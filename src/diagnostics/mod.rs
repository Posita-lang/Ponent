//! Ponent diagnostic system.
//!
//! A comprehensive, multi-format diagnostic system modeled after:
//! - **rustc**: error codes (`E030`), `^`-underline annotations, `--explain`, suggestions
//! - **Zig**: `referenced by` call chains
//! - **Austral**: multi-format output (plain / JSON / HTML), error type classification
//! - **Vale**: per-pass error humanizers, structured error types

// ── i18n: translation macro ────────────────────────────────────
//
// Delegates to `i18n::render`: the message is a stable Fluent slug
// (`type-mismatch`) resolved from `locales/*.ftl` in the active
// language; unknown keys fall back verbatim to the key itself.
//
// Usage:
//   tr!("type-mismatch", expected = "Int<32>", found = "Bool")
#[macro_export]
macro_rules! tr {
    ($msg:literal $(, $($key:ident = $val:expr),+ $(,)?)?) => {{
        // SmallVec<[T; 4]> keeps the pair list on the stack for the common
        // case (≤4 args; all current call sites use 1–2).  The old Vec
        // allocated on the heap even for a single argument.
        //
        // Values are lifted into `Cow<'_, str>` via `TrValue`: a `&str`,
        // `Cow<'_, str>` or `String` argument converts without re-formatting
        // (borrowed values stay borrowed); any other `Display` value must be
        // wrapped in `DisplayArg(...)` so the allocation is explicit at the
        // call site.
        let pairs: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'_, str>); 4]> = {
            let mut v: $crate::smallvec::SmallVec<[(&'static str, std::borrow::Cow<'_, str>); 4]> =
                $crate::smallvec::smallvec![];
            $(
                $(
                    v.push((
                        stringify!($key),
                        $crate::diagnostics::i18n::TrValue::into_cow($val),
                    ));
                )+
            )?
            v
        };
        $crate::diagnostics::i18n::render_slug($msg, &pairs)
    }};
}

pub(crate) mod ascii;
pub mod chain;
pub mod collector;
pub mod emitter;
pub mod error_code;
pub mod explain_server;
pub mod fix;
pub mod glyph;
pub mod i18n;
pub mod kind;
pub mod label;
pub mod level;
pub mod render;
pub mod renderers;
pub mod sink;
pub mod term;
pub(crate) mod wrap;

#[cfg(all(test, not(feature = "shuttle")))]
mod snapshots;

pub use chain::CallChain;
pub use collector::DiagCtxt;
pub use emitter::Emitter;
pub use error_code::{ErrCode, ErrorCategory};
pub use fix::{FixEdit, FixHints, FixSet, apply_fixes, apply_fixes_for_file};
pub use glyph::GlyphRenderer;
pub use i18n::DisplayArg;
pub use kind::{ComptimeReason, DiagnosticKind, Humanizer, TypeCtx};
pub use label::{AnnotationKind, Label, Snippet, SourcePos};
pub use level::DiagnosticLevel;
pub use render::{RenderCtx, Renderer};
pub use renderers::{HtmlRenderer, HumanRenderer, JsonRenderer, ShortRenderer};
pub use sink::{DiagnosticSink, MakeSink, StringSink};

use crate::ast::FileId;
use crate::ast::Span;
use crate::smallvec::{SmallVec, smallvec};
use arrayvec::ArrayVec;
use compact_str::{CompactString, format_compact};
use covariant_vec::CovariantVec;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use swrite::{SWrite, swriteln};

// ── Diagnostic options (inspired by GHC's DiagOpts) ────────────────

/// Default target line width (display columns) for rendered diagnostics and
/// `--explain` text.  Matches rustc's default of 100.
pub const DEFAULT_MAX_WIDTH: usize = 100;

/// When to emit ANSI colour codes, mirroring `--color=auto/always/never`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    /// Probe the terminal (see [`term::detect_color`]); no codes when piped.
    #[default]
    Auto,
    /// Always emit colour codes, even when piped.
    Always,
    /// Never emit colour codes.
    Never,
}

/// `--color` value that is not one of `auto`/`always`/`never`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseColorError;

impl std::fmt::Display for ParseColorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("expected one of `auto`, `always`, `never`")
    }
}

impl std::str::FromStr for ColorChoice {
    type Err = ParseColorError;

    /// Strict: an unknown spelling is an **error**, not `Auto`.
    ///
    /// The old inherent `from_str` mapped every typo (`--color=alwyas`) to `Auto`, i.e.
    /// silently made the answer depend on whether stdout is a tty — the "works on my
    /// terminal" bug class, at the one boundary where strictness is cheap.  (The CLI path
    /// in `main.rs` already rejected unknown values with `usage_error`; this removes the
    /// lenient twin that nothing called, so there is one parser and it is strict.)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            _ => Err(ParseColorError),
        }
    }
}

impl ColorChoice {
    /// Resolve to a concrete `use_color` bit via [`term::detect_color`].
    pub fn resolve(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => term::detect_color(),
        }
    }
}

/// Diagnostic output format, centralising the CLI `--json` flag mapping so
/// every subcommand picks the same renderer for the same flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
}

impl OutputFormat {
    pub fn from_json_flag(json: bool) -> Self {
        if json { Self::Json } else { Self::Human }
    }
}

/// Centralized configuration for the diagnostic system.
///
/// Controls filtering, formatting, and display behaviour across all emitter
/// backends (terminal, HTML, JSON, etc.).
#[derive(Debug, Clone)]
pub struct DiagOpts {
    /// Maximum number of errors to report before stopping (`None` = unlimited).
    pub max_errors: Option<usize>,
    /// Treat all warnings as errors.
    pub warn_is_error: bool,
    /// Number of source-context lines to show above and below the primary span.
    pub context_lines: usize,
    /// Whether to emit ANSI colour codes (only relevant for terminal emitters).
    ///
    /// Sharp edge: [`DiagOpts::default`] hard-codes `use_color: true` while
    /// [`ColorChoice::default`] (see `term.rs`) is `Auto`.  Any caller that
    /// builds `DiagOpts::default()` *without* following up with
    /// [`DiagOpts::detect`] (or an explicit `--color` flag override) forces
    /// colour on even when stdout is a pipe.  This is intentional — the
    /// snapshot tests pin `DiagOpts::default()` output as colored — but it
    /// means batch callers must use `detect()` (or set `use_color` themselves)
    /// to get environment-correct behaviour.
    pub use_color: bool,
    /// Whether to emit Unicode box-drawing and highlight glyphs (only
    /// relevant for terminal emitters); `false` selects the ASCII charset.
    pub use_unicode: bool,
    /// Reverse the order in which errors are reported (most recent first).
    pub reverse_errors: bool,
    /// Target line width for rendered diagnostics (display columns).
    /// Messages longer than this are word-wrapped; continuation lines
    /// repeat the box prefix.  Matches rustc's default of 100.
    pub max_width: usize,
    /// Compact layout mode: drops the sub-box spacing and close rows and
    /// simplifies the sub-box header (`│  ┌─ file:1:1` → `│  file:1:1`).
    /// Off by default; the full box-drawing layout is unchanged.
    pub compact: bool,
    /// Terminal colour depth — controls whether 256-colour palette sequences
    /// are emitted or downgraded to the 16-colour ANSI palette.  Probed from
    /// `COLORTERM` / `TERM` by [`DiagOpts::detect`]; the default
    /// ([`ColorDepth::TwoFiftySix`]) matches the historical hard-coded
    /// behaviour.
    pub color_depth: term::ColorDepth,
    /// Tab stop width for display-column calculations.  Tabs expand to the
    /// next `tab_width`-column stop.  Defaults to 4 (matches the historical
    /// hard-coded `TAB_WIDTH`); projects using 2/8-space indents can override.
    pub tab_width: usize,
    /// Show only these diagnostic codes (exact, case-insensitive match against
    /// the **canonical** spelling; entries are canonicalized once when the
    /// filter is built).
    ///
    /// **A diagnostic with no code is excluded** — matching is keyed on the code,
    /// exactly as for `only_categories`.  `allow_codes` wins when a code appears
    /// in both lists, and the three `only_*` filters compose with AND.
    pub only_codes: Vec<CompactString>,
    /// Keep only diagnostics whose error code belongs to one of these
    /// categories.
    ///
    /// **A diagnostic with no code is excluded**, because a category is a
    /// property of the code: `Diagnostic::error("...")` built without
    /// `with_code` has nothing to classify.  `DiagnosticKind`-built
    /// diagnostics that carry a code participate normally.  Callers that need
    /// code-less findings to survive must not filter by category.
    pub only_categories: Vec<ErrorCategory>,
    /// Show only diagnostics whose display file name contains one of these
    /// patterns — **substring** match (via `memmem`), not glob or exact match.
    /// Unlike `only_codes`/`only_categories`, a code-less diagnostic *does*
    /// participate: the file name is independent of the code.
    ///
    /// Empty patterns are ignored (see `DiagFilter::from_opts`).
    pub only_files: Vec<CompactString>,
    /// Codes to *suppress*, rustc `-A`-style: a diagnostic whose code is in
    /// this list is filtered out (not rendered, not counted).  The name
    /// follows rustc's `-A`/`--allow` lint levels — an "allowed" diagnostic
    /// is permitted to fail silently.  For the opposite (show *only* these
    /// codes) use [`DiagOpts::only_codes`].
    pub allow_codes: Vec<CompactString>,
    pub quiet: bool,
    pub no_wrap: bool,
}

impl DiagOpts {
    /// Build options with colour, width, and charset probed from the
    /// environment and terminal, suitable for a CLI's default before flag
    /// parsing.
    ///
    /// A `--color=always/never` flag should overwrite `use_color` afterwards;
    /// `max_width` likewise for `--width`.  [`term::supports_hyperlinks`]
    /// exists for the renderer-side follow-up that consumes it.
    /// Clamp the layout knobs at the model boundary.
    ///
    /// `DiagOpts`'s fields are public, so a caller can write `tab_width = 1 << 60` straight past
    /// [`detect`](Self::detect).  The renderers each clamp what they consume (`glyph` clamps the
    /// width to `[MIN_WIDTH, 8 * DEFAULT_MAX_WIDTH]` and the tab stop to `>= 1`), but the *model*
    /// is the one place that can state one bound for every consumer, including the ones that do
    /// arithmetic before clamping (`LineIndex::display_col`'s slot key, the wrap budget).
    ///
    /// Rustc-compatible callers never hit either clamp: 32 columns is twice the widest plausible
    /// tab stop, and 4096 is far past any real terminal.
    pub fn clamp_layout(&mut self) -> &mut Self {
        self.tab_width = self.tab_width.clamp(1, Self::MAX_TAB_WIDTH);
        self.max_width = self.max_width.clamp(1, Self::MAX_LAYOUT_WIDTH);
        self
    }

    pub fn detect() -> Self {
        let caps = term::TermCaps::detect();
        let mut opts = DiagOpts {
            use_color: caps.color,
            use_unicode: caps.unicode,
            // `max_width == 0` degenerates wrapping and tab expansion; a
            // detected width of 0 means "unknown", not "zero columns".
            max_width: caps.width.max(1),
            color_depth: caps.depth,
            tab_width: 4,
            ..Default::default()
        };
        opts.clamp_layout();
        opts
    }
}

impl DiagOpts {
    /// Widest accepted tab stop.  `tab_width * 2` still fits any display arithmetic below.
    pub const MAX_TAB_WIDTH: usize = 32;
    /// Widest accepted layout width (display columns).
    pub const MAX_LAYOUT_WIDTH: usize = 4096;
}

impl Default for DiagOpts {
    fn default() -> Self {
        DiagOpts {
            max_errors: None,
            warn_is_error: false,
            context_lines: 2,
            use_color: true,
            use_unicode: true,
            reverse_errors: false,
            max_width: DEFAULT_MAX_WIDTH,
            compact: false,
            color_depth: term::ColorDepth::TwoFiftySix,
            tab_width: 4,
            only_codes: Vec::new(),
            only_categories: Vec::new(),
            only_files: Vec::new(),
            allow_codes: Vec::new(),
            quiet: false,
            no_wrap: false,
        }
    }
}

/// A reusable filter: build once, then check many diagnostics.
///
/// [`Diagnostic::matches_filter`] rebuilds the filter per call (`O(filter_size)`
/// allocations); hoist it with [`DiagFilter::from_opts`] and use
/// [`Diagnostic::matches_filter_with`] when checking a batch.
///
/// The exact-match filters (`allow_codes`, `only_codes`, `only_categories`) are
/// hashed once at construction.  `only_files` holds `memmem::Finder`s, matched by
/// substring — the field's own doc records that a parallel `Vec` of patterns was
/// kept "in lockstep" and then never read, and this sentence used to repeat that
/// stale claim.  Building this once per batch turns filter matching from
/// `O(diagnostics × filter_size)` into `O(filter_size + diagnostics)`; the common
/// empty-filter case builds three empty sets and an empty vec, so a diagnostic
/// pays no lookup work at all.
/// Registry slots as a 512-bit map.
///
/// Slots come from a fixed table (`CODE_TABLE`, ~452 entries, slot < 512), so a
/// bit test replaces the `u16` hashing a `FxHashSet` did on the per-diagnostic
/// path — no hasher, no probe, one cache line, and a smaller `DiagFilter`.
/// `SLOT_BITS` is asserted against the registry size in `error_code.rs`
/// (`code_slot_is_a_bijection_onto_the_direct_address_range`).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SlotSet([u64; SLOT_WORDS]);

const SLOT_WORDS: usize = 8;
const SLOT_BITS: usize = SLOT_WORDS * 64;

impl SlotSet {
    fn insert(&mut self, slot: u16) {
        let slot = slot as usize;
        // `assert!`, not `debug_assert!` + a silent `if`: the table is static, so this is
        // unreachable in a correct build, but a registry refactor that yields an
        // out-of-range slot would otherwise filter silently in release — `--allow E030`
        // would stop suppressing and `only_codes` would start dropping diagnostics with no
        // signal at all.  Loud at the definition site, like the `Send + Sync` const-assert
        // and the 4 GiB scan assert.
        assert!(slot < SLOT_BITS, "registry slot {slot} exceeds the bitmap");
        self.0[slot / 64] |= 1u64 << (slot % 64);
    }

    #[inline]
    fn contains(&self, slot: u16) -> bool {
        let slot = slot as usize;
        slot < SLOT_BITS && self.0[slot / 64] >> (slot % 64) & 1 == 1
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
}

#[derive(Debug, Clone)]
pub struct DiagFilter {
    /// Code filters as **interned table slots** for every pattern that names a
    /// registry code, plus the verbatim uppercased spelling for patterns that do
    /// not.  Matching a diagnostic whose code is registered is then one bit test
    /// instead of hashing a string and possibly allocating an uppercased copy of
    /// it.
    allow_slots: SlotSet,
    allow_verbatim: rustc_hash::FxHashSet<CompactString>,
    only_slots: SlotSet,
    only_verbatim: rustc_hash::FxHashSet<CompactString>,
    /// Longest pattern stored in either verbatim set.
    ///
    /// A verbatim match compares the *whole* canonical spelling, so a diagnostic whose code is
    /// longer than every pattern cannot match — and `canonical()` on an untrusted spelling is
    /// attacker-controlled in length (`with_code_str` documents that it is safe for untrusted input).
    /// Without this bound, a filter turns `matches()` into O(code length) hashing plus, on the
    /// uppercase retry, an allocation and copy of the whole spelling — per diagnostic, in the emit
    /// loop.  This is the `MAX_CODE_CHARS` argument applied to *matching* rather than rendering.
    verbatim_max_len: usize,
    /// `ErrorCategory` bits (see [`ErrorCategory::mask_bit`]): one AND instead of a
    /// hash-and-probe per diagnostic, matching the `SlotSet` treatment of codes.
    only_category_mask: u16,
    /// Prebuilt SIMD substring searchers for `only_files`, and the *only* copy of
    /// that filter: `str::contains` runs a two-way setup per call on the
    /// per-diagnostic path, while `memmem::FinderOwned` pays it once at
    /// construction.  A parallel `Vec<String>` was kept in lockstep here and then
    /// never read, i.e. a second source of truth whose only future was to drift.
    only_files_finders: Vec<memchr::memmem::Finder<'static>>,
    /// Precomputed "no filter is set": the common unfiltered batch turns
    /// `matches` into one predictable branch instead of four `is_empty()`
    /// checks per diagnostic.
    is_empty: bool,
}

impl Default for DiagFilter {
    /// The derived `Default` produced `is_empty: false` with all four sets
    /// empty — the exact state the field exists to preclude.  Behaviour stayed
    /// correct (each guard fails), but the field's documented invariant is
    /// load-bearing for `matches`'s fast path, so it is constructed explicitly.
    fn default() -> Self {
        Self {
            allow_slots: SlotSet::default(),
            allow_verbatim: rustc_hash::FxHashSet::default(),
            only_slots: SlotSet::default(),
            only_verbatim: rustc_hash::FxHashSet::default(),
            verbatim_max_len: 0,
            only_category_mask: 0,
            only_files_finders: Vec::new(),
            is_empty: true,
        }
    }
}

impl DiagFilter {
    /// Build the fast-match view from a [`DiagOpts`].
    /// Build a filter from the options once, for reuse across many diagnostics.
    pub fn from_opts(opts: &DiagOpts) -> Self {
        // Codes are resolved case-insensitively everywhere else in the
        // codebase (`ErrCode::new`, `try_new`, `suggest_code`), and
        // `matches` compares against `ErrCode::code()` — which is the
        // canonical table spelling.  Collecting the raw CLI strings here made
        // `--allow e030` silently match nothing.  Canonicalize once at
        // construction; unknown codes stay verbatim so they still filter.
        // A pattern that names a registry code becomes its table slot; one that
        // does not keeps the uppercased spelling, so genuinely unknown codes
        // still filter.
        // Empty patterns are dropped, matching `only_files` below: a `--allow ""`
        // would otherwise admit an uppercased empty string into the verbatim set,
        // which no code equals — a filter that reports itself active while matching
        // nothing, instead of the caller's empty value being rejected as empty.
        let classify = |pats: &[CompactString]| {
            let mut slots = SlotSet::default();
            let mut verbatim: rustc_hash::FxHashSet<CompactString> =
                rustc_hash::FxHashSet::default();
            for pat in pats.iter().filter(|p| !p.is_empty()) {
                match crate::diagnostics::error_code::lookup_uppercased(pat) {
                    Some(e) => {
                        slots.insert(e.slot);
                    }
                    None => {
                        // In-place uppercase: `CompactString::from(pat.to_ascii_uppercase())`
                        // allocated a `String` and then copied it into the
                        // `CompactString`.  This idiom already exists in
                        // `suggest_code`.
                        let mut upper = CompactString::from(pat.as_str());
                        upper.as_mut_str().make_ascii_uppercase();
                        verbatim.insert(upper);
                    }
                }
            }
            (slots, verbatim)
        };
        let (allow_slots, allow_verbatim) = classify(&opts.allow_codes);
        let (only_slots, only_verbatim) = classify(&opts.only_codes);
        let only_category_mask = opts
            .only_categories
            .iter()
            .fold(0u16, |mask, c| mask | c.mask_bit());
        // Empty patterns are dropped instead of admitted: a `memmem::Finder` for
        // the empty needle matches at position 0 of *every* file name, so
        // `--only-files ""` would silently become a no-op while `is_empty`
        // reported the filter active — ANDing a match-everything term against the
        // other filters instead of being rejected at the boundary.
        // One SIMD `memmem` pass per pattern.  A single Aho-Corasick automaton would
        // be one pass regardless of pattern count, but `memmem::Finder`'s SIMD
        // constants win for the shapes this flag actually sees (`--only-files` lists
        // are short); the crossover is around eight patterns, so the swap is recorded
        // here as a measured-threshold decision rather than taken blind.
        let only_files_finders: Vec<memchr::memmem::Finder<'static>> = opts
            .only_files
            .iter()
            .filter(|pat| !pat.is_empty())
            .map(|pat| memchr::memmem::Finder::new(pat.as_bytes()).into_owned())
            .collect();
        let is_empty = allow_slots.is_empty()
            && allow_verbatim.is_empty()
            && only_slots.is_empty()
            && only_verbatim.is_empty()
            && only_category_mask == 0
            && only_files_finders.is_empty();
        let verbatim_max_len = allow_verbatim
            .iter()
            .chain(only_verbatim.iter())
            .map(|p| p.len())
            .max()
            .unwrap_or(0);
        DiagFilter {
            allow_slots,
            allow_verbatim,
            only_slots,
            only_verbatim,
            verbatim_max_len,
            only_category_mask,
            only_files_finders,
            is_empty,
        }
    }

    fn code_matches(
        slots: &SlotSet,
        verbatim: &rustc_hash::FxHashSet<CompactString>,
        verbatim_max_len: usize,
        code: &crate::diagnostics::error_code::ErrCode,
    ) -> bool {
        // Registry codes: one `u16` lookup, no string hashing.
        //
        // **Cross-module contract** (lives in `error_code.rs`, pinned there by
        // `unknown_code_resolving_to_a_registered_spelling_has_an_entry`): for the
        // `Other` variant, `entry()` must return `Some` whenever the *uppercased*
        // spelling is registered — that is what makes rustc-style `-A e030` suppress
        // `ErrCode::unknown("e030")`.  If a future refactor made `entry()` return
        // `None` for every `Other`, the `verbatim` set below would still catch the
        // exact spelling, silently degrading case-insensitive filtering.
        if let Some(entry) = code.entry()
            && slots.contains(entry.slot)
        {
            return true;
        }
        let canonical = code.canonical();
        // A verbatim pattern matches only on whole-string equality, so a longer spelling cannot hit:
        // skip the hash and the uppercase-retry allocation below.  (The slot path above is
        // length-independent and already ran.)
        if canonical.len() > verbatim_max_len {
            return false;
        }
        if verbatim.contains(canonical) {
            return true;
        }
        // `from_opts` stores every pattern uppercased.  Known and resolved
        // spellings are already canonical uppercase; only genuinely unknown
        // mixed-case spellings need one allocation, and unknown codes are
        // rare.  This replaces an O(filter_size) case-insensitive scan on the
        // per-diagnostic path.
        if canonical.bytes().all(|b| !b.is_ascii_lowercase()) {
            return false;
        }
        // Stack buffer, not `String`: codes are ≤ 5 bytes and the no-allocation
        // property is the whole point of replacing the old O(filter_size) scan.
        // Uppercasing ASCII keeps the slice valid UTF-8, so the `from_utf8`
        // cannot fail.
        let mut upper = smallvec::SmallVec::<[u8; 32]>::with_capacity(canonical.len());
        upper.extend_from_slice(canonical.as_bytes());
        upper.make_ascii_uppercase();
        match std::str::from_utf8(&upper) {
            Ok(u) => verbatim.contains(u),
            // Unreachable (ASCII-only transform), but `false` is the safe answer.
            Err(_) => false,
        }
    }

    /// Whether `diag` passes the filter — mirrors the original
    /// [`Diagnostic::matches_filter`] exactly, with hash lookups for the
    /// exact-match fields.
    /// Whether `diag` passes this filter.
    pub fn matches(&self, diag: &Diagnostic<'_>) -> bool {
        // Once per process, not once per diagnostic: every construction path goes through
        // `from_opts`/`Default` with private fields, so the invariant cannot change after
        // construction — recomputing four set emptiness tests plus the mask here showed up
        // in debug-build suites (this is the filter on every emit-loop iteration).
        {
            static CHECKED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if cfg!(debug_assertions) && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                debug_assert!(
                    self.is_empty
                        == (self.allow_slots.is_empty()
                            && self.allow_verbatim.is_empty()
                            && self.only_slots.is_empty()
                            && self.only_verbatim.is_empty()
                            && self.only_category_mask == 0
                            && self.only_files_finders.is_empty()),
                    "DiagFilter::is_empty disagrees with the sets it summarises"
                );
            }
        }
        if self.is_empty {
            return true;
        }
        let code = diag.code();
        // Suppress list (rustc `-A` semantics): a code in `allow_codes` is
        // *dropped*.  Pinned by `allow_codes_suppresses_matching_diagnostics`.
        if (!self.allow_slots.is_empty() || !self.allow_verbatim.is_empty())
            && let Some(c) = code
            && Self::code_matches(
                &self.allow_slots,
                &self.allow_verbatim,
                self.verbatim_max_len,
                c,
            )
        {
            return false;
        }
        if !self.only_slots.is_empty() || !self.only_verbatim.is_empty() {
            match code {
                Some(c)
                    if Self::code_matches(
                        &self.only_slots,
                        &self.only_verbatim,
                        self.verbatim_max_len,
                        c,
                    ) => {}
                _ => return false,
            }
        }
        if self.only_category_mask != 0 {
            match code {
                Some(c) if self.only_category_mask & c.category().mask_bit() != 0 => {}
                _ => return false,
            }
        }
        if !self.only_files_finders.is_empty() {
            let file = diag.file_name().as_bytes();
            if !self
                .only_files_finders
                .iter()
                .any(|finder| finder.find(file).is_some())
            {
                return false;
            }
        }
        true
    }
}

// ── Emission guarantee (inspired by rustc's ErrorGuaranteed) ─────

/// A token proving that a diagnostic (error or warning) has been emitted
/// (or intentionally suppressed).  `EmissionGuarantee::emitted()` means the
/// diagnostic was *actually* recorded; `EmissionGuarantee::suppressed()`
/// means it was filtered out (e.g. a warning when `can_emit_warnings` is
/// false).  Callers can call [`was_emitted`](Self::was_emitted) to
/// distinguish the two cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionGuarantee {
    _private: (),
    emitted: bool,
}

impl EmissionGuarantee {
    /// The emitted diagnostic was recorded.
    pub(crate) fn emitted() -> Self {
        EmissionGuarantee {
            _private: (),
            emitted: true,
        }
    }

    /// The diagnostic was suppressed (e.g. warning when warnings are off).
    pub(crate) fn suppressed() -> Self {
        EmissionGuarantee {
            _private: (),
            emitted: false,
        }
    }

    /// Returns `true` if the diagnostic was actually recorded.
    pub fn was_emitted(&self) -> bool {
        self.emitted
    }
}

/// A `Result` that guarantees a diagnostic (error or warning) has been
/// emitted on the `Err` path.  Analogous to rustc's `Result<T, ErrorGuaranteed>`
/// pattern — the `Err` variant carries an `EmissionGuarantee` proving that
/// the diagnostic was recorded, rather than carrying the diagnostic itself.
pub type DiagResult<T> = Result<T, EmissionGuaranteed>;

/// Opaque wrapper around `EmissionGuarantee` for use in `DiagResult`.
/// Functions that return `DiagResult<T>` produce an `EmissionGuaranteed`
/// on the error path, proving a diagnostic was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionGuaranteed(EmissionGuarantee);

impl From<EmissionGuarantee> for EmissionGuaranteed {
    fn from(g: EmissionGuarantee) -> Self {
        EmissionGuaranteed(g)
    }
}

/// A lazily-formatted diagnostic message, analogous to rustc's
/// `DiagMessage`: the message is DATA until a renderer commits to a
/// concrete string.
///
/// Two shapes:
/// - [`DiagMessage::Static`] — the full English text (or any pre-rendered
///   string), borrowed when possible (`Cow<'src, str>`), so fully static
///   messages allocate NOTHING.
/// - [`DiagMessage::Slug`] — a Fluent slug plus typed arguments, resolved
///   through the active language's bundle at materialization time.  Word
///   order lives in `locales/*.ftl`, not in `format!` templates.
///
/// Materialization is [`DiagMessage::render_into`] (buffer-reusing) /
/// [`DiagMessage::to_string`]; `Diagnostic::humanize` is the single
/// boundary where the collector commits.
#[derive(Debug, Clone)]
pub enum DiagMessage<'src> {
    Static(Cow<'src, str>),
    /// A Fluent slug (`locales/*.ftl` message id) with its arguments.
    /// `pairs` is ordered for deterministic rendering; keys must match
    /// the `$placeholders` in the .ftl entries.
    Slug {
        slug: &'static str,
        pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>,
    },
    /// Several messages rendered in order.
    ///
    /// Exists so a composed message can stay **lazy**: `Comptime` with a traceback
    /// used to render its base message eagerly (freezing it in whichever language was
    /// active at construction) and append a pre-built English header and frame list,
    /// which is the module's one documented mixed-language window.  With `Joined` the
    /// base slug, the header slug and every frame slug all resolve at the render
    /// boundary, i.e. in the language active *then*.
    Joined(Box<[DiagMessage<'src>]>),
}

impl<'src> DiagMessage<'src> {
    /// Static text, borrowing a `'static` string when possible.
    pub fn static_str(s: &'static str) -> Self {
        DiagMessage::Static(Cow::Borrowed(s))
    }

    /// Fluent slug with arguments (see [`DiagMessage::Slug`]).
    pub fn slug(slug: &'static str, pairs: SmallVec<[(&'static str, Cow<'src, str>); 4]>) -> Self {
        DiagMessage::Slug { slug, pairs }
    }

    /// Several messages rendered in order (see [`DiagMessage::Joined`]).
    ///
    /// Takes a `Vec` and boxes it: the variant contains `Self`, so it needs
    /// indirection, and one boxed slice is a single allocation with no per-part
    /// pointer chase.
    pub fn joined(parts: Vec<DiagMessage<'src>>) -> Self {
        // Flatten nested compositions to a **fixpoint**, depth-first and in order.  The
        // previous one-level loop left `joined(joined(joined(..)))` at depth 2 while this
        // comment claimed "depth is now 1 by construction": `joined` is public and nestable,
        // the render path recursed once per level, and `compute_stable_id` saw a deeper tag
        // sequence for identical output.  A work stack makes the claim true instead of
        // softening it — every part is expanded once, so the cost stays linear in the subtree.
        // (Ids for previously-nested-by-two-or-more input change once — the same latitude the
        // accessor's "cross-run key, not persisted" contract granted the previous id changes.)
        let mut flat: Vec<DiagMessage<'src>> = Vec::with_capacity(parts.len());
        let mut work = parts;
        work.reverse();
        while let Some(part) = work.pop() {
            match part {
                DiagMessage::Joined(inner) => {
                    work.extend(inner.into_vec().into_iter().rev());
                }
                other => flat.push(other),
            }
        }
        DiagMessage::Joined(flat.into_boxed_slice())
    }

    /// Materialize into `out` (reusing its capacity across calls — the
    /// collector's hot path), routing slugs through the i18n bundle.
    pub fn render_into(&self, out: &mut String) {
        match self {
            DiagMessage::Static(s) => out.push_str(s),
            DiagMessage::Slug { slug, pairs } => {
                // `Slug` *is* the "this is a slug" tag, so the generic entry
                // point's `is_ftl_slug` scan only re-derived it.
                crate::diagnostics::i18n::render_into_slug_ref(slug, pairs, out);
            }
            DiagMessage::Joined(parts) => {
                for part in parts.iter() {
                    part.render_into(out);
                }
            }
        }
    }

    /// Consuming variant of [`render_into`](Self::render_into): moves
    /// `Slug`-pair arguments into the Fluent map instead of cloning them.
    /// Only callable when the caller owns the message (e.g. `humanize`,
    /// which gets a fresh `DiagMessage` from `humanize_into`).
    pub fn render_into_owned(self, out: &mut String) {
        match self {
            DiagMessage::Static(s) => out.push_str(&s),
            DiagMessage::Slug { slug, pairs } => {
                crate::diagnostics::i18n::render_into_slug(slug, pairs, out);
            }
            DiagMessage::Joined(parts) => {
                // `Box<[T]>` → `Vec<T>` moves the slice out, so the parts are
                // consumed rather than cloned (the point of the owned variant).
                for part in parts.into_vec() {
                    part.render_into_owned(out);
                }
            }
        }
    }

    /// Materialize into a fresh `String` (cold paths, tests).
    pub fn to_rendered(&self) -> String {
        let mut s = String::new();
        self.render_into(&mut s);
        s
    }

    /// Borrow the static text, if this message is fully static.
    pub fn as_static(&self) -> Option<&str> {
        match self {
            DiagMessage::Static(s) => Some(s),
            // A `Joined` message is never "fully static": its parts are rendered in
            // the active language, so callers that want one borrowed string must go
            // through `render_into`.
            DiagMessage::Slug { .. } | DiagMessage::Joined(_) => None,
        }
    }
}

impl Default for DiagMessage<'_> {
    fn default() -> Self {
        DiagMessage::Static(Cow::Borrowed(""))
    }
}

impl From<String> for DiagMessage<'_> {
    fn from(s: String) -> Self {
        DiagMessage::Static(Cow::Owned(s))
    }
}

impl From<&str> for DiagMessage<'_> {
    fn from(s: &str) -> Self {
        DiagMessage::Static(Cow::Owned(s.to_owned()))
    }
}

// NOTE: no `From<&'static str>` specialization — it conflicts with
// `From<&str>` (the impls overlap for every 'static input).  Zero-copy
// static text goes through the explicit `DiagMessage::static_str`
// constructor instead.

impl<'a> std::fmt::Display for DiagMessage<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagMessage::Static(s) => f.write_str(s),
            DiagMessage::Slug { slug, pairs } => {
                // The variant **is** the "known slug" tag, so route through the slug-tagged
                // renderer: `render` re-derives the tag with an `is_ftl_slug` byte scan on every
                // `Display`, which the `render_into` path already avoids.  One shape, one cost
                // model.
                f.write_str(&crate::diagnostics::i18n::render_slug(slug, pairs))
            }
            DiagMessage::Joined(parts) => {
                for part in parts.iter() {
                    write!(f, "{part}")?;
                }
                Ok(())
            }
        }
    }
}

/// A set of spans that a diagnostic points to, supporting multiple primary
/// locations.  Analogous to rustc's `MultiSpan`.  When there is a single
/// primary span, `primary` contains one element; for multiple (e.g. a
/// duplicate definition + the previous definition), both are primary.
#[derive(Debug, Default)]
pub struct MultiSpan {
    // Invariant: when `!dirty`, sorted by `(file, start, end)` ascending
    // (`Span`'s `Ord`), no duplicates. All mutation goes through `push`,
    // which preserves this.  When `dirty`, the vector is unsorted and may
    // contain duplicates; `primary()` sorts + dedups before returning.
    primary: SmallVec<[Span; 2]>,
    dirty: bool,
    /// Cached sorted+deduped view for the `dirty` state, populated on the
    /// first `primary()`/`primary_ref()` read after a dirtying `push` (a
    /// boxed slice, not a clone of the inline `SmallVec`: the cache must not
    /// grow `Diagnostic` past its size guard, and the dirty path is rare
    /// enough that one heap allocation is noise).
    /// Interior mutability is safe here: `MultiSpan` only ever crosses
    /// threads by *move* (`&mut Diagnostic` chunks in the emitter's scoped
    /// threads need `Send`, not `Sync`, and `OnceLock<Box<[..]>>` stays
    /// `Send` *and* `Sync` — the latter is required because `Diagnostic`
    /// values are stored in the query cache (`FnBodyCheck`), whose storage
    /// is `Sync`).
    /// Invalidated by every `push`/`clear`.
    sorted_cache: std::sync::OnceLock<Box<[Span]>>,
}

impl Clone for MultiSpan {
    fn clone(&self) -> Self {
        // `sorted_cache` is a pure function of `primary`; copying its boxed
        // slice makes cloning a diagnostic O(spans) and defeats the cache.
        // Recompute lazily if the clone is ever read in the dirty state.
        MultiSpan {
            primary: self.primary.clone(),
            dirty: self.dirty,
            sorted_cache: std::sync::OnceLock::new(),
        }
    }
}

impl MultiSpan {
    pub fn new(span: Span) -> Self {
        MultiSpan {
            primary: smallvec![span],
            dirty: false,
            sorted_cache: std::sync::OnceLock::new(),
        }
    }

    /// Insert a primary span, appending in O(1).
    ///
    /// Duplicates of the *tail* are ignored immediately; a duplicate of an
    /// earlier element is appended and removed by the deferred sort+dedup on
    /// the next [`primary`](Self::primary) / [`first`](Self::first) read, so
    /// the observable span set stays idempotent (safe for repeated
    /// `humanize`) while the `SmallVec` may transiently hold a duplicate.
    ///
    /// Compilers emit spans in ascending source order, so the fast path
    /// compares against the tail and appends in O(1).  An out-of-order span
    /// is appended in O(1) and the vector is marked `dirty`; the sort +
    /// dedup is deferred to the next [`primary`](Self::primary) or
    /// [`first`](Self::first) read, so a fully reverse-sorted push stream
    /// costs `O(n log n)` total instead of `O(n²)` memmoves.
    /// **`max_seen` is deliberately absent.**  Tracking the running maximum so
    /// an "ascending" push against a sorted-cached view avoids a re-sort does
    /// not work: the sorted view only exists while `dirty == true`, i.e. while
    /// `primary` is *unsorted*, and no append can restore sortedness there — the
    /// re-sort is required, not incidental.  In the one state where an ascending
    /// push is free (`dirty == false`, `primary` sorted), the tail **is** the
    /// maximum, so `last() > span` already decides it and a `max_seen` field
    /// would be a second copy of the same fact.  What does hold — and is pinned
    /// by `multispan_push_read_interleaving_keeps_the_view_exact` — is that
    /// repeated reads between pushes pay at most one sort.
    pub fn push(&mut self, span: Span) {
        // `push` is documented as idempotent (safe for a repeated `humanize`),
        // and a duplicate tail changes nothing — including the sorted view, so
        // it must not be dropped.  The old shape reset the cache before the
        // match and made the idempotent path re-sort on the next read.
        if self.primary.last() == Some(&span) {
            return;
        }
        // Drop the sorted view cache on every real push by replacing the lock
        // (the old `OnceLock::take` requires Rust 1.90; MSRV is 1.88).
        // Re-population only happens on the next dirty read.
        self.sorted_cache = std::sync::OnceLock::new();
        // No equality arm: the early return above already handled `last == span`
        // (and resetting the cache for a duplicate tail would be wrong there).
        match self.primary.last() {
            Some(last) if *last > span => {
                self.primary.push(span);
                self.dirty = true;
            }
            _ => self.primary.push(span),
        }
    }

    /// The earliest primary span, or `None` if empty.
    ///
    /// Complexity: `O(1)` when the set is clean; when `dirty`, `O(n log n)` once
    /// through [`primary_ref`] (which sorts *and caches*), then `O(1)` for every
    /// later call.  The old text described an `O(n)` min-scan that no longer
    /// exists — a reader optimizing against it would believe repeated `first()`
    /// calls were cheap-min when they are cached lookups.
    pub fn first(&self) -> Option<Span> {
        if self.dirty {
            // Route through `primary_ref`, which sorts once and caches: the old
            // `iter().min()` rescanned `O(n)` on every call between pushes
            // (`stable_id`, `augment` and every renderer call `first()`).
            self.primary_ref().first().copied()
        } else {
            self.primary.first().copied()
        }
    }

    /// Clear all primary spans.
    pub fn clear(&mut self) {
        self.primary.clear();
        self.dirty = false;
        self.sorted_cache = std::sync::OnceLock::new();
    }

    /// Return the spans in sorted order.  When `dirty`, sorts + dedups once
    /// and caches the result in `sorted_cache` — repeated reads between
    /// pushes skip the `O(n log n)` re-sort; otherwise clones the
    /// already-sorted vector (inline for `n ≤ 2`, so no heap allocation in
    /// Owned copy of the sorted, deduplicated primary spans.
    ///
    /// Renderers must use [`Self::primary_ref`]: this routes through it and then
    /// `.collect()`s a `SmallVec`, which is the clone `primary_ref` exists to avoid — so
    /// using it in a per-span render loop pays one allocation per call for nothing.  It
    /// survives for callers that need an owned value (tests, one-shot inspection).
    /// the common case).
    pub fn primary(&self) -> SmallVec<[Span; 2]> {
        // Route through `primary_ref` so the sort/dedup/cache discipline has
        // exactly one implementation: a dirty span sorts once and both
        // accessors then read the cached view.
        self.primary_ref().iter().copied().collect()
    }

    /// The primary spans as a borrowed slice when possible, avoiding the
    /// [`SmallVec`] clone that [`primary`](Self::primary) performs even when
    /// the vector is already sorted (`!dirty`, the common case —
    /// [`push`](Self::push) keeps it in ascending order).  On a `dirty`
    /// read it either borrows the cached sorted view or owns a freshly
    /// sorted + deduped `Vec` (which is then cached, so subsequent dirty
    /// reads borrow too).  Callers that only iterate or read `len`/`first`
    /// pay no clone on the fast path.
    pub fn primary_ref(&self) -> Cow<'_, [Span]> {
        if !self.dirty {
            Cow::Borrowed(self.primary.as_ref())
        } else if let Some(cached) = self.sorted_cache.get() {
            // The cache is behind `&self`, so the borrowed view is exact.
            Cow::Borrowed(cached)
        } else {
            // `collect` on a `TrustedLen` iterator reserves exactly `len`, so the
            // `into_boxed_slice` below is a no-op; cloning the `SmallVec` first could
            // carry spare capacity and make it shrink (a second allocation).
            let mut out: Vec<Span> = self.primary.iter().copied().collect();
            out.sort_unstable();
            out.dedup();
            debug_assert!(
                // Strict: the documented invariant is sorted **and deduped**, so
                // a dropped `dedup()` must trip here rather than pass on `<=`.
                out.windows(2).all(|w| w[0] < w[1]),
                "MultiSpan::primary must be sorted by (file, start)"
            );
            // Cache the sorted view so repeated reads between pushes skip
            // the sort; the boxed-slice allocation is one small heap copy
            // against an `O(n log n)` sort per read.  Return a borrow of the
            // cache rather than cloning it: the cache lives behind `&self`,
            // so the borrow is exact and the dirty path costs no extra copy.
            match self.sorted_cache.set(out.into_boxed_slice()) {
                Ok(()) => {
                    // Two threads may both compute and both attempt `set`; a
                    // successful `set` or the winner's value is visible here
                    // because `OnceLock` is acquire/release.
                    Cow::Borrowed(self.sorted_cache.get().expect("just set"))
                }
                // We lost the set race.  Our sorted+deduped view is identical
                // to the winner's; return it instead of discarding the work.
                Err(own) => Cow::Owned(own.into_vec()),
            }
        }
    }
}

/// How confident we are that a suggestion is correct.
/// Inspired by rustc's `Applicability`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    /// The suggestion can be applied automatically without review.
    MachineApplicable,
    /// The suggestion might be correct, but needs human review.
    MaybeIncorrect,
    /// The suggestion has placeholders that need to be filled in.
    HasPlaceholders,
    /// The applicability is unspecified.
    Unspecified,
}

impl Default for Applicability {
    /// `Unspecified` is the safe fallback: an auto-applied suggestion the
    /// author never classified (API misuse of the default) must not be
    /// treated as machine-applicable without review.
    fn default() -> Self {
        Applicability::Unspecified
    }
}

/// How a suggestion should be displayed.
/// Inspired by rustc's `SuggestionStyle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SuggestionStyle {
    /// Show the suggestion inline with the diagnostic.
    #[default]
    ShowAlways,
    /// Show the suggestion code, but hide the message.
    ShowCode,
    /// Show the message, but hide the inline code.
    HideCodeInline,
    /// Hide the suggestion entirely from the user.
    HideCodeAlways,
    /// Completely hidden (only visible to tools, e.g. IDE).
    CompletelyHidden,
}

/// The JSON wire name of an [`Applicability`] variant.
pub fn applicability_name(app: Applicability) -> &'static str {
    match app {
        Applicability::MachineApplicable => "MachineApplicable",
        Applicability::MaybeIncorrect => "MaybeIncorrect",
        Applicability::HasPlaceholders => "HasPlaceholders",
        Applicability::Unspecified => "Unspecified",
    }
}

/// The JSON wire name of a [`SuggestionStyle`] variant.
pub fn suggestion_style_name(style: SuggestionStyle) -> &'static str {
    match style {
        SuggestionStyle::ShowAlways => "ShowAlways",
        SuggestionStyle::ShowCode => "ShowCode",
        SuggestionStyle::HideCodeInline => "HideCodeInline",
        SuggestionStyle::HideCodeAlways => "HideCodeAlways",
        SuggestionStyle::CompletelyHidden => "CompletelyHidden",
    }
}

/// The JSON wire name of an [`AnnotationKind`] variant.
pub fn annotation_kind_name(k: AnnotationKind) -> &'static str {
    match k {
        AnnotationKind::Primary => "Primary",
        AnnotationKind::Secondary => "Secondary",
        AnnotationKind::Note => "Note",
        AnnotationKind::Help => "Help",
    }
}

/// The JSON wire name of a [`chain::ChainEntryKind`] variant.
pub fn chain_entry_kind_name(k: chain::ChainEntryKind) -> &'static str {
    match k {
        chain::ChainEntryKind::Call => "Call",
        chain::ChainEntryKind::Inlined => "Inlined",
        chain::ChainEntryKind::Reference => "Reference",
        chain::ChainEntryKind::Defines => "Defines",
    }
}

/// A structured suggestion with confidence and style metadata.
#[derive(Debug, Clone)]
pub struct Suggestion<'src> {
    pub message: Cow<'src, str>,
    pub applicability: Applicability,
    pub style: SuggestionStyle,
    /// Source span the suggestion replaces.  When present together with
    /// `replacement`, renderers emit a two-line `-`/`+` diff instead of a
    /// one-line help/note.
    pub span: Option<Span>,
    /// Replacement text for `span`.
    pub replacement: Option<Cow<'src, str>>,
}

impl<'src> std::fmt::Display for Suggestion<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag = match self.applicability {
            Applicability::MachineApplicable => "",
            Applicability::MaybeIncorrect => " (maybe incorrect)",
            Applicability::HasPlaceholders => " (has placeholders)",
            Applicability::Unspecified => "",
        };
        write!(f, "{}{}", self.message, tag)
    }
}

/// A segment of styled text, used in notes and help messages.
/// Inspired by rustc's `StringPart`.
#[derive(Debug, Clone)]
pub struct StringPart<'src> {
    pub content: Cow<'src, str>,
    pub style: StringPartStyle,
}

/// Whether a `StringPart` should be highlighted or rendered normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPartStyle {
    Normal,
    Highlighted,
}

/// A sequence of styled text segments, used for rich diagnostic messages.
/// When rendered, highlighted segments are shown in a distinct color
/// (e.g. bright blue for type names) while normal segments use the default color.
#[derive(Debug, Clone)]
pub struct StyledString<'src>(pub Vec<StringPart<'src>>);

impl<'src> StyledString<'src> {
    pub fn new(parts: Vec<StringPart<'src>>) -> Self {
        StyledString(parts)
    }

    pub fn plain(text: impl Into<Cow<'src, str>>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Normal,
        }])
    }

    pub fn highlighted(text: impl Into<Cow<'src, str>>) -> Self {
        StyledString(vec![StringPart {
            content: text.into(),
            style: StringPartStyle::Highlighted,
        }])
    }

    /// Render the styled string as plain text (without ANSI codes).
    pub fn to_plain(&self) -> String {
        self.0.iter().map(|p| p.content.as_ref()).collect()
    }
}

/// A complete compiler diagnostic (error, warning, help, note, or info).
///
/// Architecture inspired by multiple compilers:
/// - **rustc**: Error codes + `^`-underline annotations + suggestions + sub-diagnostics
/// - **Zig**: `referenced by` call chains for traceability
/// - **Austral**: Error type categorization + multi-format output
/// - **Vale**: Structured error variants per compiler pass
///
/// # Lazy humanization
///
/// When built from a structured [`DiagnosticKind`] via
/// [`Diagnostic::error_kind`] / [`fatal_kind`] / [`warning_kind`],
/// `message` / `labels` / `spans` / `help` / `suggestions` stay
/// unfilled until [`humanize`](Self::humanize) runs.  The constructors
/// deliberately skip that step so:
///
/// - **Filtered-out diagnostics pay nothing** — `DiagCtxt::emit` humanizes through
///   the emitter's own [`DiagFilter`] ([`DiagCtxt::humanize_pending_for`]), so a
///   `--only-files`/`--allow`/`--only-codes` run pays no Fluent render for a rejected
///   diagnostic.  What is *not* skipped: diagnostics dropped by `max_errors` (the budget
///   is applied while rendering, so the count is only known afterwards) and diagnostics
///   suppressed by `can_emit_warnings` (those are dropped at `push`, before this).
/// - **i18n language resolves at render time** — Fluent runs against
///   the *current* `Lang`, so a long-lived process (LSP) can
///   `Lang::set` between emits without freezing messages in the
///   language that happened to be active at construction.
///
/// Humanization is the *caller's* step for batches: [`Emitter::emit`] (the
/// single-diagnostic path) humanizes at its render boundary, but every batch entry
/// point renders `&Diagnostic` and cannot mutate, so it requires diagnostics that
/// have already been humanized (`DiagCtxt::humanize_pending`).  The `&mut` batch
/// variants (`emit_all_to`, `emit_all_streaming_to`) do it for you.  Reading these
/// fields before that boundary sees the constructor's defaults: an empty message and
/// a `None` primary span.
/// `kind` is retained (never dropped) as the structured source of
/// truth so `humanize()` can re-run if the kind is replaced.
/// Restores a temporarily-taken `Diagnostic::kind` if `humanize_into` panics.
/// `disarm` on the success path returns the kind without restoring twice.
/// The soundness argument for [`MultiSpan`]'s interior `OnceLock` cache is that
/// `Diagnostic` values live in the query cache (`FnBodyCheck`), whose storage is
/// `Sync` — and today that is an unwritten assumption.  `error_code.rs` pins the
/// same property for `ErrCode`; this is the stronger claim, so pin it at the
/// definition site: a future non-`Sync` field then fails here instead of at a
/// distant use site (or, worse, at runtime behind a `dyn` erasure).
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Diagnostic<'static>>();
    assert_send_sync::<MultiSpan>();
};

#[derive(Debug)]
pub struct Diagnostic<'src> {
    level: DiagnosticLevel,
    message: Cow<'src, str>,
    code: Option<ErrCode>,
    /// Structured error kind (Vale-style ADT).  When present, the humanizer
    /// can derive `message`, `labels`, and other fields automatically.
    kind: Option<std::sync::Arc<DiagnosticKind<'src>>>,
    /// Primary spans (multiple for multi-location diagnostics).
    spans: MultiSpan,
    help: Option<Cow<'src, str>>,
    /// Structured suggestions with applicability and style metadata.
    suggestions: CovariantVec<'src, Suggestion<'src>, 1>,
    /// Multi-span annotations for source-context rendering
    labels: CovariantVec<'src, Label<'src>, 1>,
    /// Zig-style call chain tracing how the error was reached
    call_chain: Option<std::sync::Arc<CallChain<'src>>>,
    /// Retained source text for `^`-underline rendering.  Borrowed from the
    /// driver's source buffer, so attaching it to every diagnostic in a
    /// compilation costs nothing.
    source: Option<&'src str>,
    /// Source file name displayed in the `┌─ <input>:1:1` location header.
    file_name: Cow<'src, str>,
    /// Child diagnostics (notes, help messages attached to this diagnostic).
    /// Sub-diagnostics.
    ///
    /// Measured, not assumed (`diagnostic_size_bounded` prints all three):
    /// `Diagnostic` is 392 bytes against a 416-byte guard, `Subdiag` is 88.  An
    /// inline-1 shape (`enum Children { None, One(Subdiag), Many(Vec<Subdiag>) }`)
    /// needs `max(88, 24) + 8 = 96` bytes where the `Vec` header uses 24, i.e.
    /// `392 - 24 + 96 = 464` — 48 bytes past the guard.  The `One(Box<Subdiag>)`
    /// variant that *would* fit (400 bytes) still heap-allocates, so it saves
    /// nothing.  Hence this stays a `Vec` and the "one note" case pays one
    /// allocation.
    children: Vec<Subdiag<'src>>,
    /// Related errors displayed alongside this diagnostic in a single merged box.
    related_errors: Vec<RelatedError<'src>>,
    /// Copy-on-write: `augment` used to deep-copy the whole tuple vector even
    /// when the target merely inherited the same source set, which is the common
    /// case for a derived diagnostic.  `Arc` makes that a refcount bump and
    /// `Arc::make_mut` clones only on a real mutation.
    foreign_sources: Option<std::sync::Arc<Vec<(FileId, Cow<'src, str>, &'src str)>>>,
    /// Whether `spans` were set by a user-facing builder rather than left
    /// at the constructor default.  `humanize()` leaves them alone in that
    /// case instead of deriving fresh primary spans from the kind's labels.
    spans_user_set: bool,
    /// Whether `help` was set by a user-facing builder rather than derived
    /// from `kind`. `humanize()` leaves it alone in that case instead of
    /// silently overwriting it.
    help_user_set: bool,
    /// Whether `suggestions` were set by a user-facing builder; `humanize()`
    /// must not clear/replace them, matching the spans precedent.
    suggestions_user_set: bool,
    /// How many entries at the *front* of `labels` were derived from `kind`
    /// by the last [`humanize`](Self::humanize).  User labels appended after
    /// that block (`with_label`, `with_secondary_label`, `augment`) survive
    /// re-humanization: the derived block is truncated and re-derived in the
    /// new language, the user block is preserved.  Labels are kind-derived
    /// first, then user-added — the same order the builders establish, since
    /// user builders only ever append.
    kind_derived_labels: u32,
    /// Whether [`humanize`](Self::humanize) has already populated `message`
    /// and `labels` from `kind`. Guards against redundant re-rendering on a
    /// second call (the operation is idempotent, but not free).
    humanized: bool,
    /// The [`Lang`](crate::diagnostics::i18n::Lang) discriminant active at
    /// the last [`humanize`](Self::humanize) (`0` = never humanized).
    ///
    /// The sentinel is sound only because no real language uses discriminant 0
    /// (`LANG_EN = 1`, pinned by `no_language_uses_the_zero_discriminant` in
    /// `i18n.rs`).  It is nevertheless read *only* under `humanized == true`, so a
    /// future renumbering cannot resurrect a bug through this field alone.  When
    /// the process language changes, the next `humanize()` re-derives
    /// `message`/`labels`/`help` from the retained `kind` in the new
    /// language — the LSP re-emit contract documented on
    /// [`error_kind`](Self::error_kind).
    humanized_lang: u8,
    /// Cached result of [`stable_id`](Self::stable_id). `0` means "not yet computed".
    ///
    /// `compute_stable_id` feeds its digest through a word-at-a-time mixer (the FNV-1a
    /// walk this comment used to cite is retired) and steers an exact `0` to `1`, so the
    /// sentinel collides only with a full 64-bit avalanche output — a ~2⁻⁶⁴ coincidence
    /// per diagnostic, and even then the only consequence is recomputing on every call,
    /// which still yields a consistent result.  See [`Self::invalidate_stable_id`].
    stable_id_cache: AtomicU64,
}

impl<'src> Clone for Diagnostic<'src> {
    fn clone(&self) -> Self {
        Diagnostic {
            level: self.level,
            message: self.message.clone(),
            code: self.code.clone(),
            kind: self.kind.clone(),
            spans: self.spans.clone(),
            help: self.help.clone(),
            suggestions: self.suggestions.clone(),
            labels: self.labels.clone(),
            call_chain: self.call_chain.clone(),
            source: self.source,
            file_name: self.file_name.clone(),
            children: self.children.clone(),
            related_errors: self.related_errors.clone(),
            foreign_sources: self.foreign_sources.clone(),
            spans_user_set: self.spans_user_set,
            help_user_set: self.help_user_set,
            suggestions_user_set: self.suggestions_user_set,
            kind_derived_labels: self.kind_derived_labels,
            humanized: self.humanized,
            humanized_lang: self.humanized_lang,
            stable_id_cache: AtomicU64::new(self.stable_id_cache.load(Ordering::Relaxed)),
        }
    }
}

impl<'src> Diagnostic<'src> {
    // ── Read-only accessors ─────────────────────────────────────────
    pub fn level(&self) -> DiagnosticLevel {
        self.level
    }
    pub fn message(&self) -> &str {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read message() before humanize()"
        );
        &self.message
    }
    pub(crate) fn message_unchecked(&self) -> &str {
        &self.message
    }

    /// Has this diagnostic been through [`Self::humanize`]?
    ///
    /// The batch entry points render `&Diagnostic` and never mutate, so they
    /// `debug_assert!` this in their filtering phase: a caller that skipped
    /// `DiagCtxt::humanize_pending` used to get silently wrong output (files keyed on
    /// pre-materialization spans, an empty message), which is the one failure mode
    /// this subsystem keeps re-discovering.
    #[inline]
    pub(crate) fn is_humanized(&self) -> bool {
        self.humanized
    }

    /// The message a renderer should show, with the release-mode fallback applied.
    ///
    /// The legitimate path pays one bool read and returns a borrow.  Only a
    /// diagnostic that skipped `humanize()` — the contract breach the batch entry
    /// points can merely `debug_assert!`, because they take `&Diagnostic` — takes the
    /// kind-rendering arm.  Routing the renderers through this turns "empty
    /// user-visible diagnostic in release" into degraded-but-correct output, which is
    /// this subsystem's stated worst-case policy ("a fallback for the fallback must be
    /// total"); `unhumanized_fallback` already existed and was consumed by `Display`
    /// alone, so the batch path shipped nothing.
    #[inline]
    pub(crate) fn display_message(&self) -> std::borrow::Cow<'_, str> {
        if self.humanized {
            return std::borrow::Cow::Borrowed(&self.message);
        }
        match self.unhumanized_fallback() {
            Some(fallback) => std::borrow::Cow::Owned(fallback),
            // No kind means it was built with a literal message, which is already
            // what `self.message` holds.
            None => std::borrow::Cow::Borrowed(self.message_unchecked()),
        }
    }

    /// The message to print for a kind-built diagnostic that was never humanized.
    ///
    /// `Some` only for that (contract-violating) shape; `Display` uses it as a
    /// release-mode fallback so a skipped `humanize()` cannot ship an empty
    /// user-visible message.  `None` on every legitimate path, so the hot path pays
    /// one `Option` check and no allocation.
    ///
    /// `message()` deliberately does **not** use this: it returns `&str`, so a
    /// fallback would have to leak or cache the rendering.  `Display` is the
    /// user-facing surface, which is why the guarantee lives there.
    pub(crate) fn unhumanized_fallback(&self) -> Option<String> {
        if self.humanized {
            return None;
        }
        let kind = self.kind.as_deref()?;
        let mut out = String::new();
        kind.message().render_into_owned(&mut out);
        if out.is_empty() {
            // A kind whose slug is missing from every bundle renders nothing; make
            // the breach visible instead of empty.
            out.push_str("<unhumanized diagnostic>");
        }
        Some(out)
    }
    pub fn code(&self) -> Option<&ErrCode> {
        self.code.as_ref()
    }
    pub fn kind(&self) -> Option<&DiagnosticKind<'src>> {
        self.kind.as_deref()
    }
    pub fn spans(&self) -> &MultiSpan {
        &self.spans
    }
    pub fn help(&self) -> Option<&str> {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read help() before humanize()"
        );
        self.help.as_deref()
    }
    pub fn suggestions(&self) -> &[Suggestion<'src>] {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read suggestions() before humanize()"
        );
        &self.suggestions
    }
    pub fn labels(&self) -> &[Label<'src>] {
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "read labels() before humanize()"
        );
        &self.labels
    }
    pub fn call_chain(&self) -> Option<&CallChain<'src>> {
        self.call_chain.as_deref()
    }
    pub fn source(&self) -> Option<&str> {
        self.source
    }
    pub fn file_name(&self) -> &str {
        &self.file_name
    }
    pub fn children(&self) -> &[Subdiag<'src>] {
        &self.children
    }
    pub fn children_mut(&mut self) -> &mut Vec<Subdiag<'src>> {
        &mut self.children
    }
    pub fn related_errors(&self) -> &[RelatedError<'src>] {
        &self.related_errors
    }

    // ── Mutable accessors ───────────────────────────────────────────
    pub fn labels_mut(&mut self) -> &mut CovariantVec<'src, Label<'src>, 1> {
        &mut self.labels
    }
    pub fn related_errors_mut(&mut self) -> &mut Vec<RelatedError<'src>> {
        &mut self.related_errors
    }
    /// Attach the file segment this diagnostic came from.
    ///
    /// Takes `&'src str`, not `Option`: a diagnostic's source identifies *which
    /// file segment* it came from, and a wrong one is structurally
    /// undetectable at emission time (a wrong snippet looks exactly like a
    /// right one).  Keeping the "source was cleared" state unrepresentable is
    /// the only reliable defence, so there is deliberately no path that sets it
    /// back to `None` — first attach wins, later ones are ignored by
    /// [`DiagCtxt::attach_source`](crate::diagnostics::collector::DiagCtxt::attach_source)'s
    /// `is_none` guard.
    pub fn set_source(&mut self, source: &'src str) {
        self.source = Some(source);
    }
    pub fn set_file_name(&mut self, file_name: impl AsRef<str>) {
        // Deliberately `AsRef<str>`, not `Into<Cow<'src, str>>`: the `Cow` form
        // would let a borrowed rename stay borrow-free, but it forces every
        // caller's *signature chain* to name `'src` (three E0621 sites in
        // `main.rs`), so the cure costs more than the copy it saves — a rename
        // happens once per diagnostic, while the lifetime plumbing would be
        // permanent.  `with_file_name` remains the borrow-preserving
        // constructor for callers that do know the lifetime.
        self.file_name = Cow::Owned(file_name.as_ref().to_owned());
        // `stable_id` hashes the file name, so a rename must invalidate a
        // previously computed id (otherwise a stale id keyed to the old name
        // is returned, reintroducing cross-file collisions).
        self.invalidate_stable_id();
    }

    /// Mark the current spans as deliberately set, so a later [`humanize`]
    /// keeps them instead of rebuilding them from the primary labels.
    ///
    /// [`humanize`]: Self::humanize
    pub(crate) fn mark_spans_user_set(&mut self) {
        self.spans_user_set = true;
    }

    pub fn with_foreign_source(
        mut self,
        file: FileId,
        name: impl Into<Cow<'src, str>>,
        source: &'src str,
    ) -> Self {
        let sources = self
            .foreign_sources
            .get_or_insert_with(std::sync::Arc::default);
        let sources = std::sync::Arc::make_mut(sources);
        if !sources.iter().any(|(f, _, _)| *f == file) {
            sources.push((file, name.into(), source));
        }
        self
    }

    pub fn source_for_file(&self, file: FileId) -> Option<&'src str> {
        if self.spans().first().is_some_and(|s| s.file == file) {
            return self.source;
        }
        self.foreign_sources
            .as_deref()
            .and_then(|sources| sources.iter().find(|(f, _, _)| *f == file))
            .map(|(_, _, s)| *s)
    }

    pub fn file_name_for(&self, file: FileId) -> Option<&str> {
        if self.spans().first().is_some_and(|s| s.file == file) {
            return Some(self.file_name());
        }
        self.foreign_sources
            .as_deref()
            .and_then(|sources| sources.iter().find(|(f, _, _)| *f == file))
            .map(|(_, n, _)| n.as_ref())
    }

    /// A deterministic identifier derived from `(code, first primary span,
    /// message, file_name)`, for tools to deduplicate / track diagnostics
    /// across runs.  `file_name` participates so identical code/spans/messages
    /// in different files do not collide.
    ///
    /// **`level` is deliberately excluded**: an error and a warning that share
    /// code, span and message are the same *finding* at two severities, and a
    /// `set_level` escalation should not change its identity (which is why
    /// `set_level` is absent from the invalidation list below).  Consumers that
    /// need to distinguish them must key on `(stable_id, level)`.
    ///
    /// Computed once with FNV-1a and cached in an [`AtomicU64`]; when the
    /// inputs are unchanged, a second call is just a relaxed load. Every
    /// entry point that can mutate the fields above
    /// ([`with_code`](Self::with_code), [`with_code_str`](Self::with_code_str),
    /// [`with_span`](Self::with_span), [`with_additional_span`](Self::with_additional_span),
    /// [`augment`](Self::augment), [`humanize`](Self::humanize),
    /// [`set_file_name`](Self::set_file_name), and
    /// [`adorn_with_span`](Self::adorn_with_span)) clears the cache via
    /// [`invalidate_stable_id`](Self::invalidate_stable_id), so callers
    /// always see a value consistent with the current inputs.
    ///
    /// **Lifecycle:** the id is stable *after* the first `humanize()`.  For a
    /// kind-built diagnostic the constructor's primary span is `None`; the first
    /// derive fills it in from the kind's labels, so that one call changes a hashed
    /// input.  Every later derive (e.g. an LSP language switch) changes nothing
    /// hashed — the message identity comes from the kind's slug and arguments, and
    /// the spans are language-invariant structural data.  A tool that dedups on this
    /// id must therefore read it at or after the render boundary (where the emitter
    /// has humanized), not from a freshly constructed diagnostic.
    pub fn stable_id(&self) -> u64 {
        let cached = self.stable_id_cache.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }
        let id = self.compute_stable_id();
        self.stable_id_cache.store(id, Ordering::Relaxed);
        id
    }

    #[inline]
    fn compute_stable_id(&self) -> u64 {
        // Word-at-a-time mixer, the same core as `content_hash_of` (and the same
        // reason): FNV-1a paid one loop-carried multiply *per byte*, so a
        // 200-byte message was 200 dependent multiplies before the diagnostic
        // could be deduped.  This mixes 8 bytes per step with an `h ^= h >> 29`
        // avalanche, cutting the dependency chain by ~8x on the dedup path.
        //
        // Swapping the core is allowed by this accessor's own contract ("a
        // cross-*run* key, not a persisted one"); what must NOT change is the
        // length-prefix discipline below, which is the fix for the NUL-boundary
        // collision, so it is preserved verbatim — and the tail is additionally
        // tagged with its own length so `"a\0"` and `"a"` cannot collapse into
        // one zero-padded word.
        const K: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut h: u64 = K ^ 0x2545_F491_4F6C_DD1D;
        let mut feed_bytes = |h: &mut u64, bytes: &[u8]| {
            let mut chunks = bytes.chunks_exact(8);
            for chunk in &mut chunks {
                let w = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"));
                *h = (*h ^ w).wrapping_mul(K);
                *h ^= *h >> 29;
            }
            let rem = chunks.remainder();
            if !rem.is_empty() {
                let mut buf = [0u8; 8];
                buf[..rem.len()].copy_from_slice(rem);
                let w = u64::from_le_bytes(buf) ^ (1u64 << (rem.len() * 8));
                *h = (*h ^ w).wrapping_mul(K);
                *h ^= *h >> 29;
            }
        };
        // A macro wrapper because the closure needs `&mut h`, which a second
        // closure could not borrow alongside it.
        macro_rules! feed_bytes {
            ($bytes:expr) => {
                feed_bytes(&mut h, $bytes)
            };
        }
        // Every variable-length field is length-prefixed rather than
        // NUL-separated: a message containing `\0` used to shift the field
        // boundaries, so `message = "a\0b", file = "c"` hashed identically to
        // `message = "a", file = "b\0c"` and two distinct diagnostics collapsed
        // in a tool's dedup set — exactly the failure `stable_id` exists to
        // prevent.  (This changes every stable id once; the field is a
        // cross-*run* key, not a persisted one.)
        // A macro, not a second closure: `feed` already borrows `h` mutably.
        macro_rules! feed_field {
            ($field:expr) => {{
                let field: &str = $field;
                feed_bytes!(&(field.len() as u64).to_le_bytes());
                feed_bytes!(field.as_bytes());
            }};
        }
        match self.code() {
            Some(c) => {
                feed_bytes!(&[1u8]);
                // `canonical`, not `code`: the latter echoes the verbatim
                // spelling for `ErrCode::unknown("e030")`, while
                // `PartialEq`/`Hash` treat it as `E030`.  Hashing the verbatim
                // form would let two `==` diagnostics emit different ids,
                // breaking the dedup contract this accessor exists for.
                feed_field!(c.canonical());
            }
            None => feed_bytes!(&[0u8]),
        }
        match self.spans().first() {
            Some(span) => {
                feed_bytes!(&[1u8]);
                feed_bytes!(&span.file.as_u32().to_le_bytes());
                feed_bytes!(&span.start.to_le_bytes());
                feed_bytes!(&span.end.to_le_bytes());
            }
            None => feed_bytes!(&[0u8]),
        }
        // Feed the message's *identity*, not its rendering.
        //
        // For a kind-built diagnostic the identity is the Fluent slug plus its
        // argument values — every one of them language-invariant — so `Lang::set`
        // (the LSP re-translation contract) cannot change a `stable_id`.  Hashing
        // the rendered text made the same finding emitted under `En` in one run and
        // `Zh` in another carry two different ids, i.e. the exact silent
        // double-report this accessor exists to prevent; the doc only reasoned
        // about excluding `level`.
        //
        // Manually built diagnostics have no slug, so they keep hashing their text
        // (there is nothing more stable to hash).  This changes every kind-built id
        // once — allowed by the accessor's own contract ("a cross-*run* key, not a
        // persisted one"), the same latitude the length-prefix fix used.
        let mut fed_message_identity = false;
        if let Some(kind) = self.kind.as_deref() {
            // An explicit work stack, so the walk is recursive in behaviour but linear in
            // code.  The previous hand-nested arms fed a *nested* `Joined`'s tag and bare
            // slug names only — dropping its pairs and its `Static` parts — so two
            // diagnostics differing solely in an inner slug's argument values (or inner
            // text) hashed identically and silently merged in a tool's dedup set.  `joined`
            // is public, so `joined(vec![joined(vec![...])])` is constructible, which made
            // that a real collision class; ids at depth >= 2 change once, the same latitude
            // as the length-prefix fix.
            //
            // The variant tags (`0`/`1`/`2`) and the length prefixes are exactly the ones the
            // inline arms used, so this refactors the walk, not the encoding.  Parts are
            // pushed reversed so they are fed in order.
            let msg = kind.message();
            // A **frame** stack, not a per-part stack.  `joined()` flattens nesting
            // *depth* to 1 but does nothing to bound *width*, so the previous
            // four-slot-per-part walk ran off its end on the first wide message:
            // `stack[4]` is a hard index panic in release and its own debug assert in
            // debug — and a comptime diagnostic with a traceback is a **flat** `Joined`
            // of `8 + 5 * frames` parts through this exact path, so the flagship
            // comptime feature panicked on a `pub` accessor whose contract is external
            // tool dedup.
            //
            // Each frame is `(remaining slice, cursor)`: width costs nothing, depth
            // still works (a hand-built nested `Joined` is constructible through the
            // public enum), and `SmallVec` keeps the one-frame case — a plain
            // `Static`/`Slug` — allocation-free, spilling only on real nesting.
            let mut frames: smallvec::SmallVec<[(&[DiagMessage<'_>], usize); 4]> =
                smallvec::SmallVec::new();
            frames.push((std::slice::from_ref(&msg), 0));
            loop {
                let Some(frame) = frames.last_mut() else {
                    break;
                };
                // Both fields are `Copy`, so the slice reference is lifted out of the
                // frame's borrow and the borrow is released before the `push` below.
                let (slice, cursor) = (frame.0, frame.1);
                if cursor == slice.len() {
                    frames.pop();
                    continue;
                }
                frame.1 = cursor + 1;
                match &slice[cursor] {
                    DiagMessage::Static(text) => {
                        feed_bytes!(&[0u8]);
                        feed_field!(text.as_ref());
                    }
                    DiagMessage::Slug { slug, pairs } => {
                        feed_bytes!(&[1u8]);
                        feed_field!(slug);
                        for (key, value) in pairs {
                            feed_field!(key);
                            feed_field!(value.as_ref());
                        }
                    }
                    DiagMessage::Joined(parts) => {
                        feed_bytes!(&[2u8]);
                        frames.push((parts.as_ref(), 0));
                    }
                }
            }
            fed_message_identity = true;
        }
        if !fed_message_identity {
            // `message_unchecked`, not `message()`: `stable_id` is a public
            // accessor with no documented humanize precondition, so it must not
            // debug-panic on a kind-built diagnostic read before `humanize()`.
            feed_field!(self.message_unchecked());
        }
        // The file NAME (not `FileId`, which is process-local) so two
        // diagnostics with identical code/span/message in different files do
        // not collide and silently suppress each other in a tool's dedupe
        // set.  `set_file_name`/`with_file_name` therefore invalidate the
        // cache: a caller that reads the id, renames, and reads again must see
        // a new id keyed to the new name, not the stale one.
        feed_field!(self.file_name());
        // The JSON renderer reserves one `id` for its internal-error placeholder
        // (`INTERNAL_ERROR_ID`).  A real id that happens to equal it — the hash
        // is unbounded, so ~2^-64 per diagnostic — would make a tool that
        // dedups or suppresses by `id` merge a genuine diagnostic with the
        // placeholder.  Steer the collision away deterministically rather than
        // asserting: this accessor must stay total.
        let h = if h == crate::diagnostics::renderers::json::INTERNAL_ERROR_ID {
            h ^ 1
        } else {
            h
        };
        h
    }

    /// Clears the [`stable_id`](Self::stable_id) cache.
    ///
    /// Every mutator that changes `code`, the first primary span, `message`, or
    /// **`file_name`** must call this.  Entry points that only touch fields not
    /// covered by the hash (`source`, `level`, `labels`, `related_errors`,
    /// `children`, `suggestions`, `help`, `call_chain`) do not need to — the
    /// cache is naturally immune to those changes.
    ///
    /// `file_name` belongs in the *covered* list: the id hashes the file *name*
    /// (not the `FileId`) so identical diagnostics in different files stay
    /// distinct in a tool's dedup set.  An earlier version of this comment listed
    /// it as unhashed while `set_file_name`/`with_file_name` correctly
    /// invalidated — exactly the reading that would reintroduce the cross-file
    /// collision.  `kind` is special-cased: `with_kind` invalidates because
    /// `humanize` turns the kind into the hashed message.
    #[inline]
    fn invalidate_stable_id(&mut self) {
        self.stable_id_cache.store(0, Ordering::Relaxed);
    }

    /// Whether this diagnostic passes the filter described by `opts`.
    ///
    /// Builds a [`DiagFilter`] from `opts` on every call, which is O(filter_size)
    /// allocations.  For checking many diagnostics against the same options
    /// (e.g. inside a loop), build the filter once with
    /// [`DiagFilter::from_opts`] and call [`matches_filter_with`] instead.
    ///
    /// [`matches_filter_with`]: Diagnostic::matches_filter_with
    pub fn matches_filter(&self, opts: &DiagOpts) -> bool {
        DiagFilter::from_opts(opts).matches(self)
    }

    /// Whether this diagnostic passes a pre-built [`DiagFilter`].
    ///
    /// Loop-safe variant of [`matches_filter`]: the caller builds the filter
    /// once and reuses it across many diagnostics, so the per-diagnostic cost
    /// is O(1) hash lookups rather than O(filter_size) allocations.
    /// Whether this diagnostic passes a pre-built [`DiagFilter`] (the
    /// loop-safe counterpart of [`Diagnostic::matches_filter`]).
    pub fn matches_filter_with(&self, filter: &DiagFilter) -> bool {
        filter.matches(self)
    }
}

/// A child diagnostic attached to a parent `Diagnostic`.
/// Inspired by rustc's `Subdiag` — used for notes, help, or secondary
/// labels that are logically part of the same diagnostic group.
#[derive(Debug, Clone)]
pub struct Subdiag<'src> {
    pub level: DiagnosticLevel,
    /// Plain-text form of the message.
    ///
    /// `None` when the message exists only as [`styled_message`]
    /// ([`styled_message`]: Self::styled_message) — the plain form is then
    /// derived on demand by [`plain_message`](Self::plain_message) instead of
    /// being materialized at construction time.
    pub message: Option<Cow<'src, str>>,
    /// Optional styled version of the message, with highlighted parts.
    /// When present, the glyph renderer uses this instead of the plain
    /// `message` to render rich text with color highlights.
    pub styled_message: Option<StyledString<'src>>,
    /// Optional source span for the child.  Retained for tooling and other
    /// backends; the human renderer renders only the message, so this is
    /// currently carried but not drawn.
    pub span: Option<Span>,
    /// Child labels.  Retained for tooling and other backends; the human
    /// renderer renders only the message, so these are currently carried
    /// but not drawn.
    pub labels: CovariantVec<'src, Label<'src>, 1>,
}

/// A related error displayed alongside the primary diagnostic in a single
/// merged box.  Each related error has its own error code, message, and
/// optional source span — for example, a type mismatch [E030] aggregated
/// into a duplicate definition [E019] diagnostic.
#[derive(Debug, Clone)]
pub struct RelatedError<'src> {
    pub code: Option<ErrCode>,
    pub message: Cow<'src, str>,
    pub span: Option<Span>,
    /// Optional label for the related span.  Retained for tooling and other
    /// backends; the human renderer shows only `message`/`code`.
    pub label: Option<Cow<'src, str>>,
}

impl<'src> Subdiag<'src> {
    /// Create a new sub-diagnostic.
    pub fn new(level: DiagnosticLevel, message: impl Into<Cow<'src, str>>) -> Self {
        Subdiag {
            level,
            message: Some(message.into()),
            styled_message: None,
            span: None,
            labels: CovariantVec::new(),
        }
    }

    /// Plain-text message, borrowing the stored form when present and
    /// deriving it from [`styled_message`](Self::styled_message) otherwise.
    pub fn plain_message(&self) -> Cow<'_, str> {
        match (&self.message, &self.styled_message) {
            (Some(message), _) => Cow::Borrowed(message.as_ref()),
            (None, Some(styled)) => Cow::Owned(styled.to_plain()),
            (None, None) => Cow::Borrowed(""),
        }
    }

    /// Create a note sub-diagnostic.
    pub fn note(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    /// Create a help sub-diagnostic.
    pub fn help_sub(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    /// Attach a source span to this sub-diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a label to this sub-diagnostic.
    pub fn with_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }
}

/// Humanize a slice in parallel above a shared threshold.
///
/// `keep` selects which diagnostics are humanized — the emitter's `&mut` entry points pass their
/// `DiagFilter`, the collector passes `|_| true`.  Both sites used to carry their own copy of the
/// threshold, the chunking formula and a cached `available_parallelism`, which is exactly the
/// drift this helper removes.
///
/// Sound for the reasons the collector documents: `Lang` is a process-global atomic, the chunks
/// are disjoint `&mut`, every per-diagnostic cache is diagnostic-local, and each `humanize` goes
/// through [`Diagnostic::humanize_contained`], so one broken diagnostic costs a placeholder
/// instead of the batch.  Output is order-independent by construction — each call touches only
/// its own diagnostic — so the split cannot change a byte.
pub(crate) fn humanize_slice_parallel<'a, F>(diags: &mut [Diagnostic<'a>], keep: F)
where
    F: Fn(&Diagnostic<'a>) -> bool + Sync,
{
    /// A spawn costs ~50 µs, so chunking pays only once the unit count is well past it.
    const PARALLEL_HUMANIZE_MIN: usize = 64;
    // `available_parallelism` is a `sched_getaffinity` syscall on Linux and a process's
    // affinity mask does not change under it, so the answer is cached.
    static AVAILABLE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threads = if diags.len() >= PARALLEL_HUMANIZE_MIN {
        (*AVAILABLE.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get())))
            .min(diags.len() / PARALLEL_HUMANIZE_MIN)
            .max(1)
    } else {
        1
    };
    if threads > 1 {
        let chunk = diags.len().div_ceil(threads);
        let keep = &keep;
        std::thread::scope(|scope| {
            for slice in diags.chunks_mut(chunk) {
                scope.spawn(move || {
                    for diag in slice {
                        if keep(diag) {
                            diag.humanize_contained();
                        }
                    }
                });
            }
        });
    } else {
        for diag in diags.iter_mut() {
            if keep(diag) {
                diag.humanize_contained();
            }
        }
    }
}

impl<'src> Diagnostic<'src> {
    // ── Constructors ───────────────────────────────────────────────

    pub fn new(level: DiagnosticLevel, message: impl Into<Cow<'src, str>>) -> Self {
        Diagnostic {
            level,
            message: message.into(),
            code: None,
            kind: None,
            spans: MultiSpan::default(),
            help: None,
            suggestions: CovariantVec::new(),
            labels: CovariantVec::new(),
            call_chain: None,
            source: None,
            file_name: "<input>".into(),
            children: Vec::new(),
            related_errors: Vec::new(),
            foreign_sources: None,
            humanized: false,
            humanized_lang: 0,
            spans_user_set: false,
            help_user_set: false,
            suggestions_user_set: false,
            kind_derived_labels: 0,
            stable_id_cache: AtomicU64::new(0),
        }
    }

    pub fn error(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    pub fn fatal(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Fatal, message)
    }

    pub fn warning(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Create an error diagnostic from a structured [`DiagnosticKind`].
    ///
    /// # Lazy humanization
    ///
    /// `message` / `labels` / `spans` / `help` / `suggestions` stay
    /// unfilled until [`humanize`](Self::humanize) runs.  The constructor
    /// deliberately skips that step so:
    ///
    /// - **Filtered-out diagnostics pay nothing** — those dropped by
    ///   `max_errors` or suppressed warnings (`can_emit_warnings = false`)
    ///   are never humanized.
    /// - **i18n language resolves at render time** — Fluent runs against
    ///   the *current* `Lang`, so a long-lived process (LSP) can
    ///   `Lang::set` between emits without freezing messages in the
    ///   language that happened to be active at construction.
    ///
    /// Humanization is the *caller's* step for batches: [`Emitter::emit`] (the
    /// single-diagnostic path) humanizes at its render boundary, but every batch entry
    /// point renders `&Diagnostic` and cannot mutate, so it requires diagnostics that
    /// have already been humanized (`DiagCtxt::humanize_pending`).  The `&mut` batch
    /// variants (`emit_all_to`, `emit_all_streaming_to`) do it for you.  Reading these
    /// fields before that boundary sees the constructor's defaults: an empty message and
    /// a `None` primary span.
    /// `kind` is retained (never dropped) as the structured source of
    /// truth so `humanize()` can re-run if the kind is replaced.
    pub fn error_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Error, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    /// Create a fatal diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn fatal_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Fatal, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    /// Create a warning diagnostic from a structured [`DiagnosticKind`].
    ///
    /// See [`error_kind`](Self::error_kind) for the lazy-humanization
    /// contract: derived fields stay unfilled until `humanize()` runs.
    pub fn warning_kind(kind: DiagnosticKind<'src>) -> Self {
        let mut diag = Self::new(DiagnosticLevel::Warning, "");
        diag.kind = Some(std::sync::Arc::new(kind));
        diag
    }

    pub fn help_diag(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Help, message)
    }

    pub fn note(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    pub fn info(message: impl Into<Cow<'src, str>>) -> Self {
        Self::new(DiagnosticLevel::Info, message)
    }

    // ── Builder methods ────────────────────────────────────────────

    /// Set the primary source span for this diagnostic.
    ///
    /// Spans set here are preserved across [`humanize`](Self::humanize);
    /// they are not overwritten by spans derived from a `kind`.
    ///
    /// This replaces the whole span list, dropping any spans previously added
    /// by [`with_additional_span`](Self::with_additional_span); call
    /// `with_span` first when combining the two.
    pub fn with_span(mut self, span: Span) -> Self {
        self.spans = MultiSpan::new(span);
        self.spans_user_set = true;
        self.invalidate_stable_id();
        self
    }

    /// Add a secondary primary span (for multi-location diagnostics).
    ///
    /// Like [`with_span`](Self::with_span), this survives `humanize`.
    pub fn with_additional_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self.spans_user_set = true;
        self.invalidate_stable_id();
        self
    }

    /// Set the error code (e.g. `ErrCode::TypeMismatch` → `E030`).
    pub fn with_code(mut self, code: ErrCode) -> Self {
        self.code = Some(code);
        self.invalidate_stable_id();
        self
    }

    /// Convenience: set error code by string (backward compat).
    /// Prefer `with_code(ErrCode::new(...))` for the new API.
    ///
    /// Unlike `ErrCode::new`, this never panics on an unknown code: the code
    /// is validated against the table and, if absent, stored as-is so
    /// `title()` / `category()` / `explain()` fall back to their generic
    /// "unknown" values.  Safe for untrusted or dynamically-built strings.
    pub fn with_code_str(mut self, code: impl AsRef<str>) -> Self {
        let code = code.as_ref();
        // An empty code is neither "code-less" nor a code: `ErrCode::unknown("")` made
        // `Display` print empty brackets (`[ ERROR] …`) and gave the diagnostic a filter
        // identity (`allow_codes` containing `""`) that no CLI spelling can produce.
        // Code-less is a well-defined state; this maps the degenerate spelling onto it.
        self.code = if code.is_empty() {
            None
        } else {
            Some(match ErrCode::try_new(code) {
                Ok(ec) => ec,
                Err(e) => ErrCode::unknown(e.0.into_owned()),
            })
        };
        self.invalidate_stable_id();
        self
    }

    /// Add a help message.
    pub fn with_help(mut self, help: impl Into<Cow<'src, str>>) -> Self {
        self.help = Some(help.into());
        // User help survives `humanize`; kind-derived help fills only
        // when the caller has not supplied its own.
        self.help_user_set = true;
        self
    }

    /// Add a single suggestion with the default (`Unspecified`) applicability
    /// and `ShowAlways` style.
    ///
    /// Classify via [`with_suggestion_style`](Self::with_suggestion_style)
    /// when the applicability is known (e.g. compiler-derived fixes).
    pub fn with_suggestion(mut self, suggestion: impl Into<Cow<'src, str>>) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability: Applicability::default(),
            style: SuggestionStyle::ShowAlways,
            span: None,
            replacement: None,
        });
        self.suggestions_user_set = true;
        self
    }

    /// Add a suggestion with explicit applicability and style.
    pub fn with_suggestion_style(
        mut self,
        suggestion: impl Into<Cow<'src, str>>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: suggestion.into(),
            applicability,
            style,
            span: None,
            replacement: None,
        });
        self.suggestions_user_set = true;
        self
    }

    /// Add a **structural** suggestion: a source span plus its replacement
    /// text.  The glyph renderer emits this as a two-line diff inside the
    /// source section, prefixed with a dim `√` (U+221A SQUARE ROOT,
    /// MachineApplicable) or `?` (MaybeIncorrect) applicability glyph on
    /// each diff line.  Single-line spans only: multi-line spans fall back
    /// to the text-only suggestion path.
    pub fn with_span_suggestion(
        mut self,
        span: Span,
        replacement: impl Into<Cow<'src, str>>,
        message: impl Into<Cow<'src, str>>,
        applicability: Applicability,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: message.into(),
            applicability,
            style: SuggestionStyle::ShowAlways,
            span: Some(span),
            replacement: Some(replacement.into()),
        });
        self.suggestions_user_set = true;
        self
    }

    /// Replace suggestions list.
    pub fn with_suggestions(mut self, suggestions: Vec<Suggestion<'src>>) -> Self {
        self.suggestions = suggestions.into();
        self.suggestions_user_set = true;
        self
    }

    /// Add a labeled sub-span for multi-span highlighting.
    ///
    /// **Decision, pinned by `user_primary_labels_do_not_extend_the_derived_spans`:** a user label
    /// does *not* contribute to [`MultiSpan`](crate::diagnostics::MultiSpan) when `humanize` derives
    /// the span set from the kind.  Only the kind's own `Primary` labels populate it; user labels are
    /// drawn but never widen the section's span set.  This is a choice, not an oversight — including
    /// them would make a section's headers depend on decorative labels a caller added for
    /// highlighting, and would silently change output for every type-built diagnostic that also
    /// carries a user label.  Callers that want the span set to include their span use
    /// [`with_span`](Self::with_span), which also disables derivation.
    pub fn with_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::new(span, label));
        self
    }

    /// Add a secondary (tilde-underline) label.
    pub fn with_secondary_label(mut self, span: Span, label: impl Into<Cow<'src, str>>) -> Self {
        self.labels.push(Label::secondary(span, label));
        self
    }

    /// Attach a related error to be displayed alongside this diagnostic
    /// in a single merged box (e.g. a type mismatch aggregated into a
    /// duplicate definition diagnostic).
    pub fn with_related(mut self, err: RelatedError<'src>) -> Self {
        self.related_errors.push(err);
        self
    }

    /// Attach a call chain (Zig-style `referenced by`).
    pub fn with_call_chain(mut self, chain: CallChain<'src>) -> Self {
        self.call_chain = Some(std::sync::Arc::new(chain));
        self
    }

    /// Attach source text for `^`-underline rendering.
    ///
    /// The source is borrowed from the caller (e.g. the driver's source
    /// buffer), so no copy is made.
    pub fn with_source(mut self, source: &'src str) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the source file name for the `┌─ <file>:1:1` location header.
    pub fn with_file_name(mut self, file_name: impl Into<Cow<'src, str>>) -> Self {
        self.file_name = file_name.into();
        self.invalidate_stable_id();
        self
    }

    /// Set the structured diagnostic kind (Vale-style ADT).
    ///
    /// When present, the humanizer can derive `message`, `labels`, and other
    /// fields from the kind's structured data, producing more precise error
    /// messages than a flat string.
    pub fn with_kind(mut self, kind: DiagnosticKind<'src>) -> Self {
        self.kind = Some(std::sync::Arc::new(kind));
        // A new kind must be materialized by the next `humanize()`, and the
        // cached `stable_id` (hashed over `(code, span, message)`) is stale
        // until then — invalidate it exactly like every other mutator.
        self.humanized = false;
        self.invalidate_stable_id();
        self
    }

    /// Add a highlighted note child (styled text with type/name highlights).
    ///
    /// The plain-text form is derived lazily by
    /// [`Subdiag::plain_message`], so a note that is never rendered (filtered
    /// out by `max_errors` / `only_codes`, say) never allocates it.
    pub fn with_highlighted_note(mut self, styled: StyledString<'src>) -> Self {
        self.children.push(Subdiag {
            level: DiagnosticLevel::Note,
            message: None,
            styled_message: Some(styled),
            span: None,
            labels: CovariantVec::new(),
        });
        self
    }

    /// Add a highlighted help child (styled text with type/name highlights).
    pub fn with_highlighted_help(mut self, styled: StyledString<'src>) -> Self {
        self.children.push(Subdiag {
            level: DiagnosticLevel::Help,
            message: None,
            styled_message: Some(styled),
            span: None,
            labels: CovariantVec::new(),
        });
        self
    }

    /// Run the humanizer on `self.kind` (if present) to populate `message`,
    /// `labels`, `help`, and `suggestions` from the structured data.
    ///
    /// Idempotent for a fixed language: a `None` kind latches `humanized`
    /// and repeated calls short-circuit, so the emitter rendering the same
    /// batch twice pays nothing.  `with_kind` resets the latch so a
    /// re-kind'ed diagnostic is re-derived.
    ///
    /// The latch is keyed to the current [`Lang`] as well: a diagnostic
    /// humanized in one language re-derives `message`/`labels`/`help` in
    /// the new language when the process language changes between emits —
    /// the LSP re-translation contract documented on
    /// [`error_kind`](Self::error_kind).  The `kind` is retained (never
    /// dropped) as the structured source of truth that makes this possible;
    /// the one-`clone`-per-humanize cost replaces the old take-and-drain,
    /// which froze the rendered language forever and left `kind()` empty.
    ///
    /// Until this runs, `message` / `labels` / `spans` / `help` /
    /// `suggestions` of a `*_kind`-built diagnostic hold their
    /// constructor defaults.  The [`Emitter`] calls this at its render
    /// boundary; direct readers of those fields must call it themselves.
    ///
    /// **Spans, help and suggestions share one precedent** — the `spans_user_set`
    /// / `help_user_set` / `suggestions_user_set` flags: once a builder supplies
    /// one, `humanize` preserves it and the kind fills only the fields still
    /// unset.  So attaching the kind first and the span/help second is the
    /// supported builder ordering, and a `with_help` survives a later `with_kind`
    /// too (the flags are sticky).  This is not an "exception" for spans: it is
    /// the rule for every user-owned field, and the kinds derive only what no
    /// caller claimed.  Overwriting user help here silently lost data once
    /// already (the user-help regression the inline comment records).
    ///
    /// `message` is **kind-owned** as well: text passed to a message-taking
    /// constructor (`Diagnostic::error("…")`) is a placeholder that this call
    /// replaces with the kind-derived wording.  Stated because it is a decision,
    /// not an accident: `help`/`suggestions`/`spans` carry `*_user_set` flags so
    /// a deliberate override survives, but for `message` the kind *is* the
    /// wording, and a flag would make every kind-built diagnostic keep its stale
    /// placeholder forever (with no way to say "derive it").  Pinned by
    /// `humanize_replaces_a_constructor_message`.
    /// `humanize` with this crate's containment policy applied.
    ///
    /// `Emitter::emit` already wraps its single-diagnostic humanize in `catch_unwind` and
    /// documents "the same policy as the batch paths" — which was false: the batch entry
    /// points (`emit_all_to`, `emit_all_streaming_to`) and `DiagCtxt::humanize_pending*`
    /// let a panic abort the entire emit.  Worse, `humanize` mutates before it can fail (it
    /// drains `labels` and `mem::take`s `message`), so the abort also left the batch
    /// half-derived.
    ///
    /// One broken diagnostic now costs one placeholder line, which is the subsystem's
    /// stated worst case ("a fallback for the fallback must be total").  This family has
    /// already shipped one panic on a `pub` path (`stable_id`'s fixed stack), so the
    /// exposure is not hypothetical.
    pub(crate) fn humanize_contained(&mut self) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.humanize())).is_err() {
            self.poison_for_render();
        }
    }

    /// Make a diagnostic renderable after `humanize` panicked mid-way.
    ///
    /// Deliberately **total**: every field `humanize` can have touched before the panic is
    /// overwritten, so no half-derived state survives (a drained label list, a `mem::take`n
    /// message, a stale derived-label count).  Spans and user-set structural fields are left
    /// alone — they are the diagnostic's identity — and the placeholder message is what makes
    /// the failure visible instead of empty.
    fn poison_for_render(&mut self) {
        const PLACEHOLDER: &str = "<internal: diagnostic humanization panicked>";
        self.humanized = true;
        self.humanized_lang = crate::diagnostics::i18n::Lang::current() as u8;
        self.message = Cow::Borrowed(PLACEHOLDER);
        self.labels.clear();
        self.help = None;
        self.suggestions.clear();
        self.kind_derived_labels = 0;
        // The id hashes the message, so it must be recomputed rather than left describing
        // text that no longer exists.
        self.invalidate_stable_id();
    }

    pub fn humanize(&mut self) {
        let lang = crate::diagnostics::i18n::Lang::current() as u8;
        if self.humanized {
            // No kind: message/labels were set manually — there is nothing
            // to re-resolve in another language.  Same language: the latch
            // holds and the call is free.
            if self.kind.is_none() || self.humanized_lang == lang {
                return;
            }
        }
        // Only the *first* derive can change something `stable_id` hashes: it fills
        // in the derived primary span (the constructor default is `None`), while a
        // re-derive under another language changes nothing hashed at all — the
        // kind-built id is computed from the kind's slug+args, its structural label
        // spans (language-invariant `Span` data) and the untouched code/file name.
        // The old condition invalidated unconditionally, so every LSP language switch
        // paid a full FNV recompute per diagnostic for an identical value — the exact
        // waste the neighbouring no-kind rationale already rejected.
        if self.kind.is_some() && !self.humanized {
            self.invalidate_stable_id();
        }
        let Some(kind_arc) = self.kind.as_ref().map(std::sync::Arc::clone) else {
            // No kind: message/labels were set manually.  Latch so later
            // calls skip the `kind` lookup entirely.
            self.humanized = true;
            self.humanized_lang = lang;
            return;
        };
        // `kind` stays installed throughout — no `mem::take`, no restore guard.  The
        // derive reads through a cheap `Arc` clone and its four accessors are
        // infallible field reads, so the fallible work (Fluent `render_into_owned`)
        // runs after this borrow ends.  The `RestoreKindOnDrop` guard that used to
        // live here protected a region that cannot panic from machinery
        // (`humanize_into`) that is no longer on this path at all.
        let message = kind_arc.message();
        let labels = kind_arc.labels();
        // Skipped when the caller overrode the field: `help()` runs the payload cap,
        // `join_with_or` and the Fluent `did_you_mean` chain (several allocations) and
        // `suggestions()` builds a vector — both were then discarded by the
        // `help_user_set`/`suggestions_user_set` guards below.
        let help = if self.help_user_set {
            None
        } else {
            kind_arc.help()
        };
        let suggestions = if self.suggestions_user_set {
            // `suggestions()` yields an owned `Vec`; the guard below discards it, so
            // skipping the derive means never building it in the first place.
            Vec::new()
        } else {
            kind_arc.suggestions()
        };
        drop(kind_arc);
        // Static messages move straight into `message`: a borrowed static
        // stays borrowed (zero allocation), an owned one is moved — only
        // the Fluent-slug path needs the render buffer.
        self.message = match message {
            DiagMessage::Static(cow) => cow,
            slug => {
                // Reuse the previous owned buffer: an LSP language switch
                // re-humanizes every diagnostic, and each slug render used to
                // allocate a fresh `String` and drop the old one.
                let mut buf = match std::mem::take(&mut self.message) {
                    Cow::Owned(mut s) => {
                        s.clear();
                        s
                    }
                    Cow::Borrowed(_) => String::new(),
                };
                slug.render_into_owned(&mut buf);
                Cow::Owned(buf)
            }
        };
        // Populate `spans` from the primary labels so the source-context
        // renderer knows which lines to display — but only when the spans
        // are still the constructor default.  Spans attached by a builder
        // (`with_span`, `with_additional_span`, `augment`,
        // `adorn_with_span`) were set deliberately and are preserved.
        // Spans are derived from the *same* label vector the block below
        // consumes, so the population is folded into that loop (see
        // `derive_spans`) instead of walking the labels twice.
        let derive_spans = !self.spans_user_set;
        if derive_spans {
            self.spans.clear();
        }
        // Labels: replace exactly the previous kind-derived block and
        // re-append user labels after the fresh block — kind-derived first,
        // then user-added (`with_label` / `with_secondary_label` /
        // `augment` only ever append), the order the builders establish.
        // Truncating just the derived block makes re-humanization on a
        // language change idempotent: derived labels refresh, user labels
        // neither duplicate nor vanish.  (A >255-label kind would saturate
        // the counter; kinds derive a handful of labels, so the bound is
        // unreachable in practice.  `min` below also tolerates external
        // mutation through `labels_mut()` that shrank the vector under our
        // bookkeeping.)
        let prev_derived = (self.kind_derived_labels as usize).min(self.labels.len());
        // Reuse both allocations instead of building fresh containers: move the
        // user-added tail aside (`drain` leaves the buffer's capacity intact),
        // truncate to zero, push the fresh derived block, then re-append the user
        // labels.  Not an in-place overwrite — an equal-count copy-over would be a
        // different fast path, and nothing needs it: the capacity, which is what
        // actually costs, is already reused.
        // Inline capacity 4, not 1: the staging vector holds only the user-added
        // tail, and an LSP language switch re-humanizes every diagnostic, so a
        // `^`+`~` pair (the common two-label shape) paid a heap allocation per
        // re-humanize.  This is a local type, so it does not touch `Diagnostic`'s
        // 384-byte footprint.
        let mut user: CovariantVec<'src, Label<'src>, 4> = CovariantVec::new();
        for lbl in self.labels.drain(prev_derived..) {
            user.push(lbl);
        }
        self.labels.truncate(0);
        for lbl in labels {
            if derive_spans && matches!(lbl.kind, AnnotationKind::Primary) {
                self.spans.push(lbl.span);
            }
            self.labels.push(lbl);
        }
        debug_assert!(
            self.labels.len() <= u32::MAX as usize,
            "kind derives more labels than u32 can represent"
        );
        self.kind_derived_labels = self.labels.len().min(u32::MAX as usize) as u32;
        for lbl in user.drain(..) {
            self.labels.push(lbl);
        }
        // help/suggestions are re-derived from `kind` on every humanize: a
        // diagnostic can be re-kind'ed, and guarded rebuilding would keep
        // k1's derived notes on a diagnostic that no longer is k1.  Caller-
        // supplied help/suggestions are preserved, matching the spans
        // precedent — overwriting them silently lost user data.
        if !self.help_user_set {
            self.help = help;
        }
        if !self.suggestions_user_set {
            // Reuse the suggestions allocation across re-humanization
            // (language switches in an LSP re-derive every diagnostic).
            self.suggestions.clear();
            self.suggestions
                .extend(suggestions.into_iter().map(|s| Suggestion {
                    message: s,
                    // `humanize_into` yields plain strings with no
                    // per-suggestion classification, so these are *not* known
                    // to be mechanically applicable.  `Unspecified` is the
                    // crate's documented safe default: an auto-apply pass must
                    // not treat them as machine-applicable without review.
                    applicability: Applicability::Unspecified,
                    style: SuggestionStyle::ShowAlways,
                    span: None,
                    replacement: None,
                }));
        }
        self.humanized = true;
        self.humanized_lang = lang;
    }

    // ── Query methods ──────────────────────────────────────────────

    pub fn is_error(&self) -> bool {
        self.level.is_error()
    }

    pub fn is_warning(&self) -> bool {
        matches!(self.level, DiagnosticLevel::Warning)
    }

    /// Change the severity of this diagnostic.
    ///
    /// This is the single place through which severity changes should flow, so
    /// that any state derived from `level` (now or in the future) is kept
    /// consistent.  Used by [`DiagCtxt`](crate::diagnostics::DiagCtxt)'s
    /// downgrade helpers.
    pub fn set_level(&mut self, level: DiagnosticLevel) {
        self.level = level;
    }

    /// Get the main location string for display.
    pub fn location_string(&self) -> String {
        let mut out = String::new();
        self.write_location_into(&mut out);
        out
    }

    /// [`location_string`](Self::location_string) into an existing buffer.
    ///
    /// `Display` for the diagnostic already writes straight into the formatter;
    /// this is the same treatment for the "at …" fragment that renderers write
    /// once per span.
    pub fn write_location_into(&self, out: &mut String) {
        use swrite::{SWrite, swrite};
        if let Some(s) = self.spans.first() {
            swrite!(out, "at {s}");
        }
    }

    /// Merge context from another diagnostic into `self`.
    ///
    /// If `other` has a span, source, labels, or a call chain and `self`
    /// does not yet have them, they are copied over.  This is useful for
    /// progressively enriching a diagnostic as it passes through compiler
    /// passes (the "context augmentation" pattern inspired by Austral).
    pub fn augment(&mut self, other: &Diagnostic<'src>) {
        // Copy **every** primary span, not just the first: `augment` exists for
        // context augmentation, and multi-location diagnostics (duplicate
        // definition + previous definition, built with `with_additional_span`)
        // are exactly where it is used.  Keeping only `first()` silently dropped
        // every location after the anchor.
        let other_primary = other.spans.primary_ref();
        if self.spans.first().is_none()
            && let Some(&first_span) = other_primary.first()
        {
            self.spans = MultiSpan::new(first_span);
            for &span in &other_primary[1..] {
                self.spans.push(span);
            }
            self.spans_user_set = true;
            self.invalidate_stable_id();
        }
        if self.source.is_none() {
            self.source = other.source;
        }
        if self.call_chain.is_none() {
            self.call_chain.clone_from(&other.call_chain);
        }
        // Merge labels that aren't already present.  The straightforward
        // scan is O(self × other) in the label counts; the keyed set is
        // O(self + other).  The threshold guards *both* sides: a small
        // `self` merged with a large `other` would otherwise scan the
        // growing `to_add` on every iteration and degrade to O(|other|²).
        // Small merges on both sides keep the allocation-free scan (the
        // common case is one or two labels).
        // `to_add` collects first: the key set borrows `self.labels`, so it
        // must be dropped before the pushes.
        if !other.labels.is_empty() {
            // Inline capacity covers the whole admitted range: `to_add` collects from
            // `other` **alone**, and the scan threshold admits up to 16 labels there,
            // so an 8-slot buffer spilled at the 9th addition — inside the band it
            // exists to keep allocation-free ("8 per side" was arithmetic about a sum,
            // not about this vector).  `&Label` is 8 bytes, so 16 slots are 128 bytes
            // of stack versus one malloc.
            let mut to_add: SmallVec<[&Label<'src>; 16]> = SmallVec::new();
            // Threshold 16, not 8: the scan path allocates nothing, and an LSP
            // re-emit merges two `^`+`~`-scale label sets (≤ 6) far more often than
            // it builds a 16-element set.  32 would make the O(n·m) scan carry
            // `&str` comparisons for 1024 pairs, which is where the keyed path
            // starts winning.
            if self.labels.len() <= 16 && other.labels.len() <= 16 {
                for lbl in &other.labels {
                    // `to_add` is checked too: a duplicate *within* `other`
                    // must not be added twice (the scan sees earlier pushes).
                    if !self
                        .labels
                        .iter()
                        .chain(to_add.iter().copied())
                        .any(|l| l.dedup_key() == lbl.dedup_key())
                    {
                        to_add.push(lbl);
                    }
                }
            } else {
                let mut seen: rustc_hash::FxHashSet<(FileId, u32, u32, &str)> =
                    self.labels.iter().map(Label::dedup_key).collect();
                for lbl in &other.labels {
                    if seen.insert(lbl.dedup_key()) {
                        to_add.push(lbl);
                    }
                }
            }
            for lbl in to_add {
                self.labels.push(lbl.clone());
            }
        }
        if let Some(other_sources) = other.foreign_sources.as_deref() {
            // No intermediate `Vec`: `other_sources` borrows `other` (a shared
            // reference to a *different* diagnostic), so mutating ours inside the
            // loop is fine — `Arc::make_mut` may clone **our** vector, which the
            // borrow does not touch.  (An earlier draft cloned the tuples first; that
            // shape is gone, and the sentence describing it only contradicted this one.)
            // One `make_mut` for the whole loop, not one per tuple: the old form
            // re-checked the strong count on every iteration, and its dedup was an
            // O(n·m) scan — the same quadratic shape the labels merge above was
            // explicitly reworked away from.  Same both-sides threshold: small
            // merges keep the allocation-free scan (0–2 foreign sources is the
            // common case), larger ones use a keyed set.
            let self_sources = self
                .foreign_sources
                .get_or_insert_with(std::sync::Arc::default);
            let self_sources = std::sync::Arc::make_mut(self_sources);
            if self_sources.len() <= 8 && other_sources.len() <= 8 {
                for (file, name, source) in other_sources.iter() {
                    if !self_sources.iter().any(|(f, _, _)| f == file) {
                        self_sources.push((*file, name.clone(), *source));
                    }
                }
            } else {
                let mut seen: rustc_hash::FxHashSet<crate::ast::FileId> =
                    self_sources.iter().map(|(f, _, _)| *f).collect();
                for (file, name, source) in other_sources.iter() {
                    if seen.insert(*file) {
                        self_sources.push((*file, name.clone(), *source));
                    }
                }
            }
        }
    }
}

/// Wrap a function that may produce a diagnostic, augmenting any errors
/// it emits with the given span and source context.
///
/// This is the Ponent equivalent of Austral's `adorn_error_with_span`:
/// instead of catching exceptions, the closure receives a `&mut DiagCtxt`
/// and any diagnostics it pushes are augmented with `span` and `source`.
///
/// # Convention
///
/// This function **augments the last diagnostic** pushed to `ctxt` when the
/// closure returns `Err`.  It relies on the invariant that the diagnostic
/// produced by the failing closure is the most recent one in the context.
/// If the closure pushes multiple diagnostics, only the last one is
/// augmented — earlier ones are left untouched.
///
/// This is an intentional design choice: the "root cause" diagnostic is
/// typically the last one pushed, and augmenting it with the outer call
/// site's span/source is the most useful behaviour.  Callers that push
/// multiple diagnostics before failing should order them so that the
/// one needing augmentation is last.
///
/// # Guarantee
///
/// `EmissionGuaranteed` can only be obtained by calling `DiagCtxt::push()`.
/// Note, however, that `push` may return a *suppressed* guarantee without
/// recording anything — a warning pushed while `can_emit_warnings` is false.
/// Therefore a growing diagnostic count is *not* guaranteed on the `Err`
/// path; augmentation is applied only when the closure actually grew the
/// context (diagnostic count strictly larger than before the call), so a
/// pre-existing diagnostic is never silently re-spanned by an unrelated
/// failure.
///
/// # Example
/// ```ignore
/// adorn_with_span(ctxt, span, source, |ctxt| {
///     check_something(ctxt)?;
///     Ok(())
/// });
/// ```
pub fn adorn_with_span<'src, T>(
    ctxt: &mut DiagCtxt<'src>,
    span: Span,
    source: Option<&'src str>,
    f: impl FnOnce(&mut DiagCtxt<'src>) -> Result<T, EmissionGuaranteed>,
) -> Result<T, EmissionGuaranteed> {
    let before = ctxt.len();
    let result = f(ctxt);
    // If the closure failed, augment the last diagnostic with context.
    if result.is_err() {
        // The closure may have returned `Err` from a *suppressed* warning
        // push (which records nothing) — or from a check that failed without
        // pushing at all.  In both cases augmenting the last diagnostic would
        // touch an *earlier, unrelated* one and corrupt its span/source, so
        // augment only when the closure actually grew the context.
        //
        // `augment_last_diagnostic` is the collector's purpose-built mutator:
        // this call site needs the span/source fields, and a raw
        // `&mut Diagnostic` would also expose level rewrites that silently
        // desynchronise the O(1) severity tallies.
        if ctxt.len() > before {
            ctxt.augment_last_diagnostic(span, source);
        }
    }
    result
}

// ── ErrCode is now a string-based code — use `ErrCode::E030` ──
// The old `from_str` method with enum variants was removed in favour of
// the lookup table in `error_code.rs`.  Any string is accepted as a code.

// ── Display ───────────────────────────────────────────────────────

impl fmt::Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `Display` reads `self.message` directly, while every message
        // accessor enforces the humanize precondition.  A kind-built
        // diagnostic rendered before `humanize()` would otherwise silently
        // print an empty message; make that contract violation loud in debug
        // builds instead.
        debug_assert!(
            self.kind.is_none() || self.humanized,
            "Display on a kind-built Diagnostic before humanize(): message is empty"
        );
        // `canonical`, not `code`: filtering (`--allow`/`--only-codes`) and
        // `stable_id` both key on the canonical spelling, so printing the
        // verbatim one made `unknown("e030")` render as `e030` while behaving
        // as `E030` — the same logical diagnostic looking like two.
        let code_str = self.code.as_ref().map(|c| c.canonical()).unwrap_or("?");
        // `label_upper` is the per-language cached Unicode-uppercased label: the
        // old `label().to_uppercase()` allocated a fresh `String` per render for
        // a value fully determined by `(level, language)`.
        let level_str = self.level.label_upper();

        // The `debug_assert` above is compiled out in release, so a caller that
        // skipped `humanize()` would print `[E030 ERROR]` with **no message** — a
        // user-visible error that says nothing, and only in release.  Fall back to
        // the kind's own message (the source `humanize` would have rendered), and
        // keep the assert so CI still fails loudly on the contract breach.
        let fallback = self.unhumanized_fallback();
        let msg: &str = fallback.as_deref().unwrap_or(&self.message);

        // Write the span straight into the formatter instead of
        // `format!("at {s}")` into a temporary — same bytes, no allocation.
        match self.spans.first() {
            Some(span) => writeln!(f, "[{code_str} {level_str}] {msg} at {span}")?,
            None => writeln!(f, "[{code_str} {level_str}] {msg}")?,
        }

        // Labels.  `underline_char` is an `Option` because `Help` has no
        // underline; this flat text form keeps its historical single space for
        // that case (the glyph renderer supplies the `| help:` form instead).
        for lbl in &self.labels {
            writeln!(
                f,
                "  {}--> {}: {}",
                lbl.underline_char().unwrap_or(' '),
                lbl.span,
                lbl.message
            )?;
        }

        // Call chain
        if let Some(ref chain) = self.call_chain
            && !chain.is_empty()
        {
            write!(f, "{}", chain.render(false, self.source, &self.file_name))?;
        }

        // Help
        if let Some(help) = &self.help {
            writeln!(f, "help: {}", help)?;
        }

        // Suggestions
        for suggestion in &self.suggestions {
            writeln!(f, "suggestion: {}", suggestion)?;
        }

        Ok(())
    }
}

/// Render the `--explain` output for a given error code string.
/// When `use_color` is true, output includes syntax highlighting and ANSI
/// colors; otherwise the body is plain text with no color codes.
pub fn explain_error_code(code_str: &str, use_color: bool) -> String {
    // Share the process-wide probe the emit paths use (one `OnceLock` load)
    // instead of re-running `detect_color_depth` per `--explain` call — the two
    // surfaces describe the same terminal, so they must not be able to disagree.
    let opts = detected_opts();
    // All three terminal-dependent parameters come from the same snapshot now.
    // This wrapper used to forward `color_depth` and then hardcode the *other*
    // two, so `--width` narrowed diagnostics while `--explain` bodies stayed at
    // 100 columns — two surfaces describing one configuration, disagreeing.
    explain_error_code_with_depth(
        code_str,
        use_color,
        opts.color_depth,
        opts.max_width,
        opts.tab_width,
    )
}

/// Whether an `--explain` body line is a code/example line.
///
/// The `CODE_TABLE` text format is "two spaces or one tab of indentation means
/// code", and that contract was only half enforced: a tab-indented example fell
/// through to the prose path, where it was word-wrapped and syntax-highlighted
/// as ordinary text (visually breaking the example).  Both spellings are
/// accepted here so a future entry cannot be mis-rendered by an invisible
/// indentation choice.
#[inline]
fn explain_line_is_code(line: &str) -> bool {
    (line.starts_with("  ") || line.starts_with('\t')) && !line.trim().is_empty()
}

/// Like [`explain_error_code`] but with an explicit colour depth, wrap width
/// and tab width, so the caller can share a probed [`term::ColorDepth`] and a
/// configured `--width`/`--tab-width` across multiple calls.
///
/// All three parameters are forwarded by [`explain_error_code`] as well: this
/// variant used to hardcode [`DEFAULT_MAX_WIDTH`] for wrapping, so `--width`
/// narrowed the diagnostic surface but not `--explain` output; the forwarded
/// parameters close that disagreement.
pub fn explain_error_code_with_depth(
    code_str: &str,
    use_color: bool,
    color_depth: term::ColorDepth,
    max_width: usize,
    tab_width: usize,
) -> String {
    use std::fmt::Write;

    // Use `try_new` to validate the code exists in the lookup table.
    let code = match ErrCode::try_new(code_str) {
        Ok(code) => code,
        Err(_) => {
            // `code_str` is untrusted (CLI input) and is echoed into terminal
            // output, so sanitize the copy shown while using the raw string
            // for suggestion matching.
            let error_label = DiagnosticLevel::Error.label();
            let unknown_msg = crate::diagnostics::i18n::fluent::unknown_error_code(
                sanitize_for_terminal(code_str),
            );
            return if use_color {
                format!(
                    "\x1b[1;31m{error_label}\x1b[0m {unknown_msg}\n\
                     {}",
                    suggest_code(code_str, true),
                )
            } else {
                // No `[E000]` placeholder: `E000` is asserted absent from the
                // registry, so printing it would invite users to look up (or
                // report) a code that cannot exist.
                format!(
                    "{error_label} {unknown_msg}\n{}",
                    suggest_code(code_str, false),
                )
            };
        }
    };
    // The explanation body dominates the output; pre-size from it so the
    // buffer is not regrown repeatedly while it is written.  Colourised code
    // lines add a background + reset pair (and sometimes highlight escapes) per
    // line — ~16 bytes — which the old `+ 128` did not cover, so a code-heavy
    // body regrew several times on the `--explain` hot path.
    let explain = code.explain();
    // Exact pre-size, at the cost of one counting pass — **kept deliberately**.
    // The alternative is `with_capacity(len + 256)` plus amortized growth.  The
    // trade was priced, not assumed: the pass is one allocation-free line scan
    // over a body already in cache, while the alternative pays a realloc **and
    // a copy of the whole multi-KB buffer** whenever a code-heavy body (the
    // `--explain` display case) exceeds the flat slack by even one escape
    // sequence.  One pass beats one copy.  That the count duplicates work
    // `push_explain_body` could fold in is true, but that fold would put the
    // sizing *inside* the writer, which cannot reserve ahead of its first
    // line.
    let extra = if use_color {
        explain
            .lines()
            .filter(|l| explain_line_is_code(l))
            .count()
            .saturating_mul(16)
    } else {
        0
    };
    let mut out = String::with_capacity(explain.len() + 128 + extra);

    // ── Header ──
    if use_color {
        swriteln!(
            out,
            "\x1b[1m{code}:\x1b[0m \x1b[36m{title}\x1b[0m",
            code = code.code(),
            title = code.title(),
        );
    } else {
        swriteln!(out, "{}: {}", code.code(), code.title());
    }
    swriteln!(out);

    // ── Body: parse explanation text, highlight code blocks, wrap text ──
    push_explain_body(
        &mut out,
        explain,
        use_color,
        color_depth,
        max_width,
        tab_width,
    );

    let explicit = crate::diagnostics::error_code::related_codes_of(code.code());
    let related: &[&'static str] = if explicit.is_empty() {
        crate::diagnostics::error_code::category_siblings_of(code.code(), 4)
    } else {
        explicit
    };
    if !related.is_empty() {
        swriteln!(out);
        // Localized: every neighbouring string on this surface is Fluent-resolved,
        // and a mixed-language explain body is exactly what the comptime
        // traceback fix removed.
        let see_also = crate::diagnostics::i18n::fluent::see_also();
        if use_color {
            out.push_str("\x1b[1m");
            out.push_str(&see_also);
            out.push_str("\x1b[0m\n");
        } else {
            out.push_str(&see_also);
            out.push('\n');
        }
        for rel in related {
            out.push_str("  ");
            out.push_str(rel);
            if let Some(entry) = crate::diagnostics::error_code::lookup(rel) {
                out.push_str("  ");
                out.push_str(entry.title);
            }
            out.push('\n');
        }
    }

    out
}

/// Render an explanation body into `out`, forming the second half of
/// [`explain_error_code_with_depth`]'s output.
///
/// Extracted so the line-classification rules (which code line opens or closes
/// a block, when a blank line separates) can be exercised against synthetic
/// bodies.  Before this, the only reachable inputs were the registered
/// `CODE_TABLE` bodies, and every one of them happens to place a blank line
/// immediately before each code block — which is exactly how the separator bug
/// fixed alongside this extraction survived.
///
/// Separator contract: a code block is preceded by exactly one blank line,
/// *unless* the preceding output line already is blank (the source's own blank
/// then does the separating).  This requires tracking the previous output line,
/// and clearing the flag whenever a line is emitted — a sticky `prev_blank`
/// silently disabled the separator for every block after the first one.
fn push_explain_body(
    out: &mut String,
    explain: &str,
    use_color: bool,
    color_depth: term::ColorDepth,
    max_width: usize,
    tab_width: usize,
) {
    use std::fmt::Write;

    // `--width` / `--tab-width` are caller-supplied and reach a public entry
    // point directly.  `DiagOpts::detect` clamps its own copy, but a zero width
    // would make `wrap_optimal` emit one word per line and a zero tab width
    // would make the display-column math divide by zero; clamp at the sole
    // consumer so the guarantee does not depend on every caller remembering.
    let max_width = max_width.max(1);
    let tab_width = tab_width.max(1);

    let mut in_code_block = false;
    // Dark background for code blocks: 256-colour dark gray, or reverse on
    // 16-colour terminals.
    let code_bg = if color_depth.supports_256() {
        "\x1b[48;5;236m"
    } else {
        "\x1b[7m"
    };
    const RESET: &str = "\x1b[0m";

    // Starts `true`: the caller writes a blank line after the header before
    // handing over the body, so a body that *opens* with a code block already
    // has its separating blank.  Starting `false` doubled that blank.
    let mut prev_blank = true;
    for line in explain.lines() {
        let trimmed = line.trim();
        if explain_line_is_code(line) {
            // Code/example line — apply syntax highlighting + background
            if !in_code_block {
                if !prev_blank {
                    swriteln!(out);
                }
                in_code_block = true;
            }
            let highlighted = crate::diagnostics::glyph::highlight_code(line, use_color);
            if use_color {
                swriteln!(out, "  {code_bg}{highlighted}{RESET}");
            } else {
                swriteln!(out, "  {highlighted}");
            }
            // A code line was emitted, so the previous output line is not blank.
            prev_blank = false;
        } else if trimmed.is_empty() {
            in_code_block = false;
            swriteln!(out);
            prev_blank = true;
        } else {
            if in_code_block {
                in_code_block = false;
                swriteln!(out);
            }
            // Regular text — wrap at the shared diagnostic width so the
            // explain text and rendered diagnostics agree on one width.
            for seg in crate::diagnostics::wrap::wrap_optimal(line, max_width, tab_width) {
                out.push_str(seg.as_ref());
                out.push('\n');
            }
            // Prose was emitted (at least one non-blank line).
            prev_blank = false;
        }
    }
}
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Damerau-Levenshtein (OSA) edit distance between `a` and `b`.
///
/// Three-row rolling DP: O(min(|a|, |b|)) auxiliary space, O(|a|·|b|) time.
/// Counts an adjacent transposition (`lne`↔`len`, `E003`↔`E030`) as a single
/// edit, where plain Levenshtein scores 2 — the radius-1 `suggest_code` scan
/// otherwise silently drops the most common keyboard typo.  Error codes are
/// four characters, so this is effectively constant-time.
///
/// Every buffer (the two char slices and the three rows) lives inline on the
/// stack for the realistic sizes — codes are ≤ 5 chars — so a call allocates
/// nothing; a longer needle spills to the heap via `SmallVec` rather than
/// being rejected, keeping the function total.
fn osa_distance(a: &str, b: &str) -> usize {
    const INLINE: usize = 8;
    let mut a: SmallVec<[char; INLINE]> = a.chars().collect();
    let mut b: SmallVec<[char; INLINE]> = b.chars().collect();
    // Keep the shorter string on the `a` side so the DP rows stay small.
    if a.len() > b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    let mut prev_prev: SmallVec<[usize; INLINE]> = smallvec![0; n + 1];
    let mut prev: SmallVec<[usize; INLINE]> = (0..=n).collect();
    let mut curr: SmallVec<[usize; INLINE]> = smallvec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[j - 1] == b[i - 1] { 0 } else { 1 };
            let mut d = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            if i >= 2 && j >= 2 && a[j - 1] == b[i - 2] && a[j - 2] == b[i - 1] {
                d = d.min(prev_prev[j - 2] + 1);
            }
            curr[j] = d;
        }
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// OSA distance restricted to the ≤ 1 question: `Some(d)` with `d ∈ {0, 1}`
/// when the Damerau-Levenshtein (OSA) distance of `a` and `b` is at most 1,
/// `None` when it is ≥ 2.
///
/// The full DP answers a Boolean question with `O(|a|·|b|)` work; for
/// `max_dist == 1` — every realistic code typo, since `suggest_code`'s radius
/// is `max(1, needle_len / 3)` and codes are ≤ 5 chars — the answer is a
/// 4-case check: equality (0), same length with exactly one substitution
/// (1), same length with one adjacent transposition (1), or length differing
/// by one with the shorter a subsequence-with-one-hole of the longer (1).
/// Comparison is over `char`s, matching [`osa_distance`]'s semantics for
/// non-ASCII needles.
///
/// `None` from this function does *not* pin the distance to ≥ 2 for needles
/// longer than 8 chars by itself — callers with `max_dist > 1` must fall
/// back to [`osa_distance`] (which `suggest_code` does).
fn osa_dist_le_1(a: &str, b: &str) -> Option<usize> {
    const INLINE: usize = 8;
    let a: SmallVec<[char; INLINE]> = a.chars().collect();
    osa_dist_le_1_prepared(&a, b)
}

/// [`osa_dist_le_1`] with the needle's chars already collected.
///
/// The 452-entry scan ran this once per table code, and the wrapper re-collected (and
/// re-allocated) the same needle 452 times per `--explain` typo.  This is the same body —
/// the split only moves the needle's collection to the caller.
fn osa_dist_le_1_prepared(a: &[char], b: &str) -> Option<usize> {
    // Every table code is ≤ 5 chars, so an 8-char inline buffer covers the other side
    // without a heap allocation.
    const INLINE: usize = 8;
    let b: SmallVec<[char; INLINE]> = b.chars().collect();
    if a == b.as_slice() {
        return Some(0);
    }
    let (s, l) = if a.len() <= b.len() {
        (a, b.as_slice())
    } else {
        (b.as_slice(), a)
    };
    match l.len() - s.len() {
        0 => {
            // Same length: either one substitution or one adjacent
            // transposition of the two differing positions.
            let mut d0 = 0usize;
            let mut d1 = 0usize;
            let mut n = 0usize;
            for i in 0..s.len() {
                if s[i] != l[i] {
                    if n >= 2 {
                        return None;
                    }
                    if n == 0 {
                        d0 = i;
                    } else {
                        d1 = i;
                    }
                    n += 1;
                }
            }
            match n {
                1 => Some(1), // substitution
                2 if d1 == d0 + 1 && s[d0] == l[d1] && s[d1] == l[d0] => {
                    Some(1) // transposition
                }
                _ => None,
            }
        }
        1 => {
            // One insertion/deletion: the shared prefix, then the tail of
            // the longer string must equal the tail of the shorter.
            let mut i = 0;
            while i < s.len() && s[i] == l[i] {
                i += 1;
            }
            if l[i + 1..] == s[i..] { Some(1) } else { None }
        }
        _ => None,
    }
}

/// Summarise the known error codes as contiguous ranges, e.g.
/// `E001–E004, E007, W113`.  Derived from `CODE_TABLE` so the fallback help
/// text cannot drift out of sync as codes are added.
///
/// `CODE_TABLE` is a compile-time constant, so the rendered summary is
/// computed once and cached; recomputing it copies, sorts, and formats the
/// whole table on every unknown-code explanation otherwise.
fn known_code_ranges() -> &'static str {
    // Generated, not built: this used to allocate a `Vec<&str>` of every registry
    // code and sort it at first use, for text that is fully determined at build
    // time.  `KNOWN_CODE_RANGES` is emitted by `tools/codegen` and pinned against
    // the runtime formatter below (`known_code_ranges_match_the_runtime_formatter`),
    // so the two implementations cannot drift silently.
    crate::diagnostics::error_code::KNOWN_CODE_RANGES
}

/// Summarise code strings as contiguous numeric ranges (see
/// [`known_code_ranges`]).  Split out so the ordering can be tested with
/// synthetic codes the current table does not contain.
#[cfg_attr(not(test), allow(dead_code))]
fn format_code_ranges(codes: &[&str]) -> String {
    // Test-only by design: production reads the generated `KNOWN_CODE_RANGES`, and this
    // is the runtime oracle that `known_code_ranges_match_the_runtime_formatter` pins it
    // against (so the generator cannot encode a different range table than the runtime
    // formatter produces).  Marked like its siblings (`build_line_sparse`,
    // `has_c1_sequence`): unmarked it is either a `dead_code` warning in release builds or
    // an accidental-deletion hazard for the oracle.
    let mut keyed: Vec<(u8, u64, &str)> = codes
        .iter()
        .map(|&c| {
            (
                c.as_bytes().first().copied().unwrap_or(b'?'),
                c.get(1..)
                    .and_then(|s| s.parse::<u32>().ok())
                    .map(u64::from)
                    .unwrap_or(u64::MAX),
                c,
            )
        })
        .collect();
    keyed.sort_unstable();
    let codes: Vec<&str> = keyed.into_iter().map(|(_, _, c)| c).collect();
    let numeric = |c: &str| {
        c.get(1..)
            .and_then(|s| s.parse::<u32>().ok())
            .map(u64::from)
    };
    // Build the result in place.  The old shape pushed a `String` per code (or
    // per range) into a `Vec` and joined it — one allocation per element plus
    // the join copy, for text that is *already* borrowed.
    //
    // `itoa` is deliberately not used here even though the parts look numeric:
    // the emitted text is the registry's spelling (`E030`, zero-padded to three
    // digits), and `itoa::Buffer::format` renders `30`.  Re-deriving the digits
    // from the parsed value would change the output for every padded code, and
    // pushing the original `&str` is strictly cheaper than re-formatting a number
    // we already parsed.  (The real `itoa` win in this subsystem is
    // `renderers::fmt::push_uint`, which formats actual numbers.)
    let mut out = String::with_capacity(codes.len().saturating_mul(9));
    let mut i = 0;
    while i < codes.len() {
        let start = codes[i];
        let prefix = start.get(..1).unwrap_or("?");
        let mut end = start;
        while i + 1 < codes.len() {
            let next = codes[i + 1];
            if next.get(..1).unwrap_or("?") != prefix {
                break;
            }
            match (numeric(end), numeric(next)) {
                (Some(a), Some(b)) if a.checked_add(1) == Some(b) => {
                    end = next;
                    i += 1;
                }
                _ => break,
            }
        }
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(start);
        if start != end {
            out.push('–');
            out.push_str(end);
        }
        i += 1;
    }
    out
}

/// Insert `(dist, prefix, code)` into a fixed-size top-3 by
/// `(dist asc, prefix desc, code asc)` — the ranking `suggest_code` pins.
///
/// Tie-break at the boundary: when `top` is already full, an entry that ties
/// all three incumbents is **dropped** ("first encountered wins at the
/// boundary") — it is not re-sorted into the full `(dist asc, prefix desc,
/// code asc)` order by evicting the worst incumbent.  Only an entry
/// *strictly better* than some incumbent takes its slot (`position` finds the
/// first strictly-better position; `pos == 3` means no incumbent is worse, so
/// the entry is discarded).
#[inline]
fn insert_top3<'a>(
    top: &mut ArrayVec<(usize, usize, &'a str), 3>,
    dist: usize,
    prefix: usize,
    code: &'a str,
) {
    let entry = (dist, prefix, code);
    let pos = top
        .iter()
        .position(|t| {
            entry.0 < t.0
                || (entry.0 == t.0 && (entry.1 > t.1 || (entry.1 == t.1 && entry.2 < t.2)))
        })
        .unwrap_or(top.len());
    if pos < 3 {
        if top.is_full() {
            top.pop();
        }
        top.insert(pos, entry);
    }
}

/// Whether `a` (4 digit bytes) with exactly one deletion equals `b`
/// (3 digit bytes) — i.e. `b` is a subsequence of `a` obtained by deleting
/// one character.  One forward mismatch scan + one tail compare: no DP.
fn digits_match_with_one_deletion(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), 4);
    debug_assert_eq!(b.len(), 3);
    let mut i = 0;
    while i < 3 && a[i] == b[i] {
        i += 1;
    }
    // Deleting `a[i]` splices the tail over the mismatch; when the loop
    // exhausts (first three digits equal), the deletion is at the end.
    a[i + 1..] == b[i..]
}

/// Whether `nb` has the registry code shape (`[EW]` + 3–4 digits).
///
/// The three callers (`suggest_code`, `suggest_top3_scan`, `suggest_top3_direct`)
/// route between the direct-address enumeration and the table scan on this
/// decision; three hand-copied predicates were one registry change away from
/// disagreeing, and the differential test compares *outputs*, so a drift that
/// happened to agree on the tested inputs would survive.
fn is_shaped_needle(nb: &[u8]) -> bool {
    matches!(nb.len(), 4 | 5)
        && matches!(nb.first(), Some(b'E' | b'W'))
        && nb[1..].iter().all(u8::is_ascii_digit)
}

/// Generate a "did you mean?" suggestion for an unknown error code.
///
/// Uses OSA (Damerau-Levenshtein) edit distance against every entry in `CODE_TABLE`
/// and returns up to three closest candidates within a small radius
/// (scaled by input length).  Comparison is case-insensitive so a
/// lowercase typo like `e030` still matches `E030`.
///
/// Ranking, in order of priority:
///   1. smaller edit distance
///   2. longer common prefix with the input (so `E03` → `E030`, not `E003`)
///   3. lexicographically smaller code (stable, deterministic output)
fn suggest_code(input: &str, use_color: bool) -> CompactString {
    // Normalise case so `e030` matches the canonical `E030`.
    let mut needle = CompactString::from(input);
    needle.as_mut_str().make_ascii_uppercase();
    let needle_len = needle.chars().count();
    // Longer inputs tolerate a slightly wider search radius; short ones
    // stay tight so we don't suggest unrelated codes.
    let max_dist = std::cmp::max(1, needle_len / 3);
    // Top-3 by (dist asc, prefix desc, code asc) via fixed-size insertion
    // sort — avoids the `Vec` + `O(n log n)` sort for a 3-element result.
    let nb = needle.as_bytes();
    let bucketed = is_shaped_needle(nb);
    let top = if bucketed {
        suggest_top3_direct(&needle)
    } else {
        suggest_top3_scan(&needle, max_dist, needle_len)
    };

    let candidates: ArrayVec<&str, 3> = top.iter().map(|(_, _, code)| *code).collect();

    if candidates.is_empty() {
        let help_label = DiagnosticLevel::Help.label();
        if use_color {
            format_compact!(
                "\x1b[1;34m{help_label}\x1b[0m: {valid_codes} — \n       {hint}",
                valid_codes =
                    crate::diagnostics::i18n::fluent::help_valid_codes(known_code_ranges()),
                hint = crate::diagnostics::i18n::fluent::run_explain_hint_example(),
            )
        } else {
            format_compact!(
                "{help_label}: {valid_codes} — \n       {hint}",
                valid_codes =
                    crate::diagnostics::i18n::fluent::help_valid_codes(known_code_ranges()),
                hint = crate::diagnostics::i18n::fluent::run_explain_hint_example(),
            )
        }
    } else {
        let help_label = DiagnosticLevel::Help.label();
        let did_you_mean = crate::diagnostics::i18n::fluent::did_you_mean({
            // Each candidate is quoted individually and the *conjunction* is
            // localized; `join("` or `")` baked English "or" in.
            let mut it = candidates.iter();
            let first = Cow::Owned(format!("`{}`", it.next().expect("non-empty")));
            crate::diagnostics::i18n::join_with_or(first, it.map(|c| Cow::Owned(format!("`{c}`"))))
        });
        let hint = crate::diagnostics::i18n::fluent::run_explain_hint_detail();
        if use_color {
            format_compact!("\x1b[1;34m{help_label}\x1b[0m: {did_you_mean}\n       {hint}")
        } else {
            format_compact!("{help_label}: {did_you_mean}\n       {hint}")
        }
    }
}

/// The general candidate scan: OSA distance against the (prefiltered) table,
/// plus the first-char bucketing shortcut for `[EW]\d{3,4}` needles.  In
/// production this only serves needles that do NOT match the registry shape
/// (short/long/non-EW/non-digit input); the shape-needle bucketed branch is
/// retained as the reference implementation that
/// `suggest_code_direct_matches_scan` differentially tests against
/// [`suggest_top3_direct`].
fn suggest_top3_scan(
    needle: &str,
    max_dist: usize,
    // #14: the caller already counts the needle's chars; recomputing it per
    // scan was a second full pass on every `--explain` typo.
    needle_len: usize,
) -> ArrayVec<(usize, usize, &'static str), 3> {
    // Top-3 by (dist asc, prefix desc, code asc) via fixed-size insertion
    // sort — avoids the `Vec` + `O(n log n)` sort for a 3-element result.
    let mut top: ArrayVec<(usize, usize, &str), 3> = ArrayVec::new();
    let nb = needle.as_bytes();

    // ── First-char bucketing ─────────────────────────────────────
    // The registry is validated to `^[EW]\d{3}$`, so a 4-char needle that
    // itself matches that shape can only match entries in one of two
    // buckets:
    //
    //   * same-prefix (`E###` vs `E###`): up to `max_dist` edits anywhere;
    //     run `osa_distance` on the whole code.
    //
    //   * cross-prefix (`E###` vs `W###`): a single prefix substitution
    //     consumes the *entire* `max_dist == 1` budget, so the 3-digit
    //     remainder must be an **exact** match — no OSA pass needed.
    //
    // No other prefixes exist, so the cross-prefix remainder check plus
    // the same-prefix OSA pass covers every match, halving the OSA calls.
    // `needle_len == 4` forces `max_dist == 1` (4/3 == 1); so does
    // `needle_len == 5` — but a 5-char needle can only match a 4-char code
    // at distance 1 by a *pure insertion* (the length delta already spends
    // the whole budget), so its same-prefix branch is an exact
    // delete-one-digit test and its cross-prefix branch can never match
    // (substitution + insertion = distance 2 > max_dist).  Extending the
    // bucket to 5 keeps an `E0308`-shaped typo from walking the whole table
    // through the full rolling DP.  Non-shape needles (shorter, longer,
    // non-EW prefix, non-digit body) fall through to the unrestricted scan
    // below.
    let bucketed = is_shaped_needle(nb);
    if bucketed {
        let same_prefix = nb[0];
        let rest = &needle[1..];
        // Collected once for the whole 452-entry scan (see `osa_dist_le_1_prepared`).
        let needle_chars: SmallVec<[char; 8]> = needle.chars().collect();
        for e in crate::diagnostics::error_code::CODE_TABLE.iter() {
            let eb = e.code.as_bytes();
            if eb[0] != same_prefix {
                // Cross-prefix: for a 4-char needle a prefix substitution
                // consumes the whole `max_dist == 1` budget, so the digits
                // must match exactly; for a 5-char needle substitution +
                // insertion is distance 2 — never within budget.
                if needle_len == 4 && eb[0] == (if same_prefix == b'E' { b'W' } else { b'E' }) {
                    if &e.code[1..] != rest {
                        continue;
                    }
                    let prefix = common_prefix_len(&needle, e.code);
                    insert_top3(&mut top, 1, prefix, e.code);
                }
                continue;
            }
            // The table is validated to `^[EW]\d{3}$`, so every code is
            // exactly 4 bytes — `debug_assert_eq!` pins it.
            debug_assert_eq!(eb.len(), 4, "CODE_TABLE codes must be 4 bytes");
            let dist = if needle_len == 4 {
                // `max_dist == 1` here, so ask the ≤ 1 question directly: a
                // ~4-compare check instead of the full rolling DP.
                match osa_dist_le_1_prepared(&needle_chars, e.code) {
                    Some(d) => d,
                    None => continue, // distance ≥ 2 > max_dist
                }
            } else {
                // `max_dist == 1` and the needle is one digit longer, so the
                // only surviving edit is a single insertion into the code:
                // the code's digits must be the needle's digits minus one.
                if !digits_match_with_one_deletion(rest.as_bytes(), &e.code.as_bytes()[1..]) {
                    continue;
                }
                1
            };
            if dist > max_dist {
                continue;
            }
            let prefix = common_prefix_len(&needle, e.code);
            insert_top3(&mut top, dist, prefix, e.code);
        }
    } else {
        // Collected once for the whole scan (see `osa_dist_le_1_prepared`).
        let needle_chars: SmallVec<[char; 8]> = needle.chars().collect();
        for e in crate::diagnostics::error_code::CODE_TABLE
            .iter()
            // Length prefilter: each edit changes length by at most one, so
            // |Δlen| is a lower bound on the distance — an entry whose code
            // length differs by more than the radius can never match, and its
            // DP is skipped.  The needle uses a char count (not byte length) so
            // the bound stays exact for non-ASCII typos; every table code is
            // validated to `^[EW]\d{3}$`, so `e.code.len()` is already its char
            // count (constant 4) and skips the per-code char walk.
            .filter(|e| needle_len.abs_diff(e.code.len()) <= max_dist)
        {
            // `max_dist == 1` covers every needle up to 5 chars — the common
            // typo shape — so the ≤ 1 oracle answers it without the DP;
            // wider radii fall back to the full DP, which remains the
            // reference the differential tests pin.
            let dist = if max_dist == 1 {
                match osa_dist_le_1_prepared(&needle_chars, e.code) {
                    Some(d) => d,
                    None => continue,
                }
            } else {
                osa_distance(&needle, e.code)
            };
            if dist > max_dist {
                continue;
            }
            let prefix = common_prefix_len(&needle, e.code);
            insert_top3(&mut top, dist, prefix, e.code);
        }
    }

    top
}

/// Direct-address neighbourhood enumeration for `[EW]\d{3,4}`-shaped needles,
/// replacing the 452-entry table walk of [`suggest_top3_scan`].
///
/// Every registry code at OSA distance ≤ 1 from a shaped needle is reachable
/// by exactly one of:
///
/// * identity (distance 0);
/// * one substitution — the prefix (`E↔W`, which forces an exact digit
///   remainder) or any single digit lane (10 digits each): 1 + 3·10 slots;
/// * one transposition of an adjacent digit pair: 2 slots (three digit lanes have
///   exactly two adjacent pairs — `(1,2)` and `(2,3)`, which is what the `p in 0..2`
///   loop enumerates);
/// * for a 5-digit needle, one insertion *into the code* — the length delta
///   already spends the whole radius, so the code's digits must be the
///   needle's digits minus one (the 4 deletion candidates).
///
/// That is ≤ 41 candidate slots per query (deduped), each resolved `O(1)`
/// through [`lookup`]'s `code_slot` + `CODE_INDEX` read, versus a full
/// 452-entry table walk whose per-entry test is itself several compares.
/// [`insert_top3`]'s comparator is a strict total order on distinct
/// `(dist, prefix, code)` entries, so enumeration order cannot change the
/// top-3 — pinned differentially against [`suggest_top3_scan`].
fn suggest_top3_direct(needle: &str) -> ArrayVec<(usize, usize, &'static str), 3> {
    // Only `[EW]\d{3}` / `[EW]\d{4}` needles have an enumerable
    // direct-address neighbourhood.  Anything else (typos like "E03",
    // longer strings, noise) has no enumeration shape — fall back to the
    // table scan so the entry point is total (the differential test feeds
    // exactly such inputs).  Production callers route shaped needles here
    // and unshaped ones straight to the scan, so this branch is cold.
    let nb = needle.as_bytes();
    let shaped = is_shaped_needle(nb);
    if !shaped {
        // Character count, matching `suggest_code`'s own radius derivation.
        // Using the *byte* length here made an unshaped multi-byte needle
        // (4 chars, 8 bytes) get twice the radius the scan reference uses, so
        // the differential equivalence held only for ASCII inputs.
        let max_dist = std::cmp::max(1, needle.chars().count() / 3);
        return suggest_top3_scan(needle, max_dist, needle.chars().count());
    }
    /// Resolve one candidate slot (if a known code) into the top-3.
    fn consider(
        top: &mut ArrayVec<(usize, usize, &'static str), 3>,
        needle: &str,
        cand: &[u8; 4],
        dist: usize,
    ) {
        // SAFETY: `cand` is assembled only from a `E`/`W` prefix and ASCII
        // digits, so it is always valid UTF-8.  The previous `simdutf8`
        // validation ran once per enumerated candidate (up to 41 per query) on
        // the path whose whole point is being op-minimal; the debug assert keeps
        // the argument checked.
        debug_assert!(cand.iter().all(u8::is_ascii));
        let code = unsafe { std::str::from_utf8_unchecked(cand) };
        if let Some(e) = crate::diagnostics::error_code::lookup(code) {
            // Dedup by code: the 5-char deletion branch can enumerate the
            // same 4-char code at two `skip` positions when the needle's
            // digits repeat, and `insert_top3` does not dedup.  The same
            // code from the same needle always carries the same claimed
            // distance, so skipping a duplicate never changes the ranking.
            if top.iter().any(|t| t.2 == e.code) {
                return;
            }
            insert_top3(top, dist, common_prefix_len(needle, e.code), e.code);
        }
    }

    let nb = needle.as_bytes();
    let prefix = nb[0];
    let digits = &nb[1..];
    let mut top: ArrayVec<(usize, usize, &'static str), 3> = ArrayVec::new();

    if digits.len() == 3 {
        // Identity (distance 0) — the table lookup simply confirms presence.
        let mut cand = [prefix, digits[0], digits[1], digits[2]];
        consider(&mut top, needle, &cand, 0);
        // Prefix substitution (distance 1, whole budget).
        cand[0] = if prefix == b'E' { b'W' } else { b'E' };
        consider(&mut top, needle, &cand, 1);
        // Digit substitutions, one lane at a time (distance 1 each).  The
        // candidate is rebuilt from the needle every iteration so lanes set
        // by previous substitutions never leak into the next one.
        for p in 0..3 {
            for d in 0..10u32 {
                let dv = b'0' + d as u8;
                if dv == digits[p] {
                    continue; // identity, already considered
                }
                cand = [prefix, digits[0], digits[1], digits[2]];
                cand[p + 1] = dv;
                consider(&mut top, needle, &cand, 1);
            }
        }
        // Adjacent-digit transpositions (OSA distance 1 each).  Transposing
        // two *equal* digits reproduces the needle verbatim — a distance-0
        // duplicate, not a distance-1 neighbour — so skip that candidate;
        // `insert_top3` would otherwise record it at the claimed distance 1.
        for p in 0..2 {
            cand = [prefix, digits[0], digits[1], digits[2]];
            cand.swap(p + 1, p + 2);
            if cand == [prefix, digits[0], digits[1], digits[2]] {
                continue;
            }
            consider(&mut top, needle, &cand, 1);
        }
    } else {
        debug_assert_eq!(digits.len(), 4, "shaped needle is [EW]\\d{{3,4}}");
        // 5-char needle: only a single insertion into a 4-char code stays
        // within the radius — the code's digits are the needle's minus one.
        for skip in 0..4 {
            let mut cand = [prefix, 0u8, 0, 0];
            let mut k = 1;
            for (i, &d) in digits.iter().enumerate() {
                if i != skip {
                    cand[k] = d;
                    k += 1;
                }
            }
            consider(&mut top, needle, &cand, 1);
        }
    }
    top
}

/// List all available error codes with their titles, formatted for terminal output.
pub fn list_error_codes(use_color: bool) -> String {
    use std::fmt::Write;
    let (bold, cyan, dim, reset) = if use_color {
        ("\x1b[1m", "\x1b[36m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    let mut out = String::new();
    swriteln!(
        out,
        "{bold}{}{reset}\n",
        crate::diagnostics::i18n::fluent::available_error_codes()
    );
    for (cat, entries) in crate::diagnostics::error_code::code_entries_by_category() {
        swriteln!(out, " {cyan}── {} ──{reset}", cat.label(),);
        for entry in entries {
            swriteln!(out, "  {bold}{}{reset}  {}", entry.code, entry.title,);
        }
    }
    swriteln!(
        out,
        "\n{dim}{}{reset}",
        crate::diagnostics::i18n::fluent::run_explain_detail()
    );
    out
}

/// Convenience function to emit diagnostics to stderr.
/// Chooses plain or colored rendering based on the `use_color` flag.
///
/// Diagnostics are humanized at the emitter's render boundary, so their
/// i18n language resolves against the current `Lang` at emit time.
/// Terminal caps are process-stable, so the three `emit_*` helpers share one
/// snapshot instead of re-probing env vars and the terminal per call.
fn detected_opts() -> DiagOpts {
    /// Frozen **on purpose**: one process has one terminal snapshot, and re-probing per call would
    /// let a terminal resize change `max_width`/`color_depth` midway through a diagnostic dump, so
    /// two surfaces could describe one configuration differently — the exact disagreement
    /// `explain_error_code_with_depth`'s `&DiagOpts` forwarding exists to close.  A caller that
    /// configures width *after* the first `emit_*`/`--explain` call is therefore ignored here by
    /// design; such a caller must pass its own `DiagOpts` (the `*_with_opts` entry points),
    /// not mutate the environment and hope.
    static DETECTED: std::sync::OnceLock<DiagOpts> = std::sync::OnceLock::new();
    DETECTED.get_or_init(DiagOpts::detect).clone()
}

/// Convenience wrapper that deliberately **ignores** write errors: a closed
/// stdout pipe during a diagnostic dump is not actionable, so this path only
/// best-effort reports.  Callers that need the failure must use
/// [`emit_diagnostics_iter`], the entry point that propagates it as
/// `io::Result`; the split is a recorded policy decision, not an oversight.
pub fn emit_diagnostics(diags: &mut [Diagnostic<'_>], use_color: bool) {
    let mut opts = detected_opts();
    opts.use_color = use_color;
    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);

    swallow_it::swallow!(emitter.emit_all(diags));
}

/// Convenience function to emit a DiagCtxt without cloning its diagnostics.
///
/// Takes `&mut` because the emitter humanizes each diagnostic at its
/// render boundary (see [`Diagnostic`]'s lazy-humanization contract).
///
/// Deliberately **ignores** write errors (e.g. a closed stdout pipe), matching
/// [`emit_diagnostics`]: this is the batch convenience path, not a reporting
/// API.  [`emit_diagnostics_iter`] is the entry point that propagates them as
/// `io::Result`; the split is a recorded policy decision, not an oversight.
pub fn emit_collector(collector: &mut DiagCtxt<'_>, use_color: bool) {
    let mut opts = detected_opts();
    opts.use_color = use_color;
    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);
    // Humanize first, then hand out read-only references: the emitter's batch
    // paths take `&Diagnostic`, so nothing downstream can mutate a diagnostic
    // (and therefore nothing can desynchronise the collector's tallies).
    // Humanize only what the filter keeps, and emit only the unreported tail — the
    // same partition discipline `DiagCtxt::emit` uses (emit `diags[reported_len..]`,
    // advance the split only after a successful write, which is what makes `emit`
    // retry-safe).  Emitting `collector.iter()` re-rendered the *reported* prefix
    // whenever a caller mixed the two surfaces, called this twice, or ran it per
    // module over a shared context, and it paid full Fluent humanization for every
    // diagnostic the filter then dropped.
    //
    // Filtering first is sound because the emitter drops filtered diagnostics
    // *before* its `is_humanized` assertion (see `emit_ordered`), so a dropped
    // diagnostic is never handed to a renderer, humanized or not.
    //
    // The source-attachment audit belongs on **every** path that pushes diagnostics into emission,
    // not only on `DiagCtxt::emit`: without this line a sourceless diagnostic emitted here bypassed
    // the "reached emission without a source" invariant that `collector.rs` documents as terminal,
    // and `internal_bug_count()` silently depended on which surface the caller used.  O(1) unless
    // the unsourced suffix flag is armed.
    collector.audit_source_attachment();
    // Bound the filter outside the closure: `humanize_pending_for` now requires `F: … + Sync`, and a
    // closure capturing `&Emitter` is not `Sync` (the emitter owns `RefCell` scratch).  The
    // predicate itself is `Sync`, which is the honest requirement — the filter is the only state the
    // parallel chunks share.
    let filter = emitter.filter();
    collector.humanize_pending_for(|d| filter.matches(d));
    if emitter.emit_all_iter(collector.unreported().iter()).is_ok() {
        collector.mark_all_reported();
    }
    // The tallies can no longer drift through the emitter; this stays as the
    // repair path for an external checkpoint consumer that mutated and did not
    // recount.
    collector.recount_derived_state();
    #[cfg(debug_assertions)]
    swallow_it::swallow!(collector.summary());
}

/// Emit an iterator of diagnostics through a fresh sink.
///
/// Unlike [`emit_collector`], this avoids cloning each diagnostic into a
/// `Vec` first.  The diagnostics must already be humanized (the emitter takes
/// `&Diagnostic`); callers that own a `DiagCtxt` get that from
/// [`DiagCtxt::humanize_pending`].
pub fn emit_diagnostics_iter<'r, 'src, I>(diags: I, use_color: bool) -> io::Result<()>
where
    'src: 'r,
    I: IntoIterator<Item = &'r Diagnostic<'src>>,
    I::IntoIter: DoubleEndedIterator,
{
    let mut opts = detected_opts();
    opts.use_color = use_color;
    let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
    let mut emitter = Emitter::new(renderer, sink::stderr_buffered(), opts);
    emitter.emit_all_iter(diags)
}

/// Largest char-boundary offset `<= offset` (clamped to `source.len()`).
///
/// Spans are byte offsets and may — through arithmetic or malformed input —
/// land inside a multi-byte character.  Slicing `&source[..offset]` at such
/// an offset panics, so callers that index by byte must floor first.
pub(crate) fn floor_char_boundary(source: &str, offset: usize) -> usize {
    let mut o = std::cmp::min(offset, source.len());
    while o > 0 && !source.is_char_boundary(o) {
        o -= 1;
    }
    o
}

/// `char_col = b − cumulative_extra` for non-ASCII chars starting before `b`.
/// `sparse` is a sorted `[(start_byte, cumulative_extra)]` table; one
/// `partition_point` + one array read gives the answer in `O(log K)`.
/// `None` means the line is ASCII, so char col equals byte col (`b`).
/// Kept as the reference implementation for differential testing; the
/// production path uses the hinted variant below.
/// Build one line's sparse `(start_byte, cumulative_extra)` table.
///
/// Reachable in production through `char_col`'s fallback (a line at or past the
/// prefix gate's frontier, or a source whose tables were never built), so the
/// `#[cfg_attr(not(test), allow(dead_code))]` that used to sit here was stale — it
/// suppressed exactly the dead-code signal that would report the fallback going
/// unused.
///
/// Only the high-byte-gate fallback uses this now: the prebuilt tables come from
/// `scan_line_starts_and_ascii`.  It is the same walk the per-line cache-miss path
/// used — `next_non_ascii_byte` jumps the ASCII runs, then each non-ASCII *char*
/// contributes its `len_utf8 - 1` to the running extra, so
/// `char_col = b − extra(last char starting before b)` holds for char boundaries.
fn build_line_sparse(line: &str) -> SmallVec<[(u32, u32); 8]> {
    let mut sparse: SmallVec<[(u32, u32); 8]> = SmallVec::new();
    let mut extra = 0u32;
    let bytes = line.as_bytes();
    let mut i = 0usize;
    loop {
        let off = ascii::next_non_ascii_byte(&bytes[i..]);
        if off == usize::MAX {
            break;
        }
        let start = i + off;
        debug_assert!(line.is_char_boundary(start));
        let ch = line[start..].chars().next().unwrap();
        extra += (ch.len_utf8() - 1) as u32;
        sparse.push((start as u32, extra));
        i = start + ch.len_utf8();
    }
    sparse
}

fn char_col_from_sparse(sparse: Option<&[(u32, u32)]>, b: usize) -> usize {
    match sparse {
        Some(s) => {
            let k = s.partition_point(|&(p, _)| (p as usize) < b);
            let extra = if k > 0 { s[k - 1].1 as usize } else { 0 };
            b - extra
        }
        None => b,
    }
}

/// Convert a byte-offset span into a 0-based (line, col) pair.
/// Single source of truth — previously duplicated verbatim in
/// `glyph::byte_to_linecol` and `label::byte_to_linecol`.
pub(crate) fn byte_to_linecol(source: &str, byte_offset: usize) -> label::SourcePos {
    let clamped = floor_char_boundary(source, byte_offset);
    let prefix = &source[..clamped];
    let line = memchr::memchr_iter(b'\n', prefix.as_bytes()).count();
    let start_of_line = memchr::memrchr(b'\n', prefix.as_bytes())
        .map(|i| i + 1)
        .unwrap_or(0);
    label::SourcePos {
        line,
        col: clamped - start_of_line,
    }
}

/// Whether `name` looks like a real on-disk file path rather than a
/// synthetic placeholder such as `<synthetic>` or an empty string.
pub(crate) fn is_real_file_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('<')
}

/// [`sanitize_for_terminal`] variant for one-line renderers that flatten
/// breaks themselves: identical SWAR gates and replacement, but `\n`, `\r`
/// and `\t` survive as layout bytes — the caller consumes them as break runs
/// (the plain variant would turn them into U+FFFD first and defeat the
/// flattening).  The short renderer's `push_flattened` collapses all three
/// to a single space, so a tab never reaches the one-line output verbatim.
pub(crate) fn sanitize_for_terminal_flatten(s: &str) -> Cow<'_, str> {
    sanitize_for_terminal_flatten_with_break(s).0
}

pub(crate) fn sanitize_for_terminal_flatten_with_break(s: &str) -> (Cow<'_, str>, Option<usize>) {
    let bytes = s.as_bytes();
    let scan = scan_terminal_control_flags_flatten(bytes);
    let (has_high, dirty, has_c1, break_at) =
        (scan.has_high, scan.dirty, scan.has_c1, scan.first_break);
    if !has_high && !dirty {
        return (Cow::Borrowed(s), break_at);
    }
    // Non-ASCII with no C0/C1/DEL *and* no bidi override/isolate: borrow.
    // Bidi formatting characters (U+202A–U+202E, U+2066–U+2069) are neither
    // controls nor visible, so the byte-level scan cannot see them — they are
    // probed separately (see [`sanitize_for_terminal`] for the spoofing
    // rationale) and must reach the owned path for removal.  Hoisted above both
    // guards, like the plain variant: the owned branch used to re-run the same
    // O(n) probe on the bidi-present path.
    //
    // `has_e2` (from the control scan) gates the probe: every bidi override is a
    // leading-`0xE2` sequence, so a string without that byte cannot contain one
    // and skips the `memchr` pass entirely — the Cyrillic (`0xD0`/`0xD1`) and CJK
    // (`0xE4..=0xE9`) cases, i.e. the dominant i18n inputs.
    let bidi = scan.has_e2 && has_bidi_override_bytes(bytes);
    if has_high && !dirty && !has_c1 && !bidi {
        return (Cow::Borrowed(s), break_at);
    }
    // Layout-only ASCII (`\n`/`\r`/`\t` are the only controls) is already
    // covered by the first branch: the scanner excludes those bytes from
    // `dirty`, so `!has_high && !dirty` is exactly the layout-only case.
    // Bidi overrides/isolates are *removed* (not U+FFFD'd), matching
    // [`sanitize_filename`] and the source-row renderer: they carry no
    // content, only rendering direction, so replacement would just inject
    // noise while `\n`/`\r`/`\t` survive for the flattening caller.
    let owned: String = if bidi {
        s.chars()
            .filter(|c| !is_bidi_override(*c))
            .map(|c| {
                if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                    '\u{FFFD}'
                } else {
                    c
                }
            })
            .collect()
    } else {
        s.chars()
            .map(|c| {
                if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                    '\u{FFFD}'
                } else {
                    c
                }
            })
            .collect()
    };
    // `first_break` was measured in the original bytes; bidi removal and
    // U+FFFD growth shift it.  The consumer wants the first break in the
    // *sanitized* string, so recompute (memchr3 is one SIMD pass, rare path).
    let break_at = memchr::memchr3(b'\n', b'\r', b'\t', owned.as_bytes());
    (Cow::Owned(owned), break_at)
}

/// Replace control characters with U+FFFD so external text cannot inject
/// terminal escapes when echoed to the terminal.
///
/// Single source of truth — previously duplicated in `glyph`
/// (`filename_display`) and `error_code`.  Applied at every boundary that
/// writes untrusted text into ANSI output (OSC 8 hyperlink text, SGR colour
/// runs), where a raw `\x1b` or `\x07` would terminate the sequence early and
/// let the remaining bytes act as terminal commands.
///
/// Two fast paths keep the common cases allocation-free:
///
/// 1. **Printable ASCII** — one SWAR scan (see `ascii::is_printable_ascii`);
///    no control character is possible, so the input is borrowed.
/// 2. **Non-ASCII with no controls** — the dominant i18n case (CJK,
///    Cyrillic, accented Latin, …).  A `chars().any(is_control)` scan
///    short-circuits on the first control; if none is present, the input
///    is borrowed.  Only strings that actually contain a control character
///    pay the allocate-and-copy path.  `is_control` covers C0
///    (`0x00..=0x1F`), C1 (`0x80..=0x9F`) and DEL (`0x7F`).
///
/// Bidi overrides/isolates (U+202A–U+202E, U+2066–U+2069) are neither
/// controls nor visible, so they escape both fast paths' scans; a gated
/// `memchr(0xE2)` precheck catches them and they are **removed** in the
/// owned path (same convention as [`sanitize_filename`]) so a path like
/// `evil.cod\u{202E}exe` cannot spoof its extension in terminal output.
pub(crate) fn sanitize_for_terminal(s: &str) -> Cow<'_, str> {
    // One fused SWAR pass answers both gates.  The old sequence
    // (`is_printable_ascii` then `has_control_byte_level`) loaded the same
    // clean bytes twice — the printable-ASCII lane already rejects every
    // byte < 0x20, so the control probe re-scanned what the first pass had
    // just cleared.
    let bytes = s.as_bytes();
    let scan = scan_terminal_control_flags(bytes);
    let (has_high, dirty, has_c1) = (scan.has_high, scan.dirty, scan.has_c1);
    if !has_high && !dirty {
        // Every byte in 0x20..=0x7E: printable ASCII, control-free.
        return Cow::Borrowed(s);
    }
    // The dominant i18n case (pure Cyrillic / CJK / accented Latin, no
    // controls): non-ASCII, but nothing to replace — borrow.  C1 is only
    // possible where high bytes exist, so the `memchr(0xC2)` probe is gated.
    // Bidi overrides/isolates (U+202A–U+202E, U+2066–U+2069) are invisible
    // and not controls, so they escape the SWAR scan — but they reorder
    // surrounding display text (`evil.cod\u{202E}exe` spoofs its extension),
    // so they are probed behind `scan.has_e2` and force the owned path.  Owned:
    // they are *removed*, matching [`sanitize_filename`] — direction-only
    // characters have no content, and U+FFFD would just inject noise.
    //
    // `has_e2`, not `has_high`: every bidi override/isolate is an `E2 80 …` /
    // `E2 81 …` sequence and `0xE2` is always a UTF-8 lead byte, so the flag has
    // no false negatives — gating on `has_high` made every CJK/Cyrillic/accented
    // string (the dominant i18n inputs this comment names) pay a full `memchr`
    // pass the scan had already proved unnecessary.
    let bidi = scan.has_e2 && has_bidi_override_bytes(bytes);
    if has_high && !dirty && !has_c1 && !bidi {
        return Cow::Borrowed(s);
    }
    if bidi {
        Cow::Owned(
            s.chars()
                .filter(|c| !is_bidi_override(*c))
                .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
                .collect(),
        )
    } else {
        Cow::Owned(
            s.chars()
                .map(|c| if c.is_control() { '\u{FFFD}' } else { c })
                .collect(),
        )
    }
}

/// One SWAR pass classifying a byte slice for the terminal sanitizers.
///
/// Returns `(has_high, dirty, has_c1)`:
///
/// * `has_high` — any byte ≥ 0x80 (non-ASCII UTF-8 byte, lead or
///   continuation);
/// * `dirty` — any byte that a sanitizer must replace: byte < 0x20 (C0) or
///   `0x7F` (DEL);
/// * `has_c1` — a `0xC2 0x80..=0x9F` C1 control sequence.
///
/// Three masks per 8-byte chunk (`hasless(0x20)`, `has_zero(0x7F)`, the
/// high-bit OR) instead of the two full scans the previous two-gate
/// structure ran over clean input.  C1 (`0xC2 0x80..=0x9F`) is *not* folded
/// in — it needs a positional two-byte check, and it is only possible when
/// `has_high` is set, so callers probe it separately via
/// [`has_c1_sequence`].
/// Result of one fused SWAR pass over a byte slice (see the three
/// `scan_terminal_control_flags*` scanners).
///
/// Named fields rather than a positional tuple: the scanners return four to five
/// booleans that callers destructure, and a bare `(bool, bool, bool, bool)` made
/// a swapped pair (`dirty`/`has_c1`) type-check.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ControlScan {
    /// Any byte ≥ 0x80.
    pub has_high: bool,
    /// Any byte the sanitizer must replace (C0 outside the caller's layout set,
    /// or DEL).
    pub dirty: bool,
    /// A C1 sequence (`0xC2 0x80..=0x9F`).
    pub has_c1: bool,
    /// Any `0xE2` byte.  Every bidi override/isolate is a three-byte sequence
    /// with `0xE2` as its lead, and in valid UTF-8 `0xE2` is always a lead byte,
    /// so this has no false negatives and lets the sanitizers gate the
    /// `has_bidi_override_bytes` probe on a flag the control scan already
    /// computed instead of re-walking the whole string with `memchr`.
    pub has_e2: bool,
    /// Byte offset of the first `\t`/`\n`/`\r` (flatten scanner only).
    pub first_break: Option<usize>,
}

/// `haszero` mask for `x & HI == 0` per lane — the borrow-free equality idiom.
#[inline]
fn eq_byte_lanes(x: u64, byte: u8, hi: u64, ones: u64) -> u64 {
    let q = x ^ (u64::from(byte) * 0x0101_0101_0101_0101);
    !q.wrapping_sub(ones) & !q & hi
}

/// **The `dirty`/`has_*` masks this returns are existentially exact, not
/// lane-exact**: a borrow-propagated lane may be flagged even though its own byte
/// is not a control character, so the mask must be used for *"is there any"* only.
/// Positional extraction needs the `| HI`-borrow-proofed lane masks (the ones the
/// newline/tab scanners build), and reaching for this mask instead is exactly the
/// bug class `line_scan_does_not_invent_newlines_after_vt` documents.
fn scan_terminal_control_flags(s: &[u8]) -> ControlScan {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let mut out = ControlScan::default();
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let high = x & HI != 0;
        out.has_high |= high;
        // `0xE2` is only possible in a chunk with a high bit, so the probe is
        // folded into the same gate the C1 probe uses.
        if high {
            out.has_e2 |= eq_byte_lanes(x, 0xE2, HI, ONES) != 0;
            out.has_c1 |= c1_in_chunk(s, i, x);
        }
        // C0: any byte < 0x20 (existentially exact).
        out.dirty |= x.wrapping_sub(SP) & !x & HI != 0;
        // DEL: any byte == 0x7F (exact — any flag implies a real lane).
        let q = x ^ DEL;
        out.dirty |= q.wrapping_sub(ONES) & !q & HI != 0;
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        out.has_high |= b >= 0x80;
        out.dirty |= b < 0x20 || b == 0x7F;
        out.has_e2 |= b == 0xE2;
        out.has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    out
}

/// Whether one 8-byte chunk contains a C1 sequence start (`0xC2` followed by
/// `0x80..=0x9F`).  The follow byte may live in the next chunk, so `s` and
/// `base` are used for the positional check rather than the chunk alone.
#[inline]
fn c1_in_chunk(s: &[u8], base: usize, x: u64) -> bool {
    const HI: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const C2: u64 = 0xC2C2_C2C2_C2C2_C2C2;
    let q = x ^ C2;
    let mut mask = q.wrapping_sub(ONES) & !q & HI;
    while mask != 0 {
        let lane = (mask.trailing_zeros() / 8) as usize;
        let k = base + lane;
        if s.get(k) == Some(&0xC2) && s.get(k + 1).is_some_and(|b| (0x80..=0x9F).contains(b)) {
            return true;
        }
        mask &= mask - 1;
    }
    false
}

/// Whether `s` contains a C1 control sequence (`0xC2` followed by
/// `0x80..=0x9F`) — the UTF-8 encoding of Unicode category `Cc` above DEL.
/// Only worth calling when the slice is known to contain high bytes.
#[cfg_attr(not(test), allow(dead_code))]
fn has_c1_sequence(s: &[u8]) -> bool {
    let mut j = 0;
    while let Some(off) = memchr::memchr(0xC2, &s[j..]) {
        let k = j + off;
        if k + 1 < s.len() && (0x80..=0x9F).contains(&s[k + 1]) {
            return true;
        }
        j = k + 1;
    }
    false
}

/// One fused SWAR pass classifying a byte slice for
/// [`sanitize_for_terminal_multiline`]: returns `(has_high, dirty, has_c1)`
/// where `has_high` is any byte ≥ 0x80, `dirty` is any byte the multiline
/// sanitizer must replace — a C0 byte other than `\t`/`\n`, or DEL — and
/// `has_c1` is a `0xC2 0x80..=0x9F` sequence.
///
/// The `\t`/`\n` exclusion is done **per lane**: the `| HI` before each
/// subtraction keeps every lane ≥ 0x80, so the borrow cannot cross a lane and
/// the `< 0x20` mask is exact per lane — the tab/newline masks are subtracted
/// from it directly, with no flagged-lane resolution and no byte re-checks.  A
/// chunk full of `\t`/`\n` therefore costs the same as a clean one.
/// (`trailing_zeros`/8 still appears, but for the *C1* probe, not for this.)
/// C1 is probed separately via [`has_c1_sequence`], only when `has_high`,
/// because a C1 control needs a leading `0xC2`.
pub(crate) fn scan_terminal_control_flags_multiline(s: &[u8]) -> ControlScan {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const TAB: u64 = 0x0909_0909_0909_0909;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;

    let mut out = ControlScan::default();
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let high = x & HI != 0;
        out.has_high |= high;
        // `0xE2` and a C1 sequence both need a high-bit lane, so both probes
        // ride the same gate: an all-ASCII chunk keeps its exact answer
        // (`false`) and pays nothing.
        if high {
            out.has_e2 |= eq_byte_lanes(x, 0xE2, HI, ONES) != 0;
            out.has_c1 |= c1_in_chunk(s, i, x);
        }
        // DEL: exact — any flag implies a genuine 0x7F lane.
        let q = x ^ DEL;
        if q.wrapping_sub(ONES) & !q & HI != 0 {
            out.dirty = true;
        }
        // C0: exact per-lane `< 0x20` mask (the `| HI` before the subtraction
        // prevents an inter-lane borrow), with `\t`/`\n` masked back out.
        // Unlike the inexact `hasless` idiom this needs no flagged-lane
        // resolution, so tab/newline-dense text stays O(chunks), not O(bytes).
        // High-byte lanes are excluded explicitly: this scanner cannot
        // early-return on `x & HI` (unlike `ascii.rs`), and the raw form also
        // flags bytes in `0x80..=0x9F`.
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI & !(x & HI);
        if lt20 != 0 {
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI;
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI;
            if lt20 & !tab & !nl != 0 {
                out.dirty = true;
            }
        }
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        out.has_high |= b >= 0x80;
        out.dirty |= (b < 0x20 && b != b'\t' && b != b'\n') || b == 0x7F;
        out.has_e2 |= b == 0xE2;
        out.has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    out
}

/// [`scan_terminal_control_flags_multiline`] variant for
/// [`sanitize_for_terminal_flatten`]: `\t`, `\n` *and* `\r` are excluded from
/// the dirty set, because those are exactly the layout bytes the flatten path
/// keeps or consumes (`push_flattened`).  Returns `(has_high, dirty,
/// has_c1, first_break)`, where `first_break` is the byte offset of the first
/// `\t`/`\n`/`\r`.
fn scan_terminal_control_flags_flatten(s: &[u8]) -> ControlScan {
    const HI: u64 = 0x8080_8080_8080_8080;
    const SP: u64 = 0x2020_2020_2020_2020;
    const DEL: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const TAB: u64 = 0x0909_0909_0909_0909;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const CR: u64 = 0x0D0D_0D0D_0D0D_0D0D;

    let mut out = ControlScan::default();
    let mut i = 0;
    while i + 8 <= s.len() {
        let x = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let high = x & HI != 0;
        out.has_high |= high;
        // Both probes need a high-bit lane: `0xE2` is ≥ 0x80 and a C1 sequence
        // starts with `0xC2`, so an all-ASCII chunk answers `false` for free.
        if high {
            out.has_e2 |= eq_byte_lanes(x, 0xE2, HI, ONES) != 0;
            out.has_c1 |= c1_in_chunk(s, i, x);
        }
        let q = x ^ DEL;
        if q.wrapping_sub(ONES) & !q & HI != 0 {
            out.dirty = true;
        }
        // Exact C0 mask, minus high-byte lanes (see the multiline scanner),
        // then minus the layout bytes `push_flattened` consumes.
        let lt20 = !((x | HI).wrapping_sub(SP)) & HI & !(x & HI);
        if lt20 != 0 {
            let xh = x & HI;
            let tab = !(((x ^ TAB) | HI).wrapping_sub(ONES)) & HI & !xh;
            let nl = !(((x ^ NL) | HI).wrapping_sub(ONES)) & HI & !xh;
            let cr = !(((x ^ CR) | HI).wrapping_sub(ONES)) & HI & !xh;
            let breaks = tab | nl | cr;
            if out.first_break.is_none() && breaks != 0 {
                out.first_break = Some(i + breaks.trailing_zeros() as usize / 8);
            }
            if lt20 & !tab & !nl & !cr != 0 {
                out.dirty = true;
            }
        }
        i += 8;
    }
    for j in i..s.len() {
        let b = s[j];
        out.has_high |= b >= 0x80;
        if out.first_break.is_none() && matches!(b, b'\t' | b'\n' | b'\r') {
            out.first_break = Some(j);
        }
        out.dirty |= (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r') || b == 0x7F;
        out.has_e2 |= b == 0xE2;
        out.has_c1 |= b == 0xC2 && s.get(j + 1).is_some_and(|b| (0x80..=0x9F).contains(b));
    }
    out
}

/// Unicode bidirectional formatting characters: the U+200E–U+200F direction
/// marks (not overrides, but equally invisible and reordering-adjacent), the
/// U+202A–U+202E overrides, and the U+2066–U+2069 isolates.  Invisible, but they
/// reorder the surrounding display text — a path like `evil.cod\u{202E}exe` can
/// render with a spoofed extension.
pub(crate) fn is_bidi_override(c: char) -> bool {
    ('\u{200E}'..='\u{200F}').contains(&c)
        || ('\u{202A}'..='\u{202E}').contains(&c)
        || ('\u{2066}'..='\u{2069}').contains(&c)
}

/// Byte-level precheck for [`is_bidi_override`]: every bidi override/isolate
/// is a three-byte UTF-8 sequence starting with `0xE2` (`E2 80 AA..AE` for the
/// overrides, `E2 81 A6..A9` for the isolates).  Returns `false` for any
/// string that cannot contain one, letting callers skip the char-by-char
/// scan entirely.
pub(crate) fn has_bidi_override_bytes(s: &[u8]) -> bool {
    let mut i = 0;
    while let Some(off) = memchr::memchr(0xE2, &s[i..]) {
        let j = i + off;
        if j + 2 < s.len() {
            let b1 = s[j + 1];
            let b2 = s[j + 2];
            let is_override = (b1 == 0x80 && (0x8E..=0x8F).contains(&b2))
                || (b1 == 0x80 && (0xAA..=0xAE).contains(&b2))
                || (b1 == 0x81 && (0xA6..=0xA9).contains(&b2));
            if is_override {
                return true;
            }
        }
        i = j + 1;
    }
    false
}

/// Sanitize a filename for terminal display: [`sanitize_for_terminal`],
/// plus bidirectional formatting characters removed outright.  Unlike
/// control characters they carry no content — only rendering direction —
/// so replacement with U+FFFD would just inject noise into the path;
/// removal kills the spoofing.
pub(crate) fn sanitize_filename(s: &str) -> Cow<'_, str> {
    // `sanitize_for_terminal` already *removes* bidirectional
    // overrides/isolates whenever `has_bidi_override_bytes` is true (its
    // `bidi` branch), and control characters were replaced on the same pass.
    // The old second `has_bidi_override_bytes` filter here was therefore
    // unreachable and cost a full `memchr(0xE2)` scan per filename render.
    // The composed contract is pinned by
    // `sanitize_filename_removes_bidi_overrides` in the tests.
    sanitize_for_terminal(s)
}

/// Multi-line variant of [`sanitize_for_terminal`] for text whose `\n`/`\t`
/// are layout intent (user messages, help, notes) rather than raw terminal
/// bytes: keeps them verbatim and replaces every other control character —
/// CR and the C1 range included — with U+FFFD.  Same fast paths as
/// [`sanitize_for_terminal`]: pure printable ASCII (plus `\n`/`\t`) is
/// borrowed, and so is any non-ASCII string whose only control characters
/// are `\n`/`\t` — the common i18n case, which the naive `collect` would
/// otherwise allocate and copy on every render.
pub(crate) fn sanitize_for_terminal_multiline(s: &str) -> Cow<'_, str> {
    // One fused SWAR pass answers both gates (see
    // [`scan_terminal_control_flags_multiline`]): the printable-ASCII probe
    // already rejects every byte < 0x20, so the control probe re-scanned the
    // same clean bytes a second time.
    let bytes = s.as_bytes();
    let scan = scan_terminal_control_flags_multiline(bytes);
    let (has_high, dirty, has_c1) = (scan.has_high, scan.dirty, scan.has_c1);
    if !has_high && !dirty {
        // Printable ASCII plus `\t`/`\n` layout controls: borrow verbatim.
        return Cow::Borrowed(s);
    }
    // Byte-level gate: `\t`/`\n` preserved, every other C0/DEL/C1 rejected.
    // Pure i18n text without controls exits here after a bounded byte scan,
    // skipping the per-char `is_control` iteration entirely.  C1 is only
    // possible where high bytes exist, so the probe is gated.  As in
    // [`sanitize_for_terminal`], invisible bidi overrides/isolates escape the
    // control scan and force the owned path, where they are removed (same
    // convention as [`sanitize_filename`]).  Gated on `scan.has_e2`, not
    // `has_high`: see the argument on [`sanitize_for_terminal`] — every bidi
    // sequence starts with a `0xE2` lead byte, so the flag is exact.
    let bidi = scan.has_e2 && has_bidi_override_bytes(bytes);
    if has_high && !dirty && !has_c1 && !bidi {
        return Cow::Borrowed(s);
    }
    let needs_replace = |c: char| c.is_control() && c != '\n' && c != '\t';
    if bidi {
        Cow::Owned(
            s.chars()
                .filter(|c| !is_bidi_override(*c))
                .map(|c| if needs_replace(c) { '\u{FFFD}' } else { c })
                .collect(),
        )
    } else {
        Cow::Owned(
            s.chars()
                .map(|c| if needs_replace(c) { '\u{FFFD}' } else { c })
                .collect(),
        )
    }
}

/// Precomputed line-start table for O(log n) byte-offset → line lookup.
///
/// The four table fields a [`LineIndex`] shares with the batch cache.
///
/// Fields are private and construction goes through [`SharedSourceTables::of`] (from a
/// freshly built index — the only place that knows all four agree) or
/// [`SharedSourceTables::from_cache`] (the cache layer, whose entries were themselves
/// produced by `of`).  A positional `(starts, ascii, char_prefixes, generation)` list — the
/// shape that let a caller mix tables from different sources — is no longer spellable.
///
/// Airtightness note: `from_cache` still accepts four loose values, so the *cache* layer can
/// in principle mix them; `from_shared` therefore keeps its release-mode consistency checks
/// and falls back to a fresh scan instead of rendering wrong columns.  Removing even that
/// window means the cache entries themselves hold a `SharedSourceTables`, which is a
/// cross-module field change (`emitter`'s `Prebuilt`, `render`'s two index structs).
#[derive(Clone)]
pub(crate) struct SharedSourceTables {
    starts: Arc<[u32]>,
    ascii: bool,
    char_prefixes: LinePrefixTables,
    generation: u64,
}

impl SharedSourceTables {
    /// From a freshly built index: the only path that knows the four fields agree.
    pub(crate) fn of(index: &LineIndex<'_>) -> Self {
        Self {
            starts: Arc::clone(index.starts_arc()),
            ascii: index.ascii(),
            char_prefixes: index.char_prefixes_arc().clone(),
            generation: index.generation(),
        }
    }

    /// Re-assemble from a cache entry's fields (cache layer only — see the type doc).
    pub(crate) fn from_cache(
        starts: Arc<[u32]>,
        ascii: bool,
        char_prefixes: LinePrefixTables,
        generation: u64,
    ) -> Self {
        Self {
            starts,
            ascii,
            char_prefixes,
            generation,
        }
    }
}

/// The render path converts hundreds of spans per diagnostic; rescanning
/// the file from the start on every conversion (as `byte_to_linecol`
/// does) makes rendering O(spans × file_size). Build once per render and
/// share it across all conversions instead. `col` is a byte column within
/// the line, exactly like `byte_to_linecol`; display-cell conversion
/// happens later in the glyph renderer.
#[derive(Clone, Debug)]
pub(crate) struct LineIndex<'a> {
    /// Byte offset of the first byte of every line; always starts with 0.
    /// A trailing `'\n'` yields a final phantom entry equal to `total`.
    ///
    /// `Arc`-shared so a cached index can be handed to a later render by an
    /// O(1) clone instead of rebuilding or copying the table (see
    /// [`RenderCtx::line_index`](crate::diagnostics::render::RenderCtx::line_index)).
    starts: Arc<[u32]>,
    source: &'a str,
    /// Whether `source` is pure ASCII.  When true, byte columns equal
    /// character columns, so [`LineIndex::linecol_display`] skips the
    /// O(byte_col) char-count walk — the only remaining per-span linear
    /// cost in the JSON/HTML/short renderers on the ASCII sources that
    /// dominate real compiler output.  `is_ascii()` is a SIMD stdlib scan
    /// run once per source per batch (in [`LineIndex::new`]); the cached
    /// path carries the flag through [`LineIndex::from_shared`] so a cache
    /// hit never rescans.
    ascii: bool,
    /// Lazily-populated sparse non-ASCII char-extra table per line, keyed by
    /// line index.  Each entry is
    /// `Option<(CharPrefixTable, AtomicUsize)>`:
    /// `None` marks an ASCII line (char col == byte col, no table needed),
    /// `Some` holds one `(start_byte, cumulative_extra)` pair per non-ASCII
    /// char, where `cumulative_extra` is the running sum of `len_utf8 − 1`.
    /// Then `char_col(b) = b − cumulative_extra` for all non-ASCII chars whose
    /// `start_byte < b`, found by one `partition_point` + one array read.
    ///
    /// Only built on the non-ASCII path of [`linecol_display`](Self::linecol_display):
    /// without it, M spans landing on one long non-ASCII line each re-walk the
    /// line prefix (`O(M · byte_col)`); the table is built once per line in
    /// `O(K)` writes (K = non-ASCII char count) and reused by every later span
    /// on that line.  Memory is `O(K)` (8 bytes per non-ASCII char) vs the
    /// old dense prefix's `O(line.len())` (4 bytes per byte) — a single CJK
    /// char in a 1 MB line uses 8 bytes instead of 4 MB.  The map is
    /// `Arc`-shared across every index built from the same cached source
    /// (threaded through [`from_shared`](Self::from_shared) and the
    /// `RenderCtx` LRU), so a batch of M diagnostics over one long non-ASCII
    /// line pays one build, not M.  Behind a `RwLock` so the parallel render
    /// path can share one map across workers; storing `None` for ASCII lines
    /// (instead of an
    /// empty table) avoids any allocation per ASCII line in a mixed
    /// CJK/ASCII source — the common case for real code with comments.  The
    /// per-line table itself is an inline `SmallVec<[(u32, u32); 8]>`, not an
    /// inner `Arc`: the map is shared by reference, so the inner refcount was
    /// always 1 and only bought a heap allocation per non-ASCII line.
    ///
    /// Succinct rank/select (Jacobson 1989; Munro–Paterson) gives this
    /// bitvector `O(1)` rank in `n/8 + o(n)` bits, which is the asymptotic
    /// optimum.  Here K is typically a handful of non-ASCII chars per line,
    /// so `partition_point`'s two or three compares beat the succinct
    /// structure's constants; revisit only if a renderer ever ranks on
    /// multi-KB CJK lines with thousands of distinct offsets.
    ///
    /// CAVEAT: the `AtomicUsize` hint is shared through the map.  A racy
    /// `store` can only make a later query re-anchor via `partition_point`
    /// (the hint is a heuristic, never the answer), so the shared map is
    /// `Send + Sync` and one `LineIndex`/`RenderCtx` may hand its
    /// `char_prefixes` to parallel workers — see
    /// [`RenderCtx::preload_line_index`](crate::diagnostics::render::RenderCtx::preload_line_index).
    char_prefixes: LinePrefixTables,
    /// One-slot memo for lines **past** the high-byte gate, where no scan-built table
    /// exists and the sparse table is rebuilt on demand.
    ///
    /// The gate exists so a pathological source (16 MiB of high bytes) cannot make the
    /// line table cost `O(file)` memory; without a memo the fallback instead made *time*
    /// `O(line)` per query, so a 50 MB single-line CJK bundle degraded to the
    /// `O(spans × file)` behaviour `LineIndex` exists to prevent.  Spans cluster on one
    /// line in real renders, so one slot captures nearly all reuse, and keeping the memo
    /// here (not in `LinePrefixTables`, which is scan-owned and immutable) preserves the
    /// immutability the parallel render path relies on.  `Mutex`, not `RefCell`: an index
    /// is `Arc`-shared across workers.
    ///
    /// `Arc` (not a bare `Mutex`) because `LineIndex` is `Clone`: a clone shares one memo
    /// for the same source, which is both cheaper and the right sharing model.
    untrusted_memo: std::sync::Arc<std::sync::Mutex<[(usize, SmallVec<[(u32, u32); 8]>); 4]>>,
    /// Display-column checkpoint tables, direct-mapped by `(line, tab_width)`.
    ///
    /// The last O(prefix) per-span cost in the renderers: `byte_offset_to_display` walked
    /// the line's graphemes from byte 0 for every header/label/note query.  Four slots
    /// cover a diagnostic's header + label + note rows; a slot is rebuilt only when the key
    /// changes, and a pure-ASCII tab-free line builds no checkpoints at all.
    display_cols: std::sync::Arc<
        std::sync::Mutex<[(usize, usize, crate::diagnostics::glyph::DisplayColumns); 4]>,
    >,
    /// Process-wide monotonic id, assigned in [`LineIndex::new`] and carried
    /// through [`from_shared`](Self::from_shared).  Paired with the
    /// `char_col`'s tables are prebuilt by the scan and keyed by line index, so
    /// this id's remaining consumers are the renderer's cross-batch caches
    /// (`SharedSources`), which use it to tell a reused index from a fresh one.
    generation: u64,
}

/// Source of [`LineIndex`]'s generation ids.  One relaxed `fetch_add` per
/// index construction — `LineIndex::new` runs once per source per batch, so
/// this is never on the span-conversion hot path.
static LINE_INDEX_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Word-at-a-time content hash: 8-byte chunks, one multiply+shift each.
///
/// Used as the cross-batch identity of a source's text (the emitter's
/// `file_index_cache` key).  It must be a *content* hash — the cheap
/// `source_fingerprint` (first/last 8 + len) is only an ABA tripwire, and a
/// middle-only edit preserves it.  No cryptographic strength is needed: the
/// cache is also keyed by `FileId`.
///
/// [`scan_line_starts_and_ascii`] accumulates this exact function while it is
/// already walking the bytes, so a cache miss costs one pass instead of two.
pub(crate) fn content_hash_of(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = K ^ (bytes.len() as u64).wrapping_mul(K);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut w = 0u64;
        for (i, &b) in tail.iter().enumerate() {
            w |= u64::from(b) << (i * 8);
        }
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(K);
    h ^= h >> 29;
    h
}

/// Fused newline-start collection, ASCII predicate and [`content_hash_of`] in
/// one `u64`-chunk pass, replacing the two `memchr_iter` scans plus `is_ascii`
/// the old `LineIndex::new` ran.  Each 8-byte lane is tested for any high bit
/// (non-ASCII) and any `0x0A` (newline) via the standard `has_zero` SWAR
/// primitive (`(x ^ NL).wrapping_sub(ONES) & !(x ^ NL) & HI`).  The `Vec`
/// starts at `with_capacity(16)` and grows amortised — the previous
/// exact-capacity count pass is dropped because the combined scan already
/// sees every newline, so the win is one fewer full pass over the source
/// (the `is_ascii` SIMD scan), not a capacity saving.  For a ≥ 1 MB source
/// this is measurable; for small files the difference is noise.
///
/// The line scan already walks every 8-byte word, so folding the hash in costs
/// a multiply+shift per word and saves the separate `O(len)` hashing pass the
/// emitter used to pay on a cache miss.  The fused accumulation is bit-identical
/// to [`content_hash_of`] (same chunking, same tail assembly, same avalanche),
/// pinned by `fused_scan_hash_matches_the_standalone_function`.
///
/// Newline starts are `u32`: a diagnostic source over 4 GiB is not a real
/// input, and the halved table halves `line_of`'s binary-search cache
/// footprint (the table is read on every span conversion).  An `assert!`
/// guards the cast in release too (see the body); sources are ≤ 4 GiB in
/// every supported configuration.
/// One sparse char table per line, `None` for a pure-ASCII line.
///
/// `Arc<[Option<Arc<[(u32, u32)]>>]>`: 8 bytes per line, and only a non-ASCII line
/// allocates a table.  Shared by reference across parallel render workers.
///
/// `trusted` is the frontier the slots are authoritative for.  The high-byte gate
/// can fire **mid-source**, and the tables already built are inside the budget by
/// construction, so they are kept; only lines at or past the frontier fall back to
/// `char_col`'s on-demand build.  Without the frontier, a gated source had to
/// discard every table (a `None` slot must mean "proven ASCII", so it cannot also
/// mean "unknown") and every later render of that file silently reverted to
/// per-query builds for *all* lines, the pre-fusion cost.
#[derive(Clone, Debug, Default)]
pub(crate) struct LinePrefixTables {
    /// Concatenated per-line sparse tables (`(start_byte, cumulative_extra)`).
    ///
    /// One allocation for the whole source.  Each non-ASCII line used to own an
    /// `Arc<[(u32, u32)]>`, so a CJK-dense source just under the high-byte gate paid
    /// a tiny allocation *per non-ASCII line* on every index build — hundreds of
    /// thousands for a large file.  Two levels instead: this buffer plus the per-line
    /// index below, i.e. two allocations total.
    entries: Arc<[(u32, u32)]>,
    /// Per line: `Some((offset into `entries`, len))`, or `None` for a line the scan
    /// proved ASCII (char col == byte col).
    spans: Arc<[Option<(u32, u32)>]>,
    trusted: usize,
}

impl LinePrefixTables {
    /// The authoritative sparse table for `line`, or `None` when the line is
    /// (proven) ASCII or has no slot at all.
    #[inline]
    pub(crate) fn get(&self, line: usize) -> Option<&[(u32, u32)]> {
        let (off, len) = (*self.spans.get(line)?)?;
        Some(&self.entries[off as usize..off as usize + len as usize])
    }

    /// Is this line's slot authoritative (a real table, or a proven-ASCII line)?
    #[inline]
    pub(crate) fn is_trusted(&self, line: usize) -> bool {
        line < self.trusted
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// High-byte budget for the prebuilt tables.
///
/// Past this the scan stops building them (`prefixes_truncated`) and `char_col`
/// falls back to building one line's table on demand — so a pathological
/// all-CJK source keeps O(1) memory per query instead of 12 bytes per non-ASCII
/// byte.  16 Mi entries is ~128 MB of `(u32, u32)`; real sources are orders of
/// magnitude below it.
const MAX_PREFIX_HIGH_BYTES: usize = 16 * 1024 * 1024;

/// Bytes of continuation following the lead byte at `p` (0..=3).
#[inline]
fn continuation_len(bytes: &[u8], p: usize) -> u32 {
    let mut k = 0usize;
    while k < 3 && p + 1 + k < bytes.len() && bytes[p + 1 + k] & 0xC0 == 0x80 {
        k += 1;
    }
    k as u32
}

fn scan_line_starts_and_ascii(source: &str) -> (Vec<u32>, bool, u64, LinePrefixTables) {
    scan_line_starts_and_ascii_capped(source, MAX_PREFIX_HIGH_BYTES)
}

/// [`scan_line_starts_and_ascii`] with an explicit high-byte budget, so the gated
/// path is testable without a 16 Mi-char fixture.
/// Test-only switch that forces the 8-byte SWAR loop, so the differential test can
/// compare the two implementations over one corpus.  Not a tuning knob.
#[cfg(test)]
static FORCE_SWAR_SCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
fn force_swar_scan() -> bool {
    #[cfg(test)]
    {
        return FORCE_SWAR_SCAN.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Exact newline and high-byte masks for one 32-byte chunk.
///
/// # Safety
///
/// `ptr` must be valid for 32 readable bytes, and the caller must have verified
/// `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_chunk_avx2(ptr: *const u8) -> (u32, u32) {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
    };
    // SAFETY: the caller's contract covers both the load and the compare.
    let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
    let nl = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'\n' as i8));
    // `movemask` takes each lane's sign bit: the compare result for the newline mask,
    // and for `v` itself exactly `byte & 0x80` (which is what the SWAR `x & HI` tests).
    (
        _mm256_movemask_epi8(nl) as u32,
        _mm256_movemask_epi8(v) as u32,
    )
}

/// The high-byte gate's terminal step.
///
/// Fires in three places (the AVX2 marks loop, the SWAR marks loop, and the scalar tail) because the
/// loops are separately unrolled; one definition because they must not drift.  Zeroing `extra`
/// beside `cur` is what makes "the gate fired ⇒ nothing more is pushed until line close" a *local*
/// fact rather than an argument spanning the rest of the function: today every later high byte
/// re-enters this branch and `close_line!` resets `extra`, but the doc contemplates a per-line
/// budget as the memory escape hatch — resuming table building would then carry a stale cumulative
/// offset into fresh entries, i.e. wrong character columns with no crash.
macro_rules! gate_fired {
    ($trusted:ident, $spans:ident, $cur:ident, $extra:ident) => {{
        if $trusted == usize::MAX {
            $trusted = $spans.len();
        }
        $cur.clear();
        $extra = 0;
    }};
}

fn scan_line_starts_and_ascii_capped(
    source: &str,
    max_high_bytes: usize,
) -> (Vec<u32>, bool, u64, LinePrefixTables) {
    const HI: u64 = 0x8080_8080_8080_8080;
    const NL: u64 = 0x0A0A_0A0A_0A0A_0A0A;
    const ONES: u64 = 0x0101_0101_0101_0101;

    let bytes = source.as_bytes();
    // Enforced in release too.  The line table is `Vec<u32>`: a source past
    // 4 GiB used to truncate every offset in release builds, making *every*
    // downstream line/column wrong with no signal at all.  A loud panic is
    // strictly better than silent corruption.  With this assertion the
    // `as u32` casts below are provably in range (`off < len <= u32::MAX`).
    assert!(
        bytes.len() <= u32::MAX as usize,
        "scan_line_starts_and_ascii: source exceeds the 4 GiB u32-offset limit ({} bytes)",
        bytes.len()
    );
    // Pre-size for large sources so the 16→2× growth chain doesn't pay ~16
    // reallocations (and megabytes of copies) on a 50 MB / 1M-line file.  A
    // heuristic average line length (~32 bytes) avoids the extra full `memchr`
    // counting pass; `Vec` growth absorbs the residual.  The 1 MB gate keeps
    // typical small sources from over-reserving.
    const LARGE_SOURCE_THRESHOLD: usize = 1 << 20;
    let capacity = if bytes.len() >= LARGE_SOURCE_THRESHOLD {
        (bytes.len() / 32 + 16).min(1 << 20)
    } else {
        16
    };
    let mut starts = Vec::with_capacity(capacity);
    starts.push(0);
    // Per-line sparse char tables, built here because this scan already touches
    // every byte and computes the high-bit mask per chunk: the old shape rebuilt
    // each line's table lazily inside a `RwLock` write on the render hot path.
    let mut spans: Vec<Option<(u32, u32)>> = Vec::with_capacity(capacity);
    // Grown, not pre-sized: the pathological single-line case then allocates only for
    // the entries it has (the pre-size above is what `shrink_to_fit` reclaims for
    // `starts`/`spans`).
    let mut entries: Vec<(u32, u32)> = Vec::new();
    // Current line's pending table + running `len_utf8 - 1` sum.
    let mut cur: SmallVec<[(u32, u32); 8]> = SmallVec::new();
    let mut extra: u32 = 0;
    let mut line_start: u32 = 0;
    let mut high_bytes: usize = 0;
    // Slot count reached when the gate fired; `usize::MAX` while it has not.
    let mut trusted: usize = usize::MAX;

    /// Close the current line: record its table (or `None` for a pure-ASCII line).
    macro_rules! close_line {
        () => {{
            // Always push, even past the gate: skipping a push would break the
            // one-slot-per-line invariant and force the wholesale reset that used to
            // throw away the tables built *before* the gate.
            match cur.len() {
                0 => spans.push(None),
                len => {
                    let off = entries.len() as u32;
                    entries.extend_from_slice(&cur);
                    spans.push(Some((off, len as u32)));
                }
            }
            cur.clear();
            extra = 0;
        }};
    }

    let mut ascii = true;
    // Fused content hash (see `content_hash_of`): identical chunking, so the
    // two implementations cannot drift.
    //
    // `memchr_iter` was considered here and declined: this loop is *fused* —
    // one pass yields line starts, the ASCII flag and the content hash.  A
    // vectorised memchr would skip the bytes the hash must consume, so it would
    // buy a faster newline scan at the price of a second full pass.  The fused
    // single pass is the right shape until profiling says otherwise.
    const HASH_K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut hash = HASH_K ^ (bytes.len() as u64).wrapping_mul(HASH_K);
    let mut i = 0;
    // 32-byte AVX2 fast path.  The kernel below returns the newline mask and the
    // high-byte mask *exactly* (a byte equality compare and the sign bits), which is
    // simpler than the SWAR form's `0x8A`-aliasing argument — that trap exists only
    // because SWAR subtracts across lanes.  The hash is still consumed as four 8-byte
    // words through the identical mixer, so `content_hash_of`'s fused equality is
    // preserved by construction rather than by a second implementation agreeing.
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && !force_swar_scan() {
        while i + 32 <= bytes.len() {
            for k in 0..4 {
                let at = i + k * 8;
                let x = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
                hash = (hash ^ x).wrapping_mul(HASH_K);
                hash ^= hash >> 29;
                ascii &= (x & HI) == 0;
            }
            // SAFETY: `i + 32 <= bytes.len()` bounds the 32-byte read, and the `avx2`
            // feature was detected immediately above.
            let (nl_mask, hi_mask) = unsafe { scan_chunk_avx2(bytes.as_ptr().add(i)) };
            let mut marks = (nl_mask | hi_mask) as u64;
            while marks != 0 {
                let lane = marks.trailing_zeros() as usize;
                let pos = i + lane;
                marks &= marks - 1;
                if (nl_mask >> lane) & 1 != 0 {
                    starts.push((pos + 1) as u32);
                    close_line!();
                    line_start = (pos + 1) as u32;
                } else {
                    high_bytes += 1;
                    if high_bytes > max_high_bytes {
                        gate_fired!(trusted, spans, cur, extra);
                    } else if bytes[pos] & 0xC0 != 0x80 {
                        let k = continuation_len(bytes, pos);
                        extra += k;
                        cur.push((pos as u32 - line_start, extra));
                    }
                }
            }
            i += 32;
        }
    }
    while i + 8 <= bytes.len() {
        let x = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        hash = (hash ^ x).wrapping_mul(HASH_K);
        hash ^= hash >> 29;
        ascii &= (x & HI) == 0;
        let q = x ^ NL;
        // Exact newline mask.  `(q | HI)` is 0x80 at a newline and also at
        // `x == 0x8A` (q == 0x80), a valid UTF-8 continuation byte (`Ċ`, `Ŋ`,
        // `ꀀ`…).  `!q` has bit 7 set only for q < 0x80, so it keeps only the
        // true newline lane while preserving the no-borrow property (`| HI`
        // makes every lane ≥ 0x80 before the subtraction).
        let mask = !((q | HI).wrapping_sub(ONES)) & !q & HI;
        let hi = x & HI;
        // One ascending pass over the flagged lanes: a newline closes the line, a
        // high byte contributes its char's extra bytes.  A lane can never be both
        // (0x0A is not ≥ 0x80), and ascending order is what makes the per-line
        // attribution identical to the `starts` pushes.
        let mut marks = mask | hi;
        while marks != 0 {
            let lane = marks.trailing_zeros() as usize / 8;
            let pos = i + lane;
            marks &= marks - 1;
            if bytes[pos] == b'\n' {
                starts.push((pos + 1) as u32);
                close_line!();
                line_start = (pos + 1) as u32;
            } else {
                // A high byte: count it against the memory gate, then open an
                // entry only at a *lead* byte (continuation bytes add no entry —
                // the convention is one `(start, cumulative_extra)` per char).
                high_bytes += 1;
                if high_bytes > max_high_bytes {
                    gate_fired!(trusted, spans, cur, extra);
                } else if bytes[pos] & 0xC0 != 0x80 {
                    let k = continuation_len(bytes, pos);
                    extra += k;
                    cur.push((pos as u32 - line_start, extra));
                }
            }
        }
        i += 8;
    }
    let mut tail_word = 0u64;
    let mut tail_len = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        ascii &= b < 0x80;
        tail_word |= u64::from(b) << (tail_len * 8);
        tail_len += 1;
        if b == b'\n' {
            starts.push((i + 1) as u32);
            close_line!();
            line_start = (i + 1) as u32;
        } else if b >= 0x80 {
            high_bytes += 1;
            if high_bytes > max_high_bytes {
                gate_fired!(trusted, spans, cur, extra);
            } else if b & 0xC0 != 0x80 {
                let k = continuation_len(bytes, i);
                extra += k;
                cur.push((i as u32 - line_start, extra));
            }
        }
        i += 1;
    }
    if tail_len != 0 {
        hash = (hash ^ tail_word).wrapping_mul(HASH_K);
        hash ^= hash >> 29;
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(HASH_K);
    hash ^= hash >> 29;
    // The heuristic above sizes for ~32 bytes per line; a source that is one
    // 50 MB line (minified bundles are real diagnostic inputs) therefore held a
    // 1 M-entry / 4 MB allocation for a one-entry table.  The transient is really
    // 12 MB: `prefixes` is pre-sized with the same capacity and `Option<Arc<[..]>>`
    // is 8 bytes, so that case holds 4 MB of `starts` plus 8 MB of slots until the
    // `shrink_to_fit` below reclaims both.  For typical sources capacity is already
    // within a growth step of `len`, so this is a no-op check there and one small
    // realloc (plus the reclaim) in exactly the pathological case.
    starts.shrink_to_fit();
    // The final line: `starts` begins with line 0 and gains one entry per newline,
    // so this leaves `prefixes.len() == starts.len()` — with or without the gate.
    close_line!();
    debug_assert_eq!(spans.len(), starts.len(), "one prefix slot per line");
    if trusted == usize::MAX {
        trusted = spans.len();
    }
    debug_assert!(trusted <= spans.len());
    spans.shrink_to_fit();
    entries.shrink_to_fit();
    (
        starts,
        ascii,
        hash,
        LinePrefixTables {
            entries: entries.into(),
            spans: spans.into(),
            trusted,
        },
    )
}

/// Uppercase hex digits for percent-encoding (`%XX`) in URLs.
///
/// One definition for the whole diagnostic layer: `term` (local `file://`
/// links), `error_code` (`--explain` URLs) and the renderer primitives all
/// encode path segments.  A shared const is the only way two encoders cannot
/// drift, and living here — rather than inside a renderer module — keeps
/// non-renderer subsystems from importing a table *through* a renderer.
pub(crate) const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

impl<'a> LineIndex<'a> {
    pub(crate) fn new(source: &'a str) -> Self {
        Self::new_hashed(source).0
    }

    /// [`Self::new`] plus the fused content hash: the emitter keys its
    /// cross-batch table cache by that hash, so returning it from the scan it
    /// already performs turns a cache miss from two full passes (hash, then
    /// index) into one.
    pub(crate) fn new_hashed(source: &'a str) -> (Self, u64) {
        let (starts, ascii, content_hash, char_prefixes) = scan_line_starts_and_ascii(source);
        let idx = Self {
            starts: Arc::from(starts),
            source,
            ascii,
            char_prefixes,
            untrusted_memo: std::sync::Arc::new(std::sync::Mutex::new(std::array::from_fn(|_| {
                (usize::MAX, SmallVec::new())
            }))),
            display_cols: std::sync::Arc::new(std::sync::Mutex::new(std::array::from_fn(|_| {
                (
                    usize::MAX,
                    usize::MAX,
                    crate::diagnostics::glyph::DisplayColumns::new("", 1),
                )
            }))),
            generation: LINE_INDEX_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        (idx, content_hash)
    }

    /// Build an index for `source` from a table already computed for that same
    /// source (shared by [`RenderCtx::line_index`](crate::diagnostics::render::RenderCtx::line_index)).
    /// The caller guarantees `starts` describes `source`'s newlines,
    /// `ascii` equals `source.is_ascii()`, and `char_prefixes` was created for
    /// the same source — passing a stale flag makes
    /// [`linecol_display`](Self::linecol_display) produce wrong columns on
    /// non-ASCII sources, so cache insertions always source the flag and the
    /// prefix map from [`ascii`](Self::ascii) and
    /// [`char_prefixes_arc`](Self::char_prefixes_arc) of a freshly-built index.
    pub(crate) fn from_shared(source: &'a str, tables: SharedSourceTables) -> Self {
        let SharedSourceTables {
            starts,
            ascii,
            char_prefixes,
            generation,
        } = tables;
        // **Release-checked, and a fallback rather than an assert.**  These were
        // `debug_assert!`s, so in release a stale or wrong-source cache entry rendered
        // silently wrong columns: `ascii` selects the byte==column shortcut, and the prefix
        // tables drive every char column.  The invariants that are O(1) — `starts[0] == 0`
        // and the table-length match below — are now enforced in every build, and a
        // violation degrades to a fresh scan instead of wrong output.
        //
        // The `ascii` check is the only O(source) one, so it runs unconditionally for small
        // sources (cheap, and where the cache is least valuable) and stays a debug assert for large
        // ones.  Closing that last window properly means making the four fields impossible to
        // mismatch (an opaque `SharedSourceTables`), which requires the cache entry type in
        // `emitter` to hold it — a separate, larger change.
        //
        // **The residual window, stated exactly** (it is the one remaining way this function can
        // render wrong columns in release): for `source.len() > ASCII_RECHECK_MAX` a stale
        // `ascii == true` paired with a non-ASCII source is *not* caught, and `ascii` selects the
        // byte==column shortcut, so every char column on that source is wrong.  Reaching it needs a
        // cache entry whose `(starts, char_prefixes)` describe one source while the reused `ascii`
        // flag describes a different, all-ASCII one — i.e. a producer that reuses a `Prebuilt`
        // across differing content, which the `(ptr, len)` + content-hash keying exists to prevent.
        // `ascii` cannot simply be derived from `char_prefixes.is_empty()`: that predicate is the
        // *missing-table* sentinel, not "all ASCII" (an all-ASCII source still has one `spans` slot
        // per line).  The fix is the opaque-table change named above; until then this is the
        // documented boundary rather than an unnoticed one.
        debug_assert_eq!(
            ascii,
            source.is_ascii(),
            "from_shared: ascii flag does not match source"
        );
        const ASCII_RECHECK_MAX: usize = 4096;
        if starts.first().copied() != Some(0)
            || (!char_prefixes.is_empty() && char_prefixes.len() != starts.len())
            || (source.len() <= ASCII_RECHECK_MAX && ascii != source.is_ascii())
        {
            return Self::new(source);
        }
        // Structural identity check: the tables are sized by line count (one slot
        // per line) unless the high-byte gate left them empty, so a table vector
        // with a different length was built for a different source.  That catches
        // the swap-whole-tables mistake — the realistic one — for free.
        debug_assert!(
            char_prefixes.is_empty() || char_prefixes.len() == starts.len(),
            "from_shared: char_prefixes has {} slots for {} lines — the tables belong \
             to a different source",
            char_prefixes.len(),
            starts.len()
        );
        Self {
            starts,
            source,
            ascii,
            char_prefixes,
            untrusted_memo: std::sync::Arc::new(std::sync::Mutex::new(std::array::from_fn(|_| {
                (usize::MAX, SmallVec::new())
            }))),
            display_cols: std::sync::Arc::new(std::sync::Mutex::new(std::array::from_fn(|_| {
                (
                    usize::MAX,
                    usize::MAX,
                    crate::diagnostics::glyph::DisplayColumns::new("", 1),
                )
            }))),
            generation,
        }
    }

    /// Display column of `byte_offset` within `line_idx`, with the per-line checkpoint
    /// table memoized.
    ///
    /// Same answer as [`crate::diagnostics::glyph::byte_offset_to_display`], which is the
    /// differential oracle (`display_columns_matches_the_scalar_walk_at_every_boundary`).
    /// Keyed by `tab_width` because tab expansion depends on it; the memo is per-`LineIndex`
    /// (a fresh index for another source gets a fresh memo — `Clone` shares, and clones
    /// share the source).
    pub(crate) fn display_col(
        &self,
        line_idx: usize,
        byte_offset: usize,
        tab_width: usize,
    ) -> usize {
        let line = self.line(line_idx);
        let slot = (line_idx ^ (tab_width << 8)) & 3;
        let mut memo = self.display_cols.lock().unwrap_or_else(|e| e.into_inner());
        if memo[slot].0 != line_idx || memo[slot].1 != tab_width {
            memo[slot] = (
                line_idx,
                tab_width,
                crate::diagnostics::glyph::DisplayColumns::new(line, tab_width),
            );
        }
        memo[slot].2.col(line, byte_offset, tab_width)
    }

    /// The index's generation id (assigned in `new`, carried through
    /// `from_shared`).  The renderer's caches use it to distinguish a reused
    /// `LineIndex` from a rebuilt one.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The shared line-start table, for caching by an O(1) `Arc` clone.
    pub(crate) fn starts_arc(&self) -> &Arc<[u32]> {
        &self.starts
    }

    /// The shared char-prefix cache, for caching by an O(1) `Arc` clone so a
    /// batch of diagnostics over one non-ASCII source reuses the per-line
    /// prefixes instead of rebuilding them per render.  The cache is
    /// `Send + Sync` (the hint is an `AtomicUsize`, the map behind a
    /// `RwLock`), so the parallel render path shares one map across workers.
    pub(crate) fn char_prefixes_arc(&self) -> &LinePrefixTables {
        &self.char_prefixes
    }

    /// Whether the indexed source is pure ASCII (byte cols == char cols).
    pub(crate) fn ascii(&self) -> bool {
        self.ascii
    }

    /// 0-based line containing `offset` (clamped to the source length).
    pub(crate) fn line_of(&self, offset: usize) -> usize {
        match self
            .starts
            .binary_search(&(offset.min(self.source.len()) as u32))
        {
            Ok(line) => line,
            Err(0) => 0,
            Err(next) => next - 1,
        }
    }

    /// Same mapping as [`byte_to_linecol`], via binary search.
    pub(crate) fn to_linecol(&self, byte_offset: usize) -> label::SourcePos {
        let clamped = floor_char_boundary(self.source, byte_offset);
        let line = self.line_of(clamped);
        label::SourcePos {
            line,
            col: clamped - self.starts[line] as usize,
        }
    }

    /// `(line, char_col)` for `byte_offset`: the line index and the
    /// character (not byte) column within that line, for display purposes.
    /// Byte length of the source this index was built from.
    ///
    /// Exists so a renderer holding a caller-supplied index can prove the index describes the
    /// source it is about to render (`chain.rs`'s debug integrity check): a stale index produces
    /// silently wrong `file:line:col`, which is the one output the chain renderer treats as
    /// sacrosanct.
    pub(crate) fn source_len(&self) -> usize {
        self.source.len()
    }

    pub(crate) fn linecol_display(&self, byte_offset: usize) -> (usize, usize) {
        let pos = self.to_linecol(byte_offset);
        // ASCII fast path: byte columns equal character columns, so skip the
        // O(byte_col) `chars().count()` walk.  This is the only per-span
        // linear cost left in the JSON/HTML/short renderers, and real
        // compiler output is overwhelmingly ASCII.
        if self.ascii {
            return (pos.line, pos.col);
        }
        // O(1) table access: `lines().nth(pos.line)` would rescan the whole
        // file on every span conversion (JSON, HTML, short, call chains).
        // `line()` strips the same `\n` / `\r\n` terminator that
        // `str::lines` does, so the column arithmetic is unchanged.  The
        // char-count walk is amortised across every span on this line (and
        // every render sharing this index) by the per-line prefix cache in
        // [`char_col`](Self::char_col).
        (pos.line, self.char_col(pos.line, pos.col))
    }

    /// Character column of `byte_col` within line `line_idx`, via a lazily
    /// built per-line cache.
    ///
    /// `byte_col` must be a char boundary within the line (callers arrive via
    /// `floor_char_boundary`).  On the first query for a line, an ASCII line
    /// caches an empty table (char count = byte count, so the answer is
    /// `byte_col`); a non-ASCII line builds a sparse
    /// `(start_byte, cumulative_extra)` table — one entry per non-ASCII char,
    /// where `cumulative_extra` is the running sum of `len_utf8 − 1`.
    /// Later queries on the same line — across every span and every render in
    /// the batch that shares this index — are `O(1)` for ASCII lines and
    /// `O(log K)` (K = non-ASCII char count) for non-ASCII lines, both via
    /// `partition_point`.  Caching ASCII lines too avoids an `O(line_len)`
    /// `is_ascii()` rescan per span on non-ASCII sources with ASCII lines.
    /// Memory is `O(K)` (8 bytes per non-ASCII char) vs the old dense
    /// prefix's `O(line.len())`.
    fn char_col(&self, line_idx: usize, byte_col: usize) -> usize {
        // Today every caller derives `line_idx` from `line_of`/`to_linecol`, so
        // this is an internal invariant — but a stale index (a caller holding
        // one across an index rebuild) would panic inside `line()` with no
        // context.
        debug_assert!(
            line_idx < self.starts.len(),
            "char_col: line_idx {line_idx} out of range ({} lines)",
            self.starts.len()
        );
        let line = self.line(line_idx);
        let b = byte_col.min(line.len());
        // Every caller arrives through `floor_char_boundary`, but nothing
        // enforced it: a mid-character `b` silently counts the containing
        // char's full width.  Make the invariant loud under fuzzing/debug.
        debug_assert!(
            line.is_char_boundary(b),
            "char_col: byte_col {byte_col} is not a char boundary on line {line_idx}"
        );
        // Pure lookup: the tables are built by the source scan, so there is no
        // lock, no per-line build and no memo on this path.  An empty vector is
        // the high-byte gate's sentinel (a real source always has ≥1 line), and
        // then this line's table is built on demand — the pre-fusion behaviour,
        // kept for pathological sources only.
        // Frontier first: below it every slot is authoritative (a real table, or a
        // line the scan proved ASCII); at or past it nothing is known and the line is
        // built on demand.
        if self.char_prefixes.is_trusted(line_idx) {
            return match self.char_prefixes.get(line_idx) {
                Some(table) => char_col_from_sparse(Some(table), b),
                None => b,
            };
        }
        // Past the gate: rebuild on demand, but reuse the last line's table.  Without the
        // memo every span on the line paid a full `build_line_sparse` walk, which is the
        // `O(spans × file)` cliff the gate was supposed to avoid trading into.
        {
            let mut memo = self
                .untrusted_memo
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Four direct-mapped slots, not one: interleaved lines (label + source row
            // + note on different lines) thrashed a single slot into a rebuild per query,
            // which is the `O(spans × line)` cost this memo exists to remove.
            let slot = line_idx & 3;
            if memo[slot].0 == line_idx {
                let table = &memo[slot].1;
                return char_col_from_sparse(
                    if table.is_empty() {
                        None
                    } else {
                        Some(table.as_slice())
                    },
                    b,
                );
            }
            let sparse = build_line_sparse(line);
            let answer = char_col_from_sparse(
                if sparse.is_empty() {
                    None
                } else {
                    Some(sparse.as_slice())
                },
                b,
            );
            memo[slot] = (line_idx, sparse);
            answer
        }
    }

    /// `(start_line, end_line)` for a span: the lines of the first and last
    /// byte the span covers. `Span::end` is exclusive, so an end sitting
    /// exactly on a line start belongs to the *previous* line — attributing
    /// it to the following line made renderers suppress a label's message on
    /// the line that drew its underline and then treat it as already shown,
    /// so it was never emitted anywhere. A zero-width span covers no byte
    /// and keeps the position's own line.
    pub(crate) fn span_lines(&self, span: Span) -> (usize, usize) {
        let start = span.start_as_usize();
        let end = span.end_as_usize();
        let start_line = self.line_of(floor_char_boundary(self.source, start));
        let end_offset = if end > start { end - 1 } else { end };
        // The end is almost always on `start_line` or the next one, so this
        // second lookup gallops from the first answer instead of running an
        // independent `O(log L)` search (`line_of_from` re-anchors when the
        // stream is descending, so it is exact for any input).
        let end_line = self.line_of_from(floor_char_boundary(self.source, end_offset), start_line);
        (start_line, end_line)
    }

    /// [`Self::line_of`] with a starting hint: a line known to be at or before
    /// the answer (an ascending render stream).
    ///
    /// Gallops forward over a short window first — consecutive spans of one
    /// diagnostic are on the same or adjacent lines, so this is usually zero or
    /// one comparisons — and falls back to `partition_point` for a hint that
    /// overshot (a descending stream) or a jump past the window.
    fn line_of_from(&self, offset: usize, hint: usize) -> usize {
        let off = offset.min(self.source.len()) as u32;
        let n = self.starts.len();
        if n == 0 {
            return 0;
        }
        let mut k = hint.min(n - 1);
        // `starts` is strictly ascending, so the answer is exactly
        // `partition_point(.., <= off) - 1`; only the route differs.
        const GALLOP: usize = 8;
        let mut walked = 0;
        while walked < GALLOP && k + 1 < n && self.starts[k + 1] <= off {
            k += 1;
            walked += 1;
        }
        if self.starts[k] > off || (k + 1 < n && self.starts[k + 1] <= off) {
            k = self.starts.partition_point(|&s| s <= off).saturating_sub(1);
        }
        k
    }

    pub(crate) fn starts(&self) -> &[u32] {
        &self.starts[..]
    }

    pub(crate) fn line_count(&self) -> usize {
        self.starts.len()
    }

    pub(crate) fn source(&self) -> &'a str {
        self.source
    }

    /// `str::lines()`-style line count: a trailing `\n` does not create an
    /// extra "ghost" line, and an empty source counts zero lines.
    pub(crate) fn text_line_count(&self) -> usize {
        if self.source.is_empty() {
            return 0;
        }
        let last = self.starts.last().copied().unwrap_or(0) as usize;
        if last == self.source.len() && self.source.ends_with('\n') {
            self.starts.len() - 1
        } else {
            self.starts.len()
        }
    }

    /// Return the content of line `line_idx` (0-based), with the trailing `\n` / `\r\n` removed.
    pub(crate) fn line(&self, line_idx: usize) -> &'a str {
        let start = self.starts[line_idx] as usize;
        let end = self
            .starts
            .get(line_idx + 1)
            .map_or(self.source.len(), |&e| e as usize);
        let raw = &self.source[start..end];
        match raw.as_bytes().last() {
            Some(b'\n') => {
                let raw = &raw[..raw.len() - 1];
                raw.strip_suffix('\r').unwrap_or(raw)
            }
            _ => raw,
        }
    }
}

// The module itself is not gated: the SWAR scanner, `LineIndex`, `MultiSpan` and
// `DiagFilter` suites are pure and single-threaded, so they must also run under
// the concurrency checker.  Only the emitter-driven tests inside it carry their
// own `#[cfg(not(feature = "shuttle"))]`, which is what the compiler verifies.
#[cfg(test)]
mod tests {
    use super::*;
    /// A panic mid-`humanize` must leave a *renderable* diagnostic, not half-derived state.
    ///
    /// `humanize_contained` is the containment half of the policy (it wraps the batch paths
    /// and `DiagCtxt::humanize_pending*`); this pins the repair half, against exactly the
    /// state `humanize` creates before it can fail: message `mem::take`n, labels drained, the
    /// derived-label count stale.
    #[test]
    fn poison_for_render_overwrites_every_half_derived_field() {
        let span = crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 1);
        let mut d =
            Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::TypeMismatch {
                expected: "int".into(),
                found: "string".into(),
                span,
                found_span: None,
                reason: None,
                context: None,
            });
        d.set_file_name("src/poison.pn");
        // The half-derived shape a panic can leave behind.
        d.message = std::borrow::Cow::Borrowed("");
        d.labels.clear();
        d.poison_for_render();

        assert!(d.is_humanized(), "the latch must hold so nothing retries");
        assert!(
            d.message().contains("humanization panicked"),
            "the failure must be visible, not empty: {:?}",
            d.message()
        );
        assert!(d.help().is_none());
        assert!(d.suggestions().is_empty());
        assert!(d.labels().is_empty());
        // Still a usable diagnostic: the id is recomputed over the placeholder rather than
        // left describing the text that vanished.
        assert_ne!(d.stable_id(), 0);
    }

    use crate::ast::Span;

    /// The fused scan must be *bit-identical* to the standalone mixer, and must
    /// still produce the same line table as `LineIndex::new`.  If the two ever
    /// drift, every emitter cache lookup would miss forever (and the cache
    /// would silently stop working rather than fail loudly).
    #[test]
    fn fused_scan_hash_matches_the_standalone_function() {
        let mut cases: Vec<String> = vec![
            String::new(),
            "a".into(),
            "abc".into(),
            "a\nb".into(),
            "x".repeat(7),
            "x".repeat(8),
            "x".repeat(9),
            "line\n".repeat(37),
            format!("{}\n{}", "y".repeat(1023), "誤り"),
            "\u{0080}\u{202E}".into(),
            "\n".repeat(64),
        ];
        cases.push(format!("{}\t{}", "z".repeat(64), "w".repeat(64)));
        for s in &cases {
            let (starts, ascii, hash, _tables) = super::scan_line_starts_and_ascii(s);
            assert_eq!(hash, super::content_hash_of(s.as_bytes()), "{s:?}");
            let idx = super::LineIndex::new(s);
            assert_eq!(idx.starts_arc().as_ref(), starts.as_slice(), "{s:?}");
            assert_eq!(idx.ascii(), ascii, "{s:?}");
            let (idx2, h2) = super::LineIndex::new_hashed(s);
            assert_eq!(h2, hash, "{s:?}");
            assert_eq!(idx2.starts_arc().as_ref(), starts.as_slice(), "{s:?}");
        }
        // A middle-only edit must move the hash (the whole point of hashing the
        // content instead of the first/last 8 bytes).
        let a = format!("{}{}{}", "a".repeat(64), 'b', "a".repeat(64));
        let b = format!("{}{}{}", "a".repeat(64), 'c', "a".repeat(64));
        assert_ne!(
            super::content_hash_of(a.as_bytes()),
            super::content_hash_of(b.as_bytes())
        );
    }

    #[test]
    fn test_basic_error() {
        let diag = Diagnostic::error("something went wrong")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 5, 12))
            .with_suggestion("try using `as` to cast");
        assert!(diag.is_error());
        assert_eq!(diag.code.as_ref().unwrap().code(), "E030");
    }

    /// Identity must follow the *canonical* spelling.  A verbatim
    /// `unknown("e030")` equals `new("e030")` under `PartialEq`, so it must
    /// also hash to the same `stable_id` and be matched by `--allow e030`.
    /// A kind-built diagnostic that was never humanized must not render an empty
    /// message in release (where the `debug_assert` is gone).
    ///
    /// `Display` falls back to the kind's own message; `unhumanized_fallback` is
    /// the testable half of that guarantee, so this pins it in *both* profiles
    /// instead of only under a release run the suite does not do.
    #[test]
    fn display_falls_back_to_the_kind_message_when_not_humanized() {
        let mut d =
            Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::TypeMismatch {
                expected: "int".into(),
                found: "string".into(),
                span: crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 1),
                found_span: None,
                reason: None,
                context: None,
            });
        d.set_file_name("src/x.pn");
        let fallback = d
            .unhumanized_fallback()
            .expect("a kind-built, un-humanized diagnostic needs a fallback");
        assert!(!fallback.is_empty());
        assert!(
            fallback.contains("type mismatch"),
            "the fallback must be the rendered kind message: {fallback:?}"
        );

        // Once humanized there is nothing to fall back to (the hot path pays one
        // `Option` and no allocation).
        d.humanize();
        assert!(d.unhumanized_fallback().is_none());

        // A manually built diagnostic is never in the violating state.
        assert!(Diagnostic::error("plain").unhumanized_fallback().is_none());
    }

    /// `stable_id` must not depend on the rendering language.
    ///
    /// `humanize` advertises mid-session `Lang::set` (the LSP re-translation
    /// contract) and invalidates the cached id, so hashing the rendered message
    /// gave the *same kind* two ids — one per language — which is precisely the
    /// silent double-report a cross-run dedup key exists to prevent.  The identity
    /// hashed for a kind-built diagnostic is now the slug plus its argument values.
    #[test]
    fn stable_id_is_language_independent_for_kind_built_diagnostics() {
        use crate::diagnostics::i18n::{Lang, acquire_test_lang_lock};
        let _guard = acquire_test_lang_lock();

        let make = || {
            let mut d =
                Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::TypeMismatch {
                    expected: "int".into(),
                    found: "string".into(),
                    span: crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 1),
                    found_span: None,
                    reason: None,
                    context: None,
                });
            d.set_file_name("src/x.pn");
            d
        };

        Lang::set(Lang::En);
        let mut en = make();
        en.humanize();
        let en_id = en.stable_id();
        let en_msg = en.message().to_string();

        Lang::set(Lang::Ru);
        let mut ru = make();
        ru.humanize();
        let ru_id = ru.stable_id();
        let ru_msg = ru.message().to_string();

        Lang::set(Lang::En);
        assert_ne!(
            en_msg, ru_msg,
            "the test needs the two renderings to differ"
        );
        assert_eq!(
            en_id, ru_id,
            "the same kind must keep one stable_id across languages"
        );

        // Manually built diagnostics have no slug, so they still hash their text —
        // two different texts must stay two ids.
        let a = Diagnostic::error("first");
        let b = Diagnostic::error("second");
        assert_ne!(a.stable_id(), b.stable_id());
    }

    /// `joined()` flattens nesting *depth* to 1; it does nothing to bound
    /// *width*.  A comptime diagnostic with a traceback is a **flat** `Joined` of
    /// `8 + 5 * frames` parts, so the four-slot per-part stack walked off its end:
    /// hard `stack[4]` index panic in release, its own debug assert in debug — on a
    /// `pub` accessor whose whole contract is external tool dedup.
    #[test]
    fn stable_id_handles_a_wide_flat_joined_message() {
        let span = crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 4);
        let traceback = vec![
            (
                crate::diagnostics::kind::ComptimeReason::ComptimeBlock,
                None,
            ),
            (
                crate::diagnostics::kind::ComptimeReason::ComptimeFnCall,
                Some(span),
            ),
        ];
        // 4 layout parts + 5 per frame = 14 parts, i.e. well past the old four.
        let make = || {
            let mut d =
                Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::Comptime {
                    kind: crate::diagnostics::kind::ComptimeErrorKind::DivisionByZero,
                    span,
                    traceback: traceback.clone(),
                });
            d.set_file_name("src/wide.pn");
            d
        };

        let diag = make();
        let id = diag.stable_id();
        assert_ne!(
            id, 0,
            "a real diagnostic must not hash to the empty sentinel"
        );
        // The id must be readable twice and equal across clones: a cache-only pass
        // would hide the panic on the second read, not fix it.
        assert_eq!(id, diag.stable_id());
        assert_eq!(id, make().stable_id());

        // Width must be *free*, and nesting depth must still be walked (a hand-built
        // nested `Joined` is constructible through the public enum).
        let wide: Vec<DiagMessage<'_>> = (0..8)
            .map(|i| DiagMessage::Static(std::borrow::Cow::Owned(format!("p{i}"))))
            .collect();
        let nested = DiagMessage::Joined(
            vec![
                DiagMessage::Static(std::borrow::Cow::Borrowed("a")),
                DiagMessage::Joined(wide.into_boxed_slice()),
            ]
            .into_boxed_slice(),
        );
        let mut rendered = String::new();
        nested.render_into(&mut rendered);
        assert!(rendered.contains("p7"), "nested parts must still be walked");
    }

    #[test]
    fn stable_id_and_filter_use_canonical_code() {
        let known = Diagnostic::error("same").with_code(ErrCode::new("e030"));
        let verbatim = Diagnostic::error("same").with_code(ErrCode::unknown("e030"));
        assert_eq!(
            known.code(),
            verbatim.code(),
            "unknown() must echo verbatim"
        );
        assert_eq!(
            known.stable_id(),
            verbatim.stable_id(),
            "stable_id must hash the canonical code, not the verbatim spelling"
        );

        let opts = DiagOpts {
            allow_codes: vec![CompactString::const_new("e030")],
            ..DiagOpts::default()
        };
        assert!(!known.matches_filter(&opts));
        assert!(
            !verbatim.matches_filter(&opts),
            "--allow e030 must suppress unknown(\"e030\")"
        );
    }

    #[test]
    fn test_with_labels() {
        let diag = Diagnostic::error("no field `z` on type Point")
            .with_code_str(ErrCode::E010)
            .with_span(Span::new(FileId::DUMMY, 50, 53))
            .with_label(Span::new(FileId::DUMMY, 0, 30), "type defined here");
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.labels[0].message, "type defined here");
    }

    #[test]
    fn test_error_code_explain() {
        let explain = explain_error_code("E030", true);
        assert!(explain.contains("type mismatch"));
        assert!(explain.contains("E030"));
    }

    #[test]
    fn test_explain_unknown_code() {
        let explain = explain_error_code("E03", true);
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains("E03"));
        assert!(explain.contains("E030")); // did you mean suggestion
        assert!(explain.contains("help"));

        // Far from every known code — falls back to generic help.  The needle is derived
        // at runtime, not hardcoded: `"Z999"` silently depended on the registry containing
        // no `E999`/`W999`, so registering one would have flipped this case onto the
        // did-you-mean branch with no code change here.
        let far = ["Z999", "Z9999", "Q888", "J000", "XJJJ"]
            .into_iter()
            .find(|candidate| !explain_error_code(candidate, true).contains("did you mean"))
            .expect("some probe code must be far from every registered code");
        let explain = explain_error_code(far, true);
        assert!(explain.contains("unknown error code"));
        assert!(explain.contains(far));
        assert!(explain.contains("valid error codes include"));
    }

    /// The unknown code is echoed into the printed explanation, so control
    /// characters from untrusted CLI input must not reach the terminal.
    #[test]
    fn explain_unknown_code_is_sanitized() {
        let out = explain_error_code("E0\x1b]8;;evil\x07", true);
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
    }

    /// Multi-line sanitizer: `\n`/`\t` are layout intent and must survive;
    /// every other control character (ESC, BEL, CR, NEL) becomes U+FFFD.
    #[test]
    fn multiline_sanitizer_keeps_layout_controls_neuters_the_rest() {
        let out = sanitize_for_terminal_multiline("ok\n\tindent\x1b]8;;x\x07mid\r\u{85}");
        assert!(out.contains('\n'), "newline lost: {out:?}");
        assert!(out.contains('\t'), "tab lost: {out:?}");
        assert!(!out.contains('\x1b'), "ESC leaked: {out:?}");
        assert!(!out.contains('\x07'), "BEL leaked: {out:?}");
        assert!(!out.contains('\r'), "CR leaked: {out:?}");
        assert!(!out.contains('\u{85}'), "NEL leaked: {out:?}");
        assert!(
            out.contains('\u{FFFD}'),
            "control char not replaced: {out:?}"
        );
        assert!(matches!(
            sanitize_for_terminal_multiline("plain\n\ttext"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// A genuine C0 control byte sitting in the lane *after* a `\t`/`\n` must
    /// not be swallowed by the SWAR tab/newline exclusion's borrow
    /// propagation: `has_zero`'s mask is not position-exact, so a backspace
    /// adjacent to a tab used to be masked away and the sanitizer returned
    /// the raw control byte borrowed.
    #[test]
    fn multiline_sanitizer_catches_control_adjacent_to_layout_control() {
        for s in [
            "\t\x08abcdef", // backspace right after tab
            "\n\x08abcdef", // backspace right after newline
            "ab\t\x08cd",   // mid-string, chunk-interior
            "\x08\tabcdef", // backspace right *before* tab (borrow source)
            "ab\x7F\tcd",   // DEL adjacent to tab
            "ab\t\x7Fcd",   // tab adjacent to DEL
        ] {
            let out = sanitize_for_terminal_multiline(s);
            assert!(
                matches!(out, std::borrow::Cow::Owned(_)),
                "raw control byte leaked as Borrowed for {s:?}"
            );
            assert!(out.contains('\u{FFFD}'), "control not replaced: {out:?}");
            assert!(!out.contains('\x08') && !out.contains('\x7F'), "{out:?}");
        }
        // Layout controls themselves survive, alone or adjacent to each other.
        for s in ["\t\n", "a\t\nb", "\t\ta\n\n"] {
            assert!(
                matches!(
                    sanitize_for_terminal_multiline(s),
                    std::borrow::Cow::Borrowed(_)
                ),
                "layout controls forced owned for {s:?}"
            );
        }
    }

    /// The non-ASCII no-op fast path must borrow (not allocate) for
    /// localized text with no control characters — the dominant i18n case.
    /// Pure-CJK / Cyrillic / accented text contains no controls, so the
    /// sanitizer must return `Cow::Borrowed`, and the owned path must still
    /// replace only the control characters.
    #[test]
    fn sanitizer_borrows_non_ascii_without_controls() {
        // Borrowed (no allocation) for the i18n hot case.
        for s in [
            "функция main не найдена",
            "crate 中没有找到 main 函数",
            "Datei nicht gefunden",
            "café crème",
            "안녕하세요",
        ] {
            assert!(
                matches!(sanitize_for_terminal(s), std::borrow::Cow::Borrowed(_)),
                "sanitize_for_terminal allocated for {s:?}"
            );
            assert!(
                matches!(
                    sanitize_for_terminal_multiline(s),
                    std::borrow::Cow::Borrowed(_)
                ),
                "sanitize_for_terminal_multiline allocated for {s:?}"
            );
        }
        // A control character in a non-ASCII string still triggers replacement.
        let out = sanitize_for_terminal("中\x1b文");
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert!(out.contains('\u{FFFD}'), "{out:?}");
        assert!(!out.contains('\x1b'), "{out:?}");
        // `\n`/`\t` survive multiline without forcing the owned path to
        // replace them, and other controls become U+FFFD.
        let out = sanitize_for_terminal_multiline("中\n文\tx\x07");
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert!(out.contains('\n') && out.contains('\t'), "{out:?}");
        assert!(!out.contains('\x07'), "{out:?}");
    }

    #[test]
    fn test_suggest_code_osa_distance() {
        // Single substituted digit — same-length typo the old heuristic missed.
        let s = suggest_code("E031", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Missing trailing digit.
        let s = suggest_code("E03", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Lowercase typo still matches the canonical code.
        let s = suggest_code("e030", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Transposed digits: `E300`↔`E030` is OSA-distance 1, so the
        // radius-1 scan must now suggest `E030` (plain Levenshtein scored 2
        // and silently dropped it).  `E300` is not a known code, so this
        // exercises the real unknown-code suggestion path.
        let s = suggest_code("E300", true);
        assert!(s.contains("E030"), "expected E030 suggestion, got: {s}");

        // Complete garbage — no suggestions, fall back to generic help.
        let s = suggest_code("????", true);
        assert!(s.contains("valid error codes include"), "got: {s}");
    }

    #[test]
    fn test_osa_distance() {
        assert_eq!(osa_distance("", ""), 0);
        assert_eq!(osa_distance("abc", "abc"), 0);
        assert_eq!(osa_distance("abc", "abd"), 1);
        assert_eq!(osa_distance("abc", "ab"), 1);
        assert_eq!(osa_distance("abc", "abcd"), 1);
        assert_eq!(osa_distance("abc", ""), 3);
        assert_eq!(osa_distance("", "abc"), 3);
        assert_eq!(osa_distance("kitten", "sitting"), 3);
        assert_eq!(osa_distance("E03", "E030"), 1);
        assert_eq!(osa_distance("E03", "E001"), 2);
        // Adjacent transpositions cost 1 under OSA, 2 under plain Levenshtein.
        // These are the keyboard-typo cases the radius-1 scan used to drop.
        assert_eq!(osa_distance("lne", "len"), 1);
        assert_eq!(osa_distance("E003", "E030"), 1);
    }

    #[test]
    fn test_call_chain() {
        let mut chain = CallChain::new();
        chain.push(chain::ChainEntry::new(
            chain::ChainEntryKind::Call,
            Some(Span::new(FileId::DUMMY, 80, 90)),
            "main",
        ));
        let diag = Diagnostic::error("type mismatch")
            .with_code_str(ErrCode::E030)
            .with_call_chain(chain);
        assert!(diag.call_chain.is_some());
        assert!(diag.to_string().contains("referenced by"));
    }

    #[test]
    fn test_from_string_code() {
        let diag = Diagnostic::error("test").with_code_str(ErrCode::E030);
        assert_eq!(diag.code.unwrap().code(), "E030");
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagCtxt::new();
        collector.push(Diagnostic::error("first error").with_code_str(ErrCode::E030));
        collector.push(Diagnostic::warning("first warning"));
        assert_eq!(collector.len(), 2);
        assert_eq!(collector.summary(), (1, 1));
        assert!(collector.has_errors());
    }

    /// `emit_collector` must advance the same reported/unreported partition as
    /// `DiagCtxt::emit`.
    ///
    /// It used to emit `collector.iter()` — every diagnostic — and never call
    /// `mark_all_reported`, so a caller that mixed the two surfaces (or simply ran
    /// two collections over one context) re-rendered the first batch.  The
    /// observable half of that bug is the frontier, and that is what this pins:
    /// after a successful emit the unreported tail is empty, so a second call has
    /// nothing to re-render.  (Writes to stderr; the assertion is on the partition,
    /// not on captured bytes.)
    #[test]
    fn emit_collector_marks_the_batch_reported() {
        let mut ctxt = DiagCtxt::new();
        ctxt.push(Diagnostic::error("first"));
        ctxt.push(Diagnostic::warning("second"));
        assert_eq!(ctxt.unreported_len(), 2, "nothing is reported yet");

        emit_collector(&mut ctxt, false);
        assert_eq!(
            ctxt.unreported_len(),
            0,
            "a successful emit must advance the reported frontier"
        );

        // Second call is a no-op — the retry-safe property `DiagCtxt::emit` has.
        emit_collector(&mut ctxt, false);
        assert_eq!(ctxt.unreported_len(), 0);
    }

    #[test]
    fn test_multiple_suggestions() {
        let diag = Diagnostic::error("test")
            .with_suggestion("first option")
            .with_suggestion("second option");
        assert_eq!(diag.suggestions.len(), 2);
        let text = diag.to_string();
        assert!(text.contains("first option"));
        assert!(text.contains("second option"));
    }

    #[test]
    fn test_help_text() {
        let diag = Diagnostic::error("test").with_help("try adding a type annotation");
        assert_eq!(diag.help.as_deref(), Some("try adding a type annotation"));
        let text = diag.to_string();
        assert!(text.contains("try adding a type annotation"));
    }

    /// A highlighted note stores only the styled form; the plain text is
    /// derived on demand, so a note that is never rendered never allocates it.
    #[test]
    fn highlighted_note_derives_plain_message_on_demand() {
        let styled = StyledString::new(vec![
            StringPart {
                content: "expected ".into(),
                style: StringPartStyle::Normal,
            },
            StringPart {
                content: "Int<32>".into(),
                style: StringPartStyle::Highlighted,
            },
        ]);
        let diag = Diagnostic::error("mismatch").with_highlighted_note(styled);
        let child = &diag.children()[0];
        assert!(child.message.is_none(), "plain text materialized eagerly");
        assert_eq!(child.plain_message(), "expected Int<32>");
    }

    #[test]
    fn test_error_category() {
        assert_eq!(ErrCode::new(ErrCode::E030).category(), ErrorCategory::Type);
        assert_eq!(ErrCode::new(ErrCode::E001).category(), ErrorCategory::Parse);
        assert_eq!(
            ErrCode::new(ErrCode::E020).category(),
            ErrorCategory::Contract
        );
        assert_eq!(ErrCode::new(ErrCode::E041).category(), ErrorCategory::Trait);
    }

    #[test]
    fn test_fatal_counts_as_error() {
        assert!(Diagnostic::fatal("boom").is_error());
        assert!(Diagnostic::error("boom").is_error());
        assert!(!Diagnostic::warning("boom").is_error());

        let mut ctxt = DiagCtxt::new();
        ctxt.push(Diagnostic::fatal("boom"));
        assert!(ctxt.has_errors());
        assert_eq!(ctxt.error_count(), 1);
        assert!(ctxt.check().is_err());
    }

    #[test]
    fn test_adorn_with_suppressed_warning_does_not_panic() {
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        ctxt.set_can_emit_warnings(false);
        let result: Result<(), EmissionGuaranteed> = adorn_with_span(
            &mut ctxt,
            Span::new(FileId::DUMMY, 0, 5),
            Some("hello"),
            |ctxt| {
                // `push` returns a *suppressed* guarantee here and records
                // nothing, so the old debug_assert would have fired.
                Err(ctxt.push(Diagnostic::warning("suppressed")).into())
            },
        );
        assert!(result.is_err());
        assert!(ctxt.is_empty());
    }

    #[test]
    fn adorn_with_span_survives_humanize() {
        // The `adorn_with_span` + `*_kind` interaction: the adornment
        // attaches the closure's call-site span to a kind-built
        // diagnostic, and the emitter's later `humanize()` must keep it
        // instead of replacing it with kind-derived primary spans.
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        let adorn_span = Span::new(FileId::DUMMY, 10, 15);
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let result: Result<(), EmissionGuaranteed> =
            adorn_with_span(&mut ctxt, adorn_span, Some("src"), |ctxt| {
                Err(ctxt
                    .push(Diagnostic::error("placeholder").with_kind(
                        DiagnosticKind::TypeMismatch {
                            expected: Cow::Borrowed("Int<32>"),
                            found: Cow::Borrowed("Str"),
                            span: kind_span,
                            found_span: None,
                            reason: None,
                            context: None,
                        },
                    ))
                    .into())
            });
        assert!(result.is_err());
        // `humanize_pending` is the collector's boundary step now that the batch
        // emitters take `&Diagnostic` (no `iter_all_mut` mutation channel).
        ctxt.humanize_pending();
        for diag in ctxt.iter() {
            assert_eq!(diag.spans.primary().as_slice(), &[adorn_span][..]);
            // Kind derivation itself is unaffected.
            assert_eq!(diag.labels.len(), 1);
            assert_eq!(diag.labels[0].span, kind_span);
        }
    }

    /// The **generated** [`KNOWN_CODE_RANGES`] must be byte-identical to the runtime
    /// formatter over the same table.
    ///
    /// This is the pin that makes the build-time/runtime duplication safe: the
    /// formatter is the tested oracle (`code_ranges_*` below), the generator is the
    /// shipped value, and drift between them fails here rather than silently
    /// changing the "valid codes are …" hint.
    #[test]
    fn known_code_ranges_match_the_runtime_formatter() {
        let codes: Vec<&str> = crate::diagnostics::error_code::CODE_TABLE
            .iter()
            .map(|e| e.code)
            .collect();
        assert_eq!(known_code_ranges(), format_code_ranges(&codes));
        assert!(!known_code_ranges().is_empty());
        // Sanity: the generated value really is the registry summary, not a stub.
        assert!(known_code_ranges().contains('E') && known_code_ranges().contains('W'));
    }

    #[test]
    fn test_known_code_ranges_cover_table() {
        let ranges = known_code_ranges();
        assert!(ranges.contains("E001"), "got: {ranges}");
        assert!(ranges.contains("W113"), "got: {ranges}");
        assert!(ranges.contains("E601"), "got: {ranges}");
    }

    #[test]
    fn code_ranges_do_not_overflow_at_u32_max() {
        let ranges = format_code_ranges(&["E4294967295", "E004294967295"]);
        assert!(ranges.contains("E4294967295"), "{ranges}");
    }

    #[test]
    fn code_ranges_sort_numerically_within_prefix() {
        // A 4-digit code would sort BEFORE E999 lexicographically; the
        // numeric-aware key keeps numeric order (E1000 after E999), and
        // E999–E1000 merge because they are numerically contiguous.
        let ranges = format_code_ranges(&["E999", "E1000", "E001", "E002", "W113", "W9"]);
        assert_eq!(ranges, "E001–E002, E999–E1000, W9, W113");
    }

    #[test]
    fn err_code_resolves_case_insensitively() {
        // A lowercase code resolves to its canonical table entry, matching
        // `suggest_code`'s case-insensitive comparison — not unknown-code
        // metadata.
        assert_eq!(ErrCode::new("e030").code(), "E030");
        assert_eq!(ErrCode::new("e030").category(), ErrorCategory::Type);
        assert_eq!(ErrCode::try_new("e030").unwrap().code(), "E030");
        // Truly unknown strings stay verbatim (case is never invented).
        assert_eq!(ErrCode::new("e9999").code(), "e9999");
        let err = ErrCode::try_new("zzz").unwrap_err();
        assert_eq!(err.0.as_ref(), "zzz");
    }

    #[test]
    fn test_e067_is_registered() {
        // Regression: `?with_code_str(ErrCode::E067)` in the PANN pattern-annotation
        // path would `debug_assert!`-panic if the code were absent.
        assert!(ErrCode::try_new("E067").is_ok());
    }

    #[test]
    fn test_with_code_str_unknown_does_not_panic() {
        // `with_code_str` is public and may receive untrusted input; it must
        // not hit `ErrCode::new`'s debug assertion.
        let diag = Diagnostic::error("boom").with_code_str("ZZZ999");
        let code = diag.code().expect("code set");
        assert_eq!(code.code(), "ZZZ999");
        assert_eq!(code.title(), "unknown error");
    }

    #[test]
    fn tr_does_not_double_substitute() {
        // A substituted value that itself contains a placeholder must be
        // emitted verbatim, not re-expanded by a later replacement pass.
        let s = crate::tr!("type-mismatch", expected = "{found}", found = "Int<32>");
        assert_eq!(s, "type mismatch: expected `{found}`, found `Int<32>`");
    }

    #[test]
    fn tr_handles_unknown_and_unterminated_placeholders() {
        assert_eq!(
            crate::tr!("no placeholders here", x = "ignored"),
            "no placeholders here"
        );
    }

    #[test]
    fn humanize_after_rekind_refreshes_derived_notes() {
        // A diagnostic can be re-kind'ed (with_kind(k1).humanize() then
        // with_kind(k2).humanize()); k1's derived help must not survive on a
        // diagnostic that is no longer k1.
        let mut diag = Diagnostic::error("placeholder").with_kind(DiagnosticKind::TypeMismatch {
            expected: Cow::Borrowed("Int<32>"),
            found: Cow::Borrowed("Str"),
            span: Span::new(FileId::DUMMY, 0, 3),
            found_span: None,
            reason: None,
            context: None,
        });
        diag.humanize();
        assert_eq!(
            diag.help.as_deref(),
            Some("try using `as` to cast, or change the expression type")
        );
        let mut diag = diag.with_kind(DiagnosticKind::NoSuchField {
            field_name: Cow::Borrowed("z"),
            type_name: Cow::Borrowed("Point"),
            span: Span::new(FileId::DUMMY, 0, 3),
        });
        diag.humanize();
        assert_eq!(diag.help, None);
        assert!(
            diag.message.contains("no field `z` found on type `Point`"),
            "message: {}",
            diag.message
        );
    }

    #[test]
    fn humanize_preserves_user_set_spans() {
        // `with_kind` first, then a builder span: the span is deliberate
        // user context and must survive `humanize()` instead of being
        // replaced by spans derived from the kind's labels.
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let user_span = Span::new(FileId::DUMMY, 40, 45);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: kind_span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_span(user_span);
        diag.humanize();
        // User span kept; the kind-derived span was NOT merged in.
        assert_eq!(diag.spans.primary().as_slice(), &[user_span][..]);
        // The rest of the kind derivation is unchanged.
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(
            diag.help.as_deref(),
            Some("try using `as` to cast, or change the expression type")
        );
    }

    /// `humanize` replaces a constructor-supplied message with the kind-derived
    /// wording.  Pinned because it is a *decision* (see `humanize`'s doc): unlike
    /// spans/help/suggestions there is no `message_user_set` flag, so the kind is
    /// the single owner of the message text.  A regression here would silently
    /// keep a placeholder in user-visible output.
    #[test]
    fn humanize_replaces_a_constructor_message() {
        let mut diag =
            Diagnostic::error("placeholder text").with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: Span::new(FileId::DUMMY, 0, 3),
                found_span: None,
                reason: None,
                context: None,
            });
        diag.humanize();
        assert!(
            diag.message().contains("type mismatch"),
            "kind must own the message: {:?}",
            diag.message()
        );
        assert!(
            !diag.message().contains("placeholder"),
            "constructor text must not survive humanize: {:?}",
            diag.message()
        );
    }

    #[test]
    fn humanize_preserves_user_set_multiple_spans() {
        // `with_span` + `with_additional_span` on a kind-built diagnostic:
        // every user-set primary survives re-derivation.
        let user1 = Span::new(FileId::DUMMY, 40, 45);
        let user2 = Span::new(FileId::DUMMY, 100, 103);
        let mut diag = Diagnostic::error_kind(DiagnosticKind::NoSuchField {
            field_name: Cow::Borrowed("z"),
            type_name: Cow::Borrowed("Point"),
            span: Span::new(FileId::DUMMY, 0, 3),
        })
        .with_span(user1)
        .with_additional_span(user2);
        diag.humanize();
        assert_eq!(diag.spans.primary().as_slice(), &[user1, user2][..]);
    }

    #[test]
    fn humanize_derives_spans_when_not_user_set() {
        // No builder span: humanize must still derive primary spans from
        // the kind's labels (the original behaviour, unchanged).
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let mut diag = Diagnostic::error("placeholder").with_kind(DiagnosticKind::TypeMismatch {
            expected: Cow::Borrowed("Int<32>"),
            found: Cow::Borrowed("Str"),
            span: kind_span,
            found_span: None,
            reason: None,
            context: None,
        });
        diag.humanize();
        assert_eq!(diag.spans.primary().as_slice(), &[kind_span][..]);
    }

    /// `allow_codes` is a *suppress* list (rustc `-A` semantics): a code in
    /// the set is dropped, everything else passes.  Pins the direction of the
    /// filter so the branch can never be "fixed" into the opposite silently.
    #[test]
    fn allow_codes_suppresses_matching_diagnostics() {
        let mut opts = DiagOpts::default();
        opts.allow_codes = vec![CompactString::const_new("E030")];
        let suppressed = Diagnostic::error("x").with_code_str("E030");
        let kept = Diagnostic::error("y").with_code_str("E031");
        let no_code = Diagnostic::error("z");
        assert!(!suppressed.matches_filter(&opts), "E030 must be suppressed");
        assert!(kept.matches_filter(&opts), "E031 must pass");
        assert!(no_code.matches_filter(&opts), "code-less diagnostics pass");
        // Empty list: no-op.
        let unfiltered = DiagOpts::default();
        assert!(suppressed.matches_filter(&unfiltered));
    }

    /// The slot-keyed filter must answer exactly like the string-set version it
    /// replaced: the reference below is the old `canon` + `code_matches` logic,
    /// evaluated over plain `Vec<String>` sets.
    #[test]
    fn slot_keyed_filter_matches_the_string_set_reference() {
        fn canon(pats: &[&str]) -> Vec<String> {
            pats.iter()
                .map(|p| {
                    crate::diagnostics::error_code::lookup_uppercased(p)
                        .map(|e| e.code.to_string())
                        .unwrap_or_else(|| p.to_ascii_uppercase())
                })
                .collect()
        }
        fn str_match(set: &[String], c: &crate::diagnostics::error_code::ErrCode) -> bool {
            let canonical = c.canonical();
            if set.iter().any(|s| s == canonical) {
                return true;
            }
            if canonical.bytes().all(|b| !b.is_ascii_lowercase()) {
                return false;
            }
            set.iter().any(|s| *s == canonical.to_ascii_uppercase())
        }

        // Built at runtime: `literal_codes_are_all_in_code_table` scans this
        // source for code literals and correctly rejects an unregistered one, so
        // the test cannot spell it out.
        let unregistered = ["E", "999"].concat();
        let diags = [
            Diagnostic::error("a").with_code_str("E030"),
            Diagnostic::error("b").with_code_str("e030"),
            Diagnostic::error("c").with_code_str(&unregistered),
            Diagnostic::error("d").with_code_str("w001"),
            Diagnostic::error("e"),
        ];
        let cases: [&[&str]; 5] = [
            &[],
            &["e030"],
            &["E999"],
            &["e030", "E999"],
            &["w001", "E0300"],
        ];
        for allow in cases {
            for only in cases {
                let mut opts = DiagOpts::default();
                opts.allow_codes = allow.iter().map(|s| CompactString::from(*s)).collect();
                opts.only_codes = only.iter().map(|s| CompactString::from(*s)).collect();
                let filter = DiagFilter::from_opts(&opts);
                let (allow_c, only_c) = (canon(allow), canon(only));
                for d in &diags {
                    let reference = match d.code() {
                        Some(c) if str_match(&allow_c, c) => false,
                        Some(c) => only_c.is_empty() || str_match(&only_c, c),
                        None => only_c.is_empty(),
                    };
                    assert_eq!(
                        filter.matches(d),
                        reference,
                        "allow {allow:?} only {only:?} diag {:?}",
                        d.message()
                    );
                }
            }
        }
    }

    /// Filter codes resolve case-insensitively, like every other code API:
    /// `--allow e030` must suppress the canonical `E030` diagnostic instead
    /// of silently matching nothing.
    /// An empty `--only-files` pattern is dropped, not admitted: `memmem`'s
    /// empty needle matches at position 0 of *every* file name, so admitting it
    /// would leak every diagnostic through the filter while `is_empty` reported
    /// the filter active (ANDing a match-everything term against the others).
    #[test]
    fn empty_only_files_pattern_is_dropped() {
        let mut opts = DiagOpts::default();
        opts.only_files = vec![CompactString::const_new("")];
        let filter = DiagFilter::from_opts(&opts);
        assert!(filter.is_empty, "an empty pattern must not arm the filter");
        assert!(filter.matches(&Diagnostic::error("x").with_file_name("src/lib.pn")));

        opts.only_files = vec![CompactString::const_new("lib")];
        let filter = DiagFilter::from_opts(&opts);
        assert!(!filter.is_empty);
        assert!(filter.matches(&Diagnostic::error("x").with_file_name("src/lib.pn")));
        assert!(!filter.matches(&Diagnostic::error("x").with_file_name("src/main.pn")));
    }

    /// A junk code's category falls back to the `Generic` registry entry, so
    /// `only_categories=[Generic]` *does* select it.  That is a decision (a
    /// filter on "generic" should not silently skip the codes the compiler could
    /// not classify), not an accident — pin it, since the `only_categories` doc
    /// only discusses code-less diagnostics.
    #[test]
    fn generic_category_filter_selects_unresolvable_codes() {
        use crate::diagnostics::error_code::ErrorCategory;
        let mut opts = DiagOpts::default();
        opts.only_categories = vec![ErrorCategory::Generic];
        let filter = DiagFilter::from_opts(&opts);
        let junk = Diagnostic::error("x").with_code_str("zz");
        assert_eq!(
            junk.code().map(ErrCode::category),
            Some(ErrorCategory::Generic),
            "an unresolvable code falls back to the Generic registry entry"
        );
        assert!(filter.matches(&junk));

        // A code in a different category is still rejected, so this is a filter
        // and not a match-everything fallback.
        let parse = Diagnostic::error("y").with_code_str("e030");
        assert_ne!(
            parse.code().map(ErrCode::category),
            Some(ErrorCategory::Generic)
        );
        assert!(!filter.matches(&parse));
    }

    /// The help/suggestion half had no pin — only spans did — so the guards that
    /// preserve user-supplied help could have been "simplified" away silently.
    #[test]
    fn humanize_preserves_user_set_help_and_suggestions() {
        let span = Span::new(FileId::DUMMY, 0, 3);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_help("user help")
            .with_span_suggestion(
                span,
                "user replacement",
                "user suggestion",
                Applicability::MachineApplicable,
            );
        diag.humanize();
        assert_eq!(
            diag.help(),
            Some("user help"),
            "user help must survive humanize"
        );
        assert_eq!(diag.suggestions().len(), 1, "user suggestion must survive");
        assert_eq!(
            diag.suggestions()[0].replacement.as_deref(),
            Some("user replacement")
        );
    }

    /// `stable_id` must hash the **sorted-first** span.  `MultiSpan` sorts
    /// lazily, so a diagnostic built with reverse-ordered primary spans must hash
    /// identically to the same diagnostic built in ascending order — otherwise the
    /// dedup key depends on builder order.
    #[test]
    fn stable_id_hashes_the_sorted_first_span() {
        let lo = Span::new(FileId::DUMMY, 0, 3);
        let hi = Span::new(FileId::DUMMY, 20, 25);
        let ascending = Diagnostic::error("boom")
            .with_span(lo)
            .with_additional_span(hi);
        let descending = Diagnostic::error("boom")
            .with_span(hi)
            .with_additional_span(lo);
        assert_eq!(ascending.stable_id(), descending.stable_id());
    }

    /// The `only_categories` doc's headline claim — a code-less diagnostic is
    /// excluded, because a category is a property of the code — had no test.
    #[test]
    fn only_categories_excludes_code_less_diagnostics() {
        let mut opts = DiagOpts::default();
        opts.only_categories = vec![ErrorCategory::Type];
        let filter = DiagFilter::from_opts(&opts);
        assert!(
            !filter.matches(&Diagnostic::error("no code")),
            "a code-less diagnostic must not pass a category filter"
        );
        assert!(filter.matches(&Diagnostic::error("typed").with_code_str(ErrCode::E030)));
    }

    /// Codes in a filter resolve case-insensitively, like every other code API:
    /// `--allow e030` suppresses a `Known` `E030` *and* an `ErrCode::unknown`
    /// spelling, because the filter compares canonical forms.
    #[test]
    fn filter_codes_resolve_case_insensitively() {
        let mut opts = DiagOpts::default();
        opts.allow_codes = vec![CompactString::const_new("e030")];
        let suppressed = Diagnostic::error("x").with_code_str("E030");
        assert!(
            !suppressed.matches_filter(&opts),
            "lowercase allow pattern must suppress E030"
        );

        let mut only = DiagOpts::default();
        only.only_codes = vec![CompactString::const_new("e031")];
        assert!(
            Diagnostic::error("x")
                .with_code_str("E031")
                .matches_filter(&only),
            "lowercase only-code must match E031"
        );
        assert!(
            !Diagnostic::error("x")
                .with_code_str("E030")
                .matches_filter(&only),
            "only-code still excludes other codes"
        );
    }

    /// Regression: the SWAR newline mask is a *candidate* set, not exact — a
    /// `\n` followed by `0x0B` (VT) makes the borrow out of the zero lane set
    /// the next lane's high bit, and the old code pushed a fake line start.
    /// Line numbers after that point were off by one.
    #[test]
    fn line_scan_does_not_invent_newlines_after_vt() {
        // Lay the poison pattern across every 8-byte lane alignment.
        let mut src = String::new();
        for pad in 0..8 {
            src.push_str(&"x".repeat(pad));
            src.push_str("a\n\u{0B}b\n");
        }
        let idx = LineIndex::new(&src);
        // A `0x0B` is valid UTF-8 but never a line break: the only newlines
        // in `src` are the two per repetition.
        let mut expected = vec![0u32];
        let mut off = 0usize;
        for pad in 0..8 {
            // "xxx…a\n" — the `\n` after `a` sits at `off + pad + 1`, so the
            // poisoned line (`\x0Bb`) starts at `off + pad + 2`.
            expected.push((off + pad + 2) as u32);
            // "\x0Bb\n" — the `\n` after `b` sits at `off + pad + 4`, so the
            // next repetition's line starts at `off + pad + 5`.  The `0x0B`
            // itself must NOT have produced a start.
            off += pad + 5;
            expected.push(off as u32);
        }
        // Exactly 2 starts per repetition + the initial 0: a `0x0B` never
        // contributes a line start.
        assert_eq!(expected.len(), 1 + 2 * 8);
        assert_eq!(idx.starts(), expected.as_slice(), "src = {src:?}");
        // And the mapping round-trips for every line.
        for i in 0..idx.line_count() {
            let pos = idx.to_linecol(idx.starts()[i] as usize);
            assert_eq!((pos.line, pos.col), (i, 0));
        }
    }

    /// The AVX2 chunk loop must produce exactly the SWAR loop's line table, ASCII flag
    /// and fused content hash.
    ///
    /// The AVX2 masks are "exact by construction" (a byte-equality compare and the sign
    /// bits), but the *loop shape* differs — 32-byte chunks with a 32-bit lane mask
    /// instead of 8-byte chunks with a 64-bit one — so the lane iteration, the chunk
    /// boundary and the tail handoff all need a differential rather than an argument.
    /// `FORCE_SWAR_SCAN` exists only for this comparison.
    ///
    /// The corpus deliberately mixes the two known traps: `0x8A` (a UTF-8 continuation
    /// byte that SWAR's subtraction sees as a newline candidate), lead/continuation runs,
    /// newlines at every offset modulo 32, and lengths on every side of the 8/32-byte
    /// boundaries so the tail paths are exercised too.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_line_scan_matches_the_swar_scan() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // the fast path does not exist here
        }
        let mut corpus: Vec<String> = Vec::new();
        corpus.push(String::new());
        for n in 0..=40usize {
            corpus.push("a".repeat(n));
            corpus.push("\n".repeat(n));
            corpus.push("a\n".repeat(n));
            // `Ċ` is 0xC4 0x8A: the continuation byte is the SWAR trap.
            corpus.push("Ċ".repeat(n));
            corpus.push("Ċ\n中".repeat(n));
            corpus.push("中".repeat(n));
        }
        // Newlines swept across every modulo-32 offset inside a longer line.
        for shift in 0..32usize {
            corpus.push(format!(
                "{}x\n{}",
                "a".repeat(shift),
                "中".repeat(shift % 7)
            ));
        }
        for src in &corpus {
            std::sync::atomic::Ordering::Relaxed;
            FORCE_SWAR_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
            let (starts_swar, ascii_swar, hash_swar, tables_swar) =
                scan_line_starts_and_ascii_capped(src, usize::MAX);
            FORCE_SWAR_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);
            let (starts_avx, ascii_avx, hash_avx, tables_avx) =
                scan_line_starts_and_ascii_capped(src, usize::MAX);
            assert_eq!(starts_swar, starts_avx, "line starts differ for {src:?}");
            assert_eq!(ascii_swar, ascii_avx, "ascii flag differs for {src:?}");
            assert_eq!(hash_swar, hash_avx, "content hash differs for {src:?}");
            assert_eq!(
                tables_swar.len(),
                tables_avx.len(),
                "table slot count differs for {src:?}"
            );
            for line in 0..tables_swar.len() {
                assert_eq!(
                    tables_swar.get(line).map(<[(u32, u32)]>::to_vec),
                    tables_avx.get(line).map(<[(u32, u32)]>::to_vec),
                    "char table differs on line {line} of {src:?}"
                );
            }
        }
        // The **gate** path.  With `usize::MAX` the prefix scan never truncates, so the
        // AVX2 marks loop's gate bookkeeping (`cur.clear()`, per-line offset attribution
        // after the frontier) was only ever compared on whichever kernel the host takes —
        // i.e. never, in both directions.  A small cap over the CJK members is exactly
        // where a `line_start`-relative offset mistake would corrupt every column past the
        // frontier: wrong columns, no crash.
        for cap in [3usize, 5, 7] {
            for src in corpus.iter().filter(|s| !s.is_ascii()) {
                FORCE_SWAR_SCAN.store(true, std::sync::atomic::Ordering::Relaxed);
                let (starts_swar, ascii_swar, hash_swar, tables_swar) =
                    scan_line_starts_and_ascii_capped(src, cap);
                FORCE_SWAR_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);
                let (starts_avx, ascii_avx, hash_avx, tables_avx) =
                    scan_line_starts_and_ascii_capped(src, cap);
                assert_eq!(
                    starts_swar, starts_avx,
                    "gated line starts differ (cap {cap}) for {src:?}"
                );
                assert_eq!(
                    ascii_swar, ascii_avx,
                    "gated ascii flag differs (cap {cap}) for {src:?}"
                );
                assert_eq!(
                    hash_swar, hash_avx,
                    "gated content hash differs (cap {cap}) for {src:?}"
                );
                assert_eq!(tables_swar.len(), tables_avx.len());
                for line in 0..tables_swar.len() {
                    assert_eq!(
                        tables_swar.get(line).map(<[(u32, u32)]>::to_vec),
                        tables_avx.get(line).map(<[(u32, u32)]>::to_vec),
                        "gated char table differs on line {line} of {src:?} (cap {cap})"
                    );
                }
            }
        }
        FORCE_SWAR_SCAN.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn line_scan_ignores_utf8_continuation_0x8a() {
        // `Ċ` is C4 8A and `ꀀ` is F0 8A 80 80: `q = x ^ 0x0A` is 0x80 for the
        // continuation byte, which the old `(q | HI)` mask mistook for `\n`.
        // Lay the pattern across every 8-byte alignment and compare the whole
        // table with a scalar newline scan.
        let mut src = String::new();
        for pad in 0..8 {
            src.push_str(&"x".repeat(pad));
            src.push('Ċ');
            src.push('\n');
            src.push('ꀀ');
            src.push_str("z\n");
        }
        let idx = LineIndex::new(&src);
        let expected: Vec<u32> = std::iter::once(0)
            .chain(
                src.bytes()
                    .enumerate()
                    .filter(|(_, byte)| *byte == b'\n')
                    .map(|(i, _)| (i + 1) as u32),
            )
            .collect();
        assert_eq!(idx.starts(), expected.as_slice(), "src = {src:?}");
        for i in 0..idx.line_count() {
            let pos = idx.to_linecol(idx.starts()[i] as usize);
            assert_eq!((pos.line, pos.col), (i, 0));
        }
    }

    /// Boundary matrix for `LineIndex`: empty input, single byte, with and
    /// without trailing newlines, multi-byte leading char, and CRLF. All
    /// conversions (including EOF and past-EOF offsets) must stay panic-free
    /// and self-consistent: every line start maps back to (line, 0), and
    /// `to_linecol` never produces an out-of-range line.
    #[test]
    fn line_index_boundaries_stay_consistent() {
        let sources = ["", "a", "a\n", "a\n\n", "中文\n", "\t\n", "abc\r\ndef\r\n"];
        for src in sources {
            let idx = LineIndex::new(src);
            let eof = idx.to_linecol(src.len());
            assert!(
                eof.line < idx.line_count(),
                "eof line {} out of range for {src:?}",
                eof.line
            );
            let past = idx.to_linecol(usize::MAX);
            assert!(past.line < idx.line_count());
            let eof_span = Span::new(FileId::DUMMY, src.len() as u32, src.len() as u32);
            let (first, last) = idx.span_lines(eof_span);
            assert!(first <= last && last < idx.line_count(), "span {src:?}");
            let over = Span::new(FileId::DUMMY, 0, (src.len() as u32).saturating_add(8));
            let (first, last) = idx.span_lines(over);
            assert!(first <= last && last < idx.line_count(), "over {src:?}");
            for i in 0..idx.line_count() {
                let pos = idx.to_linecol(idx.starts()[i] as usize);
                assert_eq!((pos.line, pos.col), (i, 0), "start of line {i} in {src:?}");
            }
        }
        // Known values: trailing newline does not create a phantom text line.
        let idx = LineIndex::new("a\n");
        assert_eq!(idx.text_line_count(), 1);
        assert_eq!(idx.line(0), "a");
        let idx = LineIndex::new("a\n\n");
        assert_eq!(idx.text_line_count(), 2);
        assert_eq!(idx.line(1), "");
        let idx = LineIndex::new("中文\n");
        assert_eq!(idx.line(0), "中文");
        let pos = idx.to_linecol("中文".len());
        assert_eq!((pos.line, pos.col), (0, "中文".len()));
    }

    /// The high-byte gate must degrade **partially**: tables built before the gate
    /// are kept, and only the frontier onward takes `char_col`'s on-demand build.
    ///
    /// The previous shape reset `prefixes` to empty on truncation — which (a) threw
    /// away tables already inside the budget and (b) made a `None` slot ambiguous
    /// between "proven ASCII" and "unknown", so every later render of that file
    /// silently reverted to a per-query build for *all* lines.
    #[test]
    fn prefix_gate_truncates_partially_and_columns_stay_exact() {
        // Line 0 = 6 high bytes, closed before the budget of 7 runs out inside line 2.
        let src = "中文\nascii\n中文四\nascii\n";
        let (starts, ascii, _hash, tables) = super::scan_line_starts_and_ascii_capped(src, 7);
        assert!(!ascii);
        assert_eq!(
            tables.len(),
            starts.len(),
            "one slot per line even past the gate"
        );
        assert!(tables.is_trusted(0) && tables.is_trusted(1));
        assert!(
            !tables.is_trusted(2) && !tables.is_trusted(3),
            "the frontier sits at the line the gate fired in"
        );
        assert!(
            tables.get(0).is_some(),
            "line 0's table was built inside the budget and must survive the gate"
        );
        assert!(tables.get(1).is_none(), "line 1 is proven ASCII");
        // Columns stay exact on both sides of the frontier — the trusted table and
        // the on-demand fallback are the same rule.
        let idx = super::LineIndex::from_shared(
            src,
            SharedSourceTables::from_cache(starts.into(), ascii, tables, 0),
        );
        let lines = idx.starts.len();
        for line_idx in 0..lines {
            let line = idx.line(line_idx);
            let boundaries: Vec<usize> = line
                .char_indices()
                .map(|(b, _)| b)
                .chain([line.len()])
                .collect();
            for (want_col, &b) in boundaries.iter().enumerate() {
                assert_eq!(
                    idx.char_col(line_idx, b),
                    want_col,
                    "line {line_idx} byte {b} of {src:?}"
                );
            }
        }
    }

    /// The scan-built tables must equal the on-demand builder, line by line.
    ///
    /// This is the correctness pin for fusing the prefix build into
    /// `scan_line_starts_and_ascii`: the fast path (prebuilt, no lock) and the
    /// fallback (`build_line_sparse`, used when the high-byte gate fires) are two
    /// implementations of one rule, so they must agree byte for byte — and both
    /// must agree with the character count, which is the property renderers use.
    #[test]
    fn prebuilt_char_prefix_tables_match_the_on_demand_builder() {
        let corpus = [
            "pure ascii\nsecond line\n",
            "中文注释\nascii line\n混合 CJK and ascii\n",
            "é\n\u{1F600}\u{1F600}\n",
            "\n\n\n",
            "no trailing newline 中文",
            "a\tb\t中\n",
        ];
        for src in corpus {
            let idx = LineIndex::new(src);
            assert_eq!(
                idx.char_prefixes_arc().len(),
                idx.starts.len(),
                "one slot per line for {src:?}"
            );
            for line_idx in 0..idx.starts.len() {
                let line = idx.line(line_idx);
                let want = build_line_sparse(line);
                match idx.char_prefixes_arc().get(line_idx) {
                    Some(t) => assert_eq!(
                        t,
                        want.as_slice(),
                        "line {line_idx} of {src:?} disagrees with the builder"
                    ),
                    None => assert!(
                        want.is_empty(),
                        "line {line_idx} of {src:?} was cached as ASCII but has non-ASCII bytes"
                    ),
                }
                // Every char boundary maps to its character index, both through
                // the table and through the on-demand walk.
                let chars: Vec<usize> = line
                    .char_indices()
                    .map(|(b, _)| b)
                    .chain([line.len()])
                    .collect();
                for (want_col, &b) in chars.iter().enumerate() {
                    assert_eq!(
                        idx.char_col(line_idx, b),
                        want_col,
                        "{src:?} line {line_idx} b={b}"
                    );
                }
            }
        }
    }

    /// `linecol_display` returns *character* columns on non-ASCII sources,
    /// backed by the lazily-built per-line char-count prefix cache.  This
    /// pins the cache's correctness (dense prefix build + lookup), the
    /// ASCII-line-in-a-non-ASCII-source fast path, reuse across repeated
    /// queries on the same line, and sharing across indexes built from the
    /// same cached source via [`LineIndex::from_shared`].
    #[test]
    fn linecol_display_char_columns_via_prefix_cache() {
        // Line 0: two 3-byte chars "中" "文"; line 1: ASCII "ab"; line 2:
        // a 4-byte char "𝄞" (U+1D11E) then ASCII.
        let src = "中文\nab\n𝄞x\n";
        let idx = LineIndex::new(src);
        assert!(!idx.ascii());
        // Byte layout: 中=0..3, 文=3..6, \n=6, a=7, b=8, \n=9, 𝄞=10..14
        // (4 bytes), x=14, \n=15.  Char boundaries only — `linecol_display`
        // floors non-boundary offsets, so test the boundaries directly.
        assert_eq!(idx.linecol_display(0), (0, 0));
        assert_eq!(idx.linecol_display(3), (0, 1));
        assert_eq!(idx.linecol_display(6), (0, 2));
        // Line 1 is ASCII inside a non-ASCII source: char col == byte col,
        // exercised through the per-line `is_ascii` fast path in `char_col`.
        assert_eq!(idx.linecol_display(7), (1, 0));
        assert_eq!(idx.linecol_display(8), (1, 1));
        assert_eq!(idx.linecol_display(9), (1, 2));
        // Line 2: "𝄞" is 4 bytes, "x" follows at byte 14.
        assert_eq!(idx.linecol_display(10), (2, 0));
        assert_eq!(idx.linecol_display(14), (2, 1));
        assert_eq!(idx.linecol_display(15), (2, 2));

        // Repeated queries on the same line hit the cached prefix (the build
        // runs once); the answers must be stable across calls.
        assert_eq!(idx.linecol_display(3), (0, 1));
        assert_eq!(idx.linecol_display(14), (2, 1));

        // A second index sharing the cache (the RenderCtx hit path) reuses
        // the prefixes the first index built: build on `idx`, then verify
        // `shared` agrees without rebuilding.  Correctness is observable via
        // the answers; sharing via `from_shared` is what makes a batch of M
        // diagnostics over one long non-ASCII line pay one build, not M.
        idx.linecol_display(3);
        // Built from the index itself: the whole point of the bundle is that these four
        // fields cannot be assembled by hand (see `SharedSourceTables`).
        let shared = LineIndex::from_shared(src, SharedSourceTables::of(&idx));
        assert_eq!(shared.linecol_display(3), (0, 1));
        assert_eq!(shared.linecol_display(14), (2, 1));
    }

    /// A non-empty span's end is exclusive: an end sitting exactly on a line
    /// start belongs to the previous line (the line holding the last covered
    /// byte), so renderers never attribute a label to a line they cannot
    /// draw it on. Zero-width spans have no covered byte and keep the
    /// position's own line.
    /// `line_of_from` must be exact for *any* hint (ascending, descending, past
    /// the answer), because `span_lines` feeds it the start line and a producer
    /// may emit spans in any order.
    #[test]
    fn hinted_line_lookup_matches_binary_search_for_every_hint() {
        let src = "aaa\nbb\n\ncccc\nd\n";
        let idx = LineIndex::new(src);
        let lines = idx.line_count();
        for off in 0..=src.len() {
            let want = idx.line_of(off);
            for hint in 0..=lines {
                assert_eq!(
                    idx.line_of_from(off, hint),
                    want,
                    "offset {off} hint {hint}"
                );
            }
            // A hint past the end must not escape the table either.
            assert_eq!(idx.line_of_from(off, usize::MAX), want, "offset {off}");
        }
    }

    #[test]
    fn span_lines_end_on_line_start_belongs_to_previous_line() {
        let idx = LineIndex::new("aaa\nbbb\n");
        // End exactly on the start of line 1: the last covered byte ('a'
        // at 2) is on line 0.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 0, 4)), (0, 0));
        // Zero-width at the same boundary keeps the following line.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 4, 4)), (1, 1));
        // Span covering a whole line plus its newline.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 0, 8)), (0, 1));
        // End inside a later line: attribution unchanged.
        assert_eq!(idx.span_lines(Span::new(FileId::DUMMY, 1, 5)), (0, 1));
    }

    // ── Renderer timing harness (benchmark-first, no speed asserts) ──
    // Run: `cargo test perf_ -- --nocapture`. Plain `#[test]`s (not
    // criterion) because the crate is binary-only, so an external
    // `benches/` target cannot import the renderer. The `[perf]` lines
    // are the before/after comparison for the renderer optimizations.

    fn perf_report(name: &str, samples_ms: &mut [f64], out_len: usize) {
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples_ms[samples_ms.len() / 2];
        let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
        println!(
            "[perf] {name}: median {median:.2} ms, mean {mean:.2} ms over {} iters (output {out_len} bytes)",
            samples_ms.len()
        );
    }

    /// (a) Very long single-line file: stresses O(n) offset rescans per
    /// conversion plus the per-label grapheme re-walk in
    /// `byte_range_to_display`.
    #[test]
    #[ignore = "perf harness: cargo test perf_ -- --ignored --nocapture"]
    fn perf_long_single_line() {
        let renderer = GlyphRenderer::new(false);
        let source = "x".repeat(500_000);
        let mut diag = Diagnostic::error("long line boom")
            .with_span(Span::new(FileId::DUMMY, 499_000, 499_010))
            .with_source(&source)
            .with_file_name("<input>");
        for i in 0..20 {
            let s = 480_000 + i * 500;
            diag = diag
                .with_secondary_label(Span::new(FileId::DUMMY, s, s + 10), format!("label {i}"));
        }
        for _ in 0..2 {
            let _ = std::hint::black_box(renderer.render_diagnostic(&diag));
        }
        let mut samples = Vec::with_capacity(5);
        let mut len = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let out = renderer.render_diagnostic(&diag);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            len = std::hint::black_box(out).len();
        }
        perf_report("long_single_line(500KB,20 labels)", &mut samples, len);
    }

    /// (b) Many labels + primary spans on a multi-line file: stresses
    /// per-span section recomputation, the per-row label scan, and the
    /// message-string multiline matching.
    #[test]
    #[ignore = "perf harness: cargo test perf_ -- --ignored --nocapture"]
    fn perf_many_labels_spans() {
        let renderer = GlyphRenderer::new(false);
        let mut source = String::new();
        let mut line_off: Vec<u32> = Vec::with_capacity(5_000);
        for i in 0..5_000 {
            line_off.push(u32::try_from(source.len()).unwrap_or(u32::MAX));
            source.push_str(&format!("let v{i} = {i} + {i};\n"));
        }
        let mut diag = Diagnostic::error("many labels")
            .with_span(Span::new(FileId::DUMMY, line_off[100], line_off[100] + 5))
            .with_source(&source)
            .with_file_name("<input>");
        for k in 1..8 {
            let l = k * 600;
            diag =
                diag.with_additional_span(Span::new(FileId::DUMMY, line_off[l], line_off[l] + 5));
        }
        for i in 0..200 {
            let l = (i * 25) % 5_000;
            diag = diag.with_secondary_label(
                Span::new(FileId::DUMMY, line_off[l], line_off[l] + 4),
                format!("note {i}"),
            );
        }
        for _ in 0..1 {
            let _ = std::hint::black_box(renderer.render_diagnostic(&diag));
        }
        let mut samples = Vec::with_capacity(5);
        let mut len = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let out = renderer.render_diagnostic(&diag);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            len = std::hint::black_box(out).len();
        }
        perf_report(
            "many_labels(5k lines,8 spans,200 labels)",
            &mut samples,
            len,
        );
    }

    /// Multi-diagnostic batch over one large file: without a shared index,
    /// every diagnostic rescans the whole file (`render_diagnostic`); with
    /// the per-batch [`RenderCtx`] cache the scan happens once.  A/B in one
    /// run, no asserts (benchmark-first harness).
    #[test]
    #[ignore = "perf harness: cargo test perf_ -- --ignored --nocapture"]
    fn perf_many_diags_one_source() {
        let source = "let x = 1;\n".repeat(20_000); // ~220 KB
        let diags: Vec<Diagnostic<'_>> = (0..200)
            .map(|i| {
                let off = u32::try_from((i * 997) % (source.len() - 2)).unwrap_or(u32::MAX);
                Diagnostic::error("boom")
                    .with_span(Span::new(FileId::DUMMY, off, off.saturating_add(1)))
                    .with_source(&source)
                    .with_file_name("<input>")
            })
            .collect();
        let renderer = GlyphRenderer::new(false);

        // Uncached: a fresh index per diagnostic (the old behaviour).
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for d in &diags {
                let _ = std::hint::black_box(renderer.render_diagnostic(d));
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        perf_report("many_diags_uncached(220KB,200 diags)", &mut samples, 0);

        // Cached: one index per batch, shared by every diagnostic.
        let opts = DiagOpts::default();
        let cx = RenderCtx::new(&opts, None);
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for d in &diags {
                let idx = cx.line_index(d.source().expect("source attached"));
                let _ = std::hint::black_box(renderer.render_diagnostic_with_index(d, Some(&idx)));
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        perf_report("many_diags_cached(220KB,200 diags)", &mut samples, 0);
    }

    /// Isolation: is the section header's O(prefix) column walk worth a
    /// per-`RenderCtx` [`LineColumns`] cache?
    ///
    /// Each rendered diagnostic asks `byte_offset_to_display` for a walk over the
    /// line prefix (header, plus once per note locator).  On a long line that is
    /// the same O(prefix) pass repeated per diagnostic, i.e. exactly what a
    /// `(line, tab_width)`-keyed table amortises to one walk per line.  This
    /// measures the walk alone, so its share of a render is visible without being
    /// buried in the 200 KB line copy that every long-line render also pays.
    #[test]
    #[ignore = "perf probe (~70s in debug): run with --ignored --nocapture"]
    fn perf_header_column_walk_on_long_line() {
        let line = "x".repeat(200_000);
        let offs: Vec<usize> = (0..200).map(|i| 150_000 + i * 3).collect();
        for _ in 0..2 {
            for &o in &offs {
                let _ = std::hint::black_box(super::glyph::byte_offset_to_display(&line, o, 4));
            }
        }
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for &o in &offs {
                let _ = std::hint::black_box(super::glyph::byte_offset_to_display(&line, o, 4));
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        perf_report(
            "byte_offset_to_display x200 (200KB line, offset 150KB)",
            &mut samples,
            0,
        );
    }

    /// Micro-isolation of indexed conversions near the end of
    /// a 220KB file. Index build is O(n) once; each lookup is O(log n).
    #[test]
    #[ignore = "perf harness: cargo test perf_ -- --ignored --nocapture"]
    fn perf_byte_to_linecol() {
        let source = "let x = 1;\n".repeat(20_000);
        let offs: Vec<usize> = (0..200).map(|i| source.len() - 1 - i * 100).collect();
        let idx = LineIndex::new(&source);
        for _ in 0..2 {
            for &o in &offs {
                std::hint::black_box(idx.to_linecol(o));
            }
        }
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for &o in &offs {
                std::hint::black_box(idx.to_linecol(o));
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("[perf] LineIndex.to_linecol x200 on 220KB: {ms:.2} ms/iter");
    }

    /// Micro-isolation of the wrap fast paths: (1) short ASCII messages hit
    /// the `len <= max_width` guard without a grapheme walk; (2) long
    /// printable-ASCII prose exercises `ascii_break_opportunities`'s `memchr`
    /// space scan.  Run: `cargo test perf_wrap_optimal -- --nocapture`.
    #[test]
    #[ignore = "perf harness: cargo test perf_ -- --ignored --nocapture"]
    fn perf_wrap_optimal() {
        // (1) Short ASCII: the len guard makes this ~O(1) per call.
        let short_msgs: Vec<String> = (0..200)
            .map(|i| format!("error code E{i:03} in module mod_{i}"))
            .collect();
        for _ in 0..2 {
            for m in &short_msgs {
                std::hint::black_box(wrap::wrap_optimal(m, 100, 4));
            }
        }
        let iters = 1000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for m in &short_msgs {
                std::hint::black_box(wrap::wrap_optimal(m, 100, 4));
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("[perf] wrap_optimal short ASCII x200: {ms:.4} ms/iter");

        // (2) Long prose: ~80 words repeated, wraps to several lines,
        // exercises the ascii_break_opportunities memchr space scan.
        let long_prose = "the quick brown fox jumps over the lazy dog and then \
            runs through the forest looking for a place to sleep but finds only \
            rocks and trees and a small stream that flows down the mountain side \
            into a valley where the wind blows cold and the leaves rustle loudly \
            while a distant owl hoots twice before falling silent again forever"
            .repeat(4);
        for _ in 0..2 {
            std::hint::black_box(wrap::wrap_optimal(&long_prose, 60, 4));
        }
        let iters = 2000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wrap::wrap_optimal(&long_prose, 60, 4));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "[perf] wrap_optimal long prose ({} bytes): {ms:.4} ms/iter",
            long_prose.len()
        );
    }

    #[test]
    fn multispan_push_dedup_and_sort() {
        let f1 = FileId::from_raw(1);
        let f2 = FileId::from_raw(2);
        let s_a = Span::new(f1, 10, 20);
        let s_b = Span::new(f1, 30, 40);
        let s_c = Span::new(f2, 0, 5);
        let s_d = Span::new(f1, 0, 4);

        let mut ms = MultiSpan::new(s_b);
        ms.push(s_a);
        ms.push(s_c);
        ms.push(s_d);
        ms.push(s_a);
        ms.push(s_b);
        ms.push(s_c);

        assert_eq!(ms.primary().as_slice(), &[s_d, s_a, s_b, s_c]);
    }

    /// `MultiSpan::push` keeps the sorted+dedup invariant for interleaved
    /// traffic: descending pushes (binary-insert path), a duplicate landing
    /// in the middle, and an insertion between two existing entries.
    #[test]
    fn multispan_push_reverse_middle_and_dup() {
        let f1 = FileId::from_raw(1);
        let s_a = Span::new(f1, 10, 20);
        let s_mid = Span::new(f1, 20, 25);
        let s_b = Span::new(f1, 30, 40);
        let s_c = Span::new(f1, 50, 60);

        let mut ms = MultiSpan::new(s_c);
        // `push` appends and marks the vector dirty; the deferred sort+dedup on
        // the next read is what produces the observable order.  (The old
        // comments here described a binary-insert implementation.)
        ms.push(s_b); // out of order → appended, sort deferred
        ms.push(s_a);
        ms.push(s_b); // duplicate → merged by the deferred dedup
        ms.push(s_mid);
        ms.push(s_c); // duplicate of the tail-most element (pos 0 after sort)

        assert_eq!(ms.primary().as_slice(), &[s_a, s_mid, s_b, s_c]);
    }

    /// Structural guard for the attribute-miswiring family.
    ///
    /// A test function whose `#[test]` slid onto its neighbour still *compiles*: it
    /// simply never runs and the suite stays green.  Three such sites existed in this
    /// subsystem (`augment_spans_survive_a_later_humanize`, `push_u32_matches_push_uint`,
    /// `related_errors_with_a_label_round_trip`), which is a comment-proof failure
    /// mode — so it gets a mechanical guard instead of another comment:
    ///
    /// * every `fn` at test-module indentation must carry exactly one `#[test]`, or
    /// * be referenced elsewhere in the same file (a helper: `render(...)`, `span(..)`)
    ///   — an *orphaned* test body is referenced nowhere, which is exactly what makes
    ///   it invisible.
    ///
    /// Two `#[test]` lines on one function is the same accident's other half and is
    /// rejected too.
    #[test]
    fn every_test_module_fn_is_collected() {
        fn test_module_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
            let mut out = Vec::new();
            let mut from = 0usize;
            while let Some(rel) = text[from..].find("mod tests {") {
                let at = from + rel;
                let open = at + "mod tests ".len();
                let bytes = text.as_bytes();
                let mut depth = 0usize;
                let mut i = open;
                let mut end = text.len();
                while i < bytes.len() {
                    match bytes[i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i;
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                out.push(at..end);
                from = end.max(at + 1);
                if from >= text.len() {
                    break;
                }
            }
            out
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/diagnostics");
        let mut files = vec![root.join("mod.rs")];
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|x| x == "rs") && path != files[0] {
                    files.push(path);
                }
            }
        }
        files.sort();

        let mut offenders: Vec<String> = Vec::new();
        for path in &files {
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let name = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            let ranges = test_module_ranges(&text);
            if ranges.is_empty() {
                continue;
            }
            // Uses are counted over the *test* code only.  Counting over the whole file
            // inflated both sides with production identifiers of the same name — a helper
            // called `render`, `span` or `wrap` was reported as "used" because production
            // code elsewhere in the file contains `render(`/`fn span(`, which is exactly
            // the false negative that matters for the most likely helper names.
            let test_code: String = ranges.iter().map(|r| &text[r.clone()]).collect();
            let lines: Vec<&str> = text.lines().collect();
            let mut offset = 0usize;
            for (lineno, line) in lines.iter().enumerate() {
                let line_start = offset;
                offset += line.len() + 1;
                if !ranges.iter().any(|r| r.contains(&line_start)) {
                    continue;
                }
                let Some(rest) = line.strip_prefix("    fn ") else {
                    continue;
                };
                let Some(fname) = rest.split(['(', '<']).next() else {
                    continue;
                };
                if fname.is_empty() {
                    continue;
                }
                // Attribute stack: walk back over attributes and doc comments.
                let mut attrs: Vec<&str> = Vec::new();
                let mut k = lineno;
                while k > 0 {
                    k -= 1;
                    let prev = lines[k].trim();
                    if prev.starts_with("///") || prev.starts_with("//") || prev.starts_with("#[") {
                        attrs.push(prev);
                    } else {
                        break;
                    }
                }
                let tests = attrs.iter().filter(|a| a.starts_with("#[test]")).count();
                if tests > 1 {
                    offenders.push(format!(
                        "{name}:{}: `{fname}` carries {tests} `#[test]` attributes",
                        lineno + 1
                    ));
                    continue;
                }
                if tests == 0 {
                    // A helper must be *used*; an orphaned test body is not.
                    // Count *uses*: occurrences of `name(` that are not the
                    // definition itself.  A generic helper defines as
                    // `fn name<'a>(` — which `name(` does not match — so the
                    // subtraction is what keeps legit helpers out of the report.
                    // A bare `matches("{fname}(")` counted `span_lines(` as a call to
                    // `span`, so a helper whose name is a *prefix* of a sibling could hide
                    // a true orphan — the exact failure this guard exists to catch.
                    let needle = format!("{fname}(");
                    let calls = test_code
                        .match_indices(&needle)
                        .filter(|(at, _)| {
                            *at == 0 || {
                                let prev = test_code.as_bytes()[at - 1];
                                !prev.is_ascii_alphanumeric() && prev != b'_'
                            }
                        })
                        .count();
                    let defined = test_code.matches(&format!("fn {fname}(")).count();
                    if calls <= defined {
                        offenders.push(format!(
                            "{name}:{}: `{fname}` has no `#[test]` and is never called — an \
                             orphaned test body (its attribute slid onto a neighbour)",
                            lineno + 1
                        ));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "test-attribute wiring is broken:\n{}",
            offenders.join("\n")
        );
    }

    /// Interleaved `push`/`primary_ref` must stay exact and must not re-sort more
    /// than once per dirty stretch (this pins the analysis recorded on
    /// `MultiSpan::push`: the cache is only consulted while `dirty`, so one sort
    /// serves every read until the next real push).
    #[test]
    fn multispan_push_read_interleaving_keeps_the_view_exact() {
        let f = FileId::DUMMY;
        let mut ms = MultiSpan::new(Span::new(f, 5, 6));
        // A descending push makes the inline vector unsorted.
        ms.push(Span::new(f, 1, 2));
        assert_eq!(
            ms.primary_ref().as_ref(),
            &[Span::new(f, 1, 2), Span::new(f, 5, 6)],
            "the dirty read must sort"
        );
        // Repeated reads are cache hits and identical.
        for _ in 0..4 {
            assert_eq!(
                ms.primary_ref().as_ref(),
                &[Span::new(f, 1, 2), Span::new(f, 5, 6)]
            );
        }
        // A duplicate tail push changes nothing, including the sorted view.
        ms.push(Span::new(f, 5, 6));
        assert_eq!(
            ms.primary_ref().as_ref(),
            &[Span::new(f, 1, 2), Span::new(f, 5, 6)],
            "an idempotent push must not dirty the view"
        );
        // A new ascending element joins the view in order.
        ms.push(Span::new(f, 9, 10));
        assert_eq!(
            ms.primary_ref().as_ref(),
            &[Span::new(f, 1, 2), Span::new(f, 5, 6), Span::new(f, 9, 10)]
        );
        // And a descending one is re-sorted, not appended.
        ms.push(Span::new(f, 3, 4));
        assert_eq!(
            ms.primary_ref().as_ref(),
            &[
                Span::new(f, 1, 2),
                Span::new(f, 3, 4),
                Span::new(f, 5, 6),
                Span::new(f, 9, 10)
            ]
        );
    }

    /// `augment` copies the other diagnostic's spans and sets `spans_user_set`, so a
    /// *later* `humanize()` must not replace them with kind-derived spans — even when
    /// `other`'s spans were themselves kind-derived.  Deliberate adorn semantics, and
    /// exactly the shape of the past `spans_user_set` regressions, so pin it.
    ///
    /// The `#[test]` here had slid onto the neighbouring function during an edit, so
    /// this pin never ran (`every_test_module_fn_is_collected` below now catches that
    /// family structurally).
    #[test]
    fn augment_spans_survive_a_later_humanize() {
        let span = |s: u32, e: u32| Span::new(FileId::DUMMY, s, e);
        let mut other =
            Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::TypeMismatch {
                expected: "int".into(),
                found: "string".into(),
                span: span(7, 9),
                found_span: None,
                reason: None,
                context: None,
            });
        other.humanize();
        assert_eq!(other.spans().first(), Some(span(7, 9)));

        // Target has no spans at all, so `augment` takes the anchor.
        let mut target = Diagnostic::error("composed");
        target.augment(&other);
        assert_eq!(target.spans().first(), Some(span(7, 9)));

        // A later humanize on the target must not overwrite the augmented span
        // (the target has no kind, so it would have nothing to derive from — the
        // meaningful case is a *kind-built* target).
        let mut kind_target =
            Diagnostic::error_kind(crate::diagnostics::kind::DiagnosticKind::TypeMismatch {
                expected: "bool".into(),
                found: "int".into(),
                span: span(100, 104),
                found_span: None,
                reason: None,
                context: None,
            });
        kind_target.augment(&other);
        assert_eq!(
            kind_target.spans().first(),
            Some(span(7, 9)),
            "augment must win over the kind's own span"
        );
        kind_target.humanize();
        assert_eq!(
            kind_target.spans().first(),
            Some(span(7, 9)),
            "humanize must not re-derive over user/augmented spans"
        );
    }

    /// The dedup key deliberately **excludes the label kind**: two labels on the same
    /// span with the same text are one annotation even when one is primary and one
    /// secondary.  That rule's only consumer is `augment`'s label merge, and both
    /// augment tests above compare labels of the *same* kind — so a "fix" that added
    /// `kind` to `Label::dedup_key` would pass the entire suite.  Pinned here.
    #[test]
    fn augment_dedups_across_label_kinds() {
        let span = Span::new(FileId::DUMMY, 1, 3);
        let mut target = Diagnostic::error("target").with_label(span, "same text");
        let other = Diagnostic::error("other").with_secondary_label(span, "same text");
        target.augment(&other);
        assert_eq!(
            target.labels().len(),
            1,
            "a primary and a secondary label on the same span with the same text must \
             collapse: {:?}",
            target
                .labels()
                .iter()
                .map(|l| &l.message)
                .collect::<Vec<_>>()
        );
    }

    /// `Diagnostic::augment` merges labels without duplicating a `(span, message)`
    /// pair — on both the small allocation-free scan and the keyed-set path used when
    /// the target already carries many labels — and a label duplicated *inside*
    /// `other` is added only once.
    #[test]
    fn augment_dedups_labels_on_both_paths() {
        for n in [2usize, 20] {
            // n = 2 exercises the scan; n = 20 the keyed-set branch.
            let mut target = Diagnostic::error("target");
            let mut other = Diagnostic::error("other");
            for i in 0..n {
                let s = Span::new(FileId::DUMMY, (i * 4) as u32, (i * 4 + 2) as u32);
                target = target.with_secondary_label(s, format!("m{i}"));
                other = other.with_secondary_label(s, format!("m{i}"));
            }
            let dup = Span::new(FileId::DUMMY, 1000, 1002);
            other = other.with_secondary_label(dup, "dup");
            other = other.with_secondary_label(dup, "dup"); // duplicate within `other`
            let fresh = Span::new(FileId::DUMMY, 2000, 2002);
            other = other.with_secondary_label(fresh, "fresh");

            let before = target.labels.len();
            target.augment(&other);
            assert_eq!(
                target.labels.len(),
                before + 2,
                "n={n}: only `dup` (once) and `fresh` should be added"
            );
        }
    }

    /// `augment` with a small `self` and a large `other` must take the
    /// keyed-set path.  A threshold that only guards `self.labels.len()`
    /// would route this to the scan path, chaining the growing `to_add`
    /// on every iteration — O(|other|²) string comparisons.  This locks
    /// the both-sides threshold: the merge must stay linear and dedup
    /// correctly regardless of which side is large.
    #[test]
    fn augment_small_self_large_other_takes_keyed_path() {
        let mut target = Diagnostic::error("target");
        let s0 = Span::new(FileId::DUMMY, 0, 2);
        target = target.with_secondary_label(s0, "self0");

        let mut other = Diagnostic::error("other");
        let n = 500;
        for i in 0..n {
            let s = Span::new(FileId::DUMMY, (i * 4 + 100) as u32, (i * 4 + 102) as u32);
            other = other.with_secondary_label(s, format!("m{i}"));
        }

        target.augment(&other);
        // `self0` plus all `n` distinct labels from `other`.
        assert_eq!(target.labels.len(), 1 + n);
    }

    /// The `--explain` body format: two spaces **or** one tab means code.
    /// A tab-indented example used to fall through to the prose path (wrapped
    /// and highlighted as text), and nothing caught it because every registered
    /// body happens to use spaces.
    #[test]
    fn explain_code_line_detection_accepts_both_indentations() {
        assert!(explain_line_is_code("  let x = 1;"));
        assert!(explain_line_is_code("\tlet x = 1;"));
        assert!(explain_line_is_code("\t  nested();"));
        // Prose, blank, and whitespace-only lines are not code.
        assert!(!explain_line_is_code("plain prose"));
        assert!(!explain_line_is_code(""));
        assert!(!explain_line_is_code("  "));
        assert!(!explain_line_is_code("\t"));
        // A single leading space is prose (the format requires two).
        assert!(!explain_line_is_code(" one space"));
    }

    /// Every code block is preceded by exactly one blank line, and never by
    /// two.
    ///
    /// `prev_blank` was set by the source's blank line and never cleared, so
    /// only the *first* block after any blank ever got a separator: with a body
    /// shaped `prose / "" / code / prose / code` the second code block was
    /// glued to the preceding prose line.  No registered body has that shape,
    /// which is why the bug survived — hence the synthetic body here, exercised
    /// through the extracted boundary rather than through a registry entry.
    #[test]
    fn every_code_block_is_preceded_by_exactly_one_blank_line() {
        let body = "prose one\n\n  code one\nprose two\n  code two\n";
        let mut out = String::new();
        push_explain_body(&mut out, body, false, term::ColorDepth::Sixteen, 80, 4);
        assert_eq!(
            out, "prose one\n\n    code one\n\nprose two\n\n    code two\n",
            "code blocks must each get exactly one separating blank line"
        );

        // The original double-blank case: a source blank immediately before the
        // block must not be doubled by the opening blank.
        let mut out = String::new();
        push_explain_body(
            &mut out,
            "prose\n\n  code\n",
            false,
            term::ColorDepth::Sixteen,
            80,
            4,
        );
        assert_eq!(out, "prose\n\n    code\n");

        // Back-to-back blocks separated by a source blank keep exactly one blank.
        let mut out = String::new();
        push_explain_body(
            &mut out,
            "  a\n\n  b\n",
            false,
            term::ColorDepth::Sixteen,
            80,
            4,
        );
        assert_eq!(out, "    a\n\n    b\n");
    }

    /// A zero `--width` / `--tab-width` from the CLI must not degenerate the
    /// explain body (one word per line) or divide by zero in the column math.
    #[test]
    fn explain_body_clamps_a_zero_width() {
        let mut out = String::new();
        push_explain_body(
            &mut out,
            "alpha beta gamma\n",
            false,
            term::ColorDepth::Sixteen,
            0,
            0,
        );
        // A zero width is clamped to 1, and width 1 legitimately degenerates
        // (no word fits) — the contract pinned here is totality: no panic, and
        // nothing dropped.  A caller asking for width 0 gets the same body as
        // width 1 rather than an error or a missing character.
        let mut one = String::new();
        push_explain_body(
            &mut one,
            "alpha beta gamma\n",
            false,
            term::ColorDepth::Sixteen,
            1,
            1,
        );
        assert_eq!(out, one, "width 0 must clamp to width 1");
        let joined: String = out.chars().filter(|c| *c != '\n').collect();
        assert_eq!(
            joined, "alphabetagamma",
            "clamping must not drop characters: {out:?}"
        );
    }

    #[test]
    fn diagnostic_size_bounded() {
        let sz = std::mem::size_of::<Diagnostic<'static>>();
        let multi = std::mem::size_of::<MultiSpan>();
        let kind = std::mem::size_of::<DiagnosticKind<'static>>();
        println!(
            "[size] Diagnostic={sz}  MultiSpan={multi}  DiagnosticKind={kind}  \
             Box<DiagnosticKind>={}  Subdiag={}  RelatedError={}",
            std::mem::size_of::<Option<std::sync::Arc<DiagnosticKind<'static>>>>(),
            std::mem::size_of::<Subdiag<'static>>(),
            std::mem::size_of::<RelatedError<'static>>()
        );
        assert!(sz <= 416, "Diagnostic grew to {sz} bytes");
        // 416 = the historical 384-byte bound + 16 B for `MultiSpan`'s
        // sorted-view cache (`OnceLock<Box<[Span]>>`, added so a dirty
        // `MultiSpan` sorts once per dirty epoch instead of on every
        // `primary()` read) + 8 B of padding for the `kind_derived_labels`
        // bookkeeping field that makes `humanize` preserve user labels.
    }

    #[test]
    fn stable_id_cache_reflects_hashed_field_mutations() {
        let base = Diagnostic::error("boom")
            .with_code_str(ErrCode::E030)
            .with_span(Span::new(FileId::DUMMY, 0, 5));
        let base_id = base.stable_id();
        assert_eq!(base.stable_id(), base_id);

        let new_code = base.clone().with_code_str(ErrCode::E031);
        assert_ne!(new_code.stable_id(), base_id);

        let new_span = base.clone().with_span(Span::new(FileId::DUMMY, 20, 25));
        assert_ne!(new_span.stable_id(), base_id);

        let mut empty = Diagnostic::error("boom");
        let empty_id = empty.stable_id();
        empty.augment(&base);
        assert_ne!(empty.stable_id(), empty_id);

        let mut d = base.clone();
        d.set_level(DiagnosticLevel::Warning);
        d.set_source("ignored");
        // `stable_id` hashes the file NAME, so a rename must invalidate
        // the cache and yield a new id (the old code returned the stale id,
        // reintroducing cross-file collisions).
        let before_rename = d.stable_id();
        d.set_file_name("ignored.pn");
        assert_ne!(d.stable_id(), before_rename);
        assert_ne!(d.stable_id(), base_id);

        // `with_kind` changes the eventual (humanized) message, so a cached
        // id computed before it must be invalidated, not returned stale.
        let kind_a = DiagnosticKind::NameNotFound {
            name: std::borrow::Cow::Borrowed("a"),
            span: Span::new(FileId::DUMMY, 0, 1),
            suggestions: smallvec![],
        };
        let kind_b = DiagnosticKind::NameNotFound {
            name: std::borrow::Cow::Borrowed("b"),
            span: Span::new(FileId::DUMMY, 0, 1),
            suggestions: smallvec![],
        };
        let mut d = base.clone().with_kind(kind_a);
        d.humanize();
        let id_a = d.stable_id();
        // `with_kind` invalidates the cache; the id only changes once the new
        // kind is materialized, which is what `humanize` does.
        d = d.with_kind(kind_b);
        d.humanize();
        assert_ne!(d.stable_id(), id_a);
    }

    /// The direct-address neighbourhood enumeration must produce the same
    /// top-3 as the full table scan for: every valid registry code (distance
    /// 0), single-edit typos of them, the corner codes E000/E999/W000/W999,
    /// cross-prefix typos, 5-digit insertions, and unrelated noise.
    #[test]
    fn suggest_code_direct_matches_scan() {
        let table_codes: Vec<&str> = crate::diagnostics::error_code::CODE_TABLE
            .iter()
            .map(|e| e.code)
            .collect();
        let mut inputs: Vec<String> = table_codes.iter().map(|c| c.to_string()).collect();
        // Lowercase spellings, corners, cross-prefix, insertions, noise.
        for c in table_codes.iter().step_by(17) {
            inputs.push(c.to_ascii_lowercase());
        }
        for c in ["E000", "E999", "W000", "W999", "E0000", "W1234", "E12345"] {
            inputs.push(c.to_string());
        }
        // Digit-corrupted typos of a sample of codes (all four lanes, plus
        // transpositions) and non-shaped noise.
        for c in table_codes.iter().step_by(5) {
            let b = c.as_bytes();
            for lane in 1..4 {
                for d in b'0'..=b'9' {
                    if d != b[lane] {
                        let mut t = b.to_vec();
                        t[lane] = d;
                        inputs.push(String::from_utf8(t).unwrap());
                    }
                }
            }
            let mut t = b.to_vec();
            t.swap(1, 2);
            inputs.push(String::from_utf8(t).unwrap());
        }
        inputs.push("????".into());
        inputs.push("E".into());
        inputs.push("ERR0123".into());
        for input in &inputs {
            // The scan reference needs the same uppercased needle.
            let mut needle = input.as_str().to_owned();
            needle.make_ascii_uppercase();
            let needle_len = needle.chars().count();
            let max_dist = std::cmp::max(1, needle_len / 3);
            let direct = suggest_top3_direct(&needle);
            let scan = suggest_top3_scan(&needle, max_dist, needle_len);
            assert_eq!(
                direct.as_slice(),
                scan.as_slice(),
                "input {input:?} (needle {needle:?})"
            );
        }
    }

    /// Plain (`use_color == false`) explain output for an unknown/typo'd code
    /// must contain no ANSI escapes — `suggest_code` used to hard-code its
    /// `\x1b[1;34m` help label into both color modes.
    #[test]
    fn plain_explain_of_unknown_code_has_no_ansi() {
        assert!(!suggest_code("E031", false).contains('\x1b'));
        assert!(suggest_code("E031", true).contains('\x1b'));
        for code in ["zzz", "E03", "????"] {
            let out = explain_error_code(code, false);
            assert!(!out.contains('\x1b'), "plain explain of {code}: {out:?}");
        }
    }

    /// `sanitize_filename` composes `sanitize_for_terminal`, which already
    /// removes bidi overrides/isolates.  Pin that composed contract: the
    /// (removed) second filter in `sanitize_filename` was unreachable, so
    /// this test is the guard if `sanitize_for_terminal` ever weakens.
    #[test]
    fn sanitize_filename_removes_bidi_overrides() {
        let spoof = "evil.code\u{202E}exe";
        assert_eq!(&*sanitize_filename(spoof), "evil.codeexe");
        // Controls are replaced by the same composed pass.
        assert_eq!(&*sanitize_filename("a\x07b"), "a\u{FFFD}b");
        // Clean ASCII stays borrowed (no extra scan cost).
        assert!(matches!(sanitize_filename("src/main.rs"), Cow::Borrowed(_)));
    }

    /// Bidi overrides/isolates must not survive any terminal sanitizer: they
    /// are invisible, so the control scans cannot see them (`evil.code\u{202E}exe`
    /// used to leak through with a spoofed extension).
    #[test]
    fn sanitizers_strip_bidi_overrides_and_isolates() {
        let spoof = "evil.code\u{202E}exe";
        for out in [
            sanitize_for_terminal(spoof).to_string(),
            sanitize_for_terminal_multiline(spoof).to_string(),
            sanitize_for_terminal_flatten(spoof).to_string(),
        ] {
            assert_eq!(out, "evil.codeexe", "spoof survived: {out:?}");
            assert!(!out.contains('\u{202E}'));
        }
        // Isolates (U+2066–U+2069) are removed too.
        let isolates = "a\u{2066}b\u{2067}c\u{2068}d\u{2069}e";
        let out = sanitize_for_terminal(isolates);
        assert_eq!(&*out, "abcde");
        // i18n text without bidi/controls still borrows.  Neither string contains
        // a `0xE2` byte, so both now skip the bidi `memchr` entirely — the gate the
        // comment on each sanitizer claims.
        let clean = "тип: тип（中文）";
        assert!(matches!(sanitize_for_terminal(clean), Cow::Borrowed(_)));
        assert!(matches!(
            sanitize_for_terminal_multiline(clean),
            Cow::Borrowed(_)
        ));
        // The gate's *open* side: `0xE2` present (em dash, U+2014) but no bidi
        // override, so the probe runs, finds nothing, and the string still borrows.
        // This is what distinguishes `has_e2` from the old `has_high`: the CJK
        // string above would pay the `memchr` under `has_high`, and an over-eager
        // gate would wrongly own this one.
        let em_dash = "a\u{2014}b";
        assert!(matches!(sanitize_for_terminal(em_dash), Cow::Borrowed(_)));
        assert!(matches!(
            sanitize_for_terminal_multiline(em_dash),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            sanitize_for_terminal_flatten(em_dash),
            Cow::Borrowed(_)
        ));
        // Layout controls survive their respective sanitizers alongside
        // bidi removal.
        let mixed = "a\u{202E}b\nc\td";
        assert_eq!(&*sanitize_for_terminal_multiline(mixed), "ab\nc\td");
        let mixed_cr = "a\u{202E}b\nc\rd";
        assert_eq!(&*sanitize_for_terminal_flatten(mixed_cr), "ab\nc\rd");
    }

    /// The flatten scanner excludes `\t`/`\n`/`\r` from the dirty set, so a
    /// multi-line message with no other controls stays `Cow::Borrowed` — and
    /// the SWAR flags must agree with the scalar predicate over random
    /// tab/newline/CR-heavy slices.
    #[test]
    fn flatten_scan_borrows_layout_only_controls() {
        for s in ["a\nb\rc\td", "\n\r\t", "line one\nline two\r\n\tthree"] {
            assert!(matches!(sanitize_for_terminal_flatten(s), Cow::Borrowed(_)));
        }
        // Still replaced: DEL, VT, ESC, C1.
        for s in ["a\x7Fb", "a\x0Bb", "a\x1Bb", "a\u{0085}b"] {
            assert!(matches!(sanitize_for_terminal_flatten(s), Cow::Owned(_)));
        }
        // Regression: C1 in the SIMD/SWAR body (not just the scalar tail).
        // The u64 kernel previously XORed with 0x02 instead of 0xC2, so a
        // long string whose only control was C1 borrowed unsanitized.
        for pad in 0..32 {
            let s = format!("{}a\u{0085}bcdefghijklmnopqrstuvwxyz", "x".repeat(pad));
            assert!(has_c1_sequence(s.as_bytes()), "pad {pad}");
            assert!(
                scan_terminal_control_flags(s.as_bytes()).has_c1,
                "pad {pad}"
            );
            assert!(matches!(sanitize_for_terminal(&s), Cow::Owned(_)));
        }
        // Non-ASCII without C1 must not falsely report C1.
        let clean = format!("{}é", "x".repeat(64));
        assert!(!scan_terminal_control_flags(clean.as_bytes()).has_c1);
        assert!(matches!(sanitize_for_terminal(&clean), Cow::Borrowed(_)));
        let reference = |s: &[u8]| {
            s.iter()
                .any(|&b| (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r') || b == 0x7F)
        };
        // Poison bytes that previously aliased tab/nl/cr in the SWAR mask:
        // U+0089/U+008A/U+008D are 2-byte UTF-8 and must not report a break.
        for s in ["CAFÉ", "ĉ", "‽"] {
            assert_eq!(
                scan_terminal_control_flags_flatten(s.as_bytes()).first_break,
                None,
                "{s:?}"
            );
            assert!(matches!(
                sanitize_for_terminal_flatten_with_break(s),
                (Cow::Borrowed(_), None)
            ));
        }
        // Owned path: break offset is in the sanitized string, after U+FFFD
        // growth. "a\x07b\tc" → "a�b\tc", tab at byte 5.
        let (owned, break_at) = sanitize_for_terminal_flatten_with_break("a\x07b\tc");
        assert!(matches!(owned, Cow::Owned(_)));
        assert_eq!(break_at, owned.find('\t'));
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_0001;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(7);
            seed
        };
        for _ in 0..10_000 {
            let len = (next() % 48) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    let r = next() & 0xFF;
                    if r < 180 {
                        [b'\t', b'\n', b'\r'][(next() % 3) as usize]
                    } else {
                        r as u8
                    }
                })
                .collect();
            let scan = scan_terminal_control_flags_flatten(&bytes);
            let ControlScan {
                dirty,
                first_break: break_at,
                ..
            } = scan;
            assert_eq!(dirty, reference(&bytes), "{bytes:?}");
            let want_break = bytes
                .iter()
                .position(|&b| matches!(b, b'\t' | b'\n' | b'\r'));
            assert_eq!(break_at, want_break, "{bytes:?}");
        }
    }

    /// Alternating pushes and reads: the sorted-view cache must never serve a
    /// stale view, and every `primary()` result stays sorted + deduped.
    #[test]
    fn multispan_push_primary_interleave_stays_sorted() {
        let mut rng: u64 = 0xC0FF_EE00_0000_0001;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(11);
            rng
        };
        let mut ms = MultiSpan::default();
        let mut expect: Vec<Span> = Vec::new();
        for _ in 0..2_000 {
            let start = (next() % 64) as u32;
            let span = Span::new(FileId::DUMMY, start, start + 1 + (next() % 8) as u32);
            if next() & 1 == 0 {
                ms.push(span);
                if !expect.contains(&span) {
                    expect.push(span);
                }
            } else if !expect.is_empty() {
                expect.sort_unstable();
                expect.dedup();
                // `primary()` is contractually sorted + deduped; deduping here
                // would mask a regression in that invariant.
                let got = ms.primary();
                assert_eq!(got.as_slice(), expect.as_slice());
                // `first()` must agree with the sorted view it exposed, and
                // reading it here exercises the dirty-read cache path.
                assert_eq!(ms.first(), expect.first().copied());
                match ms.primary_ref() {
                    Cow::Borrowed(s) => assert_eq!(s, expect.as_slice()),
                    Cow::Owned(s) => assert_eq!(s, expect.as_slice()),
                }
            }
            if next() % 97 == 0 {
                ms.clear();
                expect.clear();
            }
        }
    }

    /// A kind-built diagnostic enriched with `with_label` keeps BOTH the
    /// kind-derived labels and the user label across humanize, and
    /// re-humanizing after a language change does not duplicate either.
    #[test]
    fn humanize_preserves_user_labels_alongside_kind_derived() {
        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let user_span = Span::new(FileId::DUMMY, 40, 45);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: kind_span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_label(user_span, "user context");
        diag.humanize();
        assert_eq!(diag.labels.len(), 2, "kind + user label");
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(diag.labels[1].span, user_span);
        // Re-humanize (latch cleared, same language): no duplication, order
        // stable — the derived block is rebuilt, the user label kept.
        diag.humanized = false;
        diag.humanize();
        assert_eq!(diag.labels.len(), 2);
        assert_eq!(diag.labels[0].span, kind_span);
        assert_eq!(diag.labels[1].span, user_span);
    }

    /// The `humanize` latch is keyed to [`Lang`]: a language change must
    /// re-derive `message`/`labels` from the retained kind, and a kind-less
    /// diagnostic must keep the message a caller set manually no matter how
    /// often the language changes.  Pins the `humanized_lang != lang` branch
    /// that no other test exercises.
    ///
    /// [`Lang`]: crate::diagnostics::i18n::Lang
    #[test]
    fn humanize_rederives_on_language_change_and_keeps_manual_messages() {
        use crate::diagnostics::i18n::{Lang, acquire_test_lang_lock};
        let _lang_guard = acquire_test_lang_lock();
        let original_lang = Lang::current();
        Lang::set(Lang::En);

        let kind_span = Span::new(FileId::DUMMY, 0, 3);
        let user_span = Span::new(FileId::DUMMY, 40, 45);
        let mut diag = Diagnostic::error("placeholder")
            .with_kind(DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Str"),
                span: kind_span,
                found_span: None,
                reason: None,
                context: None,
            })
            .with_label(user_span, "user context");
        diag.humanize();
        let english = diag.message().to_string();

        Lang::set(Lang::Zh);
        diag.humanize();
        assert_ne!(
            diag.message(),
            english.as_str(),
            "a language change must re-derive the kind's message"
        );
        assert_eq!(diag.labels().len(), 2, "kind-derived + user label");
        assert_eq!(diag.labels()[1].span, user_span);
        assert_eq!(
            diag.labels()[1].message,
            "user context",
            "the user-added label must survive re-humanization"
        );

        // Kind-less: there is nothing to re-resolve, so the manually set
        // message is stable across a language change.
        let mut manual = Diagnostic::error("hand-written message");
        manual.humanize();
        Lang::set(Lang::En);
        manual.humanize();
        assert_eq!(manual.message(), "hand-written message");

        Lang::set(original_lang);
    }

    /// A user `Primary` label must **not** widen the kind-derived span set.
    ///
    /// `humanize` derives `MultiSpan` only from the kind's own labels; the decision is documented on
    /// [`Diagnostic::with_label`] and this pins it.  Without a pin, "why doesn't my span show up in
    /// the section header" has two possible answers and only one of them is implemented.
    #[test]
    fn user_primary_labels_do_not_extend_the_derived_spans() {
        let mut diag =
            Diagnostic::error("boom").with_label(Span::new(FileId::DUMMY, 10, 12), "a user label");
        // The builder never set spans explicitly, so derivation is active.
        assert!(
            !diag.spans_user_set,
            "the fixture must not disable derivation"
        );
        diag.humanize();
        let spans = diag.spans().primary_ref();
        assert!(
            !spans.iter().any(|s| s.start == 10 && s.end == 12),
            "a user label leaked into the derived span set: {:?}",
            spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>()
        );
    }

    /// Regression: a closure that pushes a *suppressed* warning and returns
    /// `Err` must not corrupt an earlier, unrelated diagnostic via
    /// `last_mut()`.
    #[test]
    fn adorn_with_span_does_not_augment_unrelated_prior_diagnostic() {
        use crate::diagnostics::EmissionGuaranteed;
        let mut ctxt = DiagCtxt::new();
        ctxt.set_can_emit_warnings(false);
        // Pre-existing unreported diagnostic — `last_mut()` would return this
        // if the guard were missing.
        ctxt.push(Diagnostic::error("prior").with_span(Span::new(FileId::DUMMY, 100, 105)));
        let result: Result<(), EmissionGuaranteed> = adorn_with_span(
            &mut ctxt,
            Span::new(FileId::DUMMY, 0, 5),
            Some("hello"),
            |ctxt| {
                // Suppressed (records nothing), then fail without pushing.
                Err(ctxt.push(Diagnostic::warning("suppressed")).into())
            },
        );
        assert!(result.is_err());
        let prior = &ctxt.into_inner()[0];
        assert_eq!(
            prior.spans.primary().as_slice(),
            &[Span::new(FileId::DUMMY, 100, 105)][..],
            "the prior diagnostic's span must be untouched"
        );
        assert_eq!(prior.source, None);
    }
}
