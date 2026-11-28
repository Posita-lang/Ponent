use crate::ast::Span;
use crate::diagnostics::DiagMessage;
use crate::diagnostics::i18n::LiftCow;
use crate::diagnostics::label::{AnnotationKind, Label};
use crate::smallvec::{SmallVec, smallvec};
use covariant_vec::CovariantVec;
use std::borrow::Cow;
use std::fmt::Write as _;

/// Describes the context in which a type was expected.
/// Inspired by Elm's `Expected` type which carries `Context` — knowing
/// *why* a type was expected allows the renderer to produce more precise
/// error messages like "parameter 1 of `foo` expects `Int<32>`" instead
/// of just "expected `Int<32>`".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCtx {
    /// No specific context (general/unspecified).
    Unspecified,
    /// From a variable or parameter type annotation.
    Annotation,
    /// From a function return type annotation.
    ReturnType,
    /// From a binary operator's operand type.
    BinOp,
    /// From a function call argument position.
    FunctionArg,
    /// From a record field type.
    Field,
    /// From a type alias or generic constraint.
    TypeAlias,
    /// From a contract condition (requires / ensures).
    Contract,
    /// From a variable definition's inferred type.
    Inference,
}

impl TypeCtx {
    /// Localized parenthetical label.  `Unspecified` is empty: callers skip
    /// the suffix entirely rather than emitting `()`.
    ///
    /// Returns `&'static str`, not `Cow`: the cache table is a `'static`
    /// `OnceLock`, so the `Cow::Owned` branch was unreachable — a 32-byte shell
    /// around a static slice on a per-frame path.  `level.rs`'s
    /// `DiagnosticLevel::label` made the same move for the same reason.
    pub fn label(&self) -> &'static str {
        let variant = match self {
            TypeCtx::Unspecified => return "",
            TypeCtx::Annotation => 1,
            TypeCtx::ReturnType => 2,
            TypeCtx::BinOp => 3,
            TypeCtx::FunctionArg => 4,
            TypeCtx::Field => 5,
            TypeCtx::TypeAlias => 6,
            TypeCtx::Contract => 7,
            TypeCtx::Inference => 8,
        };
        crate::diagnostics::i18n::cached_label_static! {
            n = 9,
            variant = variant,
            // Slot 0 (`Unspecified`) is unreachable — it returns early above —
            // but the empty slug keeps the array anchored to the discriminant.
            slugs = [
                "",
                "type-ctx-annotation",
                "type-ctx-return-type",
                "type-ctx-binop",
                "type-ctx-function-arg",
                "type-ctx-field",
                "type-ctx-type-alias",
                "type-ctx-contract",
                "type-ctx-inference",
            ],
        }
    }
}

impl std::fmt::Display for TypeCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Kept for diagnostics/tests that format a TypeCtx directly; the
        // humanizer uses `label()` so the suffix follows `Lang`.
        if matches!(self, TypeCtx::Unspecified) {
            return Ok(());
        }
        f.write_str(self.label())
    }
}

/// A structured diagnostic kind, carrying the exact data relevant to the
/// error or warning being reported.
///
/// Inspired by Vale's `ICompileErrorT` ADT: instead of stuffing everything
/// into a `message: String`, each variant holds typed fields that a
/// [`Humanizer`] can use to produce precise, context-aware error messages
/// and annotations.
///
/// The `'src` lifetime binds all borrowed string data (type names,
/// identifiers, field names, etc.) to the source session so that names
/// originating from the AST/interner can be stored without cloning into
/// a `String`.  Constructed values may still own dynamic text via
/// `Cow::Owned`.
///
/// # Example
///
/// ```ignore
/// Diagnostic::error(DiagnosticKind::TypeMismatch {
///     expected: "Int<32>",
///     found: "&Str",
///     span: some_span,
///     found_span: some_other_span,
/// })
/// .with_code_str(ErrCode::E030);
/// ```
#[derive(Debug, Clone)]
pub enum DiagnosticKind<'src> {
    /// A value of one type was used where another type was expected.
    TypeMismatch {
        expected: Cow<'src, str>,
        found: Cow<'src, str>,
        /// The span of the expression with the wrong type.
        span: Span,
        /// Optional span of the value that produced the found type.
        found_span: Option<Span>,
        /// Optional explanation of WHY the types don't match
        /// (e.g. "Int<16> is not a subtype of Int<23>").
        reason: Option<Cow<'src, str>>,
        /// The context in which the expected type was determined
        /// (e.g. from a type annotation, a function return type, etc.).
        context: Option<TypeCtx>,
    },
    /// A field access referred to a field that doesn't exist on the type.
    NoSuchField {
        field_name: Cow<'src, str>,
        type_name: Cow<'src, str>,
        span: Span,
    },
    /// A function/method call had argument type mismatches.
    ArgumentTypeMismatch {
        callee: Cow<'src, str>,
        param_name: Cow<'src, str>,
        expected: Cow<'src, str>,
        found: Cow<'src, str>,
        span: Span,
        param_span: Option<Span>,
    },
    /// A name could not be resolved in the current scope.
    NameNotFound {
        name: Cow<'src, str>,
        span: Span,
        // `'static`, deliberately — the narrowing looks unnecessary, but it is
        // real and load-bearing: `hir`'s
        // `FnBodyCheck`/`VfsError` store the kind for longer than the caller's
        // `'input` borrow lives, so `Cow<'src, str>` here makes those two producers
        // require `'input: 'static` (verified by attempting the change).  Widening
        // means making both producers own their suggestion strings, which is a
        // separate, larger change — and nothing in
        // `name_not_found_help{,_owned}`/`join_with_or` would benefit.
        suggestions: SmallVec<[Cow<'static, str>; 2]>,
    },
    /// A duplicate definition (variable, function, type).
    DuplicateDefinition {
        name: Cow<'src, str>,
        this_span: Span,
        original_span: Span,
    },
    /// A contract condition (`requires` / `ensures`) was not boolean.
    ContractNonBool {
        clause: Cow<'src, str>,
        found: Cow<'src, str>,
        span: Span,
    },
    /// A trait implementation is missing a required method.
    ImplMissingMethod {
        trait_name: Cow<'src, str>,
        method_name: Cow<'src, str>,
        impl_span: Span,
        trait_span: Span,
    },
    /// A compile-time evaluation error (comptime).
    Comptime {
        kind: ComptimeErrorKind,
        span: Span,
        /// Call stack, most recent first.  `None` marks a frame with no
        /// source position (the abandoned `(0,0)` sentinel — a real
        /// zero-length span at offset 0 stays distinguishable as
        /// `Some`).
        traceback: Vec<(ComptimeReason, Option<Span>)>,
    },
}

/// Why a block is being evaluated at compile time — used for error
/// context backtracking (like Zig's `BlockComptimeReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComptimeReason {
    /// User wrote `comptime { ... }` block.
    ComptimeBlock,
    /// Inside a `comptime def` function body.
    ComptimeFnDef,
    /// Calling a comptime function with `!`.
    ComptimeFnCall,
    /// Evaluating `@comptime_test` function.
    ComptimeTest,
    /// Evaluating `assert!()`.
    Assertion,
    /// Evaluating `@typeInfo!()`.
    TypeInfo,
    /// Evaluating `layout_of!()`.
    LayoutOf,
}

impl ComptimeReason {
    /// Human-readable description of this comptime context frame, resolved
    /// through Fluent so it follows the active
    /// [`Lang`](super::i18n::Lang).
    pub fn as_str(&self) -> &'static str {
        // Explicit variant → slot mapping, like every other label site
        // (`level.rs`, `error_code.rs`, `chain.rs`): reordering the enum
        // below must be a compile error here, not a silent label swap.
        let variant = match self {
            ComptimeReason::ComptimeBlock => 0,
            ComptimeReason::ComptimeFnDef => 1,
            ComptimeReason::ComptimeFnCall => 2,
            ComptimeReason::ComptimeTest => 3,
            ComptimeReason::Assertion => 4,
            ComptimeReason::TypeInfo => 5,
            ComptimeReason::LayoutOf => 6,
        };
        // `cached_label!`'s `Owned` branch was unreachable here too: the table
        // lives in a `'static OnceLock`, so borrow instead of wrapping a slice
        // in a `Cow` once per traceback frame.
        crate::diagnostics::i18n::cached_label_static! {
            n = 7,
            variant = variant,
            slugs = [
                "comptime-reason-block",
                "comptime-reason-fndef",
                "comptime-reason-fncall",
                "comptime-reason-test",
                "comptime-reason-assert",
                "comptime-reason-typeinfo",
                "comptime-reason-layoutof",
            ],
        }
    }
}

/// Payload text for comptime diagnostics.
///
/// `Arc<str>` rather than `Cow<'src, str>`: every producer in
/// `hir::comptime` constructs these from an owned `ComptimeError` string, so
/// there is no borrowed source text to preserve.  Threading `'src` here would
/// only add a lifetime parameter that leaks into the query cache
/// (`FnBodyCheck`/`VfsError` diagnostics are `'input`) without eliminating a
/// single allocation, while `Arc` makes the repeated `humanize` clones O(1).
#[derive(Debug, Clone)]
pub enum ComptimeErrorKind {
    /// Step limit exceeded (possible infinite loop).
    StepLimitExceeded,
    /// Division or remainder by zero.
    DivisionByZero,
    /// Integer overflow (trap policy).
    Overflow,
    /// Type mismatch in a comptime operation.
    TypeError(std::sync::Arc<str>),
    /// Assertion failed at compile time.
    AssertionFailed(std::sync::Arc<str>),
    /// Unknown identifier in comptime context.
    UnknownIdentifier(std::sync::Arc<str>),
    /// A runtime-only construct encountered in comptime context.
    NotComptimeAllowed(std::sync::Arc<str>),
    /// The expression cannot be evaluated at compile time.
    Deferred,
    /// A comptime sandbox violation.
    SandboxViolation(std::sync::Arc<str>),
    /// Memory limit exceeded during comptime evaluation.
    MemoryLimitExceeded(std::sync::Arc<str>),
    /// An internal comptime error.
    Internal(std::sync::Arc<str>),
}

/// Map a frame span to the traceback's `Option<Span>`: the abandoned
/// `(0,0)` sentinel becomes `None` (a frame with no source position).
///
/// The conversion is ambiguous by construction: a *real* zero-length span at
/// file offset 0 is exactly `(0,0)` and is also mapped to `None`, silently
/// losing its position.  The sentinel lives in the traceback source
/// representation (`Vec<(reason, Option<Span>)>` upstream stores the real
/// `Option`), so this helper only bridges legacy `(reason, Span)` lists; a
/// caller that needs start-of-file zero-length frames must store
/// `Option<Span>` at construction instead of recovering it here.
/// # Legacy bridge — do not add callers
///
/// This exists only to adapt the two `hir` producers that still store the
/// abandoned `(reason, Span)` traceback form (`hir/checker`, `hir/comptime`) to
/// `Option<Span>`.  New code must store `Option<Span>` at construction; running a
/// real position through this helper *destroys* it (see the ambiguity above).
pub(crate) fn maybe_span(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

/// Converts a [`DiagnosticKind`] into a human-readable message and a set of
/// labels/annotations for source-context rendering.
///
/// This is the Ponent equivalent of Vale's `*ErrorHumanizer` objects — each
/// variant knows how to format itself precisely.
pub trait Humanizer<'src> {
    /// Produce the primary error message — as DATA (`DiagMessage`):
    /// static text is borrowed, parameterized text is a Fluent slug.
    /// The string is materialized only at the rendering boundary
    /// (`Diagnostic::humanize` / renderers), never at construction.
    fn message(&self) -> DiagMessage<'src>;

    /// Produce labels (annotations) for source-context rendering.
    ///
    /// Returns the same inline-capacity container [`Diagnostic::labels`]
    /// stores, so `humanize` moves it straight in: a single-label kind —
    /// the overwhelming majority — never touches the heap, and a two-label
    /// kind skips the intermediate `Vec` allocation the old `Vec` return
    /// forced.  (The inline capacity stays 1: widening it to 2 grows every
    /// `Diagnostic` past its 384-byte size guard.)
    ///
    /// **Labels are resolved eagerly**, against the `Lang` active at call time
    /// (`Label` carries text, not a slug), while messages stay lazy.  That is the
    /// existing design, not drift: `humanize` runs immediately before render, so
    /// the window is small — but a caller that switches `Lang` in between sees
    /// the labels in the old language.
    fn labels(&self) -> CovariantVec<'src, Label<'src>, 1>;

    /// Optional help text.
    fn help(&self) -> Option<Cow<'src, str>> {
        None
    }

    /// Structured suggestions — the lazy twin of `humanize_into`'s fourth tuple
    /// slot.
    ///
    /// **No variant overrides this today**: `NameNotFound` and friends put their
    /// advice in `help()` (a `DiagMessage`), so both humanize paths return an
    /// empty vector and `Diagnostic::suggestions` is populated by the caller, not
    /// by a kind.  Kept as the documented extension point (a future variant with
    /// structured data — import paths, similar-identifier spans — overrides here
    /// *and* in `humanize_into`, which `humanizer_and_humanize_into_agree` pins
    /// equal) rather than deleted and re-added: it costs one empty `Vec` (no
    /// allocation) and keeps the two humanize paths symmetric.
    fn suggestions(&self) -> Vec<Cow<'src, str>> {
        vec![]
    }
}

/// Localized message for a comptime failure.
///
/// Shared by the borrowing [`Humanizer::message`] path and the consuming
/// `humanize_into` path so the two cannot drift, and routed through Fluent so
/// the most user-visible error family follows the active language instead of
/// emitting English unconditionally.
fn comptime_message<'a>(kind: &ComptimeErrorKind) -> DiagMessage<'a> {
    match kind {
        ComptimeErrorKind::StepLimitExceeded => DiagMessage::Slug {
            slug: "comptime-step-limit",
            pairs: smallvec![],
        },
        ComptimeErrorKind::DivisionByZero => DiagMessage::Slug {
            slug: "comptime-division-by-zero",
            pairs: smallvec![],
        },
        ComptimeErrorKind::Overflow => DiagMessage::Slug {
            slug: "comptime-overflow",
            pairs: smallvec![],
        },
        // `Cow::Borrowed(&**e)` would remove this allocation, but it cannot type-check:
        // `Humanizer::message(&self) -> DiagMessage<'src>` returns data tied to `'src`
        // while `&**e` borrows from the `Arc` the kind owns, i.e. from `&self`'s
        // anonymous lifetime.  Making it borrow requires `&'src self` and the
        // `OnceLock`-per-field redesign — a trait-shape change, not a local edit.  What
        // *was* redundant is gone: `pair` caps in place now (`capped_payload` truncates an
        // `Owned` payload without re-collecting), so this is one allocation, not two.
        ComptimeErrorKind::TypeError(e) => DiagMessage::Slug {
            slug: "comptime-type-error",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::AssertionFailed(e) => DiagMessage::Slug {
            slug: "comptime-assertion-failed",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::UnknownIdentifier(e) => DiagMessage::Slug {
            slug: "comptime-unknown-identifier",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::NotComptimeAllowed(e) => DiagMessage::Slug {
            slug: "comptime-not-allowed",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::Deferred => DiagMessage::Slug {
            slug: "comptime-deferred",
            pairs: smallvec![],
        },
        ComptimeErrorKind::SandboxViolation(e) => DiagMessage::Slug {
            slug: "comptime-sandbox-violation",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::MemoryLimitExceeded(e) => DiagMessage::Slug {
            slug: "comptime-memory-limit",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::Internal(e) => DiagMessage::Slug {
            slug: "comptime-internal",
            pairs: smallvec![pair("e", Cow::Owned(e.to_string()))],
        },
    }
}

/// [`name_not_found_help`] for the **consuming** path.
///
/// `humanize_into` owns the `SmallVec` and has no other consumer for it, yet it
/// passed `&suggestions`, so `join_with_or` deep-cloned every `String` — on the
/// emitter path, in the same function whose doc promises "one clone instead of
/// two".  Taking the vec by value lets `join_with_or` move them.
/// How many "did you mean" candidates reach the help line.
///
/// Suggestion *count* and suggestion *length* were both unbounded: N suggestions
/// meant N Fluent renders (N−1 buffer trades) and a help line of unbounded width
/// flowing into every renderer.  rustc caps its list for the same reason.  Three is
/// what a reader can act on.
///
/// Deliberately **unmarked**: unlike `MAX_PAYLOAD_CHARS` (a bounded `…` per payload) and
/// `MAX_TRACEBACK_FRAMES` (a trailing bullet line), this list is the tail of a "did you
/// mean X or Y" sentence, where a marker after the conjunction reads as truncated text
/// rather than a truncated list.  The cap is stated in the doc instead of drawn in the
/// output; if a caller ever needs the "more candidates exist" signal, it belongs in the
/// structured suggestion list, not in this sentence.
const MAX_SUGGESTIONS_SHOWN: usize = 3;

fn name_not_found_help_owned<'src>(
    suggestions: SmallVec<[Cow<'src, str>; 2]>,
) -> Option<Cow<'src, str>> {
    let mut it = suggestions
        .into_iter()
        .take(MAX_SUGGESTIONS_SHOWN)
        .map(capped_payload);
    let first = it.next()?;
    let joined = crate::diagnostics::i18n::join_with_or(first, it);
    Some(crate::diagnostics::i18n::fluent::did_you_mean(joined).lift())
}

fn name_not_found_help<'src>(suggestions: &[Cow<'src, str>]) -> Option<Cow<'src, str>> {
    let shown = &suggestions[..suggestions.len().min(MAX_SUGGESTIONS_SHOWN)];
    let (first, rest) = shown.split_first()?;
    // By value: `first` is a temporary, and `&capped_payload(first.clone())` copied an
    // over-cap owned payload twice — the capped `String` and then again into
    // `join_with_or` — while the original was never used again.  The owned twin already
    // does this.
    let first = capped_payload(first.clone());
    // The conjunction is language-specific: the old `join("` or `")` baked
    // English "or" into every locale's output.
    let joined = crate::diagnostics::i18n::join_with_or(
        first,
        rest.iter().map(|s| capped_payload(s.clone())),
    );
    Some(crate::diagnostics::i18n::fluent::did_you_mean(joined).lift())
}

/// Hard cap on a type/identifier payload rendered into a message or a label.
///
/// Every other unbounded surface in this module has a budget — traceback frames,
/// chain depth, chain entries — but a macro-generated (or adversarial) type name
/// flowed verbatim into the message pairs, the labels *and* the rendered buffer.
/// 512 chars is far past any real name and still bounds the worst-case renderer
/// work; the marker matches the traceback's `…` convention.
const MAX_PAYLOAD_CHARS: usize = 512;

/// Apply [`MAX_PAYLOAD_CHARS`], borrowing when the payload already fits.
fn capped_payload<'a>(text: Cow<'a, str>) -> Cow<'a, str> {
    // `char_count <= byte_count`, so a payload of at most `MAX_PAYLOAD_CHARS` **bytes**
    // cannot exceed the cap: an exact O(1) short-circuit that skips the char walk for
    // every ASCII payload, i.e. the common case.  The walk below stays for multi-byte
    // payloads, where bytes overcount chars.
    if text.len() <= MAX_PAYLOAD_CHARS {
        return text;
    }
    // One walk, not two: `char_indices().nth` yields the byte offset of the first char
    // past the cap (a char boundary by construction), so the prefix is a plain slice.
    // The old shape walked once to test and again to collect, and re-collected even for
    // an `Owned` payload that could be truncated in place.
    let Some((cut, _)) = text.char_indices().nth(MAX_PAYLOAD_CHARS) else {
        return text;
    };
    match text {
        Cow::Owned(mut owned) => {
            owned.truncate(cut);
            owned.push('…');
            Cow::Owned(owned)
        }
        Cow::Borrowed(borrowed) => {
            let mut out = String::with_capacity(cut + '…'.len_utf8());
            out.push_str(&borrowed[..cut]);
            out.push('…');
            Cow::Owned(out)
        }
    }
}

/// Cap a message pair's value at [`MAX_PAYLOAD_CHARS`].
///
/// One enforcement point for every variant.  The cap's own doc promised "the message
/// pairs, the labels *and* the rendered buffer", but only `TypeMismatch` was wired:
/// six other families — plus every comptime payload, the most attacker-influenced text
/// in this module — flowed verbatim into Fluent and the render buffer.
#[inline]
fn pair<'src>(key: &'static str, value: Cow<'src, str>) -> (&'static str, Cow<'src, str>) {
    (key, capped_payload(value))
}

/// Build the `TypeMismatch` message.
///
/// `found` is taken **by value**: the consuming `humanize_into` path moves it in
/// unless a `found_span` label also needs it, which is the only case where the
/// old `&found` form's clone was actually required.  The borrowing
/// `Humanizer::message` path clones once, as it must.
fn type_mismatch_message<'src>(
    expected: &Cow<'src, str>,
    found: Cow<'src, str>,
    // By value: `humanize_into` owns the `Cow` and never uses it again, so
    // passing a reference forced a clone of an owned reason (a `String` copy for
    // runtime-composed subtype explanations).  The borrowing path clones once,
    // which is its minimum.
    reason: Option<Cow<'src, str>>,
    context: Option<TypeCtx>,
) -> DiagMessage<'src> {
    // The suffix is composed by Fluent MESSAGE REFERENCES, not by appending
    // English punctuation here: the em dash and the parenthetical are
    // language-specific, and the old imperative append path also resolved the
    // language at construction time instead of at the render boundary (every
    // other variant returns a `Slug`).
    // One predicate, not two: `TypeCtx::label()` already returns `""` for
    // `Unspecified`, so the explicit `matches!` filter duplicated that rule —
    // exactly the two-sources-of-truth shape this module keeps removing.
    // Decide on the **variant**, not on an empty string: `cached_label_static!`
    // explicitly permits empty labels for padding variants, so a lost translation of
    // e.g. `type-ctx-field` would have silently downgraded the message from
    // `type-mismatch-ctx` to `type-mismatch` and dropped the pair.  The
    // non-emptiness of every real variant is pinned per locale by
    // `type_ctx_labels_are_localized_everywhere`.
    let ctx_label = match context {
        None | Some(TypeCtx::Unspecified) => None,
        Some(c) => Some(c.label()),
    };
    let reason = reason.filter(|r| !r.is_empty());
    let slug = match (ctx_label.is_some(), reason.is_some()) {
        (false, false) => "type-mismatch",
        (false, true) => "type-mismatch-reason",
        (true, false) => "type-mismatch-ctx",
        (true, true) => "type-mismatch-ctx-reason",
    };
    // Budgeted: the only unbounded payloads that reach a rendered message.
    let mut pairs = smallvec![
        ("expected", capped_payload(expected.clone())),
        ("found", capped_payload(found)),
    ];
    if let Some(label) = ctx_label {
        // `TypeCtx::label()` returns `&'static str`; the message slug wants a
        // `Cow`, and borrowing is free (no copy, no refcount).
        pairs.push(("ctx", Cow::Borrowed(label)));
    }
    if let Some(r) = reason {
        pairs.push(("reason", capped_payload(r)));
    }
    DiagMessage::Slug { slug, pairs }
}

/// Frames rendered from a comptime traceback — message lines and secondary
/// labels alike.
///
/// Comptime recursion depth is recursion- or attacker-controlled, so an
/// unbounded traceback produces hundreds of message lines *and* hundreds of
/// secondary labels for the renderer to lay out.  Every other unbounded surface
/// in this crate has a budget (`cap_char_prefixes`, the size-guard tests); this
/// is the traceback's.
const MAX_TRACEBACK_FRAMES: usize = 16;

/// Secondary labels for a comptime traceback: run-deduped by span, capped like
/// the message path, and **marked** when the cap cut something.
///
/// Shared by both humanize arms so the two cannot drift (they had: the consuming
/// arm once lost the dedup entirely).  The truncation marker mirrors the message
/// path's ` …` terminator — without it the renderer showed 16 underlines that just
/// stopped while the message directly above said "…", i.e. the two surfaces
/// disagreed about completeness.
fn comptime_traceback_labels<'src>(
    span: Span,
    traceback: &[(ComptimeReason, Option<Span>)],
) -> crate::diagnostics::CovariantVec<'src, Label<'src>, 1> {
    let mut labels: crate::diagnostics::CovariantVec<'src, Label<'src>, 1> =
        crate::diagnostics::CovariantVec::new();
    labels.push(Label::new(
        span,
        crate::diagnostics::i18n::fluent::label_comptime_eval_error().lift(),
    ));
    // Dedup key is the span alone — one underline cannot show two texts — and it
    // is a **set**, not a run: `[fnDef@(7,9), Assertion@(2,3), fnDef@(7,9)]` is a
    // legitimate recursive traceback shape, and a run-based check emitted two
    // identical underlines at (7,9) for it.  The message path keeps its
    // `(reason, span)` run policy because a *line* can carry two reasons; that
    // asymmetry is deliberate and documented there.
    //
    // Frames located at the primary span are excluded: a secondary underline under
    // the primary annotation is noise, and the message still lists them.
    // Inline capacity = the frame cap (`MAX_TRACEBACK_FRAMES` = 16 slots): the buffer only spills
    // once a traceback carries more distinct frames than the cap allows, i.e. never on input the cap
    // accepts.  Derive the capacity from the constant — the literal `4` this comment used to cite
    // predated the cap and would have made every recursive traceback spill.
    let mut emitted_spans: crate::smallvec::SmallVec<[Span; MAX_TRACEBACK_FRAMES]> =
        crate::smallvec::SmallVec::new();
    let mut emitted = 0usize;
    let mut cut = false;
    for (reason, tb_span) in traceback {
        let Some(tb) = *tb_span else {
            // `None` frames have no position; message-only, and transparent to the
            // set check (they never occupy a span).
            continue;
        };
        if tb == span || emitted_spans.contains(&tb) {
            continue;
        }
        // Dedup before the cap, exactly as the message path does: a duplicate
        // costs no budget and must not produce a marker for output not omitted.
        if emitted == MAX_TRACEBACK_FRAMES {
            cut = true;
            break;
        }
        emitted_spans.push(tb);
        emitted += 1;
        labels.push(Label::secondary(tb, reason.as_str()));
    }
    if cut && let Some(last) = labels.last().map(|l| l.span) {
        labels.push(Label::secondary(
            last,
            format!(
                "{} …",
                crate::diagnostics::i18n::fluent::comptime_frame_bullet()
            ),
        ));
    }
    labels
}

/// Build a `Comptime` message, appending the call stack when there is one.
///
/// **Language contract (asymmetric on purpose).**  With an empty traceback this
/// returns a lazy [`DiagMessage`] slug, resolved against the active `Lang` at
/// *render* time like every other variant.  With a non-empty traceback the base
/// message is rendered **now** (it is concatenated with the frame lines, and
/// `DiagMessage` has no composed form), so it freezes whatever `Lang` was active
/// at `humanize` time.  A caller that switches `Lang` between `humanize` and
/// render therefore sees mixed-language output for this one shape.  The frame
/// labels are eager for the same reason (see `Humanizer::labels`).
fn comptime_message_with_traceback<'src>(
    kind: &ComptimeErrorKind,
    traceback: &[(ComptimeReason, Option<Span>)],
) -> DiagMessage<'src> {
    let msg = comptime_message(kind);
    if traceback.is_empty() {
        return msg;
    }
    // **Lazy composition.**  This used to render the base message here and append a
    // pre-built header and frame list, which froze every piece in whichever language
    // was active at *construction* — the module's one mixed-language window (the base
    // message could be English while the frames, rendered a moment later, were
    // Russian).  `DiagMessage::Joined` defers all of it to the render boundary, so a
    // mid-session language switch reproduces the whole block in the new language.
    //
    // The layout pieces (`"\n\n"`, the per-frame `"\n"` separator, the two-space
    // indent) stay static because they are layout, not words; the bullet is
    // locale-owned and is fetched here for the same reason it cannot live inside the
    // Fluent templates (pattern edges are trimmed).
    let bullet = crate::diagnostics::i18n::fluent::comptime_frame_bullet();
    // Four layout parts up front, five per frame (`"\n"`, indent, bullet, gap, slug) and
    // up to four for the truncation marker.  The old `4 + frames` under-provisioned by
    // ~3x, so a 16-frame traceback regrew this ~130-byte-element vector twice and then
    // shrank it again in `into_boxed_slice`.
    let frames_cap = traceback.len().min(MAX_TRACEBACK_FRAMES);
    // Derived, not hand-summed: the three leading layout parts plus `msg` plus the per-frame
    // five plus the four-part truncation marker.  The old `8 + 5*frames` was *one* part short at
    // the cap (87 vs 88), so the exact-fit claim held only by luck — and the next added part
    // (a localized separator, say) would have reintroduced a silent reallocation.
    const PARTS_BEFORE_FRAMES: usize = 4; // msg, blank line, header, leading newline
    const PARTS_PER_FRAME: usize = 5; // newline, indent, bullet, gap, slug
    const PARTS_FOR_MARKER: usize = 4;
    let mut parts: Vec<DiagMessage<'src>> =
        Vec::with_capacity(PARTS_BEFORE_FRAMES + PARTS_PER_FRAME * frames_cap + PARTS_FOR_MARKER);
    parts.push(msg);
    parts.push(DiagMessage::Static(Cow::Borrowed("\n\n")));
    parts.push(DiagMessage::Slug {
        slug: "comptime-call-stack-header",
        pairs: SmallVec::new(),
    });
    parts.push(DiagMessage::Static(Cow::Borrowed("\n")));
    // Dedup key is `(reason, span)`, not the span alone: comptime recursion produces
    // runs of *identical* frames and collapsing those is the point, while two
    // different reasons at one span (macro expansion can share a span) are distinct
    // context and both belong in the message.  The labels path below collapses by
    // span alone, because one underline cannot show two texts — that asymmetry is
    // deliberate, not drift.
    //
    // `None` frames are emitted as a visible line here, so they legitimately
    // interrupt a run; the labels path skips them *before* the run check, making
    // them transparent there.  Each behaviour matches what its surface shows.
    let mut emitted = 0usize;
    let mut prev: Option<(ComptimeReason, Option<Span>)> = None;
    // Frames actually printed, so the truncation marker can ask "was anything *new*
    // omitted?" instead of "did we stop before the end?".
    let mut shown: SmallVec<[(ComptimeReason, Option<Span>); MAX_TRACEBACK_FRAMES]> =
        SmallVec::new();
    let mut truncated = false;
    for (idx, (reason, span)) in traceback.iter().enumerate() {
        // Dedup **before** the cap: a frame identical to the last emitted one would be
        // skipped anyway, so counting it against the budget both wasted a slot and —
        // when it was the only remaining frame — printed a truncation marker for
        // output that omitted nothing.
        if prev == Some((*reason, *span)) {
            continue;
        }
        if emitted == MAX_TRACEBACK_FRAMES {
            // Mark only when something genuinely **new** was omitted.  A *non-adjacent*
            // repeat of an already-shown frame adds no information, and marking it made
            // the message claim unseen output that the labels surface — which set-dedups
            // before the cap — does not corroborate.  (A new reason at an already-shown
            // span still marks here and not there; that is the documented
            // span-vs-`(reason, span)` asymmetry's shadow, not drift.)
            truncated = traceback[idx..]
                .iter()
                .any(|frame| prev != Some(*frame) && !shown.contains(frame));
            break;
        }
        prev = Some((*reason, *span));
        shown.push((*reason, *span));
        if emitted > 0 {
            parts.push(DiagMessage::Static(Cow::Borrowed("\n")));
        }
        emitted += 1;
        // Indent is layout; the marker itself is locale-owned.  Three static parts
        // instead of one `format!`: the bullet comes from the cached zero-arg table, so
        // its `Clone` is a pointer copy, and this removes up to 16 small allocations per
        // traceback message.
        parts.push(DiagMessage::Static(Cow::Borrowed("  ")));
        parts.push(DiagMessage::Static(bullet.clone()));
        parts.push(DiagMessage::Static(Cow::Borrowed(" ")));
        parts.push(match span {
            Some(span) => DiagMessage::Slug {
                slug: "comptime-frame-offset",
                pairs: smallvec::smallvec![
                    pair("reason", Cow::Borrowed(reason.as_str())),
                    pair("offset", Cow::Owned(span.start.to_string())),
                ],
            },
            None => DiagMessage::Slug {
                slug: "comptime-frame",
                pairs: smallvec::smallvec![pair("reason", Cow::Borrowed(reason.as_str()))],
            },
        });
    }
    if truncated {
        // Pure punctuation plus the locale's marker, so the truncation needs no
        // translated words and cannot drift from the locales.
        parts.push(DiagMessage::Static(Cow::Borrowed("\n")));
        parts.push(DiagMessage::Static(Cow::Borrowed("  ")));
        parts.push(DiagMessage::Static(bullet.clone()));
        parts.push(DiagMessage::Static(Cow::Borrowed(" …")));
    }
    DiagMessage::joined(parts)
}

impl<'src> Humanizer<'src> for DiagnosticKind<'src> {
    fn message(&self) -> DiagMessage<'src> {
        match self {
            DiagnosticKind::TypeMismatch {
                expected,
                found,
                reason,
                context,
                ..
            } => type_mismatch_message(expected, found.clone(), reason.clone(), *context),
            DiagnosticKind::NoSuchField {
                field_name,
                type_name,
                ..
            } => DiagMessage::Slug {
                slug: "field-not-found",
                pairs: smallvec![
                    pair("field", field_name.clone()),
                    pair("type", type_name.clone()),
                ],
            },
            DiagnosticKind::ArgumentTypeMismatch {
                callee,
                param_name,
                expected,
                found,
                ..
            } => DiagMessage::Slug {
                slug: "argument-type-mismatch",
                pairs: smallvec![
                    pair("callee", callee.clone()),
                    pair("param", param_name.clone()),
                    pair("expected", expected.clone()),
                    pair("found", found.clone()),
                ],
            },
            DiagnosticKind::NameNotFound { name, .. } => DiagMessage::Slug {
                slug: "name-not-found",
                pairs: smallvec![pair("name", name.clone())],
            },
            DiagnosticKind::DuplicateDefinition { name, .. } => DiagMessage::Slug {
                slug: "duplicate-definition",
                pairs: smallvec![pair("name", name.clone())],
            },
            DiagnosticKind::ContractNonBool { clause, found, .. } => DiagMessage::Slug {
                slug: "contract-non-bool",
                pairs: smallvec![pair("clause", clause.clone()), pair("found", found.clone())],
            },
            DiagnosticKind::ImplMissingMethod {
                trait_name,
                method_name,
                ..
            } => DiagMessage::Slug {
                slug: "impl-missing-method",
                pairs: smallvec![
                    pair("trait", trait_name.clone()),
                    pair("method", method_name.clone()),
                ],
            },
            DiagnosticKind::Comptime {
                kind, traceback, ..
            } => comptime_message_with_traceback(kind, traceback),
        }
    }

    fn labels(&self) -> CovariantVec<'src, Label<'src>, 1> {
        match self {
            DiagnosticKind::TypeMismatch {
                span,
                found_span,
                expected,
                found,
                ..
            } => {
                let mut labels = CovariantVec::new();
                // Pass the `Cow` straight through: the generated wrapper is
                // lifetime-parametric, so a borrowed type name stays borrowed
                // instead of being formatted into a fresh `String`.
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_expected(capped_payload(
                        expected.clone(),
                    ))
                    .lift(),
                ));
                if let Some(fs) = found_span {
                    // `found` is already a `Cow`: a borrowed type name
                    // clones as a pointer copy with no `format!` buffer.
                    // (An owned one still copies, exactly as before.)
                    labels.push(Label::secondary(*fs, capped_payload(found.clone())));
                }
                labels
            }
            DiagnosticKind::NoSuchField { span, .. } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_field_not_found().lift(),
                ));
                labels
            }
            DiagnosticKind::ArgumentTypeMismatch {
                span,
                param_span,
                expected,
                found,
                ..
            } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_expected_found(
                        capped_payload(expected.clone()),
                        capped_payload(found.clone()),
                    )
                    .lift(),
                ));
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(
                        *ps,
                        crate::diagnostics::i18n::fluent::label_declared_here(capped_payload(
                            expected.clone(),
                        ))
                        .lift(),
                    ));
                }
                labels
            }
            DiagnosticKind::NameNotFound { span, .. } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_not_in_scope().lift(),
                ));
                labels
            }
            DiagnosticKind::DuplicateDefinition {
                this_span,
                original_span,
                ..
            } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *this_span,
                    crate::diagnostics::i18n::fluent::label_duplicate_definition().lift(),
                ));
                labels.push(Label::secondary(
                    *original_span,
                    crate::diagnostics::i18n::fluent::label_first_defined_here().lift(),
                ));
                labels
            }
            DiagnosticKind::ContractNonBool { span, .. } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_expected_bool().lift(),
                ));
                labels
            }
            DiagnosticKind::ImplMissingMethod {
                impl_span,
                trait_span,
                ..
            } => {
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *impl_span,
                    crate::diagnostics::i18n::fluent::label_method_missing().lift(),
                ));
                labels.push(Label::secondary(
                    *trait_span,
                    crate::diagnostics::i18n::fluent::label_required_by_trait().lift(),
                ));
                labels
            }
            DiagnosticKind::Comptime {
                span, traceback, ..
            } => {
                // One implementation for both arms (dedup, cap, truncation
                // marker); see `comptime_traceback_labels`.
                // Positioned frames become secondary annotations (see the
                // helper's doc for the dedup/cap/marker policy, which the message
                // path mirrors).
                comptime_traceback_labels(*span, traceback)
            }
        }
    }

    fn help(&self) -> Option<Cow<'src, str>> {
        match self {
            DiagnosticKind::TypeMismatch { .. } => {
                Some(crate::diagnostics::i18n::fluent::help_cast_or_change().lift())
            }
            DiagnosticKind::NameNotFound { suggestions, .. } => {
                name_not_found_help(suggestions.as_slice())
            }
            _ => None,
        }
    }
}

impl<'src> DiagnosticKind<'src> {
    /// Consume the kind, producing `(message, labels, help, suggestions)` in one pass.
    ///
    /// **Oracle-only in production.**  `Diagnostic::humanize` uses the borrowing
    /// `Humanizer` methods, so this consuming twin exists to keep the two paths
    /// differentially pinned (`humanizer_and_humanize_into_agree`).  Declaring that status
    /// beats leaving it accidental — and the paragraph below describes what resurrecting
    /// it would buy, not what production does today (it used to lead with the production
    /// framing, which is how a dead function gets "optimised" by mistake).
    ///
    /// **Decision (review D6): keep it as the differential oracle, do not resurrect it.**
    /// Resurrecting it would mean `Diagnostic::humanize` consumes a clone of the kind
    /// (`Arc::make_mut`), which reintroduces the per-emit kind clone the borrowing path
    /// exists to avoid on the *cold* path where it would not pay.  The oracle's value is
    /// that `humanizer_and_humanize_into_agree` compares both shapes field by field, so a
    /// change to either is caught; the cost is one `#[cfg_attr]` and a doc block, not a
    /// maintained production path.
    ///
    /// Why it is kept rather than deleted: shared fields (e.g. `TypeMismatch.expected`,
    /// used by both the message pairs and a label) move into one output and clone into the
    /// other — one clone instead of the two (`message` + `labels` each cloning under
    /// `&self`) that the separate `Humanizer` methods pay.  Paired with
    /// `DiagMessage::render_into_owned`, the Fluent argument map would receive the pairs by
    /// move too, so a `Cow::Owned` pair would skip its `String::clone` entirely.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn humanize_into(
        self,
    ) -> (
        DiagMessage<'src>,
        CovariantVec<'src, Label<'src>, 1>,
        Option<Cow<'src, str>>,
        Vec<Cow<'src, str>>,
    ) {
        use crate::diagnostics::i18n::fluent;
        match self {
            DiagnosticKind::TypeMismatch {
                expected,
                found,
                span,
                found_span,
                reason,
                context,
            } => {
                // `found` is needed twice only when there is a `found_span`
                // label; with no label the old `&found` signature cloned an owned
                // type name that was then dropped unused.
                let found_for_label = found_span.is_some().then(|| found.clone());
                // `reason` is owned here and not used again: move it.
                let message = type_mismatch_message(&expected, found, reason, context);
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    span,
                    fluent::label_expected(capped_payload(expected)).lift(),
                ));
                if let (Some(fs), Some(found)) = (found_span, found_for_label) {
                    labels.push(Label::secondary(fs, capped_payload(found)));
                }
                (
                    message,
                    labels,
                    Some(fluent::help_cast_or_change().lift()),
                    vec![],
                )
            }
            DiagnosticKind::NoSuchField {
                field_name,
                type_name,
                span,
            } => {
                let message: DiagMessage = DiagMessage::Slug {
                    slug: "field-not-found",
                    pairs: smallvec![pair("field", field_name), pair("type", type_name)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_field_not_found().lift()));
                (message, labels, None, vec![])
            }
            DiagnosticKind::ArgumentTypeMismatch {
                callee,
                param_name,
                expected,
                found,
                span,
                param_span,
            } => {
                let message = DiagMessage::Slug {
                    slug: "argument-type-mismatch",
                    pairs: smallvec![
                        pair("callee", callee),
                        pair("param", param_name),
                        pair("expected", expected.clone()),
                        pair("found", found.clone()),
                    ],
                };
                let mut labels = CovariantVec::new();
                // Three consumers of `expected` (message, first label, optional
                // second label); only the *last* may move.  Without a param span
                // the first label is the last, so it takes the value instead of a
                // clone — the common shape for a call-site argument.
                match param_span {
                    Some(ps) => {
                        labels.push(Label::new(
                            span,
                            fluent::label_expected_found(expected.clone(), found).lift(),
                        ));
                        labels.push(Label::secondary(
                            ps,
                            fluent::label_declared_here(expected).lift(),
                        ));
                    }
                    None => {
                        labels.push(Label::new(
                            span,
                            fluent::label_expected_found(expected, found).lift(),
                        ));
                    }
                }
                (message, labels, None, vec![])
            }
            DiagnosticKind::NameNotFound {
                name,
                span,
                suggestions,
            } => {
                let message: DiagMessage = DiagMessage::Slug {
                    slug: "name-not-found",
                    pairs: smallvec![pair("name", name)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_not_in_scope().lift()));
                let help = name_not_found_help_owned(suggestions);
                (message, labels, help, vec![])
            }
            DiagnosticKind::DuplicateDefinition {
                name,
                this_span,
                original_span,
            } => {
                let message = DiagMessage::Slug {
                    slug: "duplicate-definition",
                    pairs: smallvec![pair("name", name)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    this_span,
                    fluent::label_duplicate_definition().lift(),
                ));
                labels.push(Label::secondary(
                    original_span,
                    fluent::label_first_defined_here().lift(),
                ));
                (message, labels, None, vec![])
            }
            DiagnosticKind::ContractNonBool {
                clause,
                found,
                span,
            } => {
                let message = DiagMessage::Slug {
                    slug: "contract-non-bool",
                    pairs: smallvec![pair("clause", clause), pair("found", found)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_expected_bool().lift()));
                (message, labels, None, vec![])
            }
            DiagnosticKind::ImplMissingMethod {
                trait_name,
                method_name,
                impl_span,
                trait_span,
            } => {
                let message = DiagMessage::Slug {
                    slug: "impl-missing-method",
                    pairs: smallvec![pair("trait", trait_name), pair("method", method_name)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(impl_span, fluent::label_method_missing().lift()));
                labels.push(Label::secondary(
                    trait_span,
                    fluent::label_required_by_trait().lift(),
                ));
                (message, labels, None, vec![])
            }
            DiagnosticKind::Comptime {
                kind,
                span,
                traceback,
            } => {
                let message: DiagMessage<'src> = comptime_message_with_traceback(&kind, &traceback);
                // The borrowing arm and this one share `comptime_traceback_labels`
                // — this arm had once lost the dedup entirely (the differential
                // corpus had no consecutive-duplicate case).
                let labels = comptime_traceback_labels(span, &traceback);
                (message, labels, None, vec![])
            }
        }
    }
}

#[cfg(test)]
mod humanizer_differential_tests {
    use super::*;
    use crate::ast::{FileId, Span};

    fn span(a: usize, b: usize) -> Span {
        Span::new(FileId::DUMMY, a as u32, b as u32)
    }

    fn corpus() -> Vec<DiagnosticKind<'static>> {
        vec![
            DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("Int<32>"),
                found: Cow::Borrowed("Bool"),
                span: span(0, 3),
                found_span: Some(span(4, 8)),
                reason: Some(Cow::Borrowed("nope")),
                context: Some(TypeCtx::FunctionArg),
            },
            DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("A"),
                found: Cow::Borrowed("B"),
                span: span(0, 1),
                found_span: None,
                reason: None,
                context: None,
            },
            DiagnosticKind::TypeMismatch {
                expected: Cow::Borrowed("A"),
                found: Cow::Borrowed("B"),
                span: span(0, 1),
                found_span: None,
                reason: None,
                context: Some(TypeCtx::Unspecified),
            },
            DiagnosticKind::NoSuchField {
                field_name: Cow::Borrowed("x"),
                type_name: Cow::Borrowed("T"),
                span: span(2, 3),
            },
            DiagnosticKind::ArgumentTypeMismatch {
                callee: Cow::Borrowed("f"),
                param_name: Cow::Borrowed("a"),
                expected: Cow::Borrowed("Int"),
                found: Cow::Borrowed("Bool"),
                span: span(5, 6),
                param_span: Some(span(7, 8)),
            },
            DiagnosticKind::ArgumentTypeMismatch {
                callee: Cow::Borrowed("f"),
                param_name: Cow::Borrowed("a"),
                expected: Cow::Borrowed("Int"),
                found: Cow::Borrowed("Bool"),
                span: span(5, 6),
                param_span: None,
            },
            DiagnosticKind::NameNotFound {
                name: Cow::Borrowed("n"),
                span: span(0, 1),
                suggestions: smallvec![Cow::Borrowed("m")],
            },
            DiagnosticKind::NameNotFound {
                name: Cow::Borrowed("n"),
                span: span(0, 1),
                suggestions: SmallVec::new(),
            },
            DiagnosticKind::DuplicateDefinition {
                name: Cow::Borrowed("d"),
                this_span: span(0, 1),
                original_span: span(2, 3),
            },
            DiagnosticKind::ContractNonBool {
                clause: Cow::Borrowed("requires"),
                found: Cow::Borrowed("Int"),
                span: span(4, 5),
            },
            DiagnosticKind::ImplMissingMethod {
                trait_name: Cow::Borrowed("Tr"),
                method_name: Cow::Borrowed("m"),
                impl_span: span(0, 1),
                trait_span: span(2, 3),
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("boom".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::Assertion, Some(span(0, 1))),
                    (ComptimeReason::ComptimeBlock, None),
                ],
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::StepLimitExceeded,
                span: span(0, 1),
                traceback: Vec::new(),
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::TypeError("x".into()),
                span: span(0, 1),
                traceback: Vec::new(),
            },
            // Traceback shapes: the differential test's only populated traceback
            // used to have no consecutive-duplicate case, which is exactly how
            // the consuming path lost its run-dedup unnoticed.  Each shape below
            // exercises one branch of the run detectors in both paths.
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("dup".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                ],
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("dup-primary".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::Assertion, Some(span(0, 1))),
                    (ComptimeReason::Assertion, Some(span(0, 1))),
                    (ComptimeReason::ComptimeFnCall, Some(span(4, 6))),
                ],
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("none-interleaved".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                    (ComptimeReason::ComptimeBlock, None),
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                ],
            },
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("non-adjacent".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                    (ComptimeReason::Assertion, Some(span(2, 3))),
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                ],
            },
            // Distinct reasons at one span: the label path collapses them (one
            // underline cannot show two texts) while the message keeps both
            // lines; the asymmetry is pinned rather than accidental.
            DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::AssertionFailed("distinct-reasons".into()),
                span: span(0, 1),
                traceback: vec![
                    (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                    (ComptimeReason::Assertion, Some(span(7, 9))),
                ],
            },
        ]
    }

    fn render(m: DiagMessage<'_>) -> String {
        let mut out = String::new();
        m.render_into_owned(&mut out);
        out
    }

    fn label_key(l: &Label<'_>) -> (usize, usize, String) {
        (
            l.span.start_as_usize(),
            l.span.end_as_usize(),
            l.message.to_string(),
        )
    }

    /// Golden text, not self-consistency: the differential test above compares
    /// two hand-written paths that share every slug constant, so a misspelled
    /// slug or a missing pair renders verbatim in BOTH and still passes.  These
    /// assertions pin the rendered English (punctuation, order, composed
    /// suffix, frame dedup) against fixed expectations.
    #[test]
    fn golden_english_composition() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        // `LangGuard`, not a bare `Lang::set`: a failed assertion must not leave the
        // process-global latch at `En` for every later lock-holding test on this
        // (reused) worker thread.
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);

        let plain = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: None,
            context: None,
        };
        assert_eq!(
            render(plain.message()),
            "type mismatch: expected `int`, found `string`"
        );

        // The suffix punctuation and order live in Fluent
        // (`type-mismatch-ctx-reason`), so this is the real oracle for the
        // composition the old imperative `write!` append got wrong.
        let ctx_label = TypeCtx::ReturnType.label();
        let composed = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: Some("covariant position".into()),
            context: Some(TypeCtx::ReturnType),
        };
        assert_eq!(
            render(composed.message()),
            format!(
                "type mismatch: expected `int`, found `string` ({ctx_label}) — covariant position"
            )
        );

        // Comptime frames: localized frame text plus run-dedup of identical
        // consecutive spans (comptime recursion produces runs of them).
        let repeated = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::TypeError("boom".into()),
            span: span(0, 1),
            traceback: vec![
                (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
            ],
        };
        let out = render(repeated.message());
        let frame = format!("  · {} (offset 7)", ComptimeReason::ComptimeFnDef.as_str());
        assert_eq!(
            out.matches(&frame).count(),
            1,
            "consecutive identical frames must be deduped: {out}"
        );
        assert!(out.contains("comptime call stack"), "{out}");
        // The annotation set must follow the same run-dedup policy as the
        // message, or the renderer gets overlapping underlines the message
        // text does not explain.  Primary label + one collapsed frame.
        assert_eq!(
            repeated.labels().len(),
            2,
            "consecutive identical frames must collapse in labels too: {:?}",
            repeated
                .labels()
                .iter()
                .map(|l| l.message.to_string())
                .collect::<Vec<_>>()
        );
    }

    /// The richest message family — `TypeMismatch` with **both** ctx and reason —
    /// pushes four pairs.  `DiagMessage::Slug.pairs` is `SmallVec<[_; 4]>`, so that
    /// construction must stay inline; the constraint lives in another module and
    /// nothing pinned it, which means a change to `DiagMessage`'s capacity would
    /// silently heap-allocate on the most common rich error.
    #[test]
    fn four_pair_type_mismatch_message_stays_inline() {
        let kind = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: Some("why".into()),
            context: Some(TypeCtx::ReturnType),
        };
        match kind.message() {
            DiagMessage::Slug { pairs, .. } => {
                assert_eq!(pairs.len(), 4);
                assert!(
                    !pairs.spilled(),
                    "the four-pair TypeMismatch must fit the inline buffer"
                );
            }
            other => panic!("expected a slug, got {other:?}"),
        }
    }

    /// Two different reasons at one span: the message keeps both lines (a line can
    /// carry two texts), the labels collapse them to one underline (one underline
    /// cannot).  The asymmetry is documented; nothing pinned the labels half.
    #[test]
    fn distinct_reasons_at_one_span_collapse_in_labels_but_not_in_the_message() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let tb = vec![
            (ComptimeReason::ComptimeFnDef, Some(span(5, 6))),
            (ComptimeReason::ComptimeFnCall, Some(span(5, 6))),
        ];
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::Deferred,
            span: span(0, 1),
            traceback: tb,
        };
        assert_eq!(
            kind.labels().len(),
            2,
            "primary + one collapsed frame span: {:?}",
            kind.labels()
                .iter()
                .map(|l| l.message.to_string())
                .collect::<Vec<_>>()
        );
        let out = render(kind.message());
        assert_eq!(
            out.matches("(offset 5)").count(),
            2,
            "the message keeps both reasons: {out}"
        );
    }

    /// Every `TypeCtx` label must be non-empty in **every** locale, because
    /// `type_mismatch_message` now decides on the variant rather than on an empty
    /// string: an empty `type-ctx-field` translation would otherwise render the
    /// `type-mismatch-ctx` slug with an empty `{$ctx}`.
    #[test]
    fn type_ctx_labels_are_localized_everywhere() {
        use crate::diagnostics::i18n::{Lang, acquire_test_lang_lock};
        let _guard = acquire_test_lang_lock();
        let all = [
            TypeCtx::Annotation,
            TypeCtx::ReturnType,
            TypeCtx::BinOp,
            TypeCtx::FunctionArg,
            TypeCtx::Field,
            TypeCtx::TypeAlias,
            TypeCtx::Contract,
            TypeCtx::Inference,
        ];
        for lang in [Lang::En, Lang::Ru, Lang::Zh] {
            Lang::set(lang);
            for ctx in all {
                let label = ctx.label();
                assert!(
                    !label.is_empty() && !label.contains("type-ctx-"),
                    "{ctx:?} has no {lang:?} label: {label:?}"
                );
            }
            // The padding variant stays empty everywhere, and never produces a pair.
            assert!(TypeCtx::Unspecified.label().is_empty());
        }
        Lang::set(Lang::En);
    }

    /// The payload budget: a pathological type name must not flow into the message
    /// verbatim (the one unbounded surface the traceback/chain caps did not cover).
    #[test]
    fn type_mismatch_payload_is_capped() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let huge = "T".repeat(10_000);
        let kind = DiagnosticKind::TypeMismatch {
            expected: huge.clone().into(),
            found: "int".into(),
            span: span(0, 1),
            found_span: None,
            reason: Some(huge.clone().into()),
            context: None,
        };
        let out = render(kind.message());
        assert!(
            out.chars().count() < MAX_PAYLOAD_CHARS * 3,
            "the rendered message must stay bounded, got {} chars",
            out.chars().count()
        );
        assert!(out.contains('…'), "truncation must be marked: {out:.80}");
        // A normal-length name is untouched (borrowed, no allocation).
        let small = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: None,
            context: None,
        };
        assert_eq!(
            render(small.message()),
            "type mismatch: expected `int`, found `string`"
        );

        // Every *other* family goes through the same cap.  Before it, six of seven
        // families and the whole label path flowed verbatim; the comptime payloads are
        // the most attacker-influenced text in the module (they embed evaluated user
        // expressions).
        let capped_kinds = [
            (
                "field-not-found",
                DiagnosticKind::NoSuchField {
                    field_name: huge.clone().into(),
                    type_name: huge.clone().into(),
                    span: span(0, 1),
                },
            ),
            (
                "argument-type-mismatch",
                DiagnosticKind::ArgumentTypeMismatch {
                    callee: huge.clone().into(),
                    param_name: huge.clone().into(),
                    expected: huge.clone().into(),
                    found: huge.clone().into(),
                    span: span(0, 1),
                    param_span: None,
                },
            ),
            (
                "name-not-found",
                DiagnosticKind::NameNotFound {
                    name: huge.clone().into(),
                    span: span(0, 1),
                    suggestions: SmallVec::new(),
                },
            ),
            (
                "contract-non-bool",
                DiagnosticKind::ContractNonBool {
                    clause: huge.clone().into(),
                    found: huge.clone().into(),
                    span: span(0, 1),
                },
            ),
        ];
        for (slug, kind) in capped_kinds {
            let out = render(kind.message());
            // The bound is per *payload*, not per message: `argument-type-mismatch`
            // carries four of them (callee/param/expected/found), so the worst legal
            // message is ~4 × the cap plus the localized chrome.  Uncapped these were
            // 4 × 10 000 chars.
            assert!(
                out.chars().count() < MAX_PAYLOAD_CHARS * 4 + 512,
                "{slug}: rendered message must stay bounded, got {} chars",
                out.chars().count()
            );
            assert!(out.contains('…'), "{slug}: truncation must be marked");
            // The label path is capped too (`label_expected`/`found`).  A label may
            // interpolate *two* capped payloads (`label_expected_found`), hence 2×cap;
            // uncapped, this label was 20 021 chars.
            for label in kind.labels() {
                assert!(
                    label.message.chars().count() <= MAX_PAYLOAD_CHARS * 2 + 64,
                    "{slug}: label payload unbounded ({})",
                    label.message.chars().count()
                );
            }
        }

        // `NameNotFound`: both dimensions of the suggestion list are bounded — count
        // (at most three Fluent renders) and length (each payload capped).
        let many = (0..64)
            .map(|i| Cow::Owned(format!("candidate{i}{}", "X".repeat(2_000))))
            .collect::<SmallVec<[Cow<'_, str>; 2]>>();
        let help = name_not_found_help(&many).expect("suggestions produce help");
        assert!(
            help.chars().count() < MAX_SUGGESTIONS_SHOWN * (MAX_PAYLOAD_CHARS + 64),
            "help must stay bounded, got {} chars",
            help.chars().count()
        );
        assert_eq!(
            help.matches("candidate").count(),
            MAX_SUGGESTIONS_SHOWN,
            "only {MAX_SUGGESTIONS_SHOWN} suggestions may be rendered: {help:.120}"
        );
        assert!(
            help.matches('…').count() >= 1,
            "each shown suggestion is capped"
        );
        // The owned twin must agree (they render through the same slug).
        assert_eq!(
            name_not_found_help_owned(many.clone()).as_deref(),
            Some(&*help),
            "owned and borrowed help builders must agree"
        );
    }

    /// Frames at the *primary* span are excluded from the labels but kept in the
    /// message — the third dedup rule in `comptime_traceback_labels`, stated on its
    /// doc and pinned here.
    #[test]
    fn frames_at_the_primary_span_are_message_only() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let tb = vec![
            (ComptimeReason::ComptimeFnDef, Some(span(0, 1))),
            (ComptimeReason::ComptimeFnCall, Some(span(4, 5))),
        ];
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::Deferred,
            span: span(0, 1),
            traceback: tb,
        };
        assert_eq!(
            kind.labels().len(),
            2,
            "primary + the one non-primary frame: {:?}",
            kind.labels()
                .iter()
                .map(|l| l.span.start_as_usize())
                .collect::<Vec<_>>()
        );
        let out = render(kind.message());
        assert!(
            out.contains("(offset 0)"),
            "the message still lists the primary-span frame: {out}"
        );
    }

    /// Non-adjacent identical spans must collapse too.
    ///
    /// A run-based dedup (`prev == span`) collapsed the *adjacent* case and let
    /// `[fnDef@(7,9), Assertion@(2,3), fnDef@(7,9)]` emit two identical underlines
    /// at (7,9) — a legitimate recursive-traceback shape, and the exact
    /// "accidental behavior" this module keeps hunting.
    #[test]
    fn non_adjacent_same_span_frames_collapse_to_one_label() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let tb = vec![
            (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
            (ComptimeReason::Assertion, Some(span(2, 3))),
            (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
        ];
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::Deferred,
            span: span(0, 1),
            traceback: tb,
        };
        let labels = kind.labels();
        assert_eq!(
            labels.len(),
            3,
            "primary + two distinct spans, not three underlines: {:?}",
            labels
                .iter()
                .map(|l| (l.span.start_as_usize(), l.span.end_as_usize()))
                .collect::<Vec<_>>()
        );
        // The message path keeps both fnDef lines (it dedups on `(reason, span)`),
        // so this also pins the documented asymmetry.
        let out = render(kind.message());
        assert_eq!(
            out.matches("(offset 7)").count(),
            2,
            "the message keeps both frames: {out}"
        );
    }

    /// `[A@s, None, A@s]`: the `None` frame is a visible message line, so it
    /// interrupts the run there, while the labels path skips it *before* the run
    /// check and therefore collapses the two `A@s` frames into one underline.
    /// Both behaviours are documented; this pins the consequence.
    #[test]
    fn none_frame_interrupts_message_runs_but_is_transparent_in_labels() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _restore = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let tb = vec![
            (ComptimeReason::ComptimeFnDef, Some(span(5, 6))),
            (ComptimeReason::ComptimeFnDef, None),
            (ComptimeReason::ComptimeFnDef, Some(span(5, 6))),
        ];
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::Deferred,
            span: span(0, 1),
            traceback: tb,
        };
        let out = render(kind.message());
        assert_eq!(
            out.matches("comptime def function body").count(),
            3,
            "None-frame interrupts the run: {out}"
        );
        assert_eq!(kind.labels().len(), 2, "primary + one collapsed span");
    }

    /// The whole i18n apparatus is otherwise tested only in English, so a bundle
    /// or locale-plumbing regression is invisible.  One non-En render, asserting it
    /// actually differs and leaks no raw slug.
    #[test]
    fn comptime_message_renders_in_another_locale() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::Overflow,
            span: span(0, 1),
            traceback: vec![(ComptimeReason::ComptimeFnDef, Some(span(2, 3)))],
        };
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
        let en = render(kind.message());
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::Ru);
        let ru = render(kind.message());
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
        assert_ne!(en, ru, "Ru must not fall back to the En text: {ru}");
        assert!(
            !ru.contains("comptime-") && !ru.contains("type-mismatch"),
            "raw slug leaked in Ru: {ru}"
        );
    }

    /// `maybe_span`'s ambiguity is documented ("a real zero-length span at offset
    /// 0 becomes `None`"); pin it so a future "fix" to the sentinel rule is a
    /// deliberate, explicit change rather than a silent one — and so the
    /// helper's information loss is visible to whoever reads the test names.
    #[test]
    fn maybe_span_pins_the_documented_sentinels() {
        let f = crate::ast::FileId::DUMMY;
        // The sentinel maps to `None`.
        assert_eq!(maybe_span(crate::ast::Span::new(f, 0, 0)), None);
        // Everything else is preserved…
        assert_eq!(
            maybe_span(crate::ast::Span::new(f, 0, 1)),
            Some(crate::ast::Span::new(f, 0, 1))
        );
        assert_eq!(
            maybe_span(crate::ast::Span::new(f, 5, 5)),
            Some(crate::ast::Span::new(f, 5, 5))
        );
        // …including a *genuine* zero-width span at offset 0, which is exactly the
        // loss the doc admits: indistinguishable from the sentinel.
        assert_eq!(
            maybe_span(crate::ast::Span::new(crate::ast::FileId::from_raw(7), 0, 0)),
            None,
            "documented loss: a real (0,0) frame is swallowed"
        );
    }

    /// Every `TypeCtx` variant has a label; only `Unspecified` is empty.  The
    /// macro-generated table is indexed by a hand-written variant number, so a
    /// table/variant mismatch should be a test failure rather than a wrong suffix.

    /// Pin the enum's own size: this file leans on inline capacities
    /// (`SmallVec` tracebacks, `CovariantVec<_, 1>` labels) and the crate has a
    /// 384-byte `Diagnostic` guard, but nothing pinned the kind itself — the
    /// largest component.  A future variant with a fat inline field should fail
    /// here rather than silently doubling every `Diagnostic`.
    #[test]
    fn diagnostic_kind_size_is_pinned() {
        // Measured: the `Comptime` arm dominates (a `Vec` + span + the Arc
        // payload tag).  The bound pins today's value minus headroom for a smaller
        // variant — growth means a new inline payload, which multiplies through
        // every `Diagnostic` in flight.
        let actual = std::mem::size_of::<DiagnosticKind<'static>>();
        assert!(
            actual <= 120,
            "DiagnosticKind grew to {actual} bytes (bound 120) — check whether a new \
             variant introduced an inline payload"
        );
    }

    /// Golden coverage for the *content* of every slug family this file renders.
    ///
    /// The differential test (`humanizer_and_humanize_into_agree`) compares two
    /// paths that share every slug constant, so a misspelled or missing slug
    /// renders identical garbage in both and it passes. Goldens are the only real
    /// oracle — and before this test only 3 of 11 comptime slugs and 2 of 4
    /// `TypeMismatch` shapes were ever rendered.
    #[test]
    fn golden_every_comptime_slug_and_the_missing_type_mismatch_shapes() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);

        // --- the two TypeMismatch shapes no golden covered ---
        let reason_only = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: Some("covariant position".into()),
            context: None,
        };
        assert_eq!(
            render(reason_only.message()),
            "type mismatch: expected `int`, found `string` — covariant position"
        );
        let ctx_only = DiagnosticKind::TypeMismatch {
            expected: "int".into(),
            found: "string".into(),
            span: span(0, 1),
            found_span: None,
            reason: None,
            context: Some(TypeCtx::ReturnType),
        };
        let ctx_label = TypeCtx::ReturnType.label();
        assert_eq!(
            render(ctx_only.message()),
            format!("type mismatch: expected `int`, found `string` ({ctx_label})")
        );

        // --- every ComptimeErrorKind variant ---
        let payload = || "x".into();
        let cases: [(ComptimeErrorKind, String); 11] = [
            (
                ComptimeErrorKind::StepLimitExceeded,
                "comptime step limit exceeded (possible infinite loop)".to_string(),
            ),
            (
                ComptimeErrorKind::DivisionByZero,
                "division by zero in comptime expression".to_string(),
            ),
            (
                ComptimeErrorKind::Overflow,
                "integer overflow in comptime expression".to_string(),
            ),
            (
                ComptimeErrorKind::TypeError(payload()),
                "comptime type error: x".to_string(),
            ),
            (
                ComptimeErrorKind::AssertionFailed(payload()),
                "comptime assertion failed: x".to_string(),
            ),
            (
                ComptimeErrorKind::UnknownIdentifier(payload()),
                "unknown identifier in comptime: x".to_string(),
            ),
            (
                ComptimeErrorKind::NotComptimeAllowed(payload()),
                "not allowed in comptime: x".to_string(),
            ),
            (
                ComptimeErrorKind::Deferred,
                "expression cannot be evaluated at compile time".to_string(),
            ),
            (
                ComptimeErrorKind::SandboxViolation(payload()),
                "comptime sandbox violation: x".to_string(),
            ),
            (
                ComptimeErrorKind::MemoryLimitExceeded(payload()),
                "comptime memory limit exceeded: x".to_string(),
            ),
            (
                ComptimeErrorKind::Internal(payload()),
                "internal comptime error: x".to_string(),
            ),
        ];
        for (kind, want) in cases {
            let k = DiagnosticKind::Comptime {
                kind,
                span: span(0, 1),
                traceback: vec![],
            };
            let out = render(k.message());
            assert_eq!(out, want, "comptime slug regression");
            // A slug that is missing from the bundle renders its own message id,
            // so this catches a *future* variant without a new golden line.
            assert!(
                !out.contains("comptime-") && !out.contains("type-mismatch"),
                "raw slug leaked into the message: {out}"
            );
        }

        // --- NameNotFound with three suggestions: the conjunction path ---
        let suggestions: SmallVec<[Cow<'static, str>; 2]> = smallvec![
            Cow::Borrowed("alpha"),
            Cow::Borrowed("beta"),
            Cow::Borrowed("gamma")
        ];
        let kind = DiagnosticKind::NameNotFound {
            name: "deta".into(),
            span: span(0, 4),
            suggestions,
        };
        assert_eq!(render(kind.message()), "name not found: `deta`");
        let help = kind.help().expect("suggestions must yield a help line");
        // `suggestion-or` is `{ $first } or { $next }`, applied pairwise.
        assert_eq!(&*help, "did you mean alpha or beta or gamma?");
    }

    /// Comptime recursion depth is input-controlled, so the traceback is capped
    /// in **both** surfaces (message lines and secondary labels) and the
    /// truncation is marked.  Without the cap a 1000-deep traceback would emit a
    /// thousand message lines and a thousand overlapping underlines.
    #[test]
    fn traceback_is_capped_in_message_and_labels() {
        // These render/read localized labels, so they race any other test
        // that mutates the process-global `Lang` latch without the lock.
        let _lang = crate::diagnostics::i18n::acquire_test_lang_lock();
        let traceback: Vec<(ComptimeReason, Option<Span>)> = (0..40usize)
            .map(|i| (ComptimeReason::ComptimeFnDef, Some(span(i, i + 1))))
            .collect();
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::TypeError("boom".into()),
            span: span(100, 101),
            traceback,
        };
        let out = render(kind.message());
        let marker = crate::diagnostics::i18n::fluent::comptime_frame_bullet();
        let frame_lines = out
            .lines()
            .filter(|l| l.trim_start().starts_with(&*marker))
            .count();
        assert_eq!(
            frame_lines,
            MAX_TRACEBACK_FRAMES + 1,
            "capped frames + one truncation marker: {out}"
        );
        assert!(out.contains('…'), "truncation must be marked: {out}");
        // Primary label + capped *secondary* frames: the cap check runs before
        // each push with `len` starting at 1 (the primary), so pushes stop at
        // `MAX + 1` total.  The old `+ 2` and its "MAX + 1 frames plus the
        // primary" comment were both off by one, which left the one guard on
        // renderer underline counts unable to notice a label leaking past the cap.
        // Exact: one primary + `MAX` positioned frames + the truncation marker.
        // `<=` tolerated a regression that silently *lost* labels, which is the
        // same failure class the old `+ 2` off-by-one caused.
        assert_eq!(
            kind.labels().len(),
            MAX_TRACEBACK_FRAMES + 2,
            "labels must be capped too: {}",
            kind.labels().len()
        );
    }

    /// The message path dedupes on `(reason, span)`: two *different* reasons at
    /// one span are distinct context and both render.  (The labels path collapses
    /// by span alone, because one underline cannot carry two texts — that
    /// asymmetry is intentional.)
    #[test]
    fn distinct_reasons_at_one_span_are_not_deduped() {
        // These render/read localized labels, so they race any other test
        // that mutates the process-global `Lang` latch without the lock.
        let _lang = crate::diagnostics::i18n::acquire_test_lang_lock();
        let kind = DiagnosticKind::Comptime {
            kind: ComptimeErrorKind::TypeError("boom".into()),
            span: span(0, 1),
            traceback: vec![
                (ComptimeReason::ComptimeFnDef, Some(span(7, 9))),
                (ComptimeReason::ComptimeBlock, Some(span(7, 9))),
            ],
        };
        let out = render(kind.message());
        assert!(
            out.contains(ComptimeReason::ComptimeFnDef.as_str()),
            "{out}"
        );
        assert!(
            out.contains(ComptimeReason::ComptimeBlock.as_str()),
            "{out}"
        );
    }

    /// The borrowing `Humanizer` impl and the consuming `humanize_into`
    /// duplicate every variant's formatting with no oracle. Pin them equal for
    /// a corpus covering every variant and the main option combinations.
    /// The `Comptime` message is lazy (slug) without a traceback and eager
    /// (rendered text) with one, because the frames are concatenated and
    /// `DiagMessage` has no composed form.  That asymmetry is the only place a
    /// message freezes the active `Lang`, so pin the *shape* — a future
    /// `DiagMessage` variant could legitimately change it, but not silently.
    #[test]
    fn comptime_message_stays_lazy_with_and_without_a_traceback() {
        // These render/read localized labels, so they race any other test
        // that mutates the process-global `Lang` latch without the lock.
        let _lang = crate::diagnostics::i18n::acquire_test_lang_lock();
        use super::{ComptimeErrorKind, ComptimeReason};

        let span = crate::ast::Span::new(crate::ast::FileId::DUMMY, 0, 1);
        let kind = || ComptimeErrorKind::TypeError(std::sync::Arc::from("Int vs Unit"));

        let plain = DiagnosticKind::Comptime {
            kind: kind(),
            span,
            traceback: Vec::new(),
        };
        assert!(
            matches!(plain.message(), DiagMessage::Slug { .. }),
            "no traceback is a single lazy slug"
        );

        // With frames the composition is `Joined`, **not** a pre-rendered string: the
        // base message used to be rendered here, freezing it in whichever language was
        // active at construction while the header and frames resolved later.  Every
        // part must now resolve at the render boundary.
        let with_frames = DiagnosticKind::Comptime {
            kind: kind(),
            span,
            traceback: vec![
                (ComptimeReason::ComptimeBlock, Some(span)),
                (ComptimeReason::ComptimeFnDef, Some(span)),
            ],
        };
        let DiagMessage::Joined(parts) = with_frames.message() else {
            panic!("a traceback must compose lazily, got a pre-rendered message");
        };
        assert!(
            parts
                .iter()
                .all(|p| !matches!(p, DiagMessage::Static(s) if s.contains("mismatch"))),
            "no part may embed the localized base message: {parts:?}"
        );
        assert!(
            parts
                .iter()
                .any(|p| matches!(p, DiagMessage::Slug { slug, .. } if *slug == "comptime-call-stack-header")),
            "the header must be a slug so it localizes at render time"
        );
        assert_eq!(
            parts
                .iter()
                .filter(|p| matches!(p, DiagMessage::Slug { slug, .. } if *slug == "comptime-frame-offset"))
                .count(),
            2,
            "one lazy slug per frame"
        );

        // A language switch after construction must reproduce the whole block in the
        // new language — the property the eager form could not provide.
        let body = {
            let _en = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
            with_frames.message().to_rendered()
        };
        let rendered_ru = {
            let _ru = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::Ru);
            with_frames.message().to_rendered()
        };
        assert!(
            body.lines().count() > 2 && rendered_ru.lines().count() == body.lines().count(),
            "the frame block must keep its shape across languages"
        );
        assert_ne!(
            body, rendered_ru,
            "a post-construction language switch must change the rendered block"
        );
    }

    /// The truncation marker must mean "content you have not seen", not "we stopped
    /// early".
    ///
    /// The labels surface set-dedups *before* the cap, so a non-adjacent repeat of an
    /// already-shown frame never reaches its cap check.  The message surface run-dedups
    /// (correctly — `distinct_reasons_at_one_span_are_not_deduped` pins that), so a
    /// non-adjacent repeat used to consume budget and then mark truncation for output
    /// that omitted nothing.  Both directions are pinned here, because "…" that the
    /// annotation set does not corroborate is exactly the two-surfaces-disagree bug the
    /// marker exists to prevent.
    #[test]
    fn comptime_marker_reflects_omitted_content_not_truncation_alone() {
        let _lang = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _en = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);
        let span = |i: u32| crate::ast::Span::new(crate::ast::FileId::DUMMY, i, i + 1);
        let distinct = |n: u32| -> Vec<(ComptimeReason, Option<crate::ast::Span>)> {
            (0..n)
                .map(|i| (ComptimeReason::ComptimeBlock, Some(span(i))))
                .collect()
        };
        let kind =
            |traceback: Vec<(ComptimeReason, Option<crate::ast::Span>)>| DiagnosticKind::Comptime {
                kind: ComptimeErrorKind::TypeError(std::sync::Arc::from("Int vs Unit")),
                span: span(0),
                traceback,
            };

        // Cap reached, and the frame past it is a *repeat* of one already shown.
        let mut repeat = distinct(MAX_TRACEBACK_FRAMES as u32);
        repeat.push((ComptimeReason::ComptimeBlock, Some(span(0))));
        let rendered = kind(repeat).message().to_rendered();
        assert!(
            !rendered.contains('…'),
            "marker claims unseen output, but the omitted frame was already shown: {rendered}"
        );

        // Cap reached with genuinely new content: the marker must appear.
        let mut fresh = distinct(MAX_TRACEBACK_FRAMES as u32);
        fresh.push((ComptimeReason::ComptimeBlock, Some(span(999))));
        let rendered = kind(fresh).message().to_rendered();
        assert!(
            rendered.contains('…'),
            "a genuinely omitted frame must be marked: {rendered}"
        );
    }

    /// Exhaustive English golden for the two hand-written slot↔slug mappings.
    ///
    /// The slot number and the slug-array position are aligned only by construction, and
    /// the existing checks (non-empty, no slug leak) pass under any permutation.  A
    /// two-line swap — `Assertion`↔`ComptimeTest`, `BinOp`↔`Field` — produces
    /// wrong-but-plausible labels in every locale and nothing would fail.  Pinning the
    /// rendered English text for **every** variant makes such a swap a named failure.
    #[test]
    fn type_ctx_and_comptime_reason_labels_are_exhaustive_goldens() {
        let _lang = crate::diagnostics::i18n::acquire_test_lang_lock();
        let _en = crate::diagnostics::i18n::LangGuard::new(crate::diagnostics::i18n::Lang::En);

        let ctx: &[(super::TypeCtx, &str)] = &[
            (super::TypeCtx::Unspecified, ""),
            (super::TypeCtx::Annotation, "from type annotation"),
            (super::TypeCtx::ReturnType, "from return type"),
            (super::TypeCtx::BinOp, "from operator"),
            (super::TypeCtx::FunctionArg, "from function argument"),
            (super::TypeCtx::Field, "from field type"),
            (super::TypeCtx::TypeAlias, "from type alias"),
            (super::TypeCtx::Contract, "from contract"),
            (super::TypeCtx::Inference, "from type inference"),
        ];
        for (variant, want) in ctx {
            assert_eq!(variant.label(), *want, "TypeCtx::{variant:?}");
        }

        let reasons: &[(super::ComptimeReason, &str)] = &[
            (
                super::ComptimeReason::ComptimeBlock,
                "comptime { ... } block",
            ),
            (
                super::ComptimeReason::ComptimeFnDef,
                "comptime def function body",
            ),
            (
                super::ComptimeReason::ComptimeFnCall,
                "comptime function call",
            ),
            (
                super::ComptimeReason::ComptimeTest,
                "@comptime_test function",
            ),
            (super::ComptimeReason::Assertion, "assert!()"),
            (super::ComptimeReason::TypeInfo, "@typeInfo!()"),
            (super::ComptimeReason::LayoutOf, "layout_of!()"),
        ];
        for (variant, want) in reasons {
            assert_eq!(variant.as_str(), *want, "ComptimeReason::{variant:?}");
        }

        // A swap makes two variants share one string; distinctness is the cheap invariant
        // that catches it even if the golden table above drifts with the ftl file.
        for (i, (_, a)) in ctx.iter().enumerate() {
            for (_, b) in &ctx[i + 1..] {
                assert_ne!(a, b, "two TypeCtx labels collide");
            }
        }
        for (i, (_, a)) in reasons.iter().enumerate() {
            for (_, b) in &reasons[i + 1..] {
                assert_ne!(a, b, "two ComptimeReason labels collide");
            }
        }
    }

    #[test]
    fn humanizer_and_humanize_into_agree() {
        let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
        crate::diagnostics::i18n::Lang::set(crate::diagnostics::i18n::Lang::En);
        for (i, k) in corpus().into_iter().enumerate() {
            let lazy_msg = render(k.message());
            let lazy_labels: Vec<_> = k.labels().iter().map(label_key).collect();
            let lazy_help = k.help().map(|c| c.into_owned());
            let lazy_sug: Vec<String> = k.suggestions().iter().map(|c| c.to_string()).collect();

            let (m, labels, help, sug) = k.clone().humanize_into();
            assert_eq!(lazy_msg, render(m), "message drift, corpus #{i}");
            let got: Vec<_> = labels.iter().map(label_key).collect();
            assert_eq!(lazy_labels, got, "labels drift, corpus #{i}");
            assert_eq!(
                lazy_help,
                help.map(|c| c.into_owned()),
                "help drift, corpus #{i}"
            );
            let got_sug: Vec<String> = sug.iter().map(|c| c.to_string()).collect();
            assert_eq!(lazy_sug, got_sug, "suggestions drift, corpus #{i}");
        }
    }
}
