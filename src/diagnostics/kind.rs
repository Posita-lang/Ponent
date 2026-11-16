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
    pub fn label(&self) -> Cow<'static, str> {
        let variant = match self {
            TypeCtx::Unspecified => return Cow::Borrowed(""),
            TypeCtx::Annotation => 1,
            TypeCtx::ReturnType => 2,
            TypeCtx::BinOp => 3,
            TypeCtx::FunctionArg => 4,
            TypeCtx::Field => 5,
            TypeCtx::TypeAlias => 6,
            TypeCtx::Contract => 7,
            TypeCtx::Inference => 8,
        };
        crate::diagnostics::i18n::cached_label! {
            n = 9,
            variant = variant,
            init = [
                String::new(),
                crate::diagnostics::i18n::fluent::type_ctx_annotation().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_return_type().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_binop().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_function_arg().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_field().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_type_alias().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_contract().into_owned(),
                crate::diagnostics::i18n::fluent::type_ctx_inference().into_owned(),
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
        f.write_str(&self.label())
    }
}

#[derive(Debug, Clone)]
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
    pub fn as_str(&self) -> Cow<'static, str> {
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
        crate::diagnostics::i18n::cached_label! {
            n = 7,
            variant = variant,
            init = [
                crate::diagnostics::i18n::fluent::comptime_reason_block().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_fndef().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_fncall().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_test().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_assert().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_typeinfo().into_owned(),
                crate::diagnostics::i18n::fluent::comptime_reason_layoutof().into_owned(),
            ],
        }
    }
}

/// Specific kinds of comptime evaluation errors.
#[derive(Debug, Clone)]
/// Payload text for comptime diagnostics.
///
/// `Arc<str>` rather than `Cow<'src, str>`: every producer in
/// `hir::comptime` constructs these from an owned `ComptimeError` string, so
/// there is no borrowed source text to preserve.  Threading `'src` here would
/// only add a lifetime parameter that leaks into the query cache
/// (`FnBodyCheck`/`VfsError` diagnostics are `'input`) without eliminating a
/// single allocation, while `Arc` makes the repeated `humanize` clones O(1).
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
pub fn maybe_span(span: Span) -> Option<Span> {
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
    fn labels(&self) -> CovariantVec<'src, Label<'src>, 1>;

    /// Optional help text.
    fn help(&self) -> Option<Cow<'src, str>> {
        None
    }

    /// Optional suggestions.
    fn suggestions(&self) -> Vec<Cow<'src, str>> {
        vec![]
    }
}

/// Localized message for a comptime failure (C4).
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
        ComptimeErrorKind::TypeError(e) => DiagMessage::Slug {
            slug: "comptime-type-error",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::AssertionFailed(e) => DiagMessage::Slug {
            slug: "comptime-assertion-failed",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::UnknownIdentifier(e) => DiagMessage::Slug {
            slug: "comptime-unknown-identifier",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::NotComptimeAllowed(e) => DiagMessage::Slug {
            slug: "comptime-not-allowed",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::Deferred => DiagMessage::Slug {
            slug: "comptime-deferred",
            pairs: smallvec![],
        },
        ComptimeErrorKind::SandboxViolation(e) => DiagMessage::Slug {
            slug: "comptime-sandbox-violation",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::MemoryLimitExceeded(e) => DiagMessage::Slug {
            slug: "comptime-memory-limit",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
        ComptimeErrorKind::Internal(e) => DiagMessage::Slug {
            slug: "comptime-internal",
            pairs: smallvec![("e", Cow::Owned(e.to_string()))],
        },
    }
}

fn name_not_found_help<'src>(
    suggestions: &[Cow<'src, str>],
) -> Option<Cow<'src, str>> {
    if suggestions.is_empty() {
        None
    } else {
        Some(crate::diagnostics::i18n::fluent::did_you_mean(suggestions.join("` or `")).lift())
    }
}

fn type_mismatch_message<'src>(
    expected: &Cow<'src, str>,
    found: &Cow<'src, str>,
    reason: Option<&Cow<'src, str>>,
    context: Option<TypeCtx>,
) -> DiagMessage<'src> {
    let has_suffix =
        reason.is_some() || context.is_some_and(|c| !matches!(c, TypeCtx::Unspecified));
    if !has_suffix {
        return DiagMessage::Slug {
            slug: "type-mismatch",
            pairs: smallvec![("expected", expected.clone()), ("found", found.clone())],
        };
    }
    let mut msg = String::with_capacity(
        expected.as_ref().len()
            + found.as_ref().len()
            + 48
            + context.map_or(0, |_| 24)
            + reason.as_ref().map_or(0, |r| r.len() + 8),
    );
    crate::diagnostics::i18n::render_into(
        "type-mismatch",
        &[("expected", expected.clone()), ("found", found.clone())],
        &mut msg,
    );
    if let Some(ctx) = context
        && !matches!(ctx, TypeCtx::Unspecified)
    {
        let label = ctx.label();
        if !label.is_empty() {
            let _ = write!(msg, " ({label})");
        }
    }
    if let Some(r) = reason {
        let _ = write!(msg, " — {r}");
    }
    msg.into()
}

fn comptime_message_with_traceback<'src>(
    kind: &ComptimeErrorKind,
    traceback: &[(ComptimeReason, Option<Span>)],
) -> DiagMessage<'src> {
    let msg = comptime_message(kind);
    if traceback.is_empty() {
        return msg;
    }
    let mut s = String::with_capacity(64 + traceback.len() * 64);
    msg.render_into_owned(&mut s);
    s.push_str("\n\n");
    s.push_str(&crate::diagnostics::i18n::fluent::comptime_call_stack_header());
    s.push('\n');
    for (i, (reason, span)) in traceback.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        let reason_str = reason.as_str();
        match span {
            Some(span) => {
                let _ = write!(s, "  · {reason_str} at offset {}", span.start);
            }
            None => {
                let _ = write!(s, "  · {reason_str}");
            }
        }
    }
    s.into()
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
            } => type_mismatch_message(expected, found, reason.as_ref(), *context),
            DiagnosticKind::NoSuchField {
                field_name,
                type_name,
                ..
            } => DiagMessage::Slug {
                slug: "field-not-found",
                pairs: smallvec![
                    ("field", field_name.clone()),
                    ("type", type_name.clone()),
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
                    ("callee", callee.clone()),
                    ("param", param_name.clone()),
                    ("expected", expected.clone()),
                    ("found", found.clone()),
                ],
            },
            DiagnosticKind::NameNotFound { name, .. } => DiagMessage::Slug {
                slug: "name-not-found",
                pairs: smallvec![("name", name.clone())],
            },
            DiagnosticKind::DuplicateDefinition { name, .. } => DiagMessage::Slug {
                slug: "duplicate-definition",
                pairs: smallvec![("name", name.clone())],
            },
            DiagnosticKind::ContractNonBool { clause, found, .. } => DiagMessage::Slug {
                slug: "contract-non-bool",
                pairs: smallvec![("clause", clause.clone()), ("found", found.clone())],
            },
            DiagnosticKind::ImplMissingMethod {
                trait_name,
                method_name,
                ..
            } => DiagMessage::Slug {
                slug: "impl-missing-method",
                pairs: smallvec![
                    ("trait", trait_name.clone()),
                    ("method", method_name.clone()),
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
                    crate::diagnostics::i18n::fluent::label_expected(expected.clone()).lift(),
                ));
                if let Some(fs) = found_span {
                    // `found` is already a `Cow`: a borrowed type name
                    // clones as a pointer copy with no `format!` buffer.
                    // (An owned one still copies, exactly as before.)
                    labels.push(Label::secondary(*fs, found.clone()));
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
                        expected.clone(),
                        found.clone(),
                    )
                    .lift(),
                ));
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(
                        *ps,
                        crate::diagnostics::i18n::fluent::label_declared_here(expected.clone())
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
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    *span,
                    crate::diagnostics::i18n::fluent::label_comptime_eval_error().lift(),
                ));
                // Surface each positioned traceback frame as a secondary
                // annotation so the comptime call stack is visible in the
                // source area, not just in the message text.  `None`
                // frames (the old `(0,0)` sentinel) have no position to
                // point at and are message-only.
                for (reason, tb_span) in traceback {
                    let Some(tb) = tb_span else {
                        continue;
                    };
                    if *tb == *span {
                        continue;
                    }
                    labels.push(Label::secondary(*tb, reason.as_str().lift()));
                }
                labels
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

    fn suggestions(&self) -> Vec<Cow<'src, str>> {
        vec![]
    }
}

impl<'src> DiagnosticKind<'src> {
    /// Consume the kind, producing `(message, labels, help, suggestions)`
    /// in one pass.  Shared fields (e.g. `TypeMismatch.expected`, used by
    /// both the message pairs and a label) move into one output and clone
    /// into the other — one clone instead of the two (`message` + `labels`
    /// each cloning under `&self`) that the separate `Humanizer` methods
    /// paid.  Paired with `DiagMessage::render_into_owned`, the Fluent
    /// argument map receives the pairs by move too, so a `Cow::Owned` pair
    /// skips its `String::clone` entirely.
    pub(crate) fn humanize_into(
        self,
    ) -> (
        DiagMessage<'src>,
        CovariantVec<'src, Label<'src>, 1>,
        Option<Cow<'src, str>>,
        Vec<Cow<'src, str>>,
    ) {
        use crate::diagnostics::i18n::LiftCow;
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
                let message = type_mismatch_message(&expected, &found, reason.as_ref(), context);
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_expected(expected).lift()));
                if let Some(fs) = found_span {
                    labels.push(Label::secondary(fs, found));
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
                    pairs: smallvec![("field", field_name), ("type", type_name)],
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
                        ("callee", callee),
                        ("param", param_name),
                        ("expected", expected.clone()),
                        ("found", found.clone()),
                    ],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(
                    span,
                    fluent::label_expected_found(expected.clone(), found).lift(),
                ));
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(
                        ps,
                        fluent::label_declared_here(expected).lift(),
                    ));
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
                    pairs: smallvec![("name", name)],
                };
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_not_in_scope().lift()));
                let help = name_not_found_help(suggestions.as_slice());
                (message, labels, help, vec![])
            }
            DiagnosticKind::DuplicateDefinition {
                name,
                this_span,
                original_span,
            } => {
                let message = DiagMessage::Slug {
                    slug: "duplicate-definition",
                    pairs: smallvec![("name", name)],
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
                    pairs: smallvec![("clause", clause), ("found", found)],
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
                    pairs: smallvec![("trait", trait_name), ("method", method_name)],
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
                let message: DiagMessage<'src> =
                    comptime_message_with_traceback(&kind, &traceback);
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_comptime_eval_error().lift()));
                for (reason, tb_span) in &traceback {
                    let Some(tb) = tb_span else {
                        continue;
                    };
                    if *tb == span {
                        continue;
                    }
                    labels.push(Label::secondary(*tb, reason.as_str().lift()));
                }
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
        ]
    }

    fn render(m: DiagMessage<'_>) -> String {
        let mut out = String::new();
        m.render_into_owned(&mut out);
        out
    }

    fn label_key(l: &Label<'_>) -> (usize, usize, String) {
        (l.span.start_as_usize(), l.span.end_as_usize(), l.message.to_string())
    }

    /// The borrowing `Humanizer` impl and the consuming `humanize_into`
    /// duplicate every variant's formatting with no oracle. Pin them equal for
    /// a corpus covering every variant and the main option combinations.
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
