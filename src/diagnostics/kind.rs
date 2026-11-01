use crate::ast::Span;
use crate::diagnostics::DiagMessage;
use crate::diagnostics::i18n::LiftCow;
use crate::diagnostics::label::{AnnotationKind, Label};
use crate::smallvec::smallvec;
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

impl std::fmt::Display for TypeCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeCtx::Unspecified => Ok(()),
            TypeCtx::Annotation => write!(f, "from type annotation"),
            TypeCtx::ReturnType => write!(f, "from return type"),
            TypeCtx::BinOp => write!(f, "from operator"),
            TypeCtx::FunctionArg => write!(f, "from function argument"),
            TypeCtx::Field => write!(f, "from field type"),
            TypeCtx::TypeAlias => write!(f, "from type alias"),
            TypeCtx::Contract => write!(f, "from contract"),
            TypeCtx::Inference => write!(f, "from type inference"),
        }
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
        suggestions: Vec<String>,
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
        crate::diagnostics::i18n::cached_label! {
            n = 7,
            variant = *self as usize,
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
pub enum ComptimeErrorKind {
    /// Step limit exceeded (possible infinite loop).
    StepLimitExceeded,
    /// Division or remainder by zero.
    DivisionByZero,
    /// Integer overflow (trap policy).
    Overflow,
    /// Type mismatch in a comptime operation.
    TypeError(String),
    /// Assertion failed at compile time.
    AssertionFailed(String),
    /// Unknown identifier in comptime context.
    UnknownIdentifier(String),
    /// A runtime-only construct encountered in comptime context.
    NotComptimeAllowed(String),
    /// The expression cannot be evaluated at compile time.
    Deferred,
    /// A comptime sandbox violation.
    SandboxViolation(String),
    /// Memory limit exceeded during comptime evaluation.
    MemoryLimitExceeded(String),
    /// An internal comptime error.
    Internal(String),
}

/// Map a frame span to the traceback's `Option<Span>`: the abandoned
/// `(0,0)` sentinel becomes `None` (a frame with no source position),
/// while a real zero-length span at offset 0 stays `Some` — same
/// conversion `ChainEntry` made when it moved off the sentinel.
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

impl<'src> Humanizer<'src> for DiagnosticKind<'src> {
    fn message(&self) -> DiagMessage<'src> {
        match self {
            DiagnosticKind::TypeMismatch {
                expected,
                found,
                reason,
                context,
                ..
            } => {
                // Base sentence goes through the `type-mismatch` Fluent slug
                // (word order lives in locales/*.ftl), so it follows the
                // active language.  The context/reason suffixes are
                // structural ASCII (parentheses, em dash) wrapping content
                // that is itself English (TypeCtx::Display, free-form
                // reason), so they are appended verbatim — localizing the
                // punctuation alone would mix scripts mid-sentence.
                let has_suffix =
                    reason.is_some() || context.is_some_and(|c| !matches!(c, TypeCtx::Unspecified));
                if has_suffix {
                    let mut msg = String::new();
                    crate::diagnostics::i18n::render_into(
                        "type-mismatch",
                        &[("expected", expected.clone()), ("found", found.clone())],
                        &mut msg,
                    );
                    if let Some(ctx) = context
                        && !matches!(ctx, TypeCtx::Unspecified)
                    {
                        use std::fmt::Write;
                        let _ = write!(msg, " ({ctx})");
                    }
                    if let Some(r) = reason {
                        use std::fmt::Write;
                        let _ = write!(msg, " — {r}");
                    }
                    msg.into()
                } else {
                    DiagMessage::Slug {
                        slug: "type-mismatch",
                        pairs: smallvec![("expected", expected.clone()), ("found", found.clone()),],
                    }
                }
            }
            DiagnosticKind::NoSuchField {
                field_name,
                type_name,
                ..
            } => format!("no field `{field_name}` on type `{type_name}`").into(),
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
            DiagnosticKind::NameNotFound { name, .. } => format!("name not found: `{name}`").into(),
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
            } => {
                let msg = match kind {
                    ComptimeErrorKind::StepLimitExceeded => DiagMessage::static_str(
                        "comptime step limit exceeded (possible infinite loop)",
                    ),
                    ComptimeErrorKind::DivisionByZero => {
                        DiagMessage::static_str("division by zero in comptime expression")
                    }
                    ComptimeErrorKind::Overflow => {
                        DiagMessage::static_str("integer overflow in comptime expression")
                    }
                    ComptimeErrorKind::TypeError(s) => format!("comptime type error: {s}").into(),
                    ComptimeErrorKind::AssertionFailed(s) => {
                        format!("comptime assertion failed: {s}").into()
                    }
                    ComptimeErrorKind::UnknownIdentifier(s) => {
                        format!("unknown identifier in comptime: {s}").into()
                    }
                    ComptimeErrorKind::NotComptimeAllowed(s) => {
                        format!("not allowed in comptime: {s}").into()
                    }
                    ComptimeErrorKind::Deferred => {
                        DiagMessage::static_str("expression cannot be evaluated at compile time")
                    }
                    ComptimeErrorKind::SandboxViolation(s) => {
                        format!("comptime sandbox violation: {s}").into()
                    }
                    ComptimeErrorKind::MemoryLimitExceeded(s) => {
                        format!("comptime memory limit exceeded: {s}").into()
                    }
                    ComptimeErrorKind::Internal(s) => {
                        format!("internal comptime error: {s}").into()
                    }
                };
                if traceback.is_empty() {
                    msg
                } else {
                    let mut s = String::with_capacity(64 + traceback.len() * 32);
                    msg.render_into(&mut s);
                    s.push_str("\n\ncomptime call stack (most recent first):\n");
                    for (i, (reason, span)) in traceback.iter().enumerate() {
                        if i > 0 {
                            s.push('\n');
                        }
                        let reason_str = reason.as_str();
                        match span {
                            Some(span) => {
                                write!(s, "  · {reason_str} at offset {}", span.start).ok()
                            }
                            None => write!(s, "  · {reason_str}").ok(),
                        };
                    }
                    s.into()
                }
            }
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
                if suggestions.is_empty() {
                    None
                } else {
                    Some(
                        crate::diagnostics::i18n::fluent::did_you_mean(suggestions.join("` or `"))
                            .lift(),
                    )
                }
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
                let has_suffix =
                    reason.is_some() || context.is_some_and(|c| !matches!(c, TypeCtx::Unspecified));
                let message = if has_suffix {
                    let mut msg = String::new();
                    crate::diagnostics::i18n::render_into(
                        "type-mismatch",
                        &[("expected", expected.clone()), ("found", found.clone())],
                        &mut msg,
                    );
                    if let Some(ctx) = context
                        && !matches!(ctx, TypeCtx::Unspecified)
                    {
                        use std::fmt::Write;
                        let _ = write!(msg, " ({ctx})");
                    }
                    if let Some(r) = reason {
                        use std::fmt::Write;
                        let _ = write!(msg, " — {r}");
                    }
                    msg.into()
                } else {
                    DiagMessage::Slug {
                        slug: "type-mismatch",
                        pairs: smallvec![("expected", expected.clone()), ("found", found.clone()),],
                    }
                };
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
                let message: DiagMessage =
                    format!("no field `{field_name}` on type `{type_name}`").into();
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
                let message: DiagMessage = format!("name not found: `{name}`").into();
                let mut labels = CovariantVec::new();
                labels.push(Label::new(span, fluent::label_not_in_scope().lift()));
                let help = if suggestions.is_empty() {
                    None
                } else {
                    Some(fluent::did_you_mean(suggestions.join("` or `")).lift())
                };
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
                let message: DiagMessage = {
                    let msg = match kind {
                        ComptimeErrorKind::StepLimitExceeded => DiagMessage::static_str(
                            "comptime step limit exceeded (possible infinite loop)",
                        ),
                        ComptimeErrorKind::DivisionByZero => {
                            DiagMessage::static_str("division by zero in comptime expression")
                        }
                        ComptimeErrorKind::Overflow => {
                            DiagMessage::static_str("integer overflow in comptime expression")
                        }
                        ComptimeErrorKind::TypeError(s) => {
                            format!("comptime type error: {s}").into()
                        }
                        ComptimeErrorKind::AssertionFailed(s) => {
                            format!("comptime assertion failed: {s}").into()
                        }
                        ComptimeErrorKind::UnknownIdentifier(s) => {
                            format!("unknown identifier in comptime: {s}").into()
                        }
                        ComptimeErrorKind::NotComptimeAllowed(s) => {
                            format!("not allowed in comptime: {s}").into()
                        }
                        ComptimeErrorKind::Deferred => DiagMessage::static_str(
                            "expression cannot be evaluated at compile time",
                        ),
                        ComptimeErrorKind::SandboxViolation(s) => {
                            format!("comptime sandbox violation: {s}").into()
                        }
                        ComptimeErrorKind::MemoryLimitExceeded(s) => {
                            format!("comptime memory limit exceeded: {s}").into()
                        }
                        ComptimeErrorKind::Internal(s) => {
                            format!("internal comptime error: {s}").into()
                        }
                    };
                    if traceback.is_empty() {
                        msg
                    } else {
                        let mut s = String::with_capacity(64 + traceback.len() * 32);
                        msg.render_into(&mut s);
                        s.push_str("\n\ncomptime call stack (most recent first):\n");
                        for (i, (reason, sp)) in traceback.iter().enumerate() {
                            if i > 0 {
                                s.push('\n');
                            }
                            let reason_str = reason.as_str();
                            match sp {
                                Some(sp) => {
                                    write!(s, "  · {reason_str} at offset {}", sp.start).ok()
                                }
                                None => write!(s, "  · {reason_str}").ok(),
                            };
                        }
                        s.into()
                    }
                };
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
