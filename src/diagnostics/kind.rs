use crate::ast::Span;
use crate::diagnostics::DiagMessage;
use crate::diagnostics::i18n::DisplayArg;
use crate::diagnostics::i18n::LiftCow;
use crate::diagnostics::label::{AnnotationKind, Label};
use crate::smallvec::{SmallVec, smallvec};
use std::borrow::Cow;

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
/// .with_code(ErrCode::new("E030"));
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
        match self {
            ComptimeReason::ComptimeBlock => crate::tr!("comptime-reason-block"),
            ComptimeReason::ComptimeFnDef => crate::tr!("comptime-reason-fndef"),
            ComptimeReason::ComptimeFnCall => crate::tr!("comptime-reason-fncall"),
            ComptimeReason::ComptimeTest => crate::tr!("comptime-reason-test"),
            ComptimeReason::Assertion => crate::tr!("comptime-reason-assert"),
            ComptimeReason::TypeInfo => crate::tr!("comptime-reason-typeinfo"),
            ComptimeReason::LayoutOf => crate::tr!("comptime-reason-layoutof"),
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
    fn labels(&self) -> Vec<Label<'src>>;

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
                // Fluent slug for the base sentence (word order lives in
                // locales/*.ftl; the en entry is verbatim the old
                // format! template).  The context/reason suffixes are
                // NOT part of the .ftl pattern, and Fluent silently
                // drops arguments its template never mentions — so a
                // suffixed message composes the owned form to stay
                // byte-identical with the old output.
                let has_suffix =
                    reason.is_some() || context.is_some_and(|c| !matches!(c, TypeCtx::Unspecified));
                if has_suffix {
                    let mut msg = format!("type mismatch: expected `{expected}`, found `{found}`");
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
                    let tb_lines: Vec<String> = traceback
                        .iter()
                        .map(|(reason, span)| {
                            let reason_str = reason.as_str();
                            match span {
                                Some(span) => {
                                    format!("  · {reason_str} at offset {}", span.start)
                                }
                                None => format!("  · {reason_str}"),
                            }
                        })
                        .collect();
                    format!(
                        "{msg}\n\ncomptime call stack (most recent first):\n{}",
                        tb_lines.join("\n")
                    )
                    .into()
                }
            }
        }
    }

    fn labels(&self) -> Vec<Label<'src>> {
        match self {
            DiagnosticKind::TypeMismatch {
                span,
                found_span,
                expected,
                found,
                ..
            } => {
                let mut labels = vec![Label::new(
                    *span,
                    crate::tr!("label-expected", expected = DisplayArg(expected)).lift(),
                )];
                if let Some(fs) = found_span {
                    labels.push(Label::secondary(*fs, format!("{found}")));
                }
                labels
            }
            DiagnosticKind::NoSuchField { span, .. } => {
                vec![Label::new(
                    *span,
                    crate::tr!("label-field-not-found").lift(),
                )]
            }
            DiagnosticKind::ArgumentTypeMismatch {
                span,
                param_span,
                expected,
                found,
                ..
            } => {
                let mut labels = vec![Label::new(
                    *span,
                    crate::tr!(
                        "label-expected-found",
                        expected = DisplayArg(expected),
                        found = DisplayArg(found)
                    )
                    .lift(),
                )];
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(
                        *ps,
                        crate::tr!("label-declared-here", expected = DisplayArg(expected))
                            .lift(),
                    ));
                }
                labels
            }
            DiagnosticKind::NameNotFound { span, .. } => {
                vec![Label::new(
                    *span,
                    crate::tr!("label-not-in-scope").lift(),
                )]
            }
            DiagnosticKind::DuplicateDefinition {
                this_span,
                original_span,
                ..
            } => {
                vec![
                    Label::new(
                        *this_span,
                        crate::tr!("label-duplicate-definition").lift(),
                    ),
                    Label::secondary(
                        *original_span,
                        crate::tr!("label-first-defined-here").lift(),
                    ),
                ]
            }
            DiagnosticKind::ContractNonBool { span, .. } => {
                vec![Label::new(
                    *span,
                    crate::tr!("label-expected-bool").lift(),
                )]
            }
            DiagnosticKind::ImplMissingMethod {
                impl_span,
                trait_span,
                ..
            } => {
                vec![
                    Label::new(
                        *impl_span,
                        crate::tr!("label-method-missing").lift(),
                    ),
                    Label::secondary(
                        *trait_span,
                        crate::tr!("label-required-by-trait").lift(),
                    ),
                ]
            }
            DiagnosticKind::Comptime {
                span, traceback, ..
            } => {
                let mut labels = vec![Label::new(
                    *span,
                    crate::tr!("label-comptime-eval-error").lift(),
                )];
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
                Some(crate::tr!("help-cast-or-change").lift())
            }
            DiagnosticKind::NameNotFound { suggestions, .. } => {
                if suggestions.is_empty() {
                    None
                } else {
                    Some(
                        crate::tr!(
                            "did-you-mean",
                            candidates = DisplayArg(suggestions.join("` or `"))
                        )
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
