use crate::ast::Span;
use crate::diagnostics::label::{AnnotationKind, Label};

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
pub enum DiagnosticKind {
    /// A value of one type was used where another type was expected.
    TypeMismatch {
        expected: String,
        found: String,
        /// The span of the expression with the wrong type.
        span: Span,
        /// Optional span of the value that produced the found type.
        found_span: Option<Span>,
        /// Optional explanation of WHY the types don't match
        /// (e.g. "Int<16> is not a subtype of Int<23>").
        reason: Option<String>,
        /// The context in which the expected type was determined
        /// (e.g. from a type annotation, a function return type, etc.).
        context: Option<TypeCtx>,
    },
    /// A field access referred to a field that doesn't exist on the type.
    NoSuchField {
        field_name: String,
        type_name: String,
        span: Span,
    },
    /// A function/method call had argument type mismatches.
    ArgumentTypeMismatch {
        callee: String,
        param_name: String,
        expected: String,
        found: String,
        span: Span,
        param_span: Option<Span>,
    },
    /// A name could not be resolved in the current scope.
    NameNotFound {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    /// A duplicate definition (variable, function, type).
    DuplicateDefinition {
        name: String,
        this_span: Span,
        original_span: Span,
    },
    /// A contract condition (`requires` / `ensures`) was not boolean.
    ContractNonBool {
        clause: String,
        found: String,
        span: Span,
    },
    /// A trait implementation is missing a required method.
    ImplMissingMethod {
        trait_name: String,
        method_name: String,
        impl_span: Span,
        trait_span: Span,
    },
    /// A compile-time evaluation error (comptime).
    Comptime {
        kind: ComptimeErrorKind,
        span: Span,
        traceback: Vec<(ComptimeReason, Span)>,
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

/// Converts a [`DiagnosticKind`] into a human-readable message and a set of
/// labels/annotations for source-context rendering.
///
/// This is the Ponent equivalent of Vale's `*ErrorHumanizer` objects — each
/// variant knows how to format itself precisely.
pub trait Humanizer {
    /// Produce the primary error message.
    fn message(&self) -> String;

    /// Produce labels (annotations) for source-context rendering.
    fn labels(&self) -> Vec<Label>;

    /// Optional help text.
    fn help(&self) -> Option<String> {
        None
    }

    /// Optional suggestions.
    fn suggestions(&self) -> Vec<String> {
        vec![]
    }
}

impl Humanizer for DiagnosticKind {
    fn message(&self) -> String {
        match self {
            DiagnosticKind::TypeMismatch {
                expected,
                found,
                reason,
                context,
                ..
            } => {
                let mut msg = format!("type mismatch: expected `{expected}`, found `{found}`");
                if let Some(ctx) = context {
                    if !matches!(ctx, TypeCtx::Unspecified) {
                        use std::fmt::Write;
                        let _ = write!(msg, " ({ctx})");
                    }
                }
                if let Some(r) = reason {
                    use std::fmt::Write;
                    let _ = write!(msg, " — {r}");
                }
                msg
            }
            DiagnosticKind::NoSuchField {
                field_name,
                type_name,
                ..
            } => {
                format!("no field `{field_name}` on type `{type_name}`")
            }
            DiagnosticKind::ArgumentTypeMismatch {
                callee,
                param_name,
                expected,
                found,
                ..
            } => {
                format!(
                    "argument type mismatch in call to `{callee}`: \
                     parameter `{param_name}` expected `{expected}`, found `{found}`"
                )
            }
            DiagnosticKind::NameNotFound { name, .. } => {
                format!("name not found: `{name}`")
            }
            DiagnosticKind::DuplicateDefinition { name, .. } => {
                format!("duplicate definition of `{name}`")
            }
            DiagnosticKind::ContractNonBool { clause, found, .. } => {
                format!("`{clause}` clause must be boolean, found `{found}`")
            }
            DiagnosticKind::ImplMissingMethod {
                trait_name,
                method_name,
                ..
            } => {
                format!("impl of `{trait_name}` is missing method `{method_name}`")
            }
            DiagnosticKind::Comptime {
                kind, traceback, ..
            } => {
                let msg = match kind {
                    ComptimeErrorKind::StepLimitExceeded => {
                        "comptime step limit exceeded (possible infinite loop)".into()
                    }
                    ComptimeErrorKind::DivisionByZero => {
                        "division by zero in comptime expression".into()
                    }
                    ComptimeErrorKind::Overflow => "integer overflow in comptime expression".into(),
                    ComptimeErrorKind::TypeError(s) => format!("comptime type error: {s}"),
                    ComptimeErrorKind::AssertionFailed(s) => {
                        format!("comptime assertion failed: {s}")
                    }
                    ComptimeErrorKind::UnknownIdentifier(s) => {
                        format!("unknown identifier in comptime: {s}")
                    }
                    ComptimeErrorKind::NotComptimeAllowed(s) => {
                        format!("not allowed in comptime: {s}")
                    }
                    ComptimeErrorKind::Deferred => {
                        "expression cannot be evaluated at compile time".into()
                    }
                    ComptimeErrorKind::SandboxViolation(s) => {
                        format!("comptime sandbox violation: {s}")
                    }
                    ComptimeErrorKind::MemoryLimitExceeded(s) => {
                        format!("comptime memory limit exceeded: {s}")
                    }
                    ComptimeErrorKind::Internal(s) => format!("internal comptime error: {s}"),
                };
                if traceback.is_empty() {
                    msg
                } else {
                    let tb_lines: Vec<String> = traceback
                        .iter()
                        .map(|(reason, span)| {
                            let reason_str = match reason {
                                ComptimeReason::ComptimeBlock => "comptime { ... } block",
                                ComptimeReason::ComptimeFnDef => "comptime def function body",
                                ComptimeReason::ComptimeFnCall => "comptime function call",
                                ComptimeReason::ComptimeTest => "@comptime_test function",
                                ComptimeReason::Assertion => "assert!()",
                                ComptimeReason::TypeInfo => "@typeInfo!()",
                                ComptimeReason::LayoutOf => "layout_of!()",
                            };
                            if span.start != 0 || span.end != 0 {
                                format!("  · {reason_str} at offset {}", span.start)
                            } else {
                                format!("  · {reason_str}")
                            }
                        })
                        .collect();
                    format!(
                        "{msg}\n\ncomptime call stack (most recent first):\n{}",
                        tb_lines.join("\n")
                    )
                }
            }
        }
    }

    fn labels(&self) -> Vec<Label> {
        match self {
            DiagnosticKind::TypeMismatch {
                span,
                found_span,
                expected,
                found,
                ..
            } => {
                let mut labels = vec![Label::new(*span, format!("expected {expected}"))];
                if let Some(fs) = found_span {
                    labels.push(Label::secondary(*fs, format!("{found}")));
                }
                labels
            }
            DiagnosticKind::NoSuchField { span, .. } => {
                vec![Label::new(*span, "field not found")]
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
                    format!("expected `{expected}`, found `{found}`"),
                )];
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(*ps, format!("`{expected}` declared here")));
                }
                labels
            }
            DiagnosticKind::NameNotFound { span, .. } => {
                vec![Label::new(*span, "not found in this scope")]
            }
            DiagnosticKind::DuplicateDefinition {
                this_span,
                original_span,
                ..
            } => {
                vec![
                    Label::new(*this_span, "duplicate definition"),
                    Label::secondary(*original_span, "first defined here"),
                ]
            }
            DiagnosticKind::ContractNonBool { span, .. } => {
                vec![Label::new(*span, "expected bool")]
            }
            DiagnosticKind::ImplMissingMethod {
                impl_span,
                trait_span,
                ..
            } => {
                vec![
                    Label::new(*impl_span, "method missing here"),
                    Label::secondary(*trait_span, "required by trait declaration here"),
                ]
            }
            DiagnosticKind::Comptime { span, .. } => {
                vec![Label::new(*span, "comptime evaluation error")]
            }
        }
    }

    fn help(&self) -> Option<String> {
        match self {
            DiagnosticKind::TypeMismatch { .. } => {
                Some("try using `as` to cast, or change the expression type".into())
            }
            DiagnosticKind::NameNotFound { suggestions, .. } => {
                if suggestions.is_empty() {
                    None
                } else {
                    Some(format!("did you mean `{}`?", suggestions.join("` or `")))
                }
            }
            _ => None,
        }
    }
}

// ── Private helpers ──────────────────────────────────────────────

impl DiagnosticKind {
    fn expected_str(&self) -> String {
        match self {
            DiagnosticKind::TypeMismatch { expected, .. } => expected.clone(),
            _ => String::new(),
        }
    }

    fn found_str(&self) -> String {
        match self {
            DiagnosticKind::TypeMismatch { found, .. } => found.clone(),
            _ => String::new(),
        }
    }
}
