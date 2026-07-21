use crate::ast::Span;
use crate::diagnostics::label::{AnnotationKind, Label};

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
#[derive(Debug, Clone)]
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
    fn help(&self) -> Option<String> { None }

    /// Optional suggestions.
    fn suggestions(&self) -> Vec<String> { vec![] }
}

impl Humanizer for DiagnosticKind {
    fn message(&self) -> String {
        match self {
            DiagnosticKind::TypeMismatch { expected, found, reason, .. } => {
                let mut msg = format!("type mismatch: expected `{expected}`, found `{found}`");
                if let Some(r) = reason {
                    use std::fmt::Write;
                    let _ = write!(msg, " — {r}");
                }
                msg
            }
            DiagnosticKind::NoSuchField { field_name, type_name, .. } => {
                format!("no field `{field_name}` on type `{type_name}`")
            }
            DiagnosticKind::ArgumentTypeMismatch { callee, param_name, expected, found, .. } => {
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
                format!(
                    "`{clause}` clause must be boolean, found `{found}`"
                )
            }
            DiagnosticKind::ImplMissingMethod { trait_name, method_name, .. } => {
                format!(
                    "impl of `{trait_name}` is missing method `{method_name}`"
                )
            }
        }
    }

    fn labels(&self) -> Vec<Label> {
        match self {
            DiagnosticKind::TypeMismatch { span, found_span, .. } => {
                let mut labels = vec![
                    Label::new(*span, format!("expected {}", self.expected_str())),
                ];
                if let Some(fs) = found_span {
                    labels.push(Label::secondary(*fs, self.found_str()));
                }
                labels
            }
            DiagnosticKind::NoSuchField { span, .. } => {
                vec![Label::new(*span, "field not found")]
            }
            DiagnosticKind::ArgumentTypeMismatch { span, param_span, expected, found, .. } => {
                let mut labels = vec![
                    Label::new(*span, format!("expected `{expected}`, found `{found}`")),
                ];
                if let Some(ps) = param_span {
                    labels.push(Label::secondary(*ps, format!("`{expected}` declared here")));
                }
                labels
            }
            DiagnosticKind::NameNotFound { span, .. } => {
                vec![Label::new(*span, "not found in this scope")]
            }
            DiagnosticKind::DuplicateDefinition { this_span, original_span, .. } => {
                vec![
                    Label::new(*this_span, "duplicate definition"),
                    Label::secondary(*original_span, "first defined here"),
                ]
            }
            DiagnosticKind::ContractNonBool { span, .. } => {
                vec![Label::new(*span, "expected bool")]
            }
            DiagnosticKind::ImplMissingMethod { impl_span, trait_span, .. } => {
                vec![
                    Label::new(*impl_span, "method missing here"),
                    Label::secondary(*trait_span, "required by trait declaration here"),
                ]
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