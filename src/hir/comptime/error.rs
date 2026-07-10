use crate::ast::Span;

/// Errors that can occur during comptime evaluation.
#[derive(Debug, Clone)]
pub enum ComptimeError {
    /// Step limit reached; possible infinite loop.
    StepLimitExceeded,
    /// Division or remainder by zero.
    DivisionByZero,
    /// Integer overflow (trap policy) in a comptime expression.
    Overflow,
    /// Type mismatch in a comptime operation.
    TypeError(String),
    /// Assertion failed at compile time.
    AssertionFailed(String),
    /// Unknown identifier in comptime context.
    UnknownIdentifier(String),
    /// A runtime-only construct encountered in comptime context.
    NotComptimeAllowed(String),
    /// The expression cannot be evaluated at compile time (defer to runtime).
    Deferred,
    /// An internal comptime error.
    Internal(String),
}

impl std::fmt::Display for ComptimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComptimeError::StepLimitExceeded => write!(f, "comptime step limit exceeded (possible infinite loop)"),
            ComptimeError::DivisionByZero => write!(f, "division by zero in comptime expression"),
            ComptimeError::Overflow => write!(f, "integer overflow in comptime expression"),
            ComptimeError::TypeError(msg) => write!(f, "comptime type error: {}", msg),
            ComptimeError::AssertionFailed(msg) => write!(f, "comptime assertion failed: {}", msg),
            ComptimeError::UnknownIdentifier(name) => write!(f, "unknown identifier in comptime: {}", name),
            ComptimeError::NotComptimeAllowed(msg) => write!(f, "{}", msg),
            ComptimeError::Deferred => write!(f, "expression cannot be evaluated at compile time"),
            ComptimeError::Internal(msg) => write!(f, "internal comptime error: {}", msg),
        }
    }
}

impl ComptimeError {
    /// Create a `NotComptimeAllowed` error from a span context.
    pub fn not_allowed(msg: impl Into<String>) -> Self {
        ComptimeError::NotComptimeAllowed(msg.into())
    }

    /// Create a `TypeError` from a message.
    pub fn type_error(msg: impl Into<String>) -> Self {
        ComptimeError::TypeError(msg.into())
    }
}
