use crate::hir::types::TypeId;

/// The result of evaluating a comptime expression.
#[derive(Debug, Clone)]
pub enum ComptimeValue {
    /// Unit `()` value.
    Unit,
    /// A boolean literal.
    Bool(bool),
    /// An integer literal.
    Int(i128),
    /// A floating-point literal.
    Float(f64),
    /// A string literal.
    String(String),
    /// A type value (returned by type factories).
    Type(TypeId),
}
