use crate::hir::types::TypeId;
use std::sync::Arc;

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
    /// A string literal, stored as `Arc<str>` to avoid deep copies on clone.
    String(Arc<str>),
    /// A type value (returned by type factories).
    Type(TypeId),
}
