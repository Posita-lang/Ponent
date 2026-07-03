use crate::hir::types::TypeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expectation {
    None,
    HasType(TypeId),
    CastableToType(TypeId),
}

/// Describes what context a type check is happening in,
/// used to produce more precise error messages.
#[derive(Debug, Clone, Copy)]
pub enum TypingContext {
    /// No specific context
    None,
    /// Checking an argument to a function call
    Argument { index: usize, total: usize },
    /// Checking the body of a closure
    ClosureBody,
    /// Checking the condition of an if/while (expression must be boolean)
    Condition,
    /// Checking a field initializer in a struct literal
    StructFieldInit,
    /// Checking the return value of a function
    ReturnValue,
    /// Checking an array/slice index expression (must be integer)
    Index,
}
