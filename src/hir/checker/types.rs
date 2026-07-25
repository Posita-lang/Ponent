use crate::diagnostics::TypeCtx;
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

/// Convert a `TypingContext` to a `TypeCtx` for use in `TypeMismatch` diagnostics.
pub fn typing_context_to_type_ctx(ctx: &TypingContext) -> TypeCtx {
    match ctx {
        TypingContext::None => TypeCtx::Unspecified,
        TypingContext::Argument { .. } => TypeCtx::FunctionArg,
        TypingContext::ClosureBody => TypeCtx::Inference,
        TypingContext::Condition => TypeCtx::Unspecified,
        TypingContext::StructFieldInit => TypeCtx::Field,
        TypingContext::ReturnValue => TypeCtx::ReturnType,
        TypingContext::Index => TypeCtx::Unspecified,
    }
}
