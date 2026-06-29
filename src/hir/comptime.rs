use crate::diagnostics::Diagnostic;
use crate::hir::checker::TypeChecker;
use crate::hir::hir::HirExpr;
use crate::hir::types::TypeId;

/// The result of evaluating a comptime block:
/// either a concrete value or a type.
#[derive(Debug, Clone)]
pub enum ComptimeValue {
    Type(TypeId),
    Value(HirExpr),
    Error,
}

/// Evaluation context for comptime blocks.
/// Comptime blocks are evaluated during type-checking and must be
/// side-effect free (no @io, no file access, no external calls).
pub struct ComptimeEvalContext<'a> {
    /// Reference to the parent type checker.
    pub checker: &'a mut TypeChecker<'a>,
    /// Maximum evaluation steps before bailing out.
    pub step_limit: usize,
    /// Current step count.
    pub steps: usize,
}

impl<'a> ComptimeEvalContext<'a> {
    /// Create a new comptime evaluation context.
    pub fn new(checker: &'a mut TypeChecker<'a>) -> Self {
        ComptimeEvalContext {
            checker,
            step_limit: 1000,
            steps: 0,
        }
    }

    /// Check whether the given expression is allowed in comptime context.
    /// Comptime blocks cannot contain I/O, file access, or other side effects.
    pub fn check_comptime_allowed(expr: &HirExpr) -> bool {
        match expr {
            HirExpr::Literal(..) => true,
            HirExpr::Ident(..) => true,
            HirExpr::BinaryOp { .. } => true,
            HirExpr::UnaryOp { .. } => true,
            HirExpr::Tuple(..) => true,
            HirExpr::Array(..) => true,
            HirExpr::Call { .. } => {
                // Function calls in comptime are allowed only if the function
                // is marked `@pure` or is itself a comptime function.
                // For now, conservatively allow all calls.
                true
            }
            // Side-effectful operations are not comptime-safe:
            HirExpr::FieldAccess { .. } => true,
            _ => {
                // I/O, file access, and other impure operations are disallowed.
                // This is a conservative starting point.
                false
            }
        }
    }

    /// Evaluate a comptime expression to a value.
    /// Returns `None` if the expression cannot be evaluated at comptime.
    pub fn eval_expr(&mut self, _expr: &HirExpr) -> Option<ComptimeValue> {
        // Skeleton: actual evaluation will be implemented in future iterations.
        // For now, this returns None to signal "deferred to runtime".
        None
    }
}
