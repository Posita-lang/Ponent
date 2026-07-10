use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::types::{TypeContext, TypeId};

use super::error::ComptimeError;
use super::value::ComptimeValue;

/// Compute the representable range for a signed integer of `bits` width.
fn signed_range(bits: u8) -> (i128, i128) {
    if bits == 0 {
        (0, 0)
    } else if bits >= 127 {
        (i128::MIN, i128::MAX)
    } else {
        let max = (1i128 << (bits - 1)) - 1;
        let min = -(1i128 << (bits - 1));
        (min, max)
    }
}

/// Compute the representable range for an unsigned integer of `bits` width.
fn unsigned_range(bits: u8) -> (i128, i128) {
    if bits >= 128 {
        (0, i128::MAX)
    } else {
        let max = (1i128 << bits) - 1;
        (0, max)
    }
}

/// Check `result` against the type's bit width.  Returns `Overflow` if the
/// value falls outside the representable range of the declared type.
/// Returns `Internal` if called on a non-integer type — this is a HIR invariant
/// violation that must be caught, not silently forwarded.
fn check_range(result: i128, ty: TypeId, ctx: &TypeContext) -> Result<i128, ComptimeError> {
    let (min, max) = match ctx.get(ty) {
        crate::hir::types::TypeData::Int { bits, .. } => signed_range(*bits),
        crate::hir::types::TypeData::UInt { bits, .. } => unsigned_range(*bits),
        _ => {
            return Err(ComptimeError::Internal(format!(
                "check_range called on non-integer type: {:?}",
                ctx.get(ty)
            )));
        }
    };
    if result < min || result > max {
        Err(ComptimeError::Overflow)
    } else {
        Ok(result)
    }
}

/// Evaluation context for comptime blocks.
/// Tracks step budget and provides expression evaluation.
pub struct ComptimeEvalContext<'a> {
    ctx: &'a TypeContext,
    steps: usize,
    step_limit: usize,
}

impl<'a> ComptimeEvalContext<'a> {
    pub fn new(ctx: &'a TypeContext) -> Self {
        ComptimeEvalContext {
            ctx,
            steps: 0,
            step_limit: 10_000,
        }
    }

    /// Set a custom step limit (for testing).
    pub fn set_step_limit(&mut self, limit: usize) {
        self.step_limit = limit;
    }

    /// Evaluate a comptime block (sequence of statements) and return the result.
    pub fn eval_block(&mut self, stmts: &[HirStmt]) -> Result<ComptimeValue, ComptimeError> {
        let mut result = ComptimeValue::Unit;
        for stmt in stmts {
            match stmt {
                HirStmt::Expression(expr) => {
                    result = self.eval_expr(expr)?;
                }
                _ => {
                    return Err(ComptimeError::not_allowed(
                        "only expressions are allowed in comptime blocks",
                    ));
                }
            }
        }
        Ok(result)
    }

    /// Evaluate a comptime expression to a value.
    pub fn eval_expr(&mut self, expr: &HirExpr) -> Result<ComptimeValue, ComptimeError> {
        if self.steps >= self.step_limit {
            return Err(ComptimeError::StepLimitExceeded);
        }
        self.steps = self.steps.saturating_add(1);

        match expr {
            HirExpr::Literal(lit, _ty, _span) => match lit {
                crate::ast::Literal::Int(n) => Ok(ComptimeValue::Int(*n)),
                crate::ast::Literal::Bool(b) => Ok(ComptimeValue::Bool(*b)),
                _ => Err(ComptimeError::Deferred),
            },
            HirExpr::Block(stmts, _ty, _span) => self.eval_block(stmts),
            HirExpr::BinaryOp { left, op, right, ty, .. } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                match (l, r, op) {
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Add) => {
                        let result = a.checked_add(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Sub) => {
                        let result = a.checked_sub(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Mul) => {
                        let result = a.checked_mul(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Div) => {
                        if b == 0 {
                            Err(ComptimeError::DivisionByZero)
                        } else if a == i128::MIN && b == -1 {
                            // i128::MIN / -1 overflows (can't represent as i128)
                            Err(ComptimeError::Overflow)
                        } else {
                            let result = a / b;
                            check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                        }
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Rem) => {
                        if b == 0 {
                            Err(ComptimeError::DivisionByZero)
                        } else if a == i128::MIN && b == -1 {
                            // i128::MIN % -1 overflows in the same way as division
                            Err(ComptimeError::Overflow)
                        } else {
                            let result = a % b;
                            check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                        }
                    }
                    _ => Err(ComptimeError::type_error("unsupported binary operation")),
                }
            }
            HirExpr::If { cond, then_branch, else_branch, .. } => {
                let cond_val = self.eval_expr(cond)?;
                match cond_val {
                    ComptimeValue::Bool(true) => self.eval_block(then_branch),
                    ComptimeValue::Bool(false) => {
                        if let Some(else_branch) = else_branch {
                            self.eval_block(else_branch)
                        } else {
                            Ok(ComptimeValue::Unit)
                        }
                    }
                    _ => Err(ComptimeError::type_error("if condition must be a boolean")),
                }
            }
            _ => Err(ComptimeError::Deferred),
        }
    }
}
