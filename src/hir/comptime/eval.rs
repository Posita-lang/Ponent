use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::types::TypeContext;

use super::error::ComptimeError;
use super::value::ComptimeValue;

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
        self.steps += 1;

        match expr {
            HirExpr::Literal(lit, _ty, _span) => match lit {
                crate::ast::Literal::Int(n) => Ok(ComptimeValue::Int(*n)),
                crate::ast::Literal::Bool(b) => Ok(ComptimeValue::Bool(*b)),
                _ => Err(ComptimeError::Deferred),
            },
            HirExpr::Block(stmts, _ty, _span) => self.eval_block(stmts),
            HirExpr::BinaryOp { left, op, right, .. } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                match (l, r, op) {
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Add) => {
                        Ok(ComptimeValue::Int(a + b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Sub) => {
                        Ok(ComptimeValue::Int(a - b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Mul) => {
                        Ok(ComptimeValue::Int(a * b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Div) => {
                        if b == 0 {
                            Err(ComptimeError::DivisionByZero)
                        } else {
                            Ok(ComptimeValue::Int(a / b))
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
