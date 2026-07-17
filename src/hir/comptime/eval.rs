use crate::hir::hir::{HirExpr, HirPattern, HirProgram, HirStmt};
use crate::hir::types::{TypeContext, TypeId};
use crate::hir::symbol::SymbolTable;
use crate::symbol::Symbol;

use super::error::ComptimeError;
use super::value::ComptimeValue;

use std::collections::HashMap;
use std::sync::Arc;

/// A registered comptime function: (parameter_names, body_statements).
type ComptimeFn = (Vec<Symbol>, Vec<HirStmt>);

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

/// Apply the overflow policy to `result` given the type's representable range.
/// Returns the corrected value, or `Overflow` error if the policy is `Trap`.
fn apply_overflow_policy(result: i128, min: i128, max: i128, policy: &crate::ast::OverflowPolicy) -> Result<i128, ComptimeError> {
    if result >= min && result <= max {
        return Ok(result);
    }
    match policy {
        crate::ast::OverflowPolicy::Wrap => {
            // Two's complement wrapping within [min, max].
            let range = max.wrapping_sub(min).wrapping_add(1);
            if range == 0 {
                return Ok(result);
            }
            Ok(result.wrapping_sub(min).wrapping_rem_euclid(range).wrapping_add(min))
        }
        crate::ast::OverflowPolicy::Saturate => {
            if result < min { Ok(min) } else { Ok(max) }
        }
        crate::ast::OverflowPolicy::Trap => Err(ComptimeError::Overflow),
    }
}

/// Check `result` against the type's bit width and overflow policy.
/// Returns the (possibly adjusted) value, or `Overflow` if the policy is `Trap`.
fn check_range(result: i128, ty: TypeId, ctx: &TypeContext) -> Result<i128, ComptimeError> {
    match ctx.get(ty) {
        crate::hir::types::TypeData::Int { bits, overflow_policy, .. } => {
            let (min, max) = signed_range(*bits);
            apply_overflow_policy(result, min, max, overflow_policy)
        }
        crate::hir::types::TypeData::UInt { bits, overflow_policy, .. } => {
            let (min, max) = unsigned_range(*bits);
            apply_overflow_policy(result, min, max, overflow_policy)
        }
        // During comptime evaluation within type checking, the result type
        // may still be an un-resolved InferVar (e.g. TypeVariableKind::Numeric
        // from a BinaryOp).  Skip range checking in that case — the type
        // checker will resolve it later and catch any mismatches.
        crate::hir::types::TypeData::InferVar { .. } => Ok(result),
        _ => Err(ComptimeError::Internal(format!(
            "check_range called on non-integer type: {:?}",
            ctx.get(ty)
        ))),
    }
}

/// Evaluation context for comptime blocks.
/// Tracks step budget and provides expression evaluation.
pub struct ComptimeEvalContext<'a> {
    ctx: &'a TypeContext,
    steps: usize,
    step_limit: usize,
    /// The HIR program, used to lookup comptime function definitions.
    /// Optional because the HirProgram is not available during type checking
    /// (it is the output of check_program).  Will be populated when comptime
    /// function calls are implemented.
    hir_program: Option<&'a HirProgram>,
    /// The symbol table, used for name resolution.
    symbols: &'a SymbolTable,
    /// Local variable bindings within the current comptime block.
    pub variables: HashMap<Symbol, ComptimeValue>,
    /// Registry of comptime functions: name → (param_names, body).
    /// Populated by the checker as it encounters comptime function definitions.
    fn_registry: HashMap<Symbol, ComptimeFn>,
}

impl<'a> ComptimeEvalContext<'a> {
    pub fn new(ctx: &'a TypeContext, symbols: &'a SymbolTable) -> Self {
        ComptimeEvalContext {
            ctx,
            symbols,
            hir_program: None,
            steps: 0,
            step_limit: 10_000,
            variables: HashMap::new(),
            fn_registry: HashMap::new(),
        }
    }

    /// Register a comptime function so it can be called from within comptime blocks.
    pub fn register_fn(&mut self, name: Symbol, params: Vec<Symbol>, body: Vec<HirStmt>) {
        self.fn_registry.insert(name, (params, body));
    }

    /// Set a custom step limit (for testing).
    pub fn set_step_limit(&mut self, limit: usize) {
        self.step_limit = limit;
    }

    /// Evaluate a comptime block (sequence of statements) and return the result.
    #[must_use]
    pub fn eval_block(&mut self, stmts: &[HirStmt]) -> Result<ComptimeValue, ComptimeError> {
        let mut result = ComptimeValue::Unit;
        for stmt in stmts {
            match stmt {
                HirStmt::Expression(expr) => {
                    result = self.eval_expr(expr)?;
                }
                HirStmt::VariableDef { name, value, .. } => {
                    let val = match value {
                        Some(e) => self.eval_expr(e)?,
                        None => return Err(ComptimeError::not_allowed(
                            "variable definitions in comptime blocks must have a value",
                        )),
                    };
                    if let Some(n) = name {
                        self.variables.insert(n.clone(), val.clone());
                        result = val;
                    } else {
                        return Err(ComptimeError::not_allowed(
                            "unnamed variables are not allowed in comptime blocks",
                        ));
                    }
                }
                HirStmt::Assign { target, value, .. } => {
                    let val = self.eval_expr(value)?;
                    if let HirExpr::Ident(name, _, _) = target.as_ref() {
                        if self.variables.contains_key(name) {
                            self.variables.insert(name.clone(), val.clone());
                            result = val;
                        } else {
                            return Err(ComptimeError::UnknownIdentifier(name.as_str()));
                        }
                    } else {
                        return Err(ComptimeError::not_allowed(
                            "only simple variable assignments are supported in comptime blocks",
                        ));
                    }
                }
                HirStmt::While { cond, body, .. } => {
                    loop {
                        if self.steps >= self.step_limit {
                            return Err(ComptimeError::StepLimitExceeded);
                        }
                        let cond_val = self.eval_expr(cond)?;
                        match cond_val {
                            ComptimeValue::Bool(true) => {
                                self.eval_block(body)?;
                            }
                            ComptimeValue::Bool(false) => break,
                            ComptimeValue::Float(_) => return Err(ComptimeError::type_error(
                                "while condition must be a boolean, found Float",
                            )),
                            ComptimeValue::String(_) => return Err(ComptimeError::type_error(
                                "while condition must be a boolean, found String",
                            )),
                            _ => return Err(ComptimeError::type_error(
                                "while condition must be a boolean",
                            )),
                        }
                    }
                    result = ComptimeValue::Unit;
                }
                _ => {
                    return Err(ComptimeError::not_allowed(
                        "only expressions, variable definitions, and assignments are allowed in comptime blocks",
                    ));
                }
            }
        }
        Ok(result)
    }

    /// Evaluate a comptime expression to a value.
    #[must_use]
    pub fn eval_expr(&mut self, expr: &HirExpr) -> Result<ComptimeValue, ComptimeError> {
        if self.steps >= self.step_limit {
            return Err(ComptimeError::StepLimitExceeded);
        }
        self.steps = self.steps.saturating_add(1);

        match expr {
            HirExpr::Literal(lit, _ty, _span) => match lit {
                crate::ast::Literal::Int(n) => Ok(ComptimeValue::Int(*n)),
                crate::ast::Literal::Float(f) => Ok(ComptimeValue::Float(*f)),
                crate::ast::Literal::Char(c) => Ok(ComptimeValue::Int(*c as i128)),
                crate::ast::Literal::Bool(b) => Ok(ComptimeValue::Bool(*b)),
                crate::ast::Literal::String(s) => Ok(ComptimeValue::String(Arc::from(s.as_str()))),
                crate::ast::Literal::ByteString(b) => {
                    Ok(ComptimeValue::String(
                        Arc::from(String::from_utf8_lossy(b).as_ref()),
                    ))
                }
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
                    // Comparison operators: return Bool
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Lt) => {
                        Ok(ComptimeValue::Bool(a < b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Gt) => {
                        Ok(ComptimeValue::Bool(a > b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Le) => {
                        Ok(ComptimeValue::Bool(a <= b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Ge) => {
                        Ok(ComptimeValue::Bool(a >= b))
                    }
                    // ── Float arithmetic ───────────────────────────────
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Add) => {
                        Ok(ComptimeValue::Float(a + b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Sub) => {
                        Ok(ComptimeValue::Float(a - b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Mul) => {
                        Ok(ComptimeValue::Float(a * b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Div) => {
                        Ok(ComptimeValue::Float(a / b))
                    }
                    // ── Float comparisons ──────────────────────────────
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Lt) => {
                        Ok(ComptimeValue::Bool(a < b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Gt) => {
                        Ok(ComptimeValue::Bool(a > b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Le) => {
                        Ok(ComptimeValue::Bool(a <= b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Ge) => {
                        Ok(ComptimeValue::Bool(a >= b))
                    }
                    // ── String concatenation ───────────────────────────
                    (ComptimeValue::String(a), ComptimeValue::String(b), crate::ast::BinOp::Add) => {
                        let mut result = String::with_capacity(a.len() + b.len());
                        result.push_str(&a);
                        result.push_str(&b);
                        Ok(ComptimeValue::String(Arc::from(result)))
                    }
                    // ── String equality ────────────────────────────────
                    (ComptimeValue::String(a), ComptimeValue::String(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::String(a), ComptimeValue::String(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
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
                    ComptimeValue::Float(_) => Err(ComptimeError::type_error(
                        "if condition must be a boolean, found Float",
                    )),
                    ComptimeValue::String(_) => Err(ComptimeError::type_error(
                        "if condition must be a boolean, found String",
                    )),
                    _ => Err(ComptimeError::type_error("if condition must be a boolean")),
                }
            }
            HirExpr::Ident(name, _ty, _span) => {
                // 1. Check local variables first.
                if let Some(val) = self.variables.get(name) {
                    return Ok(val.clone());
                }
                // 2. Check if the name is a zero-argument comptime function
                //    (e.g. `comptime def N() -> Int<32> { 5 }` referenced as `N`).
                if let Some((params, body)) = self.fn_registry.get(name).cloned() {
                    if params.is_empty() {
                        let saved = std::mem::take(&mut self.variables);
                        let result = self.eval_block(&body);
                        self.variables = saved;
                        return result;
                    }
                }
                // 3. Check the symbol table for comptime-known values.
                //    (e.g. comptime function parameters, imported constants).
                //    For now, this is a placeholder — full symbol table integration
                //    will be added in a later phase.
                Err(ComptimeError::UnknownIdentifier(name.as_str()))
            }
            HirExpr::Call { callee, args, comptime, .. } if *comptime => {
                // Resolve the callee to a function name.
                let fn_name = match callee.as_ref() {
                    HirExpr::Ident(name, _, _) => *name,
                    _ => return Err(ComptimeError::type_error(
                        "comptime call target must be a simple function name",
                    )),
                };
                // Built-in: assert(condition)
                if fn_name.as_str() == "assert" {
                    if args.len() != 1 {
                        return Err(ComptimeError::type_error(
                            "assert takes exactly one argument",
                        ));
                    }
                    let cond = self.eval_expr(&args[0])?;
                    match cond {
                        ComptimeValue::Bool(true) => Ok(ComptimeValue::Unit),
                        ComptimeValue::Bool(false) => Err(ComptimeError::AssertionFailed(
                            "assertion failed".into(),
                        )),
                        _ => Err(ComptimeError::type_error(
                            "assert argument must be a boolean",
                        )),
                    }
                } else {
                    // Look up the function in the registry.
                    let (params, body) = self.fn_registry.get(&fn_name).ok_or_else(|| {
                        ComptimeError::UnknownIdentifier(fn_name.as_str())
                    })?.clone();
                // Evaluate arguments.
                let arg_vals: Vec<ComptimeValue> = args
                    .iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                if arg_vals.len() != params.len() {
                    return Err(ComptimeError::type_error(format!(
                        "comptime function `{}` expected {} arguments, got {}",
                        fn_name, params.len(), arg_vals.len(),
                    )));
                }
                // Save the current variable scope and bind parameters.
                let saved = std::mem::take(&mut self.variables);
                for (param, val) in params.iter().zip(arg_vals.into_iter()) {
                    self.variables.insert(param.clone(), val);
                }
                // Evaluate the function body.
                let result = self.eval_block(&body);
                // Restore the previous variable scope.
                self.variables = saved;
                result
                }
            }
            HirExpr::TypeInfo(ty, _) => {
                // @typeInfo(T) returns the type itself as a comptime value.
                Ok(ComptimeValue::Type(*ty))
            }
            HirExpr::CompileError(msg, _) => {
                Err(ComptimeError::AssertionFailed(msg.clone()))
            }
            _ => Err(ComptimeError::Deferred),
        }
    }
}
