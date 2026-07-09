use crate::ast::{BinOp, Literal, Span, UnaryOp};
use crate::diagnostics::{Diagnostic, DiagnosticCollector};
use crate::hir::checker::TypeChecker;
use crate::hir::hir::HirExpr;
use crate::hir::symbol::FunctionBinding;
use crate::hir::types::{AdtKind, DefId, TypeData, TypeId};

/// Errors that can occur during comptime evaluation.
#[derive(Debug, Clone)]
pub enum ComptimeError {
    /// The expression cannot be evaluated at compile time (defer to runtime).
    Deferred,
    /// Step limit reached; possible infinite loop.
    StepLimitExceeded,
    /// Division or remainder by zero.
    DivisionByZero,
    /// Type mismatch in a comptime operation.
    TypeError(String),
    /// Assertion failed at compile time.
    AssertionFailed(String),
    /// Unknown comptime function.
    UnknownFunction(String),
    /// Unknown identifier.
    UnknownIdentifier(String),
    /// A runtime-only construct encountered in comptime context.
    NotComptimeAllowed(String),
    /// Internal comptime error.
    Internal(String),
}

impl std::fmt::Display for ComptimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComptimeError::Deferred => write!(f, "expression cannot be evaluated at compile time"),
            ComptimeError::StepLimitExceeded => write!(f, "comptime step limit exceeded (possible infinite loop)"),
            ComptimeError::DivisionByZero => write!(f, "division by zero in comptime expression"),
            ComptimeError::TypeError(msg) => write!(f, "comptime type error: {}", msg),
            ComptimeError::AssertionFailed(msg) => write!(f, "comptime assertion failed: {}", msg),
            ComptimeError::UnknownFunction(name) => write!(f, "unknown comptime function: {}", name),
            ComptimeError::UnknownIdentifier(name) => write!(f, "unknown identifier in comptime: {}", name),
            ComptimeError::NotComptimeAllowed(msg) => write!(f, "{}", msg),
            ComptimeError::Internal(msg) => write!(f, "internal comptime error: {}", msg),
        }
    }
}

/// The result of evaluating a comptime block:
/// either a concrete value, a type, a control-flow signal, or an error.
#[derive(Debug)]
pub enum ComptimeValue {
    Type(TypeId),
    /// A concrete HIR expression value (cheap to clone via Arc).
    Value(Arc<HirExpr>),
    /// Signal: break out of the current comptime loop with an optional value.
    Break(Option<Box<ComptimeValue>>),
    /// Signal: skip to the next iteration of the current comptime loop.
    Continue,
    Error,
}

impl Clone for ComptimeValue {
    fn clone(&self) -> Self {
        match self {
            ComptimeValue::Type(ty) => ComptimeValue::Type(*ty),
            ComptimeValue::Value(hir) => ComptimeValue::Value(Arc::clone(hir)),
            ComptimeValue::Break(val) => ComptimeValue::Break(val.clone()),
            ComptimeValue::Continue => ComptimeValue::Continue,
            ComptimeValue::Error => ComptimeValue::Error,
        }
    }
}

impl ComptimeValue {
    /// If this is a `Value`, return a reference to the inner `HirExpr`.
    pub fn as_hir_expr(&self) -> Option<&HirExpr> {
        match self {
            ComptimeValue::Value(hir) => Some(hir.as_ref()),
            _ => None,
        }
    }
}

/// A lazy iterator over comptime-generated values, following the KSP
/// `has_next()`/`next()` pattern (see YenTopKShortestPathsAlg).
/// `priority` is a function that returns an ordering key for candidates
/// (lower = explored first).  Callers supply the priority function to
/// control the exploration order (e.g. simpler types before complex ones).
#[derive(Debug, Clone)]
pub struct SeqGen<T: Clone> {
    candidates: Vec<T>,
    limit: usize,
    count: usize,
    generator: Option<fn(&T) -> Option<Vec<T>>>,
    priority: fn(&T) -> usize,
}

impl<T: Clone> SeqGen<T> {
    pub fn new(first: T, limit: usize, generator: fn(&T) -> Option<Vec<T>>) -> Self {
        SeqGen {
            candidates: vec![first],
            limit,
            count: 0,
            generator: Some(generator),
            priority: |_| 0,
        }
    }

    /// Create a new `SeqGen` with a custom priority function.
    /// Lower priority values are explored first (like Yen's path cost).
    pub fn with_priority(
        first: T,
        limit: usize,
        generator: fn(&T) -> Option<Vec<T>>,
        priority: fn(&T) -> usize,
    ) -> Self {
        SeqGen {
            candidates: vec![first],
            limit,
            count: 0,
            generator: Some(generator),
            priority,
        }
    }

    pub fn has_next(&self) -> bool {
        !self.candidates.is_empty() && self.count < self.limit
    }

    pub fn next(&mut self) -> Option<T> {
        if self.candidates.is_empty() || self.count >= self.limit {
            return None;
        }

        let result = self.candidates.remove(0);
        self.count += 1;

        if self.count < self.limit {
            if let Some(gen) = self.generator {
                if let Some(new_candidates) = gen(&result) {
                    for c in new_candidates {
                        let pos = self.candidates.iter().position(|x| {
                            (self.priority)(x) > (self.priority)(&c)
                        });
                        match pos {
                            Some(p) => self.candidates.insert(p, c),
                            None => self.candidates.push(c),
                        }
                    }
                }
            }
        }

        Some(result)
    }
}

/// A comptime iterator that yields types from a type factory function.
#[derive(Debug, Clone)]
pub struct TypeSeqGen {
    inner: SeqGen<TypeId>,
}

impl TypeSeqGen {
    pub fn new(
        first: TypeId,
        limit: usize,
        generator: fn(&TypeId) -> Option<Vec<TypeId>>,
    ) -> Self {
        TypeSeqGen {
            inner: SeqGen::new(first, limit, generator),
        }
    }

    pub fn has_next(&self) -> bool {
        self.inner.has_next()
    }

    pub fn next(&mut self) -> Option<TypeId> {
        self.inner.next()
    }
}

/// A comptime iterator that yields expressions from an expression factory.
#[derive(Debug, Clone)]
pub struct ExprSeqGen {
    inner: SeqGen<HirExpr>,
}

impl ExprSeqGen {
    pub fn new(
        first: HirExpr,
        limit: usize,
        generator: fn(&HirExpr) -> Option<Vec<HirExpr>>,
    ) -> Self {
        ExprSeqGen {
            inner: SeqGen::new(first, limit, generator),
        }
    }

    pub fn has_next(&self) -> bool {
        self.inner.has_next()
    }

    pub fn next(&mut self) -> Option<HirExpr> {
        self.inner.next()
    }
}

/// Information about the current comptime call site, used for
/// error reporting and call-stack tracking.
#[derive(Debug, Clone)]
pub struct ComptimeFrame {
    /// Name of the function or "comptime block".
    pub name: String,
    /// Source span of the call site.
    pub span: Span,
}

/// Evaluation context for comptime blocks.
pub struct ComptimeEvalContext<'a> {
    /// Reference to the parent type checker.
    pub checker: &'a mut TypeChecker<'a>,
    /// Maximum evaluation steps before bailing out.
    pub step_limit: usize,
    /// Current step count.
    pub steps: usize,
    /// Call stack for error reporting and recursion depth tracking.
    pub call_stack: Vec<ComptimeFrame>,
    /// Local variable bindings within the current comptime scope.
    pub local_values: HashMap<String, ComptimeValue>,
}

use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

impl<'a> ComptimeEvalContext<'a> {
    /// Create a new comptime evaluation context.
    pub fn new(checker: &'a mut TypeChecker<'a>) -> Self {
        ComptimeEvalContext {
            checker,
            step_limit: 1000,
            steps: 0,
            call_stack: Vec::new(),
            local_values: HashMap::default(),
        }
    }

    /// Push a call frame for stack-trace tracking.
    pub fn push_frame(&mut self, name: String, span: Span) {
        self.call_stack.push(ComptimeFrame { name, span });
    }

    /// Pop the most recent call frame.
    pub fn pop_frame(&mut self) {
        self.call_stack.pop();
    }

    /// Format the current call stack as a human-readable trace.
    pub fn format_stack_trace(&self) -> String {
        if self.call_stack.is_empty() {
            return String::new();
        }
        let mut trace = String::from("comptime call stack:\n");
        for (i, frame) in self.call_stack.iter().enumerate() {
            trace.push_str(&format!("  {}: {} at {:?}\n", i, frame.name, frame.span));
        }
        trace
    }

    /// Check whether the given expression is allowed in comptime context.
    pub fn check_comptime_allowed(expr: &HirExpr) -> bool {
        match expr {
            HirExpr::Literal(..) => true,
            HirExpr::Ident(..) => true,
            HirExpr::BinaryOp { .. } => true,
            HirExpr::UnaryOp { .. } => true,
            HirExpr::Tuple(..) => true,
            HirExpr::Array(..) => true,
            HirExpr::Call { .. } => true,
            HirExpr::FieldAccess { .. } => true,
            _ => false,
        }
    }

    /// Evaluate a comptime expression to a value.
    /// Returns `Err(ComptimeError::Deferred)` if the expression cannot be evaluated
    /// at comptime (caller should fall back to runtime codegen).
    /// Returns `Err(...)` for other errors that should halt compilation.
    pub fn eval_expr(&mut self, expr: &HirExpr) -> Result<ComptimeValue, ComptimeError> {
        if self.steps >= self.step_limit {
            return Err(ComptimeError::StepLimitExceeded);
        }
        self.steps += 1;

        match expr {
            // ── Literal ──────────────────────────────────────────────
            HirExpr::Literal(lit, ty, _) => {
                Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(lit.clone(), *ty, Span::new(0, 0)))))
            }

            // ── Identifier ───────────────────────────────────────────
            HirExpr::Ident(name, _ty, _span) => {
                // 1. Check local comptime bindings (set by variable defs in comptime blocks)
                if let Some(val) = self.local_values.get(name) {
                    return Ok(val.clone());
                }
                // 2. Look up in checker's local variable types
                if let Some(&ty) = self.checker.local_variable_types.get(name) {
                    return Ok(ComptimeValue::Type(ty));
                }
                // 3. Look up in type param cache
                if let Some(&ty) = self.checker.local_type_param_cache.get(name) {
                    return Ok(ComptimeValue::Type(ty));
                }
                // 4. Look up as a type alias in the resolution map
                if let Some(&def_id) = self.checker.resolution_map.type_def_ids.get(name) {
                    if let Some(type_id) = self.checker.ctx.get_type_id_for_def_id(def_id) {
                        return Ok(ComptimeValue::Type(type_id));
                    }
                }
                Err(ComptimeError::UnknownIdentifier(name.clone()))
            }

            // ── BinaryOp ─────────────────────────────────────────────
            HirExpr::BinaryOp { left, op, right, ty, span } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.eval_binary_op(l, r, *op, *ty, *span)
            }

            // ── UnaryOp ──────────────────────────────────────────────
            HirExpr::UnaryOp { op, expr: operand, ty, span } => {
                let val = self.eval_expr(operand)?;
                self.eval_unary_op(val, *op, *ty, *span)
            }

            // ── Block: evaluate all statements, return last value ─────
            HirExpr::Block(stmts, _ty, _span) => {
                self.eval_body_stmts(stmts)
            }

            // ── If: evaluate condition, branch accordingly ────────────
            HirExpr::If { cond, then_branch, else_branch, .. } => {
                let cond_val = self.eval_expr(cond)?;
                match cond_val {
                    ComptimeValue::Value(hir) => match hir.as_ref() {
                        HirExpr::Literal(Literal::Bool(true), _, _) => {
                            self.eval_body_stmts(then_branch)
                        }
                        HirExpr::Literal(Literal::Bool(false), _, _) => {
                            if let Some(else_branch) = else_branch {
                                self.eval_body_stmts(else_branch)
                            } else {
                                Ok(ComptimeValue::Value(Arc::new(
                                    HirExpr::Literal(Literal::Bool(false), self.checker.ctx.unit(), Span::new(0, 0))
                                )))
                            }
                        }
                        _ => Err(ComptimeError::Deferred),
                    },
                    _ => Err(ComptimeError::Deferred),
                }
            }

            // ── Loop: bounded comptime loop with Break/Continue support ──
            HirExpr::Loop { body, .. } => {
                loop {
                    if self.steps >= self.step_limit {
                        return Err(ComptimeError::StepLimitExceeded);
                    }
                    self.steps += 1;
                    match self.eval_body_stmts(body)? {
                        ComptimeValue::Break(val) => {
                            return Ok(ComptimeValue::Break(val));
                        }
                        ComptimeValue::Continue => continue,
                        ComptimeValue::Error => return Ok(ComptimeValue::Error),
                        _ => continue, // Normal values are ignored; keep looping
                    }
                }
            }

            // ── Tuple ────────────────────────────────────────────────
            HirExpr::Tuple(elems, ty, span) => {
                let mut evaled = Vec::with_capacity(elems.len());
                for elem in elems {
                    match self.eval_expr(elem)? {
                        ComptimeValue::Value(v) => evaled.push(HirExpr::clone(v.as_ref())),
                        _ => return Err(ComptimeError::TypeError("tuple element not a comptime value".into())),
                    }
                }
                Ok(ComptimeValue::Value(Arc::new(HirExpr::Tuple(evaled, *ty, *span))))
            }

            // ── Array ────────────────────────────────────────────────
            HirExpr::Array(elems, ty, span) => {
                let mut evaled = Vec::with_capacity(elems.len());
                for elem in elems {
                    match self.eval_expr(elem)? {
                        ComptimeValue::Value(v) => evaled.push(HirExpr::clone(v.as_ref())),
                        _ => return Err(ComptimeError::TypeError("array element not a comptime value".into())),
                    }
                }
                Ok(ComptimeValue::Value(Arc::new(HirExpr::Array(evaled, *ty, *span))))
            }

            // ── TypeAnnotated ────────────────────────────────────────
            HirExpr::TypeAnnotated { expr: inner, .. } => {
                self.eval_expr(inner)
            }

            // ── Cast ─────────────────────────────────────────────────
            HirExpr::Cast { expr: inner, ty, .. } => {
                let val = self.eval_expr(inner)?;
                match val {
                    ComptimeValue::Value(hir) => match hir.as_ref() {
                        HirExpr::Literal(lit, _, _) => {
                            Ok(ComptimeValue::Value(Arc::new(
                                HirExpr::Literal(lit.clone(), *ty, Span::new(0, 0))
                            )))
                        }
                        _ => {
                            Ok(ComptimeValue::Value(Arc::new(HirExpr::Cast {
                                expr: Box::new(HirExpr::clone(inner.as_ref())),
                                ty: *ty,
                                safe: true,
                                rounding: None,
                                span: Span::new(0, 0),
                            })))
                        }
                    },
                    _ => {
                        Ok(ComptimeValue::Value(Arc::new(HirExpr::Cast {
                            expr: Box::new(HirExpr::clone(inner.as_ref())),
                            ty: *ty,
                            safe: true,
                            rounding: None,
                            span: Span::new(0, 0),
                        })))
                    }
                }
            }

            // ── Range ────────────────────────────────────────────────
            HirExpr::Range { .. } => Err(ComptimeError::Deferred),

            // ── Call: comptime function calls ─────────────────────────
            HirExpr::Call { callee, args, comptime: true, ty, span } => {
                self.eval_comptime_call(callee, args, *ty, *span)
            }
            HirExpr::Call { .. } => {
                // Non-comptime calls are deferred to runtime
                Err(ComptimeError::Deferred)
            }

            // ── Everything else: defer to runtime ────────────────────
            _ => Err(ComptimeError::Deferred),
        }
    }

    /// Evaluate a comptime call. Handles built-in functions and user-defined
    /// comptime functions.
    fn eval_comptime_call(
        &mut self,
        callee: &HirExpr,
        args: &[HirExpr],
        ret_ty: TypeId,
        span: Span,
    ) -> Result<ComptimeValue, ComptimeError> {
        // Resolve the callee to a name
        let callee_name = match callee {
            HirExpr::Ident(name, _, _) => name.clone(),
            HirExpr::AttrAccess { base, attr, .. } => {
                // Resolve method-style calls: Module::function or Type::method
                if let HirExpr::Ident(base_name, _, _) = base.as_ref() {
                    format!("{}::{}", base_name, attr)
                } else {
                    return Err(ComptimeError::UnknownFunction(format!(
                        "unsupported comptime callee: {:?}", callee
                    )));
                }
            }
            _ => return Err(ComptimeError::UnknownFunction(format!(
                "unsupported comptime callee: {:?}", callee
            ))),
        };

        // ── Built-in comptime functions ──────────────────────────
        match callee_name.as_str() {
            "assert" | "@assert" => {
                if args.len() == 1 {
                    let cond = self.eval_expr(&args[0])?;
                    match cond {
                        ComptimeValue::Value(hir) => match hir.as_ref() {
                            HirExpr::Literal(Literal::Bool(true), _, _) => {
                                Ok(ComptimeValue::Value(Arc::new(
                                    HirExpr::Literal(Literal::Bool(true), ret_ty, Span::new(0, 0))
                                )))
                            }
                            HirExpr::Literal(Literal::Bool(false), _, _) => {
                                Err(ComptimeError::AssertionFailed(format!(
                                    "assertion failed{}{}",
                                    if self.call_stack.is_empty() { String::new() } else { " at ".to_string() },
                                    self.format_stack_trace(),
                                )))
                            }
                            _ => Err(ComptimeError::TypeError(
                                "assert requires a comptime boolean expression".into()
                            )),
                        },
                        ComptimeValue::Error => Ok(ComptimeValue::Error),
                        _ => Err(ComptimeError::TypeError(
                            "assert requires a comptime boolean expression".into()
                        )),
                    }
                } else {
                    Err(ComptimeError::TypeError("assert requires exactly 1 argument".into()))
                }
            }

            "@compile_error" => {
                if let Some(arg) = args.first() {
                    if let HirExpr::Literal(Literal::String(msg), _, _) = arg {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!("compile error: {}", msg))
                                .with_span(span)
                                .with_suggestions(vec![self.format_stack_trace()]),
                        );
                    }
                }
                Ok(ComptimeValue::Error)
            }

            "@typeInfo" | "@typeInfo!" => {
                if args.len() == 1 {
                    let ty_val = self.eval_expr(&args[0])?;
                    match ty_val {
                        ComptimeValue::Type(type_id) => {
                            Ok(self.reflect_type_info(type_id, span))
                        }
                        ComptimeValue::Value(_) => Err(ComptimeError::TypeError(
                            "@typeInfo requires a type argument".into()
                        )),
                        ComptimeValue::Error => Ok(ComptimeValue::Error),
                    }
                } else {
                    Err(ComptimeError::TypeError("@typeInfo requires exactly 1 argument".into()))
                }
            }

            "sizeof" | "@sizeof" => {
                if args.len() == 1 {
                    let ty_val = self.eval_expr(&args[0])?;
                    match ty_val {
                        ComptimeValue::Type(type_id) => {
                            let size = self.estimate_type_size(type_id);
                            let usize_ty = self.checker.ctx.usize();
                            Ok(ComptimeValue::Value(Arc::new(
                                HirExpr::Literal(Literal::Int(size as i64), usize_ty, Span::new(0, 0))
                            )))
                        }
                        _ => Err(ComptimeError::TypeError("sizeof requires a type argument".into())),
                    }
                } else {
                    Err(ComptimeError::TypeError("sizeof requires exactly 1 argument".into()))
                }
            }

            _ => {
                // User-defined comptime function: look up in the symbol table
                self.eval_user_comptime_function(&callee_name, args, ret_ty, span)
            }
        }
    }

    /// Evaluate a user-defined comptime function by looking it up in the
    /// symbol table and evaluating its body.
    fn eval_user_comptime_function(
        &mut self,
        name: &str,
        args: &[HirExpr],
        _ret_ty: TypeId,
        span: Span,
    ) -> Result<ComptimeValue, ComptimeError> {
        // Look up the function in the symbol table
        let func_binding = self.checker.symbols.lookup_function(name)
            .ok_or_else(|| ComptimeError::UnknownFunction(name.to_string()))?;

        if !func_binding.is_comptime {
            return Err(ComptimeError::NotComptimeAllowed(format!(
                "function '{}' is not a comptime function; use `comptime def` to declare it",
                name
            )));
        }

        let body = func_binding.signature.body.as_ref()
            .ok_or_else(|| ComptimeError::Internal(format!(
                "comptime function '{}' has no body", name
            )))?;

        // Push a frame for stack trace
        self.push_frame(name.to_string(), span);

        // Evaluate arguments and bind them as local comptime values
        let params = &func_binding.signature.params;
        if args.len() != params.len() {
            self.pop_frame();
            return Err(ComptimeError::TypeError(format!(
                "comptime function '{}' expects {} arguments, got {}",
                name, params.len(), args.len()
            )));
        }

        // Save previous local values
        let prev_locals = self.local_values.clone();

        for (param, arg) in params.iter().zip(args.iter()) {
            let arg_val = self.eval_expr(arg)?;
            self.local_values.insert(param.name.clone(), arg_val);
        }

        // Evaluate the function body (the last expression)
        let result = self.eval_body_stmts(body);

        // Restore previous local values
        self.local_values = prev_locals;

        self.pop_frame();
        result
    }

    /// Evaluate a sequence of statements and return the value of the last expression.
    /// Handles VariableDef, Expression, Return (early exit), Leave (break), Continue.
    fn eval_body_stmts(&mut self, stmts: &[HirStmt]) -> Result<ComptimeValue, ComptimeError> {
        let mut last_value = None;
        for stmt in stmts {
            match stmt {
                HirStmt::VariableDef { name: Some(name), value: Some(value), .. } => {
                    let val = self.eval_expr(value)?;
                    self.local_values.insert(name.clone(), val.clone());
                    last_value = Some(val);
                }
                HirStmt::VariableDef { name: Some(name), value: None, ty, .. } => {
                    let comptime_ty = if !self.checker.ctx.is_error(*ty) {
                        *ty
                    } else {
                        self.checker.local_variable_types.get(name).copied()
                    };
                    if let Some(ty) = comptime_ty {
                        let val = ComptimeValue::Type(ty);
                        self.local_values.insert(name.clone(), val.clone());
                        last_value = Some(val);
                    }
                }
                HirStmt::Expression(expr) => {
                    let val = self.eval_expr(expr)?;
                    // Check for control-flow signals from expression evaluation
                    match &val {
                        ComptimeValue::Break(_) | ComptimeValue::Continue => return Ok(val),
                        _ => last_value = Some(val),
                    }
                }
                HirStmt::Return { value, .. } => {
                    if let Some(val_expr) = value {
                        return self.eval_expr(val_expr);
                    }
                    return Ok(ComptimeValue::Value(Arc::new(
                        HirExpr::Literal(Literal::Bool(false), self.checker.ctx.unit(), Span::new(0, 0))
                    )));
                }
                HirStmt::Leave { .. } => {
                    // `leave` inside comptime: break out of the current loop
                    return Ok(ComptimeValue::Break(None));
                }
                HirStmt::Continue { .. } => {
                    return Ok(ComptimeValue::Continue);
                }
                _ => {} // Skip other statements
            }
        }
        last_value.ok_or(ComptimeError::Deferred)
    }

/// Reflect on a type and produce a comptime representation of its metadata.
/// Returns a structured HirExpr tuple: (kind_name: &str, fields...)
/// For composite types, includes field/variant details from the symbol table.
fn reflect_type_info(&mut self, type_id: TypeId, _span: Span) -> ComptimeValue {
    let ctx = &mut self.checker.ctx;
    let unit_ty = ctx.unit();
    let usize_ty = ctx.usize();
    let bool_ty = ctx.bool();
    let str_ty = ctx.str_ref(); // &Str — proper type for comptime string labels
    // Build a single reusable tuple type for (kind_str, ...) pairs.
    let info_ty = ctx.unit(); // Simplified: all info tuples share unit type.

    // Helper closures — use the correct type per literal kind.
    let str_lit = |s: &str| HirExpr::Literal(Literal::String(s.into()), str_ty, Span::new(0, 0));
    let int_lit = |n: i64| HirExpr::Literal(Literal::Int(n), usize_ty, Span::new(0, 0));
    let bool_lit = |b: bool| HirExpr::Literal(Literal::Bool(b), bool_ty, Span::new(0, 0));

    // Build a proper tuple type for (kind, field...) info tuples.
    let info_tup_ty = |elems: &[HirExpr]| {
        let elem_tys: Vec<TypeId> = elems.iter().map(|e| match e {
            HirExpr::Literal(Literal::String(..), _, _) => str_ty,
            HirExpr::Literal(Literal::Int(..), _, _) => usize_ty,
            HirExpr::Literal(Literal::Bool(..), _, _) => bool_ty,
            HirExpr::Tuple(.., ty, _) => *ty,
            _ => unit_ty,
        }).collect();
        if elem_tys.is_empty() { unit_ty } else { ctx.tuple(elem_tys) }
    };

    /// Resolve an AST type expression to a DefId, handling simple paths
    /// and generic type names like `Int<32>` (returns the base type's DefId).
    fn resolve_ast_type_def_id(ty: &Type, resolution_map: &ResolutionMap, symbols: &SymbolTable) -> Option<i64> {
        match ty {
            Type::Path(path, _) => {
                if let Some(&did) = resolution_map.type_def_ids.get(&path[0]) {
                    return Some(did.0 as i64);
                }
                if let Some(did) = symbols.lookup_type_by_path(path) {
                    return Some(did.0 as i64);
                }
                None
            }
            Type::Generic(base, _args, _) => {
                if let Type::Path(path, _) = base.as_ref() {
                    if let Some(&did) = resolution_map.type_def_ids.get(&path[0]) {
                        return Some(did.0 as i64);
                    }
                }
                None
            }
            _ => None,
        }
    }

    let make_info = |elems: Vec<HirExpr>| -> ComptimeValue {
        let ty = info_tup_ty(&elems);
        ComptimeValue::Value(Arc::new(HirExpr::Tuple(elems, ty, Span::new(0, 0))))
    };

    let make_nested_info = |label: HirExpr, inner: HirExpr| -> ComptimeValue {
        let ty = info_tup_ty(&[label.clone(), inner.clone()]);
        ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![label, inner], ty, Span::new(0, 0))))
    };

    match ctx.get(type_id) {
        TypeData::Int { bits, signed } => {
            make_info(vec![str_lit("Int"), int_lit(*bits as i64), bool_lit(*signed)])
        }
        TypeData::UInt { bits } => {
            make_info(vec![str_lit("UInt"), int_lit(*bits as i64)])
        }
        TypeData::Float { bits } => {
            make_info(vec![str_lit("Float"), int_lit(*bits as i64)])
        }
        TypeData::Bool => ComptimeValue::Value(Arc::new(str_lit("Bool"))),
        TypeData::Char => ComptimeValue::Value(Arc::new(str_lit("Char"))),
        TypeData::Byte => ComptimeValue::Value(Arc::new(str_lit("Byte"))),
        TypeData::USize => ComptimeValue::Value(Arc::new(str_lit("USize"))),
        TypeData::Unit => ComptimeValue::Value(Arc::new(str_lit("Unit"))),
        TypeData::Never => ComptimeValue::Value(Arc::new(str_lit("Never"))),
        TypeData::Adt { kind: AdtKind::Struct, def_id, .. } => {
            let mut fields_tuple = Vec::new();
            if let Some(binding) = self.checker.symbols.lookup_type_by_def_id(*def_id) {
                for field in &binding.fields {
                    let field_info = make_info(vec![
                        str_lit(&field.name),
                        int_lit(field.ty.0 as i64),
                        str_lit(&format!("{:?}", ctx.get(field.ty))),
                    ]);
                    if let ComptimeValue::Value(hir) = &field_info {
                        fields_tuple.push(HirExpr::clone(hir.as_ref()));
                    }
                }
            }
            let fields_ty = ctx.tuple(
                fields_tuple.iter().map(|f| match f {
                    HirExpr::Tuple(.., ty, _) => *ty,
                    _ => unit_ty,
                }).collect()
            );
            make_info(vec![
                str_lit("Struct"),
                int_lit(def_id.0 as i64),
                int_lit(fields_tuple.len() as i64),
                HirExpr::Tuple(fields_tuple, fields_ty, Span::new(0, 0)),
            ])
        }
        TypeData::Adt { kind: AdtKind::Enum, def_id, .. } => {
            let mut variants_tuple = Vec::new();
            if let Some(binding) = self.checker.symbols.lookup_type_by_def_id(*def_id) {
                for variant in &binding.variants {
                    let payload_id = resolve_ast_type_def_id(
                        &variant.payload.as_ref().unwrap_or(&Type::Error(Span::new(0, 0))),
                        &self.checker.resolution_map,
                        self.checker.symbols,
                    ).unwrap_or(0);
                    let is_none = variant.payload.is_none();
                    let vinfo = make_info(vec![
                        str_lit(&variant.name),
                        int_lit(if is_none { -1i64 } else { payload_id }),
                    ]);
                    if let ComptimeValue::Value(hir) = &vinfo {
                        variants_tuple.push(HirExpr::clone(hir.as_ref()));
                    }
                }
            }
            let vars_ty = ctx.tuple(
                variants_tuple.iter().map(|f| match f {
                    HirExpr::Tuple(.., ty, _) => *ty,
                    _ => unit_ty,
                }).collect()
            );
            make_info(vec![
                str_lit("Enum"),
                int_lit(def_id.0 as i64),
                int_lit(variants_tuple.len() as i64),
                HirExpr::Tuple(variants_tuple, vars_ty, Span::new(0, 0)),
            ])
        }
        TypeData::Tuple { elems } => {
            let elem_ids: Vec<HirExpr> = elems.iter().map(|e| int_lit(e.0 as i64)).collect();
            let ids_ty = ctx.tuple(elems.iter().map(|_| usize_ty).collect());
            make_info(vec![
                str_lit("Tuple"),
                int_lit(elems.len() as i64),
                HirExpr::Tuple(elem_ids, ids_ty, Span::new(0, 0)),
            ])
        }
        TypeData::Array { elem, size } => {
            make_info(vec![str_lit("Array"), int_lit(elem.0 as i64), int_lit(*size as i64)])
        }
        TypeData::Slice { elem } => {
            make_info(vec![str_lit("Slice"), int_lit(elem.0 as i64)])
        }
        TypeData::Fn { params, ret } => {
            // Return: ("Fn", param_count, [param_ty_id, ...], ret_ty_id)
            let param_ids: Vec<HirExpr> = params.iter().map(|p| int_lit(p.0 as i64)).collect();
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Fn"),
                int_lit(params.len() as i64),
                HirExpr::Tuple(param_ids, unit_ty, Span::new(0, 0)),
                int_lit(ret.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Ref { ty, mutable } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Ref"),
                int_lit(ty.0 as i64),
                bool_lit(*mutable),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Pointer { ty } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Pointer"),
                int_lit(ty.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Ptr { size, pointee } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Ptr"),
                int_lit(size.0 as i64),
                int_lit(pointee.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Coproduct { alternatives } => {
            let alt_ids: Vec<HirExpr> = alternatives.iter().map(|a| int_lit(a.0 as i64)).collect();
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Coproduct"),
                int_lit(alternatives.len() as i64),
                HirExpr::Tuple(alt_ids, unit_ty, Span::new(0, 0)),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Forall { param_index, param_name, body } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Forall"),
                int_lit(*param_index as i64),
                str_lit(param_name),
                int_lit(body.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Exists { param_index: _, name, base } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Exists"),
                str_lit(name),
                int_lit(base.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Poly { quantifiers, body } => {
            let q_strings: Vec<HirExpr> = quantifiers.iter()
                .map(|(idx, name)| str_lit(&format!("{}:{}", idx, name)))
                .collect();
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Poly"),
                int_lit(quantifiers.len() as i64),
                HirExpr::Tuple(q_strings, unit_ty, Span::new(0, 0)),
                int_lit(body.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Rational { int_bits, frac_bits } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Rational"),
                int_lit(*int_bits as i64),
                int_lit(*frac_bits as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Mu { param_index, param_name, body } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Mu"),
                int_lit(*param_index as i64),
                int_lit(body.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::Nu { param_index, param_name, body } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("Nu"),
                int_lit(*param_index as i64),
                int_lit(body.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::DynTrait { traits } => {
            let trait_ids: Vec<HirExpr> = traits.iter()
                .map(|t| int_lit(t.0 as i64))
                .collect();
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("DynTrait"),
                int_lit(traits.len() as i64),
                HirExpr::Tuple(trait_ids, unit_ty, Span::new(0, 0)),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::GenericParam { index, name } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("GenericParam"),
                int_lit(*index as i64),
                str_lit(name),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::AssociatedType { trait_id, name, self_ty } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("AssociatedType"),
                int_lit(trait_id.0 as i64),
                str_lit(name),
                int_lit(self_ty.0 as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        TypeData::InferVar { id } => {
            ComptimeValue::Value(Arc::new(HirExpr::Tuple(vec![
                str_lit("InferVar"),
                int_lit(*id as i64),
            ], unit_ty, Span::new(0, 0)))
        }
        _ => {
            ComptimeValue::Value(Arc::new(str_lit("Other")))
        }
    }
}

    /// Estimate the compile-time size of a type (in bytes), including padding.
    fn estimate_type_size(&self, type_id: TypeId) -> u64 {
        const PTR_SIZE: u64 = 8;
        let data = self.checker.ctx.get(type_id);
        match data {
            TypeData::Int { bits, .. } | TypeData::UInt { bits } => {
                let bytes = (*bits as u64 + 7) / 8;
                // Round up to next power-of-two alignment
                let align = bytes.next_power_of_two();
                ((bytes + align - 1) / align) * align
            }
            TypeData::Float { bits } => {
                let bytes = (*bits as u64 + 7) / 8;
                bytes.max(4)
            }
            TypeData::Bool | TypeData::Char | TypeData::Byte => 1,
            TypeData::USize => PTR_SIZE,
            TypeData::Unit | TypeData::Never => 0,
            TypeData::Tuple { elems } => {
                let mut total = 0u64;
                let mut max_align = 1u64;
                for e in elems {
                    let size = self.estimate_type_size(*e);
                    let align = size.next_power_of_two();
                    max_align = max_align.max(align);
                    // Align current offset
                    total = ((total + align - 1) / align) * align;
                    total += size;
                }
                // Final alignment
                ((total + max_align - 1) / max_align) * max_align
            }
            TypeData::Array { elem, size } => {
                self.estimate_type_size(*elem) * size
            }
            TypeData::Adt { def_id, .. } => {
                // Compute struct size by summing field types (with alignment).
                // Field types are already resolved to TypeId by the resolver.
                if let Some(binding) = self.checker.symbols.lookup_type_by_def_id(*def_id) {
                    let mut total = 0u64;
                    let mut max_align = 1u64;
                    for field in &binding.fields {
                        let size = self.estimate_type_size(field.ty);
                        let align = size.next_power_of_two().max(1);
                        max_align = max_align.max(align);
                        total = ((total + align - 1) / align) * align;
                        total += size;
                    }
                    ((total + max_align - 1) / max_align) * max_align
                } else {
                    PTR_SIZE * 4
                }
            }
            TypeData::Adt { def_id, .. } => {
                // Enum size = max variant payload + discriminant tag.
                if let Some(binding) = self.checker.symbols.lookup_type_by_def_id(*def_id) {
                    let mut max_payload = 0u64;
                    for variant in &binding.variants {
                        match &variant.payload {
                            Some(ast_ty) => {
                                // Resolve AST type to a TypeId via the resolution map
                                if let Type::Path(path, _) = ast_ty.as_ref() {
                                    if let Some(&def_id) = self.checker.resolution_map.type_def_ids.get(&path[0]) {
                                        if let Some(ty_id) = self.checker.ctx.get_type_id_for_def_id(def_id) {
                                            let size = self.estimate_type_size(ty_id);
                                            max_payload = max_payload.max(size);
                                        }
                                    }
                                }
                            }
                            None => {} // No payload
                        }
                    }
                    let total = max_payload + 1; // +1 for discriminant
                    let align = max_payload.next_power_of_two().max(1);
                    ((total + align - 1) / align) * align
                } else {
                    PTR_SIZE * 4
                }
            }
            TypeData::Ref { .. } | TypeData::Pointer { .. } => PTR_SIZE,
            _ => PTR_SIZE,
        }
    }

    /// Evaluate a binary operation on two comptime values.
    fn eval_binary_op(
        &mut self,
        left: ComptimeValue,
        right: ComptimeValue,
        op: BinOp,
        ty: TypeId,
        span: Span,
    ) -> Result<ComptimeValue, ComptimeError> {
        match (left, right) {
            (ComptimeValue::Value(l), ComptimeValue::Value(r)) => {
                match (l.as_ref(), r.as_ref()) {
                    (HirExpr::Literal(l1, _, _), HirExpr::Literal(l2, _, _)) => {
                        match (l1, l2) {
                            // Integer arithmetic
                            (Literal::Int(a), Literal::Int(b)) => {
                                let result = match op {
                                    BinOp::Add => Literal::Int(a.wrapping_add(b)),
                                    BinOp::Sub => Literal::Int(a.wrapping_sub(b)),
                                    BinOp::Mul => Literal::Int(a.wrapping_mul(b)),
                                    BinOp::Div => {
                                        if *b == 0 { return Err(ComptimeError::DivisionByZero); }
                                        Literal::Int(a / b)
                                    }
                                    BinOp::Rem => {
                                        if *b == 0 { return Err(ComptimeError::DivisionByZero); }
                                        Literal::Int(a % b)
                                    }
                                    BinOp::Eq => Literal::Bool(a == b),
                                    BinOp::Neq => Literal::Bool(a != b),
                                    BinOp::Lt => Literal::Bool(a < b),
                                    BinOp::Gt => Literal::Bool(a > b),
                                    BinOp::Le => Literal::Bool(a <= b),
                                    BinOp::Ge => Literal::Bool(a >= b),
                                    BinOp::BitAnd => Literal::Int(a & b),
                                    BinOp::BitOr => Literal::Int(a | b),
                                    BinOp::BitXor => Literal::Int(a ^ b),
                                    BinOp::Shl => Literal::Int(a.wrapping_shl(*b as u32)),
                                    BinOp::Shr => Literal::Int(a.wrapping_shr(*b as u32)),
                                    _ => return Err(ComptimeError::Deferred),
                                };
                                Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(result, ty, span))))
                            }
                            // Float arithmetic
                            (Literal::Float(a), Literal::Float(b)) => {
                                let result = match op {
                                    BinOp::Add => Literal::Float(a + b),
                                    BinOp::Sub => Literal::Float(a - b),
                                    BinOp::Mul => Literal::Float(a * b),
                                    BinOp::Div => Literal::Float(a / b),
                                    BinOp::Eq => Literal::Bool(a == b),
                                    BinOp::Neq => Literal::Bool(a != b),
                                    BinOp::Lt => Literal::Bool(a < b),
                                    BinOp::Gt => Literal::Bool(a > b),
                                    BinOp::Le => Literal::Bool(a <= b),
                                    BinOp::Ge => Literal::Bool(a >= b),
                                    _ => return Err(ComptimeError::Deferred),
                                };
                                Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(result, ty, span))))
                            }
                            // Boolean logic
                            (Literal::Bool(a), Literal::Bool(b)) => {
                                let result = match op {
                                    BinOp::And => Literal::Bool(*a && *b),
                                    BinOp::Or => Literal::Bool(*a || *b),
                                    BinOp::Eq => Literal::Bool(a == b),
                                    BinOp::Neq => Literal::Bool(a != b),
                                    _ => return Err(ComptimeError::Deferred),
                                };
                                Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(result, ty, span))))
                            }
                            _ => Err(ComptimeError::Deferred),
                        }
                    }
                    _ => Err(ComptimeError::Deferred),
                }
            }
            (ComptimeValue::Value(l), ComptimeValue::Value(r)) => {
                Ok(ComptimeValue::Value(Arc::new(HirExpr::BinaryOp {
                    left: Box::new(HirExpr::clone(l.as_ref())),
                    op,
                    right: Box::new(HirExpr::clone(r.as_ref())),
                    ty,
                    span,
                })))
            }
            _ => Err(ComptimeError::Deferred),
        }
    }

    /// Evaluate a unary operation on a comptime value.
    fn eval_unary_op(
        &mut self,
        val: ComptimeValue,
        op: UnaryOp,
        ty: TypeId,
        span: Span,
    ) -> Result<ComptimeValue, ComptimeError> {
        match val {
            ComptimeValue::Value(hir) => match hir.as_ref() {
                HirExpr::Literal(lit, _, _) => {
                    match (op, lit) {
                        (UnaryOp::Neg, Literal::Int(v)) => {
                            Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(Literal::Int(-v), ty, span))))
                        }
                        (UnaryOp::Neg, Literal::Float(v)) => {
                            Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(Literal::Float(-v), ty, span))))
                        }
                        (UnaryOp::Not, Literal::Bool(v)) => {
                            Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(Literal::Bool(!v), ty, span))))
                        }
                        (UnaryOp::BitNot, Literal::Int(v)) => {
                            Ok(ComptimeValue::Value(Arc::new(HirExpr::Literal(Literal::Int(!v), ty, span))))
                        }
                        _ => Err(ComptimeError::Deferred),
                    }
                }
                _ => Ok(ComptimeValue::Value(Arc::new(HirExpr::UnaryOp {
                    op,
                    expr: Box::new(HirExpr::clone(hir.as_ref())),
                    ty,
                    span,
                }))),
            },
            _ => Err(ComptimeError::Deferred),
        }
    }
}

// We need HirStmt for the body evaluation
use crate::hir::hir::HirStmt;
