use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCollector, DiagnosticLevel};
use crate::hir::hir::*;
use crate::hir::infer::*;
use crate::hir::symbol::*;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use std::collections::HashSet;

pub struct TypeChecker<'a> {
    ctx: &'a mut TypeContext,
    symbols: &'a SymbolTable,
    trait_env: &'a TraitEnv,
    diagnostics: DiagnosticCollector,
    current_function: Option<DefId>,
    current_return_type: Option<TypeId>,
    current_scope: usize,
    next_var_id: usize,
    subst: Subst,
    resolving_aliases: HashSet<DefId>,
    infer: InferenceContext,
    infer_stack: Vec<InferenceContext>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        symbols: &'a SymbolTable,
        trait_env: &'a TraitEnv,
    ) -> Self {
        TypeChecker {
            ctx,
            symbols,
            trait_env,
            diagnostics: DiagnosticCollector::new(),
            current_function: None,
            current_return_type: None,
            current_scope: 0,
            next_var_id: 0,
            subst: Subst::new(),
            resolving_aliases: HashSet::new(),
            infer: InferenceContext::new(),
            infer_stack: Vec::new(),
        }
    }

    fn enter_inference_scope(&mut self) {
        let old = std::mem::replace(&mut self.infer, InferenceContext::new());
        self.infer_stack.push(old);
    }

    fn exit_inference_scope(&mut self) -> Result<(), DiagnosticCollector> {
        let mut current =
            std::mem::replace(&mut self.infer, self.infer_stack.pop().unwrap_or_default());
        if let Err(err) = current.solve(self.ctx, self.trait_env) {
            let diag = Diagnostic::error(format!("type inference error: {:?}", err))
                .with_span(Span::new(0, 0));
            self.diagnostics.push(diag);
            return Err(std::mem::take(&mut self.diagnostics));
        }
        let solution = current.finalize(self.ctx);
        Ok(())
    }

    pub fn check_program(&mut self, program: &Program) -> Result<HirProgram, DiagnosticCollector> {
        let mut items = Vec::new();
        for stmt in &program.items {
            match self.check_stmt(stmt) {
                Ok(hir) => items.push(hir),
                Err(diag) => {
                    self.diagnostics.push(diag);
                    items.push(HirStmt::Error);
                }
            }
        }

        if self.diagnostics.has_errors() {
            Err(std::mem::take(&mut self.diagnostics))
        } else {
            Ok(HirProgram {
                items,
                span: program.span,
            })
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<HirStmt, Diagnostic> {
        match stmt {
            Stmt::VariableDef {
                kind,
                mutable,
                name,
                pattern,
                ty,
                value,
                else_branch,
                span,
                attributes,
                doc,
            } => {
                let ty_id = if let Some(ty) = ty {
                    self.resolve_type(ty)?
                } else {
                    self.new_infer_var(TypeVariableKind::Unconstrained)
                };

                let value_hir = if let Some(value) = value {
                    let (hir, inferred_ty) = self.synthesize(value)?;
                    self.unify(ty_id, inferred_ty, *span)?;
                    Some(Box::new(hir))
                } else {
                    None
                };

                let pattern_hir = if let Some(pattern) = pattern {
                    Some(self.check_pattern(pattern, ty_id)?)
                } else {
                    None
                };

                let else_hir = if let Some(else_branch) = else_branch {
                    let mut stmts = Vec::new();
                    for s in else_branch {
                        stmts.push(self.check_stmt(s)?);
                    }
                    Some(stmts)
                } else {
                    None
                };

                let final_ty = if ty_id != self.ctx.error() {
                    ty_id
                } else if let Some(hir) = &value_hir {
                    hir.ty()
                } else {
                    self.ctx.error()
                };

                Ok(HirStmt::VariableDef {
                    kind: *kind,
                    mutable: *mutable,
                    name: name.clone(),
                    pattern: pattern_hir,
                    ty: final_ty,
                    value: value_hir,
                    else_branch: else_hir,
                    span: *span,
                })
            }
            Stmt::FunctionDef {
                span,
                attributes,
                contracts,
                doc,
                name,
                params,
                return_type,
                body,
                type_params,
                where_clause,
                finally,
                is_comptime,
                is_async,
            } => {
                let return_ty = self.resolve_type(return_type)?;
                let mut hir_params = Vec::new();
                for param in params {
                    let param_ty = if let Some(ty) = &param.ty {
                        self.resolve_type(ty)?
                    } else {
                        self.ctx.error()
                    };
                    hir_params.push(HirParam {
                        name: param.name.clone(),
                        ty: param_ty,
                        default: param.default.clone(),
                        span: param.span,
                    });
                }

                let old_function = self.current_function;
                let old_return = self.current_return_type;
                self.current_return_type = Some(return_ty);

                self.enter_inference_scope();

                let body_hir = if let Some(body) = body {
                    let mut stmts = Vec::new();
                    for s in body {
                        stmts.push(self.check_stmt(s)?);
                    }
                    Some(stmts)
                } else {
                    None
                };

                if let Err(err) = self.exit_inference_scope() {
                    return Err(Diagnostic::error("inference failure").with_span(*span));
                }

                self.current_return_type = old_return;
                self.current_function = old_function;

                let finally_hir = if let Some(finally) = finally {
                    let mut stmts = Vec::new();
                    for s in finally {
                        stmts.push(self.check_stmt(s)?);
                    }
                    Some(stmts)
                } else {
                    None
                };

                Ok(HirStmt::FunctionDef {
                    span: *span,
                    attributes: attributes.clone(),
                    contracts: contracts.clone(),
                    doc: doc.clone(),
                    name: name.clone(),
                    params: hir_params,
                    return_type: return_ty,
                    body: body_hir,
                    type_params: type_params.clone(),
                    where_clause: where_clause.clone().map(|_| ()),
                    finally: finally_hir,
                    is_comptime: *is_comptime,
                    is_async: *is_async,
                })
            }
            Stmt::Expression(expr) => {
                let (hir, _) = self.synthesize(expr)?;
                Ok(HirStmt::Expression(Box::new(hir)))
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                let (cond_hir, cond_ty) = self.synthesize(cond)?;
                if !self.ctx.is_bool(cond_ty) {
                    self.diagnostics
                        .push(Diagnostic::error("if condition must be boolean").with_span(*span));
                }

                let then_hir = self.check_block(then_branch)?;
                let else_hir = if let Some(else_branch) = else_branch {
                    Some(self.check_block(else_branch)?)
                } else {
                    None
                };

                Ok(HirStmt::If {
                    cond: Box::new(cond_hir),
                    then_branch: then_hir,
                    else_branch: else_hir,
                    span: *span,
                })
            }
            Stmt::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.synthesize(scrutinee)?;
                let pattern_hir = self.check_pattern(pattern, scrut_ty)?;
                let then_hir = self.check_block(then_branch)?;
                let else_hir = if let Some(else_branch) = else_branch {
                    Some(self.check_block(else_branch)?)
                } else {
                    None
                };

                Ok(HirStmt::IfLet {
                    pattern: pattern_hir,
                    scrutinee: Box::new(scrut_hir),
                    then_branch: then_hir,
                    else_branch: else_hir,
                    span: *span,
                })
            }
            Stmt::While {
                cond,
                body,
                invariant,
                decreases,
                span,
            } => {
                let (cond_hir, cond_ty) = self.synthesize(cond)?;
                if !self.ctx.is_bool(cond_ty) {
                    self.diagnostics.push(
                        Diagnostic::error("while condition must be boolean").with_span(*span),
                    );
                }

                let inv_hir = if let Some(inv) = invariant {
                    Some(self.synthesize(inv)?.0)
                } else {
                    None
                };

                let dec_hir = if let Some(dec) = decreases {
                    Some(self.synthesize(dec)?.0)
                } else {
                    None
                };

                let body_hir = self.check_block(body)?;

                Ok(HirStmt::While {
                    cond: Box::new(cond_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::WhileLet {
                pattern,
                scrutinee,
                body,
                invariant,
                decreases,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.synthesize(scrutinee)?;
                let pattern_hir = self.check_pattern(pattern, scrut_ty)?;

                let inv_hir = if let Some(inv) = invariant {
                    Some(self.synthesize(inv)?.0)
                } else {
                    None
                };

                let dec_hir = if let Some(dec) = decreases {
                    Some(self.synthesize(dec)?.0)
                } else {
                    None
                };

                let body_hir = self.check_block(body)?;

                Ok(HirStmt::WhileLet {
                    pattern: pattern_hir,
                    scrutinee: Box::new(scrut_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::For {
                pattern,
                iterable,
                body,
                invariant,
                decreases,
                span,
            } => {
                let (iter_hir, iter_ty) = self.synthesize(iterable)?;
                let elem_ty = if let Some(slice_ty) = self.ctx.elem_of_slice(iter_ty) {
                    slice_ty
                } else if let Some(arr_ty) = self.ctx.elem_of_array(iter_ty) {
                    arr_ty
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("for loop iterable must be an array or slice")
                            .with_span(*span),
                    );
                    self.ctx.error()
                };

                let pattern_hir = self.check_pattern(pattern, elem_ty)?;

                let inv_hir = if let Some(inv) = invariant {
                    Some(self.synthesize(inv)?.0)
                } else {
                    None
                };

                let dec_hir = if let Some(dec) = decreases {
                    Some(self.synthesize(dec)?.0)
                } else {
                    None
                };

                let body_hir = self.check_block(body)?;

                Ok(HirStmt::For {
                    pattern: pattern_hir,
                    iterable: Box::new(iter_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::Loop { body, span } => {
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::Loop {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::Leave { label, span } => Ok(HirStmt::Leave {
                label: label.clone(),
                span: *span,
            }),
            Stmt::Continue { label, span } => Ok(HirStmt::Continue {
                label: label.clone(),
                span: *span,
            }),
            Stmt::Return { value, span } => {
                if let Some(value) = value {
                    let (hir, ty) = self.synthesize(value)?;
                    if let Some(ret_ty) = self.current_return_type {
                        self.unify(ret_ty, ty, *span)?;
                    }
                    Ok(HirStmt::Return {
                        value: Some(Box::new(hir)),
                        span: *span,
                    })
                } else {
                    if let Some(ret_ty) = self.current_return_type {
                        if !self.ctx.is_unit(ret_ty) && !self.ctx.is_never(ret_ty) {
                            self.diagnostics.push(
                                Diagnostic::error("return without value in non-unit function")
                                    .with_span(*span),
                            );
                        }
                    }
                    Ok(HirStmt::Return {
                        value: None,
                        span: *span,
                    })
                }
            }
            Stmt::Assign {
                target,
                op,
                value,
                span,
            } => {
                let (target_hir, target_ty) = self.synthesize(target)?;
                let (value_hir, value_ty) = self.synthesize(value)?;

                if let Some(op) = op {
                    let result_ty = self.binary_op_type(*op, target_ty, value_ty, *span)?;
                    self.unify(target_ty, result_ty, *span)?;
                } else {
                    self.unify(target_ty, value_ty, *span)?;
                }

                Ok(HirStmt::Assign {
                    target: Box::new(target_hir),
                    op: *op,
                    value: Box::new(value_hir),
                    span: *span,
                })
            }
            Stmt::ComptimeBlock { body, span } => {
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::ComptimeBlock {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::ScopeCleanup {
                name,
                body,
                propagates,
                overrides,
                span,
            } => {
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::ScopeCleanup {
                    name: name.clone(),
                    body: body_hir,
                    propagates: *propagates,
                    overrides: *overrides,
                    span: *span,
                })
            }
            Stmt::Trigger { name, span } => Ok(HirStmt::Trigger {
                name: name.clone(),
                span: *span,
            }),
            Stmt::Unsafe { body, span } => {
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::Unsafe {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::GhostVariableDef { inner, span } => {
                let inner_hir = self.check_stmt(inner)?;
                Ok(HirStmt::GhostVariableDef {
                    inner: Box::new(inner_hir),
                    span: *span,
                })
            }
            Stmt::Isolate { body, span } => {
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::Isolate {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::TypeDef { .. } => Ok(HirStmt::Error),
            Stmt::TraitDef { .. } => Ok(HirStmt::Error),
            Stmt::Import { .. } => Ok(HirStmt::Error),
            Stmt::ExternFunction { .. } => Ok(HirStmt::Error),
            Stmt::Constraint { .. } => Ok(HirStmt::Error),
            Stmt::Edition(..) => Ok(HirStmt::Error),
            Stmt::ImplBlock { .. } => Ok(HirStmt::Error),
            Stmt::Error(span) => Err(Diagnostic::error("invalid statement").with_span(*span)),
        }
    }

    fn check_block(&mut self, stmts: &[Stmt]) -> Result<Vec<HirStmt>, Diagnostic> {
        let mut result = Vec::new();
        for stmt in stmts {
            result.push(self.check_stmt(stmt)?);
        }
        Ok(result)
    }

    fn check_pattern(
        &mut self,
        pattern: &Pattern,
        expected_ty: TypeId,
    ) -> Result<HirPattern, Diagnostic> {
        match pattern {
            Pattern::Wildcard(span) => Ok(HirPattern::Wildcard(*span)),
            Pattern::Ident(name, span) => Ok(HirPattern::Ident(name.clone(), expected_ty, *span)),
            Pattern::Literal(expr, span) => {
                let (hir, ty) = self.synthesize(expr)?;
                self.unify(expected_ty, ty, *span)?;
                Ok(HirPattern::Literal(Box::new(hir), *span))
            }
            Pattern::Tuple(patterns, span) => {
                let expected_elems = if let Some(elems) = self.ctx.tuple_elems(expected_ty) {
                    elems.to_vec()
                } else if self.ctx.is_error(expected_ty) {
                    vec![self.ctx.error(); patterns.len()]
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "expected tuple type, found {:?}",
                            self.ctx.get(expected_ty)
                        ))
                        .with_span(*span),
                    );
                    vec![self.ctx.error(); patterns.len()]
                };

                let mut hir_patterns = Vec::new();
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_ty = if i < expected_elems.len() {
                        expected_elems[i]
                    } else {
                        self.ctx.error()
                    };
                    hir_patterns.push(self.check_pattern(pat, elem_ty)?);
                }
                Ok(HirPattern::Tuple(hir_patterns, *span))
            }
            Pattern::Struct { path, fields, span } => {
                let def_id = self.resolve_def_id(path)?;
                let binding = self
                    .symbols
                    .lookup_type_by_def_id(def_id)
                    .ok_or_else(|| Diagnostic::error("struct not found").with_span(*span))?;
                if let TypeKind::Struct = binding.kind {
                    let type_args = self.collect_type_args(path, *span)?;
                    let mut subst = Subst::new();
                    for (i, param) in binding.params.iter().enumerate() {
                        if let Some(&arg) = type_args.get(i) {
                            subst.insert(i, arg);
                        }
                    }
                    let mut hir_fields = Vec::new();
                    for (name, pat) in fields {
                        let field_ty = binding
                            .fields
                            .iter()
                            .find(|f| f.name == *name)
                            .map(|f| self.ctx.subst(f.ty, &subst))
                            .ok_or_else(|| {
                                Diagnostic::error(format!("field '{}' not found", name))
                                    .with_span(*span)
                            })?;
                        let hir_pat = self.check_pattern(pat, field_ty)?;
                        hir_fields.push((name.clone(), Box::new(hir_pat)));
                    }
                    Ok(HirPattern::Struct {
                        path: path.clone(),
                        fields: hir_fields,
                        span: *span,
                    })
                } else {
                    Err(Diagnostic::error("pattern type is not a struct").with_span(*span))
                }
            }
            Pattern::Enum {
                path,
                variant,
                inner,
                span,
            } => {
                let def_id = self.resolve_def_id(path)?;
                let binding = self
                    .symbols
                    .lookup_type_by_def_id(def_id)
                    .ok_or_else(|| Diagnostic::error("enum not found").with_span(*span))?;
                if let TypeKind::Enum = binding.kind {
                    let type_args = self.collect_type_args(path, *span)?;
                    let mut subst = Subst::new();
                    for (i, param) in binding.params.iter().enumerate() {
                        if let Some(&arg) = type_args.get(i) {
                            subst.insert(i, arg);
                        }
                    }
                    let variant_def = binding
                        .variants
                        .iter()
                        .find(|v| v.name == *variant)
                        .ok_or_else(|| {
                            Diagnostic::error(format!("variant '{}' not found", variant))
                                .with_span(*span)
                        })?;
                    let inner_ty = match &variant_def.payload {
                        Some(payload_ty) => self.resolve_type(payload_ty)?,
                        None => self.ctx.error(),
                    };
                    let inner_hir = if let Some(inner) = inner {
                        Some(Box::new(self.check_pattern(inner, inner_ty)?))
                    } else {
                        None
                    };
                    Ok(HirPattern::Enum {
                        path: path.clone(),
                        variant: variant.clone(),
                        inner: inner_hir,
                        span: *span,
                    })
                } else {
                    Err(Diagnostic::error("pattern type is not an enum").with_span(*span))
                }
            }
            Pattern::Or(patterns, span) => {
                let mut hir_patterns = Vec::new();
                for pat in patterns {
                    hir_patterns.push(self.check_pattern(pat, expected_ty)?);
                }
                Ok(HirPattern::Or(hir_patterns, *span))
            }
            Pattern::Error(span) => Ok(HirPattern::Error(*span)),
        }
    }

    fn synthesize(&mut self, expr: &Expr) -> Result<(HirExpr, TypeId), Diagnostic> {
        match expr {
            Expr::Literal(lit, span) => {
                let kind = match lit {
                    Literal::Int(_) => TypeVariableKind::Integer,
                    Literal::Float(_) => TypeVariableKind::Float,
                    Literal::Bool(_) => TypeVariableKind::Bool,
                    Literal::Char(_) => TypeVariableKind::Any,
                    Literal::String(_) | Literal::ByteString(_) => TypeVariableKind::Any,
                };
                let ty = self.new_infer_var(kind);
                Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
            }
            Expr::Ident(name, span) => {
                if let Some(binding) = self.symbols.lookup_variable(name, *span) {
                    Ok((HirExpr::Ident(name.clone(), binding.ty, *span), binding.ty))
                } else if let Some(func) = self.symbols.lookup_function(name) {
                    let sig = func.signature.clone();
                    let ty = self
                        .ctx
                        .function(sig.params.iter().map(|p| p.ty).collect(), sig.return_type);
                    Ok((HirExpr::Ident(name.clone(), ty, *span), ty))
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined name: {}", name)).with_span(*span),
                    );
                    Ok((HirExpr::Error(*span), self.ctx.error()))
                }
            }
            Expr::TypeAnnotated { expr, ty, span } => {
                let expected_ty = self.resolve_type(ty)?;
                let (hir, actual_ty) = self.synthesize(expr)?;
                self.unify(expected_ty, actual_ty, *span)?;
                Ok((
                    HirExpr::TypeAnnotated {
                        expr: Box::new(hir),
                        ty: expected_ty,
                        span: *span,
                    },
                    expected_ty,
                ))
            }
            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let trait_id = self
                    .get_trait_id_for_binop(*op, *span)
                    .unwrap_or(DefId(usize::MAX));
                let result_ty = match op {
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                    | BinOp::And | BinOp::Or => self.ctx.bool(),
                    _ => self.new_infer_var(TypeVariableKind::Numeric),
                };

                let (left_hir, left_ty) = self.synthesize(left)?;
                let (right_hir, right_ty) = self.synthesize(right)?;

                self.unify(left_ty, right_ty, *span)?;
                // Only add trait constraints if the trait was found
                if trait_id != DefId(usize::MAX) {
                    self.add_constraint(Constraint::Impl(left_ty, trait_id, *span));
                    self.add_constraint(Constraint::Impl(right_ty, trait_id, *span));
                }
                // Only unify result with left_ty for non-comparison ops
                match op {
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                    | BinOp::And | BinOp::Or => {}
                    _ => {
                        self.add_constraint(Constraint::Eq(result_ty, left_ty, *span));
                    }
                }

                Ok((
                    HirExpr::BinaryOp {
                        left: Box::new(left_hir),
                        op: *op,
                        right: Box::new(right_hir),
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::UnaryOp { op, expr, span } => {
                let operand_ty = self.new_infer_var(TypeVariableKind::Numeric);
                let result_ty = self.new_infer_var(TypeVariableKind::Numeric);
                let (hir, actual) = self.synthesize(expr)?;
                self.add_constraint(Constraint::Eq(operand_ty, actual, *span));
                self.add_constraint(Constraint::Eq(result_ty, operand_ty, *span));

                Ok((
                    HirExpr::UnaryOp {
                        op: *op,
                        expr: Box::new(hir),
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::Call {
                callee,
                args,
                comptime,
                span,
            } => {
                let (callee_hir, callee_ty) = self.synthesize(callee)?;

                let (param_tys, ret_ty) = if let Some(params) = self.ctx.params_of_fn(callee_ty) {
                    (
                        params.to_vec(),
                        self.ctx.ret_of_fn(callee_ty).unwrap_or(self.ctx.error()),
                    )
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("called expression is not a function").with_span(*span),
                    );
                    (vec![self.ctx.error(); args.len()], self.ctx.error())
                };

                if param_tys.len() != args.len() {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "wrong number of arguments: expected {}, found {}",
                            param_tys.len(),
                            args.len()
                        ))
                        .with_span(*span),
                    );
                }

                let mut hir_args = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    let expected = if i < param_tys.len() {
                        param_tys[i]
                    } else {
                        self.ctx.error()
                    };
                    let (hir, actual) = self.synthesize(arg)?;
                    self.unify(expected, actual, hir.span())?;
                    hir_args.push(hir);
                }
                Ok((
                    HirExpr::Call {
                        callee: Box::new(callee_hir),
                        args: hir_args,
                        comptime: *comptime,
                        ty: ret_ty,
                        span: *span,
                    },
                    ret_ty,
                ))
            }
            Expr::Index { base, index, span } => {
                let base_ty = self.new_infer_var(TypeVariableKind::Any);
                let index_ty = self.new_infer_var(TypeVariableKind::Integer);
                let elem_ty = self.new_infer_var(TypeVariableKind::Any);
                let (base_hir, base_actual) = self.synthesize(base)?;
                let (index_hir, index_actual) = self.synthesize(index)?;
                self.add_constraint(Constraint::Eq(base_ty, base_actual, *span));
                self.add_constraint(Constraint::Eq(index_ty, index_actual, *span));
                let resolved_elem = if let Some(slice_ty) = self.ctx.elem_of_slice(base_actual) {
                    slice_ty
                } else if let Some(arr_ty) = self.ctx.elem_of_array(base_actual) {
                    arr_ty
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("indexing on non-array/non-slice type").with_span(*span),
                    );
                    self.ctx.error()
                };
                self.add_constraint(Constraint::Eq(elem_ty, resolved_elem, *span));
                if !self.ctx.is_integer(index_actual) && !self.ctx.is_usize(index_actual) {
                    self.diagnostics
                        .push(Diagnostic::error("index must be an integer").with_span(*span));
                }
                Ok((
                    HirExpr::Index {
                        base: Box::new(base_hir),
                        index: Box::new(index_hir),
                        ty: elem_ty,
                        span: *span,
                    },
                    elem_ty,
                ))
            }
            Expr::FieldAccess { base, field, span } => {
                let base_ty = self.new_infer_var(TypeVariableKind::Any);
                let (base_hir, base_actual) = self.synthesize(base)?;
                self.add_constraint(Constraint::Eq(base_ty, base_actual, *span));
                let field_ty = self.lookup_field(base_actual, field, *span)?;
                Ok((
                    HirExpr::FieldAccess {
                        base: Box::new(base_hir),
                        field: field.clone(),
                        ty: field_ty,
                        span: *span,
                    },
                    field_ty,
                ))
            }
            Expr::AttrAccess { base, attr, span } => {
                let base_ty = self.new_infer_var(TypeVariableKind::Any);
                let (base_hir, base_actual) = self.synthesize(base)?;
                self.add_constraint(Constraint::Eq(base_ty, base_actual, *span));
                let attr_ty = self.lookup_attr(base_actual, attr, *span)?;
                Ok((
                    HirExpr::AttrAccess {
                        base: Box::new(base_hir),
                        attr: attr.clone(),
                        ty: attr_ty,
                        span: *span,
                    },
                    attr_ty,
                ))
            }
            Expr::Cast {
                expr,
                ty,
                safe,
                rounding,
                span,
            } => {
                let (hir, actual_ty) = self.synthesize(expr)?;
                let target_ty = self.resolve_type(ty)?;
                let cast_ty = self.check_cast(actual_ty, target_ty, *safe, *span)?;
                Ok((
                    HirExpr::Cast {
                        expr: Box::new(hir),
                        ty: cast_ty,
                        safe: *safe,
                        rounding: *rounding,
                        span: *span,
                    },
                    cast_ty,
                ))
            }
            Expr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let start_hir = if let Some(start) = start {
                    Some(Box::new(self.synthesize(start)?.0))
                } else {
                    None
                };
                let end_hir = if let Some(end) = end {
                    Some(Box::new(self.synthesize(end)?.0))
                } else {
                    None
                };
                let int_ty = self.ctx.int(32, true);
                let ty = self.ctx.tuple(vec![int_ty, int_ty]);
                Ok((
                    HirExpr::Range {
                        start: start_hir,
                        end: end_hir,
                        inclusive: *inclusive,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::StructLit { path, fields, span } => {
                let def_id = self.resolve_def_id(path)?;
                let binding = self.symbols.lookup_type_by_def_id(def_id).ok_or_else(|| {
                    Diagnostic::error(format!("struct not found: {:?}", path)).with_span(*span)
                })?;
                let type_args = self.collect_type_args(path, *span)?;
                let struct_ty = self.ctx.struct_ty(def_id, type_args.clone());
                let mut subst = Subst::new();
                for (i, param) in binding.params.iter().enumerate() {
                    if let Some(&arg) = type_args.get(i) {
                        subst.insert(i, arg);
                    }
                }

                let mut hir_fields = Vec::new();
                for (name, value) in fields {
                    let field_def =
                        binding
                            .fields
                            .iter()
                            .find(|f| f.name == *name)
                            .ok_or_else(|| {
                                Diagnostic::error(format!("field '{}' not found", name))
                                    .with_span(*span)
                            })?;
                    let field_ty = self.ctx.subst(field_def.ty, &subst);
                    let (hir, _) = self.synthesize(value)?;
                    self.unify(field_ty, hir.ty(), *span)?;
                    hir_fields.push((name.clone(), Box::new(hir)));
                }

                Ok((
                    HirExpr::StructLit {
                        path: path.clone(),
                        fields: hir_fields,
                        ty: struct_ty,
                        span: *span,
                    },
                    struct_ty,
                ))
            }
            Expr::EnumLit {
                path,
                variant,
                payload,
                span,
            } => {
                let def_id = self.resolve_def_id(path)?;
                let binding = self.symbols.lookup_type_by_def_id(def_id).ok_or_else(|| {
                    Diagnostic::error(format!("enum not found: {:?}", path)).with_span(*span)
                })?;
                let type_args = self.collect_type_args(path, *span)?;
                let enum_ty = self.ctx.enum_ty(def_id, type_args.clone());
                let mut subst = Subst::new();
                for (i, param) in binding.params.iter().enumerate() {
                    if let Some(&arg) = type_args.get(i) {
                        subst.insert(i, arg);
                    }
                }

                let payload_hir = if let Some(payload) = payload {
                    let variant_def = binding
                        .variants
                        .iter()
                        .find(|v| v.name == *variant)
                        .ok_or_else(|| {
                            Diagnostic::error(format!("variant '{}' not found", variant))
                                .with_span(*span)
                        })?;
                    let payload_ty = match &variant_def.payload {
                        Some(payload_ty) => self.resolve_type(payload_ty)?,
                        None => self.ctx.error(),
                    };
                    let (hir, _) = self.synthesize(payload)?;
                    self.unify(payload_ty, hir.ty(), *span)?;
                    Some(Box::new(hir))
                } else {
                    None
                };

                Ok((
                    HirExpr::EnumLit {
                        path: path.clone(),
                        variant: variant.clone(),
                        payload: payload_hir,
                        ty: enum_ty,
                        span: *span,
                    },
                    enum_ty,
                ))
            }
            Expr::Move(expr, span) => {
                let (hir, ty) = self.synthesize(expr)?;
                Ok((HirExpr::Move(Box::new(hir), ty, *span), ty))
            }
            Expr::Tuple(exprs, span) => {
                let mut hirs = Vec::new();
                let mut types = Vec::new();
                for e in exprs {
                    let (hir, ty) = self.synthesize(e)?;
                    hirs.push(hir);
                    types.push(ty);
                }
                let ty = self.ctx.tuple(types);
                Ok((HirExpr::Tuple(hirs, ty, *span), ty))
            }
            Expr::Array(exprs, span) => {
                let mut hirs = Vec::new();
                let mut elem_ty = None;
                for e in exprs {
                    let (hir, ty) = self.synthesize(e)?;
                    hirs.push(hir);
                    if elem_ty.is_none() {
                        elem_ty = Some(ty);
                    } else {
                        self.unify(elem_ty.unwrap(), ty, *span)?;
                    }
                }
                let ty = self
                    .ctx
                    .array(elem_ty.unwrap_or(self.ctx.error()), exprs.len() as u64);
                Ok((HirExpr::Array(hirs, ty, *span), ty))
            }
            Expr::Closure {
                params,
                return_type,
                captures,
                body,
                span,
            } => {
                let mut hir_params = Vec::new();
                let mut param_tys = Vec::new();
                for param in params {
                    let ty = if let Some(ty) = &param.ty {
                        self.resolve_type(ty)?
                    } else {
                        self.ctx.error()
                    };
                    hir_params.push(HirParam {
                        name: param.name.clone(),
                        ty,
                        default: None,
                        span: param.span,
                    });
                    param_tys.push(ty);
                }

                let ret_ty = if let Some(ret) = return_type {
                    self.resolve_type(ret)?
                } else {
                    self.ctx.unit()
                };

                let body_hir = self.check_block(body)?;
                let ty = self.ctx.function(param_tys, ret_ty);

                Ok((
                    HirExpr::Closure {
                        params: hir_params,
                        return_type: ret_ty,
                        captures: captures.clone(),
                        body: body_hir,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::Try { expr, span } => {
                let (hir, ty) = self.synthesize(expr)?;
                let result_ty = self.check_result_type(ty, *span)?;
                Ok((
                    HirExpr::Try {
                        expr: Box::new(hir),
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::UnsafeBlock { body, span } => {
                let body_hir = self.check_block(body)?;
                let ty = self.ctx.unit();
                Ok((
                    HirExpr::UnsafeBlock {
                        body: body_hir,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::Catch {
                expr,
                branches,
                span,
            } => {
                let (expr_hir, expr_ty) = self.synthesize(expr)?;
                let (ok_ty, error_ty) = self.extract_result_types(expr_ty, *span)?;

                let mut hir_branches = Vec::new();
                for branch in branches {
                    let pattern_hir = self.check_pattern(&branch.pattern, error_ty)?;
                    let body_hir = self.check_block(&branch.body)?;
                    hir_branches.push(HirCatchBranch {
                        pattern: pattern_hir,
                        bind: branch.bind.clone(),
                        body: body_hir,
                        span: branch.span,
                    });
                }

                Ok((
                    HirExpr::Catch {
                        expr: Box::new(expr_hir),
                        branches: hir_branches,
                        ty: ok_ty,
                        span: *span,
                    },
                    ok_ty,
                ))
            }
            Expr::LeaveWith { expr, span } => {
                let (hir, ty) = self.synthesize(expr)?;
                let never = self.ctx.never();
                Ok((
                    HirExpr::LeaveWith {
                        expr: Box::new(hir),
                        ty: never,
                        span: *span,
                    },
                    never,
                ))
            }
            Expr::Await { expr, span } => {
                let (hir, ty) = self.synthesize(expr)?;
                let future_ty = self.check_future_type(ty, *span)?;
                Ok((
                    HirExpr::Await {
                        expr: Box::new(hir),
                        ty: future_ty,
                        span: *span,
                    },
                    future_ty,
                ))
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                is_expression,
                span,
            } => {
                let (cond_hir, cond_ty) = self.synthesize(cond)?;
                if !self.ctx.is_bool(cond_ty) {
                    self.diagnostics
                        .push(Diagnostic::error("if condition must be boolean").with_span(*span));
                }

                let then_hir = self.check_block(then_branch)?;
                let then_ty = self.block_type(&then_hir);

                let else_hir = if let Some(else_branch) = else_branch {
                    Some(self.check_block(else_branch)?)
                } else {
                    None
                };

                let result_ty = if *is_expression {
                    let else_ty = if let Some(else_hir) = &else_hir {
                        self.block_type(else_hir)
                    } else {
                        self.ctx.unit()
                    };
                    self.unify(then_ty, else_ty, *span)?;
                    then_ty
                } else {
                    self.ctx.unit()
                };

                Ok((
                    HirExpr::If {
                        cond: Box::new(cond_hir),
                        then_branch: then_hir,
                        else_branch: else_hir,
                        is_expression: *is_expression,
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.synthesize(scrutinee)?;
                let pattern_hir = self.check_pattern(pattern, scrut_ty)?;
                let then_hir = self.check_block(then_branch)?;
                let else_hir = if let Some(else_branch) = else_branch {
                    Some(self.check_block(else_branch)?)
                } else {
                    None
                };

                let ty = self.ctx.unit();
                Ok((
                    HirExpr::IfLet {
                        pattern: pattern_hir,
                        scrutinee: Box::new(scrut_hir),
                        then_branch: then_hir,
                        else_branch: else_hir,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::Match {
                scrutinee,
                arms,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.synthesize(scrutinee)?;

                let mut hir_arms = Vec::new();
                let mut arm_ty = None;
                for arm in arms {
                    let pattern_hir = self.check_pattern(&arm.pattern, scrut_ty)?;
                    let guard_hir = if let Some(guard) = &arm.guard {
                        let (hir, ty) = self.synthesize(guard)?;
                        if !self.ctx.is_bool(ty) {
                            self.diagnostics.push(
                                Diagnostic::error("match guard must be boolean")
                                    .with_span(arm.span),
                            );
                        }
                        Some(Box::new(hir))
                    } else {
                        None
                    };
                    let (body_hir, body_ty) = self.synthesize(&arm.body)?;
                    if arm_ty.is_none() {
                        arm_ty = Some(body_ty);
                    } else {
                        self.unify(arm_ty.unwrap(), body_ty, arm.span)?;
                    }
                    hir_arms.push(HirMatchArm {
                        pattern: pattern_hir,
                        guard: guard_hir,
                        body: Box::new(body_hir),
                        span: arm.span,
                    });
                }

                let result_ty = arm_ty.unwrap_or(self.ctx.unit());
                Ok((
                    HirExpr::Match {
                        scrutinee: Box::new(scrut_hir),
                        arms: hir_arms,
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::Block(stmts, span) => {
                let hir_stmts = self.check_block(stmts)?;
                let ty = self.block_type(&hir_stmts);
                Ok((HirExpr::Block(hir_stmts, ty, *span), ty))
            }
            Expr::Error(span) => Ok((HirExpr::Error(*span), self.ctx.error())),
        }
    }

    fn resolve_type(&mut self, ty: &Type) -> Result<TypeId, Diagnostic> {
        match ty {
            Type::Path(path, span) => {
                let def_id = match self.resolve_def_id(path) {
                    Ok(did) => did,
                    Err(_) => {
                        let name = &path[0];
                        return match name.as_str() {
                            "Bool" => Ok(self.ctx.bool()),
                            "Char" => Ok(self.ctx.char()),
                            "Byte" => Ok(self.ctx.byte()),
                            "USize" => Ok(self.ctx.usize()),
                            "Unit" => Ok(self.ctx.unit()),
                            "Never" => Ok(self.ctx.never()),
                            _ => Err(
                                Diagnostic::error(format!("type '{}' not found", name))
                                    .with_span(*span),
                            ),
                        };
                    }
                };
                let binding = self.symbols.lookup_type_by_def_id(def_id).ok_or_else(|| {
                    Diagnostic::error(format!("type not found: {:?}", path)).with_span(*span)
                })?;
                match binding.kind {
                    TypeKind::Alias => {
                        if self.resolving_aliases.contains(&def_id) {
                            return Err(
                                Diagnostic::error("circular alias definition").with_span(*span)
                            );
                        }
                        self.resolving_aliases.insert(def_id);
                        let result = if let Some(ast) = &binding.alias_ast {
                            self.resolve_type(ast)
                        } else {
                            Err(Diagnostic::error("alias has no body").with_span(*span))
                        };
                        self.resolving_aliases.remove(&def_id);
                        result
                    }
                    TypeKind::Struct => {
                        let args = self.collect_type_args(path, *span)?;
                        Ok(self.ctx.struct_ty(def_id, args))
                    }
                    TypeKind::Enum => {
                        let args = self.collect_type_args(path, *span)?;
                        Ok(self.ctx.enum_ty(def_id, args))
                    }
                    _ => {
                        Err(Diagnostic::error("expected type, found something else")
                            .with_span(*span))
                    }
                }
            }
            Type::Generic(base, args, span) => {
                // Handle generic built-in types
                if let Type::Path(path, _) = base.as_ref() {
                    if path.len() == 1 {
                        match path[0].as_str() {
                            "Int" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(32);
                                return Ok(self.ctx.int(bits, true));
                            }
                            "UInt" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(32);
                                return Ok(self.ctx.int(bits, false));
                            }
                            "Float" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(64);
                                return Ok(self.ctx.float(bits));
                            }
                            _ => {}
                        }
                    }
                }
                let base_ty = self.resolve_type(base)?;
                let expanded_base = self.expand_base_type(base_ty, *span)?;
                let mut arg_tys = Vec::new();
                for arg in args {
                    arg_tys.push(self.resolve_type(arg)?);
                }
                match self.ctx.get(expanded_base) {
                    TypeData::Struct { def_id, .. } => {
                        let binding =
                            self.symbols.lookup_type_by_def_id(*def_id).ok_or_else(|| {
                                Diagnostic::error("struct definition not found").with_span(*span)
                            })?;
                        if arg_tys.len() != binding.params.len() {
                            return Err(Diagnostic::error(format!(
                                "wrong number of type arguments: expected {}, got {}",
                                binding.params.len(),
                                arg_tys.len()
                            ))
                            .with_span(*span));
                        }
                        Ok(self.ctx.struct_ty(*def_id, arg_tys))
                    }
                    TypeData::Enum { def_id, .. } => {
                        let binding =
                            self.symbols.lookup_type_by_def_id(*def_id).ok_or_else(|| {
                                Diagnostic::error("enum definition not found").with_span(*span)
                            })?;
                        if arg_tys.len() != binding.params.len() {
                            return Err(Diagnostic::error(format!(
                                "wrong number of type arguments: expected {}, got {}",
                                binding.params.len(),
                                arg_tys.len()
                            ))
                            .with_span(*span));
                        }
                        Ok(self.ctx.enum_ty(*def_id, arg_tys))
                    }
                    _ => Err(
                        Diagnostic::error("generic type arguments on non-generic type")
                            .with_span(*span),
                    ),
                }
            }
            Type::Reference(ty, mutable, span) => {
                let inner = self.resolve_type(ty)?;
                Ok(self.ctx.reference(inner, *mutable))
            }
            Type::Pointer(ty, span) => {
                let inner = self.resolve_type(ty)?;
                Ok(self.ctx.pointer(inner))
            }
            Type::Slice(ty, span) => {
                let inner = self.resolve_type(ty)?;
                Ok(self.ctx.slice(inner))
            }
            Type::Array(ty, size, span) => {
                let inner = self.resolve_type(ty)?;
                if let Expr::Literal(Literal::Int(size_val), _) = size.as_ref() {
                    Ok(self.ctx.array(inner, *size_val as u64))
                } else {
                    Err(
                        Diagnostic::error("array size must be a compile-time constant integer")
                            .with_span(*span),
                    )
                }
            }
            Type::Tuple(tys, span) => {
                let mut elems = Vec::new();
                for t in tys {
                    elems.push(self.resolve_type(t)?);
                }
                Ok(self.ctx.tuple(elems))
            }
            Type::Function { params, ret, span } => {
                let mut param_tys = Vec::new();
                for p in params {
                    param_tys.push(self.resolve_type(p)?);
                }
                let ret_ty = self.resolve_type(ret)?;
                Ok(self.ctx.function(param_tys, ret_ty))
            }
            Type::Projection(base, name, span) => Ok(self.ctx.error()),
            Type::DynTrait(traits, span) => {
                let mut trait_ids = Vec::new();
                for t in traits {
                    if let Type::Path(path, _) = t {
                        if let Some(def_id) = self.resolve_def_id(path).ok() {
                            trait_ids.push(def_id);
                        }
                    }
                }
                Ok(self.ctx.dyn_trait(trait_ids))
            }
            Type::Exists {
                name,
                base,
                invariant,
                span,
            } => {
                let base_ty = self.resolve_type(base)?;
                let (inv_hir, inv_ty) = self.synthesize(invariant)?;
                if !self.ctx.is_bool(inv_ty) {
                    self.diagnostics
                        .push(Diagnostic::error("invariant must be boolean").with_span(*span));
                }
                Ok(self.ctx.exists(name.clone(), base_ty, *invariant.clone()))
            }
            Type::Literal(expr, span) => {
                let (_, ty) = self.synthesize(expr)?;
                Ok(ty)
            }
            Type::Never(span) => Ok(self.ctx.never()),
            Type::Union(tys, span) => Ok(self.ctx.error()),
            Type::Error(span) => Ok(self.ctx.error()),
        }
    }

    fn expand_base_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(def_id) = self.ctx.get_def_id_for_type(ty) {
            let binding = self.symbols.lookup_type_by_def_id(def_id);
            if let Some(binding) = binding {
                if binding.kind == TypeKind::Alias {
                    if self.resolving_aliases.contains(&def_id) {
                        return Err(Diagnostic::error("circular alias definition").with_span(span));
                    }
                    self.resolving_aliases.insert(def_id);
                    let result = if let Some(ast) = &binding.alias_ast {
                        self.resolve_type(ast)
                    } else {
                        Err(Diagnostic::error("alias has no body").with_span(span))
                    };
                    self.resolving_aliases.remove(&def_id);
                    return result;
                }
            }
        }
        Ok(ty)
    }

    fn collect_type_args(&self, path: &[String], span: Span) -> Result<Vec<TypeId>, Diagnostic> {
        if path.len() > 1 {
            return Err(Diagnostic::error("nested paths not supported yet").with_span(span));
        }
        Ok(Vec::new())
    }

    fn resolve_def_id(&self, path: &[String]) -> Result<DefId, Diagnostic> {
        if path.is_empty() {
            return Err(Diagnostic::error("empty path").with_span(Span::new(0, 0)));
        }
        let name = &path[0];
        if let Some(binding) = self.symbols.lookup_type(name) {
            if path.len() == 1 {
                return Ok(binding.def_id);
            }
            Err(Diagnostic::error(format!(
                "nested paths not supported yet: {}",
                path.join("::")
            ))
            .with_span(Span::new(0, 0)))
        } else {
            Err(Diagnostic::error(format!("type '{}' not found", name)).with_span(Span::new(0, 0)))
        }
    }

    fn unify(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        match self.ctx.unify(expected, actual) {
            Ok(ty) => Ok(ty),
            Err(err) => Err(Diagnostic::error(format!(
                "type mismatch: expected {:?}, found {:?}",
                self.ctx.get(expected),
                self.ctx.get(actual)
            ))
            .with_span(span)),
        }
    }

    fn binary_op_type(
        &mut self,
        op: BinOp,
        left: TypeId,
        right: TypeId,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                self.unify(left, right, span)?;
                if self.ctx.is_numeric(left) {
                    Ok(left)
                } else {
                    Err(
                        Diagnostic::error("arithmetic operators require numeric types")
                            .with_span(span),
                    )
                }
            }
            BinOp::AddWrap
            | BinOp::SubWrap
            | BinOp::MulWrap
            | BinOp::AddSaturate
            | BinOp::SubSaturate
            | BinOp::MulSaturate
            | BinOp::AddTrap
            | BinOp::SubTrap
            | BinOp::MulTrap => {
                self.unify(left, right, span)?;
                if self.ctx.is_integer(left) {
                    Ok(left)
                } else {
                    Err(Diagnostic::error(
                        "wrapping/saturating/trapping operators require integer types",
                    )
                    .with_span(span))
                }
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                self.unify(left, right, span)?;
                if self.ctx.is_integer(left) {
                    Ok(left)
                } else {
                    Err(Diagnostic::error("bitwise operators require integer types")
                        .with_span(span))
                }
            }
            BinOp::Eq | BinOp::Neq => {
                self.unify(left, right, span)?;
                Ok(self.ctx.bool())
            }
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                self.unify(left, right, span)?;
                if self.ctx.is_numeric(left) {
                    Ok(self.ctx.bool())
                } else {
                    Err(
                        Diagnostic::error("comparison operators require numeric types")
                            .with_span(span),
                    )
                }
            }
            BinOp::And | BinOp::Or => {
                if !self.ctx.is_bool(left) {
                    Err(Diagnostic::error("logical operators require boolean types")
                        .with_span(span))
                } else {
                    self.unify(left, right, span)?;
                    Ok(self.ctx.bool())
                }
            }
        }
    }

    fn unary_op_type(&mut self, op: UnaryOp, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        match op {
            UnaryOp::Neg => {
                if self.ctx.is_numeric(ty) {
                    Ok(ty)
                } else {
                    Err(Diagnostic::error("negation requires numeric type").with_span(span))
                }
            }
            UnaryOp::Not => {
                if self.ctx.is_bool(ty) {
                    Ok(self.ctx.bool())
                } else {
                    Err(Diagnostic::error("logical not requires boolean type").with_span(span))
                }
            }
            UnaryOp::BitNot => {
                if self.ctx.is_integer(ty) {
                    Ok(ty)
                } else {
                    Err(Diagnostic::error("bitwise not requires integer type").with_span(span))
                }
            }
            UnaryOp::Deref => {
                if let Some(pointee) = self.ctx.pointee_of_ref(ty) {
                    Ok(pointee)
                } else if let Some(pointee) = self.ctx.pointee_of_pointer(ty) {
                    Ok(pointee)
                } else {
                    Err(
                        Diagnostic::error("dereference requires reference or pointer type")
                            .with_span(span),
                    )
                }
            }
            UnaryOp::Ref | UnaryOp::RefMut => {
                let mutable = matches!(op, UnaryOp::RefMut);
                Ok(self.ctx.reference(ty, mutable))
            }
        }
    }

    fn check_cast(
        &mut self,
        from: TypeId,
        to: TypeId,
        safe: bool,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        if safe {
            if self.ctx.is_numeric(from) && self.ctx.is_numeric(to) {
                Ok(to)
            } else if self.ctx.is_bool(from) && self.ctx.is_integer(to) {
                Ok(to)
            } else if self.ctx.is_integer(from) && self.ctx.is_bool(to) {
                Ok(to)
            } else {
                Err(
                    Diagnostic::error("safe cast only allowed between numeric and boolean types")
                        .with_span(span),
                )
            }
        } else {
            if self.ctx.is_numeric(from) && self.ctx.is_numeric(to) {
                Ok(to)
            } else if self.ctx.is_reference(from) && self.ctx.is_pointer(to) {
                Ok(to)
            } else if self.ctx.is_pointer(from) && self.ctx.is_reference(to) {
                Ok(to)
            } else if let TypeData::Ptr { .. } = self.ctx.get(from) {
                if let TypeData::Ptr { .. } = self.ctx.get(to) {
                    Ok(to)
                } else {
                    Err(
                        Diagnostic::error("bitcast requires compatible pointer/ref types")
                            .with_span(span),
                    )
                }
            } else {
                Err(Diagnostic::error("unsafe cast requires compatible types").with_span(span))
            }
        }
    }

    fn check_result_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(ok_ty) = self.extract_ok_type(ty) {
            Ok(ok_ty)
        } else {
            Err(Diagnostic::error("try operator requires Result type").with_span(span))
        }
    }

    fn check_future_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(future_ty) = self.extract_future_type(ty) {
            Ok(future_ty)
        } else {
            Err(Diagnostic::error("await operator requires Future type").with_span(span))
        }
    }

    fn extract_ok_type(&self, ty: TypeId) -> Option<TypeId> {
        if let TypeData::Enum { def_id: _, args } = self.ctx.get(ty) {
            if args.len() == 2 {
                return Some(args[0]);
            }
        }
        None
    }

    fn extract_future_type(&self, ty: TypeId) -> Option<TypeId> {
        if let TypeData::Enum { def_id: _, args } = self.ctx.get(ty) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
        None
    }

    fn extract_result_types(&self, ty: TypeId, span: Span) -> Result<(TypeId, TypeId), Diagnostic> {
        if let TypeData::Enum { def_id: _, args } = self.ctx.get(ty) {
            if args.len() == 2 {
                return Ok((args[0], args[1]));
            }
        }
        Err(Diagnostic::error("catch requires Result type").with_span(span))
    }

    fn lookup_field(&self, ty: TypeId, name: &str, span: Span) -> Result<TypeId, Diagnostic> {
        if let TypeData::Struct { def_id, args } = self.ctx.get(ty) {
            let binding = self
                .symbols
                .lookup_type_by_def_id(*def_id)
                .ok_or_else(|| Diagnostic::error("struct definition not found").with_span(span))?;
            let field = binding
                .fields
                .iter()
                .find(|f| f.name == name)
                .ok_or_else(|| {
                    Diagnostic::error(format!("field '{}' not found", name)).with_span(span)
                })?;
            let mut subst = Subst::new();
            for (i, param) in binding.params.iter().enumerate() {
                if let Some(&arg) = args.get(i) {
                    subst.insert(i, arg);
                }
            }
            let field_ty = self.ctx.subst(field.ty, &subst);
            Ok(field_ty)
        } else {
            Err(Diagnostic::error("field access on non-struct type").with_span(span))
        }
    }

    fn lookup_attr(&self, ty: TypeId, name: &str, span: Span) -> Result<TypeId, Diagnostic> {
        match name {
            "len" => {
                if self.ctx.is_array(ty) || self.ctx.is_slice(ty) {
                    Ok(self.ctx.usize())
                } else {
                    Err(
                        Diagnostic::error("'len attribute requires array or slice type")
                            .with_span(span),
                    )
                }
            }
            "size" => {
                if self.ctx.is_integer(ty) || self.ctx.is_float(ty) || self.ctx.is_pointer(ty) {
                    Ok(self.ctx.usize())
                } else {
                    Err(Diagnostic::error(
                        "'size attribute requires integer, float, or pointer type",
                    )
                    .with_span(span))
                }
            }
            "align" => Ok(self.ctx.usize()),
            "default" => Ok(ty),
            _ => Err(Diagnostic::error(format!("unknown attribute '{}'", name)).with_span(span)),
        }
    }

    fn block_type(&self, stmts: &[HirStmt]) -> TypeId {
        for stmt in stmts.iter().rev() {
            if let HirStmt::Expression(expr) = stmt {
                return expr.ty();
            }
            if let HirStmt::Return { value, .. } = stmt {
                if let Some(value) = value {
                    return value.ty();
                }
                return self.ctx.never();
            }
        }
        self.ctx.unit()
    }

    fn get_trait_id_for_binop(&self, op: BinOp, span: Span) -> Result<DefId, Diagnostic> {
        let trait_name = match op {
            BinOp::Add => "Add",
            BinOp::Sub => "Sub",
            BinOp::Mul => "Mul",
            BinOp::Div => "Div",
            BinOp::Rem => "Rem",
            BinOp::BitAnd => "BitAnd",
            BinOp::BitOr => "BitOr",
            BinOp::BitXor => "BitXor",
            BinOp::Shl => "Shl",
            BinOp::Shr => "Shr",
            BinOp::Eq => "Eq",
            BinOp::Neq => "Neq",
            BinOp::Lt => "Lt",
            BinOp::Gt => "Gt",
            BinOp::Le => "Le",
            BinOp::Ge => "Ge",
            BinOp::And => "And",
            BinOp::Or => "Or",
            _ => {
                return Err(
                    Diagnostic::error("overflow operators not yet supported via traits")
                        .with_span(span),
                );
            }
        };
        self.symbols
            .lookup_trait(trait_name)
            .map(|b| b.def_id)
            .ok_or_else(|| {
                Diagnostic::error(format!("trait `{}` not found", trait_name)).with_span(span)
            })
    }

    fn literal_type(&mut self, lit: &Literal) -> TypeId {
        match lit {
            Literal::Int(_) => self.ctx.int(32, true),
            Literal::Float(_) => self.ctx.float(64),
            Literal::Char(_) => self.ctx.char(),
            Literal::String(_) => self.ctx.slice(self.ctx.byte()),
            Literal::ByteString(_) => self.ctx.slice(self.ctx.byte()),
            Literal::Bool(_) => self.ctx.bool(),
        }
    }

    fn extract_int_from_type(&self, ty: &Type) -> Option<u8> {
        match ty {
            Type::Literal(expr, _) => {
                if let Expr::Literal(Literal::Int(val), _) = expr.as_ref() {
                    Some(*val as u8)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn new_infer_var(&mut self, kind: TypeVariableKind) -> TypeId {
        self.infer.new_type_var(self.ctx, kind)
    }

    fn add_constraint(&mut self, c: Constraint) {
        self.infer.add_constraint(c);
    }
}
