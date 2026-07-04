use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCollector};
use crate::hir::hir::*;
use crate::hir::infer::*;
use crate::hir::resolver::ResolutionMap;
use crate::hir::symbol::*;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::mem;

pub mod autoderef;
pub mod context;
pub mod contract;
pub mod fn_ctxt;
pub mod helpers;
pub mod region;
pub mod types;
use self::autoderef::*;
use self::helpers::*;
use self::types::*;
pub use context::*;
pub use contract::*;
pub use fn_ctxt::*;
pub use region::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtxKind {
    /// A normaw function body or top-wevel bwock (•́ω•̀)
    Function,
    /// A cwosuwe body (can't bweak/continue fwom outside) (/ω＼)
    Closure,
    /// An async bwock (wike a cwosuwe) ☆ﾟ.*･｡ﾟ
    AsyncBlock,
    /// A widdle woop (wike `loop { ... }`)
    Loop,
    /// A `whiwe` woop
    While,
    /// A `fow` woop
    For,
    /// A wabewed bwock (can be bweaked via `bweak 'wabew`) (｀・ω・´)
    LabeledBlock,
    /// A comptime evawuation bwock — `wetuwn` inside is comptime contwow fwow, not an ewwow. (◕‿◕)
    Comptime,
}

/// A fwame howding the context kind and its span (*/ω＼*)
#[derive(Debug, Clone)]
pub struct CtxFrame {
    pub kind: CtxKind,
    span: Span,
    /// Optionaw wabew name (onwy used by WabewedBwock)
    label: Option<String>,
}
pub struct TypeChecker<'a> {
    ctx: &'a mut TypeContext,
    symbols: &'a SymbolTable,
    trait_env: &'a mut TraitEnv,
    diagnostics: DiagnosticCollector,
    current_function: Option<DefId>,
    current_return_type: Option<TypeId>,
    resolving_aliases: HashSet<DefId>,
    infer: InferenceContext,
    infer_stack: Vec<InferenceContext>,
    /// Wegion twee: twacks cuwwent function, woop, cwosuwe, etc.
    /// Wepwaces the owd wineaw `woop_stack` with a twee stwuctuwe
    /// suppowting pawtiaw genewawization (OmniML §3.2). (｀・ω・´)
    region_tree: RegionTree,
    /// Locaw cache of variabwe types, updated by check_stmt for each VawiabweDef.
    /// Ovewwides the wesowvew's pwacehowdew `ewrow` type. (◕‿◕)
    local_variable_types: HashMap<String, TypeId>,
    /// Pre-resolved by NameResolver: variable name → TypeId
    resolution_map: ResolutionMap,
    /// Local cache of generic type parameter types (e.g. `T` in `def foo<T>(x: T)`).
    /// Populated when processing function definitions with type_params.
    local_type_param_cache: HashMap<String, TypeId>,
    /// SCAP-style guarantee chain: tracks outstanding postconditions that must
    /// be discharged on function return (Feng & Shao 2006 §4).
    guarantee_chain: GuaranteeChain,
}

/// Error type for comptime control flow within comptime blocks.
/// These are not real errors — they are control-flow signals that propagate
/// out of a comptime evaluation context (like `return` inside `comptime { }`).
#[derive(Debug, Clone)]
pub enum ComptimeControlFlow {
    Return(Option<HirExpr>),
    Break(Option<String>),
    Continue(Option<String>),
}

impl std::fmt::Display for ComptimeControlFlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComptimeControlFlow::Return(_) => write!(f, "comptime return"),
            ComptimeControlFlow::Break(_) => write!(f, "comptime break"),
            ComptimeControlFlow::Continue(_) => write!(f, "comptime continue"),
        }
    }
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        symbols: &'a SymbolTable,
        trait_env: &'a mut TraitEnv,
        resolution_map: ResolutionMap,
    ) -> Self {
        let mut checker = TypeChecker {
            ctx,
            symbols,
            trait_env,
            diagnostics: DiagnosticCollector::new(),
            current_function: None,
            current_return_type: None,
            resolving_aliases: HashSet::new(),
            infer: InferenceContext::new(),
            infer_stack: Vec::new(),
            region_tree: RegionTree::new(),
            local_variable_types: HashMap::new(),
            local_type_param_cache: HashMap::new(),
            resolution_map,
            guarantee_chain: GuaranteeChain::new(),
        };
        // Pre-populate from the name resolver's results
        for (name, ty) in &checker.resolution_map.variable_types {
            checker.local_variable_types.insert(name.clone(), *ty);
        }
        checker
    }

    /// Find the innermost bweak tawget (Woop, Whiwe, Fow, WabewedBwock) (*＾▽＾)／
    /// Wetuwns the tawget's span and optionaw wabew. If `wabew` is Some, onwy match same-named WabewedBwock.
    /// Find the innermost continue tawget (onwy Woop, Whiwe, Fow) ☆ﾟ.*･｡ﾟ
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
            Err(mem::take(&mut self.diagnostics))
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
                ..
            } => {
                // 'set' does not support pattern destructuring
                if *kind == VariableKind::Set && pattern.is_some() {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "`set` does not support pattern destructuring; use `let` instead",
                        )
                        .with_code("E001")
                        .with_span(*span),
                    );
                }

                // 'let' must have an explicit initializer
                if *kind == VariableKind::Let && value.is_none() {
                    self.diagnostics.push(
                        Diagnostic::error("`let` requires an explicit initializer; it cannot rely on a type's default value")
                            .with_code("E002")
                            .with_span(*span),
                    );
                }

                // Resolve the declared type, or leave as an inference variable if not provided.
                let declared_ty = if let Some(ty) = ty {
                    self.resolve_type(ty)?
                } else {
                    self.new_infer_var(TypeVariableKind::Unconstrained)
                };

                // Determine the actual initializer (value) and its type.
                let (value_hir, inferred_ty) = if let Some(value) = value {
                    // Explicit initializer present
                    if ty.is_some() {
                        let hir = self.check_expr(
                            value,
                            Expectation::HasType(declared_ty),
                            TypingContext::None,
                        )?;
                        let ty = hir.ty();
                        (Some(hir), ty)
                    } else {
                        let (hir, ty) = self.infer_expr(value)?;
                        (Some(hir), ty)
                    }
                } else {
                    // No explicit initializer: try type's default value
                    let default_expr = self.lookup_type_default_expr(declared_ty, *span)?;
                    if let Some(default_expr) = default_expr {
                        let hir = self.check_expr(
                            &default_expr,
                            Expectation::HasType(declared_ty),
                            TypingContext::None,
                        )?;
                        let ty = hir.ty();
                        (Some(hir), ty)
                    } else {
                        // Neither default nor initializer – error
                        self.diagnostics.push(
                            Diagnostic::error(
                                "type has no default value and no initializer provided",
                            )
                            .with_code("E003")
                            .with_span(*span),
                        );
                        (None, declared_ty)
                    }
                };

                // Unify declared type with inferred type (if we have both)
                if let Some(ref value_hir) = value_hir {
                    self.unify_with(declared_ty, inferred_ty, *span, TypingContext::None)?;
                }

                let pattern_hir = if let Some(pattern) = pattern {
                    Some(self.check_pattern(pattern, declared_ty)?)
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

                let final_ty = if declared_ty != self.ctx.error() {
                    declared_ty
                } else if let Some(hir) = &value_hir {
                    hir.ty()
                } else {
                    self.ctx.error()
                };

                // Cache the variable's type for subsequent references
                if let Some(var_name) = name {
                    self.local_variable_types.insert(var_name.clone(), final_ty);
                }

                Ok(HirStmt::VariableDef {
                    kind: *kind,
                    mutable: *mutable,
                    name: name.clone(),
                    pattern: pattern_hir,
                    ty: final_ty,
                    value: value_hir.map(Box::new),
                    else_branch: else_hir,
                    span: *span,
                })
            }
            Stmt::FunctionDef {
                span,
                attributes,
                contracts,
                name,
                params,
                return_type,
                body,
                type_params,
                where_clause,
                finally,
                is_comptime,
                is_async,
                ..
            } => {
                // Register generic type parameters FIRST so that `T` in parameter types,
                // return types, and where clauses can be resolved.
                for (i, tp) in type_params.iter().enumerate() {
                    let generic_id = self.ctx.generic_param(i, tp.name.clone());
                    self.local_type_param_cache
                        .insert(tp.name.clone(), generic_id);
                }

                let return_ty = self.resolve_type(return_type)?;
                let mut hir_params = Vec::new();
                for param in params {
                    let param_ty = if let Some(ty) = &param.ty {
                        self.resolve_type(ty)?
                    } else {
                        self.ctx.error()
                    };
                    self.require_type_sized(param_ty, param.span);
                    hir_params.push(HirParam {
                        name: param.name.clone(),
                        ty: param_ty,
                        default: param.default.clone(),
                        span: param.span,
                    });
                }

                let guard = ScopeGuard::new(self);
                guard.checker.current_function = Some(DefId(0));
                guard.checker.current_return_type = Some(return_ty);
                guard.checker.enter_inference_scope();
                guard.checker.push_ctx(CtxKind::Function, *span, None);

                // Pre-populate the local variable cache with function parameters
                // and `result` so that ensures clauses can reference them.
                for p in &hir_params {
                    guard
                        .checker
                        .local_variable_types
                        .insert(p.name.clone(), p.ty);
                }
                guard
                    .checker
                    .local_variable_types
                    .insert("result".to_string(), return_ty);

                // SCAP: collect ensures conditions into the guarantee chain.
                // Each `ensures` becomes a postcondition that must hold at return.
                for contract in contracts {
                    if let Contract::Ensures { expr, .. } = contract {
                        let (_, ensures_ty) = guard
                            .checker
                            .infer_expr(expr)
                            .unwrap_or_else(|_| (HirExpr::Error(*span), guard.checker.ctx.bool()));
                        let g = Guarantee::new(None, Some(ensures_ty), None);
                        guard.checker.guarantee_chain.push(g);
                    }
                }
                // Generate where-clause constraints as Impl(clause_ty, trait_id)
                // so the solver can verify trait bounds on generic parameters.
                if let Some(wc) = where_clause {
                    for pred in &wc.predicates {
                        let pred_ty = guard.checker.resolve_type(&pred.ty)?;
                        for bound in &pred.bounds {
                            if let Some(trait_id) = guard.checker.resolve_trait_path(bound) {
                                guard
                                    .checker
                                    .add_constraint(Constraint::Impl(pred_ty, trait_id, pred.span));
                            }
                        }
                    }
                }

                let body_result = if let Some(body) = body {
                    let mut stmts = Vec::new();
                    for s in body {
                        stmts.push(guard.checker.check_stmt(s)?);
                    }
                    Ok(Some(stmts))
                } else {
                    Ok(None)
                };

                guard.checker.pop_ctx();

                let body_hir = match body_result {
                    Ok(body) => body,
                    Err(e) => return Err(e),
                };

                let exit_res = guard.checker.exit_inference_scope();
                guard.defuse();

                if let Err(_err) = exit_res {
                    return Err(Diagnostic::error("inference failure").with_span(*span));
                }

                if let Some(ref body_stmts) = body_hir {
                    let body_ty = self.block_type(body_stmts);
                    self.unify_with(return_ty, body_ty, *span, TypingContext::ReturnValue)?;
                }

                // Contract verification skeleton: check that requires/ensures are bool,
                // and decreases/terminates are integer types.
                for contract in contracts {
                    match contract {
                        Contract::Requires(expr, cspan) | Contract::Invariant(expr, cspan) => {
                            let (_, ty) = self.infer_expr(expr)?;
                            if !self.ctx.is_bool(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error("contract condition must be boolean")
                                        .with_code("E020")
                                        .with_span(*cspan),
                                );
                            }
                        }
                        Contract::Ensures {
                            expr, span: cspan, ..
                        } => {
                            let (_, ty) = self.infer_expr(expr)?;
                            if !self.ctx.is_bool(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error("ensures clause must be boolean")
                                        .with_code("E020")
                                        .with_span(*cspan),
                                );
                            }
                        }
                        Contract::Decreases(expr, cspan) | Contract::Terminates(expr, cspan) => {
                            let (_, ty) = self.infer_expr(expr)?;
                            if !self.ctx.is_numeric(ty) && !self.ctx.is_integer(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "decreases/terminates expression must be an integer",
                                    )
                                    .with_code("E021")
                                    .with_span(*cspan),
                                );
                            }
                        }
                    }
                }

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
                    doc: None,
                    name: name.clone(),
                    params: hir_params,
                    return_type: return_ty,
                    body: body_hir,
                    type_params: type_params.clone(),
                    where_clause: where_clause.clone().map(|_| ()),
                    finally: finally_hir,
                    is_comptime: *is_comptime,
                    is_async: *is_async,
                    is_ieee_contracts: attributes.iter().any(|a| a.name == "ieee_contracts"),
                    hints: attributes
                        .iter()
                        .filter(|a| a.name == "hint")
                        .flat_map(|a| a.args.clone())
                        .collect(),
                })
            }
            Stmt::Expression(expr) => {
                let (hir, _) = self.infer_expr(expr)?;
                Ok(HirStmt::Expression(Box::new(hir)))
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                let (cond_hir, cond_ty) = self.infer_expr(cond)?;
                let cond_is_bool = self.ctx.is_bool(cond_ty)
                    || matches!(self.ctx.get(cond_ty), TypeData::InferVar { id }
                        if self.infer.get_var_kind(*id) == Some(TypeVariableKind::Bool));
                if !cond_is_bool {
                    self.diagnostics.push(
                        Diagnostic::error("if condition must be boolean")
                            .with_code("E004")
                            .with_span(*span)
                            .with_label(cond.span(), format!("got {:?}", self.ctx.get(cond_ty))),
                    );
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
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee)?;
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
                let (cond_hir, cond_ty) = self.infer_expr(cond)?;
                let cond_is_bool = self.ctx.is_bool(cond_ty)
                    || matches!(self.ctx.get(cond_ty), TypeData::InferVar { id }
                        if self.infer.get_var_kind(*id) == Some(TypeVariableKind::Bool));
                if !cond_is_bool {
                    self.diagnostics.push(
                        Diagnostic::error("while condition must be boolean")
                            .with_span(*span)
                            .with_label(cond.span(), format!("got {:?}", self.ctx.get(cond_ty))),
                    );
                }
                let inv_hir = invariant
                    .as_ref()
                    .map(|inv| self.infer_expr(inv).map(|(h, _)| h))
                    .transpose()?;
                let dec_hir = decreases
                    .as_ref()
                    .map(|dec| self.infer_expr(dec).map(|(h, _)| h))
                    .transpose()?;
                self.push_ctx(CtxKind::While, *span, None);
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
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
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee)?;
                let pattern_hir = self.check_pattern(pattern, scrut_ty)?;
                let inv_hir = invariant
                    .as_ref()
                    .map(|inv| self.infer_expr(inv).map(|(h, _)| h))
                    .transpose()?;
                let dec_hir = decreases
                    .as_ref()
                    .map(|dec| self.infer_expr(dec).map(|(h, _)| h))
                    .transpose()?;
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
                let (iter_hir, iter_ty) = self.infer_expr(iterable)?;
                let elem_ty = self
                    .ctx
                    .elem_of_slice(iter_ty)
                    .or_else(|| self.ctx.elem_of_array(iter_ty))
                    .unwrap_or_else(|| {
                        self.diagnostics.push(
                            Diagnostic::error("for loop iterable must be an array or slice")
                                .with_span(*span),
                        );
                        self.ctx.error()
                    });
                let pattern_hir = self.check_pattern(pattern, elem_ty)?;
                let inv_hir = invariant
                    .as_ref()
                    .map(|inv| self.infer_expr(inv).map(|(h, _)| h))
                    .transpose()?;
                let dec_hir = decreases
                    .as_ref()
                    .map(|dec| self.infer_expr(dec).map(|(h, _)| h))
                    .transpose()?;
                self.push_ctx(CtxKind::For, *span, None);
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
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
                self.push_ctx(CtxKind::Loop, *span, None);
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
                Ok(HirStmt::Loop {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::Leave { label, span } => {
                let target = self.find_break_target(label.as_deref());
                match target {
                    None => {
                        // Check if we're inside a cwosuwe (>_<)
                        let enclosing_closure =
                            self.region_tree
                                .iter_frames_rev()
                                .find_map(|f| match f.kind {
                                    CtxKind::Closure | CtxKind::AsyncBlock => Some(f.span),
                                    _ => None,
                                });
                        if enclosing_closure.is_some() {
                            self.diagnostics.push(
                                Diagnostic::error("cannot `leave` out of a closure or async block")
                                    .with_code("E005")
                                    .with_span(*span),
                            );
                        } else if label.is_some() {
                            self.diagnostics.push(
                                Diagnostic::error(format!("cannot `leave` with label `{}` – no matching labeled block or loop found", label.as_ref().unwrap()))
                                    .with_code("E005")
                                    .with_span(*span)
                            );
                        } else {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "`leave` statement outside of loop; use `return` instead",
                                )
                                .with_code("E005")
                                .with_span(*span)
                                .with_suggestion("use `return` to exit the current function"),
                            );
                        }
                        Ok(HirStmt::Leave {
                            label: label.clone(),
                            span: *span,
                        })
                    }
                    Some(_) => Ok(HirStmt::Leave {
                        label: label.clone(),
                        span: *span,
                    }),
                }
            }
            Stmt::Continue { label, span } => {
                let target = self.find_continue_target(label.as_deref());
                match target {
                    None => {
                        let enclosing_closure =
                            self.region_tree
                                .iter_frames_rev()
                                .find_map(|f| match f.kind {
                                    CtxKind::Closure | CtxKind::AsyncBlock => Some(f.span),
                                    _ => None,
                                });
                        if enclosing_closure.is_some() {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "cannot `continue` out of a closure or async block",
                                )
                                .with_code("E006")
                                .with_span(*span),
                            );
                        } else {
                            self.diagnostics.push(
                                Diagnostic::error("`continue` statement outside of loop")
                                    .with_code("E006")
                                    .with_span(*span)
                                    .with_suggestion("use `leave` or `return` instead"),
                            );
                        }
                        Ok(HirStmt::Continue {
                            label: label.clone(),
                            span: *span,
                        })
                    }
                    Some(_) => Ok(HirStmt::Continue {
                        label: label.clone(),
                        span: *span,
                    }),
                }
            }
            Stmt::Return { value, span } => {
                // Check if we're inside a comptime block — if so, return is comptime
                // control flow, not a real function return.
                let in_comptime = self
                    .region_tree
                    .iter_frames_rev()
                    .any(|f| matches!(f.kind, CtxKind::Comptime));
                if in_comptime {
                    // Inside comptime, `return` acts as comptime control flow:
                    // the value is evaluated and propagated out of the comptime block.
                    if let Some(value) = value {
                        let (hir, _) = self.infer_expr(value)?;
                        return Err(Diagnostic::error(format!(
                            "comptime return with value: {:?}",
                            hir
                        )));
                    }
                    return Err(Diagnostic::error("comptime return".to_string()));
                }

                // SCAP: discharging the innermost guarantee on return.
                // If there's an ensures clause, it acts as the postcondition
                // and must be satisfied at this return point.
                if let Some(g) = self.guarantee_chain.current() {
                    if let Some(post) = g.post {
                        // The postcondition type must be bool and should hold
                        // at the return point.  We verify this by type-checking
                        // unify(post, bool) as a basic consistency check.
                        if !self.ctx.is_bool(post) {
                            self.diagnostics.push(
                                Diagnostic::error("ensures condition must be boolean at return")
                                    .with_code("E022")
                                    .with_span(*span),
                            );
                        }
                    }
                }

                // Check that return is inside a function or closure context
                let in_function = self
                    .region_tree
                    .iter_frames_rev()
                    .any(|f| matches!(f.kind, CtxKind::Function | CtxKind::Closure));
                if !in_function {
                    self.diagnostics.push(
                        Diagnostic::error("`return` statement outside of function")
                            .with_code("E007")
                            .with_span(*span),
                    );
                }
                // Ban `return Err(...)` — use `leave with` instead
                if let Some(Expr::EnumLit { path, variant, .. }) = value {
                    if variant == "Err" && path.len() == 1 && path[0] == "Result" {
                        self.diagnostics.push(
                            Diagnostic::error("`return Err(...)` is not valid; use `leave with` instead")
                                .with_code("E008")
                                .with_span(*span)
                                .with_suggestion("write `leave with error_value;` instead of `return Err(error_value);`")
                        );
                    }
                }
                if let Some(value) = value {
                    if let Some(ret_ty) = self.current_return_type {
                        let hir = self.check_expr(
                            value,
                            Expectation::HasType(ret_ty),
                            TypingContext::ReturnValue,
                        )?;
                        Ok(HirStmt::Return {
                            value: Some(Box::new(hir)),
                            span: *span,
                        })
                    } else {
                        let (hir, _) = self.infer_expr(value)?;
                        Ok(HirStmt::Return {
                            value: Some(Box::new(hir)),
                            span: *span,
                        })
                    }
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
                // Validate that the target is a valid lvalue
                if !is_valid_lvalue(target) {
                    self.diagnostics.push(
                        Diagnostic::error("invalid left-hand side for assignment; expected variable, field access, or index")
                            .with_span(*span)
                    );
                }
                let (target_hir, target_ty) = self.infer_expr(target)?;
                let value_hir = if let Some(op) = op {
                    let result_ty = self.binary_op_type(*op, target_ty, target_ty, *span)?;
                    self.unify_with(target_ty, result_ty, *span, TypingContext::None)?;
                    self.check_expr(value, Expectation::HasType(target_ty), TypingContext::None)?
                } else {
                    self.check_expr(value, Expectation::HasType(target_ty), TypingContext::None)?
                };
                Ok(HirStmt::Assign {
                    target: Box::new(target_hir),
                    op: *op,
                    value: Box::new(value_hir),
                    span: *span,
                })
            }
            Stmt::ComptimeBlock { body, span } => {
                // Push a comptime context frame so that `return` inside comptime
                // blocks is treated as comptime control flow, not an error.
                self.push_ctx(CtxKind::Comptime, *span, None);
                let body_hir = match self.check_block(body) {
                    Ok(hir) => {
                        self.pop_ctx();
                        Ok(HirStmt::ComptimeBlock {
                            body: hir,
                            span: *span,
                        })
                    }
                    Err(diag) => {
                        self.pop_ctx();
                        Err(diag)
                    }
                };
                body_hir
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
            Stmt::TypeDef { .. } => {
                // Type definitions are already handled by the resolver;
                // no additional checking needed here.
                Ok(HirStmt::Error)
            }
            Stmt::Edition(..) => {
                // Edition declarations are handled by the parser; skip silently.
                Ok(HirStmt::Error)
            }
            Stmt::TraitDef { .. } => {
                // Trait definitions are handled by the resolver; skip silently.
                Ok(HirStmt::Error)
            }
            Stmt::Import { .. } | Stmt::ExternFunction { .. } | Stmt::Constraint { .. } => {
                self.diagnostics.push(
                    Diagnostic::error("top-level item not yet supported in type checker")
                        .with_span(stmt.span()),
                );
                Ok(HirStmt::Error)
            }
            Stmt::ImplBlock { .. } => {
                let (trait_path, for_type, methods, span, attributes, type_params) = match stmt {
                    Stmt::ImplBlock {
                        span,
                        trait_path,
                        for_type,
                        methods,
                        attributes,
                        type_params,
                        ..
                    } => (
                        trait_path,
                        for_type,
                        methods,
                        *span,
                        attributes,
                        type_params,
                    ),
                    _ => panic!("check_stmt: expected ImplBlock, got {:?}", stmt),
                };
                if let Some(tp) = &trait_path {
                    // ── Trait impl block ─────────────────────────────────
                    let trait_id = match self.resolve_def_id(tp) {
                        Ok(id) => id,
                        Err(diag) => {
                            self.diagnostics.push(diag);
                            return Ok(HirStmt::Error);
                        }
                    };
                    let trait_binding = match self.symbols.lookup_trait_by_def_id(trait_id) {
                        Some(b) => b,
                        None => {
                            self.diagnostics.push(
                                Diagnostic::error("trait not found")
                                    .with_code("E100")
                                    .with_span(span),
                            );
                            return Ok(HirStmt::Error);
                        }
                    };

                    // Register generic type parameters so `T` in `impl<T> Foo for T` resolves
                    for (i, tp) in type_params.iter().enumerate() {
                        let generic_id = self.ctx.generic_param(i, tp.name.clone());
                        self.local_type_param_cache
                            .insert(tp.name.clone(), generic_id);
                    }

                    // Resolve the for_type
                    let for_ty = self.resolve_type(for_type)?;

                    // Check that all required trait methods are provided
                    let auto_deref = attributes.iter().any(|a| a.name == "auto_deref");
                    let impl_method_names: HashSet<String> =
                        methods.iter().map(|m| m.name.clone()).collect();
                    let self_ty = &for_type;

                    for (tm_name, _tm_sig) in &trait_binding.methods {
                        if !impl_method_names.contains(tm_name) {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "impl missing method `{}` required by trait `{}`",
                                    tm_name,
                                    tp.join("::"),
                                ))
                                .with_code("E101")
                                .with_help("every trait method must be implemented — add a `def` for it in this impl block")
                                .with_span(span));
                        }
                    }

                    // Ensure all required associated types are provided (or have defaults)
                    for (at_name, at_default) in &trait_binding.associated_types {
                        if at_default.is_none() {
                            // No default — the impl must provide this associated type.
                            // This check is deferred until impl-block associated types are parsed.
                        }
                    }

                    // Resolve method param/return types and register the impl
                    let mut method_infos = Vec::new();
                    for m in methods {
                        let param_tys = m
                            .params
                            .iter()
                            .map(|p| {
                                if let Some(ty) = &p.ty {
                                    let resolved = self.resolve_self_ty(ty, self_ty);
                                    self.resolve_type(&resolved)
                                } else {
                                    Ok(self.ctx.error())
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let ret_ty = {
                            let resolved = self.resolve_self_ty(&m.return_type, self_ty);
                            self.resolve_type(&resolved)?
                        };

                        // Signature compatibility: compare against trait declaration
                        if let Some((_, trait_sig)) =
                            trait_binding.methods.iter().find(|(n, _)| n == &m.name)
                        {
                            if m.params.len() != trait_sig.params.len() {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "impl method `{}` has {} parameters but trait expects {}",
                                        m.name,
                                        m.params.len(),
                                        trait_sig.params.len(),
                                    ))
                                    .with_code("E103")
                                    .with_span(m.span),
                                );
                            }
                        }

                        method_infos.push(crate::hir::traits::MethodInfo {
                            name: m.name.clone(),
                            param_tys,
                            ret_ty,
                            span: m.span,
                            has_auto_deref: auto_deref,
                        });
                    }

                    let candidate = crate::hir::traits::ImplCandidate {
                        trait_id,
                        for_type: for_ty,
                        methods: methods.clone(),
                        assoc_tys: Vec::new(),
                        span,
                        has_auto_deref: auto_deref,
                        context: {
                            // Populate context from where clause and type param bounds,
                            // for Paterson/Coverage condition checking.
                            let mut ctx_tys = Vec::new();
                            for (i, tp) in type_params.iter().enumerate() {
                                if !tp.bounds.is_empty() {
                                    let param_id = self.ctx.generic_param(i, tp.name.clone());
                                    ctx_tys.push(param_id);
                                }
                            }
                            ctx_tys
                        },
                    };

                    if let Err(orphan) =
                        self.trait_env
                            .add_impl(candidate, self.symbols, self.ctx, false)
                    {
                        self.diagnostics.push(
                            Diagnostic::error(format!("{}", orphan))
                                .with_code("E102")
                                .with_span(span),
                        );
                    }

                    // Also register the resolved methods for method resolution
                    if let TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } =
                        self.ctx.get(for_ty)
                    {
                        self.trait_env.add_inherent_methods(*def_id, method_infos);
                    }

                    Ok(HirStmt::ImplBlock {
                        span,
                        attributes: attributes.clone(),
                        trait_path: trait_path.clone(),
                        for_type: for_ty,
                        methods: methods.clone(),
                        associated_types: Vec::new(),
                    })
                } else {
                    // Inherent impl block: resolve the type and register methods
                    let for_ty = self.resolve_type(for_type)?;
                    let for_def_id = match self.ctx.get(for_ty) {
                        TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => *def_id,
                        _ => {
                            self.diagnostics.push(
                                Diagnostic::error("inherent impl on non-struct/enum type")
                                    .with_span(span),
                            );
                            return Ok(HirStmt::Error);
                        }
                    };
                    // Resolve method param/return types, replacing `Self` with for_type
                    let self_ty = &for_type; // The original AST type for Self
                    let auto_deref = attributes.iter().any(|a| a.name == "auto_deref");
                    let mut method_infos = Vec::new();
                    for m in methods {
                        let param_tys = m
                            .params
                            .iter()
                            .map(|p| {
                                if let Some(ty) = &p.ty {
                                    let resolved = self.resolve_self_ty(ty, self_ty);
                                    self.resolve_type(&resolved)
                                } else {
                                    Ok(self.ctx.error())
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let ret_ty = {
                            let resolved = self.resolve_self_ty(&m.return_type, self_ty);
                            self.resolve_type(&resolved)?
                        };
                        method_infos.push(crate::hir::traits::MethodInfo {
                            name: m.name.clone(),
                            param_tys,
                            ret_ty,
                            span: m.span,
                            has_auto_deref: auto_deref,
                        });
                    }
                    self.trait_env
                        .add_inherent_methods(for_def_id, method_infos);
                    Ok(HirStmt::ImplBlock {
                        span,
                        attributes: attributes.clone(),
                        trait_path: trait_path.clone(),
                        for_type: for_ty,
                        methods: methods.clone(),
                        associated_types: Vec::new(),
                    })
                }
            }
            Stmt::Error(span) => Err(Diagnostic::error("invalid statement").with_span(*span)),
        }
    }

    fn check_block(&mut self, stmts: &[Stmt]) -> Result<Vec<HirStmt>, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_block(stmts)
    }


    fn infer_expr(&mut self, expr: &Expr) -> Result<(HirExpr, TypeId), Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.infer_expr(expr)
    }

    fn check_expr(
        &mut self,
        expr: &Expr,
        expected: Expectation,
        ctx: TypingContext,
    ) -> Result<HirExpr, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_expr(expr, expected, ctx)
    }
    fn check_pattern(
        &mut self,
        pattern: &Pattern,
        expected_ty: TypeId,
    ) -> Result<HirPattern, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_pattern(pattern, expected_ty)
    }

    fn resolve_type(&mut self, ty: &Type) -> Result<TypeId, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.resolve_type(ty)
    }

    /// Recursively replace `Self` / `self` occurrences in a type with the
    /// concrete `self_ty` (the type being implemented for).
    fn resolve_self_ty(&self, ty: &Type, self_ty: &Type) -> Type {
        match ty {
            Type::Path(p, s) if p.len() == 1 && (p[0] == "Self" || p[0] == "self") => {
                self_ty.clone()
            }
            Type::Reference { inner, mutable, span: s, .. } => {
                Type::Reference {
                    inner: Box::new(self.resolve_self_ty(inner, self_ty)),
                    mutable: *mutable,
                    lifetime: None,
                    span: *s,
                }
            }
            Type::Pointer(inner, s) => {
                Type::Pointer(Box::new(self.resolve_self_ty(inner, self_ty)), *s)
            }
            Type::Generic(base, args, span) => {
                let new_base = self.resolve_self_ty(base, self_ty);
                let new_args: Vec<GenericArg> = args.iter().map(|a| {
                    match a {
                        GenericArg::Positional(t) => GenericArg::Positional(self.resolve_self_ty(t, self_ty)),
                        GenericArg::Named(n, t) => GenericArg::Named(n.clone(), self.resolve_self_ty(t, self_ty)),
                    }
                }).collect();
                Type::Generic(Box::new(new_base), new_args, *span)
            }
            Type::Tuple(tys, span) => {
                Type::Tuple(tys.iter().map(|t| self.resolve_self_ty(t, self_ty)).collect(), *span)
            }
            Type::Slice(inner, span) => {
                Type::Slice(Box::new(self.resolve_self_ty(inner, self_ty)), *span)
            }
            Type::Array(inner, size, span) => {
                Type::Array(Box::new(self.resolve_self_ty(inner, self_ty)), size.clone(), *span)
            }
            Type::DynTrait(traits, span) => {
                Type::DynTrait(traits.iter().map(|t| self.resolve_self_ty(t, self_ty)).collect(), *span)
            }
            Type::Function { params, ret, span } => {
                Type::Function {
                    params: params.iter().map(|p| self.resolve_self_ty(p, self_ty)).collect(),
                    ret: Box::new(self.resolve_self_ty(ret, self_ty)),
                    span: *span,
                }
            }
            Type::Projection { impl_type, trait_path, assoc_name, span } => {
                Type::Projection {
                    impl_type: Box::new(self.resolve_self_ty(impl_type, self_ty)),
                    trait_path: Box::new(self.resolve_self_ty(trait_path, self_ty)),
                    assoc_name: assoc_name.clone(),
                    span: *span,
                }
            }
            other => other.clone(),
        }
    }

    fn expand_base_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.expand_base_type(ty, span)
    }

    fn resolve_type_to_struct_or_enum(
        &self,
        ty: TypeId,
        span: Span,
    ) -> Result<(DefId, Vec<TypeId>), Diagnostic> {
        let resolved = self.ctx.resolve_binding(ty);
        match self.ctx.get(resolved) {
            TypeData::Struct { def_id, args } | TypeData::Enum { def_id, args } => {
                Ok((*def_id, args.clone()))
            }
            TypeData::Error => Err(Diagnostic::error("type error").with_span(span)),
            _ => Err(Diagnostic::error("expected struct or enum type").with_span(span)),
        }
    }

    fn resolve_def_id(&self, path: &[String]) -> Result<DefId, Diagnostic> {
        if path.is_empty() {
            return Err(Diagnostic::error("empty path").with_span(Span::new(0, 0)));
        }
        // Check the resolution map first (populated by NameResolver)
        if path.len() == 1 {
            if let Some(&def_id) = self.resolution_map.type_def_ids.get(&path[0]) {
                return Ok(def_id);
            }
        }
        // Check if this is a generic type parameter (e.g. `T` in `def foo<T>(x: T)`)
        if path.len() == 1 {
            if self.local_type_param_cache.contains_key(&path[0]) {
                // Return a sentinel DefId to signal "this is a generic param, not a concrete type"
                // The caller (resolve_type) will handle this by looking up local_type_param_cache.
                return Ok(DefId(usize::MAX - 1));
            }
        }
        self.symbols
            .lookup_type(&path[0])
            .map(|b| b.def_id)
            .or_else(|| self.symbols.lookup_trait(&path[0]).map(|b| b.def_id))
            .ok_or_else(|| {
                Diagnostic::error(format!("'{}' not found", path[0])).with_span(Span::new(0, 0))
            })
    }

    /// Suggest a cast for common type mismatches (e.g. Int ↔ Float).
    fn suggest_cast(&self, expected: TypeId, actual: TypeId) -> Option<String> {
        let (e, a) = (self.ctx.get(expected), self.ctx.get(actual));
        match (e, a) {
            (TypeData::Int { .. }, TypeData::Float { .. })
            | (TypeData::Float { .. }, TypeData::Int { .. }) => {
                Some("try using `as` to cast between integer and float types".into())
            }
            (TypeData::Bool, TypeData::Int { .. }) => {
                Some("try `x != 0` to convert Int to Bool".into())
            }
            (TypeData::Int { .. }, TypeData::Bool) => {
                Some("try `if x { 1 } else { 0 }` to convert Bool to Int".into())
            }
            _ => None,
        }
    }

    fn unify(&mut self, expected: TypeId, actual: TypeId, span: Span) -> Result<(), Diagnostic> {
        self.ctx
            .unify(expected, actual)
            .map(|_| ())
            .map_err(|_err| {
                let msg = format!(
                    "type mismatch: expected {:?}, found {:?}",
                    self.ctx.get(expected),
                    self.ctx.get(actual)
                );
                let mut diag = Diagnostic::error(msg).with_code("E030").with_span(span);
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion);
                }
                diag
            })
    }

    fn unify_with(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
        ctx: TypingContext,
    ) -> Result<(), Diagnostic> {
        self.ctx
            .unify(expected, actual)
            .map(|_| ())
            .map_err(|_err| {
                let msg = match ctx {
                    TypingContext::ReturnValue => {
                        format!(
                            "return value type mismatch: expected {:?}, found {:?}",
                            self.ctx.get(expected),
                            self.ctx.get(actual)
                        )
                    }
                    TypingContext::StructFieldInit => {
                        format!(
                            "field initializer type mismatch: expected {:?}, found {:?}",
                            self.ctx.get(expected),
                            self.ctx.get(actual)
                        )
                    }
                    TypingContext::Condition => {
                        format!("condition must be boolean, got {:?}", self.ctx.get(actual))
                    }
                    TypingContext::Argument { index, total } => {
                        format!(
                            "argument {} of {} has wrong type: expected {:?}, found {:?}",
                            index + 1,
                            total,
                            self.ctx.get(expected),
                            self.ctx.get(actual)
                        )
                    }
                    TypingContext::ClosureBody => {
                        format!(
                            "closure body type mismatch: expected {:?}, found {:?}",
                            self.ctx.get(expected),
                            self.ctx.get(actual)
                        )
                    }
                    TypingContext::None => {
                        format!(
                            "type mismatch: expected {:?}, found {:?}",
                            self.ctx.get(expected),
                            self.ctx.get(actual)
                        )
                    }
                    TypingContext::Index => {
                        format!("index must be an integer, got {:?}", self.ctx.get(actual))
                    }
                };
                let mut diag = Diagnostic::error(msg)
                    .with_code("E030")
                    .with_span(span)
                    .with_label(span, format!("expected {:?}", self.ctx.get(expected)));
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion);
                }
                diag
            })
    }

    fn binary_op_type(
        &mut self,
        op: BinOp,
        left: TypeId,
        right: TypeId,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        self.unify_with(left, right, span, TypingContext::None)?;

        // Helper: check if a type is or may become numeric (handles InferVar).
        let is_or_may_be_numeric = |ty: TypeId| -> bool {
            self.ctx.is_numeric(ty)
                || matches!(self.ctx.get(ty), TypeData::InferVar { id }
                    if self.infer.get_var_kind(*id) == Some(TypeVariableKind::Numeric)
                        || self.infer.get_var_kind(*id) == Some(TypeVariableKind::Integer)
                        || self.infer.get_var_kind(*id) == Some(TypeVariableKind::Float))
        };
        let is_or_may_be_integer = |ty: TypeId| -> bool {
            self.ctx.is_integer(ty)
                || matches!(self.ctx.get(ty), TypeData::InferVar { id }
                    if self.infer.get_var_kind(*id) == Some(TypeVariableKind::Integer)
                        || self.infer.get_var_kind(*id) == Some(TypeVariableKind::Numeric))
        };
        let is_or_may_be_bool = |ty: TypeId| -> bool {
            self.ctx.is_bool(ty)
                || matches!(self.ctx.get(ty), TypeData::InferVar { id }
                    if self.infer.get_var_kind(*id) == Some(TypeVariableKind::Bool))
        };

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
                if is_or_may_be_numeric(left) =>
            {
                Ok(left)
            }
            BinOp::AddWrap
            | BinOp::SubWrap
            | BinOp::MulWrap
            | BinOp::AddSaturate
            | BinOp::SubSaturate
            | BinOp::MulSaturate
            | BinOp::AddTrap
            | BinOp::SubTrap
            | BinOp::MulTrap
                if is_or_may_be_integer(left) =>
            {
                Ok(left)
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
                if is_or_may_be_integer(left) =>
            {
                Ok(left)
            }
            BinOp::Eq | BinOp::Neq => Ok(self.ctx.bool()),
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge if is_or_may_be_numeric(left) => {
                Ok(self.ctx.bool())
            }
            BinOp::And | BinOp::Or if is_or_may_be_bool(left) => {
                self.unify_with(left, right, span, TypingContext::None)?;
                Ok(self.ctx.bool())
            }
            _ => Err(Diagnostic::error("invalid operands for binary operator").with_span(span)),
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
            if (self.ctx.is_numeric(from) && self.ctx.is_numeric(to))
                || (self.ctx.is_bool(from) && self.ctx.is_integer(to))
                || (self.ctx.is_integer(from) && self.ctx.is_bool(to))
            {
                Ok(to)
            } else if self.ctx.is_reference(from) {
                Err(Diagnostic::error(
                    "safe cast from reference type requires explicit dereference or unsafe cast",
                )
                .with_code("E601")
                .with_span(span)
                .with_suggestion("consider dereferencing first: `*expr as TargetType`")
                .with_suggestion("or use `as!` for an unsafe bitcast"))
            } else {
                Err(
                    Diagnostic::error("safe cast only allowed between numeric and boolean types")
                        .with_code("E601")
                        .with_span(span)
                        .with_suggestion("use `From` trait for non-primitive type conversions"),
                )
            }
        } else {
            if (self.ctx.is_numeric(from) && self.ctx.is_numeric(to))
                || (self.ctx.is_reference(from) && self.ctx.is_pointer(to))
                || (self.ctx.is_pointer(from) && self.ctx.is_reference(to))
            {
                Ok(to)
            } else if let (TypeData::Ptr { .. }, TypeData::Ptr { .. }) =
                (self.ctx.get(from), self.ctx.get(to))
            {
                Ok(to)
            } else if self.ctx.is_reference(from) && self.ctx.is_integer(to) {
                Err(
                    Diagnostic::error("unsafe cast from reference to integer not yet supported")
                        .with_code("E601")
                        .with_span(span)
                        .with_suggestion("consider using `*expr as usize` via a pointer cast"),
                )
            } else {
                Err(Diagnostic::error("unsafe cast requires compatible types (numeric<->numeric, ref<->ptr, ptr<->ptr)")
                    .with_code("E601")
                    .with_span(span))
            }
        }
    }

    /// Check that a type satisfies the `Sized` bound.
    /// Concrete types are implicitly `Sized`.  Type parameters are assumed
    /// sized by default (the standard conservative choice).  Unresolved
    /// infer vars get a deferred `Impl` constraint.
    fn require_type_sized(&mut self, ty: TypeId, span: Span) {
        let resolved = self.ctx.resolve_binding(ty);
        match self.ctx.get(resolved) {
            TypeData::InferVar { .. } => {
                self.add_constraint(Constraint::Impl(ty, DefId(0), span));
            }
            _ => {} // concrete types and generic params: assumed Sized
        }
    }

    fn check_result_type(&self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(ok_ty) = self.extract_ok_type(ty) {
            Ok(ok_ty)
        } else {
            Err(Diagnostic::error("try operator requires Result type").with_span(span))
        }
    }

    fn check_future_type(&self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(future_ty) = self.extract_future_type(ty) {
            Ok(future_ty)
        } else {
            Err(Diagnostic::error("await operator requires Future type").with_span(span))
        }
    }

    fn extract_ok_type(&self, ty: TypeId) -> Option<TypeId> {
        if let TypeData::Enum { def_id: did, args } = self.ctx.get(ty) {
            if let Some(result_id) = self.known_def_id("Result") {
                if *did == result_id && args.len() == 2 {
                    return Some(args[0]);
                }
            }
        }
        None
    }

    fn extract_future_type(&self, ty: TypeId) -> Option<TypeId> {
        if let TypeData::Enum { def_id: did, args } = self.ctx.get(ty) {
            if let Some(future_id) = self.known_def_id("Future") {
                if *did == future_id && args.len() == 1 {
                    return Some(args[0]);
                }
            }
        }
        None
    }

    fn extract_result_types(&self, ty: TypeId, span: Span) -> Result<(TypeId, TypeId), Diagnostic> {
        if let TypeData::Enum { def_id: did, args } = self.ctx.get(ty) {
            if let Some(result_id) = self.known_def_id("Result") {
                if *did == result_id && args.len() == 2 {
                    return Ok((args[0], args[1]));
                }
            }
        }
        Err(Diagnostic::error("catch requires Result type").with_span(span))
    }

    fn known_def_id(&self, name: &str) -> Option<DefId> {
        self.symbols.lookup_type(name).map(|b| b.def_id)
    }

    /// Resolve a trait path from a bound `Type` (e.g. `Add` or `Add<Int<32>>`) to a `DefId`.
    fn resolve_trait_path(&self, bound: &Type) -> Option<DefId> {
        let name = match bound {
            Type::Path(path, _) => path.first()?,
            Type::Generic(base, ..) => match base.as_ref() {
                Type::Path(path, _) => path.first()?,
                _ => return None,
            },
            _ => return None,
        };
        self.symbols.lookup_trait(name).map(|b| b.def_id)
    }

    /// Attempt to dereference a type once using built-in rules.
    /// Handles `&T` / `&mut T`, `*T`, `Ptr<pointee = T>`, and known wrapper types.
    fn builtin_deref_ty(&self, ty: TypeId) -> Option<TypeId> {
        // Deweference `&T` / `&mut T` → `T` uwu
        if let Some(inner) = self.ctx.pointee_of_ref(ty) {
            return Some(inner);
        }
        // Deweference `*T` → `T` (つω`｡)
        if let Some(inner) = self.ctx.pointee_of_pointer(ty) {
            return Some(inner);
        }
        // Deweference `Ptr<pointee = T>` → `T` (*＾▽＾)／
        if let TypeData::Ptr { pointee, .. } = self.ctx.get(ty) {
            return Some(*pointee);
        }
        // Try dewefewence via `Deref` twait with `@auto_dewef` mawk uwu
        self.try_deref_trait_step(ty)
    }

    /// Attempt to dereference through a `Deref` trait impl marked `@auto_deref`.
    fn try_deref_trait_step(&self, ty: TypeId) -> Option<TypeId> {
        let deref_trait_id = self.symbols.lookup_trait("Deref").map(|b| b.def_id)?;
        let candidates = self.trait_env.lookup_impls_for_type(ty);
        for cand in candidates {
            if cand.trait_id == deref_trait_id && cand.has_auto_deref {
                if let Some(target_ty) = cand
                    .assoc_tys
                    .iter()
                    .find(|(name, _)| name == "Target")
                    .map(|(_, ty)| *ty)
                {
                    return Some(target_ty);
                }
            }
        }
        None
    }

    /// Walk the autoderef chain up to MAX_DEREFS steps, yielding each intermediate type.
    fn autoderef_chain<'s>(&'s self, ty: TypeId) -> AutoderefIter<'s> {
        AutoderefIter {
            checker: self,
            current: Some(ty),
            depth: 0,
            max_depth: 10,
        }
    }

    /// Local type argument synthesis (Pierce & Turner 2000, §3).
    /// When a function type's parameters contain GenericParam (uninstantiated type
    /// variables), this creates fresh InferVars for them, infers argument types,
    /// unifies to bind the InferVars, and returns the resolved call result.
    fn try_synthesize_type_args(
        &mut self,
        callee_hir: &HirExpr,
        callee_ty: TypeId,
        args: &[Expr],
        comptime: bool,
        expected: Option<TypeId>,
        span: Span,
    ) -> Result<Option<(HirExpr, TypeId)>, Diagnostic> {
        // Peel off Forall layers to get the underlying Fn type.
        // For polymorphic functions, the type is wrapped as:
        //   Forall(0, "T", Forall(1, "U", Fn { params: [...], ret: ... }))
        // We strip the Forall nodes and recover the Fn body.
        let mut inner_ty = callee_ty;
        loop {
            match self.ctx.get(inner_ty) {
                TypeData::Forall { body, .. } => inner_ty = *body,
                _ => break,
            }
        }

        // Only works on Fn types
        let (params, ret) = match self
            .ctx
            .params_of_fn(inner_ty)
            .zip(self.ctx.ret_of_fn(inner_ty))
        {
            Some(p) => p,
            None => return Ok(None),
        };
        let param_tys = params.to_vec();

        // Collect GenericParam indices from parameter types AND return type
        let mut generic_indices: Vec<usize> = Vec::new();
        for &pt in &param_tys {
            Self::collect_generic_param_indices(pt, &self.ctx, &mut generic_indices);
        }
        Self::collect_generic_param_indices(ret, &self.ctx, &mut generic_indices);
        generic_indices.sort();
        generic_indices.dedup();
        if generic_indices.is_empty() {
            return Ok(None);
        }

        // Create fresh InferVars for each GenericParam index
        let mut infer_var_for_index: Vec<TypeId> = Vec::new();
        for _ in &generic_indices {
            let var = self.new_infer_var(TypeVariableKind::Any);
            infer_var_for_index.push(var);
        }

        // Build substitution: GenericParam index → fresh InferVar
        let mut subst = Subst::new();
        for (&gp_idx, &var) in generic_indices.iter().zip(infer_var_for_index.iter()) {
            subst.insert(gp_idx, var);
        }

        // Substitute the InferVars into param types and return type
        let substituted_params: Vec<TypeId> = param_tys
            .iter()
            .map(|&pt| self.ctx.subst(pt, &subst))
            .collect();
        let substituted_ret = self.ctx.subst(ret, &subst);

        // Check arity
        if substituted_params.len() != args.len() {
            return Err(Diagnostic::error(format!(
                "wrong number of arguments: expected {}, found {}",
                substituted_params.len(),
                args.len()
            ))
            .with_span(span));
        }

        // If an expected type is provided (checking mode), proceed conservatively:
        // if the return type contains any InferVar in contravariant position (e.g.
        // inside Fn params), fall back to let the normal call path handle it.
        // Otherwise, try unifying with the expected type — if that fails, fall back
        // rather than erroring, since the normal path may produce a better diagnostic.
        if let Some(exp_ty) = expected {
            // Quick check for contravariant occurrences: if any InferVar appears
            // inside Fn params within the return type, fall back.
            let has_contra = Self::type_var_in_problematic_position(
                substituted_ret,
                &infer_var_for_index,
                &self.ctx,
            );
            if has_contra {
                return Ok(None);
            }
            // Try unification — if it fails, don't error; just fall back.
            if self.ctx.unify(substituted_ret, exp_ty).is_err() {
                return Ok(None);
            }
        }

        // Infer argument types and unify with substituted parameter types
        let mut hir_args = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let expected_param_ty = substituted_params
                .get(i)
                .copied()
                .unwrap_or(self.ctx.error());
            let hir_arg = self.check_expr(
                arg,
                Expectation::HasType(expected_param_ty),
                TypingContext::Argument {
                    index: i,
                    total: args.len(),
                },
            )?;
            hir_args.push(hir_arg);
        }

        // After unification, the InferVars have been bound to concrete types.
        // Create a final substitution from GenericParam indices to their resolved types.
        let mut final_subst = Subst::new();
        for (&gp_idx, &var) in generic_indices.iter().zip(infer_var_for_index.iter()) {
            let resolved = self.ctx.resolve_binding(var);
            // Cannot resolve — reuse the InferVar itself; the caller will fallback
            if self.ctx.is_error(resolved) || self.ctx.is_infer_var(resolved) {
                return Ok(None);
            }
            final_subst.insert(gp_idx, resolved);
        }

        // Apply the resolved substitution to the return type
        let final_ret = self.ctx.subst(ret, &final_subst);
        Ok(Some((
            HirExpr::Call {
                callee: Box::new(callee_hir.clone()),
                args: hir_args,
                comptime,
                ty: final_ret,
                span,
            },
            final_ret,
        )))
    }

    /// Collect all GenericParam indices appearing in a type.
    fn collect_generic_param_indices(ty: TypeId, ctx: &TypeContext, out: &mut Vec<usize>) {
        match ctx.get(ty) {
            TypeData::GenericParam { index, .. } => out.push(*index),
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                for &a in args {
                    Self::collect_generic_param_indices(a, ctx, out);
                }
            }
            TypeData::Tuple { elems } => {
                for &e in elems {
                    Self::collect_generic_param_indices(e, ctx, out);
                }
            }
            TypeData::Array { elem, .. } => Self::collect_generic_param_indices(*elem, ctx, out),
            TypeData::Slice { elem } => Self::collect_generic_param_indices(*elem, ctx, out),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                Self::collect_generic_param_indices(*ty, ctx, out);
            }
            TypeData::Ptr { pointee, .. } => {
                Self::collect_generic_param_indices(*pointee, ctx, out)
            }
            TypeData::Fn { params, ret } => {
                for &p in params {
                    Self::collect_generic_param_indices(p, ctx, out);
                }
                Self::collect_generic_param_indices(*ret, ctx, out);
            }
            TypeData::AssociatedType { self_ty, .. } => {
                Self::collect_generic_param_indices(*self_ty, ctx, out)
            }
            TypeData::Exists { base, .. } => Self::collect_generic_param_indices(*base, ctx, out),
            TypeData::Poly { body, .. } => Self::collect_generic_param_indices(*body, ctx, out),
            _ => {}
        }
    }

    /// Check if any of the given InferVars appear in a position where
    /// unification with an expected type could be unsound:
    /// - Inside Fn params (contravariant)
    /// - Inside Ref/Pointer/Ptr (invariant)
    /// If so, we conservatively fall back to normal call handling.
    fn type_var_in_problematic_position(ty: TypeId, vars: &[TypeId], ctx: &TypeContext) -> bool {
        match ctx.get(ty) {
            TypeData::Fn { params, ret } => {
                // Fn params are contravariant — check each param for vars
                for &p in params {
                    for &v in vars {
                        if Self::type_tree_contains(p, v, ctx) {
                            return true;
                        }
                    }
                }
                // Return type is covariant — safe to recurse normally
                Self::type_var_in_problematic_position(*ret, vars, ctx)
            }
            // Ref/Pointer/Ptr are invariant — if any var appears inside, it's risky
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                for &v in vars {
                    if Self::type_tree_contains(*ty, v, ctx) {
                        return true;
                    }
                }
                false
            }
            TypeData::Ptr { pointee, .. } => {
                for &v in vars {
                    if Self::type_tree_contains(*pointee, v, ctx) {
                        return true;
                    }
                }
                false
            }
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => args
                .iter()
                .any(|&a| Self::type_var_in_problematic_position(a, vars, ctx)),
            TypeData::Tuple { elems } => elems
                .iter()
                .any(|&e| Self::type_var_in_problematic_position(e, vars, ctx)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                Self::type_var_in_problematic_position(*elem, vars, ctx)
            }
            TypeData::AssociatedType { self_ty, .. } => {
                Self::type_var_in_problematic_position(*self_ty, vars, ctx)
            }
            TypeData::Exists { base, .. } => {
                Self::type_var_in_problematic_position(*base, vars, ctx)
            }
            TypeData::Poly { body, .. } => Self::type_var_in_problematic_position(*body, vars, ctx),
            _ => false,
        }
    }

    /// Check if a specific TypeId appears anywhere in a type tree.
    fn type_tree_contains(ty: TypeId, target: TypeId, ctx: &TypeContext) -> bool {
        let resolved = ctx.resolve_binding(ty);
        if resolved == ctx.resolve_binding(target) {
            return true;
        }
        match ctx.get(resolved) {
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => args
                .iter()
                .any(|&a| Self::type_tree_contains(a, target, ctx)),
            TypeData::Tuple { elems } => elems
                .iter()
                .any(|&e| Self::type_tree_contains(e, target, ctx)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                Self::type_tree_contains(*elem, target, ctx)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                Self::type_tree_contains(*ty, target, ctx)
            }
            TypeData::Ptr { pointee, .. } => Self::type_tree_contains(*pointee, target, ctx),
            TypeData::Fn { params, ret } => {
                params
                    .iter()
                    .any(|&p| Self::type_tree_contains(p, target, ctx))
                    || Self::type_tree_contains(*ret, target, ctx)
            }
            TypeData::AssociatedType { self_ty, .. } => {
                Self::type_tree_contains(*self_ty, target, ctx)
            }
            TypeData::Exists { base, .. } => Self::type_tree_contains(*base, target, ctx),
            TypeData::Poly { body, .. } => Self::type_tree_contains(*body, target, ctx),
            _ => false,
        }
    }

    fn lookup_field(&self, ty: TypeId, name: &str, span: Span) -> Result<TypeId, Diagnostic> {
        // Collect field names from all types in the deref chain for error reporting
        let mut all_field_names: Vec<String> = Vec::new();

        // Try direct lookup first
        if let TypeData::Struct { def_id, args } = self.ctx.get(ty) {
            let binding = self
                .symbols
                .lookup_type_by_def_id(*def_id)
                .ok_or_else(|| Diagnostic::error("struct definition not found").with_span(span))?;
            all_field_names.extend(binding.fields.iter().map(|f| f.name.clone()));
            if let Some(field) = binding.fields.iter().find(|f| f.name == name) {
                let mut subst = Subst::new();
                for (i, _param) in binding.params.iter().enumerate() {
                    if let Some(&arg) = args.get(i) {
                        subst.insert(i, arg);
                    }
                }
                return Ok(self.ctx.subst(field.ty, &subst));
            }
        }

        // Walk autoderef chain, skipping the original type (already tried)
        for deref_ty in self.autoderef_chain(ty).skip(1) {
            if let TypeData::Struct { def_id, args } = self.ctx.get(deref_ty) {
                let binding = self.symbols.lookup_type_by_def_id(*def_id).ok_or_else(|| {
                    Diagnostic::error("struct definition not found").with_span(span)
                })?;
                all_field_names.extend(binding.fields.iter().map(|f| f.name.clone()));
                if let Some(field) = binding.fields.iter().find(|f| f.name == name) {
                    let mut subst = Subst::new();
                    for (i, _param) in binding.params.iter().enumerate() {
                        if let Some(&arg) = args.get(i) {
                            subst.insert(i, arg);
                        }
                    }
                    return Ok(self.ctx.subst(field.ty, &subst));
                }
            }
        }

        // Build an informative error message
        let mut diag = Diagnostic::error(format!("no field `{}` found on type", name))
            .with_code("E010")
            .with_span(span);

        if !all_field_names.is_empty() {
            diag =
                diag.with_suggestion(format!("available fields: {}", all_field_names.join(", ")));
            if let Some(suggestion) = did_you_mean_suggestion(name, &all_field_names) {
                diag = diag.with_suggestion(suggestion);
            }
        }

        Err(diag)
    }

    /// Look up a method by name on a type, walking the autoderef chain.
    /// Returns `(param_types, return_type)` if found.
    fn lookup_method(&mut self, ty: TypeId, name: &str) -> Option<(Vec<TypeId>, TypeId)> {
        for current_ty in self.autoderef_chain(ty) {
            // Check inherent methods first (registered via `impl Type { ... }`)
            for method in self.trait_env.lookup_inherent_methods(current_ty, self.ctx) {
                if method.name == name {
                    return Some((method.param_tys.clone(), method.ret_ty));
                }
            }

            // Collect trait impl methods with matching name, then resolve types
            // outside the borrow of self.trait_env.
            let mut pending: Vec<(Vec<crate::ast::Param>, crate::ast::Type)> = Vec::new();
            for cand in self.trait_env.lookup_impls_for_type(current_ty) {
                for method in &cand.methods {
                    if method.name == name {
                        pending.push((method.params.clone(), method.return_type.clone()));
                    }
                }
            }
            for (params, return_type) in pending {
                let mut param_tys = Vec::with_capacity(params.len());
                for p in &params {
                    if let Some(ref param_ty) = p.ty {
                        match self.resolve_type(param_ty) {
                            Ok(ty_id) => param_tys.push(ty_id),
                            Err(_) => return None,
                        }
                    } else {
                        param_tys.push(self.ctx.error());
                    }
                }
                let ret_ty = self.resolve_type(&return_type).ok()?;
                return Some((param_tys, ret_ty));
            }
        }
        None
    }

    fn lookup_attr(&self, ty: TypeId, name: &str, span: Span) -> Result<TypeId, Diagnostic> {
        match name {
            "len" if self.ctx.is_array(ty) || self.ctx.is_slice(ty) => Ok(self.ctx.usize()),
            "size"
                if self.ctx.is_integer(ty) || self.ctx.is_float(ty) || self.ctx.is_pointer(ty) =>
            {
                Ok(self.ctx.usize())
            }
            "align" => Ok(self.ctx.usize()),
            "default" => Ok(ty),
            _ => Err(Diagnostic::error(format!("unknown attribute '{}'", name)).with_span(span)),
        }
    }

    fn lookup_type_default_expr(
        &mut self,
        ty_id: TypeId,
        span: Span,
    ) -> Result<Option<Expr>, Diagnostic> {
        let resolved = self.ctx.resolve_binding(ty_id);
        let def_id = match self.ctx.get(resolved) {
            TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => Some(*def_id),
            _ => None,
        };
        if let Some(def_id) = def_id {
            if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id) {
                if binding.no_default {
                    self.diagnostics.push(
                        Diagnostic::error("type forbids implicit initialization (no_default)")
                            .with_span(span),
                    );
                    return Ok(None);
                }
                if let Some(ref default_expr) = binding.default_value {
                    return Ok(Some(default_expr.clone()));
                }
            }
        }
        Ok(None)
    }

    fn block_type(&self, stmts: &[HirStmt]) -> TypeId {
        for stmt in stmts.iter().rev() {
            match stmt {
                HirStmt::Expression(expr) => {
                    if !matches!(expr.as_ref(), HirExpr::Error(_)) {
                        return expr.ty();
                    }
                }
                HirStmt::Return {
                    value: Some(value), ..
                } => {
                    if !matches!(value.as_ref(), HirExpr::Error(_)) {
                        return value.ty();
                    }
                }
                HirStmt::Return { value: None, .. }
                | HirStmt::Leave { .. }
                | HirStmt::Continue { .. } => return self.ctx.never(),
                _ => {}
            }
        }
        self.ctx.unit()
    }

    fn get_trait_id_for_binop(&self, op: BinOp, span: Span) -> Result<Option<DefId>, Diagnostic> {
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
        Ok(self.symbols.lookup_trait(trait_name).map(|b| b.def_id))
    }

    fn extract_int_from_type(&self, ty: &Type) -> Option<u8> {
        if let Type::Literal(expr, _) = ty {
            if let Expr::Literal(Literal::Int(val), _) = expr.as_ref() {
                if *val > 64 {
                    return None; // reject out-of-range bit widths silently
                }
                return Some(*val as u8);
            }
        }
        None
    }

    fn new_infer_var(&mut self, kind: TypeVariableKind) -> TypeId {
        self.infer.new_type_var(self.ctx, kind)
    }
    fn add_constraint(&mut self, c: Constraint) {
        self.infer.add_constraint(c);
    }

    /// Look up the TypeBinding for a Struct or Enum type, if available.
    fn lookup_type_binding(&self, ty: TypeId) -> Option<TypeBinding> {
        let resolved = self.ctx.resolve_binding(ty);
        match self.ctx.get(resolved) {
            TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => {
                self.symbols.lookup_type_by_def_id(*def_id).cloned()
            }
            _ => None,
        }
    }
}

#[cfg(test)]
pub mod tests;
