use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCollector};
use crate::hir::hir::*;
use crate::hir::infer::*;
use crate::hir::resolver::ResolutionMap;
use crate::hir::symbol::*;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use crate::symbol::Symbol;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;
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
/// A scoped map of variable name → TypeId.
///
/// Maintains a stack of `HashMap` frames. New bindings are always
/// inserted into the innermost frame. Lookups search from innermost
/// to outermost, implementing lexical shadowing.
///
/// Uses `Rc<RefCell<...>>` for interior mutability so that
/// `VarScopeGuard` can own a separate `Rc` reference and pop frames
/// in its `Drop` without holding any borrow on the `TypeChecker`.
///
/// This replaces a flat `HashMap` that leaked bindings across scope
/// boundaries (e.g. `if let Some(x) = ... { }` would leave `x` in
/// scope after the block).
#[derive(Debug, Clone)]
pub struct ScopedVarMap {
    frames: Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>>,
}

impl ScopedVarMap {
    pub fn new() -> Self {
        ScopedVarMap {
            frames: Rc::new(RefCell::new(vec![HashMap::new()])),
        }
    }

    /// Push a new, empty scope frame.
    pub fn push_frame(&self) {
        self.frames.borrow_mut().push(HashMap::new());
    }

    /// Pop the innermost scope frame, discarding its bindings.
    pub fn pop_frame(&self) {
        self.frames.borrow_mut().pop();
    }

    /// Insert a binding into the innermost scope frame.
    pub fn insert(&self, name: Symbol, ty: TypeId) {
        self.frames.borrow_mut().last_mut().unwrap().insert(name, ty);
    }

    /// Insert a binding into the base (outermost) scope frame.
    /// Used for caching global/module‑level variable types so they
    /// persist across all nested scopes.
    pub fn insert_global(&self, name: Symbol, ty: TypeId) {
        self.frames.borrow_mut()[0].insert(name, ty);
    }

    /// Look up a binding, searching from innermost to outermost scope.
    pub fn get(&self, name: Symbol) -> Option<TypeId> {
        let frames = self.frames.borrow();
        for frame in frames.iter().rev() {
            if let Some(&ty) = frame.get(&name) {
                return Some(ty);
            }
        }
        None
    }

    /// Extend the innermost frame with an iterator of bindings.
    pub fn extend(&self, iter: impl IntoIterator<Item = (Symbol, TypeId)>) {
        self.frames.borrow_mut().last_mut().unwrap().extend(iter);
    }

    /// Return a clone of the inner `Rc` so a guard can
    /// operate independently of any borrow on this struct.
    fn rc_clone(&self) -> Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>> {
        Rc::clone(&self.frames)
    }
}

/// RAII guard that pops a variable scope frame on drop.
///
/// Returned by `TypeChecker::enter_var_scope()`. Ensures the frame is
/// popped even when the enclosing function returns early via `?`.
///
/// Owns its own `Rc` reference to the frames vector, completely
/// independent of any borrow on the `TypeChecker` or `ScopedVarMap`.
pub(crate) struct VarScopeGuard {
    frames: Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>>,
}

impl VarScopeGuard {
    fn new(frames: Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>>) -> Self {
        frames.borrow_mut().push(HashMap::new());
        VarScopeGuard { frames }
    }
}

impl Drop for VarScopeGuard {
    fn drop(&mut self) {
        self.frames.borrow_mut().pop();
    }
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
    /// Stack of (inference_context, region_tree_snapshot) for scope management.
    /// Storing a snapshot of the region tree allows abort_inference_scope
    /// to roll back any CtxFrames and region structure changes made inside
    /// the aborted scope.
    infer_stack: Vec<(InferenceContext, region::RegionTree)>,
    /// Wegion twee: twacks cuwwent function, woop, cwosuwe, etc.
    /// Wepwaces the owd wineaw `woop_stack` with a twee stwuctuwe
    /// suppowting pawtiaw genewawization (OmniML §3.2). (｀・ω・´)
    region_tree: RegionTree,
    /// Scoped cache of variable types, managed as a stack of frames.
    /// A new frame is pushed on block entry and popped on block exit.
    /// Ovewwides the wesowvew's pwacehowdew `ewrow` type. (◕‿◕)
    local_variable_types: ScopedVarMap,
    /// Pre-resolved by NameResolver: variable name → TypeId
    resolution_map: ResolutionMap,
    /// Local cache of generic type parameter types (e.g. `T` in `def foo<T>(x: T)`).
    /// Populated when processing function definitions with type_params.
    local_type_param_cache: HashMap<Symbol, TypeId>,
    /// SCAP-style guarantee chain: tracks outstanding postconditions that must
    /// be discharged on function return (Feng & Shao 2006 §4).
    guarantee_chain: GuaranteeChain,
    /// Names of mutable global variables (top-level `set mut`).
    /// These can only be read/written inside `@trusted` functions.
    mutable_globals: HashSet<Symbol>,
    /// Whether the current function is annotated `@trusted`.
    current_function_trusted: bool,
    /// Registry of comptime functions: name → (param_names, body).
    /// Populated as the checker encounters `comptime def` functions and
    /// passed to ComptimeEvalContext for comptime block evaluation.
    comptime_fn_registry: HashMap<Symbol, (Vec<Symbol>, Vec<HirStmt>)>,
    /// Whether we are currently in the comptime-function-body pass (Pass 2).
    /// When true, ComptimeBlock evaluation is deferred to after Pass 2 so
    /// that forward references between comptime functions work correctly.
    comptime_fn_pass: bool,
    /// Deferred comptime blocks collected during Pass 2.  Evaluated after
    /// all comptime function bodies are registered.
    deferred_comptime_blocks: Vec<(Vec<HirStmt>, TypeId, Span)>,
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
            local_variable_types: ScopedVarMap::new(),
            local_type_param_cache: HashMap::new(),
            resolution_map,
            guarantee_chain: GuaranteeChain::new(),
            mutable_globals: HashSet::new(),
            current_function_trusted: false,
            comptime_fn_registry: HashMap::new(),
            comptime_fn_pass: false,
            deferred_comptime_blocks: Vec::new(),
        };
        checker
    }

    /// Find the innermost bweak tawget (Woop, Whiwe, Fow, WabewedBwock) (*＾▽＾)／
    /// Wetuwns the tawget's span and optionaw wabew. If `wabew` is Some, onwy match same-named WabewedBwock.
    /// Find the innermost continue tawget (onwy Woop, Whiwe, Fow) ☆ﾟ.*･｡ﾟ
    pub fn check_program(&mut self, program: &Program) -> Result<HirProgram, DiagnosticCollector> {
        let mut items = Vec::new();

        // Wrap the entire program in an inference scope so that
        // top‑level statements (variable defs, expression stmts, etc.)
        // also have their Eq/Impl/Match constraints solved and finalized.
        // Previously the solver only ran inside function bodies via
        // enter_inference_scope in check_stmt(FunctionDef).
        self.enter_inference_scope();

        // Pass 1: register all comptime function signatures (name + param names)
        // WITHOUT checking bodies, so that forward references between comptime
        // functions work correctly (e.g. `comptime def f() { g() }` followed by
        // `comptime def g() { 42 }`).
        let comptime_fn_indices: Vec<usize> = program.items.iter().enumerate().filter_map(|(i, stmt)| {
            if let Stmt::FunctionDef { name, params, is_comptime, .. } = stmt {
                if *is_comptime {
                    let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
                    self.comptime_fn_registry.insert(*name, (param_names, Vec::new()));
                    Some(i)
                } else { None }
            } else { None }
        }).collect();

        // Pass 2: type-check all comptime function bodies (all signatures are now available).
        // During this pass, comptime blocks inside comptime function bodies are deferred
        // so that forward references to comptime functions defined later work correctly.
        self.comptime_fn_pass = true;
        for &i in &comptime_fn_indices {
            match self.check_stmt(&program.items[i]) {
                Ok(hir) => items.push(hir),
                Err(diag) => {
                    self.diagnostics.push(diag);
                    items.push(HirStmt::Error);
                }
            }
        }
        self.comptime_fn_pass = false;

        // Evaluate deferred comptime blocks from Pass 2.  Now all comptime function
        // bodies are registered, so forward references will resolve correctly.
        for (hir, ty, span) in self.deferred_comptime_blocks.drain(..) {
            let mut eval = crate::hir::comptime::ComptimeEvalContext::new(self.ctx, self.symbols);
            for (name, (params, body)) in &self.comptime_fn_registry {
                eval.register_fn(name.clone(), params.clone(), body.clone());
            }
            if let Err(e) = eval.eval_block(&hir) {
                self.diagnostics.push(
                    Diagnostic::error(format!("comptime error: {}", e))
                        .with_code_str("E080")
                        .with_span(span),
                );
            }
        }

        // Pass 3: type-check remaining items (non-comptime functions,
        // comptime blocks, type defs, etc.) in order.
        for (i, stmt) in program.items.iter().enumerate() {
            if comptime_fn_indices.contains(&i) {
                continue; // already processed in pass 2
            }
            match self.check_stmt(stmt) {
                Ok(hir) => items.push(hir),
                Err(diag) => {
                    self.diagnostics.push(diag);
                    items.push(HirStmt::Error);
                }
            }
        }

        // Expand `generate` blocks before solving constraints.  (This step
        // is now performed before name resolution, so the resolver never sees
        // unexpanded template bodies.  The expander call here is retained as
        // a safety net for any `Generate` nodes that might survive — but in
        // normal operation the list should already be fully expanded.)
        // Solve all queued constraints, finalize inference variables,
        // and commit the transaction.  On failure the transaction is
        // rolled back and the region tree is restored to its pre-scope state.
        // Generalization runs AFTER commit so that its side-effects
        // (gen_statuses, pool membership) are not split across a transaction
        // boundary — if the commit failed, there is nothing to roll back.
        let (prev, saved_tree) = self.infer_stack.pop().expect(
            "check_program: infer_stack is empty — \
             enter_inference_scope was never called",
        );
        let mut current = mem::replace(&mut self.infer, prev);
        let result = self.solve_current_ctx(&mut current);
        match result {
            Ok(()) => {
                self.ctx.commit_transaction();
                // Generalize all regions (OmniML §6 force_root_generalization),
                // AFTER the transaction is committed.  This is safe because
                // generalization only mutates the inference context (gen_statuses,
                // pools), which will be discarded along with `current` when this
                // function returns.  The TypeContext bindings are already finalized
                // by commit_transaction and are not affected by generalization.
                let _generalized = current.force_root_generalization(self.ctx);
            }
            Err(diags) => {
                self.ctx.rollback_transaction();
                current.region_tree.rollback_pool();
                self.region_tree = saved_tree;
                return Err(diags);
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
                type_captures,
                ..
            } => {
                // 'set' does not support pattern destructuring
                if *kind == VariableKind::Set && pattern.is_some() {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "`set` does not support pattern destructuring; use `let` instead",
                        )
                        .with_code_str("E001")
                        .with_span(*span),
                    );
                }

                // 'let' must have an explicit initializer
                if *kind == VariableKind::Let && value.is_none() {
                    self.diagnostics.push(
                        Diagnostic::error("`let` requires an explicit initializer; it cannot rely on a type's default value")
                            .with_code_str("E002")
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
                            .with_code_str("E003")
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

                // Track mutable global variables (top-level `set mut`).
                // These require `@trusted` context to be read/written.
                if *mutable && self.current_function.is_none() {
                    if let Some(var_name) = name {
                        self.mutable_globals.insert(var_name.clone());
                    }
                }

                // `set auto<T> = expr` — bind captured type names to the inferred type.
                // Each name in `type_captures` becomes available as a type alias in
                // comptime reflection (e.g., `@typeInfo!(T)`).
                if !type_captures.is_empty() {
                    if let Some(capture) = type_captures.first() {
                        self.local_type_param_cache
                            .insert(capture.name.clone(), final_ty);
                    }
                    // Future: support multiple captures (T, N, L) with compile-time
                    // constant extraction for non-type captures.
                    if type_captures.len() > 1 {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "multiple type captures not yet supported; only the first name is bound",
                            )
                            .with_span(*span),
                        );
                    }
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
                    type_captures: type_captures.clone(),
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
                // Collect names before insertion so we can clean up after the function body
                // is fully processed, preventing cross-function cache pollution.
                let fn_param_names: Vec<Symbol> = type_params.iter().map(|tp| tp.name).collect();
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

                // ── @interrupt handler checks ─────────────────────────
                let is_interrupt = attributes.iter().any(|a| a.name.eq_str("interrupt"));
                if is_interrupt {
                    // Rule 1: return type must be Never (!)
                    if !self.ctx.is_never(return_ty) {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler must return `!` (never type)")
                                .with_code_str("E050")
                                .with_span(*span)
                                .with_help("interrupt handlers must have return type `!` because they never return")
                        );
                    }
                    // Rule 2: no custom parameters
                    if !params.is_empty() {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler cannot have parameters")
                                .with_code_str("E051")
                                .with_span(*span)
                                .with_help("interrupt handlers take no arguments — state is read via MMIO or ghost variables")
                        );
                    }
                    // Rule 3: must have @no_alloc and @no_panic (both required for interrupt handlers)
                    let has_no_alloc = attributes.iter().any(|a| a.name.eq_str("no_alloc"));
                    let has_no_panic = attributes.iter().any(|a| a.name.eq_str("no_panic"));
                    if !has_no_alloc {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler must satisfy @no_alloc")
                                .with_code_str("E052")
                                .with_span(*span)
                                .with_suggestion("add `@no_alloc` to this function (redundant with `@no_panic`?)")
                        );
                    }
                    if !has_no_panic {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler must satisfy @no_panic")
                                .with_code_str("E053")
                                .with_span(*span)
                                .with_suggestion("add `@no_panic` to this function")
                        );
                    }
                    // Rule 4: @interrupt + @alloc is incompatible
                    if attributes.iter().any(|a| a.name.eq_str("alloc")) {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler cannot have @alloc")
                                .with_code_str("E054")
                                .with_span(*span)
                                .with_help("@interrupt and @alloc are incompatible — interrupt handlers must not allocate")
                        );
                    }
                    // Rule 5: @interrupt + @io is incompatible
                    if attributes.iter().any(|a| a.name.eq_str("io")) {
                        self.diagnostics.push(
                            Diagnostic::error("@interrupt handler cannot have @io")
                                .with_code_str("E055")
                                .with_span(*span)
                                .with_help("@interrupt and @io are incompatible — interrupt handlers must not perform I/O")
                        );
                    }
                }

                let guard = ScopeGuard::new(self);
                guard.checker.current_function = Some(DefId(0));
                guard.checker.current_function_trusted =
                    attributes.iter().any(|a| a.name.eq_str("trusted"));
                guard.checker.current_return_type = Some(return_ty);
                guard.checker.enter_inference_scope();
                guard.checker.push_ctx(CtxKind::Function, *span, None);

                // Enter a variable scope for the function body
                let _scope = guard.checker.enter_var_scope();

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
                    .insert(Symbol::intern("result"), return_ty);

                // SCAP: collect ensures conditions into the guarantee chain.
                // Each `ensures` becomes a postcondition that must hold at return.
                for contract in contracts {
                    if let Contract::Ensures { expr, .. } = contract {
                        let (_, ensures_ty) = guard
                            .checker
                            .infer_expr(expr)
                            .unwrap_or_else(|_| (HirExpr::Error(*span), guard.checker.ctx.bool()));
                        let g = Guarantee::new(Predicate::True, Predicate::Type(ensures_ty), None);
                        guard.checker.guarantee_chain.push(g);
                    }
                }
                // Generate where-clause constraints as Impl(clause_ty, trait_id)
                // so the solver can verify trait bounds on generic parameters.
                // Also expand constraint aliases (e.g. `where C: SortableContainer`
                // → Impl(C, Container) + Impl(C::Item, Ord) + ...).
                if let Some(wc) = where_clause {
                    for pred in &wc.predicates {
                        let pred_ty = guard.checker.resolve_type(&pred.ty)?;
                        for bound in &pred.bounds {
                            if let Some(trait_id) = guard.checker.resolve_trait_path(bound) {
                                guard
                                    .checker
                                    .add_constraint(Constraint::Impl(pred_ty, trait_id, pred.span));
                            } else if let Some(name) = TypeChecker::extract_bound_name(bound) {
                                if let Some(constraint) = guard.checker.symbols.lookup_constraint(name) {
                                    for &bound_ty in &constraint.bounds {
                                        if let Some(trait_id) = guard.checker.ctx.get_def_id_for_type(bound_ty) {
                                            guard.checker.add_constraint(
                                                Constraint::Impl(pred_ty, trait_id, pred.span),
                                            );
                                        }
                                    }
                                }
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

                let exit_res = guard.commit();

                if let Err(diags) = exit_res {
                    let details: Vec<String> = diags.iter().map(|d| d.message.clone()).collect();
                    return Err(Diagnostic::error(format!(
                        "inference failure: {}",
                        details.join("; ")
                    ))
                    .with_span(*span));
                }

                if let Some(ref body_stmts) = body_hir {
                    // Function bodies require explicit `return` — no implicit trailing expression.
                    let body_ty = self.block_type_impl(body_stmts, false);
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
                                        .with_code_str("E020")
                                        .with_span(*cspan)
                                        .with_label(expr.span(), format!("got {:?}", self.ctx.get(ty))),
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
                                        .with_code_str("E020")
                                        .with_span(*cspan)
                                        .with_label(expr.span(), format!("got {:?}", self.ctx.get(ty))),
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
                                    .with_code_str("E021")
                                    .with_span(*cspan)
                                    .with_label(expr.span(), format!("got {:?}", self.ctx.get(ty))),
                                );
                            }
                        }
                    }
                }

                // Pop variable scope — removes function params and `result` — via RAII
                // (the _scope guard above drops here on the normal path; on `?` it drops
                // implicitly via its Drop impl, preventing frame leaks.)

                let finally_hir = if let Some(finally) = finally {
                    let mut stmts = Vec::new();
                    for s in finally {
                        stmts.push(self.check_stmt(s)?);
                    }
                    Some(stmts)
                } else {
                    None
                };

                // ── Clean up generic parameter cache ─────────────────
                // Remove the inserted generic params so they don't leak into subsequent
                // function or block scopes.  `fn_param_names` was collected at entry.
                for name in &fn_param_names {
                    self.local_type_param_cache.remove(name);
                }

                // Register comptime functions in the global registry so that
                // `comptime { ... }` blocks can call them.
                if *is_comptime {
                    let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
                    if let Some(ref body) = body_hir {
                        self.comptime_fn_registry.insert(
                            name.clone(),
                            (param_names, body.clone()),
                        );
                    }
                }

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
                    is_ieee_contracts: attributes.iter().any(|a| a.name.eq_str("ieee_contracts")),
                    hints: attributes
                        .iter()
                        .filter(|a| a.name.eq_str("hint"))
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
                            .with_code_str("E004")
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
                let (pattern_hir, then_hir) = {
                    let _scope = self.enter_var_scope();
                    let p = self.check_pattern(pattern, scrut_ty)?;
                    let t = self.check_block(then_branch)?;
                    (p, t)
                }; // _scope dropped: pattern + then-branch bindings removed
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
                let _scope = self.enter_var_scope();
                let pattern_hir = self.check_pattern(pattern, scrut_ty)?;
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
                // scope drops here — removes pattern bindings
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
                let _scope = self.enter_var_scope();
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
                // scope drops here — removes pattern + block bindings
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
                let label_str = label.map(|l| l.as_str());
                let target = self.find_break_target(label_str.as_deref());
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
                                    .with_code_str("E005")
                                    .with_span(*span),
                            );
                        } else if label.is_some() {
                            self.diagnostics.push(
                                Diagnostic::error(format!("cannot `leave` with label `{}` – no matching labeled block or loop found", label.as_ref().unwrap()))
                                    .with_code_str("E005")
                                    .with_span(*span)
                            );
                        } else {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "`leave` statement outside of loop; use `return` instead",
                                )
                                .with_code_str("E005")
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
                let label_str = label.map(|l| l.as_str());
                let target = self.find_continue_target(label_str.as_deref());
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
                                .with_code_str("E006")
                                .with_span(*span),
                            );
                        } else {
                            self.diagnostics.push(
                                Diagnostic::error("`continue` statement outside of loop")
                                    .with_code_str("E006")
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
                    // The postcondition type (if present) must be bool,
                    // indicating the ensures clause holds at the return point.
                    if let Predicate::Type(post) = g.post {
                        if !self.ctx.is_bool(post) {
                            self.diagnostics.push(
                                Diagnostic::error("ensures condition must be boolean at return")
                                    .with_code_str("E022")
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
                            .with_code_str("E007")
                            .with_span(*span),
                    );
                }
                // Ban `return Err(...)` — use `leave with` instead
                if let Some(Expr::EnumLit { path, variant, .. }) = value {
                    if variant.eq_str("Err") && path.len() == 1 && path[0].eq_str("Result") {
                        self.diagnostics.push(
                            Diagnostic::error("`return Err(...)` is not valid; use `leave with` instead")
                                .with_code_str("E008")
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
                // Check that mutable globals are only assigned inside @trusted functions
                if let Expr::Ident(name, _) = target.as_ref() {
                    if self.mutable_globals.contains(name) && !self.current_function_trusted {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot assign to mutable global `{}` outside `@trusted` function",
                                name,
                            ))
                            .with_code_str("E040")
                            .with_span(*span)
                            .with_help("wrap the function in `@trusted` and add `requires`/`ensures` contracts")
                        );
                    }
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
                        // Extract the type of the comptime block from its last expression,
                        // so that `def f() -> Int<32> { comptime { 42 } }` type-checks.
                        let ty = hir.last().and_then(|s| match s {
                            HirStmt::Expression(e) => Some(e.ty()),
                            _ => None,
                        }).unwrap_or_else(|| self.ctx.unit());
                        if self.comptime_fn_pass {
                            // During Pass 2 (comptime function body checking), defer
                            // evaluation so that forward references to comptime functions
                            // defined later in the source are available at evaluation time.
                            // After Pass 2 completes, all deferred blocks are evaluated.
                            self.deferred_comptime_blocks.push((hir.clone(), ty, *span));
                        } else {
                            // Evaluate the comptime block at compile time.
                            let mut eval = crate::hir::comptime::ComptimeEvalContext::new(self.ctx, self.symbols);
                            // Register pre-collected comptime functions.
                            for (name, (params, body)) in &self.comptime_fn_registry {
                                eval.register_fn(name.clone(), params.clone(), body.clone());
                            }
                            if let Err(e) = eval.eval_block(&hir) {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("comptime error: {}", e))
                                        .with_code_str("E080")
                                        .with_span(*span),
                                );
                            }
                        }
                        Ok(HirStmt::ComptimeBlock {
                            body: hir,
                            ty,
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
            Stmt::LayoutDef { name, attributes, span } => {
                // Layout alias definitions are handled by the resolver.
                // The checker just passes them through.
                Ok(HirStmt::LayoutDef {
                    name: name.clone(),
                    attributes: attributes.clone(),
                    span: *span,
                })
            }
            Stmt::TypeDef { .. } => {
                // Type definitions are already handled by the resolver;
                // no additional checking needed here.
                Ok(HirStmt::Error)
            }
            Stmt::Edition(version, span) => {
                // Edition is validated and stored by the resolver.
                // The checker simply passes it through.
                Ok(HirStmt::Edition(version.clone(), *span))
            }
            Stmt::TraitDef { .. } => {
                // Trait definitions are handled by the resolver; skip silently.
                Ok(HirStmt::Error)
            }
            Stmt::Import {
                path,
                items,
                alias,
                span,
            } => {
                // Imports are already resolved by the NameResolver and registered
                // in the resolution map. The checker just passes them through.
                Ok(HirStmt::Import {
                    path: path.clone(),
                    items: items.clone(),
                    alias: alias.clone(),
                    span: *span,
                })
            }
            Stmt::ExternFunction {
                abi,
                name,
                params,
                return_type,
                span,
                attributes,
            } => {
                let ret_ty = self.resolve_type(return_type)?;
                let mut hir_params = Vec::new();
                for p in params {
                    let p_ty = if let Some(ref ty) = p.ty {
                        self.resolve_type(ty)?
                    } else {
                        self.new_infer_var(TypeVariableKind::Unconstrained)
                    };
                    hir_params.push(HirParam {
                        name: p.name.clone(),
                        ty: p_ty,
                        default: p.default.clone(),
                        span: p.span,
                    });
                }
                Ok(HirStmt::ExternFunction {
                    abi: abi.clone(),
                    name: name.clone(),
                    params: hir_params,
                    return_type: ret_ty,
                    span: *span,
                    attributes: attributes.clone(),
                })
            }
            Stmt::Constraint { name, bounds, span } => {
                let resolved_bounds: Vec<TypeId> = bounds
                    .iter()
                    .map(|b| self.resolve_type(b))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(HirStmt::Constraint {
                    name: name.clone(),
                    bounds: resolved_bounds,
                    span: *span,
                })
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
                    _ => {
                        let msg = format!("check_stmt: expected ImplBlock, got {:?}", stmt);
                        self.diagnostics
                            .push(Diagnostic::error(&msg).with_span(stmt.span()));
                        return Ok(HirStmt::Error);
                    }
                };
                if let Some(tp) = &trait_path {
                    // ── Trait impl block ─────────────────────────────────
                    // Resolve the trait path to get its DefId.
                    // For simple paths like `Show`, look up the trait directly;
                    // for complex types like `Add<Int<32>>`, resolve as a type.
                    let trait_id = match tp.as_ref() {
                        Type::Path(path, _) => {
                            match self.symbols.lookup_trait_by_path(path) {
                                Some(id) => id,
                                None => {
                                    self.diagnostics.push(
                                        Diagnostic::error("trait not found")
                                            .with_code_str("E100")
                                            .with_span(span),
                                    );
                                    return Ok(HirStmt::Error);
                                }
                            }
                        }
                        _ => {
                            let trait_ty = self.resolve_type(tp.as_ref())?;
                            match self.ctx.get_def_id_for_type(trait_ty) {
                                Some(id) => id,
                                None => {
                                    self.diagnostics.push(
                                        Diagnostic::error("trait not found")
                                            .with_code_str("E100")
                                            .with_span(span),
                                    );
                                    return Ok(HirStmt::Error);
                                }
                            }
                        }
                    };
                    let trait_binding = match self.symbols.lookup_trait_by_def_id(trait_id) {
                        Some(b) => b,
                        None => {
                            self.diagnostics.push(
                                Diagnostic::error("trait not found")
                                    .with_code_str("E100")
                                    .with_span(span),
                            );
                            return Ok(HirStmt::Error);
                        }
                    };

                    // Register generic type parameters so `T` in `impl<T> Foo for T` resolves
                    // Collect names before insertion so we can clean up after the impl block
                    // is fully processed, preventing cross-impl cache pollution.
                    let impl_param_names: Vec<Symbol> = type_params.iter().map(|tp| tp.name).collect();
                    for (i, tp) in type_params.iter().enumerate() {
                        let generic_id = self.ctx.generic_param(i, tp.name.clone());
                        self.local_type_param_cache
                            .insert(tp.name.clone(), generic_id);
                    }

                    // Resolve the for_type
                    let for_ty = self.resolve_type(for_type)?;

                    // Check that all required trait methods are provided
                    let auto_deref = attributes.iter().any(|a| a.name.eq_str("auto_deref"));
                    let impl_method_names: HashSet<Symbol> =
                        methods.iter().map(|m| m.name).collect();
                    let self_ty = &for_type;

                    for (tm_name, _tm_sig) in &trait_binding.methods {
                        if !impl_method_names.contains(tm_name) {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "impl missing method `{}` required by trait `{}`",
                                    tm_name,
                                    Self::type_to_string(tp.as_ref()),
                                ))
                                .with_code_str("E101")
                                .with_help("every trait method must be implemented — add a `def` for it in this impl block")
                                .with_span(span)
                                .with_label(trait_binding.span, "required by trait declaration here"));
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
                                    // Bare `self`, `&self`, `&mut self` params: resolve to `for_ty`
                                    Ok(for_ty)
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
                                    .with_code_str("E103")
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
                        resolved_methods: method_infos.clone(),
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
                                .with_code_str("E102")
                                .with_span(span),
                        );
                    }

                    // Also register the resolved methods for method resolution
                    if let TypeData::Adt { def_id, .. } = self.ctx.get(for_ty)
                    {
                        self.trait_env.add_inherent_methods(*def_id, method_infos);
                    }

                    // ── Clean up generic parameter cache for trait impl ──
                    for name in &impl_param_names {
                        self.local_type_param_cache.remove(name);
                    }

                    Ok(HirStmt::ImplBlock {
                        span,
                        attributes: attributes.clone(),
                        trait_path: Some(trait_id),
                        for_type: for_ty,
                        methods: methods.clone(),
                        associated_types: Vec::new(),
                    })
                } else {
                    // Inherent impl block: resolve the type and register methods
                    let for_ty = self.resolve_type(for_type)?;
                    let for_def_id = match self.ctx.get(for_ty) {
                        TypeData::Adt { def_id, .. } => *def_id,
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
                    let auto_deref = attributes.iter().any(|a| a.name.eq_str("auto_deref"));
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
                                    // Bare `self`, `&self`, `&mut self` params: resolve to `for_ty`
                                    Ok(for_ty)
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
                        trait_path: None,
                        for_type: for_ty,
                        methods: methods.clone(),
                        associated_types: Vec::new(),
                    })
                }
            }
            Stmt::Error(span) => Err(Diagnostic::error("invalid statement").with_span(*span)),
            // Stmt::Generate is expanded before name resolution, so it
            // should never reach the checker.  If it does, the pipeline
            // is misconfigured.
            Stmt::Generate { span, .. } => Err(Diagnostic::error("generate block not expanded before type checking")
                .with_span(*span)),
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
            Type::Path(p, s) if p.len() == 1 && (p[0].eq_str("Self") || p[0].eq_str("self")) => {
                self_ty.clone()
            }
            Type::Reference {
                inner,
                mutable,
                span: s,
                ..
            } => Type::Reference {
                inner: Box::new(self.resolve_self_ty(inner, self_ty)),
                mutable: *mutable,
                lifetime: None,
                span: *s,
            },
            Type::Pointer(inner, s) => {
                Type::Pointer(Box::new(self.resolve_self_ty(inner, self_ty)), *s)
            }
            Type::Generic(base, args, span) => {
                let new_base = self.resolve_self_ty(base, self_ty);
                let new_args: Vec<GenericArg> = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Positional(t) => {
                            GenericArg::Positional(self.resolve_self_ty(t, self_ty))
                        }
                        GenericArg::Named(n, t) => {
                            GenericArg::Named(n.clone(), self.resolve_self_ty(t, self_ty))
                        }
                    })
                    .collect();
                Type::Generic(Box::new(new_base), new_args, *span)
            }
            Type::Tuple(tys, span) => Type::Tuple(
                tys.iter()
                    .map(|t| self.resolve_self_ty(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Slice(inner, span) => {
                Type::Slice(Box::new(self.resolve_self_ty(inner, self_ty)), *span)
            }
            Type::Array(inner, size, span) => Type::Array(
                Box::new(self.resolve_self_ty(inner, self_ty)),
                size.clone(),
                *span,
            ),
            Type::DynTrait(traits, span) => Type::DynTrait(
                traits
                    .iter()
                    .map(|t| self.resolve_self_ty(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Function { params, ret, span } => Type::Function {
                params: params
                    .iter()
                    .map(|p| self.resolve_self_ty(p, self_ty))
                    .collect(),
                ret: Box::new(self.resolve_self_ty(ret, self_ty)),
                span: *span,
            },
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name,
                span,
            } => Type::Projection {
                impl_type: Box::new(self.resolve_self_ty(impl_type, self_ty)),
                trait_path: Box::new(self.resolve_self_ty(trait_path, self_ty)),
                assoc_name: assoc_name.clone(),
                span: *span,
            },
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
            TypeData::Adt { kind: _, def_id, args } => Ok((*def_id, args.clone())),
            TypeData::Error => Err(Diagnostic::error("type error").with_span(span)),
            _ => Err(Diagnostic::error("expected struct or enum type").with_span(span)),
        }
    }

    fn resolve_def_id(&self, path: &[Symbol]) -> Result<DefId, Diagnostic> {
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
            .lookup_type(path[0])
            .map(|b| b.def_id)
            .or_else(|| self.symbols.lookup_trait(path[0]).map(|b| b.def_id))
            .ok_or_else(|| {
                Diagnostic::error(format!("'{}' not found", path[0].as_str())).with_span(Span::new(0, 0))
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
                let mut diag = Diagnostic::error(msg).with_code_str("E030").with_span(span);
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
                    .with_code_str("E030")
                    .with_span(span);
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
                .with_code_str("E601")
                .with_span(span)
                .with_suggestion("consider dereferencing first: `*expr as TargetType`")
                .with_suggestion("or use `as!` for an unsafe bitcast"))
            } else {
                Err(
                    Diagnostic::error("safe cast only allowed between numeric and boolean types")
                        .with_code_str("E601")
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
                        .with_code_str("E601")
                        .with_span(span)
                        .with_suggestion("consider using `*expr as usize` via a pointer cast"),
                )
            } else {
                Err(Diagnostic::error("unsafe cast requires compatible types (numeric<->numeric, ref<->ptr, ptr<->ptr)")
                    .with_code_str("E601")
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

    fn check_future_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(future_ty) = self.extract_future_type(ty) {
            Ok(future_ty)
        } else {
            Err(Diagnostic::error("await operator requires Future type").with_span(span))
        }
    }

    fn extract_ok_type(&self, ty: TypeId) -> Option<TypeId> {
        if let TypeData::Adt { kind: _, def_id: did, args } = self.ctx.get(ty) {
            if let Some(result_id) = self.known_def_id(Symbol::intern("Result")) {
                if *did == result_id && args.len() == 2 {
                    return Some(args[0]);
                }
            }
        }
        None
    }

    fn extract_future_type(&mut self, ty: TypeId) -> Option<TypeId> {
        // Use the trait-based associated type projection:
        // resolve `<ty as Future>::Output`
        let future_id = self.known_def_id(Symbol::intern("Future"))?;
        self.trait_env
            .resolve_assoc_type(future_id, ty, "Output", self.ctx, self.symbols)
    }

    fn extract_result_types(&self, ty: TypeId, span: Span) -> Result<(TypeId, TypeId), Diagnostic> {
        if let TypeData::Adt { kind: _, def_id: did, args } = self.ctx.get(ty) {
            if let Some(result_id) = self.known_def_id(Symbol::intern("Result")) {
                if *did == result_id && args.len() == 2 {
                    return Ok((args[0], args[1]));
                }
            }
        }
        Err(Diagnostic::error("catch requires Result type").with_span(span))
    }

    fn known_def_id(&self, name: Symbol) -> Option<DefId> {
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
        self.symbols.lookup_trait(*name).map(|b| b.def_id)
    }

    /// Extract the name from a bound `Type` for constraint alias lookup.
    fn extract_bound_name(bound: &Type) -> Option<Symbol> {
        let base = match bound {
            Type::Path(path, _) => return path.first().copied(),
            Type::Generic(base, _, _) => base.as_ref(),
            _ => return None,
        };
        match base {
            Type::Path(path, _) => path.first().cloned(),
            _ => None,
        }
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
        let deref_trait_id = self.symbols.lookup_trait(Symbol::intern("Deref")).map(|b| b.def_id)?;
        let candidates = self.trait_env.lookup_impls_for_type(ty);
        // Check Deref first
        for cand in &candidates {
            if cand.trait_id == deref_trait_id && cand.has_auto_deref {
                if let Some(target_ty) = cand
                    .assoc_tys
                    .iter()
                    .find(|(name, _)| name.eq_str("Target"))
                    .map(|(_, ty)| *ty)
                {
                    return Some(target_ty);
                }
            }
        }
        // Also try DerefMut: same Target as Deref
        let deref_mut_id = self.symbols.lookup_trait(Symbol::intern("DerefMut")).map(|b| b.def_id);
        if let Some(deref_mut_id) = deref_mut_id {
            for cand in &candidates {
                if cand.trait_id == deref_mut_id && cand.has_auto_deref {
                    if let Some(target_ty) = self
                        .trait_env
                        .lookup_impl(deref_trait_id, ty)
                        .and_then(|dc| {
                            dc.assoc_tys
                                .iter()
                                .find(|(name, _)| name.eq_str("Target"))
                                .map(|(_, ty)| *ty)
                        })
                    {
                        return Some(target_ty);
                    }
                }
            }
        }
        None
    }

    /// Walk the autoderef chain up to MAX_DEREFS steps, yielding each intermediate type.
    fn autoderef_chain<'s>(&'s self, ty: TypeId) -> AutoderefIter<'s> {
        AutoderefIter::with_max_depth(self, ty, DEFAULT_MAX_DEREF_DEPTH)
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
            self.ctx.begin_transaction();
            let unify_ok = self.ctx.unify(substituted_ret, exp_ty).is_ok();
            if !unify_ok {
                self.ctx.rollback_transaction();
                return Ok(None);
            }
            self.ctx.commit_transaction();
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
            TypeData::Adt { args, .. } => {
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
            TypeData::Adt { args, .. } => args
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
            TypeData::Adt { args, .. } => args
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

    fn lookup_field(&mut self, ty: TypeId, name: Symbol, span: Span) -> Result<TypeId, Diagnostic> {
        // Collect field names from all types in the deref chain for error reporting
        let mut all_field_names: Vec<String> = Vec::new();

        // Try direct lookup first
        {
            let data = self.ctx.get(ty);
            let def_id = match data {
                TypeData::Adt { def_id, .. } => Some(*def_id),
                _ => None,
            };
            if let Some(def_id) = def_id {
                let args: &[TypeId] = match data {
                    TypeData::Adt { args, .. } => args.as_slice(),
                    _ => &[],
                };
                let binding = self.symbols.lookup_type_by_def_id(def_id).ok_or_else(|| {
                    Diagnostic::error("struct definition not found").with_span(span)
                })?;
                all_field_names.extend(binding.fields.iter().map(|f| f.name.as_str()));
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

        // Walk autoderef chain, skipping the original type (already tried)
        for deref_ty in self.autoderef_chain(ty).skip(1) {
            let data = self.ctx.get(deref_ty);
            let def_id = match data {
                TypeData::Adt { def_id, .. } => Some(*def_id),
                _ => None,
            };
            if let Some(def_id) = def_id {
                let args: &[TypeId] = match data {
                    TypeData::Adt { args, .. } => args.as_slice(),
                    _ => &[],
                };
                let binding = self.symbols.lookup_type_by_def_id(def_id).ok_or_else(|| {
                    Diagnostic::error("struct definition not found").with_span(span)
                })?;
                all_field_names.extend(binding.fields.iter().map(|f| f.name.as_str()));
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
            .with_code_str("E010")
            .with_span(span);

        // If we found the type definition, show where it was defined
        if let TypeData::Adt { def_id, .. } = self.ctx.get(ty) {
            if let Some(binding) = self.symbols.lookup_type_by_def_id(*def_id) {
                diag = diag.with_label(binding.span, "type defined here");
            }
        }

        if !all_field_names.is_empty() {
            diag =
                diag.with_suggestion(format!("available fields: {}", all_field_names.join(", ")));
            if let Some(suggestion) = did_you_mean_suggestion(&name.as_str(), &all_field_names) {
                diag = diag.with_suggestion(suggestion);
            }
        }

        Err(diag)
    }

    /// Look up a method by name on a type, walking the autoderef chain.
    /// Returns `(param_types, return_type)` if found.
    fn lookup_method(&mut self, ty: TypeId, name: Symbol) -> Option<(Vec<TypeId>, TypeId)> {
        // Collect autoderef chain first to avoid borrow conflicts with self.ctx.
        let chain: Vec<TypeId> = self.autoderef_chain(ty).collect();
        // Pre-collect all unique trait IDs.
        let all_trait_ids: Vec<DefId> = {
            let mut seen = std::collections::HashSet::new();
            self.trait_env
                .all_impls()
                .iter()
                .filter(|c| seen.insert(c.trait_id))
                .map(|c| c.trait_id)
                .collect()
        };

        for current_ty in chain {
            // Check inherent methods first.
            for method in self.trait_env.lookup_inherent_methods(current_ty, self.ctx) {
                if method.name == name {
                    return Some((method.param_tys.clone(), method.ret_ty));
                }
            }

            // Check trait impl methods via exact match.
            for cand in self.trait_env.lookup_impls_for_type(current_ty) {
                for method in &cand.resolved_methods {
                    if method.name == name {
                        return Some((method.param_tys.clone(), method.ret_ty));
                    }
                }
            }

            // Fallback: try generic impl matching for every trait.
            for &trait_id in &all_trait_ids {
                if let Some((cand, subst)) =
                    self.trait_env
                        .lookup_impl_generic(trait_id, current_ty, self.ctx, self.symbols)
                {
                    for method in &cand.resolved_methods {
                        if method.name == name {
                            let param_tys: Vec<TypeId> = method
                                .param_tys
                                .iter()
                                .map(|&p| self.ctx.subst(p, &subst))
                                .collect();
                            let ret_ty = self.ctx.subst(method.ret_ty, &subst);
                            return Some((param_tys, ret_ty));
                        }
                    }
                }
            }
        }
        None
    }

    fn lookup_attr(&self, ty: TypeId, name: Symbol, span: Span) -> Result<TypeId, Diagnostic> {
        if name.eq_str("len") && (self.ctx.is_array(ty) || self.ctx.is_slice(ty) || ty == self.ctx.builtin_str || ty == self.ctx.builtin_str_ref) {
            Ok(self.ctx.usize())
        } else if name.eq_str("size") && (self.ctx.is_integer(ty) || self.ctx.is_float(ty) || self.ctx.is_pointer(ty)) {
            Ok(self.ctx.usize())
        } else if name.eq_str("align") {
            Ok(self.ctx.usize())
        } else if name.eq_str("default") {
            Ok(ty)
        } else {
            Err(Diagnostic::error(format!("unknown attribute '{}'", name)).with_span(span))
        }
    }

    fn lookup_type_default_expr(
        &mut self,
        ty_id: TypeId,
        span: Span,
    ) -> Result<Option<Expr>, Diagnostic> {
        let resolved = self.ctx.resolve_binding(ty_id);
        let def_id = match self.ctx.get(resolved) {
            TypeData::Adt { def_id, .. } => Some(*def_id),
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
        self.block_type_impl(stmts, true)
    }

    /// Whether an implicit trailing expression counts as the block's return type.
    /// Functions (`def`) require explicit `return`; closures and blocks allow
    /// trailing expressions as implicit return values.
    fn block_type_impl(&self, stmts: &[HirStmt], allow_implicit: bool) -> TypeId {
        for stmt in stmts.iter().rev() {
            match stmt {
                HirStmt::ComptimeBlock { ty, .. } => {
                    if *ty != self.ctx.error() {
                        return *ty;
                    }
                }
                HirStmt::Expression(expr) if allow_implicit => {
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
        Ok(self.symbols.lookup_trait(Symbol::intern(trait_name)).map(|b| b.def_id))
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
            TypeData::Adt { def_id, .. } => self.symbols.lookup_type_by_def_id(*def_id).cloned(),
            _ => None,
        }
    }

    /// Convert an AST type to a user-friendly string for diagnostics.
    fn type_to_string(ty: &Type) -> String {
        match ty {
            Type::Path(path, _) => path.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("::"),
            Type::Generic(base, args, _) => {
                let base_str = Self::type_to_string(base);
                let args_str: Vec<String> = args.iter().map(|a| match a {
                    crate::ast::GenericArg::Positional(t) => Self::type_to_string(t),
                    crate::ast::GenericArg::Named(n, t) => format!("{} = {}", n, Self::type_to_string(t)),
                }).collect();
                format!("{}<{}>", base_str, args_str.join(", "))
            }
            Type::Reference { inner, mutable, .. } => {
                if *mutable {
                    format!("&mut {}", Self::type_to_string(inner))
                } else {
                    format!("&{}", Self::type_to_string(inner))
                }
            }
            _ => format!("{:?}", ty),
        }
    }
}

#[cfg(test)]
pub mod tests;
