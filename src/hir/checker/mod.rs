use crate::ast::*;
use crate::diagnostics::{DiagCtxt, Diagnostic, Label};
use crate::hir::hir::*;
use crate::hir::infer::*;
use crate::hir::resolver::ResolutionMap;
use crate::hir::symbol::*;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::project::ProjectionCache;
use crate::hir::traits::solver::select::SelectionContext;
use crate::hir::traits::solver::{
    FulfillmentContext, Obligation, ObligationCause, ObligationCauseCode,
    Predicate as TraitPredicate,
};
use crate::hir::types::*;
use crate::symbol::Symbol;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::mem;
use std::rc::Rc;

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
        self.frames
            .borrow_mut()
            .last_mut()
            .unwrap()
            .insert(name, ty);
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

    /// Check whether a binding exists in the innermost (current) scope frame only.
    /// Returns `true` if the name is bound in the current frame, `false` otherwise.
    /// Unlike `get()`, this does NOT search enclosing scopes, so it correctly
    /// allows shadowing of outer-scope variables.
    pub fn current_frame_contains(&self, name: Symbol) -> bool {
        self.frames
            .borrow()
            .last()
            .map_or(false, |frame| frame.contains_key(&name))
    }

    /// Iterate over all bindings across all scope frames.
    /// Yields each (name, type) pair exactly once (innermost frame wins on duplicates).
    pub fn iter(&self) -> Vec<(Symbol, TypeId)> {
        let frames = self.frames.borrow();
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for frame in frames.iter().rev() {
            for (name, ty) in frame {
                if seen.insert(*name) {
                    result.push((*name, *ty));
                }
            }
        }
        result
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
    span_frames: Rc<RefCell<Vec<HashMap<Symbol, Span>>>>,
}

impl VarScopeGuard {
    fn new(
        frames: Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>>,
        span_frames: Rc<RefCell<Vec<HashMap<Symbol, Span>>>>,
    ) -> Self {
        frames.borrow_mut().push(HashMap::new());
        span_frames.borrow_mut().push(HashMap::new());
        VarScopeGuard {
            frames,
            span_frames,
        }
    }
}

impl Drop for VarScopeGuard {
    fn drop(&mut self) {
        self.frames.borrow_mut().pop();
        self.span_frames.borrow_mut().pop();
    }
}

pub struct TypeChecker<'a> {
    ctx: &'a mut TypeContext,
    symbols: &'a SymbolTable,
    trait_env: &'a mut TraitEnv,
    diagnostics: DiagCtxt,
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
    /// Map of variable name → definition span, for type origin tracing.
    /// Populated alongside `local_variable_types` at variable definition sites.
    /// Used by `resolve_type_origin` to show where a type originates.
    /// Scoped alongside `local_variable_types` via `VarScopeGuard`.
    local_variable_spans: Rc<RefCell<Vec<HashMap<Symbol, Span>>>>,
    /// Pre-resolved by NameResolver: variable name → TypeId
    resolution_map: ResolutionMap,
    /// Local cache of generic type parameter types (e.g. `T` in `def foo<T>(x: T)`).
    /// Populated when processing function definitions with type_params.
    /// Also used by `set auto<T> = expr` to bind captured type names.
    ///
    /// # Scope leak note
    /// `auto<T>` inserts entries that are never removed when the block scope
    /// exits.  This is safe because the **resolver** uses lexical scoping
    /// (a `Scope` stack in `SymbolTable`), so `T` is unresolvable after the
    /// block exits — the checker never runs.  Example:
    /// ```posita
    /// def foo() {
    ///     {
    ///         set auto<T> = 42;  // cache: T → Int<32>
    ///     }                        // resolver: T no longer in scope
    ///     set x: T = 1;            // RESOLVER ERROR — unreachable path
    /// }
    /// ```
    /// Function generic parameters (`def bar<T>(...)`) **do** clean up after
    /// themselves via `local_type_param_cache.remove(name)`, so any stale
    /// `auto<T>` entry is overwritten when a later function declares its own
    /// `T` as a generic parameter.  The leak is real but unexploitable.
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
    /// Registry of builtin trait DefIds for fast lookup during trait resolution.
    builtin_registry: BuiltinTraitRegistry,
    /// Cache for associated type projection normalization.
    proj_cache: ProjectionCache,
    /// Trait obligations accumulated during function body checking (from
    /// `binary_op_type`, `require_type_sized`, and other non-where-clause
    /// sources).  Merged with `caller_bounds` and processed by the new
    /// trait solver in `check_stmt(FunctionDef)`.
    ///
    /// NOTE: This is a transitional field.  Once all trait constraints are
    /// routed through the new solver, `Constraint::Impl` will be removed
    /// from the old solver and this field will become the sole collection
    /// point for trait obligations.
    trait_obligations: Vec<(Span, TraitPredicate)>,
    /// Residual obligations from function bodies that failed before their
    /// solver pass ran.  These are processed at the `check_program` top-level
    /// solver pass, preventing obligation loss when a function body errors
    /// before the `trait_obligations` drain site.
    residual_trait_obligations: Vec<(Span, TraitPredicate)>,
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
            diagnostics: DiagCtxt::new(),
            current_function: None,
            current_return_type: None,
            resolving_aliases: HashSet::new(),
            infer: InferenceContext::new(),
            infer_stack: Vec::new(),
            region_tree: RegionTree::new(),
            local_variable_types: ScopedVarMap::new(),
            local_variable_spans: Rc::new(RefCell::new(vec![HashMap::new()])),
            local_type_param_cache: HashMap::new(),
            resolution_map,
            guarantee_chain: GuaranteeChain::new(),
            mutable_globals: HashSet::new(),
            current_function_trusted: false,
            comptime_fn_registry: HashMap::new(),
            comptime_fn_pass: false,
            deferred_comptime_blocks: Vec::new(),
            builtin_registry: BuiltinTraitRegistry::new(),
            proj_cache: ProjectionCache::new(),
            trait_obligations: Vec::new(),
            residual_trait_obligations: Vec::new(),
        };

        // ── Register builtin trait DefIds ──
        // This populates the BuiltinTraitRegistry so that the trait solver
        // can identify builtin traits (Sized, Copy, Clone, etc.) by their
        // DefId during candidate assembly.  Without this, the solver would
        // never recognize any trait as builtin and would rely solely on
        // user-defined impls.
        for name_str in &[
            "Sized",
            "Copy",
            "Clone",
            "Drop",
            "Default",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Rem",
            "Neg",
            "Eq",
            "Ord",
            "Index",
            "IndexMut",
            "Deref",
            "Display",
            "Serialize",
            "Write",
        ] {
            if let Some(binding) = checker.symbols.lookup_trait(Symbol::intern(name_str)) {
                checker
                    .builtin_registry
                    .register(binding.def_id, &Symbol::intern(name_str));
            }
        }

        // Built-in traits and impls are registered by `register_builtins`
        // inside `NameResolver::new`.  The debug assertion below was removed
        // because it fired in test configurations where the TraitEnv is empty
        // (e.g. unit tests that parse and check without a full resolver).
        // The registration chain is verified by the `check_source` test helper.

        checker
    }

    /// Find the innermost bweak tawget (Woop, Whiwe, Fow, WabewedBwock) (*＾▽＾)／
    /// Wetuwns the tawget's span and optionaw wabew. If `wabew` is Some, onwy match same-named WabewedBwock.
    /// Find the innermost continue tawget (onwy Woop, Whiwe, Fow) ☆ﾟ.*･｡ﾟ
    /// Type-check a parsed program and produce HIR.
    ///
    /// # Errors
    ///
    /// Returns `Err(DiagCtxt)` containing all type errors found
    /// during checking.  The checker continues after each error to collect
    /// as many diagnostics as possible.
    #[must_use]
    pub fn check_program(&mut self, program: &Program) -> Result<HirProgram, DiagCtxt> {
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
        let comptime_fn_indices: Vec<usize> = program
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, stmt)| {
                if let Stmt::FunctionDef {
                    name,
                    params,
                    is_comptime,
                    ..
                } = stmt
                {
                    if *is_comptime {
                        let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
                        self.comptime_fn_registry
                            .insert(*name, (param_names, Vec::new()));
                        Some(i)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

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

        // ── New trait solver: resolve top-level trait obligations ──
        // After all statements are type-checked, drain trait_obligations
        // accumulated from non-function contexts (module-level variable
        // initializers, constant expressions, etc.) and run the new solver.
        // This ensures that binary_op_type and require_type_sized calls
        // outside of function bodies are also verified.
        // Note: Function bodies handle their own solver pass inside
        // check_stmt(FunctionDef), so by the time we reach here, only
        // top-level obligations remain.
        //
        // Save the obligations in a persistent local so that the retry pass
        // (after the old solver resolves inference variables) can reuse them.
        // The first pass drains the vector; the retry pass uses the saved copy.
        let mut top_obligations: Vec<(Span, TraitPredicate)> =
            self.trait_obligations.drain(..).collect();
        // Also process any residual obligations salvaged from failed function bodies.
        top_obligations.extend(self.residual_trait_obligations.drain(..));
        if !top_obligations.is_empty() {
            let ctx: &mut TypeContext = &mut self.ctx;
            let mut selcx = SelectionContext::new(
                ctx,
                self.trait_env,
                self.symbols,
                &self.builtin_registry,
                &self.proj_cache,
                &[], // no caller bounds at top level
            );
            let mut fulfill = FulfillmentContext::new(&mut selcx);
            fulfill.set_infer_data_from(&self.infer);
            for (obl_span, bound) in &top_obligations {
                let obligation = Obligation {
                    cause: crate::hir::traits::solver::ObligationCause {
                        span: *obl_span,
                        code: crate::hir::traits::solver::ObligationCauseCode::Misc,
                    },
                    predicate: match bound {
                        TraitPredicate::Trait {
                            trait_id,
                            self_ty,
                            args,
                        } => crate::hir::traits::solver::Predicate::Trait {
                            trait_id: *trait_id,
                            self_ty: *self_ty,
                            args: args.clone(),
                        },
                        TraitPredicate::Sized { ty } => {
                            crate::hir::traits::solver::Predicate::Sized { ty: *ty }
                        }
                        _ => continue,
                    },
                    recursion_depth: 0,
                };
                fulfill.register_obligation(obligation);
            }
            if let Err(errors) = fulfill.evaluate_all() {
                let mut msgs: Vec<String> = Vec::new();
                for e in &errors {
                    use crate::hir::traits::solver::obligation::SolveError;
                    let (trait_id, self_ty) = match e {
                        SolveError::Ambiguous {
                            trait_id, self_ty, ..
                        }
                        | SolveError::NotFound {
                            trait_id, self_ty, ..
                        } => (*trait_id, *self_ty),
                        _ => continue,
                    };
                    let trait_name = self
                        .symbols
                        .lookup_trait_by_def_id(trait_id)
                        .and_then(|tb| self.symbols.trait_name_by_def_id(trait_id))
                        .map(|s| s.as_str())
                        .unwrap_or_else(|| format!("{:?}", trait_id));
                    let ty = self.ctx.get(self_ty).display_with(self.ctx);
                    msgs.push(format!(
                        "no trait implementation found for `{}` on type `{}`",
                        trait_name, ty
                    ));
                }
                if msgs.is_empty() {
                    let msg = errors
                        .iter()
                        .map(|e| format!("{}", e))
                        .collect::<Vec<_>>()
                        .join("; ");
                    msgs.push(msg);
                }
                let msg = msgs.join("; ");
                let span = errors
                    .first()
                    .and_then(|e| e.span())
                    .unwrap_or(crate::ast::Span::new(0, 0));
                self.diagnostics.push(
                    Diagnostic::error(format!("trait solver error: {}", msg))
                        .with_code_str("E030")
                        .with_span(span),
                );
            }
        }

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
                // ── Retry deferred top-level trait obligations ──
                // After the old solver has resolved all inference variables,
                // run the new solver again to retry any obligations that were
                // deferred due to unresolved infer vars during the first pass.
                // The types are now concrete, so the solver should be able to
                // resolve all remaining obligations.
                //
                // IMPORTANT: use the saved top_obligations, NOT
                // self.trait_obligations — the first pass already drained
                // the vector and the deferred obligations were lost when the
                // transient FulfillmentContext was dropped.
                if !top_obligations.is_empty() {
                    let ctx: &mut TypeContext = &mut self.ctx;
                    let mut selcx = SelectionContext::new(
                        ctx,
                        self.trait_env,
                        self.symbols,
                        &self.builtin_registry,
                        &self.proj_cache,
                        &[],
                    );
                    let mut fulfill = FulfillmentContext::new(&mut selcx);
                    fulfill.set_infer_data_from(&self.infer);
                    for (obl_span, bound) in &top_obligations {
                        let obligation = Obligation {
                            cause: crate::hir::traits::solver::ObligationCause {
                                span: *obl_span,
                                code: crate::hir::traits::solver::ObligationCauseCode::Misc,
                            },
                            predicate: match bound {
                                TraitPredicate::Trait {
                                    trait_id,
                                    self_ty,
                                    args,
                                } => crate::hir::traits::solver::Predicate::Trait {
                                    trait_id: *trait_id,
                                    self_ty: *self_ty,
                                    args: args.clone(),
                                },
                                TraitPredicate::Sized { ty } => {
                                    crate::hir::traits::solver::Predicate::Sized { ty: *ty }
                                }
                                _ => continue,
                            },
                            recursion_depth: 0,
                        };
                        fulfill.register_obligation(obligation);
                    }
                    if let Err(errors) = fulfill.evaluate_all_final() {
                        let msg = format_solve_errors(&self.symbols, &self.ctx, &errors);
                        let span = errors
                            .first()
                            .and_then(|e| e.span())
                            .unwrap_or(crate::ast::Span::new(0, 0));
                        self.diagnostics.push(
                            Diagnostic::error(format!("trait solver error: {}", msg))
                                .with_code_str("E030")
                                .with_span(span),
                        );
                    }
                }
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

                // ── Duplicate variable detection ──
                // Check BEFORE the RHS is evaluated, so the error is reported
                // even if the initializer expression fails type-checking.
                // When a duplicate is detected, subsequent errors from the RHS
                // are aggregated as children of this diagnostic.
                let mut dup_diag: Option<Diagnostic> = None;
                if let Some(var_name) = name {
                    if self
                        .local_variable_types
                        .current_frame_contains(var_name.clone())
                    {
                        let prev_span = self.span_get(var_name).unwrap_or(*span);
                        dup_diag = Some(
                            Diagnostic::error(format!("duplicate definition of `{}`", var_name,))
                                .with_code_str("E019")
                                .with_span(*span)
                                .with_additional_span(prev_span)
                                .with_secondary_label(
                                    prev_span,
                                    "previous definition here",
                                ),
                        );
                    } else if self.local_variable_types.get(var_name.clone()).is_some() {
                        // Shadowing is allowed but warns.
                        let prev_span = self.span_get(var_name).unwrap_or(*span);
                        self.diagnostics.push(
                            Diagnostic::warning(format!("shadowing definition of `{}`", var_name,))
                                .with_code_str("W113")
                                .with_span(*span)
                                .with_additional_span(prev_span)
                                .with_secondary_label(
                                    prev_span,
                                    "previous definition here",
                                ),
                        );
                    }
                }

                // Resolve the declared type, or leave as an inference variable if not provided.
                let declared_ty = if let Some(ty) = ty {
                    self.resolve_type(ty)?
                } else {
                    self.new_infer_var(
                        TypeVariableKind::Unconstrained,
                        crate::hir::infer::VarOrigin::Expression(Some(*span)),
                    )
                };

                // Determine the actual initializer (value) and its type.
                // Wrap in a closure so errors from the RHS can be aggregated
                // into the duplicate definition diagnostic.
                let rhs_result = (|| -> Result<(Option<HirExpr>, TypeId, Option<HirPattern>, Option<Vec<HirStmt>>), Diagnostic> {
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
                    Ok((value_hir, inferred_ty, pattern_hir, else_hir))
                })();
                let (value_hir, inferred_ty, pattern_hir, else_hir) = match rhs_result {
                    Ok(r) => r,
                    Err(rhs_err) => {
                        if let Some(ref mut d) = dup_diag {
                            d.related_errors.push(crate::diagnostics::RelatedError {
                                code: rhs_err.code.clone(),
                                message: rhs_err.message.clone(),
                                span: rhs_err.spans.first(),
                                label: None,
                            });
                        } else {
                            self.diagnostics.push(rhs_err);
                        }
                        (None, self.ctx.error(), None, None)
                    }
                };
                if let Some(ref d) = dup_diag {
                    self.diagnostics.push(d.clone());
                }

                let final_ty = if declared_ty != self.ctx.error() {
                    declared_ty
                } else if let Some(hir) = &value_hir {
                    hir.ty()
                } else {
                    self.ctx.error()
                };

                // Cache the variable's type for subsequent references.
                // If this is a duplicate definition, preserve the original
                // type and span — do NOT overwrite them with the duplicate's
                // values, otherwise resolve_type_origin and "previous
                // definition here" labels would point to the wrong location,
                // and downstream error recovery would see the wrong type.
                if let Some(var_name) = name {
                    if dup_diag.is_none() {
                        self.local_variable_types.insert(var_name.clone(), final_ty);
                        self.span_insert(var_name.clone(), *span);
                    }
                }

                // Track mutable global variables (top-level `set mut`).
                // These require `@trusted` context to be read/written.
                if *mutable && self.current_function.is_none() {
                    if let Some(var_name) = name {
                        self.mutable_globals.insert(var_name.clone());
                    }
                }

                // `set auto<T, N> = expr` — bind captured type names to the inferred type.
                // Each name in `type_captures` becomes available as a type alias in
                // comptime reflection (e.g., `@typeInfo!(T)`).
                for capture in type_captures {
                    self.local_type_param_cache
                        .insert(capture.name.clone(), final_ty);
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
                // ── Salvage per-function trait_obligations ──
                // Each function body starts with a fresh accumulator.
                // If a previous function failed, its stale obligations
                // would leak into this function's trait-solving context,
                // causing spurious errors or silent acceptance of invalid
                // obligations.  Instead of clearing (which would lose
                // obligations from a failed function), salvage them into
                // residual_trait_obligations for processing at the top
                // level by `check_program`.
                let residual: Vec<_> = self.trait_obligations.drain(..).collect();
                if !residual.is_empty() {
                    self.residual_trait_obligations.extend(residual);
                }

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

                // SAFETY: Raw pointers to `symbols` and `ctx` are taken before
                // `ScopeGuard::new(self)` borrows `self` mutably.  While the guard
                // is alive we cannot access `self.symbols` / `self.ctx` through
                // the normal borrow path, but the pointers remain valid because:
                //
                // 1. `ScopeGuard` only stores a `&mut` reference — it does NOT
                //    move or destroy `self`, so the addresses are stable.
                // 2. On the error path (where these pointers are dereferenced) the
                //    guard's `Drop` calls `rollback_transaction()` and
                //    `abort_inference_scope()`, neither of which mutates `symbols`.
                // 3. `ctx` uses `RefCell` internally, which provides runtime
                //    borrow-checking even if accessed through a raw pointer.
                // 4. The dereference happens AFTER `fulfill.evaluate_all()` has
                //    returned, so there is no concurrent access.
                let symbols_ptr = std::ptr::addr_of!(self.symbols);
                let ctx_ptr = std::ptr::addr_of!(self.ctx);

                let guard = ScopeGuard::new(self);
                guard.checker.current_function = Some(DefId(0));
                guard.checker.current_function_trusted =
                    attributes.iter().any(|a| a.name.eq_str("trusted"));

                // Enter inference scope BEFORE creating return_ty so that the
                // return‑type inference variable lives in the fresh context
                // (not the old one pushed onto the infer stack).
                guard.checker.enter_inference_scope();

                let return_ty = if let Some(rt) = return_type {
                    guard.checker.resolve_type(rt)?
                } else {
                    guard
                        .checker
                        .new_infer_var(TypeVariableKind::Any, VarOrigin::Expression(Some(*span)))
                };
                guard.checker.current_return_type = Some(return_ty);

                // ── @interrupt handler checks ─────────────────────────
                let is_interrupt = attributes.iter().any(|a| a.name.eq_str("interrupt"));
                if is_interrupt {
                    // Rule 1: return type must be Never (!)
                    if return_type.is_none() || !guard.checker.ctx.is_never(return_ty) {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("@interrupt handler must return `!` (never type)")
                                .with_code_str("E050")
                                .with_span(*span)
                                .with_help("interrupt handlers must have return type `!` because they never return")
                        );
                    }
                    // Rule 2: no custom parameters
                    if !params.is_empty() {
                        guard.checker.diagnostics.push(
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
                        guard.checker.diagnostics.push(
                            Diagnostic::error("@interrupt handler must satisfy @no_alloc")
                                .with_code_str("E052")
                                .with_span(*span)
                                .with_suggestion("add `@no_alloc` to this function (redundant with `@no_panic`?)")
                        );
                    }
                    if !has_no_panic {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("@interrupt handler must satisfy @no_panic")
                                .with_code_str("E053")
                                .with_span(*span)
                                .with_suggestion("add `@no_panic` to this function"),
                        );
                    }
                    // Rule 4: @interrupt + @alloc is incompatible
                    if attributes.iter().any(|a| a.name.eq_str("alloc")) {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("@interrupt handler cannot have @alloc")
                                .with_code_str("E054")
                                .with_span(*span)
                                .with_help("@interrupt and @alloc are incompatible — interrupt handlers must not allocate")
                        );
                    }
                    // Rule 5: @interrupt + @io is incompatible
                    if attributes.iter().any(|a| a.name.eq_str("io")) {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("@interrupt handler cannot have @io")
                                .with_code_str("E055")
                                .with_span(*span)
                                .with_help("@interrupt and @io are incompatible — interrupt handlers must not perform I/O")
                        );
                    }
                }

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
                    guard.checker.span_insert(p.name.clone(), p.span);
                }
                guard
                    .checker
                    .local_variable_types
                    .insert(Symbol::intern("codomain"), return_ty);
                guard.checker.span_insert(Symbol::intern("codomain"), *span);

                // SCAP: collect ensures conditions into the guarantee chain.
                // Each `ensures` becomes a postcondition that must hold at return.
                for contract in contracts {
                    if let Contract::Ensures { expr, .. } = contract {
                        let expr_labels = extract_labels_from_expr(expr);
                        // Inject each label as a scoped variable with the
                        // return type, so that the expression can reference
                        // `@label` as a placeholder for the return value.
                        for label in &expr_labels {
                            guard.checker.local_variable_types.insert(*label, return_ty);
                            guard.checker.span_insert(*label, *span);
                        }
                        let (_, ensures_ty) = match guard.checker.infer_expr(expr) {
                            Ok(result) => result,
                            Err(diag) => {
                                // ── Collect the error, don't swallow it ──
                                // If the ensures expression fails to type-check
                                // (e.g. a type mismatch in the contract), we
                                // must still report the error rather than silently
                                // defaulting to `bool`.  The checker continues
                                // with a default value so that subsequent errors
                                // in the same function body can also be collected.
                                guard.checker.diagnostics.push(diag);
                                (HirExpr::Error(*span), guard.checker.ctx.bool())
                            }
                        };
                        let g = Guarantee::new_with_expr(
                            Predicate::True,
                            Predicate::Type(ensures_ty),
                            None,
                            Some(Box::new(expr.clone())),
                        );
                        guard.checker.guarantee_chain.push(g);
                    }
                }
                // Generate where-clause constraints as Impl(clause_ty, trait_id)
                // so the solver can verify trait bounds on generic parameters.
                // Also expand constraint aliases (e.g. `where C: SortableContainer`
                // → Impl(C, Container) + Impl(C::Item, Ord) + ...) and collect
                // caller_bounds for the new trait solver.
                //
                // NOTE: This is a transitional dual-solver architecture.
                //   - Old solver: Constraint::Impl (from where-clause bounds AND
                //     function-body trait requirements like method calls).
                //   - New solver: FulfillmentContext with caller_bounds (from
                //     where-clause bounds only).
                //   Both solvers use the same TraitEnv for impl lookup, so they
                //   should produce consistent results.  The long-term plan is to
                //   route ALL trait constraints through the new solver and remove
                //   Constraint::Impl from the old solver.
                //
                // Track‑B (Tuple subjects):
                //   `where (X, Y): Rel` with `constraint Rel<T, U> { T: Foo<U> }`
                //   builds Subst{ 0 → X, 1 → Y }, substitutes every predicate's
                //   subject and bounds, and emits Impl(substituted_subject, trait, span).
                let mut caller_bounds: Vec<TraitPredicate> = Vec::new();
                if let Some(wc) = where_clause {
                    for pred in &wc.predicates {
                        // Resolve the subject(s).  A `Type::Tuple` means
                        // `where (A, B, …): Bound` — resolve each element.
                        let subject_tys: Vec<TypeId> = if let Type::Tuple(elems, _) = &pred.ty {
                            elems
                                .iter()
                                .map(|e| guard.checker.resolve_type(e))
                                .collect::<Result<Vec<_>, _>>()?
                        } else {
                            vec![guard.checker.resolve_type(&pred.ty)?]
                        };

                        for bound in &pred.bounds {
                            // ── Direct trait bound ──────────────────────────
                            if let Some(trait_id) = guard.checker.resolve_trait_path(bound) {
                                if subject_tys.len() > 1 {
                                    // A single trait bound applied to multiple
                                    // types is ambiguous — reject it.
                                    guard.checker.diagnostics.push(
                                        Diagnostic::error(
                                            "a single trait bound cannot be applied \
                                             to multiple types in a tuple subject; \
                                             use separate `where` clauses",
                                        )
                                        .with_code_str("E004")
                                        .with_span(pred.span),
                                    );
                                } else {
                                    // ── Extract trait generic args from the bound ──
                                    // For `T: Add<Int<32>>`, the bound is parsed as
                                    // `Type::Generic(Path(["Add"]), [Positional(Int<32>)])`.
                                    // We extract positional args here and resolve them
                                    // to TypeIds for the new solver's TraitPredicate.
                                    let mut trait_args: Vec<TypeId> = Vec::new();
                                    if let Type::Generic(_, args, _) = bound {
                                        for arg in args {
                                            match arg {
                                                GenericArg::Positional(ty) => {
                                                    match guard.checker.resolve_type(ty) {
                                                        Ok(resolved) => trait_args.push(resolved),
                                                        Err(diag) => {
                                                            guard.checker.diagnostics.push(diag);
                                                        }
                                                    }
                                                }
                                                GenericArg::Named(_, _) => {
                                                    // Handled below as ProjectionEq
                                                }
                                            }
                                        }
                                    }

                                    // Register with the new trait solver via caller_bounds
                                    // (already done above — this is a no-op placeholder).
                                    // The old solver's Constraint::Impl was removed in the
                                    // unified solver migration.
                                    // Also register with new trait solver as caller bound
                                    caller_bounds.push(TraitPredicate::Trait {
                                        trait_id,
                                        self_ty: subject_tys[0],
                                        args: trait_args,
                                    });

                                    // ── Extract associated type constraints (Named args) ──
                                    // Handle `T: Iterator<Item = U>` — the bound is
                                    // parsed as `Type::Generic(Path(["Iterator"]),
                                    // [Named("Item", Path(["U"]))])`.  Each `Named`
                                    // arg is an associated type projection that must
                                    // be resolved.
                                    if let Type::Generic(_, args, _) = bound {
                                        for arg in args {
                                            match arg {
                                                GenericArg::Named(assoc_name, assoc_ty) => {
                                                    // Resolve the associated type value
                                                    match guard.checker.resolve_type(assoc_ty) {
                                                        Ok(assoc_ty_id) => {
                                                            // Register ProjectionEq with old solver
                                                            // Register ProjectionEq with new solver
                                                            caller_bounds.push(
                                                                TraitPredicate::ProjectionEq {
                                                                    trait_id,
                                                                    self_ty: subject_tys[0],
                                                                    assoc_name: *assoc_name,
                                                                    value: assoc_ty_id,
                                                                },
                                                            );
                                                        }
                                                        Err(diag) => {
                                                            guard.checker.diagnostics.push(diag);
                                                        }
                                                    }
                                                }
                                                GenericArg::Positional(_) => {
                                                    // Already handled above in trait_args extraction
                                                }
                                            }
                                        }
                                    }
                                }
                                continue;
                            }

                            // ── Constraint alias ────────────────────────────
                            let Some(name) = TypeChecker::extract_bound_name(bound) else {
                                continue;
                            };

                            let Some(constraint) = guard.checker.symbols.lookup_constraint(name)
                            else {
                                continue;
                            };

                            // Validate arity: tuple-element count must match
                            // the constraint's type-param count.
                            if subject_tys.len() != constraint.params.len() {
                                guard.checker.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "constraint `{}` expects {} type parameter(s), \
                                         but {} {} given",
                                        name,
                                        constraint.params.len(),
                                        subject_tys.len(),
                                        if subject_tys.len() > 1 { "were" } else { "was" },
                                    ))
                                    .with_code_str("E004")
                                    .with_span(pred.span),
                                );
                                continue;
                            }

                            // Build a positional substitution:
                            //   Subst{ 0 → subject_tys[0], 1 → subject_tys[1], … }
                            let mut subst = crate::hir::types::Subst::new();
                            for (i, &ty) in subject_tys.iter().enumerate() {
                                subst.insert(i, ty);
                            }

                            for cp in &constraint.predicates {
                                // Substitute the predicate's subject too, so
                                // that generic-param references in the subject
                                // (or bounds) are replaced by the actual types.
                                let subst_subject = guard.checker.ctx.subst(cp.subject, &subst);
                                for &bound_ty in &cp.bounds {
                                    let substituted = guard.checker.ctx.subst(bound_ty, &subst);
                                    if let Some(trait_id) =
                                        guard.checker.ctx.get_def_id_for_type(substituted)
                                    {
                                        // Also register with new trait solver
                                        caller_bounds.push(TraitPredicate::Trait {
                                            trait_id,
                                            self_ty: subst_subject,
                                            args: vec![],
                                        });
                                    } else {
                                        guard.checker.diagnostics.push(
                                            Diagnostic::warning(format!(
                                                "bound `{:?}` does not resolve \
                                                 to a trait",
                                                bound
                                            ))
                                            .with_span(pred.span),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Filter caller_bounds for the solver ──
                // Where-clause bounds on types that contain generic parameters
                // (e.g., `where T: SomeTrait`) are passed as assumptions to the
                // solver so they can be used as Param candidates for matching
                // obligations on the same type.  Bounds on fully concrete types
                // (e.g., `where i32: SomeTrait`) must NOT be passed as assumptions
                // — the solver would treat them as Param candidates and succeed
                // without verifying that an impl actually exists.  Instead, they
                // are only registered as obligations (via all_bounds below) so
                // the solver must find a real impl to satisfy them.
                let solver_caller_bounds: Vec<TraitPredicate> = {
                    let ctx = &*guard.checker.ctx;
                    caller_bounds
                        .iter()
                        .filter(|b| {
                            let self_ty = match b {
                                TraitPredicate::Trait { self_ty, .. }
                                | TraitPredicate::AutoTrait { self_ty, .. }
                                | TraitPredicate::ProjectionEq { self_ty, .. } => *self_ty,
                                TraitPredicate::ProjectionNormalize { projection, .. } => {
                                    projection.self_ty
                                }
                                // Sized and CopyLike don't encode a where-clause subject
                                _ => return true,
                            };
                            let mut indices = Vec::new();
                            Self::collect_generic_param_indices(self_ty, ctx, &mut indices);
                            !indices.is_empty()
                        })
                        .cloned()
                        .collect()
                };

                let body_result = if let Some(body) = body {
                    let mut stmts = Vec::new();
                    let mut body_err = None;
                    for s in body {
                        match guard.checker.check_stmt(s) {
                            Ok(hir) => stmts.push(hir),
                            Err(e) => {
                                body_err = Some(e);
                                break;
                            }
                        }
                    }
                    match body_err {
                        Some(e) => Err(e),
                        None => Ok(Some(stmts)),
                    }
                } else {
                    Ok(None)
                };

                guard.checker.pop_ctx();

                // ── Defer body error propagation ──
                // The solver pass (below) must run INSIDE the inference scope
                // so that inference variables from the function body are still
                // alive and the solver can resolve trait obligations correctly.
                // If we propagated the body error immediately, the guard would
                // be dropped, the inference scope would be aborted, and any
                // trait obligations pushed during ensures/contract checking
                // (e.g. `Ord` from `ensures @s > 1`) would lose their inference
                // variables — causing false positives like "Ord not found on Int".
                //
                // Instead, we save the body error and run the solver pass first,
                // then propagate the error after.  This is consistent with the
                // OmniML region/level design (omniml/lib/constraint_solver/
                // generalization.ml): inference variables are resolved within
                // their defining region before the region is exited.
                let mut body_hir: Option<Vec<HirStmt>> = None;
                let mut saved_body_err: Option<Diagnostic> = None;
                match body_result {
                    Ok(body) => {
                        body_hir = body;
                    }
                    Err(e) => {
                        saved_body_err = Some(e);
                    }
                }

                // If no explicit return type was written and the body has no
                // return statements, default the inferred return type to Never.
                // This must happen BEFORE the solver runs (inside the inference
                // scope) so the solver doesn't see an unresolved Any-kind
                // infer var and report CannotInfer.
                if return_type.is_none() {
                    if let Some(ref body_stmts) = body_hir {
                        // Recursively check for return statements inside nested
                        // blocks (if, while, for, etc.) — not just top-level.
                        fn has_return_recursive(stmts: &[HirStmt]) -> bool {
                            for s in stmts {
                                match s {
                                    HirStmt::Return { .. } => return true,
                                    HirStmt::If {
                                        then_branch,
                                        else_branch,
                                        ..
                                    } => {
                                        if has_return_recursive(then_branch) {
                                            return true;
                                        }
                                        if let Some(else_stmts) = else_branch {
                                            if has_return_recursive(else_stmts) {
                                                return true;
                                            }
                                        }
                                    }
                                    HirStmt::While { body, .. }
                                    | HirStmt::WhileLet { body, .. }
                                    | HirStmt::For { body, .. }
                                    | HirStmt::Loop { body, .. } => {
                                        if has_return_recursive(body) {
                                            return true;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            false
                        }
                        let has_return = has_return_recursive(body_stmts);
                        if !has_return {
                            let _ = guard
                                .checker
                                .ctx
                                .unify(return_ty, guard.checker.ctx.never());
                        }
                    }
                }

                // ── Validate path labels ──
                // Every label referenced in `ensures @label expr` must appear on
                // at least one `return @label` in the function body.  Labels that
                // appear on return but are never referenced in ensures are allowed
                // (they are simply ignored).  Labels that appear in ensures but
                // never on any return are a compile-time error.
                if let Some(ref body_stmts) = body_hir {
                    // Collect all labels from ensures clauses (extracted from `@identifier`
                    // references in the expression, e.g. `ensures @even % 2 == 0`).
                    let ensures_labels: Vec<Symbol> = contracts
                        .iter()
                        .filter_map(|c| match c {
                            Contract::Ensures { expr, .. } => {
                                let labels = extract_labels_from_expr(expr);
                                if labels.is_empty() {
                                    None
                                } else {
                                    Some(labels)
                                }
                            }
                            _ => None,
                        })
                        .flatten()
                        .collect();
                    if !ensures_labels.is_empty() {
                        // Collect all labels from return statements in the body,
                        // recursively walking nested blocks (if, while, for, etc.).
                        fn collect_return_labels(stmts: &[HirStmt]) -> Vec<Symbol> {
                            let mut labels = Vec::new();
                            for s in stmts {
                                match s {
                                    HirStmt::Return { labels: l, .. } => {
                                        labels.extend(l.iter().copied());
                                    }
                                    HirStmt::If {
                                        then_branch,
                                        else_branch,
                                        ..
                                    } => {
                                        labels.extend(collect_return_labels(then_branch));
                                        if let Some(else_stmts) = else_branch {
                                            labels.extend(collect_return_labels(else_stmts));
                                        }
                                    }
                                    HirStmt::While { body, .. }
                                    | HirStmt::WhileLet { body, .. }
                                    | HirStmt::For { body, .. }
                                    | HirStmt::Loop { body, .. } => {
                                        labels.extend(collect_return_labels(body));
                                    }
                                    _ => {}
                                }
                            }
                            labels
                        }
                        let return_labels = collect_return_labels(body_stmts);
                        // Check: every ensures label must appear on at least one return.
                        for label in &ensures_labels {
                            if !return_labels.contains(label) {
                                let label_str = label.as_str();
                                let label_name = label_str.strip_prefix('@').unwrap_or(&label_str);
                                guard.checker.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "label `@{}` used in `ensures` but never attached to a `return`",
                                        label_name,
                                    ))
                                    .with_code_str("E030")
                                    .with_help("each label in `ensures @label` must have a matching `return @label`")
                                    .with_suggestion(format!(
                                        "add `return @{} <value>` to the function body, or remove `@{}` from the ensures clause",
                                        label_name, label_name,
                                    )),
                                );
                            }
                        }
                        // Check: every return label must have a matching ensures clause.
                        //
                        // TODO: Once reachability analysis (constant propagation +
                        // SMT-based branch evaluation) is available, reduce this to
                        // a warning for provably-unreachable return paths.  Currently
                        // we conservatively error on all unlabeled returns, even if
                        // the branch condition is statically determined (e.g.
                        // `if true { return @s x; } else { return @r y; }` where
                        // the else branch is dead code).
                        for label in &return_labels {
                            if !ensures_labels.contains(label) {
                                let label_str = label.as_str();
                                let label_name = label_str.strip_prefix('@').unwrap_or(&label_str);
                                guard.checker.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "label `@{}` attached to a `return` but never referenced in an `ensures` clause",
                                        label_name,
                                    ))
                                    .with_code_str("E030")
                                    .with_help("each `return @label` must have a matching `ensures @label` clause")
                                    .with_suggestion(format!(
                                        "add `ensures @{} <property>` to the function's contracts, or remove `@{}` from the return statement",
                                        label_name, label_name,
                                    )),
                                );
                            }
                        }
                    }
                }

                // ── New trait solver: resolve all trait obligations ──
                // After the function body is fully checked, run the new
                // FulfillmentContext to verify that all trait constraints
                // (where-clause bounds, binary ops, Sized checks, etc.)
                // are satisfied.  This runs INSIDE the inference scope so
                // that any unification from trait matching is captured by
                // the transaction and rolled back on failure.
                //
                // IMPORTANT: caller_bounds (from where-clause) are passed
                // as the SelectionContext's caller_bounds for candidate
                // matching.  trait_obligations (from binary_op_type,
                // require_type_sized) are registered as obligations but
                // NOT passed as caller_bounds, because they would match
                // themselves as Param candidates and cause ambiguity.
                let trait_obs: Vec<(Span, TraitPredicate)> =
                    guard.checker.trait_obligations.drain(..).collect();
                // Save all obligations for potential retry after guard.commit().
                let all_bounds: Vec<(Span, TraitPredicate)> = {
                    let mut bounds: Vec<(Span, TraitPredicate)> =
                        caller_bounds.iter().map(|b| (*span, b.clone())).collect();
                    bounds.extend(trait_obs.clone());
                    bounds
                };
                let has_obligations = !all_bounds.is_empty();
                if has_obligations {
                    // We need separate borrows of ctx for the solver.
                    let ctx: &mut TypeContext = guard.checker.ctx;
                    let mut selcx = SelectionContext::new(
                        ctx,
                        guard.checker.trait_env,
                        guard.checker.symbols,
                        &guard.checker.builtin_registry,
                        &guard.checker.proj_cache,
                        &solver_caller_bounds, // only where-clause bounds on generic-param types as assumptions
                    );
                    let mut fulfill = FulfillmentContext::new(&mut selcx);
                    // Pass inference variable data for the defaulting step.
                    fulfill.set_infer_data_from(&guard.checker.infer);
                    // Register ALL obligations (where-clause + body-check-time)
                    for (obl_span, bound) in &all_bounds {
                        let obligation = Obligation {
                            cause: crate::hir::traits::solver::ObligationCause {
                                span: *obl_span,
                                code:
                                    crate::hir::traits::solver::ObligationCauseCode::WhereClause {
                                        span: *obl_span,
                                    },
                            },
                            predicate: match bound {
                                TraitPredicate::Trait {
                                    trait_id,
                                    self_ty,
                                    args,
                                } => crate::hir::traits::solver::Predicate::Trait {
                                    trait_id: *trait_id,
                                    self_ty: *self_ty,
                                    args: args.clone(),
                                },
                                TraitPredicate::ProjectionEq {
                                    trait_id,
                                    self_ty,
                                    assoc_name,
                                    value,
                                } => crate::hir::traits::solver::Predicate::ProjectionEq {
                                    trait_id: *trait_id,
                                    self_ty: *self_ty,
                                    assoc_name: *assoc_name,
                                    value: *value,
                                },
                                TraitPredicate::AutoTrait { trait_id, self_ty } => {
                                    crate::hir::traits::solver::Predicate::AutoTrait {
                                        trait_id: *trait_id,
                                        self_ty: *self_ty,
                                    }
                                }
                                TraitPredicate::Sized { ty } => {
                                    crate::hir::traits::solver::Predicate::Sized { ty: *ty }
                                }
                                TraitPredicate::ProjectionNormalize { projection, target } => {
                                    crate::hir::traits::solver::Predicate::ProjectionNormalize {
                                        projection: crate::hir::traits::solver::ProjectionTy {
                                            trait_id: projection.trait_id,
                                            self_ty: projection.self_ty,
                                            args: projection.args.clone(),
                                            assoc_name: projection.assoc_name,
                                        },
                                        target: *target,
                                    }
                                }
                                TraitPredicate::CopyLike { kind, ty } => {
                                    crate::hir::traits::solver::Predicate::CopyLike {
                                        kind: match kind {
                                            crate::hir::traits::solver::CopyKind::Copy => {
                                                crate::hir::traits::solver::CopyKind::Copy
                                            }
                                            crate::hir::traits::solver::CopyKind::Clone => {
                                                crate::hir::traits::solver::CopyKind::Clone
                                            }
                                        },
                                        ty: *ty,
                                    }
                                }
                                TraitPredicate::Eq { a, b } => {
                                    crate::hir::traits::solver::Predicate::Eq { a: *a, b: *b }
                                }
                                TraitPredicate::Sub { sub, sup } => {
                                    crate::hir::traits::solver::Predicate::Sub {
                                        sub: *sub,
                                        sup: *sup,
                                    }
                                }
                                TraitPredicate::Match {
                                    scrutinee,
                                    branches_id,
                                } => crate::hir::traits::solver::Predicate::Match {
                                    scrutinee: *scrutinee,
                                    branches_id: *branches_id,
                                },
                                TraitPredicate::Forall { body } => {
                                    crate::hir::traits::solver::Predicate::Forall {
                                        body: body.clone(),
                                    }
                                }
                                TraitPredicate::Exists { body } => {
                                    crate::hir::traits::solver::Predicate::Exists {
                                        body: body.clone(),
                                    }
                                }
                                TraitPredicate::Instance {
                                    scheme_ty,
                                    instantiation_ty,
                                } => crate::hir::traits::solver::Predicate::Instance {
                                    scheme_ty: *scheme_ty,
                                    instantiation_ty: *instantiation_ty,
                                },
                                TraitPredicate::Let { def, body } => {
                                    crate::hir::traits::solver::Predicate::Let {
                                        def: def.clone(),
                                        body: body.clone(),
                                    }
                                }
                            },
                            recursion_depth: 0,
                        };
                        fulfill.register_obligation(obligation);
                    }
                    if let Err(errors) = fulfill.evaluate_all() {
                        // Abort: unresolved trait obligations must fail the check.
                        // SAFETY: `symbols_ptr` and `ctx_ptr` were taken before the
                        // guard was created (see the safety comment at the declaration
                        // site).  The dereference is safe because:
                        // - `guard` Drop does not mutate `symbols`.
                        // - `ctx` uses `RefCell` for interior mutability.
                        // - `evaluate_all()` has already returned, no concurrent access.
                        let symbols = unsafe { &*symbols_ptr };
                        let ctx = unsafe { &*ctx_ptr };
                        let msg = format_solve_errors(symbols, ctx, &errors);
                        let err_span = errors.first().and_then(|e| e.span()).unwrap_or(*span);
                        return Err(Diagnostic::error(format!("trait solver error: {}", msg))
                            .with_code_str("E030")
                            .with_span(err_span));
                    }
                }

                // ── Propagate saved body error ──
                // If the function body failed, we must abort the inference scope
                // (via guard drop) rather than committing it, because the body's
                // inference results are partial/incomplete.  The solver pass has
                // already run inside the inference scope, so trait obligations
                // from ensures/contracts were resolved correctly before the
                // inference variables were lost.
                if let Some(body_err) = saved_body_err {
                    return Err(body_err);
                }

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
                    if return_type.is_some() {
                        // User wrote an explicit return type — check body against it.
                        let body_ty = self.block_type_impl(body_stmts, false);
                        self.unify_with(return_ty, body_ty, *span, TypingContext::ReturnValue)?;
                    }
                    // When return_type is None, the infer var was already unified
                    // with return values during body checking (via current_return_type),
                    // or defaulted to Never before the solver ran (see above).
                }

                // Contract verification skeleton: check that requires/ensures are bool,
                // and decreases/terminates are integer types.
                for contract in contracts {
                    match contract {
                        Contract::Requires(expr, cspan) | Contract::Invariant(expr, cspan) => {
                            let (_, ty) = self.infer_expr(expr)?;
                            // Regex types cannot appear in contracts (SYNTAX.md).
                            if self.ctx.contains_regex(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "Regex types cannot appear in contracts; they are for runtime use only",
                                    )
                                    .with_code_str("E030")
                                    .with_span(*cspan)
                                    .with_help("use a boolean predicate instead of a Regex type")
                                    .with_suggestion("replace the Regex type with a compatible boolean expression"),
                                );
                            } else if !self.ctx.is_bool(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error("contract condition must be boolean")
                                        .with_code_str("E020")
                                        .with_span(*cspan)
                                        .with_label(
                                            expr.span(),
                                            format!("got {:?}", self.ctx.get(ty)),
                                        ),
                                );
                            }
                        }
                        Contract::Ensures {
                            expr, span: cspan, ..
                        } => {
                            let (_, ty) = self.infer_expr(expr)?;
                            // Regex types cannot appear in contracts (SYNTAX.md).
                            if self.ctx.contains_regex(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "Regex types cannot appear in contracts; they are for runtime use only",
                                    )
                                    .with_code_str("E030")
                                    .with_span(*cspan)
                                    .with_help("use a boolean predicate instead of a Regex type")
                                    .with_suggestion("replace the Regex type with a compatible boolean expression"),
                                );
                            } else if !self.ctx.is_bool(ty) {
                                self.diagnostics.push(
                                    Diagnostic::error("ensures clause must be boolean")
                                        .with_code_str("E020")
                                        .with_span(*cspan)
                                        .with_label(
                                            expr.span(),
                                            format!("got {:?}", self.ctx.get(ty)),
                                        ),
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

                // ── Retry deferred trait obligations ──
                // After the old solver has resolved all inference variables,
                // and after contract expressions have been checked, drain any
                // remaining trait_obligations and run the new solver one final
                // time.  This catches obligations from both the function body
                // and contract expressions (requires, ensures, etc.) that were
                // deferred due to unresolved infer vars during the first pass.
                let final_obs: Vec<(Span, TraitPredicate)> =
                    self.trait_obligations.drain(..).collect();
                if !final_obs.is_empty() {
                    let ctx: &mut TypeContext = &mut self.ctx;
                    let mut selcx = SelectionContext::new(
                        ctx,
                        self.trait_env,
                        self.symbols,
                        &self.builtin_registry,
                        &self.proj_cache,
                        &solver_caller_bounds,
                    );
                    let mut fulfill = FulfillmentContext::new(&mut selcx);
                    // Pass inference variable data for the defaulting step.
                    fulfill.set_infer_data_from(&self.infer);
                    // Collect all obligations: original all_bounds (which includes
                    // where-clause bounds and body-check-time obligations) plus
                    // any new ones from contract expressions.
                    let all_final: Vec<&(Span, TraitPredicate)> =
                        all_bounds.iter().chain(final_obs.iter()).collect();
                    for (obl_span, bound) in all_final {
                        let obligation = Obligation {
                            cause: crate::hir::traits::solver::ObligationCause {
                                span: *obl_span,
                                code:
                                    crate::hir::traits::solver::ObligationCauseCode::WhereClause {
                                        span: *obl_span,
                                    },
                            },
                            predicate: match bound {
                                TraitPredicate::Trait {
                                    trait_id,
                                    self_ty,
                                    args,
                                } => crate::hir::traits::solver::Predicate::Trait {
                                    trait_id: *trait_id,
                                    self_ty: *self_ty,
                                    args: args.clone(),
                                },
                                TraitPredicate::ProjectionEq {
                                    trait_id,
                                    self_ty,
                                    assoc_name,
                                    value,
                                } => crate::hir::traits::solver::Predicate::ProjectionEq {
                                    trait_id: *trait_id,
                                    self_ty: *self_ty,
                                    assoc_name: *assoc_name,
                                    value: *value,
                                },
                                TraitPredicate::AutoTrait { trait_id, self_ty } => {
                                    crate::hir::traits::solver::Predicate::AutoTrait {
                                        trait_id: *trait_id,
                                        self_ty: *self_ty,
                                    }
                                }
                                TraitPredicate::Sized { ty } => {
                                    crate::hir::traits::solver::Predicate::Sized { ty: *ty }
                                }
                                TraitPredicate::ProjectionNormalize { projection, target } => {
                                    crate::hir::traits::solver::Predicate::ProjectionNormalize {
                                        projection: crate::hir::traits::solver::ProjectionTy {
                                            trait_id: projection.trait_id,
                                            self_ty: projection.self_ty,
                                            args: projection.args.clone(),
                                            assoc_name: projection.assoc_name,
                                        },
                                        target: *target,
                                    }
                                }
                                TraitPredicate::CopyLike { kind, ty } => {
                                    crate::hir::traits::solver::Predicate::CopyLike {
                                        kind: match kind {
                                            crate::hir::traits::solver::CopyKind::Copy => {
                                                crate::hir::traits::solver::CopyKind::Copy
                                            }
                                            crate::hir::traits::solver::CopyKind::Clone => {
                                                crate::hir::traits::solver::CopyKind::Clone
                                            }
                                        },
                                        ty: *ty,
                                    }
                                }
                                TraitPredicate::Eq { a, b } => {
                                    crate::hir::traits::solver::Predicate::Eq { a: *a, b: *b }
                                }
                                TraitPredicate::Sub { sub, sup } => {
                                    crate::hir::traits::solver::Predicate::Sub {
                                        sub: *sub,
                                        sup: *sup,
                                    }
                                }
                                TraitPredicate::Match {
                                    scrutinee,
                                    branches_id,
                                } => crate::hir::traits::solver::Predicate::Match {
                                    scrutinee: *scrutinee,
                                    branches_id: *branches_id,
                                },
                                TraitPredicate::Forall { body } => {
                                    crate::hir::traits::solver::Predicate::Forall {
                                        body: body.clone(),
                                    }
                                }
                                TraitPredicate::Exists { body } => {
                                    crate::hir::traits::solver::Predicate::Exists {
                                        body: body.clone(),
                                    }
                                }
                                TraitPredicate::Instance {
                                    scheme_ty,
                                    instantiation_ty,
                                } => crate::hir::traits::solver::Predicate::Instance {
                                    scheme_ty: *scheme_ty,
                                    instantiation_ty: *instantiation_ty,
                                },
                                TraitPredicate::Let { def, body } => {
                                    crate::hir::traits::solver::Predicate::Let {
                                        def: def.clone(),
                                        body: body.clone(),
                                    }
                                }
                            },
                            recursion_depth: 0,
                        };
                        fulfill.register_obligation(obligation);
                    }
                    if let Err(errors) = fulfill.evaluate_all_final() {
                        // SAFETY: `symbols_ptr` and `ctx_ptr` were taken before the
                        // guard was created (see the safety comment at the declaration
                        // site).  The dereference is safe because:
                        // - `guard` Drop does not mutate `symbols`.
                        // - `ctx` uses `RefCell` for interior mutability.
                        // - `evaluate_all_final()` has already returned.
                        let symbols = unsafe { &*symbols_ptr };
                        let ctx = unsafe { &*ctx_ptr };
                        let msg = format_solve_errors(symbols, ctx, &errors);
                        let err_span = errors.first().and_then(|e| e.span()).unwrap_or(*span);
                        return Err(Diagnostic::error(format!("trait solver error: {}", msg))
                            .with_code_str("E030")
                            .with_span(err_span));
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
                        self.comptime_fn_registry
                            .insert(name.clone(), (param_names, body.clone()));
                    }
                }

                // Patch the resolver's placeholder return type (unit()) with the
                // actual inferred/concrete type so that cross‑function call sites
                // see the correct return type rather than the stale placeholder.
                // Using Cell<TypeId> allows mutation through the shared &SymbolTable
                // reference that the checker holds.
                self.symbols.update_function_return_type(*name, return_ty);

                Ok(HirStmt::FunctionDef {
                    span: *span,
                    attributes: attributes.clone(),
                    contracts: contracts.clone(),
                    doc: None,
                    name: name.clone(),
                    params: hir_params,
                    return_type: Some(return_ty),
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
            Stmt::Return {
                value,
                labels,
                span,
            } => {
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
                    // ── Semantic-equivalence fast path ──
                    // If the ensures expression is structurally equivalent to
                    // the return expression (after normalization + simplification),
                    // we can skip the SMT check.  This handles trivial cases like
                    // `ensures codomain == x + x` with `return x + x`.
                    //
                    // Only applies when the return has a value and the guarantee
                    // carries an AST expression.
                    if let Some(ref ast_expr) = g.ast_expr {
                        if let Some(return_value) = value {
                            let fast_path_ok = try_fast_path(ast_expr, return_value);
                            if fast_path_ok {
                                // Fast path succeeded — guarantee is trivially satisfied.
                                // Skip the SMT check entirely.
                            } else {
                                // Fast path failed — fall through to the type check below.
                                let _ = fast_path_ok;
                            }
                        }
                    }

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
                            labels: labels.clone(),
                            span: *span,
                        })
                    } else {
                        let (hir, _) = self.infer_expr(value)?;
                        Ok(HirStmt::Return {
                            value: Some(Box::new(hir)),
                            labels: labels.clone(),
                            span: *span,
                        })
                    }
                } else {
                    if let Some(ret_ty) = self.current_return_type {
                        if self.ctx.is_infer_var(ret_ty) {
                            // Infer var — unify with unit
                            let _ = self.unify(ret_ty, self.ctx.unit(), *span);
                        } else if !self.ctx.is_unit(ret_ty) && !self.ctx.is_never(ret_ty) {
                            self.diagnostics.push(
                                Diagnostic::error("return without value in non-unit function")
                                    .with_span(*span),
                            );
                        }
                    }
                    Ok(HirStmt::Return {
                        value: None,
                        labels: labels.clone(),
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
                    let result_ty =
                        self.binary_op_type(*op, target_ty, target_ty, None, None, *span)?;
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
                        let ty = hir
                            .last()
                            .and_then(|s| match s {
                                HirStmt::Expression(e) => Some(e.ty()),
                                _ => None,
                            })
                            .unwrap_or_else(|| self.ctx.unit());
                        if self.comptime_fn_pass {
                            // During Pass 2 (comptime function body checking), defer
                            // evaluation so that forward references to comptime functions
                            // defined later in the source are available at evaluation time.
                            // After Pass 2 completes, all deferred blocks are evaluated.
                            self.deferred_comptime_blocks.push((hir.clone(), ty, *span));
                        } else {
                            // Evaluate the comptime block at compile time.
                            let mut eval = crate::hir::comptime::ComptimeEvalContext::new(
                                self.ctx,
                                self.symbols,
                            );
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
            Stmt::LayoutDef {
                name,
                attributes,
                span,
            } => {
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
                        self.new_infer_var(
                            TypeVariableKind::Unconstrained,
                            crate::hir::infer::VarOrigin::Expression(Some(p.span)),
                        )
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
            Stmt::Constraint {
                name,
                params,
                predicates,
                span,
            } => {
                let resolved_bounds: Vec<TypeId> = predicates
                    .iter()
                    .flat_map(|p| {
                        let subject = self.resolve_type(&p.ty);
                        let mut bs: Vec<TypeId> = p
                            .bounds
                            .iter()
                            .map(|b| self.resolve_type(b))
                            .collect::<Result<_, _>>()
                            .unwrap_or_default();
                        if subject.is_ok() {
                            bs.insert(0, subject.unwrap());
                        }
                        bs
                    })
                    .collect();
                Ok(HirStmt::Constraint {
                    name: *name,
                    bounds: resolved_bounds,
                    span: *span,
                })
            }
            Stmt::ImplBlock { .. } => {
                let (trait_path, for_type, methods, span, attributes, type_params, where_clause) =
                    match stmt {
                        Stmt::ImplBlock {
                            span,
                            trait_path,
                            for_type,
                            methods,
                            attributes,
                            type_params,
                            where_clause,
                            ..
                        } => (
                            trait_path,
                            for_type,
                            methods,
                            *span,
                            attributes,
                            type_params,
                            where_clause,
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
                            // Use lookup_trait_by_path (path-based) instead of
                            // scope-based lookup_trait, so that qualified paths
                            // like `std::ops::Add` are resolved correctly.
                            // The old code was changed to scope-based lookup
                            // to avoid builtin trait interference, but the
                            // path-based lookup is correct for trait impls.
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
                    let impl_param_names: Vec<Symbol> =
                        type_params.iter().map(|tp| tp.name).collect();
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
                            // Add where-clause predicate types as context.
                            // Each predicate's subject type (e.g. `T` in `where T: Foo`)
                            // must be present so the Coverage condition can verify that
                            // every bare type variable in the head type appears in at
                            // least one context type.
                            if let Some(wc) = where_clause {
                                for pred in &wc.predicates {
                                    match self.resolve_type(&pred.ty) {
                                        Ok(resolved) => ctx_tys.push(resolved),
                                        Err(diag) => {
                                            self.diagnostics.push(diag);
                                        }
                                    }
                                }
                            }
                            // Add type params that have bounds to context.
                            // `impl<T: Bar>` implicitly constrains T.
                            for (i, tp) in type_params.iter().enumerate() {
                                if !tp.bounds.is_empty() {
                                    let param_id = self.ctx.generic_param(i, tp.name.clone());
                                    ctx_tys.push(param_id);
                                }
                            }
                            ctx_tys
                        },
                        arity: type_params.len(),
                        trait_args: {
                            // ── Resolve trait generic args from the trait_path ──
                            // For `impl Add<Int<32>> for MyType`, trait_path is
                            // `Type::Generic(Path(["Add"]), [Positional(Int<32>)])`.
                            // Extract positional args and resolve them to TypeIds.
                            let mut args = Vec::new();
                            if let Some(tp) = &trait_path {
                                if let Type::Generic(_, generic_args, _) = tp.as_ref() {
                                    for arg in generic_args {
                                        if let GenericArg::Positional(ty) = arg {
                                            match self.resolve_type(ty) {
                                                Ok(resolved) => args.push(resolved),
                                                Err(diag) => {
                                                    self.diagnostics.push(diag);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            args
                        },
                        where_clause_bounds: {
                            // Populate where-clause bounds for sub-obligation generation.
                            // Each bound `T: Foo` becomes (T, Foo, args) in where_clause_bounds.
                            let mut bounds = Vec::new();
                            // Extract from where clause predicates
                            if let Some(wc) = where_clause {
                                for pred in &wc.predicates {
                                    if let Ok(subject_ty) = self.resolve_type(&pred.ty) {
                                        for bound in &pred.bounds {
                                            if let Some(trait_id) = self.resolve_trait_path(bound) {
                                                let mut bound_args = Vec::new();
                                                if let Type::Generic(_, args, _) = bound {
                                                    for arg in args {
                                                        if let GenericArg::Positional(ty) = arg {
                                                            if let Ok(resolved) =
                                                                self.resolve_type(ty)
                                                            {
                                                                bound_args.push(resolved);
                                                            }
                                                        }
                                                    }
                                                }
                                                bounds.push((subject_ty, trait_id, bound_args));
                                            }
                                        }
                                    }
                                }
                            }
                            // Extract from type param bounds (e.g. `impl<T: Clone>`)
                            for (i, tp) in type_params.iter().enumerate() {
                                if !tp.bounds.is_empty() {
                                    let param_id = self.ctx.generic_param(i, tp.name.clone());
                                    for bound in &tp.bounds {
                                        if let Some(trait_id) = self.resolve_trait_path(bound) {
                                            let mut bound_args = Vec::new();
                                            if let Type::Generic(_, args, _) = bound {
                                                for arg in args {
                                                    if let GenericArg::Positional(ty) = arg {
                                                        if let Ok(resolved) = self.resolve_type(ty)
                                                        {
                                                            bound_args.push(resolved);
                                                        }
                                                    }
                                                }
                                            }
                                            bounds.push((param_id, trait_id, bound_args));
                                        }
                                    }
                                }
                            }
                            bounds
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
                    } else {
                        // Clear projection cache — new impl may change
                        // normalization results for associated types.
                        self.proj_cache.clear();
                    }

                    // Also register the resolved methods for method resolution
                    if let TypeData::Adt { def_id, .. } = self.ctx.get(for_ty) {
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
            Stmt::Generate { span, .. } => Err(Diagnostic::error(
                "generate block not expanded before type checking",
            )
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
            TypeData::Adt {
                kind: _,
                def_id,
                args,
            } => Ok((*def_id, args.clone())),
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
                Diagnostic::error(format!("'{}' not found", path[0].as_str()))
                    .with_span(Span::new(0, 0))
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
                let expected_str = self.ctx.get(expected).display_with(self.ctx);
                let actual_str = self.ctx.get(actual).display_with(self.ctx);
                let msg = match ctx {
                    TypingContext::ReturnValue => {
                        format!(
                            "return value type mismatch: expected {}, found {}",
                            expected_str, actual_str,
                        )
                    }
                    TypingContext::StructFieldInit => {
                        format!(
                            "field initializer type mismatch: expected {}, found {}",
                            expected_str, actual_str,
                        )
                    }
                    TypingContext::Condition => {
                        format!("condition must be boolean, got {}", actual_str)
                    }
                    TypingContext::Argument { index, total } => {
                        format!(
                            "argument {} of {} has wrong type: expected {}, found {}",
                            index + 1,
                            total,
                            expected_str,
                            actual_str,
                        )
                    }
                    TypingContext::ClosureBody => {
                        format!(
                            "closure body type mismatch: expected {}, found {}",
                            expected_str, actual_str,
                        )
                    }
                    TypingContext::None => {
                        format!(
                            "type mismatch: expected {}, found {}",
                            expected_str, actual_str,
                        )
                    }
                    TypingContext::Index => {
                        format!("index must be an integer, got {}", actual_str)
                    }
                };
                let mut diag = match ctx {
                    TypingContext::ReturnValue => {
                        Diagnostic::error(msg).with_code_str("E036").with_span(span)
                    }
                    TypingContext::Argument { .. } => {
                        Diagnostic::error(msg).with_code_str("E037").with_span(span)
                    }
                    TypingContext::Condition => {
                        Diagnostic::error(msg).with_code_str("E038").with_span(span)
                    }
                    TypingContext::Index => {
                        Diagnostic::error(msg).with_code_str("E039").with_span(span)
                    }
                    _ => Diagnostic::error(msg).with_code_str("E030").with_span(span),
                };
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
        left_span: Option<Span>,
        right_span: Option<Span>,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        // Logical And/Or are NOT trait-routed in the desugaring table.
        // They stay as hard-coded bool operators.
        if matches!(op, BinOp::And | BinOp::Or) {
            let ok =
                self.ctx.is_bool(left) || matches!(self.ctx.get(left), TypeData::InferVar { .. });
            if !ok {
                return Err(
                    Diagnostic::error("logical operators require bool operands").with_span(span)
                );
            }
            // Check kind compatibility early so that e.g. `true and infer_var(Integer)`
            // produces "type mismatch: expected integer type, found Bool" at the operator
            // site rather than a confusing unification failure later.
            self.check_kind_compat(left, left_span, right, right_span, span)?;
            self.check_kind_compat(right, right_span, left, left_span, span)?;
            self.unify_with(left, right, span, TypingContext::None)?;
            return Ok(self.ctx.bool());
        }

        // Overflow-suffixed operators (+%, +?, +!, -%, etc.) are compiler
        // intrinsics — not overloadable via traits (§Spec: Operator Desugaring).
        // They require integer types.
        if matches!(
            op,
            BinOp::AddWrap
                | BinOp::SubWrap
                | BinOp::MulWrap
                | BinOp::AddSaturate
                | BinOp::SubSaturate
                | BinOp::MulSaturate
                | BinOp::AddTrap
                | BinOp::SubTrap
                | BinOp::MulTrap
        ) {
            let is_int = self.ctx.is_integer(left)
                || matches!(self.ctx.get(left), TypeData::InferVar { .. });
            if !is_int {
                return Err(Diagnostic::error(
                    "overflow-suffixed operators require integer operands",
                )
                .with_span(span));
            }
            // Check kind compatibility early, before unify, so that
            // e.g. `infer_var(Float) +% 1` produces "expected integer type, found Float"
            // at the operator site rather than a confusing error later.
            self.check_kind_compat(left, left_span, right, right_span, span)?;
            self.check_kind_compat(right, right_span, left, left_span, span)?;
            self.unify_with(left, right, span, TypingContext::None)?;
            return Ok(left);
        }

        // Trait-routed operators: check kind compatibility early so that
        // e.g. `1 + "hello"` produces a clear diagnostic at the operator
        // site rather than a confusing inference failure later.
        self.check_kind_compat(left, left_span, right, right_span, span)?;
        self.check_kind_compat(right, right_span, left, left_span, span)?;

        // All other operators route through traits (§Spec: Operator Desugaring).
        let Some(trait_id) = self.get_trait_id_for_binop(op, span)? else {
            return Err(Diagnostic::error("operator not supported via traits").with_span(span));
        };

        self.trait_obligations.push((
            span,
            TraitPredicate::Trait {
                trait_id,
                self_ty: left,
                args: vec![],
            },
        ));
        self.trait_obligations.push((
            span,
            TraitPredicate::Trait {
                trait_id,
                self_ty: right,
                args: vec![],
            },
        ));

        // Comparison operators return bool.
        // Unify operands so that inference variables are resolved before
        // the trait solver processes the obligation.  Without this, an
        // infer var from a literal (e.g. `0` in `b != 0`) would remain
        // unresolved and the trait obligation would be deferred forever.
        if matches!(
            op,
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
        ) {
            self.unify_with(left, right, span, TypingContext::None)?;
            return Ok(self.ctx.bool());
        }

        // Arithmetic/bitwise: unify operands, create infer var for result.
        // The Impl constraint verifies the trait exists; the infer var is
        // unified with the expected result type downstream.
        self.unify_with(left, right, span, TypingContext::None)?;
        let result_ty = self.new_infer_var(
            TypeVariableKind::Numeric,
            crate::hir::infer::VarOrigin::Expression(Some(span)),
        );
        // Resolve the result type to the operand type.  Without this, the
        // result infer var would never be resolved to a concrete type, and
        // any comparison operator that follows (e.g. `>=` in `a + b >= 0`)
        // would receive two infer vars and fail to resolve either.
        self.unify_with(left, result_ty, span, TypingContext::None)?;
        Ok(result_ty)
    }

    /// Trace a type back to its origin variable, if any.
    /// Scans `local_variable_types` for a variable whose resolved type matches
    /// `ty`, then returns the variable's definition span from
    /// `local_variable_spans`.  Returns `None` if the type doesn't match any
    /// tracked variable (e.g. it's a literal type or a function result).
    fn resolve_type_origin(&self, ty: TypeId) -> Option<(Symbol, Span)> {
        let resolved = self.ctx.resolve_binding(ty);
        // Never match the error sentinel type — it's not a real type and
        // would cause "type originates here" labels on every cascaded error.
        if matches!(self.ctx.get(resolved), TypeData::Error) {
            return None;
        }
        for (sym, var_ty) in self.local_variable_types.iter() {
            if self.ctx.resolve_binding(var_ty) == resolved {
                if let Some(def_span) = self.span_get(&sym) {
                    return Some((sym, def_span));
                }
            }
        }
        None
    }

    /// Look up a variable's definition span in the scoped `local_variable_spans`
    /// stack, searching from innermost to outermost frame.
    fn span_get(&self, name: &Symbol) -> Option<Span> {
        let frames = self.local_variable_spans.borrow();
        for frame in frames.iter().rev() {
            if let Some(&span) = frame.get(name) {
                return Some(span);
            }
        }
        None
    }

    /// Insert a variable's definition span into the innermost frame of
    /// the scoped `local_variable_spans` stack.
    fn span_insert(&self, name: Symbol, span: Span) {
        self.local_variable_spans
            .borrow_mut()
            .last_mut()
            .unwrap()
            .insert(name, span);
    }

    /// Check that an inference variable's kind constraint is compatible with the
    /// resolved type of another type.  This prevents situations like
    /// `true` (InferVar with kind Bool) being unified with `Int<32>`.
    /// Only fires when the other side resolves to a concrete (non-type-variable) type.
    fn check_kind_compat(
        &self,
        maybe_var: TypeId,
        maybe_var_span: Option<Span>,
        other: TypeId,
        other_span: Option<Span>,
        span: Span,
    ) -> Result<(), Diagnostic> {
        if let TypeData::InferVar { id } = self.ctx.get(maybe_var) {
            let kind = self.infer.get_var_kind(*id);
            let resolved_other = self.ctx.resolve_binding(other);
            // Only check when the other side is a concrete type (not a type variable).
            // Type variables (InferVar, GenericParam, SkolemVar) are placeholders
            // that can be unified with any compatible type.
            // Also skip the error sentinel type — cascading errors from a
            // previously failed expression add no useful information.
            match self.ctx.get(resolved_other) {
                TypeData::InferVar { .. }
                | TypeData::GenericParam { .. }
                | TypeData::SkolemVar { .. }
                | TypeData::Error => return Ok(()),
                _ => {}
            }
            let other_type_str = if matches!(self.ctx.get(resolved_other), TypeData::Error) {
                "no suitable type exists".to_string()
            } else {
                format!("{}", self.ctx.get(resolved_other).display_with(self.ctx))
            };
            let mut diag = match kind {
                Some(TypeVariableKind::Bool) => {
                    if !self.ctx.is_bool(resolved_other) {
                        Some(
                            Diagnostic::error(format!(
                                "type mismatch: expected `Bool`, found `{}`",
                                other_type_str,
                            ))
                            .with_code_str("E031")
                            .with_span(span),
                        )
                    } else {
                        None
                    }
                }
                Some(TypeVariableKind::Integer) => {
                    if !self.ctx.is_integer(resolved_other)
                        && !matches!(self.ctx.get(resolved_other), TypeData::Rational { .. })
                    {
                        Some(
                            Diagnostic::error(format!(
                                "type mismatch: expected integer type, found `{}`",
                                other_type_str,
                            ))
                            .with_code_str("E031")
                            .with_span(span),
                        )
                    } else {
                        None
                    }
                }
                Some(TypeVariableKind::Float) => {
                    if !self.ctx.is_float(resolved_other) {
                        Some(
                            Diagnostic::error(format!(
                                "type mismatch: expected float type, found `{}`",
                                other_type_str,
                            ))
                            .with_code_str("E031")
                            .with_span(span),
                        )
                    } else {
                        None
                    }
                }
                Some(TypeVariableKind::Numeric) => {
                    if !self.ctx.is_numeric(resolved_other) {
                        Some(
                            Diagnostic::error(format!(
                                "type mismatch: expected numeric type, found `{}`",
                                other_type_str,
                            ))
                            .with_code_str("E031")
                            .with_span(span),
                        )
                    } else {
                        None
                    }
                }
                // Any / Unconstrained are compatible with everything
                Some(TypeVariableKind::Any) | Some(TypeVariableKind::Unconstrained) | None => None,
            };
            if let Some(ref mut d) = diag {
                // Add a secondary label for the "other" operand (the concrete type).
                if let Some(os) = other_span {
                    d.labels.push(Label::secondary(os, other_type_str));
                }
                // Add a note label for the "maybe_var" operand (the infer var).
                if let Some(ms) = maybe_var_span {
                    d.labels.push(Label::secondary(ms, "expected integer type"));
                }
                // Trace the type origin: if the "other" operand's type came from
                // a variable definition, show where it originated.
                if let Some((_origin_name, origin_span)) = self.resolve_type_origin(resolved_other)
                {
                    if origin_span != other_span.unwrap_or(origin_span) {
                        d.labels
                            .push(Label::secondary(origin_span, "type originates here"));
                    }
                    // If the type is a string reference (&Str / &[Byte]),
                    // suggest that the programmer might have meant a numeric literal.
                    // Place this note at the origin span (the definition site),
                    // right after the "type originates here" label.
                    if matches!(self.ctx.get(resolved_other), TypeData::Ref { .. }) {
                        let inner = match self.ctx.get(resolved_other) {
                            TypeData::Ref { ty, .. } => self.ctx.get(*ty),
                            _ => &TypeData::Error,
                        };
                        if matches!(inner, TypeData::Adt { def_id, .. } if *def_id == DefId(usize::MAX))
                            || matches!(inner, TypeData::Byte)
                        {
                            d.labels.push(Label::help(
                                origin_span,
                                "this value is a string, not a number. \
                                     Remove the quotes to use it as a numeric literal.",
                            ));
                        }
                    }
                }
                return Err(std::mem::replace(d, Diagnostic::error("placeholder")));
            }
        }
        Ok(())
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
                // Register with the new trait solver.  The new solver handles
                // Sized via Predicate::Sized, which triggers the builtin Sized
                // check in candidate assembly.  If the type is still an infer var,
                // the obligation is deferred and retried after the old solver runs.
                self.trait_obligations
                    .push((span, TraitPredicate::Sized { ty }));
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
        if let TypeData::Adt {
            kind: _,
            def_id: did,
            args,
        } = self.ctx.get(ty)
        {
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
        if let TypeData::Adt {
            kind: _,
            def_id: did,
            args,
        } = self.ctx.get(ty)
        {
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
        let path = match bound {
            Type::Path(path, _) => path,
            Type::Generic(base, ..) => match base.as_ref() {
                Type::Path(path, _) => path,
                _ => return None,
            },
            _ => return None,
        };
        self.symbols.lookup_trait_by_path(path)
    }

    /// Extract the name from a bound `Type` for constraint alias lookup.
    fn extract_bound_name(bound: &Type) -> Option<Symbol> {
        let base = match bound {
            Type::Path(path, _) => return path.last().copied(),
            Type::Generic(base, _, _) => base.as_ref(),
            _ => return None,
        };
        match base {
            Type::Path(path, _) => path.last().cloned(),
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
        let deref_trait_id = self
            .symbols
            .lookup_trait(Symbol::intern("Deref"))
            .map(|b| b.def_id)?;
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
        let deref_mut_id = self
            .symbols
            .lookup_trait(Symbol::intern("DerefMut"))
            .map(|b| b.def_id);
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
            let var = self.new_infer_var(
                TypeVariableKind::Any,
                crate::hir::infer::VarOrigin::GenericParam,
            );
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
        if name.eq_str("len")
            && (self.ctx.is_array(ty)
                || self.ctx.is_slice(ty)
                || ty == self.ctx.builtin_str
                || ty == self.ctx.builtin_str_ref)
        {
            Ok(self.ctx.usize())
        } else if name.eq_str("size")
            && (self.ctx.is_integer(ty) || self.ctx.is_float(ty) || self.ctx.is_pointer(ty))
        {
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
                | HirStmt::Continue { .. }
                | HirStmt::Loop { .. } => return self.ctx.never(),
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
            // Eq/Neq both desugar to Eq::eq (§Spec: Operator Desugaring)
            BinOp::Eq | BinOp::Neq => "Eq",
            // Lt/Gt/Le/Ge all desugar to Ord methods (§Spec: Operator Desugaring)
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => "Ord",
            // And/Or are NOT trait-routed — handled directly by binary_op_type
            BinOp::And | BinOp::Or => return Ok(None),
            // Overflow-suffixed operators are compiler intrinsics, not overloadable
            _ => {
                return Err(
                    Diagnostic::error("overflow operators not yet supported via traits")
                        .with_span(span),
                );
            }
        };
        Ok(self
            .symbols
            .lookup_trait(Symbol::intern(trait_name))
            .map(|b| b.def_id))
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

    fn new_infer_var(
        &mut self,
        kind: TypeVariableKind,
        origin: crate::hir::infer::VarOrigin,
    ) -> TypeId {
        self.infer.new_type_var(self.ctx, kind, origin)
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
            Type::Path(path, _) => path
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("::"),
            Type::Generic(base, args, _) => {
                let base_str = Self::type_to_string(base);
                let args_str: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        crate::ast::GenericArg::Positional(t) => Self::type_to_string(t),
                        crate::ast::GenericArg::Named(n, t) => {
                            format!("{} = {}", n, Self::type_to_string(t))
                        }
                    })
                    .collect();
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

// ── Label extraction helpers ────────────────────────────────────
// Extract `@identifier` labels from AST expressions.  These are
// `Expr::Ident` with `@`-prefixed names, used in `ensures @label expr`
// as placeholders for the return value on specific paths.

fn extract_labels_from_expr(e: &Expr) -> Vec<Symbol> {
    let mut labels = Vec::new();
    match e {
        Expr::Ident(name, _) if name.as_str().starts_with('@') => {
            labels.push(*name);
        }
        Expr::BinaryOp { left, right, .. } => {
            labels.extend(extract_labels_from_expr(left));
            labels.extend(extract_labels_from_expr(right));
        }
        Expr::UnaryOp { expr, .. } => {
            labels.extend(extract_labels_from_expr(expr));
        }
        Expr::Call { callee, args, .. } => {
            labels.extend(extract_labels_from_expr(callee));
            for arg in args {
                labels.extend(extract_labels_from_expr(arg));
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            labels.extend(extract_labels_from_expr(cond));
            for stmt in then_branch {
                labels.extend(extract_labels_from_stmt(stmt));
            }
            if let Some(stmts) = else_branch {
                for stmt in stmts {
                    labels.extend(extract_labels_from_stmt(stmt));
                }
            }
        }
        _ => {}
    }
    labels
}

fn extract_labels_from_stmt(s: &Stmt) -> Vec<Symbol> {
    match s {
        Stmt::Expression(e) => extract_labels_from_expr(e),
        Stmt::Return { value: Some(v), .. } => extract_labels_from_expr(v),
        _ => Vec::new(),
    }
}

// ── Semantic-equivalence fast path ─────────────────────────────
// Try to prove that the return expression satisfies the ensures
// clause WITHOUT calling Z3, using algebraic simplification and
// structural comparison.

// ── Semantic-equivalence fast path ─────────────────────────────
// Try to prove that the return expression satisfies the ensures
// clause WITHOUT calling Z3, using algebraic simplification and
// structural comparison.

/// Format a list of `SolveError` into a human-readable error message.
/// Resolves `DefId` to trait names and `TypeId` to type names.
fn format_solve_errors(
    symbols: &crate::hir::symbol::SymbolTable,
    ctx: &crate::hir::types::TypeContext,
    errors: &[crate::hir::traits::solver::obligation::SolveError],
) -> String {
    use crate::hir::traits::solver::obligation::SolveError;
    let mut msgs: Vec<String> = Vec::new();
    for e in errors {
        let (trait_id, self_ty) = match e {
            SolveError::Ambiguous {
                trait_id, self_ty, ..
            }
            | SolveError::NotFound {
                trait_id, self_ty, ..
            } => (*trait_id, *self_ty),
            _ => continue,
        };
        let trait_name = symbols
            .trait_name_by_def_id(trait_id)
            .map(|s| s.as_str())
            .unwrap_or_else(|| format!("trait#{}", trait_id.0));
        let resolved = ctx.resolve_binding(self_ty);
        let type_tag = if matches!(resolved.tag(), crate::hir::types::TypeTag::InferVar) {
            "unknown type".to_string()
        } else {
            format!("{:?}", resolved.tag())
        };
        msgs.push(format!(
            "no trait implementation found for `{}` on type `{}`",
            trait_name, type_tag
        ));
    }
    if msgs.is_empty() {
        errors
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join("; ")
    } else {
        msgs.join("; ")
    }
}

/// Try the fast path: check if `return_value` satisfies `ensures_expr`.
///
/// Strategy: replace `codomain` (and any `@label`) in the ensures
/// expression with the return value, then check if the result is
/// semantically equivalent to `true`.
fn try_fast_path(ensures_expr: &Expr, return_value: &Expr) -> bool {
    false
}

/// Replace the `codomain` identifier (and any `@label` identifiers)
/// in an expression with the return value expression.
/// Used by the SMT-based contract verification path.
#[allow(dead_code)]
fn replace_codomain(expr: &Expr, replacement: &Expr) -> Expr {
    match expr {
        Expr::Ident(name, _) if name.as_str() == "codomain" || name.as_str().starts_with('@') => {
            replacement.clone()
        }
        Expr::BinaryOp {
            left,
            op,
            right,
            span,
        } => Expr::BinaryOp {
            left: Box::new(replace_codomain(left, replacement)),
            op: *op,
            right: Box::new(replace_codomain(right, replacement)),
            span: *span,
        },
        Expr::UnaryOp {
            op,
            expr: inner,
            span,
        } => Expr::UnaryOp {
            op: *op,
            expr: Box::new(replace_codomain(inner, replacement)),
            span: *span,
        },
        Expr::Call {
            callee,
            args,
            comptime,
            span,
        } => Expr::Call {
            callee: Box::new(replace_codomain(callee, replacement)),
            args: args
                .iter()
                .map(|a| replace_codomain(a, replacement))
                .collect(),
            comptime: *comptime,
            span: *span,
        },
        Expr::FieldAccess { base, field, span } => Expr::FieldAccess {
            base: Box::new(replace_codomain(base, replacement)),
            field: *field,
            span: *span,
        },
        Expr::Index { base, index, span } => Expr::Index {
            base: Box::new(replace_codomain(base, replacement)),
            index: Box::new(replace_codomain(index, replacement)),
            span: *span,
        },
        _ => expr.clone(),
    }
}

#[cfg(test)]
pub mod tests;
