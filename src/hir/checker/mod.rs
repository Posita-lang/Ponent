use crate::ast::*;
use crate::diagnostics::{
    Applicability, ComptimeReason, DiagCtxt, Diagnostic, DiagnosticKind, Label, Suggestion,
    SuggestionStyle, TypeCtx,
};
use crate::hir::comptime::value::ComptimeValue;
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
use std::cell::Cell;
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
    /// An `isolate` bwock — no access to extewnaw mutabwe state. (｀・ω・´)
    Isolate,
}

/// A fwame howding the context kind and its span (*/ω＼*)
#[derive(Debug, Clone)]
pub struct CtxFrame {
    pub kind: CtxKind,
    span: Span,
    /// Optionaw wabew name (onwy used by WabewedBwock)
    label: Option<String>,
    /// Why this block is comptime (if applicable).
    comptime_reason: Option<ComptimeReason>,
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
    ghost_frames: Rc<RefCell<Vec<HashSet<Symbol>>>>,
    runtime_frames: Rc<RefCell<Vec<HashSet<Symbol>>>>,
}

impl VarScopeGuard {
    fn new(
        frames: Rc<RefCell<Vec<HashMap<Symbol, TypeId>>>>,
        span_frames: Rc<RefCell<Vec<HashMap<Symbol, Span>>>>,
        ghost_frames: Rc<RefCell<Vec<HashSet<Symbol>>>>,
        runtime_frames: Rc<RefCell<Vec<HashSet<Symbol>>>>,
    ) -> Self {
        frames.borrow_mut().push(HashMap::new());
        span_frames.borrow_mut().push(HashMap::new());
        ghost_frames.borrow_mut().push(HashSet::new());
        runtime_frames.borrow_mut().push(HashSet::new());
        VarScopeGuard {
            frames,
            span_frames,
            ghost_frames,
            runtime_frames,
        }
    }
}

impl Drop for VarScopeGuard {
    fn drop(&mut self) {
        self.frames.borrow_mut().pop();
        self.span_frames.borrow_mut().pop();
        self.ghost_frames.borrow_mut().pop();
        self.runtime_frames.borrow_mut().pop();
    }
}

pub struct TypeChecker<'a> {
    ctx: &'a mut TypeContext,
    symbols: &'a SymbolTable,
    trait_env: &'a mut TraitEnv,
    diagnostics: DiagCtxt,
    /// Source text for converting byte offsets to line:column in tracebacks.
    source: Option<&'a str>,
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
    /// Names of ghost variables (compile-time-only, from `ghost set`).
    /// Maintained as a scope stack, pushed/popped with `VarScopeGuard`.
    /// Searched top-to-bottom in `contains_runtime_ident` so that ghost
    /// declarations in an inner scope don't leak outward.
    ghost_var_scopes: Rc<RefCell<Vec<HashSet<Symbol>>>>,
    /// Variable names that carry `@must_handle` obligations (assigned from
    /// a call to a `@must_handle` function).  Checked by `Expr::Try` and
    /// `Expr::Catch` to prevent bypassing `@must_handle` by storing the
    /// result in a variable before using `?` or `catch`.
    must_handle_sources: RefCell<HashSet<Symbol>>,
    /// Names of RUNTIME variables (`set x = ...`), kept in a scope stack
    /// parallel to `ghost_var_scopes`.  `contains_runtime_ident` checks the
    /// INNERMOST binding first: a runtime variable shadowing an outer ghost
    /// variable must be treated as runtime (rejected), not as ghost.
    runtime_var_scopes: Rc<RefCell<Vec<HashSet<Symbol>>>>,
    /// True while checking the inner statement of a `ghost set ...` so that
    /// its name is registered as ghost (not runtime) in `VariableDef`.
    in_ghost_var_def: Cell<bool>,
    /// Functions that access mutable globals (by DefId).
    /// Populated during body checking; used to enforce isolate block restrictions.
    functions_accessing_mutables: HashSet<DefId>,
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
    /// Each entry is (captures, body_hir, ty, span).
    deferred_comptime_blocks: Vec<(Vec<(Symbol, Span)>, Vec<HirStmt>, TypeId, Span)>,
    /// Per-arm exist skolem mapping for the current GADT arm.
    /// Populated during pattern checking and consumed by
    /// `apply_gadt_refinement` so that both use the SAME skolem
    /// TypeIds for existentially quantified type variables.
    /// Cleared after `pop_gadt_arm()`.
    ///
    /// A STACK of per-variant scopes: each existential variant pushes its
    /// OWN scope, indexed by the binder's position in `exists_params`
    /// (name → index is a fixed per-variant table; the SKOLEM IDENTITY is
    /// the index, not the name — GHC `realUnique` / OCaml `id: int`).
    /// Same-named binders in different variants map to independent
    /// indices and can never conflate.
    /// (Moved into `TypeContext::gadt` — see `GadtContext`.)
    /// Compile-time constant values for immutable `set` variable declarations
    /// with literal initializers.  Used by `comptime [x] { ... }` capture lists.
    ///
    /// This is a scope-aware value stack: each variable maps to a stack of
    /// values pushed in nested scopes.  On scope exit, the top value is popped,
    /// restoring the outer scope's value.  The companion `scope_var_stack`
    /// tracks which variables were defined in each scope frame so they can be
    /// correctly popped.
    ///
    /// # Invariants
    /// - `scope_var_stack.len()` equals the current nesting depth managed by
    ///   `enter_var_scope()` / `VarScopeGuard`.
    /// - For every `Symbol` in each inner `Vec<Symbol>` of `scope_var_stack`,
    ///   `literal_values[Symbol]` has at least `scope_var_stack.len() - frame_index`
    ///   entries (enough to pop back to the outer scope).
    literal_values: HashMap<Symbol, Vec<ComptimeValue>>,
    /// Scope stack for literal values: each entry is the list of variable names
    /// that were defined (and thus pushed onto `literal_values`) in that scope.
    /// Popped in lockstep with `enter_var_scope()` / `VarScopeGuard`.
    scope_var_stack: Vec<Vec<Symbol>>,
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
    /// Whether the compiler is running in strict mode.
    /// In strict mode, all `@trusted` functions must have `@link_proof` or
    /// `@comptime_test` evidence; otherwise, compilation fails.
    strict_mode: bool,
    /// Whether experimental features are enabled.
    /// When false, items marked @experimental cause a compile error.
    enable_experimental: bool,
    /// Conditional compilation features from `--feature xxx`.
    pub(crate) features: Vec<String>,
    /// Whether debug mode is enabled (`@cfg(debug)` = true).
    pub(crate) debug: bool,
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

/// RAII guard that pops the GADT arm registry on drop if any pushes
/// remain (detected via `TypeContext::gadt_arm_depth`).  Saves the depth
/// at creation time and pops until depth ≤ saved value on drop, correctly
/// handling nested GADT matches.
///
/// The guard takes ownership of the current `current_gadt_exist_skolems`
/// via `std::mem::take` (caller empties the field before creating the
/// guard).  On early return the guard's `Drop` drops the skolems to
/// `None`, preventing stale skolem reuse.  On the happy path the caller
/// assigns back (or leaves `None`).
///
/// # Why raw pointer (unsafe)?
///
/// A `&'a TypeContext` reference would conflict with the enclosing
/// method's `&mut self` borrow, because `self.ctx: &'a mut TypeContext`
/// cannot be re-borrowed as shared while the method borrows `self`
/// mutably for other operations.  A raw pointer bypasses this, and is
/// sound because the guard is always a local variable created and
/// consumed within a single method invocation — it never outlives the
/// `TypeContext` it points to.
/// One existential scope frame: the variant it was created for (its NAME is
/// the frame identity — `check_pattern_inner` reuses the frame that
/// `precreate_exist_skolems` pushed for the SAME top-level variant, so the
/// payload type and `apply_gadt_refinement` share one witness set) and the
/// `ExistScopeFrame` and `PendingInnerGadtEq` now live in
/// `crate::hir::types` (alongside `GadtContext`, which owns the GADT
/// state formerly scattered across these two structs).

#[must_use = "the guard is an RAII scope guard — it must be a method-local \
              variable; storing it in a struct field would break the raw \
              pointer lifetime invariant"]
pub(crate) struct GadtArmGuard {
    /// Pointer to the `GadtContext` whose arm depth / fact registry this
    /// guard manages.  Raw pointer because a `&GadtContext` reference
    /// would conflict with the enclosing method's `&mut self` borrow (see
    /// struct doc).  Sound because the guard is always a method-local
    /// variable that never outlives the `TypeContext` owning the
    /// `GadtContext`.
    gadt: *const crate::hir::types::GadtContext,
    /// Existential-frame stack depth at arm entry: on drop the stack is
    /// truncated back to this depth, preserving OUTER arms' witnesses while
    /// discarding this arm's own frames.
    saved_exist_depth: usize,
    saved_depth: usize,
    /// The region id entered by `enter_region` (the level *before*
    /// entering).  The `Drop` impl restores it via the `infer_raw` raw
    /// pointer — even on early return, the region is popped (RAII).
    pub(crate) prev_region: crate::hir::infer::InferRegionId,
    /// Raw pointer to the `InferenceContext` so the `Drop` impl can restore
    /// the TcLevel region (via `exit_level`) on ALL paths, including early
    /// returns.  SAFETY: the guard is a method-local variable that never
    /// outlives the `TypeChecker` owning the `InferenceContext` (same
    /// lifetime discipline as the `gadt` field).
    infer_raw: *mut crate::hir::infer::InferenceContext,
    /// Whether the region has already been restored by an explicit call to
    /// `restore_region()` (the happy path).  When `true`, the `Drop` impl
    /// skips the region restore to avoid a double-restore bug.
    region_restored: bool,
}

impl GadtArmGuard {
    /// Create a guard that also enters a fresh TcLevel region.  The region
    /// is restored on drop (even on early return), so InferVars created in
    /// the arm body are at a deeper level and escaping them is caught by
    /// the TcLevel escape check in `unify_internal_impl`.
    pub fn enter_region(
        ctx: &TypeContext,
        infer: &mut crate::hir::infer::InferenceContext,
        saved_depth: usize,
        saved_exist_depth: usize,
    ) -> Self {
        let prev_region = infer.enter_level();
        GadtArmGuard {
            gadt: &ctx.gadt as *const crate::hir::types::GadtContext,
            saved_depth,
            saved_exist_depth,
            prev_region,
            infer_raw: infer as *mut crate::hir::infer::InferenceContext,
            region_restored: false,
        }
    }

    /// Restore the TcLevel region entered by `enter_region` — called on the
    /// happy path when the arm processing succeeds.  Sets the
    /// `region_restored` flag so the `Drop` impl does not restore again.
    pub fn restore_region(&mut self) {
        if !self.infer_raw.is_null() && !self.region_restored {
            let infer = unsafe { &mut *self.infer_raw };
            infer.exit_level(self.prev_region);
            self.region_restored = true;
        }
    }
}

/// RAII: restore the existential-frame stack to the pre-push depth if the
/// arm's precreate is not consumed (an error before the caller's own
/// truncation).  The success path calls `commit()` — the frame stays for
/// the arm's own guard to manage.
pub(crate) struct PrecreateGuard {
    ctx: *const crate::hir::types::GadtContext,
    depth: usize,
    committed: bool,
}

impl PrecreateGuard {
    pub fn enter(ctx: &TypeContext, depth: usize) -> Self {
        PrecreateGuard {
            ctx: &ctx.gadt as *const crate::hir::types::GadtContext,
            depth,
            committed: false,
        }
    }

    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for PrecreateGuard {
    fn drop(&mut self) {
        if !self.committed {
            // SAFETY: the `GadtContext` outlives this method-local guard
            // (same lifetime argument as `GadtArmGuard`'s pointer).
            unsafe { &*self.ctx }
                .exist_skolems
                .borrow_mut()
                .truncate(self.depth);
        }
    }
}

impl Drop for GadtArmGuard {
    fn drop(&mut self) {
        // SAFETY: `self.gadt` is always non-null — set by `enter_region`
        // from a `&GadtContext` reference (which is never null).
        debug_assert!(!self.gadt.is_null());
        // Pop GADT arms back to the saved depth — the arm's refinements
        // are discarded.  (SAFETY: `self.gadt` points to the `GadtContext`
        // owned by the `TypeContext` that outlives this guard — the guard
        // is a method-local variable that never outlives its creator.  The
        // raw pointer is necessary because a `&GadtContext` reference would
        // conflict with the enclosing method's `&mut self` borrow.)
        let gadt = unsafe { &*self.gadt };
        while gadt.arm_depth.get() > self.saved_depth {
            gadt.exit_arm();
        }
        // Restore the existential-frame stack to the pre-arm depth —
        // OUTER arms' witnesses survive nested arms; this arm's own
        // frames are discarded.  (SAFETY: the same lifetime argument as
        // the `gadt` deref above — the guard never outlives its creator.
        // The write goes through the `RefCell` interior mutability, the
        // GadtContext convention alongside `facts`/`arm_depth`.)
        // Restore the TcLevel region entered by `enter_region` — even on
        // early return (`?`), the region must be popped to keep the
        // inference context consistent (SAFETY: the same lifetime
        // argument as the `gadt` deref above — the guard never outlives
        // its creator).  Only restore if the happy path hasn't already
        // done so (via `restore_region()`).
        if !self.infer_raw.is_null() && !self.region_restored {
            let infer = unsafe { &mut *self.infer_raw };
            infer.exit_level(self.prev_region);
        }
        let gadt = unsafe { &*self.gadt };
        let current_len = gadt.exist_skolems.borrow().len();
        // Depth-discipline violations are caught in debug builds; in
        // release we SATURATE the truncation to the current length so a
        // stray push cannot silently no-op — and a panic on the unwind
        // path cannot abort the compiler process.
        debug_assert!(
            self.saved_exist_depth <= current_len,
            "GadtArmGuard: saved_exist_depth {} exceeds stack len {} — \
             a frame was pushed without this guard's depth discipline",
            self.saved_exist_depth,
            current_len,
        );
        gadt.exist_skolems
            .borrow_mut()
            .truncate(self.saved_exist_depth.min(current_len));
    }
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        symbols: &'a SymbolTable,
        trait_env: &'a mut TraitEnv,
        resolution_map: ResolutionMap,
        strict_mode: bool,
        enable_experimental: bool,
        features: Vec<String>,
        debug: bool,
    ) -> Self {
        Self::new_with_source(
            ctx,
            symbols,
            trait_env,
            resolution_map,
            strict_mode,
            enable_experimental,
            features,
            debug,
            None,
        )
    }

    pub fn new_with_source(
        ctx: &'a mut TypeContext,
        symbols: &'a SymbolTable,
        trait_env: &'a mut TraitEnv,
        resolution_map: ResolutionMap,
        strict_mode: bool,
        enable_experimental: bool,
        features: Vec<String>,
        debug: bool,
        source: Option<&'a str>,
    ) -> Self {
        let mut checker = TypeChecker {
            ctx,
            symbols,
            trait_env,
            diagnostics: DiagCtxt::new(),
            source,
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
            ghost_var_scopes: Rc::new(RefCell::new(Vec::new())),
            runtime_var_scopes: Rc::new(RefCell::new(Vec::new())),
            must_handle_sources: RefCell::new(HashSet::new()),
            in_ghost_var_def: Cell::new(false),
            functions_accessing_mutables: HashSet::new(),
            current_function_trusted: false,
            comptime_fn_registry: HashMap::new(),
            comptime_fn_pass: false,
            deferred_comptime_blocks: Vec::new(),
            literal_values: HashMap::new(),
            scope_var_stack: Vec::new(),
            builtin_registry: BuiltinTraitRegistry::new(),
            proj_cache: ProjectionCache::new(),
            trait_obligations: Vec::new(),
            residual_trait_obligations: Vec::new(),
            strict_mode,
            enable_experimental,
            features,
            debug,
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

        // Wrap the entire program in a literal value scope so that
        // top-level variable definitions (`set x = 42`) have a scope
        // frame to track into.  Popped at the end of this function.
        self.push_literal_scope();

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
        // Also skip items whose @cfg condition is not met.
        let comptime_fn_indices: Vec<usize> = program
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, stmt)| {
                if self.should_skip_due_to_cfg(stmt) {
                    return None;
                }
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
        for (captures, hir, _ty, span) in self.deferred_comptime_blocks.drain(..) {
            // Collect literal values for capture names before any mutable borrow of self.
            let captured_literals: Vec<(Symbol, Option<ComptimeValue>)> = captures
                .iter()
                .map(|(sym, _span)| {
                    (
                        *sym,
                        self.literal_values.get(sym).and_then(|v| v.last()).cloned(),
                    )
                })
                .collect();
            let traceback = {
                let frames: Vec<&CtxFrame> = self
                    .region_tree
                    .iter_frames_rev()
                    .filter(|f| matches!(f.kind, CtxKind::Comptime))
                    .collect();
                format_comptime_traceback_inner(&frames, self.source)
            };
            let mut eval = crate::hir::comptime::ComptimeEvalContext::new_with_source(
                self.ctx,
                self.symbols,
                &mut self.diagnostics,
                traceback.clone(),
                self.source,
            );
            // Inject captured variable values (from pre-collected literals).
            for (capture, val) in &captured_literals {
                if let Some(val) = val {
                    let slot = eval.allocate_slot();
                    eval.cur_slot.insert(*capture, slot);
                    eval.variables.insert(slot, val.clone());
                }
            }
            for (name, (params, body)) in &self.comptime_fn_registry {
                eval.register_fn(*name, params.clone(), body.clone());
            }
            let eval_result = eval.eval_block(&hir);
            drop(eval);
            let _ = crate::diagnostics::adorn_with_span(
                &mut self.diagnostics,
                span,
                None,
                |ctxt| {
                    match eval_result {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            let msg = if traceback.is_empty() {
                                crate::tr!("comptime error: {e}", e = e)
                            } else {
                                let tb_str: Vec<String> = traceback
                                    .iter()
                                    .map(|(reason, span)| {
                                        let r = match reason {
                                            ComptimeReason::ComptimeBlock => {
                                                "comptime { ... } block"
                                            }
                                            ComptimeReason::ComptimeFnDef => {
                                                "comptime def function body"
                                            }
                                            ComptimeReason::ComptimeFnCall => {
                                                "comptime function call"
                                            }
                                            ComptimeReason::ComptimeTest => {
                                                "@comptime_test function"
                                            }
                                            ComptimeReason::Assertion => "assert!()",
                                            ComptimeReason::TypeInfo => "@typeInfo!()",
                                            ComptimeReason::LayoutOf => "layout_of!()",
                                        };
                                        if span.start != 0 || span.end != 0 {
                                            format!("  · {r} at offset {}", span.start)
                                        } else {
                                            format!("  · {r}")
                                        }
                                    })
                                    .collect();
                                format!(
                                    "comptime error: {}\ncomptime call stack (most recent first):\n{}",
                                    e,
                                    tb_str.join("\n"),
                                )
                            };
                            // Push once; adorn_with_span will add span/source.
                            Err(ctxt
                                .push(Diagnostic::error(msg).with_code_str("E080"))
                                .into())
                        }
                    }
                },
            );
        }

        // Pass 3: type-check remaining items (non-comptime functions,
        // comptime blocks, type defs, etc.) in order.
        // Skip items whose @cfg condition is not met.
        for (i, stmt) in program.items.iter().enumerate() {
            if comptime_fn_indices.contains(&i) {
                continue; // already processed in pass 2
            }
            // In strict mode, check that @cfg conditions are provably reachable.
            if self.strict_mode {
                self.check_cfg_reachability(stmt);
            }
            self.validate_auto_ro_placement(stmt);
            if self.should_skip_due_to_cfg(stmt) {
                items.push(HirStmt::Stripped { span: stmt.span() }); // item excluded by @cfg
                continue;
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
                    let ty = self
                        .ctx
                        .get(self_ty)
                        .display_with(self.ctx, Some(self.symbols));
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
                    Diagnostic::error(crate::tr!("trait solver error: {msg}", msg = msg))
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

        self.pop_literal_scope();
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
                    if self.local_variable_types.current_frame_contains(*var_name) {
                        let prev_span = self.span_get(var_name).unwrap_or(*span);
                        dup_diag = Some(
                            Diagnostic::error_kind(DiagnosticKind::DuplicateDefinition {
                                name: var_name.to_string(),
                                this_span: *span,
                                original_span: prev_span,
                            })
                            .with_code_str("E019"),
                        );
                    } else if self.local_variable_types.get(*var_name).is_some() {
                        // Shadowing is allowed but warns.
                        let prev_span = self.span_get(var_name).unwrap_or(*span);
                        self.diagnostics.push(
                            Diagnostic::warning(crate::tr!(
                                "shadowing definition of `{name}`",
                                name = var_name.as_str()
                            ))
                            .with_code_str("W113")
                            .with_span(*span)
                            .with_additional_span(prev_span)
                            .with_secondary_label(prev_span, "previous definition here"),
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
                            let (hir, ty) = self.infer_expr(value, None)?;
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
                            d.related_errors_mut()
                                .push(crate::diagnostics::RelatedError {
                                    code: rhs_err.code().cloned(),
                                    message: rhs_err.message().to_string(),
                                    span: rhs_err.spans().first(),
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
                if let Some(var_name) = name
                    && dup_diag.is_none()
                {
                    self.local_variable_types.insert(*var_name, final_ty);
                    self.span_insert(*var_name, *span);
                    // Register the name as RUNTIME unless this is the
                    // inner statement of a `ghost set` (already added to
                    // `ghost_var_scopes` by `GhostVariableDef`).
                    if !self.in_ghost_var_def.get() {
                        let mut rscopes = self.runtime_var_scopes.borrow_mut();
                        if let Some(rscope) = rscopes.last_mut() {
                            rscope.insert(*var_name);
                        }
                    }
                    // Track comptime-known literal values for explicit captures.
                    if !*mutable {
                        if let Some(ref value_hir) = value_hir {
                            if let HirExpr::Literal(lit, _, _) = value_hir {
                                let cv = match lit {
                                    Literal::Int(v) => Some(ComptimeValue::Int(*v)),
                                    Literal::Float(v) => Some(ComptimeValue::Float(*v)),
                                    Literal::Bool(v) => Some(ComptimeValue::Bool(*v)),
                                    Literal::String(s) => Some(ComptimeValue::String(
                                        std::sync::Arc::from(s.as_str()),
                                    )),
                                    _ => None,
                                };
                                if let Some(val) = cv {
                                    self.insert_literal_value(*var_name, val);
                                }
                            } else {
                                // Non-literal initializer — try evaluating it
                                // as a comptime expression (e.g. `40 + 2`,
                                // `fibonacci!(5)`).  If it evaluates
                                // successfully, the result is stored as a
                                // comptime-known value for explicit captures.
                                let result = {
                                    // Direct capture: `set y = x` where `x` is
                                    // already tracked as a comptime-known literal.
                                    if let HirExpr::Ident(name, _, _) = value_hir {
                                        if let Some(val) = self.get_literal_value(name) {
                                            Some(Ok(val.clone()))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                    .unwrap_or_else(|| {
                                        let mut ec = crate::hir::comptime::ComptimeEvalContext::new(
                                            self.ctx,
                                            self.symbols,
                                            &mut self.diagnostics,
                                        );
                                        // Register comptime functions so that
                                        // pure comptime function calls in the
                                        // initializer resolve correctly.
                                        for (fn_name, (fn_params, fn_body)) in
                                            &self.comptime_fn_registry
                                        {
                                            ec.register_fn(
                                                *fn_name,
                                                fn_params.clone(),
                                                fn_body.clone(),
                                            );
                                        }
                                        ec.eval_expr(value_hir)
                                    })
                                };
                                match result {
                                    Ok(val) => {
                                        self.insert_literal_value(*var_name, val);
                                    }
                                    Err(_) => {
                                        // Expression is not comptime-evaluable —
                                        // skip.  The capture list will report
                                        // a proper error later.
                                    }
                                }
                            }
                        }
                    }
                }

                // Track mutable global variables (top-level `set mut`).
                // These require `@trusted` context to be read/written.
                if *mutable
                    && self.current_function.is_none()
                    && let Some(var_name) = name
                {
                    self.mutable_globals.insert(*var_name);
                }

                // `set auto<T, N> = expr` — bind captured type names to the inferred type.
                // Each name in `type_captures` becomes available as a type alias in
                // comptime reflection (e.g., `@typeInfo!(T)`).
                for capture in type_captures {
                    self.local_type_param_cache.insert(capture.name, final_ty);
                }

                // Track `@must_handle` sources: if the value expression is a
                // call to a `@must_handle` function, record the variable name
                // so that `Expr::Try` and `Expr::Catch` can fire even when
                // the result is stored in a variable before `?` or `catch`.
                if let Some(n) = name
                    && let Some(v) = value
                    && let Expr::Call { callee, .. } = v
                {
                    // `@must_handle` source tracking: cover direct calls,
                    // static-method calls, and method calls — the
                    // store-then-propagate bypass must not hold for methods
                    // (SYNTAX.md accountability covers ALL call sites).
                    let is_must_handle = match callee.as_ref() {
                        Expr::Ident(callee_name, _) => self
                            .symbols
                            .lookup_function(*callee_name)
                            .map_or(false, |b| {
                                b.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            }),
                        Expr::Path(path, _) if path.len() >= 2 => {
                            self.symbols.lookup_function(path[1]).map_or(false, |b| {
                                b.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            })
                        }
                        Expr::FieldAccess { field, .. } => {
                            // Method call — look up `@must_handle` through
                            // the receiver type's impl blocks (mirror of the
                            // `Expr::Try` path).
                            let base_ty = value_hir.as_ref().and_then(|vh| match vh {
                                HirExpr::Call { callee: c, .. } => match c.as_ref() {
                                    HirExpr::FieldAccess { base, .. } => Some(base.ty()),
                                    _ => None,
                                },
                                _ => None,
                            });
                            base_ty.map_or(false, |base_ty| {
                                self.trait_env
                                    .lookup_inherent_methods(base_ty, self.ctx)
                                    .iter()
                                    .any(|m| {
                                        m.name == *field
                                            && m.attributes
                                                .iter()
                                                .any(|a| a.name.eq_str("must_handle"))
                                    })
                                    || self
                                        .trait_env
                                        .lookup_impls_for_type(base_ty)
                                        .iter()
                                        .flat_map(|ic| &ic.methods)
                                        .any(|m| {
                                            m.name == *field
                                                && m.attributes
                                                    .iter()
                                                    .any(|a| a.name.eq_str("must_handle"))
                                        })
                            })
                        }
                        _ => false,
                    };
                    if is_must_handle {
                        self.must_handle_sources.borrow_mut().insert(*n);
                    }
                }

                Ok(HirStmt::VariableDef {
                    kind: *kind,
                    mutable: *mutable,
                    name: *name,
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
                // ── Save per-function state for nested `def` support ──
                let prev_frozen = self.ctx.frozen_vars.borrow().clone();
                let prev_seal = self.ctx.seal_violations.get();
                let prev_return_ty = self.current_return_type;
                let prev_must_handle = self.must_handle_sources.borrow().clone();
                // `&ro`-freeze bookkeeping is function-scoped: a `&ro r`
                // borrow in one function must not freeze an unrelated `r`
                // in another (SYNTAX.md — the freeze lasts for the borrow's
                // lifetime, i.e. within the function).
                self.ctx.frozen_vars.borrow_mut().clear();
                // The GADT-seal violation counter is function-scoped too —
                // a violation in one function must not be misattributed to
                // a later, clean function in strict mode.
                self.ctx.seal_violations.set(0);
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
                // Const generic params (kind Const) are EXEMPT from the
                // generality check: they monomorphize
                // per concrete constant value (SYNTAX.md §Const Generics),
                // so each instantiation is checked separately.
                let const_param_names: Vec<Symbol> = type_params
                    .iter()
                    .filter(|tp| matches!(tp.kind, crate::ast::TypeParamKind::Const { .. }))
                    .map(|tp| tp.name)
                    .collect();
                // Register the type params into the binder scope FIRST, so a
                // const parameter's declared VALUE type may reference them
                // (e.g. `const N: T`) when it is resolved below.
                for (i, tp) in type_params.iter().enumerate() {
                    let generic_id = self.ctx.generic_param(i, tp.name);
                    self.local_type_param_cache.insert(tp.name, generic_id);
                }
                // THEN resolve each const parameter's declared VALUE type
                // (`const N: usize` → usize).  The narrowed E104 exemption
                // allows a const param to be bound only to a type consistent
                // with its value type (the monomorphization) — an unrelated
                // concrete type (e.g. N := Bool) is a generality violation.
                let mut const_param_value_types: std::collections::HashMap<Symbol, TypeId> =
                    std::collections::HashMap::default();
                for tp in type_params.iter() {
                    if let crate::ast::TypeParamKind::Const { ty, .. } = &tp.kind {
                        if let Ok(value_ty) = self.resolve_type(ty) {
                            const_param_value_types.insert(tp.name, value_ty);
                        }
                    }
                }

                // ── Where-equality given constraints (Q14) ────────────
                // `where T == U` / `where T == Int<32>` establish a given
                // equivalence within the function body (committee ruling):
                // unify the two sides so T and U become interchangeable,
                // and a concrete RHS resolves T.  Plain unification has
                // union-find semantics — circular equalities are harmless
                // (already-equivalent types unify trivially), NOT a rewrite
                // chain (so E064's cycle concern does not apply).
                //
                // Founder's ruling: cycles are ACCEPTED (union-find), but a
                // redundant closing edge (the two sides already equivalent)
                // is flagged with a WARNING — the committee's "looks odd ⇒
                // likely a programmer error" tradition, minus the overly
                // aggressive compile error.
                if let Some(wc) = where_clause.as_ref() {
                    for eq in &wc.equalities {
                        if let (Ok(l), Ok(r)) =
                            (self.resolve_type(&eq.left), self.resolve_type(&eq.right))
                        {
                            // Redundancy check: the two sides are "already
                            // equivalent" only if they RESOLVE to the same
                            // type.  `can_unify` would bind a fresh
                            // GenericParam inside a rolled-back transaction
                            // and report "redundant" for the FIRST constraint
                            // on the parameter — a false positive.
                            if self.ctx.resolve_binding(l) == self.ctx.resolve_binding(r) {
                                self.diagnostics.push(
                                    Diagnostic::warning(
                                        "where equality is redundant: the two sides are already equivalent",
                                    )
                                    .with_span(eq.span),
                                );
                            }
                            if let Err(_) = self.ctx.unify_tracked(l, r, eq.span) {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "where equality contradicts a prior constraint: the two sides cannot be unified",
                                    )
                                    .with_code_str("E107")
                                    .with_span(eq.span)
                                    .with_help(
                                        "remove the contradictory equality, or split into separate functions",
                                    ),
                                );
                            }
                        }
                    }
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
                        name: param.name,
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

                // ── Proof obligation check (strict mode) ────────────────
                // In strict mode, all @trusted functions must have @link_proof
                // or @comptime_test evidence.  This ensures that trust
                // boundaries are backed by formal proofs or test coverage.
                if guard.checker.current_function_trusted && guard.checker.strict_mode {
                    let has_link_proof = attributes.iter().any(|a| a.name.eq_str("link_proof"));
                    let has_comptime_test =
                        attributes.iter().any(|a| a.name.eq_str("comptime_test"));
                    if !has_link_proof && !has_comptime_test {
                        guard.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "@trusted function `{}` must have @link_proof or @comptime_test evidence in strict mode",
                                name,
                            ))
                            .with_code_str("E091")
                            .with_span(*span)
                            .with_help("add `@link_proof(path, hash)` referencing an external proof, or `@comptime_test` to validate at compile time")
                            .with_suggestion("add `@link_proof(\"path/to/proof.coq\", \"sha256:...\")` or `@comptime_test` above this function"),
                        );
                    }
                }

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
                if *is_comptime {
                    guard
                        .checker
                        .push_comptime_ctx(ComptimeReason::ComptimeFnDef, *span);
                }

                // Enter a variable scope for the function body
                let _scope = guard.checker.enter_var_scope();

                // Pre-populate the local variable cache with function parameters
                // and `result` so that ensures clauses can reference them.
                for p in &hir_params {
                    guard.checker.local_variable_types.insert(p.name, p.ty);
                    guard.checker.span_insert(p.name, p.span);
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
                        let (_, ensures_ty) = match guard.checker.infer_expr(expr, None) {
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
                                                GenericArg::Const(_) => {
                                                    // Const generic args are handled by
                                                    // comptime evaluation, not type resolution.
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
                                                GenericArg::Const(_) => {
                                                    // Const generic args: currently no resolution needed
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

                // ── @auto_ro (SYNTAX.md §Local Relaxation) ─────────────
                // `@auto_ro` allows `&mut T` to be implicitly coerced to
                // `&T` at call sites / method resolution within this
                // function.  Without it, the unifier rejects the coercion
                // by default and the transition must be explicit (`&ro`).
                // Save/restore so nested function definitions nest
                // correctly.
                let prev_auto_ro = guard.checker.ctx.auto_ro.get();
                let prev_auto_coerce = guard.checker.ctx.auto_coerce.get();
                // DCE guardrail (SYNTAX.md §Reference Coercion): the
                // implicit freeze must NOT apply inside `@trusted` functions
                // or in Strict Mode, even when `@auto_ro` is present.
                let has_auto_ro = attributes.iter().any(|a| a.name.eq_str("auto_ro"));
                let has_auto_coerce = attributes.iter().any(|a| a.name.eq_str("auto_coerce"));
                let has_trusted = attributes.iter().any(|a| a.name.eq_str("trusted"));
                let auto_ro_active = has_auto_ro && !guard.checker.strict_mode && !has_trusted;
                guard.checker.ctx.auto_ro.set(auto_ro_active);
                let auto_coerce_active =
                    has_auto_coerce && !guard.checker.strict_mode && !has_trusted;
                guard.checker.ctx.auto_coerce.set(auto_coerce_active);
                // SYNTAX.md §Local Relaxation: `@auto_ro` is NOT permitted
                // inside `@trusted` functions or in Strict Mode — report it
                // rather than silently ignoring the attribute.
                if has_auto_ro {
                    if guard.checker.strict_mode {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("`@auto_ro` is not permitted in Strict Mode")
                                .with_span(*span),
                        );
                    } else if has_trusted {
                        guard.checker.diagnostics.push(
                            Diagnostic::error(
                                "`@auto_ro` is not permitted on `@trusted` functions",
                            )
                            .with_span(*span),
                        );
                    }
                }
                if has_auto_coerce {
                    if guard.checker.strict_mode {
                        guard.checker.diagnostics.push(
                            Diagnostic::error("`@auto_coerce` is not permitted in Strict Mode")
                                .with_span(*span),
                        );
                    } else if has_trusted {
                        guard.checker.diagnostics.push(
                            Diagnostic::error(
                                "`@auto_coerce` is not permitted on `@trusted` functions",
                            )
                            .with_span(*span),
                        );
                    }
                }
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
                guard.checker.ctx.auto_ro.set(prev_auto_ro);
                guard.checker.ctx.auto_coerce.set(prev_auto_coerce);
                // Restore per-function state (for nested `def` support):
                // the outer function's `frozen_vars` and `seal_violations`
                // must survive the nested function's body checking.
                *guard.checker.ctx.frozen_vars.borrow_mut() = prev_frozen;
                guard.checker.ctx.seal_violations.set(prev_seal);
                guard.checker.current_return_type = prev_return_ty;
                *guard.checker.must_handle_sources.borrow_mut() = prev_must_handle;
                // Fail closed (strict mode): a nonzero seal-violation count
                // means a GenericParam binding was skipped inside a GADT arm
                // and never recovered — surface it instead of silently
                // accepting a possibly-unresolved program.
                if guard.checker.ctx.seal_violations.get() > 0 {
                    let diag = if guard.checker.strict_mode {
                        Diagnostic::error("internal: GADT arm seal violations were not recovered")
                    } else {
                        Diagnostic::warning("internal: GADT arm seal violations were not recovered")
                    };
                    guard.checker.diagnostics.push(diag.with_span(*span));
                }

                guard.checker.pop_ctx();
                if *is_comptime {
                    guard.checker.pop_ctx();
                }

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
                if return_type.is_none()
                    && let Some(ref body_stmts) = body_hir
                {
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
                                    if let Some(else_stmts) = else_branch
                                        && has_return_recursive(else_stmts)
                                    {
                                        return true;
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
                                TraitPredicate::NormalizesTo { projection, target } => {
                                    crate::hir::traits::solver::Predicate::NormalizesTo {
                                        projection: projection.clone(),
                                        target: *target,
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
                        // SAFETY: `symbols_ptr` outlives this block (borrowed from
                        // `self.symbols` by the caller); no other code mutates it.
                        let symbols = unsafe { &*symbols_ptr };
                        // SAFETY: `ctx_ptr` points to the `TypeContext` owned by the
                        // caller; all access goes through `RefCell` interior mutability,
                        // and no concurrent access is possible here.
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
                    let details: Vec<String> =
                        diags.iter().map(|d| d.message().to_string()).collect();
                    return Err(Diagnostic::error(format!(
                        "inference failure: {}",
                        details.join("; ")
                    ))
                    .with_span(*span));
                }

                if let Some(ref body_stmts) = body_hir
                    && return_type.is_some()
                {
                    // User wrote an explicit return type — check body against it.
                    let body_ty = self.block_type_impl(body_stmts, false);
                    self.unify_with(return_ty, body_ty, *span, TypingContext::ReturnValue)?;
                }
                // When return_type is None, the infer var was already unified
                // with return values during body checking (via current_return_type),
                // or defaulted to Never before the solver ran (see above).

                // Contract verification skeleton: check that requires/ensures are bool,
                // and decreases/terminates are integer types.
                for contract in contracts {
                    match contract {
                        Contract::Requires(expr, cspan) | Contract::Invariant(expr, cspan) => {
                            let (_, ty) = self.infer_expr(expr, None)?;
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
                            let (_, ty) = self.infer_expr(expr, None)?;
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
                            let (_, ty) = self.infer_expr(expr, None)?;
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
                                TraitPredicate::NormalizesTo { projection, target } => {
                                    crate::hir::traits::solver::Predicate::NormalizesTo {
                                        projection: projection.clone(),
                                        target: *target,
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
                        // SAFETY: `symbols_ptr` outlives this block (borrowed from
                        // `self.symbols` by the caller); no other code mutates it.
                        let symbols = unsafe { &*symbols_ptr };
                        // SAFETY: `ctx_ptr` points to the `TypeContext` owned by the
                        // caller; all access goes through `RefCell` interior mutability,
                        // and no concurrent access is possible here.
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

                // ── Generality check ─────────────────────────
                // A generic parameter must not be solved to a concrete
                // type by the function body: the definition must
                // type-check for ALL instantiations (rustc's rigid
                // `TyKind::Param` / GHC & OCaml's skolem discipline).
                // After the seal, GADT refinements never write
                // global bindings, so any binding remaining on a function
                // generic param is body-driven and is a generality
                // violation (E104).
                // Params explicitly constrained by `where T == Concrete`
                // are exempt: the constraint is declared in
                // the signature, so the body may rely on it.
                let where_eq_exempt: Vec<Symbol> = where_clause
                    .as_ref()
                    .map(|wc| {
                        // A single-segment path is a type parameter ONLY if
                        // it names a declared generic parameter of this
                        // function — built-in types that parse as
                        // single-segment paths (`Bool`, `String`, `USize`,
                        // ...) are CONCRETE types for the exemption logic.
                        let is_param = |ty: &crate::ast::Type| {
                            matches!(
                                ty,
                                crate::ast::Type::Path(p, _)
                                    if p.len() == 1 && fn_param_names.contains(&p[0])
                            )
                        };
                        wc.equalities
                            .iter()
                            .flat_map(|eq| {
                                // Exempt a side ONLY when the equality
                                // constrains it to a CONCRETE type (Q6:
                                // `where T == Int<32>`).  A param-to-param
                                // equality (`where T == U`) exempts NEITHER
                                // side: the given-equality registration
                                // (unify at function entry) already makes T
                                // and U interchangeable, which the
                                // generality check tolerates (binding one to
                                // the other is a GenericParam binding, not a
                                // violation) — but a BODY-driven concrete
                                // binding of the equivalence class must
                                // still fire E104.
                                let mut out = Vec::new();
                                if !is_param(&eq.right)
                                    && let crate::ast::Type::Path(p, _) = &eq.left
                                    && fn_param_names.contains(&p[0])
                                {
                                    out.push(p[0]);
                                }
                                if !is_param(&eq.left)
                                    && let crate::ast::Type::Path(p, _) = &eq.right
                                    && fn_param_names.contains(&p[0])
                                {
                                    out.push(p[0]);
                                }
                                out
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                // Params that participate in a param-to-param equality
                // (`where T == U`): the given-equivalence makes T and U
                // interchangeable, but ONLY the GenericParam interchange is
                // exempt from E104 — a concrete resolution (the body
                // specializing the equivalence class) is a violation.
                let where_eq_class: Vec<Symbol> = where_clause
                    .as_ref()
                    .map(|wc| {
                        wc.equalities
                            .iter()
                            .flat_map(|eq| {
                                let mut out = Vec::new();
                                let is_param = |ty: &crate::ast::Type| {
                                    matches!(
                                        ty,
                                        crate::ast::Type::Path(p, _)
                                            if p.len() == 1 && fn_param_names.contains(&p[0])
                                    )
                                };
                                if is_param(&eq.left) && is_param(&eq.right) {
                                    if let crate::ast::Type::Path(p, _) = &eq.left {
                                        out.push(p[0]);
                                    }
                                    if let crate::ast::Type::Path(p, _) = &eq.right {
                                        out.push(p[0]);
                                    }
                                }
                                out
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                // ── E104 exemption categories ──────────────────────────
                // A generic parameter is exempt from the generality check
                // when it falls into one of:
                //   1. const generic params (monomorphized per value);
                //   2. `where T == ConcreteType` params (explicitly
                //      constrained by the signature — Q6);
                //   3. `where T == U` params resolving to ANOTHER
                //      GenericParam (the declared given equivalence — Q14).
                // A `where T == U` param resolving to a CONCRETE type (the
                // body specializing the equivalence class) is NOT exempt —
                // that is a body-driven specialization and fires E104.
                for name in &fn_param_names {
                    let Some(&tid) = self.local_type_param_cache.get(name) else {
                        continue;
                    };
                    let resolved = self.ctx.resolve_binding(tid);
                    if const_param_names.contains(name) {
                        // Narrowed exemption: a const param may only be bound
                        // to a type consistent with its declared VALUE type
                        // (the monomorphization).  An unrelated concrete type
                        // (e.g. N := Bool for `const N: usize`) is a
                        // generality violation and falls through to the E104
                        // below — the broad exemption silently accepted it.
                        if let Some(value_ty) = const_param_value_types.get(name) {
                            // Probe WITHOUT committing: try_unify mutates the
                            // inference state (it drives unify_internal), so
                            // wrap the check in a transaction and roll back —
                            // a pure unifiability probe must not bind the
                            // const param here.
                            let depth = self.ctx.transaction_depth();
                            self.ctx.begin_transaction();
                            let compatible = self.ctx.try_unify(resolved, *value_ty, None).is_ok();
                            self.ctx.rollback_to(depth);
                            if compatible {
                                continue;
                            }
                        }
                    } else if where_eq_exempt.contains(name) {
                        continue; // where-constrained params are exempt
                    }
                    // Param-to-param where-equalities (`where T == U`)
                    // establish a given equivalence: T resolving to U
                    // (another GenericParam) is the declared interchange and
                    // is exempt — but a CONCRETE resolution (the body
                    // specializing the equivalence class) is a generality
                    // violation.
                    if where_eq_class.contains(name)
                        && matches!(self.ctx.get_raw(resolved), TypeData::GenericParam { .. })
                    {
                        continue;
                    }
                    // Generality violation: the parameter was bound to a
                    // concrete type, OR to a DIFFERENT generic parameter
                    // (`T := U` — the body must be parametric in each
                    // distinct parameter; rustc/GHC/OCaml all reject
                    // rigid-param-to-rigid-param unifications).  Resolution
                    // to itself (unbound) or to an unbound InferVar /
                    // skolem / error is fine — still abstract.
                    let violation = resolved != tid
                        && !matches!(
                            self.ctx.get_raw(resolved),
                            TypeData::InferVar { .. }
                                | TypeData::SkolemVar { .. }
                                | TypeData::Error
                        );
                    if violation {
                        // Point at the precise binding site when recorded
                        // (see `unify_tracked` / `set_binding` origin
                        // capture); otherwise fall back to the function span.
                        let origin = self.ctx.generic_binding_origins.borrow().get(&tid).copied();
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "generic parameter `{}` is constrained to a specific type by the function body; the body must type-check for every instantiation",
                                name,
                            ))
                            .with_code_str("E104")
                            .with_span(origin.unwrap_or(*span)),
                        );
                    }
                }

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
                            .insert(*name, (param_names, body.clone()));
                    }
                }

                // ── @comptime_test: execute at compile time ──────────────
                // If the function has the `@comptime_test` attribute, evaluate
                // its body at compile time.  Test failures (assertion failures)
                // cause a compile error.  The function is stripped from the
                // final binary (not emitted to HIR).
                let is_comptime_test = attributes.iter().any(|a| a.name.eq_str("comptime_test"));
                if is_comptime_test {
                    // @comptime_test functions must have no parameters — eval_block
                    // cannot bind parameters, so any params would fail with
                    // UnknownIdentifier when referenced in the body.
                    if !params.is_empty() {
                        self.diagnostics.push(
                            Diagnostic::error("@comptime_test function must have no parameters")
                                .with_code_str("E090")
                                .with_span(*span)
                                .with_help("@comptime_test functions are executed at compile time with no arguments")
                                .with_suggestion("remove the parameters, or use a regular comptime def function instead"),
                        );
                        return Ok(HirStmt::Error);
                    }
                    if let Some(ref body) = body_hir {
                        // If type-checking already produced Error nodes anywhere
                        // in the body (including nested inside if branches, loop
                        // bodies, closures, etc.), skip comptime evaluation to
                        // avoid confusing secondary errors.
                        let has_error = contains_error(body);
                        if !has_error {
                            let outer_tb = {
                                let frames: Vec<&CtxFrame> = self
                                    .region_tree
                                    .iter_frames_rev()
                                    .filter(|f| matches!(f.kind, CtxKind::Comptime))
                                    .collect();
                                format_comptime_traceback_inner(&frames, self.source)
                            };
                            let mut eval =
                                crate::hir::comptime::ComptimeEvalContext::new_with_source(
                                    self.ctx,
                                    self.symbols,
                                    &mut self.diagnostics,
                                    outer_tb,
                                    self.source,
                                );
                            for (fn_name, (fn_params, fn_body)) in &self.comptime_fn_registry {
                                eval.register_fn(*fn_name, fn_params.clone(), fn_body.clone());
                            }
                            match eval.eval_block(body) {
                                Ok(_) => {
                                    // Test passed — strip from HIR.
                                    return Ok(HirStmt::Stripped { span: *span });
                                }
                                Err(e) => {
                                    self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "@comptime_test `{}` failed: {}",
                                        name, e,
                                    ))
                                    .with_code_str("E090")
                                    .with_span(*span)
                                    .with_help("@comptime_test functions are executed at compile time; fix the assertion failure"),
                                );
                                    return Ok(HirStmt::Error);
                                }
                            }
                        } // end if !has_error
                        // If the body contains Error nodes (type errors), the test
                        // cannot be evaluated — strip it to avoid leaking an errored
                        // @comptime_test function into the final HIR.
                        return Ok(HirStmt::Stripped { span: *span });
                    }
                    if body_hir.is_none() {
                        // body_hir is None (extern function) — emit error below.
                        self.diagnostics.push(
                        Diagnostic::error("@comptime_test requires a function body")
                            .with_code_str("E090")
                            .with_span(*span)
                            .with_help("@comptime_test functions must have a body to execute at compile time; extern declarations cannot be tested")
                            .with_suggestion("remove `@comptime_test` from this function, or provide a body"),
                    );
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
                    name: *name,
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
                    generated_by_comptime: false,
                })
            }
            Stmt::Expression(expr) => {
                let (hir, _) = self.infer_expr(expr, None)?;
                Ok(HirStmt::Expression(Box::new(hir)))
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                let (cond_hir, cond_ty) = self.infer_expr(cond, None)?;
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
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee, None)?;
                // Shared GADT arm lifecycle (enter/pop/region-restore
                // encapsulated in `with_gadt_arm` — same sequence as the
                // other three pattern-matching sites).
                let (pattern_hir, then_hir, _gadt_reachable) =
                    self.with_gadt_arm(scrut_ty, pattern, *span, |ck, _| {
                        ck.check_block(then_branch)
                    })?;
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
                let (cond_hir, cond_ty) = self.infer_expr(cond, None)?;
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
                    .map(|inv| self.infer_expr(inv, None).map(|(h, _)| h))
                    .transpose()?;
                let dec_hir = decreases
                    .as_ref()
                    .map(|dec| self.infer_expr(dec, None).map(|(h, _)| h))
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
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee, None)?;
                // Shared GADT arm lifecycle (enter/pop/region-restore
                // encapsulated in `with_gadt_arm` — same sequence as the
                // other three pattern-matching sites).
                let (pattern_hir, (inv_hir, dec_hir, body_hir), _gadt_reachable) = self
                    .with_gadt_arm(scrut_ty, pattern, *span, |ck, _| {
                        let inv_hir = invariant
                            .as_ref()
                            .map(|inv| ck.infer_expr(inv, None).map(|(h, _)| h))
                            .transpose()?;
                        let dec_hir = decreases
                            .as_ref()
                            .map(|dec| ck.infer_expr(dec, None).map(|(h, _)| h))
                            .transpose()?;
                        ck.push_ctx(CtxKind::While, *span, None);
                        let body_hir = ck.check_block(body)?;
                        ck.pop_ctx();
                        Ok((inv_hir, dec_hir, body_hir))
                    })?;
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
                let (iter_hir, iter_ty) = self.infer_expr(iterable, None)?;
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
                    .map(|inv| self.infer_expr(inv, None).map(|(h, _)| h))
                    .transpose()?;
                let dec_hir = decreases
                    .as_ref()
                    .map(|dec| self.infer_expr(dec, None).map(|(h, _)| h))
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
                            label: *label,
                            span: *span,
                        })
                    }
                    Some(_) => Ok(HirStmt::Leave {
                        label: *label,
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
                            label: *label,
                            span: *span,
                        })
                    }
                    Some(_) => Ok(HirStmt::Continue {
                        label: *label,
                        span: *span,
                    }),
                }
            }
            Stmt::Return {
                value,
                labels,
                span,
            } => {
                // Check if we're inside a comptime BLOCK — if so, return is
                // comptime control flow, not a real function return.  A
                // comptime FUNCTION body (`comptime def f() -> T { return v; }`,
                // SYNTAX.md §"Type Factories") is NOT a block: its `return`
                // is a real function return and must carry a value.
                let in_comptime_block = self
                    .region_tree
                    .iter_frames_rev()
                    .find(|f| matches!(f.kind, CtxKind::Comptime))
                    .map_or(false, |f| {
                        !matches!(f.comptime_reason, Some(ComptimeReason::ComptimeFnDef))
                    });
                if in_comptime_block {
                    // Inside comptime, `return` acts as comptime control flow:
                    // the value is evaluated and propagated out of the comptime block.
                    if let Some(value) = value {
                        let (hir, _) = self.infer_expr(value, None)?;
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
                    if let Some(ref ast_expr) = g.ast_expr
                        && let Some(return_value) = value
                    {
                        let fast_path_ok = try_fast_path(ast_expr, return_value);
                        if fast_path_ok {
                            // Fast path succeeded — guarantee is trivially satisfied.
                            // Skip the SMT check entirely.
                        } else {
                            // Fast path failed — fall through to the type check below.
                            let _ = fast_path_ok;
                        }
                    }

                    // The postcondition type (if present) must be bool,
                    // indicating the ensures clause holds at the return point.
                    if let Predicate::Type(post) = g.post
                        && !self.ctx.is_bool(post)
                    {
                        self.diagnostics.push(
                            Diagnostic::error("ensures condition must be boolean at return")
                                .with_code_str("E022")
                                .with_span(*span),
                        );
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
                // Ban `return Err(...)` — use `leave with` instead.
                // Unified with the `Expr::LeaveWith` path via
                // `is_result_err_constructor` (follows the alias chain of
                // `Result` — multi-level aliases cannot bypass the lint).
                if let Some(Expr::EnumLit { path, variant, .. }) = value
                    && self.is_result_err_constructor(path, variant)
                {
                    self.emit_return_err_lint(*span);
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
                        let (hir, _) = self.infer_expr(value, None)?;
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
                            let _ = self.unify(ret_ty, self.ctx.unit(), *span, TypeCtx::ReturnType);
                        } else if !self.ctx.is_unit(ret_ty) && !self.ctx.is_never(ret_ty) {
                            self.diagnostics.push(
                                Diagnostic::error("return without value in non-unit function")
                                    .with_span(*span)
                                    .with_suggestion(format!(
                                        "return a value of the declared return type `{}`; for \
                                         error exits in a Result-returning function use \
                                         `leave with Err(..)` or `return Ok(..)`",
                                        self.ctx.get(ret_ty),
                                    )),
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
                // Reject mutation of a variable frozen by an active `&ro`
                // borrow (SYNTAX.md §Reference Coercion).
                let frozen_target = expr_root_ident(target);
                if let Some(name) = frozen_target
                    && self.ctx.frozen_vars.borrow().contains(&name)
                {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "cannot mutate a variable frozen by an active `&ro` borrow",
                        )
                        .with_span(*span),
                    );
                }
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
                    // Track which functions access mutable globals (for isolate checking)
                    if self.mutable_globals.contains(name)
                        && let Some(def_id) = self.current_function
                    {
                        self.functions_accessing_mutables.insert(def_id);
                    }
                    // Mutable globals are also forbidden inside comptime blocks
                    if self.mutable_globals.contains(name) && self.is_in_comptime() {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot assign to mutable global `{}` inside comptime context",
                                name,
                            ))
                            .with_code_str("E082")
                            .with_span(*span)
                            .with_help(
                                "comptime code is sandboxed and cannot access mutable global state",
                            ),
                        );
                    }
                    // Mutable globals are also forbidden inside isolate blocks
                    if self.mutable_globals.contains(name) && self.is_in_isolate() {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot assign to mutable global `{}` inside isolate block",
                                name,
                            ))
                            .with_code_str("E093")
                            .with_span(*span)
                            .with_help("isolate blocks must not access external mutable state"),
                        );
                    }
                }
                let (target_hir, target_ty) = self.infer_expr(target, None)?;
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
            Stmt::ComptimeBlock {
                captures,
                body,
                trusted,
                span,
                ..
            } => {
                // Push a comptime context frame so that `return` inside comptime
                // blocks is treated as comptime control flow, not an error.
                self.push_comptime_ctx(ComptimeReason::ComptimeBlock, *span);
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
                            self.deferred_comptime_blocks.push((
                                captures.clone(),
                                hir.clone(),
                                ty,
                                *span,
                            ));
                        } else {
                            // Evaluate the comptime block at compile time.
                            // Pre-collect literal values for captures before any mutable borrow of self.
                            let captured_literals: Vec<(Symbol, Option<ComptimeValue>)> = captures
                                .iter()
                                .map(|(sym, _span)| {
                                    (
                                        *sym,
                                        self.literal_values
                                            .get(sym)
                                            .and_then(|v| v.last())
                                            .cloned(),
                                    )
                                })
                                .collect();
                            // ── Check capture errors BEFORE eval creation ──
                            // so they appear before any comptime evaluation errors.
                            // Iterate over original captures (which carry spans) so
                            // each error can point to the specific capture name.
                            for (capture, capture_span) in captures.iter() {
                                let val = self
                                    .literal_values
                                    .get(capture)
                                    .and_then(|v| v.last())
                                    .cloned();
                                if val.is_some() {
                                    continue;
                                }
                                if let Some(binding) = self
                                    .symbols
                                    .lookup_variable(*capture, crate::ast::Span::new(0, 0))
                                {
                                    if binding.mutable {
                                        self.diagnostics.push(
                                            Diagnostic::error(format!(
                                                "cannot capture mutable variable `{}` in comptime block",
                                                capture,
                                            ))
                                            .with_code_str("E082")
                                            .with_span(*capture_span),
                                        );
                                    } else {
                                        self.diagnostics.push(
                                            Diagnostic::error(format!(
                                                "captured variable `{}` must be a compile-time \
                                                 constant (initializer must be a literal or \
                                                 another comptime-known value)",
                                                capture,
                                            ))
                                            .with_code_str("E082")
                                            .with_span(*capture_span),
                                        );
                                    }
                                } else {
                                    self.diagnostics.push(
                                        Diagnostic::error(format!(
                                            "unknown variable `{}` in comptime capture list",
                                            capture,
                                        ))
                                        .with_code_str("E082")
                                        .with_span(*capture_span),
                                    );
                                }
                            }
                            let outer_tb = {
                                let frames: Vec<&CtxFrame> = self
                                    .region_tree
                                    .iter_frames_rev()
                                    .filter(|f| matches!(f.kind, CtxKind::Comptime))
                                    .collect();
                                format_comptime_traceback_inner(&frames, self.source)
                            };
                            let mut eval =
                                crate::hir::comptime::ComptimeEvalContext::new_with_source(
                                    self.ctx,
                                    self.symbols,
                                    &mut self.diagnostics,
                                    outer_tb,
                                    self.source,
                                );
                            eval.set_trusted(*trusted);
                            // Inject captured literal values into eval context.
                            for (capture, val) in &captured_literals {
                                if let Some(val) = val {
                                    let slot = eval.allocate_slot();
                                    eval.cur_slot.insert(*capture, slot);
                                    eval.variables.insert(slot, val.clone());
                                }
                            }
                            // Register pre-collected comptime functions.
                            for (name, (params, body)) in &self.comptime_fn_registry {
                                eval.register_fn(*name, params.clone(), body.clone());
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
                            captures: captures.clone(),
                            trusted: *trusted,
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
                when_condition,
                span,
            } => {
                let body_hir = self.check_block(body)?;
                // Convert when_condition from AST Expr to HirExpr and
                // validate that the condition is Boolean AND a compile-time
                // predicate (SYNTAX.md: "may reference only ghost variables
                // and other compile-time-constant expressions").
                let when_hir = when_condition
                    .as_ref()
                    .map(|cond| {
                        self.infer_expr(cond, None).and_then(|(h, ty)| {
                            let bool_ty = self.ctx.bool();
                            self.unify_with(bool_ty, ty, *span, TypingContext::None)?;
                            // Reject runtime variable references in the
                            // condition — only literals and comptime constants
                            // are permitted.
                            if contains_runtime_ident(
                                &h,
                                &*self.ghost_var_scopes.borrow(),
                                &*self.runtime_var_scopes.borrow(),
                                &self.local_type_param_cache,
                            ) {
                                return Err(Diagnostic::error(
                                    "`scope_cleanup when` condition must be a compile-time predicate"
                                )
                                .with_help(
                                    "use only literals and comptime constants"
                                )
                                .with_span(*span));
                            }
                            Ok(Box::new(h))
                        })
                    })
                    .transpose()?;
                Ok(HirStmt::ScopeCleanup {
                    name: *name,
                    when_condition: when_hir,
                    body: body_hir,
                    propagates: *propagates,
                    overrides: *overrides,
                    span: *span,
                })
            }
            Stmt::Trigger { name, span } => Ok(HirStmt::Trigger {
                name: *name,
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
                // Extract the variable name from the inner `set mut? name = ...`.
                let var_name = match inner.as_ref() {
                    crate::ast::Stmt::VariableDef { name: Some(n), .. } => Some(*n),
                    _ => None,
                };
                if let Some(name) = var_name {
                    // Push to the current (topmost) scope, or create one.
                    // The fallback is unreachable in practice: `VarScopeGuard`
                    // pushes a ghost frame before any statement is checked.
                    let mut scopes = self.ghost_var_scopes.borrow_mut();
                    if let Some(scope) = scopes.last_mut() {
                        scope.insert(name);
                    } else {
                        debug_assert!(
                            false,
                            "ghost variable declared outside an active variable scope"
                        );
                        let mut scope = HashSet::new();
                        scope.insert(name);
                        scopes.push(scope);
                    }
                }
                // Mark that the inner statement is a ghost definition, so
                // `VariableDef` registers the name as ghost (not runtime).
                self.in_ghost_var_def.set(true);
                let inner_hir = self.check_stmt(inner);
                self.in_ghost_var_def.set(false);
                let inner_hir = inner_hir?;
                Ok(HirStmt::GhostVariableDef {
                    inner: Box::new(inner_hir),
                    span: *span,
                })
            }
            Stmt::Isolate { body, span, .. } => {
                // Push an Isolate context frame so that the body can be
                // verified to not access external mutable state.
                // ── TODO: complete tracking ───────────────────────────
                // The current check only flags calls to @trusted/@io/mutating
                // functions inside the isolate block.  A complete tracking pass
                // should also verify that:
                //   - No writes occur through &mut references from outside
                //   - No `isolate`-captured variables are mutated
                //   - Calls to `unsafe` / extern functions are rejected
                // Full implementation deferred to a follow-up pass.
                // ───────────────────────────────────────────────────────
                self.push_ctx(CtxKind::Isolate, *span, None);
                let body_hir = match self.check_block(body) {
                    Ok(hir) => {
                        self.pop_ctx();
                        Ok(HirStmt::Isolate {
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
            Stmt::LayoutDef {
                name,
                attributes,
                span,
            } => {
                // Layout alias definitions are handled by the resolver.
                // The checker just passes them through.
                Ok(HirStmt::LayoutDef {
                    name: *name,
                    attributes: attributes.clone(),
                    span: *span,
                })
            }
            Stmt::TypeDef { span, .. } => {
                // Type definitions are already handled by the resolver;
                // no additional checking needed here.
                Ok(HirStmt::Stripped { span: *span })
            }
            Stmt::Edition(version, span) => {
                // Edition is validated and stored by the resolver.
                // The checker simply passes it through.
                Ok(HirStmt::Edition(version.clone(), *span))
            }
            Stmt::TraitDef { span, .. } => {
                // Trait definitions are handled by the resolver; skip silently.
                Ok(HirStmt::Stripped { span: *span })
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
                    alias: *alias,
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
                        name: p.name,
                        ty: p_ty,
                        default: p.default.clone(),
                        span: p.span,
                    });
                }
                Ok(HirStmt::ExternFunction {
                    abi: abi.clone(),
                    name: *name,
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
                let (
                    trait_path,
                    for_type,
                    methods,
                    span,
                    attributes,
                    type_params,
                    where_clause,
                    associated_types,
                ) = match stmt {
                    Stmt::ImplBlock {
                        span,
                        trait_path,
                        for_type,
                        methods,
                        attributes,
                        type_params,
                        where_clause,
                        associated_types,
                        ..
                    } => (
                        trait_path,
                        for_type,
                        methods,
                        *span,
                        attributes,
                        type_params,
                        where_clause,
                        associated_types,
                    ),
                    _ => {
                        let msg = format!("check_stmt: expected ImplBlock, got {:?}", stmt);
                        self.diagnostics
                            .push(Diagnostic::error(&msg).with_span(stmt.span()));
                        return Ok(HirStmt::Error);
                    }
                };
                // Collect impl type-param names once — shared by all methods
                // of this impl.  Passed to `check_method_body` so the E104
                // generality check runs per method (methods are generic
                // definitions too).
                let impl_param_names: Vec<Symbol> = type_params.iter().map(|tp| tp.name).collect();
                // Const generic params monomorphize per concrete constant
                // value (SYNTAX.md §Const Generics) — the narrowed E104
                // exemption allows a const param to be bound only to a type
                // consistent with its declared value type.  Mirrors the
                // `Stmt::FunctionDef` arm's const-param handling.
                let const_param_names: Vec<Symbol> = type_params
                    .iter()
                    .filter(|tp| matches!(tp.kind, crate::ast::TypeParamKind::Const { .. }))
                    .map(|tp| tp.name)
                    .collect();
                let mut const_param_value_types: std::collections::HashMap<Symbol, TypeId> =
                    std::collections::HashMap::default();
                for tp in type_params.iter() {
                    if let crate::ast::TypeParamKind::Const { ty, .. } = &tp.kind {
                        if let Ok(value_ty) = self.resolve_type(ty) {
                            const_param_value_types.insert(tp.name, value_ty);
                        }
                    }
                }
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
                    // ── @experimental check ───────────────────────────
                    for attr in &trait_binding.attributes {
                        if attr.name.eq_str("experimental") && !self.enable_experimental {
                            let trait_name = match tp.as_ref() {
                                Type::Path(p, _) => p
                                    .last()
                                    .map(|s| s.as_str().to_string())
                                    .unwrap_or_else(|| "?".to_string()),
                                _ => "?".to_string(),
                            };
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "use of experimental trait `{}`",
                                    trait_name,
                                ))
                                .with_code_str("E094")
                                .with_span(span)
                                .with_help("experimental features are not enabled; use `--enable-experimental` to use this trait"),
                            );
                        }
                    }

                    // Register generic type parameters so `T` in `impl<T> Foo for T` resolves
                    // (names were collected at the impl-block arm top in
                    // `impl_param_names` for post-impl cleanup).
                    for (i, tp) in type_params.iter().enumerate() {
                        let generic_id = self.ctx.generic_param(i, tp.name);
                        self.local_type_param_cache.insert(tp.name, generic_id);
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
                                Diagnostic::error_kind(DiagnosticKind::ImplMissingMethod {
                                    trait_name: Self::type_to_string(tp.as_ref()),
                                    method_name: tm_name.to_string(),
                                    impl_span: span,
                                    trait_span: trait_binding.span,
                                })
                                .with_code_str("E101")
                                .with_help("every trait method must be implemented — add a `def` for it in this impl block"));
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
                            && m.params.len() != trait_sig.params.len()
                        {
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

                        method_infos.push(crate::hir::traits::MethodInfo {
                            name: m.name,
                            param_tys,
                            ret_ty,
                            span: m.span,
                            attributes: m.attributes.clone(),
                            has_auto_deref: auto_deref,
                        });
                        // Type-check the method body (methods are
                        // functions — same gating and body checking).
                        self.check_method_body(
                            m,
                            &for_type,
                            &impl_param_names,
                            &const_param_names,
                            &const_param_value_types,
                        )?;
                    }

                    // Populate the associated types (`type Target = ...`)
                    // so deref coercions (`try_deref_trait_step`) can find
                    // the `Target` through the impl.
                    let assoc_tys = associated_types
                        .iter()
                        .map(|at| {
                            let ty = match &at.default {
                                Some(d) => {
                                    let resolved = self.resolve_self_ty(d, self_ty);
                                    self.resolve_type(&resolved)?
                                }
                                None => self.ctx.error(),
                            };
                            Ok((at.name, ty))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let candidate = crate::hir::traits::ImplCandidate {
                        trait_id,
                        for_type: for_ty,
                        methods: methods.clone(),
                        resolved_methods: method_infos.clone(),
                        assoc_tys,
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
                                    let param_id = self.ctx.generic_param(i, tp.name);
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
                            if let Some(tp) = &trait_path
                                && let Type::Generic(_, generic_args, _) = tp.as_ref()
                            {
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
                                                        if let GenericArg::Positional(ty) = arg
                                                            && let Ok(resolved) =
                                                                self.resolve_type(ty)
                                                        {
                                                            bound_args.push(resolved);
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
                                    let param_id = self.ctx.generic_param(i, tp.name);
                                    for bound in &tp.bounds {
                                        if let Some(trait_id) = self.resolve_trait_path(bound) {
                                            let mut bound_args = Vec::new();
                                            if let Type::Generic(_, args, _) = bound {
                                                for arg in args {
                                                    if let GenericArg::Positional(ty) = arg
                                                        && let Ok(resolved) = self.resolve_type(ty)
                                                    {
                                                        bound_args.push(resolved);
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
                            name: m.name,
                            param_tys,
                            ret_ty,
                            span: m.span,
                            attributes: m.attributes.clone(),
                            has_auto_deref: auto_deref,
                        });
                        // Type-check the method body (methods are
                        // functions — same gating and body checking).
                        self.check_method_body(
                            m,
                            &for_type,
                            &impl_param_names,
                            &const_param_names,
                            &const_param_value_types,
                        )?;
                    }
                    self.trait_env
                        .add_inherent_methods(for_def_id, method_infos);
                    // ── Clean up generic parameter cache for inherent impl ──
                    // Symmetric with the trait-impl cleanup above — the impl's
                    // type params must not leak into subsequent top-level items.
                    for name in &impl_param_names {
                        self.local_type_param_cache.remove(name);
                    }
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

    /// Type-check an impl method's body — methods are functions, so they
    /// get the same `@auto_ro`/`@auto_coerce` gating (incl. the
    /// `@trusted`/Strict Mode rejection) and body checking as a function
    /// definition.  (Previously method bodies were only registered as
    /// signatures — they were never type-checked.)
    fn check_method_body(
        &mut self,
        m: &crate::ast::ImplMethod,
        self_ty: &crate::ast::Type,
        impl_param_names: &[Symbol],
        const_param_names: &[Symbol],
        const_param_value_types: &std::collections::HashMap<Symbol, TypeId>,
    ) -> Result<(), Diagnostic> {
        let Some(body) = &m.body else {
            return Ok(());
        };
        // Method bodies are function bodies — the same per-function state
        // management applies (frozen-vars scope, GADT-seal violation
        // counter, `@must_handle` source tracking).  See `check_stmt`'s
        // `Stmt::FunctionDef` arm.
        let prev_must_handle = self.must_handle_sources.borrow().clone();
        self.ctx.frozen_vars.borrow_mut().clear();
        self.ctx.seal_violations.set(0);
        let has_auto_ro = m.attributes.iter().any(|a| a.name.eq_str("auto_ro"));
        let has_auto_coerce = m.attributes.iter().any(|a| a.name.eq_str("auto_coerce"));
        let has_trusted = m.attributes.iter().any(|a| a.name.eq_str("trusted"));
        let prev_auto_ro = self.ctx.auto_ro.get();
        let prev_auto_coerce = self.ctx.auto_coerce.get();
        self.ctx
            .auto_ro
            .set(has_auto_ro && !self.strict_mode && !has_trusted);
        self.ctx
            .auto_coerce
            .set(has_auto_coerce && !self.strict_mode && !has_trusted);
        if has_auto_ro || has_auto_coerce {
            if self.strict_mode {
                self.diagnostics.push(
                    Diagnostic::error("`@auto_ro`/`@auto_coerce` is not permitted in Strict Mode")
                        .with_span(m.span),
                );
            } else if has_trusted {
                self.diagnostics.push(
                    Diagnostic::error(
                        "`@auto_ro`/`@auto_coerce` is not permitted on `@trusted` functions",
                    )
                    .with_span(m.span),
                );
            }
        }
        self.push_ctx(CtxKind::Function, m.span, None);
        let _scope = self.enter_var_scope();
        for p in &m.params {
            let ty = match &p.ty {
                Some(t) => {
                    let resolved = self.resolve_self_ty(t, self_ty);
                    self.resolve_type(&resolved)?
                }
                None => self.ctx.error(),
            };
            self.local_variable_types.insert(p.name, ty);
            self.span_insert(p.name, p.span);
        }
        let ret_ty = {
            let resolved = self.resolve_self_ty(&m.return_type, self_ty);
            self.resolve_type(&resolved)?
        };
        let prev_return = self.current_return_type;
        self.current_return_type = Some(ret_ty);
        let prev_function = self.current_function;
        self.current_function = Some(DefId(0));
        let result = self.check_block(body);
        self.current_function = prev_function;
        self.current_return_type = prev_return;
        // ── E104 generality check ──────────────────────────────────────
        // Methods are generic definitions too: the impl's type params must
        // not be specialized to a concrete type by the method body (the
        // body must type-check for EVERY instantiation).  Same discipline
        // as the `Stmt::FunctionDef` arm's E104 check.
        for name in impl_param_names {
            let Some(&tid) = self.local_type_param_cache.get(name) else {
                continue;
            };
            let resolved = self.ctx.resolve_binding(tid);
            // Narrowed const-param exemption: a const param may only be
            // bound to a type consistent with its declared VALUE type (the
            // monomorphization).  An unrelated concrete type (e.g. N := Bool
            // for `const N: usize`) is a generality violation.
            if const_param_names.contains(name) {
                if let Some(value_ty) = const_param_value_types.get(name) {
                    // Probe WITHOUT committing (transactional).
                    let depth = self.ctx.transaction_depth();
                    self.ctx.begin_transaction();
                    let compatible = self.ctx.try_unify(resolved, *value_ty, None).is_ok();
                    self.ctx.rollback_to(depth);
                    if compatible {
                        continue;
                    }
                }
            }
            let violation = resolved != tid
                && !matches!(
                    self.ctx.get_raw(resolved),
                    TypeData::InferVar { .. } | TypeData::SkolemVar { .. } | TypeData::Error
                );
            if violation {
                let origin = self.ctx.generic_binding_origins.borrow().get(&tid).copied();
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "generic parameter `{}` is constrained to a specific type by the method body; the body must type-check for every instantiation",
                        name,
                    ))
                    .with_code_str("E104")
                    .with_span(origin.unwrap_or(m.span)),
                );
            }
        }
        self.pop_ctx();
        self.ctx.auto_ro.set(prev_auto_ro);
        self.ctx.auto_coerce.set(prev_auto_coerce);
        // Restore the `@must_handle` source tracking — a method body's
        // `@must_handle` assignments must not leak into sibling methods.
        *self.must_handle_sources.borrow_mut() = prev_must_handle;
        // Emit a diagnostic for nonzero seal-violation count: a GADT arm
        // skipped a GenericParam binding inside the method body and never
        // recovered.  Same pattern as the `Stmt::FunctionDef` arm.
        if self.ctx.seal_violations.get() > 0 {
            let diag = if self.strict_mode {
                Diagnostic::error("internal: GADT arm seal violations were not recovered")
            } else {
                Diagnostic::warning("internal: GADT arm seal violations were not recovered")
            };
            self.diagnostics.push(diag.with_span(m.span));
        }
        result.map(|_| ())
    }

    fn check_block(&mut self, stmts: &[Stmt]) -> Result<Vec<HirStmt>, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_block(stmts)
    }

    fn infer_expr(
        &mut self,
        expr: &Expr,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr, TypeId), Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.infer_expr(expr, expected)
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
        let exist_depth = self.ctx.gadt.exist_skolems.borrow().len();
        let mut fc = FnCtxt::new(self);
        fc.check_pattern(pattern, expected_ty, exist_depth)
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
                            GenericArg::Named(*n, self.resolve_self_ty(t, self_ty))
                        }
                        GenericArg::Const(ac) => {
                            // Const generic args: clone as-is (self-type doesn't affect
                            // const expressions at this stage).
                            GenericArg::Const(crate::ast::AnonConst {
                                value: ac.value.clone(),
                                span: ac.span,
                            })
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
                assoc_name: *assoc_name,
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
        if path.len() == 1
            && let Some(&def_id) = self.resolution_map.type_def_ids.get(&path[0])
        {
            return Ok(def_id);
        }
        // Check if this is a generic type parameter (e.g. `T` in `def foo<T>(x: T)`)
        if path.len() == 1 && self.local_type_param_cache.contains_key(&path[0]) {
            // Return a sentinel DefId to signal "this is a generic param, not a concrete type"
            // The caller (resolve_type) will handle this by looking up local_type_param_cache.
            return Ok(DefId(usize::MAX - 1));
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
    fn suggest_cast(&self, expected: TypeId, actual: TypeId) -> Option<Suggestion> {
        let (e, a) = (self.ctx.get(expected), self.ctx.get(actual));
        let msg = match (e, a) {
            (TypeData::Int { .. }, TypeData::Float { .. })
            | (TypeData::Float { .. }, TypeData::Int { .. }) => {
                Some("try using `as` to cast between integer and float types")
            }
            (TypeData::Bool, TypeData::Int { .. }) => Some("try `x != 0` to convert Int to Bool"),
            (TypeData::Int { .. }, TypeData::Bool) => {
                Some("try `if x { 1 } else { 0 }` to convert Bool to Int")
            }
            _ => None,
        };
        msg.map(|m| Suggestion {
            message: m.into(),
            applicability: Applicability::MaybeIncorrect,
            style: SuggestionStyle::ShowAlways,
        })
    }

    /// Generate a human-readable reason for a type mismatch between two
    /// types, explaining *why* they are incompatible.  Returns `None` when
    /// no specific reason can be determined (the types are simply different).
    pub(crate) fn type_mismatch_reason(&self, expected: TypeId, actual: TypeId) -> Option<String> {
        let (e, a) = (self.ctx.get(expected), self.ctx.get(actual));
        match (e, a) {
            (
                TypeData::Int {
                    bits: eb,
                    signed: es,
                    ..
                },
                TypeData::Int {
                    bits: ab,
                    signed: as_,
                    ..
                },
            ) => {
                if eb != ab {
                    Some(format!("Int<{eb}> is not the same width as Int<{ab}>"))
                } else if es != as_ {
                    Some(format!(
                        "signed Int<{eb}> is not the same as unsigned Int<{eb}>"
                    ))
                } else {
                    None
                }
            }
            (TypeData::Int { .. }, TypeData::UInt { bits: ab, .. }) => Some(format!(
                "signed integer is not the same as unsigned UInt<{ab}>"
            )),
            (TypeData::UInt { bits: eb, .. }, TypeData::Int { .. }) => Some(format!(
                "unsigned UInt<{eb}> is not the same as signed integer"
            )),
            (TypeData::UInt { bits: eb, .. }, TypeData::UInt { bits: ab, .. }) => {
                if eb != ab {
                    Some(format!("UInt<{eb}> is not the same width as UInt<{ab}>"))
                } else {
                    None
                }
            }
            (TypeData::Float { bits: eb }, TypeData::Float { bits: ab }) => {
                if eb != ab {
                    Some(format!("Float<{eb}> is not the same width as Float<{ab}>"))
                } else {
                    None
                }
            }
            (TypeData::Ref { .. }, _) => {
                Some("a reference type is not a value type; try dereferencing with `*`".into())
            }
            (_, TypeData::Ref { .. }) => Some(
                "expected a reference type, but got a value type; try taking a reference with `&`"
                    .into(),
            ),
            _ => None,
        }
    }

    fn unify(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
        context: TypeCtx,
    ) -> Result<(), Diagnostic> {
        self.ctx
            .unify_tracked(expected, actual, span)
            .map(|_| ())
            .map_err(|_err| {
                let reason = self.type_mismatch_reason(expected, actual);
                let mut diag = Diagnostic::error_kind(DiagnosticKind::TypeMismatch {
                    expected: format!("{:?}", self.ctx.get(expected)),
                    found: format!("{:?}", self.ctx.get(actual)),
                    span,
                    found_span: None,
                    reason,
                    context: Some(context),
                })
                .with_code_str("E030");
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion.message);
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
            .unify_tracked(expected, actual, span)
            .map(|_| ())
            .map_err(|_err| {
                let expected_str = self
                    .ctx
                    .get(expected)
                    .display_with(self.ctx, Some(self.symbols));
                let actual_str = self
                    .ctx
                    .get(actual)
                    .display_with(self.ctx, Some(self.symbols));
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
                    diag = diag.with_suggestion(suggestion.message);
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
        // ── Resolve operands through bindings AND the GADT fact registry ──
        // before kind checks / trait obligations.  Without this, a refined
        // generic param (e.g. `x : T` where the arm's facts say T → Int<32>)
        // would register a `T: Add` obligation that is solved LATER at
        // function end (after the arm's facts are popped), binding
        // T := Int<32> into the global table — a seal leak that the
        // generality check (E104) then surfaces as a false rejection.  The
        // call path already resolves arguments through facts this way.
        let left = self.ctx.resolve_binding(left);
        let right = self.ctx.resolve_binding(right);
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
            if self.ctx.resolve_binding(var_ty) == resolved
                && let Some(def_span) = self.span_get(&sym)
            {
                return Some((sym, def_span));
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
                format!(
                    "{}",
                    self.ctx
                        .get(resolved_other)
                        .display_with(self.ctx, Some(self.symbols))
                )
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
                    d.labels_mut().push(Label::secondary(os, other_type_str));
                }
                // Add a note label for the "maybe_var" operand (the infer var).
                if let Some(ms) = maybe_var_span {
                    d.labels_mut()
                        .push(Label::secondary(ms, "expected integer type"));
                }
                // Trace the type origin: if the "other" operand's type came from
                // a variable definition, show where it originated.
                if let Some((_origin_name, origin_span)) = self.resolve_type_origin(resolved_other)
                {
                    if origin_span != other_span.unwrap_or(origin_span) {
                        d.labels_mut()
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
                            d.labels_mut().push(Label::help(
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
            && let Some(result_id) = self.known_def_id(Symbol::intern("Result"))
            && *did == result_id
            && args.len() == 2
        {
            return Some(args[0]);
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
            && let Some(result_id) = self.known_def_id(Symbol::intern("Result"))
            && *did == result_id
            && args.len() == 2
        {
            return Ok((args[0], args[1]));
        }
        Err(Diagnostic::error("catch requires Result type").with_span(span))
    }

    fn known_def_id(&self, name: Symbol) -> Option<DefId> {
        self.symbols.lookup_type(name).map(|b| b.def_id)
    }

    /// Emit the canonical `return Err(...)` lint (E008).  Shared by the
    /// `Stmt::Return` path and the `Expr::LeaveWith { is_return: true }`
    /// path so the message, code, and help cannot drift between them, and
    /// both push the diagnostic and CONTINUE type-checking (uniform error
    /// recovery — follow-on errors are still reported).
    pub(crate) fn emit_return_err_lint(&mut self, span: Span) {
        self.diagnostics.push(
            Diagnostic::error("`return Err(...)` is not valid; use `leave with` instead")
                .with_code_str("E008")
                .with_span(span)
                .with_help(
                    "`leave with` is the only valid error exit in Posita \
                     (SYNTAX.md §\"Error Handling\"); it is recorded as an \
                     `ErrorExit` in the control-flow graph for audit",
                )
                .with_suggestion(
                    "write `leave with error_value;` instead of `return Err(error_value);`",
                ),
        );
    }

    /// Unified `return Err(e)` / `leave with Err(e)` lint (SYNTAX.md
    /// §"Error Handling"): is this enum literal an `Err` constructor of
    /// `Result` (or of an alias of `Result`)?  Follows the alias chain of
    /// the enum path, so multi-level aliases (`type A = Result<...>`;
    /// `type B = A`) cannot bypass the lint.  Used by BOTH the
    /// `Stmt::Return` path and the `Expr::LeaveWith` path so the two
    /// implementations cannot drift semantically.
    pub(crate) fn is_result_err_constructor(&self, path: &[Symbol], variant: &Symbol) -> bool {
        if !variant.eq_str("Err") || path.is_empty() {
            return false;
        }
        // Check the LAST path segment: `HirExpr::EnumLit` stores the TYPE
        // path (without the variant name), so `Result::Err` has path=`[Result]`
        // and `core::Result::Err` has path=`[core, Result]`.  Checking only
        // `path[0]` would miss multi-segment paths like `core::Result::Err`.
        let mut current = path[path.len() - 1];
        if current.eq_str("Result") {
            return true;
        }
        // Bounded alias-chain walk: cyclic aliases are rejected by the
        // resolver, but keep the lint robust against pathological input.
        const MAX_ALIAS_DEPTH: usize = 64;
        let mut depth = 0;
        loop {
            let binding = match self.symbols.lookup_type(current) {
                Some(b) => b,
                None => return false,
            };
            let alias_ast = match binding.alias_ast.as_ref() {
                Some(t) => t,
                None => return false,
            };
            match alias_ast {
                Type::Path(p, _) if p.len() == 1 => {
                    if p[0].eq_str("Result") {
                        return true;
                    }
                    current = p[0];
                }
                Type::Generic(base, _, _) => match &**base {
                    Type::Path(p, _) if p.len() == 1 => {
                        if p[0].eq_str("Result") {
                            return true;
                        }
                        current = p[0];
                    }
                    _ => return false,
                },
                _ => return false,
            }
            depth += 1;
            if depth > MAX_ALIAS_DEPTH {
                return false;
            }
        }
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
            if cand.trait_id == deref_trait_id
                && cand.has_auto_deref
                && let Some(target_ty) = cand
                    .assoc_tys
                    .iter()
                    .find(|(name, _)| name.eq_str("Target"))
                    .map(|(_, ty)| *ty)
            {
                return Some(target_ty);
            }
        }
        // Also try DerefMut: same Target as Deref
        let deref_mut_id = self
            .symbols
            .lookup_trait(Symbol::intern("DerefMut"))
            .map(|b| b.def_id);
        if let Some(deref_mut_id) = deref_mut_id {
            for cand in &candidates {
                if cand.trait_id == deref_mut_id
                    && cand.has_auto_deref
                    && let Some(target_ty) = self
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
            let unify_ok = self
                .ctx
                .unify_tracked(substituted_ret, exp_ty, span)
                .is_ok();
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
        let type_name = format!("{:?}", self.ctx.get(ty));
        let mut diag = Diagnostic::error_kind(DiagnosticKind::NoSuchField {
            field_name: name.to_string(),
            type_name,
            span,
        })
        .with_code_str("E010");

        // If we found the type definition, show where it was defined
        if let TypeData::Adt { def_id, .. } = self.ctx.get(ty)
            && let Some(binding) = self.symbols.lookup_type_by_def_id(*def_id)
        {
            diag = diag.with_label(binding.span, "type defined here");
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
        // Resolve bindings and dereference `&T` before checking attributes:
        // `s'len` on `s: &[T]` must look at the pointee `[T]`.
        let mut base = self.ctx.resolve_binding(ty);
        if matches!(self.ctx.get(base), TypeData::Ref { .. })
            && let TypeData::Ref { ty: inner, .. } = self.ctx.get(base)
        {
            base = self.ctx.resolve_binding(*inner);
        }
        if name.eq_str("len")
            && (self.ctx.is_array(base)
                || self.ctx.is_slice(base)
                || base == self.ctx.builtin_str
                || base == self.ctx.builtin_str_ref)
        {
            Ok(self.ctx.usize())
        } else if name.eq_str("size")
            && (self.ctx.is_integer(base) || self.ctx.is_float(base) || self.ctx.is_pointer(base))
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
        if let Some(def_id) = def_id
            && let Some(binding) = self.symbols.lookup_type_by_def_id(def_id)
        {
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
        if let Type::Literal(expr, _) = ty
            && let Expr::Literal(Literal::Int(val), _) = expr.as_ref()
        {
            if *val > 64 {
                return None; // reject out-of-range bit widths silently
            }
            return Some(*val as u8);
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

    /// Check whether a GADT variant's `when` constraints are satisfiable
    /// given the scrutinee's concrete type arguments.  Used for dead variant
    /// elimination before GADT refinement.
    pub fn is_gadt_variant_reachable(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
    ) -> bool {
        let (binding, vd, args) = match self.resolve_gadt_variant_info(scrut_ty, pattern, span) {
            Some(info) => info,
            None => return true,
        };
        if vd.eq_spec.is_empty() {
            return true;
        }
        // For existential GADT variants, create fresh InferVars for each
        // exist param and check if the constraints are satisfiable (using
        // a transaction to avoid persistent side effects).  Previously this
        // returned `true` unconditionally, which was unsound: a variant
        // with `when T == [X]` is NOT reachable when `T = Int<32>`.
        //
        // NOTE: we deliberately do NOT restore the inference var cursor
        // after probing.  The probe InferVars allocate TypeIds in the type
        // arena which cannot be freed; restoring `next_var_id` would make a
        // later real variable reuse the same var_id, breaking the
        // "var_id is unique" invariant.  The ids are monotonically
        // increasing; the small id waste per reachability check is
        // acceptable.
        if !vd.exists_params.is_empty() {
            let mut exist_vars: Vec<TypeId> = Vec::new();
            for _ep in &vd.exists_params {
                let var = self.infer.new_type_var(
                    &mut self.ctx,
                    crate::hir::infer::TypeVariableKind::Any,
                    crate::hir::infer::VarOrigin::Synthetic,
                );
                exist_vars.push(var);
            }
            // Use a single transaction for all constraints so that
            // shared substitutions are preserved across the conjunction.
            // Also save/restore `unify_seen` (same discipline as `can_unify`)
            // so the cycle-detection cache is not left in a partial state.
            let saved_seen = self.ctx.save_unify_seen();
            self.ctx.begin_transaction();
            let mut all_satisfied = true;
            for (pn, ct) in &vd.eq_spec {
                let declared = self.resolve_type_with_skolems(ct, &vd.exists_params, &exist_vars);
                let Some(declared) = declared else {
                    all_satisfied = false;
                    break;
                };
                if let Some((param_idx, _)) = binding
                    .params
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.name == *pn)
                {
                    if let Some(actual) = args.get(param_idx).copied()
                        && self.ctx.try_unify(declared, actual, None).is_err()
                    {
                        all_satisfied = false;
                        break;
                    }
                } else if let Some(&var) = vd
                    .exists_params
                    .iter()
                    .position(|p| p == pn)
                    .and_then(|i| exist_vars.get(i))
                    && self.ctx.try_unify(declared, var, None).is_err()
                {
                    all_satisfied = false;
                    break;
                }
            }
            self.ctx.rollback_transaction();
            self.ctx.restore_unify_seen(saved_seen);
            return all_satisfied;
        }
        // Non-existential GADT reachability: use a single transaction
        // across all constraints so that shared substitutions are preserved.
        // Also save/restore `unify_seen` (same discipline as `can_unify`).
        let saved_seen = self.ctx.save_unify_seen();
        self.ctx.begin_transaction();
        let mut all_satisfied = true;
        for (pn, ct) in &vd.eq_spec {
            let Some((i, _)) = binding
                .params
                .iter()
                .enumerate()
                .find(|(_, p)| p.name == *pn)
            else {
                continue;
            };
            let Some(actual) = args.get(i).copied() else {
                continue;
            };
            // If the constraint type cannot be resolved, treat the variant
            // as SATISFIED (reachable) — the SAFE direction for
            // exhaustiveness: requiring the user to handle a variant is
            // conservative, while silently excluding it could accept a
            // non-exhaustive match.  This path should be unreachable (the
            // resolver already validated the RHS), so assert it in debug
            // builds to catch future regressions.
            let Ok(declared) = self.resolve_type(ct) else {
                debug_assert!(
                    false,
                    "unreachable: resolve_type failed on a GADT `when` RHS in is_gadt_variant_reachable"
                );
                all_satisfied = true;
                break;
            };
            if self.ctx.try_unify(declared, actual, None).is_err() {
                all_satisfied = false;
                break;
            }
        }
        self.ctx.rollback_transaction();
        self.ctx.restore_unify_seen(saved_seen);
        all_satisfied
    }

    /// Shared lookup: given a pattern and scrutinee type, find the matching
    /// GADT variant and its type arguments.  Returns the `TypeBinding` (owned),
    /// the variant definition, and the resolved type arguments.
    /// Returns `None` for non-enum patterns or when lookups fail.
    /// Used by `is_gadt_variant_reachable` and `apply_gadt_refinement`.
    fn resolve_gadt_variant_info(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
    ) -> Option<(TypeBinding, crate::ast::EnumVariant, Vec<TypeId>)> {
        let variant_name = match pattern {
            crate::ast::Pattern::Enum { variant, .. } => *variant,
            _ => return None,
        };
        let binding = self.lookup_type_binding(scrut_ty)?;
        let vd = binding
            .variants
            .iter()
            .find(|v| v.name == variant_name)?
            .clone();
        let (_, args) = self.resolve_type_to_struct_or_enum(scrut_ty, span).ok()?;
        Some((binding, vd, args))
    }

    /// Look up the TypeBinding for a Struct or Enum type, if available.
    fn lookup_type_binding(&self, ty: TypeId) -> Option<TypeBinding> {
        let resolved = self.ctx.resolve_binding(ty);
        match self.ctx.get(resolved) {
            TypeData::Adt { def_id, .. } => self.symbols.lookup_type_by_def_id(*def_id).cloned(),
            _ => None,
        }
    }

    /// Register GADT equalities for a pattern arm using the GADT equality
    /// registry.
    ///
    /// For each `when` clause constraint, registers the scrutinee's type
    /// argument → concrete type mapping in the scoped GADT registry.
    /// `resolve_binding` then transparently sees the refinement within
    /// the arm.
    ///
    /// If the matched variant has existentially quantified type variables
    /// (`exists X`), this method creates fresh skolem variables for them
    /// and substitutes them into the `when` constraint types.
    ///
    /// The caller must have called `ctx.push_gadt_arm()` before this
    /// method and `ctx.pop_gadt_arm()` after processing the arm body.
    ///
    /// See `TypeContext::push_gadt_arm` for the OCaml `unify_gadt`
    /// reference (ctype.ml:3926-3949).
    /// Pre-create existential skolem variables for a GADT pattern arm
    /// BEFORE check_pattern runs.  This ensures that check_pattern_inner
    /// and apply_gadt_refinement use the SAME skolem TypeIds for
    /// existentially quantified type variables, avoiding the soundness
    /// gap of having two independent witness sets.
    /// Always clears stale skolems first.
    /// Reset the arm-scoped `pending_eqs` accumulator before processing a
    /// new GADT match arm.  Stale equalities from a previous (possibly
    /// unreachable or error-aborted) arm must not leak into the current
    /// arm — `pending_eqs` is only consumed by `apply_gadt_refinement`
    /// when the current arm is reachable (cross-arm contamination).
    ///
    /// The existential-frame stack is deliberately NOT cleared here: outer
    /// arms' witnesses must survive nested arms (SYNTAX.md — witnesses are
    /// "kept distinct" within their branch).  The current arm's own frames
    /// are popped when its arm guard drops.
    pub(crate) fn clear_pending_eqs(&mut self) {
        self.ctx.gadt.pending_eqs.clear();
    }

    pub(crate) fn precreate_exist_skolems(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
    ) {
        // NOTE: the arm-scoped `pending_eqs` reset is an explicit separate
        // step (`clear_pending_eqs`) — this helper advertises only what its
        // name says (skolem precreation), no hidden side effects.
        let info = self.resolve_gadt_variant_info(scrut_ty, pattern, span);
        let (binding, vd, _) = match info {
            Some(i) => i,
            None => return,
        };
        if vd.exists_params.is_empty() {
            return;
        }
        // Skolem identity is the binder's INDEX in `exists_params`, not its
        // name (GHC `realUnique` / OCaml `id: int`): allocate one fresh
        // skolem per exist param, in declaration order.  The frame carries
        // the VARIANT name as its identity so `check_pattern_inner` reuses
        // this frame for the same top-level variant (payload and GADT
        // equations share ONE witness set) and only pushes new frames for
        // nested existential variants.
        let skolems: Vec<TypeId> = vd
            .exists_params
            .iter()
            .map(|_| self.ctx.fresh_gadt_skolem())
            .collect();
        self.ctx
            .gadt
            .exist_skolems
            .borrow_mut()
            .push(ExistScopeFrame {
                def_id: binding.def_id,
                variant_name: vd.name,
                used: false,
                skolems,
            });
    }

    /// Shared GADT arm setup for all four pattern-matching sites
    /// (`Expr::Match` arms, `Expr::IfLet`, `Stmt::IfLet`, `Stmt::WhileLet`):
    /// pre-create exist skolems, check the pattern, run dead-variant
    /// elimination, push the GADT arm + apply refinement, and create the
    /// RAII guard.  The caller owns the returned guard (scoped to its body
    /// check) and pops explicitly on the normal path; the guard pops on
    /// early return.  Centralizing this sequence keeps the four sites from
    /// drifting (previously `Stmt::IfLet` silently lacked it).
    fn begin_gadt_arm(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
    ) -> Result<(HirPattern, bool, GadtArmGuard), Diagnostic> {
        // Record the existential-frame depth BEFORE this arm's frames are
        // pushed — outer arms' witnesses must survive nested arms, and the
        // guard truncates the stack back to this depth on drop.
        let exist_depth = self.ctx.gadt.exist_skolems.borrow().len();
        // RAII: any error between the precreate push and the caller's own
        // truncation restores the stack (a future error-prone step added
        // in this gap cannot leak frames).  The success path commits.
        let mut _precreate_guard = PrecreateGuard::enter(&self.ctx, exist_depth);
        // Reset the arm-scoped pending-eq accumulator, then pre-create
        // exist skolems BEFORE check_pattern.
        self.clear_pending_eqs();
        self.precreate_exist_skolems(scrut_ty, pattern, span);
        let pattern_hir = match self.check_pattern(pattern, scrut_ty) {
            Ok(h) => h,
            Err(e) => {
                // The guard's Drop truncates to `exist_depth`.
                return Err(e);
            }
        };
        _precreate_guard.commit();
        // GADT dead variant elimination (before refinement).
        let gadt_reachable = self.is_gadt_variant_reachable(scrut_ty, pattern, span);
        // ── Top-level unreachable GADT arm warning ─────────────────
        // A GADT variant that is deterministically unreachable (its `when`
        // constraints conflict with the scrutinee's type arguments) is
        // silently excluded from exhaustiveness — warn so the user notices
        // a likely logic error (wrong variant or wrong type argument).
        // `is_gadt_variant_reachable` only returns false on a CONCRETE
        // constraint conflict (resolution failure defaults to reachable),
        // so this does not false-positive on uncertain cases.
        if !gadt_reachable
            && let Some((_, vd, _)) = self.resolve_gadt_variant_info(scrut_ty, pattern, span)
            && !vd.eq_spec.is_empty()
        {
            self.diagnostics.push(
                Diagnostic::warning(format!(
                    "GADT variant '{}' is unreachable: its `when` constraints are incompatible with the scrutinee type",
                    vd.name,
                ))
                .with_span(span),
            );
        }
        // Save depth BEFORE push so the guard can pop on early return.
        let saved_depth = self.ctx.gadt.arm_depth.get();
        if gadt_reachable {
            self.ctx.push_gadt_arm();
            self.apply_gadt_refinement(scrut_ty, pattern, span);
        }
        // Guard ensures pop on early return (using pre-push depth), and
        // restores the TcLevel region entered for the arm.  The frames stay
        // on the stack — the guard truncates to `exist_depth` on drop, so
        // nested arms cannot destroy an outer arm's witnesses.
        let guard = GadtArmGuard::enter_region(self.ctx, &mut self.infer, saved_depth, exist_depth);
        Ok((pattern_hir, gadt_reachable, guard))
    }

    /// Run `body` inside a GADT arm scope: `enter_var_scope` +
    /// `begin_gadt_arm`, then the arm body, then pop the refinement BEFORE
    /// returning (so arm-local equalities don't leak into cross-arm
    /// unification or the else-branch), and restore the TcLevel region.
    /// The `GadtArmGuard`'s Drop still handles early-return paths
    /// (idempotent depth check + region flag).
    fn with_gadt_arm<T>(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
        body: impl FnOnce(&mut Self, bool) -> Result<T, Diagnostic>,
    ) -> Result<(HirPattern, T, bool), Diagnostic> {
        let _scope = self.enter_var_scope();
        let (p, gadt_reachable, mut guard) = self.begin_gadt_arm(scrut_ty, pattern, span)?;
        let t = body(self, gadt_reachable)?;
        // Pop the refinement BEFORE the caller continues (cross-arm
        // unification / else-branch) so this arm's equalities do not leak.
        // The guard's idempotent depth check handles the early-return path.
        if gadt_reachable {
            self.ctx.pop_gadt_arm();
        }
        // Restore the TcLevel region entered by the arm guard (sets the
        // flag so the guard's Drop does not restore a second time).
        guard.restore_region();
        Ok((p, t, gadt_reachable))
    }

    pub fn apply_gadt_refinement(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: Span,
    ) {
        // begin_gadt_arm guarantees precreate_exist_skolems ran first,
        // which pushes the existential frame for this variant exactly when
        // it has existential params.  Assert the invariant instead of
        // silently pushing a frame that no code would ever pop (the old
        // fallback was load-bearing-by-accident: it could never fire on the
        // begin_gadt_arm path, but a future reordering would leak the
        // unpopped frame into the next arm's precreate reuse check).
        debug_assert!(
            !self.ctx.gadt.exist_skolems.borrow().is_empty()
                || self
                    .resolve_gadt_variant_info(scrut_ty, pattern, span)
                    .map_or(true, |(_, vd, _)| vd.exists_params.is_empty()),
            "apply_gadt_refinement: precreate_exist_skolems must have pushed the existential frame"
        );
        // Register ALL collected `when` equalities — the top-level variant
        // (collected by `check_pattern_inner` for the scrutinee's pattern)
        // plus every nested GADT constructor in the pattern tree (nested
        // GADT refinement, SYNTAX.md §"Nested GADT Refinement").  This runs
        // after `push_gadt_arm`, so the GADT fact registry has an active
        // arm to write into.
        let pending = std::mem::take(&mut self.ctx.gadt.pending_eqs);
        for eq in pending {
            // Nested GADT satisfiability gate (see audit #3): an impossible
            // nested constructor must not contribute equalities to the arm.
            // Report a warning (aggregated in DiagCtxt's `unreported` list,
            // flushed together) so the user sees the unreachable nested
            // constructor — same severity as the top-level unreachable warn.
            if !self.nested_eq_satisfiable(
                &eq.binding,
                &eq.args,
                &eq.param_name,
                &eq.concrete_ty,
                &eq.exist_params,
                &eq.skolems,
                span,
            ) {
                self.diagnostics.push(
                    Diagnostic::warning(format!(
                        "nested GADT constructor is unreachable: `when {} == ...` is incompatible with the inner scrutinee type",
                        eq.param_name,
                    ))
                    .with_span(eq.concrete_ty.span()),
                );
                continue;
            }
            self.register_single_gadt_eq(
                &eq.param_name,
                &eq.concrete_ty,
                &eq.binding,
                &eq.args,
                &eq.exist_params,
                &eq.skolems,
            );
        }
    }

    /// Register a single GADT equality for one `when` constraint entry.
    /// Looks up the parameter index by name in `binding.params`, gets the
    /// corresponding argument from `args`, resolves the concrete type,
    /// and calls `register_gadt_eq` to record the mapping.
    ///
    /// If `exist_skolems` is non-empty (the variant has `exists` variables),
    /// any reference to an existentially quantified type parameter in the
    /// concrete type is replaced with the corresponding skolem TypeId.
    /// Transactionally verify that a pending NESTED GADT `when` equality
    /// is satisfiable under the inner scrutinee's type arguments.  Returns
    /// `false` for an impossible nested constructor (its equality must not
    /// be registered into the arm).  Equalities whose RHS contains an
    /// existential skolem are always accepted (opacity semantics — the
    /// witness equation is inert regardless of satisfiability).
    fn nested_eq_satisfiable(
        &mut self,
        binding: &TypeBinding,
        args: &[TypeId],
        param_name: &Symbol,
        concrete_ty: &crate::ast::Type,
        exist_params: &[Symbol],
        skolems: &[TypeId],
        span: crate::ast::Span,
    ) -> bool {
        // Resolve the `when` RHS (skolem-aware, same as
        // `register_single_gadt_eq`).
        let d = match self.resolve_type_with_skolems(concrete_ty, exist_params, skolems) {
            Some(d) => d,
            None => match self.resolve_type(concrete_ty) {
                Ok(d) => d,
                Err(_) => return false,
            },
        };
        // RHS contains an existential skolem → inert equation; accept.
        if self.ctx.contains_gadt_skolem(d) {
            return true;
        }
        // Extract the inner scrutinee's actual argument for this param.
        let Some((pi, _)) = binding
            .params
            .iter()
            .enumerate()
            .find(|(_, p)| p.name == *param_name)
        else {
            return true;
        };
        let Some(&a) = args.get(pi) else {
            return true;
        };
        // Canonicalize both sides through `resolve_binding` so the probe
        // sees OUTER GADT refinements (the fact stack IS consulted by
        // resolve_binding).  E.g. if an outer arm refined `T → Int<32>`,
        // an inner `when T == Bool` equality resolves `a` to `Int<32>`
        // and the probe correctly rejects it.
        let d = self.ctx.resolve_binding(d);
        let a = self.ctx.resolve_binding(a);
        // Transactional satisfiability probe (same discipline as
        // `is_gadt_variant_reachable`).
        let saved_seen = self.ctx.save_unify_seen();
        self.ctx.begin_transaction();
        let ok = self
            .ctx
            .try_unify(d, a, Some(&self.infer.region_tree))
            .is_ok();
        self.ctx.rollback_transaction();
        self.ctx.restore_unify_seen(saved_seen);
        ok
    }

    /// or-pattern GADT refinement intersection (rules 1-6): a type
    /// parameter is refined only when ALL alternatives produce the SAME
    /// equality; conflicting equalities raise E066; an unconstrained
    /// alternative leaves the parameter abstract.  `alt_eqs[i]` holds the
    /// `when` equalities collected (in isolation) by the i-th alternative
    /// (see `Pattern::Or` handling).  GHC's OrPat does NOT propagate
    /// or-pattern refinements to the body: each alternative's pattern is
    /// checked separately, but the body (`thing_inside`) is checked once
    /// and the per-alternative GADT givens are hopped/discarded
    /// (`Note [Hopping the LIE]`; GHC's testsuite `Or4.hs` is rejected
    /// because the body cannot use the refinement).  Expressing
    /// refinement as an equality intersection fits Posita's registry
    /// architecture more directly, and propagating the intersection
    /// (all alternatives agree) is sound — it is the disjunction's
    /// common facts.
    pub(crate) fn apply_or_alt_intersection(
        &mut self,
        alt_eqs: &[Vec<PendingInnerGadtEq>],
        alt_reachable: &[bool],
        span: Span,
    ) {
        if alt_eqs.is_empty() {
            return;
        }
        // Issue 2 (order-independent): anchor the intersection on the
        // first REACHABLE alternative.  Anchoring on `alt_eqs[0]`
        // unconditionally made the result order-dependent — an
        // unreachable first alternative (e.g. `Eq(_) | Lit(_)` on a
        // concrete scrutinee where `Eq` is dead) still served as the
        // base and falsely raised E066 against reachable alternatives,
        // while `Lit(_) | Eq(_)` compiled cleanly.  If NO alternative is
        // reachable, nothing can be intersected or propagated.
        let Some(base_idx) = alt_reachable.iter().position(|&r| r) else {
            return;
        };
        let base = &alt_eqs[base_idx];
        for eq in base {
            let mut all_same = true;
            let mut conflict = false;
            for (i, other) in alt_eqs.iter().enumerate() {
                // Per-alternative reachability (Issue 2): an unreachable
                // alternative's equalities can neither conflict with nor
                // contribute to the reachable ones — ignore them.  The
                // base itself is reachable by construction (base_idx).
                if i == base_idx || !alt_reachable.get(i).copied().unwrap_or(true) {
                    continue;
                }
                // Key: (binding.def_id, param_name) — compare bindings by
                // def_id (TypeBinding has no PartialEq).  The RHS types are
                // compared structurally after resolution (ignoring AST
                // span — the same type at different source positions has
                // different spans, and direct AST equality would falsely
                // report a conflict).
                let same = other.iter().any(|o| {
                    o.param_name == eq.param_name
                        && o.binding.def_id == eq.binding.def_id
                        && self.or_eq_concrete_equal(eq, o)
                });
                let different = other.iter().any(|o| {
                    o.param_name == eq.param_name
                        && o.binding.def_id == eq.binding.def_id
                        && !self.or_eq_concrete_equal(eq, o)
                });
                if different {
                    conflict = true;
                    break;
                }
                if !same {
                    all_same = false;
                    // Do NOT break here — a later alternative may conflict
                    // with the base on this param, and we must detect that
                    // conflict (E066) rather than silently skipping it.
                }
            }
            if conflict {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "conflicting GADT refinements in or-pattern: `{}` is refined to different types by the alternatives",
                        eq.param_name,
                    ))
                    .with_code_str("E066")
                    .with_span(span),
                );
            } else if all_same {
                // All alternatives agree → propagate to the branch body
                // (into pending_eqs; apply_gadt_refinement registers it).
                self.ctx.gadt.pending_eqs.push(eq.clone());
            }
            // Some alternative unconstrained (!all_same && !conflict) →
            // do not propagate (the parameter stays abstract).
        }
    }

    /// Compare two or-pattern alternatives' `when` RHS types for semantic
    /// equality: resolve each to a TypeId (skolem-aware) and compare the
    /// TypeData STRUCTURE (ignoring AST span — the same type at different
    /// source positions has different spans and would falsely compare
    /// unequal).  Falls back to AST equality if resolution fails.
    fn or_eq_concrete_equal(&mut self, a: &PendingInnerGadtEq, b: &PendingInnerGadtEq) -> bool {
        let resolve = |ctx: &mut Self, eq: &PendingInnerGadtEq| {
            ctx.resolve_type_with_skolems(&eq.concrete_ty, &eq.exist_params, &eq.skolems)
                .or_else(|| ctx.resolve_type(&eq.concrete_ty).ok())
        };
        match (resolve(self, a), resolve(self, b)) {
            // Alpha-equivalence: GADT existential skolems compare equal BY
            // KIND (any existential witness is alpha-equivalent to any
            // other), not by identity or binder position — two alternatives
            // with the same existential SHAPE (`T == [X]` vs `T == [Y]`,
            // independent witnesses) are NOT conflicting.
            (Some(ta), Some(tb)) => {
                let da = self.ctx.get(ta).clone();
                let db = self.ctx.get(tb).clone();
                self.type_data_alpha_eq(&da, &db)
            }
            _ => {
                // Symmetric opacity: a path is an opaque `exists` witness if
                // EITHER side's existential scope binds it — a one-sided
                // list would make the equality result depend on argument
                // order (asymmetric / non-deterministic).
                let both_exists: Vec<Symbol> = a
                    .exist_params
                    .iter()
                    .chain(b.exist_params.iter())
                    .copied()
                    .collect();
                crate::hir::type_eq::type_eq_ignoring_spans(
                    &a.concrete_ty,
                    &b.concrete_ty,
                    &both_exists,
                    &self.symbols,
                )
            }
        }
    }

    /// Structural equality for `TypeData` with existential alpha-equivalence:
    /// GADT skolems compare equal by KIND (any existential witness is
    /// alpha-equivalent to any other), while all non-skolem structure
    /// compares exactly.
    fn type_data_alpha_eq(&self, a: &TypeData, b: &TypeData) -> bool {
        match (a, b) {
            (TypeData::Slice { elem: ea }, TypeData::Slice { elem: eb }) => {
                let da = self.ctx.get(*ea).clone();
                let db = self.ctx.get(*eb).clone();
                self.type_data_alpha_eq(&da, &db)
            }
            (TypeData::Ref { ty: ea, .. }, TypeData::Ref { ty: eb, .. }) => {
                let da = self.ctx.get(*ea).clone();
                let db = self.ctx.get(*eb).clone();
                self.type_data_alpha_eq(&da, &db)
            }
            (TypeData::Tuple { elems: ea }, TypeData::Tuple { elems: eb }) => {
                ea.len() == eb.len()
                    && ea.iter().zip(eb.iter()).all(|(&x, &y)| {
                        let da = self.ctx.get(x).clone();
                        let db = self.ctx.get(y).clone();
                        self.type_data_alpha_eq(&da, &db)
                    })
            }
            (
                TypeData::Adt {
                    kind: ka,
                    def_id: da,
                    args: aa,
                },
                TypeData::Adt {
                    kind: kb,
                    def_id: db,
                    args: ab,
                },
            ) => {
                ka == kb
                    && da == db
                    && aa.len() == ab.len()
                    && aa.iter().zip(ab.iter()).all(|(&x, &y)| {
                        let da = self.ctx.get(x).clone();
                        let db = self.ctx.get(y).clone();
                        self.type_data_alpha_eq(&da, &db)
                    })
            }
            (TypeData::Array { elem: ea, .. }, TypeData::Array { elem: eb, .. }) => {
                let da = self.ctx.get(*ea).clone();
                let db = self.ctx.get(*eb).clone();
                self.type_data_alpha_eq(&da, &db)
            }
            (TypeData::Pointer { ty: ea }, TypeData::Pointer { ty: eb }) => {
                let da = self.ctx.get(*ea).clone();
                let db = self.ctx.get(*eb).clone();
                self.type_data_alpha_eq(&da, &db)
            }
            (
                TypeData::Fn {
                    params: pa,
                    ret: ra,
                    ..
                },
                TypeData::Fn {
                    params: pb,
                    ret: rb,
                    ..
                },
            ) => {
                pa.len() == pb.len()
                    && pa.iter().zip(pb.iter()).all(|(&x, &y)| {
                        let da = self.ctx.get(x).clone();
                        let db = self.ctx.get(y).clone();
                        self.type_data_alpha_eq(&da, &db)
                    })
                    && {
                        let da = self.ctx.get(*ra).clone();
                        let db = self.ctx.get(*rb).clone();
                        self.type_data_alpha_eq(&da, &db)
                    }
            }
            // GADT existential skolems are alpha-equivalent (positional),
            // not identity-equal — this is the fix for or-pattern false
            // conflicts on existential variants.
            (
                TypeData::SkolemVar {
                    universe_num: ua, ..
                },
                TypeData::SkolemVar {
                    universe_num: ub, ..
                },
            ) if *ua == TypeContext::GADT_SKOLEM_UNIVERSE
                && *ub == TypeContext::GADT_SKOLEM_UNIVERSE =>
            {
                true
            }
            _ => a == b,
        }
    }

    fn register_single_gadt_eq(
        &mut self,
        param_name: &Symbol,
        concrete_ty: &Type,
        binding: &TypeBinding,
        args: &[TypeId],
        exist_params: &[Symbol],
        skolems: &[TypeId],
    ) {
        // Resolve the right-hand side of the constraint (the concrete type).
        // When existential skolems are in scope, resolve WITH skolems FIRST:
        // `exists X` must shadow any outer type name (builtin alias, type
        // parameter, etc.), so `when T == [X]` refers to the existential
        // binder, not to an outer `type X = ...`.  Fall back to normal
        // resolution only when the skolem-aware path cannot resolve it.
        let d = if !skolems.is_empty() {
            match self.resolve_type_with_skolems(concrete_ty, exist_params, skolems) {
                Some(d) => d,
                None => match self.resolve_type(concrete_ty) {
                    Ok(d) => d,
                    Err(_) => return,
                },
            }
        } else {
            match self.resolve_type(concrete_ty) {
                Ok(d) => d,
                Err(_) => return,
            }
        };
        // Check if the constraint targets an enum type parameter or an
        // existential variable.
        if let Some((pi, _)) = binding
            .params
            .iter()
            .enumerate()
            .find(|(_, p)| p.name == *param_name)
        {
            let Some(&a) = args.get(pi) else { return };
            // Conflict checking happens INSIDE `register_gadt_eq_directional`
            // (after canonicalization) so the check key always matches the
            // registration key.
            self.register_gadt_eq_directional(a, d, concrete_ty.span());
        } else if let Some(&skolem) = exist_params
            .iter()
            .position(|p| p == param_name)
            .and_then(|i| skolems.get(i))
        {
            // Constraint targets an existential variable (e.g.,
            // `when X == Int<32>` where X is an `exists` param).
            //
            // Witness solving: when the RHS is CLOSED (no unresolved vars),
            // the language-designer ruling (2026-08-04) permits refinement
            // via `register_gadt_eq_directional` (which handles the skolem
            // branch with `ParamRefinement` + warn).  An open RHS stays
            // inert as an `ExistentialEquation` — the witness remains
            // opaque (opacity is the DEFAULT, solving is opt-in per
            // consumer — GHC-style explicit coercion / OCaml's GADT
            // constraint mode).
            if self.type_has_unresolved_vars(d) {
                self.ctx.register_existential_equation(skolem, d);
            } else {
                self.register_gadt_eq_directional(skolem, d, concrete_ty.span());
            }
        }
    }

    /// Nested GADT refinement conflict detection (SYNTAX.md §"Nested GADT
    /// Refinement"): if an outer constructor already refined the same type
    /// parameter to a DIFFERENT concrete type, the inner `when` equality
    /// conflicts — report a compile error instead of silently
    /// overwriting the refinement.
    fn check_gadt_refinement_conflict(&mut self, from: TypeId, to: TypeId, span: Span) {
        let conflict = self.ctx.gadt.facts.borrow().iter().rev().any(|arm| {
            arm.iter().any(|f| match f {
                crate::hir::types::GadtFact::ParamRefinement {
                    from: f_from,
                    to: f_to,
                } => *f_from == from && *f_to != to,
                // An existential equation constrains the same `from` with a
                // type whose constructor differs from the new `to` — e.g.,
                // `ExistentialEquation(T, [S])` (T is a slice) vs
                // `ParamRefinement(T, Int<32>)` (T is an integer).  Same
                // constructor is allowed (the equation is inert; the
                // refinement is the active constraint).
                crate::hir::types::GadtFact::ExistentialEquation { lhs, rhs } => {
                    *lhs == from && rhs.tag() != to.tag()
                }
                _ => false,
            })
        });
        if conflict {
            self.diagnostics.push(
                Diagnostic::error(
                    "conflicting GADT refinements: the same type parameter is refined to two different types by nested `when` constraints",
                )
                .with_code_str("E061")
                .with_span(span),
            );
        }
    }

    /// Whether a type contains any `GenericParam`, `InferVar`, or GADT
    /// skolem anywhere in its structure.  Used to gate existential witness
    /// solving: `S → d` may only be a `ParamRefinement` when `d` is closed.
    pub(crate) fn type_has_unresolved_vars(&self, ty: TypeId) -> bool {
        // Follow bindings first: a bound InferVar is no longer "unresolved"
        // even if the arena still shows `TypeData::InferVar`.
        let resolved = self.ctx.resolve_binding(ty);
        match self.ctx.get(resolved) {
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => true,
            TypeData::SkolemVar { universe_num, .. }
                if *universe_num == TypeContext::GADT_SKOLEM_UNIVERSE =>
            {
                true
            }
            TypeData::Slice { elem } => self.type_has_unresolved_vars(*elem),
            TypeData::Ref { ty: inner, .. } => self.type_has_unresolved_vars(*inner),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.type_has_unresolved_vars(e)),
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.type_has_unresolved_vars(a)),
            TypeData::Array { elem, .. } => self.type_has_unresolved_vars(*elem),
            TypeData::Pointer { ty: inner } => self.type_has_unresolved_vars(*inner),
            TypeData::Ptr { size, pointee } => {
                self.type_has_unresolved_vars(*size) || self.type_has_unresolved_vars(*pointee)
            }
            TypeData::Fn { params, ret, .. } => {
                params.iter().any(|&p| self.type_has_unresolved_vars(p))
                    || self.type_has_unresolved_vars(*ret)
            }
            TypeData::Exists { base, .. } => self.type_has_unresolved_vars(*base),
            TypeData::Forall { body, .. } => self.type_has_unresolved_vars(*body),
            TypeData::AssociatedType { self_ty, .. } => self.type_has_unresolved_vars(*self_ty),
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.type_has_unresolved_vars(a)),
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                self.type_has_unresolved_vars(*body)
            }
            TypeData::Poly { body, .. } => self.type_has_unresolved_vars(*body),
            _ => false,
        }
    }

    /// Register a GADT equality in the correct direction:
    /// - If `a` is a refinable variable (GenericParam, InferVar), register `a → d`.
    /// - If `a` is concrete, register `d → a` and decompose structurally.
    ///
    /// The conflict check lives HERE (after canonicalization) rather than at
    /// the call sites, so the check key is ALWAYS the same as the
    /// registration key — a call site passing a non-root TypeId cannot
    /// silently miss an existing refinement on the chain root.
    fn register_gadt_eq_directional(&mut self, a: TypeId, d: TypeId, span: Span) {
        // Canonicalize `a` to its binding-chain ROOT first (following ONLY
        // bindings, never the GADT registry).  If `a` is a non-root node of
        // a path-compressed chain, `resolve_binding` would skip it and the
        // refinement would be invisible — keying to the root keeps the
        // refinement reachable from any chain head.
        let a = self.ctx.resolve_binding_no_gadt(a);
        // Conflict check uses the canonicalized key (same as registration),
        // scanning ALL active arms (outer refinements are still relevant
        // when checking inner ones).
        self.check_gadt_refinement_conflict(a, d, span);
        match self.ctx.get(a) {
            crate::hir::types::TypeData::GenericParam { .. }
            | crate::hir::types::TypeData::InferVar { .. } => {
                // Refinable variable → param refinement (visible to
                // resolve_binding) ONLY when the target is closed.  If `d`
                // contains ANY unresolved variable (a GADT skolem like
                // `when T == [X]`, another GenericParam, or an InferVar),
                // the equality is existential — register it as an INERT
                // equation so the witness never becomes visible through
                // `resolve_binding(a)`.  (A GenericParam/InferVar target
                // would otherwise create a spurious cross-parameter chain.)
                if self.type_has_unresolved_vars(d) {
                    self.ctx.register_existential_equation(a, d);
                } else {
                    self.ctx.register_param_refinement(a, d);
                }
            }
            // GADT existential skolem on the SCRUTINEE side: the equality
            // involves an opaque witness.  When the RHS is CLOSED (no
            // unresolved vars), the language-designer ruling (2026-08-04)
            // permits refinement: register a `ParamRefinement` + emit a
            // warning (the syntax is unusual).  An open RHS (containing
            // another skolem, GenericParam, or InferVar) stays inert as
            // an `ExistentialEquation` — the witness remains opaque.
            crate::hir::types::TypeData::SkolemVar { universe_num, .. }
                if *universe_num == TypeContext::GADT_SKOLEM_UNIVERSE =>
            {
                if self.type_has_unresolved_vars(d) {
                    // RHS has unresolved vars → inert equation
                    self.ctx.register_existential_equation(a, d);
                } else {
                    // RHS closed → refine the existential (unusual syntax)
                    self.ctx.register_param_refinement(a, d);
                    self.diagnostics.push(
                        Diagnostic::warning(
                            "existential variable refined by `when` constraint; \
                             this syntax is unusual",
                        )
                        .with_span(span),
                    );
                }
            }
            _ => {
                self.register_structural_gadt_eq(d, a, span);
            }
        }
    }

    /// Walk a pair of types structurally and register GADT equalities for
    /// any skolem-to-concrete mappings discovered during the walk.
    /// This solves existential witnesses inside compound types: from
    /// `[S] == [Int<32>]` it registers `S → Int<32>`, not merely
    /// `[S] → [Int<32>]`.
    ///
    /// CRITICAL: For existential GADTs the whole equality is registered as
    /// an INERT `ExistentialEquation` (never consulted by `resolve_binding`),
    /// so the inner skolem-to-concrete equality must NOT be exposed — that
    /// would violate the opacity guarantee (SYNTAX.md §"Existential
    /// Quantification": "they remain opaque except as dictated by the
    /// `when` constraint").
    fn register_structural_gadt_eq(&mut self, declared: TypeId, actual: TypeId, span: Span) {
        // If the declared (when-clause) type OR the actual scrutinee-side
        // type CONTAINS a GADT skolem, the whole equality is existential:
        // register it as an INERT equation (never consulted by
        // resolve_binding), so the witness stays opaque (SYNTAX.md
        // §"Existential Quantification").  Do NOT register a top-level
        // rewrite and do NOT decompose — the inner `skolem → concrete`
        // equality would leak through resolve_binding.  (Checking `actual`
        // as well is defense-in-depth: `is_gadt_variant_reachable` gates
        // this path today, but the invariant should be local, not
        // dependent on call-site ordering.)
        if self.ctx.contains_gadt_skolem(declared) || self.ctx.contains_gadt_skolem(actual) {
            self.ctx.register_existential_equation(declared, actual);
            return;
        }
        // No skolem: register the top-level param refinement ONLY if
        // `declared` is a refinable variable (GenericParam / InferVar).
        // A concrete→concrete equality is NOT a refinement — `resolve_binding`
        // must never rewrite a closed type.  (Equal concrete pairs are
        // no-ops; unequal pairs are contradictions that reachability
        // should have filtered.)
        match self.ctx.get(declared) {
            crate::hir::types::TypeData::GenericParam { .. }
            | crate::hir::types::TypeData::InferVar { .. } => {
                // Every ParamRefinement insertion — including the
                // sub-constraints discovered by structural decomposition —
                // must be conflict-checked.  The top-level check in
                // register_gadt_eq_directional covers only the outer
                // equality; without a check here, two nested `when`
                // constraints could refine the same variable to different
                // types and silently overwrite (determinism loss).
                self.check_gadt_refinement_conflict(declared, actual, span);
                self.ctx.register_param_refinement(declared, actual);
            }
            _ => {}
        }
        let d_data = self.ctx.get(declared).clone();
        let a_data = self.ctx.get(actual).clone();
        match (&d_data, &a_data) {
            (TypeData::Slice { elem: d_elem }, TypeData::Slice { elem: a_elem }) => {
                self.register_structural_gadt_eq(*d_elem, *a_elem, span);
            }
            (TypeData::Ref { ty: d_ty, .. }, TypeData::Ref { ty: a_ty, .. }) => {
                self.register_structural_gadt_eq(*d_ty, *a_ty, span);
            }
            (TypeData::Tuple { elems: d_elems }, TypeData::Tuple { elems: a_elems }) => {
                for (d, a) in d_elems.iter().zip(a_elems.iter()) {
                    self.register_structural_gadt_eq(*d, *a, span);
                }
            }
            (TypeData::Adt { args: d_args, .. }, TypeData::Adt { args: a_args, .. }) => {
                for (d, a) in d_args.iter().zip(a_args.iter()) {
                    self.register_structural_gadt_eq(*d, *a, span);
                }
            }
            (TypeData::Array { elem: d_elem, .. }, TypeData::Array { elem: a_elem, .. }) => {
                self.register_structural_gadt_eq(*d_elem, *a_elem, span);
            }
            (TypeData::Pointer { ty: d_ty }, TypeData::Pointer { ty: a_ty }) => {
                self.register_structural_gadt_eq(*d_ty, *a_ty, span);
            }
            (
                TypeData::Fn {
                    params: d_p,
                    ret: d_r,
                    ..
                },
                TypeData::Fn {
                    params: a_p,
                    ret: a_r,
                    ..
                },
            ) => {
                for (d, a) in d_p.iter().zip(a_p.iter()) {
                    self.register_structural_gadt_eq(*d, *a, span);
                }
                self.register_structural_gadt_eq(*d_r, *a_r, span);
            }
            _ => {} // Mismatched or non-decomposable — inner equalities already covered
        }
    }

    /// Trace-instrumented wrapper around `resolve_type_with_skolems_impl`.
    /// When `PONENT_TRACE` is set, each recursive resolution step is
    /// printed with indentation, showing the type, binder names, skolem
    /// list, and the outcome — invaluable for debugging payload type
    /// resolution (GADT/existential/generic) failures.
    fn resolve_type_with_skolems(
        &mut self,
        ty: &Type,
        exist_params: &[Symbol],
        skolems: &[TypeId],
    ) -> Option<TypeId> {
        #[cfg(debug_assertions)]
        {
            let _guard = crate::hir::anya::TraceGuard::enter();
            let result = self.resolve_type_with_skolems_impl(ty, exist_params, skolems);
            crate::hir::anya::trace_resolve(ty, exist_params, skolems, &result);
            result
        }
        #[cfg(not(debug_assertions))]
        {
            self.resolve_type_with_skolems_impl(ty, exist_params, skolems)
        }
    }

    /// Walk an AST type, replacing references to existentially quantified
    /// type variables with their skolem TypeIds.  `exist_params` is the
    /// variant's fixed name→index table; skolem IDENTITY is the index
    /// (GHC `realUnique` / OCaml `id: int`), so same-named binders in
    /// different variants resolve to their own scope's skolem.
    /// Returns `None` if the type cannot be resolved.
    fn resolve_type_with_skolems_impl(
        &mut self,
        ty: &Type,
        exist_params: &[Symbol],
        skolems: &[TypeId],
    ) -> Option<TypeId> {
        match ty {
            Type::Path(path, _) if path.len() == 1 => {
                // Single-name type: resolve by binder INDEX, not by name.
                let idx = exist_params.iter().position(|p| p == &path[0]);
                match idx.and_then(|i| skolems.get(i)) {
                    Some(&skolem) => Some(skolem),
                    None => {
                        // Try normal resolution
                        self.resolve_type(ty).ok()
                    }
                }
            }
            Type::Slice(elem, _) => {
                let elem_ty = self.resolve_type_with_skolems(elem, exist_params, skolems)?;
                Some(self.ctx.alloc(TypeData::Slice { elem: elem_ty }))
            }
            Type::Generic(base, args, _) => {
                // Generic type: resolve the base and build with skolem-substituted args.
                let base_id = match &**base {
                    Type::Path(path, _) => {
                        match self.resolve_type(&Type::Path(path.clone(), ty.span())) {
                            Ok(t) => t,
                            // Base may be a built-in that only resolves as
                            // part of the full generic type (e.g. `Int` in
                            // `Int<32>`); fall back to normal resolution.
                            Err(_) => return self.resolve_type(ty).ok(),
                        }
                    }
                    _ => match self.resolve_type(base) {
                        Ok(t) => t,
                        Err(_) => return self.resolve_type(ty).ok(),
                    },
                };
                let resolved_args: Vec<TypeId> = args
                    .iter()
                    .map(|arg| match arg {
                        crate::ast::GenericArg::Positional(t)
                        | crate::ast::GenericArg::Named(_, t) => {
                            self.resolve_type_with_skolems(t, exist_params, skolems)
                        }
                        crate::ast::GenericArg::Const(ac) => self
                            .resolve_type(&Type::Expr(ac.value.clone(), ac.span))
                            .ok(),
                    })
                    .collect::<Option<Vec<_>>>()?;
                if let TypeData::Adt { kind, def_id, .. } = self.ctx.get(base_id).clone() {
                    Some(self.ctx.alloc(TypeData::Adt {
                        kind,
                        def_id,
                        args: resolved_args,
                    }))
                } else {
                    // The base is a built-in generic type (e.g. `Int<32>`
                    // where `Int` is `TypeData::Int`, or `Float<N>`), NOT an
                    // ADT.  Fall back to normal resolution so these resolve
                    // correctly instead of returning None.
                    self.resolve_type(ty).ok()
                }
            }
            Type::Reference { inner, mutable, .. } => {
                let inner_ty = self.resolve_type_with_skolems(inner, exist_params, skolems)?;
                Some(self.ctx.alloc(TypeData::Ref {
                    ty: inner_ty,
                    mutable: *mutable,
                }))
            }
            Type::Pointer(inner, _) => {
                let inner_ty = self.resolve_type_with_skolems(inner, exist_params, skolems)?;
                Some(self.ctx.alloc(TypeData::Pointer { ty: inner_ty }))
            }
            Type::Tuple(elems, _) => {
                let resolved: Vec<TypeId> = elems
                    .iter()
                    .map(|e| self.resolve_type_with_skolems(e, exist_params, skolems))
                    .collect::<Option<Vec<_>>>()?;
                Some(self.ctx.tuple(resolved))
            }
            Type::Array(elem, size, _) => {
                let elem_ty = self.resolve_type_with_skolems(elem, exist_params, skolems)?;
                // Evaluate the array size expression.  Literal integers and
                // comptime-evaluable expressions are supported; for anything
                // else we return None so the caller falls back to normal
                // resolve_type.
                let size_val = match size.as_ref() {
                    crate::ast::Expr::Literal(crate::ast::Literal::Int(n), _) => {
                        if *n >= 0 {
                            *n as u64
                        } else {
                            return None;
                        }
                    }
                    // For non-literal size expressions, attempt comptime eval
                    // via the HirExpr if available, otherwise skip.
                    _ => {
                        // Check if we can evaluate via the already-resolved
                        // type.  If not, fall back to normal resolution.
                        return None;
                    }
                };
                Some(self.ctx.alloc(TypeData::Array {
                    elem: elem_ty,
                    size: size_val,
                }))
            }
            Type::Function { params, ret, .. } => {
                let resolved_params: Vec<TypeId> = params
                    .iter()
                    .map(|p| self.resolve_type_with_skolems(p, exist_params, skolems))
                    .collect::<Option<Vec<_>>>()?;
                let resolved_ret = self.resolve_type_with_skolems(ret, exist_params, skolems)?;
                Some(self.ctx.function(resolved_params, resolved_ret))
            }
            // For Path with multiple segments or other forms, try normal resolution
            _ => self.resolve_type(ty).ok(),
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
                        crate::ast::GenericArg::Const(ac) => {
                            format!(
                                "const {}",
                                Self::type_to_string(&crate::ast::Type::Expr(
                                    ac.value.clone(),
                                    ac.span
                                ))
                            )
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

    /// Check if we are currently inside a `comptime { ... }` block.
    ///
    /// Used to enforce comptime sandbox restrictions: calling `@trusted` or
    /// `@io` functions from comptime context is a compile-time error.
    pub(crate) fn is_in_comptime(&self) -> bool {
        self.region_tree
            .iter_frames_rev()
            .any(|f| matches!(f.kind, CtxKind::Comptime))
    }

    /// Check if we are currently inside an `isolate { ... }` block.
    pub(crate) fn is_in_isolate(&self) -> bool {
        self.region_tree
            .iter_frames_rev()
            .any(|f| matches!(f.kind, CtxKind::Isolate))
    }

    // ── Literal value scope helpers ─────────────────────────────
    //
    // literal_values tracks comptime-known constant values as a
    // scope-aware value stack.  These helpers manage the stack:

    /// Push a new scope frame for literal value tracking.
    /// Must be called alongside `enter_var_scope()` (a new frame is
    /// pushed before checking a block's statements).
    pub(crate) fn push_literal_scope(&mut self) {
        self.scope_var_stack.push(Vec::new());
    }

    /// Pop the innermost literal scope frame and remove all values
    /// that were defined in that scope, restoring outer-scope values.
    ///
    /// Must be called when the corresponding `VarScopeGuard` drops
    /// (i.e. on scope exit), paired with the matching `push_literal_scope`.
    ///
    /// # Panics
    /// Panics if `scope_var_stack` is empty (scope imbalance bug).
    pub(crate) fn pop_literal_scope(&mut self) {
        let frame = self
            .scope_var_stack
            .pop()
            .expect("pop_literal_scope without matching push_literal_scope");
        for name in &frame {
            if let Some(stack) = self.literal_values.get_mut(name) {
                stack.pop();
                if stack.is_empty() {
                    self.literal_values.remove(name);
                }
            }
        }
    }

    /// Record a comptime-known value for `name` in the current scope.
    /// Pushes onto the per-variable value stack and records `name` in
    /// the current scope frame so `pop_literal_scope` can undo it.
    ///
    /// # Panics
    /// Panics if no literal scope frame has been pushed (must be
    /// called inside a block after `push_literal_scope`).
    pub(crate) fn insert_literal_value(&mut self, name: Symbol, value: ComptimeValue) {
        self.literal_values.entry(name).or_default().push(value);
        self.scope_var_stack
            .last_mut()
            .expect("insert_literal_value outside any literal scope")
            .push(name);
    }

    /// Get the current (innermost) comptime-known value for `name`,
    /// or `None` if the variable has no comptime-known value in the
    /// current or any enclosing scope.
    pub(crate) fn get_literal_value(&self, name: &Symbol) -> Option<&ComptimeValue> {
        self.literal_values.get(name)?.last()
    }

    /// Check whether a statement should be skipped due to `@cfg` evaluation.
    /// `@auto_ro` (SYNTAX.md §Local Relaxation) is only meaningful on
    /// function definitions.  A placement on any other item is a surface
    /// contract violation and must be reported, not silently ignored.
    fn validate_auto_ro_placement(&mut self, stmt: &Stmt) {
        let attributes = match stmt {
            // Function definitions are the ONLY valid placement.
            Stmt::FunctionDef { attributes, .. }
            | Stmt::TypeDef { attributes, .. }
            | Stmt::TraitDef { attributes, .. }
            | Stmt::ImplBlock { attributes, .. }
            | Stmt::ExternFunction { attributes, .. }
            | Stmt::LayoutDef { attributes, .. }
            | Stmt::Generate { attributes, .. }
            | Stmt::VariableDef { attributes, .. }
            | Stmt::ComptimeBlock { attributes, .. }
            | Stmt::Isolate { attributes, .. } => attributes,
            _ => {
                // A future attribute-bearing `Stmt` variant must not
                // silently bypass the `@auto_ro`/`@auto_coerce` placement
                // validation.  Within this crate the match stays exhaustive
                // (new variants are compile errors); the `#[non_exhaustive]`
                // attribute covers downstream crates.
                return;
            }
        };
        for attr in attributes {
            if attr.name.eq_str("auto_ro") {
                if !matches!(stmt, Stmt::FunctionDef { .. }) {
                    self.diagnostics.push(
                        Diagnostic::error("`@auto_ro` is only valid on function definitions")
                            .with_span(attr.span)
                            .with_suggestion(
                                "move the `@auto_ro` attribute to the function that needs \
                                 the local relaxation",
                            ),
                    );
                }
            } else if attr.name.eq_str("auto_coerce") {
                if !matches!(stmt, Stmt::FunctionDef { .. }) {
                    self.diagnostics.push(
                        Diagnostic::error("`@auto_coerce` is only valid on function definitions")
                            .with_span(attr.span),
                    );
                }
            }
        }
    }

    ///
    /// Returns `true` if the item has a `@cfg(condition)` attribute whose
    /// condition is not met on the current target, meaning the item should
    /// be excluded from compilation.
    fn should_skip_due_to_cfg(&mut self, stmt: &Stmt) -> bool {
        let attributes = match stmt {
            Stmt::FunctionDef { attributes, .. }
            | Stmt::TypeDef { attributes, .. }
            | Stmt::TraitDef { attributes, .. }
            | Stmt::ImplBlock { attributes, .. }
            | Stmt::ExternFunction { attributes, .. }
            | Stmt::LayoutDef { attributes, .. }
            | Stmt::Generate { attributes, .. }
            | Stmt::VariableDef { attributes, .. }
            | Stmt::ComptimeBlock { attributes, .. }
            | Stmt::Isolate { attributes, .. } => attributes,
            _ => return false,
        };
        for attr in attributes {
            if !crate::hir::cfg::eval_cfg(
                attr,
                &self.ctx.target,
                &self.features,
                self.debug,
                &mut self.diagnostics,
            ) {
                return true;
            }
        }
        false
    }

    /// In strict mode, check that all `@cfg` conditions on this statement
    /// are provably reachable under some target configuration.
    /// If a condition is contradictory (e.g. `all(target_os = "linux", target_os = "windows")`),
    /// emit a diagnostic error.
    fn check_cfg_reachability(&mut self, stmt: &Stmt) {
        let (attributes, span) = match stmt {
            Stmt::FunctionDef {
                attributes, span, ..
            }
            | Stmt::TypeDef {
                attributes, span, ..
            }
            | Stmt::TraitDef {
                attributes, span, ..
            }
            | Stmt::ImplBlock {
                attributes, span, ..
            }
            | Stmt::ExternFunction {
                attributes, span, ..
            }
            | Stmt::LayoutDef {
                attributes, span, ..
            }
            | Stmt::Generate {
                attributes, span, ..
            }
            | Stmt::VariableDef {
                attributes, span, ..
            }
            | Stmt::ComptimeBlock {
                attributes, span, ..
            }
            | Stmt::Isolate {
                attributes, span, ..
            } => (attributes, *span),
            _ => return,
        };
        for attr in attributes {
            if !crate::hir::cfg::is_provably_reachable(
                attr,
                self.strict_mode,
                &mut self.diagnostics,
            ) {
                self.diagnostics.push(
                    Diagnostic::error(
                        "@cfg condition is unreachable: no target configuration can satisfy it"
                    )
                    .with_code_str("E092")
                    .with_span(span)
                    .with_help("the @cfg condition contradicts itself (e.g. target_os = \"linux\" and target_os = \"windows\" simultaneously)")
                );
            }
        }
    }

    /// Take the collected diagnostics out of the checker (including warnings).
    /// Returns the `DiagCtxt` with all accumulated diagnostics.
    pub fn take_diagnostics(&mut self) -> DiagCtxt {
        std::mem::replace(&mut self.diagnostics, DiagCtxt::new())
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

use crate::hir::hir::{HirExpr, HirStmt};

// Internal node type for the iterative Error-walker stack so we can
// push both HirStmt and HirExpr references onto one Vec.
enum Node<'a> {
    Stmt(&'a HirStmt),
    Expr(&'a HirExpr),
}

/// Iteratively check whether a tree of `HirStmt` / `HirExpr` contains any
/// `Error` node, using an explicit stack to avoid stack overflow on deeply
/// nested code (e.g. 1000+ nested `if` expressions).
///
/// This traversal is scoped to a single function body (passed as `stmts`),
/// not the entire program, so O(n) complexity is acceptable — the
/// subsequent comptime evaluation is also O(n) for the same body.
/// Returns `true` the moment an `Error` is found (short-circuit).
fn contains_error(stmts: &[HirStmt]) -> bool {
    let mut stack: Vec<Node<'_>> = stmts.iter().map(Node::Stmt).collect();

    while let Some(node) = stack.pop() {
        match node {
            Node::Stmt(HirStmt::Error) => return true,
            Node::Stmt(HirStmt::Expression(e)) => stack.push(Node::Expr(e)),
            Node::Stmt(
                HirStmt::If {
                    then_branch,
                    else_branch,
                    ..
                }
                | HirStmt::IfLet {
                    then_branch,
                    else_branch,
                    ..
                },
            ) => {
                stack.extend(then_branch.iter().map(Node::Stmt));
                if let Some(b) = else_branch {
                    stack.extend(b.iter().map(Node::Stmt));
                }
            }
            Node::Stmt(
                HirStmt::While { body, .. }
                | HirStmt::WhileLet { body, .. }
                | HirStmt::For { body, .. }
                | HirStmt::Loop { body, .. }
                | HirStmt::ComptimeBlock { body, .. }
                | HirStmt::ScopeCleanup { body, .. }
                | HirStmt::Unsafe { body, .. }
                | HirStmt::Isolate { body, .. }
                | HirStmt::Generate { body, .. },
            ) => {
                stack.extend(body.iter().map(Node::Stmt));
            }
            Node::Stmt(HirStmt::GhostVariableDef { inner, .. }) => stack.push(Node::Stmt(inner)),
            // VariableDef and Assign carry a value expression that may contain Error.
            Node::Stmt(HirStmt::VariableDef { value, .. }) => {
                if let Some(e) = value {
                    stack.push(Node::Expr(e));
                }
            }
            Node::Stmt(HirStmt::Assign { value, .. }) => stack.push(Node::Expr(value)),
            // All other HirStmt variants have no nested Error-carrying nodes.
            Node::Stmt(_) => {}

            // ── HirExpr ──
            Node::Expr(HirExpr::Error(_)) => return true,
            Node::Expr(e) => push_expr_children(&mut stack, e),
        }
    }
    false
}

/// Push the children of an expression onto the stack.
fn push_expr_children<'a>(stack: &mut Vec<Node<'a>>, expr: &'a HirExpr) {
    match expr {
        HirExpr::Error(_) => unreachable!("handled by caller"),
        // Leaf variants.
        HirExpr::Literal(..)
        | HirExpr::Ident(..)
        | HirExpr::TypeInfo(..)
        | HirExpr::LayoutOf(..)
        | HirExpr::CompileError(..) => {}
        // Single child.
        HirExpr::TypeAnnotated { expr: e, .. }
        | HirExpr::UnaryOp { expr: e, .. }
        | HirExpr::FieldAccess { base: e, .. }
        | HirExpr::AttrAccess { base: e, .. }
        | HirExpr::Cast { expr: e, .. }
        | HirExpr::Move(e, _, _)
        | HirExpr::Try { expr: e, .. }
        | HirExpr::LeaveWith { expr: e, .. }
        | HirExpr::Return { value: e, .. }
        | HirExpr::Await { expr: e, .. }
        | HirExpr::Old { expr: e, .. }
        | HirExpr::PolyBox { expr: e, .. }
        | HirExpr::PolyUnbox { expr: e, .. } => stack.push(Node::Expr(e)),
        // Two children.
        HirExpr::BinaryOp { left, right, .. }
        | HirExpr::Quantified {
            range: left,
            body: right,
            ..
        } => {
            stack.push(Node::Expr(right));
            stack.push(Node::Expr(left));
        }
        HirExpr::Index { base, index, .. } => {
            stack.push(Node::Expr(index));
            stack.push(Node::Expr(base));
        }
        HirExpr::Range { start, end, .. } => {
            if let Some(e) = end {
                stack.push(Node::Expr(e));
            }
            if let Some(e) = start {
                stack.push(Node::Expr(e));
            }
        }
        // Collections of children.
        HirExpr::Call { callee, args, .. } => {
            stack.push(Node::Expr(callee));
            stack.extend(args.iter().map(Node::Expr));
        }
        HirExpr::StructLit { fields, .. } => {
            stack.extend(fields.iter().map(|(_, e)| Node::Expr(e)));
        }
        HirExpr::EnumLit { payload, .. } => {
            if let Some(e) = payload {
                stack.push(Node::Expr(e));
            }
        }
        HirExpr::Tuple(items, _, _) | HirExpr::Array(items, _, _) => {
            stack.extend(items.iter().map(Node::Expr));
        }
        // Blocks / stmt-containers.
        HirExpr::Closure { body, .. }
        | HirExpr::UnsafeBlock { body, .. }
        | HirExpr::Block(body, _, _)
        | HirExpr::Task { block: body, .. } => {
            stack.extend(body.iter().map(Node::Stmt));
        }
        HirExpr::Catch {
            expr: e, branches, ..
        } => {
            stack.push(Node::Expr(e));
            for b in branches {
                stack.extend(b.body.iter().map(Node::Stmt));
            }
        }
        HirExpr::If {
            then_branch,
            else_branch,
            ..
        }
        | HirExpr::IfLet {
            then_branch,
            else_branch,
            ..
        } => {
            stack.extend(then_branch.iter().map(Node::Stmt));
            if let Some(b) = else_branch {
                stack.extend(b.iter().map(Node::Stmt));
            }
        }
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            stack.push(Node::Expr(scrutinee));
            for arm in arms {
                if let Some(g) = &arm.guard {
                    stack.push(Node::Expr(g));
                }
                stack.push(Node::Expr(&arm.body));
            }
        }
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
        Expr::Ident(name, _) if name.eq_str("codomain") || name.as_str().starts_with('@') => {
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

/// Collect a comptime traceback from a slice of comptime frames.
/// Returns structured (ComptimeReason, Span) pairs for flexible rendering.
fn format_comptime_traceback_inner(
    frames: &[&CtxFrame],
    _source: Option<&str>,
) -> Vec<(ComptimeReason, Span)> {
    frames
        .iter()
        .filter_map(|frame| frame.comptime_reason.map(|reason| (reason, frame.span)))
        .collect()
}

/// Check whether a HirExpr tree contains any runtime identifier references.
/// Used to enforce that `scope_cleanup when` conditions are compile-time
/// predicates.  `ghost_var_scopes` contains names of ghost variables that
/// are allowed in compile-time predicates (SYNTAX.md: "may reference only
/// ghost variables and other compile-time-constant expressions").
/// `runtime_var_scopes` contains names of RUNTIME variables; a runtime
/// binding in an INNER scope shadows an outer ghost of the same name and
/// must be treated as runtime (rejected).
/// Walk an access chain (`FieldAccess` / `Index` / `Deref`) to its root
/// `Ident` — used for `&ro` freeze registration and the frozen-mutation
/// check so `&ro x[i]`, `x.freeze!().f`, `*x.f`, etc. all track `x`.
fn expr_root_ident(e: &crate::ast::Expr) -> Option<Symbol> {
    match e {
        crate::ast::Expr::Ident(name, _) => Some(*name),
        crate::ast::Expr::FieldAccess { base, .. } => expr_root_ident(base),
        crate::ast::Expr::Index { base, .. } => expr_root_ident(base),
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Deref,
            expr,
            ..
        } => expr_root_ident(expr),
        _ => None,
    }
}

fn contains_runtime_ident(
    expr: &HirExpr,
    ghost_var_scopes: &[HashSet<Symbol>],
    runtime_var_scopes: &[HashSet<Symbol>],
    local_params: &HashMap<Symbol, TypeId>,
) -> bool {
    match expr {
        HirExpr::Ident(name, _, _) => {
            // Find the INNERMOST scope containing this name.  If that
            // binding is runtime, the ident is a runtime variable even if
            // an outer scope declares a ghost of the same name.
            let depth = ghost_var_scopes.len().max(runtime_var_scopes.len());
            for d in (0..depth).rev() {
                let is_runtime = runtime_var_scopes
                    .get(d)
                    .map_or(false, |s| s.contains(name));
                let is_ghost = ghost_var_scopes.get(d).map_or(false, |s| s.contains(name));
                if is_runtime || is_ghost {
                    return is_runtime;
                }
            }
            // Generic parameters (incl. `const` ones) are compile-time
            // entities — SYNTAX.md §"Structured Resource Cleanup" allows
            // const generic parameters in `when` conditions.  (The caller
            // still enforces the Boolean type of the whole condition.)
            if local_params.contains_key(name) {
                return false;
            }
            // Unbound name — conservatively treat as runtime.
            true
        }
        HirExpr::Literal(_, _, _) => false,
        // A comptime call (`DEBUG!()`) is a compile-time-constant
        // expression (SYNTAX.md §"Comptime"): it is evaluated at compile
        // time, so the whole call is a valid compile-time predicate.
        // Its ARGUMENTS are still checked recursively — a comptime call
        // with a runtime argument (`DEBUG!(flag)`) must be rejected,
        // since the predicate would depend on runtime state.  A runtime
        // call is conservatively rejected.
        HirExpr::Call {
            comptime: true,
            args,
            ..
        } => args
            .iter()
            .any(|a| contains_runtime_ident(a, ghost_var_scopes, runtime_var_scopes, local_params)),
        HirExpr::Call {
            comptime: false, ..
        } => true,
        HirExpr::Block(stmts, _, _) => stmts.iter().any(|s| {
            contains_stmt_runtime_ident(s, ghost_var_scopes, runtime_var_scopes, local_params)
        }),
        HirExpr::UnaryOp { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::BinaryOp { left, right, .. } => {
            contains_runtime_ident(left, ghost_var_scopes, runtime_var_scopes, local_params)
                || contains_runtime_ident(right, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::FieldAccess { base, .. } => {
            contains_runtime_ident(base, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Index { base, index, .. } => {
            contains_runtime_ident(base, ghost_var_scopes, runtime_var_scopes, local_params)
                || contains_runtime_ident(index, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Cast { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            contains_runtime_ident(cond, ghost_var_scopes, runtime_var_scopes, local_params)
                || then_branch.iter().any(|s| {
                    contains_stmt_runtime_ident(
                        s,
                        ghost_var_scopes,
                        runtime_var_scopes,
                        local_params,
                    )
                })
                || else_branch.as_ref().map_or(false, |b| {
                    b.iter().any(|s| {
                        contains_stmt_runtime_ident(
                            s,
                            ghost_var_scopes,
                            runtime_var_scopes,
                            local_params,
                        )
                    })
                })
        }
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            contains_runtime_ident(
                scrutinee,
                ghost_var_scopes,
                runtime_var_scopes,
                local_params,
            ) || arms.iter().any(|arm| {
                contains_runtime_ident(
                    &arm.body,
                    ghost_var_scopes,
                    runtime_var_scopes,
                    local_params,
                )
            })
        }
        HirExpr::Tuple(elems, _, _) => elems
            .iter()
            .any(|e| contains_runtime_ident(e, ghost_var_scopes, runtime_var_scopes, local_params)),
        HirExpr::Array(elems, _, _) => elems
            .iter()
            .any(|e| contains_runtime_ident(e, ghost_var_scopes, runtime_var_scopes, local_params)),
        HirExpr::PolyBox { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::UnsafeBlock { body, .. } => body.iter().any(|s| {
            contains_stmt_runtime_ident(s, ghost_var_scopes, runtime_var_scopes, local_params)
        }),
        HirExpr::Catch { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Try { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Await { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Old { expr: inner, .. } => {
            contains_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirExpr::Error(_) => false,
        HirExpr::Return { value, .. } => {
            contains_runtime_ident(value, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        // Default: conservatively treat unknown/edge expressions as runtime.
        // This is safe (rejects valid code) but not ideal — if a new HirExpr
        // variant is added without updating this match, it will be rejected.
        // (The `#[non_exhaustive]` attribute on `HirExpr` ensures downstream
        // crates add a wildcard arm; this wildcard is the safe fallback.)
        _ => true,
    }
}

/// Check whether a HirStmt tree contains any runtime identifier references.
/// Used by `contains_runtime_ident` for block traversal.
fn contains_stmt_runtime_ident(
    stmt: &HirStmt,
    ghost_var_scopes: &[HashSet<Symbol>],
    runtime_var_scopes: &[HashSet<Symbol>],
    local_params: &HashMap<Symbol, TypeId>,
) -> bool {
    match stmt {
        HirStmt::VariableDef { value, .. } => value.as_ref().map_or(false, |v| {
            contains_runtime_ident(v, ghost_var_scopes, runtime_var_scopes, local_params)
        }),
        HirStmt::Expression(expr) => {
            contains_runtime_ident(expr, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        HirStmt::GhostVariableDef { inner, .. } => {
            contains_stmt_runtime_ident(inner, ghost_var_scopes, runtime_var_scopes, local_params)
        }
        _ => true,
    }
}

#[cfg(test)]
pub mod tests;
