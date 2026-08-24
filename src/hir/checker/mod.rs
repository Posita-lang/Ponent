use crate::ast::{
    Attribute, BinOp, Contract, Expr, GenericArg, Literal, Pattern, Program, Span, Stmt, Type,
    UnaryOp, VariableKind,
};
use crate::diagnostics::{
    Applicability, ComptimeReason, DiagCtxt, Diagnostic, DiagnosticKind, Label, Suggestion,
    SuggestionStyle, TypeCtx,
};
use crate::hir::comptime::value::ComptimeValue;
use crate::hir::hir::{HirCatchBranch, HirMatchArm, HirParam, HirPattern, HirProgram};
use crate::hir::infer::{Constraint, InferenceContext, TypeVariableKind, VarOrigin};
use crate::hir::resolver::ResolutionMap;
use crate::hir::symbol::{SymbolTable, TypeBinding, TypeKind};
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::project::ProjectionCache;
use crate::hir::traits::solver::select::SelectionContext;
use crate::hir::traits::solver::{
    FulfillmentContext, Obligation, ObligationCause, ObligationCauseCode,
    Predicate as TraitPredicate,
};
use crate::hir::types::{
    Characteristic, CrateId, DefId, ExistScopeFrame, PendingInnerGadtEq, Subst, TypeContext,
    TypeData, TypeId,
};
use crate::symbol::Symbol;
use num_bigint::BigInt;
use num_traits::Zero;
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
pub mod probe;
pub mod region;
pub mod types;
use self::autoderef::{AutoderefIter, DEFAULT_MAX_DEREF_DEPTH};
use self::context::ScopeGuard;
use self::helpers::{
    did_you_mean_suggestion, find_similar_names, is_valid_lvalue, levenshtein_distance,
};
use self::types::{Expectation, TypingContext, typing_context_to_type_ctx};
pub use contract::{Guarantee, GuaranteeChain, Predicate};
pub use fn_ctxt::FnCtxt;
pub use region::{Region, RegionFrameIter, RegionId, RegionTree};

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
            .expect("insert called with empty scope frame stack (push_frame missing)")
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
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("extend called with empty scope frame stack (push_frame missing)")
            .extend(iter);
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

pub(crate) struct PendingGadtConstruct {
    pub enum_ty: crate::hir::types::TypeId,
    pub variant: Symbol,
    pub span: crate::ast::Span,
}

pub struct TypeChecker<'a, 'input> {
    ctx: &'a mut TypeContext<'input>,
    symbols: &'a SymbolTable<'input>,
    trait_env: &'a mut TraitEnv<'input>,
    diagnostics: DiagCtxt,
    /// Reused SMT solver for the loop-inference hint gate — spawns at most
    /// once per checker run instead of once per `while` loop.
    smt_solver: Option<crate::hir::smt::SmtSolver>,
    /// Deterministic global counter for generated invariant-binder names
    /// (`_inv_{id}`) — unlike the old span-based name, the id depends on
    /// compilation order and cannot be predicted by user code, so the
    /// capture-avoidance retry loop exhausts essentially never.
    next_inv_binder_id: usize,
    /// The most recently checked program, kept even when the borrow-check
    /// post-pass's diagnostics reject it — lets the Polonius equivalence
    /// tests inspect the HIR bodies of REJECTED programs (the error
    /// direction).
    pub(crate) last_checked_program: Option<HirProgram<'input>>,
    /// The per-function borrow signature data — (function name, the
    /// input-borrow parameter POSITIONS, the A(ρ) signature facts) — for
    /// the extractor's cross-function call-site mapping.
    pub(crate) signature_facts: Vec<(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>, // the receiver type (methods only — None for free functions)
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )>,
    /// Two-phase refactor: function bodies whose HIR is fully checked
    /// and whose signature facts are finalized — the flow-sensitive
    /// borrow-check + move post-passes run over ALL of them in Phase B,
    /// AFTER the signature registry is complete (order-independent).
    /// The pending runtime post-checks (Phase B): the function name → its
    /// HIR body (the borrow + move checks run over these AFTER the
    /// signature registry is complete).
    pub(crate) pending_runtime_checks: Vec<(Symbol, Vec<HirStmt<'input>>)>,
    /// The `finally` blocks of the pending functions (name → block
    /// statements): passed to `borrow_check_function` so the finally
    /// statements participate in the borrow check (SYNTAX.md §finally —
    /// the block runs on every function-exit edge).
    function_finally: HashMap<Symbol, Vec<HirStmt<'input>>>,
    /// GADT constructions deferred to the post-solve validation phase
    /// (the committee ruling — solve → default → validate).
    pub(crate) pending_gadt_constructs: Vec<PendingGadtConstruct>,
    /// Method-lookup cache (the rustc tcx-query pattern): (ty, name) →
    /// resolved (param_tys, ret_ty, method_def_id) or None.  Cleared
    /// whenever inherent methods / impls are registered.
    method_cache: HashMap<
        (crate::hir::types::TypeId, Symbol),
        Option<(
            Vec<crate::hir::types::TypeId>,
            crate::hir::types::TypeId,
            DefId,
        )>,
    >,
    /// Source text for converting byte offsets to line:column in tracebacks.
    source: Option<&'a str>,
    current_function: Option<DefId>,
    current_return_type: Option<TypeId>,
    /// The current function's `@hint(assertion)` list — verified against
    /// loop candidates during body checking (a user hint is an
    /// assertion, not decoration).
    pending_function_hints: Vec<crate::ast::Expr<'input>>,
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
    resolution_map: ResolutionMap<'input>,
    /// Local cache of generic type parameter types (e.g. `T` in `def foo<T>(x: T)`).
    /// Populated when processing function definitions with type_params.
    /// Also used by `set auto<T> = expr` to bind captured type names.
    ///
    /// # Scope leak note
    /// `auto<T>` inserts entries that are never removed when the block scope
    /// exits.  This is safe because the **resolver** uses lexical scoping
    /// (a `Scope<'input>` stack in `SymbolTable<'input>`), so `T` is unresolvable after the
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
    guarantee_chain: GuaranteeChain<'input>,
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
    /// Per-isolate-frame set of variables declared INSIDE the isolate
    /// block (a stack parallel to the CtxKind::Isolate frames).
    /// An assignment inside an isolate block to a variable NOT
    /// declared in the block (a captured outer mutable local or a
    /// function parameter) mutates external mutable state and must be
    /// rejected (SYNTAX.md §Task Isolation — "does not access any
    /// external mutable state"); variables declared within the block are
    /// internal and may be mutated.
    isolate_declared: Vec<HashSet<Symbol>>,
    /// True while checking the inner statement of a `ghost set ...` so that
    /// its name is registered as ghost (not runtime) in `VariableDef`.
    in_ghost_var_def: Cell<bool>,
    /// Functions that access mutable globals (by DefId).
    /// Populated during body checking; used to enforce isolate block restrictions.
    functions_accessing_mutables: HashSet<DefId>,
    /// Side-effect labels per function, computed over the call graph from
    /// the AST BEFORE any body check (order-independent): a function's
    /// label is its DIRECT effects (mutable-global reads, unsafe, @io,
    /// panic, comptime calls) unioned with its transitive callees'.  The
    /// isolate block check requires `MUTABLE_GLOBAL` to be clear; `@pure`
    /// requires a wider forbidden set — one map serves both.
    effect_of: HashMap<DefId, EffectSet>,
    /// Side-effect labels per IMPL METHOD, keyed by the method's OWN DefId
    /// (the "assoc item" identity, mirroring rustc's `AssocItem.def_id` —
    /// allocated in the resolver, so methods are addressable independently
    /// of their impl).  Computed in `collect_function_effects`: a method's
    /// DIRECT effects (mutable-global reads, unsafe, @io/@diverges/
    /// comptime calls — collected by the same BodyInfoCollector) unioned
    /// with its free-function callees' (already-propagated) effects, then
    /// closed transitively over method→method edges.  The isolate/@pure
    /// checks look this map up at the call site via `lookup_method`'s
    /// returned method DefId.
    method_effect_of: HashMap<DefId, EffectSet>,
    /// Explicit-lifetime OUTLIVES constraints collected for the CURRENT
    /// function signature: each pair `('a, 'b)` means `'a: 'b` (region
    /// `'a` outlives region `'b`).  Collected from the explicit `&'a T`
    /// annotations in the parameter and return types (SYNTAX.md
    /// §Explicit Lifetime Parameters) and solved by the region solver —
    /// a return reference may only use a lifetime that provably outlives
    /// its source (rustc's "lifetime may not live long enough").
    region_outlives: Vec<(Symbol, Symbol)>,
    /// The CURRENT function's signature region sets, retained across the
    /// body check so the region solver can run a SECOND pass AFTER the
    /// body (the body's collected `region_subtype_outlives` pairs are
    /// merged into `region_outlives` at the end — signature-only solving
    /// would silently accept body-level `&'a`/`&'b` mismatches, the
    /// fail-open).
    current_param_regions: HashSet<Symbol>,
    current_ret_regions: Vec<Symbol>,
    /// Whether the current function is annotated `@trusted`.
    current_function_trusted: bool,
    /// Whether the current function is annotated `@pure` — the method-call
    /// effect check at the call site (receiver type known there) uses this
    /// to enforce the @pure forbidden set on methods too (the function-level
    /// check in the FunctionDef arm covers free-function chains only).
    current_function_pure: bool,
    /// Registry of comptime functions: name → (param_names, body).
    /// Populated as the checker encounters `comptime def` functions and
    /// passed to ComptimeEvalContext for comptime block evaluation.
    comptime_fn_registry: HashMap<Symbol, (Vec<Symbol>, Vec<HirStmt<'input>>)>,
    /// Whether we are currently in the comptime-function-body pass (Pass 2).
    /// When true, ComptimeBlock evaluation is deferred to after Pass 2 so
    /// that forward references between comptime functions work correctly.
    comptime_fn_pass: bool,
    /// Deferred comptime blocks collected during Pass 2.  Evaluated after
    /// all comptime function bodies are registered.
    /// Each entry is (captures, body_hir, ty, span).
    deferred_comptime_blocks: Vec<(Vec<(Symbol, Span)>, Vec<HirStmt<'input>>, TypeId, Span)>,
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
    literal_values: HashMap<Symbol, Vec<ComptimeValue<'input>>>,
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
    /// trait solver in `check_stmt(FunctionDef<'input>)`.
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
pub enum ComptimeControlFlow<'input> {
    Return(Option<HirExpr<'input>>),
    Break(Option<String>),
    Continue(Option<String>),
}

impl<'input> std::fmt::Display for ComptimeControlFlow<'input> {
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
/// A `&'a TypeContext<'input>` reference would conflict with the enclosing
/// method's `&mut self` borrow, because `self.ctx: &'a mut TypeContext<'input>`
/// cannot be re-borrowed as shared while the method borrows `self`
/// mutably for other operations.  A raw pointer bypasses this, and is
/// sound because the guard is always a local variable created and
/// consumed within a single method invocation — it never outlives the
/// `TypeContext<'input>` it points to.
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
pub(crate) struct GadtArmGuard<'input> {
    /// Pointer to the `GadtContext` whose arm depth / fact registry this
    /// guard manages.  Raw pointer because a `&GadtContext` reference
    /// would conflict with the enclosing method's `&mut self` borrow (see
    /// struct doc).  Sound because the guard is always a method-local
    /// variable that never outlives the `TypeContext<'input>` owning the
    /// `GadtContext`.
    gadt: *const crate::hir::types::GadtContext<'input>,
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

impl<'input> GadtArmGuard<'input> {
    /// Create a guard that also enters a fresh TcLevel region.  The region
    /// is restored on drop (even on early return), so InferVars created in
    /// the arm body are at a deeper level and escaping them is caught by
    /// the TcLevel escape check in `unify_internal_impl`.
    pub fn enter_region(
        ctx: &TypeContext<'input>,
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
pub(crate) struct PrecreateGuard<'input> {
    ctx: *const crate::hir::types::GadtContext<'input>,
    depth: usize,
    committed: bool,
}

impl<'input> PrecreateGuard<'input> {
    pub fn enter(ctx: &TypeContext<'input>, depth: usize) -> Self {
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

impl<'input> Drop for PrecreateGuard<'input> {
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

impl<'input> Drop for GadtArmGuard<'input> {
    fn drop(&mut self) {
        // SAFETY: `self.gadt` is always non-null — set by `enter_region`
        // from a `&GadtContext` reference (which is never null).
        debug_assert!(!self.gadt.is_null());
        // Pop GADT arms back to the saved depth — the arm's refinements
        // are discarded.  (SAFETY: `self.gadt` points to the `GadtContext`
        // owned by the `TypeContext<'input>` that outlives this guard — the guard
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
        // Depth-discipline violations are FAIL-CLOSED in ALL builds (the
        // shared exist-skolem stack has multiple
        // RAII guards; a depth mismatch means a guard's truncation would
        // discard another arm's frames — must panic, not saturate).
        if !std::thread::panicking() {
            assert!(
                self.saved_exist_depth <= current_len,
                "GadtArmGuard: saved_exist_depth {} exceeds stack len {} — \
                 a frame was pushed without this guard's depth discipline",
                self.saved_exist_depth,
                current_len,
            );
        } else {
            // During unwinding a second panic would abort the process and
            // mask the original error — saturate instead; the truncate
            // below is already clamped with `.min(current_len)`.
            eprintln!(
                "GadtArmGuard: saved_exist_depth {} exceeds stack len {} (saturating during unwinding)",
                self.saved_exist_depth, current_len,
            );
        }
        gadt.exist_skolems
            .borrow_mut()
            .truncate(self.saved_exist_depth.min(current_len));
    }
}

impl<'input: 'a, 'a> TypeChecker<'a, 'input> {
    pub fn new(
        ctx: &'a mut TypeContext<'input>,
        symbols: &'a SymbolTable<'input>,
        trait_env: &'a mut TraitEnv<'input>,
        resolution_map: ResolutionMap<'input>,
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
        ctx: &'a mut TypeContext<'input>,
        symbols: &'a SymbolTable<'input>,
        trait_env: &'a mut TraitEnv<'input>,
        resolution_map: ResolutionMap<'input>,
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
            smt_solver: None,
            next_inv_binder_id: 0,
            last_checked_program: None,
            signature_facts: Vec::new(),
            pending_runtime_checks: Vec::new(),
            function_finally: HashMap::new(),
            pending_gadt_constructs: Vec::new(),
            method_cache: HashMap::new(),
            source,
            current_function: None,
            current_return_type: None,
            pending_function_hints: Vec::new(),
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
            isolate_declared: Vec::new(),
            must_handle_sources: RefCell::new(HashSet::new()),
            in_ghost_var_def: Cell::new(false),
            functions_accessing_mutables: HashSet::new(),
            effect_of: HashMap::new(),
            method_effect_of: HashMap::new(),
            region_outlives: Vec::new(),
            current_param_regions: HashSet::new(),
            current_ret_regions: Vec::new(),
            current_function_trusted: false,
            current_function_pure: false,
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
    pub fn check_program(
        &mut self,
        program: &Program<'input>,
    ) -> Result<HirProgram<'input>, DiagCtxt> {
        let mut items = Vec::new();

        // Wrap the entire program in a literal value scope so that
        // top-level variable definitions (`set x = 42`) have a scope
        // frame to track into.  Popped at the end of this function.
        self.push_literal_scope();

        // Wrap the entire program in an inference scope so that
        // top‑level statements (variable defs, expression stmts, etc.)
        // also have their Eq/Impl/Match constraints solved and finalized.
        // Previously the solver only ran inside function bodies via
        // enter_inference_scope in check_stmt(FunctionDef<'input>).
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
                            // Dispatch the error code by the cause: a
                            // comptime sandbox violation (e.g. calling
                            // @trusted/@io, or an item declaration inside a
                            // comptime block) is E081 ("comptime sandbox
                            // violation"); other evaluation errors are E080.
                            let code = if matches!(
                                e,
                                crate::hir::comptime::error::ComptimeError::SandboxViolation(_)
                            ) {
                                "E081"
                            } else {
                                "E080"
                            };
                            Err(ctxt.push(Diagnostic::error(msg).with_code_str(code)).into())
                        }
                    }
                },
            );
        }

        // Pass 3: type-check remaining items (non-comptime functions,
        // comptime blocks, type defs, etc.) in order.
        // Skip items whose @cfg condition is not met.
        // The borrow-signature pre-registration: the first
        // pass registers every function's A(ρ) facts BEFORE any body check
        // — the cross-function loan issuance becomes order-independent.
        pre_register_signatures(self, program);
        // Per-function side-effect labels over the call graph, also before
        // any body check (order-independent): the isolate block check
        // below rejects MUTABLE_GLOBAL effects (indirect access too — A
        // calls B, B reads a mutable global ⇒ A is rejected in an isolate
        // block), and `@pure` checks a wider forbidden set.
        collect_function_effects(self, program);
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
        // check_stmt(FunctionDef<'input>), so by the time we reach here, only
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
            let ctx: &mut TypeContext<'input> = &mut self.ctx;
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
                        .lookup_trait_by_def_id(
                            trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)),
                        )
                        .and_then(|tb| {
                            self.symbols.trait_name_by_def_id(
                                trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)),
                            )
                        })
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
                // The post-solve GADT validation phase (the committee
                // ruling — solve → default → validate): constructions
                // deferred during inference are validated now that the
                // type arguments are concrete.
                self.validate_pending_gadt_constructs();
                // ── Phase B: flow-sensitive borrow/move post-passes ──
                // All function bodies were type-checked to HIR and their
                // A(ρ) signature facts finalized (Phase A).  Run the
                // borrow check + affine move check over ALL of them NOW,
                // against the COMPLETE signature registry — cross-function
                // loan facts are order-independent (a caller appearing
                // before its callee sees the callee's finalized HIR-level
                // signature, not the AST-level under-approximation).
                // Runs AFTER solve+commit so `type_is_copy` in
                // `collect_non_copy_roots` sees resolved types.
                for (_name, body_stmts) in &self.pending_runtime_checks {
                    let finally = self
                        .function_finally
                        .get(_name)
                        .map_or(&[][..], |v| v.as_slice());
                    for err in crate::hir::polonius::borrow_check_function(
                        body_stmts,
                        finally,
                        &self.signature_facts,
                        &self.ctx,
                    ) {
                        self.diagnostics
                            .push(crate::hir::cfg_graph::borrow_error_diagnostic(&err));
                    }
                    // The static move check (the CFG-level affine
                    // use-after-move).  The non-Copy roots come from the
                    // variable types (the §Copy `type_is_copy` — the
                    // String and the aggregates containing it).
                    let mut non_copy_roots: Vec<Symbol> = Vec::new();
                    collect_non_copy_roots(body_stmts, &self.ctx, &mut non_copy_roots);
                    let finally = self
                        .function_finally
                        .get(_name)
                        .map_or(&[][..], |v| v.as_slice());
                    for msg in crate::hir::cfg_graph::check_function_moves(
                        body_stmts,
                        finally,
                        &non_copy_roots,
                    ) {
                        self.diagnostics
                            .push(Diagnostic::error(&msg).with_code_str("E114"));
                    }
                }
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
                    let ctx: &mut TypeContext<'input> = &mut self.ctx;
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
                // function returns.  The TypeContext<'input> bindings are already finalized
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
        let prog = HirProgram {
            items,
            span: program.span,
        };
        if self.diagnostics.has_errors() {
            // Keep the fully-built HIR accessible even when the
            // borrow-check post-pass rejects the program — the Polonius
            // equivalence tests inspect the bodies of REJECTED programs
            // (the error direction).
            self.last_checked_program = Some(prog.clone());
            Err(mem::take(&mut self.diagnostics))
        } else {
            Ok(prog)
        }
    }

    /// Loop-invariant inference (integration, hint-only per the
    /// 2026-08-13 committee ruling): translate the HIR loop, run the
    /// widened fixpoint, gate the candidates through the SMT consistency
    /// check, and report the survivors as a note.  Candidates NEVER
    /// discharge obligations by themselves — the solver stays the
    /// authority.  Non-translatable loops are skipped silently
    /// (fail-closed).
    ///
    /// Note (2026-08-13/14 rulings, INVARIANT VERIFICATION WIRED): the
    /// declared `while` loop `invariant` clause is an OBLIGATION — per the
    /// language-owner ruling, the clause is verified, not decorative.  The
    /// inferred BII candidates are the premise; the check is exact
    /// difference-constraint self-discharge first (`expr_entails_typed`,
    /// founder bifurcation), SMT discharge fallback (`SmtSolver::discharge`),
    /// and a compile error when the invariant cannot be proven.  The
    /// `decreases` clause and `@hint(assertion)` wiring follow separately.
    /// The BII candidates themselves stay advisory notes (the solver stays
    /// the authority for what is INferred; only what the USER declares is
    /// an obligation).
    /// Does a HIR expression use an explicit wrap-around operator
    /// (`+%`/`-%`/`*%`)?  HIR-side mirror of `expr_uses_wrap` — used to
    /// keep synthesis (`use_bv`) and verification (wrap-routing) in
    /// agreement: a loop whose body/condition wraps is synthesized under
    /// bit-vector semantics and its obligations are discharged via
    /// `SmtSolver::discharge_bv`.
    fn hir_uses_wrap(&self, e: &crate::hir::hir::HirExpr<'input>) -> bool {
        match e {
            crate::hir::hir::HirExpr::BinaryOp {
                op, left, right, ..
            } => {
                matches!(
                    op,
                    crate::ast::BinOp::AddWrap
                        | crate::ast::BinOp::SubWrap
                        | crate::ast::BinOp::MulWrap
                ) || self.hir_uses_wrap(left)
                    || self.hir_uses_wrap(right)
            }
            crate::hir::hir::HirExpr::UnaryOp { expr, .. }
            | crate::hir::hir::HirExpr::TypeAnnotated { expr, .. }
            | crate::hir::hir::HirExpr::Try { expr, .. }
            | crate::hir::hir::HirExpr::Catch { expr, .. }
            | crate::hir::hir::HirExpr::LeaveWith { expr, .. }
            | crate::hir::hir::HirExpr::Move(expr, ..) => self.hir_uses_wrap(expr),
            crate::hir::hir::HirExpr::Call { callee, args, .. } => {
                // Wrap propagation through calls: if the callee is a free
                // function whose effect label carries WRAP (it — or
                // transitively something it calls — uses `+%`/`-%`/`*%`),
                // the call site is wrap-semantics even though no wrap
                // operator is syntactically present here.  Unknown callees
                // (unresolved names) conservatively fall back to
                // syntactic recursion only.
                let callee_wraps = matches!(&**callee, crate::hir::hir::HirExpr::Ident(f, _, _))
                    && self
                        .symbols
                        .lookup_function(*match &**callee {
                            crate::hir::hir::HirExpr::Ident(f, _, _) => f,
                            _ => unreachable!(),
                        })
                        .and_then(|b| self.effect_of.get(&b.def_id))
                        .is_some_and(|e| e.contains(EffectSet::WRAP));
                // Method-call propagation: `r.foo()` where `foo` (or
                // transitively something it calls) uses `+%` — resolve the
                // receiver variable's type to the method's DefId and check
                // `method_effect_of`.  Only simple-variable receivers are
                // resolved (`r.foo()`); complex receivers (`get().foo()`,
                // `self.foo()` outside an impl body) conservatively fall
                // back to syntactic recursion.
                let method_wraps =
                    if let crate::hir::hir::HirExpr::FieldAccess { base, field, .. } = &**callee
                        && let crate::hir::hir::HirExpr::Ident(r, _, _) = &**base
                        && let Some(receiver_ty) = self.local_variable_types.get(*r)
                        && let Some(receiver_def) = self.ctx.get_def_id_for_type(receiver_ty)
                        && let Some(method_def) =
                            self.symbols.lookup_method_def_id(receiver_def, *field)
                        && let Some(eff) = self.method_effect_of.get(&method_def)
                    {
                        eff.contains(EffectSet::WRAP)
                    } else {
                        false
                    };
                callee_wraps
                    || method_wraps
                    || self.hir_uses_wrap(callee)
                    || args.iter().any(|a| self.hir_uses_wrap(a))
            }
            crate::hir::hir::HirExpr::Index { base, index, .. } => {
                self.hir_uses_wrap(base) || self.hir_uses_wrap(index)
            }
            crate::hir::hir::HirExpr::Tuple(elems, ..)
            | crate::hir::hir::HirExpr::Array(elems, ..) => {
                elems.iter().any(|x| self.hir_uses_wrap(x))
            }
            crate::hir::hir::HirExpr::Closure { body, .. }
            | crate::hir::hir::HirExpr::UnsafeBlock { body, .. } => {
                body.iter().any(|s| self.hir_stmt_uses_wrap(s))
            }
            _ => false,
        }
    }

    /// Does a HIR statement (or anything nested in it) use a wrap operator?
    fn hir_stmt_uses_wrap(&self, s: &crate::hir::hir::HirStmt<'input>) -> bool {
        match s {
            crate::hir::hir::HirStmt::Expression(e) => self.hir_uses_wrap(e),
            crate::hir::hir::HirStmt::VariableDef { value, .. } => {
                value.as_ref().is_some_and(|v| self.hir_uses_wrap(v))
            }
            crate::hir::hir::HirStmt::While { cond, body, .. } => {
                self.hir_uses_wrap(cond) || body.iter().any(|s| self.hir_stmt_uses_wrap(s))
            }
            crate::hir::hir::HirStmt::WhileLet {
                scrutinee, body, ..
            } => self.hir_uses_wrap(scrutinee) || body.iter().any(|s| self.hir_stmt_uses_wrap(s)),
            crate::hir::hir::HirStmt::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                self.hir_uses_wrap(cond)
                    || then_branch.iter().any(|s| self.hir_stmt_uses_wrap(s))
                    || else_branch
                        .as_ref()
                        .is_some_and(|b| b.iter().any(|s| self.hir_stmt_uses_wrap(s)))
            }
            crate::hir::hir::HirStmt::For { body, .. }
            | crate::hir::hir::HirStmt::Loop { body, .. }
            | crate::hir::hir::HirStmt::ComptimeBlock { body, .. } => {
                body.iter().any(|s| self.hir_stmt_uses_wrap(s))
            }
            crate::hir::hir::HirStmt::Assign { target, value, .. } => {
                self.hir_uses_wrap(target) || self.hir_uses_wrap(value)
            }
            crate::hir::hir::HirStmt::Return { value, .. } => {
                value.as_ref().is_some_and(|v| self.hir_uses_wrap(v))
            }
            _ => false,
        }
    }

    /// Extract the loop postcondition from the current function's
    /// `ensures` clause. Returns `Some` only when the `ensures`
    /// expression converts to a `Cond` over the loop variables;
    /// otherwise `None` (ϕ₃ vacuously true).
    ///
    /// NOTE: this is a free associated function, not a `&self` method —
    /// `infer_loop_hints` holds `&mut self.smt_solver` via `smt` when
    /// calling it, and a method call would borrow all of `self` and
    /// conflict; passing `&self.guarantee_chain` keeps the borrow
    /// field-scoped and coexists.
    fn extract_loop_postcondition(
        chain: &GuaranteeChain<'input>,
        vars: &[Symbol],
        signed: &[bool],
    ) -> Option<crate::hir::loop_ir::Cond> {
        let guarantee = chain.current()?;
        let ast_expr = guarantee.ast_expr.as_ref()?;
        ast_expr_to_cond(ast_expr, vars, signed)
    }

    fn infer_loop_hints(
        &mut self,
        cond: &crate::hir::hir::HirExpr<'input>,
        body: &[crate::hir::hir::HirStmt<'input>],
        span: crate::ast::Span,
        invariant: Option<&crate::ast::Expr<'input>>,
        decreases: Option<&crate::ast::Expr<'input>>,
    ) {
        let Some((vars, instrs)) = crate::hir::loop_infer::hir_loop_to_loop_instrs(
            cond,
            body,
            // Type-level overflow policy (suffix >
            // type policy > default trap) — a plain `+`/`-` on a
            // saturating type lowers to AddSat.
            &|sym| {
                self.local_variable_types
                    .get(sym)
                    .map(|ty| self.ctx.overflow_policy_of(ty))
            },
        ) else {
            // Untranslatable loop — inference skipped, ничего страшного
            // (fail-closed).  But a DECLARED invariant/decreases is an
            // obligation that cannot be discharged without a model —
            // reject it.
            if let Some(inv) = invariant {
                self.reject_unverified_invariant(inv, span);
            }
            if let Some(dec) = decreases {
                self.reject_unverified_decreases(dec, span);
            }
            return;
        };
        let Some(arena) = self.ctx.arena else {
            return;
        };
        // Seed the DBM with the PRE-LOOP initial values of the loop
        // variables.  Without an `init` the matrix starts at top (all ∞)
        // and `join(top, _) = top`, so the fixpoint converges to top at
        // once and `dbm_to_invariant_exprs` yields zero candidates — the
        // hint channel would be wired but inert.  Only comptime-known
        // integer bindings seed a `ConstVar`; unknown values are omitted
        // (that variable simply starts at top — fail-closed, the hint
        // stays advisory and SMT remains the authority).
        let init: Vec<crate::hir::loop_infer::LoopInstr> = vars
            .iter()
            .enumerate()
            .filter_map(|(idx, sym)| match self.get_literal_value(sym) {
                Some(crate::hir::comptime::value::ComptimeValue::Int(c)) => {
                    Some(crate::hir::loop_infer::LoopInstr::ConstVar(idx, *c))
                }
                _ => None,
            })
            .collect();
        // use_bv (synthesis/verification agreement): a loop whose
        // body/condition uses an explicit wrap-around operator
        // (`+%`/`-%`/`*%`) is synthesized under BIT-VECTOR semantics
        // (`use_bv: true`) — matching the verification-side wrap-routing
        // in `obligation_provable`, which discharges wrap obligations via
        // `SmtSolver::discharge_bv`.  LIA cannot express modular
        // arithmetic, so both sides must agree on the wrap-around reading.
        // Computed BEFORE `smt` is borrowed (both touch `self` —
        // `hir_uses_wrap` needs `&self`).
        let use_bv = self.hir_uses_wrap(cond) || body.iter().any(|s| self.hir_stmt_uses_wrap(s));
        let smt = self
            .smt_solver
            .get_or_insert_with(|| crate::hir::smt::SmtSolver::new("z3"));
        // Per-variable bit-widths for the template rows (Posita
        // has bit-width types — Int<N>/UInt<N>); a variable whose type is
        // unknown defaults to 64 bits.
        let bit_widths: Vec<u8> = vars
            .iter()
            .map(|sym| {
                self.local_variable_types
                    .get(*sym)
                    .and_then(|ty| self.ctx.bits_of_int(ty))
                    .map(|b| u8::try_from(b).unwrap_or(u8::MAX))
                    .unwrap_or(64)
            })
            .collect();
        // Per-variable signedness for the template queries: `Int<N>` rows
        // compare SIGNED (`bvsle`/`bvsge`), `UInt<N>` and unknowns UNSIGNED
        // (`bvule`/`bvuge`) — the same decision the verification side makes
        // (`verify_loop_decreases` / `obligation_provable`), so a guard is
        // read the same way in synthesis and verification.
        let signed: Vec<bool> = vars
            .iter()
            .map(|sym| {
                self.local_variable_types
                    .get(*sym)
                    .map(|ty| self.ctx.is_signed(ty))
                    .unwrap_or(false)
            })
            .collect();
        // Pre-fill the DBM with compile-time-known type
        // ranges (Int<N> / UInt<N> value domains) so the fixpoint starts
        // tighter. `apply_type_bounds` (the lower-bound encoding bug) was
        // fixed earlier; enabling the seeding is the remaining wiring.
        // Widths > 127 bits exceed the i128 DBM domain — skipped (None).
        let type_bounds: Vec<(Option<i128>, Option<i128>)> = vars
            .iter()
            .map(|sym| {
                match self
                    .local_variable_types
                    .get(*sym)
                    .map(|ty| self.ctx.get(ty))
                {
                    Some(crate::hir::types::TypeData::Int { bits, .. }) if *bits <= 127 => {
                        let half = 1i128 << (*bits as usize - 1);
                        (Some(-half), Some(half - 1))
                    }
                    Some(crate::hir::types::TypeData::UInt { bits, .. }) if *bits <= 127 => {
                        (Some(0), Some((1i128 << *bits as usize) - 1))
                    }
                    _ => (None, None),
                }
            })
            .collect();
        // Propose-and-refine over the template domain first.  The
        // The BiiLoopProblem path takes priority (Phase A: edge-based
        // transition encoding + independent verification
        // `verify_template_against_problem`, sharing
        // `encode_edge_inductiveness` with synthesis); on failure (lowering
        // / synthesis / verification) fall back to the old
        // `synthesize_bitwise_bii`, then to `dbm_fixpoint`, preserving the
        // baseline hint behavior.
        // Candidates from either path encode at the VARIABLES'
        // signedness downstream (verify_loop_decreases /
        // obligation_provable); DBM-fallback candidates are
        // difference-constraint expressions over the same variables.

        // Independent verification gets a LONGER budget than synthesis
        // (committee ruling on the verifier-overhead decision point):
        // re-tune the shared solver instance in place, then restore.
        // Counterexample → report an error (synthesis-layer bug); the
        // report is deferred until the `smt` borrow ends (see below).
        // Inconclusive (timeout / unknown) → fall back silently.
        // Separate unmodified variables (external
        // symbols such as `n` in `while i < n`) into params for the
        // BiiLoopProblem path — the loop-carried variables are the ones
        // the body modifies; the rest become read-only params (their
        // type-range conditions are spliced into the quantified
        // antecedents). Hints are produced only over
        // the loop variables.
        let separated =
            crate::hir::loop_ir::separate_loop_params(&vars, &init, &instrs, &bit_widths, &signed);
        let (sep_vars, sep_init, sep_body, sep_bws, sep_signed, sep_params): (
            &[crate::symbol::Symbol],
            &[crate::hir::loop_infer::LoopInstr],
            &[crate::hir::loop_infer::LoopInstr],
            &[u8],
            &[bool],
            &[crate::hir::loop_ir::BiiVar],
        ) = match &separated {
            Some(sep) => (
                &sep.vars,
                &sep.init,
                &sep.body,
                &sep.bit_widths,
                &sep.signed,
                &sep.params,
            ),
            None => (&vars, &init, &instrs, &bit_widths, &signed, &[]),
        };
        let mut verify_counterexample = false;
        // Extract the postcondition from the ensures clause (computed
        // outside the closure to avoid borrow conflicts). Pass a
        // field-scoped `&self.guarantee_chain` rather than a method call:
        // `smt` holds `&mut self.smt_solver`, and a method call would
        // borrow all of `self` and conflict.
        let post_cond: Option<crate::hir::loop_ir::Cond> =
            Self::extract_loop_postcondition(&self.guarantee_chain, sep_vars, sep_signed);
        // Wrap loops (`use_bv`) carry no trap
        // definedness — their arithmetic is total (`+%`/`-%`).
        let problem_path = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            sep_vars, sep_init, sep_body, sep_bws, sep_signed, sep_params, !use_bv,
        )
        .and_then(|mut problem| {
            problem.post = post_cond.clone();
            // Budget floor applied HERE (production caller), not inside
            // the drivers: the paper's raw count ≈ 2×4W (Theorem 5.5),
            // so 8W + 64 covers wide templates without inflating a
            // test's explicit budget.
            let bws_all: Vec<u8> = sep_bws
                .iter()
                .copied()
                .chain(sep_params.iter().map(|p| p.bw))
                .collect();
            let signed_all: Vec<bool> = sep_signed
                .iter()
                .copied()
                .chain(sep_params.iter().map(|p| p.signed))
                .collect();
            let budget = crate::hir::bii::query_budget_floor(
                bws_all.len(),
                &bws_all,
                &signed_all,
                &problem.saturates,
            )
            .max(512);
            let tpl = crate::hir::bii::synthesize_problem_bii(smt, &problem, budget, use_bv)?;
            smt.set_timeout(crate::hir::smt::VERIFY_TIMEOUT_MS);
            let outcome =
                crate::hir::bii::verify_template_against_problem(smt, &problem, &tpl, use_bv);
            smt.set_timeout(crate::hir::smt::Z3_TIMEOUT_MS);
            match outcome {
                // `TrapUnproven` — checks 1–2 passed (the BII is
                // inductive, the hint is sound) but check 3 (trap
                // absence, `A ∧ G ⟹ def`) failed: the template domain
                // cannot prove the loop trap-free (e.g. a strided
                // counter whose interval BII cannot express the
                // stride). NOT a synthesis bug — the hint is emitted
                // exactly as with Verified. Surfacing the trap signal
                // as a user-visible diagnostic (error vs warning,
                // strict vs non-strict) is an L3 decision — deferred,
                // not silently dropped. (To instead BLOCK compilation
                // on check-3 failure, route this arm alongside
                // Counterexample.)
                crate::hir::bii::VerifyOutcome::Verified
                | crate::hir::bii::VerifyOutcome::TrapUnproven
                | crate::hir::bii::VerifyOutcome::PostUnproven => Some(
                    crate::hir::bii::template_to_invariant_exprs(arena, &tpl, sep_vars),
                ),
                crate::hir::bii::VerifyOutcome::Counterexample => {
                    verify_counterexample = true;
                    None
                }
                crate::hir::bii::VerifyOutcome::Inconclusive => None,
            }
        });
        let candidates = match problem_path {
            Some(rows) if !rows.is_empty() => rows,
            Some(_) => {
                // The new path converged but all rows are trivial
                // (full-range) — same template domain and result as the
                // old path, so fall straight back to DBM (an empty
                // candidate set would make the decreases query range over
                // states the loop can never reach and falsely reject).
                crate::hir::loop_infer::infer_loop_invariant_exprs(
                    arena,
                    &vars,
                    &init,
                    &instrs,
                    100,
                    2,
                    Some(&type_bounds),
                )
            }
            None => match crate::hir::bii::synthesize_bitwise_bii(
                smt,
                &vars,
                &init,
                &instrs,
                &bit_widths,
                &signed,
                crate::hir::bii::query_budget_floor(vars.len(), &bit_widths, &signed, &[]).max(512),
                use_bv,
            ) {
                Some(tpl) => {
                    let rows = crate::hir::bii::template_to_invariant_exprs(arena, &tpl, &vars);
                    if rows.is_empty() {
                        crate::hir::loop_infer::infer_loop_invariant_exprs(
                            arena,
                            &vars,
                            &init,
                            &instrs,
                            100,
                            2,
                            Some(&type_bounds),
                        )
                    } else {
                        rows
                    }
                }
                None => crate::hir::loop_infer::infer_loop_invariant_exprs(
                    arena,
                    &vars,
                    &init,
                    &instrs,
                    100,
                    2,
                    Some(&type_bounds),
                ),
            },
        };
        if !candidates.is_empty() && smt.check_hints(&candidates) {
            let msgs: Vec<String> = candidates.iter().map(|e| format!("{:?}", e)).collect();
            self.diagnostics.push(
                crate::diagnostics::Diagnostic::note("inferred loop invariant (hint candidate)")
                    .with_span(span)
                    .with_label(span, msgs.join(" and ")),
            );
        }
        // Independent verification found a counterexample: the synthesized
        // template is NOT inductive — a synthesis-layer bug. Report an
        // error (committee ruling on the verifier-overhead decision point).
        if verify_counterexample {
            self.diagnostics.push(
                crate::diagnostics::Diagnostic::error(
                    "internal: synthesized loop invariant failed independent verification",
                )
                .with_span(span)
                .with_label(
                    span,
                    "the BII synthesis result is not inductive — a compiler bug",
                ),
            );
        }
        // The DECLARED loop `invariant` is an obligation — verify it
        // against the inferred candidates (exact self-discharge first,
        // SMT discharge fallback, error when unprovable).  Per the
        // language-owner ruling the clause must be verified, not decorative.
        if let Some(inv) = invariant {
            self.verify_loop_invariant(inv, &candidates, &vars, span, use_bv);
        }
        // The DECLARED loop `decreases` measure must strictly decrease
        // on every iteration — an ∃∀ query, reject when unprovable.
        if let Some(dec) = decreases {
            self.verify_loop_decreases(dec, &candidates, &vars, &instrs, span, use_bv);
        }
        // The function's `@hint(assertion)` list — verify each against
        // the loop candidates (a user hint is an assertion, not dead
        // storage).  Take the list out to avoid borrowing `self` while
        // `verify_loop_hint` takes `&mut self`.
        let pending_hints = std::mem::take(&mut self.pending_function_hints);
        for hint in &pending_hints {
            self.verify_loop_hint(hint, &candidates, &vars, span, use_bv);
        }
        self.pending_function_hints = pending_hints;
        // The `decreases` candidate — wired into the same
        // advisory-only channel (SMT stays the authority).  Re-acquire the
        // solver here: the earlier `smt` borrow must end before
        // `verify_loop_invariant` (which takes `&mut self`).
        if let Some(dec) = crate::hir::loop_infer::infer_loop_decreases_expr(arena, &vars, &instrs)
        {
            let smt = self
                .smt_solver
                .get_or_insert_with(|| crate::hir::smt::SmtSolver::new("z3"));
            if smt.check_hints(&[dec]) {
                self.diagnostics.push(
                    crate::diagnostics::Diagnostic::note(
                        "inferred loop decreases (hint candidate)",
                    )
                    .with_span(span)
                    .with_label(span, format!("{:?}", dec)),
                );
            }
        }
    }

    /// Verify a DECLARED loop `invariant` clause — an obligation per
    /// the language-owner ruling, not decoration.  Exact
    /// difference-constraint self-discharge first (`expr_entails_typed`,
    /// founder bifurcation: an exact decision is self-verifying), then SMT
    /// discharge (`SmtSolver::discharge`, `∧candidates ⟹ invariant` =
    /// unsat), and finally a compile error when unprovable (fail-closed).
    fn verify_loop_invariant(
        &mut self,
        inv: &crate::ast::Expr<'input>,
        candidates: &[&'input crate::ast::Expr<'input>],
        vars: &[Symbol],
        span: crate::ast::Span,
        use_bv: bool,
    ) {
        if !self.obligation_provable(inv, candidates, vars, use_bv) {
            self.reject_unverified_invariant(inv, span);
        }
    }

    /// Is `obligation` entailed by `∧candidates`?  Exact
    /// difference-constraint self-discharge first (`expr_entails_typed` —
    /// the founder-bifurcation exact decision), then SMT discharge
    /// (`SmtSolver::discharge`, unsat = entailed).  Fail-closed: `false`
    /// on unknown/unavailable/untranslatable — the caller rejects.
    fn obligation_provable(
        &mut self,
        obligation: &crate::ast::Expr<'input>,
        candidates: &[&'input crate::ast::Expr<'input>],
        vars: &[Symbol],
        use_bv: bool,
    ) -> bool {
        // Discreteness-aware comparison (`X > 0` ≡ `X >= 1`) applies when
        // every loop variable is an integer type.
        let is_int = vars.iter().all(|sym| {
            self.local_variable_types
                .get(*sym)
                .map(|ty| self.ctx.is_integer(ty))
                .unwrap_or(false)
        });
        // Exact path: `∧candidates ⟹ obligation` inside the
        // difference-constraint sub-language. BV gate (explicit
        // contract, committee ruling boundary): a wrap-semantics loop
        // (use_bv) skips the LIA exact channel entirely — its
        // candidates/obligations belong to the BV discharge below (the
        // same routing as the synthesis side). LIA order relations
        // happen to be preserved for pure bounds, but relying on that
        // is an implicit argument from expression shapes; the gate
        // keeps the boundary explicit.
        if !use_bv && !candidates.is_empty() {
            let Some(arena) = self.ctx.arena else {
                return false;
            };
            let mut conj = candidates[0];
            for c in &candidates[1..] {
                conj = arena.alloc(crate::ast::Expr::BinaryOp {
                    left: conj,
                    op: crate::ast::BinOp::And,
                    right: *c,
                    span: crate::ast::Span::new(0, 0),
                });
            }
            match crate::hir::type_eq::expr_entails_typed(conj, obligation, is_int) {
                Some(true) => return true, // exact self-discharge — verified.
                Some(false) | None => { /* fall through to SMT */ }
            }
        }
        // SMT path: `∧candidates ⟹ obligation` (assert the negation of the
        // obligation, expect `unsat`).  Unavailable/unknown fail closed.
        //
        // Wrap-routing: if the obligation or ANY candidate uses an
        // explicit wrap-around operator (`+%`/`-%`/`*%`), LIA cannot express
        // modular arithmetic — discharge under BIT-VECTOR semantics
        // (`SmtSolver::discharge_bv`) instead, with the loop variables'
        // bit-width.  This matches the synthesis side: a wrap-loop is
        // synthesized with `use_bv: true` (see `infer_loop_hints`), so the
        // candidate is only acceptable under the same wrap-around reading.
        // Wrap reached THROUGH a function call is also routed: the
        // `callee_wraps` callback resolves the callee's WRAP effect label
        // (computed by `collect_function_effects` over the call graph, so
        // `f` wrapping — directly or transitively — marks its callers).
        // A free-function callee (`f(...)`) is looked up in `effect_of`;
        // a method callee (`r.foo(...)`) resolves the receiver variable's
        // type to the method's DefId and checks `method_effect_of`.
        let callee_wraps = |callee: &crate::ast::Expr| match callee {
            crate::ast::Expr::Ident(f, _) => self
                .symbols
                .lookup_function(*f)
                .and_then(|b| self.effect_of.get(&b.def_id))
                .is_some_and(|e| e.contains(EffectSet::WRAP)),
            crate::ast::Expr::FieldAccess { base, field, .. } => {
                if let crate::ast::Expr::Ident(r, _) = *base
                    && let Some(receiver_ty) = self.local_variable_types.get(*r)
                    && let Some(receiver_def) = self.ctx.get_def_id_for_type(receiver_ty)
                    && let Some(method_def) =
                        self.symbols.lookup_method_def_id(receiver_def, *field)
                    && let Some(eff) = self.method_effect_of.get(&method_def)
                {
                    eff.contains(EffectSet::WRAP)
                } else {
                    false
                }
            }
            _ => false,
        };
        // Wrap routing (with use_bv, matching the synthesis side): a
        // wrap-around loop (`use_bv == true`) discharges its obligations
        // under BIT-VECTOR semantics EVEN when the declared invariant /
        // hint is an ordinary bound (`i <= 254`) with no wrap operator —
        // otherwise the same loop's `decreases` is verified under BV while
        // its `invariant` would silently fall to LIA (the routing gap).
        let uses_wrap = use_bv
            || crate::hir::smt::expr_uses_wrap(obligation, &callee_wraps)
            || candidates
                .iter()
                .any(|c| crate::hir::smt::expr_uses_wrap(c, &callee_wraps));
        let smt = self
            .smt_solver
            .get_or_insert_with(|| crate::hir::smt::SmtSolver::new("z3"));
        if uses_wrap {
            // Per-variable bit-widths for the BV query: each `Int<N>` /
            // `UInt<N>` variable declares at its OWN width (a variable of
            // unknown width defaults to 64 — the checker's default).  A
            // uniform width would truncate mixed-width loops (an `Int<8>`
            // loop with an `Int<16>` variable) and falsely discharge —
            // or falsely reject — their obligations.
            let widths: std::collections::HashMap<Symbol, u8> = vars
                .iter()
                .filter_map(|sym| {
                    let ty = self.local_variable_types.get(*sym)?;
                    self.ctx
                        .bits_of_int(ty)
                        .map(|b| (*sym, u8::try_from(b).unwrap_or(u8::MAX)))
                })
                .collect();
            // Per-variable signedness for the BV query: `Int<N>` variables
            // are SIGNED (comparators `bvsle`/`bvsge`); `UInt<N>` and
            // unknowns are unsigned (`bvule`/`bvuge`).  Without this set, a
            // negative-bound wrap obligation (e.g. `x ≥ -5` on `Int<8>`,
            // where `-5` is the two's-complement `0xFB`) would be compared
            // as an unsigned pattern and misjudged.
            let signed: std::collections::HashSet<String> = vars
                .iter()
                .filter_map(|sym| {
                    let ty = self.local_variable_types.get(*sym)?;
                    if matches!(self.ctx.get(ty), crate::hir::types::TypeData::Int { .. }) {
                        Some(sym.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            // Hints encode at the VARIABLES' signedness
            // (hints_unsigned = false) — the same stale-policy fix as
            // `verify_loop_decreases`' candidate encoding: post-refactor
            // signed rows carry true signed bounds, and forcing them
            // unsigned empties negative-bound premises (vacuous
            // entailment). `discharge_bv`'s `hints_unsigned` parameter
            // is now vestigial at this call site (always false);
            // retire it when smt.rs is next open.
            return smt.discharge_bv(candidates, obligation, &widths, Some(&signed), false);
        }
        smt.discharge(candidates, obligation)
    }

    /// Verify a function-level `@hint(assertion)` against the loop
    /// candidates — the hint is a user assertion, not decoration (same
    /// entailment machinery as the declared invariant).
    fn verify_loop_hint(
        &mut self,
        hint: &crate::ast::Expr<'input>,
        candidates: &[&'input crate::ast::Expr<'input>],
        vars: &[Symbol],
        span: crate::ast::Span,
        use_bv: bool,
    ) {
        if !self.obligation_provable(hint, candidates, vars, use_bv) {
            self.diagnostics.push(
                crate::diagnostics::Diagnostic::error(
                    "declared @hint(assertion) could not be verified",
                )
                .with_span(span)
                .with_label(
                    span,
                    format!("hint `{:?}` is not provable from the loop body", hint),
                ),
            );
        }
    }

    /// The declared loop invariant could not be proven — the clause is
    /// an obligation, so reject (fail-closed).
    fn reject_unverified_invariant(
        &mut self,
        inv: &crate::ast::Expr<'input>,
        span: crate::ast::Span,
    ) {
        self.diagnostics.push(
            crate::diagnostics::Diagnostic::error("declared loop invariant could not be verified")
                .with_span(span)
                .with_label(
                    span,
                    format!("invariant `{:?}` is not provable from the loop body", inv),
                ),
        );
    }

    /// Verify a DECLARED loop `decreases` measure — it must strictly
    /// decrease on every iteration.  Query:
    /// `∀X,X'. (∧candidates(X) ∧ G(X) ∧ T(X,X')) → dec(X') < dec(X)`; the
    /// SMT text asserts the NEGATION and expects `unsat`.  Unprovable /
    /// unavailable / unknown → reject (fail-closed, per the
    /// language-owner ruling).
    #[allow(clippy::too_many_arguments)] // full verification context — grouped by the caller
    fn verify_loop_decreases(
        &mut self,
        dec: &crate::ast::Expr<'input>,
        candidates: &[&'input crate::ast::Expr<'input>],
        vars: &[Symbol],
        body: &[crate::hir::loop_infer::LoopInstr],
        span: crate::ast::Span,
        use_bv: bool,
    ) {
        let Some(arena) = self.ctx.arena else {
            self.reject_unverified_decreases(dec, span);
            return;
        };
        // Primed copy of the decreases expr: `i` → `i_p`, so the query can
        // compare `dec(X') < dec(X)`.
        let mut dec_p = dec;
        for sym in vars {
            let primed = Symbol::intern(&format!("{}_p", sym.as_str()));
            dec_p = crate::ast::visit::replace_ident_in_expr(arena, dec_p, *sym, primed);
        }
        // Synthesis/verification agreement (same routing as
        // `obligation_provable`): a wrap-around loop (`use_bv`) proves
        // `dec(X') < dec(X)` under BIT-VECTOR semantics (modular
        // `bvadd`/`bvsub`, signed `bvslt`) — LIA's unbounded integers
        // cannot model the wrap that the loop actually executes.
        if use_bv {
            // Per-variable bit-widths, index-aligned with `vars` (each
            // `Int<N>`/`UInt<N>` variable declares at its OWN width; a
            // variable of unknown width defaults to 64 — the checker's
            // default).  A uniform width (the first variable's) would
            // truncate mixed-width loops and misjudge their guards.
            let widths: Vec<u8> = vars
                .iter()
                .map(|sym| {
                    self.local_variable_types
                        .get(*sym)
                        .and_then(|ty| self.ctx.bits_of_int(ty))
                        .map(|b| u8::try_from(b).unwrap_or(u8::MAX))
                        .unwrap_or(64)
                })
                .collect();
            // `widths_map` carries BOTH `i` and its primed copy `i_p`:
            // the decreases expression is re-encoded with
            // `replace_ident_in_expr` (i → i_p), so a compound measure
            // like `10 - i` becomes `10 - i_p` — WITHOUT the primed
            // entry the literal `10` stays 64-bit (the checker default)
            // while `i_p` declares at its own width (e.g. 8), Z3 rejects
            // the sort mismatch, and every compound `decreases` on a
            // non-64-bit loop fails closed.  The primed entry mirrors
            // the declared `_p` constant (same width).
            let widths_map: std::collections::HashMap<Symbol, u8> = vars
                .iter()
                .copied()
                .zip(widths.iter().copied())
                .flat_map(|(sym, w)| {
                    [
                        (sym, w),
                        (Symbol::intern(&format!("{}_p", sym.as_str())), w),
                    ]
                })
                .collect();
            // Signedness set: `Int<N>` variables compare with `bvslt`.
            // The primed copies (`i_p`) take the same signedness — they
            // appear as sub-expressions of the re-encoded decreases
            // measure and must compare consistently with the unprimed
            // side.
            let signed: std::collections::HashSet<String> = vars
                .iter()
                .filter_map(|sym| {
                    let ty = self.local_variable_types.get(*sym)?;
                    if matches!(self.ctx.get(ty), crate::hir::types::TypeData::Int { .. }) {
                        Some(sym.as_str())
                    } else {
                        None
                    }
                })
                .flat_map(|s| [s.to_string(), format!("{}_p", s)])
                .collect();
            let mut smt = String::new();
            smt.push_str("(set-logic BV)\n");
            for (idx, sym) in vars.iter().enumerate() {
                smt.push_str(&format!(
                    "(declare-const {} (_ BitVec {}))\n",
                    sym.as_str(),
                    widths[idx]
                ));
                smt.push_str(&format!(
                    "(declare-const {}_p (_ BitVec {}))\n",
                    sym.as_str(),
                    widths[idx]
                ));
            }
            // Declare the SSA intermediate variables `xs_{step}_{i}` used
            // by the sequential transition BEFORE any assert references
            // them — the previous code asserted them without declaring,
            // so Z3 rejected the query ("unknown constant xs_0_0") and
            // every wrap-loop `decreases` verification failed closed.
            let mut decl_step = 0usize;
            for instr in body {
                match instr {
                    crate::hir::loop_infer::LoopInstr::AddVar(i, _)
                    | crate::hir::loop_infer::LoopInstr::ConstVar(i, _)
                    | crate::hir::loop_infer::LoopInstr::CopyVar(i, _) => {
                        smt.push_str(&format!(
                            "(declare-const xs_{}_{} (_ BitVec {}))\n",
                            decl_step, i, widths[*i]
                        ));
                        decl_step += 1;
                    }
                    _ => {}
                }
            }
            // Candidates encode at the VARIABLES' signedness
            // (per-expression via `expr_to_smt_bv`: an expression over
            // `Int<N>` variables compares bvsge/bvsle, over `UInt<N>`
            // bvuge/bvule). The old `rows_unsigned` policy — BII rows
            // forced unsigned — predates the signed-row refactor, since
            // when signed rows carry TRUE signed bounds validated with
            // bvsle (`row_ge_le` / `template_formula`). A signed interval
            // like [-127, 0] forced-unsigned becomes
            // bvuge(x, 129) ∧ bvule(x, 0) — an EMPTY premise, and an
            // empty premise makes the negation query vacuously unsat
            // ("verified"): the false acceptance that flipped
            // test_set_in_loop_body_is_assignment once the A′-1 guard
            // fix stopped the BV synthesis from collapsing the BII to
            // the init singleton (the swap loop's false `decreases i` —
            // i goes -127 → 0 — started passing). The old
            // sign-boundary-crossing concern ([127, 129] rows) is OBE:
            // a signed Interval row's tops are exactly
            // [-2^(bw-1), 2^(bw-1)-1], so it never crosses, and rows
            // over unsigned variables pick unsigned comparators on
            // their own. Diff/Sum rows with |bound| ≥ 2^(bw-1) still
            // fail closed (the diff in-bounds guard / the literal-width
            // rejection) — the conservative direction.
            let candidate_signed = Some(&signed);
            for c in candidates {
                let mut e = String::new();
                if !crate::hir::smt::expr_to_smt_bv(c, &mut e, &widths_map, candidate_signed) {
                    // Candidate involves a Diff row whose bound exceeds
                    // the variable's native bit-width (computed at the
                    // synthesis-side value-preserving lift width).  Such a bound is
                    // trivially true at the variable's width (the
                    // difference is always within range), so skipping
                    // the candidate is safe — it adds no useful
                    // constraint to the premise.
                    continue;
                }
                smt.push_str(&format!("(assert {})\n", e));
            }
            // Guard + transition (sequential SSA with intermediate vars).
            // Guards encode via the value-preserving lift (same source of truth as the
            // synthesis side in bii.rs): each operand is lifted to a
            // signed-faithful representation, sign-extended to a common
            // width, and compared with SIGNED predicates.  This replaces
            // the old direct bvsle/bvule encoding which used different
            // widths for synthesis vs. verification when operands were
            // extended to different bit-widths.
            // cond_to_smt / encode_cmp_bv emit positional names (x_0,
            // x_1, …) for variables; remap them to the actual names
            // declared in this query (i, j, …).  Sort by descending
            // index so longer names (x_10) are replaced before shorter
            // ones (x_1) to avoid partial-match corruption.
            let remap_vars = |s: &str| -> String {
                let mut out = s.to_string();
                for idx in (0..vars.len()).rev() {
                    let from = format!("x_{idx}");
                    let to = vars[idx].as_str();
                    out = out.replace(&from, &to);
                }
                out
            };
            let signed_vec: Vec<bool> = vars
                .iter()
                .map(|sym| signed.contains(&sym.as_str()))
                .collect();
            let mut cur: Vec<String> = (0..vars.len())
                .map(|i| format!("{}", vars[i].as_str()))
                .collect();
            let mut step = 0usize;
            for instr in body {
                match instr {
                    crate::hir::loop_infer::LoopInstr::TestLe(i, c) => {
                        let cond = crate::hir::loop_ir::Cond::Cmp {
                            op: crate::hir::loop_ir::CmpOp::Le,
                            lhs: Box::new(crate::hir::loop_ir::ScalarExpr::Var(*i)),
                            rhs: Box::new(crate::hir::loop_ir::ScalarExpr::Const(BigInt::from(*c))),
                            signed: signed_vec[*i],
                        };
                        match crate::hir::bii::cond_to_smt(
                            &cond,
                            true,
                            &widths,
                            &signed_vec,
                            vars.len(),
                        ) {
                            Some(g) => smt.push_str(&format!("(assert {})\n", remap_vars(&g))),
                            None => {
                                self.reject_unverified_decreases(dec, span);
                                return;
                            }
                        }
                    }
                    crate::hir::loop_infer::LoopInstr::TestDiffLe(i, j, c) => {
                        let cond = crate::hir::loop_ir::Cond::Cmp {
                            op: crate::hir::loop_ir::CmpOp::Le,
                            lhs: Box::new(crate::hir::loop_ir::ScalarExpr::Sub(
                                Box::new(crate::hir::loop_ir::ScalarExpr::Var(*i)),
                                Box::new(crate::hir::loop_ir::ScalarExpr::Var(*j)),
                                crate::hir::loop_ir::ArithSem::Wrap,
                            )),
                            rhs: Box::new(crate::hir::loop_ir::ScalarExpr::Const(BigInt::from(*c))),
                            signed: signed_vec[*i] || signed_vec[*j],
                        };
                        match crate::hir::bii::cond_to_smt(
                            &cond,
                            true,
                            &widths,
                            &signed_vec,
                            vars.len(),
                        ) {
                            Some(g) => smt.push_str(&format!("(assert {})\n", remap_vars(&g))),
                            None => {
                                self.reject_unverified_decreases(dec, span);
                                return;
                            }
                        }
                    }
                    _ => {}
                }
            }
            for instr in body {
                match instr {
                    crate::hir::loop_infer::LoopInstr::AddVar(i, c) => {
                        let name = format!("xs_{}_{}", step, i);
                        step += 1;
                        smt.push_str(&format!(
                            "(assert (= {} (bvadd {} {})))\n",
                            name,
                            cur[*i],
                            crate::hir::smt::bv_const_pub(*c, widths[*i])
                        ));
                        cur[*i] = name;
                    }
                    crate::hir::loop_infer::LoopInstr::ConstVar(i, c) => {
                        let name = format!("xs_{}_{}", step, i);
                        step += 1;
                        smt.push_str(&format!(
                            "(assert (= {} {}))\n",
                            name,
                            crate::hir::smt::bv_const_pub(*c, widths[*i])
                        ));
                        cur[*i] = name;
                    }
                    crate::hir::loop_infer::LoopInstr::CopyVar(i, j) => {
                        let name = format!("xs_{}_{}", step, i);
                        step += 1;
                        smt.push_str(&format!("(assert (= {} {}))\n", name, cur[*j]));
                        cur[*i] = name;
                    }
                    _ => {}
                }
            }
            for (i, cur_name) in cur.iter().enumerate() {
                smt.push_str(&format!(
                    "(assert (= {}_p {}))\n",
                    vars[i].as_str(),
                    cur_name
                ));
            }
            let mut dec_cur = String::new();
            let mut dec_next = String::new();
            if !crate::hir::smt::expr_to_smt_bv(dec, &mut dec_cur, &widths_map, Some(&signed))
                || !crate::hir::smt::expr_to_smt_bv(
                    dec_p,
                    &mut dec_next,
                    &widths_map,
                    Some(&signed),
                )
            {
                self.reject_unverified_decreases(dec, span);
                return;
            }
            // The decrease comparison itself is signedness-aware: a
            // decreases expression over `Int<N>` variables compares signed
            // (`bvslt`); over `UInt<N>` variables, unsigned (`bvult`).
            let dec_cmp = if crate::hir::smt::expr_involves_signed(dec, &signed) {
                "bvslt"
            } else {
                "bvult"
            };
            smt.push_str(&format!(
                "(assert (not ({} {} {})))\n",
                dec_cmp, dec_next, dec_cur
            ));
            smt.push_str("(check-sat)\n");
            let smt_solver = self
                .smt_solver
                .get_or_insert_with(|| crate::hir::smt::SmtSolver::new("z3"));
            match smt_solver.run_raw_query(&smt) {
                crate::hir::smt::RawQueryOutcome::Unsat => { /* strictly decreasing — verified */
                }
                crate::hir::smt::RawQueryOutcome::Sat(m) => {
                    eprintln!(
                        "[verify_loop_decreases BV] SAT witness found — \
                         decreases `{:?}` unprovable: {}",
                        dec,
                        &m[..m.len().min(512)]
                    );
                    self.reject_unverified_decreases(dec, span);
                }
                crate::hir::smt::RawQueryOutcome::Unknown => {
                    eprintln!(
                        "[verify_loop_decreases BV] z3 returned unknown — \
                         decreases `{:?}` inconclusive (fail closed)",
                        dec
                    );
                    self.reject_unverified_decreases(dec, span);
                }
                crate::hir::smt::RawQueryOutcome::Error(e) => {
                    eprintln!("[verify_loop_decreases BV] SMT error: {}", e);
                    self.reject_unverified_decreases(dec, span);
                }
            }
            return;
        }
        let mut smt = String::new();
        smt.push_str("(set-logic LIA)\n");
        for sym in vars {
            smt.push_str(&format!("(declare-const {} Int)\n", sym.as_str()));
            smt.push_str(&format!("(declare-const {}_p Int)\n", sym.as_str()));
        }
        // The inferred invariant candidates as the premise.
        for c in candidates {
            let mut e = String::new();
            if !crate::hir::smt::expr_to_smt(c, &mut e) {
                self.reject_unverified_decreases(dec, span);
                return;
            }
            smt.push_str(&format!("(assert {})\n", e));
        }
        // Guard + transition from the body instructions.
        let mut assigned = vec![false; vars.len()];
        for instr in body {
            match instr {
                crate::hir::loop_infer::LoopInstr::TestLe(i, c) => {
                    smt.push_str(&format!("(assert (<= {} {}))\n", vars[*i].as_str(), c));
                }
                crate::hir::loop_infer::LoopInstr::TestDiffLe(i, j, c) => {
                    smt.push_str(&format!(
                        "(assert (<= (- {} {}) {}))\n",
                        vars[*i].as_str(),
                        vars[*j].as_str(),
                        c
                    ));
                }
                _ => {}
            }
        }
        // Guard + transition from the body instructions.  `cur[i]` tracks
        // the CURRENT value name of variable `i` (its `_p` primed name once
        // assigned) so a later `CopyVar` reads the post-update value —
        // sequential read-after-write semantics, matching the BV path and
        // `encode_sequential_transition` (bii.rs).
        let mut cur: Vec<String> = vars.iter().map(|s| s.as_str()).collect();
        for instr in body {
            match instr {
                crate::hir::loop_infer::LoopInstr::AddVar(i, c) => {
                    smt.push_str(&format!(
                        "(assert (= {}_p (+ {} {})))\n",
                        vars[*i].as_str(),
                        cur[*i],
                        c
                    ));
                    cur[*i] = format!("{}_p", vars[*i].as_str());
                    assigned[*i] = true;
                }
                crate::hir::loop_infer::LoopInstr::ConstVar(i, c) => {
                    smt.push_str(&format!("(assert (= {}_p {}))\n", vars[*i].as_str(), c));
                    cur[*i] = format!("{}_p", vars[*i].as_str());
                    assigned[*i] = true;
                }
                crate::hir::loop_infer::LoopInstr::CopyVar(i, j) => {
                    // Sequential read-after-write: `t := t + 1; a := t`
                    // gives `a_p` the CURRENT (post-update) `t`, not the
                    // pre-state value — the old code read `vars[*j]`
                    // directly, mis-modeling the transition when `j` was
                    // assigned earlier in the body.
                    smt.push_str(&format!(
                        "(assert (= {}_p {}))\n",
                        vars[*i].as_str(),
                        cur[*j]
                    ));
                    cur[*i] = format!("{}_p", vars[*i].as_str());
                    assigned[*i] = true;
                }
                _ => {}
            }
        }
        // Unassigned variables keep their value across the transition.
        for (i, a) in assigned.iter().enumerate() {
            if !a {
                smt.push_str(&format!(
                    "(assert (= {}_p {}))\n",
                    vars[i].as_str(),
                    vars[i].as_str()
                ));
            }
        }
        // The negation of `dec(X') < dec(X)` — expect `unsat`.
        let mut dec_cur = String::new();
        let mut dec_next = String::new();
        if !crate::hir::smt::expr_to_smt(dec, &mut dec_cur)
            || !crate::hir::smt::expr_to_smt(dec_p, &mut dec_next)
        {
            self.reject_unverified_decreases(dec, span);
            return;
        }
        smt.push_str(&format!("(assert (not (< {} {})))\n", dec_next, dec_cur));
        smt.push_str("(check-sat)\n");
        let smt_solver = self
            .smt_solver
            .get_or_insert_with(|| crate::hir::smt::SmtSolver::new("z3"));
        match smt_solver.run_raw_query(&smt) {
            crate::hir::smt::RawQueryOutcome::Unsat => { /* strictly decreasing — verified */ }
            crate::hir::smt::RawQueryOutcome::Sat(m) => {
                eprintln!(
                    "[verify_loop_decreases LIA] SAT witness found — \
                     decreases `{:?}` unprovable: {}",
                    dec,
                    &m[..m.len().min(512)]
                );
                self.reject_unverified_decreases(dec, span);
            }
            crate::hir::smt::RawQueryOutcome::Unknown => {
                eprintln!(
                    "[verify_loop_decreases LIA] z3 returned unknown — \
                     decreases `{:?}` inconclusive (fail closed)",
                    dec
                );
                self.reject_unverified_decreases(dec, span);
            }
            crate::hir::smt::RawQueryOutcome::Error(e) => {
                eprintln!("[verify_loop_decreases LIA] SMT error: {}", e);
                self.reject_unverified_decreases(dec, span);
            }
        }
    }

    /// The declared loop `decreases` measure could not be proven to
    /// strictly decrease — an obligation, so reject (fail-closed).
    fn reject_unverified_decreases(
        &mut self,
        dec: &crate::ast::Expr<'input>,
        span: crate::ast::Span,
    ) {
        self.diagnostics.push(
            crate::diagnostics::Diagnostic::error("declared loop decreases could not be verified")
                .with_span(span)
                .with_label(
                    span,
                    format!(
                        "decreases `{:?}` is not provably strictly decreasing per iteration",
                        dec
                    ),
                ),
        );
    }

    fn check_stmt(&mut self, stmt: &Stmt<'input>) -> Result<HirStmt<'input>, Diagnostic> {
        match stmt {
            Stmt::VariableDef { .. } => self.check_variable_def_stmt(stmt),
            Stmt::FunctionDef { .. } => self.check_function_def_stmt(stmt),
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
                    || matches!(self.ctx.get(cond_ty), TypeData::InferVar { id, .. }
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
                label,
                cond,
                body,
                invariant,
                decreases,
                span,
            } => {
                let (cond_hir, cond_ty) = self.infer_expr(cond, None)?;
                let cond_is_bool = self.ctx.is_bool(cond_ty)
                    || matches!(self.ctx.get(cond_ty), TypeData::InferVar { id, .. }
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
                self.push_ctx(
                    CtxKind::While,
                    *span,
                    label.as_ref().map(|l| l.as_str().to_string()),
                );
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
                self.infer_loop_hints(
                    &cond_hir,
                    &body_hir,
                    *span,
                    invariant.as_ref(),
                    decreases.as_ref(),
                );
                Ok(HirStmt::While {
                    label: *label,
                    cond: Box::new(cond_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::WhileLet {
                label,
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
                        ck.push_ctx(
                            CtxKind::While,
                            *span,
                            label.as_ref().map(|l| l.as_str().to_string()),
                        );
                        let body_hir = ck.check_block(body)?;
                        ck.pop_ctx();
                        Ok((inv_hir, dec_hir, body_hir))
                    })?;
                Ok(HirStmt::WhileLet {
                    label: *label,
                    pattern: pattern_hir,
                    scrutinee: Box::new(scrut_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::For {
                label,
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
                self.push_ctx(
                    CtxKind::For,
                    *span,
                    label.as_ref().map(|l| l.as_str().to_string()),
                );
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
                // scope drops here — removes pattern + block bindings
                Ok(HirStmt::For {
                    label: *label,
                    pattern: pattern_hir,
                    iterable: Box::new(iter_hir),
                    body: body_hir,
                    invariant: inv_hir.map(Box::new),
                    decreases: dec_hir.map(Box::new),
                    span: *span,
                })
            }
            Stmt::Loop { label, body, span } => {
                self.push_ctx(
                    CtxKind::Loop,
                    *span,
                    label.as_ref().map(|l| l.as_str().to_string()),
                );
                let body_hir = self.check_block(body)?;
                self.pop_ctx();
                Ok(HirStmt::Loop {
                    label: *label,
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
                        } else if label.is_some() {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "cannot `continue` with label `{}` – no matching labeled loop found",
                                    label.as_ref().unwrap()
                                ))
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
            Stmt::Return { .. } => self.check_return_stmt(stmt),
            Stmt::Assign { .. } => self.check_assign_stmt(stmt),
            Stmt::ComptimeBlock { .. } => self.check_comptime_block_stmt(stmt),
            Stmt::ScopeCleanup {
                name,
                body,
                propagates,
                overrides,
                when_condition,
                span,
            } => {
                let body_hir = self.check_block(body)?;
                // Convert when_condition from AST Expr<'input> to HirExpr<'input> and
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
                // Strict Mode forbids `unsafe` blocks completely
                // (SYNTAX.md §Strict Mode).
                if self.strict_mode {
                    self.diagnostics.push(
                        Diagnostic::error("`unsafe` blocks are forbidden in Strict Mode")
                            .with_span(*span),
                    );
                }
                let body_hir = self.check_block(body)?;
                Ok(HirStmt::Unsafe {
                    body: body_hir,
                    span: *span,
                })
            }
            Stmt::GhostVariableDef { inner, span } => {
                // Extract the variable name from the inner `set mut? name = ...`.
                let var_name = match inner {
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
                // A fresh per-frame set of variables declared
                // INSIDE this isolate block; the Assign arm rejects writes
                // to variables NOT in this set (captured outer mutable
                // locals / parameters — external mutable state).
                self.isolate_declared.push(HashSet::new());
                let body_hir = match self.check_block(body) {
                    Ok(hir) => {
                        self.pop_ctx();
                        self.isolate_declared.pop();
                        Ok(HirStmt::Isolate {
                            body: hir,
                            span: *span,
                        })
                    }
                    Err(diag) => {
                        self.pop_ctx();
                        self.isolate_declared.pop();
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
            Stmt::ImplBlock { .. } => self.check_impl_block_stmt(stmt),
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

    fn check_return_stmt(&mut self, stmt: &Stmt<'input>) -> Result<HirStmt<'input>, Diagnostic> {
        let Stmt::Return {
            value,
            labels,
            span,
        } = stmt
        else {
            unreachable!("check_stmt dispatch guarantees the statement variant");
        };
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

    fn check_assign_stmt(&mut self, stmt: &Stmt<'input>) -> Result<HirStmt<'input>, Diagnostic> {
        let Stmt::Assign {
            target,
            op,
            value,
            span,
        } = stmt
        else {
            unreachable!("check_stmt dispatch guarantees the statement variant");
        };
        // Mutation of a frozen place is enforced by the flow-sensitive
        // point-level borrow-check post-pass (see the FunctionDef<'input>
        // arm) — loans die at their borrow variable's last use.
        // Validate that the target is a valid lvalue
        if !is_valid_lvalue(target) {
            self.diagnostics.push(
                Diagnostic::error("invalid left-hand side for assignment; expected variable, field access, or index")
                    .with_span(*span)
            );
        }
        // Check that mutable globals are only assigned inside @trusted functions
        if let Expr::Ident(name, _) = target {
            if self.mutable_globals.contains(name) && !self.current_function_trusted {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot assign to mutable global `{}` outside `@trusted` function",
                        name,
                    ))
                    .with_code_str("E040")
                    .with_span(*span)
                    .with_help(
                        "wrap the function in `@trusted` and add `requires`/`ensures` contracts",
                    ),
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
                    .with_help("comptime code is sandboxed and cannot access mutable global state"),
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
            // Assigning to a CAPTURED outer mutable
            // local (or a parameter) inside an isolate block
            // mutates external mutable state — SYNTAX.md §Task
            // Isolation ("does not access any external mutable
            // state").  Variables declared INSIDE the block
            // (`isolate_declared` — recorded by the VariableDef
            // arm) are internal and may be mutated; anything else
            // is captured and must be frozen.
            if self.is_in_isolate() {
                let internal = self
                    .isolate_declared
                    .last()
                    .map_or(false, |frame| frame.contains(name));
                if !internal {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "cannot assign to captured variable `{}` inside isolate block",
                            name,
                        ))
                        .with_code_str("E093")
                        .with_span(*span)
                        .with_help(
                            "isolate blocks must not mutate captured outer state — \
                             declare the variable inside the block instead",
                        ),
                    );
                }
            }
        }
        // An assign TARGET is written, not read — the
        // mutation check below handles the write side (the old
        // read-side suppression is removed ).
        let target_res = self.infer_expr(target, None);
        let (target_hir, target_ty) = target_res?;
        let value_hir = if let Some(op) = op {
            let result_ty = self.binary_op_type(*op, target_ty, target_ty, None, None, *span)?;
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

    fn check_comptime_block_stmt(
        &mut self,
        stmt: &Stmt<'input>,
    ) -> Result<HirStmt<'input>, Diagnostic> {
        let Stmt::ComptimeBlock {
            captures,
            body,
            trusted,
            span,
            ..
        } = stmt
        else {
            unreachable!("check_stmt dispatch guarantees the statement variant");
        };
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
                    self.deferred_comptime_blocks
                        .push((captures.clone(), hir.clone(), ty, *span));
                } else {
                    // Evaluate the comptime block at compile time.
                    // Pre-collect literal values for captures before any mutable borrow of self.
                    let captured_literals: Vec<(Symbol, Option<ComptimeValue>)> = captures
                        .iter()
                        .map(|(sym, _span)| {
                            (
                                *sym,
                                self.literal_values.get(sym).and_then(|v| v.last()).cloned(),
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
                    let mut eval = crate::hir::comptime::ComptimeEvalContext::new_with_source(
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

    fn check_variable_def_stmt(
        &mut self,
        stmt: &Stmt<'input>,
    ) -> Result<HirStmt<'input>, Diagnostic> {
        let Stmt::VariableDef {
            kind,
            mutable,
            name,
            pattern,
            ty,
            value,
            else_branch,
            span,
            type_captures,
            type_modifiers,
            ..
        } = stmt
        else {
            unreachable!("check_stmt dispatch guarantees the statement variant");
        };
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

        // ── Loop-body `set` → assignment lowering ──
        // Inside a loop body, `set x = v` on a loop-carried variable that
        // already exists (e.g. `set i = i + 1`) is the harness form of the
        // plain assignment `i = i + 1` — lower it to the assignment path
        // so the value is checked against the existing binding and the
        // mutation rules of the plain syntax apply.  Without this
        // lowering, a second `set` of a loop-carried variable in the same
        // body reported E019 (duplicate definition) and the W113
        // shadowing diverged from assignment semantics.  Only the
        // innermost frame matters: `set` of a variable declared in an
        // OUTER frame outside the loop still warns (W113), because the
        // outer scope is not a loop-carried update.
        let in_loop_body = self
            .region_tree
            .iter_frames_rev()
            .next()
            .is_some_and(|f| matches!(f.kind, CtxKind::While | CtxKind::For | CtxKind::Loop));
        if let Some(arena) = self.ctx.arena
            && *kind == VariableKind::Set
            && in_loop_body
            && let Some(var_name) = name
            && let Some(value_expr) = value
            && self.local_variable_types.get(*var_name).is_some()
        {
            let assign = Stmt::Assign {
                target: arena.alloc(crate::ast::Expr::Ident(*var_name, *span)),
                op: None,
                value: (*value_expr).clone(),
                span: *span,
            };
            return self.check_assign_stmt(&assign);
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
        // Type-level overflow policy: apply the
        // `with overflow = ...` modifiers parsed on the annotated type.
        let declared_ty = self.apply_overflow_modifiers(declared_ty, type_modifiers);

        // Determine the actual initializer (value) and its type.
        // Wrap in a closure so errors from the RHS can be aggregated
        // into the duplicate definition diagnostic.
        let rhs_result = (|| -> Result<(Option<HirExpr<'input>>, TypeId, Option<HirPattern<'input>>, Option<Vec<HirStmt<'input>>>), Diagnostic> {
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
                // No explicit initializer: try type's default value.
                // The default may live on the ALIAS binding (`type
                // MyInt = Int<32> with default = 5` — SYNTAX.md §535)
                // — the ADT-only lookup below cannot see it because
                // the alias resolves THROUGH to its base type.  Look
                // up the annotation path's binding directly first.
                let default_expr = if let Some(crate::ast::Type::Path(p, _)) = ty {
                    match self
                        .symbols
                        .lookup_type_by_path(p)
                        .and_then(|d| self.symbols.lookup_type_by_def_id(d))
                    {
                        Some(b) if b.no_default => {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "type forbids implicit initialization (no_default)",
                                )
                                .with_span(*span),
                            );
                            None
                        }
                        Some(b) => b.default_value.clone(),
                        None => self.lookup_type_default_expr(declared_ty, *span)?,
                    }
                } else {
                    self.lookup_type_default_expr(declared_ty, *span)?
                };
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
                        // The semantic code for "no
                        // default value" is E016 — E003 is
                        // "unexpected token" (parser).
                        .with_code_str("E016")
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
                // The let-else branch — the closure pattern keeps
                // the `?` early-return semantics (the old loan
                // truncation is removed — `ctx.loans` is
                // dead bookkeeping).
                let res = (|| {
                    let mut stmts = Vec::new();
                    for s in else_branch {
                        stmts.push(self.check_stmt(s)?);
                    }
                    Ok(stmts)
                })();
                Some(res?)
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
                // A variable declared INSIDE an
                // isolate block is internal state — record it in
                // the current isolate frame so the Assign arm
                // allows mutating it (only CAPTURED outer
                // variables are external).
                if let Some(frame) = self.isolate_declared.last_mut() {
                    frame.insert(*var_name);
                }
            }
            // Track comptime-known literal values for explicit captures
            // AND for the loop-invariant hint pipeline (the checker
            // seeds `infer_loop_invariant_exprs` with the pre-loop
            // values of loop variables — which are typically `mut`
            // counters — via `get_literal_value`).
            if let Some(ref value_hir) = value_hir {
                if let HirExpr::Literal(lit, _, _) = value_hir {
                    let cv = match lit {
                        Literal::Int(v) => v.to_i128().map(ComptimeValue::Int),
                        Literal::Float(v) => Some(ComptimeValue::Float(*v)),
                        Literal::Bool(v) => Some(ComptimeValue::Bool(*v)),
                        Literal::String(s) => {
                            Some(ComptimeValue::String(std::sync::Arc::from(s.as_str())))
                        }
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
                            for (fn_name, (fn_params, fn_body)) in &self.comptime_fn_registry {
                                ec.register_fn(*fn_name, fn_params.clone(), fn_body.clone());
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
            let is_must_handle = match callee {
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
                                    && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            })
                            || self
                                .trait_env
                                .lookup_impls_for_type(base_ty)
                                .iter()
                                .flat_map(|ic| &ic.methods)
                                .any(|m| {
                                    m.name == *field
                                        && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
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

    fn check_impl_block_stmt(
        &mut self,
        stmt: &Stmt<'input>,
    ) -> Result<HirStmt<'input>, Diagnostic> {
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
            let trait_id = match tp {
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
                    let trait_ty = self.resolve_type(tp)?;
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
                    let trait_name = match tp {
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
            let impl_method_names: HashSet<Symbol> = methods.iter().map(|m| m.name).collect();
            let self_ty = &for_type;

            for (tm_name, _tm_sig) in &trait_binding.methods {
                if !impl_method_names.contains(tm_name) {
                    self.diagnostics.push(
                        Diagnostic::error_kind(DiagnosticKind::ImplMissingMethod {
                            trait_name: Self::type_to_string(tp),
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
                let mut param_tys = Vec::new();
                for p in &m.params {
                    let ty = if let Some(ty) = &p.ty {
                        let resolved = self.resolve_self_ty(ty, self_ty);
                        self.resolve_type(&resolved)?
                    } else {
                        // Bare `self`, `&self`, `&mut self` params: resolve to `for_ty`
                        for_ty
                    };
                    param_tys.push(ty);
                }
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
                    // The method's OWN DefId — allocated by the
                    // resolver and registered under (receiver DefId,
                    // name); look it back up so the checker and the
                    // AST pre-scan share the same method identity.
                    // Fallback: allocate fresh (defensive — should
                    // never fire, resolver runs first).
                    def_id: self
                        .ctx
                        .get_def_id_for_type(for_ty)
                        .and_then(|receiver_def| {
                            self.symbols.lookup_method_def_id(receiver_def, m.name)
                        })
                        .unwrap_or_else(crate::hir::types::alloc_def_id),
                    name: m.name,
                    param_tys,
                    ret_ty,
                    span: m.span,
                    attributes: m.attributes.clone(),
                    has_auto_deref: auto_deref,
                });
            }

            // Populate the associated types (`type Target = ...`)
            // so deref coercions (`try_deref_trait_step`) can find
            // the `Target` through the impl.
            let mut assoc_tys = Vec::new();
            for at in associated_types {
                let ty = match &at.default {
                    Some(d) => {
                        let resolved = self.resolve_self_ty(d, self_ty);
                        self.resolve_type(&resolved)?
                    }
                    None => self.ctx.error(),
                };
                assoc_tys.push((at.name, ty));
            }
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
                        && let Type::Generic(_, generic_args, _) = tp
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
                                                    && let Ok(resolved) = self.resolve_type(ty)
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

            if let Err(orphan) = self
                .trait_env
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
                // The new trait impl also invalidates
                // the method-lookup cache (a cached miss would mask
                // the newly available methods).
                self.method_cache.clear();
            }

            // Also register the resolved methods for method resolution
            if let TypeData::Adt { def_id, .. } = self.ctx.get(for_ty) {
                self.trait_env.add_inherent_methods(*def_id, method_infos);
            }

            // Type-check each method body AFTER registration (the
            // impl is visible via add_impl/add_inherent_methods now,
            // so a body may call a sibling method `self.b()`).
            for m in methods {
                self.check_method_body(
                    m,
                    &for_type,
                    &impl_param_names,
                    &const_param_names,
                    &const_param_value_types,
                )?;
            }

            // ── Clean up generic parameter cache for trait impl ──
            // AFTER the method bodies: `check_method_body` re-resolves
            // the signatures via `resolve_self_ty` (e.g. `&self` →
            // `&Box<T>`), which needs the impl's type params still in
            // the cache.  Clearing first made every generic trait impl
            // with methods fail with "type 'T' not found".
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
                        Diagnostic::error("inherent impl on non-struct/enum type").with_span(span),
                    );
                    return Ok(HirStmt::Error);
                }
            };
            // Resolve method param/return types, replacing `Self` with for_type
            let self_ty = &for_type; // The original AST type for Self
            let auto_deref = attributes.iter().any(|a| a.name.eq_str("auto_deref"));
            let mut method_infos = Vec::new();
            for m in methods {
                let mut param_tys = Vec::new();
                for p in &m.params {
                    let ty = if let Some(ty) = &p.ty {
                        let resolved = self.resolve_self_ty(ty, self_ty);
                        self.resolve_type(&resolved)?
                    } else {
                        // Bare `self`, `&self`, `&mut self` params: resolve to `for_ty`
                        for_ty
                    };
                    param_tys.push(ty);
                }
                let ret_ty = {
                    let resolved = self.resolve_self_ty(&m.return_type, self_ty);
                    self.resolve_type(&resolved)?
                };
                method_infos.push(crate::hir::traits::MethodInfo {
                    // The method's OWN DefId — allocated by the
                    // resolver and registered under (receiver DefId,
                    // name); look it back up so the checker and the
                    // AST pre-scan share the same method identity.
                    // Fallback: allocate fresh (defensive — should
                    // never fire, resolver runs first).
                    def_id: self
                        .symbols
                        .lookup_method_def_id(for_def_id, m.name)
                        .unwrap_or_else(crate::hir::types::alloc_def_id),
                    name: m.name,
                    param_tys,
                    ret_ty,
                    span: m.span,
                    attributes: m.attributes.clone(),
                    has_auto_deref: auto_deref,
                });
            }
            // Register ALL methods BEFORE checking any body: a
            // method body may call a sibling method (`self.b()`),
            // and method lookup reads `trait_env.inherent_methods`
            // — which must already contain the whole impl (rustc
            // registers `associated_items` before body typeck).
            self.trait_env
                .add_inherent_methods(for_def_id, method_infos);
            // The new methods invalidate the method-lookup cache.
            self.method_cache.clear();
            // Type-check each method body AFTER registration.
            for m in methods {
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

    /// `@no_panic` body verification (SYNTAX.md §Effect Annotations): the
    /// function must never panic.  Scan the HIR body for
    /// - default-trap floating-point arithmetic (the float default
    ///   overflow policy is `trap`, aligned with integers — the
    ///   2026-08-08 committee ruling);
    /// - explicit trap operators (`+!` / `-!` / `*!`);
    /// - `panic` calls;
    /// - calls to functions that are not themselves `@no_panic`.
    /// Violations are compile-time errors in strict mode, warnings
    /// otherwise (SYNTAX.md: "Verification failure is a compile-time
    /// error in strict mode; in non-strict mode, the compiler emits a
    /// warning").
    fn check_no_panic_body(&mut self, body: &[HirStmt<'input>], fn_span: Span) {
        let _ = fn_span;
        let mut stack: Vec<Node<'_, 'input>> = body.iter().map(Node::Stmt).collect();
        while let Some(node) = stack.pop() {
            match node {
                Node::Stmt(stmt) => match stmt {
                    HirStmt::Expression(e) => stack.push(Node::Expr(e)),
                    HirStmt::If {
                        then_branch,
                        else_branch,
                        ..
                    }
                    | HirStmt::IfLet {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        stack.extend(then_branch.iter().map(Node::Stmt));
                        if let Some(b) = else_branch {
                            stack.extend(b.iter().map(Node::Stmt));
                        }
                    }
                    HirStmt::While { body, .. }
                    | HirStmt::WhileLet { body, .. }
                    | HirStmt::For { body, .. }
                    | HirStmt::Loop { body, .. }
                    | HirStmt::ComptimeBlock { body, .. }
                    | HirStmt::ScopeCleanup { body, .. }
                    | HirStmt::Unsafe { body, .. }
                    | HirStmt::Isolate { body, .. }
                    | HirStmt::Generate { body, .. } => {
                        stack.extend(body.iter().map(Node::Stmt));
                    }
                    HirStmt::GhostVariableDef { inner, .. } => stack.push(Node::Stmt(inner)),
                    HirStmt::VariableDef { value, .. } => {
                        if let Some(e) = value {
                            stack.push(Node::Expr(e));
                        }
                    }
                    HirStmt::Assign { value, .. } => stack.push(Node::Expr(value)),
                    HirStmt::Return { value, .. } => {
                        if let Some(e) = value {
                            stack.push(Node::Expr(e));
                        }
                    }
                    _ => {}
                },
                Node::Expr(expr) => {
                    match expr {
                        HirExpr::BinaryOp {
                            op,
                            left,
                            right,
                            span,
                            ..
                        } => {
                            let explicit_trap =
                                matches!(op, BinOp::AddTrap | BinOp::SubTrap | BinOp::MulTrap);
                            // Default float arithmetic traps: the float
                            // default overflow policy is `trap` (aligned
                            // with integers — committee ruling), so a
                            // plain `+`/`-`/`*`/`/`/`%` on floats can
                            // panic (NaN/∞/div-by-zero).  `+%` (AddWrap)
                            // is IEEE — allowed; `+?` (saturate) —
                            // allowed; `+!` (trap) — caught above.
                            let default_float_trap = matches!(
                                op,
                                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
                            ) && (self.no_panic_operand_is_float(left)
                                || self.no_panic_operand_is_float(right));
                            if explicit_trap || default_float_trap {
                                self.push_no_panic_violation(
                                    if explicit_trap {
                                        "explicit trap operator (`+!`/`-!`/`*!`) in `@no_panic` function"
                                    } else {
                                        "default-trap floating-point arithmetic in `@no_panic` function (use `+%`/`+?` or `with overflow = wrap|saturate|ieee`)"
                                    },
                                    *span,
                                );
                            }
                        }
                        HirExpr::Call { callee, span, .. } => {
                            if let HirExpr::Ident(name, _, _) = callee.as_ref() {
                                if name.eq_str("panic") {
                                    self.push_no_panic_violation(
                                        "call to `panic` in `@no_panic` function",
                                        *span,
                                    );
                                } else if let Some(fb) = self.symbols.lookup_function(*name) {
                                    let callee_no_panic =
                                        fb.attributes.iter().any(|a| a.name.eq_str("no_panic"));
                                    if !callee_no_panic {
                                        self.push_no_panic_violation(
                                            &format!(
                                                "call to non-`@no_panic` function `{}` in `@no_panic` function",
                                                name,
                                            ),
                                            *span,
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    push_expr_children(&mut stack, expr);
                }
            }
        }
    }

    /// Emit a `@no_panic` verification violation: an error in strict
    /// mode, a warning otherwise (SYNTAX.md §Effect Annotations).
    fn push_no_panic_violation(&mut self, msg: &str, span: Span) {
        let diag = if self.strict_mode {
            Diagnostic::error(format!("`@no_panic` violation: {msg}"))
        } else {
            Diagnostic::warning(format!("`@no_panic` violation: {msg}"))
        };
        self.diagnostics.push(diag.with_span(span));
    }

    /// Whether `e` denotes a FLOAT operand for the `@no_panic` scan.
    ///
    /// A float literal's HIR type is a Float-kind inference variable
    /// (still unresolved at body-scan time), so `is_float(ty)` alone
    /// would miss `1.0 + 2.0`; recognise the literal directly and check
    /// the infer-var kind as well.
    fn no_panic_operand_is_float(&self, e: &HirExpr<'input>) -> bool {
        if matches!(e, HirExpr::Literal(crate::ast::Literal::Float(_), _, _)) {
            return true;
        }
        if self.ctx.is_float(e.ty()) {
            return true;
        }
        if let TypeData::InferVar { id, .. } = self.ctx.get(e.ty())
            && self.infer.get_var_kind(*id) == Some(TypeVariableKind::Float)
        {
            return true;
        }
        false
    }

    fn check_function_def_stmt(
        &mut self,
        stmt: &Stmt<'input>,
    ) -> Result<HirStmt<'input>, Diagnostic> {
        let Stmt::FunctionDef {
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
        } = stmt
        else {
            unreachable!("check_stmt dispatch guarantees the statement variant");
        };
        // ── Save per-function state for nested `def` support ──
        let prev_seal = self.ctx.seal_violations.get();
        let prev_return_ty = self.current_return_type;
        let prev_must_handle = self.must_handle_sources.borrow().clone();
        // The function's `@hint(assertion)` list — verified against
        // loop candidates during body checking (a user hint is an
        // assertion, not decoration).
        let prev_hints = std::mem::take(&mut self.pending_function_hints);
        self.pending_function_hints = attributes
            .iter()
            .filter(|a| a.name.eq_str("hint"))
            .flat_map(|a| a.args.clone())
            .collect();
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

        // ── Where-equality given constraints ────────────
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
                if let (Ok(l), Ok(r)) = (self.resolve_type(&eq.left), self.resolve_type(&eq.right))
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
        guard.checker.current_function_pure = attributes.iter().any(|a| a.name.eq_str("pure"));

        // ── Proof obligation check (strict mode) ────────────────
        // In strict mode, all @trusted functions must have @link_proof
        // or @comptime_test evidence.  This ensures that trust
        // boundaries are backed by formal proofs or test coverage.
        if guard.checker.current_function_trusted && guard.checker.strict_mode {
            let has_link_proof = attributes.iter().any(|a| a.name.eq_str("link_proof"));
            let has_comptime_test = attributes.iter().any(|a| a.name.eq_str("comptime_test"));
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

        // `@trusted` must carry `requires`/`ensures`
        // contracts (SYNTAX.md:1039 — "Requires `requires`/
        // `ensures` contracts"; :1202 — "it must carry
        // `requires`/`ensures` contracts").  The trust boundary is
        // DOCUMENTED by its contracts; previously only the
        // strict-mode @link_proof evidence was checked, so a
        // `@trusted` function with no contracts at all slipped
        // through.
        if guard.checker.current_function_trusted {
            let has_requires = contracts
                .iter()
                .any(|c| matches!(c, crate::ast::Contract::Requires(..)));
            let has_ensures = contracts
                .iter()
                .any(|c| matches!(c, crate::ast::Contract::Ensures { .. }));
            if !has_requires || !has_ensures {
                guard.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "@trusted function `{}` must carry `requires`/`ensures` contracts",
                        name,
                    ))
                    .with_code_str("E092")
                    .with_span(*span)
                    .with_help(
                        "add `requires <condition>` and `ensures <condition>` \
                                 clauses documenting the trust boundary",
                    ),
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

        // ── Explicit-lifetime region constraints ─────────────
        // (SYNTAX.md §Explicit Lifetime Parameters: "verified by
        // the borrow checker; mismatches cause compile errors".)
        // Collect the explicit `&'a T` regions of the parameter
        // and return types and SOLVE the outlives constraints: a
        // return reference may only use a lifetime PROVIDED BY a
        // parameter (the caller hands it in) — otherwise the
        // returned reference could outlive its source (rustc:
        // "lifetime ... does not appear in any of the function's
        // parameters" / E0623).  The solver computes the
        // transitive closure, so `'a: 'b` chains are honored.
        let param_regions: HashSet<Symbol> = hir_params
            .iter()
            .flat_map(|p| collect_ref_lifetimes(p.ty, &guard.checker.ctx))
            .collect();
        let ret_regions = collect_ref_lifetimes(return_ty, &guard.checker.ctx);
        // Retain the signature region sets for the POST-BODY
        // second solve (the body's collected subtype pairs are
        // merged into `region_outlives` at the end).
        guard.checker.current_param_regions = param_regions.clone();
        guard.checker.current_ret_regions = ret_regions.clone();
        // Enable SUBTYPE-path region collection for this function:
        // `subtype`'s Ref arm now records `&'a T <: &'b T` as the
        // covariance constraint `'a: 'b` (rustc `make_subregion`)
        // instead of rejecting — the solver consumes the pairs.
        guard.checker.ctx.region_subtype_collect.set(true);
        guard
            .checker
            .ctx
            .region_subtype_outlives
            .borrow_mut()
            .clear();
        // Register the (trivially-satisfied) outlives records for
        // every shared region before solving, PLUS the explicit
        // `where 'a: 'b` lifetime-outlives predicates (rustc's
        // `WherePredicateKind::RegionPredicate`) as solver input
        // edges — `'a: 'b` chains are honored by the transitive
        // closure.
        guard.checker.region_outlives.clear();
        for r in &ret_regions {
            guard.checker.region_outlives.push((*r, *r));
        }
        if let Some(wc) = where_clause.as_ref() {
            for (lt, outlives) in &wc.lifetime_outlives {
                for bound in outlives {
                    guard.checker.region_outlives.push((*lt, *bound));
                }
            }
        }
        for uncovered in
            solve_region_outlives(&param_regions, &ret_regions, &guard.checker.region_outlives)
        {
            guard.checker.diagnostics.push(
                Diagnostic::error(format!(
                    "lifetime `{}` does not appear in any of the function's parameters",
                    uncovered,
                ))
                .with_span(*span)
                .with_help(
                    "the returned reference's lifetime must be provided by a \
                             parameter — add `&'<lifetime> T` to a parameter type or \
                             remove the lifetime from the return type",
                ),
            );
        }

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
                        .with_suggestion(
                            "add `@no_alloc` to this function (redundant with `@no_panic`?)",
                        ),
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

                    let Some(constraint) = guard.checker.symbols.lookup_constraint(name) else {
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
        // solver so they can be used as Param<'input> candidates for matching
        // obligations on the same type.  Bounds on fully concrete types
        // (e.g., `where i32: SomeTrait`) must NOT be passed as assumptions
        // — the solver would treat them as Param<'input> candidates and succeed
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
        // guardrail (SYNTAX.md §Reference Coercion): the
        // implicit freeze must NOT apply inside `@trusted` functions
        // or in Strict Mode, even when `@auto_ro` is present.
        let has_auto_ro = attributes.iter().any(|a| a.name.eq_str("auto_ro"));
        let has_auto_coerce = attributes.iter().any(|a| a.name.eq_str("auto_coerce"));
        let has_trusted = attributes.iter().any(|a| a.name.eq_str("trusted"));
        let has_runtime_check = attributes.iter().any(|a| a.name.eq_str("runtime_check"));
        let auto_ro_active = has_auto_ro && !guard.checker.strict_mode && !has_trusted;
        guard.checker.ctx.auto_ro.set(auto_ro_active);
        let auto_coerce_active = has_auto_coerce && !guard.checker.strict_mode && !has_trusted;
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
                    Diagnostic::error("`@auto_ro` is not permitted on `@trusted` functions")
                        .with_span(*span),
                );
            }
        }
        // SYNTAX.md: `@runtime_check` — "Only allowed in non-strict
        // mode."  In Strict Mode it would defer contract checking
        // to runtime, undermining the static-proof requirement.
        if has_runtime_check && guard.checker.strict_mode {
            guard.checker.diagnostics.push(
                Diagnostic::error("`@runtime_check` is not permitted in Strict Mode")
                    .with_span(*span),
            );
        }
        if has_auto_coerce {
            if guard.checker.strict_mode {
                guard.checker.diagnostics.push(
                    Diagnostic::error("`@auto_coerce` is not permitted in Strict Mode")
                        .with_span(*span),
                );
            } else if has_trusted {
                guard.checker.diagnostics.push(
                    Diagnostic::error("`@auto_coerce` is not permitted on `@trusted` functions")
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
        // the outer function's `seal_violations` must survive the
        // nested function's body checking (the loans bookkeeping is
        // removed ).
        guard.checker.ctx.seal_violations.set(prev_seal);
        guard.checker.current_return_type = prev_return_ty;
        guard.checker.pending_function_hints = prev_hints;
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
        let mut body_hir: Option<Vec<HirStmt<'input>>> = None;
        let mut saved_body_err: Option<Diagnostic> = None;
        match body_result {
            Ok(body) => {
                body_hir = body;
            }
            Err(e) => {
                saved_body_err = Some(e);
            }
        }

        // ── @no_panic body verification ─────────────────────────
        // SYNTAX.md §Effect Annotations: a `@no_panic` function must
        // never panic.  Scan the body for default-trap float arithmetic
        // (float default overflow policy is `trap`), explicit trap
        // operators (`+!`), `panic` calls, and calls to non-`@no_panic`
        // functions.  Comptime functions execute at compile time — a
        // panic there surfaces directly as a compile error, so the
        // runtime no-panic guarantee is vacuous; skip them.
        if !*is_comptime
            && attributes.iter().any(|a| a.name.eq_str("no_panic"))
            && let Some(ref body_stmts) = body_hir
        {
            guard.checker.check_no_panic_body(body_stmts, *span);
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
            fn has_return_recursive<'input>(stmts: &[HirStmt<'input>]) -> bool {
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
        // The comptime functions are compile-time — they never
        // exist at runtime — so the RUNTIME post-passes (the A(ρ)
        // signature collection, the borrow check, the move check)
        // do not apply to their bodies.
        if !*is_comptime && let Some(ref body_stmts) = body_hir {
            // The borrow signature facts (the A(ρ) constraints)
            // from the resolved reference types + the output
            // derivation.
            guard.checker.collect_signature_facts(
                *name,
                &hir_params,
                Some(return_ty),
                body_stmts,
                false,
                None,
            );
            // Two-phase refactor: the flow-sensitive borrow
            // check + move check are NOT run here — they run in
            // Phase B (check_program) over ALL function bodies,
            // AFTER the signature registry is complete.  Running
            // them interleaved here made cross-function loan
            // facts order-dependent: a caller appearing before
            // its callee was checked against the callee's
            // AST-level (under-approximated) signature, and the
            // later HIR-level replacement never re-checked it.
            guard
                .checker
                .pending_runtime_checks
                .push((*name, body_stmts.clone()));
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
                fn collect_return_labels<'input>(stmts: &[HirStmt<'input>]) -> Vec<Symbol> {
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
        // themselves as Param<'input> candidates and cause ambiguity.
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
            let ctx: &mut TypeContext<'input> = guard.checker.ctx;
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
                        code: crate::hir::traits::solver::ObligationCauseCode::WhereClause {
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
                            crate::hir::traits::solver::Predicate::Forall { body: body.clone() }
                        }
                        TraitPredicate::Exists { body } => {
                            crate::hir::traits::solver::Predicate::Exists { body: body.clone() }
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
                // SAFETY: `ctx_ptr` points to the `TypeContext<'input>` owned by the
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

        // ── Return-body equality: eager check + constraint-queue dispatch ──
        // (moved BEFORE guard.commit so the queued Constraint::Eq is solved
        // by THIS function's inference scope — dispatching after commit
        // would leak the constraint into the outer inference context).
        //
        // Reachability dependency: the `saved_body_err` short-circuit above
        // returns BEFORE this block, so a body-scope error and a return
        // mismatch never surface in the same check — the eager `?` below
        // also returns before the queued Eq is dispatched, so the same
        // equality is never double-reported.  If `saved_body_err` handling
        // ever becomes non-short-circuiting (e.g. for better error
        // recovery), the double-error precedence question reopens with no
        // test pinning it — revisit the ordering here if that changes.
        if let Some(ref body_stmts) = body_hir
            && return_type.is_some()
        {
            // User wrote an explicit return type — check body against it.
            let body_ty = guard.checker.block_type_impl(body_stmts, false);
            // Eager unification remains the diagnostic authority (DCE S4):
            // a mismatch reports the rich E030 immediately.  On success the
            // SAME equality is ALSO dispatched to the constraint queue so the
            // old solver's solve() (inside guard.commit below) processes a
            // real Constraint::Eq — the recovered OmniML production dispatch.
            guard
                .checker
                .unify_with(return_ty, body_ty, *span, TypingContext::ReturnValue)?;
            guard.checker.add_constraint(Constraint::Eq(
                return_ty,
                body_ty,
                *span,
                crate::hir::infer::EqOrigin::Normal,
            ));
        }
        // When return_type is None, the infer var was already unified
        // with return values during body checking (via current_return_type),
        // or defaulted to Never before the solver ran (see above).

        let exit_res = guard.commit();

        if let Err(diags) = exit_res {
            let details: Vec<String> = diags.iter().map(|d| d.message().to_string()).collect();
            return Err(
                Diagnostic::error(format!("inference failure: {}", details.join("; ")))
                    .with_span(*span),
            );
        }

        // Contract<'input> verification skeleton: check that requires/ensures are bool,
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
                                .with_label(expr.span(), format!("got {:?}", self.ctx.get(ty))),
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
                                .with_label(expr.span(), format!("got {:?}", self.ctx.get(ty))),
                        );
                    }
                }
                Contract::Decreases(expr, cspan) | Contract::Terminates(expr, cspan) => {
                    let (_, ty) = self.infer_expr(expr, None)?;
                    if !self.ctx.is_numeric(ty) && !self.ctx.is_integer(ty) {
                        self.diagnostics.push(
                            Diagnostic::error("decreases/terminates expression must be an integer")
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
        let final_obs: Vec<(Span, TraitPredicate)> = self.trait_obligations.drain(..).collect();
        if !final_obs.is_empty() {
            let ctx: &mut TypeContext<'input> = &mut self.ctx;
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
                        code: crate::hir::traits::solver::ObligationCauseCode::WhereClause {
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
                            crate::hir::traits::solver::Predicate::Forall { body: body.clone() }
                        }
                        TraitPredicate::Exists { body } => {
                            crate::hir::traits::solver::Predicate::Exists { body: body.clone() }
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
                // SAFETY: `ctx_ptr` points to the `TypeContext<'input>` owned by the
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
            // The Infallible constraint (SYNTAX.md §Structured
            // Resource Cleanup): a `finally` block is reserved for
            // infallible cleanup — no `?` propagation, `leave
            // with`, `return`, or `panic` is allowed inside it.
            for s in &stmts {
                // Recursive check: an early-exit at ANY depth
                // (nested in `if`/`while`/`match`/assignment)
                // violates the infallibility constraint.
                let infallible_ok = !contains_early_exit(std::slice::from_ref(s));
                if !infallible_ok {
                    self.diagnostics.push(
                                Diagnostic::error(
                                    "finally blocks are infallible: no `?`, `leave with`, `return`, or `panic` allowed",
                                )
                                .with_span(s.span()),
                            );
                }
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
                        // constrains it to a CONCRETE type
                        // (`where T == Int<32>`).  A param-to-param
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
        //      constrained by the signature);
        //   3. `where T == U` params resolving to ANOTHER
        //      GenericParam (the declared given equivalence).
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
            // Param<'input>-to-param where-equalities (`where T == U`)
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
                    TypeData::InferVar { .. } | TypeData::SkolemVar { .. } | TypeData::Error
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
                    let mut eval = crate::hir::comptime::ComptimeEvalContext::new_with_source(
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
        // Using Cell<TypeId> allows mutation through the shared &SymbolTable<'input>
        // reference that the checker holds.
        self.symbols.update_function_return_type(*name, return_ty);

        // ── `@pure` verification ────────────────────────────────
        // A `@pure` function must have NO forbidden side effects,
        // transitively (its callees' effects are unioned into its
        // own label by `collect_function_effects`): touching a
        // mutable global, I/O, unsafe, panic, or comptime calls are
        // all disallowed.  `effect_of` is keyed by the REAL DefId
        // (`current_function` is the DefId(0) placeholder here).
        if attributes.iter().any(|a| a.name.eq_str("pure"))
            && let Some(fb) = self.symbols.lookup_function(*name)
        {
            let forbidden = EffectSet::MUTABLE_GLOBAL
                | EffectSet::IO
                | EffectSet::UNSAFE
                | EffectSet::PANIC
                | EffectSet::COMPTIME;
            if let Some(eff) = self.effect_of.get(&fb.def_id)
                && eff.intersects(forbidden)
            {
                let what = {
                    let mut parts = Vec::new();
                    if eff.contains(EffectSet::MUTABLE_GLOBAL) {
                        parts.push("reads or writes a mutable global");
                    }
                    if eff.contains(EffectSet::IO) {
                        parts.push("performs I/O");
                    }
                    if eff.contains(EffectSet::UNSAFE) {
                        parts.push("uses unsafe");
                    }
                    if eff.contains(EffectSet::PANIC) {
                        parts.push("may panic");
                    }
                    if eff.contains(EffectSet::COMPTIME) {
                        parts.push("calls a comptime function");
                    }
                    parts.join(", ")
                };
                self.diagnostics.push(
                    Diagnostic::error(format!("`@pure` function `{}` {}", name, what,))
                        .with_code_str("E117")
                        .with_span(*span)
                        .with_help(
                            "a @pure function must have no side effects, transitively — \
                                 remove the offending call or drop @pure",
                        ),
                );
            }
        }

        // The `finally` block is collected SEPARATELY (name →
        // block) and passed to the borrow check — it runs on
        // EVERY function-exit edge, wired by
        // `CfgBuilder::attach_finally` (SYNTAX.md §finally); it
        // is NOT merged into the body.
        if let Some(f) = &finally_hir {
            self.function_finally.insert(*name, f.clone());
        }

        // Disable SUBTYPE-path region collection and merge the
        // collected covariance constraints (`&'a T <: &'b T` →
        // `'a: 'b`) into the outlives graph — the function body
        // may have gone through subtype checks that recorded
        // region relationships beyond the signature's own.
        // NOTE: `guard` has been dropped by now (the code below
        // uses `self` directly) — access the ctx via `self`.
        self.ctx.region_subtype_collect.set(false);
        let collected = std::mem::take(&mut *self.ctx.region_subtype_outlives.borrow_mut());

        // Verify each BODY-collected region
        // requirement `(a, b)` (meaning `'a: 'b`) against the
        // GIVEN closure — the signature self-loops plus the
        // explicit `where 'a: 'b` edges — captured BEFORE the
        // collected edges are merged (a requirement must not
        // prove itself).  `def pick<'a,'b>(a:&'a Int, b:&'b Int)
        // -> &'b Int { return a; }` collects `('a,'b)` with no
        // `where 'a: 'b` → not satisfiable → diagnosed (rustc
        // E0623; SYNTAX.md §Explicit Lifetime Parameters
        // "mismatches cause compile errors").
        let given = self.region_outlives.clone();
        for (a, b) in &collected {
            if !region_reaches(*a, *b, &given) {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "lifetime `{}` may not live long enough — it must outlive \
                                 `{}` but no `where '_: '_:` predicate proves it",
                        a, b,
                    ))
                    .with_span(*span)
                    .with_help(
                        "add a `where '<lifetime1>: '<lifetime2>` outlives \
                                 predicate to the function signature",
                    ),
                );
            }
        }
        self.region_outlives.extend(collected);

        // SECOND region solve AFTER the body — the
        // body's subtype/unify checks collected `&'a`/`&'b`
        // covariance pairs into `region_outlives`; solving only at
        // the signature (before the body) silently accepted
        // body-level lifetime mismatches (`def pick<'a,'b>(a:&'a
        // Int, b:&'b Int) -> &'b Int { return a; }` — rustc
        // E0623).  Re-run the solver over the merged constraints;
        // uncovered return regions are now diagnosed (SYNTAX.md
        // §Explicit Lifetime Parameters: "mismatches cause
        // compile errors").
        let param_regions = self.current_param_regions.clone();
        let ret_regions = self.current_ret_regions.clone();
        for uncovered in solve_region_outlives(&param_regions, &ret_regions, &self.region_outlives)
        {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "lifetime `{}` does not appear in any of the function's parameters",
                    uncovered,
                ))
                .with_span(*span)
                .with_help(
                    "the returned reference's lifetime must be provided by a \
                             parameter — add `&'<lifetime> T` to a parameter type or \
                             remove the lifetime from the return type",
                ),
            );
        }

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

    /// Type-check an impl method's body — methods are functions, so they
    /// get the same `@auto_ro`/`@auto_coerce` gating (incl. the
    /// `@trusted`/Strict Mode rejection) and body checking as a function
    /// definition.  (Previously method bodies were only registered as
    /// signatures — they were never type-checked.)
    fn check_method_body(
        &mut self,
        m: &crate::ast::ImplMethod<'input>,
        self_ty: &crate::ast::Type<'input>,
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
        let mut hir_params: Vec<HirParam<'input>> = Vec::new();
        for p in &m.params {
            let ty = match &p.ty {
                Some(t) => {
                    let resolved = self.resolve_self_ty(t, self_ty);
                    self.resolve_type(&resolved)?
                }
                None => self.ctx.error(),
            };
            hir_params.push(HirParam {
                name: p.name,
                ty,
                default: None,
                span: p.span,
            });
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
        // Two-phase refactor: method bodies are borrow-checked in
        // Phase B (check_program) together with free-function bodies,
        // against the COMPLETE signature registry (previously they were
        // checked here, interleaved, against whatever signatures existed
        // at this point — the same order-dependence as the FunctionDef
        // arm).
        if let Ok(ref body_stmts) = result {
            self.pending_runtime_checks
                .push((m.name, body_stmts.clone()));
            // The HIR-level signature facts (the A(ρ)
            // constraints from the RESOLVED reference types) — replaces the
            // AST-level facts registered by `pre_register_signatures`.
            let receiver_ty = self.resolve_type(self_ty).ok();
            self.collect_signature_facts(
                m.name,
                &hir_params,
                Some(ret_ty),
                body_stmts,
                true,
                receiver_ty,
            );
        }
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

    fn check_block(&mut self, stmts: &[Stmt<'input>]) -> Result<Vec<HirStmt<'input>>, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_block(stmts)
    }

    fn infer_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.infer_expr(expr, expected)
    }

    fn check_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Expectation,
        ctx: TypingContext,
    ) -> Result<HirExpr<'input>, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.check_expr(expr, expected, ctx)
    }
    fn check_pattern(
        &mut self,
        pattern: &Pattern<'input>,
        expected_ty: TypeId,
    ) -> Result<HirPattern<'input>, Diagnostic> {
        let exist_depth = self.ctx.gadt.exist_skolems.borrow().len();
        let mut fc = FnCtxt::new(self);
        fc.check_pattern(pattern, expected_ty, exist_depth)
    }

    /// The post-solve GADT validation phase (the committee ruling —
    /// solve → default → validate): constructions deferred during
    /// inference are validated now that the type arguments are concrete.
    /// The new E060 trigger: after solve + defaulting, a `when` constraint
    /// is not satisfied.
    fn validate_pending_gadt_constructs(&mut self) {
        let pending = std::mem::take(&mut self.pending_gadt_constructs);
        for p in pending {
            // Copy the def_id/args out so the ctx borrow ends before the
            // mutable calls below (resolve_type / try_unify).
            let (def_id, args) = match self.ctx.get(p.enum_ty) {
                crate::hir::types::TypeData::Adt { def_id, args, .. } => (*def_id, args.clone()),
                _ => continue,
            };
            let Some(binding) = self.symbols.lookup_type_by_def_id(def_id) else {
                continue;
            };
            let Some(variant_def) = binding.variants.iter().find(|v| v.name == p.variant) else {
                continue;
            };
            // Existential GADT parameters (`Slice(exists X: &[X]) when T
            // == [X]`) are NOT in scope at the construction site — they
            // are variant-scoped and only skolemized during pattern
            // matching.  Generate fresh inference variables for them and
            // resolve the constraint THROUGH them (the same skolemization
            // the pattern-matching path uses); a plain `resolve_type`
            // would fail on `X` and spuriously emit E060.
            let mut exist_vars: Vec<TypeId> = Vec::new();
            if !variant_def.exists_params.is_empty() {
                for _ep in &variant_def.exists_params {
                    let var = self.infer.new_type_var(
                        &mut self.ctx,
                        crate::hir::infer::TypeVariableKind::Any,
                        crate::hir::infer::VarOrigin::Synthetic,
                    );
                    exist_vars.push(var);
                }
            }
            for (param_name, concrete_ty) in &variant_def.eq_spec {
                let Some((param_idx, _)) = binding
                    .params
                    .iter()
                    .enumerate()
                    .find(|(_, tp)| tp.name == *param_name)
                else {
                    continue;
                };
                let Some(actual_arg) = args.get(param_idx).copied() else {
                    continue;
                };
                // A failed resolution must NOT be silently
                // skipped — report it (fail-closed), and probe the unify in
                // a transaction (try_unify does not carry one — a failed
                // unify can leave partial bindings that pollute the other
                // pending constructions).
                let declared = match self.resolve_type_with_skolems(
                    concrete_ty,
                    &variant_def.exists_params,
                    &exist_vars,
                ) {
                    Some(d) => d,
                    None => {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "GADT variant `{}` constraint not satisfied (unresolvable)",
                                p.variant,
                            ))
                            .with_code_str("E060")
                            .with_span(p.span),
                        );
                        continue;
                    }
                };
                // Fail-closed — if the type argument is STILL
                // unresolved after solving, `try_unify` would trivially
                // succeed (an InferVar unifies with anything); the
                // construction must not be silently accepted.
                if self.type_has_unresolved_vars(actual_arg) {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "GADT variant `{}` constraint cannot be verified: type argument is still unresolved after solving",
                            p.variant,
                        ))
                        .with_code_str("E060")
                        .with_span(p.span),
                    );
                    continue;
                }
                // Also save/restore `unify_seen` and `seal_violations`
                // around the transactional probe (same discipline as
                // `is_gadt_variant_reachable`): `try_unify` mutates the
                // cycle-detection cache and the seal counter as side
                // effects, and `rollback_transaction` only restores type
                // bindings — without the save/restore, stale `unify_seen`
                // entries can make later unifications falsely detect a
                // cycle, and leaked seal-violation counts can corrupt the
                // strict-mode GADT arm check.
                let saved_seen = self.ctx.save_unify_seen();
                let saved_seal = self.ctx.seal_violations.get();
                self.ctx.begin_transaction();
                let ok = self
                    .ctx
                    .try_unify(declared, actual_arg, Some(&self.infer.region_tree))
                    .is_ok();
                self.ctx.rollback_transaction();
                self.ctx.restore_unify_seen(saved_seen);
                self.ctx.seal_violations.set(saved_seal);
                if !ok {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "GADT variant `{}` constraint not satisfied",
                            p.variant,
                        ))
                        .with_code_str("E060")
                        .with_span(p.span)
                        .with_help(format!(
                            "`{}` requires `{} == {}`, but the type argument is constrained differently",
                            p.variant,
                            param_name,
                            crate::ast::ast_type_display(concrete_ty),
                        )),
                    );
                }
            }
        }
    }

    /// Apply type-level overflow modifiers (`with overflow = ...`) to a
    /// resolved type. `Int`/`UInt` get the policy;
    /// `Float` (IEEE only — the type carries no policy field) and other
    /// types are returned unchanged.
    fn apply_overflow_modifiers(
        &mut self,
        ty: TypeId,
        modifiers: &[crate::ast::TypeModifier<'input>],
    ) -> TypeId {
        let mut ty = ty;
        for m in modifiers {
            if let crate::ast::TypeModifier::Overflow(policy) = m {
                ty = match self.ctx.get(ty) {
                    crate::hir::types::TypeData::Int { bits, signed, .. } => {
                        self.ctx.int_with_overflow(*bits, *signed, *policy)
                    }
                    crate::hir::types::TypeData::UInt { bits, .. } => {
                        self.ctx.uint_with_overflow(*bits, *policy)
                    }
                    _ => ty,
                };
            }
        }
        ty
    }

    fn resolve_type(&mut self, ty: &Type<'input>) -> Result<TypeId, Diagnostic> {
        let mut fc = FnCtxt::new(self);
        fc.resolve_type(ty)
    }

    /// Recursively replace `Self` / `self` occurrences in a type with the
    /// concrete `self_ty` (the type being implemented for).
    fn resolve_self_ty(&self, ty: &Type<'input>, self_ty: &Type<'input>) -> Type<'input> {
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
                inner: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(inner, self_ty)),
                mutable: *mutable,
                lifetime: None,
                span: *s,
            },
            Type::Pointer(inner, s) => Type::Pointer(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(inner, self_ty)),
                *s,
            ),
            Type::Generic(base, args, span) => {
                let new_base = self.resolve_self_ty(base, self_ty);
                let new_args: Vec<GenericArg<'input>> = args
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
                                value: ac.value,
                                span: ac.span,
                            })
                        }
                    })
                    .collect();
                Type::Generic(
                    self.ctx
                        .arena
                        .expect("arena required for type construction")
                        .alloc(new_base),
                    new_args,
                    *span,
                )
            }
            Type::Tuple(tys, span) => Type::Tuple(
                tys.iter()
                    .map(|t| self.resolve_self_ty(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Slice(inner, span) => Type::Slice(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(inner, self_ty)),
                *span,
            ),
            Type::Array(inner, size, span) => Type::Array(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(inner, self_ty)),
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
                ret: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(ret, self_ty)),
                span: *span,
            },
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name,
                span,
            } => Type::Projection {
                impl_type: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(impl_type, self_ty)),
                trait_path: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_ty(trait_path, self_ty)),
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
                // ── Inference-origin trace ──────────────────────────
                for (ty, label) in [(expected, "expected type"), (actual, "found type")] {
                    if let Some((vid, origin_span)) = self.infer_origin(ty) {
                        diag = diag.with_secondary_label(
                            origin_span,
                            format!("{} was originally inferred as `?{}` here", label, vid),
                        );
                    }
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
        // They accept integer and float operands (float `+%` is IEEE).
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
            // The committee ruling (float default `trap` — aligned with
            // integers): overflow-suffixed operators accept BOTH integer
            // (wrap/saturate/trap) and float operands (float `+%` is IEEE
            // semantics — an explicit opt-in; `+?` saturates; `+!` traps).
            let is_num = self.ctx.is_integer(left)
                || self.ctx.is_float(left)
                || matches!(self.ctx.get(left), TypeData::InferVar { .. });
            if !is_num {
                return Err(Diagnostic::error(
                    "overflow-suffixed operators require integer or float operands",
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

    /// Look up the inference-variable origin of a type WITHOUT following
    /// `set_binding` chains.  See `FnCtxt::infer_origin` for the rationale.
    fn infer_origin(&self, ty: TypeId) -> Option<(usize, crate::ast::Span)> {
        let var_id = self.ctx.get_infer_var_id(ty)?;
        match self.infer.var_origins().get(var_id)? {
            crate::hir::infer::VarOrigin::Expression(Some(span)) => Some((var_id, *span)),
            _ => None,
        }
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
        if let TypeData::InferVar { id, .. } = self.ctx.get(maybe_var) {
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
            // A refined type (`exists n: T invariant P(n)`) has the CARRIER's
            // kind — `exists n: Int<32> invariant n != 0` IS an integer type
            // (SYNTAX.md: the binder "is erased at runtime"; the invariant is
            // verified at construction points, not by the kind check).
            let kind_other = match self.ctx.get(resolved_other) {
                TypeData::Exists { base, .. } => *base,
                _ => resolved_other,
            };
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
                    if !self.ctx.is_bool(kind_other) {
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
                    if !self.ctx.is_integer(kind_other)
                        && !matches!(self.ctx.get(kind_other), TypeData::Rational { .. })
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
                    if !self.ctx.is_float(kind_other) {
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
                    if !self.ctx.is_numeric(kind_other) {
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
                // ── Inference-origin trace ──────────────────────────
                // Additionally, if the "other" operand's TypeId was originally
                // an InferVar (now bound to a concrete type), show where that
                // inference variable was created.
                if let Some((vid, infer_span)) = self.infer_origin(resolved_other) {
                    if Some(infer_span) != other_span {
                        d.labels_mut().push(Label::secondary(
                            infer_span,
                            format!("type was originally inferred as `?{}` here", vid),
                        ));
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
    fn resolve_trait_path(&self, bound: &Type<'input>) -> Option<DefId> {
        let path = match bound {
            Type::Path(path, _) => path,
            Type::Generic(base, ..) => match base {
                Type::Path(path, _) => path,
                _ => return None,
            },
            _ => return None,
        };
        self.symbols.lookup_trait_by_path(path)
    }

    /// Extract the name from a bound `Type` for constraint alias lookup.
    fn extract_bound_name(bound: &Type<'input>) -> Option<Symbol> {
        let base = match bound {
            Type::Path(path, _) => return path.last().copied(),
            Type::Generic(base, _, _) => base,
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

    /// Attempt to dereference through a `Deref` trait impl.
    ///
    /// `require_auto_deref` distinguishes the two consumption modes:
    /// - `true` (the AUTO form — the method-call receiver chain and the
    ///   `@auto_coerce` deref coercions): only impls marked `@auto_deref`
    ///   participate (SYNTAX.md §Method-Call Auto-Dereferencing);
    /// - `false` (the EXPLICIT form — the `*x` operator): any `Deref`
    ///   impl applies (the manual form does not need the attribute — an
    ///   unmarked impl still supports `(*x).method()`).
    fn try_deref_trait_step(&self, ty: TypeId) -> Option<TypeId> {
        self.deref_trait_step(ty, true)
    }

    fn deref_trait_step(&self, ty: TypeId, require_auto_deref: bool) -> Option<TypeId> {
        let deref_trait_id = self
            .symbols
            .lookup_trait(Symbol::intern("Deref"))
            .map(|b| b.def_id)?;
        // The type interner does NOT dedupe Adt instantiations
        // (`ctx.struct_ty` allocates a fresh id every call), so an exact
        // TypeId match against the impl's `for_type` misses most lookups.
        // Collect candidates by exact id AND by DefId equivalence.
        let ty_def_id = self.ctx.get_def_id_for_type(ty);
        let mut candidates: Vec<&crate::hir::traits::ImplCandidate<'input>> =
            self.trait_env.lookup_impls_for_type(ty);
        let extras: Vec<&crate::hir::traits::ImplCandidate<'input>> = self
            .trait_env
            .all_impls()
            .iter()
            .filter(|cand| {
                !candidates.iter().any(|c| std::ptr::eq(*c, *cand))
                    && self.ctx.get_def_id_for_type(cand.for_type) == ty_def_id
            })
            .collect();
        candidates.extend(extras);
        // Check Deref first
        for cand in &candidates {
            if cand.trait_id == deref_trait_id
                && (!require_auto_deref || cand.has_auto_deref)
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
                    && (!require_auto_deref || cand.has_auto_deref)
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
    fn autoderef_chain<'s>(&'s self, ty: TypeId) -> AutoderefIter<'s, 'input> {
        AutoderefIter::with_max_depth(self, ty, DEFAULT_MAX_DEREF_DEPTH)
    }

    /// Local type argument synthesis (Pierce & Turner 2000, §3).
    /// When a function type's parameters contain GenericParam (uninstantiated type
    /// variables), this creates fresh InferVars for them, infers argument types,
    /// unifies to bind the InferVars, and returns the resolved call result.
    fn try_synthesize_type_args(
        &mut self,
        callee_hir: &HirExpr<'input>,
        callee_ty: TypeId,
        args: &[Expr<'input>],
        comptime: bool,
        expected: Option<TypeId>,
        span: Span,
    ) -> Result<Option<(HirExpr<'input>, TypeId)>, Diagnostic> {
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
    fn collect_generic_param_indices(ty: TypeId, ctx: &TypeContext<'input>, out: &mut Vec<usize>) {
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
    fn type_var_in_problematic_position(
        ty: TypeId,
        vars: &[TypeId],
        ctx: &TypeContext<'input>,
    ) -> bool {
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
    fn type_tree_contains(ty: TypeId, target: TypeId, ctx: &TypeContext<'input>) -> bool {
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

        // Walk autoderef chain, skipping the original type (already tried).
        // Collect the chain first so the iterator's `&self` borrow ends
        // before the loop body mutates `self.ctx` / `self.symbols`.
        let deref_chain: Vec<TypeId> = self.autoderef_chain(ty).skip(1).collect();
        for deref_ty in deref_chain {
            let data = self.ctx.get(deref_ty);
            // Extract everything needed from `data` in ONE match so its
            // immutable borrow of `self.ctx` ends before the mutable
            // `self.ctx.subst` below.
            let (def_id, args) = match data {
                TypeData::Adt { def_id, args, .. } => (Some(*def_id), args.to_vec()),
                _ => (None, Vec::new()),
            };
            if let Some(def_id) = def_id {
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
    /// Returns `(param_types, return_type, method_def_id)` if found — the
    /// DefId is the method's OWN identity (allocated in the resolver), so
    /// callers can address the method independently of its impl.
    /// Method lookup with a cache (the rustc tcx-query pattern): the
    /// autoderef chain and the per-type method scan are recomputed only on
    /// a miss.  The cache is cleared whenever inherent methods/impls are
    /// registered.
    fn lookup_method(&mut self, ty: TypeId, name: Symbol) -> Option<(Vec<TypeId>, TypeId, DefId)> {
        // The cache is keyed on the RESOLVED type — an unresolved
        // InferVar queried BEFORE binding would cache a stale `None`
        // (the method "missing" even after the var is bound to a
        // concrete type).  The same resolve-before-query discipline as
        // `TypeContext::characteristic` / `check_variance`.
        let ty = self.ctx.resolve_binding(ty);
        if let Some(cached) = self.method_cache.get(&(ty, name)) {
            return cached.clone();
        }
        let result = self.lookup_method_uncached(ty, name);
        self.method_cache.insert((ty, name), result.clone());
        result
    }

    fn lookup_method_uncached(
        &mut self,
        ty: TypeId,
        name: Symbol,
    ) -> Option<(Vec<TypeId>, TypeId, DefId)> {
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
                    return Some((method.param_tys.clone(), method.ret_ty, method.def_id));
                }
            }

            // Check trait impl methods via exact match.
            for cand in self.trait_env.lookup_impls_for_type(current_ty) {
                for method in &cand.resolved_methods {
                    if method.name == name {
                        return Some((method.param_tys.clone(), method.ret_ty, method.def_id));
                    }
                }
            }

            // Fallback: try generic impl matching for every trait.  Each
            // attempt runs in a `CheckerProbe`: `lookup_impl_generic`
            // COMMITS its unification when the impl's `for_type` unifies
            // with the receiver (traits/mod.rs — the transaction is
            // commit-on-success), but if that impl does not provide the
            // requested method we must not keep the attempt's bindings —
            // the unification pinned the generic params to this receiver
            // even though the candidate is skipped.  Commit only when the
            // method is actually found; dropping the probe rolls the
            // attempt back (bindings, inference state, diagnostics).
            for &trait_id in &all_trait_ids {
                let mut probe = self.begin_probe();
                let found = probe
                    .with(|c| {
                        let Some((cand, subst)) = c
                            .trait_env
                            .lookup_impl_generic(trait_id, current_ty, c.ctx, c.symbols)
                        else {
                            return Ok::<_, ()>(None);
                        };
                        for method in &cand.resolved_methods {
                            if method.name == name {
                                let param_tys: Vec<TypeId> = method
                                    .param_tys
                                    .iter()
                                    .map(|&p| c.ctx.subst(p, &subst))
                                    .collect();
                                let ret_ty = c.ctx.subst(method.ret_ty, &subst);
                                return Ok::<_, ()>(Some((param_tys, ret_ty, method.def_id)));
                            }
                        }
                        Ok::<_, ()>(None)
                    })
                    .expect("the probe closure is infallible");
                if let Some(hit) = found {
                    probe.commit();
                    return Some(hit);
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
    ) -> Result<Option<Expr<'input>>, Diagnostic> {
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

    fn block_type(&self, stmts: &[HirStmt<'input>]) -> TypeId {
        self.block_type_impl(stmts, true)
    }

    /// Whether an implicit trailing expression counts as the block's return type.
    /// Functions (`def`) require explicit `return`; closures and blocks allow
    /// trailing expressions as implicit return values.
    fn block_type_impl(&self, stmts: &[HirStmt<'input>], allow_implicit: bool) -> TypeId {
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
                // A function body whose FINAL statement is an `if` whose
                // branches return values (`if c { return x; } else
                // { return y; }`) must be typed by the branch returns —
                // the old code fell through to Unit, wrongly rejecting
                // valid bodies ("expected T, found Unit").
                HirStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    let then_ty = self.block_type_impl(then_branch, allow_implicit);
                    let else_ty = else_branch
                        .as_ref()
                        .map(|b| self.block_type_impl(b, allow_implicit))
                        .unwrap_or(self.ctx.unit());
                    if then_ty != self.ctx.unit() && then_ty == else_ty {
                        return then_ty;
                    }
                }
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

    fn extract_int_from_type(&self, ty: &Type<'input>) -> Option<u32> {
        if let Type::Literal(expr, _) = ty
            && let Expr::Literal(Literal::Int(val), _) = expr
        {
            match val.to_u64() {
                Some(n) if n <= 64 => Some(n as u32),
                _ => None, // reject out-of-range or negative bit widths silently
            }
        } else {
            None
        }
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
        // Own the resolved data so the `&mut self` borrow from
        // `resolve_gadt_variant_info` ends before the reachability probe.
        let (binding, vd, args) = (binding.clone(), vd.clone(), args);
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
            // A reachability PROBE must not pollute
            // the `seal_violations` soundness counter — probe-induced
            // guard hits are always recovered by the rollback.
            let saved_seal = self.ctx.seal_violations.get();
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
            self.ctx.seal_violations.set(saved_seal);
            return all_satisfied;
        }
        // Non-existential GADT reachability: use a single transaction
        // across all constraints so that shared substitutions are preserved.
        // Also save/restore `unify_seen` (same discipline as `can_unify`).
        let saved_seen = self.ctx.save_unify_seen();
        // The probe must not pollute the seal counter.
        let saved_seal = self.ctx.seal_violations.get();
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
        self.ctx.seal_violations.set(saved_seal);
        all_satisfied
    }

    /// Shared lookup: given a pattern and scrutinee type, find the matching
    /// GADT variant and its type arguments.  Returns the `TypeBinding<'input>` (owned),
    /// the variant definition, and the resolved type arguments.
    /// Returns `None` for non-enum patterns or when lookups fail.
    /// Used by `is_gadt_variant_reachable` and `apply_gadt_refinement`.
    fn resolve_gadt_variant_info(
        &self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern,
        span: crate::ast::Span,
    ) -> Option<(
        TypeBinding<'input>,
        crate::ast::EnumVariant<'input>,
        Vec<TypeId>,
    )> {
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

    /// Look up the TypeBinding<'input> for a Struct or Enum type, if available.
    fn lookup_type_binding(&self, ty: TypeId) -> Option<TypeBinding<'input>> {
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
        pattern: &crate::ast::Pattern<'input>,
        span: crate::ast::Span,
    ) -> Result<(HirPattern<'input>, bool, GadtArmGuard<'input>), Diagnostic> {
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
        pattern: &crate::ast::Pattern<'input>,
        span: crate::ast::Span,
        body: impl FnOnce(&mut Self, bool) -> Result<T, Diagnostic>,
    ) -> Result<(HirPattern<'input>, T, bool), Diagnostic> {
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
        let pending = self.ctx.gadt.take_pending_eqs();
        for eq in pending {
            // Nested GADT satisfiability gate: an impossible
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
        binding: &TypeBinding<'input>,
        args: &[TypeId],
        param_name: &Symbol,
        concrete_ty: &crate::ast::Type<'input>,
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
        alt_eqs: &[Vec<PendingInnerGadtEq<'input>>],
        alt_reachable: &[bool],
        span: Span,
    ) {
        if alt_eqs.is_empty() {
            return;
        }
        // The order-independent intersection: anchor the intersection on the
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
                // Per-alternative reachability  : an unreachable
                // alternative's equalities can neither conflict with nor
                // contribute to the reachable ones — ignore them.  The
                // base itself is reachable by construction (base_idx).
                if i == base_idx || !alt_reachable.get(i).copied().unwrap_or(true) {
                    continue;
                }
                // Key: (binding.def_id, param_name) — compare bindings by
                // def_id (TypeBinding<'input> has no PartialEq).  The RHS types are
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
                self.ctx.gadt.push_pending_eq(eq.clone());
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
    fn or_eq_concrete_equal(
        &mut self,
        a: &PendingInnerGadtEq<'input>,
        b: &PendingInnerGadtEq<'input>,
    ) -> bool {
        let resolve = |ctx: &mut Self, eq: &PendingInnerGadtEq<'input>| {
            // Resolve through the skolems first, falling back to a plain
            // type resolution (a nested `||` closure would capture `eq`,
            // whose lifetime is only valid inside this closure body).
            if let Some(t) =
                ctx.resolve_type_with_skolems(&eq.concrete_ty, &eq.exist_params, &eq.skolems)
            {
                Some(t)
            } else {
                ctx.resolve_type(&eq.concrete_ty).ok()
            }
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
                self.type_data_alpha_eq(&da, &db) && self.exists_invariants_alpha_eq(ta, tb)
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

    /// § identity: two resolved `exists` types are the same type only if
    /// their INVARIANTS are alpha-equivalent — `exists X. Int invariant
    /// X > 0` and `exists Y. Int invariant Y < 0` are DIFFERENT types (the
    /// invariant restricts the valid instances).  The invariant lives in
    /// the side `TypeMeta` (NOT in `TypeData::Exists`), so the comparison
    /// happens here with the TypeIds at hand — mirroring the AST-level
    /// comparison in `type_eq` (same alpha-renaming + capture guards).
    /// Non-`exists` pairs return `true` (the structural compare decided).
    fn exists_invariants_alpha_eq(&self, ta: TypeId, tb: TypeId) -> bool {
        let (na, ia) = match self.ctx.get(ta) {
            TypeData::Exists { name, .. } => (*name, self.ctx.get_invariant(ta)),
            _ => return true,
        };
        let (nb, ib) = match self.ctx.get(tb) {
            TypeData::Exists { name, .. } => (*name, self.ctx.get_invariant(tb)),
            _ => return true,
        };
        match (ia, ib) {
            (Some(a), Some(b)) => {
                // L1/L2: the invariant comparison is discreteness-aware for
                // INTEGER bases (`X > 0` ≡ `X >= 1`); floats/rationals are
                // dense and compare the bounds exactly.
                let is_int = self
                    .ctx
                    .base_of_exists(ta)
                    .map(|base| self.ctx.is_integer(base))
                    .unwrap_or(false);
                if na == nb {
                    crate::hir::type_eq::expr_eq_ignoring_spans_typed(a, b, is_int)
                } else if !crate::hir::type_eq::expr_free_in(nb, a)
                    && !crate::hir::type_eq::expr_free_in(na, b)
                {
                    let map = [(na, nb)];
                    crate::hir::type_eq::expr_eq_ignoring_spans_renamed_typed(a, b, &map, is_int)
                } else {
                    false
                }
            }
            (None, None) => true,
            // One side carries an invariant, the other does not — the two
            // types admit different value sets.
            _ => false,
        }
    }

    fn register_single_gadt_eq(
        &mut self,
        param_name: &Symbol,
        concrete_ty: &Type<'input>,
        binding: &TypeBinding<'input>,
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
            // Rule 3 (the committee ruling — the reserved extension point): a
            // `when X == Y` equality where the RHS references ANOTHER
            // existential variable.  The two witnesses are equivalent and
            // interchangeable.  Register the equivalence deterministically
            // (by skolem id — the smaller id is the canonical lhs, so the
            // ordering never depends on processing order).
            if let crate::ast::Type::Path(segments, _) = concrete_ty {
                if segments.len() == 1
                    && let Some(&other) = exist_params
                        .iter()
                        .position(|p| *p == segments[0])
                        .and_then(|i| skolems.get(i))
                    && other != skolem
                {
                    let (lhs, rhs) = if skolem.raw() < other.raw() {
                        (skolem, other)
                    } else {
                        (other, skolem)
                    };
                    self.ctx.register_existential_equation(lhs, rhs);
                    return;
                }
            }
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
        ty: &Type<'input>,
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
        ty: &Type<'input>,
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
                        match self.resolve_type(&Type::Path(
                            smallvec::SmallVec::from(path.clone()),
                            ty.span(),
                        )) {
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
                        crate::ast::GenericArg::Const(ac) => {
                            self.resolve_type(&Type::Expr(ac.value, ac.span)).ok()
                        }
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
            Type::Reference {
                inner,
                mutable,
                lifetime,
                ..
            } => {
                let inner_ty = self.resolve_type_with_skolems(inner, exist_params, skolems)?;
                Some(self.ctx.alloc(TypeData::Ref {
                    ty: inner_ty,
                    mutable: *mutable,
                    // The explicit lifetime annotation survives the
                    // lowering (previously dropped by `..`) — the borrow
                    // checker maps it to the early-bound region.
                    lifetime: *lifetime,
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
                let size_val = match size {
                    crate::ast::Expr::Literal(crate::ast::Literal::Int(n), _) => {
                        match n.to_u64() {
                            Some(sz) => sz,
                            None => return None, // If the value is negative or exceeds the range of u64, fallback to an ordinary resolve_type.
                        }
                    }
                    // For non-literal size expressions, attempt comptime eval
                    // via the HirExpr<'input> if available, otherwise skip.
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
    fn type_to_string(ty: &Type<'input>) -> String {
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
                                Self::type_to_string(&crate::ast::Type::Expr(ac.value, ac.span))
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
    pub(crate) fn insert_literal_value(&mut self, name: Symbol, value: ComptimeValue<'input>) {
        self.literal_values.entry(name).or_default().push(value);
        self.scope_var_stack
            .last_mut()
            .expect("insert_literal_value outside any literal scope")
            .push(name);
    }

    /// Get the current (innermost) comptime-known value for `name`,
    /// or `None` if the variable has no comptime-known value in the
    /// current or any enclosing scope.
    pub(crate) fn get_literal_value<'s>(
        &'s self,
        name: &Symbol,
    ) -> Option<&'s ComptimeValue<'input>> {
        self.literal_values.get(name)?.last()
    }

    /// Check whether a statement should be skipped due to `@cfg` evaluation.
    /// `@auto_ro` (SYNTAX.md §Local Relaxation) is only meaningful on
    /// function definitions.  A placement on any other item is a surface
    /// contract violation and must be reported, not silently ignored.
    /// Resolve a function's borrow signature from its resolved
    /// reference types (the `TypeData::Ref` detection) + the output-
    /// derivation analysis, and store the A(ρ) signature facts.
    pub(crate) fn collect_signature_facts(
        &mut self,
        name: Symbol,
        params: &[HirParam<'input>],
        return_type: Option<TypeId>,
        body: &[HirStmt<'input>],
        is_method: bool,
        receiver_ty: Option<TypeId>,
    ) {
        let param_refs: Vec<(Symbol, bool, bool, Option<Symbol>)> = params
            .iter()
            .map(|p| match self.ctx.get(p.ty) {
                TypeData::Ref {
                    mutable, lifetime, ..
                } => (p.name, true, *mutable, *lifetime),
                _ => (p.name, false, false, None),
            })
            .collect();
        let return_refs = return_type
            .map(|t| crate::hir::polonius::count_return_refs(&self.ctx, t))
            .unwrap_or(0);
        let sig = crate::hir::polonius::extract_borrow_signature(&param_refs, return_refs);
        let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
        let deriving = crate::hir::polonius::derive_output_origins(body, &param_names);
        let facts = crate::hir::polonius::signature_facts(&sig, &deriving);
        // rustc's dangling-reference rejection: a REFERENCE return
        // that references a LOCAL (non-parameter) place would dangle
        // after the function returns.
        if return_refs > 0 {
            for span in crate::hir::polonius::dangling_return_spans(body, &param_names) {
                self.diagnostics.push(
                    Diagnostic::error(
                        "cannot return a reference to a local value (dangling borrow)",
                    )
                    .with_span(span),
                );
            }
        }
        // The input-borrow POSITIONS (the reference-param indices) — for
        // the extractor's call-site mapping.
        let input_positions: Vec<usize> = params
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(self.ctx.get(p.ty), TypeData::Ref { .. }))
            .map(|(i, _)| i)
            .collect();
        // `pre_register_signatures` (AST-level) pushes first,
        // and the consumer `find`-s the FIRST match — so the more precise
        // HIR-level facts would be shadowed by the AST-level entry.  Remove
        // the AST-level entry before pushing (the HIR is authoritative).
        // The dedup is scoped to entries of the SAME kind: an impl method
        // with the same name but a different receiver (or a free function
        // vs. a method) must NOT be evicted — the name-only `retain` used
        // to drop the method's A(ρ) facts (the order-dependent
        // cross-function freeze).
        self.signature_facts.retain(|(nm, mi, receiver, _, _)| {
            !(*nm == name && *mi == is_method && *receiver == receiver_ty)
        });
        self.signature_facts
            .push((name, is_method, receiver_ty, input_positions, facts));
    }

    fn validate_auto_ro_placement(&mut self, stmt: &Stmt<'input>) {
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
                // A future attribute-bearing `Stmt<'input>` variant must not
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
    fn should_skip_due_to_cfg(&mut self, stmt: &Stmt<'input>) -> bool {
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
    fn check_cfg_reachability(&mut self, stmt: &Stmt<'input>) {
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

fn extract_labels_from_expr<'input>(e: &Expr<'input>) -> Vec<Symbol> {
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

fn extract_labels_from_stmt<'input>(s: &Stmt<'input>) -> Vec<Symbol> {
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
            .trait_name_by_def_id(trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)))
            .map(|s| s.as_str())
            .unwrap_or_else(|| {
                format!(
                    "trait#{}",
                    trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)).0
                )
            });
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
// push both HirStmt<'input> and HirExpr<'input> references onto one Vec.
enum Node<'a, 'input> {
    Stmt(&'a HirStmt<'input>),
    Expr(&'a HirExpr<'input>),
}

/// Iteratively check whether a tree of `HirStmt<'input>` / `HirExpr<'input>` contains any
/// `Error` node, using an explicit stack to avoid stack overflow on deeply
/// nested code (e.g. 1000+ nested `if` expressions).
///
/// This traversal is scoped to a single function body (passed as `stmts`),
/// not the entire program, so O(n) complexity is acceptable — the
/// subsequent comptime evaluation is also O(n) for the same body.
/// Returns `true` the moment an `Error` is found (short-circuit).
fn contains_error<'input>(stmts: &[HirStmt<'input>]) -> bool {
    let mut stack: Vec<Node<'_, '_>> = stmts.iter().map(Node::Stmt).collect();

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
            // All other HirStmt<'input> variants have no nested Error-carrying nodes.
            Node::Stmt(_) => {}

            // ── HirExpr<'input> ──
            Node::Expr(HirExpr::Error(_)) => return true,
            Node::Expr(e) => push_expr_children(&mut stack, e),
        }
    }
    false
}

/// Whether any statement in the slice (or anything nested inside it —
/// `if`/`if let`/`while`/`loop` bodies, `match`/`call`/binary operands
/// inside expressions, assignment values) is an early-exit /
/// error-propagation term: `leave with` (incl. the `?` lowering),
/// `return`, or `try`.  The Infallible constraint (SYNTAX.md
/// §Structured Resource Cleanup) forbids these inside `finally` blocks
/// at ANY depth — a top-level-only `matches!` would let `if true {
/// return; }` or `x = f()?;` bypass the check.
fn contains_early_exit(stmts: &[crate::hir::hir::HirStmt<'_>]) -> bool {
    let mut stack: Vec<Node<'_, '_>> = stmts.iter().map(Node::Stmt).collect();
    while let Some(node) = stack.pop() {
        match node {
            Node::Expr(
                crate::hir::hir::HirExpr::LeaveWith { .. }
                | crate::hir::hir::HirExpr::Return { .. }
                | crate::hir::hir::HirExpr::Try { .. },
            ) => return true,
            Node::Expr(crate::hir::hir::HirExpr::Error(_)) => {}
            Node::Expr(e @ crate::hir::hir::HirExpr::Call { callee, .. }) => {
                // A call to the `panic` builtin (a `@diverges` function) is
                // an early exit — `finally` blocks are Infallible
                // (SYNTAX.md §finally) and must not panic.
                if let crate::hir::hir::HirExpr::Ident(s, _, _) = callee.as_ref()
                    && s.eq_str("panic")
                {
                    return true;
                }
                push_expr_children(&mut stack, e);
            }
            Node::Expr(e) => push_expr_children(&mut stack, e),
            Node::Stmt(crate::hir::hir::HirStmt::Leave { .. })
            | Node::Stmt(crate::hir::hir::HirStmt::Return { .. }) => return true,
            Node::Stmt(crate::hir::hir::HirStmt::Expression(e)) => {
                stack.push(Node::Expr(e));
            }
            Node::Stmt(
                crate::hir::hir::HirStmt::If {
                    then_branch,
                    else_branch,
                    ..
                }
                | crate::hir::hir::HirStmt::IfLet {
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
                crate::hir::hir::HirStmt::While { body, .. }
                | crate::hir::hir::HirStmt::WhileLet { body, .. }
                | crate::hir::hir::HirStmt::For { body, .. }
                | crate::hir::hir::HirStmt::Loop { body, .. }
                | crate::hir::hir::HirStmt::ComptimeBlock { body, .. }
                | crate::hir::hir::HirStmt::ScopeCleanup { body, .. }
                | crate::hir::hir::HirStmt::Unsafe { body, .. }
                | crate::hir::hir::HirStmt::Isolate { body, .. }
                | crate::hir::hir::HirStmt::Generate { body, .. },
            ) => {
                stack.extend(body.iter().map(Node::Stmt));
            }
            Node::Stmt(crate::hir::hir::HirStmt::Assign { value, .. }) => {
                stack.push(Node::Expr(value));
            }
            Node::Stmt(crate::hir::hir::HirStmt::VariableDef {
                value: Some(value), ..
            }) => {
                stack.push(Node::Expr(value));
            }
            Node::Stmt(_) => {}
        }
    }
    false
}

/// Push the children of an expression onto the stack.
fn push_expr_children<'a, 'input>(stack: &mut Vec<Node<'a, 'input>>, expr: &'a HirExpr<'input>) {
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
fn try_fast_path<'input>(ensures_expr: &Expr<'input>, return_value: &Expr<'input>) -> bool {
    false
}

/// Replace the `codomain` identifier (and any `@label` identifiers)
/// in an expression with the return value expression.
/// Used by the SMT-based contract verification path.
#[allow(dead_code)]
fn replace_codomain<'input>(
    arena: &'input bumpalo::Bump,
    expr: &'input Expr<'input>,
    replacement: &'input Expr<'input>,
) -> &'input Expr<'input> {
    match expr {
        Expr::Ident(name, _) if name.eq_str("codomain") || name.as_str().starts_with('@') => {
            replacement
        }
        Expr::BinaryOp {
            left,
            op,
            right,
            span,
        } => arena.alloc(Expr::BinaryOp {
            left: replace_codomain(arena, left, replacement),
            op: *op,
            right: replace_codomain(arena, right, replacement),
            span: *span,
        }),
        Expr::UnaryOp {
            op,
            expr: inner,
            span,
        } => arena.alloc(Expr::UnaryOp {
            op: *op,
            expr: replace_codomain(arena, inner, replacement),
            span: *span,
        }),
        Expr::Call {
            callee,
            args,
            comptime,
            span,
        } => arena.alloc(Expr::Call {
            callee: replace_codomain(arena, callee, replacement),
            args: args
                .iter()
                .map(|a| replace_codomain(arena, a, replacement).clone())
                .collect(),
            comptime: *comptime,
            span: *span,
        }),
        Expr::FieldAccess { base, field, span } => arena.alloc(Expr::FieldAccess {
            base: replace_codomain(arena, base, replacement),
            field: *field,
            span: *span,
        }),
        Expr::Index { base, index, span } => arena.alloc(Expr::Index {
            base: replace_codomain(arena, base, replacement),
            index: replace_codomain(arena, index, replacement),
            span: *span,
        }),
        _ => expr,
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

/// Check whether a HirExpr<'input> tree contains any runtime identifier references.
/// Used to enforce that `scope_cleanup when` conditions are compile-time
/// predicates.  `ghost_var_scopes` contains names of ghost variables that
/// are allowed in compile-time predicates (SYNTAX.md: "may reference only
/// ghost variables and other compile-time-constant expressions").
/// `runtime_var_scopes` contains names of RUNTIME variables; a runtime
/// binding in an INNER scope shadows an outer ghost of the same name and
/// must be treated as runtime (rejected).
/// Whether `target` is a PREFIX of `frozen` (equality included): mutating
/// `target` would touch the storage of the frozen place.  E.g. `a.b` is a
/// prefix of `a.b.c` (mutating `a.b` clobbers `a.b.c`), but `a.c` is not.
/// The shared `place_is_prefix_of` predicate lives in `hir::place`
/// (the duplicated copy was removed).
pub(crate) use crate::hir::place::place_is_prefix_of;

fn contains_runtime_ident<'input>(
    expr: &HirExpr<'input>,
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
        // This is safe (rejects valid code) but not ideal — if a new HirExpr<'input>
        // variant is added without updating this match, it will be rejected.
        // (The `#[non_exhaustive]` attribute on `HirExpr<'input>` ensures downstream
        // crates add a wildcard arm; this wildcard is the safe fallback.)
        _ => true,
    }
}

/// Check whether a HirStmt<'input> tree contains any runtime identifier references.
/// Used by `contains_runtime_ident` for block traversal.
fn contains_stmt_runtime_ident<'input>(
    stmt: &HirStmt<'input>,
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

/// Try to convert an AST expression into a loop-IR `Cond`.
/// Only simple comparisons are handled (==, !=, <, >, <=, >=, &&, ||, !
/// between variables/constants). Returns `None` if the expression
/// references a non-loop variable (e.g. `codomain`).
fn ast_expr_to_cond(
    expr: &crate::ast::Expr,
    vars: &[Symbol],
    signed: &[bool],
) -> Option<crate::hir::loop_ir::Cond> {
    use crate::hir::loop_ir::{CmpOp, Cond};

    match expr {
        // ── Boolean literals ──────────────────────────────────────
        // `true` / `false` map directly to Cond::True / Cond::False.
        crate::ast::Expr::Literal(crate::ast::Literal::Bool(true), _) => Some(Cond::True),
        crate::ast::Expr::Literal(crate::ast::Literal::Bool(false), _) => Some(Cond::False),

        // ── Type-annotated expression: strip annotation, recurse ──
        // `(expr: Type)` carries no semantic effect on the value;
        // peel the annotation and convert the inner expression.
        crate::ast::Expr::TypeAnnotated { expr: inner, .. } => {
            ast_expr_to_cond(inner, vars, signed)
        }

        // ── Binary operators ──────────────────────────────────────
        crate::ast::Expr::BinaryOp {
            left, op, right, ..
        } => match op {
            crate::ast::BinOp::And => {
                let l = ast_expr_to_cond(left, vars, signed)?;
                let r = ast_expr_to_cond(right, vars, signed)?;
                Some(Cond::And(Box::new(l), Box::new(r)))
            }
            crate::ast::BinOp::Or => {
                let l = ast_expr_to_cond(left, vars, signed)?;
                let r = ast_expr_to_cond(right, vars, signed)?;
                Some(Cond::Or(Box::new(l), Box::new(r)))
            }
            crate::ast::BinOp::Eq
            | crate::ast::BinOp::Neq
            | crate::ast::BinOp::Lt
            | crate::ast::BinOp::Gt
            | crate::ast::BinOp::Le
            | crate::ast::BinOp::Ge => {
                let lhs = ast_expr_to_scalar(left, vars)?;
                let rhs = ast_expr_to_scalar(right, vars)?;
                let cmp_op = match op {
                    crate::ast::BinOp::Eq => CmpOp::Eq,
                    crate::ast::BinOp::Neq => CmpOp::Neq,
                    crate::ast::BinOp::Lt => CmpOp::Lt,
                    crate::ast::BinOp::Gt => CmpOp::Gt,
                    crate::ast::BinOp::Le => CmpOp::Le,
                    crate::ast::BinOp::Ge => CmpOp::Ge,
                    _ => unreachable!(),
                };
                let s = scalar_uses_signed(&lhs, signed) || scalar_uses_signed(&rhs, signed);
                Some(Cond::Cmp {
                    op: cmp_op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    signed: s,
                })
            }
            _ => None,
        },

        // ── Unary NOT ─────────────────────────────────────────────
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Not,
            expr: inner,
            ..
        } => {
            let c = ast_expr_to_cond(inner, vars, signed)?;
            Some(Cond::Not(Box::new(c)))
        }

        _ => None,
    }
}

/// Try to convert an AST expression into a loop-IR `ScalarExpr`.
/// Handles: identifiers (loop variables), integer literals, Add, Sub,
/// unary negation, and type-annotated expressions.
/// Returns `None` for any form outside the linear subset.
fn ast_expr_to_scalar(
    expr: &crate::ast::Expr,
    vars: &[Symbol],
) -> Option<crate::hir::loop_ir::ScalarExpr> {
    use crate::hir::loop_ir::{ArithSem, ScalarExpr};

    let idx = |s: Symbol| -> Option<usize> { vars.iter().position(|v| *v == s) };

    match expr {
        // Loop variable reference
        crate::ast::Expr::Ident(name, _) => Some(ScalarExpr::Var(idx(*name)?)),

        // Integer literal
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => {
            Some(ScalarExpr::Const(num_bigint::BigInt::from(v.to_i128()?)))
        }

        // Addition: a + b
        crate::ast::Expr::BinaryOp {
            left,
            op: crate::ast::BinOp::Add,
            right,
            ..
        } => {
            let l = ast_expr_to_scalar(left, vars)?;
            let r = ast_expr_to_scalar(right, vars)?;
            Some(ScalarExpr::Add(Box::new(l), Box::new(r), ArithSem::Wrap))
        }

        // Subtraction: a - b
        crate::ast::Expr::BinaryOp {
            left,
            op: crate::ast::BinOp::Sub,
            right,
            ..
        } => {
            let l = ast_expr_to_scalar(left, vars)?;
            let r = ast_expr_to_scalar(right, vars)?;
            Some(ScalarExpr::Sub(Box::new(l), Box::new(r), ArithSem::Wrap))
        }

        // ── Unary negation: -x → 0 - x ──────────────────────────
        // The loop IR has no unary neg node; encode as Sub(0, x).
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr: inner,
            ..
        } => {
            let inner_s = ast_expr_to_scalar(inner, vars)?;
            Some(ScalarExpr::Sub(
                Box::new(ScalarExpr::Const(num_bigint::BigInt::zero())),
                Box::new(inner_s),
                ArithSem::Wrap,
            ))
        }

        // ── Type-annotated expression: strip annotation, recurse ──
        // `(x: Int<32>)` carries no semantic effect on the value;
        // peel the annotation and convert the inner expression.
        crate::ast::Expr::TypeAnnotated { expr: inner, .. } => ast_expr_to_scalar(inner, vars),

        _ => None,
    }
}

/// Whether a `ScalarExpr` involves a signed variable.
fn scalar_uses_signed(e: &crate::hir::loop_ir::ScalarExpr, signed: &[bool]) -> bool {
    match e {
        crate::hir::loop_ir::ScalarExpr::Var(i) => signed.get(*i).copied().unwrap_or(false),
        crate::hir::loop_ir::ScalarExpr::Const(_) => false,
        crate::hir::loop_ir::ScalarExpr::Add(l, r, _)
        | crate::hir::loop_ir::ScalarExpr::Sub(l, r, _) => {
            scalar_uses_signed(l, signed) || scalar_uses_signed(r, signed)
        }
        crate::hir::loop_ir::ScalarExpr::Ite(_, t, f) => {
            scalar_uses_signed(t, signed) || scalar_uses_signed(f, signed)
        }
    }
}

#[cfg(test)]
pub mod tests;

/// Collect the non-Copy variable roots for the static move check —
/// recursively through the nested blocks (if/while/for bodies), so an
/// independent non-Copy binding inside a nested block (e.g. a function
/// call returning a `String`) is tracked too.
pub(crate) fn collect_non_copy_roots<'input>(
    stmts: &[HirStmt<'input>],
    ctx: &TypeContext<'input>,
    out: &mut Vec<Symbol>,
) {
    for stmt in stmts {
        match stmt {
            HirStmt::VariableDef {
                name: Some(n), ty, ..
            } => {
                if !ctx.type_is_copy(*ty) {
                    out.push(*n);
                }
            }
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            }
            | HirStmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_non_copy_roots(then_branch, ctx, out);
                if let Some(eb) = else_branch {
                    collect_non_copy_roots(eb, ctx, out);
                }
            }
            HirStmt::While { body, .. }
            | HirStmt::WhileLet { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Loop { body, .. } => {
                collect_non_copy_roots(body, ctx, out);
            }
            // The expression statements: recurse into EVERY expression
            // form that carries statement containers (Block / If / IfLet
            // / Match arms / Closure / Task / Catch branches) — a
            // non-Copy binding inside an expression-position if/match
            // was previously invisible to the move tracking
            // (use-after-move false negatives).
            HirStmt::Expression(e) => collect_expr_non_copy_roots(e, ctx, out),
            _ => {}
        }
    }
}

/// Recurse into the statement containers of an expression (the same
/// surface as `used_vars_in_expr_into` — a shared helper would be
/// better; the two must not drift again).
fn collect_expr_non_copy_roots<'input>(
    e: &HirExpr<'input>,
    ctx: &TypeContext<'input>,
    out: &mut Vec<Symbol>,
) {
    match e {
        HirExpr::Block(stmts, _, _) => collect_non_copy_roots(stmts, ctx, out),
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
            collect_non_copy_roots(then_branch, ctx, out);
            if let Some(eb) = else_branch {
                collect_non_copy_roots(eb, ctx, out);
            }
        }
        HirExpr::Match { arms, .. } => {
            for arm in arms {
                collect_expr_non_copy_roots(&arm.body, ctx, out);
            }
        }
        HirExpr::Closure { body, .. } => collect_non_copy_roots(body, ctx, out),
        HirExpr::Task { block, .. } => collect_non_copy_roots(block, ctx, out),
        HirExpr::Catch { branches, .. } => {
            for b in branches {
                collect_non_copy_roots(&b.body, ctx, out);
            }
        }
        _ => {}
    }
}

/// The AST-level pre-registration: the FIRST pass
/// registers every function's borrow signature BEFORE any body check —
/// the cross-function loan issuance becomes order-independent (a `main`
/// before its callee no longer skips the callee's A(ρ) facts).  The
/// AST-level reference detection (`Type::Reference` — `&`/`&mut`)
/// covers the direct-reference parameters; the HIR-level collection in
/// the FunctionDef<'input> arm remains for the resolved-type precision.
pub(crate) fn pre_register_signatures<'input: 'a, 'a>(
    checker: &mut TypeChecker<'a, 'input>,
    program: &crate::ast::Program<'input>,
) {
    for item in &program.items {
        if let crate::ast::Stmt::FunctionDef {
            name,
            params,
            return_type,
            body,
            // The comptime functions are compile-time — they never exist
            // at runtime (SYNTAX.md §const — "never exist at runtime"), so
            // they do NOT participate in the runtime borrow-signature
            // registry (the A(ρ) cross-function freeze).
            is_comptime: false,
            ..
        } = item
        {
            let param_refs: Vec<(Symbol, bool, bool, Option<Symbol>)> = params
                .iter()
                .map(|p| match p.ty.as_ref() {
                    Some(crate::ast::Type::Reference {
                        mutable, lifetime, ..
                    }) => (p.name, true, *mutable, *lifetime),
                    _ => (p.name, false, false, None),
                })
                .collect();
            let return_refs = return_type
                .as_ref()
                .map(|t| crate::hir::polonius::count_return_refs_ast(t))
                .unwrap_or(0);
            let sig = crate::hir::polonius::extract_borrow_signature(&param_refs, return_refs);
            let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
            let deriving = derive_output_origins_ast(body.as_deref().unwrap_or(&[]), &param_names);
            let facts = crate::hir::polonius::signature_facts(&sig, &deriving);
            let input_positions: Vec<usize> = params
                .iter()
                .enumerate()
                .filter(|(_, p)| matches!(p.ty.as_ref(), Some(crate::ast::Type::Reference { .. })))
                .map(|(i, _)| i)
                .collect();
            checker
                .signature_facts
                .push((*name, false, None, input_positions, facts));
        } else if let crate::ast::Stmt::ImplBlock {
            for_type, methods, ..
        } = item
        {
            // The method definitions: register each impl-block method's
            // borrow signature
            // (the `self`/param references + the output derivation) so
            // the method-call cross-function freeze is order-independent.
            // The RECEIVER type (the impl's target — `impl for A`) is
            // part of the registry key: same-name methods on DIFFERENT
            // receivers no longer collide (the consumer matches on it
            // instead of find-first-match).
            let receiver_ty = checker.resolve_type(for_type).ok();
            for m in methods {
                let param_refs: Vec<(Symbol, bool, bool, Option<Symbol>)> = m
                    .params
                    .iter()
                    .map(|p| match p.ty.as_ref() {
                        Some(crate::ast::Type::Reference {
                            mutable, lifetime, ..
                        }) => (p.name, true, *mutable, *lifetime),
                        _ => (p.name, false, false, None),
                    })
                    .collect();
                let return_refs = crate::hir::polonius::count_return_refs_ast(&m.return_type);
                let sig = crate::hir::polonius::extract_borrow_signature(&param_refs, return_refs);
                let param_names: Vec<Symbol> = m.params.iter().map(|p| p.name).collect();
                let deriving =
                    derive_output_origins_ast(m.body.as_deref().unwrap_or(&[]), &param_names);
                let facts = crate::hir::polonius::signature_facts(&sig, &deriving);
                let input_positions: Vec<usize> = m
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        matches!(p.ty.as_ref(), Some(crate::ast::Type::Reference { .. }))
                    })
                    .map(|(i, _)| i)
                    .collect();
                checker
                    .signature_facts
                    .push((m.name, true, receiver_ty, input_positions, facts));
            }
        }
    }
}

/// Side-effect labels for a function, computed over the call graph
/// (pre-check, order-independent).  The isolate block check requires
/// `MUTABLE_GLOBAL` to be clear; `@pure` requires a wider forbidden set.
/// One graph, one fixpoint, one query — instead of a separate transitive
/// set per attribute.
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct EffectSet: u8 {
        /// Reads or writes a top-level `set mut` variable.
        const MUTABLE_GLOBAL = 1 << 0;
        /// Calls an `@io` function (external I/O).
        const IO = 1 << 1;
        /// Contains an `unsafe` block.
        const UNSAFE = 1 << 2;
        /// Contains a reachable `panic` (or calls an `@diverges` function).
        const PANIC = 1 << 3;
        /// Allocates (forbidden in interrupt handlers).
        const ALLOC = 1 << 4;
        /// Calls a comptime function (`f!()`).
        const COMPTIME = 1 << 5;
        /// Contains an explicit wrap-around operator (`+%`/`-%`/`*%`).
        /// Propagates up the call graph like the other effects: a caller
        /// of a wrap-using function inherits WRAP, so a loop body that
        /// calls `f(x)` (where `f` wraps) is recognized as wrap-semantics
        /// by the bit-vector routing (synthesis `use_bv` and verification
        /// `discharge_bv`).
        const WRAP = 1 << 6;
    }
}

/// Collect every EXPLICIT lifetime annotation (`&'a T` → `'a`) inside a
/// resolved type, recursively (nested references, tuples, arrays, slices,
/// function types, ADT args).  Feeds the region solver: the explicit
/// regions appearing in a function's parameter/return types form its
/// early-bound UniversalRegions (SYNTAX.md §Explicit Lifetime Parameters
/// — "verified by the borrow checker; mismatches cause compile errors").
fn collect_ref_lifetimes(ty: TypeId, ctx: &crate::hir::types::TypeContext) -> Vec<Symbol> {
    use crate::hir::types::TypeData;
    let mut out = Vec::new();
    match ctx.get(ty) {
        TypeData::Ref { lifetime, ty, .. } => {
            if let Some(l) = lifetime {
                out.push(*l);
            }
            out.extend(collect_ref_lifetimes(*ty, ctx));
        }
        TypeData::Tuple { elems } => {
            for &e in elems {
                out.extend(collect_ref_lifetimes(e, ctx));
            }
        }
        TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
            out.extend(collect_ref_lifetimes(*elem, ctx));
        }
        TypeData::Fn { params, ret } => {
            for &p in params {
                out.extend(collect_ref_lifetimes(p, ctx));
            }
            out.extend(collect_ref_lifetimes(*ret, ctx));
        }
        TypeData::Adt { args, .. } => {
            for &a in args {
                out.extend(collect_ref_lifetimes(a, ctx));
            }
        }
        _ => {}
    }
    out
}

/// Reachability in the GIVEN outlives graph: does `a` provably outlive
/// `b` (`'a: 'b`) via the transitive closure of `edges`?  Used by the
/// post-body verification — a BODY-collected requirement `(a, b)`
/// must be proved by the signature-level `where 'a: 'b` edges alone (the
/// requirement itself is not part of the given graph, so it cannot
/// self-prove).
fn region_reaches(a: Symbol, b: Symbol, edges: &[(Symbol, Symbol)]) -> bool {
    let mut graph: HashMap<Symbol, Vec<Symbol>> = HashMap::new();
    for (x, y) in edges {
        graph.entry(*x).or_default().push(*y);
    }
    let mut seen: HashSet<Symbol> = HashSet::new();
    let mut stack: Vec<Symbol> = vec![a];
    while let Some(node) = stack.pop() {
        if node == b {
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        if let Some(next) = graph.get(&node) {
            stack.extend(next.iter().copied());
        }
    }
    false
}

/// The region solver: given the collected explicit `'a: 'b` outlives
/// constraints and the parameter/return regions of a function signature,
/// return the set of RETURN regions that are NOT covered by any parameter
/// region — i.e. regions for which no parameter region provably outlives
/// them.  Computes the transitive closure of the outlives graph first
/// (`'a: 'b` and `'b: 'c` ⇒ `'a: 'c`), then checks each return region
/// against the parameter regions (rustc's "lifetime may not live long
/// enough" / E0623 family).  Parameter regions are the free/early-bound
/// regions the caller supplies; a return reference may only use one of
/// them (or something they outlive).
fn solve_region_outlives(
    param_regions: &HashSet<Symbol>,
    ret_regions: &[Symbol],
    constraints: &[(Symbol, Symbol)],
) -> Vec<Symbol> {
    // The outlives graph: `'a` → the set of regions it outlives.
    let mut graph: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();
    for (a, b) in constraints {
        graph.entry(*a).or_default().insert(*b);
    }
    // Transitive closure (worklist over the graph — the constraint set is
    // tiny: bounded by the number of explicit lifetimes in a signature).
    let mut closure: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();
    for &r in param_regions.iter().chain(ret_regions) {
        let mut reach: HashSet<Symbol> = HashSet::new();
        let mut stack: Vec<Symbol> = vec![r];
        while let Some(node) = stack.pop() {
            if !reach.insert(node) {
                continue;
            }
            if let Some(next) = graph.get(&node) {
                for &n in next {
                    stack.push(n);
                }
            }
        }
        closure.insert(r, reach);
    }
    // A return region is covered iff some parameter region reaches it.
    let mut uncovered = Vec::new();
    for r in ret_regions {
        let covered = param_regions
            .iter()
            .any(|p| closure.get(p).is_some_and(|reach| reach.contains(r)));
        if !covered {
            uncovered.push(*r);
        }
    }
    uncovered
}

/// Visitor that collects, for one function body: the set of free-function
/// callees (direct `f(...)` calls resolvable via `lookup_function`), the
/// set of method callees whose RECEIVER type is resolvable at AST scan
/// time (`self` — the impl's `for_type` — and explicitly-typed
/// parameters), and the DIRECT side-effect labels of the body itself
/// (mutable-global reads, `unsafe` blocks, `@io`/`@diverges`/comptime
/// calls).
struct BodyInfoCollector<'a, 'input> {
    checker: &'a TypeChecker<'a, 'input>,
    globals: &'a HashSet<Symbol>,
    callees: &'a mut HashSet<DefId>,
    /// Receiver variable name → its type's DefId (`self` → impl's
    /// `for_type`; explicitly-typed params → their type).  Method calls
    /// `r.foo()` on a receiver in this map record a method→method edge.
    receivers: &'a HashMap<Symbol, DefId>,
    /// Method callees with a resolvable receiver — the CALLEE METHOD's own
    /// DefId (resolved via `symbols.lookup_method_def_id`; the key space
    /// of `method_effect_of`).
    method_callees: &'a mut HashSet<DefId>,
    effects: &'a mut EffectSet,
}

impl<'ast, 'input: 'a, 'a> crate::ast::visit::Visitor<'ast, 'input>
    for BodyInfoCollector<'a, 'input>
{
    type Result = ();
    fn visit_expr(&mut self, expr: &'ast crate::ast::Expr<'input>) {
        match expr {
            crate::ast::Expr::Ident(name, _) => {
                if self.globals.contains(name) {
                    self.effects.insert(EffectSet::MUTABLE_GLOBAL);
                }
            }
            crate::ast::Expr::Call {
                callee, comptime, ..
            } => {
                // Free-function calls: `f(...)` — matching the isolate
                // check's own callee resolution (`lookup_function`).
                if let crate::ast::Expr::Ident(f, _) = *callee
                    && let Some(b) = self.checker.symbols.lookup_function(*f)
                {
                    self.callees.insert(b.def_id);
                    // Direct effect labels derived from the CALLEE's
                    // attributes: calling an `@io` function is itself I/O;
                    // calling an `@diverges` function reaches a panic.
                    if b.attributes.iter().any(|a| a.name.eq_str("io")) {
                        self.effects.insert(EffectSet::IO);
                    }
                    if b.attributes.iter().any(|a| a.name.eq_str("diverges")) {
                        self.effects.insert(EffectSet::PANIC);
                    }
                }
                // Method calls with a RESOLVABLE receiver type: `self.foo()`
                // or `x.bar()` where `x` is `self`/an explicitly-typed
                // parameter.  The receiver's type is known from the
                // signature at AST scan time — resolve to the callee
                // method's OWN DefId and record the method→method edge so
                // the fixpoint unions the callee's effects.
                else if let crate::ast::Expr::FieldAccess { base, field, .. } = *callee
                    && let crate::ast::Expr::Ident(r, _) = *base
                    && let Some(receiver_def) = self.receivers.get(r)
                    && let Some(method_def) = self
                        .checker
                        .symbols
                        .lookup_method_def_id(*receiver_def, *field)
                {
                    self.method_callees.insert(method_def);
                }
                if *comptime {
                    self.effects.insert(EffectSet::COMPTIME);
                }
            }
            crate::ast::Expr::UnsafeBlock { .. } => {
                self.effects.insert(EffectSet::UNSAFE);
            }
            crate::ast::Expr::BinaryOp { op, .. } => {
                // An explicit wrap-around operator (`+%`/`-%`/`*%`) marks
                // this function's effect label with WRAP — the label
                // propagates up the call graph (a caller inherits WRAP),
                // which the wrap-routing uses to recognize wrap-semantics
                // loops that reach a wrap through a function call.
                if matches!(
                    op,
                    crate::ast::BinOp::AddWrap
                        | crate::ast::BinOp::SubWrap
                        | crate::ast::BinOp::MulWrap
                ) {
                    self.effects.insert(EffectSet::WRAP);
                }
            }
            _ => {}
        }
        crate::ast::visit::walk_expr(self, expr)
    }
    fn visit_stmt(&mut self, stmt: &'ast crate::ast::Stmt<'input>) {
        if let crate::ast::Stmt::Unsafe { .. } = stmt {
            self.effects.insert(EffectSet::UNSAFE);
        }
        crate::ast::visit::walk_stmt(self, stmt)
    }
}

/// Pre-compute the per-function side-effect labels over the call graph: a
/// function's label is its DIRECT effects (mutable-global reads, `unsafe`
/// blocks, calls to `@io`/`@diverges` functions, comptime calls) unioned
/// with its transitive callees'.  Run BEFORE any body check
/// (order-independent — the callee may be defined after the caller);
/// the isolate and `@pure` checks consult this map.
pub(crate) fn collect_function_effects<'input: 'a, 'a>(
    checker: &mut TypeChecker<'a, 'input>,
    program: &crate::ast::Program<'input>,
) {
    // 1. Top-level mutable globals: `set mut x = ...` at module scope.
    let mut globals: HashSet<Symbol> = HashSet::new();
    for item in &program.items {
        if let crate::ast::Stmt::VariableDef {
            kind: crate::ast::VariableKind::Set,
            mutable: true,
            name: Some(n),
            ..
        } = item
        {
            globals.insert(*n);
        }
    }

    // 2. Direct effects + call edges (callee → its callers).
    let mut callers_of: HashMap<DefId, Vec<DefId>> = HashMap::new();
    let mut direct: HashMap<DefId, EffectSet> = HashMap::new();
    for item in &program.items {
        if let crate::ast::Stmt::FunctionDef {
            name,
            body,
            is_comptime: false,
            ..
        } = item
        {
            // The comptime functions are compile-time — they never exist
            // at runtime, so they cannot read a runtime mutable global.
            let Some(caller) = checker.symbols.lookup_function(*name).map(|b| b.def_id) else {
                continue;
            };
            let mut callees: HashSet<DefId> = HashSet::new();
            let mut effects = EffectSet::empty();
            // Free functions have no `self`/typed-parameter receiver
            // types to resolve — pass empty maps (no method→method edges).
            let receivers: HashMap<Symbol, DefId> = HashMap::new();
            let mut method_callees: HashSet<DefId> = HashSet::new();
            let mut v = BodyInfoCollector {
                checker,
                globals: &globals,
                callees: &mut callees,
                receivers: &receivers,
                method_callees: &mut method_callees,
                effects: &mut effects,
            };
            for s in body.as_deref().unwrap_or(&[]) {
                crate::ast::visit::walk_stmt(&mut v, s);
            }
            direct.insert(caller, effects);
            for c in callees {
                callers_of.entry(c).or_default().push(caller);
            }
        }
    }

    // 3. Propagate effects UP the call graph (a caller inherits its
    // callees' effects — A calls B, B touches a mutable global ⇒ A's
    // label includes MUTABLE_GLOBAL) to a fixpoint.
    let mut effect_of = direct.clone();
    let mut stack: Vec<DefId> = direct
        .iter()
        .filter(|(_, e)| !e.is_empty())
        .map(|(k, _)| *k)
        .collect();
    while let Some(f) = stack.pop() {
        let feffects = effect_of[&f];
        if let Some(callers) = callers_of.get(&f) {
            for &c in callers {
                let before = effect_of.get(&c).copied().unwrap_or(EffectSet::empty());
                let after = before | feffects;
                if after != before {
                    effect_of.insert(c, after);
                    stack.push(c);
                }
            }
        }
    }

    // 4. Method effects: keyed by the method's OWN DefId (the "assoc
    // item" identity — allocated in the resolver and registered under
    // (receiver type DefId, name) in `symbols.method_def_ids`, mirroring
    // rustc's `AssocItem.def_id`).  A method's label is its DIRECT effects
    // (mutable-global reads, unsafe, @io/@diverges/comptime calls —
    // collected by the same BodyInfoCollector) unioned with its
    // free-function callees' already-propagated effects (the fixpoint
    // above is complete by now).  Method→method call edges are resolved
    // when the receiver type is known AT AST scan time — `self` (the
    // impl's `for_type`) and explicitly-typed parameters — and stored in
    // `method_edges` (DefId → callee DefIds) for the multi-hop fixpoint
    // below.
    let mut method_effect_of: HashMap<DefId, EffectSet> = HashMap::new();
    let mut method_edges: HashMap<DefId, HashSet<DefId>> = HashMap::new();
    for item in &program.items {
        if let crate::ast::Stmt::ImplBlock {
            for_type, methods, ..
        } = item
        {
            // The receiver's DefId: `resolve_type` may fail (forward
            // references not yet resolvable) — skip that impl.
            let Ok(receiver_ty) = checker.resolve_type(for_type) else {
                continue;
            };
            let Some(receiver_def) = checker.ctx.get_def_id_for_type(receiver_ty) else {
                continue;
            };
            for m in methods {
                // The method's OWN DefId — the resolver allocated and
                // registered it under (receiver DefId, name); look it back
                // up (fallback: allocate fresh, defensive — resolver runs
                // first).
                let method_def = checker
                    .symbols
                    .lookup_method_def_id(receiver_def, m.name)
                    .unwrap_or_else(crate::hir::types::alloc_def_id);
                // Receiver map for method→method edges: `self` is always
                // the impl's receiver (the parser synthesizes its type as
                // `&Self`/`Self`); explicitly-typed parameters resolve to
                // their declared type's DefId (pointee for `&T`).
                let mut receivers: HashMap<Symbol, DefId> = HashMap::new();
                receivers.insert(Symbol::intern("self"), receiver_def);
                for p in &m.params {
                    if p.name.eq_str("self") {
                        continue;
                    }
                    if let Some(ty) = &p.ty
                        && let Ok(ty_id) = checker.resolve_type(ty)
                    {
                        let pointee = match checker.ctx.get(ty_id) {
                            TypeData::Ref { ty: inner, .. } => *inner,
                            _ => ty_id,
                        };
                        if let Some(def_id) = checker.ctx.get_def_id_for_type(pointee) {
                            receivers.insert(p.name, def_id);
                        }
                    }
                }
                let mut callees: HashSet<DefId> = HashSet::new();
                let mut method_callees: HashSet<DefId> = HashSet::new();
                let mut effects = EffectSet::empty();
                let mut v = BodyInfoCollector {
                    checker,
                    globals: &globals,
                    callees: &mut callees,
                    receivers: &receivers,
                    method_callees: &mut method_callees,
                    effects: &mut effects,
                };
                for s in m.body.as_deref().unwrap_or(&[]) {
                    crate::ast::visit::walk_stmt(&mut v, s);
                }
                // Union the propagated effects of the method's
                // free-function callees into its own label.
                for c in &callees {
                    effects |= effect_of.get(c).copied().unwrap_or(EffectSet::empty());
                }
                method_effect_of.insert(method_def, effects);
                // Record the method→method edges (caller DefId → callee
                // DefIds) for the multi-hop fixpoint below.
                method_edges.insert(method_def, method_callees);
            }
        }
    }

    // 5. Multi-hop propagation over the method→method edges: A calls
    // `self.B`, B calls `self.C`, C reads a mutable global ⇒ B's label
    // gains MUTABLE_GLOBAL, and so does A's.  Reverse edges
    // (callee → its callers) + worklist, mirroring the free-function
    // fixpoint above — so a method chain is closed transitively.
    let mut method_callers_of: HashMap<DefId, Vec<DefId>> = HashMap::new();
    for (caller, callees) in &method_edges {
        for &callee in callees {
            method_callers_of.entry(callee).or_default().push(*caller);
        }
    }
    let mut stack: Vec<DefId> = method_effect_of
        .iter()
        .filter(|(_, e)| !e.is_empty())
        .map(|(k, _)| *k)
        .collect();
    while let Some(f) = stack.pop() {
        let feffects = method_effect_of[&f];
        if let Some(callers) = method_callers_of.get(&f) {
            for &c in callers {
                let before = method_effect_of
                    .get(&c)
                    .copied()
                    .unwrap_or(EffectSet::empty());
                let after = before | feffects;
                if after != before {
                    method_effect_of.insert(c, after);
                    stack.push(c);
                }
            }
        }
    }
    checker.effect_of = effect_of;
    checker.method_effect_of = method_effect_of;
}

/// The AST-level return-place root (the counterpart of the HIR
/// `place_root` in `polonius.rs`).
fn place_root_ast(e: &crate::ast::Expr) -> Option<Symbol> {
    match e {
        crate::ast::Expr::Ident(name, _) => Some(*name),
        crate::ast::Expr::FieldAccess { base, .. } => place_root_ast(base),
        crate::ast::Expr::Index { base, .. } => place_root_ast(base),
        crate::ast::Expr::UnaryOp {
            op:
                crate::ast::UnaryOp::Deref
                | crate::ast::UnaryOp::Ref
                | crate::ast::UnaryOp::RefMut
                | crate::ast::UnaryOp::Ro,
            expr,
            ..
        } => place_root_ast(expr),
        _ => None,
    }
}

/// The AST-level output-derivation analysis (the counterpart of the HIR
/// `derive_output_origins` in `polonius.rs`).
fn derive_output_origins_ast(stmts: &[crate::ast::Stmt], params: &[Symbol]) -> Vec<usize> {
    let mut deriving = Vec::new();
    collect_returns_ast(stmts, params, &mut deriving);
    deriving
}

/// AST-level counterpart of the HIR `collect_returns`: recurse into nested
/// control flow (If/While/WhileLet/For/Loop) so pre-registered signatures
/// are order-independent for nested returns too (the
/// AST-level analysis missed nested returns, under-approximating the
/// cross-function borrow freeze — the unsound direction).
fn collect_returns_ast(stmts: &[crate::ast::Stmt], params: &[Symbol], out: &mut Vec<usize>) {
    for stmt in stmts {
        match stmt {
            crate::ast::Stmt::Return { value: Some(v), .. } => {
                if let Some(root) = place_root_ast(v) {
                    if let Some(i) = params.iter().position(|p| *p == root) {
                        if !out.contains(&i) {
                            out.push(i);
                        }
                    }
                }
            }
            crate::ast::Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_returns_ast(then_branch, params, out);
                if let Some(eb) = else_branch {
                    collect_returns_ast(eb, params, out);
                }
            }
            // `if let` — mirrors the HIR-level `collect_returns` (the
            // missing arm under-approximated the A(ρ) derivations in the
            // pre-registration window, when only the AST-level facts are
            // available).
            crate::ast::Stmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_returns_ast(then_branch, params, out);
                if let Some(eb) = else_branch {
                    collect_returns_ast(eb, params, out);
                }
            }
            crate::ast::Stmt::While { body, .. }
            | crate::ast::Stmt::WhileLet { body, .. }
            | crate::ast::Stmt::For { body, .. }
            | crate::ast::Stmt::Loop { body, .. } => {
                collect_returns_ast(body, params, out);
            }
            _ => {}
        }
    }
}
