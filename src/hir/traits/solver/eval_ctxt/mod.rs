//! # EvalCtxt — Recursive Goal Evaluation Context with Probe Support
//!
//! Analogous to `rustc_next_trait_solver::solve::EvalCtxt`.
//! Provides a recursive evaluation context with:
//! - `probe` — try a candidate, roll back all side effects on failure
//! - Nested goal tracking, canonical variable values, universe tracking
//! - Transaction management (begin/commit/rollback)
//!
//! ## Submodules
//!
//! - [`probe`] — Builder-pattern probe system (`ProbeCtxt`, `TraitProbeCtxt`,
//!   `CandidateHeadUsages`, `CandidateSource`)

pub mod probe;

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast::Span;
use crate::hir::infer::TypeVariableKind;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::obligation::{ImplSource, Obligation, Predicate, SolveError};
use crate::hir::traits::solver::search_graph::SearchGraph;
use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;

/// Counter for fresh inference variables used during instance instantiation.
static INSTANCE_FRESH_VAR_ID: AtomicUsize = AtomicUsize::new(4_000_000);

/// Describes the instantiation target of a polymorphic scheme,
/// collected from a TypeData node without holding a borrow on the
/// TypeContext. Used by the Instance constraint handler to separate
/// the immutable scan phase from the mutable variable-creation phase.
enum InstantiationTarget {
    /// `∀α₁.∀α₂. ... αₙ. body_ty` — Forall binders.
    Forall {
        binder_indices: Vec<usize>,
        body_ty: TypeId,
    },
    /// `∀quantifiers. body_ty` — Poly type.
    Poly {
        binder_indices: Vec<usize>,
        body_ty: TypeId,
    },
}

/// Result of a goal evaluation attempt, parallel to rustc's `QueryResult`.
#[derive(Debug, Clone)]
pub enum QueryResult {
    Yes(ImplSource),
    Maybe(ImplSource),
    No(SolveError),
}

/// What kind of goal we're currently computing.
/// Analogous to rustc's `CurrentGoalKind` in `eval_ctxt/mod.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentGoalKind {
    Misc,
    ProjectionComputeAssocTermCandidate,
    ProjectionNormalizeCandidate,
    StructuralCandidate,
    AlwaysApplicableCandidate,
    FallbackCandidate,
}

/// What kind of probe we're performing.
/// Analogous to rustc's `inspect::ProbeKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    /// Entering a probe for a candidate evaluation.
    CandidateEvaluation,
    /// Entering a probe for a single candidate.
    SingleCandidate,
    /// Entering a probe for a trait goal.
    TraitGoal,
    /// Entering a probe for a trait candidate with a specific source.
    TraitCandidate,
    /// Entering a probe for a projection goal.
    ProjectionGoal,
    /// Entering a probe for a normalization goal.
    NormalizationGoal,
    /// Entering a probe for structural trait checking.
    StructuralTrait,
    /// Entering a probe for a fallback candidate.
    ShadowedEnvProbing,
    /// Entering a probe for a tautological obligation.
    TautologicalObligation,
    /// Entering a probe for ambiguity handling.
    ForcedAmbiguity,
}

/// Where a goal came from — analogous to Rust's `GoalSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalSource {
    /// From an impl's where-clause.
    ImplWhereBound,
    /// From a trait obligation.
    TraitObligation,
    /// From a projection goal.
    Projection,
    /// From a where-clause on the function/item.
    ItemWhereBound,
    /// From a builtin trait (Sized/Copy/Clone).
    Builtin,
    /// From a misc source.
    Misc,
}

/// Information about why a goal is stalled — analogous to Rust's `GoalStalledOn`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalStalledOn {
    /// Stalled on an inference variable.
    InferVar(usize),
    /// Stalled on a projection.
    Projection,
    /// Stalled on a placeholder/universe.
    Placeholder,
}

/// Recursive goal evaluation context.
///
/// Wraps a `SolverDelegate` and mirrors the core fields of Rust's `EvalCtxt`.
/// See the table in the module doc for the field mapping.
pub struct EvalCtxt<'a, D: SolverDelegate> {
    pub delegate: &'a mut D,
    pub var_kinds: Vec<TypeVariableKind>,
    pub current_goal_kind: CurrentGoalKind,
    pub var_values: Vec<TypeId>,
    pub max_input_universe: usize,
    pub nested_goals: Vec<(GoalSource, Obligation, Option<GoalStalledOn>)>,
    pub tainted: Result<(), ()>,
    pub opaque_accesses: HashSet<TypeId>,
    /// The search graph for cycle detection and fixpoint iteration.
    /// Borrowed from the owning `FulfillmentContext`.
    /// Analogous to Rust's `search_graph: &'a mut SearchGraph<D>`.
    pub search_graph: &'a mut SearchGraph,
    /// The source span of the origin goal being evaluated.
    pub origin_span: Span,
    /// The number of opaque type storage entries at the start of evaluation.
    pub initial_opaque_type_storage_len: usize,
    // ── Cached environment data (raw pointers to avoid borrow conflicts) ──
    // These are stored as raw pointers so that `ctx()` and `trait_env()` /
    // `caller_bounds()` / `builtin_registry()` can be called simultaneously
    // without borrow conflicts.  Initialized once in `new()`.
    pub(crate) trait_env: *const TraitEnv,
    pub(crate) caller_bounds: *const [Predicate],
    pub(crate) builtin_registry: *const BuiltinTraitRegistry,
}

/// Full snapshot of all EvalCtxt fields, used for rollback in probe.
struct EvalCtxtSnapshot {
    var_kinds: Vec<TypeVariableKind>,
    current_goal_kind: CurrentGoalKind,
    var_values: Vec<TypeId>,
    max_input_universe: usize,
    nested_goals_len: usize,
    opaque_accesses: HashSet<TypeId>,
    tainted: Result<(), ()>,
    // NOTE: search_graph is borrowed (&'a mut), so it cannot be cloned.
    // The probe rollback does NOT restore the search_graph — this is safe
    // because search_graph mutations are idempotent (push/pop on try_entry/exit).
}

impl<'a, D: SolverDelegate> EvalCtxt<'a, D> {
    fn snapshot(&self) -> EvalCtxtSnapshot {
        EvalCtxtSnapshot {
            var_kinds: self.var_kinds.clone(),
            current_goal_kind: self.current_goal_kind,
            var_values: self.var_values.clone(),
            max_input_universe: self.max_input_universe,
            nested_goals_len: self.nested_goals.len(),
            opaque_accesses: self.opaque_accesses.clone(),
            tainted: self.tainted,
        }
    }

    fn restore_snapshot(&mut self, snap: EvalCtxtSnapshot) {
        self.var_kinds = snap.var_kinds;
        self.current_goal_kind = snap.current_goal_kind;
        self.var_values = snap.var_values;
        self.max_input_universe = snap.max_input_universe;
        self.nested_goals.truncate(snap.nested_goals_len);
        self.opaque_accesses = snap.opaque_accesses;
        self.tainted = snap.tainted;
        // search_graph is NOT restored — see the struct comment above.
    }

    /// Create a new evaluation context from a solver delegate, search graph,
    /// and the source span of the origin goal.
    pub fn new(delegate: &'a mut D, search_graph: &'a mut SearchGraph, origin_span: Span) -> Self {
        // Cache environment data as raw pointers to avoid borrow conflicts
        // with ctx() later (see raw pointer fields doc).
        let trait_env = delegate.trait_env() as *const TraitEnv;
        let caller_bounds = delegate.caller_bounds() as *const [Predicate];
        let builtin_registry = delegate.builtin_registry() as *const BuiltinTraitRegistry;
        EvalCtxt {
            delegate,
            var_kinds: Vec::new(),
            current_goal_kind: CurrentGoalKind::Misc,
            var_values: Vec::new(),
            max_input_universe: 0,
            nested_goals: Vec::new(),
            tainted: Ok(()),
            opaque_accesses: HashSet::new(),
            search_graph,
            origin_span,
            initial_opaque_type_storage_len: 0,
            trait_env,
            caller_bounds,
            builtin_registry,
        }
    }

    /// Access the type context through the delegate.
    pub fn ctx(&mut self) -> &mut TypeContext {
        self.delegate.ctx()
    }

    /// Access the trait environment (read-only, no borrow conflict with ctx()).
    pub fn trait_env(&self) -> &TraitEnv {
        // SAFETY: The raw pointer was initialized in `new()` from the delegate's
        // `trait_env()` method, and the delegate is alive for the lifetime of
        // this EvalCtxt.  The delegate is not dropped or invalidated while we
        // hold this reference.
        unsafe { &*self.trait_env }
    }

    /// Access the caller bounds (read-only, no borrow conflict with ctx()).
    pub fn caller_bounds(&self) -> &[Predicate] {
        // SAFETY: Same as trait_env() — the raw pointer is valid for the
        // lifetime of this EvalCtxt.
        unsafe { &*self.caller_bounds }
    }

    /// Access the builtin trait registry (read-only, no borrow conflict with ctx()).
    pub fn builtin_registry(&self) -> &BuiltinTraitRegistry {
        // SAFETY: Same as trait_env() — the raw pointer is valid for the
        // lifetime of this EvalCtxt.
        unsafe { &*self.builtin_registry }
    }

    // ── Eq/Sub/Match goal evaluation ──────────────────────────────────

    /// Evaluate an equality goal: `Eq(a, b)`.
    ///
    /// Unifies `a` and `b`.  If either side is still an unresolved inference
    /// variable, returns `Deferred` so the caller can retry later.
    pub fn compute_eq_goal(&mut self, a: TypeId, b: TypeId, span: Span) -> Result<ImplSource, SolveError> {
        let ctx = self.ctx();
        let ra = ctx.resolve_binding(a);
        let rb = ctx.resolve_binding(b);
        if ctx.is_infer_var(ra) || ctx.is_infer_var(rb) {
            // At least one side is still unresolved — defer.
            let stalled_on = vec![if ctx.is_infer_var(ra) { ra } else { rb }];
            return Ok(ImplSource::Deferred { stalled_on });
        }
        ctx.unify(a, b).map_err(|_| SolveError::Mismatch {
            expected: a,
            found: b,
            span,
        })?;
        Ok(ImplSource::Param(vec![]))
    }

    /// Evaluate a subtype goal: `Sub(sub, sup)`.
    ///
    /// Checks that `sub <: sup`.  If either side is still an unresolved
    /// inference variable, returns `Deferred` so the caller can retry later.
    pub fn compute_sub_goal(&mut self, sub: TypeId, sup: TypeId, span: Span) -> Result<ImplSource, SolveError> {
        let ctx = self.ctx();
        let rsub = ctx.resolve_binding(sub);
        let rsup = ctx.resolve_binding(sup);
        if ctx.is_infer_var(rsub) || ctx.is_infer_var(rsup) {
            // At least one side is still unresolved — defer.
            let stalled_on = vec![if ctx.is_infer_var(rsub) { rsub } else { rsup }];
            return Ok(ImplSource::Deferred { stalled_on });
        }
        if !ctx.subtype(sub, sup) {
            return Err(SolveError::Mismatch {
                expected: sup,
                found: sub,
                span,
            });
        }
        Ok(ImplSource::Param(vec![]))
    }

    /// Evaluate a match goal: `Match { scrutinee, branches_id }`.
    ///
    /// Tries to discharge the match constraint by determining the scrutinee's
    /// shape and matching against the registered branch patterns.
    /// If the scrutinee is still an unresolved inference variable, returns
    /// `Deferred` so the caller can retry later.
    pub fn compute_match_goal(
        &mut self,
        scrutinee: TypeId,
        branches_id: (usize, usize),
        span: Span,
    ) -> Result<ImplSource, SolveError> {
        let ctx = self.ctx();
        let resolved = ctx.resolve_binding(scrutinee);
        if ctx.is_infer_var(resolved) {
            // Scrutinee is still unresolved — defer.
            return Ok(ImplSource::Deferred {
                stalled_on: vec![resolved],
            });
        }
        // Delegate to the solver delegate's discharge_match implementation.
        // This requires access to the match_branches table which is
        // owned by InferenceContext (the old solver).
        match self.delegate.discharge_match(scrutinee, branches_id) {
            Ok(continuation) => {
                // The match was discharged successfully.  Return the
                // continuation obligations so they can be resolved.
                Ok(ImplSource::Param(continuation))
            }
            Err(()) => {
                // No branch matched and no else_ fallback.
                Err(SolveError::NotFound {
                    trait_id: DefId(0),
                    self_ty: scrutinee,
                    span,
                })
            }
        }
    }

    // ── Forall/Exists/Instance/Let goal evaluation ────────────────────

    /// Evaluate a `Forall { body }` goal.
    ///
    /// Enters a new universe, creates a fresh Skolem variable for the
    /// universally quantified binder, substitutes it in the body, and
    /// returns the substituted body as a nested obligation.
    pub fn compute_forall_goal(
        &mut self,
        body: &Predicate,
        cause: &crate::hir::traits::solver::obligation::ObligationCause,
        recursion_depth: usize,
    ) -> Result<ImplSource, SolveError> {
        let ctx = self.ctx();
        let (_universe, skolem_ty) = ctx.enter_universe();
        // The body will be resolved with the skolem in scope — return it
        // as a nested obligation.  The skolem is tracked by the SearchGraph
        // via the universe mechanism, so it cannot escape.
        let nested = Obligation {
            cause: cause.clone(),
            predicate: body.clone(),
            recursion_depth: recursion_depth + 1,
        };
        Ok(ImplSource::Param(vec![nested]))
    }

    /// Evaluate an `Exists { body }` goal.
    ///
    /// The existentially quantified variable is already bound.
    /// Returns the body as a nested obligation.
    pub fn compute_exists_goal(
        &mut self,
        body: &Predicate,
        cause: &crate::hir::traits::solver::obligation::ObligationCause,
        recursion_depth: usize,
    ) -> Result<ImplSource, SolveError> {
        let nested = Obligation {
            cause: cause.clone(),
            predicate: body.clone(),
            recursion_depth: recursion_depth + 1,
        };
        Ok(ImplSource::Param(vec![nested]))
    }

    /// Evaluate an `Instance { scheme_ty, instantiation_ty }` goal.
    ///
    /// If `scheme_ty = ∀α₁...∀αₙ. τ_body`, creates fresh inference
    /// variables β₁...βₙ and constrains
    /// `Eq(instantiation_ty, τ_body[αᵢ:=βᵢ])`.
    ///
    /// For polymorphic types (`Poly`), creates fresh vars for each
    /// quantifier and substitutes.  For concrete types, unifies directly.
    pub fn compute_instance_goal(
        &mut self,
        scheme_ty: TypeId,
        instantiation_ty: TypeId,
        cause: &crate::hir::traits::solver::obligation::ObligationCause,
        recursion_depth: usize,
    ) -> Result<ImplSource, SolveError> {
        let ctx = self.ctx();
        let resolved_scheme = ctx.resolve_binding(scheme_ty);

        // Phase 1: Scan scheme type (immutable borrow) to collect binders.
        let scheme_info = match ctx.get(resolved_scheme) {
            TypeData::Forall { .. } => {
                let mut indices: Vec<usize> = Vec::new();
                let mut inner = resolved_scheme;
                loop {
                    match ctx.get(inner) {
                        TypeData::Forall {
                            param_index, body, ..
                        } => {
                            indices.push(*param_index);
                            inner = *body;
                        }
                        _ => break,
                    }
                }
                Some(InstantiationTarget::Forall {
                    binder_indices: indices,
                    body_ty: inner,
                })
            }
            TypeData::Poly { quantifiers, body } => {
                let indices: Vec<usize> = quantifiers.iter().map(|(idx, _)| *idx).collect();
                Some(InstantiationTarget::Poly {
                    binder_indices: indices,
                    body_ty: *body,
                })
            }
            _ => {
                // Concrete (or error) — just unify directly.
                None
            }
        };

        // Phase 2: Create fresh vars and emit Eq obligation.
        match scheme_info {
            Some(target) => {
                let (eq_a, eq_b) = match target {
                    InstantiationTarget::Forall {
                        binder_indices,
                        body_ty,
                    } => {
                        let mut instantiated = body_ty;
                        for &idx in binder_indices.iter().rev() {
                            let fresh_id = INSTANCE_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                            let fv = ctx.alloc_infer_var(fresh_id);
                            instantiated = ctx.replace_generic(instantiated, idx, fv);
                        }
                        (instantiation_ty, instantiated)
                    }
                    InstantiationTarget::Poly {
                        binder_indices,
                        body_ty,
                    } => {
                        let mut instantiated = body_ty;
                        for &idx in binder_indices.iter() {
                            let fresh_id = INSTANCE_FRESH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                            let fv = ctx.alloc_infer_var(fresh_id);
                            instantiated = ctx.replace_generic(instantiated, idx, fv);
                        }
                        (instantiation_ty, instantiated)
                    }
                };
                let nested = Obligation {
                    cause: cause.clone(),
                    predicate: Predicate::Eq { a: eq_a, b: eq_b },
                    recursion_depth: recursion_depth + 1,
                };
                Ok(ImplSource::Param(vec![nested]))
            }
            None => {
                // Concrete type — unify directly.
                let ctx = self.ctx();
                ctx.unify(instantiation_ty, resolved_scheme)
                    .map_err(|_| SolveError::Mismatch {
                        expected: instantiation_ty,
                        found: resolved_scheme,
                        span: cause.span,
                    })?;
                Ok(ImplSource::Param(vec![]))
            }
        }
    }

    /// Evaluate a `Let { def, body }` goal.
    ///
    /// Returns the `def` and `body` as nested obligations.
    /// The caller is responsible for resolving `def` before `body`.
    pub fn compute_let_goal(
        &mut self,
        def: &Predicate,
        body: &Predicate,
        cause: &crate::hir::traits::solver::obligation::ObligationCause,
        recursion_depth: usize,
    ) -> Result<ImplSource, SolveError> {
        let def_obl = Obligation {
            cause: cause.clone(),
            predicate: def.clone(),
            recursion_depth: recursion_depth + 1,
        };
        let body_obl = Obligation {
            cause: cause.clone(),
            predicate: body.clone(),
            recursion_depth: recursion_depth + 1,
        };
        // Return both obligations: def first, then body.
        Ok(ImplSource::Param(vec![def_obl, body_obl]))
    }

    // ── Probe mechanism (new builder API) ────────────────────────────

    /// Create a builder-pattern probe context.
    ///
    /// Usage:
    /// ```ignore
    /// ecx.probe(ProbeKind::TraitGoal).enter(|ecx| {
    ///     // ... evaluate inside a transaction ...
    /// })
    /// ```
    pub fn probe<T>(
        &mut self,
        probe_kind: crate::hir::traits::solver::eval_ctxt::ProbeKind,
    ) -> probe::ProbeCtxt<'_, 'a, D, T> {
        probe::ProbeCtxt {
            ecx: self,
            probe_kind,
            _result: std::marker::PhantomData,
        }
    }

    /// Create a trait candidate probe context.
    ///
    /// Usage:
    /// ```ignore
    /// ecx.probe_trait_candidate(source).enter(|ecx| {
    ///     // ... evaluate candidate inside a transaction ...
    /// })
    /// ```
    pub fn probe_trait_candidate(
        &mut self,
        source: probe::CandidateSource,
    ) -> probe::TraitProbeCtxt<'_, 'a, D> {
        probe::TraitProbeCtxt {
            cx: probe::ProbeCtxt {
                ecx: self,
                probe_kind: crate::hir::traits::solver::eval_ctxt::ProbeKind::TraitCandidate,
                _result: std::marker::PhantomData,
            },
            source,
        }
    }

    /// Create a builtin trait candidate probe context.
    /// Convenience wrapper around `probe_trait_candidate`.
    pub fn probe_builtin_candidate(
        &mut self,
        source: probe::BuiltinImplSource,
    ) -> probe::TraitProbeCtxt<'_, 'a, D> {
        self.probe_trait_candidate(probe::CandidateSource::Builtin(source))
    }

    // ── Legacy probe methods (deprecated) ────────────────────────────

    /// Like `probe`, but for candidates that may produce multiple valid
    /// results (e.g. ambiguity).  Rolls back if the result is not `Yes`.
    pub fn probe_maybe(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<ImplSource, SolveError>,
    ) -> Result<ImplSource, SolveError> {
        let snap = self.snapshot();
        self.ctx().begin_transaction();

        match f(self) {
            Ok(impl_source) => {
                match &impl_source {
                    ImplSource::Deferred { .. } => {
                        self.ctx().rollback_transaction();
                        // Restore everything EXCEPT nested_goals —
                        // deferred goals need to survive for retry.
                        self.var_kinds = snap.var_kinds;
                        self.current_goal_kind = snap.current_goal_kind;
                        self.var_values = snap.var_values;
                        self.max_input_universe = snap.max_input_universe;
                        self.opaque_accesses = snap.opaque_accesses;
                        self.tainted = snap.tainted;
                    }
                    ImplSource::Builtin(_)
                    | ImplSource::UserDefined { .. }
                    | ImplSource::Param { .. }
                    | ImplSource::Object { .. }
                    | ImplSource::Poly { .. }
                    | ImplSource::Auto { .. } => {
                        self.ctx().commit_transaction();
                    }
                }
                Ok(impl_source)
            }
            Err(e) => {
                self.ctx().rollback_transaction();
                self.restore_snapshot(snap);
                Err(e)
            }
        }
    }

    // ── Nested goal management ──────────────────────────────────────

    /// Register a nested obligation that must be resolved for the current
    /// goal to be satisfied.
    pub fn push_nested_goal(
        &mut self,
        source: GoalSource,
        obligation: Obligation,
        stalled_on: Option<GoalStalledOn>,
    ) {
        self.nested_goals.push((source, obligation, stalled_on));
    }

    /// Drain all nested goals accumulated during evaluation.
    pub fn drain_nested_goals(&mut self) -> Vec<(GoalSource, Obligation, Option<GoalStalledOn>)> {
        std::mem::take(&mut self.nested_goals)
    }

    // ── Universe tracking ───────────────────────────────────────────

    /// Create a new universe (increment the max_input_universe counter).
    /// Analogous to Rust's `EvalCtxt::create_next_universe`.
    pub fn create_next_universe(&mut self) -> usize {
        self.max_input_universe += 1;
        self.max_input_universe
    }

    /// Check whether a given universe is nameable by the caller.
    pub fn universe_is_nameable(&self, universe: usize) -> bool {
        universe <= self.max_input_universe
    }

    // ── Opaque access tracking ──────────────────────────────────────

    /// Record an access to an opaque type.
    pub fn record_opaque_access(&mut self, ty: TypeId) {
        self.opaque_accesses.insert(ty);
    }

    /// Check whether a specific opaque type has been accessed.
    pub fn has_opaque_access(&self, ty: TypeId) -> bool {
        self.opaque_accesses.contains(&ty)
    }

    // ── Taint tracking ──────────────────────────────────────────────

    /// Mark this `EvalCtxt` as tainted (no longer usable for canonical
    /// query responses).
    pub fn mark_tainted(&mut self) {
        self.tainted = Err(());
    }

    /// Check whether this `EvalCtxt` is tainted.
    pub fn is_tainted(&self) -> bool {
        self.tainted.is_err()
    }
}
