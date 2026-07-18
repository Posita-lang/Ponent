//! # SolverDelegate — Trait Solver Abstraction Layer
//!
//! Decouples the generic trait-solving logic from the concrete compiler
//! types (`TypeContext`, `TraitEnv`, `SymbolTable`, …), following the
//! same pattern as `rustc_next_trait_solver::delegate::SolverDelegate`.
//!
//! ## Design
//!
//! The solver (`FulfillmentContext`, `evaluate_goal`, `SelectionContext`)
//! previously depended directly on concrete types like `&mut TypeContext`
//! and `&TraitEnv`.  This made it impossible to:
//! - Unit‑test the solver with a mock environment
//! - Swap out the internals without touching every solver method
//!
//! `SolverDelegate` bundles all of those dependencies into a single trait.
//! Any struct that implements `SolverDelegate` can be used as the solver's
//! "environment", and the solver code accesses everything through `self.delegate.ctx()`,
//! `self.delegate.trait_env()`, etc.
//!
//! ## Usage
//!
//! ```ignore
//! fn my_solver_pass(delegate: &mut impl SolverDelegate) {
//!     let ctx = delegate.ctx();
//!     let trait_env = delegate.trait_env();
//!     // …
//! }
//! ```

use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::builtins::{BuiltinTrait, BuiltinTraitRegistry};
use crate::hir::traits::solver::obligation::{ImplSource, Obligation, Predicate, SolveError};
use crate::hir::traits::solver::project::ProjectionCache;
use crate::hir::types::{DefId, TypeContext, TypeId};

/// The trait solver's environment — abstracts over the concrete compiler
/// types so that the solver logic can be written generically.
///
/// Implementors are expected to provide shared (read‑only) access to the
/// type context, trait environment, symbol table, and so on, plus the
/// core selection method that drives candidate assembly → winnowing →
/// confirmation.
pub trait SolverDelegate {
    /// Mutable reference to the type context (arena, bindings, …).
    fn ctx(&mut self) -> &mut TypeContext;

    /// Read‑only reference to the trait environment (registered impls).
    fn trait_env(&self) -> &TraitEnv;

    /// Read‑only reference to the symbol table.
    fn symbols(&self) -> &SymbolTable;

    /// Built‑in trait registry (Sized, Copy, Clone, …).
    fn builtin_registry(&self) -> &BuiltinTraitRegistry;

    /// Projection cache for associated‑type normalization.
    fn proj_cache(&self) -> &ProjectionCache;

    /// Caller‑provided bounds (from where‑clauses in scope).
    fn caller_bounds(&self) -> &[Predicate];

    // ── Core solver operations ──────────────────────────────────────

    /// Select a candidate for the given obligation by delegating to the
    /// `GoalKind`-based assembly engine (see `assembly::assemble_and_evaluate_candidates`).
    ///
    /// The default implementation routes through the `GoalKind` trait,
    /// so implementors only need to provide the environment accessors
    /// (`ctx()`, `trait_env()`, etc.) and the projection handlers.
    /// Override only if custom selection logic is required.
    fn select(&mut self, obligation: &Obligation) -> Result<ImplSource, SolveError>
    where
        Self: Sized,
    {
        // Default implementation: use the GoalKind-based assembly engine.
        // This requires a mutable EvalCtxt, which we create inline.
        let mut search_graph = crate::hir::traits::solver::search_graph::SearchGraph::new();
        let span = obligation.cause.span;
        let mut ecx =
            crate::hir::traits::solver::eval_ctxt::EvalCtxt::new(self, &mut search_graph, span);
        crate::hir::traits::solver::assembly::assemble_and_evaluate_candidates(&mut ecx, obligation)
    }

    /// Resolve the self‑ty and arguments of an obligation through bindings.
    fn resolve_obligation(&self, obligation: &Obligation) -> super::select::ResolvedObligation;

    // ── Trait classification helpers ─────────────────────────────────

    /// Whether the trait identified by `def_id` is coinductive
    /// (auto‑traits like `Sized`, `Copy`, `Clone`).
    fn trait_is_coinductive(&self, def_id: DefId) -> bool;

    /// If the given `def_id` identifies a built‑in trait, return its
    /// `BuiltinTrait` variant; otherwise `None`.
    fn is_builtin_trait(&self, def_id: DefId) -> Option<BuiltinTrait>;

    // ── Projection helpers ──────────────────────────────────────────

    /// Normalize a projection type (associated type projection) by looking
    /// up the impl and unifying with the target.
    fn handle_projection_eq(
        &mut self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: crate::symbol::Symbol,
        target: TypeId,
        cause: &super::obligation::ObligationCause,
    ) -> Result<ImplSource, SolveError>;

    /// Normalize a projection type with a possibly‑unknown target.
    fn handle_projection_normalize(
        &mut self,
        projection: &super::obligation::ProjectionTy,
        target: TypeId,
        cause: &super::obligation::ObligationCause,
    ) -> Result<ImplSource, SolveError>;

    // ── Match constraint helpers ─────────────────────────────────────

    /// Discharge a match constraint by determining the scrutinee's shape
    /// and matching against the registered branch patterns.
    ///
    /// Returns `Ok(continuation_obligations)` if the match was discharged,
    /// or `Err(())` if no branch matched and no else_ fallback exists.
    ///
    /// The default implementation returns `Err(())` — the actual match
    /// discharging logic is in `InferenceContext` (old solver) and will
    /// be migrated incrementally.
    fn discharge_match(
        &mut self,
        _scrutinee: TypeId,
        _branches_id: (usize, usize),
    ) -> Result<Vec<super::obligation::Obligation>, ()> {
        Err(())
    }

    // ── Defaulting ──────────────────────────────────────────────────

    /// Default unresolved inference variables that have guided kinds
    /// (Integer, Float, Bool, Numeric) to their default types.
    ///
    /// This is called by `FulfillmentContext::evaluate_all` after all
    /// obligations and constraints have been processed.  The defaulting
    /// logic is shared with the old solver via `defaulting::default_variables`.
    ///
    /// The default implementation is a no-op — the actual defaulting is
    /// still performed by `InferenceContext::solve` in the old solver.
    /// Subtypes that have access to the inference variable tables can
    /// override this to provide full defaulting in the new solver.
    fn default_variables(&mut self) -> Result<(), crate::hir::types::TypeError> {
        Ok(())
    }
}

/// Extension trait for `SolverDelegate` that provides methods used by `EvalCtxt`.
///
/// Analogous to Rust's `SolverDelegateEvalExt` in `eval_ctxt/mod.rs`.
pub trait SolverDelegateEvalExt: SolverDelegate + Sized {
    /// Evaluate a goal from outside the trait solver.
    fn evaluate_root_goal(
        &mut self,
        goal: &crate::hir::traits::solver::obligation::Obligation,
    ) -> Result<
        crate::hir::traits::solver::obligation::ImplSource,
        crate::hir::traits::solver::obligation::SolveError,
    >;

    /// Check whether evaluating a goal may hold.
    fn root_goal_may_hold(
        &mut self,
        goal: &crate::hir::traits::solver::obligation::Obligation,
    ) -> bool;
}

/// Blanket implementation of `SolverDelegateEvalExt` for all `Sized` `SolverDelegate` types.
impl<D: SolverDelegate + Sized> SolverDelegateEvalExt for D {
    fn evaluate_root_goal(
        &mut self,
        goal: &crate::hir::traits::solver::obligation::Obligation,
    ) -> Result<
        crate::hir::traits::solver::obligation::ImplSource,
        crate::hir::traits::solver::obligation::SolveError,
    > {
        let mut search_graph = crate::hir::traits::solver::search_graph::SearchGraph::new();
        let span = goal.cause.span;
        let mut ecx =
            crate::hir::traits::solver::eval_ctxt::EvalCtxt::new(self, &mut search_graph, span);
        crate::hir::traits::solver::eval::evaluate_goal(&mut ecx, goal)
    }

    fn root_goal_may_hold(
        &mut self,
        goal: &crate::hir::traits::solver::obligation::Obligation,
    ) -> bool {
        self.evaluate_root_goal(goal).is_ok()
    }
}
