use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::builtins::{BuiltinTrait, BuiltinTraitRegistry};
use crate::hir::traits::solver::obligation::{
    BuiltinImplSource, ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate,
    ProjectionTy, SolveError,
};
use crate::hir::traits::solver::project::{self, ProjectionCache};
use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;

/// The core trait resolution engine.
///
/// Analogous to rustc's `SelectionContext`. Responsible for:
/// 1. Candidate assembly (gathering possible impls/bounds/builtins)
/// 2. Winnowing (removing ambiguous/overlapping candidates)
/// 3. Confirmation (verifying the selected candidate and producing sub-obligations)
///
/// Uses `TraitEnv` as a read-only data source for registered impls.
/// Does NOT modify `TraitEnv` — all state mutations go through `TypeContext` transactions.
///
/// ## Refactoring Note
/// Candidate assembly (impls, caller_bounds, builtins, object_ty, poly) has been
/// moved to the `assembly` module via the `GoalKind` trait.  The `select` method
/// now delegates to `assembly::assemble_and_evaluate_candidates`.
/// The old private assembly methods (`assemble_candidates_from_*`, `try_match_impl`,
/// `winnow`, `specificity`, `confirm_candidate`) have been removed from this file.
/// See `src/hir/traits/solver/assembly/mod.rs` for the new implementation.
pub struct SelectionContext<'a> {
    pub ctx: &'a mut TypeContext,
    pub trait_env: &'a TraitEnv,
    pub symbols: &'a SymbolTable,
    pub builtin_registry: &'a BuiltinTraitRegistry,
    /// Caller bounds (from where-clauses in scope).
    pub caller_bounds: &'a [Predicate],
    /// Projection cache for associated type normalization.
    pub proj_cache: &'a ProjectionCache,
}

/// Maximum recursion depth for trait resolution before overflow.
pub(crate) const MAX_RECURSION_DEPTH: usize = 64;

/// A set of selection candidates.
#[derive(Clone, Debug)]
pub struct Candidates {
    pub vec: Vec<Candidate>,
    pub ambiguous: bool,
}

#[derive(Clone, Debug)]
pub enum Candidate {
    /// User-defined impl, identified by index in TraitEnv::impls.
    ///
    /// SAFETY: The `impl_source` field contains an `ImplSource::UserDefined`
    /// whose `Subst` holds `TypeId` values that were allocated inside a
    /// **rolled-back transaction** during candidate assembly
    /// (`assemble_candidates_from_impls`).  These `TypeId` values are valid
    /// (they were allocated by `alloc_infer_var` which does not go through
    /// the undo log) but the **unification bindings** between them and the
    /// obligation's types were undone by the rollback.
    ///
    /// Therefore the `impl_source` must NOT be used directly.  Instead,
    /// `confirm_candidate` re-runs `try_match_impl` inside a **fresh**
    /// transaction, which re-creates the bindings, and commits only if
    /// the candidate wins.  The `Subst` from the assembly phase is used
    /// only as a quick check that matching is possible — the actual
    /// bindings come from the fresh confirmation run.
    ///
    /// If you modify this code, ensure that:
    /// 1. The `impl_source` stored here is never used for code generation
    ///    or type resolution without re-confirmation.
    /// 2. The `idx` is the sole source of truth for identifying which impl
    ///    was matched.
    Impl { idx: usize, impl_source: ImplSource },
    /// Caller-provided bound (where-clause).
    /// Stores the (self_ty, args) that were matched during assembly so that
    /// confirm_candidate can re-apply the unification in a fresh transaction.
    Param { self_ty: TypeId, args: Vec<TypeId> },
    /// Builtin trait (Sized, Copy, Clone, etc.).
    Builtin(BuiltinImplSource),
    /// Object type bound (dyn Trait).
    Object {
        object_trait_id: DefId,
        nested: Vec<Obligation>,
    },
    /// Poly/unbox (Posita-specific).
    /// During assembly, the allocation ran inside a rolled-back transaction;
    /// confirm_candidate will re-apply it inside a fresh transaction.
    /// Only the quantifier count is needed for confirmation.
    Poly {
        /// Number of quantifiers on the poly type.  Used during confirmation
        /// to re-create fresh inference variables inside a committed transaction.
        quantifier_count: usize,
    },
}

/// A resolved obligation — self_ty has been followed through bindings.
#[derive(Clone, Debug)]
pub struct ResolvedObligation {
    pub trait_id: DefId,
    pub self_ty: TypeId,
    pub args: Vec<TypeId>,
    /// Whether the self_ty is still an inference variable, meaning the
    /// obligation cannot be resolved yet and should be retried later.
    pub ambiguous: bool,
    /// The recursion depth of the parent obligation that produced this one.
    /// Used to propagate depth when creating nested obligations during
    /// confirmation (e.g., `Candidate::Poly`).
    pub parent_depth: usize,
    /// The source span from the original obligation, preserved for error reporting.
    pub span: crate::ast::Span,
}

impl<'a> SelectionContext<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        trait_env: &'a TraitEnv,
        symbols: &'a SymbolTable,
        builtin_registry: &'a BuiltinTraitRegistry,
        proj_cache: &'a ProjectionCache,
        caller_bounds: &'a [Predicate],
    ) -> Self {
        SelectionContext {
            ctx,
            trait_env,
            symbols,
            builtin_registry,
            proj_cache,
            caller_bounds,
        }
    }

    /// Select a candidate for the given obligation.
    ///
    /// Returns the resolved `ImplSource` on success, or a `SolveError` on failure.
    ///
    /// This method now delegates to the `GoalKind`-based assembly engine
    /// (see `assembly::assemble_and_evaluate_candidates`), which replaces
    /// the old inline candidate assembly → winnowing → confirmation pipeline.
    /// The engine uses the `GoalKind` trait to dispatch assembly logic per
    /// predicate type, making it extensible without modifying the core engine.
    #[must_use]
    pub fn select(&mut self, obligation: &Obligation) -> Result<ImplSource, SolveError> {
        // Create an EvalCtxt wrapping self and delegate to the assembly engine.
        let mut search_graph = crate::hir::traits::solver::search_graph::SearchGraph::new();
        let span = obligation.cause.span;
        let mut ecx = crate::hir::traits::solver::eval_ctxt::EvalCtxt::new(
            self,
            &mut search_graph,
            span,
        );
        crate::hir::traits::solver::assembly::assemble_and_evaluate_candidates(&mut ecx, obligation)
    }

    /// Resolve the self_ty through bindings and extract the trait predicate.
    fn resolve_obligation(&self, obligation: &Obligation) -> ResolvedObligation {
        match &obligation.predicate {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => {
                let resolved_self = self.ctx.resolve_binding(*self_ty);
                let resolved_args: Vec<TypeId> =
                    args.iter().map(|a| self.ctx.resolve_binding(*a)).collect();
                let ambiguous = self.ctx.is_infer_var(resolved_self);
                ResolvedObligation {
                    trait_id: *trait_id,
                    self_ty: resolved_self,
                    args: resolved_args,
                    ambiguous,
                    parent_depth: obligation.recursion_depth,
                    span: obligation.cause.span,
                }
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = self.ctx.resolve_binding(*self_ty);
                let ambiguous = self.ctx.is_infer_var(resolved_self);
                ResolvedObligation {
                    trait_id: *trait_id,
                    self_ty: resolved_self,
                    args: vec![],
                    ambiguous,
                    parent_depth: obligation.recursion_depth,
                    span: obligation.cause.span,
                }
            }
            Predicate::Sized { ty } => {
                let resolved_ty = self.ctx.resolve_binding(*ty);
                let ambiguous = self.ctx.is_infer_var(resolved_ty);
                ResolvedObligation {
                    trait_id: DefId(usize::MAX), // sentinel
                    self_ty: resolved_ty,
                    args: vec![],
                    ambiguous,
                    parent_depth: obligation.recursion_depth,
                    span: obligation.cause.span,
                }
            }
            _ => {
                // Fallback for other predicate types (ProjectionEq, etc.)
                ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: self.ctx.error(),
                    args: vec![],
                    ambiguous: false,
                    parent_depth: obligation.recursion_depth,
                    span: obligation.cause.span,
                }
            }
        }
    }

    // ── Projection handling ──

    /// Handle `<SelfTy as Trait>::AssocName == Value` — resolve the projection
    /// and unify with the expected value.
    fn handle_projection_eq(
        &mut self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: &Symbol,
        value: TypeId,
        cause: &ObligationCause,
    ) -> Result<ImplSource, SolveError> {
        let resolved_self = self.ctx.resolve_binding(self_ty);
        let proj = ProjectionTy {
            trait_id,
            self_ty: resolved_self,
            args: vec![],
            assoc_name: *assoc_name,
        };
        let normalized = project::normalize_projection(
            &proj,
            self.trait_env,
            self.ctx,
            self.proj_cache,
            self.symbols,
        );
        match normalized {
            Some(concrete_ty) => {
                self.ctx
                    .unify(value, concrete_ty)
                    .map_err(|_| SolveError::Mismatch {
                        expected: value,
                        found: concrete_ty,
                        span: cause.span,
                    })?;
                Ok(ImplSource::Param(vec![]))
            }
            None => Err(SolveError::NotFound {
                trait_id,
                self_ty: resolved_self,
                span: cause.span,
            }),
        }
    }

    /// Handle `<SelfTy as Trait>::AssocName` — normalize the projection
    /// and return the concrete type via an ImplSource.
    fn handle_projection_normalize(
        &mut self,
        projection: &ProjectionTy,
        target: TypeId,
        cause: &ObligationCause,
    ) -> Result<ImplSource, SolveError> {
        let resolved_self = self.ctx.resolve_binding(projection.self_ty);
        let proj = ProjectionTy {
            trait_id: projection.trait_id,
            self_ty: resolved_self,
            args: projection.args.clone(),
            assoc_name: projection.assoc_name,
        };
        let normalized = project::normalize_projection(
            &proj,
            self.trait_env,
            self.ctx,
            self.proj_cache,
            self.symbols,
        );
        match normalized {
            Some(concrete_ty) => {
                self.ctx
                    .unify(target, concrete_ty)
                    .map_err(|_| SolveError::Mismatch {
                        expected: target,
                        found: concrete_ty,
                        span: cause.span,
                    })?;
                Ok(ImplSource::Param(vec![]))
            }
            None => Err(SolveError::NotFound {
                trait_id: projection.trait_id,
                self_ty: resolved_self,
                span: cause.span,
            }),
        }
    }

    // ── Projection handling ──
}

// ── SolverDelegate implementation ───────────────────────────────────

impl SolverDelegate for SelectionContext<'_> {
    fn ctx(&mut self) -> &mut TypeContext {
        self.ctx
    }

    fn trait_env(&self) -> &TraitEnv {
        self.trait_env
    }

    fn symbols(&self) -> &SymbolTable {
        self.symbols
    }

    fn builtin_registry(&self) -> &BuiltinTraitRegistry {
        self.builtin_registry
    }

    fn proj_cache(&self) -> &ProjectionCache {
        self.proj_cache
    }

    fn caller_bounds(&self) -> &[Predicate] {
        self.caller_bounds
    }

    fn resolve_obligation(&self, obligation: &Obligation) -> ResolvedObligation {
        SelectionContext::resolve_obligation(self, obligation)
    }

    fn trait_is_coinductive(&self, def_id: DefId) -> bool {
        self.builtin_registry
            .lookup(def_id)
            .is_some_and(|bt| bt.is_coinductive())
    }

    fn is_builtin_trait(&self, def_id: DefId) -> Option<BuiltinTrait> {
        self.builtin_registry.lookup(def_id)
    }

    fn handle_projection_eq(
        &mut self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: crate::symbol::Symbol,
        target: TypeId,
        cause: &ObligationCause,
    ) -> Result<ImplSource, SolveError> {
        SelectionContext::handle_projection_eq(self, trait_id, self_ty, &assoc_name, target, cause)
    }

    fn handle_projection_normalize(
        &mut self,
        projection: &ProjectionTy,
        target: TypeId,
        cause: &ObligationCause,
    ) -> Result<ImplSource, SolveError> {
        SelectionContext::handle_projection_normalize(self, projection, target, cause)
    }

    fn default_variables(&mut self) -> Result<(), crate::hir::types::TypeError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::traits::TraitEnv;
    use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
    use crate::hir::traits::solver::obligation::{ObligationCause, ObligationCauseCode, Predicate};
    use crate::hir::traits::solver::project::ProjectionCache;
    use crate::hir::types::{CrateId, DefId};

    #[test]
    fn test_overflow_at_max_depth() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // Create the type BEFORE passing &mut ctx to SelectionContext.
        let int_ty = ctx.int(32, true);

        let mut selcx = SelectionContext::new(
            &mut ctx,
            &trait_env,
            &symbols,
            &builtin_registry,
            &proj_cache,
            &caller_bounds,
        );

        // Obligation at exactly MAX_RECURSION_DEPTH: must overflow
        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: int_ty },
            recursion_depth: MAX_RECURSION_DEPTH,
        };

        let result = selcx.select(&obligation);
        match result {
            Err(SolveError::Overflow { depth, .. }) => {
                assert_eq!(depth, MAX_RECURSION_DEPTH);
            }
            other => {
                panic!(
                    "expected Overflow at depth {}, got {:?}",
                    MAX_RECURSION_DEPTH, other
                );
            }
        }
    }

    #[test]
    fn test_no_overflow_below_max_depth() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // Create the type BEFORE passing &mut ctx to SelectionContext.
        let int_ty = ctx.int(32, true);

        let mut selcx = SelectionContext::new(
            &mut ctx,
            &trait_env,
            &symbols,
            &builtin_registry,
            &proj_cache,
            &caller_bounds,
        );

        // Obligation at MAX_RECURSION_DEPTH - 1: must NOT overflow.
        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: int_ty },
            recursion_depth: MAX_RECURSION_DEPTH - 1,
        };

        let result = selcx.select(&obligation);
        match result {
            Err(SolveError::Overflow { .. }) => {
                panic!(
                    "should NOT overflow at depth {} < MAX_RECURSION_DEPTH ({})",
                    MAX_RECURSION_DEPTH - 1,
                    MAX_RECURSION_DEPTH
                );
            }
            _ => {
                // Any other result (NotFound, Ambiguous, Deferred, Ok) is fine —
                // the point is that it did NOT overflow.
            }
        }
    }

    #[test]
    fn test_deferred_stalled_on_populated() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // Create an inference variable as the self_ty — this guarantees
        // select() returns Deferred { stalled_on }.
        let infer_var = ctx.alloc_infer_var(999);

        let mut selcx = SelectionContext::new(
            &mut ctx,
            &trait_env,
            &symbols,
            &builtin_registry,
            &proj_cache,
            &caller_bounds,
        );

        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: infer_var },
            recursion_depth: 0,
        };

        let result = selcx.select(&obligation);
        match result {
            Ok(ImplSource::Deferred { stalled_on }) => {
                // stalled_on must contain the inference variable that
                // was blocking resolution.
                assert!(!stalled_on.is_empty(), "stalled_on should not be empty");
                assert!(
                    stalled_on.contains(&infer_var),
                    "stalled_on should contain the blocking infer var (id=999), got {:?}",
                    stalled_on,
                );
            }
            other => {
                panic!(
                    "expected Deferred {{ stalled_on }} for infer var self_ty, got {:?}",
                    other
                );
            }
        }
    }

    #[test]
    fn test_forest_next_pending_skips_unresolved_deferred() {
        // Verify that next_pending skips a deferred node whose stalled_on
        // variables are still unresolved inference variables.
        let mut ctx = TypeContext::new();
        let mut forest = crate::hir::traits::solver::forest::ObligationForest::new();

        let infer_var = ctx.alloc_infer_var(1001);
        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: infer_var },
            recursion_depth: 0,
        };

        let idx = forest.register(obligation);
        // Mark it as deferred — simulating what select() + mark_deferred does.
        forest.mark_deferred(idx, vec![infer_var]);

        // next_pending should skip this node because the infer var is unresolved.
        assert!(
            forest.next_pending().is_none(),
            "next_pending should skip deferred node with unresolved stalled_on"
        );
    }

    #[test]
    fn test_forest_next_pending_returns_resolved_deferred() {
        // Verify that next_pending returns a deferred node when at least one
        // stalled_on variable has been resolved (bound to a concrete type).
        let mut ctx = TypeContext::new();
        let mut forest = crate::hir::traits::solver::forest::ObligationForest::new();

        let infer_var = ctx.alloc_infer_var(1002);
        let int_ty = ctx.int(32, true);
        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: infer_var },
            recursion_depth: 0,
        };

        let idx = forest.register(obligation);
        forest.mark_deferred(idx, vec![infer_var]);

        // Resolve the inference variable by binding it to Int<32>.
        ctx.set_binding(infer_var, int_ty);

        // Now next_pending should return the node because the stalled_on
        // variable is no longer an inference variable (it was recycled
        // to Pending by recycle_ready_deferred).
        forest.recycle_ready_deferred(&ctx);
        assert!(
            forest.next_pending().is_some(),
            "next_pending should return deferred node when stalled_on is resolved"
        );
    }

    #[test]
    fn test_forest_has_ready_deferred() {
        // Verify that has_ready_deferred correctly identifies whether any
        // deferred node has a resolved stalled_on variable.
        let mut ctx = TypeContext::new();
        let mut forest = crate::hir::traits::solver::forest::ObligationForest::new();

        let infer_var = ctx.alloc_infer_var(1003);
        let obligation = Obligation {
            cause: ObligationCause {
                span: crate::ast::Span::new(0, 0),
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty: infer_var },
            recursion_depth: 0,
        };

        let idx = forest.register(obligation);
        forest.mark_deferred(idx, vec![infer_var]);

        // Before resolution: no ready deferred nodes.
        assert!(!forest.has_ready_deferred(&ctx));

        // Resolve the inference variable.
        let int_ty = ctx.int(32, true);
        ctx.set_binding(infer_var, int_ty);

        // After resolution: has_ready_deferred should return true.
        assert!(forest.has_ready_deferred(&ctx));
    }
}
