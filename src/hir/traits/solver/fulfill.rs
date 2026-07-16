use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::forest::{ObligationForest, MAX_NODES};
use crate::hir::traits::solver::obligation::{ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate, SolveError};
use crate::hir::traits::solver::project::ProjectionCache;
use crate::hir::traits::solver::select::SelectionContext;
use crate::hir::traits::TraitEnv;
use crate::hir::types::TypeContext;
use crate::hir::infer::InferenceContext;
use crate::hir::symbol::SymbolTable;

/// Drives iterative trait resolution.
///
/// Owns the `ObligationForest` and manages the selection + propagation loop.
/// Modeled after rustc's `FulfillmentContext` but simplified:
/// - No separate `ObligationProcessor` trait (we use direct `SelectionContext` methods)
/// - No `ObligationForest` cache keys (Posita has no `ParamEnv` at this level)
///
/// Usage:
/// ```ignore
/// let mut fulfill = FulfillmentContext::new(ctx, infer, trait_env, &caller_bounds);
/// fulfill.register_obligation(obligation);
/// match fulfill.evaluate_all() {
///     Ok(()) => { /* all obligations resolved */ }
///     Err(errors) => { /* report errors */ }
/// }
/// ```
pub struct FulfillmentContext<'a> {
    forest: ObligationForest,
    selcx: SelectionContext<'a>,
}

impl<'a> FulfillmentContext<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        infer: &'a mut InferenceContext,
        trait_env: &'a TraitEnv,
        symbols: &'a SymbolTable,
        builtin_registry: &'a BuiltinTraitRegistry,
        proj_cache: &'a ProjectionCache,
        caller_bounds: &'a [Predicate],
    ) -> Self {
        FulfillmentContext {
            forest: ObligationForest::new(),
            selcx: SelectionContext::new(ctx, infer, trait_env, symbols, builtin_registry, proj_cache, caller_bounds),
        }
    }

    /// Register a new obligation to be fulfilled.
    pub fn register_obligation(&mut self, obligation: Obligation) {
        self.forest.register(obligation);
    }

    /// Register an obligation from a predicate.
    /// Convenience wrapper that creates an Obligation from a Predicate
    /// with default cause and recursion depth.
    pub fn register_predicate(&mut self, predicate: Predicate, span: crate::ast::Span) {
        let obligation = Obligation {
            cause: crate::hir::traits::solver::obligation::ObligationCause {
                span,
                code: crate::hir::traits::solver::obligation::ObligationCauseCode::Misc,
            },
            predicate,
            recursion_depth: 0,
        };
        self.forest.register(obligation);
    }

    pub fn evaluate_all(&mut self) -> Result<(), Vec<SolveError>> {
        self.evaluate_all_inner(false)
    }

    /// Like `evaluate_all`, but returns an error if any obligations remain
    /// deferred after the solver stalls.  This is the version to use for the
    /// final pass, after the old solver has resolved all inference variables.
    pub fn evaluate_all_final(&mut self) -> Result<(), Vec<SolveError>> {
        self.evaluate_all_inner(true)
    }

    fn evaluate_all_inner(&mut self, error_on_deferred: bool) -> Result<(), Vec<SolveError>> {
        let mut errors = Vec::new();
        let mut iteration_count: usize = 0;
        let mut last_deferred_count: usize = 0;

        loop {
            // Compact the forest periodically to prevent unbounded memory growth.
            iteration_count += 1;
            if iteration_count % 100 == 0 && self.forest.len() > MAX_NODES {
                self.forest.compact();
            }

            // ── Progress check (BEFORE next_pending) ──
            // If the only remaining pending nodes are Deferred and the count
            // hasn't changed since the last iteration, we are stalled — no
            // progress can be made until the types are resolved by the old
            // solver.  Exit the loop and return the deferred obligations.
            //
            // IMPORTANT: This check must happen BEFORE next_pending() to avoid
            // dequeuing a node that would be leaked if the loop exits here.
            // Also, we must verify that pending_count == 0 (no non-deferred
            // pending nodes remain) — the deferred_count alone is not enough,
            // because there could be Pending nodes that haven't been tried yet.
            let deferred_count = self.forest.deferred_count();
            let pending_count = self.forest.pending_count();
            if pending_count == 0 && deferred_count > 0 && deferred_count == last_deferred_count {
                // All remaining nodes are deferred and no progress was made.
                // Exit — the checker will retry after the old solver runs.
                break;
            }
            last_deferred_count = deferred_count;

            // Get the next pending obligation
            let Some(idx) = self.forest.next_pending() else {
                break; // all processed
            };

            // Try to enter evaluation (with cycle detection)
            if !self.forest.mark_evaluating(idx, self.selcx.ctx) {
                // Cycle detected.  The key was NOT inserted by this node
                // (it was already in active_path from an ancestor that nests
                // the same predicate).  Therefore we must NOT call
                // leave_evaluating — that would remove the ancestor's key
                // and corrupt cycle detection for the rest of the ancestor's
                // evaluation.  The ancestor will call leave_evaluating when
                // it finishes.
                match self.forest.state_at(idx) {
                    crate::hir::traits::solver::forest::ObligationState::CycleDetected => {
                        // Coinductive cycle — treat as resolved
                        self.forest.mark_resolved(idx);
                    }
                    _ => {
                        // Error was set by mark_evaluating (non-coinductive cycle)
                        if let crate::hir::traits::solver::forest::ObligationState::Error(e) =
                            self.forest.state_at(idx)
                        {
                            errors.push(e.clone());
                        }
                    }
                }
                continue;
            }

            // Select a candidate
            let obligation = self.forest.obligation_at(idx).clone();
            let result = self.selcx.select(&obligation);

            // Leave the evaluating state
            self.forest.leave_evaluating(idx);

            match result {
                Ok(impl_source) => {
                    match impl_source {
                        ImplSource::Deferred => {
                            // Cannot resolve yet — defer and retry later.
                            // The node's state is still Evaluating; reset to Deferred.
                            self.forest.mark_deferred(idx);
                        }
                        _ => {
                            // Register sub-obligations
                            for sub in impl_source.nested_obligations() {
                                self.forest.register_child(sub, idx);
                            }
                            self.forest.mark_resolved(idx);
                        }
                    }
                }
                Err(e) => {
                    self.forest.mark_error(idx, e.clone());
                    errors.push(e);
                }
            }
        }

        if errors.is_empty() {
            if error_on_deferred && self.forest.deferred_count() > 0 {
                // Deferred obligations remain after the final solver pass.
                // These are obligations whose self_ty is still an inference
                // variable — they could not be resolved even after the old
                // solver has run.  Report them as errors.
                Err(vec![SolveError::Ambiguous {
                    trait_id: crate::hir::types::DefId(0),
                    self_ty: self.selcx.ctx.error(),
                    span: crate::ast::Span::new(0, 0),
                    num_candidates: 0,
                }])
            } else {
                Ok(())
            }
        } else {
            Err(errors)
        }
    }

    /// Check if there are still pending obligations.
    pub fn has_pending(&self) -> bool {
        self.forest.has_pending()
    }

    /// Get the number of pending obligations.
    pub fn pending_count(&self) -> usize {
        self.forest.pending_count()
    }

    /// Check if there are deferred obligations that need retry.
    pub fn has_deferred(&self) -> bool {
        self.forest.deferred_count() > 0
    }

    /// Get a reference to the underlying obligation forest.
    pub fn forest(&self) -> &ObligationForest {
        &self.forest
    }
}

/// Helper: extract nested obligations from an ImplSource.
trait NestedObligations {
    fn nested_obligations(&self) -> Vec<Obligation>;
}

impl NestedObligations for crate::hir::traits::solver::obligation::ImplSource {
    fn nested_obligations(&self) -> Vec<Obligation> {
        match self {
            crate::hir::traits::solver::obligation::ImplSource::UserDefined { nested, .. } => {
                nested.clone()
            }
            crate::hir::traits::solver::obligation::ImplSource::Param(nested) => nested.clone(),
            crate::hir::traits::solver::obligation::ImplSource::Builtin(_) => vec![],
            crate::hir::traits::solver::obligation::ImplSource::Object { nested, .. } => {
                nested.clone()
            }
            crate::hir::traits::solver::obligation::ImplSource::Auto { nested } => nested.clone(),
            crate::hir::traits::solver::obligation::ImplSource::Poly { nested, .. } => nested.clone(),
            crate::hir::traits::solver::obligation::ImplSource::Deferred => vec![],
        }
    }
}