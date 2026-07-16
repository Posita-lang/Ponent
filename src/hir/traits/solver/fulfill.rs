use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
use crate::hir::traits::solver::eval::evaluate_goal;
use crate::hir::traits::solver::forest::{ObligationForest, MAX_NODES};
use crate::hir::traits::solver::obligation::{ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate, SolveError};
use crate::hir::traits::solver::project::ProjectionCache;
use crate::hir::traits::solver::select::SelectionContext;
use crate::hir::traits::TraitEnv;
use crate::hir::types::TypeContext;
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
/// let mut fulfill = FulfillmentContext::new(ctx, trait_env, &caller_bounds);
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
        trait_env: &'a TraitEnv,
        symbols: &'a SymbolTable,
        builtin_registry: &'a BuiltinTraitRegistry,
        proj_cache: &'a ProjectionCache,
        caller_bounds: &'a [Predicate],
    ) -> Self {
        FulfillmentContext {
            forest: ObligationForest::new(),
            selcx: SelectionContext::new(ctx, trait_env, symbols, builtin_registry, proj_cache, caller_bounds),
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

        loop {
            // Compact the forest periodically to prevent unbounded memory growth.
            iteration_count += 1;
            if iteration_count % 100 == 0 && self.forest.len() > MAX_NODES {
                self.forest.compact();
            }

            // ── Progress check (BEFORE next_pending) ──
            // If no pending nodes remain and no deferred node has a resolved
            // stalled_on variable, we are stalled — no progress can be made
            // until the old solver resolves more inference variables.
            //
            // If there ARE ready deferred nodes, recycle them back to Pending
            // so next_pending can pick them up.
            let pending_count = self.forest.pending_count();
            if pending_count == 0 {
                if self.forest.has_ready_deferred(self.selcx.ctx) {
                    self.forest.recycle_ready_deferred(self.selcx.ctx);
                } else {
                    break;
                }
            }

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

            // Select a candidate and recursively evaluate nested goals.
            let obligation = self.forest.obligation_at(idx).clone();
            let result = evaluate_goal(&mut self.selcx, &obligation);

            // Leave the evaluating state
            self.forest.leave_evaluating(idx);

            match result {
                Ok(ImplSource::Deferred { stalled_on }) => {
                    // Cannot resolve yet — defer and retry later.
                    // Store the blocking inference variables so the
                    // caller can selectively re-evaluate when they
                    // are resolved.
                    self.forest.mark_deferred(idx, stalled_on);
                }
                Ok(_) => {
                    // All nested goals were resolved recursively inside
                    // evaluate_goal — no need to register children here.
                    self.forest.mark_resolved(idx);
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