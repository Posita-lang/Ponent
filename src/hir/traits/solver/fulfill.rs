use crate::ast::Span;
use crate::hir::infer::defaulting;
use crate::hir::infer::{GenStatus, InferenceContext, TypeVariableKind, VarOrigin};
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval::evaluate_goal;
use crate::hir::traits::solver::eval_ctxt::EvalCtxt;
use crate::hir::traits::solver::forest::{MAX_NODES, ObligationForest};
use crate::hir::traits::solver::obligation::{
    ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate, SolveError,
};
use crate::hir::traits::solver::search_graph::SearchGraph;
use crate::hir::types::{TypeContext, TypeId};

/// Drives iterative trait resolution.
///
/// Owns the `ObligationForest` and manages the selection + propagation loop.
/// Generic over `D: SolverDelegate` so it can be used with any solver backend
/// (production `SelectionContext`, mock delegates for testing, etc.).
///
/// Eq/Sub/Match constraints are now handled through `Predicate::Eq`,
/// `Predicate::Sub`, and `Predicate::Match` registered as regular obligations
/// (see `register_predicate`).  The old inline constraint structs
/// (`EqConstraint`, `SubConstraint`, `MatchConstraint`) and their evaluation
/// methods have been removed as part of the EvalCtxt migration.
///
/// Usage:
/// ```ignore
/// let mut fulfill = FulfillmentContext::new(&mut delegate);
/// fulfill.register_obligation(obligation);
/// match fulfill.evaluate_all() {
///     Ok(()) => { /* all obligations resolved */ }
///     Err(errors) => { /* report errors */ }
/// }
/// ```
pub struct FulfillmentContext<'a, D: SolverDelegate> {
    forest: ObligationForest,
    delegate: &'a mut D,
    /// Owns the search graph for cycle detection and fixpoint iteration.
    /// Passed as `&mut` to `EvalCtxt` during goal evaluation.
    search_graph: SearchGraph,
    /// Inference variable data for the defaulting step.
    /// Set by `set_infer_data` after construction.
    infer_var_type_ids: Vec<TypeId>,
    infer_type_vars: Vec<(TypeVariableKind, VarOrigin)>,
    infer_gen_statuses: Vec<GenStatus>,
}

impl<'a, D: SolverDelegate> FulfillmentContext<'a, D> {
    pub fn new(delegate: &'a mut D) -> Self {
        FulfillmentContext {
            forest: ObligationForest::new(),
            delegate,
            search_graph: SearchGraph::new(),
            infer_var_type_ids: Vec::new(),
            infer_type_vars: Vec::new(),
            infer_gen_statuses: Vec::new(),
        }
    }

    /// Set the inference variable data from the `InferenceContext`.
    /// This enables the defaulting step in `evaluate_all_inner`.
    pub fn set_infer_data(
        &mut self,
        var_type_ids: &[TypeId],
        type_vars: &[(TypeVariableKind, VarOrigin)],
        gen_statuses: &[GenStatus],
    ) {
        self.infer_var_type_ids = var_type_ids.to_vec();
        self.infer_type_vars = type_vars.to_vec();
        self.infer_gen_statuses = gen_statuses.to_vec();
    }

    /// Convenience wrapper: extract inference variable data from an
    /// `InferenceContext` and forward it to `set_infer_data`.
    pub fn set_infer_data_from(&mut self, infer: &InferenceContext) {
        let type_vars: Vec<(TypeVariableKind, VarOrigin)> = infer
            .type_vars()
            .iter()
            .enumerate()
            .map(|(i, tv)| {
                let origin = infer
                    .var_origins()
                    .get(i)
                    .copied()
                    .unwrap_or(VarOrigin::Synthetic);
                (tv.kind, origin)
            })
            .collect();
        self.set_infer_data(infer.var_type_ids(), &type_vars, infer.gen_statuses());
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

    #[must_use]
    pub fn evaluate_all(&mut self) -> Result<(), Vec<SolveError>> {
        self.evaluate_all_inner(false)
    }

    /// Like `evaluate_all`, but returns an error if any obligations remain
    /// deferred after the solver stalls.  This is the version to use for the
    /// final pass, after the old solver has resolved all inference variables.
    #[must_use]
    pub fn evaluate_all_final(&mut self) -> Result<(), Vec<SolveError>> {
        self.evaluate_all_inner(true)
    }

    fn evaluate_all_inner(&mut self, error_on_deferred: bool) -> Result<(), Vec<SolveError>> {
        let mut errors = Vec::new();
        let mut iteration_count: usize = 0;
        // Track the most recently processed obligation's span for error reporting.
        let mut last_span: Option<crate::ast::Span> = None;

        self.search_graph.begin_fixpoint();

        loop {
            // Compact the forest periodically to prevent unbounded memory growth.
            iteration_count += 1;
            if iteration_count % 100 == 0 && self.forest.len() > MAX_NODES {
                self.forest.compact();
            }

            // ── Defaulting ──
            if !self.infer_var_type_ids.is_empty() {
                if let Err(e) = defaulting::default_variables(
                    self.delegate.ctx(),
                    &self.infer_var_type_ids,
                    &self.infer_type_vars,
                    &self.infer_gen_statuses,
                ) {
                    let span = last_span.unwrap_or(crate::ast::Span::new(0, 0));
                    errors.push(SolveError::Ambiguous {
                        trait_id: crate::hir::types::DefId(0),
                        self_ty: self.delegate.ctx().error(),
                        span,
                        num_candidates: 0,
                    });
                    break;
                }
            }

            // ── Fixpoint check ──
            // If no goal was entered in this iteration, we have converged.
            if !self.search_graph.has_changed() {
                break;
            }
            // Try to advance the fixpoint iteration. If the limit is reached,
            // report overflow (analogous to Rust's fixpoint_overflow_result).
            if !self.search_graph.try_fixpoint_step() {
                let span = last_span.unwrap_or(crate::ast::Span::new(0, 0));
                errors.push(SolveError::Overflow {
                    obligation: Box::new(Obligation {
                        cause: crate::hir::traits::solver::ObligationCause {
                            span,
                            code: crate::hir::traits::solver::ObligationCauseCode::Misc,
                        },
                        predicate: crate::hir::traits::solver::Predicate::Sized {
                            ty: self.delegate.ctx().error(),
                        },
                        recursion_depth: 0,
                    }),
                    depth: 0,
                });
                break;
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
                if self.forest.has_ready_deferred(self.delegate.ctx()) {
                    self.forest.recycle_ready_deferred(self.delegate.ctx());
                } else {
                    break;
                }
            }

            // Get the next pending obligation
            let Some(idx) = self.forest.next_pending() else {
                break; // all processed
            };

            // Try to enter evaluation (with cycle detection)
            if !self.forest.mark_evaluating(idx, self.delegate.ctx()) {
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
            let span = obligation.cause.span;
            last_span = Some(span);
            let mut ecx = EvalCtxt::new(&mut *self.delegate, &mut self.search_graph, span);
            let result = evaluate_goal(&mut ecx, &obligation);

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
                    self_ty: self.delegate.ctx().error(),
                    span: last_span.unwrap_or(crate::ast::Span::new(0, 0)),
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
