use crate::ast::Span;
use crate::hir::infer::defaulting;
use crate::hir::infer::{GenStatus, InferenceContext, TypeVariableKind, VarOrigin};
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval::evaluate_goal;
use crate::hir::traits::solver::eval_ctxt::EvalCtxt;
use crate::hir::traits::solver::forest::{EnterEvaluation, MAX_NODES, ObligationForest};
use crate::hir::traits::solver::obligation::{
    ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate, SolveError,
};
use crate::hir::traits::solver::search_graph::SearchGraph;
use crate::hir::types::{DefId, TypeContext, TypeId};

/// Hard failsafe for the evaluation loop: the maximum number of loop
/// iterations (one obligation evaluated or recycled per iteration) before
/// the pass reports overflow instead of hanging.  Generous enough that a
/// large program never reaches it (each obligation evaluates a small
/// constant number of times), but bounded so a solver invariant violation
/// (a defer/recycle flip-flop) terminates the compile.
pub const MAX_FULFILLMENT_ITERATIONS: usize = 1_000_000;

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
pub struct FulfillmentContext<'a, D> {
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

impl<'a, 'input, D: SolverDelegate<'input>> FulfillmentContext<'a, D> {
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
        // An obligation whose self type contains the ERROR recovery sentinel
        // is a recovery artifact: the expression that produced it already
        // failed to type-check (e.g. a deref of a non-reference falls back to
        // the error type, fn_ctxt) and was silently recovered.  Enforcing
        // traits on the recovery type would surface spurious `no trait
        // implementation found for ... on type Error` errors on top of
        // already-recovered expressions — skip them (the recovery path owns
        // the diagnostics).  `contains_error` (not the shallow `is_error`)
        // also catches composite recoveries like `Vec<Error>`.
        let ty = obligation.predicate.self_ty();
        let resolved = self.delegate.ctx().resolve_binding(ty);
        if self.delegate.ctx().contains_error(resolved) {
            return;
        }
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
        self.register_obligation(obligation);
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
        // Track the most recently processed real obligation (not just its
        // span) so overflow/final-pass reports can anchor on a genuine goal.
        let mut last_goal: Option<Obligation> = None;

        self.search_graph.begin_fixpoint();

        loop {
            // Compact the forest periodically to prevent unbounded memory growth.
            iteration_count += 1;
            if iteration_count % 100 == 0 && self.forest.len() > MAX_NODES {
                self.forest.compact();
            }

            // ── Progress check (BEFORE next_pending) ──
            // Convergence: with no pending nodes and no deferred node whose
            // `stalled_on` variables have been resolved, the fixpoint has
            // converged — every obligation is either resolved, errored, or
            // parked on an unresolved inference variable.  Re-evaluation is
            // driven by the recycle loop below.  The search graph's
            // `has_changed` flag must NOT be used as the loop's convergence
            // signal: it is reset by every `evaluate_goal` call and only
            // reflects the last goal's internal fixpoint progress (it would
            // either skip evaluation entirely or break early with pending
            // nodes left unprocessed).
            //
            // Hard failsafe: terminate the loop past a generous iteration
            // budget and report overflow.  The termination argument relies
            // on invariants (defer only on unbound top-level infer vars,
            // monotone committed bindings, single-pass defaulting); if a
            // future change violates one of them, this cap — and only this
            // cap — stops the hang.  (It cannot live in the search graph:
            // nested goal evaluation calls `begin_fixpoint` and would reset
            // its counter mid-pass.)
            if iteration_count > MAX_FULFILLMENT_ITERATIONS {
                // Report the overflow against the most recent real
                // obligation — the synthetic `Sized<Error>` is only a last
                // resort when no goal was ever processed (unreachable in
                // practice: every iteration records the obligation it
                // touched, including cycle-detected ones).
                let obligation = last_goal.clone().unwrap_or_else(|| Obligation {
                    cause: crate::hir::traits::solver::ObligationCause {
                        span: crate::ast::DUMMY_SPAN,
                        code: crate::hir::traits::solver::ObligationCauseCode::Misc,
                    },
                    predicate: crate::hir::traits::solver::Predicate::Sized {
                        ty: self.delegate.ctx().error(),
                    },
                    recursion_depth: 0,
                });
                errors.push(SolveError::Overflow {
                    obligation: Box::new(obligation),
                    depth: 0,
                });
                break;
            }
            //
            // If there ARE ready deferred nodes, recycle them back to Pending
            // so next_pending can pick them up.
            let pending_count = self.forest.pending_count();
            if pending_count == 0 {
                if self.forest.has_ready_deferred(self.delegate.ctx()) {
                    self.forest.recycle_ready_deferred(self.delegate.ctx());
                } else {
                    // No pending and no ready deferred: defaulting is the
                    // LAST resort.  It binds unconstrained inference
                    // variables to their default types, which may unblock
                    // deferred nodes (their `stalled_on` variables become
                    // concrete).  It runs only here — NOT at the top of
                    // every iteration — because it only matters once the
                    // solver has run out of obligations to process; running
                    // it per-iteration was O(iterations × vars) waste.
                    match self.try_default_infer_vars() {
                        Ok(true) => {}
                        Ok(false) => break, // truly converged
                        Err(e) => {
                            errors.push(e);
                            break;
                        }
                    }
                }
            }

            // Get the next pending obligation
            let Some(idx) = self.forest.next_pending() else {
                break; // all processed
            };

            // Try to enter evaluation (with cycle detection).  For trait
            // goals, report whether the trait is a user-declared coinductive
            // trait (`@coinductive`) so the forest can classify the cycle.
            let coinductive_trait = match &self.forest.obligation_at(idx).predicate {
                Predicate::Trait { trait_id, .. } => self.delegate.trait_is_coinductive(*trait_id),
                _ => false,
            };
            match self
                .forest
                .mark_evaluating(idx, self.delegate.ctx(), coinductive_trait)
            {
                EnterEvaluation::CycleDetected => {
                    // Cycle detected.  The key was NOT inserted by this node
                    // (it was already in active_path from an ancestor that
                    // nests the same predicate).  Therefore we must NOT call
                    // leave_evaluating — that would remove the ancestor's
                    // key and corrupt cycle detection for the rest of the
                    // ancestor's evaluation.  The ancestor will call
                    // leave_evaluating when it finishes.  (This is enforced
                    // by the `EnterEvaluation` enum: `leave_evaluating` is
                    // only reachable in the `Entered` arm below.)
                    // The obligation is still recorded so the iteration-cap
                    // overflow report never falls back to the synthetic
                    // obligation on a pure cycle-detection spin.
                    last_goal = Some(self.forest.obligation_at(idx).clone());
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
                EnterEvaluation::Entered => {}
            }

            // Select a candidate and recursively evaluate nested goals.
            let obligation = self.forest.obligation_at(idx).clone();
            let span = obligation.cause.span;
            let mut ecx = EvalCtxt::new(&mut *self.delegate, &mut self.search_graph, span);
            let result = evaluate_goal(&mut ecx, &obligation);

            // Leave the evaluating state
            self.forest.leave_evaluating(idx);
            // Record the real goal for error reporting (overflow cap and
            // final-pass deferred reports).  The move is safe here — the
            // evaluation above only borrowed it.
            last_goal = Some(obligation);

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
                    // The recovery-sentinel skip (mirror of the
                    // registration-time skip): an obligation whose self_ty
                    // has since RESOLVED to the error sentinel is a
                    // recovery artifact — the expression it came from
                    // already failed elsewhere (e.g. a silently recovered
                    // deref).  Enforcing traits on the recovery type
                    // surfaces cascading `... on type Error` errors; the
                    // recovery path owns the diagnostics.  The var could
                    // not be recognized at registration (still an
                    // inference variable), so it is filtered here — only
                    // for the self-typed error kinds (a NotFound/Ambiguous
                    // on a real type is still a genuine error).
                    if e.is_recovery_artifact(self.delegate.ctx()) {
                        self.forest.mark_resolved(idx);
                        continue;
                    }
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
                // solver has run.  Report them as errors, anchored at the
                // FIRST deferred obligation's actual span and self type
                // (the sentinel `!!` + the last processed span would be
                // misleading: the stalled var may be a real type).
                let (self_ty, span) = match self.forest.first_deferred() {
                    Some(ob) => (ob.predicate.self_ty(), ob.cause.span),
                    None => (
                        last_goal
                            .as_ref()
                            .map(|ob| ob.predicate.self_ty())
                            .unwrap_or_else(|| self.delegate.ctx().error()),
                        last_goal
                            .as_ref()
                            .map(|ob| ob.cause.span)
                            .unwrap_or(crate::ast::DUMMY_SPAN),
                    ),
                };
                Err(vec![SolveError::Ambiguous {
                    trait_id: None,
                    self_ty,
                    span,
                    num_candidates: 0,
                }])
            } else {
                Ok(())
            }
        } else {
            Err(errors)
        }
    }

    /// The best available `(trait_id, self_ty)` for error reporting when a
    /// solver step fails without an obligation in hand: the current goal on
    /// the search-graph stack, else the first pending
    /// Trait/AutoTrait/ProjectionEq obligation in the forest.  `None` when
    /// neither is available (no trait-shaped goal in flight) — the caller
    /// falls back to synthetic values.
    fn current_goal_info(&self) -> Option<(Option<DefId>, TypeId)> {
        if let Some(entry) = self.search_graph.current_goal() {
            return Some((entry.key.trait_id, entry.key.self_ty));
        }
        // Peek (do NOT pop — a popped-but-unprocessed node would be
        // silently dropped).
        let idx = self.forest.peek_pending()?;
        let ob = self.forest.obligation_at(idx);
        match &ob.predicate {
            Predicate::Trait {
                trait_id, self_ty, ..
            }
            | Predicate::AutoTrait {
                trait_id, self_ty, ..
            }
            | Predicate::ProjectionEq {
                trait_id, self_ty, ..
            } => Some((Some(*trait_id), *self_ty)),
            _ => None,
        }
    }

    /// Last-resort progress: run inference-variable defaulting and report
    /// whether it unblocked deferred nodes.
    ///
    /// Returns `Ok(true)` when defaulting succeeded and recycled deferred
    /// nodes are ready (the caller continues the loop), `Ok(false)` when no
    /// progress is possible (the fixpoint has converged), and `Err(e)`
    /// when defaulting failed (the caller reports `e` and stops).
    ///
    /// The defaulting error is never silently swallowed: its own span
    /// (more precise than the last-processed obligation's) anchors the
    /// reported ambiguity.  Note `default_variables` currently cannot
    /// construct an error — its Pass 2 binds unresolved `Any`/Unconstrained
    /// expression variables to the `Error` sentinel instead of failing (see
    /// `src/hir/infer/defaulting.rs`), so this arm is defensive: it keeps
    /// the error's span so the report stays anchored if that changes.
    fn try_default_infer_vars(&mut self) -> Result<bool, SolveError> {
        if self.infer_var_type_ids.is_empty() {
            return Ok(false);
        }
        if let Err(e) = defaulting::default_variables(
            self.delegate.ctx(),
            &self.infer_var_type_ids,
            &self.infer_type_vars,
            &self.infer_gen_statuses,
        ) {
            // Prefer the current goal from the search graph stack; if the
            // stack is empty, fall back to the first pending obligation in
            // the forest.  `trait_id` stays `Option` — a `DefId(0)` sentinel
            // would leak into user-facing error messages.
            let (trait_id, self_ty) = self
                .current_goal_info()
                .unwrap_or((None, self.delegate.ctx().error()));
            return Err(SolveError::Ambiguous {
                trait_id,
                self_ty,
                span: e.span(),
                num_candidates: 0,
            });
        }
        if self.forest.has_ready_deferred(self.delegate.ctx()) {
            self.forest.recycle_ready_deferred(self.delegate.ctx());
            Ok(true)
        } else {
            Ok(false)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::traits::ImplCandidate;
    use crate::hir::traits::TraitEnv;
    use crate::hir::traits::solver::SelectionContext;
    use crate::hir::traits::solver::builtins::BuiltinTraitRegistry;
    use crate::hir::traits::solver::project::ProjectionCache;
    use crate::hir::types::CrateId;
    use crate::hir::types::TypeData;

    /// Helper: a self-recursive impl `impl<T: Foo> Foo for T`.  The
    /// where-clause bound re-enters the same predicate after the impl's
    /// generic parameter is unified with the obligation's self_ty, which
    /// is what drives the solver into cycle detection.  `is_trusted`
    /// skips the orphan/overlap checks so the fixture does not need
    /// SymbolTable type bindings.
    fn make_recursive_impl(trait_id: DefId, for_type: TypeId) -> ImplCandidate<'static> {
        ImplCandidate {
            trait_id,
            for_type,
            methods: vec![],
            resolved_methods: vec![],
            assoc_tys: vec![],
            span: crate::ast::DUMMY_SPAN,
            has_auto_deref: false,
            context: vec![for_type],
            where_clause_bounds: vec![(for_type, trait_id, vec![])],
            arity: 1,
            trait_args: vec![],
        }
    }

    /// An obligation `S: Foo` (concrete self_ty) with a self-recursive impl
    /// `impl<T: Foo> Foo for T` must terminate: user traits are inductive
    /// unless declared `@coinductive` (the attribute is what
    /// `trait_is_coinductive` consults for user traits), so the cycle is
    /// reported as an overflow error.
    #[test]
    fn test_inductive_cycle_reports_overflow() {
        let mut ctx = TypeContext::new();
        let mut trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        let trait_id = DefId(42);
        let for_type = ctx.generic_param(0, crate::symbol::Symbol::intern("T"));
        let impl_cand = make_recursive_impl(trait_id, for_type);
        trait_env
            .add_impl(impl_cand, &symbols, &mut ctx, true)
            .expect("self-recursive impl must register");

        let goal_ty = ctx.struct_ty(DefId(100), vec![]);

        let result = {
            let mut selcx = SelectionContext::new(
                &mut ctx,
                &trait_env,
                &symbols,
                &builtin_registry,
                &proj_cache,
                &caller_bounds,
            );
            let mut fulfill = FulfillmentContext::new(&mut selcx);
            fulfill.register_predicate(
                Predicate::Trait {
                    trait_id,
                    self_ty: goal_ty,
                    args: vec![],
                },
                crate::ast::DUMMY_SPAN,
            );
            fulfill.evaluate_all()
        };
        assert!(
            matches!(result, Err(ref errs) if matches!(errs.as_slice(), [SolveError::Overflow { .. }])),
            "an inductive cycle must be reported as Overflow, got {:?}",
            result
        );
    }

    /// Regression for the generic-param unify direction: a goal whose
    /// self_ty is a generic param (`T0: Foo`) with the same self-recursive
    /// impl must CYCLE like the concrete case.  Previously `TypeContext::unify`
    /// bound the generic param TO the impl's fresh variable, leaving the
    /// var dangling — `SgGoalKey::from_obligation` returned `None`, cycle
    /// detection was skipped, and the goal deferred instead of overflowing.
    /// The InferVar arm now wins over the GenericParam arm (var := param,
    /// mirroring `InferenceContext::unify` and rustc).
    #[test]
    fn test_generic_param_self_ty_cycle_reports_overflow() {
        let mut ctx = TypeContext::new();
        let mut trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        let trait_id = DefId(42);
        let for_type = ctx.generic_param(0, crate::symbol::Symbol::intern("T"));
        let impl_cand = make_recursive_impl(trait_id, for_type);
        trait_env
            .add_impl(impl_cand, &symbols, &mut ctx, true)
            .expect("self-recursive impl must register");

        let goal_ty = ctx.generic_param(0, crate::symbol::Symbol::intern("T"));

        let result = {
            let mut selcx = SelectionContext::new(
                &mut ctx,
                &trait_env,
                &symbols,
                &builtin_registry,
                &proj_cache,
                &caller_bounds,
            );
            let mut fulfill = FulfillmentContext::new(&mut selcx);
            fulfill.register_predicate(
                Predicate::Trait {
                    trait_id,
                    self_ty: goal_ty,
                    args: vec![],
                },
                crate::ast::DUMMY_SPAN,
            );
            fulfill.evaluate_all()
        };
        assert!(
            matches!(result, Err(ref errs) if matches!(errs.as_slice(), [SolveError::Overflow { .. }])),
            "a generic-param self_ty cycle must be reported as Overflow, got {:?}",
            result
        );
    }

    /// The same self-recursive impl as the inductive test, but the trait is
    /// declared `@coinductive` (the user-level analog of rustc's
    /// `#[rustc_coinductive]`): the cycle is now productive — coinductive
    /// traits' impl where-clauses are "productive steps" — and the goal
    /// must resolve instead of overflowing.
    #[test]
    fn test_coinductive_cycle_resolves() {
        let mut ctx = TypeContext::new();
        let mut trait_env = TraitEnv::new();
        let mut symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        let trait_id = DefId(42);
        symbols
            .insert_trait(
                crate::symbol::Symbol::intern("Foo"),
                crate::hir::symbol::TraitBinding {
                    def_id: trait_id,
                    methods: vec![],
                    associated_types: vec![],
                    super_traits: vec![],
                    span: crate::ast::DUMMY_SPAN,
                    attributes: vec![crate::ast::Attribute {
                        name: crate::symbol::Symbol::intern("coinductive"),
                        args: vec![],
                        named_args: vec![],
                        span: crate::ast::DUMMY_SPAN,
                    }],
                    crate_id: symbols.local_crate_id,
                },
                crate::ast::DUMMY_SPAN,
            )
            .expect("trait must register");
        let for_type = ctx.generic_param(0, crate::symbol::Symbol::intern("T"));
        let impl_cand = make_recursive_impl(trait_id, for_type);
        trait_env
            .add_impl(impl_cand, &symbols, &mut ctx, true)
            .expect("self-recursive impl must register");

        let goal_ty = ctx.struct_ty(DefId(100), vec![]);

        let result = {
            let mut selcx = SelectionContext::new(
                &mut ctx,
                &trait_env,
                &symbols,
                &builtin_registry,
                &proj_cache,
                &caller_bounds,
            );
            let mut fulfill = FulfillmentContext::new(&mut selcx);
            fulfill.register_predicate(
                Predicate::Trait {
                    trait_id,
                    self_ty: goal_ty,
                    args: vec![],
                },
                crate::ast::DUMMY_SPAN,
            );
            fulfill.evaluate_all()
        };
        assert!(
            result.is_ok(),
            "a @coinductive cycle must resolve (productive steps), got {:?}",
            result
        );
    }

    /// Regression (defaulting is a last resort): an obligation whose
    /// self_ty is an Integer-kind inference variable must defer, then be
    /// resolved once defaulting binds the variable to `Int<32>` — even
    /// though defaulting now runs ONLY when the pending queue is exhausted
    /// (not at the top of every iteration).  Moving it must not lose the
    /// defer → default → recycle → resolve unblocking behavior.
    #[test]
    fn test_defaulting_unblocks_deferred_obligation() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let mut builtin_registry = BuiltinTraitRegistry::new();
        // `Predicate::Sized` addresses the builtin Sized trait via the
        // DefId(usize::MAX) sentinel (see assembly) — register it so the
        // goal resolves once its self_ty is concrete.
        builtin_registry.register(
            crate::hir::types::DefId(usize::MAX),
            &crate::symbol::Symbol::intern("Sized"),
        );
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // Integer-kind inference variable: guided by defaulting to Int<32>.
        let infer_var = ctx.alloc_infer_var(777, 0);
        let type_vars = vec![(
            TypeVariableKind::Integer,
            VarOrigin::Expression(Some(crate::ast::DUMMY_SPAN)),
        )];
        let gen_statuses = vec![GenStatus::Ungeneralized];

        let result = {
            let mut selcx = SelectionContext::new(
                &mut ctx,
                &trait_env,
                &symbols,
                &builtin_registry,
                &proj_cache,
                &caller_bounds,
            );
            let mut fulfill = FulfillmentContext::new(&mut selcx);
            fulfill.set_infer_data(&[infer_var], &type_vars, &gen_statuses);
            fulfill.register_predicate(Predicate::Sized { ty: infer_var }, crate::ast::DUMMY_SPAN);
            fulfill.evaluate_all()
        };
        assert!(
            result.is_ok(),
            "Sized<?integer> must resolve after last-resort defaulting: {:?}",
            result
        );
        // The inference variable must have been defaulted to Int<32>.
        let resolved = ctx.resolve_binding(infer_var);
        assert!(
            matches!(ctx.get(resolved), TypeData::Int { bits: 32, .. }),
            "Integer infer var must be defaulted to Int<32>, got {:?}",
            ctx.get(resolved)
        );
    }

    /// Multiple obligations stalled on the SAME inference variable must all
    /// be unblocked by one defaulting step (recycle → resolve), not just the
    /// first one to be re-evaluated.
    #[test]
    fn test_defaulting_unblocks_multiple_deferred() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let mut builtin_registry = BuiltinTraitRegistry::new();
        builtin_registry.register(
            crate::hir::types::DefId(usize::MAX),
            &crate::symbol::Symbol::intern("Sized"),
        );
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // One Integer-kind inference variable, three obligations on it.
        let infer_var = ctx.alloc_infer_var(779, 0);
        let type_vars = vec![(
            TypeVariableKind::Integer,
            VarOrigin::Expression(Some(crate::ast::DUMMY_SPAN)),
        )];
        let gen_statuses = vec![GenStatus::Ungeneralized];

        let mut selcx = SelectionContext::new(
            &mut ctx,
            &trait_env,
            &symbols,
            &builtin_registry,
            &proj_cache,
            &caller_bounds,
        );
        let mut fulfill = FulfillmentContext::new(&mut selcx);
        fulfill.set_infer_data(&[infer_var], &type_vars, &gen_statuses);
        for _ in 0..3 {
            fulfill.register_predicate(Predicate::Sized { ty: infer_var }, crate::ast::DUMMY_SPAN);
        }

        let result = fulfill.evaluate_all();
        assert!(
            result.is_ok(),
            "all three Sized<?integer> obligations must resolve after defaulting: {:?}",
            result
        );
        assert_eq!(
            fulfill.forest().deferred_count(),
            0,
            "no obligation may remain deferred once the variable is defaulted"
        );
        assert_eq!(
            fulfill.forest().pending_count(),
            0,
            "no obligation may remain pending once the variable is defaulted"
        );
        drop(fulfill);
        drop(selcx);
        let resolved = ctx.resolve_binding(infer_var);
        assert!(
            matches!(ctx.get(resolved), TypeData::Int { bits: 32, .. }),
            "Integer infer var must be defaulted to Int<32>, got {:?}",
            ctx.get(resolved)
        );
    }

    /// An obligation that stalls on an inference variable with NO infer
    /// data (defaulting unavailable) must not hang: the non-final pass
    /// converges and returns `Ok`, and the final pass reports the deferred
    /// obligation as an ambiguity error.
    #[test]
    fn test_stalled_obligation_converges_ok_nonfinal_reports_final() {
        let mut ctx = TypeContext::new();
        let trait_env = TraitEnv::new();
        let symbols = crate::hir::symbol::SymbolTable::new(CrateId(DefId(0)));
        let builtin_registry = BuiltinTraitRegistry::new();
        let proj_cache = ProjectionCache::new();
        let caller_bounds: [Predicate; 0] = [];

        // Any-kind inference variable, but NO infer data is set — defaulting
        // has nothing to work with.
        let infer_var = ctx.alloc_infer_var(778, 0);

        let mut selcx = SelectionContext::new(
            &mut ctx,
            &trait_env,
            &symbols,
            &builtin_registry,
            &proj_cache,
            &caller_bounds,
        );
        let mut fulfill = FulfillmentContext::new(&mut selcx);
        fulfill.register_predicate(Predicate::Sized { ty: infer_var }, crate::ast::DUMMY_SPAN);

        let non_final = fulfill.evaluate_all();
        assert!(
            non_final.is_ok(),
            "a stalled obligation must converge (not hang) in the non-final pass: {:?}",
            non_final
        );
        let final_result = fulfill.evaluate_all_final();
        assert!(
            matches!(final_result, Err(ref errs) if matches!(errs.as_slice(), [SolveError::Ambiguous { .. }])),
            "the final pass must report the stalled obligation as Ambiguous, got {:?}",
            final_result
        );
        // No infer data was supplied, so defaulting never ran and the
        // variable stays unbound.
        drop(fulfill);
        drop(selcx);
        assert!(
            ctx.is_infer_var(ctx.resolve_binding(infer_var)),
            "without infer data the stalled variable must remain unbound"
        );
    }
}
