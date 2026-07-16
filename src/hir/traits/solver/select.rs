use crate::hir::traits::solver::obligation::{
    BuiltinImplSource, ImplSource, Obligation, ObligationCause, ObligationCauseCode, Predicate,
    ProjectionTy, SolveError,
};
use crate::hir::traits::solver::builtins::{self, BuiltinTrait, BuiltinTraitRegistry};
use crate::hir::traits::solver::project::{self, ProjectionCache};
use crate::hir::traits::TraitEnv;
use crate::hir::types::{DefId, Subst, TypeContext, TypeData, TypeId};
use crate::hir::infer::InferenceContext;
use crate::hir::symbol::SymbolTable;
use crate::symbol::Symbol;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counter for fresh inference variables used during generic impl matching.
static GENERIC_MATCH_VAR_ID: AtomicUsize = AtomicUsize::new(2_000_000);

/// The core trait resolution engine.
///
/// Analogous to rustc's `SelectionContext`. Responsible for:
/// 1. Candidate assembly (gathering possible impls/bounds/builtins)
/// 2. Winnowing (removing ambiguous/overlapping candidates)
/// 3. Confirmation (verifying the selected candidate and producing sub-obligations)
///
/// Uses `TraitEnv` as a read-only data source for registered impls.
/// Does NOT modify `TraitEnv` — all state mutations go through `TypeContext` transactions.
pub struct SelectionContext<'a> {
    pub ctx: &'a mut TypeContext,
    pub infer: &'a mut InferenceContext,
    pub trait_env: &'a TraitEnv,
    pub symbols: &'a SymbolTable,
    pub builtin_registry: &'a BuiltinTraitRegistry,
    /// Caller bounds (from where-clauses in scope).
    pub caller_bounds: &'a [Predicate],
    /// Projection cache for associated type normalization.
    pub proj_cache: &'a ProjectionCache,
    /// Current recursion depth.
    recursion_depth: usize,
}

/// Maximum recursion depth for trait resolution before overflow.
const MAX_RECURSION_DEPTH: usize = 64;

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
    Impl {
        idx: usize,
        impl_source: ImplSource,
    },
    /// Caller-provided bound (where-clause).
    /// Stores the (self_ty, args) that were matched during assembly so that
    /// confirm_candidate can re-apply the unification in a fresh transaction.
    Param {
        self_ty: TypeId,
        args: Vec<TypeId>,
    },
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
}

impl<'a> SelectionContext<'a> {
    pub fn new(
        ctx: &'a mut TypeContext,
        infer: &'a mut InferenceContext,
        trait_env: &'a TraitEnv,
        symbols: &'a SymbolTable,
        builtin_registry: &'a BuiltinTraitRegistry,
        proj_cache: &'a ProjectionCache,
        caller_bounds: &'a [Predicate],
    ) -> Self {
        SelectionContext {
            ctx,
            infer,
            trait_env,
            symbols,
            builtin_registry,
            proj_cache,
            caller_bounds,
            recursion_depth: 0,
        }
    }

    /// Select a candidate for the given obligation.
    ///
    /// Returns the resolved `ImplSource` on success, or a `SolveError` on failure.
    pub fn select(&mut self, obligation: &Obligation) -> Result<ImplSource, SolveError> {
        if self.recursion_depth >= MAX_RECURSION_DEPTH {
            return Err(SolveError::Overflow {
                trait_id: DefId(0),
                self_ty: TypeId::from_raw(0),
                depth: self.recursion_depth,
            });
        }
        self.recursion_depth += 1;

        // ── Handle ProjectionEq / ProjectionNormalize directly ──
        // These are not trait obligations — they are resolved by looking up
        // the associated type in the impl and unifying with the target.
        // Must decrement recursion_depth before returning to avoid leaks.
        match &obligation.predicate {
            Predicate::ProjectionEq { trait_id, self_ty, assoc_name, value } => {
                self.recursion_depth -= 1;
                return self.handle_projection_eq(*trait_id, *self_ty, assoc_name, *value, &obligation.cause);
            }
            Predicate::ProjectionNormalize { projection, target } => {
                self.recursion_depth -= 1;
                return self.handle_projection_normalize(projection, *target, &obligation.cause);
            }
            _ => {}
        }

        // ── Resolve self_ty ──
        let resolved = self.resolve_obligation(obligation);

        // ── If self_ty is still an infer var, defer ──
        // The obligation cannot be resolved until the type variable is
        // bound to a concrete type.  Return Deferred so the caller can
        // retry after the type is resolved.
        if resolved.ambiguous {
            self.recursion_depth -= 1;
            return Ok(ImplSource::Deferred);
        }

        // ── Candidate assembly ──
        let mut candidates = Candidates {
            vec: Vec::new(),
            ambiguous: false,
        };

        self.assemble_candidates_from_impls(&resolved, &mut candidates);
        self.assemble_candidates_from_caller_bounds(&resolved, &mut candidates);
        self.assemble_candidates_from_builtins(&resolved, &mut candidates);
        self.assemble_candidates_from_object_ty(&resolved, &mut candidates);
        self.assemble_candidates_from_poly(&resolved, &mut candidates);

        // ── Winnowing ──
        self.winnow(&mut candidates, &resolved)?;

        self.recursion_depth -= 1;

        // ── Confirmation ──
        match candidates.vec.len() {
            0 => {
                if candidates.ambiguous {
                    Err(SolveError::Ambiguous {
                        trait_id: resolved.trait_id,
                        self_ty: resolved.self_ty,
                        span: obligation.cause.span,
                        num_candidates: 0,
                    })
                } else {
                    Err(SolveError::NotFound {
                        trait_id: resolved.trait_id,
                        self_ty: resolved.self_ty,
                        span: obligation.cause.span,
                    })
                }
            }
            1 => self.confirm_candidate(&resolved, &candidates.vec[0]),
            _ => {
                // Multiple candidates survived winnowing → ambiguity
                Err(SolveError::Ambiguous {
                    trait_id: resolved.trait_id,
                    self_ty: resolved.self_ty,
                    span: obligation.cause.span,
                    num_candidates: candidates.vec.len(),
                })
            }
        }
    }

    /// Resolve the self_ty through bindings and extract the trait predicate.
    fn resolve_obligation(&self, obligation: &Obligation) -> ResolvedObligation {
        match &obligation.predicate {
            Predicate::Trait { trait_id, self_ty, args } => {
                let resolved_self = self.ctx.resolve_binding(*self_ty);
                let resolved_args: Vec<TypeId> = args.iter().map(|a| self.ctx.resolve_binding(*a)).collect();
                let ambiguous = self.ctx.is_infer_var(resolved_self);
                ResolvedObligation {
                    trait_id: *trait_id,
                    self_ty: resolved_self,
                    args: resolved_args,
                    ambiguous,
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
                }
            }
            _ => {
                // Fallback for other predicate types (ProjectionEq, etc.)
                ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: self.ctx.error(),
                    args: vec![],
                    ambiguous: false,
                }
            }
        }
    }

    // ── Candidate assembly ──

    fn assemble_candidates_from_impls(
        &mut self,
        obligation: &ResolvedObligation,
        candidates: &mut Candidates,
    ) {
        let mut impl_count = 0;
        for (idx, impl_cand) in self.trait_env.all_impls().iter().enumerate() {
            if impl_cand.trait_id != obligation.trait_id {
                continue;
            }
            impl_count += 1;
            // Try unification inside a transaction, then ROLL BACK regardless.
            self.ctx.begin_transaction();
            let result = self.try_match_impl(idx, impl_cand, obligation);
            self.ctx.rollback_transaction();
            match result {
                Ok(impl_source) => {
                    candidates.vec.push(Candidate::Impl { idx, impl_source });
                }
                Err(_) => {}
            }
        }
        
    }

    fn try_match_impl(
        &mut self,
        cand_idx: usize,
        impl_cand: &crate::hir::traits::ImplCandidate,
        obligation: &ResolvedObligation,
    ) -> Result<ImplSource, SolveError> {
        let arity = impl_cand.arity;

        // Generate fresh infer vars for each generic param
        let mut subst = Subst::new();
        for i in 0..arity {
            let id = GENERIC_MATCH_VAR_ID.fetch_add(1, Ordering::Relaxed);
            let fresh = self.ctx.alloc_infer_var(id);
            subst.insert(i, fresh);
        }

        // Substitute the candidate's for_type with fresh infer vars.
        let substituted_for_type = self.ctx.subst(impl_cand.for_type, &subst);

        // Unify substituted for_type with obligation's self_ty
        self.ctx.unify(obligation.self_ty, substituted_for_type).map_err(|_| {
            SolveError::NotFound {
                trait_id: obligation.trait_id,
                self_ty: obligation.self_ty,
                span: crate::ast::Span::new(0, 0),
            }
        })?;

        // ── Unify trait generic args ──
        // Match the impl's trait args (e.g. `Int<32>` in `impl Add<Int<32>> for T`)
        // against the obligation's args from the where-clause bound
        // (e.g. `Add<Rhs = Int<32>>`).  Each impl trait_arg is substituted with
        // fresh infer vars so that generic params (e.g. `impl<R> Add<R> for T`)
        // are correctly matched.
        let substituted_trait_args: Vec<TypeId> = impl_cand.trait_args
            .iter()
            .map(|&arg| self.ctx.subst(arg, &subst))
            .collect();

        // Both impl and obligation must agree on the number of trait args.
        // If they differ, the impl cannot match this obligation.
        if substituted_trait_args.len() != obligation.args.len() {
            return Err(SolveError::NotFound {
                trait_id: obligation.trait_id,
                self_ty: obligation.self_ty,
                span: crate::ast::Span::new(0, 0),
            });
        }

        for (impl_arg, ob_arg) in substituted_trait_args.iter().zip(obligation.args.iter()) {
            self.ctx.unify(*impl_arg, *ob_arg).map_err(|_| {
                SolveError::Mismatch {
                    expected: *ob_arg,
                    found: *impl_arg,
                    span: crate::ast::Span::new(0, 0),
                }
            })?;
        }

        // ── Generate sub-obligations from impl's where-clause ──
        // Each bound `T: Foo` in `impl<T: Foo> Bar for T` becomes a
        // Predicate::Trait obligation after applying the substitution.
        let mut nested: Vec<Obligation> = Vec::new();
        for &(ref_self_ty, bound_trait_id, ref bound_args) in &impl_cand.where_clause_bounds {
            let substituted_self = self.ctx.subst(ref_self_ty, &subst);
            let substituted_args: Vec<TypeId> = bound_args
                .iter()
                .map(|&arg| self.ctx.subst(arg, &subst))
                .collect();
            nested.push(Obligation {
                cause: crate::hir::traits::solver::obligation::ObligationCause {
                    span: impl_cand.span,
                    code: ObligationCauseCode::ImplBound { impl_def_id: impl_cand.trait_id },
                },
                predicate: Predicate::Trait {
                    trait_id: bound_trait_id,
                    self_ty: substituted_self,
                    args: substituted_args,
                },
                recursion_depth: 0,
            });
        }

        Ok(ImplSource::UserDefined {
            cand_idx,
            subst,
            nested,
        })
    }

    fn assemble_candidates_from_caller_bounds(
        &mut self,
        obligation: &ResolvedObligation,
        candidates: &mut Candidates,
    ) {
        for bound in self.caller_bounds {
            let (trait_id, self_ty, args) = match bound {
                Predicate::Trait { trait_id, self_ty, args } => (trait_id, self_ty, Some(args)),
                Predicate::AutoTrait { trait_id, self_ty } => (trait_id, self_ty, None),
                _ => continue,
            };
            if *trait_id == obligation.trait_id {
                self.ctx.begin_transaction();

                // Unify self_ty
                let ok = self.ctx.unify(obligation.self_ty, *self_ty).is_ok();

                // Also unify trait generic args (e.g. Add<i32> vs Add<i64>)
                let args_ok = if ok {
                    if let Some(bound_args) = args {
                        if bound_args.len() == obligation.args.len() {
                            bound_args.iter().zip(obligation.args.iter())
                                .all(|(ba, oa)| self.ctx.unify(*ba, *oa).is_ok())
                        } else {
                            false
                        }
                    } else {
                        true
                    }
                } else {
                    false
                };

                // Roll back — candidate assembly must be side-effect-free.
                // confirm_candidate will re-apply the unification.
                self.ctx.rollback_transaction();
                if args_ok {
                    candidates.vec.push(Candidate::Param {
                        self_ty: *self_ty,
                        args: args.cloned().unwrap_or_default(),
                    });
                    // Do NOT return here — continue checking all bounds.
                    // Multiple matching bounds for the same trait should
                    // be collected and winnowed, producing an Ambiguous
                    // error if more than one survives.
                }
            }
        }
    }

    fn assemble_candidates_from_builtins(
        &mut self,
        obligation: &ResolvedObligation,
        candidates: &mut Candidates,
    ) {
        let builtin_kind = self.builtin_registry.lookup(obligation.trait_id);

        match builtin_kind {
            Some(BuiltinTrait::Sized) => {
                let self_ty = obligation.self_ty;
                // If the self_ty is an inference var, we don't know yet — mark ambiguous
                if self.ctx.is_infer_var(self_ty) {
                    candidates.ambiguous = true;
                } else if builtins::compute_sized(self_ty, self.ctx) {
                    candidates.vec.push(Candidate::Builtin(BuiltinImplSource::Sized));
                }
                // If unsized, no candidate is added — the obligation fails
                // (which is correct: unsized types do not satisfy `Sized`).
            }
            Some(BuiltinTrait::Copy) => {
                if builtins::compute_copy(obligation.self_ty, self.ctx) {
                    candidates.vec.push(Candidate::Builtin(BuiltinImplSource::Copy));
                }
            }
            Some(BuiltinTrait::Clone) => {
                // Clone auto-derives from Copy: if the type is Copy, it's also Clone.
                // This covers both explicit Clone impls (from from_impls) and the
                // automatic derive (SYNTAX.md § Automatic Clone for Copy Types).
                if builtins::compute_clone(obligation.self_ty, self.ctx) {
                    candidates.vec.push(Candidate::Builtin(BuiltinImplSource::Clone));
                }
            }
            Some(BuiltinTrait::Drop) => {
                // Drop is a user-implemented trait — rely on from_impls.
            }
            Some(BuiltinTrait::Default) => {
                // Default is a user-implemented trait — rely on from_impls.
            }
            Some(_) => {
                // Other builtins (Add, Sub, Eq, Ord, Deref, etc.) have no automatic
                // structural derivation — they require a user-defined impl.
                // Rely on from_impls for these.
            }
            None => {}
        }
    }

    fn assemble_candidates_from_object_ty(
        &mut self,
        obligation: &ResolvedObligation,
        candidates: &mut Candidates,
    ) {
        // If the self_ty is a dyn Trait, extract bounds from the vtable.
        if let TypeData::DynTrait { traits, .. } = self.ctx.get(obligation.self_ty) {
            for trait_id in traits {
                if *trait_id == obligation.trait_id {
                    candidates.vec.push(Candidate::Object {
                        object_trait_id: *trait_id,
                        nested: vec![],
                    });
                }
            }
        }
    }

    fn assemble_candidates_from_poly(
        &mut self,
        obligation: &ResolvedObligation,
        candidates: &mut Candidates,
    ) {
        // Poly types: `Poly { quantifiers, body }` — a boxed polymorphic value.
        // Check that the self_ty is a poly type, and record the quantifier count.
        // The actual unboxing happens in confirm_candidate inside a committed
        // transaction.
        let quantifier_count = match self.ctx.get(obligation.self_ty) {
            TypeData::Poly { quantifiers, .. } => quantifiers.len(),
            _ => return,
        };

        candidates.vec.push(Candidate::Poly {
            quantifier_count,
        });
    }

    // ── Winnowing ──

    fn winnow(
        &mut self,
        candidates: &mut Candidates,
        _obligation: &ResolvedObligation,
    ) -> Result<(), SolveError> {
        if candidates.vec.len() <= 1 {
            return Ok(());
        }

        // Sort by specificity: concrete > generic, impl > param > builtin
        candidates.vec.sort_by(|a, b| self.specificity(a, b));

        // Keep only the most specific ones
        let mut i = 1;
        while i < candidates.vec.len() {
            if self.candidate_should_be_dropped(&candidates.vec[i], &candidates.vec[0]) {
                candidates.vec.swap_remove(i);
            } else {
                i += 1;
            }
        }

        if candidates.vec.len() > 1 {
            candidates.ambiguous = true;
        }

        Ok(())
    }

    /// Order candidates by specificity (most specific first).
    fn specificity(&self, a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
        match (a, b) {
            // Param candidates are most specific (caller knows best)
            (Candidate::Param { .. }, _) => std::cmp::Ordering::Less,
            (_, Candidate::Param { .. }) => std::cmp::Ordering::Greater,
            // Impl candidates are more specific than builtins
            (Candidate::Impl { .. }, Candidate::Builtin(_)) => std::cmp::Ordering::Less,
            (Candidate::Builtin(_), Candidate::Impl { .. }) => std::cmp::Ordering::Greater,
            // Impl vs Impl: compare constructor depth of for_type.
            // A concrete type (depth ≥ 1) is more specific than a generic
            // parameter (depth 0).  Equal depths means equally specific,
            // which should be treated as ambiguous.
            (Candidate::Impl { idx: ai, .. }, Candidate::Impl { idx: bi, .. }) => {
                let a_cand = &self.trait_env.all_impls()[*ai];
                let b_cand = &self.trait_env.all_impls()[*bi];
                let a_depth = self.ctx.type_constructor_depth(a_cand.for_type);
                let b_depth = self.ctx.type_constructor_depth(b_cand.for_type);
                b_depth.cmp(&a_depth) // higher depth = more specific = Ordering::Less
            }
            // Otherwise equal
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Check if a candidate should be dropped in favor of another.
    /// Only strictly less specific candidates are dropped.  Equally specific
    /// candidates survive — they will trigger an Ambiguous error, which is
    /// the correct behaviour for overlapping impls.
    fn candidate_should_be_dropped(&self, victim: &Candidate, other: &Candidate) -> bool {
        self.specificity(victim, other) == std::cmp::Ordering::Greater
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
            &proj, self.trait_env, self.ctx, self.proj_cache, self.symbols,
        );
        match normalized {
            Some(concrete_ty) => {
                self.ctx.unify(value, concrete_ty).map_err(|_| {
                    SolveError::Mismatch {
                        expected: value,
                        found: concrete_ty,
                        span: cause.span,
                    }
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
            &proj, self.trait_env, self.ctx, self.proj_cache, self.symbols,
        );
        match normalized {
            Some(concrete_ty) => {
                self.ctx.unify(target, concrete_ty).map_err(|_| {
                    SolveError::Mismatch {
                        expected: target,
                        found: concrete_ty,
                        span: cause.span,
                    }
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

    // ── Confirmation ──

    fn confirm_candidate(
        &mut self,
        obligation: &ResolvedObligation,
        candidate: &Candidate,
    ) -> Result<ImplSource, SolveError> {
        match candidate {
            Candidate::Impl { idx, .. } => {
                // Re-apply the bindings for the winning candidate.
                // The candidate assembly phase rolled back all transactions,
                // so we must re-run the matching inside a fresh transaction
                // and commit it here.
                let impl_cand = &self.trait_env.all_impls()[*idx];
                self.ctx.begin_transaction();
                let result = self.try_match_impl(*idx, impl_cand, obligation);
                match result {
                    Ok(impl_source) => {
                        self.ctx.commit_transaction();
                        Ok(impl_source)
                    }
                    Err(e) => {
                        self.ctx.rollback_transaction();
                        Err(e)
                    }
                }
            }
            Candidate::Param { self_ty, args } => {
                // Re-apply the unification for the matched caller bound.
                // The candidate assembly phase rolled back the transaction,
                // so we must re-unify in a fresh transaction and commit.
                self.ctx.begin_transaction();
                let ok = self.ctx.unify(obligation.self_ty, *self_ty).is_ok()
                    && args.len() == obligation.args.len()
                    && args.iter().zip(obligation.args.iter())
                        .all(|(a, b)| self.ctx.unify(*a, *b).is_ok());
                if ok {
                    self.ctx.commit_transaction();
                    Ok(ImplSource::Param(vec![]))
                } else {
                    self.ctx.rollback_transaction();
                    Err(SolveError::NotFound {
                        trait_id: obligation.trait_id,
                        self_ty: obligation.self_ty,
                        span: crate::ast::Span::new(0, 0),
                    })
                }
            }
            Candidate::Builtin(kind) => {
                Ok(ImplSource::Builtin(*kind))
            }
            Candidate::Object { object_trait_id, nested } => {
                Ok(ImplSource::Object {
                    object_trait_id: *object_trait_id,
                    nested: nested.clone(),
                })
            }
            Candidate::Poly { quantifier_count } => {
                // Re-apply the allocation inside a fresh transaction.
                // The assembly phase rolled back, so the infer vars are stale.
                // We need to re-create them and commit only if the candidate wins.
                let body = match self.ctx.get(obligation.self_ty) {
                    TypeData::Poly { body, .. } => *body,
                    _ => {
                        return Err(SolveError::NotFound {
                            trait_id: obligation.trait_id,
                            self_ty: obligation.self_ty,
                            span: crate::ast::Span::new(0, 0),
                        });
                    }
                };
                self.ctx.begin_transaction();
                let mut fresh_subst = Subst::new();
                for i in 0..*quantifier_count {
                    let id = GENERIC_MATCH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                    let fresh = self.ctx.alloc_infer_var(id);
                    fresh_subst.insert(i, fresh);
                }
                let unboxed_body = self.ctx.subst(body, &fresh_subst);
                let confirmed_obligation = Obligation {
                    cause: ObligationCause {
                        span: crate::ast::Span::new(0, 0),
                        code: ObligationCauseCode::PolyUnbox { span: crate::ast::Span::new(0, 0) },
                    },
                    predicate: Predicate::Trait {
                        trait_id: obligation.trait_id,
                        self_ty: unboxed_body,
                        args: obligation.args.clone(),
                    },
                    recursion_depth: self.recursion_depth,
                };
                self.ctx.commit_transaction();
                Ok(ImplSource::Poly {
                    subst: fresh_subst,
                    nested: vec![confirmed_obligation],
                })
            }
        }
    }
}