use crate::ast::FileId;
use super::*;

impl<'input> TypeContext<'input> {
    /// Unify with a span hint: like `unify`, but records the operation
    /// span so `set_binding` can capture WHERE a GenericParam got bound
    /// (precise E104 error location).  The hint is restored afterwards —
    /// bindings created outside a span-carrying unify (e.g. by the trait
    /// solver) record no origin, and E104 falls back to the function span.
    pub(crate) fn unify_tracked(
        &mut self,
        a: TypeId,
        b: TypeId,
        span: crate::ast::Span,
    ) -> Result<TypeId, TypeError> {
        let prev = std::mem::replace(&mut self.current_unify_span, Some(span));
        let r = self.unify(a, b);
        self.current_unify_span = prev;
        r
    }

    #[must_use]
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
        // FIX: binder scope — clear stack at top-level entry.
        self.binder_stack.clear();

        // ── Transaction: capture current bindings for rollback ──
        self.begin_transaction();

        // Clear the seen-set before each top-level unification.
        self.unify_seen.clear();

        // @auto_ro / @auto_coerce: coerce `&mut U → &U` at the CallSite
        // entry point ONLY — never inside the symmetric unifier.  The
        // recursion passes `coercion_depth + 1` so this fires exactly once.
        // `resolve_binding` is required before `self.get()`: without it,
        // bound InferVars return `InferVar` from `get()` and the pattern
        // match fails silently, skipping the coercion entirely.
        let resolved_a = self.resolve_binding_compressed(a);
        let resolved_b = self.resolve_binding_compressed(b);
        let (a, b) =
            if CoercionContext::from_u8(self.current_coercion_ctx.load(AtomicOrdering::Relaxed))
                == CoercionContext::CallSite
                && (self.auto_ro || self.auto_coerce)
                && let TypeData::Ref {
                    ty: t_a,
                    mutable: m_a,
                    lifetime: l_a,
                    ..
                } = self.get(resolved_a)
                && let TypeData::Ref {
                    ty: t_b,
                    mutable: m_b,
                    lifetime: l_b,
                    ..
                } = self.get(resolved_b)
                && !*m_a
                && *m_b
            {
                let b_coerced = self.reference_with_lifetime(*t_b, false, *l_b);
                (a, b_coerced)
            } else {
                (a, b)
            };

        let result = self.unify_internal(a, b, Variance::Invariant, None, 0);

        // ── Commit or rollback ──
        match result {
            Ok(ty) => {
                self.commit_transaction();
                Ok(ty)
            }
            Err(e) => {
                self.rollback_transaction();
                Err(e)
            }
        }
    }

    /// Try to unify two types WITHOUT managing a transaction.
    ///
    /// Unlike `unify()`, this method does NOT call `begin_transaction`/
    /// `commit_transaction`/`rollback_transaction`.  The caller is responsible
    /// for managing the transaction lifecycle.  This is useful for operations
    /// like overlap checking where the caller already has an outer transaction
    /// and does not want nested transaction management.
    ///
    /// If unification succeeds, the type bindings are modified in place.
    /// The caller MUST call `rollback_transaction()` to undo them if the
    /// result is only being used for a check (like overlap detection).
    #[must_use]
    pub fn try_unify(
        &mut self,
        a: TypeId,
        b: TypeId,
        // Optional region tree for TcLevel escape checking.
        // When present, binding an InferVar from a shallower level
        // to a type from a deeper scope is rejected. (GHC §TcLevel)
        region_tree: Option<&crate::hir::infer::InferRegionTree>,
    ) -> Result<TypeId, TypeError> {
        // FIX: binder scope — clear stack at top-level entry.
        self.binder_stack.clear();
        self.unify_seen.clear();
        self.unify_internal(a, b, Variance::Invariant, region_tree, 0)
    }
    /// Stage 5a (Pistone & Tranchini roadmap): structural unification
    /// fallback — canonical-form (βη) equivalence query.
    ///
    /// When `try_unify`/`can_unify` fails structurally, the types may still
    /// be Yoneda/βη-isomorphic (e.g. `∀X.(A⇒X)⇒B ≡ B[A/X]`).  This pure
    /// query lowers both sides to canonical trees (`canonical::to_neg`) and
    /// compares them — the paper's decidability boundary applies: sound for
    /// the κ≤0 fragment (decidable βη theory), a *one-sided approximation*
    /// in the κ≤1 fragment (μ/ν present ⇒ βη axiomatization incomplete).
    ///
    /// **Gated**: returns `None` when the compilation unit has broken
    /// parametricity (`dyn Trait` reification) — equivalence-strength
    /// consumers must treat `None` as "unknown" and keep the conservative
    /// answer (fail-closed).  Diagnostics-only callers can use
    /// `neg_beta_eta_equiv` directly.
    pub fn yoneda_equiv_check(&mut self, a: TypeId, b: TypeId) -> Option<bool> {
        self.parametricity.equivalence_allowed()?;
        Some(self.neg_beta_eta_equiv(a, b))
    }

    /// Pure query: can `a` and `b` be unified?  Returns `true`/`false`
    /// without mutating the global unification state.  Uses a temporary
    /// transaction internally to discard any side effects.
    pub fn can_unify(&mut self, a: TypeId, b: TypeId) -> bool {
        // FIX: binder scope — clear stack at top-level entry.
        self.binder_stack.clear();
        self.begin_transaction();
        let saved_seen = self.unify_seen.clone();
        self.unify_seen.clear();
        let result = self.unify_internal(a, b, Variance::Invariant, None, 0);
        *&mut self.unify_seen = saved_seen;
        self.rollback_transaction();
        result.is_ok()
    }

    /// Biased unification for GADT type refinement (Peyton Jones et al.
    /// PLDI'06 §6.4).  Like `unify`, but respects the rigidity distinction
    /// between user-specified (rigid) and inferred (wobbly) types:
    ///
    /// - **Rigid vs Rigid**: both types are user-specified.  Unification
    ///   proceeds normally (strict equality check).  A mismatch is a
    ///   genuine type error.
    ///
    /// - **Rigid vs Wobbly**: the rigid type is user-committed; the wobbly
    ///   type is an unconstrained inference variable.  The wobbly variable
    ///   is BOUND to the rigid type (not an equality constraint).  This is
    ///   the "bias" — the rigid type wins.
    ///
    /// - **Wobbly vs Wobbly**: both are inference variables.  An equality
    ///   constraint is created (standard MGU behavior).
    ///
    /// This prevents unsound refinement of inferred types: a GADT `when`
    /// constraint that refines a fresh inference variable to a concrete
    /// type produces a binding (`?a := Int<32>`), not an equality
    /// constraint that could later be rolled back or conflict.
    ///
    /// # Arguments
    /// * `a` - The first type (typically the scrutinee's type argument)
    /// * `b` - The second type (typically the concrete type from the
    ///          `when` constraint)
    /// * `a_rigid` - Rigidity of type `a`
    /// * `b_rigid` - Rigidity of type `b`
    pub fn unify_biased(
        &mut self,
        a: TypeId,
        b: TypeId,
        a_rigid: Rigidity,
        b_rigid: Rigidity,
    ) -> Result<TypeId, TypeError> {
        self.binder_stack.clear();
        self.begin_transaction();
        self.unify_seen.clear();

        // Resolve bindings before inspecting rigidity and dispatching.
        // `get` reads raw arena slots; an already-bound InferVar would
        // appear fresh, causing illegal overwrites or spurious failures.
        let a = self.resolve_binding_compressed(a);
        let b = self.resolve_binding_compressed(b);

        let result = match (a_rigid, b_rigid) {
            (Rigidity::Rigid, Rigidity::Rigid) | (Rigidity::Wobbly, Rigidity::Wobbly) => {
                // Both same rigidity: standard MGU (equality constraint).
                self.unify_internal(a, b, Variance::Invariant, None, 0)
            }
            (Rigidity::Rigid, Rigidity::Wobbly) => {
                // `a` is rigid (user-specified), `b` is wobbly (inferred).
                // Bias: bind `b` to `a` if `b` is a fresh InferVar.
                if let TypeData::InferVar { .. } = self.get(b) {
                    self.bias_fresh_vars(b, a)
                } else {
                    self.unify_internal(a, b, Variance::Invariant, None, 0)
                }
            }
            (Rigidity::Wobbly, Rigidity::Rigid) => {
                // `a` is wobbly (inferred), `b` is rigid (user-specified).
                // Bias: bind `a` to `b` if `a` is a fresh InferVar.
                if let TypeData::InferVar { .. } = self.get(a) {
                    self.bias_fresh_vars(a, b)
                } else {
                    self.unify_internal(a, b, Variance::Invariant, None, 0)
                }
            }
        };

        match result {
            Ok(ty) => {
                self.commit_transaction();
                Ok(ty)
            }
            Err(e) => {
                self.rollback_transaction();
                Err(e)
            }
        }
    }

    /// Bind a fresh inference variable `var` to a concrete type `target`,
    /// walking into compound types to bind nested fresh variables as well
    /// (Peyton Jones et al. PLDI'06 §6.4, biased MGU).
    ///
    /// This is the core of the rigid/wobbly distinction: when a fresh
    /// inference variable appears in a position that the GADT refinement
    /// constrains, this method binds it directly to the target type
    /// rather than creating an equality constraint.  The binding is
    /// recorded in the transaction undo-log via `set_binding`.
    ///
    /// For compound types (tuples, ADTs, refs, etc.), the method
    /// recursively biases nested fresh variables toward the corresponding
    /// parts of the target.  If the target is not a compatible compound
    /// type, the top-level binding is attempted (which may fail if the
    /// types are structurally incompatible).
    pub(crate) fn bias_fresh_vars(
        &mut self,
        var: TypeId,
        target: TypeId,
    ) -> Result<TypeId, TypeError> {
        // Resolve aliases before occurs-check and structural bias walk.
        // `get` reads raw arena slots; an aliased InferVar would be
        // treated as fresh, causing incorrect bias or occurs-check failures.
        let var = self.resolve_binding_compressed(var);
        let target = self.resolve_binding_compressed(target);

        // Occurs check: prevent recursive types.
        if self.occurs_check(var, target) {
            return Err(TypeError::RecursiveType {
                ty: var,
                span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
            });
        }
        // GADT skolem escape guard: don't bind a variable to a type
        // containing a GADT existential skolem (witness would escape).
        if self.contains_gadt_skolem(target) {
            return Err(TypeError::SkolemEscape {
                var_id: usize::MAX,
                var_level: Self::GADT_SKOLEM_UNIVERSE,
                current_level: 0,
                span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
            });
        }
        // Structural bias: walk into compound types to bind nested
        // fresh variables.
        let var_data = self.get(var);
        let target_data = self.get(target);
        match (&*var_data, &*target_data) {
            // Both tuples of same length: bias each element pair.
            (
                TypeData::Tuple { elems: var_elems },
                TypeData::Tuple {
                    elems: target_elems,
                },
            ) if var_elems.len() == target_elems.len() => {
                let pairs: Vec<(TypeId, TypeId)> = var_elems
                    .iter()
                    .zip(target_elems.iter())
                    .map(|(&a, &b)| (a, b))
                    .collect();
                for (ve, te) in pairs {
                    if self.is_infer_var(ve) {
                        self.bias_fresh_vars(ve, te)?;
                    } else {
                        self.unify_internal(ve, te, Variance::Invariant, None, 0)?;
                    }
                }
            }
            // Both ADTs with same def_id and same arg count: bias each arg.
            (
                TypeData::Adt {
                    def_id: vd1,
                    args: var_args,
                    ..
                },
                TypeData::Adt {
                    def_id: vd2,
                    args: target_args,
                    ..
                },
            ) if vd1 == vd2 && var_args.len() == target_args.len() => {
                let pairs: Vec<(TypeId, TypeId)> = var_args
                    .iter()
                    .zip(target_args.iter())
                    .map(|(&a, &b)| (a, b))
                    .collect();
                for (va, ta) in pairs {
                    if self.is_infer_var(va) {
                        self.bias_fresh_vars(va, ta)?;
                    } else {
                        self.unify_internal(va, ta, Variance::Invariant, None, 0)?;
                    }
                }
            }
            // Both arrays of same size: bias element.
            (TypeData::Array { elem: ve, size: vs }, TypeData::Array { elem: te, size: ts })
                if vs == ts =>
            {
                if self.is_infer_var(*ve) {
                    self.bias_fresh_vars(*ve, *te)?;
                } else {
                    self.unify_internal(*ve, *te, Variance::Invariant, None, 0)?;
                }
            }
            // Both slices: bias element.
            (TypeData::Slice { elem: ve }, TypeData::Slice { elem: te }) => {
                if self.is_infer_var(*ve) {
                    self.bias_fresh_vars(*ve, *te)?;
                } else {
                    self.unify_internal(*ve, *te, Variance::Invariant, None, 0)?;
                }
            }
            // Both refs (same mutability): bias inner type.
            (
                TypeData::Ref {
                    ty: ve,
                    mutable: vm,
                    ..
                },
                TypeData::Ref {
                    ty: te,
                    mutable: tm,
                    ..
                },
            ) if vm == tm => {
                if self.is_infer_var(*ve) {
                    self.bias_fresh_vars(*ve, *te)?;
                } else {
                    self.unify_internal(*ve, *te, Variance::Invariant, None, 0)?;
                }
            }
            // Both pointers: bias inner type.
            (TypeData::Pointer { ty: ve }, TypeData::Pointer { ty: te }) => {
                if self.is_infer_var(*ve) {
                    self.bias_fresh_vars(*ve, *te)?;
                } else {
                    self.unify_internal(*ve, *te, Variance::Invariant, None, 0)?;
                }
            }
            // Both Fns with same param count: bias params and return.
            (
                TypeData::Fn {
                    params: vp,
                    ret: vr,
                },
                TypeData::Fn {
                    params: tp,
                    ret: tr,
                },
            ) if vp.len() == tp.len() => {
                let ret_var = *vr;
                let ret_target = *tr;
                let pairs: Vec<(TypeId, TypeId)> =
                    vp.iter().zip(tp.iter()).map(|(&a, &b)| (a, b)).collect();
                for (ve, te) in pairs {
                    if self.is_infer_var(ve) {
                        self.bias_fresh_vars(ve, te)?;
                    } else {
                        self.unify_internal(ve, te, Variance::Invariant, None, 0)?;
                    }
                }
                if self.is_infer_var(ret_var) {
                    self.bias_fresh_vars(ret_var, ret_target)?;
                } else {
                    self.unify_internal(ret_var, ret_target, Variance::Invariant, None, 0)?;
                }
            }
            // Fallback: direct binding (covers leaf types and
            // structurally incompatible compound types).
            _ => {}
        }
        // Final binding: var → target (only if var is a free InferVar).
        // When var is a compound type (Tuple, Adt, etc.), the recursive
        // match arms above already bound the nested inference variables.
        // Attempting set_binding on a compound TypeId fails — skip it.
        if self.is_infer_var(var) {
            if !self.set_binding(var, target) {
                return Err(TypeError::Mismatch {
                    expected: var,
                    found: target,
                    span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                });
            }
        }
        Ok(target)
    }

    fn variance_tag(v: Variance) -> u8 {
        match v {
            Variance::Invariant => 0,
            Variance::Covariant => 1,
            Variance::Contravariant => 2,
        }
    }

    /// Snapshot the `unify_seen` cycle-detection cache so a transactional
    /// query can restore it afterwards (same discipline as `can_unify`).
    pub(crate) fn save_unify_seen(&self) -> std::collections::HashSet<(TypeId, TypeId, u8)> {
        self.unify_seen.clone()
    }

    /// Restore a previously saved `unify_seen` cache snapshot.
    pub(crate) fn restore_unify_seen(
        &mut self,
        saved: std::collections::HashSet<(TypeId, TypeId, u8)>,
    ) {
        self.unify_seen = saved;
    }

    /// Whether a type contains a GADT existential skolem anywhere in its
    /// structure (not just at the top level).  Used by variable-binding
    /// unification rules to prevent an arm-local witness from escaping
    /// inside a compound type (e.g. `GenericParam ~ [S]`).
    pub(crate) fn contains_gadt_skolem(&self, ty: TypeId) -> bool {
        // Follow inference aliases to detect hidden GADT skolems.
        // `get` reads raw arena slots; an InferVar unified with a
        // SkolemVar would appear as InferVar, allowing the skolem to
        // bypass the escape check and leak into the outer context.
        let ty = self.resolve_binding(ty);

        match self.get(ty) {
            TypeData::SkolemVar { universe_num, .. }
                if *universe_num == Self::GADT_SKOLEM_UNIVERSE =>
            {
                true
            }
            // Non-GADT skolems (HRTB / higher-ranked, different universe)
            // are NOT existential witnesses — do not over-approximate them.
            TypeData::SkolemVar { .. } => false,
            TypeData::Slice { elem } => self.contains_gadt_skolem(*elem),
            TypeData::Ref { ty: inner, .. } => self.contains_gadt_skolem(*inner),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.contains_gadt_skolem(e)),
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.contains_gadt_skolem(a)),
            TypeData::Array { elem, .. } => self.contains_gadt_skolem(*elem),
            TypeData::Pointer { ty: inner } => self.contains_gadt_skolem(*inner),
            TypeData::Ptr { size, pointee } => {
                self.contains_gadt_skolem(*size) || self.contains_gadt_skolem(*pointee)
            }
            TypeData::Fn { params, ret, .. } => {
                params.iter().any(|&p| self.contains_gadt_skolem(p))
                    || self.contains_gadt_skolem(*ret)
            }
            TypeData::Exists { base, .. } => self.contains_gadt_skolem(*base),
            TypeData::Forall { body, .. } => self.contains_gadt_skolem(*body),
            TypeData::AssociatedType { self_ty, .. } => self.contains_gadt_skolem(*self_ty),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.contains_gadt_skolem(a))
            }
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                self.contains_gadt_skolem(*body)
            }
            TypeData::Poly { body, .. } => self.contains_gadt_skolem(*body),
            // Leaf / closed types never contain a GADT skolem.
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Unit
            | TypeData::Never
            | TypeData::Error
            | TypeData::GenericParam { .. }
            | TypeData::InferVar { .. }
            | TypeData::Rational { .. }
            | TypeData::Regex { .. } => false,
            // Conservative: unknown forms may contain a skolem.
            _ => true,
        }
    }

    /// Whether the type contains a GenericParam that the current GADT
    /// context cannot refine (no active `ParamRefinement` fact and no
    /// binding).  `get` resolves bindings + facts first, so this arm only
    /// fires for a COMPLETELY unbound GenericParam — one the in-arm unify
    /// would bind into the global table (seal).  Used by the seal
    /// discharge guard: a compound expected/body type with an unrefined
    /// interior GenericParam (e.g. an inert existential witness, or an
    /// unrelated parameter) must skip the in-arm discharge.
    pub(crate) fn type_contains_unrefined_generic_param(&self, ty: TypeId) -> bool {
        // Follow bindings first: a bound GenericParam is no longer
        // "unrefined" even if the arena still shows the raw param.
        let resolved = self.resolve_binding(ty);
        match self.get(resolved) {
            TypeData::GenericParam { .. } => true,
            TypeData::Slice { elem } => self.type_contains_unrefined_generic_param(*elem),
            TypeData::Ref { ty: inner, .. } => self.type_contains_unrefined_generic_param(*inner),
            TypeData::Tuple { elems } => elems
                .iter()
                .any(|&e| self.type_contains_unrefined_generic_param(e)),
            TypeData::Adt { args, .. } => args
                .iter()
                .any(|&a| self.type_contains_unrefined_generic_param(a)),
            TypeData::Array { elem, .. } => self.type_contains_unrefined_generic_param(*elem),
            TypeData::Pointer { ty: inner } => self.type_contains_unrefined_generic_param(*inner),
            TypeData::Ptr { size, pointee } => {
                self.type_contains_unrefined_generic_param(*size)
                    || self.type_contains_unrefined_generic_param(*pointee)
            }
            TypeData::Fn { params, ret, .. } => {
                params
                    .iter()
                    .any(|&p| self.type_contains_unrefined_generic_param(p))
                    || self.type_contains_unrefined_generic_param(*ret)
            }
            TypeData::Exists { base, .. } => self.type_contains_unrefined_generic_param(*base),
            TypeData::Forall { body, .. } => self.type_contains_unrefined_generic_param(*body),
            TypeData::AssociatedType { self_ty, .. } => {
                self.type_contains_unrefined_generic_param(*self_ty)
            }
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.type_contains_unrefined_generic_param(a)),
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                self.type_contains_unrefined_generic_param(*body)
            }
            TypeData::Poly { body, .. } => self.type_contains_unrefined_generic_param(*body),
            // Leaf / closed types contain no GenericParam.
            _ => false,
        }
    }

    /// Internal unification with variance-aware subtyping.
    /// Recursively decomposes compound types and unifies sub-components
    /// according to the given variance.
    ///
    /// Variance propagation rules:
    /// - Invariant: all sub-components unified with Invariant (strict equality)
    /// - Covariant (T <: U): sub-components in covariant positions keep Covariant,
    ///   those in contravariant positions flip to Contravariant
    /// - Contravariant (T :> U): sub-components in covariant positions flip to
    ///   Contravariant, those in contravariant positions flip to Covariant
    fn unify_internal(
        &mut self,
        a: TypeId,
        b: TypeId,
        variance: Variance,
        region_tree: Option<&crate::hir::infer::InferRegionTree>,
        coercion_depth: usize,
    ) -> Result<TypeId, TypeError> {
        // Resolve to canonical roots before caching and unification.
        // This restores the union-find invariant: `get_arc` reads raw
        // arena slots and does NOT follow binding chains, so operating
        // on unresolved TypeIds would break transitive equality
        // propagation and fragment the cycle-detection cache.
        let a = self.resolve_binding_compressed(a);
        let b = self.resolve_binding_compressed(b);

        // ── Cycle detection, NOT a result cache ─────────────────────
        // `unify_seen` records which (a, b, variance) pairs are already
        // being visited in the CURRENT recursive descent, so that cyclic
        // type structures (e.g. recursive ADTs) terminate instead of
        // looping forever.  It deliberately does NOT memoize the outcome
        // of a unification: it holds no "success/failure" result, so a
        // probe inside a transaction (see `can_unify` /
        // `is_gadt_variant_reachable`) cannot poison later real checks —
        // entries are removed on error below so a failed attempt can be
        // retried after rollback.
        let tag = Self::variance_tag(variance);
        let key = (a, b, tag);
        if !self.unify_seen.insert(key) {
            // Already visited this pair — assume success to break cycles.
            return Ok(a);
        }

        let result = self.unify_internal_impl(a, b, variance, region_tree, coercion_depth);

        // On error, remove the cache entry so future attempts can retry.
        if result.is_err() {
            self.unify_seen.remove(&key);
        }
        result
    }

    /// The actual unification logic, called by `unify_internal` which wraps
    /// it with cache management.
    fn unify_internal_impl(
        &mut self,
        a: TypeId,
        b: TypeId,
        variance: Variance,
        region_tree: Option<&crate::hir::infer::InferRegionTree>,
        coercion_depth: usize,
    ) -> Result<TypeId, TypeError> {
        // Fail-closed: a NONE TypeId is the error-path sentinel (`HirExpr::Error`
        // reports `ty() == TypeId::NONE`).  It must never reach arena indexing —
        // the originating error was already reported, so the unify resolves to
        // the non-NONE side instead of panicking (an ICE on user code).
        if a == TypeId::NONE {
            return Ok(b);
        }
        if b == TypeId::NONE {
            return Ok(a);
        }
        let data_a = self.get_arc(a);
        let data_b = self.get_arc(b);

        // Reflexivity fast path: `unify(T, T)` always succeeds for the
        // SAME TypeId, including a GADT existential skolem unified with
        // itself.  This guard runs BEFORE the match below, so the rigid
        // GADT-skolem arms (which reject skolem vs. anything-else) never
        // see an identical pair — the `(SkolemVar, SkolemVar)` equality
        // arm is not shadowed and same-skolem reflexivity is preserved.
        if *data_a == *data_b {
            return Ok(a);
        }

        match (&*data_a, &*data_b) {
            (TypeData::Error, _) => Ok(b),
            (_, TypeData::Error) => Ok(a),
            (
                TypeData::GenericParam { index: i1, .. },
                TypeData::GenericParam { index: i2, .. },
            ) if i1 == i2 => Ok(a),
            // GADT existential skolems are RIGID: a GADT skolem (sentinelled
            // by `GADT_SKOLEM_UNIVERSE`) can only unify with ITSELF.  It must
            // never be bound to, or bound by, a `GenericParam`, an outer
            // `InferVar`, or a concrete type — otherwise the arm-local
            // existential witness would escape through the outer binder
            // (SYNTAX.md §"Existential Quantification": "prevented from
            // leaking into the surrounding context").  Placed BEFORE the
            // `(_, GenericParam)` / `(InferVar, _)` binding arms so that a
            // GADT skolem never falls through to a variable-binding rule.
            (
                TypeData::SkolemVar {
                    universe_num: u1, ..
                },
                _,
            ) if *u1 == Self::GADT_SKOLEM_UNIVERSE => {
                return Err(TypeError::SkolemEscape {
                    var_id: usize::MAX,
                    var_level: *u1,
                    current_level: 0,
                    span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                });
            }
            (
                _,
                TypeData::SkolemVar {
                    universe_num: u2, ..
                },
            ) if *u2 == Self::GADT_SKOLEM_UNIVERSE => {
                return Err(TypeError::SkolemEscape {
                    var_id: usize::MAX,
                    var_level: *u2,
                    current_level: 0,
                    span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                });
            }
            (TypeData::GenericParam { index, .. }, _) => {
                // FIX: binder scope guard — if this GenericParam is currently
                // bound in an active binder, unifying it with a foreign type
                // would be a scope escape.
                if self.binder_stack.contains(index) {
                    return Err(TypeError::ScopeEscape {
                        index: *index,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                // A GADT skolem must not be bound into a variable through a
                // compound type (e.g. `GenericParam ~ [S]`): the witness
                // would escape the arm (SYNTAX.md §"Existential
                // Quantification").
                if self.contains_gadt_skolem(b) {
                    return Err(TypeError::SkolemEscape {
                        var_id: usize::MAX,
                        var_level: Self::GADT_SKOLEM_UNIVERSE,
                        current_level: 0,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }
            (_, TypeData::GenericParam { index, .. }) => {
                // FIX: binder scope guard — symmetric.
                if self.binder_stack.contains(index) {
                    return Err(TypeError::ScopeEscape {
                        index: *index,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if self.contains_gadt_skolem(a) {
                    return Err(TypeError::SkolemEscape {
                        var_id: usize::MAX,
                        var_level: Self::GADT_SKOLEM_UNIVERSE,
                        current_level: 0,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if !self.set_binding(b, a) {
                    return Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(a)
            }
            // SkolemVar: identical skolems are equal; different skolems cannot unify.
            (
                TypeData::SkolemVar {
                    id: id1,
                    universe_num: u1,
                },
                TypeData::SkolemVar {
                    id: id2,
                    universe_num: u2,
                },
            ) if id1 == id2 && u1 == u2 => Ok(a),
            (TypeData::InferVar { .. }, _) => {
                // TcLevel escape check (GHC §TcLevel): if the InferVar is
                // from a shallower region level than the current level,
                // binding it to a type from a deeper scope is forbidden.
                if let Some(rt) = region_tree
                    && let TypeData::InferVar { id, .. } = self.get(a)
                {
                    let var_region = rt.region_of_var(*id);
                    let var_level = rt.get_level(var_region);
                    let current = rt.current;
                    let cur_level = rt.get_level(current);
                    if var_level < cur_level {
                        return Err(TypeError::SkolemEscape {
                            var_id: *id,
                            var_level,
                            current_level: cur_level,
                            span: (*&self.current_unify_span)
                                .unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                        });
                    }
                }
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                // A GADT skolem must not be bound into an InferVar through
                // a compound type (arm-local witness escaping the arm).
                if self.contains_gadt_skolem(b) {
                    return Err(TypeError::SkolemEscape {
                        var_id: usize::MAX,
                        var_level: Self::GADT_SKOLEM_UNIVERSE,
                        current_level: 0,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                // Per-variable universe escape check (rustc `can_define`):
                // binding a type that CONTAINS a higher-universe InferVar
                // into this var lets a forall-introduced variable escape.
                // The solver-local path checks this; the checker-global
                // path must too (canonical instantiation introduces
                // high-universe vars that flow back through this unify).
                if let TypeData::InferVar { universe, .. } = self.get(a)
                    && ty_contains_foreign_universe(self, b, *universe)
                {
                    return Err(TypeError::SkolemEscape {
                        var_id: usize::MAX,
                        var_level: *universe,
                        current_level: 0,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }
            (_, TypeData::InferVar { .. }) => {
                // Symmetric TcLevel escape check (GHC §TcLevel): if the
                // InferVar `b` is from a shallower region level than the
                // current level, binding it to a type from a deeper scope
                // is forbidden.  Mirrors the check in the (InferVar, _) arm.
                if let Some(rt) = region_tree
                    && let TypeData::InferVar { id, .. } = self.get(b)
                {
                    let var_region = rt.region_of_var(*id);
                    let var_level = rt.get_level(var_region);
                    let current = rt.current;
                    let cur_level = rt.get_level(current);
                    if var_level < cur_level {
                        return Err(TypeError::SkolemEscape {
                            var_id: *id,
                            var_level,
                            current_level: cur_level,
                            span: (*&self.current_unify_span)
                                .unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                        });
                    }
                }
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                // The FOREIGN-UNIVERSE guard (mirrors the `(InferVar, _)`
                // arm): binding a type that CONTAINS a higher-universe
                // InferVar into this variable would let forall-introduced
                // variables escape.  (The GADT-skolem containment check is
                // NOT mirrored here: the REGION-LEVEL arm scope — the
                // TcLevel check above — already rejects a binding into an
                // OUTER (shallower) variable, which is precisely the
                // existential-witness escape through a compound type
                // (`unify(&[S], outer_var)`: the outer var's region level
                // is below the current arm's).  An arm-LOCAL unwrap
                // (`MkA(MkB(s))` binding the witness to an arm-fresh
                // variable, region level >= current) is legal — a naive
                // `contains_gadt_skolem` mirror cannot tell these apart
                // (all GADT skolems share GADT_SKOLEM_UNIVERSE) and
                // rejects legal same-name nested existential patterns.)
                if let TypeData::InferVar { universe, .. } = self.get(b)
                    && ty_contains_foreign_universe(self, a, *universe)
                {
                    return Err(TypeError::SkolemEscape {
                        var_id: usize::MAX,
                        var_level: *universe,
                        current_level: 0,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                if !self.set_binding(b, a) {
                    return Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(a)
            }

            // ── Compound types: same variant, recursive sub-component unification ──

            // Adt (struct/enum): same def_id, same args length, unify args pairwise (invariant).
            (
                TypeData::Adt {
                    kind: _,
                    def_id: d1,
                    args: a1,
                },
                TypeData::Adt {
                    kind: _,
                    def_id: d2,
                    args: a2,
                },
            ) if d1 == d2 && a1.len() == a2.len() => {
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify_internal(
                        *t1,
                        *t2,
                        Variance::Invariant,
                        region_tree,
                        coercion_depth + 1,
                    )?;
                }
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Tuple: same length, elements are COVARIANT
            (TypeData::Tuple { elems: e1 }, TypeData::Tuple { elems: e2 })
                if e1.len() == e2.len() =>
            {
                let elem_variance = variance.xform(Variance::Covariant);
                for (t1, t2) in e1.iter().zip(e2.iter()) {
                    self.unify_internal(*t1, *t2, elem_variance, region_tree, coercion_depth + 1)?;
                }
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Function: params are CONTRAVARIANT, return is COVARIANT
            (
                TypeData::Fn {
                    params: p1,
                    ret: r1,
                },
                TypeData::Fn {
                    params: p2,
                    ret: r2,
                },
            ) if p1.len() == p2.len() => {
                let param_variance = variance.xform(Variance::Contravariant);
                for (t1, t2) in p1.iter().zip(p2.iter()) {
                    self.unify_internal(*t1, *t2, param_variance, region_tree, coercion_depth + 1)?;
                }
                let ret_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*r1, *r2, ret_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Array: same size, element is COVARIANT
            (TypeData::Array { elem: e1, size: s1 }, TypeData::Array { elem: e2, size: s2 })
                if s1 == s2 =>
            {
                let elem_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*e1, *e2, elem_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Slice: element is COVARIANT
            (TypeData::Slice { elem: e1 }, TypeData::Slice { elem: e2 }) => {
                let elem_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*e1, *e2, elem_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Ref: pointee is INVARIANT.  Mutability must match exactly in
            // all variance positions — the `&mut T → &T` coercion is handled
            // at the `unify()` entry point under CallSite context only.
            (
                TypeData::Ref {
                    ty: t1,
                    mutable: m1,
                    lifetime: l1,
                    ..
                },
                TypeData::Ref {
                    ty: t2,
                    mutable: m2,
                    lifetime: l2,
                    ..
                },
            ) => {
                // Explicit-lifetime consistency (SYNTAX.md §Explicit
                // Lifetime Parameters — "verified by the borrow checker;
                // mismatches cause compile errors"): `&'a T` unifies with
                // `&'b T` only when the annotations AGREE (or either side
                // is elided).  The region solver verifies `'a: 'b`
                // outlives at the signature level; at the UNIFICATION
                // site two different explicit regions are a mismatch
                // (rustc: "lifetime mismatch").
                if let (Some(l1), Some(l2)) = (l1, l2)
                    && l1 != l2
                {
                    // Region SUBTYPE collection (symmetric with the
                    // subtype Ref arm — rustc `make_subregion(b, a)`, the
                    // covariance constraint `'a: 'b`): when the checker
                    // has enabled per-signature collection, record the
                    // pair and continue (the region solver decides
                    // satisfiability against the `where 'a: 'b`
                    // predicates); otherwise two different explicit
                    // regions are a unification mismatch (rustc:
                    // "lifetime mismatch").
                    //
                    // ORIENTATION: `unify` is called as
                    // `unify_with(expected, actual)` — `l1` is the
                    // EXPECTED region, `l2` the ACTUAL one.  The required
                    // outlives for `&'actual T` flowing into an
                    // `&'expected T` position is `actual : expected`, so
                    // record `(l2, l1)` (the subtype arm records
                    // `(sub, sup)` — its l1 IS the sub side, already
                    // correct).
                    if self.region_subtype_collect {
                        self.region_subtype_outlives.push((*l2, *l1));
                    } else {
                        return Err(TypeError::Mismatch {
                            expected: b,
                            found: a,
                            span: (*&self.current_unify_span)
                                .unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                        });
                    }
                }
                let allow_mutable_coerce = match variance {
                    // `@auto_ro` coercion is handled at the `unify()` entry
                    // point (CallSite context only).  Inside the symmetric
                    // unifier we require an EXACT mutability match in every
                    // variance position.  The old covariant `&mut T → &T`
                    // permission was unsound and has been removed.
                    _ => *m1 == *m2,
                };
                if !allow_mutable_coerce {
                    return Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    });
                }
                let ty_variance = variance.xform(Variance::Invariant);
                self.unify_internal(*t1, *t2, ty_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Pointer: INVARIANT (per compute_variance_edges signing it sign: 0).
            // While some languages treat raw pointers as covariant, this design
            // conservatively marks them invariant for type safety.
            (TypeData::Pointer { ty: t1 }, TypeData::Pointer { ty: t2 }) => {
                let ty_variance = variance.xform(Variance::Invariant);
                self.unify_internal(*t1, *t2, ty_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Ptr: invariant for safety
            (
                TypeData::Ptr {
                    size: s1,
                    pointee: p1,
                },
                TypeData::Ptr {
                    size: s2,
                    pointee: p2,
                },
            ) => {
                self.unify_internal(
                    *s1,
                    *s2,
                    Variance::Invariant,
                    region_tree,
                    coercion_depth + 1,
                )?;
                self.unify_internal(
                    *p1,
                    *p2,
                    Variance::Invariant,
                    region_tree,
                    coercion_depth + 1,
                )?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Coproduct: same length, alternatives COVARIANT
            (
                TypeData::Coproduct { alternatives: a1 },
                TypeData::Coproduct { alternatives: a2 },
            ) if a1.len() == a2.len() => {
                let alt_variance = variance.xform(Variance::Covariant);
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify_internal(*t1, *t2, alt_variance, region_tree, coercion_depth + 1)?;
                }
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Forall: α-convert then COVARIANT body
            (
                TypeData::Forall {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Forall {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) => {
                let body_variance = variance.xform(Variance::Covariant);
                // FIX: binder scope — push binder index before recursing into body.
                let result = if *pi1 != *pi2 {
                    // α-conversion with capture avoidance: rename BOTH bodies
                    // to a FRESH index that cannot appear free in either body.
                    let fresh_idx = self.fresh_param_index();
                    let fresh_gp = self.generic_param(fresh_idx, *pn2);
                    let b1_renamed = self.replace_generic(*b1, *pi1, fresh_gp);
                    let b2_renamed = self.replace_generic(*b2, *pi2, fresh_gp);
                    self.binder_stack.push(fresh_idx);
                    let res = self.unify_internal(
                        b1_renamed,
                        b2_renamed,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                } else {
                    self.binder_stack.push(*pi1);
                    let res = self.unify_internal(
                        *b1,
                        *b2,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                };
                result?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Exists: α-convert then COVARIANT base
            (
                TypeData::Exists {
                    param_index: pi1,
                    name: n1,
                    base: b1,
                },
                TypeData::Exists {
                    param_index: pi2,
                    name: n2,
                    base: b2,
                },
            ) => {
                let base_variance = variance.xform(Variance::Covariant);
                // FIX: binder scope — push binder index before recursing into base.
                let result = if *pi1 != *pi2 {
                    let fresh_idx = self.fresh_param_index();
                    let fresh_gp = self.generic_param(fresh_idx, *n2);
                    let b1_renamed = self.replace_generic(*b1, *pi1, fresh_gp);
                    let b2_renamed = self.replace_generic(*b2, *pi2, fresh_gp);
                    self.binder_stack.push(fresh_idx);
                    let res = self.unify_internal(
                        b1_renamed,
                        b2_renamed,
                        base_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                } else {
                    self.binder_stack.push(*pi1);
                    let res = self.unify_internal(
                        *b1,
                        *b2,
                        base_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                };
                result?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Poly: α-convert quantifiers then COVARIANT body
            (
                TypeData::Poly {
                    quantifiers: q1,
                    body: b1,
                },
                TypeData::Poly {
                    quantifiers: q2,
                    body: b2,
                },
            ) if q1.len() == q2.len() => {
                let body_variance = variance.xform(Variance::Covariant);
                // α-conversion with capture avoidance: rename BOTH sides to
                // fresh indices for each mismatched quantifier.
                let mut b1_renamed = *b1;
                let mut b2_renamed = *b2;
                // FIX: binder scope — collect indices that are in scope for the body.
                let mut scope_indices = Vec::with_capacity(q1.len());
                for ((i1, _), (i2, pn2)) in q1.iter().zip(q2.iter()) {
                    if i1 != i2 {
                        let fresh_idx = self.fresh_param_index();
                        let fresh_gp = self.generic_param(fresh_idx, *pn2);
                        b1_renamed = self.replace_generic(b1_renamed, *i1, fresh_gp);
                        b2_renamed = self.replace_generic(b2_renamed, *i2, fresh_gp);
                        scope_indices.push(fresh_idx);
                    } else {
                        scope_indices.push(*i1);
                    }
                }
                // Push all scope indices (in order).
                for &idx in &scope_indices {
                    self.binder_stack.push(idx);
                }
                let res = self.unify_internal(
                    b1_renamed,
                    b2_renamed,
                    body_variance,
                    region_tree,
                    coercion_depth + 1,
                );
                // Pop in reverse order.
                for _ in 0..scope_indices.len() {
                    self.binder_stack.pop();
                }
                res?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Mu: α-convert then COVARIANT body
            (
                TypeData::Mu {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Mu {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) => {
                let body_variance = variance.xform(Variance::Covariant);
                // FIX: binder scope — push binder index before recursing into body.
                let result = if *pi1 != *pi2 {
                    let fresh_idx = self.fresh_param_index();
                    let fresh_gp = self.generic_param(fresh_idx, *pn2);
                    let b1_renamed = self.replace_generic(*b1, *pi1, fresh_gp);
                    let b2_renamed = self.replace_generic(*b2, *pi2, fresh_gp);
                    self.binder_stack.push(fresh_idx);
                    let res = self.unify_internal(
                        b1_renamed,
                        b2_renamed,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                } else {
                    self.binder_stack.push(*pi1);
                    let res = self.unify_internal(
                        *b1,
                        *b2,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                };
                result?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Nu: α-convert with capture avoidance then COVARIANT body
            (
                TypeData::Nu {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Nu {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) => {
                let body_variance = variance.xform(Variance::Covariant);
                // FIX: binder scope — push binder index before recursing into body.
                let result = if *pi1 != *pi2 {
                    let fresh_idx = self.fresh_param_index();
                    let fresh_gp = self.generic_param(fresh_idx, *pn2);
                    let b1_renamed = self.replace_generic(*b1, *pi1, fresh_gp);
                    let b2_renamed = self.replace_generic(*b2, *pi2, fresh_gp);
                    self.binder_stack.push(fresh_idx);
                    let res = self.unify_internal(
                        b1_renamed,
                        b2_renamed,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                } else {
                    self.binder_stack.push(*pi1);
                    let res = self.unify_internal(
                        *b1,
                        *b2,
                        body_variance,
                        region_tree,
                        coercion_depth + 1,
                    );
                    self.binder_stack.pop();
                    res
                };
                result?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // Rational: same int_bits and frac_bits (invariant)
            (
                TypeData::Rational {
                    int_bits: i1,
                    frac_bits: f1,
                },
                TypeData::Rational {
                    int_bits: i2,
                    frac_bits: f2,
                },
            ) if i1 == i2 && f1 == f2 => {
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // DynTrait: same trait list (invariant)
            (TypeData::DynTrait { traits: t1 }, TypeData::DynTrait { traits: t2 }) if t1 == t2 => {
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // AssociatedType: same trait_id + name, self_ty is COVARIANT
            (
                TypeData::AssociatedType {
                    trait_id: ti1,
                    name: n1,
                    self_ty: s1,
                },
                TypeData::AssociatedType {
                    trait_id: ti2,
                    name: n2,
                    self_ty: s2,
                },
            ) if ti1 == ti2 && n1 == n2 => {
                let self_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*s1, *s2, self_variance, region_tree, coercion_depth + 1)?;
                if !self.set_binding(a, b) {
                    return Err(TypeError::Mismatch {
                        expected: a,
                        found: b,
                        span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                    });
                }
                Ok(b)
            }

            // ── Under non-Invariant variance, try subtype fallback ──
            _ if variance != Variance::Invariant => {
                let (sub, sup) = match variance {
                    Variance::Covariant => (a, b),
                    Variance::Contravariant => (b, a),
                    _ => unreachable!(),
                };
                if self.subtype(sub, sup) {
                    if !self.set_binding(a, b) {
                        return Err(TypeError::Mismatch {
                            expected: a,
                            found: b,
                            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
                        });
                    }
                    Ok(b)
                } else {
                    Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
                    })
                }
            }

            _ => Err(TypeError::Mismatch {
                expected: b,
                found: a,
                span: (*&self.current_unify_span).unwrap_or(crate::ast::Span::new(FileId::DUMMY, 0, 0)),
            }),
        }
    }
}
