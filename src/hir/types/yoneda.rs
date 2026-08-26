use super::*;

impl<'input> TypeContext<'input> {
    /// Apply Yoneda reduction if this type matches the pattern
    /// `∀X. (A ⇒ X) ⇒ B⟨X⟩`  →  `B[X ↦ A]`
    /// or `∀X. (X ⇒ A) ⇒ B⟨X⟩`  →  `B[X ↦ A]`.
    /// Matches both explicit `Forall` nodes and implicit `Fn`-encoded patterns.
    ///
    /// Uses iteration with convergence detection (max 10 rounds) to handle
    /// chained reductions like Forall(X, Forall(Y, ...)) where reducing the
    /// outer Forall exposes a new reducible inner Forall. This follows the
    /// same convergence principle as Yen's KSP algorithm:
    /// "keep iterating until no more candidates can be generated".
    pub fn try_yoneda_reduce(&mut self, ty: TypeId) -> TypeId {
        // Limit iterations to prevent DoS from maliciously constructed types.
        // In practice, Yoneda/co-Yoneda reduction converges in ≤3 iterations
        // because each pass either eliminates a Forall node or reaches a
        // fixed point.  10 is a generous safety ceiling.
        const MAX_ITERATIONS: usize = 10;
        let mut result = ty;
        for _iteration in 0..MAX_ITERATIONS {
            let before = result;
            result = self.yoneda_reduce_once(result);
            if result == before {
                break; // converged
            }
        }
        result
    }

    /// Single-pass Yoneda reduction (used internally by `try_yoneda_reduce`).
    ///
    /// Matches the ≡_X / ≡_X schemas from Pistone & Tranchini (2022) §2.
    /// These are type-level instances of the Yoneda Lemma and its dual
    /// (Mac Lane §III.2): Nat(D(r, —), K) ≅ K(r), applied to the
    /// representable functor D(X, —) in the category of types.
    ///
    /// **≡_X (Yoneda)** – each branch's *return* is the bound variable X:
    /// ```text
    /// ∀X. ∀Y⃗. ⟨ ∀Z⃗ₖ. ⟨Aⱼₖ⟨X⟩⟩ⱼ ⇒ X ⟩ₖ ⇒ B⟨X⟩
    ///   ≡_X
    /// ∀Y⃗. B⟨X ↦ Σₖ ∃Z⃗ₖ. Πⱼ Aⱼₖ⟨X⟩⟩
    /// ```
    ///
    /// **≡_X (co-Yoneda)** – each branch's *first param* is the bound variable X:
    /// ```text
    /// ∀X. ∀Y⃗. ⟨ ∀Z⃗ₖ. X ⇒ Aⱼ⟨X⟩ ⟩ₖ ⇒ B⟨X⟩
    ///   ≡_X
    /// ∀Y⃗. B⟨X ↦ νX. ∀Z⃗ₖ. Πⱼ Aⱼ⟨X⟩⟩
    /// ```
    ///
    /// Terms: `Σₖ` = sum (multi-branch → Tuple of results), `Πⱼ` = product (Tuple),
    /// `∃Z⃗ₖ` = Exists node when the branch has inner quantifiers.
    /// μX/νX fixpoints are elided when X does not appear in B⟨X⟩ (common case).
    ///
    /// ## Note on partial solving (2026-07)
    ///
    /// We considered extending this function to perform "partial" Yoneda reduction
    /// when the type contains unresolved inference variables (InferVar) or other
    /// non-standard shapes that don't match ≡_X / ≡^X exactly.  The idea would be
    /// to reduce what can be reduced and suspend the rest, resuming once more type
    /// information becomes available (akin to OmniML's suspended match constraints).
    ///
    /// The paper (Pistone & Tranchini 2022) defines ≡_X / ≡^X as deterministic,
    /// all-or-nothing rewrite rules — partial solving would be an extension beyond
    /// what the paper specifies, lacking formal guarantees (soundness, completeness,
    /// termination) for the new "partial" rules.  We chose not to pursue this.
    ///
    /// If κ(A) imprecision from abandoned reductions becomes a practical problem,
    /// consider revisiting with a well-scoped extension (e.g. limited to InferVar
    /// only, or as a separate κ-only pass).
    fn yoneda_reduce_once(&mut self, ty: TypeId) -> TypeId {
        // ── Case A: explicit Forall node ──────────────────────────────
        let ty_data = self.get_arc(ty);
        if let TypeData::Forall {
            param_index,
            param_name: _,
            body,
        } = &*ty_data
        {
            // Strip leading ∀Y⃗ outer quantifiers before the Fn kernel.
            // Paper (Fig.3): ≡_X / ≡^X preserves ∀Y⃗ on both sides.
            //   ∀X. ∀Y⃗. ⟨...⟩ₖ ⇒ B⟨X⟩   ≡_X   ∀Y⃗. B⟨X ↦ ...⟩
            let mut outer_quantifiers: Vec<(usize, Symbol)> = Vec::new();
            let mut inner = *body;
            loop {
                let inner_data = self.get_arc(inner);
                match &*inner_data {
                    TypeData::Forall {
                        param_index: oi,
                        param_name: on,
                        body: ob,
                    } => {
                        outer_quantifiers.push((*oi, *on));
                        inner = *ob;
                    }
                    _ => break,
                }
            }
            let body_data = self.get_arc(inner);
            if let TypeData::Fn { params, ret } = &*body_data {
                let pi = *param_index;
                let ret = *ret;
                let mut branch_replacements: Vec<TypeId> = Vec::new();
                let mut is_coyoneda = false;
                // co-Yoneda (≡_X): no Σₖ — multiple branches combine via product,
                // not coproduct. Each branch's Aⱼ = whole function tail after X.
                // (Pistone & Tranchini 2022 §2, ≡_X formula)
                let mut coyoneda_replacements: Vec<TypeId> = Vec::new();
                // βη normalization: expand Tuple-of-Fn branches into separate branches.
                // (Pistone & Tranchini 2022 §2, βη-isomorphisms: (A→C)×(B→C) ≅ (A+B)→C)
                // A single branch that is a Tuple of Fns is expanded so that each
                // Fn component becomes an independent branch for Yoneda/co-Yoneda matching.
                let mut normalized_params: Vec<TypeId> = Vec::with_capacity(params.len());
                for &b in params.iter() {
                    match self.get(b) {
                        TypeData::Tuple { elems } => {
                            // Each element of the tuple becomes a separate branch.
                            normalized_params.extend(elems.iter().copied());
                        }
                        _ => normalized_params.push(b),
                    }
                }
                for &branch in &normalized_params {
                    // Peel outer Forall layers (∀Z⃗ₖ).
                    let mut inner_quantifiers: Vec<(usize, Symbol)> = Vec::new();
                    let mut inner = branch;
                    loop {
                        let inner_data = self.get_arc(inner);
                        match &*inner_data {
                            TypeData::Forall {
                                param_index: fi,
                                param_name: fn_,
                                body: b,
                            } => {
                                inner_quantifiers.push((*fi, *fn_));
                                inner = *b;
                            }
                            _ => break,
                        }
                    }
                    let inner_data = self.get_arc(inner);
                    if let TypeData::Fn {
                        params: ips,
                        ret: ir,
                    } = &*inner_data
                    {
                        // Check ≡_X (Yoneda): ir = GenericParam(pi)
                        let yoneda_match = match self.get(*ir) {
                            TypeData::GenericParam { index, .. } if *index == pi => true,
                            _ => false,
                        };
                        // Check ≡_X (co-Yoneda): ips[0] = GenericParam(pi)
                        let coyoneda_match = if !ips.is_empty() {
                            match self.get(ips[0]) {
                                TypeData::GenericParam { index, .. } if *index == pi => true,
                                _ => false,
                            }
                        } else {
                            false
                        };
                        // Process Yoneda case
                        if yoneda_match {
                            let product = if ips.len() == 1 {
                                ips[0]
                            } else {
                                self.tuple(ips.clone())
                            };
                            let repl = if inner_quantifiers.is_empty() {
                                product
                            } else {
                                let mut w = product;
                                // Barendregt-inspired renaming: replace each peeled index
                                // with a globally-unique index (via fresh_param_index) that
                                // also cannot collide with pi, preventing false positives
                                // in needs_fix while guaranteeing full capture avoidance.
                                for (eq, en) in &inner_quantifiers {
                                    let mut fresh_idx = self.fresh_param_index();
                                    if fresh_idx == pi {
                                        fresh_idx = self.fresh_param_index();
                                    }
                                    let fresh_gp = self.generic_param(fresh_idx, *en);
                                    w = self.replace_generic(w, *eq, fresh_gp);
                                    w = self.exists(
                                        fresh_idx,
                                        *en,
                                        w,
                                        crate::ast::Expr::Literal(
                                            crate::ast::Literal::Bool(true),
                                            crate::ast::Span::new(0, 0),
                                        ),
                                    );
                                }
                                w
                            };
                            branch_replacements.push(repl);
                        }
                        // Process co-Yoneda case (only if not already handled by Yoneda)
                        if !yoneda_match && coyoneda_match {
                            is_coyoneda = true;
                            // ≡_X: each branch's Aⱼ = the whole function tail after X
                            // (Pistone & Tranchini 2022 §2, ≡_X formula).
                            // Multiple branches combine via product, NOT coproduct.
                            let replacement = if ips.len() <= 1 {
                                *ir
                            } else {
                                self.function(ips[1..].to_vec(), *ir)
                            };
                            let repl = if inner_quantifiers.is_empty() {
                                replacement
                            } else {
                                let mut w = replacement;
                                for (eq, en) in &inner_quantifiers {
                                    let mut fresh_idx = self.fresh_param_index();
                                    if fresh_idx == pi {
                                        fresh_idx = self.fresh_param_index();
                                    }
                                    let fresh_gp = self.generic_param(fresh_idx, *en);
                                    w = self.replace_generic(w, *eq, fresh_gp);
                                    w = self.forall(fresh_idx, *en, w);
                                }
                                w
                            };
                            coyoneda_replacements.push(repl);
                        }
                    }
                }
                // ≡_X and ≡^X are exclusive global patterns (paper §2):
                // ALL branches must match the SAME schema.  Mixed branches
                // (some Yoneda, some co-Yoneda) cannot be reduced.
                if !branch_replacements.is_empty() && !coyoneda_replacements.is_empty() {
                    return ty;
                }
                // EVERY parameter must match the schema — a non-matching
                // parameter (e.g. `Int` in `∀X. (A → X) → Int → X`) aborts
                // the reduction (fail-closed): silently dropping it would
                // equate `A` with `Int → A`, breaking type soundness.
                let total_matches = branch_replacements.len() + coyoneda_replacements.len();
                if total_matches == 0 || total_matches != normalized_params.len() {
                    return ty;
                }

                // ── Variance preconditions (Pistone & Tranchini 2022 §2) ──
                // Yoneda (≡_X) requires B⟨X⟩ to be entirely positive, and each
                // replacement Σₖ ∃Z⃗ₖ. Πⱼ Aⱼₖ⟨X⟩ to be positive for μ legality.
                // co-Yoneda (≡^X) requires B⟨X⟩ to be entirely negative, and each
                // replacement Πⱼ Aⱼ⟨X⟩ to be positive for ν legality.
                if !branch_replacements.is_empty() {
                    // Yoneda case
                    if !self.check_positive_only(pi, ret)
                        || branch_replacements
                            .iter()
                            .any(|&r| !self.check_positive_only(pi, r))
                    {
                        return ty;
                    }
                } else if !coyoneda_replacements.is_empty() {
                    // co-Yoneda case
                    if !self.check_negative_only(pi, ret)
                        || coyoneda_replacements
                            .iter()
                            .any(|&r| !self.check_positive_only(pi, r))
                    {
                        return ty;
                    }
                }

                let sigma = if is_coyoneda {
                    // ≡_X: no Σₖ — multiple branches combine via product (tuple),
                    // not coproduct.  (Pistone & Tranchini 2022 §2, ≡_X formula)
                    if coyoneda_replacements.len() == 1 {
                        coyoneda_replacements[0]
                    } else {
                        self.tuple(coyoneda_replacements.clone())
                    }
                } else {
                    // Σₖ is the categorical coproduct (sum type), NOT a product.
                    // For ∀X.(A₁⇒X)⇒(A₂⇒X)⇒X  →  A₁ + A₂
                    self.coproduct(branch_replacements)
                };
                // Wrap with μX/νX only when the branch product(s) depend on X
                // (Pistone & Tranchini 2022 §2, eq.3 & eq.4):
                //   Yoneda (A⟨X⟩⇒X):    B⟨X⟩ → B⟨X↦μX.A⟨X⟩⟩
                //   co-Yoneda (X⇒A⟨X⟩): B⟨X⟩ → B⟨X↦νX.A⟨X⟩⟩
                // When A⟨X⟩ = Int (no X), no fixpoint needed:
                //   ∀X.(Int⇒X)⇒B⟨X⟩  →  B⟨X↦Int⟩
                let needs_fix = self.type_contains_param(pi, sigma);
                let replacement = if needs_fix {
                    if is_coyoneda {
                        self.alloc(TypeData::Nu {
                            param_index: pi,
                            param_name: "X".into(),
                            body: sigma,
                        })
                    } else {
                        self.alloc(TypeData::Mu {
                            param_index: pi,
                            param_name: "X".into(),
                            body: sigma,
                        })
                    }
                } else {
                    sigma
                };
                let mut result = self.replace_generic(ret, pi, replacement);
                // Re-wrap preserved outer quantifiers ∀Y⃗ (paper Fig.3).
                for (oi, on) in outer_quantifiers.into_iter().rev() {
                    result = self.forall(oi, on, result);
                }
                return result;
            }
            return ty;
        }

        // No explicit Forall → no reduction (Case B was removed, as all legal
        // Yoneda/co-Yoneda patterns are captured by the explicit Forall case).
        ty
    }

    /// Check whether all occurrences of `param` in `ty` appear only in
    /// **positive** (covariant) positions throughout the type tree
    /// (Pistone & Tranchini 2022 §2).
    fn check_positive_only(&self, param: usize, ty: TypeId) -> bool {
        self.check_variance(param, ty, 1)
    }

    /// Check whether all occurrences of `param` in `ty` appear only in
    /// **negative** (contravariant) positions.
    fn check_negative_only(&self, param: usize, ty: TypeId) -> bool {
        self.check_variance(param, ty, -1)
    }
}
