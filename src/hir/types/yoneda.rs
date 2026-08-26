use super::*;

impl<'input> TypeContext<'input> {
    /// Apply Yoneda reduction if this type matches the pattern
    ///  `∀X. (A ⇒ X) ⇒ B⟨X⟩`   →   `B[X ↦ A]`
    /// or  `∀X. (X ⇒ A) ⇒ B⟨X⟩`   →   `B[X ↦ A]` .
    /// Matches both explicit  `Forall`  nodes and implicit  `Fn` -encoded patterns.
    ///
    /// Uses iteration with convergence detection (max 10 rounds) to handle
    /// chained reductions like Forall(X, Forall(Y, ...)) where reducing the
    /// outer Forall exposes a new reducible inner Forall. This follows the
    /// same convergence principle as Yen's KSP algorithm:
    ///  "keep iterating until no more candidates can be generated ".
    pub fn try_yoneda_reduce(&mut self, ty: TypeId) -> TypeId {
        // Limit iterations to prevent DoS from maliciously constructed types.
        // In practice, Yoneda/co-Yoneda reduction converges in ≤3 iterations
        // because each pass either eliminates a Forall node or reaches a
        // fixed point. 10 is a generous safety ceiling.
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
    /// Matches the ≡_X / ≡^X schemas from Pistone & Tranchini (2022) §2.
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
    /// **≡^X (co-Yoneda)** – each branch's *first param* is the bound variable X:
    /// ```text
    /// ∀X. ∀Y⃗. ⟨ ∀Z⃗ₖ. X ⇒ Aⱼ⟨X⟩ ⟩ₖ ⇒ B⟨X⟩
    ///   ≡^X
    /// ∀Y⃗. B⟨X ↦ νX. ∀Z⃗ₖ. Πⱼ Aⱼ⟨X⟩⟩
    /// ```
    ///
    /// Terms: `Σₖ` = coproduct (multi-branch → sum of branch replacements), `Πⱼ` = product (Tuple),
    /// `∃Z⃗ₖ` = Exists node when the branch has inner quantifiers.
    /// μX/νX fixpoints are elided when X does not appear in B⟨X⟩ (common case).
    ///
    /// ## Dual-Track Candidate Strategy
    ///
    /// When a branch is `X ⇒ X`, it simultaneously matches the Yoneda pattern
    /// (`ret == X`) and the co-Yoneda pattern (`ips[0] == X`). A greedy local
    /// choice would commit to Yoneda and potentially fail the global variance
    /// check, missing a valid co-Yoneda reduction (e.g., `∀X.(X⇒X)⇒(X⇒D)`).
    ///
    /// To solve this, we collect candidates for BOTH schemas independently
    /// (Dual-Track), and then adjudicate globally based on the variance of `B⟨X⟩`.
    /// If `X ∉ FV(ret)`, both variance checks vacuously pass, and both paths
    /// yield the exact same result (the replacement is ignored by `replace_generic`),
    /// making the Yoneda-first preference mathematically safe.
    ///
    /// ## Conservatism & Scope
    ///
    /// This implementation is intentionally a **conservative, fail-closed subset**
    /// of the full paper's reduction:
    /// - **Root-only rewriting:** We only attempt reduction at the head redex
    ///   immediately after stripping outer `∀Y⃗`. Nested redexes (e.g. inside Tuples
    ///   or ADT args) are not recursively traversed here. This is deliberate; nested
    ///   normalization is handled by the outer type equivalence/solver loop.
    /// - **Shallow βη pre-normalization:** We only flatten one level of
    ///   Tuple-of-Fn branches. We deliberately skip deep associativity normalization
    ///   and bare `X` branches (which would imply `∀X.X ≡ 1`). This avoids the most
    ///   contentious instances of ε-theory (parametricity) that may not hold in
    ///   standard Set semantics with empty types.
    ///
    /// ## Soundness Invariants
    ///
    /// The correctness of this reduction relies on the following system-level invariants:
    /// 1. `check_variance` correctly handles binder shadowing (flips signs for Fn params,
    ///    preserves for Tuple/Coproduct/Forall/Exists/Mu/Nu bodies).
    /// 2. `replace_generic` is capture-avoiding (respects binder shadowing).
    /// 3. `fresh_param_index` generates globally unique, monotonically increasing indices
    ///    that never collide with existing bound variables in the context.
    /// 4. `TypeId` equality is structural (or interned), ensuring convergence detection
    ///    (`result == before`) works correctly.
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

                // βη normalization: expand Tuple-of-Fn branches into separate branches.
                // (Pistone & Tranchini 2022 §2, βη-isomorphisms: currying (A×B)→C ≡ A→(B→C))
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

                if normalized_params.is_empty() {
                    return ty;
                }

                // ── Dual-Track Candidate Collection ─────────────────────────
                // We collect candidates for BOTH schemas independently.
                let mut yoneda_candidates: Vec<Option<TypeId>> =
                    Vec::with_capacity(normalized_params.len());
                let mut coyoneda_candidates: Vec<Option<TypeId>> =
                    Vec::with_capacity(normalized_params.len());

                for &branch in &normalized_params {
                    // Peel outer Forall layers (∀Z⃗ₖ).
                    let mut inner_quantifiers: Vec<(usize, Symbol)> = Vec::new();
                    let mut inner_branch = branch;
                    loop {
                        let inner_data = self.get_arc(inner_branch);
                        match &*inner_data {
                            TypeData::Forall {
                                param_index: fi,
                                param_name: fn_,
                                body: b,
                            } => {
                                inner_quantifiers.push((*fi, *fn_));
                                inner_branch = *b;
                            }
                            _ => break,
                        }
                    }

                    let mut yoneda_repl = None;
                    let mut coyoneda_repl = None;

                    let inner_data = self.get_arc(inner_branch);
                    if let TypeData::Fn {
                        params: ips,
                        ret: ir,
                    } = &*inner_data
                    {
                        // Check ≡_X (Yoneda): ir = GenericParam(pi)
                        let yoneda_match = matches!(self.get(*ir), TypeData::GenericParam { index, .. } if *index == pi);

                        // Check ≡^X (co-Yoneda): ips[0] = GenericParam(pi)
                        let coyoneda_match = !ips.is_empty()
                            && matches!(self.get(ips[0]), TypeData::GenericParam { index, .. } if *index == pi);

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
                                for (eq, en) in inner_quantifiers.iter().rev() {
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
                            yoneda_repl = Some(repl);
                        }

                        if coyoneda_match {
                            // ≡^X: each branch's Aⱼ = the whole function tail after X
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
                                for (eq, en) in inner_quantifiers.iter().rev() {
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
                            coyoneda_repl = Some(repl);
                        }
                    }

                    yoneda_candidates.push(yoneda_repl);
                    coyoneda_candidates.push(coyoneda_repl);
                }

                let all_yoneda = yoneda_candidates.iter().all(|c| c.is_some());
                let all_coyoneda = coyoneda_candidates.iter().all(|c| c.is_some());

                if !all_yoneda && !all_coyoneda {
                    return ty;
                }

                // ── Global Adjudication ─────────────────────────────────────
                // Try Yoneda first. If it fails variance checks, fall through to co-Yoneda.
                if all_yoneda {
                    let branch_replacements: Vec<TypeId> =
                        yoneda_candidates.iter().map(|c| c.unwrap()).collect();
                    // Yoneda (≡_X) requires B⟨X⟩ to be entirely positive, and each
                    // replacement Σₖ ∃Z⃗ₖ. Πⱼ Aⱼₖ⟨X⟩ to be positive for μ legality.
                    if self.check_positive_only(pi, ret)
                        && branch_replacements
                            .iter()
                            .all(|&r| self.check_positive_only(pi, r))
                    {
                        // Σₖ is the categorical coproduct (sum type), NOT a product.
                        let sigma = self.coproduct(branch_replacements);
                        let needs_fix = self.type_contains_param(pi, sigma);
                        let replacement = if needs_fix {
                            self.alloc(TypeData::Mu {
                                param_index: pi,
                                param_name: "X".into(),
                                body: sigma,
                            })
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
                }

                if all_coyoneda {
                    let coyoneda_replacements: Vec<TypeId> =
                        coyoneda_candidates.iter().map(|c| c.unwrap()).collect();
                    // co-Yoneda (≡^X) requires B⟨X⟩ to be entirely negative, and each
                    // replacement Πⱼ Aⱼ⟨X⟩ to be positive for ν legality.
                    if self.check_negative_only(pi, ret)
                        && coyoneda_replacements
                            .iter()
                            .all(|&r| self.check_positive_only(pi, r))
                    {
                        // ≡^X: no Σₖ — multiple branches combine via product (tuple).
                        let sigma = if coyoneda_replacements.len() == 1 {
                            coyoneda_replacements[0]
                        } else {
                            self.tuple(coyoneda_replacements)
                        };
                        let needs_fix = self.type_contains_param(pi, sigma);
                        let replacement = if needs_fix {
                            self.alloc(TypeData::Nu {
                                param_index: pi,
                                param_name: "X".into(),
                                body: sigma,
                            })
                        } else {
                            sigma
                        };
                        let mut result = self.replace_generic(ret, pi, replacement);
                        for (oi, on) in outer_quantifiers.into_iter().rev() {
                            result = self.forall(oi, on, result);
                        }
                        return result;
                    }
                }

                return ty;
            }
            return ty;
        }
        // No explicit Forall → no reduction.
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
