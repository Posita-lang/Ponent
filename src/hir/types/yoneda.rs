//! Yoneda reduction driver (Pistone & Tranchini 2020, §3/§5).
//!
//! `try_yoneda_reduce` peels `∀X. (A ⇒ X) ⇒ B⟨X⟩ → B[X↦A]` (and the (X ⇒ A)
//! dual) using a strictly-decreasing `∀`-count fuel (guaranteed
//! termination, §3 quantifier measure).
//!
//! ## Roadmap
//!
//! **Stage 4 — ε-theory-strength equivalence (the gate).**
//! The canonical-form (βη) equivalence in `canonical.rs`
//! (`neg_beta_eta_equiv`) is a *sound one-sided approximation* of the
//! partial Fig. 4 axiomatization — it is **not** full ε-theory.  Consumers
//! that need ε-theory-strength isomorphisms (the Yoneda schemas
//! ≅_X / ≅_X̄) must gate on the compilation-unit parametricity flag
//! (`mod.rs` `parametricity`, set by a `dyn Trait` reification): when a
//! parametricity-breaking construct is present the ε-strength isomorphisms
//! are refused (fail-closed).  This is the "Stage 4 note" referenced from
//! `canonical.rs` and `mod.rs`.
//!
//! **P3 — performance & safety hardening (implemented).**
//! All items below are implemented; none affect the fail-closed
//! correctness of the tree machinery:
//! - `is_coherent` (2-SAT, `coherence.rs`): `FlatTree` now precomputes a
//!   (label, color) → [node] index (`labeled`) at construction;
//!   `nodes_labeled` is an O(1) lookup — the O(n²) clause pass no longer
//!   rescans the nodes (O(N + n²·avg) total).
//! - `has_cyclic_alternating_path`: the up-move ancestor query (γ ⪯ β)
//!   uses Euler entry/exit times (`tin`/`tout` on `FlatTree`, built in
//!   one iterative pass) — an O(1) subtree-interval test instead of a
//!   per-query parent walk.
//! - `to_neg` / `to_pos` (`canonical.rs`): the FV pass is INCREMENTAL —
//!   `to_neg_fv`/`to_pos_fv` return the subtree's free-variable set along
//!   with the canonical tree in ONE structural pass; the vacuous-binder
//!   test is an O(1) `contains` instead of a per-binder `fvs()` rescan.
//! - The recursive walkers: `build_kappa_graph` (kappa.rs) is fully
//!   iterative (explicit frame stack); `walk_sum`/`walk_prod`
//!   (rewrite.rs) are depth-capped (`WALK_DEPTH_CAP`, fail-closed `None`).
//!   The remaining recursive walkers (`to_neg_fv`/`to_pos_fv`,
//!   `FlatTree::flat`, `PolyTree::subst`, `collect_fvs`) are bounded above
//!   by the parser's `max_recursion_depth` (256): a type too deep to
//!   parse can never reach them from source.
//! - `FUNCTOR_ACTION_FUEL` (`polar_action.rs`): BASE of a size-scaled
//!   budget — `functor_action_fuel` uses `max(BASE, 10  node_count)`;
//!   the binder-shadow rule remains the real termination argument.

use crate::ast::FileId;
use super::*;

impl<'input> TypeContext<'input> {
    /// Apply Yoneda reduction if this type matches the pattern
    ///   `∀X. (A ⇒ X) ⇒ B⟨X⟩`    →    `B[X ↦ A]`
    /// or   `∀X. (X ⇒ A) ⇒ B⟨X⟩`    →    `B[X ↦ A]`  .
    /// Matches both explicit   `Forall`   nodes and implicit   `Fn`  -encoded patterns.
    ///
    /// ## Termination: quantifier-measure fuel (Pistone & Tranchini 2022 §3)
    ///
    /// Each ≡_X / ≡^X reduction eliminates exactly one `∀X`: the retained
    /// `∀Y⃗` binders are re-wrapped 1:1 and the peeled `∀Z⃗ₖ` are re-introduced
    /// as ∃/∀ binders with the same count.  The number of `Forall` nodes in
    /// the tree is therefore a strictly decreasing measure — using it as
    /// fuel gives guaranteed termination and replaces the previous
    /// arbitrary `MAX_ITERATIONS = 10` safety cap.  (If the fuel is ever
    /// exhausted mid-convergence — which the measure argument rules out —
    /// we stop and return the current type: fail-closed.)
    pub fn try_yoneda_reduce(&mut self, ty: TypeId) -> TypeId {
        let mut fuel = self.count_foralls(ty);

        let mut result = ty;
        loop {
            let before = result;
            let head = self.yoneda_reduce_once(result);
            result = self.yoneda_reduce_nested(head);
            if result == before || fuel == 0 {
                break;
            }
            fuel -= 1;
        }

        result
    }

    /// Count the `Forall` nodes in a type tree (following `resolve_binding`).
    /// This is the termination measure for Yoneda reduction: each reduction
    /// eliminates exactly one ∀X (see `try_yoneda_reduce`).
    fn count_foralls(&self, root: TypeId) -> usize {
        let mut count = 0usize;
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(t) = stack.pop() {
            if !visited.insert(t) {
                continue;
            }
            if matches!(self.get(t), TypeData::Forall { .. }) {
                count += 1;
            }
            visit_type_children(t, self, &mut |c| stack.push(c));
        }
        count
    }

    /// Recursively drive Yoneda reduction through the *whole* type tree, not
    /// just the head redex.
    ///
    /// The root attempt (`yoneda_reduce_once`) handles the head redex and the
    /// `∀Y⃗.F`-kernel currying permutation (the paper's `A⇒∀X.B ≡ ∀X.A⇒B`
    /// fixpoint).  This traversal covers the remaining nested positions:
    /// quantifiers inside Tuple / Coproduct elements, Fn parameter and return
    /// positions, and Mu / Nu / Exists bodies.  `Forall` bodies are
    /// deliberately not entered here (binder-capture risk; the head-redex
    /// path already drives reduction through quantified spines), and
    /// ADT / SkolemVar / GenericParam subtrees stay opaque — quantifiers do
    /// not appear under a named type's arguments in HIR.
    ///
    /// Fail-closed discipline (Pistone & Tranchini 2022, "all-or-nothing"):
    /// every position either fully matches a schema and reduces, or is
    /// returned unchanged — partial matches never degrade silently.
    fn yoneda_reduce_nested(&mut self, ty: TypeId) -> TypeId {
        let data = self.get_arc(ty);
        match &*data {
            TypeData::Tuple { elems } => {
                let mut changed = false;
                let new_elems = elems
                    .iter()
                    .map(|&e| {
                        let r = self.try_yoneda_reduce(e);
                        changed |= r != e;
                        r
                    })
                    .collect::<SmallVec<[TypeId; 4]>>();
                if changed { self.tuple(new_elems) } else { ty }
            }
            TypeData::Coproduct { alternatives } => {
                let mut changed = false;
                let new_alts = alternatives
                    .iter()
                    .map(|&a| {
                        let r = self.try_yoneda_reduce(a);
                        changed |= r != a;
                        r
                    })
                    .collect::<SmallVec<[TypeId; 4]>>();
                if changed {
                    self.coproduct(new_alts)
                } else {
                    ty
                }
            }
            TypeData::Fn { params, ret } => {
                let mut changed = false;
                let new_params = params
                    .iter()
                    .map(|&p| {
                        let r = self.try_yoneda_reduce(p);
                        changed |= r != p;
                        r
                    })
                    .collect::<SmallVec<[TypeId; 4]>>();
                let new_ret = self.try_yoneda_reduce(*ret);
                changed |= new_ret != *ret;
                if changed {
                    self.function(new_params, new_ret)
                } else {
                    ty
                }
            }
            TypeData::Mu {
                param_index,
                param_name,
                body,
            } => {
                let new_body = self.try_yoneda_reduce(*body);
                if new_body != *body {
                    self.alloc(TypeData::Mu {
                        param_index: *param_index,
                        param_name: *param_name,
                        body: new_body,
                    })
                } else {
                    ty
                }
            }
            TypeData::Nu {
                param_index,
                param_name,
                body,
            } => {
                let new_body = self.try_yoneda_reduce(*body);
                if new_body != *body {
                    self.alloc(TypeData::Nu {
                        param_index: *param_index,
                        param_name: *param_name,
                        body: new_body,
                    })
                } else {
                    ty
                }
            }
            TypeData::Exists {
                param_index,
                name,
                base,
            } => {
                let new_base = self.try_yoneda_reduce(*base);
                if new_base != *base {
                    self.exists(
                        *param_index,
                        *name,
                        new_base,
                        crate::ast::Expr::Literal(
                            crate::ast::Literal::Bool(true),
                            crate::ast::Span::new(FileId::DUMMY, 0, 0),
                        ),
                    )
                } else {
                    ty
                }
            }
            _ => ty,
        }
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

                                // Build the ∃Z⃗ block in peel order.
                                //
                                // Peeling `∀Z₁. ∀Z₂. ...` gives:
                                //
                                //     [(Z₁), (Z₂)]
                                //
                                // Folding in that order produces:
                                //
                                //     ∃Z₂. ∃Z₁. ...
                                //
                                // This is the nesting pinned by
                                // `test_yoneda_two_inner_quantifiers`.
                                //
                                // Reuse the peeled binder indices unless they would be captured.
                                // Only freshen when the index collides with the outer `X`, a
                                // preserved outer `∀Y`, or an already introduced existential.
                                let mut bound: Vec<usize> = vec![pi];
                                bound.extend(outer_quantifiers.iter().map(|(idx, _)| *idx));

                                let peeled_indices: Vec<usize> =
                                    inner_quantifiers.iter().map(|(idx, _)| *idx).collect();

                                for (eq, en) in inner_quantifiers.iter() {
                                    let mut idx = *eq;

                                    if bound.contains(&idx) {
                                        idx = self.fresh_param_index();

                                        while bound.contains(&idx)
                                            || idx == pi
                                            || peeled_indices.contains(&idx)
                                        {
                                            idx = self.fresh_param_index();
                                        }

                                        let fresh_gp = self.generic_param(idx, *en);
                                        w = self.replace_generic(w, *eq, fresh_gp);
                                    }

                                    w = self.exists(
                                        idx,
                                        *en,
                                        w,
                                        crate::ast::Expr::Literal(
                                            crate::ast::Literal::Bool(true),
                                            crate::ast::Span::new(FileId::DUMMY, 0, 0),
                                        ),
                                    );

                                    bound.push(idx);
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
                                self.function(
                                    ips[1..].iter().copied().collect::<SmallVec<[TypeId; 4]>>(),
                                    *ir,
                                )
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
                    let branch_replacements: SmallVec<[TypeId; 4]> =
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
                    let coyoneda_replacements: SmallVec<[TypeId; 4]> =
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
