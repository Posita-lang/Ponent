use super::*;

impl<'input> TypeContext<'input> {
    pub fn subtype(&mut self, sub: TypeId, sup: TypeId) -> bool {
        if sub == sup {
            return true;
        }

        // Clone Arcs to release the immutable borrow from self.get(), since
        // self.subtype() calls inside match arms require &mut self.
        let sub_data = self.get_arc(sub);
        let sup_data = self.get_arc(sup);

        match (&*sub_data, &*sup_data) {
            (TypeData::Error, _) => true,
            (_, TypeData::Error) => true,
            (TypeData::Never, _) => true,

            // ── Higher-Ranked Types: `∀X.T <: ∀Y.U` ────────────
            (
                TypeData::Forall {
                    param_index: pi1,
                    param_name: _,
                    body: b1,
                },
                TypeData::Forall {
                    param_index: pi2,
                    param_name: _,
                    body: b2,
                },
            ) => {
                if *pi1 == *pi2 {
                    // Same binder index: compare bodies directly.
                    self.subtype(*b1, *b2)
                } else {
                    // α-conversion with capture avoidance: rename BOTH bodies
                    // to a FRESH index that cannot appear free in either body.
                    // Simply renaming pi2 → pi1 would capture any free
                    // GenericParam(pi1) already present in b2.
                    let fresh_idx = self.fresh_param_index();
                    let fresh_name = "α".into();
                    let fresh_gp = self.generic_param(fresh_idx, fresh_name);
                    let b1_renamed = self.replace_generic(*b1, *pi1, fresh_gp);
                    let b2_renamed = self.replace_generic(*b2, *pi2, fresh_gp);
                    self.subtype(b1_renamed, b2_renamed)
                }
            }
            // ∀X.T <: U (U not a Forall): skolemize X in a higher universe so
            // it cannot accidentally unify with free variables in U.
            (
                TypeData::Forall {
                    param_index: pi,
                    param_name: _,
                    body,
                },
                _,
            ) => {
                let (universe, skolem) = self.enter_universe();
                let body_skolemized = self.replace_generic(*body, *pi, skolem);
                let ok = self.subtype(body_skolemized, sup);
                // The skolem must not escape into sup.  The subtype check
                // is currently read-only (no bindings), so escape cannot
                // happen today — this is defense-in-depth for future changes.
                ok && self
                    .check_skolem_escape(sup, universe.saturating_sub(1))
                    .is_none()
            }
            // T <: ∀X.U: peel the right-side binder.
            (_, TypeData::Forall { body, .. }) => self.subtype(sub, *body),

            (TypeData::Unit, TypeData::Unit) => true,
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
                // Aligned with unify_internal_impl's Ref handling:
                // - &mut T <: &T allowed (borrow shortening), invariant inner type
                // - &T <: &mut T NEVER allowed
                // - same mutability → invariant inner type
                //
                // Explicit-lifetime consistency (SYNTAX.md §Explicit
                // Lifetime Parameters — "verified by the borrow checker;
                // mismatches cause compile errors"): two DIFFERENT explicit
                // regions are not in a subtype relation at the pure-type
                // level (the region solver verifies `'a: 'b` outlives at
                // the signature level); an ELIDED side does not constrain.
                //
                // Resolve bindings so that inference variables that have been
                // bound to concrete types are compared by their resolved form.
                let r1 = self.resolve_binding(*t1);
                let r2 = self.resolve_binding(*t2);
                if *m1 != *m2 {
                    // SYNTAX.md §Reference Coercion: by default Posita
                    // does NOT allow `&mut T` to be implicitly coerced to
                    // `&T` — the (gated) coercion path in unification is
                    // the ONLY place that permits it (@auto_ro/@auto_coerce
                    // + CallSite + depth == 0).  `subtype` is a pure
                    // relation and must reject it (an ungated
                    // `&mut T <: &T` here would silently reintroduce the
                    // forbidden coercion from any future call site).
                    return false;
                }
                if let (Some(l1), Some(l2)) = (l1, l2)
                    && l1 != l2
                {
                    // Region SUBTYPE collection (rustc's
                    // `make_subregion(b, a)` — the covariance constraint
                    // `'a: 'b` for `&'a T <: &'b T`): when the checker has
                    // enabled per-signature collection, record the pair
                    // and ACCEPT (the solver decides satisfiability
                    // against the where-constraints); otherwise this is a
                    // pure-relation call (tests, structural checks) and
                    // the strict rejection stands.
                    if self.region_subtype_collect.get() {
                        self.region_subtype_outlives.borrow_mut().push((*l1, *l2));
                        // The pointee must still match (invariant) — the
                        // lifetime covariance is the only relaxation.
                        return r1 == r2;
                    }
                    return false;
                }
                r1 == r2 // same mutability, invariant
            }
            (TypeData::Pointer { ty: t1 }, TypeData::Pointer { ty: t2 }) => {
                // Invariant — exact equality required after resolving bindings.
                self.resolve_binding(*t1) == self.resolve_binding(*t2)
            }
            (
                TypeData::Fn {
                    params: p1,
                    ret: r1,
                },
                TypeData::Fn {
                    params: p2,
                    ret: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return false;
                }
                // Use explicit loop instead of .all() closure to satisfy &mut self
                for (a, b) in p1.iter().zip(p2.iter()) {
                    if !self.subtype(*b, *a) {
                        return false;
                    }
                }
                self.subtype(*r1, *r2)
            }
            (TypeData::Array { elem: e1, size: s1 }, TypeData::Array { elem: e2, size: s2 }) => {
                *s1 == *s2 && self.subtype(*e1, *e2)
            }
            (TypeData::Slice { elem: e1 }, TypeData::Slice { elem: e2 }) => self.subtype(*e1, *e2),
            (TypeData::Tuple { elems: e1 }, TypeData::Tuple { elems: e2 }) => {
                if e1.len() != e2.len() {
                    return false;
                }
                for (a, b) in e1.iter().zip(e2.iter()) {
                    if !self.subtype(*a, *b) {
                        return false;
                    }
                }
                true
            }
            (
                TypeData::Coproduct { alternatives: a1 },
                TypeData::Coproduct { alternatives: a2 },
            ) => {
                if a1.len() != a2.len() {
                    return false;
                }
                for (a, b) in a1.iter().zip(a2.iter()) {
                    if !self.subtype(*a, *b) {
                        return false;
                    }
                }
                true
            }
            (
                TypeData::Int {
                    bits: b1,
                    signed: s1,
                    ..
                },
                TypeData::Int {
                    bits: b2,
                    signed: s2,
                    ..
                },
            ) => *s1 == *s2 && *b1 == *b2,
            (TypeData::Float { bits: b1 }, TypeData::Float { bits: b2 }) => *b1 == *b2,
            (
                TypeData::Rational {
                    int_bits: p1,
                    frac_bits: q1,
                },
                TypeData::Rational {
                    int_bits: p2,
                    frac_bits: q2,
                },
            ) => *p1 == *p2 && *q1 == *q2,
            // Ptr: invariant on both size and pointee — resolve bindings first.
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
                self.resolve_binding(*s1) == self.resolve_binding(*s2)
                    && self.resolve_binding(*p1) == self.resolve_binding(*p2)
            }
            // Poly: α-convert quantifiers → covariant body
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
                let mut b1_renamed = *b1;
                let mut b2_renamed = *b2;
                for ((i1, _), (i2, pn2)) in q1.iter().zip(q2.iter()) {
                    if i1 != i2 {
                        let fresh_idx = self.fresh_param_index();
                        let fresh_gp = self.generic_param(fresh_idx, *pn2);
                        b1_renamed = self.replace_generic(b1_renamed, *i1, fresh_gp);
                        b2_renamed = self.replace_generic(b2_renamed, *i2, fresh_gp);
                    }
                }
                self.subtype(b1_renamed, b2_renamed)
            }
            // SkolemVar: identical skolems are equal; no subtyping otherwise.
            (
                TypeData::SkolemVar {
                    id: id1,
                    universe_num: u1,
                },
                TypeData::SkolemVar {
                    id: id2,
                    universe_num: u2,
                },
            ) => id1 == id2 && u1 == u2,
            _ => false,
        }
    }
}
