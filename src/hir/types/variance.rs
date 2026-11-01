use super::*;

impl<'input> TypeContext<'input> {
    /// Check whether every occurrence of `param` in `ty` is at a
    /// position whose cumulative sign matches `expected_sign`.
    ///
    /// Sign propagation rules:
    ///   - Fn params: contravariant → cumulative sign flips
    ///   - Fn ret: covariant → cumulative sign unchanged
    ///   - Ref/Pointer/Ptr: invariant → param cannot appear inside
    ///   - Tuple/Array/Slice/Struct args/Enum args/Forall body/Exists base:
    ///     covariant → cumulative sign unchanged
    pub(crate) fn check_variance(&self, param: usize, ty: TypeId, expected_sign: isize) -> bool {
        let mut scope = Vec::new();
        self.check_variance_with_sign(param, ty, expected_sign, 1, &mut scope)
    }

    fn check_variance_with_sign(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
        scope: &mut Vec<usize>,
    ) -> bool {
        let ty = self.resolve_binding(ty);
        let scope_was_empty = scope.is_empty();

        // Only when the current scope is empty (i.e., no binder shadowing) do we use the global cache
        if scope_was_empty {
            let cache_key = (param, ty, expected_sign, cumulative_sign);
            if let Some(&cached) = self.interner.variance_cache.read().get(&cache_key) {
                return cached;
            }
        }

        // Handle the binder of the current type (if any), pushing its index onto the stack
        let data = self.get(ty);
        let binder_count = match data {
            TypeData::Forall { param_index, .. }
            | TypeData::Mu { param_index, .. }
            | TypeData::Nu { param_index, .. }
            | TypeData::Exists { param_index, .. } => {
                scope.push(*param_index);
                1
            }
            TypeData::Poly { quantifiers, .. } => {
                for &(pi, _) in quantifiers {
                    scope.push(pi);
                }
                quantifiers.len()
            }
            _ => 0,
        };

        let result = self.check_variance_uncached(param, ty, expected_sign, cumulative_sign, scope);

        // Pop the current binder(s)
        for _ in 0..binder_count {
            scope.pop();
        }

        // If the original scope was empty, store the result in the cache
        if scope_was_empty {
            let cache_key = (param, ty, expected_sign, cumulative_sign);
            self.interner
                .variance_cache
                .write()
                .insert(cache_key, result);
        }

        result
    }

    fn check_variance_uncached(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
        scope: &mut Vec<usize>,
    ) -> bool {
        let edges = self.get_variance_edges(ty);

        // Leaf node: no child edges
        if edges.is_empty() {
            if let TypeData::GenericParam { index, .. } = self.get(ty) {
                // Check if this parameter is shadowed by the current scope
                if scope.iter().rev().any(|&s| s == *index) {
                    return true; // Bound, so it does not affect the check for external free parameters
                }
                return *index == param && cumulative_sign == expected_sign;
            }
            return true; // Other leaf types do not contain parameters
        }

        // Non-leaf node: check edges one by one
        for edge in &edges {
            // Only process this edge if the target type contains the free parameter
            if self.type_contains_free_param(param, edge.target, scope) {
                match edge.sign {
                    -1 => {
                        // Contravariant: flip cumulative_sign
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            -cumulative_sign,
                            scope,
                        ) {
                            return false;
                        }
                    }
                    0 => {
                        // Invariant position: free parameter is not allowed here
                        return false;
                    }
                    _ => {
                        // Covariant: keep cumulative_sign
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            cumulative_sign,
                            scope,
                        ) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Get (or compute) the variance-annotated outgoing edges for a TypeId.
    /// Edges represent "this type → child type with given variance sign".
    fn get_variance_edges(&self, ty: TypeId) -> Vec<VarianceEdge> {
        if let Some(edges) = self.interner.variance_edges.read().get(&ty) {
            return edges.clone();
        }
        let edges = self.compute_variance_edges(ty);
        self.interner
            .variance_edges
            .write()
            .insert(ty, edges.clone());
        edges
    }

    /// Build the outgoing variance edges for a TypeId by inspecting its TypeData.
    pub(crate) fn compute_variance_edges(&self, ty: TypeId) -> Vec<VarianceEdge> {
        match self.get(ty) {
            TypeData::Fn { params, ret } => {
                let mut edges: Vec<VarianceEdge> = params
                    .iter()
                    .map(|&p| VarianceEdge {
                        target: p,
                        sign: -1,
                    })
                    .collect();
                edges.push(VarianceEdge {
                    target: *ret,
                    sign: 1,
                });
                edges
            }
            TypeData::Adt { def_id, args, .. } => {
                // Per-slot ADT variance, derived from the
                // registered AdtDef fields (field→slot convention of
                // `adt_field_is_copy`: GenericParam{i} marks arg slot i),
                // composed with Variance::xform exactly as the
                // unify/subtype consumers define it (Fn params
                // contravariant, *mut/&mut/Ptr-mut invariant, everything
                // else covariant).  A slot's sign is the AGREE over ALL
                // paths from the ADT surface to GenericParam{slot}: all
                // +1 ⇒ +1, all −1 ⇒ −1, anything else ⇒ 0.  A path
                // re-entering a def already on the derivation stack
                // contributes 0 (cycle-conservative): self-referential
                // ADTs (List/Tree) land invariant — the path-aggregation
                // approach loses them,
                // matching the former uniform-invariant status quo with
                // NO regression; the rustc-style fixpoint (the
                // `variances` pass) is the follow-up that would recover
                // them.
                // Non-recursive wrappers — Option/Vec/Result, the
                // dominant parametric shapes — derive exactly.  An
                // UNREGISTERED def also lands all-invariant (fail-closed
                // = the former behavior).  The variance_edges cache
                // inherits the registration-order caveat: register_adt
                // must precede the first edge computation for the type.
                let mut stack: Vec<DefId> = Vec::new();
                let mut memo: HashMap<(DefId, usize), isize> = HashMap::default();
                let mut edges: Vec<VarianceEdge> = Vec::with_capacity(args.len());
                for (j, &a) in args.iter().enumerate() {
                    let sign = self.derive_adt_slot_sign(*def_id, j, &mut stack, &mut memo);
                    edges.push(VarianceEdge { target: a, sign });
                }
                edges
            }
            TypeData::Tuple { elems } => elems
                .iter()
                .map(|&e| VarianceEdge { target: e, sign: 1 })
                .collect(),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                vec![VarianceEdge {
                    target: *elem,
                    sign: 1,
                }]
            }
            TypeData::Ref { ty, .. } => {
                // &T/&mut T: invariant in T (the borrow checker owns
                // the region distinction, variance stays strict).
                vec![VarianceEdge {
                    target: *ty,
                    sign: 0,
                }]
            }
            TypeData::Pointer { ty, mutable, .. } => {
                // §2.4: *const T is covariant in T; *mut T is INVARIANT
                // in T.  Never fall through to a generic covariant walk —
                // that misalignment is the G-class layer mismatch
                // (κ vs polar_action).
                vec![VarianceEdge {
                    target: *ty,
                    sign: if *mutable { 0 } else { 1 },
                }]
            }
            TypeData::Ptr {
                size,
                pointee,
                mutable,
                ..
            } => {
                // §2.4: the size axis (the pointer's own width) is
                // invariant; pointee follows mutability.
                vec![
                    VarianceEdge {
                        target: *pointee,
                        sign: if *mutable { 0 } else { 1 },
                    },
                    // size must also be traversed — it may carry
                    // GenericParam/SkolemVar.
                    VarianceEdge {
                        target: *size,
                        sign: 0,
                    },
                ]
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. }
            | TypeData::Poly { body, .. } => {
                vec![VarianceEdge {
                    target: *body,
                    sign: 1,
                }]
            }
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .map(|&a| VarianceEdge { target: a, sign: 1 })
                .collect(),
            TypeData::AssociatedType { self_ty, .. } => {
                vec![VarianceEdge {
                    target: *self_ty,
                    sign: 1,
                }]
            }
            // Leaves: no edges (GenericParam, primitives, etc.)
            _ => Vec::new(),
        }
    }

    /// The per-slot variance sign of a registered ADT —
    /// agree over ALL field paths from the def's surface to
    /// `GenericParam{slot}` (all +1 ⇒ +1, all −1 ⇒ −1, anything else ⇒
    /// 0).  Unregistered def ⇒ 0 (fail-closed = the former
    /// uniform-invariant behavior).  `stack` holds the defs currently on
    /// the derivation path; `memo` caches (def_id, slot) within one
    /// derive call.
    fn derive_adt_slot_sign(
        &self,
        def_id: DefId,
        slot: usize,
        stack: &mut Vec<DefId>,
        memo: &mut HashMap<(DefId, usize), isize>,
    ) -> isize {
        // Cycle-conservative: a def re-entering its own derivation
        // contributes 0 — self-referential ADTs land invariant (a
        // fixpoint derive would lift this).
        if stack.contains(&def_id) {
            return 0;
        }
        if let Some(&s) = memo.get(&(def_id, slot)) {
            return s;
        }
        let Some(def) = self.interner.adt_defs.read().get(&def_id).cloned() else {
            return 0;
        };
        stack.push(def_id);
        let mut paths: Vec<isize> = Vec::new();
        for &f in &def.fields {
            self.adt_field_paths(f, slot, Variance::Covariant, &mut paths, stack, memo);
        }
        stack.pop();
        let sign = if !paths.is_empty() && paths.iter().all(|&s| s == 1) {
            1
        } else if !paths.is_empty() && paths.iter().all(|&s| s == -1) {
            -1
        } else {
            0
        };
        memo.insert((def_id, slot), sign);
        sign
    }

    /// Collect the composed signs of every path from `ty` down to
    /// `GenericParam{slot}`, threading the position variance with
    /// `Variance::xform` exactly as the unify/subtype consumers define
    /// it.  A path reaching a def already on the derivation stack
    /// contributes 0 and stops (cycle-conservative).
    fn adt_field_paths(
        &self,
        ty: TypeId,
        slot: usize,
        pol: Variance,
        out: &mut Vec<isize>,
        stack: &mut Vec<DefId>,
        memo: &mut HashMap<(DefId, usize), isize>,
    ) {
        let sign_of = |v: Variance| match v {
            Variance::Covariant => 1,
            Variance::Contravariant => -1,
            Variance::Invariant => 0,
        };
        match self.get(ty) {
            TypeData::GenericParam { index, .. } if *index == slot => {
                out.push(sign_of(pol));
            }
            TypeData::GenericParam { .. } => {}
            TypeData::Fn { params, ret } => {
                self.adt_field_paths(*ret, slot, pol.xform(Variance::Covariant), out, stack, memo);
                for &p in params {
                    self.adt_field_paths(
                        p,
                        slot,
                        pol.xform(Variance::Contravariant),
                        out,
                        stack,
                        memo,
                    );
                }
            }
            // §2.4 conventions, same as polar_action/κ: &T/&mut T
            // invariant, *const covariant, *mut invariant, Ptr-mut
            // invariant pointee + invariant size axis.
            TypeData::Ref { ty: t, .. } => {
                self.adt_field_paths(*t, slot, pol.xform(Variance::Invariant), out, stack, memo)
            }
            TypeData::Pointer { ty: t, mutable, .. } => {
                let v = if *mutable {
                    Variance::Invariant
                } else {
                    Variance::Covariant
                };
                self.adt_field_paths(*t, slot, pol.xform(v), out, stack, memo)
            }
            TypeData::Ptr {
                size,
                pointee,
                mutable,
                ..
            } => {
                let v = if *mutable {
                    Variance::Invariant
                } else {
                    Variance::Covariant
                };
                self.adt_field_paths(*pointee, slot, pol.xform(v), out, stack, memo);
                self.adt_field_paths(
                    *size,
                    slot,
                    pol.xform(Variance::Invariant),
                    out,
                    stack,
                    memo,
                );
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => {
                for &e in elems {
                    self.adt_field_paths(e, slot, pol.xform(Variance::Covariant), out, stack, memo);
                }
            }
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => self.adt_field_paths(
                *elem,
                slot,
                pol.xform(Variance::Covariant),
                out,
                stack,
                memo,
            ),
            TypeData::Adt { def_id, args, .. } => {
                if stack.contains(def_id) {
                    // Re-entering a def on the derivation path: the
                    // occurrence variance is unknowable without a
                    // fixpoint derive ⇒ 0.
                    out.push(0);
                    return;
                }
                for (j, &a) in args.iter().enumerate() {
                    let s = self.derive_adt_slot_sign(*def_id, j, stack, memo);
                    let v = match s {
                        1 => Variance::Covariant,
                        -1 => Variance::Contravariant,
                        _ => Variance::Invariant,
                    };
                    self.adt_field_paths(a, slot, pol.xform(v), out, stack, memo);
                }
            }
            // Binder bodies are covariant in their single child; other
            // leaves terminate the path.
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. }
            | TypeData::Poly { body, .. } => self.adt_field_paths(
                *body,
                slot,
                pol.xform(Variance::Covariant),
                out,
                stack,
                memo,
            ),
            _ => {}
        }
    }

    /// Check if a GenericParam with the given index appears freely (not bound)
    /// anywhere in a type, respecting binder scopes.
    pub fn type_contains_free_param(
        &self,
        param: usize,
        ty: TypeId,
        scope: &mut Vec<usize>,
    ) -> bool {
        let ty = self.resolve_binding(ty);
        let data = self.get(ty);

        // Handle the binder of the current type (push onto stack)
        let binder_count = match data {
            TypeData::Forall { param_index, .. }
            | TypeData::Mu { param_index, .. }
            | TypeData::Nu { param_index, .. }
            | TypeData::Exists { param_index, .. } => {
                scope.push(*param_index);
                1
            }
            TypeData::Poly { quantifiers, .. } => {
                for &(pi, _) in quantifiers {
                    scope.push(pi);
                }
                quantifiers.len()
            }
            _ => 0,
        };

        let result = match data {
            TypeData::GenericParam { index, .. } => {
                // Check if shadowed by current scope
                if scope.iter().rev().any(|&s| s == *index) {
                    false
                } else {
                    *index == param
                }
            }
            TypeData::Fn { params, ret } => {
                params
                    .iter()
                    .any(|&p| self.type_contains_free_param(param, p, scope))
                    || self.type_contains_free_param(param, *ret, scope)
            }
            TypeData::Adt { args, .. } => args
                .iter()
                .any(|&a| self.type_contains_free_param(param, a, scope)),
            TypeData::Tuple { elems } => elems
                .iter()
                .any(|&e| self.type_contains_free_param(param, e, scope)),
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.type_contains_free_param(param, a, scope)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                self.type_contains_free_param(param, *elem, scope)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty, .. } => {
                self.type_contains_free_param(param, *ty, scope)
            }
            TypeData::Ptr { size, pointee, .. } => {
                self.type_contains_free_param(param, *pointee, scope)
                    || self.type_contains_free_param(param, *size, scope)
            }
            TypeData::AssociatedType { self_ty, .. } => {
                self.type_contains_free_param(param, *self_ty, scope)
            }
            // A revealed Opaque's hidden type participates in the scan
            // (aligned with `visit_type_children`) — a param living only
            // inside a revealed TAIT hidden type must be seen, otherwise
            // `replace_generic`'s early-return guard lets stale free
            // params survive instantiation.  `hidden: None` is a leaf.
            TypeData::Opaque {
                hidden: Some(h), ..
            } => self.type_contains_free_param(param, *h, scope),
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. }
            | TypeData::Poly { body, .. } => {
                // Note: the current binder is already on the stack, so parameters with the same name in the body will be shadowed
                self.type_contains_free_param(param, *body, scope)
            }
            _ => false, // Other leaf types do not contain parameters
        };

        // Pop the current binder(s)
        for _ in 0..binder_count {
            scope.pop();
        }

        result
    }

    /// Legacy wrapper: checks if `param` appears anywhere in `ty`,
    /// ignoring binder scopes (i.e., treating all occurrences as free).
    /// This matches the original behavior before binder-aware fixes.
    pub fn type_contains_param(&self, param: usize, ty: TypeId) -> bool {
        let mut scope = Vec::new();
        self.type_contains_free_param(param, ty, &mut scope)
    }
}
