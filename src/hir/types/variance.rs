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
        self.check_variance_with_sign(param, ty, expected_sign, 1)
    }

    fn check_variance_with_sign(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
    ) -> bool {
        // Resolve bindings first: an unresolved InferVar would be treated as a
        // leaf node with no outgoing variance edges, causing any variance check
        // to silently return `true`.  This would allow InferVars to later be
        // bound to types containing restricted-parameter occurrences, bypassing
        // the variance constraint entirely.
        let ty = self.resolve_binding(ty);
        // Check cache first
        let cache_key = (param, ty, expected_sign, cumulative_sign);
        if let Some(&cached) = self.variance_cache.borrow().get(&cache_key) {
            return cached;
        }
        let result = self.check_variance_uncached(param, ty, expected_sign, cumulative_sign);
        self.variance_cache.borrow_mut().insert(cache_key, result);
        result
    }

    fn check_variance_uncached(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
    ) -> bool {
        // Use pre-computed variance edges instead of pattern-matching TypeData.
        // This is faster because edges are computed once and reused.
        let edges = self.get_variance_edges(ty);
        for edge in &edges {
            if self.type_contains_param(param, edge.target) {
                // Propagate sign: contravariant flips, invariant blocks
                match edge.sign {
                    -1 => {
                        // Contravariant: flip cumulative sign
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            -cumulative_sign,
                        ) {
                            return false;
                        }
                    }
                    0 => {
                        // Invariant: param cannot appear
                        return false;
                    }
                    _ => {
                        // Covariant: keep cumulative sign
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            cumulative_sign,
                        ) {
                            return false;
                        }
                    }
                }
            }
        }
        // No edges → no sub-types (leaf node). Check if THIS node is the param.
        if edges.is_empty()
            && let TypeData::GenericParam { index, .. } = self.get(ty)
            && *index == param
        {
            return cumulative_sign == expected_sign;
        }
        true
    }

    /// Get (or compute) the variance-annotated outgoing edges for a TypeId.
    /// Edges represent "this type → child type with given variance sign".
    fn get_variance_edges(&self, ty: TypeId) -> Vec<VarianceEdge> {
        if let Some(edges) = self.variance_edges.borrow().get(&ty) {
            return edges.clone();
        }
        let edges = self.compute_variance_edges(ty);
        self.variance_edges.borrow_mut().insert(ty, edges.clone());
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
            TypeData::Adt { args, .. } => args
                .iter()
                .map(|&a| VarianceEdge { target: a, sign: 0 }) // invariant — nominal types have invariant params
                .collect(),
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
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                vec![VarianceEdge {
                    target: *ty,
                    sign: 0,
                }]
            }
            TypeData::Ptr { size, pointee, .. } => {
                let mut edges = vec![VarianceEdge {
                    target: *pointee,
                    sign: 0,
                }];
                // size must also be traversed — it may carry GenericParam/SkolemVar
                edges.push(VarianceEdge {
                    target: *size,
                    sign: 0,
                });
                edges
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

    /// Check if a GenericParam with the given index appears anywhere in a type.
    pub fn type_contains_param(&self, param: usize, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::GenericParam { index, .. } => *index == param,
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.type_contains_param(param, p))
                    || self.type_contains_param(param, *ret)
            }
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.type_contains_param(param, a)),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.type_contains_param(param, e)),
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.type_contains_param(param, a)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                self.type_contains_param(param, *elem)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                self.type_contains_param(param, *ty)
            }
            TypeData::Ptr { size, pointee, .. } => {
                self.type_contains_param(param, *pointee) || self.type_contains_param(param, *size)
            }
            TypeData::AssociatedType { self_ty, .. } => self.type_contains_param(param, *self_ty),
            TypeData::Poly { body, .. } => self.type_contains_param(param, *body),
            TypeData::Forall { body, .. } => self.type_contains_param(param, *body),
            TypeData::Exists { base, .. } => self.type_contains_param(param, *base),
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                self.type_contains_param(param, *body)
            }
            _ => false,
        }
    }
}
