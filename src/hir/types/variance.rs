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
            if let Some(&cached) = self.variance_cache.borrow().get(&cache_key) {
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
            self.variance_cache.borrow_mut().insert(cache_key, result);
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
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                self.type_contains_free_param(param, *ty, scope)
            }
            TypeData::Ptr { size, pointee, .. } => {
                self.type_contains_free_param(param, *pointee, scope)
                    || self.type_contains_free_param(param, *size, scope)
            }
            TypeData::AssociatedType { self_ty, .. } => {
                self.type_contains_free_param(param, *self_ty, scope)
            }
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
