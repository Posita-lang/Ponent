use super::*;

/// The characteristic κ(A) of a type, describing its inhabitant count:
/// - `FiniteExhaustible(usize)` → κ=0: finite inhabitants (e.g. `Bool` has 2)
/// - `InfiniteEnumerable` → κ=1: infinite but enumerable (recursive types with only covariant cycles)
/// - `Undecidable` → κ=∞: cannot decide (cycles through contravariant/invariant positions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Characteristic {
    FiniteExhaustible(usize),
    InfiniteEnumerable,
    Undecidable,
}

/// A variance-annotated type graph used for κ(A) computation.
/// Nodes are TypeIds; edges carry a variance sign (+1 covariant, -1 contravariant, 0 invariant).
#[allow(dead_code)]
struct KappaGraph {
    nodes: Vec<TypeId>,
    /// (from_idx, to_idx, sign)
    edges: Vec<(usize, usize, isize)>,
}

impl<'input> TypeContext<'input> {
    /// Compute the characteristic κ(A) of a type, used for exhaustiveness checking.
    ///
    /// Two-phase algorithm (Pistone & Tranchini 2022 §5):
    /// 1. Yoneda-reduce the type to eliminate quantifiers.
    /// 2. Compute κ on the reduced (monomorphic) type via simple combinatoric rules.
    pub fn characteristic(&mut self, ty: TypeId) -> Characteristic {
        // Resolve bindings first: if ty is an InferVar or GenericParam that has
        // been bound to a concrete type, compute κ on the resolved type instead
        // of the unbound variable.
        let ty = self.resolve_binding(ty);
        // Check cache first.
        if let Some(&cached) = self.kappa_cache.borrow().get(&ty) {
            return cached;
        }
        let reduced = self.try_yoneda_reduce(ty);
        // The κ computation is the FIXED-POINT GRAPH SOLVER (the
        // `rewrite κ(A) as fixed-point graph solver + cache` design):
        // build the type-dependency graph rooted at the reduced type and
        // solve it — recursive (μ/ν) types are solved by the graph's
        // fixpoint iteration (termination is structural), not by a
        // recursive `match` that relies on the Yoneda reduction to
        // terminate.
        let result = self.solve_kappa(&self.build_kappa_graph(reduced));
        // Cache the result.
        self.kappa_cache.borrow_mut().insert(ty, result);
        result
    }

    /// Build the type graph from root, collecting all reachable nodes,
    /// variance edges, and axiom links for bound GenericParam occurrences.
    fn build_kappa_graph(&self, root: TypeId) -> KappaGraph {
        use std::collections::HashSet as Set;

        let mut nodes: Vec<TypeId> = Vec::new();
        let mut edges: Vec<(usize, usize, isize)> = Vec::new();
        let mut node_map: HashMap<TypeId, usize> = HashMap::default();
        let mut visited: Set<TypeId> = Set::default();
        // Stack of active binder scopes: (param_index, binder_node_idx)
        let mut binder_stack: Vec<(usize, usize)> = Vec::new();
        // GenericParam occurrences grouped by (param_index, binder_node_idx).
        // Each entry collects all occurrences of a specific variable bound by a specific binder.
        let mut param_occurrences: HashMap<(usize, usize), Vec<usize>> = HashMap::default();

        // Recursive traversal.
        fn traverse<'input>(
            ty: TypeId,
            ctx: &TypeContext<'input>,
            nodes: &mut Vec<TypeId>,
            edges: &mut Vec<(usize, usize, isize)>,
            node_map: &mut HashMap<TypeId, usize>,
            visited: &mut Set<TypeId>,
            binder_stack: &mut Vec<(usize, usize)>,
            param_occurrences: &mut HashMap<(usize, usize), Vec<usize>>,
        ) -> usize {
            if let Some(&idx) = node_map.get(&ty) {
                return idx;
            }
            let idx = nodes.len();
            nodes.push(ty);
            node_map.insert(ty, idx);
            visited.insert(ty);

            let data = ctx.get(ty);
            match data {
                TypeData::GenericParam { index, .. } => {
                    // Check if this GPIO is bound by an active binder.
                    if let Some(&(pi, binder_idx)) =
                        binder_stack.iter().rev().find(|(p, _)| *p == *index)
                    {
                        param_occurrences
                            .entry((pi, binder_idx))
                            .or_default()
                            .push(idx);
                        // Add a self-loop to mark this GPIO as bound.
                        // This prevents leaf_kappa from resolving it immediately
                        // and ensures the binder's fixed-point cycle is detected.
                        edges.push((idx, idx, 1));
                    }
                }
                TypeData::Forall {
                    param_index, body, ..
                }
                | TypeData::Mu {
                    param_index, body, ..
                }
                | TypeData::Nu {
                    param_index, body, ..
                } => {
                    // Push binder FIRST, then traverse body so GenericParam
                    // occurrences register with the correct binder scope.
                    binder_stack.push((*param_index, idx));
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    // Binder → body (covariant)
                    edges.push((idx, body_idx, 1));
                    binder_stack.pop();
                }
                TypeData::Poly { quantifiers, body } => {
                    // Push all quantifier indices as binders for the body.
                    for &(pi, _) in quantifiers {
                        binder_stack.push((pi, idx));
                    }
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    for _ in quantifiers {
                        binder_stack.pop();
                    }
                    // Poly → body (covariant)
                    edges.push((idx, body_idx, 1));
                }
                TypeData::Exists { base: body, .. } => {
                    // Not introducing a binder for GenericParam — treat body as covariant child.
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    edges.push((idx, body_idx, 1));
                }
                _ => {
                    // Generic case: emit variance edges for all children.
                    let variance_edges = ctx.compute_variance_edges(ty);
                    for ve in &variance_edges {
                        let child_idx = traverse(
                            ve.target,
                            ctx,
                            nodes,
                            edges,
                            node_map,
                            visited,
                            binder_stack,
                            param_occurrences,
                        );
                        edges.push((idx, child_idx, ve.sign));
                    }
                }
            }
            idx
        }

        traverse(
            root,
            self,
            &mut nodes,
            &mut edges,
            &mut node_map,
            &mut visited,
            &mut binder_stack,
            &mut param_occurrences,
        );

        // Build axiom links: for each (variable, binder), connect all GPIO occurrences
        // pairwise as bidirectional covariant edges so they participate in the fixed-point solver.
        for (_key, occurrences) in &param_occurrences {
            for i in 0..occurrences.len() {
                for j in (i + 1)..occurrences.len() {
                    let a = occurrences[i];
                    let b = occurrences[j];
                    edges.push((a, b, 1)); // a → b (covariant)
                    edges.push((b, a, 1)); // b → a (covariant)
                }
            }
        }

        KappaGraph { nodes, edges }
    }

    /// Solve κ for a graph using fixed-point iteration.
    /// Returns the κ of the root node (graph.nodes[0]).
    #[allow(dead_code)]
    fn solve_kappa(&self, graph: &KappaGraph) -> Characteristic {
        let n = graph.nodes.len();
        // result[i] = None (unknown) or Some(κ)
        let mut result: Vec<Option<Characteristic>> = vec![None; n];
        // Maps TypeId → Characteristic for quick child lookup during combine.
        let mut type_kappa: HashMap<TypeId, Characteristic> = HashMap::default();

        let mut out_degree: Vec<usize> = vec![0; n];
        for &(from, _to, _sign) in &graph.edges {
            out_degree[from] += 1;
        }

        // Determine initial κ for base-type leaf nodes (out_degree == 0).
        let mut queue: Vec<usize> = Vec::new();
        for i in 0..n {
            if out_degree[i] == 0 {
                let k = self.leaf_kappa(graph.nodes[i]);
                result[i] = Some(k);
                type_kappa.insert(graph.nodes[i], k);
                queue.push(i);
            }
        }

        // Build reverse adjacency: for each node, which nodes have an edge TO it?
        let mut reverse_edges: Vec<Vec<(usize, isize)>> = vec![Vec::new(); n];
        for &(from, to, sign) in &graph.edges {
            reverse_edges[to].push((from, sign));
        }

        // Track how many outgoing edges are still unresolved for each node.
        let mut unresolved_count: Vec<usize> = out_degree.clone();

        // BFS-based propagation: pop determined nodes and check their predecessors.
        while let Some(determined) = queue.pop() {
            let det_kappa = result[determined].unwrap();

            // Check all predecessors (nodes that depend on `determined`).
            for &(pred, _sign) in &reverse_edges[determined] {
                if result[pred].is_some() {
                    continue;
                }
                unresolved_count[pred] = unresolved_count[pred].saturating_sub(1);
                if unresolved_count[pred] == 0 {
                    let k = self.combine_kappa(graph.nodes[pred], &type_kappa);
                    result[pred] = Some(k);
                    type_kappa.insert(graph.nodes[pred], k);
                    queue.push(pred);
                }
            }
        }

        // After propagation, check for remaining undetermined nodes (cycles).
        let undetermined: Vec<usize> = (0..n).filter(|i| result[*i].is_none()).collect();

        if undetermined.is_empty() {
            return result[0].unwrap();
        }

        // Phase 2: classify remaining cycle(s).  Check edge variance.
        use std::collections::HashSet as Set;
        let undetermined_set: Set<usize> = undetermined.iter().copied().collect();
        let mut has_non_covariant = false;

        for &(from, to, sign) in &graph.edges {
            // Only consider edges where BOTH ends are in the remaining subgraph.
            if undetermined_set.contains(&from) && undetermined_set.contains(&to) && sign != 1 {
                has_non_covariant = true;
                break;
            }
        }

        if has_non_covariant {
            for &i in &undetermined {
                result[i] = Some(Characteristic::Undecidable);
            }
        } else {
            for &i in &undetermined {
                result[i] = Some(Characteristic::InfiniteEnumerable);
            }
        }

        result[0].unwrap()
    }

    /// Return the κ of a leaf type (no outgoing edges).
    #[allow(dead_code)]
    fn leaf_kappa(&self, ty: TypeId) -> Characteristic {
        // Does NOT go through `characteristic_body` — this is the base case.
        let data = self.get(ty);
        match data {
            TypeData::Int { bits, .. } => Characteristic::FiniteExhaustible(
                1usize.checked_shl(*bits as u32).unwrap_or(usize::MAX),
            ),
            TypeData::UInt { bits, .. } => Characteristic::FiniteExhaustible(
                1usize.checked_shl(*bits as u32).unwrap_or(usize::MAX),
            ),
            TypeData::Float { .. } | TypeData::USize => {
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Rational {
                int_bits,
                frac_bits,
                ..
            } => {
                let total_bits = *int_bits as u32 + *frac_bits as u32;
                // Use (usize::BITS - 1) so we can safely represent 1 << total_bits.
                // The previous hard-coded threshold of 16 would misclassify even
                // modest fixed-point types like Rational<8,8> as `usize::MAX`,
                // degrading pattern-match exhaustiveness precision.
                if total_bits >= (usize::BITS - 1) {
                    Characteristic::FiniteExhaustible(usize::MAX)
                } else {
                    Characteristic::FiniteExhaustible(1usize << total_bits)
                }
            }
            TypeData::Bool => Characteristic::FiniteExhaustible(2),
            TypeData::Char => Characteristic::FiniteExhaustible(256),
            TypeData::Byte => Characteristic::FiniteExhaustible(256),
            TypeData::Unit => Characteristic::FiniteExhaustible(1),
            TypeData::Never => Characteristic::FiniteExhaustible(0),
            TypeData::Error => Characteristic::FiniteExhaustible(0),
            TypeData::GenericParam { .. } => {
                // GenericParam with no axiom links → unknown but finite.
                // (If it HAS axiom links, it'll be part of a cycle and get
                //  classified during Phase 2.)
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Adt { .. } => Characteristic::FiniteExhaustible(usize::MAX),
            TypeData::InferVar { .. } => Characteristic::FiniteExhaustible(usize::MAX),
            TypeData::DynTrait { .. } => Characteristic::InfiniteEnumerable,

            // The following types are NOT leaf types in practice because they
            // have outgoing edges.  This arm is a fallback.
            _ => Characteristic::FiniteExhaustible(usize::MAX),
        }
    }

    /// Combine children κ values into a node's κ, given the type constructor.
    /// Called when all of a node's outgoing edges point to determined nodes.
    /// Combine children κ values into a node's κ, given the type constructor.
    /// Called when all of a node's outgoing edges point to determined nodes.
    /// `kappa_map` maps child TypeId → determined Characteristic.
    #[allow(dead_code)]
    fn combine_kappa(
        &self,
        ty: TypeId,
        kappa_map: &HashMap<TypeId, Characteristic>,
    ) -> Characteristic {
        /// Helper: look up a child's κ — must be resolved at this point.
        fn ck<'input>(
            ctx: &TypeContext<'input>,
            child: TypeId,
            map: &HashMap<TypeId, Characteristic>,
        ) -> Characteristic {
            *map.get(&child)
                .expect("child kappa not resolved: graph construction missed a dependency edge")
        }

        let data = self.get(ty);
        match data {
            TypeData::Tuple { elems } => {
                let mut total = 1usize;
                let mut has_infinite = false;
                for &e in elems {
                    match ck(self, e, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => total = total.saturating_mul(n),
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(total)
                }
            }
            TypeData::Adt { args, .. } => {
                let mut has_infinite = false;
                for &a in args {
                    match ck(self, a, kappa_map) {
                        Characteristic::FiniteExhaustible(_) => {}
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(usize::MAX)
                }
            }
            TypeData::Array { elem, size } => match ck(self, *elem, kappa_map) {
                Characteristic::FiniteExhaustible(n) => {
                    Characteristic::FiniteExhaustible(n.saturating_pow(*size as u32))
                }
                Characteristic::InfiniteEnumerable => Characteristic::InfiniteEnumerable,
                Characteristic::Undecidable => Characteristic::Undecidable,
            },
            TypeData::Slice { .. }
            | TypeData::Ref { .. }
            | TypeData::Pointer { .. }
            | TypeData::Ptr { .. } => Characteristic::InfiniteEnumerable,
            TypeData::Fn { params, ret } => {
                let mut domain_product = 1usize;
                let mut domain_infinite = false;
                for &p in params {
                    match ck(self, p, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => {
                            domain_product = domain_product.saturating_mul(n)
                        }
                        Characteristic::InfiniteEnumerable => domain_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                match ck(self, *ret, kappa_map) {
                    Characteristic::Undecidable => Characteristic::Undecidable,
                    Characteristic::FiniteExhaustible(c) => {
                        if domain_product == 0 {
                            Characteristic::FiniteExhaustible(1)
                        } else if domain_infinite {
                            if c == 0 {
                                Characteristic::FiniteExhaustible(0)
                            } else if c == 1 {
                                Characteristic::FiniteExhaustible(1)
                            } else {
                                Characteristic::InfiniteEnumerable
                            }
                        } else {
                            Characteristic::FiniteExhaustible(
                                c.saturating_pow(domain_product as u32),
                            )
                        }
                    }
                    Characteristic::InfiniteEnumerable => {
                        if domain_product == 0 {
                            Characteristic::FiniteExhaustible(1)
                        } else {
                            Characteristic::InfiniteEnumerable
                        }
                    }
                }
            }
            TypeData::Coproduct { alternatives } => {
                let mut total = 0usize;
                let mut has_infinite = false;
                for &a in alternatives {
                    match ck(self, a, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => total = total.saturating_add(n),
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(total)
                }
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. } => ck(self, *body, kappa_map),
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => ck(self, *body, kappa_map),
            TypeData::AssociatedType { self_ty, .. } => ck(self, *self_ty, kappa_map),
            _ => Characteristic::FiniteExhaustible(usize::MAX),
        }
    }
}
