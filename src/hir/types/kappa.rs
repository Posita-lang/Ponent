use super::*;

/// The characteristic κ(A) of a type, describing its inhabitant count:
/// - `FiniteExhaustible(usize)` → κ=0: finite inhabitants (e.g. `Bool` has 2)
/// - `InfiniteEnumerable` → κ=1: infinite but enumerable (recursive types with only covariant cycles)
/// - `Undecidable` → κ=∞: cannot decide (cycles through contravariant/invariant positions)
///
/// NB: this is NOT the paper's Def 5.8 κ(A) ∈ {0, 1, ×, ∞} of
/// Pistone & Tranchini (arXiv:1907.03481).  That tree-level verdict —
/// quantifier-elimination convergence on *polymorphic* Λ2 types — lives
/// in `coherence.rs::FlatTree::characteristic` (2-SAT coherence + cyclic
/// alternating paths), and the tree-path reduct in
/// `rewrite.rs::standard_reduce`.  THIS module computes inhabitant
/// *counts* of mostly monomorphic types (ADTs, Int, tuples, arrows) for
/// exhaustiveness checking (`fn_ctxt.rs`); it uses `try_yoneda_reduce`
/// (yoneda.rs, schema-shaped head matching) only as a pre-pass.  The two
/// share a name and the paper's §5 vocabulary but answer different
/// questions on different type domains — do not merge them: the tree path
/// fails closed on non-Λ2 types (Int, ADTs, …), which are exactly this
/// module's bread and butter.
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
    ///
    /// Dual-κ API note: `FlatTree::characteristic` (coherence.rs) computes
    /// the paper's tree-level κ ∈ {0, 1, ∞} from the occurrence tree;
    /// THIS method computes the RESIDENCY count the exhaustiveness checker
    /// consumes (FiniteExhaustible(n)/InfiniteEnumerable/...).  The two are
    /// intentionally different: the checker needs inhabitant counts, not
    /// the paper's rewrite-theory verdict — merging them would force the
    /// exhaustiveness logic through the ε-theoretic lens.
    pub fn characteristic(&mut self, ty: TypeId) -> Characteristic {
        // Resolve bindings first: if ty is an InferVar or GenericParam that has
        // been bound to a concrete type, compute κ on the resolved type instead
        // of the unbound variable.
        let ty = self.resolve_binding_compressed(ty);
        // Check cache first.
        if let Some(&cached) = self.kappa_cache.get(&ty) {
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
        self.kappa_cache.insert(ty, result);
        result
    }

    /// B5a: true if a variant-payload expansion is registered for this
    /// Adt `TypeId` (see `TypeContext::adt_expansions`).
    pub(crate) fn adt_expansion_known(&self, ty: TypeId) -> bool {
        self.adt_expansions.contains_key(&ty)
    }

    /// B5a: the registered variant-payload expansion (per-variant payload
    /// `TypeId`s, nullary variant = unit), if any.
    pub(crate) fn adt_expansion(&self, ty: TypeId) -> Option<&[TypeId]> {
        self.adt_expansions.get(&ty).map(|v| v.as_slice())
    }

    /// B5a: register the variant-payload expansion for an Adt `TypeId`.
    /// Idempotent — the first registration wins (all use sites register
    /// the same payloads for the same instantiation).
    pub(crate) fn register_adt_expansion(&mut self, ty: TypeId, payloads: Vec<TypeId>) {
        self.adt_expansions.entry(ty).or_insert(payloads);
    }

    /// Build the type graph from root, collecting all reachable nodes,
    /// variance edges, and axiom links for bound GenericParam occurrences.
    ///
    /// Iterative (P3d): an explicit frame stack replaces recursion on type
    /// depth, so a pathologically deep type cannot overflow the call stack.
    /// Binder scopes are pushed/popped on the frame's enter/exit, which
    /// preserves exactly the path-dependent shadowing semantics of the
    /// original recursive walk (innermost binder wins).
    fn build_kappa_graph(&self, root: TypeId) -> KappaGraph {
        /// One node on the traversal stack. `slots` holds, per child slot,
        /// the node index of the completed child (`None` while pending).
        /// Child frames are pushed ONE AT A TIME (never all at once): a
        /// frame pops only while its parent is the stack top, so the
        /// parent's `slots` are always the ones receiving the record.
        struct KFrame {
            ty: TypeId,
            idx: usize,
            /// None = enter phase not yet run (children not resolved).
            children: Option<Vec<(TypeId, isize)>>,
            /// Number of binder scopes pushed at enter.
            pushed: usize,
            slots: Vec<Option<usize>>,
            /// Child frames not yet pushed, in slot order (children whose
            /// slot was filled inline — shared subtree / back-edge — never
            /// appear here).
            pending: Vec<(usize, TypeId)>,
            pending_pos: usize,
            /// Slot in the parent frame to record into (`None` for root).
            parent_slot: Option<usize>,
        }

        /// Resolve the outgoing child edges of `ty` (binder pushes included).
        /// Returns (children with variance signs, binders pushed).
        fn resolve_children<'input>(
            ty: TypeId,
            idx: usize,
            ctx: &TypeContext<'input>,
            binder_stack: &mut Vec<(usize, usize)>,
            edges: &mut Vec<(usize, usize, isize)>,
            param_occurrences: &mut HashMap<(usize, usize), Vec<usize>>,
        ) -> (Vec<(TypeId, isize)>, usize) {
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
                    (Vec::new(), 0)
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
                    // Push binder FIRST so GenericParam occurrences in the
                    // body register with the correct binder scope.
                    binder_stack.push((*param_index, idx));
                    (vec![(*body, 1)], 1)
                }
                TypeData::Poly { quantifiers, body } => {
                    // Push all quantifier indices as binders for the body.
                    for &(pi, _) in quantifiers {
                        binder_stack.push((pi, idx));
                    }
                    (vec![(*body, 1)], quantifiers.len())
                }
                TypeData::Exists {
                    param_index,
                    base: body,
                    ..
                } => {
                    // ∃X.A binds X: push it as a binder (like ∀/μ/ν) so that
                    // GenericParam occurrences of X in A register as bound,
                    // not free (B5 counting gap).  Body edge stays covariant.
                    binder_stack.push((*param_index, idx));
                    (vec![(*body, 1)], 1)
                }
                TypeData::Adt {
                    kind: AdtKind::Enum, ..
                } => {
                    // Enum ADT: the generic variance edges (invariant edges
                    // to the nominal args), PLUS the B5a variant-payload
                    // expansion — a self-referential enum produces a
                    // back-edge here, classified by the Phase-2 SCC
                    // classifier (`solve_kappa`) by the cycle's variance.
                    // Unregistered ⇒ no expansion edges (fail-closed:
                    // recursion not detected, the pre-B5a behavior).
                    // Struct ADTs fall to the generic arm: their fields
                    // live in the symbol table (not in the TypeId), so
                    // struct recursion stays an honest boundary.
                    let mut out = ctx
                        .compute_variance_edges(ty)
                        .into_iter()
                        .map(|ve| (ve.target, ve.sign))
                        .collect::<Vec<_>>();
                    if let Some(payloads) = ctx.adt_expansion(ty) {
                        for &p in payloads {
                            out.push((p, 1));
                        }
                    }
                    (out, 0)
                }
                _ => (
                    ctx.compute_variance_edges(ty)
                        .into_iter()
                        .map(|ve| (ve.target, ve.sign))
                        .collect(),
                    0,
                ),
            }
        }

        let mut nodes: Vec<TypeId> = Vec::new();
        let mut edges: Vec<(usize, usize, isize)> = Vec::new();
        let mut node_map: HashMap<TypeId, usize> = HashMap::default();
        // Stack of active binder scopes: (param_index, binder_node_idx)
        let mut binder_stack: Vec<(usize, usize)> = Vec::new();
        // GenericParam occurrences grouped by (param_index, binder_node_idx).
        // Each entry collects all occurrences of a specific variable bound by a specific binder.
        let mut param_occurrences: HashMap<(usize, usize), Vec<usize>> = HashMap::default();

        let mut stack: Vec<KFrame> = Vec::new();
        let root_idx = {
            let idx = nodes.len();
            nodes.push(root);
            node_map.insert(root, idx);
            idx
        };
        stack.push(KFrame {
            ty: root,
            idx: root_idx,
            children: None,
            pushed: 0,
            slots: Vec::new(),
            pending: Vec::new(),
            pending_pos: 0,
            parent_slot: None,
        });

        'traverse: loop {
            if stack.last().unwrap().children.is_none() {
                // Enter phase: resolve this frame's children.  Children
                // already interned (shared subtree / back-edge) complete
                // inline; the rest become `pending` frames, pushed one at
                // a time by the advance branch below.
                let (ty, idx) = {
                    let f = &stack.last().unwrap();
                    (f.ty, f.idx)
                };
                let (children, pushed) = resolve_children(
                    ty,
                    idx,
                    self,
                    &mut binder_stack,
                    &mut edges,
                    &mut param_occurrences,
                );
                let frame = stack.last_mut().unwrap();
                frame.pushed = pushed;
                frame.slots = vec![None; children.len()];
                for (slot, &(c, _)) in children.iter().enumerate() {
                    if let Some(&cidx) = node_map.get(&c) {
                        frame.slots[slot] = Some(cidx);
                    }
                }
                frame.pending = children
                    .iter()
                    .enumerate()
                    .filter(|(slot, _)| frame.slots[*slot].is_none())
                    .map(|(slot, &(c, _))| (slot, c))
                    .collect();
                frame.children = Some(children);
                continue;
            }
            // Advance: the top frame is entered but still has pending
            // children — push exactly ONE child frame (next in slot
            // order).  This is what keeps the parent the stack top the
            // moment its current child pops.
            if stack.last().unwrap().pending_pos < stack.last().unwrap().pending.len() {
                let (slot, c) =
                    stack.last().unwrap().pending[stack.last().unwrap().pending_pos];
                stack.last_mut().unwrap().pending_pos += 1;
                let cidx = {
                    let i = nodes.len();
                    nodes.push(c);
                    node_map.insert(c, i);
                    i
                };
                stack.push(KFrame {
                    ty: c,
                    idx: cidx,
                    children: None,
                    pushed: 0,
                    slots: Vec::new(),
                    pending: Vec::new(),
                    pending_pos: 0,
                    parent_slot: Some(slot),
                });
                continue;
            }
            // Exit phase: the top frame is entered and every child frame
            // above it has already popped and recorded (LIFO discipline),
            // so all slots are filled.
            let (idx, pushed, slots, children, parent_slot) = {
                let f = &stack.last().unwrap();
                debug_assert!(f.slots.iter().all(|s| s.is_some()));
                (
                    f.idx,
                    f.pushed,
                    f.slots.clone(),
                    f.children.clone().unwrap(),
                    f.parent_slot,
                )
            };
            for (slot, &(_, sign)) in children.iter().enumerate() {
                edges.push((idx, slots[slot].unwrap(), sign));
            }
            for _ in 0..pushed {
                binder_stack.pop();
            }
            stack.pop();
            match parent_slot {
                Some(slot) => {
                    stack.last_mut().unwrap().slots[slot] = Some(idx);
                }
                None => break 'traverse,
            }
        }

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
            // (The node's class was recorded at queue time; unwrapping here
            // would be a dead read — the loop only propagates `determined`.)
            let _ = result[determined].is_some();

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

        // Phase 2 (B5b): classify the remaining undetermined nodes per
        // strongly connected component of the undetermined subgraph.
        //
        // Tarjan (iterative) emits SCCs in REVERSE topological order of the
        // condensation — scc_nodes[0] is a SINK (a cycle with no outgoing
        // edge to another undetermined cycle). "Classify successors before
        // their predecessors" therefore means ASCENDING cid order: a
        // singleton that depends on a cycle is visited only after that
        // cycle's class is in `class_map`, so `combine_kappa` sees every
        // child (the B5 fail-closed fallback never fires on a legitimate
        // graph).
        //
        // - A genuine cycle (SCC size > 1, or a self-loop) is classified by
        //   the variance of its internal edges: any non-covariant edge ⇒
        //   Undecidable (a fixpoint through a contravariant/invariant
        //   position cannot be counted); all covariant ⇒
        //   InfiniteEnumerable (Set semantics: the least fixpoint of a
        //   covariant polynomial with a self-reference is infinite — the
        //   conservative reading for the degenerate μX.X case, whose least
        //   fixpoint is empty; we never claim a finite count for a cycle).
        // - A singleton SCC (a node that merely DEPENDS on a cycle) is
        //   classified by the real combination rules (`combine_kappa`) over
        //   its children's classes — exact for every constructor, including
        //   the constant arms (Slice/Ref/Ptr ⇒ InfiniteEnumerable
        //   regardless of the element type).  This replaces the previous
        //   global-flag approximation that classified ALL undetermined
        //   nodes with the worst SCC's verdict.
        use std::collections::HashSet as Set;
        let undetermined_set: Set<usize> = undetermined.iter().copied().collect();
        // Adjacency restricted to the undetermined subgraph.
        let mut adj: Vec<Vec<(usize, isize)>> = vec![Vec::new(); n];
        for &(from, to, sign) in &graph.edges {
            if undetermined_set.contains(&from) && undetermined_set.contains(&to) {
                adj[from].push((to, sign));
            }
        }
        // Iterative Tarjan SCC (same style as `FlatTree::is_coherent`).
        let mut index = vec![usize::MAX; n];
        let mut low = vec![0usize; n];
        let mut comp = vec![usize::MAX; n];
        let mut on_stack = vec![false; n];
        let mut scc_stack: Vec<usize> = Vec::new();
        let mut next_index = 0usize;
        let mut n_comp = 0usize;
        let mut scc_nodes: Vec<Vec<usize>> = Vec::new();
        for root in 0..n {
            if !undetermined_set.contains(&root) || index[root] != usize::MAX {
                continue;
            }
            index[root] = next_index;
            low[root] = next_index;
            next_index += 1;
            scc_stack.push(root);
            on_stack[root] = true;
            let mut frames: Vec<(usize, usize)> = vec![(root, 0)];
            while let Some(&mut (v, ref mut ci)) = frames.last_mut() {
                if *ci < adj[v].len() {
                    let w = adj[v][*ci].0;
                    *ci += 1;
                    if index[w] == usize::MAX {
                        index[w] = next_index;
                        low[w] = next_index;
                        next_index += 1;
                        scc_stack.push(w);
                        on_stack[w] = true;
                        frames.push((w, 0));
                    } else if on_stack[w] {
                        low[v] = low[v].min(index[w]);
                    }
                } else {
                    frames.pop();
                    if let Some(&mut (p, _)) = frames.last_mut() {
                        low[p] = low[p].min(low[v]);
                    }
                    if low[v] == index[v] {
                        let mut scc = Vec::new();
                        loop {
                            let w = scc_stack.pop().unwrap();
                            on_stack[w] = false;
                            comp[w] = n_comp;
                            scc.push(w);
                            if w == v {
                                break;
                            }
                        }
                        scc_nodes.push(scc);
                        n_comp += 1;
                    }
                }
            }
        }
        // Classify successors before their predecessors (ascending id
        // order — see the note above the loop).
        let mut class_map: HashMap<TypeId, Characteristic> = type_kappa.clone();
        for cid in 0..n_comp {
            let scc = &scc_nodes[cid];
            let has_self_loop = scc
                .iter()
                .any(|&v| adj[v].iter().any(|&(to, _)| to == v));
            if scc.len() > 1 || has_self_loop {
                // A genuine cycle: internal variance decides.
                let scc_set: Set<usize> = scc.iter().copied().collect();
                let non_covariant = scc.iter().any(|&v| {
                    adj[v].iter().any(|&(to, sign)| scc_set.contains(&to) && sign != 1)
                });
                let cls = if non_covariant {
                    Characteristic::Undecidable
                } else {
                    Characteristic::InfiniteEnumerable
                };
                for &v in scc {
                    result[v] = Some(cls);
                    class_map.insert(graph.nodes[v], cls);
                }
            } else {
                // Singleton without a self-loop: a node that depends on an
                // already-classified cycle.  Exact combination semantics.
                let v = scc[0];
                let cls = self.combine_kappa(graph.nodes[v], &class_map);
                result[v] = Some(cls);
                class_map.insert(graph.nodes[v], cls);
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
            // Fail-closed (B5): a missing child κ (graph construction missed
            // a dependency edge) means the recursive κ can't be determined —
            // report it as infinite (never claim finite exhaustiveness)
            // instead of panicking and crashing the compiler.
            *map.get(&child).unwrap_or(&Characteristic::InfiniteEnumerable)
        }

        let data = self.get(ty);
        match data {
            TypeData::Tuple { elems } => {
                let mut total = 1usize;
                let mut has_infinite = false;
                let mut has_undecidable = false;

                for &e in elems {
                    match ck(self, e, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => total = total.saturating_mul(n),
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => has_undecidable = true,
                    }
                }

                // 0 absorbs everything: ∅ × A ≅ ∅ for any A (including infinite/undecidable)
                if total == 0 {
                    Characteristic::FiniteExhaustible(0)
                } else if has_undecidable {
                    Characteristic::Undecidable
                } else if has_infinite {
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
            TypeData::Array { elem, size } => {
                // Empty array: exactly one inhabitant, regardless of element type
                if *size == 0 {
                    return Characteristic::FiniteExhaustible(1);
                }

                match ck(self, *elem, kappa_map) {
                    Characteristic::FiniteExhaustible(n) => {
                        Characteristic::FiniteExhaustible(n.saturating_pow(*size as u32))
                    }
                    Characteristic::InfiniteEnumerable => Characteristic::InfiniteEnumerable,
                    Characteristic::Undecidable => Characteristic::Undecidable,
                }
            }
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
                                // B5b fix: an INFINITE domain with codomain
                                // c ≥ 2 gives |B^A| ≥ 2^ℵ₀ — UNCOUNTABLE.
                                // The κ hierarchy has no "uncountable" grade,
                                // so the honest verdict is Undecidable (the
                                // previous InfiniteEnumerable wrongly claimed
                                // countable enumerability of the function
                                // space).
                                Characteristic::Undecidable
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
                        } else if domain_infinite {
                            // B5b fix: |B^A| with A infinite and
                            // |B| ≥ ℵ₀ is uncountable (≥ 2^ℵ₀) —
                            // Undecidable, symmetric with the
                            // FiniteExhaustible(c≥2) arm above.
                            Characteristic::Undecidable
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
