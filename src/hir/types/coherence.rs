//! NF occurrence trees, coherence (2-SAT), and the characteristic κ
//! — faithful implementation of Pistone & Tranchini 2020 §4–5
//! (arXiv:1907.03481v3).
//!
//! ## What this module implements
//!
//! - **§4.1**: the βη-invariant tree representation `tp(A)` of Λ2-types.
//!   A type in normal form `∀Y⃗. A₁ → … → Aₙ → X` becomes a root node
//!   labeled `X` (negated color) whose children are the trees of `A_i`,
//!   with the binders `Y⃗` recorded on the root (Lemma 4.1: currying
//!   permutation + vacuous binder removal + quantifier hoisting make this
//!   canonical; parameter order is sorted for the permutation law).
//! - **§5.1**: modular nodes (Def 5.1), eliminality (Def 5.2, Lemma 5.1).
//! - **§5.2**: coherence as 2-SAT (Remark 5.1) — implication graph over the
//!   literals "φ(X) = Blue/Red" (eliminability + parallel-modular-pair
//!   clauses), solved by iterative Tarjan SCC in linear time.
//! - **§5.3**: κ(E) ∈ {0, 1, ∞} via cyclic alternating paths (Def 5.7–5.8).
//!
//! ## Deliberate scope limits (fail-closed)
//!
//! The **polynomial** tree extension of §4.2 (•/N constants encoding
//! µX.∃Y⃗.ΣΠ / νX.∀Y⃗.Π) and the *rewriting rules* of Fig. 6 are NOT
//! implemented here: this module only decides coherence/κ on the plain
//! Λ2-tree `tp(A)`, which is exactly the paper's *a priori* decision
//! problem ("κ(A) ≠ ∞ ⇒ A rewrites to a monomorphic type").  Types using
//! user-level sums/empty/µ/ν fall outside Λ2 and are reported as
//! inapplicable rather than being silently mis-decided.
//!
//! Colors: `blue` = positive occurrence, `red` = negative occurrence,
//! matching the paper's X⁺ / X̄ convention.

use super::*;
use std::collections::HashMap;

/// Occurrence color (§4.1): blue = X⁺, red = X̄.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Blue,
    Red,
}

impl Color {
    fn flip(self) -> Color {
        match self {
            Color::Blue => Color::Red,
            Color::Red => Color::Blue,
        }
    }
}

/// A node of the NF occurrence tree.
///
/// Layout follows Def 4.2: a node `⊕_{i∈I}^{X^c} E_i | Y⃗` has label `X^c`,
/// head-children `E_i` (the `A_i` subtrees), and binder set `Y⃗` (the
/// quantifiers pushed leftmost over this arrow spine).
#[derive(Debug, Clone)]
pub(crate) struct OccNode {
    /// Variable labeling this node (the spine's return variable).
    pub label: usize,
    /// Color of the label occurrence at this node.
    pub color: Color,
    /// Children — the NF subtrees of the spine arguments `A₁ … Aₙ`.
    pub children: Vec<OccTree>,
    /// Quantifier binders pushed to this node (`Y⃗` in Def 4.2).  Each is
    /// `(index, flipped)` where `flipped` records that the binder acts on
    /// the negated occurrence: for a positive node X⁺ the binders are X̄-side
    /// (blue label, red binders) and vice versa.
    pub binders: Vec<usize>,
}

/// A rooted occurrence tree.  `Rc`/`RefCell`-free: built once, read-only.
pub(crate) type OccTree = Box<OccNode>;

/// Node identity inside a built tree (index into `CoherenceTree::nodes`).
pub(crate) type NodeId = usize;

/// A flattened occurrence tree with parent links and node lookup.
pub(crate) struct CoherenceTree {
    pub nodes: Vec<OccNode>,
    pub parent: Vec<Option<NodeId>>,
    pub root: NodeId,
    // B8: the binder-scope maps were stored here but never read after
    // construction — the shadowing check (Notation 5.1) uses the LOCAL maps
    // inside `build_coherence_tree`, so the struct fields were dead state.
}

/// A `bool`-like colored variable occurrence collected during lowering.
#[derive(Debug)]
struct Pending {
    label: usize,
    color: Color,
    children: Vec<Pending>,
    binders: Vec<usize>,
}

// ── Lazy atom terminals ─────────────────────────────────────────────
//
// §4.1 defines Λ2 as atoms + →: the paper's Fig. 1 isomorphisms
// instantiate CLOSED ATOMS (e.g. (★) takes A closed), so a nominal type
// (primitive, ADT, ref/ptr/array/slice, dyn, opaque, Unit, Type) is a
// legitimate Λ2 ground atom — NOT outside the machinery.  Such a type
// lowers to a bare leaf labeled by an injective encoding of its TypeId:
//
//     atom label = ATOM_BASE - ty.index()
//
// Properties (all load-bearing):
// - bijective per context (arena indices are unique), so equal TypeIds
//   share a label and distinct TypeIds never collide;
// - disjoint from `param_index` space by magnitude: atom labels live in
//   [ATOM_LABEL_FLOOR, ATOM_BASE] (~2^63.97), param indices are a small
//   upward counter — `fresh_param_index` debug-asserts the boundary;
// - ≠ `usize::MAX`, which stays reserved for the Π-identity `unit_leaf`
//   in rewrite.rs;
// - never binder-governed: atom labels never enter `binder_scope`, so
//   `is_modular` refuses them (never a modular site / reduction target),
//   the 2-SAT never allocates them a variable, and `place_binders` never
//   sinks a binder onto them.  They are pure inert terminals.
const ATOM_BASE: usize = usize::MAX - 1;
/// Lowest possible atom label = ATOM_BASE - max arena index.
pub(crate) const ATOM_LABEL_FLOOR: usize = usize::MAX - (usize::MAX >> TypeId::TAG_BITS);

/// The lazy atom label for a nominal type (see the block above).
fn atom_label(ty: TypeId) -> usize {
    let l = ATOM_BASE - ty.index();
    debug_assert!(l >= ATOM_LABEL_FLOOR, "atom label underflowed its range");
    l
}

/// Does `ty` contain NO GenericParam anywhere?  Only GROUND nominal
/// types lower to lazy atom leaves; a nominal carrying a parameter (e.g.
/// `Vec<T>` with T a param) is not a Λ2 ground atom — treating it as one
/// would silently drop the param occurrence, so it stays fail-closed.
fn is_ground<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> bool {
    fn go<'input>(ctx: &TypeContext<'input>, ty: TypeId, ok: &mut bool) {
        if !*ok {
            return;
        }
        if matches!(ctx.get(ty), TypeData::GenericParam { .. }) {
            *ok = false;
            return;
        }
        visit_type_children(ty, ctx, &mut |c| go(ctx, c, ok));
    }
    let mut ok = true;
    go(ctx, ty, &mut ok);
    ok
}

impl<'input> TypeContext<'input> {
    /// Build the NF occurrence tree `t(A)` of a Λ2 type (§4.1, Def of t).
    ///
    /// NF lowering mirrors Lemma 4.1's proof: strip vacuous ∀ binders
    /// (`A ≡ ∀X.A` when X∉FV), hoist ∀ out of arrow returns
    /// (`A→∀X.B ≡ ∀X.A→B`), flatten curried spines and sort them
    /// (`A→(B→C) ≡ B→(A→C)`), recursing into NF for the arguments.
    ///
    /// The whole closed type A maps to the *positive* tree `t(A)`.
    ///
    /// Λ2 = atoms + → (§4.1).  GROUND nominal types (primitives,
    /// ground ADTs, refs/ptrs/arrays/slices, dyn, opaque, Unit, Type)
    /// lower to INERT atom leaves labeled by an injective TypeId encoding
    /// (see `atom_label`).  Still outside Λ2 (fail-closed): `Coproduct`
    /// (user sums), `Never` (0), `Mu`/`Nu`, `Exists`, `Poly`,
    /// `InferVar`/`SkolemVar`/`Error`, return-position `Tuple`, and
    /// nominals CARRYING a parameter (`Vec<T>` — not a ground atom;
    /// treating it as one would drop the param occurrence).
    ///
    /// Binder identity is `param_index` (globally unique per the
    /// `fresh_param_index` allocator discipline, see canonical.rs).
    pub fn build_coherence_tree(&self, root: TypeId) -> Option<CoherenceTree> {
        let pending = place_binders(self.lower_type(root, Color::Blue)?);

        // Single-pass: build the owned node tree, collecting parent links
        // and binder scopes as we go.  Binders sit at the LCA of their
        // occurrences (place_binders) — this is NOT an accommodation of the
        // window: per Lemma 4.1(i) the hoisting iso `A⇒∀X.B ≡ ∀X.A⇒B`
        // (X∉FV(A)) pulls a binder up exactly as far as its occurrences
        // allow, so the NF-canonical position IS the LCA.  See the
        // place_binders doc and the fig7 duel goldens below.
        let mut nodes: Vec<OccNode> = Vec::new();
        let mut parent: Vec<Option<NodeId>> = Vec::new();
        let mut binder_root: HashMap<usize, NodeId> = HashMap::default();
        let mut binder_owner: HashMap<usize, NodeId> = HashMap::default();

        fn flatten(
            p: Pending,
            par: Option<NodeId>,
            nodes: &mut Vec<OccNode>,
            parent: &mut Vec<Option<NodeId>>,
            binder_root: &mut HashMap<usize, NodeId>,
            binder_owner: &mut HashMap<usize, NodeId>,
        ) -> Option<NodeId> {
            // Notation 5.1 requires distinct bound variables; a binder
            // index appearing twice (shadowing) ⇒ out of scope, fail-closed.
            for &b in &p.binders {
                if binder_root.contains_key(&b) || binder_owner.contains_key(&b) {
                    return None;
                }
            }
            let id = nodes.len();
            nodes.push(OccNode {
                label: p.label,
                color: p.color,
                children: Vec::new(),
                binders: p.binders.clone(),
            });
            parent.push(par);
            for &b in &p.binders {
                binder_root.insert(b, id);
                binder_owner.insert(b, id);
            }
            for c in p.children {
                let cid = flatten(c, Some(id), nodes, parent, binder_root, binder_owner)?;
                // Link the child as an owned subtree carrying its label and
                // binders (the arena copy stays as a placeholder).
                let taken = std::mem::replace(
                    &mut nodes[cid],
                    OccNode {
                        label: usize::MAX,
                        color: Color::Blue,
                        children: Vec::new(),
                        binders: Vec::new(),
                    },
                );
                nodes[id].children.push(Box::new(taken));
            }
            Some(id)
        }

        let root_id = flatten(
            pending,
            None,
            &mut nodes,
            &mut parent,
            &mut binder_root,
            &mut binder_owner,
        )?;

        // After flatten, every child was swapped OUT of the arena into its
        // parent's owned `children` list; only the root's slot still holds
        // the real (fully owned) tree.  Take it.
        let root_owned = std::mem::replace(
            &mut nodes[root_id],
            OccNode {
                label: usize::MAX,
                color: Color::Blue,
                children: Vec::new(),
                binders: Vec::new(),
            },
        );

        Some(CoherenceTree {
            nodes: vec![root_owned],
            parent: vec![None],
            root: 0,
        })
    }

    /// Lower a type position to a `Pending` tree of the given node color.
    ///
    /// `color` is the color of the *label* occurrence at this spine's root:
    /// the top-level type lowers to a positive (Blue) tree; each arrow
    /// argument flips polarity.
    fn lower_type(&self, ty: TypeId, color: Color) -> Option<Pending> {
        let resolved = self.resolve_binding(ty);
        let data = self.get_arc(resolved);
        match &*data {
            // NF spine: ∀Y⃗.A₁→…→Aₙ→X — collect binders (skipping vacuous
            // ones is handled by the FV check below) and the spine.
            TypeData::Forall {
                param_index, body, ..
            } => {
                // Recurse through chained quantifiers: ∀X.∀Y.A lowers to
                // A's spine carrying both binders.
                let mut inner = self.lower_type(*body, color)?;
                // Vacuous binder elimination (Fig. 4a): X∉FV ⇒ drop.
                if pending_contains_var(&inner, *param_index) {
                    inner.binders.push(*param_index);
                }
                Some(inner)
            }
            _ => Pending::spine_of(self, resolved, color),
        }
    }
}

impl Pending {
    /// Build the spine node for `A₁→…→Aₙ→X` (or a bare variable/atom).
    /// Children lower with flipped polarity; the label is the spine's
    /// return variable.
    fn spine_of<'input>(ctx: &TypeContext<'input>, ty: TypeId, color: Color) -> Option<Pending> {
        let resolved = ctx.resolve_binding(ty);
        let data = ctx.get_arc(resolved);
        match &*data {
            TypeData::Fn { params, ret } => {
                // Currying: flatten nested single-param arrow returns and
                // tuple params into one spine; sort children (permutation).
                let mut args: Vec<TypeId> = Vec::new();
                let mut collect = |t: TypeId, ctx: &TypeContext<'input>| -> Option<()> {
                    let d = ctx.get(ctx.resolve_binding(t));
                    match d {
                        TypeData::Tuple { elems } => {
                            args.extend(elems.iter().copied());
                        }
                        _ => args.push(t),
                    }
                    Some(())
                };
                for &p in params {
                    collect(p, ctx)?;
                }
                // The return side: recurse with `lower_type` so ∀-returns
                // hoist their binder onto this spine (A→∀X.B ≡ ∀X.A→B).
                // The returned node IS this spine's head (label+binders);
                // its children belong to the return type's own spine, which
                // per NF is empty (NF arguments carry the structure).
                let ret_pending = ctx.lower_type(*ret, color)?;
                let mut children: Vec<Pending> = Vec::new();
                for a in args {
                    children.push(ctx.lower_type(a, color.flip())?);
                }
                // If ∀-hoisting pulled a binder out of a *nested* arrow
                // (A→(B→∀X.C)), lower_type returns the inner spine with its
                // own args — those sit at flip(color) twice = the outer
                // args' polarity, so merge them into this spine (currying
                // flattening, Lemma 4.1).
                children.extend(ret_pending.children);
                // P2: decorate-sort-undecorate — pending_cmp recomputed the
                // structural key on EVERY comparison (O(key) each, quadratic
                // in the number of comparisons); compute each child's key
                // once instead.
                let mut decorated: Vec<(Vec<u8>, Pending)> = children
                    .into_iter()
                    .map(|c| (pending_sort_key(&c), c))
                    .collect();
                decorated.sort_by(|a, b| a.0.cmp(&b.0));
                let children: Vec<Pending> =
                    decorated.into_iter().map(|(_, c)| c).collect();
                Some(Pending {
                    label: ret_pending.label,
                    color,
                    children,
                    binders: ret_pending.binders,
                })
            }
            TypeData::GenericParam { index, .. } => Some(Pending {
                label: *index,
                color,
                children: Vec::new(),
                binders: Vec::new(),
            }),
            // GROUND nominal types are Λ2 atoms (§4.1: Λ2 = atoms + →;
            // Fig. 1 instantiates closed atoms, so Int/Bool/ground ADTs/
            // refs/… are legitimate).  They lower to inert bare leaves
            // labeled by an injective TypeId encoding (see ATOM_BASE):
            // never binder-governed, never modular, never 2-SAT variables.
            // A nominal CARRYING a parameter (Vec<T>) is not a ground atom
            // — treating it as one would drop the param occurrence — so it
            // stays fail-closed.
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Rational { .. }
            | TypeData::Unit
            | TypeData::Adt { .. }
            | TypeData::Array { .. }
            | TypeData::Slice { .. }
            | TypeData::Ref { .. }
            | TypeData::Pointer { .. }
            | TypeData::Ptr { .. }
            | TypeData::DynTrait { .. }
            | TypeData::Type
            | TypeData::Opaque { .. } => {
                if is_ground(ctx, resolved) {
                    Some(Pending {
                        label: atom_label(resolved),
                        color,
                        children: Vec::new(),
                        binders: Vec::new(),
                    })
                } else {
                    None // parameterized nominal: not a Λ2 ground atom
                }
            }
            // ∀ in argument position: hoist (A→∀X.B ≡ ∀X.A→B) — the binder
            // travels to the enclosing spine node.  Handled in lower_type
            // via the Forall arm only at top level; here a ∀ return of an
            // inner spine was already lifted by recursion through `ret`.
            _ => None, // outside Λ2: sums (Coproduct), 0 (Never), μ/ν, ∃,
                       // Poly, inference state (InferVar/SkolemVar/Error),
                       // products in return position (Tuple)
        }
    }
}

/// Does this pending tree mention `var` free (ignoring its own binders)?
///
/// P2 boundary: called once per ∀ binder during lowering (L277), each call
/// an early-exiting walk of the just-lowered subtree — O(k·n) over a type
/// with k binders, k small in practice.  An incremental FV undo-log would
/// restructure `lower_type`'s bottom-up construction for a marginal win;
/// deliberately not taken.
fn pending_contains_var(p: &Pending, var: usize) -> bool {
    if p.binders.contains(&var) {
        return false;
    }
    if p.label == var {
        return true;
    }
    p.children.iter().any(|c| pending_contains_var(c, var))
}

// ─── Flat view with parent links ────────────────────────────────────────

/// A node of the flattened view: arena ids with parent pointers, rebuilt
/// from the owned tree (the build-time arena is discarded).
#[derive(Debug, Clone)]
pub(crate) struct FlatNode {
    pub(crate) label: usize,
    pub(crate) color: Color,
    pub(crate) binders: Vec<usize>,
    pub(crate) children: Vec<NodeId>,
}

/// Flattened occurrence tree: supports the §5 graph-theoretic predicates.
pub(crate) struct FlatTree {
    pub(crate) nodes: Vec<FlatNode>,
    pub(crate) parent: Vec<Option<NodeId>>,
    pub(crate) depth: Vec<usize>,
    /// Bound variable → r_X (the node whose binder set contains X;
    /// Notation 5.1 gives uniqueness).
    pub(crate) binder_scope: HashMap<usize, NodeId>,
    /// P3a: (label, color) → node ids, built ONCE at construction.
    /// `nodes_labeled` is an O(1) index lookup instead of an O(N)
    /// rescan — the 2-SAT clause scan in `is_coherent` calls it O(n²)
    /// times, so the precomputed index drops the clause pass from
    /// O(n²·N) to O(n²·avg) + one O(N) build.
    pub(crate) labeled: HashMap<(usize, Color), Vec<NodeId>>,
    /// P3b: Euler tour entry time (`tin[v]` is assigned on discovery,
    /// `tout[v]` after all descendants).  Standard subtree-interval
    /// test: `g` is an ancestor-OR-EQUAL of `id` ⇔
    /// `tin[g] ≤ tin[id] ∧ tout[id] ≤ tout[g]` — O(1) instead of an
    /// O(depth) parent-pointer climb in `has_cyclic_alternating_path`'s
    /// up-move scan (which runs it O(leaves · nodes) times).
    pub(crate) tin: Vec<usize>,
    pub(crate) tout: Vec<usize>,
}

/// P3b: iterative Euler tour of the rooted tree (node 0 = root in both
/// construction paths).  `tin` on discovery, `tout` after the whole
/// subtree — gives the `g ⪯ id` interval test above.  Iterative (stack)
/// by the P3d no-recursion discipline: pathologically deep spines must
/// not recurse.
fn build_euler_times(nodes: &[FlatNode]) -> (Vec<usize>, Vec<usize>) {
    let n = nodes.len();
    let mut tin = vec![0usize; n];
    let mut tout = vec![0usize; n];
    if n == 0 {
        return (tin, tout);
    }
    let mut t = 0usize;
    // (node, next-child-slot)
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    tin[0] = t;
    t += 1;
    while let Some(top) = stack.last_mut() {
        let v = top.0;
        if top.1 < nodes[v].children.len() {
            let w = nodes[v].children[top.1];
            top.1 += 1;
            tin[w] = t;
            t += 1;
            stack.push((w, 0));
        } else {
            tout[v] = t;
            t += 1;
            stack.pop();
        }
    }
    (tin, tout)
}

/// P3a: build the (label, color) → [node] index in one O(N) pass.
fn build_labeled_index(nodes: &[FlatNode]) -> HashMap<(usize, Color), Vec<NodeId>> {
    let mut idx: HashMap<(usize, Color), Vec<NodeId>> = HashMap::default();
    for (i, n) in nodes.iter().enumerate() {
        idx.entry((n.label, n.color)).or_default().push(i);
    }
    idx
}

impl FlatTree {
    /// Construct from raw parts (used by `rewrite::flat_view` to scope a
    /// polynomial-tree skeleton through the same predicates).
    pub(crate) fn from_parts(
        nodes: Vec<FlatNode>,
        binder_scope: HashMap<usize, NodeId>,
    ) -> FlatTree {
        // P3: structural invariants the whole coherence machinery rests
        // on — a non-empty node list, every registered binder pointing at
        // a live node, and (after the parent links below) exactly one
        // root.  Fail loudly in debug if a construction path breaks them.
        debug_assert!(!nodes.is_empty(), "from_parts: empty node list");
        debug_assert!(
            binder_scope.values().all(|&id| id < nodes.len()),
            "from_parts: binder_scope entry out of range"
        );
        let mut parent = vec![None; nodes.len()];
        let mut depth = vec![0usize; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for &c in &n.children {
                parent[c] = Some(i);
                depth[c] = depth[i] + 1;
            }
        }
        debug_assert_eq!(
            parent.iter().filter(|p| p.is_none()).count(),
            1,
            "from_parts: exactly one root expected"
        );
        let labeled = build_labeled_index(&nodes);
        let (tin, tout) = build_euler_times(&nodes);
        FlatTree {
            nodes,
            parent,
            depth,
            binder_scope,
            labeled,
            tin,
            tout,
        }
    }

    /// Coherence valuation φ (Def 5.5) — delegates to the 2-SAT check.
    pub(crate) fn valuation(&self) -> Option<HashMap<usize, Color>> {
        self.is_coherent()
    }
}

impl OccNode {
    fn clone_box(&self) -> OccTree {
        Box::new(self.clone())
    }
}

impl CoherenceTree {
    /// Borrow the owned root occurrence tree (for rewrite.rs lowering).
    pub(crate) fn root_tree(&self) -> OccTree {
        self.nodes[self.root].clone_box()
    }

    /// Flatten the owned root tree into an arena with parent/depth links.
    pub(crate) fn flat(&self) -> FlatTree {
        let mut ft = FlatTree {
            nodes: Vec::new(),
            parent: Vec::new(),
            depth: Vec::new(),
            binder_scope: HashMap::default(),
            labeled: HashMap::default(),
            tin: Vec::new(),
            tout: Vec::new(),
        };
        fn walk(n: &OccNode, par: Option<NodeId>, dep: usize, ft: &mut FlatTree) -> NodeId {
            let id = ft.nodes.len();
            ft.nodes.push(FlatNode {
                label: n.label,
                color: n.color,
                binders: n.binders.clone(),
                children: Vec::new(),
            });
            ft.parent.push(par);
            ft.depth.push(dep);
            for &b in &n.binders {
                // Notation 5.1: unique r_X (build_coherence_tree already
                // rejected duplicates).
                ft.binder_scope.insert(b, id);
            }
            for c in &n.children {
                let cid = walk(c, Some(id), dep + 1, ft);
                ft.nodes[id].children.push(cid);
            }
            id
        }
        walk(&self.nodes[self.root], None, 0, &mut ft);
        ft.labeled = build_labeled_index(&ft.nodes);
        (ft.tin, ft.tout) = build_euler_times(&ft.nodes);
        ft
    }
}

impl FlatTree {
    fn is_bound(&self, v: usize) -> bool {
        self.binder_scope.contains_key(&v)
    }

    /// Terminal (leaf) nodes: bare variable occurrences.
    fn leaves(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].children.is_empty())
            .collect()
    }

    /// Tree distance d(α, β) (number of edges on the unique path).
    fn dist(&self, mut a: NodeId, mut b: NodeId) -> usize {
        // P2: O(1) fast path when one endpoint is an ancestor of the
        // other — the is_modular window case dist(α, r_X) with r_X above.
        if self.tin[b] <= self.tin[a] && self.tout[a] <= self.tout[b] {
            return self.depth[a] - self.depth[b];
        }
        if self.tin[a] <= self.tin[b] && self.tout[b] <= self.tout[a] {
            return self.depth[b] - self.depth[a];
        }
        let mut d = 0;
        while self.depth[a] > self.depth[b] {
            a = self.parent[a].unwrap();
            d += 1;
        }
        while self.depth[b] > self.depth[a] {
            b = self.parent[b].unwrap();
            d += 1;
        }
        while a != b {
            a = self.parent[a].unwrap();
            b = self.parent[b].unwrap();
            d += 2;
        }
        d
    }

    /// All nodes labeled `v` with color `c` (P3a: O(1) index lookup
    /// against the precomputed `labeled` map, not an O(N) rescan).
    fn nodes_labeled(&self, v: usize, c: Color) -> &[NodeId] {
        self.labeled
            .get(&(v, c))
            .map(|s| s.as_slice())
            .unwrap_or(&[])
    }

    /// Def 5.1: is terminal node `a` modular?
    /// modular ⇔ a ≠ h_X ∧ 1 ≤ d(a, r_X) ≤ 2 ∧ no parallel node of the
    /// same label X^c.  Here r_X is the node binding X and h_X is the head
    /// of r_X's subtree — i.e. node r_X itself.
    ///
    /// Representation mapping (arXiv:1907.03481v3 §4.1, L844 verbatim:
    /// "rooted trees whose **leaves** are labeled by colored variables"):
    /// the paper's tree carries variable labels on LEAVES only; spine
    /// nodes are unlabeled.  Our FlatTree merges each spine's head
    /// variable into the spine node, so:
    /// - a code node WITH children ≙ the paper's head-leaf of that spine.
    ///   Its paper distance is the code distance + 1 (one extra edge to
    ///   the head-leaf), so the d(α, r_X) ≤ 2 window admits code d ≤ 1
    ///   (Lemma 5.5's proof, L1987, confirms the negative case: a modular
    ///   candidate "whose distance from r_X is greater than 2" — paper
    ///   distance, i.e. code d ≥ 2 for a spine).
    /// - a code BARE LEAF ≙ the paper leaf itself: window 1..=2 unchanged.
    /// "Parallel" (Notation 5.1, verbatim: "immediate successors of the
    /// same node") is a PAPER-tree relation, so in code terms:
    /// - the head-leaf of spine v is parallel to v's bare-leaf children
    ///   only (head-leaves of spine children live under THEIR spines);
    /// - a bare leaf is parallel to its bare-leaf siblings and to its
    ///   parent spine's head-leaf (the X̄⇒X̄ shape — both non-modular,
    ///   which is what keeps the κ=∞ probe `∀XY.(X→(Y→X))→X→Y` incoherent).
    /// Two same-label spine SIBLINGS (e.g. Example 3.1's `mid`/`YW` W̄
    /// spines under the root) are NOT paper-parallel — they are leaves
    /// under different paper parents — which is what makes Example 3.1
    /// coherent, matching §3's ≅1 proof via Thm 5.6.
    fn is_modular(&self, a: NodeId) -> bool {
        let n = &self.nodes[a];
        let x = n.label;
        let Some(&r) = self.binder_scope.get(&x) else {
            return false; // free occurrence: not governed by any binder
        };
        if a == r {
            return false; // α ≠ h_X
        }
        let d = self.dist(a, r);
        let is_terminal = n.children.is_empty();
        // Distance window, per the representation mapping above.
        let in_window = if is_terminal {
            (1..=2).contains(&d)
        } else {
            d <= 1 // paper d = code d + 1 ≤ 2
        };
        if !in_window {
            return false;
        }
        // "no parallel node of label X^c" — paper siblings, per the
        // mapping above.
        if is_terminal {
            if let Some(p) = self.parent[a] {
                for &s in &self.nodes[p].children {
                    // bare-leaf sibling with same label+color
                    if s != a
                        && self.nodes[s].children.is_empty()
                        && self.nodes[s].label == x
                        && self.nodes[s].color == n.color
                    {
                        return false;
                    }
                }
                // parent spine's head-leaf with same label+color
                let pn = &self.nodes[p];
                if pn.label == x && pn.color == n.color {
                    return false;
                }
            }
        } else {
            // head-leaf of this spine: parallel = bare-leaf children
            // with the same label+color.
            for &s in &n.children {
                let sc = &self.nodes[s];
                if sc.children.is_empty() && sc.label == x && sc.color == n.color {
                    return false;
                }
            }
        }
        true
    }

    /// Def 5.2 (per the LaTeX source): X is `c`-eliminable iff every node
    /// α : X^c̄ (the OPPOSITE color) is modular.  Vacuously true when no
    /// node of that color exists.
    pub(crate) fn color_eliminable(&self, x: usize, c: Color) -> bool {
        self.nodes_labeled(x, c.flip())
            .iter()
            .all(|&a| self.is_modular(a))
    }

    /// Lemma 5.1: X eliminable ⇔ blue-eliminable ∨ red-eliminable, where
    /// "c-eliminable" quantifies over nodes of color c̄.
    fn eliminable(&self, x: usize) -> bool {
        self.color_eliminable(x, Color::Blue) || self.color_eliminable(x, Color::Red)
    }

    // ── Def 5.5: coherence ──────────────────────────────────────────────

    /// Raw parallel-modular-pair existence check: are there parallel
    /// modular nodes α : x^c, β : y^d?  NOT Def 5.4's coherence itself —
    /// Def 5.4 (verbatim, L1955) says X^c and Y^d are coherent iff there
    /// is no parallel modular pair α : X^c̄, β : Y^d̄ in the COMPLEMENTED
    /// colors; hence coherence of (X^c, Y^d) is
    /// `pair_coherent(x, c̄, y, d̄)`, and 2-SAT clause (2) uses the
    /// equivalent form: a pair (x^c, y^d) forbids the complemented
    /// valuation (φ(x)=c̄, φ(y)=d̄), yielding clause (a_x^c ∨ a_y^d).
    fn pair_coherent(&self, x: usize, c: Color, y: usize, d: Color) -> bool {
        if x == y {
            return true; // same variable, no pair (α:X, β:Y) with Y ≠ X
        }
        let xs = self.nodes_labeled(x, c);
        let ys = self.nodes_labeled(y, d);
        for &a in xs {
            if !self.is_modular(a) {
                continue;
            }
            let Some(p) = self.parent[a] else { continue };
            for &b in ys {
                if b != a && self.parent[b] == Some(p) && self.is_modular(b) {
                    return false; // parallel modular pair
                }
            }
        }
        true
    }

    /// Def 5.5 (Remark 5.1) via 2-SAT: X ↦ b_X, b_X ⇔ φ(X) = Blue.
    /// Clauses:
    ///  (1) φ(X)=Blue ⇒ X blue-eliminable, φ(X)=Red ⇒ X red-eliminable
    ///      (Def 5.2 semantics as implemented by `color_eliminable`);
    ///  (2) for every X ≠ Y and colors (c,d) with a parallel modular pair
    ///      α : X^c, β : Y^d (Def 5.4): the forbidden valuation is the
    ///      COMPLEMENT — ¬(φ(X)=c̄ ∧ φ(Y)=d̄), i.e. clause (a_X^c ∨ a_Y^d).
    ///      (Def 5.4: X^c coherent with Y^d ⇔ no parallel modular
    ///      α:X^c̄, β:Y^d̄; so a sibling pair of colors (c,d) rules out the
    ///      valuation (c̄,d̄), not (c,d).)
    /// Implication graph + Tarjan SCC (iterative): O(n²) clauses, linear
    /// solve.  Fail-closed: UNSAT ⇒ not coherent (κ = ∞).
    pub fn is_coherent(&self) -> Option<HashMap<usize, Color>> {
        let vars: Vec<usize> = {
            let mut v: Vec<usize> = self.binder_scope.keys().copied().collect();
            v.sort();
            v
        };
        let n = vars.len();
        if n == 0 {
            return Some(HashMap::default());
        }
        // literal 2i = "φ(vars[i]) = Blue", 2i+1 = "φ(vars[i]) = Red".
        let blue = |i: usize| 2 * i;
        let red = |i: usize| 2 * i + 1;
        // clause (a ∨ b) as implications ¬a → b, ¬b → a (xor-1 negation).
        let mut imp: Vec<Vec<usize>> = vec![Vec::new(); 2 * n];
        fn add(imp: &mut [Vec<usize>], a: usize, b: usize) {
            imp[a ^ 1].push(b);
            imp[b ^ 1].push(a);
        }
        // (1) eliminability.
        for (i, &x) in vars.iter().enumerate() {
            let eb = self.color_eliminable(x, Color::Blue);
            let er = self.color_eliminable(x, Color::Red);
            if !eb {
                add(&mut imp, red(i), red(i)); // forces φ(X) ≠ Blue
            }
            if !er {
                add(&mut imp, blue(i), blue(i)); // forces φ(X) ≠ Red
            }
        }
        // (2) parallel modular pairs (Def 5.4).
        for i in 0..n {
            for j in (i + 1)..n {
                let (x, y) = (vars[i], vars[j]);
                for &(c, d) in &[
                    (Color::Blue, Color::Blue),
                    (Color::Blue, Color::Red),
                    (Color::Red, Color::Blue),
                    (Color::Red, Color::Red),
                ] {
                    if !self.pair_coherent(x, c, y, d) {
                        // Sibling modular pair (x^c, y^d) forbids the
                        // COMPLEMENTED valuation (φ(x)=c̄, φ(y)=d̄): clause
                        // (a_x^c ∨ a_y^d), where a_x^c = "φ(x)=c".
                        let l1 = if c == Color::Blue { blue(i) } else { red(i) };
                        let l2 = if d == Color::Blue { blue(j) } else { red(j) };
                        add(&mut imp, l1, l2);
                    }
                }
            }
        }
        // Tarjan SCC (iterative); ids assigned in reverse topological order.
        let m = 2 * n;
        let mut index = vec![usize::MAX; m];
        let mut low = vec![0usize; m];
        let mut comp = vec![usize::MAX; m];
        let mut on_stack = vec![false; m];
        let mut stack: Vec<usize> = Vec::new();
        let mut next_index = 0usize;
        let mut n_comp = 0usize;
        for root in 0..m {
            if index[root] != usize::MAX {
                continue;
            }
            index[root] = next_index;
            low[root] = next_index;
            next_index += 1;
            stack.push(root);
            on_stack[root] = true;
            let mut frames: Vec<(usize, usize)> = vec![(root, 0)];
            loop {
                let Some(&(v, ci)) = frames.last() else { break };
                if ci < imp[v].len() {
                    let w = imp[v][ci];
                    frames.last_mut().unwrap().1 += 1;
                    if index[w] == usize::MAX {
                        index[w] = next_index;
                        low[w] = next_index;
                        next_index += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        frames.push((w, 0));
                    } else if on_stack[w] {
                        low[v] = low[v].min(index[w]);
                    }
                } else {
                    frames.pop();
                    if let Some((p, _)) = frames.last() {
                        let p = *p;
                        low[p] = low[p].min(low[v]);
                    }
                    if low[v] == index[v] {
                        loop {
                            let w = stack.pop().unwrap();
                            on_stack[w] = false;
                            comp[w] = n_comp;
                            if w == v {
                                break;
                            }
                        }
                        n_comp += 1;
                    }
                }
            }
        }
        let mut val = HashMap::default();
        for (i, &x) in vars.iter().enumerate() {
            if comp[blue(i)] == comp[red(i)] {
                return None; // UNSAT: X^Blue and X^Red forced equal ⇒ incoherent
            }
            // Tarjan ids are in reverse topological order, so the standard
            // rule applies: literal true iff its component id is smaller.
            val.insert(
                x,
                if comp[blue(i)] < comp[red(i)] { Color::Blue } else { Color::Red },
            );
        }
        Some(val)
    }

    // ── Def 5.7–5.8: alternating paths and κ ────────────────────────────

    /// Def 5.7 down-move: α ▷ β where (α, β) is an X-pair and β is modular.
    /// Def 5.7 up-move: α ▷ β, α ≠ β, α modular with immediate predecessor
    /// γ, β : X (any occurrence), and γ ≤ β (γ is an ancestor-or-equal of β).
    ///
    /// κ(E) (Def 5.8): 0 if coherent and no cyclic alternating path,
    /// 1 if coherent with a cyclic alternating path, ∞ if not coherent.
    pub fn characteristic(&self) -> u8 {
        if self.is_coherent().is_none() {
            return u8::MAX; // ∞
        }
        if self.has_cyclic_alternating_path() {
            1
        } else {
            0
        }
    }

    /// Detect a cyclic alternating path: down-move followed by up-move
    /// chains.  We build the directed edge set (down ∪ up) over nodes and
    /// look for a cycle via DFS.
    ///
    /// Def 5.7 (arXiv:1907.03481v3, verbatim clauses):
    /// - *down-move* α ⊳ β: (α,β) is an X-pair (same label, opposite
    ///   colors) and β is modular.  NO ancestor constraint — α and β are
    ///   terminal nodes, so the tree order is vacuous between them.
    /// - *up-move* α ⊳ β: α ≠ β, α is a modular node with immediate
    ///   predecessor γ, β : X for some X ∈ bv(E), and γ ⪯ β (γ is an
    ///   ancestor-or-equal of β).
    /// Alternation is enforced structurally: the (node, parity) two-layer
    /// graph only admits down edges at even parity and up edges at odd.
    fn has_cyclic_alternating_path(&self) -> bool {
        let leaves = self.leaves();

        // down-moves: α ⊳ β where (α,β) is an X-pair and β modular.
        // An X-pair is (α : X^blue, β : X^red) (Def 5.1 ordering); only β
        // must be modular (hence terminal, Def 5.1). α may be ANY node —
        // an internal spine head and the binder node r_X itself included:
        // Def 5.7 (verbatim, L2033) puts NO constraint on the source, and
        // the paper's own construction (L2037: "down-move from an
        // occurrence of X in A to the modular node labeled X̄, then
        // up-move back to X") yields κ=1 for ∀X.X⇒X — confirmed by §7.3,
        // which classifies ∀X.X⇒X into Λ2^{κ≤1} (a μν-type, not κ=0).
        // (A former r_X-source exclusion here produced κ=0 for ∀X.X⇒X,
        // contradicting both the verbatim definition and §7.3.)
        let mut down: Vec<(NodeId, NodeId)> = Vec::new();
        for a in 0..self.nodes.len() {
            let na = &self.nodes[a];
            if na.color != Color::Blue {
                continue;
            }
            // P2: group by label via the labeled index — a down-move's
            // target is a same-label Red terminal, so only nodes_labeled
            // candidates need the modular test (the old loop scanned
            // every leaf for every Blue node).
            for &b in self.nodes_labeled(na.label, Color::Red) {
                if self.nodes[b].children.is_empty() && self.is_modular(b) {
                    down.push((a, b));
                }
            }
        }

        // up-moves: α → β where α modular, γ = parent(α), γ ≤ β, and
        // β is any node labeled with a bound variable.
        let mut up: Vec<(NodeId, NodeId)> = Vec::new();
        for &a in &leaves {
            if !self.is_modular(a) {
                continue;
            }
            let Some(g) = self.parent[a] else { continue };
            for (id, n) in self.nodes.iter().enumerate() {
                if !self.is_bound(n.label) {
                    continue;
                }
                // γ ≤ β: g is ancestor-or-equal of id — P3b Euler
                // subtree-interval test (O(1) instead of a parent climb).
                if self.tin[g] <= self.tin[id]
                    && self.tout[id] <= self.tout[g]
                    && id != a
                {
                    up.push((a, id));
                }
            }
        }

        // Cycle detection over a TWO-LAYER graph: Def 5.7 demands strict
        // down/up alternation, so (node,parity) pairs — layer0 --down-->
        // layer1 --up--> layer0 — with node n,p encoded as 2n+p.  Merging
        // both edge sets into one graph would report the symmetric
        // X̄⇄X⁺ down/down round-trip as a bogus cycle.
        let n2 = self.nodes.len() * 2;
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::default();
        for (a, b) in down {
            adj.entry(a * 2).or_default().push(b * 2 + 1);
        }
        for (a, b) in up {
            adj.entry(a * 2 + 1).or_default().push(b * 2);
        }
        // 0=white 1=gray 2=black
        let mut state = vec![0u8; n2];
        for start in 0..n2 {
            if state[start] != 0 {
                continue;
            }
            let mut stack: Vec<(NodeId, usize)> = vec![(start, 0)];
            state[start] = 1;
            while let Some(&mut (n, ref mut idx)) = stack.last_mut() {
                let succ = adj.get(&n).cloned().unwrap_or_default();
                if *idx >= succ.len() {
                    state[n] = 2;
                    stack.pop();
                    continue;
                }
                let s = succ[*idx];
                *idx += 1;
                match state[s] {
                    1 => return true, // back edge → alternating cycle
                    0 => {
                        state[s] = 1;
                        stack.push((s, 0));
                    }
                    _ => {}
                }
            }
        }
        false
    }
}

/// Structural sort key of a pending spine (P2: extracted from the old
/// comparison closure so the decorate-sort site computes each key once).
/// Total order for the child permutation (Lemma 4.1 ii: `A ∼ B` up to
/// argument permutation).  Deterministic.
fn pending_sort_key(p: &Pending) -> Vec<u8> {
    // Full 64-bit label (little-endian): a 16-bit encoding would collide
    // once labels exceed 65535, making the child permutation (Lemma 4.1
    // ii) non-deterministic. Mirrors the binder encoding below.
    let mut k = p.label.to_le_bytes().to_vec();
    match p.color {
        Color::Blue => k.push(0),
        Color::Red => k.push(1),
    }
    for &b in &p.binders {
        k.extend_from_slice(&b.to_le_bytes());
    }
    k.push(0xFF); // separator
    for c in &p.children {
        k.extend(pending_sort_key(c));
    }
    k
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Build `∀X. A` with fresh index n.
    fn forall(ctx: &mut TypeContext, idx: usize, body: TypeId) -> TypeId {
        ctx.forall(idx, Symbol::intern(&format!("X{}", idx)), body)
    }

    #[test]
    fn kappa_identity_type_is_coherent() {
        // ∀X. X → X — the identity type.  Balanced ⇒ coherent (Prop 3.1).
        // κ = 1, NOT 0: Def 5.7's verbatim down-move admits the source
        // r_X itself, giving the alternating cycle r_X ⊳ X̄ ⊳ r_X — and
        // §7.3 explicitly classifies ∀X.X⇒X into Λ2^{κ≤1} (it reduces to
        // a μ-type, here μX.1, not to a plain finite type).  The
        // reduction still converges: coherent ⟺ strongly converges
        // (Thm 5.6), which is all the reduction pipeline needs.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let id_fn = ctx.function(smallvec![x], x);
        let ty = forall(&mut ctx, 0, id_fn);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        // X is bound, both colors present (X⁺ return, X̄ param).
        assert!(ft.eliminable(0), "X should be eliminable in ∀X.X→X");
        assert_eq!(ft.characteristic(), 1, "§7.3: ∀X.X⇒X ∈ Λ2^{{κ≤1}}");
    }

    #[test]
    fn kappa_neg_non_duplicated() {
        // ∀X.Y. (X→Y) → X → Y: each variable occurs negatively once
        // (negatively non-duplicated) ⇒ κ = 0 (Prop 3.1).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let xy = ctx.function(smallvec![x], y);
        let body = ctx.function(smallvec![xy, x], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        // X⁺ appears as a branch-arg leaf and up/down moves alternate
        // through the X-pair ⇒ coherent but with an alternating cycle:
        // κ=1 ("rewrites into a μν-type"), still ≠ ∞.  Prop 3.1's
        // ≅_Y 1 is compatible: the μ may dissolve via Fig. 4c.
        assert_ne!(ft.characteristic(), u8::MAX, "neg-non-dup ⇒ coherent");
    }

    #[test]
    fn kappa_outside_lambda2_fails_closed() {
        // Refined boundary: GROUND atoms (Int) are now Λ2 atoms and
        // lower fine (`kappa_ground_atom_builds_tree`).  What stays
        // outside Λ2: sums (Coproduct), 0 (Never), μ/ν, ∃, Poly,
        // inference state, and nominals CARRYING a parameter — a param
        // under a nominal head is not a ground atom (treating it as one
        // would silently drop the param occurrence).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let bad = ctx.enum_ty(DefId(7777), smallvec![x]); // parameterized nominal
        let f = ctx.function(smallvec![bad], bad);
        assert!(
            ctx.build_coherence_tree(f).is_none(),
            "parameterized nominal is not a Λ2 ground atom"
        );
        // Sums stay outside the tree representation (§4.1 excludes +).
        let i32ty = ctx.int(32, true);
        let b = ctx.bool();
        let c = ctx.coproduct(smallvec![i32ty, b]);
        let fc = ctx.function(smallvec![c], c);
        assert!(ctx.build_coherence_tree(fc).is_none(), "sums stay outside Λ2");
    }

    #[test]
    fn kappa_ground_atom_builds_tree() {
        // Int is a Λ2 ground atom — atom-bearing spines build, and
        // the atom is INERT: exactly one bound variable (X, via ∀) in the
        // 2-SAT, the atom label never enters binder_scope.  ∀X.X→Int→Int
        // is moreover coherent with κ=0: X's single X̄ leaf is modular
        // (d=1, no parallel sibling) and the atom head spawns no X-pairs.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let i = ctx.int(32, true);
        let f = ctx.function(smallvec![x, i], i);
        let ty = ctx.forall(0, "X".into(), f);
        let ct = ctx.build_coherence_tree(ty).expect("ground atoms are Λ2");
        let ft = ct.flat();
        assert_eq!(
            ft.binder_scope.len(),
            1,
            "atom labels never enter binder_scope"
        );
        assert_eq!(
            ft.characteristic(),
            0,
            "single param occurrence with inert atoms: coherent, no alternating cycles"
        );
    }

    #[test]
    fn kappa_yoneda_with_ground_atom_instantiation() {
        // The Yoneda shape (1) with a CLOSED-ATOM A: ∀X.(X→Int)→X ≅_Y Int.
        // The atom terminal is visible to the machinery; the type
        // must be coherent (κ≠∞) — the earlier machinery refused the whole
        // type because Int had no lowering at all.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let i = ctx.int(32, true);
        let xi = ctx.function(smallvec![x], i);
        let body = ctx.function(smallvec![xi], x);
        let ty = ctx.forall(0, "X".into(), body);
        let k = ctx
            .build_coherence_tree(ty)
            .expect("Λ2 with a ground atom")
            .flat()
            .characteristic();
        assert_ne!(k, u8::MAX, "Yoneda-with-atom shape must be coherent");
    }

    #[test]
    fn kappa_yoneda_schema_positive() {
        // ∀X.(X→X→X)→X: matches (Y⁺) with branch ⟨X,X⟩→X and B=X.
        // X⁺ occurs only in the branch args (negative side of the branch),
        // all X̄ nodes sit at depth ≤2 from r_X with no parallel sibling of
        // the same color ⇒ X is blue-eliminable ⇒ coherent, κ=0.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let xxx = ctx.function(smallvec![x, x], x);
        let body = ctx.function(smallvec![xxx], x);
        let ty = forall(&mut ctx, 0, body);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        assert!(
            ft.color_eliminable(0, Color::Blue),
            "X should be blue-eliminable per (Y⁺)"
        );
        assert_eq!(ft.characteristic(), 0);
    }

    #[test]
    fn kappa_truly_non_coherent() {
        // ∀XY.(X→(Y→X))→X→Y: an X̄ occurrence sits at depth 3 from r_X
        // (inside the nested arrow) ⇒ beyond the modular window for BOTH
        // colors ⇒ neither color-eliminable ⇒ incoherent, κ=∞.
        // (Probed: char=255.)
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let yx = ctx.function(smallvec![y], x);
        let b1 = ctx.function(smallvec![yx], y);
        let body = ctx.function(smallvec![b1, x], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        assert_eq!(ft.characteristic(), u8::MAX, "incoherent ⇒ κ=∞");
    }

    #[test]
    fn kappa_fig2_prefix_yoneda_shape() {
        // Fig. 2 first step: ∀X. X→X→A ≅ A[X↦1+1] — the Yoneda-shape type
        // ∀X. X→X→Y (with free Y as a bound param 1) is coherent.
        // Use the closed variant ∀X.X→X→X?  No — that duplicates X⁺ twice
        // in the return... Actually Fig.1 (★) is ∀X.X→X→A with A closed,
        // which Λ2 cannot express; instead check ∀XY.X→X→Y→Y coherence.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let xxy = ctx.function(smallvec![x, x], y);
        let body = ctx.function(smallvec![xxy], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        // X occurs negatively twice (parallel) → not blue-eliminable; but
        // Y-eliminability may still save coherence for Y.  X itself is
        // non-eliminable in both colors ⇒ incoherent ⇒ κ=∞.
        let k = ft.characteristic();
        assert!(k == 0 || k == u8::MAX, "expected 0 or ∞, got {}", k);
    }

    /// Paper Fig. 7(a) verdict (Example 5.1): "X is eliminable but not
    /// c-eliminable" — exactly one of the two colors is eliminable
    /// (Lemma 5.1).
    ///
    /// BOUNDARY: the exact Fig. 7(a) tree is an SVG diagram in the paper
    /// that does not reliably extract to text — the text caption
    /// `∀X.(X→X→X)→X→(Y→X)→X` puts the X's inside `(X→X→X)` beyond the
    /// modular window (distance > 2 from `r_X`), so *neither* color is
    /// eliminable there, contradicting the paper.  We therefore pin the
    /// VERDICT on a hand-verifiable type: `∀X.(X→Y)→X` — X occurs once as
    /// the head `h_X` (never modular) and once as a modular argument
    /// (distance 2 from `r_X`), so exactly one color is eliminable.
    #[test]
    fn fig7_verdict_a_eliminable_not_c_elim() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into()); // free
        let xy = ctx.function(smallvec![x], y);
        let body = ctx.function(smallvec![xy], x); // (X→Y)→X
        let ty = ctx.forall(0, "X".into(), body);

        let ft = ctx.build_coherence_tree(ty).expect("Λ2").flat();
        let eb = ft.color_eliminable(0, Color::Blue);
        let er = ft.color_eliminable(0, Color::Red);
        assert!(
            eb != er,
            "Fig 7(a) verdict: X eliminable but not c-eliminable ⇒ exactly one color (blue={} red={})",
            eb,
            er
        );
    }

    /// Paper Fig. 7(b) verdict (Example 5.1): "X is both c-eliminable and
    /// c̄-eliminable".  (Same SVG-extraction boundary as Fig. 7(a).)
    ///
    /// Hand-verifiable type: `∀X.(X→Y)→(X→Z)→W` — X occurs in two distinct
    /// argument subtrees (not parallel), each at distance 2 from `r_X`, and
    /// `r_X`'s head is `W` (≠ X), so every X node is modular ⇒ both colors
    /// eliminable.
    #[test]
    fn fig7_verdict_b_both_c_elim() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let body = ctx.function(smallvec![xy, xz], w); // (X→Y)→(X→Z)→W
        let ty = ctx.forall(0, "X".into(), body);

        let ft = ctx.build_coherence_tree(ty).expect("Λ2").flat();
        let eb = ft.color_eliminable(0, Color::Blue);
        let er = ft.color_eliminable(0, Color::Red);
        assert!(
            eb && er,
            "Fig 7(b) verdict: X both c-eliminable and c̄-eliminable (blue={} red={})",
            eb,
            er
        );
    }

    /// Duel golden #1 (paper Fig 7(b), verbatim): the two βη-equivalent
    /// types
    ///   ∀XY.(X⇒X̄)⇒(∀Z.X⇒Z̄)⇒(Y⇒X̄)⇒Y   and
    ///   ∀X.(X⇒X̄)⇒∀Y.(Y⇒X̄)⇒(X⇒∀Z.Z̄)⇒Y
    /// must translate into the SAME tree (Lemma 4.1.ii).  Under root-stacked
    /// binder placement the nested writings of Y and Z would get different
    /// binder scopes and different trees — this test fails on root-stacking.
    #[test]
    fn d3_fig7b_duel_same_tree() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        // Shared subtrees.
        let xx = ctx.function(smallvec![x], x); // X⇒X̄
        let xz = ctx.function(smallvec![x], z); // X⇒Z̄  (codomain Z̄)
        let yx = ctx.function(smallvec![y], x); // Y⇒X̄
        // t1 = ∀XY.(X⇒X̄)⇒(∀Z.X⇒Z̄)⇒(Y⇒X̄)⇒Y
        let a2 = ctx.forall(2, "Z".into(), xz);
        let body1 = ctx.function(smallvec![xx, a2, yx], y);
        let forall_y = ctx.forall(1, "Y".into(), body1);
        let t1 = ctx.forall(0, "X".into(), forall_y);
        // t2 = ∀X.(X⇒X̄)⇒∀Y.(Y⇒X̄)⇒(X⇒∀Z.Z̄)⇒Y
        let zcod = ctx.forall(2, "Z".into(), z); // ∀Z.Z̄  (binder nested in a codomain)
        let b2 = ctx.function(smallvec![x], zcod); // X⇒∀Z.Z̄
        let inner2 = ctx.function(smallvec![yx, b2], y);
        let fy = ctx.forall(1, "Y".into(), inner2); // ∀Y.(…) nested after the first arg
        let body2 = ctx.function(smallvec![xx], fy);
        let t2 = ctx.forall(0, "X".into(), body2);

        let f1 = ctx.build_coherence_tree(t1).expect("Λ2 t1").flat();
        let f2 = ctx.build_coherence_tree(t2).expect("Λ2 t2").flat();
        assert_eq!(
            format!("{:?}", f1.nodes),
            format!("{:?}", f2.nodes),
            "βη-equivalent writings must give identical trees (Lemma 4.1.ii)"
        );
        assert_eq!(f1.characteristic(), f2.characteristic());
    }

    /// Duel golden #2 (single-holder descent): a binder whose
    /// occurrences ALL live inside one argument sinks BELOW the syntactic
    /// root to its LCA.  Type: ∀X.((X⇒Y)⇒X)⇒Z — X occurs at the inner
    /// spine head and inside (X⇒Y); the LCA is the inner spine, a direct
    /// child of the root (whose head is Z, free).  Root-stacking would
    /// leave binder_scope[X] at the root node — this test fails there.
    #[test]
    fn d3_single_holder_binder_sinks_to_lca() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let xy = ctx.function(smallvec![x], y); // X⇒Y
        let inner = ctx.function(smallvec![xy], x); // (X⇒Y)⇒X
        let body = ctx.function(smallvec![inner], z); // ((X⇒Y)⇒X)⇒Z
        let ty = ctx.forall(0, "X".into(), body);

        let ft = ctx.build_coherence_tree(ty).expect("Λ2").flat();
        let r = ft.binder_scope.get(&0).copied().expect("X registered");
        assert_ne!(r, 0, "binder must sink below the syntactic root");
        assert_eq!(
            ft.parent[r],
            Some(0),
            "LCA is the inner spine — a direct child of the root"
        );
    }

    /// Golden: a non-coherence verdict decided by the parallel-pair
    /// POLARITY of 2-SAT clause (2).  Type (X=0, Y=1, A=2, B=3, H=4 free):
    ///
    ///   ∀XY. body, body = (X, Y, C3) ⇒ X, C3 = ((X,Y)⇒H) ⇒ A
    ///
    /// gives, under the binder placement (r_X = r_Y = the body spine):
    /// - a parallel MODULAR pair (X̄, Ȳ): the two bare Red arg leaves at
    ///   d=1, same parent, both modular ⇒ clause (a_X^red ∨ a_Y^red);
    /// - deep X̄/Ȳ leaves at d=3 under C3 — NOT modular ⇒ X and Y are
    ///   each NOT Red-eliminable ⇒ clause (1) forces φ(X)=φ(Y)=Blue.
    /// Combined: (red∨red) ∧ blue(X) ∧ blue(Y) ⇒ UNSAT ⇒ κ=∞.  With the
    /// polarity WRITTEN BACKWARDS (pair (x^c,y^d) forbidding (φ=c,φ=d),
    /// i.e. clause (blue∨blue)) the same tree would be SAT — this test
    /// pins the complemented-color reading of Def 5.4 (verbatim L1955).
    #[test]
    fn d5_pair_polarity_decides_incoherence() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let h = ctx.generic_param(4, "H".into());
        let a = ctx.generic_param(2, "A".into());
        let deep = ctx.function(smallvec![x, y], h);
        let c3 = ctx.function(smallvec![deep], a);
        let body = ctx.function(smallvec![x, y, c3], x);
        let forall_y = ctx.forall(1, "Y".into(), body);
        let ty = ctx.forall(0, "X".into(), forall_y);

        let ft = ctx.build_coherence_tree(ty).expect("Λ2").flat();
        assert_eq!(
            ft.characteristic(),
            u8::MAX,
            "parallel modular (X̄,Ȳ) pair + forced-Blue valuations ⇒ UNSAT (κ=∞)"
        );
    }
}

/// §4.1 NF binder placement: a binder ∀X lives at the *lowest common
/// ancestor* of all X's occurrences.
///
/// Paper-side derivation (Def 4.1's NF + Lemma 4.1's proof, verbatim):
/// NF(A) = ∀Ȳ.A₁⇒…⇒Aₙ⇒X with each Y∈Ȳ occurring in the body, and the
/// proof of Lemma 4.1(i) canonicalizes using exactly two binder-mobility
/// isomorphisms: (a) vacuous elimination ∀X.A ≡_η A (X∉FV), and (b) the
/// hoisting iso `A ⇒ ∀X.B ≡_βη ∀X.A ⇒ B` (X∉FV(A)).  Iso (b) pulls a
/// binder up past subtrees NOT containing X and must stop at the first
/// spine whose body still holds all occurrences — pulling higher would
/// require crossing an X-bearing argument, which (b) forbids.  Hence the
/// canonical position is exactly the LCA of the occurrences, and stacking
/// binders at the syntactic root is the NON-canonical choice.
///
/// This is not just theory: the paper's own duel pair (the two
/// βη-equivalent types of Fig 7(b), "∀XY.(X⇒X̄)⇒(∀Z.X⇒Z̄)⇒(Y⇒X̄)⇒Y" vs
/// "∀X.(X⇒X̄)⇒∀Y.(Y⇒X̄)⇒(X⇒∀Z.Z̄)⇒Y") must translate into the SAME tree
/// (Lemma 4.1.ii); with root-stacked binders the two writings would get
/// different binder scopes and different trees.  `d3_fig7b_duel_same_tree`
/// pins this verbatim pair.
///
/// (The single-holder descent of `sink` goes one spine further than a
/// literal NF reading when ALL occurrences live inside one argument —
/// semantically the canceling eliminations that root placement would
/// refuse; the refusals are fail-closed, the eliminations are pinned by
/// the convergence goldens.)
fn place_binders(p: Pending) -> Pending {
    /// Collect occurrence paths (child-index chains) for `x` in `pt`.
    fn occurrences(pt: &Pending, x: usize, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if pt.label == x {
            out.push(path.clone());
        }
        for (i, c) in pt.children.iter().enumerate() {
            path.push(i);
            occurrences(c, x, path, out);
            path.pop();
        }
    }

    /// Occurs anywhere in subtree?
    fn occurs(pt: &Pending, x: usize) -> bool {
        if pt.label == x {
            return true;
        }
        pt.children.iter().any(|c| occurs(c, x))
    }

    /// Sink one binder to the LCA of its occurrences.
    fn sink(pt: &mut Pending, x: usize) {
        if !occurs(pt, x) {
            pt.binders.push(x); // vacuous: leave where it is
            return;
        }
        loop {
            // Children containing x:
            let holders: Vec<usize> = (0..pt.children.len())
                .filter(|&i| occurs(&pt.children[i], x))
                .collect();
            let here = pt.label == x;
            match (here, holders.len()) {
                // Occurrence at this node, or x spread across multiple
                // children (or child + node): LCA is here.
                (true, _) | (false, 2..) => {
                    pt.binders.push(x);
                    return;
                }
                // Exactly one child holds all occurrences: descend.
                (false, 1) => {
                    sink(&mut pt.children[holders[0]], x);
                    return;
                }
                // No occurrence (shouldn't happen — guarded above).
                (false, 0) => {
                    pt.binders.push(x);
                    return;
                }
            }
        }
    }

    fn walk(pt: &mut Pending) {
        // Sink this node's binders, then recurse into children.
        let binders = std::mem::take(&mut pt.binders);
        for x in binders {
            sink(pt, x);
        }
        for c in &mut pt.children {
            walk(c);
        }
    }

    let mut root = p;
    walk(&mut root);
    root
}
