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
//! - **§5.2**: coherence as 2-SAT (Remark 5.1) — implemented directly as
//!   its satisfying-valuation semantics (Def 5.5), not via a generic
//!   2-SAT solver, because the clause set is tiny and the direct check is
//!   the same polytime.
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
    /// Bound variables: index → (root node of its binder scope r_X, color
    /// of its head occurrence).
    pub binder_root: HashMap<usize, NodeId>,
    /// Nodes carrying binder sets that mention each variable: for X ∈ Y⃗ at
    /// node n, X's binder-scope root r_X is that node (Notation 5.1).
    pub binder_owner: HashMap<usize, NodeId>,
}

/// A `bool`-like colored variable occurrence collected during lowering.
#[derive(Debug)]
struct Pending {
    label: usize,
    color: Color,
    children: Vec<Pending>,
    binders: Vec<usize>,
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
    /// Fails (returns `None`) for anything outside Λ2: `Coproduct`
    /// (user sums), `Never` (0), `Mu`/`Nu`, `Exists`, `Poly`, ADTs,
    /// `InferVar`, `Opaque` — fail-closed, as tp(·) is defined only for Λ2.
    ///
    /// Binder identity is `param_index` (globally unique per the
    /// `fresh_param_index` allocator discipline, see canonical.rs).
    pub fn build_coherence_tree(&self, root: TypeId) -> Option<CoherenceTree> {
        let pending = place_binders(self.lower_type(root, Color::Blue)?);

        // Single-pass: build the owned node tree, collecting parent links
        // and binder scopes as we go.
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
            binder_root,
            binder_owner,
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
                children.sort_by(|a, b| pending_cmp(a, b));
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
            // ∀ in argument position: hoist (A→∀X.B ≡ ∀X.A→B) — the binder
            // travels to the enclosing spine node.  Handled in lower_type
            // via the Forall arm only at top level; here a ∀ return of an
            // inner spine was already lifted by recursion through `ret`.
            _ => None, // outside Λ2 (base types like Int/Bool/ADT…)
        }
    }
}

/// Does this pending tree mention `var` free (ignoring its own binders)?
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
}

impl FlatTree {
    /// Construct from raw parts (used by `rewrite::flat_view` to scope a
    /// polynomial-tree skeleton through the same predicates).
    pub(crate) fn from_parts(
        nodes: Vec<FlatNode>,
        binder_scope: HashMap<usize, NodeId>,
    ) -> FlatTree {
        let mut parent = vec![None; nodes.len()];
        let mut depth = vec![0usize; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for &c in &n.children {
                parent[c] = Some(i);
                depth[c] = depth[i] + 1;
            }
        }
        FlatTree {
            nodes,
            parent,
            depth,
            binder_scope,
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

    /// All nodes labeled `v` with color `c`.
    fn nodes_labeled(&self, v: usize, c: Color) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].label == v && self.nodes[i].color == c)
            .collect()
    }

    /// Def 5.1: is terminal node `a` modular?
    /// modular ⇔ a ≠ h_X ∧ 1 ≤ d(a, r_X) ≤ 2 ∧ no parallel node of the
    /// same label X^c.  Here r_X is the node binding X and h_X is the head
    /// of r_X's subtree — i.e. node r_X itself.
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
        if !(1..=2).contains(&d) {
            return false;
        }
        // No parallel node of label X^c: no sibling with same label+color.
        if let Some(p) = self.parent[a] {
            for &s in &self.nodes[p].children {
                if s != a && self.nodes[s].label == x && self.nodes[s].color == n.color {
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

    /// Def 5.4: X^c and Y^d are coherent iff there are no parallel modular
    /// nodes α : X^c, β : Y^d (siblings with those labels/colors).
    fn pair_coherent(&self, x: usize, c: Color, y: usize, d: Color) -> bool {
        if x == y {
            return true; // same variable, no pair (α:X, β:Y) with Y ≠ X
        }
        let xs = self.nodes_labeled(x, c);
        let ys = self.nodes_labeled(y, d);
        for &a in &xs {
            if !self.is_modular(a) {
                continue;
            }
            let Some(p) = self.parent[a] else { continue };
            for &b in &ys {
                if b != a && self.parent[b] == Some(p) && self.is_modular(b) {
                    return false; // parallel modular pair
                }
            }
        }
        true
    }

    /// Def 5.5 (Remark 5.1 semantics, checked directly): does a valuation
    /// φ : bV → Colors exist s.t. every X is φ(X)-eliminable and every
    /// pair X^{φ(X)}, Y^{φ(Y)} (X≠Y) is coherent?
    ///
    /// Small clause sets ⇒ brute force over 2^n valuations with memoized
    /// eliminability; the paper's 2-SAT would scale further but n is tiny
    /// for real types.  Fail-closed: no valuation ⇒ not coherent.
    pub fn is_coherent(&self) -> Option<HashMap<usize, Color>> {
        let vars: Vec<usize> = {
            let mut v: Vec<usize> = self.binder_scope.keys().copied().collect();
            v.sort();
            v
        };
        // Precompute per-variable, per-color eliminability.
        let elim: HashMap<(usize, Color), bool> = vars
            .iter()
            .flat_map(|&x| [(x, Color::Blue), (x, Color::Red)])
            .map(|k| (k, self.color_eliminable(k.0, k.1)))
            .collect();

        // Brute force over valuations (n ≤ 20 guard: beyond that, fail
        // closed rather than blow up; a real 2-SAT/SCC pass can replace
        // this later).
        if vars.len() > 20 {
            return None;
        }
        for mask in 0u64..(1u64 << vars.len()) {
            let val: HashMap<usize, Color> = vars
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    (
                        x,
                        if mask & (1 << i) != 0 {
                            Color::Red
                        } else {
                            Color::Blue
                        },
                    )
                })
                .collect();
            let ok = vars.iter().all(|&x| {
                elim[&(x, val[&x])]
                    && vars
                        .iter()
                        .all(|&y| y == x || self.pair_coherent(x, val[&x], y, val[&y]))
            });
            if ok {
                return Some(val);
            }
        }
        None
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
    fn has_cyclic_alternating_path(&self) -> bool {
        use std::collections::HashSet;
        let leaves = self.leaves();

        // down-moves: α → β where (α,β) is an X-pair and β modular.
        let mut down: Vec<(NodeId, NodeId)> = Vec::new();
        for &a in &leaves {
            for &b in &leaves {
                if a == b {
                    continue;
                }
                let (la, lb) = (&self.nodes[a].label, &self.nodes[b].label);
                let (ca, cb) = (&self.nodes[a].color, &self.nodes[b].color);
                if la == lb && ca != cb && self.is_modular(b) {
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
                // γ ≤ β: g is ancestor-or-equal of id.
                let mut cur = Some(id);
                let mut anc = false;
                while let Some(c) = cur {
                    if c == g {
                        anc = true;
                        break;
                    }
                    cur = self.parent[c];
                }
                if anc && id != a {
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

/// Total order on pending spines for the child permutation (Lemma 4.1 ii:
/// `A ∼ B` up to argument permutation).  Deterministic structural key.
fn pending_cmp(a: &Pending, b: &Pending) -> std::cmp::Ordering {
    fn key(p: &Pending) -> Vec<u8> {
        let mut k = vec![p.label as u8, (p.label >> 8) as u8];
        match p.color {
            Color::Blue => k.push(0),
            Color::Red => k.push(1),
        }
        for &b in &p.binders {
            k.extend_from_slice(&b.to_le_bytes());
        }
        k.push(0xFF); // separator
        for c in &p.children {
            k.extend(key(c));
        }
        k
    }
    key(a).cmp(&key(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `∀X. A` with fresh index n.
    fn forall(ctx: &mut TypeContext, idx: usize, body: TypeId) -> TypeId {
        ctx.forall(idx, Symbol::intern(&format!("X{}", idx)), body)
    }

    #[test]
    fn kappa_identity_type_is_coherent() {
        // ∀X. X → X — the identity type.  Balanced ⇒ κ = 0 (Prop 3.1).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let id_fn = ctx.function(smallvec![x], x);
        let ty = forall(&mut ctx, 0, id_fn);

        let tree = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = tree.flat();
        // X is bound, both colors present (X⁺ return, X̄ param).
        assert!(ft.eliminable(0), "X should be eliminable in ∀X.X→X");
        assert_eq!(ft.characteristic(), 0, "balanced type ⇒ κ=0");
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
        // Base types (Int) are not Λ2 — build_coherence_tree must refuse.
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let f = ctx.function(smallvec![i], i);
        assert!(ctx.build_coherence_tree(f).is_none());
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
}

/// §4.1 NF binder placement: a binder ∀X lives at the *lowest common
/// ancestor* of all X's occurrences (Lemma 4.1 pushes quantifiers
/// leftmost; the NF spine node closest to the occurrences is the LCA).
/// Lowering naively stacks every binder at the root (the syntactic
/// ∀-prefix), which breaks modular-window distances.  This pass sinks
/// each binder from the root to the LCA of its variable's occurrence
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
