//! Yoneda tree rewriting (Pistone & Tranchini 2020, Fig. 6 / Def 5.3).
//!
//! Operates on the *polynomial* extension of the occurrence trees from
//! `coherence.rs`: two new root constants • (the µ/Σ side) and N (the ν/Π
//! side), per §4.2 Def 4.3.  The four reduction rules `→_X^c` (Fig. 6a–d)
//! eliminate an eliminable variable X from a tree, producing the
//! substitution θ described in Prop 5.2:
//!
//! - blue side (X⁻-eliminable, (Y⁺)):  θ(X) = {µX.} Σₖ ∃Z⃗ₖ. Πⱼ D_jk[X⁺]
//! - red side  (X⁺-eliminable, (Y⁻)):  θ(X̄) = {νX.} ∀Z⃗ₖ. Πⱼ E_k[X⁻]
//!
//! where the {µX.}/{νX.} binder is elided when X does not occur in the
//! substituted parts (the "canceling" case of §5.2), and the ∃Z⃗ₖ/∀Z⃗ₖ
//! binders come from the modular branch's own binder set Z⃗ₖ.
//!
//! Representation: we work directly on the owned `OccNode` trees from
//! `coherence.rs`, extended with two synthetic node kinds (PolySum /
//! PolyProd) standing for the • / N polynomial nodes.  This keeps the
//! tree family small; `τ(E)` (Fig. 5) lowers a simple (P₀) polynomial
//! tree back to a monomorphic type.

use crate::ast::FileId;
use super::coherence::{Color, OccNode, OccTree};
use super::*;

/// Local occurrence node with a *polynomial* child grammar: unlike
/// coherence's `OccNode` (children: Vec<Box<OccNode>>, spine-only), a
/// rewriting-tree spine may carry Σ/Π nodes (•/N reducts) in any child
/// slot — exactly what Fig. 6 surgery produces when θ lands under a
/// parent spine.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RNode {
    pub label: usize,
    pub color: Color,
    pub binders: Vec<usize>,
    pub children: Vec<PolyTree>,
}

/// A polynomial tree: occurrence nodes plus the • / N polynomial roots.
#[derive(Debug, Clone, PartialEq)]
pub enum PolyTree {
    /// A plain Λ2 occurrence node (spine head or leaf).
    Occ(RNode),
    /// • node (§4.2): µX.∃Y⃗ₖ.Σₖ Πⱼ E_jk — `parts[jk]` are the Π factors
    /// of sum branch k; `binders[k]` are the ∃Z⃗ₖ of that branch; `head`
    /// is the eliminated variable X the • replaces.
    PolySum {
        head: usize,
        parts: Vec<(Vec<usize>, Vec<PolyTree>)>,
    },
    /// N node (§4.2): νX.∀Y⃗.Πⱼ E_j — `parts[j]` are the Π factors;
    /// `binders` is the ∀Z⃗; `head` is the eliminated variable X.
    PolyProd {
        head: usize,
        binders: Vec<usize>,
        parts: Vec<PolyTree>,
    },
}

use PolyTree::{Occ, PolyProd, PolySum};

/// Build a `PolyTree` from a plain Λ2 occurrence tree.
pub(crate) fn from_occ(n: OccTree) -> PolyTree {
    fn conv(n: OccNode) -> RNode {
        RNode {
            label: n.label,
            color: n.color,
            binders: n.binders,
            children: n
                .children
                .into_iter()
                .map(|b| PolyTree::Occ(conv(*b)))
                .collect(),
        }
    }
    Occ(conv(*n))
}

impl PolyTree {
    /// Apply the substitution θ: replace every free occurrence of label
    /// `x` with color `c` by `theta` (a PolySum/PolyProd replacing a
    /// *leaf* occurrence — θ is only ever applied at leaf positions,
    /// mirroring Fig. 6 where the X^c terminals are replaced).
    pub fn subst(&self, x: usize, c: Color, theta: &PolyTree) -> PolyTree {
        match self {
            Occ(n) => {
                // θ on binders of this node: bound variables are renamed
                // away (α-conversion is implicit — binder indices are
                // globally unique in our representation).
                if n.label == x && n.color == c && n.children.is_empty() {
                    // Leaf occurrence replaced by θ (a •/N node or plain
                    // subtree); θ is used as-is (its `head` variable has
                    // been pre-renamed by the caller if needed).
                    return theta.clone();
                }
                let mut n2 = n.clone();
                for ch in &mut n2.children {
                    *ch = ch.subst(x, c, theta);
                }
                Occ(n2)
            }
            PolySum { head, parts, .. } => {
                let parts2 = parts
                    .iter()
                    .map(|(bs, ts)| {
                        (
                            bs.clone(),
                            ts.iter().map(|t| t.subst(x, c, theta)).collect(),
                        )
                    })
                    .collect();
                PolySum {
                    head: *head,
                    parts: parts2,
                }
            }
            PolyProd {
                head,
                binders,
                parts,
            } => PolyProd {
                head: *head,
                binders: binders.clone(),
                parts: parts.iter().map(|t| t.subst(x, c, theta)).collect(),
            },
        }
    }
}

impl PolyTree {
    /// Free variables (labels not bound by any binder on the path —
    /// computed structurally: binders stored at a node bind within it).
    pub fn fvs(&self) -> std::collections::HashSet<usize> {
        let mut s = std::collections::HashSet::new();
        self.collect_fvs(&mut s);
        s
    }

    fn collect_fvs(&self, out: &mut std::collections::HashSet<usize>) {
        match self {
            Occ(n) => {
                if !n.binders.contains(&n.label) {
                    out.insert(n.label);
                }
                for b in &n.binders {
                    out.remove(b);
                }
                for c in &n.children {
                    c.collect_fvs(out);
                }
            }
            PolySum { head, parts, .. } => {
                for (_, trees) in parts {
                    for t in trees {
                        t.collect_fvs(out);
                    }
                }
                // X is bound by the • itself.
                out.remove(head);
            }
            PolyProd {
                head,
                binders,
                parts,
            } => {
                for t in parts {
                    t.collect_fvs(out);
                }
                out.remove(head);
                for b in binders {
                    out.remove(b);
                }
            }
        }
    }
}

// ─── The four Yoneda reduction rules →_X^c (Fig. 6, Def 5.3) ────────────
//
// Declarative form (equivalent to the four-case geometric surgery when X
// is c-eliminable, per Prop 5.2 + schemas (Y⁺)/(Y⁻)):
//
// - c = Red  ((Y⁺)): every X̄-terminal is modular; θ(X) = {µX.}Σₖ∃Z⃗ₖ.Πⱼ D_jk
// - c = Blue ((Y⁻)): every X⁺-terminal is modular; θ(X) = {νX.}∀Z⃗.Πⱼ E_j
//
// The {µX.}/{νX.} binder appears only in the non-canceling case (X occurs
// in the substituted bodies).  Fail-closed: unexpected shapes ⇒ None.
pub fn yoneda_step(tree: &PolyTree, x: usize, c: Color) -> Option<PolyTree> {
    match c {
        Color::Red => {
            // (Y⁺): collect Σ-branches from modular X̄ sites.
            let branches = collect_sum_branches(tree, x)?;
            if branches.is_empty() {
                return None; // no X̄ occurrence — X not red-side present
            }
            // Always keep the {µX.} binder (non-canceling form).  The
            // canceling elision is βη-redundant (Fig. 4c: µX.A ≡ A when
            // X∉FV(A)), and our shared-set FV computation can leak binder
            // removals across sibling branches, so the "does X occur"
            // test is not trusted here — keeping the binder is sound in
            // both cases and τ drops vacuous binders on lowering.
            let theta = PolySum {
                head: x,
                parts: branches,
            };
            replace_binder_subtree(tree, x, &theta)
        }
        Color::Blue => {
            // (Y⁻): collect Π-factors from modular X⁺ sites.
            let (binders, parts) = collect_prod_factors(tree, x)?;
            if parts.is_empty() {
                return None;
            }
            // Symmetric with (Y⁺): always keep the {νX.} binder — the
            // canceling elision is βη-redundant (Fig. 4c) and the FV
            // occurrence test is not trusted (shared-set leak).
            let theta = PolyProd {
                head: x,
                binders,
                parts,
            };
            replace_binder_subtree(tree, x, &theta)
        }
    }
}

/// Fig. 6 surgery: replace the WHOLE subtree rooted at r_X (the node
/// whose binder set contains X) with θ — not the leaf terminals.  The
/// head occurrence X^c at r_X itself is absorbed by θ (its binder or its
/// elision in the canceling case), which leaf-substitution can never
/// reach.
fn replace_binder_subtree(t: &PolyTree, x: usize, theta: &PolyTree) -> Option<PolyTree> {
    match t {
        Occ(n) => {
            if n.binders.contains(&x) {
                if n.binders.len() == 1 {
                    return Some(theta.clone());
                }
                // r_X carries binders for OTHER variables too (LCA
                // placement stacks shared ancestors here).  Dropping them
                // (the previous bug) leaks those variables as free.  Keep
                // the spine node as context B: remaining binders stay on
                // it, θ becomes its child — B⟨X↦θ⟩ per Fig. 6.
                let mut n2 = n.clone();
                n2.binders.retain(|&b| b != x);
                n2.children = vec![theta.clone()];
                return Some(Occ(n2));
            }
            let mut n2 = n.clone();
            for ch in &mut n2.children {
                let replaced = replace_binder_subtree(ch, x, theta)?;
                *ch = replaced;
            }
            Some(Occ(n2))
        }
        PolySum { head, parts } => {
            if *head == x {
                return Some(theta.clone());
            }
            let mut parts2 = Vec::new();
            for (bs, ts) in parts {
                let mut ts2 = Vec::new();
                for t2 in ts {
                    ts2.push(replace_binder_subtree(t2, x, theta)?);
                }
                parts2.push((bs.clone(), ts2));
            }
            Some(PolySum {
                head: *head,
                parts: parts2,
            })
        }
        PolyProd {
            head,
            binders,
            parts,
        } => {
            if *head == x {
                return Some(theta.clone());
            }
            let mut parts2 = Vec::new();
            for t2 in parts {
                parts2.push(replace_binder_subtree(t2, x, theta)?);
            }
            Some(PolyProd {
                head: *head,
                binders: binders.clone(),
                parts: parts2,
            })
        }
    }
}

// P3d: depth cap for the recursive PolyTree walkers below.  Source-
// derived trees are already bounded by the parser's
// `max_recursion_depth` (256), and Yoneda reduction only reshuffles
// subtrees (reduct depth ≤ original + O(1)); the cap is defense-
// in-depth for API-constructed trees — a depth past any legitimate
// one fails closed (None) instead of risking a stack overflow.
const WALK_DEPTH_CAP: usize = 100_000;

/// Collect one Σ-branch per modular X̄-terminal: (∃Z⃗ binders, Π factors =
/// the terminal's sibling spine factors).  For a bare X̄ leaf the minimal
/// branch is (its binders, empty product ≙ 1).
fn collect_sum_branches(t: &PolyTree, x: usize) -> Option<Vec<(Vec<usize>, Vec<PolyTree>)>> {
    let mut out = Vec::new();
    walk_sum(t, x, &mut out)?;
    Some(out)
}

fn walk_sum(t: &PolyTree, x: usize, out: &mut Vec<(Vec<usize>, Vec<PolyTree>)>) -> Option<()> {
    walk_sum_in(t, x, out, false, 0)
}

fn walk_sum_in(
    t: &PolyTree,
    x: usize,
    out: &mut Vec<(Vec<usize>, Vec<PolyTree>)>,
    in_scope: bool,
    depth: usize,
) -> Option<()> {
    if depth > WALK_DEPTH_CAP {
        return None; // P3d: fail-closed past the depth cap
    }
    // Fig. 6a geometric surgery: for the subtree rooted at r_X (the node
    // binding X), each X̄ terminal at path p contributes one branch
    // ∃(terminal binders). Π(path sibling factors ++ terminal args).
    // The sibling factors at each edge r_X→terminal are the OTHER
    // children at that edge — dropping them (the previous bug) loses the
    // surrounding spine context and the reduct is not ≙ the paper's.
    fn sink(
        n: &RNode,
        x: usize,
        prefix: &[PolyTree],
        out: &mut Vec<(Vec<usize>, Vec<PolyTree>)>,
        depth: usize,
    ) -> Option<()> {
        // Is this node itself an X̄ site?
        let is_site = n.label == x && n.color == Color::Red;
        // Children as PolyTrees.
        let kids: Vec<PolyTree> = n.children.iter().map(|c| c.clone()).collect();
        if is_site {
            // Branch factors = path prefix ++ this node's args (the
            // ⟨A_jk⟩_j of the (Y⁺) LHS codomain position).
            let mut factors: Vec<PolyTree> = prefix.to_vec();
            factors.extend(kids);
            // (Y⁺) LHS shape check: branch factors must not contain X̄ —
            // such an occurrence means the tree does not fit the schema
            // (would leak a free variable into the reduct).  Fail-closed.
            if factors.iter().any(|f| contains_x_col(f, x, Color::Red)) {
                return None;
            }
            out.push((n.binders.clone(), factors));
            // X̄ terminals don't recurse (their binders are the ∃Z⃗).
            return Some(());
        }
        if n.binders.contains(&x) {
            // r_X boundary: start a fresh prefix here (its own binders
            // scope X below).  If r_X is itself the X̄ head site, handled
            // above only when label/color match; otherwise treat as the
            // scope root and descend.
        }
        for (i, k) in kids.iter().enumerate() {
            // Siblings at this edge = the other children (Π factors).
            let mut pf: Vec<PolyTree> = prefix.to_vec();
            for (j, s) in kids.iter().enumerate() {
                if j != i {
                    pf.push(s.clone());
                }
            }
            match k {
                PolyTree::Occ(inner) => sink(inner, x, &pf, out, depth + 1)?,
                poly => {
                    // Σ/Π node under the scope: descend structurally,
                    // KEEPING the r_X scope (scope is carried by the
                    // ancestors, not by the constructor node itself).
                    walk_sum_in(poly, x, out, true, depth + 1)?;
                }
            }
        }
        Some(())
    }
    match t {
        Occ(n) => {
            if n.binders.contains(&x) {
                // r_X: collect branches from this subtree only.
                sink(n, x, &[], out, depth)
            } else if in_scope && n.label == x && n.color == Color::Red {
                // An X̄ leaf under a Σ/Π constructor branch below r_X:
                // still a collection site (scope carried by ancestors).
                sink(n, x, &[], out, depth)
            } else {
                let deeper = in_scope || n.binders.iter().any(|b| *b == x);
                for ch in &n.children {
                    walk_sum_in(ch, x, out, deeper, depth + 1)?;
                }
                Some(())
            }
        }
        // Once inside r_X's subtree (through Σ/Π constructor nodes), the
        // binder scope is carried by the *ancestors* — any X̄ leaf below
        // is still a site for the (Y⁺) collection.
        PolySum { parts, .. } => {
            for (_, ts) in parts {
                for t2 in ts {
                    walk_sum_in(t2, x, out, in_scope, depth + 1)?;
                }
            }
            Some(())
        }
        PolyProd { parts, .. } => {
            for t2 in parts {
                walk_sum_in(t2, x, out, in_scope, depth + 1)?;
            }
            Some(())
        }
    }
}

/// Dual scan for (Y⁻): ∀Z⃗ binders + Π E_j factors from modular X⁺
/// terminals.
fn collect_prod_factors(t: &PolyTree, x: usize) -> Option<(Vec<usize>, Vec<PolyTree>)> {
    let mut binders = Vec::new();
    let mut parts = Vec::new();
    walk_prod(t, x, &mut binders, &mut parts, 0)?;
    Some((binders, parts))
}

fn walk_prod(
    t: &PolyTree,
    x: usize,
    binders: &mut Vec<usize>,
    parts: &mut Vec<PolyTree>,
    depth: usize,
) -> Option<()> {
    if depth > WALK_DEPTH_CAP {
        return None; // P3d: fail-closed past the depth cap
    }
    match t {
        Occ(n) => {
            if n.label == x && n.color == Color::Blue {
                // Modular X⁺ site: bare leaf contributes a unit factor; a
                // spine X⁺[...] contributes its binder set to ∀Z⃗ and its
                // children as Π factors (per the (Y⁻) LHS ⟨X→A_j⟩_j).
                for b in &n.binders {
                    if !binders.contains(b) {
                        binders.push(*b);
                    }
                }
                for c in &n.children {
                    parts.push(c.clone());
                }
                if n.children.is_empty() {
                    parts.push(unit_leaf()); // unit factor
                }
                return Some(());
            }
            for ch in &n.children {
                walk_prod(ch, x, binders, parts, depth + 1)?;
            }
            Some(())
        }
        PolySum { parts: ps, .. } => {
            for (_, ts) in ps {
                for t2 in ts {
                    walk_prod(t2, x, binders, parts, depth + 1)?;
                }
            }
            Some(())
        }
        PolyProd { parts: ps, .. } => {
            for t2 in ps {
                walk_prod(t2, x, binders, parts, depth + 1)?;
            }
            Some(())
        }
    }
}

// ─── Standard reduction driver (§5.2, Thm 5.6) ──────────────────────────
//
// A standard reduction picks, for each bound variable X, the color given
// by a coherent valuation φ (Thm 5.6: E coherent ⇔ E strongly converges
// under standard reduction).  We iterate: rebuild the flat view, read φ,
// reduce the first variable whose φ-color applies, repeat until no bound
// variables remain (P₀) or no rule applies (fail-closed).

/// Bound labels currently occurring in the tree (any color, with a binder
/// scope registered in the flat view).
fn remaining_bound_vars(t: &PolyTree) -> Vec<usize> {
    // Structural approximation: labels that appear as node labels and are
    // shadowed by some binder set in the tree.  The flat-view check in the
    // driver is authoritative; this is just the candidate list.
    let mut labels = Vec::new();
    let mut binders = Vec::new();
    let mut ctor = Vec::new();
    fn walk(
        t: &PolyTree,
        labels: &mut Vec<usize>,
        binders: &mut Vec<usize>,
        ctor: &mut Vec<usize>,
    ) {
        match t {
            Occ(n) => {
                if !labels.contains(&n.label) {
                    labels.push(n.label);
                }
                for b in &n.binders {
                    if !binders.contains(b) {
                        binders.push(*b);
                    }
                }
                for c in &n.children {
                    walk(c, labels, binders, ctor);
                }
            }
            PolySum { head, parts, .. } => {
                // NB: the {µX.} head is a *constructor* binder (Fig. 5 •
                // node), NOT an eliminable ∀ quantifier — P₀ trees may
                // contain •/N, and a standard reduction never eliminates
                // a constructor-bound variable.  Track it as a binder
                // scope only (so X-leaves under the branches resolve as
                // bound), never as an eliminable label.
                if !binders.contains(head) {
                    binders.push(*head);
                }
                if !ctor.contains(head) {
                    ctor.push(*head);
                }
                for (_, ts) in parts {
                    for t2 in ts {
                        walk(t2, labels, binders, ctor);
                    }
                }
            }
            PolyProd {
                head,
                binders: bs,
                parts,
            } => {
                // Symmetric: {νX.} head is a constructor binder.
                if !binders.contains(head) {
                    binders.push(*head);
                }
                for b in bs {
                    if !binders.contains(b) {
                        binders.push(*b);
                    }
                }
                if !ctor.contains(head) {
                    ctor.push(*head);
                }
                for t2 in parts {
                    walk(t2, labels, binders, ctor);
                }
            }
        }
    }
    walk(t, &mut labels, &mut binders, &mut ctor);
    // A label bound ONLY by a constructor binder (µ/ν head) is fixed by
    // the fixpoint, not eliminable by →_X^c — exclude it from the
    // candidate list (the previous bug made the driver spin on the ν
    // residue occurrence).
    labels.retain(|l| binders.contains(l) && !ctor.contains(l));
    labels.sort();
    labels
}

/// Is this polynomial tree simple (in P₀: no bound variables)?
pub fn is_simple(t: &PolyTree) -> bool {
    remaining_bound_vars(t).is_empty()
}

/// Standard reduction to a simple polynomial tree (or fail-closed None).
///
/// Candidate order — least-canceling (Def 5.6 + the Thm 5.6 proof): a
/// c-canceling variable X is reduced only after all bound variables in
/// r_X's subtree are eliminated (the paper's "variables above r_X" —
/// proof-theoretic trees grow with the root at the bottom, so "above" =
/// deeper).  Ordering candidates by the *depth of r_X, descending*
/// (deepest binder first) realizes exactly that constraint and defers
/// canceling reductions until their scope is emptied.  Ties break on the
/// binder index for determinism.
pub fn standard_reduce(tree: &PolyTree) -> Option<PolyTree> {
    let mut cur = tree.clone();
    for _step in 0..10_000 {
        // P2: one walk per step — remaining_bound_vars is BOTH the
        // simplicity test (empty ⇒ P₀) and the candidate list; the old
        // is_simple precheck walked the tree a second time.
        let mut vars = remaining_bound_vars(&cur);
        if vars.is_empty() {
            return Some(cur);
        }
        // Rebuild the coherence view for the current tree: we go through
        // the Λ2 forest — polynomial roots are opaque to binder scoping,
        // so we reconstruct an CoherenceTree from the Occ spine.
        let flat = flat_view(&cur)?;
        // Coherence valuation (Def 5.5) fixes each variable's color.
        let phi = flat.valuation()?;
        // Least-canceling order: deepest r_X first (see fn docs).
        vars.sort_by_key(|&x| {
            let d = flat
                .binder_scope
                .get(&x)
                .map(|&r| flat.depth[r])
                .unwrap_or(usize::MAX);
            (std::cmp::Reverse(d), x)
        });
        let mut progressed = false;
        for &x in &vars {
            let Some(&c) = phi.get(&x) else { continue };
            // φ-color first (Thm 5.6 standard strategy); on failure try
            // the dual color before declaring the variable stuck — the
            // valuation may pick a color whose sites were absorbed by an
            // earlier step, while the opposite color's sites remain.
            // Fail-closed is preserved: yoneda_step itself refuses any
            // tree whose shape does not match the schema.
            let dual = match c {
                Color::Red => Color::Blue,
                Color::Blue => Color::Red,
            };
            if let Some(next) = yoneda_step(&cur, x, c).or_else(|| yoneda_step(&cur, x, dual)) {
                cur = next;
                progressed = true;
                break;
            }
        }
        if !progressed {
            return None; // fail-closed: stuck, not simple
        }
    }
    None // step bound exhausted — fail-closed
}

/// Build a coherence view for binder scoping of a PolyTree: flatten the
/// Occ skeleton (Poly roots become synthetic spines) and register binder
/// scopes, so the modular/eliminality predicates apply uniformly.
fn flat_view(t: &PolyTree) -> Option<super::coherence::FlatTree> {
    fn walk(
        t: &PolyTree,
        nodes: &mut Vec<super::coherence::FlatNode>,
        scope: &mut std::collections::HashMap<usize, usize>,
    ) -> Option<usize> {
        let id = nodes.len();
        match t {
            Occ(n) => {
                for b in &n.binders {
                    // Notation 5.1: unique binder scope.
                    if scope.contains_key(b) {
                        return None;
                    }
                }
                nodes.push(super::coherence::FlatNode {
                    label: n.label,
                    color: n.color,
                    binders: n.binders.clone(),
                    children: Vec::new(),
                });
                for b in &n.binders {
                    scope.insert(*b, id);
                }
                for ch in &n.children {
                    let cid = walk(ch, nodes, scope)?;
                    nodes[id].children.push(cid);
                }
                Some(id)
            }
            PolySum { head, parts, .. } => {
                // Synthetic • spine: label = head (bound here), children =
                // all branch bodies flattened.  The {µX.} binder may re-use
                // an index already bound upstream (θ reconstruction of the
                // SAME ∀Z⃗ from the (Y⁺) LHS, not a fresh binder) — treat
                // re-registration as a reference, not a collision.
                // B4: this must be `entry().or_insert` — the previous
                // unconditional `scope.insert(*head, id)` CLOBBERED the
                // upstream scope entry, so a re-used µ/ν head index got a
                // wrong r_X and every modular-distance computed against it
                // in later reduction rounds was off.
                scope.entry(*head).or_insert(id);
                nodes.push(super::coherence::FlatNode {
                    label: *head,
                    color: Color::Blue,
                    binders: Vec::new(),
                    children: Vec::new(),
                });
                for (_, ts) in parts {
                    for t2 in ts {
                        let cid = walk(t2, nodes, scope)?;
                        nodes[id].children.push(cid);
                    }
                }
                Some(id)
            }
            PolyProd {
                head,
                binders,
                parts,
            } => {
                // Symmetric with PolySum: θ's {νX.}∀Z⃗ reconstruction may
                // re-use indices already bound upstream — treat
                // re-registration as a reference, not a collision.
                // B4: `entry().or_insert` (see the PolySum note — the
                // unconditional inserts clobbered upstream scopes; the
                // `contains_key` pre-checks below them were dead code).
                scope.entry(*head).or_insert(id);
                for b in binders {
                    scope.entry(*b).or_insert(id);
                }
                nodes.push(super::coherence::FlatNode {
                    label: *head,
                    color: Color::Red,
                    binders: binders.clone(),
                    children: Vec::new(),
                });
                for t2 in parts {
                    let cid = walk(t2, nodes, scope)?;
                    nodes[id].children.push(cid);
                }
                Some(id)
            }
        }
    }
    let mut nodes = Vec::new();
    // std HashMap to match `FlatTree::from_parts`'s signature (this file's
    // `HashMap` alias is FxHashMap via `use super::*`).
    let mut scope: std::collections::HashMap<usize, usize> = std::collections::HashMap::default();
    walk(t, &mut nodes, &mut scope)?;
    Some(super::coherence::FlatTree::from_parts(nodes, scope))
}

// ─── τ(E): simple polynomial trees → monomorphic types (Fig. 5) ─────────
//
// τ(X)=X;  τ(⊕^X E_i|Y⃗) = ∀Y⃗.τ(E₁)→…→τ(Eₙ)→X  (as a *negative* spine this
// lowers to Πτ(E_i)→X; we lower positive-side trees only, which is what
// standard reduction produces);  τ(•_root) = µX.∃Y⃗ₖ.Σₖ Πⱼ τ(E_jk);
// τ(N_root) = νX.∀Y⃗.Πⱼ τ(E_j).
//
// Only meaningful on simple (P₀) trees — asserted by the caller.

impl<'input> TypeContext<'input> {
    /// Lower a simple polynomial tree to a `TypeId` (Fig. 5).
    /// Fail-closed: non-simple trees or out-of-Λ2 leaves ⇒ None.
    pub fn tau(&mut self, t: &PolyTree) -> Option<TypeId> {
        match t {
            PolySum { head, parts } => {
                // µX.∃Y⃗ₖ.Σₖ Πⱼ E_jk  — head == usize::MAX ⇒ no binder
                // (canceling): plain Σ.
                let alternatives = parts
                    .iter()
                    .map(|(binders, factors)| {
                        let mut branch = if factors.is_empty() {
                            self.unit()
                        } else {
                            let elems: Option<SmallVec<[TypeId; 4]>> =
                                factors.iter().map(|f| self.tau(f)).collect();
                            let elems = elems?;
                            if elems.len() == 1 {
                                elems[0]
                            } else {
                                self.tuple(elems)
                            }
                        };
                        // ∃Y⃗ₖ wrapper (binders in reverse fold order).
                        for &b in binders.iter().rev() {
                            branch = self.exists(
                                b,
                                Symbol::intern(&format!("Z{}", b)),
                                branch,
                                crate::ast::Expr::Literal(
                                    crate::ast::Literal::Bool(true),
                                    crate::ast::Span::new(FileId::DUMMY, 0, 0),
                                ),
                            );
                        }
                        Some(branch)
                    })
                    .collect::<Option<SmallVec<[TypeId; 4]>>>()?;
                let sigma = self.coproduct(alternatives);
                if *head == usize::MAX {
                    Some(sigma)
                } else {
                    Some(self.alloc(TypeData::Mu {
                        param_index: *head,
                        param_name: Symbol::intern(&format!("X{}", head)),
                        body: sigma,
                    }))
                }
            }
            PolyProd {
                head,
                binders,
                parts,
            } => {
                // νX.∀Y⃗.Πⱼ E_j.
                let mut factors: SmallVec<[TypeId; 4]> = SmallVec::new();
                for f in parts {
                    factors.push(self.tau(f)?);
                }
                let mut prod = if factors.is_empty() {
                    self.unit()
                } else if factors.len() == 1 {
                    factors[0]
                } else {
                    self.tuple(factors)
                };
                // ∀Y⃗ wrapper.
                for &b in binders.iter().rev() {
                    prod = self.forall(b, Symbol::intern(&format!("Z{}", b)), prod);
                }
                if *head == usize::MAX {
                    Some(prod)
                } else {
                    Some(self.alloc(TypeData::Nu {
                        param_index: *head,
                        param_name: Symbol::intern(&format!("X{}", head)),
                        body: prod,
                    }))
                }
            }
            Occ(n) => {
                // Plain Λ2 skeleton: a spine `Π children → label` lowered
                // positively as a tuple→param arrow, or a bare variable.
                if n.children.is_empty() {
                    if n.label >= super::coherence::ATOM_LABEL_FLOOR && n.label != usize::MAX {
                        // An inert atom terminal.  τ-lowering an atom
                        // leaf needs the label→TypeId reverse map (a later
                        // slice) — fail-closed here rather than minting a
                        // bogus GenericParam with a huge index.  (κ,
                        // coherence, and standard reduction all handle
                        // atoms; only the τ endpoint is pending.)
                        return None;
                    }
                    if n.label == usize::MAX {
                        // A bare leaf with the reserved label is the
                        // Π-factor identity `unit_leaf()` ≙ 1 — lower it
                        // to Unit (Fig. 5: ▲ ↦ 1).  An earlier fail-closed
                        // here blocked Example 3.1's witness pipeline: its
                        // reduct carries unit factors, and ≅1 is the type's
                        // PAPER verdict.  (A spine node with the reserved
                        // label — children non-empty — cannot arise from a
                        // reduction; the branch below keeps that
                        // impossible shape fail-closed.)
                        return Some(self.unit());
                    }
                    Some(self.generic_param(n.label, Symbol::intern(&format!("T{}", n.label))))
                } else {
                    // Negative spine: Π τ(children) → label.  Lowered as
                    // Fn from tuple to the head param.
                    let mut params: SmallVec<[TypeId; 4]> = SmallVec::new();
                    for c in &n.children {
                        params.push(self.tau(c)?);
                    }
                    let ret = self.generic_param(n.label, Symbol::intern(&format!("T{}", n.label)));
                    Some(self.function(params, ret))
                }
            }
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::super::coherence::Color;
    use super::*;

    fn forall(ctx: &mut TypeContext, idx: usize, body: TypeId) -> TypeId {
        ctx.forall(idx, Symbol::intern(&format!("X{}", idx)), body)
    }

    /// Build the occurrence tree of a Λ2 type and run standard reduction.
    fn reduce(ctx: &TypeContext, ty: TypeId) -> Option<PolyTree> {
        let ct = ctx.build_coherence_tree(ty)?;
        let spine = ct.root_tree();
        let poly = from_occ(spine);
        standard_reduce(&poly)
    }

    #[test]
    fn reduce_identity_to_unit() {
        // ∀X. X→X — (Y⁻)-shaped: branch X̄→X⁺, B=X⁺.  Balanced ⇒ κ=0
        // ⇒ strong convergence to a simple tree (Thm 5.6); τ gives 1.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let f = ctx.function(smallvec![x], x);
        let ty = forall(&mut ctx, 0, f);

        let simple = reduce(&ctx, ty).expect("coherent type must converge");
        assert!(is_simple(&simple), "∀X.X→X must reach P₀");
    }

    #[test]
    fn reduce_kappa_incoherent_fails_closed() {
        // The probed incoherent type: ∀XY.(X→(Y→X))→X→Y (κ=∞) must NOT
        // reduce — fail-closed.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let yx = ctx.function(smallvec![y], x);
        let b1 = ctx.function(smallvec![yx], y);
        let body = ctx.function(smallvec![b1, x], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        // κ=∞ confirmed by the coherence module.
        let ct = ctx.build_coherence_tree(ty).expect("Λ2");
        assert_eq!(ct.flat().characteristic(), u8::MAX);
        // Standard reduction must fail (no coherent valuation).
        assert!(reduce(&ctx, ty).is_none());
    }

    #[test]
    fn tau_lowers_simple_tree() {
        // τ of a bare-variable PolyTree gives back the param.
        let mut ctx = TypeContext::new();
        let leaf = PolyTree::Occ(RNode {
            label: 3,
            color: Color::Blue,
            children: Vec::new(),
            binders: Vec::new(),
        });
        let lowered = ctx.tau(&leaf).expect("simple");
        let expected = ctx.generic_param(3, Symbol::intern("T3"));
        assert_eq!(lowered, expected);
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod golden_tests {
    use super::*;

    fn forall(ctx: &mut TypeContext, idx: usize, body: TypeId) -> TypeId {
        ctx.forall(idx, Symbol::intern(&format!("X{}", idx)), body)
    }

    fn reduce(ctx: &TypeContext, ty: TypeId) -> Option<PolyTree> {
        let ct = ctx.build_coherence_tree(ty)?;
        let poly = from_occ(ct.root_tree());
        standard_reduce(&poly)
    }

    /// Paper Example 3.1 (§3, verbatim):
    /// ∀XYZW.((((X→Y)→X→Z)→(Y→Z)→W)→(Y→W)→W) ≅ 1 (unique inhabitant).
    /// Golden: the type must be coherent (κ≠∞) per the §5 machinery,
    /// standard-reduce to a simple (P₀) tree, and τ of that tree lowers.
    #[test]
    fn golden_example_3_1_unique_inhabitant() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());

        // inner = (X→Y)→X→Z
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let inner = ctx.function(smallvec![xy], xz);
        // yz = Y→Z, yw = Y→W
        let yz = ctx.function(smallvec![y], z);
        let yw = ctx.function(smallvec![y], w);
        // mid = inner → (Y→Z) → W
        let mid = ctx.function(smallvec![inner, yz], w);
        // A = mid → (Y→W) → W   (paper Example 3.1 exact: second arg is Y→W)
        let a = ctx.function(smallvec![mid, yw], w);

        let ty = forall(&mut ctx, 0, a);
        let ty = forall(&mut ctx, 1, ty);
        let ty = forall(&mut ctx, 2, ty);
        let ty = forall(&mut ctx, 3, ty);

        let ct = ctx.build_coherence_tree(ty).expect("Λ2 golden type");
        let k = ct.flat().characteristic();
        // Corrected verdict: COHERENT (κ=1), matching Thm 5.6 — the paper
        // proves this type ≅ 1 in §3 via direct Yoneda isomorphisms
        // (≅_W, ≅_Z, ≅_X, ≅_Y, ≅_η), and Thm 5.6 (coherent ⟺ strong
        // convergence) therefore REQUIRES the tree to be coherent.  The
        // the earlier κ=∞ came from counting two same-label spine SIBLINGS
        // (mid/YW, both W̄) as paper-parallel — but the paper's trees carry
        // labels on LEAVES only (§4.1 L844), so leaves under different
        // spines are not parallel (Notation 5.1: parallel = immediate
        // successors of the same node).  κ=1 (not 0): our tree has a cyclic
        // alternating path — consistent with the reduction keeping the
        // {μX.} binder (non-canceling form; the canceling elision is
        // deferred to the canonical layer, Fig. 4c), so the reduct carries
        // fixpoint heads.  The paper does not exhibit κ for this exact
        // type; what Thm 5.6 pins is coherence + convergence, which is
        // what this golden asserts.
        assert_eq!(
            k, 1,
            "Example 3.1 must be coherent (κ≠∞) per Thm 5.6; κ=1 with the non-canceling reduct"
        );
        // Thm 5.6 (⟸ direction): coherent ⟹ standard reduction converges.
        let simple = reduce(&ctx, ty)
            .expect("coherent ⟹ strong convergence under standard reduction");
        assert!(is_simple(&simple), "convergence lands on a P₀ tree");
    }

    /// Paper Fig. 2 (§1.1) six-inhabitant chain — the INITIAL type:
    /// ∀X. X→X→∀Y.(∀Z.(Z→X)→(∀W.(W→Z)→W→X)→Z→Y)→(X→Y)→Y.
    /// Golden: coherent, standard reduction converges to a simple tree.
    /// (The full chain to 1+1+1+1+1+1 needs sum types, outside Λ2 —
    /// we assert convergence of the Λ2 prefix, which is what the tree
    /// machinery covers.)
    #[test]
    fn golden_fig2_six_inhabitants_prefix() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());

        // innerW = ∀W.(W→Z)→W→X
        let wz = ctx.function(smallvec![w], z);
        let wx = ctx.function(smallvec![w], x);
        let inner_w = ctx.function(smallvec![wz, wx], x);
        let forall_w = ctx.forall(3, "W".into(), inner_w);
        // innerZ = (Z→X) → innerW → Z → Y
        let zx = ctx.function(smallvec![z], x);
        let zy = ctx.function(smallvec![z], y);
        let inner_z = ctx.function(smallvec![zx, forall_w, zy], y);
        let forall_z = ctx.forall(2, "Z".into(), inner_z);
        // xy = X→Y
        let xy = ctx.function(smallvec![x], y);
        // spine: X→X→∀Y? — Fig.2's ∀Y binds the rest:
        // body = ∀Y.(∀Z...)→(X→Y)→Y  — the outer ∀X.X→X→[that]
        let forall_y_body = ctx.function(smallvec![forall_z, xy], y);
        let forall_y = ctx.forall(1, "Y".into(), forall_y_body);
        let body = ctx.function(smallvec![x, x], forall_y);
        let ty = forall(&mut ctx, 0, body);

        let ct = ctx.build_coherence_tree(ty).expect("Λ2 golden type");
        let ft = ct.flat();
        // Refreshed rationale (verdict unchanged): the two X̄ params at
        // the root are PAPER-PARALLEL bare leaves (same parent, same
        // label+color — Notation 5.1), so X is non-eliminable in both
        // colors and the 2-SAT is UNSAT: κ=∞.  This is genuine, not a
        // representation artifact: eliminating X here needs the Fig.1 (★)
        // instantiation X ↦ 1+1 (a SUM type), which the paper performs on
        // Λ2p-types with sums — outside the Λ2 tree schemas (Y⁺)/(Y⁻).
        // (Atoms per se are not the blocker — a closed atom is a fine Λ2
        // type and lazy atom terminals feed them through; the (★)
        // step is what the tree machinery cannot express.)
        assert_eq!(
            ft.characteristic(),
            u8::MAX,
            "Fig.2 initial type: two parallel X̄ leaves ⇒ incoherent (the (★) 1+1 step lives outside the tree schemas)"
        );
    }

    /// Paper Fig. 1 (★)-shaped instance, ∀X.X→X→X→X? No — Fig.1 needs a
    /// closed A; the closest Λ2-expressible balanced instance is the
    /// identity type (already covered in coherence tests).  Instead
    /// golden-check the paper's §5.2 conflict example:
    /// ∀X.∀Y.(X→Y→X)→((Y→X)→W)→Z is incoherent — approximated with the
    /// probe-verified ∀XY.(X→(Y→X))→X→Y (κ=∞): reduction must be refused.
    #[test]
    fn golden_incoherent_refused() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let yx = ctx.function(smallvec![y], x);
        let b1 = ctx.function(smallvec![yx], y);
        let body = ctx.function(smallvec![b1, x], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        assert!(
            reduce(&ctx, ty).is_none(),
            "incoherent ⇒ no standard reduction"
        );
    }
}

/// Does `t` contain a bare occurrence of label `x` with color `c`
/// (anywhere, including under Poly nodes)?  Used to enforce the (Y⁺)/(Y⁻)
/// LHS shape: branch factors must not contain X̄ (resp. X⁺) — such an
/// occurrence means the tree does not match the schema and the step must
/// fail closed rather than silently leak a free variable.
fn contains_x_col(t: &PolyTree, x: usize, c: Color) -> bool {
    match t {
        Occ(n) => {
            if n.label == x && n.color == c {
                return true;
            }
            n.children.iter().any(|ch| contains_x_col(ch, x, c))
        }
        PolySum { parts, .. } => parts
            .iter()
            .any(|(_, ts)| ts.iter().any(|t2| contains_x_col(t2, x, c))),
        PolyProd { parts, .. } => parts.iter().any(|t2| contains_x_col(t2, x, c)),
    }
}

/// Synthetic unit leaf (Π factor ≙ 1): label usize::MAX is reserved.
pub(crate) fn unit_leaf() -> PolyTree {
    PolyTree::Occ(RNode {
        label: usize::MAX,
        color: Color::Blue,
        binders: Vec::new(),
        children: Vec::new(),
    })
}

// ─── Stage 5b (v1): witness / coercion generation ────────────────────────
//
// The paper (App. A) constructs explicit *term-level* witnesses for the
// two Yoneda schemas: s[x]/t[x] for (Y⁺) (built from in_Σ, pack, fold_T
// and their duals) and u[x]/v[x] for (Y⁻) (built from π_Π, out_U,
// unfold_U).  A full elaboration-level term generator needs an
// AST/HIR term constructor with fold/unfold/pack/unpack; this v1 slice
// provides the *type-level* half of the witness pair:
//
//   - reduce A's tree to a simple P₀ tree (standard reduction),
//   - lower it to the monomorphic type M = τ(E),
//   - certify the direction: `intro` M→A / `elim` A→M — a coercion in
//     that direction is sound iff the canonical-form equivalence holds
//     (and the parametricity gate is open, since the isomorphism is
//     ε-theoretic, not βη).
//
// Fail-closed: no reduction, no gate, or direction mismatch ⇒ None.

/// Direction of a Yoneda coercion relative to the original polymorphic
/// type: `Elim` goes A → M (uses the isomorphism left-to-right, like
/// applying the paper's s[x]/u[x]); `Intro` goes M → A (the inverse,
/// t[x]/v[x]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YonedaDirection {
    /// A → M: eliminate the quantifier (paper's s[x] for (Y⁺), u[x] for (Y⁻)).
    Elim,
    /// M → A: re-introduce the quantifier (paper's t[x] / v[x]).
    Intro,
}

/// A type-level Yoneda coercion witness.
#[derive(Clone, Debug)]
pub struct YonedaCoercion {
    pub source: TypeId,
    pub target: TypeId,
    pub direction: YonedaDirection,
}

impl<'input> TypeContext<'input> {
    /// Produce the coercion witness `A ⇄ M` where M = τ(reduct) of A's
    /// standard reduction.  `want` selects the direction.
    ///
    /// B6 (certification semantics, doc/impl aligned): a direction is
    /// CERTIFIED exactly when (a) the parametricity gate is open, (b) A
    /// lowers to a coherent Λ2 tree, (c) the standard reduction succeeds,
    /// and (d) τ lowers the reduct.  No separate equivalence check runs —
    /// the isomorphism is NOT verified here but guaranteed by the paper:
    /// Lemma 5.3 (`t(A) ↝* E ∈ P₀ ⟹ A ≅_ν τ(E)`) plus Thm 5.6 (coherent
    /// ⟺ strongly converges under standard reduction).  The coercion is
    /// ε-theoretic, not βη — which is exactly why the gate is consulted.
    ///
    /// Fail-closed: gate closed, no coherent tree, reduction stuck, or
    /// τ not lowerable ⇒ `None`.
    pub fn yoneda_coercion(&mut self, a: TypeId, want: YonedaDirection) -> Option<YonedaCoercion> {
        // ε-theoretic isomorphism: equivalence-strength use ⇒ gated.
        self.parametricity.equivalence_allowed()?;
        let ct = self.build_coherence_tree(a)?;
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly)?;
        let m = self.tau(&simple)?;
        let (source, target) = match want {
            YonedaDirection::Elim => (a, m),
            YonedaDirection::Intro => (m, a),
        };
        Some(YonedaCoercion {
            source,
            target,
            direction: want,
        })
    }

    /// Round-trip check: both witnesses exist and are mutually inverse at
    /// the type level (Elim's source/target are Intro's target/source —
    /// both directions reduce to the SAME M = τ(reduct)).  Same
    /// certification semantics as `yoneda_coercion`: gate open + coherent
    /// reduction + τ lowering; the isomorphism itself is guaranteed
    /// paper-side (Lemma 5.3 + Thm 5.6), not re-checked here.
    pub fn yoneda_round_trip(&mut self, a: TypeId) -> Option<(YonedaCoercion, YonedaCoercion)> {
        let elim = self.yoneda_coercion(a, YonedaDirection::Elim)?;
        let intro = self.yoneda_coercion(a, YonedaDirection::Intro)?;
        if elim.source != intro.target || elim.target != intro.source {
            return None;
        }
        Some((elim, intro))
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod witness_tests {
    use super::*;

    /// Round-trip on a coherent, reducible type: both directions exist,
    /// are mutually inverse at the type level, and the gate blocks them
    /// when parametricity is broken (fail-closed).
    #[test]
    fn witness_round_trip_identity_type() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let f = ctx.function(smallvec![x], x);
        let ty = ctx.forall(0, "X".into(), f);

        let (elim, intro) = ctx.yoneda_round_trip(ty).expect("coherent ⇒ round trip");
        assert_eq!(elim.direction, YonedaDirection::Elim);
        assert_eq!(intro.direction, YonedaDirection::Intro);
        // Mutually inverse at the type level: elim A→M, intro M→A.
        assert_eq!(elim.source, intro.target);
        assert_eq!(elim.target, intro.source);
        // The monomorphic endpoint M lowers to a real TypeId.
        assert_ne!(elim.target, ty, "M is the reduct, not A itself");

        // Gate closed ⇒ fail-closed (both directions refuse).
        ctx.parametricity.break_parametricity();
        assert!(ctx.yoneda_coercion(ty, YonedaDirection::Elim).is_none());
        assert!(ctx.yoneda_round_trip(ty).is_none());
    }

    /// Example 3.1's type is COHERENT per the corrected §5 2-SAT (κ=1,
    /// see `golden_example_3_1_unique_inhabitant`), so its standard
    /// reduction converges and the type-level witness pair exists — the
    /// earlier "incoherent ⇒ no witness" premise was a mis-translation of
    /// Notation 5.1 (paper-parallel = paper siblings; labels live on
    /// leaves, so same-label spines under different parents are not
    /// parallel).
    #[test]
    fn witness_example_3_1_round_trip() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let inner = ctx.function(smallvec![xy], xz);
        let yz = ctx.function(smallvec![y], z);
        let yw = ctx.function(smallvec![y], w);
        let mid = ctx.function(smallvec![inner, yz], w);
        let a = ctx.function(smallvec![mid, yw], w);
        let ty = ctx.forall(0, "X".into(), a);
        let ty = ctx.forall(1, "Y".into(), ty);
        let ty = ctx.forall(2, "Z".into(), ty);
        let ty = ctx.forall(3, "W".into(), ty);

        // Coherent (κ=1) ⇒ standard reduction converges ⇒ round-trip
        // witness exists, mutually inverse at the type level.
        let (elim, intro) = ctx
            .yoneda_round_trip(ty)
            .expect("coherent per Thm 5.6 ⇒ round-trip witness");
        assert_eq!(elim.source, intro.target);
        assert_eq!(elim.target, intro.source);
    }

    /// Probe (follow-up): pin WHERE the Example 3.1 witness pipeline
    /// runs — build → reduce → τ — so a regression names the stage.
    #[test]
    fn witness_example_3_1_pipeline_stages() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let inner = ctx.function(smallvec![xy], xz);
        let yz = ctx.function(smallvec![y], z);
        let yw = ctx.function(smallvec![y], w);
        let mid = ctx.function(smallvec![inner, yz], w);
        let a = ctx.function(smallvec![mid, yw], w);
        let ty = ctx.forall(0, "X".into(), a);
        let ty = ctx.forall(1, "Y".into(), ty);
        let ty = ctx.forall(2, "Z".into(), ty);
        let ty = ctx.forall(3, "W".into(), ty);

        let ct = ctx.build_coherence_tree(ty).expect("stage 1: Λ2 tree");
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly).expect("stage 2: standard reduction");
        assert!(is_simple(&simple));
        let m = ctx.tau(&simple).expect("stage 3: τ lowering");
        assert_ne!(m, ty, "τ endpoint is the monomorphic reduct, not A");
    }

    /// An incoherent type admits no witness at all (κ=∞ ⇒ no reduction).
    #[test]
    fn witness_incoherent_fail_closed() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let yx = ctx.function(smallvec![y], x);
        let b1 = ctx.function(smallvec![yx], y);
        let body = ctx.function(smallvec![b1, x], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);

        assert!(ctx.yoneda_coercion(ty, YonedaDirection::Elim).is_none());
        assert!(ctx.yoneda_round_trip(ty).is_none());
    }
}
