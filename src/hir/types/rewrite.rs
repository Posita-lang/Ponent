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

use super::coherence::{Color, OccNode, OccTree};
use super::*;
use crate::ast::FileId;

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
                // Scope-lifted reduct node: when the LAST child is a Poly
                // node whose head equals this node's label, or whose head
                // occurs in an earlier F_l child (residue), the µ/ν from
                // that Poly scopes over the WHOLE node — the label is not
                // free, and the F_l residues are bound (removed below).
                let lifted_head = match n.children.last() {
                    Some(PolySum { head, .. }) | Some(PolyProd { head, .. }) => {
                        let h = *head;
                        let residues = n.children[..n.children.len() - 1].iter().any(|f| {
                            contains_x_col(f, h, Color::Red) || contains_x_col(f, h, Color::Blue)
                        });
                        if h == n.label || residues {
                            Some(h)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if !lifted_head.is_some() && !n.binders.contains(&n.label) {
                    out.insert(n.label);
                }
                for b in &n.binders {
                    out.remove(b);
                }
                for c in &n.children {
                    c.collect_fvs(out);
                }
                if let Some(h) = lifted_head {
                    out.remove(&h); // F_l residues are µ/ν-bound
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
    // GATE READING (text-exhausted; the former gate rationale cited an
    // "X^c̄-labeled root" that was a misreading of a figure-embedded
    // fragment). Prop 5.2 verbatim (ar5iv L192-196):
    //   (a) blue/positive: E as in Fig 6(a) left "for some (D_jk⟨X⟩), (F_l⟨X⟩)
    //       and blue variable Y (POSSIBLY X ITSELF)" — the site root is ANY
    //       blue variable, NOT necessarily X;
    //   (b) red/positive: root = "blue variable Y ≠ X" (derived: h_X = X⁺
    //       would be non-modular, killing red-eliminability);
    //   (c)/(d): the negative-tree duals ("switching blue and red everywhere",
    //       L200) — UNIMPLEMENTED here.
    // Def 5.3 (L201): the site is "the subtree E of F rooted in r_X" ✓ (our
    // site choice), and the proof's F_l/G_k split matches collect_sum_branches
    // / walk_prod verbatim ✓. So the HARVEST is right; the GATE is wrong:
    // the current `rx.label == x` refutes (a)'s "possibly X itself".
    // PRECISION (retires the former "fail-closed = sound superset"
    // slogan, imprecise in BOTH directions). The engine's accepted set is
    // exactly two families —
    //   (i) blue at an X̄-headed r_X (= rule (a), Y = X; text ✓, goldens ✓),
    //   (ii) red at an X̄-headed r_X (NO positive-tree textual match — the
    //       candidate Fig-6(c)/(d) mechanism; sound ×3:
    //       red_side_fires_internal_x_labeled_root,
    //       cancellation_family_reduces_soundly_via_red_rule,
    //       red_side_quotient_reduces_to_y).
    // The paper's accepted set is strictly larger (rule (a): blue Y ≠ X;
    // rule (b): red at blue Y ≠ X) — t1 is pinned engine-refused /
    // paper-accepted (pinned by fig7b_t1_probe_coherent_but_engine_stuck).
    // The Y ≠ X refusals are
    // now PROVABLY forced, not merely figure-pending:
    //   • red Y ≠ X: keep-root+F_l+θ gives νX.((Y→X)→1) ≅ ∀Y.1 on t1 vs
    //     truth(t1) = ∅ (the Y := ∅ instance is uninhabitable) ⇒ unsound.
    //   • blue Y ≠ X: the ∀X.((X→A)→(X→B)→C) family (r_X = the C⁺-
    //     headed root; rule-(a) LHS match, K = ∅, F_l = both branches)
    //     DISCRIMINATES the candidate groupings: µ-lift (τ's current form) ⇒
    //     µX.((X→A)→(X→B)→Σ∅) ≅ 0; substitution (X⁺ := 0) ⇒ ≅ C = truth,
    //     but 0-leaves are OUTSIDE the Λ2 tree grammar; a K ≥ 1 reading
    //     of "as in Fig 6(a) left" would make the family LHS-non-matching
    //     (gate already text-exact). The figure's grouping decision is
    //     SET-LOAD-BEARING. (The former "conflict" probes dissolved on
    //     re-derivation: ∀X.(A⇒X)⇒(B⇒X) fires TODAY under family (i),
    //     sound via µX.(B→A) ≅ B→A; ∀X.(X→X)→Y is X̄-headed ⇒ family (ii),
    //     not a Y≠X case.)
    // The absorb shortcut below is the Y = X case and stays correct.
    let rx = find_binder_node(tree, x)?;
    let head_ok = match c {
        Color::Blue => rx.label == x && rx.color == Color::Blue,
        Color::Red => rx.label == x && rx.color == Color::Red,
    };
    if !head_ok {
        return None;
    }
    match c {
        // Fig. 6(a) — X BLUE-eliminable (Def 5.2: every X̄ node modular):
        // reduct θ(X) = {µX.}Σₖ∃Z⃗ₖ.Πⱼ D_jk[X⁺]; the F_l children of r_X
        // stay in parameter position, the head slot becomes θ.
        Color::Blue => {
            let (f_ctx, branches) = collect_sum_branches(tree, x)?;
            // Zero branches are text-legal: τ(•) = µX.∃Y⃗.Σ_{k∈K}Π_j τ(E_jk)
            // (paper L152) quantifies over an arbitrary family K — K=∅ gives
            // Σ_∅ = 0, and τ lowers PolySum{parts: []} to Never.  (This is
            // the Thm 5.6 un-stick path for X⁺-only shapes such as
            // ∀X.(X⇒Y)⇒X ↠ µX.((X⇒Y)⇒0) ≅ ∅.)
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
            splice_reduct(tree, x, &theta, f_ctx, true)
        }
        // Fig. 6(b) — X RED-eliminable (every X⁺ node modular):
        // reduct θ(X̄) = {νX.}∀Z⃗ₖ.Πⱼ E_k[X⁻]; the F_l children stay in
        // parameter position and r_X's OWN head survives (red-eliminable
        // implies h_X ≠ X⁺, so the head is not absorbed).
        Color::Red => {
            let mut f_ctx = Vec::new();
            let mut binders = Vec::new();
            let mut parts = Vec::new();
            for ch in &rx.children {
                if contains_x_col(ch, x, Color::Blue) {
                    // X⁺-bearing child: collect its ∀Z⃗ / E_j material.
                    walk_prod(ch, x, &mut binders, &mut parts, 0)?;
                } else {
                    // F_l (red side, paper L199 dual): no X⁺ terminals by
                    // construction (those took the walk_prod arm above).
                    // Paper permits F_l⟨X̄⟩ here — its X̄ residues are
                    // bound by the {νX.} scoping over the WHOLE reduct
                    // node (τ lifts the scope accordingly).
                    f_ctx.push(ch.clone());
                }
            }
            // Zero factors are text-legal (dual of the blue side): the
            // N-form Π_{j∈∅} E_j = 1 is the paper's unit constant, and τ
            // lowers PolyProd{parts: []} to Unit.
            // Symmetric with (Y⁺): always keep the {νX.} binder — the
            // canceling elision is βη-redundant (Fig. 4c) and the FV
            // occurrence test is not trusted (shared-set leak).
            let theta = PolyProd {
                head: x,
                binders,
                parts,
            };
            splice_reduct(tree, x, &theta, f_ctx, false)
        }
    }
}

/// Fig. 6 surgery: rebuild the root→r_X path, replacing r_X with the
/// reduct node — NOT a whole-subtree swap.  The former code dropped
/// r_X's F_l children (and, when several binders stacked on r_X, the
/// context B shape), destroying legitimate reducts such as
/// ∀X.(Int⇒X)⇒(Bool⇒Int)⇒X, whose correct reduct keeps (Bool⇒Int).
///
/// Blue side (`absorb_head`): the r_X head X̄ is absorbed by θ's {µX.} —
/// the reduct node's head label is x itself; with an empty F_l context
/// the reduct is θ alone (matching the ∀X.X→X ⇒ µX.1 golden).
/// Red side: r_X's own head label survives (never X⁺ under
/// red-eliminability), with the N-form θ as its last child.
fn splice_reduct(
    t: &PolyTree,
    x: usize,
    theta: &PolyTree,
    f_ctx: Vec<PolyTree>,
    absorb_head: bool,
) -> Option<PolyTree> {
    match t {
        Occ(n) => {
            if n.binders.contains(&x) {
                let mut binders = n.binders.clone();
                binders.retain(|&b| b != x);
                if absorb_head && f_ctx.is_empty() {
                    return Some(theta.clone());
                }
                let mut children = f_ctx;
                children.push(theta.clone());
                // Head label: the reduct node KEEPS r_X's own head label
                // when F_l is non-empty — overwriting it with x would
                // erase r_X's genuine head occurrence (e.g. the Z̄ head of
                // the two-continuation inner spine), shifting the
                // occurrence multiset and sending later 2-SAT rounds UNSAT.
                // (The empty-F_l case returned θ directly above, so no
                // label is needed there.)  F_l's X-residues stay bound by
                // θ's µ via τ's scope lift (the scope lift), which fires
                // on the residue condition regardless of the node label.
                let label = n.label;
                return Some(Occ(RNode {
                    label,
                    color: n.color,
                    binders,
                    children,
                }));
            }
            let mut n2 = n.clone();
            for ch in &mut n2.children {
                // Only the (unique, globally fresh) subtree containing
                // r_X changes; siblings are cloned verbatim.
                if find_binder_node(ch, x).is_some() {
                    *ch = splice_reduct(ch, x, theta, f_ctx.clone(), absorb_head)?;
                }
            }
            Some(Occ(n2))
        }
        PolySum { head, parts } => {
            if *head == x {
                // x is already bound by a previous reduction's • node —
                // never an eliminable candidate; reaching here means a
                // scope bug.  Fail-closed.
                return None;
            }
            let mut parts2 = Vec::new();
            for (bs, ts) in parts {
                let mut ts2 = Vec::new();
                for t2 in ts {
                    if find_binder_node(t2, x).is_some() {
                        ts2.push(splice_reduct(t2, x, theta, f_ctx.clone(), absorb_head)?);
                    } else {
                        ts2.push(t2.clone());
                    }
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
                return None; // same defensive fail-closed as above
            }
            let mut parts2 = Vec::new();
            for t2 in parts {
                if find_binder_node(t2, x).is_some() {
                    parts2.push(splice_reduct(t2, x, theta, f_ctx.clone(), absorb_head)?);
                } else {
                    parts2.push(t2.clone());
                }
            }
            Some(PolyProd {
                head: *head,
                binders: binders.clone(),
                parts: parts2,
            })
        }
    }
}

// Depth cap for the recursive PolyTree walkers below.  Source-
// derived trees are already bounded by the parser's
// `max_recursion_depth` (256), and Yoneda reduction only reshuffles
// subtrees (reduct depth ≤ original + O(1)); the cap is defense-
// in-depth for API-constructed trees — a depth past any legitimate
// one fails closed (None) instead of risking a stack overflow.
const WALK_DEPTH_CAP: usize = 100_000;

/// Prop 5.2 decomposition (paper L196-200): locate r_X (the Occ node
/// whose binder set contains x) and partition its children into
///
///  - F_l: children containing NO X̄ terminal — kept verbatim as the
///    reduct node's parameter context (paper L72: (Y⇒Z) survives in the
///    parameter position of the ≡_W reduct);
///  - G_k: children whose OWN head leaf is the modular X̄ — each
///    contributes one Σ-branch (∃Z⃗ = G.binders, D_jk = G's own
///    arguments).  A bare X̄-leaf child gives the branch (∅, []) — the
///    paper's "possibly I = Z_k = ∅, so that G_k = X̄".
///
/// The former path-prefix-sibling machinery polluted D_jk: the paper's
/// D_jk are only the site's own spine arguments; sibling branches and
/// F_l are NOT factors (each branch is an independent Σ-alternative and
/// F_l stays in parameter position).
///
/// Fail-closed: any X̄ NOT at a G-child's head (deeper inside a child,
/// inside G's arguments, inside Poly children) ⇒ None — after the 2-SAT
/// such positions cannot lie on a modular site; this is defense.
fn collect_sum_branches(
    t: &PolyTree,
    x: usize,
) -> Option<(Vec<PolyTree>, Vec<(Vec<usize>, Vec<PolyTree>)>)> {
    let rx = find_binder_node(t, x)?;
    let mut f_ctx = Vec::new();
    let mut branches = Vec::new();
    for c in &rx.children {
        match c {
            Occ(g) if g.label == x && g.color == Color::Red => {
                // G_k: the X̄ is this child's own head leaf.  Its
                // arguments D_jk must be X̄-free — an X̄ there would leak
                // a free variable into the reduct (schema violation).
                if g.children.iter().any(|d| contains_x_col(d, x, Color::Red)) {
                    return None;
                }
                branches.push((g.binders.clone(), g.children.clone()));
            }
            _ => {
                // Paper L197 (color-formatted): F_l = subtrees with no X̄
                // (RED) terminals; F_l⟨X⁺⟩ is LEGAL — its X⁺ residues are
                // bound by the {µX.} that scopes over the WHOLE reduct node
                // (µ over F_l included; τ lifts the scope accordingly).
                // The former any-color guard was over-tight and
                // blocked three paper-mandated reductions (Thm 5.6 forward
                // direction).
                if contains_x_col(c, x, Color::Red) {
                    return None;
                }
                f_ctx.push(c.clone());
            }
        }
    }
    Some((f_ctx, branches))
}

/// Locate the node whose binder set contains x (the r_X of Fig. 6).
/// Binder scopes are globally fresh, so at most one node matches.
fn find_binder_node(t: &PolyTree, x: usize) -> Option<&RNode> {
    match t {
        Occ(n) => {
            if n.binders.contains(&x) {
                return Some(n);
            }
            n.children.iter().find_map(|c| find_binder_node(c, x))
        }
        PolySum { parts, .. } => parts
            .iter()
            .flat_map(|(_, ts)| ts.iter())
            .find_map(|t2| find_binder_node(t2, x)),
        PolyProd { parts, .. } => parts.iter().find_map(|t2| find_binder_node(t2, x)),
    }
}

fn walk_prod(
    t: &PolyTree,
    x: usize,
    binders: &mut Vec<usize>,
    parts: &mut Vec<PolyTree>,
    depth: usize,
) -> Option<()> {
    if depth > WALK_DEPTH_CAP {
        return None; // fail-closed past the depth cap
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
            // Foreign-θ guard (defense-in-depth): a Poly node reached
            // here is an EARLIER round's θ — the x-material in its parts is
            // bound by THAT θ's constructor ({µx'.}/{νx'}, x' ≠ x; x itself
            // is ctor-bound and never a candidate again).  Descending would
            // lift x-sites out of the foreign ctor's scope: their factors
            // land under the CURRENT θ, stranding the foreign-bound
            // residues FREE (regression: red_harvest_refuses_foreign_theta_
            // descent builds the exact shape — fvs gains a free X).
            // Reachable today: NO — the modular window forces every site
            // inside a foreign θ to flat-d ≥ 2, non-terminals there are
            // non-modular (er false ⇒ the step never fires) and only bare
            // d2-Blue leaves pass, for which walk_prod emits a bare unit
            // (no material).  The guard turns that window-arithmetic
            // accident into a structural guarantee.  Skip when the θ
            // carries no x-material (opaque w.r.t. x); refuse otherwise.
            if contains_x_col(t, x, Color::Red) || contains_x_col(t, x, Color::Blue) {
                return None;
            }
            for (_, ts) in ps {
                for t2 in ts {
                    walk_prod(t2, x, binders, parts, depth + 1)?;
                }
            }
            Some(())
        }
        PolyProd { parts: ps, .. } => {
            // Foreign-θ guard — same rationale as the PolySum arm above.
            if contains_x_col(t, x, Color::Red) || contains_x_col(t, x, Color::Blue) {
                return None;
            }
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
                // Symmetric with the • node: {νX.} AND the ∀Z⃗ are
                // constructor binders of the polynomial constant — τ
                // re-wraps the ∀Z⃗, so they are never eliminable by →_X^c.
                if !binders.contains(head) {
                    binders.push(*head);
                }
                if !ctor.contains(head) {
                    ctor.push(*head);
                }
                for b in bs {
                    if !binders.contains(b) {
                        binders.push(*b);
                    }
                    if !ctor.contains(b) {
                        ctor.push(*b);
                    }
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
/// Step/depth bound (why the tree walkers below need no stack cap):
/// each successful `yoneda_step` removes x from r_X's binder set and x
/// reappears only as θ's CONSTRUCTOR head (`remaining_bound_vars`
/// excludes ctor-bound labels), and Notation 5.1 gives x a unique r_X —
/// so every bound variable is eliminated at most once and the loop runs
/// ≤ (#source binders) ≤ 256 steps (the parser's recursion bound).  Each
/// step adds at most one wrapping level, so tree depth stays ≤ ~512 —
/// the same magnitude the polar-action walkers are capped at.
/// `WALK_DEPTH_CAP` is a termination backstop only; if the
/// step-elimination invariant ever breaks (a step that keeps its
/// variable bound), re-derive this bound before trusting the walkers'
/// recursion.
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
    // The (variable, color) pairs already reduced — Def 5.6's standardness
    // index (each bound variable is eliminated at most once, so the
    // variable identifies its E₀ entry).
    let mut chosen: Vec<(usize, Color)> = Vec::new();
    for _step in 0..10_000 {
        // One walk per step — remaining_bound_vars is BOTH the
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
        // Def 5.6 (standard reduction): the sequence's colors must be
        // PAIRWISE coherent (Def 5.4) — NOT the whole tree coherent (Def
        // 5.5).  The former `flat.valuation()?` refused every incoherent
        // tree although the paper itself reduces non-coherent types
        // (post-Cor.-5.2, L380–382: "such a type A need not even be
        // coherent … reduces, by eliminating X …").  Track the reduced
        // (variable, color) pairs; a coherent valuation, when one exists,
        // is still preferred (the Thm 5.6 strategy) and automatically
        // satisfies the pairwise condition.  Measuring coherence on the
        // current view is sound: Lemma 5.5 shows parallel pairs (hence
        // incoherence) only persist, never appear, along standard steps.
        let phi = flat.valuation();
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
            // Def 5.2 license: c-eliminable.  Def 5.6 license: coherent
            // with every previously reduced variable (at-most-once
            // elimination ⇒ the variable identifies the E₀ entry).
            let standard = |c: Color| {
                chosen
                    .iter()
                    .all(|&(y, d)| y == x || flat.pair_coherent(x, c, y, d))
            };
            let mut colors: Vec<Color> = Vec::new();
            if let Some(&c) = phi.as_ref().and_then(|v| v.get(&x)) {
                // φ-color first (Thm 5.6 standard strategy).
                colors.push(c);
            }
            for c in [Color::Blue, Color::Red] {
                if !colors.contains(&c) {
                    // Dual-color fallback: the valuation may pick a color
                    // whose sites were absorbed by an earlier step, while
                    // the opposite color's sites remain.  A dual step is
                    // admissible only under the SAME Def 5.6 pairwise
                    // coherence — a non-standard step is not licensed by
                    // Lemma 5.3.  Fail-closed is preserved: yoneda_step
                    // itself refuses any tree whose shape does not match
                    // the schema.
                    colors.push(c);
                }
            }
            for &c in &colors {
                if !flat.color_eliminable(x, c) || !standard(c) {
                    continue;
                }
                if let Some(next) = yoneda_step(&cur, x, c) {
                    cur = next;
                    chosen.push((x, c));
                    progressed = true;
                    break;
                }
            }
            if progressed {
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
                // This must be `entry().or_insert` — the previous
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
                // `entry().or_insert` (see the PolySum note — the
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
    /// Invert the lazy atom encoding (coherence.rs `atom_label`):
    /// label = ATOM_BASE - arena_index, so the index — and via
    /// `TypeTag::from(&data)` the full TypeId — is recovered in O(1)
    /// (same slot, same tag ⇒ the recovered TypeId IS the original).
    /// Returns None out-of-range (fail-closed).
    fn atom_tyid(&self, label: usize) -> Option<TypeId> {
        let idx = super::coherence::atom_label_inverse(label)?;
        let data = self.interner.types.get(idx)?;
        debug_assert!(
            !matches!(**data, TypeData::GenericParam { .. }),
            "atom labels never carry params (is_ground invariant)"
        );
        let tag = TypeTag::from(&**data) as usize;
        Some(TypeId::from_raw(((idx + 1) << TypeId::TAG_BITS) | tag))
    }

    /// The ∃Y⃗ₖ.Σₖ Πⱼ E_jk body of a • node WITHOUT the {µX.} wrapper —
    /// used both by `tau`'s PolySum arm (which adds the µ itself) and by
    /// the Occ reduct arm's scope lift (where the µ moves OUTSIDE the
    /// whole reduct node to bind F_l⟨X⁺⟩ residues; paper L152/L197).
    fn tau_sigma_body(&mut self, parts: &[(Vec<usize>, Vec<PolyTree>)]) -> Option<TypeId> {
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
        Some(self.coproduct(alternatives))
    }

    /// The ∀Y⃗.Πⱼ E_j body of an N node WITHOUT the {νX.} wrapper (dual of
    /// `tau_sigma_body`; paper L152/L199).
    fn tau_prod_body(&mut self, binders: &[usize], parts: &[PolyTree]) -> Option<TypeId> {
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
        Some(prod)
    }

    /// Lower a simple polynomial tree to a `TypeId` (Fig. 5).
    /// Fail-closed: non-simple trees or out-of-Λ2 leaves ⇒ None.
    pub fn tau(&mut self, t: &PolyTree) -> Option<TypeId> {
        match t {
            PolySum { head, parts } => {
                let sigma = self.tau_sigma_body(parts)?;
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
                let prod = self.tau_prod_body(binders, parts)?;
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
                        // An inert atom terminal: recover the original
                        // TypeId via the O(1) reverse map; fail-closed
                        // out-of-range rather than minting a bogus
                        // GenericParam with a huge index.
                        return self.atom_tyid(n.label);
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
                    // Fig. 5: τ(⊕^X E | Y⃗) = ∀Y⃗.… — node binders must be
                    // re-wrapped as ∀ (they are the shared-LCA variables
                    // of the reduct node; dropping them let a bound
                    // variable escape as a free GenericParam).
                    let mut leaf =
                        self.generic_param(n.label, Symbol::intern(&format!("T{}", n.label)));
                    for &b in n.binders.iter().rev() {
                        leaf = self.forall(b, Symbol::intern(&format!("Y{}", b)), leaf);
                    }
                    return Some(leaf);
                } else {
                    // Negative spine: Π τ(children) → label.
                    let atom_ret =
                        if n.label >= super::coherence::ATOM_LABEL_FLOOR && n.label != usize::MAX {
                            // Atom-headed spine: recover the atom TypeId via the
                            // reverse map (None ⇒ fail-closed).
                            Some(self.atom_tyid(n.label))
                        } else {
                            None
                        };
                    if let Some(None) = atom_ret {
                        return None;
                    }
                    //
                    // Head-slot θ with SCOPE LIFT (Fig. 6 RHS after a
                    // splice; paper L152/L197/L199): when the LAST child is
                    // a Poly node whose head (a) equals this node's label
                    // (blue absorb) or (b) occurs in an EARLIER child (an
                    // F_l⟨X⁺⟩/F_l⟨X̄⟩ residue — legal per Prop 5.2's proof),
                    // the {µX.}/{νX.} must scope over the WHOLE reduct node
                    // — F_l included — not just θ's own body.  Lowering:
                    //   Mu{h, Fn(τ(F…), σ-body)}   (blue, σ-body without µ)
                    //   Nu{h, Fn(τ(F…), π-body)}   (red,  π-body without ν)
                    // The former τ(θ)-as-return left the residues free and
                    // the θ's inner µ could not reach them.
                    let lifted_head = match n.children.last() {
                        Some(PolySum { head, .. }) | Some(PolyProd { head, .. }) => {
                            let h = *head;
                            let residues = n.children[..n.children.len() - 1].iter().any(|f| {
                                contains_x_col(f, h, Color::Red)
                                    || contains_x_col(f, h, Color::Blue)
                            });
                            if h == n.label || residues {
                                Some(h)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    // Parameters: ALL children EXCEPT the trailing Poly θ
                    // when a scope lift fires.  The former unconditional
                    // take(len-1) dropped the last REAL parameter of any
                    // spine without a Poly tail (e.g. F₁ = (X→Int) inside a
                    // lifted reduct) — masked until atom τ was restored
                    // (see `atom_tyid`).
                    let param_count = if lifted_head.is_some() {
                        n.children.len() - 1
                    } else {
                        n.children.len()
                    };
                    let mut params: SmallVec<[TypeId; 4]> = SmallVec::new();
                    for c in n.children.iter().take(param_count) {
                        params.push(self.tau(c)?);
                    }
                    if let Some(h) = lifted_head {
                        // Scope lift: µ/ν wraps the WHOLE reduct node, and
                        // the node's own binders (shared-LCA variables, x
                        // already removed by the splice) wrap OUTSIDE the
                        // µ/ν — the binder sat upstream of the head before
                        // the surgery (Fig. 5: ∀Y⃗ outside).
                        let name = Symbol::intern(&format!("X{}", h));
                        let lifted = match n.children.last() {
                            Some(PolySum { parts, .. }) => {
                                let sigma = self.tau_sigma_body(parts)?;
                                let body = self.function(params, sigma);
                                self.alloc(TypeData::Mu {
                                    param_index: h,
                                    param_name: name,
                                    body,
                                })
                            }
                            Some(PolyProd {
                                head,
                                binders,
                                parts,
                            }) => {
                                let prod = self.tau_prod_body(binders, parts)?;
                                let body = self.function(params, prod);
                                self.alloc(TypeData::Nu {
                                    param_index: *head,
                                    param_name: name,
                                    body,
                                })
                            }
                            _ => unreachable!("lifted_head checked a Poly last child"),
                        };
                        let mut ty = lifted;
                        for &b in n.binders.iter().rev() {
                            ty = self.forall(b, Symbol::intern(&format!("Y{}", b)), ty);
                        }
                        return Some(ty);
                    }
                    let ret = match atom_ret {
                        // Atom head recovered via the reverse map.
                        Some(Some(t)) => t,
                        // Atom label recognized but the reverse map failed:
                        // fail closed, mirroring the leaf arm — do NOT mint
                        // a GenericParam with an atom-label-sized index.
                        Some(None) => return None,
                        None => {
                            self.generic_param(n.label, Symbol::intern(&format!("T{}", n.label)))
                        }
                    };
                    // Fig. 5: re-wrap the node binders as ∀Y⃗ (see the leaf
                    // arm note — shared-LCA variables must not escape).
                    let mut ty = self.function(params, ret);
                    for &b in n.binders.iter().rev() {
                        ty = self.forall(b, Symbol::intern(&format!("Y{}", b)), ty);
                    }
                    Some(ty)
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

    /// ∀X.X — X has NO occurrence at all, so the
    /// Fig. 6(a) LHS matches trivially (L=∅, K=∅) and
    ///   θ = µX.Σ_∅ = 0   (paper L152: family quantifier over K).
    /// Reduct: µX.0 ≅ Never — the paper's zero type, text-legal.
    #[test]
    fn reduce_no_occurrence_to_never() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let ty = forall(&mut ctx, 0, x);
        let simple = reduce(&ctx, ty).expect("K=∅ blue rule must fire");
        assert!(is_simple(&simple));
        match &simple {
            PolySum { head: 0, parts } => assert!(
                parts.is_empty(),
                "∀X.X must reduce to µX.Σ_∅ (zero branches)"
            ),
            other => panic!("expected µX.Σ_∅, got {:?}", other),
        }
    }

    /// ∀X.(X→Int)→X — X occurs only as h_X and as
    /// the X⁺ parameter of (X→Int)⟨X⁺⟩; L=∅, K=∅ ⇒
    ///   ∀X.(X→Int)→X ↠ µX.((X→Int)⇒0) ≅ ∅   (X:=0 ⇒ 1⇒0 = 0).
    /// (The τ endpoint stays fail-closed on the Int atom — pending the
    /// label→TypeId reverse map; this pins the PolyTree reduct.)
    #[test]
    fn reduce_yoneda_atom_shape_to_never() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let i = ctx.int(32, true);
        let xi = ctx.function(smallvec![x], i); // (X→Int)
        let body = ctx.function(smallvec![xi], x); // (X→Int)→X
        let ty = forall(&mut ctx, 0, body);
        let simple = reduce(&ctx, ty).expect("K=∅ blue rule must fire");
        assert!(is_simple(&simple));
        // Reduct shape: the spliced reduct node
        //   Occ{label: X, children: [F₁=(X→Int)⟨X⁺⟩, PolySum{head: X, ∅}]}
        // ≙ µX.((X→Int)⇒Σ_∅) — µ scope lifted over F₁, Σ_∅ = 0
        // (K=∅), F₁'s X⁺ residue bound by the lifted µ.
        match &simple {
            Occ(n) if n.label == 0 => match n.children.last() {
                Some(PolySum { head: 0, parts }) => assert!(
                    parts.is_empty(),
                    "∀X.(X→Int)→X must end in Σ_∅, got parts={:?}",
                    parts.len()
                ),
                other => panic!("expected a trailing PolySum head=X, got {:?}", other),
            },
            other => panic!("expected the lifted reduct node, got {:?}", other),
        }
    }

    #[test]
    fn tau_lowers_atom_via_reverse_map() {
        // τ of the ∀X.(X→Int)→X reduct — the Int atom lowers via the
        // reverse map, the F_l⟨X⁺⟩ residue is µ-bound (scope lift),
        // Σ_∅ = Never. Expected ≙ µX.((Int⇒X)⇒0) — the paper's ≡_Y
        // form (NOT a Set-domain type: F(0)=1, F(1)=0 — µ-non-monotone,
        // so no Set fixpoint exists; the tree form is still faithful).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let i = ctx.int(32, true);
        let xi = ctx.function(smallvec![x], i);
        let body = ctx.function(smallvec![xi], x);
        let ty = forall(&mut ctx, 0, body);
        let ct = ctx.build_coherence_tree(ty).expect("Λ2");
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly).expect("converges");
        let m = ctx
            .tau(&simple)
            .expect("Int atom lowers via the reverse map");
        match ctx.get(m) {
            TypeData::Mu {
                param_index: 0,
                body,
                ..
            } => match ctx.get(*body) {
                TypeData::Fn { params, ret } => {
                    assert_eq!(params.len(), 1);
                    match ctx.get(params[0]) {
                        TypeData::Fn {
                            params: ip,
                            ret: ir,
                        } => {
                            assert_eq!(ip.len(), 1);
                            // The X⁺ leaf lowers to a (freshly minted)
                            // GenericParam for index 0 — τ mints per call.
                            assert!(ctx.is_generic_param(ip[0]), "µ-bound residue");
                            // The Int atom sits in the RETURN position of
                            // F₁=(X→Int) and must be recovered via the
                            // reverse map.
                            assert_eq!(*ir, i, "Int atom via the reverse map");
                        }
                        other => panic!("F₁ shape: {other:?}"),
                    }
                    assert!(ctx.is_never(*ret), "Σ_∅ = 0");
                }
                other => panic!("µ body: {other:?}"),
            },
            other => panic!("expected µX.((Int⇒X)⇒0), got {other:?}"),
        }
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

    /// Paper Example 3.1 (arXiv:1907.03481v3 L70-72, verbatim):
    /// A = ((((X⇒Y)⇒X⇒Z)⇒(Y⇒Z)⇒W)⇒(Y⇒Z)⇒W — the second parameter is
    /// (Y⇒Z), NOT (Y⇒W).  The ≡_W step of §3,
    ///   ∀XYZW.A ≡_W ∀XYZ.(Y⇒Z)⇒(((X⇒Y)⇒X⇒Z)×(Y⇒Z)),
    /// keeps the W-free (Y⇒Z) as an F_l context — it only holds because
    /// the second parameter's W is free.
    ///
    /// §5 tree-level verdict: X̄(inner arg-2, d=4), X⁺(d=5), Ȳ(xy head
    /// d=5, yz-in-mid arg d=3), Y⁺(yz-arg arg d=3), Z̄(yz-in-mid head
    /// d=4), Z⁺(inner head d=4) all fall outside the modular window ⇒
    /// X,Y,Z are eliminable in NEITHER color ⇒ 2-SAT UNSAT ⇒ κ=∞.  This
    /// does NOT contradict the paper: §3's ≅_Y 1 chain uses the FULL ≡
    /// isomorphism theory of App. A, strictly stronger than the §5 tree
    /// rewriting; §5 only gives sufficient conditions (L183), and the
    /// discussion after Cor. 5.2 (L369-372) exhibits a ≡_Y-finite type
    /// that is tree-incoherent.  Fail-closed IS the paper-faithful
    /// behavior here.
    #[test]
    fn golden_example_3_1_tree_is_incoherent() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());

        // inner = (X→Y)→X→Z
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let inner = ctx.function(smallvec![xy], xz);
        let yz = ctx.function(smallvec![y], z);
        // mid = inner → (Y→Z) → W
        let mid = ctx.function(smallvec![inner, yz], w);
        // A = mid → (Y→Z) → W   (paper L71: second parameter is yz)
        let a = ctx.function(smallvec![mid, yz], w);

        let ty = forall(&mut ctx, 0, a);
        let ty = forall(&mut ctx, 1, ty);
        let ty = forall(&mut ctx, 2, ty);
        let ty = forall(&mut ctx, 3, ty);

        let ct = ctx.build_coherence_tree(ty).expect("Λ2 golden type");
        assert_eq!(
            ct.flat().characteristic(),
            u8::MAX,
            "X,Y,Z eliminable in neither color ⇒ κ=∞"
        );
        // Incoherent ⇒ no standard reduction, no witness.
        assert!(
            reduce(&ctx, ty).is_none(),
            "incoherent ⇒ standard reduction refused"
        );
    }

    /// The (Y⇒W) variant of Example 3.1 — the type the old golden
    /// accidentally encoded.  It is a legitimate coherent (κ=1) type with
    /// two continuation branches; pinned at the tree level only.  Its
    /// correct red-side reduct depends on the Fig. 6(b) dual surgery, so
    /// the reduct shape is deliberately NOT pinned here yet.
    #[test]
    fn golden_two_continuation_branches_coherent() {
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

        let ty = forall(&mut ctx, 0, a);
        let ty = forall(&mut ctx, 1, ty);
        let ty = forall(&mut ctx, 2, ty);
        let ty = forall(&mut ctx, 3, ty);

        let ct = ctx.build_coherence_tree(ty).expect("Λ2 type");
        let ft = ct.flat();
        // Machine-checked derivation chain (proof in the comment below;
        // each premise pinned so a future lowering change names the DRIFTED
        // LEMMA, not a bare "κ changed"):
        assert!(
            !ft.color_eliminable(3, Color::Red),
            "L1 drifted: er(W)=false required (W⁺ = {{h_W}}, a==r ⇒ non-modular)"
        );
        assert!(
            !ft.color_eliminable(1, Color::Blue),
            "L2 drifted: eb(Y)=false required (yz-branch Ȳ head at d=3, out of window)"
        );
        // L3: with L1∧L2 forced, the rule-(ii) clause (a_W^red ∨ a_Y^blue)
        // is the sole UNSAT source — pinned by the verdict itself.
        assert_eq!(
            ft.characteristic(),
            u8::MAX,
            "rule-(ii) clause (a_W^red ∨ a_Y^blue) + forced φ(W)=Blue ∧ φ(Y)=Red ⇒ UNSAT"
        );
    }

    /// Red-side firing smoke + τ end-to-end (the first pinned red-side
    /// firing): φ(X)=Red forced —
    /// eb(X)=false (the S=(X⇒Unit) branch's bare X̄ leaf sits at d2 RED,
    /// outside the color-asymmetric window) and er(X)=true (the d1 X⁺
    /// bare leaf) — with r_X = the X̄-labeled INNER spine, so the head
    /// gate passes and the RED rule fires (the first pinned red-side
    /// reduction).  Reduct ≙ νX.(S → Π₁): S enters as F_l⟨X̄⟩ (paper
    /// L199, residues bound by the lifted ν) and the bare X⁺ site
    /// gives the single Π₁ factor.  τ (via the atom reverse map for
    /// the Unit atom head) lowers end-to-end.
    #[test]
    fn red_side_fires_internal_x_labeled_root() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let u = ctx.unit();
        let s = ctx.function(smallvec![x], u); // (X⇒Unit)
        let inner = ctx.function(smallvec![s, x], x); // (X⇒Unit)⇒X
        let a = ctx.function(smallvec![inner], u); // inner ⇒ Unit
        let ty = forall(&mut ctx, 0, a);

        let ct = ctx.build_coherence_tree(ty).expect("Λ2");
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly)
            .expect("φ(X)=Red forced, X̄-labeled internal root ⇒ red rule fires");
        assert!(is_simple(&simple));
        // Reduct shape (per flat dump): the ROOT is the Unit-atom spine
        // above r_X (A's return is the Unit atom); the spliced X̄-headed
        // reduct node is its single child — keeping its X̄ head, with the
        // S spine (F_l⟨X̄⟩ residue) and the N-form θ = νX.Π₁ (single unit
        // factor from the bare X⁺ site) as children.
        match &simple {
            Occ(root) => {
                assert_eq!(root.children.len(), 1, "root spine: one child (inner)");
                match root.children.last().expect("checked len") {
                    Occ(n) if n.label == 0 && n.color == Color::Red => {
                        match n.children.last() {
                            Some(PolyProd {
                                head: 0,
                                binders,
                                parts,
                            }) => {
                                assert!(binders.is_empty(), "no ∀Z⃗ for a bare site");
                                assert_eq!(parts.len(), 1, "Π₁ from the bare X⁺ site");
                            }
                            other => panic!("expected trailing PolyProd head=X, got {other:?}"),
                        }
                        assert_eq!(n.children.len(), 2, "S (F_l⟨X̄⟩) + θ");
                    }
                    other => panic!("expected the spliced X̄-headed reduct node, got {other:?}"),
                }
            }
            other => panic!("expected an Occ root spine, got {other:?}"),
        }
        // τ endpoint (reverse map + scope lift): the Unit atom head lowers
        // via the reverse map; S's X̄ residue is ν-bound (a GenericParam
        // inside the ν scope).
        let m = ctx
            .tau(&simple)
            .expect("reverse map + scope lift lower the red reduct");
        match ctx.get(m) {
            TypeData::Fn { params, ret } => {
                assert_eq!(params.len(), 1, "the Unit-atom spine has one param (S)");
                match ctx.get(params[0]) {
                    TypeData::Nu {
                        param_index: 0,
                        body,
                        ..
                    } => match ctx.get(*body) {
                        TypeData::Fn {
                            params: p2,
                            ret: r2,
                        } => {
                            assert_eq!(p2.len(), 1, "S = (X̄⇒Unit) as F_l⟨X̄⟩");
                            match ctx.get(p2[0]) {
                                TypeData::Fn {
                                    params: p3,
                                    ret: r3,
                                } => {
                                    assert_eq!(p3.len(), 1);
                                    assert!(ctx.is_generic_param(p3[0]), "ν-bound X̄ residue");
                                    assert_eq!(*r3, u, "S's codomain is the Unit atom");
                                }
                                other => panic!("S shape: {other:?}"),
                            }
                            assert_eq!(*r2, u, "Π₁ ≙ Unit");
                        }
                        other => panic!("ν body: {other:?}"),
                    },
                    other => panic!("expected ν, got {other:?}"),
                }
                assert_eq!(*ret, u, "atom head via the reverse map");
            }
            other => panic!("expected the Unit-atom spine, got {other:?}"),
        }
    }

    /// Thm 5.6 probe on the paper's own Example 5.1 type (Fig 7(b) t1 —
    /// the d3 duel's t1). t1 is coherent with κ=0 (no down-moves: every
    /// X̄ occurrence is a NON-terminal spine, so no alternating cycle), so
    /// the paper's a-priori promise (κ≠∞ ⇒ rewrites to a monomorphic
    /// type) applies. Engine trace (full, three iterations): X has NO head
    /// occurrence — its occurrences are two X̄ branch-head spines and two
    /// X⁺ d2 leaves, while r_X is the root headed Y⁺ — so the head gate
    /// can never pass for X in EITHER color ⇒ X is never eliminable ⇒
    /// standard_reduce = None. A LIVE Thm 5.6 violation on the paper's own
    /// example: Fig. 9(a)'s figure ruling is LOAD-BEARING here. (If this
    /// returns Some, the gate semantics moved — re-derive the figure
    /// reading and Example 5.1's trace against the new semantics.)
    #[test]
    fn fig7b_t1_probe_coherent_but_engine_stuck() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let xx = ctx.function(smallvec![x], x);
        let xz = ctx.function(smallvec![x], z);
        let forall_z = ctx.forall(2, "Z".into(), xz);
        let yx = ctx.function(smallvec![y], x);
        let body = ctx.function(smallvec![xx, forall_z, yx], y);
        let ty = ctx.forall(0, "X".into(), body);
        let ty = ctx.forall(1, "Y".into(), ty);
        let ct = ctx.build_coherence_tree(ty).expect("Λ2 t1");
        // κ=1 (was 0 under the bare-only down-move target): Fig. 9(a)
        // illustrates an alternate path in t1 itself — the X⁺ d2 leaf
        // down-moves to the X̄ d1 BRANCH HEAD (modular per Def 5.7's
        // "β is modular", the same predicate Def 5.2 quantifies over) and
        // up-moves back ⇒ cyclic.  Thm 5.6 violation record stands and
        // SHARPENS: a κ=1 type must strongly converge per Thm 5.6, yet the
        // engine still head-gates (standard_reduce = None) — the violation
        // moved from "κ=0 yet stuck" to "κ=1 yet stuck".
        assert_eq!(ct.flat().characteristic(), 1, "coherent, cyclic (κ=1)");
        let poly = from_occ(ct.root_tree());
        assert!(
            standard_reduce(&poly).is_none(),
            "predicted: X head-gated forever (no X-colored head occurrence) ⇒ stuck"
        );
    }

    /// Cancellation-family probe — ADJUDICATED (probe-refuted then
    /// re-derived from the dump): ∀X.((A⇒X)⇒X)⇒C (A, C free).  The former
    /// prediction ("coherent but refused") was WRONG on both counts:
    ///   • place_binders sinks ∀X onto INNER, not the root — inner's own
    ///     X̄ label IS an X occurrence, so LCA({inner, ax}) = inner (the
    ///     sink's `here` short-circuit), and the current red gate
    ///     (X̄-labeled root — the red_side_fires_internal_x_labeled_root
    ///     family) legitimately fires;
    ///   • the fired reduct is SOUND: the root C-spine survives, and the
    ///     inner subtree ≙ νX.Π[A] ≅ A — matching the parametric truth
    ///     ∀X.((A→X)→X) ≅ A (X:=A, feed id) ⇒ the whole type ≅ A→C.  (The
    ///     former "truth ≅ C" claim used only the instantiation direction
    ///     X:=A→0 — sound but incomplete; it cannot establish ≅.)
    /// Textual note: this firing family (X̄-labeled r_X, red rule —
    /// red_side_fires_internal_x_labeled_root and this probe, both sound)
    /// has NO positive-tree rule-(b) textual LHS match under the verbatim
    /// Prop 5.2 reading ("blue variable Y ≠ X"); it is the candidate
    /// Fig-6(c)/(d) mechanism.  Parked with the same figure dependency as
    /// the Y ≠ X RHS question — but the current red gate+RHS now stands
    /// UNREFUTED on two sound probes.
    #[test]
    fn cancellation_family_reduces_soundly_via_red_rule() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let a = ctx.generic_param(1, "A".into());
        let c = ctx.generic_param(2, "C".into());
        let ax = ctx.function(smallvec![a], x); // (A⇒X)
        let inner = ctx.function(smallvec![ax], x); // (A⇒X)⇒X
        let body = ctx.function(smallvec![inner], c); // ((A⇒X)⇒X)⇒C
        let ty = ctx.forall(0, "X".into(), body);
        let ct = ctx.build_coherence_tree(ty).expect("Λ2");
        assert_ne!(ct.flat().characteristic(), u8::MAX, "coherent (κ=0)");
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly).expect("red rule fires at the X̄-labeled r_X");
        assert!(is_simple(&simple));
        // Reduct shape (per dump): the C-headed root spine SURVIVES; the
        // spliced inner node keeps its X̄ label; its sole child is the
        // N-form θ = PolyProd{head: X, parts: [A]} (the A residue).
        match &simple {
            Occ(root) if root.label == 2 && root.color == Color::Blue => {
                assert_eq!(root.children.len(), 1);
                match root.children.last().expect("checked len") {
                    Occ(n) if n.label == 0 && n.color == Color::Red => {
                        assert!(n.binders.is_empty());
                        match n.children.last() {
                            Some(PolyProd {
                                head: 0,
                                binders,
                                parts,
                            }) => {
                                assert!(binders.is_empty());
                                assert_eq!(parts.len(), 1, "single A residue factor");
                                match &parts[0] {
                                    Occ(leaf) => assert_eq!(leaf.label, 1, "the A leaf"),
                                    other => panic!("expected the A leaf, got {other:?}"),
                                }
                            }
                            other => panic!("expected θ = PolyProd head=X, got {other:?}"),
                        }
                    }
                    other => panic!("expected the X̄-headed red reduct node, got {other:?}"),
                }
            }
            other => panic!("expected the surviving C-headed root spine, got {other:?}"),
        }
        // τ end-to-end: m ≙ (νX.A)⇒C ≅ A⇒C — the parametric truth.
        let m = ctx.tau(&simple).expect("τ lowers the red reduct");
        match ctx.get(m) {
            TypeData::Fn { params, ret } => {
                assert_eq!(params.len(), 1, "the C-spine has one param (the ν image)");
                assert!(ctx.is_generic_param(*ret), "C survives as the codomain");
                match ctx.get(params[0]) {
                    TypeData::Nu {
                        param_index: 0,
                        body,
                        ..
                    } => match ctx.get(*body) {
                        TypeData::Fn {
                            params: p2,
                            ret: r2,
                        } => {
                            assert!(p2.is_empty(), "the ν body is the A image (no params)");
                            assert!(ctx.is_generic_param(*r2), "the A residue");
                        }
                        other => panic!("ν body: {other:?}"),
                    },
                    other => panic!("expected νX wrapping the A image, got {other:?}"),
                }
            }
            other => panic!("expected the C-spine, got {other:?}"),
        }
    }

    /// Regression pin for the foreign-θ guard: a Ȳ-headed red site whose
    /// child is an EARLIER round's θ (µX.Σ[Y⁺spine⟨X⁺⟩] — X ctor-bound by
    /// the FOREIGN µ).  Descending would lift the Y⁺ site out of the
    /// foreign ctor's scope and strand the X⁺ leaf as a free factor under
    /// the CURRENT θ (fvs() gains a free 0).  The guard refuses instead;
    /// pinned by calling `yoneda_step` directly (the modular window makes
    /// this shape unreachable from `standard_reduce` today — the guard
    /// turns that window-arithmetic accident into a structural one).
    #[test]
    fn red_harvest_refuses_foreign_theta_descent() {
        use super::super::coherence::Color as C2;
        let x_leaf = Occ(RNode {
            label: 0,
            color: C2::Blue,
            binders: Vec::new(),
            children: Vec::new(),
        });
        let y_spine = Occ(RNode {
            label: 1,
            color: C2::Blue,
            binders: Vec::new(),
            children: vec![x_leaf],
        });
        let foreign_theta = PolySum {
            head: 0,
            parts: vec![(Vec::new(), vec![y_spine])],
        };
        let site = Occ(RNode {
            label: 1, // Ȳ-headed site — the red gate passes (binders: [1])
            color: C2::Red,
            binders: vec![1],
            children: vec![foreign_theta],
        });
        let step = yoneda_step(&site, 1, C2::Red);
        assert!(
            step.is_none(),
            "foreign-θ guard must refuse descent into an earlier round's θ"
        );
    }

    /// F2's third sound probe, end-to-end: ∀X.(X→X)→Y — the first
    /// probe covering the "in-branch r_X + empty F_l" shape.  Premises
    /// (machine-checked): eb(X)=false (h_X IS r_X, a==r ⇒ non-modular),
    /// er(X)=true (the bare X⁺ leaf at d1) ⇒ φ(X)=Red forced.  Red rule
    /// fires at the X̄-headed r_X; θ = νX.Π₁; reduct ≙ (νX.1)→Y ≅ 1→Y ≅ Y
    /// — the parametric truth (∀X.((X→X)→Y) ≅ (∀X.X→X)→Y, feed the
    /// witness to id).  νX.Π₁ is exactly the ∀X.X→X ≅ 1 quotient — F2's
    /// ν-quotient semantics verified a third time
    /// (red_side_fires_internal_x_labeled_root,
    /// cancellation_family_reduces_soundly_via_red_rule, this).
    #[test]
    fn red_side_quotient_reduces_to_y() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let xx = ctx.function(smallvec![x], x); // (X→X)
        let body = ctx.function(smallvec![xx], y); // (X→X)→Y
        let ty = ctx.forall(0, "X".into(), body);
        let ct = ctx.build_coherence_tree(ty).expect("Λ2");
        let ft = ct.flat();
        // Premises: φ(X)=Red forced.
        assert!(
            !ft.color_eliminable(0, Color::Blue),
            "eb(X)=false required (h_X is r_X itself, a==r ⇒ non-modular)"
        );
        assert!(
            ft.color_eliminable(0, Color::Red),
            "er(X)=true required (bare X⁺ leaf at d1, modular)"
        );
        assert_ne!(ft.characteristic(), u8::MAX, "coherent");
        let poly = from_occ(ct.root_tree());
        let simple = standard_reduce(&poly).expect("red rule fires at the X̄-headed r_X");
        assert!(is_simple(&simple));
        // Reduct shape: the Y-headed root spine survives; the spliced node
        // keeps its X̄ label; its sole child is θ = νX.Π₁ (unit factor).
        match &simple {
            Occ(root) if root.label == 1 && root.color == Color::Blue => {
                assert_eq!(root.children.len(), 1);
                match root.children.last().expect("checked len") {
                    Occ(n) if n.label == 0 && n.color == Color::Red => {
                        assert!(n.binders.is_empty(), "x removed from the binder set");
                        match n.children.last() {
                            Some(PolyProd {
                                head: 0,
                                binders,
                                parts,
                            }) => {
                                assert!(binders.is_empty());
                                assert_eq!(parts.len(), 1, "Π₁ from the bare X⁺ site");
                            }
                            other => panic!("expected θ = PolyProd head=X, got {other:?}"),
                        }
                    }
                    other => panic!("expected the X̄-headed red reduct node, got {other:?}"),
                }
            }
            other => panic!("expected the surviving Y-headed root spine, got {other:?}"),
        }
        // τ end-to-end: m ≙ (νX.1)→Y ≅ Y — τ mints a fresh param name
        // ("T1"), so assert is_generic_param, not equality (same precedent
        // as tau_lowers_simple_tree).
        let m = ctx.tau(&simple).expect("τ lowers the quotient reduct");
        match ctx.get(m) {
            TypeData::Fn { params, ret } => {
                assert_eq!(params.len(), 1, "the Y-spine has one param (the ν image)");
                assert!(ctx.is_generic_param(*ret), "Y survives as the codomain");
                match ctx.get(params[0]) {
                    TypeData::Nu {
                        param_index: 0,
                        body,
                        ..
                    } => match ctx.get(*body) {
                        TypeData::Fn {
                            params: p2,
                            ret: r2,
                        } => {
                            assert!(p2.is_empty(), "the ν body is the Unit quotient");
                            assert_eq!(*r2, ctx.unit(), "Π₁ ≙ Unit (the ∀X.X→X ≅ 1 quotient)");
                        }
                        other => panic!("ν body: {other:?}"),
                    },
                    other => panic!("expected νX wrapping the Unit quotient, got {other:?}"),
                }
            }
            other => panic!("expected the Y-spine, got {other:?}"),
        }
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

// ─── Witness / coercion generation (App. A) ──────────────────────────────
//
// The paper (App. A) constructs explicit *term-level* witnesses for the
// two Yoneda schemas: s[x]/t[x] for (Y⁺) (built from in_Σ, pack, fold_T
// and their duals) and u[x]/v[x] for (Y⁻) (built from π_Π, out_U,
// unfold_U).  A full elaboration-level term generator needs an
// AST/HIR term constructor with fold/unfold/pack/unpack; this slice
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
    /// Certification semantics (doc/impl aligned): a direction is
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

    /// Paper Example 3.1 (arXiv:1907.03481v3 L70-72, verbatim — second
    /// parameter is (Y⇒Z), see golden_example_3_1_tree_is_incoherent).
    /// Tree-level verdict: κ=∞ ⇒ no Yoneda coercion, no round-trip
    /// witness — the App. A ≡ theory that proves ≅ 1 in §3 is strictly
    /// stronger than the §5 machinery, which fails closed here.
    #[test]
    fn witness_example_3_1_incoherent_no_witness() {
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let y = ctx.generic_param(1, "Y".into());
        let z = ctx.generic_param(2, "Z".into());
        let w = ctx.generic_param(3, "W".into());
        let xy = ctx.function(smallvec![x], y);
        let xz = ctx.function(smallvec![x], z);
        let inner = ctx.function(smallvec![xy], xz);
        let yz = ctx.function(smallvec![y], z);
        let mid = ctx.function(smallvec![inner, yz], w);
        let a = ctx.function(smallvec![mid, yz], w);
        let ty = ctx.forall(0, "X".into(), a);
        let ty = ctx.forall(1, "Y".into(), ty);
        let ty = ctx.forall(2, "Z".into(), ty);
        let ty = ctx.forall(3, "W".into(), ty);

        // Gate open (tree builds), κ=∞ ⇒ everything downstream refused.
        assert!(ctx.build_coherence_tree(ty).is_some(), "tree builds");
        assert!(ctx.yoneda_coercion(ty, YonedaDirection::Elim).is_none());
        assert!(ctx.yoneda_round_trip(ty).is_none());
    }

    /// The (Y⇒W) variant — coherent (κ=1, golden unchanged), but the head gate
    /// refuses the reduction: X's r_X is the Z-headed inner spine
    /// (label Z ≠ X) — not an X-labeled LHS instance.  The former
    /// "manually verified" 4-step trace checked CONVERGENCE only; the
    /// semantic soundness of that step family is refuted by the
    /// non-X-head counterexamples (∀X.(X⇒B)⇒(X⇒C)⇒D ↠ …≅ 0 vs ≅ D).
    /// The type joins the coherent-but-stuck family (see yoneda_step's comment).
    #[test]
    fn witness_two_continuation_branches_round_trip() {
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

        // κ=1 (tree-level golden unchanged) but the X step's site root
        // is Z-headed ⇒ no coercion, no round-trip witness.
        assert!(ctx.yoneda_round_trip(ty).is_none());
    }

    /// Probe (follow-up): pin WHERE the Example 3.1 witness pipeline
    /// runs — build → reduce → τ — so a regression names the stage.
    /// Paper type (second parameter (Y⇒Z), κ=∞): the gate stays open but
    /// standard reduction is refused, so stages 2–3 never fire.
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
        let mid = ctx.function(smallvec![inner, yz], w);
        let a = ctx.function(smallvec![mid, yz], w);
        let ty = ctx.forall(0, "X".into(), a);
        let ty = ctx.forall(1, "Y".into(), ty);
        let ty = ctx.forall(2, "Z".into(), ty);
        let ty = ctx.forall(3, "W".into(), ty);

        let ct = ctx.build_coherence_tree(ty).expect("stage 1: Λ2 tree");
        let poly = from_occ(ct.root_tree());
        assert!(
            standard_reduce(&poly).is_none(),
            "stage 2: incoherent (κ=∞) ⇒ standard reduction refused"
        );
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
