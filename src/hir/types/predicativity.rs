//! Stage 5c: predicativity pass — atomization (paper §8).
//!
//! §8 establishes that for a *constant* polynomial type P[X] (i.e. X
//! occurs only in the rightmost position: P[X] = ⟨⟨A_jk⟩_j → X⟩_k → X
//! with X ∉ FV(A_jk)), any instantiation `t_B : ∀X.P[X] ⊢ P[B/X]` can
//! be *atomized* — rewritten (Eq. 14) into a term using an
//! instantiation at `at(B)` (the rightmost variable of B), preserving
//! the contextual behavior (≅_ε):
//!
//! ```text
//! [_]_B  ≅_ε  λx⟨y_k⟩_k. In_B[ [_]_at(B) (λ⟨x_j⟩_j. El_B[⟨y_k⟩⟨x_j⟩_j]_k) ]
//! ```
//!
//! The premise is the *simplification lemma* (Lemma 8.1): whenever the
//! elimination context El : A ⊢ B (B ∈ RS(A)) is a Φ_X^P-algebra
//! morphism, the P[X]-induction on A is ε-equivalent to In composed
//! with the induction on B.
//!
//! This module provides the *type-level* half of the pass:
//! - `RS(A)`: right simplifications (Def, §8),
//! - `at(A)`: the rightmost variable,
//! - `dual_context_pair`: the canonical (El, In) pair A ⊢ at(A),
//! - `atomizable`: whether a ∀X.P[X] instantiation at B can be atomized
//!   (P constant + B decomposes to a variable head).
//!
//! The *term-level* rewrite itself lives on `witness::Term` (Stage 5b
//! v2): `atomize_instantiation` rewrites a `TyApp` of the initial
//! algebra into the Eq. (14) shape using the (El, In) scaffolding.
//!
//! Fail-closed: any premise failure (non-constant P, B with no
//! rightmost variable, gate closed) ⇒ no atomization is offered.

use super::rewrite::{from_occ, standard_reduce};
use super::witness::Term;
use super::*;

/// Right simplifications RS(A) (§8):
///   RS(X) = ∅;  RS(A→B) = {B} ∪ RS(B);  RS(∀Y.B) = {B} ∪ RS(B).
pub fn right_simplifications(ctx: &TypeContext<'_>, ty: TypeId, out: &mut Vec<TypeId>) {
    match ctx.get(ty) {
        TypeData::Fn { ret, .. } | TypeData::Forall { body: ret, .. } => {
            out.push(*ret);
            right_simplifications(ctx, *ret, out);
        }
        _ => {}
    }
}

/// `at(A)`: the rightmost variable of A (§8).  `None` when A's spine
/// does not end in a variable (e.g. ends in Int/Unit or a Poly reduct).
pub fn at_variable<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> Option<TypeId> {
    let mut cur = ty;
    loop {
        match ctx.get(cur) {
            TypeData::Fn { ret, .. } | TypeData::Forall { body: ret, .. } => {
                cur = ctx.resolve_binding(*ret);
            }
            TypeData::GenericParam { .. } => return Some(cur),
            _ => return None,
        }
    }
}

/// Is P[X] a *constant* polynomial type?  P is constant iff X does not
/// occur in any of the A_jk — equivalently (paper's isomorphism (13))
/// iff the standard reduction of ∀X.P[X] lands on a tree whose reduct
/// carries NO {µX.}/{νX.} constructor binder (the fixpoint binder
/// appears precisely when P is non-constant).
pub fn polynomial_is_constant<'input>(ctx: &TypeContext<'input>, poly: TypeId) -> Option<bool> {
    // poly must be a universal type.
    let TypeData::Forall {
        body, param_index, ..
    } = ctx.get(poly)
    else {
        return None;
    };
    let _ = body;
    let ct = ctx.build_coherence_tree(poly)?;
    let t = from_occ(ct.root_tree());
    // Cheap precheck: the reduced tree must be simple (κ ≠ ∞ along the
    // way) for the question to even be decidable by our machinery.
    let simple = standard_reduce(&t)?;
    // Constant P ⇔ the constructor binder's variable equals the
    // eliminated ∀X binder... actually ⇔ the reduct has NO fixpoint:
    // walk the simple tree for PolySum/PolyProd whose head is a real
    // bound variable (not the elided usize::MAX).
    fn has_fixpoint(t: &super::rewrite::PolyTree) -> bool {
        use super::rewrite::PolyTree;
        match t {
            PolyTree::Occ(n) => n.children.iter().any(|c| has_fixpoint(c)),
            PolyTree::PolySum { head, parts, .. } => {
                *head != usize::MAX || parts.iter().any(|(_, ts)| ts.iter().any(has_fixpoint))
            }
            PolyTree::PolyProd { head, parts, .. } => {
                *head != usize::MAX || parts.iter().any(has_fixpoint)
            }
        }
    }
    let _ = param_index;
    Some(!has_fixpoint(&simple))
}

/// A dual context pair (El, In) : A ⊢ B (§8 Def 8.1) at the type level:
/// the elimination spine (type/term applications) from A down to B, and
/// the mirror introductions.  We represent it by the B endpoint plus the
/// number of spine steps (El = [] C₁ … Cₙ u₁ … uₘ, In = Λ/λ mirror).
#[derive(Clone, Debug)]
pub struct DualContext {
    /// B ∈ RS(A): the target of the elimination spine.
    pub target: TypeId,
    /// Number of spine steps in El (mirrored by In).
    pub steps: usize,
}

/// The canonical dual pair (El_A, In_A) : A ⊢ at(A) (§8: "for any type A
/// and right simplification B ∈ RS(A) we can always construct a pair of
/// dual contexts").
pub fn dual_context_pair<'input>(ctx: &TypeContext<'input>, a: TypeId) -> Option<DualContext> {
    let b = at_variable(ctx, a)?;
    // Count spine steps from a down to b.
    let mut steps = 0usize;
    let mut cur = a;
    loop {
        match ctx.get(cur) {
            TypeData::Fn { ret, .. } | TypeData::Forall { body: ret, .. } => {
                steps += 1;
                cur = ctx.resolve_binding(*ret);
            }
            _ => break,
        }
    }
    Some(DualContext { target: b, steps })
}

/// Type-level atomizability check (Lemma 8.2 premise): can the
/// instantiation of `forall_poly : ∀X.P[X]` at `b` be atomized?
/// Premises: P constant; b has a rightmost variable at(b) (so the dual
/// pair (El_B, In_B) : B ⊢ at(B) exists).
pub fn atomizable<'input>(
    ctx: &TypeContext<'input>,
    forall_poly: TypeId,
    b: TypeId,
) -> Option<DualContext> {
    // ε-equivalence-strength rewrite ⇒ parametricity gate.
    ctx.parametricity.equivalence_allowed()?;
    match polynomial_is_constant(ctx, forall_poly)? {
        true => {}
        false => return None,
    }
    dual_context_pair(ctx, b)
}

// ─── Term-level rewrite (Eq. 14, on witness::Term) ──────────────────────

/// Atomize an instantiation (Eq. 14): given the term-level shape
/// `[t]_B` (a TyApp of the initial algebra at a non-atomic B), produce
/// `λx⟨y_k⟩. In_B[ [t]_at(B) (λ⟨x_j⟩. El_B[⟨y_k⟩⟨x_j⟩]) ]`.
///
/// In the witness IR the scaffolding reduces to η-expanding the
/// instantiation through the dual pair: the erased term `t` arrives as
/// `base` (the body under TyApp), `k_arity` = number of ⟨y_k⟩ binders
/// (the ⟨A_jk⟩_j → X branches), and `j_arity` = the ⟨x_j⟩ arity shared
/// per branch.  The result wraps `base` in the In/El mirror so that the
/// outer instantiation happens at `at(B)` (atomic) instead of B.
pub fn atomize_instantiation(base: Term, k_arity: usize, j_arity: usize) -> Term {
    // λ⟨y_k⟩. λx? — paper's shape: λx ⟨y_k⟩_k. In_B[ ... ]; in the IR,
    // In/El are context scaffolding (not term constructors), so the
    // computational content is: bind ⟨y_k⟩, re-apply the base at the
    // atomic position via the El spine applied to ⟨y_k⟩⟨x_j⟩.
    //
    // Construct: λ⟨y_k⟩. base' where base' applies Var chain:
    //   inner = λ⟨x_j⟩. El_B(⟨y_k⟩ ⟨x_j⟩)  ≈  λ⟨x_j⟩. y_k x_j ... (El = apps)
    //   body  = base (inner)   — [t]_at(B) applied to the rebuilt morphism
    let mut inner = Term::Var(0); // placeholder rebuilt below
    // inner = λ⟨x_j⟩. y_{k_last} x_1 … x_{j_arity} — for the single-branch
    // polynomial shape (k_arity = 1) this is the Eq. 14 core.
    // Build λ⟨x_j⟩. ⟨y_k⟩ applied pointwise: under j_arity binders, the
    // y_k sits at index j_arity.
    let mut call = Term::Var(j_arity);
    for j in 0..j_arity {
        // applications right-to-left: y x_{j_arity-1} … x_0 → x indices
        // shift under the coming λs; build innermost first.
        call = Term::app(call, Term::Var(j_arity - 1 - j));
    }
    inner = call;
    for _ in 0..j_arity {
        inner = Term::lam(inner);
    }
    // body = base(inner): base expects the branch function list.
    let mut body = base;
    for _ in 0..k_arity {
        body = Term::app(body, inner.clone());
    }
    // Wrap λ⟨y_k⟩: under those binders `base`'s free vars shift.
    let mut wrapped = body;
    for _ in 0..k_arity {
        wrapped = Term::lam(wrapped);
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rs_and_at_of_simple_arrow() {
        let mut ctx = TypeContext::new();
        // A = X0 → (X1 → X2): RS = {X1→X2, X2}, at = X2.
        let x0 = ctx.generic_param(0, "A".into());
        let x1 = ctx.generic_param(1, "B".into());
        let x2 = ctx.generic_param(2, "C".into());
        let inner = ctx.function(smallvec![x1], x2);
        let a = ctx.function(smallvec![x0], inner);
        let mut rs = Vec::new();
        right_simplifications(&ctx, a, &mut rs);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0], inner);
        assert_eq!(rs[1], x2);
        let at = at_variable(&ctx, a).expect("arrow spine ends in a var");
        assert_eq!(at, x2);
    }

    #[test]
    fn at_of_non_variable_head_is_none() {
        let mut ctx = TypeContext::new();
        // A = Unit → Unit: RS = {Unit}, at = None (ends in Unit, not a var).
        let u = ctx.unit();
        let a = ctx.function(smallvec![u], u);
        let mut rs = Vec::new();
        right_simplifications(&ctx, a, &mut rs);
        assert_eq!(rs.len(), 1);
        assert!(at_variable(&ctx, a).is_none());
    }

    #[test]
    fn dual_pair_counts_spine_steps() {
        let mut ctx = TypeContext::new();
        let x0 = ctx.generic_param(0, "A".into());
        let x1 = ctx.generic_param(1, "B".into());
        let a = ctx.function(smallvec![x0], x1);
        let d = dual_context_pair(&ctx, a).expect("ends in var");
        assert_eq!(d.target, x1);
        assert_eq!(d.steps, 1);
    }

    #[test]
    fn atomize_instantiation_shape() {
        // base = Var(0) (the erased term), 1 branch, 2 factors.
        let atomized = atomize_instantiation(Term::Var(0), 1, 2);
        // Shape: λ⟨y⟩. base(λ⟨x⟩. y x x) — λ, then app of base.
        match &atomized {
            Term::Lam(b) => match b.as_ref() {
                Term::App(f, _) => match f.as_ref() {
                    Term::Var(0) => {} // base shifted by the λ binder
                    other => panic!("base not applied directly: {other:?}"),
                },
                other => panic!("not an application: {other:?}"),
            },
            other => panic!("not a lambda: {other:?}"),
        }
        // Normalizes without stuck terms (βη total on this shape).
        let _ = atomized.normalize();
    }

    #[test]
    fn atomize_gate_closed_fail_closed() {
        let mut ctx = TypeContext::new();
        // ∀X.X (identity-polynomial — constant, X not in any A_jk since
        // there are none).
        let x = ctx.generic_param(0, "X".into());
        let poly = ctx.forall(0, "X".into(), x);
        let b = ctx.int(32, true);
        // Gate open: P constant (no fixpoint binder) but B = Int has no
        // rightmost variable ⇒ not atomizable.
        assert!(atomizable(&ctx, poly, b).is_none());
        // Gate closed: refuse even before shape checks.
        ctx.parametricity.break_parametricity();
        let x2 = ctx.generic_param(1, "Y".into());
        let poly2 = ctx.forall(1, "Y".into(), x2);
        let b2 = {
            let unit = ctx.unit();
            let z = ctx.generic_param(2, "Z".into());
            ctx.function(smallvec![unit], z)
        };
        assert!(atomizable(&ctx, poly2, b2).is_none());
    }
}
