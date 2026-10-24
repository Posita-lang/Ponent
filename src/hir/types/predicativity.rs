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

use super::rewrite::from_occ;
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

/// Is P[X] a *constant* polynomial type (§8)?  P is constant iff X does
/// not occur in any of the A_jk (isomorphism (13)).
///
/// B5: decided by DIRECT OCCURRENCE on the unreduced tree, not by
/// constructor-binder presence — `yoneda_step` always KEEPS the {µX.}
/// binder (the canceling elision is βη-redundant but deliberately not
/// performed, see rewrite.rs), so a reduct carries the fixpoint binder
/// even for a constant P and `head != usize::MAX` cannot discriminate.
///
/// Scope (fail-closed): §8 defines the polynomial shape as
/// P[X] = ⟨(A_jk(X) ⇒ X̄) ⇒ X⟩_k ⇒ X — every branch of r_X must be
/// X̄-headed.  X's occurrences are then exactly: the r_X head, the branch
/// heads (d=1), and — when P is NON-constant — deeper occurrences inside
/// some A_jk (d ≥ 2).  A tree where some r_X-branch is NOT X̄-headed does
/// not fit the §8 shape (its ⟨X⇒A_jk(X̄)⟩-style branch domains are
/// indistinguishable from A_jk occurrences at this tree granularity) ⇒
/// `None`.  Incoherent P (κ=∞) likewise ⇒ `None`.
pub fn polynomial_is_constant<'input>(ctx: &TypeContext<'input>, poly: TypeId) -> Option<bool> {
    // poly must be a universal type.
    let TypeData::Forall { param_index, .. } = ctx.get(poly) else {
        return None;
    };
    let x = *param_index;
    let ct = ctx.build_coherence_tree(poly)?;
    if ct.flat().characteristic() == u8::MAX {
        return None; // κ=∞: fail-closed
    }
    let t = from_occ(ct.root_tree());
    // Single walk over the pre-reduction tree (Occ nodes only): locate
    // r_X (the unique node whose binders contain x — place_binders sinks
    // binders to the LCA of their occurrences, so every x occurrence lies
    // inside r_X's subtree), classify x occurrences by distance, and
    // check the (Y⁺) branch shape at r_X.
    fn walk(
        t: &super::rewrite::PolyTree,
        x: usize,
        d_from_r: Option<usize>,
        st: &mut (bool, bool, bool), // (shape_ok, non_constant, found_r)
    ) {
        use super::rewrite::PolyTree;
        let PolyTree::Occ(n) = t else {
            st.0 = false; // pre-reduction trees are Occ-only; fail closed
            return;
        };
        let is_r = n.binders.contains(&x);
        if is_r {
            st.2 = true;
        }
        let d_here = if is_r { Some(0) } else { d_from_r.map(|d| d + 1) };
        if let Some(d) = d_here {
            if n.label == x && d >= 2 {
                st.1 = true; // X inside some A_jk
            }
            if d == 0 {
                // (Y⁺) shape: every branch of r_X is X̄-headed.
                for c in &n.children {
                    match c {
                        PolyTree::Occ(cn) if cn.label == x => {}
                        _ => st.0 = false,
                    }
                }
            }
        }
        for c in &n.children {
            walk(c, x, d_here, st);
        }
    }
    let mut st = (true, false, false);
    walk(&t, x, None, &mut st);
    let (shape_ok, non_constant, found_r) = st;
    if !shape_ok {
        return None; // not the §8 (Y⁺) polynomial shape
    }
    if !found_r {
        // Vacuous ∀X (binder dropped at lowering): X occurs nowhere ⇒
        // trivially constant.
        return Some(true);
    }
    Some(!non_constant)
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
/// `λ⟨y_k⟩. In_B[ [t]_at(B) (λ⟨x_j⟩. El_B[⟨y_k⟩⟨x_j⟩]) ]`.
///
/// The dual pair (El_B, In_B) : B ⊢ at(B) is consumed as follows:
/// - *El_B* (elimination): B = C₁→…→C_s→at(B), so eliminating through
///   B applies the ⟨y_k⟩ handler chain straight to the ⟨x_j⟩ factors —
///   the El spine is exactly the y·x application below;
/// - *In_B* (introduction): rebuilding a B-typed value from an at(B)
///   value η-expands by the s = `dual.steps` spine steps:
///   In_B[w] = λz₁…λz_s. w z₁ … z_s.
///
/// `k_arity` = number of ⟨y_k⟩ binders (the ⟨A_jk⟩_j → X branches),
/// `j_arity` = the ⟨x_j⟩ arity shared per branch.  Fail-closed premises
/// (P constant, gate open, B with a rightmost variable) are checked by
/// `atomizable` before this rewrite is offered.
pub fn atomize_instantiation(
    base: Term,
    dual: &DualContext,
    k_arity: usize,
    j_arity: usize,
) -> Term {
    // inner = λ⟨x_j⟩. y x_1 … x_{j_arity} — El_B applied to ⟨y_k⟩⟨x_j⟩:
    // under the j_arity binders the (last) y sits at index j_arity.
    let mut call = Term::Var(j_arity);
    for j in 0..j_arity {
        // applications right-to-left: y x_{j_arity-1} … x_0 → x indices
        // shift under the coming λs; build innermost first.
        call = Term::app(call, Term::Var(j_arity - 1 - j));
    }
    let mut inner = call;
    for _ in 0..j_arity {
        inner = Term::lam(inner);
    }
    // body = base(inner): [t]_at(B) applied to the rebuilt morphism.
    let mut body = base;
    for _ in 0..k_arity {
        body = Term::app(body, inner.clone());
    }
    // In_B[body] = λz₁…λz_s. body z₁ … z_s (s = dual.steps, verbatim the
    // In_B clause above).  Build the application spine with final-depth
    // indices in ONE pass: the previous iterative `λa. body a` wrapping
    // embedded a β-redex at every step ((λa. body a) b), and that redex's
    // β-normal form collapses back to a SINGLE η layer — losing the s-ary
    // spine the dual pair prescribes.
    let s = dual.steps;
    if s > 0 {
        // z₁ is the OUTERMOST η binder: under the full s-deep prefix its
        // index is s-1; … z_s is Var(0).  Applications go z₁ first.
        for j in 0..s {
            body = Term::app(body, Term::Var(s - 1 - j));
        }
        for _ in 0..s {
            body = Term::lam(body);
        }
    }
    // Wrap λ⟨y_k⟩: under those binders `body`'s free vars shift.
    let mut wrapped = body;
    for _ in 0..k_arity {
        wrapped = Term::lam(wrapped);
    }
    wrapped
}

#[cfg(all(test, not(feature = "shuttle")))]
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
        // base = Var(0) (the erased term), 1 branch, 2 factors, and a
        // dual pair with s = 0 spine steps (B already ends in at(B)).
        let mut ctx = TypeContext::new();
        let dual = DualContext { target: ctx.unit(), steps: 0 };
        let atomized = atomize_instantiation(Term::Var(0), &dual, 1, 2);
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
    fn atomize_instantiation_consumes_dual_steps() {
        // In_B η-expands by exactly `dual.steps` λ layers on top of the
        // base application, mirroring B = C₁→…→C_s→at(B).  The expanded
        // binders appear as λa. body a wrapped around the base (the
        // application is η, not necessarily consecutive at the spine),
        // so compare TOTAL lambda counts: steps=s adds exactly s λs.
        fn count_lams(t: &Term) -> usize {
            let inner = match t {
                Term::Lam(b) | Term::TyLam(b) => return 1 + count_lams(b),
                Term::App(f, a) => return count_lams(f) + count_lams(a),
                Term::InSum { value, .. } | Term::Proj { value, .. } => {
                    return count_lams(value)
                }
                Term::SumCase {
                    scrutinee,
                    branches,
                } => {
                    let mut n = count_lams(scrutinee);
                    for b in branches {
                        n += count_lams(b);
                    }
                    return n;
                }
                Term::Pair(ps) => {
                    let mut n = 0;
                    for p in ps {
                        n += count_lams(p);
                    }
                    return n;
                }
                Term::Fold { body } | Term::Unfold { body } => return count_lams(body),
                Term::Var(_) | Term::TyApp(..) => 0,
            };
            let _ = inner;
            0
        }
        let mut ctx = TypeContext::new();
        let unit = ctx.unit();
        let a0 = atomize_instantiation(
            Term::Var(0),
            &DualContext { target: unit, steps: 0 },
            1,
            2,
        );
        let a2 = atomize_instantiation(
            Term::Var(0),
            &DualContext { target: unit, steps: 2 },
            1,
            2,
        );
        let n0 = count_lams(&a0);
        let n2 = count_lams(&a2);
        assert_eq!(
            n2,
            n0 + 2,
            "In_B must η-expand by exactly dual.steps λ layers: {a2:?}"
        );
        // Normalize without stuck terms.
        let _ = a2.normalize();
        // steps = 0 has no η layer: the outermost λ is the y binder and
        // the body applies base directly (no application of the base's
        // result to a spine variable).
        match &a0 {
            Term::Lam(b) => {
                assert!(
                    !matches!(b.as_ref(), Term::Lam(_)),
                    "steps=0 adds no η layer: {a0:?}"
                );
            }
            other => panic!("unexpected shape: {other:?}"),
        }
    }

    #[test]
    fn polynomial_constancy_by_occurrence() {
        // B5 golden: constancy is decided by whether X occurs in the
        // A_jk, NOT by whether the reduct carries a {µX.} binder —
        // yoneda_step always keeps that binder (non-canceling form), so
        // the old has_fixpoint check reported "non-constant" even for a
        // constant P.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        // Constant: ∀X.(Int→X)→X — A_jk = Int contains no X ⇒ constant
        // (isomorphism (1): ≅ Int, no fixpoint needed).
        let i = ctx.int(32, true);
        let ix = ctx.function(smallvec![i], x);
        let body = ctx.function(smallvec![ix], x);
        let ty = ctx.forall(0, "X".into(), body);
        assert_eq!(
            polynomial_is_constant(&ctx, ty),
            Some(true),
            "A_jk = Int: no X occurrence in any factor ⇒ constant"
        );
        // Non-constant: ∀X.(X→X)→X — A_jk = X ⇒ a genuine X occurrence
        // survives in the reduct (the μX recursion).
        let xx = ctx.function(smallvec![x], x);
        let body2 = ctx.function(smallvec![xx], x);
        let ty2 = ctx.forall(0, "X".into(), body2);
        assert_eq!(
            polynomial_is_constant(&ctx, ty2),
            Some(false),
            "A_jk = X: genuine X occurrence in the reduct ⇒ non-constant"
        );
        // Fail-closed: a non-(Y⁺) shape (branch NOT X̄-headed — here the
        // (Y⁻) shape ⟨X⇒Int⟩⇒X, Yoneda-equivalent to Int by cancellation)
        // is outside §8's constancy definition ⇒ None.
        let xi = ctx.function(smallvec![x], i);
        let body3 = ctx.function(smallvec![xi], x);
        let ty3 = ctx.forall(0, "X".into(), body3);
        assert_eq!(
            polynomial_is_constant(&ctx, ty3),
            None,
            "branch ⟨X⇒Int⟩ is not X̄-headed: outside the §8 (Y⁺) shape ⇒ None"
        );
    }

    #[test]
    fn atomize_in_beta_normal() {
        // B1 golden: the In_B η-expansion is built in ONE pass with
        // final-depth indices — the previous iterative `λa. body a`
        // wrapping embedded a β-redex at every step, and that redex's
        // β-normal form collapsed back to a single η layer, losing the
        // s-ary spine In_B[w] = λz₁…λz_s. w z₁…z_s.  Golden: the result
        // is already βη-normal (normalize fixpoint) with the verbatim
        // spine shape.
        let mut ctx = TypeContext::new();
        let unit = ctx.unit();
        let a2 = atomize_instantiation(
            Term::Var(0),
            &DualContext { target: unit, steps: 2 },
            1,
            2,
        );
        // βη fixpoint: no β-redexes, no η-collapsible layers.
        assert_eq!(
            a2.clone().normalize(),
            a2,
            "atomized term must be βη-normal: {a2:?}"
        );
        // Shape: λy. λz₁. λz₂. (base …) z₁ z₂ — applications in z₁-then-z₂
        // order (de Bruijn under the s=2 prefix: z₁ = Var(1), z₂ = Var(0)).
        match &a2 {
            Term::Lam(y) => match y.as_ref() {
                Term::Lam(z1) => match z1.as_ref() {
                    Term::Lam(z2) => match z2.as_ref() {
                        Term::App(f21, a_z2) => {
                            assert!(
                                matches!(a_z2.as_ref(), Term::Var(0)),
                                "z₂ is the innermost binder: {a_z2:?}"
                            );
                            match f21.as_ref() {
                                Term::App(f1, a_z1) => {
                                    assert!(
                                        matches!(a_z1.as_ref(), Term::Var(1)),
                                        "z₁ is the outer η binder: {a_z1:?}"
                                    );
                                    assert!(
                                        matches!(f1.as_ref(), Term::App(..)),
                                        "base application spine intact"
                                    );
                                }
                                other => panic!("no z₁ application: {other:?}"),
                            }
                        }
                        other => panic!("body not an application: {other:?}"),
                    },
                    other => panic!("missing z₂ layer: {other:?}"),
                },
                other => panic!("missing z₁ layer: {other:?}"),
            },
            other => panic!("not a lambda: {other:?}"),
        }
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
