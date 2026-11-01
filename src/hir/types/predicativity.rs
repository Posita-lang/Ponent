//! Predicativity pass — atomization (paper §8).
//!
//! §8 establishes that for a *constant* polynomial type P[X] (i.e. X
//! occurs only in the rightmost position: P[X] = ⟨⟨A_jk⟩_j → X⟩_k → X
//! with X ∉ FV(A_jk)), any instantiation `t_B : ∀X.P[X] ⊢ P[B/X]` can
//! be *atomized* — rewritten (Eq. 14) into a term using an
//! instantiation at `at(B)` (the rightmost variable of B), preserving
//! the contextual behavior (≅_ε):
//!
//! ```text
//! [_]_B  ≅_ε  λ⟨y_k⟩_k. In_B[ [_]_at(B) (λ⟨x_j⟩_j. El_B[⟨y_k⟩⟨x_j⟩_j])_k ]
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
//! The *term-level* rewrite itself lives on `witness::Term`:
//! `atomize_instantiation` rewrites a `TyApp` of the initial
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
/// Decided by DIRECT OCCURRENCE on the unreduced tree, not by
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
/// `None`.
///
/// NO coherence gate: Lemma 8.2 (ar5iv L561) quantifies over "all
/// constant polynomial type P[X] and type B of Λ2" — its only premises
/// are P constant + the (Y⁺) shape (+ El being the canonical,
/// trivially-algebraic pair).  κ (§5's a-priori quantifier-elimination
/// verdict) is NOT a premise, and the shape walk below is purely
/// structural.  The former κ=∞ refusal rejected atomizations the
/// theorem licenses.
pub fn polynomial_shape<'input>(
    ctx: &TypeContext<'input>,
    poly: TypeId,
) -> Option<PolynomialShape> {
    // poly must be a universal type.
    let TypeData::Forall { param_index, .. } = ctx.get(poly) else {
        return None;
    };
    let x = *param_index;
    let ct = ctx.build_coherence_tree(poly)?;
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
        at_root: bool,
        st: &mut (bool, bool, bool, Vec<usize>), // (shape_ok, non_constant, found_r, j_arities)
    ) {
        use super::rewrite::PolyTree;
        let PolyTree::Occ(n) = t else {
            st.0 = false; // pre-reduction trees are Occ-only; fail closed
            return;
        };
        let is_r = n.binders.contains(&x);
        if is_r {
            st.2 = true;
            // §8's r_X is the ∀-body's FINAL spine — P[X] =
            // ⟨⟨A_jk⟩⇒X̄⟩_k⇒X is the WHOLE type: the binder node must be
            // the tree ROOT with an X⁺ head.  The former walk checked
            // only the color, so an INTERIOR Blue LCA (X buried in an
            // argument, e.g. ∀X.(((X⇒X)⇒Int)⇒C)) or a non-X-headed root
            // (e.g. ∀X.(X⇒X⇒Int)) passed the branch gate and was
            // misreported constant — an Eq. 14 offer with no Lemma 8.2
            // license.  Internalizability reading: Eq. 14's In_B wrap
            // needs the conclusion type's HEAD to be B, which holds iff
            // the polynomial IS the whole type (see the (Y⁻) note
            // below).
            if !at_root || n.label != x {
                st.0 = false;
            }
            // (Y⁻) FAIL-CLOSED JUSTIFICATION (upgraded from "no textual
            // basis" to a structural one).  A RED r_X means X's LCA is an
            // interior X̄ spine — P does not end in X — outside §8's scope
            // entirely (the (Y⁻)-fragment family, e.g. ∀X.((X→X)→C)).
            // (i) §8's engine is INITIALITY of the positive P-algebra
            // (ar5iv L560); Lemma 8.2's premises are constant-P + B ∈ Λ2
            // only (L561–2) and no dual statement exists — the paper
            // itself defers the "intrinsic duality in the tree-shapes of
            // Fig. 6" to future work (L572).  (ii) The mirrored equation,
            // while TRUE in the relational model (hexagon + both pair
            // laws + ξ-congruence, for all s), is NOT internalizable as a
            // closed same-type rewrite: Def 8.1's E-grammar is
            // binder-free (E ::= [] | E C | E u), and the dual conclusion
            // type H[S(B)] is headed by the X-free R — no In-site exists
            // to discharge the El plumb's binders.  (iii) The type-level
            // (Y⁻) reduction IS covered — rewrite.rs's red rule — which
            // is the granularity the paper provides.  A term-level dual
            // would be a type-changing η-long elaboration
            // (λz⃗.t_B(s) ≃ λz⃗.t_at(B)(plumb(s)) :
            // (C₁⇒…⇒at(B))⇒R), out of `atomize`'s type-preserving
            // contract; future witness-elaboration work.
            if n.color != super::coherence::Color::Blue {
                st.0 = false;
            }
        }
        let d_here = if is_r {
            Some(0)
        } else {
            d_from_r.map(|d| d + 1)
        };
        if let Some(d) = d_here {
            if n.label == x && d >= 2 {
                st.1 = true; // X inside some A_jk
            }
            if d == 0 {
                // (Y⁺) shape: every branch is X̄-headed — label AND color.
                // Under a Blue r_X parity already forces Red children; the
                // color test makes the documented invariant explicit, not
                // parity-accidental.  Branch k passes the gate ⇒ its
                // factor count IS j_k (P[X] = ⟨⟨A_jk⟩_j ⇒ X⟩_k ⇒ X,
                // paper L544–545).
                for c in &n.children {
                    match c {
                        PolyTree::Occ(cn)
                            if cn.label == x && cn.color == super::coherence::Color::Red =>
                        {
                            st.3.push(cn.children.len())
                        }
                        _ => st.0 = false,
                    }
                }
            }
        }
        for c in &n.children {
            walk(c, x, d_here, false, st);
        }
    }
    let mut st = (true, false, false, Vec::new());
    walk(&t, x, None, true, &mut st);
    let (shape_ok, non_constant, found_r, j_arities) = st;
    if !shape_ok {
        return None; // not the §8 (Y⁺) polynomial shape
    }
    if !found_r {
        // Vacuous ∀X (binder dropped at lowering): X occurs nowhere ⇒
        // trivially constant, k=0.
        return Some(PolynomialShape {
            constant: true,
            j_arities: Vec::new(),
        });
    }
    Some(PolynomialShape {
        constant: !non_constant,
        j_arities,
    })
}

/// The §8 polynomial shape of `∀X.P[X]`, read off the occurrence tree:
/// `constant` = X occurs at no d ≥ 2 (X ∉ FV(A_jk), isomorphism (13));
/// `j_arities[k]` = branch k's factor count ⟨A_jk⟩_j.  Arities are in
/// tree (sorted-branch) order — Eq. 14's handler order is the same
/// order, and the branch permutation is Lemma 4.1.ii-neutral.
/// k = 0 (vacuous ∀X) degenerates to `j_arities = []`, for which
/// `atomize_instantiation` returns `In_B[base]`/`base` — Eq. 14's k=0
/// degeneration verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialShape {
    pub constant: bool,
    pub j_arities: Vec<usize>,
}

/// Backward-compatible wrapper (existing goldens unchanged).
pub fn polynomial_is_constant<'input>(ctx: &TypeContext<'input>, poly: TypeId) -> Option<bool> {
    polynomial_shape(ctx, poly).map(|s| s.constant)
}

/// The blessed entry point of the §8 atomization: runs EVERY premise
/// (gate, P constant, the (Y⁺) shape, the dual pair), extracts the
/// per-branch arities from the type itself, and only then constructs
/// the Eq. 14 term.  Turns `atomize_instantiation`'s documented
/// precondition into a STRUCTURAL one — unvetted (k, j_arities) can no
/// longer reach the raw constructor from the pipeline.
pub fn atomize<'input>(
    ctx: &TypeContext<'input>,
    forall_poly: TypeId,
    b: TypeId,
    base: Term,
) -> Option<Term> {
    ctx.parametricity.equivalence_allowed()?;
    let shape = polynomial_shape(ctx, forall_poly)?;
    if !shape.constant {
        return None;
    }
    let dual = dual_context_pair(ctx, b)?;
    Some(atomize_instantiation(base, &dual, &shape.j_arities))
}

/// One elimination step of a dual context spine (§8 Def 8.1).  The spine
/// conflates TWO levels — the former bare `steps: usize` count built a
/// term-level λ/application mirror for a leading ∀ step, which is
/// ill-typed: a ∀ step's mirror is Λ/type-application (Eq. 14's El/In
/// act at the type level there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// `C ⇒ …`: term-level mirror (λ / application).
    Fn,
    /// `∀Y. …`: type-level mirror (Λ = TyLam / TyApp).  `ty` is the
    /// TyApp marker for the bound variable — witness::Term carries type
    /// args as bare `usize` markers, so we store the ∀'s param index
    /// (a GenericParam TypeId would have no Term-level carrier).  The
    /// marker is kept SPINE-DISTINCT by `dual_context_pair`'s α-renaming
    /// of shadowed binders (a fresh index replaces a repeat).  CLOSED as
    /// CLOSED (the upgrade recipe is recorded): the usize marker is
    /// sufficient for the atomize term's consumption (structural match +
    /// βη).  If renderable, de-Bruijn-independent markers are ever
    /// needed, the upgrade is a mechanical slice: ① widen
    /// `Term::TyApp(Box<Term>, TypeId)`; ② have `dual_context_pair` mint
    /// `ctx.alloc(TypeData::GenericParam{..})` and store it here; ③ sync
    /// `atomize_forall_step_is_type_level`'s `TyApp(_, 0)` matcher.
    Forall { ty: usize },
}

/// A dual context pair (El, In) : A ⊢ B (§8 Def 8.1) at the type level:
/// the elimination spine (type/term applications) from A down to B, and
/// the mirror introductions.  We represent it by the B endpoint plus the
/// typed spine steps (El = [] C₁ … Cₙ u₁ … uₘ, In = Λ/λ mirror).
#[derive(Clone, Debug)]
pub struct DualContext {
    /// B ∈ RS(A): the target of the elimination spine.
    pub target: TypeId,
    /// Spine steps OUTERMOST FIRST.  Replaces the former `steps: usize`
    /// (the bare count conflated the two levels → ill-typed term under a
    /// leading ∀).  `step_count()` preserves the old read.
    pub steps: Vec<StepKind>,
}

impl DualContext {
    /// Total spine-step count across BOTH levels (Fn + Forall).  Under
    /// the two-pool discipline only `Fn` steps occupy term de Bruijn
    /// slots — any consumer doing index arithmetic must count Fn steps
    /// itself (as `atomize_instantiation` does with its `f`), never use
    /// this number as a slot count.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// The canonical dual pair (El_A, In_A) : A ⊢ at(A) (§8: "for any type A
/// and right simplification B ∈ RS(A) we can always construct a pair of
/// dual contexts").
pub fn dual_context_pair<'input>(ctx: &TypeContext<'input>, a: TypeId) -> Option<DualContext> {
    let b = at_variable(ctx, a)?;
    // Record the typed spine steps from a down to b.
    let mut steps: Vec<StepKind> = Vec::new();
    // Notation 5.1: bound variables are pairwise distinct.  A repeated
    // param index on the spine would make the TyApp marker ambiguous
    // (two ∀ steps sharing one tag) — discharged by α-renaming below.
    let mut seen: Vec<usize> = Vec::new();
    let mut cur = a;
    loop {
        match ctx.get(cur) {
            TypeData::Fn { ret, .. } => {
                steps.push(StepKind::Fn);
                cur = ctx.resolve_binding(*ret);
            }
            TypeData::Forall {
                param_index,
                body: ret,
                ..
            } => {
                // The paper fixes a dual pair "for ALL type A" (L561) —
                // unconditionally.  Notation 5.1's pairwise-distinctness
                // is a NAMING convention discharged by α-renaming; the
                // marker only needs to be SPINE-DISTINCT (TyApp erasure
                // is positional — witness.rs's TyApp(TyLam(b), _) arm
                // ignores the marker).  A repeated index is therefore
                // α-renamed to a fresh marker instead of refusing: the
                // former fail-closed None rejected a pair the paper
                // guarantees.
                let marker = if seen.contains(param_index) {
                    // Rename past the WHOLE seen set: a fresh index could
                    // itself collide with a real binder index already on
                    // the spine (e.g. a counter that starts at 0 while
                    // param 0 is on the spine).
                    let mut m = ctx.fresh_param_index();
                    while seen.contains(&m) {
                        m = ctx.fresh_param_index();
                    }
                    m
                } else {
                    *param_index
                };
                seen.push(marker);
                steps.push(StepKind::Forall { ty: marker });
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
    // Delegate to the SINGLE shape-walk source — the former inlined
    // second walk inevitably drifted (the root/head check was
    // exactly the kind of fix one walk got and the other missed).
    if !polynomial_shape(ctx, forall_poly)?.constant {
        return None;
    }
    dual_context_pair(ctx, b)
}

// ─── Term-level rewrite (Eq. 14, on witness::Term) ──────────────────────

/// Atomize an instantiation (Eq. 14): given the term-level shape
/// `[t]_B` (a TyApp of the initial algebra at a non-atomic B), produce
/// `λ⟨y_k⟩. In_B[ [t]_at(B) (λ⟨x_j⟩. El_B[⟨y_k⟩⟨x_j⟩]) ]`.
///
/// INTERNAL — prefer `atomize`, which runs EVERY premise structurally
/// (gate, P constant, the (Y⁺) shape incl. r_X = root ∧ X-headed, the
/// dual pair) and extracts `j_arities` from the type itself.  This raw
/// constructor performs no premise checks: an unvetted (j_arities,
/// dual) pair can silently produce an Eq. 14 term with no Lemma 8.2
/// license.  Signature kept `pub` — witness/yoneda-side callers pass
/// zero changes.
///
/// Precondition (fail-closed at the type level): the caller must have
/// checked `atomizable(ctx, poly, b).is_some()` (P constant + gate open +
/// B with a rightmost variable) — this function performs no premise
/// checks of its own.
///
/// The dual pair (El_B, In_B) : B ⊢ at(B) is consumed as follows:
/// - *El_B* (elimination): B = C₁→…→C_s→at(B); per Def 8.1's E-grammar
///   (`E ::= [] | E C | E u`) eliminating through B applies the ⟨y_k⟩
///   handler chain straight to the ⟨x_j⟩ factors AND forwards the mirror
///   binders into the handler — the El spine is exactly the y·x·z⃗·@Ȳ
///   application below;
/// - *In_B* (introduction): per Def 8.1's I-grammar
///   (`I ::= [] | ΛY.I | λy.I`) it is a PURE mirror of abstractions —
///   `In_B[w] = ΛȲ.λz⃗. w`, never `w z⃗ @Ȳ`.  The dual-pair law
///   `In ∘ El ≃_η []` (Def 8.1) holds by βη only for the pure mirror,
///   and Eq. 14's typing (`In_B[W : at(B)] : B`) closes by pure wrapping
///   alone — an at(B)-typed term has no type grounds to consume B's spine
///   arguments.  The z⃗/Ȳ binders introduced by In are consumed
///   EXCLUSIVELY by the handlers' El forwarding.
///
/// `j_arities[i]` = the ⟨x_j⟩ arity of branch i — j is indexed PER
/// BRANCH (the polynomial shape P[X] = ⟨⟨A_jk⟨X⟩⟩_j ⇒ X⟩_k ⇒ X; a
/// uniform j_arity silently mis-types heterogeneous P).
///
/// # Base-index contract (two-pool)
///
/// `base` is passed in AMBIENT scope (index 0 = the caller's `[t]_at(B)`)
/// and is shifted INTERNALLY by `k_arity + f`, where `f` = the number of
/// `Fn` steps (`StepKind::Forall` is a Λ/TyApp marker — two-pool: it
/// consumes no term slot).  The rewrite introduces `k_arity` y-binders
/// OUTERMOST plus the `Fn` η-binders between them and the body; without
/// the shift the ambient `Var(0)` would silently resolve to the innermost
/// introduced binder.  Callers therefore do NOT pre-shift.
pub fn atomize_instantiation(base: Term, dual: &DualContext, j_arities: &[usize]) -> Term {
    let steps = &dual.steps;
    let k_arity = j_arities.len();
    // Two-pool: only Fn steps occupy term slots; ∀ steps are Λ/TyApp
    // markers, transparent to the de Bruijn space.
    let fn_count = |range: &[StepKind]| {
        range
            .iter()
            .filter(|s| matches!(s, StepKind::Fn { .. }))
            .count()
    };
    let f = fn_count(steps);
    // Eq. (14): each of the k branch handlers y_k gets its OWN
    // λ⟨x_j⟩.El_B[y_k⟨x_j⟩] argument.  Branch i (0-based application
    // order; i = 0 is the outermost arrow = y_1) consumes y_{i+1}, which
    // under the final binder scope (branch-i x's innermost, then the Fn
    // η-binders, then y_1..y_k outermost) sits at index
    // j_i + f + (k_arity - 1 - i).  The former single Var(j_arity + n)
    // (= y_k) fed the SAME handler to every branch, leaving
    // y_1..y_{k-1} unused and mis-typing every k > 1 instantiation.
    //
    // El FORWARDING: Eq. 14's El_B[⟨y_k⟩⟨x_j⟩] must forward the SAME
    // mirror binders into the handler — the base needs a ⟨A⟩⇒at(B)-shaped
    // handler, but y⟨x⟩ lands in B-shaped territory; each spine step is
    // applied inside the handler (term apps for Fn steps, TyApp for ∀
    // steps; the canonical pair shares one binder skeleton).  Under the
    // handler scope (branch-i x's innermost, then the Fn η-binders
    // outward), the z binder introduced by the Fn step at position s sits
    // at j_i + (Fn steps AFTER s) — ∀ steps add no slot.
    // body = base(inner): [t]_at(B) applied to the rebuilt morphisms.
    // base arrives in ambient scope and is shifted internally past the
    // k_arity y-binders + f Fn η-binders that this rewrite will wrap
    // around it (see the base-index contract above).
    let mut body = base.shift(k_arity + f);
    for (i, &j_i) in j_arities.iter().enumerate() {
        let mut call = Term::Var(j_i + f + (k_arity - 1 - i));
        for j in 0..j_i {
            // applications right-to-left: y x_{j_i-1} … x_0 → x indices
            // shift under the coming λs; build innermost first.
            call = Term::app(call, Term::Var(j_i - 1 - j));
        }
        for (s, step) in steps.iter().enumerate() {
            call = match step {
                StepKind::Fn => {
                    let z_idx = j_i + fn_count(&steps[s + 1..]);
                    Term::app(call, Term::Var(z_idx))
                }
                StepKind::Forall { ty } => Term::TyApp(Box::new(call), *ty),
            };
        }
        let mut inner = call;
        for _ in 0..j_i {
            inner = Term::lam(inner);
        }
        // [t]_at(B)'s i-th curried arrow consumes the i-th handler.
        body = Term::app(body, inner);
    }
    // In_B[body] = the PURE introduction mirror (Def 8.1: I ::= [] |
    // ΛY.I | λy.I; dual-pair law In ∘ El ≃_η []): wrapping ONLY — the
    // at(B)-typed spine is NEVER applied to the In binders.  The z/Y
    // binders flow into the handlers' El forwarding exclusively (the
    // E-grammar pass above).  The former application pass emitted
    // (…) z⃗ @Ȳ after the base spine — outside the I-grammar, ill-typed
    // (applying an at(B)-typed term to B's spine arguments),
    // double-consuming the z's, and breaking In ∘ El ≃_η [].  Wrapping
    // goes in REVERSE so that step_0 ends up OUTERMOST (λ for Fn steps,
    // Λ = TyLam for ∀ steps).
    for step in steps.iter().rev() {
        body = match step {
            StepKind::Fn => Term::lam(body),
            StepKind::Forall { .. } => Term::tylam(body),
        };
    }
    // Wrap λ⟨y_k⟩.
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
        assert_eq!(d.step_count(), 1);
        assert_eq!(d.steps, vec![StepKind::Fn]);
    }

    #[test]
    fn atomize_instantiation_k2_regression() {
        // k=2：λy₁.λy₂. base (λx. y₁ x) (λx. y₂ x) — each branch gets its
        // OWN handler binder (the former single-Var version fed the SAME
        // binder to BOTH branches, leaving the other unused).  Eq. (14)'s
        // spine is LEFT-NESTED with base at the head:
        // App(App(base, H₁), H₂).
        //
        // Base-index contract: base is passed in AMBIENT scope (Var(0))
        // and shifted internally by k+f = 2+0 — the two y-binders the
        // rewrite introduces occupy slots 0/1, so base surfaces at slot 2.
        let mut ctx = TypeContext::new();
        let dual = DualContext {
            target: ctx.unit(),
            steps: vec![],
        };
        let t = atomize_instantiation(Term::Var(0), &dual, &[1, 1]).normalize();
        let t = match t {
            Term::Lam(b) => *b,
            other => panic!("expected outermost λ, got {other:?}"),
        };
        let t = match t {
            Term::Lam(b) => *b,
            other => panic!("expected second λ, got {other:?}"),
        };
        match t {
            Term::App(f, a) => {
                // f = App(base, H₁): base at slot 2 (slots 0/1 = the two
                // y-binders), H₁ = λx. y₁ x with head Var(2) (= y₁ under
                // x=0, y₂=1, y₁=2).
                let (base_slot, h1) = match f.as_ref() {
                    Term::App(b, h) => (b.as_ref(), h.as_ref()),
                    other => panic!("branch-1 spine must be App(base, handler): {other:?}"),
                };
                assert!(
                    matches!(base_slot, Term::Var(2)),
                    "base = Var(2): {base_slot:?}"
                );
                let h1_ok = match h1 {
                    Term::Lam(h) => matches!(
                        h.as_ref(),
                        Term::App(g, _) if matches!(g.as_ref(), Term::Var(2))
                    ),
                    other => panic!("H₁ must be a λ: {other:?}"),
                };
                assert!(h1_ok, "branch-1 handler: {h1:?}");
                // H₂ = λx. y₂ x — head Var(1) (= y₂ under x=0, y₂=1).  The
                // two heads DIFFER ⇒ each branch consumes its OWN binder.
                let a_ok = match a.as_ref() {
                    Term::Lam(h) => matches!(
                        h.as_ref(),
                        Term::App(g, _) if matches!(g.as_ref(), Term::Var(1))
                    ),
                    other => panic!("H₂ must be a λ: {other:?}"),
                };
                assert!(a_ok, "branch-2 spine: {a:?}");
            }
            other => panic!("expected per-branch application spine, got {other:?}"),
        }
    }

    #[test]
    fn atomize_instantiation_shape() {
        // base = Var(0) (the erased term), 1 branch, 2 factors, and a
        // dual pair with s = 0 spine steps (B already ends in at(B)).
        let mut ctx = TypeContext::new();
        let dual = DualContext {
            target: ctx.unit(),
            steps: vec![],
        };
        let atomized = atomize_instantiation(Term::Var(0), &dual, &[2]);
        // Shape: λ⟨y⟩. base(λ⟨x⟩. y x x) — λ, then app of base.  base is
        // passed ambient (Var(0)) and shifted internally by k+f = 1: the
        // y-binder the rewrite introduces occupies slot 0, base surfaces
        // at slot 1.
        match &atomized {
            Term::Lam(b) => match b.as_ref() {
                Term::App(f, _) => match f.as_ref() {
                    Term::Var(1) => {} // base internally shifted: ambient 0 + k+f = 1
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
                Term::InSum { value, .. } | Term::Proj { value, .. } => return count_lams(value),
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
            &DualContext {
                target: unit,
                steps: vec![],
            },
            &[2],
        );
        let a2 = atomize_instantiation(
            Term::Var(0),
            &DualContext {
                target: unit,
                steps: vec![StepKind::Fn, StepKind::Fn],
            },
            &[2],
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
        // Constancy golden: constancy is decided by whether X occurs in the
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
    fn polynomial_constancy_rejects_red_r_x_fragments() {
        // X's LCA is an interior X̄ spine (P does not end in X) — the
        // cancellation-family shapes of
        // rewrite.rs::cancellation_family_reduces_soundly_via_red_rule are
        // outside §8's scope. The former
        // label-only test misreported this P as CONSTANT (Some(true)),
        // offering an unsound Eq. 14 atomization (Lemma 8.1's premise does
        // not hold for a P that does not end in X).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let c = ctx.generic_param(1, "C".into());
        let xx = ctx.function(smallvec![x], x); // (X→X)
        // ∀X.((X→X)→C): r_X = the X̄-headed (X→X) spine (RED) ⇒ None.
        let body = ctx.function(smallvec![xx], c);
        let ty = ctx.forall(0, "X".into(), body);
        assert_eq!(
            polynomial_is_constant(&ctx, ty),
            None,
            "Red r_X ⇒ P does not end in X ⇒ outside the §8 (Y⁺) shape"
        );
        // Control (the fix must NOT over-reject): ∀X.((X→X)→X) — Blue r_X,
        // X̄-headed branch, X at d2 inside it ⇒ in scope, NON-constant.
        let body2 = ctx.function(smallvec![xx], x);
        let ty2 = ctx.forall(0, "X".into(), body2);
        assert_eq!(polynomial_is_constant(&ctx, ty2), Some(false));
    }

    #[test]
    fn atomize_in_beta_normal() {
        // In_B is the PURE introduction mirror (Def 8.1's I-grammar): it
        // wraps abstractions only — the at(B)-typed spine is never applied
        // to the In binders.  Golden: the result is already βη-normal
        // (normalize fixpoint) with no η-collapsible layers, and the z
        // applications live exclusively inside the handler's El forwarding.
        let mut ctx = TypeContext::new();
        let unit = ctx.unit();
        let a2 = atomize_instantiation(
            Term::Var(0),
            &DualContext {
                target: unit,
                steps: vec![StepKind::Fn, StepKind::Fn],
            },
            &[2],
        );
        // βη fixpoint: no β-redexes, no η-collapsible layers.
        assert_eq!(
            a2.clone().normalize(),
            a2,
            "atomized term must be βη-normal: {a2:?}"
        );
        // Shape: λy. λz₁. λz₂. base (λx₁. λx₂. ((y x₁ x₂) z₁) z₂) — the
        // z applications are ALL inside the handler (El forwarding); the
        // base spine is NOT applied to the In binders.
        // Exponents under the full prefix: x₂=0, x₁=1, z₂=2, z₁=3, y=4;
        // base = k+f = 3.
        match &a2 {
            Term::Lam(y) => match y.as_ref() {
                Term::Lam(z1) => match z1.as_ref() {
                    Term::Lam(z2) => match z2.as_ref() {
                        Term::App(base_app, handler) => {
                            assert!(
                                matches!(base_app.as_ref(), Term::Var(3)),
                                "base at k+f=3: {base_app:?}"
                            );
                            match handler.as_ref() {
                                Term::Lam(x1) => match x1.as_ref() {
                                    Term::Lam(x2) => match x2.as_ref() {
                                        Term::App(app_z1, a_z2) => {
                                            assert!(
                                                matches!(a_z2.as_ref(), Term::Var(2)),
                                                "z₂ at 2: {a_z2:?}"
                                            );
                                            match app_z1.as_ref() {
                                                Term::App(app_x2, a_z1) => {
                                                    assert!(
                                                        matches!(a_z1.as_ref(), Term::Var(3)),
                                                        "z₁ at 3: {a_z1:?}"
                                                    );
                                                    match app_x2.as_ref() {
                                                        Term::App(app_x1, a_x2) => {
                                                            assert!(matches!(
                                                                a_x2.as_ref(),
                                                                Term::Var(0)
                                                            ));
                                                            match app_x1.as_ref() {
                                                                Term::App(yv, a_x1) => {
                                                                    assert!(matches!(
                                                                        a_x1.as_ref(),
                                                                        Term::Var(1)
                                                                    ));
                                                                    assert!(
                                                                        matches!(
                                                                            yv.as_ref(),
                                                                            Term::Var(4)
                                                                        ),
                                                                        "y at j+f+(k-1)=4"
                                                                    );
                                                                }
                                                                other => panic!("{other:?}"),
                                                            }
                                                        }
                                                        other => panic!("{other:?}"),
                                                    }
                                                }
                                                other => panic!("{other:?}"),
                                            }
                                        }
                                        other => panic!("{other:?}"),
                                    },
                                    other => panic!("missing x₁ layer"),
                                },
                                other => panic!("missing x₂ layer"),
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
    fn dual_pair_renames_shadowed_forall_markers() {
        // §8 (L561): dual pairs exist for ALL type A — unconditionally.
        // A shadowed ∀ binder is α-renamed to a fresh spine-distinct
        // marker; the former fail-closed None refused a guaranteed pair.
        let mut ctx = TypeContext::new();
        let unit = ctx.unit();
        let z = ctx.generic_param(1, "Z".into());
        let unitfnz = ctx.function(smallvec![unit], z);
        // ∀Yₚ.(∀Yₚ.(Unit→Z)): the inner ∀ reuses param index 0.
        let inner = ctx.forall(0, "Y".into(), unitfnz);
        let outer = ctx.forall(0, "Y".into(), inner);
        let dual = dual_context_pair(&ctx, outer).expect("dual pairs always exist");
        match &dual.steps[..] {
            [
                StepKind::Forall { ty: m0 },
                StepKind::Forall { ty: m1 },
                StepKind::Fn,
            ] => {
                assert_eq!(*m0, 0, "first occurrence keeps the real index");
                assert_ne!(*m1, 0, "shadowed binder α-renamed to a fresh marker");
                assert_ne!(m0, m1, "markers spine-distinct (Notation 5.1)");
            }
            other => panic!("unexpected steps: {other:?}"),
        }
        // Control unchanged: distinct indices pass through verbatim.
        let inner_ok = ctx.forall(1, "W".into(), unitfnz);
        let outer_ok = ctx.forall(0, "Y".into(), inner_ok);
        assert_eq!(
            dual_context_pair(&ctx, outer_ok).expect("distinct").steps,
            vec![
                StepKind::Forall { ty: 0 },
                StepKind::Forall { ty: 1 },
                StepKind::Fn
            ]
        );
    }

    #[test]
    fn polynomial_shape_and_structural_atomize_entry() {
        // P[X] = (Int⇒X) ⇒ (Bool⇒Unit⇒X) ⇒ X — heterogeneous j = [1, 2]:
        // j is indexed PER BRANCH (paper L544–545); a uniform j_arity
        // silently mis-types this shape.
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let (i, b, u) = (ctx.int(32, true), ctx.bool(), ctx.unit());
        let br0 = ctx.function(smallvec![i], x);
        let br1 = ctx.function(smallvec![b, u], x);
        let body = ctx.function(smallvec![br0, br1], x);
        let ty = ctx.forall(0, "X".into(), body);
        let shape = polynomial_shape(&ctx, ty).expect("(Y⁺) shape");
        assert!(shape.constant);
        assert_eq!(shape.j_arities, vec![1, 2]);
        // Structural entry: premises checked, arities read from the TYPE.
        let (p0, p1) = (
            ctx.generic_param(2, "P".into()),
            ctx.generic_param(3, "Q".into()),
        );
        let bty = ctx.function(smallvec![p0], p1); // at(b) = Q
        assert!(atomize(&ctx, ty, bty, Term::Var(0)).is_some());
        // Non-constant control: refused BEFORE any term is built.
        let xx = ctx.function(smallvec![x], x);
        let body2 = ctx.function(smallvec![xx], x);
        let ty2 = ctx.forall(0, "X".into(), body2);
        assert!(atomize(&ctx, ty2, bty, Term::Var(0)).is_none());
    }

    #[test]
    fn polynomial_shape_requires_r_x_at_root_and_x_headed() {
        // §8's r_X is the ∀-body's FINAL spine — the polynomial IS
        // the whole type.  Two former false-accept classes, both must be
        // None now:
        // (a) interior Blue X-headed LCA — X buried in an argument, the
        //     type ENDS in C, not X: ∀X.(((X⇒X)⇒Int)⇒C);
        // (b) non-X-headed root: ∀X.(X⇒X⇒Int).
        // An Eq. 14 offer for either has no Lemma 8.2 license (Eq. 14's
        // In_B wrap needs the conclusion type's HEAD to be B).
        let mut ctx = TypeContext::new();
        let x = ctx.generic_param(0, "X".into());
        let c = ctx.generic_param(1, "C".into());
        let i = ctx.int(32, true);
        // (a): ((X⇒X)⇒Int)⇒C — X's LCA is the interior Blue spine.
        let xx = ctx.function(smallvec![x], x);
        let xxi = ctx.function(smallvec![xx], i);
        let body_a = ctx.function(smallvec![xxi], c);
        let ty_a = ctx.forall(0, "X".into(), body_a);
        assert!(
            polynomial_shape(&ctx, ty_a).is_none(),
            "(a) interior Blue X-headed LCA, type ends in C ⇒ not §8 shape"
        );
        // (b): (X⇒X⇒Int) — X̄ leaves' LCA is the root, head Int ≠ X.
        let xxi2 = ctx.function(smallvec![x, x], i);
        let ty_b = ctx.forall(0, "X".into(), xxi2);
        assert!(
            polynomial_shape(&ctx, ty_b).is_none(),
            "(b) non-X-headed root ⇒ not §8 shape"
        );
        // Control group (all four prior verdicts unchanged):
        let ix = ctx.function(smallvec![i], x);
        let body_ok = ctx.function(smallvec![ix], x); // (Int→X)→X
        let ty_ok = ctx.forall(0, "X".into(), body_ok);
        assert_eq!(
            polynomial_shape(&ctx, ty_ok).map(|s| s.constant),
            Some(true),
            "(Int→X)→X stays Some(true)"
        );
        let body_nonc = ctx.function(smallvec![xx], x); // (X→X)→X
        let ty_nonc = ctx.forall(0, "X".into(), body_nonc);
        assert_eq!(
            polynomial_shape(&ctx, ty_nonc).map(|s| s.constant),
            Some(false),
            "(X→X)→X stays Some(false)"
        );
        let xi = ctx.function(smallvec![x], i);
        let body_red = ctx.function(smallvec![xi], x); // (X⇒Int)⇒X
        let ty_red = ctx.forall(0, "X".into(), body_red);
        assert_eq!(
            polynomial_shape(&ctx, ty_red).map(|s| s.constant),
            None,
            "(X⇒Int)⇒X stays None (Red r_X)"
        );
        let body_rc = ctx.function(smallvec![xx], c); // (X→X)→C
        let ty_rc = ctx.forall(0, "X".into(), body_rc);
        assert_eq!(
            polynomial_shape(&ctx, ty_rc).map(|s| s.constant),
            None,
            "(X→X)→C stays None (now root-checked AND red-checked)"
        );
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

    #[test]
    fn atomize_forall_step_is_type_level() {
        // H: a leading ∀ step's mirror is Λ/TyApp (TYPE level), not λ —
        // the former bare steps-count built a λ mirror for it and the
        // term was ill-typed.  B-path: ∀Y.(Unit→Z) — steps
        // [Forall{Y=0}, Fn]; at(B) = Z.
        let mut ctx = TypeContext::new();
        let unit = ctx.unit();
        let z = ctx.generic_param(1, "Z".into());
        let unitfnz = ctx.function(smallvec![unit], z);
        let a = ctx.forall(0, "Y".into(), unitfnz);
        let dual = dual_context_pair(&ctx, a).expect("∀Y.(Unit→Z) ends in Z");
        assert_eq!(dual.step_count(), 2);
        assert_eq!(dual.steps, vec![StepKind::Forall { ty: 0 }, StepKind::Fn]);

        let atomized = atomize_instantiation(Term::Var(0), &dual, &[1]);
        // Shape: λy. ΛY. λz. base (λx. ((y x)@Y) z) — the ∀ step is TyLam
        // outermost, In_B is the PURE mirror (no trailing @Y z on the base
        // spine), and the z's flow into the handler's El forwarding only.
        // Two-pool: y sits at j+f = 2 (the ΛY step occupies no term slot;
        // only x and z do), base at k+f = 2.
        match &atomized {
            Term::Lam(y) => match y.as_ref() {
                Term::TyLam(tyb) => match tyb.as_ref() {
                    Term::Lam(zbody) => match zbody.as_ref() {
                        Term::App(base_app, handler) => {
                            assert!(
                                matches!(base_app.as_ref(), Term::Var(2)),
                                "base at k+f=2 (pure mirror): {base_app:?}"
                            );
                            match handler.as_ref() {
                                Term::Lam(hbody) => match hbody.as_ref() {
                                    Term::App(app_z, a_z) => {
                                        assert!(
                                            matches!(a_z.as_ref(), Term::Var(1)),
                                            "z forwarded: {a_z:?}"
                                        );
                                        match app_z.as_ref() {
                                            Term::TyApp(g, 0) => match g.as_ref() {
                                                Term::App(yv, a_x) => {
                                                    assert!(matches!(a_x.as_ref(), Term::Var(0)));
                                                    assert!(
                                                        matches!(yv.as_ref(), Term::Var(2)),
                                                        "handler head y at j+f=2"
                                                    );
                                                }
                                                other => panic!("y not applied: {other:?}"),
                                            },
                                            other => panic!("no @Y on y: {other:?}"),
                                        }
                                    }
                                    other => panic!("handler body not an application: {other:?}"),
                                },
                                other => panic!("handler not a λ: {other:?}"),
                            }
                        }
                        other => panic!("z-body not an application: {other:?}"),
                    },
                    other => panic!("no z layer: {other:?}"),
                },
                other => panic!("∀ step not TyLam: {other:?}"),
            },
            other => panic!("not a lambda: {other:?}"),
        }
        // βη-normal.
        assert_eq!(atomized.clone().normalize(), atomized);
    }

    #[test]
    fn atomize_in_is_pure_mirror_semantic_pin() {
        // Eq. 14 typing pin (P[X]=(A→X)→X, B=C→X₀, s=1, k=1, j=1):
        // the correct output is λy.λz. base(λx. y x z) — the free-theorem
        // instance ev_z(t_B y) = t_{X₀}(ev_z∘y) has NO trailing apply.
        // The former In_B pass emitted (…) z after the base spine.
        let mut ctx = TypeContext::new();
        let dual = DualContext {
            target: ctx.unit(),
            steps: vec![StepKind::Fn],
        };
        let t = atomize_instantiation(Term::Var(0), &dual, &[1]);
        match &t {
            Term::Lam(y) => match y.as_ref() {
                Term::Lam(z) => match z.as_ref() {
                    Term::App(f, h) => {
                        assert!(matches!(f.as_ref(), Term::Var(2)), "base at k+f=2: {f:?}");
                        assert!(
                            matches!(h.as_ref(), Term::Lam(_)),
                            "In_B is a PURE mirror: no z application at the top: {h:?}"
                        );
                    }
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn atomize_mixed_spine_z_indices() {
        // steps = [∀Y₇, Fn, ∀Y₈, Fn]: pins the Fn-count window formula,
        // the two-pool slot count (y at j+f=3, NOT unified-pool's j+n=5),
        // and the pure-mirror In (Λ₇.λz₁.Λ₈.λz₂. base H).
        let mut ctx = TypeContext::new();
        let dual = DualContext {
            target: ctx.unit(),
            steps: vec![
                StepKind::Forall { ty: 7 },
                StepKind::Fn,
                StepKind::Forall { ty: 8 },
                StepKind::Fn,
            ],
        };
        let t = atomize_instantiation(Term::Var(0), &dual, &[1]);
        match &t {
            Term::Lam(y) => match y.as_ref() {
                Term::TyLam(b7) => match b7.as_ref() {
                    Term::Lam(z1) => match z1.as_ref() {
                        Term::TyLam(b8) => match b8.as_ref() {
                            Term::Lam(z2) => match z2.as_ref() {
                                Term::App(f, h) => {
                                    assert!(matches!(f.as_ref(), Term::Var(3)), "base at k+f=3");
                                    // h = λx. (((y x)@7 z₁)@8) z₂
                                    match h.as_ref() {
                                        Term::Lam(xb) => match xb.as_ref() {
                                            Term::App(app8, a8) => {
                                                assert!(
                                                    matches!(a8.as_ref(), Term::Var(1)),
                                                    "z₂=Var(1)"
                                                );
                                                match app8.as_ref() {
                                                    Term::TyApp(g8, 8) => match g8.as_ref() {
                                                        Term::App(app7, a7) => {
                                                            assert!(
                                                                matches!(a7.as_ref(), Term::Var(2)),
                                                                "z₁=Var(2)"
                                                            );
                                                            match app7.as_ref() {
                                                                Term::TyApp(g7, 7) => {
                                                                    match g7.as_ref() {
                                                                        Term::App(yv, a_x) => {
                                                                            assert!(matches!(
                                                                                a_x.as_ref(),
                                                                                Term::Var(0)
                                                                            ));
                                                                            assert!(
                                                                                matches!(
                                                                                    yv.as_ref(),
                                                                                    Term::Var(3)
                                                                                ),
                                                                                "y at j+f=3 (two-pool), not j+n=5"
                                                                            );
                                                                        }
                                                                        other => {
                                                                            panic!("{other:?}")
                                                                        }
                                                                    }
                                                                }
                                                                other => {
                                                                    panic!("marker ≠ Y₇: {other:?}")
                                                                }
                                                            }
                                                        }
                                                        other => panic!("{other:?}"),
                                                    },
                                                    other => panic!("marker ≠ Y₈: {other:?}"),
                                                }
                                            }
                                            other => panic!("{other:?}"),
                                        },
                                        other => panic!("{other:?}"),
                                    }
                                }
                                other => panic!("{other:?}"),
                            },
                            other => panic!("{other:?}"),
                        },
                        other => panic!("{other:?}"),
                    },
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        let _ = t.clone().normalize();
    }
}
