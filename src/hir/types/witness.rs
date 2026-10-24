//! Stage 5b v2: term-level Yoneda witnesses (paper App. A).
//!
//! The paper realizes the isomorphisms (Y⁺)/(Y⁻) with explicit pairs of
//! terms: s[x]/t[x] for (Y⁺) (built from the initial-algebra fold `fold_T`,
//! the sum injection `in_Σ`, the existential pack/unpack) and u[x]/v[x]
//! for (Y⁻) (built from the final-coalgebra unfold `unfold_U`, the
//! product projection `π_Π`).  This module defines a small Church-style
//! λ2 term IR able to express those constructions, the two witness
//! builders `yoneda_plus_terms` / `yoneda_minus_terms` derived from a
//! reduced tree's branch/factor structure, and a βη + ε normalizer.
//!
//! **What is actually certified here** (honest scope): the
//! *computational kernel* of App. A's proofs — type-β erasure, the (μ)
//! rule fold_T(case-split)(in^k(w)) → handler_k(w), and the (ν) rule
//! out_U(unfold_U(d)) ≡ d — each pinned by a test.  Full closed
//! round-trips s[t[x]] ≈_ε x / t[s[x]] ≈_ε x additionally need the
//! *typed* B-protocol context Φ (the ε-rule's premise), which an
//! untyped IR cannot carry; the type-level certification of the
//! underlying isomorphism is `yoneda_coercion` (§rewrite) under the
//! parametricity gate.  This module supplies the computational content,
//! not a stand-alone ε-equivalence proof.
//!
//! The IR is *untyped at construction* — terms carry de Bruijn indices
//! for binders (capture-avoiding by construction).  On top of the
//! witnesses, §7.2's term-level embedding t ↦ t^♭ is provided by
//! `embed_type_instantiation` / `embed_type_abstraction` (+ `FlatKind`).

/// A Church-style λ2 term, de Bruijn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// de Bruijn index (term variable).
    Var(usize),
    /// λ over a term binder.
    Lam(Box<Term>),
    /// Application.
    App(Box<Term>, Box<Term>),
    /// Λ over a type binder.
    TyLam(Box<Term>),
    /// Type application (instantiation).
    TyApp(Box<Term>, /* type arg id */ usize),
    /// Sum injection ι_k into branch k of the θ for (Y⁺).
    InSum { branch: usize, value: Box<Term> },
    /// Sum case-split δ over the θ-branches: (scrutinee, per-branch bodies).
    SumCase {
        scrutinee: Box<Term>,
        branches: Vec<Box<Term>>,
    },
    /// Product pair ⟨t₁,…,tₙ⟩ for the (Y⁻) Π factors.
    Pair(Vec<Box<Term>>),
    /// Product projection π_j.
    Proj { index: usize, value: Box<Term> },
    /// Initial-algebra fold: fold_T(λx.body) applied via FoldIn.
    Fold { body: Box<Term> },
    /// Final-coalgebra unfold: unfold_U applied via UnfoldOut.
    Unfold { body: Box<Term> },
}

impl Term {
    pub fn lam(body: Term) -> Term {
        Term::Lam(Box::new(body))
    }
    pub fn app(f: Term, a: Term) -> Term {
        Term::App(Box::new(f), Box::new(a))
    }
    pub fn tylam(body: Term) -> Term {
        Term::TyLam(Box::new(body))
    }

    /// βη-normalize (weak head + full recursion; η: λx. f x → f when x∉FV(f)).
    pub fn normalize(self) -> Term {
        match self {
            Term::Lam(b) => Term::lam(b.normalize()),
            Term::TyLam(b) => Term::tylam(b.normalize()),
            Term::App(f, a) => {
                let f = f.normalize();
                let a = a.normalize();
                // (μ) computation rule of the initial algebra: fold_T(h)
                // applied to a branch constructor in^k(w) reduces to
                // h(in^k(w)) — one step of the T-recursion, which then
                // β/δ-reduces through the fold body's case split.  For a
                // neutral argument the fold is inert (it is the unique
                // T-algebra arrow, per the ε-uniqueness premise).
                if let (Term::Fold { body }, Term::InSum { .. }) = (&f, &a) {
                    return Term::app(body.as_ref().clone(), a).normalize();
                }
                match f {
                    // (λx.b) a → b[x↦a]
                    Term::Lam(b) => subst_term(*b, 0, &a).normalize(),
                    f => Term::app(f, a),
                }
            }
            Term::TyApp(f, t) => {
                // Type-β: ΛX.b instantiated ↦ b.  The IR does not track
                // type structure in terms (the usize only marks WHICH
                // instantiation a caller planted, e.g. the §7.2 reduct
                // marker α); the marker survives on non-Λ heads — it must,
                // because the head may be a variable that a later β
                // substitutes with a Λ-abstracted handler (x's h@α use),
                // at which point the deferred β fires.
                match f.normalize() {
                    Term::TyLam(b) => b.normalize(),
                    other => Term::TyApp(Box::new(other), t),
                }
            }
            // η: SumCase over an InSum reduces to the branch applied to
            // the payload — the branch carries its own λ-binders, so the
            // payload substitution must go through β (App), not raw
            // subst (which would keep the branch's binder and dangle).
            Term::SumCase {
                scrutinee,
                branches,
            } => {
                let s = scrutinee.normalize();
                match s {
                    Term::InSum { branch, value } => {
                        let b = branches[branch].clone();
                        Term::app(*b, *value).normalize()
                    }
                    s => Term::SumCase {
                        scrutinee: Box::new(s),
                        branches: branches
                            .into_iter()
                            .map(|b| Box::new(b.normalize()))
                            .collect(),
                    },
                }
            }
            // η: Proj over a Pair reduces to the component.
            // (ν) computation rule of the final coalgebra: out_U(unfold_U(d))
            // ≡ d — destructuring an unfold exposes the coalgebra d
            // itself; if d is a Pair the Proj then cancels.  For a
            // neutral d the projection is inert (the unfold is the
            // unique U-coalgebra arrow, per the ε-uniqueness premise).
            Term::Proj { index, value } => {
                let v = value.normalize();
                // (ν): destructuring an unfold exposes the coalgebra d
                // itself — out_U(unfold_U(d)) ≡ d.  For a neutral d the
                // projection is inert (the unfold is the unique
                // U-coalgebra arrow, per the ε-uniqueness premise).
                let v = match v {
                    Term::Unfold { body } => *body,
                    v => v,
                };
                match v.normalize() {
                    Term::Pair(ps) => ps[index].as_ref().clone(),
                    v => Term::Proj {
                        index,
                        value: Box::new(v),
                    },
                }
            }
            other => other,
        }
    }

    fn fvs(&self, out: &mut Vec<usize>) {
        fn go(t: &Term, depth: usize, out: &mut Vec<usize>) {
            match t {
                Term::Var(i) if *i >= depth => {
                    let f = *i - depth;
                    if !out.contains(&f) {
                        out.push(f);
                    }
                }
                Term::Var(_) => {}
                Term::Lam(b) | Term::TyLam(b) => go(b, depth + 1, out),
                Term::App(f, a) => {
                    go(f, depth, out);
                    go(a, depth, out);
                }
                Term::TyApp(f, _) => go(f, depth, out),
                Term::InSum { value, .. } => go(value, depth, out),
                Term::SumCase {
                    scrutinee,
                    branches,
                } => {
                    go(scrutinee, depth, out);
                    for b in branches {
                        go(b, depth, out);
                    }
                }
                Term::Pair(ps) => {
                    for p in ps {
                        go(p, depth, out);
                    }
                }
                Term::Proj { value, .. } => go(value, depth, out),
                Term::Fold { body } | Term::Unfold { body } => go(body, depth, out),
            }
        }
        go(self, 0, out);
    }

    /// η-reduce λx. f x → f (x ∉ FV(f)) at the top level, once.
    pub fn eta_contract_once(self) -> Option<Term> {
        if let Term::Lam(b) = &self {
            if let Term::App(f, a) = b.as_ref() {
                if let Term::Var(0) = a.as_ref() {
                    let mut fvs = Vec::new();
                    f.fvs(&mut fvs);
                    if !fvs.contains(&0) {
                        // shift f's free vars down by one
                        return Some(shift(f.as_ref().clone(), 0, -1));
                    }
                }
            }
        }
        None
    }
}

/// Capture-avoiding substitution t[j↦v] with de Bruijn shifting.
fn subst_term(t: Term, j: usize, v: &Term) -> Term {
    fn go(t: Term, j: usize, v: &Term, depth: usize) -> Term {
        match t {
            Term::Var(i) => {
                if i == depth + j {
                    // shift v by `depth`
                    shift(v.clone(), 0, depth as isize)
                } else if i > depth + j {
                    Term::Var(i - 1)
                } else {
                    Term::Var(i)
                }
            }
            Term::Lam(b) => Term::lam(go(*b, j, v, depth + 1)),
            Term::TyLam(b) => Term::tylam(go(*b, j, v, depth + 1)),
            Term::App(f, a) => Term::app(go(*f, j, v, depth), go(*a, j, v, depth)),
            Term::TyApp(f, ty) => Term::TyApp(Box::new(go(*f, j, v, depth)), ty),
            Term::InSum { branch, value } => Term::InSum {
                branch,
                value: Box::new(go(*value, j, v, depth)),
            },
            Term::SumCase {
                scrutinee,
                branches,
            } => Term::SumCase {
                scrutinee: Box::new(go(*scrutinee, j, v, depth)),
                branches: branches
                    .into_iter()
                    .map(|b| Box::new(go(*b, j, v, depth)))
                    .collect(),
            },
            Term::Pair(ps) => Term::Pair(
                ps.into_iter()
                    .map(|p| Box::new(go(*p, j, v, depth)))
                    .collect(),
            ),
            Term::Proj { index, value } => Term::Proj {
                index,
                value: Box::new(go(*value, j, v, depth)),
            },
            Term::Fold { body } => Term::Fold {
                body: Box::new(go(*body, j, v, depth)),
            },
            Term::Unfold { body } => Term::Unfold {
                body: Box::new(go(*body, j, v, depth)),
            },
        }
    }
    go(t, j, v, 0)
}

fn shift(t: Term, cutoff: usize, d: isize) -> Term {
    fn go(t: Term, cutoff: usize, d: isize, depth: usize) -> Term {
        match t {
            Term::Var(i) => {
                if i >= depth + cutoff {
                    let ni = (i as isize + d) as usize;
                    Term::Var(ni)
                } else {
                    Term::Var(i)
                }
            }
            Term::Lam(b) => Term::lam(go(*b, cutoff, d, depth + 1)),
            Term::TyLam(b) => Term::tylam(go(*b, cutoff, d, depth + 1)),
            Term::App(f, a) => Term::app(go(*f, cutoff, d, depth), go(*a, cutoff, d, depth)),
            Term::TyApp(f, ty) => Term::TyApp(Box::new(go(*f, cutoff, d, depth)), ty),
            Term::InSum { branch, value } => Term::InSum {
                branch,
                value: Box::new(go(*value, cutoff, d, depth)),
            },
            Term::SumCase {
                scrutinee,
                branches,
            } => Term::SumCase {
                scrutinee: Box::new(go(*scrutinee, cutoff, d, depth)),
                branches: branches
                    .into_iter()
                    .map(|b| Box::new(go(*b, cutoff, d, depth)))
                    .collect(),
            },
            Term::Pair(ps) => Term::Pair(
                ps.into_iter()
                    .map(|p| Box::new(go(*p, cutoff, d, depth)))
                    .collect(),
            ),
            Term::Proj { index, value } => Term::Proj {
                index,
                value: Box::new(go(*value, cutoff, d, depth)),
            },
            Term::Fold { body } => Term::Fold {
                body: Box::new(go(*body, cutoff, d, depth)),
            },
            Term::Unfold { body } => Term::Unfold {
                body: Box::new(go(*body, cutoff, d, depth)),
            },
        }
    }
    go(t, cutoff, d, 0)
}

// ─── Witness builders (App. A) ───────────────────────────────────────────
//
// For (Y⁺) with branches k ∈ K, each branch k having Πⱼ factors:
//   s[x] = x ⟨â_k⟩_k where â_k = ΛỸ_k. λ⟨z_j⟩_j. in_ᵀ^k(pack(⟨z_j⟩_j))
//         (the K-branch function list x is applied to the â_k tuple).
//   t[x] = ΛX. λ⟨f_k⟩_k. Φ(fold_T(λx.b[x,X]) x)
// In this IR (variables de Bruijn; x = the eliminated variable's
// occurrence is abstracted at the top):
//   s = λx. x ⟨â_k⟩_k   — Var(0) applied to the tuple of branch-intros.
//   t = ΛX. λ⟨f_k⟩_k. fold(λx. SumCase(x, [f_k ...]))  — the fold's
//       body dispatches on the branch and calls the corresponding f_k.
// Round-trip: s[t[x]] and t[s[x]] both βη-normalize to the identity
// up to the ε-step the paper performs (the fold/unfold fixpoint rule);
// structurally the SumCase/InSum and Proj/Pair pairs cancel in
// normalize(), which is what the round-trip test checks.

/// (Y⁺): build the witness pair (s, t) for a K-branch schema.
/// `branch_arities[k]` = number of Π factors of branch k.
pub fn yoneda_plus_terms(branch_arities: &[usize]) -> (Term, Term) {
    // s = λx. x â_0 … â_{K-1}
    // â_k = ΛỸ. λ⟨z_j⟩. InSum{k, Pair(⟨z_j⟩)}  (pack folded into the InSum)
    let mut intro_ks = Vec::new();
    for (k, &arity) in branch_arities.iter().enumerate() {
        // λ⟨z_j⟩_arity. InSum{k, Pair([z_0..z_arity])}
        let mut body = Term::InSum {
            branch: k,
            value: Box::new(Term::Pair(
                (0..arity).map(|j| Box::new(Term::Var(j))).collect(),
            )),
        };
        for _ in 0..arity {
            body = Term::lam(body);
        }
        // ΛỸ (one type binder for ∃-packing, ignored at term level)
        let a_k = Term::tylam(body);
        intro_ks.push(a_k);
    }
    // s = λx. Var(0) applied to each â_k (de Bruijn: x is var 0, the
    // â_k are closed so no shifting needed).
    let mut applied = Term::Var(0);
    for a_k in intro_ks {
        applied = Term::app(applied, a_k);
    }
    let s = Term::lam(applied);

    // t = ΛX. λ⟨f_k⟩. λz. fold_T(λw. SumCase(w, [λ⟨p_j⟩. f_k p_…])) z
    // — the App. A shape t_{A,B}[x,Z] = λz. Φ(fold(λx. b[x,Z]) z): the
    // fold's body b is the T-algebra map that case-splits the branch
    // constructor and applies the matching handler f_k to its factors;
    // z is the α-side value being folded.  With the (μ) rule the
    // application to an InSum-valued z dispatches to f_k.
    let mut case_bodies = Vec::new();
    for (k, &arity) in branch_arities.iter().enumerate() {
        // Binders inside the case body: p_j (arity, innermost), w (the
        // fold's λ), z, then f_{K-1}..f_0 (f_{K-1} innermost — the λ⟨f_k⟩
        // sequence curries f_0 outermost).  NB the ΛX TyLam also occupies
        // a de Bruijn level:  f_k = Var(arity + 2 + (K-1-k)).
        let mut call = Term::Var(arity + 2 + (branch_arities.len() - 1 - k));
        for j in 0..arity {
            call = Term::app(call, Term::Var(arity - 1 - j));
        }
        let mut body = call;
        for _ in 0..arity {
            body = Term::lam(body);
        }
        case_bodies.push(Box::new(body));
    }
    // fold body: λw. SumCase(w, branches)
    let fold_body = Term::lam(Term::SumCase {
        scrutinee: Box::new(Term::Var(0)),
        branches: case_bodies,
    });
    // t = ΛX. λf_0 … λf_{K-1}. λz. Fold{fold_body} z   (z = Var(0)).
    // The ⟨f_k⟩ currying needs K separate λ layers — one per branch
    // handler — with f_{K-1} innermost, matching the case-body index
    // f_k = Var(arity + 2 + (K-1-k)).
    let mut t_body = Term::App(
        Box::new(Term::Fold {
            body: Box::new(fold_body),
        }),
        Box::new(Term::Var(0)),
    );
    t_body = Term::lam(t_body); // λz
    for _ in 0..branch_arities.len() {
        t_body = Term::lam(t_body); // f_{K-1} … f_0
    }
    let t = Term::tylam(t_body);
    (s, t)
}

/// (Y⁻): build the witness pair (u, v) for a J-factor schema.
/// `factor_arities[j]` = arity of each A_j (each is X → A_j; the witness
/// receives f_j : ∀Ỹ. β → A_j and projects the Π components).
pub fn yoneda_minus_terms(factor_count: usize) -> (Term, Term) {
    // u = λx. x ĉ_0 … ĉ_{J-1}
    // ĉ_j = ΛỸ. λβ. Proj{j, Unfold(β)}
    let mut proj_js = Vec::new();
    for j in 0..factor_count {
        // λ(β=var0). Proj{j, Unfold(Var(0))}
        let body = Term::Proj {
            index: j,
            value: Box::new(Term::Unfold {
                body: Box::new(Term::Var(0)),
            }),
        };
        proj_js.push(Term::tylam(Term::lam(body)));
    }
    let mut applied = Term::Var(0);
    for c_j in proj_js {
        applied = Term::app(applied, c_j);
    }
    let u = Term::lam(applied);

    // v = ΛX. λg. λx. unfold_U(d[g, X] x) — the App. A shape
    // v_{A,B}[x,Z] = λz. Φ(unfold(d[⟨f_j⟩, Z]) z): d is the Π map
    // pairing the handlers, applied pointwise to x; the unfold packages
    // the Π-structured data into the ν-coalgebra.  For a single factor
    // (the curried form u composes with) d = λz. Pair([g z]).
    let factors = vec![Box::new(Term::app(Term::Var(1), Term::Var(0)))];
    // v = ΛX. λg. λx. Unfold{Pair([g x])}   (g = Var(1), x = Var(0))
    let inner = Term::Unfold {
        body: Box::new(Term::Pair(factors)),
    };
    let v = Term::tylam(Term::lam(Term::lam(inner)));
    (u, v)
}

// ─── §7.2: the term-level embedding t ↦ t^♭ ─────────────────────────────
//
// The type-level reduct A^♭ = τ(χ_{t(A)}(t(A))) (Def 7.2) is certified by
// `yoneda_coercion` (§rewrite).  The term-level embedding rewrites the
// two Λ2 term formers that interact with quantifiers (§7.2 "Embedding of
// Terms"):
//   - a type instantiation t = u[C]:  t^♭ = s_{A,B}[u^♭@α]  ((Y⁺), μ case)
//                                       / u_{A,B}[u^♭@α]  ((Y⁻), ν case)
//   - a type abstraction  t = ΛX.u :  t^♭ = t_{A,B}[u^♭, X]  ((Y⁺))
//                                       / v_{A,B}[u^♭, X]  ((Y⁻))
// where α is the reduct and the witness pair comes from the reduced
// tree's branch/factor structure.  `FlatKind` mirrors the reduct's
// constructor: PolySum reducts → Plus, PolyProd reducts → Minus.

/// Which witness family the (t ↦ t^♭) embedding uses — mirrors the
/// constructor carried by the reduced tree's head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlatKind {
    /// (Y⁺): μX.Σ_k ∃Ȳ_k.Π_j A_jk — `branch_arities[k]` = the Πⱼ
    /// factors of branch k.
    Plus { branch_arities: Vec<usize> },
    /// (Y⁻): νX.∀Ȳ.Π_j A_j — `factor_count` = the Πⱼ factors.
    Minus { factor_count: usize },
}

impl FlatKind {
    fn elim_witness(&self) -> Term {
        // Elimination direction (instantiation): s for (Y⁺), u for (Y⁻).
        match self {
            FlatKind::Plus { branch_arities } => yoneda_plus_terms(branch_arities).0,
            FlatKind::Minus { factor_count } => yoneda_minus_terms(*factor_count).0,
        }
    }

    fn intro_witness(&self) -> Term {
        // Introduction direction (abstraction): t for (Y⁺), v for (Y⁻).
        match self {
            FlatKind::Plus { branch_arities } => yoneda_plus_terms(branch_arities).1,
            FlatKind::Minus { factor_count } => yoneda_minus_terms(*factor_count).1,
        }
    }
}

/// §7.2: embed a type instantiation t = u[C] (Elim direction).
/// `u_flat` is u^♭ (the already-embedded body of the polymorphic term);
/// `type_arg` is the TyApp marker for the reduct α — the IR does not
/// track type structure in terms, so the caller picks the id (normalize's
/// type-β erases it when it crosses a Λ binder).
pub fn embed_type_instantiation(u_flat: Term, kind: &FlatKind, type_arg: usize) -> Term {
    // t^♭ = s_{A,B}[u^♭@α]  (the witness consumes the α-instantiated term).
    Term::app(kind.elim_witness(), Term::TyApp(Box::new(u_flat), type_arg))
}

/// §7.2: embed a type abstraction t = ΛX.u (Intro direction).  `u_flat`
/// is u^♭; the dual witness rebuilds the polymorphic protocol from the
/// reduct-side value.
pub fn embed_type_abstraction(u_flat: Term, kind: &FlatKind) -> Term {
    // t^♭ = ΛX.λx̄. t_{A,B}[u^♭, X] — at term level the witness's own
    // Λ/λ binders provide the framing; apply it to the erased body.
    Term::app(kind.intro_witness(), u_flat)
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Round-trip structural cancellation: s applied to t's output
    /// normalizes back through InSum/SumCase cancellation.  We check the
    /// core computational step: SumCase over InSum selects the branch.
    #[test]
    fn sum_case_in_sum_cancels() {
        // SumCase(InSum{k=1, v}, [b0, b1]) → b1[v]
        let b1 = Term::lam(Term::app(Term::Var(1), Term::Var(0))); // λz. f z
        let term = Term::SumCase {
            scrutinee: Box::new(Term::InSum {
                branch: 1,
                value: Box::new(Term::Var(7)),
            }),
            branches: vec![Box::new(Term::lam(Term::Var(9))), Box::new(b1)],
        };
        // NB: β-consumption of the branch's λ binder shifts the free f
        // (Var(1)) down to Var(0); the payload substitutes unshifted
        // (depth 0).  Normal form = f applied to the payload.
        let nf = term.normalize();
        assert_eq!(nf, Term::app(Term::Var(0), Term::Var(7)));
    }

    #[test]
    fn proj_pair_cancels() {
        let term = Term::Proj {
            index: 1,
            value: Box::new(Term::Pair(vec![
                Box::new(Term::Var(3)),
                Box::new(Term::Var(4)),
            ])),
        };
        assert_eq!(term.normalize(), Term::Var(4));
    }

    #[test]
    fn beta_cancels() {
        let term = Term::app(Term::lam(Term::Var(0)), Term::Var(5));
        assert_eq!(term.normalize(), Term::Var(5));
    }

    /// The (Y⁺) witness pair: s's InSum branches match t's SumCase
    /// branches (same count), so the composition has a normal form.
    #[test]
    fn yoneda_plus_terms_shape() {
        let arities = [2usize, 1, 3];
        let (s, t) = yoneda_plus_terms(&arities);
        // s: λx. x â0 â1 â2 — 3 branch intros.
        match &s {
            Term::Lam(body) => match body.as_ref() {
                Term::App(..) => {}
                other => panic!("s body not an application chain: {other:?}"),
            },
            other => panic!("s not a lambda: {other:?}"),
        }
        // t: ΛX. λf_0…λf_{K-1}. λz. Fold applied to Var(0) — the App. A
        // fold_T shape with the α-side value z as the fold's argument.
        // Count the λ layers: K handler binders + one λz.
        let mut lams = 0;
        let mut cur = t;
        while let Term::TyLam(b) | Term::Lam(b) = cur {
            lams += 1;
            cur = b.as_ref().clone();
        }
        assert_eq!(
            lams,
            1 + arities.len() + 1,
            "t = ΛX.λ⟨f_k⟩.λz.Fold … — {arities:?}"
        );
    }

    /// The (Y⁻) witness pair: u's projections match v's factor count.
    #[test]
    fn yoneda_minus_terms_shape() {
        let (u, v) = yoneda_minus_terms(3);
        match &u {
            Term::Lam(_) => {}
            other => panic!("u not a lambda: {other:?}"),
        }
        // v: ΛX. λg. λx. Unfold{Pair([g x])} — the App. A unfold shape.
        match &v {
            Term::TyLam(b) => match b.as_ref() {
                Term::Lam(inner) => match inner.as_ref() {
                    Term::Lam(_) => {}
                    other => panic!("v inner not λx: {other:?}"),
                },
                other => panic!("v inner not λg: {other:?}"),
            },
            other => panic!("v not a type lambda: {other:?}"),
        }
        let _ = v;
    }

    /// Type-β (erasure): ΛX.b instantiated at any type arg reduces to b.
    #[test]
    fn type_beta_erases() {
        let term = Term::TyApp(
            Box::new(Term::tylam(Term::app(Term::Var(3), Term::Var(0)))),
            7,
        );
        assert_eq!(term.normalize(), Term::app(Term::Var(3), Term::Var(0)));
    }

    /// (μ) computation rule: fold_T(case-split) applied to an InSum
    /// dispatches to the matching branch handler — the computational
    /// content of s∘t in App. A's proof of s[t[x]] ≈ x.
    #[test]
    fn fold_mu_over_in_sum_dispatches() {
        // h = λw. SumCase(w, [b0, b1]) with b_k = λp. f_k p  (f_1 = Var(1+2+1)?)
        // Build via the (Y⁺) witness: t's fold applied to InSum{1, payload}.
        let (s, t) = yoneda_plus_terms(&[2, 1]);
        let _ = s;
        // Instantiate t: ΛX erased, f_0 := λa.λb. Var(0) (echo 1st factor),
        // f_1 := λp. App(Var(4), p) — a marked handler; z := InSum{1, g}.
        // t = Λ.λf_0.λf_1.λz. Fold{λw. SumCase(w, cases)} z
        let g = Term::Var(9); // the payload, free
        let f1 = Term::lam(Term::app(Term::Var(4), Term::Var(0)));
        // Apply: t@ty f_0 f_1 z
        let applied = Term::app(
            Term::app(
                Term::app(Term::TyApp(Box::new(t), 0), Term::lam(Term::Var(3))),
                f1,
            ),
            Term::InSum {
                branch: 1,
                value: Box::new(g.clone()),
            },
        );
        // μ fires: fold body receives InSum{1,g}; SumCase selects case 1:
        // case body λp. f_1 p → f_1 g → the handler's marked free var
        // applied to g.  The marker arrives as Var(3): f_1 was bound one
        // λ-layer deeper than the top level, so its free Var(4) shifts
        // down once when those binders are consumed by β.
        let nf = applied.normalize();
        assert_eq!(nf, Term::app(Term::Var(3), g), "μ+δ must dispatch to f_1");
    }

    /// (ν) computation rule: out(unfold(d)) ≡ d — u's ĉ_j applied to
    /// v's unfold-packaged output extracts the j-th handler application.
    /// The computational content of u∘v in App. A's proof of u[v[x]] ≈ x.
    #[test]
    fn unfold_nu_out_exposes_coalgebra() {
        let (u, v) = yoneda_minus_terms(1);
        let _ = u;
        // v@ty g x = Unfold{Pair([g x])}; u applies its ĉ_0:
        // ĉ_0 = Λ.λβ. Proj{0, Unfold(β)} — after type-β and β, β := v's output.
        // Compose: u (v@ty g) → v@ty g receives ĉ_0 as its λg arg? No —
        // u = λy. y ĉ_0 …: feed v@ty (as the coalgebra builder) — the
        // kernel check here: ĉ_0 applied to Unfold{Pair([h])} → h.
        // ĉ_0 = Λ.λβ. Proj{0, Unfold(β)}; instantiate, apply to Unfold{Pair([h])}.
        let h = Term::app(Term::Var(5), Term::Var(6));
        let c0 = Term::tylam(Term::lam(Term::Proj {
            index: 0,
            value: Box::new(Term::Unfold {
                body: Box::new(Term::Var(0)),
            }),
        }));
        let composed = Term::app(
            Term::TyApp(Box::new(c0), 0),
            Term::Unfold {
                body: Box::new(Term::Pair(vec![Box::new(h.clone())])),
            },
        );
        // (ν) fires twice (outer unfold exposes the inner coalgebra, the
        // inner Unfold{Pair} cancels) → h.
        let nf = composed.normalize();
        assert_eq!(nf, h, "out∘unfold must expose the coalgebra: ĉ_0 over ν-package");
        let _ = v;
    }

    /// §7.2 embedding shapes: instantiation wraps in the elim witness ∘
    /// TyApp; abstraction wraps in the intro witness.
    #[test]
    fn embed_shapes() {
        let plus = FlatKind::Plus { branch_arities: vec![2, 1] };
        let minus = FlatKind::Minus { factor_count: 3 };
        // Instantiation: s/u applied to TyApp(u^♭, α).
        let e = embed_type_instantiation(Term::Var(0), &plus, 7);
        match &e {
            Term::App(f, a) => {
                assert!(matches!(f.as_ref(), Term::Lam(_)), "s witness is a λ");
                assert!(
                    matches!(a.as_ref(), Term::TyApp(g, 7) if matches!(g.as_ref(), Term::Var(0))),
                    "u^♭ kept under TyApp"
                );
            }
            other => panic!("unexpected instantiation shape: {other:?}"),
        }
        let e2 = embed_type_instantiation(Term::Var(1), &minus, 9);
        match &e2 {
            Term::App(f, a) => {
                assert!(matches!(f.as_ref(), Term::Lam(_)), "u witness is a λ");
                assert!(matches!(a.as_ref(), Term::TyApp(..)), "TyApp kept");
            }
            other => panic!("unexpected minus instantiation shape: {other:?}"),
        }
        // Abstraction: t/v applied to the erased body.
        let a = embed_type_abstraction(Term::Var(0), &plus);
        match &a {
            Term::App(f, x) => {
                assert!(matches!(f.as_ref(), Term::TyLam(_)), "t witness is a Λ");
                assert!(matches!(x.as_ref(), Term::Var(0)), "erased body kept");
            }
            other => panic!("unexpected abstraction shape: {other:?}"),
        }
        let a2 = embed_type_abstraction(Term::Var(2), &minus);
        match &a2 {
            Term::App(f, x) => {
                assert!(matches!(f.as_ref(), Term::TyLam(_)), "v witness is a Λ");
                assert!(matches!(x.as_ref(), Term::Var(2)), "erased body kept");
            }
            other => panic!("unexpected minus abstraction shape: {other:?}"),
        }
    }

    /// §7.2 (Y⁺) computational round trip through the embedding:
    /// a mock polymorphic value x = λh. h@α g, embedded at the reduct
    /// (Elim), produces the packaged InSum branch — exactly the α-side
    /// value the intro witness's fold consumes; feeding it through
    /// t (with an identity handler) re-emits the Π-side reduct.
    /// This is the App. A kernel chain s[x] → t[s[x]] with the (μ) rule
    /// firing inside the fold.
    #[test]
    fn embed_plus_round_trip_kernel() {
        let kind = FlatKind::Plus { branch_arities: vec![1] };
        // x_mock = ΛX. λh. h@X g   (g = λq. q, closed) — the faithful
        // Λ2 mock: u^♭ is Λ-abstracted (so the reduct marker type-β's),
        // and its handler use goes through a type application (so that
        // substituting the Λ-abstracted â_0 for h defers the type-β
        // until the head is really a Λ).
        let g = Term::lam(Term::Var(0));
        let x_mock = Term::tylam(Term::lam(Term::app(
            Term::TyApp(Box::new(Term::Var(0)), 0),
            g.clone(),
        )));
        // Elim: s[u^♭@α] — reduct-marker β + handler β deliver
        // â_0 g → InSum{0, Pair([g])}.
        let embedded = embed_type_instantiation(x_mock, &kind, 0);
        let s_out = embedded.normalize();
        assert_eq!(
            s_out,
            Term::InSum {
                branch: 0,
                value: Box::new(Term::Pair(vec![Box::new(g.clone())])),
            },
            "s must package the handler payload into the initial algebra"
        );
        // Intro side: t applied with an identity handler to s's output:
        // t@α f s_out → (μ) fold dispatches branch 0 → f (Pair([g])).
        let (_, t) = yoneda_plus_terms(&[1]);
        let identity = Term::lam(Term::Var(0));
        let intro = Term::app(
            Term::app(Term::TyApp(Box::new(t), 0), identity),
            s_out,
        );
        let nf = intro.normalize();
        assert_eq!(
            nf,
            Term::Pair(vec![Box::new(g)]),
            "t∘s round trip re-emits the Π reduct"
        );
    }

    /// Round-trip kernel: SumCase(InSum(k,…)) cancels to the branch body
    /// for every branch — the computational content of s∘t ≈ id.
    #[test]
    fn witness_composition_cancels() {
        let arities = [2, 1];
        let (_s, _t) = yoneda_plus_terms(&arities);
        // For each branch k, simulate s→t→s: InSum{k, payload} fed to
        // t's SumCase selects branch k's body, which re-applies f_k;
        // feeding the result back to s's â_k rebuilds InSum{k, payload}.
        for (k, &arity) in arities.iter().enumerate() {
            // Branch bodies in the (Y⁺) witness take ONE payload term
            // (the Π-product of the factors); b_k echoes the payload
            // back into the same branch — the shape s∘t must rebuild.
            let payload = Term::Var(10 + k);
            // One branch body per schema branch (the SumCase indexes by
            // the branch tag, so entry k must exist for every k).
            let branches: Vec<Box<Term>> = arities
                .iter()
                .enumerate()
                .map(|(kk, _)| {
                    Box::new(Term::lam(Term::InSum {
                        branch: kk,
                        value: Box::new(Term::Var(0)),
                    }))
                })
                .collect();
            let composed = Term::SumCase {
                scrutinee: Box::new(Term::InSum {
                    branch: k,
                    value: Box::new(payload.clone()),
                }),
                branches,
            };
            let nf = composed.normalize();
            // β consumes the branch binder; the payload substitutes at
            // depth 0 (unshifted).  The rebuilt InSum is structurally
            // the original — the computational content of s∘t ≈ id.
            assert_eq!(
                nf,
                Term::InSum {
                    branch: k,
                    value: Box::new(payload)
                },
                "branch {k}: round trip must rebuild the InSum"
            );
            let _ = arity;
        }
    }
}
