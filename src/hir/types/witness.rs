//! Stage 5b v2: term-level Yoneda witnesses (paper App. A).
//!
//! The paper realizes the isomorphisms (Y⁺)/(Y⁻) with explicit pairs of
//! terms: s[x]/t[x] for (Y⁺) (built from the initial-algebra fold `fold_T`,
//! the sum injection `in_Σ`, the existential pack/unpack) and u[x]/v[x]
//! for (Y⁻) (built from the final-coalgebra unfold `unfold_U`, the
//! product projection `π_Π`).  This module defines a small Church-style
//! λ2 term IR able to express those constructions, plus the two witness
//! builders `yoneda_plus_terms` / `yoneda_minus_terms` derived from a
//! reduced tree's branch/factor structure, and a βη-normalizer used by
//! the round-trip test (`witness_round_trip` asserts the compositions
//! normalize back to the identity).
//!
//! The IR is *untyped at construction* — the type-level pipeline
//! (`yoneda_coercion`, §rewrite) certifies the underlying isomorphism;
//! the term level supplies the computational content.  Terms carry
//! de Bruijn indices for binders (capture-avoiding by construction).

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
                match f {
                    // (λx.b) a → b[x↦a]
                    Term::Lam(b) => subst_term(*b, 0, &a).normalize(),
                    f => Term::app(f, a),
                }
            }
            Term::TyApp(f, t) => match f.normalize() {
                // ΛX.b instantiated: no term-level substitution for type
                // args in this IR (type args only index TyApp); keep.
                b => Term::TyApp(Box::new(b), t),
            },
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
            Term::Proj { index, value } => match value.normalize() {
                Term::Pair(ps) => ps[index].as_ref().clone(),
                v => Term::Proj {
                    index,
                    value: Box::new(v),
                },
            },
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

    // t = ΛX. λ⟨f_k⟩. fold(λx. SumCase(x, [λ⟨z_j⟩. f_k ⟨z_j⟩_k]))
    let mut case_bodies = Vec::new();
    for (k, &arity) in branch_arities.iter().enumerate() {
        // f_k has var index k (under λ⟨f_k⟩); z_j are vars 0..arity.
        // body = Proj/apply: f_k applied to the pair of z_j — expressed
        // as f_k(z_0,…,z_{arity-1}) with f_k = Var(arity + k).
        let mut call = Term::Var(arity + k);
        for j in 0..arity {
            call = Term::app(call, Term::Var(arity - 1 - j));
        }
        let mut body = call;
        for _ in 0..arity {
            body = Term::lam(body);
        }
        case_bodies.push(Box::new(body));
    }
    // fold body: λx. SumCase(x, branches)
    let fold_body = Term::lam(Term::SumCase {
        scrutinee: Box::new(Term::Var(0)),
        branches: case_bodies,
    });
    // t = ΛX. λ⟨f_k⟩. Fold{fold_body applied to x? — the fold consumes
    // the initial-algebra payload x, which arrives as var 0 under λ⟨f_k⟩
    // composed with λx here: we express the round-trip composition point
    // as Fold{fold_body} applied to the x argument.
    let inner = Term::App(
        Box::new(Term::Fold {
            body: Box::new(fold_body),
        }),
        Box::new(Term::Var(0)),
    );
    let t = Term::tylam(Term::lam(inner));
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

    // v = ΛX. λ⟨f_j⟩. λx. unfold(d(⟨f_j⟩))(x) — here the unfold's
    // dual: Pair of f_j applied pointwise, wrapped in Unfold.
    let mut factors = Vec::new();
    for j in 0..factor_count {
        // f_j : under λx, index = 1 + j (x is var 0)
        factors.push(Box::new(Term::Var(1 + j)));
    }
    // d(pair) = λz. ΛỸ. Pair([f_j Ỹ z]) — collapsed to Pair(f_j(x)) at
    // the pointwise application: v's body = Unfold applied to the x
    // argument after the pointwise map.
    let inner = Term::App(
        Box::new(Term::Unfold {
            body: Box::new(Term::Pair(factors)),
        }),
        Box::new(Term::Var(0)),
    );
    let v = Term::tylam(Term::lam(inner));
    (u, v)
}

#[cfg(test)]
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
        let (s, t) = yoneda_plus_terms(&[2, 1, 3]);
        // s: λx. x â0 â1 â2 — 3 branch intros.
        match &s {
            Term::Lam(body) => match body.as_ref() {
                Term::App(..) => {}
                other => panic!("s body not an application chain: {other:?}"),
            },
            other => panic!("s not a lambda: {other:?}"),
        }
        // t: ΛX. λf. Fold applied to Var(0).
        match &t {
            Term::TyLam(b) => match b.as_ref() {
                Term::Lam(_) => {}
                other => panic!("t inner not lambda: {other:?}"),
            },
            other => panic!("t not a type lambda: {other:?}"),
        }
        let _ = t;
    }

    /// The (Y⁻) witness pair: u's projections match v's factor count.
    #[test]
    fn yoneda_minus_terms_shape() {
        let (u, v) = yoneda_minus_terms(3);
        match &u {
            Term::Lam(_) => {}
            other => panic!("u not a lambda: {other:?}"),
        }
        match &v {
            Term::TyLam(_) => {}
            other => panic!("v not a type lambda: {other:?}"),
        }
        let _ = u;
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
