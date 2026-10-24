//! Stage 5b v3: typed witness IR — carries the typed B-protocol context Φ
//! (the ε-rule premise) so that the witness round trips s[t[x]] ≈_ε x /
//! t[s[x]] ≈_ε x close *fully*: every reduction step is type-checked
//! (fail-closed), and Φ is definitionally expanded over the polynomial
//! fragment, reducing the ε-rule to βη-congruence of that expansion plus
//! the (μ)/(ν) rules.
//!
//! ## Why a typed IR is the missing piece
//!
//! In App. A, Φⁿ_Ajk is the functor action of the branch type A_jk on a
//! morphism — it can only be *built* by recursing over A_jk's type
//! structure (`A(X) = Int ⇒ X` gives `Φ(f) = λz. f ∘ z`).  An untyped IR
//! cannot carry that structure, which is why the v2 kernel test stopped
//! at the computational kernel.  Here every node is stamped with a
//! `TypeId` and `typed_normalize` re-derives the type at each step:
//! a step whose type cannot be derived aborts with `None` — the B-protocol
//! premise is *checked*, not assumed.
//!
//! ## The ε-rule, definitionally
//!
//! The central ε-equivalence of App. A (`Φ(fold(h))` commuting with the
//! fold) becomes, under the definitional Φ expansion, the typed (μ) rule:
//!
//! ```text
//! fold(h)(in^k(w))  →  h( Φ_{branch_k}(fold(h))(w) )
//! ```
//!
//! where `Φ_{branch_k}` is `functor_action` over the branch's type and
//! `fold(h)` is the recursive call.  The premise checked at each firing:
//! the carrier's body must be (canonically) the coproduct of the branch
//! types — exactly the initial-algebra typing of the reduct τ(E).
//!
//! ## Honest scope (v1 fragment)
//!
//! - `functor_action` supports: direct param, const (param-free) types,
//!   tuples (pointwise), and `Fn(P, A')` with the param occurring only in
//!   the codomain (post-composition `g ∘ z`).  μ/ν/∃/∀-shaped functors
//!   are fail-closed `None` for now.
//! - Type equality uses structural `TypeId` equality with the canonical
//!   βη fallback (`neg_beta_eta_equiv`).
//!
//! ## Gate (Stage 4)
//!
//! This layer is ε-theory-strength: every `normalize` step that fires the
//! typed (μ) rule relies on the ε-rule premise (the Φ-algebra morphism).
//! Any PRODUCTION consumer must therefore consult
//! `TypeContext::parametricity` first — a compilation unit that reifies
//! types at runtime (`dyn Trait` dispatch) must refuse ε-strength
//! judgments, fail-closed.  The internal `ty_eq` uses the UNGATED
//! canonical βη equivalence deliberately: βη-strength equality is sound
//! regardless of parametricity; only the ε-rule firings themselves need
//! the gate.  (The layer is currently standalone — no production call
//! sites — so the gate obligation applies to the first wiring.)

use super::*;

/// A typed term: node + its `TypeId`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TTerm {
    pub kind: TKind,
    pub ty: TypeId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TKind {
    Var(usize),
    /// λ over a term binder of type `binder`.
    Lam { binder: TypeId, body: Box<TTerm> },
    App(Box<TTerm>, Box<TTerm>),
    /// Λ over the type binder `param` (a `GenericParam` index).
    TyLam { param: usize, body: Box<TTerm> },
    /// Type application with the caller-supplied instantiated result type.
    /// `typed_normalize` verifies `result ≡ subst(f's ∀-body, param ↦ arg)`
    /// — the reduct marker of the §7.2 embedding is checked, not trusted.
    TyApp { f: Box<TTerm>, arg: TypeId, result: TypeId },
    /// Sum injection into branch k.  The stamped `ty` is EITHER the
    /// T-view `Σ(branches)` or the μ-view (the carrier whose body is
    /// that coproduct — the constructor application `in^k(w)`); both are
    /// verified against the branches at normalization.
    InSum { branch: usize, value: Box<TTerm>, branches: Vec<TypeId> },
    SumCase { scrutinee: Box<TTerm>, branches: Vec<Box<TTerm>>, result: TypeId },
    Pair(Vec<Box<TTerm>>),
    Proj { index: usize, value: Box<TTerm> },
    /// Initial-algebra fold: `body : F(carrier) → carrier`, `ty = carrier`.
    /// `carrier` must be a `Mu`.
    Fold { body: Box<TTerm>, carrier: TypeId },
    /// Final-coalgebra unfold: `body : carrier`, `ty = carrier`'s body.
    /// `carrier` must be a `Nu`.
    Unfold { body: Box<TTerm>, carrier: TypeId },
}

fn mk(kind: TKind, ty: TypeId) -> TTerm {
    TTerm { kind, ty }
}

/// Structural type equality with the canonical βη fallback.
fn ty_eq(ctx: &mut TypeContext, a: TypeId, b: TypeId) -> bool {
    a == b || ctx.neg_beta_eta_equiv(a, b)
}

/// Does `param` occur in `ty`?  Binder-aware: a `Mu`/`Nu`/`Forall`/`Exists`
/// whose own `param_index` equals `param` shadows its body.
fn mentions(ctx: &TypeContext, ty: TypeId, param: usize) -> bool {
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved) {
        TypeData::GenericParam { index, .. } => *index == param,
        TypeData::Fn { params, ret } => {
            params.iter().any(|&p| mentions(ctx, p, param)) || mentions(ctx, *ret, param)
        }
        TypeData::Tuple { elems } => elems.iter().any(|&e| mentions(ctx, e, param)),
        TypeData::Coproduct { alternatives } => {
            alternatives.iter().any(|&a| mentions(ctx, a, param))
        }
        TypeData::Mu { param_index, body, .. } | TypeData::Nu { param_index, body, .. } => {
            *param_index != param && mentions(ctx, *body, param)
        }
        TypeData::Forall { param_index, body, .. } => {
            *param_index != param && mentions(ctx, *body, param)
        }
        TypeData::Exists { param_index, base, .. } => {
            *param_index != param && mentions(ctx, *base, param)
        }
        _ => false,
    }
}

/// Φ^param_A(g): the functor action of `A` (polynomial in `param`) on the
/// morphism `g : src → dst`, as a definitional βη expansion.  The result
/// has type `A[src] → A[dst]`.
///
/// Fragment (v1): direct param (`Φ = g`), const types (`Φ = id_A`),
/// tuples (pointwise), and `Fn(P, A')` with `param` only in the codomain
/// (post-composition `A(g)(z) = g ∘ z`).  Everything else — including
/// μ/ν/∃/∀-shaped functors — is fail-closed `None`.
pub fn functor_action(
    ctx: &mut TypeContext,
    a: TypeId,
    param: usize,
    g: TTerm,
) -> Option<TTerm> {
    // g must be a single-param morphism src → dst.
    let (src, dst) = match ctx.get(g.ty) {
        TypeData::Fn { params, ret } if params.len() == 1 => (params[0], *ret),
        _ => return None,
    };
    if !mentions(ctx, a, param) {
        // Const functor: the morphism map is the identity at A.
        return Some(mk(
            TKind::Lam { binder: a, body: Box::new(mk(TKind::Var(0), a)) },
            ctx.function(smallvec![a], a),
        ));
    }
    let resolved = ctx.resolve_binding(a);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::GenericParam { index, .. } if index == param => {
            // Φ = g itself; A[src]→A[dst] = src→dst = g's type.
            Some(g)
        }
        TypeData::Tuple { elems } => {
            // Pointwise: λp. Pair([Φ_{a_j}(g) (π_j p)]_j) — varying
            // components are mapped, const components pass through.
            let a_src = ctx.subst(a, &Subst::from_single(param, src));
            let a_dst = ctx.subst(a, &Subst::from_single(param, dst));
            let mut parts: Vec<Box<TTerm>> = Vec::new();
            for (j, &a_j) in elems.iter().enumerate() {
                let proj_j = mk(
                    TKind::Proj { index: j, value: Box::new(mk(TKind::Var(0), a_src)) },
                    ctx.subst(a_j, &Subst::from_single(param, src)),
                );
                if mentions(ctx, a_j, param) {
                    let phi_j = functor_action(ctx, a_j, param, g.clone())?;
                    let applied_ty = match ctx.get(phi_j.ty) {
                        TypeData::Fn { params, ret } if params.len() == 1 => {
                            debug_assert!(ty_eq_static(params[0], proj_j.ty));
                            *ret
                        }
                        _ => return None,
                    };
                    parts.push(Box::new(mk(
                        TKind::App(Box::new(phi_j), Box::new(proj_j)),
                        applied_ty,
                    )));
                } else {
                    parts.push(Box::new(proj_j));
                }
            }
            let pair = mk(TKind::Pair(parts), a_dst);
            Some(mk(
                TKind::Lam { binder: a_src, body: Box::new(pair) },
                ctx.function(smallvec![a_src], a_dst),
            ))
        }
        TypeData::Fn { params, ret } => {
            // Covariant fragment: param only in the codomain.
            if params.iter().any(|&p| mentions(ctx, p, param)) {
                return None; // negative/contravariant occurrence: v2
            }
            if !mentions(ctx, ret, param) {
                return None; // handled by the const path above
            }
            // A(g)(z) = g ∘ z :  λz:a_src. λī:params. g (z i₁ … iₙ)
            // (post-composition: the functor's action on the morphism g).
            let a_src = ctx.subst(a, &Subst::from_single(param, src));
            let a_dst = ctx.subst(a, &Subst::from_single(param, dst));
            let n = params.len();
            // Binder order outermost→innermost: z, i_0, …, i_{n-1}; the
            // body sits under n+1 binders, so the embedded g shifts by n+1.
            let g_shifted = tshift(g.clone(), 0, (n + 1) as isize);
            // g (z i₁ … iₙ) — z applied to the spine args yields src,
            // which g consumes.  Left-associated application chain:
            // call = z i₁ … iₙ, then body = g call.
            let mut call = mk(TKind::Var(n), a_src); // z : A[src]
            for j in 0..n {
                let call_ty = call.ty;
                call = mk(
                    TKind::App(Box::new(call), Box::new(mk(TKind::Var(n - 1 - j), params[j]))),
                    {
                        // each step's type: the remaining function type of z
                        match ctx.get(call_ty) {
                            TypeData::Fn { ret, .. } => *ret,
                            _ => return None,
                        }
                    },
                );
            }
            let body = mk(TKind::App(Box::new(g_shifted), Box::new(call)), dst);
            // Nest binders innermost-first: i_{n-1} … i_0, then z.
            let mut term = body;
            let mut cur_ty = dst;
            for j in (0..n).rev() {
                let layer_ty = ctx.function(smallvec![params[j]], cur_ty);
                term = mk(TKind::Lam { binder: params[j], body: Box::new(term) }, layer_ty);
                cur_ty = layer_ty;
            }
            term = mk(TKind::Lam { binder: a_src, body: Box::new(term) }, ctx.function(smallvec![a_src], a_dst));
            Some(term)
        }
        _ => None, // μ/ν/∃/∀-shaped functors: v2 (fail-closed)
    }
}

/// Static (context-free) structural equality — used only for debug_asserts.
fn ty_eq_static(a: TypeId, b: TypeId) -> bool {
    a == b
}

/// de Bruijn substitution for typed terms.  Node types are absolute
/// (`GenericParam` indices belong to TYPE binders, not term binders), so
/// only `Var` indices move.
fn tsubst(t: TTerm, j: usize, v: &TTerm, depth: usize) -> TTerm {
    match t.kind {
        TKind::Var(i) => {
            if i == depth + j {
                tshift(v.clone(), 0, depth as isize)
            } else if i > depth + j {
                mk(TKind::Var(i - 1), t.ty)
            } else {
                t
            }
        }
        TKind::Lam { binder, body } => mk(
            TKind::Lam { binder, body: Box::new(tsubst(*body, j, v, depth + 1)) },
            t.ty,
        ),
        TKind::App(f, a) => mk(
            TKind::App(Box::new(tsubst(*f, j, v, depth)), Box::new(tsubst(*a, j, v, depth))),
            t.ty,
        ),
        TKind::TyLam { param, body } => mk(
            TKind::TyLam { param, body: Box::new(tsubst(*body, j, v, depth)) },
            t.ty,
        ),
        TKind::TyApp { f, arg, result } => mk(
            TKind::TyApp { f: Box::new(tsubst(*f, j, v, depth)), arg, result },
            t.ty,
        ),
        TKind::InSum { branch, value, branches } => mk(
            TKind::InSum { branch, value: Box::new(tsubst(*value, j, v, depth)), branches },
            t.ty,
        ),
        TKind::SumCase { scrutinee, branches, result } => mk(
            TKind::SumCase {
                scrutinee: Box::new(tsubst(*scrutinee, j, v, depth)),
                branches: branches.into_iter().map(|b| Box::new(tsubst(*b, j, v, depth))).collect(),
                result,
            },
            t.ty,
        ),
        TKind::Pair(ps) => mk(
            TKind::Pair(ps.into_iter().map(|p| Box::new(tsubst(*p, j, v, depth))).collect()),
            t.ty,
        ),
        TKind::Proj { index, value } => mk(
            TKind::Proj { index, value: Box::new(tsubst(*value, j, v, depth)) },
            t.ty,
        ),
        TKind::Fold { body, carrier } => mk(
            TKind::Fold { body: Box::new(tsubst(*body, j, v, depth)), carrier },
            t.ty,
        ),
        TKind::Unfold { body, carrier } => mk(
            TKind::Unfold { body: Box::new(tsubst(*body, j, v, depth)), carrier },
            t.ty,
        ),
    }
}

fn tshift(t: TTerm, cutoff: usize, d: isize) -> TTerm {
    fn go(t: TTerm, cutoff: usize, d: isize, depth: usize) -> TTerm {
        let kind = match t.kind {
            TKind::Var(i) => {
                if i >= depth + cutoff {
                    TKind::Var((i as isize + d) as usize)
                } else {
                    TKind::Var(i)
                }
            }
            TKind::Lam { binder, body } => {
                TKind::Lam { binder, body: Box::new(go(*body, cutoff, d, depth + 1)) }
            }
            TKind::App(f, a) => TKind::App(
                Box::new(go(*f, cutoff, d, depth)),
                Box::new(go(*a, cutoff, d, depth)),
            ),
            TKind::TyLam { param, body } => {
                TKind::TyLam { param, body: Box::new(go(*body, cutoff, d, depth)) }
            }
            TKind::TyApp { f, arg, result } => TKind::TyApp {
                f: Box::new(go(*f, cutoff, d, depth)),
                arg,
                result,
            },
            TKind::InSum { branch, value, branches } => TKind::InSum {
                branch,
                value: Box::new(go(*value, cutoff, d, depth)),
                branches,
            },
            TKind::SumCase { scrutinee, branches, result } => TKind::SumCase {
                scrutinee: Box::new(go(*scrutinee, cutoff, d, depth)),
                branches: branches
                    .into_iter()
                    .map(|b| Box::new(go(*b, cutoff, d, depth)))
                    .collect(),
                result,
            },
            TKind::Pair(ps) => TKind::Pair(
                ps.into_iter().map(|p| Box::new(go(*p, cutoff, d, depth))).collect(),
            ),
            TKind::Proj { index, value } => {
                TKind::Proj { index, value: Box::new(go(*value, cutoff, d, depth)) }
            }
            TKind::Fold { body, carrier } => {
                TKind::Fold { body: Box::new(go(*body, cutoff, d, depth)), carrier }
            }
            TKind::Unfold { body, carrier } => {
                TKind::Unfold { body: Box::new(go(*body, cutoff, d, depth)), carrier }
            }
        };
        mk(kind, t.ty)
    }
    go(t, cutoff, d, 0)
}

impl TTerm {
    /// Typed βηε-normalization: every step re-derives its type; a step
    /// whose type cannot be derived aborts with `None` (fail-closed) —
    /// this is the B-protocol context Φ being *checked* at each rewrite.
    pub fn normalize(&self, ctx: &mut TypeContext) -> Option<TTerm> {
        let mut env: Vec<TypeId> = Vec::new();
        self.go(ctx, &mut env)
    }

    fn go(&self, ctx: &mut TypeContext, env: &mut Vec<TypeId>) -> Option<TTerm> {
        match &self.kind {
            TKind::Var(i) => {
                let ty = *env.get(env.len().checked_sub(1)?.checked_sub(*i)?)?;
                Some(mk(TKind::Var(*i), ty))
            }
            TKind::Lam { binder, body } => {
                env.push(*binder);
                let body = body.go(ctx, env);
                env.pop();
                let body = body?;
                let body_ty = body.ty;
                Some(mk(
                    TKind::Lam { binder: *binder, body: Box::new(body) },
                    ctx.function(smallvec![*binder], body_ty),
                ))
            }
            TKind::App(f, a) => {
                let f = f.go(ctx, env)?;
                let a = a.go(ctx, env)?;
                // (μ) rule with the definitional Φ expansion: the fold
                // consuming a constructor application commutes with the
                // functor action on the branch payload.
                if let (
                    TTerm { kind: TKind::Fold { body: h, carrier }, .. },
                    TTerm { kind: TKind::InSum { branch, value, branches }, .. },
                ) = (&f, &a)
                {
                    let carrier_data = ctx.get(ctx.resolve_binding(*carrier)).clone();
                    let (carrier_param, carrier_body, is_mu) = match carrier_data {
                        TypeData::Mu { param_index, body, .. } => (param_index, body, true),
                        TypeData::Nu { param_index, body, .. } => (param_index, body, false),
                        _ => return None, // fold/unfold at a non-fixpoint: fail-closed
                    };
                    let _ = is_mu;
                    // Premise: the carrier's body is the coproduct of the
                    // branch types (initial-algebra typing of τ(E)).
                    let sigma = ctx.coproduct(branches.clone().into());
                    if !ty_eq(ctx, carrier_body, sigma) {
                        return None;
                    }
                    // The recursive call: fold(h) as the morphism
                    // μ → carrier (its codomain is the algebra's target,
                    // checked ≡ carrier at the Fold node).
                    let fold_fn_ty = ctx.function(smallvec![*carrier], *carrier);
                    let fold_term = mk(
                        TKind::Fold { body: h.clone(), carrier: *carrier },
                        fold_fn_ty,
                    );
                    // Φ_{branch_k}(fold(h))(w) — the definitional ε-step.
                    let fmap = functor_action(ctx, branches[*branch], carrier_param, fold_term)?;
                    let fmap_ty = fmap.ty;
                    let fmap_ty_data = ctx.get(fmap_ty).clone();
                    let applied = mk(
                        TKind::App(Box::new(fmap), Box::new((**value).clone())),
                        match fmap_ty_data {
                            TypeData::Fn { params, .. } => {
                                if !ty_eq(ctx, params[0], value.ty) {
                                    return None;
                                }
                                params[0]
                            }
                            _ => return None,
                        },
                    );
                    // h receives the mapped payload; result type = carrier.
                    let h_dom = match ctx.get(h.ty).clone() {
                        TypeData::Fn { params, ret } => {
                            if !ty_eq(ctx, params[0], applied.ty) {
                                return None;
                            }
                            ret
                        }
                        _ => return None,
                    };
                    if !ty_eq(ctx, h_dom, *carrier) {
                        return None;
                    }
                    return mk(
                        TKind::App(Box::new((**h).clone()), Box::new(applied)),
                        *carrier,
                    )
                    .go(ctx, env);
                }
                match &f.kind {
                    TKind::Lam { binder, body } => {
                        if !ty_eq(ctx, *binder, a.ty) {
                            return None;
                        }
                        tsubst(body.as_ref().clone(), 0, &a, 0).go(ctx, env)
                    }
                    _ => {
                        let f_data = ctx.get(f.ty).clone();
                        let ret = match f_data {
                            TypeData::Fn { params, ret } if params.len() == 1 => {
                                if !ty_eq(ctx, params[0], a.ty) {
                                    return None;
                                }
                                ret
                            }
                            _ => return None,
                        };
                        Some(mk(TKind::App(Box::new(f), Box::new(a)), ret))
                    }
                }
            }
            TKind::TyLam { param, body } => {
                let body = body.go(ctx, env)?;
                let body_ty = body.ty;
                Some(mk(
                    TKind::TyLam { param: *param, body: Box::new(body) },
                    ctx.forall(*param, Symbol::intern("X"), body_ty),
                ))
            }
            TKind::TyApp { f, arg, result } => {
                let f = f.go(ctx, env)?;
                // Verify the caller-supplied result against the real
                // substitution — the reduct marker is checked.
                let (param_index, forall_body) = match ctx.get(f.ty) {
                    TypeData::Forall { param_index, body, .. } => (*param_index, *body),
                    _ => return None,
                };
                let expected = ctx.subst(forall_body, &Subst::from_single(param_index, *arg));
                if !ty_eq(ctx, expected, *result) {
                    return None;
                }
                Some(mk(
                    TKind::TyApp { f: Box::new(f), arg: *arg, result: *result },
                    *result,
                ))
            }
            TKind::InSum { branch, value, branches } => {
                let value = value.go(ctx, env)?;
                if !ty_eq(ctx, value.ty, branches[*branch]) {
                    return None;
                }
                // The stamp must be one of the two views: Σ(branches) or
                // a fixpoint whose body is that coproduct.
                let sigma = ctx.coproduct(branches.clone().into());
                let stamp_ok = ty_eq(ctx, self.ty, sigma)
                    || match ctx.get(ctx.resolve_binding(self.ty)) {
                        TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                            ty_eq(ctx, *body, sigma)
                        }
                        _ => false,
                    };
                if !stamp_ok {
                    return None;
                }
                Some(mk(
                    TKind::InSum { branch: *branch, value: Box::new(value), branches: branches.clone() },
                    self.ty,
                ))
            }
            TKind::SumCase { scrutinee, branches, result } => {
                let s = scrutinee.go(ctx, env)?;
                let alts = match ctx.get(s.ty) {
                    TypeData::Coproduct { alternatives } => alternatives.clone(),
                    _ => return None,
                };
                if alts.len() != branches.len() {
                    return None;
                }
                let mut out = Vec::new();
                for b in branches {
                    let b = b.go(ctx, env)?;
                    out.push(Box::new(b));
                }
                Some(mk(
                    TKind::SumCase { scrutinee: Box::new(s), branches: out, result: *result },
                    *result,
                ))
            }
            TKind::Pair(ps) => {
                let mut out = Vec::new();
                for p in ps {
                    out.push(Box::new(p.go(ctx, env)?));
                }
                let tys: SmallVec<[TypeId; 4]> = out.iter().map(|p| p.ty).collect();
                let ty = match tys.len() {
                    0 => ctx.unit(),
                    1 => tys[0],
                    _ => ctx.tuple(tys),
                };
                Some(mk(TKind::Pair(out), ty))
            }
            TKind::Proj { index, value } => {
                let v = value.go(ctx, env)?;
                // (ν) rule: out(unfold(d)) ≡ d — destructuring an unfold
                // exposes the coalgebra; if it is a Pair the projection
                // then cancels.
                let v = match &v.kind {
                    TKind::Unfold { body, carrier } => {
                        let d = body.go(ctx, env)?;
                        // carrier must be a Nu whose body types d.
                        let nu_body = match ctx.get(ctx.resolve_binding(*carrier)) {
                            TypeData::Nu { body, .. } => *body,
                            _ => return None,
                        };
                        if !ty_eq(ctx, d.ty, nu_body) {
                            return None;
                        }
                        d
                    }
                    _ => v,
                };
                match &v.kind {
                    TKind::Pair(ps) => {
                        let p = ps.get(*index)?;
                        Some((**p).clone())
                    }
                    _ => {
                        let elems = match ctx.get(v.ty) {
                            TypeData::Tuple { elems } => elems.clone(),
                            _ => return None,
                        };
                        let ty = *elems.get(*index)?;
                        Some(mk(
                            TKind::Proj { index: *index, value: Box::new(v) },
                            ty,
                        ))
                    }
                }
            }
            TKind::Fold { body, carrier } => {
                let b = body.go(ctx, env)?;
                // carrier must be a Mu (initial algebra).
                let carrier_body = match ctx.get(ctx.resolve_binding(*carrier)) {
                    TypeData::Mu { body, .. } => *body,
                    _ => return None,
                };
                let (dom, ret) = match ctx.get(b.ty) {
                    TypeData::Fn { params, ret } if params.len() == 1 => (params[0], *ret),
                    _ => return None,
                };
                // The algebra's codomain is the carrier; its domain is
                // (canonically) the carrier's body — the functor at the
                // fixpoint.  This IS the τ(E) typing premise.
                if !ty_eq(ctx, ret, *carrier) || !ty_eq(ctx, dom, carrier_body) {
                    return None;
                }
                Some(mk(
                    TKind::Fold { body: Box::new(b), carrier: *carrier },
                    *carrier,
                ))
            }
            TKind::Unfold { body, carrier } => {
                let b = body.go(ctx, env)?;
                let nu_body = match ctx.get(ctx.resolve_binding(*carrier)) {
                    TypeData::Nu { body, .. } => *body,
                    _ => return None,
                };
                if !ty_eq(ctx, b.ty, *carrier) {
                    return None;
                }
                Some(mk(
                    TKind::Unfold { body: Box::new(b), carrier: *carrier },
                    nu_body,
                ))
            }
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// The fully typed (Y⁺) kernel round trip on the const-payload
    /// instance A(X) = Int: α = μX.Σ(Int), the identity algebra
    /// f = in : Int → α.  fold(f)(in(g)) → h(Φ_{Int}(fold(f))(g)) —
    /// Φ over the const branch is the identity map, so the chain
    /// collapses back to in(g).  EVERY step is type-checked; the
    /// closed result is the initial-algebra equation in(g) ≈ fold(in)(in(g))
    /// with the B-protocol context (carrier body ≡ Σ(branches)) verified
    /// at the (μ) firing.
    #[test]
    fn const_payload_round_trip_closes_typed() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        // α = μX.Σ(Int)  (T = const Int — the branch type never varies)
        let sigma_int = ctx.coproduct(smallvec![int]);
        let alpha = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: Symbol::intern("X"),
            body: sigma_int,
        });
        // The whole chain abstracted over the payload g : Int so the
        // free variable has an environment entry.
        // f = λw:Int. in^0(w)  — the identity algebra : Int → α
        let f = mk(
            TKind::Lam {
                binder: int,
                body: Box::new(mk(
                    TKind::InSum { branch: 0, value: Box::new(mk(TKind::Var(0), int)), branches: vec![int] },
                    alpha,
                )),
            },
            ctx.function(smallvec![int], alpha),
        );
        // in(g) : α  (constructor application, μ-view stamp)
        let in_g = mk(
            TKind::InSum { branch: 0, value: Box::new(mk(TKind::Var(0), int)), branches: vec![int] },
            alpha,
        );
        // fold(f)(in(g))
        let fold_f = mk(
            TKind::Fold { body: Box::new(f), carrier: alpha },
            alpha,
        );
        let chain = mk(
            TKind::App(Box::new(fold_f), Box::new(in_g)),
            alpha,
        );
        let body = mk(
            TKind::Lam { binder: int, body: Box::new(chain) },
            ctx.function(smallvec![int], alpha),
        );
        let nf = body.normalize(&mut ctx).expect("typed round trip must close");
        // The normal form's body must be in(g) again — the initial-algebra
        // equation, fully type-checked.
        match &nf.kind {
            TKind::Lam { body, .. } => {
                assert_eq!(body.ty, alpha, "closed chain has type α = μX.Σ(Int)");
                match &body.kind {
                    TKind::InSum { branch: 0, value, branches } => {
                        assert_eq!(branches.len(), 1);
                        assert_eq!(value.ty, int, "payload is the Int argument");
                        assert!(matches!(value.kind, TKind::Var(0)), "payload is g");
                    }
                    other => panic!("round trip must land on in(g), got {other:?}"),
                }
            }
            other => panic!("expected the λw wrapper, got {other:?}"),
        }
    }

    /// Φ over a covariant Fn functor: A(X) = Int ⇒ X gives
    /// Φ(g) = λz. g ∘ z : (Int⇒src) → (Int⇒dst) — the definitional
    /// expansion the ε-rule's Φ-unrolling is made of.
    #[test]
    fn functor_action_fn_covariant_expands() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let src = ctx.generic_param(1, Symbol::intern("S"));
        let dst = ctx.generic_param(2, Symbol::intern("D"));
        // g : src → dst (abstract, bound by an outer λ in the environment)
        let g_ty = ctx.function(smallvec![src], dst);
        let g = mk(TKind::Var(0), g_ty);
        let gp_x = ctx.generic_param(0, Symbol::intern("X"));
        let a = ctx.function(smallvec![int], gp_x);
        let phi = functor_action(&mut ctx, a, 0, g).expect("Fn-covariant is in the fragment");
        let int_src = ctx.function(smallvec![int], src);
        let int_dst = ctx.function(smallvec![int], dst);
        let expected_ty = ctx.function(smallvec![int_src], int_dst);
        assert_eq!(phi.ty, expected_ty, "Φ : (Int⇒src) → (Int⇒dst)");
        // Term shape: λz:Int⇒src. λi:Int. g (z i)
        match &phi.kind {
            TKind::Lam { binder, body } => {
                assert_eq!(*binder, ctx.function(smallvec![int], src));
                match &body.kind {
                    TKind::Lam { binder: b2, body: inner } => {
                        assert_eq!(*b2, int);
                        match &inner.kind {
                            TKind::App(f, a) => {
                                assert!(matches!(f.kind, TKind::Var(2)), "g is Var(2) under two binders");
                                match &a.kind {
                                    TKind::App(z, i) => {
                                        assert!(matches!(z.kind, TKind::Var(1)), "z is Var(1)");
                                        assert!(matches!(i.kind, TKind::Var(0)), "i is Var(0)");
                                    }
                                    other => panic!("z applied to i, got {other:?}"),
                                }
                            }
                            other => panic!("g applied, got {other:?}"),
                        }
                    }
                    other => panic!("λi binder, got {other:?}"),
                }
            }
            other => panic!("λz binder, got {other:?}"),
        }
        // The expanded Φ type-checks under normalization.  g is abstract
        // (Var(0) shifted to Var(2) under Φ's two new binders), so wrap
        // the whole thing in λg:g_ty to give the free variable an
        // environment entry; the normal form is the term itself.
        let wrapped = mk(
            TKind::Lam { binder: g_ty, body: Box::new(phi) },
            ctx.function(smallvec![g_ty], expected_ty),
        );
        let nf = wrapped.normalize(&mut ctx).expect("Φ expansion normalizes");
        assert_eq!(nf.ty, ctx.function(smallvec![g_ty], expected_ty));
        match &nf.kind {
            TKind::Lam { body, .. } => assert_eq!(body.ty, expected_ty, "Φ body type kept"),
            other => panic!("expected λg wrapper, got {other:?}"),
        }
    }

    /// Fail-closed: the (μ) rule refuses to fire when the carrier's body
    /// is not (canonically) the coproduct of the branch types — the
    /// B-protocol premise is checked, not assumed.
    #[test]
    fn mu_rule_rejects_carrier_mismatch() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let boolean = ctx.bool();
        // carrier = μX.Σ(Int) but the InSum claims branch type Bool.
        let sigma_int = ctx.coproduct(smallvec![int]);
        let alpha = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: Symbol::intern("X"),
            body: sigma_int,
        });
        let f = mk(
            TKind::Lam {
                binder: boolean,
                body: Box::new(mk(
                    TKind::InSum { branch: 0, value: Box::new(mk(TKind::Var(0), boolean)), branches: vec![boolean] },
                    alpha,
                )),
            },
            ctx.function(smallvec![boolean], alpha),
        );
        let in_g = mk(
            TKind::InSum { branch: 0, value: Box::new(mk(TKind::Var(0), boolean)), branches: vec![boolean] },
            alpha,
        );
        let chain = mk(
            TKind::App(
                Box::new(mk(TKind::Fold { body: Box::new(f), carrier: alpha }, alpha)),
                Box::new(in_g),
            ),
            alpha,
        );
        let wrapped = mk(
            TKind::Lam { binder: boolean, body: Box::new(chain) },
            ctx.function(smallvec![boolean], alpha),
        );
        assert!(
            wrapped.normalize(&mut ctx).is_none(),
            "carrier body Σ(Int) ≠ Σ(Bool): the (μ) rule must fail closed"
        );
    }

    /// Fail-closed: TyApp's reduct marker is verified against the real
    /// substitution — a lying marker is rejected.
    #[test]
    fn tyapp_marker_is_checked() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let boolean = ctx.bool();
        let gp = ctx.generic_param(0, Symbol::intern("X"));
        let id_fn_ty = ctx.function(smallvec![gp], gp);
        // f = ΛX. λx:X. x : ∀X. X → X
        let f = mk(
            TKind::TyLam {
                param: 0,
                body: Box::new(mk(
                    TKind::Lam { binder: gp, body: Box::new(mk(TKind::Var(0), gp)) },
                    id_fn_ty,
                )),
            },
            ctx.forall(0, Symbol::intern("X"), id_fn_ty),
        );
        // Honest instantiation at Int: result = Int → Int.
        let int_int = ctx.function(smallvec![int], int);
        let honest = mk(
            TKind::TyApp {
                f: Box::new(f.clone()),
                arg: int,
                result: int_int,
            },
            int_int,
        );
        assert!(honest.normalize(&mut ctx).is_some(), "honest marker passes");
        // Lying marker: claims the instantiation at Bool while arg = Int.
        let liar = mk(
            TKind::TyApp {
                f: Box::new(f),
                arg: int,
                result: ctx.function(smallvec![boolean], boolean),
            },
            ctx.function(smallvec![boolean], boolean),
        );
        assert!(
            liar.normalize(&mut ctx).is_none(),
            "reduct marker must be checked: subst(∀X.X→X, X↦Int) ≠ Bool→Bool"
        );
    }
}
