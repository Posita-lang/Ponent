//! Stage 5d: polarized functor action + recursive payload closure.
//!
//! θ(X) = {µX.}Σₖ∃Z⃗ₖ.Πⱼ D_jk[X⁺] (blue, (Y⁺)) / {νX.}∀Z⃗ₖ.Πⱼ E_k[X⁻] (red, (Y⁻))
//! is substituted into each branch payload P[X] — the recursive payload
//! instance, e.g. the `Int⇒X` branch of a fold. The substitution is
//! FUNCTORIAL: it threads occurrence polarity, because X may sit in a
//! negative (contravariant) position (`X⇒A`) where the covariant image
//! does not apply. The separately-tested Fn expansion (params/ret unwrap)
//! is preserved; what this stage adds is the contravariant image plumbing
//! and the fold-termination discipline.
//!
//! ## Integration note (evaluated against the yoneda.rs reducer)
//!
//! NOT wired into the default reducer path (`yoneda.rs` μ/ν assembly,
//! ~`yoneda_reducer_once`): the reducer's fixpoint assembly already
//! produces the paper-correct *compact* form `µX.Σₖ Bₖ[X]` — X free in
//! the body is bound by the µ (the fixpoint equation's RHS), exactly
//! θ(X)'s D_jk[X⁺] residues — and golden tests pin that shape (e.g.
//! `∀X.(∀Z.(Z,X)⇒X)⇒X → µX.∃Z.(Z,X)`).  Auto-closing there would emit
//! the unfolded form `µX.Σₖ Bₖ[µ…]`, breaking the pins.  The compact and
//! unfolded forms are ε-equivalent (µX.F = F[µX.F]), so
//! `close_recursive_payloads` is an OPT-IN transformer for consumers that
//! need the payload instantiated at the whole fold (functor-payload
//! materialization for a typed-witness Φ, fold-structure-sensitive κ
//! counting).  Covariant-only pre-5d call sites keep working through
//! `functor_action_pos`.

use super::*;

/// Polarity of the position an occurrence sits in — the `TypeId`-level
/// mirror of coherence.rs's colors (Blue = X⁺ = positive, Red = X̄ = negative).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Covariant position (X⁺): the covariant image applies.
    Pos,
    /// Contravariant position (X̄): only the contravariant image applies.
    Neg,
}

impl Polarity {
    /// Compose polarities through a constructor position — the same
    /// multiplication table as `Variance::xform` (invariant slots are
    /// refused explicitly, see the `Ptr` / `&mut` arms).
    pub fn compose(self, position: Polarity) -> Polarity {
        match (self, position) {
            (Polarity::Pos, Polarity::Pos) => Polarity::Pos,
            (Polarity::Pos, Polarity::Neg) => Polarity::Neg,
            (Polarity::Neg, Polarity::Pos) => Polarity::Neg,
            (Polarity::Neg, Polarity::Neg) => Polarity::Pos,
        }
    }
}

/// P3e: BASE of the size-scaled fuel budget for `functor_action` (see
/// `functor_action_fuel`); the binder-shadow rule is the real
/// termination argument — the fuel is only the fail-closed backstop.
const FUNCTOR_ACTION_FUEL: u32 = 100_000;

/// P3e: size-scaled fuel budget: `max(BASE, 10  node_count)` — a budget
/// proportional to the payload's type-graph size tightens the fail-
/// closed window on large types (a fixed constant could be exhausted by
/// a large-but-fine type, or be too generous for a small one).  The
/// count is iterative with a visited set (types are DAGs through the
/// interner — the same TypeId recurs); `visit_type_children` covers all
/// TypeId-child positions.
fn functor_action_fuel<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> u32 {
    let mut seen: std::collections::HashSet<TypeId> = std::collections::HashSet::new();
    let mut stack = vec![ty];
    let mut count = 0usize;
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        count += 1;
        visit_type_children(t, ctx, &mut |c| {
            if !seen.contains(&c) {
                stack.push(c);
            }
        });
    }
    FUNCTOR_ACTION_FUEL.max((10 * count) as u32)
}

/// Does `ty` mention param index `idx` FREE (binder shadowing respected)?
fn mentions_param_free<'input>(ctx: &TypeContext<'input>, ty: TypeId, idx: usize) -> bool {
    fn go<'input>(
        ctx: &TypeContext<'input>,
        ty: TypeId,
        idx: usize,
        bound: &mut Vec<usize>,
    ) -> bool {
        let data = ctx.get_arc(ty);
        match &*data {
            TypeData::GenericParam { index, .. } => *index == idx && !bound.contains(index),
            TypeData::Forall { param_index, body, .. }
            | TypeData::Exists { param_index, base: body, .. }
            | TypeData::Mu { param_index, body, .. }
            | TypeData::Nu { param_index, body, .. } => {
                bound.push(*param_index);
                let found = go(ctx, *body, idx, bound);
                bound.pop();
                found
            }
            TypeData::Poly { quantifiers, body } => {
                let n = quantifiers.len();
                for (qi, _) in quantifiers {
                    bound.push(*qi);
                }
                let found = go(ctx, *body, idx, bound);
                for _ in 0..n {
                    bound.pop();
                }
                found
            }
            _ => {
                let mut found = false;
                visit_type_children(ty, ctx, &mut |c| {
                    if !found {
                        found = go(ctx, c, idx, bound);
                    }
                });
                found
            }
        }
    }
    go(ctx, ty, idx, &mut Vec::new())
}

/// Capture guard: refuse to insert an image under a binder whose index
/// occurs free in the image. (Param indices are globally unique via
/// `fresh_param_index`, so this defends against hand-built types.)
fn capture_guard<'input>(
    ctx: &TypeContext<'input>,
    binder: usize,
    image_pos: TypeId,
    image_neg: Option<TypeId>,
) -> Option<()> {
    if mentions_param_free(ctx, image_pos, binder) {
        return None;
    }
    if let Some(neg) = image_neg {
        if mentions_param_free(ctx, neg, binder) {
            return None;
        }
    }
    Some(())
}

/// Polarized functorial action: rewrite every FREE occurrence of the
/// variable `x` inside `payload` into the image for its polarity.
///
/// - `image_pos`: covariant image (the Σ/µ reduct of Fig. 6), used at X⁺.
/// - `image_neg`: contravariant image, used at X̄ (function parameters).
///   `None` ⇒ fail-closed on a negative occurrence (a covariant-only
///   reduct has no contravariant action; the (Y⁻) dual θ(X̄) supplies one).
///
/// # Termination under fold (the recursive payload case)
///
/// Any binder whose `param_index == x` SHADOWS the substitution: the map
/// returns that subtree unchanged. This is what lets a recursive payload
/// `Int⇒X` be closed against `µX.Σₖ Bₖ[X]` in ONE pass: the inserted
/// image is opaque (never re-entered), the fold binder is opaque, and
/// the (possibly vacuous) outer binder is NOT dissolved here — Fig. 4c
/// elision is deferred to canonicalization, since dissolving it would
/// re-open the recursion and diverge.
pub fn functor_action<'input>(
    ctx: &mut TypeContext<'input>,
    payload: TypeId,
    x: usize,
    image_pos: TypeId,
    image_neg: Option<TypeId>,
) -> Option<TypeId> {
    // Fast path: no free occurrence ⇒ identity.
    if !mentions_param_free(ctx, payload, x) {
        return Some(payload);
    }
    // P3e: budget scales with the payload's type-graph size.
    let mut fuel = functor_action_fuel(ctx, payload);
    functor_action_go(ctx, payload, x, image_pos, image_neg, Polarity::Pos, &mut fuel)
}

/// Covariant-only wrapper preserving the pre-Stage-5d call sites.
pub fn functor_action_pos<'input>(
    ctx: &mut TypeContext<'input>,
    payload: TypeId,
    x: usize,
    image: TypeId,
) -> Option<TypeId> {
    functor_action(ctx, payload, x, image, None)
}

fn functor_action_go<'input>(
    ctx: &mut TypeContext<'input>,
    ty: TypeId,
    x: usize,
    image_pos: TypeId,
    image_neg: Option<TypeId>,
    pol: Polarity,
    fuel: &mut u32,
) -> Option<TypeId> {
    if *fuel == 0 {
        return None; // fail-closed: fuel exhausted
    }
    *fuel -= 1;
    let data = ctx.get_arc(ty);
    match &*data {
        // ── The substituted variable ────────────────────────────────────
        TypeData::GenericParam { index, .. } if *index == x => match pol {
            Polarity::Pos => Some(image_pos),
            Polarity::Neg => image_neg,
        },
        // ── Fn: the (already-tested) expansion, now polarity-threaded ──
        TypeData::Fn { params, ret } => {
            let neg_pol = pol.compose(Polarity::Neg);
            let pos_pol = pol.compose(Polarity::Pos);
            let mut new_params: SmallVec<[TypeId; 4]> = SmallVec::with_capacity(params.len());
            for &p in params {
                new_params.push(functor_action_go(ctx, p, x, image_pos, image_neg, neg_pol, fuel)?);
            }
            let new_ret = functor_action_go(ctx, *ret, x, image_pos, image_neg, pos_pol, fuel)?;
            Some(ctx.function(new_params, new_ret))
        }
        // ── Covariant composites ────────────────────────────────────────
        TypeData::Tuple { elems } => {
            let p = pol.compose(Polarity::Pos);
            let mut new_elems = SmallVec::with_capacity(elems.len());
            for &e in elems {
                new_elems.push(functor_action_go(ctx, e, x, image_pos, image_neg, p, fuel)?);
            }
            Some(ctx.tuple(new_elems))
        }
        TypeData::Coproduct { alternatives } => {
            let p = pol.compose(Polarity::Pos);
            let mut new_alts = SmallVec::with_capacity(alternatives.len());
            for &a in alternatives {
                new_alts.push(functor_action_go(ctx, a, x, image_pos, image_neg, p, fuel)?);
            }
            Some(ctx.coproduct(new_alts))
        }
        TypeData::Adt { kind, def_id, args } => {
            // Nominal application: args instantiate covariantly (the same
            // treatment replace_generic and the κ counting graph give).
            let p = pol.compose(Polarity::Pos);
            let mut new_args = SmallVec::with_capacity(args.len());
            for &a in args {
                new_args.push(functor_action_go(ctx, a, x, image_pos, image_neg, p, fuel)?);
            }
            Some(ctx.alloc(TypeData::Adt { kind: *kind, def_id: *def_id, args: new_args }))
        }
        TypeData::Array { elem, size } => {
            let new_elem = functor_action_go(ctx, *elem, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.array(new_elem, *size))
        }
        TypeData::Slice { elem } => {
            let new_elem = functor_action_go(ctx, *elem, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.slice(new_elem))
        }
        TypeData::Ref { ty: inner, mutable, lifetime } => {
            if *mutable {
                // &mut T is invariant in T: no functorial image.
                if mentions_param_free(ctx, *inner, x) {
                    return None;
                }
                return Some(ty);
            }
            let new_inner = functor_action_go(ctx, *inner, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.reference_with_lifetime(new_inner, false, *lifetime))
        }
        TypeData::Pointer { ty: inner } => {
            let new_inner = functor_action_go(ctx, *inner, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.pointer(new_inner))
        }
        TypeData::AssociatedType { trait_id, name, self_ty } => {
            let new_self = functor_action_go(ctx, *self_ty, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.associated_type(*trait_id, *name, new_self))
        }
        TypeData::Opaque { def_id, hidden: Some(h) } => {
            let new_h = functor_action_go(ctx, *h, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.alloc(TypeData::Opaque { def_id: *def_id, hidden: Some(new_h) }))
        }
        // ── Invariant slot: no functorial action ────────────────────────
        TypeData::Ptr { size, pointee } => {
            if mentions_param_free(ctx, *size, x) || mentions_param_free(ctx, *pointee, x) {
                return None;
            }
            Some(ty)
        }
        // ── Binders: shadow ⇒ fold-stop; else guard + descend ──────────
        TypeData::Forall { param_index, param_name, body } => {
            if *param_index == x {
                return Some(ty); // binder shadows x — occurrences are bound
            }
            capture_guard(ctx, *param_index, image_pos, image_neg)?;
            let new_body = functor_action_go(ctx, *body, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.forall(*param_index, *param_name, new_body))
        }
        TypeData::Exists { param_index, name, base } => {
            if *param_index == x {
                return Some(ty);
            }
            capture_guard(ctx, *param_index, image_pos, image_neg)?;
            let new_base = functor_action_go(ctx, *base, x, image_pos, image_neg, pol, fuel)?;
            let new_id = ctx.alloc(TypeData::Exists {
                param_index: *param_index,
                name: *name,
                base: new_base,
            });
            // Preserve invariant/default metadata, mirroring subst().
            let src_meta = ctx.interner.meta.read().get(&ty).cloned();
            if let Some(meta) = src_meta {
                ctx.interner.meta.write().entry(new_id).or_insert(meta);
            }
            Some(new_id)
        }
        TypeData::Mu { param_index, param_name, body } => fold_arm(
            ctx, *param_index, *param_name, *body, ty, x, image_pos, image_neg, pol, fuel, true,
        ),
        TypeData::Nu { param_index, param_name, body } => fold_arm(
            ctx, *param_index, *param_name, *body, ty, x, image_pos, image_neg, pol, fuel, false,
        ),
        TypeData::Poly { quantifiers, body } => {
            if quantifiers.iter().any(|(i, _)| *i == x) {
                return Some(ty);
            }
            for (qi, _) in quantifiers {
                capture_guard(ctx, *qi, image_pos, image_neg)?;
            }
            let new_body = functor_action_go(ctx, *body, x, image_pos, image_neg, pol, fuel)?;
            Some(ctx.poly(quantifiers.clone(), new_body))
        }
        // ── Leaves & inference state: untouched ─────────────────────────
        _ => Some(ty),
    }
}

/// Shared `Mu`/`Nu` arm: shadow ⇒ opaque stop; else guard, recurse, rebuild.
#[allow(clippy::too_many_arguments)]
fn fold_arm<'input>(
    ctx: &mut TypeContext<'input>,
    param_index: usize,
    param_name: Symbol,
    body: TypeId,
    original: TypeId,
    x: usize,
    image_pos: TypeId,
    image_neg: Option<TypeId>,
    pol: Polarity,
    fuel: &mut u32,
    is_mu: bool,
) -> Option<TypeId> {
    if param_index == x {
        // THE FOLD: the reduct's own {µX.}/{νX.} binder shadows x — stop.
        // No unfold; no Fig. 4c dissolution of a (possibly vacuous)
        // binder here — re-opening the fold after substitution is exactly
        // the divergence the recursive closure must prevent.
        return Some(original);
    }
    capture_guard(ctx, param_index, image_pos, image_neg)?;
    let new_body = functor_action_go(ctx, body, x, image_pos, image_neg, pol, fuel)?;
    Some(if is_mu {
        ctx.alloc(TypeData::Mu { param_index, param_name, body: new_body })
    } else {
        ctx.alloc(TypeData::Nu { param_index, param_name, body: new_body })
    })
}

// ─── Recursive payload closure ──────────────────────────────────────────

/// Complete closure of the recursive payload instances of a fixpoint
/// reduct. Given `µX.Σₖ Bₖ[X]` (resp. ν) whose branch payloads still
/// mention the recursion variable — the `Int⇒X` branch being canonical —
/// re-instantiate each payload at the WHOLE fold in one pass:
///
/// ```text
/// µX.Σₖ Bₖ[X]  ⇒  µX.Σₖ Bₖ[µX.Σₖ Bₖ[X]]   (one unfold; image inserted opaque)
/// ```
///
/// Termination: `functor_action` stops at the fold binder, never re-enters
/// an inserted image, and the outer (possibly now vacuous) binder is kept,
/// NOT dissolved — dissolution would re-open the recursion (`Int⇒µX.F →
/// dissolve → …` loops). The interner shares the inserted image, so the
/// closed body is finite. Closing twice is the identity.
///
/// # Positivity (fail-closed)
///
/// A fold body mentioning the bound variable free in NEGATIVE position
/// admits no Set-semantics fixpoint to close against (Cantor: no
/// X ≅ 2^X); the κ solver classifies exactly those cycles `Undecidable`
/// (kappa.rs Phase 2). The closure refuses in lockstep.
pub fn close_recursive_payloads<'input>(
    ctx: &mut TypeContext<'input>,
    fixpoint: TypeId,
) -> Option<TypeId> {
    let data = ctx.get_arc(fixpoint);
    let (is_mu, x, param_name, body) = match &*data {
        TypeData::Mu { param_index, param_name, body } => (true, *param_index, *param_name, *body),
        TypeData::Nu { param_index, param_name, body } => (false, *param_index, *param_name, *body),
        _ => return None, // not a fold — nothing to close
    };
    if !is_positive_functor(ctx, body, x) {
        return None;
    }
    // One unfold. image_neg = None: after the positivity check no free
    // negative occurrence can remain; a masked one (defensive) fails
    // closed rather than fabricating an image.
    let closed_body = functor_action(ctx, body, x, fixpoint, None)?;
    debug_assert!(
        !mentions_param_free(ctx, closed_body, x),
        "closure left a free recursion variable"
    );
    Some(if is_mu {
        ctx.alloc(TypeData::Mu { param_index: x, param_name, body: closed_body })
    } else {
        ctx.alloc(TypeData::Nu { param_index: x, param_name, body: closed_body })
    })
}

/// Positivity: does `x` occur free in `body` ONLY in positive positions?
fn is_positive_functor<'input>(ctx: &TypeContext<'input>, body: TypeId, x: usize) -> bool {
    positive_in(ctx, body, x, Polarity::Pos, &mut Vec::new())
}

fn positive_in<'input>(
    ctx: &TypeContext<'input>,
    ty: TypeId,
    x: usize,
    pol: Polarity,
    bound: &mut Vec<usize>,
) -> bool {
    let data = ctx.get_arc(ty);
    match &*data {
        TypeData::GenericParam { index, .. } => {
            *index != x || bound.contains(index) || pol == Polarity::Pos
        }
        TypeData::Fn { params, ret } => {
            let neg_pol = pol.compose(Polarity::Neg);
            let pos_pol = pol.compose(Polarity::Pos);
            params.iter().all(|&p| positive_in(ctx, p, x, neg_pol, bound))
                && positive_in(ctx, *ret, x, pos_pol, bound)
        }
        TypeData::Forall { param_index, body, .. }
        | TypeData::Exists { param_index, base: body, .. }
        | TypeData::Mu { param_index, body, .. }
        | TypeData::Nu { param_index, body, .. } => {
            if *param_index == x {
                return true; // the fold's own binder — bound occurrences
            }
            bound.push(*param_index);
            let ok = positive_in(ctx, *body, x, pol, bound);
            bound.pop();
            ok
        }
        TypeData::Poly { quantifiers, body } => {
            if quantifiers.iter().any(|(i, _)| *i == x) {
                return true;
            }
            let n = quantifiers.len();
            for (qi, _) in quantifiers {
                bound.push(*qi);
            }
            let ok = positive_in(ctx, *body, x, pol, bound);
            for _ in 0..n {
                bound.pop();
            }
            ok
        }
        TypeData::Ptr { size, pointee } => {
            !mentions_param_free(ctx, *size, x) && !mentions_param_free(ctx, *pointee, x)
        }
        TypeData::Ref { ty, mutable, .. } => {
            if *mutable {
                // &mut T is INVARIANT in T (B6): an occurrence of x in the
                // pointee is not a positive occurrence — treat as non-positive
                // (aligns with Rust's actual variance, unlike the generic
                // covariant fallback below).
                !mentions_param_free(ctx, *ty, x)
            } else {
                // &T is covariant in T.
                positive_in(ctx, *ty, x, pol, bound)
            }
        }
        _ => {
            let mut ok = true;
            visit_type_children(ty, ctx, &mut |c| {
                if ok {
                    ok = positive_in(ctx, c, x, pol, bound);
                }
            });
            ok
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn functor_action_fn_expansion_positive() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let payload = ctx.function(smallvec![int], xv); // Int⇒X (X⁺ at ret)
        let img = ctx.bool();
        let out = functor_action(&mut ctx, payload, 0, img, None).expect("positive action");
        let expected = ctx.function(smallvec![int], img);
        assert_eq!(out, expected);
    }

    #[test]
    fn functor_action_negative_occurrence_contravariant() {
        let mut ctx = TypeContext::new();
        let u = ctx.unit();
        let xv = ctx.generic_param(0, "X".into());
        let payload = ctx.function(smallvec![xv], u); // X⇒Unit (X̄ at param)
        let img_pos = ctx.int(32, true);
        let img_neg = ctx.bool();
        let out =
            functor_action(&mut ctx, payload, 0, img_pos, Some(img_neg)).expect("both images");
        let expected = ctx.function(smallvec![img_neg], u);
        assert_eq!(out, expected);
        // Covariant-only reduct: no contravariant action ⇒ fail-closed.
        assert!(functor_action(&mut ctx, payload, 0, img_pos, None).is_none());
    }

    #[test]
    fn functor_action_double_negative_is_positive() {
        // (X⇒Unit)⇒Unit — X at Neg·Neg = Pos (double negation).
        let mut ctx = TypeContext::new();
        let u = ctx.unit();
        let xv = ctx.generic_param(0, "X".into());
        let inner = ctx.function(smallvec![xv], u);
        let payload = ctx.function(smallvec![inner], u);
        let img = ctx.bool();
        let out =
            functor_action(&mut ctx, payload, 0, img, None).expect("double-neg ⇒ positive");
        let expected_inner = ctx.function(smallvec![img], u);
        let expected = ctx.function(smallvec![expected_inner], u);
        assert_eq!(out, expected);
    }

    #[test]
    fn functor_action_stops_at_fold_binder() {
        // Tuple (µX.(Int⇒X), X): the fold part must stay opaque while the
        // free X maps — exercises the shadow rule, not just the fast path.
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let fn_int_x = ctx.function(smallvec![int], xv);
        let fold = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: "X".into(),
            body: fn_int_x,
        });
        let payload = ctx.tuple(smallvec![fold, xv]);
        let img = ctx.bool();
        let out = functor_action(&mut ctx, payload, 0, img, None).expect("shadow + free mix");
        let expected = ctx.tuple(smallvec![fold, img]);
        assert_eq!(out, expected);
    }

    #[test]
    fn closure_closes_int_arrow_branch() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let branch = ctx.function(smallvec![int], xv); // Int⇒X
        let u = ctx.unit();
        let body = ctx.coproduct(smallvec![branch, u]);
        let fold = ctx.alloc(TypeData::Mu { param_index: 0, param_name: "X".into(), body });
        let closed = close_recursive_payloads(&mut ctx, fold).expect("positive functor closes");
        assert_ne!(closed, fold);
        let closed_body = match ctx.get(closed) {
            TypeData::Mu { body, .. } => *body,
            _ => panic!("expected a fold"),
        };
        let alts = match ctx.get(closed_body) {
            TypeData::Coproduct { alternatives } => alternatives.to_vec(),
            _ => panic!("expected Σ"),
        };
        let ret = match ctx.get(alts[0]) {
            TypeData::Fn { ret, .. } => *ret,
            _ => panic!("expected the Int⇒… branch"),
        };
        assert_eq!(ret, fold, "branch return is the whole fold (opaque image)");
        // Termination/idempotence: the inserted image's binder shadows x
        // and the outer binder is NOT dissolved ⇒ second close is identity.
        let again = close_recursive_payloads(&mut ctx, closed).expect("idempotent");
        assert_eq!(again, closed);
    }

    #[test]
    fn closure_is_identity_without_free_occurrences() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let u = ctx.unit();
        let body = ctx.coproduct(smallvec![int, u]);
        let fold = ctx.alloc(TypeData::Mu { param_index: 0, param_name: "X".into(), body });
        assert_eq!(close_recursive_payloads(&mut ctx, fold), Some(fold));
    }

    #[test]
    fn closure_nu_side_closes_symmetrically() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let branch = ctx.function(smallvec![int], xv);
        let fold = ctx.alloc(TypeData::Nu { param_index: 0, param_name: "X".into(), body: branch });
        let closed = close_recursive_payloads(&mut ctx, fold).expect("positive functor closes");
        match ctx.get(closed) {
            TypeData::Nu { body, .. } => match ctx.get(*body) {
                TypeData::Fn { ret, .. } => assert_eq!(*ret, fold),
                other => panic!("expected the Int⇒… branch, got {other:?}"),
            },
            other => panic!("expected a ν-fold, got {other:?}"),
        }
    }

    #[test]
    fn closure_refuses_negative_payload_and_kappa_agrees() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let branch = ctx.function(smallvec![xv], int); // X⇒Int — X̄ inside the fold
        let u = ctx.unit();
        let body = ctx.coproduct(smallvec![branch, u]);
        let fold = ctx.alloc(TypeData::Mu { param_index: 0, param_name: "X".into(), body });
        assert!(
            close_recursive_payloads(&mut ctx, fold).is_none(),
            "non-positive functor ⇒ fail-closed"
        );
        assert!(
            matches!(ctx.characteristic(fold), Characteristic::Undecidable),
            "κ solver must agree: non-covariant cycle ⇒ Undecidable"
        );
    }

    #[test]
    fn functor_action_capture_guard_fails_closed() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let inner = ctx.function(smallvec![int], xv); // Int⇒X
        let payload = ctx.forall(1, "Y".into(), inner); // ∀Y.(Int⇒X)
        let img = ctx.bool();
        let ok = functor_action(&mut ctx, payload, 0, img, None).expect("no capture");
        let inner2 = ctx.function(smallvec![int], img);
        let expected = ctx.forall(1, "Y".into(), inner2);
        assert_eq!(ok, expected);
        // Image mentioning Y free would be captured ⇒ fail-closed.
        let capturing = ctx.generic_param(1, "Y".into());
        assert!(functor_action(&mut ctx, payload, 0, capturing, None).is_none());
    }
}
