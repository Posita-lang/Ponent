//! Polarized functor action + recursive payload closure.
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
//! counting).  Covariant-only legacy call sites keep working through
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

/// BASE of the size-scaled fuel budget for `functor_action` (see
/// `functor_action_fuel`); the binder-shadow rule is the real
/// termination argument — the fuel is only the fail-closed backstop.
const FUNCTOR_ACTION_FUEL: u32 = 100_000;

/// Size-scaled fuel budget: `max(BASE, 10 node_count)` — scaled
/// ABOVE the base for large types (a fixed constant could be exhausted
/// by a large-but-fine type) and deliberately NEVER below it: fuel is
/// consumed per VISIT, while the count is of DISTINCT nodes (visited
/// set) — a DAG with shared subtrees visits more nodes than it
/// contains, so a purely proportional budget could starve a small but
/// heavily-shared type.  The
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

/// The variance sign of the edge parent→child, AGREE-OR-REFUSE: when the
/// same child TypeId fills SEVERAL slots of the parent (S<X, X> with
/// mixed slot variances), a target-equality lookup cannot tell the slots
/// apart — the action is admissible only if every slot hosting that
/// target agrees on the sign.  `None` = no edge or disagreeing signs ⇒
/// the caller fails closed.
fn child_edge_sign<'input>(
    ctx: &TypeContext<'input>,
    parent: TypeId,
    child: TypeId,
) -> Option<isize> {
    let mut signs = ctx
        .compute_variance_edges(parent)
        .into_iter()
        .filter(|ve| ve.target == child)
        .map(|ve| ve.sign);
    let s0 = signs.next()?;
    signs.all(|s| s == s0).then_some(s0)
}

/// Hard recursion-DEPTH cap.  Source-derived types are bounded by
/// the parser's max_recursion_depth (256) and never approach this; the
/// cap exists because the walkers RECURSE and fuel bounds total WORK,
/// not stack — a hand-built deep type would drive proportionally many
/// frames through the stack before the fuel mattered.  512 is 2× the
/// source bound; in DEBUG builds the walker frames are fat (~2 KB), so
/// 512 frames ≈ 1 MB — inside even 2 MB worker stacks (the former
/// 2048-figure assumed ~400 B frames and overflowed test threads in
/// debug).  Fail-closed beyond.  Same discipline as coherence.rs's
/// iterative walkers, in capped-recursion rather than iterative form.
const POLAR_DEPTH_CAP: usize = 512;

/// Does `ty` mention param index `idx` FREE (binder shadowing respected)?
fn mentions_param_free<'input>(ctx: &TypeContext<'input>, ty: TypeId, idx: usize) -> bool {
    fn go<'input>(
        ctx: &TypeContext<'input>,
        ty: TypeId,
        idx: usize,
        bound: &mut Vec<usize>,
        depth: usize,
    ) -> bool {
        // Depth over-limit ⇒ TRUE: the fail-closed direction is unique —
        // the functor_action fast path turns true into "enter the
        // depth-capped go" (⇒ None), and capture_guard turns true into a
        // refusal.  A false here would silently admit.
        if depth > POLAR_DEPTH_CAP {
            return true;
        }
        let data = ctx.get_arc(ty);
        match &*data {
            TypeData::GenericParam { index, .. } => *index == idx && !bound.contains(index),
            TypeData::Forall {
                param_index, body, ..
            }
            | TypeData::Exists {
                param_index,
                base: body,
                ..
            }
            | TypeData::Mu {
                param_index, body, ..
            }
            | TypeData::Nu {
                param_index, body, ..
            } => {
                bound.push(*param_index);
                let found = go(ctx, *body, idx, bound, depth + 1);
                bound.pop();
                found
            }
            TypeData::Poly { quantifiers, body } => {
                let n = quantifiers.len();
                for (qi, _) in quantifiers {
                    bound.push(*qi);
                }
                let found = go(ctx, *body, idx, bound, depth + 1);
                for _ in 0..n {
                    bound.pop();
                }
                found
            }
            _ => {
                let mut found = false;
                visit_type_children(ty, ctx, &mut |c| {
                    if !found {
                        found = go(ctx, c, idx, bound, depth + 1);
                    }
                });
                found
            }
        }
    }
    go(ctx, ty, idx, &mut Vec::new(), 0)
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
    // Budget scales with the payload's type-graph size.
    let mut fuel = functor_action_fuel(ctx, payload);
    functor_action_go(
        ctx,
        payload,
        x,
        image_pos,
        image_neg,
        Polarity::Pos,
        &mut fuel,
        0,
    )
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
    depth: usize,
) -> Option<TypeId> {
    // Depth-capped fail-closed — fuel bounds total WORK, not
    // stack; the depth cap bounds the recursion itself (see
    // POLAR_DEPTH_CAP).
    if depth > POLAR_DEPTH_CAP {
        return None;
    }
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
                new_params.push(functor_action_go(
                    ctx,
                    p,
                    x,
                    image_pos,
                    image_neg,
                    neg_pol,
                    fuel,
                    depth + 1,
                )?);
            }
            let new_ret =
                functor_action_go(ctx, *ret, x, image_pos, image_neg, pos_pol, fuel, depth + 1)?;
            Some(ctx.function(new_params, new_ret))
        }
        // ── Covariant composites ────────────────────────────────────────
        TypeData::Tuple { elems } => {
            let p = pol.compose(Polarity::Pos);
            let mut new_elems = SmallVec::with_capacity(elems.len());
            for &e in elems {
                new_elems.push(functor_action_go(
                    ctx,
                    e,
                    x,
                    image_pos,
                    image_neg,
                    p,
                    fuel,
                    depth + 1,
                )?);
            }
            Some(ctx.tuple(new_elems))
        }
        TypeData::Coproduct { alternatives } => {
            let p = pol.compose(Polarity::Pos);
            let mut new_alts = SmallVec::with_capacity(alternatives.len());
            for &a in alternatives {
                new_alts.push(functor_action_go(
                    ctx,
                    a,
                    x,
                    image_pos,
                    image_neg,
                    p,
                    fuel,
                    depth + 1,
                )?);
            }
            Some(ctx.coproduct(new_alts))
        }
        TypeData::Adt { kind, def_id, args } => {
            // Nominal application: consult the declared variance for each
            // argument that mentions x.  sign == 1 ⇒ covariant (Pos);
            // sign == -1 ⇒ recurse contravariantly (requires a negative
            // image); sign == 0 or unknown variance ⇒ fail-closed rather
            // than fabricating an image.  This keeps the functor action in
            // agreement with the positivity gate and the κ solver.
            let mut new_args = SmallVec::with_capacity(args.len());
            for &a in args {
                if !mentions_param_free(ctx, a, x) {
                    new_args.push(a);
                    continue;
                }
                // AGREE-OR-REFUSE (child_edge_sign): the former
                // target-equality `find` handed the FIRST slot's sign to
                // every slot hosting the same TypeId — an invariant slot
                // (a *mut pointee field) would receive the covariant
                // image, the exact mis-threading the §2.4 invariant-slot
                // discipline forbids.  All slots hosting `a` must agree,
                // else the whole substitution fails closed
                // (is_positive_functor's Adt arm already refuses mixed
                // slots on the positivity side — the two stay in
                // lockstep).
                let sign = child_edge_sign(ctx, ty, a);
                match sign {
                    Some(1) => {
                        let p = pol.compose(Polarity::Pos);
                        new_args.push(functor_action_go(
                            ctx,
                            a,
                            x,
                            image_pos,
                            image_neg,
                            p,
                            fuel,
                            depth + 1,
                        )?);
                    }
                    Some(-1) => {
                        let neg = pol.compose(Polarity::Neg);
                        let img_neg = image_neg?;
                        new_args.push(functor_action_go(
                            ctx,
                            a,
                            x,
                            image_pos,
                            Some(img_neg),
                            neg,
                            fuel,
                            depth + 1,
                        )?);
                    }
                    _ => return None,
                }
            }
            Some(ctx.alloc(TypeData::Adt {
                kind: *kind,
                def_id: *def_id,
                args: new_args,
            }))
        }
        TypeData::Array { elem, size } => {
            let new_elem =
                functor_action_go(ctx, *elem, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.array(new_elem, *size))
        }
        TypeData::Slice { elem } => {
            let new_elem =
                functor_action_go(ctx, *elem, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.slice(new_elem))
        }
        TypeData::Ref {
            ty: inner,
            mutable,
            lifetime,
        } => {
            if *mutable {
                // &mut T is invariant in T: no functorial image.
                if mentions_param_free(ctx, *inner, x) {
                    return None;
                }
                return Some(ty);
            }
            let new_inner =
                functor_action_go(ctx, *inner, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.reference_with_lifetime(new_inner, false, *lifetime))
        }
        TypeData::Pointer {
            ty: inner, mutable, ..
        } => {
            if *mutable {
                // *mut T is INVARIANT in T: no functorial image — same
                // rule as the Ref { mutable: true } arm.
                if mentions_param_free(ctx, *inner, x) {
                    return None;
                }
                return Some(ty);
            }
            // *const T is covariant in T (the current `*T` sugar).
            let new_inner =
                functor_action_go(ctx, *inner, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.pointer(new_inner))
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            // AssociatedType — the same variance-authority ruling as the Opaque arm: the
            // projection's self-type edge sign is the variance module's
            // call (the authority kappa's graph builder defers to).
            if !mentions_param_free(ctx, *self_ty, x) {
                return Some(ty);
            }
            match child_edge_sign(ctx, ty, *self_ty) {
                Some(1) => {
                    let new_self = functor_action_go(
                        ctx,
                        *self_ty,
                        x,
                        image_pos,
                        image_neg,
                        pol,
                        fuel,
                        depth + 1,
                    )?;
                    Some(ctx.associated_type(*trait_id, *name, new_self))
                }
                _ => None,
            }
        }
        TypeData::Opaque {
            def_id,
            hidden: Some(h),
        } => {
            // Polarity through a TAIT is whatever the variance module
            // declares for the opaque→hidden edge — the SAME authority
            // kappa's graph builder defers to (resolve_children's `_`
            // arm).  The former hard-coded transparency handed the
            // covariant image to an invariant Opaque slot whenever
            // variance.rs marks the edge invariant (Rust's own rule for
            // opaque types), and the positivity gate passed the same
            // occurrence the κ solver called non-covariant.  Covariant
            // edge ⇒ thread; anything else ⇒ fail-closed (the
            // invariant-slot discipline of the Ptr / &mut arms).
            // `hidden: None` stays a leaf.
            if !mentions_param_free(ctx, *h, x) {
                return Some(ty);
            }
            match child_edge_sign(ctx, ty, *h) {
                Some(1) => {
                    let new_h =
                        functor_action_go(ctx, *h, x, image_pos, image_neg, pol, fuel, depth + 1)?;
                    Some(ctx.alloc(TypeData::Opaque {
                        def_id: *def_id,
                        hidden: Some(new_h),
                    }))
                }
                _ => None,
            }
        }
        // ── Invariant slot: no functorial action ────────────────────────
        TypeData::Ptr {
            size,
            pointee,
            mutable,
            ..
        } => {
            // size: the pointer's own width axis — invariant, never
            // functorial (§2.4).
            if mentions_param_free(ctx, *size, x) {
                return None;
            }
            if *mutable {
                // Ptr<..., mut>: invariant in pointee (§2.4).
                if mentions_param_free(ctx, *pointee, x) {
                    return None;
                }
                return Some(ty);
            }
            // Ptr<..., const>: covariant in pointee.
            let new_pointee =
                functor_action_go(ctx, *pointee, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.ptr(*size, new_pointee))
        }
        // ── Binders: shadow ⇒ fold-stop; else guard + descend ──────────
        TypeData::Forall {
            param_index,
            param_name,
            body,
        } => {
            if *param_index == x {
                return Some(ty); // binder shadows x — occurrences are bound
            }
            capture_guard(ctx, *param_index, image_pos, image_neg)?;
            let new_body =
                functor_action_go(ctx, *body, x, image_pos, image_neg, pol, fuel, depth + 1)?;
            Some(ctx.forall(*param_index, *param_name, new_body))
        }
        TypeData::Exists {
            param_index,
            name,
            base,
        } => {
            if *param_index == x {
                return Some(ty);
            }
            capture_guard(ctx, *param_index, image_pos, image_neg)?;
            let new_base =
                functor_action_go(ctx, *base, x, image_pos, image_neg, pol, fuel, depth + 1)?;
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
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => fold_arm(
            ctx,
            *param_index,
            *param_name,
            *body,
            ty,
            x,
            image_pos,
            image_neg,
            pol,
            fuel,
            depth,
            true,
        ),
        TypeData::Nu {
            param_index,
            param_name,
            body,
        } => fold_arm(
            ctx,
            *param_index,
            *param_name,
            *body,
            ty,
            x,
            image_pos,
            image_neg,
            pol,
            fuel,
            depth,
            false,
        ),
        TypeData::Poly { quantifiers, body } => {
            if quantifiers.iter().any(|(i, _)| *i == x) {
                return Some(ty);
            }
            for (qi, _) in quantifiers {
                capture_guard(ctx, *qi, image_pos, image_neg)?;
            }
            let new_body =
                functor_action_go(ctx, *body, x, image_pos, image_neg, pol, fuel, depth + 1)?;
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
    depth: usize,
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
    let new_body = functor_action_go(ctx, body, x, image_pos, image_neg, pol, fuel, depth + 1)?;
    Some(if is_mu {
        ctx.alloc(TypeData::Mu {
            param_index,
            param_name,
            body: new_body,
        })
    } else {
        ctx.alloc(TypeData::Nu {
            param_index,
            param_name,
            body: new_body,
        })
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
/// (kappa.rs Phase 2 via the function-space rule) and the closure
/// refuses in lockstep — FOR THE CONTRAVARIANT CASE.  Invariant-slot
/// cycles (e.g. µX.(*mut X), or an ADT self-payload hiding a `*mut`
/// behind its back-edge) refuse here at the positivity gate while κ
/// counts inhabitants InfiniteEnumerable — semantically right for
/// inhabitant counting (X ≅ Addr is a solution), a DIFFERENT question
/// (see the kappa.rs header).  The former blanket "lockstep" claim was
/// over-broad.
pub fn close_recursive_payloads<'input>(
    ctx: &mut TypeContext<'input>,
    fixpoint: TypeId,
) -> Option<TypeId> {
    let data = ctx.get_arc(fixpoint);
    let (is_mu, x, param_name, body) = match &*data {
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => (true, *param_index, *param_name, *body),
        TypeData::Nu {
            param_index,
            param_name,
            body,
        } => (false, *param_index, *param_name, *body),
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
        ctx.alloc(TypeData::Mu {
            param_index: x,
            param_name,
            body: closed_body,
        })
    } else {
        ctx.alloc(TypeData::Nu {
            param_index: x,
            param_name,
            body: closed_body,
        })
    })
}

/// Positivity: does `x` occur free in `body` ONLY in positive positions?
fn is_positive_functor<'input>(ctx: &TypeContext<'input>, body: TypeId, x: usize) -> bool {
    positive_in(ctx, body, x, Polarity::Pos, &mut Vec::new(), 0)
}

fn positive_in<'input>(
    ctx: &TypeContext<'input>,
    ty: TypeId,
    x: usize,
    pol: Polarity,
    bound: &mut Vec<usize>,
    depth: usize,
) -> bool {
    // Depth-capped fail-closed — over-limit ⇒ FALSE (non-positive
    // ⇒ the closure refuses).  The reverse direction (true) would admit
    // an occurrence the depth never examined.
    if depth > POLAR_DEPTH_CAP {
        return false;
    }
    let data = ctx.get_arc(ty);
    match &*data {
        TypeData::GenericParam { index, .. } => {
            *index != x || bound.contains(index) || pol == Polarity::Pos
        }
        TypeData::Fn { params, ret } => {
            let neg_pol = pol.compose(Polarity::Neg);
            let pos_pol = pol.compose(Polarity::Pos);
            params
                .iter()
                .all(|&p| positive_in(ctx, p, x, neg_pol, bound, depth + 1))
                && positive_in(ctx, *ret, x, pos_pol, bound, depth + 1)
        }
        TypeData::Forall {
            param_index, body, ..
        }
        | TypeData::Exists {
            param_index,
            base: body,
            ..
        }
        | TypeData::Mu {
            param_index, body, ..
        }
        | TypeData::Nu {
            param_index, body, ..
        } => {
            if *param_index == x {
                return true; // the fold's own binder — bound occurrences
            }
            bound.push(*param_index);
            let ok = positive_in(ctx, *body, x, pol, bound, depth + 1);
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
            let ok = positive_in(ctx, *body, x, pol, bound, depth + 1);
            for _ in 0..n {
                bound.pop();
            }
            ok
        }
        TypeData::Ptr {
            size,
            pointee,
            mutable,
            ..
        } => {
            // size: the pointer's own width axis — invariant (§2.4): an
            // occurrence there is never a positive occurrence.
            if mentions_param_free(ctx, *size, x) {
                return false;
            }
            if *mutable {
                // Ptr<..., mut>: invariant in pointee ⇒ non-positive.
                !mentions_param_free(ctx, *pointee, x)
            } else {
                // Ptr<..., const>: covariant in pointee.
                positive_in(ctx, *pointee, x, pol, bound, depth + 1)
            }
        }
        TypeData::Ref { ty, mutable, .. } => {
            if *mutable {
                // &mut T is INVARIANT in T: an occurrence of x in the
                // pointee is not a positive occurrence — treat as non-positive
                // (aligns with Rust's actual variance, unlike the generic
                // covariant fallback below).
                !mentions_param_free(ctx, *ty, x)
            } else {
                // &T is covariant in T.
                positive_in(ctx, *ty, x, pol, bound, depth + 1)
            }
        }
        TypeData::Pointer {
            ty: inner, mutable, ..
        } => {
            if *mutable {
                // *mut T is INVARIANT in T: an occurrence of x in the
                // pointee is not a positive occurrence (same rule as the
                // Ref { mutable: true } arm above).
                !mentions_param_free(ctx, *inner, x)
            } else {
                // *const T (the `*T` sugar) is covariant in T.
                positive_in(ctx, *inner, x, pol, bound, depth + 1)
            }
        }
        TypeData::Adt { .. } => {
            // Nominal application: consult the declared variance.  An
            // argument mentioning x in a non-covariant slot is NOT a
            // positive occurrence — the former covariant fallback let a
            // negative/invariant ADT parameter slip past the positivity
            // gate while κ called the same cycle Undecidable.
            for ve in ctx.compute_variance_edges(ty) {
                if ve.sign != 1 && mentions_param_free(ctx, ve.target, x) {
                    return false;
                }
            }
            let mut ok = true;
            visit_type_children(ty, ctx, &mut |c| {
                if ok {
                    ok = positive_in(ctx, c, x, pol, bound, depth + 1);
                }
            });
            ok
        }
        TypeData::Opaque {
            hidden: Some(h), ..
        } => {
            // An x-occurrence inside a revealed TAIT hidden type is a
            // positive occurrence ONLY if the variance module declares
            // the opaque→hidden edge covariant — same authority as
            // functor_action_go's Opaque arm and kappa's graph builder.
            // The former `_` walk threaded pol through unconditionally.
            match child_edge_sign(ctx, ty, *h) {
                Some(1) => positive_in(ctx, *h, x, pol, bound, depth + 1),
                _ => !mentions_param_free(ctx, *h, x),
            }
        }
        _ => {
            let mut ok = true;
            visit_type_children(ty, ctx, &mut |c| {
                if ok {
                    ok = positive_in(ctx, c, x, pol, bound, depth + 1);
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
        let out = functor_action(&mut ctx, payload, 0, img, None).expect("double-neg ⇒ positive");
        let expected_inner = ctx.function(smallvec![img], u);
        let expected = ctx.function(smallvec![expected_inner], u);
        assert_eq!(out, expected);
    }

    #[test]
    fn deep_handbuilt_types_fail_closed_not_crash() {
        // The three walkers recurse — fuel bounds total WORK, not
        // stack.  The pin builds just PAST the cap: pre-fix the walkers
        // happily recursed to this depth (returning Some), post-fix the
        // depth cap fires first and yields None.  (Building at the fuel's
        // nominal ~100k magnitude is not even reachable — the interner's
        // own volatility pre-check recurses over the whole chain and
        // overflows the stack during CONSTRUCTION; that auxiliary walk is
        // outside this module's scope, but it is the same no-recursion
        // discipline debt.)
        let mut ctx = TypeContext::new();
        let xv = ctx.generic_param(0, "X".into());
        let img = ctx.bool();
        let mut deep = xv;
        for _ in 0..POLAR_DEPTH_CAP + 100 {
            deep = ctx.tuple(smallvec![deep]);
        }
        assert!(
            functor_action(&mut ctx, deep, 0, img, None).is_none(),
            "depth-capped fail-closed past the cap, NOT deep recursion"
        );
        // Shallow control substitutes as usual.
        let shallow = ctx.tuple(smallvec![xv]);
        let out = functor_action(&mut ctx, shallow, 0, img, None).expect("shallow");
        match ctx.get(out) {
            TypeData::Tuple { elems } => assert_eq!(elems[0], img),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn opaque_slot_polarity_follows_variance_module() {
        // Whatever variance.rs declares for the opaque→hidden edge,
        // the functor action and the positivity gate must AGREE with it —
        // and with kappa's graph builder, which defers to the same
        // function.
        let mut ctx = TypeContext::new();
        let xv = ctx.generic_param(0, "X".into());
        let int = ctx.int(32, true);
        let hidden = ctx.function(smallvec![int], xv);
        let op = ctx.alloc(TypeData::Opaque {
            def_id: DefId(8101),
            hidden: Some(hidden),
        });
        let cov = ctx
            .compute_variance_edges(op)
            .iter()
            .any(|ve| ve.target == hidden && ve.sign == 1);
        let img = ctx.bool();
        let acted = functor_action(&mut ctx, op, 0, img, None);
        let positive = is_positive_functor(&mut ctx, op, 0);
        assert_eq!(
            acted.is_some(),
            cov,
            "action threads iff the edge is covariant"
        );
        assert_eq!(positive, cov, "positivity agrees with the same edge");
    }

    #[test]
    fn adt_duplicate_target_mixed_variance_fails_closed() {
        // The same child TypeId filling SEVERAL slots of an Adt —
        // target-equality sign lookup handed the FIRST slot's sign to all
        // slots; an invariant slot (*mut pointee) would receive the
        // covariant image.  Agree-or-refuse ⇒ None.
        let mut ctx = TypeContext::new();
        // Prototype: struct S<A, B> { f0: A, f1: *mut B } — slot variance
        // via the adt-def field convention (GenericParam{i} marks arg
        // slot i, as in adt_field_is_copy).
        let p0 = ctx.generic_param(0, "A".into());
        let p1 = ctx.generic_param(1, "B".into());
        let mut_p1 = ctx.pointer_mut(p1);
        ctx.register_adt(
            DefId(8001),
            AdtDef {
                fields: vec![p0, mut_p1],
                has_drop: false,
            },
        );
        let xv = ctx.generic_param(9, "X".into());
        let s = ctx.struct_ty(DefId(8001), smallvec![xv, xv]); // S<X, X>
        // Premise guard against the ACTUAL authority: with the per-slot
        // derive in place, the registered field convention
        // (GenericParam{i} marks arg slot i) makes this the REACHABLE
        // mixed-sign case the agree-or-refuse lookup was built for —
        // field 0 (covariant) vs field 1 (*mut pointee, invariant).
        // The former uniform-invariant derive masked it ([0, 0]).
        let signs: Vec<_> = ctx
            .compute_variance_edges(s)
            .iter()
            .filter(|ve| ve.target == xv)
            .map(|ve| ve.sign)
            .collect();
        assert_eq!(signs, vec![1, 0], "premise: mixed slot variances for X");
        let img = ctx.bool();
        assert!(
            functor_action(&mut ctx, s, 9, img, None).is_none(),
            "agree-or-refuse ⇒ None (mixed slots refuse; the former find() would have threaded the covariant image into the invariant slot)"
        );
        // The helper's OWN contract, on the reachable duplicate-target
        // shapes: Fn carries the same target at sign -1 AND +1 — mixed
        // ⇒ None (the former find() would have taken -1); a Tuple
        // hosting the target twice at +1 agrees ⇒ Some(1).
        let dup_fn = ctx.function(smallvec![xv], xv); // fn(X) → X
        assert_eq!(
            child_edge_sign(&ctx, dup_fn, xv),
            None,
            "mixed signs (param -1, ret +1) ⇒ refuse"
        );
        let dup_tuple = ctx.tuple(smallvec![xv, xv]);
        assert_eq!(
            child_edge_sign(&ctx, dup_tuple, xv),
            Some(1),
            "agreeing covariant slots ⇒ Some(1)"
        );
        // No over-rejection on the agreeing path: the covariant composite
        // (Tuple — every edge sign +1) hosting the target twice
        // substitutes fully.
        let out = functor_action(&mut ctx, dup_tuple, 9, img, None)
            .expect("agreeing covariant slots substitute");
        match ctx.get(out) {
            TypeData::Tuple { elems } => assert!(elems.iter().all(|&a| a == img)),
            other => panic!("{other:?}"),
        }
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
        let fold = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: "X".into(),
            body,
        });
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
        let fold = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: "X".into(),
            body,
        });
        assert_eq!(close_recursive_payloads(&mut ctx, fold), Some(fold));
    }

    #[test]
    fn closure_nu_side_closes_symmetrically() {
        let mut ctx = TypeContext::new();
        let int = ctx.int(32, true);
        let xv = ctx.generic_param(0, "X".into());
        let branch = ctx.function(smallvec![int], xv);
        let fold = ctx.alloc(TypeData::Nu {
            param_index: 0,
            param_name: "X".into(),
            body: branch,
        });
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
        let fold = ctx.alloc(TypeData::Mu {
            param_index: 0,
            param_name: "X".into(),
            body,
        });
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

    #[test]
    fn pointer_mut_is_invariant_const_pointer_covariant() {
        let mut ctx = TypeContext::new();
        let xv = ctx.generic_param(0, "X".into());
        let img = ctx.bool();
        // *mut X: hand-minted (no surface form yet) — invariant slot.
        let pm = ctx.alloc(TypeData::Pointer {
            ty: xv,
            mutable: true,
        });
        assert!(
            functor_action(&mut ctx, pm, 0, img, None).is_none(),
            "*mut T is invariant: no functorial image when T mentions x"
        );
        let u = ctx.unit();
        let tup = ctx.tuple(smallvec![pm, u]);
        assert!(
            !is_positive_functor(&mut ctx, tup, 0),
            "*mut X is not a positive occurrence of x"
        );
        // *const X (today's only surface form): still covariant.
        let pc = ctx.pointer(xv);
        let out = functor_action(&mut ctx, pc, 0, img, None).expect("*const covariant");
        assert_eq!(
            ctx.get(out),
            &TypeData::Pointer {
                ty: img,
                mutable: false
            }
        );
    }

    #[test]
    fn variance_edges_match_pointer_mutability() {
        let mut ctx = TypeContext::new();
        let xv = ctx.generic_param(0, "X".into());
        let us = ctx.usize();
        // *mut X: the pointee edge is INVARIANT (§2.4).
        let pm = ctx.alloc(TypeData::Pointer {
            ty: xv,
            mutable: true,
        });
        assert!(
            ctx.compute_variance_edges(pm)
                .iter()
                .any(|ve| ve.target == xv && ve.sign == 0),
            "*mut must emit an INVARIANT edge (§2.4)"
        );
        // *const X: the pointee edge is covariant.
        let pc = ctx.pointer(xv);
        assert!(
            ctx.compute_variance_edges(pc)
                .iter()
                .any(|ve| ve.target == xv && ve.sign == 1)
        );
        // Ptr<size, mut>: size axis invariant, pointee invariant.
        let psm = ctx.ptr_mut(us, xv);
        let edges = ctx.compute_variance_edges(psm);
        assert!(
            edges.iter().any(|ve| ve.target == us && ve.sign == 0),
            "size axis invariant (§2.4)"
        );
        assert!(edges.iter().any(|ve| ve.target == xv && ve.sign == 0));
        // Ptr<size, const>: size invariant, pointee covariant.
        let psc = ctx.ptr(us, xv);
        let edges = ctx.compute_variance_edges(psc);
        assert!(edges.iter().any(|ve| ve.target == us && ve.sign == 0));
        assert!(edges.iter().any(|ve| ve.target == xv && ve.sign == 1));
    }
}
