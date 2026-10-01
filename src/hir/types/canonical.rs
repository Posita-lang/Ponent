//! Canonical tree IR for Yoneda reduction (Pistone & Tranchini 2022 §3).
//!
//! The paper defines Yoneda reduction not directly over types but over a
//! class of finite trees representing the types of Λ2 (crucially excluding
//! those built with `0`, `+`, …) *up to βη-isomorphism*.  This module
//! provides that intermediate representation together with the Fig. 4(a)
//! βη-normalization used to bring a `TypeData` tree into canonical form.
//!
//! ## Design decisions
//!
//! - **Positive / negative polarization.**  Canonical negative types are
//!   curried arrow chains `Π⃗A ⇒ N` plus quantifiers; canonical positive
//!   types are products/sums/fixpoints/variables.  This matches the shapes
//!   the ≡_X / ≡^X schemas consume.
//!
//! - **User-written sums are opaque.**  The paper excludes `+`/`0` from
//!   trees.  In this codebase `Coproduct` is (today) only *generated* by
//!   Yoneda reduction itself, never written by users; we still treat it as
//!   an **opaque positive atom** (`Pos::Opaque`) so that any future
//!   user-facing sum type cannot silently participate in βη normalization.
//!   Only the `Σₖ` produced by the rewriter is recognized by the rewriter.
//!
//! - **Binder identity = `param_index`.**  HIR binder indices come from the
//!   globally-unique monotonic `fresh_param_index()` allocator (plus item-
//!   scoped GenericParam indices that never collide with fresh ones inside
//!   the types we normalize).  We therefore keep explicit indices instead of
//!   shifting de Bruijn levels, and lower back with fresh index allocation
//!   only where a binder must be re-created (see `from_neg` / `from_pos`).
//!   This sidesteps the fragile "fresh binder lands on 1" coupling noted in
//!   `test_coyoneda_inner_quantifier_one`.

use super::*;

/// A canonical positive type: μ / product / variable / opaque atoms.
///
/// Sum (`+`) and empty (`0`) constructors are intentionally absent — see the
/// module docs.  They are represented as `Pos::Opaque`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pos {
    /// A (free or bound) type variable occurrence.
    Var(usize),
    /// The unit type `1`.
    Unit,
    /// A user-level type constructor applied to canonical arguments
    /// (ADTs, primitives, arrays, refs, …) or any position the canonical
    /// grammar does not further decompose (Coproduct, InferVar, …).
    /// `None` args = a nullary atom (Bool, Int, Unit already lifted, …).
    Opaque(TypeId, Vec<Pos>),
    /// Product `A₁ × … × Aₙ` in **sorted** order (commutativity, Fig. 4b).
    Prod(Vec<Pos>),
    /// Least fixpoint μX.A.
    Mu(usize, Box<Pos>),
}

/// A canonical negative type: curried arrow chains + quantifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Neg {
    /// A positive type in negative position (the spine's return type).
    Atom(Pos),
    /// Curried arrow `A₁ ⇒ … ⇒ Aₙ ⇒ N`  (currying + reordering, Fig. 4a/4b).
    Arrow(Vec<Pos>, Box<Neg>),
    /// Universal quantifier ∀X.N.
    Forall(usize, Box<Neg>),
}

impl Neg {
    /// Free variable indices of this canonical type.
    pub fn fvs(&self) -> std::collections::HashSet<usize> {
        let mut s = std::collections::HashSet::new();
        self.collect_fvs(&mut s);
        s
    }

    fn collect_fvs(&self, out: &mut std::collections::HashSet<usize>) {
        match self {
            Neg::Atom(p) => p.collect_fvs(out),
            Neg::Arrow(params, ret) => {
                for p in params {
                    p.collect_fvs(out);
                }
                ret.collect_fvs(out);
            }
            Neg::Forall(idx, body) => {
                body.collect_fvs(out);
                out.remove(idx);
            }
        }
    }

    /// True if this canonical type is βη-equal to `1` under the paper's
    /// partial axiomatization (spine of vacuous ∀-binders returning Unit).
    pub fn is_unit(&self) -> bool {
        match self {
            Neg::Atom(Pos::Unit) => true,
            Neg::Forall(_, body) => body.is_unit(),
            _ => false,
        }
    }
}

impl Pos {
    /// Free variable indices of this canonical positive type.
    fn fvs(&self) -> std::collections::HashSet<usize> {
        let mut s = std::collections::HashSet::new();
        self.collect_fvs(&mut s);
        s
    }

    fn collect_fvs(&self, out: &mut std::collections::HashSet<usize>) {
        match self {
            Pos::Var(i) => {
                out.insert(*i);
            }
            Pos::Unit => {}
            Pos::Opaque(_, args) if args.is_empty() => {}
            Pos::Prod(items) => {
                for p in items {
                    p.collect_fvs(out);
                }
            }
            Pos::Mu(idx, body) => {
                body.collect_fvs(out);
                out.remove(idx);
            }
            Pos::Opaque(_, args) => {
                for a in args {
                    a.collect_fvs(out);
                }
            }
        }
    }
}

impl<'input> TypeContext<'input> {
    /// Lower a `TypeId` to a canonical **negative** tree, applying the
    /// Fig. 4(a) βη-normalizations:
    ///   - currying `(A×B)⇒C ≡ A⇒(B⇒C)` and argument reordering
    ///     `A⇒(B⇒C) ≡ B⇒(A⇒C)` (params sorted by canonical form);
    ///   - quantifier hoisting `A⇒∀X.B ≡ ∀X.(A⇒B)`;
    ///   - vacuous binder elimination `A ≡ ∀X.A (X∉FV(A))`.
    pub fn to_neg(&self, ty: TypeId) -> Neg {
        let resolved = self.resolve_binding(ty);
        let data = self.get_arc(resolved);
        match &*data {
            TypeData::Forall {
                param_index, body, ..
            } => {
                let inner = self.to_neg(*body);
                // Vacuous binder elimination (Fig. 4a): A ≡ ∀X.A when X∉FV(A).
                if !inner.fvs().contains(param_index) {
                    inner
                } else {
                    Neg::Forall(*param_index, Box::new(inner))
                }
            }
            TypeData::Fn { params, ret } => {
                // Currying: flatten nested single-param arrows and tuple
                // params into one sorted spine (Fig. 4a + 4b).
                let mut spine: Vec<Pos> = Vec::new();
                let mut collect = |t: TypeId, ctx: &Self| {
                    let p = ctx.to_pos(t);
                    match p {
                        // (A×B)⇒C ≡ A⇒(B⇒C)
                        Pos::Prod(items) => spine.extend(items),
                        other => spine.push(other),
                    }
                };
                for &p in params {
                    collect(p, self);
                }
                spine.sort();
                // Flatten nested arrows into one curried spine
                // ((A⇒B)⇒C ≡ A⇒(B⇒C) orientation), then hoist foralls.
                let ret_neg = self.to_neg(*ret);
                let (spine, ret_neg) = match ret_neg {
                    Neg::Arrow(spine2, ret2) => {
                        spine.extend(spine2);
                        spine.sort();
                        (spine, *ret2)
                    }
                    other => (spine, other),
                };
                // Quantifier hoisting (Fig. 4a): A⇒∀X.B ≡ ∀X.(A⇒B).
                match ret_neg {
                    Neg::Forall(idx, body) => Neg::Forall(idx, Box::new(Neg::Arrow(spine, body))),
                    other => Neg::Arrow(spine, Box::new(other)),
                }
            }
            // A non-arrow, non-forall type in negative position is an atom.
            _ => Neg::Atom(self.to_pos(ty)),
        }
    }

    /// Lower a `TypeId` to a canonical **positive** tree.
    pub fn to_pos(&self, ty: TypeId) -> Pos {
        let resolved = self.resolve_binding(ty);
        let data = self.get_arc(resolved);
        match &*data {
            TypeData::GenericParam { index, .. } => Pos::Var(*index),
            TypeData::Unit => Pos::Unit,
            TypeData::Tuple { elems } => {
                // Product: flatten nested tuples, sort (commutativity Fig. 4b),
                // drop unit factors (A×1 ≡ A).
                let mut flat: Vec<Pos> = Vec::new();
                for &e in elems {
                    match self.to_pos(e) {
                        Pos::Unit => {}
                        Pos::Prod(items) => flat.extend(items),
                        other => flat.push(other),
                    }
                }
                flat.sort();
                match flat.len() {
                    0 => Pos::Unit,
                    1 => flat.into_iter().next().unwrap(),
                    _ => Pos::Prod(flat),
                }
            }
            TypeData::Mu {
                param_index, body, ..
            } => {
                let inner = self.to_pos(*body);
                // Vacuous fixpoint elimination (Fig. 4c): μX.A ≡ A when X∉FV(A).
                if !inner.fvs().contains(param_index) {
                    inner
                } else {
                    Pos::Mu(*param_index, Box::new(inner))
                }
            }
            _ => {
                // Opaque positive atom: primitives, ADTs, Coproduct, Never,
                // InferVar, …  Recurse into children as canonical args.
                let args: Vec<Pos> = {
                    let mut acc = Vec::new();
                    visit_type_children(ty, self, &mut |c| acc.push(self.to_pos(c)));
                    acc.sort();
                    acc
                };
                Pos::Opaque(resolved, args)
            }
        }
    }

    /// Raise a canonical negative tree back to a `TypeId`.
    pub fn from_neg(&mut self, neg: &Neg) -> TypeId {
        match neg {
            Neg::Atom(p) => self.from_pos(p),
            Neg::Arrow(params, ret) => {
                let ret_ty = self.from_neg(ret);
                // Rebuild the curried chain right-to-left.  Empty spine =
                // the return type itself (0-ary arrow is not representable
                // in TypeData; `0⇒A ≡ 1` cases never produce empty spines).
                let mut acc = ret_ty;
                for p in params.iter().rev() {
                    let arg = self.from_pos(p);
                    acc = self.function(smallvec![arg], acc);
                }
                acc
            }
            Neg::Forall(idx, body) => {
                let inner = self.from_neg(body);
                self.forall(*idx, Symbol::intern(&format!("X{}", idx)), inner)
            }
        }
    }

    /// Raise a canonical positive tree back to a `TypeId`.
    pub fn from_pos(&mut self, pos: &Pos) -> TypeId {
        match pos {
            Pos::Var(i) => self.generic_param(*i, Symbol::intern(&format!("T{}", i))),
            Pos::Unit => self.unit(),
            Pos::Prod(items) => match items.len() {
                0 => self.unit(),
                1 => self.from_pos(&items[0]),
                _ => {
                    let elems = items.iter().map(|p| self.from_pos(p)).collect();
                    self.tuple(elems)
                }
            },
            Pos::Mu(idx, body) => {
                let inner = self.from_pos(body);
                self.alloc(TypeData::Mu {
                    param_index: *idx,
                    param_name: Symbol::intern(&format!("X{}", idx)),
                    body: inner,
                })
            }
            Pos::Opaque(ty, _) => {
                // Composite opaques are compared in canonical form directly
                // (see `neg_equiv`); raising them back to TypeData by child
                // substitution is not needed for equivalence queries and
                // would risk capture.  Fail-closed: return the original.
                *ty
            }
        }
    }
}

/// Sort key for canonical forms so that product factors and arrow params
/// have a total order (commutativity normalization, Fig. 4b).
impl Ord for Pos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        fn rank(p: &Pos) -> u8 {
            match p {
                Pos::Unit => 0,
                Pos::Var(_) => 1,
                Pos::Prod(_) => 2,
                Pos::Mu(_, _) => 3,
                Pos::Opaque(_, _) => 4,
            }
        }
        fn key(p: &Pos) -> Vec<u8> {
            // Structural byte-key: sufficient for a deterministic total order.
            let mut k = vec![rank(p)];
            match p {
                Pos::Var(i) => k.extend(i.to_le_bytes()),
                Pos::Mu(i, b) => {
                    k.extend(i.to_le_bytes());
                    k.extend(key(b));
                }
                Pos::Prod(items) => {
                    for it in items {
                        k.extend(key(it));
                    }
                }
                Pos::Opaque(t, args) => {
                    k.extend(t.index().to_le_bytes());
                    for a in args {
                        k.extend(key(a));
                    }
                }
                Pos::Unit => {}
            }
            k
        }
        key(self).cmp(&key(other)).then(Ordering::Equal)
    }
}

impl PartialOrd for Pos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'input> TypeContext<'input> {
    /// βη-equivalence query over canonical trees (Pistone & Tranchini 2022
    /// §3): two types are equivalent iff their canonical forms are equal.
    ///
    /// Sound one-sided approximation only: the canonical forms realize the
    /// *partial* Fig. 4 axiomatization (currying/commutativity/unit/
    /// vacuous-binder laws), not full ε-theory.  Consumers needing
    /// ε-theory-strength must gate on parametricity (see the roadmap
    /// Stage 4 note in `yoneda.rs`).
    pub fn neg_beta_eta_equiv(&self, a: TypeId, b: TypeId) -> bool {
        self.to_neg(a) == self.to_neg(b)
    }

    /// Parametricity-gated variant of `neg_beta_eta_equiv` for consumers
    /// that use the result as an *equivalence judgment* (impl overlap,
    /// where-clause equivalence): returns `None` when the compilation unit
    /// has broken parametricity (`dyn Trait` reification), fail-closed.
    /// Diagnostics-only callers use `neg_beta_eta_equiv` directly.
    pub fn neg_beta_eta_equiv_gated(&self, a: TypeId, b: TypeId) -> Option<bool> {
        self.parametricity.equivalence_allowed()?;
        Some(self.neg_beta_eta_equiv(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> crate::ast::Span {
        crate::ast::Span::new(0, 0)
    }

    #[test]
    fn canonical_currying_reorder() {
        // (A×B)⇒C ≡ B⇒(A⇒C) modulo commutativity:
        // ((Int,Bool)⇒Unit) vs (Bool⇒(Int⇒Unit)) — same canonical form.
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let b = ctx.bool();
        let u = ctx.unit();
        let ib = ctx.tuple(smallvec![i, b]);
        let a1 = ctx.function(smallvec![ib], u);
        let inner = ctx.function(smallvec![i], u);
        let a2 = ctx.function(smallvec![b], inner);
        assert!(ctx.neg_beta_eta_equiv(a1, a2));
    }

    #[test]
    fn canonical_vacuous_forall_eliminated() {
        // ∀X.(Int⇒Unit) ≡ Int⇒Unit  (vacuous binder, Fig. 4a).
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let u = ctx.unit();
        let plain = ctx.function(smallvec![i], u);
        let quantified = ctx.forall(0, "X".into(), plain);
        assert!(ctx.neg_beta_eta_equiv(plain, quantified));
    }

    #[test]
    fn canonical_quantifier_hoisting() {
        // Int⇒(∀X.Unit) ≡ ∀X.(Int⇒Unit)  (A⇒∀X.B ≡ ∀X.A⇒B).
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let u = ctx.unit();
        let forall_u = ctx.forall(0, "X".into(), u);
        let lifted = ctx.function(smallvec![i], forall_u);
        let fn_i_u = ctx.function(smallvec![i], u);
        let hoisted = ctx.forall(0, "X".into(), fn_i_u);
        assert!(ctx.neg_beta_eta_equiv(lifted, hoisted));
    }

    #[test]
    fn canonical_non_equiv_fails_closed() {
        // Int⇒Bool vs Int⇒Int must NOT be equivalent.
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let b = ctx.bool();
        let f1 = ctx.function(smallvec![i], b);
        let f2 = ctx.function(smallvec![i], i);
        assert!(!ctx.neg_beta_eta_equiv(f1, f2));
    }

    #[test]
    fn canonical_unit_factor_dropped() {
        // (Int, Unit) ≡ Int  (A×1 ≡ A, Fig. 4b).
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let with_unit = ctx.tuple(smallvec![i, ctx.unit()]);
        assert_eq!(ctx.to_pos(with_unit), ctx.to_pos(i));
    }

    #[test]
    fn canonical_vacuous_mu_eliminated() {
        // μX.Int ≡ Int  (vacuous fixpoint, Fig. 4c).
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let m = ctx.alloc(TypeData::Mu {
            param_index: 7,
            param_name: "X".into(),
            body: i,
        });
        assert_eq!(ctx.to_pos(m), ctx.to_pos(i));
    }

    #[test]
    fn canonical_roundtrip_neg() {
        // from_neg(to_neg(t)) must produce a TypeId again (smoke test).
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let b = ctx.bool();
        let f = ctx.function(smallvec![i], b);
        let neg = ctx.to_neg(f);
        let _raised = ctx.from_neg(&neg);
    }
}
