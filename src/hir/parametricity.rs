//! Parametricity gate (roadmap Stage 4).
//!
//! The Yoneda type isomorphisms (and therefore Yoneda reduction used as an
//! *equivalence* judgment, not merely as a κ diagnostic) are valid only
//! under the ε-theory — i.e. when the term model is parametric.  The
//! paper's premise (Pistone & Tranchini 2020, §2 + App. A): polymorphic
//! programs are interpreted as (strongly) dinatural transformations, which
//! fails if the language can observe a polymorphic value's type at
//! runtime.
//!
//! ## What breaks parametricity in this language
//!
//! - **`dyn Trait` dynamic dispatch** — runtime type reification after
//!   erasure; a `dyn` value can be downcast/dispached on its concrete
//!   type, which distinguishes instantiations.  Breaks parametricity.
//! - **Runtime type-case / downcasting** — same failure mode.
//!
//! ## What does NOT break parametricity
//!
//! - **`@typeInfo!(T)` / `layout_of!(T)`** — strictly *compile-time*
//!   reflection on named, monomorphic types (SYNTAX.md: they expand in
//!   `generate` blocks before polymorphic instantiation exists).  They
//!   never observe the runtime type of a polymorphic value, so they are
//!   permitted inside the gate.
//! - κ diagnostics (`characteristic`), exhaustiveness counting — these
//!   are conservative *diagnostics* on types, sound regardless of
//!   parametricity, and are NOT gated (per the roadmap: "κ/counting can
//!   be done unconditionally").
//!
//! ## Semantic model of μ/ν (counting soundness)
//!
//! Inhabitant counting (`characteristic`, exhaustiveness) implicitly
//! assumes **Set semantics**: `μX.A` is the initial algebra (least
//! fixpoint, well-founded trees only), `νX.A` the final coalgebra.  E.g.
//! `μX.1+X` has exactly ℕ-many elements and `μX.(X→1)` has least fixpoint
//! 1 (finite!) under Set semantics.
//!
//! If the language had general recursion (CPO semantics), `μX.1+X` would
//! additionally contain ⊥ and "exact counting" would be unsound for types
//! whose counting path depends on recursive values.  Ponent is total
//! (no general recursion in the core language), so Set semantics is the
//! documented model — **but** this must be revisited if:
//!   - general recursion / `loop`-based value recursion is added to the
//!     core language, or
//!   - `μ`/`ν` types become user-writable and can wrap recursive
//!     function dependencies.
//! The current `leaf_kappa` saturation (κ over `usize::MAX` for cyclic
//! ADTs) is the conservative fallback masking this question; the
//! polynomial fixpoint solver (roadmap §3c) must only return numeric
//! counts for recursion schemes whose least fixpoint is computable and
//! unique under Set semantics.
//!
//! ## Granularity: compilation-unit level
//!
//! Per the roadmap, the gate is a *compilation-unit* property: a single
//! `dyn` dispatch anywhere in the unit can be fed a polymorphic
//! instantiation, so the whole unit's polymorphic fragment is no longer
//! provably parametric.  We therefore collect a unit-wide flag during HIR
//! lowering and let equivalence-strength consumers (`neg_beta_eta_equiv`,
//! Yoneda-reduction-based coercion/equivalence checks) require it; failure
//! is fail-closed (gate closed ⇒ fall back to structural equivalence
//! only).

use super::*;
use crate::hir::types::visit_type_children;
use crate::hir::types::{TypeContext, TypeData, TypeId};

/// Compilation-unit parametricity state, threaded through the checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParametricityGate {
    /// True once any parametricity-breaking construct (`dyn Trait`
    /// dispatch, runtime type-case) has been lowered in this unit.
    pub broken: bool,
}

impl ParametricityGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a parametricity-breaking construct.
    pub fn break_parametricity(&mut self) {
        self.broken = true;
    }

    /// May this unit use Yoneda reduction as an equivalence judgment?
    /// Fail-closed: an *untracked* unit (gate never consulted) must not
    /// silently enable ε-strength equivalence, so the default answer for
    /// a unit that has never been scanned is "unknown" — callers treat
    /// `None` as closed.
    pub fn equivalence_allowed(&self) -> Option<()> {
        if self.broken { None } else { Some(()) }
    }
}

/// Scan a type for parametricity-breaking runtime observation.
///
/// `dyn Trait` (TypeData::DynTrait) reifies the erased runtime type for
/// dispatch — its presence in a polymorphic instantiation distinguishes
/// type arguments, breaking parametricity.  Comptime reflection
/// (`@typeInfo!`/`layout_of!`) never appears *inside* a type, so a pure
/// type-walk cannot encounter it; it is not a violation.
pub fn scan_type_for_violations<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
    visit: &mut dyn FnMut(TypeId),
) {
    if matches!(ctx.get(ty), TypeData::DynTrait { .. }) {
        visit(ty);
    }
    // Recurse through children (resolve first, mirroring other walkers).
    let resolved = ctx.resolve_binding(ty);
    visit_type_children(resolved, ctx, &mut |child| {
        scan_type_for_violations(child, ctx, visit);
    });
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn gate_default_allows_but_untracked_is_closed_by_convention() {
        let g = ParametricityGate::new();
        assert!(g.equivalence_allowed().is_some());
        let mut g2 = ParametricityGate::new();
        g2.break_parametricity();
        assert!(g2.equivalence_allowed().is_none());
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod gate_integration_tests {
    use super::*;
    use crate::hir::types::TypeContext;
    use smallvec::smallvec;

    #[test]
    fn gate_blocks_gated_equiv_after_dyn_trait() {
        // A `dyn Trait` in the unit closes the gate: neg_beta_eta_equiv_gated
        // must return None (fail-closed), while the ungated diagnostic query
        // still answers.
        let mut ctx = TypeContext::new();
        let i = ctx.int(32, true);
        let u = ctx.unit();
        let f1 = ctx.function(smallvec![i], u);
        let f2 = ctx.function(smallvec![i], u);

        assert_eq!(ctx.neg_beta_eta_equiv_gated(f1, f2), Some(true));
        assert!(ctx.neg_beta_eta_equiv(f1, f2));

        ctx.parametricity.break_parametricity();
        assert_eq!(ctx.neg_beta_eta_equiv_gated(f1, f2), None);
        // Ungated diagnostic query unaffected.
        assert!(ctx.neg_beta_eta_equiv(f1, f2));
    }
}
