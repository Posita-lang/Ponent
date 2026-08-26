use super::*;

/// A fact established by a GADT `when` clause, scoped to one match arm.
/// Split so that existential opacity is enforced structurally rather than
/// by convention: only `ParamRefinement` is consulted by `resolve_binding`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GadtFact {
    /// Non-existential refinement: a refinable variable (`GenericParam` or
    /// a non-arm-local `InferVar`) maps to a concrete type.  Safe for
    /// `resolve_binding` to follow.
    ParamRefinement { from: TypeId, to: TypeId },
    /// An equality involving existential skolems (e.g. `[S] ~ [Int<32>]`).
    /// Never used as a rewrite rule — the witness must remain opaque
    /// (SYNTAX.md §"Existential Quantification").
    ///
    /// Read only via explicit `resolve_existential_witness` calls; never
    /// through `resolve_binding` (which would transparently expose the
    /// witness and break opacity).  It is retained because it (a) makes
    /// opacity a STRUCTURAL guarantee — the compiler cannot accidentally
    /// rewrite an existential equation, since `resolve_gadt_eq` only
    /// matches `ParamRefinement` — and (b) is the future carrier for the
    /// existential elimination rules (branch-boundary occurs-check,
    /// length/projection rules, injectivity checks).
    ExistentialEquation { lhs: TypeId, rhs: TypeId },
}

/// One existential scope frame: the variant it was created for (identified
/// by enum DefId AND variant name — occurrence identity) and the skolems
/// allocated for that variant, indexed by binder position in
/// `exists_params` (GHC `realUnique` / OCaml `id: int` identity).
/// `check_pattern_inner` reuses the frame that `precreate_exist_skolems`
/// pushed for the SAME top-level variant occurrence (same DefId + name +
/// not yet consumed); nested existential variants, same-named variants in
/// DIFFERENT enums, and recursive variants (frame already `used`) all push
/// their own frame.
#[derive(Debug, Clone)]
pub(crate) struct ExistScopeFrame {
    pub(crate) def_id: DefId,
    pub(crate) variant_name: Symbol,
    /// Whether the top-level `check_pattern_inner` has already consumed
    /// this frame.  Once consumed, a recursive re-encounter of the same
    /// (DefId, variant) must push a fresh frame instead of reusing.
    pub(crate) used: bool,
    /// Skolems for each `exists` binder, in `exists_params` order.
    pub(crate) skolems: Vec<TypeId>,
}

/// One inner GADT pattern `when` equality collected during
/// `check_pattern_inner` (which runs BEFORE `push_gadt_arm`).  Registered
/// by `apply_gadt_refinement` after the arm is pushed, so the GADT fact
/// registry has an active arm to write into — nested GADT refinement
/// (SYNTAX.md §"Nested GADT Refinement").
#[derive(Debug, Clone)]
pub(crate) struct PendingInnerGadtEq<'input> {
    pub(crate) param_name: Symbol,
    pub(crate) concrete_ty: crate::ast::Type<'input>,
    pub(crate) binding: crate::hir::symbol::TypeBinding<'input>,
    pub(crate) args: Vec<TypeId>,
    pub(crate) exist_params: Vec<Symbol>,
    pub(crate) skolems: Vec<TypeId>,
}

/// All GADT-related state for the type checker, aggregated into a single
/// structure so that arm entry/exit is ATOMIC and the invariant "stack
/// depth == registry depth" can be verified locally instead of across
/// scattered fields (`TypeChecker::current_gadt_exist_skolems` +
/// `pending_inner_gadt_eqs` and `TypeContext::{gadt_facts,gadt_arm_depth}`
/// were previously four unrelated mutable fields with implicit ordering
/// constraints).
///
/// Exists refinement — three mechanical rules:
///   Rule 1 — `when X == ConcreteType`: X resolves to `ConcreteType` and is
///     REMOVED from the facts; every use of X ≡ a use of `ConcreteType`.
///   Rule 2 — `when T == Expr<X₁, X₂, ...>`: T is refined to the compound
///     type; the Xᵢ stay opaque (registered as inert `ExistentialEquation`s).
///   Rule 3 — `when X == Y` (equality between TWO exists vars): X ≡ Y, both
///     opaque, interchangeable.  No dedicated syntax: the common case (two
///     payload components sharing a type) is expressed by REUSING the same
///     exists variable (`MkPair(exists A: (A, A))` — the same skolem
///     appearing twice unifies to equivalence).  Explicit cross-variant
///     exists equivalence is reserved for future use cases.
pub(crate) struct GadtContext<'input> {
    /// The GADT fact registry: a stack of per-arm fact lists.
    pub(crate) facts: RefCell<Vec<Vec<GadtFact>>>,
    /// Depth counter for `facts`, kept in sync with its length.
    pub(crate) arm_depth: Cell<usize>,
    /// Per-variant existential skolem scope stack (occurrence identity).
    pub(crate) exist_skolems: RefCell<Vec<ExistScopeFrame>>,
    /// Inner GADT `when` equalities collected before `push_gadt_arm`.
    pub(crate) pending_eqs: Vec<PendingInnerGadtEq<'input>>,
}

impl<'input> GadtContext<'input> {
    pub(crate) fn new() -> Self {
        GadtContext {
            facts: RefCell::new(Vec::new()),
            arm_depth: Cell::new(0),
            exist_skolems: RefCell::new(Vec::new()),
            pending_eqs: Vec::new(),
        }
    }

    /// Enter a new GADT arm: push a fresh fact list and bump the depth.
    pub(crate) fn enter_arm(&self) {
        self.facts.borrow_mut().push(Vec::new());
        self.arm_depth.set(self.arm_depth.get() + 1);
    }

    /// Exit the current GADT arm, discarding its equalities.
    pub(crate) fn exit_arm(&self) {
        self.facts.borrow_mut().pop();
        let d = self.arm_depth.get();
        self.arm_depth.set(d.saturating_sub(1));
    }

    /// Register a GADT **param refinement** within the current arm.
    pub(crate) fn register_param_refinement(&self, from: TypeId, to: TypeId) {
        if let Some(arm) = self.facts.borrow_mut().last_mut() {
            arm.push(GadtFact::ParamRefinement { from, to });
        }
    }

    /// Register an **inert existential equation** within the current arm.
    pub(crate) fn register_existential_equation(&self, lhs: TypeId, rhs: TypeId) {
        if let Some(arm) = self.facts.borrow_mut().last_mut() {
            arm.push(GadtFact::ExistentialEquation { lhs, rhs });
        }
    }

    /// Push an existential scope frame (occurrence identity).
    /// Collect one inner `when` equality for later registration.
    pub(crate) fn push_pending_eq(&mut self, eq: PendingInnerGadtEq<'input>) {
        self.pending_eqs.push(eq);
    }

    /// Take the collected inner `when` equalities (clearing the queue).
    pub(crate) fn take_pending_eqs(&mut self) -> Vec<PendingInnerGadtEq<'input>> {
        std::mem::take(&mut self.pending_eqs)
    }
}
