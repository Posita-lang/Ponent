use super::*;
use crate::sync::RwLock;
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;

/// A fact established by a GADT `when` clause, scoped to one match arm.
/// Split so that existential opacity is enforced structurally rather than
/// by convention: only `ParamRefinement` is consulted by `resolve_binding`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GadtFact {
    /// Non-existential refinement: a refinable variable (`GenericParam` or
    /// a non-arm-local `InferVar`) maps to a concrete type.  Safe for
    /// `resolve_binding` to follow.
    /// (OutsideIn(X) §3.2: implication constraint resolution — the constraint
    /// is treated as an assumption rather than a global binding.)
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

// ══ Ambivalent types — the GADT solving core ═══════════════════════════
// (Garrigue & Rémy, "Ambivalent types for principal type inference with
// GADTs," APLAS 2013, LNCS 8301 pp.257–272; HAL hal-00914560)
//
// Replaces the PLDI'06 Rigid/Wobbly gate (the historical role of
// `data.rs::Rigidity`):
//  • M-USE (Fig. 3) has NO rigidity premise — the equation τ₁≐τ₂ enters the
//    branch context, and any variable may be unified through it;
//  • a node's ambivalent set ψ (§3.1, ζ ::= ψ^α) widens only when
//    unification actually CONSUMES an equation, and the widening lands on a
//    FRESH label (f's `1:int` stays a singleton — §2: "nothing forces us to
//    use this equation");
//  • the ambiguity boundary is the branch EXIT: an escaping node's ψ must be
//    entailed by the enclosing equations (Definition 1 plus M-USE's
//    well-formedness requirement on ζ₂), otherwise the arm is rejected —
//    restoring principality and monotonicity (§4).
//
// Ambivalence is pure metadata (a per-node ledger keyed by TypeId) and never
// enters `TypeData`, so the Yoneda pipeline and the OmniML solver only ever
// see ordinary types; the ledger frame is discarded with the arm (§4:
// "ambivalent types … are dropped during the translation").
/// Per-arm ledger: node label → the raw type set ψ assumed interchangeable
/// at that node (§3.1).
pub(crate) type AmbivalentLedger = HashMap<TypeId, SmallVec<[TypeId; 4]>>;

/// Branch-exit rejection: an ambivalent type escaped the scope of an
/// equation it depends on — §2: "an ambiguity arises only when an ambivalent
/// type becomes incoherent by escaping the scope of an equation it depends
/// on".  The checker renders it as an ambiguity diagnostic at the match span.
#[derive(Debug, Clone)]
pub struct AmbiguityError {
    /// The escaping node (a widened ambivalent label).
    pub node: TypeId,
    /// ψ — the elements that cannot be proved equal outside the branch.
    pub set: Vec<TypeId>,
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
    pub(crate) args: smallvec::SmallVec<[crate::hir::types::TypeId; 4]>,
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
/// ## Theoretical basis
/// - **Skolemization and implication constraints:**
///   Vytiniotis et al., "OutsideIn(X): Modular Type Inference with Local
///   Assumptions," *Journal of Functional Programming* 21(4-5), 2011.
///   The per-arm `facts` stack and `GADT_SKOLEM_UNIVERSE` sentinel implement
///   OutsideIn(X)'s mechanism for preventing local GADT type equalities from
///   escaping their pattern-match scope.
/// - **Rigid vs Wobbly types:**
///   Peyton Jones et al., "Simple unification-based type inference for GADTs,"
///   POPL 2006.  Rigid annotations (introduced by programmer-supplied type
///   signatures) restrict refinement to typed-bound variables; bare inferred
///   bindings remain wobbly and receive no refinement (PLDI'06 §3 order-
///   insensitivity guarantee).
/// - **Existential opacity:**
///   Rule 3 of the existential refinement discipline — the same skolem
///   appearing twice unifies to equivalence (`MkPair(exists A: (A, A))`) —
///   corresponds to the parametricity argument for existential packages
///   (Abadi et al., "A Theory of Objects," 1996).
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
    pub(crate) facts: RwLock<Vec<Vec<GadtFact>>>,
    /// Depth counter for `facts`, kept in sync with its length.
    pub(crate) arm_depth: AtomicUsize,
    /// Per-arm index for O(1) ParamRefinement lookup.  The innermost
    /// arm (last entry) is consulted first by `resolve_gadt_eq`.
    pub(crate) param_refinements: RwLock<Vec<HashMap<TypeId, TypeId>>>,
    /// Per-variant existential skolem scope stack (occurrence identity).
    pub(crate) exist_skolems: RwLock<Vec<ExistScopeFrame>>,
    /// Inner GADT `when` equalities collected before `push_gadt_arm`.
    pub(crate) pending_eqs: Vec<PendingInnerGadtEq<'input>>,
    /// Per-arm ambivalent ledgers (§3.1), frame i ↔ arm depth i.  `exit_arm`
    /// drops the frame: after the exit check the remaining sets are entailed
    /// by the enclosing context, so discarding them is the paper's
    /// translation step (§4: "ambivalent types … are dropped during the
    /// translation").
    pub(crate) ambivalent: RwLock<Vec<AmbivalentLedger>>,
    /// Exit-entailment window: arms with index ≥ floor are invisible to
    /// `resolve_gadt_eq`, so the coherence check resolves under the enclosing
    /// equations (M-USE's Γ, not Γ, τ₁ ≐ τ₂).  `usize::MAX` = no suppression
    /// (the default).
    pub(crate) entailment_floor: AtomicUsize,
}

impl<'input> GadtContext<'input> {
    pub(crate) fn new() -> Self {
        GadtContext {
            facts: RwLock::new(Vec::new()),
            arm_depth: AtomicUsize::new(0),
            param_refinements: RwLock::new(Vec::new()),
            exist_skolems: RwLock::new(Vec::new()),
            pending_eqs: Vec::new(),
            ambivalent: RwLock::new(Vec::new()),
            entailment_floor: AtomicUsize::new(usize::MAX),
        }
    }

    /// Enter a new GADT arm: push a fresh fact list and bump the depth.
    pub(crate) fn enter_arm(&self) {
        self.facts.write().push(Vec::new());
        self.param_refinements.write().push(HashMap::default());
        // Push the arm's ambivalent ledger in lockstep with the fact stack
        // (§3.1: ambivalence is per-arm state, dropped at the arm boundary).
        self.ambivalent.write().push(AmbivalentLedger::default());
        self.arm_depth.store(
            self.arm_depth.load(AtomicOrdering::Relaxed) + 1,
            AtomicOrdering::Relaxed,
        );
        debug_assert_eq!(
            self.facts.read().len(),
            self.arm_depth.load(AtomicOrdering::Relaxed),
            "GadtContext invariant violated: facts stack depth != arm_depth after enter_arm"
        );
    }

    /// Exit the current GADT arm, discarding its equalities.
    pub(crate) fn exit_arm(&self) {
        self.facts.write().pop();
        self.param_refinements.write().pop();
        // Drop the arm's ambivalent ledger (§4: the sets are entailed by the
        // enclosing equations at this point, so discarding them is the
        // paper's translation step, not a loss of information).
        self.ambivalent.write().pop();
        self.entailment_floor
            .store(usize::MAX, AtomicOrdering::Relaxed);
        let d = self.arm_depth.load(AtomicOrdering::Relaxed);
        self.arm_depth
            .store(d.saturating_sub(1), AtomicOrdering::Relaxed);
        debug_assert_eq!(
            self.facts.read().len(),
            self.arm_depth.load(AtomicOrdering::Relaxed),
            "GadtContext invariant violated: facts stack depth != arm_depth after exit_arm"
        );
    }

    /// Register a GADT **param refinement** within the current arm.
    pub(crate) fn register_param_refinement(&self, from: TypeId, to: TypeId) {
        if let Some(arm) = self.facts.write().last_mut() {
            arm.push(GadtFact::ParamRefinement { from, to });
        }
        if let Some(map) = self.param_refinements.write().last_mut() {
            map.insert(from, to);
        }
    }

    /// Register an **inert existential equation** within the current arm.
    pub(crate) fn register_existential_equation(&self, lhs: TypeId, rhs: TypeId) {
        if let Some(arm) = self.facts.write().last_mut() {
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

    /// ψ(node) in the current arm (`None` ⇒ the implicit singleton `{node}`,
    /// §3.1's `{t}^t`).  Read-only: the ledger is pure metadata, so a query never
    /// mutates inference state.
    pub(crate) fn set_of_current(&self, node: TypeId) -> Option<SmallVec<[TypeId; 4]>> {
        self.ambivalent
            .read()
            .last()
            .and_then(|l| l.get(&node).cloned())
    }

    /// Record ψ for `node` in the current arm.  The caller enforces
    /// WF-TYPE-AMBIVALENT's |ψ \\ Vr| ≤ 1 (Fig. 1); see
    /// `TypeContext::mint_widened_label`.
    pub(crate) fn widen_current(&self, node: TypeId, set: SmallVec<[TypeId; 4]>) {
        if let Some(l) = self.ambivalent.write().last_mut() {
            l.insert(node, set);
        }
    }

    /// M-ANN (§3.5: "sound to decrease them explicitly, using a type
    /// annotation"): replace ψ(node) with the singleton {τ}.
    pub(crate) fn narrow_current(&self, node: TypeId, tau: TypeId) {
        if let Some(l) = self.ambivalent.write().last_mut() {
            if l.contains_key(&node) {
                l.insert(node, smallvec::smallvec![tau]);
            }
        }
    }

    /// Snapshot of the current arm's ledger entries.  A widened set lives on a
    /// fresh label β with `β → root`, so callers that know only the resolved
    /// type (e.g. M-ANN's annotation type) must scan the (tiny) frame to find
    /// the label; see `TypeContext::narrow_ambivalent`.
    pub(crate) fn current_entries(&self) -> Vec<(TypeId, SmallVec<[TypeId; 4]>)> {
        self.ambivalent
            .read()
            .last()
            .map(|l| l.iter().map(|(&k, v)| (k, v.clone())).collect())
            .unwrap_or_default()
    }

    /// Equation pairs of every arm with index < floor — the enclosing
    /// context Γ at exit depth `floor`.  Existential equations are inert and
    /// never participate (opacity preserved).
    ///
    /// The read guard is dropped before returning, so callers may re-enter
    /// `resolve_gadt_eq` (which takes the same lock) without nesting reads.
    pub(crate) fn equations_below(&self, floor: usize) -> Vec<(TypeId, TypeId)> {
        let refs = self.param_refinements.read();
        let mut out = Vec::new();
        for (i, map) in refs.iter().enumerate() {
            if i >= floor {
                break;
            }
            for (&f, &t) in map.iter() {
                out.push((f, t));
            }
        }
        out
    }
}
