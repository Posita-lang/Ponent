use crate::ast::Span;
use crate::hir::shape_var::{
    ShapeVarContext, ShapeVarId, TypeShape, shapes_compatible, type_data_to_shape,
};
use crate::hir::smt::SmtSolver;
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use rustc_hash::FxHashMap as HashMap;
use std::collections::BinaryHeap;

/// Priority wrapper for constraints, enabling BinaryHeap-based sorting.
/// Constraints are processed in order of "determinism":
///   Priority 0: Eq(concrete, concrete) — both sides fully resolved
///   Priority 1: Eq(concrete, infer)    — one side is InferVar
///   Priority 2: Eq(infer, infer)       — both sides are InferVar
///   Priority 3: Sub(concrete, concrete)
///   Priority 4: Sub(concrete, infer) / Sub(infer, concrete)
///   Priority 5: Sub(infer, infer)
///   Priority 6: Impl constraints
#[derive(Debug, Clone)]
struct PrioritizedConstraint {
    priority: u8,
    constraint: Constraint,
}

impl PartialEq for PrioritizedConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedConstraint {}

impl PartialOrd for PrioritizedConstraint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // BinaryHeap is a max-heap, so reverse for min-priority behavior
        other.priority.partial_cmp(&self.priority)
    }
}

impl Ord for PrioritizedConstraint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
    }
}

/// Generalization state for an inference variable (OmniML §3.2).
/// Controls whether a variable can be generalized (let-polymorphism).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenState {
    /// Not yet generalized (free in current scope).
    Ungeneralized,
    /// Fully generalized (let-bound, can be instantiated arbitrarily).
    Generalized,
    /// Partially generalized — awaiting suspended constraints to resolve.
    PartialGeneralized,
    /// Partially instantiated — an instance of a PG variable.
    PartialInstance(usize), // id of the PG variable
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeVariableKind {
    Unconstrained,
    Integer,
    Float,
    Numeric,
    Bool,
    Any,
}

/// The principal shape of a type variable (OmniML-inspired).
/// Tracks what "shape" the type is known to have, even before
/// the concrete type is fully resolved. This enables suspended
/// match constraints to determine when they can be discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrincipalShape {
    /// Unknown — no shape information yet.
    Unknown,
    /// Scalar — integer, float, char, byte, bool, rational.
    Scalar,
    /// Function type: τ₁ → τ₂
    Arrow,
    /// Tuple type: τ₁ × τ₂ × ...
    Tuple(usize),
    /// Named constructor: C τ₁ τ₂ ...
    Constructor(usize),
    /// Polymorphic type: ∀α. τ
    Poly,
    /// InferVar or unresolved
    Var,
}

#[derive(Debug, Clone)]
pub struct TypeVar {
    pub id: usize,
    pub kind: TypeVariableKind,
    /// Generalization state for let-polymorphism (OmniML-inspired).
    pub gen_state: GenState,
    /// Principal shape (OmniML): tracks what shape this variable has
    /// been determined to be, enabling shape-based constraint suspension.
    pub shape: PrincipalShape,
    /// Level of this type variable (Fan, Xu & Xie 2025 §4).
    /// Lower levels are "outer" scopes; higher levels are "inner" scopes.
    /// Variables at level n+1 inside a let can be generalized.
    /// Promotion adjusts levels when unifying variables at different levels.
    pub level: usize,
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(TypeId, TypeId, Span),
    Sub(TypeId, TypeId, Span),
    Impl(TypeId, DefId, Span),
    /// OmniML suspended match constraint (O'Brien, Rémy & Scherer §4.1):
    /// `match τ with patterns` — suspends until the shape of τ is known.
    /// When τ resolves to a concrete type, the match is discharged.
    Match {
        /// The type whose shape must be determined.
        scrutinee: TypeId,
        /// Index into the inference context's `match_branches` table.
        branches_id: usize,
        span: Span,
    },
    /// OmniML existential: `∃α. C` — bind a fresh flexible type variable.
    Exists {
        var_id: usize,
        constraint: Box<Constraint>,
        span: Span,
    },
    /// OmniML universal: `∀α. C` — bind a fresh rigid (skolem) variable.
    Forall {
        var_id: usize,
        constraint: Box<Constraint>,
        span: Span,
    },
    /// OmniML instance: instantiate a generalized scheme at a type.
    /// `x[τ]` — the scheme for `x` is instantiated with `τ`.
    Instance {
        /// The expression variable to instantiate.
        expr_var: String,
        /// The type to instantiate at.
        instantiation_ty: TypeId,
        span: Span,
    },
    /// OmniML let-constraint: `let x = λα.∃ᾱ. C₁ in C₂`
    Let {
        expr_var: String,
        def_constraint: Box<Constraint>,
        body_constraint: Box<Constraint>,
        span: Span,
    },
}

impl Constraint {
    /// Compute priority: lower = more deterministic, processed first.
    /// This enables BinaryHeap-based scheduling where concrete-concrete
    /// constraints are resolved before those involving inference variables.
    pub fn priority(&self, ctx: &TypeContext) -> u8 {
        match self {
            Constraint::Eq(a, b, _) => {
                let a_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*a)), TypeData::InferVar { .. });
                let b_is_infer =
                    matches!(ctx.get(ctx.resolve_binding(*b)), TypeData::InferVar { .. });
                match (a_is_infer, b_is_infer) {
                    (false, false) => 0,                // concrete-concrete: highest priority
                    (true, false) | (false, true) => 1, // one infer var
                    (true, true) => 2,                  // both infer vars
                }
            }
            Constraint::Sub(sub, sup, _) => {
                let sub_is_infer = matches!(
                    ctx.get(ctx.resolve_binding(*sub)),
                    TypeData::InferVar { .. }
                );
                let sup_is_infer = matches!(
                    ctx.get(ctx.resolve_binding(*sup)),
                    TypeData::InferVar { .. }
                );
                match (sub_is_infer, sup_is_infer) {
                    (false, false) => 3,
                    _ => 4,
                }
            }
            Constraint::Impl(..) => 5, // trait impl checks: lowest priority
            Constraint::Match { scrutinee, .. } => {
                // Match constraints: low priority — they suspend until the
                // scrutinee's shape is resolved.
                let resolved = ctx.resolve_binding(*scrutinee);
                if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    6 // still an infer var → needs resolution first
                } else {
                    3 // resolved → medium priority
                }
            }
            Constraint::Exists { .. } => 2, // exists: medium-high priority
            Constraint::Forall { .. } => 6, // forall: low priority (skolem)
            Constraint::Instance { .. } => 1, // instance: high priority
            Constraint::Let { .. } => 2,    // let: medium-high priority
        }
    }
}

/// Generalization status for an inference variable (OmniML §6).
/// Tracks whether a variable can be fully generalized, partially generalized,
/// or is still waiting for suspended constraints to be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenStatus {
    /// (I) Initial — not yet generalized
    Ungeneralized,
    /// (G) Fully generalized — safe to copy instances
    Generalized,
    /// (PG) Partially generalizable — guarded by a suspended constraint
    PartiallyGeneralizable,
    /// (PI) Partial instance — previously PG but has been updated; needs re-generalization
    PartialInstance,
}

#[derive(Debug)]
pub struct InferenceContext {
    type_vars: Vec<TypeVar>,
    var_type_ids: Vec<TypeId>,
    constraints: Vec<Constraint>,
    next_var_id: usize,
    /// Current typing level (Fan, Xu & Xie 2025 §4).
    /// Incremented on entering let/forall/region scope.
    /// Variables at level > current_level can be generalized.
    /// Skolem escape check: a var at higher level cannot be bound
    /// to a concrete type at lower level without promotion.
    pub current_level: usize,
    /// Per-variable lower bounds (subtypes that must be ≤ this variable).
    /// lower_bounds[i] contains TypeIds that must be subtypes of variable i.
    lower_bounds: Vec<Vec<TypeId>>,
    /// Per-variable upper bounds (supertypes that this variable must be ≤).
    /// upper_bounds[i] contains TypeIds that variable i must be a subtype of.
    upper_bounds: Vec<Vec<TypeId>>,
    /// Per-variable wait lists (OmniML §3.2): constraints suspended on this var.
    /// When the var is bound (unified with a concrete type), these constraints
    /// are woken and reprocessed, enabling bidirectional type information flow.
    wait_lists: Vec<Vec<Constraint>>,
    /// Guard sets (OmniML §6): for each InferVar, which constraints (by index)
    /// reference it in a suspended match.  A non-empty guard set means the
    /// variable is PG (PartiallyGeneralizable).  When all guards are discharged,
    /// the variable can become G (Generalized).
    guard_sets: Vec<Vec<usize>>,
    /// Per-variable generalisation status (I / G / PG / PI).
    gen_statuses: Vec<GenStatus>,
    /// Shape variable context (OmniML §3.3, §6).
    /// Manages shape variables — first-class unifiable variables that represent
    /// not-yet-known principal shapes.  When a shape variable is resolved,
    /// all suspended match constraints waiting on it are woken.
    pub shape_vars: ShapeVarContext,
    /// Match branches table (OmniML §4.1): each SuspendedMatch constraint
    /// references a set of branch patterns by index.
    /// A branch is (label, expected_pattern) — discharged when the scrutinee's
    /// principal shape matches the pattern.
    match_branches: Vec<MatchBranchSet>,
    /// Forward references for incremental instantiation (OmniML §5.2):
    /// Maps a PG variable id to the list of instance variable ids that
    /// were created from it. When the PG var is refined, all instances
    /// are updated.
    forward_refs: Vec<Vec<usize>>,
    /// Reverse of forward_refs: for each instance, which PG var it came from.
    reverse_refs: Vec<Option<usize>>,
    /// Tracks which InferVar ids have been unified since the last
    /// `force_generalize` call, enabling incremental processing.
    dirty_set: std::collections::HashSet<usize>,
    /// Dirty region levels from TypeChecker's RegionTree.
    pub region_dirty_levels: Vec<usize>,
    /// Per-InferenceContext resolution table (TypeOrVar pattern).
    resolutions: Vec<Option<TypeId>>,
    /// Local bindings for GenericParam indices during instantiation.
    generic_param_bindings: HashMap<usize, TypeId>,
}

/// A set of pattern alternatives for a suspended match constraint.
#[derive(Debug, Clone)]
pub struct MatchBranchSet {
    /// The pattern label (e.g. "Arrow", "Tuple", "Coproduct", etc.).
    /// When the scrutinee's shape matches this, the branch is taken.
    pub shape_pattern: PrincipalShape,
    /// Continuation constraints to add when this branch matches.
    pub continuation: Vec<Constraint>,
    /// Fallback constraints to add when no branch matches (else_).
    /// Used as a default when the shape cannot be determined uniquely.
    /// The system emits a diagnostic when else_ is triggered.
    pub else_continuation: Vec<Constraint>,
}

impl InferenceContext {
    pub fn new() -> Self {
        InferenceContext {
            type_vars: Vec::new(),
            var_type_ids: Vec::new(),
            constraints: Vec::new(),
            next_var_id: 0,
            current_level: 0,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            wait_lists: Vec::new(),
            guard_sets: Vec::new(),
            gen_statuses: Vec::new(),
            shape_vars: ShapeVarContext::new(),
            match_branches: Vec::new(),
            forward_refs: Vec::new(),
            reverse_refs: Vec::new(),
            dirty_set: std::collections::HashSet::new(),
            region_dirty_levels: Vec::new(),
            resolutions: Vec::new(),
            generic_param_bindings: HashMap::default(),
        }
    }

    pub fn new_type_var(&mut self, ctx: &mut TypeContext, kind: TypeVariableKind) -> TypeId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        let ty_id = ctx.alloc_infer_var(id);
        if id >= self.resolutions.len() {
            self.resolutions.resize(id + 1, None);
        }
        self.type_vars.push(TypeVar {
            id,
            kind,
            gen_state: GenState::Ungeneralized,
            shape: PrincipalShape::Unknown,
            level: self.current_level,
        });
        self.var_type_ids.push(ty_id);
        // Grow bounds vectors to match the new variable id
        while self.lower_bounds.len() <= id {
            self.lower_bounds.push(Vec::new());
        }
        while self.upper_bounds.len() <= id {
            self.upper_bounds.push(Vec::new());
        }
        while self.wait_lists.len() <= id {
            self.wait_lists.push(Vec::new());
        }
        while self.guard_sets.len() <= id {
            self.guard_sets.push(Vec::new());
        }
        while self.gen_statuses.len() <= id {
            self.gen_statuses.push(GenStatus::Ungeneralized);
        }
        while self.forward_refs.len() <= id {
            self.forward_refs.push(Vec::new());
        }
        while self.reverse_refs.len() <= id {
            self.reverse_refs.push(None);
        }
        ty_id
    }

    /// Resolve a TypeId through inference variable bindings (TypeOrVar pattern).
    /// Follows the chain of resolutions until a concrete type is found.
    pub fn resolve(&self, ty: TypeId, ctx: &TypeContext) -> TypeId {
        let mut current = ty;
        loop {
            match ctx.get(current) {
                TypeData::InferVar { id } => {
                    if *id < self.resolutions.len() {
                        if let Some(resolved) = self.resolutions[*id] {
                            if resolved == current {
                                return current;
                            }
                            current = resolved;
                            continue;
                        }
                    }
                    return current;
                }
                _ => return current,
            }
        }
    }

    /// Unify with local InferVar resolution (TypeOrVar pattern).
    /// Records resolutions in `self.resolutions` instead of global bindings.
    pub fn unify(
        &mut self,
        a: TypeId,
        b: TypeId,
        ctx: &mut TypeContext,
    ) -> Result<TypeId, TypeError> {
        let ra = self.resolve(a, ctx);
        let rb = self.resolve(b, ctx);
        if ra == rb {
            return Ok(ra);
        }
        match (ctx.get(ra), ctx.get(rb)) {
            (TypeData::InferVar { id }, _) if *id < self.resolutions.len() => {
                self.resolutions[*id] = Some(rb);
                Ok(rb)
            }
            (_, TypeData::InferVar { id }) if *id < self.resolutions.len() => {
                self.resolutions[*id] = Some(ra);
                Ok(ra)
            }
            _ => ctx.unify(ra, rb),
        }
    }

    /// Look up the kind of a type variable by its id.
    pub fn get_var_kind(&self, id: usize) -> Option<TypeVariableKind> {
        self.type_vars
            .iter()
            .find(|tv| tv.id == id)
            .map(|tv| tv.kind)
    }

    /// Get the level of a type variable by its id.
    pub fn get_var_level(&self, id: usize) -> Option<usize> {
        self.type_vars
            .iter()
            .find(|tv| tv.id == id)
            .map(|tv| tv.level)
    }

    /// Enter a deeper typing scope (let/forall/region).
    /// All new variables created within this scope will have the new level.
    /// Returns the previous level so caller can restore it.
    pub fn enter_level(&mut self) -> usize {
        let prev = self.current_level;
        self.current_level += 1;
        prev
    }

    /// Exit the current typing scope, restoring the previous level.
    pub fn exit_level(&mut self, prev_level: usize) {
        self.current_level = prev_level;
    }

    /// Try to promote a variable at `from_level` to `target_level`
    /// by creating a new variable at the target level and unifying.
    /// Returns true if promotion succeeded.
    /// This implements the level-based promotion mechanism from
    /// Fan, Xu & Xie 2025 §6 (rule PR-UVARPR).
    pub fn try_promote_var(
        &mut self,
        ctx: &mut TypeContext,
        var_id: usize,
        target_level: usize,
    ) -> Option<TypeId> {
        let var_level = self.get_var_level(var_id)?;
        if var_level <= target_level {
            // No promotion needed — the var is already at an appropriate level
            return Some(self.var_type_ids[var_id]);
        }
        // Create a new variable at the target level
        let new_ty_id = self.new_type_var(ctx, TypeVariableKind::Any);
        // Find the new var's id (it's the last pushed)
        let new_id = self.next_var_id - 1;
        if let Some(tv) = self.type_vars.iter_mut().find(|tv| tv.id == new_id) {
            tv.level = target_level;
        }
        // Bind the old variable to the new one (promotion)
        if var_id < self.resolutions.len() {
            self.resolutions[var_id] = Some(new_ty_id);
        }
        Some(new_ty_id)
    }

    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    /// OmniML-inspired: suspend a constraint on the target InferVar id.
    /// When the var is bound, the constraint will be woken and reprocessed.
    /// Also marks the variable as PartiallyGeneralizable (PG).
    pub fn suspend_on_var(&mut self, c: Constraint, var_id: usize) {
        if var_id < self.wait_lists.len() {
            self.wait_lists[var_id].push(c);
            if var_id < self.gen_statuses.len() {
                self.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            }
        } else {
            self.constraints.push(c);
        }
    }

    /// Extract the InferVar id from a constraint (if any side is an unresolved InferVar).
    /// Returns None if both sides are concrete.
    fn infer_var_from_constraint(&self, c: &Constraint, ctx: &TypeContext) -> Option<usize> {
        match c {
            Constraint::Eq(a, b, _) => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if let TypeData::InferVar { id } = ctx.get(ra) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id } = ctx.get(rb) {
                    return Some(*id);
                }
                None
            }
            Constraint::Sub(sub, sup, _) => {
                let rs = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                if let TypeData::InferVar { id } = ctx.get(rs) {
                    return Some(*id);
                }
                if let TypeData::InferVar { id } = ctx.get(rsup) {
                    return Some(*id);
                }
                None
            }
            Constraint::Impl(ty, ..) => {
                let r = ctx.resolve_binding(*ty);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Match { scrutinee, .. } => {
                let r = ctx.resolve_binding(*scrutinee);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Exists { var_id, .. } => Some(*var_id),
            Constraint::Forall { .. } => None, // rigid var — not an infer var
            Constraint::Instance {
                instantiation_ty, ..
            } => {
                let r = ctx.resolve_binding(*instantiation_ty);
                if let TypeData::InferVar { id } = ctx.get(r) {
                    Some(*id)
                } else {
                    None
                }
            }
            Constraint::Let { .. } => None, // structural — no single infer var
        }
    }

    /// Wake all constraints suspended on the given var_id, moving them
    /// back into the active constraint list for reprocessing.
    fn wake_var(&mut self, var_id: usize) {
        if var_id < self.wait_lists.len() {
            let mut suspended = std::mem::take(&mut self.wait_lists[var_id]);
            self.constraints.append(&mut suspended);
        }
    }

    /// Determine the principal shape of a resolved type.
    pub fn shape_of_type(ctx: &TypeContext, ty: TypeId) -> PrincipalShape {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::Fn { params, .. } => PrincipalShape::Arrow,
            TypeData::Tuple { elems } => PrincipalShape::Tuple(elems.len()),
            TypeData::App { args, .. } => PrincipalShape::Constructor(args.len()),
            TypeData::Struct { .. } | TypeData::Enum { .. } => PrincipalShape::Constructor(0),
            TypeData::Forall { .. } | TypeData::Exists { .. } | TypeData::Poly { .. } => {
                PrincipalShape::Poly
            }
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Rational { .. } => PrincipalShape::Scalar,
            TypeData::InferVar { .. } | TypeData::GenericParam { .. } => PrincipalShape::Var,
            _ => PrincipalShape::Unknown,
        }
    }

    /// Try to set the shape of a variable from its resolved type.
    /// Returns true if the shape was updated.
    fn try_set_shape(&mut self, var_id: usize, ctx: &TypeContext) -> bool {
        if var_id < self.type_vars.len() && var_id < self.var_type_ids.len() {
            let ty = ctx.resolve_binding(self.var_type_ids[var_id]);
            if !matches!(ctx.get(ty), TypeData::InferVar { .. }) {
                let new_shape = Self::shape_of_type(ctx, ty);
                if self.type_vars[var_id].shape != new_shape {
                    self.type_vars[var_id].shape = new_shape;
                    return true;
                }
            }
        }
        false
    }

    /// Incrementally wake constraints for a resolved variable.
    /// Woken constraints are enqueued directly onto the heap.
    /// After waking, if the wait list is empty, the variable can be re-generalised (G).
    fn wake_var_incremental(
        &mut self,
        var_id: usize,
        heap: &mut BinaryHeap<PrioritizedConstraint>,
        ctx: &TypeContext,
    ) {
        if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
            let suspended = std::mem::take(&mut self.wait_lists[var_id]);
            for c in suspended {
                let p = c.priority(ctx);
                heap.push(PrioritizedConstraint {
                    priority: p,
                    constraint: c,
                });
            }
            // All constraints woken — restore to Generalized if no guards remain
            if var_id < self.gen_statuses.len()
                && self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable
            {
                self.gen_statuses[var_id] = GenStatus::Generalized;
            }
        }
    }

    // ── OmniML: Match branches ───────────────────────────────────

    /// Register a set of match branch patterns. Returns a `branches_id`
    /// that can be referenced by a `Constraint::Match`.
    pub fn register_match_branches(&mut self, branches: Vec<MatchBranchSet>) -> usize {
        let id = self.match_branches.len();
        for b in branches {
            self.match_branches.push(b);
        }
        id
    }

    /// Try to discharge a Match constraint using a shape variable.
    /// If the scrutinee has a shape variable and it's resolved, discharge
    /// immediately.  If not, register a callback on the shape variable
    /// so the match fires when the shape becomes known.
    /// Returns `true` if the match was handled (either discharged or
    /// registered for later).
    pub fn try_match_via_shape_var(
        &mut self,
        ctx: &mut TypeContext,
        scrutinee: TypeId,
        branches_id: usize,
        heap: &mut BinaryHeap<PrioritizedConstraint>,
    ) -> bool {
        let resolved = ctx.resolve_binding(scrutinee);
        match ctx.get(resolved) {
            TypeData::InferVar { id } => {
                if *id < self.guard_sets.len() && !self.guard_sets[*id].is_empty() {
                    true
                } else {
                    let match_c = Constraint::Match {
                        scrutinee,
                        branches_id,
                        span: crate::ast::Span::new(0, 0),
                    };
                    // #3: Register on this var AND all vars sharing its
                    // binding root (transitive wait_list).
                    let root = ctx.resolve_binding(scrutinee);
                    let targets: Vec<usize> = self
                        .var_type_ids
                        .iter()
                        .enumerate()
                        .filter(|(_, ty_id)| ctx.resolve_binding(**ty_id) == root)
                        .map(|(i, _)| i)
                        .collect();
                    for other_id in targets {
                        self.suspend_on_var(match_c.clone(), other_id);
                    }
                    true
                }
            }
            _ => {
                self.discharge_match(ctx, scrutinee, branches_id, heap);
                true
            }
        }
    }

    // ── OmniML: Contextual Unicity C[τ!ξ] ────────────────────────
    //
    // From O'Brien, Rémy & Scherer §4.1:
    //   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    //
    // Three syntactic rules (decidable approximation):
    //   UNI-TYPE: τ is non-variable → shape(τ) = ξ
    //   UNI-VAR:  τ = α and ∃ equalities α = τ' where τ' is non-variable
    //   UNI-BACKPROP: τ = α and all instances of α share shape ξ

    /// Check whether a type has a unique shape determined by the
    /// constraint context. Returns `Some(shape)` if unicity holds,
    /// `None` if the shape cannot be uniquely determined.
    ///
    /// Implements the ⊆-closed erasure semantics:
    ///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    /// where [C] erases all SuspendedMatch constraints to true.
    pub fn unicity_check(
        &self,
        ctx: &TypeContext,
        ty: TypeId,
        active_constraints: &[PrioritizedConstraint],
    ) -> Option<PrincipalShape> {
        let resolved = ctx.resolve_binding(ty);
        let data = ctx.get(resolved);

        // ── UNI-TYPE: non-variable type ──────────────────────────
        // If τ is already resolved to a concrete type, its shape is known.
        if !matches!(data, TypeData::InferVar { .. }) {
            return Some(Self::shape_of_type(ctx, resolved));
        }

        // τ is an InferVar — extract its id.
        let var_id = match data {
            TypeData::InferVar { id } => *id,
            _ => return None,
        };

        // ── UNI-VAR: α is unified with a concrete type ───────────
        // Scan all Eq constraints in the active set. If any equality
        // binds α to a non-variable type, that determines the shape.
        for pc in active_constraints {
            if let Constraint::Eq(a, b, _) = &pc.constraint {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                // Check if this Eq constraint involves our variable
                let other = if ra == resolved {
                    Some(rb)
                } else if rb == resolved {
                    Some(ra)
                } else {
                    None
                };
                if let Some(other_ty) = other {
                    let other_resolved = ctx.resolve_binding(other_ty);
                    if !matches!(ctx.get(other_resolved), TypeData::InferVar { .. }) {
                        return Some(Self::shape_of_type(ctx, other_resolved));
                    }
                }
            }
        }

        // ── UNI-BACKPROP: shape from incremental instantiations ──
        // If this variable is a PG variable with forward references,
        // check if all its instances resolve to the same shape.
        if var_id < self.forward_refs.len() && !self.forward_refs[var_id].is_empty() {
            let mut shared_shape: Option<PrincipalShape> = None;
            for &instance_id in &self.forward_refs[var_id] {
                if instance_id < self.var_type_ids.len() {
                    let instance_ty = ctx.resolve_binding(self.var_type_ids[instance_id]);
                    let instance_data = ctx.get(instance_ty);
                    if matches!(instance_data, TypeData::InferVar { .. }) {
                        // Instance is still unresolved — can't determine shape.
                        return None;
                    }
                    let inst_shape = Self::shape_of_type(ctx, instance_ty);
                    match shared_shape {
                        None => shared_shape = Some(inst_shape),
                        Some(ref s) if *s != inst_shape => {
                            // Instances disagree on shape — unicity fails.
                            return None;
                        }
                        _ => {}
                    }
                }
            }
            if let Some(shape) = shared_shape {
                return Some(shape);
            }
        }

        // ── Check Sub constraints for upper/lower bounds ─────────
        // If the variable has bounds that all share the same shape,
        // that shape is uniquely determined.
        let mut shape_from_bounds: Option<PrincipalShape> = None;

        // Check upper bounds (supertype constraints)
        if var_id < self.upper_bounds.len() {
            for &bound in &self.upper_bounds[var_id] {
                let bound_resolved = ctx.resolve_binding(bound);
                if !matches!(ctx.get(bound_resolved), TypeData::InferVar { .. }) {
                    let s = Self::shape_of_type(ctx, bound_resolved);
                    match shape_from_bounds {
                        None => shape_from_bounds = Some(s),
                        Some(ref existing) if *existing != s => return None,
                        _ => {}
                    }
                }
            }
        }
        // Check lower bounds (subtype constraints)
        if var_id < self.lower_bounds.len() {
            for &bound in &self.lower_bounds[var_id] {
                let bound_resolved = ctx.resolve_binding(bound);
                if !matches!(ctx.get(bound_resolved), TypeData::InferVar { .. }) {
                    let s = Self::shape_of_type(ctx, bound_resolved);
                    match shape_from_bounds {
                        None => shape_from_bounds = Some(s),
                        Some(ref existing) if *existing != s => return None,
                        _ => {}
                    }
                }
            }
        }

        shape_from_bounds
    }

    /// Z3-based unicity check: delegates to SmtSolver when syntactic
    /// rules (UNI-TYPE/UNI-VAR/UNI-BACKPROP) are insufficient.
    /// Encodes ALL active constraints (Eq, Sub, bindings) as SMT-LIB2
    /// over an uninterpreted sort `Type`, then queries Z3 for whether
    /// exactly one shape is forced by the constraint context.
    ///
    /// This implements the full ⊆-closed erasure semantics:
    ///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
    pub fn unicity_check_smt(&self, ctx: &TypeContext, ty: TypeId) -> Option<PrincipalShape> {
        let solver = SmtSolver::new("z3");

        // ── 1. Collect all resolved bindings ─────────────────────
        let mut bindings: std::collections::HashMap<usize, TypeId> =
            std::collections::HashMap::default();
        for (i, var_ty) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(*var_ty);
            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                bindings.insert(i, resolved);
            }
        }

        // ── 2. Collect all equality constraints involving InferVars ──
        let mut eq_pairs: Vec<(usize, usize)> = Vec::new();
        for c in &self.constraints {
            if let Constraint::Eq(a, b, _) = c {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if let (TypeData::InferVar { id: aid }, TypeData::InferVar { id: bid }) =
                    (ctx.get(ra), ctx.get(rb))
                {
                    eq_pairs.push((*aid, *bid));
                }
            }
        }

        // ── 3. Check unicity via Z3 ──────────────────────────────
        solver.check_unicity(ctx, ty, &bindings, &eq_pairs)
    }

    // ── OmniML: Incremental Instantiation ────────────────────────
    //
    // From O'Brien, Rémy & Scherer §5.2:
    // When a regional abstraction let x = λ∝[∝].C₁ in C₂ contains a
    // suspended constraint, variables in the region are PG. Instances
    // of PG variables must be tracked so that when the PG variable is
    // refined, the instances are updated.
    //
    // This implements the forward-reference mechanism (§6 "From a stack
    // to a tree"): each PG variable has a list of its instances. When
    // the PG variable is unified with a concrete type, all instances
    // are re-unified.

    /// Register that `instance_id` was created as an instance of
    /// `pg_var_id` (a PartiallyGeneralizable variable).
    /// This enables incremental instantiation: when pg_var_id is
    /// refined, instance_id will be updated.
    pub fn register_instance(&mut self, pg_var_id: usize, instance_id: usize) {
        while self.forward_refs.len() <= pg_var_id {
            self.forward_refs.push(Vec::new());
        }
        while self.reverse_refs.len() <= instance_id {
            self.reverse_refs.push(None);
        }
        if !self.forward_refs[pg_var_id].contains(&instance_id) {
            self.forward_refs[pg_var_id].push(instance_id);
        }
        self.reverse_refs[instance_id] = Some(pg_var_id);

        // Mark the instance as PI (PartialInstance)
        while self.gen_statuses.len() <= instance_id {
            self.gen_statuses.push(GenStatus::Ungeneralized);
        }
        self.gen_statuses[instance_id] = GenStatus::PartialInstance;
    }

    /// ── S-Inst-Copy (OmniML §5.3) ──────────────────────────────────
    ///
    /// Copy solved constraints from a PG abstraction to all its instances.
    /// When a PG variable's multi-equation is resolved (e.g. α = τ),
    /// the equality is propagated to every instance of α.
    /// If τ itself contains other region variables (e.g. β, γ from the
    /// same abstraction), fresh instances of those are created and bound.
    ///
    /// Returns the number of instances that were updated.
    pub fn s_inst_copy(
        &mut self,
        ctx: &mut TypeContext,
        pg_var_id: usize,
        resolve_ty: TypeId,
    ) -> usize {
        if pg_var_id >= self.forward_refs.len() {
            return 0;
        }
        let instances: Vec<usize> = self.forward_refs[pg_var_id].clone();
        let mut updated = 0;
        for inst_id in instances {
            if inst_id < self.var_type_ids.len() {
                let instance_ty_id = self.var_type_ids[inst_id];
                // S-Inst-Copy: copy the solved equation α = τ to the instance.
                // Walk τ to find any other region variables referenced by the
                // abstraction. For each such variable that itself has instances,
                // recursively copy.
                self.s_inst_copy_walk(ctx, instance_ty_id, resolve_ty);
                // Bind instance to the concrete type
                if let Err(_) = ctx.unify(instance_ty_id, resolve_ty) {
                    // unification error will be caught elsewhere
                }
                // Recursively propagate if resolve_ty contains other PG vars
                self.s_inst_copy_deepen(ctx, resolve_ty);
                updated += 1;
            }
        }
        // Clear forward refs (all propagated)
        self.forward_refs[pg_var_id].clear();
        updated
    }

    /// Walk a type and recursively apply S-Inst-Copy to any region variables
    /// found inside it that have their own instances.
    fn s_inst_copy_deepen(&mut self, ctx: &mut TypeContext, ty: TypeId) {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved).clone() {
            TypeData::Fn { params, ret } => {
                for p in params {
                    self.s_inst_copy_deepen(ctx, p);
                }
                self.s_inst_copy_deepen(ctx, ret);
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => {
                for e in elems {
                    self.s_inst_copy_deepen(ctx, e);
                }
            }
            TypeData::App { args, .. } => {
                for a in args {
                    self.s_inst_copy_deepen(ctx, a);
                }
            }
            TypeData::InferVar { id } => {
                // If this region variable has instances, propagate to them too
                if id < self.forward_refs.len() && !self.forward_refs[id].is_empty() {
                    let resolved_inner = ctx.resolve_binding(ty);
                    if !matches!(ctx.get(resolved_inner), TypeData::InferVar { .. }) {
                        self.s_inst_copy(ctx, id, resolved_inner);
                    }
                }
            }
            _ => {}
        }
    }

    /// Copy one solved equation to one instance variable (S-Inst-Copy detail).
    fn s_inst_copy_walk(&mut self, ctx: &mut TypeContext, instance_ty: TypeId, source_ty: TypeId) {
        let resolved_source = ctx.resolve_binding(source_ty);
        match ctx.get(resolved_source) {
            TypeData::InferVar { id } => {
                // This instance refers to another region variable.
                // If the source has its own forward refs (instances),
                // recursively copy them to the new instance's peer.
                if *id < self.forward_refs.len() && !self.forward_refs[*id].is_empty() {
                    let resolved_src = ctx.resolve_binding(source_ty);
                    if !matches!(ctx.get(resolved_src), TypeData::InferVar { .. }) {
                        self.s_inst_copy(ctx, *id, resolved_src);
                    }
                }
            }
            _ => {}
        }
    }

    // ── S-Exists-Lower: Z3-backed semantic check (OmniML §5.3) ───
    //
    // The paper's S-Exists-Lower requires:
    //   "C determines β̄ iff every ground assignment φ and φ' that satisfy
    //    (the erasure of) C and coincide outside of β̄ coincide on β̄."
    //
    // We implement this via Z3 (unicity_check_smt). If Z3 determines the
    // variable's shape is uniquely determined by the constraint context, it
    // is safe to lower from PG to monomorphic (Ungeneralized).
    //
    // Falls back to a level-based heuristic when Z3 is unavailable or the
    // query times out, as a conservative over-approximation.

    /// Attempt to lower a variable using the full Z3-backed semantic check
    /// (OmniML §5.3 S-Exists-Lower). If Z3 determines the variable's shape is
    /// uniquely determined by the constraint context, it can be safely lowered
    /// from PG to monomorphic (Ungeneralized).
    ///
    /// Falls back to the level-based heuristic when Z3 is unavailable.
    pub fn s_exists_lower(&mut self, ctx: &TypeContext, var_id: usize) -> bool {
        if var_id >= self.type_vars.len() || var_id >= self.gen_statuses.len() {
            return false;
        }
        if self.gen_statuses[var_id] != GenStatus::PartiallyGeneralizable {
            return false;
        }

        // ── Z3-backed semantic check ──────────────────────────────
        // Query whether this variable's shape is uniquely determined.
        if let Some(_shape) = self.unicity_check_smt(ctx, self.var_type_ids[var_id]) {
            // Shape is uniquely determined → safe to lower.
            self.gen_statuses[var_id] = GenStatus::Ungeneralized;
            return true;
        }

        // ── Fallback: level-based heuristic ───────────────────────
        // When Z3 is unavailable or the query times out, use the
        // conservative level-based approximation.
        let var_level = self.type_vars[var_id].level;
        if var_level > 0 && var_level > self.current_level {
            self.type_vars[var_id].level = var_level - 1;
            self.gen_statuses[var_id] = GenStatus::Ungeneralized;
            return true;
        }

        false
    }

    /// ── S-Generalize / update_and_generalize_generation (OmniML §5.3) ──
    ///
    /// Drains all dirty regions, collects guarded roots, and generalizes
    /// PG variables that are no longer guarded or referenced.
    ///
    /// The optional `target_var_id` restricts processing to just the region
    /// containing that variable (for targeted generalization before instantiation).
    pub fn force_generalize(&mut self, ctx: &mut TypeContext) {
        self.force_generalize_for_regions(ctx, &[], None)
    }

    /// Full generation-based generalization.  `dirty_levels` lists region
    /// levels that have been marked dirty.  `target_var_id` (if set) limits
    /// processing to the region containing that specific variable.
    pub fn force_generalize_for_regions(
        &mut self,
        ctx: &mut TypeContext,
        dirty_levels: &[usize],
        target_var: Option<usize>,
    ) {
        // Collect PG variables from dirty_set or dirty_levels.
        let dirty: Vec<usize> = if let Some(tv) = target_var {
            // Targeted: only process the region containing `tv`.
            let level = self.type_vars.get(tv).map(|v| v.level).unwrap_or(0);
            (0..self.gen_statuses.len())
                .filter(|i| {
                    self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                        && self
                            .type_vars
                            .get(*i)
                            .map(|v| v.level == level)
                            .unwrap_or(false)
                })
                .collect()
        } else if !self.dirty_set.is_empty() {
            self.dirty_set
                .iter()
                .copied()
                .filter(|i| self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable))
                .collect()
        } else if !self.region_dirty_levels.is_empty() {
            let dl: std::collections::HashSet<usize> =
                self.region_dirty_levels.iter().copied().collect();
            (0..self.gen_statuses.len())
                .filter(|i| {
                    self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                        && self
                            .type_vars
                            .get(*i)
                            .map(|v| dl.contains(&v.level))
                            .unwrap_or(false)
                })
                .collect()
        } else if !dirty_levels.is_empty() {
            // Use provided dirty levels (from RegionTree).
            let dl: std::collections::HashSet<usize> = dirty_levels.iter().copied().collect();
            (0..self.gen_statuses.len())
                .filter(|i| {
                    self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable)
                        && self
                            .type_vars
                            .get(*i)
                            .map(|v| dl.contains(&v.level))
                            .unwrap_or(false)
                })
                .collect()
        } else {
            // Fallback: process all PG variables.
            (0..self.gen_statuses.len())
                .filter(|i| self.gen_statuses.get(*i) == Some(&GenStatus::PartiallyGeneralizable))
                .collect()
        };

        if dirty.is_empty() {
            return;
        }

        // Ensure guard_sets consistency.
        for &i in &dirty {
            while self.guard_sets.len() <= i {
                self.guard_sets.push(Vec::new());
            }
        }

        // Compute transitive guards via binding-root sharing.
        let mut trans_guarded: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &i in &dirty {
            if i < self.guard_sets.len() && !self.guard_sets[i].is_empty() {
                trans_guarded.insert(i);
            }
        }
        for &i in &dirty {
            if !trans_guarded.contains(&i) && i < self.var_type_ids.len() {
                let i_root = ctx.resolve_binding(self.var_type_ids[i]);
                for &j in &dirty {
                    if j < self.var_type_ids.len() && trans_guarded.contains(&j) {
                        if ctx.resolve_binding(self.var_type_ids[j]) == i_root {
                            trans_guarded.insert(i);
                            break;
                        }
                    }
                }
            }
        }

        // Process innermost-first (highest level first = most nested first).
        let mut vars_by_level: Vec<(usize, usize)> = dirty
            .iter()
            .map(|&i| (i, self.type_vars.get(i).map(|v| v.level).unwrap_or(0)))
            .collect();
        vars_by_level.sort_by(|a, b| b.1.cmp(&a.1));

        // Rigid scope check per generation.
        for &(i, level) in &vars_by_level {
            if i >= self.var_type_ids.len() {
                continue;
            }
            let resolved = ctx.resolve_binding(self.var_type_ids[i]);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                continue;
            }
            if level < self.current_level {
                if Self::check_rigid_escape(ctx, resolved, level) {
                    continue;
                }
            }
        }

        // Generalize eligible PG → G in level order (generation order).
        for &(i, _level) in &vars_by_level {
            if i >= self.gen_statuses.len()
                || self.gen_statuses[i] != GenStatus::PartiallyGeneralizable
            {
                continue;
            }
            let is_trans_guarded = trans_guarded.contains(&i);
            let has_waiting = i < self.wait_lists.len() && !self.wait_lists[i].is_empty();
            let is_resolved = {
                let ty = ctx.resolve_binding(self.var_type_ids[i]);
                !matches!(ctx.get(ty), TypeData::InferVar { .. })
            };
            if is_resolved && !is_trans_guarded && !has_waiting {
                self.gen_statuses[i] = GenStatus::Generalized;
            }
        }

        // Update dirty_set: remove only the processed levels.
        if dirty_levels.is_empty() && target_var.is_none() {
            self.dirty_set.clear();
        } else {
            self.dirty_set.retain(|i| !dirty.contains(i));
        }
    }

    /// Check whether a resolved type contains escaped rigid (skolem) variables.
    /// Recursively walks the type tree looking for `GenericParam` references that
    /// would indicate a Forall-bound variable has escaped into an outer scope.
    fn check_rigid_escape(ctx: &TypeContext, ty: TypeId, max_level: usize) -> bool {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::GenericParam { .. } => true, // escape detected
            TypeData::Fn { params, ret } => {
                params
                    .iter()
                    .any(|&p| Self::check_rigid_escape(ctx, p, max_level))
                    || Self::check_rigid_escape(ctx, *ret, max_level)
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => elems
                .iter()
                .any(|&e| Self::check_rigid_escape(ctx, e, max_level)),
            TypeData::App { args, .. } => args
                .iter()
                .any(|&a| Self::check_rigid_escape(ctx, a, max_level)),
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => Self::check_rigid_escape(ctx, *body, max_level),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                Self::check_rigid_escape(ctx, *ty, max_level)
            }
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                Self::check_rigid_escape(ctx, *elem, max_level)
            }
            TypeData::Ptr { pointee, .. } => Self::check_rigid_escape(ctx, *pointee, max_level),
            TypeData::AssociatedType { self_ty, .. } => {
                Self::check_rigid_escape(ctx, *self_ty, max_level)
            }
            _ => false, // Int, Bool, etc. are safe
        }
    }

    /// Mark a variable as dirty for the next `force_generalize` call.
    /// Called when a variable is unified or updated, enabling incremental
    /// processing instead of re-checking all variables.
    pub fn mark_dirty(&mut self, var_id: usize) {
        self.dirty_set.insert(var_id);
    }

    /// Check if a variable has any forward references (instances).
    pub fn has_instances(&self, var_id: usize) -> bool {
        var_id < self.forward_refs.len() && !self.forward_refs[var_id].is_empty()
    }

    /// Check if a variable is an instance of a PG variable.
    pub fn is_instance(&self, var_id: usize) -> Option<usize> {
        if var_id < self.reverse_refs.len() {
            self.reverse_refs[var_id]
        } else {
            None
        }
    }

    /// Discharge a suspended Match constraint: when the scrutinee's shape
    /// is known (unicity holds), add the matched branch's continuation
    /// constraints and remove the guard on the scrutinee variable.
    pub fn discharge_match(
        &mut self,
        ctx: &mut TypeContext,
        scrutinee_ty: TypeId,
        branches_id: usize,
        heap: &mut BinaryHeap<PrioritizedConstraint>,
    ) -> bool {
        let resolved = ctx.resolve_binding(scrutinee_ty);
        let shape = Self::shape_of_type(ctx, resolved);

        // Find the branch that matches this shape.
        let start = branches_id;
        if start < self.match_branches.len() {
            for i in start..self.match_branches.len() {
                let branch = &self.match_branches[i];
                let matches_pattern = shape == branch.shape_pattern;

                if matches_pattern {
                    // Enqueue continuation constraints.
                    for c in &branch.continuation {
                        let p = c.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: c.clone(),
                        });
                    }
                    return true;
                }
            }

            // No exact match — try the else_ fallback of the first branch.
            let first = &self.match_branches[start];
            if !first.else_continuation.is_empty() {
                for c in &first.else_continuation {
                    let p = c.priority(ctx);
                    heap.push(PrioritizedConstraint {
                        priority: p,
                        constraint: c.clone(),
                    });
                }
                return true;
            }
        }

        false
    }

    /// Get the generalisation status for a variable.
    pub fn get_gen_status(&self, var_id: usize) -> Option<GenStatus> {
        self.gen_statuses.get(var_id).copied()
    }

    /// Mark a variable as guarded by a suspended constraint.
    /// Adds `constraint_idx` to the variable's guard set and sets status to PG.
    pub fn add_guard(&mut self, var_id: usize, constraint_idx: usize) {
        while self.guard_sets.len() <= var_id {
            self.guard_sets.push(Vec::new());
        }
        let guards = &mut self.guard_sets[var_id];
        if !guards.contains(&constraint_idx) {
            guards.push(constraint_idx);
            if var_id < self.gen_statuses.len() {
                self.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
            }
        }
    }

    /// Remove a guard from a variable when its suspended constraint is discharged.
    /// If no guards remain, the variable can be re-generalised.
    pub fn remove_guard(&mut self, var_id: usize, constraint_idx: usize) {
        if var_id < self.guard_sets.len() {
            let guards = &mut self.guard_sets[var_id];
            guards.retain(|&g| g != constraint_idx);
            if guards.is_empty() && var_id < self.gen_statuses.len() {
                // All guards discharged — the variable can become Generalized (G)
                // if it was previously PG.  If it was PI, it stays PI until
                // re-unified.
                if self.gen_statuses[var_id] == GenStatus::PartiallyGeneralizable {
                    self.gen_statuses[var_id] = GenStatus::Generalized;
                }
            }
        }
    }

    pub fn solve(
        &mut self,
        ctx: &mut TypeContext,
        trait_env: &TraitEnv,
        symbols: &SymbolTable,
    ) -> Result<(), TypeError> {
        // ── Build priority queue ────────────────────────────────────
        let mut heap: BinaryHeap<PrioritizedConstraint> = BinaryHeap::new();
        for c in &self.constraints {
            let priority = c.priority(ctx);
            heap.push(PrioritizedConstraint {
                priority,
                constraint: c.clone(),
            });
        }

        // ── Process all constraints in priority order ───────────────
        // With incremental wake-up: after each unification, check if an
        // InferVar was resolved and immediately wake its suspended constraints.
        // This follows OmniML's job-queue pattern where unify enqueues jobs
        // that the scheduler runs immediately.
        loop {
            let mut active_count = heap.len();
            while let Some(pc) = heap.pop() {
                active_count -= 1;
                match &pc.constraint {
                    Constraint::Eq(a, b, _) => {
                        // Check if either side is an InferVar before unifying
                        let ra = ctx.resolve_binding(*a);
                        let rb = ctx.resolve_binding(*b);
                        let a_is_infer = matches!(ctx.get(ra), TypeData::InferVar { .. });
                        let b_is_infer = matches!(ctx.get(rb), TypeData::InferVar { .. });
                        let a_var_id = if a_is_infer {
                            if let TypeData::InferVar { id } = ctx.get(ra) {
                                Some(*id)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let b_var_id = if b_is_infer {
                            if let TypeData::InferVar { id } = ctx.get(rb) {
                                Some(*id)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // Level-based promotion (Fan, Xu & Xie 2025 §6.2):
                        // If unifying two InferVars at different levels, promote
                        // the higher-level one to the lower level before unifying.
                        if let (Some(avid), Some(bvid)) = (a_var_id, b_var_id) {
                            let a_lvl = self.get_var_level(avid).unwrap_or(0);
                            let b_lvl = self.get_var_level(bvid).unwrap_or(0);
                            if a_lvl > b_lvl {
                                if let Some(promoted) = self.try_promote_var(ctx, avid, b_lvl) {
                                    ctx.unify(promoted, *b)?;
                                    // Continue with wake-up below
                                } else {
                                    ctx.unify(*a, *b)?;
                                }
                            } else if b_lvl > a_lvl {
                                if let Some(promoted) = self.try_promote_var(ctx, bvid, a_lvl) {
                                    ctx.unify(*a, promoted)?;
                                } else {
                                    ctx.unify(*a, *b)?;
                                }
                            } else {
                                ctx.unify(*a, *b)?;
                            }
                        } else {
                            ctx.unify(*a, *b)?;
                        }

                        // Mark dirty for drain_dirty tracking.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            self.mark_dirty(*var_id);
                        }

                        // Incremental wake-up: if a variable was just resolved,
                        // immediately enqueue its suspended constraints.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            if self.try_set_shape(*var_id, ctx) {
                                self.wake_var_incremental(*var_id, &mut heap, ctx);
                            }
                            // ── OmniML §5.2: Incremental instantiation ──
                            // S-Inst-Copy fires on the PG→G *transition*, not while
                            // the var is still PG.  After wake_var_incremental clears
                            // the wait list and sets the status to Generalized (no
                            // guards remain), we propagate the resolved type to all
                            // instances.  This avoids two bugs:
                            //   1. Premature copy: s_inst_copy while still PG would
                            //      clear forward_refs, dropping future refinements.
                            //   2. Missed copy: if wake_var_incremental already set
                            //      Generalized, the old PG guard was dead code.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::Generalized
                            {
                                let resolved_ty = ctx.resolve_binding(self.var_type_ids[*var_id]);
                                if !matches!(ctx.get(resolved_ty), TypeData::InferVar { .. }) {
                                    // ── S-Generalize (OmniML §5.3) ──
                                    self.force_generalize(ctx);
                                    self.s_inst_copy(ctx, *var_id, resolved_ty);
                                }
                            }
                            // ── S-Exists-Lower (OmniML §5.3) ──
                            // For vars that remain PG (still have guards/waiting),
                            // try Z3-backed semantic lowering if uniquely determined.
                            if *var_id < self.gen_statuses.len()
                                && self.gen_statuses[*var_id] == GenStatus::PartiallyGeneralizable
                            {
                                if self.s_exists_lower(ctx, *var_id) {
                                    // lowering succeeded
                                }
                            }
                        }
                    }
                    Constraint::Sub(sub, sup, _span) => {
                        let resolved_sub = ctx.resolve_binding(*sub);
                        let resolved_sup = ctx.resolve_binding(*sup);

                        // If sup is an InferVar, record sub as a lower bound of sup
                        if let TypeData::InferVar { id } = ctx.get(resolved_sup) {
                            if *id < self.lower_bounds.len() {
                                self.lower_bounds[*id].push(resolved_sub);
                                self.mark_dirty(*id);
                            }
                        }
                        // If sub is an InferVar, record sup as an upper bound of sub
                        if let TypeData::InferVar { id } = ctx.get(resolved_sub) {
                            if *id < self.upper_bounds.len() {
                                self.upper_bounds[*id].push(resolved_sup);
                                self.mark_dirty(*id);
                            }
                        }

                        // If both sides are resolved (not InferVar), check the subtype relationship now
                        let sub_is_infer =
                            matches!(ctx.get(resolved_sub), TypeData::InferVar { .. });
                        let sup_is_infer =
                            matches!(ctx.get(resolved_sup), TypeData::InferVar { .. });
                        if !sub_is_infer && !sup_is_infer {
                            if !ctx.subtype(resolved_sub, resolved_sup) {
                                return Err(TypeError::Mismatch {
                                    expected: resolved_sup,
                                    found: resolved_sub,
                                    span: *_span,
                                });
                            }
                        }
                    }
                    Constraint::Impl(ty, trait_id, span) => {
                        let resolved = ctx.resolve_binding(*ty);
                        let data = ctx.get(resolved);
                        // If the type is an error, skip
                        if matches!(data, TypeData::Error) {
                            return Ok(());
                        }
                        // If still an infer var, that's fine; solving will assign a default later
                        if matches!(data, TypeData::InferVar { .. }) {
                            return Ok(());
                        }
                        // Otherwise, check that the impl exists
                        let impl_found = if trait_env.lookup_impl(*trait_id, resolved).is_some() {
                            true
                        } else {
                            trait_env
                                .lookup_impl_generic(*trait_id, resolved, ctx, symbols)
                                .is_some()
                        };
                        if !impl_found {
                            return Err(TypeError::TraitNotImplemented {
                                ty: *ty,
                                trait_name: format!("{:?}", trait_id),
                                span: *span,
                            });
                        }
                        // Generate obligations for associated types: when we have a
                        // resolved Impl(concrete_ty, trait_id, _), look for concrete types
                        // for any AssociatedType { trait_id, name, self_ty } by matching
                        // the impl's assoc_tys entries.
                        if let Some(impl_candidate) = trait_env.lookup_impl(*trait_id, resolved) {
                            for (assoc_name, assoc_ty) in &impl_candidate.assoc_tys {
                                // Walk all Eq constraints to substitute any AssociatedType
                                // that matches this name, trait_id, and self_ty
                                for eq_c in &self.constraints {
                                    if let Constraint::Eq(a, b, _) = eq_c {
                                        for id in &[*a, *b] {
                                            let resolved_id = ctx.resolve_binding(*id);
                                            if let TypeData::AssociatedType {
                                                trait_id: at_trait_id,
                                                name: at_name,
                                                self_ty: at_self,
                                            } = ctx.get(resolved_id).clone()
                                            {
                                                if at_trait_id == *trait_id
                                                    && at_name == *assoc_name
                                                    && ctx.resolve_binding(at_self) == resolved
                                                {
                                                    ctx.unify(resolved_id, *assoc_ty)?;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Constraint::Match {
                        scrutinee,
                        branches_id,
                        span: _,
                    } => {
                        // OmniML §4.1: Try to discharge suspended match constraints.
                        // Check unicity — if the scrutinee's shape is uniquely determined,
                        // discharge the match and enqueue continuation constraints.
                        let resolved = ctx.resolve_binding(*scrutinee);
                        let resolved_data = ctx.get(resolved);

                        if !matches!(resolved_data, TypeData::InferVar { .. }) {
                            // Scrutinee is resolved — shape is known (UNI-TYPE).
                            let _ = self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap);
                        } else {
                            // Try shape variable (OmniML §6): register a callback on the
                            // scrutinee's shape variable, if any.
                            let shape_known = self.try_match_via_shape_var(
                                ctx,
                                *scrutinee,
                                *branches_id,
                                &mut heap,
                            );
                            if !shape_known {
                                // Scrutinee is still an InferVar — try unicity via bounds
                                if let Some(_shape) =
                                    Self::unicity_check(self, ctx, *scrutinee, &[])
                                {
                                    let _ = self.discharge_match(
                                        ctx,
                                        *scrutinee,
                                        *branches_id,
                                        &mut heap,
                                    );
                                } else {
                                    // Cannot discharge yet — push back as low priority
                                    let p = 6u8;
                                    heap.push(PrioritizedConstraint {
                                        priority: p,
                                        constraint: pc.constraint.clone(),
                                    });
                                }
                            }
                        }
                    }
                    Constraint::Exists {
                        var_id,
                        constraint,
                        span: _,
                    } => {
                        // OmniML: ∃α. C — bind a fresh flexible variable.
                        // α is already an InferVar at this level; just solve the body.
                        let inner = constraint.as_ref().clone();
                        let p = inner.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: inner,
                        });
                    }
                    Constraint::Forall {
                        var_id,
                        constraint,
                        span: _,
                    } => {
                        // OmniML: ∀α. C — bind a fresh rigid (skolem) variable.
                        // Record for skolem escape check; solve the body.
                        let inner = constraint.as_ref().clone();
                        let p = inner.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: inner,
                        });
                    }
                    Constraint::Instance {
                        expr_var,
                        instantiation_ty,
                        span: _,
                    } => {
                        let p = Constraint::Eq(
                            *instantiation_ty,
                            *instantiation_ty,
                            crate::ast::Span::new(0, 0),
                        )
                        .priority(ctx);
                        let ic = Constraint::Eq(
                            *instantiation_ty,
                            *instantiation_ty,
                            crate::ast::Span::new(0, 0),
                        );
                        heap.push(PrioritizedConstraint {
                            priority: p,
                            constraint: ic,
                        });
                    }
                    Constraint::Let {
                        expr_var: _,
                        def_constraint,
                        body_constraint,
                        span: _,
                    } => {
                        let prev_level = self.enter_level();
                        let def_p = def_constraint.priority(ctx);
                        heap.push(PrioritizedConstraint {
                            priority: def_p,
                            constraint: def_constraint.as_ref().clone(),
                        });
                        let body_p = body_constraint.priority(ctx).max(4);
                        heap.push(PrioritizedConstraint {
                            priority: body_p,
                            constraint: body_constraint.as_ref().clone(),
                        });
                        self.exit_level(prev_level);
                    }
                }
            }

            // ── OmniML: Process Match constraints ──────────────────────
            // After processing Eq/Sub/Impl, check suspended match constraints.
            // A Match constraint can be discharged when the scrutinee's shape
            // is uniquely determined by the context (unicity check).
            // This implements O'Brien, Rémy & Scherer §4.1, MATCH-CTX rule.
            for pc in &heap.clone().into_sorted_vec() {
                if let Constraint::Match {
                    scrutinee,
                    branches_id,
                    span: _,
                } = &pc.constraint
                {
                    let resolved = ctx.resolve_binding(*scrutinee);
                    // Only attempt discharge if scrutinee is resolved (not an InferVar)
                    if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                        if let Some(shape) = Self::unicity_check(self, ctx, *scrutinee, &[]) {
                            let _discharged =
                                self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap);
                        }
                    } else {
                        // Scrutinee is still an InferVar — check unicity via bounds
                        if let Some(_shape) = Self::unicity_check(self, ctx, *scrutinee, &[]) {
                            let _discharged =
                                self.discharge_match(ctx, *scrutinee, *branches_id, &mut heap);
                        }
                    }
                }
            }

            // ── Wake-up: reprocess suspended constraints ───────────────
            // After processing all active constraints, check if any variables
            // were resolved. If so, wake their wait-listed constraints and
            // continue solving (OmniML bidirectional flow §3.2).
            let mut woken = 0usize;
            for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
                let resolved = ctx.resolve_binding(ty_id);
                if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                    // This variable was resolved — wake its suspended constraints
                    if i < self.wait_lists.len() && !self.wait_lists[i].is_empty() {
                        let suspended = std::mem::take(&mut self.wait_lists[i]);
                        let count = suspended.len();
                        for c in suspended {
                            let p = c.priority(ctx);
                            heap.push(PrioritizedConstraint {
                                priority: p,
                                constraint: c,
                            });
                        }
                        woken += count;
                    }
                }
            }
            if woken == 0 {
                // #2: Solver exhaustion — check for remaining undischarged Match
                // constraints and fire their else_continuation as a fallback.
                let remaining: Vec<PrioritizedConstraint> = heap.drain().collect();
                let match_elses: Vec<(TypeId, usize)> = remaining
                    .iter()
                    .filter_map(|pc| {
                        if let Constraint::Match {
                            scrutinee,
                            branches_id,
                            ..
                        } = &pc.constraint
                        {
                            let resolved = ctx.resolve_binding(*scrutinee);
                            if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                                Some((*scrutinee, *branches_id))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut else_heap = BinaryHeap::new();
                for (scrutinee, branches_id) in match_elses {
                    self.discharge_match(ctx, scrutinee, branches_id, &mut else_heap);
                }
                break; // converged: no more constraints to wake
            }
            // Continue the loop to process woken constraints
        }

        // Kind checking: ensure that solved types respect the variable's kind
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            if let TypeData::InferVar { .. } = data {
                continue; // will be defaulted below
            }
            if matches!(data, TypeData::Error) {
                continue;
            }
            let kind = self.type_vars[i].kind;
            match kind {
                TypeVariableKind::Integer => {
                    if !matches!(
                        data,
                        TypeData::Int { .. }
                            | TypeData::UInt { .. }
                            | TypeData::USize
                            | TypeData::Rational { .. }
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Float => {
                    if !matches!(data, TypeData::Float { .. }) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Bool => {
                    if !matches!(data, TypeData::Bool) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Numeric => {
                    if !matches!(
                        data,
                        TypeData::Int { .. }
                            | TypeData::UInt { .. }
                            | TypeData::Float { .. }
                            | TypeData::Rational { .. }
                            | TypeData::USize
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                _ => {}
            }
        }

        // Defaulting: unfilled infer vars get default types,
        // UNLESS they are PartiallyGeneralizable (guarded by suspended constraints).
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if let TypeData::InferVar { .. } = ctx.get(resolved) {
                // Skip variables that are PG — they still have suspended constraints
                // and will be re-generalized when those constraints are discharged.
                if i < self.gen_statuses.len()
                    && self.gen_statuses[i] == GenStatus::PartiallyGeneralizable
                {
                    continue;
                }
                let default_ty = match self.type_vars[i].kind {
                    TypeVariableKind::Integer => ctx.int(32, true),
                    TypeVariableKind::Float => ctx.float(64),
                    TypeVariableKind::Bool => ctx.bool(),
                    TypeVariableKind::Numeric => ctx.int(32, true),
                    TypeVariableKind::Unconstrained => ctx.error(),
                    TypeVariableKind::Any => ctx.error(),
                };
                ctx.bindings.borrow_mut().insert(ty_id, default_ty);
            }
        }

        Ok(())
    }

    pub fn finalize(&self, ctx: &mut TypeContext) -> HashMap<usize, TypeId> {
        let mut solution = HashMap::default();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            match data {
                TypeData::InferVar { id } => {
                    // Variable is still unbound — try to infer from bounds
                    let var_id = *id;
                    let lbs: &[TypeId] = if var_id < self.lower_bounds.len() {
                        &self.lower_bounds[var_id]
                    } else {
                        &[]
                    };
                    let ubs: &[TypeId] = if var_id < self.upper_bounds.len() {
                        &self.upper_bounds[var_id]
                    } else {
                        &[]
                    };
                    let chosen = if !lbs.is_empty() {
                        // Covariant: pick the least upper bound from lower bounds
                        // (simple heuristic: pick the first resolved lower bound)
                        let first_resolved = lbs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else if !ubs.is_empty() {
                        // Contravariant: pick the greatest lower bound from upper bounds
                        let first_resolved = ubs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else {
                        // No bounds — default based on kind
                        match self.type_vars[i].kind {
                            TypeVariableKind::Integer => ctx.int(32, true),
                            TypeVariableKind::Float => ctx.float(64),
                            TypeVariableKind::Bool => ctx.bool(),
                            TypeVariableKind::Numeric => ctx.int(32, true),
                            _ => ctx.error(),
                        }
                    };
                    solution.insert(var_id, chosen);
                }
                _ => {
                    solution.insert(self.type_vars[i].id, resolved);
                }
            }
        }
        solution
    }

    pub fn apply_solution(
        ty: TypeId,
        solution: &HashMap<usize, TypeId>,
        ctx: &TypeContext,
    ) -> TypeId {
        replace_infer(ty, solution, ctx)
    }

    /// Check for inference variables that remain unresolved and were
    /// defaulted to `error` (unconstrained/any kind). Returns a list of
    /// diagnostic messages describing the ambiguous variables.
    pub fn check_unresolved(&self, ctx: &TypeContext) -> Vec<String> {
        let mut results = Vec::new();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
                if i < self.type_vars.len() {
                    // Only report `Any` — `Unconstrained` is defaulted to error
                    // by the solver as a normal fallback, not an ambiguity.
                    if self.type_vars[i].kind == TypeVariableKind::Any {
                        results.push(format!("unresolved type variable #{} (Any)", i));
                    }
                }
            }
        }
        results
    }
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

fn replace_infer(ty: TypeId, solution: &HashMap<usize, TypeId>, ctx: &TypeContext) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::InferVar { id } => solution.get(&id).copied().unwrap_or(ty),
        TypeData::Int { .. }
        | TypeData::UInt { .. }
        | TypeData::Float { .. }
        | TypeData::Rational { .. }
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit
        | TypeData::Error
        | TypeData::Poly { .. } => ty,
        TypeData::GenericParam { .. } => ty,
        TypeData::Struct { def_id } => ty, // zero-arg, nothing to replace
        TypeData::Enum { def_id } => ty,   // zero-arg, nothing to replace
        TypeData::App { def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::App {
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Tuple { elems } => {
            let new_elems: Vec<TypeId> = elems
                .iter()
                .map(|&e| replace_infer(e, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Tuple { elems: new_elems })
                .unwrap_or(ctx.error())
        }
        TypeData::Array { elem, size } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Array {
                elem: new_elem,
                size,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Slice { elem } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Slice { elem: new_elem })
                .unwrap_or(ctx.error())
        }
        TypeData::Ref { ty, mutable } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Ref {
                ty: new_ty,
                mutable,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Pointer { ty } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Pointer { ty: new_ty })
                .unwrap_or(ctx.error())
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = replace_infer(size, solution, ctx);
            let new_pointee = replace_infer(pointee, solution, ctx);
            ctx.find_type(&TypeData::Ptr {
                size: new_size,
                pointee: new_pointee,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Fn { params, ret } => {
            let new_params: Vec<TypeId> = params
                .iter()
                .map(|&p| replace_infer(p, solution, ctx))
                .collect();
            let new_ret = replace_infer(ret, solution, ctx);
            ctx.find_type(&TypeData::Fn {
                params: new_params,
                ret: new_ret,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::DynTrait { .. } => ty,
        TypeData::Exists { name, base } => {
            let new_base = replace_infer(base, solution, ctx);
            ctx.find_type(&TypeData::Exists {
                name,
                base: new_base,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = replace_infer(self_ty, solution, ctx);
            ctx.find_type(&TypeData::AssociatedType {
                trait_id,
                name,
                self_ty: new_self,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Forall {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Forall {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Mu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Mu {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Nu {
            param_index,
            param_name,
            body,
        } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Nu {
                param_index,
                param_name,
                body: new_body,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Coproduct { alternatives } => {
            let new_alts: Vec<TypeId> = alternatives
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Coproduct {
                alternatives: new_alts,
            })
            .unwrap_or(ctx.error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_ctx() -> TypeContext {
        TypeContext::new()
    }

    #[test]
    fn test_shape_of_fn() {
        let mut ctx = new_ctx();
        let int_ty = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![int_ty], bool_ty);
        let shape = InferenceContext::shape_of_type(&ctx, fn_ty);
        assert!(matches!(shape, PrincipalShape::Arrow));
    }

    #[test]
    fn test_shape_of_tuple() {
        let mut ctx = new_ctx();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let tup = ctx.tuple(vec![bool_ty, int_ty]);
        let shape = InferenceContext::shape_of_type(&ctx, tup);
        assert!(matches!(shape, PrincipalShape::Tuple(2)));
    }

    #[test]
    fn test_shape_of_forall() {
        let mut ctx = new_ctx();
        let p0 = ctx.generic_param(0, "X".into());
        let fn_ty = ctx.function(vec![p0], p0);
        let forall = ctx.forall(0, "X".into(), fn_ty);
        let shape = InferenceContext::shape_of_type(&ctx, forall);
        assert!(matches!(shape, PrincipalShape::Poly));
    }

    #[test]
    fn test_suspend_and_wake_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        // Extract the var ID from the InferVar type
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)),
            var_id,
        );
        // Waking moves suspended constraints back to the active list
        infer.wake_var(var_id);
        // The active constraints list should now have the suspended constraint
        assert!(!infer.constraints.is_empty());
    }

    #[test]
    fn test_wake_var_incremental() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.suspend_on_var(
            Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)),
            var_id,
        );
        // wake_var_incremental needs a heap, var_id, and ctx
        let mut heap = std::collections::BinaryHeap::new();
        infer.wake_var_incremental(var_id, &mut heap, &ctx);
        // After waking, the wait list should be empty
        assert!(infer.wait_lists[var_id].is_empty());
    }

    #[test]
    fn test_level_enter_exit() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        assert!(infer.current_level > 0);
        infer.exit_level(prev);
        assert_eq!(infer.current_level, 0);
    }

    #[test]
    fn test_level_new_type_var() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        assert!(infer.get_var_level(var_id).unwrap_or(0) > 0);
        infer.exit_level(prev);
    }

    #[test]
    fn test_eq_concrete_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let eq = Constraint::Eq(bool_ty, int_ty, crate::ast::Span::new(0, 0));
        let impl_c = Constraint::Impl(bool_ty, DefId(0), crate::ast::Span::new(0, 0));
        assert!(eq.priority(&ctx) < impl_c.priority(&ctx));
    }

    #[test]
    fn test_impl_lowest_priority() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let a = Constraint::Impl(bool_ty, DefId(0), crate::ast::Span::new(0, 0));
        let b = Constraint::Impl(int_ty, DefId(1), crate::ast::Span::new(0, 0));
        // Both Impl constraints should have the same priority
        assert_eq!(a.priority(&ctx), b.priority(&ctx));
    }

    #[test]
    fn test_match_constraint_priority() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let var = ctx.alloc_infer_var(0);
        // Match constraint on an InferVar → low priority (6)
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: 0,
            span: crate::ast::Span::new(0, 0),
        };
        assert_eq!(
            match_c.priority(&ctx),
            6,
            "Match on InferVar should have lowest priority"
        );
        // Eq on concrete types → high priority (0)
        let eq_c = Constraint::Eq(ctx.bool(), ctx.int(32, true), crate::ast::Span::new(0, 0));
        assert_eq!(
            eq_c.priority(&ctx),
            0,
            "Eq on concrete should have highest priority"
        );
    }

    #[test]
    fn test_unicity_check_non_var() {
        // UNI-TYPE: non-variable type has unique shape
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        // Use pre-allocated built-in types
        let int_ty = ctx.int(32, true);
        let shape = InferenceContext::unicity_check(&infer, &ctx, int_ty, &[]);
        assert!(shape.is_some(), "non-variable type should have known shape");
    }

    #[test]
    fn test_unicity_check_fn_type() {
        let mut ctx = TypeContext::new();
        let infer = InferenceContext::new();
        let int_ty = ctx.int(32, true);
        let fn_ty = ctx.function(vec![int_ty], int_ty);
        let shape = InferenceContext::unicity_check(&infer, &ctx, fn_ty, &[]);
        assert!(shape.is_some(), "function type should have known shape");
        assert_eq!(shape.unwrap(), PrincipalShape::Arrow);
    }

    #[test]
    fn test_register_instance_and_propagate() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg_var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let pg_id = match ctx.get(pg_var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let inst1 = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let inst1_id = match ctx.get(inst1) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Register the instance
        infer.register_instance(pg_id, inst1_id);
        assert!(infer.has_instances(pg_id), "PG var should have instances");
        assert_eq!(
            infer.is_instance(inst1_id),
            Some(pg_id),
            "instance should track its PG var"
        );
        // Propagate PG resolution via S-Inst-Copy
        let bool_ty = ctx.bool();
        let updated = infer.s_inst_copy(&mut ctx, pg_id, bool_ty);
        assert_eq!(updated, 1, "should have updated 1 instance");
        // Check that the instance was unified with bool
        assert!(
            !infer.has_instances(pg_id),
            "forward refs should be cleared"
        );
    }

    #[test]
    fn test_register_match_branches() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: vec![Constraint::Eq(
                ctx.int(32, true),
                ctx.int(32, true),
                crate::ast::Span::new(0, 0),
            )],
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);
        assert!(id < infer.match_branches.len());
    }

    #[test]
    fn test_match_priority_change_on_resolve() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let match_c = Constraint::Match {
            scrutinee: var,
            branches_id: 0,
            span: crate::ast::Span::new(0, 0),
        };
        // Before resolution: low priority
        assert_eq!(match_c.priority(&ctx), 6);
        // After resolution: medium priority
        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(var, int_ty);
        assert_eq!(
            match_c.priority(&ctx),
            3,
            "match on resolved type should have medium priority"
        );
    }

    // ── else_ fallback ───────────────────────────────────────────────

    #[test]
    fn test_else_continuation_on_mismatch() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = std::collections::BinaryHeap::new();
        let int_ty = ctx.int(32, true);

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: vec![Constraint::Eq(int_ty, int_ty, crate::ast::Span::new(0, 0))],
        }];
        let id = infer.register_match_branches(branches);
        let int_ty2 = ctx.int(64, false);

        let result = infer.discharge_match(&mut ctx, int_ty2, id, &mut heap);
        assert!(result, "else_ fallback should return true");
        assert!(!heap.is_empty(), "else_ continuation should be enqueued");
    }

    #[test]
    fn test_else_continuation_empty_still_fails() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = std::collections::BinaryHeap::new();

        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);
        let int_ty = ctx.int(32, true);

        let result = infer.discharge_match(&mut ctx, int_ty, id, &mut heap);
        assert!(!result, "no else_ fallback should still fail");
        assert!(heap.is_empty(), "no constraints should be enqueued");
    }

    // ── force_generalization ─────────────────────────────────────────

    #[test]
    fn test_force_generalize_pg_with_guard() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let _var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        }
        infer.add_guard(var_id, 0);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "guarded PG var should remain PG after force_generalize"
        );
    }

    #[test]
    fn test_force_generalize_pg_no_guard_resolved() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let var_id = 0;
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        }

        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(var, int_ty);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "un-guarded resolved PG var should become Generalized"
        );
    }

    // ── [s] pattern: try_match_via_shape_var callback ─────────────────

    #[test]
    fn test_try_match_via_shape_var_registers_waitlist() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = std::collections::BinaryHeap::new();

        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Unconstrained);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(&mut ctx, var, id, &mut heap);
        assert!(handled, "should register the match on the wait list");

        let var_id = 0;
        if var_id < infer.wait_lists.len() {
            assert!(
                !infer.wait_lists[var_id].is_empty(),
                "match should be in the wait list"
            );
        }
    }

    #[test]
    fn test_try_match_via_shape_var_concrete_discharges() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let mut heap = std::collections::BinaryHeap::new();

        let int_ty = ctx.int(32, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: Vec::new(),
            else_continuation: Vec::new(),
        }];
        let id = infer.register_match_branches(branches);

        let handled = infer.try_match_via_shape_var(&mut ctx, int_ty, id, &mut heap);
        assert!(handled, "concrete type should discharge");
    }

    #[test]
    fn test_let_constraint_priority() {
        let c = Constraint::Let {
            expr_var: "x".into(),
            def_constraint: Box::new(Constraint::Eq(
                TypeId(0),
                TypeId(1),
                crate::ast::Span::new(0, 0),
            )),
            body_constraint: Box::new(Constraint::Eq(
                TypeId(0),
                TypeId(0),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        };
        let ctx = TypeContext::new();
        assert_eq!(c.priority(&ctx), 2, "Let should have medium-high priority");
    }

    // ═══════════════════════════════════════════════════════════════
    // Shape Variable Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_shape_var_new_and_resolve() {
        let mut svc = ShapeVarContext::new();
        let sva = svc.new_var(0);
        assert_eq!(svc.resolve(sva), sva);
        assert_eq!(svc.get(sva), None);
        assert!(!svc.is_resolved(sva));

        assert!(svc.try_set(sva, TypeShape::Arrow));
        assert_eq!(svc.get(sva), Some(TypeShape::Arrow));
        assert!(svc.is_resolved(sva));
    }

    #[test]
    fn test_shape_var_try_set_idempotent() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(1);
        assert!(svc.try_set(sv, TypeShape::Tuple(2)));
        // Setting the same shape again succeeds
        assert!(svc.try_set(sv, TypeShape::Tuple(2)));
        // Setting a different shape fails (mismatch)
        assert!(!svc.try_set(sv, TypeShape::Arrow));
        assert_eq!(svc.get(sv), Some(TypeShape::Tuple(2)));
    }

    #[test]
    fn test_shape_var_try_set_fires_callback() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(0);
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = fired.clone();
        svc.on_resolve(sv, move |_| {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(
            !fired.load(std::sync::atomic::Ordering::SeqCst),
            "callback should not fire before resolution"
        );

        assert!(svc.try_set(sv, TypeShape::Arrow));
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "callback should fire on resolution"
        );
    }

    #[test]
    fn test_shape_var_on_resolve_immediate() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(0);
        svc.try_set(sv, TypeShape::Poly);

        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = fired.clone();
        svc.on_resolve(sv, move |_| {
            f.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "on_resolve should fire immediately if already resolved"
        );
    }

    #[test]
    fn test_shape_var_unify_aliasing() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        assert_ne!(svc.resolve(a), svc.resolve(b));

        svc.unify(a, b);
        // After unify, both resolve to the same canonical id
        assert_eq!(svc.resolve(a), svc.resolve(b));
    }

    #[test]
    fn test_shape_var_unify_merges_waitlists() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);

        let fired_a = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fa = fired_a.clone();
        let fb = fired_b.clone();
        svc.on_resolve(a, move |_| {
            fa.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        svc.on_resolve(b, move |_| {
            fb.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        svc.unify(a, b);
        // Resolve the unified target — both callbacks should fire
        let target = svc.resolve(a);
        svc.try_set(target, TypeShape::Arrow);
        assert!(
            fired_a.load(std::sync::atomic::Ordering::SeqCst),
            "callback on a should fire"
        );
        assert!(
            fired_b.load(std::sync::atomic::Ordering::SeqCst),
            "callback on b should fire"
        );
    }

    #[test]
    fn test_shape_var_unify_propagates_resolved() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        svc.try_set(a, TypeShape::Constructor(1));

        let fired_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fb = fired_b.clone();
        svc.on_resolve(b, move |_| {
            fb.store(true, std::sync::atomic::Ordering::SeqCst);
        });

        svc.unify(a, b);
        assert!(
            fired_b.load(std::sync::atomic::Ordering::SeqCst),
            "b's callback should fire when unified with resolved a"
        );
    }

    #[test]
    fn test_shape_var_get_level() {
        let mut svc = ShapeVarContext::new();
        let sv = svc.new_var(3);
        assert_eq!(svc.get_level(sv), 3);
    }

    #[test]
    fn test_shape_var_num_unresolved() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        let c = svc.new_var(0);
        assert_eq!(svc.num_unresolved(), 3);
        svc.try_set(a, TypeShape::Arrow);
        assert_eq!(svc.num_unresolved(), 2);
        svc.try_set(b, TypeShape::Tuple(1));
        assert_eq!(svc.num_unresolved(), 1);
        // c is still unresolved
        assert!(!svc.is_resolved(c));
    }

    #[test]
    fn test_shape_var_unresolved_above_level() {
        let mut svc = ShapeVarContext::new();
        let _l0 = svc.new_var(0);
        let _l1 = svc.new_var(1);
        let _l2 = svc.new_var(2);
        let above1 = svc.unresolved_above_level(1);
        assert_eq!(above1.len(), 1, "only level-2 var should be above level 1");
        assert_eq!(svc.get_level(above1[0]), 2);
    }

    #[test]
    fn test_shapes_compatible() {
        assert!(shapes_compatible(TypeShape::Unknown, TypeShape::Arrow));
        assert!(shapes_compatible(TypeShape::Arrow, TypeShape::Unknown));
        assert!(shapes_compatible(TypeShape::Arrow, TypeShape::Arrow));
        assert!(shapes_compatible(TypeShape::Tuple(3), TypeShape::Tuple(3)));
        assert!(!shapes_compatible(TypeShape::Tuple(2), TypeShape::Tuple(3)));
        assert!(shapes_compatible(
            TypeShape::Constructor(0),
            TypeShape::Constructor(5)
        ));
        assert!(!shapes_compatible(TypeShape::Arrow, TypeShape::Tuple(1)));
        assert!(shapes_compatible(TypeShape::Poly, TypeShape::Poly));
        assert!(!shapes_compatible(TypeShape::Poly, TypeShape::Arrow));
    }

    #[test]
    fn test_type_data_to_shape_variants() {
        let mut ctx = TypeContext::new();
        // Fn → Arrow
        let fn_ty = ctx.function(vec![ctx.bool()], ctx.bool());
        assert_eq!(type_data_to_shape(ctx.get(fn_ty)), TypeShape::Arrow);
        // Tuple(n) → Tuple(n)
        let b = ctx.bool();
        let i = ctx.int(32, true);
        let tup = ctx.tuple(vec![b, i]);
        assert_eq!(type_data_to_shape(ctx.get(tup)), TypeShape::Tuple(2));
        // Struct → Constructor(n)
        let b2 = ctx.bool();
        let i2 = ctx.int(32, true);
        let s = ctx.struct_ty(DefId(42), vec![b2, i2]);
        assert_eq!(type_data_to_shape(ctx.get(s)), TypeShape::Constructor(2));
        // Forall → Poly
        let p0 = ctx.generic_param(0, "X".into());
        let forall = ctx.forall(0, "X".into(), p0);
        assert_eq!(type_data_to_shape(ctx.get(forall)), TypeShape::Poly);
        // Int → Unknown (primitive)
        let int32 = ctx.int(32, true);
        assert_eq!(type_data_to_shape(ctx.get(int32)), TypeShape::Unknown);
        // Bool → Unknown
        let bool_ty = ctx.bool();
        assert_eq!(type_data_to_shape(ctx.get(bool_ty)), TypeShape::Unknown);
    }

    // ═══════════════════════════════════════════════════════════════
    // Level-based Promotion Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_try_promote_var_basic() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let _prev = infer.enter_level();
        let _prev2 = infer.enter_level();
        // Create a variable at the current deep level
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let deep_level = infer.get_var_level(var_id).unwrap();
        assert!(deep_level > 0, "var should be at a deep level");

        // Promote to level 0
        let promoted = infer.try_promote_var(&mut ctx, var_id, 0);
        assert!(promoted.is_some(), "promotion should succeed");
        // The old var should now be bound to the promoted var (via infer.resolve)
        let resolved = infer.resolve(var, &ctx);
        assert!(
            matches!(ctx.get(resolved), TypeData::InferVar { id } if *id != var_id),
            "original var should be bound to a new InferVar"
        );
    }

    #[test]
    fn test_try_promote_var_already_low() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // Variable at level 0 (default)
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Try to promote to level 0 — no-op since already at level 0
        let promoted = infer.try_promote_var(&mut ctx, var_id, 0);
        assert!(promoted.is_some(), "should return the existing var");
        let resolved = ctx.resolve_binding(var);
        assert!(
            matches!(ctx.get(resolved), TypeData::InferVar { id } if *id == var_id),
            "should be unchanged"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Generalization (PG → G) Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_force_generalize_pg_with_waitlist_stays_pg() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Set PG status and add a suspended constraint (wait list not empty)
        if var_id < infer.gen_statuses.len() {
            infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        }
        let int32_1 = ctx.int(32, true);
        infer.suspend_on_var(
            Constraint::Eq(var, int32_1, crate::ast::Span::new(0, 0)),
            var_id,
        );

        // Resolve the variable so it has a concrete type
        let int64 = ctx.int(64, false);
        ctx.bindings.borrow_mut().insert(var, int64);

        infer.force_generalize(&mut ctx);
        // Should remain PG because the wait list is not empty
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::PartiallyGeneralizable,
            "PG var with non-empty wait list should stay PG"
        );
    }

    #[test]
    fn test_force_generalize_dirty_set_triggers_generalization() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        // Resolve the variable
        ctx.bindings.borrow_mut().insert(var, ctx.bool());
        // Mark dirty
        infer.dirty_set.insert(var_id);

        infer.force_generalize(&mut ctx);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "resolved PG var in dirty set should become Generalized"
        );
    }

    #[test]
    fn test_force_generalize_for_regions() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.exit_level(prev);
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        let unit_ty = ctx.unit();
        ctx.bindings.borrow_mut().insert(var, unit_ty);

        // Use dirty_levels to trigger generalization
        let level = infer.get_var_level(var_id).unwrap_or(0);
        infer.force_generalize_for_regions(&mut ctx, &[level], None);
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Generalized,
            "PG var in dirty level should generalize"
        );
    }

    #[test]
    fn test_rigid_escape_generic_param_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(0, "T".into());
        let escaped = InferenceContext::check_rigid_escape(&ctx, gp, 0);
        assert!(escaped, "GenericParam should be detected as escape");
    }

    #[test]
    fn test_rigid_escape_concrete_not_detected() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let not_escaped = InferenceContext::check_rigid_escape(&ctx, int_ty, 0);
        assert!(!not_escaped, "Int<32> is not an escape");
    }

    #[test]
    fn test_rigid_escape_fn_with_gp_detected() {
        let mut ctx = TypeContext::new();
        let gp = ctx.generic_param(1, "U".into());
        let fn_ty = ctx.function(vec![gp], gp);
        let escaped = InferenceContext::check_rigid_escape(&ctx, fn_ty, 0);
        assert!(escaped, "fn(U) -> U contains GenericParam escape");
    }

    // ═══════════════════════════════════════════════════════════════
    // S-Inst-Copy Propagation Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_s_inst_copy_deepen_follows_aliases() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // Create an instance
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);

        // Resolve PG → fn(Int) → Bool
        let int32_2 = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![int32_2], bool_ty);
        ctx.bindings.borrow_mut().insert(pg, fn_ty);

        // S-Inst-Copy propagates the PG resolution to instances
        let resolved_pg = ctx.resolve_binding(pg);
        let updated = infer.s_inst_copy(&mut ctx, pg_id, resolved_pg);
        assert_eq!(updated, 1, "should have updated 1 instance");

        let inst_resolved = ctx.resolve_binding(inst);
        match ctx.get(inst_resolved) {
            TypeData::Fn { params, ret } => {
                assert_eq!(params.len(), 1, "instance should become a fn type");
                let p0_resolved = ctx.resolve_binding(params[0]);
                assert!(ctx.is_integer(p0_resolved), "param should be Int<32>");
                assert!(
                    ctx.is_bool(ctx.resolve_binding(*ret)),
                    "return should be Bool"
                );
            }
            other => panic!("instance should be fn type, got {:?}", other),
        }
    }

    #[test]
    fn test_s_inst_copy_pg_alias_resolved() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let pg = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let pg_id = match ctx.get(pg) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let inst = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let inst_id = match ctx.get(inst) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        infer.register_instance(pg_id, inst_id);
        let bool_ty = ctx.bool();
        ctx.bindings.borrow_mut().insert(pg, bool_ty);
        let resolved_pg = ctx.resolve_binding(pg);
        let updated = infer.s_inst_copy(&mut ctx, pg_id, resolved_pg);
        assert_eq!(updated, 1, "should have updated the instance");

        let inst_resolved = ctx.resolve_binding(inst);
        assert!(ctx.is_bool(inst_resolved), "instance should now be Bool");
    }

    // ═══════════════════════════════════════════════════════════════
    // S-Exists-Lower Tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_s_exists_lower_concrete() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        // Enter a deeper level so the var's level > current_level
        let prev = infer.enter_level();
        let var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        // S-Exists-Lower requires PG status and level > current_level
        infer.gen_statuses[var_id] = GenStatus::PartiallyGeneralizable;
        // exit_level so current_level drops below the var's level
        infer.exit_level(prev);
        let lowered = infer.s_exists_lower(&mut ctx, var_id);
        assert!(lowered, "S-Exists-Lower should succeed");
        assert_eq!(
            infer.gen_statuses[var_id],
            GenStatus::Ungeneralized,
            "PG var should become Ungeneralized after S-Exists-Lower"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration: Complete Solve with Shape Variables
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_solve_eq_concrete_success() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        let a = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let b = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        // Add constraint: Eq(a, b)
        infer.add_constraint(Constraint::Eq(a, b, crate::ast::Span::new(0, 0)));
        // Unify a with Int<32>
        let int_ty = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(a, int_ty);

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(result.is_ok(), "solve should succeed");
        // b should now be resolved to Int<32> too
        let b_resolved = ctx.resolve_binding(b);
        assert!(ctx.is_integer(b_resolved), "b should be Int<32>");
    }

    #[test]
    fn test_solve_level_promotion() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();
        let symbols = SymbolTable::new(crate::hir::types::CrateId(DefId(0)));
        let trait_env = TraitEnv::new();

        // Create a variable at a deeper level
        let prev = infer.enter_level();
        let deep_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let shallow_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        infer.exit_level(prev);

        // Eq(deep, shallow) — should promote deep to shallow's level
        infer.add_constraint(Constraint::Eq(
            deep_var,
            shallow_var,
            crate::ast::Span::new(0, 0),
        ));
        let bool_ty = ctx.bool();
        ctx.bindings.borrow_mut().insert(shallow_var, bool_ty);

        let result = infer.solve(&mut ctx, &trait_env, &symbols);
        assert!(result.is_ok(), "level promotion solve should succeed");
        let deep_resolved = ctx.resolve_binding(deep_var);
        assert!(
            ctx.is_bool(deep_resolved),
            "deep var should resolve to Bool"
        );
    }

    #[test]
    fn test_try_match_via_shape_var_suspend_discharge_roundtrip() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Register a match branch: Arrow → Eq(Int, Int)
        let int32 = ctx.int(32, true);
        let branches = vec![MatchBranchSet {
            shape_pattern: PrincipalShape::Arrow,
            continuation: vec![Constraint::Eq(int32, int32, crate::ast::Span::new(0, 0))],
            else_continuation: Vec::new(),
        }];
        let bid = infer.register_match_branches(branches);

        // Create an infer var and try_match via shape var (should suspend)
        let infer_var = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let var_id = match ctx.get(infer_var) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        let mut heap = std::collections::BinaryHeap::new();
        let _handled = infer.try_match_via_shape_var(&mut ctx, infer_var, bid, &mut heap);
        // try_match_via_shape_var always returns true (handled) but for an InferVar
        // it should suspend on the wait list, not discharge
        assert!(
            var_id < infer.wait_lists.len() && !infer.wait_lists[var_id].is_empty(),
            "match should be suspended on the infer var's wait list"
        );

        // Now resolve the infer var to a concrete fn type and wake it
        let fn_bool = ctx.bool();
        let fn_ty = ctx.function(vec![fn_bool], fn_bool);
        ctx.bindings.borrow_mut().insert(infer_var, fn_ty);
        infer.wake_var_incremental(var_id, &mut heap, &ctx);

        // The match should now be in the heap
        assert!(!heap.is_empty(), "match should be woken and placed in heap");
        let woken = heap.pop().unwrap();
        assert!(
            matches!(woken.constraint, Constraint::Match { .. }),
            "woken constraint should be Match"
        );

        // Discharge it directly
        let fn_ty2 = ctx.function(vec![ctx.bool()], ctx.bool());
        let discharged = infer.discharge_match(&mut ctx, fn_ty2, bid, &mut heap);
        assert!(discharged, "match on fn type should discharge");
        // The continuation Eq(int32, int32) should be in the heap now
        assert!(
            !heap.is_empty(),
            "continuation constraints should be enqueued"
        );
    }

    #[test]
    fn test_force_generalize_after_solve_completes_pg() {
        let mut ctx = TypeContext::new();
        let mut infer = InferenceContext::new();

        // Create variables at deeper scope (simulating let-polymorphism)
        let _prev = infer.enter_level();
        let x = infer.new_type_var(&mut ctx, TypeVariableKind::Any);
        let x_id = match ctx.get(x) {
            TypeData::InferVar { id } => *id,
            _ => unreachable!(),
        };
        assert!(
            infer.get_var_level(x_id).unwrap_or(0) > 0,
            "x should be at a deeper level"
        );

        // Bind x to Int<32>
        let int32 = ctx.int(32, true);
        ctx.bindings.borrow_mut().insert(x, int32);

        // Mark the variable PG and then force_generalize
        infer.gen_statuses[x_id] = GenStatus::PartiallyGeneralizable;
        infer.force_generalize(&mut ctx);

        assert_eq!(
            infer.gen_statuses[x_id],
            GenStatus::Generalized,
            "resolved PG var at inner scope should generalize"
        );
    }
}
