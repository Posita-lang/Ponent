use crate::ast::Span;
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
}

impl Constraint {
    /// Compute priority: lower = more deterministic, processed first.
    /// This enables BinaryHeap-based scheduling where concrete-concrete
    /// constraints are resolved before those involving inference variables.
    pub fn priority(&self, ctx: &TypeContext) -> u8 {
        match self {
            Constraint::Eq(a, b, _) => {
                let a_is_infer = matches!(ctx.get(ctx.resolve_binding(*a)), TypeData::InferVar { .. });
                let b_is_infer = matches!(ctx.get(ctx.resolve_binding(*b)), TypeData::InferVar { .. });
                match (a_is_infer, b_is_infer) {
                    (false, false) => 0, // concrete-concrete: highest priority
                    (true, false) | (false, true) => 1, // one infer var
                    (true, true) => 2, // both infer vars
                }
            }
            Constraint::Sub(sub, sup, _) => {
                let sub_is_infer = matches!(ctx.get(ctx.resolve_binding(*sub)), TypeData::InferVar { .. });
                let sup_is_infer = matches!(ctx.get(ctx.resolve_binding(*sup)), TypeData::InferVar { .. });
                match (sub_is_infer, sup_is_infer) {
                    (false, false) => 3,
                    _ => 4,
                }
            }
            Constraint::Impl(..) => 5, // trait impl checks: lowest priority
        }
    }
}

#[derive(Debug, Clone)]
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
        }
    }

    pub fn new_type_var(&mut self, ctx: &mut TypeContext, kind: TypeVariableKind) -> TypeId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        let ty_id = ctx.alloc_infer_var(id);
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
        ty_id
    }

    /// Look up the kind of a type variable by its id.
    pub fn get_var_kind(&self, id: usize) -> Option<TypeVariableKind> {
        self.type_vars.iter().find(|tv| tv.id == id).map(|tv| tv.kind)
    }

    /// Get the level of a type variable by its id.
    pub fn get_var_level(&self, id: usize) -> Option<usize> {
        self.type_vars.iter().find(|tv| tv.id == id).map(|tv| tv.level)
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
    pub fn try_promote_var(&mut self, ctx: &mut TypeContext, var_id: usize, target_level: usize) -> Option<TypeId> {
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
        ctx.bindings.borrow_mut().insert(self.var_type_ids[var_id], new_ty_id);
        Some(new_ty_id)
    }

    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    /// OmniML-inspired: suspend a constraint on the target InferVar id.
    /// When the var is bound, the constraint will be woken and reprocessed.
    pub fn suspend_on_var(&mut self, c: Constraint, var_id: usize) {
        if var_id < self.wait_lists.len() {
            self.wait_lists[var_id].push(c);
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
                if let TypeData::InferVar { id } = ctx.get(ra) { return Some(*id); }
                if let TypeData::InferVar { id } = ctx.get(rb) { return Some(*id); }
                None
            }
            Constraint::Sub(sub, sup, _) => {
                let rs = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                if let TypeData::InferVar { id } = ctx.get(rs) { return Some(*id); }
                if let TypeData::InferVar { id } = ctx.get(rsup) { return Some(*id); }
                None
            }
            Constraint::Impl(ty, ..) => {
                let r = ctx.resolve_binding(*ty);
                if let TypeData::InferVar { id } = ctx.get(r) { Some(*id) } else { None }
            }
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
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                PrincipalShape::Constructor(args.len())
            }
            TypeData::Forall { .. } | TypeData::Exists { .. } => PrincipalShape::Poly,
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
    fn wake_var_incremental(&mut self, var_id: usize, heap: &mut BinaryHeap<PrioritizedConstraint>, ctx: &TypeContext) {
        if var_id < self.wait_lists.len() && !self.wait_lists[var_id].is_empty() {
            let suspended = std::mem::take(&mut self.wait_lists[var_id]);
            for c in suspended {
                let p = c.priority(ctx);
                heap.push(PrioritizedConstraint { priority: p, constraint: c });
            }
        }
    }

    pub fn solve(&mut self, ctx: &mut TypeContext, trait_env: &TraitEnv, symbols: &SymbolTable) -> Result<(), TypeError> {
        // ── Build priority queue ────────────────────────────────────
        let mut heap: BinaryHeap<PrioritizedConstraint> = BinaryHeap::new();
        for c in &self.constraints {
            let priority = c.priority(ctx);
            heap.push(PrioritizedConstraint { priority, constraint: c.clone() });
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
                        let a_var_id = if a_is_infer { if let TypeData::InferVar { id } = ctx.get(ra) { Some(*id) } else { None } } else { None };
                        let b_var_id = if b_is_infer { if let TypeData::InferVar { id } = ctx.get(rb) { Some(*id) } else { None } } else { None };

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

                        // Incremental wake-up: if a variable was just resolved,
                        // immediately enqueue its suspended constraints.
                        for var_id in [a_var_id, b_var_id].iter().flatten() {
                            if self.try_set_shape(*var_id, ctx) {
                                self.wake_var_incremental(*var_id, &mut heap, ctx);
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
                        }
                    }
                    // If sub is an InferVar, record sup as an upper bound of sub
                    if let TypeData::InferVar { id } = ctx.get(resolved_sub) {
                        if *id < self.upper_bounds.len() {
                            self.upper_bounds[*id].push(resolved_sup);
                        }
                    }

                    // If both sides are resolved (not InferVar), check the subtype relationship now
                    let sub_is_infer = matches!(ctx.get(resolved_sub), TypeData::InferVar { .. });
                    let sup_is_infer = matches!(ctx.get(resolved_sup), TypeData::InferVar { .. });
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
                        trait_env.lookup_impl_generic(*trait_id, resolved, ctx, symbols).is_some()
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
                    heap.push(PrioritizedConstraint { priority: p, constraint: c });
                }
                woken += count;
            }
        }
    }
    if woken == 0 {
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
                        TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::USize
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

        // Defaulting: unfilled infer vars get default types
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if let TypeData::InferVar { .. } = ctx.get(resolved) {
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
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit
        | TypeData::Error => ty,
        TypeData::GenericParam { .. } => ty,
        TypeData::Struct { def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Struct {
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Enum { def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Enum {
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
        TypeData::Forall { param_index, param_name, body } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Forall { param_index, param_name, body: new_body })
                .unwrap_or(ctx.error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_ctx() -> TypeContext { TypeContext::new() }

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
        infer.suspend_on_var(Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)), var_id);
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
        infer.suspend_on_var(Constraint::Impl(ctx.bool(), DefId(0), crate::ast::Span::new(0, 0)), var_id);
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
}
