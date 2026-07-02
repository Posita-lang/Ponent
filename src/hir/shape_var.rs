//! Shape Variable System — PrinciPal Shape Variables (OmniML §3.3, §6).
//!
//! A shape variable represents a *not-yet-known* principal shape. Unlike the
//! old `PrincipalShape` enum (which is a simple tag), shape variables are
//! first-class unifiable variables that participate in unification.
//!
//! When two shape variables unify, their wait lists merge. When a shape
//! variable is assigned a concrete shape (via unification with a concrete
//! type constructor), all constraints in its wait list are woken and
//! processed — this is the "yield/resume" mechanism for suspended match
//! constraints.
//!
//! Shape variables enable the key OmniML properties:
//! - Partial generalisation (PG): a variable guarded by a shape variable can
//!   be partially generalised even though its shape is not yet known.
//! - Incremental instantiation: when the shape variable is resolved, all
//!   instances of the PG variable are re-unified via S-Inst-Copy.

use crate::hir::types::*;
use std::collections::HashMap;

// ── Re-exports ─────────────────────────────────────────────────────

pub use self::guard_set::GuardSet;
pub use self::status::DirtyStatus;

// ── Guard Set Module (OmniML §6) ──────────────────────────────────
//
// A guard set tracks how many constraints are "blocking" a variable
// from being generalised. It has two components:
//   - `direct_guards`: reference count of direct suspended match constraints
//   - `transitive_guards`: map from transitive guard ID to count
//
// When all guards are removed, the variable can transition from PG to G.

pub mod guard_set {
    use std::collections::HashMap;

    /// A guard identifier for transitive guards.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TransitiveGuardId(pub usize);

    /// Guard set: tracks direct and transitive guards on a variable.
    #[derive(Debug, Clone)]
    pub struct GuardSet {
        /// Direct guard count — number of suspended match constraints
        /// directly referencing this variable.
        pub direct_guards: usize,
        /// Transitive guards — map from guard ID to count.
        /// A transitive guard means variable A is blocked by variable B's
        /// shape resolution: A holds a transitive guard on B.
        pub transitive_guards: HashMap<usize, usize>,
    }

    impl GuardSet {
        pub fn new() -> Self {
            GuardSet {
                direct_guards: 0,
                transitive_guards: HashMap::new(),
            }
        }

        pub fn is_empty(&self) -> bool {
            self.direct_guards == 0 && self.transitive_guards.is_empty()
        }

        /// Add a direct guard.
        pub fn add_direct(&mut self) {
            self.direct_guards += 1;
        }

        /// Remove a direct guard. Returns the new count.
        pub fn remove_direct(&mut self) -> usize {
            if self.direct_guards > 0 {
                self.direct_guards -= 1;
            }
            self.direct_guards
        }

        /// Add a transitive guard on `guard_id`. Returns the new count for that guard.
        pub fn add_transitive(&mut self, guard_id: usize) -> usize {
            let entry = self.transitive_guards.entry(guard_id).or_insert(0);
            *entry += 1;
            *entry
        }

        /// Remove one occurrence of a transitive guard. Returns the remaining count.
        pub fn remove_transitive(&mut self, guard_id: usize) -> usize {
            if let Some(count) = self.transitive_guards.get_mut(&guard_id) {
                *count -= 1;
                let remaining = *count;
                if *count == 0 {
                    self.transitive_guards.remove(&guard_id);
                }
                remaining
            } else {
                0
            }
        }

        /// Clear all transitive guards for a given ID.
        pub fn clear_transitive(&mut self, guard_id: usize) {
            self.transitive_guards.remove(&guard_id);
        }

        /// Check if a transitive guard is active.
        pub fn is_transitively_guarded(&self, guard_id: usize) -> bool {
            self.transitive_guards.get(&guard_id).copied().unwrap_or(0) > 0
        }

        /// Merge two guard sets.
        pub fn union(&self, other: &Self) -> Self {
            let mut tg = self.transitive_guards.clone();
            for (&k, &v) in &other.transitive_guards {
                *tg.entry(k).or_insert(0) += v;
            }
            GuardSet {
                direct_guards: self.direct_guards + other.direct_guards,
                transitive_guards: tg,
            }
        }
    }

    impl Default for GuardSet {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ── Generalisation Status Module (OmniML §6) ──────────────────────
//
// Status tracks whether a type is fully generalised (Generic) or still
// an instance with optional dirty marking.

pub mod status {
    /// The generalisation status of a type node.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DirtyStatus {
        /// Instance — not yet fully generalised.
        /// `dirty` indicates the owning region has updates to propagate.
        Instance { dirty: bool },
        /// Fully generalised — belongs to no region.
        Generic,
    }

    impl DirtyStatus {
        pub fn is_dirty(&self) -> bool {
            match self {
                DirtyStatus::Instance { dirty } => *dirty,
                DirtyStatus::Generic => false,
            }
        }

        pub fn is_generic(&self) -> bool {
            matches!(self, DirtyStatus::Generic)
        }

        pub fn set_dirty(&mut self) {
            match self {
                DirtyStatus::Instance { dirty } => *dirty = true,
                DirtyStatus::Generic => {}
            }
        }

        /// Merge two statuses on type unification.
        /// Any update marks the result as dirty.
        pub fn merge(a: Self, b: Self) -> Self {
            match (a, b) {
                (DirtyStatus::Generic, _) | (_, DirtyStatus::Generic) => {
                    // If either is Generic, the result is Instance with dirty=true
                    // (since merging with generic preserves the instance status)
                    DirtyStatus::Instance { dirty: true }
                }
                (DirtyStatus::Instance { .. }, DirtyStatus::Instance { .. }) => {
                    DirtyStatus::Instance { dirty: true }
                }
            }
        }
    }
}

use crate::hir::types::*;
use std::collections::VecDeque;

/// Unique identifier for a shape variable.
/// Parallels `InferVar`'s id — each shape variable has a dense ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeVarId(pub usize);

/// The resolved shape of a shape variable: a type constructor tag.
/// Maps 1-to-1 to the type structure (Fn, Tuple, Struct, etc.).
/// This is what a shape variable resolves TO when it becomes known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeShape {
    /// Unknown — shape variable not yet resolved.
    Unknown,
    /// Arrow (function) type: τ₁ → τ₂
    Arrow,
    /// Tuple type: τ₁ × τ₂ × … × τₙ
    Tuple(usize),
    /// Named constructor (struct/enum): C τ₁ τ₂ … τₙ
    Constructor(usize),
    /// Polymorphic/existential/fixpoint container: ∀α.τ / ∃α.τ / μα.τ / να.τ
    Poly,
}

/// A callback to be invoked when a shape variable is resolved.
pub type ShapeCallback = Box<dyn FnOnce(TypeShape) + Send>;

/// Internal state for a single shape variable.
pub struct ShapeVar {
    pub id: ShapeVarId,
    /// The resolved shape, or `Unknown` if not yet resolved.
    pub resolved: Option<TypeShape>,
    /// The level at which this shape variable was created.
    pub level: usize,
    /// Wait list: callbacks that fire when this shape var is resolved.
    pub wait_list: Vec<ShapeCallback>,
    /// If this shape var has been unified with another, the other's id.
    pub alias: Option<ShapeVarId>,
}

impl std::fmt::Debug for ShapeVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeVar")
            .field("id", &self.id)
            .field("resolved", &self.resolved)
            .field("level", &self.level)
            .field("wait_list_len", &self.wait_list.len())
            .field("alias", &self.alias)
            .finish()
    }
}

/// The shape variable state machine, owned by `InferenceContext`.
#[derive(Debug)]
pub struct ShapeVarContext {
    vars: Vec<ShapeVar>,
    next_id: usize,
}

impl ShapeVarContext {
    pub fn new() -> Self {
        ShapeVarContext {
            vars: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a new unresolved shape variable at the given level.
    pub fn new_var(&mut self, level: usize) -> ShapeVarId {
        let id = ShapeVarId(self.next_id);
        self.next_id += 1;
        self.vars.push(ShapeVar {
            id,
            resolved: None,
            level,
            wait_list: Vec::new(),
            alias: None,
        });
        id
    }

    /// Resolve `id` to a canonical shape variable ID (following aliases).
    pub fn resolve(&self, id: ShapeVarId) -> ShapeVarId {
        let mut current = id;
        loop {
            if current.0 >= self.vars.len() {
                return current;
            }
            match self.vars[current.0].alias {
                Some(alias) => current = alias,
                None => return current,
            }
        }
    }

    /// Get the resolved shape of a shape variable, or `None` if unresolved.
    pub fn get(&self, id: ShapeVarId) -> Option<TypeShape> {
        let canonical = self.resolve(id);
        if canonical.0 >= self.vars.len() {
            return None;
        }
        self.vars[canonical.0].resolved
    }

    /// Get the level of a shape variable.
    pub fn get_level(&self, id: ShapeVarId) -> usize {
        let canonical = self.resolve(id);
        if canonical.0 >= self.vars.len() {
            return 0;
        }
        self.vars[canonical.0].level
    }

    /// Register a callback on a shape variable, to fire when resolved.
    pub fn on_resolve<F>(&mut self, id: ShapeVarId, callback: F)
    where
        F: FnOnce(TypeShape) + Send + 'static,
    {
        let canonical = self.resolve(id);
        if canonical.0 >= self.vars.len() {
            return;
        }
        // If already resolved, fire immediately.
        if let Some(shape) = self.vars[canonical.0].resolved {
            callback(shape);
            return;
        }
        self.vars[canonical.0].wait_list.push(Box::new(callback));
    }

    /// Unify two shape variables: they become the same variable.
    /// Returns `true` if any callbacks were fired.
    pub fn unify(&mut self, a: ShapeVarId, b: ShapeVarId) -> bool {
        if a == b {
            return false;
        }
        let ca = self.resolve(a);
        let cb = self.resolve(b);
        if ca == cb {
            return false;
        }

        // Merge the higher-index into the lower for deterministic ordering.
        let (target, source) = if ca.0 <= cb.0 { (ca, cb) } else { (cb, ca) };

        // Drain source's wait list into target.
        let source_wait = std::mem::take(&mut self.vars[source.0].wait_list);
        self.vars[target.0].wait_list.extend(source_wait);

        // If source is resolved, propagate to target.
        let mut fired = false;
        if let Some(shape) = self.vars[source.0].resolved {
            fired = self.try_set(target, shape);
        }

        // Mark source as alias of target.
        self.vars[source.0].alias = Some(target);

        // If target is already resolved and source wasn't, fire source's waiters.
        if !fired {
            if let Some(shape) = self.vars[target.0].resolved {
                let pending = std::mem::take(&mut self.vars[target.0].wait_list);
                for cb in pending {
                    cb(shape);
                }
                fired = true;
            }
        }

        fired
    }

    /// Try to set a shape variable to a concrete shape.
    /// Returns `true` if any callbacks were fired.
    pub fn try_set(&mut self, id: ShapeVarId, shape: TypeShape) -> bool {
        let canonical = self.resolve(id);
        if canonical.0 >= self.vars.len() {
            return false;
        }
        let sv = &mut self.vars[canonical.0];
        if let Some(existing) = sv.resolved {
            // Already resolved — shapes must match.
            return existing == shape;
        }
        sv.resolved = Some(shape);
        let pending = std::mem::take(&mut sv.wait_list);
        for cb in pending {
            cb(shape);
        }
        true
    }

    /// Check if a shape variable is resolved.
    pub fn is_resolved(&self, id: ShapeVarId) -> bool {
        self.get(id).is_some()
    }

    /// Number of unresolved shape variables.
    pub fn num_unresolved(&self) -> usize {
        self.vars.iter().filter(|sv| sv.resolved.is_none()).count()
    }

    /// Collect all unresolved shape variable IDs at levels above `max_level`.
    pub fn unresolved_above_level(&self, max_level: usize) -> Vec<ShapeVarId> {
        self.vars
            .iter()
            .filter(|sv| sv.resolved.is_none() && sv.level > max_level)
            .map(|sv| sv.id)
            .collect()
    }
}

/// Check whether two `TypeShape` values are compatible for unification.
pub fn shapes_compatible(a: TypeShape, b: TypeShape) -> bool {
    match (a, b) {
        (TypeShape::Unknown, _) | (_, TypeShape::Unknown) => true,
        (TypeShape::Arrow, TypeShape::Arrow) => true,
        (TypeShape::Tuple(n), TypeShape::Tuple(m)) => n == m,
        (TypeShape::Constructor(_), TypeShape::Constructor(_)) => true,
        (TypeShape::Poly, TypeShape::Poly) => true,
        _ => false,
    }
}

/// Decompose a `TypeData` into its shape tag.
pub fn type_data_to_shape(data: &TypeData) -> TypeShape {
    match data {
        TypeData::Fn { .. } => TypeShape::Arrow,
        TypeData::Tuple { elems } => TypeShape::Tuple(elems.len()),
        TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
            TypeShape::Constructor(args.len())
        }
        TypeData::Forall { .. }
        | TypeData::Exists { .. }
        | TypeData::Mu { .. }
        | TypeData::Nu { .. }
        | TypeData::Poly { .. } => TypeShape::Poly,
        TypeData::Coproduct { alternatives } => TypeShape::Tuple(alternatives.len()),
        _ => TypeShape::Unknown,
    }
}
