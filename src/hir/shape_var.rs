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
    Constructor(usize, usize),
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
    /// O(1) unresolved count: incremented in `new_var`, decremented at the
    /// single `None → Some` transition site in `try_set`, and repaired in
    /// `truncate_vars` for truncated variables.
    unresolved_count: usize,
    /// Unresolved (canonical, non-alias) shape vars grouped by level.
    /// Query for `unresolved_above_level` walks buckets with level >
    /// `max_level` instead of scanning the whole var table.  Buckets are
    /// indexed by level; `vars.len()` bounds the highest level in practice
    /// (levels come from the region tree, which is small).
    unresolved_by_level: Vec<Vec<ShapeVarId>>,
    /// Position of each shape var id inside its `unresolved_by_level` bucket,
    /// for O(1) swap-remove.  An entry only exists while the var is
    /// unresolved AND canonical (aliases are removed from buckets when
    /// created in `unify`).
    bucket_pos: HashMap<usize, usize>,
}

impl ShapeVarContext {
    pub fn new() -> Self {
        ShapeVarContext {
            vars: Vec::new(),
            next_id: 0,
            unresolved_count: 0,
            unresolved_by_level: Vec::new(),
            bucket_pos: HashMap::new(),
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
        // Index maintenance: this var is unresolved and canonical.
        self.unresolved_count += 1;
        if self.unresolved_by_level.len() <= level {
            self.unresolved_by_level.resize(level + 1, Vec::new());
        }
        self.unresolved_by_level[level].push(id);
        self.bucket_pos.insert(id.0, self.unresolved_by_level[level].len() - 1);
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
            if !fired && self.vars[target.0].resolved.is_some() {
                // Both source and target are resolved with incompatible shapes.
                // Do NOT set the alias — merging incompatible shapes silently
                // would violate type soundness.  Return false so callers can
                // detect the mismatch.
                return false;
            }
        }

        // Mark source as alias of target.
        self.vars[source.0].alias = Some(target);
        // Index maintenance: the source is no longer a canonical variable —
        // remove it from its level bucket so queries report canonical vars
        // only.  If the source was still unresolved, the merged class is now
        // counted exactly once (via the target), so the counter drops too.
        // (The target keeps its bucket entry; if it later resolves,
        // `try_set` removes it.)
        if self.vars[source.0].resolved.is_none() {
            self.unresolved_count -= 1;
        }
        self.remove_from_bucket(source);

        // If target is already resolved and source wasn't, fire source's waiters.
        if !fired && let Some(shape) = self.vars[target.0].resolved {
            let pending = std::mem::take(&mut self.vars[target.0].wait_list);
            for cb in pending {
                cb(shape);
            }
            fired = true;
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
        // Index maintenance: the single `None → Some` transition site.
        // The var was unresolved (it is canonical by `resolve`), so it must
        // have a bucket entry.
        self.unresolved_count -= 1;
        self.remove_from_bucket(canonical);
        for cb in pending {
            cb(shape);
        }
        true
    }

    /// Check if a shape variable is resolved.
    pub fn is_resolved(&self, id: ShapeVarId) -> bool {
        self.get(id).is_some()
    }

    /// Number of unresolved canonical shape variables (O(1) counter).
    ///
    /// Counts equivalence CLASSES whose canonical var is unresolved — an
    /// alias never counts separately.  Invariant: the counter equals the
    /// size of the union of ALL per-level buckets (verified by
    /// `test_shape_var_unresolved_index_consistency`).
    pub fn num_unresolved(&self) -> usize {
        self.unresolved_count
    }

    /// Collect all unresolved (canonical) shape variable IDs at all levels.
    /// Exposed for tests/debug; equivalent to summing every bucket.
    #[cfg(test)]
    pub(crate) fn unresolved_all(&self) -> Vec<ShapeVarId> {
        self.unresolved_by_level.iter().flatten().copied().collect()
    }

    /// Collect all unresolved (canonical) shape variable IDs at levels
    /// above `max_level`.  Walks only the per-level buckets above the
    /// threshold instead of scanning the whole var table.
    pub fn unresolved_above_level(&self, max_level: usize) -> Vec<ShapeVarId> {
        let mut out = Vec::new();
        for (level, bucket) in self.unresolved_by_level.iter().enumerate() {
            if level > max_level {
                out.extend_from_slice(bucket);
            }
        }
        out
    }

    /// O(1) bucket removal via swap-remove, using `bucket_pos` for the
    /// index.  A no-op if the var has no bucket entry (already removed,
    /// or removed as truncated — see `truncate_vars`).
    fn remove_from_bucket(&mut self, id: ShapeVarId) {
        if let Some(pos) = self.bucket_pos.remove(&id.0) {
            let level = self.vars[id.0].level;
            let bucket = &mut self.unresolved_by_level[level];
            debug_assert!(
                pos < bucket.len() && bucket[pos] == id,
                "bucket_pos desync for shape var {id:?}"
            );
            let last = bucket.len() - 1;
            if pos != last {
                let moved = bucket[last];
                bucket[pos] = moved;
                self.bucket_pos.insert(moved.0, pos);
            }
            bucket.pop();
        }
    }

    /// Number of shape variables allocated (for snapshot truncation).
    pub(crate) fn vars_len(&self) -> usize {
        self.vars.len()
    }

    /// The next fresh shape variable ID (for snapshot restoration).
    pub(crate) fn next_id_val(&self) -> usize {
        self.next_id
    }

    /// Truncate shape variables to a saved length (rollback).
    ///
    /// Repairs the unresolved-count/level-bucket index: truncated tail
    /// variables may be resolved (their resolution is NOT individually
    /// undo-logged — the whole var is dropped) or unresolved-canonical
    /// (their bucket entries must go).  Cost is O(truncated tail), i.e.
    /// amortized O(1) per rolled-back variable.
    pub(crate) fn truncate_vars(&mut self, len: usize) {
        if len >= self.vars.len() {
            return;
        }
        // Repair the index for the truncated tail: a tail var with a
        // bucket entry was canonical AND unresolved — its counter
        // contribution goes away with it.  (Resolved or alias tail vars
        // were never counted, so they only need their bucket_pos entry
        // dropped, which remove_from_bucket's no-op path handles.)
        let tail: Vec<ShapeVarId> = self.vars[len..].iter().map(|sv| sv.id).collect();
        for id in tail {
            if self.bucket_pos.remove(&id.0).is_some() {
                self.unresolved_count -= 1;
            }
        }
        // Drop bucket entries pointing into the truncated tail.
        for bucket in self.unresolved_by_level.iter_mut() {
            bucket.retain(|id| (id.0 as usize) < len);
        }
        while matches!(self.unresolved_by_level.last(), Some(b) if b.is_empty()) {
            self.unresolved_by_level.pop();
        }
        self.vars.truncate(len);
        if self.next_id > len {
            self.next_id = len;
        }
    }

    /// Restore the next_id counter (rollback).
    pub(crate) fn set_next_id(&mut self, id: usize) {
        self.next_id = id;
    }
}

/// Check whether two `TypeShape` values are compatible for unification.
pub fn shapes_compatible(a: TypeShape, b: TypeShape) -> bool {
    match (a, b) {
        (TypeShape::Unknown, _) | (_, TypeShape::Unknown) => true,
        (TypeShape::Arrow, TypeShape::Arrow) => true,
        (TypeShape::Tuple(n), TypeShape::Tuple(m)) => n == m,
        (TypeShape::Constructor(id1, _), TypeShape::Constructor(id2, _)) => id1 == id2,
        (TypeShape::Poly, TypeShape::Poly) => true,
        _ => false,
    }
}

/// Decompose a `TypeData` into its shape tag.
pub fn type_data_to_shape(data: &TypeData) -> TypeShape {
    match data {
        TypeData::Fn { .. } => TypeShape::Arrow,
        TypeData::Tuple { elems } => TypeShape::Tuple(elems.len()),
        TypeData::Adt { def_id, args, .. } => TypeShape::Constructor(def_id.0, args.len()),
        TypeData::Forall { .. }
        | TypeData::Exists { .. }
        | TypeData::Mu { .. }
        | TypeData::Nu { .. }
        | TypeData::Poly { .. }
        | TypeData::SkolemVar { .. } => TypeShape::Poly,
        TypeData::Rational { .. } => TypeShape::Unknown,
        TypeData::Coproduct { alternatives } => TypeShape::Tuple(alternatives.len()),
        _ => TypeShape::Unknown,
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_shape_var_new_and_resolve() {
        let mut svc = ShapeVarContext::new();
        let id = svc.new_var(1);
        assert!(!svc.is_resolved(id));
        assert_eq!(svc.get_level(id), 1);
        assert_eq!(svc.resolve(id), id);
    }

    #[test]
    fn test_shape_var_try_set() {
        let mut svc = ShapeVarContext::new();
        let id = svc.new_var(0);
        assert!(!svc.is_resolved(id));
        assert!(svc.try_set(id, TypeShape::Arrow));
        assert!(svc.is_resolved(id));
        assert_eq!(svc.get(id), Some(TypeShape::Arrow));
    }

    #[test]
    fn test_shape_var_unify_creates_alias() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        assert_ne!(svc.resolve(a), svc.resolve(b));
        svc.unify(a, b);
        assert_eq!(svc.resolve(a), svc.resolve(b));
    }

    #[test]
    fn test_shape_var_callback_fires_on_try_set() {
        let mut svc = ShapeVarContext::new();
        let id = svc.new_var(0);
        let fired = Arc::new(AtomicBool::new(false));
        let f = Arc::clone(&fired);
        svc.on_resolve(id, move |_| f.store(true, Ordering::SeqCst));
        assert!(!fired.load(Ordering::SeqCst));
        svc.try_set(id, TypeShape::Arrow);
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn test_shape_var_callback_fires_immediately_if_resolved() {
        let mut svc = ShapeVarContext::new();
        let id = svc.new_var(0);
        svc.try_set(id, TypeShape::Arrow);
        let fired = Arc::new(AtomicBool::new(false));
        let f = Arc::clone(&fired);
        svc.on_resolve(id, move |_| f.store(true, Ordering::SeqCst));
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn test_shape_var_unify_merges_wait_lists() {
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(0);
        let a_fired = Arc::new(AtomicBool::new(false));
        let b_fired = Arc::new(AtomicBool::new(false));
        let af = Arc::clone(&a_fired);
        let bf = Arc::clone(&b_fired);
        svc.on_resolve(a, move |_| af.store(true, Ordering::SeqCst));
        svc.on_resolve(b, move |_| bf.store(true, Ordering::SeqCst));
        svc.unify(a, b);
        svc.try_set(svc.resolve(a), TypeShape::Arrow);
        assert!(a_fired.load(Ordering::SeqCst));
        assert!(b_fired.load(Ordering::SeqCst));
    }

    #[test]
    fn test_shape_var_num_unresolved() {
        let mut svc = ShapeVarContext::new();
        assert_eq!(svc.num_unresolved(), 0);
        svc.new_var(0);
        svc.new_var(0);
        assert_eq!(svc.num_unresolved(), 2);
    }

    #[test]
    fn test_shape_var_unresolved_index_consistency() {
        // Counter + bucket index must stay consistent across the three
        // mutation paths: try_set (resolution), unify (aliasing), and
        // truncate_vars (rollback).
        let mut svc = ShapeVarContext::new();
        let a = svc.new_var(0);
        let b = svc.new_var(1);
        let c = svc.new_var(2);
        assert_eq!(svc.num_unresolved(), 3);
        assert_eq!(svc.unresolved_above_level(0).len(), 2);
        assert_eq!(svc.unresolved_above_level(1).len(), 1);

        // Resolve one — counter and bucket drop together.
        assert!(svc.try_set(b, TypeShape::Arrow));
        assert_eq!(svc.num_unresolved(), 2);
        assert_eq!(svc.unresolved_above_level(1).len(), 1); // only c
        // Re-set is a no-op for the counter.
        assert!(svc.try_set(b, TypeShape::Arrow));
        assert_eq!(svc.num_unresolved(), 2);

        // Alias the pair: unify merges the HIGHER id into the lower, so
        // c becomes an alias of a (canonical = a, level 0).  c leaves its
        // level-2 bucket; the merged class counts ONCE via a.
        svc.unify(a, c);
        let canonical = svc.resolve(a);
        assert_eq!(canonical, a);
        assert_eq!(svc.resolve(c), a);
        // Nothing above level 1 anymore: b is resolved, c is an alias.
        assert!(svc.unresolved_above_level(1).is_empty());
        // The full bucket union still holds exactly the canonical pair.
        assert_eq!(svc.unresolved_all(), vec![a]);
        assert_eq!(svc.num_unresolved(), 1);

        // Rollback truncation: cut back to just `a`'s slot.  a is
        // unresolved-canonical, so exactly one class survives.
        svc.truncate_vars(1);
        assert_eq!(svc.num_unresolved(), 1); // only a (unresolved)
        assert!(!svc.is_resolved(a));
        assert_eq!(svc.unresolved_above_level(0).len(), 0);
        assert_eq!(svc.unresolved_all(), vec![a]);
        // After truncation, new ids must not collide.
        let d = svc.new_var(0);
        assert_eq!(d.0, 1);
        assert_eq!(svc.num_unresolved(), 2);
        // Full truncation resets to empty.
        svc.truncate_vars(0);
        assert_eq!(svc.num_unresolved(), 0);
        assert!(svc.unresolved_above_level(0).is_empty());
        let e = svc.new_var(3);
        assert_eq!(e.0, 0);
        assert_eq!(svc.unresolved_above_level(2), vec![e]);
        assert_eq!(svc.unresolved_all(), vec![e]);
    }

    #[test]
    fn test_shape_var_callback_reentrancy_keeps_counter_sane() {
        // A wait-list callback that re-enters try_set/unify during the
        // resolution transition must not corrupt the counter or buckets.
        // (Raw-pointer re-entry stands in for the solver's callback path,
        // which reaches the same context through its own handle.)
        let mut svc = ShapeVarContext::new();
        let target = svc.new_var(0);
        let other = svc.new_var(1);
        let svc_ptr = &mut svc as *mut ShapeVarContext as usize;
        svc.on_resolve(target, move |shape| {
            let inner = unsafe { &mut *(svc_ptr as *mut ShapeVarContext) };
            let _ = inner.try_set(other, shape);
        });
        unsafe {
            let svc = &mut *(svc_ptr as *mut ShapeVarContext);
            assert!(svc.try_set(target, TypeShape::Poly));
            assert_eq!(svc.num_unresolved(), 0);
            assert!(svc.unresolved_all().is_empty());
            assert!(svc.unresolved_above_level(0).is_empty());
        }
    }

    #[test]
    fn test_shapes_compatible() {
        assert!(shapes_compatible(TypeShape::Unknown, TypeShape::Arrow));
        assert!(shapes_compatible(TypeShape::Arrow, TypeShape::Arrow));
        assert!(!shapes_compatible(TypeShape::Arrow, TypeShape::Tuple(2)));
        assert!(!shapes_compatible(TypeShape::Tuple(2), TypeShape::Tuple(3)));
    }
}

// ── Tree module (OmniML §6) ────────────────────────────────────────
//
// A tree structure with levels and nearest-common-ancestor lookup.
// Used by Region0 to organise regions into a hierarchy.
pub mod tree {
    /// Level of a node in the tree (depth from root).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Level(u32);

    impl Level {
        pub const ZERO: Level = Level(0);
        pub fn succ(self) -> Level {
            Level(self.0 + 1)
        }
    }

    /// A node in the tree.
    #[derive(Debug, Clone)]
    pub struct Node<T> {
        pub id: usize,
        pub level: Level,
        pub value: T,
        pub parent: Option<usize>,
    }

    impl<T> Node<T> {
        pub fn value(&self) -> &T {
            &self.value
        }
    }

    /// The root tree.
    #[derive(Debug)]
    pub struct Tree<T> {
        pub root: usize,
        nodes: Vec<Node<T>>,
    }

    impl<T> Tree<T> {
        pub fn new(value: T) -> (Self, usize) {
            let id = 0;
            let node = Node {
                id,
                level: Level::ZERO,
                value,
                parent: None,
            };
            (
                Tree {
                    root: id,
                    nodes: vec![node],
                },
                id,
            )
        }

        pub fn create_child(&mut self, parent_id: usize, value: T) -> usize {
            let id = self.nodes.len();
            let parent_level = self.nodes[parent_id].level;
            let node = Node {
                id,
                level: parent_level.succ(),
                value,
                parent: Some(parent_id),
            };
            self.nodes.push(node);
            id
        }

        pub fn get(&self, id: usize) -> &Node<T> {
            &self.nodes[id]
        }

        pub fn get_mut(&mut self, id: usize) -> &mut Node<T> {
            &mut self.nodes[id]
        }

        pub fn nearest_common_ancestor(&self, id1: usize, id2: usize) -> usize {
            if id1 == id2 {
                return id1;
            }
            let n1 = &self.nodes[id1];
            let n2 = &self.nodes[id2];
            if n1.level < n2.level {
                return self.nearest_common_ancestor(id1, n2.parent.expect("root has no parent"));
            }
            if n1.level > n2.level {
                return self.nearest_common_ancestor(n1.parent.expect("root has no parent"), id2);
            }
            // same level, both > root
            self.nearest_common_ancestor(
                n1.parent.expect("root has no parent"),
                n2.parent.expect("root has no parent"),
            )
        }
    }

    // ── With_dirty: dirty-mark propagation ──────────────────────────
    // NOTE: This module is feature-gated behind `omniml_pool` pending
    // full integration with the region tree lifecycle.  It is not yet
    // exercised by tests and should not be used in production code.

    #[cfg(feature = "omniml_pool")]
    pub mod with_dirty {
        use super::*;
        use std::collections::HashMap;

        #[derive(Debug, Clone)]
        pub struct Dirty {
            /// Children IDs that are dirty, stored by node ID (not by value)
            /// to avoid recursive type issues with Clone.
            pub children: HashMap<usize, usize>,
        }

        impl Dirty {
            pub fn new() -> Self {
                Dirty {
                    children: HashMap::new(),
                }
            }
            pub fn is_empty(&self) -> bool {
                self.children.is_empty()
            }
            pub fn add_child(&mut self, child_id: usize) {
                self.children.insert(child_id, child_id);
            }
            pub fn remove_child(&mut self, id: usize) {
                self.children.remove(&id);
            }
        }

        #[derive(Debug, Clone)]
        pub struct DirtyNodeDesc<T> {
            pub value: T,
            pub dirty: Option<Dirty>,
        }

        pub type DirtyNode<T> = super::Node<DirtyNodeDesc<T>>;

        pub fn create_desc<T>(value: T) -> DirtyNodeDesc<T> {
            DirtyNodeDesc { value, dirty: None }
        }

        /// A tree with dirty-mark propagation.
        #[derive(Debug)]
        pub struct DirtyTree<T> {
            pub tree: super::Tree<DirtyNodeDesc<T>>,
            pub dirty_roots: Dirty,
        }

        impl<T> DirtyTree<T> {
            pub fn new(value: T) -> Self {
                let (tree, _root_id) = super::Tree::new(create_desc(value));
                DirtyTree {
                    tree,
                    dirty_roots: Dirty::new(),
                }
            }

            pub fn create_child(&mut self, parent_id: usize, value: T) -> usize {
                self.tree.create_child(parent_id, create_desc(value))
            }

            pub fn get(&self, id: usize) -> &DirtyNode<T> {
                self.tree.get(id)
            }

            pub fn get_mut(&mut self, id: usize) -> &mut DirtyNode<T> {
                self.tree.get_mut(id)
            }

            pub fn root(&self) -> &DirtyNode<T> {
                self.tree.get(self.tree.root)
            }

            pub fn is_empty(&self) -> bool {
                self.dirty_roots.is_empty()
            }

            fn find_closest_dirty_ancestor(&self, node_id: usize) -> &Dirty {
                let node = self.tree.get(node_id);
                match node.parent {
                    None => &self.dirty_roots,
                    Some(parent_id) => {
                        let parent = self.tree.get(parent_id);
                        match parent.value.dirty {
                            Some(ref dirty) => dirty,
                            None => self.find_closest_dirty_ancestor(parent_id),
                        }
                    }
                }
            }

            fn find_closest_dirty_ancestor_mut(&mut self, node_id: usize) -> &mut Dirty {
                // We need to be careful with borrows here — use raw pointer for the
                // ancestor path since tree and dirty_roots are separate fields.
                let node = self.tree.get(node_id);
                let parent_id = match node.parent {
                    None => return &mut self.dirty_roots,
                    Some(id) => id,
                };
                // Check if the parent has a dirty marker without borrowing self.
                let has_dirty = self.tree.get(parent_id).value.dirty.is_some();
                if has_dirty {
                    // Return a mutable reference to the parent's dirty children.
                    // SAFETY: we are mutating the dirty field of the parent node,
                    // which is reachable through self.tree.nodes[parent_id].value.dirty.
                    let parent = &mut self.tree.nodes[parent_id];
                    parent.value.dirty.as_mut().unwrap()
                } else {
                    self.find_closest_dirty_ancestor_mut(parent_id)
                }
            }

            pub fn mark_dirty(&mut self, node_id: usize) {
                let node = self.tree.get(node_id);
                if node.value.dirty.is_some() {
                    return;
                }

                // Find the closest dirty ancestor (or dirty_roots).
                // We need to collect reparent candidates BEFORE getting the
                // mutable reference to the ancestor, to avoid borrow conflicts.
                let to_reparent: Vec<usize> = {
                    let anc_dirty = self.find_closest_dirty_ancestor(node_id);
                    anc_dirty
                        .children
                        .keys()
                        .filter(|&&child_id| self.is_ancestor(node_id, child_id))
                        .copied()
                        .collect()
                };

                // Remove reparented children from the ancestor.
                {
                    let anc_dirty = self.find_closest_dirty_ancestor_mut(node_id);
                    for &child_id in &to_reparent {
                        anc_dirty.remove_child(child_id);
                    }
                    // Add this node to the ancestor's dirty children.
                    anc_dirty.add_child(node_id);
                }

                // Set the node's dirty marker with the reparented children.
                let mut new_dirty = Dirty::new();
                for &child_id in &to_reparent {
                    new_dirty.add_child(child_id);
                }
                let node = &mut self.tree.nodes[node_id];
                node.value.dirty = Some(new_dirty);
            }

            pub fn is_ancestor(&self, ancestor_id: usize, desc_id: usize) -> bool {
                self.tree.nearest_common_ancestor(ancestor_id, desc_id) == ancestor_id
            }

            pub fn drain_dirty<F>(&mut self, node_id: usize, before: &F, after: &F, f: &F)
            where
                F: Fn(),
            {
                // We need to process nodes in a depth-first order.
                // Since we need mutable access to self, we collect the children first.
                let children: Vec<usize> = {
                    let node = self.tree.get(node_id);
                    match node.value.dirty {
                        Some(ref dirty) => dirty.children.keys().copied().collect(),
                        None => Vec::new(),
                    }
                };

                for &child_id in &children {
                    before();
                    self.drain_dirty(child_id, before, after, f);
                }

                before();
                f();
                after();

                // After processing, clear the dirty marker if no more children are dirty.
                let has_remaining = {
                    let node = self.tree.get(node_id);
                    node.value.dirty.as_ref().map_or(false, |d| !d.is_empty())
                };
                if !has_remaining {
                    let anc_dirty = self.find_closest_dirty_ancestor_mut(node_id);
                    anc_dirty.remove_child(node_id);
                    let node = &mut self.tree.nodes[node_id];
                    node.value.dirty = None;
                }
                after();
            }

            pub fn drain_dirty_roots<F>(&mut self, before: &F, after: &F, f: &F)
            where
                F: Fn(),
            {
                before();
                let roots: Vec<usize> = self.dirty_roots.children.keys().copied().collect();
                for &root_id in &roots {
                    self.drain_dirty(root_id, before, after, f);
                }
            }
        }
    }
}

// ── Status: Generalization State (OmniML §6) ────────────────────────
//
// Each type variable has a Status indicating whether it is fully
// generalized or still an Instance belonging to a region.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// The term is not fully generalized. `dirty` indicates that the
    /// term's owning region has been marked, meaning the term has updates
    /// that need propagating.
    Instance { dirty: bool },
    /// The term is fully generalized and belongs to no region.
    /// Once a term is generalized, it is removed from its pool.
    Generic,
}

impl Status {
    pub fn is_dirty(&self) -> bool {
        match self {
            Status::Instance { dirty } => *dirty,
            Status::Generic => panic!("Status::is_dirty called on Generic"),
        }
    }

    pub fn is_generic(&self) -> bool {
        matches!(self, Status::Generic)
    }

    /// Merge two statuses. Any update marks the status as dirty.
    pub fn merge(a: &Status, b: &Status) -> Status {
        match (a, b) {
            (Status::Generic, _) | (_, Status::Generic) => {
                panic!("Cannot merge Generic nodes")
            }
            (Status::Instance { .. }, Status::Instance { .. }) => Status::Instance { dirty: true },
        }
    }
}

// ── Region Pool Module (OmniML §6) ────────────────────────────────
//
// A Pool tracks all type variables that belong to a region.
// When a region is exited (generalized), the pool's types can be
// examined to determine which can become Generic vs. remain Instance.

pub mod pool {
    use super::tree;

    /// A region pool: tracks type variable IDs belonging to one region.
    pub struct Pool {
        /// Flexible type variable IDs in this region.
        pub types: Vec<usize>,
        /// Rigid (Skolem) variable IDs in this region.
        pub rigid_vars: Vec<usize>,
        /// Shape variable region node (optional — created lazily).
        pub shape_var_region: Option<usize>,
        /// Parent shape variable region node.
        pub parent_shape_var_region: usize,
        /// Callback invoked when a rigid variable escapes its scope.
        pub raise_scope_escape: Option<Box<dyn FnMut(usize) + Send>>,
    }

    impl std::fmt::Debug for Pool {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Pool")
                .field("types", &self.types)
                .field("rigid_vars", &self.rigid_vars)
                .field("shape_var_region", &self.shape_var_region)
                .field("parent_shape_var_region", &self.parent_shape_var_region)
                .field(
                    "raise_scope_escape",
                    &self.raise_scope_escape.as_ref().map(|_| "<fn>"),
                )
                .finish()
        }
    }

    impl Pool {
        pub fn new(parent_shape_var_region: usize) -> Self {
            Pool {
                types: Vec::new(),
                rigid_vars: Vec::new(),
                shape_var_region: None,
                parent_shape_var_region,
                raise_scope_escape: None,
            }
        }

        pub fn register_type(&mut self, id: usize) {
            self.types.push(id);
        }

        pub fn register_rigid_var(&mut self, id: usize) {
            self.rigid_vars.push(id);
        }

        pub fn is_dead(&self) -> bool {
            self.types.is_empty()
        }

        pub fn is_alive(&self) -> bool {
            !self.is_dead()
        }
    }

    /// Region0: wraps a Pool in a dirty tree node (OmniML §6).
    /// Each region is a node in the tree with a Pool as its value.
    #[cfg(feature = "omniml_pool")]
    pub type Region = tree::with_dirty::DirtyNode<Pool>;

    /// Get the pool from a region node.
    #[cfg(feature = "omniml_pool")]
    pub fn pool_of(region: &Region) -> &Pool {
        &region.value.value
    }

    /// Get the pool mutably from a region node.
    #[cfg(feature = "omniml_pool")]
    pub fn pool_of_mut(region: &mut Region) -> &mut Pool {
        &mut region.value.value
    }
}
