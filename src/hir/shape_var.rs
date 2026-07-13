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
        TypeData::Adt { args, .. } => TypeShape::Constructor(args.len()),
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

#[cfg(test)]
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
        pub fn succ(self) -> Level { Level(self.0 + 1) }
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
        pub fn value(&self) -> &T { &self.value }
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
            let node = Node { id, level: Level::ZERO, value, parent: None };
            (Tree { root: id, nodes: vec![node] }, id)
        }

        pub fn create_child(&mut self, parent_id: usize, value: T) -> usize {
            let id = self.nodes.len();
            let parent_level = self.nodes[parent_id].level;
            let node = Node { id, level: parent_level.succ(), value, parent: Some(parent_id) };
            self.nodes.push(node);
            id
        }

        pub fn get(&self, id: usize) -> &Node<T> { &self.nodes[id] }

        pub fn get_mut(&mut self, id: usize) -> &mut Node<T> { &mut self.nodes[id] }

        pub fn nearest_common_ancestor(&self, id1: usize, id2: usize) -> usize {
            if id1 == id2 { return id1; }
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
            pub fn new() -> Self { Dirty { children: HashMap::new() } }
            pub fn is_empty(&self) -> bool { self.children.is_empty() }
            pub fn add_child(&mut self, child_id: usize) { self.children.insert(child_id, child_id); }
            pub fn remove_child(&mut self, id: usize) { self.children.remove(&id); }
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
                DirtyTree { tree, dirty_roots: Dirty::new() }
            }

            pub fn create_child(&mut self, parent_id: usize, value: T) -> usize {
                self.tree.create_child(parent_id, create_desc(value))
            }

            pub fn get(&self, id: usize) -> &DirtyNode<T> { self.tree.get(id) }

            pub fn get_mut(&mut self, id: usize) -> &mut DirtyNode<T> { self.tree.get_mut(id) }

            pub fn root(&self) -> &DirtyNode<T> { self.tree.get(self.tree.root) }

            pub fn is_empty(&self) -> bool { self.dirty_roots.is_empty() }

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
                if node.value.dirty.is_some() { return; }

                // Find the closest dirty ancestor (or dirty_roots).
                // We need to collect reparent candidates BEFORE getting the
                // mutable reference to the ancestor, to avoid borrow conflicts.
                let to_reparent: Vec<usize> = {
                    let anc_dirty = self.find_closest_dirty_ancestor(node_id);
                    anc_dirty.children
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
            where F: Fn()
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
            where F: Fn()
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
            (Status::Instance { .. }, Status::Instance { .. }) => {
                Status::Instance { dirty: true }
            }
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
                .field("raise_scope_escape", &self.raise_scope_escape.as_ref().map(|_| "<fn>"))
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
