use rustc_hash::FxHashMap as HashMap;

/// Reference-counted guard set for tracking when inference variables
/// are captured by suspended constraints (OmniML §6).
///
/// Each type variable maintains a set of guards:
/// - **direct_guards**: reference count of constraints directly suspended on this var.
/// - **transitive_guards**: per-region counts for transitive guards from PG variables
///   that have instances in this variable's region.
///
/// A variable with any non-zero guard remains PartiallyGeneralizable (PG);
/// when all guards reach zero it may become Generalized (G).
#[derive(Debug, Clone)]
pub struct GuardSet {
    /// Number of direct guards (suspended match/impl constraints).
    pub direct_guards: usize,
    /// Per-region transitive guard counts (region_id → count).
    /// Transitive guards arise when a PG variable has an instance in
    /// another region: the instance is transitively guarded by the
    /// PG variable's region.
    pub transitive_guards: HashMap<usize, usize>,
}

impl GuardSet {
    pub fn empty() -> Self {
        GuardSet {
            direct_guards: 0,
            transitive_guards: HashMap::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.direct_guards == 0 && self.transitive_guards.is_empty()
    }

    /// Add a direct guard (increment reference count).
    pub fn add_guard(&mut self) {
        self.direct_guards = self.direct_guards.wrapping_add(1);
    }

    /// Remove a direct guard (decrement reference count).
    pub fn remove_guard(&mut self) {
        if self.direct_guards > 0 {
            self.direct_guards -= 1;
        }
    }

    /// Add a transitive guard for a region (increment per-region count).
    pub fn add_transitive_guard(&mut self, region_id: usize) {
        *self.transitive_guards.entry(region_id).or_insert(0) += 1;
    }

    /// Remove a transitive guard for a region (decrement per-region count).
    /// Panics in debug if the guard doesn't exist.
    pub fn remove_transitive_guard(&mut self, region_id: usize) {
        if let Some(count) = self.transitive_guards.get_mut(&region_id) {
            debug_assert!(*count > 0, "remove_transitive_guard: count already zero");
            *count -= 1;
            if *count == 0 {
                self.transitive_guards.remove(&region_id);
            }
        }
    }

    /// Clear ALL transitive guards for a region (used during generalization).
    pub fn clear_transitive_guard(&mut self, region_id: usize) {
        self.transitive_guards.remove(&region_id);
    }

    /// Check if this variable is transitively guarded by the given region.
    pub fn is_transitively_guarded(&self, region_id: usize) -> bool {
        self.transitive_guards
            .get(&region_id)
            .map_or(false, |&c| c > 0)
    }

    /// Merge two guard sets (union of direct guards and sum of transitive).
    pub fn union(&self, other: &Self) -> Self {
        let mut transitive = self.transitive_guards.clone();
        for (&region, &count) in &other.transitive_guards {
            *transitive.entry(region).or_insert(0) += count;
        }
        GuardSet {
            direct_guards: self.direct_guards + other.direct_guards,
            transitive_guards: transitive,
        }
    }

    /// Clear all guards (reset to empty).
    pub fn clear(&mut self) {
        self.direct_guards = 0;
        self.transitive_guards.clear();
    }
}
