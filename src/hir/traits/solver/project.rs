use crate::hir::query::{DefaultCache as QueryCache, QueryCacheType};
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::traits::solver::obligation::ProjectionTy;
use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;
use std::cell::RefCell;

/// Maximum number of entries in the projection cache.
/// When exceeded, the entire cache is cleared.  This is a simple safeguard
/// against unbounded memory growth; the `clear()` penalty is minimal
/// because cache misses only trigger a `TraitEnv` lookup, not a full
/// unification.
const MAX_CACHE_SIZE: usize = 1024;

/// Cache for normalized projection types.
///
/// Prevents infinite recursion and redundant work during
/// `<T as Trait>::Assoc` resolution.  Uses `QueryCache` for O(1) lookup
/// with capacity-limited eviction.
///
/// Migrated from `RefCell<HashMap>` to `QueryCache` — the query system's
/// generic cache provides the same functionality with built-in eviction
/// and a consistent API for all compiler caches.
pub struct ProjectionCache {
    cache: QueryCache<ProjectionKey, ProjectionCacheEntry>,
    /// For marking ambiguous entries — we need a separate tracking
    /// mechanism because `QueryCache` doesn't support the "ambiguous"
    /// state natively.  This is a temporary wrapper until the query
    /// system is extended to support multi-state entries.
    ambiguous: RefCell<rustc_hash::FxHashSet<ProjectionKey>>,
}

/// Key for the projection cache: `(trait_id, self_ty, assoc_name)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProjectionKey(DefId, TypeId, Symbol);

#[derive(Clone, Debug)]
enum ProjectionCacheEntry {
    /// Successfully normalized to this type.
    Normalized(TypeId),
    /// Normalization failed (no impl found, etc.).
    Error,
}

impl ProjectionCache {
    pub fn new() -> Self {
        ProjectionCache {
            cache: QueryCache::with_capacity(MAX_CACHE_SIZE),
            ambiguous: RefCell::new(rustc_hash::FxHashSet::default()),
        }
    }

    /// Try to get a cached normalized type.
    pub fn get(
        &self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: &Symbol,
        ctx: &TypeContext,
    ) -> Option<TypeId> {
        let resolved = ctx.resolve_binding(self_ty);
        let key = ProjectionKey(trait_id, resolved, *assoc_name);

        // Check ambiguous set first.
        if self.ambiguous.borrow().contains(&key) {
            return None;
        }

        match self.cache.lookup(&key) {
            Some(ProjectionCacheEntry::Normalized(ty)) => Some(ty),
            Some(ProjectionCacheEntry::Error) => None,
            None => None,
        }
    }

    /// Insert a normalized type into the cache.
    /// If the cache exceeds MAX_CACHE_SIZE, it is fully cleared first.
    pub fn insert(
        &self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: Symbol,
        ty: TypeId,
        ctx: &TypeContext,
    ) {
        let resolved = ctx.resolve_binding(self_ty);
        let key = ProjectionKey(trait_id, resolved, assoc_name);
        self.cache.insert(key, ProjectionCacheEntry::Normalized(ty));
    }

    /// Mark a projection as ambiguous (should not be cached permanently).
    pub fn mark_ambiguous(
        &self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: Symbol,
        ctx: &TypeContext,
    ) {
        let resolved = ctx.resolve_binding(self_ty);
        let key = ProjectionKey(trait_id, resolved, assoc_name);
        self.ambiguous.borrow_mut().insert(key);
    }

    /// Clear the cache. Called when new impls are added.
    pub fn clear(&self) {
        self.cache.clear();
        self.ambiguous.borrow_mut().clear();
    }
}

/// Resolve an associated type projection: `<SelfTy as Trait>::AssocName`.
///
/// This performs the equivalent of `<SelfTy as Trait>::AssocName`.
/// It first tries exact match on `for_type`, then falls back to
/// `lookup_impl_generic` for generic impls.
///
/// This is the solver's projection resolution function. It delegates
/// to TraitEnv for impl lookup and does NOT modify the inference context
/// (no unification, no transaction). The caller is responsible for
/// committing or rolling back.
pub fn resolve_projection(
    trait_env: &TraitEnv,
    trait_id: DefId,
    self_ty: TypeId,
    assoc_name: &Symbol,
    ctx: &mut TypeContext,
    symbols: &SymbolTable,
) -> Option<TypeId> {
    // Try exact match first
    if let Some(cand) = trait_env.lookup_impl(trait_id, self_ty) {
        for (name, ty) in &cand.assoc_tys {
            if name.eq_str(&assoc_name.as_str()) {
                return Some(ctx.resolve_binding(*ty));
            }
        }
        return None;
    }

    // Fall back to generic impl lookup
    let (cand, subst) = trait_env.lookup_impl_generic(trait_id, self_ty, ctx, symbols)?;
    for (name, ty) in &cand.assoc_tys {
        if name.eq_str(&assoc_name.as_str()) {
            let resolved = ctx.subst(*ty, &subst);
            return Some(ctx.resolve_binding(resolved));
        }
    }
    None
}

/// Normalize a projection type to its concrete value.
///
/// This is the core normalization function. It resolves the projection
/// and then recursively normalizes the result. If the projection's
/// self_ty is an inference variable, normalization is deferred
/// (returns `None`).
pub fn normalize_projection(
    proj: &ProjectionTy,
    trait_env: &TraitEnv,
    ctx: &mut TypeContext,
    cache: &ProjectionCache,
    symbols: &SymbolTable,
) -> Option<TypeId> {
    // Check cache first
    if let Some(cached) = cache.get(proj.trait_id, proj.self_ty, &proj.assoc_name, ctx) {
        return Some(cached);
    }

    // If self_ty is an inference variable, we cannot resolve yet
    if ctx.is_infer_var(proj.self_ty) {
        cache.mark_ambiguous(proj.trait_id, proj.self_ty, proj.assoc_name, ctx);
        return None;
    }

    // Resolve the projection
    let resolved = resolve_projection(
        trait_env,
        proj.trait_id,
        proj.self_ty,
        &proj.assoc_name,
        ctx,
        symbols,
    )?;

    // Recursively normalize the result (in case it's itself a projection)
    let result = match ctx.get(resolved) {
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let inner_proj = ProjectionTy {
                trait_id: *trait_id,
                self_ty: *self_ty,
                args: proj.args.clone(),
                assoc_name: *name,
            };
            normalize_projection(&inner_proj, trait_env, ctx, cache, symbols).unwrap_or(resolved)
        }
        _ => resolved,
    };

    // Cache the result
    cache.insert(proj.trait_id, proj.self_ty, proj.assoc_name, result, ctx);
    Some(result)
}
