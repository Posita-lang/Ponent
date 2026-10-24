pub mod solver;

use crate::ast::FileId;
use crate::ast::Span;
use crate::hir::symbol::{SymbolTable, TypeKind};
use crate::hir::traits::solver::coherence::check_overlap;
use crate::hir::types::{DefId, Subst, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;
use crate::sync::RwLock;
use rustc_hash::FxHashMap as HashMap;
use smallvec::{SmallVec, smallvec};
use std::sync::atomic::{AtomicUsize, Ordering};

// Base offset (1_000_000) for fresh generic-match variables: keeps them
// out of the main inference context's low ID space (regular inference
// variables are allocated from 0 upward), so the two ID families never
// collide in any table that indexes by both.  The exact value only needs
// to exceed the largest plausible count of ordinary inference variables
// in a single program — an order of magnitude above any real compile.
static GENERIC_MATCH_VAR_ID: AtomicUsize = AtomicUsize::new(1_000_000);

#[derive(Debug, Clone)]
pub struct ImplCandidate<'input> {
    pub trait_id: DefId,
    pub for_type: TypeId,
    pub methods: Vec<crate::ast::ImplMethod<'input>>,
    /// Pre-resolved method signatures (resolved during resolution using impl's
    /// type param mapping, so generic params like `T` are properly handled).
    pub resolved_methods: Vec<MethodInfo<'input>>,
    pub assoc_tys: Vec<(Symbol, TypeId)>,
    pub span: Span,
    /// Whether this impl's method calls can be auto-deref'd through.
    /// Set by `@auto_deref` attribute on the impl.
    pub has_auto_deref: bool,
    /// Context types from the impl's where clause, for Paterson condition
    /// termination checking. Each entry should be a type that appears in
    /// a constraint (e.g. `T` in `where T: Foo`). Must be strictly smaller
    /// in constructor depth than `for_type`.
    pub context: Vec<TypeId>,
    /// The actual trait bounds from the impl's where clause, stored as
    /// (self_ty, trait_id, args) triples.  Used by the trait solver to generate
    /// sub-obligations during impl matching (e.g., `impl<T: Clone> Foo for T`
    /// produces a sub-obligation `T: Clone` that must be verified).
    pub where_clause_bounds: Vec<(TypeId, DefId, Vec<TypeId>)>,
    /// Number of generic type parameters on this impl (e.g. `impl<T, U> Foo for T`
    /// has arity 2). Used by the trait solver to generate fresh inference variables
    /// during generic impl matching.
    pub arity: usize,
    /// Resolved generic arguments on the trait reference (e.g. `Int<32>` in
    /// `impl Add<Int<32>> for MyType`).  Stored as TypeIds so the trait solver
    /// can unify them with the obligation's trait args during matching.
    pub trait_args: Vec<TypeId>,
}

/// Describes a single method with resolved type IDs, ready for method lookup.
#[derive(Debug, Clone)]
pub struct MethodInfo<'input> {
    /// The method's OWN DefId — allocated once in the resolver (the
    /// "assoc item" identity, mirroring rustc's `AssocItem.def_id`), so a
    /// method is addressable independently of its impl across the whole
    /// pipeline (lookup, effect labels, diagnostics).
    pub def_id: DefId,
    pub name: Symbol,
    pub param_tys: SmallVec<[TypeId; 4]>,
    pub ret_ty: TypeId,
    pub span: Span,
    pub attributes: Vec<crate::ast::Attribute<'input>>,
    /// Whether this method's `Deref` impl is marked `@auto_deref`.
    /// Without this flag, even if a type implements `Deref`, the compiler
    /// will NOT automatically dereference through it — the user must write `*x`.
    pub has_auto_deref: bool,
}

pub struct TraitEnv<'input> {
    impls: Vec<ImplCandidate<'input>>,
    /// Inherent (non-trait) methods indexed by the DefId of the type they're implemented on.
    inherent_methods: HashMap<DefId, Vec<MethodInfo<'input>>>,
    /// Cache for `lookup_impl_generic`: (trait_id, target_ty) → Some(cand_idx, subst)
    /// on match, None on miss (negative cache).  Stores the candidate index alongside
    /// the substitution so that a cache hit returns the exact same candidate that
    /// originally produced the subst — avoiding mismatches when multiple impls
    /// share the same trait_id.
    ///
    /// Uses a lock for interior mutability so that `lookup_impl_generic` can
    /// remain `&self` (the cache is transparent to callers).
    ///
    /// Was a `RefCell`.  It cannot be: `RefCell` is `!Sync`, so `TraitEnv` —
    /// and therefore `Arc<TraitEnv>` — was not `Send`, which rules it out of
    /// anything the query provider hands to rayon (`QueryProvider: Sync`,
    /// src/hir/query/mod.rs:1511).  `RwLock` is the `Send + Sync` equivalent
    /// and suits a read-mostly memo cache.
    impl_generic_cache: RwLock<HashMap<(DefId, TypeId), Option<(usize, Subst)>>>,
}

// `RwLock<T>` is not `Clone` (unlike `RefCell<T>`), so `Clone` is
// implemented by hand.  This deep-copies the cache rather than sharing it,
// which is exactly what `#[derive(Clone)]` on the old `RefCell` did —
// keeping that behaviour avoids a subtle change where cloned environments
// would otherwise start seeing each other's negative-cache entries.
impl<'input> Clone for TraitEnv<'input> {
    fn clone(&self) -> Self {
        TraitEnv {
            impls: self.impls.clone(),
            inherent_methods: self.inherent_methods.clone(),
            impl_generic_cache: RwLock::new(self.impl_generic_cache.read().clone()),
        }
    }
}

impl<'input> TraitEnv<'input> {
    /// Clone for a short-lived worker that only LOOKS UP methods (the
    /// per-compute clones in the query layer).  Identical to `clone`
    /// except the lookup memo starts EMPTY: it is a read-mostly cache
    /// (see the field doc), so a fresh worker simply re-fills its own
    /// entries — no semantic difference (cloned entries were already
    /// isolated per clone), and the O(cache) deep copy under the read
    /// lock disappears from the per-function compute path.  The
    /// `impls`/`inherent_methods` data — what lookups actually consult
    /// — is copied in full, exactly as `clone` does.
    pub fn clone_for_worker(&self) -> Self {
        TraitEnv {
            impls: self.impls.clone(),
            inherent_methods: self.inherent_methods.clone(),
            impl_generic_cache: RwLock::new(HashMap::default()),
        }
    }
}

impl<'input> Default for TraitEnv<'input> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'input> TraitEnv<'input> {
    pub fn new() -> Self {
        TraitEnv {
            impls: Vec::new(),
            inherent_methods: HashMap::default(),
            impl_generic_cache: RwLock::new(HashMap::default()),
        }
    }

    /// Return all registered impl candidates.
    pub fn all_impls(&self) -> &[ImplCandidate<'input>] {
        &self.impls
    }

    pub fn add_impl(
        &mut self,
        candidate: ImplCandidate<'input>,
        symbols: &SymbolTable<'input>,
        ctx: &mut TypeContext<'input>,
        is_trusted: bool,
    ) -> Result<(), TraitError<'_>> {
        // Check termination conditions BEFORE the orphan rule, so that
        // generic impls get clear errors about Paterson/Coverage violations
        // rather than a misleading "orphan rule" error.
        // ... Paterson, Coverage, Orphan checks (unchanged) ...
        // ... (same as current code, just ctx changed to &mut) ...
        // Check termination conditions BEFORE the orphan rule, so that
        // generic impls get clear errors about Paterson/Coverage violations
        // rather than a misleading "orphan rule" error.

        // Paterson condition: each context type must be strictly smaller
        // (in constructor depth) than the head type `for_type`.
        // Exception: if both the context type and the head are the same
        // GenericParam (same index), the constraint is safe because it
        // can't create a growing chain — it's a self-referential bound.
        let head_depth = ctx.type_constructor_depth(candidate.for_type);
        let head_data = ctx.get(candidate.for_type);
        let head_as_generic = matches!(head_data, TypeData::GenericParam { .. });
        for &ctx_ty in &candidate.context {
            let d = ctx.type_constructor_depth(ctx_ty);
            if d >= head_depth {
                // Allow if both are the same GenericParam
                if head_as_generic
                    && let TypeData::GenericParam { index: hi, .. } = head_data
                    && let TypeData::GenericParam { index: ci, .. } = ctx.get(ctx_ty)
                    && hi == ci
                {
                    continue;
                }
                return Err(TraitError::Orphan {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
                    kind: OrphanKind::PatersonViolation,
                });
            }
        }

        // Coverage condition: every *bare* type variable in the head type
        // must appear in at least one context type.
        let bare_vars = collect_bare_generic_params(candidate.for_type, ctx);
        if !bare_vars.is_empty() {
            let mut context_vars = std::collections::HashSet::new();
            for &ctx_ty in &candidate.context {
                context_vars.extend(collect_generic_params(ctx_ty, ctx));
            }
            if !bare_vars.is_subset(&context_vars) {
                return Err(TraitError::Orphan {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
                    kind: OrphanKind::CoverageViolation,
                });
            }
        }

        // Orphan rule: if for_type is a concrete type (not a generic param),
        // check that at least one of the trait or the type is local.
        let type_def_id = ctx.get_def_id_for_type(candidate.for_type);
        if let Some(def_id) = type_def_id {
            let trait_crate = symbols
                .lookup_trait_by_def_id(candidate.trait_id)
                .map(|b| b.crate_id);
            let type_crate = symbols.lookup_type_by_def_id(def_id).map(|b| b.crate_id);

            let local = symbols.local_crate_id;

            let trait_local = trait_crate == Some(local);
            let type_local = type_crate == Some(local);

            if !is_trusted && !trait_local && !type_local {
                return Err(TraitError::Orphan {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
                    kind: OrphanKind::OrphanRule,
                });
            }
        }

        // ── Overlap check ──
        // Verify that the new impl does not overlap with any existing impl
        // of the same trait.  Uses two-phase detection:
        // 1. Structural comparison (TypeData::PartialEq) for concrete types
        // 2. try_unify for types with GenericParams
        //
        // NOTE: In parallel test runs, the global DefId allocator
        // (reset_def_id_allocator) is not thread-safe, causing race
        // conditions.  Use --test-threads=1 for reliable test results.
        if !is_trusted {
            ctx.begin_transaction();
            let conflict = check_overlap(&self.impls, &candidate, ctx);
            ctx.rollback_transaction();
            if let Some(conflict) = conflict {
                return Err(TraitError::Orphan {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
                    kind: OrphanKind::OverlapViolation {
                        existing_span: conflict.existing_span,
                    },
                });
            }
        }

        self.impls.push(candidate);
        // Invalidate the generic impl cache: a new impl may change the
        // result of future lookups, and stale entries could produce
        // incorrect candidate matches.
        self.impl_generic_cache.write().clear();
        Ok(())
    }
}

/// Collect all *free* GenericParam indices appearing in a type.
/// Tracks bound variables introduced by Forall/Exists/Mu/Nu/Poly
/// to avoid collecting variables that are bound by inner quantifiers.
fn collect_generic_params<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    let mut bound = std::collections::HashSet::new();
    collect_generic_params_rec(ty, ctx, &mut set, &mut bound);
    set
}

fn collect_generic_params_rec<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
    set: &mut std::collections::HashSet<usize>,
    bound: &mut std::collections::HashSet<usize>,
) {
    match ctx.get(ty) {
        TypeData::GenericParam { index, .. } => {
            if !bound.contains(index) {
                set.insert(*index);
            }
        }
        TypeData::Adt { args, .. } => {
            for a in args {
                collect_generic_params_rec(*a, ctx, set, bound);
            }
        }
        TypeData::Tuple { elems } => {
            for e in elems {
                collect_generic_params_rec(*e, ctx, set, bound);
            }
        }
        TypeData::Array { elem, .. } => collect_generic_params_rec(*elem, ctx, set, bound),
        TypeData::Slice { elem } => collect_generic_params_rec(*elem, ctx, set, bound),
        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
            collect_generic_params_rec(*ty, ctx, set, bound)
        }
        TypeData::Ptr { pointee, .. } => collect_generic_params_rec(*pointee, ctx, set, bound),
        TypeData::Fn { params, ret } => {
            for p in params {
                collect_generic_params_rec(*p, ctx, set, bound);
            }
            collect_generic_params_rec(*ret, ctx, set, bound);
        }
        TypeData::AssociatedType { self_ty, .. } => {
            collect_generic_params_rec(*self_ty, ctx, set, bound)
        }
        TypeData::Forall {
            param_index, body, ..
        } => {
            let inserted = bound.insert(*param_index);
            collect_generic_params_rec(*body, ctx, set, bound);
            if inserted {
                bound.remove(param_index);
            }
        }
        TypeData::Exists {
            param_index, base, ..
        } => {
            let inserted = bound.insert(*param_index);
            collect_generic_params_rec(*base, ctx, set, bound);
            if inserted {
                bound.remove(param_index);
            }
        }
        TypeData::Mu {
            param_index, body, ..
        } => {
            let inserted = bound.insert(*param_index);
            collect_generic_params_rec(*body, ctx, set, bound);
            if inserted {
                bound.remove(param_index);
            }
        }
        TypeData::Nu {
            param_index, body, ..
        } => {
            let inserted = bound.insert(*param_index);
            collect_generic_params_rec(*body, ctx, set, bound);
            if inserted {
                bound.remove(param_index);
            }
        }
        TypeData::Poly { quantifiers, body } => {
            let mut added = Vec::new();
            for (idx, _) in quantifiers {
                if bound.insert(*idx) {
                    added.push(*idx);
                }
            }
            collect_generic_params_rec(*body, ctx, set, bound);
            for idx in added {
                bound.remove(&idx);
            }
        }
        _ => {}
    }
}

/// Collect only *bare* GenericParam indices — those appearing directly as a
/// top-level constructor argument (e.g. `T` in `(T,)` but not `T` in `Option<T>`).
fn collect_bare_generic_params<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    match ctx.get(ty) {
        TypeData::GenericParam { index, .. } => {
            set.insert(*index);
        }
        TypeData::Adt { args, .. } => {
            for a in args {
                if let TypeData::GenericParam { index, .. } = ctx.get(*a) {
                    set.insert(*index);
                } else {
                    set.extend(collect_bare_generic_params(*a, ctx));
                }
            }
        }
        TypeData::Tuple { elems } => {
            for e in elems {
                if let TypeData::GenericParam { index, .. } = ctx.get(*e) {
                    set.insert(*index);
                } else {
                    set.extend(collect_bare_generic_params(*e, ctx));
                }
            }
        }
        TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
            if let TypeData::GenericParam { index, .. } = ctx.get(*elem) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*elem, ctx));
            }
        }
        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
            if let TypeData::GenericParam { index, .. } = ctx.get(*ty) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*ty, ctx));
            }
        }
        TypeData::Ptr { pointee, .. } => {
            if let TypeData::GenericParam { index, .. } = ctx.get(*pointee) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*pointee, ctx));
            }
        }
        TypeData::Fn { params, ret } => {
            for p in params {
                if let TypeData::GenericParam { index, .. } = ctx.get(*p) {
                    set.insert(*index);
                } else {
                    set.extend(collect_bare_generic_params(*p, ctx));
                }
            }
            if let TypeData::GenericParam { index, .. } = ctx.get(*ret) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*ret, ctx));
            }
        }
        TypeData::Exists { base, .. } => {
            if let TypeData::GenericParam { index, .. } = ctx.get(*base) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*base, ctx));
            }
        }
        TypeData::Poly { body, .. } => {
            if let TypeData::GenericParam { index, .. } = ctx.get(*body) {
                set.insert(*index);
            } else {
                set.extend(collect_bare_generic_params(*body, ctx));
            }
        }
        _ => {}
    }
    set
}

impl<'input> TraitEnv<'input> {
    pub fn lookup_impl(&self, trait_id: DefId, type_id: TypeId) -> Option<&ImplCandidate<'input>> {
        self.impls
            .iter()
            .find(|cand| cand.trait_id == trait_id && cand.for_type == type_id)
    }

    /// Try to find a generic impl candidate for the given trait and concrete type.
    /// For each candidate with matching trait_id, generates fresh infer vars for
    /// its generic parameters and attempts to unify the candidate's for_type with
    /// the target type. Returns the candidate if unification succeeds, along with
    /// the substitution mapping from generic param indices to the fresh infer vars.
    /// The caller must apply this substitution to the candidate's assoc_tys.
    pub fn lookup_impl_generic<'b>(
        &'b self,
        trait_id: DefId,
        target_ty: TypeId,
        ctx: &mut TypeContext<'input>,
        symbols: &SymbolTable<'input>,
    ) -> Option<(&'b ImplCandidate<'input>, Subst)> {
        let resolved = ctx.resolve_binding(target_ty);
        let cache_key = (trait_id, resolved);
        // Negative cache check.
        if self
            .impl_generic_cache
            .read()
            .get(&cache_key)
            .map_or(false, |v| v.is_none())
        {
            return None;
        }
        // Positive cache check: use the stored candidate index to retrieve
        // the exact same candidate that produced the cached substitution,
        // avoiding mismatches when multiple impls share the same trait_id.
        if let Some(Some((cached_idx, cached_subst))) =
            self.impl_generic_cache.read().get(&cache_key)
            && *cached_idx < self.impls.len()
        {
            return Some((&self.impls[*cached_idx], cached_subst.clone()));
        }
        for cand_idx in 0..self.impls.len() {
            if self.impls[cand_idx].trait_id != trait_id {
                continue;
            }
            // Get the type binding for the candidate's for_type
            let def_id = match ctx.get(self.impls[cand_idx].for_type) {
                TypeData::Adt { def_id, .. } => *def_id,
                _ => continue,
            };
            let binding = match symbols.lookup_type_by_def_id(def_id) {
                Some(b) => b,
                None => continue,
            };
            if binding.params.is_empty() {
                continue; // not generic, skip
            }
            // Generate fresh infer vars for each generic param
            let mut subst = Subst::new();
            let mut fresh_args: SmallVec<[TypeId; 4]> =
                SmallVec::with_capacity(binding.params.len());
            for i in 0..binding.params.len() {
                let id = GENERIC_MATCH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                let fresh = ctx.alloc_infer_var(id, 0);
                subst.insert(i, fresh);
                fresh_args.push(fresh);
            }
            // Substitute the candidate's for_type with the fresh args
            let substituted = match binding.kind {
                TypeKind::Struct => ctx.struct_ty(def_id, fresh_args),
                TypeKind::Enum => ctx.enum_ty(def_id, fresh_args),
                _ => continue,
            };
            // Use a transaction so that a failed unification attempt
            // does not leave stale bindings in the type context.
            ctx.begin_transaction();
            let result = ctx.unify(target_ty, substituted);
            if result.is_err() {
                ctx.rollback_transaction();
                continue; // unification failed, not a match
            }
            ctx.commit_transaction();
            // Cache the positive result with the candidate index.
            self.impl_generic_cache
                .write()
                .insert(cache_key, Some((cand_idx, subst.clone())));
            return Some((&self.impls[cand_idx], subst));
        }
        // Negative cache: no match found for this (trait_id, resolved) pair.
        self.impl_generic_cache.write().insert(cache_key, None);
        None
    }

    /// Find all impl candidates whose `for_type` matches the given type exactly.
    /// This is used for inherent method lookup (methods on a type without a trait).
    pub fn lookup_impls_for_type(&self, type_id: TypeId) -> Vec<&ImplCandidate<'input>> {
        self.impls
            .iter()
            .filter(|cand| cand.for_type == type_id)
            .collect()
    }

    /// Resolve an associated type projection: given a trait_id, self_ty, and
    /// associated type name, find the concrete type from the impl.
    ///
    /// This performs the equivalent of `<SelfTy as Trait>::AssocName`.
    /// It first tries exact match on `for_type`, then falls back to
    /// `lookup_impl_generic` for generic impls.
    pub fn resolve_assoc_type(
        &self,
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: &str,
        ctx: &mut TypeContext<'input>,
        symbols: &SymbolTable<'input>,
    ) -> Option<TypeId> {
        // Try exact match first
        if let Some(cand) = self.lookup_impl(trait_id, self_ty) {
            for (name, ty) in &cand.assoc_tys {
                if name.eq_str(assoc_name) {
                    return Some(ctx.resolve_binding(*ty));
                }
            }
            return None;
        }
        // Fall back to generic impl lookup
        let (cand, subst) = self.lookup_impl_generic(trait_id, self_ty, ctx, symbols)?;
        for (name, ty) in &cand.assoc_tys {
            if name.eq_str(assoc_name) {
                let resolved = ctx.subst(*ty, &subst);
                return Some(ctx.resolve_binding(resolved));
            }
        }
        None
    }

    /// Register resolved inherent methods for a type.
    pub fn add_inherent_methods(&mut self, for_type: DefId, methods: Vec<MethodInfo<'input>>) {
        self.inherent_methods
            .entry(for_type)
            .or_default()
            .extend(methods);
    }

    /// Look up inherent methods registered for a type.
    pub fn lookup_inherent_methods(
        &self,
        ty: TypeId,
        ctx: &TypeContext<'input>,
    ) -> &[MethodInfo<'input>] {
        match ctx.get(ty) {
            TypeData::Adt { def_id, .. } => self
                .inherent_methods
                .get(def_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            _ => &[],
        }
    }
}

#[derive(Debug, Clone)]
pub enum TraitError<'input> {
    /// Paterson condition / Coverage condition / orphan rule violation
    /// during impl registration.
    Orphan {
        trait_id: DefId,
        type_id: TypeId,
        span: Span,
        kind: OrphanKind,
    },
    /// No matching impl found for the given trait and type.
    NotFound {
        trait_id: DefId,
        type_id: TypeId,
        span: Span,
    },
    /// Multiple impl candidates matched (incoherent overlap).
    Ambiguous {
        candidates: Vec<ImplCandidate<'input>>,
        span: Span,
    },
}

/// Specific violation kind for `TraitError::Orphan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrphanKind {
    /// Paterson condition: a context type is not strictly smaller than the head.
    PatersonViolation,
    /// Coverage condition: a bare type variable in the head is not covered.
    CoverageViolation,
    /// Neither the trait nor the type is local to the current crate.
    OrphanRule,
    /// Overlap with an existing impl: the new impl's for_type and the existing
    /// impl's for_type can be unified, meaning there exists a concrete type
    /// that would satisfy both impls.
    OverlapViolation { existing_span: Span },
}

impl<'input> std::fmt::Display for TraitError<'input> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraitError::Orphan {
                kind,
                trait_id,
                type_id,
                ..
            } => {
                let msg = match kind {
                    OrphanKind::PatersonViolation => {
                        "impl violates Paterson condition: a context type is not strictly smaller than the head type"
                    }
                    OrphanKind::CoverageViolation => {
                        "impl violates Coverage condition: a bare type variable in the head is not covered by context types"
                    }
                    OrphanKind::OrphanRule => {
                        "orphan rule violation: impl for trait on type is not allowed because neither the trait nor the type is local"
                    }
                    OrphanKind::OverlapViolation { .. } => {
                        "impl overlaps with an existing impl: the head types unify, meaning there exists a concrete type that would satisfy both"
                    }
                };
                write!(f, "{} (trait={:?}, type={:?})", msg, trait_id, type_id)
            }
            TraitError::NotFound {
                trait_id, type_id, ..
            } => {
                write!(
                    f,
                    "trait impl not found for trait={:?} on type={:?}",
                    trait_id, type_id
                )
            }
            TraitError::Ambiguous { candidates, .. } => {
                write!(
                    f,
                    "multiple matching impl candidates found ({} candidates)",
                    candidates.len()
                )
            }
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::hir::types::{CrateId, DefId};

    #[test]
    fn test_trait_env_new_is_empty() {
        let env = TraitEnv::new();
        assert!(env.all_impls().is_empty());
    }

    #[test]
    fn test_add_impl_simple() {
        let mut ctx = TypeContext::new();
        let local = CrateId(DefId(0));
        let mut symbols = SymbolTable::new(local);
        let mut env = TraitEnv::new();

        let trait_id = symbols.allocate_def_id();
        let type_id = symbols.allocate_def_id();
        let for_ty = ctx.struct_ty(type_id, smallvec![]);

        let candidate = ImplCandidate {
            trait_id,
            for_type: for_ty,
            methods: vec![],
            resolved_methods: vec![],
            assoc_tys: vec![],
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
            has_auto_deref: false,
            context: vec![],
            arity: 0,
            trait_args: vec![],
            where_clause_bounds: vec![],
        };

        // Must register the type in the symbol table so the orphan rule check can find it
        let binding = crate::hir::symbol::TypeBinding {
            def_id: type_id,
            params: vec![],
            kind: crate::hir::symbol::TypeKind::Struct,
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
            alias_ast: None,
            attributes: vec![],
            fields: vec![],
            variants: vec![],
            invariant: None,
            default_value: None,
            no_default: false,
            crate_id: local,
            missing_match: None,
            exhaustive: false,
            c_layout: false,
            transparent: false,
            expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
        };
        symbols
            .insert_type("TestType".into(), binding, crate::ast::Span::new(FileId::DUMMY, 0, 0))
            .ok();

        // Register the def_id → type_id mapping so the orphan rule check
        // can find the type and actually execute, not silently skip.
        ctx.register_def_id(type_id, for_ty);

        let result = env.add_impl(candidate, &symbols, &mut ctx, false);
        assert!(
            result.is_ok(),
            "add_impl should succeed: {:?}",
            result.err()
        );
        drop(result);
        assert_eq!(env.all_impls().len(), 1);
    }

    #[test]
    fn test_add_inherent_methods() {
        let mut ctx = TypeContext::new();
        let local = CrateId(DefId(0));
        let mut env = TraitEnv::new();

        let def_id = DefId(42);
        let method = MethodInfo {
            def_id: DefId(43),
            name: "foo".into(),
            param_tys: smallvec![],
            ret_ty: ctx.int(32, true),
            span: crate::ast::Span::new(FileId::DUMMY, 0, 0),
            attributes: vec![],
            has_auto_deref: false,
        };
        env.add_inherent_methods(def_id, vec![method]);

        let ty = ctx.struct_ty(def_id, smallvec![]);
        let methods = env.lookup_inherent_methods(ty, &ctx);
        assert_eq!(methods.len(), 1);
        assert!(methods[0].name.eq_str("foo"));
    }
}
