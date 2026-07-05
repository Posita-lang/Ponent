use crate::ast::Span;
use crate::hir::symbol::{SymbolTable, TypeKind};
use crate::hir::types::{DefId, Subst, TypeContext, TypeData, TypeId};
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static GENERIC_MATCH_VAR_ID: AtomicUsize = AtomicUsize::new(1_000_000);

#[derive(Debug, Clone)]
pub struct ImplCandidate {
    pub trait_id: DefId,
    pub for_type: TypeId,
    pub methods: Vec<crate::ast::ImplMethod>,
    /// Pre-resolved method signatures (resolved during resolution using impl's
    /// type param mapping, so generic params like `T` are properly handled).
    pub resolved_methods: Vec<MethodInfo>,
    pub assoc_tys: Vec<(String, TypeId)>,
    pub span: Span,
    /// Whether this impl's method calls can be auto-deref'd through.
    /// Set by `@auto_deref` attribute on the impl.
    pub has_auto_deref: bool,
    /// Context types from the impl's where clause, for Paterson condition
    /// termination checking. Each entry should be a type that appears in
    /// a constraint (e.g. `T` in `where T: Foo`). Must be strictly smaller
    /// in constructor depth than `for_type`.
    pub context: Vec<TypeId>,
}

/// Describes a single method with resolved type IDs, ready for method lookup.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: String,
    pub param_tys: Vec<TypeId>,
    pub ret_ty: TypeId,
    pub span: Span,
    /// Whether this method's `Deref` impl is marked `@auto_deref`.
    /// Without this flag, even if a type implements `Deref`, the compiler
    /// will NOT automatically dereference through it — the user must write `*x`.
    pub has_auto_deref: bool,
}

pub struct TraitEnv {
    impls: Vec<ImplCandidate>,
    /// Inherent (non-trait) methods indexed by the DefId of the type they're implemented on.
    inherent_methods: HashMap<DefId, Vec<MethodInfo>>,
}

impl TraitEnv {
    pub fn new() -> Self {
        TraitEnv {
            impls: Vec::new(),
            inherent_methods: HashMap::default(),
        }
    }

    pub fn add_impl(
        &mut self,
        candidate: ImplCandidate,
        symbols: &SymbolTable,
        ctx: &crate::hir::types::TypeContext,
        is_trusted: bool,
    ) -> Result<(), OrphanError> {
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
                if head_as_generic {
                    if let TypeData::GenericParam { index: hi, .. } = head_data {
                        if let TypeData::GenericParam { index: ci, .. } = ctx.get(ctx_ty) {
                            if hi == ci {
                                continue;
                            }
                        }
                    }
                }
                return Err(OrphanError {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
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
                return Err(OrphanError {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
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
                return Err(OrphanError {
                    trait_id: candidate.trait_id,
                    type_id: candidate.for_type,
                    span: candidate.span,
                });
            }
        }

        self.impls.push(candidate);
        Ok(())
    }
}

/// Collect all GenericParam indices appearing in a type.
fn collect_generic_params(ty: TypeId, ctx: &TypeContext) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    collect_generic_params_rec(ty, ctx, &mut set);
    set
}

fn collect_generic_params_rec(
    ty: TypeId,
    ctx: &TypeContext,
    set: &mut std::collections::HashSet<usize>,
) {
    match ctx.get(ty) {
        TypeData::GenericParam { index, .. } => {
            set.insert(*index);
        }
        TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
            for a in args {
                collect_generic_params_rec(*a, ctx, set);
            }
        }
        TypeData::Tuple { elems } => {
            for e in elems {
                collect_generic_params_rec(*e, ctx, set);
            }
        }
        TypeData::Array { elem, .. } => collect_generic_params_rec(*elem, ctx, set),
        TypeData::Slice { elem } => collect_generic_params_rec(*elem, ctx, set),
        TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
            collect_generic_params_rec(*ty, ctx, set)
        }
        TypeData::Ptr { pointee, .. } => collect_generic_params_rec(*pointee, ctx, set),
        TypeData::Fn { params, ret } => {
            for p in params {
                collect_generic_params_rec(*p, ctx, set);
            }
            collect_generic_params_rec(*ret, ctx, set);
        }
        TypeData::AssociatedType { self_ty, .. } => collect_generic_params_rec(*self_ty, ctx, set),
        TypeData::Exists { base, .. } => collect_generic_params_rec(*base, ctx, set),
        TypeData::Poly { body, .. } => collect_generic_params_rec(*body, ctx, set),
        _ => {}
    }
}

/// Collect only *bare* GenericParam indices — those appearing directly as a
/// top-level constructor argument (e.g. `T` in `(T,)` but not `T` in `Option<T>`).
fn collect_bare_generic_params(ty: TypeId, ctx: &TypeContext) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    match ctx.get(ty) {
        TypeData::GenericParam { index, .. } => {
            set.insert(*index);
        }
        TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
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

impl TraitEnv {
    pub fn lookup_impl(&self, trait_id: DefId, type_id: TypeId) -> Option<&ImplCandidate> {
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
        ctx: &mut TypeContext,
        symbols: &SymbolTable,
    ) -> Option<(&'b ImplCandidate, Subst)> {
        for cand in &self.impls {
            if cand.trait_id != trait_id {
                continue;
            }
            // Get the type binding for the candidate's for_type
            let def_id = match ctx.get(cand.for_type) {
                TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => *def_id,
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
            let mut fresh_args = Vec::with_capacity(binding.params.len());
            for i in 0..binding.params.len() {
                let id = GENERIC_MATCH_VAR_ID.fetch_add(1, Ordering::Relaxed);
                let fresh = ctx.alloc_infer_var(id);
                subst.insert(i, fresh);
                fresh_args.push(fresh);
            }
            // Substitute the candidate's for_type with the fresh args
            let substituted = match binding.kind {
                TypeKind::Struct => ctx.struct_ty(def_id, fresh_args),
                TypeKind::Enum => ctx.enum_ty(def_id, fresh_args),
                _ => continue,
            };
            // Try to unify target_ty with the substituted type
            if ctx.unify(target_ty, substituted).is_err() {
                continue; // unification failed, not a match
            }
            return Some((cand, subst));
        }
        None
    }

    /// Find all impl candidates whose `for_type` matches the given type exactly.
    /// This is used for inherent method lookup (methods on a type without a trait).
    pub fn lookup_impls_for_type(&self, type_id: TypeId) -> Vec<&ImplCandidate> {
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
        ctx: &mut TypeContext,
        symbols: &SymbolTable,
    ) -> Option<TypeId> {
        // Try exact match first
        if let Some(cand) = self.lookup_impl(trait_id, self_ty) {
            for (name, ty) in &cand.assoc_tys {
                if name == assoc_name {
                    return Some(ctx.resolve_binding(*ty));
                }
            }
            return None;
        }
        // Fall back to generic impl lookup
        let (cand, subst) = self.lookup_impl_generic(trait_id, self_ty, ctx, symbols)?;
        for (name, ty) in &cand.assoc_tys {
            if name == assoc_name {
                let resolved = ctx.subst(*ty, &subst);
                return Some(ctx.resolve_binding(resolved));
            }
        }
        None
    }

    /// Register resolved inherent methods for a type.
    pub fn add_inherent_methods(&mut self, for_type: DefId, methods: Vec<MethodInfo>) {
        self.inherent_methods
            .entry(for_type)
            .or_default()
            .extend(methods);
    }

    /// Look up inherent methods registered for a type.
    pub fn lookup_inherent_methods(&self, ty: TypeId, ctx: &TypeContext) -> &[MethodInfo] {
        match ctx.get(ty) {
            TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => self
                .inherent_methods
                .get(def_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            _ => &[],
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrphanError {
    pub trait_id: DefId,
    pub type_id: TypeId,
    pub span: Span,
}

impl Default for TraitEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for OrphanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "orphan rule violation: impl for trait {:?} on type {:?} is not allowed because neither the trait nor the type is local",
            self.trait_id, self.type_id
        )
    }
}
