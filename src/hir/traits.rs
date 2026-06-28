use crate::ast::Span;
use crate::hir::symbol::SymbolTable;
use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use rustc_hash::FxHashMap as HashMap;

#[derive(Debug, Clone)]
pub struct ImplCandidate {
    pub trait_id: DefId,
    pub for_type: TypeId,
    pub methods: Vec<crate::ast::ImplMethod>,
    pub assoc_tys: Vec<(String, TypeId)>,
    pub span: Span,
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
        TraitEnv { impls: Vec::new(), inherent_methods: HashMap::default() }
    }

    pub fn add_impl(
        &mut self,
        candidate: ImplCandidate,
        symbols: &SymbolTable,
        ctx: &crate::hir::types::TypeContext,
        is_trusted: bool,
    ) -> Result<(), OrphanError> {
        let trait_crate = symbols
            .lookup_trait_by_def_id(candidate.trait_id)
            .map(|b| b.crate_id);
        let type_def_id = ctx.get_def_id_for_type(candidate.for_type);
        let type_crate = type_def_id
            .and_then(|did| symbols.lookup_type_by_def_id(did))
            .map(|b| b.crate_id);

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

        self.impls.push(candidate);
        Ok(())
    }

    pub fn lookup_impl(&self, trait_id: DefId, type_id: TypeId) -> Option<&ImplCandidate> {
        self.impls
            .iter()
            .find(|cand| cand.trait_id == trait_id && cand.for_type == type_id)
    }

    /// Find all impl candidates whose `for_type` matches the given type exactly.
    /// This is used for inherent method lookup (methods on a type without a trait).
    pub fn lookup_impls_for_type(&self, type_id: TypeId) -> Vec<&ImplCandidate> {
        self.impls
            .iter()
            .filter(|cand| cand.for_type == type_id)
            .collect()
    }

    /// Register resolved inherent methods for a type.
    pub fn add_inherent_methods(&mut self, for_type: DefId, methods: Vec<MethodInfo>) {
        self.inherent_methods.entry(for_type).or_default().extend(methods);
    }

    /// Look up inherent methods registered for a type.
    pub fn lookup_inherent_methods(&self, ty: TypeId, ctx: &TypeContext) -> &[MethodInfo] {
        match ctx.get(ty) {
            TypeData::Struct { def_id, .. } | TypeData::Enum { def_id, .. } => {
                self.inherent_methods.get(def_id).map(|v| v.as_slice()).unwrap_or(&[])
            }
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
