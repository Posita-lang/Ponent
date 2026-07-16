use crate::ast::Span;
use crate::hir::types::{DefId, Subst, TypeId};
use crate::symbol::Symbol;

/// Source location and context for a trait obligation.
#[derive(Clone, Debug)]
pub struct ObligationCause {
    pub span: Span,
    pub code: ObligationCauseCode,
}

#[derive(Clone, Debug)]
pub enum ObligationCauseCode {
    MethodCall { method_name: Symbol },
    WhereClause { span: Span },
    ImplBound { impl_def_id: DefId },
    BuiltinDerive { trait_name: Symbol },
    PolyUnbox { span: Span },
    Misc,
}

/// A predicate that must be proven during trait resolution.
///
/// Posita has explicit lifetime parameters (see architecture §1.2)
/// but no `OutlivesPredicate` or region subtyping, so the predicate
/// language is simpler than Rust's.
#[derive(Clone, Debug)]
pub struct Obligation {
    pub cause: ObligationCause,
    pub predicate: Predicate,
    pub recursion_depth: usize,
}

/// What we need to prove.
/// Simpler than Rust's — no `OutlivesPredicate`, no `RegionOutlives`.
/// Lifetime parameters are treated as generic indices within the `Trait` variant's args.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// `T: Trait<Args>`
    Trait {
        trait_id: DefId,
        self_ty: TypeId,
        args: Vec<TypeId>,
    },
    /// `T: Trait<Args, Item = U>` — associated type projection equality
    ProjectionEq {
        trait_id: DefId,
        self_ty: TypeId,
        assoc_name: Symbol,
        value: TypeId,
    },
    /// `<T as Trait>::Assoc` — normalize this projection to a concrete type
    ProjectionNormalize {
        projection: ProjectionTy,
        target: TypeId,
    },
    /// `T: AutoTrait` (future: Send/Sync-like marker traits)
    AutoTrait {
        trait_id: DefId,
        self_ty: TypeId,
    },
    /// `T: Sized` — special builtin
    Sized {
        ty: TypeId,
    },
    /// `T: Copy` / `T: Clone` — special builtins
    CopyLike {
        kind: CopyKind,
        ty: TypeId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopyKind {
    Copy,
    Clone,
}

/// A projection type: `<SelfTy as Trait>::AssocName`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionTy {
    pub trait_id: DefId,
    pub self_ty: TypeId,
    pub args: Vec<TypeId>,
    pub assoc_name: Symbol,
}

/// The result of selecting a single obligation.
/// Analogous to rustc's `ImplSource`.
#[derive(Clone, Debug)]
pub enum ImplSource {
    /// User-defined impl: `impl Trait for Type { ... }`
    UserDefined {
        cand_idx: usize,
        subst: Subst,
        nested: Vec<Obligation>,
    },
    /// Caller-provided bound (where-clause)
    Param(Vec<Obligation>),
    /// Builtin trait (Sized, Copy, Clone, etc.)
    Builtin(BuiltinImplSource),
    /// Object type bound (dyn Trait)
    Object {
        object_trait_id: DefId,
        nested: Vec<Obligation>,
    },
    /// Auto-derived (future: Send-like)
    Auto {
        nested: Vec<Obligation>,
    },
    /// Poly/unbox resolved (Posita-specific).
    /// Unlike UserDefined, there is no real impl — the obligation is
    /// satisfied by unboxing a polymorphic value.
    Poly {
        subst: Subst,
        nested: Vec<Obligation>,
    },
    /// The obligation cannot be resolved yet because the self_ty is still
    /// an inference variable.  The solver should retry after the type is
    /// resolved.  Contains no sub-obligations.
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinImplSource {
    Sized,
    Copy,
    Clone,
    DiscriminantKind,
    FnPtr,
}

/// Error type for trait resolution.
#[derive(Clone, Debug)]
pub enum SolveError {
    NotFound {
        trait_id: DefId,
        self_ty: TypeId,
        span: Span,
    },
    Ambiguous {
        trait_id: DefId,
        self_ty: TypeId,
        span: Span,
        num_candidates: usize,
    },
    Overflow {
        trait_id: DefId,
        self_ty: TypeId,
        depth: usize,
    },
    CycleDetected {
        predicate: Predicate,
    },
    Mismatch {
        expected: TypeId,
        found: TypeId,
        span: Span,
    },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::NotFound { trait_id, self_ty, .. } => {
                write!(f, "trait impl not found for trait={:?} on type={:?}", trait_id, self_ty)
            }
            SolveError::Ambiguous { num_candidates, .. } => {
                write!(f, "ambiguous trait impl ({} candidates)", num_candidates)
            }
            SolveError::Overflow { depth, .. } => {
                write!(f, "trait resolution overflow at depth {}", depth)
            }
            SolveError::CycleDetected { .. } => {
                write!(f, "cycle detected during trait resolution")
            }
            SolveError::Mismatch { expected, found, .. } => {
                write!(f, "type mismatch: expected {:?}, found {:?}", expected, found)
            }
        }
    }
}