use crate::ast::Span;
use crate::hir::types::{DefId, Subst, TypeContext, TypeId};
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
    AutoTrait { trait_id: DefId, self_ty: TypeId },
    /// `T: Sized` — special builtin
    Sized { ty: TypeId },
    /// `T: Copy` / `T: Clone` — special builtins
    CopyLike { kind: CopyKind, ty: TypeId },
    // ── Eq/Sub/Match constraints (migrated from old solver) ──────────
    /// `Eq(a, b)` — type equality constraint (migrated from `Constraint::Eq`).
    /// Succeeds if `a` and `b` can be unified, defers if either is an
    /// unresolved inference variable.
    Eq { a: TypeId, b: TypeId },
    /// `Sub(sub, sup)` — subtype constraint (migrated from `Constraint::Sub`).
    /// Succeeds if `sub <: sup`, defers if either is an unresolved
    /// inference variable.
    Sub { sub: TypeId, sup: TypeId },
    /// `Match { scrutinee, branches_id }` — suspended match constraint
    /// (migrated from `Constraint::Match`).  Discharged when the scrutinee's
    /// shape is uniquely determined.
    Match {
        scrutinee: TypeId,
        branches_id: (usize, usize),
    },
    // ── Forall/Exists/Instance/Let constraints (migrated from old solver) ──
    /// `Forall { body }` — universally quantified constraint.
    /// Binds a fresh rigid (skolem) variable for the body.
    Forall {
        /// The body predicate to resolve under the quantifier.
        body: Box<Predicate>,
    },
    /// `Exists { body }` — existentially quantified constraint.
    /// Binds a fresh flexible variable for the body.
    Exists {
        /// The body predicate to resolve under the quantifier.
        body: Box<Predicate>,
    },
    /// `Instance { scheme_ty, instantiation_ty }` — instantiate a polymorphic
    /// scheme.  If `scheme_ty = ∀α₁...∀αₙ. τ_body`, creates fresh inference
    /// variables β₁...βₙ and constrains `Eq(instantiation_ty, τ_body[αᵢ:=βᵢ])`.
    Instance {
        /// The polymorphic scheme to instantiate.
        scheme_ty: TypeId,
        /// The type to instantiate at.
        instantiation_ty: TypeId,
    },
    /// `Let { def, body }` — let-polymorphism constraint.
    Let {
        /// The definition predicate.
        def: Box<Predicate>,
        /// The body predicate to resolve.
        body: Box<Predicate>,
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
    Auto { nested: Vec<Obligation> },
    /// Poly/unbox resolved (Posita-specific).
    /// Unlike UserDefined, there is no real impl — the obligation is
    /// satisfied by unboxing a polymorphic value.
    Poly {
        subst: Subst,
        nested: Vec<Obligation>,
    },
    /// The obligation cannot be resolved yet because the self_ty is still
    /// an inference variable.  `stalled_on` records which inference variables
    /// are blocking resolution, enabling selective re-evaluation when those
    /// variables are bound.  Contains no sub-obligations.
    Deferred {
        /// Inference variable TypeIds that are blocking resolution.
        stalled_on: Vec<TypeId>,
    },
}

impl ImplSource {
    /// Extract nested obligations from any ImplSource variant.
    /// Returns an empty vec for `Builtin` and `Deferred`.
    pub fn nested_obligations(&self) -> Vec<Obligation> {
        match self {
            ImplSource::UserDefined { nested, .. } => nested.clone(),
            ImplSource::Param(nested) => nested.clone(),
            ImplSource::Builtin(_) => vec![],
            ImplSource::Object { nested, .. } => nested.clone(),
            ImplSource::Auto { nested } => nested.clone(),
            ImplSource::Poly { nested, .. } => nested.clone(),
            ImplSource::Deferred { .. } => vec![],
        }
    }
}

/// How certain we are that a selected impl source is correct.
///
/// Analogous to rustc's `Certainty::Yes` vs `Certainty::Maybe(MaybeCause)`.
/// When a goal is `Maybe`, it provisionally succeeded but may fail once
/// inference variables are resolved or due to other provisional conditions.
/// The caller (e.g. `FulfillmentContext`) can use the `MaybeCause` to
/// decide whether to report the ambiguity to the user or silently retry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Certainty {
    /// Definitely resolved — the impl is sound and complete.
    Yes,
    /// Provisionally resolved — the goal may still fail once inference
    /// variables are resolved, or due to overflow / coinductive cycles.
    /// The `MaybeCause` describes why the result is provisional.
    Maybe(MaybeCause),
}

/// Why a goal is only provisionally resolved (`Certainty::Maybe`).
///
/// Analogous to rustc's `MaybeCause`.  Distinguishes between:
/// - Inference variables that are still unresolved (retry later)
/// - Recursion depth exceeded (overflow — bail out)
/// - Coinductive cycle detected (auto traits — treat as success)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaybeCause {
    /// Inference variables are not yet resolved.
    /// `stalled_on` records which variables are blocking resolution,
    /// enabling selective re-evaluation when those variables are bound.
    Unresolved { stalled_on: Vec<TypeId> },
    /// The recursion depth was exceeded during trait resolution.
    /// This is a hard ambiguity — the goal should be reported as an error.
    Overflow,
    /// A coinductive cycle was detected (e.g. `Send: Send`).
    /// Auto traits and `Sized` are coinductive, so cycles are expected
    /// and treated as provisional success.
    CoinductiveCycle,
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
        /// The obligation that exceeded the recursion limit.
        obligation: Box<Obligation>,
        /// The recursion depth at which overflow occurred.
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
            SolveError::NotFound {
                trait_id, self_ty, ..
            } => {
                write!(
                    f,
                    "trait impl not found for trait={:?} on type={:?}",
                    trait_id, self_ty
                )
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
            SolveError::Mismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "type mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
        }
    }
}

// ── Inherent methods on Predicate ──

impl Predicate {
    /// The self type of the goal (after resolving through bindings if needed).
    pub fn self_ty(&self) -> TypeId {
        match self {
            Predicate::Trait { self_ty, .. } => *self_ty,
            Predicate::AutoTrait { self_ty, .. } => *self_ty,
            Predicate::Sized { ty } => *ty,
            Predicate::CopyLike { ty, .. } => *ty,
            Predicate::ProjectionEq { self_ty, .. } => *self_ty,
            Predicate::ProjectionNormalize { projection, .. } => projection.self_ty,
            Predicate::Eq { a, .. } => *a,
            Predicate::Sub { sub, .. } => *sub,
            Predicate::Match { scrutinee, .. } => *scrutinee,
            Predicate::Forall { body } | Predicate::Exists { body } => body.self_ty(),
            Predicate::Instance { scheme_ty, .. } => *scheme_ty,
            Predicate::Let { def, .. } => def.self_ty(),
        }
    }

    /// The trait def id if this is a trait goal, or `None` for builtin-only
    /// goals like `Sized` / `Copy`.
    pub fn trait_def_id(&self) -> Option<DefId> {
        match self {
            Predicate::Trait { trait_id, .. }
            | Predicate::AutoTrait { trait_id, .. }
            | Predicate::ProjectionEq { trait_id, .. } => Some(*trait_id),
            Predicate::ProjectionNormalize { projection, .. } => Some(projection.trait_id),
            Predicate::Eq { .. } | Predicate::Sub { .. } | Predicate::Match { .. } => None,
            Predicate::Forall { .. }
            | Predicate::Exists { .. }
            | Predicate::Instance { .. }
            | Predicate::Let { .. } => None,
            _ => None,
        }
    }

    /// Resolve the goal through bindings, returning a `ResolvedObligation`.
    /// This is an inherent method to avoid `E0283` inference issues with
    /// `GoalKind<D>::resolve` (where `D` cannot be inferred from arguments).
    pub fn resolve(&self, ctx: &TypeContext) -> super::select::ResolvedObligation {
        match self {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let resolved_args: Vec<TypeId> =
                    args.iter().map(|a| ctx.resolve_binding(*a)).collect();
                let ambiguous = ctx.is_infer_var(resolved_self);
                super::select::ResolvedObligation {
                    trait_id: *trait_id,
                    self_ty: resolved_self,
                    args: resolved_args,
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let ambiguous = ctx.is_infer_var(resolved_self);
                super::select::ResolvedObligation {
                    trait_id: *trait_id,
                    self_ty: resolved_self,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Sized { ty } => {
                let resolved_ty = ctx.resolve_binding(*ty);
                let ambiguous = ctx.is_infer_var(resolved_ty);
                super::select::ResolvedObligation {
                    trait_id: DefId(usize::MAX),
                    self_ty: resolved_ty,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Eq { a, b } => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                let ambiguous = ctx.is_infer_var(ra) || ctx.is_infer_var(rb);
                super::select::ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: ra,
                    args: vec![rb],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Sub { sub, sup } => {
                let rsub = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                let ambiguous = ctx.is_infer_var(rsub) || ctx.is_infer_var(rsup);
                super::select::ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: rsub,
                    args: vec![rsup],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Match { scrutinee, .. } => {
                let resolved = ctx.resolve_binding(*scrutinee);
                let ambiguous = ctx.is_infer_var(resolved);
                super::select::ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: resolved,
                    args: vec![],
                    ambiguous,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Forall { body } | Predicate::Exists { body } => body.resolve(ctx),
            Predicate::Instance { scheme_ty, .. } => {
                let resolved = ctx.resolve_binding(*scheme_ty);
                super::select::ResolvedObligation {
                    trait_id: DefId(0),
                    self_ty: resolved,
                    args: vec![],
                    ambiguous: false,
                    parent_depth: 0,
                    span: crate::ast::Span::new(0, 0),
                }
            }
            Predicate::Let { def, .. } => def.resolve(ctx),
            _ => super::select::ResolvedObligation {
                trait_id: DefId(0),
                self_ty: ctx.error(),
                args: vec![],
                ambiguous: false,
                parent_depth: 0,
                span: crate::ast::Span::new(0, 0),
            },
        }
    }
}

impl SolveError {
    /// Extract the source span from this error, if available.
    pub fn span(&self) -> Option<crate::ast::Span> {
        match self {
            SolveError::NotFound { span, .. } => Some(*span),
            SolveError::Ambiguous { span, .. } => Some(*span),
            SolveError::Overflow { obligation, .. } => Some(obligation.cause.span),
            SolveError::CycleDetected { .. } => None,
            SolveError::Mismatch { span, .. } => Some(*span),
        }
    }
}
