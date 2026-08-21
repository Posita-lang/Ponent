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
    /// `NormalizesTo(ProjectionTy, TypeId)` — dedicated goal kind for
    /// associated type projection normalization with fixpoint iteration.
    /// See `rustc_next_trait_solver::solve::normalizes_to`.
    NormalizesTo {
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
            ImplSource::Builtin(src) => {
                // The builtin-impl source is consumed here (Sized/Copy/
                // Clone/DiscriminantKind/FnPtr): a builtin trait has no
                // nested obligations, but the source kind is sanity-checked
                // in debug builds (only known builtin sources are legal).
                debug_assert!(
                    matches!(
                        *src,
                        BuiltinImplSource::Sized
                            | BuiltinImplSource::Copy
                            | BuiltinImplSource::Clone
                            | BuiltinImplSource::DiscriminantKind
                            | BuiltinImplSource::FnPtr
                    ),
                    "unknown builtin impl source"
                );
                vec![]
            }
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

/// Whether evaluating a goal changed the inference state.
///
/// Used by the fixpoint iteration loop to detect convergence:
/// when a cycle head is re-evaluated and the result hasn't changed,
/// we've reached a fixpoint and can stop.
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
pub enum HasChanged {
    Yes,
    No,
}

/// Why a goal needs to be re-evaluated after a fixpoint iteration.
///
/// A subset of rustc's `RerunReason` — we don't inline the full
/// canonicalization/opaque-type infrastructure yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RerunReason {
    /// New inference variables were introduced that may resolve ambiguity.
    NewInferenceVars,
    /// A nested goal was previously ambiguous and may now be resolved.
    NestedGoalResolved,
    /// A cycle head's provisional result has changed.
    CycleHeadChanged,
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
        trait_id: Option<DefId>,
        self_ty: TypeId,
        span: Span,
    },
    Ambiguous {
        trait_id: Option<DefId>,
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
        /// The underlying unification error, when available — carried for
        /// more precise diagnostics.
        note: String,
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
                    "trait impl not found for trait={} on type={}",
                    trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)).0,
                    self_ty.raw()
                )
            }
            SolveError::Ambiguous {
                num_candidates,
                trait_id,
                self_ty,
                ..
            } => {
                if *num_candidates == 0 {
                    write!(
                        f,
                        "no trait implementation found for trait={} on type={}",
                        trait_id.unwrap_or(crate::hir::types::DefId(usize::MAX)).0,
                        self_ty.raw()
                    )
                } else {
                    write!(f, "ambiguous trait impl ({} candidates)", num_candidates)
                }
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
            Predicate::NormalizesTo { projection, .. } => projection.self_ty,
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
    pub fn resolve<'input>(&self, ctx: &TypeContext<'input>) -> super::select::ResolvedObligation {
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
                Self::make_obligation(
                    Some(*trait_id),
                    resolved_self,
                    resolved_args,
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                let ambiguous = ctx.is_infer_var(resolved_self);
                Self::make_obligation(
                    Some(*trait_id),
                    resolved_self,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::Sized { ty } => {
                let resolved_ty = ctx.resolve_binding(*ty);
                let ambiguous = ctx.is_infer_var(resolved_ty);
                Self::make_obligation(
                    Some(DefId(usize::MAX)),
                    resolved_ty,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::Eq { a, b } => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                let ambiguous = ctx.is_infer_var(ra) || ctx.is_infer_var(rb);
                Self::make_obligation(None, ra, vec![rb], ambiguous, crate::ast::Span::new(0, 0))
            }
            Predicate::Sub { sub, sup } => {
                let rsub = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                let ambiguous = ctx.is_infer_var(rsub) || ctx.is_infer_var(rsup);
                Self::make_obligation(
                    None,
                    rsub,
                    vec![rsup],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::Match { scrutinee, .. } => {
                let resolved = ctx.resolve_binding(*scrutinee);
                let ambiguous = ctx.is_infer_var(resolved);
                Self::make_obligation(
                    None,
                    resolved,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::Forall { body } | Predicate::Exists { body } => body.resolve(ctx),
            Predicate::Instance { scheme_ty, .. } => {
                let resolved = ctx.resolve_binding(*scheme_ty);
                Self::make_obligation(None, resolved, vec![], false, crate::ast::Span::new(0, 0))
            }
            // Quantifier/let wrappers recurse into their inner predicate
            // (unified through `inner_predicate`).
            Predicate::Forall { .. } | Predicate::Exists { .. } | Predicate::Let { .. } => {
                match self.inner_predicate() {
                    Some(inner) => inner.resolve(ctx),
                    None => Self::make_obligation(
                        None,
                        ctx.error(),
                        vec![],
                        false,
                        crate::ast::Span::new(0, 0),
                    ),
                }
            }
            // Explicit arms for the remaining variants — a newly added
            // Predicate variant is a compile error here instead of being
            // silently swallowed by `_`.
            Predicate::CopyLike { ty, .. } => {
                let resolved = ctx.resolve_binding(*ty);
                let ambiguous = ctx.is_infer_var(resolved);
                Self::make_obligation(
                    None,
                    resolved,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::ProjectionEq { self_ty, .. } => {
                let resolved = ctx.resolve_binding(*self_ty);
                let ambiguous = ctx.is_infer_var(resolved);
                Self::make_obligation(
                    None,
                    resolved,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::ProjectionNormalize { projection, .. } => {
                let resolved = ctx.resolve_binding(projection.self_ty);
                let ambiguous = ctx.is_infer_var(resolved);
                Self::make_obligation(
                    None,
                    resolved,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
            Predicate::NormalizesTo { projection, .. } => {
                let resolved = ctx.resolve_binding(projection.self_ty);
                let ambiguous = ctx.is_infer_var(resolved);
                Self::make_obligation(
                    None,
                    resolved,
                    vec![],
                    ambiguous,
                    crate::ast::Span::new(0, 0),
                )
            }
        }
    }

    /// Shared `ResolvedObligation` construction — every arm only fills in
    /// the fields that differ.  `trait_id` is `Option<DefId>`: `None` for
    /// non-trait goals, `Some` for trait goals / the built-in `Sized`.
    fn make_obligation(
        trait_id: Option<DefId>,
        self_ty: TypeId,
        args: Vec<TypeId>,
        ambiguous: bool,
        span: crate::ast::Span,
    ) -> super::select::ResolvedObligation {
        super::select::ResolvedObligation {
            trait_id,
            self_ty,
            args,
            ambiguous,
            parent_depth: 0,
            span,
        }
    }

    /// The inner predicate for quantifier/let wrappers (`Forall` /
    /// `Exists` / `Let`) — unified recursion for `resolve`.
    fn inner_predicate(&self) -> Option<&Predicate> {
        match self {
            Predicate::Forall { body } | Predicate::Exists { body } => Some(body),
            Predicate::Let { def, .. } => Some(def),
            _ => None,
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

    /// The self type of the goal this error is about, when the error kind
    /// carries one.  `None` for error kinds without a self type (e.g.
    /// `Mismatch`).  Used by recovery-artifact detection.
    pub fn self_ty(&self) -> Option<TypeId> {
        match self {
            SolveError::NotFound { self_ty, .. } | SolveError::Ambiguous { self_ty, .. } => {
                Some(*self_ty)
            }
            SolveError::Overflow { obligation, .. } => Some(obligation.predicate.self_ty()),
            SolveError::CycleDetected { predicate } => Some(predicate.self_ty()),
            SolveError::Mismatch { .. } => None,
        }
    }

    /// Whether this error is a recovery artifact: an obligation whose self
    /// type has since RESOLVED to the error sentinel.  The expression it
    /// came from already failed to type-check elsewhere (e.g. a silently
    /// recovered deref); enforcing traits on the recovery type surfaces
    /// cascading `... on type Error` errors, and the recovery path owns the
    /// diagnostics.  `contains_error` (not the shallow `is_error`) also
    /// catches composite recoveries like `Vec<Error>`, and
    /// `CycleDetected`/`Overflow` goals at the sentinel are recovery
    /// artifacts too (e.g. a cyclic generic impl instantiated at the
    /// recovery type).
    pub fn is_recovery_artifact<'input>(&self, ctx: &TypeContext<'input>) -> bool {
        self.self_ty()
            .is_some_and(|ty| ctx.contains_error(ctx.resolve_binding(ty)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins `SolveError::self_ty` / `is_recovery_artifact`: every error
    /// kind that carries a self type reports it (routed through
    /// `Predicate::self_ty` for `Overflow` / `CycleDetected`), errors on
    /// the error sentinel are recovery artifacts, errors on real types are
    /// not, and kinds without a self type (`Mismatch`) are never artifacts.
    #[test]
    fn test_solve_error_self_ty_and_recovery_artifact() {
        let mut ctx = TypeContext::new();
        let err_ty = ctx.error();
        let int_ty = ctx.int(32, true);
        let span = crate::ast::DUMMY_SPAN;

        // NotFound / Ambiguous carry their self_ty directly.
        let not_found = SolveError::NotFound {
            trait_id: None,
            self_ty: int_ty,
            span,
        };
        assert_eq!(not_found.self_ty(), Some(int_ty));
        assert!(!not_found.is_recovery_artifact(&ctx));

        let not_found_err = SolveError::NotFound {
            trait_id: None,
            self_ty: err_ty,
            span,
        };
        assert!(not_found_err.is_recovery_artifact(&ctx));

        let ambiguous = SolveError::Ambiguous {
            trait_id: None,
            self_ty: int_ty,
            span,
            num_candidates: 0,
        };
        assert_eq!(ambiguous.self_ty(), Some(int_ty));
        assert!(!ambiguous.is_recovery_artifact(&ctx));

        // Overflow / CycleDetected route through the predicate's self_ty.
        let sized_ob = |ty: TypeId| Obligation {
            cause: ObligationCause {
                span,
                code: ObligationCauseCode::Misc,
            },
            predicate: Predicate::Sized { ty },
            recursion_depth: 0,
        };
        let overflow_err = SolveError::Overflow {
            obligation: Box::new(sized_ob(err_ty)),
            depth: 0,
        };
        assert_eq!(overflow_err.self_ty(), Some(err_ty));
        assert!(overflow_err.is_recovery_artifact(&ctx));

        let overflow_int = SolveError::Overflow {
            obligation: Box::new(sized_ob(int_ty)),
            depth: 0,
        };
        assert!(!overflow_int.is_recovery_artifact(&ctx));

        let cycle_err = SolveError::CycleDetected {
            predicate: Predicate::Sized { ty: err_ty },
        };
        assert_eq!(cycle_err.self_ty(), Some(err_ty));
        assert!(cycle_err.is_recovery_artifact(&ctx));

        // Mismatch has no self type — never a recovery artifact.
        let mismatch = SolveError::Mismatch {
            expected: int_ty,
            found: err_ty,
            span,
            note: String::new(),
        };
        assert_eq!(mismatch.self_ty(), None);
        assert!(!mismatch.is_recovery_artifact(&ctx));
    }

    /// A composite recovery type (`Vec<Error>`) is still an artifact —
    /// `contains_error` is checked, not the shallow `is_error`.
    #[test]
    fn test_recovery_artifact_catches_composite_recovery() {
        let mut ctx = TypeContext::new();
        let err_ty = ctx.error();
        let vec_err = ctx.struct_ty(DefId(100), vec![err_ty]);
        let span = crate::ast::DUMMY_SPAN;

        let not_found = SolveError::NotFound {
            trait_id: None,
            self_ty: vec_err,
            span,
        };
        assert!(
            not_found.is_recovery_artifact(&ctx),
            "an obligation on Vec<Error> is a recovery artifact"
        );
    }
}
