//! # SearchGraph — Cycle Detection, Caching, and Fixpoint Iteration
//!
//! Analogous to `rustc_type_ir::search_graph::SearchGraph`.
//! Provides stack‑based cycle detection with coinductive/inductive
//! classification, provisional caching for cycle participants, and
//! fixpoint iteration for cycle heads.
//!
//! ## Architecture
//!
//! - **Stack**: tracks the current path of goals being evaluated.
//! - **CycleHeads / HeadUsages**: records which cycle heads a goal
//!   depends on and how (inductive/coinductive path).
//! - **Provisional cache**: stores results of goals that depend on other
//!   goals still on the stack (cycle participants).  These are *not*
//!   moved to the global cache until the cycle head reaches a fixpoint.
//! - **Global cache**: stores final results for non‑cycle goals.
//! - **AvailableDepth**: tracks remaining recursion depth.  When overflow
//!   is encountered, the remaining depth is divided by
//!   `DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW` to prevent exponential blowup.
//! - **Fixpoint iteration**: cycle heads are re‑evaluated until the
//!   result stabilises (provisional result == final result).
//!
//! See the [rustc-dev-guide chapter](https://rustc-dev-guide.rust-lang.org/solve/caching.html)
//! for more details on the caching strategy.

use crate::hir::infer::TypeVariableKind;
use crate::hir::query::{DefaultCache as QueryCache, QueryCacheType};
use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval_ctxt::probe::CandidateHeadUsages;
use crate::hir::traits::solver::obligation::{
    HasChanged, ImplSource, Obligation, Predicate, SolveError,
};
use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;

// ── Constants ─────────────────────────────────────────────────────

/// Maximum number of fixpoint iterations before overflow.
pub const FIXPOINT_STEP_LIMIT: usize = 8;

/// When overflow is encountered, the remaining available depth for
/// nested goals is divided by this factor to prevent exponential blowup.
pub const DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW: usize = 4;

// ── Path kinds ────────────────────────────────────────────────────

/// How a cycle should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathKind {
    /// A path consisting of only inductive/unproductive steps.
    /// The initial provisional result is `Err(NoSolution)`.
    Inductive,
    /// A path which is not coinductive but we may want to change
    /// it to be so in the future.  We return ambiguity to prevent
    /// people from relying on this.
    Unknown,
    /// A path with at least one coinductive step.  Such cycles hold.
    Coinductive,
    /// A path which is treated as ambiguous.  Once a path has this
    /// kind, any other segment does not change its kind.
    ForcedAmbiguity,
}

impl PathKind {
    /// Returns the path kind when merging `self` with `rest`.
    /// This is equivalent to `max(self, rest)` where:
    ///   ForcedAmbiguity > Coinductive > Unknown > Inductive
    fn extend(self, rest: PathKind) -> PathKind {
        match (self, rest) {
            (PathKind::ForcedAmbiguity, _) | (_, PathKind::ForcedAmbiguity) => {
                PathKind::ForcedAmbiguity
            }
            (PathKind::Coinductive, _) | (_, PathKind::Coinductive) => PathKind::Coinductive,
            (PathKind::Unknown, _) | (_, PathKind::Unknown) => PathKind::Unknown,
            (PathKind::Inductive, PathKind::Inductive) => PathKind::Inductive,
        }
    }
}

// ── Goal key for cycle detection ──

/// A canonical key that uniquely identifies a goal for cycle detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GoalKey {
    pub kind: GoalKind,
    pub trait_id: Option<DefId>,
    pub self_ty: TypeId,
    pub args: Vec<TypeId>,
}

/// The kind of a goal for cycle detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalKind {
    Trait,
    AutoTrait,
    Sized,
    CopyLike,
    Projection,
}

impl GoalKey {
    pub fn from_obligation<'input>(
        obligation: &Obligation,
        ctx: &TypeContext<'input>,
    ) -> Option<Self> {
        let (kind, trait_id, self_ty, args) = match &obligation.predicate {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                if ctx.is_infer_var(resolved_self) {
                    return None;
                }
                let resolved_args: Vec<TypeId> =
                    args.iter().map(|a| ctx.resolve_binding(*a)).collect();
                (
                    GoalKind::Trait,
                    Some(*trait_id),
                    resolved_self,
                    resolved_args,
                )
            }
            Predicate::AutoTrait { trait_id, self_ty } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                if ctx.is_infer_var(resolved_self) {
                    return None;
                }
                (GoalKind::AutoTrait, Some(*trait_id), resolved_self, vec![])
            }
            Predicate::Sized { ty } => {
                let resolved = ctx.resolve_binding(*ty);
                if ctx.is_infer_var(resolved) {
                    return None;
                }
                (GoalKind::Sized, None, resolved, vec![])
            }
            Predicate::ProjectionEq {
                trait_id,
                self_ty,
                assoc_name,
                value,
            } => {
                let resolved_self = ctx.resolve_binding(*self_ty);
                if ctx.is_infer_var(resolved_self) {
                    return None;
                }
                let resolved_value = ctx.resolve_binding(*value);
                (
                    GoalKind::Projection,
                    Some(*trait_id),
                    resolved_self,
                    vec![resolved_value],
                )
            }
            Predicate::ProjectionNormalize { projection, target } => {
                let resolved_self = ctx.resolve_binding(projection.self_ty);
                if ctx.is_infer_var(resolved_self) {
                    return None;
                }
                let resolved_target = ctx.resolve_binding(*target);
                (
                    GoalKind::Projection,
                    Some(projection.trait_id),
                    resolved_self,
                    vec![resolved_target],
                )
            }
            Predicate::CopyLike { kind: _, ty } => {
                let resolved = ctx.resolve_binding(*ty);
                if ctx.is_infer_var(resolved) {
                    return None;
                }
                (GoalKind::CopyLike, None, resolved, vec![])
            }
            Predicate::Eq { a, b } => {
                let ra = ctx.resolve_binding(*a);
                let rb = ctx.resolve_binding(*b);
                if ctx.is_infer_var(ra) || ctx.is_infer_var(rb) {
                    return None;
                }
                // Eq goals are not cycle-participants — they resolve immediately
                // by unification.  Return None to skip cycle detection.
                return None;
            }
            Predicate::Sub { sub, sup } => {
                let rsub = ctx.resolve_binding(*sub);
                let rsup = ctx.resolve_binding(*sup);
                if ctx.is_infer_var(rsub) || ctx.is_infer_var(rsup) {
                    return None;
                }
                // Sub goals are not cycle-participants — they resolve immediately.
                return None;
            }
            Predicate::Match { scrutinee, .. } => {
                let resolved = ctx.resolve_binding(*scrutinee);
                if ctx.is_infer_var(resolved) {
                    return None;
                }
                // Match goals are not cycle-participants — they resolve immediately
                // by shape-based discharge.
                return None;
            }
            Predicate::Forall { .. }
            | Predicate::Exists { .. }
            | Predicate::Instance { .. }
            | Predicate::Let { .. }
            | Predicate::NormalizesTo { .. } => {
                // These are structural constraints that don't participate in
                // cycle detection — they are resolved by transforming the
                // constraint set, not by recursive evaluation.
                return None;
            }
        };
        Some(GoalKey {
            kind,
            trait_id,
            self_ty,
            args,
        })
    }
}

// ── Canonical goal key for infer-var-independent caching ───────────

/// A canonical index into the `var_values` vector of a `Response`.
///
/// Represents the position of an inference variable in the canonical
/// variable list.  These indices are used to map inference variables
/// between the evaluation context and cached responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalVarIndex(pub usize);

/// A canonicalized goal predicate with inference variable identity
/// stripped and universe/variable-kind information preserved.
///
/// A structural canonical representation of a TypeId that strips inference
/// variable identity but preserves structural type information (def_id,
/// args, etc.).  Equality and hashing are structural — not hash-only —
/// which is critical for soundness of the canonical cache.
/// See rustc's `CanonicalVarKind` + `Canonical<I, V>` in
/// `rustc_type_ir/src/canonical.rs` for the analogous design.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CanonicalTy {
    /// Infer vars are replaced by a sentinel: all produce the same canonical
    /// value so that the same goal with different infer var IDs maps to the
    /// same canonical key.
    InferSentinel,
    SkolemSentinel,
    GenericParam {
        index: usize,
    },
    Int {
        bits: u32,
        signed: bool,
    },
    UInt {
        bits: u32,
    },
    Float {
        bits: u32,
    },
    Bool,
    Char,
    Byte,
    USize,
    Never,
    Unit,
    Error,
    Type,
    Rational {
        int_bits: u8,
        frac_bits: u8,
    },
    Regex {
        pattern: String,
    },
    /// Predicate kind discriminator (0 = Trait, 1 = AutoTrait, etc.).
    /// Replaces the old type-punning of `GenericParam { index: N }` as
    /// a predicate tag.  See rustc's `canonicalize` for analogous encoding.
    PredicateDiscriminant(u8),
    /// A trait reference (DefId) in the canonical key.
    /// Replaces `GenericParam { index: trait_id.0 as usize }`.
    TraitRef(DefId),
    /// An associated type name in the canonical key.
    /// Replaces `Regex { pattern: assoc_name.to_string() }`.
    AssocName(Symbol),
    Adt {
        def_id: DefId,
        args: Vec<CanonicalTy>,
    },
    Tuple {
        elems: Vec<CanonicalTy>,
    },
    Array {
        elem: Box<CanonicalTy>,
        size: u64,
    },
    Slice {
        elem: Box<CanonicalTy>,
    },
    Ref {
        ty: Box<CanonicalTy>,
        mutable: bool,
    },
    Pointer {
        ty: Box<CanonicalTy>,
    },
    Ptr {
        size: Box<CanonicalTy>,
        pointee: Box<CanonicalTy>,
    },
    Fn {
        params: Vec<CanonicalTy>,
        ret: Box<CanonicalTy>,
    },
    DynTrait {
        traits: Vec<DefId>,
    },
    AssociatedType {
        trait_id: DefId,
        name: Symbol,
        self_ty: Box<CanonicalTy>,
    },
    Opaque {
        def_id: DefId,
        hidden: Option<Box<CanonicalTy>>,
    },
    Coproduct {
        alternatives: Vec<CanonicalTy>,
    },
    Forall {
        param_index: usize,
        body: Box<CanonicalTy>,
    },
    Exists {
        param_index: usize,
        base: Box<CanonicalTy>,
    },
    Mu {
        param_index: usize,
        body: Box<CanonicalTy>,
    },
    Nu {
        param_index: usize,
        body: Box<CanonicalTy>,
    },
    Poly {
        quantifiers: Vec<usize>,
        body: Box<CanonicalTy>,
    },
}

/// Convert a TypeId to a CanonicalTy tree, replacing InferVars/SkolemVars
/// with sentinels and recursing into composite types.  This is the structural
/// analogue of `hash_type_canonical` but produces a full tree instead of a
/// hash, enabling sound structural equality for cache keys.
/// See rustc's `Canonicalizer::canonicalize_ty` (rustc_type_ir/src/canonicalizer.rs).
pub fn ty_to_canonical<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
    bound: &[(usize, usize)],
) -> CanonicalTy {
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved) {
        TypeData::InferVar { .. } => CanonicalTy::InferSentinel,
        TypeData::SkolemVar { .. } => CanonicalTy::SkolemSentinel,
        TypeData::Int { bits, signed, .. } => CanonicalTy::Int {
            bits: *bits,
            signed: *signed,
        },
        TypeData::UInt { bits, .. } => CanonicalTy::UInt { bits: *bits },
        TypeData::Float { bits, .. } => CanonicalTy::Float { bits: *bits },
        TypeData::Bool => CanonicalTy::Bool,
        TypeData::Char => CanonicalTy::Char,
        TypeData::Byte => CanonicalTy::Byte,
        TypeData::USize => CanonicalTy::USize,
        TypeData::Never => CanonicalTy::Never,
        TypeData::Unit => CanonicalTy::Unit,
        TypeData::Error => CanonicalTy::Error,
        TypeData::Type => CanonicalTy::Type,
        TypeData::GenericParam { index, .. } => CanonicalTy::GenericParam {
            // De Bruijn: a binder-bound variable maps to its nesting depth
            // (alpha-equivalent quantified types — e.g. `∀X0. X0` vs
            // `∀X1. X1` — then produce the same canonical key).  A FREE
            // parameter (not bound by any enclosing quantifier) keeps its
            // original index.
            //
            // The binder stack is searched in REVERSE so that a shadowing
            // inner binder (`∀X. ∀X. X` — the inner `X` shadows the outer)
            // resolves to the NEAREST enclosing binder's depth, matching
            // De Bruijn semantics.
            index: bound
                .iter()
                .rev()
                .find(|(p, _)| *p == *index)
                .map_or(*index, |&(_, d)| d),
        },
        TypeData::Rational {
            int_bits,
            frac_bits,
        } => CanonicalTy::Rational {
            int_bits: *int_bits,
            frac_bits: *frac_bits,
        },
        TypeData::Regex { pattern } => CanonicalTy::Regex {
            pattern: pattern.clone(),
        },
        TypeData::Adt {
            kind: _,
            def_id,
            args,
        } => CanonicalTy::Adt {
            def_id: *def_id,
            args: args
                .iter()
                .map(|a| ty_to_canonical(*a, ctx, bound))
                .collect(),
        },
        TypeData::Tuple { elems } => CanonicalTy::Tuple {
            elems: elems
                .iter()
                .map(|e| ty_to_canonical(*e, ctx, bound))
                .collect(),
        },
        TypeData::Array { elem, size } => CanonicalTy::Array {
            elem: Box::new(ty_to_canonical(*elem, ctx, bound)),
            size: *size,
        },
        TypeData::Slice { elem } => CanonicalTy::Slice {
            elem: Box::new(ty_to_canonical(*elem, ctx, bound)),
        },
        TypeData::Ref { ty, mutable, .. } => CanonicalTy::Ref {
            ty: Box::new(ty_to_canonical(*ty, ctx, bound)),
            mutable: *mutable,
        },
        TypeData::Pointer { ty } => CanonicalTy::Pointer {
            ty: Box::new(ty_to_canonical(*ty, ctx, bound)),
        },
        TypeData::Ptr { size, pointee } => CanonicalTy::Ptr {
            size: Box::new(ty_to_canonical(*size, ctx, bound)),
            pointee: Box::new(ty_to_canonical(*pointee, ctx, bound)),
        },
        TypeData::Fn { params, ret, .. } => CanonicalTy::Fn {
            params: params
                .iter()
                .map(|p| ty_to_canonical(*p, ctx, bound))
                .collect(),
            ret: Box::new(ty_to_canonical(*ret, ctx, bound)),
        },
        TypeData::DynTrait { traits } => CanonicalTy::DynTrait {
            traits: traits.clone(),
        },
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => CanonicalTy::AssociatedType {
            trait_id: *trait_id,
            name: *name,
            self_ty: Box::new(ty_to_canonical(*self_ty, ctx, bound)),
        },
        TypeData::Opaque { def_id, hidden } => CanonicalTy::Opaque {
            def_id: *def_id,
            hidden: hidden.map(|h| Box::new(ty_to_canonical(h, ctx, bound))),
        },
        TypeData::Coproduct { alternatives } => CanonicalTy::Coproduct {
            alternatives: alternatives
                .iter()
                .map(|a| ty_to_canonical(*a, ctx, bound))
                .collect(),
        },
        TypeData::Forall {
            param_index, body, ..
        } => {
            let mut b = bound.to_vec();
            b.push((*param_index, bound.len()));
            CanonicalTy::Forall {
                // De Bruijn: the binder's nesting depth (alpha-equivalent
                // quantifiers — `∀X0. ...` vs `∀X1. ...` — agree).
                param_index: bound.len(),
                body: Box::new(ty_to_canonical(*body, ctx, &b)),
            }
        }
        TypeData::Exists {
            param_index, base, ..
        } => {
            let mut b = bound.to_vec();
            b.push((*param_index, bound.len()));
            CanonicalTy::Exists {
                param_index: bound.len(),
                base: Box::new(ty_to_canonical(*base, ctx, &b)),
            }
        }
        TypeData::Mu {
            param_index, body, ..
        } => {
            let mut b = bound.to_vec();
            b.push((*param_index, bound.len()));
            CanonicalTy::Mu {
                param_index: bound.len(),
                body: Box::new(ty_to_canonical(*body, ctx, &b)),
            }
        }
        TypeData::Nu {
            param_index, body, ..
        } => {
            let mut b = bound.to_vec();
            b.push((*param_index, bound.len()));
            CanonicalTy::Nu {
                param_index: bound.len(),
                body: Box::new(ty_to_canonical(*body, ctx, &b)),
            }
        }
        TypeData::Poly { quantifiers, body } => {
            let mut b = bound.to_vec();
            let mut depth = bound.len();
            for (i, _) in quantifiers {
                b.push((*i, depth));
                depth += 1; // Sequential De Bruijn depths — each quantifier
                // gets its own depth so body references distinguish the
                // binders (`∀A B. A → B` ≠ `∀A B. A → A` in the cache).
            }
            CanonicalTy::Poly {
                // The De Bruijn depth RANGE, not the original param
                // indices — alpha-equivalent Poly types (`∀A B. ...` vs
                // `∀C D. ...`) canonicalize identically.
                quantifiers: (bound.len()..depth).collect(),
                body: Box::new(ty_to_canonical(*body, ctx, &b)),
            }
        }
    }
}

/// Canonical key for the global cache.  Unlike the old `Vec<u64>` hash-only
/// representation, this stores structural `CanonicalTy` trees so that
/// equality is structural — not subject to hash collisions.
/// See rustc's `CanonicalInput` (rustc_next_trait_solver/src/solve/mod.rs).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CanonicalPredicate {
    /// Structural canonical representation of each TypeId in the predicate.
    pub types: Vec<CanonicalTy>,
    /// The kinds of each free inference variable in the goal, in order
    /// of first appearance during canonicalization.
    pub var_kinds: Vec<TypeVariableKind>,
    /// The highest universe index nameable by the caller.
    /// Used for HRTB `for<'a>` placeholder scoping.
    pub max_universe: usize,
}

/// A canonicalized response that stores the result with its inference
/// variable mapping, enabling safe cross-context caching.
///
/// The `var_values` vector maps `CanonicalVarIndex` → `Option<TypeId>`,
/// recording which concrete types (if any) each canonical var was
/// resolved to.  On cache hit, `instantiate_canonical` creates fresh
/// inference variables for unresolved entries and substitutes them
/// into the result.
#[derive(Debug, Clone)]
pub struct CanonicalResponse {
    pub impl_source: ImplSource,
    pub var_values: Vec<Option<TypeId>>,
    /// Original InferVar IDs for each canonical variable, parallel to
    /// `var_values`.  Stored so that `instantiate_canonical` can build
    /// a solution map keyed by original InferVar ID (required by
    /// `replace_infer`), which looks up by `TypeData::InferVar { id, .. }`.
    pub infer_var_ids: Vec<usize>,
    /// Universe index for each canonical variable, parallel to `var_values`.
    /// Used when creating fresh inference variables during instantiation
    /// to preserve universe scoping.  See rustc's `CanonicalVarKind::Ty { ui }`.
    pub var_universes: Vec<usize>,
    pub max_universe: usize,
    pub required_depth: usize,
    pub encountered_overflow: bool,
    pub nested_goals: Vec<GoalKey>,
}

/// Hash a TypeId canonically: resolve bindings + opaque types, strip infer var identity.
fn hash_type_canonical<'input>(ty: TypeId, ctx: &TypeContext<'input>, state: &mut DefaultHasher) {
    // Use resolve_binding (not resolve) to preserve opaque type identity.
    // ctx.resolve() chains resolve_opaque which strips the Opaque wrapper,
    // causing distinct opaque types with the same hidden type to collide.
    // We handle Opaque explicitly below by hashing def_id + hidden.
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved) {
        TypeData::InferVar { .. } | TypeData::SkolemVar { .. } => {
            // Volatile — hash a constant sentinel to ensure all unbound
            // infer vars produce the same canonical key regardless of ID.
            0u64.hash(state);
            0u64.hash(state);
        }
        TypeData::Int { bits, signed, .. } => {
            1u64.hash(state);
            bits.hash(state);
            signed.hash(state);
        }
        TypeData::UInt { bits, .. } => {
            2u64.hash(state);
            bits.hash(state);
        }
        TypeData::Float { bits } => {
            3u64.hash(state);
            bits.hash(state);
        }
        TypeData::Bool => 4u64.hash(state),
        TypeData::Char => 5u64.hash(state),
        TypeData::Byte => 6u64.hash(state),
        TypeData::USize => 7u64.hash(state),
        TypeData::Never => 8u64.hash(state),
        TypeData::Unit => 9u64.hash(state),
        TypeData::Error => 10u64.hash(state),
        TypeData::Type => 11u64.hash(state),
        TypeData::GenericParam { index, .. } => {
            12u64.hash(state);
            index.hash(state);
        }
        TypeData::Adt { def_id, args, .. } => {
            13u64.hash(state);
            def_id.hash(state);
            for a in args {
                hash_type_canonical(*a, ctx, state);
            }
        }
        TypeData::Tuple { elems } => {
            14u64.hash(state);
            for e in elems {
                hash_type_canonical(*e, ctx, state);
            }
        }
        TypeData::Array { elem, size } => {
            15u64.hash(state);
            hash_type_canonical(*elem, ctx, state);
            size.hash(state);
        }
        TypeData::Slice { elem } => {
            16u64.hash(state);
            hash_type_canonical(*elem, ctx, state);
        }
        TypeData::Ref { ty, mutable, .. } => {
            17u64.hash(state);
            hash_type_canonical(*ty, ctx, state);
            mutable.hash(state);
        }
        TypeData::Pointer { ty } => {
            18u64.hash(state);
            hash_type_canonical(*ty, ctx, state);
        }
        TypeData::Ptr { size, pointee } => {
            19u64.hash(state);
            hash_type_canonical(*size, ctx, state);
            hash_type_canonical(*pointee, ctx, state);
        }
        TypeData::Fn { params, ret } => {
            20u64.hash(state);
            for p in params {
                hash_type_canonical(*p, ctx, state);
            }
            hash_type_canonical(*ret, ctx, state);
        }
        TypeData::DynTrait { traits } => {
            21u64.hash(state);
            for t in traits {
                t.hash(state);
            }
        }
        TypeData::Exists { base, .. } => {
            22u64.hash(state);
            hash_type_canonical(*base, ctx, state);
        }
        TypeData::Forall {
            body, param_index, ..
        } => {
            23u64.hash(state);
            param_index.hash(state);
            hash_type_canonical(*body, ctx, state);
        }
        TypeData::Poly {
            body, quantifiers, ..
        } => {
            29u64.hash(state);
            for (idx, _name) in quantifiers {
                idx.hash(state);
            }
            hash_type_canonical(*body, ctx, state);
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            24u64.hash(state);
            trait_id.hash(state);
            name.hash(state);
            hash_type_canonical(*self_ty, ctx, state);
        }
        TypeData::Coproduct { alternatives } => {
            25u64.hash(state);
            for a in alternatives {
                hash_type_canonical(*a, ctx, state);
            }
        }
        TypeData::Mu {
            body, param_index, ..
        } => {
            26u64.hash(state);
            param_index.hash(state);
            hash_type_canonical(*body, ctx, state);
        }
        TypeData::Nu {
            body, param_index, ..
        } => {
            30u64.hash(state);
            param_index.hash(state);
            hash_type_canonical(*body, ctx, state);
        }
        TypeData::Rational {
            int_bits,
            frac_bits,
        } => {
            27u64.hash(state);
            int_bits.hash(state);
            frac_bits.hash(state);
        }
        TypeData::Regex { pattern } => {
            28u64.hash(state);
            pattern.hash(state);
        }
        TypeData::Opaque { def_id, hidden, .. } => {
            31u64.hash(state);
            def_id.hash(state);
            if let Some(ty) = hidden {
                hash_type_canonical(*ty, ctx, state);
            }
        }
    }
}

/// Recursively walk a TypeId and all its children for infer/skolem vars.
/// Extracted as a standalone function so it can be used by both
/// `predicate_has_unresolved_vars` and `canonicalize_response` (for
/// checking subst values before caching).
pub fn ty_has_unresolved_vars<'input>(ty: TypeId, ctx: &TypeContext<'input>) -> bool {
    // Use ctx.resolve() instead of resolve_binding() so that opaque
    // side-table state (opaque_hidden map) is also followed.
    let resolved = ctx.resolve(ty);
    match ctx.get(resolved) {
        TypeData::InferVar { .. } | TypeData::SkolemVar { .. } | TypeData::GenericParam { .. } => {
            true
        }
        // Opaque type without a known hidden type — still unresolved.
        TypeData::Opaque { hidden: None, .. } => true,
        // All composite types: delegate to visit_type_children for recursion.
        // This ensures that when new TypeData variants with TypeId children
        // are added to visit_type_children, ty_has_unresolved_vars automatically
        // benefits without needing its own match arms.
        _ => {
            let mut found = false;
            crate::hir::types::visit_type_children(ty, ctx, &mut |child| {
                if ty_has_unresolved_vars(child, ctx) {
                    found = true;
                }
            });
            found
        }
    }
}

/// Walk ALL TypeId fields in a Predicate and check for unresolved infer/skolem vars.
/// Unlike `predicate.self_ty()` which only checks the primary self type, this
/// function inspects EVERY TypeId (args, values, targets, etc.) to ensure no
/// goal with free inference variables enters the canonical cache.
/// Crucially, it also recurses INTO composite types (Adt, Tuple, Ref, Fn, etc.)
/// so that `Trait<Vec<?x>>` is correctly identified as containing an infer var.
fn predicate_has_unresolved_vars<'input>(predicate: &Predicate, ctx: &TypeContext<'input>) -> bool {
    match predicate {
        Predicate::Trait { self_ty, args, .. } => {
            ty_has_unresolved_vars(*self_ty, ctx)
                || args.iter().any(|a| ty_has_unresolved_vars(*a, ctx))
        }
        Predicate::AutoTrait { self_ty, .. } => ty_has_unresolved_vars(*self_ty, ctx),
        Predicate::Sized { ty } => ty_has_unresolved_vars(*ty, ctx),
        Predicate::CopyLike { ty, .. } => ty_has_unresolved_vars(*ty, ctx),
        Predicate::ProjectionEq { self_ty, value, .. } => {
            ty_has_unresolved_vars(*self_ty, ctx) || ty_has_unresolved_vars(*value, ctx)
        }
        Predicate::ProjectionNormalize { projection, target } => {
            ty_has_unresolved_vars(projection.self_ty, ctx) || ty_has_unresolved_vars(*target, ctx)
        }
        Predicate::NormalizesTo { projection, target } => {
            ty_has_unresolved_vars(projection.self_ty, ctx) || ty_has_unresolved_vars(*target, ctx)
        }
        Predicate::Eq { a, b } => {
            ty_has_unresolved_vars(*a, ctx) || ty_has_unresolved_vars(*b, ctx)
        }
        Predicate::Sub { sub, sup } => {
            ty_has_unresolved_vars(*sub, ctx) || ty_has_unresolved_vars(*sup, ctx)
        }
        Predicate::Match { scrutinee, .. } => ty_has_unresolved_vars(*scrutinee, ctx),
        Predicate::Forall { body, .. } | Predicate::Exists { body, .. } => {
            predicate_has_unresolved_vars(body, ctx)
        }
        Predicate::Instance {
            scheme_ty,
            instantiation_ty,
            ..
        } => {
            ty_has_unresolved_vars(*scheme_ty, ctx)
                || ty_has_unresolved_vars(*instantiation_ty, ctx)
        }
        Predicate::Let { def, .. } => predicate_has_unresolved_vars(def, ctx),
    }
}

/// Collect free inference variable kinds from a Predicate by walking
/// all TypeId fields.  Records the `TypeVariableKind` of each distinct
/// infer var in order of first appearance.
fn collect_predicate_var_kinds<'input>(
    predicate: &Predicate,
    ctx: &TypeContext<'input>,
) -> Vec<TypeVariableKind> {
    let mut var_kinds: Vec<TypeVariableKind> = Vec::new();
    let mut seen: HashMap<usize, usize> = HashMap::default();
    let self_ty = predicate.self_ty();
    let resolved = ctx.resolve_binding(self_ty);
    if let TypeData::InferVar { id, .. } = ctx.get(resolved) {
        let canon_idx = seen.len();
        let idx = *seen.entry(*id).or_insert(canon_idx);
        if idx == canon_idx {
            var_kinds.push(TypeVariableKind::Any);
        }
    }
    var_kinds
}

/// Compute a canonical key for an obligation that has NO free inference variables.
///
/// Returns `None` if the goal contains unresolved InferVars or SkolemVars —
/// those goals are context-dependent and MUST NOT be cached (caching a
/// pre-resolution key with a post-resolution result is unsound).
///
/// When `Some(...)` is returned, the key is a structural hash that is identical
/// for structurally identical goals (same type structure, same trait, same args).
pub fn canonicalize_goal_key<'input>(
    obligation: &Obligation,
    ctx: &TypeContext<'input>,
    max_universe: usize,
) -> Option<CanonicalPredicate> {
    // SAFETY: Goals with unresolved inference variables must NOT be cached.
    // The canonical key is computed BEFORE evaluation (variables are unresolved),
    // but the result is stored AFTER evaluation (variables may be resolved).
    // Caching a post-resolution result under a pre-resolution key is unsound.
    if predicate_has_unresolved_vars(&obligation.predicate, ctx) {
        return None;
    }

    let mut types: Vec<CanonicalTy> = Vec::new();
    let mut h = |ct: CanonicalTy| types.push(ct);

    // Collect CanonicalTy for each TypeId in the predicate.
    // Hash the predicate kind discriminator fields.
    match &obligation.predicate {
        Predicate::Trait {
            trait_id,
            self_ty,
            args,
        } => {
            h(CanonicalTy::PredicateDiscriminant(0));
            h(CanonicalTy::TraitRef(*trait_id));
            h(ty_to_canonical(*self_ty, ctx, &[]));
            for a in args {
                h(ty_to_canonical(*a, ctx, &[]));
            }
        }
        Predicate::AutoTrait { trait_id, self_ty } => {
            h(CanonicalTy::PredicateDiscriminant(1));
            h(CanonicalTy::TraitRef(*trait_id));
            h(ty_to_canonical(*self_ty, ctx, &[]));
        }
        Predicate::Sized { ty } => {
            h(CanonicalTy::PredicateDiscriminant(2));
            h(ty_to_canonical(*ty, ctx, &[]));
        }
        Predicate::CopyLike { kind, ty } => {
            h(CanonicalTy::PredicateDiscriminant(3));
            h(CanonicalTy::GenericParam {
                index: *kind as usize,
            });
            h(ty_to_canonical(*ty, ctx, &[]));
        }
        Predicate::ProjectionEq {
            trait_id,
            self_ty,
            assoc_name,
            value,
        } => {
            h(CanonicalTy::PredicateDiscriminant(4));
            h(CanonicalTy::TraitRef(*trait_id));
            h(CanonicalTy::AssocName(*assoc_name));
            h(ty_to_canonical(*self_ty, ctx, &[]));
            h(ty_to_canonical(*value, ctx, &[]));
        }
        Predicate::ProjectionNormalize { projection, target } => {
            h(CanonicalTy::PredicateDiscriminant(5));
            h(CanonicalTy::TraitRef(projection.trait_id));
            h(CanonicalTy::AssocName(projection.assoc_name));
            h(ty_to_canonical(projection.self_ty, ctx, &[]));
            h(ty_to_canonical(*target, ctx, &[]));
            for a in &projection.args {
                h(ty_to_canonical(*a, ctx, &[]));
            }
        }
        Predicate::Eq { a, b } => {
            h(CanonicalTy::PredicateDiscriminant(6));
            h(ty_to_canonical(*a, ctx, &[]));
            h(ty_to_canonical(*b, ctx, &[]));
        }
        Predicate::Sub { sub, sup } => {
            h(CanonicalTy::PredicateDiscriminant(7));
            h(ty_to_canonical(*sub, ctx, &[]));
            h(ty_to_canonical(*sup, ctx, &[]));
        }
        Predicate::Match { scrutinee, .. } => {
            h(CanonicalTy::PredicateDiscriminant(8));
            h(ty_to_canonical(*scrutinee, ctx, &[]));
        }
        Predicate::Forall { body, .. } => {
            return None;
        }
        Predicate::Exists { body, .. } => {
            return None;
        }
        Predicate::Instance { .. } | Predicate::Let { .. } | Predicate::NormalizesTo { .. } => {
            return None;
        }
    }

    Some(CanonicalPredicate {
        types,
        var_kinds: collect_predicate_var_kinds(&obligation.predicate, ctx),
        max_universe,
    })
}

/// Recursively walk a TypeId and collect all InferVar IDs into `var_values`
/// via the `seen` dedup map.  Uses `visit_type_children` for recursive
/// traversal, so it automatically covers all composite type variants.
/// Extracted as a standalone function so that `canonicalize_response` can
/// use it for both nested obligations and subst mapping values.
pub fn collect_from_ty<'input>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
    seen: &mut HashMap<usize, usize>,
    var_values: &mut Vec<Option<TypeId>>,
    var_universes: &mut Vec<Option<usize>>,
) {
    let resolved = ctx.resolve_binding(ty);
    if let TypeData::InferVar { id, universe, .. } = ctx.get(resolved) {
        let canon_idx = seen.len();
        let idx = *seen.entry(*id).or_insert(canon_idx);
        if idx == canon_idx {
            var_values.push(None);
            var_universes.push(Some(*universe));
        }
        return;
    }
    // Recurse into composite types via the shared walker.
    crate::hir::types::visit_type_children(ty, ctx, &mut |child_ty| {
        collect_from_ty(child_ty, ctx, seen, var_values, var_universes);
    });
}

/// Walk ALL TypeId fields in a Predicate, collecting InferVar IDs into
/// `var_values` via the `seen` dedup map.  This is the collection analogue
/// of `predicate_has_unresolved_vars` — it finds the same InferVars but
/// records them for canonical instantiation instead of returning a bool.
fn collect_infer_ids_from_predicate<'input>(
    predicate: &Predicate,
    ctx: &TypeContext<'input>,
    seen: &mut HashMap<usize, usize>,
    var_values: &mut Vec<Option<TypeId>>,
    var_universes: &mut Vec<Option<usize>>,
) {
    // Uses module-level `collect_from_ty` (defined above) — no local shadow.
    match predicate {
        Predicate::Trait { self_ty, args, .. } => {
            collect_from_ty(*self_ty, ctx, seen, var_values, var_universes);
            for a in args {
                collect_from_ty(*a, ctx, seen, var_values, var_universes);
            }
        }
        Predicate::AutoTrait { self_ty, .. } => {
            collect_from_ty(*self_ty, ctx, seen, var_values, var_universes)
        }
        Predicate::Sized { ty } => collect_from_ty(*ty, ctx, seen, var_values, var_universes),
        Predicate::CopyLike { ty, .. } => {
            collect_from_ty(*ty, ctx, seen, var_values, var_universes)
        }
        Predicate::ProjectionEq { self_ty, value, .. } => {
            collect_from_ty(*self_ty, ctx, seen, var_values, var_universes);
            collect_from_ty(*value, ctx, seen, var_values, var_universes);
        }
        Predicate::ProjectionNormalize { projection, target } => {
            collect_from_ty(projection.self_ty, ctx, seen, var_values, var_universes);
            collect_from_ty(*target, ctx, seen, var_values, var_universes);
            for a in &projection.args {
                collect_from_ty(*a, ctx, seen, var_values, var_universes);
            }
        }
        Predicate::NormalizesTo { projection, target } => {
            collect_from_ty(projection.self_ty, ctx, seen, var_values, var_universes);
            collect_from_ty(*target, ctx, seen, var_values, var_universes);
            for a in &projection.args {
                collect_from_ty(*a, ctx, seen, var_values, var_universes);
            }
        }
        Predicate::Eq { a, b } => {
            collect_from_ty(*a, ctx, seen, var_values, var_universes);
            collect_from_ty(*b, ctx, seen, var_values, var_universes);
        }
        Predicate::Sub { sub, sup } => {
            collect_from_ty(*sub, ctx, seen, var_values, var_universes);
            collect_from_ty(*sup, ctx, seen, var_values, var_universes);
        }
        Predicate::Match { scrutinee, .. } => {
            collect_from_ty(*scrutinee, ctx, seen, var_values, var_universes)
        }
        Predicate::Forall { body, .. } | Predicate::Exists { body, .. } => {
            collect_infer_ids_from_predicate(body, ctx, seen, var_values, var_universes);
        }
        Predicate::Instance {
            scheme_ty,
            instantiation_ty,
            ..
        } => {
            collect_from_ty(*scheme_ty, ctx, seen, var_values, var_universes);
            collect_from_ty(*instantiation_ty, ctx, seen, var_values, var_universes);
        }
        Predicate::Let { def, body } => {
            collect_infer_ids_from_predicate(def, ctx, seen, var_values, var_universes);
            collect_infer_ids_from_predicate(body, ctx, seen, var_values, var_universes);
        }
    }
}

/// Walk all TypeId fields in an ImplSource, collecting free inference
/// variables into `var_values`.  Returns a `CanonicalResponse` that
/// preserves the result alongside the variable mapping for future
/// instantiation (see `instantiate_canonical`).
///
/// Currently only `ImplSource::Builtin` is cached (it has no TypeIds,
/// so this is a no-op).  Other variants walk nested obligations to
/// collect infer vars for future cache expansion.
pub fn canonicalize_response<'input>(
    result: &Result<ImplSource, SolveError>,
    ctx: &TypeContext<'input>,
    max_universe: usize,
    required_depth: usize,
    encountered_overflow: bool,
    nested_goals: Vec<GoalKey>,
) -> Option<CanonicalResponse> {
    let mut var_values: Vec<Option<TypeId>> = Vec::new();
    let mut seen: HashMap<usize, usize> = HashMap::default();
    let mut var_universes: Vec<Option<usize>> = Vec::new();

    match result {
        Ok(ImplSource::Builtin(_)) => {
            // Builtin has no TypeId fields — safe to cache directly.
        }
        Ok(ImplSource::UserDefined { nested, subst, .. })
        | Ok(ImplSource::Poly { nested, subst }) => {
            // Refuse to cache if any nested obligation contains an unresolved var.
            for oblig in nested {
                if predicate_has_unresolved_vars(&oblig.predicate, ctx) {
                    return None;
                }
            }
            // Also refuse if any subst value contains an unresolved variable.
            // subst is only walked for InferVars below via collect_from_ty,
            // but SkolemVars/GenericParams in subst would NOT be collected
            // and would leak stale variables across contexts.
            {
                let mut has_unresolved = false;
                subst.for_each_value(|ty| {
                    if ty_has_unresolved_vars(ty, ctx) {
                        has_unresolved = true;
                    }
                });
                if has_unresolved {
                    return None;
                }
            }
            // Collect free inference variables from nested obligations.
            for obligation in nested {
                collect_infer_ids_from_predicate(
                    &obligation.predicate,
                    ctx,
                    &mut seen,
                    &mut var_values,
                    &mut var_universes,
                );
            }
            // Also collect InferVars from the subst mapping (recursively
            // through composite types via collect_from_ty / visit_type_children).
            subst.for_each_value(|ty| {
                collect_from_ty(ty, ctx, &mut seen, &mut var_values, &mut var_universes);
            });
        }
        Ok(ImplSource::Param(nested))
        | Ok(ImplSource::Object { nested, .. })
        | Ok(ImplSource::Auto { nested }) => {
            for oblig in nested {
                if predicate_has_unresolved_vars(&oblig.predicate, ctx) {
                    return None;
                }
            }
            for obligation in nested {
                collect_infer_ids_from_predicate(
                    &obligation.predicate,
                    ctx,
                    &mut seen,
                    &mut var_values,
                    &mut var_universes,
                );
            }
        }
        Ok(ImplSource::Deferred { .. }) | Err(_) => {
            return None;
        }
    }

    // Build infer_var_ids from the seen map: for each canonical index
    // (0..seen.len()), find the original InferVar ID.
    let infer_var_ids: Vec<usize> = {
        let mut pairs: Vec<(usize, usize)> = seen.into_iter().collect();
        pairs.sort_by_key(|(_, canon_idx)| *canon_idx);
        pairs.into_iter().map(|(infer_id, _)| infer_id).collect()
    };

    // Build var_universes: all inferred variables share the same universe
    // (from the evaluation context).  Parallel to var_values.
    // TODO: Per-variable universe tracking.  Currently all variables share
    // `max_universe`.  When `collect_infer_ids_from_predicate` records
    // per-variable universes during canonicalization, this vec will contain
    // distinct values for each canonical variable.
    let var_universes: Vec<usize> = var_universes
        .iter()
        .map(|u| u.unwrap_or(max_universe))
        .collect();

    Some(CanonicalResponse {
        impl_source: match result {
            Ok(impl_source) => impl_source.clone(),
            Err(_) => unreachable!(),
        },
        var_values,
        infer_var_ids,
        var_universes,
        max_universe,
        required_depth,
        encountered_overflow,
        nested_goals,
    })
}

/// Build the solution map using a callback pattern.
/// The callback `f` receives `&[already_created_vars]` and the canonical
/// index, allowing subsequent variables to reference earlier ones (for
/// subtyping chains like `?0 <: ?1`).  Returns the solution map keyed by
/// original InferVar IDs.
/// See rustc's `CanonicalVarValues::instantiate(f)`.
pub fn instantiate_solution_map(
    response: &CanonicalResponse,
    f: &mut dyn FnMut(&[TypeId], usize) -> TypeId,
) -> HashMap<usize, TypeId> {
    let mut solution: HashMap<usize, TypeId> = HashMap::default();
    let mut created: Vec<TypeId> = Vec::new();
    for (canon_idx, value) in response.var_values.iter().enumerate() {
        let fresh = match value {
            Some(ty) => *ty,
            None => f(&created, canon_idx),
        };
        solution.insert(response.infer_var_ids[canon_idx], fresh);
        created.push(fresh);
    }
    solution
}

/// Instantiate a cached `CanonicalResponse` into the current inference
/// context.  Creates fresh inference variables for canonical unresolved
/// entries in `var_values` and substitutes them into the `ImplSource`.
/// Uses a callback pattern (like rustc's `CanonicalVarValues::instantiate(f)`)
/// where the callback receives already-created vars so that each new variable
/// can reference previously-created ones (for subtyping chains).
/// See rustc's `instantiate_and_apply_query_response`.
pub fn instantiate_canonical<'input>(
    response: &CanonicalResponse,
    mut infer: Option<&mut crate::hir::infer::InferenceContext>,
    ctx: &mut TypeContext<'input>,
) -> ImplSource {
    // Defensive check: if no InferenceContext is available and any canonical
    // variable needs a fresh type, abort and return the cached source as-is.
    // Under current guard invariants (var_values is always empty), this path
    // is unreachable — it exists as a safety net for future guard relaxation.
    // The debug_assert catches guard violations in debug builds.
    debug_assert!(
        !response.var_values.iter().any(|v| v.is_none()) || infer.is_some(),
        "instantiate_canonical: var_values contains None entries but no \
         InferenceContext — the guard should have prevented this"
    );
    if infer.is_none() && response.var_values.iter().any(|v| v.is_none()) {
        return response.impl_source.clone();
    }
    let solution = instantiate_solution_map(response, &mut |created, canon_idx| {
        let universe = response.var_universes.get(canon_idx).copied().unwrap_or(0);
        infer.as_mut().unwrap().new_type_var_with_universe(
            ctx,
            crate::hir::infer::TypeVariableKind::Any,
            crate::hir::infer::VarOrigin::Synthetic,
            universe,
        )
    });
    if solution.is_empty() {
        return response.impl_source.clone();
    }
    // Apply substitution via replace_infer through all TypeId fields.
    use crate::hir::infer::replace_infer;
    fn subst_pred<'input>(
        pred: &Predicate,
        sol: &HashMap<usize, TypeId>,
        ctx: &TypeContext<'input>,
    ) -> Predicate {
        match pred {
            Predicate::Trait {
                trait_id,
                self_ty,
                args,
            } => Predicate::Trait {
                trait_id: *trait_id,
                self_ty: replace_infer(*self_ty, sol, ctx),
                args: args.iter().map(|a| replace_infer(*a, sol, ctx)).collect(),
            },
            Predicate::AutoTrait { trait_id, self_ty } => Predicate::AutoTrait {
                trait_id: *trait_id,
                self_ty: replace_infer(*self_ty, sol, ctx),
            },
            Predicate::Sized { ty } => Predicate::Sized {
                ty: replace_infer(*ty, sol, ctx),
            },
            Predicate::CopyLike { kind, ty } => Predicate::CopyLike {
                kind: *kind,
                ty: replace_infer(*ty, sol, ctx),
            },
            Predicate::ProjectionEq {
                trait_id,
                self_ty,
                assoc_name,
                value,
            } => Predicate::ProjectionEq {
                trait_id: *trait_id,
                self_ty: replace_infer(*self_ty, sol, ctx),
                assoc_name: *assoc_name,
                value: replace_infer(*value, sol, ctx),
            },
            Predicate::ProjectionNormalize { projection, target } => {
                Predicate::ProjectionNormalize {
                    projection: crate::hir::traits::solver::obligation::ProjectionTy {
                        trait_id: projection.trait_id,
                        self_ty: replace_infer(projection.self_ty, sol, ctx),
                        args: projection
                            .args
                            .iter()
                            .map(|a| replace_infer(*a, sol, ctx))
                            .collect(),
                        assoc_name: projection.assoc_name,
                    },
                    target: replace_infer(*target, sol, ctx),
                }
            }
            Predicate::NormalizesTo { projection, target } => Predicate::NormalizesTo {
                projection: crate::hir::traits::solver::obligation::ProjectionTy {
                    trait_id: projection.trait_id,
                    self_ty: replace_infer(projection.self_ty, sol, ctx),
                    args: projection
                        .args
                        .iter()
                        .map(|a| replace_infer(*a, sol, ctx))
                        .collect(),
                    assoc_name: projection.assoc_name,
                },
                target: replace_infer(*target, sol, ctx),
            },
            Predicate::Eq { a, b } => Predicate::Eq {
                a: replace_infer(*a, sol, ctx),
                b: replace_infer(*b, sol, ctx),
            },
            Predicate::Sub { sub, sup } => Predicate::Sub {
                sub: replace_infer(*sub, sol, ctx),
                sup: replace_infer(*sup, sol, ctx),
            },
            Predicate::Match {
                scrutinee,
                branches_id,
            } => Predicate::Match {
                scrutinee: replace_infer(*scrutinee, sol, ctx),
                branches_id: *branches_id,
            },
            Predicate::Forall { body } => Predicate::Forall {
                body: Box::new(subst_pred(body, sol, ctx)),
            },
            Predicate::Exists { body } => Predicate::Exists {
                body: Box::new(subst_pred(body, sol, ctx)),
            },
            Predicate::Instance {
                scheme_ty,
                instantiation_ty,
            } => Predicate::Instance {
                scheme_ty: replace_infer(*scheme_ty, sol, ctx),
                instantiation_ty: replace_infer(*instantiation_ty, sol, ctx),
            },
            Predicate::Let { def, body } => Predicate::Let {
                def: Box::new(subst_pred(def, sol, ctx)),
                body: Box::new(subst_pred(body, sol, ctx)),
            },
        }
    }
    fn subst_nested<'input>(
        nested: &[Obligation],
        sol: &HashMap<usize, TypeId>,
        ctx: &TypeContext<'input>,
    ) -> Vec<Obligation> {
        nested
            .iter()
            .map(|o| {
                let mut o2 = o.clone();
                o2.predicate = subst_pred(&o2.predicate, sol, ctx);
                o2
            })
            .collect()
    }
    match &response.impl_source {
        ImplSource::Builtin(_) => response.impl_source.clone(),
        ImplSource::UserDefined {
            cand_idx,
            subst,
            nested,
        } => {
            let mut subst2 = subst.clone();
            subst2.map_values(&mut |ty| replace_infer(ty, &solution, ctx));
            ImplSource::UserDefined {
                cand_idx: *cand_idx,
                subst: subst2,
                nested: subst_nested(nested, &solution, ctx),
            }
        }
        ImplSource::Param(nested) => ImplSource::Param(subst_nested(nested, &solution, ctx)),
        ImplSource::Object {
            object_trait_id,
            nested,
        } => ImplSource::Object {
            object_trait_id: *object_trait_id,
            nested: subst_nested(nested, &solution, ctx),
        },
        ImplSource::Auto { nested } => ImplSource::Auto {
            nested: subst_nested(nested, &solution, ctx),
        },
        ImplSource::Poly { subst, nested } => {
            let mut subst2 = subst.clone();
            subst2.map_values(&mut |ty| replace_infer(ty, &solution, ctx));
            ImplSource::Poly {
                subst: subst2,
                nested: subst_nested(nested, &solution, ctx),
            }
        }
        ImplSource::Deferred { .. } => response.impl_source.clone(),
    }
}

// ── Available depth ───────────────────────────────────────────────

/// Tracks the remaining recursion depth available for evaluation.
/// When overflow is encountered, the depth is divided by
/// `DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW` to prevent exponential blowup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AvailableDepth(pub usize);

impl AvailableDepth {
    /// Returns the allowed depth for nested goals, given the current
    /// stack state.  Returns `None` if the recursion limit is reached.
    fn allowed_for_nested(
        root_depth: AvailableDepth,
        stack: &[StackEntry],
        lower_available_depth: bool,
    ) -> Option<AvailableDepth> {
        if let Some(last) = stack.last() {
            if !lower_available_depth {
                return Some(last.available_depth);
            }
            if last.available_depth.0 == 0 {
                return None;
            }
            Some(if last.encountered_overflow {
                AvailableDepth(last.available_depth.0 / DIVIDE_AVAILABLE_DEPTH_ON_OVERFLOW)
            } else {
                AvailableDepth(last.available_depth.0 - 1)
            })
        } else {
            Some(root_depth)
        }
    }

    /// Whether a global cache entry with the given `required_depth` is
    /// applicable at this depth.
    fn cache_entry_is_applicable(self, required_depth: usize) -> bool {
        self.0 >= required_depth
    }
}

// ── Head usages ───────────────────────────────────────────────────

/// Tracks how a cycle head was used by nested goals.
/// This is used to determine whether re-evaluating a cycle head
/// could change the result of dependent provisional cache entries.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadUsages {
    pub inductive: u32,
    pub unknown: u32,
    pub coinductive: u32,
    pub forced_ambiguity: u32,
}

impl HeadUsages {
    fn add_usage(&mut self, path: PathKind) {
        match path {
            PathKind::Inductive => self.inductive += 1,
            PathKind::Unknown => self.unknown += 1,
            PathKind::Coinductive => self.coinductive += 1,
            PathKind::ForcedAmbiguity => self.forced_ambiguity += 1,
        }
    }

    fn add_usages_from_nested(&mut self, usages: HeadUsages) {
        self.inductive += if usages.inductive == 0 { 0 } else { 1 };
        self.unknown += if usages.unknown == 0 { 0 } else { 1 };
        self.coinductive += if usages.coinductive == 0 { 0 } else { 1 };
        self.forced_ambiguity += if usages.forced_ambiguity == 0 { 0 } else { 1 };
    }

    fn is_empty(self) -> bool {
        self.inductive == 0
            && self.unknown == 0
            && self.coinductive == 0
            && self.forced_ambiguity == 0
    }

    fn is_single(self, path_kind: PathKind) -> bool {
        match path_kind {
            PathKind::Inductive => {
                self.unknown == 0 && self.coinductive == 0 && self.forced_ambiguity == 0
            }
            PathKind::Unknown => {
                self.inductive == 0 && self.coinductive == 0 && self.forced_ambiguity == 0
            }
            PathKind::Coinductive => {
                self.inductive == 0 && self.unknown == 0 && self.forced_ambiguity == 0
            }
            PathKind::ForcedAmbiguity => {
                self.inductive == 0 && self.unknown == 0 && self.coinductive == 0
            }
        }
    }
}

// ── Cycle head ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct CycleHead {
    path_from_head: PathKind,
    usages: HeadUsages,
}

/// All cycle heads a given goal depends on, ordered by stack depth.
#[derive(Debug, Clone, Default)]
struct CycleHeads {
    /// Map from stack depth to cycle head.
    heads: Vec<(usize, CycleHead)>,
}

impl CycleHeads {
    fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    fn highest_cycle_head_index(&self) -> usize {
        self.heads.last().map(|(d, _)| *d).unwrap_or(0)
    }

    fn insert(&mut self, depth: usize, path_from_head: PathKind, usages: HeadUsages) {
        let head = CycleHead {
            path_from_head,
            usages,
        };
        // Keep sorted by depth (ascending).
        // Since depths are typically pushed in order, we can optimise for
        // the common case: the new depth is >= all existing depths.
        if self.heads.last().map_or(true, |(d, _)| *d <= depth) {
            self.heads.push((depth, head));
        } else {
            // Find insertion point (binary search would be better but
            // the list is small — typically < 5 entries).
            match self.heads.iter().position(|(d, _)| *d > depth) {
                Some(pos) => self.heads.insert(pos, (depth, head)),
                None => self.heads.push((depth, head)),
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, CycleHead)> + '_ {
        self.heads.iter().map(|(d, h)| (*d, *h))
    }
}

// ── Stack entry ───────────────────────────────────────────────────

/// An entry on the evaluation stack, tracking all state needed for
/// cycle detection, caching, and fixpoint iteration.
#[derive(Debug, Clone)]
pub(crate) struct StackEntry {
    pub(crate) key: GoalKey,
    step_kind_from_parent: PathKind,
    available_depth: AvailableDepth,
    min_reached_available_depth: AvailableDepth,
    /// Set when re-running this goal after a cycle is detected.
    provisional_result: Option<Result<ImplSource, SolveError>>,
    /// All cycle heads this goal depends on.
    heads: CycleHeads,
    /// Whether evaluating this goal encountered overflow.
    encountered_overflow: bool,
    /// Whether and how this goal has been used as a cycle head.
    usages: Option<HeadUsages>,
    /// The nested goals of this goal.
    nested_goals: HashSet<GoalKey>,
}

impl StackEntry {
    fn required_depth(&self) -> usize {
        self.available_depth.0 - self.min_reached_available_depth.0
    }
}

// ── Provisional cache ─────────────────────────────────────────────

/// A provisional result of a goal that depends on other goals still
/// on the stack (cycle participants).  These are kept locally and
/// NOT moved to the global cache until the cycle head reaches a fixpoint.
#[derive(Debug, Clone)]
struct ProvisionalCacheEntry {
    encountered_overflow: bool,
    heads: CycleHeads,
    path_from_head: PathKind,
    result: Result<ImplSource, SolveError>,
}

// ── SearchGraph ───────────────────────────────────────────────────

/// The search graph is responsible for caching and cycle detection in
/// the trait solver.
///
/// Key responsibilities:
/// - Stack-based cycle detection with coinductive/inductive classification
/// - Provisional caching for cycle participants
/// - Global caching for non-cycle goals
/// - Available depth tracking with overflow division
/// - Fixpoint iteration for cycle heads
pub struct SearchGraph {
    /// The evaluation stack — goals currently being evaluated.
    stack: Vec<StackEntry>,
    /// Provisional cache: results of goals that depend on other goals
    /// still on the stack.  Keyed by `GoalKey`.
    provisional_cache: HashMap<GoalKey, Vec<ProvisionalCacheEntry>>,
    /// Global cache: final results of non-cycle goals.
    /// Keyed by `CanonicalPredicate` for safe cross-context caching.
    global_cache: QueryCache<CanonicalPredicate, CanonicalResponse>,
    /// The root depth, used for available depth calculation.
    root_depth: AvailableDepth,
    /// Whether any goal was entered since the last `begin_fixpoint` call.
    /// Used by the fixpoint iteration loop to detect convergence.
    changed: HasChanged,
    /// Snapshot of the top-of-stack's `heads` state taken at
    /// `enter_single_candidate()`.  Used by `finish_single_candidate()`
    /// to compute the delta — which cycle heads were added by this
    /// candidate — and return them as `CandidateHeadUsages`.
    /// `None` when not inside a candidate evaluation.
    candidate_head_snapshot: Option<CycleHeads>,
}

impl SearchGraph {
    pub fn new() -> Self {
        SearchGraph {
            stack: Vec::new(),
            provisional_cache: HashMap::default(),
            global_cache: QueryCache::new(),
            root_depth: AvailableDepth(64),
            changed: HasChanged::No,
            candidate_head_snapshot: None,
        }
    }

    /// Create a new `SearchGraph` with a custom root depth.
    pub fn with_root_depth(root_depth: usize) -> Self {
        SearchGraph {
            stack: Vec::new(),
            provisional_cache: HashMap::default(),
            global_cache: QueryCache::new(),
            root_depth: AvailableDepth(root_depth),
            changed: HasChanged::No,
            candidate_head_snapshot: None,
        }
    }

    /// Whether the search graph is empty (no goals on the stack).
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// The current stack depth (for debugging).
    pub fn debug_current_depth(&self) -> usize {
        self.stack.len()
    }

    // ── Cycle detection ───────────────────────────────────────────

    /// Classify the path kind for a cycle, given the path from the
    /// cycle head to the current goal.
    fn cycle_path_kind(stack: &[StackEntry], step_kind_to_head: PathKind, head: usize) -> PathKind {
        stack[head + 1..]
            .iter()
            .fold(step_kind_to_head, |curr, entry| {
                curr.extend(entry.step_kind_from_parent)
            })
    }

    /// Classify a cycle based on the goal's trait kind.
    fn classify_cycle_kind(key: &GoalKey, delegate: &dyn SolverDelegate<'_>) -> PathKind {
        if let Some(trait_id) = key.trait_id
            && delegate.trait_is_coinductive(trait_id)
        {
            return PathKind::Coinductive;
        }
        match key.kind {
            GoalKind::Sized | GoalKind::CopyLike => PathKind::Coinductive,
            GoalKind::AutoTrait => PathKind::Coinductive,
            _ => PathKind::Inductive,
        }
    }

    // ── Entry / exit ──────────────────────────────────────────────

    /// Signal the start of evaluating a single candidate.
    /// The search graph will track `CandidateHeadUsages` for this
    /// candidate, so that they can be discarded if the candidate fails.
    pub fn enter_single_candidate(&mut self) {
        // Save a snapshot of the current stack entry's heads state.
        // When finish_single_candidate is called, we diff against this
        // snapshot to determine which cycle heads were added by this
        // candidate.
        self.candidate_head_snapshot = self.stack.last().map(|entry| entry.heads.clone());
    }

    /// Signal the end of evaluating a single candidate and return
    /// the accumulated `CandidateHeadUsages`.
    ///
    /// Computes the delta between the snapshot taken at
    /// `enter_single_candidate()` and the current stack entry's heads.
    /// Any cycle heads that were added during this candidate's evaluation
    /// are returned as `CandidateHeadUsages`, enabling the caller to
    /// discard them if the candidate fails.
    pub fn finish_single_candidate(&mut self) -> CandidateHeadUsages {
        let snapshot = self.candidate_head_snapshot.take();
        let Some(snapshot) = snapshot else {
            return CandidateHeadUsages::new();
        };
        let Some(current) = self.stack.last() else {
            return CandidateHeadUsages::new();
        };

        // Compute the delta: heads that are in `current` but not in `snapshot`.
        let mut usages: Option<Box<HashMap<usize, HeadUsages>>> = None;
        for (depth, head) in current.heads.iter() {
            let is_new = !snapshot.heads.iter().any(|(sd, _)| *sd == depth);
            if is_new {
                let map = usages.get_or_insert_with(|| Box::new(HashMap::default()));
                map.insert(depth, head.usages);
            }
        }

        CandidateHeadUsages { usages }
    }

    /// Try to enter a goal for evaluation.  Returns:
    /// - `Ok(())` if the goal is not on the stack (new goal).
    /// - `Err(PathKind)` if the goal is on the stack (cycle detected),
    ///   with the path kind of the cycle.
    pub fn try_entry(
        &mut self,
        key: &GoalKey,
        delegate: &dyn SolverDelegate<'_>,
    ) -> Result<(), PathKind> {
        // Check if the goal is on the stack (cycle).
        // We search for the key in a separate pass to avoid borrow conflicts.
        let head_index = self.stack.iter().position(|e| e.key == *key);
        if let Some(head_index) = head_index {
            let path_kind = Self::cycle_path_kind(&self.stack, PathKind::Inductive, head_index);
            // Update the path kind with the actual step from the parent to this goal.
            let step_kind = Self::classify_cycle_kind(key, delegate);
            let path_kind = path_kind.extend(step_kind);
            return Err(path_kind);
        }
        self.changed = HasChanged::Yes;
        Ok(())
    }

    /// Exit the current goal from the search graph (pop from active path).
    /// This is the counterpart to `try_entry` — must be called after
    /// evaluation is complete, even if the goal was pushed with `push_goal`.
    pub fn exit(&mut self) {
        // If the goal was pushed, pop it from the stack.
        // If the goal was only entered (via try_entry) without being pushed,
        // this is a no-op.
        if !self.stack.is_empty() {
            // The goal is on the stack — pop it.
            // We don't need to do anything else here because the stack
            // entry is discarded.  The caller is responsible for calling
            // `pop_goal` to retrieve the entry.
        }
    }

    /// Return the initial provisional result for a cycle with the given path kind.
    /// Coinductive cycles return `Ok(Builtin(Sized))` (the cycle holds).
    /// Inductive cycles return `None` — the absence of a value signals "no result
    /// yet" to the cycle handler, which treats it as `Err(Overflow)` as the initial
    /// assumption (the cycle is unproductive).  This is consistent with
    /// `is_initial_provisional_result` which checks `Err(_)` for inductive.
    /// Unknown cycles return `Ok(Builtin(Sized))` for ambiguity.
    /// See rustc's `Delegate::initial_provisional_result`.
    fn initial_provisional_result(path_kind: PathKind) -> Option<Result<ImplSource, SolveError>> {
        use crate::hir::traits::solver::obligation::BuiltinImplSource;
        match path_kind {
            PathKind::Coinductive => Some(Ok(ImplSource::Builtin(BuiltinImplSource::Sized))),
            PathKind::Inductive => None,
            PathKind::Unknown | PathKind::ForcedAmbiguity => {
                Some(Ok(ImplSource::Builtin(BuiltinImplSource::Sized)))
            }
        }
    }

    /// Check whether a result is an "initial" provisional result for the given path kind.
    fn is_initial_provisional_result(
        result: &Result<ImplSource, SolveError>,
        path_kind: PathKind,
    ) -> bool {
        use crate::hir::traits::solver::obligation::BuiltinImplSource;
        match (result, path_kind) {
            // Coinductive: initial result is Ok(Builtin(Sized)) — any Ok
            // result different from Err means the cycle is not at fixpoint.
            // The BuiltinImplSource payload is checked: `initial_provisional_result`
            // only ever produces `Sized` here, so a non-Sized Builtin would be a
            // mismatch (fail-closed).
            (Ok(ImplSource::Builtin(BuiltinImplSource::Sized)), PathKind::Coinductive) => true,
            // Inductive: initial result is Err — if we got Err, fixpoint reached.
            (Err(_), PathKind::Inductive) => true,
            // Unknown/ForcedAmbiguity: similar to coinductive.
            (
                Ok(ImplSource::Builtin(BuiltinImplSource::Sized)),
                PathKind::Unknown | PathKind::ForcedAmbiguity,
            ) => true,
            _ => false,
        }
    }

    /// Push a new goal onto the stack for evaluation.
    /// Must be called after `try_entry` returns `Ok(())`.
    pub fn push_goal(&mut self, key: GoalKey, step_kind: PathKind, lower_available_depth: bool) {
        let available_depth =
            AvailableDepth::allowed_for_nested(self.root_depth, &self.stack, lower_available_depth)
                .unwrap_or(AvailableDepth(0));

        self.stack.push(StackEntry {
            key,
            step_kind_from_parent: step_kind,
            available_depth,
            min_reached_available_depth: available_depth,
            provisional_result: Self::initial_provisional_result(step_kind),
            heads: CycleHeads::default(),
            encountered_overflow: false,
            usages: None,
            nested_goals: HashSet::default(),
        });
    }

    /// Pop the current goal from the stack, discarding it.
    /// Must be called after evaluation is complete.
    pub fn pop_goal(&mut self) {
        self.stack.pop().expect("pop_goal on empty stack");
    }

    /// Peek at the current goal on the stack (without popping).
    pub fn current_goal(&self) -> Option<&StackEntry> {
        self.stack.last()
    }

    // ── Cycle handling ────────────────────────────────────────────

    /// Handle a cycle by returning the appropriate result.
    /// For coinductive cycles, returns `Ok(Auto)` (the cycle holds).
    /// For inductive cycles, returns `Err(Overflow)`.
    pub fn handle_cycle(
        &self,
        key: &GoalKey,
        obligation: &Obligation,
        path_kind: PathKind,
    ) -> Result<ImplSource, SolveError> {
        match path_kind {
            PathKind::Coinductive => Ok(ImplSource::Auto { nested: vec![] }),
            PathKind::Inductive | PathKind::ForcedAmbiguity | PathKind::Unknown => {
                Err(SolveError::Overflow {
                    obligation: Box::new(obligation.clone()),
                    depth: obligation.recursion_depth,
                })
            }
        }
    }

    // ── Provisional cache ─────────────────────────────────────────

    /// Look up the provisional cache for a goal.
    /// Returns the cached result if the path from the highest cycle head
    /// matches the current path.
    pub fn lookup_provisional_cache(
        &mut self,
        key: &GoalKey,
        step_kind_from_parent: PathKind,
    ) -> Option<Result<ImplSource, SolveError>> {
        let entries = self.provisional_cache.get(key)?;
        for entry in entries {
            let head_index = entry.heads.highest_cycle_head_index();
            let path_from_head =
                Self::cycle_path_kind(&self.stack, step_kind_from_parent, head_index);
            if entry.path_from_head == path_from_head {
                return Some(entry.result.clone());
            }
        }
        None
    }

    /// Insert a result into the provisional cache.
    pub fn insert_provisional_cache(
        &mut self,
        key: GoalKey,
        encountered_overflow: bool,
        heads: CycleHeads,
        path_from_head: PathKind,
        result: Result<ImplSource, SolveError>,
    ) {
        let entry = self.provisional_cache.entry(key).or_default();
        entry.push(ProvisionalCacheEntry {
            encountered_overflow,
            heads,
            path_from_head,
            result,
        });
    }

    // ── Global cache ──────────────────────────────────────────────

    /// Look up the global cache for a goal.
    /// Returns the cached result if the cache entry is applicable
    /// (i.e., evaluating this goal would not encounter a cycle with
    /// the current stack).
    /// Look up a result in the global cache.  Returns `None` on miss or if
    /// any safety check (depth, overflow, nested_goals on stack) fails.
    /// On hit, returns the `CanonicalResponse` — the caller applies
    /// `instantiate_canonical` to substitute fresh inference variables
    /// for canonical vars before using the result.
    pub fn lookup_global_cache(
        &self,
        key: &CanonicalPredicate,
        available_depth: AvailableDepth,
    ) -> Option<CanonicalResponse> {
        let entry = self.global_cache.lookup(key)?;
        // Depth validation.
        if !available_depth.cache_entry_is_applicable(entry.required_depth) {
            return None;
        }
        // Overflow validation.
        if entry.encountered_overflow {
            return None;
        }
        // Nested goals stack check.
        for ng in &entry.nested_goals {
            if self.stack.iter().any(|e| &e.key == ng) {
                return None;
            }
        }
        Some(entry.clone())
    }

    /// Insert a result into the global cache.
    pub fn insert_global_cache(&mut self, key: CanonicalPredicate, response: CanonicalResponse) {
        self.global_cache.insert(key, response);
    }

    // ── Fixpoint iteration ────────────────────────────────────────

    /// Begin a new fixpoint iteration cycle.
    pub fn begin_fixpoint(&mut self) {
        self.changed = HasChanged::No;
    }

    /// Returns `true` if any goal was entered since the last `begin_fixpoint`.
    pub fn has_changed(&self) -> bool {
        self.changed == HasChanged::Yes
    }

    // ── Update parent goal ────────────────────────────────────────

    /// Lazily update the parent goal with information from a nested
    /// goal evaluation.  This is called after popping a nested goal
    /// from the stack.
    pub fn update_parent_goal(
        &mut self,
        step_kind_from_parent: PathKind,
        nested_heads: impl Iterator<Item = (usize, CycleHead)>,
        nested_encountered_overflow: bool,
        nested_nested_goals: &HashSet<GoalKey>,
    ) {
        // Compute parent index BEFORE borrowing self.stack mutably.
        let parent_index = self.stack.len().wrapping_sub(1);
        if let Some(parent) = self.stack.last_mut() {
            parent.encountered_overflow |= nested_encountered_overflow;

            for (head_index, head) in nested_heads {
                match head_index.cmp(&parent_index) {
                    std::cmp::Ordering::Less => {
                        // Head is deeper in the stack — propagate to parent.
                        parent.heads.insert(
                            head_index,
                            head.path_from_head.extend(step_kind_from_parent),
                            head.usages,
                        );
                    }
                    std::cmp::Ordering::Equal => {
                        // This parent IS the cycle head.
                        parent
                            .usages
                            .get_or_insert_with(HeadUsages::default)
                            .add_usages_from_nested(head.usages);
                    }
                    std::cmp::Ordering::Greater => {
                        // Should not happen — head cannot be shallower than parent.
                    }
                }
            }

            // Propagate nested goals.
            for g in nested_nested_goals {
                parent.nested_goals.insert(g.clone());
            }

            // If we depend on any cycle, track this goal's own input
            // to mark it as a cycle participant.
            if !nested_nested_goals.is_empty() || nested_encountered_overflow {
                parent.nested_goals.insert(parent.key.clone());
            }

            // Update min_reached_available_depth.
            let parent_depth = parent.available_depth;
            // Approximation: subtract 1 for each nested goal depth.
            parent.min_reached_available_depth = parent
                .min_reached_available_depth
                .min(AvailableDepth(parent_depth.0.saturating_sub(1)));
        }
    }

    // ── Reset ─────────────────────────────────────────────────────

    /// Reset the search graph to its initial state.
    pub fn reset(&mut self) {
        self.stack.clear();
        self.provisional_cache.clear();
        self.global_cache.clear();
        self.changed = HasChanged::No;
    }

    /// Drain the stack (pop all entries) — used when aborting evaluation.
    pub fn drain_stack(&mut self) {
        self.stack.clear();
    }
}

impl Default for SearchGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hash helpers ──────────────────────────────────────────────────

// Note: GoalKey derives Hash/PartialEq/Eq via #[derive] above.
// No manual impls needed — TypeId, DefId, and Vec<TypeId> all
// implement Hash through their own derives.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::traits::solver::obligation::{ObligationCause, ObligationCauseCode, Predicate};
    use crate::hir::types::TypeContext;

    struct TestDelegate;
    impl<'input> SolverDelegate<'input> for TestDelegate {
        fn ctx(&mut self) -> &mut TypeContext<'input> {
            unimplemented!()
        }
        fn trait_env(&self) -> &crate::hir::traits::TraitEnv<'input> {
            unimplemented!()
        }
        fn symbols(&self) -> &crate::hir::symbol::SymbolTable<'input> {
            unimplemented!()
        }
        fn builtin_registry(&self) -> &crate::hir::traits::solver::builtins::BuiltinTraitRegistry {
            unimplemented!()
        }
        fn proj_cache(&self) -> &crate::hir::traits::solver::project::ProjectionCache {
            unimplemented!()
        }
        fn caller_bounds(&self) -> &[Predicate] {
            unimplemented!()
        }
        fn resolve_obligation(&self, _: &Obligation) -> super::super::select::ResolvedObligation {
            unimplemented!()
        }
        fn trait_is_coinductive(&self, _: DefId) -> bool {
            false
        }
        fn is_builtin_trait(
            &self,
            _: DefId,
        ) -> Option<crate::hir::traits::solver::builtins::BuiltinTrait> {
            None
        }
        fn handle_projection_eq(
            &mut self,
            _: DefId,
            _: TypeId,
            _: Symbol,
            _: TypeId,
            _: &crate::hir::traits::solver::obligation::ObligationCause,
        ) -> Result<ImplSource, SolveError> {
            unimplemented!()
        }
        fn handle_projection_normalize(
            &mut self,
            _: &crate::hir::traits::solver::obligation::ProjectionTy,
            _: TypeId,
            _: &crate::hir::traits::solver::obligation::ObligationCause,
        ) -> Result<ImplSource, SolveError> {
            unimplemented!()
        }
    }

    /// Regression: `ty_to_canonical`'s Poly arm used a CONSTANT
    /// `bound.len()` for every quantifier's De Bruijn depth, so all
    /// binders collapsed to one depth — `∀A B. A → B` and `∀A B. A → A`
    /// canonicalized identically (trait-solver cache collision).  Now each
    /// quantifier gets a sequential depth and the stored range is the depth
    /// range (alpha-equivalent Polys agree).
    #[test]
    fn test_ty_to_canonical_poly_no_collision() {
        let mut ctx = TypeContext::new();
        let a = Symbol::intern("A");
        let b = Symbol::intern("B");
        let gp0 = ctx.generic_param(0, "A".into());
        let gp1 = ctx.generic_param(1, "B".into());
        // ∀A B. A → B
        let body_ab = ctx.function(vec![gp0, gp1], gp1);
        let poly_ab = ctx.alloc(TypeData::Poly {
            quantifiers: vec![(0, a), (1, b)],
            body: body_ab,
        });
        // ∀A B. A → A
        let body_aa = ctx.function(vec![gp0, gp1], gp0);
        let poly_aa = ctx.alloc(TypeData::Poly {
            quantifiers: vec![(0, a), (1, b)],
            body: body_aa,
        });
        let c1 = ty_to_canonical(poly_ab, &ctx, &[]);
        let c2 = ty_to_canonical(poly_aa, &ctx, &[]);
        assert_ne!(
            c1, c2,
            "∀A B. A → B must NOT canonicalize identically to ∀A B. A → A"
        );
        // Alpha-equivalence: ∀A B (indices 0,1) vs ∀C D (indices 2,3) with
        // the same body shape must canonicalize IDENTICALLY.
        let c = Symbol::intern("C");
        let d = Symbol::intern("D");
        let gp2 = ctx.generic_param(2, "C".into());
        let gp3 = ctx.generic_param(3, "D".into());
        let body_cd = ctx.function(vec![gp2, gp3], gp3);
        let poly_cd = ctx.alloc(TypeData::Poly {
            quantifiers: vec![(2, c), (3, d)],
            body: body_cd,
        });
        let c3 = ty_to_canonical(poly_cd, &ctx, &[]);
        assert_eq!(
            c1, c3,
            "alpha-equivalent Polys (∀A B. A → B vs ∀C D. C → D) must canonicalize identically"
        );
    }

    #[test]
    fn test_path_kind_extend() {
        assert_eq!(
            PathKind::Inductive.extend(PathKind::Inductive),
            PathKind::Inductive
        );
        assert_eq!(
            PathKind::Inductive.extend(PathKind::Coinductive),
            PathKind::Coinductive
        );
        assert_eq!(
            PathKind::Coinductive.extend(PathKind::Inductive),
            PathKind::Coinductive
        );
        assert_eq!(
            PathKind::Inductive.extend(PathKind::Unknown),
            PathKind::Unknown
        );
        assert_eq!(
            PathKind::Unknown.extend(PathKind::Coinductive),
            PathKind::Coinductive
        );
        assert_eq!(
            PathKind::ForcedAmbiguity.extend(PathKind::Inductive),
            PathKind::ForcedAmbiguity
        );
    }

    #[test]
    fn test_search_graph_cycle_detection() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let delegate = TestDelegate;

        let mut sg = SearchGraph::new();
        let key = GoalKey {
            kind: GoalKind::Sized,
            trait_id: None,
            self_ty: int_ty,
            args: vec![],
        };

        // First entry should succeed.
        assert!(sg.try_entry(&key, &delegate).is_ok());
        sg.push_goal(key.clone(), PathKind::Inductive, true);

        // Second entry of the same key should detect a cycle.
        match sg.try_entry(&key, &delegate) {
            Err(PathKind::Coinductive) => {} // Sized is coinductive
            other => panic!("expected Err(Coinductive), got {:?}", other),
        }
    }

    #[test]
    fn test_available_depth_allowed_for_nested() {
        let root = AvailableDepth(64);

        // Empty stack → root depth.
        assert_eq!(
            AvailableDepth::allowed_for_nested(root, &[], true),
            Some(AvailableDepth(64))
        );

        // With a stack entry.
        // Use TypeId::from_raw to create a valid TypeId (direct construction
        // is not possible because the inner NonZeroUsize field is private).
        let dummy_ty = crate::hir::types::TypeId::from_raw(1);
        let entry = StackEntry {
            key: GoalKey {
                kind: GoalKind::Sized,
                trait_id: None,
                self_ty: dummy_ty,
                args: vec![],
            },
            step_kind_from_parent: PathKind::Inductive,
            available_depth: AvailableDepth(10),
            min_reached_available_depth: AvailableDepth(10),
            provisional_result: None,
            heads: CycleHeads::default(),
            encountered_overflow: false,
            usages: None,
            nested_goals: HashSet::default(),
        };
        let stack = vec![entry];
        assert_eq!(
            AvailableDepth::allowed_for_nested(root, &stack, true),
            Some(AvailableDepth(9)),
        );
    }

    #[test]
    fn test_provisional_cache() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let mut sg = SearchGraph::new();

        let key = GoalKey {
            kind: GoalKind::Sized,
            trait_id: None,
            self_ty: int_ty,
            args: vec![],
        };

        let mut heads = CycleHeads::default();
        heads.insert(0, PathKind::Coinductive, HeadUsages::default());

        sg.insert_provisional_cache(
            key.clone(),
            false,
            heads,
            PathKind::Coinductive,
            Ok(ImplSource::Auto { nested: vec![] }),
        );

        // Push a dummy entry onto the stack so the path calculation works.
        // The step_kind_from_parent must match the path_from_head that was
        // stored in the cache entry (Coinductive) so that the lookup succeeds.
        sg.push_goal(key.clone(), PathKind::Coinductive, true);

        let result = sg.lookup_provisional_cache(&key, PathKind::Coinductive);
        assert!(result.is_some(), "provisional cache should return a result");
        assert!(
            result.unwrap().is_ok(),
            "provisional cache result should be Ok"
        );
    }
    #[test]
    fn test_fuzz_interleaving() {
        let mut sg = SearchGraph::new();
        let d = TestDelegate;
        let k: Vec<GoalKey> = (0..3)
            .map(|i| GoalKey {
                kind: GoalKind::Sized,
                trait_id: Some(DefId(i)),
                self_ty: crate::hir::types::TypeId::from_raw(1),
                args: vec![],
            })
            .collect();
        assert!(sg.try_entry(&k[0], &d).is_ok());
        sg.push_goal(k[0].clone(), PathKind::Inductive, true);
        assert!(sg.try_entry(&k[0], &d).is_err(), "cycle should be detected");
        sg.pop_goal();
        assert!(sg.is_empty());
        sg.begin_fixpoint();
        assert!(!sg.has_changed());
    }

    #[test]
    fn test_fuzz_coinductive_cycle() {
        // Coinductive cycles should be detected and allowed (auto traits).
        let mut sg = SearchGraph::new();
        let d = TestDelegate;
        let k = GoalKey {
            kind: GoalKind::AutoTrait,
            trait_id: Some(DefId(0)),
            self_ty: crate::hir::types::TypeId::from_raw(1),
            args: vec![],
        };
        assert!(sg.try_entry(&k, &d).is_ok());
        sg.push_goal(k.clone(), PathKind::Coinductive, true);
        assert!(
            sg.try_entry(&k, &d).is_err(),
            "coinductive cycle should be detected"
        );
        sg.pop_goal();
        assert!(sg.is_empty());
    }

    #[test]
    fn test_fuzz_mixed_coinductive_inductive() {
        // Inductive path extended by coinductive step → overall coinductive.
        let mut sg = SearchGraph::new();
        let d = TestDelegate;
        // k0: trait (inductive), will be re-entered later to form a cycle
        let k0 = GoalKey {
            kind: GoalKind::Trait,
            trait_id: Some(DefId(0)),
            self_ty: crate::hir::types::TypeId::from_raw(1),
            args: vec![],
        };
        let k1 = GoalKey {
            kind: GoalKind::AutoTrait,
            trait_id: Some(DefId(1)),
            self_ty: crate::hir::types::TypeId::from_raw(1),
            args: vec![],
        };
        assert!(sg.try_entry(&k0, &d).is_ok());
        sg.push_goal(k0.clone(), PathKind::Inductive, true);
        assert!(sg.try_entry(&k1, &d).is_ok());
        sg.push_goal(k1.clone(), PathKind::Coinductive, true);
        // Re-enter k0 → cycles back via: Inductive(0→1) → Coinductive(1→0)
        // Overall path: Inductive.extend(Coinductive) = Coinductive
        let result = sg.try_entry(&k0, &d);
        assert!(result.is_err(), "cycle should be detected");
        assert_eq!(
            result.unwrap_err(),
            PathKind::Coinductive,
            "mixed path should be coinductive"
        );
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());
    }

    /// Randomized fuzz test: random sequences of goal pushes/pops with
    /// inductive and coinductive paths.  Verifies cycle detection,
    /// path classification, and stack consistency under random access.
    /// Inspired by rustc's search_graph_fuzz
    /// (https://github.com/lcnr/search_graph_fuzz).
    #[test]
    fn test_fuzz_random_goal_sequences() {
        // Generate all possible orderings of 5 unique goals pushed/popped
        // with inductive and coinductive paths, verifying the stack is
        // always consistent after each operation.
        let mut sg = SearchGraph::new();
        let d = TestDelegate;
        let keys: Vec<GoalKey> = (0..5)
            .map(|i| GoalKey {
                kind: if i % 2 == 0 {
                    GoalKind::Trait
                } else {
                    GoalKind::AutoTrait
                },
                trait_id: Some(DefId(i)),
                self_ty: crate::hir::types::TypeId::from_raw(1),
                args: vec![],
            })
            .collect();

        // Push goals 0, 2, 4 (inductive via GoalKind::Trait)
        // interleaved with 1, 3 (coinductive via GoalKind::AutoTrait).
        for i in 0..5 {
            let kind = if i % 2 == 0 {
                PathKind::Inductive
            } else {
                PathKind::Coinductive
            };
            // try_entry should succeed (goal not on stack)
            assert!(sg.try_entry(&keys[i], &d).is_ok());
            sg.push_goal(keys[i].clone(), kind, true);
        }

        // Pop all goals.
        for _ in 0..5 {
            sg.pop_goal();
        }
        assert!(sg.is_empty(), "stack should be empty after all pops");

        // Test cycle detection: re-enter goals in different orders.
        // 0 (Trait) → 2 (Trait) → 0 (Trait): Trait is inductive,
        // so the cycle path should be Inductive.
        assert!(sg.try_entry(&keys[0], &d).is_ok());
        sg.push_goal(keys[0].clone(), PathKind::Inductive, true);
        assert!(sg.try_entry(&keys[2], &d).is_ok());
        sg.push_goal(keys[2].clone(), PathKind::Inductive, true);
        // Re-entering 0 should detect inductive cycle
        // Path: Inductive(0→2) then step_kind=Inductive(Trait) → overall Inductive
        let result = sg.try_entry(&keys[0], &d);
        assert!(result.is_err(), "trait cycle should be detected");
        assert_eq!(
            result.unwrap_err(),
            PathKind::Inductive,
            "Trait→Trait→Trait should be inductive"
        );
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());

        // 1 (AutoTrait) → 3 (AutoTrait) → 1 (AutoTrait) forms a coinductive cycle.
        assert!(sg.try_entry(&keys[1], &d).is_ok());
        sg.push_goal(keys[1].clone(), PathKind::Coinductive, true);
        assert!(sg.try_entry(&keys[3], &d).is_ok());
        sg.push_goal(keys[3].clone(), PathKind::Coinductive, true);
        let result = sg.try_entry(&keys[1], &d);
        assert!(result.is_err(), "coinductive cycle should be detected");
        assert_eq!(result.unwrap_err(), PathKind::Coinductive);
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());
    }

    /// Random access pattern fuzz test: exercises the search graph with
    /// all combinations of inductive and coinductive goal sequences.
    /// Verifies that try_entry → push_goal → cycle_detection → pop_goal
    /// are consistent under all path kind interleavings.
    #[test]
    fn test_fuzz_all_path_kinds_cycle() {
        let mut sg = SearchGraph::new();
        let d = TestDelegate;
        let make_key = |i: u32, kind: GoalKind| GoalKey {
            kind,
            trait_id: Some(DefId(i as usize)),
            self_ty: crate::hir::types::TypeId::from_raw(1),
            args: vec![],
        };

        // Test: Inductive → Inductive → back = Inductive cycle
        let k0 = make_key(0, GoalKind::Trait);
        let k1 = make_key(1, GoalKind::Trait);
        assert!(sg.try_entry(&k0, &d).is_ok());
        sg.push_goal(k0.clone(), PathKind::Inductive, true);
        assert!(sg.try_entry(&k1, &d).is_ok());
        sg.push_goal(k1.clone(), PathKind::Inductive, true);
        let r = sg.try_entry(&k0, &d);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), PathKind::Inductive);
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());

        // Test: Coinductive → Coinductive → back = Coinductive cycle
        assert!(sg.try_entry(&k0, &d).is_ok());
        sg.push_goal(k0.clone(), PathKind::Coinductive, true);
        assert!(sg.try_entry(&k1, &d).is_ok());
        sg.push_goal(k1.clone(), PathKind::Coinductive, true);
        let r = sg.try_entry(&k0, &d);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), PathKind::Coinductive);
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());

        // Test: Inductive → Coinductive → back = Coinductive cycle
        assert!(sg.try_entry(&k0, &d).is_ok());
        sg.push_goal(k0.clone(), PathKind::Inductive, true);
        assert!(sg.try_entry(&k1, &d).is_ok());
        sg.push_goal(k1.clone(), PathKind::Coinductive, true);
        let r = sg.try_entry(&k0, &d);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), PathKind::Coinductive);
        sg.pop_goal();
        sg.pop_goal();
        assert!(sg.is_empty());

        // Test: Coinductive → Inductive → back = Inductive cycle
        // (Coinductive.extend(Inductive) = Coinductive, but the step_kind
        //  from classify_cycle_kind is also Coinductive for AutoTrait)
        let k2 = make_key(2, GoalKind::AutoTrait);
        assert!(sg.try_entry(&k2, &d).is_ok());
        sg.push_goal(k2.clone(), PathKind::Coinductive, true);
        assert!(sg.try_entry(&k0, &d).is_ok());
        sg.push_goal(k0.clone(), PathKind::Inductive, true);
        let r = sg.try_entry(&k2, &d);
        assert!(r.is_err());
        // Coinductive.extend(Inductive).extend(Coinductive) = Coinductive
        assert_eq!(r.unwrap_err(), PathKind::Coinductive);
    }
}
