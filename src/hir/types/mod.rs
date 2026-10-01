use parking_lot::RwLock;
use rustc_hash::FxHashMap as HashMap;
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

pub(crate) use smallvec::{SmallVec, smallvec};

use crate::ast::OverflowPolicy;
use crate::symbol::Symbol;

// ── TypeFlags bitfield ────────────────────────────────────────────────
bitflags::bitflags! {
    /// Compact bitfield caching computed type properties for O(1) lookup.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TypeFlags: u8 {
        const IS_INFER_VAR = 0b0000_0001;
        const IS_CONCRETE = 0b0000_0010;
        const HAS_REGION_VAR = 0b0000_0100;
        const IS_FORALL = 0b0000_1000;
        const IS_FUNCTION = 0b0001_0000;
        const IS_STRUCT = 0b0010_0000;
        const IS_GENERIC_PARAM = 0b0100_0000;
    }
}

/// Global atomic counter for DefId allocation.
/// Used by `SymbolTable::allocate_def_id()` to ensure globally unique DefId
/// values across all SymbolTable<'input> instances.  Without this, each SymbolTable<'input>
/// starts its own counter from 0, causing DefId collisions between different
/// traits/types in different symbol tables (e.g., test isolation breaks).
/// The counter starts at 1 because `DefId(0)` is reserved as the sentinel
/// for the local crate (see `alloc_def_id`) — values begin after it.
static NEXT_DEF_ID: AtomicUsize = AtomicUsize::new(1);

/// Allocate a globally unique DefId.
/// DefId(0) is reserved as a sentinel for the local crate.
pub fn alloc_def_id() -> DefId {
    DefId(NEXT_DEF_ID.fetch_add(1, AtomicOrdering::Relaxed))
}

/// Reset the global DefId allocator back to its initial state.
/// This is intended for test use only — calling it outside of tests
/// would break DefId uniqueness guarantees across the entire program.
#[cfg(test)]
pub fn reset_def_id_allocator() {
    NEXT_DEF_ID.store(1, AtomicOrdering::Relaxed);
}

/// Posita language edition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Edition {
    Year2024,
    Year2026,
}

impl Edition {
    /// Parse an edition from a string like "2024" or "2026".
    /// Returns `None` for unknown edition strings.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "2024" => Some(Edition::Year2024),
            "2026" => Some(Edition::Year2026),
            _ => None,
        }
    }

    /// The latest (current) edition.
    pub const fn latest() -> Self {
        Edition::Year2026
    }
}

mod data;
pub use data::*;
mod coercion;
pub use coercion::*;
mod gadt;
pub use gadt::*;
mod variance;
pub use variance::*;
mod transaction;
pub use transaction::*;
mod canonical;
mod coherence;
pub mod predicativity;
mod rewrite;
pub mod witness;
mod yoneda;
pub use yoneda::*;
mod subtype;
pub use subtype::*;
mod unify;
pub use unify::*;
mod kappa;
pub use kappa::*;

/// Visit every TypeId child of a resolved type by calling `f` on each one.
/// This is the SINGLE source of truth for which TypeData variants contain
/// TypeId children.  When a new TypeData variant with TypeId fields is added,
/// update this function — all downstream walkers (ty_has_unresolved_vars,
/// collect_from_ty, type_is_volatile, etc.) automatically benefit.
/// See rustc's `TypeSuperFoldable` for the analogous pattern.
///
/// Lives at the `types` layer (not the traits solver) because it is a
/// pure TypeData structural walker — the traits solver, inference, and
/// the checker all consume it.  (Previously hosted in
/// `traits::solver::search_graph`, which forced `types` → `traits`
/// dependencies for consumers of `ty_contains_foreign_universe`.)
pub fn visit_type_children<'input, F: FnMut(TypeId)>(
    ty: TypeId,
    ctx: &TypeContext<'input>,
    f: &mut F,
) {
    let resolved = ctx.resolve_binding(ty);
    match ctx.get(resolved) {
        TypeData::Adt { args, .. } => {
            for a in args {
                f(*a);
            }
        }
        TypeData::Tuple { elems } => {
            for e in elems {
                f(*e);
            }
        }
        TypeData::Ref { ty, .. } => f(*ty),
        TypeData::Fn { params, ret, .. } => {
            for p in params {
                f(*p);
            }
            f(*ret);
        }
        TypeData::Array { elem, .. } => f(*elem),
        TypeData::Slice { elem } => f(*elem),
        TypeData::Pointer { ty } => f(*ty),
        TypeData::Ptr { size, pointee } => {
            f(*size);
            f(*pointee);
        }
        TypeData::AssociatedType { self_ty, .. } => f(*self_ty),
        TypeData::Opaque {
            hidden: Some(h), ..
        } => f(*h),
        TypeData::Coproduct { alternatives } => {
            for a in alternatives {
                f(*a);
            }
        }
        TypeData::Forall { body, .. } | TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
            f(*body)
        }
        TypeData::Exists { base, .. } => f(*base),
        TypeData::Poly { body, .. } => f(*body),
        // Leaf types have no TypeId children.
        _ => {}
    }
}

/// The universe-escape check: binding a variable to
/// a type that CONTAINS an InferVar from a DIFFERENT universe (directly or
/// nested inside a composite type) lets a forall-introduced variable
/// escape.  Recurses through composite types via the shared walker.
///
/// Lives at the `types` layer (not inference) because the checker-global
/// unify path (`types::TypeContext`) uses it; moving it here removes the
/// `types` → `infer` dependency inversion.
pub(crate) fn ty_contains_foreign_universe<'input>(
    ctx: &TypeContext<'input>,
    ty: TypeId,
    uni: usize,
) -> bool {
    match ctx.get(ctx.resolve_binding(ty)) {
        TypeData::InferVar { universe, .. } => *universe > uni,
        _ => {
            let mut found = false;
            visit_type_children(ty, ctx, &mut |c| {
                if ty_contains_foreign_universe(ctx, c, uni) {
                    found = true;
                }
            });
            found
        }
    }
}

/// The signature facts (the polonius_int.dl placeholder inputs).
///
/// Lives at the `types` layer (not the borrow checker) because it is a
/// pure data contract shared by `cfg_graph` (the CFG collector) and
/// `polonius` (the rules engine) — hosting it in either would force a
/// module-level cycle between them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SignatureFacts {
    /// `universal_region(origin)`: the input borrow origins are
    /// placeholders (the placeholder machinery is adopted — rejection
    /// of undeclared placeholder subsets via R9).
    pub universal_region: Vec<u32>,
    /// `known_placeholder_subset(input, output)`: the DECLARED A(ρ)
    /// constraint — the output borrow alive ⟹ the input borrow considered
    /// alive (the call-site instantiation turns this into per-point
    /// `subset_base` facts).
    pub known_placeholder_subset: Vec<(u32, u32)>,
    /// The MUTABILITY of each input borrow (in input order) — the
    /// call-site cross-function loans use it to issue `Exclusive` only for mutable
    /// inputs (`&mut`); read-only inputs (`&T`/`&ro`) get `ReadOnly`
    /// (previously every return-borrow was `Exclusive`).
    pub input_borrow_mutable: Vec<bool>,
}

fn is_type_volatile_inner(data: &TypeData, types: Option<&boxcar::Vec<Arc<TypeData>>>) -> bool {
    match data {
        TypeData::InferVar { .. } | TypeData::SkolemVar { .. } => true,
        TypeData::GenericParam { .. }
        | TypeData::Int { .. }
        | TypeData::UInt { .. }
        | TypeData::Float { .. }
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit
        | TypeData::Error
        | TypeData::Type
        | TypeData::Rational { .. }
        | TypeData::Regex { .. }
        | TypeData::Opaque { hidden: None, .. } => false,
        TypeData::Opaque {
            hidden: Some(ty), ..
        } => is_type_volatile_inner_by_id(*ty, types),
        TypeData::Adt { args, .. } => args.iter().any(|a| is_type_volatile_inner_by_id(*a, types)),
        TypeData::Tuple { elems } => elems
            .iter()
            .any(|e| is_type_volatile_inner_by_id(*e, types)),
        TypeData::Array { elem, .. } => is_type_volatile_inner_by_id(*elem, types),
        TypeData::Slice { elem } => is_type_volatile_inner_by_id(*elem, types),
        TypeData::Ref { ty, .. } => is_type_volatile_inner_by_id(*ty, types),
        TypeData::Pointer { ty } => is_type_volatile_inner_by_id(*ty, types),
        TypeData::Ptr { size, pointee } => {
            is_type_volatile_inner_by_id(*size, types)
                || is_type_volatile_inner_by_id(*pointee, types)
        }
        TypeData::Fn { params, ret } => {
            params
                .iter()
                .any(|p| is_type_volatile_inner_by_id(*p, types))
                || is_type_volatile_inner_by_id(*ret, types)
        }
        TypeData::DynTrait { .. } => false,
        TypeData::Exists { base, .. } => is_type_volatile_inner_by_id(*base, types),
        TypeData::Forall { body, .. } => is_type_volatile_inner_by_id(*body, types),
        TypeData::Poly { body, .. } => is_type_volatile_inner_by_id(*body, types),
        TypeData::Coproduct { alternatives } => alternatives
            .iter()
            .any(|a| is_type_volatile_inner_by_id(*a, types)),
        TypeData::Mu { body, .. } => is_type_volatile_inner_by_id(*body, types),
        TypeData::Nu { body, .. } => is_type_volatile_inner_by_id(*body, types),
        TypeData::AssociatedType { .. } => false,
    }
}

fn is_type_volatile_inner_by_id(ty: TypeId, types: Option<&boxcar::Vec<Arc<TypeData>>>) -> bool {
    // If we have access to the type arena, look up the TypeData and check.
    if let Some(types) = types {
        let idx = ty.index();
        if let Some(slot) = types.get(idx) {
            // Mirror the match in is_type_volatile_inner — handle every variant
            // explicitly so that the fallthrough below is only for the `None` case.
            return match &**slot {
                // Volatile: cannot cache.
                TypeData::InferVar { .. } | TypeData::SkolemVar { .. } => true,
                // Leaf / primitive types: never volatile.
                TypeData::GenericParam { .. }
                | TypeData::Int { .. }
                | TypeData::UInt { .. }
                | TypeData::Float { .. }
                | TypeData::Bool
                | TypeData::Char
                | TypeData::Byte
                | TypeData::USize
                | TypeData::Never
                | TypeData::Unit
                | TypeData::Error
                | TypeData::Type
                | TypeData::Rational { .. }
                | TypeData::Regex { .. }
                | TypeData::Opaque { hidden: None, .. } => false,
                // Opaque type with a revealed hidden type — recurse.
                TypeData::Opaque {
                    hidden: Some(ty), ..
                } => is_type_volatile_inner_by_id(*ty, Some(types)),
                TypeData::DynTrait { .. } => false,
                TypeData::AssociatedType { .. } => false,
                // Composite types: recurse into children.
                TypeData::Adt { args, .. } => args
                    .iter()
                    .any(|a| is_type_volatile_inner_by_id(*a, Some(types))),
                TypeData::Tuple { elems } => elems
                    .iter()
                    .any(|e| is_type_volatile_inner_by_id(*e, Some(types))),
                TypeData::Array { elem, .. } => is_type_volatile_inner_by_id(*elem, Some(types)),
                TypeData::Slice { elem } => is_type_volatile_inner_by_id(*elem, Some(types)),
                TypeData::Ref { ty, .. } => is_type_volatile_inner_by_id(*ty, Some(types)),
                TypeData::Pointer { ty } => is_type_volatile_inner_by_id(*ty, Some(types)),
                TypeData::Ptr { size, pointee } => {
                    is_type_volatile_inner_by_id(*size, Some(types))
                        || is_type_volatile_inner_by_id(*pointee, Some(types))
                }
                TypeData::Fn { params, ret } => {
                    params
                        .iter()
                        .any(|p| is_type_volatile_inner_by_id(*p, Some(types)))
                        || is_type_volatile_inner_by_id(*ret, Some(types))
                }
                TypeData::Exists { base, .. } => is_type_volatile_inner_by_id(*base, Some(types)),
                TypeData::Forall { body, .. } => is_type_volatile_inner_by_id(*body, Some(types)),
                TypeData::Poly { body, .. } => is_type_volatile_inner_by_id(*body, Some(types)),
                TypeData::Coproduct { alternatives } => alternatives
                    .iter()
                    .any(|a| is_type_volatile_inner_by_id(*a, Some(types))),
                TypeData::Mu { body, .. } => is_type_volatile_inner_by_id(*body, Some(types)),
                TypeData::Nu { body, .. } => is_type_volatile_inner_by_id(*body, Some(types)),
            };
        } else {
            // The TypeId points to an index beyond the current arena.
            // This can happen when `data` is externally constructed and
            // its TypeId children haven't been allocated yet.  We cannot
            // determine volatility without the referenced data, so be
            // conservative: treat it as volatile (uncacheable).
            return true;
        }
    }
    // Without the arena, conservatively assume volatile (uncacheable).
    true
}

/// A place (variable / field / index / deref chain) whose storage can be
/// borrowed.  Statement-level loan tracking: the freeze scope of a borrow
/// is the enclosing lexical block, and the frozen entity is the exact
/// place, not merely the root variable.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) enum FrozenPlace {
    Root(Symbol),
    Field(Box<FrozenPlace>, Symbol),
    /// Dynamic index (`a[i]`): the index VALUE is not statically known, so
    /// every element is treated as potentially touched — two dynamic
    /// indexes on the same base are indistinguishable (conservative).
    Index(Box<FrozenPlace>),
    /// Constant index (`a[0]`, `a[3]`): the offset is statically known, so
    /// `a[0]` and `a[1]` are distinct places — freezing `a[0]` does NOT
    /// freeze `a[1]` (mirrors rustc's `ProjectionElem::ConstantIndex`).
    ConstIndex(Box<FrozenPlace>, u64),
    Deref(Box<FrozenPlace>),
}

/// The kind of a borrow, which determines how strictly the source place is
/// frozen (SYNTAX.md §References / §Reference Coercion).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LoanKind {
    /// `&ro expr` / `expr.freeze!()` — read-only borrow: the source place is
    /// frozen against MUTATION for the borrow's lifetime.
    ReadOnly,
    /// `&mut expr` — exclusive borrow: the source place is frozen — neither
    /// readable nor writable — for the borrow's lifetime.  The read side is
    /// enforced by the flow-sensitive borrow-check post-pass (E109).
    Exclusive,
}

impl LoanKind {
    /// The borrow-syntax family name (the borrow diagnostics use it to
    /// name the mechanism; the key is `Hash`-able for the error dedup).
    pub fn as_str(&self) -> &'static str {
        match self {
            LoanKind::ReadOnly => "ReadOnly",
            LoanKind::Exclusive => "Exclusive",
        }
    }
}

/// The registered definition of an ADT (struct/enum), used by
/// `type_is_copy` to decide the §Copy derivation (all fields recursively
/// Copy + no Drop impl).
#[derive(Debug, Clone)]
pub(crate) struct AdtDef {
    pub fields: Vec<TypeId>,
    pub has_drop: bool,
}

/// The PER-QUERY half of the type system.
///
/// `TypeContext<'input>` owns the mutable inference state — bindings,
/// transactions, GADT refinements, and the caches that depend on them.
/// The type arena and the structural caches live in the shared
/// [`TypeFactory`] behind an `Arc`, so several contexts can exist at once
/// over one interner and still agree on `TypeId`s.  That is what lets a
/// per-function query take its own context: inference for two functions
/// never touches the same bindings, but both see the same types.
///
/// Note that `get()` reads `bindings` (via `resolve_binding`), so *all*
/// type inspection depends on this half.  Parallelism therefore comes from
/// one context per query, not from sharing one context between queries.
pub struct TypeContext<'input> {
    /// The parser's arena — set in the compiler driver (main.rs) so the
    /// checker can allocate NEW AST nodes (e.g. the `type T = Base where
    /// value > 0` desugar builds `invariant _where_N > 0`).  `None` in
    /// unit tests / target-layout tests that don't desugar.
    /// Wrapped in `SyncArena`, not a bare `&Bump`: `bumpalo::Bump` is not
    /// `Sync`, and `&T: Send`/`&T: Sync` both require `T: Sync`, so a bare
    /// `&Bump` here made `TypeContext` neither `Send` nor `Sync`.  See
    /// `crate::arena` for why the wrapper is sound.
    pub arena: Option<&'input crate::arena::SyncArena<'input>>,
    /// The SHARED interner: the type arena plus every cache that does not
    /// depend on inference state.  Behind an `Arc` so a per-query context
    /// can share it instead of copying it.
    interner: Arc<TypeFactory<'input>>,
    pub(crate) bindings: HashMap<TypeId, TypeId>,
    pub builtin_unit: TypeId,
    pub builtin_never: TypeId,
    pub builtin_error: TypeId,
    pub builtin_bool: TypeId,
    pub builtin_char: TypeId,
    pub builtin_byte: TypeId,
    pub builtin_usize: TypeId,
    /// Built-in string slice type `Str`.
    pub builtin_str: TypeId,
    /// Built-in reference to string slice `&Str` — a `Ref { ty: Str, mutable: false }`.
    pub builtin_str_ref: TypeId,
    /// `Type` — a dedicated struct-like type
    /// for comptime layout descriptors, replacing the old `ctx.error()` fallback.
    pub builtin_layout_descriptor: TypeId,
    /// Region-subtype collection switch + collected outlives pairs.
    /// When the checker enables region collection (per function
    /// signature), `subtype`'s Ref arm records `(l1, l2)` for
    /// `&'a T <: &'b T` (rustc's `make_subregion(b, a)` — the covariance
    /// constraint `'a: 'b`) instead of rejecting; the solver consumes the
    /// collected pairs.  When disabled (pure-relation calls, tests), the
    /// old strict rejection stands.
    pub(crate) region_subtype_collect: bool,
    pub(crate) region_subtype_outlives: Vec<(Symbol, Symbol)>,
    /// Compilation-unit parametricity state (roadmap Stage 4): set by the
    /// checker when a parametricity-breaking construct (`dyn Trait` type
    /// reification) is lowered.  Consulted by Yoneda-reduction-as-
    /// equivalence consumers via `equiv_yoneda_allowed`; κ diagnostics are
    /// NOT gated.  See `hir::parametricity` module docs.
    pub parametricity: crate::hir::parametricity::ParametricityGate,
    /// When an opaque type's concrete type is inferred inside the defining
    /// scope, the hidden type is stored here. `resolve_opaque` checks this
    /// map before falling back to the `TypeData::Opaque.hidden` field.
    /// Uses `RefCell` because the defining scope may set hidden types
    /// during type checking after TypeContext<'input> creation.
    opaque_hidden: HashMap<TypeId, TypeId>,
    /// Transaction stack for atomic unification (OmniML-style rollback via undo log).
    /// Each entry is a list of (key, old_value) pairs recording every binding
    /// change made during that transaction.  On rollback the changes are undone
    /// in reverse order; on commit the log is discarded.
    /// This is O(changes) instead of O(total_bindings) — a significant saving
    /// when the binding table is large and transactions are frequent.
    /// Note: the types arena (self.types) is NOT truncated on rollback because
    /// TypeId values may be held externally; the arena is append-only.
    transaction_stack: Vec<Vec<(TypeId, Option<TypeId>)>>,
    /// Parallel undo log for `opaque_hidden` changes.  Each transaction
    /// level has a list of (opaque_type_id, previous_hidden_value) pairs
    /// that are restored on rollback.
    opaque_hidden_undo: Vec<Vec<(TypeId, Option<TypeId>)>>,
    /// Cache for unification with variance: prevents infinite recursion on
    /// self-referential types.  Keyed by (a, b, variance_tag) where
    /// variance_tag = 0 for Invariant, 1 for Covariant, 2 for Contravariant.
    unify_seen: HashSet<(TypeId, TypeId, u8)>,
    /// Current "operation span" hint: set by `unify_tracked` so that
    /// `set_binding` can record WHERE a GenericParam got bound (precise
    /// E104 error location).  `None` outside span-carrying unifications,
    /// so solver-driven bindings do not inherit a stale span.
    pub(crate) current_unify_span: Option<crate::ast::Span>,
    /// The current unification's coercion context: `@auto_ro`'s
    /// `&mut T → &T` relaxation applies ONLY at function call sites
    /// (SYNTAX.md) — set to `CallSite` while checking call arguments, and
    /// `Structural` elsewhere (array/ADT elements, struct fields, ...).
    pub(crate) current_coercion_ctx: std::sync::atomic::AtomicU8,
    /// Count of seal-the-wall violations (GADT refinements attempting to
    /// NEWLY bind a GenericParam into the global table).  Incremented in
    /// ALL builds — the binding is skipped, so the global table is never
    /// polluted; the counter keeps the violation observable.
    pub(crate) seal_violations: usize,
    /// Whether the function currently being checked has `@auto_ro`
    /// (SYNTAX.md §Local Relaxation): allows `&mut T` to be implicitly
    /// coerced to `&T` within the function body.  Set/restored around
    /// each function body check (save/restore, so nested functions nest).
    pub(crate) auto_ro: bool,
    /// Whether the function currently being checked has `@auto_coerce`
    /// (SYNTAX.md §Local Relaxation): enables ALL safe implicit
    /// coercions (`&mut T` → `&T` like `@auto_ro`, plus deref coercions).
    /// Set/restored around the function body; forbidden in `@trusted`
    /// functions and Strict Mode.
    pub(crate) auto_coerce: bool,
    /// Binding origins for GenericParams: param TypeId → span of the
    /// unifying operation that bound it.  Consulted by the E104
    /// generality check to point at the precise binding site instead of
    /// the whole function definition.
    pub(crate) generic_binding_origins: HashMap<TypeId, crate::ast::Span>,
    /// Cache for κ(A) characteristic results.  Cleared when bindings change.
    kappa_cache: HashMap<TypeId, Characteristic>,
    /// Cached type property flags for O(1) lookup on hot paths.
    type_flags: HashMap<TypeId, TypeFlags>,
    /// Language edition for this compilation unit.
    edition: Edition,
    /// Target platform information (arch, ABI, sizes, alignments).
    pub target: crate::hir::target::Target,
    /// GADT equality registry — stack of per-arm equality lists.
    ///
    /// Each entry records `(from_type, to_type)` meaning "within the
    /// current arm, `from_type` is equivalent to `to_type`".
    ///
    /// This is the OCaml approach (see `ctype.ml:3926-3949`): instead of
    /// calling `set_binding` (which modifies global state and requires
    /// transaction/rollback), GADT arm processing registers equalities
    /// here.  `resolve_binding` consults this registry after following
    /// the normal binding chain, making GADT refinements transparently
    /// visible to all type operations within the arm.
    /// Scoped GADT facts for the current arm, split into two kinds:
    /// `ParamRefinement` (visible to `resolve_binding`) and
    /// `ExistentialEquation` (inert — never used as a rewrite rule, so an
    /// existential witness stays opaque even inside compound types).
    /// All GADT-related state (fact registry + depth counter + existential
    /// scope stack + pending inner equalities) aggregated into one
    /// structure; see `GadtContext`.
    /// Binder scope stack: tracks GenericParam indices currently under a
    /// Forall/Exists/Mu/Nu/Poly binder during unification.  Used to detect
    /// scope escape — a GenericParam that is bound in an active binder must
    /// not be unified with a foreign type (which would leak the quantified
    /// variable into the surrounding context).
    pub(crate) binder_stack: Vec<usize>,
    pub(crate) gadt: GadtContext<'input>,
}

impl<'input> TypeContext<'input> {
    /// Register an ADT definition (struct/enum) — used by `type_is_copy`
    /// to decide the §Copy derivation.
    pub(crate) fn register_adt(&self, def_id: DefId, def: AdtDef) {
        self.interner.adt_defs.write().insert(def_id, def);
    }

    /// Look up a registered ADT definition.
    pub(crate) fn adt_def(&self, def_id: DefId) -> Option<AdtDef> {
        self.interner.adt_defs.read().get(&def_id).cloned()
    }

    /// Mark an ADT as implementing `Drop` (an `impl Drop for T` exists) —
    /// the §Copy derivation must exclude it.
    pub(crate) fn set_adt_has_drop(&self, def_id: DefId) {
        if let Some(mut def) = self.adt_def(def_id) {
            def.has_drop = true;
            self.register_adt(def_id, def);
        }
    }

    pub fn new() -> Self {
        Self::new_with_target(crate::hir::target::Target::host())
    }

    pub fn new_with_target(target: crate::hir::target::Target) -> Self {
        let mut ctx = TypeContext {
            arena: None,
            interner: Arc::new(TypeFactory::new()),
            bindings: HashMap::default(),
            builtin_unit: TypeId::NONE,
            builtin_never: TypeId::NONE,
            builtin_error: TypeId::NONE,
            builtin_bool: TypeId::NONE,
            builtin_char: TypeId::NONE,
            builtin_byte: TypeId::NONE,
            builtin_usize: TypeId::NONE,
            builtin_str: TypeId::NONE,
            builtin_str_ref: TypeId::NONE,
            builtin_layout_descriptor: TypeId::NONE,
            region_subtype_collect: false,
            region_subtype_outlives: Vec::new(),
            parametricity: crate::hir::parametricity::ParametricityGate::new(),
            opaque_hidden: HashMap::default(),
            opaque_hidden_undo: Vec::new(),
            transaction_stack: Vec::new(),
            unify_seen: HashSet::default(),
            current_unify_span: None,
            current_coercion_ctx: std::sync::atomic::AtomicU8::new(
                CoercionContext::Structural.to_u8(),
            ),
            seal_violations: 0,
            auto_ro: false,
            auto_coerce: false,
            generic_binding_origins: HashMap::default(),
            kappa_cache: HashMap::default(),
            type_flags: HashMap::default(),
            edition: Edition::latest(),
            target,
            binder_stack: Vec::new(),
            gadt: GadtContext::new(),
        };
        ctx.builtin_unit = ctx.alloc(TypeData::Unit);
        ctx.builtin_never = ctx.alloc(TypeData::Never);
        ctx.builtin_error = ctx.alloc(TypeData::Error);
        ctx.builtin_bool = ctx.alloc(TypeData::Bool);
        ctx.builtin_char = ctx.alloc(TypeData::Char);
        ctx.builtin_byte = ctx.alloc(TypeData::Byte);
        ctx.builtin_usize = ctx.alloc(TypeData::USize);
        // Str type: represented as a zero-sized struct with a sentinel DefId.
        ctx.builtin_str = ctx.alloc(TypeData::Adt {
            kind: AdtKind::Struct,
            def_id: DefId::SENTINEL_STR,
            args: smallvec![],
        });
        // &Str = Ref { ty: Str, mutable: false }
        ctx.builtin_str_ref = ctx.reference(ctx.builtin_str, false);
        // LayoutDescriptor type for layout_of! results.
        ctx.builtin_layout_descriptor = ctx.alloc(TypeData::Adt {
            kind: AdtKind::Struct,
            def_id: DefId::SENTINEL_LAYOUT_DESC,
            args: smallvec![],
        });
        ctx
    }

    /// Build a context that shares `other`'s interner but starts with fresh
    /// inference state.
    ///
    /// This is the per-query constructor.  The new context sees exactly the
    /// same type arena, the same `meta`/`def_id_to_type_id`/`adt_defs`
    /// registrations and the same builtin `TypeId`s — so `TypeId`s minted
    /// before the split stay meaningful and comparable — but it gets its own
    /// bindings, transaction stack, GADT state and binding-dependent caches.
    ///
    /// Two contexts built this way can therefore be used concurrently: each
    /// mutates only its own inference state, and the shared half is
    /// internally synchronised.
    ///
    /// The `arena` and `edition`/`target` are copied, not shared: the arena
    /// is already a `Sync` view of the parser's bump, and the other two are
    /// plain values.
    pub fn new_sharing(other: &Self) -> Self {
        TypeContext {
            arena: other.arena,
            interner: Arc::clone(&other.interner),
            bindings: HashMap::default(),
            builtin_unit: other.builtin_unit,
            builtin_never: other.builtin_never,
            builtin_error: other.builtin_error,
            builtin_bool: other.builtin_bool,
            builtin_char: other.builtin_char,
            builtin_byte: other.builtin_byte,
            builtin_usize: other.builtin_usize,
            builtin_str: other.builtin_str,
            builtin_str_ref: other.builtin_str_ref,
            builtin_layout_descriptor: other.builtin_layout_descriptor,
            region_subtype_collect: false,
            region_subtype_outlives: Vec::new(),
            parametricity: crate::hir::parametricity::ParametricityGate::new(),
            opaque_hidden: HashMap::default(),
            opaque_hidden_undo: Vec::new(),
            transaction_stack: Vec::new(),
            unify_seen: HashSet::default(),
            current_unify_span: None,
            current_coercion_ctx: std::sync::atomic::AtomicU8::new(
                CoercionContext::Structural.to_u8(),
            ),
            seal_violations: 0,
            auto_ro: false,
            auto_coerce: false,
            generic_binding_origins: HashMap::default(),
            kappa_cache: HashMap::default(),
            type_flags: HashMap::default(),
            edition: other.edition,
            target: other.target.clone(),
            binder_stack: Vec::new(),
            gadt: GadtContext::new(),
        }
    }

    /// Returns the refinement invariant recorded for `id`, if any.
    ///
    /// Returns an owned `Expr` rather than a borrow: `meta` lives behind a
    /// lock now (it is shared state), so the reference would outlive the
    /// guard.  `Expr` is `Clone` and these are short AST fragments.
    pub fn get_invariant(&self, id: TypeId) -> Option<crate::ast::Expr<'input>> {
        self.interner
            .meta
            .read()
            .get(&id)
            .and_then(|m| m.invariant.as_ref().cloned())
    }

    /// Allocate a fresh, globally-unique parameter index for Exists/Forall binders.
    pub fn fresh_param_index(&self) -> usize {
        let idx = self.interner.next_param_index.load(AtomicOrdering::Relaxed);
        self.interner
            .next_param_index
            .store(idx + 1, AtomicOrdering::Relaxed);
        idx
    }

    /// Get the current language edition.
    pub fn edition(&self) -> Edition {
        self.edition
    }

    /// Set the language edition (e.g. from an `edition = "2026"` declaration).
    pub fn set_edition(&mut self, edition: Edition) {
        self.edition = edition;
    }

    /// Allocate a type.  Takes `&self`: the arena is append-only and
    /// lock-free (see the `boxcar::Vec` in `TypeFactory`), so interning
    /// does not need exclusive access.  That is what lets a shared
    /// `&TypeContext` still create types during checking.
    pub fn alloc(&self, data: TypeData) -> TypeId {
        self.interner.alloc(data).0
    }

    /// The shared interner.  Comptime code that needs to mint types with
    /// only `&self` uses this; every context built from the same session
    /// shares one, so the resulting `TypeId`s are directly comparable.
    pub fn factory(&self) -> &TypeFactory<'input> {
        &self.interner
    }

    /// Read a slot of the type arena by index.
    ///
    /// The arena is append-only, so a slot that `TypeId`s point at is
    /// always initialized — an out-of-range index means a `TypeId` was
    /// minted for a type that never made it into the shared interner.
    #[inline]
    pub(crate) fn slot(&self, index: usize) -> &Arc<TypeData> {
        self.interner.types.get(index).unwrap_or_else(|| {
            panic!(
                "type slot {} is not initialized (arena holds {}) — \
                 a TypeId was minted for a type that was never allocated",
                index,
                self.interner.types.count()
            )
        })
    }

    /// Check whether a `TypeData` transitively contains any volatile type
    /// (`InferVar` or `SkolemVar`).  Volatile types are scope-sensitive —
    /// they must not be interned because:
    ///
    /// - **InferVar**: the same id can be reused across inference scopes;
    ///   caching a composite that embeds an InferVar would let stale
    ///   bindings leak into the new scope.
    /// - **SkolemVar**: each `enter_universe()` call creates a fresh id;
    ///   caching composities that embed them would bloat the cache.
    ///   with single-use entries that never hit the cache.
    fn type_is_volatile(&self, data: &TypeData) -> bool {
        match data {
            TypeData::InferVar { .. } | TypeData::SkolemVar { .. } => true,
            // Leaf / primitive types are never volatile
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Unit
            | TypeData::Never
            | TypeData::Error
            | TypeData::Rational { .. }
            | TypeData::Regex { .. }
            | TypeData::GenericParam { .. }
            | TypeData::Type
            | TypeData::Opaque { hidden: None, .. } => false,
            // Opaque type with revealed hidden type — recurse.
            TypeData::Opaque {
                hidden: Some(ty), ..
            } => self.type_is_volatile_by_id(*ty),
            // Composite types: check each child TypeId
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.type_is_volatile_by_id(p))
                    || self.type_is_volatile_by_id(*ret)
            }
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.type_is_volatile_by_id(a)),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.type_is_volatile_by_id(e)),
            TypeData::Array { elem, .. } => self.type_is_volatile_by_id(*elem),
            TypeData::Slice { elem } => self.type_is_volatile_by_id(*elem),
            TypeData::Ref { ty, .. } => self.type_is_volatile_by_id(*ty),
            TypeData::Pointer { ty } => self.type_is_volatile_by_id(*ty),
            TypeData::Ptr { size, pointee } => {
                self.type_is_volatile_by_id(*size) || self.type_is_volatile_by_id(*pointee)
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => self.type_is_volatile_by_id(*body),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.type_is_volatile_by_id(a))
            }
            TypeData::AssociatedType { self_ty, .. } => self.type_is_volatile_by_id(*self_ty),
            TypeData::DynTrait { .. } => false,
        }
    }

    /// Look up a TypeId and check if its TypeData transitively contains
    /// a volatile type (InferVar or SkolemVar).
    /// Recursive helper for `type_is_volatile`.
    fn type_is_volatile_by_id(&self, id: TypeId) -> bool {
        // First check the RAW slot — if this TypeId was originally allocated as
        // an InferVar or SkolemVar, it IS volatile regardless of any bindings.
        // Using resolve_binding first would skip this check when a volatile type
        // happens to be bound to a concrete type, allowing composite types
        // containing the raw volatile TypeId to be incorrectly cached.
        if matches!(
            &**self.slot(id.index()),
            TypeData::InferVar { .. } | TypeData::SkolemVar { .. }
        ) {
            return true;
        }
        // Follow bindings and check the resolved type.
        // This catches the case where a non-volatile type (e.g. GenericParam)
        // has been bound to a volatile type.
        let resolved = self.resolve_binding(id);
        if resolved == id {
            return false;
        }
        self.type_is_volatile(&self.slot(resolved.index()))
    }

    /// The Copy determination (§Copy): the scalars are Copy; the function
    /// types and the String (an `Adt`) are not; the aggregates recurse.
    pub(crate) fn type_is_copy(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Rational { .. }
            // Unit (zero fields, no Drop) and Never (uninhabited) satisfy
            // the §Copy derivation rule vacuously — they must not fall
            // through to `_ => false` (an affine `()` would demand an
            // explicit `move`, contradicting SYNTAX.md §Value Semantics).
            | TypeData::Unit
            // Raw pointers (`Pointer`/`Ptr`) are Copy: a raw pointer is
            // essentially a usize-sized integer — it owns nothing and has
            // no `Drop`; copying an address has no memory cost and no
            // ownership consequence (committee ruling).  The pointee does
            // not matter — the pointer does not own it.
            | TypeData::Never
            | TypeData::Pointer { .. }
            | TypeData::Ptr { .. } => true,
            TypeData::Fn { .. } => false,
            TypeData::Tuple { elems } => elems.iter().all(|e| self.type_is_copy(*e)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem, .. } => self.type_is_copy(*elem),
            // The reference types: `&T` (immutable — e.g. `&Str`) is Copy;
            // `&mut T` is not (§References — exclusive, non-copyable).
            TypeData::Ref { mutable, .. } => !*mutable,
            // The §Copy derivation for ADTs (SYNTAX.md §Value Semantics):
            // no Drop impl + all fields recursively Copy — the definition
            // table is populated by the resolver (def_id → fields + Drop).
            TypeData::Adt { def_id, args, .. } => self.adt_is_copy(*def_id, args),
            // Fail closed: the remaining types
            // (Coproduct, Exists, Poly, DynTrait, Opaque, Mu, Nu,
            // AssociatedType, ...) default to non-Copy unless proven
            // bitwise-replicable — the `_ => true` fallthrough was the
            // wrong direction for an affine language.
            _ => false,
        }
    }

    /// The §Copy derivation for an ADT: no Drop impl + all fields
    /// recursively Copy.
    fn adt_is_copy(&self, def_id: DefId, args: &[TypeId]) -> bool {
        match self.adt_def(def_id) {
            Some(def) => {
                if def.has_drop {
                    return false;
                }
                def.fields.iter().all(|f| self.adt_field_is_copy(*f, args))
            }
            // Unregistered ADT (e.g. a builtin like `String`): conservative.
            None => false,
        }
    }

    /// A field of an ADT instantiation: a generic parameter is replaced by
    /// its argument (`Option<Int<32>>` — the field `T` → `Int<32>`); other
    /// fields recurse (nested ADTs carry their own args in their TypeData).
    fn adt_field_is_copy(&self, field: TypeId, args: &[TypeId]) -> bool {
        match self.get(field) {
            TypeData::GenericParam { index, .. } => {
                args.get(*index).is_some_and(|&a| self.type_is_copy(a))
            }
            _ => self.type_is_copy(field),
        }
    }

    /// Returns the resolved `TypeData` for a `TypeId`, following bindings.
    pub fn get(&self, id: TypeId) -> &TypeData {
        let resolved = self.resolve_binding(id);
        // Entry-existence, NOT `index < count()`: `boxcar::Vec::count()`
        // counts *completed* pushes, and completions happen in claim order
        // only by accident — a thread stalled between `next_index()` and its
        // final `count.fetch_add` leaves `count` below `max_index + 1` even
        // though every returned `TypeId`'s slot is fully written.  Under
        // concurrent `alloc` (rayon checker workers) an `index < count()`
        // assertion is therefore a false positive; `types.get(i).is_some()`
        // (the same check `slot()` makes) is the real invariant.
        debug_assert!(
            self.interner.types.get(resolved.index()).is_some(),
            "TypeContext::get() index {} is not an initialized arena slot \
             (types.len={}) — TypeId minted for a type that was never allocated",
            resolved.index(),
            self.interner.types.count(),
        );
        &self.slot(resolved.index())
    }

    /// Returns the raw `TypeData` for a `TypeId` WITHOUT following bindings.
    /// Unlike `get()`, this does not resolve inference variable bindings,
    /// so it can inspect the original type before any substitution.
    /// Useful for binder-scope checks where the raw type identity matters.
    pub fn get_raw(&self, id: TypeId) -> &TypeData {
        // See `get()`: entry-existence, never `index < count()` — `count()`
        // lags in-flight pushes and makes this a false positive under
        // concurrent `alloc`.
        debug_assert!(
            self.interner.types.get(id.index()).is_some(),
            "TypeContext::get_raw() index {} is not an initialized arena slot \
             (types.len={}) — TypeId minted for a type that was never allocated",
            id.index(),
            self.interner.types.count(),
        );
        &self.slot(id.index())
    }

    /// Returns an `Arc<TypeData>` instead of a borrow, enabling cheap clone via
    /// `Arc::clone` (reference-count bump only).  Use this instead of
    /// `self.get(ty).clone()` on hot paths (substitution, Yoneda reduction,
    /// unification) to avoid deep-copying `Vec<TypeId>` and `String` fields.
    pub fn get_arc(&self, id: TypeId) -> Arc<TypeData> {
        let resolved = self.resolve_binding(id);
        Arc::clone(&self.slot(resolved.index()))
    }

    pub fn is_infer_var(&self, id: TypeId) -> bool {
        matches!(self.get(id), TypeData::InferVar { .. })
    }

    /// Get cached type flags, computing lazily on first access.
    pub fn get_type_flags(&mut self, ty: TypeId) -> TypeFlags {
        if let Some(flags) = self.type_flags.get(&ty) {
            return *flags;
        }
        let flags = self.compute_type_flags(ty);
        self.type_flags.insert(ty, flags);
        flags
    }

    fn compute_type_flags(&self, ty: TypeId) -> TypeFlags {
        let mut flags = TypeFlags::empty();
        match self.get(ty) {
            TypeData::InferVar { .. } => flags |= TypeFlags::IS_INFER_VAR,
            _ => flags |= TypeFlags::IS_CONCRETE,
        }
        match self.get(ty) {
            TypeData::GenericParam { .. } => flags |= TypeFlags::IS_GENERIC_PARAM,
            TypeData::Fn { .. } => flags |= TypeFlags::IS_FUNCTION,
            TypeData::Forall { .. } => flags |= TypeFlags::IS_FORALL,
            TypeData::Adt {
                kind: AdtKind::Struct,
                ..
            } => flags |= TypeFlags::IS_STRUCT,
            _ => {}
        }
        flags
    }

    /// Check if the RAW TypeId slot (ignoring bindings) is an InferVar,
    /// and return its id if so.  Unlike `is_infer_var`, this does NOT
    /// follow `set_binding` chains, so it correctly identifies the
    /// original InferVar even after it has been unified with a concrete type.
    pub(crate) fn get_infer_var_id(&self, id: TypeId) -> Option<usize> {
        if let TypeData::InferVar { id: raw_id, .. } = &**self.slot(id.index()) {
            Some(*raw_id)
        } else {
            None
        }
    }

    /// If the type is an opaque TAIT (`type T = impl Trait`) outside its
    /// defining scope, resolve to the revealed concrete type.
    /// Uses an iterative loop with a visited set to handle cycles.
    /// Recursively follows chains of opaque indirections.
    pub fn resolve_opaque(&self, id: TypeId) -> TypeId {
        let mut current = self.resolve_binding(id);
        let mut visited = rustc_hash::FxHashSet::default();
        visited.insert(current);
        loop {
            // Resolve bindings at each hop — the hidden type may have been
            // unified with another type after being stored in opaque_hidden.
            current = self.resolve_binding(current);
            // Check the opaque_hidden map first (defining scope reveal).
            if let Some(hidden) = self.opaque_hidden.get(&current) {
                let hidden = self.resolve_binding(*hidden);
                if !visited.insert(hidden) {
                    return current; // cycle detected
                }
                current = hidden;
                continue;
            }
            match &**self.slot(current.index()) {
                TypeData::Opaque {
                    hidden: Some(hidden),
                    ..
                } => {
                    let hidden = self.resolve_binding(*hidden);
                    if !visited.insert(hidden) {
                        return current; // cycle detected
                    }
                    current = hidden;
                }
                _ => return current,
            }
        }
    }

    /// Set the hidden concrete type for an opaque type within its defining
    /// scope.  The type checker calls this when the concrete type is inferred.
    /// Records the previous value in the transaction undo log so that
    /// `rollback_transaction` can restore it.
    ///
    /// If the opaque type already has a hidden type, unifies the new value
    /// with the existing one instead of silently overwriting.  Returns an
    /// error if unification fails (conflicting defining uses).
    pub fn set_opaque_hidden(&mut self, id: TypeId, hidden: TypeId) -> Result<(), String> {
        // If there is an existing hidden type, unify rather than overwrite.
        // Copy out the existing value first to avoid borrow conflicts.
        let existing = self.opaque_hidden.get(&id).copied();
        if let Some(existing) = existing {
            self.unify(existing, hidden).map_err(|_| {
                format!(
                    "opaque type {:?} has conflicting hidden types: {:?} vs {:?}",
                    id, existing, hidden
                )
            })?;
            return Ok(());
        }
        // Record undo before inserting, if a transaction is active.
        if let Some(log) = self.opaque_hidden_undo.last_mut() {
            let old = self.opaque_hidden.get(&id).copied();
            log.push((id, old));
        }
        self.opaque_hidden.insert(id, hidden);
        Ok(())
    }

    /// Push a new GADT arm scope onto the equality registry stack.
    ///
    /// All GADT equalities registered via `register_gadt_eq` after this
    /// call are scoped to this arm and will be discarded on `pop_gadt_arm`.
    ///
    /// This replaces the old approach of `begin_transaction` + `try_unify`
    /// + `rollback_transaction` for GADT pattern matching.  Instead of
    /// writing type equalities into the global bindings table (which
    /// requires a transaction to undo), we record them in this scoped
    /// registry.  `resolve_binding` transparently consults the registry,
    /// so all type operations within the arm see the refined types.
    ///
    /// This is directly inspired by OCaml's `Pattern<'input>` unification mode
    /// (ctype.ml:446-459).  In OCaml, `link_type` (≈`set_binding`) is
    /// never called in `Pattern<'input>` mode; instead, constraint equations
    /// are collected in `equated_types`:
    ///
    /// ```ocaml
    /// (* ctype.ml:446-459 *)
    /// type unification_environment =
    ///   | Expression of { env : Env.t; in_subst : bool; }
    ///   | Pattern<'input> of
    ///       { penv : Pattern_env.t;
    ///         equated_types : TypePairs.t;  (* ← no link_type, collect here *)
    ///         assume_injective : bool;
    ///         unify_eq_set : TypePairs.t; }
    /// ```
    pub fn push_gadt_arm(&self) {
        self.gadt.enter_arm();
    }

    /// Pop the current GADT arm scope, discarding its equalities.
    ///
    /// After this call, `resolve_binding` no longer sees the
    /// GADT refinements from this arm.
    ///
    /// See `push_gadt_arm` for the OCaml `Pattern<'input>` mode reference
    /// (ctype.ml:3926-3936: `equated_types` discarded on scope exit).
    pub fn pop_gadt_arm(&self) {
        self.gadt.exit_arm();
    }

    /// Register a GADT **param refinement** within the current arm.
    ///
    /// `from` is a refinable variable (typically a `GenericParam` or a
    /// non-arm-local `InferVar`), and `to` is the concrete type from the
    /// `when` clause.  Within the arm, `resolve_binding(from)` returns `to`.
    /// Safe for `resolve_binding` to follow because `from` is a variable,
    /// not a closed type.
    pub fn register_param_refinement(&self, from: TypeId, to: TypeId) {
        self.gadt.register_param_refinement(from, to);
    }

    /// Register an **inert existential equation** within the current arm.
    ///
    /// `lhs` is a type containing an existential skolem (e.g. `[S]`), `rhs`
    /// is the concrete scrutinee side (e.g. `[Int<32>]`).  This fact is
    /// NEVER consulted by `resolve_binding`, so the existential witness
    /// stays opaque (SYNTAX.md §"Existential Quantification").
    pub fn register_existential_equation(&self, lhs: TypeId, rhs: TypeId) {
        self.gadt.register_existential_equation(lhs, rhs);
    }

    /// Register a GADT equality that holds within the current arm.
    ///
    /// `from` is the scrutinee's type argument (typically a `GenericParam`
    /// or `InferVar`), and `to` is the concrete type from the `when`
    /// clause.  Within the arm, `resolve_binding(from)` will return `to`.
    ///
    /// Multiple equalities can be registered per arm.  They are checked
    /// in reverse registration order (LIFO) during resolution.
    ///
    /// See `push_gadt_arm` for the OCaml equivalent (`record_equation`,
    /// ctype.ml:506-511).
    // (register_gadt_eq removed: it was an unclassified pub API that could
    // bypass the ParamRefinement/ExistentialEquation split.  Use
    // `register_param_refinement` or `register_existential_equation`.)

    /// Resolve a TypeId by following the binding chain, then checking
    /// the GADT equality registry.
    ///
    /// The GADT registry is consulted AFTER the binding chain: if the
    /// root of the binding chain appears as a key in the current arm's
    /// GADT equalities, the corresponding value is returned instead.
    /// This makes GADT type refinements transparently visible to all
    /// type operations (type checking, unification, etc.) without
    /// modifying the global bindings table.
    pub(crate) fn resolve_binding(&self, id: TypeId) -> TypeId {
        // Safety: guard against infinite loops from circular bindings.
        // 10 000 is generous enough for any real program while preventing
        // a maliciously constructed chain from DoS-ing the compiler.
        const MAX_CHAIN_DEPTH: usize = 10_000;

        // First pass: follow the binding chain to the root with a single
        // immutable borrow.  This is a simple linked-list traversal through
        // the bindings map until we reach an unbound TypeId.
        //
        // ALSO check the GADT equality registry at each step: if any node
        // along the chain (including the starting `id` and the final root)
        // appears as a key in the current arm's GADT equalities, the
        // mapping takes effect at that point.  We return the mapped value
        // directly (resolved through its own binding chain) WITHOUT
        // path-compressing it into the global bindings table, because the
        // GADT equality is scoped to the current arm and must not leak
        // after pop_gadt_arm().  See push_gadt_arm for the OCaml reference.
        let mut current = id;
        let mut depth = 0;
        loop {
            // Check GADT registry at each step before following bindings.
            if let Some(mapped) = self.resolve_gadt_eq(current) {
                return self.resolve_binding_tail(mapped);
            }
            let bindings = &self.bindings;
            if let Some(&next) = bindings.get(&current) {
                current = next;
                depth += 1;
                if depth > MAX_CHAIN_DEPTH {
                    break;
                }
            } else {
                break;
            }
        }
        // Also check the final node (the root of the binding chain).
        if let Some(mapped) = self.resolve_gadt_eq(current) {
            return self.resolve_binding_tail(mapped);
        }
        // NOTE: path compression is deliberately NOT performed here.
        //
        // It used to run inside this `&self` walk, mutating `bindings`
        // through a `RefCell`.  `bindings` is now a plain `HashMap` (the
        // interior mutability had to go for `TypeContext` to become
        // `Sync`), and compressing needs `&mut self` — which `get()` and
        // the other `&self` readers cannot supply.  Chains are already
        // bounded by `MAX_CHAIN_DEPTH` above, so dropping the memoization
        // costs a bounded amount of walking and changes no observable
        // result: compression was purely a cache of this same walk.
        //
        // `&mut self` receivers that walk a chain anyway can opt into the
        // compression via `resolve_binding_mut` / `resolve_binding_compressed`
        // below.

        current
    }

    /// Follow the binding chain from `id` to the root, checking the GADT
    /// registry at each step.  Used internally by `resolve_binding` to
    /// resolve the target of a GADT equality mapping.
    ///
    /// It DOES chase GADT registry entries so that transitive equalities
    /// like `A →[GADT] B →[GADT] C` resolve correctly.
    /// Follow ONLY the `bindings` chain to the root, NEVER consulting the
    /// GADT registry.  Used to canonicalize a refinement key to the chain
    /// root BEFORE registering a GADT equality, so path compression cannot
    /// hide the key from `resolve_binding` (which checks the GADT registry
    /// at each step of a possibly-compressed chain).
    pub(crate) fn resolve_binding_no_gadt(&self, id: TypeId) -> TypeId {
        const MAX_CHAIN_DEPTH: usize = 10_000;
        let mut current = id;
        let mut depth = 0;
        loop {
            let bindings = &self.bindings;
            match bindings.get(&current) {
                Some(&next) => {
                    current = next;
                    depth += 1;
                    if depth > MAX_CHAIN_DEPTH {
                        return current;
                    }
                }
                None => return current,
            }
        }
    }

    /// Walk `id`'s binding chain to its root, rewiring every intermediate
    /// node straight to that root, and return the root.
    ///
    /// Returns `None` when compression must not run.  The caller MUST then
    /// fall back to `resolve_binding`, which performs the identical walk —
    /// so the fallback is always observationally equal to this method.
    ///
    /// Correctness conditions for rewiring
    /// -----------------------------------
    /// `resolve_binding` consults the GADT equality registry at *every*
    /// node of a chain, not only the root.  Rewriting `a → b → c` to
    /// `a → c` makes any refinement keyed at `b` unreachable from `a`, and
    /// the result would silently change.  Two guards make it safe:
    ///
    /// * `arm_depth == 0`.  With no arm live, `gadt.param_refinements` is
    ///   empty (`exit_arm` pops the frame it pushed in `enter_arm`), so no
    ///   refinement can be hidden.  While an arm IS live we bail: a
    ///   refinement key canonicalised to the chain root at registration
    ///   time (`resolve_binding_no_gadt`) can still be rebound mid-arm —
    ///   the seal-the-wall guard only pins `GenericParam` keys, whereas
    ///   `InferVar` and GADT-`SkolemVar` keys are registered as
    ///   `ParamRefinement` too (`checker/mod.rs::register_gadt_eq_directional`)
    ///   and may become non-root intermediates.  Compressing through such
    ///   a node hides the refinement.
    /// * The registry must be *observed* empty.  Checked once here rather
    ///   than per node, so invariant drift degrades to "no compression"
    ///   instead of a wrong answer.
    ///
    /// Compression never changes what `resolve_binding` returns for any
    /// type — each rewired node still reaches the same root — so caches
    /// keyed on resolved types (`kappa_cache`, `type_flags`) stay valid,
    /// and no invalidation is needed.
    pub(crate) fn resolve_binding_mut(&mut self, id: TypeId) -> Option<TypeId> {
        if self.gadt.arm_depth.load(AtomicOrdering::Relaxed) > 0 {
            return None;
        }
        {
            let refs = self.gadt.param_refinements.read();
            if refs.iter().any(|arm| !arm.is_empty()) {
                return None;
            }
        }

        const MAX_CHAIN_DEPTH: usize = 10_000;

        // Inline capacity, not a heap path: this runs from `unify_internal`'s
        // recursion, so the common short-chain case must not allocate.
        let mut path: SmallVec<[TypeId; 4]> = SmallVec::new();
        let mut current = id;
        let mut depth = 0;
        loop {
            let next = {
                let bindings = &self.bindings;
                bindings.get(&current).copied()
            };
            match next {
                Some(next) => {
                    path.push(current);
                    current = next;
                    depth += 1;
                    if depth > MAX_CHAIN_DEPTH {
                        return None;
                    }
                }
                None => break,
            }
        }

        for node in path {
            self.compress_binding(node, current);
        }

        Some(current)
    }

    /// `resolve_binding` for `&mut self` receivers, compressing the chain
    /// when that is safe.  Returns exactly what `resolve_binding` returns,
    /// on every path.
    pub(crate) fn resolve_binding_compressed(&mut self, id: TypeId) -> TypeId {
        match self.resolve_binding_mut(id) {
            Some(root) => root,
            None => self.resolve_binding(id),
        }
    }

    /// Repoint `key` at `value`, recording the old value in the active
    /// transaction's undo log so `rollback_transaction` reconstructs the
    /// pre-compression chain.
    ///
    /// Deliberately NOT `set_binding`: compression must not touch
    /// `generic_binding_origins`.  That map records *binding sites* for
    /// the E104 diagnostic, and a compression write has no site of its
    /// own — recording `current_unify_span` there would relocate the
    /// error to whichever unify happened to walk the chain.  The
    /// seal-the-wall guard is also skipped, which is why callers must
    /// only reach this with `arm_depth == 0`.
    fn compress_binding(&mut self, key: TypeId, value: TypeId) {
        debug_assert_eq!(
            self.gadt.arm_depth.load(AtomicOrdering::Relaxed),
            0,
            "compress_binding reached while a GADT arm is active"
        );
        if self.bindings.get(&key) == Some(&value) {
            return;
        }
        if let Some(log) = self.transaction_stack.last_mut() {
            let old = self.bindings.get(&key).copied();
            log.push((key, old));
        }
        self.bindings.insert(key, value);
    }

    fn resolve_binding_tail(&self, id: TypeId) -> TypeId {
        const MAX_CHAIN_DEPTH: usize = 10_000;
        let mut current = id;
        let mut depth = 0;
        loop {
            // Follow bindings one step.
            let bound = {
                let bindings = &self.bindings;
                bindings.get(&current).copied()
            };
            if let Some(next) = bound {
                current = next;
                depth += 1;
                if depth > MAX_CHAIN_DEPTH {
                    break;
                }
                // After following a binding, also check GADT registry.
                if let Some(mapped) = self.resolve_gadt_eq(current) {
                    current = mapped;
                }
                continue;
            }
            // No binding found — check GADT registry before returning.
            if let Some(mapped) = self.resolve_gadt_eq(current) {
                current = mapped;
                depth += 1;
                if depth > MAX_CHAIN_DEPTH {
                    break;
                }
                continue;
            }
            break;
        }
        current
    }

    /// Look up `id` in the GADT fact registry, checking arms from
    /// innermost (current) to outermost.  Only `ParamRefinement` facts are
    /// consulted — `ExistentialEquation` facts are inert and never used as
    /// rewrite rules, so an existential witness stays opaque.
    /// Returns the mapped TypeId if found, or `None` if `id` is not
    /// registered as a param refinement.
    ///
    /// See `push_gadt_arm` (OCaml `equated_types` lookup, ctype.ml:3926-3936).
    fn resolve_gadt_eq(&self, id: TypeId) -> Option<TypeId> {
        let refs = self.gadt.param_refinements.read();
        for map in refs.iter().rev() {
            if let Some(&to) = map.get(&id) {
                return Some(to);
            }
        }
        None
    }

    /// Explicit witness-solving API (GHC-style coercion / OCaml GADT
    /// constraint mode): given a GADT existential skolem, return the
    /// concrete type it was solved to by a `when` constraint within the
    /// current arm, if any.  Opacity is the DEFAULT — `resolve_binding`
    /// NEVER follows existential equations; consumers that need the solved
    /// witness (payload type resolution, `'len` lookup) call this
    /// explicitly.  This is the single observation point for witness
    /// solving, so `resolve_binding` (transparent) and unification (rigid)
    /// cannot drift apart.
    /// Opt-in existential witness solving: consumers that legitimately need
    /// a solved witness (e.g. `'len`/projection rules) call this explicitly
    /// at the point of use — it is intentionally NOT part of
    /// `resolve_binding` (which must keep witnesses opaque).  The test
    /// consumer (`test_resolve_existential_witness_opt_in`) exercises the
    /// observation point; production consumers land with the `'len`/projection
    /// rules (future).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn resolve_existential_witness(&self, skolem: TypeId) -> Option<TypeId> {
        let facts = self.gadt.facts.read();
        for arm in facts.iter().rev() {
            for fact in arm.iter() {
                if let GadtFact::ExistentialEquation { lhs, rhs } = fact
                    && *lhs == skolem
                {
                    return Some(*rhs);
                }
            }
        }
        None
    }

    /// Resolve bindings AND resolve opaque types.
    /// This is the standard entry point for getting the "real" type.
    pub fn resolve(&self, id: TypeId) -> TypeId {
        self.resolve_opaque(self.resolve_binding(id))
    }

    pub fn alloc_infer_var(&mut self, id: usize, universe: usize) -> TypeId {
        self.alloc(TypeData::InferVar { id, universe })
    }

    /// Allocate a fresh inference variable from the TypeContext-LOCAL
    /// overlap counter (replaces the former process-global
    /// `OVERLAP_FRESH_VAR_ID`).  Per-context, so long-running processes
    /// and parallel tests never share a single unbounded counter.
    pub fn alloc_overlap_fresh_var(&mut self, universe: usize) -> TypeId {
        let id = self
            .interner
            .next_overlap_fresh_id
            .load(AtomicOrdering::Relaxed);
        self.interner
            .next_overlap_fresh_id
            .store(id + 1, AtomicOrdering::Relaxed);
        self.alloc(TypeData::InferVar { id, universe })
    }

    pub fn get_def_id_for_type(&self, id: TypeId) -> Option<DefId> {
        let resolved = self.resolve_binding(id);
        match &self.slot(resolved.index()).as_ref() {
            TypeData::Adt { def_id, .. } => Some(*def_id),
            TypeData::DynTrait { traits } => traits.first().copied(),
            _ => None,
        }
    }

    pub fn register_def_id(&mut self, def_id: DefId, type_id: TypeId) {
        self.interner
            .def_id_to_type_id
            .write()
            .insert(def_id, type_id);
    }

    pub fn get_type_id_for_def_id(&self, def_id: DefId) -> Option<TypeId> {
        self.interner.def_id_to_type_id.read().get(&def_id).copied()
    }

    pub fn int(&mut self, bits: u32, signed: bool) -> TypeId {
        debug_assert!(
            bits >= 1 && bits <= 64,
            "Int<{}> out of range (SYNTAX.md: bits 1..64)",
            bits,
        );
        self.alloc(TypeData::Int {
            bits,
            signed,
            overflow_policy: OverflowPolicy::Trap,
        })
    }

    pub fn uint(&mut self, bits: u32) -> TypeId {
        debug_assert!(
            bits >= 1 && bits <= 64,
            "UInt<{}> out of range (SYNTAX.md: bits 1..64)",
            bits,
        );
        self.alloc(TypeData::UInt {
            bits,
            overflow_policy: OverflowPolicy::Trap,
        })
    }

    /// Create an Int type with a specific overflow policy.
    pub fn int_with_overflow(&mut self, bits: u32, signed: bool, policy: OverflowPolicy) -> TypeId {
        self.alloc(TypeData::Int {
            bits,
            signed,
            overflow_policy: policy,
        })
    }

    /// Create a UInt type with a specific overflow policy.
    pub fn uint_with_overflow(&mut self, bits: u32, policy: OverflowPolicy) -> TypeId {
        self.alloc(TypeData::UInt {
            bits,
            overflow_policy: policy,
        })
    }

    /// Get the overflow policy for an integer type (defaults to Trap for non-integers).
    pub fn overflow_policy_of(&self, ty: TypeId) -> OverflowPolicy {
        match self.get(ty) {
            TypeData::Int {
                overflow_policy, ..
            }
            | TypeData::UInt {
                overflow_policy, ..
            } => *overflow_policy,
            _ => OverflowPolicy::Trap,
        }
    }

    pub fn float(&mut self, bits: u32) -> TypeId {
        self.alloc(TypeData::Float { bits })
    }

    pub fn bool(&self) -> TypeId {
        self.builtin_bool
    }

    /// Create a compile-time validated regex type.
    /// All `Regex` types must go through this constructor — it is the single
    /// chokepoint for pattern validation, covering parser, resolver, and
    /// any future deserialization path.  The parser also validates the
    /// pattern before emitting `Type::Regex`, so this `debug_assert!` is
    /// a safety net for debug builds only.
    pub fn regex(&mut self, pattern: Symbol) -> TypeId {
        let valid = *self
            .interner
            .regex_cache
            .write()
            .entry(pattern)
            .or_insert_with(|| regex_syntax::parse(pattern.as_str()).is_ok());
        debug_assert!(valid, "regex() called with an invalid pattern");
        self.alloc(TypeData::Regex { pattern })
    }

    pub fn char(&self) -> TypeId {
        self.builtin_char
    }

    pub fn byte(&self) -> TypeId {
        self.builtin_byte
    }

    pub fn usize(&self) -> TypeId {
        self.builtin_usize
    }

    pub fn str_ref(&self) -> TypeId {
        self.builtin_str_ref
    }

    pub fn unit(&self) -> TypeId {
        self.builtin_unit
    }

    pub fn never(&self) -> TypeId {
        self.builtin_never
    }

    pub fn error(&self) -> TypeId {
        self.builtin_error
    }

    pub fn struct_ty(&mut self, def_id: DefId, args: SmallVec<[TypeId; 4]>) -> TypeId {
        let id = self.alloc(TypeData::Adt {
            kind: AdtKind::Struct,
            def_id,
            args,
        });
        // Register the prototype (first instantiation) only.
        // Generic instances (e.g. Vec<i32>, Vec<bool>) share the same DefId
        // and must not overwrite the prototype — OmniML uses Constr + Ident
        // for constructor identity; our DefId is the analog of Ident.t.
        self.interner
            .def_id_to_type_id
            .write()
            .entry(def_id)
            .or_insert(id);
        id
    }

    pub fn enum_ty(&mut self, def_id: DefId, args: SmallVec<[TypeId; 4]>) -> TypeId {
        let id = self.alloc(TypeData::Adt {
            kind: AdtKind::Enum,
            def_id,
            args,
        });
        // Same as struct_ty — register prototype only, never overwrite.
        self.interner
            .def_id_to_type_id
            .write()
            .entry(def_id)
            .or_insert(id);
        id
    }

    pub fn tuple(&mut self, elems: SmallVec<[TypeId; 4]>) -> TypeId {
        self.alloc(TypeData::Tuple { elems })
    }

    /// Create a coproduct (sum type) Σ Aᵢ — "one of the alternatives".
    /// Used by Yoneda reduction to encode branch choice.
    pub fn coproduct(&mut self, alternatives: SmallVec<[TypeId; 4]>) -> TypeId {
        match alternatives.len() {
            0 => self.never(),
            1 => alternatives[0],
            _ => self.alloc(TypeData::Coproduct { alternatives }),
        }
    }

    pub fn array(&mut self, elem: TypeId, size: u64) -> TypeId {
        self.alloc(TypeData::Array { elem, size })
    }

    pub fn slice(&mut self, elem: TypeId) -> TypeId {
        self.alloc(TypeData::Slice { elem })
    }

    pub fn reference(&mut self, ty: TypeId, mutable: bool) -> TypeId {
        self.reference_with_lifetime(ty, mutable, None)
    }

    /// Construct a `&T` / `&mut T` reference, PRESERVING the explicit
    /// lifetime annotation (`&'a T` — `Some('a)`) when the caller threads
    /// it through — substitution / generic replacement must NOT silently
    /// erase user-written lifetimes (SYNTAX.md explicit lifetime rule).
    pub fn reference_with_lifetime(
        &mut self,
        ty: TypeId,
        mutable: bool,
        lifetime: Option<Symbol>,
    ) -> TypeId {
        self.alloc(TypeData::Ref {
            ty,
            mutable,
            lifetime,
        })
    }

    pub fn pointer(&mut self, ty: TypeId) -> TypeId {
        self.alloc(TypeData::Pointer { ty })
    }

    pub fn ptr(&mut self, size: TypeId, pointee: TypeId) -> TypeId {
        self.alloc(TypeData::Ptr { size, pointee })
    }

    pub fn function(&mut self, params: SmallVec<[TypeId; 4]>, ret: TypeId) -> TypeId {
        self.alloc(TypeData::Fn { params, ret })
    }

    /// Allocate a polytype `[∀ᾱ. τ]` — a boxed first-class polymorphic type.
    /// `quantifiers` are (index, name) pairs for universally quantified variables.
    /// `body` references them via `GenericParam`.
    pub fn poly(&mut self, quantifiers: SmallVec<[(usize, Symbol); 2]>, body: TypeId) -> TypeId {
        // Defense-in-depth: enforce the OmniML closedness invariant
        // (omniml/lib/constraint_solver/principal_shape.ml — `Poly.invariant`):
        // the polytope body must be closed — no `InferVar`, no nested `Poly`.
        // If a bug elsewhere creates an open body, catch it at the
        // construction site instead of silently propagating a stale
        // inference variable (which `replace_infer` would miss).
        //
        // This is a COMPILER-correctness invariant, not a user-code
        // constraint: a violation means a bug in the inference engine,
        // `replace_generic`, or the Yoneda reduction.  It FAILS CLOSED
        // (ICE) in BOTH debug and release builds — degrading to the
        // `Error` type in release would let the malformed polytope unify
        // with anything and silently cascade into type confusion.
        assert!(
            !self.poly_body_is_open(body, &quantifiers),
            "poly() invariant violated: polytope body must be closed (no InferVar, no nested Poly, no unbound GenericParam)"
        );
        self.alloc(TypeData::Poly { quantifiers, body })
    }

    /// Whether `ty` (a polytope body) violates the OmniML closedness
    /// invariant: it contains an `InferVar` or a nested `Poly`.
    /// See the `TypeData::Poly` doc comment for the invariant.
    fn poly_body_is_open(&self, ty: TypeId, quantifiers: &[(usize, Symbol)]) -> bool {
        self.poly_body_is_open_inner(ty, quantifiers, &mut Vec::new())
    }

    /// Recursive core of `poly_body_is_open` — threads the set of indices
    /// bound by INNER binders (`Forall`/`Exists`/`Mu`/`Nu`): a
    /// `GenericParam` shadowed by an inner binder is NOT a free variable
    /// of the outer `Poly`, so it must not flag the body as open.  The
    /// old implementation recursed with only the outer quantifiers and
    /// falsely rejected closed nested quantifiers; the binder stack
    /// (verified regression) fixes that.
    fn poly_body_is_open_inner(
        &self,
        ty: TypeId,
        quantifiers: &[(usize, Symbol)],
        locally_bound: &mut Vec<usize>,
    ) -> bool {
        match self.get(ty) {
            TypeData::InferVar { .. } => true,
            TypeData::Poly { .. } => true,
            // OmniML closedness (principal_shape.ml + OmniML.md): a polytope
            // annotation is closed iff every free type variable is bound by
            // its quantifiers.  A GenericParam whose index is NOT among the
            // quantifiers — and NOT shadowed by an inner binder — is a free
            // variable: the body is open.
            TypeData::GenericParam { index, .. } => {
                if locally_bound.contains(index) {
                    return false; // bound by an INNER binder — shadowed.
                }
                !quantifiers.iter().any(|(i, _)| *i == *index)
            }
            TypeData::Slice { elem } => {
                self.poly_body_is_open_inner(*elem, quantifiers, locally_bound)
            }
            TypeData::Ref { ty: inner, .. } => {
                self.poly_body_is_open_inner(*inner, quantifiers, locally_bound)
            }
            TypeData::Tuple { elems } => elems
                .iter()
                .any(|&e| self.poly_body_is_open_inner(e, quantifiers, locally_bound)),
            TypeData::Adt { args, .. } => args
                .iter()
                .any(|&a| self.poly_body_is_open_inner(a, quantifiers, locally_bound)),
            TypeData::Array { elem, .. } => {
                self.poly_body_is_open_inner(*elem, quantifiers, locally_bound)
            }
            TypeData::Pointer { ty: inner } => {
                self.poly_body_is_open_inner(*inner, quantifiers, locally_bound)
            }
            TypeData::Ptr { size, pointee } => {
                self.poly_body_is_open_inner(*size, quantifiers, locally_bound)
                    || self.poly_body_is_open_inner(*pointee, quantifiers, locally_bound)
            }
            TypeData::Fn { params, ret, .. } => {
                params
                    .iter()
                    .any(|&p| self.poly_body_is_open_inner(p, quantifiers, locally_bound))
                    || self.poly_body_is_open_inner(*ret, quantifiers, locally_bound)
            }
            TypeData::Exists {
                param_index, base, ..
            } => {
                locally_bound.push(*param_index);
                let open = self.poly_body_is_open_inner(*base, quantifiers, locally_bound);
                locally_bound.pop();
                open
            }
            TypeData::Forall {
                param_index, body, ..
            } => {
                locally_bound.push(*param_index);
                let open = self.poly_body_is_open_inner(*body, quantifiers, locally_bound);
                locally_bound.pop();
                open
            }
            TypeData::AssociatedType { self_ty, .. } => {
                self.poly_body_is_open_inner(*self_ty, quantifiers, locally_bound)
            }
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.poly_body_is_open_inner(a, quantifiers, locally_bound)),
            TypeData::Mu {
                param_index, body, ..
            }
            | TypeData::Nu {
                param_index, body, ..
            } => {
                locally_bound.push(*param_index);
                let open = self.poly_body_is_open_inner(*body, quantifiers, locally_bound);
                locally_bound.pop();
                open
            }
            TypeData::Opaque { hidden, .. } => {
                hidden.is_some_and(|h| self.poly_body_is_open_inner(h, quantifiers, locally_bound))
            }
            // Leaf / closed types never contain an inference variable, a
            // nested polytope, or an unbound generic parameter.
            _ => false,
        }
    }

    pub fn rational(&mut self, int_bits: u8, frac_bits: u8) -> TypeId {
        self.alloc(TypeData::Rational {
            int_bits,
            frac_bits,
        })
    }

    pub fn dyn_trait(&mut self, traits: Vec<DefId>) -> TypeId {
        self.alloc(TypeData::DynTrait { traits })
    }

    pub fn exists(
        &mut self,
        param_index: usize,
        name: Symbol,
        base: TypeId,
        invariant: crate::ast::Expr<'input>,
    ) -> TypeId {
        let id = self.alloc(TypeData::Exists {
            param_index,
            name,
            base,
        });
        self.interner.meta.write().entry(id).or_insert(TypeMeta {
            default_value: None,
            invariant: Some(invariant),
            no_default: false,
        });
        id
    }

    pub fn forall(&mut self, param_index: usize, param_name: Symbol, body: TypeId) -> TypeId {
        self.alloc(TypeData::Forall {
            param_index,
            param_name,
            body,
        })
    }

    /// Skip the `subst` type-pool lookup limitations and directly build
    /// the replacement type.  This avoids the `fn_ty_no_alloc().expect()`
    /// panic that occurs when `subst` tries to find a pre-existing type
    /// that hasn't been created yet.
    pub fn replace_generic(
        &mut self,
        ty: TypeId,
        param_index: usize,
        replacement: TypeId,
    ) -> TypeId {
        if !self.type_contains_param(param_index, ty) {
            return ty;
        }
        let data = self.get_arc(ty);
        match &*data {
            TypeData::GenericParam { index, .. } if *index == param_index => replacement,
            TypeData::Fn { params, ret } => {
                let new_params: SmallVec<[TypeId; 4]> = params
                    .iter()
                    .map(|&p| self.replace_generic(p, param_index, replacement))
                    .collect();
                let new_ret = self.replace_generic(*ret, param_index, replacement);
                self.function(new_params, new_ret)
            }
            TypeData::Forall {
                param_index: pi,
                param_name,
                body,
            } => {
                // Binder shadowing: if the binder's param_index matches the
                // parameter being substituted, the binder shadows it — do NOT
                // recurse into the body.
                if *pi == param_index {
                    ty
                } else {
                    let new_body = self.replace_generic(*body, param_index, replacement);
                    self.forall(*pi, *param_name, new_body)
                }
            }
            TypeData::Exists {
                param_index: pi,
                name,
                base,
            } => {
                // Binder shadowing: same as Forall.
                if *pi == param_index {
                    ty
                } else {
                    let new_base = self.replace_generic(*base, param_index, replacement);
                    let new_id = self.alloc(TypeData::Exists {
                        param_index: *pi,
                        name: *name,
                        base: new_base,
                    });
                    // Preserve Exists metadata (invariant, default_value), matching
                    // the same invariant in subst()'s Exists arm.
                    let src_meta = self.interner.meta.read().get(&ty).cloned();
                    if let Some(meta) = src_meta {
                        self.interner.meta.write().entry(new_id).or_insert(meta);
                    }
                    new_id
                }
            }
            TypeData::Mu {
                param_index: pi,
                param_name,
                body,
            } => {
                // Binder shadowing: same as Forall.
                if *pi == param_index {
                    ty
                } else {
                    let new_body = self.replace_generic(*body, param_index, replacement);
                    self.alloc(TypeData::Mu {
                        param_index: *pi,
                        param_name: *param_name,
                        body: new_body,
                    })
                }
            }
            TypeData::Nu {
                param_index: pi,
                param_name,
                body,
            } => {
                // Binder shadowing: same as Forall.
                if *pi == param_index {
                    ty
                } else {
                    let new_body = self.replace_generic(*body, param_index, replacement);
                    self.alloc(TypeData::Nu {
                        param_index: *pi,
                        param_name: *param_name,
                        body: new_body,
                    })
                }
            }
            TypeData::Poly { quantifiers, body } => {
                // Poly is a binder over all its quantifiers.  If any quantifier
                // shadows the target param_index, do not recurse.
                if quantifiers.iter().any(|(idx, _)| *idx == param_index) {
                    ty
                } else {
                    let new_body = self.replace_generic(*body, param_index, replacement);
                    self.poly(quantifiers.clone(), new_body)
                }
            }
            // ── Composite types: recurse into all sub-components ──
            TypeData::Ref {
                ty: inner,
                mutable,
                lifetime,
            } => {
                let new_inner = self.replace_generic(*inner, param_index, replacement);
                self.reference_with_lifetime(new_inner, *mutable, *lifetime)
            }
            TypeData::Pointer { ty: inner } => {
                let new_inner = self.replace_generic(*inner, param_index, replacement);
                self.pointer(new_inner)
            }
            TypeData::Ptr { size, pointee } => {
                let new_size = self.replace_generic(*size, param_index, replacement);
                let new_pointee = self.replace_generic(*pointee, param_index, replacement);
                self.ptr(new_size, new_pointee)
            }
            TypeData::Array { elem, size } => {
                let new_elem = self.replace_generic(*elem, param_index, replacement);
                self.array(new_elem, *size)
            }
            TypeData::Slice { elem } => {
                let new_elem = self.replace_generic(*elem, param_index, replacement);
                self.slice(new_elem)
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let new_self = self.replace_generic(*self_ty, param_index, replacement);
                self.associated_type(*trait_id, *name, new_self)
            }
            TypeData::Tuple { elems } => {
                let new_elems: SmallVec<[TypeId; 4]> = elems
                    .iter()
                    .map(|&e| self.replace_generic(e, param_index, replacement))
                    .collect();
                self.tuple(new_elems)
            }
            TypeData::Adt { kind, def_id, args } => {
                let new_args: SmallVec<[TypeId; 4]> = args
                    .iter()
                    .map(|&a| self.replace_generic(a, param_index, replacement))
                    .collect();
                self.alloc(TypeData::Adt {
                    kind: *kind,
                    def_id: *def_id,
                    args: new_args,
                })
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: SmallVec<[TypeId; 4]> = alternatives
                    .iter()
                    .map(|&a| self.replace_generic(a, param_index, replacement))
                    .collect();
                if new_alts.len() == 1 {
                    new_alts[0]
                } else {
                    self.alloc(TypeData::Coproduct {
                        alternatives: new_alts,
                    })
                }
            }
            _ => ty,
        }
    }

    pub fn generic_param(&mut self, index: usize, name: Symbol) -> TypeId {
        self.alloc(TypeData::GenericParam { index, name })
    }

    pub fn associated_type(&mut self, trait_id: DefId, name: Symbol, self_ty: TypeId) -> TypeId {
        self.alloc(TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        })
    }

    /// Check whether `param` occurs inside `ty` (the "occurs check").
    ///
    /// # Why no `visited` set is needed
    ///
    /// The `types` arena (`Vec<Arc<TypeData>>`) is physically a DAG — every
    /// `TypeData` is allocated before any cycles could exist, and the only
    /// way to form a cycle is through the `bindings` table.  Since this
    /// function calls `self.resolve_binding(ty)` first, the incoming `ty`
    /// is already dereferenced past any binding chain, making the recursive
    /// walk of the type structure **acyclic by construction**.
    ///
    /// A naive reader might be tempted to add a `visited: HashSet<TypeId>`
    /// to guard against infinite recursion.  **Do not.**  It would add O(n)
    /// memory overhead and mask the fact that the real cycle-safety proof
    /// lives upstream, in the binding layer.
    pub(crate) fn occurs_check(&self, param: TypeId, ty: TypeId) -> bool {
        if param == ty {
            return true;
        }
        let resolved = self.resolve_binding(ty);
        // Resolve again in case ty had a binding chain that ends at param.
        if resolved == param {
            return true;
        }
        match &self.slot(resolved.index()).as_ref() {
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.occurs_check(param, a)),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.occurs_check(param, e)),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.occurs_check(param, a))
            }
            TypeData::Array { elem, .. } => self.occurs_check(param, *elem),
            TypeData::Slice { elem } => self.occurs_check(param, *elem),
            TypeData::Ref { ty, .. } => self.occurs_check(param, *ty),
            TypeData::Pointer { ty } => self.occurs_check(param, *ty),
            TypeData::Ptr { size, pointee } => {
                self.occurs_check(param, *size) || self.occurs_check(param, *pointee)
            }
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.occurs_check(param, p))
                    || self.occurs_check(param, *ret)
            }
            TypeData::Poly { body, .. } => self.occurs_check(param, *body),
            TypeData::Exists { base, .. } => self.occurs_check(param, *base),
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => self.occurs_check(param, *body),
            TypeData::AssociatedType { self_ty, .. } => self.occurs_check(param, *self_ty),
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => false,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::Regex { .. }
            | TypeData::DynTrait { .. }
            | TypeData::SkolemVar { .. }
            | TypeData::Type
            | TypeData::Opaque { .. } => false,
        }
    }

    /// Insert a binding, recording the old value in the current transaction's
    /// undo log if one is active.  Always use this instead of
    /// `self.bindings.insert(...)` so that transactions can
    /// correctly roll back.
    pub(crate) fn set_binding(&mut self, key: TypeId, value: TypeId) -> bool {
        // ── Seal-the-wall guard ─────────────────────────────────────
        // GADT arm processing must never bind a GenericParam into the
        // global bindings table: refinements live in GadtContext.facts
        // (scoped, popped with the arm) and resolve_binding consults them
        // while an arm is active.  A GenericParam bound here while
        // `arm_depth > 0` means the refinement machinery leaked into the
        // global table (cross-arm contamination / post-arm leakage).
        //
        // Exemption: idempotent re-writes.  Pattern<'input> instantiation
        // legitimately binds a scrutinee generic param to a synthetic
        // infer var (`T → ?a`) at arm_depth 0 (before push), and later
        // unification inside the arm may re-write that SAME binding to
        // the SAME resolved type — not a new leak.
        // Fail closed in ALL builds (not just debug): a seal violation
        // means the refinement machinery leaked a GenericParam binding
        // into the global table.  Recoverable: SKIP the binding (do not
        // pollute the table) and record the violation — no panic (which
        // would abort the compiler process).
        if self.gadt.arm_depth.load(AtomicOrdering::Relaxed) > 0
            && matches!(self.get_raw(key), TypeData::GenericParam { .. })
        {
            // Check if this is path compression (re-binding to the same
            // resolved type) — the ONLY permitted GenericParam binding
            // inside an arm.  Any DIFFERENT binding is a seal violation.
            if self.bindings.contains_key(&key)
                && self.resolve_binding_no_gadt(key) == self.resolve_binding_no_gadt(value)
            {
                // Path compression to the same type — no-op, skip the
                // unnecessary write.
                return true;
            }
            self.seal_violations = self.seal_violations + 1;
            return false;
        }
        // Record the origin span for GenericParam bindings (precise E104
        // error location).  Only recorded inside a span-carrying unify
        // (`unify_tracked`); solver-driven bindings record nothing and
        // E104 falls back to the function span.
        if matches!(self.get_raw(key), TypeData::GenericParam { .. })
            && let Some(origin) = self.current_unify_span
        {
            self.generic_binding_origins.insert(key, origin);
        }
        if let Some(log) = self.transaction_stack.last_mut() {
            let old = self.bindings.get(&key).copied();
            log.push((key, old));
        }
        self.bindings.insert(key, value);
        true
    }

    /// When `self_ty` resolves to a concrete ADT, return its `DefId`.
    /// Full projection resolution (finding the impl's concrete associated
    /// type) requires `TraitEnv` and is performed by the checker.
    pub fn try_normalize_associated_type_def_id(&self, self_ty: TypeId) -> Option<DefId> {
        let resolved = self.resolve_binding(self_ty);
        match self.get(resolved) {
            TypeData::Adt { def_id, .. } => Some(*def_id),
            _ => None,
        }
    }

    pub fn enter_universe(&mut self) -> (usize, TypeId) {
        let universe = self.interner.next_universe.load(AtomicOrdering::Relaxed);
        self.interner
            .next_universe
            .store(universe + 1, AtomicOrdering::Relaxed);
        // Dynamically create a SkolemVar with the correct universe_num
        let skolem = self.alloc(TypeData::SkolemVar {
            id: universe,
            universe_num: universe,
        });
        (universe, skolem)
    }

    /// Allocate a fresh SkolemVar for a GADT existential parameter.
    /// Unlike `enter_universe`, this does NOT increment the universe
    /// counter — it uses a separate counter (`next_gadt_skolem_id`)
    /// that is not tied to the Forall-level universe hierarchy.
    /// This prevents unbounded `next_universe` growth in `while-let`
    /// loops on existential GADT variants.
    ///
    /// Sentinel universe number for GADT existential skolems; must be
    /// larger than any HRTB universe so `check_skolem_escape` can
    /// distinguish them.  The isolation from the main universe counter
    /// mirrors OutsideIn(X)'s treatment of implication-constraint
    /// skolems as a separate scope (Vytiniotis et al., JFP 2011).
    pub const GADT_SKOLEM_UNIVERSE: usize = usize::MAX;

    pub fn fresh_gadt_skolem(&mut self) -> TypeId {
        let id = self
            .interner
            .next_gadt_skolem_id
            .load(AtomicOrdering::Relaxed);
        self.interner
            .next_gadt_skolem_id
            .store(id + 1, AtomicOrdering::Relaxed);
        self.alloc(TypeData::SkolemVar {
            id,
            universe_num: Self::GADT_SKOLEM_UNIVERSE,
        })
    }

    pub fn check_skolem_escape(&self, ty: TypeId, max_universe: usize) -> Option<usize> {
        let resolved = self.resolve_binding(ty);
        match self.get(resolved) {
            TypeData::SkolemVar { universe_num, .. } if *universe_num > max_universe => {
                Some(*universe_num)
            }
            TypeData::Adt { args, .. }
            | TypeData::Tuple { elems: args, .. }
            | TypeData::Coproduct {
                alternatives: args, ..
            } => {
                for &a in args {
                    if let Some(u) = self.check_skolem_escape(a, max_universe) {
                        return Some(u);
                    }
                }
                None
            }
            TypeData::Fn { params, ret } => {
                for &p in params {
                    if let Some(u) = self.check_skolem_escape(p, max_universe) {
                        return Some(u);
                    }
                }
                self.check_skolem_escape(*ret, max_universe)
            }
            TypeData::Ref { ty, .. }
            | TypeData::Pointer { ty }
            | TypeData::Array { elem: ty, .. }
            | TypeData::Slice { elem: ty } => self.check_skolem_escape(*ty, max_universe),
            TypeData::Ptr { size, pointee, .. } => {
                let mut max = self.check_skolem_escape(*pointee, max_universe);
                if let Some(u) = self.check_skolem_escape(*size, max_universe) {
                    max = Some(max.map_or(u, |m| m.max(u)));
                }
                max
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. }
            | TypeData::Poly { body, .. } => self.check_skolem_escape(*body, max_universe),
            TypeData::AssociatedType { self_ty, .. } => {
                self.check_skolem_escape(*self_ty, max_universe)
            }
            _ => None,
        }
    }

    /// `TypeFactory::alloc` interns bottom-up — every caller allocates the
    /// child `TypeId`s first, then the parent — so identical logical types
    /// always build the same `TypeData` key and hit the same entry.
    ///
    /// Volatile types (`InferVar` / `SkolemVar` and any composite that
    /// embeds them) are scope-sensitive and NEVER interned: `alloc` skips
    /// caching them (`can_cache`), so they have no table entry — checked
    /// here with the same semantics (`type_is_volatile` mirrors
    /// `is_type_volatile_inner`, reading through `self.types`).
    pub(crate) fn find_type(&self, data: &TypeData) -> Option<TypeId> {
        if self.type_is_volatile(data) {
            return None;
        }
        self.interner.type_map.lock().get(data).copied()
    }

    pub fn subst(&mut self, ty: TypeId, subst: &Subst) -> TypeId {
        let resolved = self.resolve_binding(ty);
        // Clone the data to avoid borrow conflicts when calling self.subst() recursively.
        let data = self.slot(resolved.index()).clone();
        match &*data {
            TypeData::GenericParam { index, .. } => subst.get(*index).copied().unwrap_or(ty),
            TypeData::Int {
                bits,
                signed,
                overflow_policy,
            } => self.int_with_overflow(*bits, *signed, *overflow_policy),
            TypeData::UInt {
                bits,
                overflow_policy,
            } => self.uint_with_overflow(*bits, *overflow_policy),
            TypeData::Float { bits } => self.float(*bits),
            TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error => ty,
            TypeData::Adt { kind, def_id, args } => {
                let new_args: SmallVec<[TypeId; 4]> =
                    args.iter().map(|&a| self.subst(a, subst)).collect();
                let new_id = self.alloc(TypeData::Adt {
                    kind: *kind,
                    def_id: *def_id,
                    args: new_args,
                });
                new_id
            }
            TypeData::Tuple { elems } => {
                let new_elems: SmallVec<[TypeId; 4]> =
                    elems.iter().map(|&e| self.subst(e, subst)).collect();
                self.tuple(new_elems)
            }
            TypeData::Array { elem, size } => {
                let new_elem = self.subst(*elem, subst);
                self.array(new_elem, *size)
            }
            TypeData::Slice { elem } => {
                let new_elem = self.subst(*elem, subst);
                self.slice(new_elem)
            }
            TypeData::Ref {
                ty,
                mutable,
                lifetime,
            } => {
                let new_ty = self.subst(*ty, subst);
                self.reference_with_lifetime(new_ty, *mutable, *lifetime)
            }
            TypeData::Pointer { ty } => {
                let new_ty = self.subst(*ty, subst);
                self.pointer(new_ty)
            }
            TypeData::Ptr { size, pointee } => {
                let new_size = self.subst(*size, subst);
                let new_pointee = self.subst(*pointee, subst);
                self.ptr(new_size, new_pointee)
            }
            TypeData::Fn { params, ret } => {
                let new_params: SmallVec<[TypeId; 4]> =
                    params.iter().map(|&p| self.subst(p, subst)).collect();
                let new_ret = self.subst(*ret, subst);
                self.function(new_params, new_ret)
            }
            TypeData::Poly { quantifiers, body } => {
                // Poly is a binder over all its quantifiers.  Remove shadowed
                // keys from the substitution map and recurse, so that other
                // free variables in the body are still substituted.
                //
                // NOTE: this differs from `replace_infer()` which treats `Poly`
                // as a leaf (does not recurse).  The difference is intentional:
                // `subst()` replaces `GenericParam` indices, which CAN appear
                // in the body as bound variables (with binder shadowing).
                // `replace_infer()` replaces `InferVar` IDs, which must NOT
                // appear in a closed polytope body per the OmniML invariant
                // (omniml/lib/constraint_solver/principal_shape.ml:
                //  `| Poly _ -> assert false`).
                let shadowed_indices: Vec<usize> = quantifiers
                    .iter()
                    .map(|(idx, _)| *idx)
                    .filter(|idx| subst.get(*idx).is_some())
                    .collect();
                if shadowed_indices.is_empty() {
                    let new_body = self.subst(*body, subst);
                    self.poly(quantifiers.clone(), new_body)
                } else {
                    let filtered = subst.without_all(&shadowed_indices);
                    if filtered.is_empty() {
                        ty
                    } else {
                        let new_body = self.subst(*body, &filtered);
                        self.poly(quantifiers.clone(), new_body)
                    }
                }
            }
            TypeData::DynTrait { .. } => ty,
            TypeData::Forall {
                param_index,
                param_name,
                body,
            } => {
                // Binder shadowing: remove the shadowed key from the subst
                // map and recurse, so that other free variables in the body
                // are still substituted.
                if subst.get(*param_index).is_some() {
                    let filtered = subst.without(*param_index);
                    if filtered.is_empty() {
                        ty
                    } else {
                        let new_body = self.subst(*body, &filtered);
                        self.alloc(TypeData::Forall {
                            param_index: *param_index,
                            param_name: *param_name,
                            body: new_body,
                        })
                    }
                } else {
                    let new_body = self.subst(*body, subst);
                    self.alloc(TypeData::Forall {
                        param_index: *param_index,
                        param_name: *param_name,
                        body: new_body,
                    })
                }
            }
            TypeData::Mu {
                param_index,
                param_name,
                body,
            } => {
                if subst.get(*param_index).is_some() {
                    let filtered = subst.without(*param_index);
                    if filtered.is_empty() {
                        ty
                    } else {
                        let new_body = self.subst(*body, &filtered);
                        self.alloc(TypeData::Mu {
                            param_index: *param_index,
                            param_name: *param_name,
                            body: new_body,
                        })
                    }
                } else {
                    let new_body = self.subst(*body, subst);
                    self.alloc(TypeData::Mu {
                        param_index: *param_index,
                        param_name: *param_name,
                        body: new_body,
                    })
                }
            }
            TypeData::Nu {
                param_index,
                param_name,
                body,
            } => {
                if subst.get(*param_index).is_some() {
                    let filtered = subst.without(*param_index);
                    if filtered.is_empty() {
                        ty
                    } else {
                        let new_body = self.subst(*body, &filtered);
                        self.alloc(TypeData::Nu {
                            param_index: *param_index,
                            param_name: *param_name,
                            body: new_body,
                        })
                    }
                } else {
                    let new_body = self.subst(*body, subst);
                    self.alloc(TypeData::Nu {
                        param_index: *param_index,
                        param_name: *param_name,
                        body: new_body,
                    })
                }
            }
            TypeData::Exists {
                param_index,
                name,
                base,
            } => {
                if subst.get(*param_index).is_some() {
                    let filtered = subst.without(*param_index);
                    if filtered.is_empty() {
                        ty
                    } else {
                        let new_base = self.subst(*base, &filtered);
                        let new_id = self.alloc(TypeData::Exists {
                            param_index: *param_index,
                            name: *name,
                            base: new_base,
                        });
                        // Copy the original Exists meta (invariant, default_value) to the new node
                        let src_meta = self.interner.meta.read().get(&ty).cloned();
                        if let Some(meta) = src_meta {
                            self.interner.meta.write().entry(new_id).or_insert(meta);
                        }
                        new_id
                    }
                } else {
                    let new_base = self.subst(*base, subst);
                    let new_id = self.alloc(TypeData::Exists {
                        param_index: *param_index,
                        name: *name,
                        base: new_base,
                    });
                    // Copy the original Exists meta (invariant, default_value) to the new node
                    let src_meta = self.interner.meta.read().get(&ty).cloned();
                    if let Some(meta) = src_meta {
                        self.interner.meta.write().entry(new_id).or_insert(meta);
                    }
                    new_id
                }
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: SmallVec<[TypeId; 4]> =
                    alternatives.iter().map(|&a| self.subst(a, subst)).collect();
                self.coproduct(new_alts)
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let new_self = self.subst(*self_ty, subst);
                self.associated_type(*trait_id, *name, new_self)
            }
            _ => ty,
        }
    }

    /// Walk a type tree and replace every occurrence of any `TypeId` that
    /// appears as a key in `replacements` with the corresponding value.
    /// Uses a worklist to avoid recursive self-calls and borrow conflicts.
    /// TODO: wire into cross-arm type abstraction in match-arm exit path.
    #[allow(dead_code)]
    pub fn replace_type_ids(
        &mut self,
        ty: TypeId,
        replacements: &std::collections::HashMap<TypeId, TypeId>,
    ) -> TypeId {
        if let Some(&replacement) = replacements.get(&ty) {
            return replacement;
        }
        let resolved = self.resolve_binding(ty);
        // Also check the resolved type: if `ty → B` via bindings and
        // `replacements[B] = C`, we must return C, not process B's
        // structure as if unmodified.
        if resolved != ty
            && let Some(&replacement) = replacements.get(&resolved)
        {
            return replacement;
        }
        let data = self.slot(resolved.index()).clone();
        match &*data {
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::InferVar { .. }
            | TypeData::GenericParam { .. } => ty,
            TypeData::Adt { kind, def_id, args } => {
                let new_args: SmallVec<[TypeId; 4]> = args
                    .iter()
                    .map(|&a| self.replace_type_ids(a, replacements))
                    .collect();
                self.alloc(TypeData::Adt {
                    kind: *kind,
                    def_id: *def_id,
                    args: new_args,
                })
            }
            TypeData::Tuple { elems } => {
                let new_elems: SmallVec<[TypeId; 4]> = elems
                    .iter()
                    .map(|&e| self.replace_type_ids(e, replacements))
                    .collect();
                self.tuple(new_elems)
            }
            TypeData::Array { elem, size } => {
                let e = self.replace_type_ids(*elem, replacements);
                self.array(e, *size)
            }
            TypeData::Slice { elem } => {
                let e = self.replace_type_ids(*elem, replacements);
                self.slice(e)
            }
            TypeData::Ref {
                ty: rt,
                mutable,
                lifetime,
            } => {
                let r = self.replace_type_ids(*rt, replacements);
                self.reference_with_lifetime(r, *mutable, *lifetime)
            }
            TypeData::Pointer { ty: pt } => {
                let p = self.replace_type_ids(*pt, replacements);
                self.pointer(p)
            }
            TypeData::Ptr { size, pointee } => {
                let ns = self.replace_type_ids(*size, replacements);
                let np = self.replace_type_ids(*pointee, replacements);
                self.ptr(ns, np)
            }
            TypeData::Fn { params, ret } => {
                let new_ret = self.replace_type_ids(*ret, replacements);
                let new_params: SmallVec<[TypeId; 4]> = params
                    .iter()
                    .map(|&p| self.replace_type_ids(p, replacements))
                    .collect();
                self.function(new_params, new_ret)
            }
            TypeData::Poly { quantifiers, body } => {
                let b = self.replace_type_ids(*body, replacements);
                self.poly(quantifiers.clone(), b)
            }
            TypeData::Forall {
                param_index,
                param_name,
                body,
            } => {
                let b = self.replace_type_ids(*body, replacements);
                self.alloc(TypeData::Forall {
                    param_index: *param_index,
                    param_name: *param_name,
                    body: b,
                })
            }
            TypeData::Mu {
                param_index,
                param_name,
                body,
            } => {
                let b = self.replace_type_ids(*body, replacements);
                self.alloc(TypeData::Mu {
                    param_index: *param_index,
                    param_name: *param_name,
                    body: b,
                })
            }
            TypeData::Nu {
                param_index,
                param_name,
                body,
            } => {
                let b = self.replace_type_ids(*body, replacements);
                self.alloc(TypeData::Nu {
                    param_index: *param_index,
                    param_name: *param_name,
                    body: b,
                })
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: SmallVec<[TypeId; 4]> = alternatives
                    .iter()
                    .map(|&a| self.replace_type_ids(a, replacements))
                    .collect();
                self.coproduct(new_alts)
            }
            TypeData::Exists {
                param_index,
                name,
                base,
            } => {
                let b = self.replace_type_ids(*base, replacements);
                self.alloc(TypeData::Exists {
                    param_index: *param_index,
                    name: *name,
                    base: b,
                })
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let s = self.replace_type_ids(*self_ty, replacements);
                self.alloc(TypeData::AssociatedType {
                    trait_id: *trait_id,
                    name: *name,
                    self_ty: s,
                })
            }
            TypeData::DynTrait { .. }
            | TypeData::Rational { .. }
            | TypeData::SkolemVar { .. }
            | TypeData::Regex { .. }
            | TypeData::Opaque { .. }
            | TypeData::Type { .. } => ty,
        }
    }

    fn struct_ty_no_alloc(&self, def_id: DefId, args: SmallVec<[TypeId; 4]>) -> Option<TypeId> {
        self.find_type(&TypeData::Adt {
            kind: AdtKind::Struct,
            def_id,
            args,
        })
    }

    fn enum_ty_no_alloc(&self, def_id: DefId, args: SmallVec<[TypeId; 4]>) -> Option<TypeId> {
        self.find_type(&TypeData::Adt {
            kind: AdtKind::Enum,
            def_id,
            args,
        })
    }

    fn tuple_ty_no_alloc(&self, elems: SmallVec<[TypeId; 4]>) -> Option<TypeId> {
        self.find_type(&TypeData::Tuple { elems })
    }

    fn array_ty_no_alloc(&self, elem: TypeId, size: u64) -> Option<TypeId> {
        self.find_type(&TypeData::Array { elem, size })
    }

    fn slice_ty_no_alloc(&self, elem: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Slice { elem })
    }

    fn ref_ty_no_alloc(&self, ty: TypeId, mutable: bool) -> Option<TypeId> {
        self.find_type(&TypeData::Ref {
            ty,
            mutable,
            lifetime: None,
        })
    }

    fn pointer_ty_no_alloc(&self, ty: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Pointer { ty })
    }

    fn ptr_ty_no_alloc(&self, size: TypeId, pointee: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Ptr { size, pointee })
    }

    fn fn_ty_no_alloc(&self, params: SmallVec<[TypeId; 4]>, ret: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Fn { params, ret })
    }

    fn coproduct_ty_no_alloc(&self, alternatives: SmallVec<[TypeId; 4]>) -> Option<TypeId> {
        self.find_type(&TypeData::Coproduct { alternatives })
    }

    fn exists_ty_no_alloc(&self, param_index: usize, name: Symbol, base: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Exists {
            param_index,
            name,
            base,
        })
    }

    fn associated_ty_no_alloc(
        &self,
        trait_id: DefId,
        name: Symbol,
        self_ty: TypeId,
    ) -> Option<TypeId> {
        self.find_type(&TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        })
    }

    fn rational_ty_no_alloc(&self, int_bits: u8, frac_bits: u8) -> Option<TypeId> {
        self.find_type(&TypeData::Rational {
            int_bits,
            frac_bits,
        })
    }

    pub fn is_numeric(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. } => true,
            _ => false,
        }
    }

    pub fn is_integer(&self, ty: TypeId) -> bool {
        matches!(
            self.get(ty),
            TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::USize
        )
    }

    pub fn is_unsigned(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { signed, .. } => !*signed,
            TypeData::UInt { .. } => true,
            TypeData::USize => true,
            _ => false,
        }
    }

    pub fn is_signed(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { signed, .. } => *signed,
            _ => false,
        }
    }

    pub fn is_float(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Float { .. })
    }

    pub fn is_bool(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Bool)
    }

    pub fn is_char(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Char)
    }

    pub fn is_byte(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Byte)
    }

    pub fn is_usize(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::USize)
    }

    pub fn is_unit(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Unit)
    }

    pub fn is_never(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Never)
    }

    pub fn is_error(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Error)
    }

    /// Whether the error-recovery sentinel occurs ANYWHERE in the type —
    /// at the top level or nested inside a composite (e.g. `Vec<Error>`,
    /// `Ref<Error>`, `Adt<Error>`).  A composite type wrapping the sentinel
    /// is a recovery artifact just like a bare sentinel: enforcing traits on
    /// it surfaces cascading `... on type Vec<!!>` errors on top of an
    /// already-recovered expression.  (The resolution skips in the trait
    /// solver use this instead of the shallow `is_error` check.)
    pub fn contains_error(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Error => true,
            TypeData::Adt { args, .. } => args.iter().any(|a| self.contains_error(*a)),
            TypeData::Tuple { elems } => elems.iter().any(|e| self.contains_error(*e)),
            TypeData::Array { elem, .. } => self.contains_error(*elem),
            TypeData::Slice { elem } => self.contains_error(*elem),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => self.contains_error(*ty),
            TypeData::Ptr { size, pointee } => {
                self.contains_error(*size) || self.contains_error(*pointee)
            }
            TypeData::Fn { params, ret } => {
                params.iter().any(|p| self.contains_error(*p)) || self.contains_error(*ret)
            }
            TypeData::Exists { base, .. } | TypeData::Forall { body: base, .. } => {
                self.contains_error(*base)
            }
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => self.contains_error(*body),
            TypeData::Poly { body, .. } => self.contains_error(*body),
            TypeData::AssociatedType { self_ty, .. } => self.contains_error(*self_ty),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|a| self.contains_error(*a))
            }
            _ => false,
        }
    }

    pub fn is_reference(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Ref { .. })
    }

    pub fn is_pointer(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Pointer { .. })
    }

    /// Compute the constructor-depth of a type for Paterson-condition checking.
    /// GenericParam = 0, Int/Bool/etc = 1, Struct/Enum = 1 + max(depth of args)
    pub fn type_constructor_depth(&self, ty: TypeId) -> usize {
        match self.get(ty) {
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => 0,
            TypeData::Adt { args, .. } => {
                1 + args
                    .iter()
                    .map(|a| self.type_constructor_depth(*a))
                    .max()
                    .unwrap_or(0)
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => {
                1 + elems
                    .iter()
                    .map(|e| self.type_constructor_depth(*e))
                    .max()
                    .unwrap_or(0)
            }
            TypeData::Array { elem, .. } => 1 + self.type_constructor_depth(*elem),
            TypeData::Slice { elem } => 1 + self.type_constructor_depth(*elem),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                1 + self.type_constructor_depth(*ty)
            }
            TypeData::Ptr { size, pointee, .. } => {
                2 + self
                    .type_constructor_depth(*pointee)
                    .max(self.type_constructor_depth(*size))
            }
            TypeData::Fn { params, ret } => {
                1 + params
                    .iter()
                    .map(|p| self.type_constructor_depth(*p))
                    .max()
                    .unwrap_or(0)
                    .max(self.type_constructor_depth(*ret))
            }
            TypeData::AssociatedType { self_ty, .. } => 1 + self.type_constructor_depth(*self_ty),
            TypeData::Exists { base, .. } => 1 + self.type_constructor_depth(*base),
            TypeData::Poly {
                quantifiers: _,
                body,
            } => 1 + self.type_constructor_depth(*body),
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => 1 + self.type_constructor_depth(*body),
            TypeData::DynTrait { .. } => 1,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::Regex { .. }
            | TypeData::SkolemVar { .. }
            | TypeData::Type
            | TypeData::Opaque { .. } => 1,
        }
    }

    pub fn is_struct(&self, ty: TypeId) -> bool {
        matches!(
            self.get(ty),
            TypeData::Adt {
                kind: AdtKind::Struct,
                ..
            }
        )
    }

    pub fn is_enum(&self, ty: TypeId) -> bool {
        matches!(
            self.get(ty),
            TypeData::Adt {
                kind: AdtKind::Enum,
                ..
            }
        )
    }

    pub fn is_tuple(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Tuple { .. })
    }

    pub fn is_array(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Array { .. })
    }

    pub fn is_slice(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Slice { .. })
    }

    pub fn is_fn(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Fn { .. })
    }

    pub fn is_dyn_trait(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::DynTrait { .. })
    }

    pub fn is_exists(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Exists { .. })
    }

    pub fn is_poly(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Poly { .. })
    }

    pub fn is_rational(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Rational { .. })
    }

    /// Recursively check whether `ty` or any of its children contains a
    /// `Regex` type.  Used to enforce the rule that `Regex` types cannot
    /// appear in contracts (SYNTAX.md §Compile-Time Regular Expressions).
    pub fn contains_regex(&self, ty: TypeId) -> bool {
        let resolved = self.resolve_binding(ty);
        match self.get(resolved) {
            TypeData::Regex { .. } => true,
            // Composite types: recurse into children.
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.contains_regex(a)),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.contains_regex(e)),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.contains_regex(a))
            }
            TypeData::Array { elem, .. } => self.contains_regex(*elem),
            TypeData::Slice { elem } => self.contains_regex(*elem),
            TypeData::Ref { ty, .. } => self.contains_regex(*ty),
            TypeData::Pointer { ty } => self.contains_regex(*ty),
            TypeData::Ptr { size, pointee } => {
                self.contains_regex(*size) || self.contains_regex(*pointee)
            }
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.contains_regex(p)) || self.contains_regex(*ret)
            }
            TypeData::Poly { body, .. } => self.contains_regex(*body),
            TypeData::Exists { base, .. } => self.contains_regex(*base),
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => self.contains_regex(*body),
            TypeData::AssociatedType { self_ty, .. } => self.contains_regex(*self_ty),
            // Leaf types — no children, no Regex.
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::GenericParam { .. }
            | TypeData::InferVar { .. }
            | TypeData::DynTrait { .. }
            | TypeData::SkolemVar { .. }
            | TypeData::Type
            | TypeData::Opaque { .. } => false,
        }
    }

    pub fn bits_of_rational_int(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Rational { int_bits, .. } => Some(*int_bits),
            _ => None,
        }
    }

    pub fn bits_of_rational_frac(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Rational { frac_bits, .. } => Some(*frac_bits),
            _ => None,
        }
    }

    pub fn is_generic_param(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::GenericParam { .. })
    }

    pub fn is_associated_type(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::AssociatedType { .. })
    }

    pub fn bits_of_int(&self, ty: TypeId) -> Option<u32> {
        match self.get(ty) {
            TypeData::Int { bits, .. } | TypeData::UInt { bits, .. } => Some(*bits),
            _ => None,
        }
    }

    pub fn signedness_of_int(&self, ty: TypeId) -> Option<bool> {
        match self.get(ty) {
            TypeData::Int { signed, .. } => Some(*signed),
            TypeData::UInt { .. } => Some(false),
            _ => None,
        }
    }

    pub fn bits_of_float(&self, ty: TypeId) -> Option<u32> {
        match self.get(ty) {
            TypeData::Float { bits } => Some(*bits),
            _ => None,
        }
    }

    pub fn size_of_array(&self, ty: TypeId) -> Option<u64> {
        match self.get(ty) {
            TypeData::Array { size, .. } => Some(*size),
            _ => None,
        }
    }

    pub fn elem_of_array(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Array { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    pub fn elem_of_slice(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Slice { elem } => Some(*elem),
            _ => None,
        }
    }

    pub fn pointee_of_ref(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Ref { ty: t, .. } => Some(*t),
            _ => None,
        }
    }

    pub fn mutability_of_ref(&self, ty: TypeId) -> Option<bool> {
        match self.get(ty) {
            TypeData::Ref { mutable, .. } => Some(*mutable),
            _ => None,
        }
    }

    pub fn pointee_of_pointer(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Pointer { ty: t } => Some(*t),
            _ => None,
        }
    }

    pub fn params_of_fn(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.get(ty) {
            TypeData::Fn { params, .. } => Some(params),
            _ => None,
        }
    }

    pub fn ret_of_fn(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Fn { ret, .. } => Some(*ret),
            _ => None,
        }
    }

    pub fn tuple_elems(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.get(ty) {
            TypeData::Tuple { elems } => Some(elems),
            _ => None,
        }
    }

    pub fn base_of_exists(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Exists { base, .. } => Some(*base),
            _ => None,
        }
    }

    pub fn name_of_exists(&self, ty: TypeId) -> Option<&Symbol> {
        match self.get(ty) {
            TypeData::Exists { name, .. } => Some(name),
            _ => None,
        }
    }

    pub fn set_meta(&mut self, id: TypeId, meta: TypeMeta<'input>) {
        self.interner.meta.write().insert(id, meta);
    }
}

#[derive(Debug, Clone)]
pub struct Subst {
    map: HashMap<usize, TypeId>,
}

impl Subst {
    pub fn new() -> Self {
        Subst {
            map: HashMap::default(),
        }
    }

    pub fn insert(&mut self, index: usize, ty: TypeId) {
        self.map.insert(index, ty);
    }

    pub fn from_single(index: usize, ty: TypeId) -> Self {
        let mut map = HashMap::default();
        map.insert(index, ty);
        Subst { map }
    }

    pub fn get(&self, index: usize) -> Option<&TypeId> {
        self.map.get(&index)
    }

    pub fn extend(&mut self, other: &Subst) {
        for (&k, &v) in other.map.iter() {
            self.map.insert(k, v);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Apply a transformation to every TypeId value in this substitution.
    pub fn map_values(&mut self, f: &mut dyn FnMut(TypeId) -> TypeId) {
        for v in self.map.values_mut() {
            *v = f(*v);
        }
    }

    /// Iterate over all TypeId values in this substitution.
    pub fn for_each_value(&self, mut f: impl FnMut(TypeId)) {
        for &v in self.map.values() {
            f(v);
        }
    }

    /// Iterate over all TypeId values in this substitution.
    pub fn values(&self) -> impl Iterator<Item = &TypeId> {
        self.map.values()
    }

    /// Return a new `Subst` with the given index removed.
    /// If the index is not present, returns a clone of `self`.
    pub fn without(&self, index: usize) -> Self {
        let mut map = self.map.clone();
        map.remove(&index);
        Subst { map }
    }

    /// Return a new `Subst` with all the given indices removed.
    pub fn without_all(&self, indices: &[usize]) -> Self {
        let mut map = self.map.clone();
        for idx in indices {
            map.remove(idx);
        }
        Subst { map }
    }
}

impl Default for Subst {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch {
        expected: TypeId,
        found: TypeId,
        span: crate::ast::Span,
    },
    UndefinedName {
        name: String,
        span: crate::ast::Span,
        suggestions: Vec<String>,
    },
    TypeNotFound {
        name: String,
        span: crate::ast::Span,
    },
    CannotInfer {
        span: crate::ast::Span,
    },
    GenericArgumentCount {
        expected: usize,
        found: usize,
        span: crate::ast::Span,
    },
    TraitNotImplemented {
        ty: TypeId,
        trait_name: String,
        span: crate::ast::Span,
    },
    InvariantViolation {
        ty: TypeId,
        expr: String,
        span: crate::ast::Span,
    },
    /// Skolem escape: a type variable was unified with a type from a deeper
    /// scope (e.g. a GADT arm), violating the level invariant.
    SkolemEscape {
        var_id: usize,
        var_level: usize,
        current_level: usize,
        span: crate::ast::Span,
    },
    /// Binder scope escape: a GenericParam that is currently bound under an
    /// active Forall/Exists/Mu/Nu/Poly binder was unified with a foreign
    /// type, which would leak the quantified variable into the surrounding
    /// context.
    ScopeEscape {
        index: usize,
        span: crate::ast::Span,
    },
    MutableBorrow {
        span: crate::ast::Span,
    },
    ImmutableBorrow {
        span: crate::ast::Span,
    },
    OutOfBounds {
        index: u64,
        size: u64,
        span: crate::ast::Span,
    },
    DivisionByZero {
        span: crate::ast::Span,
    },
    Overflow {
        span: crate::ast::Span,
    },
    NeverType {
        span: crate::ast::Span,
    },
    CircularDependency {
        name: String,
        span: crate::ast::Span,
    },
    DuplicateDefinition {
        name: String,
        span: crate::ast::Span,
        previous: crate::ast::Span,
    },
    PrivateField {
        name: String,
        span: crate::ast::Span,
    },
    PrivateType {
        name: String,
        span: crate::ast::Span,
    },
    PrivateFunction {
        name: String,
        span: crate::ast::Span,
    },
    PatternNotExhaustive {
        span: crate::ast::Span,
    },
    PatternRedundant {
        span: crate::ast::Span,
    },
    PatternTypeMismatch {
        expected: TypeId,
        found: TypeId,
        span: crate::ast::Span,
    },
    RecursiveType {
        ty: TypeId,
        span: crate::ast::Span,
    },
}

impl TypeError {
    /// The source span the error is anchored at.
    pub fn span(&self) -> crate::ast::Span {
        match self {
            TypeError::Mismatch { span, .. }
            | TypeError::UndefinedName { span, .. }
            | TypeError::TypeNotFound { span, .. }
            | TypeError::GenericArgumentCount { span, .. }
            | TypeError::TraitNotImplemented { span, .. }
            | TypeError::InvariantViolation { span, .. }
            | TypeError::SkolemEscape { span, .. }
            | TypeError::ScopeEscape { span, .. }
            | TypeError::OutOfBounds { span, .. }
            | TypeError::CircularDependency { span, .. }
            | TypeError::DuplicateDefinition { span, .. }
            | TypeError::PrivateField { span, .. }
            | TypeError::PrivateType { span, .. }
            | TypeError::PrivateFunction { span, .. }
            | TypeError::PatternTypeMismatch { span, .. }
            | TypeError::RecursiveType { span, .. }
            | TypeError::CannotInfer { span }
            | TypeError::MutableBorrow { span }
            | TypeError::ImmutableBorrow { span }
            | TypeError::DivisionByZero { span }
            | TypeError::Overflow { span }
            | TypeError::NeverType { span }
            | TypeError::PatternNotExhaustive { span }
            | TypeError::PatternRedundant { span } => *span,
        }
    }

    /// Overwrite the source span of this error.
    /// Used by the solver to patch errors originating from deep in `unify`
    /// with the span of the constraint that triggered the unification.
    pub fn set_span(&mut self, new_span: crate::ast::Span) {
        match self {
            TypeError::Mismatch { span, .. }
            | TypeError::UndefinedName { span, .. }
            | TypeError::TypeNotFound { span, .. }
            | TypeError::GenericArgumentCount { span, .. }
            | TypeError::TraitNotImplemented { span, .. }
            | TypeError::InvariantViolation { span, .. }
            | TypeError::SkolemEscape { span, .. }
            | TypeError::ScopeEscape { span, .. }
            | TypeError::OutOfBounds { span, .. }
            | TypeError::CircularDependency { span, .. }
            | TypeError::DuplicateDefinition { span, .. }
            | TypeError::PrivateField { span, .. }
            | TypeError::PrivateType { span, .. }
            | TypeError::PrivateFunction { span, .. }
            | TypeError::PatternTypeMismatch { span, .. }
            | TypeError::RecursiveType { span, .. }
            | TypeError::CannotInfer { span }
            | TypeError::MutableBorrow { span }
            | TypeError::ImmutableBorrow { span }
            | TypeError::DivisionByZero { span }
            | TypeError::Overflow { span }
            | TypeError::NeverType { span }
            | TypeError::PatternNotExhaustive { span }
            | TypeError::PatternRedundant { span } => *span = new_span,
        }
    }
}

/// Auto-trait audit for the type context.
///
/// `TypeContext` is deliberately **not** `Sync` — it carries ~20
/// `Cell`/`RefCell` fields (`bindings`, `transaction_stack`,
/// `binder_stack`, `current_unify_span`, …) whose values are *positional*:
/// two threads interleaving on them would corrupt each other's unification
/// state even if each individual field were locked.
///
/// It *is* `Send`, and that is what makes it usable from a `Sync` query
/// provider: `Mutex<T>: Sync` only needs `T: Send`, so wrapping the whole
/// context in one mutex yields a `Sync` handle that hands out `&mut
/// TypeContext` to exactly one query at a time.  See
/// `crate::session::CompilerSession`.
///
/// These tests exist so that adding an `Rc`/`RefCell` field (or making an
/// existing one hold a non-`Send` type) fails at compile time here rather
/// than silently removing the option of sharing the context.
#[cfg(test)]
mod send_sync_audit {
    use super::TypeContext;
    use super::TypeData;
    use std::sync::Arc;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn type_context_is_send_and_shareable_under_a_lock() {
        // `Mutex`/`RwLock` are `Sync` iff their contents are `Send`, so
        // these three lines are what the session's design rests on.
        assert_send::<TypeContext<'static>>();
        assert_send::<parking_lot::Mutex<TypeContext<'static>>>();
        assert_send::<parking_lot::RwLock<TypeContext<'static>>>();
    }

    /// The interner half of `TypeContext` must be shareable by reference:
    /// `&T: Send` and `&T: Sync` both require `T: Sync`, so a session that
    /// hands `&TypeContext` to parallel queries cannot work without it.
    /// Every remaining mutable-inference field lives behind a lock precisely
    /// so this bound holds.
    #[test]
    fn type_context_is_sync() {
        assert_sync::<TypeContext<'static>>();
        assert_sync::<crate::hir::types::data::TypeFactory<'static>>();
        assert_sync::<crate::hir::types::gadt::GadtContext<'static>>();
    }

    /// The whole point of the split: several contexts can exist at once
    /// over one interner, be moved to other threads, and still agree on
    /// `TypeId`s.  `Send` is what lets a query own its context outright.
    #[test]
    fn per_query_contexts_share_one_interner() {
        assert_send::<TypeContext<'static>>();
        assert_send::<Arc<crate::hir::types::data::TypeFactory<'static>>>();

        let mut base = TypeContext::new();
        let a = base.alloc(TypeData::Bool);
        let mut query = TypeContext::new_sharing(&base);

        // Same interner ⇒ the same id resolves to the same type, and an id
        // minted by one context is legible to the other.
        assert_eq!(query.get(a), &TypeData::Bool);
        assert_eq!(base.get(query.alloc(TypeData::Char)), &TypeData::Char);

        // Inference state is NOT shared: binding in `query` is invisible to
        // `base`, which is what makes per-query checking independent.
        let var = base.alloc(TypeData::InferVar { id: 0, universe: 0 });
        query.set_binding(var, a);
        assert_eq!(query.resolve_binding(var), a);
        assert_eq!(base.resolve_binding(var), var);
    }

    /// `TypeContext` is `Sync`, so `&TypeContext` crosses threads — but
    /// that only buys read-only sharing of the interner.  Mutating
    /// inference state still needs `&mut`, which is why parallelism comes
    /// from one context per query rather than one shared context.
    #[test]
    fn shared_context_can_be_read_from_two_threads() {
        let ctx = TypeContext::new();
        let b = ctx.alloc(TypeData::Bool);
        let shared = &ctx;
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(move || {
                    assert_eq!(shared.get(b), &TypeData::Bool);
                });
            }
        });
    }
}

#[cfg(test)]
mod tests;
