//! The compiler's query definitions.
//!
//! MSRV NOTE: this module (and the query system generally) uses
//! `slice::is_sorted` (stable since Rust 1.82) and let-chains (edition
//! 2024 / Rust 1.88+).  If `cargo build` fails on either, the toolchain is
//! older than the crate's effective MSRV — update the toolchain, do not
//! rewrite the code.
//!
//! This is where the query system stops being infrastructure and starts
//! being used.  Each item here is a `Key -> Value` mapping computed on
//! demand, cached, and dep-tracked for incremental re-use.
//!
//! ## Value shape: owned snapshots and arena borrows
//!
//! `Query::Value` must be `Clone + Debug + Send + Sync + 'q + StableHash`
//! (see `QueryValue` in the parent module).  The compiler's own data
//! structures — `SymbolTable<'input>`, `TraitEnv<'input>`,
//! `HirStmt<'input>` — borrow the parse arena, so they can never
//! themselves be query values.
//!
//! Two shapes are used here:
//!
//! - **Owned snapshots** extracted from that state: small, position-free,
//!   arena-independent summaries (`CheckFnBody`).  A summary has no spans
//!   to accidentally hash, so it satisfies the `StableHash` contract
//!   (parent module) for free.
//! - **Arena borrows** (`Value = &'q V`, `Storage = Arena`): `compute`
//!   allocates the summary in the session arena and the cache stores the
//!   bare reference — no `Arc`, no clone on the hit path
//!   (`CheckFnSignature`, `TypeDefShape`, `DocTokens`).
//!
//! ## Which queries are wired into the pipeline
//!
//! - [`CheckFnSignature`] — driven by `pre_register_signatures` in
//!   `hir::checker`, which is the compiler's real per-function pre-pass.
//! - [`TypeDefShape`] — still test-only; it is the reference example the
//!   query machinery was validated against.

// `TypeDefShape` has no production caller yet.  Remove once it does.
// (The allow is scoped below to the TypeDefShape items instead of being
// module-wide: a blanket `#![allow(dead_code)]` here was also silencing
// dead helpers on the production queries — exactly the kind of rot the
// attribute is supposed to surface, not hide.)

use super::{
    Arena, BodyWorkerSlot, ContentClass, DefaultCache, Owned, Query, QueryDescriptor,
    QueryProvider, Scheduler, StableHash, Stored, TaggedQueryKey,
};
use as_variant::as_variant;
use crate::ast::FileId;
use crate::ast::{Item, Stmt};
use crate::hir::polonius::{count_return_refs_ast, extract_borrow_signature, signature_facts};
use crate::hir::symbol::TypeKind;
use crate::hir::types::{DefId, SignatureFacts, TypeId};
use crate::symbol::Symbol;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Bridges `fmt::Write` into a `Hasher`, so `Debug` output flows directly
/// into the hasher without an intermediate `String`.  Zero-allocation.
struct HashWriter<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> std::fmt::Write for HashWriter<'_, H> {
    #[inline]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Owned, position-free summary of one resolved type definition.
///
/// `kind: None` means the `DefId` does not name a type definition.  That is
/// a legitimate answer rather than an error: the query is total, so callers
/// get a value instead of the latch poisoning that a panic would cause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeDefSummary {
    pub def_id: DefId,
    pub kind: Option<TypeKind>,
    /// Field name -> field type, in declaration order.
    pub fields: Vec<(Symbol, TypeId)>,
    /// Number of declared generic parameters.
    pub num_params: usize,
}

impl StableHash for TypeDefSummary {
    /// Hashes semantic identity only.  There are no spans in this type by
    /// construction — see the module note — so this cannot drift.
    fn stable_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        self.def_id.hash(&mut h);
        self.kind.hash(&mut h);
        self.num_params.hash(&mut h);
        for (name, ty) in &self.fields {
            name.hash(&mut h);
            ty.hash(&mut h);
        }
        h.finish()
    }
}

/// Resolved shape of a type definition, keyed by `DefId`.
///
/// The first query wired into the real pipeline.  It reads only
/// `SymbolTable`, so it needs no `&mut TypeContext` — which is why it can
/// land before the `&mut TypeContext` sharing problem is solved.
///
/// ARENA ORPHANING — explicit design decision: `Arena` storage never
/// reclaims a slot (see `erase.rs`), so every invalidation + recompute of
/// a hot key allocates a fresh `&'q TypeDefSummary` and orphans the old
/// one until the session ends.  Accepted deliberately: the value is small,
/// the invalidation rate is low, and the orphaned memory is bounded by
/// `(recompute count) × (value size)` over one session.  If this query
/// ever becomes invalidation-hot (e.g. a watch-mode editor driving
/// per-keystroke rebuilds), revisit the storage choice — the arena WILL
/// grow unboundedly under that regime.
pub struct TypeDefShape;

impl<'q, 'src: 'q> Query<'q, 'src> for TypeDefShape {
    type Storage = Arena;
    type Key = DefId;
    type Value = &'q TypeDefSummary;
    type Cache = DefaultCache<DefId, Stored<'q, 'src, Self>>;

    #[inline]
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "type_def_shape",
            "resolved shape (kind, fields, arity) of a type definition",
        )
    }

    fn into_tagged(key: &DefId) -> TaggedQueryKey {
        TaggedQueryKey::TypeDefShape(*key)
    }

    fn from_tagged(tag: &TaggedQueryKey) -> Option<DefId> {
        as_variant!(tag, TaggedQueryKey::TypeDefShape(key) => *key)
    }

    fn compute<S: Scheduler>(
        key: &DefId,
        provider: &dyn QueryProvider<'q, 'src, S>,
    ) -> Self::Value {
        let session = provider.session().expect(
            "TypeDefShape requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let arena = session.arena().expect(
            "TypeDefShape requires a provider carrying the session arena \
             (SessionQueryProvider::with_arena)",
        );
        let symbols = session.symbols();

        let summary = match symbols.lookup_type_by_def_id(*key) {
            Some(binding) => TypeDefSummary {
                def_id: *key,
                kind: Some(binding.kind),
                fields: binding.fields.iter().map(|f| (f.name, f.ty)).collect(),
                num_params: binding.params.len(),
            },
            None => TypeDefSummary {
                def_id: *key,
                kind: None,
                fields: Vec::new(),
                num_params: 0,
            },
        };
        arena.alloc(summary)
    }
}

impl StableHash for SignatureFacts {
    /// Semantic identity only: the facts are plain integer/bool vectors
    /// with no spans (the module note), so hashing them element-wise
    /// cannot drift.
    ///
    /// `known_placeholder_subset` is hashed in `Vec` order, so the hasher
    /// depends on the producer emitting it in a canonical order.  The one
    /// producer of this query's value (`hir::polonius::signature_facts`)
    /// does: it pushes in a double index loop over `inputs` and
    /// `outputs`, and the origins it emits are position-derived, so the
    /// vector arrives sorted ascending and the loop order is the
    /// canonical one.  The sibling producer (`hir::polonius`'s
    /// `signature_facts_of`, which `extend`s in body-traversal order)
    /// builds the CHECKER's facts, not this query value.
    ///
    /// The `debug_assert` is the insurance: a future producer that emits
    /// the pairs in a different order (e.g. a `HashMap` iteration, or a
    /// reversed loop) would otherwise change every fingerprint silently.
    /// Free in release.
    fn stable_hash(&self) -> u64 {
        debug_assert!(
            self.known_placeholder_subset.is_sorted(),
            "known_placeholder_subset must be emitted in canonical \
             (input-major, output-minor ascending) order — the stable hash \
             above depends on the Vec order"
        );
        let mut h = FxHasher::default();
        self.universal_region.hash(&mut h);
        self.input_borrow_mutable.hash(&mut h);
        for &(a, b) in &self.known_placeholder_subset {
            a.hash(&mut h);
            b.hash(&mut h);
        }
        h.finish()
    }
}

/// Where one function lives in `Program::items`.
///
/// A structural index rather than a `Symbol`: method names are not unique
/// (two impl blocks can both define `len`), and resolving a receiver type
/// for the key would drag `&mut TypeContext` into key construction.  The
/// index is stable for the lifetime of the `Program` a session was built
/// from: the pre-pass that walks that same `Program` mints the `FnId`s,
/// and every query keyed on them (signature facts, doc lines) resolves
/// them against that same `Program` — a `FnId` is meaningless across
/// sessions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId {
    /// Index into `Program::items`.
    pub item: usize,
    /// `None` = the item at `item` is a free function.
    /// `Some(i)` = the `i`-th method of the impl block at `item`.
    pub method: Option<usize>,
}

impl super::QueryKey for FnId {}

/// The borrow-signature facts one function contributes to the A(ρ)
/// registry: which parameter positions are input borrows, and the
/// declared `output ⟹ input` placeholder subsets.
///
/// Everything here is derived from the function's AST alone — no resolved
/// types, no inference — which is what lets the query run before (and
/// independently of) any body check.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FnSignatureFacts {
    /// Parameter indices whose declared type is a reference (`&`/`&mut`).
    pub input_positions: Vec<usize>,
    pub facts: SignatureFacts,
}

impl StableHash for FnSignatureFacts {
    fn stable_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        self.input_positions.hash(&mut h);
        h.write_u64(self.facts.stable_hash());
        h.finish()
    }
}

/// Compute one function's borrow signature from its AST.
///
/// Shared by the free-function and impl-method arms so the two cannot
/// drift the way the duplicated bodies in `pre_register_signatures` did.
///
/// ASSUMPTION: parameter names are unique within a function.  The borrow
/// signature keys input borrows by `Symbol` (`extract_borrow_signature`
/// pushes `(name, mutable)` per reference param), while `input_positions`
/// is positional; with two same-named reference params the name-keyed and
/// positional views diverge (the call-site mapping resolves borrows by
/// name).  Duplicate value-param names are diagnosed as redefinitions in
/// the same scope (E019 class) — if that check ever stops covering
/// params, this query needs a positional key instead.
fn signature_facts_of(
    params: &[crate::ast::Param<'_>],
    return_type: Option<&crate::ast::Type<'_>>,
    body: Option<&[Stmt<'_>]>,
) -> FnSignatureFacts {
    let param_refs: Vec<(Symbol, bool, bool, Option<Symbol>)> = params
        .iter()
        .map(|p| match p.ty.as_ref() {
            Some(crate::ast::Type::Reference {
                mutable, lifetime, ..
            }) => (p.name, true, *mutable, *lifetime),
            _ => (p.name, false, false, None),
        })
        .collect();
    let return_refs = return_type.map(count_return_refs_ast).unwrap_or(0);
    let sig = extract_borrow_signature(&param_refs, return_refs);
    let param_names: Vec<Symbol> = params.iter().map(|p| p.name).collect();
    let deriving =
        crate::hir::checker::derive_output_origins_ast(body.unwrap_or(&[]), &param_names);
    let facts = signature_facts(&sig, &deriving);
    let input_positions: Vec<usize> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.ty.as_ref(), Some(crate::ast::Type::Reference { .. })))
        .map(|(i, _)| i)
        .collect();
    FnSignatureFacts {
        input_positions,
        facts,
    }
}

/// The AST-level borrow signature of one function, keyed by [`FnId`].
///
/// This is the query that makes the A(ρ) pre-registration order-independent
/// *and* parallel: the checker asks for every function's facts up front
/// (via `par_get`) instead of registering them one at a time while walking
/// the item list.  It reads only `Program`, so it needs no `&mut` state —
/// the reason it can run on rayon at all.
///
/// ARENA ORPHANING — explicit design decision: like [`TypeDefShape`], this
/// query uses `Arena` storage, so one allocation is orphaned per key per
/// epoch and is only released by `clear_all`/`clear::<Self>`.  Untracked
/// queries recompute rarely; the notable exception is a driver-rebuilt
/// `Program` (bumping `program_epoch`), which re-allocates EVERY key —
/// accepted because the values are small and the rebuild is a session-scale
/// event, but a per-keystroke driver would grow the arena without bound.
pub struct CheckFnSignature;

impl<'q, 'src: 'q> Query<'q, 'src> for CheckFnSignature {
    type Storage = Arena;
    type Key = FnId;
    type Value = &'q FnSignatureFacts;
    type Cache = DefaultCache<FnId, Stored<'q, 'src, Self>>;

    #[inline]
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "check_fn_signature",
            "AST-level borrow signature (input borrow positions + A(rho) facts) of one function",
        )
        // Untracked query: compute reads only the frozen post-parse
        // `Program` and issues no nested query gets, so dep tracking buys
        // nothing — memo only.  The memo is bounded by the default cache
        // capacity (4096) with evict-one overflow (no working-set cliff),
        // which is the deliberate capacity decision for this query: a
        // crate with more functions keeps AT MOST that many entries and
        // pays deterministic recomputes for evicted ones.  Eviction is FIFO
        // by FIRST insertion via the `VecDeque` queue beside the memo map
        // — not bucket order and not recency.  Do not read
        // tracked-cache CLOCK admission quality into it.
        //
        // Cross-program binding: entries in the untracked memo are stamped
        // with the provider's `program_epoch` (the DRIVER bumps it per
        // rebuilt Program — `SessionQueryProvider::with_program_epoch`), and
        // a memo hit from a different epoch is treated as a miss (never
        // served — a FnId is meaningless across sessions, and these facts
        // drive the A(ρ) registry).  A driver that reuses one `QuerySystem`
        // across Programs has TWO sanctioned escape hatches: `clear_all`
        // (strong: the memo generation discards stale entries outright)
        // or an epoch bump (graceful: stale entries recompute).  A rebuild
        // with NEITHER is the one uncovered hazard — the shared epoch makes
        // the aliasing indistinguishable from a current-program hit, so it
        // is the driver's contract to uphold (see `program_epoch`'s doc).
        .with_untracked(true)
    }

    fn into_tagged(key: &FnId) -> TaggedQueryKey {
        TaggedQueryKey::CheckFnSignature(key.clone())
    }

    fn from_tagged(tag: &TaggedQueryKey) -> Option<FnId> {
        as_variant!(tag, TaggedQueryKey::CheckFnSignature(key) => key.clone())
    }

    fn compute<S: Scheduler>(key: &FnId, provider: &dyn QueryProvider<'q, 'src, S>) -> Self::Value {
        let session = provider.session().expect(
            "CheckFnSignature requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        // Arena storage: the facts are allocated in the session arena and
        // the cache stores the bare `&'q FnSignatureFacts` — a hit serves
        // the same pointer with no `Arc` wrap and no clone.
        let arena = session.arena().expect(
            "CheckFnSignature requires a provider carrying the session arena \
             (SessionQueryProvider::with_arena)",
        );
        let program = session.program();
        let Some(item) = program.items.get(key.item) else {
            return arena.alloc(FnSignatureFacts::default());
        };
        let facts = match (item, key.method) {
            // The comptime functions are compile-time — they never exist
            // at runtime (SYNTAX.md §const), so they do NOT participate in
            // the runtime borrow-signature registry.  The caller never
            // builds a `FnId` for one; answering with empty facts keeps
            // the query total instead of panicking (a panic would poison
            // the latch and surface as a cycle error).
            (
                Item::FunctionDef {
                    params,
                    return_type,
                    body,
                    is_comptime: false,
                    ..
                },
                None,
            ) => signature_facts_of(&params, return_type.as_ref(), body.as_deref()),
            // Comptime methods cannot appear here: `ImplMethod` (unlike the
            // free `FunctionDef` arm above, which filters `is_comptime:
            // false`) carries no comptime flag at all — the parser only
            // accepts comptime on free functions (SYNTAX.md §const), so an
            // impl method addressed by `FnId` is always a runtime method.
            (Item::ImplBlock { methods, .. }, Some(i)) => match methods.get(i) {
                Some(m) => signature_facts_of(&m.params, Some(&m.return_type), m.body.as_deref()),
                None => FnSignatureFacts::default(),
            },
            _ => FnSignatureFacts::default(),
        };
        arena.alloc(facts)
    }
}

// ── Per-function body check ──────────────────────────────────────

/// The complete body-check result of one free function, keyed by [`FnId`]:
/// the checked HIR body plus everything the caller used to salvage off the
/// worker checker (deferred obligations, GADT constructs, diagnostics).
///
/// As a query value this borrows the session arena (`FnBodyCheck<'q>`):
/// the driving provider's session lifetime must outlive the values, and
/// the type system enforces the pairing — a provider whose session
/// lifetime is shorter than the values it would cache is rejected at
/// compile time instead of silently caching values that outlive their
/// arena.  The arena need not be `'static`: `main` owns a stack `Bump`
/// and the provider borrows it for the duration of the run.
#[derive(Debug, Clone)]
pub struct FnBodyCheck<'input> {
    /// The checked body.  A rejected body folds to [`HirStmt::Error`] with
    /// the diagnostic pushed into `diagnostics` — mirroring the driver's
    /// emit-time behavior (`Err(diag)` → `items.push(HirStmt::Error)`),
    /// because a query value cannot carry a `Result`'s borrow-bearing
    /// error type without the same lifetime obligations.
    pub hir: crate::hir::hir::HirStmt<'input>,
    pub runtime_checks: Vec<(Symbol, Vec<crate::hir::hir::HirStmt<'input>>)>,
    /// `BTreeMap`, not `HashMap`: iteration is deterministic by construction,
    /// so `stable_hash` needs no per-call sort and no future reader can
    /// accidentally iterate it in an unspecified order.
    pub function_finally:
        std::collections::BTreeMap<Symbol, Vec<crate::hir::hir::HirStmt<'input>>>,
    pub pending_gadt: Vec<crate::hir::checker::PendingGadtConstruct>,
    pub residual_obligations: Vec<(
        crate::ast::Span,
        crate::hir::traits::solver::obligation::Predicate,
    )>,
    pub trait_obligations: Vec<(
        crate::ast::Span,
        crate::hir::traits::solver::obligation::Predicate,
    )>,
    pub diagnostics: Vec<crate::diagnostics::Diagnostic<'input>>,
}

/// Length-prefixed sequence hashing: the prefix keeps `[a, b]` distinct
/// from `[ab]` under a streaming hasher.
fn hash_seq<T>(h: &mut FxHasher, xs: &[T], mut f: impl FnMut(&mut FxHasher, &T)) {
    h.write_usize(xs.len());
    for x in xs {
        f(&mut *h, x);
    }
}

/// Presence-tagged option hashing.
fn hash_opt<T>(h: &mut FxHasher, o: &Option<T>, mut f: impl FnMut(&mut FxHasher, &T)) {
    h.write_u8(o.is_some() as u8);
    if let Some(x) = o {
        f(&mut *h, x);
    }
}

/// Span-sensitive leaf fallback: `Debug`-hashes a value whole, spans
/// included.  Sound-direction only — a change here can force a recompute,
/// never suppress one.  Used for leaves whose types are too large to walk
/// (`ast::Expr`, `ast::Type`) or that only occur in nested items
/// (`Attribute`, `Contract`, `TypeParam`, `TypeDefinition`, `TraitMethod`,
/// `AssociatedType`, `ImplMethod`).
fn hash_debug(h: &mut FxHasher, x: &impl std::fmt::Debug) {
    let _ = std::fmt::write(&mut HashWriter(h), format_args!("{x:?}"));
}

/// Structured hash for an `ast::Expr` LEAF (literal-type expressions,
/// `with default` / `validate` modifiers, array sizes).  These positions
/// previously went through `hash_debug`, whose `Debug` output embeds spans,
/// so a comment-only edit inside the leaf flipped the type's fingerprint —
/// a real false-invalidation source in editor-driven sessions.  The typed
/// fast paths here hash NO span; every remaining shape is handed to the
/// fully span-free [`hash_ast_expr`] walker, so no leaf position can smuggle
/// a span into the fingerprint.
fn hash_ast_expr_leaf(h: &mut FxHasher, e: &crate::ast::Expr<'_>) {
    match e {
        crate::ast::Expr::Literal(lit, span) => {
            h.write_u8(0);
            hash_literal(h, lit);
        }
        crate::ast::Expr::Ident(sym, span) => {
            h.write_u8(1);
            sym.hash(h);
        }
        crate::ast::Expr::BinaryOp {
            left,
            op,
            right,
            span,
        } => {
            h.write_u8(2);
            hash_ast_expr_leaf(h, left);
            h.write_u8(*op as u8);
            hash_ast_expr_leaf(h, right);
        }
        crate::ast::Expr::UnaryOp { op, expr, span } => {
            h.write_u8(3);
            h.write_u8(*op as u8);
            hash_ast_expr_leaf(h, expr);
        }
        crate::ast::Expr::Path(syms, span) => {
            h.write_u8(4);
            hash_seq(h, syms, |h, s| s.hash(h));
        }
        // #5: `hash_ast_expr` is span-free for every shape, so the fallback
        // no longer reintroduces span sensitivity; `hash_debug` did.
        other => hash_ast_expr(h, other),
    }
}

// ── Typed leaf hashers ───────────────────────────────────────────────
//
// Type-driven replacements for `hash_debug` on the leaf types that only
// occur as nested items (`Attribute`, `Contract`, `TypeParam`,
// `TraitMethod`, `AssociatedType`, `ImplMethod`) and on `ast::Type`.
// `hash_debug` paid fmt vtable dispatch + a `Debug`-format round trip and
// hashed structural padding fields; these walk the fields directly.
//
// SPANS ARE NOT HASHED: every `Span` field of these AST shapes is
// diagnostic-only, so the typed hashers skip them entirely.
// A comment/whitespace edit that merely shifts byte offsets cannot flip a
// fingerprint through this layer — the two-channel trivia invariance the
// `CheckFnBody` hash achieves for HIR bodies now holds here too.  The only
// remaining span sensitivity is the `hash_debug` fallback, whose `Debug`
// output embeds spans; it is sound-direction only (forces a recompute,
// never suppresses one).
//
// Any change to the hashed field set or its ORDER is
// a fingerprint change for every value carrying these types (sound
// direction: one bulk recompute, never a suppressed one).
//
// NOT FULLY SPAN-FREE, deliberately: `hash_ast_type`'s
// `T::Expr` arm routes through `hash_ast_expr_leaf`, which covers
// `Literal`/`Ident`/`BinaryOp`/`UnaryOp`/`Path` span-free but falls back to
// `hash_debug` for anything more complex.  That fallback is sound-direction
// (over-invalidation) — do not claim `hash_ast_type` is span-free for
// arbitrary `T::Expr` leaves, and do not try to "fix" it without a typed
// expression walker (which this module deliberately does not have).
//
// Deliberately NOT typed here: `ast::Expr` — walking it would duplicate the
// expression grammar into hashers for no measured win, so `Expr` internals
// stay on the `hash_debug` fallback wherever they appear as a leaf.
// `TypeDefinition` HAS since been typed (`hash_type_definition` below): type
// definitions are frequent in real programs, and a comment/whitespace edit
// inside a definition body must not invalidate dependents (the two-channel
// story `CheckFnBody`'s hash already achieves for functions).

fn hash_ast_type(h: &mut FxHasher, ty: &crate::ast::Type<'_>) {
    use crate::ast::Type as T;
    match ty {
        T::Path(syms, span) => {
            h.write_u8(0);
            hash_seq(h, syms, |h, s| s.hash(h));
        }
        T::Generic(base, args, span) => {
            h.write_u8(1);
            hash_ast_type(h, base);
            hash_seq(h, args, hash_generic_arg);
        }
        T::Reference {
            inner,
            mutable,
            lifetime,
            span,
        } => {
            h.write_u8(2);
            hash_ast_type(h, inner);
            h.write_u8(*mutable as u8);
            hash_opt(h, lifetime, |h, s| s.hash(h));
        }
        T::Pointer(inner, span) => {
            h.write_u8(3);
            hash_ast_type(h, inner);
        }
        T::Slice(inner, span) => {
            h.write_u8(4);
            hash_ast_type(h, inner);
        }
        T::Array(inner, size, span) => {
            h.write_u8(5);
            hash_ast_type(h, inner);
            hash_ast_expr_leaf(h, size);
        }
        T::Tuple(tys, span) => {
            h.write_u8(6);
            hash_seq(h, tys, hash_ast_type);
        }
        T::Function { params, ret, span } => {
            h.write_u8(7);
            hash_seq(h, params, hash_ast_type);
            hash_ast_type(h, ret);
        }
        T::Projection {
            impl_type,
            trait_path,
            assoc_name,
            span,
        } => {
            h.write_u8(8);
            hash_ast_type(h, impl_type);
            hash_ast_type(h, trait_path);
            assoc_name.hash(h);
        }
        T::DynTrait(ts, span) => {
            h.write_u8(9);
            hash_seq(h, ts, hash_ast_type);
        }
        T::Forall {
            lifetime,
            body,
            span,
        } => {
            h.write_u8(10);
            lifetime.hash(h);
            hash_ast_type(h, body);
        }
        T::Exists {
            name,
            base,
            invariant,
            span,
        } => {
            h.write_u8(11);
            name.hash(h);
            hash_ast_type(h, base);
            hash_ast_expr_leaf(h, invariant);
        }
        T::WhereShorthand {
            base,
            invariant,
            span,
        } => {
            h.write_u8(12);
            hash_ast_type(h, base);
            hash_ast_expr_leaf(h, invariant);
        }
        T::Literal(e, span) => {
            h.write_u8(13);
            hash_ast_expr_leaf(h, e);
        }
        T::Never(span) => {
            h.write_u8(14);
        }
        T::Union(ts, span) => {
            h.write_u8(15);
            hash_seq(h, ts, hash_ast_type);
        }
        T::Expr(e, span) => {
            h.write_u8(16);
            hash_ast_expr_leaf(h, e);
        }
        T::Regex(s, span) => {
            h.write_u8(17);
            s.hash(h);
        }
        T::Error(span) => {
            h.write_u8(18);
        }
    }
}

fn hash_generic_arg(h: &mut FxHasher, ga: &crate::ast::GenericArg<'_>) {
    use crate::ast::GenericArg as G;
    match ga {
        G::Positional(ty) => {
            h.write_u8(0);
            hash_ast_type(h, ty);
        }
        G::Named(name, ty) => {
            h.write_u8(1);
            name.hash(h);
            hash_ast_type(h, ty);
        }
        G::Const(ac) => {
            h.write_u8(2);
            // `ast::Expr` leaf: span-free on covered shapes.
            hash_ast_expr_leaf(h, ac.value);
        }
    }
}

/// Hash one AST attribute.
///
/// Precondition: `Attribute` never carries doc payloads.  Doc text lives in
/// the separate `doc` field (stripped by the checker); if a future parser
/// routes `#[doc]`-style text here, a doc-only edit would flip the semantic
/// fingerprint through this path and defeat the two-channel invariance.
fn hash_attribute(h: &mut FxHasher, a: &crate::ast::Attribute<'_>) {
    a.name.hash(h);
    hash_seq(h, &a.args, |h, e| hash_ast_expr_leaf(h, e));
    hash_seq(h, &a.named_args, |h, (name, e)| {
        name.hash(h);
        hash_ast_expr_leaf(h, e);
    });
}

fn hash_ast_pattern(h: &mut FxHasher, p: &crate::ast::Pattern<'_>) {
    use crate::ast::Pattern as P;
    match p {
        P::Wildcard(span) => {
            h.write_u8(0);
        }
        P::Ident(name, span) => {
            h.write_u8(1);
            name.hash(h);
        }
        P::Literal(e, span) => {
            h.write_u8(2);
            // Literal patterns are expression leaves: span-free where the
            // leaf hasher covers the shape.
            hash_ast_expr_leaf(h, e);
        }
        P::Tuple(ps, span) => {
            h.write_u8(3);
            hash_seq(h, ps, hash_ast_pattern);
        }
        P::Struct {
            path,
            fields,
            rest,
            span,
        } => {
            h.write_u8(4);
            hash_seq(h, path, |h, s| s.hash(h));
            hash_seq(h, fields, |h, (name, sub)| {
                name.hash(h);
                hash_ast_pattern(h, sub);
            });
            h.write_u8(*rest as u8);
        }
        P::Enum {
            path,
            variant,
            inner,
            span,
        } => {
            h.write_u8(5);
            hash_seq(h, path, |h, s| s.hash(h));
            variant.hash(h);
            hash_opt(h, inner, |h, p| hash_ast_pattern(h, p));
        }
        P::Annotated { inner, ty, span } => {
            h.write_u8(6);
            hash_ast_pattern(h, inner);
            hash_ast_type(h, ty);
        }
        P::Or(ps, span) => {
            h.write_u8(7);
            hash_seq(h, ps, hash_ast_pattern);
        }
        P::Slice(before, mid, after, span) => {
            h.write_u8(8);
            hash_seq(h, before, hash_ast_pattern);
            hash_opt(h, mid, |h, p| hash_ast_pattern(h, p));
            hash_seq(h, after, hash_ast_pattern);
        }
        P::Error(span) => {
            h.write_u8(9);
        }
    }
}

fn hash_ensures_target(h: &mut FxHasher, t: &crate::ast::EnsuresTarget<'_>) {
    use crate::ast::EnsuresTarget as E;
    match t {
        E::Unconditional => h.write_u8(0),
        E::OnOk(p) => {
            h.write_u8(1);
            hash_opt(h, p, hash_ast_pattern);
        }
        E::OnErr(p) => {
            h.write_u8(2);
            hash_opt(h, p, hash_ast_pattern);
        }
        E::OnTimeout => h.write_u8(3),
        E::OnCancel => h.write_u8(4),
    }
}

fn hash_contract(h: &mut FxHasher, c: &crate::ast::Contract<'_>) {
    use crate::ast::Contract as C;
    match c {
        C::Requires(e, span) => {
            h.write_u8(0);
            hash_ast_expr_leaf(h, e);
        }
        C::Ensures {
            expr,
            span,
            target,
            labels,
        } => {
            h.write_u8(1);
            hash_ast_expr_leaf(h, expr);
            hash_ensures_target(h, target);
            hash_seq(h, labels, |h, s| s.hash(h));
        }
        C::Invariant(e, span) => {
            h.write_u8(2);
            hash_ast_expr_leaf(h, e);
        }
        C::Decreases(e, span) => {
            h.write_u8(3);
            hash_ast_expr_leaf(h, e);
        }
        C::Terminates(e, span) => {
            h.write_u8(4);
            hash_ast_expr_leaf(h, e);
        }
    }
}

fn hash_type_param(h: &mut FxHasher, tp: &crate::ast::TypeParam<'_>) {
    tp.name.hash(h);
    hash_seq(h, tp.bounds, hash_ast_type);
    match tp.kind {
        crate::ast::TypeParamKind::Type => h.write_u8(0),
        crate::ast::TypeParamKind::Lifetime => h.write_u8(1),
        crate::ast::TypeParamKind::Const { ty, default } => {
            h.write_u8(2);
            hash_ast_type(h, &ty);
            hash_opt(h, &default, |h, e| hash_ast_expr_leaf(h, e));
        }
    }
}

fn hash_ast_param(h: &mut FxHasher, p: &crate::ast::Param<'_>) {
    p.name.hash(h);
    hash_opt(h, &p.ty, hash_ast_type);
    hash_opt(h, &p.default, |h, e| hash_ast_expr_leaf(h, e));
}

fn hash_trait_method(h: &mut FxHasher, m: &crate::ast::TraitMethod<'_>) {
    // COVERAGE NOTE: if `TraitMethod` gains a default body, where-clause, or
    // any field beyond name/params/return_type/span, that field MUST be
    // hashed here — otherwise a change to a trait method's default
    // implementation would not invalidate dependent queries.  The
    // destructuring pattern below makes such an addition a compile error.
    let crate::ast::TraitMethod {
        name,
        params,
        return_type,
        span: _,
    } = m;
    name.hash(h);
    hash_seq(h, params, hash_ast_param);
    hash_ast_type(h, return_type);
}

fn hash_associated_type(h: &mut FxHasher, a: &crate::ast::AssociatedType<'_>) {
    // COVERAGE NOTE: same obligation as `hash_trait_method` — if
    // `AssociatedType` gains a where-clause or other field beyond
    // name/lifetime_params/default/span, hash it here.  The destructuring
    // pattern makes that a compile error.
    let crate::ast::AssociatedType {
        name,
        lifetime_params,
        default,
        span: _,
    } = a;
    name.hash(h);
    hash_seq(h, lifetime_params, |h, s| s.hash(h));
    hash_opt(h, default, hash_ast_type);
}

/// Typed, span-free hasher for [`crate::ast::Item`] — the AST counterpart of
/// the HIR item walkers, reached from `hash_ast_stmt`'s `Stmt::Item` arm.
///
/// This replaces the former `hash_debug(item)` fallback,
/// whose `Debug` output embeds spans: a comment or whitespace edit inside a
/// nested `def`/`type` used to flip the enclosing body fingerprint, which
/// made the "two-channel trivia invariance" claim untrue for any body
/// containing a nested item.
///
/// COVERAGE NOTE: the match is EXHAUSTIVE over `Item`, so a new variant is a
/// compile error here (forcing a hashing decision).  Within each arm every
/// semantic field is hashed; `span` and `doc` are trivia and are skipped.
fn hash_ast_item(h: &mut FxHasher, item: &crate::ast::Item<'_>) {
    use crate::ast::Item as I;
    match item {
        I::FunctionDef {
            span: _,
            attributes,
            contracts,
            doc: _,
            name,
            params,
            return_type,
            body,
            type_params,
            where_clause,
            finally,
            is_comptime,
            is_async,
        } => {
            h.write_u8(0);
            hash_seq(h, attributes, hash_attribute);
            hash_seq(h, contracts, hash_contract);
            name.hash(h);
            hash_seq(h, params, hash_ast_param);
            hash_opt(h, return_type, hash_ast_type);
            hash_opt(h, body, |h, b| hash_ast_stmts(h, b));
            hash_seq(h, type_params, hash_type_param);
            hash_where_clause(h, where_clause);
            hash_opt(h, finally, |h, b| hash_ast_stmts(h, b));
            h.write_u8(*is_comptime as u8);
            h.write_u8(*is_async as u8);
        }
        I::TypeDef {
            span: _,
            attributes,
            doc: _,
            name,
            params,
            definition,
            contracts,
        } => {
            h.write_u8(1);
            hash_seq(h, attributes, hash_attribute);
            name.hash(h);
            hash_seq(h, params, hash_type_param);
            hash_type_definition(h, definition);
            hash_seq(h, contracts, hash_contract);
        }
        I::TraitDef {
            span: _,
            attributes,
            doc: _,
            name,
            methods,
            associated_types,
        } => {
            h.write_u8(2);
            hash_seq(h, attributes, hash_attribute);
            name.hash(h);
            hash_seq(h, methods, hash_trait_method);
            hash_seq(h, associated_types, hash_associated_type);
        }
        I::ImplBlock {
            span: _,
            attributes,
            trait_path,
            for_type,
            methods,
            associated_types,
            where_clause,
            type_params,
        } => {
            h.write_u8(3);
            hash_seq(h, attributes, hash_attribute);
            hash_opt(h, trait_path, |h, t| hash_ast_type(h, t));
            hash_ast_type(h, for_type);
            hash_seq(h, methods, hash_impl_method);
            hash_seq(h, associated_types, hash_associated_type);
            hash_where_clause(h, where_clause);
            hash_seq(h, type_params, hash_type_param);
        }
        I::Import {
            path,
            items,
            alias,
            span: _,
        } => {
            h.write_u8(4);
            hash_seq(h, path, |h, s| s.hash(h));
            hash_opt(h, items, |h, xs| hash_seq(h, xs, |h, s| s.hash(h)));
            hash_opt(h, alias, |h, s| s.hash(h));
        }
        I::ExternFunction {
            abi,
            name,
            params,
            return_type,
            span: _,
            attributes,
        } => {
            h.write_u8(5);
            abi.hash(h);
            name.hash(h);
            hash_seq(h, params, hash_ast_param);
            hash_ast_type(h, return_type);
            hash_seq(h, attributes, hash_attribute);
        }
        I::Constraint {
            name,
            params,
            predicates,
            span: _,
        } => {
            h.write_u8(6);
            name.hash(h);
            hash_seq(h, params, hash_type_param);
            hash_seq(h, predicates, hash_where_predicate);
        }
        I::Edition(name, _) => {
            h.write_u8(7);
            name.hash(h);
        }
        I::LayoutDef {
            name,
            attributes,
            span: _,
        } => {
            h.write_u8(8);
            name.hash(h);
            hash_seq(h, attributes, hash_attribute);
        }
        I::ComptimeBlock {
            captures,
            trusted,
            attributes,
            body,
            span: _,
        } => {
            h.write_u8(9);
            hash_seq(h, captures, |h, (sym, _)| sym.hash(h));
            h.write_u8(*trusted as u8);
            hash_seq(h, attributes, hash_attribute);
            hash_ast_stmts(h, body);
        }
        I::Generate {
            attributes,
            for_type,
            body,
            span: _,
        } => {
            h.write_u8(10);
            hash_seq(h, attributes, hash_attribute);
            hash_ast_type(h, for_type);
            hash_ast_stmts(h, body);
        }
        I::Error(_) => h.write_u8(11),
    }
}

fn hash_where_predicate(h: &mut FxHasher, p: &crate::ast::WherePredicate<'_>) {
    hash_ast_type(h, &p.ty);
    hash_seq(h, p.bounds, hash_ast_type);
}

/// Hash a where-clause span-free.  `None` and an empty clause hash
/// differently on purpose: the parse shape is part of the semantic
/// fingerprint (it is what the E104 generality exemption keys off).
fn hash_where_clause(h: &mut FxHasher, wc: &Option<crate::ast::WhereClause<'_>>) {
    match wc {
        None => h.write_u8(0),
        Some(wc) => {
            h.write_u8(1);
            hash_seq(h, wc.predicates, hash_where_predicate);
            hash_seq(h, wc.equalities, |h, e| {
                hash_ast_type(h, &e.left);
                hash_ast_type(h, &e.right);
            });
            hash_seq(h, wc.lifetime_outlives, |h, (a, bs)| {
                a.hash(h);
                hash_seq(h, *bs, |h, s| s.hash(h));
            });
        }
    }
}

/// AST analogue of [`hash_stmt`] (which walks checked HIR).  Raw
/// impl-method bodies are un-checked `ast::Stmt`, so the HIR walker cannot
/// be reused.  Every span is skipped, including inside nested items
///: the whole
/// walker is now trivia-invariant.
fn hash_ast_stmts(h: &mut FxHasher, ss: &[crate::ast::Stmt<'_>]) {
    hash_seq(h, ss, hash_ast_stmt);
}

fn hash_ast_stmt(h: &mut FxHasher, s: &crate::ast::Stmt<'_>) {
    use crate::ast::Stmt as S;
    match s {
        S::VariableDef {
            kind,
            mutable,
            name,
            pattern,
            ty,
            value,
            else_branch,
            span: _,
            attributes,
            // `doc` is trivia: the two-channel invariant forbids hashing it.
            doc: _,
            type_captures,
            type_modifiers,
        } => {
            h.write_u8(0);
            h.write_u8(*kind as u8);
            h.write_u8(*mutable as u8);
            hash_opt(h, name, |h, n| n.hash(h));
            hash_opt(h, pattern, hash_ast_pattern);
            hash_opt(h, ty, hash_ast_type);
            hash_opt(h, value, hash_ast_expr);
            hash_opt(h, else_branch, |h, b| hash_ast_stmts(h, b));
            hash_seq(h, attributes, hash_attribute);
            hash_seq(h, type_captures, hash_type_param);
            hash_seq(h, type_modifiers, hash_type_modifier);
        }
        S::Item(item) => {
            h.write_u8(1);
            hash_ast_item(h, item);
        }
        S::Expression(e) => {
            h.write_u8(2);
            hash_ast_expr(h, e);
        }
        S::If {
            cond,
            then_branch,
            else_branch,
            span: _,
        } => {
            h.write_u8(3);
            hash_ast_expr(h, cond);
            hash_ast_stmts(h, then_branch);
            hash_opt(h, else_branch, |h, b| hash_ast_stmts(h, b));
        }
        S::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            span: _,
        } => {
            h.write_u8(4);
            hash_ast_pattern(h, pattern);
            hash_ast_expr(h, scrutinee);
            hash_ast_stmts(h, then_branch);
            hash_opt(h, else_branch, |h, b| hash_ast_stmts(h, b));
        }
        S::While {
            label,
            cond,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(5);
            hash_opt(h, label, |h, l| l.hash(h));
            hash_ast_expr(h, cond);
            hash_ast_stmts(h, body);
            hash_opt(h, invariant, hash_ast_expr);
            hash_opt(h, decreases, hash_ast_expr);
        }
        S::WhileLet {
            label,
            pattern,
            scrutinee,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(6);
            hash_opt(h, label, |h, l| l.hash(h));
            hash_ast_pattern(h, pattern);
            hash_ast_expr(h, scrutinee);
            hash_ast_stmts(h, body);
            hash_opt(h, invariant, hash_ast_expr);
            hash_opt(h, decreases, hash_ast_expr);
        }
        S::For {
            label,
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(7);
            hash_opt(h, label, |h, l| l.hash(h));
            hash_ast_pattern(h, pattern);
            hash_ast_expr(h, iterable);
            hash_ast_stmts(h, body);
            hash_opt(h, invariant, hash_ast_expr);
            hash_opt(h, decreases, hash_ast_expr);
        }
        S::Loop { label, body, span: _ } => {
            h.write_u8(8);
            hash_opt(h, label, |h, l| l.hash(h));
            hash_ast_stmts(h, body);
        }
        S::Leave { label, span: _ } => {
            h.write_u8(9);
            hash_opt(h, label, |h, l| l.hash(h));
        }
        S::Continue { label, span: _ } => {
            h.write_u8(10);
            hash_opt(h, label, |h, l| l.hash(h));
        }
        S::Return {
            value,
            labels,
            span: _,
        } => {
            h.write_u8(11);
            hash_opt(h, value, hash_ast_expr);
            hash_seq(h, labels, |h, l| l.hash(h));
        }
        S::Assign {
            target,
            op,
            value,
            span: _,
        } => {
            h.write_u8(12);
            hash_ast_expr(h, target.expr());
            h.write_u8(op.is_some() as u8);
            if let Some(op) = op {
                h.write_u8(*op as u8);
            }
            hash_ast_expr(h, value);
        }
        S::ScopeCleanup {
            name,
            when_condition,
            body,
            propagates,
            overrides,
            span: _,
        } => {
            h.write_u8(13);
            name.hash(h);
            hash_opt(h, when_condition, |h, e| hash_ast_expr(h, e));
            hash_ast_stmts(h, body);
            h.write_u8(*propagates as u8);
            h.write_u8(*overrides as u8);
        }
        S::Trigger { name, span: _ } => {
            h.write_u8(14);
            name.hash(h);
        }
        S::Unsafe { body, span: _ } => {
            h.write_u8(15);
            hash_ast_stmts(h, body);
        }
        S::GhostVariableDef { inner, span: _ } => {
            h.write_u8(16);
            hash_ast_stmt(h, inner);
        }
        S::Isolate {
            attributes,
            body,
            span: _,
        } => {
            h.write_u8(17);
            hash_seq(h, attributes, hash_attribute);
            hash_ast_stmts(h, body);
        }
        S::Error(_) => h.write_u8(18),
    }
}

fn hash_ast_expr(h: &mut FxHasher, e: &crate::ast::Expr<'_>) {
    use crate::ast::Expr as E;
    match e {
        E::Literal(lit, _) => {
            h.write_u8(0);
            hash_literal(h, lit);
        }
        E::Ident(sym, _) => {
            h.write_u8(1);
            sym.hash(h);
        }
        E::TypeAnnotated { expr, ty, span: _ } => {
            h.write_u8(2);
            hash_ast_expr(h, expr);
            hash_ast_type(h, ty);
        }
        E::BinaryOp {
            left,
            op,
            right,
            span: _,
        } => {
            h.write_u8(3);
            hash_ast_expr(h, left);
            h.write_u8(*op as u8);
            hash_ast_expr(h, right);
        }
        E::UnaryOp { op, expr, span: _ } => {
            h.write_u8(4);
            h.write_u8(*op as u8);
            hash_ast_expr(h, expr);
        }
        E::Call {
            callee,
            args,
            comptime,
            span: _,
        } => {
            h.write_u8(5);
            hash_ast_expr(h, callee);
            hash_seq(h, args, hash_ast_expr);
            h.write_u8(*comptime as u8);
        }
        E::Index {
            base,
            index,
            span: _,
        } => {
            h.write_u8(6);
            hash_ast_expr(h, base);
            hash_ast_expr(h, index);
        }
        E::FieldAccess {
            base,
            field,
            span: _,
        } => {
            h.write_u8(7);
            hash_ast_expr(h, base);
            field.hash(h);
        }
        E::AttrAccess {
            base,
            attr,
            span: _,
        } => {
            h.write_u8(8);
            hash_ast_expr(h, base);
            attr.hash(h);
        }
        E::Cast {
            expr,
            ty,
            safe,
            rounding,
            span: _,
        } => {
            h.write_u8(9);
            hash_ast_expr(h, expr);
            hash_ast_type(h, ty);
            h.write_u8(*safe as u8);
            h.write_u8(rounding.is_some() as u8);
            if let Some(r) = rounding {
                h.write_u8(*r as u8);
            }
        }
        E::Range {
            start,
            end,
            inclusive,
            span: _,
        } => {
            h.write_u8(10);
            hash_opt(h, start, |h, e| hash_ast_expr(h, e));
            hash_opt(h, end, |h, e| hash_ast_expr(h, e));
            h.write_u8(*inclusive as u8);
        }
        E::StructLit { path, fields, span: _ } => {
            h.write_u8(11);
            hash_seq(h, path, |h, s| s.hash(h));
            hash_seq(h, fields, |h, (name, e)| {
                name.hash(h);
                hash_ast_expr(h, e);
            });
        }
        E::EnumLit {
            path,
            variant,
            payload,
            span: _,
        } => {
            h.write_u8(12);
            hash_seq(h, path, |h, s| s.hash(h));
            variant.hash(h);
            hash_opt(h, payload, |h, e| hash_ast_expr(h, e));
        }
        E::Move(expr, _) => {
            h.write_u8(13);
            hash_ast_expr(h, expr);
        }
        E::Path(syms, _) => {
            h.write_u8(14);
            hash_seq(h, syms, |h, s| s.hash(h));
        }
        E::Tuple(es, _) => {
            h.write_u8(15);
            hash_seq(h, es, hash_ast_expr);
        }
        E::Array(es, _) => {
            h.write_u8(16);
            hash_seq(h, es, hash_ast_expr);
        }
        E::Closure {
            params,
            return_type,
            captures,
            body,
            span: _,
        } => {
            h.write_u8(17);
            hash_seq(h, params, hash_ast_param);
            hash_opt(h, return_type, hash_ast_type);
            hash_seq(h, captures, hash_capture);
            hash_ast_stmts(h, body);
        }
        E::Try { expr, span: _ } => {
            h.write_u8(18);
            hash_ast_expr(h, expr);
        }
        E::UnsafeBlock { body, span: _ } => {
            h.write_u8(19);
            hash_ast_stmts(h, body);
        }
        E::Catch {
            expr,
            branches,
            span: _,
        } => {
            h.write_u8(20);
            hash_ast_expr(h, expr);
            hash_seq(h, branches, hash_ast_catch_branch);
        }
        E::LeaveWith {
            expr,
            is_return,
            span: _,
        } => {
            h.write_u8(21);
            hash_ast_expr(h, expr);
            h.write_u8(*is_return as u8);
        }
        E::Await { expr, span: _ } => {
            h.write_u8(22);
            hash_ast_expr(h, expr);
        }
        E::If {
            cond,
            then_branch,
            else_branch,
            is_expression,
            span: _,
        } => {
            h.write_u8(23);
            hash_ast_expr(h, cond);
            hash_ast_stmts(h, then_branch);
            hash_opt(h, else_branch, |h, b| hash_ast_stmts(h, b));
            h.write_u8(*is_expression as u8);
        }
        E::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            is_expression,
            span: _,
        } => {
            h.write_u8(24);
            hash_ast_pattern(h, pattern);
            hash_ast_expr(h, scrutinee);
            hash_ast_stmts(h, then_branch);
            hash_opt(h, else_branch, |h, b| hash_ast_stmts(h, b));
            h.write_u8(*is_expression as u8);
        }
        E::Match {
            scrutinee,
            arms,
            span: _,
        } => {
            h.write_u8(25);
            hash_ast_expr(h, scrutinee);
            hash_seq(h, arms, hash_ast_match_arm);
        }
        E::Block(stmts, _) => {
            h.write_u8(26);
            hash_ast_stmts(h, stmts);
        }
        E::PolyBox {
            expr,
            scheme,
            span: _,
        } => {
            h.write_u8(27);
            hash_ast_expr(h, expr);
            hash_opt(h, scheme, hash_ast_type_scheme);
        }
        E::PolyUnbox {
            expr,
            scheme,
            span: _,
        } => {
            h.write_u8(28);
            hash_ast_expr(h, expr);
            hash_opt(h, scheme, hash_ast_type_scheme);
        }
        E::Quantified {
            quantifier,
            binder,
            range,
            body,
            span: _,
        } => {
            h.write_u8(29);
            h.write_u8(*quantifier as u8);
            binder.hash(h);
            hash_ast_expr(h, range);
            hash_ast_expr(h, body);
        }
        E::Old(expr, _) => {
            h.write_u8(30);
            hash_ast_expr(h, expr);
        }
        E::Task { body, span: _ } => {
            h.write_u8(31);
            hash_ast_stmts(h, body);
        }
        E::TypeInfo(ty, _) => {
            h.write_u8(32);
            hash_ast_type(h, ty);
        }
        E::LayoutOf(ty, _) => {
            h.write_u8(33);
            hash_ast_type(h, ty);
        }
        E::CompileError(msg, _) => {
            h.write_u8(34);
            msg.hash(h);
        }
        E::Error(_) => h.write_u8(35),
    }
}

fn hash_ast_catch_branch(h: &mut FxHasher, b: &crate::ast::CatchBranch<'_>) {
    hash_ast_pattern(h, &b.pattern);
    hash_opt(h, &b.bind, |h, s| s.hash(h));
    hash_ast_stmts(h, b.body);
}

fn hash_ast_match_arm(h: &mut FxHasher, a: &crate::ast::MatchArm<'_>) {
    hash_ast_pattern(h, &a.pattern);
    hash_opt(h, &a.guard, |h, e| hash_ast_expr(h, e));
    hash_ast_expr(h, &a.body);
}

fn hash_ast_type_scheme(h: &mut FxHasher, s: &crate::ast::TypeScheme<'_>) {
    // Quantifier SPANS are diagnostic-only; hash the bound names.
    hash_seq(h, s.quantifiers, |h, (_, sym)| sym.hash(h));
    hash_ast_type(h, s.body);
}

fn hash_impl_method(h: &mut FxHasher, m: &crate::ast::ImplMethod<'_>) {
    // COVERAGE NOTE: the destructuring pattern below is
    // INTENTIONAL — adding a field to `ImplMethod` becomes a compile error
    // here instead of a silently unhashed semantic field.  Only `span` is
    // skipped (trivia).  Verified against the current AST definition
    // (name, attributes, params, return_type, body, span); there is no
    // where-clause / type-param / contract set on an impl method today.
    let crate::ast::ImplMethod {
        name,
        attributes,
        params,
        return_type,
        body,
        span: _,
    } = m;
    name.hash(h);
    hash_seq(h, attributes, hash_attribute);
    hash_seq(h, params, hash_ast_param);
    hash_ast_type(h, return_type);
    // Raw impl-method bodies are AST (not HIR): use the span-free AST
    // walker above.
    hash_opt(h, body, |h, b| hash_ast_stmts(h, b));
}

/// Typed hasher for `ast::TypeDefinition` — replaces the previous
/// `hash_debug(definition)` fallback, whose `Debug` output embedded spans
/// and defeated the two-channel trivia invariance for every type
/// definition (a comment/whitespace edit inside a definition body flipped
/// the fingerprint).  This walker hashes NO span, so definition bodies are
/// now trivia-invariant — the same win `CheckFnBody`'s hash achieves for
/// function bodies.
///
/// `ast::Expr` leaves (field defaults, `Default`/`Validate` modifiers) go
/// through `hash_ast_expr_leaf` — span-free on its covered shapes,
/// `hash_debug` fallback otherwise (see the typed-leaf block note above).
///
/// Impl-method bodies are hashed SPAN-FREE via `hash_ast_stmts`
/// (`hash_impl_method` above) — the previous "KNOWN GAP" note claiming they
/// went through `hash_debug` was stale: do not "restore"
/// `hash_debug` there, that would regress trivia invariance for every type
/// definition containing an impl block.  Nested items inside a body now go
/// through `hash_ast_item` as well, so the walker is fully span-free.
///
/// SPANS ARE NOT HASHED, as for every typed AST hasher.
fn hash_type_definition(h: &mut FxHasher, d: &crate::ast::TypeDefinition<'_>) {
    use crate::ast::TypeDefinition as TD;
    match d {
        TD::Struct(fields, modifiers) => {
            h.write_u8(0);
            hash_seq(h, fields, hash_struct_field);
            hash_seq(h, modifiers, hash_type_modifier);
        }
        TD::Enum(variants, repr, modifiers) => {
            h.write_u8(1);
            hash_seq(h, variants, hash_enum_variant);
            hash_opt(h, repr, |h, s| s.hash(h));
            hash_seq(h, modifiers, hash_type_modifier);
        }
        TD::TraitDef {
            methods,
            associated_types,
        } => {
            h.write_u8(2);
            hash_seq(h, methods, hash_trait_method);
            hash_seq(h, associated_types, hash_associated_type);
        }
        TD::ImplBlock {
            trait_path,
            for_type,
            methods,
        } => {
            h.write_u8(3);
            hash_opt(h, &trait_path, |h, s| hash_seq(h, s, |h, s| s.hash(h)));
            hash_ast_type(h, for_type);
            hash_seq(h, methods, hash_impl_method);
        }
        TD::Constraint(tys) => {
            h.write_u8(4);
            hash_seq(h, tys, hash_ast_type);
        }
        TD::Alias(ty, modifiers) => {
            h.write_u8(5);
            hash_ast_type(h, ty);
            hash_seq(h, modifiers, hash_type_modifier);
        }
        TD::Opaque(ty, modifiers) => {
            h.write_u8(6);
            hash_ast_type(h, ty);
            hash_seq(h, modifiers, hash_type_modifier);
        }
    }
}

fn hash_struct_field(h: &mut FxHasher, f: &crate::ast::StructField<'_>) {
    // COVERAGE NOTE: field set verified against the current AST definition;
    // `default` is an `ast::Expr` leaf, hashed with the span-free leaf
    // hasher — a comment/whitespace edit in a default no
    // longer flips the enclosing type definition's fingerprint.
    f.name.hash(h);
    hash_ast_type(h, &f.ty);
    hash_opt(h, &f.default, |h, e| hash_ast_expr_leaf(h, e));
}

fn hash_enum_variant(h: &mut FxHasher, v: &crate::ast::EnumVariant<'_>) {
    // COVERAGE NOTE: field set verified against the current AST definition
    // (GADT `eq_spec` and `exists_params` are semantic — they MUST be
    // hashed, a `when` clause edit flips the fingerprint).
    v.name.hash(h);
    hash_opt(h, &v.payload, hash_ast_type);
    h.write_usize(v.eq_spec.len());
    for (name, ty) in &v.eq_spec {
        name.hash(h);
        hash_ast_type(h, ty);
    }
    hash_seq(h, &v.exists_params, |h, s| s.hash(h));
}

fn hash_type_modifier(h: &mut FxHasher, m: &crate::ast::TypeModifier<'_>) {
    use crate::ast::TypeModifier as M;
    match m {
        M::Overflow(p) => {
            h.write_u8(0);
            h.write_u8(*p as u8);
        }
        M::Default(e) => {
            h.write_u8(1);
            hash_ast_expr_leaf(h, e); // `ast::Expr` leaf
        }
        M::Validate(e) => {
            h.write_u8(2);
            hash_ast_expr_leaf(h, e); // `ast::Expr` leaf
        }
        M::NoDefault => h.write_u8(3),
    }
}

fn hash_literal(h: &mut FxHasher, lit: &crate::ast::Literal) {
    match lit {
        crate::ast::Literal::Int(n) => {
            h.write_u8(0);
            match n {
                crate::ast::IntLit::Small(v) => {
                    h.write_u8(0);
                    v.hash(h);
                }
                crate::ast::IntLit::Large(v) => {
                    h.write_u8(1);
                    v.hash(h);
                }
            }
        }
        crate::ast::Literal::Float(v) => {
            h.write_u8(1);
            // `to_bits`, NOT `PartialEq`-via-float-equality: `-0.0` and
            // `0.0` are `==` and hash the same under float semantics, but
            // as BIT PATTERNS they differ, and distinct NaN payloads are
            // `!=` to themselves yet hash distinctly.  Hash/eq being
            // consistent is not the goal here — `stable_hash` is a
            // CHANGE DETECTOR, not a map key, and both of these differences
            // are the CONSERVATIVE direction (they force a recompute that
            // would otherwise be skipped).  Do not "fix" this to `==`
            // semantics: that would make a `-0.0` → `0.0` edit invisible
            // and serve a stale fingerprint.
            h.write_u64(v.to_bits());
        }
        crate::ast::Literal::Char(v) => {
            h.write_u8(2);
            v.hash(h);
        }
        crate::ast::Literal::Byte(v) => {
            h.write_u8(3);
            v.hash(h);
        }
        crate::ast::Literal::String(v) => {
            h.write_u8(4);
            v.hash(h);
        }
        crate::ast::Literal::ByteString(v) => {
            h.write_u8(5);
            v.hash(h);
        }
        crate::ast::Literal::Bool(v) => {
            h.write_u8(6);
            v.hash(h);
        }
    }
}

fn hash_hir_param(h: &mut FxHasher, p: &crate::hir::hir::HirParam<'_>) {
    p.name.hash(h);
    p.ty.hash(h);
    // `default` is an AST fragment: the typed leaf hasher covers the
    // trivially-walkable shapes span-free; complex exprs fall back to
    // `hash_debug` (sound-direction over-invalidation only).
    hash_opt(h, &p.default, |h, e| hash_ast_expr_leaf(h, e));
}

fn hash_capture(h: &mut FxHasher, c: &crate::ast::Capture) {
    c.name.hash(h);
    h.write_u8(c.mode as u8);
}

fn hash_pattern(h: &mut FxHasher, p: &crate::hir::hir::HirPattern<'_>) {
    use crate::hir::hir::HirPattern as P;
    match p {
        P::Wildcard(_) => h.write_u8(0),
        P::Ident(name, ty, _) => {
            h.write_u8(1);
            name.hash(h);
            ty.hash(h);
        }
        P::Literal(e, _) => {
            h.write_u8(2);
            hash_expr(&mut *h, e);
        }
        P::Tuple(ps, _) => {
            h.write_u8(3);
            hash_seq(&mut *h, ps, hash_pattern);
        }
        P::Struct {
            path,
            fields,
            rest,
            span: _,
        } => {
            h.write_u8(4);
            hash_seq(&mut *h, path, |h, s| s.hash(h));
            hash_seq(&mut *h, fields, |h, (name, sub)| {
                name.hash(&mut *h);
                hash_pattern(&mut *h, sub);
            });
            h.write_u8(*rest as u8);
        }
        P::Enum {
            path,
            variant,
            inner,
            span: _,
        } => {
            h.write_u8(5);
            hash_seq(&mut *h, path, |h, s| s.hash(h));
            variant.hash(&mut *h);
            hash_opt(&mut *h, inner, |h, sub| hash_pattern(h, sub));
        }
        P::Or(ps, _) => {
            h.write_u8(6);
            hash_seq(&mut *h, ps, hash_pattern);
        }
        P::Slice(before, mid, after, _) => {
            h.write_u8(7);
            hash_seq(&mut *h, before, hash_pattern);
            hash_opt(&mut *h, mid, |h, sub| hash_pattern(h, sub));
            hash_seq(&mut *h, after, hash_pattern);
        }
        P::Error(_) => h.write_u8(8),
    }
}

/// Fingerprint one statement list.
///
/// The walk is exhaustive by design: a missed field is the unsound
/// (stale-serve) direction, so a cheap shape prefilter that skips the walk
/// would trade correctness for speed.  Defer such a prefilter until a
/// deterministic miss-path benchmark (`bench_miss_chain_hash`) shows the walk
/// dominates `stage`; the safe direction then is memoizing the hash on the
/// value, not weakening the walker.
fn hash_stmts(h: &mut FxHasher, ss: &[crate::hir::hir::HirStmt<'_>]) {
    hash_seq(h, ss, hash_stmt);
}

fn hash_expr(h: &mut FxHasher, e: &crate::hir::hir::HirExpr<'_>) {
    use crate::hir::hir::HirExpr as E;
    match e {
        E::Literal(lit, ty, _) => {
            h.write_u8(0);
            hash_literal(&mut *h, lit);
            ty.hash(&mut *h);
        }
        E::Ident(name, ty, _) => {
            h.write_u8(1);
            name.hash(h);
            ty.hash(h);
        }
        E::TypeAnnotated { expr, ty, span: _ } => {
            h.write_u8(2);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::BinaryOp {
            left,
            op,
            right,
            ty,
            span: _,
        } => {
            h.write_u8(3);
            hash_expr(&mut *h, left);
            h.write_u8(*op as u8);
            hash_expr(&mut *h, right);
            ty.hash(&mut *h);
        }
        E::UnaryOp {
            op,
            expr,
            ty,
            span: _,
        } => {
            h.write_u8(4);
            h.write_u8(*op as u8);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::Call {
            callee,
            args,
            comptime,
            ty,
            span: _,
        } => {
            h.write_u8(5);
            hash_expr(&mut *h, callee);
            hash_seq(&mut *h, args, hash_expr);
            h.write_u8(*comptime as u8);
            ty.hash(&mut *h);
        }
        E::Index {
            base,
            index,
            ty,
            span: _,
        } => {
            h.write_u8(6);
            hash_expr(&mut *h, base);
            hash_expr(&mut *h, index);
            ty.hash(&mut *h);
        }
        E::FieldAccess {
            base,
            field,
            ty,
            span: _,
        } => {
            h.write_u8(7);
            hash_expr(&mut *h, base);
            field.hash(&mut *h);
            ty.hash(&mut *h);
        }
        E::AttrAccess {
            base,
            attr,
            ty,
            span: _,
        } => {
            h.write_u8(8);
            hash_expr(&mut *h, base);
            attr.hash(&mut *h);
            ty.hash(&mut *h);
        }
        E::Cast {
            expr,
            ty,
            safe,
            rounding,
            span: _,
        } => {
            h.write_u8(9);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
            h.write_u8(*safe as u8);
            h.write_u8(rounding.is_some() as u8);
            if let Some(r) = rounding {
                h.write_u8(*r as u8);
            }
        }
        E::Range {
            start,
            end,
            inclusive,
            ty,
            span: _,
        } => {
            h.write_u8(10);
            hash_opt(&mut *h, start, |h, e| hash_expr(h, e));
            hash_opt(&mut *h, end, |h, e| hash_expr(h, e));
            h.write_u8(*inclusive as u8);
            ty.hash(&mut *h);
        }
        E::StructLit {
            path,
            fields,
            ty,
            span: _,
        } => {
            h.write_u8(11);
            hash_seq(&mut *h, path, |h, s| s.hash(h));
            hash_seq(&mut *h, fields, |h, (name, v)| {
                name.hash(&mut *h);
                hash_expr(&mut *h, v);
            });
            ty.hash(&mut *h);
        }
        E::EnumLit {
            path,
            variant,
            payload,
            ty,
            span: _,
        } => {
            h.write_u8(12);
            hash_seq(&mut *h, path, |h, s| s.hash(h));
            variant.hash(&mut *h);
            hash_opt(&mut *h, payload, |h, e| hash_expr(h, e));
            ty.hash(&mut *h);
        }
        E::Move(inner, ty, _) => {
            h.write_u8(13);
            hash_expr(&mut *h, inner);
            ty.hash(&mut *h);
        }
        E::Tuple(es, ty, _) => {
            h.write_u8(14);
            hash_seq(&mut *h, es, hash_expr);
            ty.hash(&mut *h);
        }
        E::Array(es, ty, _) => {
            h.write_u8(15);
            hash_seq(&mut *h, es, hash_expr);
            ty.hash(&mut *h);
        }
        E::Closure {
            params,
            return_type,
            captures,
            body,
            ty,
            span: _,
        } => {
            h.write_u8(16);
            hash_seq(&mut *h, params, hash_hir_param);
            return_type.hash(&mut *h);
            hash_seq(&mut *h, captures, hash_capture);
            hash_stmts(&mut *h, body);
            ty.hash(&mut *h);
        }
        E::Try { expr, ty, span: _ } => {
            h.write_u8(17);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::UnsafeBlock { body, ty, span: _ } => {
            h.write_u8(18);
            hash_stmts(&mut *h, body);
            ty.hash(&mut *h);
        }
        E::Catch {
            expr,
            branches,
            ty,
            span: _,
        } => {
            h.write_u8(19);
            hash_expr(&mut *h, expr);
            hash_seq(&mut *h, branches, |h, b| {
                hash_pattern(&mut *h, &b.pattern);
                hash_opt(&mut *h, &b.bind, |h, s| s.hash(h));
                hash_stmts(&mut *h, &b.body);
            });
            ty.hash(&mut *h);
        }
        E::LeaveWith { expr, ty, span: _ } => {
            h.write_u8(20);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::Return { value, ty, span: _ } => {
            h.write_u8(21);
            hash_expr(&mut *h, value);
            ty.hash(&mut *h);
        }
        E::Await { expr, ty, span: _ } => {
            h.write_u8(22);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::If {
            cond,
            then_branch,
            else_branch,
            is_expression,
            ty,
            span: _,
        } => {
            h.write_u8(23);
            hash_expr(&mut *h, cond);
            hash_stmts(&mut *h, then_branch);
            hash_opt(&mut *h, else_branch, |h, b| hash_stmts(h, b));
            h.write_u8(*is_expression as u8);
            ty.hash(&mut *h);
        }
        E::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ty,
            span: _,
        } => {
            h.write_u8(24);
            hash_pattern(&mut *h, pattern);
            hash_expr(&mut *h, scrutinee);
            hash_stmts(&mut *h, then_branch);
            hash_opt(&mut *h, else_branch, |h, b| hash_stmts(h, b));
            ty.hash(&mut *h);
        }
        E::Match {
            scrutinee,
            arms,
            ty,
            span: _,
        } => {
            h.write_u8(25);
            hash_expr(&mut *h, scrutinee);
            hash_seq(&mut *h, arms, |h, a| {
                hash_pattern(&mut *h, &a.pattern);
                hash_opt(&mut *h, &a.guard, |h, g| hash_expr(h, g));
                hash_expr(&mut *h, &a.body);
            });
            ty.hash(&mut *h);
        }
        E::Block(ss, ty, _) => {
            h.write_u8(26);
            hash_stmts(&mut *h, ss);
            ty.hash(&mut *h);
        }
        E::PolyBox { expr, ty, span: _ } => {
            h.write_u8(27);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::PolyUnbox { expr, ty, span: _ } => {
            h.write_u8(28);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::Quantified {
            quantifier,
            binder,
            range,
            body,
            ty,
            span: _,
        } => {
            h.write_u8(29);
            h.write_u8(*quantifier as u8);
            binder.hash(&mut *h);
            hash_expr(&mut *h, range);
            hash_expr(&mut *h, body);
            ty.hash(&mut *h);
        }
        E::Old { expr, ty, span: _ } => {
            h.write_u8(30);
            hash_expr(&mut *h, expr);
            ty.hash(&mut *h);
        }
        E::Task { block, ty, span: _ } => {
            h.write_u8(31);
            hash_stmts(&mut *h, block);
            ty.hash(&mut *h);
        }
        E::TypeInfo(ty, _) => {
            h.write_u8(32);
            ty.hash(h);
        }
        E::LayoutOf(ty, _) => {
            // Typed leaf hasher (`hash_ast_type` — spans skipped, so this
            // arm is trivia-invariant).  Do NOT "restore" `hash_debug`
            // here: that would reintroduce the fmt round-trip the typed
            // leaf hashers removed; see the typed-leaf note above.
            h.write_u8(33);
            hash_ast_type(&mut *h, ty);
        }
        E::CompileError(msg, _) => {
            h.write_u8(34);
            msg.hash(h);
        }
        E::Error(_) => h.write_u8(35),
    }
}

fn hash_stmt(h: &mut FxHasher, s: &crate::hir::hir::HirStmt<'_>) {
    use crate::hir::hir::HirStmt as S;
    match s {
        S::VariableDef {
            kind,
            mutable,
            name,
            pattern,
            ty,
            value,
            else_branch,
            span: _,
            type_captures,
        } => {
            h.write_u8(0);
            h.write_u8(*kind as u8);
            h.write_u8(*mutable as u8);
            hash_opt(&mut *h, name, |h, n| n.hash(h));
            hash_opt(&mut *h, pattern, |h, p| hash_pattern(h, p));
            ty.hash(&mut *h);
            hash_opt(&mut *h, value, |h, e| hash_expr(h, e));
            hash_opt(&mut *h, else_branch, |h, b| hash_stmts(h, b));
            // `TypeParam`s only occur on nested items: span-sensitive leaf.
            hash_seq(&mut *h, type_captures, hash_type_param);
        }
        S::FunctionDef {
            span: _,
            attributes,
            contracts,
            doc,
            name,
            params,
            return_type,
            body,
            type_params,
            where_clause,
            finally,
            is_comptime,
            is_async,
            is_ieee_contracts,
            hints,
            generated_by_comptime,
        } => {
            h.write_u8(1);
            hash_seq(&mut *h, attributes, hash_attribute);
            hash_seq(&mut *h, contracts, hash_contract);
            // NOTE: `doc` should be `None` in the checked HIR (the checker
            // strips doc comments into the doc channel before body-checking).
            // If it can be `Some`, hashing it here would break the
            // two-channel invariance (a doc-only edit would flip the
            // fingerprint).  The doc channel is `DocTokens`' responsibility;
            // this hash is the body's SEMANTIC identity.
            // Fail loudly at the moment the invariant is first violated,
            // not later when a doc-only edit silently reinvalidates every
            // function body.
            debug_assert!(
                doc.is_none(),
                "checker leaked a doc comment into the HIR FunctionDef body; \
                 hashing it breaks two-channel invariance (doc edits would invalidate)"
            );
            // Release arm: ignore `doc` unconditionally.  The invariant is
            // "the checker strips doc"; a release build that violates it
            // must degrade memoization NOT AT ALL rather than flipping every
            // fingerprint on a doc-only edit (the debug assert above fires
            // first in debug builds).
            // Hash `doc` uniformly in every profile: `hash_opt` writes the
            // presence byte even for `None`, so the old debug-only arm made
            // debug and release fingerprints diverge.  The debug_assert above
            // still enforces "checked HIR has doc == None".
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_seq(&mut *h, params, hash_hir_param);
            hash_opt(&mut *h, return_type, |h, t| t.hash(h));
            hash_opt(&mut *h, body, |h, b| hash_stmts(h, b));
            hash_seq(&mut *h, type_params, hash_type_param);
            // Hash the CONTENTS, not just presence: editing the bounds
            // inside the clause (e.g. `where T: Foo` → `where T: Bar`)
            // must flip the fingerprint.  `hash_debug` is the safe
            // fallback (sound direction only — it over-hashes spans, never
            // under-hashes semantics), matching the treatment of other
            // AST fragments in this walker.
            // (Note: in the CHECKED HIR the clause is `Option<()>` — the
            // resolver strips bounds — so today this hashes presence only;
            // the comment keeps the contract for any future AST-level
            // variant of this walker.)
            match where_clause {
                None => h.write_u8(0),
                Some(w) => {
                    h.write_u8(1);
                    hash_debug(&mut *h, w);
                }
            }
            hash_opt(&mut *h, finally, |h, b| hash_stmts(h, b));
            h.write_u8(*is_comptime as u8);
            h.write_u8(*is_async as u8);
            h.write_u8(*is_ieee_contracts as u8);
            // AST fragments: span-sensitive leaves (see `hash_debug`).
            hash_seq(&mut *h, hints, hash_debug);
            h.write_u8(*generated_by_comptime as u8);
        }
        S::TypeDef {
            span: _,
            attributes,
            doc,
            name,
            params,
            definition,
            contracts,
        } => {
            h.write_u8(2);
            hash_seq(&mut *h, attributes, hash_attribute);
            // Same two-channel invariant as `S::FunctionDef`'s `doc` above:
            // fail loudly at the first violation, not later when a doc-only
            // edit flips every type definition's fingerprint.
            debug_assert!(
                doc.is_none(),
                "checker leaked a doc comment into the HIR TypeDef body; \
                 hashing it breaks two-channel invariance (doc edits would invalidate)"
            );
            // Release arm: same rule as `S::FunctionDef`'s `doc` above.
            // Hash `doc` uniformly in every profile: `hash_opt` writes the
            // presence byte even for `None`, so the old debug-only arm made
            // debug and release fingerprints diverge.  The debug_assert above
            // still enforces "checked HIR has doc == None".
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_seq(&mut *h, params, hash_type_param);
            // `definition` is a `TypeDefinition` body: typed hasher (see
            // `hash_type_definition`) — restores the two-channel trivia
            // invariance that `hash_debug` defeated for all type
            // definitions.
            hash_type_definition(&mut *h, definition);
            hash_seq(&mut *h, contracts, hash_contract);
        }
        S::TraitDef {
            span: _,
            attributes,
            doc,
            name,
            methods,
            associated_types,
        } => {
            h.write_u8(3);
            hash_seq(&mut *h, attributes, hash_attribute);
            // Same two-channel invariant as `S::FunctionDef`'s `doc` above:
            // the checker strips doc into the doc channel; hashing a leaked
            // doc here would flip every trait's fingerprint on a doc-only
            // edit.  Fail loudly at the first violation.
            debug_assert!(
                doc.is_none(),
                "checker leaked a doc comment into the HIR TraitDef body; \
                 hashing it breaks two-channel invariance (doc edits would invalidate)"
            );
            // Hash `doc` uniformly in every profile: `hash_opt` writes the
            // presence byte even for `None`, so the old debug-only arm made
            // debug and release fingerprints diverge.  The debug_assert above
            // still enforces "checked HIR has doc == None".
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_seq(&mut *h, methods, hash_trait_method);
            hash_seq(&mut *h, associated_types, hash_associated_type);
        }
        S::Import {
            path,
            items,
            alias,
            span: _,
        } => {
            h.write_u8(4);
            hash_seq(&mut *h, path, |h, s| s.hash(h));
            hash_opt(&mut *h, items, |h, v| hash_seq(h, v, |h, s| s.hash(h)));
            hash_opt(&mut *h, alias, |h, s| s.hash(h));
        }
        S::ExternFunction {
            abi,
            name,
            params,
            return_type,
            span: _,
            attributes,
        } => {
            h.write_u8(5);
            abi.hash(&mut *h);
            name.hash(&mut *h);
            hash_seq(&mut *h, params, hash_hir_param);
            return_type.hash(&mut *h);
            hash_seq(&mut *h, attributes, hash_attribute);
        }
        S::Constraint {
            name,
            bounds,
            span: _,
        } => {
            h.write_u8(6);
            name.hash(&mut *h);
            hash_seq(&mut *h, bounds, |h, t| t.hash(h));
        }
        S::Edition(edition, _) => {
            h.write_u8(7);
            edition.hash(h);
        }
        S::Expression(e) => {
            h.write_u8(8);
            hash_expr(&mut *h, e);
        }
        S::If {
            cond,
            then_branch,
            else_branch,
            span: _,
        } => {
            h.write_u8(9);
            hash_expr(&mut *h, cond);
            hash_stmts(&mut *h, then_branch);
            hash_opt(&mut *h, else_branch, |h, b| hash_stmts(h, b));
        }
        S::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            span: _,
        } => {
            h.write_u8(10);
            hash_pattern(&mut *h, pattern);
            hash_expr(&mut *h, scrutinee);
            hash_stmts(&mut *h, then_branch);
            hash_opt(&mut *h, else_branch, |h, b| hash_stmts(h, b));
        }
        S::While {
            label,
            cond,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(11);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
            hash_expr(&mut *h, cond);
            hash_stmts(&mut *h, body);
            hash_opt(&mut *h, invariant, |h, e| hash_expr(h, e));
            hash_opt(&mut *h, decreases, |h, e| hash_expr(h, e));
        }
        S::WhileLet {
            label,
            pattern,
            scrutinee,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(12);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
            hash_pattern(&mut *h, pattern);
            hash_expr(&mut *h, scrutinee);
            hash_stmts(&mut *h, body);
            hash_opt(&mut *h, invariant, |h, e| hash_expr(h, e));
            hash_opt(&mut *h, decreases, |h, e| hash_expr(h, e));
        }
        S::For {
            label,
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            span: _,
        } => {
            h.write_u8(13);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
            hash_pattern(&mut *h, pattern);
            hash_expr(&mut *h, iterable);
            hash_stmts(&mut *h, body);
            hash_opt(&mut *h, invariant, |h, e| hash_expr(h, e));
            hash_opt(&mut *h, decreases, |h, e| hash_expr(h, e));
        }
        S::Loop {
            label,
            body,
            span: _,
        } => {
            h.write_u8(14);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
            hash_stmts(&mut *h, body);
        }
        S::Leave { label, span: _ } => {
            h.write_u8(15);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
        }
        S::Continue { label, span: _ } => {
            h.write_u8(16);
            hash_opt(&mut *h, label, |h, s| s.hash(h));
        }
        S::Return {
            value,
            labels,
            span: _,
        } => {
            h.write_u8(17);
            hash_opt(&mut *h, value, |h, e| hash_expr(h, e));
            hash_seq(&mut *h, labels, |h, s| s.hash(h));
        }
        S::Assign {
            target,
            op,
            value,
            span: _,
        } => {
            h.write_u8(18);
            hash_expr(&mut *h, target);
            h.write_u8(op.is_some() as u8);
            if let Some(o) = op {
                h.write_u8(*o as u8);
            }
            hash_expr(&mut *h, value);
        }
        S::ComptimeBlock {
            captures,
            trusted,
            body,
            ty,
            span: _,
        } => {
            h.write_u8(19);
            // Capture-list spans point at the capture list for errors;
            // only the names (+ mode) are semantic.  The tuple's second
            // element is a `Span`, NOT a type — a type there would make
            // this a silent under-hash.
            const _: () = {
                fn assert_second_is_span<'i>(c: &(Symbol, crate::ast::Span)) {}
                let _ = assert_second_is_span;
            };
            hash_seq(&mut *h, captures, |h, (name, _)| name.hash(h));
            h.write_u8(*trusted as u8);
            hash_stmts(&mut *h, body);
            ty.hash(&mut *h);
        }
        S::ScopeCleanup {
            name,
            when_condition,
            body,
            propagates,
            overrides,
            span: _,
        } => {
            h.write_u8(20);
            name.hash(&mut *h);
            hash_opt(&mut *h, when_condition, |h, e| hash_expr(h, e));
            hash_stmts(&mut *h, body);
            h.write_u8(*propagates as u8);
            h.write_u8(*overrides as u8);
        }
        S::Trigger { name, span: _ } => {
            h.write_u8(21);
            name.hash(h);
        }
        S::Unsafe { body, span: _ } => {
            h.write_u8(22);
            hash_stmts(&mut *h, body);
        }
        S::GhostVariableDef { inner, span: _ } => {
            h.write_u8(23);
            hash_stmt(&mut *h, inner);
        }
        S::Isolate { body, span: _ } => {
            h.write_u8(24);
            hash_stmts(&mut *h, body);
        }
        S::LayoutDef {
            name,
            attributes,
            span: _,
        } => {
            h.write_u8(25);
            name.hash(&mut *h);
            hash_seq(&mut *h, attributes, hash_attribute);
        }
        S::ImplBlock {
            span: _,
            attributes,
            trait_path,
            for_type,
            methods,
            associated_types,
        } => {
            h.write_u8(26);
            // NOTE: `HirStmt::ImplBlock` carries NO `doc` field (unlike
            // FunctionDef/TypeDef/TraitDef), so there is nothing to guard
            // here — the two-channel doc invariant is vacuously upheld.
            hash_seq(&mut *h, attributes, hash_attribute);
            hash_opt(&mut *h, trait_path, |h, d| d.hash(h));
            for_type.hash(&mut *h);
            hash_seq(&mut *h, methods, hash_impl_method);
            hash_seq(&mut *h, associated_types, hash_associated_type);
        }
        S::Generate {
            for_type,
            body,
            span: _,
        } => {
            h.write_u8(27);
            for_type.hash(&mut *h);
            hash_stmts(&mut *h, body);
        }
        S::Stripped { span: _ } => h.write_u8(28),
        S::Error => h.write_u8(29),
    }
}

impl<'input> StableHash for FnBodyCheck<'input> {
    /// Span-free structural hash at the walker level.  The previous
    /// implementation `Debug`-formatted the whole `HirStmt`, so every span
    /// in the body fed the fingerprint: a comment-only edit changed the
    /// fingerprint and fanned out into a full recompute (measured 100%
    /// false-invalidation on trivia edits, ~10ns per hashed byte).  This
    /// walker matches every `Span` with `span: _` and never hashes it
    /// directly; the typed AST hashers it calls (`hash_attribute`,
    /// `hash_contract`, `hash_type_param`, `hash_ast_type`, ...) also skip
    /// spans.
    ///
    /// Every variant match above is exhaustive (no wildcards): a new HIR
    /// variant breaks compilation here and forces a hashing decision, so a
    /// missed field cannot silently become a stale cache hit.  The only
    /// span-sensitive leaves left are the `hash_debug` fallbacks (nested
    /// `ast::Expr` fragments and impl-method bodies, whose `Debug` output
    /// embeds spans) — each is sound-direction (forces a recompute, never
    /// suppresses one).
    ///
    /// Beyond the body, every other field is hashed: `function_finally`
    /// keys are sorted first (`HashMap` order is nondeterministic), the
    /// obligation spans are skipped while their predicates are hashed, and
    /// diagnostics now include level and code (the old hash covered the
    /// message alone, so two diagnostics differing only in severity
    /// collided).
    fn stable_hash(&self) -> u64 {
        // Stack guard: `hash_stmt`/`hash_expr` recurse to HIR depth and run
        // on every `stage`.  The enclosing `get_shared` frame carries a
        // `stacker::maybe_grow`, but that check happened at FRAME ENTRY —
        // a pathological HIR can blow through the red zone INSIDE the
        // guarded compute (an overflow here would poison the key:
        // recoverable, but avoidable).  Mirror the checker's red-zone
        // discipline at the recursion entry.
        stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || {
            let mut h = FxHasher::default();
            hash_stmt(&mut h, &self.hir);
            hash_seq(&mut h, &self.runtime_checks, |h, (name, checks)| {
            name.hash(&mut *h);
            hash_stmts(&mut *h, checks);
        });
        // `function_finally` is a `BTreeMap`, so iteration order is
        // deterministic by construction — no sort, no SmallVec.
        h.write_usize(self.function_finally.len());
        for (k, checks) in &self.function_finally {
            k.hash(&mut h);
            hash_stmts(&mut h, checks);
        }
        hash_seq(&mut h, &self.pending_gadt, |h, g| {
            // Struct-pattern destructuring: a new field on
            // `PendingGadtConstruct` becomes a compile error here rather
            // than a silently unhashed semantic input.  `span` is trivia.
            let crate::hir::checker::PendingGadtConstruct {
                enum_ty,
                variant,
                span: _,
            } = g;
            enum_ty.hash(&mut *h);
            variant.hash(&mut *h);
        });
        hash_seq(&mut h, &self.residual_obligations, |h, (_, pred)| {
            pred.hash(&mut *h)
        });
        hash_seq(&mut h, &self.trait_obligations, |h, (_, pred)| {
            pred.hash(&mut *h)
        });
        hash_seq(&mut h, &self.diagnostics, |h, d| {
            h.write_u8(d.level() as u8);
            d.message_unchecked().hash(&mut *h);
            hash_opt(&mut *h, &d.code().map(|c| c.code()), |h, c| c.hash(h));
        });
        h.finish()
        })
    }
}

/// The type-checked body of one free function, keyed by [`FnId`].
///
/// Test-only compute counter: the digest-gating test asserts that an
/// unchanged function body is served from cache (zero recomputes) while
/// a changed one recomputes.
#[cfg(test)]
pub(crate) static FN_BODY_COMPUTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Test-only per-compute log: `(query-system address, key)` for every
/// `CheckFnBody` execution.  `FN_BODY_COMPUTES` is process-global, so any
/// test counting it races with every other test running `check_program`
/// on another thread (the digest-gate test saw `4 -> 6` instead of
/// `3 -> 4` for exactly this reason).  Tagging each entry with the
/// Tagged by the query system's MONOTONIC id (`system_id()`), so driving
/// `QuerySystem`'s identity lets a test filter to its own system and
/// count deterministically under `cargo test` parallelism.  Never an
/// address — addresses are reused after the system is dropped (the old
/// LATENT note's hazard; `NEXT_SYSTEM_ID` exists for exactly this).
#[cfg(test)]
pub(crate) static FN_BODY_COMPUTE_LOG: std::sync::Mutex<Vec<(usize, FnId)>> =
    std::sync::Mutex::new(Vec::new());

/// Test-only helper: current length of the compute log (a mark).
#[cfg(test)]
pub(crate) fn fn_body_compute_mark() -> usize {
    FN_BODY_COMPUTE_LOG.lock().map(|log| log.len()).unwrap_or(0)
}

/// Test-only helper: reset the compute log.  The log is process-global
/// and append-only (the mark/count helpers above rely on stable indices),
/// so a long CI session would otherwise grow it without bound.  Callers
/// must not run concurrently with tests that hold a mark across the
/// reset — take a fresh mark afterwards.
#[cfg(test)]
pub(crate) fn clear_fn_body_compute_log() {
    FN_BODY_COMPUTE_LOG
        .lock()
        .map(|mut log| log.clear())
        .ok();
}

/// Test-only helper: number of `CheckFnBody` computes logged for the
/// query system with id `system_id` since `mark`.
#[cfg(test)]
pub(crate) fn fn_body_compute_count_for(system_id: usize, mark: usize) -> usize {
    FN_BODY_COMPUTE_LOG
        .lock()
        .map(|log| {
            log.iter()
                .skip(mark)
                .filter(|(id, _)| *id == system_id)
                .count()
        })
        .unwrap_or(0)
}

/// The type-checked body of one free function, keyed by [`FnId`].
///
/// This is the query that makes the body-check pass (the compile's real
/// hot path, where the wall time actually goes) visible to the query
/// system: per-function dep nodes, fingerprints, caching, and — with the
/// digest gate — per-function invalidation, replacing the old
/// chunk-level rayon fan-out that had NO incremental identity at all.
///
/// Free functions only ([`FnId::method`] must be `None`): impl blocks are
/// barrier items — they register inherent methods into the trait env and
/// mutate `method_cache`, so they stay on the serial driver path exactly
/// as Pass 3 treats them today.  The worker is built over a
/// `TypeContext::new_sharing` clone (interner shared, bindings fresh) and
/// a fresh `TraitEnv` clone per compute — the env snapshot problem the
/// chunked pass solves with per-run clones; here the barrier ordering is
/// the DRIVER's obligation (runs must fan out between barrier items).
pub struct CheckFnBody;

impl<'q, 'src: 'q> Query<'q, 'src> for CheckFnBody {
    type Storage = Owned;
    type Key = FnId;
    type Value = FnBodyCheck<'q>;
    type Cache = DefaultCache<FnId, Stored<'q, 'src, Self>>;

    #[inline]
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "check_fn_body",
            "Type-checked body of one free function (HIR + salvaged obligations)",
        )
    }

    fn into_tagged(key: &FnId) -> TaggedQueryKey {
        TaggedQueryKey::CheckFnBody(key.clone())
    }

    fn from_tagged(tag: &TaggedQueryKey) -> Option<FnId> {
        as_variant!(tag, TaggedQueryKey::CheckFnBody(key) => key.clone())
    }

    fn compute<S: Scheduler>(key: &FnId, provider: &dyn QueryProvider<'q, 'src, S>) -> Self::Value {
        // NOTE: no `key.method` assert here — a panic inside `compute`
        // poisons the latch and surfaces as a bogus cycle error.  A key
        // that does not address a free function folds to the error value
        // in the kind guard below (the totality doctrine stated at
        // TypeDefShape / CheckFnSignature / DocTokens).
        //
        // No lifetime bridge needed: `compute` receives the provider at
        // this impl's `'q`, so the session view already borrows for
        // exactly the lifetime the cached `FnBodyCheck<'q>` values
        // require.  A provider with a shorter session lifetime cannot
        // drive `CheckFnBody` — the type system rejects it (the old
        // `transmute` plus `'input: 'static` bound are gone).
        let view = provider.session().expect(
            "CheckFnBody requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let ctx_borrow = view.ctx_sharing().expect(
            "CheckFnBody requires the provider to opt in via \
             SessionQueryProvider::with_body_check",
        );
        let harness = view.body_check_harness().expect(
            "CheckFnBody requires the provider to opt in via \
             SessionQueryProvider::with_body_check",
        );
        // Session-lifetime tripwire: the harness must be the one built
        // against THIS context.  The harness stamps its context's
        // monotonic session id (`TypeContext::session_id`); a provider
        // pairing a stale harness from another session with this session's
        // ctx view must fail LOUDLY in every build, not just debug — in
        // release a silent pass would run `TypeChecker::new_with_source`
        // against the wrong symbol table / trait env and publish the
        // result as if correct.  Deliberately NOT an address — addresses
        // are reused after the old context drops, which would let a stale
        // harness pass silently (the same ban as `NEXT_SYSTEM_ID` /
        // `NEXT_STAT_ARENA_ID`).
        //
        // `compute` cannot return `Result`, so a mismatch folds to the
        // rejected-body shape (the same arm the item-kind guard uses): the
        // release build degrades loudly instead of silently.
        #[cfg(test)]
        {
            FN_BODY_COMPUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Tag = the system's MONOTONIC id (`system_id()`), not its
            // address — the address-reuse hazard the LATENT note below
            // used to carry is gone.
            let system_id = provider.query_system().system_id();
            if let Ok(mut log) = FN_BODY_COMPUTE_LOG.lock() {
                log.push((system_id, *key));
            }
        }
        if harness.session_stamp != ctx_borrow.session_id {
            return FnBodyCheck {
                hir: crate::hir::hir::HirStmt::Error,
                runtime_checks: Vec::new(),
                function_finally: Default::default(),
                pending_gadt: Vec::new(),
                residual_obligations: Vec::new(),
                trait_obligations: Vec::new(),
                // The failure is value-shaped but must also be OBSERVABLE:
                // a caller that only inspects the diagnostics (or the
                // driver's error tally) sees this error instead of silently
                // proceeding with an empty poisoned body.
                diagnostics: vec![crate::diagnostics::Diagnostic::error(format!(
                    "CheckFnBody session/harness mismatch: harness stamp {} != ctx session id {}",
                    harness.session_stamp, ctx_borrow.session_id
                ))],
            };
        }
        // Key-shape totality guard (runs BEFORE any worker-pool work): the
        // key must address a FREE function.  A mis-minted/stale
        // `FnId { item: <free fn>, method: Some(_) }` must not be silently
        // treated as a free-function body check by the kind guard below
        // (which only inspects the item kind and the comptime flag) — it
        // folds to the rejected-body shape, the same totality doctrine
        // stated at TypeDefShape / CheckFnSignature / DocTokens.  Folding
        // here also avoids constructing a worker the guard would discard.
        if key.method.is_some() {
            return FnBodyCheck {
                hir: crate::hir::hir::HirStmt::Error,
                runtime_checks: Vec::new(),
                function_finally: Default::default(),
                pending_gadt: Vec::new(),
                residual_obligations: Vec::new(),
                trait_obligations: Vec::new(),
                diagnostics: Vec::new(),
            };
        }
        // Worker pool (2b): the pieces of worker construction that are pure
        // repeated allocation — the sharing `TypeContext` and the worker
        // `TraitEnv` — are pooled per `(thread, session, program)` and reset
        // in place instead of rebuilt.  The `TypeChecker` itself is still
        // constructed fresh per compute: it borrows `wctx`/`env` mutably for
        // its whole lifetime, so pooling it would be self-referential; its
        // construction is small compared to the table copies the pool kills.
        // On a compute panic the taken halves drop with the frame and the
        // slot's `None`s make the next compute rebuild — no poisoned reuse.
        let program = view.program();
        // Use the driver-maintained monotonic program epoch, NOT the
        // program address.  Addresses are reused after drop; within one
        // session a rebuilt `Program` reallocated at the recycled address
        // would alias the stale pool stamp — exactly the address-as-identity
        // failure class `NEXT_SYSTEM_ID` / `NEXT_STAT_ARENA_ID` exist to ban.
        let pool_stamp = (ctx_borrow.session_id, view.program_epoch());
        let pool_slot = provider
            .query_system()
            .body_worker_pool
            .get_or_default();
        let (mut wctx, mut env) = {
            let mut slot = pool_slot.borrow_mut();
            match slot.as_mut() {
                // Session match: the `TypeContext` is
                // session-scoped and only shares the interner, so a PROGRAM
                // change does not invalidate it — reuse it and rebuild only
                // the `TraitEnv` (whose table copies are the expensive half).
                // A program match reuses both; a dropped `env` (panic
                // residue) rebuilds only the env, keeping the ctx.
                Some(s) if s.stamp.0 == pool_stamp.0 && s.ctx.is_some() => {
                    let mut ctx = s.ctx.take().unwrap();
                    ctx.reset_for_worker_reuse(ctx_borrow);
                    let mut env = match (s.stamp.1 == pool_stamp.1, s.env.take()) {
                        (true, Some(env)) => {
                            let mut env = env;
                            env.reset_for_worker_reuse(view.trait_env());
                            env
                        }
                        // Program changed, or the env half was dropped:
                        // rebuild the trait environment only.
                        _ => view.trait_env().clone_for_worker(),
                    };
                    s.stamp = pool_stamp;
                    (ctx, env)
                }
                _ => {
                    // Stamp miss (new session, or dropped halves): rebuild
                    // both from scratch and adopt the slot for the next
                    // compute on this thread.
                    *slot = Some(BodyWorkerSlot {
                        stamp: pool_stamp,
                        ctx: None,
                        env: None,
                    });
                    (
                        crate::hir::types::TypeContext::new_sharing(ctx_borrow),
                        view.trait_env().clone_for_worker(),
                    )
                }
            }
        };
        // Shared handle (2a): the checker only READS the resolution map, so
        // hand the worker an `Arc::clone` — one refcount bump replaces the
        // full map clone (five `FxHashMap`s) that used to run per function.
        // Views that don't carry the Arc (bare test views) fall back to the
        // by-value clone, which the checker constructor still accepts.
        let resolution_map = view
            .resolution_map_arc()
            .unwrap_or_else(|| std::sync::Arc::new(view.resolution_map().clone()));
        let mut worker = crate::hir::checker::TypeChecker::new_with_source(
            &mut wctx,
            view.symbols(),
            &mut env,
            resolution_map,
            harness.strict_mode,
            harness.enable_experimental,
            harness.features.clone(),
            harness.debug,
            harness.source,
        );
        worker.shared = harness.shared.clone();
        // The worker stands in for the literal-value scope the driver
        // wraps the program in (top-level `insert_literal_value` needs it).
        worker.push_literal_scope();
        // Totality: an out-of-range FnId must yield a VALUE, not a panic —
        // a panic inside `compute` poisons the latch and surfaces as a
        // bogus cycle error (the same doctrine CheckFnSignature /
        // TypeDefShape / DocTokens follow).  A stale FnId from another
        // session, or an index minted past the end of a rebuilt Program,
        // folds to the same shape the driver uses for a rejected body.
        // Item-kind guard (mirrors CheckFnSignature's shape match): the
        // key must address a RUNTIME free function.  A stale/mis-minted
        // FnId pointing at an ImplBlock, a comptime free function (they
        // never exist at runtime — SYNTAX.md §const; the sibling query
        // filters `is_comptime: false` for the same reason), or an
        // out-of-range index folds to the rejected-body shape instead of
        // panicking into a latch poison — or worse, invoking the checker
        // on an item shape it does not expect.
        let Some(
            item @ Item::FunctionDef {
                is_comptime: false, ..
            },
        ) = program.items.get(key.item)
        else {
            worker.pop_literal_scope();
            restore_body_worker(pool_slot, wctx, env);
            return FnBodyCheck {
                hir: crate::hir::hir::HirStmt::Error,
                runtime_checks: Vec::new(),
                function_finally: Default::default(),
                pending_gadt: Vec::new(),
                residual_obligations: Vec::new(),
                trait_obligations: Vec::new(),
                diagnostics: Vec::new(),
            };
        };
        // `check_function_def_salvaged` takes `&Item<'q>` with a free
        // outer borrow lifetime, so the borrow from `program` (shorter
        // than `'q`) passes through unchanged — only the pointee
        // lifetime `'q` reaches the result (`FnBodyCheck<'q>`).  The old
        // `Stmt::Item(item)` wrapper forced `&'q Item<'q>` (the variant
        // is `&'input Item<'input>`, which pins both lifetimes equal)
        // and needed a transmute to satisfy; taking the item directly
        // removes that requirement.
        let out = worker.check_function_def_salvaged(item);
        worker.pop_literal_scope();
        restore_body_worker(pool_slot, wctx, env);
        out
    }
}

/// Hand a finished body-check worker's `TypeContext`/`TraitEnv` back to the
/// thread's pool slot (2b).  The stamp is already validated (the slot's
/// owner is this thread and the stamp was set at construction); on a panic
/// inside `compute` this never runs and the slot's `None` halves make the
/// next compute rebuild from scratch — no poisoned state is ever served.
#[inline]
fn restore_body_worker<'input>(
    slot: &std::cell::RefCell<Option<crate::hir::query::BodyWorkerSlot<'input>>>,
    wctx: crate::hir::types::TypeContext<'input>,
    env: crate::hir::traits::TraitEnv<'input>,
) {
    let mut slot = slot.borrow_mut();
    if let Some(s) = slot.as_mut() {
        s.ctx = Some(wctx);
        s.env = Some(env);
    }
}

/// The doc-channel lines anchored to one function, keyed by [`FnId`].
///
/// A `///` comment precedes the item it documents, so its own span lies
/// OUTSIDE the item span; the lookup matches by the comment's anchor
/// token (`DocMap::doc_lines_anchored_in`), which the parser placed at
/// the item's first token.  `//!` module docs never anchor to an item
/// and are excluded by that lookup.
///
/// Declared `ContentClass::DOC`: the value carries no semantic content —
/// the dep-graph's `finish_task` accounting turns any Semantic query
/// depending on this node into a loud panic, and
/// `dirty_readers_by_class` lets a doc-channel reparse invalidate its
/// readers without touching Semantic fingerprints.
pub struct DocTokens;

/// Newtype so the cache gets a purpose-built `StableHash` (line count +
/// per-line bytes) instead of relying on any generic container hashing.
/// No production caller yet (`DocTokens` is a reference-example query,
/// same status as `TypeDefShape` — see the module header).
#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocLines {
    pub lines: Vec<String>,
}

impl StableHash for DocLines {
    fn stable_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        h.write_usize(self.lines.len());
        for line in &self.lines {
            h.write(line.as_bytes());
            h.write_u8(0xff); // line separator, so ["ab","c"] != ["a","bc"]
        }
        h.finish()
    }
}

impl<'q, 'src: 'q> Query<'q, 'src> for DocTokens {
    type Storage = Arena;
    type Key = FnId;
    type Value = &'q DocLines;
    type Cache = DefaultCache<FnId, Stored<'q, 'src, Self>>;

    #[inline]
    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "doc_tokens",
            "Doc-channel lines anchored to one function (item docs via the doc map)",
        )
        .with_content_class(ContentClass::DOC)
    }

    fn into_tagged(key: &FnId) -> TaggedQueryKey {
        TaggedQueryKey::DocTokens(key.clone())
    }

    fn from_tagged(tag: &TaggedQueryKey) -> Option<FnId> {
        as_variant!(tag, TaggedQueryKey::DocTokens(key) => key.clone())
    }

    fn compute<S: Scheduler>(key: &FnId, provider: &dyn QueryProvider<'q, 'src, S>) -> Self::Value {
        let session = provider.session().expect(
            "DocTokens requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let arena = session.arena().expect(
            "DocTokens requires a provider carrying the session arena \
             (SessionQueryProvider::with_arena)",
        );
        let program = session.program();
        let Some(item) = program.items.get(key.item) else {
            return arena.alloc(DocLines::default());
        };
        let lines = match (item, key.method) {
            // `is_comptime: false`, mirroring CheckFnSignature: comptime
            // free functions never exist at runtime (SYNTAX.md §const) and
            // never get a `FnId` from the pre-pass, so a leaked/cross-session
            // comptime FnId must answer empty rather than return docs for a
            // function that does not exist at runtime.
            (
                Item::FunctionDef {
                    span,
                    is_comptime: false,
                    ..
                },
                None,
            ) => program
                .doc_map
                .doc_lines_anchored_in(span.start, span.end)
                .map(String::from)
                .collect(),
            (Item::ImplBlock { methods, .. }, Some(i)) => match methods.get(i) {
                Some(m) => program
                    .doc_map
                    .doc_lines_anchored_in(m.span.start, m.span.end)
                    .map(String::from)
                    .collect(),
                None => Vec::new(),
            },
            // Comptime functions never get a `FnId` from the pre-pass;
            // answering empty keeps the query total (a panic would poison
            // the latch and surface as a cycle error).
            _ => Vec::new(),
        };
        arena.alloc(DocLines { lines })
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::ast::Program;
    use crate::hir::query::QuerySystem;
    use crate::hir::resolver::{NameResolver, ResolutionMap};
    use crate::hir::symbol::SymbolTable;
    use crate::hir::traits::TraitEnv;
    use crate::hir::types::{CrateId, TypeContext};
    use crate::parser::Parser;
    use bumpalo::Bump;

    /// Resolve a tiny program through the real resolver, so the query is
    /// fed a genuine `SymbolTable` rather than a hand-built stub.
    #[allow(clippy::type_complexity)]
    fn resolved<'a>(
        source: &'a str,
        sync_arena: &'a crate::arena::SyncArena<'a>,
    ) -> (
        TypeContext<'a>,
        SymbolTable<'a>,
        TraitEnv<'a>,
        ResolutionMap<'a>,
        Program<'a, 'a>,
    ) {
        let mut parser = Parser::new(source, FileId::DUMMY, sync_arena);
        let program = parser.parse_program().expect("test source must parse");
        let mut ctx = TypeContext::new();
        ctx.arena = Some(sync_arena);
        let mut resolver = NameResolver::new(&mut ctx, CrateId(DefId(0)), Some(source), false);
        let (symbols, trait_env, _diags, resolution_map) = resolver.resolve_program(&program);
        (ctx, symbols, trait_env, resolution_map, program)
    }

    #[test]
    fn type_def_shape_reads_a_real_resolved_struct() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved(
            "type Point = struct { x: Int<32>, y: Int<32> }\n",
            &sync_arena,
        );

        let def_id = symbols
            .lookup_type(Symbol::intern("Point"))
            .expect("Point must be in the symbol table")
            .def_id;

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        let summary = system
            .get::<TypeDefShape>(def_id, &provider)
            .expect("query must not cycle");

        assert_eq!(summary.def_id, def_id);
        assert_eq!(summary.kind, Some(TypeKind::Struct));
        assert_eq!(summary.num_params, 0);
        assert_eq!(summary.fields.len(), 2, "Point has two fields");
        assert_eq!(summary.fields[0].0, Symbol::intern("x"));
        assert_eq!(summary.fields[1].0, Symbol::intern("y"));

        // Second call must return the identical cached value.
        let again = system
            .get::<TypeDefShape>(def_id, &provider)
            .expect("query must not cycle");
        assert_eq!(summary, again);
    }

    #[test]
    fn type_def_shape_is_total_for_unknown_def_ids() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved("", &sync_arena);

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        // A DefId that names nothing must yield a value, not a panic —
        // a panic would poison the latch and surface as a cycle error.
        let unknown = DefId(usize::MAX / 2);
        let summary = system
            .get::<TypeDefShape>(unknown, &provider)
            .expect("an unknown DefId must not be reported as a cycle");
        assert_eq!(summary.kind, None);
        assert!(summary.fields.is_empty());
    }

    /// The reason `SymbolTable` and `TraitEnv` had to become `Sync`:
    /// `par_get` hands the provider to rayon.
    #[test]
    fn type_def_shape_runs_under_par_get() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved(
            "type A = struct { v: Int<32> }\ntype B = struct { w: Int<32> }\n",
            &sync_arena,
        );

        let a = symbols
            .lookup_type(Symbol::intern("A"))
            .expect("A must resolve")
            .def_id;
        let b = symbols
            .lookup_type(Symbol::intern("B"))
            .expect("B must resolve")
            .def_id;

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        let results = system.par_get::<TypeDefShape>(&[a, b, a], &provider);
        assert_eq!(results.len(), 3);
        let first = results[0].as_ref().expect("A must compute");
        let second = results[1].as_ref().expect("B must compute");
        let third = results[2].as_ref().expect("repeat of A must compute");
        assert_eq!(first.kind, Some(TypeKind::Struct));
        assert_eq!(second.kind, Some(TypeKind::Struct));
        // The duplicate key is deduplicated: same cached value.
        assert_eq!(first, third);
    }

    /// `def get(x: &mut Int<32>) -> &mut Int<32> { return x; }`
    ///
    /// One input borrow `x` (origin 0, mutable) and one output borrow
    /// (origin 1).  The body returns `x`, so the derivation analysis
    /// declares the A(ρ) subset `0 ⟹ 1` — the fact that drives the
    /// cross-function freeze at every call site.
    #[test]
    fn check_fn_signature_reads_a_real_function() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved(
            "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }\n",
            &sync_arena,
        );

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        let facts = system
            .get::<CheckFnSignature>(
                FnId {
                    item: 0,
                    method: None,
                },
                &provider,
            )
            .expect("query must not cycle");

        assert_eq!(facts.input_positions, vec![0], "`x` is parameter 0");
        assert_eq!(facts.facts.universal_region, vec![0, 1], "one in, one out");
        assert_eq!(facts.facts.input_borrow_mutable, vec![true], "`&mut`");
        assert_eq!(
            facts.facts.known_placeholder_subset,
            vec![(0, 1)],
            "the return derives from `x`, so output-live implies input-live"
        );

        // Second call must return the identical cached value.
        let again = system
            .get::<CheckFnSignature>(
                FnId {
                    item: 0,
                    method: None,
                },
                &provider,
            )
            .expect("query must not cycle");
        assert_eq!(facts, again);
    }

    /// A method is addressed by its impl block plus its index within it,
    /// not by name — two impl blocks may both define `get`.
    #[test]
    fn check_fn_signature_addresses_impl_methods_by_index() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let source = concat!(
            "type Box = struct { v: Int<32> }\n",
            "impl for Box {\n",
            "    def get(&mut self) -> Int<32> { return self.v; }\n",
            "    def put(&mut self, n: &Int<32>) { self.v = n; }\n",
            "}\n",
        );
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved(source, &sync_arena);

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        // `impl` is item 1 (item 0 is the type alias).
        let get = system
            .get::<CheckFnSignature>(
                FnId {
                    item: 1,
                    method: Some(0),
                },
                &provider,
            )
            .expect("`get` must compute");
        let put = system
            .get::<CheckFnSignature>(
                FnId {
                    item: 1,
                    method: Some(1),
                },
                &provider,
            )
            .expect("`put` must compute");

        // `get(&mut self)` — `self` is parameter 0, so it is an input
        // borrow position.  Its return is `Int<32>`, not a reference, so
        // there is no output origin and therefore no A(ρ) subset.
        assert_eq!(get.input_positions, vec![0]);
        assert!(get.facts.known_placeholder_subset.is_empty());

        // `put`'s second parameter is also a reference, but a shared one —
        // so it gets a second input origin and a `false` mutability entry.
        assert_eq!(put.input_positions, vec![0, 1]);
        assert_eq!(put.facts.input_borrow_mutable, vec![true, false]);
        // Distinct cache entries, even though both live in one impl block.
        assert_ne!(get, put);
    }

    /// Out-of-range keys must yield a value, not a panic: a panic inside
    /// `compute` poisons the latch and surfaces as a bogus cycle error.
    #[test]
    fn check_fn_signature_is_total_for_out_of_range_keys() {
        let mut arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&mut arena);
        let (_ctx, symbols, trait_env, resolution_map, program) = resolved(
            "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }\n",
            &sync_arena,
        );

        let system = QuerySystem::new();
        let provider = crate::session::SessionQueryProvider::new(
            &system,
            &symbols,
            &trait_env,
            &resolution_map,
            &program,
        )
        .with_arena(&sync_arena);

        for key in [
            FnId {
                item: 99,
                method: None,
            },
            FnId {
                item: 0,
                method: Some(0),
            }, // item 0 is not an impl block
        ] {
            let facts = system
                .get::<CheckFnSignature>(key, &provider)
                .expect("an out-of-range FnId must not be reported as a cycle");
            assert_eq!(*facts, FnSignatureFacts::default(), "key {:?}", key);
        }
    }
}

/// Under `feature = "shuttle"` these are excluded with the rest of the
/// suite: `sample_body` reaches `Symbol::intern`, whose `RwLock` is
/// shuttle's under that feature and needs an `ExecutionState` libtest's
/// thread does not have (`NotSet`).  Same rationale as the `tests` module
/// above — see its `cfg`.
#[cfg(all(test, not(feature = "shuttle")))]
mod stable_hash_tests {
    use super::{FnBodyCheck, StableHash};
    use crate::ast::{BinOp, FileId, Literal, Span, VariableKind};
    use crate::hir::hir::{HirExpr, HirStmt};
    use crate::hir::types::TypeId;
    use crate::symbol::Symbol;

    /// Representative body covering the hottest arms (def, call, binop,
    /// control flow, return).  Every span derives from `base`, so the same
    /// semantic body can be rebuilt at a shifted position.
    fn sample_body(base: usize) -> HirStmt<'static> {
        let sp = |o: usize| {
            let s = u32::try_from(base + o).unwrap_or(u32::MAX);
            Span::new(FileId::DUMMY, s, s.saturating_add(1))
        };
        let ty = TypeId::NONE;
        let x = Symbol::intern("x");
        HirStmt::Expression(Box::new(HirExpr::Block(
            vec![
                HirStmt::VariableDef {
                    kind: VariableKind::Let,
                    mutable: false,
                    name: Some(x),
                    pattern: None,
                    ty,
                    value: Some(Box::new(HirExpr::BinaryOp {
                        left: Box::new(HirExpr::Ident(x, ty, sp(2))),
                        op: BinOp::Add,
                        right: Box::new(HirExpr::Literal(
                            Literal::Int(crate::ast::IntLit::Small(1)),
                            ty,
                            sp(3),
                        )),
                        ty,
                        span: sp(4),
                    })),
                    else_branch: None,
                    span: sp(0),
                    type_captures: vec![],
                },
                HirStmt::If {
                    cond: Box::new(HirExpr::Ident(x, ty, sp(6))),
                    then_branch: vec![HirStmt::Return {
                        value: Some(Box::new(HirExpr::Call {
                            callee: Box::new(HirExpr::Ident(Symbol::intern("f"), ty, sp(8))),
                            args: vec![HirExpr::Ident(x, ty, sp(9))],
                            comptime: false,
                            ty,
                            span: sp(10),
                        })),
                        labels: vec![],
                        span: sp(7),
                    }],
                    else_branch: None,
                    span: sp(5),
                },
            ],
            ty,
            sp(11),
        )))
    }

    fn sample_value(base: usize) -> FnBodyCheck<'static> {
        FnBodyCheck {
            hir: sample_body(base),
            runtime_checks: vec![],
            function_finally: Default::default(),
            pending_gadt: vec![],
            residual_obligations: vec![],
            trait_obligations: vec![],
            diagnostics: vec![],
        }
    }

    /// Trivia-shift stability: the old `Debug` hash changed on ANY span
    /// movement (measured 100% false-invalidation on comment-only edits).
    /// The span-free walker must not.
    #[test]
    fn trivia_span_shift_keeps_fingerprint() {
        assert_eq!(
            sample_value(0).stable_hash(),
            sample_value(10_000).stable_hash()
        );
    }

    /// Impl-method bodies are raw AST statements.  The span-free AST
    /// walker must ignore a comment/whitespace shift but still see a
    /// semantic edit (literal change).
    #[test]
    fn impl_body_span_shift_keeps_fingerprint() {
        use super::hash_ast_stmt;
        use crate::ast::{Expr as AExpr, IntLit, Literal, Stmt as AStmt};
        use crate::symbol::Symbol;
        use rustc_hash::FxHasher;
        use std::hash::Hasher;

        fn hash_body(base: usize, value: i128) -> u64 {
            let sp = |o: usize| {
                let s = u32::try_from(base + o).unwrap_or(u32::MAX);
                Span::new(FileId::DUMMY, s, s.saturating_add(1))
            };
            let callee = AExpr::Ident(Symbol::intern("f"), sp(1));
            let args = [AExpr::Literal(Literal::Int(IntLit::Small(value)), sp(2))];
            let call = AExpr::Call {
                callee: &callee,
                args: &args,
                comptime: false,
                span: sp(3),
            };
            let stmt = AStmt::Return {
                value: Some(call),
                labels: &[],
                span: sp(4),
            };
            let mut h = FxHasher::default();
            hash_ast_stmt(&mut h, &stmt);
            h.finish()
        }

        assert_eq!(
            hash_body(0, 7),
            hash_body(10_000, 7),
            "a span shift inside an impl-method body must not flip the fingerprint"
        );
        assert_ne!(
            hash_body(0, 7),
            hash_body(0, 8),
            "a literal edit must flip the fingerprint"
        );
    }

    /// E2 gap: the original trivia test used a body with NO attributes,
    /// contracts, or type params, so a comment inserted before one of those
    /// constructs could still flip the fingerprint.  Build a `FunctionDef`
    /// carrying all three (plus a param default) at two span bases and
    /// assert the fingerprint is identical.
    #[test]
    fn trivia_span_shift_keeps_fingerprint_with_nested_constructs() {
        use crate::ast::{Attribute, Contract, EnsuresTarget, TypeParam, TypeParamKind};
        use crate::hir::hir::HirParam;
        use crate::symbol::Symbol;

        fn fn_def(base: usize) -> HirStmt<'static> {
            let sp = |o: usize| {
                let s = u32::try_from(base + o).unwrap_or(u32::MAX);
                Span::new(FileId::DUMMY, s, s.saturating_add(1))
            };
            let ty = TypeId::NONE;
            let x = Symbol::intern("x");
            let t = Symbol::intern("T");
            let inline = Symbol::intern("inline");
            HirStmt::FunctionDef {
                span: sp(0),
                attributes: vec![Attribute {
                    name: inline,
                    args: vec![crate::ast::Expr::Ident(x, sp(1))],
                    named_args: vec![],
                    span: sp(2),
                }],
                contracts: vec![Contract::Requires(
                    crate::ast::Expr::Literal(
                        Literal::Int(crate::ast::IntLit::Small(1)),
                        sp(3),
                    ),
                    sp(4),
                )],
                doc: None,
                name: Symbol::intern("f"),
                params: vec![HirParam {
                    name: x,
                    ty,
                    default: Some(crate::ast::Expr::Literal(
                        Literal::Int(crate::ast::IntLit::Small(2)),
                        sp(5),
                    )),
                    span: sp(7),
                }],
                return_type: None,
                body: None,
                type_params: vec![TypeParam {
                    name: t,
                    bounds: &[],
                    kind: TypeParamKind::Type,
                    span: sp(6),
                }],
                where_clause: None,
                finally: None,
                is_comptime: false,
                is_async: false,
                is_ieee_contracts: false,
                hints: vec![],
                generated_by_comptime: false,
            }
        }
        let v = |base| FnBodyCheck {
            hir: fn_def(base),
            runtime_checks: vec![],
            function_finally: Default::default(),
            pending_gadt: vec![],
            residual_obligations: vec![],
            trait_obligations: vec![],
            diagnostics: vec![],
        };
        assert_eq!(
            v(0).stable_hash(),
            v(10_000).stable_hash(),
            "a comment-shifted span before an attribute/contract/type param \
             must NOT flip the fingerprint"
        );
        // Semantic flip still detected: change the contract expression.
        let mut w = v(0);
        if let HirStmt::FunctionDef { contracts, .. } = &mut w.hir {
            contracts[0] = Contract::Ensures {
                expr: crate::ast::Expr::Ident(Symbol::intern("y"), Span::new(FileId::DUMMY, 0, 1)),
                span: Span::new(FileId::DUMMY, 0, 1),
                target: EnsuresTarget::Unconditional,
                labels: vec![],
            };
        }
        assert_ne!(v(0).stable_hash(), w.stable_hash());
    }

    /// Semantic edits must flip the fingerprint — a missed field here
    /// would be a stale cache hit, the unsound direction.
    #[test]
    fn semantic_edits_flip_fingerprint() {
        let base = sample_value(0).stable_hash();
        // Changed operator.
        let mut v = sample_value(0);
        if let HirStmt::Expression(e) = &mut v.hir
            && let HirExpr::Block(stmts, _, _) = e.as_mut()
            && let HirStmt::VariableDef {
                value: Some(val), ..
            } = &mut stmts[0]
            && let HirExpr::BinaryOp { op, .. } = val.as_mut()
        {
            *op = BinOp::Sub;
        }
        assert_ne!(base, v.stable_hash(), "operator change invisible");
        // Added statement.
        let mut v = sample_value(0);
        if let HirStmt::Expression(e) = &mut v.hir
            && let HirExpr::Block(stmts, _, _) = e.as_mut()
        {
            stmts.push(HirStmt::Trigger {
                name: Symbol::intern("t"),
                span: Span::new(FileId::DUMMY, 1, 2),
            });
        }
        assert_ne!(base, v.stable_hash(), "added statement invisible");
        // New diagnostic (level + code are hashed, not just the message).
        let mut v = sample_value(0);
        v.diagnostics
            .push(crate::diagnostics::Diagnostic::error("boom"));
        assert_ne!(base, v.stable_hash(), "new diagnostic invisible");
        // New obligation predicate (spans skipped, predicates hashed).
        let mut v = sample_value(0);
        v.residual_obligations.push((
            Span::new(FileId::DUMMY, 0, 0),
            crate::hir::traits::solver::obligation::Predicate::Sized { ty: TypeId::NONE },
        ));
        assert_ne!(base, v.stable_hash(), "new obligation invisible");
    }
}
