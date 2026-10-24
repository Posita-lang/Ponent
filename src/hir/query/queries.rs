//! The compiler's query definitions.
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

use crate::ast::FileId;
use super::{
    Arena, ContentClass, DefaultCache, Owned, Query, QueryDescriptor, QueryProvider, Scheduler,
    StableHash, Stored, TaggedQueryKey,
};
use crate::ast::Stmt;
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
pub struct TypeDefShape;

impl<'q, 'src: 'q> Query<'q, 'src> for TypeDefShape {
    type Storage = Arena;
    type Key = DefId;
    type Value = &'q TypeDefSummary;
    type Cache = DefaultCache<DefId, Stored<'q, 'src, Self>>;

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
        match tag {
            TaggedQueryKey::TypeDefShape(key) => Some(*key),
            _ => None,
        }
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
pub struct CheckFnSignature;

impl<'q, 'src: 'q> Query<'q, 'src> for CheckFnSignature {
    type Storage = Arena;
    type Key = FnId;
    type Value = &'q FnSignatureFacts;
    type Cache = DefaultCache<FnId, Stored<'q, 'src, Self>>;

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
        // crate with more functions keeps a most-recent window and pays
        // deterministic recomputes for evicted entries.
        //
        // Driver obligation: `clear_all` between Programs (or
        // `clear::<CheckFnSignature>` between iterations).  A FnId is
        // meaningless across sessions and the memo carries no generation
        // stamp — clear_all's registry replace kills the vtable (and the
        // memo with it); nothing structural enforces the call.
        .with_untracked(true)
    }

    fn into_tagged(key: &FnId) -> TaggedQueryKey {
        TaggedQueryKey::CheckFnSignature(key.clone())
    }

    fn from_tagged(tag: &TaggedQueryKey) -> Option<FnId> {
        match tag {
            TaggedQueryKey::CheckFnSignature(key) => Some(key.clone()),
            _ => None,
        }
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
                Stmt::FunctionDef {
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
            (Stmt::ImplBlock { methods, .. }, Some(i)) => match methods.get(i) {
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
    pub function_finally: std::collections::HashMap<Symbol, Vec<crate::hir::hir::HirStmt<'input>>>,
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
    // `default` is an AST fragment: span-sensitive leaf (see `hash_debug`).
    hash_opt(h, &p.default, |h, e| hash_debug(h, e));
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
            // `ast::Type`: span-sensitive leaf (see `hash_debug`).
            h.write_u8(33);
            hash_debug(&mut *h, ty);
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
            hash_debug(&mut *h, type_captures);
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
            hash_debug(&mut *h, attributes);
            hash_debug(&mut *h, contracts);
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_seq(&mut *h, params, hash_hir_param);
            hash_opt(&mut *h, return_type, |h, t| t.hash(h));
            hash_opt(&mut *h, body, |h, b| hash_stmts(h, b));
            hash_debug(&mut *h, type_params);
            h.write_u8(where_clause.is_some() as u8);
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
            hash_debug(&mut *h, attributes);
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_debug(&mut *h, params);
            hash_debug(&mut *h, definition);
            hash_debug(&mut *h, contracts);
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
            hash_debug(&mut *h, attributes);
            hash_opt(&mut *h, doc, |h, d| d.hash(h));
            name.hash(&mut *h);
            hash_debug(&mut *h, methods);
            hash_debug(&mut *h, associated_types);
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
            hash_debug(&mut *h, attributes);
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
            // only the names are semantic.
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
            hash_debug(&mut *h, attributes);
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
            hash_debug(&mut *h, attributes);
            hash_opt(&mut *h, trait_path, |h, d| d.hash(h));
            for_type.hash(&mut *h);
            hash_debug(&mut *h, methods);
            hash_debug(&mut *h, associated_types);
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
    /// Span-free structural hash (report item #3).  The previous
    /// implementation `Debug`-formatted the whole `HirStmt`, so every span
    /// in the body fed the fingerprint: a comment-only edit changed the
    /// fingerprint and fanned out into a full recompute (measured 100%
    /// false-invalidation on trivia edits, ~10ns per hashed byte).  This
    /// walker hashes semantic content only and skips every `Span`.
    ///
    /// Every variant match above is exhaustive (no wildcards): a new HIR
    /// variant breaks compilation here and forces a hashing decision, so a
    /// missed field cannot silently become a stale cache hit.  The only
    /// span-sensitive leaves left are enumerated on `hash_debug` — each is
    /// sound-direction (forces a recompute, never suppresses one).
    ///
    /// Beyond the body, every other field is hashed: `function_finally`
    /// keys are sorted first (`HashMap` order is nondeterministic), the
    /// obligation spans are skipped while their predicates are hashed, and
    /// diagnostics now include level and code (the old hash covered the
    /// message alone, so two diagnostics differing only in severity
    /// collided).
    fn stable_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        hash_stmt(&mut h, &self.hir);
        hash_seq(&mut h, &self.runtime_checks, |h, (name, checks)| {
            name.hash(&mut *h);
            hash_stmts(&mut *h, checks);
        });
        // Deterministic order: `HashMap` iteration is unspecified.  A
        // `SmallVec` keeps the typical 0-8 `finally` keys entirely on the
        // stack: this runs on every `CheckFnBody::compute` (the value is
        // `stable_hash`ed before it is staged), and the old `Vec` paid a
        // heap allocation whenever the map was non-empty.
        let mut finally_keys: SmallVec<[Symbol; 8]> =
            self.function_finally.keys().copied().collect();
        finally_keys.sort();
        h.write_usize(finally_keys.len());
        for k in finally_keys {
            k.hash(&mut h);
            if let Some(checks) = self.function_finally.get(&k) {
                hash_stmts(&mut h, checks);
            }
        }
        hash_seq(&mut h, &self.pending_gadt, |h, g| {
            g.enum_ty.hash(&mut *h);
            g.variant.hash(&mut *h);
        });
        hash_seq(&mut h, &self.residual_obligations, |h, (_, pred)| {
            pred.hash(&mut *h)
        });
        hash_seq(&mut h, &self.trait_obligations, |h, (_, pred)| {
            pred.hash(&mut *h)
        });
        hash_seq(&mut h, &self.diagnostics, |h, d| {
            h.write_u8(d.level() as u8);
            d.message().hash(&mut *h);
            hash_opt(&mut *h, &d.code().map(|c| c.code()), |h, c| c.hash(h));
        });
        h.finish()
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
/// driving `QuerySystem`'s address lets a test filter to its own system
/// and count deterministically under `cargo test` parallelism.
///
/// LATENT: the tag is the system's ADDRESS, while the rest of this
/// module deliberately keys on the monotonic `NEXT_SYSTEM_ID` (mod.rs)
/// because an address is reused after the system is dropped.  Harmless
/// today because every `QuerySystem` in the test suite is leaked (they
/// hold `&'static` handles), so no two are ever live at once; a test
/// that dropped a system and allocated another would alias.  If one
/// ever needs to, filter on `system.system_id()` instead and widen the
/// tuple to `(usize, usize)`.
#[cfg(test)]
pub(crate) static FN_BODY_COMPUTE_LOG: std::sync::Mutex<Vec<(usize, FnId)>> =
    std::sync::Mutex::new(Vec::new());

/// Test-only helper: current length of the compute log (a mark).
#[cfg(test)]
pub(crate) fn fn_body_compute_mark() -> usize {
    FN_BODY_COMPUTE_LOG.lock().map(|log| log.len()).unwrap_or(0)
}

/// Test-only helper: number of `CheckFnBody` computes logged for the
/// query system at `system_addr` since `mark`.
#[cfg(test)]
pub(crate) fn fn_body_compute_count_for(system_addr: usize, mark: usize) -> usize {
    FN_BODY_COMPUTE_LOG
        .lock()
        .map(|log| {
            log.iter()
                .skip(mark)
                .filter(|(addr, _)| *addr == system_addr)
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
        match tag {
            TaggedQueryKey::CheckFnBody(key) => Some(key.clone()),
            _ => None,
        }
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
        // against THIS context.  `with_body_check` stamps the ctx address
        // into the harness; a provider pairing a stale harness from another
        // session with this session's ctx view fails loudly here in debug
        // builds instead of silently checking against the wrong context.
        debug_assert_eq!(
            harness.session_stamp, ctx_borrow as *const _ as usize,
            "CheckFnBody driven with a BodyCheckHarness from a different session \
             (harness stamp != ctx address)"
        );
        #[cfg(test)]
        {
            FN_BODY_COMPUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let system_addr = provider.query_system() as *const _ as usize;
            if let Ok(mut log) = FN_BODY_COMPUTE_LOG.lock() {
                log.push((system_addr, *key));
            }
        }
        // Fresh worker per compute — the per-item shape the profile
        // priced at ~0.65ms of construction, amortised on real bodies.
        let program = view.program();
        let mut wctx = crate::hir::types::TypeContext::new_sharing(ctx_borrow);
        // Worker clone: the env's lookup memo starts empty (the worker
        // re-fills its own entries) — the O(cache) deep copy under the
        // env's read lock drops out of the per-function compute path.
        let mut env = view.trait_env().clone_for_worker();
        let mut worker = crate::hir::checker::TypeChecker::new_with_source(
            &mut wctx,
            view.symbols(),
            &mut env,
            view.resolution_map().clone(),
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
            stmt @ Stmt::FunctionDef {
                is_comptime: false, ..
            },
        ) = program.items.get(key.item)
        else {
            worker.pop_literal_scope();
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
        let out = worker.check_function_def_salvaged(&stmt);
        worker.pop_literal_scope();
        out
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
        match tag {
            TaggedQueryKey::DocTokens(key) => Some(key.clone()),
            _ => None,
        }
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
            (Stmt::FunctionDef { span, .. }, None) => program
                .doc_map
                .doc_lines_anchored_in(span.start, span.end)
                .map(String::from)
                .collect(),
            (Stmt::ImplBlock { methods, .. }, Some(i)) => match methods.get(i) {
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
        let arena = Bump::new();
        let sync_arena = crate::arena::SyncArena::new(&arena);
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
