//! The compiler's query definitions.
//!
//! This is where the query system stops being infrastructure and starts
//! being used.  Each item here is a `Key -> Value` mapping computed on
//! demand, cached, and dep-tracked for incremental re-use.
//!
//! ## Why every value type here is owned and `'static`
//!
//! `Query::Value` requires `Clone + Debug + Send + 'static + StableHash`
//! (see `Query` in the parent module).  The compiler's own data structures
//! — `SymbolTable<'input>`, `TraitEnv<'input>`, `HirStmt<'input>` — borrow
//! the parse arena, which `main.rs` creates as a **stack-local**
//! `bumpalo::Bump`.  They therefore cannot be `'static` without leaking the
//! arena, so they can never themselves be query values.
//!
//! Values here are instead *owned snapshots extracted from* that state:
//! small, position-free, arena-independent summaries.  That also satisfies
//! the `StableHash` contract (parent module) for free, since a summary has
//! no spans to accidentally hash.
//!
//! ## Which queries are wired into the pipeline
//!
//! - [`CheckFnSignature`] — driven by `pre_register_signatures` in
//!   `hir::checker`, which is the compiler's real per-function pre-pass.
//! - [`TypeDefShape`] — still test-only; it is the reference example the
//!   query machinery was validated against.

// `TypeDefShape` has no production caller yet.  Remove once it does.
#![allow(dead_code)]

use super::{
    CompilerSessionView, ContentClass, DefaultCache, Query, QueryDescriptor, QueryProvider,
    StableHash,
};
use crate::ast::Stmt;
use crate::hir::polonius::{count_return_refs_ast, extract_borrow_signature, signature_facts};
use crate::hir::symbol::TypeKind;
use crate::hir::types::{DefId, SignatureFacts, TypeId};
use crate::symbol::Symbol;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

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

impl Query for TypeDefShape {
    type Key = DefId;
    type Value = TypeDefSummary;
    type Cache = DefaultCache<DefId, TypeDefSummary>;

    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "type_def_shape",
            "resolved shape (kind, fields, arity) of a type definition",
        )
    }

    fn compute(key: &DefId, provider: &dyn QueryProvider) -> TypeDefSummary {
        let session = provider.session().expect(
            "TypeDefShape requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let symbols = session.symbols();

        match symbols.lookup_type_by_def_id(*key) {
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
        }
    }
}

impl StableHash for SignatureFacts {
    /// Semantic identity only: the facts are plain integer/bool vectors
    /// with no spans (the module note), so hashing them element-wise
    /// cannot drift.
    fn stable_hash(&self) -> u64 {
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

impl Query for CheckFnSignature {
    type Key = FnId;
    type Value = FnSignatureFacts;
    type Cache = DefaultCache<FnId, FnSignatureFacts>;

    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "check_fn_signature",
            "AST-level borrow signature (input borrow positions + A(rho) facts) of one function",
        )
    }

    fn compute(key: &FnId, provider: &dyn QueryProvider) -> FnSignatureFacts {
        let session = provider.session().expect(
            "CheckFnSignature requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let program = session.program();
        let Some(item) = program.items.get(key.item) else {
            return FnSignatureFacts::default();
        };
        match (item, key.method) {
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
            ) => signature_facts_of(params, return_type.as_ref(), body.as_deref()),
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
        }
    }
}

// ── Per-function body check ──────────────────────────────────────

/// The complete body-check result of one free function, keyed by [`FnId`]:
/// the checked HIR body plus everything the caller used to salvage off the
/// worker checker (deferred obligations, GADT constructs, diagnostics).
///
/// `'input: 'static` is REQUIRED for this type to be a query value (the
/// value must outlive the compute into the cache): the whole pipeline
/// already satisfies it — `main.rs` leaks the parse arena
/// (`sync_arena.alloc_str`), and `checker::tests` leaks its fixtures
/// (`check_source`) — but a stack-arena caller cannot use this query.
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
    pub residual_obligations: Vec<(crate::ast::Span, crate::hir::traits::solver::obligation::Predicate)>,
    pub trait_obligations: Vec<(crate::ast::Span, crate::hir::traits::solver::obligation::Predicate)>,
    pub diagnostics: Vec<crate::diagnostics::Diagnostic<'input>>,
}

impl<'input> StableHash for FnBodyCheck<'input> {
    /// LEAF-CONSERVATIVE, deliberate deviation from the span-free rule the
    /// facts queries above follow.  Their values are position-free by
    /// construction, so their hashes cannot drift under trivia edits; the
    /// body-check value CONTAINS spans, so hashing it whole means a
    /// trivia edit inside the function changes the fingerprint and fans
    /// out.  That is sound (never a false "unchanged") and cheap: nothing
    /// reads `FnBodyCheck` downstream today, so there are no reader
    /// stamps to lose — the fingerprint only decides "recompute this
    /// function or serve the cache".  A no-span structural hasher over
    /// `HirStmt` is the follow-up if a downstream query ever starts
    /// reading these values.
    fn stable_hash(&self) -> u64 {
        let mut h = FxHasher::default();
        format!("{:?}", self.hir).hash(&mut h);
        h.write_u8(self.diagnostics.len() as u8);
        for d in &self.diagnostics {
            d.message().hash(&mut h);
        }
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
pub struct CheckFnBody<'input>(std::marker::PhantomData<fn() -> crate::hir::hir::HirStmt<'input>>);

impl<'input: 'static> Query for CheckFnBody<'input> {
    type Key = FnId;
    type Value = FnBodyCheck<'input>;
    type Cache = DefaultCache<FnId, FnBodyCheck<'input>>;

    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "check_fn_body",
            "Type-checked body of one free function (HIR + salvaged obligations)",
        )
    }

    fn compute(key: &FnId, provider: &dyn QueryProvider<'_>) -> Self::Value {
        assert!(
            key.method.is_none(),
            "CheckFnBody is keyed by FREE functions only; impl methods are \
             barrier items and stay on the serial driver path"
        );
        let view = provider.session().expect(
            "CheckFnBody requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        // SAFETY: bridge the ERASED provider lifetime to this impl's
        // `'input`.  The `Query` trait's `compute` takes
        // `&dyn QueryProvider<'_>` — the session lifetime is anonymous
        // there (the same erasure that makes `Any` unusable for the
        // session view, see the `CompilerSessionView` doc), so the
        // compiler cannot prove what is true by construction: this query
        // is driven ONLY through `check_program`, whose harness builder
        // requires `'input: 'static` (the pipeline leaks the parse arena;
        // `check_source` fixtures do the same), i.e. the provider's
        // session lifetime IS this impl's `'input`.  A provider with a
        // shorter session lifetime must never drive `CheckFnBody` — the
        // cached `FnBodyCheck<'input>` values would outlive the state
        // they borrow.  Enforced by construction (only `check_program`
        // calls `with_body_check`), not by the type system; the clean fix
        // is a lifetime-parameterized `Query::compute` (GAT), deferred
        // with the rest of the `&mut TypeContext` sharing work.
        let view: &dyn CompilerSessionView<'input> = unsafe {
            std::mem::transmute::<&dyn CompilerSessionView<'_>, &dyn CompilerSessionView<'input>>(
                view,
            )
        };
        let ctx_borrow = view.ctx_sharing().expect(
            "CheckFnBody requires the provider to opt in via \
             SessionQueryProvider::with_body_check",
        );
        let harness = view.body_check_harness().expect(
            "CheckFnBody requires the provider to opt in via \
             SessionQueryProvider::with_body_check",
        );
        #[cfg(test)]
        FN_BODY_COMPUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Fresh worker per compute — the per-item shape the profile
        // priced at ~0.65ms of construction, amortised on real bodies.
        let program = view.program();
        let stmt = program
            .items
            .get(key.item)
            .expect("FnId.item indexes Program::items for the lifetime of the session");
        let mut wctx = crate::hir::types::TypeContext::new_sharing(ctx_borrow);
        let mut env = view.trait_env().clone();
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
        let out = worker.check_function_def_salvaged(stmt);
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

impl Query for DocTokens {
    type Key = FnId;
    type Value = DocLines;
    type Cache = DefaultCache<FnId, DocLines>;

    fn descriptor() -> QueryDescriptor {
        QueryDescriptor::new(
            "doc_tokens",
            "Doc-channel lines anchored to one function (item docs via the doc map)",
        )
        .with_content_class(ContentClass::DOC)
    }

    fn compute(key: &FnId, provider: &dyn QueryProvider) -> DocLines {
        let session = provider.session().expect(
            "DocTokens requires a provider carrying a compiler session; \
             it is being driven by a bare QuerySystem",
        );
        let program = session.program();
        let Some(item) = program.items.get(key.item) else {
            return DocLines::default();
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
        DocLines { lines }
    }
}

#[cfg(test)]
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
        Program<'a>,
    ) {
        let mut parser = Parser::new(source, sync_arena);
        let program = parser.parse_program().expect("test source must parse");
        let mut ctx = TypeContext::new();
        ctx.arena = Some(sync_arena);
        let mut resolver = NameResolver::new(&mut ctx, CrateId(DefId(0)), Some(source));
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
        );

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
        );

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
        );

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
        );

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
        );

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
        );

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
            assert_eq!(facts, FnSignatureFacts::default(), "key {:?}", key);
        }
    }
}
