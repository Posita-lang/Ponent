//! The compilation session.
//!
//! One value owns the state a query needs.  Today the pipeline in
//! `main.rs` threads these pieces through as separate locals; the session
//! collects them so a query can be handed the whole thing via a
//! [`QueryProvider`].
//!
//! `TypeContext` is deliberately NOT owned by a query: every consumer of
//! it wants `&mut`, and sharing that is the separate `&mut TypeContext`
//! problem.  What IS exposed (opt-in, via
//! [`SessionQueryProvider::with_body_check`]) is a read-only borrow a
//! compute can `TypeContext::new_sharing`-clone from — the same
//! clone-the-interner discipline the parallel body-check pass already
//! uses — plus the checker config + cross-function tables a body-check
//! worker needs.  Queries may still only READ resolved state.

// `CompilerSession` itself is still assembled only by `main.rs`, which does
// not read every field back; the per-function queries reach the state
// through `SessionQueryProvider` instead.

use crate::ast::Program;
use crate::hir::checker::SharedCheckState;
use crate::hir::query::{CompilerSessionView, QueryProvider, QuerySystem};
use crate::hir::resolver::ResolutionMap;
use crate::hir::symbol::SymbolTable;
use crate::hir::target::Target;
use crate::hir::traits::TraitEnv;
use crate::hir::types::TypeContext;

/// Everything a body-check worker needs beyond the session's resolved
/// state: the checker configuration and the cross-function tables
/// ([`SharedCheckState`], behind an `Arc` exactly as the parallel
/// body-check pass shares them today).
///
/// `source` is `&'input str` (NOT the checker's independent `'src`): a
/// `CheckFnBody` compute must construct its worker with `'src = 'input`
/// so the salvaged `Diagnostic<'input>` values are `'static`-eligible for
/// the query cache.  The `'src: 'input` proof is the harness-builder
/// caller's obligation — `check_program` takes it as a where-clause.
///
/// Built by the driver (which owns the config), carried opt-in by
/// [`SessionQueryProvider`], and consumed by a `CheckFnBody` compute to
/// construct its own worker `TypeChecker` over a `new_sharing` context
/// clone.  `Send + Sync` so the provider stays handable to rayon:
/// `SharedCheckState` is already shared across rayon workers by the
/// existing pass.
#[derive(Clone)]
pub struct BodyCheckHarness<'input> {
    pub shared: std::sync::Arc<SharedCheckState<'input>>,
    pub strict_mode: bool,
    pub enable_experimental: bool,
    pub features: Vec<String>,
    pub debug: bool,
    pub source: Option<&'input str>,
}

/// All the state one compilation needs, in one place.
pub struct CompilerSession<'input> {
    pub system: QuerySystem,
    pub ctx: TypeContext<'input>,
    pub symbols: SymbolTable<'input>,
    pub trait_env: TraitEnv<'input>,
    pub resolution_map: ResolutionMap<'input>,
    pub target: Target,
}

impl<'input> CompilerSession<'input> {
    pub fn new(
        ctx: TypeContext<'input>,
        symbols: SymbolTable<'input>,
        trait_env: TraitEnv<'input>,
        resolution_map: ResolutionMap<'input>,
        target: Target,
    ) -> Self {
        CompilerSession {
            system: QuerySystem::new(),
            ctx,
            symbols,
            trait_env,
            resolution_map,
            target,
        }
    }
}

/// A `QueryProvider` that carries a borrow of the session's resolved state.
///
/// Built from *disjoint field borrows* rather than a `&CompilerSession`
/// method, so a caller can still take `&mut session.ctx` (for the checker)
/// while this provider is alive.
///
/// This is the type that forced `SymbolTable` and `TraitEnv` to become
/// `Sync`: `QueryProvider: Sync`, so everything it borrows must be too.
pub struct SessionQueryProvider<'a, 'input> {
    system: &'a QuerySystem,
    symbols: &'a SymbolTable<'input>,
    trait_env: &'a TraitEnv<'input>,
    resolution_map: &'a ResolutionMap<'input>,
    program: &'a Program<'input>,
    /// Read-only borrow of the session's `TypeContext`, opt-in via
    /// [`SessionQueryProvider::with_body_check`]: a body-check compute
    /// derives its OWN worker context with `TypeContext::new_sharing`
    /// (interner shared, bindings fresh) — the provider never hands the
    /// original context to anyone as `&mut`.
    ctx: Option<&'a TypeContext<'input>>,
    /// Checker config + cross-function tables for body-check workers,
    /// opt-in via [`SessionQueryProvider::with_body_check`].
    body_check: Option<BodyCheckHarness<'input>>,
}

impl<'a, 'input> SessionQueryProvider<'a, 'input> {
    pub fn new(
        system: &'a QuerySystem,
        symbols: &'a SymbolTable<'input>,
        trait_env: &'a TraitEnv<'input>,
        resolution_map: &'a ResolutionMap<'input>,
        program: &'a Program<'input>,
    ) -> Self {
        SessionQueryProvider {
            system,
            symbols,
            trait_env,
            resolution_map,
            program,
            ctx: None,
            body_check: None,
        }
    }

    /// Opt in to body-check worker construction: attach the read-only
    /// context borrow (a compute `new_sharing`-clones from it) and the
    /// checker harness.  Builder on top of [`SessionQueryProvider::new`]
    /// so every existing `new` call site keeps its exact shape — the
    /// provider stays Sync as long as `SharedCheckState: Sync`, which the
    /// parallel pass already relies on.
    pub fn with_body_check(
        mut self,
        ctx: &'a TypeContext<'input>,
        harness: BodyCheckHarness<'input>,
    ) -> Self {
        self.ctx = Some(ctx);
        self.body_check = Some(harness);
        self
    }
}

impl<'a, 'input> QueryProvider<'input> for SessionQueryProvider<'a, 'input> {
    fn query_system(&self) -> &QuerySystem {
        self.system
    }

    fn session(&self) -> Option<&dyn CompilerSessionView<'input>> {
        Some(self)
    }
}

impl<'a, 'input> CompilerSessionView<'input> for SessionQueryProvider<'a, 'input> {
    fn symbols(&self) -> &SymbolTable<'input> {
        self.symbols
    }

    fn trait_env(&self) -> &TraitEnv<'input> {
        self.trait_env
    }

    fn resolution_map(&self) -> &ResolutionMap<'input> {
        self.resolution_map
    }

    fn program(&self) -> &Program<'input> {
        self.program
    }

    fn ctx_sharing(&self) -> Option<&TypeContext<'input>> {
        self.ctx
    }

    fn body_check_harness(&self) -> Option<&BodyCheckHarness<'input>> {
        self.body_check.as_ref()
    }
}
