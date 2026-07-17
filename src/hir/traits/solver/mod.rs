//! Posita Trait Solver
//!
//! This module implements a trait resolution engine for the Posita compiler,
//! modeled after rustc's `SelectionContext` + `FulfillmentContext` architecture.
//! It uses the existing `TraitEnv` as a read-only data source for registered impls.
//!
//! ## Architecture
//!
//! - [`obligation`] — Core types: `Obligation`, `Predicate`, `ImplSource`, `SolveError`
//! - [`forest`] — `ObligationForest`: pending obligation tracking with cycle detection
//! - [`select`] — `SelectionContext`: candidate assembly, winnowing, confirmation
//! - [`fulfill`] — `FulfillmentContext`: iterative obligation resolution loop
//! - [`builtins`] — `BuiltinTrait` identification, `compute_copy`/`compute_sized`/`compute_clone`
//! - [`project`] — `ProjectionTy` normalization, `ProjectionCache`
//! - [`coherence`] — Overlap detection for impl registration
//!
//! ## Compatibility
//!
//! All solver types use `&TraitEnv` as a read-only data source.
//! The `TraitEnv` interface is unchanged — no modifications to the
//! existing `add_impl`, `lookup_impl`, or `lookup_impl_generic` methods.

pub mod builtins;
pub mod coherence;
pub mod eval;
pub mod forest;
pub mod fulfill;
pub mod obligation;
pub mod project;
pub mod select;

#[cfg(test)]
pub mod test;

// Re-export the most commonly used types at the solver level.
pub use eval::evaluate_goal;
pub use fulfill::FulfillmentContext;
pub use obligation::{
    BuiltinImplSource, CopyKind, ImplSource, Obligation, ObligationCause, ObligationCauseCode,
    Predicate, ProjectionTy, SolveError,
};
pub use select::SelectionContext;
