//! # ProbeCtxt — Transactional Candidate Evaluation with Rollback
//!
//! Analogous to `rustc_next_trait_solver::solve::eval_ctxt::probe`.
//! Provides a builder-pattern probe system that:
//!
//! - Wraps candidate evaluation in a transaction (rollback on failure)
//! - Tracks `CandidateHeadUsages` so that failed candidates don't
//!   pollute the cycle head dependency tracking
//! - Supports three enter modes: basic, single-candidate, and
//!   without-propagated-nested-goals
//! - Integrates with the proof tree builder via `ProbeKind`

use std::marker::PhantomData;

use rustc_hash::FxHashMap as HashMap;

use crate::hir::traits::solver::delegate::SolverDelegate;
use crate::hir::traits::solver::eval_ctxt::EvalCtxt;
use crate::hir::traits::solver::eval_ctxt::{GoalSource, GoalStalledOn, ProbeKind};
use crate::hir::traits::solver::obligation::{ImplSource, Obligation, SolveError};
use crate::hir::traits::solver::search_graph::HeadUsages;
use crate::hir::types::DefId;

// ── Candidate source ──────────────────────────────────────────────

/// Identifies where a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateSource {
    /// A user-defined impl, identified by its index in `TraitEnv`.
    Impl(usize),
    /// A builtin trait impl (Sized, Copy, Clone, etc.).
    Builtin(BuiltinImplSource),
    /// A caller-provided bound (where-clause).
    Param,
    /// An object type bound (`dyn Trait`).
    Object(DefId),
    /// A poly/unbox type (Posita-specific).
    Poly,
    /// An auto-derived trait (future: Send-like).
    Auto,
}

/// Builtin impl source kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinImplSource {
    Sized,
    Copy,
    Clone,
    Misc,
}

// ── Candidate head usages ─────────────────────────────────────────

/// Tracks which cycle heads a candidate depends on and how
/// (inductive/coinductive path).
///
/// When a candidate fails, its `CandidateHeadUsages` are discarded,
/// preventing failed candidates from polluting the cycle head
/// dependency tracking of the parent goal.
#[derive(Debug, Default, Clone)]
pub struct CandidateHeadUsages {
    /// Map from stack depth → head usages for each cycle head.
    /// The `usize` is the index in the `SearchGraph` stack.
    pub(crate) usages:
        Option<Box<HashMap<usize, crate::hir::traits::solver::search_graph::HeadUsages>>>,
}

impl CandidateHeadUsages {
    pub fn new() -> Self {
        CandidateHeadUsages { usages: None }
    }

    /// Merge usages from another `CandidateHeadUsages`.
    /// Used when merging results from multiple candidates.
    pub fn merge_usages(&mut self, other: CandidateHeadUsages) {
        if let Some(other_usages) = other.usages {
            if let Some(ref mut self_usages) = self.usages {
                for (head_index, head) in other_usages.into_iter() {
                    let entry = self_usages.entry(head_index).or_default();
                    entry.inductive += head.inductive;
                    entry.unknown += head.unknown;
                    entry.coinductive += head.coinductive;
                    entry.forced_ambiguity += head.forced_ambiguity;
                }
            } else {
                self.usages = Some(other_usages);
            }
        }
    }
}

// ── ProbeCtxt ─────────────────────────────────────────────────────

/// A builder-pattern probe context.
///
/// Created by `EvalCtxt::probe(kind)` and consumed by `.enter(f)` /
/// `.enter_single_candidate(f)` / `.enter_without_propagated_nested_goals(f)`.
pub struct ProbeCtxt<'me, 'a, D: SolverDelegate, T> {
    pub(crate) ecx: &'me mut EvalCtxt<'a, D>,
    pub(crate) probe_kind: ProbeKind,
    pub(crate) _result: PhantomData<T>,
}

impl<D: SolverDelegate, T> ProbeCtxt<'_, '_, D, T> {
    /// Enter a basic probe with full transaction rollback.
    ///
    /// Snapshots `nested_goals` before calling `f`, and propagates them
    /// to the outer context on success.  On failure, all side effects
    /// (including nested goals) are rolled back.
    pub fn enter(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> Result<T, SolveError>,
    ) -> Result<T, SolveError> {
        let nested_goals = self.ecx.nested_goals.clone();
        self.enter_inner(f, nested_goals)
    }

    /// Enter a single-candidate probe, tracking `CandidateHeadUsages`.
    ///
    /// The returned `CandidateHeadUsages` can be used to ignore the
    /// cycle head dependencies of this candidate if it fails.
    pub fn enter_single_candidate(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> Result<T, SolveError>,
    ) -> (Result<T, SolveError>, CandidateHeadUsages) {
        let mut candidate_usages = CandidateHeadUsages::new();

        // Signal the search graph to start tracking usages for this candidate.
        self.ecx.search_graph.enter_single_candidate();

        let result = self.enter(|ecx| {
            let r = f(ecx);
            candidate_usages = ecx.search_graph.finish_single_candidate();
            r
        });

        (result, candidate_usages)
    }

    /// Enter a probe without propagating nested goals.
    ///
    /// Used for tautological obligations where the nested goals of the
    /// probe should not affect the outer context.
    pub fn enter_without_propagated_nested_goals(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> Result<T, SolveError>,
    ) -> Result<T, SolveError> {
        self.enter_inner(f, Default::default())
    }

    /// Inner implementation shared by all `enter` variants.
    ///
    /// Uses a snapshot of the `EvalCtxt` state and a transaction on
    /// `TypeContext` to enable clean rollback on failure.
    fn enter_inner(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> Result<T, SolveError>,
        propagated_nested_goals: Vec<(GoalSource, Obligation, Option<GoalStalledOn>)>,
    ) -> Result<T, SolveError> {
        let ProbeCtxt {
            ecx: outer,
            probe_kind: _,
            _result,
        } = self;

        // Save the current nested_goals and replace with the propagated ones.
        let saved_nested_goals =
            std::mem::replace(&mut outer.nested_goals, propagated_nested_goals);

        // Snapshot EvalCtxt state for rollback.
        let snap = outer.snapshot();
        outer.ctx().begin_transaction();

        let result = f(outer);

        match &result {
            Ok(_) => {
                outer.ctx().commit_transaction();
                // Keep the new nested_goals (from propagation + whatever was added).
            }
            Err(_) => {
                outer.ctx().rollback_transaction();
                outer.restore_snapshot(snap);
                // Restore the original nested_goals on failure.
                outer.nested_goals = saved_nested_goals;
            }
        }

        result
    }
}

// ── TraitProbeCtxt ────────────────────────────────────────────────

/// A specialized probe for trait candidate evaluation.
///
/// Wraps a `ProbeCtxt<ImplSource>` and adds candidate source
/// tracking.  The `.enter()` method returns a `Candidate` with the
/// source and head usages attached.
pub struct TraitProbeCtxt<'me, 'a, D: SolverDelegate> {
    pub(crate) cx: ProbeCtxt<'me, 'a, D, ImplSource>,
    pub(crate) source: CandidateSource,
}

impl<D: SolverDelegate> TraitProbeCtxt<'_, '_, D> {
    /// Evaluate the candidate inside a transaction, returning a
    /// `Candidate` with the source and head usages.
    pub fn enter(
        self,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> Result<ImplSource, SolveError>,
    ) -> Result<Candidate, SolveError> {
        let (result, head_usages) = self.cx.enter_single_candidate(f);
        result.map(|r| Candidate {
            source: self.source,
            result: r,
            head_usages,
        })
    }
}

// ── Candidate ─────────────────────────────────────────────────────

/// A selected candidate with its source and cycle head usages.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub source: CandidateSource,
    pub result: ImplSource,
    pub head_usages: CandidateHeadUsages,
}
