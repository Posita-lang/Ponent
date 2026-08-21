//! # CheckerProbe — Paired Inference + Binding Transactional Probes
//!
//! rustc analog: `InferCtxt::probe` (old solver) / `ProbeCtxt` (new
//! solver — already ported in `traits/solver/eval_ctxt/probe.rs`, which
//! covers only the solver domain: nested goals, search-graph head
//! usages, and a `TypeContext` transaction).  `CheckerProbe` extends the
//! pattern with the INFERENCE domain: an `InferenceContext` snapshot is
//! paired with a `TypeContext` transaction so a failed candidate
//! evaluation rolls back inference state (vars, resolutions, guards,
//! gen statuses, forward refs — the undo log) and type bindings in
//! LOCKSTEP, together with any diagnostics the failed attempt pushed.
//!
//! OmniML note: the solver CORE is deliberately backtracking-free
//! (OmniML.md §2.5: "tractable, backtracking-free inference" — type
//! principality makes local decisions optimal, and the suspended-match
//! semantics are chosen so a non-backtracking solver is complete).
//! Probes are for SELECTION phases — candidate disambiguation, receiver
//! probing — never for constraint solving.  The OCaml reference uses
//! the same shape at the unification level (`Unify.try_unify_or_rollback`
//! in `lib/constraint_solver/generalization.ml`).
//!
//! Validity: a probe may only be opened while the current scope's
//! `InferenceContext` is LIVE (before `solve`/`finalize`).  Rolling
//! back a solved context would desync the undo log from the bindings
//! the solver already committed to `TypeContext`.
//!
//! Residual non-reversible state (documented in `rollback_to`):
//! `wait_lists` takes on wake paths and `type_vars[i].region_id` in the
//! S-Exists-Lower level fallback — conservative (a var stays PG until
//! its next wake), never unsound.  `next_var_id` is monotonic — rolled
//! back vars leave gaps in the id sequence, which is harmless (ids only
//! index per-var vectors).

use crate::hir::checker::TypeChecker;

/// An open probe guard, holding the lockstep pair of
/// (`InferenceContext` snapshot, `TypeContext` transaction depth) plus
/// the diagnostics watermark.
///
/// Access the checker through [`CheckerProbe::with`]; on success call
/// [`CheckerProbe::commit`]; on failure just drop the guard — `Drop`
/// rolls back all three domains (including on unwind).
///
/// Production consumer: the generic-impl fallback in
/// `lookup_method_uncached` (checker/mod.rs) runs each candidate
/// attempt inside a probe, committing only the attempt that finds the
/// requested method.
pub struct CheckerProbe<'a, 'c, 'input> {
    checker: &'a mut TypeChecker<'c, 'input>,
    snapshot_len: usize,
    diag_len: usize,
    committed: bool,
}

impl<'a, 'c, 'input> CheckerProbe<'a, 'c, 'input> {
    /// Run `f` with mutable access to the checker.  Returns `f`'s
    /// result verbatim; the probe stays open regardless.
    pub fn with<T, E>(
        &mut self,
        f: impl for<'b> FnOnce(&'b mut TypeChecker<'c, 'input>) -> Result<T, E>,
    ) -> Result<T, E> {
        f(&mut *self.checker)
    }

    /// Commit the probe: the snapshot is discarded and the transaction
    /// is committed (in `Drop`), keeping everything the probe did.
    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for CheckerProbe<'_, '_, '_> {
    fn drop(&mut self) {
        if self.committed {
            self.checker.infer.commit_snapshot(self.snapshot_len);
            self.checker.ctx.commit_transaction();
        } else {
            self.checker.infer.rollback_to(self.snapshot_len);
            self.checker.ctx.rollback_transaction();
            self.checker.diagnostics.truncate_unreported(self.diag_len);
        }
    }
}

impl<'input, 'c> TypeChecker<'c, 'input> {
    /// Open a probe: `InferenceContext::start_snapshot` +
    /// `TypeContext::begin_transaction` + a diagnostics watermark.
    ///
    /// The candidate-loop idiom (one probe per attempt):
    /// ```ignore
    /// for candidate in candidates {
    ///     let mut probe = self.begin_probe();
    ///     match probe.with(|c| try_candidate(c, candidate)) {
    ///         Ok(v) => { probe.commit(); return Ok(v); }
    ///         Err(_) => {} // drop rolls everything back
    ///     }
    /// }
    /// ```
    pub fn begin_probe<'a>(&'a mut self) -> CheckerProbe<'a, 'c, 'input> {
        let snapshot_len = self.infer.start_snapshot();
        let diag_len = self.diagnostics.unreported_len();
        self.ctx.begin_transaction();
        CheckerProbe {
            checker: self,
            snapshot_len,
            diag_len,
            committed: false,
        }
    }
}
