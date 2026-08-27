//! SAT/SMT solver family for Posita.
//!
//! - **Solvo** (Latin: "I solve") — a DPLL SAT solver with watched literals,
//!   VSIDS-lite activity heuristic, clause normalisation, and incremental
//!   assumption-based solving. Handles small-to-medium formulas efficiently.
//!
//! Future siblings:
//! - **Puto** (Latin: "I think") — a CDCL SAT solver with clause learning
//! - **Aptus** (Latin: "suitable") — a theory solver for SMT / configuration
//!   semantics (target triples, feature dependencies, ABI constraints).

pub mod solvo;

pub use solvo::*;

// ─── Unified solver trait ────────────────────────────────────────────────────

/// A common interface for all SAT solvers in the Posita family.
///
/// This allows the compiler (or any consumer) to switch between `Solvo`,
/// `Puto`, or an external solver without changing calling code.
pub trait SatSolver {
    /// Declare a new boolean variable and return its 1-based index.
    fn new_var(&mut self) -> usize;

    /// Declare a new variable with a debug name.
    fn new_named_var(&mut self, name: &str) -> usize;

    /// Ensure that at least `idx` variables exist (1-based).
    fn ensure_var(&mut self, idx: usize);

    /// Add a CNF clause (disjunction of signed literals).
    /// Literals are 1-based: positive = var, negative = ¬var.
    fn add_clause(&mut self, clause: &[i32]);

    /// Add a unit clause (single literal).
    fn add_unit(&mut self, lit: i32);

    /// Add an implication: `a → b` ≡ `¬a ∨ b`.
    fn add_implies(&mut self, a: i32, b: i32);

    /// Add an equivalence: `a ↔ b` ≡ `(¬a ∨ b) ∧ (a ∨ ¬b)`.
    fn add_equiv(&mut self, a: i32, b: i32);

    /// Add an at-most-one constraint over `lits`.
    fn add_at_most_one(&mut self, lits: &[i32]);

    /// Add an exactly-one constraint over `lits`.
    fn add_exactly_one(&mut self, lits: &[i32]);

    /// Solve the current formula.
    fn solve(&mut self) -> SolveResult;

    /// Solve under temporary assumptions (without polluting state).
    fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult;

    /// Number of declared variables.
    fn num_vars(&self) -> usize;

    /// Number of clauses currently in the formula.
    fn num_clauses(&self) -> usize;
}
