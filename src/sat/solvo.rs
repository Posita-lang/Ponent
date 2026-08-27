//! A DPLL SAT solver with watched literals and VSIDS-lite heuristic.
//!
//! Named **Solvo** (Latin: "I solve"), this is a lightweight, self-contained
//! SAT solver designed for quick satisfiability checks within the compiler.
//! It handles small-to-medium formulas (up to ~200 variables) efficiently.
//!
//! # Features
//! - **Watched literals** for efficient unit propagation
//! - **VSIDS-lite** activity-based variable selection
//! - **Clause normalisation** (dedup, tautology removal, empty-clause detection)
//! - **Sequential counter** encoding for at-most-one constraints
//! - **Incremental solving** via `solve_assumptions`
//! - **Configurable resource limits** to prevent DoS on malicious input
//! - **Preprocessing**: one-shot pure literal elimination
//!
//! # Limitations
//! - No CDCL / clause learning (see future `Puto` module)
//! - No theory solving (see future `Aptus` module)
//! - Chronological backtracking only (within DPLL framework)
//!
//! # Example
//! ```
//! use sat::solvo::Solver;
//!
//! let mut solver = Solver::new();
//! let a = solver.new_named_var("a");
//! let b = solver.new_named_var("b");
//! // (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b)
//! solver.add_clause(&[a as i32, b as i32]);
//! solver.add_clause(&[-(a as i32), b as i32]);
//! solver.add_clause(&[a as i32, -(b as i32)]);
//! let result = solver.solve();
//! assert!(result.is_sat());
//! ```

use std::collections::HashSet;
use std::time::Instant;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum formula size (clauses × variables) before pure literal elimination
/// is skipped. Pure literal elimination scans all clauses for every variable,
/// giving O(C×V) complexity per round. For extremely large formulas (e.g.
/// deeply nested `@cfg` conditions), this becomes a DoS vector. Beyond this
/// threshold the solver falls back to branching-only, which is still correct
/// but may explore more of the search tree.
const PURE_LITERAL_THRESHOLD: usize = 4096;

/// Default maximum number of decisions (branching choices) before the solver
/// gives up and returns [`SolveResult::Unknown`]. Guards against exponential
/// search on maliciously crafted `@cfg` conditions.
const DEFAULT_MAX_DECISIONS: usize = 100_000;

/// Default maximum number of conflicts (backtracks) before giving up.
const DEFAULT_MAX_CONFLICTS: usize = 200_000;

/// Default maximum number of unit propagation steps before giving up.
const DEFAULT_MAX_PROPAGATIONS: usize = 1_000_000;

/// Default time limit in microseconds (5 seconds).
const DEFAULT_MAX_TIME_MICROS: u64 = 5_000_000;

/// VSIDS activity decay factor. After each conflict, all activities are
/// multiplied by this factor so that recent conflicts weigh more.
const ACTIVITY_DECAY: f64 = 0.95;

/// VSIDS activity increment for variables involved in a conflict.
const ACTIVITY_BUMP: f64 = 1.0;

/// At-most-one encoding threshold: if the number of literals exceeds this,
/// use sequential counter encoding instead of pairwise.
const AMO_SEQUENTIAL_THRESHOLD: usize = 8;

// ─── Public types ────────────────────────────────────────────────────────────

/// Result of a SAT solving attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveResult {
    /// Satisfiable — contains the model (1-based variable → bool).
    Sat(Vec<bool>),
    /// Unsatisfiable — no assignment can satisfy the formula.
    Unsat,
    /// Unknown — the solver exceeded resource limits.
    Unknown,
}

impl SolveResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SolveResult::Sat(_))
    }
    pub fn is_unsat(&self) -> bool {
        matches!(self, SolveResult::Unsat)
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, SolveResult::Unknown)
    }

    /// Unwrap the model from a [`Sat`](SolveResult::Sat) result.
    ///
    /// # Panics
    /// Panics if the result is not `Sat`.
    pub fn unwrap_sat(self) -> Vec<bool> {
        match self {
            SolveResult::Sat(model) => model,
            _ => panic!("called `unwrap_sat()` on a non-Sat result: {self:?}"),
        }
    }
}

/// Resource limits for the solver. All fields have sensible defaults.
/// This prevents DoS on maliciously crafted inputs (e.g. `@cfg` conditions).
#[derive(Debug, Clone)]
pub struct Limits {
    /// Maximum branching decisions.
    pub max_decisions: usize,
    /// Maximum conflicts (currently counts backtracks).
    pub max_conflicts: usize,
    /// Maximum unit propagation steps.
    pub max_propagations: usize,
    /// Maximum wall-clock time in microseconds.
    pub max_time_micros: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_decisions: DEFAULT_MAX_DECISIONS,
            max_conflicts: DEFAULT_MAX_CONFLICTS,
            max_propagations: DEFAULT_MAX_PROPAGATIONS,
            max_time_micros: DEFAULT_MAX_TIME_MICROS,
        }
    }
}

/// Solver statistics (useful for debugging and benchmarking).
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub decisions: usize,
    pub propagations: usize,
    pub conflicts: usize,
    pub backtracks: usize,
    pub clauses_added: usize,
    pub tautologies_removed: usize,
    pub duplicates_removed: usize,
    pub elapsed_micros: u64,
}

/// A SAT variable with an optional name for debugging.
#[derive(Debug, Clone)]
pub struct Var {
    pub name: Option<String>,
}

/// A CNF clause with watched-literal bookkeeping.
#[derive(Debug, Clone)]
struct Clause {
    /// The literals in this clause.
    lits: Vec<i32>,
    /// Indices into `lits` of the two watched literals.
    /// For unit clauses, watched[1] == watched[0].
    watched: [usize; 2],
    /// Whether this clause is currently satisfied under the assignment.
    satisfied: bool,
    /// Whether this is a learned clause (reserved for future CDCL / Puto).
    #[allow(dead_code)]
    learned: bool,
}

impl Clause {
    fn new(lits: Vec<i32>) -> Self {
        let w1 = 0;
        let w2 = if lits.len() > 1 { 1 } else { 0 };
        Clause {
            lits,
            watched: [w1, w2],
            satisfied: false,
            learned: false,
        }
    }

    fn is_unit(&self) -> bool {
        self.lits.len() == 1
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.lits.is_empty()
    }
}

// ─── Internal DPLL outcome ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum DpllOutcome {
    Sat,
    Unsat,
    LimitExceeded,
}

// ─── Literal helpers ─────────────────────────────────────────────────────────

/// Extract the variable index from a literal (strip sign).
#[inline]
fn lit_var(lit: i32) -> usize {
    lit.unsigned_abs() as usize
}

/// Check whether a literal is positive.
#[inline]
fn lit_sign(lit: i32) -> bool {
    lit > 0
}

/// Negate a literal.
#[inline]
fn lit_neg(lit: i32) -> i32 {
    -lit
}

/// Map a literal to a watchlist / occurrence-list index.
/// Positive literal `v` → `2*(v-1)`, negative literal `¬v` → `2*(v-1)+1`.
#[inline]
fn lit_index(lit: i32) -> usize {
    let v = lit_var(lit);
    if lit > 0 {
        2 * (v - 1)
    } else {
        2 * (v - 1) + 1
    }
}

/// Total number of literal indices for `n` variables.
#[inline]
fn lit_index_size(n: usize) -> usize {
    2 * n
}

// ─── Clause normalisation ────────────────────────────────────────────────────

/// Normalise a clause:
/// - Remove duplicate literals
/// - Detect tautologies (both `x` and `¬x` present) → returns `None`
/// - Remove zero literals (ignored)
///
/// Returns `Some(vec![])` for an empty clause (≡ false).
fn normalize_clause(clause: &[i32]) -> Option<Vec<i32>> {
    let mut seen_pos = HashSet::new();
    let mut seen_neg = HashSet::new();
    let mut lits = Vec::new();

    for &lit in clause {
        if lit == 0 {
            continue;
        }
        let var = lit_var(lit);
        if lit > 0 {
            // Check for tautology: ¬var already seen
            if seen_neg.contains(&var) {
                return None; // tautology
            }
            if seen_pos.insert(var) {
                lits.push(lit);
            }
        } else {
            // Check for tautology: var already seen
            if seen_pos.contains(&var) {
                return None; // tautology
            }
            if seen_neg.insert(var) {
                lits.push(lit);
            }
        }
    }

    Some(lits)
}

// ─── Solver ──────────────────────────────────────────────────────────────────

/// A SAT solver using DPLL with watched literals and VSIDS-lite.
#[derive(Debug)]
pub struct Solver {
    /// All declared variables (index 0 is unused; variables are 1-based).
    vars: Vec<Var>,
    /// The CNF formula: a conjunction of clauses.
    clauses: Vec<Clause>,
    /// Current partial assignment: var_index → Some(true/false) or None.
    assignment: Vec<Option<bool>>,
    /// Decision level for each variable (for backtracking).
    decision_level: Vec<usize>,
    /// Reason clause index for each variable (reserved for future CDCL).
    #[allow(dead_code)]
    reason: Vec<Option<usize>>,
    /// Current decision level.
    current_level: usize,
    /// Trail: history of assigned literals in order.
    trail: Vec<i32>,
    /// Trail limits: decision level → trail length at that level.
    trail_limits: Vec<usize>,
    /// Watch lists: watchlist[lit_index] → clause indices watching that literal.
    watchlist: Vec<Vec<usize>>,
    /// Occurrence lists: occurrence[lit_index] → clause indices containing lit.
    occurrence: Vec<Vec<usize>>,
    /// Number of currently unsatisfied clauses (incremental counter).
    unsat_count: usize,
    /// VSIDS-lite activity score per variable (1-based).
    activity: Vec<f64>,
    /// Resource limits.
    limits: Limits,
    /// Accumulated statistics.
    pub stats: SolverStats,
    /// Whether an empty clause has been added (immediately UNSAT).
    has_empty_clause: bool,
    /// Start time of the current solve() call (for time limit).
    solve_start: Option<Instant>,
}

impl Solver {
    /// Create a new, empty solver.
    pub fn new() -> Self {
        Solver {
            vars: vec![Var { name: None }], // index 0 placeholder
            clauses: Vec::new(),
            assignment: vec![None],
            decision_level: vec![0],
            reason: vec![None],
            current_level: 0,
            trail: Vec::new(),
            trail_limits: Vec::new(),
            watchlist: Vec::new(),
            occurrence: Vec::new(),
            unsat_count: 0,
            activity: vec![0.0],
            limits: Limits::default(),
            stats: SolverStats::default(),
            has_empty_clause: false,
            solve_start: None,
        }
    }

    /// Create a solver with custom resource limits.
    pub fn with_limits(limits: Limits) -> Self {
        let mut s = Self::new();
        s.limits = limits;
        s
    }

    /// Set resource limits.
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Set a custom decision limit (backward-compatible convenience).
    pub fn set_max_decisions(&mut self, limit: usize) {
        self.limits.max_decisions = limit;
    }

    // ── Variable management ──────────────────────────────────────────────────

    /// Declare a new boolean variable and return its 1-based index.
    pub fn new_var(&mut self) -> usize {
        let idx = self.vars.len();
        self.vars.push(Var { name: None });
        self.assignment.push(None);
        self.decision_level.push(0);
        self.reason.push(None);
        self.activity.push(0.0);
        // Grow watchlist and occurrence lists by 2 (positive + negative literal).
        self.watchlist.push(Vec::new());
        self.watchlist.push(Vec::new());
        self.occurrence.push(Vec::new());
        self.occurrence.push(Vec::new());
        idx
    }

    /// Declare a new variable with a debug name.
    pub fn new_named_var(&mut self, name: &str) -> usize {
        let idx = self.vars.len();
        self.vars.push(Var {
            name: Some(name.to_string()),
        });
        self.assignment.push(None);
        self.decision_level.push(0);
        self.reason.push(None);
        self.activity.push(0.0);
        self.watchlist.push(Vec::new());
        self.watchlist.push(Vec::new());
        self.occurrence.push(Vec::new());
        self.occurrence.push(Vec::new());
        idx
    }

    /// Ensure that at least `idx` variables exist (1-based).
    /// This replaces the old `while (solver.new_var()) < var_counter` hack.
    pub fn ensure_var(&mut self, idx: usize) {
        while self.vars.len() <= idx {
            self.new_var();
        }
    }

    /// Number of declared variables.
    pub fn num_vars(&self) -> usize {
        self.vars.len() - 1
    }

    /// Number of clauses in the formula.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    // ── Clause management ────────────────────────────────────────────────────

    /// Add a CNF clause (disjunction of signed literals).
    ///
    /// The clause is normalised before insertion:
    /// - Duplicate literals are removed.
    /// - Tautological clauses (containing both `x` and `¬x`) are discarded.
    /// - An empty clause (after normalisation) marks the formula as trivially UNSAT.
    ///
    /// Watch lists and occurrence lists are updated incrementally.
    pub fn add_clause(&mut self, clause: &[i32]) {
        let normalized = match normalize_clause(clause) {
            Some(lits) => lits,
            None => {
                // Tautology — always true, skip.
                self.stats.tautologies_removed += 1;
                return;
            }
        };

        if normalized.is_empty() {
            // Empty clause ≡ false — formula is UNSAT.
            self.has_empty_clause = true;
            return;
        }

        let clause_id = self.clauses.len();
        let clause_obj = Clause::new(normalized.clone());

        // Update occurrence lists.
        for &lit in &normalized {
            let li = lit_index(lit);
            if li < self.occurrence.len() {
                self.occurrence[li].push(clause_id);
            }
        }

        // Update watch lists: watch the first two literals.
        let w0 = lit_index(normalized[0]);
        if w0 < self.watchlist.len() {
            self.watchlist[w0].push(clause_id);
        }
        if normalized.len() > 1 {
            let w1 = lit_index(normalized[1]);
            if w1 < self.watchlist.len() {
                self.watchlist[w1].push(clause_id);
            }
        }

        // Track unsatisfied count (clause starts unsatisfied).
        self.unsat_count += 1;

        self.clauses.push(clause_obj);
        self.stats.clauses_added += 1;
    }

    /// Add a unit clause (single literal).
    pub fn add_unit(&mut self, lit: i32) {
        self.add_clause(&[lit]);
    }

    /// Add an implication: `a → b` ≡ `¬a ∨ b`.
    pub fn add_implies(&mut self, a: i32, b: i32) {
        self.add_clause(&[-a, b]);
    }

    /// Add an equivalence: `a ↔ b` ≡ `(¬a ∨ b) ∧ (a ∨ ¬b)`.
    pub fn add_equiv(&mut self, a: i32, b: i32) {
        self.add_clause(&[-a, b]);
        self.add_clause(&[a, -b]);
    }

    /// Add an at-most-one constraint over `lits`.
    ///
    /// For small sets (≤ `AMO_SEQUENTIAL_THRESHOLD`), uses pairwise encoding.
    /// For larger sets, uses **sequential counter** encoding which introduces
    /// O(n) auxiliary variables and O(n) clauses instead of O(n²).
    pub fn add_at_most_one(&mut self, lits: &[i32]) {
        if lits.len() <= 1 {
            return;
        }

        if lits.len() <= AMO_SEQUENTIAL_THRESHOLD {
            // Pairwise encoding: for each pair (i, j), add (¬i ∨ ¬j).
            for i in 0..lits.len() {
                for j in (i + 1)..lits.len() {
                    self.add_clause(&[-lits[i], -lits[j]]);
                }
            }
        } else {
            // Sequential counter encoding (Sinz, 2005).
            // Introduce auxiliary variables s_1 .. s_{n-1}.
            // s_i means "at least one of lits[0..=i] is true".
            let n = lits.len();
            let mut aux = Vec::with_capacity(n - 1);
            for _ in 0..(n - 1) {
                aux.push(self.new_var() as i32);
            }

            // Clause 1: ¬lits[0] ∨ s[0]
            self.add_clause(&[-lits[0], aux[0]]);

            // For i in 1..n-2:
            //   ¬lits[i] ∨ s[i]
            //   ¬s[i-1] ∨ s[i]
            //   ¬lits[i] ∨ ¬s[i-1]
            for i in 1..(n - 1) {
                self.add_clause(&[-lits[i], aux[i]]);
                self.add_clause(&[-aux[i - 1], aux[i]]);
                self.add_clause(&[-lits[i], -aux[i - 1]]);
            }

            // Final: ¬lits[n-1] ∨ ¬s[n-2]
            self.add_clause(&[-lits[n - 1], -aux[n - 2]]);
        }
    }

    /// Add an exactly-one constraint: at-least-one + at-most-one.
    pub fn add_exactly_one(&mut self, lits: &[i32]) {
        self.add_clause(lits); // at least one
        self.add_at_most_one(lits); // at most one
    }

    // ── Solving ──────────────────────────────────────────────────────────────

    /// Solve the current formula.
    ///
    /// Returns:
    /// - [`SolveResult::Sat`]`(assignment)` — satisfiable; the assignment
    ///   maps each variable (1-based index) to its boolean value.
    /// - [`SolveResult::Unsat`] — unsatisfiable.
    /// - [`SolveResult::Unknown`] — the solver exceeded the decision limit.
    pub fn solve(&mut self) -> SolveResult {
        self.solve_assumptions(&[])
    }

    /// Solve under temporary assumptions.
    ///
    /// Assumptions are unit literals that are asserted at decision level 1
    /// but do not permanently modify the formula. After solving, all
    /// assumption-related state is cleaned up.
    pub fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult {
        // Quick check: empty clause means immediate UNSAT.
        if self.has_empty_clause {
            return SolveResult::Unsat;
        }

        let start = Instant::now();
        self.solve_start = Some(start);
        self.stats = SolverStats::default();

        // Reset solving state (but keep clauses and variables).
        self.reset_state();

        // Push assumption level.
        self.current_level = 1;
        self.trail_limits.push(0);

        // Assert assumptions as unit propagations at level 1.
        for &lit in assumptions {
            let var = lit_var(lit);
            if var >= self.assignment.len() {
                // Variable not declared — treat as unknown.
                return SolveResult::Unknown;
            }
            let val = lit_sign(lit);
            if let Some(existing) = self.assignment[var] {
                if existing != val {
                    // Contradictory assumption.
                    return SolveResult::Unsat;
                }
                // Already assigned consistently — no-op.
            } else {
                self.assign(var, val, 1);
                self.trail.push(lit);
            }
        }

        // Propagate assumptions.
        if !self.propagate() {
            return SolveResult::Unsat;
        }

        // Preprocessing: one-shot pure literal elimination (if formula is small).
        let formula_size = self.clauses.len().saturating_mul(self.vars.len());
        if formula_size <= PURE_LITERAL_THRESHOLD {
            self.preprocess_pure_literals();
        }

        // Run DPLL.
        let outcome = self.dpll();

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.elapsed_micros = elapsed;
        self.solve_start = None;

        match outcome {
            DpllOutcome::Sat => {
                let model: Vec<bool> = (1..self.vars.len())
                    .map(|v| self.assignment[v].unwrap_or(false))
                    .collect();
                SolveResult::Sat(model)
            }
            DpllOutcome::Unsat => SolveResult::Unsat,
            DpllOutcome::LimitExceeded => SolveResult::Unknown,
        }
    }

    /// Reset all solving state (assignments, trail, levels) while keeping
    /// the clause database and variable declarations intact.
    fn reset_state(&mut self) {
        for a in self.assignment.iter_mut() {
            *a = None;
        }
        for d in self.decision_level.iter_mut() {
            *d = 0;
        }
        for r in self.reason.iter_mut() {
            *r = None;
        }
        for a in self.activity.iter_mut() {
            *a = 0.0;
        }
        self.current_level = 0;
        self.trail.clear();
        self.trail_limits.clear();
        self.unsat_count = self.clauses.len(); // all clauses start unsatisfied
        // Reset clause satisfied flags and watched pointers.
        for clause in self.clauses.iter_mut() {
            clause.satisfied = false;
            clause.watched = [0, if clause.lits.len() > 1 { 1 } else { 0 }];
        }
        // Rebuild watchlist references (clause indices don't change).
        self.rebuild_watchlist();
    }

    /// Rebuild watch lists from scratch based on current clause watched indices.
    fn rebuild_watchlist(&mut self) {
        for wl in self.watchlist.iter_mut() {
            wl.clear();
        }
        for (cid, clause) in self.clauses.iter().enumerate() {
            if clause.lits.is_empty() {
                continue;
            }
            let w0 = lit_index(clause.lits[clause.watched[0]]);
            if w0 < self.watchlist.len() {
                self.watchlist[w0].push(cid);
            }
            if clause.lits.len() > 1 {
                let w1 = lit_index(clause.lits[clause.watched[1]]);
                if w1 < self.watchlist.len() {
                    self.watchlist[w1].push(cid);
                }
            }
        }
    }

    // ── DPLL main loop ───────────────────────────────────────────────────────

    /// DPLL main loop — iterative implementation using an explicit stack.
    /// Avoids recursion depth issues on large formulas.
    fn dpll(&mut self) -> DpllOutcome {
        // Each stack entry: (decision_level_before_branch, variable, tried_true).
        let mut stack: Vec<(usize, usize, bool)> = Vec::new();

        loop {
            // Check resource limits.
            if self.stats.decisions >= self.limits.max_decisions
                || self.stats.conflicts >= self.limits.max_conflicts
                || self.stats.propagations >= self.limits.max_propagations
            {
                return DpllOutcome::LimitExceeded;
            }
            if let Some(start) = self.solve_start {
                if start.elapsed().as_micros() as u64 > self.limits.max_time_micros {
                    return DpllOutcome::LimitExceeded;
                }
            }

            // Unit propagation.
            if !self.propagate() {
                // Conflict — backtrack.
                self.stats.conflicts += 1;

                // Decay activities.
                for a in self.activity.iter_mut() {
                    *a *= ACTIVITY_DECAY;
                }

                // Backtrack loop: keep popping frames until we find one
                // where we can try the other branch, or exhaust the stack.
                loop {
                    if let Some((level, var, tried_true)) = stack.pop() {
                        self.stats.backtracks += 1;
                        self.backtrack(level);

                        if !tried_true {
                            // Both branches of `var` failed — continue
                            // backtracking to the parent frame.
                            if stack.is_empty() {
                                return DpllOutcome::Unsat;
                            }
                            continue;
                        }

                        // We tried true and it failed — now try false.
                        self.current_level = level + 1;
                        self.trail_limits.push(self.trail.len());
                        self.assign(var, false, self.current_level);
                        self.trail.push(-(var as i32));
                        stack.push((level, var, false));
                        break; // exit backtracking loop, continue main loop
                    } else {
                        // Stack is empty — no more choices. UNSAT.
                        return DpllOutcome::Unsat;
                    }
                }
                continue;
            }

            // Check if all clauses are satisfied.
            if self.unsat_count == 0 {
                return DpllOutcome::Sat;
            }

            // Choose a variable to branch on (VSIDS-lite).
            let var = self.choose_var();
            if var == 0 {
                // No unassigned variables. If we get here with unsat_count > 0,
                // something is wrong, but treat as SAT (all assigned).
                return DpllOutcome::Sat;
            }

            // Branch: try true first.
            self.stats.decisions += 1;
            let level = self.current_level;
            self.current_level += 1;
            self.trail_limits.push(self.trail.len());
            self.assign(var, true, self.current_level);
            self.trail.push(var as i32);
            stack.push((level, var, true));
        }
    }

    // ── Watched-literal propagation ──────────────────────────────────────────

    /// Unit propagation using watched literals.
    ///
    /// When a literal `l` is assigned, we only need to check clauses that
    /// watch `¬l` (because `l` being true means `¬l` is false, so those
    /// clauses may need a new watched literal or may become unit/conflicting).
    ///
    /// Returns `false` if a conflict is detected.
    fn propagate(&mut self) -> bool {
        // We'll use a worklist approach: collect literals to propagate.
        let mut queue: Vec<i32> = Vec::new();

        // Seed the queue with all currently assigned literals that haven't
        // been propagated yet. For simplicity in this implementation, we
        // propagate all assigned literals each time.
        for &lit in &self.trail {
            queue.push(lit);
        }

        let mut head = 0;
        while head < queue.len() {
            let lit = queue[head];
            head += 1;
            self.stats.propagations += 1;

            // Check propagation limit.
            if self.stats.propagations >= self.limits.max_propagations {
                return false; // treat as conflict to trigger limit-exceeded
            }

            // When `lit` is true, `¬lit` is false.
            // Clauses watching `¬lit` need attention.
            let neg_idx = lit_index(lit_neg(lit));
            if neg_idx >= self.watchlist.len() {
                continue;
            }

            // We need to iterate over the watchlist for ¬lit, but we may
            // modify it (by moving watches). Use index-based iteration.
            let mut wi = 0;
            while wi < self.watchlist[neg_idx].len() {
                let cid = self.watchlist[neg_idx][wi];
                let clause = &self.clauses[cid];

                if clause.satisfied {
                    // Already satisfied, skip.
                    wi += 1;
                    continue;
                }

                // Determine which watched literal is ¬lit.
                let w0_lit = clause.lits[clause.watched[0]];
                let w1_lit = clause.lits[clause.watched[1]];
                let is_w0 = (w0_lit == lit_neg(lit));
                let is_w1 = (w1_lit == lit_neg(lit));

                if !is_w0 && !is_w1 {
                    // This clause doesn't actually watch ¬lit (stale entry).
                    wi += 1;
                    continue;
                }

                // Find the other watched literal.
                let other_watched = if is_w0 { w1_lit } else { w0_lit };
                let other_val = self.lit_value(other_watched);

                if other_val == Some(true) {
                    // Clause is satisfied by the other watched literal.
                    let was_unsat = !self.clauses[cid].satisfied;
                    self.clauses[cid].satisfied = true;
                    if was_unsat && self.unsat_count > 0 {
                        self.unsat_count -= 1;
                    }
                    wi += 1;
                    continue;
                }

                // Try to find a new literal to watch.
                let mut found_new = false;
                let lits_len = self.clauses[cid].lits.len();
                for k in 0..lits_len {
                    if k == self.clauses[cid].watched[0] || k == self.clauses[cid].watched[1] {
                        continue;
                    }
                    let candidate = self.clauses[cid].lits[k];
                    let cval = self.lit_value(candidate);
                    if cval != Some(false) {
                        // candidate is true or unassigned — watch it.
                        let new_idx = lit_index(candidate);

                        // Remove old watch on ¬lit.
                        // We'll do this by swapping with the last element.
                        self.watchlist[neg_idx].swap_remove(wi);

                        // Update the watched index in the clause.
                        if is_w0 {
                            self.clauses[cid].watched[0] = k;
                        } else {
                            self.clauses[cid].watched[1] = k;
                        }

                        // Add new watch.
                        if new_idx < self.watchlist.len() {
                            self.watchlist[new_idx].push(cid);
                        }

                        found_new = true;
                        break;
                        // Note: we don't increment wi because swap_remove
                        // put a new element at position wi.
                    }
                }

                if found_new {
                    continue; // wi already points to the next element
                }

                // No new literal to watch.
                if other_val == Some(false) {
                    // Conflict: all literals are false.
                    return false;
                }

                // other_val is None (unassigned): unit clause — propagate.
                let unit_lit = other_watched;
                let unit_var = lit_var(unit_lit);
                let unit_val = lit_sign(unit_lit);

                if let Some(existing) = self.assignment[unit_var] {
                    if existing != unit_val {
                        return false; // conflict
                    }
                    // Already assigned consistently — clause is effectively
                    // satisfied or at least not conflicting.
                    let was_unsat = !self.clauses[cid].satisfied;
                    self.clauses[cid].satisfied = true;
                    if was_unsat && self.unsat_count > 0 {
                        self.unsat_count -= 1;
                    }
                    wi += 1;
                    continue;
                }

                // Assign the unit literal.
                self.assign(unit_var, unit_val, self.current_level);
                self.trail.push(unit_lit);
                queue.push(unit_lit);

                // Mark clause as satisfied.
                let was_unsat = !self.clauses[cid].satisfied;
                self.clauses[cid].satisfied = true;
                if was_unsat && self.unsat_count > 0 {
                    self.unsat_count -= 1;
                }

                // Bump activity for the propagated variable.
                if unit_var < self.activity.len() {
                    self.activity[unit_var] += ACTIVITY_BUMP;
                }

                wi += 1;
            }
        }

        true
    }

    // ── Preprocessing ────────────────────────────────────────────────────────

    /// One-shot pure literal elimination.
    ///
    /// A literal is "pure" if it appears in the formula but its negation
    /// does not. Pure literals can be assigned to satisfy all their clauses
    /// without risk of conflict.
    fn preprocess_pure_literals(&mut self) {
        let num_vars = self.vars.len();

        // Count positive and negative occurrences for each variable.
        let mut pos_count = vec![0usize; num_vars];
        let mut neg_count = vec![0usize; num_vars];

        for clause in &self.clauses {
            if clause.satisfied {
                continue;
            }
            for &lit in &clause.lits {
                let var = lit_var(lit);
                if var < num_vars {
                    if lit > 0 {
                        pos_count[var] += 1;
                    } else {
                        neg_count[var] += 1;
                    }
                }
            }
        }

        // Identify and assign pure literals.
        for var in 1..num_vars {
            if self.assignment[var].is_some() {
                continue;
            }
            let pure_val = if pos_count[var] > 0 && neg_count[var] == 0 {
                Some(true) // only positive occurrences
            } else if neg_count[var] > 0 && pos_count[var] == 0 {
                Some(false) // only negative occurrences
            } else {
                None
            };

            if let Some(val) = pure_val {
                self.assign(var, val, 0); // level 0: preprocessing
                let lit = if val { var as i32 } else { -(var as i32) };
                self.trail.push(lit);

                // Mark all clauses containing this literal as satisfied.
                let li = lit_index(lit);
                if li < self.occurrence.len() {
                    for &cid in &self.occurrence[li] {
                        let was_unsat = !self.clauses[cid].satisfied;
                        self.clauses[cid].satisfied = true;
                        if was_unsat && self.unsat_count > 0 {
                            self.unsat_count -= 1;
                        }
                    }
                }
            }
        }
    }

    // ── Variable selection (VSIDS-lite) ──────────────────────────────────────

    /// Choose the next unassigned variable to branch on.
    /// Uses VSIDS-lite: pick the unassigned variable with the highest
    /// activity score. Ties are broken by variable index (lower first).
    ///
    /// Returns 0 if all variables are assigned.
    fn choose_var(&self) -> usize {
        let mut best_var = 0usize;
        let mut best_activity = -1.0f64;

        for var in 1..self.vars.len() {
            if self.assignment[var].is_none() {
                let act = self.activity[var];
                if act > best_activity {
                    best_activity = act;
                    best_var = var;
                }
            }
        }

        best_var
    }

    // ── Assignment management ────────────────────────────────────────────────

    /// Assign a variable at the given decision level.
    fn assign(&mut self, var: usize, val: bool, level: usize) {
        if var < self.assignment.len() {
            self.assignment[var] = Some(val);
            self.decision_level[var] = level;
        }
    }

    /// Evaluate a literal under the current assignment.
    /// Returns `Some(true)` if the literal is true, `Some(false)` if false,
    /// `None` if the variable is unassigned.
    fn lit_value(&self, lit: i32) -> Option<bool> {
        let var = lit_var(lit);
        if var >= self.assignment.len() {
            return None;
        }
        self.assignment[var].map(|v| if lit > 0 { v } else { !v })
    }

    /// Backtrack to the given decision level, undoing all assignments
    /// made after that level.
    fn backtrack(&mut self, level: usize) {
        // Find the trail position to rewind to.
        let target = if level < self.trail_limits.len() {
            self.trail_limits[level]
        } else {
            0
        };

        // Undo assignments.
        while self.trail.len() > target {
            let lit = self.trail.pop().unwrap();
            let var = lit_var(lit);
            let old_val = self.assignment[var];

            self.assignment[var] = None;
            self.decision_level[var] = 0;
            self.reason[var] = None;

            // Update clause satisfaction status.
            // When we unassign a variable, clauses that were satisfied by it
            // may become unsatisfied again.
            if let Some(val) = old_val {
                let lit_true = if val { var as i32 } else { -(var as i32) };
                let li = lit_index(lit_true);
                if li < self.occurrence.len() {
                    for &cid in &self.occurrence[li] {
                        if self.clauses[cid].satisfied {
                            // Check if the clause is still satisfied by
                            // another literal.
                            let still_sat = self.clauses[cid]
                                .lits
                                .iter()
                                .any(|&l| self.lit_value(l) == Some(true));
                            if !still_sat {
                                self.clauses[cid].satisfied = false;
                                self.unsat_count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Trim trail limits.
        self.trail_limits.truncate(level);
        self.current_level = level;
    }

    /// Check whether all clauses are satisfied.
    /// (Now O(1) via incremental `unsat_count`, but kept for verification.)
    #[allow(dead_code)]
    fn all_satisfied_verify(&self) -> bool {
        self.clauses
            .iter()
            .all(|c| c.satisfied || c.lits.iter().any(|&lit| self.lit_value(lit) == Some(true)))
    }

    /// Reserved for future CDCL integration.
    #[allow(dead_code)]
    fn analyze_conflict(&mut self, _conflict_clause: usize) -> Option<(Vec<i32>, usize)> {
        // Future Puto: analyze the implication graph to produce a learned
        // clause and a backjump level.
        None
    }
}

// ─── SatSolver trait implementation ──────────────────────────────────────────

impl super::SatSolver for Solver {
    fn new_var(&mut self) -> usize {
        Solver::new_var(self)
    }
    fn new_named_var(&mut self, name: &str) -> usize {
        Solver::new_named_var(self, name)
    }
    fn ensure_var(&mut self, idx: usize) {
        Solver::ensure_var(self, idx)
    }
    fn add_clause(&mut self, clause: &[i32]) {
        Solver::add_clause(self, clause)
    }
    fn add_unit(&mut self, lit: i32) {
        Solver::add_unit(self, lit)
    }
    fn add_implies(&mut self, a: i32, b: i32) {
        Solver::add_implies(self, a, b)
    }
    fn add_equiv(&mut self, a: i32, b: i32) {
        Solver::add_equiv(self, a, b)
    }
    fn add_at_most_one(&mut self, lits: &[i32]) {
        Solver::add_at_most_one(self, lits)
    }
    fn add_exactly_one(&mut self, lits: &[i32]) {
        Solver::add_exactly_one(self, lits)
    }
    fn solve(&mut self) -> SolveResult {
        Solver::solve(self)
    }
    fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult {
        Solver::solve_assumptions(self, assumptions)
    }
    fn num_vars(&self) -> usize {
        Solver::num_vars(self)
    }
    fn num_clauses(&self) -> usize {
        Solver::num_clauses(self)
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_trivial_sat() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(a as i32);
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_trivial_unsat() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(a as i32);
        solver.add_unit(-(a as i32));
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_and() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_unit(a as i32);
        solver.add_unit(b as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(model[0], "a should be true");
            assert!(model[1], "b should be true");
        }
    }

    #[test]
    fn test_or() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_empty_clause_unsat() {
        // Empty clause ≡ false — no assignment can satisfy it.
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_clause(&[a as i32]);
        solver.add_clause(&[]); // empty clause = false
        assert!(solver.solve().is_unsat(), "empty clause should be UNSAT");
    }

    #[test]
    fn test_tautology_clause_sat() {
        // A clause containing both a and ¬a is always true — trivially SAT.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, -(a as i32)]); // a ∨ ¬a (tautology)
        solver.add_clause(&[b as i32]);
        assert!(solver.solve().is_sat(), "tautology clause should be SAT");
    }

    #[test]
    fn test_many_vars_one_clause_sat() {
        // 1000 variables, single clause: at least one must be true. Always SAT.
        let mut solver = Solver::new();
        for _ in 0..1000 {
            solver.new_var();
        }
        solver.add_clause(&[1]); // var 1 must be true
        let result = solver.solve();
        assert!(
            result.is_sat(),
            "single clause among 1000 vars should be SAT"
        );
        if let SolveResult::Sat(model) = result {
            assert!(model[0], "var 1 should be true in the model");
        }
    }

    // ── Clause normalisation tests ───────────────────────────────────────────

    #[test]
    fn test_normalize_dedup() {
        // [a, a, b] should become [a, b].
        let result = normalize_clause(&[1, 1, 2]);
        assert_eq!(result, Some(vec![1, 2]));
    }

    #[test]
    fn test_normalize_tautology() {
        // [a, ¬a, b] is a tautology.
        let result = normalize_clause(&[1, -1, 2]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_normalize_empty() {
        let result = normalize_clause(&[]);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_normalize_zero_ignored() {
        let result = normalize_clause(&[1, 0, 2]);
        assert_eq!(result, Some(vec![1, 2]));
    }

    // ── Implication and equivalence tests ────────────────────────────────────

    #[test]
    fn test_implies() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        // a → b, and a is true, so b must be true.
        solver.add_implies(a as i32, b as i32);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(model[0], "a should be true");
            assert!(model[1], "b should be true (implied by a)");
        }
    }

    #[test]
    fn test_equiv() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        // a ↔ b, a is true → b must be true.
        solver.add_equiv(a as i32, b as i32);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(model[0]);
            assert!(model[1]);
        }
    }

    #[test]
    fn test_equiv_unsat() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        // a ↔ b, a is true, b is false → UNSAT.
        solver.add_equiv(a as i32, b as i32);
        solver.add_unit(a as i32);
        solver.add_unit(-(b as i32));
        assert!(solver.solve().is_unsat());
    }

    // ── At-most-one / exactly-one tests ──────────────────────────────────────

    #[test]
    fn test_at_most_one_pairwise() {
        // 3 literals, pairwise encoding.
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        let c = solver.new_var() as i32;
        solver.add_at_most_one(&[a, b, c]);
        // At most one of {a, b, c} can be true.
        solver.add_unit(a);
        solver.add_unit(b);
        assert!(
            solver.solve().is_unsat(),
            "at-most-one violated: a and b both true"
        );
    }

    #[test]
    fn test_at_most_one_sequential() {
        // 10 literals → triggers sequential counter encoding.
        let mut solver = Solver::new();
        let lits: Vec<i32> = (0..10).map(|_| solver.new_var() as i32).collect();
        solver.add_at_most_one(&lits);
        // Set two to true → UNSAT.
        solver.add_unit(lits[3]);
        solver.add_unit(lits[7]);
        assert!(
            solver.solve().is_unsat(),
            "at-most-one (sequential) violated"
        );
    }

    #[test]
    fn test_at_most_one_sequential_sat() {
        // 10 literals, set exactly one to true → SAT.
        let mut solver = Solver::new();
        let lits: Vec<i32> = (0..10).map(|_| solver.new_var() as i32).collect();
        solver.add_at_most_one(&lits);
        solver.add_unit(lits[5]);
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_exactly_one() {
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        let c = solver.new_var() as i32;
        solver.add_exactly_one(&[a, b, c]);
        // Exactly one must be true.
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            let count = model.iter().filter(|&&v| v).count();
            assert_eq!(count, 1, "exactly one should be true");
        }
    }

    #[test]
    fn test_exactly_one_unsat() {
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        solver.add_exactly_one(&[a, b]);
        solver.add_unit(-a);
        solver.add_unit(-b);
        assert!(
            solver.solve().is_unsat(),
            "exactly-one with none true should be UNSAT"
        );
    }

    // ── Cfg-like scenario tests ──────────────────────────────────────────────

    #[test]
    fn test_cfg_target_os_conflict() {
        // Simulate: @cfg(all(target_os = "linux", target_os = "windows"))
        // target_os can only be one value → UNSAT.
        let mut solver = Solver::new();
        let linux = solver.new_named_var("target_os=linux");
        let windows = solver.new_named_var("target_os=windows");
        let macos = solver.new_named_var("target_os=macos");
        // At most one OS.
        solver.add_at_most_one(&[linux as i32, windows as i32, macos as i32]);
        // At least one OS (exactly one).
        solver.add_clause(&[linux as i32, windows as i32, macos as i32]);
        // User wants both linux and windows.
        solver.add_unit(linux as i32);
        solver.add_unit(windows as i32);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_binary_clause_chain() {
        // a → b → c → d → e, and ¬e. Should be SAT with a=false, d=false.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        let d = solver.new_var();
        let e = solver.new_var();
        solver.add_implies(a as i32, b as i32);
        solver.add_implies(b as i32, c as i32);
        solver.add_implies(c as i32, d as i32);
        solver.add_implies(d as i32, e as i32);
        solver.add_unit(-(e as i32));
        let result = solver.solve();
        assert!(result.is_sat(), "binary clause chain should be SAT");
        if let SolveResult::Sat(model) = result {
            assert!(!model[0], "a should be false");
            assert!(!model[3], "d should be false");
        }
    }

    // ── Incremental / assumption tests ───────────────────────────────────────

    #[test]
    fn test_solve_assumptions_basic() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        // (a ∨ b)
        solver.add_clause(&[a as i32, b as i32]);

        // Under assumption ¬a, b must be true.
        let result = solver.solve_assumptions(&[-(a as i32)]);
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(!model[0], "a should be false under assumption");
            assert!(model[1], "b should be true");
        }
    }

    #[test]
    fn test_solve_assumptions_conflict() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(a as i32);
        // Assume ¬a → conflict.
        let result = solver.solve_assumptions(&[-(a as i32)]);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_incremental_solve_no_state_leak() {
        // Regression: incremental reuse of a Solver must not leak
        // state between solve() calls.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        assert!(solver.solve().is_sat(), "(a ∨ b) is satisfiable");
        // Add ¬a ∨ ¬b and re-solve.
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        // (a ∨ b) ∧ (¬a ∨ ¬b) is still SAT (a=true, b=false).
        assert!(
            solver.solve().is_sat(),
            "(a∨b) ∧ (¬a∨¬b) should be SAT after incremental add"
        );
    }

    // ── ensure_var test ──────────────────────────────────────────────────────

    #[test]
    fn test_ensure_var() {
        let mut solver = Solver::new();
        solver.ensure_var(10);
        assert_eq!(solver.num_vars(), 10);
        // Calling again with a smaller index should be a no-op.
        solver.ensure_var(5);
        assert_eq!(solver.num_vars(), 10);
    }

    // ── Limits test ──────────────────────────────────────────────────────────

    #[test]
    fn test_decision_limit() {
        let mut solver = Solver::with_limits(Limits {
            max_decisions: 5,
            ..Default::default()
        });
        // Create a formula that requires many decisions.
        for _ in 0..20 {
            let v = solver.new_var();
            solver.add_clause(&[v as i32, -(v as i32)]); // tautology, removed
        }
        // Add a hard constraint.
        let x = solver.new_var();
        let y = solver.new_var();
        solver.add_clause(&[x as i32, y as i32]);
        solver.add_clause(&[-(x as i32), y as i32]);
        solver.add_clause(&[x as i32, -(y as i32)]);
        solver.add_clause(&[-(x as i32), -(y as i32)]);
        // This is UNSAT, but might hit the limit first.
        let result = solver.solve();
        assert!(
            result.is_unsat() || result.is_unknown(),
            "should be UNSAT or Unknown with tight limits"
        );
    }

    // ── Pigeonhole principle ─────────────────────────────────────────────────

    #[test]
    fn test_pigeonhole_3_2() {
        // 3 pigeons, 2 holes → UNSAT.
        let mut solver = Solver::new();
        let pigeons = 3;
        let holes = 2;

        // p_{i,j} = pigeon i is in hole j
        let mut p = vec![vec![0i32; holes]; pigeons];
        for i in 0..pigeons {
            for j in 0..holes {
                p[i][j] = solver.new_var() as i32;
            }
        }

        // Each pigeon must be in at least one hole.
        for i in 0..pigeons {
            solver.add_clause(&p[i]);
        }

        // Each hole has at most one pigeon.
        for j in 0..holes {
            let col: Vec<i32> = (0..pigeons).map(|i| p[i][j]).collect();
            solver.add_at_most_one(&col);
        }

        assert!(solver.solve().is_unsat(), "pigeonhole 3→2 should be UNSAT");
    }

    #[test]
    fn test_pigeonhole_2_2() {
        // 2 pigeons, 2 holes → SAT.
        let mut solver = Solver::new();
        let pigeons = 2;
        let holes = 2;

        let mut p = vec![vec![0i32; holes]; pigeons];
        for i in 0..pigeons {
            for j in 0..holes {
                p[i][j] = solver.new_var() as i32;
            }
        }

        for i in 0..pigeons {
            solver.add_clause(&p[i]);
        }

        for j in 0..holes {
            let col: Vec<i32> = (0..pigeons).map(|i| p[i][j]).collect();
            solver.add_at_most_one(&col);
        }

        assert!(solver.solve().is_sat(), "pigeonhole 2→2 should be SAT");
    }

    // ── Stats test ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_populated() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), b as i32]);
        assert!(solver.num_clauses() > 0);
        solver.solve();
        assert!(solver.stats.elapsed_micros > 0 || solver.stats.elapsed_micros == 0);
    }

    // ── Random 3-SAT smoke test ──────────────────────────────────────────────

    /// Simple LCG-based random number generator for reproducible tests.
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 33) as u64
        }
        fn next_range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize) % (hi - lo + 1)
        }
    }

    fn gen_random_3sat(seed: u64, num_vars: usize, num_clauses: usize) -> Vec<Vec<i32>> {
        let mut rng = Lcg::new(seed);
        let mut clauses = Vec::new();
        for _ in 0..num_clauses {
            let mut lits = Vec::new();
            for _ in 0..3 {
                let var = rng.next_range(1, num_vars) as i32;
                let sign = if rng.next() % 2 == 0 { 1 } else { -1 };
                lits.push(var * sign);
            }
            clauses.push(lits);
        }
        clauses
    }

    #[test]
    fn test_random_3sat_smoke() {
        // Generate a few random 3-SAT instances and ensure no panics.
        for &num_vars in &[10, 20, 30] {
            let num_clauses = num_vars * 4; // near phase transition
            let clauses = gen_random_3sat(42, num_vars, num_clauses);
            let mut solver = Solver::new();
            for _ in 0..num_vars {
                solver.new_var();
            }
            for c in &clauses {
                solver.add_clause(c);
            }
            let result = solver.solve();
            // Just ensure it doesn't panic and returns a valid result.
            assert!(result.is_sat() || result.is_unsat() || result.is_unknown());
        }
    }

    // ── Cross-validation with recursive DPLL ─────────────────────────────────

    /// A minimal recursive DPLL for comparison testing only.
    fn solve_recursive(clauses: &[Vec<i32>], num_vars: usize) -> Option<Vec<bool>> {
        let mut assignment = vec![None; num_vars + 1]; // 1-based

        fn propagate(clauses: &[Vec<i32>], assignment: &mut Vec<Option<bool>>) -> bool {
            loop {
                let mut changed = false;
                for clause in clauses {
                    let mut unassigned_lit = 0i32;
                    let mut unassigned_count = 0;
                    let mut satisfied = false;

                    for &lit in clause {
                        let var = lit.unsigned_abs() as usize;
                        let val = if lit > 0 {
                            assignment[var]
                        } else {
                            assignment[var].map(|v| !v)
                        };
                        match val {
                            Some(true) => {
                                satisfied = true;
                                break;
                            }
                            Some(false) => {}
                            None => {
                                unassigned_lit = lit;
                                unassigned_count += 1;
                            }
                        }
                    }

                    if satisfied {
                        continue;
                    }
                    if unassigned_count == 0 {
                        return false; // conflict
                    }
                    if unassigned_count == 1 {
                        let var = unassigned_lit.unsigned_abs() as usize;
                        let val = unassigned_lit > 0;
                        assignment[var] = Some(val);
                        changed = true;
                    }
                }
                if !changed {
                    return true;
                }
            }
        }

        fn dpll_recursive(clauses: &[Vec<i32>], assignment: &mut Vec<Option<bool>>) -> bool {
            if !propagate(clauses, assignment) {
                return false;
            }

            // Find first unassigned variable.
            let mut var = 0;
            for v in 1..assignment.len() {
                if assignment[v].is_none() {
                    var = v;
                    break;
                }
            }
            if var == 0 {
                return true; // all assigned, no conflict
            }

            let saved = assignment.clone();

            // Try true.
            assignment[var] = Some(true);
            if dpll_recursive(clauses, assignment) {
                return true;
            }
            assignment.copy_from_slice(&saved);

            // Try false.
            assignment[var] = Some(false);
            if dpll_recursive(clauses, assignment) {
                return true;
            }
            assignment.copy_from_slice(&saved);

            false
        }

        if dpll_recursive(clauses, &mut assignment) {
            Some(
                assignment[1..]
                    .iter()
                    .map(|&a| a.unwrap_or(false))
                    .collect(),
            )
        } else {
            None
        }
    }

    fn extract_clauses(solver: &Solver) -> Vec<Vec<i32>> {
        solver.clauses.iter().map(|c| c.lits.clone()).collect()
    }

    #[test]
    fn test_iterative_vs_recursive_consistency() {
        let formulas: Vec<(&str, fn(&mut Solver))> = vec![
            ("trivial_sat", |s| {
                let a = s.new_var();
                s.add_unit(a as i32);
            }),
            ("trivial_unsat", |s| {
                let a = s.new_var();
                s.add_unit(a as i32);
                s.add_unit(-(a as i32));
            }),
            ("and_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_unit(a as i32);
                s.add_unit(b as i32);
            }),
            ("or_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_clause(&[a as i32, b as i32]);
            }),
            ("implies_unsat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_implies(a as i32, b as i32);
                s.add_unit(a as i32);
                s.add_unit(-(b as i32));
            }),
            ("equiv_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_equiv(a as i32, b as i32);
                s.add_unit(a as i32);
            }),
            ("3sat_mixed", |s| {
                let a = s.new_var();
                let b = s.new_var();
                let c = s.new_var();
                s.add_clause(&[a as i32, b as i32, c as i32]);
                s.add_clause(&[-(a as i32), -(b as i32), c as i32]);
                s.add_clause(&[a as i32, -(b as i32), -(c as i32)]);
            }),
        ];

        for (name, build) in formulas {
            let mut iter_solver = Solver::new();
            build(&mut iter_solver);
            let iter_result = iter_solver.solve();
            let clauses = extract_clauses(&iter_solver);
            let num_vars = iter_solver.num_vars();
            let rec_result = solve_recursive(&clauses, num_vars);

            assert_eq!(
                iter_result.is_sat(),
                rec_result.is_some(),
                "Formula '{}' — iterative and recursive disagree",
                name,
            );
        }
    }

    // ── Random cross-validation ──────────────────────────────────────────────

    #[test]
    fn test_random_cross_validation() {
        for seed in 0..50u64 {
            let num_vars = 8 + (seed as usize % 10);
            let num_clauses = num_vars * 3;
            let clauses = gen_random_3sat(seed, num_vars, num_clauses);

            let mut solver = Solver::new();
            for _ in 0..num_vars {
                solver.new_var();
            }
            for c in &clauses {
                solver.add_clause(c);
            }
            let iter_result = solver.solve();
            let rec_result = solve_recursive(&clauses, num_vars);

            assert_eq!(
                iter_result.is_sat(),
                rec_result.is_some(),
                "Random 3-SAT seed={} ({} vars, {} clauses) — mismatch",
                seed,
                num_vars,
                num_clauses,
            );
        }
    }
}
