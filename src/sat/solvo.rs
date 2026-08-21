/// A minimal DPLL SAT solver for propositional logic, named **Solvo** (Latin: "I solve").
///
/// Solvo is a lightweight, self-contained SAT solver designed for quick
/// satisfiability checks within the compiler.  It is NOT a replacement for
/// industrial solvers like Z3 — it handles small formulas (tens of variables)
/// efficiently, but lacks CDCL, VSIDS, and other advanced features.
///
/// # Overview
/// - Variables are represented as `usize` indices.
/// - A literal is a signed variable: `i32` where positive = variable, negative = ¬variable.
/// - A clause is a set of literals (disjunction).
/// - A CNF formula is a set of clauses (conjunction).
///
/// The solver uses the DPLL algorithm with:
/// - Unit propagation (unit clause rule)
/// - Pure literal elimination
/// - Chronological backtracking
///
/// # Example
/// ```
/// use sat::Solver;
///
/// let mut solver = Solver::new();
/// let a = solver.new_named_var("a");
/// let b = solver.new_named_var("b");
/// // (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b)
/// solver.add_clause(&[a, b]);
/// solver.add_clause(&[-a, b]);
/// solver.add_clause(&[a, -b]);
/// assert_eq!(solver.solve(), SolveResult::Sat(vec![true, true])); // a=true, b=true
/// ```

/// Maximum formula size (clauses × variables) before pure literal elimination
/// is skipped.  Pure literal elimination scans all clauses for every variable,
/// giving O(C×V) complexity per round.  For extremely large formulas (e.g.
/// deeply nested `@cfg` conditions), this becomes a DoS vector.  Beyond this
/// threshold the solver falls back to branching-only, which is still correct
/// but may explore more of the search tree.
const PURE_LITERAL_THRESHOLD: usize = 2048;

/// Default maximum number of decisions (branching choices) before the solver
/// gives up and returns [`SolveResult::Unknown`].  Guards against exponential
/// search on maliciously crafted `@cfg` conditions.  100 000 lets the solver
/// handle formulas up to ~40–50 variables comfortably while capping runaway
/// search on pathological inputs.
const DEFAULT_MAX_DECISIONS: usize = 100_000;

/// A SAT variable with an optional name for debugging.
#[derive(Debug, Clone)]
pub struct Var {
    pub name: Option<String>,
}

/// A CNF clause: a disjunction of literals.
/// Each literal is an `i32`: positive = variable, negative = ¬variable.
/// Variable indices are 1-based (0 is reserved).
pub type Clause = Vec<i32>;

/// The result of a SAT solver invocation.
///
/// Three states mirror the standard SAT/SMT outcome:
/// - [`Sat`](SolveResult::Sat) — the formula is satisfiable, with a model (assignment).
/// - [`Unsat`](SolveResult::Unsat) — the formula is unsatisfiable (no model exists).
/// - [`Unknown`](SolveResult::Unknown) — the solver could not decide within the
///   configured decision limit (`set_max_decisions`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveResult {
    /// The formula is satisfiable.  Contains a model: for each variable
    /// (1-based index), its assigned boolean value.
    Sat(Vec<bool>),
    /// The formula is unsatisfiable — no satisfying assignment exists.
    Unsat,
    /// The solver gave up after exceeding the decision limit without reaching
    /// a definite conclusion.  The formula may be satisfiable or unsatisfiable.
    Unknown,
}

impl SolveResult {
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

    /// Returns `true` if the result is [`Sat`](SolveResult::Sat).
    pub fn is_sat(&self) -> bool {
        matches!(self, SolveResult::Sat(_))
    }

    /// Returns `true` if the result is [`Unsat`](SolveResult::Unsat).
    pub fn is_unsat(&self) -> bool {
        matches!(self, SolveResult::Unsat)
    }

    /// Returns `true` if the result is [`Unknown`](SolveResult::Unknown).
    pub fn is_unknown(&self) -> bool {
        matches!(self, SolveResult::Unknown)
    }
}

/// A SAT solver using the DPLL algorithm.
#[derive(Debug)]
pub struct Solver {
    /// All declared variables.
    vars: Vec<Var>,
    /// The CNF formula: a conjunction of clauses.
    clauses: Vec<Clause>,
    /// Current partial assignment: var_index → Some(true/false) or None.
    assignment: Vec<Option<bool>>,
    /// Decision level for each variable (for backtracking).
    decision_level: Vec<usize>,
    /// Current decision level.
    current_level: usize,
    /// Trail: history of assigned literals in order.
    trail: Vec<i32>,
    /// Trail limits: decision level → trail length at that level.
    trail_limits: Vec<usize>,
    /// Reusable buffers for pure literal detection (avoids per-call allocation).
    pure_pos: Vec<bool>,
    pure_neg: Vec<bool>,
    /// Maximum number of decisions (branching choices) before giving up.
    max_decisions: usize,
    /// Number of decisions made so far in the current `solve()` call.
    decisions_made: usize,
}

/// Internal outcome of the DPLL loop — distinguishes "truly unsatisfiable"
/// from "hit the decision limit" so that `solve()` can map them to
/// [`SolveResult::Unsat`] and [`SolveResult::Unknown`] respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DpllOutcome {
    Sat,
    Unsat,
    LimitExceeded,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            vars: vec![Var { name: None }], // index 0 is unused (reserved)
            clauses: Vec::new(),
            assignment: vec![None],
            decision_level: vec![0],
            current_level: 0,
            trail: Vec::new(),
            trail_limits: vec![0],
            pure_pos: Vec::new(),
            pure_neg: Vec::new(),
            max_decisions: DEFAULT_MAX_DECISIONS,
            decisions_made: 0,
        }
    }

    /// Set a custom decision limit.  The solver will give up and return `None`
    /// after this many branching choices.  Use 0 to allow unbounded search
    /// (not recommended for untrusted input).
    pub fn set_max_decisions(&mut self, limit: usize) {
        self.max_decisions = limit;
    }

    /// Declare a new boolean variable. Returns its index (1-based).
    pub fn new_var(&mut self) -> usize {
        self.vars.push(Var { name: None });
        self.assignment.push(None);
        self.decision_level.push(0);
        self.vars.len() - 1
    }

    /// Declare a new named variable (for debugging).
    pub fn new_named_var(&mut self, name: &str) -> usize {
        let idx = self.new_var();
        self.vars[idx].name = Some(name.to_string());
        idx
    }

    /// Add a clause (disjunction of literals).  Literals are `i32` where
    /// positive = variable, negative = ¬variable.
    /// **Note:** An empty clause represents `false` and will cause the
    /// formula to become unsatisfiable.
    pub fn add_clause(&mut self, clause: &[i32]) {
        // 空節もそのまま追加する（空節 = false を表す）
        self.clauses.push(clause.to_vec());
    }

    /// Add a unit clause (a single literal).
    pub fn add_unit(&mut self, lit: i32) {
        self.add_clause(&[lit]);
    }

    /// Add an implication: antecedent → consequent.
    /// Equivalent to (¬antecedent ∨ consequent).
    pub fn add_implies(&mut self, antecedent: i32, consequent: i32) {
        self.add_clause(&[-antecedent, consequent]);
    }

    /// Add an equivalence: a ↔ b.
    /// Equivalent to (¬a ∨ b) ∧ (a ∨ ¬b).
    pub fn add_equiv(&mut self, a: i32, b: i32) {
        self.add_clause(&[-a, b]);
        self.add_clause(&[a, -b]);
    }

    /// Add an at-most-one constraint: at most one of the literals can be true.
    /// For each pair (i, j), adds (¬li ∨ ¬lj).
    pub fn add_at_most_one(&mut self, lits: &[i32]) {
        for i in 0..lits.len() {
            for j in (i + 1)..lits.len() {
                self.add_clause(&[-lits[i], -lits[j]]);
            }
        }
    }

    /// Add an exactly-one constraint: exactly one of the literals is true.
    /// Equivalent to at-least-one + at-most-one.
    pub fn add_exactly_one(&mut self, lits: &[i32]) {
        self.add_clause(lits); // at least one
        self.add_at_most_one(lits); // at most one
    }

    /// Solve the current formula.
    ///
    /// Returns:
    /// - [`SolveResult::Sat`]`(assignment)` — the formula is satisfiable; the
    ///   assignment maps each variable (1-based index) to its boolean value.
    /// - [`SolveResult::Unsat`] — the formula is unsatisfiable.
    /// - [`SolveResult::Unknown`] — the solver exceeded the decision limit
    ///   (`set_max_decisions`) without reaching a definite conclusion.
    pub fn solve(&mut self) -> SolveResult {
        self.decisions_made = 0;
        // Reset the ENTIRE DPLL state before each solve.  A previous
        // call's model (or a partial assignment left by a `LimitExceeded`
        // exit) must not leak into the next solve: with stale decisions
        // the propagator treats them as fixed assumptions, which can
        // prune the search and yield a FALSE UNSAT on incremental reuse
        // (e.g. `(a∨b)` then `(¬a∨¬b)` is satisfiable, but the leaked
        // `a=true` from the first solve makes the second appear unsat).
        self.assignment.iter_mut().for_each(|a| *a = None);
        self.decision_level.iter_mut().for_each(|l| *l = 0);
        self.current_level = 0;
        self.trail.clear();
        self.trail_limits.clear();
        match self.dpll() {
            DpllOutcome::Sat => {
                let result: Vec<bool> = self.assignment[1..]
                    .iter()
                    .map(|&a| a.unwrap_or(false))
                    .collect();
                SolveResult::Sat(result)
            }
            DpllOutcome::Unsat => SolveResult::Unsat,
            DpllOutcome::LimitExceeded => SolveResult::Unknown,
        }
    }

    /// DPLL main loop — iterative implementation using an explicit stack.
    /// Avoids recursion depth issues on large formulas.
    fn dpll(&mut self) -> DpllOutcome {
        let formula_size = self.clauses.len().saturating_mul(self.vars.len());
        let use_pure_literal = formula_size <= PURE_LITERAL_THRESHOLD;

        // Each entry is (decision_level_before_branch, variable, tried_true).
        // `tried_true` is per-frame so that backtracking through multiple
        // decision levels correctly explores the false branch of each level.
        let mut stack: Vec<(usize, usize, bool)> = Vec::new();

        loop {
            // ── Propagation phase ──
            let mut conflict = false;
            loop {
                if let Some(_conflict_lit) = self.unit_propagate() {
                    conflict = true;
                    break;
                }
                if use_pure_literal && let Some(lit) = self.pure_literal() {
                    self.assign(lit, true);
                    continue;
                }
                break;
            }

            if conflict {
                // Backtrack through decision levels.
                while let Some(&(level, var, tried_true)) = stack.last() {
                    self.backtrack(level);
                    stack.pop();
                    if !tried_true {
                        // We just tried true and it failed — now try false.
                        stack.push((level, var, true)); // mark this frame as tried_true
                        self.current_level += 1;
                        self.trail_limits.push(self.trail.len());
                        self.assign(-(var as i32), false);
                        break;
                    }
                    // Both branches failed at this level — pop it and continue
                    // backtracking to the previous decision level.
                }
                // If we exhausted all decision levels without finding an
                // unexplored branch, the formula is unsatisfiable.
                if stack.is_empty() {
                    return DpllOutcome::Unsat;
                }
                continue;
            }

            // ── Decision phase ──
            if self.all_satisfied() {
                return DpllOutcome::Sat;
            }

            let var = self.choose_var();
            if var == 0 {
                return DpllOutcome::Sat; // No unassigned variables — all satisfied.
            }

            // Enforce the decision limit to cap exponential search.
            self.decisions_made += 1;
            if self.decisions_made >= self.max_decisions && self.max_decisions > 0 {
                return DpllOutcome::LimitExceeded;
            }

            // Push decision point and try true.
            let decision_level = self.current_level;
            self.current_level += 1;
            self.trail_limits.push(self.trail.len());
            stack.push((decision_level, var, false));
            self.assign(var as i32, false);
        }
    }

    /// Unit propagation: find all unit clauses and assign their literals.
    /// Returns `Some(lit)` if a conflict is detected (both a literal and its
    /// negation are implied), or `None` if propagation succeeds.
    fn unit_propagate(&mut self) -> Option<i32> {
        let mut units = Vec::new();
        loop {
            units.clear();

            for clause in &self.clauses {
                if clause.is_empty() {
                    // Empty clause — unsatisfiable
                    return Some(0);
                }
                let mut unassigned = 0;
                let mut last_lit = 0;
                let mut all_false = true;
                let mut satisfied = false;
                // Track which literal in this clause was assigned most recently
                // (highest decision level) — that's the one that caused the conflict.
                let mut conflict_lit = 0;
                let mut conflict_level = 0;

                for &lit in clause {
                    let var = lit.unsigned_abs() as usize;
                    match self.assignment[var] {
                        Some(true) if lit > 0 => {
                            satisfied = true;
                            all_false = false;
                            break; // Clause is satisfied.
                        }
                        Some(false) if lit < 0 => {
                            satisfied = true;
                            all_false = false;
                            break; // Clause is satisfied.
                        }
                        Some(true) if lit < 0 => {
                            if self.decision_level[var] > conflict_level {
                                conflict_lit = lit;
                                conflict_level = self.decision_level[var];
                            }
                        }
                        Some(false) if lit > 0 => {
                            if self.decision_level[var] > conflict_level {
                                conflict_lit = lit;
                                conflict_level = self.decision_level[var];
                            }
                        }
                        Some(_) => {} // lit == 0 (shouldn't happen)
                        None => {
                            all_false = false;
                            unassigned += 1;
                            last_lit = lit;
                        }
                    }
                }

                if all_false {
                    return Some(conflict_lit);
                }

                // Only propagate the last unassigned literal as a unit if the
                // clause is NOT already satisfied by another literal.  Without
                // this check, an unassigned literal scanned before the satisfying
                // literal would be incorrectly treated as a unit, causing false
                // conflict on a later clause (see review diagnosis).
                if !satisfied && unassigned == 1 && last_lit != 0 {
                    units.push(last_lit);
                }
            }

            if units.is_empty() {
                return None;
            }
            for lit in units.drain(..) {
                self.assign(lit, true);
            }
        }
    }

    /// Pure literal elimination: find a literal that appears only in one
    /// polarity (always positive or always negative) and assign it.
    fn pure_literal(&mut self) -> Option<i32> {
        let n = self.vars.len();
        self.pure_pos.clear();
        self.pure_pos.resize(n, false);
        self.pure_neg.clear();
        self.pure_neg.resize(n, false);

        for clause in &self.clauses {
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var < self.vars.len() && self.assignment[var].is_none() {
                    if lit > 0 {
                        self.pure_pos[var] = true;
                    } else {
                        self.pure_neg[var] = true;
                    }
                }
            }
        }

        for var in 1..self.vars.len() {
            if self.assignment[var].is_none() {
                if self.pure_pos[var] && !self.pure_neg[var] {
                    return Some(var as i32);
                }
                if !self.pure_pos[var] && self.pure_neg[var] {
                    return Some(-(var as i32));
                }
            }
        }
        None
    }

    /// Assign a literal (set its variable to true/false).
    /// `_is_propagation` is reserved for future CDCL use (recording reason clauses).
    fn assign(&mut self, lit: i32, _is_propagation: bool) {
        let var = lit.unsigned_abs() as usize;
        let val = lit > 0;
        if self.assignment[var].is_none() {
            self.assignment[var] = Some(val);
            self.decision_level[var] = self.current_level;
            self.trail.push(lit);
        }
    }

    /// Check if all clauses are satisfied under the current assignment.
    fn all_satisfied(&self) -> bool {
        for clause in &self.clauses {
            let mut satisfied = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                match self.assignment[var] {
                    Some(true) if lit > 0 => {
                        satisfied = true;
                        break;
                    }
                    Some(false) if lit < 0 => {
                        satisfied = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !satisfied {
                return false;
            }
        }
        true
    }

    /// Choose an unassigned variable for branching.
    /// Returns the variable index (1-based), or 0 if all are assigned.
    fn choose_var(&self) -> usize {
        // Simple heuristic: choose the first unassigned variable.
        // In a more advanced solver, this would use VSIDS or similar.
        for var in 1..self.vars.len() {
            if self.assignment[var].is_none() {
                return var;
            }
        }
        0
    }

    /// Backtrack to the given decision level.
    fn backtrack(&mut self, level: usize) {
        // Unassign all variables at decision levels > level.
        while let Some(&lit) = self.trail.last() {
            let var = lit.unsigned_abs() as usize;
            if self.decision_level[var] <= level {
                break;
            }
            self.assignment[var] = None;
            self.decision_level[var] = 0;
            self.trail.pop();
        }
        self.current_level = level;
        while self.trail_limits.len() > level + 1 {
            self.trail_limits.pop();
        }
    }

    /// Analyze a conflict: backtrack to the previous decision level.
    /// Returns `false` to indicate that the current branch failed.
    /// The caller (dpll branching logic) will try the other branch.
    #[allow(dead_code)]
    fn analyze_conflict(&mut self, _conflict: i32) -> bool {
        // Reserved for future CDCL integration.  The current DPLL solver
        // handles conflict through the explicit stack in dpll() instead.
        if self.current_level == 0 {
            return false; // Unsatisfiable at decision level 0
        }
        self.backtrack(self.current_level - 1);
        false // This branch failed — caller should try the other branch
    }

    /// Get the variable name (for debugging).
    pub fn var_name(&self, var: usize) -> String {
        self.vars
            .get(var)
            .and_then(|v| v.name.as_ref())
            .cloned()
            .unwrap_or_else(|| format!("x{}", var))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let model = result.unwrap_sat();
        assert!(model[0]);
        assert!(model[1]);
    }

    #[test]
    fn test_or() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_unit(-(a as i32));
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(!model[0]);
        assert!(model[1]);
    }

    #[test]
    fn test_exactly_one() {
        let mut solver = Solver::new();
        let a = solver.new_named_var("a");
        let b = solver.new_named_var("b");
        let c = solver.new_named_var("c");
        solver.add_exactly_one(&[a as i32, b as i32, c as i32]);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        let true_count = model.iter().filter(|&&v| v).count();
        assert_eq!(true_count, 1);
    }

    #[test]
    fn test_implies() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_implies(a as i32, b as i32);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]);
        assert!(model[1]);
    }

    #[test]
    fn test_unsat_by_contradiction() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_unit(-(a as i32));
        solver.add_unit(-(b as i32));
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_mutual_exclusion() {
        let mut solver = Solver::new();
        let linux = solver.new_named_var("target_os_linux");
        let windows = solver.new_named_var("target_os_windows");
        let macos = solver.new_named_var("target_os_macos");
        solver.add_at_most_one(&[linux as i32, windows as i32, macos as i32]);
        solver.add_clause(&[linux as i32, windows as i32, macos as i32]);
        solver.add_unit(linux as i32);
        solver.add_unit(macos as i32);
        assert!(solver.solve().is_unsat());
    }

    // ── Additional tests ──────────────────────────────────────────

    #[test]
    fn test_equiv() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        // a ↔ b
        solver.add_equiv(a as i32, b as i32);
        // a
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(model[1]); // b true (by equivalence)
    }

    #[test]
    fn test_equiv_negation() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_equiv(a as i32, b as i32);
        // ¬a
        solver.add_unit(-(a as i32));
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(!model[0]); // a false
        assert!(!model[1]); // b false (by equivalence)
    }

    #[test]
    fn test_chain_implication() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        // a → b, b → c
        solver.add_implies(a as i32, b as i32);
        solver.add_implies(b as i32, c as i32);
        // a
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(model[1]); // b true
        assert!(model[2]); // c true
    }

    #[test]
    fn test_backtracking_choice() {
        // Formula that requires backtracking:
        // (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b)
        // Only solution: a=true, b=true
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), b as i32]);
        solver.add_clause(&[a as i32, -(b as i32)]);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(model[1]); // b true
    }

    #[test]
    fn test_backtracking_unsat() {
        // (a ∨ b) ∧ (¬a ∨ b) ∧ (a ∨ ¬b) ∧ (¬a ∨ ¬b)
        // Only solution would be a=true ∧ b=true ∧ a=false — unsat
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), b as i32]);
        solver.add_clause(&[a as i32, -(b as i32)]);
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_unit_propagation() {
        // (a ∨ b) ∧ (¬a) → unit propagation forces b
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_unit(-(a as i32));
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(!model[0]); // a false
        assert!(model[1]); // b true (unit propagation)
    }

    #[test]
    fn test_pure_literal() {
        // a ∧ (¬a ∨ b) — pure literal elimination
        // a is pure (always positive), b is pure (always positive)
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_unit(a as i32);
        solver.add_clause(&[-(a as i32), b as i32]);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(model[1]); // b true
    }

    #[test]
    fn test_empty_formula() {
        // Empty formula is always satisfiable
        let mut solver = Solver::new();
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_single_var_unsat() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_clause(&[a as i32]);
        solver.add_clause(&[-(a as i32)]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_three_var_exactly_one() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        // Exactly one of {a, b, c}
        solver.add_exactly_one(&[a as i32, b as i32, c as i32]);
        // Force a true
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(!model[1]); // b false
        assert!(!model[2]); // c false
    }

    #[test]
    fn test_at_most_one_unsat() {
        // At most one of {a, b}, but force both — unsat
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_at_most_one(&[a as i32, b as i32]);
        solver.add_unit(a as i32);
        solver.add_unit(b as i32);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_multiple_clauses_sat() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        // (a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ ¬c) ∧ a
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), c as i32]);
        solver.add_clause(&[-(b as i32), -(c as i32)]);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        // a=true, b=false, c=true is one solution
        // or a=true, b=true, c=false is another
    }

    #[test]
    fn test_large_formula() {
        // Chain of implications: a1 → a2 → a3 → ... → a10
        // Force a1 true — should propagate to all
        let mut solver = Solver::new();
        let vars: Vec<i32> = (0..10).map(|_| solver.new_var() as i32).collect();
        for i in 0..9 {
            solver.add_implies(vars[i], vars[i + 1]);
        }
        solver.add_unit(vars[0]);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        for (i, v) in vars.iter().enumerate() {
            assert!(model[*v as usize - 1], "var {} should be true", i);
        }
    }

    #[test]
    fn test_mutual_exclusion_three_force_one() {
        // At most one of {a, b, c}, force a — should be sat with a true
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_at_most_one(&[a as i32, b as i32, c as i32]);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true
        assert!(!model[1]); // b false
        assert!(!model[2]); // c false
    }

    #[test]
    fn test_assignment_consistency() {
        // After solving, verify that all clauses are satisfied
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), c as i32]);
        solver.add_clause(&[-(b as i32), -(c as i32)]);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        // Verify clause (a ∨ b)
        assert!(model[0] || model[1]);
        // Verify clause (¬a ∨ c)
        assert!(!model[0] || model[2]);
        // Verify clause (¬b ∨ ¬c)
        assert!(!model[1] || !model[2]);
    }

    #[test]
    fn test_iterative_propagation_chain() {
        // Chain of implications forced by unit clauses:
        // a → b, b → c, c → d, unit(a) forces a=true, propagates to d=true.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        let d = solver.new_var();
        solver.add_implies(a as i32, b as i32);
        solver.add_implies(b as i32, c as i32);
        solver.add_implies(c as i32, d as i32);
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(model[0]); // a true (unit)
        assert!(model[1]); // b true (propagated from a)
        assert!(model[2]); // c true (propagated from b)
        assert!(model[3]); // d true (propagated from c)
    }

    #[test]
    fn test_iterative_propagation_conflict() {
        // Unit propagation detects conflict: a → b, a → ¬b, unit(a).
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_implies(a as i32, b as i32);
        solver.add_implies(a as i32, -(b as i32));
        solver.add_unit(a as i32);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_propagation_with_backtrack() {
        // Unit propagation at different decision levels:
        // (a ∨ b) ∧ (a ∨ c) ∧ (¬a ∨ ¬b ∨ ¬c) ∧ unit(¬a)
        // ¬a forces a=false. Then (a ∨ b) becomes unit(b), forces b=true.
        // Then (a ∨ c) becomes unit(c), forces c=true.
        // (¬a ∨ ¬b ∨ ¬c) = (true ∨ false ∨ false) = satisfied.
        // So a=false, b=true, c=true is a valid model.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[a as i32, c as i32]);
        solver.add_clause(&[-(a as i32), -(b as i32), -(c as i32)]);
        solver.add_unit(-(a as i32));
        let result = solver.solve();
        assert!(result.is_sat());
        let model = result.unwrap_sat();
        assert!(!model[0]); // a false (unit)
        assert!(model[1]); // b true (propagated)
        assert!(model[2]); // c true (propagated)
    }

    #[test]
    fn test_iterative_deep_backtrack() {
        // Formula requiring deep backtracking through the iterative stack:
        // (a ∨ b ∨ c) ∧ (¬a ∨ b ∨ c) ∧ (a ∨ ¬b ∨ c) ∧ (a ∨ b ∨ ¬c)
        // ∧ (¬a ∨ ¬b) ∧ (¬a ∨ ¬c) ∧ (¬b ∨ ¬c)
        // Only solution: a=false, b=false, c=false? Let's check.
        // ¬a ∨ ¬b: a=false or b=false
        // ¬a ∨ ¬c: a=false or c=false
        // ¬b ∨ ¬c: b=false or c=false
        // At most one of a,b,c can be true.
        // (a ∨ b ∨ c): at least one must be true.
        // Contradiction — unsat.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_clause(&[a as i32, b as i32, c as i32]);
        solver.add_clause(&[-(a as i32), b as i32, c as i32]);
        solver.add_clause(&[a as i32, -(b as i32), c as i32]);
        solver.add_clause(&[a as i32, b as i32, -(c as i32)]);
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        solver.add_clause(&[-(a as i32), -(c as i32)]);
        solver.add_clause(&[-(b as i32), -(c as i32)]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_unsat_simple() {
        // Simple UNSAT: a ∧ ¬a
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(a as i32);
        solver.add_unit(-(a as i32));
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_unsat_chromatic() {
        // Graph coloring UNSAT: edge (a,b) with only 1 color.
        // a and b must both be colored, but they can't share a color.
        // With 1 color, at least one edge endpoint is uncolored — but
        // we enforce both must be colored. Contradiction.
        let mut solver = Solver::new();
        let a = solver.new_var(); // a is colored
        let b = solver.new_var(); // b is colored
        // Both must be colored.
        solver.add_unit(a as i32);
        solver.add_unit(b as i32);
        // But they can't both be colored with the same color (edge constraint).
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_unsat_pigeonhole() {
        // Pigeonhole principle: 3 pigeons, 2 holes. At least 2 pigeons
        // share a hole. Encode as: each pigeon in exactly one hole,
        // but only 2 holes available.
        // p0_in_h0, p0_in_h1, p1_in_h0, p1_in_h1, p2_in_h0, p2_in_h1
        let mut solver = Solver::new();
        let p0h0 = solver.new_var();
        let p0h1 = solver.new_var();
        let p1h0 = solver.new_var();
        let p1h1 = solver.new_var();
        let p2h0 = solver.new_var();
        let p2h1 = solver.new_var();
        // Each pigeon in exactly one hole.
        solver.add_exactly_one(&[p0h0 as i32, p0h1 as i32]);
        solver.add_exactly_one(&[p1h0 as i32, p1h1 as i32]);
        solver.add_exactly_one(&[p2h0 as i32, p2h1 as i32]);
        // At most one pigeon per hole.
        solver.add_at_most_one(&[p0h0 as i32, p1h0 as i32, p2h0 as i32]);
        solver.add_at_most_one(&[p0h1 as i32, p1h1 as i32, p2h1 as i32]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_unsat_force_contradiction() {
        // Force a contradiction through implication chain:
        // a → b, b → c, c → ¬a, unit(a). Cycle: a→b→c→¬a.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_implies(a as i32, b as i32);
        solver.add_implies(b as i32, c as i32);
        solver.add_implies(c as i32, -(a as i32));
        solver.add_unit(a as i32);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_iterative_unsat_all_combinations() {
        // Exhaustively unsatisfiable: all 2^n assignments blocked.
        // For 2 variables (a, b), add clauses blocking each assignment:
        // ¬a ∨ ¬b  (blocks a=T, b=T)
        // a ∨ b     (blocks a=F, b=F)
        // ¬a ∨ b    (blocks a=T, b=F)
        // a ∨ ¬b    (blocks a=F, b=T)
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[-(a as i32), b as i32]);
        solver.add_clause(&[a as i32, -(b as i32)]);
        assert!(solver.solve().is_unsat());
    }

    /// A minimal recursive DPLL for comparison testing only.
    /// This is a simplified, non-optimized reference implementation.
    fn solve_recursive(clauses: &[Vec<i32>], num_vars: usize) -> Option<Vec<bool>> {
        let mut assignment = vec![None; num_vars + 1]; // 1-based

        fn propagate(clauses: &[Vec<i32>], assignment: &mut [Option<bool>]) -> Option<Vec<i32>> {
            loop {
                let mut units = Vec::new();
                let mut conflict = false;

                for clause in clauses {
                    let mut unassigned = 0;
                    let mut last_lit = 0;
                    let mut satisfied = false;
                    let mut all_false = true;

                    for &lit in clause {
                        let var = lit.unsigned_abs() as usize;
                        match assignment[var] {
                            Some(true) if lit > 0 => {
                                satisfied = true;
                                break;
                            }
                            Some(false) if lit < 0 => {
                                satisfied = true;
                                break;
                            }
                            Some(_) => {}
                            None => {
                                all_false = false;
                                unassigned += 1;
                                last_lit = lit;
                            }
                        }
                    }

                    if !satisfied && all_false {
                        conflict = true;
                        break;
                    }
                    if !satisfied && unassigned == 1 && last_lit != 0 {
                        units.push(last_lit);
                    }
                }

                if conflict {
                    return None;
                }
                if units.is_empty() {
                    return Some(units);
                }
                for lit in units {
                    let var = lit.unsigned_abs() as usize;
                    assignment[var] = Some(lit > 0);
                }
            }
        }

        fn dpll_recursive(clauses: &[Vec<i32>], assignment: &mut [Option<bool>]) -> bool {
            // Propagate
            if propagate(clauses, assignment).is_none() {
                return false;
            }

            // Check all satisfied
            if clauses.iter().all(|c| {
                c.iter().any(|&lit| {
                    let var = lit.unsigned_abs() as usize;
                    match assignment[var] {
                        Some(true) if lit > 0 => true,
                        Some(false) if lit < 0 => true,
                        _ => false,
                    }
                })
            }) {
                return true;
            }

            // Choose variable
            let var = (1..=assignment.len() - 1)
                .find(|&v| assignment[v].is_none())
                .unwrap_or(0);
            if var == 0 {
                return true;
            }

            // Try true
            let saved = assignment.to_vec();
            assignment[var] = Some(true);
            if dpll_recursive(clauses, assignment) {
                return true;
            }

            // Backtrack and try false
            assignment.copy_from_slice(&saved);
            assignment[var] = Some(false);
            let result = dpll_recursive(clauses, assignment);
            if !result {
                // Both branches failed — restore to pre-decision state
                assignment.copy_from_slice(&saved);
            }
            result
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

    /// Helper: build clauses from a Solver for comparison.
    fn extract_clauses(solver: &Solver) -> Vec<Vec<i32>> {
        solver.clauses.clone()
    }

    #[test]
    fn test_iterative_vs_recursive_consistency() {
        // Build a set of formulas and verify both implementations agree.
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
                s.add_unit(-(a as i32));
            }),
            ("chain_propagation", |s| {
                let a = s.new_var();
                let b = s.new_var();
                let c = s.new_var();
                let d = s.new_var();
                s.add_implies(a as i32, b as i32);
                s.add_implies(b as i32, c as i32);
                s.add_implies(c as i32, d as i32);
                s.add_unit(a as i32);
            }),
            ("backtracking_choice", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_clause(&[a as i32, b as i32]);
                s.add_clause(&[-(a as i32), b as i32]);
                s.add_clause(&[a as i32, -(b as i32)]);
            }),
            ("backtracking_unsat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_clause(&[a as i32, b as i32]);
                s.add_clause(&[-(a as i32), b as i32]);
                s.add_clause(&[a as i32, -(b as i32)]);
                s.add_clause(&[-(a as i32), -(b as i32)]);
            }),
            ("equiv_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                s.add_equiv(a as i32, b as i32);
                s.add_unit(a as i32);
            }),
            ("exactly_one_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                let c = s.new_var();
                s.add_exactly_one(&[a as i32, b as i32, c as i32]);
            }),
            ("implication_cycle_unsat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                let c = s.new_var();
                s.add_implies(a as i32, b as i32);
                s.add_implies(b as i32, c as i32);
                s.add_implies(c as i32, -(a as i32));
                s.add_unit(a as i32);
            }),
            ("pigeonhole_unsat", |s| {
                let p0h0 = s.new_var();
                let p0h1 = s.new_var();
                let p1h0 = s.new_var();
                let p1h1 = s.new_var();
                let p2h0 = s.new_var();
                let p2h1 = s.new_var();
                s.add_exactly_one(&[p0h0 as i32, p0h1 as i32]);
                s.add_exactly_one(&[p1h0 as i32, p1h1 as i32]);
                s.add_exactly_one(&[p2h0 as i32, p2h1 as i32]);
                s.add_at_most_one(&[p0h0 as i32, p1h0 as i32, p2h0 as i32]);
                s.add_at_most_one(&[p0h1 as i32, p1h1 as i32, p2h1 as i32]);
            }),
            ("multi_var_sat", |s| {
                let a = s.new_var();
                let b = s.new_var();
                let c = s.new_var();
                s.add_clause(&[a as i32, b as i32]);
                s.add_clause(&[-(a as i32), c as i32]);
                s.add_clause(&[-(b as i32), -(c as i32)]);
            }),
        ];

        for (name, build) in formulas {
            // Build with iterative solver
            let mut iter_solver = Solver::new();
            build(&mut iter_solver);
            let iter_result = iter_solver.solve();
            let clauses = extract_clauses(&iter_solver);
            let num_vars = iter_solver.vars.len() - 1;

            // Build with recursive solver
            let rec_result = solve_recursive(&clauses, num_vars);

            // Both must agree on SAT/UNSAT
            assert_eq!(
                iter_result.is_sat(),
                rec_result.is_some(),
                "Formula '{}': iterative={}, recursive={}",
                name,
                iter_result.is_sat(),
                rec_result.is_some(),
            );

            // If both SAT, verify the iterative model satisfies all clauses
            if let SolveResult::Sat(model) = &iter_result {
                for clause in &clauses {
                    let satisfied = clause.iter().any(|&lit| {
                        let var = lit.unsigned_abs() as usize;
                        match lit > 0 {
                            true => model[var - 1],
                            false => !model[var - 1],
                        }
                    });
                    assert!(
                        satisfied,
                        "Formula '{}': iterative model violates clause {:?}",
                        name, clause,
                    );
                }
            }
        }
    }

    /// Generate a random 3-SAT formula for benchmarking.
    /// Uses a simple LCG PRNG (no external deps needed).
    fn gen_random_3sat(seed: u64, num_vars: usize, num_clauses: usize) -> Vec<Vec<i32>> {
        let mut state = seed;
        let mut rng = || -> usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as usize) % num_vars
        };

        let mut clauses = Vec::with_capacity(num_clauses);
        for _ in 0..num_clauses {
            let mut clause = Vec::with_capacity(3);
            for _ in 0..3 {
                let var = loop {
                    let v = rng() % num_vars + 1;
                    if !clause.contains(&(v as i32)) && !clause.contains(&(-(v as i32))) {
                        break v;
                    }
                };
                let neg = rng() % 2 == 0;
                clause.push(if neg { -(var as i32) } else { var as i32 });
            }
            clauses.push(clause);
        }
        clauses
    }

    #[test]
    fn bench_iterative_vs_recursive_large() {
        use std::time::Instant;

        let sizes = [(50, 200), (100, 400)];

        println!("\n── DPLL Benchmark: iterative vs recursive ──");
        println!(
            "{:<12} {:<10} {:<14} {:<14} {:<8}",
            "Vars", "Clauses", "Iterative(μs)", "Recursive(μs)", "Speedup"
        );
        println!("{}", "─".repeat(62));

        for (num_vars, num_clauses) in sizes {
            let clauses = gen_random_3sat(42, num_vars, num_clauses);

            // Iterative
            let mut iter_solver = Solver::new();
            for _ in 0..num_vars {
                iter_solver.new_var();
            }
            for clause in &clauses {
                iter_solver.add_clause(clause);
            }
            let start = Instant::now();
            let iter_result = iter_solver.solve();
            let iter_time = start.elapsed().as_micros();

            // Recursive
            let start = Instant::now();
            let rec_result = solve_recursive(&clauses, num_vars);
            let rec_time = start.elapsed().as_micros();

            // Verify consistency
            assert_eq!(
                iter_result.is_sat(),
                rec_result.is_some(),
                "Mismatch at {} vars: iter={}, rec={}",
                num_vars,
                iter_result.is_sat(),
                rec_result.is_some(),
            );

            let speedup = if iter_time > 0 {
                rec_time as f64 / iter_time as f64
            } else {
                0.0
            };

            println!(
                "{:<12} {:<10} {:<14} {:<14} {:<8.2}x",
                num_vars, num_clauses, iter_time, rec_time, speedup,
            );
        }
        println!();
    }

    #[test]
    fn debug_solver_divergence() {
        let (num_vars, num_clauses) = (50, 200);
        let clauses = gen_random_3sat(42, num_vars, num_clauses);

        let mut solver = Solver::new();
        for _ in 0..num_vars {
            solver.new_var();
        }
        for c in &clauses {
            solver.add_clause(c);
        }
        let iter_result = solver.solve();
        let rec_result = solve_recursive(&clauses, num_vars);

        if iter_result.is_sat() != rec_result.is_some() {
            let mut dimacs = format!("p cnf {} {}\n", num_vars, clauses.len());
            for c in &clauses {
                for lit in c {
                    dimacs.push_str(&format!("{} ", lit));
                }
                dimacs.push_str("0\n");
            }
            std::fs::write("/tmp/divergence.cnf", &dimacs).unwrap();
            eprintln!("DIMACS written to /tmp/divergence.cnf");
            panic!(
                "ITER={}, REC={}",
                iter_result.is_sat(),
                rec_result.is_some()
            );
        }
    }

    // ──────────────────────────────────────────────
    // Z3 cross-validation benchmarks
    // ──────────────────────────────────────────────

    /// Helper: solve a CNF formula with Z3 and return assignment.
    fn solve_with_z3(clauses: &[Vec<i32>], num_vars: usize) -> Option<Vec<bool>> {
        use z3::SatResult;
        use z3::ast::Bool;

        let solver = z3::Solver::new();

        // Create Boolean variables (1-based; index 0 unused)
        let vars: Vec<Bool> =
            std::iter::once(Bool::from_bool(true)) // dummy 0
                .chain((1..=num_vars).map(|i| Bool::new_const(format!("x{i}"))))
                .collect();

        // Assert clauses
        for clause in clauses {
            if clause.is_empty() {
                solver.assert(Bool::from_bool(false));
                continue;
            }
            let lits: Vec<Bool> = clause
                .iter()
                .map(|lit| {
                    let var = lit.unsigned_abs() as usize;
                    if *lit > 0 {
                        vars[var].clone()
                    } else {
                        vars[var].not()
                    }
                })
                .collect();
            solver.assert(Bool::or(&lits));
        }

        match solver.check() {
            SatResult::Sat => {
                let model = solver.get_model().unwrap();
                let mut assignment = vec![false; num_vars + 1];
                for i in 1..=num_vars {
                    let val = model.eval(&vars[i], true).unwrap().as_bool().unwrap();
                    assignment[i] = val;
                }
                Some(assignment[1..].to_vec())
            }
            SatResult::Unsat => None,
            SatResult::Unknown => panic!("Z3 returned Unknown for a SAT problem"),
        }
    }

    /// Cross-validate Solvo against Z3 on random 3-SAT formulas.
    #[test]
    fn bench_solvo_vs_z3_random_3sat() {
        use std::time::Instant;

        // Vary clause-to-variable ratio around the phase transition (~4.3)
        let configs = [
            (10, 20, "under-constrained (r=2.0)"),
            (10, 43, "phase-transition (r=4.3)"),
            (10, 60, "over-constrained (r=6.0)"),
            (15, 30, "under-constrained (r=2.0)"),
            (15, 65, "phase-transition (r=4.3)"),
            (15, 90, "over-constrained (r=6.0)"),
            (20, 40, "under-constrained (r=2.0)"),
            (20, 86, "phase-transition (r=4.3)"),
            (20, 120, "over-constrained (r=6.0)"),
        ];

        println!("\n── Solvo vs Z3: random 3-SAT ──");
        println!(
            "{:<8} {:<8} {:<30} {:<10} {:<10} {:<10} {:<10} {:<10}",
            "Vars", "Clauses", "Regime", "Solvo(μs)", "Z3(μs)", "Ratio", "Solvo", "Z3"
        );
        println!("{}", "─".repeat(96));

        for &(num_vars, num_clauses, label) in &configs {
            let clauses = gen_random_3sat(42 + num_vars as u64, num_vars, num_clauses);

            // Solvo
            let mut solver = Solver::new();
            for _ in 0..num_vars {
                solver.new_var();
            }
            for c in &clauses {
                solver.add_clause(c);
            }
            let start = Instant::now();
            let solvo_result = solver.solve();
            let solvo_time = start.elapsed().as_micros();

            // Z3
            let start = Instant::now();
            let z3_result = solve_with_z3(&clauses, num_vars);
            let z3_time = start.elapsed().as_micros();

            let ratio = if solvo_time > 0 {
                z3_time as f64 / solvo_time as f64
            } else {
                0.0
            };

            assert_eq!(
                solvo_result.is_sat(),
                z3_result.is_some(),
                "Mismatch at {} vars, {} clauses ({})",
                num_vars,
                num_clauses,
                label,
            );

            println!(
                "{:<8} {:<8} {:<30} {:<10} {:<10} {:<10.2} {:<10} {:<10}",
                num_vars,
                num_clauses,
                label,
                solvo_time,
                z3_time,
                ratio,
                if solvo_result.is_sat() {
                    "SAT"
                } else {
                    "UNSAT"
                },
                if z3_result.is_some() { "SAT" } else { "UNSAT" },
            );
        }
        println!();
    }

    /// Cross-validate Solvo against Z3 on known-hard pigeonhole formulas.
    #[test]
    fn bench_solvo_vs_z3_pigeonhole() {
        use std::time::Instant;

        let problems = [(2, 3), (3, 4), (4, 5), (5, 6)];

        println!("\n── Solvo vs Z3: Pigeonhole Principle ──");
        println!(
            "{:<8} {:<8} {:<10} {:<10} {:<10}",
            "Holes", "Pigeons", "Solvo(μs)", "Z3(μs)", "Result"
        );
        println!("{}", "─".repeat(46));

        for &(holes, pigeons) in &problems {
            let num_vars = holes * pigeons;
            let mut solver = Solver::new();
            // var[i][j]: pigeon i → hole j
            let mut vars = vec![vec![0usize; holes]; pigeons];
            for i in 0..pigeons {
                for j in 0..holes {
                    vars[i][j] = solver.new_var();
                }
            }

            // Each pigeon goes to at least one hole
            for i in 0..pigeons {
                let clause: Vec<i32> = (0..holes).map(|j| vars[i][j] as i32).collect();
                solver.add_clause(&clause);
            }

            // No two pigeons share a hole (at-most-one per hole)
            for j in 0..holes {
                for i1 in 0..pigeons {
                    for i2 in (i1 + 1)..pigeons {
                        solver.add_clause(&[-(vars[i1][j] as i32), -(vars[i2][j] as i32)]);
                    }
                }
            }

            // Extract clauses for Z3
            let clauses = extract_clauses(&solver);

            // Solvo
            let start = Instant::now();
            let solvo_result = solver.solve();
            let solvo_time = start.elapsed().as_micros();

            // Z3
            let start = Instant::now();
            let z3_result = solve_with_z3(&clauses, num_vars);
            let z3_time = start.elapsed().as_micros();

            // Pigeonhole with n+1 pigeons into n holes is UNSAT
            assert_eq!(
                solvo_result.is_sat(),
                z3_result.is_some(),
                "Pigeonhole mismatch: {} pigeons, {} holes",
                pigeons,
                holes,
            );
            assert!(
                solvo_result.is_unsat(),
                "Pigeonhole {}→{} should be UNSAT",
                pigeons,
                holes,
            );

            println!(
                "{:<8} {:<8} {:<10} {:<10} {:<10}",
                holes, pigeons, solvo_time, z3_time, "UNSAT ✓",
            );
        }
        println!();
    }

    /// Cross-validate Solvo against Z3 on all the hand-crafted formulas.
    #[test]
    fn bench_solvo_vs_z3_handcrafted() {
        use std::time::Instant;

        let formulas: Vec<(&str, fn(&mut Solver), Option<bool>)> = vec![
            (
                "trivial_sat",
                |s| {
                    let v = s.new_var();
                    s.add_unit(v as i32);
                },
                Some(true),
            ),
            (
                "trivial_unsat",
                |s| {
                    let a = s.new_var();
                    s.add_unit(a as i32);
                    s.add_unit(-(a as i32));
                },
                Some(false),
            ),
            (
                "and_sat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    s.add_unit(a as i32);
                    s.add_unit(b as i32);
                },
                Some(true),
            ),
            (
                "or_sat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    s.add_clause(&[a as i32, b as i32]);
                    s.add_unit(-(a as i32));
                },
                Some(true),
            ),
            (
                "chain_propagation",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    let c = s.new_var();
                    let d = s.new_var();
                    s.add_implies(a as i32, b as i32);
                    s.add_implies(b as i32, c as i32);
                    s.add_implies(c as i32, d as i32);
                    s.add_unit(a as i32);
                },
                Some(true),
            ),
            (
                "backtracking_choice",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    s.add_clause(&[a as i32, b as i32]);
                    s.add_clause(&[-(a as i32), b as i32]);
                    s.add_clause(&[a as i32, -(b as i32)]);
                },
                Some(true),
            ),
            (
                "backtracking_unsat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    s.add_clause(&[a as i32, b as i32]);
                    s.add_clause(&[-(a as i32), b as i32]);
                    s.add_clause(&[a as i32, -(b as i32)]);
                    s.add_clause(&[-(a as i32), -(b as i32)]);
                },
                Some(false),
            ),
            (
                "equiv_sat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    s.add_equiv(a as i32, b as i32);
                    s.add_unit(a as i32);
                },
                Some(true),
            ),
            (
                "exactly_one_sat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    let c = s.new_var();
                    s.add_exactly_one(&[a as i32, b as i32, c as i32]);
                },
                Some(true),
            ),
            (
                "implication_cycle_unsat",
                |s| {
                    let a = s.new_var();
                    let b = s.new_var();
                    let c = s.new_var();
                    s.add_implies(a as i32, b as i32);
                    s.add_implies(b as i32, c as i32);
                    s.add_implies(c as i32, -(a as i32));
                    s.add_unit(a as i32);
                },
                Some(false),
            ),
            (
                "pigeonhole_unsat",
                |s| {
                    let p0h0 = s.new_var();
                    let p0h1 = s.new_var();
                    let p1h0 = s.new_var();
                    let p1h1 = s.new_var();
                    let p2h0 = s.new_var();
                    let p2h1 = s.new_var();
                    s.add_exactly_one(&[p0h0 as i32, p0h1 as i32]);
                    s.add_exactly_one(&[p1h0 as i32, p1h1 as i32]);
                    s.add_exactly_one(&[p2h0 as i32, p2h1 as i32]);
                    s.add_at_most_one(&[p0h0 as i32, p1h0 as i32, p2h0 as i32]);
                    s.add_at_most_one(&[p0h1 as i32, p1h1 as i32, p2h1 as i32]);
                },
                Some(false),
            ),
            (
                "multi_var_sat",
                |s| {
                    for _ in 0..10 {
                        s.new_var();
                    }
                    let vars: Vec<i32> = (1..=10)
                        .map(|i| {
                            s.add_clause(&[i as i32]);
                            i as i32
                        })
                        .collect();
                    // Redundant but tests propagation
                    s.add_clause(&[-vars[0], vars[1]]);
                    s.add_clause(&[-vars[1], vars[2]]);
                },
                Some(true),
            ),
        ];

        println!("\n── Solvo vs Z3: Hand-crafted formulas ──");
        println!(
            "{:<35} {:<10} {:<10} {:<10}",
            "Formula", "Solvo(μs)", "Z3(μs)", "Result"
        );
        println!("{}", "─".repeat(65));

        for (name, setup, expected) in &formulas {
            let mut solver = Solver::new();
            setup(&mut solver);
            let clauses = extract_clauses(&solver);
            let num_vars = solver.vars.len() - 1;

            // Solvo
            let start = Instant::now();
            let solvo_result = solver.solve();
            let solvo_time = start.elapsed().as_micros();

            // Z3
            let start = Instant::now();
            let z3_result = solve_with_z3(&clauses, num_vars);
            let z3_time = start.elapsed().as_micros();

            if let Some(expected) = expected {
                assert_eq!(
                    solvo_result.is_sat(),
                    *expected,
                    "Solvo gave wrong result for {}",
                    name,
                );
            }
            assert_eq!(
                solvo_result.is_sat(),
                z3_result.is_some(),
                "Solvo vs Z3 mismatch for {}",
                name,
            );

            println!(
                "{:<35} {:<10} {:<10} {:<10}",
                name,
                solvo_time,
                z3_time,
                if solvo_result.is_sat() {
                    "SAT ✓"
                } else {
                    "UNSAT ✓"
                },
            );
        }
        println!();
    }

    /// Larger random benchmark: stress-test Solvo against Z3 at moderate scale.
    #[test]
    fn bench_solvo_vs_z3_moderate_scale() {
        use std::time::Instant;

        let configs = [
            (30, 60, "sparse"),
            (30, 129, "phase-transition"),
            (30, 180, "dense"),
        ];

        println!("\n── Solvo vs Z3: Moderate Scale ──");
        println!(
            "{:<8} {:<8} {:<20} {:<10} {:<10} {:<10}",
            "Vars", "Clauses", "Regime", "Solvo(μs)", "Z3(μs)", "Result"
        );
        println!("{}", "─".repeat(66));

        for &(num_vars, num_clauses, label) in &configs {
            let clauses = gen_random_3sat(99 + num_vars as u64, num_vars, num_clauses);

            // Solvo
            let mut solver = Solver::new();
            for _ in 0..num_vars {
                solver.new_var();
            }
            for c in &clauses {
                solver.add_clause(c);
            }
            let start = Instant::now();
            let solvo_result = solver.solve();
            let solvo_time = start.elapsed().as_micros();

            // Z3
            let start = Instant::now();
            let z3_result = solve_with_z3(&clauses, num_vars);
            let z3_time = start.elapsed().as_micros();

            assert_eq!(
                solvo_result.is_sat(),
                z3_result.is_some(),
                "Mismatch at {} vars, {} clauses ({})",
                num_vars,
                num_clauses,
                label,
            );

            println!(
                "{:<8} {:<8} {:<20} {:<10} {:<10} {:<10}",
                num_vars,
                num_clauses,
                label,
                solvo_time,
                z3_time,
                if solvo_result.is_sat() {
                    "SAT ✓"
                } else {
                    "UNSAT ✓"
                },
            );
        }
        println!();
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
            assert!(model[0], "var 1 should be true in model");
        }
    }

    #[test]
    fn test_pure_literal_elimination() {
        // x only appears positively. Pure literal rule assigns x = true.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        solver.add_clause(&[a as i32, c as i32]);
        let result = solver.solve();
        assert!(result.is_sat(), "pure literal a should be assignable");
    }

    #[test]
    fn test_all_combinations_blocked() {
        // 3 variables, all 2^3 = 8 assignments blocked. Must be UNSAT.
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        for &(va, vb, vc) in &[
            (true, true, true),
            (true, true, false),
            (true, false, true),
            (true, false, false),
            (false, true, true),
            (false, true, false),
            (false, false, true),
            (false, false, false),
        ] {
            let mut clause = Vec::new();
            if va {
                clause.push(-(a as i32));
            } else {
                clause.push(a as i32);
            }
            if vb {
                clause.push(-(b as i32));
            } else {
                clause.push(b as i32);
            }
            if vc {
                clause.push(-(c as i32));
            } else {
                clause.push(c as i32);
            }
            solver.add_clause(&clause);
        }
        assert!(
            solver.solve().is_unsat(),
            "all assignments blocked should be UNSAT"
        );
    }

    #[test]
    fn test_binary_clause_chain() {
        // a → b, b → c, c → d, d → e, ¬e. Chain propagation: ¬a, ¬b, ¬c, ¬d.
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
            assert!(!model[0], "a should be false (¬a)");
            assert!(!model[3], "d should be false (¬d)");
        }
    }

    /// Regression: incremental reuse of a `Solver` must not leak
    /// state between `solve()` calls.  `(a∨b) ∧ (¬a∨¬b)` is satisfiable
    /// (a=true, b=false), but with the first solve's model (`a=true`,
    /// `b=true` via pure-literal elimination) left in `assignment`, the
    /// second solve treats them as fixed and reports a FALSE UNSAT.
    #[test]
    fn test_incremental_solve_no_state_leak() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause(&[a as i32, b as i32]);
        assert!(solver.solve().is_sat(), "(a ∨ b) is satisfiable");
        // Add `¬a ∨ ¬b` and re-solve on the SAME solver — still SAT.
        solver.add_clause(&[-(a as i32), -(b as i32)]);
        assert!(
            solver.solve().is_sat(),
            "(a ∨ b) ∧ (¬a ∨ ¬b) is satisfiable — a leaked model must not \
             produce a false UNSAT"
        );
    }

    #[test]
    fn test_unsat_by_unit_conflict() {
        // a and ¬a as unit clauses — immediate conflict.
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(a as i32);
        solver.add_unit(-(a as i32));
        assert!(solver.solve().is_unsat(), "a and ¬a should be UNSAT");
    }
}
