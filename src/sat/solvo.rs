//! A DPLL SAT solver with watched literals and VSIDS-lite heuristic.
//!
//! Named **Solvo** (Latin: "I solve"), this is a lightweight, self-contained
//! SAT solver designed for quick satisfiability checks within the compiler.
//! It handles small-to-medium formulas (up to ~200 variables) efficiently.
//!
//! # Features
//! - **Watched literals** (by value) for efficient unit propagation
//! - **Binary-clause fast path**: implications/equivalences propagate with a
//!   single table lookup, no clause-object indirection
//! - **Incremental propagation** via a trail head (`qhead`): each propagation
//!   pass processes only newly assigned literals (MiniSat-style rewind on
//!   backtrack)
//! - **VSIDS-lite** activity-based variable selection with lazy decay
//! - **Phase saving**: the last polarity each variable was assigned persists
//!   across backtracks *and* across incremental `solve_assumptions` calls
//! - **Clause normalisation** (dedup, tautology removal, empty-clause detection)
//! - **Sequential counter** encoding for at-most-one constraints
//! - **Incremental solving** via `solve_assumptions`; activity and phases are
//!   preserved between solves (warm starts)
//! - **Configurable resource limits** to prevent DoS on malicious input;
//!   limit exhaustion is reported as [`SolveResult::Unknown`], never as a
//!   (possibly wrong) `Unsat`
//! - **Auto-declaration**: clauses may mention undeclared variables; they are
//!   declared implicitly (MiniSat-style sizing from literals)
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

use std::time::Instant;

use super::VarOrder;

// ─── Constants ───────────────────────────────────────────────────────────────

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

/// VSIDS activity decay factor. On each conflict the increment `var_inc`
/// is divided by this factor (lazy decay): bumping with a growing increment
/// is order-equivalent to multiplying every activity by the decay, but costs
/// O(1) per conflict instead of O(n).
const ACTIVITY_DECAY: f64 = 0.95;

/// Initial VSIDS activity increment.
const ACTIVITY_BUMP: f64 = 1.0;

/// Activity rescale threshold. When an activity (or the increment) exceeds
/// this, all activities and the increment are scaled down so that floats
/// never overflow. Amortised O(1): rescales are exponentially rare.
const ACTIVITY_RESCALE: f64 = 1e100;

/// Check the wall-clock time limit only every N main-loop iterations;
/// `Instant::elapsed` is cheap but per-iteration checking adds jitter to
/// `stats.elapsed_micros` on tiny formulas.
const TIME_CHECK_INTERVAL: usize = 64;

/// At-most-one pairwise encoding threshold: below this, pairwise AMO wins
/// on propagation strength.
const AMO_PAIRWISE_THRESHOLD: usize = 8;

/// At-most-one commander-encoding threshold: beyond this, the commander
/// encoding propagates better than the sequential counter (Sinz) encoding.
const AMO_COMMANDER_THRESHOLD: usize = 50;

/// Commander-encoding group size (Klieber; group AMO is pairwise).
const AMO_COMMANDER_GROUP: usize = 10;

// ─── Public types ────────────────────────────────────────────────────────────

/// Result of a SAT solving attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveResult {
    /// Satisfiable — contains the model (1-based variable → bool).
    Sat(Vec<bool>),
    /// Unsatisfiable — no assignment can satisfy the formula.
    ///
    /// When the UNSAT answer was derived *under assumptions* and the solver
    /// supports conflict-core extraction, `core` holds the subset of the
    /// assumption literals that jointly contradict the formula — "these
    /// `@cfg` flags are jointly impossible". Empty when no core was computed
    /// (e.g. plain `solve()`, or a solver without core extraction).
    Unsat {
        /// Conflicting assumption literals (subset of the assumptions passed
        /// to `solve_assumptions`), or empty.
        core: Vec<i32>,
    },
    /// Unknown — the solver exceeded resource limits.
    Unknown,
}

impl SolveResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SolveResult::Sat(_))
    }
    pub fn is_unsat(&self) -> bool {
        matches!(self, SolveResult::Unsat { .. })
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, SolveResult::Unknown)
    }

    /// The assumption conflict core of an [`Unsat`](SolveResult::Unsat)
    /// result, or `None` if the result is not `Unsat` or no core was
    /// produced.
    pub fn unsat_core(&self) -> Option<&[i32]> {
        match self {
            SolveResult::Unsat { core } if !core.is_empty() => Some(core),
            _ => None,
        }
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
///
/// Exhausting any limit makes the solver return [`SolveResult::Unknown`] —
/// never `Unsat` — so a limit can only degrade the answer, never corrupt it.
#[derive(Debug, Clone, PartialEq, Eq)]
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
///
/// `decisions`, `propagations`, `conflicts`, `backtracks` and
/// `elapsed_micros` are **per solve** and reset at the start of every
/// `solve`/`solve_assumptions` call. `clauses_added`, `tautologies_removed`
/// and `duplicates_removed` are **cumulative** over the solver's lifetime.
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
    // ── CDCL counters (Puto only; Solvo leaves them at zero) ──
    /// Learnt clauses added by 1UIP conflict analysis.
    pub learnts_added: usize,
    /// Learnt clauses removed by database reduction.
    pub learnts_removed: usize,
    /// Learnt-database reductions performed.
    pub reductions: usize,
    /// Luby restarts performed.
    pub restarts: usize,
}

/// A variable with an optional name for debugging.
#[derive(Debug, Clone)]
pub struct Var {
    pub name: Option<String>,
}

/// A CNF clause with watched-literal bookkeeping.
///
/// Watches are stored **by value** (the watched literals themselves), not as
/// indices into `lits` — this removes a double indirection in the innermost
/// propagation loop. Only clauses with ≥ 3 literals live in `Solver::clauses`
/// and participate in watch movement; binary clauses use the dedicated
/// `bin_watches` fast path and unit clauses never move their (single) watch.
#[derive(Debug, Clone)]
struct Clause {
    /// The literals in this clause.
    lits: Vec<i32>,
    /// The two watched literals (equal for unit clauses).
    w0: i32,
    w1: i32,
}

impl Clause {
    /// General clause (≥ 3 literals): watch the first two literals.
    fn new(lits: Vec<i32>) -> Self {
        debug_assert!(lits.len() >= 3);
        Clause {
            w0: lits[0],
            w1: lits[1],
            lits,
        }
    }

    /// Binary clause for the fast path (watches are fixed, never moved).
    fn binary(lits: Vec<i32>) -> Self {
        debug_assert!(lits.len() == 2);
        Clause {
            w0: lits[0],
            w1: lits[1],
            lits,
        }
    }

    /// Unit clause (single watch, duplicated).
    fn unit(lit: i32) -> Self {
        Clause {
            lits: vec![lit],
            w0: lit,
            w1: lit,
        }
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.lits.is_empty()
    }
}

// ─── Internal outcomes ───────────────────────────────────────────────────────

/// Outcome of one unit-propagation pass.
///
/// `Limit` is deliberately distinct from `Conflict`: conflating the two made
/// the solver report spurious `Unsat` on formulas that merely exhausted the
/// propagation budget (a wrong-answer-class bug).
#[derive(Debug, Clone, Copy, PartialEq)]
enum PropOutcome {
    /// Propagation complete, no conflict.
    Ok,
    /// A clause is falsified under the current assignment.
    Conflict,
    /// The propagation limit was exhausted; the assignment is inconclusive.
    Limit,
}

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
///
/// # Panics (debug builds)
/// Panics on `i32::MIN`, which has no negation. All public entry points
/// reject `i32::MIN` before literals reach this function.
#[inline]
fn lit_neg(lit: i32) -> i32 {
    debug_assert_ne!(lit, i32::MIN, "literal i32::MIN cannot be negated");
    -lit
}

/// Map a literal to a watchlist index.
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

// ─── Clause normalisation ────────────────────────────────────────────────────

/// Normalise a clause:
/// - Remove duplicate literals
/// - Detect tautologies (both `x` and `¬x` present) → returns `None`
/// - Remove zero literals (ignored)
///
/// Returns `Some(vec![])` for an empty clause (≡ false).
///
/// Allocation-free for the common case: a single output vector with linear
/// dedup scans (clauses here are short — the `@cfg` workload is dominated by
/// binary implications). Original literal order is preserved, which keeps
/// watch choice (and therefore search behaviour) reproducible.
fn normalize_clause(clause: &[i32]) -> Option<Vec<i32>> {
    if clause.len() > 32 {
        return normalize_long_clause(clause);
    }
    let mut lits: Vec<i32> = Vec::with_capacity(clause.len());
    for &lit in clause {
        if lit == 0 {
            continue;
        }
        debug_assert_ne!(lit, i32::MIN, "literal i32::MIN cannot be negated");
        let var = lit_var(lit);
        let mut dup_or_taut = false;
        for &existing in &lits {
            if lit_var(existing) == var {
                if lit_sign(existing) == lit_sign(lit) {
                    dup_or_taut = true; // duplicate — drop
                } else {
                    return None; // tautology
                }
                break;
            }
        }
        if !dup_or_taut {
            lits.push(lit);
        }
    }
    Some(lits)
}

/// Sort-based normalisation for long clauses: dedup and tautology detection
/// in O(k log k) rather than the quadratic scan used for short clauses.
fn normalize_long_clause(clause: &[i32]) -> Option<Vec<i32>> {
    let mut sorted: Vec<i32> = clause.iter().copied().filter(|&lit| lit != 0).collect();
    for &lit in &sorted {
        debug_assert_ne!(lit, i32::MIN, "literal i32::MIN cannot be negated");
    }
    sorted.sort_unstable_by_key(|&lit| (lit_var(lit), lit_sign(lit)));
    let mut lits: Vec<i32> = Vec::with_capacity(sorted.len());
    for &lit in &sorted {
        if let Some(&last) = lits.last()
            && lit_var(last) == lit_var(lit)
        {
            if lit_sign(last) == lit_sign(lit) {
                continue; // duplicate
            }
            return None; // both polarities — tautology
        }
        lits.push(lit);
    }
    Some(lits)
}

// ─── Solver ──────────────────────────────────────────────────────────────────

/// A SAT solver using DPLL with watched literals and VSIDS-lite.
#[derive(Debug)]
pub struct Solver {
    /// All declared variables (index 0 is unused; variables are 1-based).
    vars: Vec<Var>,
    /// Clauses with ≥ 3 literals (participate in watch movement).
    clauses: Vec<Clause>,
    /// Binary clauses (len == 2). Watches are static; propagation goes
    /// through `bin_watches` directly and never touches these objects.
    binary_clauses: Vec<Clause>,
    /// Current partial assignment: var_index → Some(true/false) or None.
    assignment: Vec<Option<bool>>,
    /// Decision level for each variable (for backtracking).
    decision_level: Vec<usize>,
    /// Saved phase per variable: the last polarity it was assigned.
    /// Persists across backtracks *and* incremental solves (phase saving).
    phase: Vec<bool>,
    /// Reason clause for each propagated variable (reserved for future CDCL).
    /// General clauses are their id; binary clauses use `None` (they are
    /// identified structurally).
    #[allow(dead_code)]
    reason: Vec<Option<u32>>,
    /// Current decision level.
    current_level: usize,
    /// Trail: history of assigned literals in order.
    trail: Vec<i32>,
    /// Trail limits: decision level → trail length at that level.
    trail_limits: Vec<usize>,
    /// Propagation queue head: index into `trail` of the next literal to
    /// propagate. Incremental — only newly assigned literals are processed.
    qhead: usize,
    /// Number of currently assigned variables. SAT exit condition:
    /// `assigned_count == num_vars()` with no conflict.
    assigned_count: usize,
    /// Watch lists for general clauses: watchlist[lit_index] → clause ids
    /// watching that literal (i.e. which fire when the literal is falsified).
    watchlist: Vec<Vec<u32>>,
    /// Binary-clause fast path: bin_watches[lit_index(l)] → implied partner
    /// literals. When `l` is falsified, every partner must be true.
    bin_watches: Vec<Vec<i32>>,
    /// VSIDS-lite activity score per variable (1-based).
    activity: Vec<f64>,
    /// Max-heap over currently-unassigned variables, keyed by activity.
    order: VarOrder,
    /// Current VSIDS increment (lazy decay; grows by 1/decay per conflict).
    var_inc: f64,
    /// Unit clauses, asserted eagerly at level 0 at the start of every solve.
    /// Enforcing a unit only reactively (through its watch) costs a decision
    /// plus a conflict per unit, and leaves chained implications from ternary
    /// clauses underived (E5/A4).
    units: Vec<i32>,

    /// Resource limits.
    limits: Limits,
    /// Append-only log of the *normalised input clauses* (including units and
    /// tautology-free duplicates). This is the authoritative formula for
    /// `clauses_snapshot`/`to_dimacs` (): unlike `clauses` it is never
    /// reached by inprocessing (no subsumption/tombstoning), so the audit
    /// artifact always describes what the caller actually added.
    original_clauses: Vec<Vec<i32>>,
    /// Statistics (per-solve + cumulative; see [`SolverStats`]).
    pub stats: SolverStats,
    /// Whether an empty clause has been added (immediately UNSAT).
    has_empty_clause: bool,
    /// Start time of the current solve() call (for time limit).
    solve_start: Option<Instant>,
    /// Main-loop iteration counter (throttles wall-clock limit checks).
    loop_tick: usize,
}

impl Solver {
    /// Create a new, empty solver.
    pub fn new() -> Self {
        Solver {
            vars: vec![Var { name: None }], // index 0 placeholder
            clauses: Vec::new(),
            binary_clauses: Vec::new(),
            assignment: vec![None],
            decision_level: vec![0],
            phase: vec![false],
            reason: vec![None],
            current_level: 0,
            trail: Vec::new(),
            trail_limits: Vec::new(),
            qhead: 0,
            assigned_count: 0,
            watchlist: Vec::new(),
            bin_watches: Vec::new(),
            activity: vec![0.0],
            order: VarOrder::new(),
            var_inc: ACTIVITY_BUMP,
            units: Vec::new(),
            limits: Limits::default(),
            original_clauses: Vec::new(),
            stats: SolverStats::default(),
            has_empty_clause: false,
            solve_start: None,
            loop_tick: 0,
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

    /// Grow the variable table by one and return its 1-based index.
    fn grow_var(&mut self, name: Option<String>) -> usize {
        let idx = self.vars.len();
        self.vars.push(Var { name });
        self.assignment.push(None);
        self.decision_level.push(0);
        self.phase.push(false);
        self.reason.push(None);
        self.activity.push(0.0);
        self.order.push_new(idx, &self.activity);
        // Two watchlist slots (positive + negative literal) for each structure.
        self.watchlist.push(Vec::new());
        self.watchlist.push(Vec::new());
        self.bin_watches.push(Vec::new());
        self.bin_watches.push(Vec::new());
        idx
    }

    /// Declare a new boolean variable and return its 1-based index.
    pub fn new_var(&mut self) -> usize {
        self.grow_var(None)
    }

    /// Declare a new variable with a debug name.
    pub fn new_named_var(&mut self, name: &str) -> usize {
        self.grow_var(Some(name.to_string()))
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

    /// Number of clauses in the formula (general + binary).
    pub fn num_clauses(&self) -> usize {
        self.clauses.len() + self.binary_clauses.len()
    }

    // ── Clause management ────────────────────────────────────────────────────

    /// Add a CNF clause (disjunction of signed literals).
    ///
    /// The clause is normalised before insertion:
    /// - Duplicate literals are removed.
    /// - Tautological clauses (containing both `x` and `¬x`) are discarded.
    /// - An empty clause (after normalisation) marks the formula as trivially UNSAT.
    ///
    /// Variables mentioned in the clause are **auto-declared** if they have
    /// not been declared yet (MiniSat-style sizing from literals), so it is
    /// impossible to introduce a phantom literal that indexes out of bounds.
    pub fn add_clause(&mut self, clause: &[i32]) {
        // A9: once the empty clause has been seen the formula is permanently
        // UNSAT. Dropping later adds matches Puto's `ok` gating, so an
        // operation stream produces the same snapshots on both backends.
        if self.has_empty_clause {
            return;
        }
        // An all-zero input (a DIMACS-style trailing-zeros line) is invalid
        // input, not the empty clause — match `add_unit`'s no-op policy. A
        // genuinely empty slice *is* the empty clause.
        if !clause.is_empty() && clause.iter().all(|&l| l == 0) {
            return;
        }
        // `i32::MIN` cannot be negated; reject it at the boundary rather than
        // letting the debug assertion vanish (and `ensure_var` attempt a
        // 2^31-slot allocation) in release builds.
        if clause.iter().any(|&l| l == i32::MIN) {
            return;
        }
        let normalized = match normalize_clause(clause) {
            Some(lits) => lits,
            None => {
                // Tautology — always true, skip.
                self.stats.tautologies_removed += 1;
                return;
            }
        };

        // Count duplicates removed (non-zero input literals − survivors).
        let input_nonzero = clause.iter().filter(|&&l| l != 0).count();
        self.stats.duplicates_removed += input_nonzero - normalized.len();

        if normalized.is_empty() {
            // Empty clause ≡ false — formula is UNSAT.
            self.has_empty_clause = true;
            return;
        }

        // : log the normalised input clause before any inprocessing can
        // rewrite or drop it. This is what `clauses_snapshot` returns.
        self.original_clauses.push(normalized.clone());

        // Auto-declare: every literal's variable must exist before we index
        // watchlists. (Also the  fix — no phantom literals can exist.)
        let max_var = normalized.iter().map(|&l| lit_var(l)).max().unwrap();
        self.ensure_var(max_var);

        if normalized.len() == 1 {
            // Unit clause: single (never-moved) watch. Also recorded in
            // `units` so the solve can assert it eagerly at level 0 — a unit
            // enforced only reactively costs a decision plus a conflict.
            let cid = self.clauses.len() as u32;
            let li = lit_index(normalized[0]);
            self.watchlist[li].push(cid);
            self.units.push(normalized[0]);
            self.clauses.push(Clause::unit(normalized[0]));
            self.stats.clauses_added += 1;
            return;
        }

        if normalized.len() == 2 {
            // Binary fast path: register the implied partner under each
            // literal's index. No clause object is touched during propagation.
            let (l0, l1) = (normalized[0], normalized[1]);
            self.bin_watches[lit_index(l0)].push(l1);
            self.bin_watches[lit_index(l1)].push(l0);
            self.binary_clauses.push(Clause::binary(normalized));
            self.stats.clauses_added += 1;
            return;
        }

        // General clause (≥ 3 literals): watch the first two literals.
        let cid = self.clauses.len() as u32;
        let w0 = lit_index(normalized[0]);
        self.watchlist[w0].push(cid);
        let w1 = lit_index(normalized[1]);
        self.watchlist[w1].push(cid);
        self.clauses.push(Clause::new(normalized));
        self.stats.clauses_added += 1;
    }

    /// Add a unit clause (single literal).
    ///
    /// Literal `0` (and `i32::MIN`, which cannot be negated) is invalid input
    /// and is ignored as a no-op — it is *not* interpreted as the empty
    /// clause (which would silently make the formula UNSAT).
    pub fn add_unit(&mut self, lit: i32) {
        if lit == 0 || lit == i32::MIN {
            return;
        }
        self.add_clause(&[lit]);
    }

    /// Add an implication: `a → b` ≡ `¬a ∨ b`.
    ///
    /// Invalid literals (`0`, `i32::MIN`) are a no-op: negating `i32::MIN`
    /// would overflow (`-a`) before `add_clause` could reject it.
    pub fn add_implies(&mut self, a: i32, b: i32) {
        if a == 0 || b == 0 || a == i32::MIN || b == i32::MIN {
            return;
        }
        self.add_clause(&[-a, b]);
    }

    /// Add an equivalence: `a ↔ b` ≡ `(¬a ∨ b) ∧ (a ∨ ¬b)`.
    pub fn add_equiv(&mut self, a: i32, b: i32) {
        if a == 0 || b == 0 || a == i32::MIN || b == i32::MIN {
            return;
        }
        self.add_clause(&[-a, b]);
        self.add_clause(&[a, -b]);
    }

    /// Add an at-most-one constraint over `lits`.
    ///
    /// Encoding by group size (shared audited implementations in
    /// [`super`]): pairwise ≤ 8, sequential counter (Sinz) up to 49, and
    /// **commander** beyond — the commander encoding propagates better at
    /// large n even though it uses more clauses.
    pub fn add_at_most_one(&mut self, lits: &[i32]) {
        // : validate/dedup first. Doing this before any `ensure_var` is what
        // keeps `i32::MIN` (whose `lit_var` is 2^31) from triggering a
        // release-build OOM; the encoders auto-declare via `add_clause` /
        // `new_var`, so no up-front `ensure_var` is needed at all.
        let Some(lits) = super::cardinality_lits(lits) else {
            return;
        };
        let lits = lits.as_slice();
        if lits.len() <= 1 {
            return;
        }

        if lits.len() <= AMO_PAIRWISE_THRESHOLD {
            super::encode_at_most_one_pairwise(self, lits);
        } else if lits.len() < AMO_COMMANDER_THRESHOLD {
            super::encode_at_most_one_sequential(self, lits);
        } else {
            super::encode_at_most_one_commander(self, lits, AMO_COMMANDER_GROUP);
        }
    }

    /// Add an exactly-one constraint: at-least-one + at-most-one.
    ///
    /// An empty literal set is invalid input and is ignored as a no-op — it
    /// is *not* added as an empty clause (which would silently make the
    /// formula UNSAT).
    pub fn add_exactly_one(&mut self, lits: &[i32]) {
        if lits.is_empty() {
            return;
        }
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
    /// - [`SolveResult::Unknown`] — the solver exceeded a resource limit.
    pub fn solve(&mut self) -> SolveResult {
        self.solve_assumptions(&[])
    }

    /// Solve under temporary assumptions.
    ///
    /// Assumptions are unit literals that are asserted at decision level 1
    /// but do not permanently modify the formula. After solving, all
    /// assumption-related state is cleaned up.
    ///
    /// Undeclared assumption variables are auto-declared (matching
    /// `add_clause`'s policy). Resource-limit exhaustion yields
    /// [`SolveResult::Unknown`] — a limit can degrade the answer to
    /// "unknown" but never fabricate an `Unsat`.
    pub fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult {
        // Quick check: empty clause means immediate UNSAT.
        if self.has_empty_clause {
            return SolveResult::Unsat { core: Vec::new() };
        }

        let start = Instant::now();
        self.solve_start = Some(start);
        // Reset only per-solve stats; cumulative counters (clauses_added,
        // tautologies_removed, duplicates_removed) survive across solves.
        self.stats.decisions = 0;
        self.stats.propagations = 0;
        self.stats.conflicts = 0;
        self.stats.backtracks = 0;

        // Reset solving state (but keep clauses, variables, activity and
        // saved phases — warm start for incremental solves).
        self.reset_state();

        // Assert unit clauses eagerly at level 0 (below the assumption level), so
        // propagation starts from them instead of deciding, falsifying and
        // conflicting. They persist for the whole solve.
        for i in 0..self.units.len() {
            let lit = self.units[i];
            let var = lit_var(lit);
            let val = lit_sign(lit);
            match self.assignment[var] {
                Some(existing) => {
                    if existing != val {
                        self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                        self.solve_start = None;
                        return SolveResult::Unsat { core: Vec::new() };
                    }
                }
                None => {
                    self.assign(var, val, 0);
                    self.trail.push(lit);
                }
            }
        }

        // Push assumption level. The level-1 boundary is the *post-unit* trail
        // length (A3): level 0 owns `trail[0..U]`. Pushing 0 would claim level 1
        // starts at index 0, so any future `backtrack(0)` would pop the eager
        // units — and units are only asserted here, at solve start, so they
        // would be silently lost for the rest of the solve (wrong answers).
        self.current_level = 1;
        self.trail_limits.push(self.trail.len());

        // Assert assumptions as unit propagations at level 1.
        for &lit in assumptions {
            debug_assert_ne!(lit, 0, "assumption literal 0 is invalid");
            debug_assert_ne!(lit, i32::MIN, "assumption literal i32::MIN is invalid");
            if lit == 0 || lit == i32::MIN {
                return SolveResult::Unknown;
            }
            self.ensure_var(lit_var(lit));
            let var = lit_var(lit);
            let val = lit_sign(lit);
            if let Some(existing) = self.assignment[var] {
                if existing != val {
                    // Contradictory assumptions. The contradicting pair is
                    // known — return it, mirroring Puto's core shape.
                    return SolveResult::Unsat {
                        core: vec![lit, -lit],
                    };
                }
                // Already assigned consistently — no-op.
            } else {
                self.assign(var, val, 1);
                self.trail.push(lit);
            }
        }

        // Propagate assumptions.
        match self.propagate() {
            PropOutcome::Conflict => return SolveResult::Unsat { core: Vec::new() },
            // Budget exhausted mid-propagation: inconclusive, not UNSAT.
            PropOutcome::Limit => {
                self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                self.solve_start = None;
                return SolveResult::Unknown;
            }
            PropOutcome::Ok => {}
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
                // Free full-bug-class check: in debug builds, a returned
                // model must satisfy every clause (including binaries and
                // clauses added after a previous solve).
                debug_assert!(
                    self.verify_model(),
                    "internal error: returned model violates the clause database"
                );
                SolveResult::Sat(model)
            }
            DpllOutcome::Unsat => SolveResult::Unsat { core: Vec::new() },
            DpllOutcome::LimitExceeded => SolveResult::Unknown,
        }
    }

    /// Reset all solving state (assignments, trail, levels) while keeping
    /// the clause database, variable declarations, **VSIDS activity and
    /// saved phases** intact.
    ///
    /// Watch state needs no rebuild: the watched-literal invariant is
    /// assignment-independent (watches are literals, not positions), so
    /// well-placed watches survive across incremental solves.
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
        self.current_level = 0;
        self.trail.clear();
        self.trail_limits.clear();
        self.qhead = 0;
        self.assigned_count = 0;
        // Every variable is unassigned again: refill the branch order.
        self.order.reset(self.num_vars(), &self.activity);
    }

    // ── DPLL main loop ───────────────────────────────────────────────────────

    /// DPLL main loop — iterative implementation using an explicit stack.
    /// Avoids recursion depth issues on large formulas.
    ///
    /// Stack entries: `(decision_level_before_branch, variable, polarity,
    /// is_first_attempt)`. `polarity` is the branch currently assigned for
    /// `var`; `is_first_attempt` distinguishes the initial branch (the saved
    /// phase) from its retry with the opposite polarity — the two are
    /// independent because the saved phase may be `false`, so polarity alone
    /// cannot tell "retry pending" from "exhausted".
    fn dpll(&mut self) -> DpllOutcome {
        let mut stack: Vec<(usize, usize, bool, bool)> = Vec::new();

        loop {
            self.loop_tick = self.loop_tick.wrapping_add(1);
            // Check counting limits every iteration (cheap)…
            if self.stats.decisions >= self.limits.max_decisions
                || self.stats.conflicts >= self.limits.max_conflicts
            {
                return DpllOutcome::LimitExceeded;
            }
            // …and the wall-clock limit only every TIME_CHECK_INTERVAL
            // iterations, reducing measurement jitter on tiny formulas.
            if self.loop_tick.is_multiple_of(TIME_CHECK_INTERVAL)
                && let Some(start) = self.solve_start
                && start.elapsed().as_micros() as u64 > self.limits.max_time_micros
            {
                return DpllOutcome::LimitExceeded;
            }

            // Unit propagation.
            match self.propagate() {
                PropOutcome::Limit => {
                    // Budget exhausted: inconclusive. Return without touching
                    // the stack — no backtrack accounting, no activity decay.
                    return DpllOutcome::LimitExceeded;
                }
                PropOutcome::Conflict => {
                    self.stats.conflicts += 1;

                    // Activity bump (A5): Solvo previously bumped only on
                    // the binary fast path, so a formula without binaries
                    // left every activity at 0.0 and `choose_var` degenerated
                    // to fixed variable-index order. Bump the variables
                    // assigned at the current decision level (the conflict
                    // cone's spine) — MiniSat's pre-CDCL heuristic, and the
                    // cheapest correction available without clause ids in
                    // `PropOutcome`.
                    let level_start = self.trail_limits.last().copied().unwrap_or(0);
                    for i in level_start..self.trail.len() {
                        let var = lit_var(self.trail[i]);
                        self.bump_activity(var);
                    }

                    // Lazy VSIDS decay: grow the increment instead of
                    // decaying every activity (O(1) per conflict).
                    self.var_inc /= ACTIVITY_DECAY;
                    if self.var_inc > ACTIVITY_RESCALE {
                        for a in self.activity.iter_mut() {
                            *a *= 1e-100;
                        }
                        self.var_inc *= 1e-100;
                    }

                    // Backtrack loop: keep popping frames until we find one
                    // whose second branch is untried, or exhaust the stack.
                    loop {
                        if let Some((level, var, val_tried, is_first)) = stack.pop() {
                            self.stats.backtracks += 1;
                            self.backtrack(level);

                            if is_first {
                                // First branch (the saved phase) failed —
                                // retry with the opposite polarity.
                                self.current_level = level + 1;
                                self.trail_limits.push(self.trail.len());
                                let other = !val_tried;
                                self.assign(var, other, self.current_level);
                                self.trail
                                    .push(if other { var as i32 } else { -(var as i32) });
                                stack.push((level, var, other, false));
                                break; // resume main loop
                            }

                            // Both branches failed — continue backtracking
                            // to the parent frame.
                            if stack.is_empty() {
                                return DpllOutcome::Unsat;
                            }
                            continue;
                        } else {
                            // Stack is empty — no more choices. UNSAT.
                            return DpllOutcome::Unsat;
                        }
                    }
                    continue;
                }
                PropOutcome::Ok => {}
            }

            // SAT exit: every variable assigned and no conflict means every
            // clause (none empty, none all-false) contains a true literal.
            if self.assigned_count == self.num_vars() {
                return DpllOutcome::Sat;
            }

            // Choose a variable to branch on (VSIDS-lite).
            let var = self.choose_var();
            if var == 0 {
                // No unassigned variables but the count disagreed — treat as
                // SAT (defensive; unreachable when the counters agree).
                return DpllOutcome::Sat;
            }

            // Branch on the saved phase first (phase saving).
            self.stats.decisions += 1;
            let level = self.current_level;
            self.current_level += 1;
            self.trail_limits.push(self.trail.len());
            let val = self.phase[var];
            self.assign(var, val, self.current_level);
            self.trail
                .push(if val { var as i32 } else { -(var as i32) });
            stack.push((level, var, val, true));
        }
    }

    // ── Watched-literal propagation ──────────────────────────────────────────

    /// Unit propagation (incremental, MiniSat-style).
    ///
    /// Processes only literals not yet propagated (from `qhead` to the trail
    /// end). When a literal `l` is assigned true, `¬l` is false, so clauses
    /// watching `¬l` need attention. Binary clauses are handled first through
    /// a direct implied-literal table (no clause object, no watch search);
    /// general clauses (≥ 3 literals) go through watch movement.
    ///
    /// On backtrack, `qhead` is rewound (`backtrack`), which re-processes
    /// everything assigned after the rewind point on the next pass — the
    /// same guarantee the old full-trail re-seed provided, at O(new work)
    /// instead of O(trail).
    fn propagate(&mut self) -> PropOutcome {
        while self.qhead < self.trail.len() {
            // Propagation limit: a *limit*, reported as such — never
            // conflated with a conflict (which would fabricate `Unsat`).
            if self.stats.propagations >= self.limits.max_propagations {
                return PropOutcome::Limit;
            }
            let lit = self.trail[self.qhead];
            self.qhead += 1;
            self.stats.propagations += 1;

            let neg = lit_neg(lit);
            let neg_idx = lit_index(neg);
            debug_assert!(
                neg_idx < self.bin_watches.len() && neg_idx < self.watchlist.len(),
                "watchlist not sized for literal {lit}"
            );

            // ── Binary-clause fast path ─────────────────────────────────
            // Every clause registered under ¬lit's index asserts its partner
            // literal. Watches are static; no entry is ever moved or removed.
            let mut bi = 0;
            while bi < self.bin_watches[neg_idx].len() {
                let implied = self.bin_watches[neg_idx][bi];
                bi += 1;
                match self.lit_value(implied) {
                    Some(true) => {} // clause already satisfied
                    Some(false) => return PropOutcome::Conflict,
                    None => {
                        let var = lit_var(implied);
                        debug_assert!(var < self.assignment.len());
                        self.assign(var, lit_sign(implied), self.current_level);
                        self.reason[var] = None; // binary clauses are structural
                        self.trail.push(implied);
                        self.bump_activity(var);
                    }
                }
            }

            // ── General clauses (≥ 3 literals) ───────────────────────────
            let mut wi = 0;
            while wi < self.watchlist[neg_idx].len() {
                let cid = self.watchlist[neg_idx][wi];
                debug_assert!((cid as usize) < self.clauses.len());
                let (w0, w1) = {
                    let c = &self.clauses[cid as usize];
                    (c.w0, c.w1)
                };
                let is_w0 = w0 == neg;
                let is_w1 = w1 == neg;
                if !is_w0 && !is_w1 {
                    // With by-value watches the watchlist mirrors the clause
                    // exactly; a non-matching entry is an invariant break.
                    debug_assert!(false, "stale watchlist entry for clause {cid}");
                    wi += 1;
                    continue;
                }
                let other = if is_w0 { w1 } else { w0 };

                // Replacement search runs in BOTH non-satisfied arms below:
                // whenever a watched literal is falsified the clause must
                // re-secure a non-false watch if one exists — asserting
                // `other` as a unit without this check would assert a "unit"
                // that is not one when a third literal is true or unassigned.
                if self.lit_value(other) != Some(true) {
                    let replacement = {
                        let clause = &self.clauses[cid as usize];
                        let mut found = None;
                        for &cand in &clause.lits {
                            if cand == w0 || cand == w1 {
                                continue;
                            }
                            if self.lit_value(cand) != Some(false) {
                                found = Some(cand);
                                break;
                            }
                        }
                        found
                    };
                    if let Some(new_lit) = replacement {
                        self.watchlist[neg_idx].swap_remove(wi);
                        {
                            let c = &mut self.clauses[cid as usize];
                            if is_w0 {
                                c.w0 = new_lit;
                            } else {
                                c.w1 = new_lit;
                            }
                        }
                        let new_idx = lit_index(new_lit);
                        self.watchlist[new_idx].push(cid);
                        // wi deliberately not incremented (swap_remove).
                        continue;
                    }
                }

                match self.lit_value(other) {
                    Some(true) => {
                        // Satisfied by the other watch — nothing to do.
                        wi += 1;
                    }
                    Some(false) => {
                        // Both watches false and no replacement exists:
                        // every literal is false — genuine conflict.
                        return PropOutcome::Conflict;
                    }
                    None => {
                        // No replacement watch and the other watch is
                        // unassigned: the clause is unit — assert it.
                        //
                        // The previous implementation skipped this arm
                        // entirely ("may be satisfied later"), which silently
                        // disabled unit propagation for every clause of
                        // length ≥ 3 — the core DPLL inference. Chained
                        // ternary implications then degenerated into
                        // exponential decide-falsify-conflict cycles.
                        let var = lit_var(other);
                        debug_assert!(self.assignment[var].is_none());
                        self.assign(var, lit_sign(other), self.current_level);
                        self.reason[var] = None;
                        self.trail.push(other);
                        self.bump_activity(var);
                        wi += 1;
                    }
                }
            }
        }
        PropOutcome::Ok
    }

    /// Bump a variable's VSIDS activity with the current (lazily decaying)
    /// increment, rescaling the whole table if values approach overflow.
    fn bump_activity(&mut self, var: usize) {
        self.activity[var] += self.var_inc;
        if self.activity[var] > ACTIVITY_RESCALE {
            for a in self.activity.iter_mut() {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
        // Uniform rescale preserves order; otherwise the var only moves up.
        self.order.bumped(var, &self.activity);
    }

    // ── Variable selection (VSIDS-lite) ──────────────────────────────────────

    /// Choose the next unassigned variable to branch on.
    /// Uses VSIDS-lite: pick the unassigned variable with the highest
    /// activity score. Ties are broken by variable index (lower first).
    ///
    /// Returns 0 if all variables are assigned.
    fn choose_var(&mut self) -> usize {
        // Assigned entries that surface are discarded (they re-enter the heap
        // on backtrack); ties break to the lowest index via the heap ordering.
        while let Some(var) = self.order.pop_max(&self.activity) {
            if self.assignment[var].is_none() {
                return var;
            }
        }
        0
    }

    // ── Assignment management ────────────────────────────────────────────────

    /// Assign a variable at the given decision level and record its phase.
    fn assign(&mut self, var: usize, val: bool, level: usize) {
        debug_assert!(var < self.assignment.len());
        debug_assert!(
            self.assignment[var].is_none(),
            "double assignment of var {var}"
        );
        self.assignment[var] = Some(val);
        self.decision_level[var] = level;
        self.phase[var] = val; // phase saving
        self.assigned_count += 1;
    }

    /// Evaluate a literal under the current assignment.
    /// Returns `Some(true)` if the literal is true, `Some(false)` if false,
    /// `None` if the variable is unassigned.
    fn lit_value(&self, lit: i32) -> Option<bool> {
        let var = lit_var(lit);
        if var >= self.assignment.len() {
            // Unreachable with auto-declaration at every entry point; kept
            // as a guard because this runs in the innermost loop.
            return None;
        }
        self.assignment[var].map(|v| if lit > 0 { v } else { !v })
    }

    /// Backtrack to the given decision level, undoing all assignments
    /// made after that level.
    ///
    /// No clause-flag restoration is needed (there are no flags): the
    /// watched-literal state is assignment-independent. The only extra work
    /// is rewinding `qhead` so the next propagation pass re-processes the
    /// rewound region's successors (MiniSat's backjump mechanism).
    fn backtrack(&mut self, level: usize) {
        // Find the trail position to rewind to.
        let target = if level < self.trail_limits.len() {
            self.trail_limits[level]
        } else {
            0
        };

        // Undo assignments (phases are deliberately kept — phase saving).
        while self.trail.len() > target {
            let lit = self.trail.pop().unwrap();
            let var = lit_var(lit);
            debug_assert!(self.assignment[var].is_some());
            self.assignment[var] = None;
            self.decision_level[var] = 0;
            self.reason[var] = None;
            self.assigned_count -= 1;
            // Unassigned again — eligible for branching.
            self.order.insert(var, &self.activity);
        }

        // Re-process everything above the rewind point on the next pass.
        self.qhead = self.qhead.min(target);

        // Trim trail limits.
        self.trail_limits.truncate(level);
        self.current_level = level;
    }

    /// Verify that the current assignment satisfies every clause in the
    /// database — general, binary, and clauses added after a previous solve.
    /// (Only invoked under `debug_assert!`, so release builds never pay the
    /// cost.)
    fn verify_model(&self) -> bool {
        let general_ok = self
            .clauses
            .iter()
            .all(|c| c.lits.iter().any(|&lit| self.lit_value(lit) == Some(true)));
        let binary_ok = self
            .binary_clauses
            .iter()
            .all(|c| c.lits.iter().any(|&lit| self.lit_value(lit) == Some(true)));
        general_ok && binary_ok
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
    fn clauses_snapshot(&self) -> Vec<Vec<i32>> {
        // : read the append-only input log, not the internal database —
        // inprocessing (subsumption, tombstoning, strengthening) never touches
        // the log, so the snapshot always describes the caller's formula.
        // Canonical order (contract shared with `puto::Solver`): see the
        // corresponding note there.
        let mut out: Vec<Vec<i32>> = self.original_clauses.clone();
        for c in &mut out {
            c.sort_unstable();
        }
        out.sort_unstable();
        out
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
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
        assert!(solver.stats.elapsed_micros > 0 || solver.stats.decisions > 0);
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
            self.state >> 33
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
                let sign = if rng.next().is_multiple_of(2) { 1 } else { -1 };
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

        fn propagate(clauses: &[Vec<i32>], assignment: &mut [Option<bool>]) -> bool {
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

        fn dpll_recursive(clauses: &[Vec<i32>], assignment: &mut [Option<bool>]) -> bool {
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

            let saved = assignment.to_vec();

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

    /// Extract the full clause database (general + binary) for oracle testing.
    fn extract_clauses(solver: &Solver) -> Vec<Vec<i32>> {
        solver
            .clauses
            .iter()
            .chain(solver.binary_clauses.iter())
            .map(|c| c.lits.clone())
            .collect()
    }

    #[test]
    fn test_iterative_vs_recursive_consistency() {
        type FormulaCase = (&'static str, fn(&mut Solver));
        let formulas: Vec<FormulaCase> = vec![
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

    // ═════════════════════════════════════════════════════════════════════════
    // ── Regression tests for the 2025-11 audit (A/B/C findings) ─────────────
    // ═════════════════════════════════════════════════════════════════════════

    // ── : limit exhaustion must yield Unknown, never a fabricated Unsat ────

    #[test]
    fn test_propagation_limit_yields_unknown_not_unsat() {
        // A long implication chain a→b→c→… needs ~n propagations to resolve;
        // a tight propagation budget must report Unknown, not Unsat.
        let mut solver = Solver::with_limits(Limits {
            max_propagations: 5,
            ..Default::default()
        });
        let n = 50;
        let lits: Vec<i32> = (0..n).map(|_| solver.new_var() as i32).collect();
        for w in lits.windows(2) {
            solver.add_implies(w[0], w[1]);
        }
        solver.add_unit(lits[0]);
        let result = solver.solve_assumptions(&[]);
        assert!(
            result.is_unknown(),
            "propagation-limit exhaustion must be Unknown, got {result:?}"
        );
    }

    #[test]
    fn test_propagation_limit_inside_dpll_yields_unknown() {
        // Same guarantee when the limit fires mid-search (the old code could
        // drain a stack of exhausted frames into a wrong `Unsat`).
        let mut solver = Solver::with_limits(Limits {
            max_propagations: 3,
            ..Default::default()
        });
        let n = 40;
        let lits: Vec<i32> = (0..n).map(|_| solver.new_var() as i32).collect();
        // Satisfiable formula requiring a decision before heavy propagation.
        solver.add_clause(&[lits[0], lits[1]]);
        for w in lits[1..].windows(2) {
            solver.add_implies(w[0], w[1]);
        }
        solver.add_unit(lits[1]);
        let result = solver.solve();
        assert!(
            result.is_unknown() || result.is_sat(),
            "limit exhaustion must never fabricate Unsat, got {result:?}"
        );
    }

    // ── : undeclared literals must auto-declare, never panic ───────────────

    #[test]
    fn test_undeclared_literal_auto_declares() {
        // Off-by-one in var numbering must not crash the solver: the clause's
        // variable is auto-declared, then the clause participates normally.
        // (Kept to 5k vars: choose_var is an O(n) scan per decision, and a
        // 50k-var decision sweep exceeds the default 5s time limit in debug
        // builds — the limit then correctly reports Unknown.)
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_clause(&[5_000]); // undeclared literal
        solver.add_clause(&[-(a as i32), 5_000]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert_eq!(model.len(), 5_000, "auto-declared var must be in the model");
            assert!(model[4_999], "unit clause (v5000) must be true");
        }
    }

    #[test]
    fn test_undeclared_literal_binary_falsified_watch() {
        // The old OOB panic path: a phantom literal in a binary clause whose
        // declared watch gets falsified → the implied literal propagates.
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_clause(&[-(a as i32), 1_000]); // a → v1000
        solver.add_unit(a as i32);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(model[999], "v1000 implied by a must be true");
        }
    }

    // ── A3: i32::MIN rejected at the boundary (checked in normalize debug) ───

    #[test]
    fn test_zero_and_min_literals_are_rejected_not_bottom() {
        // Literal 0 passed to add_unit is invalid input — it must NOT be
        // interpreted as the empty clause (which would silently UNSAT the
        // formula).
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_unit(0);
        assert!(
            !solver.has_empty_clause,
            "add_unit(0) must not create an empty clause"
        );
        solver.add_unit(a as i32);
        assert!(solver.solve().is_sat());

        // add_exactly_one(&[]) is invalid input too — no silent UNSAT.
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_exactly_one(&[]);
        assert!(
            !solver.has_empty_clause,
            "add_exactly_one(&[]) must not create an empty clause"
        );
        solver.add_unit(a as i32);
        assert!(solver.solve().is_sat());
    }

    // ── A4: stats hygiene ────────────────────────────────────────────────────

    #[test]
    fn test_cumulative_stats_survive_solve() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        solver.add_clause(&[a as i32, a as i32, a as i32]); // 2 duplicates
        solver.add_clause(&[a as i32, -(a as i32)]); // tautology
        assert_eq!(solver.stats.clauses_added, 1);
        assert_eq!(solver.stats.duplicates_removed, 2);
        assert_eq!(solver.stats.tautologies_removed, 1);

        let before = (solver.stats.clauses_added, solver.stats.duplicates_removed);
        solver.solve();
        // Per-solve counters reset, cumulative counters survive the solve.
        assert_eq!(solver.stats.clauses_added, before.0);
        assert_eq!(solver.stats.duplicates_removed, before.1);
        solver.add_clause(&[a as i32, a as i32]);
        assert_eq!(solver.stats.clauses_added, before.0 + 1);
        assert_eq!(solver.stats.duplicates_removed, before.1 + 1);
    }

    // ── B2: activity and phases persist across incremental solves ────────────

    #[test]
    fn test_vsids_activity_persists_across_solves() {
        // Build a formula with real conflicts so activities become nonzero.
        let mut solver = Solver::new();
        let x = solver.new_var() as i32;
        let y = solver.new_var() as i32;
        solver.add_clause(&[x, y]);
        solver.add_clause(&[-x, y]);
        solver.add_clause(&[x, -y]);
        solver.add_clause(&[-x, -y]); // 2-var pigeonhole → conflicts
        assert!(solver.solve().is_unsat());

        let activity_before: Vec<f64> = solver.activity.clone();
        assert!(
            activity_before.iter().any(|&a| a > 0.0),
            "conflicts must bump activity"
        );

        // Adding nothing (tautology is dropped) and re-solving must not
        // reset activity — warm-start incremental solves. With lazy decay,
        // per-variable activity never decreases (no per-conflict decay, and
        // no rescale at these magnitudes), so the second solve can only add
        // to the accumulated scores. If reset_state zeroed activity, the
        // previously-bumped variables would come back strictly smaller.
        solver.add_clause(&[x, -x]); // tautology, removed
        let _ = solver.solve();
        for (i, (&after, &before)) in solver
            .activity
            .iter()
            .zip(activity_before.iter())
            .enumerate()
        {
            assert!(
                after >= before,
                "activity of var {i} decreased across solves: {before} → {after} \
                 (reset_state must not zero VSIDS activity)"
            );
        }
        assert!(
            solver.activity.iter().any(|&a| a > 0.0),
            "accumulated activity must survive into the next solve"
        );
    }

    #[test]
    fn test_phase_persists_across_solves() {
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        // Force b true via a's unit.
        solver.add_implies(a, b);
        solver.add_unit(a);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = &result {
            assert!(model[1]);
        }
        assert!(solver.phase[2], "b's saved phase should be true");
        // A second solve (nothing changed) must keep the saved phase.
        let _ = solver.solve();
        assert!(solver.phase[2], "phase must persist across solves");
    }

    // ── E5: qhead rewind — conflict deep in a subtree, SAT in sibling ────────

    #[test]
    fn test_qhead_rewind_after_deep_conflict() {
        // Deciding a=true forces b, then conflicts; backtracking must rewind
        // qhead so the sibling subtree re-propagates correctly.
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        solver.add_unit(a); // a must be true
        solver.add_implies(a, b); // a=true → b=true
        solver.add_clause(&[-a, -b]); // …but ¬a ∨ ¬b → conflict under a=true
        // The escape: clause below forces c once a's subtree fails? No — the
        // formula (a) ∧ (a→b) ∧ (¬a∨¬b) is UNSAT. Verify that:
        let result = solver.solve();
        assert!(result.is_unsat());

        // Now the SAT sibling case: drop the conflicting clause, add a deep
        // chain under a decision variable so rewind + re-propagation runs.
        let mut solver = Solver::new();
        let d = solver.new_var() as i32;
        let e = solver.new_var() as i32;
        let f = solver.new_var() as i32;
        let g = solver.new_var() as i32;
        solver.add_clause(&[d, e]); // branch point
        solver.add_clause(&[-d, -e]); // at most one true
        solver.add_implies(d, f);
        solver.add_implies(f, g);
        solver.add_unit(-g); // g must be false
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            // d=true would force g=true → conflict; solver must recover via
            // the d=false sibling.
            assert!(
                !model[0],
                "d must be false after rewinding the d=true subtree"
            );
            assert!(model[1], "e must be true");
        }
    }

    // ── Backtrack/flag stress: deep chain implication + conflict + retry ─────

    #[test]
    fn test_deep_chain_conflict_retry() {
        // x1↔x2↔…↔x20, x20 true → forces all true (forward and backward);
        // then ¬x11 conflicts deep inside the propagated region and the
        // solver must retry/backtrack correctly to UNSAT.
        let mut solver = Solver::new();
        let n = 20;
        let lits: Vec<i32> = (0..n).map(|_| solver.new_var() as i32).collect();
        for w in lits.windows(2) {
            solver.add_equiv(w[0], w[1]);
        }
        solver.add_unit(lits[n - 1]); // forces the whole chain true
        assert!(solver.solve().is_sat());
        if let SolveResult::Sat(model) = solver.solve() {
            assert!(model[..n].iter().all(|&v| v), "chain must fully propagate");
        }

        // Now add a contradiction in the middle and re-solve incrementally.
        solver.add_unit(-lits[10]);
        assert!(solver.solve().is_unsat());
    }

    // ── Model verification over random 3-SAT (property test) ─────────────────

    fn verify_model_against_clauses(clauses: &[Vec<i32>], model: &[bool]) -> bool {
        clauses.iter().all(|c| {
            c.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                let val = model[var - 1];
                if lit > 0 { val } else { !val }
            })
        })
    }

    #[test]
    fn test_model_satisfies_all_clauses_random() {
        // For every SAT answer over randomized instances, the returned model
        // must satisfy every clause — including binaries (fast path) and
        // units.
        for seed in 0..60u64 {
            let num_vars = 6 + (seed as usize % 12);
            let num_clauses = num_vars * 4;
            let clauses = gen_random_3sat(seed, num_vars, num_clauses);

            let mut solver = Solver::new();
            for _ in 0..num_vars {
                solver.new_var();
            }
            for c in &clauses {
                solver.add_clause(c);
            }
            if let SolveResult::Sat(model) = solver.solve() {
                assert!(
                    verify_model_against_clauses(&clauses, &model),
                    "seed={seed}: returned model violates a clause"
                );
            }
        }
    }

    #[test]
    fn test_model_satisfies_clauses_added_incrementally() {
        // Solve, add clauses, re-solve: the new model must satisfy BOTH the
        // old and new clauses (covers warm-start watch reuse).
        let mut solver = Solver::new();
        let lits: Vec<i32> = (0..8).map(|_| solver.new_var() as i32).collect();
        solver.add_clause(&[lits[0], lits[1]]);
        let first = solver.solve();
        let first_clauses = extract_clauses(&solver);
        if let SolveResult::Sat(m) = &first {
            assert!(verify_model_against_clauses(&first_clauses, m));
        }

        solver.add_implies(lits[0], lits[2]);
        solver.add_implies(lits[2], -lits[0]);
        solver.add_exactly_one(&[lits[3], lits[4], lits[5]]);
        let second = solver.solve();
        let all_clauses = extract_clauses(&solver);
        if let SolveResult::Sat(m) = second {
            assert!(verify_model_against_clauses(&all_clauses, &m));
        }
    }

    // ── Binary fast path correctness (heavier implications workload) ─────────

    #[test]
    fn test_binary_fast_path_equivalence_ring() {
        // Ring of equivalences over 30 vars — pure binary clauses.
        let mut solver = Solver::new();
        let n = 30;
        let lits: Vec<i32> = (0..n).map(|_| solver.new_var() as i32).collect();
        for i in 0..n {
            solver.add_equiv(lits[i], lits[(i + 1) % n]);
        }
        solver.add_unit(lits[0]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SolveResult::Sat(model) = result {
            assert!(model[..n].iter().all(|&v| v), "equivalence ring must unify");
        }

        // Flip one link: ring becomes unsatisfiable.
        let mut solver = Solver::new();
        let lits: Vec<i32> = (0..n).map(|_| solver.new_var() as i32).collect();
        for i in 0..(n - 1) {
            solver.add_equiv(lits[i], lits[i + 1]);
        }
        solver.add_equiv(lits[n - 1], -lits[0]);
        solver.add_unit(lits[0]);
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_binary_watch_conflict_and_propagation() {
        // Binary clauses in both polarities: conflict via the fast path and
        // propagation through chains reaching the same variable from two sides.
        let mut solver = Solver::new();
        let a = solver.new_var() as i32;
        let b = solver.new_var() as i32;
        solver.add_implies(a, b);
        solver.add_implies(-a, b);
        solver.add_unit(-b); // b must be false → both a and ¬a conflict → UNSAT
        assert!(solver.solve().is_unsat());
    }

    #[test]
    fn test_unit_propagation_from_ternary_clauses() {
        // E5/A4: chained unit implications through clauses of length ≥ 3 must
        // be derived by propagation alone — zero decisions. Before the fix the
        // `None` arm asserted nothing, so every unit cost a decision + conflict.
        let mut s = Solver::new();
        s.ensure_var(6);
        s.add_clause(&[1, 2, 3]);
        s.add_unit(-1);
        s.add_unit(-2);
        s.add_clause(&[4, 5, 6]);
        s.add_unit(-4);
        s.add_unit(-5);
        assert!(s.solve().is_sat());
        assert_eq!(
            s.stats.decisions, 0,
            "units implied by ternary clauses must propagate without decisions"
        );
    }

    #[test]
    fn test_conflict_bumps_activity_without_binaries() {
        // E5/A5: on a formula with no binary clauses, conflicts must still
        // update VSIDS — otherwise `choose_var` degenerates to fixed order.
        let mut s = Solver::new();
        s.ensure_var(6);
        // A 3-SAT-ish core that forces at least one conflict.
        for c in [
            [1, 2, 3],
            [-1, -2, 3],
            [-1, 2, -3],
            [1, -2, -3],
            [-1, -2, -3],
            [1, 2, -3],
        ] {
            s.add_clause(&c);
        }
        let _ = s.solve();
        assert!(
            s.activity.iter().skip(1).any(|&a| a > 0.0),
            "conflicts must bump variable activity even without binary clauses"
        );
    }

    /// A3: `trail_limits[0]` must be the *post-unit* trail length. The old
    /// `push(0)` claimed level 1 started at index 0, so any `backtrack(0)`
    /// (a restart policy, an assumption rewind) popped the eagerly asserted
    /// units — which are only asserted at solve start — silently losing them.
    #[test]
    fn test_backtrack_zero_keeps_level_zero_units() {
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        s.add_unit(a);
        assert!(s.solve().is_sat());
        s.backtrack(0);
        assert_eq!(
            s.assignment[a as usize],
            Some(true),
            "level-0 units must survive backtrack(0)"
        );
    }
}
