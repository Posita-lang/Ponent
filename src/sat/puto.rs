//! A CDCL SAT solver with 1UIP clause learning.
//!
//! Named **Puto** (Latin: "I think"), this is the clause-learning sibling of
//! [`crate::sat::solvo`](Solvo). Where Solvo backtracks chronologically and
//! learns nothing, Puto performs conflict-driven clause learning (CDCL):
//!
//! - **1UIP conflict analysis**: on every conflict, resolution over the
//!   implication graph (`reason[]`) produces a learnt clause asserting the
//!   *first unique implication point*; the learnt clause is a pure resolution
//!   consequence of the input formula — never of assumptions — so it is
//!   globally valid and persists across incremental solves.
//! - **Backjumping**: `cancel_until` rewinds directly to the highest level
//!   remaining in the learnt clause, skipping irrelevant subtrees that
//!   chronological backtracking would re-explore.
//! - **Restarts**: Luby-scaled conflict budgets (base 512); combined with
//!   phase saving this makes the search provably terminating per budget.
//! - **Learnt clause database management**: LBD ("glue") scoring, learnt
//!   clause activity (bumped when used as a conflict-analysis reason), and
//!   half-database reduction against a growing high-water mark. Binary
//!   learnts are never deleted; deleted clauses remain as tombstones in the
//!   watchlists (lazy deletion) until the next database reduction compacts
//!   them — id-stable and sound, at a bounded memory cost.
//! - **VSIDS**: variables are bumped when their literals enter conflict
//!   analysis; the propagation-time bump Solvo uses is deliberately absent
//!   (nonstandard there, and irrelevant once learnt-clause bumping exists).
//!   Auxiliary encoding variables (Sinz/totalizer/commander) therefore start
//!   at zero activity and are only promoted if they participate in conflicts.
//! - **Assumption conflict cores** (see [`SolveResult::Unsat`]): when a
//!   solve under assumptions fails, MiniSat-style `analyze_final` extracts
//!   the subset of assumptions that jointly contradict the formula.
//! - **Inprocessing** (behind `preprocess`, on by default): forward/backward
//!   subsumption on every clause add, and bounded variable elimination
//!   (DP-style resolution) between solves, with model reconstruction so the
//!   returned model still satisfies the *original* formula.
//!
//! Soundness contract (mirrors Solvo): an answer is either sound
//! (`Sat`/`Unsat`) or honest (`Unknown` on limit exhaustion); resource
//! limits can degrade an answer but never fabricate one.
//!
//! # Example
//! ```
//! use sat::puto::Solver;
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

use super::{Limits, SolveResult, SolverStats, VarOrder};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default maximum number of decisions before returning
/// [`SolveResult::Unknown`].
const DEFAULT_MAX_DECISIONS: usize = 500_000;

/// Default maximum number of conflicts before returning
/// [`SolveResult::Unknown`].
const DEFAULT_MAX_CONFLICTS: usize = 500_000;

/// Default maximum number of unit propagation steps before returning
/// [`SolveResult::Unknown`].
const DEFAULT_MAX_PROPAGATIONS: usize = 5_000_000;

/// Default time limit in microseconds (5 seconds).
const DEFAULT_MAX_TIME_MICROS: u64 = 5_000_000;

/// VSIDS activity decay factor (lazy decay: `var_inc` grows by `1/decay`
/// per conflict instead of decaying every activity — O(1) per conflict).
const ACTIVITY_DECAY: f64 = 0.95;

/// Initial VSIDS activity increment.
const ACTIVITY_BUMP: f64 = 1.0;

/// Activity rescale threshold (activities and `var_inc` are scaled down by
/// 1e100 when exceeded; exponentially rare, amortised O(1)).
const ACTIVITY_RESCALE: f64 = 1e100;

/// Luby restart base: the first restart fires after this many conflicts,
/// then the budget follows the Luby sequence 1,1,2,1,1,2,4,… × base.
const RESTART_BASE: usize = 512;

/// Initial learnt-database high-water mark; grows by
/// [`REDUCTION_GROWTH`] after each reduction.
const MAX_LEARNTS_BASE: usize = 4_000;

/// High-water mark growth per reduction (MiniSat's 500·le growth).
const REDUCTION_GROWTH: usize = 500;

/// Check the wall-clock time limit only every N search iterations.
const TIME_CHECK_INTERVAL: usize = 64;

/// At-most-one pairwise encoding threshold (small groups).
const AMO_PAIRWISE_THRESHOLD: usize = 8;

/// At-most-one commander-encoding threshold (large groups): between the two
/// thresholds the sequential-counter (Sinz) encoding is used; beyond it the
/// commander encoding propagates better.
const AMO_COMMANDER_THRESHOLD: usize = 50;

/// Commander-encoding group size (Klieber; group AMO is pairwise).
const AMO_COMMANDER_GROUP: usize = 10;

/// Bounded-variable-elimination gate: a variable is only eliminated when
/// `pos_occurrences × neg_occurrences ≤ BVE_PRODUCT_LIMIT` (bounds the
/// resolvent blow-up; DP with an unbounded product is a compile-time DoS).
const BVE_PRODUCT_LIMIT: usize = 20;
/// Hard cap on a single resolvent's length (see `A8#9`): the product gate
/// bounds how many resolvents the pair of polarities may produce, not how
/// long each one is; a resolvent longer than this refuses the elimination.
const BVE_MAX_RESOLVENT_LEN: usize = 100;

/// Per-add subsumption is only attempted for clauses up to this literal
/// count. Long clauses are poor subsumption candidates and their occurrence
/// scans are expensive; they are left to the periodic global pass.
const SUBSUME_MAX_LEN: usize = 32;

/// Per-add subsumption is disabled once the database grows past this many
/// clauses — at that scale the O(k·m) scans dominate construction and the
/// BVE pass provides the same pruning more cheaply.
const SUBSUME_MAX_DB: usize = 50_000;

/// On-the-fly learnt subsumption only scans asserting-literal occurrence
/// lists up to this size, keeping the per-conflict cost bounded.
const SUBSUME_SCAN_MAX: usize = 4_096;

/// Vivification bounds: only original clauses up to this length and
/// databases up to this size are probed during preprocessing.
const VIVIFY_MAX_LEN: usize = 8;
const VIVIFY_MAX_DB: usize = 20_000;

// ─── Internal types ──────────────────────────────────────────────────────────

/// A variable with an optional debug name.
#[derive(Debug, Clone)]
pub struct Var {
    pub name: Option<String>,
}

/// A CNF clause with watched-literal bookkeeping.
///
/// Watches are stored **by value**. The clause database is a single vector
/// (`clauses`) shared by original and learnt clauses — learnt clauses carry
/// the `learnt` flag plus LBD/activity scoring. Deleted clauses remain as
/// tombstones (`deleted = true`, id-stable) until the next database
/// reduction compacts the watchlists around them.
#[derive(Debug, Clone)]
struct Clause {
    /// The literals in this clause.
    ///
    /// C1: inline capacity 4 keeps the overwhelmingly common binary/ternary
    /// clause — the compiler workload is implication-dominated — off the heap
    /// entirely. Longer clauses spill to the heap exactly as a `Vec` did.
    lits: crate::smallvec::SmallVec<[i32; 4]>,
    /// The two watched literals (equal for unit clauses; units do not live
    /// in the database — see [`Solver::level0_units`] — so in practice the
    /// two watches are always distinct).
    w0: i32,
    w1: i32,
    /// Whether this is a learnt clause.
    learnt: bool,
    /// LBD ("glue") score: number of distinct decision levels spanned.
    /// Lower is better; the primary reduction key.
    lbd: u32,
    /// Learnt-clause activity: bumped when the clause is used as a conflict
    /// analysis reason; the secondary reduction key.
    activity: f64,
    /// Tombstone flag (lazy deletion from subsumption / DB reduction).
    deleted: bool,
    /// Number of live assignments currently using this clause as a propagation
    /// reason (A6). A locked clause must never be deleted by `reduce_db`:
    /// deleting a reason degrades `analyze`/`analyze_final` cores and pruning,
    /// and the whole point of this solver is assumption-core quality.
    locked: u32,
    /// 64-bit literal signature (C9): `OR` of `1 << (var % 64)` over the
    /// clause's variables. A candidate whose signature is not a subset of
    /// the query's cannot subsume it — one `AND` rejects it. A *missed*
    /// subsumption is sound (subsumption is optional), so a stale signature
    /// can only cost performance, never correctness; the exact subset check
    /// is still performed for every surviving candidate.
    sig: u64,
}

/// Signature of a literal slice (C9).
#[inline]
fn lits_signature(lits: &[i32]) -> u64 {
    lits.iter()
        .fold(0u64, |acc, &l| acc | (1u64 << (lit_var(l) & 63)))
}

/// Outcome of one propagation pass. `Limit` is deliberately distinct from
/// `Conflict`: a limit must degrade to [`SolveResult::Unknown`], never
/// fabricate an `Unsat`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PropOutcome {
    Ok,
    /// Falsified clause id (for conflict analysis).
    Conflict(u32),
    Limit,
}

// ─── Literal helpers ─────────────────────────────────────────────────────────
// (Same encoding as Solvo: literals are ±var with 1-based vars; watchlist
// index = 2*(v-1) for positive, 2*(v-1)+1 for negative.)

#[inline]
fn lit_var(lit: i32) -> usize {
    lit.unsigned_abs() as usize
}

#[inline]
fn lit_sign(lit: i32) -> bool {
    lit > 0
}

/// Negate a literal. Panics (debug) on `i32::MIN`, which has no negation;
/// all entry points reject it before literals reach the solver core.
#[inline]
fn lit_neg(lit: i32) -> i32 {
    debug_assert_ne!(lit, i32::MIN, "literal i32::MIN cannot be negated");
    -lit
}

#[inline]
fn lit_index(lit: i32) -> usize {
    let v = lit_var(lit);
    if lit > 0 {
        2 * (v - 1)
    } else {
        2 * (v - 1) + 1
    }
}

/// Normalise a clause: drop zeros, dedup, detect tautologies (`None`).
/// Short clauses use a linear-scan dedup (construction is hot and they are
/// the common case); long clauses sort first to avoid quadratic scans.
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
                    dup_or_taut = true;
                } else {
                    return None;
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

/// Sort-based normalisation for long clauses: sort by (variable, sign), then
/// drop adjacent duplicates and reject a variable appearing in both
/// polarities. O(k log k).
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

/// Linear membership test — clauses are short; no hashing needed.
#[inline]
fn lits_contains(lits: &[i32], lit: i32) -> bool {
    lits.contains(&lit)
}

// ─── Luby restart sequence ───────────────────────────────────────────────────

/// Luby sequence 1, 1, 2, 1, 1, 2, 4, 1, … (MiniSat's `luby(2, x)`), used
/// for restart budgets: `budget = LUBY * luby(restart_index)`.
fn luby_pow2(x: usize) -> usize {
    // Find the finite subsequence containing index x and its size.
    let (mut size, mut seq) = (1usize, 0usize);
    while size < x + 1 {
        seq += 1;
        size = 2 * size + 1;
    }
    let mut x = x;
    while size - 1 != x {
        // MiniSat's `luby(y, x)` decrements `seq` here as it walks back down
        // the sequence. Omitting it makes the budget double at every restart
        // (1, 2, 2, 4, 4, 4, 4, 8, …) instead of following Luby
        // (1, 2, 1, 1, 2, 4, 1, …), which silently disables restarts after a
        // few dozen intervals and collapses portfolio restart diversity.
        size = (size - 1) / 2;
        seq -= 1;
        x %= size;
    }
    1usize << seq
}

// ─── Solver ──────────────────────────────────────────────────────────────────

/// One bounded-variable-elimination log entry: the eliminated variable and
/// its original clauses (with `v` / `¬v` still present — resurrection
/// re-adds them verbatim; model reconstruction strips `v` when evaluating).
type ElimEntry = (usize, Vec<Vec<i32>>, Vec<Vec<i32>>);

/// A CDCL SAT solver with 1UIP clause learning and assumption cores.
#[derive(Debug)]
pub struct Solver {
    /// All declared variables (index 0 is unused; variables are 1-based).
    vars: Vec<Var>,
    /// Unified clause database: originals + learnts (flagged). Ids are u32
    /// indices into this vector; deleted clauses stay as tombstones.
    clauses: Vec<Clause>,
    /// Watch lists: watchlist[lit_index] → clause ids watching that literal
    /// (firing when the literal is falsified).
    ///
    /// C2: inline capacity 4 keeps the typical literal's watch list (≈2–4
    /// entries under 2-watched-literal propagation) off the heap; the arena
    /// refactor that would share one backing buffer is a larger change, but
    /// this removes the per-literal allocation directly.
    watchlist: Vec<crate::smallvec::SmallVec<[u32; 4]>>,
    /// Occurrence lists for **all** clauses (originals + learnts):
    /// occurrence[lit_index] → clause ids containing the literal. Used by
    /// subsumption-on-add, on-the-fly learnt subsumption, and bounded variable
    /// elimination (which filters to non-learnt clauses explicitly). Deletion
    /// is lazy (tombstones are skipped at scan time and compacted by
    /// [`Solver::rebuild_occurrence`]).
    occurrence: Vec<Vec<u32>>,

    /// Current partial assignment: var → Some(true/false) or None.
    assignment: Vec<Option<bool>>,
    /// Decision level per variable.
    level: Vec<usize>,
    /// Reason clause id per propagated variable (`None` = decision or
    /// level-0 fact).
    reason: Vec<Option<u32>>,
    /// Saved phase per variable (phase saving; persists across solves).
    saved_phase: Vec<bool>,
    /// BVE-eliminated variables: no clause in the database mentions them;
    /// branching skips them and the model is reconstructed from
    /// [`Solver::elim_order`] after `Sat`.
    eliminated: Vec<bool>,
    /// Elimination log for model reconstruction, in elimination order.
    elim_order: Vec<ElimEntry>,

    /// Current decision level.
    current_level: usize,
    /// Trail: assigned literals in order (true literals).
    trail: Vec<i32>,
    /// Trail limits: decision level → trail length at that level's start.
    trail_lim: Vec<usize>,
    /// Propagation queue head into `trail` (incremental propagation).
    qhead: usize,
    /// Number of assigned variables.
    assigned_count: usize,

    /// Level-0 facts (unit clauses): permanent, re-asserted every solve.
    level0_units: Vec<i32>,
    /// Fast check: which variables carry a level-0 fact (BVE skips these).
    fact_vars: Vec<bool>,

    /// VSIDS activity per variable (1-based).
    activity: Vec<f64>,
    /// Max-heap over unassigned, non-eliminated variables keyed by activity.
    order: VarOrder,
    /// Current VSIDS increment (lazy decay).
    var_inc: f64,
    /// Learnt-clause activity increment (lazy decay), kept **separate** from
    /// `var_inc` (A5). Sharing one increment meant a clause-table rescale also
    /// divided the variable increment (freezing VSIDS for ~4.5k conflicts) and
    /// a variable rescale left clause activities at 1e100 (a rescale storm on
    /// every clause bump). Each table now rescales only with its own
    /// increment.
    cla_inc: f64,

    /// Conflict-analysis scratch, reused across conflicts (B1): the `seen`
    /// marks, the O(cone) clear-list, and the LBD level buffer.
    analyze_seen: Vec<bool>,
    analyze_marks: Vec<usize>,
    analyze_levels: Vec<usize>,
    /// Scratch buffers for the remaining per-conflict allocations (B1):
    /// the watch-ordering buffer, the recursive-minimisation stack/added
    /// lists, and `analyze_final`'s seen marks and conflict clause.
    analyze_ordered: Vec<i32>,
    redundant_stack: Vec<i32>,
    redundant_added: Vec<usize>,
    analyze_final_seen: Vec<bool>,
    analyze_final_conflict: Vec<i32>,
    /// Reusable candidate-id buffer for subsumption-on-add (B3).
    scratch_candidates: Vec<u32>,
    /// Journal of `(var, old_phase)` pairs touched while probing, so
    /// vivification can restore phases without cloning the whole vector (B4).
    phase_journal: Vec<(usize, bool)>,
    phase_journal_mark: Vec<bool>,
    phase_journaling: bool,

    /// Variables mentioned by user clauses added/resurrected since the last BVE
    /// sweep (B2) and a membership flag. The sweep visits these first so an
    /// incremental workload does not pay the full O(vars) scan before reaching
    /// the variables that actually changed.
    touched_vars: Vec<usize>,
    touched_flag: Vec<bool>,

    /// Short-window EMA of the decisions-between-conflicts gap and its value at
    /// the last restart (C5), used to defer restarts while the conflict rate
    /// is still improving. `None` means "no sample yet" — a legitimate EMA of
    /// `0.0` (conflict chains with zero decisions) must not be mistaken for the
    /// first sample forever, as the old `== 0.0` sentinel did.
    conflict_ema: Option<f64>,
    prev_conflict_ema: Option<f64>,
    /// C5: short/long-window EMAs of the learnt clauses' LBD (Glucose-style).
    /// A degraded recent LBD (`fast > slow`) is a more robust restart trigger
    /// than the conflict-gap EMA alone, which can look "improving" while the
    /// learnt DB is getting worse.
    lbd_fast: Option<f64>,
    lbd_slow: Option<f64>,
    last_conflict_decisions: usize,

    /// Append-only log of the *normalised input clauses* (). This is the
    /// authoritative formula for `clauses_snapshot`/`to_dimacs`: inprocessing
    /// (BVE resolvents, subsumption tombstoning, vivification strengthening)
    /// never touches it, so the audit artifact always describes what the
    /// caller actually added — not the solver's derived database.
    original_clauses: Vec<Vec<i32>>,

    /// Set when the *input formula* changed since the last solve (B3): a user
    /// `add_clause`/`add_fact` or a resurrection. Learnts do **not** dirty it —
    /// they are pure consequences and never mention level-0 literals, so the
    /// level-0 trail's propagation state stays valid. When clean, a repeated
    /// `solve_assumptions` on a stable formula skips the O(formula)
    /// occurrence rebuild, the BVE sweep and the full re-propagation.
    formula_dirty: bool,

    /// Reverse-index depth (B6): for each variable, how many logged BVE
    /// elimination entries mention it in their stored clauses. Lets
    /// `restore_eliminated_dependency` answer "does any entry reference `v`?"
    /// in O(1) instead of scanning the whole accumulated `elim_order` (and
    /// every stored clause literal) once per candidate variable — the hidden
    /// quadratic that made incremental query latency grow with usage.
    elim_mentions: Vec<u32>,

    /// Number of live *original* (non-learnt, non-deleted) clauses. Used to
    /// gate the subsumption-on-add pass: `clauses.len()` counts tombstones
    /// and learnts, so on a churny database it would silently switch
    /// preprocessing off long before the formula actually got large.
    live_originals: usize,

    /// Learnt clause ids (for database reduction).
    learnts: Vec<u32>,
    /// Current learnt-database high-water mark.
    max_learnts: usize,
    /// Upper bound on `max_learnts` (configurable via
    /// [`Solver::set_max_learnts_cap`]). The default mirrors the historical
    /// hard cap.
    max_learnts_cap: usize,
    /// Conflicts accumulated across **all** solves. The dynamic high-water
    /// mark is derived from this, not from `stats.conflicts` (which resets at
    /// the start of every solve, so a mark grown from it shrank back to the
    /// base each time).
    total_conflicts: usize,

    /// Assumptions for the running solve, one decision level each.
    assumptions: Vec<i32>,

    /// Conflicts since the last restart (Luby budget counter).
    conflicts_since_restart: usize,
    /// Luby sequence index.
    restart_seq: usize,
    /// Conflict budget for the current restart interval (`RESTART_BASE ×
    /// luby(restart_seq)`), recomputed only when the interval advances.
    restart_budget: usize,
    /// Base restart interval for this solver instance (portfolio knob).
    restart_base: usize,
    /// VSIDS decay factor applied per conflict (portfolio knob).
    var_decay: f64,
    /// Initial polarity for free decisions (portfolio knob).
    default_phase: bool,
    /// Optional cooperative-cancellation flag (portfolio coordination).
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// C7: optional portfolio-wide learnt exchange. Each worker publishes its
    /// low-LBD (glue) learnts into this slab at restart boundaries and imports
    /// entries it has not seen yet. Every published clause is a logical
    /// consequence of the *original* formula, so importing it is sound
    /// regardless of the exporting worker's state. `None` outside a sharing
    /// portfolio.
    learnt_share: Option<std::sync::Arc<std::sync::Mutex<Vec<Vec<i32>>>>>,
    /// Number of shared-slab entries this solver has already imported.
    import_cursor: usize,
    /// Clause-id high-water mark already published (avoids rescans).
    shared_published_cid: u32,

    /// Global unsatisfiability (empty clause learnt at level 0). Once set,
    /// every future solve returns `Unsat` immediately.
    ok: bool,
    /// Whether an empty clause was added explicitly.
    has_empty_clause: bool,

    /// Inprocessing switch (subsumption-on-add + BVE + vivification).
    preprocess: bool,
    /// C8: optional DRAT proof trail. Enabled via [`enable_drat`], which also
    /// disables preprocessing: BVE/vivify derivations are not RUP, so a valid
    /// external proof must consist only of learnt clauses (RUP) plus the final
    /// empty clause. `None` in normal operation.
    drat: Option<String>,
    /// Bounded-variable elimination sub-switch (vivification/subsumption stay
    /// governed by [`Solver::preprocess`]).
    bve_enabled: bool,
    /// Set when a clause was added/resurrected/derived since the last
    /// vivification pass; cleared once the clause database has been probed.
    needs_vivify: bool,

    limits: Limits,
    /// Statistics (per-solve + cumulative; see [`SolverStats`]).
    pub stats: SolverStats,
    /// Start time of the current solve (time limit).
    solve_start: Option<Instant>,
    /// Search iteration counter (throttles time-limit checks).
    loop_tick: usize,
}

impl Solver {
    /// Create a new, empty solver.
    pub fn new() -> Self {
        Solver {
            vars: vec![Var { name: None }],
            clauses: Vec::new(),
            watchlist: Vec::new(),
            occurrence: Vec::new(),
            assignment: vec![None],
            level: vec![0],
            reason: vec![None],
            saved_phase: vec![false],
            eliminated: vec![false],
            elim_order: Vec::new(),
            current_level: 0,
            trail: Vec::new(),
            trail_lim: Vec::new(),
            qhead: 0,
            assigned_count: 0,
            level0_units: Vec::new(),
            fact_vars: vec![false],
            activity: vec![0.0],
            order: VarOrder::new(),
            var_inc: ACTIVITY_BUMP,
            cla_inc: ACTIVITY_BUMP,
            learnts: Vec::new(),
            max_learnts: MAX_LEARNTS_BASE,
            max_learnts_cap: MAX_LEARNTS_BASE + REDUCTION_GROWTH * 64,
            total_conflicts: 0,
            assumptions: Vec::new(),
            conflicts_since_restart: 0,
            restart_seq: 0,
            restart_budget: RESTART_BASE,
            restart_base: RESTART_BASE,
            var_decay: ACTIVITY_DECAY,
            default_phase: false,
            live_originals: 0,
            original_clauses: Vec::new(),
            touched_vars: Vec::new(),
            touched_flag: vec![false],
            formula_dirty: true,
            elim_mentions: Vec::new(),
            conflict_ema: None,
            prev_conflict_ema: None,
            lbd_fast: None,
            lbd_slow: None,
            last_conflict_decisions: 0,
            analyze_seen: Vec::new(),
            analyze_marks: Vec::new(),
            analyze_levels: Vec::new(),
            analyze_ordered: Vec::new(),
            redundant_stack: Vec::new(),
            redundant_added: Vec::new(),
            analyze_final_seen: Vec::new(),
            analyze_final_conflict: Vec::new(),
            scratch_candidates: Vec::new(),
            phase_journal: Vec::new(),
            phase_journal_mark: Vec::new(),
            phase_journaling: false,
            cancel: None,
            learnt_share: None,
            import_cursor: 0,
            shared_published_cid: 0,
            ok: true,
            has_empty_clause: false,
            preprocess: true,
            drat: None,
            bve_enabled: true,
            needs_vivify: true,
            limits: Limits {
                max_decisions: DEFAULT_MAX_DECISIONS,
                max_conflicts: DEFAULT_MAX_CONFLICTS,
                max_propagations: DEFAULT_MAX_PROPAGATIONS,
                max_time_micros: DEFAULT_MAX_TIME_MICROS,
            },
            stats: SolverStats::default(),
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

    /// Set the Luby base restart interval (portfolio knob).
    pub fn set_restart_base(&mut self, base: usize) {
        self.restart_base = base.max(1);
        self.restart_budget = self
            .restart_base
            .saturating_mul(luby_pow2(self.restart_seq));
    }

    /// Set the per-conflict VSIDS decay factor (portfolio knob). Must be in
    /// `(0, 1)`; values outside that range are clamped.
    pub fn set_var_decay(&mut self, decay: f64) {
        if decay > 0.0 && decay < 1.0 {
            self.var_decay = decay;
        }
    }

    /// Set the initial polarity for free decisions (portfolio knob). Only
    /// affects variables declared from this point on.
    pub fn set_default_phase(&mut self, phase: bool) {
        self.default_phase = phase;
    }

    /// Attach a cooperative-cancellation flag. When set, the search stops
    /// (returning `Unknown`) shortly after the flag is raised. Used by the
    /// parallel portfolio to retire losing workers.
    pub fn set_cancel_flag(&mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.cancel = Some(flag);
    }

    /// C7: join a portfolio-wide learnt-exchange slab.
    pub fn set_learnt_share(&mut self, share: std::sync::Arc<std::sync::Mutex<Vec<Vec<i32>>>>) {
        self.learnt_share = Some(share);
    }

    /// C8: emit a DRAT proof trail of every learnt clause (and the final empty
    /// clause) into an in-memory string, retrievable with [`drat_proof`].
    ///
    /// Preprocessing is switched **off** because BVE/vivify derivations are
    /// not RUP steps: a checker given the original formula could not replay
    /// them. With preprocessing off, the only clauses added during the solve
    /// are CDCL learnt clauses, each of which is a RUP consequence of the
    /// formula plus the previously emitted clauses, so the trail is a valid
    /// DRAT proof against the original CNF.
    pub fn enable_drat(&mut self) {
        self.drat = Some(String::new());
        self.preprocess = false;
    }

    /// The accumulated DRAT trail, if [`enable_drat`] was called.
    pub fn drat_proof(&self) -> Option<&str> {
        self.drat.as_deref()
    }

    /// Append one DRAT addition line (`l1 l2 … 0`) when recording.
    fn drat_add(&mut self, lits: &[i32]) {
        let Some(buf) = self.drat.as_mut() else {
            return;
        };
        for &l in lits {
            buf.push_str(&l.to_string());
            buf.push(' ');
        }
        buf.push_str("0\n");
    }

    /// C7: publish this solver's glue learnts and import unseen ones.
    ///
    /// Runs at decision level 0 (from `restart`), so imported clauses cannot
    /// invalidate any live assignment. Importing is sound because every
    /// exported clause is a consequence of the original formula; the only
    /// cost of a stale/duplicate import is a redundant `add_learnt_clause`
    /// (deduped by subsumption-on-add).
    fn share_and_import_learnts(&mut self) {
        // C8: an imported clause is RUP w.r.t. its exporter's proof, not
        // necessarily w.r.t. this one, so it must never enter a DRAT trail.
        if self.drat.is_some() {
            return;
        }
        const SLAB_CAP: usize = 4096;
        let Some(share) = self.learnt_share.clone() else {
            return;
        };
        let mut imported: Vec<Vec<i32>> = Vec::new();
        {
            let mut slab = share.lock().unwrap_or_else(|e| e.into_inner());
            // Import first, then publish, then advance the cursor past our own
            // pushes so the next restart does not re-import them.
            let end = slab.len();
            let from = self.import_cursor.min(end);
            imported.extend_from_slice(&slab[from..end]);
            let start = self.shared_published_cid as usize;
            let stop = self.clauses.len();
            for cid in start..stop {
                if slab.len() >= SLAB_CAP {
                    break;
                }
                let c = &self.clauses[cid];
                if c.learnt && !c.deleted && c.lbd <= 2 && (2..=8).contains(&c.lits.len()) {
                    slab.push(c.lits.to_vec());
                }
            }
            self.shared_published_cid = stop as u32;
            self.import_cursor = slab.len();
        }
        let had_import = !imported.is_empty();
        for lits in imported {
            self.add_learnt_clause(lits, 2);
        }
        // A just-imported clause can be falsified by the level-0 trail, but the
        // trail is already fully propagated (qhead == trail.len()), so nothing
        // would inspect it. Rewinding qhead re-scans the (small) level-0 trail
        // and surfaces any such conflict on the next propagate.
        if had_import {
            self.qhead = 0;
        }
    }

    /// Enable/disable inprocessing (subsumption-on-add and bounded variable
    /// elimination). On by default; disable for bit-exact comparability
    /// with a no-preprocess baseline (the differential test harness does
    /// exactly that).
    pub fn set_preprocess(&mut self, enable: bool) {
        self.preprocess = enable;
    }

    /// Enable/disable bounded variable elimination alone (leaving
    /// subsumption-on-add and vivification governed by
    /// [`Solver::set_preprocess`]).
    pub fn set_bve(&mut self, enable: bool) {
        self.bve_enabled = enable;
    }

    /// Set the upper bound on the learnt-clause high-water mark.
    pub fn set_max_learnts_cap(&mut self, cap: usize) {
        self.max_learnts_cap = cap.max(1);
        self.max_learnts = self.max_learnts.min(self.max_learnts_cap);
    }

    // ── Variable management ──────────────────────────────────────────────────

    fn grow_var(&mut self, name: Option<String>) -> usize {
        let idx = self.vars.len();
        self.vars.push(Var { name });
        self.assignment.push(None);
        self.level.push(0);
        self.reason.push(None);
        self.saved_phase.push(self.default_phase);
        self.eliminated.push(false);
        self.activity.push(0.0);
        self.order.push_new(idx, &self.activity);
        self.fact_vars.push(false);
        self.watchlist.push(crate::smallvec::SmallVec::new());
        self.watchlist.push(crate::smallvec::SmallVec::new());
        self.occurrence.push(Vec::new());
        self.occurrence.push(Vec::new());
        self.touched_flag.push(false);
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
    pub fn ensure_var(&mut self, idx: usize) {
        while self.vars.len() <= idx {
            self.new_var();
        }
    }

    /// Number of declared variables.
    pub fn num_vars(&self) -> usize {
        self.vars.len() - 1
    }

    /// Number of live (non-deleted) clauses — originals plus learnts.
    pub fn num_clauses(&self) -> usize {
        self.clauses.iter().filter(|c| !c.deleted).count()
    }

    // ── Clause management ────────────────────────────────────────────────────

    /// Attach a clause to the watchlists under its two watched literals.
    fn attach(&mut self, cid: u32, w0: i32, w1: i32) {
        self.watchlist[lit_index(w0)].push(cid);
        self.watchlist[lit_index(w1)].push(cid);
        let c = &mut self.clauses[cid as usize];
        c.w0 = w0;
        c.w1 = w1;
    }

    /// Tombstone a clause, keeping [`Solver::live_originals`] in step.
    /// Idempotent-safe: a second call on the same clause would double-count,
    /// so the debug assertion makes the (currently impossible) case loud.
    fn tombstone_clause(&mut self, cid: u32) {
        {
            let c = &self.clauses[cid as usize];
            debug_assert!(!c.deleted, "tombstone_clause on an already-deleted clause");
            if c.deleted {
                return;
            }
            if !c.learnt {
                self.live_originals -= 1;
            }
        }
        self.clauses[cid as usize].deleted = true;
    }

    /// Assert a level-0 fact (unit clause). Returns `false` on contradiction
    /// with an existing fact — the formula is globally UNSAT.
    fn add_fact(&mut self, lit: i32) -> bool {
        debug_assert_ne!(lit, 0);
        let v = lit_var(lit);
        self.ensure_var(v);
        if self.fact_vars[v] {
            // Already have a fact for this var — consistent?
            let existing = self.level0_units.iter().find(|&&f| lit_var(f) == v);
            if let Some(&f) = existing
                && lit_sign(f) != lit_sign(lit)
            {
                return false;
            }
            return true;
        }
        self.fact_vars[v] = true;
        self.level0_units.push(lit);
        self.formula_dirty = true;
        true
    }

    /// Add a CNF clause (disjunction of signed literals).
    ///
    /// Normalisation (dedup / tautology removal / zero-literal dropping) and
    /// auto-declaration of mentioned variables match Solvo. Unit clauses
    /// become permanent level-0 facts. With preprocessing enabled, the clause
    /// participates in forward/backward subsumption against the existing
    /// database before insertion.
    pub fn add_clause(&mut self, clause: &[i32]) {
        if !self.ok {
            return; // globally UNSAT — accept and ignore for API stability
        }
        // An all-zero input (a DIMACS-style trailing-zeros line) is invalid
        // input, not the empty clause: match `add_unit`'s documented no-op
        // policy. A genuinely empty slice *is* the empty clause.
        if !clause.is_empty() && clause.iter().all(|&l| l == 0) {
            return;
        }
        // `i32::MIN` has no negation; in release the debug assertion inside
        // the normalisers would be compiled out and `lit_var` would demand a
        // 2^31-slot allocation. Reject it loudly at the boundary instead.
        if clause.iter().any(|&l| l == i32::MIN) {
            return;
        }
        let normalized = match normalize_clause(clause) {
            Some(lits) => lits,
            None => {
                self.stats.tautologies_removed += 1;
                return;
            }
        };
        let input_nonzero = clause.iter().filter(|&&l| l != 0).count();
        self.stats.duplicates_removed += input_nonzero - normalized.len();

        if normalized.is_empty() {
            self.has_empty_clause = true;
            self.ok = false;
            return;
        }

        let max_var = normalized.iter().map(|&l| lit_var(l)).max().unwrap();
        self.ensure_var(max_var);

        // : log the normalised input clause before subsumption/BVE/vivify can
        // rewrite or drop it. `clauses_snapshot`/`to_dimacs` read this log.
        self.original_clauses.push(normalized.clone());

        // B2: remember which variables this add touched so the next BVE sweep
        // starts with them instead of scanning the whole variable space.
        for &lit in &normalized {
            let v = lit_var(lit);
            if v < self.touched_flag.len() && !self.touched_flag[v] {
                self.touched_flag[v] = true;
                self.touched_vars.push(v);
            }
        }

        // A clause mentioning an eliminated variable invalidates that
        // elimination — resurrect it (its original clauses are re-added)
        // before the new clause joins the database. Without this, the model
        // reconstruction would only account for the stale stored clauses.
        for &lit in &normalized {
            let v = lit_var(lit);
            if self.eliminated[v] {
                self.resurrect(v);
            }
        }

        if normalized.len() == 1 {
            if !self.add_fact(normalized[0]) {
                self.ok = false;
            }
            self.stats.clauses_added += 1;
            return;
        }

        // Subsumption against the existing database (preprocessing). Gated so
        // construction stays near-linear on large formulas: long clauses and
        // big databases are pruned by the periodic BVE pass instead.
        if self.preprocess
            && normalized.len() <= SUBSUME_MAX_LEN
            && self.live_originals < SUBSUME_MAX_DB
            && !self.subsume_on_add(&normalized)
        {
            return; // subsumed by an existing clause — nothing to add
        }

        let cid = self.clauses.len() as u32;
        let (w0, w1) = (normalized[0], normalized[1]);
        self.clauses.push(Clause {
            w0,
            w1,
            sig: lits_signature(&normalized),
            lits: crate::smallvec::SmallVec::from_vec(normalized),
            learnt: false,
            lbd: 0,
            activity: 0.0,
            deleted: false,
            locked: 0,
        });
        self.attach(cid, w0, w1);
        for &lit in &self.clauses[cid as usize].lits {
            self.occurrence[lit_index(lit)].push(cid);
        }
        self.live_originals += 1;
        self.formula_dirty = true;
        self.needs_vivify = true;
        self.stats.clauses_added += 1;
    }

    /// Subsumption on add: (a) if an existing clause is subsumed by `lits`,
    /// delete it (or, on exact duplicate, keep it and skip the add);
    /// (b) if an existing clause subsumes `lits`, return `false` to skip the
    /// add entirely. Candidates come from the occurrence lists of `lits`'
    /// literals — subsumption requires a shared literal, so this is complete.
    ///
    /// Cost is bounded by the occurrence-list sizes at the literal indices;
    /// at the compiler's formula scale (hundreds of short clauses) that is
    /// negligible next to the propagation savings.
    fn subsume_on_add(&mut self, lits: &[i32]) -> bool {
        // B3: reuse the candidate buffer instead of allocating per add.
        let mut candidates = std::mem::take(&mut self.scratch_candidates);
        candidates.clear();
        for &lit in lits {
            let idx = lit_index(lit);
            for i in 0..self.occurrence[idx].len() {
                let cid = self.occurrence[idx][i];
                let c = &self.clauses[cid as usize];
                if c.deleted || c.learnt {
                    continue;
                }
                candidates.push(cid);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut keep = true;
        // C9: signature precheck — one AND rejects every candidate that
        // cannot contain, or be contained by, `lits`.
        let qsig = lits_signature(lits);
        for &cid in &candidates {
            let sig_d = self.clauses[cid as usize].sig;
            if (sig_d & !qsig) != 0 && (qsig & !sig_d) != 0 {
                continue;
            }
            let (d_lits_len, d_subsumes, subsumed_by) = {
                let d = &self.clauses[cid as usize].lits;
                let d_subsumes = d.len() >= lits.len() && lits.iter().all(|&l| lits_contains(d, l));
                let subsumed_by = d.len() < lits.len() && d.iter().all(|&l| lits_contains(lits, l));
                (d.len(), d_subsumes, subsumed_by)
            };
            if d_subsumes {
                if d_lits_len == lits.len() {
                    // Exact duplicate — keep the existing clause.
                    self.stats.duplicates_removed += 1;
                    self.scratch_candidates = candidates;
                    return false;
                }
                // `lits` strictly subsumes the existing clause — delete it.
                self.tombstone_clause(cid);
            } else if subsumed_by {
                keep = false;
            }
        }
        self.scratch_candidates = candidates;
        keep
    }

    /// Add a unit clause (single literal) as a permanent level-0 fact.
    /// Literal `0` (and `i32::MIN`) is invalid input, ignored as a no-op.
    pub fn add_unit(&mut self, lit: i32) {
        if lit == 0 || lit == i32::MIN {
            return;
        }
        // Route through add_clause: it handles auto-declaration, tombstone
        // bookkeeping, and resurrection of eliminated variables.
        self.add_clause(&[lit]);
    }

    /// Add an implication: `a → b` ≡ `¬a ∨ b`.
    ///
    /// Invalid literals (`0`, `i32::MIN`) are a no-op: negating `i32::MIN`
    /// would overflow before `add_clause` could reject it.
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
    /// Encoding by group size: pairwise ≤ 8 (best propagation at tiny n),
    /// sequential counter (Sinz) up to 49, commander beyond — the commander
    /// encoding's clause count is higher but its propagation is stronger at
    /// large n. Auxiliary variables start at zero VSIDS activity and are
    /// only promoted by participation in conflict analysis.
    pub fn add_at_most_one(&mut self, lits: &[i32]) {
        // : validate/dedup before any variable growth. The old up-front
        // `ensure_var(max_var)` also made `i32::MIN` (var 2^31) a release-build
        // OOM; it was redundant anyway (the encoders auto-declare through
        // `add_clause`/`new_var`), so it is gone.
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

    /// Add an at-most-`k` constraint over `lits` (totalizer encoding —
    /// see [`super::encode_at_most_k`]).
    pub fn add_at_most_k(&mut self, lits: &[i32], k: usize) {
        // The totalizer auto-declares every variable it mentions (and validates
        // the literal set), so no up-front `ensure_var` — which would have made
        // `i32::MIN` trigger a 2^31-slot allocation.
        super::encode_at_most_k(self, lits, k);
    }

    /// Add an exactly-one constraint: at-least-one + at-most-one.
    /// An empty literal set is invalid input, ignored as a no-op (matching
    /// Solvo — a debug panic on harmless input is not a contract).
    pub fn add_exactly_one(&mut self, lits: &[i32]) {
        if lits.is_empty() {
            return;
        }
        self.add_clause(lits);
        self.add_at_most_one(lits);
    }

    // ── Search state management ──────────────────────────────────────────────

    /// Current decision level (level 0 = permanent facts).
    #[inline]
    fn decision_level(&self) -> usize {
        self.current_level
    }

    /// Value of a literal under the current assignment.
    #[inline]
    fn lit_value(&self, lit: i32) -> Option<bool> {
        let var = lit_var(lit);
        if var >= self.assignment.len() {
            return None;
        }
        self.assignment[var].map(|v| if lit > 0 { v } else { !v })
    }

    /// Enqueue a literal at the current decision level with an optional
    /// reason clause. Returns `false` if the literal is already falsified.
    fn enqueue(&mut self, lit: i32, reason: Option<u32>) -> bool {
        let var = lit_var(lit);
        debug_assert!(var < self.assignment.len());
        match self.assignment[var] {
            Some(val) => val == lit_sign(lit),
            None => {
                self.assignment[var] = Some(lit_sign(lit));
                self.level[var] = self.current_level;
                // A6: a clause used as a reason is locked against deletion.
                if let Some(rcid) = reason {
                    self.clauses[rcid as usize].locked += 1;
                }
                self.reason[var] = reason;
                // B4: journal the pre-probe phase so vivification can restore
                // only the variables its probes touched.
                if self.phase_journaling {
                    if self.phase_journal_mark.len() < self.assignment.len() {
                        self.phase_journal_mark.resize(self.assignment.len(), false);
                    }
                    if !self.phase_journal_mark[var] {
                        self.phase_journal_mark[var] = true;
                        self.phase_journal.push((var, self.saved_phase[var]));
                    }
                }
                self.saved_phase[var] = lit_sign(lit); // phase saving
                self.assigned_count += 1;
                self.trail.push(lit);
                true
            }
        }
    }

    /// Backjump to `lvl`: undo all assignments above it, rewind `qhead`
    /// (MiniSat's backjump mechanism — everything above the rewind point is
    /// re-propagated on the next pass). Phases are kept (phase saving).
    fn cancel_until(&mut self, lvl: usize) {
        if self.current_level <= lvl {
            return;
        }
        // `trail_lim.len() == current_level` is a maintained invariant (only
        // `new_decision_level`/`cancel_until` touch either), so reaching this
        // point implies `lvl < trail_lim.len()`. The fallback is belt-and-
        // braces for a future refactor — it must never drop level-0 facts,
        // which are permanent by the solver's own contract, hence the assert.
        debug_assert!(
            lvl < self.trail_lim.len(),
            "level/trail_lim invariant broken"
        );
        let target = if lvl < self.trail_lim.len() {
            self.trail_lim[lvl]
        } else {
            0
        };
        while self.trail.len() > target {
            let lit = self.trail.pop().unwrap();
            let var = lit_var(lit);
            debug_assert!(self.assignment[var].is_some());
            // A6: release the reason clause's deletion lock.
            if let Some(rcid) = self.reason[var] {
                let c = &mut self.clauses[rcid as usize];
                c.locked = c.locked.saturating_sub(1);
            }
            self.assignment[var] = None;
            self.level[var] = 0;
            self.reason[var] = None;
            self.assigned_count -= 1;
            self.stats.backtracks += 1;
            // Back to candidate status: re-enter the branch order.
            self.order.insert(var, &self.activity);
        }
        self.qhead = self.qhead.min(target);
        self.trail_lim.truncate(lvl);
        self.current_level = lvl;
    }

    /// Open a new decision level (possibly a "dummy" one — MiniSat opens
    /// dummy levels for already-satisfied assumptions so that decision level
    /// `i+1` always corresponds to assumption `i`).
    fn new_decision_level(&mut self) {
        self.trail_lim.push(self.trail.len());
        self.current_level += 1;
    }

    /// Bump a variable's VSIDS activity with the current lazy increment.
    fn bump_activity(&mut self, var: usize) {
        self.activity[var] += self.var_inc;
        if self.activity[var] > ACTIVITY_RESCALE {
            for a in self.activity.iter_mut() {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
        // Activity only increases (or is uniformly rescaled, preserving
        // order), so the variable can only move up.
        self.order.bumped(var, &self.activity);
    }

    /// Bump a learnt clause's activity (used as a conflict-analysis reason),
    /// rescaling all learnt activities when values approach overflow.
    fn bump_clause_activity(&mut self, cid: u32) {
        let new_act = self.clauses[cid as usize].activity + self.cla_inc;
        if new_act > ACTIVITY_RESCALE {
            for &id in &self.learnts {
                self.clauses[id as usize].activity *= 1e-100;
            }
            // A5: rescale only the clause increment — the variable table is
            // untouched by this, so `var_inc` must stay put.
            self.cla_inc *= 1e-100;
        } else {
            self.clauses[cid as usize].activity = new_act;
        }
    }

    // ── Watched-literal propagation ──────────────────────────────────────────

    /// Unit propagation (incremental from `qhead`), recording reason clauses
    /// for propagated assignments (the substrate conflict analysis walks).
    /// Tombstoned clauses are skipped lazily.
    fn propagate(&mut self) -> PropOutcome {
        while self.qhead < self.trail.len() {
            if self.stats.propagations >= self.limits.max_propagations {
                return PropOutcome::Limit;
            }
            // A single pass can touch the whole watch database; poll the
            // cancel flag here so a cancelled portfolio worker returns
            // promptly instead of after the pass (A6).
            if self.stats.propagations.is_multiple_of(4096) && self.cancelled() {
                return PropOutcome::Limit;
            }
            let lit = self.trail[self.qhead];
            self.qhead += 1;
            self.stats.propagations += 1;

            let neg = lit_neg(lit);
            let neg_idx = lit_index(neg);
            debug_assert!(neg_idx < self.watchlist.len());

            let mut wi = 0;
            while wi < self.watchlist[neg_idx].len() {
                let cid = self.watchlist[neg_idx][wi];
                debug_assert!((cid as usize) < self.clauses.len());

                if self.clauses[cid as usize].deleted {
                    // Tombstone — lazy deletion; entry is compacted at the
                    // next database reduction / solve start.
                    self.watchlist[neg_idx].swap_remove(wi);
                    continue;
                }

                let (w0, w1) = {
                    let c = &self.clauses[cid as usize];
                    (c.w0, c.w1)
                };
                let is_w0 = w0 == neg;
                let is_w1 = w1 == neg;
                if !is_w0 && !is_w1 {
                    debug_assert!(false, "stale watchlist entry for clause {cid}");
                    wi += 1;
                    continue;
                }
                let other = if is_w0 { w1 } else { w0 };

                // Replacement search runs in BOTH arms below: whenever a
                // watched literal is falsified, the clause must re-secure a
                // non-false watch if any exists — propagating `other` without
                // this check would assert a "unit" that isn't one when a
                // third literal is true or unassigned.
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
                        continue;
                    }
                    Some(false) => {
                        // All literals false — conflict.
                        return PropOutcome::Conflict(cid);
                    }
                    None => {
                        // Unit: propagate `other`, record the reason clause.
                        // (The enqueue must be unconditional — wrapping it in
                        // debug_assert! would make it a no-op in release.)
                        let enqueued = self.enqueue(other, Some(cid));
                        debug_assert!(enqueued);
                        wi += 1;
                    }
                }
            }
        }
        PropOutcome::Ok
    }

    // ── Conflict analysis (1UIP) ─────────────────────────────────────────────

    /// 1UIP conflict analysis (MiniSat-style).
    ///
    /// Starting from the falsified conflict clause, resolve over `reason[]`
    /// backwards through the trail until exactly one literal of the current
    /// decision level remains — the *first unique implication point*. The
    /// resulting learnt clause is a pure resolution consequence of the input
    /// formula: decisions (and therefore assumptions) are never resolved
    /// over, they only bound where the walk stops, so learnts are globally
    /// valid and persist across incremental solves.
    ///
    /// Returns `(learnt, backjump_level, lbd)` where the learnt clause is
    /// ordered `[asserting_lit, highest_level_lit, …]` (that order becomes
    /// the two watches) and `backjump_level` is the highest level among the
    /// non-asserting literals.
    fn analyze(&mut self, confl_cid: u32) -> (Vec<i32>, usize, u32) {
        let current = self.decision_level();
        let mut learnt: Vec<i32> = Vec::new();
        let mut path_c: usize = 0;
        let mut p: Option<i32> = None; // literal being resolved out (None first)
        let mut index = self.trail.len(); // trail scan position (exclusive)
        let mut confl = confl_cid;

        // Reusable scratch (B1). Allocating `seen` per conflict was the classic
        // CDCL killer: at a 500k-conflict budget on a 100k-var formula it is tens
        // of GB of cumulative memset. The buffers are moved out of `self` so
        // `&mut self` calls (`bump_activity`, `lit_redundant`) stay borrowable.
        let mut seen = std::mem::take(&mut self.analyze_seen);
        let mut seen_list = std::mem::take(&mut self.analyze_marks);
        let mut levels = std::mem::take(&mut self.analyze_levels);
        debug_assert!(
            seen.iter().all(|&b| !b),
            "conflict-analysis scratch was not cleaned"
        );
        if seen.len() < self.assignment.len() {
            seen.resize(self.assignment.len(), false);
        }

        loop {
            // Bump learnt clauses used in the derivation.
            if self.clauses[confl as usize].learnt {
                self.bump_clause_activity(confl);
            }

            // Collect unseen literals of the clause (skip `p` by value).
            {
                let p_lit = p;
                let clause_len = self.clauses[confl as usize].lits.len();
                for j in 0..clause_len {
                    let q = self.clauses[confl as usize].lits[j];
                    if let Some(pl) = p_lit
                        && q == pl
                    {
                        continue;
                    }
                    let v = lit_var(q);
                    if !seen[v] && self.level[v] > 0 {
                        seen[v] = true;
                        seen_list.push(v);
                        self.bump_activity(v);
                        if self.level[v] >= current {
                            path_c += 1;
                        } else {
                            learnt.push(q);
                        }
                    }
                }
            }

            // Select the next literal to resolve over: the most recent
            // trail literal whose variable is marked.
            loop {
                index -= 1;
                let lit = self.trail[index];
                if seen[lit_var(lit)] {
                    p = Some(lit);
                    seen[lit_var(lit)] = false;
                    break;
                }
            }
            path_c -= 1;
            if path_c == 0 {
                break;
            }
            // Continue with the reason of the resolved-out literal. A
            // decision literal (reason None) at level > 0 must never be
            // resolved over before path_c reaches 0 — that would mean the
            // current level had no unique implication point.
            let p_var = lit_var(p.unwrap());
            confl = self.reason[p_var]
                .unwrap_or_else(|| panic!("1UIP analysis reached a decision literal"));
        }

        // Asserting literal: the negation of the last resolved-out literal.
        // `path_c == 0` guarantees at least one current-level literal was
        // seen and popped, so `p` is always `Some` here; use `expect` to make
        // a violated invariant loud instead of an opaque unwrap.
        let asserting = -p.expect("1UIP analysis: path_c reached 0 without an asserting literal");

        if learnt.is_empty() {
            // Every conflict literal sat at the current decision level — the
            // learnt clause is unit and asserts globally (backjump level 0).
            learnt.push(asserting);
            for &v in &seen_list {
                seen[v] = false;
            }
            seen_list.clear();
            levels.clear();
            self.analyze_seen = seen;
            self.analyze_marks = seen_list;
            self.analyze_levels = levels;
            return (learnt, 0, 1);
        }

        // Backjump level + watch order: the asserting literal must be
        // position 0 (it becomes watch 0 and the re-assertion target), and
        // the highest-level non-asserting literal position 1 (watch 1).
        let mut max_pos = 0;
        let mut max_lvl = 0;
        for (i, &l) in learnt.iter().enumerate() {
            let lv = self.level[lit_var(l)];
            if lv > max_lvl {
                max_lvl = lv;
                max_pos = i;
            }
        }
        let max_lit = learnt[max_pos];
        // B1: build the watch-ordered clause in a reused buffer, then swap it
        // into `learnt` (the previous `learnt` allocation becomes the next
        // call's scratch) instead of allocating a fresh Vec per conflict.
        let mut ordered = std::mem::take(&mut self.analyze_ordered);
        ordered.clear();
        ordered.reserve(learnt.len() + 1);
        ordered.push(asserting);
        ordered.push(max_lit);
        for (i, &l) in learnt.iter().enumerate() {
            if i != max_pos {
                ordered.push(l);
            }
        }
        self.analyze_ordered = std::mem::replace(&mut learnt, ordered);
        let backjump = max_lvl;

        // Recursive learnt-clause minimisation (MiniSat's `ccmin_mode = 2`):
        // a literal `q` is redundant when its implication graph, resolved
        // transitively through `reason`s, bottoms out only in other literals
        // of this learnt clause or in level-0 facts. Removing it yields a
        // strictly shorter clause with the same pruning power.
        //
        // `seen` marks exactly the non-asserting literals of `learnt` at this
        // point (the analysis loop unmarks each resolved literal). `keep`
        // preserves the asserting literal at position 0.
        // Abstract level set (B8, MiniSat's `ccmin_mode = 2`). A reason literal
        // whose level does not occur in the learnt clause cannot make the
        // literal being checked redundant, so reject it before walking its
        // implication cone. Levels ≥ 64 fall back to the unbounded walk
        // (the u64 mask cannot represent them).
        let mut abstract_levels: u64 = 0;
        for &l in &learnt {
            let lv = self.level[lit_var(l)];
            if lv < 64 {
                abstract_levels |= 1u64 << lv;
            }
        }

        {
            // B1: reuse the scratch buffer for the minimised clause too.
            let mut keep = std::mem::take(&mut self.analyze_ordered);
            keep.clear();
            keep.reserve(learnt.len());
            keep.push(learnt[0]);
            for i in 1..learnt.len() {
                let q = learnt[i];
                // C6: binary-reason fast path. A literal whose reason is the
                // binary clause `(q ∨ other)` is redundant iff `other` is
                // already level-0 or marked `seen` — the exact case
                // `lit_redundant`'s walk resolves in one step, but without the
                // DFS/scratch round-trip. Anything else falls back to the full
                // recursive check, so semantics are unchanged.
                let redundant = match self.reason[lit_var(q)] {
                    Some(rid) if self.clauses[rid as usize].lits.len() == 2 => {
                        let lits = &self.clauses[rid as usize].lits;
                        let other = if lits[0] == q { lits[1] } else { lits[0] };
                        let ov = lit_var(other);
                        self.level[ov] == 0
                            || seen[ov]
                            || self.lit_redundant(q, &mut seen, &mut seen_list, abstract_levels)
                    }
                    Some(_) => self.lit_redundant(q, &mut seen, &mut seen_list, abstract_levels),
                    None => false,
                };
                if !redundant {
                    keep.push(q);
                }
            }
            self.analyze_ordered = std::mem::replace(&mut learnt, keep);
        }

        // A7: LBD ("glue") is computed from the *minimised* clause. Computing
        // it before minimisation systematically over-counts levels (minimisation
        // can only shrink), which let `reduce_db` delete clauses whose true LBD
        // was 2 and made glue-2 protection a lie.
        levels.clear();
        levels.extend(learnt.iter().map(|&l| self.level[lit_var(l)]));
        levels.sort_unstable();
        levels.dedup();
        let lbd = levels.len() as u32;

        // Reset the seen marks and hand the scratch buffers back.
        for &v in &seen_list {
            seen[v] = false;
        }
        seen_list.clear();
        levels.clear();
        self.analyze_seen = seen;
        self.analyze_marks = seen_list;
        self.analyze_levels = levels;

        (learnt, backjump, lbd)
    }

    /// `true` when literal `q0` is implied by the rest of the learnt clause
    /// and level-0 facts alone — the recursive minimization predicate.
    ///
    /// Walks the implication graph backwards from `q0`'s reason. Every
    /// encountered literal must either be at level 0, already marked `seen`
    /// (i.e. part of the learnt clause / an already-proven redundant
    /// literal), or itself have a reason to recurse into. A decision literal
    /// that is not part of the clause breaks the proof.
    ///
    /// On success, freshly marked variables stay marked (they are proven
    /// redundant and may short-circuit later literals). On failure the marks
    /// added by this call are rolled back so a failed attempt cannot lend
    /// false support to a later one.
    fn lit_redundant(
        &mut self,
        q0: i32,
        seen: &mut [bool],
        marks: &mut Vec<usize>,
        abstract_levels: u64,
    ) -> bool {
        // B1: the DFS stack and the rollback list are reused across literals
        // (and across conflicts) instead of reallocated per literal.
        let mut stack = std::mem::take(&mut self.redundant_stack);
        let mut added = std::mem::take(&mut self.redundant_added);
        stack.clear();
        added.clear();
        stack.push(q0);
        let mut ok = true;
        'walk: while let Some(q) = stack.pop() {
            let Some(rid) = self.reason[lit_var(q)] else {
                // Decision literal outside the clause — not redundant.
                ok = false;
                break 'walk;
            };
            for &r in &self.clauses[rid as usize].lits {
                if r == q {
                    continue;
                }
                let v = lit_var(r);
                let lv = self.level[v];
                if lv == 0 || seen[v] {
                    continue;
                }
                // Level abstraction (B8): a reason literal at a level absent
                // from the learnt clause cannot participate in a redundancy
                // proof, so fail immediately instead of walking its whole
                // implication cone (a hidden quadratic on conflict-dense runs).
                if lv < 64 && (abstract_levels & (1u64 << lv)) == 0 {
                    ok = false;
                    break 'walk;
                }
                if self.reason[v].is_none() {
                    ok = false;
                    break 'walk;
                }
                seen[v] = true;
                added.push(v);
                // Successful marks are deliberately kept for the remaining
                // literals of this clause; record them so the caller can
                // clear every mark it owns.
                marks.push(v);
                stack.push(r);
            }
        }
        if !ok {
            for &v in &added {
                seen[v] = false;
            }
        }
        self.redundant_stack = stack;
        self.redundant_added = added;
        ok
    }

    /// Assumption-conflict analysis (MiniSat's `analyzeFinal`).
    ///
    /// Called with the true literal `p` that the assumption set forces
    /// (i.e. `-p` is the assumption that came out false). Walks the
    /// implication graph from `p` down to decision literals and returns the
    /// conflict clause `[p, ¬d₁, …, ¬dₘ]` where `dᵢ` are the decision
    /// literals implying `p`. Under assumptions, every decision literal at
    /// levels 1..=k *is* an assumption, so negating the clause's literals
    /// and intersecting with the assumption list yields the core.
    fn analyze_final(&mut self, p: i32) -> Vec<i32> {
        // B1: both the conflict clause and the seen marks are reused buffers.
        // The returned clause is owned by the caller, which stores it back.
        let mut conflict = std::mem::take(&mut self.analyze_final_conflict);
        conflict.clear();
        conflict.push(p);
        if self.decision_level() == 0 {
            return conflict;
        }
        let mut seen = std::mem::take(&mut self.analyze_final_seen);
        if seen.len() < self.assignment.len() {
            seen.resize(self.assignment.len(), false);
        }
        seen[lit_var(p)] = true;
        let bottom = if self.trail_lim.is_empty() {
            0
        } else {
            self.trail_lim[0]
        };
        for i in (bottom..self.trail.len()).rev() {
            let lit = self.trail[i];
            let v = lit_var(lit);
            if seen[v] {
                match self.reason[v] {
                    None => {
                        // Decision literal (an assumption under this scheme).
                        debug_assert!(self.level[v] > 0);
                        conflict.push(-lit);
                    }
                    Some(rid) => {
                        // Iterate by index: cloning the reason clause for
                        // every seen variable is free to remove (B9#1).
                        let rlen = self.clauses[rid as usize].lits.len();
                        for j in 0..rlen {
                            let r = self.clauses[rid as usize].lits[j];
                            if r != lit && self.level[lit_var(r)] > 0 {
                                seen[lit_var(r)] = true;
                            }
                        }
                    }
                }
                seen[v] = false;
            }
        }
        // All marks were cleared by the walk above; return the buffer.
        self.analyze_final_seen = seen;
        conflict
    }

    /// Extract the assumption core from an `analyze_final` conflict clause:
    /// the assumption literals whose negations appear in the clause, plus
    /// the failed assumption itself (the clause's first literal).
    fn core_from_conflict(&self, conflict: &[i32]) -> Vec<i32> {
        let mut core: Vec<i32> = Vec::new();
        for &l in conflict {
            let neg = -l;
            if self.assumptions.contains(&neg) && !core.contains(&neg) {
                core.push(neg);
            }
        }
        core
    }

    // ── Learnt database management ───────────────────────────────────────────

    /// Reduce the learnt database: delete the worst half (highest LBD
    /// first, then lowest activity), never delete binary learnts.
    ///
    /// Learnt clauses *are* occurrence-indexed (see
    /// [`Solver::add_learnt_clause`]), so tombstones are compacted out of the
    /// occurrence lists here as well as out of the watchlists — otherwise
    /// every later occurrence scan (subsumption, on-the-fly learnt
    /// subsumption, BVE candidate collection) would walk dead entries in a
    /// long single solve, where `rebuild_occurrence` may never run.
    fn reduce_db(&mut self) {
        // Learnts may have been tombstoned by on-the-fly subsumption; drop
        // their ids so the length reflects the live database.
        self.learnts
            .retain(|&cid| !self.clauses[cid as usize].deleted);
        if self.learnts.len() < 2 {
            return;
        }
        // Sort learnt ids worst-first: highest LBD, then lowest activity.
        // (Low LBD / high activity are the best clauses and must be kept.)
        self.learnts.sort_by(|&a, &b| {
            let (ca, cb) = (&self.clauses[a as usize], &self.clauses[b as usize]);
            cb.lbd.cmp(&ca.lbd).then(
                ca.activity
                    .partial_cmp(&cb.activity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        let half = self.learnts.len() / 2;
        let mut removed = 0usize;
        let mut kept: Vec<u32> = Vec::with_capacity(self.learnts.len() - half);
        for (i, &cid) in self.learnts.iter().enumerate() {
            // C6: never delete binary learnts (lits.len() <= 2) nor glue
            // clauses (LBD ≤ 2) — glue-2 clauses are the single most
            // valuable class and were previously unprotected.
            let deletable = {
                let c = &self.clauses[cid as usize];
                // A6: never delete a clause that is a live propagation reason.
                c.lits.len() > 2 && c.lbd > 2 && c.locked == 0
            };
            if i < half && deletable {
                self.clauses[cid as usize].deleted = true;
                removed += 1;
            } else {
                kept.push(cid);
            }
        }
        self.learnts = kept;
        self.stats.learnts_removed += removed;
        self.stats.reductions += 1;

        // Compact watchlists and occurrence lists around the tombstones.
        for wl in self.watchlist.iter_mut() {
            wl.retain(|cid| !self.clauses[*cid as usize].deleted);
        }
        for ol in self.occurrence.iter_mut() {
            ol.retain(|&cid| !self.clauses[cid as usize].deleted);
        }

        // Grow the high-water mark (C6/A9): scale with the *cumulative* conflict
        // count (per-solve `stats.conflicts` resets each solve) and clamp to the
        // configurable cap.
        let dynamic = MAX_LEARNTS_BASE + self.total_conflicts / 2;
        self.max_learnts = dynamic.min(self.max_learnts_cap);
    }

    // ── Search loop ──────────────────────────────────────────────────────────

    /// Pick the next branch variable: highest VSIDS activity among
    /// unassigned, non-eliminated variables; ties break to the lowest index.
    /// Returns 0 when every branchable variable is assigned (SAT condition).
    fn pick_branch_lit(&mut self) -> usize {
        // The heap holds candidates; assigned (or BVE-eliminated) entries that
        // surface are discarded here and re-inserted when they become
        // eligible again (on backtrack / resurrection).
        while let Some(var) = self.order.pop_max(&self.activity) {
            if self.assignment[var].is_none() && !self.eliminated[var] {
                return var;
            }
        }
        0
    }

    /// Insert a learnt clause into the database (watched, activity-tracked)
    /// and return its id.
    fn add_learnt_clause(&mut self, lits: Vec<i32>, lbd: u32) -> u32 {
        debug_assert!(!lits.is_empty());
        // On-the-fly subsumption (every clause is occurrence-indexed):
        //  • a live clause that is a subset of `lits` and contains the
        //    asserting literal makes the learnt redundant — reuse it as the
        //    propagation reason instead of inserting a weaker duplicate;
        //  • learnt clauses strictly subsumed by the new learnt are tombstoned.
        if lits.len() >= 2 && lits.len() <= SUBSUME_MAX_LEN {
            let idx = lit_index(lits[0]);
            if self.occurrence[idx].len() <= SUBSUME_SCAN_MAX {
                // Take the list out instead of cloning it (B2): the ids are
                // needed across `self.clauses` mutations below.
                let candidates = std::mem::take(&mut self.occurrence[idx]);
                for &cid in &candidates {
                    let c = &self.clauses[cid as usize];
                    if c.deleted || c.lits.len() > lits.len() {
                        continue;
                    }
                    if c.lits.iter().all(|l| lits_contains(&lits, *l)) {
                        self.stats.duplicates_removed += 1;
                        self.occurrence[idx] = candidates;
                        return cid;
                    }
                }
                for &cid in &candidates {
                    let redundant = {
                        let c = &self.clauses[cid as usize];
                        c.learnt
                            && !c.deleted
                            && c.lits.len() > lits.len()
                            && lits.iter().all(|l| lits_contains(&c.lits, *l))
                    };
                    if redundant {
                        self.clauses[cid as usize].deleted = true;
                    }
                }
                self.occurrence[idx] = candidates;
            }
        }
        let cid = self.clauses.len() as u32;
        let w0 = lits[0];
        let w1 = if lits.len() > 1 { lits[1] } else { lits[0] };
        self.clauses.push(Clause {
            w0,
            w1,
            sig: lits_signature(&lits),
            lits: crate::smallvec::SmallVec::from_vec(lits),
            learnt: true,
            lbd,
            activity: 0.0,
            deleted: false,
            locked: 0,
        });
        if self.drat.is_some() {
            let drat_lits = self.clauses[cid as usize].lits.to_vec();
            self.drat_add(&drat_lits);
        }
        if self.clauses[cid as usize].lits.len() >= 2 {
            self.attach(cid, w0, w1);
        } else {
            // Unit learnt: never moved and never used as a propagation reason
            // (the search enqueues it with `reason = None`). Registering the
            // same watch twice would leave two permanent, dead watchlist
            // entries that are only compacted if the clause is deleted.
            self.clauses[cid as usize].w0 = w0;
            self.clauses[cid as usize].w1 = w1;
        }
        for &lit in &self.clauses[cid as usize].lits {
            self.occurrence[lit_index(lit)].push(cid);
        }
        self.learnts.push(cid);
        self.stats.learnts_added += 1;
        cid
    }

    /// Perform one restart: drop all decisions (facts and phases survive).
    fn restart(&mut self) {
        self.cancel_until(0);
        self.conflicts_since_restart = 0;
        self.restart_seq += 1;
        // Advance the cached Luby budget once per restart instead of
        // recomputing it on every search iteration.
        self.restart_budget = self
            .restart_base
            .saturating_mul(luby_pow2(self.restart_seq));
        self.stats.restarts += 1;
        // C4: at decision level 0, drop clauses satisfied by a permanent fact
        // every few restarts. Long single solves previously never
        // re-inprocessed, so satisfied clauses lingered in the watchlists for
        // the whole run. Skipping locked clauses (A6) keeps every current
        // propagation reason alive; a clause with a true literal cannot
        // propagate anyway, and tombstoned watch entries are skipped lazily.
        if self.restart_seq % 8 == 0 {
            self.remove_satisfied_at_level0();
            // C7: exchange glue learnts with the other portfolio workers.
            self.share_and_import_learnts();
        }
    }

    /// C4: tombstone clauses satisfied by a level-0 fact. Must be called at
    /// decision level 0. Returns the number of clauses dropped.
    fn remove_satisfied_at_level0(&mut self) -> usize {
        debug_assert_eq!(self.decision_level(), 0, "remove_satisfied below level 0");
        let mut removed = 0usize;
        for cid in 0..self.clauses.len() {
            // Evaluate the predicate in a borrow that ends before the
            // mutation; `lit_value` reads the level-0 assignment.
            let satisfied = {
                let c = &self.clauses[cid];
                !c.deleted
                    && c.locked == 0
                    && c.lits.iter().any(|&l| self.lit_value(l) == Some(true))
            };
            if satisfied {
                self.tombstone_clause(cid as u32);
                removed += 1;
            }
        }
        removed
    }

    /// Cheap cooperative-cancellation check for the long inner loops
    /// (propagation passes, BVE, vivification) that `limits_exceeded` cannot
    /// reach: a worker wedged in one of those would otherwise delay the
    /// portfolio's decisive result by the whole pass.
    #[inline]
    fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Check the decision/conflict limits (every iteration) and the
    /// wall-clock limit (throttled). `true` → limits exhausted.
    fn limits_exceeded(&mut self) -> bool {
        if self.stats.decisions >= self.limits.max_decisions
            || self.stats.conflicts >= self.limits.max_conflicts
        {
            return true;
        }
        self.loop_tick += 1;
        if self.loop_tick.is_multiple_of(TIME_CHECK_INTERVAL) {
            if let Some(start) = self.solve_start
                && start.elapsed().as_micros() as u64 > self.limits.max_time_micros
            {
                return true;
            }
            if let Some(flag) = &self.cancel
                && flag.load(std::sync::atomic::Ordering::Relaxed)
            {
                return true;
            }
        }
        false
    }

    /// The CDCL search loop.
    ///
    /// Propagate → on conflict: analyze (1UIP), backjump, add the learnt
    /// clause; on no conflict: restart if the Luby budget expired, else
    /// decide the next assumption (levels 1..k) or branch on the best
    /// VSIDS variable. Termination is guarded by the resource limits
    /// (returning [`Outcome::Unknown`]).
    fn search(&mut self) -> Outcome {
        loop {
            if self.limits_exceeded() {
                return Outcome::Unknown;
            }

            match self.propagate() {
                PropOutcome::Limit => return Outcome::Unknown,
                PropOutcome::Conflict(cid) => {
                    self.stats.conflicts += 1;
                    self.total_conflicts += 1;
                    self.conflicts_since_restart += 1;

                    // Short-window conflict-interval EMA for the restart gate
                    // (C5), plus the lazy VSIDS decay.
                    let gap = self
                        .stats
                        .decisions
                        .saturating_sub(self.last_conflict_decisions);
                    self.last_conflict_decisions = self.stats.decisions;
                    self.conflict_ema = Some(match self.conflict_ema {
                        Some(prev) => 0.99 * prev + 0.01 * gap as f64,
                        None => gap as f64,
                    });

                    // Lazy VSIDS decay (A5). Both increments decay together so their relative
                    // scale is preserved, but each table rescales only with its own increment.
                    self.var_inc /= self.var_decay;
                    self.cla_inc /= self.var_decay;
                    if self.var_inc > ACTIVITY_RESCALE {
                        for a in self.activity.iter_mut() {
                            *a *= 1e-100;
                        }
                        self.var_inc *= 1e-100;
                    }

                    // Conflict at level 0: the formula is globally UNSAT.
                    if self.decision_level() == 0 {
                        self.ok = false;
                        return Outcome::Unsat { core: Vec::new() };
                    }

                    let (learnt, backjump, lbd) = self.analyze(cid);

                    // C5: track LBD trend (fast ≈ last ~32 conflicts, slow ≈
                    // last ~512) so the restart gate can see a degrading
                    // learnt database even while the conflict-gap EMA improves.
                    let l = lbd as f64;
                    self.lbd_fast = Some(match self.lbd_fast {
                        Some(p) => p + (l - p) / 32.0,
                        None => l,
                    });
                    self.lbd_slow = Some(match self.lbd_slow {
                        Some(p) => p + (l - p) / 512.0,
                        None => l,
                    });

                    if learnt.len() == 1 {
                        // Unit learnt: globally valid consequence — assert at
                        // level 0 and keep permanently.
                        self.cancel_until(0);
                        let lit = learnt[0];
                        if !self.enqueue(lit, None) {
                            self.ok = false;
                            return Outcome::Unsat { core: Vec::new() };
                        }
                        self.add_learnt_clause(vec![lit], 1);
                    } else {
                        // Capture the asserting literal *before* the clause is
                        // moved: `add_learnt_clause` may reuse an existing
                        // subsuming clause, whose `w0` is an arbitrary watch
                        // (possibly a falsified non-asserting literal) — using
                        // `w0` would then enqueue a false literal (debug
                        // assertion) or silently fail to assert (release).
                        let asserting = learnt[0];
                        self.cancel_until(backjump);
                        let cid = self.add_learnt_clause(learnt, lbd);
                        // Re-assert the implied literal at the backjump level
                        // with the clause as its reason (MiniSat).
                        if !self.enqueue(asserting, Some(cid)) {
                            // Only an already-false literal makes `enqueue`
                            // fail — a real level-0 contradiction.
                            self.ok = false;
                            return Outcome::Unsat { core: Vec::new() };
                        }
                    }

                    // Learnt database reduction against the high-water mark.
                    if self.learnts.len() >= self.max_learnts {
                        self.reduce_db();
                    }
                    continue;
                }
                PropOutcome::Ok => {}
            }

            // Restart when the Luby conflict budget expires — but defer while the
            // short-window conflict rate is still improving (C5): fixed restarts
            // demonstrably hurt runs making steady progress. Luby stays the
            // backstop via the 2× cap, so restarts can never stop entirely.
            if self.conflicts_since_restart >= self.restart_budget {
                let improving = matches!(
                    (self.conflict_ema, self.prev_conflict_ema),
                    (Some(cur), Some(prev)) if cur < prev
                );
                // C5: also restart when the recent LBD has clearly degraded
                // relative to the long-run trend, even if the conflict-gap EMA
                // still looks improving.
                let lbd_stalled = matches!(
                    (self.lbd_fast, self.lbd_slow),
                    (Some(fast), Some(slow)) if fast > slow * 1.1
                );
                if !improving
                    || lbd_stalled
                    || self.conflicts_since_restart >= self.restart_budget.saturating_mul(2)
                {
                    self.prev_conflict_ema = self.conflict_ema;
                    self.restart();
                    continue;
                }
            }

            // Assumption levels: decision level i (1-based) ↔ assumptions[i-1].
            let k = self.assumptions.len();
            if self.decision_level() < k {
                let p = self.assumptions[self.decision_level()];
                match self.lit_value(p) {
                    Some(true) => {
                        // Already satisfied — dummy level keeps the
                        // assumption-index ↔ level alignment.
                        self.new_decision_level();
                    }
                    Some(false) => {
                        // The assumption is falsified by the formula together
                        // with earlier assumptions — extract the core.
                        let conflict = self.analyze_final(-p);
                        let core = self.core_from_conflict(&conflict);
                        // B1: hand the buffer back for reuse.
                        self.analyze_final_conflict = conflict;
                        return Outcome::Unsat { core };
                    }
                    None => {
                        self.stats.decisions += 1;
                        self.new_decision_level();
                        let enqueued = self.enqueue(p, None);
                        debug_assert!(enqueued);
                    }
                }
                continue;
            }

            // Free decision on the saved phase.
            match self.pick_branch_lit() {
                0 => return Outcome::Sat,
                v => {
                    self.stats.decisions += 1;
                    let val = self.saved_phase[v];
                    self.new_decision_level();
                    let branch = if val { v as i32 } else { -(v as i32) };
                    let enqueued = self.enqueue(branch, None);
                    debug_assert!(enqueued);
                }
            }
        }
    }

    // ── Solving entry point ──────────────────────────────────────────────────

    /// Solve the current formula.
    pub fn solve(&mut self) -> SolveResult {
        let result = self.solve_assumptions(&[]);
        // C8: close the trail with the empty clause on UNSAT. The empty clause
        // is RUP at that point (the solver derived a level-0 conflict), so a
        // checker accepts the proof against the original CNF.
        if self.drat.is_some() && matches!(result, SolveResult::Unsat { .. }) {
            self.drat_add(&[]);
        }
        result
    }

    /// Solve under temporary assumptions.
    ///
    /// Assumptions are fixed literals asserted at decision levels 1..k; the
    /// search never branches below them. On `Unsat`, the returned
    /// [`SolveResult::Unsat`] carries the conflict **core**: the subset of
    /// assumption literals that jointly contradict the formula (with the
    /// input formula — learnt clauses are pure consequences of it, so the
    /// core is sound). On `Sat`, the model satisfies the *original* formula,
    /// including variables eliminated by inprocessing (reconstructed).
    ///
    /// Resource-limit exhaustion yields [`SolveResult::Unknown`], never a
    /// fabricated answer.
    pub fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult {
        if self.has_empty_clause || !self.ok {
            return SolveResult::Unsat { core: Vec::new() };
        }

        let start = Instant::now();
        self.solve_start = Some(start);
        // Per-solve stats reset; cumulative counters survive.
        self.stats.decisions = 0;
        self.stats.propagations = 0;
        self.stats.conflicts = 0;
        self.stats.backtracks = 0;
        self.conflicts_since_restart = 0;

        // Normalise assumptions: reject invalid literals, dedup, and detect
        // direct contradictions (core = the contradicting pair).
        let mut norm: Vec<i32> = Vec::new();
        for &a in assumptions {
            if a == 0 || a == i32::MIN {
                self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                self.solve_start = None;
                return SolveResult::Unknown;
            }
            self.ensure_var(lit_var(a));
            if norm.contains(&a) {
                continue;
            }
            if norm.contains(&-a) {
                self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                self.solve_start = None;
                return SolveResult::Unsat { core: vec![a, -a] };
            }
            norm.push(a);
        }

        // Reset search state to level 0 (facts re-asserted below; phases,
        // activity, clauses and learnts persist for warm incremental solves).
        self.cancel_until(0);

        // An eliminated variable mentioned in an assumption must be
        // resurrected first — its clauses were removed from the database.
        for &a in &norm {
            let v = lit_var(a);
            if self.eliminated[v] {
                self.resurrect(v);
            }
        }
        self.assumptions = norm;

        // Inprocessing between solves: bounded variable elimination. (For-
        // ward/backward subsumption already ran per add_clause.) This runs
        // BEFORE fact assertion: BVE may derive new unit facts, and all
        // freshly attached clauses (resolvents, resurrected clauses, and
        // user clauses added between solves) must be present before the
        // single propagation pass below examines the trail.
        // B3: skip the O(formula) inprocessing pass when the input formula has
        // not changed since the last solve (the documented incremental
        // workload — many queries against a stable formula).
        let dirty = self.formula_dirty;
        if self.preprocess && self.bve_enabled && dirty {
            self.bounded_variable_elimination();
            if !self.ok {
                self.assumptions.clear();
                self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                self.solve_start = None;
                return SolveResult::Unsat { core: Vec::new() };
            }
        }

        // Assert level-0 facts (including any BVE-derived unit resolvents —
        // a contradiction here means the formula is globally UNSAT). Iterate
        // by index rather than cloning the fact list on every query (B9#2 —
        // nothing in this loop mutates the vector).
        for i in 0..self.level0_units.len() {
            let f = self.level0_units[i];
            if !self.enqueue(f, None) {
                self.ok = false;
                self.assumptions.clear();
                self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                self.solve_start = None;
                return SolveResult::Unsat { core: Vec::new() };
            }
        }

        // Full re-propagation from the trail start: clauses attached since
        // the last propagation (user adds between solves, resurrected
        // clauses, BVE resolvents) may watch already-false literals and have
        // no pending trigger — re-examining every level-0 literal's watch
        // list against the current database restores the watch invariant.
        // Idempotent, and O(level-0 trail) once per solve.
        if dirty {
            self.qhead = 0;
            match self.propagate() {
                PropOutcome::Conflict(_) => {
                    self.ok = false;
                    self.assumptions.clear();
                    self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                    self.solve_start = None;
                    return SolveResult::Unsat { core: Vec::new() };
                }
                PropOutcome::Limit => {
                    self.assumptions.clear();
                    self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                    self.solve_start = None;
                    return SolveResult::Unknown;
                }
                PropOutcome::Ok => {}
            }
        }

        // Vivification: probe short original clauses and shorten those whose
        // assumption prefix is already contradictory. Strengthening moves
        // watches, so re-propagate the level-0 trail afterwards to restore the
        // watch invariant (and surface any conflict exposed by a prefix).
        if self.vivify() {
            self.qhead = 0;
            match self.propagate() {
                PropOutcome::Conflict(_) => {
                    self.ok = false;
                    self.assumptions.clear();
                    self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                    self.solve_start = None;
                    return SolveResult::Unsat { core: Vec::new() };
                }
                PropOutcome::Limit => {
                    self.assumptions.clear();
                    self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
                    self.solve_start = None;
                    return SolveResult::Unknown;
                }
                PropOutcome::Ok => {}
            }
        }

        // Vivification may now derive level-0 facts; a contradiction there is
        // a genuine UNSAT (the probe proved the formula entails both
        // polarities).
        if !self.ok {
            self.assumptions.clear();
            self.stats.elapsed_micros = start.elapsed().as_micros() as u64;
            self.solve_start = None;
            return SolveResult::Unsat { core: Vec::new() };
        }

        // The formula is now fully absorbed into the database and the
        // level-0 trail: nothing below needs to redo the O(formula) work
        // until a caller dirties the formula again (B3).
        self.formula_dirty = false;

        let outcome = self.search();

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.elapsed_micros = elapsed;
        self.solve_start = None;
        self.assumptions.clear();

        match outcome {
            Outcome::Sat => {
                let mut model: Vec<bool> = (1..self.vars.len())
                    .map(|v| self.assignment[v].unwrap_or(false))
                    .collect();
                // Reconstruct BVE-eliminated variables (reverse elimination
                // order) so the model satisfies the original formula.
                self.reconstruct_model(&mut model);
                debug_assert!(
                    self.verify_model(&model),
                    "internal error: Puto returned a model violating the formula; \
                     clauses={:?} facts={:?} elim={:?} model={model:?}",
                    self.clauses_snapshot(),
                    self.level0_units,
                    self.elim_order,
                );
                SolveResult::Sat(model)
            }
            Outcome::Unsat { core } => SolveResult::Unsat { core },
            Outcome::Unknown => SolveResult::Unknown,
        }
    }

    /// Snapshot of the original formula: the append-only input log ().
    ///
    /// Deliberately *not* the live clause database: after preprocessing that
    /// holds BVE resolvents, drops subsumed originals, and shortened
    /// (vivified) bodies — none of which the caller wrote. The log also makes
    /// the snapshot stable across solves, which is what the differential
    /// harness and DIMACS export rely on.
    pub fn clauses_snapshot(&self) -> Vec<Vec<i32>> {
        let mut out: Vec<Vec<i32>> = self.original_clauses.clone();
        // Canonical order (contract shared with `solvo::Solver`): clauses are
        // compared across backends by the differential harness, and the two
        // backends store originals/units/binaries in different internal
        // layouts. Sorting literals within each clause and the clause list
        // itself makes the snapshot backend-independent.
        for c in &mut out {
            c.sort_unstable();
        }
        out.sort_unstable();
        out
    }

    /// DIMACS CNF export of the current formula (1-based variables, as here).
    pub fn to_dimacs(&self) -> String {
        use swrite::{SWrite, swrite, swriteln};
        let clauses = self.clauses_snapshot();
        let mut out = String::with_capacity(24 + clauses.len() * 8);
        swriteln!(out, "p cnf {} {}", self.num_vars(), clauses.len());
        for c in &clauses {
            for l in c {
                swrite!(out, "{l} ");
            }
            out.push_str("0\n");
        }
        out
    }

    /// Debug dump of the inprocessing state (test harness only).
    ///
    /// Deliberately *not* `#[cfg(debug_assertions)]`: the differential test
    /// harness calls it from `cargo test --release`, and a method that only
    /// exists in debug builds makes release test builds fail to compile.
    pub(crate) fn debug_dump(&self) -> String {
        format!(
            "clauses={:?} facts={:?} assumptions={:?} elim={:?}",
            self.clauses_snapshot(),
            self.level0_units,
            self.assumptions,
            self.elim_order,
        )
    }

    // ── Bounded variable elimination (inprocessing) ──────────────────────────

    /// DP-style bounded variable elimination, run between solves.
    ///
    /// For each candidate variable `v` (not eliminated, no level-0 fact, not
    /// an assumption variable) with `pos × neg ≤ BVE_PRODUCT_LIMIT`, all
    /// clauses containing `v` are replaced by their pairwise resolvents —
    /// classical DP elimination, equisatisfiable with the original formula.
    /// Pure-polarity variables (one side empty) are eliminated without
    /// resolvents. Learnt clauses mentioning `v` are deleted: DP preserves
    /// entailment only for clauses *not mentioning* the eliminated variable.
    ///
    /// The eliminated variable's original clause bodies are logged in
    /// `elim_order` for model reconstruction. If a later `add_clause`
    /// mentions an eliminated variable, it is resurrected (its clauses
    /// re-added), which restores the original constraints soundly.
    /// Strengthen a live clause to the given (strictly shorter) literal slice,
    /// rewiring its watch entries. The caller guarantees `new_lits.len() >= 2`
    /// and that `new_lits` is a prefix of the old clause.
    fn strengthen_clause(&mut self, cid: u32, new_lits: &[i32]) {
        debug_assert!(new_lits.len() >= 2);
        let (old_w0, old_w1) = {
            let c = &self.clauses[cid as usize];
            (c.w0, c.w1)
        };
        // Drop the clause's entries under its old watches (each watched
        // literal has exactly one entry).
        for w in [old_w0, old_w1] {
            let idx = lit_index(w);
            self.watchlist[idx].retain(|c| *c != cid);
        }
        let (w0, w1) = (new_lits[0], new_lits[1]);
        {
            let c = &mut self.clauses[cid as usize];
            c.lits = crate::smallvec::SmallVec::from_slice(new_lits);
            c.sig = lits_signature(&c.lits);
        }
        // `attach` records the new watches and re-registers the clause.
        self.attach(cid, w0, w1);
    }

    /// Start journaling phase changes (B4). Only variables `enqueue` touches
    /// during the journal window are recorded, so `end_phase_journal` restores in
    /// O(touched) instead of cloning the whole `saved_phase` vector.
    fn begin_phase_journal(&mut self) {
        self.phase_journal.clear();
        if self.phase_journal_mark.len() < self.assignment.len() {
            self.phase_journal_mark.resize(self.assignment.len(), false);
        }
        self.phase_journaling = true;
    }

    /// Restore every phase recorded since `begin_phase_journal`.
    fn end_phase_journal(&mut self) {
        self.phase_journaling = false;
        for &(v, old) in &self.phase_journal {
            self.saved_phase[v] = old;
            self.phase_journal_mark[v] = false;
        }
        self.phase_journal.clear();
    }

    /// C3: self-subsuming resolution (SatELite's OBS rule).
    ///
    /// If `occurrence[¬l]` holds a clause `D` with `¬l ∈ D` and
    /// `D ∖ {¬l} ⊆ C ∖ {l}`, resolving `C` with `D` on `l` yields a clause
    /// contained in `C ∖ {l}`, so `C ∖ {l}` is implied and `l` can be dropped.
    /// Returns the index of a removable literal.
    ///
    /// Reuses the occurrence lists built for subsumption/vivification and the
    /// same `SUBSUME_SCAN_MAX` bound. Stale entries (clauses strengthened since
    /// the last `rebuild_occurrence`) are harmless: every candidate is
    /// re-validated with `lits_contains`, so a dead entry is skipped rather
    /// than trusted.
    fn self_subsume_literal(&self, cid: u32, lits: &[i32]) -> Option<usize> {
        for (i, &l) in lits.iter().enumerate() {
            let neg = -l;
            let occ = &self.occurrence[lit_index(neg)];
            if occ.len() > SUBSUME_SCAN_MAX {
                continue;
            }
            for &d in occ {
                if d == cid {
                    continue;
                }
                let cd = &self.clauses[d as usize];
                if cd.deleted {
                    continue;
                }
                if !lits_contains(&cd.lits, neg) {
                    continue;
                }
                // Every other literal of `D` must already be in `C ∖ {l}`.
                // `dl == l` is rejected: a tautological `D` would leave `l` in
                // the resolvent, which is not a subset of `C ∖ {l}`.
                let subset = cd
                    .lits
                    .iter()
                    .all(|&dl| dl == neg || (dl != l && lits_contains(lits, dl)));
                if subset {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Clause vivification (preprocessing).
    ///
    /// For each short original clause `l₁ ∨ … ∨ lₙ`, assume `¬l₁, ¬l₂, …`
    /// one at a time and unit-propagate. If the prefix `¬l₁ … ¬lᵢ` is already
    /// contradictory, then `l₁ ∨ … ∨ lᵢ` is a logical consequence of the
    /// current formula, so the clause can be replaced by that shorter prefix.
    /// This removes literals that later propagation proves redundant and is
    /// one of the strongest modern preprocessing techniques on structured
    /// problems.
    ///
    /// Returns `true` when at least one clause was shortened (the caller must
    /// then re-propagate to restore the watch invariant). Runs at decision
    /// level 0 with the level-0 trail fully propagated.
    fn vivify(&mut self) -> bool {
        if !self.preprocess || !self.needs_vivify {
            return false;
        }
        // A9: gate on live *originals* — `clauses.len()` counts learnts and
        // tombstones, so a churny long run permanently disabled vivification
        // even for a tiny original formula.
        if self.live_originals > VIVIFY_MAX_DB {
            self.needs_vivify = false;
            return false;
        }
        let candidates: Vec<u32> = self
            .clauses
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                !c.deleted && !c.learnt && (3..=VIVIFY_MAX_LEN).contains(&c.lits.len())
            })
            .map(|(i, _)| i as u32)
            .collect();

        // Vivification probes call `enqueue`, which writes `saved_phase`.
        // Those probe polarities are search-irrelevant and would otherwise
        // pollute the phase heuristic across solves; journal only the touched
        // variables instead of cloning the whole vector (B4).
        self.begin_phase_journal();

        let mut changed = false;
        for cid in candidates {
            // Vivification probes every short clause; a cancelled worker must
            // not finish the whole sweep before reporting (A6).
            if self.cancelled() {
                self.cancel_until(0);
                self.end_phase_journal();
                // A8: earlier iterations may already have strengthened /
                // tombstoned clauses; the occurrence lists must be rebuilt
                // before this early return or they describe dead literals.
                if changed {
                    self.rebuild_occurrence();
                }
                return changed;
            }
            let mut lits = {
                let c = &self.clauses[cid as usize];
                if c.deleted || c.learnt || c.lits.len() < 3 {
                    continue;
                }
                // Already satisfied by level-0 facts: probing is pointless.
                if c.lits.iter().any(|&l| self.lit_value(l) == Some(true)) {
                    continue;
                }
                c.lits.clone()
            };
            // C3: drop literals implied redundant by self-subsuming
            // resolution before the (more expensive) prefix probe. Keep at
            // least two literals so `strengthen_clause`'s invariant holds; a
            // two-literal clause that could still be reduced is left to the
            // probe, which handles the unit case explicitly.
            let mut obs_dropped = false;
            while lits.len() > 2 {
                match self.self_subsume_literal(cid, &lits) {
                    Some(i) => {
                        lits.remove(i);
                        obs_dropped = true;
                    }
                    None => break,
                }
            }
            if obs_dropped {
                self.strengthen_clause(cid, &lits);
                changed = true;
            }
            let n = lits.len();
            let mut keep_len = n;
            for (i, &l) in lits.iter().enumerate() {
                self.new_decision_level();
                if !self.enqueue(-l, None) {
                    // `l` is already implied true under the prefix — the
                    // prefix `lits[..=i]` is entailed.
                    keep_len = i + 1;
                    break;
                }
                match self.propagate() {
                    PropOutcome::Conflict(_) => {
                        keep_len = i + 1;
                        break;
                    }
                    PropOutcome::Limit => {
                        // Out of budget: abandon the pass, keep the flag so a
                        // later solve retries once limits allow.
                        self.cancel_until(0);
                        self.end_phase_journal();
                        // A8: same stale-occurrence hazard as the cancel arm.
                        if changed {
                            self.rebuild_occurrence();
                        }
                        return changed;
                    }
                    PropOutcome::Ok => {}
                }
            }
            self.cancel_until(0);
            if keep_len < n {
                if keep_len == 1 {
                    // The first probe already conflicted: the clause is
                    // entailed as a *unit* — the strongest inprocessing
                    // result there is. It was previously discarded (the
                    // clause stayed at full length, its strongest
                    // consequence lost). Assert it permanently and drop the
                    // clause (level-0 facts are never in the database).
                    //
                    // The fact must also be *enqueued*: this runs after the
                    // solve's fact-assertion loop, so recording it alone
                    // would leave it unpropagated (the following propagation
                    // pass only walks the trail).
                    if !self.add_fact(lits[0]) || !self.enqueue(lits[0], None) {
                        self.ok = false;
                    }
                    self.tombstone_clause(cid);
                    changed = true;
                } else {
                    self.strengthen_clause(cid, &lits[..keep_len]);
                    changed = true;
                }
            }
        }

        self.needs_vivify = false;
        if changed {
            // Occurrence lists still describe the pre-vivification literals.
            self.rebuild_occurrence();
        }
        self.end_phase_journal();
        changed
    }

    fn rebuild_occurrence(&mut self) {
        for list in self.occurrence.iter_mut() {
            list.clear();
        }
        for (cid, clause) in self.clauses.iter().enumerate() {
            if clause.deleted {
                continue;
            }
            for &lit in &clause.lits {
                self.occurrence[lit_index(lit)].push(cid as u32);
            }
        }
    }

    fn bounded_variable_elimination(&mut self) {
        // Compact the occurrence lists: subsumption and previous BVE runs
        // leave tombstones behind, which would otherwise be rescanned by
        // every elimination candidate. One O(database) pass per solve.
        self.rebuild_occurrence();
        // Variables eliminated in this pass (deferred learnt purge, B5).
        let mut eliminated_now: Vec<bool> = vec![false; self.vars.len()];
        // B2: visit the variables touched since the last sweep first, then the
        // remainder. On the documented incremental workload (many queries
        // against a mostly-stable formula) this front-loads the work that can
        // actually pay off.
        let mut order_vars: Vec<usize> = Vec::with_capacity(self.vars.len());
        order_vars.extend(
            self.touched_vars
                .iter()
                .copied()
                .filter(|&v| v >= 1 && v < self.vars.len()),
        );
        for v in 1..self.vars.len() {
            if !self.touched_flag.get(v).copied().unwrap_or(false) {
                order_vars.push(v);
            }
        }
        for &v in &self.touched_vars {
            if v < self.touched_flag.len() {
                self.touched_flag[v] = false;
            }
        }
        self.touched_vars.clear();
        for v in order_vars {
            if self.eliminated[v] || self.fact_vars[v] {
                continue;
            }
            // A BVE sweep is bounded by the whole variable set; poll the
            // cancel flag so a cancelled portfolio worker can return (A6).
            if v.is_multiple_of(64) && self.cancelled() {
                return;
            }
            if self.assumptions.iter().any(|&a| lit_var(a) == v) {
                continue;
            }
            // Dependency restore: an earlier elimination whose stored
            // clauses mention `v` makes the accumulated elim_order
            // dependency-inconsistent — reverse-order reconstruction would
            // evaluate that entry before `v`'s final value exists. Undo
            // (resurrect) every such elimination, transitively, before `v`
            // is eliminated.
            self.restore_eliminated_dependency(v);
            let pos_lit = v as i32;
            let neg_lit = -(v as i32);

            let mut pos_ids: Vec<u32> = Vec::new();
            let mut neg_ids: Vec<u32> = Vec::new();
            // B2: iterate by index instead of cloning each occurrence list.
            let pos_idx = lit_index(pos_lit);
            for i in 0..self.occurrence[pos_idx].len() {
                let cid = self.occurrence[pos_idx][i];
                if !self.clauses[cid as usize].deleted && !self.clauses[cid as usize].learnt {
                    pos_ids.push(cid);
                }
            }
            let neg_idx = lit_index(neg_lit);
            for i in 0..self.occurrence[neg_idx].len() {
                let cid = self.occurrence[neg_idx][i];
                if !self.clauses[cid as usize].deleted && !self.clauses[cid as usize].learnt {
                    neg_ids.push(cid);
                }
            }
            if pos_ids.len() * neg_ids.len() > BVE_PRODUCT_LIMIT {
                continue;
            }

            // Non-pure case: produce resolvents first (formula unchanged on
            // failure paths below).
            if !pos_ids.is_empty() && !neg_ids.is_empty() {
                let mut resolvents: Vec<Vec<i32>> = Vec::new();
                for &pid in &pos_ids {
                    for &nid in &neg_ids {
                        let (pl, nl) = {
                            let (pc, nc) =
                                (&self.clauses[pid as usize], &self.clauses[nid as usize]);
                            (pc.lits.clone(), nc.lits.clone())
                        };
                        // R = (P \ {v}) ∪ (N \ {¬v}); skip tautologies.
                        let mut r: Vec<i32> = Vec::with_capacity(pl.len() + nl.len() - 2);
                        let mut taut = false;
                        for side in [&pl, &nl] {
                            for &l in side {
                                if l == pos_lit || l == neg_lit {
                                    continue;
                                }
                                if lits_contains(&r, l) {
                                    continue;
                                }
                                if lits_contains(&r, -l) {
                                    taut = true;
                                    break;
                                }
                                r.push(l);
                            }
                            if taut {
                                break;
                            }
                        }
                        if taut {
                            continue;
                        }
                        debug_assert!(!r.is_empty(), "empty resolvent requires unit v-clauses");
                        resolvents.push(r);
                    }
                }
                // Dedup (A8#8): normalise each resolvent so duplicates from
                // different clause pairs compare equal, then drop them — DP
                // allows duplicates but they bloat the database and skew
                // learnt/activity accounting.
                for r in &mut resolvents {
                    r.sort_unstable();
                }
                resolvents.sort_unstable();
                resolvents.dedup();
                // Bound the resolvent *length* (A8#9): BVE_PRODUCT_LIMIT
                // bounds the count only, and two long clauses yield a
                // resolvent that would enter the database unchecked. Refuse
                // the whole elimination rather than dropping a resolvent —
                // dropping one weakens the formula and forfeits UNSAT.
                if resolvents.iter().any(|r| r.len() > BVE_MAX_RESOLVENT_LEN) {
                    continue;
                }
                for r in resolvents {
                    if r.len() == 1 {
                        // Unit resolvent — entailed by the formula; a
                        // permanent fact.
                        if !self.add_fact(r[0]) {
                            self.ok = false;
                            return;
                        }
                    } else {
                        let cid = self.clauses.len() as u32;
                        let (w0, w1) = (r[0], r[1]);
                        self.clauses.push(Clause {
                            w0,
                            w1,
                            sig: lits_signature(&r),
                            lits: crate::smallvec::SmallVec::from_vec(r),
                            learnt: false,
                            lbd: 0,
                            activity: 0.0,
                            deleted: false,
                            locked: 0,
                        });
                        self.attach(cid, w0, w1);
                        for &lit in &self.clauses[cid as usize].lits {
                            self.occurrence[lit_index(lit)].push(cid);
                        }
                        self.live_originals += 1;
                        self.needs_vivify = true;
                    }
                }
            }

            // Delete the v-clauses (tombstones).
            for cid in pos_ids.iter().chain(neg_ids.iter()) {
                self.tombstone_clause(*cid);
            }
            // Deferred learnt purge (B5): record `v` and drop every learnt
            // mentioning any eliminated variable in one pass after the loop.
            // Doing it per variable rescans the whole learnt database for
            // every eliminated variable — quadratic on learnt-rich instances.
            eliminated_now[v] = true;

            // Log the ORIGINAL clauses (v / ¬v still present — resurrection
            // re-adds them verbatim; model reconstruction strips v).
            let pos_clauses: Vec<Vec<i32>> = pos_ids
                .iter()
                .map(|&cid| self.clauses[cid as usize].lits.to_vec())
                .collect();
            let neg_clauses: Vec<Vec<i32>> = neg_ids
                .iter()
                .map(|&cid| self.clauses[cid as usize].lits.to_vec())
                .collect();
            self.eliminated[v] = true;
            self.note_elim_mentions(&pos_clauses, &neg_clauses, true);
            self.elim_order.push((v, pos_clauses, neg_clauses));
        }

        // One learnt purge for everything eliminated in this pass (B5).
        // Learnts mentioning an eliminated variable are no longer valid
        // consequences of the (reduced) database.
        self.learnts.retain(|&lid| {
            let mentions = self.clauses[lid as usize]
                .lits
                .iter()
                .any(|&l| eliminated_now.get(lit_var(l)).copied().unwrap_or(false));
            if mentions {
                self.clauses[lid as usize].deleted = true;
            }
            !mentions
        });
    }

    /// Adjust [`Solver::elim_mentions`] for one elimination entry: every
    /// variable its stored clauses reference gains (or loses) a mention.
    fn note_elim_mentions(&mut self, pos: &[Vec<i32>], neg: &[Vec<i32>], add: bool) {
        let mut vars: Vec<usize> = Vec::new();
        for clause in pos.iter().chain(neg.iter()) {
            for &l in clause {
                vars.push(lit_var(l));
            }
        }
        vars.sort_unstable();
        vars.dedup();
        for u in vars {
            if u >= self.elim_mentions.len() {
                self.elim_mentions.resize(u + 1, 0);
            }
            if add {
                self.elim_mentions[u] += 1;
            } else {
                self.elim_mentions[u] = self.elim_mentions[u].saturating_sub(1);
            }
        }
    }

    /// Resurrect an eliminated variable: re-add its original clauses (the
    /// resolvents can stay — they are consequences of the restored set).
    fn resurrect(&mut self, v: usize) {
        let Some(idx) = self.elim_order.iter().position(|&(ev, _, _)| ev == v) else {
            debug_assert!(self.eliminated[v], "resurrect of a non-eliminated var");
            return;
        };
        let (_, pos_clauses, neg_clauses) = self.elim_order.remove(idx);
        self.note_elim_mentions(&pos_clauses, &neg_clauses, false);
        self.eliminated[v] = false;
        // `v` is branchable again.
        self.order.insert(v, &self.activity);
        self.needs_vivify = true;
        self.formula_dirty = true;

        for clause in pos_clauses.into_iter().chain(neg_clauses) {
            debug_assert!(clause.len() >= 2);
            let cid = self.clauses.len() as u32;
            let (w0, w1) = (clause[0], clause[1]);
            self.clauses.push(Clause {
                w0,
                w1,
                sig: lits_signature(&clause),
                lits: crate::smallvec::SmallVec::from_vec(clause),
                learnt: false,
                lbd: 0,
                activity: 0.0,
                deleted: false,
                locked: 0,
            });
            self.attach(cid, w0, w1);
            for &lit in &self.clauses[cid as usize].lits {
                self.occurrence[lit_index(lit)].push(cid);
            }
            self.live_originals += 1;
        }
    }

    /// Undo every elimination whose stored clauses mention `v`, transitively.
    ///
    /// The reconstruction processes `elim_order` in reverse: an entry may
    /// only reference variables whose final values are already fixed at that
    /// point — i.e. variables eliminated *earlier* in the list. If a later
    /// BVE run eliminates a variable that an existing entry's clauses
    /// reference, the order would invert and the entry would be evaluated
    /// against a stale value. Resurrecting the referencing entries (and,
    /// recursively, any eliminated variable their restored clauses mention)
    /// restores a dependency-consistent order.
    fn restore_eliminated_dependency(&mut self, v: usize) {
        // O(1) fast path (B6): no logged entry references `v` — nothing to
        // undo. This runs for every BVE candidate, so the scan it replaces
        // was the dominant cost of an incremental query.
        if self.elim_mentions.get(v).copied().unwrap_or(0) == 0 {
            return;
        }
        while let Some(idx) = self.elim_order.iter().position(|(_, pos, neg)| {
            pos.iter()
                .chain(neg.iter())
                .any(|c| c.iter().any(|&l| lit_var(l) == v))
        }) {
            let (dv, pos_clauses, neg_clauses) = self.elim_order.remove(idx);
            self.note_elim_mentions(&pos_clauses, &neg_clauses, false);
            debug_assert!(dv != v);
            self.eliminated[dv] = false;
            self.order.insert(dv, &self.activity);

            let mut new_cids: Vec<u32> = Vec::new();
            for clause in pos_clauses.into_iter().chain(neg_clauses) {
                debug_assert!(clause.len() >= 2);
                let cid = self.clauses.len() as u32;
                let (w0, w1) = (clause[0], clause[1]);
                self.clauses.push(Clause {
                    w0,
                    w1,
                    sig: lits_signature(&clause),
                    lits: crate::smallvec::SmallVec::from_vec(clause),
                    learnt: false,
                    lbd: 0,
                    activity: 0.0,
                    deleted: false,
                    locked: 0,
                });
                self.attach(cid, w0, w1);
                for &lit in &self.clauses[cid as usize].lits {
                    self.occurrence[lit_index(lit)].push(cid);
                }
                self.live_originals += 1;
                new_cids.push(cid);
            }
            self.needs_vivify = true;

            // The restored clauses may mention OTHER eliminated variables —
            // those entries are now stale in the same way; recurse.
            let mut restored: Vec<usize> = new_cids
                .iter()
                .flat_map(|&cid| self.clauses[cid as usize].lits.iter().map(|&l| lit_var(l)))
                .filter(|&u| u != dv && self.eliminated[u])
                .collect();
            restored.sort_unstable();
            restored.dedup();
            for u in restored {
                self.restore_eliminated_dependency(u);
            }
        }
    }

    /// Reconstruct eliminated variables' model values in reverse elimination
    /// order. For variable `v`: if any v-negative original body is falsified
    /// by the current partial model, `v` must be false (the corresponding
    /// clause is then satisfied by ¬v, and the DP argument guarantees all
    /// v-positive bodies are satisfied); otherwise `v` is true.
    fn reconstruct_model(&self, model: &mut [bool]) {
        fn sat_body(model: &[bool], v: usize, body: &[i32]) -> bool {
            body.iter()
                .filter(|&&l| lit_var(l) != v) // strip v / ¬v — v is being decided
                .any(|&l| {
                    let val = model[lit_var(l) - 1];
                    if l > 0 { val } else { !val }
                })
        }
        for &(v, ref pos, ref neg) in self.elim_order.iter().rev() {
            let plus_unsat = pos.iter().any(|c| !sat_body(model, v, c));
            let minus_unsat = neg.iter().any(|c| !sat_body(model, v, c));
            debug_assert!(
                !(plus_unsat && minus_unsat),
                "BVE reconstruction: both polarities falsified (unsound elimination) \
                 v={v} plus={pos:?} minus={neg:?} model={model:?}"
            );
            model[v - 1] = !minus_unsat;
        }
    }

    /// Verify that `model` satisfies every live database clause, every level-0
    /// fact, and every *original* clause of a BVE-eliminated variable.
    ///
    /// The database alone is insufficient: after BVE it holds resolvents, and
    /// a model can satisfy those while violating the pre-elimination clauses.
    /// Checks `elim_order` against the **reconstructed** model so a
    /// reconstruction bug is caught instead of silently returning a wrong
    /// `Sat`. (Only invoked under `debug_assert!`.)
    fn verify_model(&self, model: &[bool]) -> bool {
        let val = |lit: i32| {
            let v = model[lit_var(lit) - 1];
            if lit > 0 { v } else { !v }
        };
        let db_ok = self
            .clauses
            .iter()
            .filter(|c| !c.deleted)
            .all(|c| c.lits.iter().any(|&l| val(l)))
            && self.level0_units.iter().all(|&l| val(l));
        if !db_ok {
            return false;
        }
        // Each logged entry holds the original clauses with `v` / `¬v` still
        // present; against the final reconstructed model every one of them
        // must be satisfied.
        self.elim_order.iter().all(|&(_, ref pos, ref neg)| {
            pos.iter()
                .chain(neg.iter())
                .all(|clause| clause.iter().any(|&l| val(l)))
        })
    }
}

/// Internal search outcome (before model/core assembly).
#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    Sat,
    Unsat { core: Vec<i32> },
    Unknown,
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
        Solver::clauses_snapshot(self)
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
        let mut s = Solver::new();
        let a = s.new_var();
        s.add_unit(a as i32);
        assert!(s.solve().is_sat());
    }

    #[test]
    fn test_trivial_unsat() {
        let mut s = Solver::new();
        let a = s.new_var();
        s.add_unit(a as i32);
        s.add_unit(-(a as i32));
        assert!(s.solve().is_unsat());
    }

    #[test]
    fn test_and_or_equiv() {
        let mut s = Solver::new();
        let a = s.new_var();
        let b = s.new_var();
        s.add_unit(a as i32);
        s.add_unit(b as i32);
        let r = s.solve();
        assert!(matches!(&r, SolveResult::Sat(m) if m[0] && m[1]));

        let mut s = Solver::new();
        let a = s.new_var();
        let b = s.new_var();
        s.add_clause(&[a as i32, b as i32]);
        assert!(s.solve().is_sat());

        // a ↔ b, a true → b true; conflicting ¬b → UNSAT.
        let mut s = Solver::new();
        let a = s.new_var();
        let b = s.new_var();
        s.add_equiv(a as i32, b as i32);
        s.add_unit(a as i32);
        s.add_unit(-(b as i32));
        assert!(s.solve().is_unsat());
    }

    #[test]
    fn test_exactly_one_and_amo() {
        let mut s = Solver::new();
        let lits: Vec<i32> = (0..10).map(|_| s.new_var() as i32).collect();
        s.add_exactly_one(&lits);
        s.add_unit(lits[3]);
        let r = s.solve();
        assert!(matches!(&r, SolveResult::Sat(m)
            if m[..10].iter().filter(|&&v| v).count() == 1 && m[3]));

        let mut s = Solver::new();
        let lits: Vec<i32> = (0..10).map(|_| s.new_var() as i32).collect();
        s.add_exactly_one(&lits);
        s.add_unit(lits[3]);
        s.add_unit(lits[7]);
        assert!(s.solve().is_unsat());
    }

    #[test]
    fn test_pigeonhole_3_2() {
        // 3 pigeons, 2 holes → UNSAT (exponentially hard for resolution,
        // but tiny; CDCL must still derive the empty clause).
        let mut s = Solver::new();
        let mut p = vec![vec![0i32; 2]; 3];
        for i in 0..3 {
            for j in 0..2 {
                p[i][j] = s.new_var() as i32;
            }
        }
        for i in 0..3 {
            s.add_clause(&p[i]);
        }
        for j in 0..2 {
            let col: Vec<i32> = (0..3).map(|i| p[i][j]).collect();
            s.add_at_most_one(&col);
        }
        assert!(s.solve().is_unsat());
    }

    #[test]
    fn test_binary_chain_propagation() {
        // a → b → c → d → e, ¬e: a must be false (fast propagation through
        // qhead + learnts).
        let mut s = Solver::new();
        let lits: Vec<i32> = (0..5).map(|_| s.new_var() as i32).collect();
        for w in lits.windows(2) {
            s.add_implies(w[0], w[1]);
        }
        s.add_unit(-(lits[4]));
        let r = s.solve();
        assert!(matches!(&r, SolveResult::Sat(m) if !m[0]));
    }

    #[test]
    fn test_binary_chain_bve_scale() {
        // Bench workload shape (n=50): BVE dissolves the pure-positive chain
        // vars; the reconstructed model must still satisfy every clause.
        for n in [10usize, 50, 200] {
            let mut s = Solver::new();
            let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
            for w in lits.windows(2) {
                s.add_implies(w[0], w[1]);
            }
            s.add_unit(-lits[n - 1]);
            let r = s.solve();
            assert!(r.is_sat(), "n={n}: chain + falsified tail must be SAT");
            if let SolveResult::Sat(m) = r {
                for w in lits.windows(2) {
                    assert!(
                        !m[w[0] as usize - 1] || m[w[1] as usize - 1],
                        "n={n}: implication violated by reconstructed model"
                    );
                }
                assert!(!m[n - 1], "n={n}: tail fact violated");
            }
        }
    }

    // ── Assumptions & conflict cores ─────────────────────────────────────────

    #[test]
    fn test_assumptions_basic() {
        let mut s = Solver::new();
        let a = s.new_var();
        let b = s.new_var();
        s.add_clause(&[a as i32, b as i32]);
        let r = s.solve_assumptions(&[-(a as i32)]);
        assert!(matches!(&r, SolveResult::Sat(m) if !m[0] && m[1]));
        // A second, different assumption reuses the warm state.
        let r = s.solve_assumptions(&[-(b as i32)]);
        assert!(matches!(&r, SolveResult::Sat(m) if m[0] && !m[1]));
    }

    #[test]
    fn test_assumption_conflict_direct() {
        // Assumption contradicts a level-0 fact → core = the assumption.
        let mut s = Solver::new();
        let a = s.new_var();
        s.add_unit(a as i32);
        let r = s.solve_assumptions(&[-(a as i32)]);
        assert_eq!(r.unsat_core(), Some(&[-(a as i32)][..]));
    }

    #[test]
    fn test_assumption_core_multi() {
        // a → c, b → ¬c: assuming a and b forces a core containing both.
        // Formula alone is SAT.
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        let c = s.new_var() as i32;
        s.add_implies(a, c);
        s.add_implies(b, -c);
        assert!(s.solve().is_sat());

        let r = s.solve_assumptions(&[a, b]);
        assert!(r.is_unsat());
        let core = r.unsat_core().expect("core must be present");
        assert_eq!(core.len(), 2, "core: {core:?}");
        assert!(core.contains(&a) && core.contains(&b), "core: {core:?}");

        // Soundness cross-check: the core assumptions genuinely contradict.
        let mut s2 = Solver::new();
        s2.add_implies(a, c);
        s2.add_implies(b, -c);
        assert!(s2.solve_assumptions(core).is_unsat());
    }

    #[test]
    fn test_assumption_core_deeper_implication() {
        // Core must reach assumptions several implication steps deep:
        // p1 → p2 → p3 → conflict-with-unit; assuming p1 alone is the core.
        let mut s = Solver::new();
        let p1 = s.new_var() as i32;
        let p2 = s.new_var() as i32;
        let p3 = s.new_var() as i32;
        let x = s.new_var() as i32;
        s.add_implies(p1, p2);
        s.add_implies(p2, p3);
        s.add_implies(p3, x);
        s.add_unit(-x); // x must be false
        let r = s.solve_assumptions(&[p1]);
        assert!(r.is_unsat());
        assert_eq!(r.unsat_core(), Some(&[p1][..]));
    }

    #[test]
    fn test_assumptions_survive_learnts() {
        // Learnts derived under assumptions are globally valid — a second
        // solve WITHOUT assumptions must not be corrupted by them.
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        s.add_clause(&[a, b]);
        assert!(s.solve_assumptions(&[-a]).is_sat());
        // Without assumptions both a and b are free.
        let r = s.solve();
        assert!(r.is_sat());
        if let SolveResult::Sat(m) = r {
            // (a ∨ b) must hold under the model.
            assert!(m[0] || m[1]);
        }
    }

    // ── Inprocessing: subsumption + BVE ──────────────────────────────────────

    #[test]
    fn test_subsumption_reduces_database() {
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        let c = s.new_var() as i32;
        s.add_clause(&[a, b, c]);
        let before = s.num_clauses();
        s.add_clause(&[a, b, c]); // exact duplicate — skipped
        s.add_clause(&[a, b]); // subsumes the first — first deleted
        let after = s.num_clauses();
        assert_eq!(
            after, 1,
            "duplicate + subsumed clause must collapse: {before} → {after}"
        );
        assert!(s.solve().is_sat());
    }

    #[test]
    fn test_bve_matches_no_preprocess_random() {
        // BVE must not change verdicts. Run the same random formulas with
        // preprocessing on and off.
        for seed in 0..40u64 {
            let num_vars = 8 + (seed as usize % 10);
            let clauses = crate::sat::tests::gen_random_3sat(seed + 1000, num_vars, num_vars * 4);
            let verdict = |pre: bool| {
                let mut s = Solver::new();
                s.set_preprocess(pre);
                for _ in 0..num_vars {
                    s.new_var();
                }
                for c in &clauses {
                    s.add_clause(c);
                }
                let r = s.solve();
                (r.is_sat(), r.is_unsat())
            };
            assert_eq!(
                verdict(true),
                verdict(false),
                "BVE changed the verdict, seed {seed}"
            );
        }
    }

    #[test]
    fn test_bve_model_satisfies_original_formula() {
        // The killer BVE property: after eliminating variables, the returned
        // model must satisfy the ORIGINAL formula (reconstruction check).
        for seed in 0..40u64 {
            let num_vars = 10 + (seed as usize % 12);
            let clauses = crate::sat::tests::gen_random_3sat(seed + 2000, num_vars, num_vars * 3);
            let mut s = Solver::new();
            for _ in 0..num_vars {
                s.new_var();
            }
            for c in &clauses {
                s.add_clause(c);
            }
            let r = s.solve();
            // Reference verdict from the recursive oracle.
            let oracle = crate::sat::tests::solve_recursive(&clauses, num_vars);
            assert_eq!(r.is_sat(), oracle.is_some(), "seed {seed}");
            if let SolveResult::Sat(model) = r {
                for (i, c) in clauses.iter().enumerate() {
                    assert!(
                        c.iter().any(|&l| {
                            let v = model[lit_var(l) - 1];
                            if l > 0 { v } else { !v }
                        }),
                        "seed {seed}: clause {i} {:?} violated by reconstructed model",
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_bve_resurrection_on_incremental_add() {
        // Solve (triggers BVE), then add a clause mentioning an eliminated
        // variable → resurrection must keep answers sound. Note (a∨b)∧(¬a∨b)
        // ENTAILS b — BVE derives that fact during solve 1.
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        s.add_clause(&[a, b]);
        s.add_clause(&[-a, b]);
        let r1 = s.solve();
        assert!(r1.is_sat());
        if let SolveResult::Sat(m) = &r1 {
            assert!(m[1], "b is entailed by the formula");
        }

        // Add ¬a — mentions the eliminated var a → resurrection re-adds
        // a's clauses; result: SAT with a=false, b=true.
        s.add_unit(-a);
        let r2 = s.solve();
        assert!(r2.is_sat(), "(a∨b) ∧ (¬a∨b) ∧ ¬a is SAT (a=false, b=true)");
        if let SolveResult::Sat(m) = r2 {
            assert!(!m[0] && m[1], "a=false, b=true; got {m:?}");
        }

        // And the contradiction with the entailed b → UNSAT.
        s.add_unit(-b);
        assert!(s.solve().is_unsat());
    }

    // ── Restarts, limits, stats ──────────────────────────────────────────────

    #[test]
    fn test_decision_limit_yields_unknown() {
        let mut s = Solver::with_limits(Limits {
            max_decisions: 5,
            ..Default::default()
        });
        let x = s.new_var();
        let y = s.new_var();
        s.add_clause(&[x as i32, y as i32]);
        s.add_clause(&[-(x as i32), y as i32]);
        s.add_clause(&[x as i32, -(y as i32)]);
        s.add_clause(&[-(x as i32), -(y as i32)]);
        let r = s.solve();
        assert!(r.is_unsat() || r.is_unknown(), "got {r:?}");
    }

    #[test]
    fn test_propagation_limit_yields_unknown_not_unsat() {
        // Preprocessing disabled: BVE would dissolve the chain via resolution
        // (with zero propagations) and legitimately return Sat.
        let mut s = Solver::with_limits(Limits {
            max_propagations: 3,
            ..Default::default()
        });
        s.set_preprocess(false);
        let lits: Vec<i32> = (0..40).map(|_| s.new_var() as i32).collect();
        for w in lits.windows(2) {
            s.add_implies(w[0], w[1]);
        }
        s.add_unit(lits[0]);
        let r = s.solve();
        assert!(
            r.is_unknown(),
            "propagation limit must be Unknown, got {r:?}"
        );
    }

    #[test]
    fn test_incremental_warm_state() {
        // Solve → add clause → solve: learnts, phases and activity persist,
        // answers stay correct.
        let mut s = Solver::new();
        let lits: Vec<i32> = (0..6).map(|_| s.new_var() as i32).collect();
        s.add_clause(&[lits[0], lits[1]]);
        assert!(s.solve().is_sat());
        s.add_implies(lits[0], lits[2]);
        s.add_implies(lits[2], -(lits[0]));
        let r = s.solve();
        assert!(r.is_sat());
        if let SolveResult::Sat(m) = r {
            assert!(!m[0], "a must be false: a → c → ¬a");
        }
        s.add_exactly_one(&[lits[3], lits[4], lits[5]]);
        assert!(s.solve().is_sat());
    }

    #[test]
    fn test_stats_counters_populated() {
        // Pigeonhole 3→2 forces conflicts → learnts, restart machinery.
        // Preprocessing is disabled: BVE legitimately dissolves this tiny
        // instance (deriving the contradiction at level 0) without any
        // conflict-driven search.
        let mut s = Solver::new();
        s.set_preprocess(false);
        let mut p = vec![vec![0i32; 2]; 3];
        for i in 0..3 {
            for j in 0..2 {
                p[i][j] = s.new_var() as i32;
            }
        }
        for i in 0..3 {
            s.add_clause(&p[i]);
        }
        for j in 0..2 {
            let col: Vec<i32> = (0..3).map(|i| p[i][j]).collect();
            s.add_at_most_one(&col);
        }
        assert!(s.solve().is_unsat());
        assert!(s.stats.conflicts > 0, "expected conflicts");
        assert!(s.stats.learnts_added > 0, "expected learnt clauses");
        assert!(s.stats.decisions + s.stats.propagations > 0);
    }

    #[test]
    fn test_global_unsat_persists() {
        let mut s = Solver::new();
        let a = s.new_var();
        s.add_unit(a as i32);
        s.add_unit(-(a as i32));
        assert!(s.solve().is_unsat());
        // Once globally UNSAT (level-0 conflict), later solves stay UNSAT
        // even after adding more clauses.
        let b = s.new_var();
        s.add_unit(b as i32);
        assert!(s.solve().is_unsat());
    }

    #[test]
    fn test_on_the_fly_learnt_subsumption() {
        let mut s = Solver::new();
        s.ensure_var(5);
        // Insert learnt (1 ∨ 2 ∨ 3).
        let base = s.add_learnt_clause(vec![1, 2, 3], 3);
        assert_eq!(base, 0);

        // A superset containing the asserting literal is subsumed: reuse the
        // existing clause as the reason and do not insert a weaker duplicate.
        let before = s.clauses.len();
        let reused = s.add_learnt_clause(vec![1, 2, 3, 4], 4);
        assert_eq!(reused, base);
        assert_eq!(
            s.clauses.len(),
            before,
            "subsumed learnt must not be inserted"
        );

        // A strictly stronger learnt tombstones the weaker one it subsumes.
        let stronger = s.add_learnt_clause(vec![1, 2], 2);
        assert_eq!(stronger, before as u32);
        assert!(
            s.clauses[base as usize].deleted,
            "subsumed learnt must be tombstoned"
        );
    }

    #[test]
    fn test_vivification_shortens_clause() {
        // Isolate vivification from BVE: with units ¬3, ¬4, probing the
        // clause (1∨2∨3∨4) derives a conflict after assuming ¬1 and ¬2, so
        // it must be strengthened to (1∨2).
        let mut s = Solver::new();
        s.set_bve(false);
        s.ensure_var(4);
        s.add_unit(-3);
        s.add_unit(-4);
        s.add_clause(&[1, 2, 3, 4]);
        assert!(s.solve().is_sat());

        let live: Vec<Vec<i32>> = s
            .clauses
            .iter()
            .filter(|c| !c.deleted && !c.learnt)
            .map(|c| c.lits.to_vec())
            .chain(s.level0_units.iter().map(|&l| vec![l]))
            .collect();
        let mut vivified: Vec<i32> = live
            .iter()
            .find(|c| c.len() == 2 && c.contains(&1) && c.contains(&2))
            .cloned()
            .unwrap_or_default();
        vivified.sort_unstable();
        assert_eq!(vivified, vec![1, 2], "expected vivified (1 ∨ 2): {live:?}");
        assert!(
            !live.iter().any(|c| c.len() == 4),
            "original clause should be shortened: {live:?}"
        );
        // : the *snapshot* still reports the caller's original formula —
        // vivification shortens the live database, not the audit artifact.
        let snapshot = s.clauses_snapshot();
        assert!(
            snapshot.iter().any(|c| c == &vec![1, 2, 3, 4]),
            "snapshot must preserve the original clause: {snapshot:?}"
        );
    }

    // ── Regression tests ───────────────────────────────────────────────────

    #[test]
    fn test_luby_sequence_golden() {
        // E1: MiniSat's luby(2, x) = 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, …
        // The missing `seq -= 1` produced 1, 2, 2, 4, 4, 4, 4, 8, … instead.
        let expected = [1usize, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1];
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(luby_pow2(i), want, "luby_pow2({i})");
        }
    }

    #[test]
    fn test_learnt_reuse_watch_differs_from_asserting_literal() {
        // E2/: when `add_learnt_clause` reuses a subsuming clause, that
        // clause's `w0` need not be the asserting literal. The search must
        // enqueue the captured asserting literal, never `clauses[cid].w0`.
        let mut s = Solver::new();
        s.set_preprocess(false);
        s.ensure_var(4);
        s.add_clause(&[2, 3]);
        let cid = 0u32;
        // Force the reused clause's watches so w0 != the asserting literal 2.
        s.clauses[cid as usize].w0 = 3;
        s.clauses[cid as usize].w1 = 2;

        let reused = s.add_learnt_clause(vec![2, 3, 4], 3);
        assert_eq!(
            reused, cid,
            "the learnt must be subsumed by the existing clause"
        );
        assert_eq!(
            s.clauses[reused as usize].w0, 3,
            "precondition: the reused clause's w0 is not the asserting literal"
        );
        // The asserting literal is 2; enqueuing it must succeed.
        assert!(s.enqueue(2, Some(reused)), "asserting literal must enqueue");
        assert_eq!(s.lit_value(2), Some(true));
    }

    #[test]
    fn test_warm_level0_resolve_needs_no_conflicts() {
        // A fact-only formula re-solved must not re-derive anything.
        let mut s = Solver::new();
        s.ensure_var(3);
        s.add_unit(1);
        s.add_unit(-2);
        assert!(s.solve().is_sat());
        assert!(s.solve().is_sat(), "second solve must remain SAT");
        assert_eq!(
            s.stats.conflicts, 0,
            "re-solving a fact-only formula must not re-derive facts"
        );
    }

    #[test]
    fn test_cancel_flag_is_honoured_quickly() {
        // E6/A6: with the cooperative-cancel flag already raised, a solve must
        // return `Unknown` promptly — even though the flag is only polled by
        // the throttled limit check — instead of running the whole search.
        let mut s = Solver::new();
        s.set_preprocess(false);
        s.ensure_var(400);
        // A dense 3-SAT-ish core that would otherwise take many conflicts.
        for i in 1..=400i32 {
            for j in 1..=3 {
                let a = ((i * 7 + j * 13) % 400) + 1;
                let b = ((i * 11 + j * 17) % 400) + 1;
                s.add_clause(&[i, a, -b]);
            }
        }
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        s.set_cancel_flag(std::sync::Arc::clone(&flag));
        let start = std::time::Instant::now();
        let result = s.solve();
        assert!(
            matches!(result, SolveResult::Unknown),
            "a pre-cancelled solve must be Unknown, got {result:?}"
        );
        assert!(
            start.elapsed().as_secs() < 5,
            "cancellation must take effect promptly (took {:?})",
            start.elapsed()
        );
    }

    #[test]
    fn test_bve_assumption_reconstruction_is_model_sound() {
        // Assumptions after BVE exercise resurrection and
        // `restore_eliminated_dependency`; the model must still satisfy the
        // *original* clauses.
        let clauses: Vec<Vec<i32>> = vec![
            vec![1, 2],
            vec![-1, 3],
            vec![2, 3, 4],
            vec![-2, 4],
            vec![-3, -4, 5],
            vec![5, 6],
            vec![-5, 6],
        ];
        let mut s = Solver::new();
        s.ensure_var(6);
        for c in &clauses {
            s.add_clause(c);
        }
        let first = s.solve();
        assert!(first.is_sat(), "formula is satisfiable");
        // Assumptions on variables that inprocessing may have eliminated.
        for a in [4i32, -4, 5, -5, 6] {
            match s.solve_assumptions(&[a]) {
                SolveResult::Sat(model) => {
                    for c in &clauses {
                        let sat = c.iter().any(|&l| {
                            let v = l.unsigned_abs() as usize;
                            if l > 0 { model[v - 1] } else { !model[v - 1] }
                        });
                        assert!(
                            sat,
                            "model violates original clause {c:?} under assumption {a}"
                        );
                    }
                }
                SolveResult::Unsat { .. } => {} // acceptable for some assumptions
                SolveResult::Unknown => panic!("unexpected Unknown under assumption {a}"),
            }
        }
    }

    /// A6: a clause currently used as a propagation reason must never be
    /// deleted by `reduce_db` — deleting a reason degrades conflict cores,
    /// which are this solver's headline feature.
    #[test]
    fn test_reduce_db_protects_reason_clauses() {
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        let c = s.new_var() as i32;
        let locked = s.add_learnt_clause(vec![a, b, c], 5);
        let other = s.add_learnt_clause(vec![-a, -b, -c], 5);
        s.learnts = vec![locked, other];
        s.new_decision_level();
        assert!(s.enqueue(a, Some(locked)));
        s.reduce_db();
        assert!(
            !s.clauses[locked as usize].deleted,
            "a locked reason clause was deleted by reduce_db"
        );
    }

    /// A5: rescaling the learnt-clause table must not touch the variable
    /// increment (doing so froze VSIDS for thousands of conflicts).
    #[test]
    fn test_clause_rescale_preserves_var_inc() {
        let mut s = Solver::new();
        let a = s.new_var() as i32;
        let b = s.new_var() as i32;
        let cid = s.add_learnt_clause(vec![a, b], 5);
        s.learnts = vec![cid];
        s.clauses[cid as usize].activity = ACTIVITY_RESCALE;
        s.cla_inc = ACTIVITY_RESCALE;
        let var_inc = s.var_inc;
        s.bump_clause_activity(cid);
        assert_eq!(
            s.var_inc, var_inc,
            "clause rescale must leave the variable increment untouched"
        );
        assert!(
            s.cla_inc < ACTIVITY_RESCALE,
            "clause increment must rescale"
        );
    }

    /// Vivification must leave the occurrence lists consistent — every
    /// entry must reference a live clause that actually contains the literal.
    /// A cancel/budget early return that skipped `rebuild_occurrence` left
    /// stale entries (dead literals) for later subsumption scans.
    #[test]
    fn test_vivification_keeps_occurrence_lists_consistent() {
        let mut s = Solver::new();
        s.set_bve(false);
        s.ensure_var(4);
        s.add_unit(-3);
        s.add_unit(-4);
        s.add_clause(&[1, 2, 3, 4]);
        assert!(s.solve().is_sat());
        for (idx, list) in s.occurrence.iter().enumerate() {
            for &cid in list {
                let c = &s.clauses[cid as usize];
                assert!(!c.deleted, "occurrence list holds a deleted clause");
                assert!(
                    c.lits.iter().any(|&l| lit_index(l) == idx),
                    "occurrence list holds a clause lacking the literal"
                );
            }
        }
    }

    /// C3: self-subsuming resolution drops a literal whose negation occurs in
    /// a clause whose remainder is already a subset of the other literals.
    #[test]
    fn test_self_subsuming_resolution_strengthens_clause() {
        let mut s = Solver::new();
        s.set_bve(false);
        s.ensure_var(3);
        // C = (1 ∨ 2 ∨ 3); D = (¬1 ∨ 2). Resolving on 1 gives (2 ∨ 3), which
        // is C ∖ {1}, so OBS removes 1 from C.
        s.add_clause(&[1, 2, 3]);
        s.add_clause(&[-1, 2]);
        s.rebuild_occurrence();
        assert!(s.vivify(), "OBS did not strengthen the clause");
        let live: Vec<Vec<i32>> = s
            .clauses
            .iter()
            .filter(|c| !c.deleted && !c.learnt)
            .map(|c| c.lits.to_vec())
            .collect();
        assert!(
            live.iter()
                .any(|c| c.len() == 2 && c.contains(&2) && c.contains(&3)),
            "expected (2 ∨ 3) after OBS, got {live:?}"
        );
    }

    /// C8: the DRAT trail records learnt additions and closes with the empty
    /// clause on UNSAT (with preprocessing off, so every addition is RUP).
    #[test]
    fn test_drat_trail_records_learnts_and_empty_clause() {
        let mut s = Solver::new();
        s.enable_drat();
        // Unsatisfiable 2-var formula that requires branching to refute.
        s.add_clause(&[1, 2]);
        s.add_clause(&[-1, 2]);
        s.add_clause(&[1, -2]);
        s.add_clause(&[-1, -2]);
        assert!(matches!(s.solve(), SolveResult::Unsat { .. }));
        let proof = s.drat_proof().expect("drat enabled");
        assert!(
            proof.ends_with("0\n"),
            "proof must end with the empty clause: {proof:?}"
        );
        // Every non-empty line is a well-formed addition line ending in `0`.
        for line in proof.lines() {
            assert!(
                line == "0" || line.ends_with(" 0"),
                "malformed DRAT line: {line:?}"
            );
        }
        // The trail is not merely the closing empty clause: the solver learned
        // at least one clause while refuting the formula.
        assert!(
            proof.lines().count() > 1,
            "expected learnt additions before the empty clause: {proof:?}"
        );
    }
}
