//! SAT/SMT solver family for Posita.
//!
//! - **Solvo** (Latin: "I solve") — a DPLL SAT solver with watched literals,
//!   VSIDS-lite activity heuristic, clause normalisation, and incremental
//!   assumption-based solving. Handles small-to-medium formulas efficiently.
//! - **Puto** (Latin: "I think") — a CDCL SAT solver with 1UIP clause
//!   learning, backjumping, Luby restarts, learnt-database management
//!   (LBD scoring + reduction), assumption conflict cores, and inprocessing
//!   (subsumption + bounded variable elimination). Orders of magnitude
//!   faster than Solvo on hard instances; the default choice for anything
//!   beyond trivial `@cfg` checks.
//!
//! Future siblings:
//! - **Aptus** (Latin: "suitable") — a theory solver for SMT / configuration
//!   semantics (target triples, feature dependencies, ABI constraints).
//!
//! # Shared encodings
//!
//! The cardinality encodings below are generic over [`SatSolver`] so both
//! solver backends (and future ones) share one audited implementation:
//! pairwise / sequential-counter / commander at-most-one, and a totalizer
//! at-most-k. Auxiliary variables are declared through the solver's
//! `new_var` and start at zero VSIDS activity — they are only promoted by
//! participation in conflict analysis.

pub mod puto;
pub mod solvo;

// Explicit re-exports: glob-re-exporting both backends made `sat::Solver`
// ambiguous for any consumer importing both.
pub use solvo::{Limits, SolveResult, SolverStats};

// ─── Variable ordering (VSIDS priority queue) ────────────────────────────────

/// Indexed binary max-heap over variable indices, keyed by VSIDS activity.
///
/// Both solver backends previously rescanned every variable on each decision
/// to find the highest-activity unassigned one — O(n) per decision, i.e.
/// O(n²) per solve. This queue makes selection O(log n).
///
/// It follows MiniSat's contract: the heap holds *candidate* variables. At
/// selection time an assigned (or Puto-eliminated) variable that surfaces is
/// popped and discarded; the caller re-inserts it when it becomes eligible
/// again (on backtrack). Activities only ever increase between inserts, so a
/// bumped variable just percolates upward. Ties break to the lower variable
/// index to keep runs deterministic.
///
/// `pos[var]` is the heap slot of `var`, or [`VarOrder::ABSENT`].
#[derive(Debug)]
pub(crate) struct VarOrder {
    heap: Vec<usize>,
    pos: Vec<usize>,
}

impl VarOrder {
    const ABSENT: usize = usize::MAX;

    pub(crate) fn new() -> Self {
        // Index 0 is an unused placeholder (variables are 1-based).
        Self {
            heap: Vec::new(),
            pos: vec![Self::ABSENT],
        }
    }

    /// Drop every entry and rebuild for `1..=max_var` (all assumed eligible).
    pub(crate) fn reset(&mut self, max_var: usize, activity: &[f64]) {
        self.heap.clear();
        self.pos.clear();
        self.pos.resize(max_var + 1, Self::ABSENT);
        for var in 1..=max_var {
            self.pos[var] = self.heap.len();
            self.heap.push(var);
        }
        // Heapify bottom-up.
        for i in (0..self.heap.len() / 2).rev() {
            self.sift_down(i, activity);
        }
    }

    #[inline]
    fn ensure(&mut self, var: usize) {
        if self.pos.len() <= var {
            self.pos.resize(var + 1, Self::ABSENT);
        }
    }

    #[inline]
    fn contains(&self, var: usize) -> bool {
        self.pos.get(var).copied().unwrap_or(Self::ABSENT) != Self::ABSENT
    }

    /// Insert `var` if absent; otherwise restore heap order at its position
    /// (activities may have changed while it was present).
    pub(crate) fn insert(&mut self, var: usize, activity: &[f64]) {
        self.ensure(var);
        if self.contains(var) {
            let i = self.pos[var];
            self.sift_up(i, activity);
            return;
        }
        self.pos[var] = self.heap.len();
        self.heap.push(var);
        let i = self.heap.len() - 1;
        self.sift_up(i, activity);
    }

    /// Add a newly declared variable (always absent beforehand).
    pub(crate) fn push_new(&mut self, var: usize, activity: &[f64]) {
        self.ensure(var);
        if self.contains(var) {
            return;
        }
        self.pos[var] = self.heap.len();
        self.heap.push(var);
        let i = self.heap.len() - 1;
        self.sift_up(i, activity);
    }

    /// Percolate `var` upward after its activity increased.
    pub(crate) fn bumped(&mut self, var: usize, activity: &[f64]) {
        if let Some(&p) = self.pos.get(var)
            && p != Self::ABSENT
        {
            self.sift_up(p, activity);
        }
    }

    /// Pop the highest-activity entry, or `None` when empty.
    pub(crate) fn pop_max(&mut self, activity: &[f64]) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        Some(self.remove_at(0, activity))
    }

    fn remove_at(&mut self, i: usize, activity: &[f64]) -> usize {
        let removed = self.heap[i];
        self.pos[removed] = Self::ABSENT;
        let last = self.heap.pop().unwrap();
        if i < self.heap.len() {
            self.heap[i] = last;
            self.pos[last] = i;
            // Either `last` belongs above (sift up) or below (sift down).
            if i > 0 && Self::better(self.heap[i], self.heap[(i - 1) / 2], activity) {
                self.sift_up(i, activity);
            } else {
                self.sift_down(i, activity);
            }
        }
        removed
    }

    /// True when `a` ranks above `b`: higher activity, lower index on ties.
    #[inline]
    fn better(a: usize, b: usize, activity: &[f64]) -> bool {
        match activity[a].partial_cmp(&activity[b]) {
            Some(std::cmp::Ordering::Greater) => true,
            Some(std::cmp::Ordering::Equal) => a < b,
            _ => false,
        }
    }

    fn sift_up(&mut self, mut i: usize, activity: &[f64]) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if Self::better(self.heap[i], self.heap[parent], activity) {
                self.heap.swap(i, parent);
                self.pos[self.heap[i]] = i;
                self.pos[self.heap[parent]] = parent;
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize, activity: &[f64]) {
        let len = self.heap.len();
        loop {
            let left = 2 * i + 1;
            if left >= len {
                break;
            }
            let right = left + 1;
            let mut best = i;
            if Self::better(self.heap[left], self.heap[best], activity) {
                best = left;
            }
            if right < len && Self::better(self.heap[right], self.heap[best], activity) {
                best = right;
            }
            if best == i {
                break;
            }
            self.heap.swap(i, best);
            self.pos[self.heap[i]] = i;
            self.pos[self.heap[best]] = best;
            i = best;
        }
    }
}

// ─── Parallel portfolio ─────────────────────────────────────────────────────

/// One worker's search strategy in [`solve_portfolio`].
///
/// Workers share the input formula but differ in preprocessing, restart
/// pacing, VSIDS decay and initial polarity, so they explore different search
/// trajectories. The first worker to produce a conclusive answer wins.
#[derive(Clone, Copy, Debug)]
pub struct PortfolioConfig {
    /// Enable BVE / subsumption / vivification preprocessing.
    pub preprocess: bool,
    /// Luby restart base interval.
    pub restart_base: usize,
    /// Per-conflict VSIDS decay factor.
    pub var_decay: f64,
    /// Initial polarity for free decisions.
    pub default_phase: bool,
}

impl PortfolioConfig {
    /// The default CDCL configuration (used for single-core runs).
    pub const DEFAULT: PortfolioConfig = PortfolioConfig {
        preprocess: true,
        restart_base: 512,
        var_decay: 0.95,
        default_phase: false,
    };
}

/// Curated portfolio: classic CDCL, inprocessing-free, aggressive short
/// restarts with eager polarity, and slow restarts. Diversity matters more
/// than any single "best" configuration.
pub const PORTFOLIO: [PortfolioConfig; 4] = [
    PortfolioConfig::DEFAULT,
    PortfolioConfig {
        preprocess: false,
        restart_base: 512,
        var_decay: 0.95,
        default_phase: false,
    },
    PortfolioConfig {
        preprocess: true,
        restart_base: 128,
        var_decay: 0.90,
        default_phase: true,
    },
    PortfolioConfig {
        preprocess: true,
        restart_base: 2048,
        var_decay: 0.99,
        default_phase: false,
    },
];

/// Resource limits for a portfolio worker: Puto's defaults, not Solvo's.
///
/// `Limits::default()` is defined by [`solvo`] (100k decisions / 200k
/// conflicts / 1M propagations) because Solvo is the learning-free DPLL and
/// the smaller budget keeps it responsive. Puto, the CDCL workhorse, ships a
/// 5× larger budget in its own `new()`. A portfolio worker is a Puto
/// instance, so handing it Solvo's default silently makes every worker
/// *weaker* than a plain sequential `puto.solve()` on hard instances — the
/// exact inverse of the portfolio's purpose. This mirrors `puto::Solver::new`.
pub fn portfolio_default_limits() -> Limits {
    Limits {
        max_decisions: 500_000,
        max_conflicts: 500_000,
        max_propagations: 5_000_000,
        max_time_micros: 5_000_000,
    }
}

/// Run one portfolio worker on a fresh `Puto` instance.
fn run_portfolio_worker(
    cfg: &PortfolioConfig,
    num_vars: usize,
    clauses: &[Vec<i32>],
    limits: &Limits,
    cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
    // C7: shared glue-learnt slab, `None` for a single-worker run.
    learnt_share: Option<&std::sync::Arc<std::sync::Mutex<Vec<Vec<i32>>>>>,
) -> SolveResult {
    let mut solver = puto::Solver::with_limits(limits.clone());
    solver.set_preprocess(cfg.preprocess);
    solver.set_restart_base(cfg.restart_base);
    solver.set_var_decay(cfg.var_decay);
    solver.set_default_phase(cfg.default_phase);
    if let Some(flag) = cancel {
        solver.set_cancel_flag(std::sync::Arc::clone(flag));
    }
    if let Some(share) = learnt_share {
        solver.set_learnt_share(std::sync::Arc::clone(share));
    }
    if num_vars > 0 {
        solver.ensure_var(num_vars);
    }
    for clause in clauses {
        solver.add_clause(clause);
    }
    solver.solve()
}

/// Solve a CNF with a parallel portfolio of differently configured CDCL
/// solvers.
///
/// Each worker gets its own copy of the formula and the same resource
/// `limits`. The first conclusive (`Sat`/`Unsat`) answer is returned; the
/// remaining workers are cancelled cooperatively and joined before this
/// function returns. On a single core (or with no threads available) this
/// degrades to the default configuration inline.
///
/// An `Unsat` answer from any worker is sound: every worker solves the same
/// formula with a sound solver, and workers never share learnt clauses.
pub fn solve_portfolio(num_vars: usize, clauses: &[Vec<i32>], limits: Limits) -> SolveResult {
    // A4: portfolio workers are Puto instances. If the caller passed Solvo's
    // `Limits::default()` they almost certainly meant "the family defaults",
    // not "Solvo's tighter budget" — substitute Puto's scale so the portfolio
    // is never weaker per worker than a sequential Puto solve.
    let limits = if limits == Limits::default() {
        portfolio_default_limits()
    } else {
        limits
    };
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = available.min(PORTFOLIO.len()).max(1);
    if workers == 1 {
        return run_portfolio_worker(
            &PortfolioConfig::DEFAULT,
            num_vars,
            clauses,
            &limits,
            None,
            None,
        );
    }

    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // C7: one glue-learnt slab shared by every worker.
    let learnt_share = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<i32>>::new()));
    // Capture one shared reference; each `move` closure copies the reference
    // instead of trying to move the `Limits` value.
    let limits_ref = &limits;
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<SolveResult>();
        for cfg in &PORTFOLIO[..workers] {
            let tx = tx.clone();
            let cancel = std::sync::Arc::clone(&cancel);
            let learnt_share = std::sync::Arc::clone(&learnt_share);
            scope.spawn(move || {
                // Panic containment: a worker that unwinds must not
                // abort the whole portfolio — the other workers may already
                // hold a sound answer. `AssertUnwindSafe` is justified
                // because each worker exclusively owns its solver; the only
                // shared state is the cancel atomic.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_portfolio_worker(
                        cfg,
                        num_vars,
                        clauses,
                        limits_ref,
                        Some(&cancel),
                        Some(&learnt_share),
                    )
                }))
                .unwrap_or(SolveResult::Unknown);
                let _ = tx.send(result);
            });
        }
        drop(tx);

        let mut last_unknown: Option<SolveResult> = None;
        for result in rx {
            match result {
                SolveResult::Sat(_) | SolveResult::Unsat { .. } => {
                    // Retire the other workers.
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    return result;
                }
                SolveResult::Unknown => last_unknown = Some(result),
            }
        }
        last_unknown.unwrap_or(SolveResult::Unknown)
    })
}

/// Solve a CNF with an escalating backend strategy (B5).
///
/// Solvo is a learning-free DPLL: on hard instances near the 3-SAT phase
/// transition its default budget can exhaust before it finds an answer, even
/// when a CDCL solver would finish. This wrapper makes the family story real:
/// try Solvo first (cheap on easy/structured formulas), and only when it
/// returns `Unknown` re-run the *same* formula on Puto, which learns.
///
/// The two backends must be semantics-equivalent (they are: the differential
/// harness pins that), so the escalated result is sound either way. On `Sat`
/// the returned model is from whichever backend produced it.
pub fn solve_escalating(num_vars: usize, clauses: &[Vec<i32>], limits: Limits) -> SolveResult {
    let mut solvo = solvo::Solver::with_limits(limits.clone());
    if num_vars > 0 {
        solvo.ensure_var(num_vars);
    }
    for clause in clauses {
        solvo.add_clause(clause);
    }
    match solvo.solve() {
        SolveResult::Unknown => {
            let mut puto = puto::Solver::with_limits(limits);
            if num_vars > 0 {
                puto.ensure_var(num_vars);
            }
            for clause in clauses {
                puto.add_clause(clause);
            }
            puto.solve()
        }
        conclusive => conclusive,
    }
}

// ─── Cardinality encodings (generic over the solver interface) ───────────────

/// Validate and canonicalise a cardinality-constraint literal set.
///
/// Drops the invalid literal `0`, rejects `i32::MIN` (it has no negation and
/// `lit_var(i32::MIN)` would make `ensure_var` request a 2³¹-slot
/// allocation), and removes repeated literals so an at-most-one / at-most-k
/// count is computed over the *set* the caller meant.
///
/// Returns `None` for invalid input (contains `i32::MIN`); callers must then
/// do nothing. The dedup preserves first-occurrence order for small sets and
/// falls back to sort+dedup for large ones (the encodings are order-insensitive,
/// but a deterministic order keeps clause output reproducible).
fn cardinality_lits(lits: &[i32]) -> Option<Vec<i32>> {
    if lits.iter().any(|&l| l == i32::MIN) {
        return None;
    }
    let filtered: Vec<i32> = lits.iter().copied().filter(|&l| l != 0).collect();
    if filtered.len() <= 1 {
        return Some(filtered);
    }
    if filtered.len() <= 64 {
        let mut out: Vec<i32> = Vec::with_capacity(filtered.len());
        for l in filtered {
            if !out.contains(&l) {
                out.push(l);
            }
        }
        return Some(out);
    }
    let mut out = filtered;
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// Encoding dispatch thresholds, matching the per-solver
/// `add_at_most_one` split (pairwise ≤ 8, sequential < 50, commander beyond).
/// Duplicated here so the generic hierarchical recursion can pick the same
/// strategy at every level.
const AMO_PAIRWISE_THRESHOLD: usize = 8;
const AMO_COMMANDER_THRESHOLD: usize = 50;

/// Pairwise at-most-one: for each pair (i, j), add (¬i ∨ ¬j). O(n²) clauses
/// but the strongest propagation — best for small groups.
pub fn encode_at_most_one_pairwise<S>(s: &mut S, lits: &[i32])
where
    S: SatSolver + ?Sized,
{
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            s.add_clause(&[-lits[i], -lits[j]]);
        }
    }
}

/// Sequential counter at-most-one (Sinz, 2005): O(n) clauses, O(n) aux
/// variables. `s_i` means "at least one of lits[0..=i] is true"; the
/// induction `s_i ≡ OR(x_0..=x_i)` makes the conflicts exactly the pairwise
/// violations.
pub fn encode_at_most_one_sequential<S>(s: &mut S, lits: &[i32])
where
    S: SatSolver + ?Sized,
{
    let n = lits.len();
    debug_assert!(n >= 2);
    debug_assert!(
        lits.iter().all(|&l| l != 0 && l != i32::MIN),
        "cardinality encodings require nonzero, non-i32::MIN literals"
    );
    let mut aux = Vec::with_capacity(n - 1);
    for _ in 0..(n - 1) {
        aux.push(s.new_var() as i32);
    }

    // Clause 1: ¬lits[0] ∨ s[0]
    s.add_clause(&[-lits[0], aux[0]]);

    // For i in 1..n-1:
    //   ¬lits[i] ∨ s[i]
    //   ¬s[i-1] ∨ s[i]
    //   ¬lits[i] ∨ ¬s[i-1]
    for i in 1..(n - 1) {
        s.add_clause(&[-lits[i], aux[i]]);
        s.add_clause(&[-aux[i - 1], aux[i]]);
        s.add_clause(&[-lits[i], -aux[i - 1]]);
    }

    // Final: ¬lits[n-1] ∨ ¬s[n-2]
    s.add_clause(&[-lits[n - 1], -aux[n - 2]]);
}

/// Commander at-most-one (Klieber): partition the literals into groups of
/// `group_size`; encode pairwise AMO within each group, make the group's
/// commander equivalent to "some member is true", and encode AMO over the
/// commanders. Two literals in different groups then cannot both be true.
/// More clauses than Sinz but strictly better propagation at large n.
pub fn encode_at_most_one_commander<S>(s: &mut S, lits: &[i32], group_size: usize)
where
    S: SatSolver + ?Sized,
{
    debug_assert!(group_size >= 2);
    let mut commanders: Vec<i32> = Vec::new();
    for group in lits.chunks(group_size) {
        if group.len() == 1 {
            // A singleton group needs no commander (its literal can never
            // conflict across groups through this group).
            commanders.push(group[0]);
            continue;
        }
        let c = s.new_var() as i32;
        // (1) At most one member true within the group.
        encode_at_most_one_pairwise(s, group);
        // (2) member → commander.
        for &x in group {
            s.add_clause(&[-x, c]);
        }
        // (3) commander → some member true.
        let mut or = Vec::with_capacity(group.len() + 1);
        or.push(-c);
        or.extend_from_slice(group);
        s.add_clause(&or);
        commanders.push(c);
    }
    // (4) At most one commander true.
    //
    // Hierarchical: the flat pairwise AMO over the commanders is
    // quadratic in the *number of commanders*, so n literals produce
    // O((n/g)²) clauses — at n = 100k, g = 10 that is ~5×10⁷ clauses built
    // at add time, bypassing every database-size gate. Recursing instead of
    // going flat keeps the encoding O(n log n) with log-depth propagation.
    if commanders.len() <= AMO_PAIRWISE_THRESHOLD {
        encode_at_most_one_pairwise(s, &commanders);
    } else if commanders.len() < AMO_COMMANDER_THRESHOLD {
        encode_at_most_one_sequential(s, &commanders);
    } else {
        encode_at_most_one_commander(s, &commanders, group_size);
    }
}

/// Totalizer at-most-`k` (Bailleux & Boufkhad): a binary tree of "counting"
/// aux variables where node output `s_t` means "at least `t` of this node's
/// inputs are true", built from merge clauses `(¬a_i ∨ ¬b_j ∨ s_{i+j})`, with
/// counters capped at `k+1` and the final constraint `¬s_{k+1}`. Aux
/// variables start at zero VSIDS activity in both backends.
///
/// `k == 0` forces every literal false; `lits.len() ≤ k` is trivially true.
pub fn encode_at_most_k<S>(s: &mut S, lits: &[i32], k: usize)
where
    S: SatSolver + ?Sized,
{
    // : validate at the boundary. Without this, a duplicated input literal is
    // counted twice (the encoding then no longer matches at-most-k over the
    // *set*), and `i32::MIN` would reach `lit_neg`. The cleaned set is the
    // authoritative input for the whole recursion.
    let Some(lits) = cardinality_lits(lits) else {
        return;
    };
    let lits = lits.as_slice();
    if lits.len() <= k {
        return; // trivially satisfied
    }
    if k == 0 {
        for &x in lits {
            s.add_clause(&[-x]);
        }
        return;
    }
    let cap = k + 1; // counters only need to reach k+1
    // Each node: counter literals [s_1, …, s_t] with t ≤ cap.
    let mut level: Vec<Vec<i32>> = lits.iter().map(|&l| vec![l]).collect();
    while level.len() > 1 {
        let mut next: Vec<Vec<i32>> = Vec::with_capacity(level.len() / 2 + 1);
        for pair in level.chunks(2) {
            if pair.len() == 1 {
                next.push(pair[0].clone());
                continue;
            }
            let (a, b) = (&pair[0], &pair[1]);
            let out_len = (a.len() + b.len()).min(cap);
            let mut out: Vec<i32> = Vec::with_capacity(out_len);
            for t in 1..=out_len {
                let sv = s.new_var() as i32;
                out.push(sv);
                // Merge clauses: a_i ∧ b_j → s_{i+j} for i + j = t
                // (a_0 / b_0 stand for "at least 0" — no antecedent).
                let lo = t.saturating_sub(b.len());
                let hi = t.min(a.len());
                for i in lo..=hi {
                    let j = t - i;
                    let mut clause = Vec::with_capacity(3);
                    if i >= 1 {
                        clause.push(-a[i - 1]);
                    }
                    if j >= 1 {
                        clause.push(-b[j - 1]);
                    }
                    clause.push(sv);
                    s.add_clause(&clause);
                }
            }
            next.push(out);
        }
        level = next;
    }
    let top = &level[0];
    if top.len() >= cap {
        // At most k: the (k+1)-th counting output must be false.
        s.add_clause(&[-top[cap - 1]]);
    }
}

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

    /// Add an at-most-`k` constraint over `lits` (totalizer encoding).
    /// `k == 0` forces all literals false; `lits.len() ≤ k` is a no-op.
    fn add_at_most_k(&mut self, lits: &[i32], k: usize) {
        encode_at_most_k(self, lits, k);
    }

    /// Solve the current formula.
    fn solve(&mut self) -> SolveResult;

    /// Solve under temporary assumptions (without polluting state).
    /// On `Unsat`, implementations supporting core extraction return the
    /// conflicting assumption subset via [`SolveResult::unsat_core`].
    fn solve_assumptions(&mut self, assumptions: &[i32]) -> SolveResult;

    /// Number of declared variables.
    fn num_vars(&self) -> usize;

    /// Number of live clauses currently in the solver's internal database
    /// (input clauses plus learnts / internally derived clauses; tombstoned
    /// clauses excluded). Backends can differ here — for a cross-backend
    /// comparable count use [`SatSolver::num_original_clauses`].
    fn num_clauses(&self) -> usize;

    /// Number of *input* clauses: the length of [`SatSolver::clauses_snapshot`].
    /// This is the backend-independent count, unaffected by learnts,
    /// resolvents, tombstones or vivification.
    fn num_original_clauses(&self) -> usize {
        self.clauses_snapshot().len()
    }

    /// Snapshot of the live original formula (input clauses only — learnt
    /// clauses and internally derived simplifications are excluded).
    /// Used by the differential test harness and DIMACS export.
    fn clauses_snapshot(&self) -> Vec<Vec<i32>>;

    /// DIMACS CNF export of the current formula (variables are 1-based, as
    /// in this family). Useful for cross-checking with external solvers.
    fn to_dimacs(&self) -> String {
        use swrite::{SWrite, swrite, swriteln};
        let clauses = self.clauses_snapshot();
        // Rough preallocation: header + an average of ~8 bytes per literal.
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
}

// ─── Differential test harness ─────────────────────────────────────────────

#[cfg(all(test, not(feature = "shuttle")))]
pub(crate) mod tests {
    use super::*;

    /// Simple LCG-based RNG for reproducible tests (same generator as the
    /// Solvo test module — keep the two in sync or factor out if it grows).
    pub(crate) struct Lcg {
        state: u64,
    }
    impl Lcg {
        pub(crate) fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        pub(crate) fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state >> 33
        }
        pub(crate) fn next_range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize) % (hi - lo + 1)
        }
    }

    pub(crate) fn gen_random_3sat(seed: u64, num_vars: usize, num_clauses: usize) -> Vec<Vec<i32>> {
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

    /// Minimal recursive DPLL oracle for cross-validation testing.
    pub(crate) fn solve_recursive(clauses: &[Vec<i32>], num_vars: usize) -> Option<Vec<bool>> {
        let mut assignment = vec![None; num_vars + 1];

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
                        return false;
                    }
                    if unassigned_count == 1 {
                        let var = unassigned_lit.unsigned_abs() as usize;
                        assignment[var] = Some(unassigned_lit > 0);
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
            let mut var = 0;
            for v in 1..assignment.len() {
                if assignment[v].is_none() {
                    var = v;
                    break;
                }
            }
            if var == 0 {
                return true;
            }
            let saved = assignment.to_vec();
            assignment[var] = Some(true);
            if dpll_recursive(clauses, assignment) {
                return true;
            }
            assignment.copy_from_slice(&saved);
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

    /// Check that `model` satisfies every clause.
    pub(crate) fn model_satisfies(clauses: &[Vec<i32>], model: &[bool]) -> bool {
        clauses.iter().all(|c| {
            c.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                let val = model[var - 1];
                if lit > 0 { val } else { !val }
            })
        })
    }

    // ── Differential: Puto (CDCL) vs Solvo (DPLL) vs recursive oracle ───────

    #[test]
    fn test_differential_puto_solvo_recursive() {
        for seed in 0..80u64 {
            let num_vars = 8 + (seed as usize % 12);
            let clauses = gen_random_3sat(seed, num_vars, num_vars * 3);

            let oracle = solve_recursive(&clauses, num_vars);

            let mut puto = puto::Solver::new();
            for _ in 0..num_vars {
                puto.new_var();
            }
            for c in &clauses {
                puto.add_clause(c);
            }
            let puto_result = puto.solve();

            let mut solvo = solvo::Solver::new();
            for _ in 0..num_vars {
                solvo.new_var();
            }
            for c in &clauses {
                solvo.add_clause(c);
            }
            let solvo_result = solvo.solve();

            assert_eq!(
                puto_result.is_sat(),
                oracle.is_some(),
                "Puto vs oracle mismatch, seed {seed}"
            );
            assert_eq!(
                puto_result.is_sat(),
                solvo_result.is_sat(),
                "Puto vs Solvo mismatch, seed {seed}"
            );
            if let SolveResult::Sat(m) = puto_result {
                assert!(
                    model_satisfies(&clauses, &m),
                    "Puto model violates the formula, seed {seed}"
                );
            }
        }
    }

    // ── Property-based fuzz of the SatSolver trait, both backends ───────────

    /// Randomized interleavings of add_* / solve / solve_assumptions across
    /// both backends with a shared oracle — where incremental-state bugs
    /// hide. Each backend sees the same operation stream.
    #[test]
    fn test_differential_incremental_interleavings() {
        for seed in 0..40u64 {
            let mut rng = Lcg::new(seed + 9999);
            let num_vars = 6 + seed as usize % 8;

            let mut puto = puto::Solver::new();
            let mut solvo = solvo::Solver::new();
            let mut accumulated: Vec<Vec<i32>> = Vec::new();

            for _ in 0..num_vars {
                puto.new_var();
                solvo.new_var();
            }

            for step in 0..30 {
                match rng.next_range(0, 3) {
                    0..=1 => {
                        // Random clause of length 1..=3 over declared vars.
                        let len = rng.next_range(1, 3);
                        let mut clause = Vec::new();
                        for _ in 0..len {
                            let var = rng.next_range(1, num_vars) as i32;
                            let sign = if rng.next().is_multiple_of(2) { 1 } else { -1 };
                            clause.push(var * sign);
                        }
                        puto.add_clause(&clause);
                        solvo.add_clause(&clause);
                        accumulated.push(clause);
                    }
                    _ => {
                        // Random (possibly empty) assumption set, then both
                        // solvers must agree.
                        let n_assumps = rng.next_range(0, 2);
                        let mut assumptions = Vec::new();
                        for _ in 0..n_assumps {
                            let var = rng.next_range(1, num_vars) as i32;
                            let sign = if rng.next().is_multiple_of(2) { 1 } else { -1 };
                            assumptions.push(var * sign);
                        }
                        let rp = puto.solve_assumptions(&assumptions);
                        let rs = solvo.solve_assumptions(&assumptions);
                        assert_eq!(
                            rp.is_sat(),
                            rs.is_sat(),
                            "seed {seed} step {step}: assumptions {assumptions:?} — \
                             puto {rp:?} vs solvo {rs:?}"
                        );

                        // Oracle: run the recursive solver on the accumulated
                        // formula with the assumption literals added as units.
                        let mut oracle_clauses = accumulated.clone();
                        for &a in &assumptions {
                            oracle_clauses.push(vec![a]);
                        }
                        let oracle = solve_recursive(&oracle_clauses, num_vars);
                        assert_eq!(
                            rp.is_sat(),
                            oracle.is_some(),
                            "seed {seed} step {step}: assumptions {assumptions:?} — \
                             puto {rp:?} vs oracle {oracle:?}"
                        );

                        // Core soundness: puto's core assumptions alone (with
                        // the formula) must still be UNSAT.
                        if let Some(core) = rp.unsat_core() {
                            let mut s2 = solvo::Solver::new();
                            for _ in 0..num_vars {
                                s2.new_var();
                            }
                            for c in &accumulated {
                                s2.add_clause(c);
                            }
                            assert!(
                                s2.solve_assumptions(core).is_unsat(),
                                "seed {seed} step {step}: unsound core {core:?}"
                            );
                        }

                        // Sat models must satisfy the accumulated formula
                        // (assumptions included).
                        if let SolveResult::Sat(m) = &rp {
                            for (ci, c) in oracle_clauses.iter().enumerate() {
                                assert!(
                                    model_satisfies(std::slice::from_ref(c), m),
                                    "seed {seed} step {step}: puto model {m:?} violates \
                                     clause {ci} {c:?}\nputo state: {}",
                                    puto.debug_dump(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Encoding property tests ──────────────────────────────────────────────

    #[test]
    fn test_at_most_k_encoding_correctness() {
        // For n and k across the interesting range: exactly the assignments
        // with ≤ k true literals must satisfy the encoding (checked with the
        // recursive oracle against exhaustive enumeration).
        for n in [1usize, 3, 5, 9] {
            for k in 0..=n {
                let mut s = solvo::Solver::new();
                let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                s.add_at_most_k(&lits, k);
                let clauses = s.clauses_snapshot();

                // Enumerate all 2^n assignments: the encoding (including its
                // aux counter vars) must be SAT exactly when ≤ k literals
                // are true. Aux vars are existential — decided by the oracle.
                for mask in 0..(1u32 << n) {
                    let count = mask.count_ones() as usize;
                    let mut oracle_clauses = clauses.clone();
                    for i in 0..n {
                        oracle_clauses.push(vec![if (mask >> i) & 1 == 1 {
                            lits[i]
                        } else {
                            -lits[i]
                        }]);
                    }
                    let ok = solve_recursive(&oracle_clauses, s.num_vars()).is_some();
                    assert_eq!(
                        ok,
                        count <= k,
                        "n={n} k={k} mask={mask}: encoding disagrees with at-most-{k}"
                    );
                }
                // And satisfiability: at-most-k + at-least-one is SAT when
                // 1 ≤ k < n; for k = 0 it is (correctly) UNSAT.
                let sp = {
                    let mut s = puto::Solver::new();
                    let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
                    s.add_at_most_k(&lits, k);
                    s.add_clause(&lits); // force at least one true
                    s.solve().is_sat()
                };
                if k == 0 {
                    assert!(!sp, "n={n}: at-most-0 + at-least-one must be UNSAT");
                } else {
                    assert!(
                        sp,
                        "n={n} k={k}: puto says UNSAT for at-most-k + at-least-one"
                    );
                }
            }
        }
    }

    #[test]
    fn test_commander_amo_correctness() {
        // 60 literals → commander encoding (n ≥ 50). AMO semantics must hold
        // and exactly-one must be satisfiable.
        let n = 60;
        let mut s = solvo::Solver::new();
        let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
        s.add_at_most_one(&lits);
        let clauses = s.clauses_snapshot();

        // AMO semantics: the encoding (including its aux vars) must be UNSAT
        // whenever two input literals are true, SAT for each single-true /
        // all-false assignment. Checked with the recursive oracle over the
        // full clause set — the aux commander vars are decided by the solver.
        let sat_of = |mask: u64| {
            let mut oracle_clauses = clauses.clone();
            for i in 0..n {
                oracle_clauses.push(vec![if (mask >> i) & 1 == 1 {
                    lits[i]
                } else {
                    -lits[i]
                }]);
            }
            let num_all_vars = s.num_vars();
            solve_recursive(&oracle_clauses, num_all_vars).is_some()
        };
        assert!(sat_of(1), "single-true assignment must satisfy AMO");
        assert!(sat_of(0), "all-false assignment must satisfy AMO");
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    !sat_of((1u64 << i) | (1u64 << j)),
                    "two-true assignment ({i},{j}) must violate AMO"
                );
            }
        }

        // Exactly-one with commander AMO: SAT with exactly one input literal
        // true (commander aux vars are decided by the solver — not counted).
        let mut s = puto::Solver::new();
        let lits: Vec<i32> = (0..n).map(|_| s.new_var() as i32).collect();
        s.add_exactly_one(&lits);
        let r = s.solve();
        assert!(
            matches!(&r, SolveResult::Sat(m)
                if m[..n].iter().filter(|&&v| v).count() == 1),
            "commander exactly-one must be satisfiable with one true input literal"
        );
    }

    // ── DIMACS export sanity ─────────────────────────────────────────────────

    #[test]
    fn test_dimacs_export_wellformed() {
        let mut s = puto::Solver::new();
        let a = s.new_var();
        let b = s.new_var();
        s.add_clause(&[a as i32, b as i32]);
        s.add_clause(&[-(a as i32)]);
        let dimacs = s.to_dimacs();
        let mut lines = dimacs.lines();
        assert_eq!(lines.next(), Some("p cnf 2 2"));
        let bodies: Vec<&str> = lines.collect();
        assert_eq!(bodies.len(), 2);
        for body in &bodies {
            assert!(
                body.ends_with(" 0"),
                "clause line must end with 0: {body:?}"
            );
            for tok in body
                .trim_end()
                .strip_suffix("0")
                .unwrap()
                .split_whitespace()
            {
                let n: i32 = tok.parse().expect("DIMACS literal must be an integer");
                assert!((1..=2).contains(&n.abs()), "literal out of range: {n}");
            }
        }
    }

    #[test]
    fn test_solver_family_is_comparable() {
        // Both backends produce identical DIMACS for the same input — the
        // compiler's differential/debug tooling relies on this.
        let build = |s: &mut dyn SatSolver| {
            let a = s.new_var() as i32;
            let b = s.new_var() as i32;
            let c = s.new_var() as i32;
            s.add_clause(&[a, b, c]);
            s.add_implies(a, b);
            s.add_at_most_one(&[a, b, c]);
        };
        let mut p = puto::Solver::new();
        build(&mut p);
        let mut v = solvo::Solver::new();
        build(&mut v);
        assert_eq!(p.to_dimacs(), v.to_dimacs());
    }

    // ── Parallel portfolio ───────────────────────────────────────────────────

    #[test]
    fn test_portfolio_matches_sequential() {
        for seed in 0..8u64 {
            let clauses = gen_random_3sat(seed, 12, 40);
            let mut s = puto::Solver::new();
            s.ensure_var(12);
            for c in &clauses {
                s.add_clause(c);
            }
            let direct = s.solve();
            let portfolio = solve_portfolio(12, &clauses, Limits::default());
            assert_eq!(direct.is_sat(), portfolio.is_sat(), "seed {seed}");
            assert_eq!(direct.is_unsat(), portfolio.is_unsat(), "seed {seed}");
            if let SolveResult::Sat(model) = portfolio {
                assert!(
                    model_satisfies(&clauses, &model),
                    "portfolio model violates the formula (seed {seed})"
                );
            }
        }
    }

    #[test]
    fn test_portfolio_detects_unsat() {
        // (x) ∧ (¬x) is trivially unsatisfiable; every worker must agree.
        let clauses = vec![vec![1], vec![-1]];
        let result = solve_portfolio(1, &clauses, Limits::default());
        assert!(result.is_unsat(), "expected UNSAT, got {result:?}");
    }

    #[test]
    fn test_solver_family_snapshot_matches_with_units() {
        // E4/A8#5: the backends must expose the *same* original formula,
        // including unit clauses interleaved with binaries and ternaries —
        // the old snapshots ordered units differently and silently diverged.
        let mut p = puto::Solver::new();
        let mut v = solvo::Solver::new();
        let formula: [&[i32]; 4] = [&[1], &[-2, 3], &[4, 5, 6], &[-7]];
        for c in formula {
            p.add_clause(c);
            v.add_clause(c);
        }
        assert_eq!(
            p.clauses_snapshot(),
            v.clauses_snapshot(),
            "Puto and Solvo snapshots must agree on unit-bearing formulas"
        );
    }

    #[test]
    fn test_commander_amo_stays_subquadratic() {
        // The flat commander AMO is quadratic in the number of
        // commanders. At 10k literals the hierarchical encoding must stay well
        // below what the flat one would build. The old bound (<600k) was
        // above the flat encoding's ~555k clause count, so it could not catch
        // the regression it guarded; <150k is comfortably above the
        // hierarchical result and below flat.
        let mut s = puto::Solver::new();
        s.set_preprocess(false);
        s.ensure_var(10_000);
        let lits: Vec<i32> = (1..=10_000).collect();
        s.add_at_most_one(&lits);
        let n = s.num_clauses();
        assert!(n < 150_000, "commander AMO blew up: {n} clauses");
    }

    // ── Post-solve snapshot equality () ──────────────────────────────────

    #[test]
    fn test_post_solve_snapshot_equality() {
        // Both backends must expose the *input* formula after a solve, even
        // though Puto's live database may hold BVE resolvents, dropped
        // subsumed clauses, and vivified (shortened) bodies.
        let clauses: [&[i32]; 5] = [&[1, 2, 3], &[-1, 2], &[2, -3], &[1], &[-2, 3]];
        let mut p = puto::Solver::new();
        let mut v = solvo::Solver::new();
        p.ensure_var(3);
        v.ensure_var(3);
        for c in clauses {
            p.add_clause(c);
            v.add_clause(c);
        }
        let rp = p.solve();
        let rv = v.solve();
        assert_eq!(rp.is_sat(), rv.is_sat());
        assert_eq!(
            p.clauses_snapshot(),
            v.clauses_snapshot(),
            "snapshots diverged after a solve with preprocessing on"
        );
        assert_eq!(p.to_dimacs(), v.to_dimacs());
        assert_eq!(p.num_original_clauses(), clauses.len());
        assert_eq!(v.num_original_clauses(), clauses.len());
    }

    // ── Boundary-literal matrix () ───────────────────────────────────────

    #[test]
    fn test_cardinality_boundary_literals() {
        fn exercise<S: SatSolver>(s: &mut S) {
            // Each of these used to panic (`-i32::MIN`) or trigger a 2^31-slot
            // `ensure_var` (release OOM). They must all be harmless no-ops.
            s.add_implies(i32::MIN, 1);
            s.add_implies(1, i32::MIN);
            s.add_equiv(0, 1);
            s.add_equiv(i32::MIN, 2);
            s.add_at_most_one(&[i32::MIN, 1]);
            s.add_at_most_one(&[0, 1]);
            s.add_at_most_k(&[i32::MIN, 1], 1);
            s.add_at_most_k(&[0, 0], 0);
            s.add_exactly_one(&[]);
            s.add_unit(0);
            s.add_unit(i32::MIN);
            assert!(
                s.num_vars() < 1_000_000,
                "an invalid literal must not grow the variable set"
            );
        }
        exercise(&mut puto::Solver::new());
        exercise(&mut solvo::Solver::new());

        // Duplicated at-most-k literals are counted once: at most 1 of the
        // *set* {1,2} is satisfiable (the old code counted each occurrence,
        // so it would have encoded at-most-1 over four slots).
        let mut s = puto::Solver::new();
        s.add_at_most_k(&[1, 1, 2, 2], 1);
        s.add_clause(&[1, 2]); // one of them true
        assert!(s.solve().is_sat());
        let mut s = puto::Solver::new();
        s.add_at_most_k(&[1, 1, 2, 2], 1);
        s.add_clause(&[1]);
        s.add_clause(&[2]);
        assert!(s.solve().is_unsat());
    }

    // ── Portfolio vs sequential on a hard UNSAT instance ───────────────────

    #[test]
    fn test_portfolio_matches_sequential_on_pigeonhole() {
        // PHP(6,5): 6 pigeons, 5 holes — UNSAT. Small enough to finish, hard
        // enough that a worker with a tighter budget than sequential Puto
        // (the A4 bug) could diverge in outcome.
        const PIGEONS: usize = 6;
        const HOLES: usize = 5;
        let var = |p: usize, h: usize| (p * HOLES + h + 1) as i32;
        let clauses = {
            let mut c: Vec<Vec<i32>> = Vec::new();
            for p in 0..PIGEONS {
                c.push((0..HOLES).map(|h| var(p, h)).collect());
                for h1 in 0..HOLES {
                    for h2 in (h1 + 1)..HOLES {
                        c.push(vec![-var(p, h1), -var(p, h2)]);
                    }
                }
            }
            for h in 0..HOLES {
                for p1 in 0..PIGEONS {
                    for p2 in (p1 + 1)..PIGEONS {
                        c.push(vec![-var(p1, h), -var(p2, h)]);
                    }
                }
            }
            c
        };
        let num_vars = PIGEONS * HOLES;
        let mut seq = puto::Solver::new();
        seq.ensure_var(num_vars);
        for c in &clauses {
            seq.add_clause(c);
        }
        let sequential = seq.solve();
        let portfolio = solve_portfolio(num_vars, &clauses, Limits::default());
        assert!(sequential.is_unsat(), "PHP(6,5) must be UNSAT");
        assert_eq!(
            sequential.is_unsat(),
            portfolio.is_unsat(),
            "portfolio disagreed with sequential on PHP(6,5)"
        );
    }
}
