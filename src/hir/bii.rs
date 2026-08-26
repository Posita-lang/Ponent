//! Best Inductive Invariant (BII) synthesis over a template domain —
//! following Hanrui Zuo, Peisen Yao, and Kui Ren, "A Fresh Look at Best
//! Inductive Loop Invariant Synthesis for Bit-Vector Relations"
//! (ACM, July 2026).
//!
//! The BII is the strongest inductive invariant expressible in a template
//! domain `L ≤ F(X) ≤ U` (Definition 3.5). This module implements the
//! propose-and-refine framework (Algorithm 2) with both the linear search
//! strategy (Algorithm 3) and the bitwise greedy strategy (Algorithm 4)
//! augmented by boundary-limit bounded leap (Algorithm 5).
//!
//! # Semantics operating point
//!
//! The paper fixes ONE semantics (unsigned bit-vectors, §3.3). This
//! pipeline serves a language with THREE overflow policies, each
//! mapped to its exact encoding:
//!   trap (language default) → LIA + definedness antecedents
//!       (mathematical integers are EXACT for partial arithmetic:
//!        the inductive step is conditioned on non-trapping states)
//!   wrap (`+%`)             → BV (the paper's setting), opt-in
//!   saturate (`+?`)         → Clamp rows (beyond the paper)
//! The LIA default is the exact encoding of the default semantics,
//! not a divergence; the paper's setting is the opt-in for loops
//! that have it. Signed support (sign-extended operands, shifted-
//! domain bit-walk, signed boundary seeds) lies OUTSIDE the paper's
//! validated unsigned scope (Lemma 5.2 / Theorem 5.3) and carries
//! its own in-line soundness arguments. The template family is the
//! sparse template-polyhedra hierarchy of the paper's Section 4
//! (Interval ⊂ Zone ⊂ Octagon ⊂ SparsePoly — "additionally include"),
//! exercised at the full SparsePoly level.
//!
//! All bound arithmetic uses `num_bigint::BigInt` to support bit-widths
//! up to 128 and beyond without overflow. The SMT encoding supports both
//! LIA (linear integer arithmetic) and BV (fixed-width bit-vectors with
//! modular wrap-around semantics; quantified queries use `(set-logic BV)`).
//!
//! Per the 2026-08-13/14 rulings the synthesized bounds are submitted
//! through the `@hint` channel (the solver stays the authority); bounds
//! inside the exact difference-constraint sub-language may additionally
//! self-discharge via `expr_entails_typed` (wired in the checker's
//! `obligation_provable`; wrap-semantics loops gate to the BV discharge).
//!
//! The LIA encoding (`i128` + `(set-logic LIA)`) is the DEFAULT solver
//! encoding; a QF_BV / `BV` bit-vector encoding — per-width
//! `(_ BitVec N)` sorts, modular `bvadd`/`bvsub` with the paper's
//! wrap-around semantics — is also AVAILABLE via
//! `synthesize_*_bii(.., use_bv: true)` (and exercised by the wrap-around
//! defensive tests). The loop-level obligation point that was missing
//! ([L3], Layer S) has since landed: while `invariant`/`decreases`/`@hint`
//! clauses are VERIFIED (see `verify_loop_invariant`/
//! `verify_loop_decreases`/`verify_loop_hint` in the checker), so a
//! BV-synthesized candidate now feeds a real verification path, not just
//! the advisory `@hint` channel. The checker still defaults to LIA
//! (`use_bv: false`) because the unbounded-integer semantics match the
//! difference-constraint candidates and the exact self-discharge path
//! (`expr_entails_typed`); the BV encoding plugs in behind the existing
//! `RowKind`/template abstraction without touching the propose-and-refine
//! driver.
//!
//! # Deviations from the paper (rationale)
//!
//! - **Bounded leap strict-tightening** (Algorithm 5, Theorem 5.5).  The
//!   leap query conjoins a STRICT-tightening obligation:
//!   `A′ ⊏ A ≜ ∃r. l′_r > A.l_r ∨ u′_r < A.u_r` — the witness must
//!   tighten at least one row.  UNSAT then proves `A = A*`: any
//!   `A* ∈ [B, A]` with `A* ⊏ A` would satisfy the query, so
//!   unsatisfiability contradicts the existence of a strictly tighter
//!   inductive template in the bounded region.  The strict conjunct
//!   adds no query — the 4W bound (Theorem 5.5) is unchanged.
//!
//! - **Position-pointer advancement** (Algorithm 4, Lemma 5.2): pointers
//!   advance only after UNSAT (the template is unchanged), never after
//!   SAT (the witness changed the template).  This is Algorithm 4's
//!   `A = A_last` test, realized as an explicit `prev_unsat` flag.
//!
//! - **Negative-bound bitwise encoding** (§5.1): Diff rows with negative
//!   lower bounds use an offset encoding that maps the signed range
//!   `[-m, m]` to the unsigned range `[0, 2m]` before bit manipulation,
//!   preserving the O(Σ bw) query bound.
//!
//! - **Sequential transition fail-closed**: `encode_sequential_transition`
//!   returns `Option`, and any unsupported `LoopInstr` causes the entire
//!   synthesis to fail closed rather than silently ignoring instructions.
//!
//! - **Solver-call counting**: the query budget counts every raw
//!   `run_raw_query` invocation, including the separate `(get-model)`
//!   round-trip that follows each SAT outcome. Theorem 5.3 of Zuo et al. (2026) /
//!   5.5 bounds (`2W` / `4W`) count high-level `Refine` queries, so the
//!   raw count is a constant factor (~2×) above the paper's bound; the
//!   budget parameter must be sized accordingly (the checker passes a
//!   generous 512).
//!
//! - **Sum rows**: the template domain now includes `x_i + x_j` rows
//!   (the full octagon), with correct top bounds `2·(2^bw − 1)` and
//!   encoded bit-width `bw + 1`.
//!
//! - **Signed relational rows**: Sum and Support3 rows are
//!   generated over uniform-signedness operand sets with TRUE signed
//!   tops (signed Sum: [−2^bw, 2^bw−2]; signed Support3:
//!   signed_s3_range); mixed-signature pairs were once skipped (symmetric
//!   mixed tops would EXCLUDE reachable states — Int<8> − UInt<8>
//!   reaches −383 < −255). Mixed-signedness relational rows carry exact
//!   asymmetric tops in BOTH modes: LIA reasons over mathematical
//!   integers; under BV each operand is extended by its OWN signedness
//!   (per-operand ext_operand), and comparisons (guards / definedness)
//!   use the value-preserving lift with SIGNED predicates
//!   (lift_cmp_operand — its LEMMA and the W_c/W_d width rules are stated in situ); the
//!   row-level signed flag no longer drives operand extension, and the
//!   comparison-level flag no longer selects predicates. Under BV,
//!   relational-row operands
//!   SIGN-extend (a two's-complement negative value zero-extended
//!   inflates by 2^bw and corrupts the offset-domain comparison).

use crate::ast::{self, IntLit};
use crate::hir::loop_infer::LoopInstr;
use crate::hir::octagon::{ClosureMode, Dbm};
use crate::hir::smt::{RawQueryOutcome, SmtSolver};
use crate::symbol::Symbol;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use sexp::Sexp;
use std::ops::Neg;

/// A single template row `l ≤ f(X) ≤ u` over the loop variables.
///
/// Bounds are stored as arbitrary-precision integers to support
/// bit-widths up to 128 (unsigned top = 2^128 − 1 exceeds `i128::MAX`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BiiRow {
    pub(crate) kind: RowKind,
    /// Bit-width of the *variables* in this row (unsigned interpretation).
    /// For Interval rows this is the variable's own width; for Diff/Sum
    /// rows it is the maximum of the two operand widths.
    pub(crate) bw: u8,
    pub(crate) lb: BigInt,
    pub(crate) ub: BigInt,
    /// true when the row's variables are signed (`Int<N>`): Interval rows
    /// then carry TRUE signed bounds and compare with `bvsle`/`bvsge`
    /// under BV (the signed row tranche: Interval + Diff). For MIXED
    /// relational rows this flag is the OR of the operands' signedness —
    /// only Interval/Clamp's domain selection consumes it; the exact
    /// per-operand signedness lives in `full_lb`/`full_ub` (via
    /// `full_range`).
    pub(crate) signed: bool,
    /// The row's FULL value range (the construction tops): bounds can never
    /// leave it. Cached from `full_range` at construction — `is_trivial` /
    /// `offset` / `BoundaryLimits` read these instead of recomputing, because
    /// MIXED-signedness rows need per-operand signedness, which the row's
    /// single `signed` flag no longer encodes.
    pub(crate) full_lb: BigInt,
    pub(crate) full_ub: BigInt,
}

/// The linear form `f(X)` of a template row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowKind {
    /// `x_i`
    Interval(usize),
    /// `x_i - x_j`
    Diff(usize, usize),
    /// `x_i + x_j`
    Sum(usize, usize),
    /// `x_i + s_j·x_j + s_k·x_k` (i < j < k), sparse template polyhedra.
    /// `sj`/`sk` are the signs of `x_j`/`x_k` (`true` = +1, `false` = −1);
    /// `x_i` always has coefficient +1 (the paper's canonical form).
    Support3(usize, usize, usize, bool, bool),
    /// `clamp(x_i + c)` — the clamped successor of a saturating
    /// assignment `x_i := x_i +? c`. The row
    /// constrains the clamped value: `l ≤ clamp(x_i + c) ≤ u`; the
    /// SMT encoding uses an `ite` for the clamp (piecewise).
    Clamp(usize, i128),
}

/// Template row-set level (Zuo et al. 2026, §4.2): how rich the template domain is.
/// `Interval` = single-variable rows only; `Zone` adds difference rows
/// (`x − y ≤ c`); `Octagon` further adds sum rows (`x + y ≤ c`);
/// `SparsePoly` (the default) further adds the sparse template-polyhedra
/// Support3 rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemplateLevel {
    /// Interval rows only.
    Interval,
    /// Interval + octagon Diff rows (zone domain: `x − y ≤ c` only, no sums).
    Zone,
    /// Interval + octagon Diff/Sum rows.
    Octagon,
    /// Interval + octagon + sparse Support3 rows (the full domain).
    SparsePoly,
}

/// A variable's full value range at width `bw`: signed `[−2^(bw−1), 2^(bw−1)−1]`,
/// unsigned `[0, 2^bw − 1]`.
fn var_bounds(bw: u8, signed: bool) -> (BigInt, BigInt) {
    if signed {
        let half = BigInt::one() << (bw as usize - 1);
        (-half.clone(), half - 1)
    } else {
        (BigInt::zero(), BiiRow::max_ub(bw))
    }
}

/// The EXACT full value range `[lo, hi]` of a row's linear form, computed from
/// the per-variable signedness `var_signed` (each operand is treated as if it
/// had the row's `bw` — exact when widths are equal, a sound superset when
/// they differ, matching the existing uniform-row convention). This UNIFIES
/// and REPLACES `s3_range` / `signed_s3_range` (their values are the
/// uniform-signedness special cases) and extends to MIXED-signedness rows:
/// Diff(i,j) → [min_i − max_j, max_i − min_j]; Sum(i,j) → [min_i + min_j,
/// max_i + max_j]; Support3 → the sum of the per-term ranges (term `−x`
/// spans `[−max, −min]`).
fn full_range(kind: RowKind, bw: u8, var_signed: &[bool]) -> (BigInt, BigInt) {
    let vb = |i: usize| var_bounds(bw, var_signed.get(i).copied().unwrap_or(false));
    match kind {
        RowKind::Interval(i) | RowKind::Clamp(i, _) => vb(i),
        RowKind::Diff(i, j) => {
            let (min_i, max_i) = vb(i);
            let (min_j, max_j) = vb(j);
            (min_i - max_j, max_i - min_j)
        }
        RowKind::Sum(i, j) => {
            let (min_i, max_i) = vb(i);
            let (min_j, max_j) = vb(j);
            (min_i + min_j, max_i + max_j)
        }
        RowKind::Support3(i, j, k, sj, sk) => {
            let term = |(lo, hi): (BigInt, BigInt), pos: bool| {
                if pos { (lo, hi) } else { (-hi, -lo) }
            };
            let (a0, a1) = term(vb(i), true);
            let (b0, b1) = term(vb(j), sj);
            let (c0, c1) = term(vb(k), sk);
            (a0 + b0 + c0, a1 + b1 + c1)
        }
    }
}

impl BiiRow {
    /// The largest unsigned value at `bw` bits: `2^bw − 1`.
    fn max_ub(bw: u8) -> BigInt {
        (BigInt::one() << bw as usize) - BigInt::one()
    }

    /// The smallest signed value for a Diff row: `−(2^bw − 1)`.
    #[allow(dead_code)] // referenced by tests; production reads full_lb/full_ub.
    fn min_diff(bw: u8) -> BigInt {
        -Self::max_ub(bw)
    }

    /// The largest value for a Sum row: `2·(2^bw − 1)`.
    /// Two `bw`-bit unsigned values sum to at most twice the max.
    #[allow(dead_code)] // referenced by tests; production reads full_lb/full_ub.
    fn max_sum_ub(bw: u8) -> BigInt {
        Self::max_ub(bw) * BigInt::from(2)
    }

    /// The offset that maps the row's full range onto `[0, width]`:
    /// `−full_lb` for relational rows (Diff/Sum/Support3 — including the
    /// mixed-signedness rows, whose asymmetric tops this generalization
    /// exists for); 0 for Interval/Clamp (unsigned rows are already
    /// non-negative; signed rows compare in the TRUE signed domain, not the
    /// offset domain).
    fn offset(&self) -> BigInt {
        match self.kind {
            RowKind::Interval(_) | RowKind::Clamp(..) => BigInt::zero(),
            _ => -self.full_lb.clone(),
        }
    }

    /// The encoded width in bits (after offsetting / range expansion):
    /// - Interval: `bw` (range `[0, 2^bw − 1]`)
    /// - Diff: `bw + 1` (offset range `[0, 2·(2^bw − 1)]`)
    /// - Sum: `bw + 1` (range `[0, 2·(2^bw − 1)]`)
    /// - Support3: `bw + 2` (offset range `[0, 3·(2^bw − 1)]`)
    /// Mixed-signedness rows keep the same encoded widths: every operand
    /// spreads ≤ 2^bw − 1 regardless of signedness.
    fn enc_bw(&self) -> u32 {
        match self.kind {
            RowKind::Diff(..) | RowKind::Sum(..) => self.bw as u32 + 1,
            RowKind::Support3(..) => self.bw as u32 + 2,
            _ => self.bw as u32,
        }
    }

    /// A row still at its full-range tops carries no information — skip it
    /// when reporting candidates.
    fn is_trivial(&self) -> bool {
        self.lb <= self.full_lb && self.ub >= self.full_ub
    }
}

/// The template domain: interval rows per variable plus octagon
/// difference and sum rows, each with an unsigned bound interpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BiiTemplate {
    pub(crate) n_vars: usize,
    pub(crate) rows: Vec<BiiRow>,
}

impl BiiTemplate {
    /// Interval + octagon-diff + octagon-sum + sparse support-three rows
    /// over ALL operand sets: uniform pairs keep the exact symmetric
    /// tops; MIXED pairs carry exact ASYMMETRIC tops from `full_range`
    /// (e.g. `Int<8> − UInt<8>` → [−383, 127]). Sound in BOTH modes:
    /// LIA is mathematical-integer arithmetic (no promotion needed);
    /// under BV each operand is extended by its OWN signedness
    /// (per-operand `ext_operand`), so the offset-domain comparison is
    /// faithful for mixtures too.
    pub(crate) fn new(n_vars: usize, bit_widths: &[u8], signed: &[bool]) -> BiiTemplate {
        Self::with_level(n_vars, bit_widths, signed, TemplateLevel::SparsePoly)
    }

    /// Build the template with an explicit row-set level
    /// (Zuo et al. 2026, §4.2): `Interval` / `Octagon` / `SparsePoly` (default).
    pub(crate) fn with_level(
        n_vars: usize,
        bit_widths: &[u8],
        signed: &[bool],
        level: TemplateLevel,
    ) -> BiiTemplate {
        Self::build(n_vars, bit_widths, signed, level)
    }
    fn build(
        n_vars: usize,
        bit_widths: &[u8],
        signed: &[bool],
        level: TemplateLevel,
    ) -> BiiTemplate {
        let bw_of = |v: usize| bit_widths.get(v).copied().unwrap_or(64);
        let signed_of = |v: usize| signed.get(v).copied().unwrap_or(false);
        let mk = |kind: RowKind, bw: u8, row_signed: bool| -> BiiRow {
            let (lo, hi) = full_range(kind, bw, signed);
            BiiRow {
                kind,
                bw,
                lb: lo.clone(),
                ub: hi.clone(),
                signed: row_signed,
                full_lb: lo,
                full_ub: hi,
            }
        };
        let mut rows = Vec::new();
        for i in 0..n_vars {
            rows.push(mk(RowKind::Interval(i), bw_of(i), signed_of(i)));
        }
        // Zone/Octagon difference rows `x_i − x_j` (i < j): uniform pairs
        // keep the exact symmetric tops `[−m, m]`; mixed pairs get their
        // exact ASYMMETRIC tops from `full_range`.
        if matches!(level, TemplateLevel::Zone | TemplateLevel::Octagon | TemplateLevel::SparsePoly)
        {
            for i in 0..n_vars {
                for j in (i + 1)..n_vars {
                    let bw = bw_of(i).max(bw_of(j));
                    rows.push(mk(RowKind::Diff(i, j), bw, signed_of(i) || signed_of(j)));
                }
            }
        }
        // Octagon sum rows `x_i + x_j` (i < j); uniform tops `[0, 2m]` /
        // signed `[−2^bw, 2^bw−2]`; mixed tops from `full_range`.
        if matches!(level, TemplateLevel::Octagon | TemplateLevel::SparsePoly) {
            for i in 0..n_vars {
                for j in (i + 1)..n_vars {
                    let bw = bw_of(i).max(bw_of(j));
                    rows.push(mk(RowKind::Sum(i, j), bw, signed_of(i) || signed_of(j)));
                }
            }
        }
        // Sparse template-polyhedra rows `x_i + s_j·x_j + s_k·x_k`
        // (i < j < k), all four sign combinations.
        if matches!(level, TemplateLevel::SparsePoly) {
            for i in 0..n_vars {
                for j in (i + 1)..n_vars {
                    for k in (j + 1)..n_vars {
                        let si = signed_of(i);
                        let bw = bw_of(i).max(bw_of(j)).max(bw_of(k));
                        for (sj, sk) in [(true, true), (true, false), (false, true), (false, false)]
                        {
                            rows.push(mk(
                                RowKind::Support3(i, j, k, sj, sk),
                                bw,
                                si || signed_of(j) || signed_of(k),
                            ));
                        }
                    }
                }
            }
        }
        BiiTemplate { n_vars, rows }
    }
    /// Template construction for a problem with saturating assignments:
    /// the fixed rows plus one Clamp row per `x_i := x_i +? c`, over ALL
    /// operand sets.
    pub(crate) fn with_saturates(
        n_vars: usize,
        bit_widths: &[u8],
        signed: &[bool],
        saturates: &[(usize, i128)],
    ) -> BiiTemplate {
        let mut tpl = Self::build(n_vars, bit_widths, signed, TemplateLevel::SparsePoly);
        for &(i, c) in saturates {
            if i >= n_vars {
                continue; // defensive: saturates reference loop variables.
            }
            let bw = bit_widths.get(i).copied().unwrap_or(64);
            let is_signed = signed.get(i).copied().unwrap_or(false);
            let (lo, hi) = var_bounds(bw, is_signed);
            tpl.rows.push(BiiRow {
                kind: RowKind::Clamp(i, c),
                bw,
                lb: lo.clone(),
                ub: hi.clone(),
                signed: is_signed,
                full_lb: lo,
                full_ub: hi,
            });
        }
        tpl
    }

    /// A clone with a single row bound tightened to `(lb, ub)`.
    fn with_row(&self, idx: usize, lb: BigInt, ub: BigInt) -> BiiTemplate {
        let mut rows = self.rows.clone();
        rows[idx].lb = lb;
        rows[idx].ub = ub;
        BiiTemplate {
            n_vars: self.n_vars,
            rows,
        }
    }
}

/// Synthesize the BII over interval + octagon template rows for a
/// loop given as `LoopInstr`s (the `body` slice includes the guard first,
/// as produced by `hir_loop_to_loop_instrs`; `init` seeds the pre-state).
///
/// Returns the converged template (whose rows are the BII bounds), or
/// `None` on any fail-closed path: solver unavailable / `unknown` /
/// unparsable witness / step budget exhausted.
/// Linear propose-and-refine BII synthesis (O(Σbw · bound-width) solver
/// calls). TEST-ONLY reference implementation: the production path uses
/// the bitwise greedy strategy (`synthesize_bitwise_bii`, O(Σbw)) — linear
/// is kept as an independent cross-check that both strategies converge to
/// the SAME BII (BII is unique per loop), and as a fallback for future
/// small-width scenarios. Gated `#[cfg(test)]` so the reference does not
/// bloat the production binary; remove the gate if it is ever used in
/// production again.
#[cfg(test)]
pub(crate) fn synthesize_linear_bii(
    solver: &SmtSolver,
    vars: &[Symbol],
    init: &[LoopInstr],
    body: &[LoopInstr],
    bit_widths: &[u8],
    signed: &[bool],
    max_refinements: usize,
    use_bv: bool,
) -> Option<BiiTemplate> {
    let mut cur = BiiTemplate::new(vars.len(), bit_widths, signed);
    let mut calls = 0usize;
    for _ in 0..max_refinements {
        // Propose: every immediate neighbor — tighten one row's lower
        // bound by 1, or its upper bound by 1.
        let candidates = propose(&cur);
        if candidates.is_empty() {
            return Some(cur); // nothing tighter is representable — BII.
        }
        // Refine step 1: the ∃∀ query without `(get-model)` — z3 errors
        // on `(get-model)` after `unsat`, so the model request is issued
        // as a SEPARATE query only when the outcome is sat.
        let query = build_refine_query(&candidates, init, body, false, use_bv, bit_widths, signed)?;
        calls += 1;
        if calls > max_refinements * 2 {
            return None;
        }
        match solver.run_raw_query(&query) {
            RawQueryOutcome::Unsat => return Some(cur), // no tighter inductive invariant.
            RawQueryOutcome::Sat(_) => {
                // Refine step 2: same query plus `(get-model)` to read the
                // witness bounds (the re-solve is cached only across
                // identical texts; this one differs by the model request).
                // Budget check before the model round-trip: the SAT path
                // would otherwise overshoot `max_refinements * 2` by one.
                if calls >= max_refinements * 2 {
                    return None;
                }
                let query_model =
                    build_refine_query(&candidates, init, body, true, use_bv, bit_widths, signed)?;
                calls += 1;
                match solver.run_raw_query(&query_model) {
                    RawQueryOutcome::Sat(model) => match parse_witness(&model, &cur.rows, use_bv) {
                        Some(bounds) => {
                            apply_bounds(&mut cur, &bounds);
                        }
                        None => return None, // witness unparsable — fail closed.
                    },
                    RawQueryOutcome::Unknown
                    | RawQueryOutcome::Error(_)
                    | RawQueryOutcome::Unsat => {
                        return None; // fail closed.
                    }
                }
            }
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => return None, // fail closed.
        }
    }
    None // step budget exhausted without convergence — fail closed.
}

/// The immediate lattice neighbors of `A` (Algorithm 3's `Propose`): for
/// each row, tighten the lower bound by 1 and the upper bound by 1.
fn propose(cur: &BiiTemplate) -> Vec<BiiTemplate> {
    let mut out = Vec::new();
    for (idx, row) in cur.rows.iter().enumerate() {
        if row.lb < row.ub {
            out.push(cur.with_row(idx, row.lb.clone() + BigInt::one(), row.ub.clone()));
        }
        if row.ub > row.lb {
            out.push(cur.with_row(idx, row.lb.clone(), row.ub.clone() - BigInt::one()));
        }
    }
    out
}

/// Per-row bit positions for the bitwise greedy strategy (Algorithm 4):
/// `lpos[i]`/`upos[i]` are the highest still-undecided bound bit of row
/// `i`. A position of `-1` means every bit of that bound is fixed.
///
/// Positions are in the ENCODED (offset) space, using `enc_bw()` so
/// that Diff and Sum rows (which need `bw + 1` bits) are covered.
#[derive(Clone, Debug)]
struct BitPositions {
    lpos: Vec<isize>,
    upos: Vec<isize>,
}

impl BitPositions {
    /// Start at the top bit of each row's encoded bound width.
    fn new(tpl: &BiiTemplate) -> BitPositions {
        BitPositions {
            lpos: tpl.rows.iter().map(|r| r.enc_bw() as isize - 1).collect(),
            upos: tpl.rows.iter().map(|r| r.enc_bw() as isize - 1).collect(),
        }
    }

    /// Advance every position by one (Algorithm 4: "if A = A_last then
    /// lpos_i -= 1, upos_i -= 1") — the previous query was UNSAT, so the
    /// bit stays and the search moves to the next lower bit.
    fn advance(&mut self) {
        for p in &mut self.lpos {
            *p -= 1;
        }
        for p in &mut self.upos {
            *p -= 1;
        }
    }
}

/// Algorithm 4's `Propose`: decide each bound bit from high to low.
/// For the lower bound of row `i`, hypothesize that the optimal bound has
/// the highest unfixed unset bit set (and all lower bits cleared); for the
/// upper bound, hypothesize the highest unfixed set bit cleared (lower
/// bits set). Rows whose current bound is negative (octagon `Diff` rows
/// with a negative lower bound) use offset encoding so bit semantics
/// remain meaningful; rows already non-negative use direct bit ops.
fn propose_bitwise(
    cur: &BiiTemplate,
    pos: &BitPositions,
    limits: &BoundaryLimits,
) -> Vec<BiiTemplate> {
    let mut out = Vec::new();
    let one = BigInt::one();
    for (idx, row) in cur.rows.iter().enumerate() {
        let offset = row.offset();
        let enc_bw = row.enc_bw() as usize;
        // Explicit `enc_bw`-bit mask: the `!` below is two's-complement
        // BigInt negation (infinite-width `...1111`), which works but is
        // subtle; masking the result to `enc_bw` bits makes the bit
        // manipulation explicit and bounds the value.
        let width_mask = (BigInt::one() << enc_bw) - BigInt::one();

        // ── Lower bound ──
        if pos.lpos[idx] >= 0 {
            // Signed Interval rows walk the bits in the
            // SHIFTED domain (`enc = true + 2^(bw−1)`), where the signed
            // order is monotone with the bit pattern — a two's-complement
            // walk would jump into the negative region when tightening a
            // non-negative bound (0 → 0x80 = −128), stalling the search.
            let shift =
                if row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
                    BigInt::one() << (enc_bw as usize - 1)
                } else {
                    offset.clone()
                };
            let enc_lb = &row.lb + &shift;
            debug_assert!(!enc_lb.is_negative());

            // Highest unset bit at or below the position pointer.
            let mut k = pos.lpos[idx] as usize;
            loop {
                if k >= enc_bw {
                    break;
                }
                let bit = (&enc_lb >> k) & &one;
                if bit.is_zero() {
                    break;
                }
                if k == 0 {
                    k = usize::MAX; // sentinel: no unset bit found
                    break;
                }
                k -= 1;
            }
            if k != usize::MAX && k < enc_bw {
                let kk = k;
                let bit_val = &one << kk;
                let mask = if kk == 0 {
                    BigInt::zero()
                } else {
                    (&one << kk) - &one
                };
                let new_enc_lb = ((&enc_lb & !&mask) | &bit_val) & &width_mask;
                // Shifted-domain walk result: `new_enc_lb − shift` is the
                // true signed value (monotone with the bit pattern).
                let new_lb = new_enc_lb - &shift;
                // `<=` admits singleton candidates (`new_lb == row.ub`):
                // Propose (Algorithm 4) of Zuo et al. (2026) only requires the
                // proposed bound to be a valid bit hypothesis, and the
                // BII can be a singleton interval (Table 2 iterates
                // `[101, 110]` → `[101, 101]`). Rejecting the equality
                // case made the search stall on width-1 rows.
                //
                // BoundaryLimits pruning: if the proposed lower bound
                // already exceeds the maximum possible optimal lower
                // bound `limits.lb[idx]`, it cannot be part of the BII
                // and is pruned before issuing a solver query.
                if new_lb <= row.ub && limits.lower_candidate_feasible(idx, &new_lb) {
                    out.push(cur.with_row(idx, new_lb, row.ub.clone()));
                }
            }
        }

        // ── Upper bound ──
        if pos.upos[idx] >= 0 {
            // Shifted domain for signed Interval rows
            // (mirrors the lower-bound branch).
            let shift =
                if row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
                    BigInt::one() << (enc_bw as usize - 1)
                } else {
                    offset.clone()
                };
            let enc_ub = &row.ub + &shift;
            debug_assert!(!enc_ub.is_negative());

            // Highest set bit at or below the position pointer.
            let mut k = pos.upos[idx] as usize;
            loop {
                if k >= enc_bw {
                    break;
                }
                let bit = (&enc_ub >> k) & &one;
                if !bit.is_zero() {
                    break;
                }
                if k == 0 {
                    k = usize::MAX;
                    break;
                }
                k -= 1;
            }
            if k != usize::MAX && k < enc_bw {
                let kk = k;
                let mask = if kk == 0 {
                    BigInt::zero()
                } else {
                    (&one << kk) - &one
                };
                let bit_val = &one << kk;
                let new_enc_ub = ((&enc_ub | &mask) & !&bit_val) & &width_mask;
                // Shifted-domain walk result (see lower-bound branch).
                let new_ub = new_enc_ub - &shift;
                // `>=` admits singleton candidates — see the lower-bound
                // branch above.
                //
                // BoundaryLimits pruning: if the proposed upper bound
                // already falls below the minimum possible optimal upper
                // bound `limits.ub[idx]`, it cannot be part of the BII
                // and is pruned before issuing a solver query.
                if new_ub >= row.lb && limits.upper_candidate_feasible(idx, &new_ub) {
                    out.push(cur.with_row(idx, row.lb.clone(), new_ub));
                }
            }
        }
    }
    out
}

/// Boundary limits (Algorithm 5 §5.2): `lb[i]` is the maximum possible
/// value of the optimal lower bound `l_i*`; `ub[i]` the minimum possible
/// value of the optimal upper bound `u_i*`. Every failed (UNSAT) proposal
/// prunes a limit; every adopted witness tightens it. The conjunction
/// `B = ⋀_i [lb[i], ub[i]]` is an under-approximation of the BII, driving
/// the bounded leap.
#[derive(Clone, Debug)]
struct BoundaryLimits {
    lb: Vec<BigInt>,
    ub: Vec<BigInt>,
}

impl BoundaryLimits {
    fn new(tpl: &BiiTemplate) -> BoundaryLimits {
        // lb[i] = the MAX possible optimal upper bound = the row's
        // full-range top; ub[i] = the MIN possible optimal lower bound =
        // the full-range bottom. Generalizes the former per-kind match:
        // identical values for every uniform row kind (including the
        // signed Interval seeding — full_lb of a signed Interval row IS
        // −2^(bw−1), preserving the fix for the "u* ≥ 0" corruption),
        // and correct for mixed-signedness rows by construction.
        BoundaryLimits {
            lb: tpl.rows.iter().map(|r| r.full_ub.clone()).collect(),
            ub: tpl.rows.iter().map(|r| r.full_lb.clone()).collect(),
        }
    }

    /// The under-approximation is non-empty (`B ≠ ⊥`) iff every row's
    /// limits still admit a valid interval.
    fn is_active(&self) -> bool {
        self.lb.iter().zip(&self.ub).all(|(l, u)| l <= u)
    }

    /// UNSAT pruning: the failed candidates tightened exactly ONE bound
    /// each (relative to the current template `cur`). A failed lower-bound
    /// tighten to `row.lb` means no invariant exists with a lower bound
    /// that high → `l* < row.lb`; symmetrically for the upper bound.
    fn prune_unsat(&mut self, cur: &BiiTemplate, cands: &[BiiTemplate]) {
        for c in cands {
            for (i, row) in c.rows.iter().enumerate() {
                let base = &cur.rows[i];
                if row.lb > base.lb {
                    let new_lb = &row.lb - BigInt::one();
                    if new_lb < self.lb[i] {
                        self.lb[i] = new_lb;
                    }
                }
                if row.ub < base.ub {
                    let new_ub = &row.ub + BigInt::one();
                    if new_ub > self.ub[i] {
                        self.ub[i] = new_ub;
                    }
                }
            }
        }
    }

    /// SAT tightening from the adopted witness bounds: l* ≤ witness.ub and
    /// u* ≥ witness.lb.
    fn tighten_sat(&mut self, bounds: &[(BigInt, BigInt)]) {
        for (i, (l, u)) in bounds.iter().enumerate() {
            if *u < self.lb[i] {
                self.lb[i] = u.clone();
            }
            if *l > self.ub[i] {
                self.ub[i] = l.clone();
            }
        }
    }

    /// Whether a lower-bound candidate can still contain the optimal lower
    /// bound. Zuo et al. 2026, §5.2: the optimum satisfies `l* ≤ limits.lb[idx]`, so
    /// a proposed `new_lb` above `limits.lb[idx]` cannot be part of the BII
    /// search region and is pruned before issuing a solver query.
    fn lower_candidate_feasible(&self, idx: usize, new_lb: &BigInt) -> bool {
        *new_lb <= self.lb[idx]
    }

    /// Whether an upper-bound candidate can still contain the optimal upper
    /// bound. Zuo et al. 2026, §5.2: the optimum satisfies `u* ≥ limits.ub[idx]`, so
    /// a proposed `new_ub` below `limits.ub[idx]` cannot be part of the BII
    /// search region and is pruned before issuing a solver query.
    fn upper_candidate_feasible(&self, idx: usize, new_ub: &BigInt) -> bool {
        *new_ub >= self.ub[idx]
    }
}

/// Synthesize the BII with the bitwise greedy strategy (Algorithm 4) and
/// the boundary-limit bounded leap (Algorithm 5). Solver-call count is
/// bounded by `2·Σ bwᵢ + 1` queries (Theorem 5.3) instead of the linear
/// search's `O(Σ 2^bwᵢ)`.
///
/// Return contract:
/// - `Some(tpl)` — the BII (every bit position decided, or a bounded-leap
///   UNSAT proving no tighter inductive template exists), OR a PARTIAL
///   inductive invariant: on budget exhaustion, solver `unknown`/error,
///   or a witness-parse failure after at least one refinement was
///   adopted, the last adopted template is returned — the paper's
///   any-time property (§4, remark after Algorithm 2). Only best-ness
///   is forfeited: every adopted witness passed the ∃∀ query carrying
///   both the Init and the Inductiveness obligation.
/// - `None` — nothing was adopted before the failure (the result would
///   be the uninformative ⊤), or a query encoding failed.
///
/// Pointers advance ONLY when the previous refine left `cur` unchanged
/// (UNSAT) — Algorithm 4's `A = A_last` test, realized as the
/// `prev_unsat` flag. After SAT the witness changed `cur`, so the new
/// bits may still need examination.
pub(crate) fn synthesize_bitwise_bii(
    solver: &SmtSolver,
    vars: &[Symbol],
    init: &[LoopInstr],
    body: &[LoopInstr],
    bit_widths: &[u8],
    signed: &[bool],
    max_queries: usize,
    use_bv: bool,
) -> Option<BiiTemplate> {
    let mut cur = BiiTemplate::new(vars.len(), bit_widths, signed);
    let mut pos = BitPositions::new(&cur);
    let mut limits = BoundaryLimits::new(&cur);
    let mut calls = 0usize;
    // Whether the previous refine step left `cur` unchanged (= UNSAT).
    // Only then do position pointers advance (Algorithm 4 line 7-8).
    let mut prev_unsat = false;
    // Any-time partial invariant: whether at least one refinement was
    // adopted (see the budget-exhaustion return below).
    let mut adopted = false;

    loop {
        if calls >= max_queries {
            // Budget exhausted: `cur` is inductive by construction — every
            // adopted witness passed the ∃∀ query carrying BOTH the Init and
            // the Inductiveness obligation. Return the PARTIAL invariant (the
            // any-time property of Zuo et al. (2026), §4 remark after Algorithm 2); `None`
            // only when nothing was adopted (the result would be the
            // uninformative ⊤).
            return if adopted { Some(cur) } else { None };
        }

        if prev_unsat {
            pos.advance();
        }
        prev_unsat = false;

        let candidates = propose_bitwise(&cur, &pos, &limits);
        if candidates.is_empty() {
            // EXHAUSTED pointers (every bit passed — the true
            // termination) vs WINDOW-INVALID proposals: an empty
            // candidate set can mean bits remain whose hypotheses are
            // window-invalid (every candidate was filtered by the l ≤ u
            // pre-check). Propose of Zuo et al. (2026) emits ALL bit hypotheses
            // and lets the SOLVER reject impossible ones (the l ≤ u
            // conjunct makes the disjunct trivially false → UNSAT →
            // pointers advance, Lemma 5.2); the Rust pre-filter
            // short-circuits that path, so advance while any pointer is
            // live, return only when every position is exhausted.
            //
            // Candidates can be empty for three reasons:
            //
            // 1. every bit position has already been exhausted;
            // 2. every generated bit hypothesis is window-invalid (`l > u`);
            // 3. every generated bit hypothesis was pruned by BoundaryLimits.
            //
            // In case 3, advancement is still sound:
            //
            // - a pruned lower-bound hypothesis has `new_lb > limits.lb[idx]`,
            //   so even the smallest value with this bit set is too large;
            //   the corresponding BII bit must be 0;
            //
            // - a pruned upper-bound hypothesis has `new_ub < limits.ub[idx]`,
            //   so even the largest value with this bit cleared is too small;
            //   the corresponding BII bit must be 1.
            //
            // Therefore passing the current positions preserves Lemma 5.2's
            // bit-preservation and progress argument.
            if pos.lpos.iter().chain(pos.upos.iter()).all(|&p| p < 0) {
                return Some(cur); // every bit position passed — BII reached.
            }
            pos.advance();
            continue;
        }

        // Refine step 1: ∃∀ query without get-model.
        // Encoding failure: if witnesses were already adopted, return the
        // partial invariant rather than discarding it (any-time property).
        let Some(query) =
            build_refine_query(&candidates, init, body, false, use_bv, bit_widths, signed)
        else {
            return if adopted { Some(cur) } else { None };
        };
        calls += 1;

        match solver.run_raw_query(&query) {
            RawQueryOutcome::Unsat => {
                // No candidate at these bit positions is inductive — prune
                // the boundary limits (A stays, so the next iteration
                // advances the position pointers).
                prev_unsat = true;
                limits.prune_unsat(&cur, &candidates);

                // Bounded leap (Alg 5): with a non-empty under-approximation,
                // a single query can jump straight to the BII in the bounded
                // region — SAT adopts the witness, UNSAT proves A is the BII.
                if limits.is_active() {
                    if calls >= max_queries {
                        return if adopted { Some(cur) } else { None };
                    }
                    let Some(leap) = build_bounded_leap_query(
                        &cur, &limits, init, body, false, use_bv, bit_widths, signed,
                    ) else {
                        return if adopted { Some(cur) } else { None };
                    };
                    calls += 1;
                    match solver.run_raw_query(&leap) {
                        RawQueryOutcome::Unsat => return Some(cur),
                        RawQueryOutcome::Sat(_) => {
                            if calls >= max_queries {
                                return if adopted { Some(cur) } else { None };
                            }
                            let Some(leap_m) = build_bounded_leap_query(
                                &cur, &limits, init, body, true, use_bv, bit_widths, signed,
                            ) else {
                                return if adopted { Some(cur) } else { None };
                            };
                            calls += 1;
                            match solver.run_raw_query(&leap_m) {
                                RawQueryOutcome::Sat(model) => {
                                    match parse_witness(&model, &cur.rows, use_bv) {
                                        Some(bounds) => {
                                            apply_bounds(&mut cur, &bounds);
                                            limits.tighten_sat(&bounds);
                                            adopted = true;
                                            // After a successful leap the pointers
                                            // should advance past the rejected
                                            // directions (Algorithm 5 line 12).
                                            prev_unsat = true;
                                        }
                                        None => {
                                            eprintln!(
                                                "witness parse failure in leap — synthesis bug"
                                            );
                                            return if adopted { Some(cur) } else { None };
                                        }
                                    }
                                }
                                RawQueryOutcome::Unknown
                                | RawQueryOutcome::Error(_)
                                | RawQueryOutcome::Unsat => {
                                    return if adopted { Some(cur) } else { None };
                                }
                            }
                        }
                        RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                            return if adopted { Some(cur) } else { None };
                        }
                    }
                }
            }
            RawQueryOutcome::Sat(_) => {
                // Refine step 2: get model.
                if calls >= max_queries {
                    return if adopted { Some(cur) } else { None };
                }
                let Some(query_model) =
                    build_refine_query(&candidates, init, body, true, use_bv, bit_widths, signed)
                else {
                    return if adopted { Some(cur) } else { None };
                };
                calls += 1;
                match solver.run_raw_query(&query_model) {
                    RawQueryOutcome::Sat(model) => {
                        match parse_witness(&model, &cur.rows, use_bv) {
                            Some(bounds) => {
                                apply_bounds(&mut cur, &bounds);
                                limits.tighten_sat(&bounds);
                                adopted = true;
                                // SAT changed cur — do NOT advance pointers.
                                // prev_unsat stays false.
                            }
                            None => {
                                eprintln!("witness parse failure in refine — synthesis bug");
                                return if adopted { Some(cur) } else { None };
                            }
                        }
                    }
                    RawQueryOutcome::Unknown
                    | RawQueryOutcome::Error(_)
                    | RawQueryOutcome::Unsat => {
                        return if adopted { Some(cur) } else { None };
                    }
                }
            }
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                return if adopted { Some(cur) } else { None };
            }
        }
    }
}

/// Apply witness bounds to the current template (row-wise).
fn apply_bounds(cur: &mut BiiTemplate, bounds: &[(BigInt, BigInt)]) {
    for (idx, (lb, ub)) in bounds.iter().enumerate() {
        cur.rows[idx].lb = lb.clone();
        cur.rows[idx].ub = ub.clone();
    }
}

/// SMT-LIB2 bit-vector constant literal `(_ bvN W)`. Negative values are
/// encoded as `(bvneg (_ bv|N| W))` (QF_BV has no signed literals).
fn bv_const(val: &BigInt, bw: u32) -> String {
    // Normalize to the declared width (two's complement): SMT-LIB requires
    // `(_ bvN W)` with 0 ≤ N < 2^W. Negative values map to their
    // two's-complement representative (e.g. -5 @ 8 bits → 251), and
    // out-of-width constants are reduced modulo 2^bw.
    let modulus = BigInt::one() << bw as usize;
    let repr = ((val % &modulus) + &modulus) % &modulus;
    format!("(_ bv{} {})", repr, bw)
}

/// Encode the loop-body transition as SEQUENTIAL (SSA-style) constraints.
///
/// Each assignment reads the CURRENT value of its operands through
/// intermediate variables, so `x = x + 1; y = x;` gives `y` the NEW `x`
/// (the old parallel encoding `(= xp_y x_x)` used the PRE-state value,
/// mis-modeling sequential read-after-write — unsound). The intermediate
/// variables are quantified in the enclosing `forall` (definitional
/// equalities force them to the unique sequential value).
///
/// In `bv` mode the arithmetic is modular (`bvadd`/`bvsub` with the
/// variable's bit-width) — the paper's bit-vector wrap-around semantics —
/// instead of unbounded LIA `+`/`-`.
///
/// Returns `(intermediate_var_names, transition_formula)`, or `None` if
/// an unsupported `LoopInstr` variant is encountered (fail-closed).
fn encode_sequential_transition(
    body: &[LoopInstr],
    n: usize,
    bv: bool,
    bws: &[u8],
) -> Option<(Vec<String>, String)> {
    let mut cur: Vec<String> = (0..n).map(|i| format!("x_{i}")).collect();
    let mut inter: Vec<String> = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    let mut step = 0usize;
    for instr in body {
        match instr {
            LoopInstr::AddVar(i, c) => {
                let name = format!("xs_{}_{}", step, i);
                step += 1;
                if bv {
                    let c_big = BigInt::from(*c);
                    parts.push(format!(
                        "(= {} (bvadd {} {}))",
                        name,
                        cur[*i],
                        bv_const(&c_big, bws[*i] as u32)
                    ));
                } else {
                    parts.push(format!("(= {} (+ {} {}))", name, cur[*i], c));
                }
                cur[*i] = name.clone();
                inter.push(name);
            }
            LoopInstr::ConstVar(i, c) => {
                let name = format!("xs_{}_{}", step, i);
                step += 1;
                if bv {
                    let c_big = BigInt::from(*c);
                    parts.push(format!("(= {} {})", name, bv_const(&c_big, bws[*i] as u32)));
                } else {
                    parts.push(format!("(= {} {})", name, c));
                }
                cur[*i] = name.clone();
                inter.push(name);
            }
            LoopInstr::CopyVar(i, j) => {
                let name = format!("xs_{}_{}", step, i);
                step += 1;
                parts.push(format!("(= {} {})", name, cur[*j]));
                cur[*i] = name.clone();
                inter.push(name);
            }
            // Guards are handled separately in the query builder — skip.
            // Explicit match (not `_ => {}`) so that future LoopInstr
            // variants cause a compile error rather than silent omission.
            LoopInstr::TestLe(..) | LoopInstr::TestGe(..) | LoopInstr::TestDiffLe(..) => {}
            // The OLD LoopInstr path (sequential SSA
            // transition) does not support `if` — fail closed so the
            // checker falls back (the BiiLoopProblem path handles `if`
            // via `Ite`).
            LoopInstr::If(..) => return None,
            // The OLD LoopInstr path (sequential SSA
            // transition) does not support saturating arithmetic — fail
            // closed so the checker falls back.
            LoopInstr::AddSat(..) => return None,
        }
    }
    for i in 0..n {
        parts.push(format!("(= xp_{} {})", i, cur[i]));
    }
    let trans = if parts.len() > 1 {
        format!("(and {})", parts.join(" "))
    } else if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        "true".to_string()
    };
    Some((inter, trans))
}

/// The formula `A'(X)` (or `A'(X')` when `primed`) for a template: the
/// conjunction `⋀_r l_r ≤ f_r(X) ≤ u_r` over every row — using the
/// WITNESS bound constants `l_r`/`u_r` (declared by the caller), so the
/// inductiveness formula constrains exactly the existential part of the
/// `Refine` query.
///
/// In BV mode the row term `f_r(X)` is zero-extended to the row's ENCODED
/// width `enc_bw` (Diff/Sum rows need `bw + 1` bits, and mixed-width
/// operands must share a sort before `bvsub`/`bvadd`), and the comparison
/// happens in the ENCODED (offset) domain: Diff rows shift the signed
/// range `[-m, m]` onto `[0, 2m]` so that `l_r ≤ f + m ≤ u_r` is a plain
/// unsigned comparison (the paper's unsigned interpretation).
/// The SMT encoding of `clamp(x + c)` for a saturating assignment
/// (a saturating update is segmented — the clamped successor lives in
/// the template's Clamp rows). The clamp is piecewise-linear, encoded
/// with nested `ite`s; MIN/MAX are the variable's type range. `x` is
/// the value being clamped: the CURRENT `x_i` for a successor's
/// clamped value (`clamp(x_i + c)` is the true transfer), or the
/// PRIMED `xp_i` when constraining the next state's own clamped
/// successor.
fn clamp_expr(x: &str, c: i128, bv: bool, bw: u8, signed: bool) -> String {
    let (min, max) = if signed {
        let half = BigInt::one() << (bw as usize - 1);
        (BigInt::zero() - half.clone(), half - 1)
    } else {
        (BigInt::zero(), (BigInt::one() << bw as usize) - 1)
    };
    if bv {
        // The clamp bounds compare the ADDITION-PRE operand, never the
        // `bvadd` result: `bvadd` wraps mod 2^bw, so `clamp(x + c)` must
        // decide on x alone (`x > MAX−c` ⟹ saturate up, `x < MIN−c` ⟹
        // saturate down).  Comparing the WRAPPED sum misjudges the
        // boundary (UInt8 x=255, c=1: bvadd = 0, neither comparison
        // fires, successor 0 instead of 255; Int8 x=127, c=1: bvadd =
        // −128, successor −128 instead of 127).  `c` is a compile-time
        // constant, so the per-sign comparison constant is representable:
        // for c > 0 only the upper bound can be crossed (MAX−c stays in
        // range), for c < 0 only the lower (MIN−c stays in range).
        let c_big = BigInt::from(c);
        let abs_c = c_big.abs();
        let span = &max - &min; // 2^bw - 1
        let c_bv = bv_const(&BigInt::from(c), bw as u32);
        let min_c = bv_const(&min, bw as u32);
        let max_c = bv_const(&max, bw as u32);

        if abs_c > span {
            return if c > 0 { max_c } else { min_c };
        }

        let (gt, lt) = if signed {
            ("bvsgt", "bvslt")
        } else {
            ("bvugt", "bvult")
        };
        let add = format!("(bvadd {x} {c_bv})");
        if c > 0 {
            let limit = bv_const(&(&max - BigInt::from(c)), bw as u32);
            format!("(ite ({gt} {x} {limit}) {max_c} {add})")
        } else if c < 0 {
            let limit = bv_const(&(&min - BigInt::from(c)), bw as u32);
            format!("(ite ({lt} {x} {limit}) {min_c} {add})")
        } else {
            x.to_string() // c == 0: clamp(x + 0) = x — no boundary crossed.
        }
    } else {
        let add = format!("(+ {x} {c})");
        format!("(ite (> {add} {max}) {max} (ite (< {add} {min}) {min} {add}))")
    }
}

fn template_formula(
    rows: &[BiiRow],
    vars_len: usize,
    primed: bool,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
) -> String {
    let var = |i: usize| -> String {
        // The first `vars_len` indices are loop variables (x_i / xp_i
        // generations); the rest are external-symbol parameters (read-only:
        // always named n_{..}, no primed version).
        if i < vars_len {
            if primed {
                format!("xp_{i}")
            } else {
                format!("x_{i}")
            }
        } else {
            format!("n_{}", i - vars_len)
        }
    };
    // Per-operand signedness: the row-level `signed` flag is the OR
    // of the operands and cannot drive extension for MIXED rows — a wrong
    // choice misreads the operand by ±2^from and shifts the row value,
    // misaligning the offset-domain window in BOTH directions (false
    // violations block synthesis; false satisfactions are UNSOUND).
    let signed_of = |v: usize| signed.get(v).copied().unwrap_or(false);
    // The clamped successor of a saturating variable —
    // `x_i := x_i +? c`, recovered from the Clamp rows.
    let clamp_of = |i: usize| -> Option<(i128, bool)> {
        rows.iter().find_map(|r| match r.kind {
            RowKind::Clamp(j, c) if j == i => Some((c, r.signed)),
            _ => None,
        })
    };
    let mut parts = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let f = match row.kind {
            RowKind::Interval(i) => {
                // A saturating variable's PRIMED successor is the clamped
                // value `clamp(x_i + c)` (the true transfer), not the raw
                // `x_i + c`. The clamp uses the CURRENT `x_i`.
                if primed {
                    if let Some((c, signed)) = clamp_of(i) {
                        clamp_expr(&format!("x_{i}"), c, bv, bws[i], signed)
                    } else {
                        var(i)
                    }
                } else {
                    var(i)
                }
            }
            RowKind::Clamp(i, c) => {
                // The row constrains the clamped value of the state the
                // row lives in: the CURRENT state's successor for `A(x)`,
                // the NEXT state's own successor for `A(xp)` (primed).
                clamp_expr(&var(i), c, bv, row.bw, row.signed)
            }
            RowKind::Diff(i, j) => {
                if bv {
                    let ebw = row.enc_bw();
                    // Sign-extend the operands of a signed (uniform)
                    // row; zero-extend unsigned ones — byte-identical to
                    // the old emission for unsigned rows.
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    format!("(bvsub {xi} {xj})")
                } else {
                    format!("(- {} {})", var(i), var(j))
                }
            }
            RowKind::Sum(i, j) => {
                if bv {
                    // As Diff — sign-extend signed rows, zero-extend
                    // unsigned ones (byte-identical for unsigned).
                    let ebw = row.enc_bw();
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    format!("(bvadd {xi} {xj})")
                } else {
                    format!("(+ {} {})", var(i), var(j))
                }
            }
            RowKind::Support3(i, j, k, sj, sk) => {
                if bv {
                    // Three operands, extended per `row.signed`
                    // (sign- or zero-), joined by `bvadd`/`bvsub` per the
                    // row's signs; the result is `bw + 2` bits, matching
                    // the bound constants.
                    let ebw = row.enc_bw();
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    let xk = ext_operand(&var(k), bws[k] as u32, ebw, signed_of(k));
                    let op_j = if sj { "bvadd" } else { "bvsub" };
                    let op_k = if sk { "bvadd" } else { "bvsub" };
                    format!("({op_k} ({op_j} {xi} {xj}) {xk})")
                } else {
                    let op_j = if sj { "+" } else { "-" };
                    let op_k = if sk { "+" } else { "-" };
                    format!("({op_k} ({op_j} {} {}) {})", var(i), var(j), var(k))
                }
            }
        };
        if bv {
            // Compare in the encoded domain: `f + offset` is non-negative
            // and fits `enc_bw` bits (Interval/Sum rows have offset 0).
            let off = row.offset();
            let f = if off.is_zero() {
                f
            } else {
                format!("(bvadd {} {})", f, bv_const(&off, row.enc_bw()))
            };
            // Signed Interval rows compare in the TRUE
            // signed domain (`bvsle`/`bvsge` on the raw bound); all other
            // rows compare in the encoded (offset) domain with unsigned
            // order. Clamp rows are signed like their
            // variable.
            if row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
                parts.push(format!("(bvsle l_{r} {f})"));
                parts.push(format!("(bvsle {f} u_{r})"));
            } else {
                parts.push(format!("(bvule l_{r} {f})"));
                parts.push(format!("(bvule {f} u_{r})"));
            }
        } else {
            parts.push(format!("(<= l_{r} {f})"));
            parts.push(format!("(<= {f} u_{r})"));
        }
    }
    if parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", parts.join(" "))
    }
}

/// Concrete-bound template formula: `⋀_r lb_r ≤ f_r(X) ≤ ub_r` using the
/// `BiiRow`'s actual bound values (BV mode compares in the offset-encoded
/// domain). Used by the independent verifier
/// (`verify_template_against_problem`) — synthesis queries use the
/// witness-constant `template_formula` (`l_r`/`u_r` constrained by the
/// candidate disjunction). Infallible: every row kind encodes
/// unconditionally, so the plain `String` return carries no `Option`.
fn template_formula_concrete(
    rows: &[BiiRow],
    vars_len: usize,
    primed: bool,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
) -> String {
    let var = |i: usize| -> String {
        if i < vars_len {
            if primed {
                format!("xp_{i}")
            } else {
                format!("x_{i}")
            }
        } else {
            format!("n_{}", i - vars_len)
        }
    };
    // Per-operand signedness — see template_formula for the
    // rationale: the row-level flag is the OR and cannot drive extension
    // for mixed rows.
    let signed_of = |v: usize| signed.get(v).copied().unwrap_or(false);
    // The clamped successor of a saturating variable
    // (see template_formula).
    let clamp_of = |i: usize| -> Option<(i128, bool)> {
        rows.iter().find_map(|r| match r.kind {
            RowKind::Clamp(j, c) if j == i => Some((c, r.signed)),
            _ => None,
        })
    };
    let mut parts = Vec::new();
    for row in rows {
        let f = match row.kind {
            RowKind::Interval(i) => {
                if primed {
                    if let Some((c, signed)) = clamp_of(i) {
                        clamp_expr(&format!("x_{i}"), c, bv, bws[i], signed)
                    } else {
                        var(i)
                    }
                } else {
                    var(i)
                }
            }
            RowKind::Clamp(i, c) => clamp_expr(&var(i), c, bv, row.bw, row.signed),
            RowKind::Diff(i, j) => {
                if bv {
                    let ebw = row.enc_bw();
                    // Sign-extend the operands of a signed (uniform)
                    // row; zero-extend unsigned ones — byte-identical to
                    // the old emission for unsigned rows.
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    format!("(bvsub {xi} {xj})")
                } else {
                    format!("(- {} {})", var(i), var(j))
                }
            }
            RowKind::Sum(i, j) => {
                if bv {
                    // As Diff — sign-extend signed rows, zero-extend
                    // unsigned ones (byte-identical for unsigned).
                    let ebw = row.enc_bw();
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    format!("(bvadd {xi} {xj})")
                } else {
                    format!("(+ {} {})", var(i), var(j))
                }
            }
            RowKind::Support3(i, j, k, sj, sk) => {
                if bv {
                    // Three operands, extended per `row.signed`
                    // (sign- or zero-), joined by `bvadd`/`bvsub` per the
                    // row's signs; the result is `bw + 2` bits, matching
                    // the bound constants.
                    let ebw = row.enc_bw();
                    let xi = ext_operand(&var(i), bws[i] as u32, ebw, signed_of(i));
                    let xj = ext_operand(&var(j), bws[j] as u32, ebw, signed_of(j));
                    let xk = ext_operand(&var(k), bws[k] as u32, ebw, signed_of(k));
                    let op_j = if sj { "bvadd" } else { "bvsub" };
                    let op_k = if sk { "bvadd" } else { "bvsub" };
                    format!("({op_k} ({op_j} {xi} {xj}) {xk})")
                } else {
                    let op_j = if sj { "+" } else { "-" };
                    let op_k = if sk { "+" } else { "-" };
                    format!("({op_k} ({op_j} {} {}) {})", var(i), var(j), var(k))
                }
            }
        };
        if bv {
            let off = row.offset();
            let f = if off.is_zero() {
                f
            } else {
                format!("(bvadd {} {})", f, bv_const(&off, row.enc_bw()))
            };
            let lb = bv_const(&(&row.lb + &off), row.enc_bw());
            let ub = bv_const(&(&row.ub + &off), row.enc_bw());
            // Signed Interval rows compare in the TRUE
            // signed domain (bvsle/bvsge); the two's-complement bound
            // pattern is interpreted signed.
            // Signed Interval AND Clamp rows compare
            // in the TRUE signed domain (bvsle/bvsge); the two's-
            // complement bound pattern is interpreted signed.
            if row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
                parts.push(format!("(bvsle {lb} {f})"));
                parts.push(format!("(bvsle {f} {ub})"));
            } else {
                parts.push(format!("(bvule {lb} {f})"));
                parts.push(format!("(bvule {f} {ub})"));
            }
        } else {
            parts.push(format!("(<= {} {f})", row.lb));
            parts.push(format!("(<= {f} {})", row.ub));
        }
    }
    if parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", parts.join(" "))
    }
}

/// Build the SMT-LIB2 query for `Refine`:
/// `∃A'.(⋁_c A'⊑c) ∧ ∀X,X'.P(A',X,X')`, where `P` = Init ∧ Inductiveness.
/// `get_model` appends `(get-model)` — issued only on a separate query
/// after a sat outcome (z3 errors on `(get-model)` after `unsat`).
///
/// Returns `None` if the transition encoding fails (fail-closed).
fn build_refine_query(
    candidates: &[BiiTemplate],
    init: &[LoopInstr],
    body: &[LoopInstr],
    get_model: bool,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
) -> Option<String> {
    let n = candidates[0].n_vars;
    let rows = &candidates[0].rows;

    // Per-variable sort: LIA `Int` or QF_BV `(_ BitVec W)`.
    // Uses enc_bw() so Diff/Sum rows get bw+1 bits.
    let sort = |bw: u32| -> String {
        if bv {
            format!("(_ BitVec {})", bw)
        } else {
            "Int".to_string()
        }
    };
    // `off` maps a TEMPLATE-domain bound to the ENCODED domain used by
    // the BV sorts (Diff rows shift the signed range `[-m, m]` onto
    // `[0, 2m]`). LIA has no width, so the offset is ignored there.
    let lit = |v: &BigInt, bw: u32, off: &BigInt| -> String {
        if bv {
            bv_const(&(v + off), bw)
        } else {
            v.to_string()
        }
    };
    let ge = if bv { "bvuge" } else { ">=" };
    let le = if bv { "bvule" } else { "<=" };

    let mut smt = String::new();
    if bv {
        smt.push_str("(set-logic BV)\n");
    } else {
        smt.push_str("(set-logic LIA)\n");
    }
    // Bound constants for the witness template A' (the existential part).
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        smt.push_str(&format!(
            "(declare-const l_{r} {})\n(declare-const u_{r} {})\n",
            sort(ebw),
            sort(ebw)
        ));
    }
    // Candidate disjunction: A' ⊑ c for some candidate c — A' is at least
    // as tight as c in every row, and never empty (l ≤ u). Bounds are
    // emitted in the ENCODED domain (Diff rows offset).
    emit_candidate_disjunction(&mut smt, candidates, &lit, bv);

    // The inductiveness formula ∀X,X'.P(A',X,X') with A'(X) = ⋀_r l_r ≤
    // f_r(X) ≤ u_r. The transition is SEQUENTIAL (SSA) — compute it first
    // so its intermediate variables can be quantified alongside X and X'.
    let (inter, trans) = encode_sequential_transition(body, n, bv, bws)?;

    smt.push_str("(assert (forall (");
    for i in 0..n {
        smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
    }
    for i in 0..n {
        smt.push_str(&format!("(xp_{i} {}) ", sort(bws[i] as u32)));
    }
    for name in &inter {
        let var_idx = name
            .strip_prefix("xs_")
            .and_then(|s| s.split('_').nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        smt.push_str(&format!("({} {}) ", name, sort(bws[var_idx] as u32)));
    }
    smt.push_str(")\n");

    // A'(X): the template rows over the current-state variables.
    let a_x = template_formula(rows, n, false, bv, bws, signed);
    let a_xp = template_formula(rows, n, true, bv, bws, signed);

    // Pre(X): the seeded pre-state (`i = c`). Any non-constant init
    // instruction cannot be encoded as a plain seed — fail closed rather
    // than silently weakening the precondition (matches
    // `encode_sequential_transition`'s policy).
    let mut pre_parts = Vec::new();
    for instr in init {
        match instr {
            LoopInstr::ConstVar(i, c) => {
                let c_big = BigInt::from(*c);
                pre_parts.push(format!(
                    "(= x_{i} {})",
                    lit(&c_big, bws[*i] as u32, &BigInt::zero())
                ));
            }
            _ => return None,
        }
    }
    let pre = if pre_parts.is_empty() {
        "true".to_string()
    } else if pre_parts.len() == 1 {
        pre_parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", pre_parts.join(" "))
    };

    // G(X): the guard — every TestLe/TestDiffLe in `body`. Under BV,
    // guards encode via the value-preserving lift (`encode_cmp_bv` /
    // `lift_cmp_operand`):
    // each operand is lifted to a signed-faithful representation,
    // sign-extended to a common width, and compared with SIGNED
    // predicates — the same encoding used by the verification side
    // (`verify_loop_decreases` in the checker, which delegates to
    // `cond_to_smt`).
    let mut guard_parts = Vec::new();
    for instr in body {
        match instr {
            LoopInstr::TestLe(i, c) => {
                if bv {
                    use crate::hir::loop_ir::ScalarExpr;
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Le,
                        &ScalarExpr::Var(*i),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(<= x_{i} {})", c));
                }
            }
            LoopInstr::TestGe(i, c) => {
                if bv {
                    use crate::hir::loop_ir::ScalarExpr;
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Ge,
                        &ScalarExpr::Var(*i),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(>= x_{i} {})", c));
                }
            }
            LoopInstr::TestDiffLe(i, j, c) => {
                if bv {
                    use crate::hir::loop_ir::{ArithSem, ScalarExpr};
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Le,
                        &ScalarExpr::Sub(
                            Box::new(ScalarExpr::Var(*i)),
                            Box::new(ScalarExpr::Var(*j)),
                            ArithSem::Wrap,
                        ),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(<= (- x_{i} x_{j}) {})", c));
                }
            }
            LoopInstr::AddVar(..) | LoopInstr::ConstVar(..) | LoopInstr::CopyVar(..) => {}
            // `if` conditions are path conditions, not loop-header
            // guards — skipped here.
            LoopInstr::If(..) => {}
            // `+?`/`-?` assignments are ignored for guard collection
            // (the transfer encoding fails closed on them).
            LoopInstr::AddSat(..) => {}
        }
    }
    let guard = if guard_parts.is_empty() {
        "true".to_string()
    } else if guard_parts.len() == 1 {
        guard_parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", guard_parts.join(" "))
    };

    // T(X,X') is the SEQUENTIAL (SSA) transition computed above via
    // `encode_sequential_transition` (its intermediate variables were
    // quantified in the `forall` binder). `trans` is used verbatim below.
    // Init: Pre(X) → A'(X). Inductiveness:
    // A'(X) ∧ G(X) ∧ T(X,X') → A'(X').
    smt.push_str(&format!(
        "  (and (=> {} {}) (=> (and {} {} {}) {}))\n",
        pre, a_x, a_x, guard, trans, a_xp
    ));
    smt.push_str("))\n");
    smt.push_str("(check-sat)\n");
    if get_model {
        smt.push_str("(get-model)\n");
    }
    Some(smt)
}

// ── BiiLoopProblem synthesis entry  ──

/// Emit the candidate disjunction `(assert (or ⋁_c ⋀_r A'⊑c))` — A' is at
/// least as tight as some candidate c in every row, and never empty
/// (l ≤ u); bounds are emitted in the ENCODED domain (Diff rows offset).
/// Shared by the LoopInstr and BiiLoopProblem refine builders to avoid
/// encoding drift between the two query paths.
/// The bound comparison operators for a row: signed
/// Interval rows compare in the TRUE signed domain (`bvsge`/`bvsle`);
/// every other row compares in the encoded (offset) domain with unsigned
/// order. LIA always uses the mathematical operators.
fn row_ge_le(row: &BiiRow, bv: bool) -> (&'static str, &'static str) {
    if bv && row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
        ("bvsge", "bvsle")
    } else if bv {
        ("bvuge", "bvule")
    } else {
        (">=", "<=")
    }
}

/// Strict versions of `row_ge_le` (for the STRICT tightening clause).
fn row_gt_lt(row: &BiiRow, bv: bool) -> (&'static str, &'static str) {
    if bv && row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
        ("bvsgt", "bvslt")
    } else if bv {
        ("bvugt", "bvult")
    } else {
        (">", "<")
    }
}

fn emit_candidate_disjunction(
    smt: &mut String,
    candidates: &[BiiTemplate],
    lit: &dyn Fn(&BigInt, u32, &BigInt) -> String,
    bv: bool,
) {
    smt.push_str("(assert (or");
    for c in candidates {
        smt.push_str(" (and");
        for (r, row) in c.rows.iter().enumerate() {
            let ebw = row.enc_bw();
            let off = row.offset();
            let (rge, rle) = row_ge_le(row, bv);
            smt.push_str(&format!(
                " ({} l_{r} {}) ({} u_{r} {})",
                rge,
                lit(&c.rows[r].lb, ebw, &off),
                rle,
                lit(&c.rows[r].ub, ebw, &off)
            ));
            smt.push_str(&format!(" ({} l_{r} u_{r})", rle));
        }
        smt.push_str(")");
    }
    smt.push_str("))\n");
}

/// Under LIA, external-symbol params are unbounded
/// `Int` — constrain each to its type range (unsigned [0, 2^bw−1] /
/// signed symmetric) so template rows referencing them can converge. BV
/// mode needs no constraint (params are bounded bit-vectors). Returns the
/// conjunction of range conditions, to be spliced into each ∀ body's
/// antecedent — params are quantifier-scoped variables, so a top-level
/// `assert` would reference an undeclared `n_j`.
fn param_range_conds(params: &[crate::hir::loop_ir::BiiVar], bv: bool) -> String {
    if bv || params.is_empty() {
        return "true".to_string();
    }
    let mut parts = Vec::new();
    for (j, p) in params.iter().enumerate() {
        // The template's Interval row for a signed
        // param carries TRUE signed top bounds, so the type-range
        // condition uses the signed range again (the unsigned workaround
        // is no longer needed — the two domains agree). Unsigned params
        // keep `[0, 2^bw−1]`.
        let (lo, hi) = if p.signed {
            let half = BigInt::one() << (p.bw as usize - 1);
            (-half.clone(), half - 1)
        } else {
            (BigInt::zero(), (BigInt::one() << p.bw as usize) - 1)
        };
        parts.push(format!("(and (<= {} n_{j}) (<= n_{j} {}))", lo, hi));
    }
    if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", parts.join(" "))
    }
}

/// Encode a `ScalarExpr` as an SMT term, returning the term and its
/// bit-width (BV mode; LIA ignores widths). `vars_len` separates loop
/// variables (x_i) from external-symbol parameters (n_{i-vars_len});
/// `ctx_bw` is the fallback width for constants. Unsupported shapes return
/// None (fail-closed).
fn expr_to_smt_bw(
    e: &crate::hir::loop_ir::ScalarExpr,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
    vars_len: usize,
    ctx_bw: u32,
) -> Option<(String, u32)> {
    use crate::hir::loop_ir::ScalarExpr as E;
    match e {
        E::Var(i) => {
            // `i < vars_len` → loop variable x_i; otherwise → parameter
            // n_{i - vars_len} (the template-domain index space is
            // vars ++ params).
            let name = if *i < vars_len {
                format!("x_{i}")
            } else {
                format!("n_{}", i - vars_len)
            };
            let bw = bws.get(*i).map(|b| *b as u32).unwrap_or(ctx_bw);
            Some((name, bw))
        }
        E::Const(c) => Some((
            if bv {
                bv_const(c, ctx_bw)
            } else {
                c.to_string()
            },
            ctx_bw,
        )),
        E::Add(l, r, sem) => {
            // Saturating arithmetic is not expressible as a plain `+`
            // (the clamp is piecewise — it lives in the Clamp rows and
            // the transfer's dedicated `clamp_expr` path in
            // `encode_edge_inductiveness`).  Any OTHER Saturate shape
            // reaching the generic encoder (nested expressions, guards,
            // init scalars) fails closed — same defense as
            // `lin_of_scalar`.
            if *sem == crate::hir::loop_ir::ArithSem::Saturate {
                return None;
            }
            let (ls, lbw) = expr_to_smt_bw(l, bv, bws, signed, vars_len, ctx_bw)?;
            let (rs, rbw) = expr_to_smt_bw(r, bv, bws, signed, vars_len, ctx_bw)?;
            if bv {
                // NB: the zext+bvadd/bvsub here encodes WRAP-semantics
                // transition arithmetic — modular arithmetic is
                // extension-agnostic at a FIXED width, so uniform zext is
                // correct for the (wrapped) value. Comparison faithfulness
                // is handled separately by lift_cmp_operand (the value-preserving lift); do
                // NOT route transition arithmetic through the lift.
                // Mixed-width operands must share a sort: zero-extend both
                // to the common width (modular `bvadd` on the widened
                // operands stays exact for the wrap-around semantics).
                let bw = lbw.max(rbw);
                Some((
                    format!(
                        "(bvadd {} {})",
                        zext_term(ls, lbw, bw),
                        zext_term(rs, rbw, bw)
                    ),
                    bw,
                ))
            } else {
                Some((format!("(+ {ls} {rs})"), ctx_bw))
            }
        }
        E::Sub(l, r, sem) => {
            if *sem == crate::hir::loop_ir::ArithSem::Saturate {
                return None; // saturating arithmetic — fail closed (see Add).
            }
            let (ls, lbw) = expr_to_smt_bw(l, bv, bws, signed, vars_len, ctx_bw)?;
            let (rs, rbw) = expr_to_smt_bw(r, bv, bws, signed, vars_len, ctx_bw)?;
            if bv {
                let bw = lbw.max(rbw);
                Some((
                    format!(
                        "(bvsub {} {})",
                        zext_term(ls, lbw, bw),
                        zext_term(rs, rbw, bw)
                    ),
                    bw,
                ))
            } else {
                Some((format!("(- {ls} {rs})"), ctx_bw))
            }
        }
        E::Ite(c, t, f) => {
            let (ts, tbw) = expr_to_smt_bw(t, bv, bws, signed, vars_len, ctx_bw)?;
            let (fs, fbw) = expr_to_smt_bw(f, bv, bws, signed, vars_len, ctx_bw)?;
            let cond = cond_to_smt(c, bv, bws, signed, vars_len)?;
            if bv {
                let bw = tbw.max(fbw);
                Some((
                    format!(
                        "(ite {cond} {} {})",
                        zext_term(ts, tbw, bw),
                        zext_term(fs, fbw, bw)
                    ),
                    bw,
                ))
            } else {
                Some((format!("(ite {cond} {ts} {fs})"), ctx_bw))
            }
        }
    }
}

/// Zero-extend `s` from `from` bits to `to` bits (no-op when equal).
fn zext_term(s: String, from: u32, to: u32) -> String {
    if from == to {
        s
    } else {
        format!("((_ zero_extend {}) {s})", to - from)
    }
}

/// Sign-extend `s` from `from` bits to `to` bits (no-op when equal) —
/// the signed-comparison counterpart of `zext_term`.
fn sext_term(s: String, from: u32, to: u32) -> String {
    if from == to {
        s
    } else {
        format!("((_ sign_extend {}) {s})", to - from)
    }
}

/// Extend operand `var_str` from its variable width (`from`) to the row's
/// encoded width (`to`), by the OPERAND'S OWN signedness (`signed` here is
/// the PER-VARIABLE flag passed by the caller, NOT the row's OR):
/// - (a) sign-extension preserves a two's-complement value;
/// - (b) zero-extension preserves an unsigned value.
fn ext_operand(var_str: &str, from: u32, to: u32, signed: bool) -> String {
    debug_assert!(from < to, "extension must widen (enc_bw > variable width)");
    let op = if signed { "sign_extend" } else { "zero_extend" };
    format!("((_ {op} {}) {var_str})", to - from)
}

/// Value-preserving lift (comparison layer). Returns `(term, W)`
/// such that the W-bit TWO'S-COMPLEMENT reading of `term` equals the
/// mathematical value of `e` (all variables within their type ranges) and
/// |value| < 2^(W−1).
///
/// LEMMA (value-preserving lift):
/// (a) a signed variable's pattern is its true value's two's complement —
///     `sext` to any W ≥ bw preserves the reading;
/// (b) an unsigned variable's value t ≤ 2^bw − 1 < 2^bw, so `zext` to
///     bw + 1 yields a pattern with top bit 0 whose SIGNED reading is t.
///
/// WIDTH RULES:
/// - W_d (addend/difference): Add/Sub nodes take W = max(W1, W2) + 1 —
///   one extra bit absorbs carry/borrow: |v1 ± v2| < 2^(W−1) whenever
///   |vi| < 2^(Wi−1). Without it the modular bvsub WRAPS (two same-width
///   signed vars span a difference up to 2^bw − 1, overflowing bw-bit
///   two's complement — the same reason Diff rows encode at bw + 1).
/// - W_c (comparison, applied by `encode_cmp_bv`): W = max of the two
///   lifted widths, both sides SIGN-extended to it, signed predicates.
/// - Constants lift at W = bits(|c|) + 1 (smallest two's-complement fit).
fn lift_cmp_operand(
    e: &crate::hir::loop_ir::ScalarExpr,
    bws: &[u8],
    signed: &[bool],
    vars_len: usize,
) -> Option<(String, u32)> {
    use crate::hir::loop_ir::{ArithSem, ScalarExpr as E};
    match e {
        E::Var(i) => {
            let name = if *i < vars_len {
                format!("x_{i}")
            } else {
                format!("n_{}", i - vars_len)
            };
            let bw = bws.get(*i).map(|b| *b as u32).unwrap_or(64);
            let s = signed.get(*i).copied().unwrap_or(false);
            if s {
                Some((name, bw)) // two's-complement-faithful by definition
            } else {
                Some((zext_term(name, bw, bw + 1), bw + 1)) // LEMMA (b)
            }
        }
        E::Const(c) => {
            if c.abs().bits() > 126 {
                return None; // absurdly wide guard constant — fail closed
            }
            let w = (c.abs().bits() + 1) as u32;
            Some((bv_const(c, w), w))
        }
        E::Add(l, r, sem) => {
            if *sem == ArithSem::Saturate {
                return None; // clamp is piecewise — not a faithful lift
            }
            let (ls, lw) = lift_cmp_operand(l, bws, signed, vars_len)?;
            let (rs, rw) = lift_cmp_operand(r, bws, signed, vars_len)?;
            let w = lw.max(rw) + 1; // WIDTH RULE W_d
            Some((
                format!("(bvadd {} {})", sext_term(ls, lw, w), sext_term(rs, rw, w)),
                w,
            ))
        }
        E::Sub(l, r, sem) => {
            if *sem == ArithSem::Saturate {
                return None;
            }
            let (ls, lw) = lift_cmp_operand(l, bws, signed, vars_len)?;
            let (rs, rw) = lift_cmp_operand(r, bws, signed, vars_len)?;
            let w = lw.max(rw) + 1; // WIDTH RULE W_d
            Some((
                format!("(bvsub {} {})", sext_term(ls, lw, w), sext_term(rs, rw, w)),
                w,
            ))
        }
        // Comparisons over Ite were never produced by the lowering — fail
        // closed rather than guess a semantics.
        E::Ite(..) => None,
    }
}

/// Shared BV comparison encoding (value-preserving lift): lift both sides, sign-extend
/// to the common width, compare with SIGNED predicates. Used by BOTH
/// `cond_to_smt` and the LoopInstr query builders' guard encoding — one
/// source of truth, no drift. Fixes: (1) unsigned diff guards with a
/// negative constant were VACUOUS under the old bvule encoding
/// (`bv_const(−1)` becomes the all-ones pattern and `bvule(·, 2^w−1)` is
/// always true — `i < n` guards were silently dropped); (2) same-width
/// signed differences wrapped (`bvsub` at bw bits wraps −255 to 1);
/// (3) mixed-signedness comparisons had no correct single-flag encoding.
fn encode_cmp_bv(
    op: crate::hir::loop_ir::CmpOp,
    lhs: &crate::hir::loop_ir::ScalarExpr,
    rhs: &crate::hir::loop_ir::ScalarExpr,
    bws: &[u8],
    signed: &[bool],
    vars_len: usize,
) -> Option<String> {
    use crate::hir::loop_ir::CmpOp as O;
    let (ls, lw) = lift_cmp_operand(lhs, bws, signed, vars_len)?;
    let (rs, rw) = lift_cmp_operand(rhs, bws, signed, vars_len)?;
    let w = lw.max(rw); // WIDTH RULE W_c
    let l = sext_term(ls, lw, w);
    let r = sext_term(rs, rw, w);
    Some(match op {
        O::Lt => format!("(bvslt {l} {r})"),
        O::Le => format!("(bvsle {l} {r})"),
        O::Gt => format!("(bvsgt {l} {r})"),
        O::Ge => format!("(bvsge {l} {r})"),
        O::Eq => format!("(= {l} {r})"),
        O::Neq => format!("(not (= {l} {r}))"),
    })
}

/// Encode a `ScalarExpr` as an SMT term (`ctx_bw` fallback width for
/// constants; `vars_len` separates loop variables from params).
fn expr_to_smt(
    e: &crate::hir::loop_ir::ScalarExpr,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
    vars_len: usize,
    ctx_bw: u32,
) -> Option<String> {
    expr_to_smt_bw(e, bv, bws, signed, vars_len, ctx_bw).map(|(s, _)| s)
}

/// Encode a `Cond` as an SMT Boolean. LIA uses the mathematical
/// operators. Under BV, comparisons go through the value-preserving lift
/// (`lift_cmp_operand`): each side is lifted so its
/// W-bit two's-complement reading equals its mathematical value (LEMMA:
/// sext preserves a signed pattern; zext to bw+1 makes an unsigned value
/// a valid non-negative two's-complement pattern), both sides are
/// sign-extended to the common width (W_c) and compared with SIGNED
/// predicates — the mathematical order for EVERY signedness mixture.
/// The `Cond::Cmp.signed` flag is therefore NO LONGER consumed here (it
/// was the OR of the operands' signedness and had no correct value for
/// mixtures); the field stays in the IR for compatibility.
pub(crate) fn cond_to_smt(
    c: &crate::hir::loop_ir::Cond,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
    vars_len: usize,
) -> Option<String> {
    use crate::hir::loop_ir::{CmpOp as O, Cond as C};
    match c {
        C::True => Some("true".to_string()),
        C::False => Some("false".to_string()),
        C::Cmp { op, lhs, rhs, .. } => {
            if bv {
                return encode_cmp_bv(*op, lhs, rhs, bws, signed, vars_len);
            }
            let l = expr_to_smt_bw(lhs, false, bws, signed, vars_len, 64)?.0;
            let r = expr_to_smt_bw(rhs, false, bws, signed, vars_len, 64)?.0;
            let pick = |lt: &str, le: &str, gt: &str, ge: &str| -> String {
                match op {
                    O::Lt => format!("({lt} {l} {r})"),
                    O::Le => format!("({le} {l} {r})"),
                    O::Gt => format!("({gt} {l} {r})"),
                    O::Ge => format!("({ge} {l} {r})"),
                    O::Eq => format!("(= {l} {r})"),
                    O::Neq => format!("(not (= {l} {r}))"),
                }
            };
            Some(pick("<", "<=", ">", ">="))
        }
        C::And(a, b) => Some(format!(
            "(and {} {})",
            cond_to_smt(a, bv, bws, signed, vars_len)?,
            cond_to_smt(b, bv, bws, signed, vars_len)?
        )),
        C::Or(a, b) => Some(format!(
            "(or {} {})",
            cond_to_smt(a, bv, bws, signed, vars_len)?,
            cond_to_smt(b, bv, bws, signed, vars_len)?
        )),
        C::Not(a) => Some(format!(
            "(not {})",
            cond_to_smt(a, bv, bws, signed, vars_len)?
        )),
    }
}

/// Shared edge-wise inductiveness snippet: one back edge →
/// `(=> (and A(X) G(X) [guard] [def] next(X,X')) A(X'))`.
/// Used by both synthesis (`build_refine_query_problem`) and the
/// independent verifier (`verify_template_against_problem`) to
/// avoid encoding drift.
fn encode_edge_inductiveness(
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    tpl: &BiiTemplate,
    edge: &crate::hir::loop_ir::TransitionEdge,
    bv: bool,
    bws: &[u8],
    concrete: bool,
) -> Option<String> {
    use crate::hir::loop_ir::EdgeKind;
    if edge.kind != EdgeKind::Back {
        return None; // only Back edges participate in inductiveness.
    }
    let n = problem.vars.len();
    let signed_all: Vec<bool> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.signed)
        .collect();
    let rows = &tpl.rows;
    // Synthesis uses the witness-constant version (l_r/u_r constrained by
    // the candidate disjunction); the verifier uses the concrete-bound
    // version (checks tpl's actual bound values directly).
    let a_x = if concrete {
        template_formula_concrete(rows, n, false, bv, bws, &signed_all)
    } else {
        template_formula(rows, n, false, bv, bws, &signed_all)
    };
    let a_xp = if concrete {
        template_formula_concrete(rows, n, true, bv, bws, &signed_all)
    } else {
        template_formula(rows, n, true, bv, bws, &signed_all)
    };
    let g = cond_to_smt(&problem.loop_guard, bv, bws, &signed_all, n)?;
    let mut ante = vec![a_x, g];
    // Param type-range conditions are spliced into the
    // quantified antecedent — params are quantifier-scoped.
    ante.push(param_range_conds(&problem.params, bv));
    if let Some(guard) = &edge.guard {
        ante.push(cond_to_smt(guard, bv, bws, &signed_all, n)?);
    }
    if let Some(def) = &edge.definedness {
        ante.push(cond_to_smt(def, bv, bws, &signed_all, n)?);
    }
    let mut next_parts = Vec::new();
    for (i, next_e) in edge.next_values.iter().enumerate() {
        let ctx_bw = bws.get(i).copied().unwrap_or(64) as u32;
        // A saturating transfer (`x_i := x_i +? c`) encodes the successor
        // as `clamp(x_i + c)` — the same piecewise semantics the Clamp
        // rows carry (`template_formula`'s primed Interval branch).  A
        // raw `+` would silently drop the clamp, mixing two different
        // transfer semantics inside one query (the antecedent's Clamp
        // rows say `clamp(x + c) ≤ u'`, the transition says
        // `xp = x + c` — inconsistent whenever the addition saturates).
        let next_s = match next_e {
            crate::hir::loop_ir::ScalarExpr::Add(l, r, sem)
                if *sem == crate::hir::loop_ir::ArithSem::Saturate =>
            {
                let c = match r.as_ref() {
                    crate::hir::loop_ir::ScalarExpr::Const(c) => c,
                    _ => return None, // non-constant saturate step — fail closed.
                };
                let x = expr_to_smt(l, bv, bws, &signed_all, n, ctx_bw)?;
                // next_values are indexed by the loop variables — an
                // out-of-range index is an IR inconsistency, NOT a
                // recoverable default: fail closed rather than clamp with
                // a guessed (8, false) width/signedness (LIA would then
                // reason over the WRONG [0,255] range silently).
                let (bw, signed) = problem.vars.get(i).map(|v| (v.bw, v.signed))?;
                clamp_expr(&x, c.to_i128()?, bv, bw, signed)
            }
            _ => expr_to_smt(next_e, bv, bws, &signed_all, n, ctx_bw)?,
        };
        next_parts.push(format!("(= xp_{i} {next_s})"));
    }
    let next_cond = if next_parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", next_parts.join(" "))
    };
    let ante_cond = if ante.len() == 1 {
        ante.into_iter().next().unwrap()
    } else {
        format!("(and {})", ante.join(" "))
    };
    Some(format!("(=> (and {ante_cond} {next_cond}) {a_xp})"))
}

/// BiiLoopProblem `Refine` query : one ∀ for Init plus one
/// ∀ per back-edge implication. Returns None on an encoding failure
/// (fail-closed).
fn build_refine_query_problem(
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    candidates: &[BiiTemplate],
    get_model: bool,
    bv: bool,
    bws: &[u8],
) -> Option<String> {
    let vars_len = problem.vars.len();
    let params_len = problem.params.len();
    if candidates[0].n_vars != vars_len + params_len {
        return None; // template var count must equal loop vars + params.
    }
    let rows = &candidates[0].rows;
    let signed_all: Vec<bool> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.signed)
        .collect();

    let sort = |bw: u32| -> String {
        if bv {
            format!("(_ BitVec {})", bw)
        } else {
            "Int".to_string()
        }
    };
    let lit = |v: &BigInt, bw: u32, off: &BigInt| -> String {
        if bv {
            bv_const(&(v + off), bw)
        } else {
            v.to_string()
        }
    };
    let ge = if bv { "bvuge" } else { ">=" };
    let le = if bv { "bvule" } else { "<=" };

    let mut smt = String::new();
    if bv {
        smt.push_str("(set-logic BV)\n");
    } else {
        smt.push_str("(set-logic LIA)\n");
    }
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        smt.push_str(&format!(
            "(declare-const l_{r} {})\n(declare-const u_{r} {})\n",
            sort(ebw),
            sort(ebw)
        ));
    }
    // Candidate disjunction (same as build_refine_query).
    emit_candidate_disjunction(&mut smt, candidates, &lit, bv);

    // Init: ∀X. ⋀_i (x_i = init_i) ⇒ A'(X). init_i = Var(i) is a tautology
    // (unconstrained). Loop variables only (params have no initial
    // constraint — provided by the caller's state).
    let mut init_parts = Vec::new();
    for (i, init_e) in problem.init.iter().enumerate() {
        let ctx_bw = bws.get(i).copied().unwrap_or(64) as u32;
        init_parts.push(format!(
            "(= x_{i} {})",
            expr_to_smt(init_e, bv, bws, &signed_all, vars_len, ctx_bw)?
        ));
    }
    let init_cond = if init_parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", init_parts.join(" "))
    };
    let a_x_init = template_formula(rows, vars_len, false, bv, bws, &signed_all);
    smt.push_str("(assert (forall (");
    for i in 0..vars_len {
        smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
    }
    for (j, p) in problem.params.iter().enumerate() {
        smt.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
    }
    smt.push_str(&format!(
        ") (=> (and {} {}) {})))\n",
        init_cond,
        param_range_conds(&problem.params, bv),
        a_x_init
    ));

    // One ∀ implication per back edge (x/xp loop variables, n read-only params).
    for edge in &problem.back_edges {
        let imp = encode_edge_inductiveness(problem, &candidates[0], edge, bv, bws, false)?;
        smt.push_str("(assert (forall (");
        for i in 0..vars_len {
            smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            smt.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        for i in 0..vars_len {
            smt.push_str(&format!("(xp_{i} {}) ", sort(bws[i] as u32)));
        }
        smt.push_str(&format!(") {imp}))\n"));
    }

    // Property Strengthening (Yao et al., "Demystifying...", §IV-A):
    // ∀X. (A'(X) ∧ ¬G(X)) ⇒ Post(X)
    if let Some(post) = &problem.post {
        let post_s = cond_to_smt(post, bv, bws, &signed_all, vars_len)?;
        let g_s = cond_to_smt(&problem.loop_guard, bv, bws, &signed_all, vars_len)?;
        let a_x = template_formula(rows, vars_len, false, bv, bws, &signed_all);

        smt.push_str("(assert (forall (");
        for i in 0..vars_len {
            smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            smt.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        smt.push_str(&format!(
            ") (=> (and {} (not {})) {})))\n",
            a_x, g_s, post_s
        ));
    }

    smt.push_str("(check-sat)\n");
    if get_model {
        smt.push_str("(get-model)\n");
    }
    Some(smt)
}

/// BiiLoopProblem bounded-leap query (Algorithm 5): `B ⊑ A′ ⊏ A` as in
/// the paper; the transition part is Init plus one implication per back
/// edge (single ∀).
fn build_bounded_leap_query_problem(
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    cur: &BiiTemplate,
    limits: &BoundaryLimits,
    get_model: bool,
    bv: bool,
    bws: &[u8],
) -> Option<String> {
    let vars_len = problem.vars.len();
    let params_len = problem.params.len();
    if cur.n_vars != vars_len + params_len {
        return None; // template var count must equal loop vars + params.
    }
    let rows = &cur.rows;
    let signed_all: Vec<bool> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.signed)
        .collect();

    let sort = |bw: u32| -> String {
        if bv {
            format!("(_ BitVec {})", bw)
        } else {
            "Int".to_string()
        }
    };
    let lit = |v: &BigInt, bw: u32, off: &BigInt| -> String {
        if bv {
            bv_const(&(v + off), bw)
        } else {
            v.to_string()
        }
    };
    let ge = if bv { "bvuge" } else { ">=" };
    let le = if bv { "bvule" } else { "<=" };
    let gt = if bv { "bvugt" } else { ">" };
    let lt = if bv { "bvult" } else { "<" };

    let mut smt = String::new();
    if bv {
        smt.push_str("(set-logic BV)\n");
    } else {
        smt.push_str("(set-logic LIA)\n");
    }
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        smt.push_str(&format!(
            "(declare-const l_{r} {})\n(declare-const u_{r} {})\n",
            sort(ebw),
            sort(ebw)
        ));
    }
    // B ⊑ A′ ⊏ A (Algorithm 5).
    smt.push_str("(assert (and ");
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        let off = row.offset();
        let (rge, rle) = row_ge_le(row, bv);
        smt.push_str(&format!(
            " ({} l_{r} {}) ({} l_{r} {}) ({} u_{r} {}) ({} u_{r} {})",
            rge,
            lit(&row.lb, ebw, &off),
            rle,
            lit(&limits.lb[r], ebw, &off),
            rge,
            lit(&limits.ub[r], ebw, &off),
            rle,
            lit(&row.ub, ebw, &off)
        ));
        smt.push_str(&format!(" ({} l_{r} u_{r})", rle));
    }
    smt.push_str("))\n");
    // `A′ ⊏ A` conjunct (Algorithm 5) of Zuo et al. (2026) — the strict-bound
    // disjunction excludes the trivial witness A' = A (Theorem 5.5).
    smt.push_str("(assert (or");
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        let off = row.offset();
        let (rgt, rlt) = row_gt_lt(row, bv);
        smt.push_str(&format!(
            " ({} l_{r} {}) ({} u_{r} {})",
            rgt,
            lit(&row.lb, ebw, &off),
            rlt,
            lit(&row.ub, ebw, &off)
        ));
    }
    smt.push_str("))\n");

    // Single ∀: Init + one implication per back edge.
    let mut parts = Vec::new();
    let mut init_parts = Vec::new();
    for (i, init_e) in problem.init.iter().enumerate() {
        let ctx_bw = bws.get(i).copied().unwrap_or(64) as u32;
        init_parts.push(format!(
            "(= x_{i} {})",
            expr_to_smt(init_e, bv, bws, &signed_all, vars_len, ctx_bw)?
        ));
    }
    let init_cond = if init_parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", init_parts.join(" "))
    };
    parts.push(format!(
        "(=> (and {} {}) {})",
        init_cond,
        param_range_conds(&problem.params, bv),
        template_formula(rows, vars_len, false, bv, bws, &signed_all)
    ));
    for edge in &problem.back_edges {
        parts.push(encode_edge_inductiveness(
            problem, cur, edge, bv, bws, false,
        )?);
    }
    let body = if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", parts.join(" "))
    };

    smt.push_str("(assert (forall (");
    for i in 0..vars_len {
        smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
    }
    for (j, p) in problem.params.iter().enumerate() {
        smt.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
    }
    for i in 0..vars_len {
        smt.push_str(&format!("(xp_{i} {}) ", sort(bws[i] as u32)));
    }
    smt.push_str(&format!(") {body}))\n"));

    // Property Strengthening (Yao et al., "Demystifying...", §IV-A):
    // ∀X. (A'(X) ∧ ¬G(X)) ⇒ Post(X)
    if let Some(post) = &problem.post {
        let post_s = cond_to_smt(post, bv, bws, &signed_all, vars_len)?;
        let g_s = cond_to_smt(&problem.loop_guard, bv, bws, &signed_all, vars_len)?;
        let a_x = template_formula(rows, vars_len, false, bv, bws, &signed_all);

        smt.push_str("(assert (forall (");
        for i in 0..vars_len {
            smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            smt.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        smt.push_str(&format!(
            ") (=> (and {} (not {})) {})))\n",
            a_x, g_s, post_s
        ));
    }

    smt.push_str("(check-sat)\n");
    if get_model {
        smt.push_str("(get-model)\n");
    }
    Some(smt)
}

/// The query-budget floor for a template: the paper's raw query count
/// is ~2× the 4W high-level bound (Theorem 5.5), so 8W raw is the
/// correct ceiling; a fixed budget under-budgets wide/multi-variable
/// templates (2 vars @ 64b: W = 258, 8W ≈ 2064) and silently drops the
/// hint (fail-closed).  The checker applies this floor when sizing its
/// synthesis budget; the drivers themselves take the caller's budget
/// verbatim so tests can assert tight logarithmic bounds (the floor
/// must not inflate a test's explicit budget).
pub(crate) fn query_budget_floor(
    n_vars: usize,
    bit_widths: &[u8],
    signed: &[bool],
    saturates: &[(usize, i128)],
) -> usize {
    // Both modes include mixed-signedness rows — the floor covers the
    // full template.
    let tpl = BiiTemplate::with_saturates(n_vars, bit_widths, signed, saturates);
    let w: usize = tpl.rows.iter().map(|r| r.enc_bw() as usize).sum();
    w * 8 + 64
}

/// BiiLoopProblem BII synthesis (Algorithm 4 + 5 driver unchanged; the
/// transition encoding is edge-wise ∀). Return contract identical to
/// `synthesize_bitwise_bii`: `Some` = the BII or an any-time partial
/// invariant (at least one refinement adopted); `None` = nothing was
/// adopted, or an encoding failure.
pub(crate) fn synthesize_problem_bii(
    solver: &SmtSolver,
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    max_queries: usize,
    use_bv: bool,
) -> Option<BiiTemplate> {
    // The template domain covers loop variables + external-symbol params
    // (read-only, enter the template rows).
    let bws_all: Vec<u8> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.bw)
        .collect();
    let signed_all: Vec<bool> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.signed)
        .collect();
    let mut cur =
        BiiTemplate::with_saturates(bws_all.len(), &bws_all, &signed_all, &problem.saturates);
    let mut pos = BitPositions::new(&cur);
    let mut limits = BoundaryLimits::new(&cur);
    let mut calls = 0usize;
    let mut prev_unsat = false;
    // Any-time partial invariant: whether at least one refinement was
    // adopted (see the budget-exhaustion return below).
    let mut adopted = false;
    // Adaptive leap cooldown: when the solver struggles on the bounded-leap
    // query (unknown/error), back off to the bitwise search for a few
    // iterations instead of thrashing — the leap is retried after the
    // cooldown expires.
    let mut leap_cooldown = 0;

    loop {
        if calls >= max_queries {
            // Budget exhausted: `cur` is inductive by construction — every
            // adopted witness passed the ∃∀ query carrying BOTH the Init and
            // the Inductiveness obligation. Return the PARTIAL invariant (the
            // any-time property of Zuo et al. (2026), §4 remark after Algorithm 2); `None`
            // only when nothing was adopted (the result would be the
            // uninformative ⊤).
            return if adopted { Some(cur) } else { None };
        }
        if prev_unsat {
            pos.advance();
            // Each standard UNSAT tick decays the cooldown.
            if leap_cooldown > 0 {
                leap_cooldown -= 1;
            }
        }
        prev_unsat = false;

        let candidates = propose_bitwise(&cur, &pos, &limits);
        if candidates.is_empty() {
            // EXHAUSTED pointers (every bit passed — the true
            // termination) vs WINDOW-INVALID proposals: an empty
            // candidate set can mean bits remain whose hypotheses are
            // window-invalid (every candidate was filtered by the l ≤ u
            // pre-check). Propose of Zuo et al. (2026) emits ALL bit hypotheses
            // and lets the SOLVER reject impossible ones (the l ≤ u
            // conjunct makes the disjunct trivially false → UNSAT →
            // pointers advance, Lemma 5.2); the Rust pre-filter
            // short-circuits that path, so advance while any pointer is
            // live, return only when every position is exhausted.
            //
            // Candidates can be empty for three reasons:
            //
            // 1. every bit position has already been exhausted;
            // 2. every generated bit hypothesis is window-invalid (`l > u`);
            // 3. every generated bit hypothesis was pruned by BoundaryLimits.
            //
            // In case 3, advancement is still sound:
            //
            // - a pruned lower-bound hypothesis has `new_lb > limits.lb[idx]`,
            //   so even the smallest value with this bit set is too large;
            //   the corresponding BII bit must be 0;
            //
            // - a pruned upper-bound hypothesis has `new_ub < limits.ub[idx]`,
            //   so even the largest value with this bit cleared is too small;
            //   the corresponding BII bit must be 1.
            //
            // Therefore passing the current positions preserves Lemma 5.2's
            // bit-preservation and progress argument.
            if pos.lpos.iter().chain(pos.upos.iter()).all(|&p| p < 0) {
                return Some(cur); // every bit position passed — BII reached.
            }
            pos.advance();
            continue;
        }

        let Some(query) = build_refine_query_problem(problem, &candidates, false, use_bv, &bws_all)
        else {
            return if adopted { Some(cur) } else { None };
        };
        calls += 1;
        match solver.run_raw_query(&query) {
            RawQueryOutcome::Unsat => {
                prev_unsat = true;
                limits.prune_unsat(&cur, &candidates);
                // Only attempt the bounded leap once the cooldown has
                // expired (a struggling solver on the leap query backs
                // off to the bitwise search instead of thrashing).
                if limits.is_active() && leap_cooldown == 0 {
                    if calls >= max_queries {
                        return if adopted { Some(cur) } else { None };
                    }
                    let Some(leap) = build_bounded_leap_query_problem(
                        problem, &cur, &limits, false, use_bv, &bws_all,
                    ) else {
                        return if adopted { Some(cur) } else { None };
                    };
                    calls += 1;
                    match solver.run_raw_query(&leap) {
                        RawQueryOutcome::Unsat => return Some(cur),
                        RawQueryOutcome::Sat(_) => {
                            if calls >= max_queries {
                                return if adopted { Some(cur) } else { None };
                            }
                            let Some(leap_m) = build_bounded_leap_query_problem(
                                problem, &cur, &limits, true, use_bv, &bws_all,
                            ) else {
                                return if adopted { Some(cur) } else { None };
                            };
                            calls += 1;
                            match solver.run_raw_query(&leap_m) {
                                RawQueryOutcome::Sat(model) => {
                                    match parse_witness(&model, &cur.rows, use_bv) {
                                        Some(bounds) => {
                                            apply_bounds(&mut cur, &bounds);
                                            limits.tighten_sat(&bounds);
                                            adopted = true;
                                            prev_unsat = true;
                                        }
                                        None => {
                                            eprintln!(
                                                "witness parse failure in leap — synthesis bug"
                                            );
                                            return if adopted { Some(cur) } else { None };
                                        }
                                    }
                                }
                                RawQueryOutcome::Unknown
                                | RawQueryOutcome::Error(_)
                                | RawQueryOutcome::Unsat => {
                                    return if adopted { Some(cur) } else { None };
                                }
                            }
                        }
                        RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                            // The solver struggles on the leap query —
                            // activate the cooldown and back off to the
                            // bitwise search instead of returning early.
                            leap_cooldown = 5;
                        }
                    }
                }
            }
            RawQueryOutcome::Sat(_) => {
                if calls >= max_queries {
                    return if adopted { Some(cur) } else { None };
                }
                let Some(query_model) =
                    build_refine_query_problem(problem, &candidates, true, use_bv, &bws_all)
                else {
                    return if adopted { Some(cur) } else { None };
                };
                calls += 1;
                match solver.run_raw_query(&query_model) {
                    RawQueryOutcome::Sat(model) => match parse_witness(&model, &cur.rows, use_bv) {
                        Some(bounds) => {
                            apply_bounds(&mut cur, &bounds);
                            limits.tighten_sat(&bounds);
                            adopted = true;
                        }
                        None => {
                            eprintln!("witness parse failure in refine — synthesis bug");
                            return if adopted { Some(cur) } else { None };
                        }
                    },
                    RawQueryOutcome::Unknown
                    | RawQueryOutcome::Error(_)
                    | RawQueryOutcome::Unsat => {
                        return if adopted { Some(cur) } else { None };
                    }
                }
            }
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                return if adopted { Some(cur) } else { None };
            }
        }
    }
}

// ═══ Exact DBM-based inductiveness verification ═══
//
// Committee ruling ("if it is strong enough to prove soundness, do not
// bother z3"): for templates inside the octagon fragment the DBM strong
// closure proves inductiveness EXACTLY, replacing every solver
// round-trip of `verify_template_against_problem`.
/// Maximum number of `if`-paths the DBM verifier enumerates before
/// falling back to the SMT verifier (fail-closed — exponential path
/// blowups stay on the z3 path).  16 covers every real branch structure
/// seen in practice (nested ifs explode at 2^n — 4 nested ifs = 16
/// paths), while keeping the fast path's enumeration bounded.
const DBM_VERIFY_MAX_PATHS: usize = 16;
/// EXACT difference-constraint verification of template
/// inductiveness, mirroring `verify_template_against_problem`'s three
/// obligations so the fast path is semantically equivalent to the SMT
/// path it short-circuits:
///
/// 1. `init ∧ param-ranges ⟹ A(X)` — per row half-bound;
/// 2. per back edge, per `if`-path: `A(X) ∧ G ∧ guard ∧ def ∧ T ⟹ A(X′)`
///    — per row half-bound over the PRIMED loop variables;
/// 3. per back edge: `A ∧ G ∧ guard ⟹ def` (trap absence, when
///    `definedness` is present).
///
/// Each check is refutation: the premise DBM plus the NEGATED half-bound
/// (still a difference constraint over the integers: ¬(E ≤ c) ⟺ E ≥ c+1)
/// must close to ⊥. Strong closure only derives constraints IMPLIED by
/// the input (paper Theorem 4.3 — the closure is the normal form of the
/// same V-domain), so an unsat closed DBM PROVES the real system unsat:
/// `Some(true)` is a sound proof and discharges without z3 (the
/// founder's bifurcation).
///
/// The converse is NOT trusted: real-domain closure can miss integer
/// tightenings (Zuo et al. 2026, §V.D), so a satisfiable query might be a
/// fabricated model. `Some(false)` therefore only means "fall back to
/// the SMT verifier for the final verdict" — never a direct
/// `Counterexample` report.
///
/// `None` — outside the fragment: Support3/Clamp rows, Saturate
/// transitions, `Or`-disjunctive conditions, `Neq` comparisons,
/// non-affine next-values, more than `DBM_VERIFY_MAX_PATHS` if-paths,
/// bounds outside `i128`, or a template/domain size mismatch. The
/// caller falls back to the SMT verifier unchanged.
///
/// Every closure here runs `ClosureMode::IntegerExact` (the
/// Harvey–Stuckey closed-and-rounded fixpoint, Zuo et al. 2026, §V.D). This
/// verifier reasons over INTEGERS by construction (the fast path is
/// gated `!bv` — LIA mathematical-integer semantics — over
/// `Int<N>`/`UInt<N>` variables), so the integer-only tightening is
/// sound here and strictly widens the Some(true) hit rate: the
/// negated row half-bound `¬(2x ≤ 2ub) = 2x ≥ 2ub+1` carries exactly
/// the odd-2x tension the rounding resolves (the Figure 6 pattern).
/// `Some(false)` still only means "fall back to the SMT verifier" —
/// the fixpoint's non-⊥ result is NOT claimed to be an integer-
/// satisfiability witness.
///
/// A check-3 failure returns Some(false) — the fast path cannot
/// distinguish "domain too weak to prove trap absence" from a genuine
/// counterexample, so it defers to the SMT verifier, which reports
/// TrapUnproven (the BII itself remains verified inductive).
fn dbm_proves_inductiveness(
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    tpl: &BiiTemplate,
) -> Option<bool> {
    use crate::hir::loop_ir::EdgeKind;
    // ── Fragment gate ─────────────────────────────────────────────
    for row in &tpl.rows {
        match row.kind {
            RowKind::Interval(_) | RowKind::Diff(_, _) | RowKind::Sum(_, _) => {}
            RowKind::Support3(..) | RowKind::Clamp(..) => return None,
        }
        if row.lb.to_i128().is_none() || row.ub.to_i128().is_none() {
            return None; // bounds beyond i128 — stay on the SMT path
        }
    }
    let n_loop = problem.vars.len();
    let n_params = problem.params.len();
    let n_total = n_loop + n_params;
    if n_loop == 0 || tpl.n_vars != n_total {
        return None;
    }
    for p in &problem.params {
        if p.bw > 127 {
            return None; // 1i128 << bw would overflow — stay on the SMT path
        }
    }
    // Node mapping: template-domain variable i occupies DBM variable
    // `map(i)`; primed loop variables are appended after the whole
    // template domain. DBM variable v spans nodes 2v (X⁺) / 2v+1 (X⁻).
    let map =
        |i: usize, primed: bool| -> usize { if primed && i < n_loop { n_total + i } else { i } };
    let node = |i: usize, neg: bool, primed: bool| -> usize {
        2 * map(i, primed) + if neg { 1 } else { 0 }
    };
    // One half-bound of row r as a difference edge `node_a − node_b ≤ c`.
    // `upper` picks `f ≤ ub`; else `f ≥ lb` (encoded as `−f ≤ −lb`).
    // Node selection follows Figure 5 translation of Zuo et al. (2026).
    let row_half = |row: &BiiRow, upper: bool, primed: bool| -> Option<(usize, usize, i128)> {
        let lb = row.lb.to_i128()?;
        let ub = row.ub.to_i128()?;
        match row.kind {
            // f = xᵢ: f ≤ ub ⟺ 2xᵢ ≤ 2ub (self-dual edge).
            RowKind::Interval(i) => {
                if upper {
                    Some((
                        node(i, false, primed),
                        node(i, true, primed),
                        ub.checked_mul(2)?,
                    ))
                } else {
                    Some((
                        node(i, true, primed),
                        node(i, false, primed),
                        lb.checked_mul(-2)?,
                    ))
                }
            }
            // f = xᵢ − xⱼ ≤ ub; f ≥ lb ⟺ xⱼ − xᵢ ≤ −lb.
            RowKind::Diff(i, j) => {
                if upper {
                    Some((node(i, false, primed), node(j, false, primed), ub))
                } else {
                    Some((
                        node(j, false, primed),
                        node(i, false, primed),
                        lb.checked_neg()?,
                    ))
                }
            }
            // f = xᵢ + xⱼ ⟺ xᵢ − (−xⱼ) ≤ ub; f ≥ lb ⟺ −xᵢ − xⱼ ≤ −lb.
            RowKind::Sum(i, j) => {
                if upper {
                    Some((node(i, false, primed), node(j, true, primed), ub))
                } else {
                    Some((
                        node(i, true, primed),
                        node(j, false, primed),
                        lb.checked_neg()?,
                    ))
                }
            }
            _ => None,
        }
    };
    // ¬(a − b ≤ c) ⟺ a − b ≥ c+1 ⟺ b − a ≤ −(c+1) (integers).
    let negate = |(a, b, c): (usize, usize, i128)| -> Option<(usize, usize, i128)> {
        Some((b, a, 0i128.checked_sub(c)?.checked_sub(1)?))
    };
    // Param type ranges — the LIA verifier splices the same ranges into
    // its antecedents; the DBM path mirrors them on
    // the params' self-dual edges.
    let mut param_edges: Vec<(usize, usize, i128)> = Vec::new();
    for (j, p) in problem.params.iter().enumerate() {
        let (lo, hi) = if p.signed {
            let half = 1i128 << (p.bw as usize - 1);
            (-half, half - 1)
        } else {
            (0, (1i128 << p.bw as usize) - 1)
        };
        let v = n_loop + j;
        param_edges.push((2 * v, 2 * v + 1, hi.checked_mul(2)?)); // p ≤ hi
        param_edges.push((2 * v + 1, 2 * v, lo.checked_mul(-2)?)); // p ≥ lo
    }
    // Refute one counter-query against a closed premise: SATISFIABLE
    // means a potential counterexample (Some(false) — the SMT verifier
    // decides); ⊥ means the half-bound is proven.
    let refuted = |dbm: &Dbm, neg: (usize, usize, i128)| -> Option<bool> {
        let mut m = dbm.clone();
        m.set_mirrored(neg.0, neg.1, neg.2);
        if m.close_with(ClosureMode::IntegerExact) {
            Some(false) // satisfiable — fall back to SMT for the verdict
        } else {
            None // refuted — this half-bound holds
        }
    };
    // ── Check 1: init ∧ param-ranges ⟹ A(X) ──────────────────────
    {
        let mut dbm = Dbm::new(n_total);
        for &(a, b, c) in &param_edges {
            dbm.set_mirrored(a, b, c);
        }
        for (i, e) in problem.init.iter().enumerate() {
            use crate::hir::loop_ir::ScalarExpr as E;
            match e {
                E::Var(v) if *v == i => {} // the tautology xᵢ = xᵢ — skip
                E::Const(c) => {
                    let k = c.to_i128()?;
                    dbm.set_mirrored(2 * i, 2 * i + 1, k.checked_mul(2)?);
                    dbm.set_mirrored(2 * i + 1, 2 * i, k.checked_mul(-2)?);
                }
                _ => return None, // unsupported init shape — fallback
            }
        }
        if dbm.close_with(ClosureMode::IntegerExact) {
            for row in &tpl.rows {
                for upper in [true, false] {
                    let neg = negate(row_half(row, upper, false)?)?;
                    if let Some(false) = refuted(&dbm, neg) {
                        return Some(false);
                    }
                }
            }
        }
        // ⊥ init+ranges: vacuous — the SMT verifier's exists-query is
        // unsat the same way (check 1 trivially holds).
    }
    // ── Check 2: per back edge, per if-path:
    //    A ∧ G ∧ guard ∧ def ∧ path-conds ∧ T ⟹ A(X′) ─────────────
    for edge in &problem.back_edges {
        if edge.kind != EdgeKind::Back {
            continue;
        }
        let mut base_conds: Vec<crate::hir::loop_ir::Cond> = vec![problem.loop_guard.clone()];
        if let Some(g) = &edge.guard {
            base_conds.push(g.clone());
        }
        // definedness is an ANTECEDENT of inductiveness (mirroring
        // `encode_edge_inductiveness`) and the CONSEQUENT of check 3.
        if let Some(d) = &edge.definedness {
            base_conds.push(d.clone());
        }
        let mut paths: Vec<(
            Vec<crate::hir::loop_ir::Cond>,
            Vec<crate::hir::loop_ir::ScalarExpr>,
        )> = Vec::new();
        if !expand_if_paths(&edge.next_values, &base_conds, &mut paths, &problem.vars) {
            return None; // too many paths / nested shape — fallback
        }
        for (conds, values) in &paths {
            let mut dbm = Dbm::new(n_total + n_loop);
            for &(a, b, c) in &param_edges {
                dbm.set_mirrored(a, b, c);
            }
            // A(X): every row, both halves, original side.
            for row in &tpl.rows {
                for upper in [true, false] {
                    let (a, b, c) = row_half(row, upper, false)?;
                    dbm.set_mirrored(a, b, c);
                }
            }
            // G ∧ guard ∧ def ∧ path conditions.
            for c in conds {
                let mut edges = Vec::new();
                if !cond_to_edges(c, false, &mut edges) {
                    return None;
                }
                for (a, b, cc) in edges {
                    dbm.set_mirrored(a, b, cc);
                }
            }
            // T(X, X′): primed difference edges per loop variable.
            for (i, e) in values.iter().enumerate() {
                if !next_value_edges(i, e, n_total, &mut dbm) {
                    return None;
                }
            }
            if !dbm.close_with(ClosureMode::IntegerExact) {
                continue; // contradictory premise — vacuous path
            }
            // A(X′): every row, both halves, PRIMED side.
            for row in &tpl.rows {
                for upper in [true, false] {
                    let neg = negate(row_half(row, upper, true)?)?;
                    if let Some(false) = refuted(&dbm, neg) {
                        return Some(false);
                    }
                }
            }
        }
    }
    // ── Check 3: A ∧ G ∧ guard ⟹ def (trap absence) ───────────────
    for edge in &problem.back_edges {
        if edge.kind != EdgeKind::Back {
            continue;
        }
        let Some(def) = &edge.definedness else {
            continue;
        };
        let mut dbm = Dbm::new(n_total);
        for &(a, b, c) in &param_edges {
            dbm.set_mirrored(a, b, c);
        }
        for row in &tpl.rows {
            for upper in [true, false] {
                let (a, b, c) = row_half(row, upper, false)?;
                dbm.set_mirrored(a, b, c);
            }
        }
        let mut conds = vec![problem.loop_guard.clone()];
        if let Some(g) = &edge.guard {
            conds.push(g.clone());
        }
        for c in &conds {
            let mut edges = Vec::new();
            if !cond_to_edges(c, false, &mut edges) {
                return None;
            }
            for (a, b, cc) in edges {
                dbm.set_mirrored(a, b, cc);
            }
        }
        if dbm.close_with(ClosureMode::IntegerExact) {
            // def must be an And-chain (the lowering only produces
            // And(Cmp)); each conjunct's negation is separately refuted
            // (A ⟹ c1 ∧ c2 ⟺ ∀i: A ⟹ ci).
            let mut conjuncts = Vec::new();
            if !split_and(def, &mut conjuncts) {
                return None;
            }
            for c in &conjuncts {
                let mut edges = Vec::new();
                if !cond_to_edges(c, true, &mut edges) {
                    return None;
                }
                let mut m = dbm.clone();
                for (a, b, cc) in edges {
                    m.set_mirrored(a, b, cc);
                }
                if m.close_with(ClosureMode::IntegerExact) {
                    return Some(false);
                }
            }
        }
    }
    // ── Check 4: A(X) ∧ ¬G(X) ⇒ Post(X) ──────────────────────
    if let Some(post) = &problem.post {
        let mut dbm = Dbm::new(n_total);
        for &(a, b, c) in &param_edges {
            dbm.set_mirrored(a, b, c);
        }
        for row in &tpl.rows {
            for upper in [true, false] {
                let (a, b, c) = row_half(row, upper, false)?;
                dbm.set_mirrored(a, b, c);
            }
        }
        // ¬G(X): encode the negated loop_guard.
        // If loop_guard is an And-chain, ¬(A∧B) = ¬A∨¬B is disjunctive,
        // outside the difference-constraint fragment → cond_to_edges
        // returns false → fall back to SMT.
        let mut neg_guard_edges = Vec::new();
        if !cond_to_edges(&problem.loop_guard, true, &mut neg_guard_edges) {
            return None; // ¬G outside the fragment → fall back to SMT
        }
        for (a, b, cc) in neg_guard_edges {
            dbm.set_mirrored(a, b, cc);
        }
        if dbm.close_with(ClosureMode::IntegerExact) {
            // Negate each Post conjunct and refute it separately.
            let mut conjuncts = Vec::new();
            if !split_and(post, &mut conjuncts) {
                return None;
            }
            for c in &conjuncts {
                let mut neg_edges = Vec::new();
                if !cond_to_edges(c, true, &mut neg_edges) {
                    return None;
                }
                let mut m = dbm.clone();
                for (a, b, cc) in neg_edges {
                    m.set_mirrored(a, b, cc);
                }
                if m.close_with(ClosureMode::IntegerExact) {
                    return Some(false); // satisfiable → SMT decides
                }
            }
        }
        // ⊥ antecedent: vacuously true — Check 4 holds trivially.
    }
    Some(true) // every counter-query refuted — inductiveness PROVEN
}
/// A linear form `Σ aᵥ·xᵥ + k` over the template-domain variables
/// (arbitrary variable count and coefficients — the callers enforce
/// their own fragment limits). `None` on non-affine shapes, `Saturate`
/// arithmetic (the clamp is piecewise — not a difference transfer),
/// `Ite` (the caller expands top-level Ites first), or `i128` overflow.
fn lin_of_scalar(e: &crate::hir::loop_ir::ScalarExpr) -> Option<(Vec<(usize, i128)>, i128)> {
    use crate::hir::loop_ir::{ArithSem, ScalarExpr as E};
    match e {
        E::Var(i) => Some((vec![(*i, 1)], 0)),
        E::Const(c) => c.to_i128().map(|k| (Vec::new(), k)),
        E::Add(l, r, sem) => {
            if *sem == ArithSem::Saturate {
                return None;
            }
            let (mut lc, lk) = lin_of_scalar(l)?;
            let (rc, rk) = lin_of_scalar(r)?;
            for (v, a) in rc {
                if let Some(slot) = lc.iter_mut().find(|(w, _)| *w == v) {
                    slot.1 = slot.1.checked_add(a)?;
                } else {
                    lc.push((v, a));
                }
            }
            lc.retain(|(_, a)| *a != 0);
            Some((lc, lk.checked_add(rk)?))
        }
        E::Sub(l, r, sem) => {
            if *sem == ArithSem::Saturate {
                return None;
            }
            let (mut lc, lk) = lin_of_scalar(l)?;
            let (rc, rk) = lin_of_scalar(r)?;
            for (v, a) in rc {
                let na = a.checked_neg()?;
                if let Some(slot) = lc.iter_mut().find(|(w, _)| *w == v) {
                    slot.1 = slot.1.checked_add(na)?;
                } else {
                    lc.push((v, na));
                }
            }
            lc.retain(|(_, a)| *a != 0);
            Some((lc, lk.checked_sub(rk)?))
        }
        E::Ite(..) => None,
    }
}
/// Encode a `Cond` as difference edges `(node_a, node_b, c)` on the
/// ORIGINAL-side nodes (2v / 2v+1). `negated` flips the comparison
/// (Lt↔Ge, Le↔Gt, Neq→Eq); `¬Eq` is `Neq` — non-convex, rejected.
/// And-chains conjoin; an Or under negation (de Morgan) conjoins the
/// negated arms; a bare Or is disjunctive — rejected. A literally
/// false condition is encoded as the contradictory self-loop
/// `(0, 0, −1)` (any negative diagonal closes to ⊥).
fn cond_to_edges(
    c: &crate::hir::loop_ir::Cond,
    negated: bool,
    out: &mut Vec<(usize, usize, i128)>,
) -> bool {
    use crate::hir::loop_ir::Cond as C;
    match c {
        C::True => {
            if negated {
                out.push((0, 0, -1)); // ¬true — contradictory premise
            }
            true
        }
        C::False => {
            if !negated {
                out.push((0, 0, -1)); // false — contradictory premise
            }
            true
        }
        C::Cmp { op, lhs, rhs, .. } => {
            let effective = if negated {
                negate_cmp_op(*op)
            } else {
                Some(*op)
            };
            match effective {
                Some(op) => cmp_to_edges(op, lhs, rhs, out),
                None => false, // ¬Eq = Neq — non-convex
            }
        }
        C::And(a, b) => {
            if negated {
                return false; // ¬(a∧b) = ¬a∨¬b — disjunctive, outside
            }
            cond_to_edges(a, false, out) && cond_to_edges(b, false, out)
        }
        C::Or(a, b) => {
            if !negated {
                return false; // a∨b — disjunctive, outside
            }
            cond_to_edges(a, true, out) && cond_to_edges(b, true, out)
        }
        C::Not(inner) => cond_to_edges(inner, !negated, out),
    }
}
fn negate_cmp_op(op: crate::hir::loop_ir::CmpOp) -> Option<crate::hir::loop_ir::CmpOp> {
    use crate::hir::loop_ir::CmpOp as O;
    Some(match op {
        O::Lt => O::Ge,
        O::Le => O::Gt,
        O::Gt => O::Le,
        O::Ge => O::Lt,
        O::Neq => O::Eq,
        O::Eq => return None, // ¬(=) is ≠ — non-convex
    })
}
/// One comparison as difference edges: `lhs op rhs` with the difference
/// `E = lhs − rhs` reduced to at most two variables with ±1
/// coefficients (paper Figure 5 node selection; single-variable bounds
/// ride the self-dual edges with the doubled constant). Integer
/// discreteness: `E < 0 ⟺ E ≤ −1`, `E > 0 ⟺ −E ≤ −1`, `E ≥ 0 ⟺ −E ≤ 0`;
/// `Eq` emits both directions; `Neq` is non-convex — rejected.
fn cmp_to_edges(
    op: crate::hir::loop_ir::CmpOp,
    lhs: &crate::hir::loop_ir::ScalarExpr,
    rhs: &crate::hir::loop_ir::ScalarExpr,
    out: &mut Vec<(usize, usize, i128)>,
) -> bool {
    use crate::hir::loop_ir::CmpOp as O;
    let (lc, lk) = match lin_of_scalar(lhs) {
        Some(v) => v,
        None => return false,
    };
    let (rc, rk) = match lin_of_scalar(rhs) {
        Some(v) => v,
        None => return false,
    };
    // E = lhs − rhs: coefficients lc − rc, constant lk − rk.
    let mut coefs = lc;
    for (v, a) in rc {
        let na = match a.checked_neg() {
            Some(n) => n,
            None => return false,
        };
        if let Some(slot) = coefs.iter_mut().find(|(w, _)| *w == v) {
            slot.1 = match slot.1.checked_add(na) {
                Some(s) => s,
                None => return false,
            };
        } else {
            coefs.push((v, na));
        }
    }
    coefs.retain(|(_, a)| *a != 0);
    if coefs.len() > 2 || coefs.iter().any(|(_, a)| a.abs() != 1) {
        return false; // outside the difference-constraint fragment
    }
    let k = match lk.checked_sub(rk) {
        Some(v) => v,
        None => return false,
    };
    // `s·E ≤ slack` per operator (Eq: both directions).
    let dirs: &[(i128, i128)] = match op {
        O::Le => &[(1, 0)],
        O::Lt => &[(1, -1)],
        O::Ge => &[(-1, 0)],
        O::Gt => &[(-1, -1)],
        O::Eq => &[(1, 0), (-1, 0)],
        O::Neq => return false,
    };
    for &(s, slack) in dirs {
        // s·(Σ aᵥxᵥ + k) ≤ slack ⟺ Σ (s·aᵥ)·xᵥ ≤ slack − s·k.
        let smk = match s.checked_mul(k) {
            Some(v) => v,
            None => return false,
        };
        let bound = match slack.checked_sub(smk) {
            Some(b) => b,
            None => return false,
        };
        let scaled: Vec<(usize, i128)> = coefs.iter().map(|(v, a)| (*v, s * a)).collect();
        match scaled.as_slice() {
            [] => return false, // constant-only comparison — fallback
            [(i, ci)] => {
                // ±xᵢ ≤ B ⟺ 2·(±xᵢ) ≤ 2B on the self-dual edge.
                let (a, b) = if *ci == 1 {
                    (2 * i, 2 * i + 1)
                } else {
                    (2 * i + 1, 2 * i)
                };
                let c2 = match bound.checked_mul(2) {
                    Some(v) => v,
                    None => return false,
                };
                out.push((a, b, c2));
            }
            [(i, ci), (j, cj)] => {
                // ±xᵢ ± xⱼ ≤ B ⟺ (ci·xᵢ) − (−cj·xⱼ) ≤ B (Figure 5).
                let na = 2 * i + (*ci < 0) as usize;
                let nb = 2 * j + (*cj > 0) as usize;
                out.push((na, nb, bound));
            }
            _ => return false,
        }
    }
    true
}
/// The transition for loop variable `dst` as primed difference edges
/// written into `dbm` over the extended space (vars ++ params ++
/// primed loop vars — `Dbm::new(n_total + n_loop)`). Accepts `Var(j)`
/// (copy — j may be a param, read-only and unprimed), `Const(k)` (pin
/// on the primed self-dual edge), and affine `±xⱼ + k` shapes with
/// `Wrap`/`Trap` semantics (mathematical integers — matching the LIA
/// mode this fast path gates on). `Saturate` and any other shape
/// return `false` (fall back to the SMT verifier).
fn next_value_edges(
    dst: usize,
    e: &crate::hir::loop_ir::ScalarExpr,
    n_total: usize,
    dbm: &mut Dbm,
) -> bool {
    let (coefs, k) = match lin_of_scalar(e) {
        Some(v) => v,
        None => return false,
    };
    // Primed nodes of the destination (a loop variable).
    let p = 2 * (n_total + dst); // x′⁺
    let q = p + 1; // x′⁻
    match coefs.as_slice() {
        [] => {
            // x′ = k — pin on the primed self-dual edge (Figure 5).
            let hi = match k.checked_mul(2) {
                Some(v) => v,
                None => return false,
            };
            let lo = match k.checked_mul(-2) {
                Some(v) => v,
                None => return false,
            };
            dbm.set_mirrored(p, q, hi);
            dbm.set_mirrored(q, p, lo);
            true
        }
        [(j, a)] if a.abs() == 1 => {
            // x′ = a·xⱼ + k — two difference edges between the primed
            // node and (±)xⱼ's ORIGINAL node (params are unprimed).
            let s = if *a == 1 { 2 * j } else { 2 * j + 1 };
            let neg_k = match k.checked_neg() {
                Some(v) => v,
                None => return false,
            };
            dbm.set_mirrored(p, s, k);
            dbm.set_mirrored(s, p, neg_k);
            true
        }
        _ => false, // multi-variable or |coef| ≠ 1 — outside the fragment
    }
}
/// Expand the top-level `Ite`s and saturating assignments of the
/// per-variable next-values into explicit paths: each path carries the
/// branch conditions (the else-arm as `Not(cond)` — negation happens at
/// the `Cmp` level in `cond_to_edges`) and the branch-selected, Ite-free
/// next-values. Nested Ites recurse. A saturating `Add`/`Sub`
/// (`ArithSem::Saturate`) expands into three linear paths (overflow to
/// `max`, underflow to `min`, and the in-range successor). Returns
/// `false` when the path count would exceed `DBM_VERIFY_MAX_PATHS`
/// (fail-closed to the SMT verifier).
fn expand_if_paths(
    values: &[crate::hir::loop_ir::ScalarExpr],
    conds: &[crate::hir::loop_ir::Cond],
    out: &mut Vec<(
        Vec<crate::hir::loop_ir::Cond>,
        Vec<crate::hir::loop_ir::ScalarExpr>,
    )>,
    vars: &[crate::hir::loop_ir::BiiVar],
) -> bool {
    use crate::hir::loop_ir::{ArithSem, ScalarExpr as E};
    use num_bigint::BigInt;

    let mut split_at = None;
    let mut is_saturate = false;

    for (i, v) in values.iter().enumerate() {
        if matches!(v, E::Ite(..)) {
            split_at = Some(i);
            break;
        }
        if let E::Add(_, _, ArithSem::Saturate) | E::Sub(_, _, ArithSem::Saturate) = v {
            split_at = Some(i);
            is_saturate = true;
            break;
        }
    }

    let Some(i) = split_at else {
        if out.len() >= DBM_VERIFY_MAX_PATHS {
            return false;
        }
        out.push((conds.to_vec(), values.to_vec()));
        return true;
    };

    if is_saturate {
        // Resolve the saturating expression `Var(x) +/- Const(c)`.
        let Some((var_idx, c_val, signed, bw)) = (match &values[i] {
            E::Add(l, r, ArithSem::Saturate) => {
                if let (E::Var(v), E::Const(c)) = (l.as_ref(), r.as_ref()) {
                    Some((*v, c.clone(), vars[*v].signed, vars[*v].bw))
                } else if let (E::Const(c), E::Var(v)) = (l.as_ref(), r.as_ref()) {
                    Some((*v, c.clone(), vars[*v].signed, vars[*v].bw))
                } else {
                    None
                }
            }
            E::Sub(l, r, ArithSem::Saturate) => {
                if let (E::Var(v), E::Const(c)) = (l.as_ref(), r.as_ref()) {
                    Some((*v, -c.clone(), vars[*v].signed, vars[*v].bw))
                } else {
                    None
                }
            }
            _ => None,
        }) else {
            return false;
        };

        let (min, max): (BigInt, BigInt) = if signed {
            let half = BigInt::one() << (bw as usize - 1);
            (-half.clone(), half - 1)
        } else {
            (BigInt::zero(), (BigInt::one() << bw as usize) - 1)
        };

        let limit_high = &max - &c_val;
        let limit_low = &min - &c_val;

        // Path 1: overflow (x > limit_high => x' = max).
        let mut v1 = values.to_vec();
        v1[i] = E::Const(max.clone());
        let mut c1 = conds.to_vec();
        c1.push(crate::hir::loop_ir::Cond::Cmp {
            op: crate::hir::loop_ir::CmpOp::Gt,
            lhs: Box::new(E::Var(var_idx)),
            rhs: Box::new(E::Const(limit_high.clone())),
            signed,
        });

        // Path 2: underflow (x < limit_low => x' = min).
        let mut v2 = values.to_vec();
        v2[i] = E::Const(min.clone());
        let mut c2 = conds.to_vec();
        c2.push(crate::hir::loop_ir::Cond::Cmp {
            op: crate::hir::loop_ir::CmpOp::Lt,
            lhs: Box::new(E::Var(var_idx)),
            rhs: Box::new(E::Const(limit_low.clone())),
            signed,
        });

        // Path 3: in-range (limit_low ≤ x ≤ limit_high => x' = x + c).
        let mut v3 = values.to_vec();
        v3[i] = E::Add(
            Box::new(E::Var(var_idx)),
            Box::new(E::Const(c_val.clone())),
            ArithSem::Wrap,
        );
        let mut c3 = conds.to_vec();
        c3.push(crate::hir::loop_ir::Cond::And(
            Box::new(crate::hir::loop_ir::Cond::Cmp {
                op: crate::hir::loop_ir::CmpOp::Le,
                lhs: Box::new(E::Var(var_idx)),
                rhs: Box::new(E::Const(limit_high)),
                signed,
            }),
            Box::new(crate::hir::loop_ir::Cond::Cmp {
                op: crate::hir::loop_ir::CmpOp::Ge,
                lhs: Box::new(E::Var(var_idx)),
                rhs: Box::new(E::Const(limit_low)),
                signed,
            }),
        ));

        expand_if_paths(&v1, &c1, out, vars)
            && expand_if_paths(&v2, &c2, out, vars)
            && expand_if_paths(&v3, &c3, out, vars)
    } else {
        let E::Ite(c, t, f) = &values[i] else {
            unreachable!()
        };
        let mut then_values = values.to_vec();
        then_values[i] = (**t).clone();
        let mut else_values = values.to_vec();
        else_values[i] = (**f).clone();
        let mut then_conds = conds.to_vec();
        then_conds.push((**c).clone());
        let mut else_conds = conds.to_vec();
        else_conds.push(crate::hir::loop_ir::Cond::Not(Box::new((**c).clone())));
        expand_if_paths(&then_values, &then_conds, out, vars)
            && expand_if_paths(&else_values, &else_conds, out, vars)
    }
}
/// Split an And-chain into its conjuncts (for check 3's per-conjunct
/// refutation). `True` contributes nothing; `False`/`Or` shapes return
/// `false` (fall back).
fn split_and(c: &crate::hir::loop_ir::Cond, out: &mut Vec<crate::hir::loop_ir::Cond>) -> bool {
    use crate::hir::loop_ir::Cond as C;
    match c {
        C::And(a, b) => split_and(a, out) && split_and(b, out),
        C::True => true,
        C::False => false,
        C::Or(..) => false,
        C::Not(..) | C::Cmp { .. } => {
            out.push(c.clone());
            true
        }
    }
}
/// Result of independent template verification.
#[derive(Debug)]
pub(crate) enum VerifyOutcome {
    /// Every check UNSAT (inductiveness AND trap absence) — the
    /// template is an inductive invariant.
    Verified,
    /// Checks 1–2 UNSAT — the template IS an inductive invariant —
    /// but check 3 (trap absence, `A ∧ G ∧ guard ⟹ def`) found a
    /// counterexample: the template domain cannot prove the loop
    /// trap-free. NOT a synthesis-layer bug: the BII may simply be
    /// weaker than the reachable set (e.g. a strided counter whose
    /// interval BII cannot express the stride — the loop may still
    /// be trap-free in reality), or the loop may genuinely trap.
    /// Distinct from `Counterexample`: the invariant itself is
    /// verified, so the hint channel stays sound. Surfacing the
    /// trap signal as a user-visible diagnostic (error vs
    /// warning, strict vs non-strict) is an L3 decision —
    /// deliberately deferred, not silently dropped.
    TrapUnproven,
    /// Checks 1–2 UNSAT (inductiveness holds), but Check 4 (ϕ₃ exit
    /// sufficiency) is SAT: the BII within the template domain cannot
    /// entail the postcondition. Either a domain expressiveness gap
    /// (the BII is too weak), or the program genuinely fails Post.
    /// The BII itself is verified inductive, so the hint channel stays
    /// intact.
    PostUnproven,
    /// Some check found a counterexample (sat) — the template is NOT
    /// inductive (a synthesis-layer bug; the checker reports it).
    Counterexample,
    /// solver unknown / error / encoding failure (timeout included) —
    /// inconclusive; the caller falls back.
    Inconclusive,
}

/// Independent template verification : returns
/// `VerifyOutcome::Verified` only if every check holds. Does not trust the
/// synthesizer's internal state — re-verify before emitting a hint:
///
/// 1. ∀X. init(X) ⇒ A(X)                                  -- Pre ⇒ Inv
/// 2. Per back edge: A ∧ G ∧ guard ∧ def ∧ next ⇒ A'      -- Inv preservation
/// 3. Per trap back edge: A ∧ G ∧ guard ⇒ def             -- trap absence
///
/// A sat (counterexample) result → `Counterexample`; unknown / error /
/// encoding failure (timeout included) → `Inconclusive` (the caller falls
/// back). Shares `encode_edge_inductiveness` with synthesis to avoid
/// encoding drift. `definedness` is all None in the current lowering
/// (item 3 dormant); params (external symbols) are universally
/// quantified.
pub(crate) fn verify_template_against_problem(
    solver: &SmtSolver,
    problem: &crate::hir::loop_ir::BiiLoopProblem,
    tpl: &BiiTemplate,
    bv: bool,
) -> VerifyOutcome {
    use crate::hir::loop_ir::EdgeKind;
    // DBM fast path (committee ruling: "if it is strong enough to prove
    // soundness, do not bother z3"): inside the octagon fragment
    // (Interval/Diff/Sum rows, difference-expressible transitions) the
    // DBM closure proves inductiveness EXACTLY — no solver round-trip.
    // Only the LIA reading is covered (BV mode has modular semantics
    // the mathematical DBM cannot express); `Some(false)` (a satisfiable
    // counter-query — the real-domain closure may have missed an integer
    // tightening, Zuo et al. 2026, §V.D) and `None` (outside the fragment) both
    // fall through to the authoritative SMT verifier below.
    if !bv {
        if let Some(true) = dbm_proves_inductiveness(problem, tpl) {
            return VerifyOutcome::Verified;
        }
    }
    let vars_len = problem.vars.len();
    let params_len = problem.params.len();
    let bws_all: Vec<u8> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.bw)
        .collect();
    let signed_all: Vec<bool> = problem
        .vars
        .iter()
        .chain(problem.params.iter())
        .map(|v| v.signed)
        .collect();
    if tpl.n_vars != vars_len + params_len {
        return VerifyOutcome::Inconclusive; // template var count must equal loop vars + params.
    }
    let rows = &tpl.rows;
    let sort = |bw: u32| -> String {
        if bv {
            format!("(_ BitVec {})", bw)
        } else {
            "Int".to_string()
        }
    };
    let a_x = template_formula_concrete(rows, vars_len, false, bv, &bws_all, &signed_all);

    let mut logic = String::new();
    if bv {
        logic.push_str("(set-logic BV)\n");
    } else {
        logic.push_str("(set-logic LIA)\n");
    }

    // 1. Pre ⇒ Inv: refutation query `¬∀X. init(X) ⇒ A(X)`; UNSAT = holds.
    let mut init_parts = Vec::new();
    for (i, init_e) in problem.init.iter().enumerate() {
        let ctx_bw = bws_all.get(i).copied().unwrap_or(64) as u32;
        match expr_to_smt(init_e, bv, &bws_all, &signed_all, vars_len, ctx_bw) {
            Some(s) => init_parts.push(format!("(= x_{i} {s})")),
            None => return VerifyOutcome::Inconclusive,
        }
    }
    let init_cond = if init_parts.is_empty() {
        "true".to_string()
    } else {
        format!("(and {})", init_parts.join(" "))
    };
    let mut q = logic.clone();
    q.push_str("(assert (not (forall (");
    for i in 0..vars_len {
        q.push_str(&format!("(x_{i} {}) ", sort(bws_all[i] as u32)));
    }
    for (j, p) in problem.params.iter().enumerate() {
        q.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
    }
    q.push_str(&format!(
        ") (=> (and {} {}) {}))))\n",
        init_cond,
        param_range_conds(&problem.params, bv),
        a_x
    ));
    q.push_str("(check-sat)\n");
    match solver.run_raw_query(&q) {
        RawQueryOutcome::Unsat => {}
        RawQueryOutcome::Sat(_) => return VerifyOutcome::Counterexample,
        RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
            return VerifyOutcome::Inconclusive;
        }
    }

    // 2. Per back edge: Inv preservation (shares encode_edge_inductiveness
    //    with synthesis).
    for edge in &problem.back_edges {
        if edge.kind != EdgeKind::Back {
            continue;
        }
        let Some(imp) = encode_edge_inductiveness(problem, tpl, edge, bv, &bws_all, true) else {
            return VerifyOutcome::Inconclusive;
        };
        let mut q = logic.clone();
        q.push_str("(assert (not (forall (");
        for i in 0..vars_len {
            q.push_str(&format!("(x_{i} {}) ", sort(bws_all[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            q.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        for i in 0..vars_len {
            q.push_str(&format!("(xp_{i} {}) ", sort(bws_all[i] as u32)));
        }
        q.push_str(&format!(") {imp})))\n"));
        q.push_str("(check-sat)\n");
        match solver.run_raw_query(&q) {
            RawQueryOutcome::Unsat => {}
            RawQueryOutcome::Sat(_) => return VerifyOutcome::Counterexample,
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                return VerifyOutcome::Inconclusive;
            }
        }
    }

    // 3. trap absence: A ∧ G ∧ guard ⇒ def. ACTIVE since the trap
    // lowering generates `definedness`
    // for every non-zero AddVar under trap semantics (the checker
    // passes `!use_bv`). A SAT result is a DOMAIN limitation (or a
    // genuine trap), not a synthesis bug → TrapUnproven.
    for edge in &problem.back_edges {
        if edge.kind != EdgeKind::Back {
            continue;
        }
        let Some(def) = &edge.definedness else {
            continue;
        };
        let Some(g) = cond_to_smt(&problem.loop_guard, bv, &bws_all, &signed_all, vars_len) else {
            return VerifyOutcome::Inconclusive;
        };
        let Some(def_s) = cond_to_smt(def, bv, &bws_all, &signed_all, vars_len) else {
            return VerifyOutcome::Inconclusive;
        };
        let mut ante = vec![a_x.clone(), g];
        // Param type-range conditions, quantifier-scoped.
        ante.push(param_range_conds(&problem.params, bv));
        if let Some(guard) = &edge.guard {
            match cond_to_smt(guard, bv, &bws_all, &signed_all, vars_len) {
                Some(s) => ante.push(s),
                None => return VerifyOutcome::Inconclusive,
            }
        }
        let ante_cond = if ante.len() == 1 {
            ante.into_iter().next().unwrap()
        } else {
            format!("(and {})", ante.join(" "))
        };
        let mut q = logic.clone();
        q.push_str("(assert (not (forall (");
        for i in 0..vars_len {
            q.push_str(&format!("(x_{i} {}) ", sort(bws_all[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            q.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        q.push_str(&format!(") (=> {} {}))))\n", ante_cond, def_s));
        q.push_str("(check-sat)\n");
        match solver.run_raw_query(&q) {
            RawQueryOutcome::Unsat => {}
            RawQueryOutcome::Sat(_) => return VerifyOutcome::TrapUnproven,
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                return VerifyOutcome::Inconclusive;
            }
        }
    }

    // 4. Exit sufficiency: A(X) ∧ ¬G(X) ⇒ Post(X)
    if let Some(post) = &problem.post {
        let Some(post_s) = cond_to_smt(post, bv, &bws_all, &signed_all, vars_len) else {
            return VerifyOutcome::Inconclusive;
        };
        let Some(g) = cond_to_smt(&problem.loop_guard, bv, &bws_all, &signed_all, vars_len) else {
            return VerifyOutcome::Inconclusive;
        };

        let mut ante = vec![a_x.clone(), format!("(not {g})")];
        ante.push(param_range_conds(&problem.params, bv));
        let ante_cond = if ante.len() == 1 {
            ante.into_iter().next().unwrap()
        } else {
            format!("(and {})", ante.join(" "))
        };

        let mut q = logic.clone();
        q.push_str("(assert (not (forall (");
        for i in 0..vars_len {
            q.push_str(&format!("(x_{i} {}) ", sort(bws_all[i] as u32)));
        }
        for (j, p) in problem.params.iter().enumerate() {
            q.push_str(&format!("(n_{j} {}) ", sort(p.bw as u32)));
        }
        q.push_str(&format!(") (=> {} {}))))\n", ante_cond, post_s));
        q.push_str("(check-sat)\n");

        match solver.run_raw_query(&q) {
            RawQueryOutcome::Unsat => {}
            RawQueryOutcome::Sat(_) => return VerifyOutcome::PostUnproven,
            RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => {
                return VerifyOutcome::Inconclusive;
            }
        }
    }

    VerifyOutcome::Verified
}

/// Algorithm 5's bounded leap query:
///  `∃A'.∀X,X'.P(A',X,X') ∧ (B ⊑ A' ⊏ A)`  — the witness must lie inside
/// the under-approximation `B`, at-or-tighter-than `A`, and STRICTLY
/// tighter in at least one row (`⊏` of Zuo et al. (2026)).
///
/// SMT has no lattice-order primitive, so `⊏` is expanded to the
/// row-wise disjunction `∃r. l'_r > A.l_r ∨ u'_r < A.u_r`; it is what
/// excludes the trivial witness A' = A and makes the Theorem-5.5 UNSAT
/// termination (`A = A*`) reachable.
///
/// Returns `None` if the transition encoding fails (fail-closed).
fn build_bounded_leap_query(
    cur: &BiiTemplate,
    limits: &BoundaryLimits,
    init: &[LoopInstr],
    body: &[LoopInstr],
    get_model: bool,
    bv: bool,
    bws: &[u8],
    signed: &[bool],
) -> Option<String> {
    let n = cur.n_vars;
    let rows = &cur.rows;

    // Per-variable sort: LIA `Int` or QF_BV `(_ BitVec W)`.
    let sort = |bw: u32| -> String {
        if bv {
            format!("(_ BitVec {})", bw)
        } else {
            "Int".to_string()
        }
    };
    // `off` maps a TEMPLATE-domain bound to the ENCODED domain used by
    // the BV sorts (Diff rows shift the signed range `[-m, m]` onto
    // `[0, 2m]`). LIA has no width, so the offset is ignored there.
    let lit = |v: &BigInt, bw: u32, off: &BigInt| -> String {
        if bv {
            bv_const(&(v + off), bw)
        } else {
            v.to_string()
        }
    };
    let ge = if bv { "bvuge" } else { ">=" };
    let le = if bv { "bvule" } else { "<=" };
    let gt = if bv { "bvugt" } else { ">" };
    let lt = if bv { "bvult" } else { "<" };

    let mut smt = String::new();
    if bv {
        smt.push_str("(set-logic BV)\n");
    } else {
        smt.push_str("(set-logic LIA)\n");
    }
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        smt.push_str(&format!(
            "(declare-const l_{r} {})\n(declare-const u_{r} {})\n",
            sort(ebw),
            sort(ebw)
        ));
    }

    // B ⊑ A′ ⊏ A in bound terms (ENCODED domain, Diff rows offset):
    // A.l_i ≤ l'_i ≤ B.lb_i and B.ub_i ≤ u'_i ≤ A.u_i
    smt.push_str("(assert (and ");
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        let off = row.offset();
        let (rge, rle) = row_ge_le(row, bv);
        smt.push_str(&format!(
            " ({} l_{r} {}) ({} l_{r} {}) ({} u_{r} {}) ({} u_{r} {})",
            rge,
            lit(&row.lb, ebw, &off),
            rle,
            lit(&limits.lb[r], ebw, &off),
            rge,
            lit(&limits.ub[r], ebw, &off),
            rle,
            lit(&row.ub, ebw, &off)
        ));
        smt.push_str(&format!(" ({} l_{r} u_{r})", rle));
    }
    smt.push_str("))\n");

    // `A′ ⊏ A` conjunct (Algorithm 5) of Zuo et al. (2026), expanded into the
    // row-wise strict-bound disjunction: `⊏` excludes the trivial witness
    // A' = A, exactly as the Theorem-5.5 UNSAT termination requires.
    smt.push_str("(assert (or");
    for (r, row) in rows.iter().enumerate() {
        let ebw = row.enc_bw();
        let off = row.offset();
        let (rgt, rlt) = row_gt_lt(row, bv);
        smt.push_str(&format!(
            " ({} l_{r} {}) ({} u_{r} {})",
            rgt,
            lit(&row.lb, ebw, &off),
            rlt,
            lit(&row.ub, ebw, &off)
        ));
    }
    smt.push_str("))\n");

    // Sequential transition.
    let (inter, trans) = encode_sequential_transition(body, n, bv, bws)?;

    smt.push_str("(assert (forall (");
    for i in 0..n {
        smt.push_str(&format!("(x_{i} {}) ", sort(bws[i] as u32)));
    }
    for i in 0..n {
        smt.push_str(&format!("(xp_{i} {}) ", sort(bws[i] as u32)));
    }
    for name in &inter {
        let var_idx = name
            .strip_prefix("xs_")
            .and_then(|s| s.split('_').nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        smt.push_str(&format!("({} {}) ", name, sort(bws[var_idx] as u32)));
    }
    smt.push_str(")\n");

    let a_x = template_formula(rows, n, false, bv, bws, signed);
    let a_xp = template_formula(rows, n, true, bv, bws, signed);

    let mut pre_parts = Vec::new();
    for instr in init {
        match instr {
            LoopInstr::ConstVar(i, c) => {
                let c_big = BigInt::from(*c);
                pre_parts.push(format!(
                    "(= x_{i} {})",
                    lit(&c_big, bws[*i] as u32, &BigInt::zero())
                ));
            }
            // Non-constant init cannot be a plain seed — fail closed.
            _ => return None,
        }
    }
    let pre = if pre_parts.is_empty() {
        "true".to_string()
    } else if pre_parts.len() == 1 {
        pre_parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", pre_parts.join(" "))
    };

    let mut guard_parts = Vec::new();
    for instr in body {
        match instr {
            LoopInstr::TestLe(i, c) => {
                if bv {
                    use crate::hir::loop_ir::ScalarExpr;
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Le,
                        &ScalarExpr::Var(*i),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(<= x_{i} {})", c));
                }
            }
            LoopInstr::TestGe(i, c) => {
                if bv {
                    use crate::hir::loop_ir::ScalarExpr;
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Ge,
                        &ScalarExpr::Var(*i),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(>= x_{i} {})", c));
                }
            }
            LoopInstr::TestDiffLe(i, j, c) => {
                if bv {
                    use crate::hir::loop_ir::{ArithSem, ScalarExpr};
                    guard_parts.push(encode_cmp_bv(
                        crate::hir::loop_ir::CmpOp::Le,
                        &ScalarExpr::Sub(
                            Box::new(ScalarExpr::Var(*i)),
                            Box::new(ScalarExpr::Var(*j)),
                            ArithSem::Wrap,
                        ),
                        &ScalarExpr::Const(BigInt::from(*c)),
                        bws,
                        signed,
                        n,
                    )?);
                } else {
                    guard_parts.push(format!("(<= (- x_{i} x_{j}) {})", c));
                }
            }
            LoopInstr::AddVar(..) | LoopInstr::ConstVar(..) | LoopInstr::CopyVar(..) => {}
            // `if` conditions are path conditions, not loop-header
            // guards — skipped here.
            LoopInstr::If(..) => {}
            // `+?`/`-?` assignments are ignored for guard collection
            // (the transfer encoding fails closed on them).
            LoopInstr::AddSat(..) => {}
        }
    }
    let guard = if guard_parts.is_empty() {
        "true".to_string()
    } else if guard_parts.len() == 1 {
        guard_parts.into_iter().next().unwrap()
    } else {
        format!("(and {})", guard_parts.join(" "))
    };

    smt.push_str(&format!(
        "  (and (=> {} {}) (=> (and {} {} {}) {}))\n",
        pre, a_x, a_x, guard, trans, a_xp
    ));
    smt.push_str("))\n");
    smt.push_str("(check-sat)\n");
    if get_model {
        smt.push_str("(get-model)\n");
    }
    Some(smt)
}

/// Parse a `(get-model)` witness: extract `(define-fun l_r () SORT v)` /
/// `(define-fun u_r () SORT v)` per row. Returns `None` on any missing or
/// malformed entry (fail closed).
///
/// In BV mode the witness bounds are in the ENCODED (offset) domain, so
/// Diff rows are mapped back to the template domain `[-m, m]` here; LIA
/// witnesses are already in the template domain.
fn parse_witness(model: &str, rows: &[BiiRow], bv: bool) -> Option<Vec<(BigInt, BigInt)>> {
    let wrapped = format!("({})", model);
    let top = sexp::parse(&wrapped).ok()?;

    let mut out = Vec::with_capacity(rows.len());

    for r in 0..rows.len() {
        let lb = parse_define_fun_from_sexp(&top, &format!("l_{r}"))?;
        let ub = parse_define_fun_from_sexp(&top, &format!("u_{r}"))?;
        let off = if bv { rows[r].offset() } else { BigInt::zero() };
        // A signed Interval row's witness is a
        // bit-vector pattern — convert it to the TRUE signed value
        // (two's complement) before storing.
        let to_real = |v: BigInt, row: &BiiRow| -> BigInt {
            if bv && row.signed && matches!(row.kind, RowKind::Interval(_) | RowKind::Clamp(..)) {
                let modulus = BigInt::one() << row.bw as usize;
                let half = BigInt::one() << (row.bw as usize - 1);
                if v >= half { v - modulus } else { v }
            } else {
                v
            }
        };
        out.push((to_real(lb - &off, &rows[r]), to_real(ub - &off, &rows[r])));
    }

    Some(out)
}

/// Extract the integer value of `(define-fun NAME () SORT <v>)`.
///
/// This uses a real S-expression parse instead of assuming any particular
/// whitespace layout. It handles Z3 models such as:
///
/// ```text
/// (define-fun l_0 () Int 0)
/// (define-fun l_0 () Int
///   (- 5))
/// (model
///   (define-fun u_0 () Int
///     7))
/// (define-fun l_0 () (_ BitVec 8)
///   #x00)
/// ```
fn parse_define_fun(model: &str, name: &str) -> Option<BigInt> {
    // `sexp::parse` parses one top-level S-expression. Model output may
    // contain several top-level forms, so wrap the whole output in one list.
    let wrapped = format!("({})", model);
    let top = sexp::parse(&wrapped).ok()?;

    parse_define_fun_from_sexp(&top, name)
}

fn parse_define_fun_from_sexp(top: &Sexp, name: &str) -> Option<BigInt> {
    let value = find_define_fun_value(top, name)?;
    sexp_value_to_bigint(value)
}

fn sexp_atom(s: &Sexp) -> Option<&str> {
    match s {
        // The `sexp` crate stores both SMT-LIB symbols and quoted strings
        // in the single `S(String)` variant.
        Sexp::Atom(sexp::Atom::S(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Recursively find:
///
/// ```text
/// (define-fun NAME () SORT VALUE)
/// ```
///
/// and return `VALUE`.
/// The sort of a define-fun constant witness must be `Int` (LIA) or
/// `(_ BitVec N)` (BV) — anything else is rejected. Defensive: witness
/// names are generated uniquely, so a mismatched same-named definition
/// elsewhere in the model is the only way this could trip.
fn is_numeric_sort(sort: &Sexp) -> bool {
    match sort {
        Sexp::Atom(sexp::Atom::S(s)) => s == "Int",
        Sexp::List(items) => {
            items.len() == 3
                && matches!(items[0], Sexp::Atom(sexp::Atom::S(ref s)) if s == "_")
                && matches!(items[1], Sexp::Atom(sexp::Atom::S(ref s)) if s == "BitVec")
                && matches!(items[2], Sexp::Atom(sexp::Atom::I(_)))
        }
        _ => false,
    }
}

fn find_define_fun_value<'a>(sx: &'a Sexp, name: &str) -> Option<&'a Sexp> {
    let Sexp::List(items) = sx else {
        return None;
    };

    // A define-fun constant witness has at least:
    // define-fun, name, params, sort, value
    if items.len() >= 5 {
        let head = sexp_atom(&items[0]);
        let fname = sexp_atom(&items[1]);

        if head == Some("define-fun") && fname == Some(name) {
            if let Sexp::List(params) = &items[2] {
                if params.is_empty() && is_numeric_sort(&items[3]) {
                    return Some(&items[4]);
                }
            }
        }
    }

    // Otherwise keep searching. This covers `(model ...)` wrappers and
    // multiple top-level commands.
    items
        .iter()
        .find_map(|child| find_define_fun_value(child, name))
}

/// Parse a model VALUE S-expression into `BigInt`.
///
/// Supports the forms we currently emit/receive:
///
/// - LIA positive: `7`
/// - LIA negative: `(- 5)`
/// - LIA bare negative atom: `-5`
/// - BV hex: `#x03`
/// - BV binary: `#b0011`
/// - BV literal: `(_ bv5 8)`
/// - BV negation: `(bvneg (_ bv5 8))`
fn sexp_value_to_bigint(value: &Sexp) -> Option<BigInt> {
    match value {
        Sexp::Atom(atom) => match atom {
            sexp::Atom::S(s) => parse_bigint_atom(s.as_str()),
            sexp::Atom::I(i) => Some(BigInt::from(*i)),
            _ => None,
        },
        Sexp::List(items) => {
            let head = sexp_atom(items.first()?)?;
            match head {
                "-" if items.len() == 2 => {
                    let n = sexp_value_to_bigint(&items[1])?;
                    Some(-n)
                }
                "bvneg" if items.len() == 2 => {
                    let n = sexp_value_to_bigint(&items[1])?;
                    Some(-n)
                }
                "_" if items.len() == 3 => {
                    let sym = sexp_atom(&items[1])?;
                    let digits = sym.strip_prefix("bv")?;
                    if digits.is_empty() {
                        return None;
                    }
                    digits.parse::<BigInt>().ok()
                }
                _ => None,
            }
        }
    }
}

fn parse_bigint_atom(atom: &str) -> Option<BigInt> {
    let atom = atom.trim();

    if atom.is_empty() {
        return None;
    }

    if let Some(hex) = atom.strip_prefix("#x") {
        if hex.is_empty() {
            return None;
        }

        return BigInt::parse_bytes(hex.as_bytes(), 16);
    }

    if let Some(bin) = atom.strip_prefix("#b") {
        if bin.is_empty() {
            return None;
        }

        return BigInt::parse_bytes(bin.as_bytes(), 2);
    }

    atom.parse::<BigInt>().ok()
}

/// Convert a converged template into Posita invariant expressions (only
/// non-trivial rows) — the candidate list for the `@hint` channel.
///
/// Bounds that exceed `i128` range are skipped (the AST `Literal::Int`
/// is `i128`-bounded; the SMT-layer result remains correct regardless).
/// This can happen for Sum rows at 128-bit width where the max bound
/// is `2^129 − 2 > i128::MAX`.
pub(crate) fn template_to_invariant_exprs<'a>(
    arena: &'a bumpalo::Bump,
    tpl: &BiiTemplate,
    vars: &[Symbol],
) -> Vec<&'a crate::ast::Expr<'a>> {
    let ident = |i: usize| -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::Ident(
            vars[i],
            crate::ast::Span::new(0, 0),
        ))
    };
    let lit = |c: i128| -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::Literal(
            crate::ast::Literal::Int(ast::IntLit::Small(c)),
            crate::ast::Span::new(0, 0),
        ))
    };
    let bin = |op: crate::ast::BinOp,
               l: &'a crate::ast::Expr<'a>,
               r: &'a crate::ast::Expr<'a>|
     -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: l,
            op,
            right: r,
            span: crate::ast::Span::new(0, 0),
        })
    };

    let mut out = Vec::new();
    for row in &tpl.rows {
        if row.is_trivial() {
            continue;
        }
        // Rows whose variable indices fall outside `vars` reference
        // external-symbol parameters (template domain = vars ++ params) —
        // they are not hints over the loop variables, so skip them (this
        // also guards `vars[i]` below against out-of-bounds panics).
        let in_vars = match row.kind {
            RowKind::Interval(i) => i < vars.len(),
            RowKind::Diff(i, j) => i < vars.len() && j < vars.len(),
            RowKind::Sum(i, j) => i < vars.len() && j < vars.len(),
            RowKind::Support3(i, j, k, ..) => i < vars.len() && j < vars.len() && k < vars.len(),
            // Clamp rows produce no hint — the AST has
            // no `clamp` expression.
            RowKind::Clamp(..) => false,
        };
        if !in_vars {
            continue;
        }
        // Bounds must fit in i128 for the AST literal.
        let lb_i128 = match row.lb.to_i128() {
            Some(v) => v,
            None => continue, // exceeds AST range — skip row.
        };
        let ub_i128 = match row.ub.to_i128() {
            Some(v) => v,
            None => continue, // exceeds AST range — skip row.
        };
        let f = match row.kind {
            RowKind::Interval(i) => ident(i),
            RowKind::Diff(i, j) => bin(crate::ast::BinOp::Sub, ident(i), ident(j)),
            RowKind::Sum(i, j) => bin(crate::ast::BinOp::Add, ident(i), ident(j)),
            RowKind::Support3(i, j, k, sj, sk) => {
                // `x_i + s_j·x_j + s_k·x_k` as nested ± binary ops.
                let inner = if sj {
                    bin(crate::ast::BinOp::Add, ident(i), ident(j))
                } else {
                    bin(crate::ast::BinOp::Sub, ident(i), ident(j))
                };
                if sk {
                    bin(crate::ast::BinOp::Add, inner, ident(k))
                } else {
                    bin(crate::ast::BinOp::Sub, inner, ident(k))
                }
            }
            // Clamp rows produce no hint (skipped by the in_vars check
            // above — the AST has no `clamp` expression).
            RowKind::Clamp(..) => unreachable!("Clamp rows are skipped by the in_vars check"),
        };
        // `l ≤ f ≤ u` ⟺ `f ≥ l ∧ f ≤ u`.
        out.push(bin(crate::ast::BinOp::Ge, f, lit(lb_i128)));
        out.push(bin(crate::ast::BinOp::Le, f, lit(ub_i128)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::loop_ir::{CmpOp, Cond, ScalarExpr};

    /// `i := 0; while i < 6 { i := i + 1 }` — the BII interval is
    /// `[0, 6]`: the init pins the lower bound to 0, inductiveness forces
    /// the upper bound to 6 (`i = 5` → `i' = 6` must stay in the
    /// invariant). Skipped when Z3 is unavailable.
    #[test]
    fn test_synthesize_linear_bii_basic_loop() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_synthesize_linear_bii_basic_loop");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        // Guard `i < 6` → `i ≤ 5`; body `i := i + 1`.
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        // Small bit-width keeps the linear search bounded.
        let bw = vec![4u8];
        let tpl = synthesize_linear_bii(&solver, &vars, &init, &body, &bw, &[false], 512, false)
            .expect("BII synthesis should converge with Z3 available");
        let row = &tpl.rows[0];
        assert_eq!(row.kind, RowKind::Interval(0));
        assert_eq!(row.lb, BigInt::zero(), "init `i := 0` pins the lower bound");
        assert_eq!(
            row.ub,
            BigInt::from(6),
            "inductiveness forces the upper bound to 6"
        );
    }

    /// Template → expression translation emits exactly `f ≥ l` and
    /// `f ≤ u` for a non-trivial row, and skips top rows.
    #[test]
    fn test_template_to_invariant_exprs() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let tpl = BiiTemplate {
            n_vars: 1,
            rows: vec![
                BiiRow {
                    kind: RowKind::Interval(0),
                    bw: 4,
                    lb: BigInt::zero(),
                    ub: BigInt::from(6),
                    signed: false,
                    full_lb: BigInt::zero(),
                    full_ub: BiiRow::max_ub(4),
                },
                BiiRow {
                    kind: RowKind::Interval(0),
                    bw: 4,
                    lb: BigInt::zero(),
                    ub: BiiRow::max_ub(4), // top — skipped.
                    signed: false,
                    full_lb: BigInt::zero(),
                    full_ub: BiiRow::max_ub(4),
                },
            ],
        };
        let exprs = template_to_invariant_exprs(arena, &tpl, &vars);
        assert_eq!(exprs.len(), 2, "only the non-trivial row is emitted");
        assert!(matches!(
            exprs[0],
            crate::ast::Expr::BinaryOp {
                op: crate::ast::BinOp::Ge,
                ..
            }
        ));
        assert!(matches!(
            exprs[1],
            crate::ast::Expr::BinaryOp {
                op: crate::ast::BinOp::Le,
                ..
            }
        ));
    }

    /// Fail-closed — an unavailable solver yields `None`, never a
    /// fabricated invariant.
    #[test]
    fn test_synthesize_linear_bii_fails_closed_without_solver() {
        let solver = SmtSolver::new("/nonexistent/z3-binary");
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        assert!(
            synthesize_linear_bii(&solver, &vars, &init, &body, &[4], &[false], 8, false).is_none(),
            "an unavailable solver must fail closed"
        );
    }

    /// The bitwise greedy strategy (Algorithm 4 + 5) computes the same
    /// BII `[0, 6]` for `i := 0; while i < 6 { i := i + 1 }` as the
    /// linear search, with a query count bounded by the bit width (not
    /// 2^bw). Skipped when Z3 is unavailable.
    #[test]
    fn test_synthesize_bitwise_bii_basic_loop() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_synthesize_bitwise_bii_basic_loop");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8]; // wider than the linear-search test: 8 bits.
        let tpl = synthesize_bitwise_bii(&solver, &vars, &init, &body, &bw, &[false], 512, false)
            .expect("bitwise BII synthesis should converge with Z3 available");
        let row = &tpl.rows[0];
        assert_eq!(row.kind, RowKind::Interval(0));
        assert_eq!(row.lb, BigInt::zero(), "init `i := 0` pins the lower bound");
        assert_eq!(
            row.ub,
            BigInt::from(6),
            "inductiveness forces the upper bound to 6"
        );
    }

    /// Any-time partial invariant (Zuo et al. 2026, §4 remark after Algorithm 2):
    /// a budget-exhausted driver returns the last INDUCTIVE template —
    /// not None — once at least one witness was adopted.  `i := 0;
    /// while i < 6 { i := i + 1 }` (BII [0,6]) with budget 3: the first
    /// refine+model pair (2 calls) adopts a witness (the first candidate
    /// set contains the inductive [0,127]), the budget check then fires.
    /// Asserts: (a) Some, never None; (b) sound — the partial bounds
    /// contain the BII row-wise (lb ≤ and ub ≥); (c) inductive — the
    /// independent verifier still says Verified.
    #[test]
    fn test_bitwise_bii_budget_exhaustion_returns_partial() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!(
                "z3 unavailable — skipping test_bitwise_bii_budget_exhaustion_returns_partial"
            );
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8];
        let tpl = synthesize_bitwise_bii(&solver, &vars, &init, &body, &bw, &[false], 3, false)
            .expect("budget exhaustion must return the partial invariant, never None");
        // (b) sound: partial ⊇ BII row-wise ([0,6] ⊆ [lb, ub]).
        assert!(tpl.rows[0].lb <= BigInt::zero(), "partial lb ≤ BII lb (0)");
        assert!(tpl.rows[0].ub >= BigInt::from(6), "partial ub ≥ BII ub (6)");
        // (c) inductive: the independent verifier confirms.
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &bw,
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
    }

    /// Same any-time property on the PRODUCTION path
    /// (synthesize_problem_bii): budget 3 exhausts after the first
    /// adopted witness — the partial invariant comes back instead of
    /// None, stays sound and verifies.
    #[test]
    fn test_problem_bii_budget_exhaustion_returns_partial() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!(
                "z3 unavailable — skipping test_problem_bii_budget_exhaustion_returns_partial"
            );
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &bw,
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let tpl = synthesize_problem_bii(&solver, &problem, 3, false)
            .expect("budget exhaustion must return the partial invariant, never None");
        assert!(tpl.rows[0].lb <= BigInt::zero(), "partial lb ≤ BII lb (0)");
        assert!(tpl.rows[0].ub >= BigInt::from(6), "partial ub ≥ BII ub (6)");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
    }

    /// With an 8-bit width, the bitwise strategy must not burn the
    /// full linear-search budget (2^8 = 256 tightenings) — it converges
    /// within a handful of queries. The assertion is deliberately loose
    /// (query budget far below 256) to keep the test robust across Z3
    /// versions while still proving the asymptotic improvement.
    #[test]
    fn test_bitwise_bii_query_count_is_logarithmic() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_bitwise_bii_query_count_is_logarithmic");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        // 8-bit width: linear search would need up to ~250 refinements,
        // bitwise greedy needs O(8) per bound. A budget of 40 (≈ 2·8·2 +
        // slack) must suffice.
        let tpl = synthesize_bitwise_bii(&solver, &vars, &init, &body, &[8], &[false], 40, false)
            .expect("bitwise BII must converge within an O(Σbw) query budget");
        assert_eq!(tpl.rows[0].ub, BigInt::from(6));
    }

    /// Z3 / SMT-LIB2 emit negative numerals in the FUNCTIONAL form
    /// `(- n)`, not `-n`. `parse_define_fun` must extract the negative
    /// value — regression: the old token scan terminated on the opening
    /// `(`, yielding `None`, so any witness with a negative bound (octagon
    /// `Diff` rows, negative intervals) failed closed.
    #[test]
    fn test_parse_define_fun_negative_numeral() {
        // Exact Z3 model format for a negative bound (verified by running
        // `z3` on a query with `(= x (- 0 5))` → `(define-fun x () Int (- 5))`.
        let model = "(set-info :status sat)\n(define-fun l_0 () Int\n  (- 5))\n(define-fun u_0 () Int\n  7)\n";
        assert_eq!(
            parse_define_fun(model, "l_0"),
            Some(BigInt::from(-5)),
            "SMT-LIB2 functional negative numeral `(- 5)` must parse as -5"
        );
        assert_eq!(
            parse_define_fun(model, "u_0"),
            Some(BigInt::from(7)),
            "positive numeral must still parse"
        );
        assert_eq!(
            parse_define_fun(model, "missing"),
            None,
            "absent name returns None"
        );
    }

    /// The transition relation must model SEQUENTIAL semantics —
    /// `x = x + 1; y = x;` gives `y` the NEW `x`, so the CopyVar must read
    /// an intermediate variable, not the pre-state `x_0`. A PARALLEL
    /// encoding (`(= xp_1 x_0)`) is unsound for read-after-write
    /// dependencies.
    #[test]
    fn test_encode_sequential_transition_reads_after_write() {
        // body: x := x + 1; y := x  (vars: [x, y] → indices 0, 1)
        let body = vec![LoopInstr::AddVar(0, 1), LoopInstr::CopyVar(1, 0)];
        let (inter, trans) = encode_sequential_transition(&body, 2, false, &[8, 8])
            .expect("transition encoding should succeed");
        // The CopyVar must read the intermediate value of x (xs_0_0), not
        // the pre-state x_0.
        assert!(
            inter.iter().any(|n| n == "xs_0_0"),
            "AddVar must create an intermediate variable: {:?}",
            inter
        );
        assert!(
            trans.contains("(= xs_1_1 xs_0_0)"),
            "CopyVar y := x must read the NEW x (xs_0_0), not pre-state x_0: {}",
            trans
        );
        assert!(
            !trans.contains("(= xp_1 x_0)"),
            "must NOT read the pre-state x_0 for y (parallel encoding): {}",
            trans
        );
        // The final equality maps xp back from the last intermediate value.
        assert!(trans.contains("(= xp_1 xs_1_1)"));
        assert!(trans.contains("(= xp_0 xs_0_0)"));
    }

    /// BV wrap-around: an 8-bit wrap-around counter `x := 255; while x ≤ 255 {
    /// x := x + 1 }`. Under QF_BV the successor wraps (255+1 = 0 mod 256),
    /// so the BII stays inside [0, 255]; under LIA the successor is
    /// unbounded (256 ∉ [0,255]), which the unbounded encoding cannot
    /// capture. Regression: LIA-only synthesis could not model the
    /// paper's bit-vector wrap-around semantics.
    #[test]
    fn test_bitwise_bii_bv_wraparound() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_bitwise_bii_bv_wraparound");
            return;
        }
        let vars = vec![Symbol::intern("x")];
        let init = vec![LoopInstr::ConstVar(0, 255)];
        // Guard `x ≤ 255` is always true in 8 bits; the body wraps.
        let body = vec![LoopInstr::TestLe(0, 255), LoopInstr::AddVar(0, 1)];
        let tpl = synthesize_bitwise_bii(&solver, &vars, &init, &body, &[8], &[false], 512, true)
            .expect("BV synthesis must converge with Z3 available");
        let row = &tpl.rows[0];
        assert_eq!(row.kind, RowKind::Interval(0));
        // Modular successor keeps the BII inside the 8-bit range — the
        // wrap-around semantics (255+1 = 0) force ub to stay at 255.
        assert!(
            row.ub <= BigInt::from(255),
            "wrap-around: BII ub must stay in [0, 255], got {}",
            row.ub
        );
        assert!(
            row.lb <= BigInt::from(255),
            "wrap-around: BII lb must stay in [0, 255], got {}",
            row.lb
        );
        assert_eq!(
            row.ub,
            BigInt::from(255),
            "the BII upper bound must be 255 (the wrap-around ceiling), got {}",
            row.ub
        );
    }

    /// Octagon `x,y := 0,0; while x < 4 { x,y := x+1, y+1 }` — the
    /// Diff row `x − y` is constantly 0, so its BII is the singleton
    /// `[0, 0]`. Exercises (a) the singleton-candidate allowance in
    /// `propose_bitwise` (Propose of Zuo et al. (2026) admits `l == u` hypotheses)
    /// and (b) the offset-encoded BV Diff encoding, whose witness is
    /// parsed back out of the encoded domain. Runs in both LIA and BV
    /// modes.
    #[test]
    fn test_bitwise_bii_octagon_diff_singleton() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_bitwise_bii_octagon_diff_singleton");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 3),
            LoopInstr::AddVar(0, 1),
            LoopInstr::AddVar(1, 1),
        ];
        let bw = vec![4u8, 4u8];
        for use_bv in [false, true] {
            let tpl = synthesize_bitwise_bii(
                &solver,
                &vars,
                &init,
                &body,
                &bw,
                &[false, false],
                512,
                use_bv,
            )
            .expect("octagon BII synthesis should converge with Z3 available");
            // Row order: Interval(x), Interval(y), Diff(x,y), Sum(x,y).
            let diff = &tpl.rows[2];
            assert_eq!(diff.kind, RowKind::Diff(0, 1));
            assert_eq!(diff.lb, BigInt::zero(), "x − y is constantly 0");
            assert_eq!(
                diff.ub,
                BigInt::zero(),
                "BII of a constant row is the singleton [0, 0] (use_bv={use_bv})"
            );
        }
    }

    /// Sparse template polyhedra — `x,y,z := 0,0,0; while x < 4 {
    /// x,y,z := x+1, y+1, z+1 }` keeps x = y = z, so each support-three
    /// row is linear in x: `x+y+z = 3x ∈ [0, 12]`, `x+y−z = x−y+z = x ∈
    /// [0, 4]`, `x−y−z = −x ∈ [−4, 0]`. Exercises the full 4·C(n,3)
    /// Support3 template in both LIA and BV modes (offset-encoded in BV).
    /// The LIA ∃∀ UNSAT queries are slow for Z3 at this template size
    /// (13 rows / ~26 candidate disjuncts): measured ≈17 min, so the test
    /// is gated behind `--features slow-tests` and excluded from the
    /// default (incl. `cargo test -r`) run.
    #[cfg(feature = "slow-tests")]
    #[test]
    fn test_bitwise_bii_support3_converges() {
        // 3 variables → 13 template rows → ~26 candidate disjuncts per
        // Refine query; the quantified LIA/BV ∃∀ queries exceed the 5 s
        // default solver timeout. Measured: Z3 needs ≈17 min for the LIA
        // UNSAT proof at this template size (300 s was insufficient), so
        // the per-instance timeout is 30 min.
        let solver = SmtSolver::with_timeout("z3", 1800_000);
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_bitwise_bii_support3_converges");
            return;
        }
        let vars = vec![
            Symbol::intern("x"),
            Symbol::intern("y"),
            Symbol::intern("z"),
        ];
        let init = vec![
            LoopInstr::ConstVar(0, 0),
            LoopInstr::ConstVar(1, 0),
            LoopInstr::ConstVar(2, 0),
        ];
        let body = vec![
            LoopInstr::TestLe(0, 3),
            LoopInstr::AddVar(0, 1),
            LoopInstr::AddVar(1, 1),
            LoopInstr::AddVar(2, 1),
        ];
        let bw = vec![4u8, 4u8, 4u8];
        for use_bv in [false, true] {
            let tpl = synthesize_bitwise_bii(
                &solver,
                &vars,
                &init,
                &body,
                &bw,
                &[false, false, false],
                1024,
                use_bv,
            )
            .expect("support-three BII synthesis should converge with Z3 available");
            let s3: Vec<&BiiRow> = tpl
                .rows
                .iter()
                .filter(|r| matches!(r.kind, RowKind::Support3(..)))
                .collect();
            assert_eq!(s3.len(), 4, "3 vars → 4·C(3,3) = 4 support-three rows");
            for row in s3 {
                let (sj, sk) = match row.kind {
                    RowKind::Support3(_, _, _, sj, sk) => (sj, sk),
                    _ => unreachable!(),
                };
                match (sj, sk) {
                    (true, true) => {
                        assert_eq!(row.lb, BigInt::zero());
                        assert_eq!(row.ub, BigInt::from(12), "x+y+z = 3x, max 3·4");
                    }
                    (true, false) => {
                        assert_eq!(row.lb, BigInt::zero());
                        assert_eq!(row.ub, BigInt::from(4), "x+y−z = x");
                    }
                    (false, true) => {
                        assert_eq!(row.lb, BigInt::zero());
                        assert_eq!(row.ub, BigInt::from(4), "x−y+z = x");
                    }
                    (false, false) => {
                        assert_eq!(row.lb, BigInt::from(-4), "x−y−z = −x");
                        assert_eq!(row.ub, BigInt::zero());
                    }
                }
            }
        }
    }

    /// BV-mode ConstVar bracket balance: `(= xs_0_0 (_ bv5 8))` must be
    /// emitted with BALANCED parentheses — a copy-paste regression once
    /// produced `(= xs_0_0 (_ bv5 8)))` (extra `)`), which Z3 rejects as
    /// a parse error and silently disabled BII synthesis for loops with a
    /// constant assignment under `use_bv: true`.
    #[test]
    fn test_encode_sequential_transition_constvar_bv_balanced() {
        // body: x := 5  (vars: [x] → index 0), 8-bit BV mode.
        let body = vec![LoopInstr::ConstVar(0, 5)];
        let (_inter, trans) = encode_sequential_transition(&body, 1, true, &[8])
            .expect("transition encoding should succeed");
        assert!(
            trans.contains("(= xs_0_0 (_ bv5 8))"),
            "ConstVar BV encoding must be balanced: {}",
            trans
        );
        assert!(
            !trans.contains("(_ bv5 8)))"),
            "must NOT emit an extra closing paren: {}",
            trans
        );
    }

    /// The bounded-leap query must contain the X→X' TRANSITION — a
    /// shadowing regression once spliced an EMPTY `trans` into the leap
    /// inductiveness formula (a `let mut trans = String::new();` after the
    /// SSA transition was computed), dropping the transition relation and
    /// letting the leap path accept non-inductive bounds.
    #[test]
    fn test_build_bounded_leap_query_includes_transition() {
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        let limits = BoundaryLimits::new(&tpl);
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::AddVar(0, 1)];
        let q = build_bounded_leap_query(&tpl, &limits, &init, &body, false, false, &[8], &[false])
            .expect("leap query should build");
        assert!(
            q.contains("xs_0_0"),
            "leap query must contain the SSA transition: {}",
            q
        );
        assert!(
            q.contains("(= xp_0 xs_0_0)"),
            "leap query must map xp back from the final intermediate: {}",
            q
        );
    }

    /// The bounded-leap query must emit the paper's `A′ ⊏ A` conjunct (as
    /// the strict-bound disjunction): dropping it admits the trivial
    /// witness A' = A and disables the Theorem-5.5 UNSAT termination.
    /// Regression pin for the `⊏` expansion.
    #[test]
    fn test_build_bounded_leap_query_has_strict_tightening() {
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        let limits = BoundaryLimits::new(&tpl);
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::AddVar(0, 1)];
        let q = build_bounded_leap_query(&tpl, &limits, &init, &body, false, false, &[8], &[false])
            .expect("leap query should build");
        // The strict tightening disjunction must be present:
        // (or (> l_0 ...) (< u_0 ...) ...)
        assert!(
            q.contains("(assert (or"),
            "leap query must contain the strict tightening disjunction: {}",
            q
        );
    }

    /// Sum rows are generated by `BiiTemplate::new` with correct top
    /// bounds `2·(2^bw − 1)`.
    #[test]
    fn test_template_new_includes_sum_rows() {
        let tpl = BiiTemplate::new(2, &[8, 8], &[false, false]);
        let sum_rows: Vec<&BiiRow> = tpl
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Sum(..)))
            .collect();
        assert_eq!(sum_rows.len(), 1, "2 vars → 1 sum row");
        let sum = sum_rows[0];
        assert_eq!(sum.kind, RowKind::Sum(0, 1));
        assert_eq!(sum.lb, BigInt::zero());
        assert_eq!(sum.ub, BiiRow::max_sum_ub(8));
        assert_eq!(sum.ub, BigInt::from(510)); // 2 * 255
    }

    /// `enc_bw` returns `bw + 1` for Diff and Sum rows, `bw` for
    /// Interval rows.
    #[test]
    fn test_enc_bw() {
        let interval = BiiRow {
            kind: RowKind::Interval(0),
            bw: 8,
            lb: BigInt::zero(),
            ub: BiiRow::max_ub(8),
            signed: false,
            full_lb: BigInt::zero(),
            full_ub: BiiRow::max_ub(8),
        };
        assert_eq!(interval.enc_bw(), 8);

        let diff = BiiRow {
            kind: RowKind::Diff(0, 1),
            bw: 8,
            lb: BiiRow::min_diff(8),
            ub: BiiRow::max_ub(8),
            signed: false,
            full_lb: BiiRow::min_diff(8),
            full_ub: BiiRow::max_ub(8),
        };
        assert_eq!(diff.enc_bw(), 9);

        let sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::zero(),
            ub: BiiRow::max_sum_ub(8),
            signed: false,
            full_lb: BigInt::zero(),
            full_ub: BiiRow::max_sum_ub(8),
        };
        assert_eq!(sum.enc_bw(), 9);
    }

    /// `is_trivial` correctly identifies top Sum rows.
    #[test]
    fn test_is_trivial_sum() {
        let top_sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::zero(),
            ub: BiiRow::max_sum_ub(8),
            signed: false,
            full_lb: BigInt::zero(),
            full_ub: BiiRow::max_sum_ub(8),
        };
        assert!(top_sum.is_trivial());

        let tight_sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::from(10),
            ub: BigInt::from(100),
            signed: false,
            full_lb: BigInt::zero(),
            full_ub: BiiRow::max_sum_ub(8),
        };
        assert!(!tight_sum.is_trivial());
    }

    /// Offset encoding for Diff rows: the offset maps `[-m, m]` to
    /// `[0, 2m]`, and the encoded value is always non-negative.
    #[test]
    fn test_diff_offset_encoding() {
        let diff = BiiRow {
            kind: RowKind::Diff(0, 1),
            bw: 4,
            lb: BigInt::from(-15),
            ub: BigInt::from(15),
            signed: false,
            full_lb: BiiRow::min_diff(4),
            full_ub: BiiRow::max_ub(4),
        };
        let offset = diff.offset();
        assert_eq!(offset, BigInt::from(15));
        let enc_lb = &diff.lb + &offset;
        assert_eq!(enc_lb, BigInt::zero());
        let enc_ub = &diff.ub + &offset;
        assert_eq!(enc_ub, BigInt::from(30));
        assert!(!enc_lb.is_negative());
        assert!(!enc_ub.is_negative());
    }

    /// Support-three rows: the four sign combinations have distinct value
    /// ranges, offsets map each onto `[0, 3m]`, and `enc_bw = bw + 2`
    /// fits `3m` bits.
    #[test]
    fn test_s3_range_and_offset() {
        let m = BiiRow::max_ub(4); // 15
        let cases = [
            // (sj, sk, min, max, offset)
            (
                true,
                true,
                BigInt::zero(),
                m.clone() * BigInt::from(3),
                BigInt::zero(),
            ),
            (
                true,
                false,
                -m.clone(),
                m.clone() * BigInt::from(2),
                m.clone(),
            ),
            (
                false,
                true,
                -m.clone(),
                m.clone() * BigInt::from(2),
                m.clone(),
            ),
            (
                false,
                false,
                -m.clone() * BigInt::from(2),
                m.clone(),
                m.clone() * BigInt::from(2),
            ),
        ];
        for (sj, sk, lo, hi, off) in cases {
            let (rlo, rhi) = full_range(
                RowKind::Support3(0, 1, 2, sj, sk),
                4,
                &[false, false, false],
            );
            assert_eq!(
                (rlo, rhi),
                (lo.clone(), hi.clone()),
                "range for (sj={sj}, sk={sk})"
            );
            let row = BiiRow {
                kind: RowKind::Support3(0, 1, 2, sj, sk),
                bw: 4,
                lb: lo.clone(),
                ub: hi.clone(),
                signed: false,
                full_lb: lo.clone(),
                full_ub: hi.clone(),
            };
            assert_eq!(row.offset(), off, "offset for (sj={sj}, sk={sk})");
            assert_eq!(row.enc_bw(), 6, "enc_bw for (sj={sj}, sk={sk})");
            // The encoded full range lands exactly on [0, 3m].
            let enc_lo = &row.lb + &row.offset();
            let enc_hi = &row.ub + &row.offset();
            assert!(!enc_lo.is_negative());
            assert_eq!(enc_hi, m.clone() * BigInt::from(3));
        }
    }

    /// `template_formula` Support3 rows: LIA mode emits plain linear
    /// arithmetic (no offset, no extension), BV mode emits zero-extended
    /// operands joined by `bvadd`/`bvsub` plus the offset constant. The
    /// LIA branch cannot be exercised end-to-end (Z3 cannot decide the
    /// quantified ∃∀ UNSAT queries at 3 variables), so it is pinned here
    /// at the formula level.
    #[test]
    fn test_template_formula_support3_lia_and_bv() {
        let rows = vec![BiiRow {
            kind: RowKind::Support3(0, 1, 2, true, false), // x + y − z
            bw: 4,
            lb: BigInt::zero(),
            ub: BigInt::from(4),
            signed: false,
            full_lb: BigInt::from(-15),
            full_ub: BigInt::from(30),
        }];
        let bws = [4u8, 4, 4];
        let f_lia = template_formula(&rows, 3, false, false, &bws, &[false, false, false]);
        assert!(
            f_lia.contains("(<= l_0 (- (+ x_0 x_1) x_2))"),
            "LIA Support3 term: {f_lia}"
        );
        assert!(
            !f_lia.contains("zero_extend"),
            "LIA must not extend: {f_lia}"
        );
        assert!(!f_lia.contains("bvadd"), "LIA must not use BV ops: {f_lia}");

        let f_bv = template_formula(&rows, 3, false, true, &bws, &[false, false, false]);
        // enc_bw = 6, each operand zero-extended by 2; (+,−) offset = m = 15.
        assert!(
            f_bv.contains(
                "(bvule l_0 (bvadd (bvsub (bvadd ((_ zero_extend 2) x_0) ((_ zero_extend 2) x_1)) ((_ zero_extend 2) x_2)) (_ bv15 6)))"
            ),
            "BV Support3 term: {f_bv}"
        );
    }

    /// `is_trivial` recognizes top (full-range) Support3 rows.
    #[test]
    fn test_is_trivial_support3() {
        let top = BiiRow {
            kind: RowKind::Support3(0, 1, 2, true, false),
            bw: 8,
            lb: -BiiRow::max_ub(8),
            ub: BiiRow::max_ub(8) * BigInt::from(2),
            signed: false,
            full_lb: BigInt::from(-255),
            full_ub: BigInt::from(510),
        };
        assert!(top.is_trivial(), "full-range Support3 row is trivial");
        let tight = BiiRow {
            kind: RowKind::Support3(0, 1, 2, true, false),
            bw: 8,
            lb: BigInt::from(0),
            ub: BigInt::from(4),
            signed: false,
            full_lb: BigInt::from(-255),
            full_ub: BigInt::from(510),
        };
        assert!(!tight.is_trivial(), "tightened Support3 row is not trivial");
    }

    /// `BoundaryLimits::new` seeds Support3 rows with their full value
    /// range (like Diff/Sum rows, not the Interval default).
    #[test]
    fn test_boundary_limits_support3() {
        let tpl = BiiTemplate::new(3, &[4, 4, 4], &[false, false, false]);
        let limits = BoundaryLimits::new(&tpl);
        for (idx, row) in tpl.rows.iter().enumerate() {
            if let RowKind::Support3(_, _, _, sj, sk) = row.kind {
                let (lo, hi) = (row.full_lb.clone(), row.full_ub.clone());
                assert_eq!(limits.lb[idx], hi, "lb limit for row {idx}");
                assert_eq!(limits.ub[idx], lo, "ub limit for row {idx}");
            }
        }
    }

    /// 128-bit support: `max_ub(128)` = 2^128 − 1, which exceeds
    /// `i128::MAX`.
    #[test]
    fn test_max_ub_128() {
        let m = BiiRow::max_ub(128);
        assert!(m > BigInt::from(i128::MAX));
        assert_eq!(m, (BigInt::one() << 128) - BigInt::one());
    }

    /// Sum row at 128-bit: `max_sum_ub(128)` = 2·(2^128 − 1) exceeds
    /// `i128::MAX`, so `template_to_invariant_exprs` must skip it.
    #[test]
    fn test_sum_128_skipped_in_ast() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let tpl = BiiTemplate {
            n_vars: 2,
            rows: vec![BiiRow {
                kind: RowKind::Sum(0, 1),
                bw: 128,
                lb: BigInt::from(1),
                ub: BiiRow::max_sum_ub(128), // > i128::MAX
                signed: false,
                full_lb: BigInt::zero(),
                full_ub: BiiRow::max_sum_ub(128),
            }],
        };
        let exprs = template_to_invariant_exprs(arena, &tpl, &vars);
        assert!(
            exprs.is_empty(),
            "Sum row with ub > i128::MAX must be skipped in AST output"
        );
    }

    /// `propose_bitwise` produces candidates for Sum rows (non-negative,
    /// no offset needed, but enc_bw = bw+1).
    #[test]
    fn test_propose_bitwise_sum_row() {
        let tpl = BiiTemplate::new(2, &[4, 4], &[false, false]);
        let pos = BitPositions::new(&tpl);
        let limits = BoundaryLimits::new(&tpl);
        let cands = propose_bitwise(&tpl, &pos, &limits);
        let sum_cands: Vec<&BiiTemplate> = cands
            .iter()
            .filter(|c| c.rows.iter().any(|r| matches!(r.kind, RowKind::Sum(..))))
            .collect();
        assert!(
            !sum_cands.is_empty(),
            "propose_bitwise must produce candidates for Sum rows"
        );
    }

    /// `BoundaryLimits::new` uses `max_sum_ub` for Sum rows' lb.
    #[test]
    fn test_boundary_limits_sum_lb() {
        let tpl = BiiTemplate::new(2, &[8, 8], &[false, false]);
        let limits = BoundaryLimits::new(&tpl);
        let sum_idx = tpl
            .rows
            .iter()
            .position(|r| matches!(r.kind, RowKind::Sum(..)))
            .expect("sum row must exist");
        assert_eq!(limits.lb[sum_idx], BiiRow::max_sum_ub(8));
    }

    /// `prev_unsat` flag: after SAT the pointers must NOT advance.
    /// This is a structural test — we verify that the `prev_unsat`
    /// flag is false after a SAT branch by checking that `propose_bitwise`
    /// produces candidates at the SAME positions (not advanced).
    #[test]
    fn test_prev_unsat_not_set_after_sat() {
        // This test verifies the logic structurally: after a SAT update,
        // prev_unsat is false, so pos.advance() is NOT called, and the
        // next propose_bitwise uses the same positions.
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        let pos_before = BitPositions::new(&tpl);
        // Simulate: after SAT, prev_unsat = false, so no advance.
        let pos_after_sat = pos_before.clone(); // no advance
        assert_eq!(pos_after_sat.lpos, pos_before.lpos);
        assert_eq!(pos_after_sat.upos, pos_before.upos);
        // After UNSAT, prev_unsat = true, so advance IS called.
        let mut pos_after_unsat = pos_before.clone();
        pos_after_unsat.advance();
        assert_ne!(pos_after_unsat.lpos, pos_before.lpos);
        assert_ne!(pos_after_unsat.upos, pos_before.upos);
    }

    /// `encode_sequential_transition` returns `Some` for valid instructions
    /// and handles all LoopInstr variants explicitly.
    #[test]
    fn test_encode_sequential_transition_all_variants() {
        let body = vec![
            LoopInstr::TestLe(0, 10),
            LoopInstr::AddVar(0, 1),
            LoopInstr::ConstVar(1, 42),
            LoopInstr::CopyVar(2, 0),
            LoopInstr::TestDiffLe(0, 1, 5),
        ];
        let result = encode_sequential_transition(&body, 3, false, &[8, 8, 8]);
        assert!(result.is_some(), "all LoopInstr variants must be handled");
        let (inter, trans) = result.unwrap();
        assert_eq!(inter.len(), 3); // AddVar, ConstVar, CopyVar
        assert!(trans.contains("xs_0_0"));
        assert!(trans.contains("xs_1_1"));
        assert!(trans.contains("xs_2_2"));
    }

    /// `bv_const` produces correct SMT-LIB2 for positive, zero, and
    /// negative values.
    #[test]
    fn test_bv_const() {
        assert_eq!(bv_const(&BigInt::from(5), 8), "(_ bv5 8)");
        assert_eq!(bv_const(&BigInt::zero(), 8), "(_ bv0 8)");
        // -5 @ 8 bits → two's complement 251.
        assert_eq!(bv_const(&BigInt::from(-5), 8), "(_ bv251 8)");
        assert_eq!(bv_const(&BigInt::from(255), 8), "(_ bv255 8)");
        // Out-of-width constants are reduced modulo 2^bw (300 mod 256 = 44).
        assert_eq!(bv_const(&BigInt::from(300), 8), "(_ bv44 8)");
    }

    /// `bv_const` with u32 bit-width (for enc_bw > 255).
    #[test]
    fn test_bv_const_wide() {
        assert_eq!(bv_const(&BigInt::from(42), 129), "(_ bv42 129)");
    }

    /// The DBM fast path proves the basic loop's template without
    /// z3 (`dbm_proves_inductiveness` needs no solver — this test runs
    /// even when z3 is unavailable). BII of `i := 0; while i < 6 {
    /// i := i + 1 }` is [0, 6]; a broken bound [0, 3] yields a
    /// satisfiable counter-query → Some(false) → the caller falls back
    /// to the SMT verifier.
    #[test]
    fn test_dbm_proves_basic_loop() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(6);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "the BII [0, 6] must be PROVEN inductive by DBM closure"
        );
        let mut bad = tpl.clone();
        bad.rows[0].ub = BigInt::from(3);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &bad),
            Some(false),
            "the broken [0, 3] has a satisfiable counter-query — SMT decides"
        );
    }

    /// Descending loop `i := 5; while i > 0 { i := i - 1 }` —
    /// BII [0, 5]; the TestGe guard is consumed by the DBM encoding.
    #[test]
    fn test_dbm_proves_descending_loop() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 5)];
        let body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(5);
        assert_eq!(dbm_proves_inductiveness(&problem, &tpl), Some(true));
    }

    /// With `if`-paths: `while i < 6 { if i < 3 { i = i + 1 } else
    /// { i = i + 2 } }`. The BII is [0, 7] (i = 5 takes the else arm →
    /// i' = 7 must be in the invariant); [0, 6] is NOT inductive — the
    /// path expansion catches both facts.
    #[test]
    fn test_dbm_proves_if_paths() {
        use crate::hir::loop_ir::{CmpOp, Cond, ScalarExpr};
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 5),
            LoopInstr::If(
                Box::new(Cond::Cmp {
                    op: CmpOp::Lt,
                    lhs: Box::new(ScalarExpr::Var(0)),
                    rhs: Box::new(ScalarExpr::Const(BigInt::from(3))),
                    signed: false,
                }),
                vec![LoopInstr::AddVar(0, 1)],
                vec![LoopInstr::AddVar(0, 2)],
            ),
        ];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(7);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "the if-path-expanded BII [0, 7] must be proven inductive"
        );
        let mut bad = tpl.clone();
        bad.rows[0].ub = BigInt::from(6);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &bad),
            Some(false),
            "[0, 6] misses the else-arm successor i' = 7"
        );
    }

    /// With params: `i := 0; while i < n { i := i + 1 }` with `n`
    /// an 8-bit unsigned param — the BII is [0, 255] (the guard bounds
    /// i ≤ n−1 ≤ 254, so i' ≤ 255). Exercises the param-range encoding
    /// AND the TestDiffLe (two-variable difference) guard encoding.
    #[test]
    fn test_dbm_proves_params() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        // i - n ≤ -1 ⟺ i < n (index 1 = the param `n` in template order).
        let body = vec![LoopInstr::TestDiffLe(0, 1, -1), LoopInstr::AddVar(0, 1)];
        let params = vec![crate::hir::loop_ir::BiiVar {
            symbol: Symbol::intern("n"),
            bw: 8,
            signed: false,
        }];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &params,
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(2, &[8, 8], &[false, false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(255);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "the param-referencing loop's BII must be proven inductive"
        );
    }

    /// Saturate transitions: the clamp is expanded by `expand_if_paths`
    /// into three linear paths (overflow to max, underflow to min,
    /// in-range successor), so the DBM path proves the loop directly —
    /// no SMT fallback.
    #[test]
    fn test_dbm_fallback_saturate() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddSat(0, 1)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(6);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "Saturate transitions are proven by the DBM path via 3-path expansion"
        );
    }

    /// Fragment fallback: Support3 rows yield `None` before any
    /// check runs (the fragment gate is first).
    #[test]
    fn test_dbm_fallback_support3() {
        use crate::hir::loop_ir::{
            BiiLoopProblem, BiiVar, Cond, EdgeKind, ScalarExpr, TransitionEdge,
        };
        let mk = |s: &str| BiiVar {
            symbol: Symbol::intern(s),
            bw: 4,
            signed: false,
        };
        let problem = BiiLoopProblem {
            vars: vec![mk("x"), mk("y"), mk("z")],
            params: vec![],
            init: vec![
                ScalarExpr::Const(BigInt::zero()),
                ScalarExpr::Const(BigInt::zero()),
                ScalarExpr::Const(BigInt::zero()),
            ],
            loop_guard: Cond::True,
            back_edges: vec![TransitionEdge {
                kind: EdgeKind::Back,
                guard: None,
                definedness: None,
                next_values: vec![ScalarExpr::Var(0), ScalarExpr::Var(1), ScalarExpr::Var(2)],
            }],
            exit_edges: vec![],
            saturates: vec![],
            post: None,
        };
        let tpl = BiiTemplate::new(3, &[4, 4, 4], &[false, false, false]);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            None,
            "Support3 rows fall back to the SMT verifier"
        );
    }

    /// DBM fast path ↔ SMT agreement (z3-gated): the DBM fast path's
    /// Some(true) coincides with the SMT verifier's Verified, and its
    /// Some(false) falls through to the SMT verdict (Counterexample).
    #[test]
    fn test_dbm_matches_smt_verifier() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_dbm_matches_smt_verifier");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(6);
        assert_eq!(dbm_proves_inductiveness(&problem, &tpl), Some(true));
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
        let mut bad = tpl.clone();
        bad.rows[0].ub = BigInt::from(3);
        assert_eq!(dbm_proves_inductiveness(&problem, &bad), Some(false));
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &bad, false),
            VerifyOutcome::Counterexample
        ));
    }

    /// Differential: a template inductive over INTEGERS but
    /// NOT over reals. Vars x, y (UInt<8>); init (0, 0); guard True;
    /// transition x' = y, y' = y. Rows: Interval(x) [0, 2],
    /// Interval(y) top, Diff(x,y) [−1, top], Sum(x,y) [0, 4].
    ///
    /// The premise derives the ODD bound 2y ≤ 5 (FW path (y−x ≤ 1) +
    /// (x+y ≤ 4); the even alternative via x's self-dual — (y−x ≤ 1) +
    /// (2x ≤ 4) + (y−x ≤ 1) = 2y ≤ 6 — is exactly one looser, so the
    /// odd bound is the TIGHTEST; Diff lb = −1 rather than 0 is what
    /// leaves the half-integer window open). The negated half-bounds
    /// ¬(x' ≤ 2) = 2x' ≥ 5 and ¬(x'+y' ≤ 4) = 2y ≥ 5 close the cycle
    /// −10 + 0 + 10 + 0 = 0 — NOT negative: the real model
    /// (x, y) = (1.5, 2.5) satisfies every premise row AND the
    /// negation. Strong closure reports Some(false) — SMT fallback.
    ///
    /// IntegerExact rounds both self-dual edges of the cycle: the
    /// premise-derived 2y ≤ 5 (stored 10) → 2y ≤ 4 (stored 8) and the
    /// negation 2x' ≥ 5 (stored −10) → 2x' ≥ 6 (stored −12 — the
    /// INTEGER negation ¬(x ≤ 2) = x ≥ 3). The cycle becomes
    /// −12 + 0 + 8 + 0 = −4 < 0 — a negative diagonal: the template is
    /// PROVEN inductive (Some(true), no solver round-trip). This is
    /// Figure-6 tension of Zuo et al. (2026) arriving through the transition:
    /// the equality chain 2x' = 2y = 5 pins the half-integer y = 2.5,
    /// which the rounding breaks.
    ///
    /// The broken variant (Interval(x) lb = 1, violated by the init
    /// x = 0) stays Some(false): the counter-model is a genuine point.
    #[test]
    fn test_dbm_integer_exact_differential() {
        use crate::hir::loop_ir::{
            BiiLoopProblem, BiiVar, Cond, EdgeKind, ScalarExpr, TransitionEdge,
        };
        let mk = |s: &str| BiiVar {
            symbol: Symbol::intern(s),
            bw: 8,
            signed: false,
        };
        let problem = BiiLoopProblem {
            vars: vec![mk("x"), mk("y")],
            params: vec![],
            init: vec![
                ScalarExpr::Const(BigInt::zero()),
                ScalarExpr::Const(BigInt::zero()),
            ],
            loop_guard: Cond::True,
            back_edges: vec![TransitionEdge {
                kind: EdgeKind::Back,
                guard: None,
                definedness: None,
                next_values: vec![ScalarExpr::Var(1), ScalarExpr::Var(1)], // x' = y, y' = y
            }],
            exit_edges: vec![],
            saturates: vec![],
            post: None,
        };
        let mut tpl = BiiTemplate::new(2, &[8, 8], &[false, false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(2); // Interval(x): 0 ≤ x ≤ 2
        tpl.rows[2].lb = BigInt::from(-1); // Diff(x,y): x − y ≥ −1 (y ≤ x + 1)
        tpl.rows[3].ub = BigInt::from(4); // Sum(x,y): x + y ≤ 4
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "IntegerExact must PROVE the integer-inductive template"
        );
        let mut bad = tpl.clone();
        bad.rows[0].lb = BigInt::from(1);
        assert_eq!(
            dbm_proves_inductiveness(&problem, &bad),
            Some(false),
            "the init counter-model (x = 0 < 1) is genuine — SMT decides"
        );
        // Cross-check (z3-gated): over LIA integers the counter-queries
        // are unsat (2y = (x+y)+(y−x) ≤ 5 ⟹ y ≤ 2), so the public
        // verifier agrees.
        let solver = SmtSolver::new("z3");
        if solver.check_version() {
            assert!(matches!(
                verify_template_against_problem(&solver, &problem, &tpl, false),
                VerifyOutcome::Verified
            ));
        }
    }

    /// End-to-end (trap semantics, guard implies def): `i := 0;
    /// while i < 240 { i := i + 10 }` on UInt<8> with trap lowering.
    /// def = i ≤ 245 (stay-in-range for +10) is implied by the guard
    /// i ≤ 239, so check 3 passes and the pipeline returns Verified.
    /// The BII is [0, 249]: i ∈ [231, 239] satisfy guard ∧ def and
    /// step to i' = 241..249 (ub ≥ 249), and ub = 249 is inductive
    /// (i ≤ 239 ⟹ i' ≤ 249). The z3-free half exercises the DBM fast
    /// path's ALL THREE checks — check 3's refutation is ¬def = i ≥ 246
    /// contradicting the premise's guard i ≤ 239.
    #[test]
    fn test_trap_absence_verified_end_to_end() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 239), LoopInstr::AddVar(0, 10)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        // The trap lowering generated the definedness condition.
        assert!(problem.back_edges[0].definedness.is_some());
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(249);
        // z3-free: DBM fast path proves inductiveness AND trap absence.
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "guard i ≤ 239 ⊂ def i ≤ 245 — all three checks proven"
        );
        let solver = SmtSolver::new("z3");
        if solver.check_version() {
            let syn = synthesize_problem_bii(&solver, &problem, 512, false)
                .expect("synthesis must converge");
            assert_eq!(syn.rows[0].lb, BigInt::zero());
            assert_eq!(syn.rows[0].ub, BigInt::from(249), "the BII is [0, 249]");
            assert!(matches!(
                verify_template_against_problem(&solver, &problem, &tpl, false),
                VerifyOutcome::Verified
            ));
        }
    }

    /// Trap semantics, domain-limited trap proof: `i := 0;
    /// while i < 250 { i := i + 10 }` on UInt<8> with trap lowering.
    /// def = i ≤ 245 is NOT implied by the guard i ≤ 249 (i ∈
    /// [246, 249] satisfy the guard but violate def), and the BII is
    /// the FULL TOP [0, 255]: the inductive step runs under the def
    /// antecedent (trap states have no successor), so [0, 255] is
    /// inductive and nothing tighter is (tightening ub below 255
    /// fails: i = 245 → i' = 255 must stay in A). The interval
    /// domain cannot express the stride — the actual reachable set
    /// {0, 10, …, 250} never traps — so check 3 fails as a DOMAIN
    /// limitation, not a synthesis bug: the outcome is TrapUnproven,
    /// NOT Counterexample. The DBM fast path returns Some(false) on
    /// the same input (its check 3 finds the satisfiable counter-
    /// query i ∈ [246, 249]) and defers to this SMT verdict.
    #[test]
    fn test_trap_absence_unproven_domain_limit() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 249), LoopInstr::AddVar(0, 10)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        assert!(problem.back_edges[0].definedness.is_some());
        // The top template IS the BII here (def-antecedent inductiveness).
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        // z3-free: DBM fast path check 3 finds the satisfiable
        // counter-query (A = [0, 255] ∧ G = i ≤ 249 admits i = 247 ≥
        // 246 = ¬def), so the fast path defers — never Some(true).
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(false),
            "check 3's counter-query is satisfiable — defer to SMT"
        );
        let solver = SmtSolver::new("z3");
        if solver.check_version() {
            let syn = synthesize_problem_bii(&solver, &problem, 512, false)
                .expect("synthesis must converge");
            assert_eq!(syn.rows[0].lb, BigInt::zero());
            assert_eq!(
                syn.rows[0].ub,
                BigInt::from(255),
                "the BII is the top [0, 255]"
            );
            // Checks 1–2 pass (the BII is inductive); check 3 fails
            // (the domain cannot prove trap absence) → TrapUnproven.
            assert!(matches!(
                verify_template_against_problem(&solver, &problem, &tpl, false),
                VerifyOutcome::TrapUnproven
            ));
        }
    }

    /// Signed Sum/Support3 rows over uniform-signedness operand
    /// sets. Two Int<8> vars: the Sum row carries TRUE signed tops
    /// [−2^bw, 2^bw−2]; the Diff row keeps the symmetric [−255, 255].
    /// Mixed pairs get EXACT asymmetric tops — sound in both modes since
    /// the per-operand extension.
    #[test]
    fn test_g3_signed_row_generation() {
        let tpl = BiiTemplate::new(2, &[8, 8], &[true, true]);
        assert_eq!(tpl.rows.len(), 4);
        let diff = &tpl.rows[2];
        assert_eq!(diff.kind, RowKind::Diff(0, 1));
        assert_eq!(
            diff.lb,
            BigInt::from(-255),
            "uniform signed Diff keeps [−m, m]"
        );
        assert_eq!(diff.ub, BigInt::from(255));
        let sum = &tpl.rows[3];
        assert_eq!(sum.kind, RowKind::Sum(0, 1));
        assert!(sum.signed);
        assert_eq!(sum.lb, BigInt::from(-256), "−2^8");
        assert_eq!(sum.ub, BigInt::from(254), "2^8 − 2");
        // Three signed vars: 3 Interval + 3 Diff + 3 Sum + 4 Support3.
        let tpl3 = BiiTemplate::new(3, &[8, 8, 8], &[true, true, true]);
        assert_eq!(tpl3.rows.len(), 13);
        // h = 128: (+,+) → [−384, 381]; (+,−)/(−,+) → [−383, 382];
        // (−,−) → [−382, 383].
        assert_eq!(
            full_range(
                RowKind::Support3(0, 1, 2, true, true),
                8,
                &[true, true, true]
            ),
            (BigInt::from(-384), BigInt::from(381))
        );
        assert_eq!(
            full_range(
                RowKind::Support3(0, 1, 2, true, false),
                8,
                &[true, true, true]
            ),
            (BigInt::from(-383), BigInt::from(382))
        );
        assert_eq!(
            full_range(
                RowKind::Support3(0, 1, 2, false, false),
                8,
                &[true, true, true]
            ),
            (BigInt::from(-382), BigInt::from(383))
        );
        for row in &tpl3.rows {
            if let RowKind::Support3(_, _, _, sj, sk) = row.kind {
                let (lo, hi) = (row.full_lb.clone(), row.full_ub.clone());
                assert_eq!(row.lb, lo, "Support3 (sj={sj}, sk={sk}) tops");
                assert_eq!(row.ub, hi);
                assert!(row.signed);
            }
        }
        // Mixed pair: no Diff, no Sum (uniform-signedness policy).
        let tpl_m = BiiTemplate::new(2, &[8, 8], &[true, false]);
        let diff_m = tpl_m
            .rows
            .iter()
            .find(|r| matches!(r.kind, RowKind::Diff(0, 1)))
            .expect("mixed Diff row exists");
        assert_eq!(diff_m.lb, BigInt::from(-383), "Int8 − UInt8 bottom");
        assert_eq!(diff_m.ub, BigInt::from(127), "Int8 − UInt8 top");
        let sum_m = tpl_m
            .rows
            .iter()
            .find(|r| matches!(r.kind, RowKind::Sum(0, 1)))
            .expect("mixed Sum row exists");
        assert_eq!(sum_m.lb, BigInt::from(-128));
        assert_eq!(sum_m.ub, BigInt::from(382));
        // Mixed triple: the (0,1) uniform pair keeps its rows.
        let tpl_m3 = BiiTemplate::new(3, &[8, 8, 8], &[true, true, false]);
        assert_eq!(
            tpl_m3
                .rows
                .iter()
                .filter(|r| matches!(r.kind, RowKind::Support3(..)))
                .count(),
            4,
            "the mixed triple now carries its 4 Support3 rows"
        );
        assert!(
            tpl_m3
                .rows
                .iter()
                .any(|r| matches!(r.kind, RowKind::Diff(0, 1)))
        );
        assert!(
            tpl_m3
                .rows
                .iter()
                .any(|r| matches!(r.kind, RowKind::Sum(0, 1)))
        );
        // Unsigned pairs: unchanged [0, 2·(2^bw−1)].
        let tpl_u = BiiTemplate::new(2, &[8, 8], &[false, false]);
        let sum_u = &tpl_u.rows[3];
        assert_eq!(sum_u.lb, BigInt::zero());
        assert_eq!(sum_u.ub, BigInt::from(510));
        assert!(!sum_u.signed);
    }

    /// Offset/enc_bw for signed rows. Signed Sum @ bw 8: tops
    /// [−256, 254], offset 256 maps onto [0, 510] — the 9-bit (bw+1)
    /// offset domain, same width as the unsigned Sum. Signed Support3
    /// (+,+) @ bw 8: tops [−384, 381], offset 384 maps onto [0, 765],
    /// enc_bw = 10 (765 < 2^10).
    #[test]
    fn test_g3_signed_offset_enc_bw() {
        let sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::from(-256),
            ub: BigInt::from(254),
            signed: true,
            full_lb: BigInt::from(-256),
            full_ub: BigInt::from(254),
        };
        assert_eq!(sum.offset(), BigInt::from(256));
        assert_eq!(sum.enc_bw(), 9);
        let enc_lb = &sum.lb + &sum.offset();
        let enc_ub = &sum.ub + &sum.offset();
        assert_eq!(enc_lb, BigInt::zero());
        assert_eq!(enc_ub, BigInt::from(510));
        assert!(!enc_lb.is_negative());
        let s3 = BiiRow {
            kind: RowKind::Support3(0, 1, 2, true, true),
            bw: 8,
            lb: BigInt::from(-384),
            ub: BigInt::from(381),
            signed: true,
            full_lb: BigInt::from(-384),
            full_ub: BigInt::from(381),
        };
        assert_eq!(s3.offset(), BigInt::from(384));
        assert_eq!(s3.enc_bw(), 10);
        assert_eq!(&s3.ub + &s3.offset(), BigInt::from(765));
    }

    /// `is_trivial` at the signed tops; tightened rows are not
    /// trivial; `BoundaryLimits::new` seeds the signed ranges.
    #[test]
    fn test_g3_signed_is_trivial_and_limits() {
        let top_sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::from(-256),
            ub: BigInt::from(254),
            signed: true,
            full_lb: BigInt::from(-256),
            full_ub: BigInt::from(254),
        };
        assert!(top_sum.is_trivial());
        let tight_sum = BiiRow {
            kind: RowKind::Sum(0, 1),
            bw: 8,
            lb: BigInt::from(-10),
            ub: BigInt::from(10),
            signed: true,
            full_lb: BigInt::from(-256),
            full_ub: BigInt::from(254),
        };
        assert!(!tight_sum.is_trivial());
        let tpl = BiiTemplate::new(2, &[8, 8], &[true, true]);
        let limits = BoundaryLimits::new(&tpl);
        let sum_idx = tpl
            .rows
            .iter()
            .position(|r| matches!(r.kind, RowKind::Sum(..)))
            .expect("sum row");
        assert_eq!(limits.lb[sum_idx], BigInt::from(254), "max ub = 2^8 − 2");
        assert_eq!(limits.ub[sum_idx], BigInt::from(-256), "min lb = −2^8");
        let tpl3 = BiiTemplate::new(3, &[8, 8, 8], &[true, true, true]);
        let limits3 = BoundaryLimits::new(&tpl3);
        for (idx, row) in tpl3.rows.iter().enumerate() {
            if let RowKind::Support3(_, _, _, sj, sk) = row.kind {
                let (lo, hi) = (row.full_lb.clone(), row.full_ub.clone());
                assert_eq!(limits3.lb[idx], hi);
                assert_eq!(limits3.ub[idx], lo);
            }
        }
    }

    /// `propose_bitwise` produces candidates for signed Sum rows —
    /// the offset-domain walk (shift = offset = 2^bw, enc_lb ≥ 0 at
    /// the tops) exercises the same machinery as Diff rows.
    #[test]
    fn test_g3_propose_bitwise_signed_sum() {
        let tpl = BiiTemplate::new(2, &[4, 4], &[true, true]);
        let pos = BitPositions::new(&tpl);
        let limits = BoundaryLimits::new(&tpl);
        let cands = propose_bitwise(&tpl, &pos, &limits);
        assert!(
            cands
                .iter()
                .any(|c| c.rows.iter().any(|r| matches!(r.kind, RowKind::Sum(..)))),
            "propose_bitwise must produce candidates for signed Sum rows"
        );
    }

    /// End-to-end (z3-gated): a signed two-variable loop with a
    /// constant sum — `x,y := 0,0; while x < 4 { x,y := x+1, y−1 }` on
    /// Int<8>. Sum(x,y)'s BII is the singleton [0, 0] (x+y is
    /// invariant); Interval(y) = [−4, 0] (y = −x, exit successor −4);
    /// Diff(x,y) = 2x ∈ [0, 8]. Runs in LIA and BV modes — the BV mode
    /// exercises the sign-extended encoding: with the OLD
    /// zero-extension the state (x,y) = (1,−1) encoded x+y as
    /// zx(1)+zx(−1) = 1+255 = 256, +offset 256 → 512 ≡ 0 (mod 512)
    /// instead of the true 256 — the [0,0] row would be judged
    /// violated and the BII could not converge.
    #[test]
    fn test_g3_signed_sum_bii_singleton() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_g3_signed_sum_bii_singleton");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 3),
            LoopInstr::AddVar(0, 1),
            LoopInstr::AddVar(1, -1),
        ];
        for use_bv in [false, true] {
            let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
                &vars,
                &init,
                &body,
                &[8, 8],
                &[true, true],
                &[],
                !use_bv,
            )
            .expect("must lower");
            let tpl = synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("signed-pair synthesis must converge");
            // Row order: Interval(x), Interval(y), Diff(x,y), Sum(x,y).
            assert_eq!(
                tpl.rows[1].lb,
                BigInt::from(-4),
                "Interval(y) lb (use_bv={use_bv})"
            );
            assert_eq!(
                tpl.rows[1].ub,
                BigInt::zero(),
                "Interval(y) ub (use_bv={use_bv})"
            );
            assert_eq!(
                tpl.rows[2].lb,
                BigInt::zero(),
                "Diff = 2x lb (use_bv={use_bv})"
            );
            assert_eq!(
                tpl.rows[2].ub,
                BigInt::from(8),
                "Diff = 2x ub (use_bv={use_bv})"
            );
            assert_eq!(
                tpl.rows[3].lb,
                BigInt::zero(),
                "Sum(x,y) is constantly 0 (use_bv={use_bv})"
            );
            assert_eq!(
                tpl.rows[3].ub,
                BigInt::zero(),
                "Sum BII is the singleton [0, 0] (use_bv={use_bv})"
            );
            // Whole-template verification: LIA's trap def (x ≤ 126,
            // y ≥ −127) is implied by the converged bounds; BV has no
            // def — both Verified.
            assert!(matches!(
                verify_template_against_problem(&solver, &problem, &tpl, use_bv),
                VerifyOutcome::Verified
            ));
        }
    }

    /// Regression (z3-gated): the signed Diff row under BV — the
    /// zero-extension hazard. `x,y := −1, 0; while x < y { x := x+1 }`
    /// on Int<8>: reachable x = −1 plus the exit successor 0, so
    /// Diff(x,y) = x − 0 ∈ [−1, 0]. Under ZERO-extension the
    /// encoding computed zx(−1) = 255 (9-bit) instead of the true −1
    /// (sign-extended 511 ≡ −1 mod 512): the Diff value encoded as
    /// 255 − 0 + 255 = 510 instead of −1 + 255 = 254, misjudging the
    /// [254, 255] bound window. LIA mode is the control (no extension
    /// — passes before AND after the fix).
    #[test]
    fn test_g3_signed_diff_bv_sign_extension() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_g3_signed_diff_bv_sign_extension");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, -1), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestDiffLe(0, 1, -1), // x < y
            LoopInstr::AddVar(0, 1),
        ];
        for use_bv in [false, true] {
            let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
                &vars,
                &init,
                &body,
                &[8, 8],
                &[true, true],
                &[],
                !use_bv,
            )
            .expect("must lower");
            let tpl = synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("signed Diff synthesis must converge");
            let diff = &tpl.rows[2];
            assert_eq!(diff.kind, RowKind::Diff(0, 1));
            assert_eq!(
                diff.lb,
                BigInt::from(-1),
                "Diff lb (use_bv={use_bv}) — the sign-extension fix"
            );
            assert_eq!(diff.ub, BigInt::zero(), "Diff ub (use_bv={use_bv})");
        }
    }

    /// Regression: the BV clamp compares the ADDITION-PRE operand, never
    /// the WRAPPED `bvadd` result.  UInt8 x=250, c=10: `bvadd` wraps to
    /// 260 mod 256 = 4, which passes both bounds and yields successor 4
    /// instead of the saturated 255 — a wrapped-sum comparison silently
    /// mis-saturates at the boundary (x=255,c=1 → 0; Int8 x=127,c=1 →
    /// −128).  The comparison constants (MAX−c / MIN−c, per sign) are
    /// representable because `c` is a compile-time constant.
    #[test]
    fn test_clamp_expr_bv_compares_pre_add() {
        // UInt8, c=10 > 0: only the upper bound can cross —
        // `(ite (bvugt x 245) 255 (bvadd x 10))`.
        assert_eq!(
            clamp_expr("x", 10, true, 8, false),
            "(ite (bvugt x (_ bv245 8)) (_ bv255 8) (bvadd x (_ bv10 8)))"
        );
        // Int8, c=1 > 0: `x > 126 → 127` — a wrapped-sum comparison lets
        // 127+1 = −128 through both bounds.
        assert_eq!(
            clamp_expr("x", 1, true, 8, true),
            "(ite (bvsgt x (_ bv126 8)) (_ bv127 8) (bvadd x (_ bv1 8)))"
        );
        // Int8, c=−1 < 0: only the lower bound can cross —
        // `x < −127 → −128` (MIN−c = −127; only x = −128 saturates).
        assert_eq!(
            clamp_expr("x", -1, true, 8, true),
            "(ite (bvslt x (_ bv129 8)) (_ bv128 8) (bvadd x (_ bv255 8)))"
        );
        // c == 0: identity — no boundary can be crossed.
        assert_eq!(clamp_expr("x", 0, true, 8, false), "x");
    }

    /// Regression (z3-gated): a signed Interval row whose BII UPPER bound
    /// is NEGATIVE — `y := −2; while y > −9 { y := y − 1 }` on Int<8>,
    /// BII = [−9, −2].  `BoundaryLimits` must seed the signed Interval
    /// row's ub at −2^(bw−1) (not 0): a 0 seed asserts "u* ≥ 0", which
    /// is false here and would empty the bounded-leap search region
    /// (leap UNSAT → premature termination on a sound-but-not-best
    /// invariant).
    #[test]
    fn test_g3_signed_interval_negative_bii_ub() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_g3_signed_interval_negative_bii_ub");
            return;
        }
        let vars = vec![Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, -2)];
        let body = vec![LoopInstr::TestGe(0, -8), LoopInstr::AddVar(0, -1)];
        for use_bv in [false, true] {
            let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
                &vars,
                &init,
                &body,
                &[8],
                &[true],
                &[],
                !use_bv,
            )
            .expect("must lower");
            let tpl = synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("signed negative-ub BII synthesis must converge");
            let row = &tpl.rows[0];
            assert_eq!(row.kind, RowKind::Interval(0));
            assert_eq!(
                row.lb,
                BigInt::from(-9),
                "lb = −9 (use_bv={use_bv}) — the exit successor"
            );
            assert_eq!(
                row.ub,
                BigInt::from(-2),
                "ub = −2 (use_bv={use_bv}) — negative BII upper bound"
            );
        }
    }

    /// Mixed-signedness relational rows: Int<8>/UInt<8> gets a
    /// Diff row with EXACT asymmetric tops [−383, 127] (min_x − max_y,
    /// max_x − min_y) and a Sum row [−128, 382]; offsets map both onto
    /// [0, 510] at enc_bw = 9.
    #[test]
    fn test_mixed_row_generation_tops() {
        let tpl = BiiTemplate::new(2, &[8, 8], &[true, false]);
        let diff = tpl
            .rows
            .iter()
            .find(|r| matches!(r.kind, RowKind::Diff(0, 1)))
            .expect("mixed Diff row must exist");
        assert_eq!(
            diff.full_lb,
            BigInt::from(-383),
            "Int8 − UInt8 bottom: −128 − 255"
        );
        assert_eq!(diff.full_ub, BigInt::from(127), "Int8 − UInt8 top: 127 − 0");
        assert_eq!(diff.offset(), BigInt::from(383));
        assert_eq!(diff.enc_bw(), 9);
        let sum = tpl
            .rows
            .iter()
            .find(|r| matches!(r.kind, RowKind::Sum(0, 1)))
            .expect("mixed Sum row must exist");
        assert_eq!(
            (sum.full_lb.clone(), sum.full_ub.clone()),
            (BigInt::from(-128), BigInt::from(382))
        );
        assert_eq!(sum.offset(), BigInt::from(128));
        assert_eq!(sum.enc_bw(), 9);
        assert_eq!(&sum.full_ub + &sum.offset(), BigInt::from(510));
    }

    /// Mixed Support3: vars (Int<8>, UInt<8>, Int<8>) — all four sign rows
    /// over the triple with exact per-term tops; enc_bw = bw + 2 still fits
    /// (each operand spreads ≤ 2^bw − 1 regardless of signedness).
    #[test]
    fn test_mixed_support3_tops() {
        let tpl = BiiTemplate::new(3, &[8, 8, 8], &[true, false, true]);
        let s3: Vec<&BiiRow> = tpl
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Support3(..)))
            .collect();
        assert_eq!(s3.len(), 4, "one triple → 4 sign combinations");
        for row in s3 {
            let (sj, sk) = match row.kind {
                RowKind::Support3(_, _, _, sj, sk) => (sj, sk),
                _ => unreachable!(),
            };
            // terms: x0 ∈ [−128, 127]; ±x1 ∈ [0,255] / [−255,0]; ±x2 ∈ [−128,127] / [−127,128].
            let (lo_j, hi_j) = if sj {
                (BigInt::from(0), BigInt::from(255))
            } else {
                (BigInt::from(-255), BigInt::from(0))
            };
            let (lo_k, hi_k) = if sk {
                (BigInt::from(-128), BigInt::from(127))
            } else {
                (BigInt::from(-127), BigInt::from(128))
            };
            assert_eq!(
                row.full_lb,
                BigInt::from(-128) + lo_j + lo_k,
                "(sj={sj}, sk={sk}) bottom"
            );
            assert_eq!(
                row.full_ub,
                BigInt::from(127) + hi_j + hi_k,
                "(sj={sj}, sk={sk}) top"
            );
            assert_eq!(row.enc_bw(), 10);
            assert!(
                &row.full_ub - &row.full_lb < BigInt::one() << 10,
                "encoded width fits"
            );
        }
    }

    /// End-to-end (z3-gated, LIA only): lockstep mixed pair —
    /// `x: Int<8>, y: UInt<8>; x,y := 0,0; while x ≤ 3 { x := x+1; y := y+1 }`
    /// keeps x − y = 0, so the mixed Diff row's BII is the singleton [0, 0].
    /// The template is synthesized under the mixed LIA domain and verified
    /// by the independent verifier (its DBM fast path is mathematical and
    /// handles mixed signedness natively).
    #[test]
    fn test_mixed_diff_bii_singleton_lia() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_mixed_diff_bii_singleton_lia");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 3),
            LoopInstr::AddVar(0, 1),
            LoopInstr::AddVar(1, 1),
        ];
        let tpl = synthesize_bitwise_bii(
            &solver,
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            512,
            false,
        )
        .expect("mixed-pair LIA synthesis must converge");
        let diff = tpl
            .rows
            .iter()
            .find(|r| matches!(r.kind, RowKind::Diff(0, 1)))
            .expect("mixed Diff row must be present in the LIA template");
        assert_eq!(diff.lb, BigInt::zero(), "x − y is constantly 0 (lb)");
        assert_eq!(diff.ub, BigInt::zero(), "x − y is constantly 0 (ub)");
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            &[],
            true,
        )
        .expect("must lower");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
    }

    /// Per-operand extension, pinned at the formula level (no solver needed):
    /// the mixed Diff row's BV emission
    /// sign-extends the Int<8> side, zero-extends the UInt<8> side, and
    /// compares in the offset domain at enc_bw 9 (offset = −full_lb = 383).
    #[test]
    fn test_mixed_diff_bv_formula_text() {
        let rows = vec![BiiRow {
            kind: RowKind::Diff(0, 1),
            bw: 8,
            lb: BigInt::from(-383),
            ub: BigInt::from(127),
            signed: true, // OR of operands — NOT consumed by extension
            full_lb: BigInt::from(-383),
            full_ub: BigInt::from(127),
        }];
        let f = template_formula(&rows, 2, false, true, &[8, 8], &[true, false]);
        assert!(
            f.contains("(bvule l_0 (bvadd (bvsub ((_ sign_extend 1) x_0) ((_ zero_extend 1) x_1)) (_ bv383 9)))"),
            "per-operand extension text: {f}"
        );
    }

    /// Semantic pins for the value-preserving lift (z3-gated) — each query fixes
    /// both variables and asks z3 whether the encoded guard agrees with the
    /// mathematical truth. Regression-pins the two defects the lift replaces:
    /// the unsigned diff guard
    /// `i < n` was VACUOUS under the old bvule encoding (c = −1 became the
    /// all-ones pattern), and the same-width signed diff wrapped (−255 → 1).
    #[test]
    fn test_guard_bv_lift_semantic_pins() {
        use crate::hir::loop_ir::{ArithSem, CmpOp, Cond, ScalarExpr};
        let mk_cond = |signed: bool| Cond::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ScalarExpr::Sub(
                Box::new(ScalarExpr::Var(0)),
                Box::new(ScalarExpr::Var(1)),
                ArithSem::Wrap,
            )),
            rhs: Box::new(ScalarExpr::Const(BigInt::from(-1))),
            signed, // ignored by the new encoding — kept for the IR
        };
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_guard_bv_lift_semantic_pins");
            return;
        }
        // (cond, x0_pattern, x1_pattern, guard_must_be_true)
        let unsigned = ([false, false], mk_cond(false));
        let signed = ([true, true], mk_cond(true));
        for ((sgn, cond), (p0, p1, want_true)) in [
            (&unsigned, ("(_ bv0 8)", "(_ bv255 8)", true)), // 0 < 255
            (&unsigned, ("(_ bv255 8)", "(_ bv0 8)", false)), // ¬(255 < 0)
            (&signed, ("(_ bv128 8)", "(_ bv127 8)", true)), // −128−127 = −255 ≤ −1 (old encoding wrapped to 1 → false)
            (&signed, ("(_ bv127 8)", "(_ bv128 8)", false)), // 255 ≤ −1 false (old encoding judged true)
        ] {
            let g = cond_to_smt(cond, true, &[8, 8], sgn, 2).expect("must encode");
            // Refutation-style construction: prove "g evaluates to want_true
            // at this pinned point" — assert its negation is unsatisfiable
            // (consistent with verify_template_against_problem's convention).
            let assert_g = if want_true {
                format!("(not {g})")
            } else {
                g.clone()
            };
            let q = format!(
                "(set-logic BV)\n(declare-const x_0 (_ BitVec 8))\n(declare-const x_1 (_ BitVec 8))\n\
                 (assert (and (= x_0 {p0}) (= x_1 {p1})))\n(assert {assert_g})\n(check-sat)\n"
            );
            assert!(
                matches!(solver.run_raw_query(&q), RawQueryOutcome::Unsat),
                "guard must evaluate {} at ({p0}, {p1}): encoded as {g}",
                if want_true { "true" } else { "false" }
            );
        }
    }

    /// End-to-end BV (z3-gated): mixed lockstep pair
    /// `x: Int<8>, y: UInt<8>; while x ≤ 3 { x := x+1; y := y+1 }`.
    /// Row order [Interval(x), Interval(y), Diff, Sum]; the BII is
    /// Ix [0,4], Iy [0,4], Diff [0,0], Sum [0,8]; the independent verifier
    /// (SMT path — the DBM fast path is gated !bv) re-confirms Verified.
    #[test]
    fn test_mixed_diff_bii_singleton_bv() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_mixed_diff_bii_singleton_bv");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 3),
            LoopInstr::AddVar(0, 1),
            LoopInstr::AddVar(1, 1),
        ];
        let tpl = synthesize_bitwise_bii(
            &solver,
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            512,
            true,
        )
        .expect("mixed-pair BV synthesis must converge");
        assert_eq!(
            (tpl.rows[0].lb.clone(), tpl.rows[0].ub.clone()),
            (BigInt::from(0), BigInt::from(4))
        );
        assert_eq!(
            (tpl.rows[1].lb.clone(), tpl.rows[1].ub.clone()),
            (BigInt::from(0), BigInt::from(4))
        );
        assert_eq!(
            (tpl.rows[2].lb.clone(), tpl.rows[2].ub.clone()),
            (BigInt::from(0), BigInt::from(0)),
            "Diff singleton"
        );
        assert_eq!(
            (tpl.rows[3].lb.clone(), tpl.rows[3].ub.clone()),
            (BigInt::from(0), BigInt::from(8)),
            "Sum = 2x ∈ [0,8]"
        );
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            &[],
            false,
        )
        .expect("must lower");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, true),
            VerifyOutcome::Verified
        ));
    }

    /// End-to-end BV with a negative mixed domain (z3-gated) —
    /// `x: Int<8> := 0; y: UInt<8> := 0; while y ≤ 99 { x := x−1; y := y+1 }`.
    /// Head states (−k, k), k ∈ [0, 100]: Ix [−100, 0], Iy [0, 100],
    /// Diff = −2k ∈ [−200, 0], Sum ≡ 0 (singleton). Exercises the offset
    /// domain around a negative window AND the unsigned single-var guard
    /// lift under BV.
    #[test]
    fn test_mixed_sum_diff_negative_bv() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_mixed_sum_diff_negative_bv");
            return;
        }
        let vars = vec![Symbol::intern("x"), Symbol::intern("y")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![
            LoopInstr::TestLe(1, 99),
            LoopInstr::AddVar(0, -1),
            LoopInstr::AddVar(1, 1),
        ];
        let tpl = synthesize_bitwise_bii(
            &solver,
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            512,
            true,
        )
        .expect("negative mixed-pair BV synthesis must converge");
        assert_eq!(
            (tpl.rows[0].lb.clone(), tpl.rows[0].ub.clone()),
            (BigInt::from(-100), BigInt::from(0))
        );
        assert_eq!(
            (tpl.rows[1].lb.clone(), tpl.rows[1].ub.clone()),
            (BigInt::from(0), BigInt::from(100))
        );
        assert_eq!(
            (tpl.rows[2].lb.clone(), tpl.rows[2].ub.clone()),
            (BigInt::from(-200), BigInt::from(0)),
            "Diff = −2k"
        );
        assert_eq!(
            (tpl.rows[3].lb.clone(), tpl.rows[3].ub.clone()),
            (BigInt::from(0), BigInt::from(0)),
            "Sum ≡ 0"
        );
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8, 8],
            &[true, false],
            &[],
            false,
        )
        .expect("must lower");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, true),
            VerifyOutcome::Verified
        ));
    }

    /// Check 4: postcondition entailed by the BII → Verified.
    /// `i := 0; while i < 6 { i := i + 1 }`, BII = [0, 6].
    /// Exit ¬(i < 6) ∧ 0 ≤ i ≤ 6 → i = 6. Post = i ≤ 10 → entailed.
    #[test]
    fn test_check4_post_implied() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_check4_post_implied");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let mut problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        // Post: i ≤ 10 (entailed by BII [0,6])
        problem.post = Some(Cond::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ScalarExpr::Var(0)),
            rhs: Box::new(ScalarExpr::Const(BigInt::from(10))),
            signed: false,
        });
        let tpl = synthesize_problem_bii(&solver, &problem, 512, false).expect("must synthesize");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
    }

    /// Check 4: postcondition NOT entailed by the BII → PostUnproven.
    /// Post = i == 100 (BII [0,6] does not entail it).
    #[test]
    fn test_check4_post_unproven() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_check4_post_unproven");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let mut problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        problem.post = Some(Cond::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ScalarExpr::Var(0)),
            rhs: Box::new(ScalarExpr::Const(BigInt::from(100))),
            signed: false,
        });
        let tpl = synthesize_problem_bii(&solver, &problem, 512, false).expect("must synthesize");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::PostUnproven
        ));
    }

    /// Check 4: post = None → skipped, does not affect the result.
    #[test]
    fn test_check4_post_none_skipped() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_check4_post_none_skipped");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        assert!(problem.post.is_none());
        let tpl = synthesize_problem_bii(&solver, &problem, 512, false).expect("must synthesize");
        assert!(matches!(
            verify_template_against_problem(&solver, &problem, &tpl, false),
            VerifyOutcome::Verified
        ));
    }

    /// DBM fast path Check 4: single-comparison guard + simple postcondition.
    /// `i := 0; while i < 6 { i := i + 1 }`, guard is a single `i ≤ 5`.
    /// ¬(i ≤ 5) = i ≥ 6, Post = i ≤ 10.
    /// DBM should handle it (the negation of a single-comparison guard
    /// stays within the difference-constraint fragment).
    #[test]
    fn test_dbm_check4_single_guard() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let mut problem = crate::hir::loop_ir::loop_instrs_to_loop_problem(
            &vars,
            &init,
            &body,
            &[8],
            &[false],
            &[],
            true,
        )
        .expect("must lower");
        problem.post = Some(Cond::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ScalarExpr::Var(0)),
            rhs: Box::new(ScalarExpr::Const(BigInt::from(10))),
            signed: false,
        });
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BigInt::from(6);
        // The DBM fast path should prove it (negated single-comparison
        // guard stays within the fragment).
        let result = dbm_proves_inductiveness(&problem, &tpl);
        assert_eq!(
            result,
            Some(true),
            "DBM must prove Check 4 for single-guard loop"
        );
    }

    /// Lower-bound candidates beyond `limits.lb` are pruned by
    /// BoundaryLimits before any solver query (Zuo et al. 2026, §5.2: the optimal
    /// lower bound satisfies `l* ≤ limits.lb`, so a candidate above it
    /// cannot be part of the BII).
    #[test]
    fn test_propose_bitwise_prunes_lower_bound_with_limits() {
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BiiRow::max_ub(8);

        let pos = BitPositions::new(&tpl);

        let mut limits = BoundaryLimits::new(&tpl);

        // Ad hoc: the optimal lower bound is at most 3.
        limits.lb[0] = BigInt::from(3);

        let cands = propose_bitwise(&tpl, &pos, &limits);

        // No lower-bound candidate may exceed 3.
        assert!(
            cands.iter().all(|c| c.rows[0].lb <= BigInt::from(3)),
            "BoundaryLimits must prune impossible lower-bound proposals"
        );
    }

    /// Upper-bound candidates below `limits.ub` are pruned by
    /// BoundaryLimits before any solver query (Zuo et al. 2026, §5.2: the optimal
    /// upper bound satisfies `u* ≥ limits.ub`, so a candidate below it
    /// cannot be part of the BII).
    #[test]
    fn test_propose_bitwise_prunes_upper_bound_with_limits() {
        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::zero();
        tpl.rows[0].ub = BiiRow::max_ub(8);

        let pos = BitPositions::new(&tpl);

        let mut limits = BoundaryLimits::new(&tpl);

        // Ad hoc: the optimal upper bound is at least 200.
        limits.ub[0] = BigInt::from(200);

        let cands = propose_bitwise(&tpl, &pos, &limits);

        // No upper-bound candidate may fall below 200.
        assert!(
            cands.iter().all(|c| c.rows[0].ub >= BigInt::from(200)),
            "BoundaryLimits must prune impossible upper-bound proposals"
        );
    }

    /// `TemplateLevel::Octagon` must NOT generate Support3 rows, but must
    /// still generate Diff rows.
    #[test]
    fn test_template_level_octagon_no_support3() {
        let tpl = BiiTemplate::with_level(
            3,
            &[8, 8, 8],
            &[false, false, false],
            TemplateLevel::Octagon,
        );
        let has_support3 = tpl
            .rows
            .iter()
            .any(|r| matches!(r.kind, RowKind::Support3(..)));
        assert!(
            !has_support3,
            "Octagon level must NOT generate Support3 rows"
        );

        let has_diff = tpl.rows.iter().any(|r| matches!(r.kind, RowKind::Diff(..)));
        assert!(has_diff, "Octagon level must generate Diff rows");
    }

    /// `TemplateLevel::Zone` generates Interval and Diff rows but NOT Sum or Support3 rows.
    #[test]
    fn test_template_level_zone() {
        let tpl = BiiTemplate::with_level(
            3,
            &[8, 8, 8],
            &[false, false, false],
            TemplateLevel::Zone,
        );
        // 3 Interval rows (one per variable)
        let interval_count = tpl
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Interval(_)))
            .count();
        assert_eq!(interval_count, 3, "Zone must generate Interval rows");
        // C(3,2) = 3 Diff rows
        let diff_count = tpl
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Diff(..)))
            .count();
        assert_eq!(diff_count, 3, "Zone must generate Diff rows");
        // No Sum rows
        let has_sum = tpl.rows.iter().any(|r| matches!(r.kind, RowKind::Sum(..)));
        assert!(!has_sum, "Zone must NOT generate Sum rows");
        // No Support3 rows
        let has_support3 = tpl
            .rows
            .iter()
            .any(|r| matches!(r.kind, RowKind::Support3(..)));
        assert!(!has_support3, "Zone must NOT generate Support3 rows");
        // Total row count: 3 + 3 = 6
        assert_eq!(tpl.rows.len(), 6);
    }

    /// Zone ⊂ Octagon ⊂ SparsePoly: each level is a strict superset of the previous.
    #[test]
    fn test_template_level_hierarchy() {
        let interval = BiiTemplate::with_level(3, &[8, 8, 8], &[false; 3], TemplateLevel::Interval);
        let zone = BiiTemplate::with_level(3, &[8, 8, 8], &[false; 3], TemplateLevel::Zone);
        let octagon = BiiTemplate::with_level(3, &[8, 8, 8], &[false; 3], TemplateLevel::Octagon);
        let sparse = BiiTemplate::with_level(3, &[8, 8, 8], &[false; 3], TemplateLevel::SparsePoly);

        assert!(interval.rows.len() < zone.rows.len());
        assert!(zone.rows.len() < octagon.rows.len());
        assert!(octagon.rows.len() < sparse.rows.len());

        // Zone = Interval + Diff
        let zone_diff_count = zone
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Diff(..)))
            .count();
        let zone_interval_count = zone
            .rows
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Interval(_)))
            .count();
        assert_eq!(zone.rows.len(), zone_interval_count + zone_diff_count);
    }

    /// Property Strengthening (Yao et al., "Demystifying...", §IV-A):
    /// the post-condition constraint
    /// `∀X. (A'(X) ∧ ¬G(X)) ⇒ Post(X)` is spliced into the refine query.
    #[test]
    fn test_property_strengthening_in_synthesis() {
        use crate::hir::loop_ir::{BiiLoopProblem, BiiVar, CmpOp, Cond, ScalarExpr};
        let mk = |s: &str| BiiVar {
            symbol: Symbol::intern(s),
            bw: 8,
            signed: false,
        };
        let problem = BiiLoopProblem {
            vars: vec![mk("i")],
            params: vec![],
            init: vec![ScalarExpr::Const(BigInt::zero())],
            loop_guard: Cond::True,
            back_edges: vec![],
            exit_edges: vec![],
            saturates: vec![],
            post: Some(Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(10))),
                signed: false,
            }),
        };
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        let candidates = vec![tpl];
        // BV mode: the post bound renders as `(_ bv10 8)`.
        let q = build_refine_query_problem(&problem, &candidates, false, true, &[8]).unwrap();

        // The query must contain the negated guard and the post bound.
        // The post constant lifts to `bits(10)+1 = 5` bits (Route B
        // width rule for constants).
        assert!(q.contains("(not true)"), "Query must contain negated guard");
        assert!(
            q.contains("(_ bv10 5)"),
            "Query must contain post-condition bound"
        );
    }

    /// Saturate DBM expansion: the DBM path can prove a loop with a
    /// saturating assignment (`x' = clamp(x + 10)`) directly, without
    /// falling back to SMT.
    #[test]
    fn test_dbm_saturate_expansion() {
        use crate::hir::loop_ir::{
            ArithSem, BiiLoopProblem, BiiVar, Cond, EdgeKind, ScalarExpr, TransitionEdge,
        };
        let mk = |s: &str| BiiVar {
            symbol: Symbol::intern(s),
            bw: 8,
            signed: false,
        };

        // A saturating assignment: x' = clamp(x + 10).
        let next_val = ScalarExpr::Add(
            Box::new(ScalarExpr::Var(0)),
            Box::new(ScalarExpr::Const(BigInt::from(10))),
            ArithSem::Saturate,
        );

        let problem = BiiLoopProblem {
            vars: vec![mk("x")],
            params: vec![],
            init: vec![ScalarExpr::Const(BigInt::from(200))],
            loop_guard: Cond::True,
            back_edges: vec![TransitionEdge {
                kind: EdgeKind::Back,
                guard: None,
                definedness: None,
                next_values: vec![next_val],
            }],
            exit_edges: vec![],
            saturates: vec![(0, 10)],
            post: None,
        };

        let mut tpl = BiiTemplate::new(1, &[8], &[false]);
        tpl.rows[0].lb = BigInt::from(200);
        tpl.rows[0].ub = BigInt::from(255); // Saturate ceiling

        // If the Saturate expansion succeeds, the DBM path returns
        // Some(true) instead of None (SMT fallback).
        assert_eq!(
            dbm_proves_inductiveness(&problem, &tpl),
            Some(true),
            "DBM must prove Saturate loops via path expansion without SMT fallback"
        );
    }

    /// Adaptive leap cooldown: the BoundaryLimits state machine stays
    /// healthy across UNSAT pruning and SAT tightening.
    #[test]
    fn test_adaptive_leap_cooldown_logic() {
        let tpl = BiiTemplate::new(1, &[8], &[false]);
        let mut limits = BoundaryLimits::new(&tpl);

        // Simulate UNSAT pruning: a candidate tightening lb to 10 and
        // ub to 20 moves the limits to lb=9 / ub=21 (one past the
        // proposal).
        let cands = vec![tpl.with_row(0, BigInt::from(10), BigInt::from(20))];
        limits.prune_unsat(&tpl, &cands);
        assert!(limits.is_active(), "Limits should be active after pruning");
        assert_eq!(
            limits.lb[0],
            BigInt::from(9),
            "prune_unsat must tighten lb to lb-1"
        );
        assert_eq!(
            limits.ub[0],
            BigInt::from(21),
            "prune_unsat must widen ub to ub+1"
        );

        // SAT tightening only strengthens: `l* ≤ 15` does not move lb
        // (already 9 < 15), and `u* ≥ 5` does not move ub (already
        // 21 > 5) — the limits stay as pruned.
        let bounds = vec![(BigInt::from(5), BigInt::from(15))];
        limits.tighten_sat(&bounds);
        assert_eq!(
            limits.lb[0],
            BigInt::from(9),
            "SAT tightening must only strengthen, never weaken"
        );
    }
}
