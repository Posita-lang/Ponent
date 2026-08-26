//! The octagon abstract domain (difference-bound matrices) — extracted
//! from  `type_eq.rs`  to keep the equality-checking module focused.
//!
//! A difference-bound matrix over  2n nodes  —  `Xᵢ⁺ = 2i`  (the variable)
//! and  `Xᵢ⁻ = 2i+1`  (its negation) — with  strong closure  ([Miné06]
//! Figure 8), meet/join/widen, and loop-transfer functions. There is NO
//! implicit zero node: single-variable bounds live on the  self-dual
//! edges `m[2i][2i+1]`  ( `2Xᵢ ≤ 2c` , i.e.  `Xᵢ ≤ c` ) and  `m[2i+1][2i]`
//! ( `−2Xᵢ ≤ −2c` , i.e.  `Xᵢ ≥ c` ); the mirror of node  `i`  is  `i ⊕ 1` , and
//! the mirror of edge  `(i, j)`  is  `(j⊕1, i⊕1)` .
//!
//! All stored bounds are  doubled  (2×c) to represent half-integers
//! exactly. Interval (self-dual) rows therefore carry 4× the interval
//! half-width in raw storage — readers must go through the semantic
//! pr ojections ( `var_ub`  /  `var_lb`  /  `diff_bound`  /  `sum_ub` ), never
//! raw cells. External API accepts plain  `c` , internal storage uses the
//! doubled space.
//!
//!  Invariant (closed matrices) :  `i128::MIN`  is the −∞ marker in the
//! saturating arithmetic ( `sat_sub` / `sat_neg` ), so a closed (non-bottom)
//! matrix never stores it as an edge bound. Path sums that would
//! saturate to  `i128::MIN`  (exact value ≤ −2^127, e.g. two −2^126 edges
//! composing) are UNREPRESENTABLE: the closure CLAMPS them to  `MIN + 4`
//! ( `sat_add_closed`  — a sound, monotone weakening, so the no-close
//! preservation lemmas keep their proofs, and the clamped value is
//! still negative, so an extreme negative cycle still reports b ottom),
//!  `sat_mul2_bound`  refuses the marker on the insertion side,
//!  `tighten_self_dual`  clamps its rounding off it, and  `close_with`
//! weakens any raw  `i128::MIN`  cell to  `DBM_INF`  as belt-and-braces.
//! Reading a cell is safe either way ( `node_bound`  would treat it as
//! the finite bound −2^126, a sound over-approximation of the true
//! −2^127), but the STORED value must have a single meaning.
//!
//!  Closed-state convention : operations that require strongly closed
//! operands ( `join` ) document it, and the debug tripwire fires on
//! violations. The one legitimate producer of non-closed states is
//!  `widen`  ([Miné06] §VI.D — closing its output would risk the
//! non-terminating chain of Figure 10 / Thm 8.2); re-close a widened
//! state before joining it. Every other transfer preserv es closure.
//!
//! # References
//!
//! Citations in this module use the following shorthand:
//!
//! -  `[Miné06]`  — Antoine Miné,  "The Octagon Abstract Domain ",
//!   Higher-Order and Symbolic Computation 19(1), 2006. The primary
//!   source for the 2n-node DBM representation, strong closure
//!   (Figure 8), coherence, join/widen, and  the loop-transfer
//!   functions. ( "The octagon paper " below always means this one.)
//! -  `[Miné01]`  — Antoine Miné,  "A New Numerical Abstract Domain Based
//!   on Difference-Bound Matrices ", PADO II, LNCS 2053, 2001. The
//!   earlier DBM-based domain that  `[Miné06]`  extends; cited here for
//!   historical context, not for a specific algorithm used in this
//!   module.
//! -  `[HS97]`  — Warwick Harvey and Peter J. Stuckey,  "A Unit Two
//!   Variable Per Inequality Integer Constraint Solver for Constraint
//!   Logic Programming ", ACSC 1997. Source of the integer-tightening
//!   step discussed in  `[Miné06]`  §V.D.
//! -  `[BHZ07]`  — Roberto Bagnara, Patricia M. Hill, and Enea
//!   Zaffanella,  "An Improved Tight Closure Algorithm for Integer
//!   Octagonal Constraints ", arXiv:0705.4618 [cs.DS], 2007. Source of
//!   the O(n³) tight-closure schedule (Theorems 2–3, Figure 2) used
//!   by  `ClosureMode::IntegerExact` : on a closed integer octagonal
//!   graph, one tightening pass plus a single floor-halved
//!   strong-coherence step computes the tight closure exactly, and an
//!   O(n) self-d ual-pair check decides Z-consistency.
//! -  `[EAGLE]`  — Peisen Yao et al., "Demystifying Template-based
//!   Invariant Generation for Bit-Vector Programs", 2024. Context for
//!   octagon templates in template-based invariant generation and the
//!   role of property strengthening / disjunctive completion.

/// `∞` marker — the largest representable bound (`i128::MAX / 4` keeps
/// saturated additions from colliding with true `i128::MAX`).
pub(crate) const DBM_INF: i128 = i128::MAX / 4;

/// Saturated `a + b` (doubled-space arithmetic).
pub(crate) fn sat_add(a: i128, b: i128) -> i128 {
    if a == DBM_INF || b == DBM_INF {
        return DBM_INF;
    }
    match a.checked_add(b) {
        Some(s) if s < DBM_INF => s,
        None if a < 0 && b < 0 => i128::MIN,
        _ => DBM_INF,
    }
}

/// Saturated  `a + b`  for the CLOSURE passes: results are CLAMPED to the
/// interval  `[i128::MIN + 4, DBM_INF]`  — any exact total ≤ −2^127 + 3 is
/// raised to  `MIN + 4`  (the −∞ marker  `i128::MIN`  itself is never a
/// legal stored value).  `MIN + 4`  is a sound weakening of any exact
/// total in that range (it is ≥ the exact value), stays finite, and is
/// a multiple of 4 (so the  `IntegerExact`  rounding cannot push it onto
/// the marker). Crucially the clamped value is still NEGATIVE, so a
/// negative cycle — however extreme — still drives a diagonal below 0
/// and re ports bottom. The interval clamp is MONOTONE in both arguments
/// (unlike a point clamp, which jumps at exactly  `i128::MIN`  and breaks
/// the lattice), so the closure operator built from it is monotone and
/// the no-close preservation lemmas ( `join` ,  `assign_add_var` ) keep
/// their proofs; it also means  `is_strongly_closed` 's triangle check
/// implies the longer C⁺ arms, and the  `IntegerExact`  tight-closure
/// round needs no iteration to converge (the clamped value is already
/// ≡ 0 (mod 4), so the tightening leaves it untouched).
fn sat_add_closed(a: i128, b: i128) -> i128 {
    sat_add(a, b).max(i128::MIN + 4)
}

/// Saturated `a - b`.
pub(crate) fn sat_sub(a: i128, b: i128) -> i128 {
    if a == DBM_INF && b == DBM_INF {
        return DBM_INF; // ∞ - ∞ conservatively treated as ∞
    }
    if a == DBM_INF {
        return DBM_INF; // ∞ - finite = ∞
    }
    if b == DBM_INF {
        return i128::MIN; // finite - ∞ = -∞
    }
    if b == i128::MIN {
        return DBM_INF; // finite - (-∞) = ∞
    }
    match a.checked_sub(b) {
        Some(s) if s < DBM_INF => s,
        None if a < 0 && b > 0 => i128::MIN,
        _ => DBM_INF,
    }
}

/// Saturated negation.
pub(crate) fn sat_neg(a: i128) -> i128 {
    if a == DBM_INF {
        return i128::MIN; // -(+∞) = -∞
    }
    if a == i128::MIN {
        return DBM_INF; // -(-∞) = +∞
    }
    match a.checked_neg() {
        Some(s) if s < DBM_INF => s,
        _ => DBM_INF,
    }
}

/// Saturating multiplication by 2 (used when converting a constraint constant  `c`  to a stored value  `2c` )
/// - If  `c == DBM_INF` , returns  `DBM_INF`
/// - If  `c`  is positive and overflows, returns  `DBM_INF`
/// - If  `c`  is negative and overflows, returns  `i128::MIN`  (representing negative infinity)
/// - Otherwise returns  `2*c`
pub(crate) fn sat_mul2(c: i128) -> i128 {
    if c == DBM_INF {
        return DBM_INF; // i128::MAX / 4
    }
    match c.checked_mul(2) {
        Some(v) => {
            if v < DBM_INF {
                v
            } else {
                DBM_INF
            }
        }
        None => {
            // Overflow
            if c > 0 { DBM_INF } else { i128::MIN }
        }
    }
}

/// Saturating multiplication by 2 for  bound insertion  ( `set`  /
///  `set_mirrored` ). Unlike  `sat_mul2` , negative overflow weakens to
///  `DBM_INF`  (no bound) instead of  `i128::MIN` :  `i128::MIN`  is reserved
/// as the negative-infinity marker in  `sat_sub` / `sat_neg` , so storing it
/// as an edge bound would let the same cell value mean a finite bound in
///  `node_bound`  and  `-∞`  in  `sat_sub`  — an unsound collision. Weakening
/// to no bound is the sound over-approximation.
fn sat_mul2_bound(c: i128) -> i128 {
    if c == DBM_INF {
        return DBM_INF;
    }
    match c.checked_mul(2) {
        Some(v) if v < DBM_INF && v > i128::MIN => v,
        _ => DBM_INF, // out of finite range: weaken to no bound
    }
}

/// Saturating multiplication `a * b` for interval arithmetic in the
/// expression evaluator. Returns `(lo, hi)` with sound over-approximation.
/// Used by `OctExpr::eval_interval` for scalar-multiply nodes.
fn sat_mul(a: i128, b: i128) -> i128 {
    if a == DBM_INF || b == DBM_INF {
        return DBM_INF;
    }
    if a == i128::MIN || b == i128::MIN {
        // Treat MIN as -∞ sentinel; product with finite is ±∞
        if a == i128::MIN && b == i128::MIN {
            return DBM_INF; // (-∞)*(-∞) = +∞
        }
        if (a == i128::MIN && b > 0) || (b == i128::MIN && a > 0) {
            return i128::MIN; // (-∞)*(+finite) = -∞
        }
        if (a == i128::MIN && b < 0) || (b == i128::MIN && a < 0) {
            return DBM_INF; // (-∞)*(-finite) = +∞
        }
        return i128::MIN;
    }
    match a.checked_mul(b) {
        Some(v) if v < DBM_INF && v > i128::MIN => v,
        Some(v) if v >= DBM_INF => DBM_INF,
        Some(_) => i128::MIN,
        None => {
            // Overflow: determine sign
            if (a > 0 && b > 0) || (a < 0 && b < 0) {
                DBM_INF
            } else {
                i128::MIN
            }
        }
    }
}

/// Compute the mirror index for coherence: node `i` mirrors to `i ⊕ 1`
/// (`Xᵢ⁺ = 2i ↔ Xᵢ⁻ = 2i+1`).
#[inline]
const fn mirror_index(i: usize) -> usize {
    i ^ 1
}

/// A difference-bound matrix over  `2n`  nodes:  `Xᵢ⁺ = Xᵢ`  and
///  `Xᵢ⁻ = -Xᵢ` .  `m[i][j]`  encodes  `node_i - node_j ≤ c` in doubled
/// space  ( `m[i][j] = 2*c` ). ([Miné06]'s convention is the transpose —
///  `m⁺ᵢⱼ`  bounds  `vⱼ − vᵢ` ; Figure 8's operators are unchanged
/// under transposition.) Single-variable bounds hang on the
/// self-dual edges:  `m[2i][2i+1]`  carries  `2Xᵢ ≤ 2c`  (stored  `4c` ),
///  `m[2i+1][2i]`  carries  `-2Xᵢ ≤ -2c`  (stored  `-4c` ).
#[derive(Clone, Debug)]
pub(crate) struct Dbm {
    ///  `2 * n_vars`  (no implicit zero node).
    pub(crate) size: usize,
    /// Flattened  `size × size`  matrix, row-major, stored in doubled space.
    pub(crate) m: Vec<i128>,
    ///  `true`  if the constraint system is unsatisfiable (⊥).
    pub(crate) bottom: bool,
    ///  `true`  if the matrix is known strongly closed (or ⊥, which is
    /// vacuously closed — see `is_strongly_closed`). Maintained by every
    /// mutation (`set`/`set_mirrored` clear it; `close_with` sets it);
    /// lets `join`'s debug tripwire be O(1) instead of an O(N³)
    /// `is_strongly_closed` check.
    pub(crate) closed: bool,
}

/// Equality compares the constraint set only — the `closed` flag is
/// metadata (derivable from `m`), so two matrices with identical values
/// but different closure bookkeeping compare equal.
impl PartialEq for Dbm {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.m == other.m && self.bottom == other.bottom
    }
}

/// Closure strategy — the Harvey–Stuckey integer-tightening profile
/// ([HS97]; see [Miné06] §V.D).
///
///  `Strong`  is [Miné06]'s Figure 8 strong closure — sound over reals,
/// rationals AND integers (an over-approximation for the latter).
///
///  `IntegerExact`  additionally computes the Bagnara–Hill–Zaffanella
/// TIGHT closure ([BHZ07], Figure 2 / Theorems 2–3): after the
/// strong closure it applies ONE Harvey–Stuckey tightening pass
/// ([Miné06] §V.D:  `2x ≤ 2c+1 ⟹ 2x ≤ 2c`  — knowing x is an integer,
/// the self-dual edge constants round down), an O(n) Z-consistency
/// check on each self-dual pair, and a single strong-coherence (S⁺)
/// pass. Total  cost O(N³) — the same order as  `Strong` , with NO
/// fixpoint iteration and NO round cap: [BHZ07] Thm 2 proves one
/// round on a closed graph yields the tight closure exactly, and
/// Thm 3 proves the self-dual-pair ch eck decides integer
/// (in)consistency, so a failed check reports ⊥ soundly and
/// completely. (This replaces a former tighten/re-close fixpoint
/// loop whose worst-case round c ount was exponential in the bit
/// width and capped at an engineering bound of 4N+64.)
/// SOUND ONLY over INTEGER domains: over rationals the rounding
/// discards solutions ( `2x ≤ 5`  admits x = 2.5; the tightened
///  `2x ≤ 4`  does not). Integer-COMPLETENESS holds only at that
/// fixpoint and only while the matrix has not since passed through a
/// Strong-close operation ( `meet` , the  `test_*`  guards,
///  `assign_const_var` ,  `assign_copy_var` ,  `forget_var`  all close with
///  `Strong`  internally and can leave self-dual edges non-≡0 (mod 4) —
/// see  `close` 's docs). Wired at the exact-discharge surface that
/// reasons over integers by construction: the BII inductiveness
/// verifier ( `dbm_proves_inductiveness`  in  `bii.rs` , LIA mode only).
/// The DBM fixpoint and the type-equality closure stay on  `Strong` .
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ClosureMode {
    Strong,
    IntegerExact,
}

// ─────────────────────────────────────────────────────────────────────
// Expression type for the interval-arithmetic fallback assignment
// ─────────────────────────────────────────────────────────────────────

/// A simple arithmetic expression over program variables, used by the
/// projection + interval-arithmetic fallback assignment ([Miné06] §VI.E,
/// the "general case" after Definition 2.6).
///
/// The octagon domain can only represent constraints of the form
/// `(±x ± y ≤ c)`, so a general assignment `X := e` cannot be encoded
/// exactly. The strategy from [Miné06] is:
///
/// 1. Extract the interval `[lb_i, ub_i]` of each variable via
///    projection (Theorem 6):
///      `lb_i = −(m+)•_{2i+1, 2i} / 2`,  `ub_i = (m+)•_{2i, 2i+1} / 2`
///    (In our doubled storage: `var_lb(i)` and `var_ub(i)`.)
///
/// 2. Evaluate `e` using interval arithmetic to obtain `[e_lo, e_hi]`
///    ⊇ { e(s) | s ∈ D⁺(m⁺) }.
///
/// 3. Produce the abstract post-state:
///    ```text
///    [m⁺(vₖ ← e)]ᵢⱼ =
///      (m⁺)•ᵢⱼ           if i, j ∉ {2k, 2k+1}
///      e_hi              if (i, j) = (2k, 2k+1)   [X ≤ e_hi]
///      −e_lo             if (i, j) = (2k+1, 2k)   [X ≥ e_lo]
///      +∞                elsewhere (X's other edges)
///    ```
///    (Definition 2.6 + the interval refinement described in §VI.E.)
///
/// This is a sound over-approximation: the result contains all
/// concrete post-states but may introduce spurious points.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OctExpr {
    /// A numeric constant.
    Const(i128),
    /// Reference to program variable `i` (0-based index).
    Var(usize),
    /// `lhs + rhs`
    Add(Box<OctExpr>, Box<OctExpr>),
    /// `lhs - rhs`
    Sub(Box<OctExpr>, Box<OctExpr>),
    /// Unary negation: `-inner`
    Neg(Box<OctExpr>),
    /// Scalar multiplication: `k * inner` where `k` is a constant.
    /// The octagon domain is linear, so only constant-coefficient
    /// scaling is supported; variable × variable would leave the domain.
    Scale(i128, Box<OctExpr>),
}

impl OctExpr {
    /// Convenience constructors.
    pub(crate) fn var(i: usize) -> Self {
        OctExpr::Var(i)
    }
    pub(crate) fn konst(c: i128) -> Self {
        OctExpr::Const(c)
    }
    pub(crate) fn add(self, rhs: OctExpr) -> Self {
        match (self, rhs) {
            (OctExpr::Const(a), OctExpr::Const(b)) => OctExpr::Const(sat_add(a, b)),
            (a, b) => OctExpr::Add(Box::new(a), Box::new(b)),
        }
    }
    pub(crate) fn sub(self, rhs: OctExpr) -> Self {
        match (self, rhs) {
            (OctExpr::Const(a), OctExpr::Const(b)) => OctExpr::Const(sat_sub(a, b)),
            (a, b) => OctExpr::Sub(Box::new(a), Box::new(b)),
        }
    }
    pub(crate) fn neg(self) -> Self {
        match self {
            OctExpr::Const(a) => OctExpr::Const(sat_neg(a)),
            a => OctExpr::Neg(Box::new(a)),
        }
    }
    pub(crate) fn scale(k: i128, inner: OctExpr) -> Self {
        if k == 0 {
            return OctExpr::Const(0); // 0 · e = 0 — fold before building the node
        }
        match inner {
            OctExpr::Const(a) => OctExpr::Const(sat_mul(k, a)),
            inner => OctExpr::Scale(k, Box::new(inner)),
        }
    }

    /// Evaluate this expression over the intervals extracted from `dbm`
    /// using standard interval arithmetic.
    ///
    /// Returns `(lo, hi)` where `lo` is the lower bound and `hi` is the
    /// upper bound of the expression's range. Either may be `DBM_INF`
    /// (unbounded) or `i128::MIN` (−∞ sentinel).
    ///
    /// # Interval arithmetic rules (sound over-approximation)
    ///
    /// - `[a,b] + [c,d] = [a+c, b+d]`
    /// - `[a,b] − [c,d] = [a−d, b−c]`
    /// - `−[a,b] = [−b, −a]`
    /// - `k·[a,b]` = `[k·a, k·b]` if k ≥ 0; `[k·b, k·a]` if k < 0
    ///
    /// All operations use saturating arithmetic to avoid UB and
    /// maintain soundness at the extremes.
    pub(crate) fn eval_interval(&self, dbm: &Dbm) -> (i128, i128) {
        match self {
            OctExpr::Const(c) => (*c, *c),

            OctExpr::Var(i) => {
                // Projection: Theorem 6 of [Miné06].
                // var_lb / var_lb return None for ∞ (unbounded).
                let lo = dbm.var_lb(*i).unwrap_or(i128::MIN);
                let hi = dbm.var_ub(*i).unwrap_or(DBM_INF);
                (lo, hi)
            }

            OctExpr::Add(lhs, rhs) => {
                let (a, b) = lhs.eval_interval(dbm);
                let (c, d) = rhs.eval_interval(dbm);
                // [a,b] + [c,d] = [a+c, b+d]
                (sat_add(a, c), sat_add(b, d))
            }

            OctExpr::Sub(lhs, rhs) => {
                let (a, b) = lhs.eval_interval(dbm);
                let (c, d) = rhs.eval_interval(dbm);
                // [a,b] − [c,d] = [a−d, b−c]
                (sat_sub(a, d), sat_sub(b, c))
            }

            OctExpr::Neg(inner) => {
                let (a, b) = inner.eval_interval(dbm);
                // −[a,b] = [−b, −a]
                (sat_neg(b), sat_neg(a))
            }

            // 0 · [a,b] = [0, 0] — even for an unbounded interval
            // (sat_mul(0, ∞) would return ∞, losing precision).
            OctExpr::Scale(0, _) => (0, 0),

            OctExpr::Scale(k, inner) => {
                let (a, b) = inner.eval_interval(dbm);
                if *k >= 0 {
                    // k·[a,b] = [k·a, k·b]
                    (sat_mul(*k, a), sat_mul(*k, b))
                } else {
                    // k·[a,b] = [k·b, k·a] (reverses order)
                    (sat_mul(*k, b), sat_mul(*k, a))
                }
            }
        }
    }

    /// Collect all variable indices referenced by this expression.
    /// Used to determine which variables' intervals to project.
    pub(crate) fn free_vars(&self, out: &mut Vec<usize>) {
        match self {
            OctExpr::Const(_) => {}
            OctExpr::Var(i) => {
                if !out.contains(i) {
                    out.push(*i);
                }
            }
            OctExpr::Add(l, r) | OctExpr::Sub(l, r) => {
                l.free_vars(out);
                r.free_vars(out);
            }
            OctExpr::Neg(inner) => inner.free_vars(out),
            OctExpr::Scale(_, inner) => inner.free_vars(out),
        }
    }
}

impl Dbm {
    /// Create a new, unconstrained (top) matrix.
    pub(crate) fn new(n_vars: usize) -> Dbm {
        let size = n_vars.checked_mul(2).expect("too many variables");
        let len = size.checked_mul(size).expect("matrix too large");
        let mut m = vec![DBM_INF; len];
        for i in 0..size {
            m[i * size + i] = 0;
        }
        Dbm {
            size,
            m,
            bottom: false,
            closed: true,
        }
    }

    /// The empty (bottom) matrix.
    pub(crate) fn bottom() -> Dbm {
        Dbm {
            size: 0,
            m: Vec::new(),
            bottom: true,
            closed: true, // ⊥ is vacuously strongly closed
        }
    }

    /// Flattened index of `m[i][j]` (row-major).
    #[inline]
    fn ix(&self, i: usize, j: usize) -> usize {
        i * self.size + j
    }

    /// Assert `i`/`j` are valid NODE indices — the write-side counterpart
    /// of `node_bound`'s total read: a bare index-out-of-bounds panic
    /// here would be a caller bug with an unhelpful message.
    #[inline]
    fn check_node(&self, i: usize, j: usize) {
        assert!(
            i < self.size && j < self.size,
            "octagon DBM node index ({i}, {j}) out of range [0, {})",
            self.size
        );
    }

    /// Assert variable index `i` is in range (write-side counterpart).
    /// Written as `i < self.size / 2` (NOT `2 * i + 1 < self.size`):
    /// the latter overflows and wraps on 64-bit for `i ≥ 2^63`,
    /// silently passing the check and corrupting the caller's variable.
    #[inline]
    fn check_var(&self, i: usize) {
        assert!(
            i < self.size / 2,
            "octagon DBM variable index {i} out of range [0, {})",
            self.size / 2
        );
    }

    /// Tighten `node_i - node_j ≤ c` **without** mirroring (internal use only).
    /// `c` is the **actual** bound; stored as `2*c`.
    pub(crate) fn set(&mut self, i: usize, j: usize, c: i128) {
        if self.bottom {
            return;
        }
        self.check_node(i, j);
        let stored = sat_mul2_bound(c);
        let idx = self.ix(i, j);
        if stored < self.m[idx] {
            self.m[idx] = stored;
            self.closed = false; // one-sided set breaks closure
        }
    }

    /// Set a bound and its mirror to maintain coherence.
    /// For constraint `node_i - node_j ≤ c`, also set `node_j̄ - node_ī ≤ c`.
    pub(crate) fn set_mirrored(&mut self, i: usize, j: usize, c: i128) {
        if self.bottom {
            return;
        }
        self.check_node(i, j);
        Self::set_mirrored_internal(&mut self.m, i, j, c, self.size);
        self.closed = false; // a new bound breaks closure
    }

    /// Internal helper: set a bound and its mirror in a raw matrix slice.
    fn set_mirrored_internal(m: &mut [i128], i: usize, j: usize, c: i128, size: usize) {
        let stored = sat_mul2_bound(c);
        let idx1 = i * size + j;
        if stored < m[idx1] {
            m[idx1] = stored;
        }
        let i_bar = mirror_index(i);
        let j_bar = mirror_index(j);
        let idx2 = j_bar * size + i_bar;
        if stored < m[idx2] {
            m[idx2] = stored;
        }
    }

    /// Read the ACTUAL bound `c` of `node_i − node_j ≤ c`. The internal
    /// storage is doubled (`2c`); this is the single halving point for
    /// readers — tests and the invariant-expression emitter share it, so
    /// the convention has one source of truth.
    ///
    /// Floor semantics: all DBM consumers in this pipeline reason over
    /// INTEGER loop variables, where `X ≤ s/2` with `s` odd floors to the
    /// exact integer bound. A rational-domain consumer would need the raw
    /// stored value (future work, note before reuse).
    /// `None` = ∞ (also ⊥ / out-of-range — readers get a total function,
    /// the "index out of bounds: len is 0" panic class dies here).
    pub(crate) fn node_bound(&self, i: usize, j: usize) -> Option<i128> {
        if self.bottom || i >= self.size || j >= self.size {
            return None;
        }
        match self.m[i * self.size + j] {
            DBM_INF => None,
            v => Some(v >> 1), // floor — sound read direction for ≤
        }
    }

    /// Strong closure ([Miné06] Figure 8) preserving coherence and deriving
    /// octagonal constraints. Returns `false` if the system is
    /// unsatisfiable (negative diagonal), setting `bottom = true`.
    ///
    /// NOTE: `Strong` closure EXITS the Harvey–Stuckey rounded normal
    /// form ([HS97]; see [Miné06] §V.D): the C⁺/S⁺ passes can derive a
    /// self-dual edge that is not ≡0 (mod 4) from non-4-multiple path
    /// sums (e.g. `2X ≤ 5`, stored 10). The IntegerExact invariant
    /// "every finite self-dual stored bound ≡ 0 (mod 4)" therefore holds
    /// only while the matrix stays under
    /// `close_with(ClosureMode::IntegerExact)`. Operations that call
    /// `close()` internally — `meet`, `test_le_var`, `test_ge_var`,
    /// `test_diff_le`, `assign_const_var`, `assign_copy_var`,
    /// `forget_var` — exit the normal form; re-close with `IntegerExact`
    /// to restore it.
    pub(crate) fn close(&mut self) -> bool {
        self.close_with(ClosureMode::Strong)
    }

    /// One pass of the strong closure, adapted from [Miné06] Figure 8
    /// (all-pivot variant — see the body comment on even-pivot
    /// equivalence): the C⁺_k / S⁺ interleaving over all k, in place.
    /// Extracted from `close_with` so it can be re-run after the
    /// coherence repair; its S⁺ tail lives in `strong_coherence_pass`,
    /// which the BHZ tight-closure round ([BHZ07]) also invokes
    /// standalone after tightening.
    fn strong_closure_pass(m: &mut [i128], size: usize) {
        // [Miné06]'s loop applies C⁺ only at even pivots (its Figure 8:
        // S⁺(C⁺_{2k}(·))); C⁺_{k̄} has the same five terms as C⁺_k, so
        // iterating over all 2n nodes applies each pivot twice —
        // redundant work, same result.
        // k iterates over ALL 2n nodes. The mirror of every node is
        // `i ⊕ 1`; with no implicit zero node the single-variable bounds
        // [Miné06] Figure 8 applies C⁺ only at EVEN pivots:
        // `m⁺_{k+1} = S⁺(C⁺_{2k}(m⁺_k))` for `k = 0..N-1`. The odd
        // pivot `k̄ = k ⊕ 1` is redundant: its five C⁺ terms are the
        // mirror of pivot `k`'s (the mirror map `i ↦ i⊕1` sends each
        // of the five arms to another arm of the mirrored entry), so
        // on a coherent matrix it re-derives the same bounds. Iterating
        // only the even pivots halves the O(N³) closure work.
        for k in (0..size).step_by(2) {
            let k_bar = mirror_index(k);
            // ---- C⁺_k ([Miné06] Figure 8). Each term is a path
            // i → … → j in the potential graph, so every arm of
            // the min is a sound bound on node_i − node_j:
            //   1. m[i][j]                         (direct)
            //   2. m[i][k]  + m[k][j]              (i→k→j)
            //   3. m[i][k̄] + m[k̄][j]              (i→k̄→j)
            //   4. m[i][k]  + m[k][k̄] + m[k̄][j]   (i→k→k̄→j)
            //   5. m[i][k̄] + m[k̄][k] + m[k][j]    (i→k̄→k→j)
            // The mirror of every term is again one of the five
            // terms of the mirrored entry (j̄, ī), so C⁺_k maps
            // coherent matrices to coherent matrices — the
            // property [Miné06] Figure 8's C⁺ is designed around
            // ([Miné06] §V.C).
            // Row offsets are hoisted out of the inner loops (pure
            // algebraic refactor — same indices, no semantic change).
            let row_k = k * size;
            let row_kb = k_bar * size;
            for i in 0..size {
                let row_i = i * size;
                for j in 0..size {
                    let mut best = m[row_i + j];
                    let t_ik = m[row_i + k];
                    let t_kj = m[row_k + j];
                    if t_ik != DBM_INF && t_kj != DBM_INF {
                        let s = sat_add_closed(t_ik, t_kj);
                        if s < best {
                            best = s;
                        }
                    }
                    let t_ikb = m[row_i + k_bar];
                    let t_kbj = m[row_kb + j];
                    if t_ikb != DBM_INF && t_kbj != DBM_INF {
                        let s = sat_add_closed(t_ikb, t_kbj);
                        if s < best {
                            best = s;
                        }
                    }
                    let t_kkb = m[row_k + k_bar];
                    if t_ik != DBM_INF && t_kkb != DBM_INF && t_kbj != DBM_INF {
                        let s = sat_add_closed(sat_add_closed(t_ik, t_kkb), t_kbj);
                        if s < best {
                            best = s;
                        }
                    }
                    let t_kbk = m[row_kb + k];
                    if t_ikb != DBM_INF && t_kbk != DBM_INF && t_kj != DBM_INF {
                        let s = sat_add_closed(sat_add_closed(t_ikb, t_kbk), t_kj);
                        if s < best {
                            best = s;
                        }
                    }
                    m[row_i + j] = best;
                }
            }
            // One S⁺ pass after each pivot's C⁺ (the interleaving of
            // [Miné06] Figure 8); the S⁺ body lives in
            // `strong_coherence_pass` so the BHZ tight-closure round
            // can invoke it standalone after tightening.
            Self::strong_coherence_pass(m, size);
        }
    }

    /// S⁺ ([Miné06] Figure 8): m[i][j] ≤ (m[i][ī] + m[j̄][j]) / 2,
    /// applied in place. Extracted from `strong_closure_pass` so the
    /// BHZ tight-closure round ([BHZ07], Figure 2, "Strong coherence")
    /// can run it standalone after tightening: on mod-4-clean operands
    /// (every finite self-dual stored bound ≡ 0 (mod 4)) the ceil
    /// halving is exact and coincides with [BHZ07]'s floor-halved sum
    /// ⌊w(i,ı)/2⌋ + ⌊w(j̄,j)/2⌋ in doubled storage (BHZ weight w = s/2).
    ///
    /// In doubled storage the derived value is (t1+t2)/2 — e.g.
    /// 2v₀ ≤ 1 ∧ 2v₁ ≤ 2 derives v₀+v₁ ≤ 1.5, stored 3. Odd sums
    /// (two half-integer bounds compose into a quarter) round UP —
    /// the sound direction for ≤. DBM_INF is never halved (DBM_INF/2
    /// is a finite value and would invent a tightening); sums are
    /// clamped off `i128::MIN` by `sat_add_closed`; sum < DBM_INF
    /// guarantees sum+1 cannot overflow.
    fn strong_coherence_pass(m: &mut [i128], size: usize) {
        for i in 0..size {
            let i_bar = mirror_index(i);
            for j in 0..size {
                let j_bar = mirror_index(j);
                let t1 = m[i * size + i_bar];
                let t2 = m[j_bar * size + j];
                if t1 != DBM_INF && t2 != DBM_INF {
                    let sum = sat_add_closed(t1, t2);
                    if sum != DBM_INF {
                        let halved = (sum + 1) >> 1; // = ceil(sum / 2)
                        let idx = i * size + j;
                        if halved < m[idx] {
                            m[idx] = halved;
                        }
                    }
                }
            }
        }
    }

    /// Harvey–Stuckey tightening ([HS97]; see [Miné06] §V.D): round every
    /// self-dual edge's stored bound down to a multiple of 4 — the
    /// edge (2v, 2v+1) carries `2x ≤ s/2` with 2x an EVEN integer, so
    /// `2x ≤ s/2` ⟺ `2x ≤ 4⌊s/4⌋` over Z ([Miné06]'s undoubled rule
    /// `2x ≤ 2c+1 ⟹ 2x ≤ 2c`). Self-dual edges are their own mirrors,
    /// so coherence is preserved. Returns whether anything changed.
    /// `DBM_INF` is never rounded (it is not a bound).
    fn tighten_self_dual(m: &mut [i128], size: usize) -> bool {
        let mut changed = false;
        for v in 0..size / 2 {
            for e in [2 * v, 2 * v + 1] {
                let idx = e * size + (e ^ 1);
                let s = m[idx];
                if s != DBM_INF && (s & 3) != 0 {
                    // Floor to a multiple of 4, CLAMPED off the −∞
                    // marker: an `s` in `(i128::MIN, i128::MIN + 4)`
                    // floors to `i128::MIN` itself — writing the sentinel
                    // that no stored cell may carry. `MIN + 4` is the
                    // tightest multiple of 4 above it and is still a
                    // sound, finite bound.
                    m[idx] = ((s >> 2) << 2).max(i128::MIN + 4);
                    changed = true;
                }
            }
        }
        changed
    }

    /// One Bagnara–Hill–Zaffanella tight-closure round ([BHZ07],
    /// Figure 2 / Theorems 2–3), applied to an ALREADY-CLOSED matrix:
    /// tighten every self-dual edge to the integer normal form
    /// (stored ≡ 0 (mod 4)), check Z-consistency, then run a single
    /// strong-coherence (S⁺) pass. Returns `false` when the constraint
    /// system has no integer solution (the caller reports ⊥).
    ///
    /// Why a single round suffices — [BHZ07] Thm 2: on a closed
    /// integer octagonal graph, `min(w(i,j), ⌊w(i,ı)/2⌋ + ⌊w(j̄,j)/2⌋)`
    /// IS the tight closure; with the self-dual weights already
    /// tightened to even, the floor-halved sum is exact, and in this
    /// module's doubled storage (BHZ weight w = stored s / 2; "w even"
    /// = "s ≡ 0 (mod 4)") it coincides with `strong_coherence_pass`'s
    /// ceil-halving on mod-4-clean operands. Thm 3: the O(n) check
    /// `m[2v][2v+1] + m[2v+1][2v] ≥ 0` after tightening decides
    /// Z-consistency — sound on ANY matrix (an empty integer-tightened
    /// interval for one variable admits no integer point) and complete
    /// on a closed coherent one. This replaces the former
    /// tighten/re-close fixpoint loop (worst case exponential rounds,
    /// engineering-capped at 4N+64 with a best-effort release mode).
    ///
    /// Corner domain: the Z-check uses `sat_add_closed`, whose clamp
    /// only RAISES sums, so it can never report ⊥ spuriously; an
    /// unrepresentable (≤ −2^127) sum is still negative after the
    /// clamp, so extreme contradictions are still caught.
    fn tight_closure_round(m: &mut [i128], size: usize) -> bool {
        Self::tighten_self_dual(m, size);
        for v in 0..size / 2 {
            let p = 2 * v;
            let q = p + 1;
            let s = sat_add_closed(m[p * size + q], m[q * size + p]);
            if s < 0 {
                return false; // Z-inconsistent ([BHZ07] Thm 3)
            }
        }
        Self::strong_coherence_pass(m, size);
        true
    }

    /// Close under the given `ClosureMode`. See the enum docs for the
    /// integer-tightening profile.
    pub(crate) fn close_with(&mut self, mode: ClosureMode) -> bool {
        // Every exit path leaves the matrix strongly closed (or ⊥, which
        // is vacuously closed), so the flag is set up front — the closure
        // passes below are atomic from the caller's perspective.
        self.closed = true;
        if self.bottom {
            return false;
        }
        let size = self.size;
        // Strong closure (all-pivot variant of [Miné06] Figure 8): one
        // pass of the C⁺_k / S⁺ interleaving over all k.
        Self::strong_closure_pass(&mut self.m, size);
        // ---- Bagnara–Hill–Zaffanella tight closure ([BHZ07], Figure 2) ----
        // `IntegerExact` applies, to the closed matrix above: ONE
        // Harvey–Stuckey tightening pass ([HS97]; see [Miné06] §V.D:
        // `2x ≤ 2c+1 ⟹ 2x ≤ 2c`; the rounding preserves the INTEGER
        // solution set of each constraint — and is unsound over
        // rationals: 2x ≤ 5 admits x = 2.5, the tightened 2x ≤ 4 does
        // not — integer domains only), an O(n) Z-consistency check on
        // each self-dual pair, and a single strong-coherence (S⁺)
        // pass. [BHZ07] Thm 2 proves that ONE round on a closed graph
        // computes the tight closure exactly — no fixpoint iteration,
        // no round cap; Thm 3 proves the self-dual-pair check decides
        // Z-consistency, so a failed check is a sound and COMPLETE ⊥
        // report.
        if mode == ClosureMode::IntegerExact {
            if !Self::tight_closure_round(&mut self.m, size) {
                self.bottom = true;
                return false;
            }
        }
        // ---- Enforce coherence (LOAD-BEARING, not just defensive) ----
        // The min-copy repairs (a) non-coherent inputs built with the
        // one-sided `set`, and (b) the mirror asymmetry the closure pass
        // itself can CREATE in the corner domain, where the clamped
        // addition is non-associative and the left-folded C⁺ arms can
        // differ from their mirrored right-folds. It only ever tightens
        // with existing valid bounds. If it changed anything, the
        // closedness conditions may no longer hold (the copied bounds
        // can open NEW paths whose closure is not yet derived), so the
        // strong closure is re-run below.
        let mut repaired = false;
        for i in 0..size {
            for j in 0..size {
                let i_bar = mirror_index(i);
                let j_bar = mirror_index(j);
                let a = self.m[i * size + j];
                let b = self.m[j_bar * size + i_bar];
                if a < b {
                    self.m[j_bar * size + i_bar] = a;
                    repaired = true;
                } else if b < a {
                    self.m[i * size + j] = b;
                    repaired = true;
                }
            }
        }
        if repaired {
            // A fired repair means the matrix was not mirror-symmetric
            // going in: re-run the strong closure to restore the
            // closedness conditions (for the corner-domain asymmetry the
            // rescue is already complete — the re-run is a no-op).
            // IntegerExact re-runs its tight-closure round so the
            // rounded normal form survives the re-close.
            Self::strong_closure_pass(&mut self.m, size);
            if mode == ClosureMode::IntegerExact {
                if !Self::tight_closure_round(&mut self.m, size) {
                    self.bottom = true;
                    return false;
                }
            }
        }
        // ---- Diagonal unsatisfiability check ----
        for i in 0..size {
            let d = self.m[i * size + i];
            if d < 0 {
                self.bottom = true;
                return false;
            }
            self.m[i * size + i] = 0; // normalize
        }
        // ---- i128::MIN normalization (belt-and-braces) ----
        // `i128::MIN` is the −∞ marker in `sat_sub`/`sat_neg`, so no
        // stored cell may carry it. The closure passes never WRITE it
        // (path sums that would saturate are CLAMPED to `MIN + 4` by
        // `sat_add_closed`), `tighten_self_dual` clamps its rounding off
        // it, and `sat_mul2_bound` refuses it on insertion; this loop
        // only cleans up raw cell writes (e.g. in tests). Weakening to
        // `DBM_INF` cannot break strong closure (the closure invariant
        // and the checker both use the clamped arithmetic). Runs AFTER
        // the diagonal check so a negative cycle is detected first.
        for cell in self.m.iter_mut() {
            if *cell == i128::MIN {
                *cell = DBM_INF;
            }
        }
        // Debug tripwire: the IntegerExact output must be in the
        // Harvey–Stuckey rounded normal form (every finite self-dual
        // stored bound ≡ 0 (mod 4)) — [BHZ07] Def. 5 property (7).
        if mode == ClosureMode::IntegerExact {
            debug_assert!(
                (0..size / 2).all(|v| {
                    let p = 2 * v;
                    let q = p + 1;
                    let a = self.m[p * size + q];
                    let b = self.m[q * size + p];
                    (a == DBM_INF || (a & 3) == 0) && (b == DBM_INF || (b & 3) == 0)
                }),
                "IntegerExact output must be in the rounded normal form"
            );
        }
        self.bottom = false;
        true
    }

    /// The unconstrained (top) matrix.
    #[allow(dead_code)]
    pub(crate) fn top(&self) -> Dbm {
        let mut m = vec![DBM_INF; self.size * self.size];
        for i in 0..self.size {
            m[i * self.size + i] = 0;
        }
        Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true, // the fresh top matrix is strongly closed
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Lattice operations
    // ─────────────────────────────────────────────────────────────────

    /// meet (⊓): intersection – tighter of each bound, then close.
    /// If either operand is bottom, result is bottom.
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    #[allow(dead_code)]
    pub(crate) fn meet(&self, other: &Dbm) -> Dbm {
        if self.bottom || other.bottom {
            return Dbm::bottom();
        }
        assert_eq!(self.size, other.size, "meet requires same-size operands");
        let m: Vec<i128> = self
            .m
            .iter()
            .zip(&other.m)
            .map(|(a, b)| (*a).min(*b))
            .collect();
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Check the strong-closure invariants ([Miné06] Figure 8), read-only —
    /// never repairs.  Four properties:
    ///
    /// 1. **Coherence**: `m[i][j] == m[j̄][ī]` (the mirrored entry).
    /// 2. **Triangle inequality** (C⁺): `m[i][j] ≤ m[i][k] + m[k][j]` for
    ///    every pivot k — the two-edge-path arm.  The longer C⁺ arms
    ///    (i→k̄→j, i→k→k̄→j, i→k̄→k→j) are implied: e.g. the triangle
    ///    gives `m[i][k̄] ≤ m[i][k] + m[k][k̄]`, so any 3- or 4-edge path
    ///    through k̄ is bounded by the 1-edge arm through k̄ already
    ///    checked here.
    /// 3. **S⁺ half-sum**: `m[i][j] ≤ ceil((m[i][ī] + m[j̄][j]) / 2)` —
    ///    the same ceil-rounding as the closure pass (INF entries skip).
    /// 4. **Zero diagonal**: `m[i][i] == 0`.
    ///
    /// ⊥ is vacuously closed (no constraints to violate).  O(N³) — used
    /// only in debug assertions (the preservation-lemma tripwire) and
    /// tests, never on a hot path.
    pub(crate) fn is_strongly_closed(&self) -> bool {
        if self.bottom {
            return true; // ⊥ is vacuously strongly closed.
        }
        let size = self.size;
        let m = &self.m;
        // 4. Zero diagonal.
        for i in 0..size {
            if m[i * size + i] != 0 {
                return false;
            }
        }
        for i in 0..size {
            let i_bar = mirror_index(i);
            for j in 0..size {
                // 1. Coherence: m[i][j] == m[j̄][ī].
                let j_bar = mirror_index(j);
                if m[i * size + j] != m[j_bar * size + i_bar] {
                    return false;
                }
                // 3. S⁺ half-sum (INF skips — no derived bound; sums are
                //    clamped off i128::MIN by sat_add_closed).
                let t1 = m[i * size + i_bar];
                let t2 = m[j_bar * size + j];
                if t1 != DBM_INF && t2 != DBM_INF {
                    let sum = sat_add_closed(t1, t2);
                    if sum != DBM_INF {
                        let halved = (sum + 1) >> 1; // ceil(sum / 2)
                        if halved < m[i * size + j] {
                            return false;
                        }
                    }
                }
                // 2. Triangle inequality over every pivot k.
                for k in 0..size {
                    let t_ik = m[i * size + k];
                    let t_kj = m[k * size + j];
                    if t_ik != DBM_INF && t_kj != DBM_INF {
                        let s = sat_add_closed(t_ik, t_kj);
                        if s != DBM_INF && s < m[i * size + j] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// join (⊔): abstract disjunction — point-wise max off the diagonal
    /// (min on the diagonal). O(N²); no closure pass.
    ///
    /// # Why the output is strongly closed (no close() needed)
    ///
    /// Let F be strong closure. F is monotone and contracting
    /// (F(M) ≤ M: closure only tightens entries). On strongly closed
    /// operands A = F(A), B = F(B), and the join M satisfies A ≤ M and
    /// B ≤ M point-wise. Monotonicity gives A ≤ F(M) and B ≤ F(M), hence
    /// M = max(A, B) ≤ F(M); contraction gives F(M) ≤ M. Therefore
    /// F(M) = M.
    ///
    /// Equivalently, point-wise: every closedness condition upper-bounds
    /// M_ij by an expression of other entries, and each branch of an
    /// outer max inherits its operand's own inequality —
    ///   - triangle: A_ij ≤ A_ik + A_kj implies
    ///     max(A_ij,B_ij) ≤ max(A_ik,B_ik) + max(A_kj,B_kj), since each
    ///     right-hand term only grows under the max;
    ///   - S⁺ saturation: same transport of A_ij ≤ (A_iī + A_j̄j)/2;
    ///   - coherence: the mirror of a point-wise max is the point-wise
    ///     max of the mirrors (M_j̄ī = max(A_j̄ī, B_j̄ī) = M_ij);
    ///   - diagonal: min(0, 0) = 0.
    /// ([Miné06], Thm 7.3 remark: "if the two arguments of ∨ are
    /// strongly closed, then the result is also strongly closed.")
    ///
    /// # Bottom is unreachable (non-bottom operands)
    ///
    /// F(M) = M with a zero diagonal admits no strictly negative cycle,
    /// so negative-diagonal detection could never fire here; the operand
    /// shortcuts below are the only bottom paths.
    ///
    /// # IntegerExact note
    ///
    /// The max of two multiples of 4 is a multiple of 4, so this join
    /// also preserves the Harvey–Stuckey rounded normal form
    /// ([HS97]; see [Miné06] §V.D)
    /// whenever both operands carry it — this operation never exits it.
    ///
    /// # Precondition
    ///
    /// Strongly closed, non-bottom operands. The debug_assert is the
    /// tripwire for the preservation argument above. A failure means
    /// either (a) a closure-preservation lemma inside this module is
    /// wrong (a bug here), or (b) the caller fed a non-closed state —
    /// most notably a `widen` result, whose output is deliberately NOT
    /// strongly closed ([Miné06] §VI.D): re-close it before joining.
    pub(crate) fn join(&self, other: &Dbm) -> Dbm {
        debug_assert!(
            self.closed && other.closed,
            "join requires strongly closed operands — a violation is either a \
             closure-preservation bug in this module, or a caller feeding a \
             non-closed state (widen's output is deliberately unclosed — \
             re-close it first; see the docs)"
        );
        if self.bottom {
            return other.clone();
        }
        if other.bottom {
            return self.clone();
        }
        assert_eq!(self.size, other.size, "join requires same-size operands");
        let size = self.size;
        // Point-wise max; the diagonal is 0 in both strongly-closed
        // operands, so force it to 0 — avoids the idx/size and idx%size
        // divisions in the hot path (min on the diagonal is identical
        // to max here, both being 0 for closed operands).
        let mut m: Vec<i128> = self
            .m
            .iter()
            .zip(&other.m)
            .map(|(a, b)| (*a).max(*b))
            .collect();
        for i in 0..size {
            m[i * size + i] = 0;
        }
        // Strongly closed ∨ strongly closed = strongly closed (see doc):
        // no closure pass, bottom unreachable.
        Dbm {
            size,
            m,
            bottom: false,
            closed: true,
        }
    }

    /// widen (∇): termination-guaranteeing upper approximation.
    /// Definition ([Miné06]): if new bound is ≤ old, keep old; otherwise ∞.
    /// Does NOT call close after widening ([Miné06] §VI.D) — the output
    /// is deliberately NOT strongly closed (closing the left operand of
    /// the next widening can create the non-terminating chain of
    /// [Miné06] Figure 10 / Thm 8.2). Callers that need a closed state
    /// (e.g. before `join`) must re-close it.
    pub(crate) fn widen(&self, new: &Dbm) -> Dbm {
        if self.bottom {
            return new.clone();
        }
        if new.bottom {
            return self.clone();
        }
        assert_eq!(self.size, new.size, "widen requires same-size operands");
        let m: Vec<i128> = self
            .m
            .iter()
            .zip(&new.m)
            .map(|(old, n)| {
                // If new is tighter (≤ old) we keep old (more conservative);
                // otherwise drop to ∞.
                if *n <= *old { *old } else { DBM_INF }
            })
            .collect();
        Dbm {
            size: self.size,
            m,
            bottom: false,
            // widen's output is deliberately NOT strongly closed
            // ([Miné06] §VI.D) — re-close before join.
            closed: false,
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Transfer functions (coherent)
    // ─────────────────────────────────────────────────────────────────

    /// `X := X + c`. O(N²); no closure pass.
    ///
    /// # Why this transfer is closure-preserving (no close() needed)
    ///
    /// The four update lines reweight the potential graph by a node
    /// potential: with σ = +1 on X⁺ (node 2i), −1 on X⁻ (node 2i+1), 0
    /// elsewhere, and δ = 2c (stored/doubled shift),
    ///     m'[u][v] = m[u][v] + δ·(σ_u − σ_v).
    /// Every path u → … → v telescopes: its weight shifts by exactly
    /// δ·(σ_u − σ_v), independent of intermediate routing (the classical
    /// argument behind Johnson's potential reweighting for APSP). Strong
    /// closure is shortest-path completeness on the potential graph, and
    /// reweighting by potentials commutes with it. Condition by
    /// condition:
    /// - triangle: the left side m'[u][v] shifts by δ(σ_u − σ_v); the
    ///   right side m'[u][k] + m'[k][v] shifts by
    ///   δ(σ_u − σ_k) + δ(σ_k − σ_v) = δ(σ_u − σ_v) — the same amount,
    ///   so the inequality is transported unchanged;
    /// - S⁺ saturation: σ_ī = −σ_i, so the right side of
    ///   m[i][j] ≤ (m[i][ī] + m[j̄][j])/2 shifts by
    ///   (δ(σ_i + σ_i) + δ(−σ_j − σ_j))/2 = δ(σ_i − σ_j), identical to
    ///   the left side's shift;
    /// - coherence: the mirror edge (j̄, ī) shifts by
    ///   δ(σ_j̄ − σ_ī) = δ(σ_i − σ_j), the same as (i, j) itself;
    /// - diagonal: σ_u − σ_u = 0, untouched — a zero diagonal cannot
    ///   become negative, so bottom is unreachable (non-bottom input).
    ///   (Saturation corners — δ = DBM_INF or δ = i128::MIN — never reach
    ///   the raw update: the potential argument requires exact
    ///   reweighting, so `assign_add_var` forgets the variable first; see
    ///   below. Even on the finite-δ path an INDIVIDUAL entry can clip
    ///   (e.g. `sat_add(20, δ) = DBM_INF` when `20 + δ ≥ DBM_INF`); the
    ///   update tracks that and falls back to a closing pass — a clipped
    ///   matrix is still a point-wise over-approximation of the exact
    ///   reweighting, so closure restores strong closure soundly.)
    ///
    /// Every stored shift is even (δ = 2c), so the ceil-halving inside
    /// S⁺'s encoded form sees unchanged input parity.
    ///
    /// # IntegerExact note
    ///
    /// A self-dual edge shifts by ±2δ = ±4c, a multiple of 4: this
    /// transfer PRESERVES the Harvey–Stuckey rounded normal form
    /// ([HS97]; see [Miné06] §V.D),
    /// so a follow-up `close_with(ClosureMode::IntegerExact)` is a no-op
    /// (the tightening step finds nothing to round).  The sole exception
    /// is the saturation regime (δ = DBM_INF or δ = i128::MIN), where the
    /// variable is forgotten instead of shifted — no rounded-form claim.
    pub(crate) fn assign_add_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        let delta = sat_mul2(c); // doubled space
        // If the additive shift saturates (δ = DBM_INF positive, δ =
        // i128::MIN negative), the potential reweighting is no longer
        // exact: raw entries land on INF/saturated values, the no-close
        // argument breaks, and `i128::MIN` cells collide with the -∞
        // marker. The shifted bounds are unrepresentable — forget the
        // variable (sound over-approximation) instead of shifting.
        if delta == DBM_INF || delta == i128::MIN {
            return self.forget_var(i);
        }
        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let mut m = self.m.clone();
        let mut saturated = false;
        // Corner-regime scan: if the input carries any finite stored
        // value ≤ −2^126 + 1, two such cells can compose to a path sum
        // < i128::MIN + 4, so the closure may have CLAMPED derived
        // bounds (`sat_add_closed` raises them to MIN + 4 — looser than
        // the exact value). The potential-reweighting preservation
        // argument requires the input to satisfy the EXACT triangle
        // inequalities; a clamped input does not, so the no-close fast
        // path could return a non-strongly-closed matrix. Treat the
        // corner regime like a saturation event: close after the shift.
        if self
            .m
            .iter()
            .any(|&v| v != DBM_INF && v <= -(1i128 << 126) + 1)
        {
            saturated = true;
        }
        for j in 0..self.size {
            let old = m[p * self.size + j];
            let new = sat_add(old, delta);
            saturated |=
                (new == DBM_INF && old != DBM_INF) || (new == i128::MIN && old != i128::MIN);
            m[p * self.size + j] = new;
            let old = m[j * self.size + p];
            let new = sat_sub(old, delta);
            saturated |=
                (new == DBM_INF && old != DBM_INF) || (new == i128::MIN && old != i128::MIN);
            m[j * self.size + p] = new;
            let old = m[q * self.size + j];
            let new = sat_sub(old, delta);
            saturated |=
                (new == DBM_INF && old != DBM_INF) || (new == i128::MIN && old != i128::MIN);
            m[q * self.size + j] = new;
            let old = m[j * self.size + q];
            let new = sat_add(old, delta);
            saturated |=
                (new == DBM_INF && old != DBM_INF) || (new == i128::MIN && old != i128::MIN);
            m[j * self.size + q] = new;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        if saturated {
            r.close();
            if r.bottom {
                return Dbm::bottom();
            }
        }
        r
    }

    /// `X := c`
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    pub(crate) fn assign_const_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        let upper = c.checked_mul(4);
        let lower = c.checked_mul(-4);
        let exact = matches!(upper, Some(v) if v < DBM_INF && v > i128::MIN)
            && matches!(lower, Some(v) if v < DBM_INF && v > i128::MIN);
        if !exact {
            return self.forget_var(i);
        }
        let p = 2 * i;
        let q = 2 * i + 1;
        let mut m = self.m.clone();
        for j in 0..self.size {
            m[p * self.size + j] = DBM_INF;
            m[q * self.size + j] = DBM_INF;
            m[j * self.size + p] = DBM_INF;
            m[j * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        let upper_actual = sat_mul2(c);
        let lower_actual = sat_mul2(sat_neg(c));
        Self::set_mirrored_internal(&mut m, p, q, upper_actual, self.size);
        Self::set_mirrored_internal(&mut m, q, p, lower_actual, self.size);
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// `X := Y` (copy assignment without offset).
    ///
    /// This is the special case of `assign_copy_add_var(i, j, 0)`.
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    pub(crate) fn assign_copy_var(&self, i: usize, j: usize) -> Dbm {
        self.assign_copy_add_var(i, j, 0)
    }

    // ─────────────────────────────────────────────────────────────────
    // Guard transfer functions
    // ─────────────────────────────────────────────────────────────────

    /// Test `X ≤ c` (coherent) — self-dual edge `(2i, 2i+1)` carries
    /// `2X ≤ 2c` (stored `4c`).
    ///
    /// Encoding from [Miné06] Figure 5:
    ///   `vᵢ ≤ c`  ⟺  `v⁺ᵢ − v⁻ᵢ ≤ 2c`
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    pub(crate) fn test_le_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * i + 1, sat_mul2(c));
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X ≥ c` (coherent) — self-dual edge `(2i+1, 2i)` carries
    /// `-2X ≤ -2c` (stored `-4c`).
    ///
    /// Encoding from [Miné06] Figure 5:
    ///   `vᵢ ≥ c`  ⟺  `v⁻ᵢ − v⁺ᵢ ≤ −2c`
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    pub(crate) fn test_ge_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        let mut r = self.clone();
        r.set_mirrored(2 * i + 1, 2 * i, sat_mul2(sat_neg(c)));
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X - Y ≤ c` (coherent) — edge `(2i, 2j)` carries
    /// `X⁺ − Y⁺ ≤ c`, i.e. `X − Y ≤ c`.
    ///
    /// Encoding from [Miné06] Figure 5:
    ///   `vᵢ − vⱼ ≤ c`  ⟺  `v⁺ᵢ − v⁺ⱼ ≤ c`  ∧  `v⁻ⱼ − v⁻ᵢ ≤ c`
    ///   (the second is the coherence mirror of the first)
    ///
    /// (Internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    pub(crate) fn test_diff_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * j, c);
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X + Y ≤ c` (coherent).
    ///
    /// Encoding from [Miné06] Figure 5:
    ///   `vᵢ + vⱼ ≤ c`  ⟺  `v⁺ᵢ − v⁻ⱼ ≤ c`  ∧  `v⁺ⱼ − v⁻ᵢ ≤ c`
    ///
    /// In our DBM indices: edge `(2i, 2j+1)` carries `X⁺ − Y⁻ ≤ c`,
    /// and its coherence mirror `((2j+1)̄, (2i)̄) = (2j, 2i+1)` carries
    /// `Y⁺ − X⁻ ≤ c`. Both encode the same octagonal constraint
    /// `X + Y ≤ c`.
    ///
    /// This is Definition 2 item 1 of [Miné06]:
    ///   `[m⁺(vₖ+vₗ≤c)]ᵢⱼ = min(m⁺ᵢⱼ, c)` if `(i,j) ∈ {(2k, 2l+1); (2l, 2k+1)}`
    ///   (transposed to our convention)
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn test_sum_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        let mut r = self.clone();
        // set_mirrored(2i, 2j+1, c) sets:
        //   (2i, 2j+1) with bound c  →  X⁺ − Y⁻ ≤ c  →  X + Y ≤ c
        //   mirror ((2j+1)^1, (2i)^1) = (2j, 2i+1) with bound c
        //                            →  Y⁺ − X⁻ ≤ c  →  X + Y ≤ c  ✓
        r.set_mirrored(2 * i, 2 * j + 1, c);
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `−X − Y ≤ c` (negated-sum guard, coherent).
    ///
    /// Encoding from [Miné06] Figure 5:
    ///   `−vᵢ − vⱼ ≤ c`  ⟺  `v⁻ⱼ − v⁺ᵢ ≤ c`  ∧  `v⁻ᵢ − v⁺ⱼ ≤ c`
    ///
    /// In our DBM indices: edge `(2j+1, 2i)` carries `Y⁻ − X⁺ ≤ c`
    /// = `−Y − X ≤ c`, and its coherence mirror `(2i+1, 2j)` carries
    /// `X⁻ − Y⁺ ≤ c` = `−X − Y ≤ c`.
    ///
    /// This is the "−vₖ − vₗ ≤ c" case of Definition 2 item 1 in
    /// [Miné06], with the translation from Figure 5.
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn test_neg_sum_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        let mut r = self.clone();
        // set_mirrored(2j+1, 2i, c) sets:
        //   (2j+1, 2i) with bound c  →  Y⁻ − X⁺ ≤ c  →  −Y − X ≤ c
        //   mirror ((2i)^1, (2j+1)^1) = (2i+1, 2j) with bound c
        //                            →  X⁻ − Y⁺ ≤ c  →  −X − Y ≤ c  ✓
        r.set_mirrored(2 * j + 1, 2 * i, c);
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X + Y = c` (equality guard for sum).
    ///
    /// Decomposition from [Miné06] Definition 2, item 3:
    ///   `m⁺(vₖ + vₗ = c) = (m⁺(vₖ + vₗ ≤ c))(−vₖ − vₗ ≤ −c)`
    ///
    /// That is, `X + Y = c` is the conjunction of:
    ///   1. `X + Y ≤ c`   (sum guard)
    ///   2. `−X − Y ≤ −c` (negated-sum guard with bound −c)
    ///
    /// The conjunction is implemented as a single pass that sets both
    /// bounds and closes once — equivalent to the meet of the two
    /// individual guard refinements (close is a kernel operator).
    ///
    /// Soundness: if either sub-guard produces ⊥, the conjunction is ⊥.
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn test_sum_eq(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        // Both bounds in one pass, then a single close: the two
        // sub-guards share the same base matrix, and closing once over
        // the combined constraint set is equivalent to closing after
        // each (close is a kernel operator).
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * j + 1, c); // X + Y ≤ c
        r.set_mirrored(2 * j + 1, 2 * i, sat_neg(c)); // −X − Y ≤ −c
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X − Y = c` (equality guard for difference).
    ///
    /// Decomposition analogous to [Miné06] Definition 2 item 3:
    ///   `m⁺(vₖ − vₗ = c) = (m⁺(vₖ − vₗ ≤ c))(vₗ − vₖ ≤ −c)`
    ///
    /// That is, `X − Y = c` is the conjunction of:
    ///   1. `X − Y ≤ c`   (difference guard)
    ///   2. `Y − X ≤ −c`  (difference guard with swapped args)
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn test_diff_eq(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        // Both bounds in one pass, then a single close: the two
        // sub-guards share the same base matrix, and closing once over
        // the combined constraint set is equivalent to closing after
        // each (close is a kernel operator).
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * j, c); // X − Y ≤ c
        r.set_mirrored(2 * j, 2 * i, sat_neg(c)); // Y − X ≤ −c
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X = c` (equality guard for a single variable).
    ///
    /// Decomposition:
    ///   `X = c`  ⟺  `X ≤ c ∧ X ≥ c`
    ///
    /// This is the conjunction of `test_le_var(i, c)` and
    /// `test_ge_var(i, c)`, merged into a single set + one close.
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn test_var_eq(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        // Both bounds in one pass, then a single close: the two
        // sub-guards share the same base matrix, and closing once over
        // the combined constraint set is equivalent to closing after
        // each (close is a kernel operator).
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * i + 1, sat_mul2(c)); // X ≤ c
        r.set_mirrored(2 * i + 1, 2 * i, sat_mul2(sat_neg(c))); // X ≥ c
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    // ─────────────────────────────────────────────────────────────────
    // Boolean combinators for guards ([Miné06] Definition 3)
    // ─────────────────────────────────────────────────────────────────

    /// Boolean AND of two guard results: `m⁺(g₁ ∧ g₂) = m⁺(g₁) ∧ m⁺(g₂)`
    ///
    /// From [Miné06] Definition 3, item 1:
    ///   `m⁺(g₁ and g₂) ≜ m⁺(g₁) ∧ m⁺(g₂)`
    ///
    /// The `∧` operator is the point-wise minimum (meet) of the two
    /// DBMs followed by closure (Theorem 7.1: intersection is exact).
    ///
    /// Both operands must be the results of applying guards to the
    /// SAME input state (i.e., both refine the same base DBM). The
    /// meet computes their conjunction.
    ///
    /// Returns ⊥ if the conjunction is unsatisfiable.
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn guard_and(&self, other: &Dbm) -> Dbm {
        // guard_and is semantically identical to meet.
        self.meet(other)
    }

    /// Boolean OR of two guard results:
    ///   `m⁺(g₁ ∨ g₂) = ((m⁺(g₁))•) ∨ ((m⁺(g₂))•)`
    ///
    /// From [Miné06] Definition 3, item 2:
    ///   `m⁺(g₁ or g₂) ≜ ((m⁺(g₁))•) ∨ ((m⁺(g₂))•)`
    ///
    /// The `∨` operator is the point-wise maximum (join) of the two
    /// STRONGLY CLOSED DBMs. By Theorem 7.3, the join of two strongly
    /// closed DBMs is the best octagonal over-approximation of the
    /// union of their V⁺-domains.
    ///
    /// Precondition: both `self` and `other` must be strongly closed
    /// (they are guard results, which close internally). The `join`
    /// method enforces this via debug_assert.
    ///
    /// Note: the union of two octagons is not necessarily an octagon,
    /// so this is an over-approximation (Theorem 7.2: ⊇).
    pub(crate) fn guard_or(&self, other: &Dbm) -> Dbm {
        if self.bottom {
            return other.clone();
        }
        if other.bottom {
            return self.clone();
        }
        // Both guard results are strongly closed (guards close internally).
        // Theorem 7.3: join of strongly closed DBMs is the best
        // octagonal over-approximation of the union.
        self.join(other)
    }

    /// Boolean NOT via De Morgan's laws ([Miné06] Definition 3, item 3):
    ///   `m⁺(¬g)` is handled by structural recursion on the guard:
    ///   - `¬(g₁ ∧ g₂)` → `(¬g₁) ∨ (¬g₂)`
    ///   - `¬(g₁ ∨ g₂)` → `(¬g₁) ∧ (¬g₂)`
    ///   - `¬(X ≤ c)` → `X ≥ c + 1` (over integers)
    ///   - `¬(X ≥ c)` → `X ≤ c − 1` (over integers)
    ///   - `¬(X + Y ≤ c)` → `X + Y ≥ c + 1` → `−X − Y ≤ −(c+1)`
    ///
    /// Since the octagon domain is NOT closed under complement (the
    /// complement of an octagon is generally not an octagon), this
    /// method implements negation only for the primitive guard forms
    /// that have exact octagonal complements. For general guards, the
    /// caller must decompose using De Morgan's laws and apply the
    /// appropriate positive guards.
    ///
    /// This method handles the atomic negation `¬(X ≤ c)` = `X ≥ c+1`
    /// for integer domains. Over reals/rationals, `¬(X ≤ c)` = `X > c`
    /// which is NOT representable as a closed bound; we approximate
    /// soundly with `X ≥ c + 1` (valid only over integers).
    ///
    /// For the general "negate an arbitrary guard" case, the caller
    /// should use the `guard_not_le_var` / `guard_not_ge_var` /
    /// `guard_not_sum_le` / `guard_not_diff_le` helpers below.
    ///
    /// `guard_not_le_var(i, c)`: ¬(X ≤ c) = X ≥ c + 1 (integer)
    pub(crate) fn guard_not_le_var(&self, i: usize, c: i128) -> Dbm {
        // ¬(X ≤ c) ⟺ X > c ⟺ X ≥ c + 1 over Z
        self.test_ge_var(i, sat_add(c, 1))
    }

    /// `guard_not_ge_var(i, c)`: ¬(X ≥ c) = X ≤ c − 1 (integer)
    pub(crate) fn guard_not_ge_var(&self, i: usize, c: i128) -> Dbm {
        // ¬(X ≥ c) ⟺ X < c ⟺ X ≤ c − 1 over Z
        self.test_le_var(i, sat_sub(c, 1))
    }

    /// `guard_not_sum_le(i, j, c)`: ¬(X + Y ≤ c) = X + Y ≥ c + 1
    /// ⟺ −X − Y ≤ −(c + 1) (integer)
    pub(crate) fn guard_not_sum_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        // ¬(X + Y ≤ c) ⟺ X + Y > c ⟺ X + Y ≥ c+1 ⟺ −X − Y ≤ −(c+1)
        self.test_neg_sum_le(i, j, sat_neg(sat_add(c, 1)))
    }

    /// `guard_not_diff_le(i, j, c)`: ¬(X − Y ≤ c) = X − Y ≥ c + 1
    /// ⟺ Y − X ≤ −(c + 1) (integer)
    pub(crate) fn guard_not_diff_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        // ¬(X − Y ≤ c) ⟺ X − Y > c ⟺ Y − X < −c ⟺ Y − X ≤ −c − 1
        self.test_diff_le(j, i, sat_sub(sat_neg(c), 1))
    }

    // ─────────────────────────────────────────────────────────────────
    // Assignment transfer functions (extended)
    // ─────────────────────────────────────────────────────────────────

    /// `X := Y + c` (copy assignment with constant offset, k ≠ l).
    ///
    /// From [Miné06] Definition 2, item 5:
    /// ```text
    /// [m⁺(vₖ ← vₗ + c)]ᵢⱼ =
    ///   c         if (j,i) ∈ {(2k, 2l); (2l+1, 2k+1)}   [X − Y ≤ c]
    ///   −c        if (j,i) ∈ {(2l, 2k); (2k+1, 2l+1)}   [Y − X ≤ −c]
    ///   (m⁺)•ᵢⱼ   if i, j ∉ {2k, 2k+1}                  [keep other info]
    ///   +∞        elsewhere                              [drop X's edges]
    /// ```
    ///
    /// Translated to our convention (m[i][j] bounds node_i − node_j):
    ///   - Drop X's rows/columns (set to +∞), keep diagonal 0.
    ///   - Set `X − Y ≤ c`:  `set_mirrored(2k, 2l, c)`
    ///     → edges (2k, 2l) and mirror (2l+1, 2k+1), both bound c.
    ///   - Set `Y − X ≤ −c`: `set_mirrored(2l, 2k, −c)`
    ///     → edges (2l, 2k) and mirror (2k+1, 2l+1), both bound −c.
    ///   - All other entries retain their (m⁺)• values from `self`.
    ///
    /// The use of `(m⁺)•` (the strongly closed form) for the retained
    /// entries prevents precision loss: implicit constraints involving
    /// X that were derivable through X's paths are already materialized
    /// in the closure, so dropping X's explicit edges does not lose
    /// information about other variables.
    ///
    /// Special case: `c = 0` gives `X := Y` (pure copy).
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn assign_copy_add_var(&self, i: usize, j: usize, c: i128) -> Dbm {
        if i == j {
            // X := X + c is an additive shift, not a copy.
            return self.assign_add_var(i, c);
        }
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        self.check_var(j);
        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let yp = 2 * j; // Y⁺
        let yq = 2 * j + 1; // Y⁻
        let mut m = self.m.clone();
        // Drop X's rows/columns (Definition 2.5: +∞ elsewhere for
        // entries involving X's node indices).
        for k in 0..self.size {
            m[p * self.size + k] = DBM_INF;
            m[q * self.size + k] = DBM_INF;
            m[k * self.size + p] = DBM_INF;
            m[k * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        // Set X − Y ≤ c:
        //   set_mirrored(p, yp, c) → edge (p, yp) = (2i, 2j) bound c
        //   mirror: (yp̄, p̄) = (2j+1, 2i+1) bound c
        //   Semantics: X⁺ − Y⁺ ≤ c → X − Y ≤ c
        //              Y⁻ − X⁻ ≤ c → −Y + X ≤ c → X − Y ≤ c  ✓
        Self::set_mirrored_internal(&mut m, p, yp, c, self.size);
        // Set Y − X ≤ −c:
        //   set_mirrored(yp, p, −c) → edge (yp, p) = (2j, 2i) bound −c
        //   mirror: (p̄, ȳp̄) = (2i+1, 2j+1) bound −c
        //   Semantics: Y⁺ − X⁺ ≤ −c → Y − X ≤ −c
        //              X⁻ − Y⁻ ≤ −c → −X + Y ≤ −c → Y − X ≤ −c  ✓
        Self::set_mirrored_internal(&mut m, yp, p, sat_neg(c), self.size);
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// `X := expr` — general assignment via projection + interval
    /// arithmetic (the fallback of [Miné06] Definition 2.6 + §VI.E).
    ///
    /// # Algorithm (from [Miné06] §VI.E, after Definition 2)
    ///
    /// For a general assignment `vₖ ← e(v₀, ..., v_{N−1})`:
    ///
    /// 1. **Project**: extract the interval of each variable referenced
    ///    by `e` using Theorem 6:
    ///    ```text
    ///    {t | ∃s ∈ D⁺(m⁺), sᵢ = t} = [−(m⁺)•_{2i+1,2i}/2, (m⁺)•_{2i,2i+1}/2]
    ///    ```
    ///    In our API: `[var_lb(i), var_ub(i)]`.
    ///
    /// 2. **Evaluate**: compute `[e_lo, e_hi]` ⊇ e([lb₀, ub₀], ...)
    ///    using interval arithmetic. The result is a sound
    ///    over-approximation: for all concrete states s ∈ D⁺(m⁺),
    ///    `e(s) ∈ [e_lo, e_hi]`.
    ///
    /// 3. **Assign**: produce the post-state:
    ///    ```text
    ///    [m⁺(vₖ ← e)]ᵢⱼ =
    ///      (m⁺)•ᵢⱼ    if i, j ∉ {2k, 2k+1}   (retain other info)
    ///      e_hi       if (i, j) = (2k, 2k+1)  (X ≤ e_hi, stored 2·e_hi)
    ///      −e_lo      if (i, j) = (2k+1, 2k)  (X ≥ e_lo, stored −2·e_lo)
    ///      +∞         elsewhere on X's rows/cols
    ///    ```
    ///
    /// # Precision
    ///
    /// This is a sound over-approximation. Precision loss comes from:
    /// - Interval arithmetic ignores correlations between variables.
    /// - The result is a "box" constraint on X, losing relational info
    ///   between X and other variables (except what the closure
    ///   materialized before dropping X's edges).
    ///
    /// To recover some relational precision, one could use the
    /// strongly closed form `(m⁺)•` for the retained entries (as we do
    /// here), which materializes all derivable constraints before
    /// dropping X's explicit edges.
    ///
    /// # Soundness argument
    ///
    /// For any concrete state `(s₀, ..., s_{N−1}) ∈ D⁺(m⁺)`:
    /// - After `X := e`, the new value of X is `e(s₀, ..., s_{N−1})`.
    /// - By interval evaluation, `e_lo ≤ e(s) ≤ e_hi`.
    /// - The post-state sets `X ≤ e_hi` and `X ≥ e_lo`.
    /// - Other variables are unchanged; their constraints are preserved
    ///   from `(m⁺)•` (minus X's rows/columns).
    /// - Therefore the post-state contains all concrete successors. ∎
    ///
    /// (Internal `close()` is `Strong`.)
    pub(crate) fn assign_expr_var(&self, i: usize, expr: &OctExpr) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);

        // Step 1 + 2: evaluate the expression using interval arithmetic
        // over the projected variable ranges.
        let (e_lo, e_hi) = expr.eval_interval(self);

        // Check representability: if the interval is too wide or
        // saturated, fall back to forgetting the variable entirely.
        // The self-dual edges store ±4·bound, so we need 4·|bound| < DBM_INF.
        let hi_representable = e_hi != DBM_INF
            && e_hi
                .checked_mul(4)
                .map_or(false, |v| v < DBM_INF && v > i128::MIN);
        let lo_representable = e_lo != i128::MIN
            && e_lo
                .checked_mul(-4)
                .map_or(false, |v| v < DBM_INF && v > i128::MIN);

        if !hi_representable && !lo_representable {
            // Both bounds unrepresentable: forget the variable.
            return self.forget_var(i);
        }

        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let mut m = self.m.clone();

        // Step 3: drop X's rows/columns, then set the computed bounds.
        for j in 0..self.size {
            m[p * self.size + j] = DBM_INF;
            m[q * self.size + j] = DBM_INF;
            m[j * self.size + p] = DBM_INF;
            m[j * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;

        // Set X ≤ e_hi on the self-dual edge (p, q):
        //   X⁺ − X⁻ ≤ 2·e_hi  (stored 4·e_hi)
        // Only set if representable; otherwise leave as +∞ (no upper bound).
        if hi_representable {
            // sat_mul2(e_hi) gives 2·e_hi; set_mirrored stores 2×(2·e_hi) = 4·e_hi
            Self::set_mirrored_internal(&mut m, p, q, sat_mul2(e_hi), self.size);
        }

        // Set X ≥ e_lo on the self-dual edge (q, p):
        //   X⁻ − X⁺ ≤ −2·e_lo  (stored −4·e_lo)
        // Only set if representable; otherwise leave as +∞ (no lower bound).
        if lo_representable {
            // sat_mul2(sat_neg(e_lo)) gives −2·e_lo;
            // set_mirrored stores 2×(−2·e_lo) = −4·e_lo
            Self::set_mirrored_internal(&mut m, q, p, sat_mul2(sat_neg(e_lo)), self.size);
        }

        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Forget variable `i` (existential projection): drop its rows and
    /// columns, then re-close. Used as the sound fallback when a
    /// transfer function's result would be unrepresentable — the
    /// weakened (unconstrained) result is an over-approximation, never
    /// a spurious `bottom`.
    ///
    /// (The internal `close()` is `Strong`: exits the Harvey–Stuckey
    /// rounded normal form — see `close`'s docs.)
    fn forget_var(&self, i: usize) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        self.check_var(i);
        let p = 2 * i;
        let q = 2 * i + 1;
        let mut m = self.m.clone();
        for j in 0..self.size {
            m[p * self.size + j] = DBM_INF;
            m[q * self.size + j] = DBM_INF;
            m[j * self.size + p] = DBM_INF;
            m[j * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    // ─────────────────────────────────────────────────────────────────
    // Semantic projections
    // ─────────────────────────────────────────────────────────────────

    /// The finite bound on `Xᵢ - Xⱼ` (actual value) or `None`.
    /// Delegates to `node_bound` (the single halving point).
    #[allow(dead_code)]
    pub(crate) fn diff_bound(&self, i: usize, j: usize) -> Option<i128> {
        self.node_bound(2 * i, 2 * j) // X−Y ≤ c
    }

    /// Semantic projection: the upper bound `Xᵢ ≤ c` or `None` (∞).
    /// Representation-independent read layer — the stable API for
    /// semantic-projection consumers and the golden bridge (the golden
    /// bridge is the representation-agnostic reading layer for the
    /// semantic-equivalence gate).
    ///
    /// 2N representation: the self-dual edge `(2i, 2i+1)` carries
    /// `2X ≤ 2c` (stored `4c`); the "/4" read is exact for both even and
    /// odd stored values (`floor(floor(s/2)/2) = floor(s/4)`).
    #[allow(dead_code)]
    pub(crate) fn var_ub(&self, i: usize) -> Option<i128> {
        self.node_bound(2 * i, 2 * i + 1).map(|v| v >> 1)
    }

    /// Semantic projection: the lower bound `Xᵢ ≥ c` or `None` (∞).
    /// The extra `>> 1` keeps the read exact on odd stored values: the
    /// self-dual edge may carry `2X ≥ 2c+1` (stored `-(4c+1)`), whose
    /// exact integer lower bound is `c+1` — `floor((4c+1)/4) = c` would
    /// be one-off-weak.
    ///
    /// Saturation behavior:
    /// The closure CLAMPS any path sum that would saturate to
    /// `i128::MIN` (exact bound −2^127, e.g. two −2^126 edges composing)
    /// to `MIN + 4` — `i128::MIN` is the −∞ marker and is never stored.
    /// The stored `MIN + 4` encodes `X⁻ - X⁺ ≤ -2^127 + 4`
    /// (i.e. `X ≥ 2^125 − 1`); `node_bound` reads it as `-2^126 + 2`,
    /// and `sat_neg(-2^125 + 1)` attempts `+2^125 − 1`, which sits at
    /// the top of the finite representable range and therefore
    /// saturates to `DBM_INF`. `DBM_INF` is the module's "no finite
    /// bound" sentinel; returning `Some(DBM_INF)` would be both wrong
    /// and ambiguous. The implementation maps that clipped value back
    /// to `None` (∞), the sound over-approximation for an
    /// unrepresentable lower bound.
    #[allow(dead_code)]
    pub(crate) fn var_lb(&self, i: usize) -> Option<i128> {
        self.node_bound(2 * i + 1, 2 * i)
            .map(|v| sat_neg(v >> 1))
            .and_then(|b| if b == DBM_INF { None } else { Some(b) })
    }

    /// Semantic projection: `Xᵢ + Xⱼ ≤ c` or `None` (∞) — the sum rides
    /// the edge `X⁺ᵢ − X⁻ⱼ` ([Miné06] Figure 5: `vᵢ + vⱼ ≤ c` ⟺
    /// `v⁺ᵢ − v⁻ⱼ ≤ c`), whose edge constant IS the semantic bound (no
    /// extra halving beyond `node_bound`'s).
    #[allow(dead_code)]
    pub(crate) fn sum_ub(&self, i: usize, j: usize) -> Option<i128> {
        self.node_bound(2 * i, 2 * j + 1) // X+Y ≤ c ⟺ X⁺−Y⁻ ≤ c
    }

    /// Equality of two matrices (must be closed and non-bottom).
    /// This is a convenience method; the derived `PartialEq` can also be used.
    pub(crate) fn eq(&self, other: &Dbm) -> bool {
        self == other
    }
}

// ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Tests
// ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strong_closure_deduces_sum() {
        // 2v₀ ≤ 1 (stored 2), 2v₁ ≤ 2 (stored 4) ⟹ S⁺: v₀+v₁ ≤ (1+2)/2 = 1.5
        // ⟹ stored 3.
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 1); // v₀⁺ - v₀⁻ ≤ 1 (self-dual: 2v₀ ≤ 1)
        d.set_mirrored(2, 3, 2); // v₁⁺ - v₁⁻ ≤ 2 (self-dual: 2v₁ ≤ 2)
        assert!(d.close());
        // v₀⁺ = node 0, v₁⁻ = node 3: v₀⁺ - v₁⁻ ≤ ceil((1+2)/2) = 1.5.
        let stored = d.m[0 * d.size + 3];
        assert_eq!(stored, 3); // stored = 2 × 1.5 = 3
    }

    #[test]
    fn test_bottom_propagation() {
        let mut d = Dbm::new(1);
        d.set(1, 0, -2); // X⁻ - X⁺ ≤ -2 ⟹ 2X ≥ 2 ⟹ X ≥ 1
        d.set(0, 1, 0); // X⁺ - X⁻ ≤ 0 ⟹ X ≤ 0
        assert!(!d.close());
        assert!(d.bottom);
        let j = d.join(&Dbm::new(1));
        assert!(!j.bottom);
        let m = d.meet(&Dbm::new(1));
        assert!(m.bottom);
    }

    #[test]
    fn test_widen_no_close() {
        let mut cur = Dbm::new(1);
        cur.set(0, 1, 0);
        cur.close();
        let mut next = Dbm::new(1);
        next.set(0, 1, 2);
        next.close();
        let w = cur.widen(&next);
        assert_eq!(w.var_ub(0), None);
        assert_ne!(w, cur);
        assert_ne!(w, next);
    }

    #[test]
    fn test_coherent_test_le_var() {
        let d = Dbm::new(1);
        let r = d.test_le_var(0, 5);
        assert!(!r.bottom);
        assert_eq!(r.m[0 * 2 + 1], 20);
        assert_eq!(r.var_ub(0), Some(5));
    }

    #[test]
    fn test_golden_projections_bridge() {
        // A. Transitivity: X−Y ≤ 4 ∧ Y−Z ≤ 2 ⟹ X−Z ≤ 6.
        let mut d = Dbm::new(3);
        d.set_mirrored(0, 2, 4);
        d.set_mirrored(2, 4, 2);
        assert!(d.close());
        assert_eq!(d.diff_bound(0, 2), Some(6), "FW transitivity");
        // B. Interval→sum (via the self-dual edges): X ≤ 5 ∧ Y ≤ 3
        // ⟹ X+Y ≤ 8.
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 10); // 2X ≤ 10 ⟺ X ≤ 5
        d.set_mirrored(2, 3, 6); // 2Y ≤ 6 ⟺ Y ≤ 3
        assert!(d.close());
        assert_eq!(d.var_ub(0), Some(5));
        assert_eq!(d.var_ub(1), Some(3));
        assert_eq!(d.sum_ub(0, 1), Some(8), "interval→sum via self-dual edges");
        // C. S⁺ self-dual composition: 2X ≤ 1 ∧ 2Y ≤ 2 ⟹ X+Y ≤ 1.5;
        // integer tight read is 1.
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 1);
        d.set_mirrored(2, 3, 2);
        assert!(d.close());
        assert_eq!(
            d.sum_ub(0, 1),
            Some(1),
            "S⁺ composition (cf. [Miné06] Figure 9)"
        );
    }

    #[test]
    fn test_closure_integer_exact_figure6() {
        let mut d = Dbm::new(1);
        d.set_mirrored(0, 1, 1); // 2x ≤ 1  (x ≤ 0.5)
        d.set_mirrored(1, 0, -1); // −2x ≤ −1 (x ≥ 0.5)
        assert!(d.close_with(ClosureMode::Strong), "real model x = 0.5");
        let mut d = Dbm::new(1);
        d.set_mirrored(0, 1, 1);
        d.set_mirrored(1, 0, -1);
        assert!(
            !d.close_with(ClosureMode::IntegerExact),
            "2x is even: no integer satisfies 2x ≤ 1 ∧ 2x ≥ 1"
        );
    }

    #[test]
    fn test_closure_integer_exact_figure9() {
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 1); // 2v₀ ≤ 1
        d.set_mirrored(2, 3, 2); // 2v₁ ≤ 2
        assert!(d.close_with(ClosureMode::IntegerExact));
        assert_eq!(d.sum_ub(0, 1), Some(1), "integer-tight v₀+v₁ ≤ 1");
        assert_eq!(d.m[0 * d.size + 3], 2, "stored 2 (Strong leaves 3)");
    }

    #[test]
    fn test_closure_integer_exact_propagates() {
        let mk = || {
            let mut d = Dbm::new(2);
            d.set_mirrored(0, 1, 5); // 2x ≤ 5    (x ≤ 2.5)
            d.set_mirrored(2, 0, 0); // y − x ≤ 0 (y ≤ x)
            d.set_mirrored(3, 2, -5); // −2y ≤ −5 (y ≥ 2.5)
            d
        };
        let mut a = mk();
        assert!(a.close_with(ClosureMode::Strong), "real model x = y = 2.5");
        let mut b = mk();
        assert!(
            !b.close_with(ClosureMode::IntegerExact),
            "x ≤ 2 ∧ y ≤ x ∧ y ≥ 3 — the tightened bounds contradict"
        );
    }

    #[test]
    fn test_closure_integer_exact_negative_half() {
        let mk = || {
            let mut d = Dbm::new(1);
            d.set_mirrored(0, 1, -5); // 2x ≤ −5
            d.set_mirrored(1, 0, 5); // −2x ≤ 5 (2x ≥ −5)
            d
        };
        let mut a = mk();
        assert!(a.close_with(ClosureMode::Strong), "real model x = −2.5");
        let mut b = mk();
        assert!(
            !b.close_with(ClosureMode::IntegerExact),
            "2x even: 2x ≤ −6 ∧ 2x ≥ −4"
        );
    }

    #[test]
    fn test_closure_integer_exact_noop_on_this_clean_input() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 10); // 2x ≤ 10 (x ≤ 5)
        a.set_mirrored(1, 0, -6); // 2x ≥ 6  (x ≥ 3)
        a.set_mirrored(2, 3, 8); // 2y ≤ 8  (y ≤ 4)
        a.set_mirrored(0, 2, 2); // x − y ≤ 2
        let mut b = a.clone();
        assert!(a.close_with(ClosureMode::Strong));
        assert!(b.close_with(ClosureMode::IntegerExact));
        assert_eq!(a, b, "integer-clean input: no self-dual entry is roundable");
    }

    #[test]
    fn test_closure_integer_exact_stored_rounding() {
        let mut a = Dbm::new(1);
        a.set_mirrored(0, 1, 5); // 2x ≤ 5
        assert!(a.close_with(ClosureMode::Strong));
        assert_eq!(a.m[0 * 2 + 1], 10, "Strong: the real bound x ≤ 2.5");
        let mut b = Dbm::new(1);
        b.set_mirrored(0, 1, 5);
        assert!(b.close_with(ClosureMode::IntegerExact));
        assert_eq!(b.m[0 * 2 + 1], 8, "IntegerExact: 2x even ⟹ x ≤ 2");
        assert_eq!(b.var_ub(0), Some(2));
    }

    // ------------------------------------------------------------------
    // Tests for the NEW features
    // ------------------------------------------------------------------

    /// Test `test_sum_le`: X + Y ≤ c guard.
    #[test]
    fn test_sum_le_guard() {
        let d = Dbm::new(2);
        let r = d.test_sum_le(0, 1, 10);
        assert!(!r.bottom);
        // The edge (0, 3) should carry X⁺ − Y⁻ ≤ 10, stored 20.
        assert_eq!(r.m[0 * r.size + 3], 20);
        assert_eq!(r.sum_ub(0, 1), Some(10));
    }

    /// Test `test_neg_sum_le`: −X − Y ≤ c guard.
    #[test]
    fn test_neg_sum_le_guard() {
        let d = Dbm::new(2);
        // −X − Y ≤ −5  ⟺  X + Y ≥ 5
        let r = d.test_neg_sum_le(0, 1, -5);
        assert!(!r.bottom);
        // Edge (2*1+1, 2*0) = (3, 0) carries Y⁻ − X⁺ ≤ −5, stored −10.
        assert_eq!(r.m[3 * r.size + 0], -10);
        // The mirror (2*0+1, 2*1) = (1, 2) carries X⁻ − Y⁺ ≤ −5, stored −10.
        assert_eq!(r.m[1 * r.size + 2], -10);
    }

    /// Test `test_sum_eq`: X + Y = c decomposes into two half-guards.
    #[test]
    fn test_sum_eq_guard() {
        let d = Dbm::new(2);
        let r = d.test_sum_eq(0, 1, 7);
        assert!(!r.bottom);
        // After X + Y = 7:
        //   X + Y ≤ 7  →  sum_ub(0,1) = Some(7)
        //   −X − Y ≤ −7  →  X + Y ≥ 7
        // Combined: X + Y is exactly 7 (within octagon precision).
        assert_eq!(r.sum_ub(0, 1), Some(7));
        // The negated sum gives a lower bound on X+Y. We can verify
        // via the edge (1, 2): X⁻ − Y⁺ ≤ −7, which means −X − Y ≤ −7.
        assert!(r.m[1 * r.size + 2] <= -14); // stored −2×7 = −14
    }

    /// Test `test_diff_eq`: X − Y = c.
    #[test]
    fn test_diff_eq_guard() {
        let d = Dbm::new(2);
        let r = d.test_diff_eq(0, 1, 3);
        assert!(!r.bottom);
        // X − Y ≤ 3 and Y − X ≤ −3 → X − Y = 3.
        assert_eq!(r.diff_bound(0, 1), Some(3));
        // The reverse: Y − X ≤ −3.
        assert_eq!(r.diff_bound(1, 0), Some(-3));
    }

    /// Test `test_var_eq`: X = c.
    #[test]
    fn test_var_eq_guard() {
        let d = Dbm::new(1);
        let r = d.test_var_eq(0, 42);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), Some(42));
        assert_eq!(r.var_lb(0), Some(42));
    }

    /// Test `assign_copy_add_var`: X := Y + c.
    #[test]
    fn test_assign_copy_add_var() {
        // Start with Y ≤ 10, Y ≥ 0.
        let mut d = Dbm::new(2);
        d.set_mirrored(2, 3, 10); // 2Y ≤ 10 → Y ≤ 5... wait, need to be careful.
        // Actually: set_mirrored(2, 3, c) sets edge (2,3) = Y⁺ − Y⁻ ≤ c.
        // This encodes 2Y ≤ c, so Y ≤ c/2.
        // For Y ≤ 10: set_mirrored(2, 3, 20) → 2Y ≤ 20 → Y ≤ 10.
        let mut d = Dbm::new(2);
        d.set_mirrored(2, 3, 20); // Y ≤ 10
        d.set_mirrored(3, 2, 0); // Y ≥ 0
        assert!(d.close());

        // X := Y + 5
        let r = d.assign_copy_add_var(0, 1, 5);
        assert!(!r.bottom);
        // X − Y ≤ 5 and Y − X ≤ −5 → X = Y + 5.
        assert_eq!(r.diff_bound(0, 1), Some(5));
        assert_eq!(r.diff_bound(1, 0), Some(-5));
        // X should be in [5, 15] (Y in [0, 10], X = Y + 5).
        // After closure, the bounds should propagate.
        assert!(r.var_ub(0).map_or(true, |v| v <= 15));
        assert!(r.var_lb(0).map_or(true, |v| v >= 5));
    }

    /// Test `assign_copy_add_var` with c = 0 reduces to pure copy.
    #[test]
    fn test_assign_copy_add_var_zero_is_copy() {
        let mut d = Dbm::new(2);
        d.set_mirrored(2, 3, 8); // Y ≤ 4
        d.set_mirrored(3, 2, -4); // Y ≥ 2
        assert!(d.close());

        let r1 = d.assign_copy_var(0, 1);
        let r2 = d.assign_copy_add_var(0, 1, 0);
        assert_eq!(r1, r2, "X := Y should equal X := Y + 0");
    }

    /// Test `assign_expr_var` with a simple constant expression.
    #[test]
    fn test_assign_expr_const() {
        let d = Dbm::new(1);
        let r = d.assign_expr_var(0, &OctExpr::Const(7));
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), Some(7));
        assert_eq!(r.var_lb(0), Some(7));
    }

    /// Test `assign_expr_var` with a variable expression.
    #[test]
    fn test_assign_expr_var_copy() {
        let mut d = Dbm::new(2);
        d.set_mirrored(2, 3, 10); // Y ≤ 5
        d.set_mirrored(3, 2, -6); // Y ≥ 3
        assert!(d.close());

        // X := Y (via expression)
        let r = d.assign_expr_var(0, &OctExpr::Var(1));
        assert!(!r.bottom);
        // X should have the same interval as Y: [3, 5]
        assert_eq!(r.var_ub(0), Some(5));
        assert_eq!(r.var_lb(0), Some(3));
    }

    /// Test `assign_expr_var` with addition.
    #[test]
    fn test_assign_expr_add() {
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 4); // X ≤ 2
        d.set_mirrored(1, 0, 0); // X ≥ 0
        d.set_mirrored(2, 3, 6); // Y ≤ 3
        d.set_mirrored(3, 2, -2); // Y ≥ 1
        assert!(d.close());

        // Z := X + Y (variable 0 gets overwritten)
        // X ∈ [0, 2], Y ∈ [1, 3] → X + Y ∈ [1, 5]
        let expr = OctExpr::Var(0).add(OctExpr::Var(1));
        let r = d.assign_expr_var(0, &expr);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), Some(5));
        assert_eq!(r.var_lb(0), Some(1));
    }

    /// Test `assign_expr_var` with negation and scaling.
    #[test]
    fn test_assign_expr_neg_scale() {
        let mut d = Dbm::new(1);
        d.set_mirrored(0, 1, 8); // X ≤ 4
        d.set_mirrored(1, 0, -2); // X ≥ 1
        assert!(d.close());

        // X := −2 * X
        // X ∈ [1, 4] → −2X ∈ [−8, −2]
        let expr = OctExpr::Scale(-2, Box::new(OctExpr::Var(0)));
        let r = d.assign_expr_var(0, &expr);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), Some(-2));
        assert_eq!(r.var_lb(0), Some(-8));
    }

    /// Test boolean combinators: AND.
    #[test]
    fn test_guard_and() {
        let d = Dbm::new(1);
        // g1: X ≤ 5, g2: X ≥ 2
        let g1 = d.test_le_var(0, 5);
        let g2 = d.test_ge_var(0, 2);
        let combined = g1.guard_and(&g2);
        assert!(!combined.bottom);
        assert_eq!(combined.var_ub(0), Some(5));
        assert_eq!(combined.var_lb(0), Some(2));
    }

    /// Test boolean combinators: AND producing contradiction.
    #[test]
    fn test_guard_and_contradiction() {
        let d = Dbm::new(1);
        // g1: X ≤ 2, g2: X ≥ 5 → contradiction
        let g1 = d.test_le_var(0, 2);
        let g2 = d.test_ge_var(0, 5);
        let combined = g1.guard_and(&g2);
        assert!(combined.bottom, "X ≤ 2 ∧ X ≥ 5 is unsatisfiable");
    }

    /// Test boolean combinators: OR.
    #[test]
    fn test_guard_or() {
        let d = Dbm::new(1);
        // g1: X ≤ 3, g2: X ≥ 7
        let g1 = d.test_le_var(0, 3);
        let g2 = d.test_ge_var(0, 7);
        let combined = g1.guard_or(&g2);
        assert!(!combined.bottom);
        // The join of X ≤ 3 and X ≥ 7 is an over-approximation.
        // The join takes the max of bounds, so:
        // upper bound: max(3's stored, ∞) = ∞ (g2 has no upper bound)
        // Actually g2: X ≥ 7 sets the lower edge but leaves upper as ∞.
        // g1: X ≤ 3 sets upper edge but leaves lower as ∞.
        // join: upper = max(3_stored, ∞) = ∞, lower = max(∞, 7_stored) = ∞
        // So the result is essentially top for this variable.
        // This is expected: the union of (-∞,3] ∪ [7,∞) is not an octagon.
    }

    /// Test boolean combinators: NOT for ≤ guard.
    #[test]
    fn test_guard_not_le() {
        let d = Dbm::new(1);
        // ¬(X ≤ 3) = X ≥ 4 (integer semantics)
        let r = d.guard_not_le_var(0, 3);
        assert!(!r.bottom);
        assert_eq!(r.var_lb(0), Some(4));
        assert_eq!(r.var_ub(0), None); // no upper bound
    }

    /// Test boolean combinators: NOT for ≥ guard.
    #[test]
    fn test_guard_not_ge() {
        let d = Dbm::new(1);
        // ¬(X ≥ 5) = X ≤ 4 (integer semantics)
        let r = d.guard_not_ge_var(0, 5);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), Some(4));
        assert_eq!(r.var_lb(0), None); // no lower bound
    }

    /// Test boolean combinators: NOT for sum guard.
    #[test]
    fn test_guard_not_sum_le() {
        let d = Dbm::new(2);
        // ¬(X + Y ≤ 5) = X + Y ≥ 6 = −X − Y ≤ −6
        let r = d.guard_not_sum_le(0, 1, 5);
        assert!(!r.bottom);
        // The edge encoding −X − Y ≤ −6 should be present.
        // Edge (2*1+1, 2*0) = (3, 0) carries Y⁻ − X⁺ ≤ −6, stored −12.
        assert!(r.m[3 * r.size + 0] <= -12);
    }

    /// Test boolean combinators: NOT for difference guard.
    #[test]
    fn test_guard_not_diff_le() {
        let d = Dbm::new(2);
        // ¬(X − Y ≤ 3) = Y − X ≤ −4
        let r = d.guard_not_diff_le(0, 1, 3);
        assert!(!r.bottom);
        // Y − X ≤ −4: edge (2*1, 2*0) = (2, 0) with bound −4, stored −8.
        assert!(r.m[2 * r.size + 0] <= -8);
    }

    // ------------------------------------------------------------------
    // Differential regression pins (from original file)
    // ------------------------------------------------------------------

    fn reference_join(a: &Dbm, b: &Dbm) -> Dbm {
        let size = a.size;
        let mut m = Vec::with_capacity(size * size);
        for idx in 0..size * size {
            let i = idx / size;
            let j = idx % size;
            let (x, y) = (a.m[idx], b.m[idx]);
            m.push(if i == j { x.min(y) } else { x.max(y) });
        }
        let mut r = Dbm {
            size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        r
    }

    #[test]
    fn test_join_differential_no_close() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 4);
        a.set_mirrored(1, 0, -2);
        a.set_mirrored(0, 2, 3);
        assert!(a.close());
        let mut b = Dbm::new(2);
        b.set_mirrored(2, 3, 2);
        b.set_mirrored(0, 2, 1);
        assert!(b.close());
        let mut c1 = Dbm::new(2);
        c1.set_mirrored(0, 1, 1);
        c1.set_mirrored(2, 3, 2);
        assert!(c1.close());
        let mut d3a = Dbm::new(3);
        d3a.set_mirrored(0, 2, 4);
        d3a.set_mirrored(2, 4, 2);
        d3a.set_mirrored(0, 1, 3);
        assert!(d3a.close());
        let mut d3b = Dbm::new(3);
        d3b.set_mirrored(4, 0, -2);
        assert!(d3b.close());
        let top2 = Dbm::new(2);
        for (x, y) in [(&a, &b), (&c1, &b), (&d3a, &d3b), (&top2, &a)] {
            let j1 = x.join(y);
            let j2 = y.join(x);
            let slow = reference_join(x, y);
            assert_eq!(j1, slow, "differential mismatch");
            assert_eq!(j1, j2, "join must be symmetric");
            assert!(!j1.bottom);
            assert!(j1.is_strongly_closed());
        }
    }

    fn reference_assign_add(a: &Dbm, i: usize, c: i128) -> Dbm {
        let p = 2 * i;
        let q = 2 * i + 1;
        let mut m = a.m.clone();
        let delta = sat_mul2(c);
        for j in 0..a.size {
            m[p * a.size + j] = sat_add(m[p * a.size + j], delta);
            m[j * a.size + p] = sat_sub(m[j * a.size + p], delta);
            m[q * a.size + j] = sat_sub(m[q * a.size + j], delta);
            m[j * a.size + q] = sat_add(m[j * a.size + q], delta);
        }
        let mut r = Dbm {
            size: a.size,
            m,
            bottom: false,
            closed: true,
        };
        r.close();
        r
    }

    #[test]
    fn test_assign_add_differential_no_close() {
        let mut a = Dbm::new(3);
        a.set_mirrored(0, 1, 4);
        a.set_mirrored(1, 0, -2);
        a.set_mirrored(0, 2, 3);
        a.set_mirrored(2, 4, 2);
        assert!(a.close());
        for &(i, c) in &[(0usize, -3i128), (0, 5), (1, 1), (2, -1)] {
            let fast = a.assign_add_var(i, c);
            let slow = reference_assign_add(&a, i, c);
            assert_eq!(fast, slow, "differential mismatch at (i={}, c={})", i, c);
            assert!(!fast.bottom);
            assert!(fast.is_strongly_closed());
        }
        for &(i, c) in &[(0usize, DBM_INF / 4 - 1), (0, -(DBM_INF / 4 - 1))] {
            let fast = a.assign_add_var(i, c);
            let slow = reference_assign_add(&a, i, c);
            assert_eq!(
                fast, slow,
                "large finite-delta mismatch at (i={}, c={})",
                i, c
            );
            assert!(!fast.bottom);
            assert!(fast.is_strongly_closed());
        }
        for &(i, c) in &[(0usize, i128::MAX / 4), (0, -(1i128 << 126))] {
            let r = a.assign_add_var(i, c);
            assert!(
                !r.bottom,
                "saturation corner (i={}, c={}) must not be bottom",
                i, c
            );
            assert_eq!(r.var_ub(i), None, "saturation corner (i={}, c={})", i, c);
            assert_eq!(r.var_lb(i), None, "saturation corner (i={}, c={})", i, c);
            assert!(r.is_strongly_closed());
        }
        let mut h = Dbm::new(2);
        h.set_mirrored(0, 1, 1);
        h.set_mirrored(2, 3, 2);
        assert!(h.close());
        assert_eq!(h.assign_add_var(0, 1), reference_assign_add(&h, 0, 1));
        assert!(h.assign_add_var(0, 1).is_strongly_closed());
    }

    #[test]
    fn test_is_strongly_closed_tripwire() {
        assert!(Dbm::new(2).is_strongly_closed());
        assert!(Dbm::bottom().is_strongly_closed());
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 4);
        d.set_mirrored(0, 2, 3);
        assert!(d.close());
        assert!(d.is_strongly_closed());
        d.set(0, 2, 1);
        assert!(!d.is_strongly_closed());
    }

    #[test]
    fn test_join_preserves_rounded_normal_form() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 5);
        a.set_mirrored(1, 0, -2);
        assert!(a.close_with(ClosureMode::IntegerExact));
        let mut b = Dbm::new(2);
        b.set_mirrored(0, 1, 3);
        b.set_mirrored(2, 3, 4);
        assert!(b.close_with(ClosureMode::IntegerExact));
        let j = a.join(&b);
        assert!(!j.bottom);
        for v in 0..j.size / 2 {
            for e in [2 * v, 2 * v + 1] {
                let s = j.m[e * j.size + (e ^ 1)];
                if s != DBM_INF {
                    assert_eq!(s & 3, 0, "self-dual bound must stay ≡ 0 (mod 4)");
                }
            }
        }
    }

    #[test]
    fn test_integer_exact_output_is_tightly_closed() {
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 5);
        d.set_mirrored(1, 0, -3);
        d.set_mirrored(2, 3, 7);
        d.set_mirrored(0, 2, 1);
        assert!(d.close_with(ClosureMode::IntegerExact));
        assert!(d.is_strongly_closed());
        for v in 0..d.size / 2 {
            for e in [2 * v, 2 * v + 1] {
                let s = d.m[e * d.size + (e ^ 1)];
                if s != DBM_INF {
                    assert_eq!((s & 3), 0, "self-dual bound must stay ≡ 0 (mod 4)");
                }
            }
        }
    }

    #[test]
    fn test_overflow_and_edge_cases() {
        let d = Dbm::new(1);
        let max_var = DBM_INF / 4 - 1;
        let r = d.assign_const_var(0, max_var);
        assert!(
            !r.bottom,
            "assign_const_var should not collapse to bottom on large valid constants"
        );
        assert_eq!(r.var_ub(0), Some(max_var));
        assert_eq!(r.var_lb(0), Some(max_var));
        let too_big = DBM_INF / 4 + 1;
        let r = Dbm::new(1).assign_const_var(0, too_big);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), None);
        assert_eq!(r.var_lb(0), None);
        let too_neg = -(DBM_INF / 4) - 1;
        let r = Dbm::new(1).assign_const_var(0, too_neg);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), None);
        assert_eq!(r.var_lb(0), None);
        let d = Dbm::new(1);
        let c = (i128::MAX / 2) + 1;
        let r = d.test_le_var(0, c);
        assert!(!r.bottom);
        assert_eq!(r.var_ub(0), None);
        let d = Dbm::new(1);
        let c_neg = i128::MIN / 2;
        let r = d.test_ge_var(0, c_neg);
        assert!(!r.bottom);
        assert_eq!(r.var_lb(0), None);
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 10);
        d.set_mirrored(2, 3, 4);
        d.close();
        let original = d.clone();
        let copied = d.assign_copy_var(0, 0);
        assert_eq!(
            original, copied,
            "assign_copy_var(i, i) must be a no-op identity"
        );
    }

    #[test]
    fn test_assign_add_var_saturation_closes() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 2);
        a.set_mirrored(1, 2, 2);
        a.set_mirrored(0, 2, 4);
        assert!(a.close());
        assert_eq!(a.m[0 * a.size + 2], 8);
        assert!(a.is_strongly_closed());
        let c = (DBM_INF - 7) / 2;
        let r = a.assign_add_var(0, c);
        assert!(!r.bottom);
        assert!(r.is_strongly_closed(), "saturated update must stay closed");
        assert_eq!(
            r.m[0 * r.size + 2],
            DBM_INF,
            "clipped edge stays a sound weakening"
        );
        assert_eq!(r.m[3 * r.size + 1], DBM_INF, "mirror clipped coherently");
    }

    #[test]
    fn test_assign_add_var_saturation_repairs_loose_input() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 2);
        a.set_mirrored(0, 2, 1);
        a.set_mirrored(2, 3, 2);
        a.set_mirrored(0, 3, 5);
        assert!(!a.is_strongly_closed(), "premise: loose input");
        let c = (1i128 << 124) - 4;
        let r = a.assign_add_var(0, c);
        assert!(!r.bottom);
        assert!(
            r.is_strongly_closed(),
            "fallback close must repair the loose input"
        );
        assert_eq!(
            r.m[0 * r.size + 3],
            (1i128 << 125) - 2,
            "direct edge re-closed to the surviving path bound"
        );
        assert_eq!(
            r.m[2 * r.size + 1],
            (1i128 << 125) - 2,
            "mirror edge recovered coherently"
        );
    }

    #[test]
    fn test_var_lb_saturated_lower_edge_is_none() {
        let mut d = Dbm::new(2);
        d.m[1 * d.size + 2] = -(1i128 << 126);
        d.m[3 * d.size + 0] = -(1i128 << 126);
        d.m[2 * d.size + 0] = -(1i128 << 126);
        d.m[1 * d.size + 3] = -(1i128 << 126);
        assert!(d.close(), "constraints are satisfiable (no negative cycle)");
        assert!(!d.bottom);
        assert!(
            d.is_strongly_closed(),
            "the skipped i128::MIN sum must not break strong closure"
        );
        assert_eq!(
            d.m[1 * d.size + 0],
            i128::MIN + 4,
            "closure clamps the unrepresentable i128::MIN path sum to MIN + 4"
        );
        assert_eq!(
            d.var_lb(0),
            None,
            "saturated lower bound projects to ∞, not Some(DBM_INF)"
        );
        assert_eq!(d.var_ub(0), None);
        let mut e = Dbm::new(1);
        e.m[1 * 2 + 0] = i128::MIN + 4;
        assert_eq!(
            e.var_lb(0),
            None,
            "DBM_INF sentinel must not leak into Some(..)"
        );
    }

    #[test]
    fn test_close_detects_extreme_negative_cycle() {
        let mut d = Dbm::new(1);
        d.set_mirrored(0, 1, -(1i128 << 125));
        d.set_mirrored(1, 0, -(1i128 << 125));
        assert!(
            !d.close(),
            "X ≤ −2^124 ∧ X ≥ 2^124 is unsatisfiable — must be bottom"
        );
        assert!(d.bottom);
    }

    #[test]
    fn test_guard_unrepresentable_bound_is_sound() {
        let r = Dbm::new(1).test_le_var(0, -(1i128 << 125));
        assert!(
            !r.bottom,
            "the guard's concrete post is non-empty — bottom would be unsound"
        );
        assert_eq!(
            r.var_ub(0),
            None,
            "unrepresentable bound drops to no bound (sound)"
        );
        let r = Dbm::new(1).test_ge_var(0, 1i128 << 125);
        assert!(!r.bottom);
        assert_eq!(r.var_lb(0), None);
        let r = Dbm::new(2).test_diff_le(0, 1, -(1i128 << 126));
        assert!(!r.bottom);
    }

    #[test]
    fn test_close_idempotent_on_corner_matrices() {
        let cases: Vec<(usize, Vec<(usize, usize, i128)>)> = vec![
            (
                2,
                vec![(0, 1, -(1i128 << 125) - 1), (1, 2, -(1i128 << 125))],
            ),
            (
                2,
                vec![(0, 1, -(1i128 << 125) + 1), (1, 2, -(1i128 << 125))],
            ),
            (
                2,
                vec![(0, 2, -(1i128 << 125) + 1), (2, 1, -(1i128 << 125))],
            ),
            (
                2,
                vec![
                    (0, 1, -(1i128 << 126) + 2),
                    (0, 2, -(1i128 << 125)),
                    (2, 1, -(1i128 << 125)),
                    (2, 3, 1),
                ],
            ),
            (
                3,
                vec![
                    (0, 4, -(1i128 << 125)),
                    (4, 5, -(1i128 << 125)),
                    (5, 2, (DBM_INF - 1) / 2),
                ],
            ),
        ];
        for (n_vars, edges) in cases {
            let mut d = Dbm::new(n_vars);
            for &(a, b, c) in &edges {
                d.set_mirrored(a, b, c);
            }
            if d.close() {
                assert!(
                    d.is_strongly_closed(),
                    "close output must pass the checker (corner matrix)"
                );
                let mut d2 = d.clone();
                assert!(d2.close());
                assert_eq!(d, d2, "close must be idempotent on corner matrices");
            } else {
                assert!(d.bottom, "failed close must be bottom");
            }
        }
    }

    #[test]
    fn test_integer_exact_converges_on_corner_input() {
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 2, -(1i128 << 125) + 1);
        d.set_mirrored(2, 1, -(1i128 << 125));
        assert!(
            d.close_with(ClosureMode::IntegerExact),
            "satisfiable corner input must converge to a non-bottom fixpoint"
        );
        assert!(!d.bottom);
    }

    #[test]
    fn test_join_stays_closed_on_corner_inputs() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, -(1i128 << 125) - 1);
        a.set_mirrored(1, 2, -(1i128 << 125));
        assert!(a.close() && a.is_strongly_closed());
        let mut b = Dbm::new(2);
        b.set_mirrored(0, 1, -(1i128 << 125) + 1);
        b.set_mirrored(1, 2, -(1i128 << 125));
        assert!(b.close() && b.is_strongly_closed());
        let m = a.join(&b);
        assert!(!m.bottom);
        assert!(
            m.is_strongly_closed(),
            "join of closed corner matrices must stay closed"
        );
        let mut m2 = m.clone();
        assert!(m2.close());
        assert_eq!(m, m2, "join output is already closed (close is a no-op)");
    }

    #[test]
    fn test_checker_pass_implies_close_noop_on_corner() {
        let mut p = Dbm::new(2);
        p.set_mirrored(0, 1, -(1i128 << 126) + 2);
        p.set_mirrored(0, 2, -(1i128 << 125));
        p.set_mirrored(2, 1, -(1i128 << 125));
        p.set_mirrored(2, 3, 1);
        assert!(
            p.is_strongly_closed(),
            "checker must accept the corner fixpoint"
        );
        let mut q = p.clone();
        assert!(q.close());
        assert_eq!(p, q, "a checker-passing matrix is a true fixpoint");
    }

    #[test]
    fn test_assign_add_corner_scan_stays_closed() {
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, -(1i128 << 125));
        a.set_mirrored(1, 2, -(1i128 << 125));
        assert!(a.close());
        assert!(a.is_strongly_closed());
        let r = a.assign_add_var(0, 1);
        assert!(!r.bottom);
        assert!(
            r.is_strongly_closed(),
            "corner-regime assign_add_var must stay strongly closed"
        );
    }
}
