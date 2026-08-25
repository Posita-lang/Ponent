//! The octagon abstract domain (difference-bound matrices) — extracted
//! from `type_eq.rs` to keep the equality-checking module focused.
//!
//! A difference-bound matrix over **2n nodes** — `Xᵢ⁺ = 2i` (the variable)
//! and `Xᵢ⁻ = 2i+1` (its negation) — with **strong closure** ([Miné06]
//! Figure 8), meet/join/widen, and loop-transfer functions. There is NO
//! implicit zero node: single-variable bounds live on the **self-dual
//! edges** `m[2i][2i+1]` (`2Xᵢ ≤ 2c`, i.e. `Xᵢ ≤ c`) and `m[2i+1][2i]`
//! (`−2Xᵢ ≤ −2c`, i.e. `Xᵢ ≥ c`); the mirror of node `i` is `i ⊕ 1`, and
//! the mirror of edge `(i, j)` is `(j⊕1, i⊕1)`.
//!
//! All stored bounds are **doubled** (2×c) to represent half-integers
//! exactly. Interval (self-dual) rows therefore carry 4× the interval
//! half-width in raw storage — readers must go through the semantic
//! projections (`var_ub` / `var_lb` / `diff_bound` / `sum_ub`), never
//! raw cells. External API accepts plain `c`, internal storage uses the
//! doubled space.
//!
//! # References
//!
//! Citations in this module use the following shorthand:
//!
//! - `[Miné06]` — Antoine Miné, "The Octagon Abstract Domain",
//!   Higher-Order and Symbolic Computation 19(1), 2006. The primary
//!   source for the 2n-node DBM representation, strong closure
//!   (Figure 8), coherence, join/widen, and the loop-transfer
//!   functions. ("The octagon paper" below always means this one.)
//! - `[Miné01]` — Antoine Miné, "A New Numerical Abstract Domain Based
//!   on Difference-Bound Matrices", PADO II, LNCS 2053, 2001. The
//!   earlier DBM-based domain that `[Miné06]` extends; cited here for
//!   historical context, not for a specific algorithm used in this
//!   module.
//! - `[HS97]` — Warwick Harvey and Peter J. Stuckey, "A Unit Two
//!   Variable Per Inequality Integer Constraint Solver for Constraint
//!   Logic Programming", ACSC 1997. Source of the integer-tightening
//!   step discussed in `[Miné06]` §V.D.

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

/// Saturating multiplication by 2 (used when converting a constraint constant `c` to a stored value `2c`)
/// - If `c == DBM_INF`, returns `DBM_INF`
/// - If `c` is positive and overflows, returns `DBM_INF`
/// - If `c` is negative and overflows, returns `i128::MIN` (representing negative infinity)
/// - Otherwise returns `2*c`
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

/// Saturating multiplication by 2 for **bound insertion** (`set` /
/// `set_mirrored`). Unlike `sat_mul2`, negative overflow weakens to
/// `DBM_INF` (no bound) instead of `i128::MIN`: `i128::MIN` is reserved
/// as the negative-infinity marker in `sat_sub`/`sat_neg`, so storing it
/// as an edge bound would let the same cell value mean a finite bound in
/// `node_bound` and `-∞` in `sat_sub` — an unsound collision. Weakening
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

/// Compute the mirror index for coherence: node `i` mirrors to `i ⊕ 1`
/// (`Xᵢ⁺ = 2i ↔ Xᵢ⁻ = 2i+1`).
#[inline]
const fn mirror_index(i: usize) -> usize {
    i ^ 1
}

/// A difference-bound matrix over `2n` nodes: `Xᵢ⁺ = Xᵢ` and
/// `Xᵢ⁻ = -Xᵢ`. `m[i][j]` encodes `node_i - node_j ≤ c` **in doubled
/// space** (`m[i][j] = 2*c`). ([Miné06]'s convention is the transpose —
/// `m⁺ᵢⱼ` bounds `vⱼ − vᵢ`; Figure 8's operators are unchanged
/// under transposition.) Single-variable bounds hang on the
/// self-dual edges: `m[2i][2i+1]` carries `2Xᵢ ≤ 2c` (stored `4c`),
/// `m[2i+1][2i]` carries `-2Xᵢ ≤ -2c` (stored `-4c`).
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Dbm {
    /// `2 * n_vars` (no implicit zero node).
    pub(crate) size: usize,
    /// Flattened `size × size` matrix, row-major, stored in doubled space.
    pub(crate) m: Vec<i128>,
    /// `true` if the constraint system is unsatisfiable (⊥).
    pub(crate) bottom: bool,
}

/// Closure strategy — the Harvey–Stuckey integer-tightening profile
/// ([HS97]; see [Miné06] §V.D).
///
/// `Strong` is [Miné06]'s Figure 8 strong closure — sound over reals,
/// rationals AND integers (an over-approximation for the latter).
///
/// `IntegerExact` additionally interleaves the HS tightening step
/// ([Miné06] §V.D: `2x ≤ 2c+1 ⟹ 2x ≤ 2c` — knowing x is an integer, the
/// self-dual edge constants round down) with re-closure until the
/// closed-and-rounded fixpoint, at O(N⁴) cost (Strong alone is O(N³)).
/// SOUND ONLY over INTEGER domains: over rationals the rounding
/// discards solutions (`2x ≤ 5` admits x = 2.5; the tightened
/// `2x ≤ 4` does not). Wired at the exact-discharge surface that
/// reasons over integers by construction: the BII inductiveness
/// verifier (`dbm_proves_inductiveness` in `bii.rs`, LIA mode only).
/// The DBM fixpoint and the type-equality closure stay on `Strong`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ClosureMode {
    Strong,
    IntegerExact,
}

impl Dbm {
    /// Create a new, unconstrained (top) matrix.
    pub(crate) fn new(n_vars: usize) -> Dbm {
        let size = 2 * n_vars;
        let mut m = vec![DBM_INF; size * size];
        for i in 0..size {
            m[i * size + i] = 0;
        }
        Dbm {
            size,
            m,
            bottom: false,
        }
    }

    /// The empty (bottom) matrix.
    pub(crate) fn bottom() -> Dbm {
        Dbm {
            size: 0,
            m: Vec::new(),
            bottom: true,
        }
    }

    /// Flattened index of `m[i][j]` (row-major).
    #[inline]
    fn ix(&self, i: usize, j: usize) -> usize {
        i * self.size + j
    }

    /// Tighten `node_i - node_j ≤ c` **without** mirroring (internal use only).
    /// `c` is the **actual** bound; stored as `2*c`.
    pub(crate) fn set(&mut self, i: usize, j: usize, c: i128) {
        if self.bottom {
            return;
        }
        let stored = sat_mul2_bound(c);
        let idx = self.ix(i, j);
        if stored < self.m[idx] {
            self.m[idx] = stored;
        }
    }

    /// Set a bound and its mirror to maintain coherence.
    /// For constraint `node_i - node_j ≤ c`, also set `node_j̄ - node_ī ≤ c`.
    pub(crate) fn set_mirrored(&mut self, i: usize, j: usize, c: i128) {
        if self.bottom {
            return;
        }
        Self::set_mirrored_internal(&mut self.m, i, j, c, self.size);
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
    pub(crate) fn close(&mut self) -> bool {
        self.close_with(ClosureMode::Strong)
    }

    /// One pass of the strong closure, adapted from [Miné06] Figure 8
    /// (all-pivot variant — see the body comment on even-pivot
    /// equivalence): the C⁺_k / S⁺ interleaving over all k, in place.
    /// Extracted from `close_with`
    /// so `IntegerExact` can re-run it between Harvey–Stuckey
    /// tightening rounds without duplicating the loop body.
    fn strong_closure_pass(m: &mut [i128], size: usize) {
        // [Miné06]'s loop applies C⁺ only at even pivots (its Figure 8:
        // S⁺(C⁺_{2k}(·))); C⁺_{k̄} has the same five terms as C⁺_k, so
        // iterating over all 2n nodes applies each pivot twice —
        // redundant work, same result.
        // k iterates over ALL 2n nodes. The mirror of every node is
        // `i ⊕ 1`; with no implicit zero node the single-variable bounds
        // hang on the self-dual edges m[2i][2i+1] / m[2i+1][2i], and the
        // five C⁺ terms below stay mirror-symmetric under `i ↦ i⊕1`.
        for k in 0..size {
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
            for i in 0..size {
                for j in 0..size {
                    let mut best = m[i * size + j];

                    let t_ik = m[i * size + k];
                    let t_kj = m[k * size + j];
                    if t_ik != DBM_INF && t_kj != DBM_INF {
                        let s = sat_add(t_ik, t_kj);
                        if s < best {
                            best = s;
                        }
                    }

                    let t_ikb = m[i * size + k_bar];
                    let t_kbj = m[k_bar * size + j];
                    if t_ikb != DBM_INF && t_kbj != DBM_INF {
                        let s = sat_add(t_ikb, t_kbj);
                        if s < best {
                            best = s;
                        }
                    }

                    let t_kkb = m[k * size + k_bar];
                    if t_ik != DBM_INF && t_kkb != DBM_INF && t_kbj != DBM_INF {
                        let s = sat_add(sat_add(t_ik, t_kkb), t_kbj);
                        if s < best {
                            best = s;
                        }
                    }

                    let t_kbk = m[k_bar * size + k];
                    if t_ikb != DBM_INF && t_kbk != DBM_INF && t_kj != DBM_INF {
                        let s = sat_add(sat_add(t_ikb, t_kbk), t_kj);
                        if s < best {
                            best = s;
                        }
                    }

                    m[i * size + j] = best;
                }
            }
            // ---- S⁺ ([Miné06] Figure 8): m[i][j] ≤ (m[i][ī] + m[j̄][j]) / 2.
            // In doubled storage the derived value is (t1+t2)/2 — e.g.
            // 2v₀ ≤ 1 ∧ 2v₁ ≤ 2 derives v₀+v₁ ≤ 1.5, stored 3. Odd sums
            // (two half-integer bounds compose
            // into a quarter) round UP — the sound direction for ≤. DBM_INF
            // is never halved (DBM_INF/2 is a finite value and would invent
            // a tightening); sum < DBM_INF guarantees sum+1 cannot overflow.
            for i in 0..size {
                let i_bar = mirror_index(i);
                for j in 0..size {
                    let j_bar = mirror_index(j);
                    let t1 = m[i * size + i_bar];
                    let t2 = m[j_bar * size + j];
                    if t1 != DBM_INF && t2 != DBM_INF {
                        let sum = sat_add(t1, t2);
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
                    m[idx] = (s >> 2) << 2; // floor to a multiple of 4
                    changed = true;
                }
            }
        }
        changed
    }

    /// Close under the given `ClosureMode`. See the enum docs for the
    /// integer-tightening profile.
    pub(crate) fn close_with(&mut self, mode: ClosureMode) -> bool {
        if self.bottom {
            return false;
        }
        let size = self.size;
        // Strong closure (all-pivot variant of [Miné06] Figure 8): one
        // pass of the C⁺_k / S⁺ interleaving over all k.
        Self::strong_closure_pass(&mut self.m, size);
        // ---- Harvey–Stuckey integer tightening ([HS97]; see [Miné06] §V.D) ----
        // `IntegerExact` interleaves the tightening around the closure
        // loops until a fixpoint (closed AND rounded — the normal form
        // Harvey–Stuckey build, O(N⁴)). The tightening rounds every
        // self-dual edge's stored bound down to a multiple of 4: the
        // edge carries `2x ≤ s/2`, and over INTEGERS 2x is even, so
        // the constraint is equivalent to `2x ≤ 4⌊s/4⌋` ([Miné06]:
        // `2x ≤ 2c+1 ⟹ 2x ≤ 2c`). Each constraint's integer solution
        // set is UNCHANGED by the rounding, so any prefix of the
        // iteration stays sound over integers — and unsound over
        // rationals (2x ≤ 5 admits x = 2.5; the tightened 2x ≤ 4 does
        // not): integer domains only. Rounds are capped defensively
        // (an engineering bound; stopping early forfeits only normal-form
        // tightness, never soundness).
        if mode == ClosureMode::IntegerExact {
            let max_rounds = size + 2;
            for _ in 0..max_rounds {
                if !Self::tighten_self_dual(&mut self.m, size) {
                    break; // stable: closed and rounded.
                }
                Self::strong_closure_pass(&mut self.m, size);
            }
        }
        let m = &mut self.m;
        // ---- Enforce coherence (defensive safety net) ----
        // This is not a repair mechanism; it ensures no internal bug
        // leaves the matrix asymmetric. All documented transfer functions
        // already maintain coherence explicitly via set_mirrored.
        for i in 0..size {
            for j in 0..size {
                let i_bar = mirror_index(i);
                let j_bar = mirror_index(j);
                let a = m[i * size + j];
                let b = m[j_bar * size + i_bar];
                if a < b {
                    m[j_bar * size + i_bar] = a;
                } else if b < a {
                    m[i * size + j] = b;
                }
            }
        }
        // ---- Diagonal unsatisfiability check ----
        for i in 0..size {
            let d = m[i * size + i];
            if d < 0 {
                self.bottom = true;
                return false;
            }
            m[i * size + i] = 0; // normalize
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
        }
    }

    // ---- Lattice operations ----

    /// meet (⊓): intersection – tighter of each bound, then close.
    /// If either operand is bottom, result is bottom.
    #[allow(dead_code)]
    pub(crate) fn meet(&self, other: &Dbm) -> Dbm {
        if self.bottom || other.bottom {
            return Dbm::bottom();
        }
        debug_assert_eq!(self.size, other.size);
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
                // 3. S⁺ half-sum (INF skips — no derived bound).
                let t1 = m[i * size + i_bar];
                let t2 = m[j_bar * size + j];
                if t1 != DBM_INF && t2 != DBM_INF {
                    let sum = sat_add(t1, t2);
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
                        let s = sat_add(t_ik, t_kj);
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
    /// tripwire for the preservation argument above — a failure means a
    /// lemma inside this module is wrong, and the fix belongs here, not
    /// in the caller.
    pub(crate) fn join(&self, other: &Dbm) -> Dbm {
        debug_assert!(
            self.is_strongly_closed() && other.is_strongly_closed(),
            "join requires strongly closed operands — a violation is a bug in a \
             closure-preservation lemma of this module, not in the caller"
        );
        if self.bottom {
            return other.clone();
        }
        if other.bottom {
            return self.clone();
        }
        debug_assert_eq!(self.size, other.size);
        let size = self.size;
        let m: Vec<i128> = self
            .m
            .iter()
            .zip(&other.m)
            .enumerate()
            .map(|(idx, (a, b))| {
                let i = idx / size;
                let j = idx % size;
                if i == j { (*a).min(*b) } else { (*a).max(*b) }
            })
            .collect();
        // Strongly closed ∨ strongly closed = strongly closed (see doc):
        // no closure pass, bottom unreachable.
        Dbm {
            size,
            m,
            bottom: false,
        }
    }

    /// widen (∇): termination-guaranteeing upper approximation.
    /// Definition ([Miné06]): if new bound is ≤ old, keep old; otherwise ∞.
    /// Does NOT call close after widening ([Miné06] §VI.D).
    pub(crate) fn widen(&self, new: &Dbm) -> Dbm {
        if self.bottom {
            return new.clone();
        }
        if new.bottom {
            return self.clone();
        }
        debug_assert_eq!(self.size, new.size);
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
        }
    }

    // ---- Transfer functions (coherent) ----

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
    /// Forget variable `i` (existential projection): drop its rows and
    /// columns, then re-close. Used as the sound fallback when a
    /// transfer function's result would be unrepresentable — the
    /// weakened (unconstrained) result is an over-approximation, never
    /// a spurious `bottom`.
    fn forget_var(&self, i: usize) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
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

        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
        };
        r.close();

        if r.bottom { Dbm::bottom() } else { r }
    }

    pub(crate) fn assign_add_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
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
        // The no-close path is exact potential reweighting,
        // m'[u][v] = m[u][v] + δ(σ_u − σ_v), which transports every
        // closure inequality unchanged. Saturation of an INDIVIDUAL
        // entry breaks that equality: a finite edge clipped to DBM_INF
        // is a weakening, so a still-finite path i→…→j can force a
        // tighter direct edge that the stored DBM_INF violates (triangle
        // closure lost); a clip to i128::MIN collides with the −∞
        // marker. Track both and fall back to a closing pass — the
        // clipped matrix is a point-wise over-approximation of the exact
        // reweighting, so closing it restores strong closure soundly.
        let mut saturated = false;
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
        // The true self-loop is always exactly 0; unconditional
        // normalization keeps the diagonal exact — the same discipline as
        // [Miné06] Fig. 7's `[C_k(n)]_ii ≜ 0`. On the finite-δ path the two
        // writes cancel exactly in checked arithmetic, so it cannot
        // over-trigger.
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
        };
        if saturated {
            // An entry clipped to a saturation sentinel breaks the exact
            // reweighting, so strong closure is no longer transported by
            // the potential-shift argument. Close the weakened matrix
            // (see loop comment); the diagonal normalization above means
            // bottom is unreachable, but keep the check for safety.
            r.close();
            if r.bottom {
                return Dbm::bottom();
            }
        }
        r
    }

    /// `X := c`
    pub(crate) fn assign_const_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }

        // Upper stored bound is 4c, lower stored bound is -4c. If either
        // would leave the finite range, the assignment is unrepresentable —
        // forget the variable (sound over-approximation) rather than encode
        // a one-sided bound that closure could treat as inconsistency.
        let upper = c.checked_mul(4);
        let lower = c.checked_mul(-4);

        let exact = matches!(upper, Some(v) if v < DBM_INF && v > i128::MIN)
            && matches!(lower, Some(v) if v < DBM_INF && v > i128::MIN);

        if !exact {
            return self.forget_var(i);
        }

        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let mut m = self.m.clone();
        // Drop X's own rows and columns.
        for j in 0..self.size {
            m[p * self.size + j] = DBM_INF;
            m[q * self.size + j] = DBM_INF;
            m[j * self.size + p] = DBM_INF;
            m[j * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        // Set X = c on the self-dual edge (no implicit zero node):
        // X ≤ c => X⁺ - X⁻ ≤ 2c (stored 4c)
        // X ≥ c => X⁻ - X⁺ ≤ -2c (stored -4c)
        // set_mirrored_internal mirrors a self-dual edge to itself.
        // With the exact-range check above, sat_mul2(c) and
        // sat_mul2(sat_neg(c)) are guaranteed finite (2c and -2c stay in
        // range), so the edges never saturate.
        let upper_actual = sat_mul2(c);
        let lower_actual = sat_mul2(sat_neg(c));
        Self::set_mirrored_internal(&mut m, p, q, upper_actual, self.size);
        Self::set_mirrored_internal(&mut m, q, p, lower_actual, self.size);
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// `X := Y`
    pub(crate) fn assign_copy_var(&self, i: usize, j: usize) -> Dbm {
        if i == j {
            return self.clone();
        }
        if self.bottom {
            return Dbm::bottom();
        }
        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let yp = 2 * j; // Y⁺
        let yq = 2 * j + 1; // Y⁻
        let mut m = self.m.clone();
        // Drop X's rows/columns.
        for k in 0..self.size {
            m[p * self.size + k] = DBM_INF;
            m[q * self.size + k] = DBM_INF;
            m[k * self.size + p] = DBM_INF;
            m[k * self.size + q] = DBM_INF;
        }
        m[p * self.size + p] = 0;
        m[q * self.size + q] = 0;
        // X = Y: coherently set both directions.
        // We need four constraints:
        // p - yp ≤ 0, yp - p ≤ 0, q - yq ≤ 0, yq - q ≤ 0
        // set_mirrored_internal(p, yp, 0) sets p->yp and its mirror
        // (yp̄, p̄) = (yq, q) → q->yq as well. The (yp, p) call covers
        // yp->p and (p̄, yp̄) = (q, yq) → yq->q.
        Self::set_mirrored_internal(&mut m, p, yp, 0, self.size);
        Self::set_mirrored_internal(&mut m, yp, p, 0, self.size);
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X ≤ c` (coherent) — self-dual edge `(2i, 2i+1)` carries
    /// `2X ≤ 2c` (stored `4c`).
    pub(crate) fn test_le_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * i + 1, sat_mul2(c));
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X ≥ c` (coherent) — self-dual edge `(2i+1, 2i)` carries
    /// `-2X ≤ -2c` (stored `-4c`).
    pub(crate) fn test_ge_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        let mut r = self.clone();
        r.set_mirrored(2 * i + 1, 2 * i, sat_mul2(sat_neg(c)));
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// Test `X - Y ≤ c` (coherent) — edge `(2i, 2j)` carries `2(X−Y) ≤ 2c`.
    pub(crate) fn test_diff_le(&self, i: usize, j: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        let mut r = self.clone();
        r.set_mirrored(2 * i, 2 * j, c);
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

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
    /// Strong closure can produce the self-dual cell
    /// `m[2i+1][2i] = i128::MIN`, which represents the lower-bound
    /// constraint `X⁻ - X⁺ ≤ -2^127` (i.e. `X ≥ 2^125` exactly).
    /// `node_bound` reads this as `-2^126`; then `sat_neg(-2^126)`
    /// attempts to compute `+2^126`, which is above the finite
    /// representable range and therefore saturates to `DBM_INF`.
    /// `DBM_INF` is the module's "no finite bound" sentinel; returning
    /// `Some(DBM_INF)` would be both wrong and ambiguous. The current
    /// implementation maps that clipped value back to `None` (∞), the
    /// sound over-approximation for an unrepresentable lower bound.
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
        // ⊥ is the identity of ⊔ ([Miné06] §VII.A): join(⊥, ⊤) = ⊤.
        // dbm_fixpoint relies on this semantics to handle infeasible
        // bodies (join(cur, ⊥) = cur).
        let j = d.join(&Dbm::new(1));
        assert!(!j.bottom);
        let m = d.meet(&Dbm::new(1));
        assert!(m.bottom);
    }

    #[test]
    fn test_widen_no_close() {
        // cur: X ≤ 0, next: X ≤ 1 (relaxed) → widen should set to ∞
        let mut cur = Dbm::new(1);
        cur.set(0, 1, 0);
        cur.close();
        let mut next = Dbm::new(1);
        next.set(0, 1, 2);
        next.close();
        let w = cur.widen(&next);
        assert_eq!(w.var_ub(0), None);
        // Ensure w is not equal to cur or next (this also tests PartialEq)
        assert_ne!(w, cur);
        assert_ne!(w, next);
    }

    #[test]
    fn test_coherent_test_le_var() {
        let d = Dbm::new(1);
        let r = d.test_le_var(0, 5);
        assert!(!r.bottom);
        // X ≤ 5 ⟹ 2X ≤ 10: the self-dual edge (0,1) carries 2·10 = 20.
        assert_eq!(r.m[0 * 2 + 1], 20);
        assert_eq!(r.var_ub(0), Some(5));
    }

    /// Golden-facts bridge: three hand-computed facts, each pinning
    /// one derivation channel — FW transitivity, the interval→sum routing
    /// through the self-dual edges, and S⁺ self-dual composition. Reads
    /// ONLY through the semantic projections — representation-independent.
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

        // C. A Figure-9-style S⁺ self-dual composition ([Miné06] Figure 9):
        // 2X ≤ 1 ∧ 2Y ≤ 2 ⟹ X+Y ≤ 1.5; the integer tight read is 1
        // (node_bound's floor semantics, see its docs).
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

    /// A Figure-6-style half-integer integer-unsatisfiable case
    /// (cf. [Miné06] Figure 6 and §V.D; Harvey–Stuckey tightening [HS97]):
    /// `2x ≤ 1 ∧ 2x ≥ 1` is real-satisfiable (x = 0.5) but
    /// integer-UNsatisfiable (2x is even); IntegerExact rounds both
    /// self-dual edges (`2x ≤ 1` → `2x ≤ 0`, `2x ≥ 1` → `2x ≥ 2`) and
    /// closes to ⊥, while Strong (the real/rational reading) stays sat.
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

    /// A [Miné06] Figure-9-style composition over integers —
    /// `2v₀ ≤ 1 ∧ 2v₁ ≤ 2`. Strong derives `v₀+v₁ ≤ 1.5` (stored 3,
    /// pinned by `test_strong_closure_deduces_sum`); IntegerExact
    /// first rounds `2v₀ ≤ 1` to `2v₀ ≤ 0`, and the re-run S⁺ then
    /// derives `v₀+v₁ ≤ 1` (stored 2). The semantic READ floors in
    /// both modes — the difference is in the STORED value (which is
    /// what propagates through the closure).
    #[test]
    fn test_closure_integer_exact_figure9() {
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 1); // 2v₀ ≤ 1
        d.set_mirrored(2, 3, 2); // 2v₁ ≤ 2
        assert!(d.close_with(ClosureMode::IntegerExact));
        assert_eq!(d.sum_ub(0, 1), Some(1), "integer-tight v₀+v₁ ≤ 1");
        assert_eq!(d.m[0 * d.size + 3], 2, "stored 2 (Strong leaves 3)");
    }

    /// Rounding PROPAGATES through difference edges —
    /// `2x ≤ 5 ∧ y − x ≤ 0 ∧ 2y ≥ 5`: real-satisfiable (x = y = 2.5),
    /// integer-UNsatisfiable (2x ≤ 5 ⟹ x ≤ 2 ⟹ y ≤ 2, but 2y ≥ 5
    /// ⟹ y ≥ 3 over the integers).
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

    /// Negative half-integer bounds round with FLOOR
    /// semantics — `2x ≤ −5 ∧ 2x ≥ −5` is real-satisfiable (x = −2.5)
    /// and integer-UNsatisfiable; the stored edges −10 and 10 round
    /// to −12 (`2x ≤ −6`) and 8 (`2x ≥ −4`), closing to ⊥.
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

    /// Integer-clean inputs (all self-dual entries multiples
    /// of 4) — the tightening is a NO-OP and the two modes produce
    /// bit-identical closures. Guards against over-tightening.
    #[test]
    fn test_closure_integer_exact_noop_on_integer_bounds() {
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

    /// The raw stored rounding — `2x ≤ 5` (stored 10) rounds
    /// to `2x ≤ 4` (stored 8) under IntegerExact; Strong keeps 10 (the
    /// exact half-integer bound). The semantic READ (var_ub) floors in
    /// both modes — the difference is in the STORED value that
    /// propagates (test_closure_integer_exact_propagates exercises it).
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
    // Differential regression pins for the closure-free fast paths.
    // The reference path re-does the same point-wise operation and then
    // pays the full strong closure; by the preservation lemmas in the
    // docs of `join` / `assign_add_var`, that closure is a no-op there,
    // so both paths must agree bit-for-bit. Any mismatch means a
    // preservation lemma is wrong — do NOT "fix" by re-adding close().
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
        };
        r.close();
        r
    }
    #[test]
    fn test_join_differential_no_close() {
        // Case 1: interval + difference constraints (2 vars).
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 4); // X ≤ 4
        a.set_mirrored(1, 0, -2); // X ≥ 1
        a.set_mirrored(0, 2, 3); // X − Y ≤ 3
        assert!(a.close());
        let mut b = Dbm::new(2);
        b.set_mirrored(2, 3, 2); // Y ≤ 2
        b.set_mirrored(0, 2, 1); // X − Y ≤ 1
        assert!(b.close());
        // Case 2: half-integer derived bounds (S⁺ composition).
        let mut c1 = Dbm::new(2);
        c1.set_mirrored(0, 1, 1); // 2X ≤ 1
        c1.set_mirrored(2, 3, 2); // 2Y ≤ 2
        assert!(c1.close());
        // Case 3: 3-var transitive chains.
        let mut d3a = Dbm::new(3);
        d3a.set_mirrored(0, 2, 4); // X − Y ≤ 4
        d3a.set_mirrored(2, 4, 2); // Y − Z ≤ 2
        d3a.set_mirrored(0, 1, 3); // X ≤ 3
        assert!(d3a.close());
        let mut d3b = Dbm::new(3);
        d3b.set_mirrored(4, 0, -2); // Z − X ≤ −2
        assert!(d3b.close());
        let top2 = Dbm::new(2); // top is trivially strongly closed
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
        // Reference path that always applies close() after the raw update.
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
        };
        r.close();
        r
    }
    #[test]
    fn test_assign_add_differential_no_close() {
        let mut a = Dbm::new(3);
        a.set_mirrored(0, 1, 4); // X ≤ 4
        a.set_mirrored(1, 0, -2); // X ≥ 1
        a.set_mirrored(0, 2, 3); // X − Y ≤ 3
        a.set_mirrored(2, 4, 2); // Y − Z ≤ 2
        assert!(a.close());
        for &(i, c) in &[(0usize, -3i128), (0, 5), (1, 1), (2, -1)] {
            let fast = a.assign_add_var(i, c);
            let slow = reference_assign_add(&a, i, c);
            assert_eq!(fast, slow, "differential mismatch at (i={}, c={})", i, c);
            assert!(!fast.bottom);
            assert!(fast.is_strongly_closed());
        }
        // A large-but-representable finite δ (c = ±(DBM_INF/4 − 1)): no
        // entry saturates, so the no-close fast path and the closing
        // reference still agree exactly.
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
        // Saturation corners: c = i128::MAX/4 drives δ to DBM_INF
        // (positive saturation); c = −2^126 drives δ to i128::MIN — the
        // exact doubled value collides with the −∞ marker. The shift is
        // unrepresentable, so assign_add_var forgets the variable (sound
        // over-approximation), never bottom.
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
        // Half-integer derived bounds shift through unchanged parity.
        let mut h = Dbm::new(2);
        h.set_mirrored(0, 1, 1); // 2X ≤ 1
        h.set_mirrored(2, 3, 2); // 2Y ≤ 2
        assert!(h.close());
        assert_eq!(h.assign_add_var(0, 1), reference_assign_add(&h, 0, 1));
        assert!(h.assign_add_var(0, 1).is_strongly_closed());
    }
    #[test]
    fn test_is_strongly_closed_tripwire() {
        assert!(Dbm::new(2).is_strongly_closed()); // top: vacuous
        assert!(Dbm::bottom().is_strongly_closed()); // ⊥: vacuous
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 4);
        d.set_mirrored(0, 2, 3);
        assert!(d.close());
        assert!(d.is_strongly_closed());
        // One-sided tighten WITHOUT closure is caught (coherence break:
        // `set` writes one direction only).
        d.set(0, 2, 1);
        assert!(!d.is_strongly_closed());
    }
    #[test]
    fn test_join_preserves_rounded_normal_form() {
        // Both operands closed under IntegerExact: every finite
        // self-dual stored bound is a multiple of 4; the point-wise max
        // of two multiples of 4 is a multiple of 4 — join never exits
        // the Harvey–Stuckey rounded normal form ([HS97]; see [Miné06] §V.D).
        let mut a = Dbm::new(2);
        a.set_mirrored(0, 1, 5); // 2x ≤ 5 → rounds to 2x ≤ 4
        a.set_mirrored(1, 0, -2); // 2x ≥ 2 (stored −4, clean)
        assert!(a.close_with(ClosureMode::IntegerExact));
        let mut b = Dbm::new(2);
        b.set_mirrored(0, 1, 3); // 2x ≤ 3 → rounds to 2x ≤ 2
        b.set_mirrored(2, 3, 4); // 2y ≤ 4 (stored 8, clean)
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
}

#[test]
fn test_overflow_and_edge_cases() {
    // 1. assign_const_var with the largest exactly representable constant:
    //    the self-dual edges store ±4c, so 4c must stay below DBM_INF.
    let d = Dbm::new(1);
    let max_var = DBM_INF / 4 - 1;
    let r = d.assign_const_var(0, max_var);
    assert!(
        !r.bottom,
        "assign_const_var should not collapse to bottom on large valid constants"
    );
    assert_eq!(r.var_ub(0), Some(max_var));
    assert_eq!(r.var_lb(0), Some(max_var));

    // 2. assign_const_var with an unrepresentable constant forgets the
    //    variable: 4c reaches/exceeds DBM_INF (positive side), or -4c
    //    does (negative side). No one-sided bound is encoded. (Note the
    //    floor division: DBM_INF = 2^125 − 1, so DBM_INF/4 = 2^123 − 1 is
    //    still representable; +1/−1 steps to the first out-of-range c.)
    let too_big = DBM_INF / 4 + 1; // 4c = 2^125 ≥ DBM_INF
    let r = Dbm::new(1).assign_const_var(0, too_big);
    assert!(!r.bottom);
    assert_eq!(r.var_ub(0), None);
    assert_eq!(r.var_lb(0), None);

    let too_neg = -(DBM_INF / 4) - 1; // -4c = 2^125 ≥ DBM_INF
    let r = Dbm::new(1).assign_const_var(0, too_neg);
    assert!(!r.bottom);
    assert_eq!(r.var_ub(0), None);
    assert_eq!(r.var_lb(0), None);

    // 3. test_le_var with an unrepresentable constant weakens to no bound
    let d = Dbm::new(1);
    let c = (i128::MAX / 2) + 1; // 2^126: 4c overflows the finite range
    let r = d.test_le_var(0, c);
    assert!(!r.bottom);
    assert_eq!(r.var_ub(0), None);

    // 4. test_ge_var with an unrepresentable constant weakens to no bound
    let d = Dbm::new(1);
    let c_neg = i128::MIN / 2; // -2^126
    let r = d.test_ge_var(0, c_neg);
    assert!(!r.bottom);
    assert_eq!(r.var_lb(0), None);

    // 5. assign_copy_var(i, i) preserves matrix exactly
    let mut d = Dbm::new(2);
    d.set_mirrored(0, 1, 10); // X <= 5
    d.set_mirrored(2, 3, 4); // Y <= 2
    d.close();
    let original = d.clone();
    let copied = d.assign_copy_var(0, 0);
    assert_eq!(
        original, copied,
        "assign_copy_var(i, i) must be a no-op identity"
    );
}

/// Regression test: individual-entry saturation in `assign_add_var`.
///
/// Even when the additive shift `δ` itself is finite, a particular matrix
/// entry may clip to a saturation sentinel. For example,
/// `c = (DBM_INF − 7) / 2 = 2^124 − 4` gives `δ = 2c = DBM_INF − 7`.
/// Then `8 + δ = 2^125 ≥ DBM_INF`, so the direct edge clips to `DBM_INF`.
/// The exact shifted bound is unrepresentable; clipping to `DBM_INF`
/// is the sound weakening. The update detects any clipped entry and
/// falls back to a closing pass. This test verifies the result remains
/// strongly closed and non-bottom.
#[test]
fn test_assign_add_var_saturation_closes() {
    let mut a = Dbm::new(2);
    a.set_mirrored(0, 1, 2); // X⁺ − X⁻ ≤ 2 (2X ≤ 2): stored 4
    a.set_mirrored(1, 2, 2); // X⁻ − Z⁺ ≤ 2: stored 4 (+ mirror (3,0))
    a.set_mirrored(0, 2, 4); // X⁺ − Z⁺ ≤ 4: stored 8 (+ mirror (3,1))
    assert!(a.close());
    assert_eq!(a.m[0 * a.size + 2], 8);
    assert!(a.is_strongly_closed());

    let c = (DBM_INF - 7) / 2; // δ = 2c = DBM_INF − 7 (even, no floor loss)
    let r = a.assign_add_var(0, c);
    assert!(!r.bottom);
    assert!(r.is_strongly_closed(), "saturated update must stay closed");
    // 8 + δ = 2^125 is unrepresentable; the sound weakening is DBM_INF.
    // The X⁻ path also dies (the self-dual edge shifts by 2δ and clips),
    // so closure cannot recover anything finite here.
    assert_eq!(
        r.m[0 * r.size + 2],
        DBM_INF,
        "clipped edge stays a sound weakening"
    );
    assert_eq!(r.m[3 * r.size + 1], DBM_INF, "mirror clipped coherently");
}

/// The same fallback, demonstrated on a deliberately loose input: the
/// direct edge clips to `DBM_INF` while a two-edge path stays finite, so
/// the raw no-close result would violate triangle closure. This is the
/// exact interaction the saturation tracking guards against. On a
/// strictly closed input, closure inequalities would force the path ≥
/// direct edge, so a clip would weaken the path too and the raw result
/// would stay closed by itself; the fallback guarantees the
/// strong-closure postcondition for ANY input, closed or not.
#[test]
fn test_assign_add_var_saturation_repairs_loose_input() {
    let mut a = Dbm::new(2);
    a.set_mirrored(0, 1, 2); // X⁺ − X⁻ ≤ 2: stored 4
    a.set_mirrored(0, 2, 1); // X⁺ − Z⁺ ≤ 1: stored 2 (+ mirror (3,1))
    a.set_mirrored(2, 3, 2); // Z⁺ − Z⁻ ≤ 2 (2Z ≤ 2): stored 4
    a.set_mirrored(0, 3, 5); // X⁺ − Z⁻ ≤ 5: stored 10 (+ mirror (2,1))
    // The direct edge (10) is looser than the path X⁺→Z⁺→Z⁻ (2 + 4):
    // coherent but NOT strongly closed.
    assert!(!a.is_strongly_closed(), "premise: loose input");

    let c = (1i128 << 124) - 4; // δ = 2c = 2^125 − 8 = DBM_INF − 7
    let r = a.assign_add_var(0, c);
    assert!(!r.bottom);
    // Raw shift: direct edge clips to DBM_INF (10 + δ ≥ DBM_INF) while
    // the shifted path X⁺→Z⁺→Z⁻ = (2 + δ) + 4 = 2^125 − 2 stays finite —
    // the raw no-close result would violate triangle closure. The
    // fallback closing pass recovers the path bound.
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

/// Saturated lower-bound projection must return `None`, not a clipped
/// sentinel.
///
/// Strong closure can produce the self-dual cell
/// `m[2i+1][2i] = i128::MIN`, which represents the constraint
/// `X⁻ − X⁺ ≤ −2^127` (i.e. `X ≥ 2^125` exactly). The projection path is:
/// `node_bound` reads this as `−2^126`, then `sat_neg(−2^126)` attempts
/// to compute `+2^126`, which exceeds the finite representable range and
/// saturates to `DBM_INF`. Because `DBM_INF` is the module's "no finite
/// bound" sentinel, returning `Some(DBM_INF)` would be both wrong and
/// ambiguous. The current implementation maps that clipped value back to
/// `None` (∞), which is the sound over-approximation for an
/// unrepresentable lower bound.
#[test]
fn test_var_lb_saturated_lower_edge_is_none() {
    let mut d = Dbm::new(2);
    // Raw writes: closure composes X⁻→Z⁺ and Z⁺→X⁺ into
    // X⁻−X⁺ ≤ −2^127 = i128::MIN (self-dual lower edge of X).
    d.m[1 * d.size + 2] = -(1i128 << 126); // X⁻ − Z⁺ ≤ −2^126
    d.m[3 * d.size + 0] = -(1i128 << 126); // mirror of (1,2)
    d.m[2 * d.size + 0] = -(1i128 << 126); // Z⁺ − X⁺ ≤ −2^126
    d.m[1 * d.size + 3] = -(1i128 << 126); // mirror of (2,0)
    assert!(d.close(), "constraints are satisfiable (no negative cycle)");
    assert!(!d.bottom);
    assert_eq!(
        d.m[1 * d.size + 0],
        i128::MIN,
        "closure must produce the saturated self-dual cell"
    );
    // The current projection maps the clipped result back to None.
    assert_eq!(
        d.var_lb(0),
        None,
        "saturated lower bound projects to ∞, not Some(DBM_INF)"
    );
    assert_eq!(d.var_ub(0), None);
}
