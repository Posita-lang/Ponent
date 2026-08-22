//! The octagon abstract domain (difference-bound matrices) — extracted
//! from `type_eq.rs` to keep the equality-checking module focused.
//!
//! A difference-bound matrix over **2n nodes** — `Xᵢ⁺ = 2i` (the variable)
//! and `Xᵢ⁻ = 2i+1` (its negation) — with **strong closure** (Figure 8),
//! meet/join/widen, and loop-transfer functions. There is NO implicit
//! zero node: single-variable bounds live on the **self-dual edges**
//! `m[2i][2i+1]` (`2Xᵢ ≤ 2c`, i.e. `Xᵢ ≤ c`) and `m[2i+1][2i]`
//! (`−2Xᵢ ≤ −2c`, i.e. `Xᵢ ≥ c`); the mirror of node `i` is `i ⊕ 1`, and
//! the mirror of edge `(i, j)` is `(j⊕1, i⊕1)`.
//!
//! All stored bounds are **doubled** (2×c) to represent half-integers
//! exactly. Interval (self-dual) rows therefore carry 4× the interval
//! half-width in raw storage — readers must go through the semantic
//! projections (`var_ub` / `var_lb` / `diff_bound` / `sum_ub`), never
//! raw cells. External API accepts plain `c`, internal storage uses the
//! doubled space.

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
    if a == DBM_INF || b == DBM_INF {
        return DBM_INF;
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
        return DBM_INF;
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
        return DBM_INF;
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

/// Compute the mirror index for coherence: node `i` mirrors to `i ⊕ 1`
/// (`Xᵢ⁺ = 2i ↔ Xᵢ⁻ = 2i+1`).
#[inline]
const fn mirror_index(i: usize) -> usize {
    i ^ 1
}

/// A difference-bound matrix over `2n` nodes: `Xᵢ⁺ = Xᵢ` and
/// `Xᵢ⁻ = -Xᵢ`. `m[i][j]` encodes `node_i - node_j ≤ c` **in doubled
/// space** (`m[i][j] = 2*c`). (The paper's convention is the transpose —
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
/// (paper §V.D).
///
/// `Strong` is the paper's Figure 8 strong closure — sound over reals,
/// rationals AND integers (an over-approximation for the latter).
///
/// `IntegerExact` additionally interleaves the HS tightening step
/// (§V.D: `2x ≤ 2c+1 ⟹ 2x ≤ 2c` — knowing x is an integer, the
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
        let stored = if c == DBM_INF { DBM_INF } else { sat_mul2(c) };
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
        let stored = if c == DBM_INF {
            DBM_INF
        } else {
            // Saturate on overflow to DBM_INF (conservative) or i128::MIN
            // to preserve negative diagonal.
            match c.checked_mul(2) {
                Some(v) => {
                    if v < DBM_INF {
                        v
                    } else {
                        DBM_INF
                    }
                }
                None => {
                    if c < 0 {
                        i128::MIN // underflow
                    } else {
                        DBM_INF
                    }
                }
            }
        };
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

    /// Strong closure (Figure 8) preserving coherence and deriving
    /// octagonal constraints. Returns `false` if the system is
    /// unsatisfiable (negative diagonal), setting `bottom = true`.
    pub(crate) fn close(&mut self) -> bool {
        self.close_with(ClosureMode::Strong)
    }

    /// One pass of the strong closure (paper Figure 8): the C⁺_k / S⁺
    /// interleaving over all k, in place. Extracted from `close_with`
    /// so `IntegerExact` can re-run it between Harvey–Stuckey
    /// tightening rounds without duplicating the loop body.
    fn strong_closure_pass(m: &mut [i128], size: usize) {
        // The paper's loop applies C⁺ only at even pivots (Figure 8:
        // S⁺(C⁺_{2k}(·))); C⁺_{k̄} has the same five terms as C⁺_k, so
        // iterating over all 2n nodes applies each pivot twice —
        // redundant work, same result.
        // k iterates over ALL 2n nodes. The mirror of every node is
        // `i ⊕ 1`; with no implicit zero node the single-variable bounds
        // hang on the self-dual edges m[2i][2i+1] / m[2i+1][2i], and the
        // five C⁺ terms below stay mirror-symmetric under `i ↦ i⊕1`.
        for k in 0..size {
            let k_bar = mirror_index(k);
            // ---- C⁺_k (paper Figure 8). Each term is a path
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
            // property Figure 8's C⁺ is designed around (§V.C).
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
            // ---- S⁺ (paper Figure 8): m[i][j] ≤ (m[i][ī] + m[j̄][j]) / 2.
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

    /// Harvey–Stuckey tightening (paper §V.D): round every
    /// self-dual edge's stored bound down to a multiple of 4 — the
    /// edge (2v, 2v+1) carries `2x ≤ s/2` with 2x an EVEN integer, so
    /// `2x ≤ s/2` ⟺ `2x ≤ 4⌊s/4⌋` over Z (the paper's undoubled rule
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
        // Strong closure (paper Figure 8): one pass of the C⁺_k / S⁺
        // interleaving over all k.
        Self::strong_closure_pass(&mut self.m, size);
        // ---- Harvey–Stuckey integer tightening (paper §V.D) ----
        // `IntegerExact` interleaves the tightening around the closure
        // loops until a fixpoint (closed AND rounded — the normal form
        // Harvey–Stuckey build, O(N⁴)). The tightening rounds every
        // self-dual edge's stored bound down to a multiple of 4: the
        // edge carries `2x ≤ s/2`, and over INTEGERS 2x is even, so
        // the constraint is equivalent to `2x ≤ 4⌊s/4⌋` (paper:
        // `2x ≤ 2c+1 ⟹ 2x ≤ 2c`). Each constraint's integer solution
        // set is UNCHANGED by the rounding, so any prefix of the
        // iteration stays sound over integers — and unsound over
        // rationals (2x ≤ 5 admits x = 2.5; the tightened 2x ≤ 4 does
        // not): integer domains only. Rounds are capped defensively
        // (HS need O(N); stopping early forfeits only normal-form
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

    /// join (⊔): abstract disjunction – looser of each bound (diagonal tighter),
    /// then close. If one operand is bottom, return the other.
    pub(crate) fn join(&self, other: &Dbm) -> Dbm {
        debug_assert!(
            {
                let mut s = self.clone();
                let mut o = other.clone();
                s.close();
                o.close();
                s == *self && o == *other
            },
            "join operands must be strongly closed"
        );

        if self.bottom {
            return other.clone();
        }
        if other.bottom {
            return self.clone();
        }
        debug_assert_eq!(self.size, other.size);
        let size: usize = self.size;
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
        let mut r = Dbm {
            size,
            m,
            bottom: false,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// widen (∇): termination-guaranteeing upper approximation.
    /// Definition (paper): if new bound is ≤ old, keep old; otherwise ∞.
    /// Does NOT call close after widening (paper §VI.D).
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

    /// `X := X + c`
    pub(crate) fn assign_add_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
        }
        let p = 2 * i; // X⁺
        let q = 2 * i + 1; // X⁻
        let mut m = self.m.clone();
        let delta = sat_mul2(c); // doubled space
        for j in 0..self.size {
            m[p * self.size + j] = sat_add(m[p * self.size + j], delta);
            m[j * self.size + p] = sat_sub(m[j * self.size + p], delta);
            m[q * self.size + j] = sat_sub(m[q * self.size + j], delta);
            m[j * self.size + q] = sat_add(m[j * self.size + q], delta);
        }
        let mut r = Dbm {
            size: self.size,
            m,
            bottom: false,
        };
        r.close();
        if r.bottom { Dbm::bottom() } else { r }
    }

    /// `X := c`
    pub(crate) fn assign_const_var(&self, i: usize, c: i128) -> Dbm {
        if self.bottom {
            return Dbm::bottom();
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
        Self::set_mirrored_internal(&mut m, p, q, 2 * c, self.size);
        Self::set_mirrored_internal(&mut m, q, p, -2 * c, self.size);
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
        r.set_mirrored(2 * i, 2 * i + 1, 2 * c);
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
        r.set_mirrored(2 * i + 1, 2 * i, -2 * c);
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
    #[allow(dead_code)]
    pub(crate) fn var_lb(&self, i: usize) -> Option<i128> {
        self.node_bound(2 * i + 1, 2 * i).map(|v| sat_neg(v >> 1))
    }

    /// Semantic projection: `Xᵢ + Xⱼ ≤ c` or `None` (∞) — the sum rides
    /// the edge `X⁺ᵢ − X⁻ⱼ` (paper Figure 5: `vᵢ + vⱼ ≤ c` ⟺
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
        // ⊥ is the identity of ⊔ (paper §VII.A): join(⊥, ⊤) = ⊤.
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

        // C. S⁺ self-dual composition (paper Figure 9): 2X ≤ 1 ∧ 2Y ≤ 2
        // ⟹ X+Y ≤ 1.5; the integer tight read is 1 (node_bound's floor
        // semantics, see its docs).
        let mut d = Dbm::new(2);
        d.set_mirrored(0, 1, 1);
        d.set_mirrored(2, 3, 2);
        assert!(d.close());
        assert_eq!(d.sum_ub(0, 1), Some(1), "S⁺ composition (Figure 9)");
    }

    /// The Figure 6 case (Harvey–Stuckey integer tightening, paper
    /// §V.D): `2x ≤ 1 ∧ 2x ≥ 1` is real-satisfiable (x = 0.5) but
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

    /// The paper's Figure 9 composition over integers —
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
}
