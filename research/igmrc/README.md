# Exact-`χ` rank-1 interval sum coloring

## Problem

Given intervals with integer endpoints, let `χ = ω = k`. Assign each interval
a lane `c(v) ∈ {1,…,k}` so that overlapping intervals get different lanes,
minimising

```
Σ_v w_v · c(v),     w_v ∈ R.
```

This is minimum sum coloring on interval graphs with the number of colours
fixed to exactly `χ`, and with the rank-1 (Monge) cost matrix `c(v,j)=w_v·j`.

## Three different questions

The literature and the experiments answer three things that must not be
conflated:

1. **General interval graphs.** The problem is NP-hard even with unit
   weights. The padding construction below forces `χ = ω = k` without
   removing the hardness.
2. **Fully saturated interval graphs** (every maximal clique has size exactly
   `k`). The padding construction no longer applies; this subclass is
   polynomial, even though its clique LP polyhedron is not integral.
3. **The clique LP on general instances.** It is not value-exact; an explicit
   half-integral counterexample is included below. Hence no compact exact LP
   exists for the general problem unless P = NP.

---

## 1. General intervals: NP-hardness

Take any `p`-colour sum-colouring instance `(G₀,p)` with `p ≥ 2ω(G₀)−1`.
Add a disjoint clique `Q` of `p` pairwise-overlapping intervals, all weight 1.
Then:

- `ω(G) = χ(G) = p = k`;
- every proper `k`-colouring uses all `k` colours on `Q`, contributing the
  constant `k(k+1)/2`;
- `Q` is disconnected from `G₀`, so the restriction to `G₀` is an arbitrary
  proper `p`-colouring.

Thus

```
OPT_{k=χ}(G) = OPT_{p-colour}(G₀) + k(k+1)/2,
```

so the exact-`χ` problem is NP-hard. The hardness lives in the **unsaturated**
maximal cliques of `G₀`: they leave idle colour slots, which are exactly what
the padding construction preserves.

### FPT/ETH status

The exact algorithm is the sweep DP with `O(n · k! · k)` time and
`O(k!)` states; for fixed `k` it is linear in `n`. Since `k! = 2^{Θ(k log k)}`,
this is an FPT algorithm, not a polynomial one in `(n,k)`.

The state is a labelled bijection “active intervals ↔ colours”, not a set or
an unlabeled partition. Two states with the same idle set but different
bijections can be distinguished by a future interval that overlaps one holder
and not the other, so idle-set compression and dominance pruning are not
available. Permutation-state DPs of this shape are the standard
`2^{Ω(k log k)}`-under-ETH family (e.g. Chromatic Number parameterised by
treewidth); representative-set compression exploits interchangeable blocks,
which a labelled bijection does not have.

Caveat: the general permutation-state lower-bound machinery is established,
but a published reduction to *this exact* rank-1-weighted, exact-`χ`,
interval-graph statement was not located. The `k!` dependence is therefore
expected to be asymptotically optimal; the final ETH-tightness reduction for
this exact variant remains to be written down.

---

## 2. Fully saturated intervals: polynomial

Assume every maximal clique has exactly `k` vertices. Summing
`Σ_j x_{v,j}=1` over a maximal clique `K` gives

```
Σ_{v∈K} Σ_j x_{v,j} = k,
```

while each lane column satisfies `Σ_{v∈K} x_{v,j} ≤ 1`. The `k` column sums
add to `k`, so every one is exactly 1:

```
Σ_{v∈K} x_{v,j} = 1     for every maximal clique K and lane j.
```

There are no idle slots in any maximal clique.

This does **not** make the LP polytope integral. For `k=5`, three maximal
cliques `C₁={a₁..a₅}`, `C₂={b₁,b₂,b₃,c₁,c₂}`, `C₃={b₁,b₂,b₃,d₁,d₂}` and the
half-integral assignment in which the `b`/`c`/`d` blocks carry `1/2` entries
is a feasible fractional vertex. It is not optimal for any rank-1 cost, but it
shows that “fully saturated” and “integral polytope” are different statements.
Fractional vertices first appear at `k ≥ 4`.

For the rank-1 objective, introduce cumulative variables

```
y_{v,t} = Σ_{j>t} x_{v,j},    t = 1..k−1.
```

Then

```
Σ_{v,j} w_v j x_{v,j}
  = Σ_v w_v + Σ_{t=1}^{k−1} Σ_v w_v y_{v,t}.
```

At each level `t`, full saturation turns the clique equations into
`A y_{·,t} = (k−t)·1`, where `A` is the maximal-clique/vertex incidence
matrix. Interval graphs have consecutive ones, so after taking consecutive row
differences `A` is a path-incidence matrix; with `0≤y≤1` this is a
`(k−t)`-unit s–t flow polytope, hence integral. The chain condition
`y_{·,t} ≥ y_{·,t+1}` asks for nested integral flows, i.e. a proper
`k`-colouring. Because all coefficients are nonnegative, a min-cost-flow /
successive-shortest-path argument preserves an integral optimum.

Therefore, on the fully saturated subclass the rank-1 LP has zero integrality
gap and the problem is polynomial: solve the LP, or equivalently run a
`poly(n,k)` nested flow / matching on the clique path. The NP-hardness padding
cannot be applied while keeping every maximal clique at size `k`; enlarging the
hard core to enforce saturation transforms it into this flow problem.

---

## 3. The general clique LP is not value-exact

LP:

```
min   Σ_{v,j} w_v · j · x_{v,j}
s.t.  Σ_j x_{v,j} = 1
      Σ_{v ∈ K} x_{v,j} ≤ 1     (maximal clique K, lane j)
      x_{v,j} ≥ 0
```

Counterexample (`n = 9`, `k = 4`, integer weights):

```
intervals: (5,10) (7,7) (6,7) (2,6) (10,10) (5,5) (8,10) (1,3) (7,9)
weights:   -4     -3    -4    -4    -4      -4    -1     -4    -4
maximal cliques:
  {3,7}, {0,3,5}, {0,2,3}, {0,1,2,8}, {0,6,8}, {0,4,6}
```

A feasible half-integral LP point has rows

```
v0: 1/2 1/2  0   0      v3:  0   0   1   0      v6:  0  1/2 1/2 0
v1: 1/2  0  1/2  0      v4:  0   0   0   1      v7:  0   0   0  1
v2:  0  1/2  0  1/2     v5:  0   0   0   1      v8:  0   0  1/2 1/2
```

with mean colours `(1.5, 2, 3, 3, 4, 4, 2.5, 4, 3.5)` and
`w·μ = -100.5`. Exhaustive enumeration gives the integral optimum `-100`, so
the LP gap is `0.5`.

The fractional mass exists because only one maximal clique has size `k`; the
unsaturated cliques let a lane split between two chains. This is the same
mechanism that makes the general problem NP-hard.

---

## 4. How large can the additive gap be?

Scale first matters. The objective is homogeneous: `w ↦ M w` multiplies both
the LP and IP optima, hence the gap, by `M`. Without a weight bound the gap is
unbounded.

With normalized weights (e.g. `|w_v| ≤ 1`), the gap still grows at least
linearly in `n`: take `m` vertex-disjoint copies of the `0.5`-gap gadget above.
The LP and IP are additive across components, so the gap is `Ω(n)` while `k`
stays 4. A trivial upper bound is `O(nK)` with `K ≤ 2k−1`, so `O(nk)`.

A uniform gap `< 1` is impossible unless P = NP: with integer weights the IP
is integral, the fractional vertices are half-integral, and a gap below 1
would let one read the optimum off the LP. This does not prove a gap ≤ 1; it
only rules out a universal sub-unit bound.

Known strengthening directions:

- maximal-clique constraints (already in this LP);
- odd-hole/odd-cycle cuts on the conflict graph `K_K □ G`, which is the
  precise source of half-integrality;
- exponential set-partitioning / column-generation LPs, strong in practice
  but not compact.

No polynomially separable compact cut family can make the LP exact for the
general problem unless P = NP.

---

## 5. Connectivity does not restore tractability

Requiring the interval graph to be connected does **not** remove the hardness.
The disjoint padding clique can be replaced by a pendant `p`-clique attached
through a single staggered bridge:

- place the hard core `H` (all maximal cliques of size `< p`) to the left;
- let `K_p = {x_1,…,x_p}` share a Helly point `t`, with staggered left
  endpoints `l_1 < … < l_p ≤ t`;
- extend the rightmost interval `u` of `H` to end at `r` with
  `l_1 < r < l_2`.

Then `u` overlaps only `x_1`, so the bridge has size 2, every core clique
keeps its original size, and only the single `K_p` clique at the far right has
size `p`. The graph is connected and has `ω = χ = p`, but it is **not
saturated**. Give `K_p` a constant total rank-1 cost (all vertices the same
weight, or weight 0), fix `x_1` to the spare colour `p`, and every optimal
colouring of `H` extends without constraint because `H` only uses colours
`1..h`, `h < p`. Hence `OPT(G) = OPT(H) + const`, and the reduction goes
through.

The real tractability boundary is **`ω`-saturation of the entire clique
path**, not connectivity. A lone `p`-clique at the end of the path is enough
to force exact-`χ` while preserving the unsaturated slack where NP-hardness
lives. In particular, the fully saturated flow/LP algorithm from §2 cannot be
applied to this connected gadget.

## 6. What to use

- General instances: exact engine is the `O(n · k! · k)` sweep DP; for
  `n ≤ 50`, `k ≤ 5` it is already linear in `n` with ≤ 120 states.
- Fully saturated instances: the rank-1 LP / nested flow is exact and
  polynomial; this is the tractable special case.
- If an LP is used as a fast path, check integrality and **fall back to the DP
  on any fractional optimum**. The LP alone is not a certificate of optimality
  outside the fully saturated subclass.

`scripts/verify_counterexample.py` checks the explicit `-100.5 / -100`
instance with only the Python standard library.