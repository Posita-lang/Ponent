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

**Traceability caveat.** No primary source was located that states
“every maximal clique has size `k` ⇒ the colouring LP is integral” in this
form; the cumulative-variable argument above is what carries the claim, and it
is conditional on the level systems really being network/TU polytopes (which is
what “min-cost flow” means). Read the polynomiality of this subclass as
**derived, not cited**. It is not conditional on the right-hand side being
`b ≡ 1`: total unimodularity is a property of the matrix, so the tightened
systems of §2.1 inherit it for any 0/1 right-hand side. If a hardness proof
were ever found for a single deficient clique, this argument — not the LP
counterexample of §3 — is what would have to fail.

### 2.1 Partial saturation: the deficiency parameter `d`

Let `𝒦` be the maximal cliques, `g_K = k − |K|` and `d = Σ_K g_K`. Summing the
lane constraints of a clique and using `Σ_j x_{v,j} = 1` gives

```
Σ_j (1 − Σ_{v∈K} x_{v,j}) = k − |K| = g_K ,
```

so `d` is exactly the total number of **idle lane slots** in the LP, and for an
integral colouring it is the number of absent colours summed over cliques.

**Derived algorithm.** Enumerate, for each clique `K`, the `g_K` colours that
are absent there — a pattern `b_{K,·} ∈ {0,1}^k` with `Σ_j b_{K,j} = |K|` —
and replace every lane inequality at `K` by the equality
`Σ_{v∈K} x_{v,j} = b_{K,j}`. By §2's TU/network argument the tightened system
is integral, so each pattern costs one flow. There are at most

```
Π_K C(k, g_K)  ≤  k^d  ≤  n^d
```

patterns, hence `n^{O(d)}` — XP in `d`, polynomial for every constant `d`.
For `d = 0` this is the fully saturated flow of §2; for `d = 1` (one clique of
size `k−1`) it is polynomial: guess which single colour is absent there
(at most `k ≤ n` flows).

**What is and is not settled.**

| question | status |
|---|---|
| every constant-`d` slice polynomial | **derived** (§2.1), so not para-NP-hard in `d` |
| FPT in `d`, i.e. `f(d)·n^{O(1)}` | **open** — `k` is part of the input, so `k^d` is not `f(d)` alone |
| parameter `Δ = max_K g_K` | `Δ = 1` does **not** bound `d` (`Θ(n)` cliques of size `k−1` give `d = Θ(n)`), so §2.1 says nothing; NP-hardness at fixed `Δ` is open |

One trap: *physically* adding dummy intervals to fill a deficient clique is
**not** a valid reduction for this objective. The dummies must take the free
colours and are charged `w_v·c(v)`, which changes the optimum. The correct
reduction is virtual — enumerate the absent-colour patterns and pay nothing.

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

### 3.1 What stays true, and which proof routes are closed

**Saturation lemma (rigorous).** For a maximum clique `K`, `|K| = k`, summing
`Σ_j x_{v,j} = 1` over `v ∈ K` gives `k`, while each lane column satisfies
`Σ_{v∈K} x_{v,j} ≤ 1`. All `k` columns are therefore tight, the `k×k` block
`(x_{v,j})` is doubly stochastic, and with `μ_v = Σ_j j·x_{v,j}`:

```
Σ_{v∈K} μ_v = k(k+1)/2 ,     μ|_K ∈ Π(1,…,k)  (permutahedron),
Σ_{v∈K\K'} x_{v,j} = Σ_{v∈K'\K} x_{v,j}       (consecutive maximum cliques).
```

This is real structure, but it is *local* and it is exactly what the
counterexample lacks: only one of its six maximal cliques is maximum, so the
lemma constrains nothing along the sweep.

**Closed routes.** Each row below was falsified, not merely left unproven:

| route | counter-evidence |
|---|---|
| LP value-exactness, equivalently the mean-colour identity `P_μ = Q` | §3: LP `<` IP by `1/2` on an `n = 9`, `k = 4` instance |
| mean-level gluing (intersect local permutahedron constraints on `μ`) | strictly weaker than the LP; explicit `k = 5` vector, gap `1.3346` |
| dual telescoping certificate `(⋆)` below | infeasible at the §3 instance, explicit Farkas certificate |
| “negative cycle of the dual ⇒ improving Kempe swap” | the violating cycle closes through an *idle* lane slot, which has no Kempe counterpart |
| nested max-weight `j`-colourable bounds `V(1..k−1)` | loose: bound 28 vs exact 29 |
| layered “switching” flow LP | strictly cheaper than the optimum on 88/200 random families |
| prefix / weight-ordered canonical DP states | 78/95 and 46/95 families have no optimal colouring of that form |

**The dual certificate, and why it fails.** With `Λ_{t,j} = Σ_{s≤t} λ_{s,j}`
over the sweep positions `t`, the LP dual collapses to

```
max_Λ  Σ_v min_j ( w_v·j + Λ_{r_v,j} − Λ_{ℓ_v−1,j} ) − Σ_j Λ_{T,j} ,
```

and LP `=` IP holds for a given optimal colouring `c*` iff there are
`λ_{t,j} ≥ 0`, vanishing on every slot `(t,j)` that is idle in `c*`, with

```
Σ_{t=ℓ_v}^{r_v} ( λ_{t,j} − λ_{t,c*(v)} )  ≥  w_v ( c*(v) − j )     for all v, j.   (⋆)
```

At the §3 instance `(⋆)` is infeasible: six of its inequalities, each weighted
`1/6`, have right-hand sides summing to `+1`, while their λ-terms telescope to
`−(λ_{1,2} + λ_{2,4} + λ_{4,1} + λ_{5,3}) ≤ 0` because only those four slots
are occupied by `c*`. The dual is therefore *not* integral on the rank-1 cone —
objective-specific TDI fails, it is not merely unproven. The infeasibility
certificate runs through an idle slot pinned by earlier prices, and `c*` is
globally optimal, so no cyclic colour exchange explains it.

---

## 4. How large can the additive gap be?

Scale first matters. The objective is homogeneous: `w ↦ M w` multiplies both
the LP and IP optima, hence the gap, by `M`. Without a weight bound the gap is
unbounded.

With normalized weights (e.g. `|w_v| ≤ 1`), the gap still grows at least
linearly in `n`: take `m` vertex-disjoint copies of the `0.5`-gap gadget above.
The LP and IP are additive across components, so the gap is `Ω(n)` while `k`
stays 4. A trivial upper bound is `O(nK)` with `K ≤ 2k−1`, so `O(nk)`. At
`k = 4` the sharp order is open: `Ω(n)` is unconditional (disjoint copies),
`O(nk)` is the ceiling, and nothing separates `Θ(n)` from `Θ(nk)`.

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

| instance | exact engine | cost |
|---|---|---|
| general | sweep DP over injective label states | `O(n · k! · k)`, FPT in `k`; `n ≤ 50`, `k ≤ 5` ⇒ ≤ 120 states |
| fully saturated (every maximal clique size `k`) | rank-1 LP / nested flow | `poly(n,k)` (§2, derived not cited) |
| bounded deficiency `d` | enumerate absent-colour patterns, one flow each | `n^{O(d)}` (XP); `d ≤ 1` polynomial (§2.1) |
| any instance, LP first | solve the clique LP, fall back to the DP on a fractional optimum | exact unconditionally; the LP is only a fast path |

Do **not** use the clique LP alone as a certificate of optimality: §3 is a
concrete instance where its optimum lies strictly below every colouring. The
greedy χ-lane sweep plus the `O(n log n)` weight-relabelling post-pass is a
heuristic — optimal only among relabellings of the one colouring the greedy
produced. That is what `src/diagnostics/glyph.rs` ships for the `n ≤ ~50`,
`χ ≤ ~5` rows, and its corpus gap to exact IGMRC has not been measured.

---

## 7. Scripts

All scripts are Python 3, standard library only, and print `ALL CHECKS PASSED`
on success.

| script | what it pins |
|---|---|
| `scripts/verify_counterexample.py` | the `-100.5 / -100` instance: LP feasibility in exact `Fraction` arithmetic, exhaustive integral optimum, gap `1/2` |
| `scripts/verify_padding_reduction.py` | the NP-hardness gadget both ways — disjoint padding clique and the connected staggered bridge; checks connectivity, `χ = ω = p`, that the core stays unsaturated, and `OPT(G) = OPT(H) + const` by brute force (including mixed-sign weights) |
| `scripts/verify_saturation_boundary.py` | the slack identity `Σ_j (1 − Σ_{v∈K} x_{v,j}) = k − |K|`; the two documented fractional points (fully saturated `k = 5`, and `d = 1` at `k = 4`); the pattern count `Π_K C(k, g_K) ≤ k^d` |
| `scripts/verify_dual_certificate.py` | `(⋆)` infeasible at the counterexample, with the six-inequality Farkas certificate re-derived in exact arithmetic |

```sh
python3 research/igmrc/scripts/verify_counterexample.py
python3 research/igmrc/scripts/verify_padding_reduction.py
python3 research/igmrc/scripts/verify_saturation_boundary.py
python3 research/igmrc/scripts/verify_dual_certificate.py
```