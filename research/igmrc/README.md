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

## Answer

The problem is **NP-hard even when all weights are 1**, and the natural
maximal-clique LP is **not value-exact**. Therefore there is no polynomial-size
LP (or polynomially separable cutting-plane description) that is exact for
all rank-1 weights unless P = NP. The exact algorithm remains the
`O(n · k! · k)` sweep DP; it is FPT in `k` and is effectively linear in `n`
for the diagnostic regime (`k ≤ 5`).

### Why exact `χ` does not avoid hardness

Take any `p`-colour sum-colouring instance `(G₀,p)` with `p ≥ 2ω(G₀)−1`.
Add a disjoint clique `Q` of `p` pairwise-overlapping intervals and give every
new interval weight 1. Then:

- `ω(G) = χ(G) = p = k` (interval graphs are perfect);
- every proper `k`-colouring uses all `k` colours on `Q`, contributing the
  constant `k(k+1)/2`;
- `Q` is disconnected from `G₀`, so the restriction to `G₀` is an arbitrary
  proper `p`-colouring.

Hence

```
OPT_{k=χ}(G) = OPT_{p-colour}(G₀) + k(k+1)/2,
```

and `p`-colour minimum sum coloring on interval graphs is NP-hard. Padding
turns any hardness instance into one with `χ = ω = p`; the `p > χ` caveat is
not an escape.

### The clique LP is not value-exact

LP:

```
min   Σ_{v,j} w_v · j · x_{v,j}
s.t.  Σ_j x_{v,j} = 1                         for every v
      Σ_{v ∈ K} x_{v,j} ≤ 1                   for every maximal clique K, lane j
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
`w·μ = -100.5`. Exhaustive enumeration of all `4^9` assignments gives the
integral optimum `-100`, so the LP gap is `0.5`.

The LP is not integral because only one maximal clique has size `k`; the
unsaturated cliques let a lane split fractional mass between two chains, which
no injection can replicate. Saturated-clique blocks are doubly stochastic, but
they do not glue across an unsaturated clique.

## What to use

- Exact engine: the `O(n · k! · k)` sweep DP (reconstruction included).
- For `n ≤ 50`, `k ≤ 5` the DP is already linear in `n` with ≤ 120 states.
- An LP may be used only as a fast path: solve it, verify the solution is
  integral and proper, and **fall back to the DP on any fractional optimum**.
  The LP alone is not a certificate of optimality.

`scripts/verify_counterexample.py` checks the LP point and recomputes the IP
by brute force; it needs only the Python standard library.
