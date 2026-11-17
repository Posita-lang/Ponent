#!/usr/bin/env python3
"""Verify the saturation/deficiency boundary claims of ``../README.md`` §2–§2.1.

Three things are checked, all in exact arithmetic:

  1. the slack identity.  For a maximal clique ``K`` of an interval graph with
     palette ``k``, the LP's lane slacks satisfy
         ``sum_j (1 - sum_{v in K} x_{v,j}) = k - |K|``.
     Summed over maximal cliques this is the deficiency ``d`` of README §2.1.

  2. two documented fractional points:
       (a) fully saturated, ``k = 5``: every maximal clique has size ``k``, the
           polytope is still not integral, and every lane column is tight;
       (b) ``d = 1``, ``k = 4``: exactly one maximal clique has size ``k - 1``,
           a half-integral point with 16 of 32 entries fractional is feasible,
           and the fractional mass is not localised.
     Feasibility (not vertex-ness) is what is re-checked.

  3. the pattern count of the §2.1 algorithm, ``prod_K C(k, g_K) <= k^d``,
     together with the observation that these fractional points are *not*
     improving directions for any rank-1 cost sampled here: their cost never
     falls below the brute-force integral optimum.

Standard library only.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from random import Random

from verify_padding_reduction import conflicts, maximal_cliques

H = Fraction(1, 2)


# ------------------------------------------------------------------ points
# (a) fully saturated: a_1..a_5 at [1,1]; b_1..b_3 at [2,3]; c_1,c_2 at [2,2];
#     d_1,d_2 at [3,3].  Three maximal cliques, all of size 5.
SAT_IV = [(1, 1)] * 5 + [(2, 3)] * 3 + [(2, 2)] * 2 + [(3, 3)] * 2
SAT_X = [
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, H, H, 0],
    [0, 1, 0, 0, 0],
    [H, 0, 0, 0, H],
    [0, 0, 0, H, H],
    [H, 0, H, 0, 0],
    [0, 0, H, 0, H],
    [H, 0, 0, H, 0],
]

# (b) d = 1: one maximal clique of size k-1 = 3, the rest size 4.
DEF_IV = [(4, 6), (5, 5), (1, 4), (1, 2), (5, 5), (6, 6), (1, 6), (1, 4)]
DEF_X = [
    [0, H, H, 0],
    [0, 0, H, H],
    [0, H, 0, H],
    [0, H, H, 0],
    [H, H, 0, 0],
    [0, H, H, 0],
    [H, 0, 0, H],
    [H, 0, H, 0],
]


def check_point(iv, x, k, tag):
    n = len(iv)
    cliques = maximal_cliques(iv)
    d = sum(k - len(q) for q in cliques)

    for v, row in enumerate(x):
        assert len(row) == k
        assert sum(row, Fraction(0)) == 1, (tag, "row", v, sum(row))
        assert all(q >= 0 for q in row)

    for q in cliques:
        for j in range(k):
            col = sum((x[v][j] for v in q), Fraction(0))
            assert col <= 1, (tag, "clique", q, "lane", j + 1, col)
        # slack identity, one clique at a time
        slack = sum((1 - sum((x[v][j] for v in q), Fraction(0))
                     for j in range(k)), Fraction(0))
        assert slack == k - len(q), (tag, q, slack, k - len(q))

    frac = sum(1 for row in x for q in row if 0 < q < 1)
    mu = [sum((j + 1) * x[v][j] for j in range(k)) for v in range(n)]
    print(f"[{tag}]  k = {k}, maximal clique sizes = "
          f"{sorted({len(q) for q in cliques})}, d = {d}")
    print(f"  feasible; {frac}/{n * k} entries fractional; "
          f"mean colours = {[str(m) for m in mu]}")
    assert d == 0 or frac > 0

    patterns = 1
    for q in cliques:
        g = k - len(q)
        patterns *= _choose(k, g)
    assert patterns <= k ** d or d == 0, (patterns, k ** d)
    print(f"  absent-colour patterns prod_K C(k, g_K) = {patterns} "
          f"(<= k^d = {k ** d})")
    return cliques, d


def _choose(n, r):
    out = 1
    for i in range(r):
        out = out * (n - i) // (i + 1)
    return out


# ------------------------------------------------------- integral optimum
def all_proper(iv, k):
    """Every proper colouring into 1..k (DFS with adjacency pruning)."""
    n = len(iv)
    adj = [[conflicts(iv[u], iv[v]) for v in range(n)] for u in range(n)]
    order = sorted(range(n), key=lambda v: (-sum(adj[v]), iv[v]))
    c = [0] * n
    out = []

    def rec(i):
        if i == n:
            out.append(tuple(c[v] for v in range(n)))
            return
        v = order[i]
        used = {c[u] for u in order[:i] if adj[v][u]}
        for q in range(1, k + 1):
            if q not in used:
                c[v] = q
                rec(i + 1)
        c[v] = 0

    rec(0)
    return out


def check_not_rank1_improving(iv, x, k, mu, tag, samples=400, seed=17):
    """Sample rank-1 weights; the fractional point must never beat the IP."""
    cols = all_proper(iv, k)
    assert cols, "no proper colouring"
    rng = Random(seed)
    worst = None
    for _ in range(samples):
        w = [rng.randint(0, 6) for _ in range(len(iv))]
        point = sum(wi * mi for wi, mi in zip(w, mu))
        best = min(sum(wi * q for wi, q in zip(w, c)) for c in cols)
        gap = point - best
        if worst is None or gap < worst:
            worst = gap
        assert point >= best, (tag, "fractional point beats every colouring", w)
    print(f"  {len(cols)} proper colourings; {samples} random rank-1 weights: "
          f"min(point - IP) = {float(worst):.3f} (never negative)")
    print()


def main() -> None:
    print("Saturation / deficiency boundary (README §2, §2.1)\n")

    k_sat = 5
    _, d_sat = check_point(SAT_IV, SAT_X, k_sat, "fully saturated, k=5")
    assert d_sat == 0
    mu_sat = [sum((j + 1) * row[j] for j in range(k_sat)) for row in SAT_X]
    check_not_rank1_improving(SAT_IV, SAT_X, k_sat, mu_sat, "saturated")

    k_def = 4
    _, d_def = check_point(DEF_IV, DEF_X, k_def, "d=1, k=4")
    assert d_def == 1
    mu_def = [sum((j + 1) * row[j] for j in range(k_def)) for row in DEF_X]
    check_not_rank1_improving(DEF_IV, DEF_X, k_def, mu_def, "d=1")

    print("ALL CHECKS PASSED")
    print("  bounded deficiency does not restore LP integrality (d=1 already "
          "admits a half-integral point),")
    print("  but these points are not improving directions for the rank-1 "
          "cost, consistent with §2.1.")


if __name__ == "__main__":
    main()