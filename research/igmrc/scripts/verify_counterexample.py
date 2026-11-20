#!/usr/bin/env python3
"""Verify the exact-chi rank-1 interval-colouring LP counterexample.

Instance: n = 9, k = 4, integer weights.  The clique LP optimum is -100.5 at
a half-integral point; the integral optimum is -100.  Standard library only.
"""

from fractions import Fraction
from itertools import product

INTERVALS = [
    (5, 10), (7, 7), (6, 7), (2, 6), (10, 10),
    (5, 5), (8, 10), (1, 3), (7, 9),
]
WEIGHTS = [-4, -3, -4, -4, -4, -4, -1, -4, -4]
K = 4
CLIQUES = [
    (3, 7),
    (0, 3, 5),
    (0, 2, 3),
    (0, 1, 2, 8),
    (0, 6, 8),
    (0, 4, 6),
]

H = Fraction(1, 2)
LP = [
    [H, H, 0, 0],
    [H, 0, H, 0],
    [0, H, 0, H],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, H, H, 0],
    [0, 0, 0, 1],
    [0, 0, H, H],
]


def conflicts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Closed-interval overlap: even touching endpoints conflict."""
    return a[0] <= b[1] and b[0] <= a[1]


def maximal_cliques(intervals: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """All maximal cliques by 2^n enumeration (n = 9 -> 511 subsets).

    The hard-coded `CLIQUES` list is an input to both the LP feasibility check and
    the IP brute force, so a *missing non-binding* clique would slip through both
    (the IP optimum is pinned by the binding ones).  Recomputing makes the list —
    and therefore the documented "maximal cliques" instance — self-validating.
    """
    n = len(intervals)
    out = []
    for mask in range(1, 1 << n):
        verts = [v for v in range(n) if mask >> v & 1]
        if any(
            not conflicts(intervals[u], intervals[v])
            for i, u in enumerate(verts)
            for v in verts[i + 1 :]
        ):
            continue
        # Maximal: no outside vertex is adjacent to every member.
        if all(
            any(not conflicts(intervals[v], intervals[w]) for w in verts)
            for v in range(n)
            if not mask >> v & 1
        ):
            out.append(tuple(verts))
    return sorted(out)


def check_lp() -> Fraction:
    # The instance's clique list must be the *complete* maximal-clique set.
    assert sorted(CLIQUES) == maximal_cliques(INTERVALS), (
        "CLIQUES is not the complete maximal-clique set: "
        f"missing {sorted(set(maximal_cliques(INTERVALS)) - set(map(tuple, CLIQUES)))}"
    )
    for v, row in enumerate(LP):
        assert sum(row) == 1, f"row {v} does not sum to 1"
        # Pinning half-integrality makes the "half-integral LP point" claim
        # machine-checked rather than prose.
        for lane, val in enumerate(row):
            assert val in (0, H, 1), f"LP[{v}][{lane}] = {val} is not half-integral"
    for clique in CLIQUES:
        for lane in range(K):
            column = sum((LP[v][lane] for v in clique), Fraction(0))
            assert column <= 1, f"clique {clique} lane {lane} = {column}"
    mu = [sum((j + 1) * LP[v][j] for j in range(K)) for v in range(len(LP))]
    cost = sum(w * m for w, m in zip(WEIGHTS, mu))
    print("LP point is feasible")
    print("  mean colours:", [str(m) for m in mu])
    print("  LP cost:     ", cost)
    assert cost == Fraction(-201, 2), cost
    return cost


def brute_force_ip() -> int:
    best = None
    best_colouring = None
    for assignment in product(range(1, K + 1), repeat=len(INTERVALS)):
        if any(len({assignment[v] for v in clique}) != len(clique) for clique in CLIQUES):
            continue
        cost = sum(w * c for w, c in zip(WEIGHTS, assignment))
        if best is None or cost < best:
            best = cost
            best_colouring = assignment
    assert best is not None
    print("integral optimum:", best)
    print("  colouring:     ", best_colouring)
    return best


def main() -> None:
    lp = check_lp()
    ip = brute_force_ip()
    gap = ip - lp
    print("gap:", gap)
    assert ip == -100
    assert gap == Fraction(1, 2)
    print("OK: LP is strictly below every integral colouring")


if __name__ == "__main__":
    main()
