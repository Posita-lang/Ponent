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


def check_lp() -> Fraction:
    for v, row in enumerate(LP):
        assert sum(row) == 1, f"row {v} does not sum to 1"
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
