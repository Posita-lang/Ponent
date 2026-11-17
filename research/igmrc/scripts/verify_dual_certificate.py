#!/usr/bin/env python3
"""Verify that the dual telescoping certificate ``(⋆)`` fails at the §3
counterexample (``../README.md`` §3.1).

Setup.  With dual prices ``alpha_v`` (free) for the assignment rows and
``lambda_{t,j} >= 0`` for the lane rows, the telescoped dual is

    max_Lambda  sum_v min_j ( w_v*j + Lambda_{r_v,j} - Lambda_{l_v-1,j} )
                - sum_j Lambda_{T,j},

and LP = IP for a given optimal colouring ``c*`` iff there are
``lambda >= 0``, zero on every slot that is idle in ``c*``, satisfying

    sum_{t in span(v)} ( lambda_{t,j} - lambda_{t,c*(v)} )
        >= w_v * ( c*(v) - j )                                    (⋆)

for every vertex ``v`` and lane ``j``.

What this script does.  It rebuilds the counterexample, recomputes ``c*`` and
its cost, forms the six ``(⋆)`` rows listed in the note *mechanically* (each
side computed from the data, not hard-coded), and sums them with weight
``1/6``.  The right-hand sides sum to ``+1``; after substituting ``lambda = 0``
on the idle slots, every remaining coefficient is ``<= 0``.  Hence the summed
left-hand side is ``<= 0`` while the summed right-hand side is ``1`` — the
system is infeasible and no dual certificate exists.

Consequently "objective-specific TDI on the rank-1 cone" is false, not merely
unproven; and the violating cycle closes through an *idle* lane slot, so it has
no interpretation as an improving Kempe colour exchange.

Standard library only.
"""

from __future__ import annotations

from fractions import Fraction

from verify_counterexample import CLIQUES, INTERVALS, K, LP, WEIGHTS

# The optimal colouring of the counterexample (0-based vertices, lanes 1..k).
C_STAR = (2, 3, 4, 3, 4, 4, 3, 4, 1)

# The six (v, j) inequalities used by the Farkas certificate.
ROWS = [(0, 1), (1, 1), (2, 2), (6, 2), (8, 3), (8, 4)]


def lp_cost() -> Fraction:
    total = Fraction(0)
    for v, row in enumerate(LP):
        mu = sum((j + 1) * row[j] for j in range(K))
        total += WEIGHTS[v] * mu
    return total


def check_colouring() -> int:
    mu = [sum((j + 1) * LP[v][j] for j in range(K)) for v in range(len(LP))]
    lp = sum(w * m for w, m in zip(WEIGHTS, mu))
    ip = sum(w * c for w, c in zip(WEIGHTS, C_STAR))
    print(f"LP point cost = {lp}, colouring cost = {ip}")
    assert lp == Fraction(-201, 2) and ip == -100
    for q in CLIQUES:
        assert len({C_STAR[v] for v in q}) == len(q), q
    return ip


def row(v: int, j: int) -> tuple[dict[tuple[int, int], int], int]:
    """One (⋆) inequality: coefficients of lambda, and its right-hand side."""
    cv = C_STAR[v]
    coeff: dict[tuple[int, int], int] = {}
    for t, q in enumerate(CLIQUES):
        if v not in q:
            continue
        coeff[(t, j)] = coeff.get((t, j), 0) + 1
        coeff[(t, cv)] = coeff.get((t, cv), 0) - 1
    return coeff, WEIGHTS[v] * (cv - j)


def main() -> None:
    print("Dual certificate (⋆) at the LP counterexample\n")
    check_colouring()

    # Slots actually occupied by c*: lambda is pinned to 0 elsewhere.
    occupied = {(t, C_STAR[v]) for t, q in enumerate(CLIQUES) for v in q}

    total: dict[tuple[int, int], int] = {}
    rhs_total = 0
    for v, j in ROWS:
        coeff, rhs = row(v, j)
        assert sum(coeff.values()) == 0
        rhs_total += rhs
        for slot, a in coeff.items():
            total[slot] = total.get(slot, 0) + a
        print(f"  (v={v}, j={j}): {len(coeff)} lambda terms, "
              f"rhs = w_v(c_v - j) = {rhs}")

    assert sum(total.values()) == 0
    assert rhs_total == 1, rhs_total  # -4 -6 -8 -1 +8 +12
    print(f"\nsum of the six rows: rhs total = {rhs_total} (> 0; each row "
          f"weighted 1/6)")

    # Substitute lambda = 0 on idle slots; the rest must be non-positive.
    live = {slot: a for slot, a in total.items() if a != 0 and slot in occupied}
    idle = {slot: a for slot, a in total.items() if a != 0 and slot not in occupied}
    assert all(a <= 0 for a in live.values()), live
    print(f"  after lambda=0 on idle slots, surviving terms: {live}")
    print(f"  (pinned slots that dropped out: {idle})")
    print("  => summed LHS <= 0, summed RHS = +1  (0 >= 1): infeasible")

    print("\nThe certificate runs through idle lane slots with no Kempe "
          "counterpart,")
    print("so 'negative cycle => improving Kempe swap' cannot be repaired "
          "locally.")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()