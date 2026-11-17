#!/usr/bin/env python3
"""Exact phase criterion for bucket occupancy, and the q* >= m/2 theorem.

Notation (derivations.md §5): ``Occ(n, m, alpha)`` is the number of
``m``-buckets hit by ``{ {i*alpha} : 1 <= i <= n }``.

Two exact statements are verified over **all** 80640 cells of the critical
arrangement at ``m = n = 64`` (integer arithmetic only):

  1. Phase criterion.  Sort the ``N = m`` points ``y_1 < ... < y_N`` and let
     ``Z`` be the number of adjacent pairs (in this order, not wrapping) that
     fall in the same ``1/N``-bucket.  Then

         ``Occ(N, N, alpha) = N - Z``,

     equivalently ``alpha`` is low-occupancy iff more than half of the gaps are
     *phase-short*: ``{N y_j} + N g_j < 1``.  This is the "three-distance +
     grid phase" criterion; the grid phase is exactly what a finite
     continued-fraction prefix does not determine, which is why no CF-only
     function decides the threshold (§5.3).

  2. ``q* >= m/2`` theorem.  With ``q* = argmin_{1<=q<=m} ||q*alpha||``
     (the largest convergent denominator ``<= m``),

         ``q* >= m/2  =>  Occ(m, m, alpha) >= ceil(m/2)``,

     with equality only on a tight layer of measure ``2*phi(m/2)/m^2``
     (``1/128`` at ``m = 64``).  The exceptional set is **empty**, so the
     one-directional criterion is exact at the ``m/2`` threshold — a different
     (much lower) threshold than the ``52`` statistic quoted in §5.3.

Also prints the exact measures ``meas(q* >= 32)``, ``meas(Occ < 32)`` and the
uniform-alpha mean ``E[Occ]`` for comparison with §5.1 and with the i.i.d.
value ``64(1 - e^-1) ~ 40.6``.

Standard library only; runs the full scan in well under a minute.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm

M = 64
D = M * lcm(*range(1, M + 1))  # every wall u/(M i) is an integer multiple of 1/D


def wall_numerators() -> list[int]:
    """All reduced-or-not rationals u/(M i), 0 <= u < M i, i <= M, plus 1."""
    nums = set()
    for i in range(1, M + 1):
        den = M * i
        step = D // den
        for u in range(den):
            nums.add(u * step)
    nums.add(D)
    return sorted(nums)


def q_star(p: int, q: int, m: int) -> int:
    """Largest convergent denominator <= m of p/q (best approximation)."""
    cf = []
    x, y = p, q
    while y:
        a = x // y
        cf.append(a)
        x, y = y, x - a * y
    # h_{-2}, h_{-1} / k_{-2}, k_{-1}
    hm2, hm1, km2, km1 = 0, 1, 1, 0
    last = 1
    for a in cf:
        h = a * hm1 + hm2
        k = a * km1 + km2
        hm2, hm1, km2, km1 = hm1, h, km1, k
        if k <= m:
            last = k
    return last


def brute_q_star(p: int, q: int, m: int) -> tuple[int, int]:
    """(q*, min_q ||q alpha||) by direct scan; used to validate q_star()."""
    best_q, best = 1, None
    for qq in range(1, m + 1):
        r = (qq * p) % q
        dist = min(r, q - r)
        if best is None or dist < best:
            best_q, best = qq, dist
    return best_q, best


def main() -> None:
    print("Phase criterion and the q* >= m/2 theorem at m = n = 64\n")
    walls = wall_numerators()
    cells = len(walls) - 1
    print(f"distinct walls (incl. 0 and 1): {len(walls)}, cells: {cells}")
    assert len(walls) == 80641 and cells == 80640, (len(walls), cells)

    half = M // 2
    len_qstar_hi = 0          # measure q* >= half  (q* over 1..m)
    len_qstar_hi63 = 0        # same, q* over 1..m-1
    len_occ_lo = 0            # measure Occ < half
    min_occ_hi = M + 1        # min Occ on q* >= half
    exceptions = 0            # q* >= half and Occ < half
    tight_cells = 0
    len_tight = 0
    weighted_occ = 0          # sum len * Occ  (numerator over D)
    checked = 0

    for a, b in zip(walls, walls[1:]):
        length = b - a
        p, q = a + b, 2 * D        # midpoint alpha = (a+b)/(2D)
        g = gcd(p, q)
        p, q = p // g, q // g
        keys = sorted((i * p) % q for i in range(1, M + 1))
        buckets = [key * M // q for key in keys]
        occ = len(set(buckets))
        same = sum(1 for x, y in zip(buckets, buckets[1:]) if x == y)
        assert occ == M - same, (Fraction(p, q), occ, same)
        # low occupancy <=> more than half of the gaps are phase-short
        assert (occ < half) == (same > half)

        qs = q_star(p, q, M)
        qs63 = q_star(p, q, M - 1)
        weighted_occ += length * occ
        if qs63 >= half:
            len_qstar_hi63 += length
        if qs >= half:
            len_qstar_hi += length
            min_occ_hi = min(min_occ_hi, occ)
            if occ < half:
                exceptions += 1
            if occ == half:
                tight_cells += 1
                len_tight += length
        if occ < half:
            len_occ_lo += length

        # validator: the convergent denominator really minimises ||q alpha||
        if checked < 200:
            _, brute = brute_q_star(p, q, M)
            r = (qs * p) % q
            assert min(r, q - r) == brute, (Fraction(p, q), qs)
        checked += 1

    print(f"cell-average Occ = {weighted_occ / D:.5f} "
          f"(= {weighted_occ} / D)")
    print(f"meas(q* >= {half})   = {Fraction(len_qstar_hi, D)} "
          f"= {len_qstar_hi / D:.6f}   (q* over 1..{M})")
    print(f"meas(q* >= {half})   = {len_qstar_hi63 / D:.6f}"
          f"   (q* over 1..{M - 1}; the §5.3 convention)")
    print(f"meas(Occ < {half})   = {Fraction(len_occ_lo, D)} "
          f"= {len_occ_lo / D:.6f}")
    print(f"min Occ on q* >= {half}: {min_occ_hi} (theorem: >= {half})")
    print(f"exceptions (q* >= {half} and Occ < {half}): {exceptions}")
    print(f"tight layer Occ = {half}: {tight_cells} cells, measure "
          f"{Fraction(len_tight, D)} = {len_tight / D:.7f}")

    # theorem-level assertions
    assert min_occ_hi == half
    assert exceptions == 0, "the E_m = empty claim failed"
    assert Fraction(len_tight, D) == Fraction(1, 128), Fraction(len_tight, D)
    assert tight_cells == 32, tight_cells
    assert 44.5 < weighted_occ / D < 44.6  # §5.1's E[Occ] ~ 44.53

    # Loose cross-checks against the published exact scan.  The q* measure is
    # convention-dependent (q <= m vs q <= m-1, and the tie-break among best
    # approximants), so only a band is asserted; the two conventions above span
    # 0.4546 .. 0.4624.
    assert 0.45 < len_qstar_hi / D < 0.47, len_qstar_hi / D
    assert 0.45 < len_qstar_hi63 / D < 0.47, len_qstar_hi63 / D
    assert abs(len_occ_lo / D - 0.1791) < 5e-3, len_occ_lo / D
    print("\nALL CHECKS PASSED")
    print(f"  Occ = N - Z holds on all {cells} cells, and Occ < 32 <=> Z > 32.")
    print("  q* >= 32 forces Occ >= 32 with no exception: the m/2 criterion "
          "is a theorem,")
    print("  unlike the threshold-52 statistic of §5.3, which is empirical.")
    print("  The mean over uniform alpha (44.53) is the uniform-alpha "
          "expectation,")
    print("  not the i.i.d. value 64(1-1/e) ~ 40.6.")


if __name__ == "__main__":
    main()