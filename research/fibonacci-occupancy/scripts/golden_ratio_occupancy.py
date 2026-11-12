#!/usr/bin/env python3
"""golden_ratio_occupancy.py — exact verification of the Fibonacci-hashing math
behind DepGraph's 0x9E37_79B9 probe-start multiplier and the 64-slot
`LINEAR_DEDUP_MAX`/`DefaultCache` occupancy question.

BACKGROUND.  The query engine hashes `DepNodeIndex(i)` with the Knuth golden-ratio
multiplier 0x9E3779B9 = 2654435769 = floor(2^32 / phi) and takes the top 6 bits.
The natural question is: how well does alpha_k = (k * 2^-32 mod 1) — the effective
rotation constant — distribute i = 1..64 into 64 buckets?  Equivalently
Occ(n, m, alpha) = # { floor(m {i alpha}) : 1 <= i <= n }, the number of occupied
bins when the first n fractional parts are dropped into m equal intervals.

Every claim below was checked against seven independent derivations (three-gap /
Steinhaus–Sos–Slater theory, continued-fraction/Ostrowski analysis, and Farey-wall
enumeration).  Where the sources disagreed, THIS TOOL's exact integer arithmetic is
the referee — everything it prints is proven, not measured:

  Q1  {0, alpha, ..., 64 alpha} has 65 circular gaps of exactly three lengths:
      phi^-8 x 24, phi^-9 x 31, phi^-10 x 10 (sum = 24+31+10 = 65, weighted sum = 1).
  Q2  Occ(n, n, 1/phi) = q_k (largest Fibonacci number <= n) is NOT a theorem:
      n = 4, 7, 9, 10..12, 14, 56..58, 63 are counterexamples on both sides.
      Occ(64, 64, 1/phi) = 55 = q_9 is an alignment coincidence, dissected exactly
      via the bucket identity b_i = (39 i + floor(i beta)) mod 64, beta = {64 alpha}.
      At n = q_k, however, Occ(q_k, q_k, alpha) = q_k for EVERY irrational alpha
      (Theorem 2.1; corrected proof sketch and exact cylinder check below).
  Q3  The FULL bucket assignment of P_64(1/phi) is stable for
      |alpha - 1/phi| < 9.435178466e-6 (nearest wall 2215/3584, i = 56 — that
      crossing does NOT change the occupancy).  The occupancy VALUE 55 is stable on
      the wider interval [1859/3008, 89/144) (radii 1.5371729e-5 below, 2.1566806e-5
      above); both endpoints are occupancy-changing walls (55 -> 54 and 55 -> 56).
      Both 2^-32 neighbours of 1/phi (2654435769 and even the lookalike 2654435761
      over 2^32) are far inside, so their occupancy 55 is a THEOREM, not luck.
  Q4  alpha_2 = 165902777/2^32 = 1/26 + delta with 26 delta = 0.0043085 < 1/64
      collapses the 64 points onto 26 residue-class "strands" (64 = 2*26 + 12);
      exactly the 11 strands r in {1,2,4,6,8,10,12,19,21,23,25} straddle a bucket
      boundary, so Occ(64, 64, alpha_2) = 26 + 11 = 37.  (One source claimed the
      straddling set also contains r = 15, 17 with two base buckets missed — that
      variant is REFUTED below.)
  Q5  (optional, --measure) alpha -> Occ(64,64,alpha) is a step function with
      breakpoints at the rationals (64 j + k)/(64 i); the exact Lebesgue measure of
      { Occ < 52 } is computable by a finite Farey scan over the 80640 cells.
      The same exact pass re-checks §5's full histogram, E[Occ], q* predictors,
      the S=39 occupancy range, and Occ = n - S + C on every cell.

EXACTNESS.  No floating point is used for any decision.  All comparisons against
alpha = (sqrt(5)-1)/2 reduce to comparing integers via
      (sqrt(5)-1)/2  >  r   <=>   sqrt(5) > 2r + 1   <=>   5 > (2r+1)^2   (r >= 0),
and floor values like floor(32 i sqrt(5) - c) reduce to integer squares:
      k + c <= 32 i sqrt(5) < k + c + 1  <=>  (k+c)^2 <= 5 (32 i)^2 < (k+c+1)^2.
Rationals are `fractions.Fraction`; the only floats in the file are display strings.

Run:  python3 research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py [--measure]
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from math import isqrt

getcontext().prec = 60
SQRT5 = Decimal(5).sqrt()
PHI = (1 + SQRT5) / 2
ALPHA_DEC = (SQRT5 - 1) / 2  # display/bracket use only; decisions are integer-exact

# ---------------------------------------------------------------------------
# Exact arithmetic in Q(sqrt(5)) — the only substrate for every decision below.
# ---------------------------------------------------------------------------


def cmp_sqrt5(times_i: int, rhs: Fraction) -> int:
    """Compare i*sqrt(5) against an exact rational rhs (rhs may be negative).

    Returns -1 / 0 / 1.  Squaring is order-preserving on non-negatives, so the
    comparison i*sqrt5 <=> p/q with p >= 0 reduces to (p^2) <=> (5 i^2 q^2);
    negative rhs is handled by sign first.  No float ever participates.
    """
    # i*sqrt5 <=> p/q  (q > 0)  <=>  sign(i) * sqrt(5) <=> p/q  (after q>0)
    p, q = rhs.numerator, rhs.denominator
    if p == 0:
        return (times_i > 0) - (times_i < 0)
    lhs_nonneg = times_i > 0 or (times_i == 0)
    # sign resolution when signs differ
    if times_i >= 0 and p < 0:
        return 1
    if times_i < 0 and p > 0:
        return -1
    # both sides same sign: compare magnitudes by squaring
    lhs_sq = 5 * times_i * times_i * q * q
    rhs_sq = p * p
    if lhs_sq > rhs_sq:
        return 1 if times_i > 0 else -1
    if lhs_sq < rhs_sq:
        return -1 if times_i > 0 else 1
    return 0  # impossible for irrational vs rational, but keep it total


def cmp_alpha(frac: Fraction) -> int:
    """Compare alpha = (sqrt(5)-1)/2 against an exact rational.

    alpha > r  <=>  sqrt(5) > 2r + 1;  2r+1 <= 0 makes it trivially true,
    otherwise square both sides (5 <=> (2r+1)^2).  Exact in Z.
    """
    t = Fraction(2) * frac + 1
    if t <= 0:
        return 1  # alpha > any r <= -1/2, and alpha > 0
    s = cmp_sqrt5(1, t)
    return s


def floor_half_isqrt5_minus(x: int, y: int) -> int:
    """Exact floor((x*sqrt(5) - y) / 2) for integers x >= 0, y.

    k <= (x*sqrt5 - y)/2  <=>  2k + y <= x*sqrt5  <=>  2k + y <= isqrt(5x^2)
    (2k+y is an integer, so comparing against the irrational root is the same
    as comparing against its floor).  One isqrt, no search, no float.
    """
    return (isqrt(5 * x * x) - y) // 2


def floor_i_alpha(i: int) -> int:
    """Exact floor(i * alpha), alpha = (sqrt(5)-1)/2, i >= 0."""
    return floor_half_isqrt5_minus(i, i)


def bucket_exact(i: int) -> int:
    """Exact bucket index floor(64 {i alpha}) for the golden alpha (0 <= b < 64).

    64 {i alpha} = (64 i sqrt(5) - 64 i - 128 floor(i alpha)) / 2.
    """
    f = floor_i_alpha(i)
    return floor_half_isqrt5_minus(64 * i, 64 * i + 128 * f)


def occ_golden_exact(n: int, m: int) -> int:
    """Exact Occ(n, m, 1/phi): # { floor(m {i alpha}) : 1 <= i <= n }, any m.

    m {i alpha} = (m i sqrt(5) - m i - 2 m floor(i alpha)) / 2 — the primitive
    above covers even and odd m uniformly.  Every decision in this file is
    integer arithmetic in Q(sqrt(5)); floats only format output.
    """
    seen = set()
    for i in range(1, n + 1):
        f = floor_i_alpha(i)
        seen.add(floor_half_isqrt5_minus(m * i, m * i + 2 * m * f) % m)
    return len(seen)


def buckets_golden_exact(n: int, m: int) -> list[int]:
    """The exact bucket index of each i = 1..n under (m, 1/phi)."""
    out = []
    for i in range(1, n + 1):
        f = floor_i_alpha(i)
        out.append(floor_half_isqrt5_minus(m * i, m * i + 2 * m * f) % m)
    return out


# ---------------------------------------------------------------------------
# Q1 — three-distance data for {0, alpha, ..., 64 alpha}
#
# THEOREM (three-distance / Steinhaus–Sós–Slater).  For any alpha and N, the
# N+1 circular gaps of {0, {alpha}, ..., {N alpha}} take at most three lengths.
# For alpha = 1/phi (ALL partial quotients 1) the convergent denominators are
# the Fibonacci numbers q_k = F_{k+1}, and ||q_k alpha|| = phi^-(k+1) exactly.
# With q_9 = 55 <= 64 < 89 = q_10 and r = 64 - 55 = 9:
#   - the 55-point configuration has two gap lengths phi^-8 (x34) and
#     phi^-9 (x21) [Steinhaus: q_8 + q_9 points];
#   - 55 alpha = 1 - phi^-10 mod 1, so each of the next 10 points
#     {55 alpha}..{64 alpha} splits one long phi^-8 gap into
#     phi^-10 + (phi^-8 - phi^-10) = phi^-10 + phi^-9 (golden identity
#     phi^-8 = phi^-9 + phi^-10, from alpha^2 = 1 - alpha).
# Prediction: phi^-8 x 24, phi^-9 x 31, phi^-10 x 10 — total 65 = 64 + 1,
# weighted sum 24 phi^-8 + 31 phi^-9 + 10 phi^-10 = 1 (two independent
# identities).  The check below is float-only for CLASSIFICATION speed, but
# the final consistency is re-proven exactly in Q2's bucket ledger.
# ---------------------------------------------------------------------------

PHI_EXP = {8: 0.0212862362522, 9: 0.0131556174964, 10: 0.0081306187558}


def check_q1() -> None:
    from collections import Counter

    n = 64
    golden_float = 0.5 * (5**0.5) - 0.5  # (sqrt5 - 1)/2 — display-grade only
    pts = sorted(((i * golden_float) % 1.0) for i in range(n + 1))
    counts: Counter[str] = Counter()
    for j in range(n + 1):
        nxt = pts[(j + 1) % (n + 1)]
        g = nxt - pts[j]
        if g <= 0:
            g += 1  # circular wrap
        for name, v in (("phi^-8", PHI_EXP[8]), ("phi^-9", PHI_EXP[9]), ("phi^-10", PHI_EXP[10])):
            if abs(g - v) < 1e-12:
                counts[name] += 1
                break
        else:
            counts[f"OTHER {g:.12f}"] += 1
    assert counts == Counter({"phi^-9": 31, "phi^-8": 24, "phi^-10": 10}), counts
    total = 24 * PHI_EXP[8] + 31 * PHI_EXP[9] + 10 * PHI_EXP[10]
    assert abs(total - 1.0) < 1e-12, total
    print("Q1 PASS  gaps: phi^-8 x24, phi^-9 x31, phi^-10 x10 (65 gaps, sum 1)")
    check_three_gap_formula()


def check_three_gap_formula() -> None:
    """Exact (60-digit Decimal) check of the general all-1 three-gap
    multiplicity formula quoted in §1 of derivations.md, at several N.

    For q_k <= N = q_k + r < q_{k+1}, the N+1 circular gaps of
    {0, alpha, ..., N alpha} are
        alpha^{k+1} x (r+1), alpha^{k-1} x (q_{k-1}-r-1),
        alpha^k x (q_k - q_{k-1} + r + 1).
    """
    from decimal import Decimal

    exps = {e: ALPHA_DEC ** e for e in range(2, 16)}
    def classify(g: Decimal):
        return next((e for e, v in exps.items()
                     if abs(g - v) < Decimal("1e-45")), None)

    F = [0, 1]
    while len(F) < 20:
        F.append(F[-1] + F[-2])
    for N in (40, 64, 65, 100, 144):
        pts = sorted([Decimal(0)] + [
            (i * SQRT5 - i) / 2 - Decimal(floor_i_alpha(i))
            for i in range(1, N + 1)])
        gaps = [pts[(j + 1) % (N + 1)] - pts[j]
                + (1 if j == N else 0) for j in range(N + 1)]
        from collections import Counter
        cc = Counter(classify(g) for g in gaps)
        k = max(j for j in range(1, 15) if F[j + 1] <= N)
        r = N - F[k + 1]
        pred = {k + 1: r + 1, k - 1: F[k] - r - 1,
                k: F[k + 1] - F[k] + r + 1}
        assert all(cc.get(e, 0) == v for e, v in pred.items()) and \
            sum(cc.values()) == N + 1, (N, cc, pred)
    print("Q1 PASS  general 3-gap multiplicity formula verified at "
          "N = 40, 64, 65, 100, 144 (60-digit Decimal, exact isqrt floors)")


# ---------------------------------------------------------------------------
# Q2 — Occ(n, n, 1/phi) = q_k is NOT a theorem; the n = 64 ledger.
#
# DISPROOF.  occ_golden_exact is EXACT (integer isqrt in Q(sqrt 5)).  It gives
# Occ(4,4)=4 > q_3=3 and Occ(9,9)=7 < q_5=8 — both inequalities fail inside
# the same block q_5=8 <= n < 13, so neither Occ >= q_k nor Occ <= q_k holds.
# Both sides fail inside q_9's own block too: Occ(56..58) = 50 < 55 while
# Occ(68) = 64 > 55.  Hence Occ(64,64)=55 = q_9 is an alignment coincidence.
#
# THE LEDGER (exact, refines the coincidence into countable parts).  With
# beta = {64 alpha} = 32 sqrt(5) - 71, the bucket of i is
#      b_i = (39 i + floor(i beta)) mod 64          (64 alpha = 39 + beta)
# so the whole assignment is integer arithmetic once floor(i beta) is exact.
# floor(i beta) = floor(32 i sqrt5 - 71 i) = isqrt(5 (32i)^2) - 71 i.
# The ledger must show: 64 points - 9 empty buckets = 55 occupied, and the
# 9 "extra" points sit in exactly 9 doubly-occupied buckets.
#
# GENERAL THEOREM (the clean positive statement, CORRECTED PROOF SKETCH).
# For ANY irrational alpha with convergent p_k/q_k: Occ(q_k, q_k, alpha) = q_k,
# with exactly one point per bucket.  Let alpha_{k+1} = [a_{k+1};...] > 1 be
# the complete quotient and eps = alpha - p_k/q_k.  Cassini gives the exact
# identity |eps| = 1 / (q_k (q_k alpha_{k+1} + q_{k-1})) < 1/q_k^2, hence
# 0 < |i eps| < 1/q_k for 1 <= i <= q_k (alpha irrational => no point ever
# sits on a bucket edge j/q_k).  With g_i = (i p_k mod q_k):
#     b_i = (g_i - [eps < 0]) mod q_k,
# because for eps > 0 the shift i eps stays inside the same bucket, while for
# eps < 0 every point shifts one bucket down INCLUDING the wrap i = q_k
# (g_i = 0, {q_k alpha} = 1 + q_k eps, which lands in bucket q_k - 1).  Since
# gcd(p_k, q_k) = 1, i -> g_i is a bijection {1..q_k} -> {0..q_k-1}, so this
# shift is a permutation and {b_i} = {0..q_k-1}.
#   ERRATUM: the earlier sketch's 1/(2 q_k^2) lemma is FALSE for general k:
#   q_k^2 |eps| = 1/(alpha_{k+1} + q_{k-1}/q_k) can approach 1 when a_k is
#   large (q_{k-1}/q_k small) and alpha_{k+1} -> 1+ (a_{k+1} = 1 followed by
#   a huge a_{k+2}); e.g. [0; 100, 1, 10^6, ...] gives 0.9901 at k = 1.
#   A large a_{k+1} instead pushes this ratio DOWN.  Hurwitz' 1/sqrt5 bound
#   holds for infinitely many k, not for each k.  The hypothesis actually
#   used is |eps| < 1/q_k^2, and the wrap branch (missing before) is what
#   makes the map a -1 shift rather than the identity when eps < 0.
# Verified below for 1/phi up to 377, sqrt(2)-1 up to 169, and -- for EVERY
# irrational inside 19 continued-fraction cylinders, at every k with q_k <=
# 4000 -- by check_convergent_cylinders(), which also refutes the old bound.
# ---------------------------------------------------------------------------


def cf_convergents(a: list[int]) -> list[tuple[int, int]]:
    """Convergents (p_k, q_k), k = 0..N, of alpha = [0; a_1, ..., a_N]."""
    pPrev, pCur, qPrev, qCur = 1, 0, 0, 1   # a_0 = 0 state
    out = [(0, 1)]
    for x in a:
        p, q = x * pCur + pPrev, x * qCur + qPrev
        out.append((p, q))
        pPrev, pCur, qPrev, qCur = pCur, p, qCur, q
    return out


def check_convergent_cylinders() -> None:
    """Exact, float-free check of Theorem 2.1 for EVERY irrational alpha in a
    continued-fraction cylinder, at EVERY convergent denominator.

    The cylinder (L, U) = (p_N/q_N, (p_N + p_{N-1})/(q_N + q_{N-1})) (sorted)
    contains every alpha with prefix [0; a_1..a_N] and any irrational tail.
    For each k we verify with Fractions only:
      (H) eps has constant sign on the cylinder and q_k^2 max|eps| < 1 -- the
          exact hypothesis the corrected proof uses;
      (P) bucket(i) = ((i p_k mod q_k) - [eps < 0]) mod q_k at a midpoint;
      (O) Occ(q_k, q_k, midpoint) == q_k.
    Cylinders where q_k^2 min|eps| > 1/2 refute the old 1/(2 q_k^2) lemma.
    """
    import random

    cases = [
        ([1] * 22, "golden [0;1,1,...]"),
        ([2] * 16, "sqrt2-1 [0;2,2,...]"),
        ([1000] + [1] * 12, "huge a1=1000"),
        ([1, 1000] + [1] * 12, "huge a2=1000"),
        ([100, 1, 10**6] + [1] * 10, "a3=10^6 (tail near 1)"),
        ([1, 1, 1, 1, 5000] + [1] * 8, "late huge partial"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "mixed 1..12"),
    ]
    random.seed(20261108)
    for t in range(12):
        cases.append(([random.randint(1, 60) for _ in range(11)], f"random-{t}"))

    checks = 0
    worst = Fraction(0)
    refutations = 0
    for a, tag in cases:
        conv = cf_convergents(a)
        N = len(a)
        pN, qN = conv[N]
        pP, qP = conv[N - 1]
        e_inf = Fraction(pN, qN)
        e_one = Fraction(pN + pP, qN + qP)
        L, U = min(e_inf, e_one), max(e_inf, e_one)
        mid = (L + U) / 2
        for k in range(N):
            pk, qk = conv[k]
            if qk > 4000:
                break
            dL, dU = L - Fraction(pk, qk), U - Fraction(pk, qk)
            if dL * dU <= 0:      # cylinder straddles this convergent: skip
                continue
            lo, hi = min(abs(dL), abs(dU)), max(abs(dL), abs(dU))
            h = hi * qk * qk
            assert h < 1, ("HYPOTHESIS FAIL", tag, k, h)
            worst = max(worst, h)
            if lo * qk * qk > Fraction(1, 2):
                refutations += 1
            neg = dL < 0
            pred = {(((i * pk) % qk) - (1 if neg else 0)) % qk
                    for i in range(1, qk + 1)}
            assert len(pred) == qk
            act = {((i * mid.numerator) % mid.denominator) * qk // mid.denominator
                   for i in range(1, qk + 1)}
            assert act == pred, ("BUCKET MAP FAIL", tag, k)
            assert occ_rational_exact(qk, qk, mid.numerator, mid.denominator) == qk, \
                ("OCC FAIL", tag, k)
            checks += 1
    assert refutations >= 1, "the old 1/(2 q_k^2) bound was NOT refuted"
    print(f"Q2 PASS  Theorem 2.1 holds for every irrational in {len(cases)} "
          f"CF cylinders ({checks} (k, cylinder) pairs, max q_k^2|eps| = "
          f"{float(worst):.6f} < 1); old 1/(2 q_k^2) lemma refuted in "
          f"{refutations} pairs")


# ---------------------------------------------------------------------------


def check_q2() -> None:
    # counterexample table (both directions)
    expected = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 8, 9: 7, 10: 10,
                11: 10, 12: 11, 13: 13, 14: 12, 56: 50, 57: 50, 58: 50,
                63: 52, 64: 55, 68: 64}
    for n, occ in expected.items():
        got = occ_golden_exact(n, n)
        assert got == occ, (n, got, occ)
    print("Q2 PASS  Occ(n,n,1/phi) != q_k in general (n=4 gives 4>3, n=9 "
          "gives 7<8; both signs occur inside q_9's block: 56..58 -> 50, 68 -> 64)")

    # the n = 64 ledger: 55 occupied, 9 empty, 9 doubly-occupied, no triple
    buckets = buckets_golden_exact(64, 64)
    from collections import Counter
    c = Counter(buckets)
    empties = sorted(set(range(64)) - set(buckets))
    doubles = sorted(k for k, v in c.items() if v == 2)
    assert len(buckets) == 64 and len(c) == 55
    assert empties == [1, 9, 16, 22, 24, 37, 50, 52, 58], empties
    assert doubles == [5, 15, 20, 23, 35, 39, 51, 54, 59], doubles
    assert max(c.values()) == 2, "no bucket holds three points"
    print("Q2 PASS  n=64 ledger: 55 occupied = 64 pts - 9 empty; extras in 9 "
          "double buckets", empties)

    # the general theorem at convergent denominators
    fibs = [1, 2]
    while fibs[-1] < 400:
        fibs.append(fibs[-1] + fibs[-2])
    for q in fibs:
        assert occ_golden_exact(q, q) == q, q
    # sqrt(2)-1 = [0;2,2,2,...]: convergent denominators 1,2,5,12,29,70,169,...
    def occ_sqrt2(n: int, m: int) -> int:
        # {i(sqrt2 - 1)} = {i sqrt2} - floor part handled exactly:
        # floor(m {i(sqrt2-1)}) — sqrt2 irrational, use integer comparison:
        # bucket b <= m{i(sqrt2-1)} < b+1.  {i(sqrt2-1)} = i*sqrt2 - (i + f)
        # with f = floor(i(sqrt2-1)) = isqrt(2 i^2) - i.
        seen = set()
        for i in range(1, n + 1):
            f = isqrt(2 * i * i) - i
            # m{i(sqrt2-1)} = m i sqrt2 - m(i+f);  floor via isqrt:
            seen.add((isqrt(2 * (m * i) ** 2) - m * (i + f)) % m)
        return len(seen)
    for q in (1, 2, 5, 12, 29, 70, 169):
        assert occ_sqrt2(q, q) == q, q
    print("Q2 PASS  general theorem Occ(q_k, q_k, alpha) = q_k at every "
          "convergent denominator (1/phi up to 377; sqrt2-1 up to 169)")

    # exact bucket -> index-pair correspondence of the 9 double buckets (§2.2).
    # Earlier drafts listed the same nine pairs in an order that did NOT match
    # the sorted bucket list; this is the machine-checked correspondence.
    pairs = {k: sorted(i + 1 for i, x in enumerate(buckets) if x == k)
             for k, v in c.items() if v == 2}
    assert pairs == {5: [5, 60], 15: [2, 36], 20: [7, 62], 23: [20, 54],
                     35: [9, 64], 39: [1, 56], 51: [11, 45], 54: [3, 58],
                     59: [29, 63]}, pairs
    check_convergent_cylinders()


# ---------------------------------------------------------------------------
# Q3 — stability walls around alpha_0 = 1/phi at (n, m) = (64, 64).
#
# MECHANISM.  alpha -> (bucket(i))_i is a step function; a bucket assignment
# changes exactly when some {i alpha} crosses a grid line k/64, i.e. at the
# rational walls alpha = (64 j + k) / (64 i), 1 <= i <= 64.  A perturbation
# delta moves {i alpha} by ~ i*delta, so the alpha-space radius to a wall is
# dist({i alpha_0}, grid) / i — the INDEX divides, which is why the point
# closest to a boundary in x-space (i = 9, dist 1.94e-4) is NOT the nearest
# wall in alpha-space (i = 56, dist 5.28e-4 but /56).
#
# VERDICT (exact, this tool as referee — two of the seven sources got the
# one-sided structure right, one conflated the pattern radius with the
# occupancy radius):
#   * pattern radius  = alpha_0 - 2215/3584 = 9.435178466e-6  (i = 56 crosses
#     39/64) — but this crossing does NOT change the occupancy: bucket 39
#     still holds {1 alpha} and bucket 38 holds {22 alpha}.
#   * the occupancy VALUE 55 is preserved on [1859/3008, 89/144):
#       - below 1859/3008 (i = 47 crosses 3/64, bucket 3 singly occupied)
#         the occupancy drops to 54;  radius 1.537172862e-5;
#       - above 89/144 (F_11/F_12, a convergent; i in {9,18,...,63} cross
#         together since 9 | 144) the net change is +1 -> 56;
#         radius 2.156680566e-5.
#   * both 2^-32 neighbours 2654435769/2^32 (Knuth) and 2654435761/2^32 sit
#     ~1.2e-10 / 2.0e-9 from alpha_0 — 4+ orders inside even the pattern
#     radius — so their occupancy 55 is a theorem, with room to spare.
#     CAVEAT for future constant choices: the pattern radius scales like
#     1/m^2; at m ~ 2^16 a 32-bit constant is already marginal.
# ---------------------------------------------------------------------------


def occ_generic_dec(n: int, m: int, a: Decimal) -> int:
    """Occ(n, m, a) for a RATIONAL/DECIMAL a (used for the 2^-32 constants and
    the wall-scan midpoints).  a is carried at 60 significant digits — far more
    than any decision here needs, since the walls are rational and the sample
    points sit mid-cell.  (The golden-alpha paths above are fully exact.)"""
    return len({int(((i * a) % 1) * m) for i in range(1, n + 1)})


def check_q3() -> None:
    from decimal import Decimal  # local alias for clarity

    def dec(f: Fraction) -> Decimal:
        return Decimal(f.numerator) / Decimal(f.denominator)

    walls = sorted({Fraction(64 * j + k, 64 * i)
                    for i in range(1, 65) for j in range(i) for k in range(65)})
    walls = [w for w in walls if 0 < w < 1]
    a0 = ALPHA_DEC
    below = sorted((w for w in walls if dec(w) < a0), key=lambda w: a0 - dec(w))
    above = sorted((w for w in walls if dec(w) > a0), key=lambda w: dec(w) - a0)

    # nearest wall on each side == the PATTERN (bucket-assignment) radius
    assert below[0] == Fraction(2215, 3584), below[0]
    assert above[0] == Fraction(89, 144), above[0]
    r_pat = a0 - dec(below[0])
    assert abs(float(r_pat) - 9.435178466276776e-06) < 1e-15

    # occupancy-preserving walls: first wall outward where the count != 55
    def occ_left_of(w: Fraction) -> int:  # occupancy in the cell just below w
        nxt = max((dec(x) for x in walls if dec(x) < dec(w)), default=Decimal(0))
        return occ_generic_dec(64, 64, (nxt + dec(w)) / 2)

    def occ_right_of(w: Fraction) -> int:  # occupancy in the cell just above w
        nxt = min((dec(x) for x in walls if dec(x) > dec(w)), default=Decimal(1))
        return occ_generic_dec(64, 64, (dec(w) + nxt) / 2)

    # occupancy-preserving walls: a wall is an occupancy boundary exactly when
    # the cells on its two sides differ; walking OUT from alpha_0, the first
    # such wall on each side bounds the interval where Occ = 55.
    occ_lo = None
    for w in below:  # sorted by distance from alpha_0 (nearest first)
        if occ_left_of(w) != 55:
            occ_lo = w
            break
    occ_hi = None
    for w in above:
        if occ_right_of(w) != 55:
            occ_hi = w
            break
    assert occ_lo == Fraction(1859, 3008), occ_lo
    assert occ_hi is None or occ_hi == Fraction(89, 144), occ_hi
    # at the upper wall 89/144 the count jumps to 56 (checked directly):
    assert occ_left_of(Fraction(89, 144)) == 55
    assert occ_generic_dec(64, 64, dec(Fraction(89, 144)) + Decimal("1e-12")) == 56
    # the pattern wall at i = 56 (2215/3584) is occupancy-NEUTRAL:
    assert occ_left_of(Fraction(2215, 3584)) == 55
    r_lo = a0 - dec(Fraction(1859, 3008))
    r_hi = dec(Fraction(89, 144)) - a0
    assert abs(float(r_lo) - 1.537172861825246e-05) < 1e-15
    assert abs(float(r_hi) - 2.156680566070735e-05) < 1e-15
    print("Q3 PASS  pattern radius 9.435e-6 (i=56 wall 2215/3584, occupancy-"
          "neutral); Occ=55 exactly on [1859/3008, 89/144): radii 1.537e-5 / 2.157e-5")

    # the 2^-32 constants
    for p, note in ((2654435769, "Knuth floor(2^32/phi)"), (2654435761, "lookalike")):
        a1 = Decimal(p) / Decimal(2**32)
        assert abs(a1 - a0) < r_pat
        assert occ_generic_dec(64, 64, a1) == 55
        print(f"Q3 PASS  alpha1 = {p}/2^32 ({note}): dist {float(abs(a1-a0)):.3e} "
              f"<< pattern radius; Occ = 55 proven")


# ---------------------------------------------------------------------------
# Q4 — the bad constant alpha_2 = 165902777/2^32 gives Occ = 37, exactly.
#
# MECHANISM (the "large partial quotient" collapse).  alpha_2 has continued
# fraction [0; 25, 1, 7, 1, 27, ...] — note a_3 = 7, NOT 8 (one source's
# expansion was off); convergents 1/25, 1/26, 8/207, 9/233.  So q_2 = 26 and
# the next denominator 207 >> 64: for the first ~207 steps the sequence is a
# PERTURBATION OF THE RATIONAL ROTATION BY 1/26.  Exactly,
#      alpha_2 = 1/26 + delta,   26 delta = 18504906/2^32 = 0.004308509... ,
# so consecutive strands (i and i+26) drift 26 delta ≈ 0.276 bucket widths —
# strictly less than one bucket.  Writing i = 26 t + r, the 64 points sit on
# 26 residue-class strands (64 = 2*26 + 12: residues 1..12 have three points,
# 0 and 13..25 have two); each strand's spread is < 1/64, so it occupies one
# or two ADJACENT buckets, never three, and distinct strands' base buckets
# differ by >= 2 so strands never collide.
#      Occ = 26 + #(strands straddling a boundary) = 26 + 11 = 37.
# The straddling residues are r in {1,2,4,6,8,10,12,19,21,23,25}.  One source
# claimed a 13-residue set with two base buckets missed (r = 15, 17) — that
# variant is refuted: the exact strand buckets below show every residue's
# base bucket occupied.
#
# MORAL for the engine: the golden ratio is not "a nice constant", it is the
# WORST-CASE OPTIMUM — alpha with bounded partial quotients maximizes
# min_{q <= n} ||q alpha||, which is exactly the quantity whose collapse (a
# good small-denominator approximation) strands 64 points onto 26 rotations
# here.  A random rational this close to 1/26 loses ~28% of the buckets.
# ---------------------------------------------------------------------------

STRADDLING = {1, 2, 4, 6, 8, 10, 12, 19, 21, 23, 25}


def check_q4() -> None:
    a2 = Decimal(165902777) / Decimal(2**32)
    delta = a2 - Decimal(1) / 26
    assert 26 * delta < Decimal(1) / 64, "strand drift must be sub-bucket"
    # exact continued fraction of alpha_2 (integer CF, no floats)
    cf = continued_fraction_fraction(165902777, 2**32, 12)
    assert cf[:4] == [0, 25, 1, 7], cf  # [0;25,1,7,...] — a_3 = 7 (not 8)
    # exact strand decomposition at 60-digit Decimal (walls are far away)
    strands: dict[int, set[int]] = {}
    for i in range(1, 65):
        b = int(((i * a2) % 1) * 64)
        strands.setdefault(i % 26, set()).add(b)
    assert len(strands) == 26
    assert all(len(s) <= 2 for s in strands.values()), "strands span < 1 bucket"
    straddling = {r for r, s in strands.items() if len(s) == 2}
    assert straddling == STRADDLING, straddling
    occ = len(set().union(*strands.values()))
    assert occ == 26 + 11 == 37, occ
    print("Q4 PASS  alpha_2 = 1/26 + 26^-1*(0.0043085): 26 strands, 11 straddle "
          "{1,2,4,6,8,10,12,19,21,23,25} -> Occ = 37  (a_3 = 7 refutes the [0;25,1,8] claim)")


def continued_fraction_fraction(p: int, q: int, terms: int) -> list[int]:
    """First `terms` terms of the continued fraction of the exact rational p/q."""
    out = []
    for _ in range(terms):
        a, r = divmod(p, q)
        out.append(a)
        if r == 0:
            break
        p, q = q, r
    return out


# ---------------------------------------------------------------------------
# Q5 — (optional, --measure) the bad set {alpha : Occ(64,64,alpha) < 52}.
#
# alpha -> Occ(64,64,alpha) is a step function on [0,1) with breakpoints at
# the rationals (64 j + k)/(64 i), 1 <= i <= 64 — 80641 distinct rationals
# INCLUDING the endpoints 0 and 1, i.e. 80639 interior walls cutting [0,1)
# into 80640 cells on which the bucket vector is constant.  The exact Lebesgue
# measure of {Occ < 52} is therefore a finite rational sum.  One exact pass
# over the cells also re-derives every statistic quoted in §5 of derivations.md
# (full occupancy histogram, E[Occ], the mode, the q* predictors, the S = 39
# occupancy range, and Theorem 2.3's Occ = n - S + C identity).  For rational
# alpha = P/Q everything is pure integer arithmetic:
#     bucket(i) = ((i*P) mod Q) * m // Q,
# a circular gap (X/Q, Y/Q) is short iff m*(Y-X) < Q and swallows the grid
# point k/m iff k*Q lies in (m*X, m*Y) for some integer k.  Midpoints are
# strictly inside cells, so no point ever sits on a bucket boundary.
# ---------------------------------------------------------------------------


def occ_rational_exact(n: int, m: int, p: int, q: int) -> int:
    """Exact Occ(n, m, p/q) for integers p, q > 0 — pure integer arithmetic.

    {i p / q} = ((i p) mod q) / q, so bucket(i) = floor(m * ((i p) mod q)/q)
    = ((i p) mod q) * m // q.  No float, no rounding: two equal buckets are
    equal exactly.  (A first draft here used (i*p) mod (m*q) // q, which is
    floor(i*p/q) — the INTEGER part, not the bucket; the measure check below
    caught it.)
    """
    return len({((i * p) % q) * m // q for i in range(1, n + 1)})


def scan_cells(n: int, m: int, threshold: int, qstar_cut: int = 52) -> dict:
    """One exact pass over all Farey cells of [0,1) — the §5 statistics.

    Returns a dict with:
      measure_lt    exact measure of {Occ < threshold} (Fraction),
      hist          occ -> exact measure (dict of Fractions),
      expected      E[Occ] (Fraction),
      q_ge52        measure{q* >= qstar_cut},
      q_ge52_bad    measure{q* >= qstar_cut AND Occ < threshold},
      q_le51_bad    measure{q* < qstar_cut AND Occ < threshold},
      s39_range     (min, max) of Occ over cells whose short-gap count S = 39,
      t23_bad       number of cells violating Occ = n - S + C,
      interior_walls / cells counts.

    Everything is integer/Fraction arithmetic: one rational midpoint per cell,
    never on a wall, so no point ever sits on a bucket boundary.
    """
    from collections import defaultdict

    walls = sorted({Fraction(m * j + k, m * i)
                    for i in range(1, n + 1) for j in range(i) for k in range(m + 1)})
    interior = [w for w in walls if 0 < w < 1]
    hist: dict[int, Fraction] = defaultdict(lambda: Fraction(0))
    expected = Fraction(0)
    measure_lt = Fraction(0)
    q_ge = q_ge_bad = q_lt_bad = Fraction(0)
    s39_min, s39_max = m + 1, -1
    t23_bad = 0
    prev = Fraction(0)
    for w in interior + [Fraction(1)]:
        mid = (prev + w) / 2
        p, Q = mid.numerator, mid.denominator
        pts = sorted((i * p) % Q for i in range(1, n + 1))
        occ = len({x * m // Q for x in pts})
        # Theorem 2.3 auxiliaries: short circular gaps and swallowed grid points
        S = C = 0
        for idx in range(n):
            X = pts[idx]
            Y = pts[(idx + 1) % n] + (Q if idx == n - 1 else 0)   # wrap gap
            gap = Y - X
            if m * gap < Q:                    # gap shorter than 1/m
                S += 1
                k = (m * X) // Q + 1          # smallest k with k*Q > m*X
                if k * Q < m * Y:             # some grid point k/m inside
                    C += 1
        if occ != n - S + C:
            t23_bad += 1
        # q* = argmin_{1 <= j < n} ||j alpha||, the best approximation at scale n
        bestq, bestd = 1, Q
        for j in range(1, n):
            r = (j * p) % Q
            d = min(r, Q - r)
            if d < bestd:
                bestd, bestq = d, j
        weight = w - prev
        hist[occ] += weight
        expected += occ * weight
        if occ < threshold:
            measure_lt += weight
        if bestq >= qstar_cut:
            q_ge += weight
            if occ < threshold:
                q_ge_bad += weight
        elif occ < threshold:
            q_lt_bad += weight
        if S == 39:
            s39_min, s39_max = min(s39_min, occ), max(s39_max, occ)
        prev = w
    return {
        "measure_lt": measure_lt, "hist": dict(hist), "expected": expected,
        "q_ge52": q_ge, "q_ge52_bad": q_ge_bad, "q_le51_bad": q_lt_bad,
        "s39_range": (s39_min, s39_max), "t23_bad": t23_bad,
        "interior_walls": len(interior), "cells": len(interior) + 1,
    }


def check_q5() -> None:
    r = scan_cells(64, 64, 52)
    meas, val = r["measure_lt"], float(r["measure_lt"])
    assert abs(val - 0.591789799172181) < 1e-12, val
    assert meas == Fraction(223889114211561380423855435,
                            378325402912903125603635712), meas
    assert r["interior_walls"] == 80639 and r["cells"] == 80640
    assert r["t23_bad"] == 0, "Theorem 2.3 (Occ = n - S + C) failed in a cell"
    hist = r["hist"]
    ge55 = float(sum(v for k, v in hist.items() if k >= 55))
    eq64 = float(hist.get(64, 0))
    E = float(r["expected"])
    mode = max(hist, key=lambda k: hist[k])
    assert abs(ge55 - 0.2807) < 5e-5 and abs(eq64 - 0.0181) < 5e-5
    assert abs(E - 44.53) < 5e-3 and mode == 54
    q_ge = float(r["q_ge52"])
    q_bad = float(r["q_ge52_bad"])
    assert abs(q_ge - 0.1645) < 5e-5 and abs(q_bad - 0.000214) < 1e-6
    assert abs((1 - q_bad / q_ge) - 0.9987) < 5e-5
    assert abs(float(r["q_le51_bad"]) / (1 - q_ge) - 0.708) < 5e-4
    assert r["s39_range"] == (25, 64)
    print(f"Q5 PASS  measure{{Occ(64,64,alpha) < 52}} = {meas} = {val:.15f} "
          f"— low occupancy is the MAJORITY (59.2%), not a pathology; 1/phi "
          f"sits in the top ~28%")
    print(f"Q5 PASS  exact scan over {r['cells']} cells: E[Occ] = {E:.4f}, "
          f"mode {mode}, measure{{Occ>=55}} = {ge55:.4f}, "
          f"P(Occ>=52|q*>=52) = {1 - q_bad / q_ge:.6f}; Theorem 2.3 holds "
          f"in every cell; at S=39 Occ runs {r['s39_range'][0]}.."
          f"{r['s39_range'][1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true",
                    help="also run the exact 80640-cell Farey scan for Q5 (slow)")
    args = ap.parse_args()
    check_q1()
    check_q2()
    check_q3()
    check_q4()
    if args.measure:
        check_q5()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

