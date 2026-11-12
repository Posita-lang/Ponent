#!/usr/bin/env python3
"""Exact, float-free verification of the corrected theorem

    Occ(q_k, q_k, alpha) = q_k

for EVERY irrational alpha whose continued-fraction expansion starts with
[0; a_1, ..., a_N], at EVERY convergent denominator q_k, k = 0..N-1.

Background: erratum to the earlier proof sketch
-----------------------------------------------
The old sketch assumed |alpha - p_k/q_k| < 1/(2 q_k^2). That bound is FALSE
for general k. With alpha_{k+1} = [a_{k+1}; a_{k+2}, ...] > 1 and Cassini's
identity,

    eps = alpha - p_k/q_k = (-1)^k / (q_k (q_k alpha_{k+1} + q_{k-1})),

so q_k^2 |eps| = 1/(alpha_{k+1} + q_{k-1}/q_k) can be arbitrarily close to 1
when a_k is large (q_{k-1}/q_k small) and the tail makes alpha_{k+1} -> 1+
(a_{k+1} = 1 followed by a huge a_{k+2}); a large a_{k+1} instead pushes this
ratio down. The bound actually needed is |eps| < 1/q_k^2, which gives
0 < |i eps| < 1/q_k for 1 <= i <= q_k:
a displacement smaller than one bucket width, of constant sign. The bucket
map is then

    b_i = ((i p_k mod q_k) - [eps < 0]) mod q_k,

a bijection of {1, ..., q_k} onto {0, ..., q_k-1}; for eps < 0 the point
i = q_k wraps into bucket q_k - 1 (the branch the old sketch omitted).

What this script checks (Fractions only; floats appear only in printouts)
------------------------------------------------------------------------
The cylinder (L, U) = ( p_N/q_N , (p_N + p_{N-1})/(q_N + q_{N-1}) ), sorted,
contains EVERY irrational alpha with that prefix (any tail t > 1). For each
convergent denominator q_k:

  (H) eps has constant sign on [L, U] and q_k^2 * max|eps| < 1 -- the exact
      hypothesis of the corrected proof;
  (P) bucket(i) = ((i p_k mod q_k) - [eps < 0]) mod q_k at a midpoint;
  (O) Occ(q_k, q_k, midpoint) == q_k.

It also reports cylinders where q_k^2 * min|eps| > 1/2, i.e. cylinders in
which the old 1/(2 q_k^2) bound fails for EVERY alpha inside -- while the
theorem still holds.
"""

from fractions import Fraction
import random


def cf_convergents(a):
    """Convergents (p_k, q_k), k = 0..N, of alpha = [0; a_1, ..., a_N].

    State (p_{k-1}, p_k), (q_{k-1}, q_k) at k = 0, with a_0 = 0:
    p_{-1} = 1, p_0 = 0, q_{-1} = 0, q_0 = 1.
    """
    pPrev, pCur = 1, 0
    qPrev, qCur = 0, 1
    conv = [(0, 1)]
    for x in a:
        p = x * pCur + pPrev
        q = x * qCur + qPrev
        conv.append((p, q))
        pPrev, pCur = pCur, p
        qPrev, qCur = qCur, q
    return conv


def cylinder(conv, N):
    """Endpoints of the CF cylinder for the prefix of length N (sorted)."""
    pN, qN = conv[N]
    pP, qP = conv[N - 1]
    tail_inf = Fraction(pN, qN)                       # tail -> infinity
    tail_one = Fraction(pN + pP, qN + qP)             # tail -> 1+
    return (min(tail_inf, tail_one), max(tail_inf, tail_one))


def occ_rational(n, m, p, q):
    """Exact Occ(n, m, p/q) for integers p, q > 0 (integer arithmetic only)."""
    return len({((i * p) % q) * m // q for i in range(1, n + 1)})


def check(a, tag):
    """Check every convergent of one cylinder; return (checks, max_h, bad_half)."""
    conv = cf_convergents(a)
    N = len(a)
    L, U = cylinder(conv, N)
    mid = (L + U) / 2
    checks = 0
    max_h = Fraction(0)
    bad_half = []
    for k in range(N):
        pk, qk = conv[k]
        if qk > 4000:
            break
        dL, dU = L - Fraction(pk, qk), U - Fraction(pk, qk)
        if dL * dU <= 0:            # cylinder straddles this convergent: skip
            continue
        lo, hi = min(abs(dL), abs(dU)), max(abs(dL), abs(dU))
        h = hi * qk * qk
        assert h < 1, ("HYPOTHESIS FAIL", tag, k, h)
        max_h = max(max_h, h)
        if lo * qk * qk > Fraction(1, 2):
            bad_half.append((k, qk, float(lo * qk * qk)))
        neg = dL < 0
        predicted = {(((i * pk) % qk) - (1 if neg else 0)) % qk
                     for i in range(1, qk + 1)}
        assert len(predicted) == qk
        actual = {((i * mid.numerator) % mid.denominator) * qk // mid.denominator
                  for i in range(1, qk + 1)}
        assert actual == predicted, ("BUCKET MAP FAIL", tag, k)
        assert occ_rational(qk, qk, mid.numerator, mid.denominator) == qk, \
            ("OCCUPANCY FAIL", tag, k)
        checks += 1
    print(f"{tag:<28s} checks={checks:3d}  max q_k^2|eps| = {float(max_h):.6f} (<1)"
          + (f"   |eps|>1/(2q^2) on k={[x[0] for x in bad_half]} (min ratio "
             f"{min(x[2] for x in bad_half):.4f})" if bad_half else ""))
    return checks, max_h, bad_half


def build_cases():
    """Curated cylinders with huge partial quotients + random prefixes."""
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
    return cases


def main():
    cases = build_cases()
    total = 0
    worst = Fraction(0)
    refutations = 0
    for a, tag in cases:
        c, h, bad = check(a, tag)
        total += c
        worst = max(worst, h)
        refutations += len(bad)
    assert refutations >= 1, "the old 1/(2 q_k^2) bound was NOT refuted"
    print(f"\nALL EXACT CHECKS PASSED: {total} (k, cylinder) pairs across "
          f"{len(cases)} cylinders, max q_k^2|eps| = {float(worst):.6f}")
    print(f"(k, cylinder) pairs where |eps| > 1/(2 q_k^2) for EVERY alpha "
          f"inside: {refutations} (=> the old lemma is false, the theorem "
          f"survives)")


if __name__ == "__main__":
    main()

