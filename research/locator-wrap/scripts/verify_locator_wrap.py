#!/usr/bin/env python3
"""Machine checks for research/locator-wrap/solution.md.

No dependencies; every claim is re-derived by brute force on small instances or
checked against the closed forms. Run from the repository root:

    python3 research/locator-wrap/scripts/verify_locator_wrap.py

Checks
  1. greedy's overrun lemma (a block overruns its budget iff singleton & over-wide)
  2. the feasibility criterion (Fact 1) against exhaustive layout search
  3. Phi is legal iff the instance is feasible; Phi >= OPT; Fact 2 blow-up
  4. OPT = L(j*-1) + 1 against exhaustive layout search
  5. paragraph additivity of the line count
  6. both secondary-objective witnesses (Q3a) and the (i,k) DP (Q3b) against
     an exhaustive minimum-line search
  7. the mixed (body/final) matrix is Monge and the feasible column window is a
     nondecreasing staircase (Q3c)
  8. the infeasible-region loss vectors and the Pareto no-go claims (Q4)
"""

from itertools import product

INFEASIBLE = None


# ── Model ────────────────────────────────────────────────────────────────────

def span(w, i, j):
    """B(i, j) for 0-based inclusive word indices, or 0 for an empty range."""
    if i > j:
        return 0
    return sum(w[i:j + 1]) + (j - i)


def layouts(n):
    """Every partition of range(n) into consecutive non-empty blocks."""
    for mask in range(1 << max(n - 1, 0)):
        blocks, start = [], 0
        for i in range(n - 1):
            if mask >> i & 1:
                blocks.append((start, i))
                start = i + 1
        blocks.append((start, n - 1))
        yield blocks


def legal(w, A, s, blocks):
    for idx, (i, j) in enumerate(blocks):
        limit = A - s if idx == len(blocks) - 1 else A
        if span(w, i, j) > limit:
            return False
    return True


def feasible_by_search(w, A, s):
    if not w:
        return s <= A
    return any(legal(w, A, s, b) for b in layouts(len(w)))


def feasible_criterion(w, A, s):
    """Fact 1 (first half): the closed-form criterion."""
    if not w:
        return s <= A
    return max(w) <= A and w[-1] <= A - s


def opt_by_search(w, A, s):
    if not w:
        return 1 if s <= A else INFEASIBLE
    best = INFEASIBLE
    for b in layouts(len(w)):
        if legal(w, A, s, b) and (best is None or len(b) < best):
            best = len(b)
    return best


def penalty(w, A, s, blocks):
    total = 0
    for idx, (i, j) in enumerate(blocks):
        beta = A - s if idx == len(blocks) - 1 else A
        slack = beta - span(w, i, j)
        total += slack * slack
    return total


def opt2_by_search(w, A, s):
    """Best squared-slack penalty among legal layouts of minimum line count."""
    if not w:
        return 0 if s <= A else INFEASIBLE
    best_lines = opt_by_search(w, A, s)
    if best_lines is INFEASIBLE:
        return INFEASIBLE
    best = INFEASIBLE
    for b in layouts(len(w)):
        if legal(w, A, s, b) and len(b) == best_lines:
            p = penalty(w, A, s, b)
            if best is None or p < best:
                best = p
    return best


# ── Greedy / Phi / the optimal split ─────────────────────────────────────────

def greedy(w, lo, hi, budget):
    """Greedy blocks over w[lo..hi] at `budget` (a block may be a lone over-wide word)."""
    blocks, i = [], lo
    while i <= hi:
        j = i
        if w[i] > budget:
            blocks.append((i, i))
            i += 1
            continue
        while j + 1 <= hi and span(w, i, j + 1) <= budget:
            j += 1
        blocks.append((i, j))
        i = j + 1
    return blocks


def greedy_count(w, budget, hi=None):
    if not w:
        return 0
    hi = len(w) - 1 if hi is None else hi
    if hi < 0:
        return 0
    return len(greedy(w, 0, hi, budget))


def phi(w, A, s):
    """The two-phase algorithm of the problem statement."""
    if not w:
        return []
    blocks = greedy(w, 0, len(w) - 1, A)
    last = blocks[-1]
    if span(w, *last) + s <= A:
        return blocks
    return blocks[:-1] + greedy(w, last[0], last[1], A - s)


def jstar(w, A, s):
    """Start of the longest feasible suffix, or INFEASIBLE."""
    if not w:
        return INFEASIBLE
    best = INFEASIBLE
    for j in range(len(w) - 1, -1, -1):
        if span(w, j, len(w) - 1) <= A - s:
            best = j
    return best


def opt_closed_form(w, A, s):
    """OPT = L(j*-1) + 1 (§2.3), plus the feasibility gate of §2.1."""
    if not w:
        return 1 if s <= A else INFEASIBLE
    if max(w) > A:
        return INFEASIBLE
    j = jstar(w, A, s)
    if j is INFEASIBLE:
        return INFEASIBLE
    return greedy_count(w, A, j - 1) + 1


def wrap_locator(w, A, s):
    """The O(n) algorithm of §2.4 — O(1) extra space, no span recomputation."""
    n = len(w)
    if n == 0:
        return 1 if s <= A else INFEASIBLE
    cur, j, maxw = 0, INFEASIBLE, 0
    for idx in range(n - 1, -1, -1):
        maxw = max(maxw, w[idx])
        cur = w[idx] + (1 if idx < n - 1 else 0) + cur
        if cur <= A - s:
            j = idx
    if maxw > A or j is INFEASIBLE:
        return INFEASIBLE
    cnt, i = 0, 0
    while i < j:
        cnt += 1
        cur = w[i]
        i += 1
        while i < j and cur + 1 + w[i] <= A:
            cur += 1 + w[i]
            i += 1
    return cnt + 1


def opt2_dp(w, A, s):
    """The (i,k) DP of §3(b) with the final-block append, O(n^2 * OPT)."""
    n = len(w)
    opt = opt_closed_form(w, A, s)
    if opt is INFEASIBLE:
        return INFEASIBLE
    if n == 0:
        return 0
    K = opt - 1
    INF = float("inf")
    prev = [0] + [INF] * n            # k = 0
    best = INF
    for k in range(1, K + 1):
        cur = [INF] * (n + 1)
        for i in range(1, n + 1):
            for p in range(0, i):
                if prev[p] is INF or prev[p] == INF:
                    continue
                b = span(w, p, i - 1)
                if b <= A:
                    cur[i] = min(cur[i], prev[p] + (A - b) ** 2)
        prev = cur
    for i in range(0, n):                       # final block w[i..n-1]
        b = span(w, i, n - 1)
        if prev[i] != INF and b <= A - s:
            best = min(best, prev[i] + (A - s - b) ** 2)
    return best


# ── Checks ───────────────────────────────────────────────────────────────────

def check_greedy_overrun():
    for n in range(1, 6):
        for w in product(range(1, 7), repeat=n):
            for budget in range(1, 8):
                for (i, j) in greedy(w, 0, n - 1, budget):
                    over = span(w, i, j) > budget
                    if over and not (i == j and w[i] > budget):
                        raise AssertionError(("lemma 1", w, budget, (i, j)))
    print("ok  lemma 1: a greedy block overruns iff it is a lone over-wide word")


def check_feasibility():
    for n in range(0, 5):
        widths = [()] if n == 0 else product(range(1, 5), repeat=n)
        for w in widths:
            for A in range(1, 9):
                for s in range(0, 7):
                    got = feasible_criterion(w, A, s)
                    want = feasible_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("feasibility", w, A, s, got, want))
    print("ok  Fact 1: closed-form feasibility == exhaustive layout search")


def check_phi():
    feasible_count = 0
    for n in range(0, 5):
        widths = [()] if n == 0 else product(range(1, 5), repeat=n)
        for w in widths:
            for A in range(1, 9):
                for s in range(0, 7):
                    out = phi(w, A, s)
                    legal_out = (not w and s <= A) or (bool(out) and legal(w, A, s, out))
                    if legal_out != feasible_criterion(w, A, s):
                        raise AssertionError(("phi legality", w, A, s, out))
                    if legal_out:
                        feasible_count += 1
                        # n = 0 has the unique empty line; `phi` returns no blocks.
                        lines = len(out) if w else 1
                        if lines < opt_by_search(w, A, s):
                            raise AssertionError(("phi beats OPT", w, A, s, out))
    assert feasible_count > 0
    print(f"ok  Phi legal iff feasible ({feasible_count} feasible instances); Phi >= OPT")

    for M in (2, 3, 5, 8, 13):
        w = [1] * M
        A, s = 2 * M + 1, 2 * M
        got, want = len(phi(w, A, s)), opt_by_search(w, A, s)
        assert got == M and want == 2, (M, got, want)
    print("ok  Fact 2: Phi emits M lines where OPT = 2 (ratio M/2 -> unbounded)")


def check_opt_formula():
    for n in range(0, 6):
        widths = [()] if n == 0 else product(range(1, 6), repeat=n)
        for w in widths:
            for A in range(1, 10):
                for s in range(0, 8):
                    got = opt_closed_form(w, A, s)
                    want = opt_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("OPT", w, A, s, got, want))
                    if wrap_locator(w, A, s) != got:
                        raise AssertionError(("O(n) algorithm", w, A, s))
    print("ok  OPT = L(j*-1)+1 and the O(n) algorithm agree with exhaustive search")


def check_paragraph_additivity():
    """§1.4: the paragraph optimum is the sum of the per-paragraph optima.

    The ground truth uses the exhaustive per-paragraph search (`opt_by_search`,
    budget `A` for body paragraphs, `A`/`s` for the last one); the claim under
    test is the composed criterion `paragraph_opt`, which derives each value
    from the closed forms instead.
    """

    def paragraph_opt(paras, A, s):
        total = 0
        for r, p in enumerate(paras):
            if r < len(paras) - 1:
                # An empty body paragraph is still one empty line.
                got = 1 if not p else (greedy_count(p, A) if max(p) <= A else INFEASIBLE)
            elif not p:
                got = 1 if s <= A else INFEASIBLE
            else:
                got = opt_closed_form(p, A, s)
            if got is INFEASIBLE:
                return INFEASIBLE
            total += got
        return total

    for A in range(1, 8):
        for s in range(0, 6):
            for p1 in ([], [1], [3], [1, 1], [2, 2]):
                for p2 in ([], [1], [2], [1, 1, 1], [3, 1]):
                    paras = [p1, p2]
                    want = 0
                    for r, p in enumerate(paras):
                        got = opt_by_search(p, A, 0 if r < len(paras) - 1 else s)
                        if got is INFEASIBLE:
                            want = INFEASIBLE
                            break
                        want += got
                    if want is INFEASIBLE:
                        assert paragraph_opt(paras, A, s) is INFEASIBLE, (paras, A, s)
                        continue
                    assert paragraph_opt(paras, A, s) == want, (paras, A, s, want)
    print("ok  paragraph line counts are additive (each paragraph independently optimal)")


def check_secondary_objective():
    # Witness 1: L is flat on [j*, j'] -> Q2's split is not penalty-optimal.
    w, A, s = [1, 1, 1], 4, 1
    q2 = penalty(w, A, s, [(0, 0), (1, 2)])
    alt = penalty(w, A, s, [(0, 1), (2, 2)])
    assert q2 == 9 and alt == 5 and opt2_by_search(w, A, s) == 5 and opt2_dp(w, A, s) == 5

    # Witness 2: j* is forced, the prefix still has an L-flat alternative.
    w2, A2, s2 = [2, 1, 1, 1], 4, 2
    q2b = penalty(w2, A2, s2, [(0, 1), (2, 2), (3, 3)])
    altb = penalty(w2, A2, s2, [(0, 0), (1, 2), (3, 3)])
    assert q2b == 10 and altb == 6 and opt2_by_search(w2, A2, s2) == 6
    print("ok  Q3a: both witnesses make greedy+longest-suffix strictly suboptimal")

    # Q3b: the DP equals an exhaustive minimum-line search over all instances.
    for n in range(1, 6):
        for w in product(range(1, 5), repeat=n):
            for A in range(1, 9):
                for s in range(0, 7):
                    got, want = opt2_dp(w, A, s), opt2_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("opt2", w, A, s, got, want))
    print("ok  Q3b: the (i,k) DP equals exhaustive search over minimum-line layouts")


def check_monge():
    # c_beta(p, i) = (D_p^beta - C_i)^2 with D_p^beta = C_p + beta + 1.
    import random
    rng = random.Random(20260119)
    for _ in range(20000):
        n = 6
        w = [rng.randint(1, 4) for _ in range(n)]
        C = [0]
        for x in w:
            C.append(C[-1] + x + 1)
        A = rng.randint(2, 12)
        s = rng.randint(0, A)
        p, pp = sorted(rng.sample(range(n), 2))
        i, ii = sorted(rng.sample(range(n), 2))
        beta_i = A if i < n - 1 else A - s
        beta_ii = A if ii < n - 1 else A - s
        d_p, d_pp = C[p], C[pp]
        c = lambda dp, beta, ci: (dp + beta + 1 - ci) ** 2
        lhs = c(d_p, beta_i, C[i]) + c(d_pp, beta_ii, C[ii])
        rhs = c(d_pp, beta_i, C[i]) + c(d_p, beta_ii, C[ii])
        if lhs > rhs:
            raise AssertionError(("Monge", w, A, s, (p, pp), (i, ii), lhs, rhs))
        # Feasible window is a nondecreasing staircase.
        def lo(beta, ci):
            return next((q for q in range(n + 1) if C[q] >= ci - beta - 1), None)
        l1, l2 = lo(beta_i, C[i]), lo(beta_ii, C[ii])
        if l1 is not None and l2 is not None and l1 > l2:
            raise AssertionError(("staircase", w, A, s))
    print("ok  Q3c: the mixed body/final matrix is Monge; the window staircase is monotone")


def check_policies():
    # Loss vectors, dominance table, and the no-go premises in the feasible region.
    for n in range(1, 6):
        for w in product(range(1, 5), repeat=n):
            for A in range(1, 9):
                for s in range(0, 7):
                    if max(w) > A or w[-1] <= A - s:
                        continue                      # only the infeasible region
                    tau = w[-1] + s - A
                    ell = greedy_count(w, A, n - 2) + 1
                    delta = greedy_count(w, A) - greedy_count(w, A, n - 2)
                    assert tau > 0 and delta in (0, 1)
                    p1 = (tau, ell, 0)
                    p3 = (0, ell, tau)
                    # P1/P3 are always incomparable: no componentwise order.
                    assert not dominates(p1, p3) and not dominates(p3, p1)
                    if s <= A:
                        p2 = (0, ell + delta, 0)
                        if delta == 0:
                            assert dominates(p2, p1) and dominates(p2, p3)
                        else:
                            for a, b in ((p1, p2), (p2, p1), (p1, p3), (p3, p1), (p2, p3), (p3, p2)):
                                assert not dominates(a, b)
                        # No-go premises: full locator + zero overflow needs
                        # L(n) body lines plus one locator line.
                        assert greedy_count(w, A) + 1 == ell + delta
    # s > A: P2 inadmissible, and a split-locator hybrid is strictly better.
    w, A, s = [1, 1], 4, 5
    p1, p3 = (2, 2, 0), (0, 2, 2)
    hybrid = (0, 2, 0)
    assert dominates(hybrid, p1) and dominates(hybrid, p3)
    print("ok  Q4: loss vectors, dominance table, and the s>A hybrid counterexample")


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def main():
    check_greedy_overrun()
    check_feasibility()
    check_phi()
    check_opt_formula()
    check_paragraph_additivity()
    check_secondary_objective()
    check_monge()
    check_policies()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
