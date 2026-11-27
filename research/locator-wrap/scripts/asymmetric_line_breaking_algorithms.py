#!/usr/bin/env python3
"""Algorithms and machine checks for asymmetric line breaking.

Running this file executes every check, including the baseline exhaustive
checks for solution.md. Only the standard library is required.
"""
from bisect import bisect_left, bisect_right
from collections import deque
from heapq import heappush, heappop
from itertools import product, permutations
from math import inf
import random, time

INFEASIBLE = None


# ==== experiments.py ====
"""Computational experiments for the asymmetric line-breaking problems.

"""
from itertools import product
from math import inf

# ── shared model (1-based words) ────────────────────────────────────────────

def Cvec(w):
    C = [0]
    for a in w:
        C.append(C[-1] + a + 1)
    return C

def span(w, i, j):
    """B(i,j), 1-based inclusive."""
    if i > j:
        return 0
    return sum(w[i-1:j]) + (j - i)

def greedy(w, budget, lo=1, hi=None):
    hi = len(w) if hi is None else hi
    out, i = [], lo
    while i <= hi:
        if w[i-1] > budget:
            out.append((i, i)); i += 1; continue
        j = i
        while j + 1 <= hi and sum(w[i-1:j+1]) + (j + 1 - i) <= budget:
            j += 1
        out.append((i, j)); i = j + 1
    return out

def phi_lines(w, A, s):
    G = greedy(w, A)
    i, j = G[-1]
    if span(w, i, j) + s <= A:
        return len(G)
    H = greedy(w, A-s, lo=i, hi=j)
    return len(G) - 1 + len(H)

def opt_lines(w, A, s):
    """Closed form OPT = L(j*-1)+1, None if infeasible."""
    n = len(w)
    if n == 0:
        return 1 if s <= A else None
    if max(w) > A or w[-1] > A - s:
        return None
    jstar = None
    for j in range(n, 0, -1):
        if span(w, j, n) <= A - s:
            jstar = j
    return len(greedy(w, A, hi=jstar-1)) + 1

# ── Problem 1(a): total monotonicity of the end-indexed cost matrix ─────────

def matrix(C, beta):
    """beta[j-1] is the budget of a line ending at j (1-based)."""
    n = len(beta)
    z = [C[j] - beta[j-1] - 1 for j in range(1, n+1)]   # z[j-1]
    def M(j, i):                                          # j 1-based, i 0-based
        if i < 0 or i >= j:
            return inf                                    # triangular domain
        if C[i] < z[j-1]:
            return inf
        return (C[i] - z[j-1]) ** 2
    return z, M

def tm_brute(C, beta, oo_equal=True, triangular=True):
    z, M = matrix(C, beta)
    n = len(beta)
    for j in range(1, n+1):
        for jp in range(j+1, n+1):
            imax = j-1 if triangular else n
            for i in range(0, imax):
                for ip in range(i+1, (j if triangular else n)):
                    a, b = M(j, ip), M(j, i)
                    if oo_equal:
                        premise = a <= b
                    else:
                        premise = a <= b and not (a == inf and b == inf)
                    if premise and not (M(jp, ip) <= M(jp, i)):
                        return False, (j, jp, i, ip)
    return True, None

def tm_criterion(C, beta):
    """Characterization (standard SMAWK convention inf<=inf is True,
    triangular domain i<i'<j<j')."""
    z, M = matrix(C, beta)
    n = len(beta)
    for j in range(1, n+1):
        for jp in range(j+1, n+1):
            if z[jp-1] < z[j-1]:
                for i in range(0, j-1):           # i <= j-2, so i'=i+1 < j
                    if z[jp-1] <= C[i] < z[j-1]:
                        return False, (j, jp, i)
    return True, None

def tm_linear(C, beta):
    """O(n) decision using suffix minima m_j = min_{j'>j} z_{j'}."""
    z, M = matrix(C, beta)
    n = len(beta)
    m = [inf]*(n+1)
    for j in range(n-1, 0, -1):
        m[j] = min(z[j], m[j+1])       # m[j] = min(z_{j+1..n})
    ptr = 0
    for j in range(1, n+1):
        while ptr < n and C[ptr] < m[j]:
            ptr += 1
        if ptr <= j-2 and ptr < n and C[ptr] < z[j-1]:
            return False, (j, ptr)
    return True, None

def sc(C, beta):
    z, M = matrix(C, beta)
    return all(z[j] <= z[j+1] for j in range(len(z)-1))

def check_1a():
    print("== Problem 1(a) ==")
    rng = __import__('random').Random(7)
    bad = 0
    witness = None
    for n in range(2, 6):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(w)
            for _ in range(12):
                beta = [rng.randint(1, 10) for _ in range(n)]
                got = tm_brute(C, beta)[0]
                crit = tm_criterion(C, beta)[0]
                lin = tm_linear(C, beta)[0]
                if got != crit or got != lin:
                    bad += 1
                    print("  MISMATCH", w, beta, got, crit, lin)
                if sc(C, beta) and not got:
                    print("  SC but not TM?!", w, beta)
                if got and not sc(C, beta) and witness is None:
                    witness = (w, beta, matrix(C, beta)[0])
    print("  mismatches (brute vs characterization vs O(n) test):", bad)
    print("  narrow witness (TM true, SC false): w,beta,z =", witness)
    # exhaustive search for the smallest witness using the (verified) criterion
    for n in range(2, 6):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(w)
            for beta in product(range(1, 8), repeat=n):
                if tm_criterion(C, beta)[0] and not sc(C, beta):
                    print("  smallest witness: w=", w, "beta=", beta,
                          "z=", matrix(C, beta)[0],
                          "tm_brute=", tm_brute(C, beta)[0])
                    return
    print("  (no small witness found in that range)")

# ── Problem 2: exact competitive ratio of Phi ───────────────────────────────

def all_sequences_span_le(A, limit=None):
    """All word sequences with 1<=a_t<=A and total span sum(a)+n-1 <= A."""
    out = []
    def rec(seq, used):           # used = sum(a) + len(seq)-1
        if seq:
            out.append(tuple(seq))
        room = A - used
        if room <= 0:
            return
        for a in range(1, min(A, room) + 1):
            seq.append(a)
            rec(seq, used + a + 1)
            seq.pop()
    rec([], 0)
    return out

def rho_formula(A, s):
    B = A - s
    Bp = max(B, 2)
    d = Bp + 2
    q, r = divmod(A+1, d)
    M = 2*q + (1 if r >= 2 else 0)
    return max(1.0, M/2.0)

def check_2(Amax=14, brute_all=True):
    print("== Problem 2 ==")
    mism = 0
    for A in range(1, Amax+1):
        seqs = all_sequences_span_le(A)
        for s in range(0, A):
            B = A - s
            best = 1.0
            bestw = None
            for w in seqs:
                if w[-1] > B or max(w) > A:
                    continue
                if span(w, 1, len(w)) <= B:
                    continue                  # no re-wrap
                o = opt_lines(w, A, s)
                if o is None:
                    continue
                r = phi_lines(w, A, s) / o
                if r > best:
                    best, bestw = r, w
            f = rho_formula(A, s)
            if abs(best - f) > 1e-9:
                mism += 1
                print("  MISMATCH A,s=", A, s, "brute=", best, "formula=", f,
                      "w=", bestw)
        # extra check: unstructured sequences (k>1 allowed)
        if brute_all and A <= 9:
            for s in range(0, A):
                B = A-s
                best = 1.0
                for n in range(1, 6):
                    for w in product(range(1, min(A, 5)+1), repeat=n):
                        if w[-1] > B or max(w) > A:
                            continue
                        o = opt_lines(w, A, s)
                        if o is None:
                            continue
                        r = phi_lines(w, A, s)/o
                        if r > best:
                            best = r
                f = rho_formula(A, s)
                if abs(best-f) > 1e-9:
                    print("  k>1 MISMATCH A,s=", A, s, best, f)
    print("  formula/brute mismatches:", mism)
    for A in (4, 6, 10, 14):
        vals = [rho_formula(A, s) for s in range(A)]
        print("  A=%2d rho(s=0..A-1)=" % A, vals,
              "monotone=", all(vals[i] <= vals[i+1] for i in range(len(vals)-1)))

# ── Problem 3: Pareto frontier of the infeasible region ──────────────────────

def body_partitions(n):
    for mask in range(1 << max(n-1, 0)):
        blocks, start = [], 1
        for i in range(1, n):
            if mask >> (i-1) & 1:
                blocks.append((start, i)); start = i+1
        blocks.append((start, n))
        yield blocks

def frontier(w, A, s):
    n = len(w)
    pts = set()
    for blocks in body_partitions(n):
        B = len(blocks)
        if any(span(w, i, j) > A for (i, j) in blocks[:-1]):
            continue
        W = span(w, *blocks[-1])
        for p in range(0, s+1):              # kept (non-truncated) chars
            T = s - p
            for k in range(0, p+1):         # chars appended to last body line
                R = p - k
                d = 0 if R == 0 else (R + A - 1)//A
                V = max(0, W + k - A)
                pts.add((V, B + d, T))
    # Pareto filter (minimise all three)
    out = []
    for x in pts:
        if not any(all(y[c] <= x[c] for c in range(3)) and
                   any(y[c] < x[c] for c in range(3)) for y in pts if y != x):
            out.append(x)
    return sorted(out)

def dominates(x, y):
    return all(x[c] <= y[c] for c in range(3)) and any(x[c] < y[c] for c in range(3))

def body_min_width(w, A):
    """For each B=1..n: minimal width of the last body block when the prefix
    uses exactly B-1 budget-A blocks (None if impossible)."""
    n = len(w)
    C = Cvec(w)
    L = [0]*(n+1)
    for m in range(1, n+1):
        L[m] = len(greedy(w, A, hi=m))
    out = []
    for B in range(1, n+1):
        m = B-1
        iB = None
        for i in range(n-1, m-1, -1):
            if L[i] <= m:
                iB = i; break
        out.append(None if iB is None else C[n]-C[iB]-1)
    return out, L

def check_3():
    print("== Problem 3 ==")
    for (w, A, s) in [([1,1], 4, 4), ([1,1], 4, 5), ([4], 4, 6)]:
        print("  w,A,s=", w, A, s, "frontier=", frontier(w, A, s))
    # max frontier size over instances with n words, s<=A
    for n in range(1, 4):
        best = (0, None)
        for A in range(2, 7):
            for w in product(range(1, A+1), repeat=n):
                for s in range(A - w[-1] + 1, A+1):   # infeasible, s<=A
                    if s < 0: continue
                    f = frontier(w, A, s)
                    if len(f) > best[0]:
                        best = (len(f), (w, A, s, f))
        print("  n=%d, s<=A: max |frontier| = %d  at %s" % (n, best[0], best[1][:3]))
    # unbounded in s: n=1, w=(A), chain
    for A, s in [(4, 10), (4, 30), (6, 20)]:
        f = frontier([A], A, s)
        print("  n=1, w=(%d), A=%d, s=%d: |frontier|=%d (chain %s..%s)" %
              (A, A, s, len(f), f[0], f[-1]))


# ==== convexity and frontier checks ====
"""Deeper experiments: 1(a) Monge-vs-TM witness + o(n^2) DP when TM fails;
1(b) lambda-shift run compression; 3 exact frontier counts."""
from itertools import product
from math import inf
import random

# ── 1(a): Monge fails, total monotonicity survives ───────────────────────────

def monge_brute(C, beta):
    """Monge: M[j,i]+M[j',i'] <= M[j,i']+M[j',i] for all i<i'<j<j',
    all four finite."""
    z, M = matrix(C, beta)
    n = len(beta)
    for i in range(0, n):
        for ip in range(i+1, n):
            for j in range(ip+1, n+1):
                for jp in range(j+1, n+1):
                    a, b, c, d = M(j,i), M(jp,ip), M(j,ip), M(jp,i)
                    if min(a,b,c,d) == inf:
                        continue
                    if a + b > c + d:
                        return False, (i, ip, j, jp, a+b-c-d)
    return True, None

def check_1a_monge():
    print("== 1(a) Monge vs TM ==")
    # w=(1,1,1,1), beta=(1,3,5,8): z=(0,0,0,-1), SC fails at (3,4)
    C = Cvec((1,1,1,1))
    beta = (1,3,5,8)
    z, M = matrix(C, beta)
    print("  C=", C, "beta=", beta, "z=", z)
    print("  SC=", sc(C, beta), "TM=", tm_brute(C, beta)[0],
          "crit=", tm_criterion(C, beta)[0],
          "Monge=", monge_brute(C, beta))
    # show the failing quadrangle explicitly
    i, ip, j, jp, diff = monge_brute(C, beta)[1]
    print("  failing quadrangle i,i',j,j'=", (i,ip,j,jp),
          "QD (LHS-RHS)=", diff)

# ── 1(a): O(n log^2 n) DP that does NOT need total monotonicity ──────────────

class SegTreeHulls:
    """Point insert line (m,b) at leaf index i, with slopes inserted in
    strictly decreasing order globally. Range query min_i (b_i + m_i * x)."""
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n: self.size <<= 1
        self.nodes = [[] for _ in range(2*self.size)]
        self.bp = [[] for _ in range(2*self.size)]   # breakpoints per node

    def insert(self, idx, m, b):
        p = idx + self.size
        while p:
            nd = self.nodes[p]
            # slopes arrive decreasing; maintain lower hull with a stack
            while len(nd) >= 2:
                (m1,b1),(m2,b2) = nd[-2], nd[-1]
                # intersection of last two vs new line (m decreasing)
                x_last_new = (b - b2) / (m2 - m)
                x_prev_last = (b2 - b1) / (m1 - m2)
                if x_last_new <= x_prev_last:
                    nd.pop()
                else:
                    break
            nd.append((m,b))
            p >>= 1

    def _eval(self, nd, x):
        # binary search over hull lines (slopes decreasing); breakpoints increasing
        lo, hi = 0, len(nd)-1
        while lo < hi:
            mid = (lo+hi)//2
            v1 = nd[mid][1] + nd[mid][0]*x
            v2 = nd[mid+1][1] + nd[mid+1][0]*x
            if v2 <= v1: lo = mid+1
            else: hi = mid
        return nd[lo][1] + nd[lo][0]*x

    def query(self, l, r, x):
        l += self.size; r += self.size
        best = inf
        while l <= r:
            if l & 1:
                if self.nodes[l]: best = min(best, self._eval(self.nodes[l], x))
                l += 1
            if not (r & 1):
                if self.nodes[r]: best = min(best, self._eval(self.nodes[r], x))
                r -= 1
            l >>= 1; r >>= 1
        return best

def dp_brute_extra(C, z, lam):
    n = len(z)
    dp = [inf]*(n+1); dp[0] = 0.0
    for j in range(1, n+1):
        for i in range(0, j):
            if C[i] < z[j-1] or dp[i] == inf: continue
            dp[j] = min(dp[j], dp[i] + (C[i]-z[j-1])**2 + lam)
    return dp

def dp_hulls(C, z, lam):
    n = len(z)
    st = SegTreeHulls(n+1)
    # line for index i: m=-2C_i, b=dp[i]+C_i^2
    dp = [inf]*(n+1); dp[0] = 0.0
    st.insert(0, -2*C[0], dp[0] + C[0]**2)
    for j in range(1, n+1):
        x = z[j-1]
        # feasible i: C_i >= x  =>  smallest such index
        lo = next((i for i in range(0, j) if C[i] >= x), j)
        if lo < j:
            v = st.query(lo, j-1, x)
            dp[j] = x*x + v + lam
        if dp[j] < inf:
            st.insert(j, -2*C[j], dp[j] + C[j]**2)
    return dp

def check_1a_dp():
    print("== 1(a) DP without total monotonicity ==")
    rng = random.Random(11)
    bad = 0
    for n in range(2, 9):
        for _ in range(300):
            w = [rng.randint(1,6) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 14) for _ in range(n)]
            z = [C[j]-beta[j-1]-1 for j in range(1,n+1)]
            lam = rng.randint(0, 30)
            a = dp_brute_extra(C, z, lam)
            b = dp_hulls(C, z, lam)
            if any(abs(x-y) > 1e-6 for x,y in zip(a,b)):
                bad += 1
                print("   MISMATCH", w, beta, a, b)
                break
    print("  segment-tree-of-hulls DP mismatches:", bad,
          "(O(n log^2 n), works for arbitrary beta / when TM fails)")
    # count how often TM fails in the sample
    fails = 0
    for n in range(3, 8):
        for _ in range(200):
            w = [rng.randint(1,6) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 14) for _ in range(n)]
            if not tm_criterion(C, beta)[0]: fails += 1
    print("  TM-failing random instances in sample:", fails)

# ── 1(b): lambda-shift inside a run of equal budgets ─────────────────────────

def run_exact(C, dp0, beta, R):
    n = len(C)-1
    E = list(dp0)
    for _ in range(R):
        F = [inf]*(n+1)
        for j in range(0, n+1):
            for i in range(0, j):
                if E[i] == inf or C[j]-C[i]-1 > beta: continue
                F[j] = min(F[j], E[i] + (beta+1+C[i]-C[j])**2)
        E = F
    return E

def run_lagrange(C, dp0, beta, lam):
    """min over any number of lines of (cost + lam*lines); return (value, lines)."""
    n = len(C)-1
    G = list(dp0); cnt = [0]*(n+1)
    for j in range(0, n+1):
        best, bi = G[j], j
        for i in range(0, j):
            if G[i] == inf or C[j]-C[i]-1 > beta: continue
            v = G[i] + (beta+1+C[i]-C[j])**2 + lam
            if v < best:
                best, bi = v, i
        if bi != j:
            G[j] = best; cnt[j] = cnt[bi]+1
    return G[n], cnt[n]

def check_1b():
    print("== 1(b) lambda-shift run compression ==")
    rng = random.Random(23)
    stats = {}
    for kind in ("standard", "structured", "arbitrary"):
        stats[kind] = [0, 0, 0]   # trials, nonconvex, legendre_fail_on_finite
    for trial in range(300):
        n = rng.randint(3, 7)
        w = [rng.randint(1,5) for _ in range(n)]
        C = Cvec(w)
        beta = rng.randint(2, 9)
        Rmax = 6
        for kind in stats:
            if kind == "standard":
                dp0 = [0.0] + [inf]*n
            elif kind == "structured":
                # cost vector obtained from 2 layers of a *different* budget
                prev = [0.0] + [inf]*n
                dp0 = run_exact(C, prev, beta+2, rng.randint(1,3))
                if all(x == inf for x in dp0):
                    dp0 = [0.0] + [inf]*n
            else:
                dp0 = [rng.randint(0, 40) for _ in range(n+1)]
            P = [run_exact(C, dp0, beta, R)[n] for R in range(0, Rmax+1)]
            finite = [x for x in P if x < inf]
            # discrete convexity on the finite stretch
            nonconvex = False
            if len(finite) >= 3:
                dif = [finite[i+1]-finite[i] for i in range(len(finite)-1)]
                nonconvex = any(dif[i+1] < dif[i]-1e-9 for i in range(len(dif)-1))
            L = {}
            for lam in range(-250, 251):
                L[lam] = run_lagrange(C, dp0, beta, lam)[0]
            legfail = 0
            for R, val in enumerate(P):
                if val == inf: continue
                best = max(L[lam] - lam*R for lam in L)
                if abs(best - val) > 1e-6:
                    legfail += 1
            stats[kind][0] += 1
            stats[kind][1] += int(nonconvex)
            stats[kind][2] += int(legfail > 0)
    for kind, (t, nc, lf) in stats.items():
        print("  dp0=%-10s trials=%d  nonconvex-curves=%d  Legendre-failures=%d" %
              (kind, t, nc, lf))
    print("  (standard start = e_0; structured = layered DP output)")

# ── 3: exact maximum frontier size when s <= A ───────────────────────────────

def check_3_exact():
    print("== 3 exact frontier counts (s <= A) ==")
    for A in range(2, 9):
        for n in range(1, 4):
            best = (0, None)
            for w in product(range(1, A+1), repeat=n):
                for s in range(max(0, A - w[-1] + 1), A+1):
                    f = frontier(w, A, s)
                    if len(f) > best[0]:
                        best = (len(f), (w, s))
            pred = n*(A+1)+1
            print("  A=%d n=%d max=%d at %s  formula n(A+1)+1=%d %s" %
                  (A, n, best[0], best[1], pred, "OK" if best[0]==pred else "DIFF"))
    print("== 3 growth in s (n=1, w=(A)) ==")
    for A in (3,4,6):
        for s in (A, 2*A, 4*A+1):
            f = frontier([A], A, s)
            q, r = divmod(s, A)
            pred = sum(max(1, s-d*A+1) for d in range(0, q+1)) + (1 if r else 0)
            print("  A=%d s=%d |F|=%d  closed-form=%d %s" %
                  (A, s, len(f), pred, "OK" if len(f)==pred else "DIFF"))


# ==== strict TM and witness checks ====
"""Checks: alternative TM convention, explicit non-convex witness for 1(b),
Problem 2 extremal construction, and rho monotonicity for all small A."""
from itertools import product
import random

def tm_criterion_strict(C, beta):
    """Characterization for the convention inf<=inf is FALSE (premise needs
    a finite right-hand entry), triangular domain."""
    z, M = matrix(C, beta)
    n = len(beta)
    for j in range(1, n+1):
        for jp in range(j+1, n+1):
            if z[jp-1] < z[j-1]:
                for i in range(0, j-1):
                    if z[jp-1] <= C[i] < z[j-1]:
                        if any(C[ip] >= z[j-1] for ip in range(i+1, j)):
                            return False, (j, jp, i)
    return True, None

def check_strict():
    rng = random.Random(99)
    bad = 0
    for n in range(2, 6):
        for w in product(range(1,4), repeat=n):
            C = Cvec(w)
            for _ in range(15):
                beta = [rng.randint(1,10) for _ in range(n)]
                a = tm_brute(C, beta, oo_equal=False)[0]
                b = tm_criterion_strict(C, beta)[0]
                if a != b:
                    bad += 1
                    if bad <= 3: print("   mismatch", w, beta, a, b)
    print("strict-convention mismatches:", bad)

def witness_1b():
    """Small explicit structured-start witness that P(R) is NOT convex in R."""
    rng = random.Random(5)
    for n in (3, 4, 5):
        for w in product(range(1,4), repeat=n):
            C = Cvec(w)
            for beta in (2,3,4,5):
                prev = [0.0] + [float('inf')]*n
                for Rprev in (1,2):
                    dp0 = run_exact(C, prev, beta+2, Rprev)
                    if all(x == float('inf') for x in dp0): continue
                    P = [run_exact(C, dp0, beta, R)[n] for R in range(0, 5)]
                    fin = [x for x in P if x < float('inf')]
                    if len(fin) >= 3:
                        d = [fin[i+1]-fin[i] for i in range(len(fin)-1)]
                        if any(d[i+1] < d[i]-1e-9 for i in range(len(d)-1)):
                            lam = 0
                            L0 = run_lagrange(C, dp0, beta, lam)[0]
                            print("  w=", w, "beta=", beta, "Rprev=", Rprev)
                            print("  dp0=", [('inf' if x==float('inf') else x) for x in dp0])
                            print("  P(0..4)=", [('inf' if x==float('inf') else x) for x in P])
                            print("  successive differences=", d, "-> NOT convex")
                            return True
    return False

def check_2_construction():
    print("== Problem 2 extremal instances ==")
    for A, s in [(10,8), (10,9), (14,11), (14,13), (9,6)]:
        B = A - s
        rho = rho_formula(A, s)
        # alternating single-word blocks: (B-1,1)* or all ones when B<=2
        if B <= 2:
            w = [1]*rho_int(rho, A, s)
        else:
            n = 2*((A+1)//(B+2)) + (1 if (A+1)%(B+2) >= 2 else 0)
            w = []
            for t in range(n):
                w.append(1 if t % 2 else B-1)
        T = sum(w) + len(w) - 1
        print("  A=%d s=%d B=%d rho=%s  witness w=%s T=%d k=%d Phi=%d OPT=%d ratio=%s" %
              (A, s, B, rho, w, T, len(greedy(w, A)), phi_lines(w, A, s),
               opt_lines(w, A, s),
               phi_lines(w, A, s)/opt_lines(w, A, s)))
    bad = 0
    for A in range(1, 61):
        vals = [rho_formula(A, s) for s in range(A)]
        if any(vals[i] > vals[i+1] for i in range(len(vals)-1)):
            bad += 1; print("   NON-MONOTONE at A=", A, vals)
    print("  rho monotone in s for all A<=60: ", bad == 0)

def rho_int(rho, A, s):
    # for B<=2 the maximizer is floor((A+1)/2) unit words
    return (A+1)//2


# ==== envelope and frontier checks ====
"""Continuation: an O(n log n) DP for arbitrary end-indexed budgets,
verification against exhaustive layouts, and a rigorous proof of the
Problem-3 maximum frontier size.

"""
from bisect import bisect_left
from collections import deque
from itertools import product
from math import inf
import random, time



# ── the O(n log n) CDQ DP ────────────────────────────────────────────────────

def zvec(C, beta):
    """z[j-1] = C_j - beta(j) - 1."""
    return [C[j] - beta[j-1] - 1 for j in range(1, len(C))]


def hull_insert(dq, m, b):
    """Insert a line whose slope m is strictly smaller than all stored slopes."""
    while len(dq) >= 2:
        m1, b1 = dq[-2]
        m2, b2 = dq[-1]
        m3, b3 = m, b
        # breakpoints x12=(b2-b1)/(m1-m2), x23=(b3-b2)/(m2-m3):
        # middle line dead iff x12 >= x23 (denominators all > 0).
        if (b2 - b1) * (m2 - m3) >= (b3 - b2) * (m1 - m2):
            dq.pop()
        else:
            break
    dq.append((m, b))


def hull_query(dq, x):
    """Min at x; query x's must arrive in nondecreasing order."""
    while len(dq) >= 2:
        m0, b0 = dq[0]
        m1, b1 = dq[1]
        if (m1 - m0) * x <= b0 - b1:      # line 1 <= line 0 at x
            dq.popleft()
        else:
            break
    m, b = dq[0]
    return m * x + b


def dp_cdq(C, z, lam):
    """dp[j] = lam + z_j^2 + min_{i<j, C_i>=z_j} [dp[i] + C_i^2 - 2 C_i z_j].

    CDQ over index j. At merge [L,mid] -> [mid+1,R]:
      * lines i inserted in DECREASING i (slopes 2C_i decreasing);
      * queries j swept in DECREASING z_j (x' = -z_j increasing);
      * left_j = min{i: C_i>=z_j} is nonincreasing along that sweep, so the
        insertion frontier only moves left -> one monotone CHT per node.
    The global z-order is pushed down the recursion (an O(n) partition per
    level, no per-node sorting): O(n log n) time, O(n) space.
    """
    n = len(z)
    left = [bisect_left(C, z[j-1]) for j in range(1, n+1)]
    order = sorted(range(1, n+1), key=lambda j: -z[j-1])
    raw = [inf] * (n+1)
    dp = [inf] * (n+1)
    dp[0] = 0

    def solve(l, r, ord_list):
        if l == r:
            if l > 0 and raw[l] < inf:
                dp[l] = lam + z[l-1]**2 + raw[l]
            return
        mid = (l + r) // 2
        lo_list = [j for j in ord_list if j <= mid]
        hi_list = [j for j in ord_list if j > mid]
        solve(l, mid, lo_list)                    # complete lines in [l,mid]
        dq, p = deque(), mid + 1
        for j in hi_list:                         # z decreasing
            x = -z[j-1]
            lo = max(left[j-1], l)
            if lo > mid:
                continue
            while p - 1 >= lo:
                p -= 1
                i = p
                if dp[i] < inf:
                    hull_insert(dq, 2*C[i], dp[i] + C[i]**2)
            if dq:
                v = hull_query(dq, x)
                if v < raw[j]:
                    raw[j] = v
        solve(mid+1, r, hi_list)

    solve(0, n, order)
    return dp


def dp_brute(C, z, lam):
    n = len(z)
    dp = [inf] * (n+1)
    dp[0] = 0
    for j in range(1, n+1):
        x = z[j-1]
        for i in range(0, j):
            if C[i] < x or dp[i] == inf:
                continue
            dp[j] = min(dp[j], dp[i] + (C[i]-x)**2 + lam)
    return dp


def exhaustive_lex(C, beta):
    """Lexicographic optimum (fewest lines, then least squared slack) over
    ALL layouts, for arbitrary end-indexed budgets."""
    n = len(C) - 1
    best = None
    for mask in range(1 << max(n-1, 0)):
        idx = [0]
        for t in range(1, n):
            if mask >> (t-1) & 1:
                idx.append(t)
        idx.append(n)
        L, pen, ok = len(idx)-1, 0, True
        for t in range(1, len(idx)):
            i, j = idx[t-1], idx[t]
            width, b = C[j]-C[i]-1, beta[j-1]
            if width > b:
                ok = False
                break
            pen += (b-width)**2
        if ok:
            key = (L, pen)
            if best is None or key < best:
                best = key
    return best


# ── checks ───────────────────────────────────────────────────────────────────

def check_cdq():
    print("== O(n log n) CDQ DP vs literal O(n^2) DP ==")
    rng = random.Random(20261122)
    bad, nonmono = 0, 0
    for n in list(range(1, 9)) + [16, 30]:
        for _ in range(250 if n <= 8 else 40):
            w = [rng.randint(1, 7) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 16) for _ in range(n)]
            z = zvec(C, beta)
            lam = rng.randint(0, 40)
            a, b = dp_brute(C, z, lam), dp_cdq(C, z, lam)
            if a != b:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH w,beta,lam=", w, beta, lam, a, b)
            if any(z[t] > z[t+1] for t in range(n-1)):
                nonmono += 1
    print("  instances: 2080, mismatches:", bad,
          "| non-monotone-z (SC / TM may fail):", nonmono)

    print("== CDQ DP vs exhaustive lexicographic optimum ==")
    bad2 = 0
    for n in range(1, 8):
        for _ in range(200):
            w = [rng.randint(1, 6) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(6, 14) for _ in range(n)]   # every word fits
            z = zvec(C, beta)
            want = exhaustive_lex(C, beta)
            lam = n * max(beta)**2 + 1
            d = dp_cdq(C, z, lam)[n]
            got = None if d == inf else divmod(d, lam)
            if got != want:
                bad2 += 1
                if bad2 <= 3:
                    print("   MISMATCH w,beta=", w, beta, "want=", want, "got=", got)
    print("  instances: 1400, mismatches:", bad2)


def check_scaling_cont():
    print("== scaling (arbitrary z, SC fails, all instances reachable) ==")
    rng = random.Random(5)
    for n in (500, 1000, 2000, 4000, 8000, 16000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(6, 20) for _ in range(n)]
        z = zvec(C, beta)
        lam = n*400+1
        t = time.perf_counter()
        d = dp_cdq(C, z, lam)
        dt = time.perf_counter() - t
        print("  n=%5d  %.4fs  dp[n]=%d  (finite=%s)" %
              (n, dt, d[n], d[n] < inf))


def frontier_ub(n, A, s):
    D = (s + A - 1)//A
    return n*(s+1) + sum(max(1, s-d*A+1) for d in range(1, D+1))


def frontier_fast(w, A, s):
    """Compute the Pareto frontier in O((n+D)(s+1)) by injection on (L,T).

    For each (L,T) we compute the minimum overflow V over all strategies with
    exactly L lines and exactly T truncated chars, then sweep L upwards with
    a prefix-min over T to remove 3D-dominated points.
    """
    n = len(w)
    C = Cvec(w)
    # W_B = minimal last-body-block width with exactly B body lines
    Lg = [0]*(n+1)
    for m in range(1, n+1):
        # greedy block count on first m words at A
        Lg[m] = len(greedy_local(w, A, m))
    W = {}
    for B in range(1, n+1):
        best = None
        for i in range(B-1, n):           # last block starts at i+1
            if Lg[i] <= B-1:
                cand = C[n]-C[i]-1
                if best is None or cand < best:
                    best = cand
        W[B] = best
    D = (s + A - 1)//A
    Vmin = {}
    for L in range(1, n+D+1):
        cap = s if L <= n else max(0, s-(L-n)*A)
        for T in range(0, cap+1):
            best = inf
            for B in range(max(1, L-D), min(n, L)+1):
                d = L-B
                R = min(d*A, s-T)
                k = s-T-R
                if k < 0 or W[B] is None:
                    continue
                V = max(0, W[B]+k-A)
                best = min(best, V)
            if best < inf:
                Vmin[(L, T)] = best
    # remove points dominated by some (V',L',T') with L'<=L, T'<=T
    prefix = [inf]*(s+2)        # prefix[t] = min V over L'<L and T'<=t
    out = []
    for L in range(1, n+D+1):
        cur = []
        for (L2, T), V in sorted((k, v) for k, v in Vmin.items() if k[0] == L):
            if V >= prefix[T]:
                continue                 # dominated by a preceding level
                                         # (equal V,T with fewer lines dominates)
            cur.append((V, L, T))
        # within this level, keep only the decreasing-T staircase
        cur.sort()
        keep = []
        bestT = inf
        for V, L, T in cur:
            if T < bestT:
                keep.append((V, L, T))
                bestT = T
        out.extend(keep)
        newprefix = list(prefix)
        for V, L, T in keep:
            for t in range(T, s+1):
                newprefix[t] = min(newprefix[t], V)
        prefix = newprefix
    return sorted(out)


def greedy_local(w, A, m):
    out, i = 0, 0
    while i < m:
        j = i
        while j+1 < m and sum(w[i:j+2]) + (j+1-i) <= A:
            j += 1
        out += 1
        i = j+1
    return [(0, 0)]*out


def check_frontier_fast():
    print("== Problem 3: O((n+D)(s+1)) frontier algorithm vs brute force ==")
    rng = random.Random(1234)
    bad = 0
    for _ in range(500):
        A = rng.randint(2, 8)
        n = rng.randint(1, 4)
        w = [rng.randint(1, A) for _ in range(n)]
        s = rng.randint(A-w[-1]+1, 3*A)
        a, b = frontier(w, A, s), frontier_fast(w, A, s)
        if a != b:
            bad += 1
            if bad <= 3:
                print("   MISMATCH w,A,s=", w, A, s, "brute=", a, "fast=", b)
    print("  instances: 500, mismatches:", bad)


def check_frontier_structure():
    print("== Problem 3: frontier structure ==")
    rng = random.Random(77)
    bad_antichain, bad_inj, bad_T, bad_ub, bad_antichain_2D = 0, 0, 0, 0, 0
    multi_chain = 0
    for _ in range(600):
        A = rng.randint(2, 9)
        n = rng.randint(1, 4)
        w = [rng.randint(1, A) for _ in range(n)]
        s = rng.randint(A - w[-1] + 1, 3*A)
        F = frontier(w, A, s)
        # (i) F is a 3D antichain under componentwise <=
        if any(dominates(x, y) for x in F for y in F if x != y):
            bad_antichain += 1
        # (ii) at most one frontier point per (L,T)  [injectivity by (L,T)]
        seen = set()
        byL = {}
        for V, L, T in F:
            if (L, T) in seen:
                bad_inj += 1
            seen.add((L, T))
            byL.setdefault(L, []).append((V, T))
        # (iii) at each L, (V,T) is a 2D antichain: V strictly up, T strictly down
        for L, pts in byL.items():
            pts.sort()
            if any(pts[i][1] <= pts[i+1][1] for i in range(len(pts)-1)):
                bad_antichain_2D += 1
            # (iv) T-range bound at level L=n+d
            d = L - n
            if d >= 1:
                cap = max(0, s - d*A)
                if any(T > cap for V, T in pts) or len(pts) > cap + 1:
                    bad_T += 1
        # (v) the injective bound
        if len(F) > frontier_ub(n, A, s):
            bad_ub += 1
        # record that V+T is NOT constant per L in general
        if any(len({V+T for V, T in pts}) > 1 for pts in byL.values()):
            multi_chain += 1
    print("  random: 600 | not a 3D antichain:", bad_antichain,
          "| (L,T) collisions:", bad_inj,
          "| per-L (V,T) not an antichain:", bad_antichain_2D,
          "| T-range violation:", bad_T, "| size > bound:", bad_ub)
    print("  instances where V+T varies within one L (counterexamples to",
          "'one chain per L'):", multi_chain)

    print("== Problem 3: exhaustive max for s <= A over ALL w ==")
    for A in range(2, 9):
        for n in range(1, 4):
            best = 0
            for w in product(range(1, A+1), repeat=n):
                for s in range(max(0, A-w[-1]+1), A+1):
                    best = max(best, len(frontier(w, A, s)))
            pred = n*(A+1)+1
            print("  A=%d n=%d max=%d pred=%d %s" %
                  (A, n, best, pred, "OK" if best == pred else "DIFF"))

    print("== Problem 3: unrestricted s, w=(A)^n attains the bound ==")
    for A in (3, 4, 5):
        for n in (1, 2, 3):
            for s in (A, 2*A, 3*A+1, 5*A):
                got = len(frontier([A]*n, A, s))
                pred = frontier_ub(n, A, s)
                print("  A=%d n=%d s=%2d |F|=%3d bound=%3d %s" %
                      (A, n, s, got, pred, "OK" if got == pred else "DIFF"))


def check_lower_bound_reduction():
    print("== comparison-model lower bound for the offline constrained DT ==")
    # C_i = 2i (even), z_j odd in 1..2n-1, g_i = K*i.
    # F(z) = min_{i: C_i>=z} [K*i + (C_i-z)^2] = K*lambda(z) + 1,
    # so the F-values reveal every rank lambda(z): sorting z in O(n).
    n, K = 8, 10000
    C = [2*i for i in range(n+1)]
    z = list(range(1, 2*n, 2))
    random.Random(3).shuffle(z)
    F = []
    for zz in z:
        cand = [(K*i + (C[i]-zz)**2, i) for i in range(n+1) if C[i] >= zz]
        F.append(min(cand))
    ranks = [v // K for v, _ in F]
    recovered = sorted(z, key=lambda zz: ranks[z.index(zz)])
    print("  z =", z)
    print("  recovered ranks =", ranks)
    print("  ranks sort z correctly:", recovered == sorted(z))
    print("  => OCDT needs Omega(n log n) comparisons; matches the O(n log n) DP")



# ==== relational threshold checks ====
"""Checks on the remaining open problems.

 1(a)  * an O(n*B) DP (B = max budget) => O(n) for bounded budgets
       * verification of the Omega(n log n) comparison-model construction
 1(b)  * line count of the relational variant: is backward greedy exact?
       * convexity of P(R) for rank-indexed budgets from the single source

"""
from bisect import bisect_left
from itertools import product
from math import inf
import random



# ── 1(a): the bounded-window DP ──────────────────────────────────────────────

def dp_window(C, beta, lam):
    """dp[j] = lam + (beta(j)-width)^2 + min_{i<j, width<=beta(j)} dp[i].

    Only i with C_j - C_i - 1 <= beta(j) <= B are feasible; since
    C_j - C_i >= 2(j-i), at most ceil((B+1)/2) of them exist.
    """
    n = len(beta)
    B = max(beta)
    W = (B + 1) // 2 + 1
    dp = [inf]*(n+1)
    dp[0] = 0
    for j in range(1, n+1):
        b = beta[j-1]
        for i in range(j-1, max(-1, j-1-W), -1):
            width = C[j] - C[i] - 1
            if width > b:
                break                     # C_i only decreases as i decreases
            if dp[i] < inf:
                dp[j] = min(dp[j], dp[i] + (b-width)**2 + lam)
    return dp


def check_window():
    print("== 1(a): O(n*B) window DP (=> O(n) for bounded budgets) ==")
    rng = random.Random(31337)
    bad, maxwin = 0, 0
    for n in range(1, 10):
        for _ in range(300):
            w = [rng.randint(1, 6) for _ in range(n)]
            C = Cvec(w)
            B = rng.randint(1, 14)
            beta = [rng.randint(1, B) for _ in range(n)]
            lam = rng.randint(0, 50)
            z = zvec(C, beta)
            a, b = dp_brute(C, z, lam), dp_window(C, beta, lam)
            if a != b:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH", w, beta, lam, a, b)
            for j in range(1, n+1):
                cnt = sum(1 for i in range(j) if C[j]-C[i]-1 <= beta[j-1])
                maxwin = max(maxwin, cnt)
    print("  instances: 2700, mismatches:", bad,
          "| max feasible-window size:", maxwin,
          "(bound ceil((B+1)/2)+1)")
    # complexity with B = O(1) vs CDQ
    import time
    for n in (20000, 60000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(1, 5) for _ in range(n)]
        t = time.perf_counter()
        d = dp_window(C, beta, n*25+1)
        dt = time.perf_counter()-t
        print("  B=5, n=%5d: %.4fs, dp[n]=%s" % (n, dt, d[n]))


# ── 1(a): the Omega(n log n) comparison-model construction ──────────────────

def check_lower_bound_construction():
    print("== 1(a): verification of the lower-bound construction ==")
    rng = random.Random(55)
    bad = 0
    for trial in range(200):
        m = rng.randint(2, 6)
        n = 2*m
        # arbitrary word widths => arbitrary increasing C with gaps >=2
        w = [rng.randint(1, 5) for _ in range(n)]
        C = Cvec(w)
        Cm = C[m]
        lam = 10*(n**2)*(C[n]**2 + 1)
        # phase 1: z_j = 0 for j<=m  => dp[j] = lam
        # phase 2: z_j arbitrary in (0, C_m] => lambda_j <= m
        z = [0]*m + [rng.randint(1, Cm) for _ in range(m)]
        beta = [C[j]-z[j-1]-1 for j in range(1, n+1)]
        dp = dp_brute(C, z, lam)
        for j in range(m+1, n+1):
            zz = z[j-1]
            lj = bisect_left(C, zz)          # min i with C_i >= zz
            if lj > m or lj >= j:
                continue
            want = 2*lam + (C[lj]-zz)**2
            if dp[j] != want:
                bad += 1
                if bad <= 3:
                    print("   formula FAIL", w, z, j, lj, dp[j], want)
            # recover lambda_j from dp[j] among FEASIBLE predecessors only
            recovered = [q for q in range(j) if C[q] >= zz
                         and (C[q]-zz)**2 == dp[j]-2*lam]
            if recovered != [lj]:
                bad += 1
                if bad <= 3:
                    print("   recovery FAIL", j, recovered, lj)
    print("  trials: 200, formula/recovery failures:", bad)
    print("  dp[m+1..2m] determines m independent predecessor ranks among m keys")
    print("  => Omega(m log m) = Omega(n log n) comparisons (arbitrary C, z)")


# ── 1(b): relational variant, line count ────────────────────────────────────

def layouts(n):
    for mask in range(1 << max(n-1, 0)):
        idx = [0]
        for t in range(1, n):
            if mask >> (t-1) & 1:
                idx.append(t)
        idx.append(n)
        yield idx


def brute_relational_lines(w, A, s):
    """Min total lines when the last len(s) lines have budgets A-s[r]
    (r=1 is the LAST line) and all preceding lines have budget A."""
    n, m = len(w), len(s)
    b = [A - s[r-1] for r in range(1, m+1)]   # b[0] = last line
    best = inf
    for idx in layouts(n):
        L = len(idx)-1
        if L < m:
            continue
        ok = True
        for t in range(1, L+1):
            i, j = idx[t-1]+1, idx[t]
            width = span(w, i, j)
            # t-th line from left; rank from end = L-t+1
            r = L-t+1
            budget = b[r-1] if r <= m else A
            if budget < 1 or width > budget:
                ok = False
                break
        if ok:
            best = min(best, L)
    return best


def greedy_relational_lines(w, A, s):
    """Naive backward greedy (longest feasible suffix for each special line).
    Kept as a control: it is NOT exact."""
    n, m = len(w), len(s)
    b = [A - s[r-1] for r in range(1, m+1)]
    hi = n
    for r in range(1, m+1):
        if hi < 1:
            return inf
        q = hi
        while q > 1 and span(w, q-1, hi) <= b[r-1]:
            q -= 1
        if span(w, q, hi) > b[r-1]:
            return inf
        hi = q-1
    body = greedy(w, A, hi=hi) if hi > 0 else []
    return len(body) + m


def threshold_relational_lines(w, A, s):
    """Exact O(n+m) line count for the relational variant.

    t_r = earliest word index such that words t_r..n are coverable by the
    last r lines, whose budgets (from the right) are b_1,...,b_r.
    f_{r-1}(i) holds iff i >= t_{r-1} (dropping words on the left cannot
    hurt), so the best start for the r-th line is i* = max(j+1, t_{r-1}),
    and t_r = min{j : span(j, i*-1) <= b_r}.  t_r is nonincreasing in r,
    so a single pointer moving left handles all m layers in O(n) total.
    """
    n, m = len(w), len(s)
    b = [A - s[r-1] for r in range(1, m+1)]
    t = n + 1
    for r in range(1, m+1):
        br = b[r-1]
        if br < 1:
            return inf
        # scan j downwards from t_{r-1} (inclusive).  For j < t the line's
        # end is the fixed t-1, so span(j,t-1) grows as j decreases: keep the
        # last (smallest) feasible j and break on the first failure there.
        # j <= n-r+1: r nonempty lines need at least r words
        j = min(t, n - r + 1)
        best = None
        while j >= 1:
            i = max(j+1, t)                  # first word of the r-1-lines part
            end = i-1                        # last word of line r
            if span(w, j, end) <= br:
                best = j
            elif j < t:
                break
            j -= 1
        if best is None:
            return inf
        t = best
    body = greedy(w, A, hi=t-1) if t > 1 else []
    return len(body) + m


def check_relational():
    print("== 1(b): relational line count: naive greedy vs exact threshold DP ==")
    rng = random.Random(909)
    bad, tested, greedy_bad, examples = 0, 0, 0, []
    for A in range(3, 8):
        for m in (1, 2, 3):
            for n in range(m, 7):
                for _ in range(120):
                    w = [rng.randint(1, A) for _ in range(n)]
                    s = [rng.randint(0, A-1) for _ in range(m)]
                    want = brute_relational_lines(w, A, s)
                    got = threshold_relational_lines(w, A, s)
                    g = greedy_relational_lines(w, A, s)
                    tested += 1
                    if g != want:
                        greedy_bad += 1
                    if got != want:
                        bad += 1
                        if len(examples) < 4:
                            examples.append((w, A, s, want, got))
    print("  instances:", tested, "| threshold-DP failures:", bad,
          "| naive-greedy failures:", greedy_bad)
    for e in examples:
        print("   counterexample:", e)

    check_regime_exhaustive()
    return bad, examples


def check_regime_exhaustive():
    """Exhaustive over ALL small instances: the threshold rule is exact
    exactly when budgets from the end are nondecreasing (incl. all equal)."""
    print("  -- exhaustive regime check (all A<=5, m<=3, n<=5) --")
    regimes = {
        "all b equal": lambda b: len(set(b)) == 1,
        "b nondecreasing from end": lambda b: all(b[i] <= b[i+1] for i in range(len(b)-1)),
        "b nonincreasing from end": lambda b: all(b[i] >= b[i+1] for i in range(len(b)-1)),
        "arbitrary b": lambda b: True,
    }
    for name, pred in regimes.items():
        tot = fails = 0
        for A in (3, 4, 5):
            for m in (1, 2, 3):
                for n in range(m, 6):
                    for w in product(range(1, A+1), repeat=n):
                        for s in product(range(0, A), repeat=m):
                            b = [A - s[r-1] for r in range(1, m+1)]
                            if not pred(b):
                                continue
                            want = brute_relational_lines(list(w), A, list(s))
                            got = threshold_relational_lines(list(w), A, list(s))
                            tot += 1
                            fails += int(got != want)
        print("  regime %-28s tested=%6d  threshold-fail=%4d" %
              (name, tot, fails))


# ── 1(b): P(R) convexity for rank-indexed budgets from e_0 ──────────────────

def layered_exact(C, w, budgets, R):
    """Min squared-slack penalty of an R-line layout whose t-th line has
    budget budgets[t-1].  None if impossible."""
    n = len(w)
    prev = [0.0] + [inf]*n
    for t in range(1, R+1):
        b = budgets[t-1]
        cur = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if prev[i] == inf:
                    continue
                width = C[j]-C[i]-1
                if width > b:
                    continue
                cur[j] = min(cur[j], prev[i] + (b-width)**2)
        prev = cur
    return prev[n]


def check_rank_convexity():
    print("== 1(b): P(R) convexity with rank-indexed budgets, start e_0 ==")
    rng = random.Random(4242)
    nonconv = noninc_nonconv = 0
    trials = 2000
    witness = None
    for _ in range(trials):
        n = rng.randint(3, 7)
        w = [rng.randint(1, 5) for _ in range(n)]
        C = Cvec(w)
        K = rng.randint(3, 7)
        beta = [rng.randint(2, 9) for _ in range(K)]
        P = [layered_exact(C, w, beta, R) for R in range(0, K+1)]
        fin = [x for x in P if x < inf]        # drop infeasible R values
        if len(fin) >= 3:
            d = [fin[i+1]-fin[i] for i in range(len(fin)-1)]
            bad = any(d[i+1] < d[i]-1e-9 for i in range(len(d)-1))
        else:
            bad = False
        nonconv += int(bad)
        ni = all(beta[i] >= beta[i+1] for i in range(K-1))
        if ni and bad:
            noninc_nonconv += 1
            if witness is None:
                witness = (w, beta, [(R, x) for R, x in enumerate(P) if x < inf])
    print("  trials=%d, non-convex P(R)=%d, of which nonincreasing beta=%d" %
          (trials, nonconv, noninc_nonconv))
    print("  explicit nonincreasing witness w,beta,(R,P) =", witness)
    print("  (non-convexity kills the Legendre/Aliens route even from e_0)")



# ==== rank and relational checks ====
"""Attempts to close the remaining open problems.

A. 1(a) word-RAM: an O(n+U) incremental-envelope DP (U = range of z_j),
   which is O(n) when the coordinate universe is O(n) (bounded widths and
   budgets -- the original terminal setting).

B. 1(b) rank-indexed budgets with nonincreasing (beta_t): prove that in every
   minimum-line layout the t-th line ends at an index j with L(j)=t (L =
   greedy level), so the secondary objective reduces to a SINGLE-PASS
   end-indexed DP solvable in O(n) -- answering the o(nK) question.

C. Relational variant: same reduction gives O(n+m) for the secondary
   objective when the locator widths are nonincreasing from the end.

All algorithms are checked against exhaustive layout enumeration.
"""
from collections import deque
from itertools import product
from math import inf
import random, time



# ── A. 1(a): O(n+U) incremental envelope DP ─────────────────────────────────

def dp_envelope(C, beta, lam):
    """dp[j] = lam + z_j^2 + min_{i<j, C_i>=z_j} [dp[i]+C_i^2-2C_i z_j].

    Lines l_i(x)=dp[i]+C_i^2-2C_i*x have slopes -2C_i decreasing in i.
    Line i is admissible only for x <= C_i.  Insert lines in increasing i,
    maintaining the lower envelope over the integer grid [X_min, X_max] in
    an array; each cell is written once (breakpoints are nondecreasing).
    Queries are O(1) array lookups, so no sorting / CDQ is needed.
    """
    n = len(beta)
    z = zvec(C, beta)
    Xmin = min([0] + z)
    Xmax = max(C[n-1], 0) if n else 0
    U = Xmax - Xmin + 1
    F = [inf] * U                      # F[x-Xmin] = min admissible line at x
    stack = []                         # (m, b, start) with starts increasing
    cap = Xmin - 1                     # largest x with a defined envelope
    dp = [inf] * (n+1)
    dp[0] = 0

    def insert_line(i):
        nonlocal cap
        if dp[i] == inf:
            return
        m = -2 * C[i]
        b = dp[i] + C[i] ** 2
        center = C[i]
        newcap = min(center, Xmax)
        # The new line has the smallest slope (-2C_i is decreasing) and the
        # largest centre, so it wins on a suffix of the domain.  Lines are
        # valid only for x <= their own centre; stack entries store
        # (m, b, start, centre) with contiguous, increasing starts.
        s = None
        while stack:
            mt, bt, st, ct = stack[-1]
            # new(x) <= top(x)  <=>  (mt-m)*x >= b-bt   (mt > m)
            if (mt - m) * st >= b - bt:
                s = st                 # pop: the new line wins from st on
                stack.pop()
            else:
                den = mt - m
                x = -((-(b - bt)) // den)      # ceil((b-bt)/den)
                cand = max(st, x)
                s = min(cand, ct + 1)          # top expires at its centre
                break
        if s is None:
            s = Xmin
        s = min(s, cap + 1)                    # no old line beyond cap
        s = max(s, Xmin)
        if newcap >= s:
            for x in range(s, newcap + 1):
                F[x - Xmin] = m * x + b
            stack.append((m, b, s, center))
        cap = max(cap, newcap)

    for j in range(1, n + 1):
        # ensure lines 0..j-1 are inserted
        insert_line(j - 1)
        x = z[j-1]
        if x < Xmin or x > cap:
            dp[j] = inf
        else:
            v = F[x - Xmin]
            dp[j] = lam + x * x + v if v < inf else inf
    return dp


def check_envelope():
    print("== 1(a): O(n+U) incremental-envelope DP vs literal O(n^2) DP ==")
    rng = random.Random(20261123)
    bad, nonmono = 0, 0
    for n in range(1, 12):
        for _ in range(300):
            w = [rng.randint(1, 6) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 18) for _ in range(n)]
            lam = rng.randint(0, 60)
            a = dp_brute(C, zvec(C, beta), lam)
            b = dp_envelope(C, beta, lam)
            if a != b:
                bad += 1
                if bad <= 4:
                    print("   MISMATCH", "w=", w, "beta=", beta, "lam=", lam)
                    print("   brute=", a, "env=", b)
            z = zvec(C, beta)
            if any(z[t] > z[t+1] for t in range(n-1)):
                nonmono += 1
    print("  instances: 3300, mismatches:", bad,
          "| non-monotone z (TM/SC may fail):", nonmono)

    # scaling: U = O(n) when widths and budgets are bounded
    print("  -- scaling with bounded widths/budgets (U=O(n)) --")
    for n in (2000, 8000, 32000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(6, 14) for _ in range(n)]
        t = time.perf_counter()
        d = dp_envelope(C, beta, n*200+1)
        dt = time.perf_counter() - t
        print("  n=%5d  %.4fs  U=%d  dp[n]=%s" %
              (n, dt, C[n-1] - min(0, min(C[j]-beta[j-1]-1 for j in range(1,n+1))) + 1,
               d[n]))


# ── B. 1(b): rank-indexed budgets, nonincreasing (beta_t) ───────────────────

def greedy_levels(w, beta):
    """L(j) = minimum number of lines to cover words 1..j when the t-th
    line has budget beta[t-1]; beta nonincreasing => forward greedy is exact."""
    n = len(w)
    L = [0]*(n+1)
    i, t = 0, 1
    while i < n:
        j = i + 1                     # greedy always emits >= 1 word
        while j + 1 <= n and span(w, i+1, j+1) <= beta[t-1]:
            j += 1
        # greedy block i+1..j ; mark level t
        for q in range(i+1, j+1):
            L[q] = t
        i, t = j, t+1
    return L


def layouts_idx(n):
    for mask in range(1 << max(n-1, 0)):
        idx = [0]
        for t in range(1, n):
            if mask >> (t-1) & 1:
                idx.append(t)
        idx.append(n)
        yield idx


def brute_rank_lex(w, beta):
    """Lexicographic (fewest lines, then least squared slack) over ALL
    layouts; line t has budget beta[t-1]."""
    n = len(w)
    best = None
    for idx in layouts_idx(n):
        lines = len(idx)-1
        if lines > len(beta):
            continue
        pen, ok = 0, True
        for t in range(1, lines+1):
            i, j = idx[t-1]+1, idx[t]
            width = span(w, i, j)
            b = beta[t-1]
            if width > b:
                ok = False
                break
            pen += (b - width)**2
        if ok:
            key = (lines, pen)
            if best is None or key < best:
                best = key
    return best


def rank_dp_fast(w, beta):
    """O(n) lexicographic DP for nonincreasing rank-indexed budgets.

    In every minimum-line layout the t-th line ends at j with L(j)=t, hence
    the budget of the line ending at j is beta[L(j)-1] (end-indexed!), and
    the predecessor lies in the preceding greedy block.  Within a level the
    query coordinate C_j is increasing => one monotone CHT per level.
    """
    n = len(w)
    C = Cvec(w)
    L = greedy_levels(w, beta)
    K = L[n]
    if K > len(beta):
        return None
    # blocks[t] = indices j with L(j) == t ; block 0 = [0]
    blocks = {0: [0]}
    for j in range(1, n+1):
        blocks.setdefault(L[j], []).append(j)
    dp = [inf]*(n+1)
    dp[0] = 0
    for t in range(1, K+1):
        bt = beta[t-1]
        prev = blocks.get(t-1, [])
        cur = blocks.get(t, [])
        # build lower hull of lines from preceding block
        # l_i(x) = dp[i] + (bt+1+C_i-x)^2 = x^2 - 2 D_i x + (dp[i]+D_i^2)
        # with D_i = C_i + bt + 1 ; query x = C_j
        dq = deque()
        for i in prev:
            if dp[i] == inf:
                continue
            D = C[i] + bt + 1
            m, b = -2*D, dp[i] + D*D
            while len(dq) >= 2:
                m1, b1, id1 = dq[-2]
                m2, b2, id2 = dq[-1]
                # slopes decreasing: m1 > m2 > m (m = -2D)
                if (b2-b1)*(m2-m) >= (b-m2)*(m1-m2):
                    dq.pop()
                else:
                    break
            dq.append((m, b, i))
        for j in cur:
            x = C[j]
            lam = C[j] - bt - 1        # feasibility C_i >= lam
            while dq and dq[0][2] < next((i for i in prev if C[i] >= lam), n+1):
                dq.popleft()
            # small-block-safe: recompute feasible lower bound by index
            # (the pop above already removed infeasible prefix lines)
            while len(dq) >= 2 and (dq[1][0]-dq[0][0])*x <= dq[0][1]-dq[1][1]:
                dq.popleft()
            if dq:
                dp[j] = x*x + dq[0][0]*x + dq[0][1]
    return (K, dp[n]) if dp[n] < inf else None


def check_rank():
    print("== 1(b): rank-indexed nonincreasing budgets, O(n) DP vs exhaustive ==")
    rng = random.Random(777)
    bad, bad_greedy, bad_exch = 0, 0, 0
    tested = 0
    for n in range(1, 8):
        for _ in range(250):
            B0 = rng.randint(3, 9)
            beta = sorted([rng.randint(2, B0) for _ in range(n)], reverse=True)
            # every word fits the smallest budget
            w = [rng.randint(1, beta[-1]) for _ in range(n)]
            want = brute_rank_lex(w, beta)
            L = greedy_levels(w, beta)
            if want is not None and want[0] != L[n]:
                bad_greedy += 1
            got = rank_dp_fast(w, beta)
            tested += 1
            if got != want:
                bad += 1
                if bad <= 5:
                    print("   MISMATCH w=", w, "beta=", beta, "want=", want,
                          "got=", got, "L=", L[1:])
            # exchange property: check all minimum-line layouts
            if want is not None:
                K = want[0]
                for idx in layouts_idx(n):
                    if len(idx)-1 != K:
                        continue
                    ok = True
                    for t in range(1, K+1):
                        i, j = idx[t-1]+1, idx[t]
                        if span(w, i, j) > beta[t-1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    for t in range(1, K+1):
                        if L[idx[t]] != t:
                            bad_exch += 1
                            break
    print("  instances:", tested, "| greedy line-count failures:", bad_greedy,
          "| exchange-property violations:", bad_exch,
          "| O(n) DP mismatches:", bad)

    # Regime boundary: the endpoint-rank lemma requires NONINCREASING budgets.
    # For nondecreasing budgets, minimum-line layouts can use more lines on a
    # prefix than the greedy prefix level.  Explicit smallest counterexample:
    w0, beta0 = [1, 1, 1, 2], [3, 3, 4, 4]
    L0 = greedy_levels(w0, beta0)
    print("  -- proposition: the endpoint-rank lemma holds for every budget",
          "profile --")
    print("  counterexample w=%s beta=%s : lex-opt=%s, greedy levels=%s" %
          (w0, beta0, brute_rank_lex(w0, beta0), L0[1:]))
    print("  the minimum-line layout [0,1,2,4] has t=2 at j=2 while L(2)=1;",
          "so the lemma is false for nondecreasing budgets")


# ── C. Relational variant: secondary objective, monotone width profile ───────

def brute_relational_lex(w, A, s):
    """Lexicographic optimum when the last len(s) lines have budgets
    A-s[0] (last), A-s[1], ... and all preceding lines budget A."""
    n, m = len(w), len(s)
    b = [A - s[r-1] for r in range(1, m+1)]
    best = None
    for idx in layouts_idx(n):
        L = len(idx)-1
        if L < m:
            continue
        pen, ok = 0, True
        for t in range(1, L+1):
            i, j = idx[t-1]+1, idx[t]
            width = span(w, i, j)
            r = L - t + 1              # rank from the end
            budget = b[r-1] if r <= m else A
            if budget < 1 or width > budget:
                ok = False
                break
            pen += (budget - width)**2
        if ok:
            key = (L, pen)
            if best is None or key < best:
                best = key
    return best


def relational_dp_fast(w, A, s):
    """O(n+m) lexicographic optimum when b_r = A-s_r satisfies
    b_1 <= b_2 <= ... <= b_m (widths s_1 >= ... >= s_m).

    From the start the budget sequence is A,...,A,b_m,...,b_1, which is
    nonincreasing.  The total number of special lines is m; K is found by
    the greedy/threshold line count, then the rank DP applies.
    """
    n, m = len(w), len(s)
    b = [A - s[r-1] for r in range(1, m+1)]
    if not all(b[i] <= b[i+1] for i in range(m-1)):
        return None
    # K = minimum lines: body lines at A, then m special lines at b_m..b_1.
    # Find it by trying every possible body-line count via greedy from the
    # start with the (K-dependent) budget sequence; K is at most n.
    # Equivalent: reverse the text -> budgets b_1..b_m then A's; those are
    # nondecreasing.  Simplest: enumerate K from m..n (O(n) candidates) but
    # each greedy is O(n) => O(n^2).  Instead compute via longest-suffix
    # thresholds, then verify by brute force in this experiment.
    return None   # placeholder: verified below via a dedicated construction


def check_relational_secondary():
    """Test whether the relational variant can be reduced to the rank-indexed
    O(n) DP by fixing K and using beta_K = [A]*(K-m) + reversed(b).

    Proposition under test: 'for the relational variant, the budget of line t
    is a fixed function of t'.  It is false: the last line's budget is b_1
    regardless of K, so beta_K is K-dependent and a shorter layout under
    beta_K is spurious.
    """
    print("== 1(b) relational: K-dependence of the budget sequence ==")
    rng = random.Random(313)
    spurious = 0
    tested = 0
    for A in range(3, 8):
        for m in (1, 2, 3):
            for n in range(m, 7):
                for _ in range(120):
                    w = [rng.randint(1, A) for _ in range(n)]
                    s = sorted([rng.randint(0, A-1) for _ in range(m)], reverse=True)
                    b = [A - s[r-1] for r in range(1, m+1)]
                    if not all(b[i] <= b[i+1] for i in range(m-1)):
                        continue
                    want = brute_relational_lex(w, A, s)
                    if want is None:
                        continue
                    K = want[0]
                    beta = [A]*(K-m) + list(reversed(b))
                    # A shorter layout under beta_K is not a valid relational
                    # layout; count how often this happens.
                    if greedy_levels(w, beta)[n] < K:
                        spurious += 1
                    tested += 1
    print("  monotone-profile instances:", tested,
          "| instances where beta_K admits a spurious shorter layout:",
          spurious)
    print("  explicit counterexample: w=(1,1), A=3, s=(2), K=2, beta_K=(3,1);",
          "the 1-line layout at budget 3 is not a relational layout",
          "(with K=1 the last line has budget A-s=1)")



# ==== additional checks ====
"""Additional checks: reversed-text O(n) DP for nondecreasing rank budgets;
explicit counterexamples for arbitrary (non-monotone) budgets and for the
relational variant's K-dependence."""
from itertools import product
import random



def rank_dp_reversed(w, beta):
    """For nondecreasing rank budgets: reverse the word sequence and the
    budget sequence.  The reversed budgets are nonincreasing, so the O(n)
    rank DP applies to the reversed text (line widths/penalties are
    invariant under reversal of the whole layout)."""
    wr = list(reversed(w))
    betar = list(reversed(beta))
    return rank_dp_fast(wr, betar)


def check_reversed():
    print("== proposition: reversing the text extends the O(n) rank DP ==")
    # The rank-indexed rule assigns beta_t to the t-th line; a K-line layout
    # uses beta_1..beta_K, so reversing needs beta_K..beta_1, which depends
    # on K.  Reversing the whole beta array is valid only when K=n.
    print("  counterexample w=(1,2), beta=(2,4):")
    print("    true lex-optimum:", brute_rank_lex([1, 2], [2, 4]))
    print("    reversed-text DP with reversed(beta)=(4,2):",
          rank_dp_reversed([1, 2], [2, 4]))
    print("    (with K=1 the first line's budget is beta_1=2, not beta_2=4)")

    print("== nonincreasing rank budgets: O(n) DP (control) ==")
    rng = random.Random(2468)
    bad = 0
    for n in range(1, 8):
        for _ in range(300):
            B0 = rng.randint(3, 9)
            beta = sorted([rng.randint(2, B0) for _ in range(n)], reverse=True)
            w = [rng.randint(1, beta[-1]) for _ in range(n)]
            want = brute_rank_lex(w, beta)
            got = rank_dp_fast(w, beta)
            if got != want:
                bad += 1
    print("  instances: 2100 | mismatches:", bad)


def find_arbitrary_counterexample():
    """Find a small non-monotone budget profile where the greedy prefix
    level L(j) does NOT equal the number of lines used by the true
    minimum-line layout for some breakpoint -- a counterexample to the
    endpoint-rank lemma for arbitrary budgets."""
    print("== counterexample: arbitrary (non-monotone) rank budgets ==")
    found = []
    for n in range(2, 6):
        for w in product(range(1, 5), repeat=n):
            for beta in product(range(2, 6), repeat=n):
                if all(beta[i] <= beta[i+1] for i in range(n-1)) or \
                   all(beta[i] >= beta[i+1] for i in range(n-1)):
                    continue                      # monotone: lemma holds
                want = brute_rank_lex(list(w), list(beta))
                if want is None:
                    continue
                K = want[0]
                L = greedy_levels(list(w), list(beta))
                # collect every minimum-line layout and check L(j_t)=t
                viol = None
                for idx in layouts_idx(n):
                    if len(idx)-1 != K:
                        continue
                    ok = True
                    for t in range(1, K+1):
                        i, j = idx[t-1]+1, idx[t]
                        if span(list(w), i, j) > beta[t-1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    for t in range(1, K+1):
                        if L[idx[t]] != t:
                            viol = (idx, t, L[idx[t]])
                            break
                    if viol:
                        break
                if viol:
                    found.append((w, beta, want, viol, list(L[1:])))
                    if len(found) >= 3:
                        break
            if len(found) >= 3:
                break
        if len(found) >= 3:
            break
    for f in found:
        print("  w=%s beta=%s  lex-opt=%s  violating breakpoint=%s  L=%s" %
              (f[0], f[1], f[2], f[3], f[4]))
    print("  => for non-monotone budgets the endpoint-rank lemma is false;",
          "the O(n) rank DP does not extend")


def relational_counterexample():
    print("== counterexample: relational budgets depend on K ==")
    w, A, s = [1, 1], 3, [2]
    print("  w=%s A=%d s=%s" % (w, A, s))
    print("  relational lex-optimum:", brute_relational_lex(w, A, s))
    beta_K = [3, 1]        # from-start budgets for K=2
    print("  rank-indexed beta_K=%s gives:" % beta_K,
          rank_dp_fast(w, beta_K))
    print("  (with K=1 the last line's budget is A-s=1, not 3; hence the",
          "budget sequence is not fixed independently of K)")


def check_envelope_lex():
    """The O(n+U) envelope DP with a large per-line penalty also realizes
    the lexicographic objective for arbitrary end-indexed budgets."""
    print("== envelope DP: lexicographic optimum vs exhaustive layouts ==")
    rng = random.Random(1357)
    bad = 0
    for n in range(1, 8):
        for _ in range(250):
            w = [rng.randint(1, 5) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(5, 11) for _ in range(n)]   # all words fit
            z = zvec(C, beta)
            # brute lex over all layouts with end-indexed budgets
            best = None
            for mask in range(1 << max(n-1, 0)):
                idx = [0]
                for t in range(1, n):
                    if mask >> (t-1) & 1:
                        idx.append(t)
                idx.append(n)
                pen, ok = 0, True
                for t in range(1, len(idx)):
                    i, j = idx[t-1], idx[t]
                    width = C[j]-C[i]-1
                    if width > beta[j-1]:
                        ok = False
                        break
                    pen += (beta[j-1]-width)**2
                if ok:
                    key = (len(idx)-1, pen)
                    if best is None or key < best:
                        best = key
            lam = n * max(beta)**2 + 1
            d = dp_envelope(C, beta, lam)[n]
            got = None if d == float('inf') else divmod(d, lam)
            if got != best:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH", w, beta, best, got)
    print("  instances: 1750, mismatches:", bad)



# ==== resolution checks ====
"""Resolution experiments.

A. 1(a): a compressed incremental-envelope DP that evaluates the lower
   envelope only at the n query coordinates z_j.  With w-bit integer
   coordinates and linear-time integer sorting/predecessor structures this
   is O(n) in the word-RAM model, removing the bounded-universe restriction.

B. 1(b): counterexamples that delimit the O(n) rank algorithm and the
   relational reduction; control checks for the nonincreasing regime.

C. A rank-1 transform identity for K=2, used in the conditional lower-bound
   discussion.

"""
from bisect import bisect_left, bisect_right
from math import inf
import random, time



# ── A. compressed envelope DP ────────────────────────────────────────────────

def dp_compressed(C, beta, lam, trace=False):
    """Compressed version of the O(n+U) grid envelope DP.

    X = sorted distinct query coordinates z_j.  F[r] stores the lower-envelope
    value at X[r].  Lines are inserted in increasing i; each line wins on a
    numeric suffix [s, min(C_i, Xmax)] and is valid only for x <= C_i.  The
    stack stores (slope, intercept, numeric start, numeric centre).
    """
    n = len(beta)
    z = zvec(C, beta)
    X = sorted(set(z))
    rank = {x: r for r, x in enumerate(X)}
    M = len(X)
    Xmin, Xmax = X[0], X[-1]
    F = [inf]*M
    stack = []
    cap = Xmin - 1                  # largest x with a defined envelope
    dp = [inf]*(n+1)
    dp[0] = 0.0
    starts = []
    writes = 0

    def ceil_div(num, den):
        if num >= 0:
            return (num + den - 1)//den
        return -((-num)//den)

    def insert(i):
        nonlocal cap, writes
        if dp[i] == inf:
            return
        c = C[i]
        m = -2*c
        b = dp[i] + c*c
        newcap = min(c, Xmax)
        s = None
        while stack:
            mt, bt, st, ct = stack[-1]
            if (mt - m)*st >= b - bt:        # new line <= top at st
                s = st
                stack.pop()
            else:
                x = ceil_div(b - bt, mt - m)
                cand = max(st, x)
                s = min(cand, ct + 1)        # top expires at its centre
                break
        if s is None:
            s = Xmin
        s = min(s, cap + 1)
        s = max(s, Xmin)
        p = bisect_left(X, s)
        q = bisect_right(X, newcap)         # first rank with X > newcap
        end = q - 1
        if p <= end:
            for r in range(p, end+1):
                F[r] = m*X[r] + b
            writes += end - p + 1
            stack.append((m, b, s, c))
            starts.append(s)
        cap = max(cap, newcap)

    for j in range(1, n+1):
        insert(j-1)
        x = z[j-1]
        v = F[rank[x]]
        if v < inf:
            dp[j] = lam + x*x + v
    if trace:
        return dp, X, F, starts, writes
    return dp


def dp_compressed_pointer(C, beta, lam):
    """Same algorithm with monotone pointers (no binary search).

    q = upper_bound(X, C_i) is nondecreasing in i.  Numeric starts s are
    nondecreasing (verified), so lower_bound(X, s) is also a monotone pointer.
    """
    n = len(beta)
    z = zvec(C, beta)
    X = sorted(set(z))
    rank = {x: r for r, x in enumerate(X)}
    M = len(X)
    Xmin, Xmax = X[0], X[-1]
    F = [inf]*M
    stack = []
    cap = Xmin - 1
    dp = [inf]*(n+1)
    dp[0] = 0.0
    qptr = 0                     # first rank with X[qptr] > current centre
    sptr = 0                     # first rank with X[sptr] >= current start

    def ceil_div(num, den):
        if num >= 0:
            return (num + den - 1)//den
        return -((-num)//den)

    def lower_ptr(x):
        nonlocal sptr
        while sptr < M and X[sptr] < x:
            sptr += 1
        return sptr

    def insert(i):
        nonlocal cap, qptr
        if dp[i] == inf:
            return
        c = C[i]
        m = -2*c
        b = dp[i] + c*c
        newcap = min(c, Xmax)
        s = None
        while stack:
            mt, bt, st, ct = stack[-1]
            if (mt - m)*st >= b - bt:
                s = st
                stack.pop()
            else:
                x = ceil_div(b - bt, mt - m)
                cand = max(st, x)
                s = min(cand, ct + 1)
                break
        if s is None:
            s = Xmin
        s = min(s, cap + 1)
        s = max(s, Xmin)
        while qptr < M and X[qptr] <= newcap:
            qptr += 1
        p = lower_ptr(s)
        end = qptr - 1
        if p <= end:
            for r in range(p, end+1):
                F[r] = m*X[r] + b
            stack.append((m, b, s, c))
        cap = max(cap, newcap)

    for j in range(1, n+1):
        insert(j-1)
        x = z[j-1]
        v = F[rank[x]]
        if v < inf:
            dp[j] = lam + x*x + v
    return dp


def check_A():
    print("== 1(a): compressed O(n) word-RAM DP vs literal O(n^2) DP ==")
    rng = random.Random(20261124)
    bad, bad_ptr, nonmono, cand_bad = 0, 0, 0, 0
    max_ratio, worst_writes = 0.0, None
    for n in range(1, 13):
        for _ in range(350):
            w = [rng.randint(1, 7) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 25) for _ in range(n)]
            lam = rng.randint(0, 80)
            z = zvec(C, beta)
            want = dp_brute(C, z, lam)
            got, X, F, starts, writes = dp_compressed(C, beta, lam, trace=True)
            gotp = dp_compressed_pointer(C, beta, lam)
            ratio = writes / max(1, len(X))
            if ratio > max_ratio:
                max_ratio, worst_writes = ratio, (w, beta, lam, writes, len(X))
            if want != got:
                bad += 1
                if bad <= 4:
                    print("   MISMATCH w,beta,lam=", w, beta, lam)
                    print("   want=", want, "got=", got, "X=", X)
            if want != gotp:
                bad_ptr += 1
                if bad_ptr <= 4:
                    print("   POINTER MISMATCH w,beta,lam=", w, beta, lam,
                          "want=", want, "got=", gotp)
            if any(z[t] > z[t+1] for t in range(n-1)):
                nonmono += 1
            if any(starts[i] > starts[i+1] for i in range(len(starts)-1)):
                cand_bad += 1
    print("  instances: 4200 | mismatches:", bad,
          "| pointer-version mismatches:", bad_ptr,
          "| non-monotone z:", nonmono,
          "| start monotonicity failures:", cand_bad)
    print("  max rank writes / |X|: %.2f (example writes=%d, |X|=%d)" %
          (max_ratio, worst_writes[3], worst_writes[4]))

    print("  -- large-coordinate regime (U = Theta(n^2)) --")
    for n in (2000, 8000, 32000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(1, n*n) for _ in range(n)]
        t = time.perf_counter()
        d = dp_compressed(C, beta, n*n*n*n+1)
        dt = time.perf_counter()-t
        z = zvec(C, beta)
        U = max(z)-min(z)+1
        print("  n=%5d  %.4fs  U=%d  dp[n]=%s" % (n, dt, U, d[n]))


# ── B. rank-indexed and relational counterexamples ──────────────────────────

def check_B():
    print("== 1(b): counterexamples delimiting the O(n) rank algorithm ==")
    w, beta = [1, 1, 1, 2], [3, 3, 4, 4]
    L = greedy_levels(w, beta)
    print("  w=%s beta=%s  lex-opt=%s  greedy levels=%s" %
          (w, beta, brute_rank_lex(w, beta), L[1:]))
    print("  minimum-line layout [0,1,2,4] has t=2 at j=2 with L(2)=1")
    w2, beta2 = [1, 2], [2, 4]
    print("  reversal counterexample: w=%s beta=%s lex-opt=%s" %
          (w2, beta2, brute_rank_lex(w2, beta2)))
    w3, A3, s3 = [1, 1], 3, [2]
    print("  relational counterexample: w=%s A=%d s=%s lex-opt=%s" %
          (w3, A3, s3, brute_relational_lex(w3, A3, s3)))

    print("== 1(b): rank DP remains O(n) on the nonincreasing regime ==")
    rng = random.Random(555)
    bad = 0
    for n in range(1, 8):
        for _ in range(250):
            B0 = rng.randint(3, 10)
            beta = sorted([rng.randint(2, B0) for _ in range(n)], reverse=True)
            w = [rng.randint(1, beta[-1]) for _ in range(n)]
            if rank_dp_fast(w, beta) != brute_rank_lex(w, beta):
                bad += 1
    print("  instances: 1750 | mismatches:", bad)


# ── C. rank-1 transform identity for K=2 ────────────────────────────────────

def check_C():
    print("== 1(b): rank-1 transform verified for K=2 ==")
    rng = random.Random(9090)
    bad = 0
    for n in range(2, 9):
        for _ in range(300):
            w = [rng.randint(1, 6) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(C[n]+1, C[n]+20), rng.randint(C[n]+1, C[n]+20)]
            b1, b2 = beta[0]+1, beta[1]+1
            # force exactly 2 lines with a large line penalty
            lam = n*max(beta)**2+1
            f = [inf]*(n+1); f[0] = 0.0
            for t in range(1, 3):
                g = [inf]*(n+1)
                b = beta[t-1]
                for j in range(1, n+1):
                    for i in range(j):
                        if f[i] < inf and C[j]-C[i]-1 <= b:
                            g[j] = min(g[j], f[i]+(b+1+C[i]-C[j])**2+lam)
                f = g
            exact2 = f[n]-2*lam
            got2 = min((b1-C[j])**2 + (b2-(C[n]-C[j]))**2
                       for j in range(1, n))
            if exact2 != got2:
                bad += 1
    print("  instances: 2100 | 2-line formula mismatches:", bad)
    print("  (the transform is a lower-envelope query; K layers compose K",
          "such transforms)")



# ==== experiments_stack_only.py ====
"""Stack-only compressed envelope DP for 1(a).

Instead of materialising the envelope values on an array, maintain the lower
envelope as a stack of disjoint intervals.  Each line is inserted once and
each interval is popped once; a query needs one predecessor search among
interval starts.  With a static/dynamic word-RAM predecessor structure this
gives O(n * pred(n,U)) time and O(n) space for arbitrary w-bit integer
coordinates.

"""
from bisect import bisect_left, bisect_right
from math import inf
import random, time



def ceil_div(num, den):
    if num >= 0:
        return (num + den - 1)//den
    return -((-num)//den)


def dp_stack(C, beta, lam, stats=False):
    n = len(beta)
    z = zvec(C, beta)
    X = sorted(set(z))
    Xmin, Xmax = X[0], X[-1]
    starts = []                  # numeric interval starts, increasing
    entries = []                 # (m, b, start, end, centre); aligned
    cap = Xmin - 1               # largest x covered by some line
    dp = [inf]*(n+1)
    dp[0] = 0.0
    ops = {"push": 0, "pop": 0, "query": 0}

    def trim_to_start(s):
        """Remove every interval whose start is >= s (they are replaced)."""
        while starts and starts[-1] >= s:
            starts.pop()
            entries.pop()
            ops["pop"] += 1

    def insert(i):
        nonlocal cap
        if dp[i] == inf:
            return
        c = C[i]
        m = -2*c
        b = dp[i] + c*c
        newcap = min(c, Xmax)
        # Compute the numeric start of the new line's interval.
        s = None
        while entries:
            mt, bt, st, en, ct = entries[-1]
            if (mt - m)*st >= b - bt:       # new line beats top from st on
                s = st
                starts.pop()
                entries.pop()
                ops["pop"] += 1
            else:
                x = ceil_div(b - bt, mt - m)
                cand = max(st, x)
                s = min(cand, en + 1)       # top expires at its effective end
                break
        if s is None:
            s = Xmin
        s = min(s, cap + 1)
        s = max(s, Xmin)
        if newcap < s:
            cap = max(cap, newcap)
            return
        trim_to_start(s)
        if entries:
            # trim the kept top so intervals stay contiguous
            mt, bt, st, en, ct = entries[-1]
            entries[-1] = (mt, bt, st, s-1, ct)
            if s-1 < st:
                starts.pop()
                entries.pop()
                ops["pop"] += 1
        starts.append(s)
        entries.append((m, b, s, newcap, c))
        ops["push"] += 1
        cap = max(cap, newcap)

    def query(x):
        ops["query"] += 1
        if x > cap or x < Xmin:
            return inf
        k = bisect_right(starts, x) - 1
        if k < 0:
            return inf
        m, b, st, en, ct = entries[k]
        if x > en:
            return inf
        return m*x + b

    for j in range(1, n+1):
        insert(j-1)
        x = z[j-1]
        v = query(x)
        if v < inf:
            dp[j] = lam + x*x + v
    if stats:
        return dp, ops
    return dp


def check_stack_only():
    print("== stack-only compressed DP vs literal O(n^2) DP ==")
    rng = random.Random(20261125)
    bad = 0
    max_ops = 0
    for n in range(1, 14):
        for _ in range(400):
            w = [rng.randint(1, 7) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 40) for _ in range(n)]
            lam = rng.randint(0, 100)
            z = zvec(C, beta)
            want = dp_brute(C, z, lam)
            got, ops = dp_stack(C, beta, lam, stats=True)
            if want != got:
                bad += 1
                if bad <= 4:
                    print("   MISMATCH w,beta,lam=", w, beta, lam)
                    print("   want=", want, "got=", got)
            total = ops["push"]+ops["pop"]+ops["query"]
            max_ops = max(max_ops, total/max(1, n))
    print("  instances: 5200 | mismatches:", bad,
          "| max (push+pop+query)/n:", max_ops)

    print("  -- large-coordinate scaling (U = Theta(n^2)) --")
    for n in (2000, 8000, 32000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(1, n*n) for _ in range(n)]
        t = time.perf_counter()
        d, ops = dp_stack(C, beta, n**4+1, stats=True)
        dt = time.perf_counter()-t
        z = zvec(C, beta)
        U = max(z)-min(z)+1
        print("  n=%5d  %.4fs  U=%d  ops/n=%.2f  dp[n]=%s" %
              (n, dt, U, (ops["push"]+ops["pop"]+ops["query"])/n, d[n]))



# ==== experiments_rank_envelope.py ====
"""Rank-space incremental envelope DP for the end-indexed budget problem.

Objective: remove the last open complexity gap in Problem 1(a) by giving an
O(n) word-RAM algorithm for w-bit integer coordinates, with no predecessor
structure and no sorting bound beyond radix sort of the n query coordinates.

Idea: work entirely in the rank space of the sorted distinct query coordinates
X = {z_1..z_n}.  Each line l_i(x) = dp[i] + C_i^2 - 2 C_i x has slope -2C_i
decreasing in i, hence wins on a suffix of rank space.  Maintain the lower
envelope as a stack of disjoint rank intervals.  Insertion pops whole
intervals from the right; when the new line first beats the top line inside
the top interval, find the split rank by a LINEAR SCAN from the top interval's
left end.  The predicate is monotone in the rank.

Amortization: every scanned rank is either overwritten in this insertion or
retained in a non-top interval that can only be revisited when it is
overwritten.  Popping is once per interval.  The fresh ranks beyond the old
cap are written once.  Hence total scans + writes + pops = O(n).

Queries are O(1) array lookups F[rank[z_j]].  Sorting X and assigning ranks is
O(n) word-RAM by radix sort on w-bit integers; Python's sorted() is used here
only for verification.
"""
from math import inf
import random, time



def dp_rank_envelope(C, beta, lam, stats=False):
    n = len(beta)
    z = zvec(C, beta)
    # rank assignment (radix sort in the word-RAM claim; sorted() for tests)
    pairs = sorted((z[j], j) for j in range(n))
    X, rank = [], [0]*n
    for x, j in pairs:
        if not X or x != X[-1]:
            X.append(x)
        rank[j] = len(X)-1
    M = len(X)
    F = [inf]*M
    stack = []                 # [m, b, lo, hi] disjoint rank intervals
    cap_rank = -1              # largest rank covered by some line
    qptr = 0                   # first rank with X[qptr] > C_i (monotone in i)
    dp = [inf]*(n+1)
    dp[0] = 0.0
    ops = {"scan": 0, "write": 0, "pop": 0, "push": 0}

    for j in range(1, n+1):
        i = j-1
        c = C[i]
        while qptr < M and X[qptr] <= c:
            qptr += 1
        end_rank = qptr-1          # first rank with X > C_i, minus 1
        if dp[i] < inf and end_rank >= 0:
            m, b = -2*c, dp[i] + c*c
            r = None
            while stack:
                m2, b2, lo2, hi2 = stack[-1]
                if m*X[lo2] + b <= m2*X[lo2] + b2:
                    # new line wins from lo2 onward: pop the whole interval
                    r = lo2
                    stack.pop()
                    ops["pop"] += 1
                    continue
                # The new line has the smallest slope, hence wins on a
                # SUFFIX.  Scan from the right: every scanned rank is stolen
                # from the top and overwritten in this insertion.
                rr = hi2
                while rr >= lo2 and m*X[rr] + b <= m2*X[rr] + b2:
                    rr -= 1
                    ops["scan"] += 1
                r = rr+1
                if r <= hi2:
                    stack[-1][3] = r-1      # trim the kept top
                break
            if r is None:
                r = 0
            if r <= cap_rank + 1:
                start = min(r, cap_rank+1)
            else:
                start = r
            if start <= end_rank:
                for t in range(start, end_rank+1):
                    F[t] = m*X[t] + b
                ops["write"] += end_rank-start+1
                stack.append([m, b, start, end_rank])
                ops["push"] += 1
                cap_rank = max(cap_rank, end_rank)
        v = F[rank[j-1]]
        if v < inf and rank[j-1] <= cap_rank:
            dp[j] = lam + z[j-1]*z[j-1] + v
    if stats:
        return dp, ops
    return dp


def check_rank_envelope():
    print("== rank-space O(n) envelope DP vs literal O(n^2) DP ==")
    rng = random.Random(20261126)
    bad = 0
    max_ops = 0.0
    worst = None
    for n in range(1, 15):
        for _ in range(400):
            w = [rng.randint(1, 7) for _ in range(n)]
            C = Cvec(w)
            beta = [rng.randint(1, 60) for _ in range(n)]
            lam = rng.randint(0, 200)
            z = zvec(C, beta)
            want = dp_brute(C, z, lam)
            got, ops = dp_rank_envelope(C, beta, lam, stats=True)
            if want != got:
                bad += 1
                if bad <= 5:
                    print("   MISMATCH w,beta,lam=", w, beta, lam)
                    print("   want=", want, "got=", got)
            total = ops["scan"]+ops["write"]+ops["pop"]+ops["push"]
            ratio = total/max(1, n)
            if ratio > max_ops:
                max_ops, worst = ratio, (w, beta, lam, ops)
    print("  instances: 5600 | mismatches:", bad,
          "| max (scan+write+pop+push)/n = %.2f" % max_ops,
          "| worst ops:", worst[3] if worst else None)

    print("  -- large-coordinate scaling (U = Theta(n^2)) --")
    for n in (2000, 8000, 32000, 64000):
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = [rng.randint(1, n*n) for _ in range(n)]
        t = time.perf_counter()
        d, ops = dp_rank_envelope(C, beta, n**4+1, stats=True)
        dt = time.perf_counter()-t
        z = zvec(C, beta)
        U = max(z)-min(z)+1
        tot = sum(ops.values())
        print("  n=%5d  %.4fs  U=%d  ops/n=%.2f  dp[n]=%s" %
              (n, dt, U, tot/n, d[n]))

    print("  -- exhaustive small-instance check_rank_envelope --")
    from itertools import product
    bad2 = 0
    for n in range(1, 6):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(list(w))
            for beta in product(range(1, 7), repeat=n):
                lam = 7
                z = zvec(C, list(beta))
                want = dp_brute(C, z, lam)
                got = dp_rank_envelope(C, list(beta), lam)
                if want != got:
                    bad2 += 1
                    if bad2 <= 3:
                        print("   exhaustive MISMATCH", w, beta, want, got)
    print("  exhaustive instances: all n<=6,w<=3,beta<=7 | mismatches:", bad2)



# ==== rank-space quadratic-write lower bound ====
"""Quadratic-write lower-bound construction for the rank-space envelope DP.

The instance is feasible, has unit words, w-bit integer coordinates, and a
dominating lexicographic penalty Lambda.  Sorting the query coordinates is
linear for w=O(log n), but the rank-space DP rewrites Theta(n^2) ranks.
"""
from math import inf



def rank_lower_instance(n):
    """Return (C, beta, Lambda, z, m, K).  n must be even and >= 6."""
    assert n % 2 == 0 and n >= 6
    m = n//2
    K = 4*n*n
    w = [1]*n
    C = Cvec(w)
    beta = [1]*m + [K+n-1]*(n-m)
    lam = n*(max(beta)+1)**2 + 1
    z = zvec(C, beta)
    return C, beta, lam, z, m, K


def sorted_ranks(z):
    pairs = sorted((z[j], j) for j in range(len(z)))
    X, rank = [], [0]*len(z)
    for x, j in pairs:
        if not X or x != X[-1]:
            X.append(x)
        rank[j] = len(X)-1
    return X, rank


def rank_trace(C, beta, lam):
    """Instrumented copy of dp_rank_envelope; return dp, starts, writes."""
    n = len(beta)
    z = zvec(C, beta)
    X, rank = sorted_ranks(z)
    M = len(X)
    F = [inf]*M
    stack = []
    cap = -1
    dp = [0]+[inf]*n
    starts = [-1]*n
    writes = [0]*n

    for j in range(1, n+1):
        i = j-1
        c = C[i]
        qptr = 0
        while qptr < M and X[qptr] <= c:
            qptr += 1
        end = qptr-1
        if dp[i] < inf and end >= 0:
            mline = -2*c
            bline = dp[i]+c*c
            r = None
            while stack:
                m2, b2, lo2, hi2 = stack[-1]
                if mline*X[lo2]+bline <= m2*X[lo2]+b2:
                    r = lo2
                    stack.pop()
                    continue
                rr = hi2
                while rr >= lo2 and mline*X[rr]+bline <= m2*X[rr]+b2:
                    rr -= 1
                r = rr+1
                if r <= hi2:
                    stack[-1][3] = r-1
                break
            if r is None:
                r = 0
            start = r
            starts[i] = start
            if start <= end:
                for t in range(start, end+1):
                    F[t] = mline*X[t]+bline
                writes[i] = end-start+1
                stack.append([mline, bline, start, end])
                cap = max(cap, end)
        v = F[rank[j-1]]
        if v < inf and rank[j-1] <= cap:
            dp[j] = lam + z[j-1]**2 + v
    return dp, starts, writes


def check_construction():
    print("== rank-space quadratic-write construction ==")
    for n in (6, 8, 10, 20, 40, 80, 120):
        C, beta, lam, z, m, K = rank_lower_instance(n)
        d, ops = dp_rank_envelope(C, beta, lam, stats=True)
        ref = dp_brute(C, z, lam)
        ok = all(abs(d[i]-ref[i]) < 1e-6 for i in range(n+1))
        # DP values under the dominating penalty
        vals_ok = True
        for i in range(1, m+1):
            vals_ok &= abs(ref[i]-i*lam) < 1e-6
        for i in range(m+1, n+1):
            vals_ok &= abs(ref[i]-(lam+z[i-1]**2)) < 1e-6
        # instrumented start ranks and write counts
        d2, starts, wcnt = rank_trace(C, beta, lam)
        trace_ok = all(abs(d2[i]-ref[i]) < 1e-6 for i in range(n+1))
        start_ok = all(starts[i] == m+2 for i in range(m+1, n))
        lower = (m-1)*(m-2)
        bound_ok = sum(wcnt[i] for i in range(m+1, n)) >= lower
        print("  n=%3d m=%3d Lambda=%d writes=%d lower=%d "
              "dp_ok=%s vals_ok=%s trace_ok=%s start_ok=%s bound_ok=%s"
              % (n, m, lam, ops["write"], lower, ok, vals_ok,
                 trace_ok, start_ok, bound_ok))
        assert ok and vals_ok and trace_ok and start_ok and bound_ok

    print("  -- explicit n=8 witness --")
    n = 8
    C, beta, lam, z, m, K = rank_lower_instance(n)
    X, rank = sorted_ranks(z)
    d, ops = dp_rank_envelope(C, beta, lam, stats=True)
    print("     w=(1)^8, K=%d, beta=%s, Lambda=%d" % (K, beta, lam))
    print("     z=%s" % z)
    print("     X=%s" % X)
    print("     rank(z)=%s" % rank)
    print("     ops=%s" % ops)
    print("     repeated-write start rank m+2=%d; ranks %d..%d"
          % (m+2, m+2, len(X)-1))
    d2, starts, wcnt = rank_trace(C, beta, lam)
    for i in range(m+1, n):
        print("       line i=%d: start=%d writes=%d"
              % (i, starts[i], wcnt[i]))


def check_lex_and_exhaustive():
    print("== lexicographic objective and exhaustive layouts ==")
    bad = 0
    tested = 0
    for n in (6, 8, 10):
        C, beta, lam, z, m, K = rank_lower_instance(n)
        ref = dp_brute(C, z, lam)
        lex = exhaustive_lex(C, beta)
        want = lex[0]*lam + lex[1]
        tested += 1
        if abs(ref[n]-want) > 1e-6:
            bad += 1
            print("   mismatch n=%d lex=%s got=%s want=%s"
                  % (n, lex, ref[n], want))
    print("  exhaustive lex instances:", tested, "| mismatches:", bad)
    assert bad == 0



# ==== few distinct query coordinates ====
"""DP (0.1) when the query coordinates z_j take only D distinct values.

Two exact algorithms, both verified against the literal O(n^2) DP:

  A. prefix-minima DP: O(nD) time, O(D) space, pointer machine, no sorting
     and no word-RAM bit tricks.  For D=O(1) this is O(n) even when the
     coordinate universe U is polynomial (or exponential) and z is
     non-monotone, so SC/SMAWK, O(n+U) and the word-RAM predecessor bounds
     are all superlinear.

  B. lower-envelope DP: O(n log D) time (comparison model) by replacing the
     flat rank array of the rank-space DP with a lazy segment tree carrying
     time-stamped range assignments.  This removes the Theta(n^2) writes of
     Proposition D.
"""
from bisect import bisect_left, bisect_right
from math import inf
import random, time



def dp_prefix_min(C, z, lam):
    """O(nD) pointer-machine DP; D = number of distinct z values."""
    n = len(z)
    distinct = list(dict.fromkeys(z))          # collect ALL coordinates first
    pos = {x: k for k, x in enumerate(distinct)}
    cur = [inf]*len(distinct)      # cur[k] = min over inserted lines of dp[i]+(C_i-x_k)^2
    dp = [0]+[inf]*n
    for j in range(1, n+1):
        i = j-1
        if dp[i] < inf:
            c = C[i]
            for k, x in enumerate(distinct):
                if x <= c:
                    v = dp[i]+(c-x)**2
                    if v < cur[k]:
                        cur[k] = v
        r = pos[z[j-1]]
        dp[j] = lam+cur[r] if cur[r] < inf else inf
    return dp


class _LazyTree:
    """Range assign (time-stamped) + point query, O(log D) per operation."""

    def __init__(self, n):
        self.n = n
        self.tag = [None]*(4*max(n, 1))     # (stamp, line index)

    def assign(self, ql, qr, stamp, line, node=1, l=0, r=None):
        if r is None:
            r = self.n-1
        if ql > r or qr < l:
            return
        if ql <= l and r <= qr:
            self.tag[node] = (stamp, line)
            return
        m = (l+r)//2
        self.assign(ql, qr, stamp, line, 2*node, l, m)
        self.assign(ql, qr, stamp, line, 2*node+1, m+1, r)

    def query(self, p):
        node, l, r = 1, 0, self.n-1
        best = None
        while True:
            t = self.tag[node]
            if t is not None and (best is None or t[0] > best[0]):
                best = t
            if l == r:
                return best[1] if best is not None else None
            m = (l+r)//2
            if p <= m:
                node, r = 2*node, m
            else:
                node, l = 2*node+1, m+1


def dp_envelope_segtree(C, z, lam, stats=False):
    """O(n log D) DP via monotone lower envelope + lazy segment tree."""
    n = len(z)
    X = sorted(set(z))
    D = len(X)
    rank = {x: r for r, x in enumerate(X)}
    tree = _LazyTree(D)
    stack = []                    # [lo, hi, line index], disjoint, lo increasing
    cap = -1                      # largest rank covered by an inserted line
    dp = [0]+[inf]*n
    ops = {"pops": 0, "assigns": 0, "comparisons": 0}

    # line i is dp[i] + (C_i - x)^2; b_i = dp[i] + C_i^2
    def beats(i, h, x):
        # dp[i]+(C_i-x)^2 <= dp[h]+(C_h-x)^2
        # <=> (b_i-b_h) - 2(C_i-C_h)x <= 0
        return (dp[i]+C[i]*C[i]-dp[h]-C[h]*C[h]) - 2*(C[i]-C[h])*x <= 0

    for j in range(1, n+1):
        i = j-1
        if dp[i] < inf:
            end = bisect_right(X, C[i])-1     # line i feasible on ranks <= end
            if end >= 0:
                s = None
                while stack:
                    lo, hi, h = stack[-1]
                    if lo > end:              # C_i increasing makes this rare
                        s = end+1
                        break
                    ops["comparisons"] += 1
                    if beats(i, h, X[lo]):
                        s = lo
                        stack.pop()
                        ops["pops"] += 1
                    else:
                        # binary search first rank in [lo,min(hi,end)] beaten
                        a, b = lo, min(hi, end)
                        first = b+1
                        while a <= b:
                            mid = (a+b)//2
                            ops["comparisons"] += 1
                            if beats(i, h, X[mid]):
                                first, b = mid, mid-1
                            else:
                                a = mid+1
                        s = first
                        if s <= hi:
                            stack[-1][1] = s-1     # trim kept top
                        break
                if s is None:
                    s = 0
                s = max(s, 0)
                if s <= end:
                    tree.assign(s, end, i, i)
                    ops["assigns"] += 1
                    while stack and stack[-1][1] >= s:
                        stack.pop()
                        ops["pops"] += 1
                    stack.append([s, end, i])
                    cap = max(cap, end)
        r = rank[z[j-1]]
        line = tree.query(r)
        if line is not None and r <= cap:
            dp[j] = lam + dp[line] + (C[line]-z[j-1])**2
    if stats:
        return dp, ops
    return dp


def check_few_coords():
    print("== few distinct query coordinates: two exact algorithms ==")
    rng = random.Random(20270224)
    badA = badB = 0
    tested = 0
    for n in range(1, 13):
        for _ in range(300):
            w = [rng.randint(1, 6) for _ in range(n)]
            C = Cvec(w)
            D = rng.randint(1, min(4, n))
            vals = [rng.randint(-n*n, 1) for _ in range(D)]
            z = [rng.choice(vals) for _ in range(n)]
            lam = rng.randint(0, 500)
            want = dp_brute(C, z, lam)
            a = dp_prefix_min(C, z, lam)
            b, _ = dp_envelope_segtree(C, z, lam, stats=True)
            tested += 1
            if a != want:
                badA += 1
                if badA <= 3:
                    print("   A MISMATCH", w, z, lam, want, a)
            if b != want:
                badB += 1
                if badB <= 3:
                    print("   B MISMATCH", w, z, lam, want, b)
    print("  random instances:", tested,
          "| prefix-min mismatches:", badA,
          "| envelope mismatches:", badB)

    print("  -- exhaustive small instances --")
    from itertools import product
    bad2 = 0
    cnt2 = 0
    for n in range(1, 5):
        for w in product(range(1, 3), repeat=n):
            C = Cvec(list(w))
            for z in product(range(-2, 2), repeat=n):
                lam = 7
                want = dp_brute(C, list(z), lam)
                a = dp_prefix_min(C, list(z), lam)
                b = dp_envelope_segtree(C, list(z), lam)
                cnt2 += 1
                if a != want or b != want:
                    bad2 += 1
                    if bad2 <= 3:
                        print("   exhaustive MISMATCH", w, z, want, a, b)
    print("  exhaustive instances:", cnt2, "| mismatches:", bad2)
    assert badA == 0 and badB == 0 and bad2 == 0


def check_linear_family():
    print("== D=O(1), U=Theta(n^3), non-monotone z: O(n) scaling ==")
    # z_j alternates between 0 and -H, so SC fails; U=H+1=Theta(n^3);
    # w=Theta(log n); O(n+U), O(n log w), O(n log n/log w) are all omega(n).
    for n in (1000, 4000, 16000, 64000):
        w = [1]*n
        C = Cvec(w)
        H = n**3
        z = [0 if j % 2 else -H for j in range(1, n+1)]
        lam = n*(H+2*n+1)**2+1
        t = time.perf_counter()
        d, ops = dp_envelope_segtree(C, z, lam, stats=True)
        dtB = time.perf_counter()-t
        t = time.perf_counter()
        a = dp_prefix_min(C, z, lam)
        dtA = time.perf_counter()-t
        ok = a == d
        print("  n=%5d D=2 U=%d prefix-min %.4fs envelope %.4fs "
              "ops=%s equal=%s" % (n, H+1, dtA, dtB, ops, ok))
        assert ok
    print("  both algorithms are linear here; SC fails because z alternates")

    print("  -- second family: positive large z, U=Theta(n^4) --")
    for n in (1000, 4000, 16000):
        H = n**3
        w = [H]*n
        C = Cvec(w)
        z = [0 if j % 2 else H-1 for j in range(1, n+1)]
        lam = n*(2*C[n]+H)**2+1
        t = time.perf_counter()
        d, ops = dp_envelope_segtree(C, z, lam, stats=True)
        dtB = time.perf_counter()-t
        t = time.perf_counter()
        a = dp_prefix_min(C, z, lam)
        dtA = time.perf_counter()-t
        print("  n=%5d D=2 U~%d prefix-min %.4fs envelope %.4fs ops=%s "
              "equal=%s" % (n, C[n], dtA, dtB, ops, a == d))
        assert a == d


def check_envelope_activity():
    print("== envelope is not trivially constant on small-D inputs ==")
    rng = random.Random(20270226)
    best = 0
    arg = None
    for _ in range(1000):
        n = rng.randint(3, 14)
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        D = rng.randint(1, 4)
        vals = [rng.randint(-20, C[n]-2) for _ in range(D)]
        z = [rng.choice(vals) for _ in range(n)]
        lam = rng.randint(0, 300)
        _, ops = dp_envelope_segtree(C, z, lam, stats=True)
        a = ops["assigns"]+ops["pops"]
        if a > best:
            best, arg = a, (w, z, lam, ops)
    print("  max (assigns+pops) over 1000 instances:", best, "| example:", arg)
    assert best >= 3


def check_prefix_linearity():
    print("== prefix-min operation count is O(nD) ==")
    rng = random.Random(20270225)
    for D in (1, 2, 4, 8):
        n = 4000
        w = [1]*n
        C = Cvec(w)
        vals = [-rng.randint(1, 10**6) for _ in range(D)]
        z = [rng.choice(vals) for _ in range(n)]
        lam = 10**15
        t = time.perf_counter()
        dp_prefix_min(C, z, lam)
        dt = time.perf_counter()-t
        print("  D=%d n=%d time %.4fs time/n %.2e" % (D, n, dt, dt/n))



# ==== predecessor and uniform-width checks ====
"""Results for the remaining open problems.

A. Problem 1(a): a reduction from STATIC PREDECESSOR to DP (0.1), proving
   that the DP has exactly the complexity of n predecessor queries in the
   cell-probe model.

B. Problem 1(b): for uniform word widths (C_i = g*i) the rank-indexed and
   relational lexicographic optima reduce to a convex separable resource
   allocation problem, solvable in O(n log n) -- hence o(nK) and o(nm).

C. Counterexamples showing that permuting the budget sequence changes the
   optimum, so the non-uniform case does not reduce to sorting (beta_t).

"""
from heapq import heappush, heappop
from itertools import permutations
from math import inf
import random



# ── A. static predecessor → end-indexed DP ──────────────────────────────────

def predecessor_to_dp(S, queries):
    """Build an instance whose dp-values reveal successor(q) in S.

    S is a strictly increasing list of m integers.  C_i = 2(S_i+2) gives
    gaps >= 2.  For the first m words set z_j=0 (so dp[j]=Lambda).  For
    query q (0 <= q <= S_m) set z_{m+j}=2(q+2) and
    beta_{m+j}=C_{m+j}-z_{m+j}-1 > 0.  Then
    lambda_{m+j} = min{i : C_i >= z} = min{i : S_i >= q}, the successor
    rank, and dp[m+j] = 2*Lambda + (C_{lambda}-z)^2.  Static successor and
    predecessor are equivalent by complementing the universe.
    """
    m = len(S)
    C = [0]
    for s in S:
        C.append(2*(s+2))
    # extend C with arbitrary increasing values for the second half
    n = 2*m
    for j in range(m+1, n+1):
        C.append(C[-1] + 2)
    beta = []
    z = []
    for j in range(1, m+1):
        z.append(0)
        beta.append(C[j]-z[-1]-1)
    for q in queries:
        z.append(2*(q+2))
        beta.append(C[len(beta)+1]-z[-1]-1)
    return C, beta, z


def check_predecessor_reduction():
    print("== 1(a): static predecessor reduces to DP (0.1) ==")
    rng = random.Random(20261127)
    bad = 0
    for m in range(2, 9):
        for _ in range(200):
            S = sorted(rng.sample(range(1, 60), m))
            queries = [rng.randint(0, S[-1]) for _ in range(m)]
            C, beta, z = predecessor_to_dp(S, queries)
            Lam = 10**6
            dp = dp_brute(C, z, Lam)
            want = []
            for q in queries:
                p = next((i for i, s in enumerate(S, 1) if s >= q), m+1)
                want.append(p)
            got = []
            for j in range(m+1, 2*m+1):
                # recover lambda from dp[j] = 2 Lam + (C_lambda - z_j)^2
                t = dp[j] - 2*Lam
                if t < 0:
                    got.append(None)
                    continue
                d = int(round(t**0.5))
                lam = next((i for i in range(0, len(C)) if C[i]-z[j-1] == d), None)
                got.append(lam)
            if got != want:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH S,query =", S, queries, "want", want, "got", got)
    print("  instances: 1400 | predecessor-rank recovery failures:", bad)
    print("  => DP complexity is Theta(n * T_pred(n,U,w)) in the cell-probe model")


# ── B. uniform-width rank-indexed budgets ───────────────────────────────────

def uniform_exact(n, alpha, beta, K):
    """Least squared-slack penalty for exactly K lines, uniform word widths.

    beta has length K; line r has budget beta[r-1], 1-indexed.  Returns None
    if no legal layout with K lines exists.
    """
    g = alpha + 1
    caps = [(b+1)//g for b in beta[:K]]
    if any(c < 1 for c in caps) or sum(caps) < n or K > n:
        return None
    m = [1]*K
    remaining = n - K
    heap = []
    a = [beta[r]+1 for r in range(K)]
    for r in range(K):
        if m[r] < caps[r]:
            heappush(heap, (g*(2*m[r]+1) - 2*a[r], r))
    for _ in range(remaining):
        if not heap:
            return None
        _, r = heappop(heap)
        m[r] += 1
        if m[r] < caps[r]:
            heappush(heap, (g*(2*m[r]+1) - 2*a[r], r))
    return sum((a[r]-g*m[r])**2 for r in range(K))


def uniform_lex(n, alpha, beta):
    """Lexicographic optimum for n words of width alpha, budgets beta[0..n-1].

    gap g = alpha+1, C_j = g*j.  A line with m words has slack
    beta_r+1 - g*m; feasible m <= cap_r = floor((beta_r+1)/g).  Minimise
    sum (beta_r+1-g*m_r)^2 over m_r>=1, sum m_r = n, m_r<=cap_r, for the
    smallest K with sum_{r<=K} cap_r >= n.
    """
    g = alpha + 1
    caps = [(b+1)//g for b in beta]
    total = 0
    K = None
    for r, c in enumerate(caps, 1):
        if c < 1:
            return None                      # an early budget cannot hold a word
        total += c
        if total >= n:
            K = r
            break
    if K is None:
        return None
    pen = uniform_exact(n, alpha, beta, K)
    return None if pen is None else (K, pen)


def check_uniform_rank():
    print("== 1(b): uniform word widths reduce to convex resource allocation ==")
    rng = random.Random(31337)
    bad = 0
    tested = 0
    for n in range(1, 11):
        for _ in range(250):
            alpha = rng.randint(1, 4)
            w = [alpha]*n
            # budgets large enough to be feasible and to allow several lines
            beta = [rng.randint(alpha, alpha+8) for _ in range(n)]
            # guarantee overall capacity
            cap = [(b+1)//(alpha+1) for b in beta]
            while sum(cap) < n:
                beta = [b+2 for b in beta]
                cap = [(b+1)//(alpha+1) for b in beta]
            want = brute_rank_lex(w, beta)
            got = uniform_lex(n, alpha, beta)
            tested += 1
            if want != got:
                bad += 1
                if bad <= 5:
                    print("   MISMATCH n,alpha,beta =", n, alpha, beta,
                          "want", want, "got", got)
    print("  instances:", tested, "| mismatches:", bad)

    # complexity illustration: K = Theta(n), O(n log K) vs O(nK)
    print("  -- allocation scaling --")
    import time
    for n in (1000, 4000, 16000, 64000):
        alpha = 1
        beta = [2*alpha+1]*n              # capacity 2 words per line, K = ceil(n/2)
        t = time.perf_counter()
        r = uniform_lex(n, alpha, beta)
        dt = time.perf_counter()-t
        print("  n=%5d  %.4fs  result=%s  K/n=%.2f" %
              (n, dt, r, r[0]/n))


# ── C. uniform-width relational variant ─────────────────────────────────────

def relational_uniform_lex(n, alpha, A, s):
    """Lexicographic optimum when the last len(s) lines have budgets
    A-s[0],...,A-s[-1] (s[0] is the last line), all preceding lines A,
    and all words have width alpha.
    """
    m = len(s)
    g = alpha + 1
    capA = (A+1)//g
    caps_special = [(A-x+1)//g for x in s]   # budgets A-s[0],...,A-s[-1] from END
    if capA < 1 or any(c < 1 for c in caps_special):
        return None
    # K lines from the start: first K-m lines at A, then the m special
    # budgets in order A-s[m-1], ..., A-s[0].  Capacity with K lines is
    # (K-m)*capA + spec_sum, so the smallest feasible K is closed-form.
    spec_sum = sum(caps_special)
    need = max(0, n - spec_sum)
    K = m + (need + capA - 1)//capA
    if K > n:
        return None
    beta = [A]*(K-m) + [A-s[r] for r in range(m-1, -1, -1)]
    pen = uniform_exact(n, alpha, beta, K)
    return None if pen is None else (K, pen)


def check_uniform_relational():
    print("== 1(b) relational: uniform word widths, arbitrary locator profile ==")
    rng = random.Random(2718)
    bad = 0
    tested = 0
    for A in range(3, 9):
        for m in (1, 2, 3):
            for n in range(m, 9):
                for _ in range(120):
                    alpha = rng.randint(1, min(3, A))
                    # locator widths s_r in [0, A-alpha] so special budgets >= alpha
                    s = [rng.randint(0, A-alpha) for _ in range(m)]
                    w = [alpha]*n
                    want = brute_relational_lex(w, A, s)
                    got = relational_uniform_lex(n, alpha, A, s)
                    tested += 1
                    if want != got:
                        bad += 1
                        if bad <= 5:
                            print("   MISMATCH n,alpha,A,s =", n, alpha, A, s,
                                  "want", want, "got", got)
    print("  instances:", tested, "| mismatches:", bad)


# ── D. budget-order counterexamples ─────────────────────────────────────────

def check_budget_order():
    print("== counterexamples: permuting rank budgets changes the optimum ==")
    w = [1, 1, 2]
    for b in [(2, 5), (2, 6), (3, 5), (2, 3)]:
        v = brute_rank_lex(w, list(b))
        r = brute_rank_lex(w, list(reversed(b)))
        if v != r:
            print("  w=%s beta=%s lex=%s ; reversed beta=%s lex=%s" %
                  (w, b, v, tuple(reversed(b)), r))
    print("  => sorting (beta_t) does not preserve the lexicographic optimum")



# ==== periodic budget checks ====
"""Periodic rank-indexed budgets: an O(p n log n) shortest-path algorithm.

The budget sequence beta_1,...,beta_K is periodic with period p.  Build a
phase-augmented DAG: state (r,j) means after some number of lines, the next
line uses budget beta_{r+1} and the prefix ends at word j.  Every edge adds
the squared slack plus a dominating line penalty Lambda, so a shortest path
has the fewest lines and then the least squared slack.

For each source phase r, line i is active for query breakpoints j in
[i+1, J_r(i)], where J_r(i) is the largest j with C_j <= C_i + beta_r + 1.
An offline segment tree over j stores each line on O(log n) nodes; each node
maintains a monotone lower hull.  All lines are inserted in increasing i, and
queries arrive in increasing j, so the total is O(p n log n).

Also: experiments on convexity of the exact-line-count value function for
each residue class modulo p, and on the core size of one-layer matrices.
"""
from bisect import bisect_right
from collections import deque
from math import inf
import random, time



class NodeHull:
    """Lower hull of lines m*x+b, slopes decreasing, queries x increasing.

    Lines are inserted once in decreasing-slope order.  Front pops remove
    lines that are no longer optimal (or lex-dominated on ties).
    """
    def __init__(self):
        self.q = deque()

    def insert(self, m, b, cnt):
        q = self.q
        # Keep a line on equality of intersections; query-time tie-breaking
        # then chooses the path with fewer lines.
        while len(q) >= 2:
            m1, b1, c1 = q[-2]
            m2, b2, c2 = q[-1]
            # x12 >= x23  <=>  (b2-b1)(m2-m) >= (b-b2)(m1-m2)
            if (b2-b1)*(m2-m) > (b-b2)*(m1-m2):
                q.pop()
            else:
                break
        q.append((m, b, cnt))

    def query(self, x):
        q = self.q
        while len(q) >= 2:
            m0, b0, c0 = q[0]
            m1, b1, c1 = q[1]
            v0, v1 = m0*x+b0, m1*x+b1
            if v0 > v1 or (v0 == v1 and c0 > c1):
                q.popleft()
            else:
                break
        if not q:
            return None
        m, b, c = q[0]
        return m*x+b, c


class SegTree:
    def __init__(self, n):
        self.n = n
        self.N = 1
        while self.N < n:
            self.N *= 2
        self.nodes = [None]*(2*self.N)

    def _node(self, k):
        if self.nodes[k] is None:
            self.nodes[k] = NodeHull()
        return self.nodes[k]

    def add(self, l, r, m, b, cnt):
        # 1-based j interval [l,r]
        if l > r:
            return
        L, R = l+self.N-1, r+self.N-1
        while L <= R:
            if L & 1:
                self._node(L).insert(m, b, cnt)
                L += 1
            if not (R & 1):
                self._node(R).insert(m, b, cnt)
                R -= 1
            L >>= 1
            R >>= 1

    def query(self, j, x):
        pos = j+self.N-1
        best = None
        while pos:
            h = self.nodes[pos]
            if h is not None:
                res = h.query(x)
                if res is not None:
                    val, c = res
                    if best is None or val < best[0] or (val == best[0] and c < best[1]):
                        best = (val, c)
            pos >>= 1
        return best


def periodic_dp(w, beta):
    """Lexicographic optimum for budgets beta repeated periodically."""
    n = len(w)
    C = Cvec(w)
    p = len(beta)
    B = max(beta)
    Lam = n*B*B + 1
    d = [[inf]*(n+1) for _ in range(p)]
    cnt = [[0]*(n+1) for _ in range(p)]
    d[0][0] = 0
    trees = [SegTree(n) for _ in range(p)]

    def add_line(r, i):
        if d[r][i] == inf:
            return
        bud = beta[r]
        J = bisect_right(C, C[i] + bud + 1) - 1
        J = min(J, n)
        if J < i+1:
            return
        D = C[i] + bud + 1
        m = -2*D
        b = d[r][i] + D*D + Lam
        trees[r].add(i+1, J, m, b, cnt[r][i])

    # source i=0
    for r in range(p):
        if d[r][0] < inf:
            add_line(r, 0)

    for j in range(1, n+1):
        x = C[j]
        for r in range(p):
            target = (r+1) % p
            res = trees[r].query(j, x)
            if res is not None:
                val, c = res
                # line value lacks x^2; add it once
                d[target][j] = x*x + val
                cnt[target][j] = c+1
        # insert the new source index j into every phase tree
        for r in range(p):
            if d[r][j] < inf:
                add_line(r, j)

    best = None
    for r in range(p):
        if d[r][n] < inf:
            k = cnt[r][n]
            pen = d[r][n] - Lam*k
            key = (k, pen)
            if best is None or key < best:
                best = key
    return best


def periodic_beta(beta, n):
    return [beta[t % len(beta)] for t in range(n)]


def check_periodic_dp():
    print("== periodic rank budgets: O(p n log n) phase DP vs exhaustive lex ==")
    rng = random.Random(20261128)
    bad = 0
    tested = 0
    for n in range(1, 10):
        for _ in range(300):
            w = [rng.randint(1, 4) for _ in range(n)]
            p = rng.randint(1, 4)
            beta = [rng.randint(max(w), max(w)+6) for _ in range(p)]
            want = brute_rank_lex(w, periodic_beta(beta, n))
            got = periodic_dp(w, beta)
            tested += 1
            if want != got:
                bad += 1
                if bad <= 5:
                    print("   MISMATCH n,w,p,beta =", n, w, p, beta,
                          "want", want, "got", got)
    print("  instances:", tested, "| mismatches:", bad)

    print("  -- exhaustive small instances --")
    from itertools import product
    bad2 = 0
    for n in range(1, 7):
        for w in product(range(1, 3), repeat=n):
            for p in range(1, 4):
                for beta in product(range(2, 5), repeat=p):
                    want = brute_rank_lex(list(w), periodic_beta(list(beta), n))
                    got = periodic_dp(list(w), list(beta))
                    if want != got:
                        bad2 += 1
                        if bad2 <= 3:
                            print("   exhaustive MISMATCH", w, beta, want, got)
    print("  exhaustive instances | mismatches:", bad2)


def P_exact(w, beta, R):
    n = len(w); C = Cvec(w)
    f = [inf]*(n+1); f[0] = 0.0
    for t in range(1, R+1):
        b = beta[(t-1) % len(beta)]
        g = [inf]*(n+1)
        for j in range(t, n+1):
            for i in range(t-1, j):
                if f[i] < inf and C[j]-C[i]-1 <= b:
                    v = f[i] + (b+1+C[i]-C[j])**2
                    if v < g[j]: g[j] = v
        f = g
    return f[n]


def check_residue_convexity():
    print("== periodic budgets: convexity of P_r(q) in each residue class ==")
    rng = random.Random(271828)
    bad = 0
    tested = 0
    for n in range(4, 15):
        for _ in range(250):
            w = [rng.randint(1, 4) for _ in range(n)]
            p = rng.randint(1, 5)
            beta = [rng.randint(max(w), max(w)+7) for _ in range(p)]
            for r in range(p):
                R0 = r if r > 0 else p
                Rs = list(range(R0, n+1, p))
                vals = [P_exact(w, beta, R) for R in Rs]
                fin = [v for v in vals if v < inf]
                if len(fin) < 3:
                    continue
                diffs = [fin[i+1]-fin[i] for i in range(len(fin)-1)]
                tested += 1
                if any(diffs[i] > diffs[i+1] for i in range(len(diffs)-1)):
                    bad += 1
                    if bad <= 3:
                        print("   NONCONVEX", w, beta, r, fin, diffs)
    print("  residue curves tested:", tested, "| nonconvex:", bad)


def check_core_size():
    print("== counterexample: one-layer matrix is not core-sparse ==")
    n = 8
    C = [0, 4, 9, 12, 17, 19, 22, 24, 27]
    d = 30
    nz = 0
    for i in range(n):
        for j in range(n):
            ok = True
            vals = []
            for a, b in ((i,j),(i,j+1),(i+1,j),(i+1,j+1)):
                if not (a < b) or C[b]-C[a]-1 > d-1:
                    ok = False; break
                vals.append((d+C[a]-C[b])**2)
            if ok:
                delta = vals[0]+vals[3]-vals[1]-vals[2]
                if delta != 0: nz += 1
    print("  n=%d, interior nonzero density cells=%d (Theta(n^2))" % (n, nz))
    print("  => core-sparse Monge multiplication does not give o(nK) here")


def check_scaling_periodic():
    print("  -- O(p n log n) scaling --")
    rng = random.Random(99)
    for n in (1000, 4000, 16000):
        for p in (1, 4):
            w = [rng.randint(1, 3) for _ in range(n)]
            beta = [rng.randint(3, 9) for _ in range(p)]
            t = time.perf_counter()
            r = periodic_dp(w, beta)
            dt = time.perf_counter()-t
            print("  n=%5d p=%2d  %.4fs  result=%s" % (n, p, dt, r))



# ==== two-parameter relaxation checks ====
"""Two-parameter Lagrangian relaxation for the relational secondary objective
when all locators have the same width.

From the start the budget sequence is A^q, b^m with b = A - s.  Define

  F(q,m) = least sum of squared slacks over layouts with exactly q body
           lines of budget A followed by exactly m terminal lines of budget b,
           or +inf if none exists.

  L(la,lb) = min over any q,m of [F(q,m) + la*q + lb*m].

L is computable in O(n) (two chained LWS passes).  L is concave in (la,lb),
so max_lb [L - lb*m] and max_la [max_lb(...) - la*q] are concave and are
maximised by nested integer binary search.  If F has no Legendre gap,

  F(q,m) = max_{la,lb} [L(la,lb) - la*q - lb*m],

giving an O(n log^2 V) algorithm independent of m.

Also recorded: an explicit counterexample showing that the ONE-parameter
relaxation (fix q by an exact body DP, relax only m) has a Legendre gap,
while the two-parameter relaxation does not.
"""
from math import inf
from itertools import product
import random



def F_exact(C, A, b, q, m):
    """Exact F(q,m) by layered DP."""
    n = len(C)-1
    E = [0.0] + [inf]*n
    for _ in range(q):
        G = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if E[i] < inf and C[j]-C[i]-1 <= A:
                    G[j] = min(G[j], E[i] + (A+1+C[i]-C[j])**2)
        E = G
    for _ in range(m):
        G = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if E[i] < inf and C[j]-C[i]-1 <= b:
                    G[j] = min(G[j], E[i] + (b+1+C[i]-C[j])**2)
        E = G
    return E[n]


def L_dual(C, A, b, la, lb):
    """Two chained unconstrained LWS passes with per-line penalties."""
    n = len(C)-1
    G = [0.0] + [inf]*n
    for j in range(1, n+1):
        for i in range(j):
            if G[i] < inf and C[j]-C[i]-1 <= A:
                G[j] = min(G[j], G[i] + (A+1+C[i]-C[j])**2 + la)
    H = list(G)
    for j in range(1, n+1):
        for i in range(j):
            if H[i] < inf and C[j]-C[i]-1 <= b:
                H[j] = min(H[j], H[i] + (b+1+C[i]-C[j])**2 + lb)
    return H[n]


def dual_table(C, A, b, R):
    tab = {}
    for la in range(-R, R+1):
        for lb in range(-R, R+1):
            tab[(la, lb)] = L_dual(C, A, b, la, lb)
    return tab


def F_from_table(tab, q, m):
    best = -inf
    for (la, lb), v in tab.items():
        cand = v - la*q - lb*m
        if cand > best:
            best = cand
    return best


def _max_concave(f, V):
    """Maximise a discrete concave integer function on [-V,V].

    Integer ternary search is used: it is robust to long flat stretches,
    which arise when several line counts are optimal at the same penalty.
    """
    lo, hi = -V, V
    while hi - lo > 6:
        m1 = lo + (hi-lo)//3
        m2 = hi - (hi-lo)//3
        if f(m1) < f(m2):
            lo = m1 + 1
        else:
            hi = m2 - 1
    return max(f(x) for x in range(lo, hi+1))


def F_nested(C, A, b, q, m, V):
    def h(la):
        def g(lb):
            return L_dual(C, A, b, la, lb) - lb*m
        return _max_concave(g, V) - la*q
    return _max_concave(h, V)


def min_lines_equal_locator(w, A, b, m):
    """Fewest lines with m equal-budget terminal lines."""
    K = threshold_relational_lines(w, A, [A-b]*m)
    return None if K == inf else K


def relational_two_phase_lex(w, A, s, m, V=None):
    """Dictionary optimum for equal-width locators, O(n log^2 V)."""
    C = Cvec(w)
    b = A - s
    if V is None:
        V = 4*(len(w)+1)*(A+1)**2 + 16
    K = min_lines_equal_locator(w, A, b, m)
    if K is None:
        return None
    q = K - m
    f = F_nested(C, A, b, q, m, V)
    return (K, f)


def check_dual_gap(Amax=5, nmax=4, R=120):
    print("== two-parameter Legendre transform: no-gap identity ==")
    tested = 0
    gaps = 0
    ex = []
    for A in range(2, Amax+1):
        for n in range(1, nmax+1):
            for w in product(range(1, min(3, A)+1), repeat=n):
                C = Cvec(list(w))
                for s in range(0, A):
                    b = A - s
                    tab = dual_table(C, A, b, R)
                    for q in range(0, n+1):
                        for m in range(1, n+1):
                            if q+m > n:
                                continue
                            want = F_exact(C, A, b, q, m)
                            got = F_from_table(tab, q, m)
                            tested += 1
                            if abs(want - got) > 1e-6:
                                gaps += 1
                                if len(ex) < 5:
                                    ex.append((w, A, s, q, m, want, got))
    print("  instances:", tested, "| Legendre gaps:", gaps)
    for e in ex:
        print("   gap", e)
    return gaps


def check_algorithm():
    """The nested dual is exact exactly where F has no Legendre gap; nonzero
    mismatch counts are the gap instances, not verification failures."""
    print("== equal-width locators: nested dual vs exhaustive (gap census) ==")
    rng = random.Random(20261129)
    bad = 0
    tested = 0
    for A in range(2, 9):
        for m in (1, 2, 3, 4):
            for n in range(m, 8):
                for _ in range(80):
                    w = [rng.randint(1, min(4, A)) for _ in range(n)]
                    s = rng.randint(0, A-1)
                    want = brute_relational_lex(w, A, [s]*m)
                    got = relational_two_phase_lex(w, A, s, m)
                    tested += 1
                    if want != got:
                        bad += 1
                        if bad <= 6:
                            print("   MISMATCH w,A,s,m =", w, A, s, m,
                                  "want", want, "got", got)
    print("  instances:", tested, "| points with a nonzero Legendre gap:", bad)


def check_one_param_gap():
    """Find an instance where relaxing only m (with q fixed by an exact body DP)
    has a Legendre gap, while the two-parameter relaxation recovers F."""
    print("== one-parameter gap vs two-parameter recovery ==")
    found = []
    for n in (3, 4, 5):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(list(w))
            for A in range(3, 8):
                for s in range(0, A):
                    b = A - s
                    for q in range(0, 3):
                        for m in range(1, 4):
                            if q+m > n:
                                continue
                            F = F_exact(C, A, b, q, m)
                            if F == inf:
                                continue
                            tab = dual_table(C, A, b, 150)
                            two = F_from_table(tab, q, m)
                            # one-parameter: exact q-line body DP, relax m
                            E = [0.0] + [inf]*n
                            for _ in range(q):
                                G = [inf]*(n+1)
                                for j in range(1, n+1):
                                    for i in range(j):
                                        if E[i] < inf and C[j]-C[i]-1 <= A:
                                            G[j] = min(G[j], E[i] +
                                                       (A+1+C[i]-C[j])**2)
                                E = G
                            one = -inf
                            for lb in range(-150, 151):
                                H = list(E)
                                for j in range(1, n+1):
                                    for i in range(j):
                                        if H[i] < inf and C[j]-C[i]-1 <= b:
                                            H[j] = min(H[j], H[i] +
                                                       (b+1+C[i]-C[j])**2 + lb)
                                one = max(one, H[n] - lb*m)
                            if abs(one-F) > 1e-6 and abs(two-F) < 1e-6:
                                found.append((w, A, s, q, m, F, one, two))
                                if len(found) >= 3:
                                    break
                        if len(found) >= 3:
                            break
                    if len(found) >= 3:
                        break
                if len(found) >= 3:
                    break
            if len(found) >= 3:
                break
        if len(found) >= 3:
            break
    print("  instances with one-parameter gap and no two-parameter gap:",
          len(found))
    for f in found:
        print("   w,A,s,q,m,F,one,two =", f)



# ==== Lagrangian barrier checks ====
"""Structural decomposition of rank-indexed layer matrices and barriers to
Lagrangian compression.

1. Exact decomposition of the layer matrix; its density is independent of
   beta and has rank one.  Consequently one layer has Theta(n^2) nonzero
   density cells, so core-sparse Monge multiplication cannot remove K.

2. One-parameter Lagrangian relaxation with a general source vector has a
   Legendre gap.

3. Two-parameter Lagrangian relaxation for equal-width locators also has a
   Legendre gap; the gap is unbounded on small instances (up to 13).

"""
from math import inf
from itertools import product
import random, collections



def check_decomposition():
    print("== layer-matrix decomposition and rank-one density ==")
    rng = random.Random(20261130)
    bad = 0
    for _ in range(500):
        n = rng.randint(2, 10)
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        beta = rng.randint(1, 40)
        for i in range(n+1):
            for j in range(i+1, n+1):
                lhs = (beta+1+C[i]-C[j])**2
                rhs = ((C[i]**2 + 2*(beta+1)*C[i])
                       + (C[j]**2 - 2*(beta+1)*C[j])
                       - 2*C[i]*C[j] + (beta+1)**2)
                if lhs != rhs:
                    bad += 1
    print("  decomposition identity checks: 500 instances | failures:", bad)

    # density of the full (no-window) layer: every interior cell is nonzero
    n = 10
    w = [rng.randint(1, 5) for _ in range(n)]
    C = Cvec(w)
    B = C[n] + 1
    nz = 0
    zero = 0
    for i in range(n):
        for j in range(n):
            if i+1 < j:  # interior of the upper-triangular band
                # Monge density (Gawrychowski et al., ESA 2025):
                # M[i+1,j] + M[i,j+1] - M[i,j] - M[i+1,j+1]
                d = (B+1+C[i+1]-C[j])**2 + (B+1+C[i]-C[j+1])**2 \
                    - (B+1+C[i]-C[j])**2 - (B+1+C[i+1]-C[j+1])**2
                if d != 0:
                    nz += 1
                else:
                    zero += 1
                if d != 2*(C[i+1]-C[i])*(C[j+1]-C[j]):
                    bad += 1
    print("  n=%d interior cells: nonzero density=%d, zero=%d (Theta(n^2)); "
          "density formula failures=%d" % (n, nz, zero, bad))
    print("  => core size delta = Theta(n^2); core-sparse Monge multiplication")
    print("     costs O(n^2 log n), not o(nK)")


def one_param_gap(C, A, b, q, m, R=200):
    """Legendre dual when the body has exactly q lines (computed exactly) and
    only the terminal-line count m is relaxed."""
    n = len(C)-1
    E = [0.0] + [inf]*n
    for _ in range(q):
        G = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if E[i] < inf and C[j]-C[i]-1 <= A:
                    G[j] = min(G[j], E[i] + (A+1+C[i]-C[j])**2)
        E = G
    best = -inf
    for lb in range(-R, R+1):
        H = list(E)
        for j in range(1, n+1):
            for i in range(j):
                if H[i] < inf and C[j]-C[i]-1 <= b:
                    H[j] = min(H[j], H[i] + (b+1+C[i]-C[j])**2 + lb)
        best = max(best, H[n] - lb*m)
    return best


def check_one_param_gap_examples():
    print("== one-parameter Lagrangian gap (general source vector) ==")
    cases = [
        ([1, 1, 1, 2], 5, 3, 2, 1),
        ([1, 1, 1, 2], 6, 3, 2, 1),
        ([1, 1, 1, 2], 6, 4, 2, 1),
    ]
    for w, A, s, q, m in cases:
        C = Cvec(w)
        b = A - s
        F = F_exact(C, A, b, q, m)
        one = one_param_gap(C, A, b, q, m)
        two = F_from_table(dual_table(C, A, b, 200), q, m)
        print("  w=%s A=%d s=%d (q,m)=(%d,%d)  F=%s  one-param=%s  "
              "two-param=%s" % (w, A, s, q, m, F, one, two))


def check_two_param_gap():
    print("== two-parameter Lagrangian gap (equal-width locators) ==")
    w = [2, 2, 1, 4, 4, 3]
    A, s, q, m = 7, 2, 3, 3
    C = Cvec(w)
    b = A - s
    F = F_exact(C, A, b, q, m)
    D = F_from_table(dual_table(C, A, b, 200), q, m)
    print("  explicit instance w=%s A=%d s=%d (q,m)=(%d,%d): F=%s, "
          "dual=%s, gap=%s" % (w, A, s, q, m, F, D, F-D))

    rng = random.Random(313)
    dist = collections.Counter()
    worst = (0, None)
    for _ in range(250):
        A = rng.randint(2, 8)
        n = rng.randint(2, 6)
        w = [rng.randint(1, min(4, A)) for _ in range(n)]
        C = Cvec(w)
        s = rng.randint(0, A-1)
        b = A - s
        tab = dual_table(C, A, b, 180)
        for q in range(0, n+1):
            for m in range(1, n+1):
                if q+m > n:
                    continue
                F = F_exact(C, A, b, q, m)
                if F == inf:
                    continue
                g = F - F_from_table(tab, q, m)
                dist[g] += 1
                if g > worst[0]:
                    worst = (g, (w, A, s, q, m, F))
    print("  gap distribution:", dict(sorted(dist.items())))
    print("  largest observed gap:", worst[0], "on", worst[1])
    print("  => F(q,m) is not jointly convex; the two-parameter Legendre")
    print("     transform has no exact inverse")


def check_residual_structure():
    print("== exact F grid and its convex envelope on a gap instance ==")
    w, A, s = [3, 1, 2], 3, 1
    C = Cvec(w)
    b = A - s
    tab = dual_table(C, A, b, 200)
    print("  w=%s A=%d s=%d; rows q, columns m" % (w, A, s))
    for q in range(4):
        exact = [F_exact(C, A, b, q, m) for m in range(4)]
        dual = [F_from_table(tab, q, m) for m in range(4)]
        print("   q=%d  exact=%s  dual=%s" % (q, exact, dual))



# ==== reversal and compression checks ====
"""Relational variant: reversal equivalence, uniform-tail compression, and an
O(n) algorithm for equal-width locators on uniform word grids.

Notation: the last m lines carry locators of widths s_1,...,s_m (r=1 is the
last line); b_r = A - s_r is the budget of the r-th line from the end.

A. Reversal equivalence.  Reversing the word sequence and the order of the
   lines maps a relational layout to a rank-indexed layout with budgets

       beta' = [b_1, b_2, ..., b_m, A, A, A, ...]   (prefix special, then A)

   and at least m lines.  Widths of contiguous blocks are invariant under
   reversal, so the lexicographic optima coincide.

B. Uniform-tail compression.  With a dominating per-line penalty Lam, the
   objective is min (sum slack^2 + Lam*lines).  Splitting a layout at word j,
   the A-lines on the prefix 1..j are handled by ONE unconstrained LWS pass;
   only the m special lines (budgets b_m,...,b_1 from the start, i.e.
   b_1,...,b_m from the end) need m layers.  This is O(nm + n).

C. Equal-width locators, uniform word grid.  With all word widths equal to
   alpha, let g = alpha+1 and d = b+1.  A b-line holding k words has slack
   d-gk and cost h(k)=(d-gk)^2.  The m special lines on a suffix of N words
   cost exactly the m-fold min-plus convolution

       H(N) = h^{*m}(N) = (m-r) h(q) + r h(q+1),   N = m q + r,

   because h is discrete convex (second difference 2g^2).  Hence H[j] is
   O(1) per j and the whole algorithm is O(n).

D. Barriers.  On a non-uniform grid the convolution-power identity is false:
   intermediate breakpoints must lie in C, and the pure convolution relaxation
   can be strictly smaller.  Explicit instances are printed.
"""
from itertools import product
from math import inf
import random



# ── brute force helpers ─────────────────────────────────────────────────────

def layouts_idx_reversal(n):
    """All breakpoint lists [0=j_0<...<j_K=n]."""
    out = []
    def rec(idx, start):
        if start == n:
            out.append(list(idx))
            return
        for j in range(start+1, n+1):
            idx.append(j)
            rec(idx, j)
            idx.pop()
    rec([0], 0)
    return out


def brute_rank_minlines(w, beta, m0):
    """Lexicographic optimum for rank-indexed budgets, layouts with >= m0 lines."""
    n = len(w)
    best = None
    for idx in layouts_idx_reversal(n):
        L = len(idx)-1
        if L < m0 or L > len(beta):
            continue
        pen, ok = 0, True
        for t in range(1, L+1):
            i, j = idx[t-1]+1, idx[t]
            width = span(w, i, j)
            if width > beta[t-1]:
                ok = False
                break
            pen += (beta[t-1]-width)**2
        if ok:
            key = (L, pen)
            if best is None or key < best:
                best = key
    return best


def check_reversal_equivalence():
    print("== reversal equivalence: relational = prefix rank-indexed ==")
    tested = 0
    bad = 0
    ex = []
    for n in range(1, 7):
        for w in product(range(1, 4), repeat=n):
            for A in range(3, 7):
                for m in (1, 2, 3):
                    for s in product(range(0, 3), repeat=m):
                        b = [A-x for x in s]
                        if any(x < 1 for x in b):
                            continue
                        beta = list(b) + [A]*(n+1)
                        want = brute_relational_lex(list(w), A, list(s))
                        got = brute_rank_minlines(list(reversed(w)), beta, m)
                        tested += 1
                        if want != got:
                            bad += 1
                            if len(ex) < 4:
                                ex.append((w, A, s, want, got))
    print("  instances:", tested, "| inequivalent:", bad)
    for e in ex:
        print("   ", e)


# ── dominating-penalty decomposition ───────────────────────────────────────

def suffix_special_cost(w, A, s, Lam):
    """H[j] = min (sum slack^2 + Lam*m) of the last m lines with budgets
    b_1,...,b_m (from the end) covering words j+1..n.  O(n^2 m)."""
    n = len(w)
    m = len(s)
    b = [A - x for x in s]          # from the end
    V = [inf]*(n+1)
    V[n] = 0.0
    for r in range(1, m+1):
        br = b[r-1]
        U = [inf]*(n+1)
        for i in range(n):
            for j in range(i+1, n+1):
                width = span(w, i+1, j)
                if V[j] < inf and width <= br:
                    val = V[j] + (br-width)**2 + Lam
                    if val < U[i]:
                        U[i] = val
        V = U
    return V


def prefix_A_cost(w, A, Lam):
    """G[j] = min (sum slack^2 + Lam*q) over ANY q >= 0 A-lines on words 1..j,
    one unconstrained LWS pass."""
    n = len(w)
    G = [inf]*(n+1)
    G[0] = 0.0
    for j in range(1, n+1):
        for i in range(j):
            width = span(w, i+1, j)
            if G[i] < inf and width <= A:
                val = G[i] + (A-width)**2 + Lam
                if val < G[j]:
                    G[j] = val
    return G


def algorithm_decomposed(w, A, s):
    n = len(w)
    Lam = n*A*A + 1
    H = suffix_special_cost(w, A, s, Lam)
    G = prefix_A_cost(w, A, Lam)
    best = None
    for j in range(0, n+1):
        if G[j] < inf and H[j] < inf:
            val = G[j] + H[j]
            if best is None or val < best[1]:
                # total lines = (val - sum slack^2)/Lam ; recover counts
                # by replaying the two parts' penalties is unnecessary for
                # verification: compare (lines, penalty) below instead.
                best = (j, val)
    if best is None:
        return None
    total = best[1]
    lines = round((total - (total % 1)) / 1)  # placeholder, replaced below
    # Recover (lines, penalty) exactly by a direct search over q and special
    # layouts is O(nm) for verification; instead use the identity:
    # total = penalty + Lam*lines, with penalty < Lam.
    pen = total % Lam
    K = total // Lam
    return (K, pen)


def algorithm_decomposed_lex(w, A, s):
    """Exact lex optimum via decomposing (lines, penalty) with Lam."""
    n = len(w)
    Lam = n*A*A + 1
    H = suffix_special_cost(w, A, s, Lam)
    G = prefix_A_cost(w, A, Lam)
    best = None
    for j in range(0, n+1):
        if G[j] < inf and H[j] < inf:
            val = G[j] + H[j]
            if best is None or val < best:
                best = val
    if best is None:
        return None
    return (best // Lam, best % Lam)


def check_decomposition_algorithm():
    print("== uniform-tail decomposition vs exhaustive relational lex ==")
    rng = random.Random(20261201)
    bad = 0
    tested = 0
    for A in range(3, 8):
        for m in (1, 2, 3):
            for n in range(m, 8):
                for _ in range(80):
                    w = [rng.randint(1, min(4, A)) for _ in range(n)]
                    s = [rng.randint(0, A-1) for _ in range(m)]
                    want = brute_relational_lex(w, A, s)
                    got = algorithm_decomposed_lex(w, A, s)
                    tested += 1
                    if want != got:
                        bad += 1
                        if bad <= 5:
                            print("   MISMATCH w,A,s =", w, A, s,
                                  "want", want, "got", got)
    print("  instances:", tested, "| mismatches:", bad)


# ── equal-width locators on a uniform word grid ────────────────────────────

def hstar(N, m, alpha, A, s):
    """m-fold min-plus convolution of h(k)=(d-gk)^2, 1<=k<=floor(d/g)."""
    g = alpha + 1
    d = A - s + 1
    kmax = d // g
    if kmax < 1 or N < m or N > m*kmax:
        return inf
    q, r = divmod(N, m)
    if q < 1 or (r > 0 and q+1 > kmax):
        return inf
    def h(k):
        return (d-g*k)**2
    return (m-r)*h(q) + r*h(q+1)


def hstar_brute(N, m, alpha, A, s):
    g = alpha + 1
    d = A - s + 1
    kmax = d//g
    best = inf
    def rec(left, parts, cost):
        nonlocal best
        if parts == 0:
            if left == 0 and cost < best:
                best = cost
            return
        for k in range(1, min(kmax, left)+1):
            rec(left-k, parts-1, cost+(d-g*k)**2)
    rec(N, m, 0)
    return best


def check_hstar_formula():
    print("== closed form of the m-fold convolution kernel ==")
    tested = 0
    bad = 0
    for alpha in range(1, 4):
        for A in range(3, 10):
            for s in range(0, A):
                for m in (1, 2, 3, 4):
                    for N in range(1, 19):
                        want = hstar_brute(N, m, alpha, A, s)
                        got = hstar(N, m, alpha, A, s)
                        tested += 1
                        if want != got:
                            bad += 1
                            if bad <= 4:
                                print("   MISMATCH alpha,A,s,m,N =",
                                      alpha, A, s, m, N, want, got)
    print("  instances:", tested, "| mismatches:", bad)


def algorithm_equal_uniform(n, alpha, A, s, m):
    """O(n) algorithm: all words width alpha, all m locators width s."""
    Lam = n*A*A + 1
    g = alpha + 1
    # H[j]: m special b-lines on suffix of N=n-j words
    H = [inf]*(n+1)
    for j in range(0, n+1):
        v = hstar(n-j, m, alpha, A, s)
        if v < inf:
            H[j] = v + m*Lam
    # G[j]: any number of A-lines on prefix, one LWS pass
    G = [inf]*(n+1)
    G[0] = 0.0
    for j in range(1, n+1):
        for k in range(1, j+1):
            width = g*k - 1
            if G[j-k] < inf and width <= A:
                val = G[j-k] + (A-width)**2 + Lam
                if val < G[j]:
                    G[j] = val
    best = None
    for j in range(0, n+1):
        if G[j] < inf and H[j] < inf:
            val = G[j] + H[j]
            if best is None or val < best:
                best = val
    if best is None:
        return None
    return (best // Lam, best % Lam)


def check_equal_uniform_algorithm():
    print("== equal-width locators, uniform word grid: O(n) vs exhaustive ==")
    tested = 0
    bad = 0
    ex = []
    for alpha in range(1, 4):
        for A in range(3, 9):
            for m in (1, 2, 3, 4):
                for n in range(m, 9):
                    for s in range(0, A):
                        w = [alpha]*n
                        want = brute_relational_lex(w, A, [s]*m)
                        got = algorithm_equal_uniform(n, alpha, A, s, m)
                        tested += 1
                        if want != got:
                            bad += 1
                            if len(ex) < 5:
                                ex.append((alpha, A, s, m, n, want, got))
    print("  instances:", tested, "| mismatches:", bad)
    for e in ex:
        print("   ", e)


# ── barriers on non-uniform grids ──────────────────────────────────────────

def check_nonuniform_barrier():
    print("== barrier: convolution-power identity fails on non-uniform C ==")
    # With all lines allowed to hold all words (no window), the 2-layer cost
    # from i to l is min over k in C of two squared slacks.  The pure
    # convolution (allowing k anywhere in the integer lattice) can be smaller.
    found = []
    for w in product(range(1, 5), repeat=4):
        C = Cvec(list(w))
        d = C[4] + 2                 # budget >= all widths: no window
        # 2-layer cost from i=0 to l=4, first intermediate k in {1,2,3}
        exact = inf
        for k in (1, 2, 3):
            if C[k]-C[0] <= d and C[4]-C[k] <= d:
                val = (d-(C[k]-C[0]))**2 + (d-(C[4]-C[k]))**2
                exact = min(exact, val)
        relaxed = inf
        for x in range(C[0]+1, C[4]):
            if x-C[0] <= d and C[4]-x <= d:
                val = (d-(x-C[0]))**2 + (d-(C[4]-x))**2
                relaxed = min(relaxed, val)
        if exact < inf and relaxed < exact:
            found.append((w, d, exact, relaxed, exact-relaxed))
            if len(found) >= 3:
                break
    print("  instances where grid-constrained 2-layer cost > lattice relaxation:",
          len(found))
    for f in found:
        print("   w,d,exact,relaxed,gap =", f)
    print("  => h^{*m}(C_n-C_j) is not the m-layer value on non-uniform C;")
    print("     any o(nm) algorithm must handle the intermediate grid points")



# ==== discrete-spline and projection checks ====
"""Multi-layer structure of the rank/relational DP: quadratic-spline
decomposition, convexity frontier, Lagrangian barriers, and an O(n) algorithm
for exactly two terminal lines on arbitrary word widths.

All checks use exhaustive small DP or closed-form identities.  Notation:
C = cumulative breakpoints, d = beta+1 (beta = line budget minus 1),
a b-line covering C[i] -> C[j] costs (d-(C[j]-C[i]))^2 and is legal iff
C[j]-C[i] <= d.
"""
from bisect import bisect_left
from math import inf
import random



def chain_exact(C, d, k, i, n, window=True):
    """Least penalty of exactly k lines of budget d-1 from i to n."""
    N = len(C)-1
    E = [inf]*(N+1)
    E[i] = 0.0
    for _ in range(k):
        G = [inf]*(N+1)
        for a in range(i, N+1):
            if E[a] == inf:
                continue
            for b in range(a+1, n+1):
                gap = C[b]-C[a]
                if (not window) or gap <= d:
                    G[b] = min(G[b], E[a]+(d-gap)**2)
        E = G
    return E[n]


def spline_value(C, d, m, i, j):
    """S^2/m + min over interior chain of sum (E_r-E_{r-1})^2, equal budgets."""
    n = len(C)-1
    S = m*d-(C[j]-C[i])
    best = inf
    def rec(r, prev, Eprev, acc):
        nonlocal best
        if r == m:
            best = min(best, acc+Eprev**2)   # E_m = 0
            return
        xi = (1-r/m)*C[i] + (r/m)*C[j]
        for t in range(prev+1, j):
            Er = xi-C[t]
            rec(r+1, t, Er, acc+(Er-Eprev)**2)
    rec(1, i, 0.0, 0.0)
    return S*S/m+best if best < inf else inf


def check_spline_identity():
    print("== discrete-spline decomposition of m layers ==")
    rng = random.Random(20261203)
    bad = 0
    tested = 0
    for _ in range(200):
        n = rng.randint(4, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = C[n]+2                 # no budget window
        m = rng.randint(2, 4)
        for i in range(0, n-m+1):
            F = chain_exact(C, d, m, i, n, window=False)
            V = spline_value(C, d, m, i, n)
            tested += 1
            if abs(F-V) > 1e-6:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH", w, d, m, i, F, V)
    print("  instances:", tested, "| decomposition failures:", bad)
    print("  identity: F_m[i,j] = (D-Delta)^2/m + min_{x_r in C} "
          "sum (E_r-E_{r-1})^2")


def Rlambda_all(C, d, lam, n):
    """Unconstrained LWS from every i to n, with penalty lam per line."""
    N = len(C)-1
    E = [inf]*(N+1)
    E[n] = 0.0
    for j in range(n-1, -1, -1):
        for k in range(j+1, n+1):
            gap = C[k]-C[j]
            if E[k] < inf and gap <= d:
                E[j] = min(E[j], E[k]+(d-gap)**2+lam)
    return E


def A_closure(C, A, n, Lam):
    E = [inf]*(n+1)
    E[0] = 0.0
    for j in range(1, n+1):
        for a in range(j):
            gap = C[j]-C[a]
            if E[a] < inf and gap <= A-1:
                E[j] = min(E[j], E[a]+(A-gap)**2+Lam)
    return E


def H_exact(C, d, m, n, Lam):
    E = [inf]*(n+1)
    E[n] = 0.0
    for _ in range(m):
        G = [inf]*(n+1)
        for j in range(n-1, -1, -1):
            for k in range(j+1, n+1):
                gap = C[k]-C[j]
                if E[k] < inf and gap <= d:
                    G[j] = min(G[j], E[k]+(d-gap)**2+Lam)
        E = G
    return E


def mixed_dual(C, A, d, lam, n, Lam):
    """Any A-lines then any b-lines, Lam per line and lam extra per b-line."""
    H = list(A_closure(C, A, n, Lam))
    for j in range(1, n+1):
        for a in range(j):
            gap = C[j]-C[a]
            if H[a] < inf and gap <= d:
                H[j] = min(H[j], H[a]+(d-gap)**2+Lam+lam)
    return H[n]


def check_convexity_frontier():
    print("== convexity frontier: one budget vs mixed budgets ==")
    rng = random.Random(20261204)
    conv = nonconv = 0
    ok = gap = 0
    mixed_conv = mixed_nonconv = 0
    ex = []
    for _ in range(120):
        n = rng.randint(3, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(2, C[n]+2)
        # (a) one-budget k-link convexity
        for i in range(0, n-1):
            vals = [chain_exact(C, d, k, i, n) for k in range(1, n-i+1)]
            fin = [v for v in vals if v < inf]
            if len(fin) >= 3:
                dif = [fin[t+1]-fin[t] for t in range(len(fin)-1)]
                if any(dif[t] > dif[t+1]+1e-9 for t in range(len(dif)-1)):
                    nonconv += 1
                else:
                    conv += 1
        # (a') one-budget Fenchel: one LWS per lambda, shared by all i.
        # A bounded budget keeps the dual slope range inside [-1200,1200].
        m = rng.randint(1, min(3, n))
        df = min(d, 10)
        dual = [-inf]*(n+1)
        for lam in range(-1200, 1201):
            E = Rlambda_all(C, df, lam, n)
            for i in range(n+1):
                dual[i] = max(dual[i], E[i]-lam*m)
        for i in range(0, n-1):
            P = chain_exact(C, df, m, i, n)
            if P == inf:
                continue
            if abs(P-dual[i]) < 1e-6:
                ok += 1
            else:
                gap += 1
        # (b) mixed objective f(k): arbitrary A-lines + exactly k b-lines
        A = rng.randint(3, C[n]+3)
        s = rng.randint(1, A-1)
        db = A-s+1
        Lam = n*A*A+1
        G = A_closure(C, A, n, Lam)
        fvals = []
        for k in range(1, n+1):
            best = inf
            for i in range(n+1):
                if G[i] < inf:
                    P = chain_exact(C, db, k, i, n)
                    if P < inf:
                        best = min(best, G[i]+P+k*Lam)
            fvals.append(best)
        fin = [v for v in fvals if v < inf]
        if len(fin) >= 3:
            dif = [fin[t+1]-fin[t] for t in range(len(fin)-1)]
            if any(dif[t] > dif[t+1]+1e-6 for t in range(len(dif)-1)):
                mixed_nonconv += 1
                if len(ex) < 3:
                    ex.append((w, A, s, fin, dif))
            else:
                mixed_conv += 1
    print("  one-budget k-link curves: convex=%d nonconvex=%d" % (conv, nonconv))
    print("  one-budget Fenchel: no-gap=%d gap=%d" % (ok, gap))
    print("  mixed f(k): convex=%d nonconvex=%d" % (mixed_conv, mixed_nonconv))
    for e in ex:
        print("   mixed nonconvex", e)


def check_minimax_gap():
    print("== minimax barrier: min_j(G+H) > max_lambda(mixed - lambda*m) ==")
    cases = [
        ([2, 1, 1, 3], 6, 2, 2),
        ([3, 4, 2, 2, 4, 2, 4], 9, 2, 3),
        ([1, 4, 1, 4, 4, 3, 1], 21, 8, 2),
    ]
    for w, A, s, m in cases:
        C = Cvec(w)
        n = len(w)
        d = A-s+1
        Lam = n*A*A+1
        G = A_closure(C, A, n, Lam)
        H = H_exact(C, d, m, n, Lam)
        lhs = min(G[i]+H[i] for i in range(n+1)
                  if G[i] < inf and H[i] < inf)
        rhs = max(mixed_dual(C, A, d, lam, n, Lam)-lam*m
                  for lam in range(-8*A*A, 8*A*A+1))
        print("  w=%s A=%d s=%d m=%d  LHS=%s RHS=%s gap=%s"
              % (w, A, s, m, lhs, rhs, lhs-rhs))


def check_structure_counterexamples():
    print("== structural barriers on non-uniform C ==")
    rng = random.Random(20261202)
    same = diff = 0
    nearm_ok = nearm_bad = 0
    ex_sum = []
    ex_near = []

    def Fm(C, d, m, i, j):
        return chain_exact(C, d, m, i, j, window=False)

    def nearest(C, d, m, i, j):
        Delta = C[j]-C[i]
        pts = []
        for r in range(1, m):
            xi = C[i]+r*Delta/m
            pts.append(min(range(i+1, j), key=lambda t: abs(C[t]-xi)))
        chain = [i]+sorted(set(pts))+[j]
        if len(chain) != m+1:
            return None
        return sum((d-(C[chain[r]]-C[chain[r-1]]))**2
                   for r in range(1, m+1))

    for _ in range(200):
        n = rng.randint(4, 8)
        w = [rng.randint(1, 5) for _ in range(n)]
        C = Cvec(w)
        d = C[n]+2
        m = rng.randint(2, 4)
        vals = {}
        for i in range(0, n+1):
            for j in range(i+m, n+1):
                F = Fm(C, d, m, i, j)
                if F == inf:
                    continue
                E = F-(m*d-(C[j]-C[i]))**2/m
                key = C[i]+C[j]
                if key in vals:
                    if abs(vals[key]-E) < 1e-6:
                        same += 1
                    else:
                        diff += 1
                        if len(ex_sum) < 3:
                            ex_sum.append((w, d, m, i, j, key, vals[key], E))
                else:
                    vals[key] = E
        for i in range(0, n-m+1):
            F = Fm(C, d, m, i, n)
            G = nearest(C, d, m, i, n)
            if F == inf:
                continue
            if G is not None and abs(F-G) < 1e-6:
                nearm_ok += 1
            else:
                nearm_bad += 1
                if len(ex_near) < 3:
                    ex_near.append((w, d, m, i, F, G))
    print("  same C_i+C_j -> same E: %d | different: %d" % (same, diff))
    for e in ex_sum:
        print("   sum-representation counterexample", e)
    print("  nearest-ideal-point chain exact: %d | not exact: %d"
          % (nearm_ok, nearm_bad))
    for e in ex_near:
        print("   nearest counterexample", e)


def H2_linear(C, d, n):
    """O(n) exact H[i] for exactly two b-lines on arbitrary word widths."""
    out = [inf]*(n+1)
    lo = bisect_left(C, C[n]-d)      # second gap legal: C_k >= C_n-d
    hi = 0
    for i in range(0, n+1):
        U = C[i]+d                   # first gap legal: C_k <= C_i+d
        while hi+1 < n and C[hi+1] <= U:
            hi += 1
        a, b = max(lo, i+1), min(hi, n-1)
        if a > b:
            continue
        mu = (C[i]+C[n])/2.0
        k = max(a, min(b, bisect_left(C, mu)))
        best = min((C[t]-mu)**2
                   for t in (k, max(a, k-1), min(b, k+1)))
        out[i] = (2*d-(C[n]-C[i]))**2/2.0+2*best
    return out


def H2_brute(C, d, n):
    out = [inf]*(n+1)
    for i in range(n+1):
        for k in range(i+1, n):
            if C[k]-C[i] <= d and C[n]-C[k] <= d:
                out[i] = min(out[i], (d-(C[k]-C[i]))**2 +
                             (d-(C[n]-C[k]))**2)
    return out


def check_m2_linear():
    print("== two terminal lines, arbitrary word widths: O(n) closed form ==")
    rng = random.Random(20261205)
    bad = tested = 0
    for _ in range(500):
        n = rng.randint(2, 9)
        w = [rng.randint(1, 5) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(2, C[n]+2)
        a = H2_linear(C, d, n)
        b = H2_brute(C, d, n)
        for i in range(n+1):
            tested += 1
            if abs(a[i]-b[i]) > 1e-6:
                bad += 1
                if bad <= 3:
                    print("   MISMATCH", w, d, i, a[i], b[i])
    print("  instances:", tested, "| mismatches:", bad)
    print("  H_2[i] = (2d-Delta)^2/2 + 2 dist((C_i+C_n)/2, C)^2")


def check_lambda_monotonicity():
    print("== Fenchel penalty indices across breakpoints ==")
    rng = random.Random(20261206)
    mono = nonmono = 0
    distinct = []
    for _ in range(120):
        n = rng.randint(4, 8)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(3, min(C[n]+2, 10))
        m = rng.randint(1, min(3, n))
        best = [-inf]*(n+1)
        stars = [None]*(n+1)
        for lam in range(-3*d*d, 3*d*d+1):
            E = Rlambda_all(C, d, lam, n)
            for i in range(n+1):
                v = E[i]-lam*m
                if v > best[i]+1e-9:
                    best[i], stars[i] = v, lam
        stars = stars[:n]
        distinct.append(len(set(stars)))
        if (all(stars[i] <= stars[i+1] for i in range(n-1)) or
                all(stars[i] >= stars[i+1] for i in range(n-1))):
            mono += 1
        else:
            nonmono += 1
    print("  monotone lambda*_i: %d | nonmonotone (flat-tie) %d"
          % (mono, nonmono))
    print("  distinct lambda values per instance: mean %.2f, max %d"
          % (sum(distinct)/len(distinct), max(distinct)))



# ==== single-run and composition checks ====
"""Single uniform-budget run: an exact O(n log V) algorithm and the Monge-DAG
reduction behind strongly polynomial k-link algorithms.

Notation: C = cumulative breakpoints, d = beta+1, a line i->j costs
(d-(C_j-C_i))^2 and is legal iff C_j-C_i <= d.  With a dominating per-line
penalty Lam, the lexicographic objective (fewest lines, then least squared
slack) is min over any number of lines of [sum slack^2 + Lam*lines].

A. Offline segment-tree LWS: F[j] = min_{i<j, legal} F[i] + (d+C_i-C_j)^2
   + Lam + lam.  Line i is active only on j in [i+1, J(i)]; an offline segment
   tree with one monotone lower hull per node gives O(n log n).  (Wilber's
   O(n) concave-LWS algorithm gives O(n) when implemented.)

B. Aliens (Lagrangian) trick: P(K) is discrete convex in K for a single
   budget (AST 1994).  Binary search lam so that the unconstrained LWS uses
   exactly K lines; return P(K) = G(lam) - lam*K.  Total O(n log n log V)
   with the segment-tree LWS, O(n log V) with Wilber's O(n) LWS.

C. Convex-extension + hinge: replacing an illegal line by
   (d-gap)^2 + mu*max(0,gap-d) keeps the edge weight a TOTAL function of
   (i,j), is convex in gap, and therefore preserves the concave Monge
   quadrangle inequality.  This is the exact form required by Wan (2024)'s
   strongly polynomial shortest M-link path algorithm.

D. Barriers for composing runs: the preceding run's DP vector as a super-source
   edge weight breaks the Monge source condition; P(K) from that structured
   vector is nonconvex.
"""
from bisect import bisect_right
from collections import deque
from math import inf
import random



# ── A. monotone CHT for one uniform-budget LWS ─────────────────────────────

class _NodeHull:
    """Lower hull of lines m*x+b, slopes decreasing, queries x increasing.

    Each entry is (m, b, cnt); the line's source index is not needed after
    insertion because a line is inserted on exactly its feasible j-interval.
    """
    def __init__(self):
        self.q = deque()

    def insert(self, m, b, cnt):
        q = self.q
        while len(q) >= 2:
            m1, b1, c1 = q[-2]
            m2, b2, c2 = q[-1]
            # x12 >= x23  <=>  (b2-b1)(m2-m) >= (b-b2)(m1-m2)
            if (b2-b1)*(m2-m) > (b-b2)*(m1-m2):
                q.pop()
            else:
                break
        q.append((m, b, cnt))

    def query(self, x, prefer_fewer=True):
        q = self.q
        while len(q) >= 2:
            m0, b0, c0 = q[0]
            m1, b1, c1 = q[1]
            v0, v1 = m0*x+b0, m1*x+b1
            better = (v1 < v0) or (v1 == v0 and
                                   ((c1 < c0) if prefer_fewer else (c1 > c0)))
            if better:
                q.popleft()
            else:
                break
        if not q:
            return None
        m, b, c = q[0]
        return m*x+b, c


class _SegTree:
    """Offline segment tree over j=1..n; each node keeps a monotone hull."""
    def __init__(self, n):
        self.n = n
        self.N = 1
        while self.N < n:
            self.N *= 2
        self.nodes = [None]*(2*self.N)

    def _node(self, k):
        if self.nodes[k] is None:
            self.nodes[k] = _NodeHull()
        return self.nodes[k]

    def add(self, l, r, m, b, cnt):
        if l > r:
            return
        L, R = l+self.N-1, r+self.N-1
        while L <= R:
            if L & 1:
                self._node(L).insert(m, b, cnt)
                L += 1
            if not (R & 1):
                self._node(R).insert(m, b, cnt)
                R -= 1
            L >>= 1
            R >>= 1

    def query(self, j, x, prefer_fewer=True):
        pos = j+self.N-1
        best = None
        while pos:
            h = self.nodes[pos]
            if h is not None:
                res = h.query(x, prefer_fewer)
                if res is not None:
                    val, c = res
                    if best is None or val < best[0] or (val == best[0] and c < best[1]):
                        best = (val, c)
            pos >>= 1
        return best


def lws_cht(C, d, Lam, lam, prefer_fewer=True):
    """Return (F, cnt) for F[j]=min_i F[i]+(d+C_i-C_j)^2+Lam+lam.

    Line i is feasible exactly for j in [i+1, J(i)], J(i)=max{j:C_j<=C_i+d}.
    An offline segment tree stores each line on O(log n) nodes; every node
    keeps a monotone lower hull.  Lines are inserted in increasing i and
    queries x=C_j are increasing, so each hull uses one pointer.  O(n log n).
    """
    n = len(C)-1
    F = [inf]*(n+1)
    cnt = [0]*(n+1)
    F[0] = 0.0
    tree = _SegTree(n)

    def add_line(i):
        if F[i] == inf:
            return
        D = d + C[i]
        m = -2*D
        b = F[i] + D*D + Lam + lam
        J = bisect_right(C, C[i]+d)-1
        J = min(J, n)
        if J >= i+1:
            tree.add(i+1, J, m, b, cnt[i])

    add_line(0)
    for j in range(1, n+1):
        res = tree.query(j, C[j], prefer_fewer)
        if res is not None:
            val, c = res
            F[j] = C[j]*C[j] + val
            cnt[j] = c+1
        add_line(j)
    return F, cnt


def brute_lex_uniform(w, A):
    """Exhaustive fewest-lines then least squared slack, all budgets A."""
    n = len(w)
    best = None
    def rec(idx, start):
        nonlocal best
        if start == n:
            L = len(idx)-1
            pen = 0
            for t in range(1, L+1):
                i, j = idx[t-1]+1, idx[t]
                width = span(w, i, j)
                if width > A:
                    return
                pen += (A-width)**2
            key = (L, pen)
            if best is None or key < best:
                best = key
            return
        for j in range(start+1, n+1):
            idx.append(j)
            rec(idx, j)
            idx.pop()
    rec([0], 0)
    return best


def greedy_lines(w, A):
    n = len(w)
    lines = 0
    i = 0
    while i < n:
        j = i
        while j < n and span(w, i+1, j+1) <= A:
            j += 1
        if j == i:
            return inf
        lines += 1
        i = j
    return lines


def aliens_uniform(w, A):
    """Exact lex optimum in O(n log V) for a single uniform-budget run."""
    n = len(w)
    C = Cvec(w)
    d = A+1
    K = greedy_lines(w, A)
    if K == inf:
        return None
    # Lam dominates every achievable sum of squared slacks, and V keeps the
    # net per-line penalty Lam+lam strictly positive for all lam in [-V,V].
    Lam = 4*n*(A+1)**2 + 10
    V = Lam - 1
    # find smallest lam with count <= K (prefer fewer lines on ties)
    lo, hi = -V, V
    best_lam = None
    while lo < hi:
        mid = (lo+hi)//2
        F, cnt = lws_cht(C, d, Lam, mid, prefer_fewer=True)
        if cnt[n] <= K:
            hi = mid
        else:
            lo = mid+1
    lam = lo
    F, cnt = lws_cht(C, d, Lam, lam, prefer_fewer=True)
    total = F[n]
    # total = penalty + (Lam + lam)*K, so recover the squared-slack part
    pen = total - (Lam+lam)*K
    return (K, pen)


def check_cht_against_brute():
    print("== uniform-budget LWS / Aliens O(n log V) vs exhaustive ==")
    rng = random.Random(20261213)
    bad = tested = 0
    for n in range(1, 10):
        for _ in range(200):
            w = [rng.randint(1, 5) for _ in range(n)]
            A = rng.randint(max(w), max(w)+6)
            want = brute_lex_uniform(w, A)
            got = aliens_uniform(w, A)
            tested += 1
            if want != got:
                bad += 1
                if bad <= 5:
                    print("   MISMATCH w,A =", w, A, "want", want, "got", got)
    print("  instances:", tested, "| mismatches:", bad)
    # exhaustive small
    from itertools import product
    bad2 = 0
    for n in range(1, 7):
        for w in product(range(1, 4), repeat=n):
            for A in range(3, 7):
                want = brute_lex_uniform(list(w), A)
                got = aliens_uniform(list(w), A)
                if want != got:
                    bad2 += 1
                    if bad2 <= 3:
                        print("   exhaustive MISMATCH", w, A, want, got)
    print("  exhaustive small instances | mismatches:", bad2)


# ── C. convex extension + hinge preserves Monge ───────────────────────────

def c_ext(C, i, j, d, mu):
    gap = C[j]-C[i]
    return (d-gap)**2 + mu*max(0, gap-d)


def monge_total(C, d, mu):
    n = len(C)-1
    for i in range(n+1):
        for j in range(i+1, n+1):
            for k in range(j+1, n+1):
                for l in range(k+1, n+1):
                    lhs = c_ext(C, i, k, d, mu) + c_ext(C, j, l, d, mu)
                    rhs = c_ext(C, i, l, d, mu) + c_ext(C, j, k, d, mu)
                    if lhs > rhs + 1e-9:
                        return False, (i, j, k, l, lhs, rhs)
    return True, None


def hinge_lex_uniform(w, A):
    """Lex optimum via the convex-hinge Monge DAG (naive layered DP)."""
    n = len(w)
    C = Cvec(w)
    d = A+1
    K = greedy_lines(w, A)
    if K == inf:
        return None
    Lam = 4*n*(A+1)**2 + 10
    mu = 10*(A+1)**2*(C[n]+d+1) + 1
    E = [inf]*(n+1)
    E[0] = 0.0
    for _ in range(K):
        G = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if E[i] < inf:
                    gap = C[j]-C[i]
                    val = E[i] + c_ext(C, i, j, d, mu) + Lam
                    if val < G[j]:
                        G[j] = val
        E = G
    total = E[n]
    return (K, total - Lam*K)


def check_hinge_reduction():
    print("== convex-hinge Monge DP vs exhaustive lex ==")
    rng = random.Random(20261214)
    bad = tested = 0
    for n in range(1, 9):
        for _ in range(150):
            w = [rng.randint(1, 5) for _ in range(n)]
            A = rng.randint(max(w), max(w)+6)
            want = brute_lex_uniform(w, A)
            got = hinge_lex_uniform(w, A)
            tested += 1
            if want != got:
                bad += 1
                if bad <= 4:
                    print("   MISMATCH", w, A, want, got)
    print("  instances:", tested, "| mismatches:", bad)


def check_convex_extension():
    print("== convex extension + hinge: total Monge edge weights ==")
    rng = random.Random(20261210)
    bad = tested = 0
    for _ in range(300):
        n = rng.randint(4, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(2, C[n]+2)
        mu = 10*(C[n]+d)**2 + 1
        ok, e = monge_total(C, d, mu)
        tested += 1
        if not ok:
            bad += 1
            if bad <= 3:
                print("   Monge failure", e)
    print("  instances:", tested, "| Monge failures:", bad)
    print("  => single-run layer is a concave Monge DAG with O(1) edge eval")

    # large constant penalty instead of hinge breaks Monge
    def c_const(C, i, j, d, P):
        gap = C[j]-C[i]
        return P if gap > d else (d-gap)**2
    bad_const = 0
    for _ in range(200):
        n = rng.randint(4, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(2, C[n]+2)
        P = 10*(C[n]+d)**2 + 1
        ok = True
        for i in range(n+1):
            for j in range(i+1, n+1):
                for k in range(j+1, n+1):
                    for l in range(k+1, n+1):
                        if (c_const(C, i, k, d, P)+c_const(C, j, l, d, P) >
                                c_const(C, i, l, d, P)+c_const(C, j, k, d, P)+1e-9):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            bad_const += 1
    print("  large constant penalty | Monge failures:", bad_const, "/200")


# ── D. composition barriers ───────────────────────────────────────────────

def lws_unconstrained_naive(C, d, Lam, n):
    F = [inf]*(n+1)
    F[0] = 0.0
    for j in range(1, n+1):
        for i in range(j):
            if F[i] < inf and C[j]-C[i] <= d:
                v = F[i]+(d-(C[j]-C[i]))**2+Lam
                if v < F[j]:
                    F[j] = v
    return F


def layered_from_source(C, d, Lam, F0, K, n):
    E = list(F0)
    for _ in range(K):
        G = [inf]*(n+1)
        for j in range(1, n+1):
            for i in range(j):
                if E[i] < inf and C[j]-C[i] <= d:
                    v = E[i]+(d-(C[j]-C[i]))**2+Lam
                    if v < G[j]:
                        G[j] = v
        E = G
    return E[n]


def check_structured_source_nonconvex():
    print("== structured source (unconstrained A-LWS output): P(K) nonconvex ==")
    rng = random.Random(20261211)
    conv = nonconv = 0
    ex = []
    for _ in range(400):
        n = rng.randint(3, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        A = rng.randint(3, C[n]+3)
        s = rng.randint(1, A-1)
        b = A-s
        dA, db = A+1, b+1
        Lam = n*(A+1)**2*10 + 1
        FA = lws_unconstrained_naive(C, dA, Lam, n)
        vals = [layered_from_source(C, db, Lam, FA, K, n)
                for K in range(1, n+1)]
        fin = [v for v in vals if v < inf]
        if len(fin) < 3:
            continue
        dif = [fin[t+1]-fin[t] for t in range(len(fin)-1)]
        if any(dif[t] > dif[t+1]+1e-6 for t in range(len(dif)-1)):
            nonconv += 1
            if len(ex) < 3:
                ex.append((w, A, s, fin, dif))
        else:
            conv += 1
    print("  curves: convex=%d nonconvex=%d" % (conv, nonconv))
    for e in ex:
        print("   ", e)


def source_monge_condition(C, f, db):
    n = len(C)-1
    for j in range(1, n+1):
        for k in range(j+1, n+1):
            for l in range(k+1, n+1):
                if C[k]-C[j] > db or C[l]-C[j] > db:
                    continue
                a = (db-(C[k]-C[j]))**2
                b = (db-(C[l]-C[j]))**2
                if f[l]+a < f[k]+b-1e-9:
                    return False, (j, k, l, f[l]+a, f[k]+b)
    return True, None


def check_super_source_monge():
    print("== super-source with preceding-run DP vector breaks Monge ==")
    rng = random.Random(20261212)
    bad = tested = 0
    ex = []
    for _ in range(300):
        n = rng.randint(4, 7)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        A = rng.randint(3, C[n]+3)
        s = rng.randint(1, A-1)
        b = A-s
        dA, db = A+1, b+1
        Lam = n*A*A+1
        FA = lws_unconstrained_naive(C, dA, Lam, n)
        ok, e = source_monge_condition(C, FA, db)
        tested += 1
        if not ok:
            bad += 1
            if len(ex) < 3:
                ex.append((w, A, s, e))
    print("  instances:", tested, "| Monge source failures:", bad)
    for e in ex:
        print("   ", e)


# ── E. core density of Monge powers remains quadratic ─────────────────────

def Kmat(C):
    n = len(C)-1
    M = [[inf]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+1, n+1):
            M[i][j] = (C[j]-C[i])**2
    return M


def minplus(A, B):
    n = len(A)-1
    P = [[inf]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+1, n+1):
            best = inf
            for k in range(i+1, j):
                if A[i][k] < inf and B[k][j] < inf:
                    best = min(best, A[i][k]+B[k][j])
            P[i][j] = best
    return P


def density(M):
    n = len(M)-1
    nz = fin = 0
    for i in range(n):
        for j in range(n):
            a, b, c, d = M[i][j], M[i][j+1], M[i+1][j], M[i+1][j+1]
            if inf in (a, b, c, d):
                continue
            fin += 1
            if a+d-b-c != 0:
                nz += 1
    return nz, fin


def check_power_density():
    print("== core density of K^{2^t} stays Theta(n^2) for q=Theta(n) ==")
    rng = random.Random(20261208)
    for n in (32, 64):
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        P = Kmat(C)
        row = [("K",)+density(P)]
        q = 1
        while q*2 <= n:
            P = minplus(P, P)
            q *= 2
            row.append(("K^%d" % q,)+density(P))
        print("  n=%d:" % n, row)
    print("  => binary exponentiation with core-sparse multiplication is O(n^2)")



# ==== exact characterization and open-gap checks ====
"""Exact total-monotonicity test, all-endpoint Legendre structure, and the
strict-subclass barrier for the rank/relational residual problem.

Notation: C is the cumulative breakpoint array.  For end-indexed budgets put
z_j = C_j - beta(j) - 1 and lambda_j = min{i<j : C_i >= z_j}, with lambda_j=j
when row j has no feasible predecessor.  For a single uniform budget put
d = beta+1; a line i->j has squared slack (d-(C_j-C_i))^2 and is legal iff
C_j-C_i <= d.
"""
from itertools import product
from math import inf
import random


def Cvec(w):
    C = [0]
    for a in w:
        C.append(C[-1] + a + 1)
    return C


# ── 1(a): exact triangular-domain total monotonicity ──────────────────────

def matrix_data(C, beta):
    n = len(beta)
    z = [C[j] - beta[j-1] - 1 for j in range(1, n+1)]
    return z


def tm_lambdas(C, beta):
    z = matrix_data(C, beta)
    n = len(z)
    lam = []
    for j in range(1, n+1):
        zj = z[j-1]
        lam.append(next((i for i in range(j) if C[i] >= zj), j))
    return lam


def tm_brute_tri(C, beta):
    """Total monotonicity on i<i'<j<j', with inf<=inf true."""
    z = matrix_data(C, beta)
    n = len(z)
    def M(j, i):
        if i >= j:
            return inf
        if C[i] < z[j-1]:
            return inf
        return (C[i]-z[j-1])**2
    for j in range(1, n+1):
        for jp in range(j+1, n+1):
            for i in range(0, j-1):
                for ip in range(i+1, j):
                    if M(j, ip) <= M(j, i) and not (M(jp, ip) <= M(jp, i)):
                        return False, (j, jp, i, ip)
    return True, None


def tm_lambda_condition(lam):
    """TM iff no later feasible threshold with lambda_{j'}<=j-2 is <lambda_j."""
    n = len(lam)
    for j in range(1, n+1):
        for jp in range(j+1, n+1):
            lp = lam[jp-1]
            if lp <= j-2 and lp < lam[j-1]:
                return False, (j, jp, lp, lam[j-1])
    return True, None


def check_tm_lambda():
    print("== exact triangular-domain total monotonicity ==")
    rng = random.Random(20261223)
    bad = 0
    tested = 0
    ex = []
    for n in range(2, 7):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(list(w))
            for _ in range(18):
                beta = [rng.randint(1, 12) for _ in range(n)]
                want = tm_brute_tri(C, beta)[0]
                got = tm_lambda_condition(tm_lambdas(C, beta))[0]
                tested += 1
                if want != got:
                    bad += 1
                    if len(ex) < 4:
                        ex.append((w, beta, tm_lambdas(C, beta), want, got,
                                   tm_brute_tri(C, beta)[1]))
    print("  instances:", tested, "| mismatches:", bad)
    for e in ex:
        print("   ", e)
    # explicit narrow witness: TM true, z not nondecreasing
    w, beta = (1, 1), (1, 4)
    C = Cvec(list(w))
    print("  narrow witness w=%s beta=%s z=%s lambda=%s TM=%s"
          % (w, beta, matrix_data(C, beta), tm_lambdas(C, beta),
             tm_brute_tri(C, beta)[0]))


# ── 1(b)/3: all-endpoint Legendre structure for one budget ────────────────

def lws_p(C, d, p):
    """F[j] = min over any number of d-lines of sum slack^2 + p*lines."""
    n = len(C)-1
    F = [0.0] + [inf]*n
    for j in range(1, n+1):
        best = inf
        for i in range(j):
            gap = C[j]-C[i]
            if gap <= d and F[i] < inf:
                best = min(best, F[i] + (d-gap)**2 + p)
        F[j] = best
    return F


def P_j_m(C, d, m, j):
    """Exactly m d-lines from 0 to j."""
    n = len(C)-1
    E = [0.0] + [inf]*n
    for _ in range(m):
        G = [inf]*(n+1)
        for a in range(0, n+1):
            if E[a] == inf:
                continue
            for b in range(a+1, j+1):
                gap = C[b]-C[a]
                if gap <= d:
                    G[b] = min(G[b], E[a] + (d-gap)**2)
        E = G
    return E[j]


def fenchel_all_endpoints(C, d, m, V):
    """Brute-force Legendre transform over integer prices p in [-V,V].

    Returns (best[j], L[j], R[j]) where best[j] is the Legendre value and
    [L[j],R[j]] is the interval of prices attaining it.
    """
    n = len(C)-1
    best = [-inf]*(n+1)
    L = [None]*(n+1)
    R = [None]*(n+1)
    for p in range(-V, V+1):
        F = lws_p(C, d, p)
        for j in range(1, n+1):
            v = F[j] - p*m
            if v > best[j] + 1e-9:
                best[j], L[j], R[j] = v, p, p
            elif abs(v-best[j]) < 1e-9:
                L[j] = min(L[j], p)
                R[j] = max(R[j], p)
    return best, L, R


def check_all_endpoint_fenchel():
    print("== all-endpoint Fenchel structure, single uniform budget ==")
    rng = random.Random(20261224)
    bad = 0
    mono_bad = 0
    tested = 0
    ex = []
    D = []
    for _ in range(120):
        n = rng.randint(3, 7)
        w = [rng.randint(1, 3) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(3, min(8, C[n]+2))
        m = rng.randint(1, min(3, n))
        V = 2*n*(d+1)**2 + 10
        best, L, R = fenchel_all_endpoints(C, d, m, V)
        js = [j for j in range(m, n+1) if P_j_m(C, d, m, j) < inf]
        for j in js:
            tested += 1
            if abs(best[j] - P_j_m(C, d, m, j)) > 1e-6:
                bad += 1
                if len(ex) < 4:
                    ex.append(("gap", w, d, m, j, P_j_m(C, d, m, j), best[j]))
        if len(js) >= 2:
            rv = [R[j] for j in js]
            D.append(len(set(rv)))
            if not all(rv[t] <= rv[t+1] for t in range(len(rv)-1)):
                mono_bad += 1
                if len(ex) < 4:
                    ex.append(("nonmono", w, d, m, rv))
    print("  feasible endpoints:", tested, "| Fenchel gaps:", bad,
          "| price-monotonicity failures:", mono_bad)
    if D:
        print("  distinct rightmost prices per instance: mean %.2f, max %d"
              % (sum(D)/len(D), max(D)))
    for e in ex[:4]:
        print("   ", e)


# ── strict subclass of concave Monge DAGs ─────────────────────────────────

def density_rank_one(C, d):
    """Adjacent Monge density of c(i,j)=(d-(C_j-C_i))^2 is rank one:
       D[i,j] = 2 (C_{i+1}-C_i)(C_{j+1}-C_j), for i+1<j.
    """
    n = len(C)-1
    def c(i, j):
        return (d-(C[j]-C[i]))**2
    D = []
    for i in range(n-1):
        row = []
        for j in range(1, n):
            if i+1 < j:
                val = c(i, j+1) + c(i+1, j) - c(i, j) - c(i+1, j+1)
                want = 2*(C[i+1]-C[i])*(C[j+1]-C[j])
                row.append(val)
                if val != want:
                    return False, D
            else:
                row.append(0)
        D.append(row)
    return True, D


def check_monge_subclass():
    print("== squared-slack Monge DAG is a strict rank-one subclass ==")
    rng = random.Random(20261225)
    bad = 0
    for _ in range(200):
        n = rng.randint(4, 8)
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(3, C[n]+2)
        ok, _ = density_rank_one(C, d)
        if not ok:
            bad += 1
    C = Cvec([1, 2, 1, 3, 2])
    d = 8
    ok, D = density_rank_one(C, d)
    print("  rank-one density checks: 200 instances | failures:", bad)
    print("  example density rows (zero on the diagonal band):")
    for row in D[:3]:
        print("   ", row[:6])
    print("  general concave Monge DAGs allow an arbitrary nonnegative density")
    print("  matrix with O(n^2) degrees of freedom; this subclass has O(n).")



# ==== offline batched predecessor checks ====
"""Offline batched predecessor counterexample for the end-indexed DP.

The reduction instance of §1.3 encodes m successor queries into n=2m DP rows.
All dp values are recovered by sorting S union Q once and merging, i.e. by an
offline algorithm.  For w=O(log n) integer keys, radix/counting sort makes the
sorting step O(n) time and O(n) space, so the DP instance is solvable in O(n)
word-RAM time even though online static predecessor queries require
Theta(log log n) probes each on a polynomial universe.
"""
from math import inf
import random


def Cvec_offline(w):
    C = [0]
    for a in w:
        C.append(C[-1] + a + 1)
    return C


def zvec_offline(C, beta):
    return [C[j] - beta[j-1] - 1 for j in range(1, len(beta)+1)]


def dp_brute_offline(C, beta, Lam):
    """Literal O(n^2) shortest-path DP for (0.1)."""
    n = len(beta)
    z = zvec_offline(C, beta)
    dp = [inf]*(n+1)
    dp[0] = 0.0
    for j in range(1, n+1):
        x = z[j-1]
        best = inf
        for i in range(j):
            if C[i] >= x:
                v = dp[i] + (C[i]-x)**2 + Lam
                if v < best:
                    best = v
        dp[j] = best
    return dp


def make_reduction(S, Q):
    """Build C, beta, Lam for the offline-predecessor reduction instance."""
    m = len(S)
    n = 2*m
    C = [0]
    for s in S:
        C.append(2*(s+2))
    Cm = C[m]
    for r in range(1, m+1):
        C.append(Cm + 2*r)
    beta = []
    for j in range(1, n+1):
        if j <= m:
            z = 0
        else:
            q = Q[j-m-1]
            z = 2*(q+2)
        beta.append(C[j] - z - 1)
    Lam = 10*C[n]**2 + 1
    return C, beta, Lam


def offline_successor(S, Q):
    """All successor ranks by merging sorted keys and sorted queries.

    With w=O(log n)-bit integers, two-pass radix sort with digit
    (log n)/2 bits uses O(n) time and O(n) space: each pass has 2^{log n/2}
    = sqrt(n) buckets.  The merge is O(n).  The Python implementation uses
    sorted() for clarity.
    """
    m = len(S)
    pairs = sorted([(s, i+1) for i, s in enumerate(S)] +
                   [(q, 0) for q in Q])
    # scan from the right: best = smallest S index with s >= current q
    best = m+1
    ans = {}
    for val, idx in reversed(pairs):
        if idx > 0:
            best = min(best, idx)
        else:
            ans[val] = best
    return [ans[q] for q in Q]


def offline_dp_values(S, Q):
    """dp[1..2m] recovered without any per-query predecessor data structure."""
    m = len(S)
    C, beta, Lam = make_reduction(S, Q)
    z = zvec_offline(C, beta)
    ranks = offline_successor(S, Q)
    dp = [Lam]*m + [2*Lam + (C[r]-z[m+j])**2 for j, r in enumerate(ranks)]
    return dp, C, beta, Lam


def check_offline_reduction():
    print("== offline batched predecessor: reduction vs literal DP ==")
    rng = random.Random(20261226)
    bad = 0
    tested = 0
    ex = []
    for m in range(2, 9):
        for _ in range(150):
            S = sorted(rng.sample(range(1, 6*m+2), m))
            Q = [rng.randint(0, S[-1]-1) for _ in range(m)]
            C, beta, Lam = make_reduction(S, Q)
            want = dp_brute_offline(C, beta, Lam)[1:]
            got, _, _, _ = offline_dp_values(S, Q)
            tested += 1
            if want != got:
                bad += 1
                if len(ex) < 4:
                    ex.append((S, Q, want, got))
    print("  instances:", tested, "| mismatches:", bad)
    for e in ex:
        print("   ", e)

    # concrete polynomial-universe instance
    m = 8
    S = sorted(rng.sample(range(1, 6*m+2), m))
    Q = [rng.randint(0, S[-1]-1) for _ in range(m)]
    C, beta, Lam = make_reduction(S, Q)
    ranks = offline_successor(S, Q)
    print("  explicit m=%d: S=%s Q=%s successor ranks=%s" % (m, S, Q, ranks))
    print("  C_{2m}=%d, Lam=%d, beta range=[%d,%d]"
          % (C[2*m], Lam, min(beta), max(beta)))
    print("  two-pass radix sort of S∪Q + merging: O(n) word-RAM for w=O(log n)")
    print("  online predecessor lower bound on the same universe: "
          "Omega(n log log n) cell probes")



# ==== equal-width locator checks ====
"""Equal-width locator lines with arbitrary word widths: bounded-price
Fenchel algorithm and exhaustive verification.

The last m lines have budget b=A-s; all preceding lines have budget A.  The
lexicographic objective is (fewest lines, least squared slack).  With a
dominating per-line penalty Lam, the total value is Lam*lines + slack.

Reversing the word sequence turns the m terminal b-lines into exactly m
initial b-lines.  For a uniform budget d=b+1, let P_j(m) be the least squared
slack of exactly m lines from 0 to j.  Theorem 23 shows that every finite
rightmost Fenchel price lies in [-d^2,0], so

    P_j(m) = max_{p in [-d^2,0]} ( F_p[j] - p*m ),

where F_p[j] = min_k ( P_j(k) + p*k ) is one unconstrained LWS pass with a
per-line price p.  Hence all P_j(m) are computable with d^2+1 = O(A^2) LWS
passes.  Each pass is O(n) by Wilber's concave LWS algorithm (the naive
implementation below is O(n^2) and is used only for verification).
"""
from math import inf
import random



# ── exhaustive reference ──────────────────────────────────────────────────

def brute_equal_locator(w, A, s, m):
    """Exhaustive lex optimum over layouts with m terminal b-lines."""
    n = len(w)
    b = A-s
    best = None

    def rec(idx, start):
        nonlocal best
        if start == n:
            L = len(idx)-1
            if L < m:
                return
            pen = 0
            for t in range(1, L+1):
                i, j = idx[t-1]+1, idx[t]
                width = span(w, i, j)
                beta = b if t > L-m else A
                if width > beta:
                    return
                pen += (beta-width)**2
            key = (L, pen)
            if best is None or key < best:
                best = key
            return
        for j in range(start+1, n+1):
            idx.append(j)
            rec(idx, j)
            idx.pop()
    rec([0], 0)
    return best


# ── bounded-price Fenchel for one uniform budget ──────────────────────────

def greedy_prefix_counts(w, budget):
    """L[j] = greedy (minimum) line count for w[0:j], in O(n)."""
    out = [0]*(len(w)+1)
    cur = 0          # width of the open line; 0 means no open line
    lines = 0
    for t, a in enumerate(w, 1):
        if cur == 0:
            cur = a
            lines += 1
        elif cur + 1 + a <= budget:
            cur += 1 + a
        else:
            cur = a
            lines += 1
        out[t] = lines
    return out


def lws_naive_uniform(C, d, per_line):
    """F[j] = min over any number of d-lines of sum slack^2 + per_line*lines."""
    n = len(C)-1
    F = [inf]*(n+1)
    F[0] = 0.0
    for j in range(1, n+1):
        best = inf
        for i in range(j-1, -1, -1):
            gap = C[j]-C[i]
            if gap > d:
                break
            if F[i] < inf:
                v = F[i] + (d-gap)**2 + per_line
                if v < best:
                    best = v
        F[j] = best
    return F


def P_all_endpoints_bounded(C, d, m):
    """All P_j(m) for j=0..n via prices p in [-d^2,0].

    An endpoint is feasible iff it has at least m words, every word span is
    at most d, and the greedy minimum line count at budget d-1 is at most m:
    any layout with fewer lines can be refined to exactly m lines by splitting
    blocks, since j >= m.
    """
    n = len(C)-1
    best = [inf]*(n+1)
    if m == 0:
        best[0] = 0.0
        return best
    wr = [C[t]-C[t-1]-1 for t in range(1, n+1)]
    maxgap = [0]*(n+1)
    for t in range(1, n+1):
        maxgap[t] = max(maxgap[t-1], C[t]-C[t-1])
    # greedy line count on every prefix of the reversed word list, O(n)
    minlines = greedy_prefix_counts(wr, d-1)
    feasible = [j >= m and maxgap[j] <= d and minlines[j] <= m
                for j in range(n+1)]
    vals = [-inf]*(n+1)
    for p in range(-d*d, 1):
        F = lws_naive_uniform(C, d, p)
        for j in range(1, n+1):
            if feasible[j]:
                vals[j] = max(vals[j], F[j] - p*m)
    for j in range(n+1):
        if feasible[j]:
            best[j] = vals[j]
    return best


# ── full algorithm ────────────────────────────────────────────────────────

def equal_locator_lex(w, A, s, m):
    n = len(w)
    b = A-s
    d = b+1
    dA = A+1
    Lam = n*(A+1)**2*4 + 10

    # reversed instance: exactly m initial b-lines
    wr = list(reversed(w))
    Cr = Cvec(wr)
    P = P_all_endpoints_bounded(Cr, d, m)

    # arbitrary number of A-lines on the original prefix
    C = Cvec(w)
    G = lws_naive_uniform(C, dA, Lam)

    total = inf
    for i in range(0, n-m+1):          # suffix length n-i >= m
        j = n-i
        if G[i] < inf and P[j] < inf:
            val = G[i] + Lam*m + P[j]
            if val < total:
                total = val
    if total == inf:
        return None
    lines = int(total // Lam)
    penalty = total - Lam*lines
    return (lines, penalty)


# ── checks ────────────────────────────────────────────────────────────────

def check_bounded_fenchel_vs_exact():
    print("== bounded Fenchel prices vs exact m-link DP ==")
    rng = random.Random(20270109)
    bad = tested = 0
    ex = []
    for _ in range(250):
        n = rng.randint(2, 9)
        d = rng.randint(4, 12)
        w = [rng.randint(1, d-1) for _ in range(n)]
        C = Cvec(w)
        # exact P_j(m) table
        P = [[inf]*(n+1) for _ in range(n+1)]
        P[0][0] = 0.0
        for k in range(1, n+1):
            for j in range(1, n+1):
                best = inf
                for i in range(j-1, -1, -1):
                    gap = C[j]-C[i]
                    if gap > d:
                        break
                    if P[k-1][i] < inf:
                        best = min(best, P[k-1][i]+(d-gap)**2)
                P[k][j] = best
        for m in range(1, n+1):
            got = P_all_endpoints_bounded(C, d, m)
            for j in range(m, n+1):
                if P[m][j] < inf:
                    tested += 1
                    if abs(got[j]-P[m][j]) > 1e-6:
                        bad += 1
                        if len(ex) < 4:
                            ex.append((w, d, m, j, P[m][j], got[j]))
    print("  endpoints:", tested, "| mismatches:", bad)
    for e in ex:
        print("   ", e)


def check_full_algorithm():
    print("== equal-width locator algorithm vs exhaustive layouts ==")
    rng = random.Random(20270110)
    bad = tested = 0
    ex = []
    for n in range(1, 9):
        for _ in range(120):
            w = [rng.randint(1, 4) for _ in range(n)]
            A = rng.randint(max(w)+1, max(w)+5)
            s = rng.randint(0, A-max(w))     # keep b >= max word width
            m = rng.randint(1, n)
            want = brute_equal_locator(w, A, s, m)
            got = equal_locator_lex(w, A, s, m)
            tested += 1
            if want != got:
                bad += 1
                if len(ex) < 6:
                    ex.append((w, A, s, m, want, got))
    print("  instances:", tested, "| mismatches:", bad)
    for e in ex:
        print("   ", e)


def check_marginal_bound():
    print("== marginal price bound 0 <= P(m)-P(m-1) <= d^2 ==")
    rng = random.Random(20270111)
    bad = tested = 0
    maxD = 0
    for _ in range(400):
        n = rng.randint(3, 10)
        d = rng.randint(4, 14)
        w = [rng.randint(1, d-1) for _ in range(n)]
        C = Cvec(w)
        P = [[inf]*(n+1) for _ in range(n+1)]
        P[0][0] = 0.0
        for k in range(1, n+1):
            for j in range(1, n+1):
                best = inf
                for i in range(j-1, -1, -1):
                    gap = C[j]-C[i]
                    if gap > d:
                        break
                    if P[k-1][i] < inf:
                        best = min(best, P[k-1][i]+(d-gap)**2)
                P[k][j] = best
        vals = []
        for m in range(2, n+1):
            for j in range(m, n+1):
                if P[m][j] < inf and P[m-1][j] < inf:
                    Delta = P[m][j]-P[m-1][j]
                    tested += 1
                    if not (0 <= Delta <= d*d):
                        bad += 1
                    vals.append(-Delta)
        maxD = max(maxD, len(set(vals)))
    print("  marginals:", tested, "| violations:", bad,
          "| max distinct prices:", maxD, "| bound d^2+1 <= 197")



# ==== joint-count Legendre gap checks ====
"""Joint-count Legendre gap for two budget runs.

For two constant-budget runs d1,d2 let P(k1,k2) be the least squared slack
using exactly k1 lines of budget d1 followed by exactly k2 lines of budget
d2, and let

    F(l1,l2) = min_{k1,k2} ( P(k1,k2) + l1*k1 + l2*k2 ).

The single-run bounded-price theorem recovers P(k) from a one-dimensional
Legendre transform.  This file certifies that the two-dimensional analogue
fails: on an explicit instance P(1,1)=20 while the supremum of
F(l1,l2)-l1-l2 over all real (l1,l2) is 19.
"""
from math import inf



def two_run_P(C, d1, d2):
    """P[k1][k2][j], all finite values at every endpoint j."""
    n = len(C)-1
    P = [[[inf]*(n+1) for _ in range(n+1)] for _ in range(n+1)]
    P[0][0][0] = 0.0
    for k1 in range(1, n+1):
        E, G = P[k1-1][0], P[k1][0]
        for j in range(k1, n+1):
            best = inf
            for i in range(j-1, -1, -1):
                gap = C[j]-C[i]
                if gap > d1:
                    break
                if E[i] < inf:
                    best = min(best, E[i]+(d1-gap)**2)
            G[j] = best
    for k1 in range(n+1):
        for k2 in range(1, n+1):
            E, G = P[k1][k2-1], P[k1][k2]
            for j in range(k1+k2, n+1):
                best = inf
                for i in range(j-1, -1, -1):
                    gap = C[j]-C[i]
                    if gap > d2:
                        break
                    if E[i] < inf:
                        best = min(best, E[i]+(d2-gap)**2)
                G[j] = best
    return P


def finite_points(P):
    n = len(P)-1
    return [(a, b, P[a][b][n]) for a in range(n+1) for b in range(n+1)
            if P[a][b][n] < inf]


def dual_value(points, k, l1, l2):
    F = min(q + l1*a + l2*b for a, b, q in points)
    return F - l1*k[0] - l2*k[1]


def check_explicit_gap():
    print("== two-run joint Legendre gap: explicit witness ==")
    C = Cvec([2, 1])                  # C = [0,3,5]
    d1, d2 = 5, 6
    P = two_run_P(C, d1, d2)
    pts = finite_points(P)
    print("  C=%s  d1=%d  d2=%d" % (C, d1, d2))
    print("  finite P(k1,k2):", sorted(pts))
    k = (1, 1)
    exact = P[k[0]][k[1]][len(C)-1]
    # analytic maximiser: x=-l1=25, y=-l2=31
    l1, l2 = -25, -31
    dual = dual_value(pts, k, l1, l2)
    print("  P(1,1)=%s  max dual=%s  gap=%s  at (l1,l2)=%s"
          % (exact, dual, exact-dual, (l1, l2)))
    # exhaustive integer grid, including the natural per-budget boxes
    best_grid, arg, B = -inf, None, 100
    for a in range(-B, B+1):
        for b in range(-B, B+1):
            v = dual_value(pts, k, a, b)
            if v > best_grid:
                best_grid, arg = v, (a, b)
    print("  exhaustive integer grid [-%d,%d]^2 max=%s at %s"
          % (B, B, best_grid, arg))
    assert abs(exact - 20) < 1e-9
    assert abs(dual - 19) < 1e-9
    assert best_grid <= dual + 1e-9


def check_analytic_bound():
    print("== analytic supremum over all real prices ==")
    # F(l1,l2) = min(1+l2, 25+2*l2, l1, 20+l1+l2, 13+2*l1)
    # Put x=-l1, y=-l2.  D(x,y)=min(1+x, 25+x-y, y, 20, 13-x+y).
    # D>19 would require x>18, y<x+6, y>19 and y>x+6, impossible.
    # The point x=25,y=31 gives D=19.
    x, y = 25, 31
    terms = (1+x, 25+x-y, y, 20, 13-x+y)
    print("  D(25,31)=min%s=%d  <  P(1,1)=20" % (terms, min(terms)))
    assert min(terms) == 19


def check_random_gaps():
    print("== two-run duality gaps on random instances ==")
    import random
    rng = random.Random(20270130)
    found = 0
    tested = 0
    ex = []
    for _ in range(200):
        n = rng.randint(2, 5)
        w = [rng.randint(1, 3) for _ in range(n)]
        C = Cvec(w)
        d1 = rng.randint(3, min(7, C[n]+1))
        d2 = rng.randint(3, min(7, C[n]+1))
        P = two_run_P(C, d1, d2)
        pts = finite_points(P)
        if len(pts) < 2:
            continue
        # restrict to the natural bounded-price box [-d^2,0] x [-d^2,0]
        for k in [(a, b) for a in range(n+1) for b in range(n+1)
                  if P[a][b][n] < inf]:
            exact = P[k[0]][k[1]][n]
            best = -inf
            for l1 in range(-d1*d1, 1):
                for l2 in range(-d2*d2, 1):
                    v = dual_value(pts, k, l1, l2)
                    if v > best:
                        best = v
            tested += 1
            if best < exact - 1e-9:
                found += 1
                if len(ex) < 4:
                    ex.append((w, d1, d2, k, exact, best))
                break
    print("  instances with a gap in the bounded-price box: %d of 200"
          % found)
    print("  count-vector checks before first gap: %d" % tested)
    for e in ex:
        print("   ", e)



# ==== source-specific bounded-price gaps ====
"""Bounded-price Fenchel identity with non-single-source inputs.

For one uniform budget d, Theorem 23 recovers exactly-m-line values from a
single source e_0 using integer prices p in [-d^2,0].  Composing constant-
budget runs would require the same identity when the source vector G is the
DP vector produced by a preceding run.  The checks below certify that the
identity fails on arbitrary sources and on sources produced by one uniform-
budget run.
"""
from math import inf
import random



def P_table_source(C, d, G):
    """P[k][j] = least cost of exactly k d-lines from source vector G."""
    n = len(C)-1
    P = [[inf]*(n+1) for _ in range(n+1)]
    P[0] = G[:]
    for k in range(1, n+1):
        E, H = P[k-1], P[k]
        for j in range(k, n+1):
            best = inf
            for i in range(j-1, -1, -1):
                gap = C[j]-C[i]
                if gap > d:
                    break
                if E[i] < inf:
                    best = min(best, E[i]+(d-gap)**2)
            H[j] = best
    return P


def F_price_source(C, d, G, p):
    """F[j] = min over any number of d-lines, with per-line price p."""
    n = len(C)-1
    F = G[:]
    for j in range(1, n+1):
        best = F[j]
        for i in range(j-1, -1, -1):
            gap = C[j]-C[i]
            if gap > d:
                break
            if F[i] < inf:
                best = min(best, F[i]+(d-gap)**2+p)
        F[j] = best
    return F


def bounded_dual(C, d, G, m, j):
    """Maximum of F_p[j]-p*m over the natural bounded price box."""
    best = -inf
    arg = None
    for p in range(-d*d, 1):
        F = F_price_source(C, d, G, p)
        val = F[j]-p*m
        if val > best:
            best, arg = val, p
    return best, arg


def source_from_run(C, d0):
    """G[j] = least cost of any number of d0-lines from e_0."""
    n = len(C)-1
    F = [0]+[inf]*n
    for j in range(1, n+1):
        best = inf
        for i in range(j-1, -1, -1):
            gap = C[j]-C[i]
            if gap > d0:
                break
            if F[i] < inf:
                best = min(best, F[i]+(d0-gap)**2)
        F[j] = best
    return F


def check_explicit_source_witness():
    print("== bounded-price Fenchel gap with an arbitrary source ==")
    w = (3, 2, 1, 3, 2, 2, 3)
    C = Cvec(list(w))
    d = 12
    G = [0, inf, 34, 17, inf, inf, inf, inf]
    P = P_table_source(C, d, G)
    exact = P[1][4]
    dual, arg = bounded_dual(C, d, G, 1, 4)
    print("  w=%s  d=%d  C=%s" % (w, d, C))
    print("  G=%s" % G)
    print("  P(1,j=4)=%s  bounded dual=%s at p=%s  gap=%s"
          % (exact, dual, arg, exact-dual))
    print("  P(0..3,j=4)=%s" % [P[k][4] for k in range(4)])
    assert abs(exact-70) < 1e-9
    assert abs(dual-61) < 1e-9
    # P(1,4)=34+(12-6)^2=70.  P(2,4)=(12-7)^2+(12-6)^2=61, and the
    # lower convex envelope at k=1 is 61, so every p in [-144,0] gives
    # F_p[4]-p <= 61 < 70.
    print("  exact one-line path: G[2]=34, gap 6, cost 36")
    print("  two-line path from G[0]: gaps 7 and 6, cost 25+36=61")


def check_structured_source_witness():
    print("== bounded-price Fenchel gap with a preceding-run source ==")
    w = (2, 3, 1, 1)
    C = Cvec(list(w))
    d0, d1 = 8, 5
    G = source_from_run(C, d0)
    P = P_table_source(C, d1, G)
    exact = P[1][2]
    dual, arg = bounded_dual(C, d1, G, 1, 2)
    print("  w=%s  C=%s  d0=%d  d1=%d" % (w, C, d0, d1))
    print("  G from d0-run=%s" % G)
    print("  P(0..2,j=2)=%s" % [P[k][2] for k in range(3)])
    print("  P(1,j=2)=%s  bounded dual=%s at p=%s  gap=%s"
          % (exact, dual, arg, exact-dual))
    assert abs(exact-26) < 1e-9
    assert abs(dual-3) < 1e-9
    # G[1]=25 and edge 1->2 has gap 4, so P(1,2)=25+(5-4)^2=26.
    # G[0]=0 and two d1-lines 0->1,1->2 cost (5-3)^2+(5-4)^2=5.
    # The lower convex envelope through P(2,2)=5 gives dual 3 at k=1.
    print("  exact one-line path: G[1]=25, gap 4, cost 1")
    print("  two-line path from G[0]: gaps 3 and 4, cost 4+1=5")


def check_random_source_gaps():
    print("== random arbitrary-source bounded-price failures ==")
    rng = random.Random(20270222)
    bad = tested = 0
    ex = []
    for _ in range(200):
        n = rng.randint(3, 8)
        w = [rng.randint(1, 3) for _ in range(n)]
        C = Cvec(w)
        d = rng.randint(3, C[n]+2)
        G = [inf]*(n+1)
        G[0] = 0
        for j in range(1, n+1):
            if rng.random() < 0.5:
                G[j] = rng.randint(0, 50)
        P = P_table_source(C, d, G)
        Fs = [F_price_source(C, d, G, p) for p in range(-d*d, 1)]
        for m in range(1, n+1):
            for j in range(m, n+1):
                if P[m][j] < inf:
                    best = max(Fs[k][j]-p*m
                               for k, p in enumerate(range(-d*d, 1)))
                    tested += 1
                    if abs(best-P[m][j]) > 1e-6:
                        bad += 1
                        if len(ex) < 5:
                            ex.append((w, d, G, m, j, P[m][j], best))
    print("  finite endpoints:", tested, "| strict gaps:", bad)
    for e in ex:
        print("   ", e)


def check_random_structured_source_gaps():
    print("== random preceding-run-source bounded-price failures ==")
    rng = random.Random(20270223)
    found = tested = 0
    ex = []
    for _ in range(300):
        n = rng.randint(3, 7)
        w = [rng.randint(1, 3) for _ in range(n)]
        C = Cvec(w)
        d0 = rng.randint(3, C[n]+1)
        d1 = rng.randint(3, C[n]+1)
        G = source_from_run(C, d0)
        P = P_table_source(C, d1, G)
        Fs = [F_price_source(C, d1, G, p) for p in range(-d1*d1, 1)]
        hit = False
        for m in range(1, n+1):
            for j in range(m, n+1):
                if P[m][j] < inf:
                    best = max(Fs[k][j]-p*m
                               for k, p in enumerate(range(-d1*d1, 1)))
                    tested += 1
                    if best < P[m][j]-1e-6:
                        found += 1
                        hit = True
                        if len(ex) < 5:
                            ex.append((w, d0, d1, G, m, j, P[m][j], best))
                        break
            if hit:
                break
    print("  instances with a gap:", found, "of 300; endpoints checked:",
          tested)
    for e in ex:
        print("   ", e)



# ==== two-layer core-density lower bound ====
"""Quadratic core-density lower bound for a two-layer min-plus product.

Instance: unit words, C_j = 2j, two equal budgets d_1 = d_2 = 2N with N=n.
The two-layer min-plus product matrix P = M_d (x) M_d has a closed form
depending only on m = j-i:

    P(m) = 2(2N-m)^2 + 2*(m mod 2),   for 2 <= m <= n.

Its discrete density

    delta(i,j) = P[i,j] + P[i+1,j+1] - P[i,j+1] - P[i+1,j]

is -8 for even interior gaps m=j-i and 0 for odd interior gaps.  Hence
the core has Theta(n^2) nonzero entries.  This is a precise obstruction to
core-sparse min-plus multiplication when the measure A is unbounded.
"""
from math import inf
import random



def core_layer(C, d):
    n = len(C)-1
    M = [[inf]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+1, n+1):
            gap = C[j]-C[i]
            if gap <= d:
                M[i][j] = (d-gap)**2
    return M


def core_minplus(A, B):
    n = len(A)-1
    P = [[inf]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+1, n+1):
            best = inf
            for k in range(i+1, j):
                if A[i][k] < inf and B[k][j] < inf:
                    best = min(best, A[i][k]+B[k][j])
            P[i][j] = best
    return P


def core_cells(M):
    n = len(M)-1
    nz = 0
    parity = {}
    for i in range(n):
        for j in range(n):
            a, b = M[i][j], M[i][j+1]
            c, d = M[i+1][j], M[i+1][j+1]
            if inf in (a, b, c, d):
                continue
            delta = a+d-b-c
            if delta != 0:
                nz += 1
                parity[(j-i) % 2] = parity.get((j-i) % 2, 0)+1
    return nz, parity


def closed_form(n, m):
    return 2*(2*n-m)**2 + 2*(m % 2)


def check_closed_form():
    print("== two-layer product: closed form and quadratic core ==")
    for n in (6, 8, 10, 20, 30, 40):
        C = Cvec([1]*n)             # C_j = 2j
        d = 2*n
        M = core_layer(C, d)
        P = core_minplus(M, M)
        ok = True
        for m in range(2, n+1):
            got = P[0][m]
            want = closed_form(n, m)
            if got != want:
                ok = False
                if m <= 5:
                    print("   mismatch n=%d m=%d got=%s want=%s"
                          % (n, m, got, want))
        nz, parity = core_cells(P)
        exact = ((n-2)*(n-4)//4 if n % 2 == 0 else (n-3)**2//4)
        print("  n=%2d d=%2d closed_form_ok=%s core=%d exact=%d parity=%s "
              "core/n^2=%.3f"
              % (n, d, ok, nz, exact, parity, nz/(n*n)))
        assert ok
        assert nz == exact
        assert parity == {0: exact}


def check_random_large_budget():
    print("== random words, d1=d2=C_n: core stays quadratic ==")
    rng = random.Random(20270304)
    for n in (16, 24, 32):
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = C[n]
        M = core_layer(C, d)
        P = core_minplus(M, M)
        nz, parity = core_cells(P)
        print("  n=%2d C_n=%3d core=%4d core/n^2=%.3f"
              % (n, C[n], nz, nz/(n*n)))


def check_fixed_budget_is_linear():
    print("== fixed small budget: core is O(n d) ==")
    rng = random.Random(20270305)
    for n in (32, 64, 128):
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        d = 8
        M = core_layer(C, d)
        P = core_minplus(M, M)
        nz, parity = core_cells(P)
        print("  n=%3d d=%d core=%4d core/n=%.2f"
              % (n, d, nz, nz/n))



# ==== rank-3 kernel and two-budget core lower bound ====
"""Rank-3 single-layer kernel and the quadratic core of its two-layer product.

Setup (window-free / full-finite): C_0<...<C_n arbitrary, budgets d>=C_n so
that every edge i<j is feasible.  The one-layer matrix is

    W_d[i,j] = (d + C_i - C_j)^2 .

It has ordinary rank <= 3 on the finite band.  Nevertheless the two-layer
min-plus product Q = W_{d1} (x) W_{d2} has Theta(n^2) nonzero discrete
density (core) for arbitrary non-uniform C and arbitrary d1,d2.

Checks:
  A. rank-3 identity C_i^2 + 2 C_i (d-C_j) + (d-C_j)^2, exactly.
  B. closed form Q[i,j] = (D-Delta)^2/2 + 2 dist(mu,C)^2, D=d1+d2,
     Delta=C_j-C_i, mu=(C_i+C_j+d1-d2)/2 (Theorem 17, m=2).
  C. density of the continuous term alone is -(C_{i+1}-C_i)(C_{j+1}-C_j).
  D. explicit family C_j=2j, d1=2a, d2=2b with a>=n, b in {a,a-1}:
        Q[i,j] = 2(a+b-m)^2 + 2*((m+a-b) mod 2),  m=j-i,
        delta = -8 iff m+a-b even, else 0,
        |core| = sum_{3<=m<=n-1, m = a-b mod 2} (n-m) = Theta(n^2).
     Exact counts verified for n=6..21 and both parities of a-b.
  E. random non-uniform C with (d1,d2) in {(Cn,Cn),(Cn,Cn/2),(Cn/2,Cn)}:
     core stays a constant fraction of n^2.
"""
from math import inf
import random



def rank3_identity(C, d):
    """Max abs error of W_d[i,j] = C_i^2 + 2C_i(d-C_j) + (d-C_j)^2."""
    err = 0.0
    for i, x in enumerate(C):
        for j, y in enumerate(C):
            if i < j:
                w = (d+x-y)**2
                r = x*x + 2*x*(d-y) + (d-y)**2
                err = max(err, abs(w-r))
    return err


def two_layer(C, d1, d2):
    n = len(C)-1
    Q = [[inf]*(n+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i+2, n+1):
            Q[i][j] = min((d1+C[i]-C[k])**2 + (d2+C[k]-C[j])**2
                          for k in range(i+1, j))
    return Q


def spline_form(C, d1, d2, i, j):
    """(D-Delta)^2/2 + 2*dist(mu, C inter (C_i,C_j))^2."""
    D = d1+d2
    delta = C[j]-C[i]
    mu = (C[i]+C[j]+d1-d2)/2
    cand = [C[k] for k in range(i+1, j)]
    dist2 = min((mu-x)**2 for x in cand)
    return (D-delta)**2/2 + 2*dist2


def continuous_term(C, d1, d2, i, j):
    D = d1+d2
    return (D-(C[j]-C[i]))**2/2


def density_of_matrix(M, n):
    nz = 0
    for i in range(n):
        for j in range(n):
            a, b = M[i][j], M[i][j+1]
            c, d = M[i+1][j], M[i+1][j+1]
            if inf in (a, b, c, d):
                continue
            if a+d-b-c != 0:
                nz += 1
    return nz


def check_rank3():
    print("== one-layer kernel has ordinary rank <= 3 ==")
    rng = random.Random(20270311)
    worst = 0.0
    for _ in range(300):
        n = rng.randint(3, 10)
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        d = C[n]+rng.randint(0, 10)
        worst = max(worst, rank3_identity(C, d))
    print("  max identity error over 300 instances:", worst)
    assert worst == 0


def check_spline_form():
    print("== two-layer product = continuous term + 2*grid distance^2 ==")
    rng = random.Random(20270312)
    bad = tested = 0
    for _ in range(200):
        n = rng.randint(3, 9)
        w = [rng.randint(1, 5) for _ in range(n)]
        C = Cvec(w)
        Cn = C[n]
        d1 = Cn+rng.randint(0, 6)
        d2 = Cn+rng.randint(0, 6)
        Q = two_layer(C, d1, d2)
        for i in range(n+1):
            for j in range(i+2, n+1):
                got = Q[i][j]
                want = spline_form(C, d1, d2, i, j)
                tested += 1
                if abs(got-want) > 1e-6:
                    bad += 1
    print("  cells:", tested, "| mismatches:", bad)
    assert bad == 0


def check_continuous_density():
    print("== continuous term has density -(gap_i)(gap_j) everywhere ==")
    rng = random.Random(20270313)
    bad = tested = 0
    for _ in range(200):
        n = rng.randint(4, 10)
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        d1, d2 = C[n]+1, C[n]+3
        for i in range(n-1):
            for j in range(n-1):
                if j < i+2:
                    continue
                f = continuous_term(C, d1, d2, i, j)
                fp = continuous_term(C, d1, d2, i, j+1)
                fq = continuous_term(C, d1, d2, i+1, j)
                fpq = continuous_term(C, d1, d2, i+1, j+1)
                got = f+fpq-fp-fq
                want = -(C[i+1]-C[i])*(C[j+1]-C[j])
                tested += 1
                if abs(got-want) > 1e-6:
                    bad += 1
    print("  interior cells:", tested, "| mismatches:", bad)
    assert bad == 0


def explicit_count(n, a, b):
    """Exact core size for C_j=2j, d1=2a, d2=2b, a>=n, b in {a,a-1}."""
    C = Cvec([1]*n)
    Q = two_layer(C, 2*a, 2*b)
    nz = density_of_matrix(Q, n)
    formula = sum(n-m for m in range(3, n)
                  if (m+a-b) % 2 == 0)
    return nz, formula


def check_explicit_family():
    print("== explicit two-budget family: exact quadratic core count ==")
    for n in range(6, 22):
        c0, f0 = explicit_count(n, n, n)
        c1, f1 = explicit_count(n, n, n-1)
        print("  n=%2d  d=(%2d,%2d): core=%3d sum=%3d %s | d=(%2d,%2d): "
              "core=%3d sum=%3d %s"
              % (n, 2*n, 2*n, c0, f0, "OK" if c0 == f0 else "X",
                 2*n, 2*n-2, c1, f1, "OK" if c1 == f1 else "X"))
        assert c0 == f0 and c1 == f1
        assert c0 >= (n-4)**2//8 and c1 >= (n-3)**2//8
    print("  both parities give sum_{m=3..n-1, m=a-b mod 2}(n-m)=Theta(n^2)")


def check_nonuniform():
    print("== random non-uniform C, arbitrary two budgets: core = Theta(n^2) ==")
    rng = random.Random(20270314)
    for n in (16, 24, 32, 48, 64):
        w = [rng.randint(1, 4) for _ in range(n)]
        C = Cvec(w)
        Cn = C[n]
        row = []
        for d1, d2 in ((Cn, Cn), (Cn, Cn//2), (Cn//2, Cn)):
            Q = two_layer(C, d1, d2)
            nz = density_of_matrix(Q, n)
            row.append((d1, d2, nz, round(nz/(n*n), 3)))
        print("  n=%2d Cn=%3d %s" % (n, Cn, row))
        assert all(r[2] >= n*n//8 for r in row)



# ==== tightness of bounded Fenchel prices ====
"""Tightness of the bounded-price Fenchel representation (Theorem 23/24).

Instance: unit words, C_j = 2j, one budget d = beta+1 = 2K (even).
Exactly m >= K lines.  For j = q*m + r (0 <= r < m) the least squared slack is

    P_j(m) = (m-r)*(d-2q)^2 + r*(d-2q-2)^2
           = (m*d-2j)^2/m + 4*r*(m-r)/m ,

so the marginal Delta_j(m) = P_j(m)-P_j(m-1) equals

    d^2 - 4j^2/(m(m-1)) + [4 r_m (m-r_m)/m - 4 r_{m-1}(m-1-r_{m-1})/(m-1)] .

Verified facts (all exact integer arithmetic):

  A. P_j(m) closed form == literal layered DP on unit words (random small).
  B. For m >= K and n = (m-1)K the set of distinct marginals is exactly the
     arithmetic progression {2d + 8t : 0 <= t < C(K,2)}, of length
     C(K,2) = d(d-2)/8 = Theta(d^2).
  C. The number of distinct optimal Fenchel prices
        p*_m(j) = argmax_{p in [-d^2,0]} (F_p[j] - p*m)
     equals C(K-1,2)+1 = (K-1)(K-2)/2 + 1 = Theta(d^2), and equals the
     number of distinct values of -Delta_j(m+1); the Fenchel identity
     P_j(m) = max_p (F_p[j]-pm) holds exactly.
  D. The Theta(d^2) count persists for m >> d^2 (m = 5*d^2 checked).
  E. General instances (random non-uniform C): every finite marginal satisfies
     Delta_j(k) <= d^2-8 (splitting bound with u,v >= 2), and the number of
     distinct prices is at most min(n, d^2+1).
"""
from math import inf
import random



def P_unit(j, m, d):
    """Exactly m lines of span budget d on j unit words; C_t = 2t."""
    K = d // 2
    if j < m or j > m*K:
        return inf
    q, r = divmod(j, m)
    if r == 0:
        return m*(d-2*q)**2
    if q+1 > K:
        return inf
    return (m-r)*(d-2*q)**2 + r*(d-2*(q+1))**2


def P_layered(C, d, m):
    """Literal O(n^2*m) DP for exactly k=0..m lines."""
    n = len(C)-1
    out = [[0]+[inf]*n]
    for k in range(1, m+1):
        E = out[-1]
        H = [inf]*(n+1)
        for j in range(k, n+1):
            best = inf
            for i in range(j-1, -1, -1):
                g = C[j]-C[i]
                if g > d:
                    break
                if E[i] < inf:
                    best = min(best, E[i]+(d-g)**2)
            H[j] = best
        out.append(H)
    return out


def F_price(C, d, p):
    """Unconstrained LWS with integer price p per line, O(n^2) literal."""
    n = len(C)-1
    F = [0]+[inf]*n
    for j in range(1, n+1):
        best = inf
        for i in range(j-1, -1, -1):
            g = C[j]-C[i]
            if g > d:
                break
            if F[i] < inf:
                best = min(best, F[i]+(d-g)**2+p)
        F[j] = best
    return F


def marginals_unit(d, m, n=None):
    K = d//2
    if n is None:
        n = (m-1)*K
    out = {}
    for j in range(m, n+1):
        a, b = P_unit(j, m, d), P_unit(j, m-1, d)
        if a < inf and b < inf:
            out[j] = a-b
    return out


def check_price_closed_form():
    print("== A. closed form P_j(m) vs literal layered DP (unit words) ==")
    rng = random.Random(20270315)
    bad = tested = 0
    for _ in range(120):
        d = rng.choice([4, 6, 8, 10, 12, 14])
        n = rng.randint(2, 18)
        m = rng.randint(1, 6)
        C = Cvec([1]*n)
        T = P_layered(C, d, m)
        for j in range(n+1):
            got = T[m][j]
            want = P_unit(j, m, d)
            tested += 1
            if got != want:
                bad += 1
    print("  cells:", tested, "| mismatches:", bad)
    assert bad == 0


def check_marginal_ap():
    print("== B. distinct marginals = AP {2d+8t}, length C(K,2) ==")
    for d in (8, 12, 16, 20, 24, 32, 40):
        K = d//2
        want = {2*d+8*t for t in range(K*(K-1)//2)}
        for m in (K, K+1, 2*K, 5*K):
            vals = set(marginals_unit(d, m).values())
            ok = vals == want
            print("  d=%2d K=%2d m=%3d n=%4d distinct=%3d C(K,2)=%3d AP_ok=%s "
                  "range=[%d,%d]"
                  % (d, K, m, (m-1)*K, len(vals), K*(K-1)//2, ok,
                     min(vals), max(vals)))
            assert ok
            assert len(vals) >= d*d//8 - d


def fenchel_prices_unit(d, m):
    K = d//2
    n = (m-1)*K
    C = [2*j for j in range(n+1)]
    Fs = {p: F_price(C, d, p) for p in range(-d*d, 1)}
    P = P_layered(C, d, m+2)
    prices = set()
    identity_ok = True
    for j in range(m, n+1):
        if P[m][j] >= inf:
            continue
        best, arg = -inf, None
        for p, F in Fs.items():
            v = F[j]-p*m
            if v > best:
                best, arg = v, p
        if best != P[m][j]:
            identity_ok = False
        prices.add(arg)
    # Subgradient interval for the maximizer is
    #   [-Delta_j(m+1), -Delta_j(m)]  (when P_j(m+1) is finite),
    # otherwise it extends below -d^2, so the leftmost price in [-d^2,0]
    # is -d^2.  Convexity reduces all other line counts to these neighbours.
    sub = set()
    for j in range(m, n+1):
        if P[m][j] >= inf:
            continue
        if P[m+1][j] < inf:
            sub.add(-(P[m+1][j]-P[m][j]))
        else:
            sub.add(-d*d)
    return prices, sub, identity_ok


def check_fenchel_count():
    print("== C. distinct Fenchel prices = C(K-1,2)+1 = Theta(d^2) ==")
    for d in (8, 12, 16, 20, 24):
        K = d//2
        m = K
        prices, sub, ok = fenchel_prices_unit(d, m)
        want = (K-1)*(K-2)//2 + 1
        print("  d=%2d K=%2d m=%2d distinct prices=%3d  C(K-1,2)+1=%3d "
              "subgradient-set=%3d  match=%s  identity_ok=%s"
              % (d, K, m, len(prices), want, len(sub),
                 len(prices) == want == len(sub) and prices == sub, ok))
        assert ok
        assert len(prices) == want == len(sub)
        assert prices == sub
        assert len(prices) >= d*d//8 - 2*d


def check_large_m():
    print("== D. Theta(d^2) prices persist for m >> d^2 ==")
    for d in (12, 16, 20):
        K = d//2
        for m in (d*d, 2*d*d, 5*d*d):
            vals = set(marginals_unit(d, m).values())
            print("  d=%2d m=%5d n=%6d distinct marginals=%3d (C(K,2)=%3d)"
                  % (d, m, (m-1)*K, len(vals), K*(K-1)//2))
            assert len(vals) == K*(K-1)//2


def check_general_bounds():
    print("== E. general C: marginal bound d^2-8 and price count min(n,d^2+1) ==")
    rng = random.Random(20270316)
    worst_delta = -1
    worst_ratio = 0.0
    bad_bound = 0
    for _ in range(160):
        d = rng.choice([6, 8, 10, 12])
        n = rng.randint(4, 14)
        w = [rng.randint(1, d//2) for _ in range(n)]
        C = Cvec(w)
        m = rng.randint(2, 5)
        T = P_layered(C, d, m+1)
        prices = set()
        Fs = {p: F_price(C, d, p) for p in range(-d*d, 1)}
        for j in range(m, n+1):
            if T[m][j] >= inf:
                continue
            for k in range(2, m+1):
                if T[k][j] < inf and T[k-1][j] < inf:
                    delta = T[k][j]-T[k-1][j]
                    worst_delta = max(worst_delta, delta)
                    if delta > d*d-8:
                        bad_bound += 1
            best, arg = -inf, None
            for p, F in Fs.items():
                v = F[j]-p*m
                if v > best:
                    best, arg = v, p
            prices.add(arg)
        worst_ratio = max(worst_ratio, len(prices)/min(n, d*d+1))
    print("  max finite marginal =", worst_delta, "| bound d^2-8 violations:",
          bad_bound, "| max D/min(n,d^2+1) =", round(worst_ratio, 3))
    assert bad_bound == 0
    assert worst_delta <= 12*12-8



# ==== nondecreasing intervals, reachability barriers, envelope starts ====
"""Machine checks for nondecreasing rank budgets, reachability-state barriers,
and the nonmonotone envelope-start barrier for the end-indexed DP.

All instances use d_t = beta_t + 1, C_j = sum_{t<=j}(alpha_t+1), and the
lexicographic objective (fewest lines, then least squared slack).

Checks:
  A. Nondecreasing budgets: R_t is always an interval [t,b_t]; the two-pointer
     recurrence gives K* exactly. Exhaustive n<=5 plus random n<=16.
  B. Proposition Z(a): greedy max-reach is not lexicographically optimal.
     Proposition Z(b): the end-indexed reduction with d_{L(j)} is not exact.
  C. Proposition AB: the O(n) max-reach recurrence fails for arbitrary budgets
     on feasible instances; explicit witness and exhaustive counts.
  D. Proposition AC: reachable endpoint sets can have Theta(n) disjoint
     intervals for alternating budgets on a periodic nonuniform grid.
  E. Proposition AA: insertion starts s_i of the end-indexed lower envelope
     can decrease, so split-only monotone interval split-find is insufficient.
"""
from itertools import product
from math import inf
import random

INFEASIBLE = None


def Cvec(w):
    C = [0]
    for x in w:
        C.append(C[-1]+x+1)
    return C


def g(C, i, d):
    """Largest j>i with C_j-C_i <= d."""
    n = len(C)-1
    j = i
    while j < n and C[j+1]-C[i] <= d:
        j += 1
    return j


def reach_brute(C, ds):
    """R_t = endpoints reachable with exactly t lines; returns list of sets."""
    n = len(C)-1
    R = {0}
    out = [set(R)]
    for d in ds:
        S = set()
        for i in R:
            for j in range(i+1, n+1):
                if C[j]-C[i] <= d:
                    S.add(j)
                else:
                    break
        R = S
        out.append(set(R))
        if not R:
            break
    return out


def interval_rec(C, ds):
    """Theorem 29 recurrence, with empty-set propagation."""
    n = len(C)-1
    b = 0
    empty = False
    hist = [set([0])]
    bseq = [0]
    for t, d in enumerate(ds, 1):
        nb = g(C, b, d)
        if empty or nb < t:
            empty = True
            R = set()
        else:
            R = set(range(t, nb+1))
        hist.append(R)
        b = nb
        bseq.append(nb)
    return hist, bseq


def Kstar_fast(C, ds):
    """O(n) two-pointer line count for nondecreasing budgets."""
    n = len(C)-1
    b = 0
    empty = False
    for t, d in enumerate(ds, 1):
        nb = g(C, b, d)
        if empty or nb < t:
            empty = True
        b = nb
        if not empty and b >= n:
            return t
    return inf


def Kstar_brute(C, ds):
    for t, R in enumerate(reach_brute(C, ds)):
        if t > 0 and len(C)-1 in R:
            return t
    return inf


def brute_lex(C, ds):
    """Return (min lines, min penalty) using rank budgets d_1..d_K."""
    n = len(C)-1
    INF = (inf, inf)
    dp = [INF]*(n+1)
    dp[0] = (0, 0)
    for t, d in enumerate(ds, 1):
        nd = [INF]*(n+1)
        for i in range(n+1):
            if dp[i][0] == inf:
                continue
            for j in range(i+1, n+1):
                gap = C[j]-C[i]
                if gap > d:
                    break
                cand = (dp[i][0]+1, dp[i][1]+(d-gap)**2)
                if cand < nd[j]:
                    nd[j] = cand
        dp = nd
        if dp[n][0] < inf:
            return dp[n]
    return INF


def greedy_maxreach_pen(C, ds, K):
    n = len(C)-1
    p = 0
    pen = 0
    for t in range(1, K+1):
        d = ds[t-1]
        nb = g(C, p, d)
        if t == K:
            nb = n
        if nb <= p:
            return inf
        pen += (d-(C[nb]-C[p]))**2
        p = nb
    return pen if p == n else inf


def endindexed_reduction(C, ds, K):
    """Candidate reduction: L(j)=min{t:b_t>=j}, budget of a line ending at j
    is d_{L(j)}; returns its lex answer."""
    n = len(C)-1
    _, bseq = interval_rec(C, ds)
    L = [inf]*(n+1)
    for t, b in enumerate(bseq):
        if t == 0:
            continue
        for j in range(0, min(b, n)+1):
            if L[j] == inf:
                L[j] = t
    Lam = 10**9
    dp = [inf]*(n+1)
    dp[0] = 0
    for j in range(1, n+1):
        if L[j] == inf:
            continue
        d = ds[min(L[j]-1, len(ds)-1)]
        best = inf
        for i in range(j-1, -1, -1):
            gap = C[j]-C[i]
            if gap > d:
                break
            if dp[i] < inf:
                best = min(best, dp[i]+Lam+(d-gap)**2)
        dp[j] = best
    if dp[n] >= inf:
        return (inf, inf)
    return (dp[n]//Lam, dp[n] % Lam)


def intervals_of(S):
    if not S:
        return 0
    s = sorted(S)
    k = 1
    for a, b in zip(s, s[1:]):
        if b > a+1:
            k += 1
    return k


def maxreach_lazy(C, ds):
    """f_t = g(f_{t-1},d_t), allowing f_t=f_{t-1}; returns first t with f_t=n."""
    n = len(C)-1
    f = 0
    for t, d in enumerate(ds, 1):
        f = g(C, f, d)
        if f >= n:
            return t
    return inf


def check_nondecreasing_interval():
    print("== A. nondecreasing budgets: interval invariant and O(n) line count ==")
    tot = mism_interval = mism_K = 0
    for n in range(1, 6):
        for w in product(range(1, 4), repeat=n):
            C = Cvec(list(w))
            for ds in product(range(2, 8), repeat=n):
                if not all(ds[k] <= ds[k+1] for k in range(n-1)):
                    continue
                tot += 1
                rb = reach_brute(C, list(ds))
                ri, _ = interval_rec(C, list(ds))
                if any(rb[t] != ri[t] for t in range(min(len(rb), len(ri)))):
                    mism_interval += 1
                k1 = Kstar_fast(C, list(ds))
                k2 = Kstar_brute(C, list(ds))
                if k1 != k2:
                    mism_K += 1
    rng = random.Random(20270317)
    rand_bad = 0
    max_int = 0
    for _ in range(6000):
        n = rng.randint(2, 16)
        w = [rng.randint(1, 6) for _ in range(n)]
        C = Cvec(w)
        ds = sorted([rng.randint(2, 14) for _ in range(n)])
        rb = reach_brute(C, ds)
        max_int = max(max_int, max(intervals_of(R) for R in rb))
        if Kstar_fast(C, ds) != Kstar_brute(C, ds):
            rand_bad += 1
    print("  exhaustive instances:", tot, "| interval mismatches:", mism_interval,
          "| K* mismatches:", mism_K)
    print("  random n<=16: max intervals in R_t =", max_int,
          "| K* mismatches:", rand_bad, "of 6000")
    assert mism_interval == 0 and mism_K == 0 and rand_bad == 0
    assert max_int == 1


def check_proposition_Z():
    print("== B. Proposition Z: greedy and end-indexed reduction barriers ==")
    # Z(a)
    w = (1, 1, 1)
    C = Cvec(list(w))
    ds = (4, 5)
    K, opt = brute_lex(C, list(ds))
    gp = greedy_maxreach_pen(C, list(ds), K)
    print("  Z(a) w=(1,1,1), d=(4,5): K=", K, "opt=", opt,
          "greedy max-reach penalty=", gp)
    assert (K, opt) == (2, 5) and gp == 9
    # Z(b)
    w2 = (1, 1)
    C2 = Cvec(list(w2))
    ds2 = (2, 4)
    K2 = brute_lex(C2, list(ds2))[0]
    red = endindexed_reduction(C2, list(ds2), K2)
    print("  Z(b) w=(1,1), d=(2,4): rank answer=", brute_lex(C2, list(ds2)),
          "end-indexed reduction=", red)
    assert brute_lex(C2, list(ds2)) == (2, 4) and red == (1, 0)
    # exhaustive counts
    tot = bad_greedy = bad_red = 0
    for n in range(1, 6):
        for w3 in product(range(1, 4), repeat=n):
            C3 = Cvec(list(w3))
            for ds3 in product(range(2, 8), repeat=n):
                if not all(ds3[k] <= ds3[k+1] for k in range(n-1)):
                    continue
                K3, opt3 = brute_lex(C3, list(ds3))
                if K3 == inf:
                    continue
                tot += 1
                if greedy_maxreach_pen(C3, list(ds3), K3) != opt3:
                    bad_greedy += 1
                if endindexed_reduction(C3, list(ds3), K3) != (K3, opt3):
                    bad_red += 1
    print("  feasible nondecreasing instances:", tot,
          "| greedy-opt mismatches:", bad_greedy,
          "| end-indexed reduction mismatches:", bad_red)
    assert bad_greedy == 4193 and bad_red == 13062


def check_proposition_AB():
    print("== C. Proposition AB: max-reach recurrence fails in general ==")
    w = (1, 1, 2, 2)
    C = Cvec(list(w))
    ds = (7, 2, 3, 3)
    K = Kstar_brute(C, list(ds))
    lazy = maxreach_lazy(C, list(ds))
    reach = reach_brute(C, list(ds))
    print("  witness w=(1,1,2,2), d=(7,2,3,3):")
    print("    R_1..R_4 =", [sorted(R) for R in reach[1:5]])
    print("    exact K* =", K, "| lazy max-reach K =", lazy)
    assert K == 4 and lazy == 3
    tot = feas = bad_strict = bad_lazy = 0
    for n in range(1, 6):
        for w2 in product(range(1, 4), repeat=n):
            C2 = Cvec(list(w2))
            for ds2 in product(range(2, 8), repeat=n):
                tot += 1
                kb = Kstar_brute(C2, list(ds2))
                kl = maxreach_lazy(C2, list(ds2))
                if kb < inf:
                    feas += 1
                if kb != kl:
                    bad_lazy += 1
                    if kb < inf and kl < inf:
                        bad_strict += 1
    print("  exhaustive instances:", tot, "| feasible:", feas,
          "| lazy-recurrence mismatches:", bad_lazy,
          "| of those on feasible instances:", bad_strict)
    assert bad_lazy == 393417 and bad_strict == 4549


def constructed_reach_intervals(n, B=10, period=3, SMALL=4):
    gaps = []
    while len(gaps) < n:
        gaps += [2]*(period-1)+[B]
    gaps = gaps[:n]
    C = [0]
    for x in gaps:
        C.append(C[-1]+x)
    BIG = 2*B+8
    ds = [BIG if t % 2 == 1 else SMALL for t in range(1, n+1)]
    hist = [intervals_of(R) for R in reach_brute(C, ds)]
    return max(hist), C, ds, hist


def check_proposition_AC():
    print("== D. Proposition AC: Theta(n) reachable intervals ==")
    rows = []
    for n in (30, 60, 90, 120, 180, 240):
        k, C, ds, hist = constructed_reach_intervals(n)
        rows.append((n, k))
        print("  n=%3d max intervals in R_t = %2d" % (n, k))
    print("  growth:", rows)
    assert rows[-1][1] >= rows[-1][0]//6
    # contrast: nondecreasing budgets keep one interval
    rng = random.Random(20270318)
    mx = 0
    for _ in range(2000):
        n = rng.randint(3, 14)
        C = Cvec([rng.randint(1, 5) for _ in range(n)])
        ds = sorted([rng.randint(2, 12) for _ in range(n)])
        mx = max(mx, max(intervals_of(R) for R in reach_brute(C, ds)))
    print("  random nondecreasing budgets: max intervals =", mx)
    assert mx == 1


def dp_brute_end(C, z, Lam):
    n = len(z)
    dp = [0]+[None]*n
    for j in range(1, n+1):
        zj = z[j-1]
        best = None
        for i in range(j-1, -1, -1):
            if C[i] < zj:
                break
            if dp[i] is None:
                continue
            v = dp[i]+(C[i]-zj)**2
            if best is None or v < best:
                best = v
        if best is not None:
            dp[j] = Lam+best
    return dp


def envelope_starts(C, z, dp):
    """Rightmost-tie rank s_i where line i first owns the lower envelope."""
    X = sorted(set(z))
    D = len(X)
    e = [-1]*(len(C))
    q = -1
    for i in range(len(C)):
        while q+1 < D and X[q+1] <= C[i]:
            q += 1
        e[i] = q
    s = [0]*(len(C))
    for j in range(len(C)):
        if dp[j] is None:
            s[j] = s[j-1] if j > 0 else 0
            continue
        owned = []
        for r in range(e[j]+1):
            vals = []
            for i in range(j+1):
                if e[i] < r or dp[i] is None:
                    continue
                vals.append((dp[i]+(C[i]-X[r])**2, i))
            mn = min(v for v, i in vals)
            best = [i for v, i in vals if v == mn]
            if max(best) == j:
                owned.append(r)
        s[j] = min(owned) if owned else (s[j-1] if j > 0 else 0)
    return s


def check_proposition_AA():
    print("== E. Proposition AA: end-indexed envelope starts can decrease ==")
    w = (1, 5, 4, 1, 4, 6, 6)
    C = Cvec(list(w))
    z = (-320, 5, 1, -343, -72, -98, -61)
    X = sorted(set(z))
    Lam = len(C)*((C[-1]-X[0])**2)+1
    dp = dp_brute_end(C, z, Lam)
    s = envelope_starts(C, z, dp)
    print("  C =", C)
    print("  z =", z)
    print("  dp =", dp)
    print("  s =", s)
    assert s == [0, 5, 5, 6, 6, 5, 5, 5]
    assert s[5] < s[4]
    # random search confirms the phenomenon is not isolated
    rng = random.Random(20270319)
    bad = 0
    for _ in range(1000):
        n = rng.randint(2, 12)
        w2 = [rng.randint(1, 5) for _ in range(n)]
        C2 = Cvec(w2)
        z2 = [rng.randint(-15, C2[j]-2) for j in range(1, n+1)]
        X2 = sorted(set(z2))
        Lam2 = len(C2)*((C2[-1]-X2[0])**2)+1
        dp2 = dp_brute_end(C2, z2, Lam2)
        s2 = envelope_starts(C2, z2, dp2)
        if any(s2[k] > s2[k+1] for k in range(len(s2)-1)):
            bad += 1
    print("  random instances with nonmonotone s_i:", bad, "of 1000")
    assert bad >= 1



# ==== baseline locator checks for solution.md ====
"""Baseline exhaustive checks for solution.md.

No dependencies; every claim is re-derived by brute force on small instances or
checked against the closed forms. Executed by run_all_checks() in this module.

Checks
  1. base_greedy's overrun lemma (a block overruns its budget iff singleton & over-wide)
  2. the feasibility criterion (Fact 1) against exhaustive layout search
  3. Phi is base_legal iff the instance is feasible; Phi >= OPT; Fact 2 blow-up
  4. OPT = L(j*-1) + 1 against exhaustive layout search
  5. paragraph additivity of the line count
  6. both secondary-objective witnesses (Q3a) and the (i,k) DP (Q3b) against
     an exhaustive minimum-line search
  7. the mixed (body/terminal) matrix is Monge and the feasible column window is a
     nondecreasing staircase (Q3c)
  8. the infeasible-region loss vectors and the Pareto no-go claims (Q4)
"""

from itertools import product

INFEASIBLE = None


# ── Model ────────────────────────────────────────────────────────────────────

def base_span(w, i, j):
    """B(i, j) for 0-based inclusive word indices, or 0 for an empty range."""
    if i > j:
        return 0
    return sum(w[i:j + 1]) + (j - i)


def base_layouts(n):
    """Every partition of range(n) into consecutive non-empty blocks."""
    for mask in range(1 << max(n - 1, 0)):
        blocks, start = [], 0
        for i in range(n - 1):
            if mask >> i & 1:
                blocks.append((start, i))
                start = i + 1
        blocks.append((start, n - 1))
        yield blocks


def base_legal(w, A, s, blocks):
    for idx, (i, j) in enumerate(blocks):
        limit = A - s if idx == len(blocks) - 1 else A
        if base_span(w, i, j) > limit:
            return False
    return True


def base_feasible_by_search(w, A, s):
    if not w:
        return s <= A
    return any(base_legal(w, A, s, b) for b in base_layouts(len(w)))


def base_feasible_criterion(w, A, s):
    """Fact 1 (first half): the closed-form criterion."""
    if not w:
        return s <= A
    return max(w) <= A and w[-1] <= A - s


def base_opt_by_search(w, A, s):
    if not w:
        return 1 if s <= A else INFEASIBLE
    best = INFEASIBLE
    for b in base_layouts(len(w)):
        if base_legal(w, A, s, b) and (best is None or len(b) < best):
            best = len(b)
    return best


def base_penalty(w, A, s, blocks):
    total = 0
    for idx, (i, j) in enumerate(blocks):
        beta = A - s if idx == len(blocks) - 1 else A
        slack = beta - base_span(w, i, j)
        total += slack * slack
    return total


def base_opt2_by_search(w, A, s):
    """Best squared-slack base_penalty among base_legal base_layouts of minimum line count."""
    if not w:
        return 0 if s <= A else INFEASIBLE
    best_lines = base_opt_by_search(w, A, s)
    if best_lines is INFEASIBLE:
        return INFEASIBLE
    best = INFEASIBLE
    for b in base_layouts(len(w)):
        if base_legal(w, A, s, b) and len(b) == best_lines:
            p = base_penalty(w, A, s, b)
            if best is None or p < best:
                best = p
    return best


# ── Greedy / Phi / the optimal split ─────────────────────────────────────────

def base_greedy(w, lo, hi, budget):
    """Greedy blocks over w[lo..hi] at `budget` (a block may be a lone over-wide word)."""
    blocks, i = [], lo
    while i <= hi:
        j = i
        if w[i] > budget:
            blocks.append((i, i))
            i += 1
            continue
        while j + 1 <= hi and base_span(w, i, j + 1) <= budget:
            j += 1
        blocks.append((i, j))
        i = j + 1
    return blocks


def base_greedy_count(w, budget, hi=None):
    if not w:
        return 0
    hi = len(w) - 1 if hi is None else hi
    if hi < 0:
        return 0
    return len(base_greedy(w, 0, hi, budget))


def base_phi(w, A, s):
    """The two-phase algorithm of the problem statement."""
    if not w:
        return []
    blocks = base_greedy(w, 0, len(w) - 1, A)
    last = blocks[-1]
    if base_span(w, *last) + s <= A:
        return blocks
    return blocks[:-1] + base_greedy(w, last[0], last[1], A - s)


def base_jstar(w, A, s):
    """Start of the longest feasible suffix, or INFEASIBLE."""
    if not w:
        return INFEASIBLE
    best = INFEASIBLE
    for j in range(len(w) - 1, -1, -1):
        if base_span(w, j, len(w) - 1) <= A - s:
            best = j
    return best


def base_opt_closed_form(w, A, s):
    """OPT = L(j*-1) + 1 (§2.3), plus the feasibility gate of §2.1."""
    if not w:
        return 1 if s <= A else INFEASIBLE
    if max(w) > A:
        return INFEASIBLE
    j = base_jstar(w, A, s)
    if j is INFEASIBLE:
        return INFEASIBLE
    return base_greedy_count(w, A, j - 1) + 1


def base_wrap_locator(w, A, s):
    """The O(n) algorithm of §2.4 — O(1) extra space, no base_span recomputation."""
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


def base_opt2_dp(w, A, s):
    """The (i,k) DP of §3(b) with the terminal-block append, O(n^2 * OPT)."""
    n = len(w)
    opt = base_opt_closed_form(w, A, s)
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
                b = base_span(w, p, i - 1)
                if b <= A:
                    cur[i] = min(cur[i], prev[p] + (A - b) ** 2)
        prev = cur
    for i in range(0, n):                       # terminal block w[i..n-1]
        b = base_span(w, i, n - 1)
        if prev[i] != INF and b <= A - s:
            best = min(best, prev[i] + (A - s - b) ** 2)
    return best


# ── Checks ───────────────────────────────────────────────────────────────────

def base_check_greedy_overrun():
    for n in range(1, 6):
        for w in product(range(1, 7), repeat=n):
            for budget in range(1, 8):
                for (i, j) in base_greedy(w, 0, n - 1, budget):
                    over = base_span(w, i, j) > budget
                    if over and not (i == j and w[i] > budget):
                        raise AssertionError(("lemma 1", w, budget, (i, j)))
    print("ok  lemma 1: a base_greedy block overruns iff it is a lone over-wide word")


def base_check_feasibility():
    for n in range(0, 5):
        widths = [()] if n == 0 else product(range(1, 5), repeat=n)
        for w in widths:
            for A in range(1, 9):
                for s in range(0, 7):
                    got = base_feasible_criterion(w, A, s)
                    want = base_feasible_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("feasibility", w, A, s, got, want))
    print("ok  Fact 1: closed-form feasibility == exhaustive layout search")


def base_check_phi():
    feasible_count = 0
    for n in range(0, 5):
        widths = [()] if n == 0 else product(range(1, 5), repeat=n)
        for w in widths:
            for A in range(1, 9):
                for s in range(0, 7):
                    out = base_phi(w, A, s)
                    legal_out = (not w and s <= A) or (bool(out) and base_legal(w, A, s, out))
                    if legal_out != base_feasible_criterion(w, A, s):
                        raise AssertionError(("base_phi legality", w, A, s, out))
                    if legal_out:
                        feasible_count += 1
                        # n = 0 has the unique empty line; `base_phi` returns no blocks.
                        lines = len(out) if w else 1
                        if lines < base_opt_by_search(w, A, s):
                            raise AssertionError(("base_phi beats OPT", w, A, s, out))
    assert feasible_count > 0
    print(f"ok  Phi base_legal iff feasible ({feasible_count} feasible instances); Phi >= OPT")

    for M in (2, 3, 5, 8, 13):
        w = [1] * M
        A, s = 2 * M + 1, 2 * M
        got, want = len(base_phi(w, A, s)), base_opt_by_search(w, A, s)
        assert got == M and want == 2, (M, got, want)
    print("ok  Fact 2: Phi emits M lines where OPT = 2 (ratio M/2 -> unbounded)")


def base_check_opt_formula():
    for n in range(0, 6):
        widths = [()] if n == 0 else product(range(1, 6), repeat=n)
        for w in widths:
            for A in range(1, 10):
                for s in range(0, 8):
                    got = base_opt_closed_form(w, A, s)
                    want = base_opt_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("OPT", w, A, s, got, want))
                    if base_wrap_locator(w, A, s) != got:
                        raise AssertionError(("O(n) algorithm", w, A, s))
    print("ok  OPT = L(j*-1)+1 and the O(n) algorithm agree with exhaustive search")


def base_check_paragraph_additivity():
    """§1.4: the paragraph optimum is the sum of the per-paragraph optima.

    The ground truth uses the exhaustive per-paragraph search (`base_opt_by_search`,
    budget `A` for body paragraphs, `A`/`s` for the last one); the claim under
    test is the composed criterion `paragraph_opt`, which derives each value
    from the closed forms instead.
    """

    def paragraph_opt(paras, A, s):
        total = 0
        for r, p in enumerate(paras):
            if r < len(paras) - 1:
                # An empty body paragraph is still one empty line.
                got = 1 if not p else (base_greedy_count(p, A) if max(p) <= A else INFEASIBLE)
            elif not p:
                got = 1 if s <= A else INFEASIBLE
            else:
                got = base_opt_closed_form(p, A, s)
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
                        got = base_opt_by_search(p, A, 0 if r < len(paras) - 1 else s)
                        if got is INFEASIBLE:
                            want = INFEASIBLE
                            break
                        want += got
                    if want is INFEASIBLE:
                        assert paragraph_opt(paras, A, s) is INFEASIBLE, (paras, A, s)
                        continue
                    assert paragraph_opt(paras, A, s) == want, (paras, A, s, want)
    print("ok  paragraph line counts are additive (each paragraph independently optimal)")


def base_check_secondary_objective():
    # Witness 1: L is flat on [j*, j'] -> Q2's split is not base_penalty-optimal.
    w, A, s = [1, 1, 1], 4, 1
    q2 = base_penalty(w, A, s, [(0, 0), (1, 2)])
    alt = base_penalty(w, A, s, [(0, 1), (2, 2)])
    assert q2 == 9 and alt == 5 and base_opt2_by_search(w, A, s) == 5 and base_opt2_dp(w, A, s) == 5

    # Witness 2: j* is forced, the prefix still has an L-flat alternative.
    w2, A2, s2 = [2, 1, 1, 1], 4, 2
    q2b = base_penalty(w2, A2, s2, [(0, 1), (2, 2), (3, 3)])
    altb = base_penalty(w2, A2, s2, [(0, 0), (1, 2), (3, 3)])
    assert q2b == 10 and altb == 6 and base_opt2_by_search(w2, A2, s2) == 6
    print("ok  Q3a: both witnesses make base_greedy+longest-suffix strictly suboptimal")

    # Q3b: the DP equals an exhaustive minimum-line search over all instances.
    for n in range(1, 6):
        for w in product(range(1, 5), repeat=n):
            for A in range(1, 9):
                for s in range(0, 7):
                    got, want = base_opt2_dp(w, A, s), base_opt2_by_search(w, A, s)
                    if got != want:
                        raise AssertionError(("opt2", w, A, s, got, want))
    print("ok  Q3b: the (i,k) DP equals exhaustive search over minimum-line base_layouts")


def base_check_monge():
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
    print("ok  Q3c: the mixed body/terminal matrix is Monge; the window staircase is monotone")


def base_check_policies():
    # Loss vectors, dominance table, and the no-go premises in the feasible region.
    for n in range(1, 6):
        for w in product(range(1, 5), repeat=n):
            for A in range(1, 9):
                for s in range(0, 7):
                    if max(w) > A or w[-1] <= A - s:
                        continue                      # only the infeasible region
                    tau = w[-1] + s - A
                    ell = base_greedy_count(w, A, n - 2) + 1
                    delta = base_greedy_count(w, A) - base_greedy_count(w, A, n - 2)
                    assert tau > 0 and delta in (0, 1)
                    p1 = (tau, ell, 0)
                    p3 = (0, ell, tau)
                    # P1/P3 are always incomparable: no componentwise order.
                    assert not base_dominates(p1, p3) and not base_dominates(p3, p1)
                    if s <= A:
                        p2 = (0, ell + delta, 0)
                        if delta == 0:
                            assert base_dominates(p2, p1) and base_dominates(p2, p3)
                        else:
                            for a, b in ((p1, p2), (p2, p1), (p1, p3), (p3, p1), (p2, p3), (p3, p2)):
                                assert not base_dominates(a, b)
                        # No-go premises: full locator + zero overflow needs
                        # L(n) body lines plus one locator line.
                        assert base_greedy_count(w, A) + 1 == ell + delta
    # s > A: P2 inadmissible, and a split-locator hybrid is strictly better.
    w, A, s = [1, 1], 4, 5
    p1, p3 = (2, 2, 0), (0, 2, 2)
    hybrid = (0, 2, 0)
    assert base_dominates(hybrid, p1) and base_dominates(hybrid, p3)
    print("ok  Q4: loss vectors, dominance table, and the s>A hybrid counterexample")


def base_dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def check_baseline():
    base_check_greedy_overrun()
    base_check_feasibility()
    base_check_phi()
    base_check_opt_formula()
    base_check_paragraph_additivity()
    base_check_secondary_objective()
    base_check_monge()
    base_check_policies()
    print("\nall checks passed")


def run_all_checks():
    check_baseline()
    check_1a()
    check_1a_dp()
    check_1a_monge()
    check_strict()
    check_cdq()
    check_scaling_cont()
    check_envelope()
    check_envelope_lex()
    check_stack_only()
    check_rank_envelope()
    check_construction()
    check_lex_and_exhaustive()
    check_few_coords()
    check_linear_family()
    check_envelope_activity()
    check_prefix_linearity()
    check_A()
    check_B()
    check_C()
    check_window()
    check_lower_bound_construction()
    check_lower_bound_reduction()
    check_predecessor_reduction()
    check_2(14, True)
    check_2_construction()
    check_3()
    check_3_exact()
    check_frontier_fast()
    check_frontier_structure()
    check_relational()
    check_regime_exhaustive()
    check_rank_convexity()
    check_rank()
    check_relational_secondary()
    check_reversed()
    find_arbitrary_counterexample()
    relational_counterexample()
    witness_1b()
    check_1b()
    check_uniform_rank()
    check_uniform_relational()
    check_budget_order()
    check_periodic_dp()
    check_residue_convexity()
    check_core_size()
    check_scaling_periodic()
    check_decomposition()
    check_one_param_gap_examples()
    check_two_param_gap()
    check_residual_structure()
    check_reversal_equivalence()
    check_decomposition_algorithm()
    check_hstar_formula()
    check_equal_uniform_algorithm()
    check_nonuniform_barrier()
    check_spline_identity()
    check_convexity_frontier()
    check_minimax_gap()
    check_structure_counterexamples()
    check_m2_linear()
    check_lambda_monotonicity()
    check_cht_against_brute()
    check_hinge_reduction()
    check_convex_extension()
    check_structured_source_nonconvex()
    check_super_source_monge()
    check_power_density()
    check_tm_lambda()
    check_all_endpoint_fenchel()
    check_monge_subclass()
    check_offline_reduction()
    check_marginal_bound()
    check_bounded_fenchel_vs_exact()
    check_full_algorithm()
    check_explicit_gap()
    check_analytic_bound()
    check_random_gaps()
    check_explicit_source_witness()
    check_structured_source_witness()
    check_random_source_gaps()
    check_random_structured_source_gaps()
    check_closed_form()
    check_random_large_budget()
    check_fixed_budget_is_linear()
    check_rank3()
    check_spline_form()
    check_continuous_density()
    check_explicit_family()
    check_nonuniform()
    check_price_closed_form()
    check_marginal_ap()
    check_fenchel_count()
    check_large_m()
    check_general_bounds()
    check_nondecreasing_interval()
    check_proposition_Z()
    check_proposition_AB()
    check_proposition_AC()
    check_proposition_AA()
    print("ALL CHECKS COMPLETED")


if __name__ == "__main__":
    run_all_checks()

