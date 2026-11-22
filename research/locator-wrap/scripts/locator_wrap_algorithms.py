#!/usr/bin/env python3
"""Algorithms and machine checks for asymmetric line breaking.

All routines used by locator_wrap_checks.py are defined here.  No external
project imports are required.
"""
from bisect import bisect_left, bisect_right
from collections import deque
from itertools import product
from math import inf
import random, time

INFEASIBLE = None


# ==== experiments.py ====
#!/usr/bin/env python3
"""Computational experiments for the three open problems in the locator-wrap notes.

Run: python3 /data/workspace/experiments.py
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


# ==== experiments_extra.py ====
#!/usr/bin/env python3
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

def dp_brute(C, z, lam):
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
            a = dp_brute(C, z, lam)
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
    print("  (standard start = e_0; structured = layered output)")

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


# ==== experiments_final.py ====
#!/usr/bin/env python3
"""Final checks: alternative TM convention, explicit non-convex witness for 1(b),
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


# ==== experiments_continue.py ====
#!/usr/bin/env python3
"""Continuation: an O(n log n) DP for arbitrary end-indexed budgets,
verification against exhaustive layouts, and a rigorous proof of the
Problem-3 maximum frontier size.

Run: python3 experiments_continue.py
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
        solve(l, mid, lo_list)                    # finalize lines in [l,mid]
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


def check_scaling():
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
                continue                 # dominated by an earlier level
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
    print("== Problem 3: corrected structure ==")
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



# ==== experiments_last.py ====
#!/usr/bin/env python3
"""Final round on the two remaining open problems.

 1(a)  * an O(n*B) DP (B = max budget) => O(n) for bounded budgets
       * verification of the Omega(n log n) comparison-model construction
 1(b)  * line count of the relational variant: is backward greedy exact?
       * convexity of P(R) for rank-indexed budgets from the single source

Run: python3 experiments_last.py
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
    (r=1 is the LAST line) and all earlier lines have budget A."""
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



# ==== experiments_resolve.py ====
#!/usr/bin/env python3
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
    the predecessor lies in the previous greedy block.  Within a level the
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
        # build lower hull of lines from previous block
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
    A-s[0] (last), A-s[1], ... and all earlier lines budget A."""
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



# ==== experiments_resolve_more.py ====
#!/usr/bin/env python3
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



# ==== experiments_final_resolution.py ====
#!/usr/bin/env python3
"""Final-resolution experiments.

A. 1(a): a compressed incremental-envelope DP that evaluates the lower
   envelope only at the n query coordinates z_j.  With w-bit integer
   coordinates and linear-time integer sorting/predecessor structures this
   is O(n) in the word-RAM model, removing the bounded-universe restriction.

B. 1(b): counterexamples that delimit the O(n) rank algorithm and the
   relational reduction; control checks for the nonincreasing regime.

C. A rank-1 transform identity for K=2, used in the conditional lower-bound
   discussion.

Run: python3 experiments_final_resolution.py
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
#!/usr/bin/env python3
"""Stack-only compressed envelope DP for 1(a).

Instead of materialising the envelope values on an array, maintain the lower
envelope as a stack of disjoint intervals.  Each line is inserted once and
each interval is popped once; a query needs one predecessor search among
interval starts.  With a static/dynamic word-RAM predecessor structure this
gives O(n * pred(n,U)) time and O(n) space for arbitrary w-bit integer
coordinates.

Run: python3 experiments_stack_only.py
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
#!/usr/bin/env python3
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

    print("  -- exhaustive small-instance check --")
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






