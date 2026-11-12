"""Independent check of every mathematical claim in the revised note."""
from itertools import product, combinations, permutations
from collections import defaultdict

# ---------- generic graph helpers ----------
def edges_from_intervals(iv):
    n = len(iv)
    return tuple((i, j) for i in range(n) for j in range(i + 1, n)
                 if max(iv[i][0], iv[j][0]) <= min(iv[i][1], iv[j][1]))

def adj_matrix(n, edges):
    A = [[False]*n for _ in range(n)]
    for a, b in edges:
        A[a][b] = A[b][a] = True
    return A

def is_proper(c, edges):
    return all(c[a] != c[b] for a, b in edges)

def normalised(n, edges, k, K):
    fixed = [None]*n
    for i, v in enumerate(K):
        fixed[v] = i + 1
    free = [v for v in range(n) if fixed[v] is None]
    out = set()
    for vals in product(range(1, k+1), repeat=len(free)):
        c = fixed[:]
        for v, x in zip(free, vals):
            c[v] = x
        if is_proper(c, edges):
            out.add(tuple(c))
    return sorted(out)

def chromatic(n, edges, k=6):
    for kk in range(1, k+1):
        for c in product(range(1, kk+1), repeat=n):
            if is_proper(c, edges):
                return kk
    return None

def clique_number(n, edges):
    best = 0
    for r in range(1, n+1):
        for S in combinations(range(n), r):
            if all((a, b) in edges or (b, a) in edges for a, b in combinations(S, 2)):
                best = r
    return best

def pleq(c, d):
    return all(x <= y for x, y in zip(c, d))

def partitions(c):
    d = defaultdict(set)
    for v, col in enumerate(c):
        d[col].add(v)
    return {frozenset(s) for s in d.values()}

def refines(p, q):
    return all(any(b <= bl for bl in q) for b in p)

# ---------- Witness A ----------
ivA = [(0, 3), (2, 4.5), (0, 10), (5.5, 8), (7, 10)]  # v1,v2,v3,u,v
eA = edges_from_intervals(ivA)
print("A edges:", eA)
assert eA == ((0,1),(0,2),(1,2),(2,3),(2,4),(3,4)), eA
assert clique_number(5, eA) == 3 and chromatic(5, eA) == 3
CA = normalised(5, eA, 3, [0,1,2])
print("A normalised colourings:", CA)
assert len(CA) == 2
c, d = CA
mn = tuple(min(x,y) for x,y in zip(c,d))
mx = tuple(max(x,y) for x,y in zip(c,d))
assert not is_proper(mn, eA) and not is_proper(mx, eA)
assert mn not in CA and mx not in CA
assert not pleq(c,d) and not pleq(d,c)
pa, pb = partitions(c), partitions(d)
assert not refines(pa,pb) and not refines(pb,pa)
# second maximum clique gives the same antichain
CA2 = normalised(5, eA, 3, [3,4,2])
print("A normalised on {u,v,v3}:", CA2)
assert len(CA2) == 2 and not pleq(CA2[0], CA2[1]) and not pleq(CA2[1], CA2[0])

# ---------- Witness B ----------
ivB = [(0, 4), (1, 3), (2, 5), (0, 0.5)]  # a,b,c,x
eB = edges_from_intervals(ivB)
print("B edges:", eB)
assert eB == ((0,1),(0,2),(0,3),(1,2)), eB
assert clique_number(4, eB) == 3 and chromatic(4, eB) == 3
CB = normalised(4, eB, 3, [0,1,2])
print("B normalised colourings:", CB)
assert len(CB) == 2
assert pleq(CB[0], CB[1]) and not pleq(CB[1], CB[0]), "B pointwise = 2-chain"
assert tuple(min(x,y) for x,y in zip(*CB)) in CB
assert tuple(max(x,y) for x,y in zip(*CB)) in CB
p1, p2 = partitions(CB[0]), partitions(CB[1])
assert not refines(p1,p2) and not refines(p2,p1)
common_ref = {frozenset(a & b) for a in p1 for b in p2 if a & b}
common_coa = {frozenset.union(*g) for g in []}  # computed explicitly below
# coarsening = transitive closure of "share a vertex" over the two partitions
blocks = list(p1) + list(p2)
parent = list(range(len(blocks)))
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x
for i in range(len(blocks)):
    for j in range(i+1, len(blocks)):
        if blocks[i] & blocks[j]:
            parent[find(i)] = find(j)
groups = defaultdict(set)
for i, b in enumerate(blocks):
    groups[find(i)] |= set(b)
common_coa = {frozenset(s) for s in groups.values()}
print("B common refinement:", common_ref)
print("B common coarsening:", common_coa)
assert len(common_ref) == 4
assert len(common_coa) == 2 and not is_proper(
    tuple(next(i for i,b in enumerate(sorted(common_coa, key=lambda z: sorted(z))) if v in b)+1 for v in range(4)), eB)
print("B: pointwise 2-chain (distributive lattice); refinement 2-antichain  OK")

# ---------- Witness C ----------
eC = tuple((a,b) for a,b in combinations(range(3),2)) + tuple((a,b) for a,b in combinations(range(3,6),2))
CC = normalised(6, eC, 3, [0,1,2])
print("C normalised colourings:", CC)
assert len(CC) == 6
assert all(not pleq(a,b) and not pleq(b,a) for a,b in combinations(CC,2))
assert all(any(not is_proper(tuple(min(x,y) for x,y in zip(a,b)), eC) for b in CC if a != b) for a in CC)
assert all(len(partitions(c)) == 3 for c in CC)

# S3 weak orders
s1, s2 = (2,1,3), (1,3,2)
def mul(p, q):  # p o q
    return tuple(p[q[i]-1] for i in range(3))
def inv_count(p):
    return sum(1 for i in range(3) for j in range(i+1,3) if p[i] > p[j])
words = defaultdict(set)
for L in range(4):
    for w in product((1,2), repeat=L):
        p = (1,2,3)
        for g in w:
            p = mul(p, s1 if g == 1 else s2)
        if inv_count(p) == L:
            words[p].add(w)
S3 = sorted(words)
assert len(S3) == 6
def make_order(kind):
    le = set()
    for u in S3:
        for v in S3:
            for wu in words[u]:
                for wv in words[v]:
                    ok = (wv[:len(wu)] == wu) if kind == 'prefix' else (wv[len(wv)-len(wu):] == wu if wu else True)
                    if ok:
                        le.add((u,v)); break
                else:
                    continue
                break
    def LE(u,v): return (u,v) in le
    def meet(a,b):
        lows=[x for x in S3 if LE(x,a) and LE(x,b)]
        return max(lows, key=lambda x: sum(1 for y in lows if LE(y,x)))
    def join(a,b):
        ups=[x for x in S3 if LE(a,x) and LE(b,x)]
        return max(ups, key=lambda x: sum(1 for y in ups if LE(x,y)))
    return LE, meet, join
for kind in ('prefix','suffix'):
    LE, meet, join = make_order(kind)
    # sanity: meet/join are actual glb/lub
    for a,b in product(S3,S3):
        m, j = meet(a,b), join(a,b)
        assert LE(m,a) and LE(m,b) and LE(a,j) and LE(b,j)
    e = (1,2,3); w0 = (3,2,1)
    s2s1 = mul(s2,s1); s1s2 = mul(s1,s2)
    lhs = join(s1, meet(s2, s2s1))
    rhs = meet(join(s1,s2), join(s1,s2s1))
    print(f"C {kind} weak order: s1 v (s2 ^ s2s1) = {lhs}, (s1 v s2) ^ (s1 v s2s1) = {rhs}")
    if kind == 'suffix':   # left weak order
        assert lhs == s1 and rhs == s2s1 and lhs != rhs
    else:                   # right weak order
        assert lhs == w0 and rhs == w0
    # mirror instance for prefix order
    lhs2 = join(s1, meet(s2, s1s2))
    rhs2 = meet(join(s1,s2), join(s1,s1s2))
    if kind == 'prefix':
        assert lhs2 == s1 and rhs2 == s1s2 and lhs2 != rhs2
        print(f"C prefix mirror: {lhs2} != {rhs2}  OK")
print("C weak order conventions  OK")

# transitivity of the normalisation-preserving symmetry group: p -> sigma o p o rho^{-1}
perms = list(permutations(range(3)))  # index i -> value perms[i]
def act(p, sigma, rho):
    inv_rho = [rho.index(i) for i in range(3)]
    return tuple(sigma[p[inv_rho[i]]-1]+1 for i in range(3))
p0 = (1,2,3)
orbit = {act(p0, s, r) for s in perms for r in perms}
print("C orbit size:", len(orbit), orbit)
assert orbit == set(c[3:] for c in CC)
# conjugation fixed points
fixed = [p for p in S3 if all(tuple(s[p[s.index(i)]-1]+1 for i in range(3)) == p for s in perms)]
print("C conjugation fixed points:", fixed)
assert fixed == [(1,2,3)]
assert w0 == (3,2,1) and inv_count(w0) == 3 and w0[0] == 3 and w0[1] == 2  # (13), a transposition
print("C transitivity + conjugation fixed point  OK")

# ---------- Stanley identities on K2 and K3 ----------
def orientations(n, edges):
    for signs in product((0,1), repeat=len(edges)):
        arcs = [(a,b) if s == 0 else (b,a) for (a,b),s in zip(edges,signs)]
        yield arcs

def acyclic(n, arcs):
    # reachability check via Floyd-Warshall
    R = [[False]*n for _ in range(n)]
    for a,b in arcs: R[a][b] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                R[i][j] = R[i][j] or (R[i][k] and R[k][j])
    return not any(R[i][i] for i in range(n))

def count_maps(n, arcs, t, strict):
    tot = 0
    for f in product(range(1,t+1), repeat=n):
        if all((f[a] < f[b] if strict else f[a] <= f[b]) for a,b in arcs):
            tot += 1
    return tot

for n, edges, chi in ((2, ((0,1),), lambda t: t*(t-1)),
                      (3, tuple(combinations(range(3),2)), lambda t: t*(t-1)*(t-2))):
    AO = [a for a in orientations(n, edges) if acyclic(n,a)]
    print(f"K{n}: #acyclic orientations =", len(AO))
    assert len(AO) == (-1)**n * chi(-1)
    for t in range(1, 6):
        dec = sum(count_maps(n,a,t,True) for a in AO)
        rec = sum(count_maps(n,a,t,False) for a in AO)
        assert dec == chi(t), (n,t,dec,chi(t))
        assert rec == (-1)**n * chi(-t), (n,t,rec,(-1)**n*chi(-t))
    print(f"K{n}: decomposition chi(t)=sum Omega^o(t) and reciprocity (-1)^n chi(-t)=sum Omega(t) verified t=1..5")
    # the doc's old formula (ii): sum_D t^{#sources(D)}
    for t in (2,3):
        src = sum(t**sum(1 for v in range(n) if not any(b == v for a,b in a)) for a in AO)
        assert src != (-1)**n * chi(-t)
    print(f"K{n}: old formula (ii) sum_D t^#sources fails  OK")

# ---------- minimality among connected graphs on <= 4 vertices ----------
def connected(n, edges):
    A = adj_matrix(n, edges)
    seen = {0}; stack=[0]
    while stack:
        x = stack.pop()
        for y in range(n):
            if A[x][y] and y not in seen:
                seen.add(y); stack.append(y)
    return len(seen) == n

def max_cliques_of(n, edges):
    out=[]
    for r in range(1,n+1):
        for S in combinations(range(n),r):
            if all((a,b) in edges or (b,a) in edges for a,b in combinations(S,2)):
                if not any(set(S) <= set(T) for T in out):
                    out = [T for T in out if not set(T) <= set(S)]
                    out.append(S)
    return out

print("\nConnected graphs on 3/4 vertices (chi=omega), normalised colouring sets:")
for n in (3,4):
    all_pairs = list(combinations(range(n),2))
    for mask in range(1<<len(all_pairs)):
        edges = tuple(p for i,p in enumerate(all_pairs) if mask>>i & 1)
        if not connected(n,edges): continue
        w = clique_number(n,edges); ch = chromatic(n,edges)
        if ch != w: continue
        K = list(max(max_cliques_of(n,edges), key=len))
        Cs = normalised(n,edges,ch,K)
        pw_chain = all(pleq(a,b) or pleq(b,a) for a,b in combinations(Cs,2))
        parts = [partitions(c) for c in Cs]
        rf_chain = all(refines(a,b) or refines(b,a) for a,b in combinations(parts,2))
        # is it an interval graph? chordal (no induced C4+) -- small brute check
        print(f"  n={n} edges={edges} chi=omega={ch} |C|={len(Cs)} pointwise_chain={pw_chain} refinement_chain={rf_chain}")
        if n == 4 and not (pw_chain and rf_chain):
            print("    ^^ failure found at n=4")

# ---------- IGMRC sweep DP vs brute force ----------
import random
random.seed(7)

def igmrc_bruteforce(iv, w):
    n=len(iv); e=edges_from_intervals(iv)
    k=max(sum(1 for v in range(n) if iv[v][0] <= t <= iv[v][1]) for t in sorted({x for a in iv for x in a}))
    best=10**9
    for c in product(range(1,k+1), repeat=n):
        if is_proper(c,e):
            best=min(best, sum(w[v]*c[v] for v in range(n)))
    return best,k

def igmrc_dp(iv, w):
    n=len(iv)
    pts=sorted({x for a in iv for x in a})
    k=max(sum(1 for v in range(n) if iv[v][0] <= t <= iv[v][1]) for t in pts)
    starts=defaultdict(list); ends=defaultdict(list)
    for v,(s,e) in enumerate(iv):
        starts[s].append(v); ends[e].append(v)
    # state: tuple of (vertex, colour) sorted by vertex; cost
    cur = {(): 0}
    for t in pts:
        for v in starts[t]:                       # process starts before ends at t
            nxt={}
            for st,cost in cur.items():
                used={col for (_,col) in st}
                for col in range(1,k+1):
                    if col not in used:
                        key=tuple(sorted(st+((v,col),)))
                        nc=cost+w[v]*col
                        if key not in nxt or nc<nxt[key]: nxt[key]=nc
            cur=nxt
        for v in ends[t]:
            merged={}
            for st,cost in cur.items():
                key=tuple((x,c) for (x,c) in st if x!=v)
                if key not in merged or cost<merged[key]: merged[key]=cost
            cur=merged
    return min(cur.values()),k

for trial in range(300):
    n=random.randint(2,7)
    iv=[]
    for _ in range(n):
        a=random.randint(0,9); b=random.randint(a+1,12)
        iv.append((a,b))
    w=[random.randint(1,9) for _ in range(n)]
    bf,k1=igmrc_bruteforce(iv,w)
    dp,k2=igmrc_dp(iv,w)
    assert k1==k2 and bf==dp, (iv,w,bf,dp,k1,k2)
print("\nIGMRC sweep DP == brute force on 300 random interval families  OK")
print("\nALL CHECKS PASSED")

