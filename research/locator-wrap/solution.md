# Line breaking with an indivisible locator — solution

Answers to [`spec.md`](spec.md). All statements below are machine-checked where
that is feasible; [`scripts/verify_locator_wrap.py`](scripts/verify_locator_wrap.py)
re-derives the feasibility criterion, the Φ blow-up, the `OPT = L(j*−1)+1`
formula, both secondary-objective witnesses, the Monge sign of the mixed matrix,
and the Pareto/no-go claims by brute force.

Throughout, indices are 1-based for words, `w_t` is the word and `|w_t|` its
width. For a prefix of length `i`,

```
W_i = Σ_{t=1..i} |w_t|,   C_i = W_i + i,   C_0 = 0,
```

so that for `0 ≤ p < i`

```
B(p+1, i) = C_i − C_p − 1 .
```

`L_B(m)` is the number of blocks produced by greedy wrapping of the first `m`
words at budget `B`; `L(m) := L_A(m)`.

---

## 1. Fact 1, the two-phase algorithm Φ, and paragraphs

### 1.1 Greedy's overrun lemma

**Lemma 1 (structure of greedy blocks).** In greedy wrapping at budget `B₀`, a
block has width `> B₀` iff it is a singleton word whose width is `> B₀`. In
particular the last block overruns `B₀` iff it is a single over-wide word.

*Proof.* Consider the moment greedy starts a block at word `i`.

* If `|w_i| > B₀`, the algorithm places `w_i` alone in a block. Its width is
  `|w_i| > B₀`; no further word can be added.
* If `|w_i| ≤ B₀`, greedy extends the block to the largest `j` with
  `B(i,j) ≤ B₀`. Every added word keeps the span `≤ B₀`, so the finished block
  has width `≤ B₀`; if `j < m` the next word starts a new block.

Hence an overrunning block can only arise in the first case, and then it is
exactly a singleton over-wide word. The last block is a special case. ∎

Corollary: if all word widths are `≤ B₀`, every greedy block has width `≤ B₀`.

### 1.2 Feasibility criterion (Fact 1, first half)

**Claim.** The instance is feasible iff either

* `n = 0` and `s ≤ A`; or
* `n ≥ 1`, `max_t |w_t| ≤ A`, and `|w_n| ≤ A − s`.

*Necessity.* Let a legal layout have blocks `S₁,…,S_L`.

* Every block is an interval of consecutive words, so its width is at least the
  width of each word in it. For `t < L`, `width(S_t) ≤ A`; for the last block
  `width(S_L) + s ≤ A`, hence `width(S_L) ≤ A − s ≤ A`. Therefore every word has
  width `≤ A`.
* The last block is a suffix `w_j,…,w_n`, so its width is at least `|w_n|`.
  Hence `|w_n| + s ≤ width(S_L) + s ≤ A`, i.e. `|w_n| ≤ A − s`.
* If `n = 0` the unique empty line has width `0`, so legality is exactly
  `0 + s ≤ A`.

*Sufficiency.*

* `n = 0, s ≤ A`: the empty line is legal.
* `n ≥ 1`: wrap the first `n−1` words greedily at budget `A`; by the corollary
  every such block has width `≤ A`. Put `w_n` alone on the last line. Its width
  is `|w_n| ≤ A − s`, so `|w_n| + s ≤ A`. This is a legal layout. ∎

### 1.3 Legality equivalence for Φ

Define `Φ(0)` to output the unique empty line. For `n ≥ 1`, Φ first greedily
wraps all words at `A`, obtaining `G₁,…,G_k`. If `B(G_k) + s ≤ A` it stops;
otherwise it drops `G_k` and re-wraps the words of `G_k` at budget `A − s`,
splicing the result after `G₁,…,G_{k−1}`.

**Claim.** Φ's output is legal iff the instance is feasible.

*Proof.* `Φ` legal ⇒ feasible is tautological (a legal layout exists), so by
§1.2 the criterion holds. Conversely assume the criterion: `max_t |w_t| ≤ A` and
`|w_n| ≤ A − s`.

* All greedy blocks `G₁,…,G_k` at budget `A` have width `≤ A`.
* If `B(G_k) + s ≤ A` the output is unchanged and legal.
* Otherwise `B(G_k) > A − s`. Re-wrap the suffix `H = G_k` at budget `A − s`.

  By Lemma 1 any block of this re-wrap that exceeds `A − s` is a singleton word
  of width `> A − s`. Such a word has width `≤ A` (global max), so it is legal
  as an ordinary line. The final block of the re-wrap ends at `w_n`; if it
  overran `A − s`, Lemma 1 would force `w_n` itself to be a singleton of width
  `> A − s`, contradicting `|w_n| ≤ A − s`. Hence the final block has width
  `≤ A − s`, so the final line plus the locator fits.

  All earlier re-wrapped blocks have width `≤ max(A − s, A) = A`. Therefore the
  whole output is legal.

So Φ is legal exactly on feasible instances. ∎

### 1.4 Paragraph structure

Let the message be paragraphs `p₁,…,p_m` separated by hard newlines. Paragraph
`r` has `n_r ≥ 0` words `w^{(r)}_1,…,w^{(r)}_{n_r}`; `n_r = 0` means the
paragraph contributes one empty line. A layout is the product of the
per-paragraph layouts; hard newlines force a break between `p_r` and `p_{r+1}`.

**Budget rule.** Every line has budget `A`, except the last line of `p_m`, which
carries the locator and has body budget `A − s`.

Let `M = max` over all words in all paragraphs.

**Feasibility criterion.**

```
feasible  ⟺  M ≤ A  and  ( (n_m ≥ 1 and |w^{(m)}_{n_m}| ≤ A − s)
                          or (n_m = 0 and s ≤ A) ).
```

*Necessity:* lines in `p₁,…,p_{m−1}` and all non-final lines of `p_m` have
budget `A`, so every word is `≤ A`; the final line of `p_m` contains its last
word, forcing `|w^{(m)}_{n_m}| ≤ A − s` (or `s ≤ A` if that paragraph is empty).

*Sufficiency:* wrap each `p_r`, `r < m`, greedily at `A` (empty paragraph → one
empty line). For `p_m` apply the single-paragraph construction of §1.2: wrap the
first `n_m − 1` words greedily at `A`, then put the last word alone on the final
line; if `n_m = 0`, the empty line carries the locator. Hard newlines do not
couple the paragraphs, so this is legal.

**Paragraph version of Φ.** Wrap `p₁,…,p_{m−1}` greedily at `A`; apply the
single-paragraph Φ to `p_m`. By §1.3 the whole output is legal iff the
paragraph criterion above holds.

**Additivity of line counts.** For any layout the total number of lines is the
sum of the lines used per paragraph, because hard newlines make the product
decomposition independent. Define

```
λ_r = 1                             if n_r = 0
    = L_A(n_r)                      if n_r ≥ 1, r < m
```

and, for the last paragraph, with `j*_m = min{ j : B_m(j, n_m) ≤ A − s }`,

```
λ_m = 1                             if n_m = 0, s ≤ A
    = L_A(j*_m − 1) + 1             if n_m ≥ 1
```

(the single-paragraph optimum of §2). Then

```
OPT_paras = Σ_{r=1..m} λ_r .
```

Every paragraph layout contributes at least `λ_r` lines, and concatenating the
optimal per-paragraph layouts attains the sum while respecting the budget rule:
all lines before `p_m`'s last line have budget `A`, and only `p_m`'s last line
has budget `A − s`.

---

## 2. Optimal line count and the O(n) algorithm

### 2.1 A subtlety about j*

As written, `j* = min{ j : B(j,n) ≤ A − s }` is nonempty iff `|w_n| ≤ A − s`,
because `B(n,n) = |w_n|`. But full feasibility also requires
`max_t |w_t| ≤ A`. If some earlier word is wider than `A`, the set can be
nonempty while no legal layout exists. Therefore:

`j*` and `OPT = L(j*−1) + 1` are used only for **feasible** instances; if the
instance is infeasible, `OPT = ∞` and `j*` is undefined by convention. Under
feasibility `max_t |w_t| ≤ A` and `|w_n| ≤ A − s`, so `j*` exists and the whole
analysis is valid.

For `n = 0`, feasible means `s ≤ A` and `OPT = 1`; there is no `j*`.

### 2.2 Greedy optimality and monotonicity of L

**Lemma 2 (greedy minimises block count).** If all word widths are `≤ A`, then
`L(m)` is the minimum possible number of blocks of width `≤ A` covering the
first `m` words.

*Proof (exchange).* Let greedy's first block be `[1,g]`, maximal with
`B(1,g) ≤ A`. Take any legal layout of the `m` words into `ℓ` blocks and let its
first block be `[1,r]`. Maximality of `g` gives `r ≤ g`. Move the words
`r+1,…,g` from the front of later blocks into the first block; removing a prefix
from a block cannot increase its width, and empty blocks are deleted, so the
number of blocks does not increase. We now have a layout whose first block is
`[1,g]`. The remaining words `g+1,…,m` are exactly the input on which greedy
continues, so by induction they need at least `L(m) − 1` blocks. Hence
`ℓ ≥ L(m)`. ∎

**Monotonicity.** `L(m)` is nondecreasing in `m`, and in fact
`L(m+1) ∈ {L(m), L(m)+1}`. Let the last greedy block of the first `m` words
start at `q`. When `w_{m+1}` is appended either `B(q, m+1) ≤ A`, in which case
that last block is extended and the count stays `L(m)`, or a new block starts
and the count becomes `L(m)+1`. Further words can only keep or increase the
count. Hence `L(j−1) ≥ L(j*−1)` for `j ≥ j*`.

### 2.3 Proof of `OPT = L(j*−1) + 1`

Assume feasibility. Note

```
B(j+1, n) = B(j, n) − |w_j| − 1 ≤ B(j, n) − 2 ,
```

so `B(j,n)` is strictly decreasing as `j` increases. Hence the set of starts `j`
whose suffix fits the last line is exactly `{ j*, j*+1, …, n }`.

Let any legal layout have last block `w_j,…,w_n`. Then `B(j,n) ≤ A − s`, so
`j ≥ j*`. Its prefix `w₁,…,w_{j−1}` is covered by `L − 1` ordinary budget-`A`
blocks, so by Lemma 2 `L − 1 ≥ L(j−1) ≥ L(j*−1)` using monotonicity. Therefore
`L ≥ L(j*−1) + 1`.

Conversely take `j = j*`. Wrap `w₁,…,w_{j*−1}` greedily at `A` (legal because
`max ≤ A`), giving `L(j*−1)` lines, and put `w_{j*}…w_n` on the last line; its
width is `B(j*, n) ≤ A − s`. This layout is legal with `L(j*−1) + 1` lines.
Hence `OPT = L(j*−1) + 1`. ∎

### 2.4 O(n) implementation

One backward pass finds `j*` and checks feasibility; one greedy pass over the
prefix computes `L(j*−1)`.

```python
def wrap_locator(w, A, s):
    n = len(w)
    if n == 0:
        return 1 if s <= A else INFEASIBLE

    cur = 0            # becomes B(j, n) while scanning j downwards
    jstar = None
    maxw = 0
    for j in range(n, 0, -1):
        maxw = max(maxw, w[j - 1])
        cur = w[j - 1] + (1 if j < n else 0) + cur      # = B(j, n)
        if cur <= A - s:
            jstar = j                                   # last assignment = smallest feasible j

    if maxw > A or jstar is None:
        return INFEASIBLE                               # jstar is None iff w_n > A - s

    m = jstar - 1
    cnt, i = 0, 1
    while i <= m:
        cnt += 1
        cur = w[i - 1]
        i += 1
        while i <= m and cur + 1 + w[i - 1] <= A:
            cur += 1 + w[i - 1]
            i += 1

    return cnt + 1
```

Correctness of the backward accumulation: at `j = n`, `cur = |w_n|`; for `j < n`,
`cur ← |w_j| + 1 + cur = B(j,n)`. Since `cur` increases as `j` decreases, the
last `j` with `cur ≤ A − s` is exactly `j*`. Checking `maxw` handles the
`max_t |w_t| ≤ A` condition. The greedy pass is `O(m) ≤ O(n)`. Total time `O(n)`,
extra space `O(1)` beyond the input.

---

## 3. Secondary objective

For a layout with `L` lines the slack of line `t` is `β_t − width(S_t)`, where
`β_t = A` for `t < L` and `β_L = A − s`. The penalty is the sum of squared
slacks.

**Invariant.** For `n ≥ 1` and fixed `L` the total slack is constant:

```
Σ_{t=1..L} β_t = L·A − s ,
Σ_{t=1..L} width(S_t) = Σ_{t=1..n} |w_t| + (n − L) ,
```

so

```
Σ_{t=1..L} (β_t − width(S_t)) = L(A + 1) − Σ_{t=1..n} |w_t| − n − s .
```

Thus minimising the sum of squares means distributing a fixed total slack as
evenly as possible. Greedy plus the longest feasible suffix typically produces
very uneven slacks.

### 3(a) Q2's solution is not secondary-optimal — two independent witnesses

**Witness 1: freedom in the last block's start (`L` flat on `[j*, j']`).**

```
n = 3,  w = (1,1,1),  A = 4,  s = 1,  A − s = 3 .
```

* `B(1,3) = 5 > 3`, `B(2,3) = 3 ≤ 3`, `B(3,3) = 1 ≤ 3`, so `j* = 2`;
  `L(1) = 1`, `OPT = 2`.
* `L(2) = 1` as well (`B(1,2) = 3 ≤ 4`), so `L` is flat on `[2,3]`.

Q2's layout (`j* = 2`): line 1 `[w₁]` width 1 slack 3; last line `[w₂,w₃]`
width 3 slack 0. Penalty `3² + 0² = 9`.

Alternative (`j = 3`): line 1 `[w₁,w₂]` width 3 slack 1; last line `[w₃]` width 1
slack 2. Penalty `1² + 2² = 5 < 9`. Both use 2 lines. ∎

**Witness 2: freedom in a non-greedy prefix (`j*` forced).**

```
n = 4,  w = (2,1,1,1),  A = 4,  s = 2,  A − s = 2 .
```

* `B(4,4) = 1 ≤ 2` but `B(3,4) = 3 > 2`, so `j* = 4` is forced.
* Prefix `(2,1,1)`: greedy at `A = 4` gives `[w₁,w₂]` (width 4) and `[w₃]`
  (width 1), so `L(3) = 2`, `OPT = 3`.

Q2's layout: `[w₁,w₂]` slack 0; `[w₃]` slack 3; `[w₄]` slack 1.
Penalty `0 + 9 + 1 = 10`.

Another 3-line layout: `[w₁]` (width 2, slack 2), `[w₂,w₃]` (width 3, slack 1),
then `[w₄]` (slack 1). Penalty `4 + 1 + 1 = 6 < 10`. Both use exactly `OPT = 3`
lines, and here the last block's start is forced. ∎

So both sources of freedom — a flat `L` interval for the last start, and
non-greedy `L`-block prefix layouts — can independently make Q2's solution
strictly suboptimal.

### 3(b) DP over `(i,k)`

Let `K = OPT − 1` be the required number of ordinary budget-`A` lines. Define

```
opt₂(i,k) = minimum squared-slack penalty for the first i words
            using exactly k blocks of budget A .
```

Base cases `opt₂(0,0) = 0`, `opt₂(i,0) = ∞` for `i > 0`,
`opt₂(0,k) = ∞` for `k > 0`. For `k ≥ 1`

```
opt₂(i,k) = min_{0 ≤ p < i, B(p+1,i) ≤ A} [ opt₂(p,k−1) + (A − B(p+1,i))² ] .
```

The last body block is `w_{p+1},…,w_i`; its penalty is `(A − B)²`.

**Correctness.** By induction on `k`. Any layout of the first `i` words into `k`
body blocks has a last body block `[p+1,i]` with `B(p+1,i) ≤ A`; the first `p`
words use `k−1` blocks and cost at least `opt₂(p,k−1)`. Conversely, appending
`[p+1,i]` to an optimal `(k−1)`-block layout of the first `p` words attains the
displayed value. Taking the minimum gives the optimum.

Finally append the special last line. For a prefix length `i` (`0 ≤ i ≤ n−1`)
the last block is `w_{i+1},…,w_n`, legal iff `B(i+1,n) ≤ A − s`, with penalty
`(A − s − B(i+1,n))²`. Hence

```
ANS = min_{0 ≤ i ≤ n−1, B(i+1,n) ≤ A−s}
        [ opt₂(i, K) + (A − s − B(i+1,n))² ] .
```

This uses exactly `K + 1 = OPT` lines and is legal.

**Complexity.** Naively `i = 1..n`, `k = 1..K`, `p = 0..i−1`: `O(n²K)` time,
`O(nK)` space. With `C_i = W_i + i` and `D_p = C_p + A + 1` we have
`B(p+1,i) = C_i − C_p − 1` and body penalty `(D_p − C_i)²`. For fixed `k−1` the
recurrence is a sliding-window minimum of lines

```
f(p) + (D_p − C_i)² = C_i² − 2·D_p·C_i + (D_p² + f(p)) ,
```

with `D_p` increasing and queries `C_i` increasing, so a monotone convex-hull
trick (deque) evaluates all `i` in `O(n)` per level: `O(nK)` time and `O(n)`
space with rolling arrays. Since `K ≤ n`, this is `O(n²)` in the worst case.

### 3(c) Monge/concavity and the Knuth–Plass speedup

Write the cost of a line from `p+1` to `i` with budget `β` as

```
c_β(p,i) = (β + 1 + C_p − C_i)² = (D_p^β − C_i)² ,   D_p^β = C_p + β + 1 ,
```

valid when `D_p^β ≥ C_i`. For body rows `β = A`; for the final row
`β_f = A − s`. Note `D_p^{A−s} = D_p^{A} − s`, so the final cost is the body cost
evaluated at a shifted coordinate:

```
c_{A−s}(p,n) = (D_p^{A} − (C_n + s))² .
```

**The Monge inequality survives.** Let rows `i < i'` have budgets `β_i, β_{i'}`
and columns `p < p'`. Put

```
δ = C_{p'} − C_p > 0 ,  ε = C_{i'} − C_i > 0 ,  Δ = β_{i'} − β_i ,  u = D_p^{β_i} − C_i .
```

Then

```
c_{β_i}(p,i)   = u² ,
c_{β_i}(p',i)  = (u + δ)² ,
c_{β_{i'}}(p,i')  = (u + Δ − ε)² ,
c_{β_{i'}}(p',i') = (u + δ + Δ − ε)² .
```

The Monge difference (`RHS − LHS`) is

```
[ c(p',i) + c(p,i') ] − [ c(p,i) + c(p',i') ] = 2·δ·(ε − Δ) .
```

* Body–body: `Δ = 0` ⇒ `2δε > 0`.
* Body–final: `Δ = −s` ⇒ `2δ(ε + s) > 0`.

Hence for all valid `p < p'`, `i < i'`

```
M[i,p] + M[i',p'] ≤ M[i,p'] + M[i',p] ,
```

i.e. the mixed matrix is Monge / totally monotone, even after adding column
constants `f(p)` and a per-line constant `λ` (both cancel in the quadrangle
difference).

**Feasibility restrictions** (`B ≤ β`, i.e. `C_p ≥ C_i − β − 1`) give column
intervals `[ℓ_i, u_i]` with `ℓ_i` and `u_i` nondecreasing in `i`; for the final
row `C_n + s > C_{n−1}`, so its left endpoint is also `≥` that of row `n−1`.
This staircase preserves total monotonicity with `∞` entries. Therefore row
minima are attained at nondecreasing column indices — exactly the SMAWK /
Knuth–Plass condition.

**Interpretation.** In the slack variable the cost `φ(u) = u²` is convex, and
that convexity is what yields the Monge inequality. (In the usual Knuth–Plass
badness parameterisation the same hypothesis is often phrased as concavity; the
invariant is the quadrangle/Monge inequality above.) The last-line perturbation
introduces no new cost function — it only translates the final query point from
`C_n` to `C_n + s`. One budget *decrease* is Monge-compatible; a budget increase
larger than `C_{i'} − C_i`, or several different special budgets, would destroy
it.

**Can the speedup be retained? Yes.**

1. *Per DP level.* The recurrence of §3(b) is a Monge row-minima problem; online
   SMAWK / the convex-hull trick computes each level in `O(n)` (`O(n log n)`
   with a generic CHT). The layered DP then costs `O(n·OPT)` instead of
   `O(n²·OPT)`.
2. *Whole problem in `O(n)` / `O(n log n)`.* The `OPT` factor comes only from the
   hard cardinality constraint. Add a large constant `λ > n·A²` per line and
   solve the unconstrained shortest-path DP

```
f(0) = 0 ,   f(i) = min_{p<i} [ f(p) + λ + c_{β_i}(p,i) ] ,
```

   with body rows `i < n` (`β = A`) and final row `i = n` (`β = A − s`).
   Because `λ` exceeds the maximum possible total penalty difference, the optimum
   first minimises the number of lines, then the secondary penalty among those.
   The matrix is the same mixed Monge matrix and the final row is the shifted
   query `x = C_n + s`. An online monotone CHT (deque) inserts one line per `p`,
   removes infeasible prefixes, and answers the strictly increasing queries
   `C₁ < C₂ < … < C_{n−1} < C_n + s` in `O(n)` total time and `O(n)` space.
   (`n = 0` is the trivial empty line.)

So the asymmetric last line does not destroy the Knuth–Plass speedup: the
one-dimensional `min_j` is precisely the last row of a totally monotone matrix.

---

## 4. Policies in the infeasible region

**Standing assumption.** To keep the policies meaningful assume
`max_t |w_t| ≤ A` (otherwise the body already has unavoidable over-wide blocks
and the loss vector needs extra coordinates). The infeasible region is

```
|w_n| > A − s          (n ≥ 1) ,   or  n = 0 with s > A .
```

Set

```
τ = |w_n| + s − A > 0 ,   ℓ = L(n − 1) + 1 ,   Δ = L(n) − L(n−1) ∈ {0,1} .
```

`Δ = 0` iff word `n` fits into the last greedy block of the first `n−1` words;
equivalently, if that block starts at `a`, `B(a, n−1) + 1 + |w_n| ≤ A`. Since
`|w_n| ≥ A − s + 1`, `Δ = 0` forces `B(a,n−1) ≤ s − 2` (so it needs `s ≥ 2`).
Also `L(n) + 1 = ℓ + Δ`.

### 4.1 Optimal layout of each policy

Let a general body layout have last block start `j`; its width is
`W_j = B(j,n)`. Legality of body lines at `A` requires `W_j ≤ A`; since
`|w_n| ≤ A`, `j = n` is always allowed, with `W_n = |w_n|`. Because `W_j` is
strictly increasing as `j` decreases, `W_j ≥ |w_n|`, with equality iff `j = n`.

**P1 — ignore the discount.** Violation is `W_j + s − A`; minimising it means
minimising `W_j`, so the unique optimum is

```
S_L = { w_n }      (for n = 0: the empty line) .
```

The prefix `w₁,…,w_{n−1}` is wrapped greedily at `A`, giving `L(n−1)` lines.
Canonical loss vector:

```
P1 = (τ, ℓ, 0) .
```

**P2 — locator on its own line.** Requires `s ≤ A`. The body is wrapped greedily
at `A` in `L(n)` lines; one extra line holds the full locator. Canonical loss
vector:

```
P2 = (0, ℓ + Δ, 0) .
```

If `s > A`, P2 is not an admissible policy.

**P3 — truncate the locator.** With last block width `W`, the last line has
`A − W` free columns; show `min(s, A − W)` characters, truncating
`s − (A − W) = W + s − A` characters. Again minimising truncation forces
`W = |w_n|`, i.e. `{w_n}` last, prefix greedy. Canonical loss vector:

```
P3 = (0, ℓ, τ) .
```

For `n = 0`, `s > A`: `τ₀ = s − A`, `ℓ₀ = 1`, `P1 = (τ₀, 1, 0)`,
`P3 = (0, 1, τ₀)`, P2 unavailable.

### 4.2 Mutual Pareto dominance

Lower is better; `x` dominates `y` iff `x_i ≤ y_i` for all `i` and `<` for at
least one.

| region | P1 | P2 | P3 | dominance |
|---|---|---|---|---|
| `s ≤ A`, `Δ = 0` | `(τ,ℓ,0)` | `(0,ℓ,0)` | `(0,ℓ,τ)` | P2 strictly dominates P1 and P3; P1 and P3 incomparable |
| `s ≤ A`, `Δ = 1` | `(τ,ℓ,0)` | `(0,ℓ+1,0)` | `(0,ℓ,τ)` | all three pairwise incomparable |
| `s > A` | `(τ,ℓ,0)` | not admissible | `(0,ℓ,τ)` | P1 and P3 incomparable; `n = 0` is the special case above |

When `Δ = 0`, P2 has the same line count as P1/P3 but zero violation and zero
truncation. When `Δ = 1`, P2 buys zero violation/truncation with one extra line,
P1 buys fewer lines with `τ` violated columns, and P3 buys fewer lines with `τ`
truncated characters; no vector is componentwise `≤` another with a strict
improvement.

### 4.3 Is there a strictly Pareto-better hybrid?

When all three policies are defined (`s ≤ A`): **no**.

Allow the most general hybrid: arbitrary body wrapping, arbitrary splitting of
the locator across the last body line and/or extra own lines, overflow, and
truncation. Let its loss be `(V, L, T)`.

To dominate P1 we need `T ≤ 0`, hence `T = 0` (full locator shown). To dominate
P3 we need `V ≤ 0`, hence `V = 0` (no overflow). To dominate P2 we need
`V = 0`, `T = 0`, and a strict improvement somewhere; since `V = T = 0` already
match P2, the strict component must be `L < L(n) + 1 = ℓ + Δ`.

But any output with `V = 0` and `T = 0` has:

* at least `L(n)` lines containing body words, because with zero overflow every
  body block has width `≤ A` and greedy minimises their number;
* at least one extra locator-only line, because the last body block has width
  `W ≥ |w_n| > A − s`, so `W + s > A` and the full locator cannot fit on the last
  body line without overflow. Even if we place `A − W < s` characters there,
  `s − (A − W) > 0` characters remain and need another line.

Therefore `L ≥ L(n) + 1 = ℓ + Δ`, contradicting the required strict inequality.
Hence no hybrid — not even one that splits the locator — can dominate P2, and
consequently none can dominate all three. In particular "partial truncation"
cannot work, because `T > 0` already fails to dominate P1.

**Remark on `s > A`.** Literally, P2 is not a policy there, so "better than all
three" is undefined. Among the two admissible policies P1 and P3 a split-locator
hybrid can be better:

```
w = (1,1) ,  A = 4 ,  s = 5 (> A) ,  n = 2 .
```

Here `L(2) = 1`, `L(1) = 1`, `ℓ = 2`, `τ = 1 + 5 − 4 = 2`.

* P1: `(2, 2, 0)`; P3: `(0, 2, 2)`.
* Hybrid: both words on one line (width 3), append 1 locator character to that
  line (width 4, no overflow), and put the remaining 4 characters on a second
  line (width 4). All 5 characters are shown, with no truncation, no overflow,
  and 2 lines: `(0, 2, 0)`.

This dominates P1 and P3. If one artificially extends P2 to `s > A` by letting
its own line overflow by `s − A = 1`, its loss would be `(1, 2, 0)`, which
`(0, 2, 0)` also dominates. (The example splits the locator across a line
break; with the literal requirement `s ≤ A` for P2, the no-go theorem above is
the complete answer for the region where all three policies exist.)

