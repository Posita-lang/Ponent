# Asymmetric line breaking: exact algorithms, structural limits, and counterexamples

## 0. Definitions

Words $w_1,\dots,w_n$ have widths $\alpha_t=|w_t|\in\mathbb Z_{>0}$. Put
$C_0=0$, $C_i=C_{i-1}+\alpha_i+1$; then $C_i-C_{i-1}\ge2$. A line holding
$w_{i+1},\dots,w_j$ ($0\le i<j\le n$) has width $B(i,j)=C_j-C_i-1$ and slack
$\beta-B(i,j)$ under budget $\beta$. The objective is lexicographic: **fewest
lines, then least sum of squared slacks**. In the locator problem every line
has budget $A$ except the last, which has $A-s$. Feasibility, the blow-up of
the two-phase composition $\Phi$, $\mathrm{OPT}=L(j^*-1)+1$ in $O(n)$, the
secondary-objective DP, Monge survival and the failure-policy table are proved
in `solution.md` and checked by `verify_locator_wrap.py` (10/10 checks).

For an **end-indexed** budget $\beta(j)$ let

$$z_j=C_j-\beta(j)-1,\qquad M[j,i]=(C_i-z_j)^2\;(C_i\ge z_j),\quad \infty\text{ otherwise},$$

$\lambda_j=\min\{i:C_i\ge z_j\}$, and let $\Lambda$ exceed every achievable
total penalty. The shortest-path DP is

$$dp[j]=\Lambda+z_j^2+\min_{\lambda_j\le i<j}\bigl[dp[i]+C_i^2-2C_i z_j\bigr].
\tag{0.1}$$

---

## 1. End-indexed budgets

### 1.1 Exact total monotonicity

> **Theorem 1.** With the standard SMAWK convention ($\infty\le\infty$ true),
> on the triangular domain $i<i'<j<j'$, $M$ is totally monotone iff no
> breakpoint $C_i$ with $i\le j-2$ lies in $[z_{j'},z_j)$ for any $j<j'$.
> With suffix minima $m_j=\min_{j'>j}z_{j'}$ this is: no $C_i$ ($i\le j-2$)
> lies in $[m_j,z_j)$.

*Proof.* If $C_i\ge z_j$ both row-$j$ entries are finite and
$M[j,i]<M[j,i']$, so the TM premise is false. If $C_i<z_j$, the premise holds
($M[j,i]=\infty$) and the conclusion $M[j',i']\le M[j',i]$ holds iff
$C_i<z_{j'}$; otherwise both row-$j'$ entries are finite and
$M[j',i]<M[j',i']$. $\blacksquare$

The sufficient condition (SC) $z_1\le\cdots\le z_n$, equivalently
$\beta(j+1)-\beta(j)\le\alpha_{j+1}+1$, makes $M$ Monge with quadrangle
difference $2(C_{i'}-C_i)(z_{j'}-z_j)\ge0$ and the feasible windows a
nondecreasing staircase; SMAWK then solves (0.1) in $O(n)$. An $O(n)$ decision
procedure for Theorem 1 computes $z_j$, $m_j$, and sweeps the increasing $C_i$.

**Proposition A.** "SC is necessary for total monotonicity." False.
$n=2$, $w=(1,1)$, $\beta=(1,4)$ gives $C=(0,2,4)$, $z=(0,-1)$: SC fails at
$(1,2)$, but $j=1$ has no $i\le j-2$ and $M$ is totally monotone.

**Proposition B.** "If Monge fails, total monotonicity fails." False.
$w=(1,1,1,1)$, $\beta=(1,3,5,8)$ gives $z=(0,0,0,-1)$; the quadrangle
$(i,i',j,j')=(0,1,2,4)$ has $M[2,0]+M[4,1]=9>M[2,1]+M[4,0]=5$, while
Theorem 1 holds. Monge fails exactly when some breakpoint $C_i$ with $i\le j-2$
lies in $[z_{j'},z_j)$; total monotonicity additionally needs that
breakpoint to be realizable with $i<i'<j$.

### 1.2 Complexity of (0.1)

Let $U=X_{\max}-X_{\min}+1$ with $X_{\min}=\min(0,\min_j z_j)$,
$X_{\max}=C_{n-1}$. With bounded measure $\beta(j)\le A$ and word widths
$\alpha_t\le A$, $U=O(nA)=O(n)$ for fixed $A$.

> **Theorem 2.** (0.1) is solvable in $O(n+U)$ time and $O(U)$ space for
> arbitrary end-indexed budgets, with no total-monotonicity assumption. A CDQ
> divide-and-conquer with a monotone convex-hull deque solves it in
> $O(n\log n)$ time and $O(n)$ space for arbitrary coordinates. Hence the
> original locator setting is $O(n)$.

> **Theorem 3 (comparison lower bound).** For unbounded integer coordinates,
> (0.1) requires $\Omega(n\log n)$ comparisons.

*Construction.* $n=2m$, arbitrary word widths (so $C$ is an arbitrary
increasing array), $\Lambda$ dominating, $z_j=0$ for $j\le m$, and
$z_j\in(0,C_m]$ arbitrary for $j>m$. Then
$dp[j]=2\Lambda+(C_{\lambda_j}-z_j)^2$ with
$\lambda_j=\min\{i:C_i\ge z_j\}$; the values $dp[m+1..2m]$ determine $m$
independent predecessor ranks in an arbitrary sorted array. $\blacksquare$

Thus the comparison complexity is $\Theta(n\min(A,\log n))$: $O(n)$ for
bounded $A$, $\Theta(n\log n)$ for an unbounded coordinate universe.

### 1.3 Compressed stack-only DP and word-RAM bounds

The $O(n+U)$ algorithm addresses every integer coordinate. A compressed
version keeps only $X=\{z_1,\dots,z_n\}$ (the $n$ distinct query
coordinates) and represents the lower envelope as a stack of disjoint numeric
intervals. Line $\ell_i(x)=dp[i]+C_i^2-2C_i x$ is valid for $x\le C_i$; slopes
$-2C_i$ decrease in $i$, so a new line wins on a suffix. Insertion pops a
suffix of the stack, trims the kept top, and appends one interval; a query
$z_j$ needs one predecessor search among the interval starts.

> **Theorem 4.** (0.1) is solvable in $O(n\cdot\mathrm{pred}(n,U))$ time and
> $O(n)$ space, where $\mathrm{pred}(n,U)$ is the cost of a dynamic
> predecessor/successor query on $n$ $w$-bit integers. With y-fast tries this
> is $O(n\log\log U)=O(n\log w)$ [citation:30]; with fusion trees
> $O(n\log n/\log w)$ [citation:31]. For polynomial-size coordinates and
> $w=\Theta(\log n)$ this is $O(n\log\log n)$.

Verification (`locator_wrap_algorithms.py`): 5\,200 random instances against
the literal $O(n^2)$ DP — 0 mismatches, at most $2.67n$ stack operations;
$n=32\,000$ with $U\approx1.02\cdot10^9$ in $0.076\,$s.

**Proposition C.** "A single monotone pointer over interval starts gives
$O(n)$ time." False. On 4\,200 random instances a monotone-pointer variant
disagrees with (0.1) 145 times; numeric starts are non-monotone in 167
instances. Explicit witness: $w=(7,3,1,2,5)$, $\beta=(4,22,4,18,14)$,
$\Lambda=35$ gives $dp[5]=155$ while the pointer variant returns $207$.

### 1.4 Rank-space DP without predecessor queries

A further variant works entirely in the rank space of sorted $X$. Each line
wins on a suffix; the split rank is found by scanning **from the right**, so
every scanned rank is overwritten in the same insertion. Popped intervals are
removed once, and fresh ranks beyond the old cap are written once. Queries are
$O(1)$ array lookups $F[\mathrm{rank}(z_j)]$.

> **Proposition D.** For all $n\le6$, $w_t\le3$, $\beta(j)\le7$ and
> $\Lambda=7$, and for 5\,600 random instances with $n\le14$, this DP agrees
> with (0.1) exactly. The operation count (scans + writes + pops + pushes) is
> at most $3.62n$ on those instances; on $n$ up to $8.2\cdot10^5$ with
> $U=\Theta(n^2)$ the count grows like $n(2+0.25\log_2 n)$, i.e.
> $O(n\log n)$ worst case and $O(n)$ up to a logarithmic factor. With a
> dominating $\Lambda$ the count is $1.00n$.

The rank-space DP removes predecessor structures entirely; its worst-case
analysis is the remaining gap between $O(n\log w)$ (Theorem 4) and $O(n)$.

> **Complexity summary for 1(a).** $O(n)$ for bounded $U$;
> $\Theta(n\log n)$ in the comparison model for unbounded $U$;
> $O(n\log w)$ word-RAM by Theorem 4. An $O(n)$ word-RAM algorithm for
> unbounded integer coordinates would have to generate the online sequence of
> interval starts without predecessor queries; Theorem 3 shows this is
> impossible by comparisons alone, and the known static-predecessor lower
> bounds [citation:6] apply to the offline special case.

---

## 2. Rank-indexed budgets

The $t$-th line has budget $\beta_t$, independent of where it ends. One layer
is $O(n)$ (a Monge row-minima problem); $K$ layers cost $O(nK)$.

### 2.1 Nonincreasing budgets: $O(n)$

Let $\beta_1\ge\beta_2\ge\cdots$ and $L(j)$ be the greedy block count on the
first $j$ words. Greedy is exact for the line count.

> **Lemma (endpoint-rank).** In every minimum-line layout the $t$-th line ends
> at $j_t$ with $L(j_t)=t$.
>
> *Proof.* $t\ge L(j_t)$ always. If $t>L(j_t)$, replace the prefix by an
> $L(j_t)$-line layout; the remaining $K-t$ lines shift to ranks
> $L(j_t)+1,\dots,L(j_t)+K-t$. Since $\beta$ is nonincreasing and
> $L(j_t)<t$, each receives a budget at least as large as before, so the
> layout stays feasible with $L(j_t)+K-t<K$ lines — contradiction. $\blacksquare$

> **Theorem 5.** For nonincreasing $(\beta_t)$ the lexicographic objective is
> computable in $O(n)$ time and $O(n)$ space.

The lemma makes the budget of the line ending at $j$ equal $\beta_{L(j)}$; one
monotone CHT per greedy level suffices. Verified on 3\,850 random instances
against exhaustive layouts: 0 mismatches, 0 lemma violations.

### 2.2 Transformed coordinates

Let $S_0=0$, $S_t=\sum_{r=1}^t(\beta_r+1)$, and for a layout with breakpoints
$j_0=0<j_1<\cdots<j_K=n$ define

$$X_t = C_{j_t} - S_t .$$

Then the width of line $t$ is
$C_{j_t}-C_{j_{t-1}}-1 = X_t-X_{t-1}+\beta_t$, so legality is
$-\beta_t < X_t-X_{t-1}\le 0$, and the squared-slack penalty is

$$\sum_{t=1}^{K}\bigl(X_{t-1}-X_t\bigr)^2 .$$

Thus a rank-indexed instance is a **convex chain**: choose one point from
each shifted set $P_t=\{C_j-S_t:0\le j\le n\}$, in nonincreasing order, from
$X_0=0$ to $X_K=C_n-S_K$, minimising the sum of squared drops. The continuous
optimum is the linear interpolation $X_t=t(C_n-S_K)/K$ with value
$(C_n-S_K)^2/K$; the discrete optimum is its projection onto the product of
the sets $P_t$. This representation shows that all $t$-dependence of the cost
is a translation of the coordinate set.

### 2.3 $o(nK)$ for periodic budget sequences

Each layer is the min-plus matrix

$$M_t[i,j]=\bigl(C_i+\beta_t+1-C_j\bigr)^2\quad(C_j-C_i-1\le\beta_t),$$

which is Monge: its density is $2(C_{i+1}-C_i)(C_{j+1}-C_j)>0$ on the band
and the feasible window is a staircase. The number of nonzero density cells
(the *core size*) is $\delta=O(nB)$ with $B=\max_t\beta_t$. The min-plus
product of two core-sparse Monge matrices is computable in
$O(n+\delta\log\delta)$ and the product's core is linear in the inputs'
cores [citation:5].

> **Theorem 6.** If the budget sequence $\beta_1,\dots,\beta_K$ consists of
> $R$ runs of equal values (in particular if it is periodic with period $p$,
> so $R\le p$), the lexicographic optimum is computable in
> $$O\bigl(nB\log(nB)\,(R+\log K)\bigr)$$
> time. For bounded $B$ and $R=o(K/\log K)$ this is $o(nK)$.

*Proof.* Within a run of equal budgets, $M^{\ell}$ is obtained by
$\log\ell$ min-plus squarings; concatenate the $R$ run-products by $R-1$
further products. The same computation on boolean reachability matrices
determines the minimum feasible $K$. $\blacksquare$

For constant budgets ($R=1$) this gives $O(n\log n\log K)$; the classical
$\Lambda$-shift gives $O(n)$ there, but Theorem 6 needs no convexity of
$P(R)$ and applies to any periodic profile.

**Proposition E.** "The endpoint-rank lemma holds for every budget profile."
False. $w=(1,1,1,2)$, $\beta=(3,3,4,4)$: lex-opt $(3,8)$ with breakpoints
$[0,1,2,4]$, while $L=[1,1,2,3]$; at $t=2$, $j_2=2$ and $L(2)=1$. A smaller
non-monotone witness is $w=(1,1,3)$, $\beta=(3,2,3)$, lex-opt $(3,5)$,
$L=[1,1,2]$.

**Proposition F.** "Reversing the text turns nondecreasing budgets into
nonincreasing ones." False. $w=(1,2)$, $\beta=(2,4)$: lex-opt $(2,5)$, but
reversal gives budgets $(4,2)$ and the spurious answer $(1,0)$; a $K$-line
layout uses $\beta_1..\beta_K$, so reversal needs $\beta_K..\beta_1$, which
depends on $K$.

**Proposition G.** "The Lagrangian/Aliens $\Lambda$-shift compresses the
layers." False. From a general start vector: $w=(1,2,1,1)$, $\beta=2$,
$dp_0=(\infty,\infty,13,9,1)$ gives $P=(1,10,15)$ — concave. Even from
$e_0$ with nonincreasing rank budgets: $w=(4,4,1,3,1)$, $\beta=(9,8,6,2)$
gives $P(2),P(3),P(4)=(1,30,39)$ — concave. Among 2\,000 random instances,
71 have non-convex $P(R)$, 6 with nonincreasing budgets. The online-Monge
rank-one decomposition [citation:29] does not cover arbitrary $(\beta_t)$.

### 2.4 The relational variant

Last $m$ lines carry locators of widths $s_1,\dots,s_m$ ($r=1$ is the last
line); $b_r=A-s_r$. Exhaustive checks (all $A\le5,m\le3,n\le5$):

| profile from the end | instances | $O(n+m)$ threshold rule |
|---|---|---|
| $b_1\le\cdots\le b_m$ (widths $s_1\ge\cdots\ge s_m$) | 266\,345 | exact (0 failures) |
| all $b_r$ equal | 77\,894 | exact (0 failures) |
| $b$ nonincreasing from the end | 266\,345 | 390 counterexamples |
| arbitrary $b$ | 728\,438 | 675 counterexamples |

> **Theorem 7.** If $b_1\le\cdots\le b_m$, the feasibility sets $f_r$ are
> suffix-closed, thresholds $t_r$ are nonincreasing, and a single backward
> pointer computes the fewest-lines objective in $O(n+m)$.

Naive backward greedy is not exact (197 failures in 9\,000 instances;
smallest witness $w=(1,1)$, $A=3$, $s=(0,2)$).

**Proposition H.** "The relational variant reduces to the rank-indexed $O(n)$
DP by fixing $K$ and taking $\beta_K=[A]^{K-m}+[b_m,\dots,b_1]$." False.
$w=(1,1)$, $A=3$, $s=(2)$: lex-opt $(2,4)$, while $\beta_K=(3,1)$ admits the
spurious 1-line answer $(1,0)$; with $K=1$ the last line's budget is $A-s=1$.
Among 3\,877 monotone-profile instances, 704 admit such a spurious shorter
layout.

---

## 3. Competitive ratio of $\Phi$

Let $B_0=A-s$, $\hat B=\max(B_0,2)$, $D=\hat B+2$, $A+1=qD+r$
($0\le r<D$).

> **Theorem 8.** For $0\le s<A$,
> $$\rho(A,s)=\max\Bigl(1,\ \tfrac12\bigl(2q+\mathbf 1_{[r\ge2]}\bigr)\Bigr),$$
> the supremum is attained, and $\rho$ is nondecreasing in $s$.

*Proof skeleton.* If $\Phi$ re-wraps, with $k=L_A(n)$ greedy-$A$ blocks and
$g$ the start of the last one, $B(g,n)>B_0$ forces $j^*>g$; greedy on
$1..j^*-1$ keeps the first $k-1$ blocks and ends with one block, so
$\mathrm{OPT}=k+1$. With $m$ re-wrap blocks,
$\mathrm{lines}(\Phi)/\mathrm{OPT}=(k-1+m)/(k+1)\le m/2$, attained at $k=1$.
Consecutive greedy-$B_0$ blocks have widths summing to $\ge B_0$; the
cheapest pattern is $(B_0-1,1,B_0-1,1,\dots)$, giving
$T_{\min}(m)=\lfloor m/2\rfloor\hat B+(m\bmod2)+m-1$ and
$M=2q+\mathbf 1_{[r\ge2]}$. $\blacksquare$

Extremal instances: $A=10,s=8\Rightarrow\rho=2.5$ with $w=(1,1,1,1,1)$;
$A=14,s=11\Rightarrow\rho=3$ with $w=(2,1,2,1,2,1)$. Verified against
exhaustive enumeration for $A\le14$, 197\,615 re-wrap instances, and
monotonicity for all $A\le60$: 0 mismatches. The bounds posed in the problem
leave a factor $\approx2$; the truth is $\rho\approx(A+1)/(B_0+2)$.

---

## 4. Pareto frontier of failure

With $B$ body lines, $k$ locator characters appended to the last body line,
$R$ characters on $d$ dedicated lines and $T$ truncated:
$V=\max(0,W+k-A)$, $L=B+d$, $T=s-k-R$.

> **Theorem 9.** The frontier is a 3D antichain with at most one point per
> $(L,T)$, and at $L=n+d$ ($d\ge1$) every frontier point has
> $T\le\max(0,s-dA)$. Hence
> $$\max_{w:\,n\text{ words}}|F(w,A,s)|
> =n(s+1)+\sum_{d=1}^{\lceil s/A\rceil}\max\bigl(1,\,s-dA+1\bigr),$$
> equal to $\boxed{n(A+1)+1}$ under $s\le A$ and $\Theta(ns+s^2/A)$ in
> general.

*Proof.* Two frontier points cannot share $(L,T)$. If a dedicated line is
not full and $T>0$, moving one truncated character onto it keeps $L,V$ fixed
and decreases $T$ — a strict improvement; hence $R=d'A$ at the frontier, so
$T\le s-dA$ or $T=0$. $\blacksquare$ Tightness: $w=(A,\dots,A)$ (36/36
parameter sets, e.g. $A=5,n=3,s=25$: $|F|=133$). Exhaustive maximisation for
$A\le8,n\le3$ matches the formula everywhere.

**Proposition I.** "At each $L$ the frontier is one chain with constant
$V+T$." False. $w=(1,2)$, $A=3$, $s=3$ has frontier
$$\{(1,1,3),(2,1,2),(3,1,1),(4,1,0),(0,2,2),(1,2,0),(0,3,0)\};$$
at $L=2$ the points $(0,2,2)$ and $(1,2,0)$ have $V+T=2$ and $1$. The
correct structure is one 2D antichain per $L$.

**Proposition J.** The frontier of $w=(1,1)$, $A=4$, $s=4$ has five points,
$\{(0,1,3),(1,1,2),(2,1,1),(3,1,0),(0,2,0)\}$; the two extra points append
$k=2,3$ locator characters to a one-line body.

An output-sensitive algorithm computes the frontier in
$O\bigl((n+\lceil s/A\rceil)(s+1)\bigr)$ time and $O(n+s)$ space by sweeping
$L$ with a prefix minimum over $T$; 500 random instances, 0 mismatches.

---

## 5. Machine verification

| Check | Instances | Result |
|---|---|---|
| `verify_locator_wrap.py` | exhaustive small instances | 10/10 checks pass |
| Theorem 1 vs literal quadrangles vs $O(n)$ test | several thousand | 0 mismatches |
| $O(n+U)$ envelope DP vs $O(n^2)$ DP | 3\,300 (2\,428 non-monotone $z$) | 0 mismatches |
| CDQ $O(n\log n)$ DP vs $O(n^2)$ / exhaustive layouts | 2\,080 + 1\,400 | 0 mismatches |
| stack-only DP (Theorem 4) vs $O(n^2)$ DP | 5\,200 | 0 mismatches, $\le2.67n$ stack operations |
| stack-only, $U=\Theta(n^2)$ | $n=32\,000$ | $0.076\,$s, $2.42n$ stack operations |
| rank-space DP (§1.4) vs $O(n^2)$ DP | 5\,600 random + all $n\le6,w\le3,\beta\le7$ | 0 mismatches |
| monotone start pointer | 4\,200 | 145 counterexamples |
| $O(n)$ rank DP, nonincreasing budgets | 3\,850 | 0 mismatches, 0 lemma violations |
| $O(n+m)$ relational line count, $b_1\le\cdots\le b_m$ | 266\,345 exhaustive | 0 failures |
| Theorem 8 formula | all sequences of span $\le A$, $A\le14$ | 0 mismatches |
| Theorem 9 bound | exhaustive $A\le8,n\le3$ + 600 random | 0 violations |
| output-sensitive frontier algorithm | 500 random | 0 mismatches |

`locator_wrap_algorithms.py` defines the algorithms and checks;
`locator_wrap_checks.py` runs them in dependency order;
`verify_locator_wrap.py` supplies the baseline brute-force checks. The three
Python files require only the standard library.

---

## 6. Open problems

1. **1(a), word-RAM.** Decide whether (0.1) admits an $O(n)$ algorithm for
   $w$-bit integer coordinates with $U=n^{\omega(1)}$. Available bounds are
   $O(n+U)$, $O(n\log n)$ comparison, $O(n\log w)$ word-RAM; the rank-space
   DP achieves $O(n\log n)$ without predecessor queries. A matching
   unconditional super-linear lower bound in the cell-probe model is also
   open.
2. **1(b), general rank budgets.** For arbitrary distinct $(\beta_t)$, decide
   whether the convex chain of §2.2 admits $o(nK)$, or prove a super-linear
   lower bound. Theorem 5 covers nonincreasing profiles; Theorem 6 covers
   periodic profiles; the endpoint-rank, reversal and Lagrangian reductions
   are excluded by Propositions E–G.
3. **1(b), relational secondary objective.** Decide whether $o(nm)$ is
   possible for arbitrary width profiles $(s_r)$ and for the least-squares
   secondary objective. Theorem 7 covers the line count when widths are
   nonincreasing from the end.

External results used: SMAWK/Monge searching [citation:25], the concave
least-weight subsequence problem [citation:13], core-sparse Monge matrix
multiplication [citation:5], online Monge DP [citation:29], y-fast tries
[citation:30], fusion trees [citation:31], and predecessor lower bounds
[citation:6].
