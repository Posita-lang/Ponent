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
in `solution.md` and checked by the baseline exhaustive checks in
`asymmetric_line_breaking_algorithms.py`.

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

> **Theorem 1$'$ (feasible-staircase form).** Define
> $$\lambda_j=\min\{i<j:C_i\ge z_j\},\qquad \lambda_j=j\ \text{if no such }i,$$
> so row $j$ is finite exactly on $i\in[\lambda_j,j-1]$. On $i<i'<j<j'$,
> $M$ is totally monotone iff
> $$\lambda_{j'}\le j-2\ \Longrightarrow\ \lambda_{j'}\ge\lambda_j
> \qquad\text{for every }j<j'.$$
> This is equivalent to Theorem 1 with $\lambda_j=j$ for infeasible rows, and
> gives an $O(n)$ decision by computing all $\lambda_j$ and checking the
> displayed implication.

*Proof.* A TM premise can hold only when $M[j,i]=\infty$, i.e. $i<\lambda_j$
(the case $i\ge j$ is outside the triangular domain). If $i\ge\lambda_{j'}$,
row $j'$ has finite increasing entries at $i,i'$, so
$M[j',i']>M[j',i]$ and the conclusion fails. Such an $i$ exists exactly when
$\lambda_{j'}\le j-2$ and $\lambda_{j'}<\lambda_j$: choose
$i=\lambda_{j'}$ and $i'=j-1$. Conversely, if no such pair exists, every
$i<\lambda_j$ with $i\le j-2$ satisfies $i<\lambda_{j'}$, so row $j'$ has
$M[j',i]=\infty$ and the conclusion $\le\infty$ holds. $\blacksquare$

Machine check: 19\,602 exhaustive/random instances, 0 mismatches between the
literal quadrangle definition, the $\lambda$-implication, and the $O(n)$
suffix-minimum test. In particular $\lambda_1\le\cdots\le\lambda_n$ is
sufficient but not necessary when a row has no feasible predecessor:
$w=(2,1)$, $\beta=(1,5)$ gives $\lambda=(1,0)$ and total monotonicity holds.

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

### 1.2 Bounded coordinates and comparison complexity

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

### 1.3 Offline batched predecessor, not online predecessor equivalence

> **Theorem 4 (upper bounds).** On $w$-bit integer coordinates, (0.1) is
> solvable in $O(n+U)$ time by direct addressing, in
> $O(n\cdot T_{\mathrm{pred}}(n,U,w))$ time and $O(n)$ space by the stack-only
> DP with a predecessor structure on interval starts, and in $O(n\log n)$
> comparisons by CDQ. In the comparison model the bound is
> $\Theta(n\log n)$ by Theorem 3.

> **Proposition U (the online-predecessor lower bound does not transfer).**
> Let $T_{\mathrm{pred}}$ be the optimal **online** static predecessor query
> time with near-linear space. The assertion
> $$\text{computing all }dp\text{ values}=\Theta(n\,T_{\mathrm{pred}})$$
> is false as a lower-bound inference. The reduction that encodes $m$ queries
> into $n=2m$ DP rows yields an instance on which all $dp$ values are
> computable in $O(n)$ word-RAM time for $w=O(\log n)$, although
> $n\,T_{\mathrm{pred}}=\Omega(n\log\log n)$.

*Explicit instance.* Let $S=\{s_1<\cdots<s_m\}$ and $Q=\{q_1,\dots,q_m\}$ be
$w$-bit integers with $0\le q_j<s_m$, $w=\gamma\log n$, $\gamma>1$. Set
$C_i=2(s_i+2)$ for $1\le i\le m$, $C_{m+r}=C_m+2r$, $z_j=0$ for $j\le m$,
$z_{m+j}=2(q_j+2)$, and $\beta(j)=C_j-z_j-1$. With $\Lambda$ dominating,
$$dp[j]=\Lambda\ (j\le m),\qquad
dp[m+j]=2\Lambda+\bigl(C_{\lambda_j}-z_{m+j}\bigr)^2,$$
where $\lambda_j=\min\{i:s_i\ge q_j\}$. All $dp$ values are therefore exactly
$m$ **offline** successor queries. Sorting $S\cup Q$ by two-pass radix sort
(digit $\tfrac12\log n$ bits, $\sqrt n$ buckets per pass) takes $O(n)$ time and
$O(n)$ space for $w=O(\log n)$; a linear merge returns every $\lambda_j$.
Machine check: 1\,050 random instances, 0 mismatches against the literal
$O(n^2)$ DP. Explicit $m=8$: $S=(3,5,20,24,32,34,37,43)$,
$Q=(11,32,40,6,27,7,22,29)$ gives ranks $(3,5,8,3,5,3,4,5)$.

Pătrașcu–Thorup (STOC 2006) proves that, for polynomial universes
$w=\ell=\gamma\log n$ and near-linear space, **online** predecessor queries
require $\Theta(\log \ell)=\Theta(\log\log n)$ cell probes, and gives the exact
time-space trade-off for all $n,w,S,\ell$ up to constant factors. Those
queries are not known at preprocessing time; the DP queries $z_j$ are part of
the input. Consequently the online lower bound does not rule out an $O(n)$
word-RAM algorithm for (0.1). What remains open is the exact word-RAM
complexity of the offline DP for arbitrary $w$ and $U$; the comparison bound
$\Theta(n\log n)$, the $O(n+U)$ direct-address bound, the $O(n\log w)$ and
$O(n\log n/\log w)$ stack bounds, and the $O(n\log n)$ CDQ bound are the
known extremes. For $U=O(n)$ the direct-address algorithm is $O(n)$.

### 1.7 Nonmonotone envelope starts and the split-find barrier

Let $X_0<\cdots<X_{D-1}$ be the sorted distinct values of $\{z_1,\dots,z_n\}$;
write each line of (0.1) as
$$\ell_i(x)=dp[i]+C_i^2-2C_i x .$$
Slopes $-2C_i$ decrease in $i$, so a newly inserted line wins on a suffix of
the active rank space. Let $s_i$ be the rank at which line $i$ first owns the
lower envelope (rightmost-tie rule), and let
$e_i=\max\{r:X_r\le C_i\}$ be its rightmost feasible rank. A split-only
interval split-find structure would give $O(n)$ time if $s_0\le s_1\le\cdots$;
Gabow–Tarjan (1985) gives $O(1)$ amortized split/find on a RAM.

> **Proposition AA.** The sequence $s_i$ need not be nondecreasing. Take
> $$w=(1,5,4,1,4,6,6),\quad C=(0,2,8,13,15,20,27,34),$$
> $$z=(-320,5,1,-343,-72,-98,-61),\quad \beta(j)=C_j-z_j-1 .$$
> With the dominating penalty of (0.1) and rightmost-tie envelope ownership,
> $$dp=(0,1239433,\infty,2376467,1254682,1142217,1146637,1140754),$$
> $$(s_0,\dots,s_7)=(0,5,5,6,6,5,5,5),$$
> so $s_5=5<s_4=6$. There are no ties. Hence the envelope update must remove
> the largest boundary and then insert a smaller boundary; it is a full
> **union-split-find** problem, not split-only. Random search over 1\,000
> instances gives 109 with nonmonotone starts.

A full interval union-split-find structure supports split, union and find in
$O(\log\log D)$ per operation (van Emde Boas / y-fast), but locating the first
rank at which a new line beats the current piecewise-linear envelope still
requires $O(\log D)$ envelope probes with the methods described here. Thus an
$O(n\log\log n)$ bound for (0.1) is **not** established by this route. The
pointer-machine lower bound $\Omega(\log\log n)$ per operation of
Mehlhorn–Näher–Alt (1990) and the cell-probe bound of Beame–Fich (2002) apply
to union-split-find itself; they do not directly decide the offline DP.


### 1.4 Compressed algorithms

The $O(n+U)$ algorithm addresses every integer coordinate. A compressed
version keeps only $X=\{z_1,\dots,z_n\}$ and represents the lower envelope as
a stack of disjoint numeric intervals. Line $\ell_i(x)=dp[i]+C_i^2-2C_i x$ is
valid for $x\le C_i$; slopes $-2C_i$ decrease in $i$, so a new line wins on a
suffix. Insertion pops a suffix of the stack, trims the kept top, and appends
one interval; a query $z_j$ needs one predecessor search among the interval
starts.

Verification: 5\,200 random instances against the literal $O(n^2)$ DP — 0
mismatches, at most $2.67n$ stack operations; $n=32\,000$ with
$U\approx1.02\cdot10^9$ in $0.076\,$s.

**Proposition C.** "A single monotone pointer over interval starts gives
$O(n)$ time." False. On 4\,200 random instances a monotone-pointer variant
disagrees with (0.1) 145 times; numeric starts are non-monotone in 167
instances. Explicit witness: $w=(7,3,1,2,5)$, $\beta=(4,22,4,18,14)$,
$\Lambda=35$ gives $dp[5]=155$ while the pointer variant returns $207$.

A rank-space variant works entirely in the rank space of sorted $X$. Each line
wins on a suffix; an insertion overwrites every rank from its split rank to
its end rank, and queries are $O(1)$ array lookups. The variant agrees with
(0.1) on 5\,600 random instances and all $n\le6$, $w_t\le3$,
$\beta(j)\le7$; correctness is not the issue.

> **Proposition D (rank-space worst case).** The assertion
> "the rank-space DP runs in $O(n)$ or $O(n\log n)$" is false. For every even
> $n=2m\ge6$ there is a feasible instance with unit words, polynomial $w$-bit
> integer coordinates ($w=O(\log n)$), and a dominating $\Lambda$ on which the
> rank-space DP performs $\Theta(n^2)$ writes. Offline sorting of $X$ alone
> does not give $O(n)$ word-RAM time for (0.1).

The construction and proof are in §1.5. Machine checks: $n=6,8,10,20,40,80,120$
all satisfy the claimed DP values, split ranks and write lower bound; an
explicit $n=8$ witness has $14$ writes against the $O(n)$ prediction;
exhaustive lexicographic layouts at $n=6,8,10$ agree with the DP.

### 1.5 Quadratic-write counterexample for rank space

Let $n=2m$, $m\ge3$, $\alpha_t\equiv1$, so $C_j=2j$. Put $K=4n^2$ and
$$\beta(j)=1\ (1\le j\le m),\qquad \beta(j)=K+n-1\ (m<j\le n).$$
Then
$$z_j=\begin{cases}2j-2,&j\le m,\\-K+2(j-m),&j>m.\end{cases}$$
Let $B=K+n-1$ and $\Lambda=n(B+1)^2+1=n(K+n)^2+1$. Every slack is at most
$B$, so every total penalty is $<nB^2<\Lambda$; $\Lambda$ dominates. Also
$K>3n$ and $\Lambda>K^2+4n^2$ for $n\ge6$. All input values are $O(n^2)$ and
all DP values are $O(n\Lambda)=O(n^6)$, hence $w=O(\log n)$.

For $i\le m$, $z_j=C_{j-1}$ on $j\le m$, so a line ending at $j$ must start
at $j-1$ or later; every layout of the first $i$ words has $i$ lines and zero
penalty, hence $dp_i=i\Lambda$. For $i>m$, $z_i\le0$, so the one-line layout
$(0,i)$ is feasible with slack $-z_i$; any layout with at least two lines
costs at least $2\Lambda>\Lambda+z_i^2$. Hence
$$dp_i=\Lambda+z_i^2\qquad(m<i\le n).$$

Let $X_0<\cdots<X_{n-1}$ be the sorted distinct $z$-values. The $m$ negative
values occupy ranks $0,\dots,m-1$, and
$$X_{m+s}=2s\qquad(0\le s\le m-1).$$
In particular $X_m=0$, $X_{m+1}=2$, $X_{m+2}=4$, and
$\max_r X_r=2m-2=n-2$. For $i\ge m+1$, $C_i=2i\ge n+2$, so line $i$ is
valid on every rank and its end rank is $n-1$.

Fix $i\in\{m+1,\dots,n-1\}$ and insert line $i$, whose value is
$\ell_i(x)=\Lambda+z_i^2+(C_i-x)^2$.

*Loss on ranks $\le m+1$.* For $x\le0$, line $0$ has value $x^2$ and is
valid; $\ell_i(x)\ge(C_i-x)^2>x^2$. For $x=2$, line $1$ has $C_1=2$,
$dp_1=\Lambda$, hence value $\Lambda$; $\ell_i(2)>\Lambda$.

*Win on ranks $\ge m+2$.* Let $x\ge4$ and let $h<i$ be any older feasible
line, so $C_h\ge x$. Lines $h=0,1$ are invalid because $C_0=0$, $C_1=2$.
For $2\le h\le m$, $dp_h=h\Lambda$ and
$$\bigl[h\Lambda+(C_h-x)^2\bigr]-\ell_i(x)
=(h-1)\Lambda+(C_h-x)^2-z_i^2-(C_i-x)^2
\ge\Lambda-(K^2+4n^2)>0 ,$$
using $z_i^2\le K^2$, $(C_i-x)^2\le C_i^2\le4n^2$.
For $m<h<i$, $dp_h=\Lambda+z_h^2$ and
$$\bigl[z_h^2+(C_h-x)^2\bigr]-\bigl[z_i^2+(C_i-x)^2\bigr]
=(z_h^2-z_i^2)-\bigl[(C_i-x)^2-(C_h-x)^2\bigr].$$
Here $z_h-z_i=-2(i-h)$ and $z_h+z_i\le-2K+2n$, so
$z_h^2-z_i^2\ge4(K-n)(i-h)$. Also
$(C_i-x)^2-(C_h-x)^2\le(C_i-C_h)(C_i+C_h)\le8n(i-h)$. Since $K>3n$, the
difference is positive. Thus line $i$ beats every older feasible line at
every rank $\ge m+2$ and loses at every rank $\le m+1$; its split rank is
exactly $m+2$.

Each of the $m-1$ insertions $i=m+1,\dots,n-1$ therefore writes the
$n-1-(m+2)+1=m-2$ ranks $m+2,\dots,n-1$. The number of writes is at least
$$(m-1)(m-2)=\Bigl(\frac n2-1\Bigr)\Bigl(\frac n2-2\Bigr)=\Theta(n^2).$$
Machine check: $n=8$ gives
$\beta=(1,1,1,1,263,263,263,263)$, $\Lambda=557569$,
$X=(-254,-252,-250,-248,0,2,4,6)$, split rank $6$, and writes
$14$; $n=120$ gives $3542$ writes against the lower bound $3422$.

### 1.6 Few distinct query coordinates: $O(nD)$ and $O(n\log D)$

Let
$$D=\bigl|\{z_1,\dots,z_n\}\bigr|$$
be the number of **distinct** query coordinates. Nothing below assumes total
monotonicity, the staircase condition (SC), or a bound on $U$ or $w$.

> **Theorem 25 (prefix-minima DP).** DP (0.1) is solvable in $O(nD)$ time and
> $O(D)$ space on a pointer machine, with exact integer arithmetic and no
> sorting, hashing being replaceable by an $O(D)$ table lookup. In particular
> $D=O(1)$ gives an $O(n)$ algorithm.

*Proof.* Rewrite (0.1) as
$$dp[j]=\Lambda+\min_{\substack{i<j\\ C_i\ge z_j}}\bigl(dp[i]+(C_i-z_j)^2\bigr).$$
Collect the $D$ distinct values $x_1,\dots,x_D$ of $z$ in one scan. Maintain
$\mathrm{cur}[k]=\min\{dp[i]+(C_i-x_k)^2 : i<j,\ C_i\ge x_k\}$ after lines
$0,\dots,j-2$ have been inserted. Inserting line $i=j-1$ updates
$\mathrm{cur}[k]$ for every $x_k\le C_i$, at most $D$ entries, and the query
$j$ reads $\mathrm{cur}[k(z_j)]$. Induction on $j$ gives (0.1). Each insertion
costs $O(D)$, so the total is $O(nD)$; the algorithm never inspects $U$ or $w$.
$\blacksquare$

> **Theorem 26 (envelope DP).** DP (0.1) is solvable in
> $O(n\log D+D\log D)$ time and $O(n+D)$ space in the comparison model, and in
> $O(n+n\log D)$ time on the word RAM when $w=O(\log n)$ (radix sorting the
> $D$ distinct $w$-bit coordinates). For $D=\Theta(n)$ this is $O(n\log n)$,
> matching CDQ; for $D=O(1)$ it is $O(n)$.

*Proof.* Sort the distinct coordinates $X_0<\cdots<X_{D-1}$ and let
$r_i=\max\{r:X_r\le C_i\}$, nondecreasing in $i$ since $C_i$ increases. Line
$i$ is feasible exactly on ranks $0,\dots,r_i$, with value
$\ell_i(r)=dp[i]+(C_i-X_r)^2$. For $h<i$,
$$\ell_i(r)-\ell_h(r)=(b_i-b_h)-2(C_i-C_h)X_r,\qquad b_i=dp[i]+C_i^2,$$
is decreasing in $X_r$; hence a new line beats every older line on a **suffix**
of ranks. The pointwise minimum is therefore a stack of disjoint intervals
$[s_i,r_i]$; inserting line $i$ pops the trailing intervals it dominates and
binary-searches the split rank $s_i$ inside the remaining top interval. Each
pop is charged once, so all pops and split searches cost $O(n\log D)$.

Replacing the flat rank array of §1.5 by a lazy segment tree with
**time-stamped** assignments fixes the $\Theta(n^2)$ writes: assigning line
$i$ to $[s_i,r_i]$ writes $O(\log D)$ nodes, each tagged $(i,\text{line})$; a
point query at rank $r$ returns the tag with the largest stamp on the
root-to-leaf path, which is exactly the most recent line covering $r$. Later
ancestor tags correctly override preceding descendant tags. Hence all $n$
insertions and all $n$ queries cost $O(n\log D)$. $\blacksquare$

**A new linear-time regime.** Let $\alpha_t\equiv1$ ($C_j=2j$), $H=n^3$, and
let $z_j$ alternate between $0$ and $-H$; then $D=2$, $U=\Theta(n^3)$,
$w=\Theta(\log n)$, and SC fails because $z$ is not nondecreasing. Every bound
that depends on $U$ or $w$ is superlinear here: $O(n+U)=O(n^3)$,
$O(n\log w)=O(n\log\log n)$, $O(n\log n/\log w)=O(n\log n/\log\log n)$, and
$O(n\log n)$ from CDQ. The $\Omega(n\log n)$ comparison lower bound of
Theorem 3 is a worst-case bound whose reduction encodes $\Theta(n)$ distinct
arbitrary coordinates; with $D=O(1)$ there is nothing to sort, so it does not
apply to this subclass. Theorems 25–26 give $O(n)$. A second family with $\alpha_t\equiv H$ and
$z_j\in\{0,H-1\}$ has $U=\Theta(n^4)$ and is also $O(n)$.

Machine checks: 3\,600 random instances with $D\le4$ and 4\,680 exhaustive
small instances, 0 mismatches for both algorithms against the literal
$O(n^2)$ DP; scaling at $n=64\,000$ with $U\approx2.6\cdot10^{14}$ confirms
linear time; the envelope is nonconstant on small-$D$ inputs (up to 5
assignments and 1 pop on 14-word instances).

Thus the exact word-RAM question remains open only when $D=\Theta(n)$; the new
upper bound is $O(n\min(D,\log n))$ (Theorem 25 for $D\le\log n$, Theorem 26
in general), and $D=O(1)$ is settled in $O(n)$.

---

## 2. Rank-indexed budgets

The $t$-th line has budget $\beta_t$, independent of where it ends. One layer
is $O(n)$ (a Monge row-minima problem); $K$ layers cost $O(nK)$.

### 2.1 Nonincreasing budgets: $O(n)$

Let $\beta_1\ge\beta_2\ge\cdots$ and $L(j)$ be the greedy block count on the
first $j$ words. Greedy is exact for the line count.

> **Theorem 5 (endpoint-rank).** In every minimum-line layout the $t$-th line
> ends at $j_t$ with $L(j_t)=t$. Consequently the lexicographic objective is
> computable in $O(n)$ time and $O(n)$ space.

*Proof.* $t\ge L(j_t)$ always. If $t>L(j_t)$, replace the prefix by an
$L(j_t)$-line layout; the remaining $K-t$ lines shift to ranks
$L(j_t)+1,\dots,L(j_t)+K-t$. Since $\beta$ is nonincreasing and
$L(j_t)<t$, each receives a budget at least as large as before, so the
layout stays feasible with $L(j_t)+K-t<K$ lines — contradiction. The lemma
makes the budget of the line ending at $j$ equal $\beta_{L(j)}$; one monotone
CHT per greedy level suffices. $\blacksquare$

Verified on 3\,850 random instances against exhaustive layouts: 0 mismatches,
0 lemma violations.

### 2.2 Transformed coordinates

Let $S_0=0$, $S_t=\sum_{r=1}^t(\beta_r+1)$, and for a layout with breakpoints
$j_0=0<j_1<\cdots<j_K=n$ define

$$X_t = C_{j_t} - S_t .$$

Then the width of line $t$ is
$C_{j_t}-C_{j_{t-1}}-1 = X_t-X_{t-1}+\beta_t$, so legality is
$-\beta_t < X_t-X_{t-1}\le 0$, and the squared-slack penalty is

$$\sum_{t=1}^{K}\bigl(X_{t-1}-X_t\bigr)^2 .$$

The problem is a convex chain through the shifted point sets
$P_t=\{C_j-S_t:0\le j\le n\}$; the continuous optimum is the linear
interpolation $X_t=t(C_n-S_K)/K$ with value $(C_n-S_K)^2/K$.

### 2.3 Periodic budget sequences

Let $\beta_1,\beta_2,\dots$ be periodic with period $p$; let $K$ be the number
of lines in the lexicographic optimum and put $p_0=\min(p,K)$.

> **Theorem 6.** Periodic rank-indexed budgets are solvable in
> $$O\bigl(\min\{nK,\ p_0 n\log n\}\bigr)$$
> time. In particular the bound is $o(nK)$ whenever $p_0=o(K/\log n)$.

*Algorithm.* Build a phase-augmented shortest-path graph. State $(r,j)$ means
that the next line has budget $\beta_{r+1}$ and the prefix ends at $j$; an edge
from $(r,i)$ to $(r+1\bmod p,j)$ costs
$(\beta_{r+1}+1+C_i-C_j)^2+\Lambda$, with $\Lambda=nB^2+1$ and
$B=\max_r\beta_r$. Because $\Lambda$ dominates every achievable total
penalty, a shortest path has the fewest lines and then the least penalty.

For source phase $r$, line $i$ is feasible for exactly those $j$ with
$i<j$ and $C_j\le C_i+\beta_{r+1}+1$; write
$J_r(i)=\max\{j:C_j\le C_i+\beta_{r+1}+1\}$. Insert the line
$\ell_i(x)=d[r][i]+(C_i+\beta_{r+1}+1-x)^2+\Lambda$ into an offline segment
tree over the interval $[i+1,J_r(i)]$. Each line is stored at $O(\log n)$
nodes; every node keeps a monotone lower hull of lines with decreasing slopes,
and queries arrive in increasing $j$, so one pointer per node suffices. All
states are processed in one sweep $j=1,\dots,n$; the $p_0$ phases give
$O(p_0 n\log n)$. The answer is recovered by subtracting
$\Lambda\cdot K$ from the shortest-path distance. $\blacksquare$

Verification: 2\,700 random instances and all small instances with $n\le6$,
$w_t\le2$, $p\le3$, $\beta_r\le4$ against exhaustive lexicographic search —
0 mismatches. Scaling: $n=16\,000$, $p=4$ in $0.29\,$s.

> **Proposition E.** "Core-sparse Monge multiplication removes the factor
> $K$ for constant-valued or few-run budget sequences." False. The core size
> $\delta(A)$ of a Monge matrix is the number of nonzero entries of its
> density matrix
> $A^\square_{i,j}=A_{i+1,j}+A_{i,j+1}-A_{i,j}-A_{i+1,j+1}$. For a layer
> with $d=\beta+1\ge C_n-C_0-1$ (all lines fit),
> $$A^\square_{i,j}=2(C_{i+1}-C_i)(C_{j+1}-C_j)>0$$
> for every interior cell $i+1<j$; hence $\delta=\Theta(n^2)$, independent
> of $\beta$. Adding row or column potentials does not change the density.
> Core-sparse min-plus multiplication therefore costs $O(n^2\log n)$ per
> product, which is worse than the $O(n)$ layered DP whenever $K=O(n)$.
> Machine check: $n=10$ has 36 nonzero interior density cells, 0 zero cells,
> and 0 failures of the density formula. **Consequently no $o(nK)$ bound
> follows from that route; for a sequence of $R$ constant-valued runs,
> $O(nK)$ remains the best known bound.**

> **Theorem 7.** For a fixed residue $r\in\{0,\dots,p-1\}$, let
> $P_r(q)$ be the least squared-slack penalty of a layout with
> $R=r+q p$ lines (with $r=0$ interpreted as $R=q p$). Then $P_r(q)$ is
> discrete convex in $q$: successive differences are nondecreasing.

*Proof sketch.* Group $p$ consecutive lines into one macro edge. The macro
matrix is the min-plus product of $p$ Monge layer matrices, hence is concave
Monge; the fixed $r$-line boundary is another Monge matrix. The theorem of
Aggarwal, Schieber and Tokuyama (1994) states that the minimum weight of a
$q$-link path in a concave Monge DAG is convex in $q$. $\blacksquare$
Verified on 1\,233 residue curves: 0 non-convex cases.

### 2.4 Layer-matrix structure and Lagrangian barriers

> **Theorem 8 (exact decomposition).** For any budget $\beta$, on the
> feasible band the layer matrix splits as
> $$M_\beta[i,j]=\underbrace{C_i^2+2(\beta+1)C_i}_{a_\beta(i)}
> +\underbrace{C_j^2-2(\beta+1)C_j}_{b_\beta(j)}
> -\underbrace{2C_iC_j}_{\text{fixed rank-one kernel}}
> +(\beta+1)^2 .$$
> Its Monge density $2(C_{i+1}-C_i)(C_{j+1}-C_j)$ is **independent of
> $\beta$ and rank one**. Therefore the $K$-layer DP is exactly an iterated
> discrete Legendre chain with a **fixed rank-one pairwise kernel**
> $-2C_iC_j$ and **time-varying unary potentials**
> $u_t(x)=2x^2-2d_t x$, where $d_t=(\beta_t+1)-(\beta_{t+1}+1)
> =\beta_t-\beta_{t+1}$.

*Identity check:* 500 random instances, 0 failures. Consequence: inside a
run of equal budgets $d_t=0$, so the unary potential is fixed; this is why
constant and periodic profiles admit compression (Theorems 5, 6, 7). Across
runs the source vector is the output of the preceding run, i.e. a general
vector, and the count dimension cannot be removed by a single penalty —
made precise by the next two propositions.

> **Proposition F (one-parameter gap).** Let $A$ be the body budget,
> $b=A-s$ the terminal budget, and
> $$F(q,m)=\min\Bigl\{\sum(\text{slack})^2:\ \text{exactly }q\text{ lines at }A
> \text{ then exactly }m\text{ lines at }b\Bigr\}.$$
> Fixing $q$ by an exact $q$-layer body DP and relaxing only $m$ with a
> penalty $\lambda$ per terminal line does **not** recover $F(q,m)$ in
> general. Explicit instances (source is the general body-DP output):
>
> | $w$ | $A$ | $s$ | $(q,m)$ | $F(q,m)$ | one-parameter dual | gap |
> |---|---|---|---|---|---|---|
> | $(1,1,1,2)$ | 5 | 3 | $(2,1)$ | 20 | 19 | 1 |
> | $(1,1,1,2)$ | 6 | 3 | $(2,1)$ | 35 | 34 | 1 |
> | $(1,1,1,2)$ | 6 | 4 | $(2,1)$ | 34 | 32 | 2 |

> **Proposition G (two-parameter gap).** Relaxing **both** counts, i.e.
> $$L(\lambda_A,\lambda_b)=\min_{q,m}\bigl[F(q,m)+\lambda_A q+\lambda_b m\bigr],
> \qquad F^{**}(q,m)=\max_{\lambda_A,\lambda_b}\bigl[L-\lambda_A q-\lambda_b m\bigr],$$
> still has a nonzero gap: $F(q,m)$ is **not jointly convex** on the integer
> lattice. Explicit instance: $w=(2,2,1,4,4,3)$, $A=7$, $s=2$,
> $(q,m)=(3,3)$ gives $F=92$ while $F^{**}=79$, a gap of $13$. A census of
> 250 random instances with $n\le6$, $A\le8$ gives gaps in $\{0,\dots,21\}$;
> the largest observed is $21$ on $w=(1,3,1,3,3,3)$, $A=8$, $s=5$,
> $(q,m)=(3,3)$, $F=123$. The exact grid and its convex envelope for
> $w=(3,1,2)$, $A=3$, $s=1$:
>
> | $q$ | exact $F(q,0..3)$ | dual $F^{**}(q,0..3)$ |
> |---|---|---|
> | 0 | $(\infty,\infty,\infty,\infty)$ | $(601,401,201,401)$ |
> | 1 | $(\infty,\infty,1,\infty)$ | $(401,201,1,201)$ |
> | 2 | $(\infty,4,\infty,\infty)$ | $(203,3,201,401)$ |
> | 3 | $(5,\infty,\infty,\infty)$ | $(5,203,401,601)$ |

**Corollary.** For general non-uniform $C$, the Lagrangian/Aliens family
(single penalty, nested penalties, or any Legendre relaxation of the line
counts) cannot produce an exact $o(nK)$ rank-indexed algorithm nor an exact
$o(nm)$ relational algorithm for the secondary objective. Any such algorithm
must be non-Lagrangian. This delimits the two remaining algorithmic questions
in §6.

### 2.5 Uniform word widths: $O(n\log n)$ for arbitrary budgets

Let all words have width $\alpha$, so $g=\alpha+1$ and $C_j=g j$. A line with
$m_r$ words has width $g m_r-1$ and slack $\beta_r+1-g m_r$; it is legal iff
$1\le m_r\le\mathrm{cap}_r=\lfloor(\beta_r+1)/g\rfloor$. For exactly $K$
lines the penalty is

$$\sum_{r=1}^{K}\bigl(\beta_r+1-g m_r\bigr)^2,\qquad
\sum_{r=1}^K m_r=n,\quad m_r\ge1 .$$

This is a convex separable integer resource allocation problem.

> **Theorem 9.** For uniform word widths the rank-indexed lexicographic
> optimum is computable in $O\bigl(n+(n-K)\log K\bigr)\le O(n\log n)$ time:
> $K$ is the smallest prefix with $\sum_{r\le K}\mathrm{cap}_r\ge n$; start
> with $m_r=1$ and allocate the remaining $n-K$ words by a min-heap keyed by
> the marginal cost
> $\Delta_r(m)=g(2m+1)-2(\beta_r+1)$, which is increasing in $m$. Each
> increment is one heap operation, and lines are removed at their capacity.

Verified on 2\,500 random instances against exhaustive lexicographic search:
0 mismatches. Scaling with budgets forcing $K=n/2$: $n=64\,000$ in
$0.022\,$s.

> **Theorem 10.** For uniform word widths the relational variant with an
> arbitrary locator profile $(s_1,\dots,s_m)$ is computable in $O(n\log n)$:
> the capacity of a $K$-line layout is $(K-m)\mathrm{cap}_A+
> \sum_r\lfloor(A-s_r+1)/g\rfloor$, giving $K$ in closed form, after which
> Theorem 9 applies to the budget sequence
> $[A]^{K-m},[A-s_m,\dots,A-s_1]$.

Verified on 15\,120 random instances against exhaustive relational search:
0 mismatches.

### 2.6 Counterexamples

* **Proposition H.** "The endpoint-rank lemma holds for every budget
  profile." False. $w=(1,1,1,2)$, $\beta=(3,3,4,4)$: lex-opt $(3,8)$ with
  breakpoints $[0,1,2,4]$, while $L=[1,1,2,3]$; at $t=2$, $j_2=2$ and
  $L(2)=1$. A smaller non-monotone witness is $w=(1,1,3)$,
  $\beta=(3,2,3)$, lex-opt $(3,5)$, $L=[1,1,2]$.
* **Proposition I.** "Reversing the text turns nondecreasing budgets into
  nonincreasing ones." False. $w=(1,2)$, $\beta=(2,4)$: lex-opt $(2,5)$, but
  reversal gives budgets $(4,2)$ and the spurious answer $(1,0)$.
* **Proposition J.** "The Lagrangian/Aliens $\Lambda$-shift compresses the
  layers for arbitrary start vectors." False. From a general start vector:
  $w=(1,2,1,1)$, $\beta=2$, $dp_0=(\infty,\infty,13,9,1)$ gives
  $P=(1,10,15)$ — concave. Even from $e_0$ with nonincreasing rank budgets:
  $w=(4,4,1,3,1)$, $\beta=(9,8,6,2)$ gives
  $P(2),P(3),P(4)=(1,30,39)$ — concave. Among 2\,000 random instances,
  71 have non-convex $P(R)$.
* **Proposition K.** "Permuting the budget sequence preserves the optimum."
  False. $w=(1,1,2)$ with $\beta=(2,5)$ gives lex-opt $(2,2)$, while
  $\beta=(5,2)$ gives $(2,4)$; $\beta=(2,6)$ gives $(2,5)$ while
  $\beta=(6,2)$ gives $(1,0)$. Sorting $(\beta_t)$ is not a valid reduction
  for non-uniform $C$.

The online DP speedup of Bar-Noy, Golin and Zhang applies to recurrences with
the online-Monge rank-one density property; rank-indexed budgets with arbitrary
$(\beta_t)$ do not have that decomposition in general.

### 2.7 Relational line count, general widths

Last $m$ lines carry locators of widths $s_1,\dots,s_m$ ($r=1$ is the last
line); $b_r=A-s_r$. Exhaustive checks (all $A\le5,m\le3,n\le5$):

| profile from the end | instances | $O(n+m)$ threshold rule |
|---|---|---|
| $b_1\le\cdots\le b_m$ (widths $s_1\ge\cdots\ge s_m$) | 266\,345 | exact (0 failures) |
| all $b_r$ equal | 77\,894 | exact (0 failures) |
| $b$ nonincreasing from the end | 266\,345 | 390 counterexamples |
| arbitrary $b$ | 728\,438 | 675 counterexamples |

> **Theorem 11.** If $b_1\le\cdots\le b_m$, the feasibility sets $f_r$ are
> suffix-closed, thresholds $t_r$ are nonincreasing, and a single backward
> pointer computes the fewest-lines objective in $O(n+m)$.

Naive backward greedy is not exact (197 failures in 9\,000 instances). The
reduction to the $O(n)$ rank DP by fixing $K$ and taking
$\beta_K=[A]^{K-m}+[b_m,\dots,b_1]$ is false: $w=(1,1)$, $A=3$, $s=(2)$
gives lex-opt $(2,4)$ while $\beta_K=(3,1)$ admits the spurious $(1,0)$.

### 2.8 Reversal equivalence and uniform-tail compression

Let $w^R=(w_n,\dots,w_1)$, $b_r=A-s_r$, and

$$\beta'=[b_1,b_2,\dots,b_m,A,A,A,\dots] .$$

> **Theorem 14.** The relational lexicographic optimum on $w$ with profile
> $(s_1,\dots,s_m)$ equals the rank-indexed lexicographic optimum on $w^R$
> with budgets $\beta'$, restricted to layouts with at least $m$ lines.

*Proof.* A line's width $B(i,j)$ is invariant under reversing both the word
sequence and the order of the lines. A $K$-line relational layout on $w$ has
budget sequence $[A]^{K-m},[b_m,\dots,b_1]$ from the start; reversing the
line order turns it into $[b_1,\dots,b_m,[A]^{K-m}]$, which is the first $K$
entries of $\beta'$, and $K\ge m$. Penalties and line counts are preserved,
so the lexicographic optima coincide. $\blacksquare$

Verified on 170\,352 exhaustive instances ($n\le6$, $A\le6$, $m\le3$): 0
inequivalences. Consequence: **Question 3 is a special case of Question 2**
with only $m+1$ constant-valued runs of $\beta_t$; any $o(nK)$ algorithm for
rank budgets that is sublinear in the number of runs yields $o(nm)$ for the
relational variant. The converse reduction also shows the prefix budgets
$b_1,\dots,b_m$ from the start are the exact obstruction to the $O(n)$ rank
DP of Theorem 5.

> **Theorem 15.** With $\Lambda=nA^2+1$, the relational lexicographic optimum
> equals
> $$\min_{0\le j\le n}\bigl(G[j]+H[j]\bigr),\qquad
> \text{lines}=V/\Lambda,\quad \text{penalty}=V\bmod\Lambda,$$
> where $V=\min_j(G[j]+H[j])$, $G[j]$ is the cost of **any** number of
> $A$-lines on words $1..j$ (one unconstrained LWS pass, $O(n)$), and
> $H[j]$ is the cost of the $m$ terminal lines on words $j+1..n$ ($m$
> layered passes, $O(nm)$). $\Lambda$ dominates every achievable sum of
> squared slacks, so minimising $\sum\text{slack}^2+\Lambda\cdot(\text{lines})$
> selects the fewest lines and then the least penalty.

Verified on 7\,200 random instances: 0 mismatches. Thus $O(nm)$ is sharp
only if the $m$ prefix layers cannot be compressed.

> **Theorem 16.** For equal-width locators ($s_1=\dots=s_m=s$) and a uniform
> word grid ($\alpha_t=\alpha$, $g=\alpha+1$), the relational lexicographic
> optimum is computable in $O(n)$ time.

*Proof.* A $b$-line holding $k$ words ($b=A-s$, $d=b+1$) has slack $d-gk$
and cost $h(k)=(d-gk)^2$, $1\le k\le\lfloor d/g\rfloor$. The $m$ terminal
lines on a suffix of $N$ words cost exactly the $m$-fold min-plus convolution
$h^{*m}(N)$; $h$ is discrete convex ($\Delta^2h=2g^2$), so with
$N=mq+r$,
$$h^{*m}(N)=(m-r)h(q)+r\,h(q+1),$$
with value $\infty$ when the split leaves its feasible range. Hence
$H[j]=h^{*m}(n-j)+m\Lambda$ in $O(1)$ per $j$; Theorem 15 supplies the
remaining $O(n)$ pass. $\blacksquare$

Verified on 9\,072 kernel instances and 2\,574 full instances: 0 mismatches.
This improves Theorem 10's $O(n\log n)$ to $O(n)$ on the intersection of
both restrictions.

> **Proposition N.** "The convolution-power identity
> $H[j]=h^{*m}(C_n-C_j)$ holds on non-uniform $C$." False. With no budget
> window ($d\ge C_n$), the two-line cost from $i=0$ to $l=4$ is
> $\min_{k\in C}[(d-(C_k-C_0))^2+(d-(C_4-C_k))^2]$, but the lattice
> relaxation $\min_{x\in\mathbb Z}$ can be strictly smaller:
>
> | $w$ | $d$ | grid-constrained | lattice relaxation | gap |
> |---|---|---|---|---|
> | $(1,1,1,3)$ | 12 | 100 | 98 | 2 |
> | $(1,1,2,2)$ | 12 | 100 | 98 | 2 |
> | $(1,1,2,3)$ | 13 | 117 | 113 | 4 |

Intermediate breakpoints must lie in $C$, so the closed form cannot be
transplanted to non-uniform widths; any $o(nm)$ algorithm in that regime
must process the intermediate grid points without touching $m$ full layers.

### 2.9 Exact multi-layer structure and compression barriers

Let $F_m[i,j]$ denote the least penalty of a chain of exactly $m$ lines with
budgets $d_1,\dots,d_m$ (where $d_r=\beta_r+1$) from breakpoint $i$ to $j$, in
the **window-free** case ($d_r$ large enough that every line fits). Put
$D=\sum_r d_r$, $D_r=\sum_{s\le r}d_s$, $\Delta=C_j-C_i$, and

$$\xi_r=\Bigl(1-\frac rm\Bigr)C_i+\frac rm C_j+D_r-\frac{rD}{m}
\qquad(1\le r<m).$$

> **Theorem 17 (discrete-spline decomposition).** Writing
> $x_r=C_{j_r}$ for the interior breakpoints and $E_r=\xi_r-x_r$ with
> $E_0=E_m=0$,
> $$F_m[i,j]=\frac{(D-\Delta)^2}{m}
> +\min_{i=j_0<\cdots<j_m=j}\ \sum_{r=1}^{m}\bigl(E_r-E_{r-1}\bigr)^2 .$$
> The first term is the continuous optimum (equalised slacks); the second is
> a nonnegative **grid-quantisation energy**, zero iff every $\xi_r\in C$.

*Proof.* With $g_r=x_r-x_{r-1}$ and $e_r=d_r-g_r$, the penalty is
$\sum e_r^2$ subject to $\sum e_r=D-\Delta=:S$. Writing $e_r=S/m+\eta_r$
with $\sum\eta_r=0$ gives $\sum e_r^2=S^2/m+\sum\eta_r^2$, and
$x_r-\xi_r=-\sum_{s\le r}\eta_s$, so $\sum\eta_r^2=\sum(E_r-E_{r-1})^2$.
$\blacksquare$ Verified on 707 random instances: 0 failures.

For $m=2$ this is exact and one-dimensional:
$$F_2[i,j]=\frac{(d_1+d_2-\Delta)^2}{2}
+2\,\mathrm{dist}\!\left(\frac{C_i+C_j+d_1-d_2}{2},\,C\right)^2,$$
where $\mathrm{dist}(x,C)=\min_k|x-C_k|$. Proposition N's gap is precisely
the second term. For a uniform grid $C_i=gi$ the energy collapses to the
$m$-fold convolution $h^{*m}$ of Theorem 16.

> **Theorem 18 (convexity frontier).** For a **single** budget, the
> $k$-link value $P_i(k)=F_k[i,n]$ is discrete convex in $k$ and its Legendre
> transform has no gap:
> $P_i(m)=\max_\lambda\bigl[\min_k(P_i(k)+\lambda k)-\lambda m\bigr]$.
> For **mixed** budgets ($\Lambda$ per line, arbitrary $A$-lines followed by
> exactly $k$ lines at $b=A-s$), the value $f(k)$ is **not** convex and the
> Legendre transform **does** have a gap.

Machine checks: 217 convex curves / 0 nonconvex and 198 no-gap / 0 gap for
one budget; 41 convex / 36 nonconvex for the mixed objective. Explicit
mixed nonconvex instance: $w=(1,3,1,2,2,1)$, $A=9$, $s=6$ gives
$f=(1475,1978,1955,2454)$ with differences $(503,-23,499)$. Convexity for a
single budget is the Aggarwal–Schieber–Tokuyama $k$-link theorem; the mixed
case is the precise obstruction behind Proposition G.

> **Proposition O (representation barriers).** For $m\ge3$, $F_m[i,j]$ is
> **not** a function of $C_i+C_j$ alone, and the chain that picks the nearest
> $C$-point to each ideal $\xi_r$ is **not** always optimal.
> - Sum-representation counterexample: $w=(2,3,2,3,4,4,2)$, $d=29$, $m=3$:
>   two endpoint pairs with the same $C_i+C_j=27$ give quantisation energies
>   $2$ and $8$ (one of them is $(i,j)=(1,6)$). A second instance
>   $w=(2,1,1,3,3,4)$, $d=22$, $m=3$ gives energies $14/3$ and $8/3$ at
>   the same $C_i+C_j=20$. Hence $F_m[i,j]$ is not a function of
>   $C_i+C_j$ plus unary terms.
> - Nearest-ideal counterexample: $w=(5,4,3,3,4,2,3)$, $d=33$, $m=3$,
>   $i=3$ gives $F=2301$ while the nearest-ideal chain costs $2307$.

Census: 96 failures of the $C_i+C_j$ representation and 60 failures of the
nearest-ideal chain. Consequently the min-plus powers $T^{2^k}$ do not admit
a compact $O(n)$-parameter representation: the quantisation energy depends on
$m-1$ distinct affine directions $(1-r/m)C_i+(r/m)C_j$, so binary
exponentiation and core-sparse multiplication (Proposition E) both fail to
remove the factor $m$.

> **Proposition P (minimax barrier).** Theorem 15's split
> $\min_j(G[j]+H[j])$ cannot be evaluated by a single Lagrangian parameter
> on the mixed graph. Writing
> $\mathrm{Mix}(\lambda)=\min\{\text{penalty}+\Lambda\cdot\text{lines}
> +\lambda\cdot(\text{$b$-lines})\}$ (one $O(n)$ two-phase DP),
> $$\min_j\bigl(G[j]+H_m[j]\bigr)
> \;>\;\max_\lambda\bigl[\mathrm{Mix}(\lambda)-\lambda m\bigr]$$
> on explicit instances:
>
> | $w$ | $A$ | $s$ | $m$ | $\min_j(G+H)$ | $\max_\lambda[\cdot]$ | gap |
> |---|---|---|---|---|---|---|
> | $(2,1,1,3)$ | 6 | 2 | 2 | 446 | 444 | 2 |
> | $(3,4,2,2,4,2,4)$ | 9 | 2 | 3 | 3501 | 2889 | 612 |
> | $(1,4,1,4,4,3,1)$ | 21 | 8 | 2 | 9458 | 7801 | 1657 |

Even though each $P_i(k)$ is convex (Theorem 18), the maximiser
$\lambda_i^*$ varies with the split point $i$; minimising over $i$ and
maximising over $\lambda$ do not commute. The optimal $\lambda_i^*$ is
monotone in $i$ up to flat-segment tie-breaking (113/120 instances; mean
$2.32$ distinct values per instance), but the number of distinct penalties is
not bounded by a constant, so a single shared $\lambda$ is insufficient.

> **Theorem 19.** For exactly **two** terminal lines and arbitrary word
> widths, all values $H_2[i]$ are computable in $O(n)$ time and the
> lexicographic relational optimum with $m=2$ is therefore $O(n)$. Here
> $d=\beta+1$ and a $b$-line is legal iff its gap $C_j-C_i\le d$:
> $$H_2[i]=\frac{\bigl(2d-(C_n-C_i)\bigr)^2}{2}
> +2\min_{\substack{k:\,i<k<n\\ C_k\in[\,C_n-d,\ C_i+d\,]}}
> \Bigl(C_k-\frac{C_i+C_n}{2}\Bigr)^2 .$$
> The feasible $k$-interval and the target $(C_i+C_n)/2$ are monotone in
> $i$, so a two-pointer scan over $C$ plus a predecessor query computes every
> $H_2[i]$ in $O(n)$ total. Verified on 3\,230 checks: 0 mismatches.

Theorem 19 supplies exactly the quantisation term that Proposition N showed
was missing: the continuous relaxation is off by up to $2\,\mathrm{dist}^2$,
but that term is itself computable in linear time when $m=2$. For $m\ge3$
the $m-1$ coupled directions of Theorem 17 remain the obstruction.

**Summary of excluded compression routes for Question 3.** Continuous
relaxation (Proposition N), nearest-ideal greedy and single-direction
representation (Proposition O), core-sparse Monge multiplication
(Proposition E), single- and two-parameter Legendre transforms
(Propositions F, G), minimax/Lagrangian on the mixed graph (Proposition P),
min-plus binary exponentiation (Proposition O), budget sorting and text
reversal (Propositions I, K). Theorems 14–16 and 19 delimit the solved
subcases; the residual problem is exactly the computation of an
$(m-1)$-direction discrete-spline projection onto the non-uniform grid $C$.

### 2.10 Single-run $o(nK)$ algorithms and the composition barrier

Let all $\beta_t\equiv A$, i.e. $R=1$ constant-valued run, and put
$d=A+1$, $\Lambda=4n(A+1)^2+10$. A line $i\to j$ has slack $d-(C_j-C_i)$
and is legal iff $C_j-C_i\le d$.

> **Theorem 20.** The lexicographic optimum of one uniform-budget run is
> computable in $O(n\log V)$ time with $V=O(nA^2)$, and in
> $$O\!\left(\sqrt{nK(n-K)\log(n-K)}\right)$$
> strongly polynomial time and $O(n)$ space. Both are $o(nK)$.

*Weakly polynomial algorithm.* With the dominating per-line penalty
$\Lambda$ and a Lagrangian price $\lambda$, the unconstrained recurrence
$F[j]=\min_{i<j,\,C_j-C_i\le d}F[i]+(d+C_i-C_j)^2+\Lambda+\lambda$ is a
concave least-weight subsequence problem; Wilber (1988) solves it in
$O(n)$, and an offline segment-tree CHT solves it in $O(n\log n)$. The
$k$-link value $P(K)$ is discrete convex (Theorem 18), so integer binary
search on $\lambda$ recovers exactly $K=L_A(n)$ lines (greedy, $O(n)$) and
$P(K)=G(\lambda)-\lambda K$. Machine check: 1\,800 random instances plus
all small instances, 0 mismatches against exhaustive lexicographic search.

*Strongly polynomial algorithm.* A constant penalty for over-wide lines
breaks Monge (121/200 failures), but the **convex extension with hinge**
$$c(i,j)=\bigl(d-(C_j-C_i)\bigr)^2
+\mu\max\bigl(0,\,C_j-C_i-d\bigr)+\Lambda$$
is a total function of $(i,j)$, convex in the gap, and hence concave Monge
(300/300 quadrangle checks). With $\mu$ dominating, over-wide edges are
never used; $c$ is evaluable in $O(1)$. The same edge weights fed to a naive
$K$-layer DP reproduce the exhaustive lexicographic optimum on 1\,200 random
instances (0 mismatches). The run is therefore a concave Monge DAG with
$n+1$ vertices, and Wan (2024)'s contract-and-conquer shortest $M$-link path
algorithm with $M=K$ gives the stated bound. $\blacksquare$

> **Proposition Q.** "For a sequence of $R$ constant-valued runs the best
> known bound is $O(nK)$." False already at $R=1$ by Theorem 20. What fails
> for $R\ge2$ is **composition**, not the single-run bound.

> **Proposition R (structured source is nonconvex).** Let $F_A[j]$ be the
> exact unconstrained $A$-run LWS vector — the junction vector of Theorem 15
> — and let $P(K)=\min_j\bigl(F_A[j]+\text{$K$-link $b$-cost from }j\text{ to }n\bigr)$.
> Then $P(K)$ is not convex in general: 140 convex / 131 nonconvex curves
> with at least three finite values. Explicit instance
> $w=(2,2,3,4)$, $A=18$, $s=12$ gives
> $P=(28967,43505,43337,57809)$ with differences
> $(14538,-168,14472)$. Hence the Aliens trick cannot be reapplied to the
> second run.

> **Proposition S (super-source breaks Monge).** To apply Wan's algorithm to
> a second run one would add a super-source with edge costs $F_A[j]$; the
> required Monge source condition
> $F_A[l]+c(j,k)\ge F_A[k]+c(j,l)$ for $j<k<l$ fails on 126/300 instances,
> e.g. $w=(4,2,4,4,3,1)$, $A=19$, $s=14$ gives
> $F_A[6]+c(4,5)=4468 < F_A[5]+c(4,6)=4504$. So the Monge-DAG reduction
> does not survive composition either.

> **Proposition T (core density does not collapse).** For the fixed kernel
> $K[i,j]=(C_j-C_i)^2$, the core size of $K^{\otimes q}$ stays quadratic for
> $q=\Theta(n)$: at $n=64$, $q=16$ has 394 nonzero density cells out of
> 1128 finite cells, and $q=32$ has 145 out of 496; at $n=32$, $q=16$ has
> 41 out of 120. Min-plus binary exponentiation with core-sparse
> multiplication therefore costs $\Theta(n^2)=\Theta(nm)$ in the hardest
> regime.

> **Theorem 21 (Wan bound applies to a strict subclass).** For
> equal-width locators with $m=\Theta(n)$ and arbitrary word widths,
> computing the single value $H_m[0]$ (the $m$-link $b$-cost from $0$ to
> $n$) is a shortest $M$-link path problem in a concave Monge DAG with
> $M=\Theta(n)$ whose adjacent density is the rank-one matrix
> $D_{i,j}=2(C_{i+1}-C_i)(C_{j+1}-C_j)$ (Theorem 8). Wan (2024) therefore
> gives
> $O(\sqrt{nM(n-M)\log(n-M)})=O(n\sqrt{n\log n})$ for the hardest
> $M=\Theta(n)$, and $O(n\,\mathrm{polylog}\,n)$ for
> $M=O(\mathrm{polylog}\,n)$ or $M=n-O(\mathrm{polylog}\,n)$.
> General concave Monge DAGs allow an arbitrary nonnegative density matrix
> with $\Theta(n^2)$ degrees of freedom; this subclass has $\Theta(n)$.
> Hence the AST/Wan open question and any lower bound for **arbitrary** Monge
> DAGs do not transfer automatically. The residual problem is to improve
> Wan's bound on this rank-one density subclass, or prove a lower bound for
> it. Machine check: 200 instances, 0 failures of the rank-one density
> formula.

**Consequence for Question 2.** With $R\ge2$ runs, the output of run $r$ is
a general row-potential vector: Theorem 18's convexity is lost (Proposition
R) and the Monge source condition fails (Proposition S), while
core-sparse exponentiation is quadratic (Proposition T). The $o(nK)$
algorithms of Theorem 20 are thus confined to $R=1$; extending them across
a budget change is the exact residual difficulty.

### 2.11 All-endpoint Legendre structure of one budget run

For one uniform budget $d=\beta+1$, let
$$P_j(m)=\min\Bigl\{\sum_{r=1}^m\bigl(d-(C_{j_r}-C_{j_{r-1}})\bigr)^2:\
0=j_0<\cdots<j_m=j\Bigr\}$$
be the exactly-$m$-line value to endpoint $j$, and
$$F_p[j]=\min_{k\ge1}\bigl(P_j(k)+p k\bigr),\qquad p\in\mathbb Z .$$
$F_p[j]$ is one unconstrained LWS pass with per-line price $p$; $p$ may be
negative because the DAG is acyclic and $k\le j$.

> **Theorem 22.** On the feasible endpoints, $P_j(m)$ is discrete convex in
> $m$ and
> $$P_j(m)=\max_{p\in\mathbb Z}\bigl(F_p[j]-p m\bigr).$$
> Moreover the rightmost maximiser $p_j^*=\max\{p:F_p[j]-pm=P_j(m)\}$ is
> nondecreasing in $j$ over feasible endpoints.

Machine checks: 203 feasible endpoints, 0 Fenchel gaps; 0 failures of
$p_j^*$ monotonicity on 120 random instances; the number of distinct
rightmost prices is small in these tests (mean $1.87$, max $4$).

> **Theorem 23 (bounded Fenchel prices).** For a uniform budget $d=\beta+1$,
> let $\Delta_j(k)=P_j(k)-P_j(k-1)$ on the finite range. Then
> $$0\le \Delta_j(k)\le d^2 .$$
> Consequently every finite rightmost price $p_j^*(k)=-\Delta_j(k)$ lies in
> $[-d^2,0]$, and when $P_j(k-1)=\infty$ the boundary price $p=0$ is a
> maximiser. Hence, after marking the endpoints that are infeasible for
> exactly $k$ lines,
> $$P_j(k)=\max_{-d^2\le p\le 0}\bigl(F_p[j]-p k\bigr).$$

*Proof of the upper bound.* Take an optimal $(k-1)$-line layout. Since a
$k$-line layout exists, $j\ge k$, so one of its $k-1$ blocks contains at
least two words. Let that block have gap $g\le d$ and split it at a word
boundary into $u+v=g$ with $u,v\ge2$. The squared-slack increase is
$$(d-u)^2+(d-v)^2-(d-g)^2=d^2-2uv\le d^2 ,$$
so $P_j(k)\le P_j(k-1)+d^2$. The lower bound is the standard monotonicity
of minimum $k$-link cost in a concave Monge DAG: deleting one link and
applying the Monge exchange yields a $(k-1)$-link layout whose cost is no
larger: the Monge exchange repartitions a window of consecutive blocks and
the cost drop is a sum of terms $2uv\ge0$ with $u+v\le d$. Convexity
(Theorem 22) then propagates $\Delta\ge0$ across the finite range. $\blacksquare$ Machine check: 1\,873 finite marginals, 0 violations
of $0\le\Delta\le d^2$; 2\,186 feasible endpoints, 0 bounded-Fenchel gaps.

An endpoint $j$ admits exactly $k$ $d$-lines iff $j\ge k$, every word span
in the prefix is $\le d$, and the greedy minimum line count $L_d(j)\le k$.
All $L_d(j)$ are computed in $O(n)$ by one greedy scan.

### 2.12 Equal-width locators: $O(nA^2)$ with arbitrary word widths

Let the last $m$ lines all have budget $b=A-s$ ($s_r\equiv s$), and put
$d=b+1=A-s+1$. By Theorem 14 the problem is the two-run rank sequence
$[b^m,A,A,\dots]$ on the reversed word list. Let $\Lambda=n(A+1)^2+1$.

> **Theorem 24.** The lexicographic optimum (fewest lines, then least squared
> slack) for $m$ equal-width locator lines and arbitrary word widths is
> computable in
> $$O(n d^2)=O\bigl(n(A-s+1)^2\bigr)\le O(nA^2)$$
> time and $O(n)$ space. For fixed measure $A$ this is $O(n)$, hence
> $o(nm)$ whenever $m=\omega(1)$ and $o(nK)$ for the two-run rank sequence
> with $K=\Theta(n)$.

*Algorithm.* Reverse the word list. Compute all $P_j(m)$ for the uniform
$b$-budget by Theorem 23: for each integer $p\in[-d^2,0]$, run one
$O(n)$ concave LWS pass with per-line price $p$ and update
$P[j]\gets\max(P[j],F_p[j]-pm)$ on the feasible endpoints. This costs
$O(nd^2)$. Compute $G[i]$, the value of an arbitrary number of $A$-lines
covering the first $i$ original words, by one $O(n)$ concave LWS pass with
per-line price $\Lambda$. The answer is
$$\min_{0\le i\le n-m}\Bigl(G[i]+\Lambda m+P[n-i]\Bigr);$$
$\lfloor V/\Lambda\rfloor$ is the line count and $V\bmod\Lambda$ the squared
slack. $\blacksquare$

Machine checks: 960 exhaustive layout comparisons, 0 mismatches;
2\,186 bounded-Fenchel endpoints, 0 mismatches; 1\,873 marginal bounds,
0 violations. The naive LWS inside the verification script is $O(n^2)$;
Wilber's $O(n)$ concave LWS gives the stated bound.

### 2.13 Two-run joint-count Legendre transform has a duality gap

The bounded-price identity of Theorem 23 recovers a single line count $k$
from a one-dimensional price $p$. For arbitrary locator profiles the natural
extension is to attach one price $\lambda_r$ to each constant-budget run and
to recover a joint count vector $(k_1,k_2,\dots)$ by a multidimensional
Legendre transform. That extension fails even for two runs.

> **Proposition V.** Let $d_1,d_2$ be two constant budgets and let
> $P(k_1,k_2)$ be the least squared slack using exactly $k_1$ lines of
> budget $d_1$ followed by exactly $k_2$ lines of budget $d_2$. Put
> $$F(\lambda_1,\lambda_2)=\min_{k_1,k_2}\bigl(P(k_1,k_2)+\lambda_1k_1+\lambda_2k_2\bigr).$$
> There is an instance with
> $$P(1,1)=20>\sup_{\lambda_1,\lambda_2\in\mathbb R}
> \bigl(F(\lambda_1,\lambda_2)-\lambda_1-\lambda_2\bigr)=19 .$$
> The gap remains 19 when the prices are restricted to the natural bounded
> boxes $\lambda_r\in[-d_r^2,0]$ of Theorem 23, and also when the price box
> is enlarged to $[-100,100]^2$.

*Explicit witness.* $w=(2,1)$, so $C=(0,3,5)$; $d_1=5$, $d_2=6$. The finite
values are
$$P(0,1)=1,\quad P(0,2)=25,\quad P(1,0)=0,\quad P(1,1)=20,\quad P(2,0)=13,$$
all other $(k_1,k_2)$ infeasible. Hence, with $x=-\lambda_1$, $y=-\lambda_2$,
$$\begin{aligned}
F&=\min\{1-y,\ 25-2y,\ -x,\ 20-x-y,\ 13-2x\},\\
D(x,y)&:=F+x+y=\min\{1+x,\ 25+x-y,\ y,\ 20,\ 13-x+y\}.
\end{aligned}$$
If $D>19$, the first, second, third and fifth terms force simultaneously
$x>18$, $y<x+6$, $y>19$ and $y>x+6$ — a contradiction. Thus $D\le19$ for
all real prices. At $(x,y)=(25,31)$, i.e.
$(\lambda_1,\lambda_2)=(-25,-31)$,
$D=\min\{26,19,31,20,19\}=19$. $\blacksquare$

Machine checks: the explicit instance gives $P(1,1)=20$ and dual optimum 19;
an exhaustive integer grid $[-100,100]^2$ has maximum 19; 83 of 200 random
two-run instances have a strict gap already inside
$[-d_1^2,0]\times[-d_2^2,0]$.

Consequently the $O(nA^2)$ algorithm of Theorem 24 cannot be obtained by
assigning one bounded price per budget value: the joint count polytope is not
the lower convex envelope of the single-run prices. Any $o(nm)$ algorithm for
arbitrary $(s_r)$ must use a different representation of the run interaction.

### 2.14 Bounded-price Fenchel identity is source-specific

For a uniform budget $d$ and a source vector $G$, let
$$P_G(k,j)=\min\{G[i]+\text{cost of exactly $k$ $d$-lines from $i$ to $j$}\},$$
with $P_G(0,j)=G[j]$, and
$$F^G_p[j]=\min_{k\ge0}\bigl(P_G(k,j)+p k\bigr).$$
Theorem 23 says that when $G=e_0$,
$P_{e_0}(k,j)=\max_{-d^2\le p\le0}(F^{e_0}_p[j]-pk)$. Composing several
constant-budget runs would need the same identity with $G$ equal to the DP
vector of a preceding run.

> **Proposition W.** The assertion
> "$P_G(k,j)=\max_{-d^2\le p\le0}(F^G_p[j]-pk)$ for every source vector $G$"
> is false. It already fails when $G$ is produced by one uniform-budget run.

*Explicit witness.* Let $w=(2,3,1,1)$, so $C=(0,3,7,9,11)$. A preceding run
with $d_0=8$ gives
$$G=(0,25,1,29,17).$$
For the next run with $d_1=5$ and endpoint $j=2$, the only paths are
$$P_G(0,2)=G[2]=1,$$
$$P_G(1,2)=G[1]+(5-(C_2-C_1))^2=25+1=26,$$
$$P_G(2,2)=G[0]+(5-(C_1-C_0))^2+(5-(C_2-C_1))^2=4+1=5.$$
Thus
$$F^G_p[2]=\min\{1,\ 26+p,\ 5+2p\},$$
and for every real $p$,
$$F^G_p[2]-p=\min\{1-p,\ 26,\ 5+p\}\le3,$$
with equality at $p=-2$. The price box of Theorem 23 is
$[-d_1^2,0]=[-25,0]$, which contains $-2$. Hence
$$P_G(1,2)=26>3=\sup_{p\in[-25,0]}\bigl(F^G_p[2]-p\bigr).$$
The gap is exactly the chord of the nonconvex sequence
$P_G(0,2),P_G(1,2),P_G(2,2)=(1,26,5)$ at $k=1$. $\blacksquare$

Machine checks: the explicit structured source gives $P_G(1,2)=26$ and dual
optimum $3$ at $p=-2$; 161 of 300 random preceding-run sources have a strict
gap. For unrestricted source vectors the smaller witness
$w=(3,2,1,3,2,2,3)$, $d=12$,
$G=(0,\infty,34,17,\infty,\infty,\infty,\infty)$ gives
$P_G(1,4)=70$ and bounded dual 61; 364 of 3\,448 finite arbitrary-source
endpoints have a strict gap.

Consequently Theorem 24 is not composable by replacing $e_0$ with the output
of a preceding run and reusing bounded prices. This separates two-run rank
sequences from $R\ge3$ runs at the level of the price representation.

### 2.15 Core-sparse multiplication is not a uniform shortcut

For a bounded difference-budget matrix $M$, let
$$\delta(M)_{i,j}=M[i,j]+M[i+1,j+1]-M[i,j+1]-M[i+1,j]$$
and let $\mathrm{core}(M)$ be the set of $(i,j)$ with all four entries finite
and $\delta(M)_{i,j}\ne0$. Core-sparse min-plus multiplication is fast when
$|\mathrm{core}(M)|=O(n)$; for fixed measure $A$ the two-layer product has
$O(nA)$ core cells, which is consistent with Theorem 24. That bound is not
uniform in $A$.

> **Proposition X.** There are two-layer rank instances with unit words and
> $d_1=d_2=\beta+1=2n$ whose min-plus product matrix has
> $$|\mathrm{core}(M_{d}\otimes M_{d})|=\Theta(n^2).$$
> Hence core-sparse min-plus multiplication, used as a black box, cannot give
> an $o(nK)$ algorithm uniformly over $A=\Theta(n)$ even for $K=2$.

*Proof.* Take $\alpha_t\equiv1$, so $C_j=2j$, and $d=2n$. Let
$Q[i,j]$ be the least cost of exactly two $d$-lines from $i$ to $j$. Put
$m=j-i$ and $a=k-i$ for the intermediate breakpoint $k$. Both windows are
feasible for every $1\le a\le m-1$ because $m\le n$ and $d=2n$. The cost is
$$(2n-2a)^2+\bigl(2n-2(m-a)\bigr)^2
=8\Bigl(n-\tfrac m2\Bigr)^2+8\Bigl(a-\tfrac m2\Bigr)^2 .$$
Minimising over integer $a$ gives
$$Q[i,j]=P(m)=2(2n-m)^2+2\,(m\bmod2),\qquad 2\le m\le n.$$
For an interior cell of gap $m=j-i\ge4$,
$$\delta(Q)_{i,j}=2P(m)-P(m-1)-P(m+1)
=\begin{cases}-8,&m\text{ even},\\0,&m\text{ odd}.\end{cases}$$
All four entries are finite exactly when $m\ge3$, so the nonzero core cells
are precisely the pairs with even $m\in\{4,6,\dots,n-1\}$. Counting
$n-m$ pairs for each such $m$ yields
$$|\mathrm{core}(Q)|=
\begin{cases}
\dfrac{(n-2)(n-4)}{4},&n\text{ even},\\[1ex]
\dfrac{(n-3)^2}{4},&n\text{ odd},
\end{cases}
=\Theta(n^2).$$
Machine checks: the closed form and the exact count agree for
$n=6,8,10,20,30,40$; random words with $d=C_n$ give core ratios
$0.266,0.337,0.338$ of $n^2$ at $n=16,24,32$; with fixed $d=8$ the core
ratio is $0.53n,0.45n,0.56n$ at $n=32,64,128$, i.e. linear. $\blacksquare$

Thus core sparsity is a fixed-measure phenomenon. Wilber's $O(n)$ concave
LWS applies to one fixed kernel; the AST $k$-link theorem applies to one
fixed concave Monge DAG; core-sparse multiplication applies when the product
core is small. None of these three results covers a sequence of rigidly
shifted nonuniform grids with unbounded $d_t$. Any unconditional $o(nK)$
algorithm must exploit a different representation of the sequence
$M_{d_1}\otimes\cdots\otimes M_{d_K}$.

### 2.16 Rank-3 kernels: a single layer is trivial, two layers are not

Take $d\ge C_n$ so that every edge $i<j$ is feasible (window-free). Put
$$W_d[i,j]=(d+C_i-C_j)^2=C_i^2+2C_i(d-C_j)+(d-C_j)^2 .$$

> **Theorem 27.** (i) On the finite band, the ordinary (non-tropical) rank of
> $W_d$ is at most $3$. For any source vector $G$ one layer is a
> **sliding-window convex-hull query** and costs $O(n)$ time and $O(n)$ space:
> $$F[j]=(d-C_j)^2+\min_{\lambda_j\le i<j}\bigl(G[i]+C_i^2+2C_i(d-C_j)\bigr),$$
> where $\lambda_j=\min\{i:C_j-C_i\le d\}$ is nondecreasing in $j$ and the
> query points $x_j=d-C_j$ are nonincreasing. Equivalently the $O(nK)$
> baseline is exactly $K$ such hull queries, and the entire difficulty of
> Question 2 is compressing their **composition**.
>
> (ii) The two-layer product $Q=W_{d_1}\otimes W_{d_2}$ has the closed form
> $$Q[i,j]=\frac{(d_1+d_2-(C_j-C_i))^2}{2}
> +2\,\mathrm{dist}\!\left(\frac{C_i+C_j+d_1-d_2}{2},\,C\right)^2 .$$
> The continuous first term alone has discrete density
> $$\delta\!\left(\tfrac12(D-\Delta)^2\right)_{i,j}
> =-(C_{i+1}-C_i)(C_{j+1}-C_j)<0$$
> at **every** interior cell.
>
> (iii) Consequently low ordinary rank does **not** imply min-plus
> compressibility, and no MinConv/APSP/OMV hardness can be embedded into a
> single layer: a rank-$3$ kernel has only $O(n)$ degrees of freedom and its
> vector product is linear-time. Any conditional lower bound must come from
> at least two composed layers.

*Proof of (ii).* Let $a=d_1+C_i$, $b=d_2-C_j$, and let $y=C_k$. The cost of
splitting at $k$ is $(a-y)^2+(b+y)^2=2(y-\tfrac{a-b}{2})^2+\tfrac{(a+b)^2}{2}$,
so minimising over $C_k\in(C_i,C_j)$ gives the formula with
$\mu=\tfrac{a-b}{2}=\tfrac{C_i+C_j+d_1-d_2}{2}$ and
$\tfrac{(a+b)^2}{2}=\tfrac{(d_1+d_2-(C_j-C_i))^2}{2}$. For the density claim,
write $t=D+C_i-C_j$, $\alpha=C_{i+1}-C_i$, $\beta=C_{j+1}-C_j$; then
$$2\delta=t^2+(t+\alpha-\beta)^2-(t-\beta)^2-(t+\alpha)^2=-2\alpha\beta,$$
so $\delta=-\alpha\beta$. $\blacksquare$ Machine checks: 300 instances with
zero rank-3 identity error; 3\,369 cells with zero spline-form mismatch;
2\,224 interior cells with zero density mismatch.

> **Proposition Y.** For **any two budgets** (equal or different) and arbitrary
> non-uniform $C$, the two-layer full-finite product has
> $|\mathrm{core}(Q)|=\Theta(n^2)$. Explicitly, with $C_j=2j$, $d_1=2a$,
> $d_2=2b$, $a\ge n$, $b\in\{a,a-1\}$, and $m=j-i$,
> $$Q[i,j]=2(a+b-m)^2+2\bigl((m+a-b)\bmod 2\bigr),$$
> $$\delta(Q)_{i,j}=\begin{cases}-8,&m\equiv a-b\pmod 2,\ m\ge3,\\0,&\text{else},\end{cases}$$
> $$|\mathrm{core}(Q)|=\sum_{\substack{3\le m\le n-1\\ m\equiv a-b\ (2)}}(n-m)
> =\Theta(n^2).$$

*Proof.* With $u=k-i\in\{1,\dots,m-1\}$ the split cost is
$4[(a-u)^2+(b-m+u)^2]=8\bigl(u-\tfrac{a-b+m}{2}\bigr)^2+2(a+b-m)^2$; the
nearest integer $u$ gives $8(\cdot)^2\in\{0,2\}$ according to the parity of
$m+a-b$. Since $a\ge n$ and $b\in\{a,a-1\}$, the unconstrained minimiser lies
inside $\{1,\dots,m-1\}$ for every $m\ge2$. Taking the second difference gives
$-8$ on the stated parity and $0$ otherwise. $\blacksquare$

Machine checks: exact counts agree with the sum for $n=6,\dots,21$ and both
parities of $a-b$ (e.g. $n=20$ gives $72$ for $d_1=d_2=40$ and $81$ for
$(40,38)$). For random non-uniform $C$ at $n=16,\dots,64$ with
$(d_1,d_2)\in\{(C_n,C_n),(C_n,C_n/2),(C_n/2,C_n)\}$, the core is
$0.285$–$0.439$ of $n^2$, increasing with $n$. This strictly generalises
Proposition X, which used equal budgets and uniform words.

**What this rules out.** Proposition Y removes the last "structured kernel"
escape: even the algebraically simplest window-free kernel, with a single
degree of freedom per layer ($d_t$), produces a quadratic core after just two
layers, for any two budgets and any word widths. Together with Proposition W
(no price reuse after a preceding run) and Proposition V (joint-count
Legendre gap), every representation so far tried — explicit DP vectors,
core-sparse block products, bounded-price Fenchel, and single-layer hardness
reductions — is excluded. An $o(nK)$ algorithm, or an $\Omega(nK)$-type
conditional lower bound, must therefore act on the **composition of at least
three** shifted hull queries.

### 2.17 The bounded-price representation is tight

Theorem 24 computes the equal-width-locator optimum with $d^2+1$ concave-LWS
passes, one per integer price in $[-d^2,0]$ (Theorem 23). It leaves open
whether the number $D$ of **distinct** optimal prices is $o(d^2)$, which would
allow a faster price search. The answer is negative; the exact worst-case value
of $D$ is $\Theta(\min(n,d^2))$.

Let $\alpha_t\equiv1$, so $C_j=2j$, let $d=\beta+1=2K$ be even, and let
$P_j(k)$ be the least squared slack of exactly $k$ $d$-lines covering the
first $j$ words. Write $j=qk+r$ with $0\le r<k$ and $a=d-2q$ (the slack of a
$q$-word line). Splitting $j$ as evenly as possible gives

$$P_j(k)=(k-r)a^2+r(a-2)^2=ka^2-4r(a-1)
=\frac{(kd-2j)^2}{k}+\frac{4r(k-r)}{k},$$

feasible for $k\le j\le kK$. The first form shows $P_j(k)$ is the continuous
optimum $(kd-2j)^2/k$ plus the exact quantisation energy $4r(k-r)/k$.

> **Theorem 28 (exact price count).** Let $m\ge K$ and $n\ge (m-1)K$. Put
> $\Delta_j(m)=P_j(m)-P_j(m-1)$ and
> $p^*_m(j)=\arg\max_{p\in[-d^2,\,0]\cap\mathbb Z}\bigl(F_p[j]-pm\bigr)$,
> taking the leftmost maximiser, where $F_p[j]=\min_k(P_j(k)+pk)$.
>
> (i) With $j=qm+\rho$, $a=d-2q$, the marginal is
> $$\Delta_j(m)=\begin{cases}
> a^2+4q(a-1),&\rho<m-1-q,\\
> a^2+4q(a-3)+8(m-1-\rho),&\rho\ge m-1-q .
> \end{cases}$$
> The set of values over all finite endpoints is exactly the arithmetic
> progression
> $$\{2d+8t:0\le t<\tbinom K2\},\qquad \tbinom K2=\frac{d(d-2)}8=\Theta(d^2).$$
>
> (ii) The leftmost maximiser of $F_p[j]-pm$ lies in the subgradient interval
> $[-\Delta_j(m+1),-\Delta_j(m)]$ when $P_j(m+1)<\infty$, and equals $-d^2$
> when $P_j(m+1)=\infty$. Hence
> $$D=\bigl|\{p^*_m(j)\}\bigr|=\binom{K-1}{2}+1
> =\frac{d^2}{8}-\frac{3d}{4}+2=\Theta(d^2),$$
> and this value is unchanged for every $m\ge K$, including $m\gg d^2$.
>
> (iii) For arbitrary word widths, $D\le\min(n,d^2+1)$, and the bound is tight
> up to constants by (ii).

*Proof sketch.* (i) $j=q(m-1)+(q+\rho)$, so the $(m-1)$-line quotient is $q'=
q$ with remainder $\sigma=q+\rho$ when $q+\rho<m-1$ (case A), and $q'=q+1$
with $\sigma=q+\rho-(m-1)$ otherwise (case B); $q+\rho\le 2(m-1)$ excludes a
second wraparound. Substituting $b=d-2q'$ into $P_j(m-1)=(m-1)b^2-4\sigma(b-1)$
and subtracting gives the two displayed formulas. Eliminating $q=(d-a)/2$
turns case A into $-a^2+2a(d+1)-2d$ and case B into
$-a^2+2a(d+3)-6d+8(m-1-\rho)$, so the union is an arithmetic progression of
step $8$; its extremes are $2d$ (at $a=2$, case A) and $d^2-8$ (at $a=d-2$),
giving $\binom K2$ values. (ii) Convexity of $P_j(\cdot)$ reduces all $k\ne m$
constraints to $k=m\pm1$; when $P_j(m+1)=\infty$ the lower constraint is
absent and the leftmost price in the search interval is $-d^2$, contributing
the extra $+1$ in $D$. (iii) $D\le n$ is trivial and $D\le d^2+1$ is
Theorem 23. $\blacksquare$

Machine checks: the closed form agrees with the literal layered DP on
1\,307 cells (0 mismatches); the two-case marginal formula and the exact AP
are verified for $d=8,12,16,20,24,32,40$ and $m\in\{K,K+1,2K,5K\}$; direct
Fenchel evaluation confirms the identity $P_j(m)=\max_p(F_p[j]-pm)$, the
subgradient characterisation, and $D=\binom{K-1}{2}+1$ for $d\le24$; the
$\Theta(d^2)$ count persists at $m=5d^2$ (e.g. $d=20$, $m=2000$, 45 distinct
marginals); on random non-uniform $C$ every finite marginal satisfies
$\Delta_j(k)\le d^2-8$ (the $u,v\ge2$ splitting bound) and
$D/\min(n,d^2+1)\le0.714$.

**Consequences.** Theorem 23's price interval $[-d^2,0]$ and Theorem 24's
$O(n d^2)$ are tight **within the price-enumeration method**: even on unit
words, $\Theta(d^2)$ distinct integer prices occur, so $\Theta(d^2)$ concave-LWS
passes are necessary if one pass is used per price. This closes the
``$D=o(m)$?'' question from Theorem 22 with exact constants: for fixed
measure $A$, $D\le d^2+1=O(A^2)=O(1)=o(m)$, but uniformly in $A$ one has
$D=\Theta(\min(n,A^2))$, and with $m=\Theta(A)$, $n=\Theta(A^2)$ the price
staircase is dense ($D=\Theta(n)$). Any $o(nA^2)$ algorithm for equal-width
locators must therefore avoid the price representation entirely — exactly as
Theorem 16 does on arithmetic grids by replacing it with a convolution power.
Combined with Propositions V, W, Y, every compact representation so far tried
(joint-count prices, price reuse after a preceding run, core-sparse products,
low ordinary rank) is excluded for the general rank/relational problem.

---


### 2.18 Nondecreasing rank budgets: interval invariant

Let $d_t=\beta_t+1$ with $d_1\le d_2\le\cdots$ and define
$$g(i,d)=\max\{j>i:C_j-C_i\le d\},\qquad b_0=0,\quad b_t=g(b_{t-1},d_t).$$
Let $R_t=\{j:\text{the first }j\text{ words admit exactly }t\text{ lines with budgets }d_1..d_t\}$.

> **Theorem 29 (interval invariant).** For every $t\ge1$,
> $$R_t=\begin{cases}\varnothing,&R_{t-1}=\varnothing\ \text{or}\ b_t<t,\\[0.5ex]\{t,t+1,\dots,b_t\},&\text{otherwise}.\end{cases}$$
> The sequence $b_t$ is nondecreasing; once $R_t=\varnothing$, all later sets
> are empty. Consequently $K^\*=\min\{t:b_t=n\text{ and no preceding layer is empty}\}$
> is computable in $O(n)$ time and $O(1)$ extra space by two pointers.

*Proof.* $R_{t-1}=\varnothing$ forces $R_t=\varnothing$. Assuming
$R_{t-1}=[t-1,b_{t-1}]$, the maximum of $R_t=\bigcup_{i\in R_{t-1}}(i,g(i,d_t)]$
is $g(b_{t-1},d_t)=b_t$. Any $t$-line layout covers at least $t$ words, so
$\min R_t\ge t$; if word $t$ cannot fit in budget $d_t$ then, by
$d_r\le d_t$ for $r\le t$, it cannot fit in any of the first $t$ budgets, so
$R_t=\varnothing$ and $b_t<t$. Otherwise the all-singleton layout gives
$t\in R_t$. For absence of holes, take $j\in R_t$ with $j+1\le b_t$: if
$j+1\le b_{t-1}$ use $i=j\in R_{t-1}$ and $d_{t-1}\le d_t$; if
$j+1>b_{t-1}$ use $i=b_{t-1}$ and $j+1\le g(i,d_t)=b_t$. $\blacksquare$

Machine checks: 73\,161 exhaustive nondecreasing instances ($n\le5$,
$w_t\le3$, $d_t\le7$) give zero interval-recurrence and zero $K^\*$ mismatches;
6\,000 random instances ($n\le16$) give $\max_t\#\text{intervals}(R_t)=1$ and
zero $K^\*$ mismatches. For contrast, nonincreasing budgets can produce 3
intervals and alternating budgets on a periodic grid produce $\Theta(n)$.

> **Proposition Z.** Nondecreasing budgets do **not** reduce the lexicographic
> objective to an end-indexed DP, and greedy max-reach is not
> lexicographically optimal.
>
> (a) $w=(1,1,1)$, $C=(0,2,4,6)$, $d=(4,5)$: $K^\*=2$; the split
> $w_1\mid w_2w_3$ has penalty $2^2+1^2=5$, while greedy max-reach gives
> $w_1w_2\mid w_3$ with penalty $0^2+3^2=9$. Exhaustively, 4\,193 feasible
> nondecreasing instances have this defect.
>
> (b) Let $L(j)=\min\{t:b_t\ge j\}$ and replace the rank budget of a line
> ending at $j$ by $d_{L(j)}$. For $w=(1,1)$, $C=(0,2,4)$, $d=(2,4)$, one has
> $b=(0,1,2)$, $L(1)=1,L(2)=2$; the end-indexed DP allows a single
> $d_2=4$ line and returns $(1,0)$, whereas the rank problem requires two
> lines and returns $(2,4)$. Exhaustively, 13\,062 feasible instances differ.
> Deleting a line shifts later budgets to smaller values, so the exchange
> argument of Theorem 5 is unavailable.

### 2.19 Reachability-state barriers for arbitrary rank budgets

For general $(d_t)$ the folklore $O(n)$ max-reach recurrence fails.

> **Proposition AB.** Let $w=(1,1,2,2)$, $C=(0,2,4,7,10)$,
> $d=(7,2,3,3)$. Then
> $$R_1=\{1,2,3\},\quad R_2=\{2\},\quad R_3=\{3\},\quad R_4=\{4\},$$
> so $K^\*=4$, while the lazy max-reach recurrence
> $f_0=0,\ f_t=g(f_{t-1},d_t)$ reports $K=3$ because $f_1=3$, $f_2=3$
> (the last word has span $3>d_2=2$), and $f_3=4$. Exhaustive enumeration
> over 2\,000\,718 instances ($n\le5$, $w_t\le3$, $d_t\le7$) gives 393\,417
> mismatches, 4\,549 of them on feasible instances; a smaller witness is
> $w=(1,1,2),\ d=(4,2,3)$, where $R_1=\{1,2\},R_2=\{2\},R_3=\{3\}$.

> **Proposition AC.** Reachable endpoint sets can have $\Theta(n)$ disjoint
> intervals. Let word gaps repeat the pattern $[2,2,B]$ with $B=10$, and let
> $$d_{2r-1}=2B+8=28,\qquad d_{2r}=4 .$$
> Then $\max_t\#\text{intervals}(R_t)$ grows as $n/6+O(1)$:
>
> | $n$ | 30 | 60 | 90 | 120 | 180 | 240 |
> |---|---|---|---|---|---|---|
> | intervals | 6 | 11 | 16 | 21 | 31 | 41 |
>
> Hence any algorithm that stores exact reachable endpoint sets layer by layer
> needs $\Omega(nK)$ bits of state on this family. This is a representation
> lower bound for that state class, not a time lower bound for the problem.

## 3. Competitive ratio of $\Phi$

Let $B_0=A-s$, $\hat B=\max(B_0,2)$, $D=\hat B+2$, $A+1=qD+r$
($0\le r<D$).

> **Theorem 12.** For $0\le s<A$,
> $$\rho(A,s)=\max\Bigl(1,\ \tfrac12\bigl(2q+\mathbf 1_{[r\ge2]}\bigr)\Bigr),$$
> the supremum is attained, and $\rho$ is nondecreasing in $s$.

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

> **Theorem 13.** The frontier is a 3D antichain with at most one point per
> $(L,T)$, and at $L=n+d$ ($d\ge1$) every frontier point has
> $T\le\max(0,s-dA)$. Hence
> $$\max_{w:\,n\text{ words}}|F(w,A,s)|
> =n(s+1)+\sum_{d=1}^{\lceil s/A\rceil}\max\bigl(1,\,s-dA+1\bigr),$$
> equal to $\boxed{n(A+1)+1}$ under $s\le A$ and $\Theta(ns+s^2/A)$ in
> general.

Tightness: $w=(A,\dots,A)$ (36/36 parameter sets). Exhaustive maximisation
for $A\le8,n\le3$ matches the formula everywhere.

**Proposition L.** "At each $L$ the frontier is one chain with constant
$V+T$." False. $w=(1,2)$, $A=3$, $s=3$ has frontier
$$\{(1,1,3),(2,1,2),(3,1,1),(4,1,0),(0,2,2),(1,2,0),(0,3,0)\};$$
at $L=2$ the points $(0,2,2)$ and $(1,2,0)$ have $V+T=2$ and $1$.

**Proposition M.** The frontier of $w=(1,1)$, $A=4$, $s=4$ has five points,
$\{(0,1,3),(1,1,2),(2,1,1),(3,1,0),(0,2,0)\}$.

An output-sensitive algorithm computes the frontier in
$O\bigl((n+\lceil s/A\rceil)(s+1)\bigr)$ time and $O(n+s)$ space; 500 random
instances, 0 mismatches.

---

## 5. Machine verification

| Check | Instances | Result |
|---|---|---|
| `solution.md` baseline exhaustive checks | exhaustive small instances | all groups pass |
| Proposition U offline batched reduction | 1\,050 random | 0 mismatches vs literal DP |
| Theorem 1$'$ $\lambda$ exact characterization | 19\,602 exhaustive/random | 0 mismatches |
| Theorem 1 vs literal quadrangles vs $O(n)$ test | several thousand | 0 mismatches |
| $O(n+U)$ envelope DP vs $O(n^2)$ DP | 3\,300 | 0 mismatches |
| CDQ $O(n\log n)$ DP vs $O(n^2)$ / exhaustive layouts | 2\,080 + 1\,400 | 0 mismatches |
| stack-only DP vs $O(n^2)$ DP | 5\,200 | 0 mismatches, $\le2.67n$ stack operations |
| rank-space DP correctness vs $O(n^2)$ DP | 5\,600 random + exhaustive small | 0 mismatches |
| Proposition D rank-space quadratic writes | $n=6,8,10,20,40,80,120$ | 0 failed checks; $3542$ writes at $n=120$ |
| Theorem 25 prefix-min DP vs $O(n^2)$ DP | 3\,600 random + 4\,680 exhaustive | 0 mismatches |
| Theorem 26 envelope DP vs $O(n^2)$ DP | 3\,600 random + 4\,680 exhaustive | 0 mismatches |
| Theorem 25–26 linear scaling, $D=2$ | $n$ up to $64\,000$, $U\approx2.6\cdot10^{14}$ | 0 mismatches; linear time |
| Theorem 26 envelope activity, small $D$ | 1\,000 instances | up to 5 assigns + 1 pop |
| exhaustive lex on lower-bound family | $n=6,8,10$ | 0 mismatches |
| static successor $\to$ DP rank recovery | 1\,400 | 0 failures |
| monotone start pointer | 4\,200 | 145 counterexamples |
| $O(n)$ rank DP, nonincreasing budgets | 3\,850 | 0 mismatches |
| periodic $O(p_0 n\log n)$ DP | 2\,700 random + exhaustive small | 0 mismatches |
| residue convexity $P_r(q)$ | 1\,233 curves | 0 non-convex cases |
| layer core size | explicit $n=10$ | 36 nonzero cells, $\delta=\Theta(n^2)$ |
| Theorem 8 decomposition identity | 500 | 0 failures |
| one-parameter Legendre gap | 3 explicit instances | gaps $1,1,2$ |
| two-parameter Legendre gap | 250 random + explicit | gaps $0..21$, explicit gap $13$ |
| uniform-width rank allocation | 2\,500 | 0 mismatches |
| uniform-width relational allocation | 15\,120 | 0 mismatches |
| budget-permutation counterexamples | explicit | 3 distinct pairs |
| $O(n+m)$ relational line count, $b_1\le\cdots\le b_m$ | 266\,345 exhaustive | 0 failures |
| Theorem 14 reversal equivalence | 170\,352 exhaustive | 0 inequivalences |
| Theorem 15 uniform-tail decomposition | 7\,200 random | 0 mismatches |
| Theorem 16 kernel closed form $h^{*m}$ | 9\,072 | 0 mismatches |
| Theorem 16 $O(n)$ equal-grid algorithm | 2\,574 exhaustive | 0 mismatches |
| Proposition N non-uniform-grid barrier | explicit | gaps $2,2,4$ |
| Theorem 17 discrete-spline decomposition | 707 random | 0 failures |
| Theorem 18 one-budget convexity / Fenchel | 217 curves / 198 points | 0 nonconvex / 0 gap |
| Theorem 18 mixed-objective nonconvexity | 77 curves | 36 nonconvex, explicit witnesses |
| Proposition O representation barriers | 200 random instances | 96 sum-representation + 60 nearest-chain failures |
| Proposition P minimax gap | 3 explicit instances | gaps $2,612,1657$ |
| Theorem 19 two-line closed form $H_2[i]$ | 3\,230 checks | 0 mismatches |
| Fenchel penalty monotonicity | 120 instances | 116 monotone, mean $2.65$ distinct penalties |
| Theorem 20 Aliens + segment-tree LWS | 1\,800 random + exhaustive small | 0 mismatches |
| Theorem 20 convex-hinge DP vs exhaustive | 1\,200 random | 0 mismatches |
| Theorem 20 convex-hinge Monge | 300 | 0 failures; constant penalty fails 121/200 |
| Proposition R structured-source convexity | 271 curves | 140 convex / 131 nonconvex |
| Proposition S super-source Monge | 300 | 126 failures |
| Proposition T core density of $K^{\otimes q}$ | $n=32,64$ | $\Theta(n^2)$ for $q=\Theta(n)$ |
| Theorem 21 rank-one Monge density | 200 | 0 failures |
| Theorem 22 all-endpoint Fenchel / price monotonicity | 203 endpoints / 120 instances | 0 gaps / 0 failures, mean $1.87$ distinct prices |
| Theorem 23 marginal bound $0\le\Delta\le d^2$ | 1\,873 marginals | 0 violations |
| Theorem 23 bounded-price Fenchel | 2\,186 feasible endpoints | 0 mismatches |
| Theorem 24 equal-width locator algorithm | 960 exhaustive layouts | 0 mismatches |
| Proposition V explicit joint Legendre gap | $w=(2,1),d=(5,6)$ | $P(1,1)=20$, dual supremum $19$ |
| Proposition V random two-run gaps | 200 instances | 83 with a gap in $[-d_r^2,0]^2$ |
| Proposition W structured-source bounded-price gap | explicit $w=(2,3,1,1)$ | $P_G(1,2)=26$, dual supremum $3$ |
| Proposition W random preceding-run sources | 300 instances | 161 with a strict gap |
| Proposition W random arbitrary sources | 3\,448 finite endpoints | 364 strict gaps |
| Proposition X two-layer closed form and exact core count | $n=6,8,10,20,30,40$ | 0 mismatches |
| Proposition X random large-budget core | $n=16,24,32$ | core ratios $0.266,0.337,0.338$ of $n^2$ |
| Proposition X fixed-budget core | $n=32,64,128,d=8$ | core $0.53n,0.45n,0.44n$ |
| Theorem 27 rank-3 identity | 300 instances | 0 identity errors |
| Theorem 27 two-layer spline form | 3\,369 cells | 0 mismatches |
| Theorem 27 continuous-term density $-\alpha\beta$ | 2\,224 interior cells | 0 mismatches |
| Proposition Y exact two-budget core counts | $n=6,\dots,21$, both parities | 0 mismatches |
| Proposition Y non-uniform core | $n=16,\dots,64$, 3 budget pairs | $0.285$–$0.439$ of $n^2$ |
| Theorem 28 closed form $P_j(k)$ vs layered DP | 1\,307 cells | 0 mismatches |
| Theorem 28 marginal AP and two-case formula | $d\le40$, $m\in\{K,K+1,2K,5K\}$ | 0 mismatches |
| Theorem 28 exact price count $D=\binom{K-1}{2}+1$ | $d=8,12,16,20,24$ direct Fenchel; $d\le40$ subgradient | 0 mismatches |
| Theorem 28 persistence at $m\gg d^2$ | $m\le5d^2$, $n\le19\,990$ | $\binom K2$ distinct marginals |
| Theorem 28 general marginal bound $\Delta\le d^2-8$ | 160 random non-uniform instances | 0 violations |
| Theorem 29 interval invariant and $O(n)$ line count | 73\,161 exhaustive + 6\,000 random | 0 mismatches |
| Proposition Z greedy / end-indexed reduction | 34\,031 feasible instances | 4\,193 / 13\,062 mismatches |
| Proposition AB max-reach recurrence | 2\,000\,718 exhaustive | 393\,417 mismatches |
| Proposition AC reachable intervals | $n=30,\dots,240$ | $41$ intervals at $n=240$ |
| Proposition AA nonmonotone envelope starts | explicit + 1\,000 random | 109 nonmonotone |
| Theorem 12 formula | all sequences of span $\le A$, $A\le14$ | 0 mismatches |
| Theorem 13 bound | exhaustive $A\le8,n\le3$ + 600 random | 0 violations |
| output-sensitive frontier algorithm | 500 random | 0 mismatches |

`asymmetric_line_breaking_algorithms.py` defines all algorithms and checks;
running that file executes them in dependency order, including the baseline
brute-force checks for `solution.md`. `open_problems_asymmetric_line_breaking.md`
restates the open questions with all definitions included. The code requires
only the standard library.

---

## 6. Open questions

1. **Exact word-RAM complexity of the end-indexed DP (0.1).** With
   $w$-bit integer coordinates, $D=|\{z_1,\dots,z_n\}|$ and $U$ the integer
   coordinate span, the known bounds are $O(n+U)$ by direct addressing,
   $O(n\log w)$ with a y-fast trie, $O(n\log n/\log w)$ with a fusion tree,
   $O(n\log n)$ comparisons by CDQ, and $O(n\min(D,\log n))$ by the
   prefix-minima/envelope methods. The comparison complexity is
   $\Theta(n\log n)$ (Theorems 2–3); $D=O(1)$ and $U=O(n)$ are solved in
   $O(n)$ (Theorems 25–26). Proposition U shows that online static-predecessor
   lower bounds do not transfer to the offline DP; Proposition D rules out the
   naive rank-space implementation; Proposition AA shows that the lower
   envelope requires full union-split-find rather than monotone split-only.
   **Decide** whether (0.1) admits an $O(n)$ word-RAM algorithm when
   $D=\Theta(n)$ and $w=O(\log n)$, or prove a super-linear cell-probe lower
   bound. A full interval union-split-find structure gives $O(\log\log D)$ per
   boundary operation, but the crossing-point search against the current
   piecewise-linear envelope is not yet reduced to $O(1)$ or $O(\log w)$
   probes.
2. **General rank-indexed budgets.** Let $K$ be the optimum line count and
   $R$ the number of constant-budget runs in $(\beta_t)$. Decide whether the
   lexicographic optimum admits $o(nK)$ on non-uniform word widths for
   $R\ge3$ runs or arbitrary nonconstant profiles, or prove a conditional
   $\Omega(nK)$-type lower bound. Solved regimes: nonincreasing budgets
   (Theorem 5, $O(n)$), nondecreasing budgets (Theorem 29, $O(n)$ for the
   line count), periodic budgets with period $p$ and $p_0=o(K/\log n)$
   (Theorem 6), uniform word widths (Theorem 9, $O(n\log n)$), a single run
   (Theorem 20, $o(nK)$ via Lagrangian search or Wan's contract-and-conquer
   bound), two constant runs $[b^m,A^\infty]$ with fixed $A,b$ (Theorem 24,
   $O(nA^2)$), and equal-width locators on arbitrary widths (Theorem 24).
   Barriers exclude: structured-source convexity (Proposition R), Monge
   source conditions (Proposition S), core-sparse powers (Proposition T),
   joint-count Legendre prices (Proposition V), source-reuse prices
   (Proposition W), quadratic two-layer cores for any two budgets on a
   rank-3 kernel (Theorem 27, Propositions X–Y), bounded-price Fenchel
   compression (Theorem 28), greedy max-reach and end-indexed reduction
   (Proposition Z), exact reachability sets (Propositions AB–AC), and
   low-ordinary-rank single-layer hardness reductions (Theorem 27). Any
   conditional lower bound must use at least three composed layers.
3. **Relational secondary objective with arbitrary locator profiles.** By
   Theorem 14 the problem is exactly Question 2 restricted to
   $[b_1,\dots,b_m,A,A,\dots]$ with $b_r=A-s_r$; by Theorem 15 it is exactly
   the compression of $m$ prefix layers into $o(nm)$; by Theorem 17 it is an
   $(m-1)$-direction discrete-spline projection onto the non-uniform grid
   $C$. Solved regimes: equal-width locators with arbitrary word widths in
   $O(nA^2)$ (Theorem 24, tight price count Theorem 28), equal-width
   locators on uniform word widths in $O(n)$ (Theorem 16), arbitrary profiles
   on uniform word widths in $O(n\log n)$ (Theorem 10), $m=2$ on arbitrary
   $C$ in $O(n)$ (Theorem 19), and the line count when locator widths are
   nonincreasing from the end in $O(n+m)$ (Theorem 11). **Decide** whether
   arbitrary $(s_1,\dots,s_m)$ on non-uniform widths admits $o(nm)$, or
   prove an $\Omega(nm)$-type conditional lower bound from the composition
   of at least three shifted hull queries.

The closed forms for the competitive ratio (Theorem 12) and the Pareto
frontier size (Theorem 13) are not open.

External results used: SMAWK/Monge searching (Aggarwal, Klawe, Moran, Shor,
Wilber, 1987), the least-weight subsequence problem (Hirschberg–Larmore,
1987; Wilber, 1988; Galil–Park, 1990), minimum-weight $k$-link paths in
concave Monge graphs and contract-and-conquer shortest $M$-link paths
(Aggarwal, Schieber, Tokuyama, 1994; Wan, arXiv:2408.00227, 2024), interval
union-split-find and its lower bounds (Gabow–Tarjan, 1985;
Mehlhorn–Näher–Alt, 1990; Beame–Fich, 2002), online Monge DP
(Bar-Noy–Golin–Zhang, 2009),
core-sparse Monge matrix multiplication (Gawrychowski et al., ESA 2025),
y-fast tries, fusion trees, linear-time word-RAM integer sorting for
$w=O(\log n)$ (radix sort; Andersson–Hagerup–Nilsson–Raman, 1998), and the
exact static predecessor time-space trade-off (Pătrașcu–Thorup, STOC 2006).
