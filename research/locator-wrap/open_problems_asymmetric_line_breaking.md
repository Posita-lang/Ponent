# Open Problems in Asymmetric Line Breaking

All notation, models, and known results needed below are stated explicitly. The
three questions concern exact algorithms and conditional lower bounds for
line-breaking DPs with nonuniform budgets. Two questions from the originating
paper are settled; their closed forms are included in Section 6 for
orientation.

## 1. Notation, models, and standard structures

### 1.1 Words, lines, and the lexicographic objective

Let $w_1,\dots,w_n$ be words with integer widths
$\alpha_t=|w_t|\in\mathbb Z_{>0}$. Put

$$
C_0=0,\qquad C_i=C_{i-1}+\alpha_i+1\quad(1\le i\le n).
$$

Thus $C_i-C_{i-1}\ge2$. A line holding the consecutive words
$w_{i+1},\dots,w_j$ has width

$$
B(i,j)=C_j-C_i-1 .
$$

Under a budget $\beta\in\mathbb Z_{>0}$ its slack is $\beta-B(i,j)$ and its
cost is $(\beta-B(i,j))^2$. Every word is assumed to fit its line unless stated
otherwise. A layout partitions $\{1,\dots,n\}$ into consecutive nonempty
blocks $S_1,\dots,S_L$. The objective is lexicographic:

$$
\text{minimise } L,\quad\text{then minimise } \sum_{t=1}^L(\beta_t-\mathrm{width}(S_t))^2 .
$$

A dominating penalty is any integer
$\Lambda>nB_{\max}^2$, where $B_{\max}$ bounds every achievable squared
slack. Adding $\Lambda$ per line encodes the lexicographic objective in a
single additive shortest-path DP.

Greedy wrapping at budget $\beta$ takes maximal feasible blocks. Let
$L_\beta(m)$ be the number of greedy blocks covering the first $m$ words.
When every word fits, greedy minimises the block count.

### 1.2 Computational models

The default model is the word RAM with machine word size $W=\Theta(\log n)$,
unit-cost arithmetic, bitwise operations, shifts, and multiplication. Integer
coordinates lie in a universe of size $U=2^W$; polynomial universes mean
$U=n^{O(1)}$. The comparison model allows only order comparisons. The pointer
machine and the cell-probe model are used for lower bounds. Let

* $D$ be the number of distinct query coordinates (defined per problem),
* $K$ be the optimum number of lines,
* $R$ be the number of constant-budget runs in a budget sequence,
* $m$ be the number of locator lines in the relational variant.

### 1.3 Structures and standard problems

For a matrix $M$, the discrete mixed second difference is

$$
\delta(M)_{i,j}=M[i,j]+M[i+1,j+1]-M[i,j+1]-M[i+1,j].
$$

$M$ is Monge when $\delta(M)_{i,j}\ge0$ for the relevant orientation, and
concave Monge when $\delta(M)_{i,j}\le0$. $M$ is totally monotone when row
minima are monotone; SMAWK (Aggarwal, Klawe, Moran, Shor, Wilber, 1987) finds
all row minima of a totally monotone matrix in linear time.

The least-weight subsequence (LWS) problem has recurrence
$F[j]=\min_{i<j}(F[i]+c(i,j))$ with a concave Monge cost matrix; Wilber
(1988) and Galil–Park (1990) give $O(n)$ algorithms for the concave case.

The min-plus product is $(A\otimes B)[i,j]=\min_k(A[i,k]+B[k,j])$. The
**core** of $A\otimes B$ is the set of cells with nonzero discrete density
$\delta(A\otimes B)_{i,j}$; core-sparse multiplication is fast when this set
is small.

A concave Monge DAG on vertices $0,\dots,n$ has edge weights $c(i,j)$ with
$c(i,l)+c(j,k)\ge c(i,k)+c(j,l)$ for $i<j<k<l$. A shortest $K$-link path is
a minimum-weight path with exactly $K$ edges. Aggarwal–Schieber–Tokuyama
(1994) and Wan (arXiv:2408.00227, 2024) give subquadratic algorithms; Wan's
bound is $O(\sqrt{N K (N-K)\log(N-K)})$ with $O(N)$ space.

The Fenchel/Legendre transform used here is
$P(k)=\max_{p\in\mathbb Z}(F(p)-pk)$ with
$F(p)=\min_k(P(k)+pk)$. A strict duality gap means this identity fails.

Interval union-split-find maintains a partition of $\{0,\dots,D-1\}$ and
supports $\mathrm{split}(x)$, $\mathrm{union}(x)$, and $\mathrm{find}(r)$.
Split-only and union-only variants admit $O(1)$ amortized RAM solutions
(Gabow–Tarjan, 1985); the full problem has $O(\log\log D)$ per operation via
van Emde Boas trees and pointer-machine $\Omega(\log\log D)$ and cell-probe
$\Omega(\log\log k/\log\log\log k)$ lower bounds (Mehlhorn–Näher–Alt, 1990;
Beame–Fich, 2002).

Conditional lower bounds may be based on SETH, APSP, min-plus convolution
(MinConv), or online matrix-vector multiplication (OMV). Static predecessor
time-space trade-offs are known exactly up to constants (Pătrașcu–Thorup,
STOC 2006).

## 2. Problem A: end-indexed budgets

Let the budget of a line be a function of its ending index $j$: line
$w_{i+1},\dots,w_j$ has budget $\beta(j)$. Define

$$
z_j=C_j-\beta(j)-1,\qquad
\lambda_j=\min\{i<j:C_i\ge z_j\},
$$

with $\lambda_j=j$ when the set is empty. Let $\Lambda$ dominate all
achievable penalties. The shortest-path DP is

$$
dp[0]=0,\qquad
dp[j]=\Lambda+\min_{\substack{i<j\\ C_i\ge z_j}}
\bigl(dp[i]+(C_i-z_j)^2\bigr).
\tag{A}
$$

Equivalently,
$dp[j]=\Lambda+z_j^2+\min_{\lambda_j\le i<j}(dp[i]+C_i^2-2C_i z_j)$. The
cost matrix is
$M[j,i]=(C_i-z_j)^2$ when $C_i\ge z_j$, and $\infty$ otherwise.

### Known results

* Theorem 1: $M$ is totally monotone iff no breakpoint $C_i$ with
  $i\le j-2$ lies in $[z_{j'},z_j)$ for any $j<j'$. With suffix minima
  $m_j=\min_{j'>j}z_{j'}$, the condition is: no such $C_i$ lies in
  $[m_j,z_j)$. An $O(n)$ decision procedure follows.
* The sufficient condition $z_1\le\cdots\le z_n$ makes $M$ Monge and SMAWK
  solves (A) in $O(n)$; it contains the classical uniform-budget case and the
  single terminal decrease $A\mapsto A-s$.
* Upper bounds for arbitrary end-indexed budgets:
  $O(n+U)$ by direct addressing; $O(n\log W)$ with a y-fast trie;
  $O(n\log n/\log W)$ with a fusion tree; $O(n\log n)$ comparisons by CDQ;
  $O(n\min(D,\log n))$ by prefix-minima and lazy-envelope methods, where
  $D=|\{z_1,\dots,z_n\}|$. When $D=O(1)$ or $U=O(n)$, (A) is $O(n)$.
* The comparison complexity is $\Theta(n\log n)$ for unbounded integer
  coordinates (Theorem 3).
* Proposition U: the online static-predecessor lower bound does not transfer.
  The reduction that encodes $m$ predecessor queries into $n=2m$ DP rows is
  solvable in $O(n)$ word-RAM time for polynomial universes by radix sorting
  and merging, although online queries cost $\Theta(\log\log n)$ each.
* Proposition D: the naive rank-space array implementation performs
  $\Theta(n^2)$ writes on a polynomial-coordinate instance with
  $W=O(\log n)$.
* Proposition AA: let $X_0<\cdots<X_{D-1}$ be the sorted distinct $z_j$ and
  let $s_i$ be the rank at which line
  $\ell_i(x)=dp[i]+C_i^2-2C_i x$ first owns the lower envelope (rightmost-tie
  rule). The sequence $s_i$ need not be nondecreasing. Witness:
  $w=(1,5,4,1,4,6,6)$, $C=(0,2,8,13,15,20,27,34)$,
  $z=(-320,5,1,-343,-72,-98,-61)$ gives
  $(s_0,\dots,s_7)=(0,5,5,6,6,5,5,5)$. Hence split-only monotone interval
  split-find is insufficient; the envelope update needs full
  union-split-find. Locating the first rank at which a new line beats the
  current piecewise-linear envelope still requires $O(\log D)$ envelope
  probes with current methods, so $O(n\log\log n)$ is not established by this
  route.

### Open Problem A

Determine the exact word-RAM complexity of (A) on polynomial integer
universes ($U=n^{O(1)}$, $W=\Theta(\log n)$) when $D=\Theta(n)$.

**A1.** Does an $O(n)$ algorithm exist, or is $\omega(n)$ necessary in the
cell-probe model?

**A2.** Is there an $O(n\log\log n)$ algorithm, i.e. can the crossing-point
search against the dynamic lower envelope be reduced to $O(\log W)$ probes
per insertion using union-split-find and integer data structures?

**A3.** Can the pointer-machine $\Omega(\log\log D)$ or cell-probe
$\Omega(\log\log k/\log\log\log k)$ bounds for union-split-find be lifted to
a super-linear lower bound for (A), or does offline access to all $z_j$ admit
linear time?

## 3. Problem B: rank-indexed budgets

Now the $t$-th line has budget $\beta_t$ independent of where it ends; this is
TeX's `\parshape` / hanging-indent model. Put $d_t=\beta_t+1$. With a
dominating penalty $\Lambda$ the layered DP is

$$
F_0[0]=0,\qquad
F_t[j]=\min_{\substack{i<j\\ C_j-C_i\le d_t}}
\bigl(F_{t-1}[i]+(d_t-(C_j-C_i))^2+\Lambda\bigr),
\tag{B}
$$

and the answer is the first $t$ with $F_t[n]<\infty$, decoded as
$(\text{lines},\text{penalty})=(F_t[n]/\Lambda,\;F_t[n]\bmod\Lambda)$.

With $S_0=0$, $S_t=\sum_{r=1}^t d_r$, and breakpoints
$0=j_0<\cdots<j_K=n$, define $X_t=C_{j_t}-S_t$. Legality is
$-\beta_t<X_t-X_{t-1}\le0$ and the penalty is
$\sum_{t=1}^K(X_{t-1}-X_t)^2$. Thus (B) is a convex chain through the
shifted point sets $P_t=\{C_j-S_t:0\le j\le n\}$.

### Solved regimes

* Nonincreasing $(\beta_t)$: $O(n)$ by the endpoint-rank lemma (Theorem 5).
* Nondecreasing $(\beta_t)$: $O(n)$ for the line count by the interval
  invariant $R_t=[t,b_t]$ with $b_t=g(b_{t-1},d_t)$ and
  $g(i,d)=\max\{j:C_j-C_i\le d\}$ (Theorem 29). The secondary objective is
  not solved.
* Periodic $(\beta_t)$ with period $p$: $O(\min\{nK,\;p_0 n\log n\})$ time,
  $p_0=\min(p,K)$ (Theorem 6).
* Uniform word widths $\alpha_t=\alpha$: $O(n\log n)$ for arbitrary budgets
  by convex separable resource allocation (Theorem 9).
* A single constant-budget run ($R=1$): $O(n\log V)$ weakly polynomial via
  Lagrangian search, or
  $O(\sqrt{nK(n-K)\log(n-K)})$ strongly polynomial via Wan's
  contract-and-conquer bound (Theorem 20).
* Two constant runs $[b^m,A^\infty]$ with fixed measure $A$ and fixed $b$:
  $O(nA^2)$ (Theorem 24).
* Equal-width locators, arbitrary word widths, arbitrary $m$: $O(nA^2)$
  (Theorem 24), with a tight $\Theta(\min(n,A^2))$ number of distinct
  Fenchel prices (Theorem 28).
* Exactly $m=2$ terminal lines on arbitrary $C$: $O(n)$ (Theorem 19).
* Line count when locator widths are nonincreasing from the end: $O(n+m)$.

### Barriers

* Lagrangian $\lambda$-shift fails from structured or general source vectors:
  $w=(1,2,1,1)$, $\beta=2$, source $(\infty,\infty,13,9,1)$ gives
  $P=(1,10,15)$ with differences $9,5$, which is concave.
* The endpoint-rank lemma fails for nondecreasing budgets; reversing the text
  is invalid; permuting $(\beta_t)$ changes the answer
  ($w=(1,1,2)$: $\beta=(2,5)$ gives $(2,2)$ while $\beta=(5,2)$ gives
  $(2,4)$).
* Core-sparse min-plus multiplication is not a uniform shortcut. The
  window-free one-layer kernel $W_d[i,j]=(d+C_i-C_j)^2$ has ordinary rank
  $\le3$ and an $O(n)$ sliding-window convex-hull product, but for any two
  budgets and any nonuniform $C$ the two-layer product has
  $\Theta(n^2)$ core (Theorem 27, Propositions X–Y). Therefore MinConv,
  APSP, and OMV hardness cannot be embedded in a single layer; any
  conditional lower bound needs at least three composed layers.
* Bounded-price Fenchel is tight but not compressible: the price interval is
  $[-d^2,0]$ and the number of distinct optimal prices is
  $\Theta(\min(n,d^2))$ (Theorem 28). Joint-count Legendre transforms for two
  runs have a strict duality gap (Proposition V), and the identity is
  source-specific (Proposition W).
* Exact reachability states are large. The $O(n)$ max-reach recurrence fails
  (Proposition AB: $w=(1,1,2,2)$, $d=(7,2,3,3)$ gives $K^\*=4$ but lazy
  max-reach reports $3$). Reachable endpoint sets can have $\Theta(n)$
  disjoint intervals on a periodic grid with alternating budgets
  (Proposition AC), so any algorithm storing exact reachable sets layer by
  layer needs $\Omega(nK)$ state on that family.
* For nondecreasing budgets, greedy max-reach is not lexicographically
  optimal and the natural end-indexed reduction fails (Proposition Z).

### Open Problem B

Decide whether the lexicographic optimum of (B) can be computed in $o(nK)$
time for nonuniform word widths, integer budgets polynomial in $n$, and

1. $R\ge3$ distinct constant-budget runs, or
2. arbitrary nonconstant $(\beta_t)$, or
3. nondecreasing $(\beta_t)$ restricted to the secondary objective.

Either give a non-Lagrangian, non-core-sparse algorithm, or prove a
conditional $\Omega(nK)$-type lower bound (SETH, APSP, MinConv, OMV) for the
layered squared-slack DP. Note that $K=\Theta(n)$ is the hardest regime, so
$o(nK)$ means $o(n^2)$.

## 4. Problem C: relational secondary objective with arbitrary locator profiles

Let the last $m$ lines each carry an indivisible locator: the $r$-th line from
the end has width $s_r$ and budget $b_r=A-s_r$, while all preceding lines
have budget $A$. The objective is again fewest lines, then least squared
slack. The known DP is $O(nm)$: longest-feasible-suffix layers for the line
count, then a backward layered DP with a monotone convex hull for the slack.

Theorem 14 (reversal equivalence): on reversed words $w^R$ the problem equals
the rank-indexed optimum with budgets
$[b_1,\dots,b_m,A,A,\dots]$, restricted to layouts with at least $m$ lines.
Theorem 15 (tail decomposition): with $\Lambda=nA^2+1$, the optimum is
$\min_j(G[j]+H[j])$, where $G[j]$ is one unconstrained $A$-budget LWS pass
($O(n)$) and $H[j]$ is the cost of $m$ terminal lines on the suffix
($O(nm)$). Theorem 17 reduces $H[j]$ to an $(m-1)$-direction discrete-spline
projection onto the nonuniform grid $C$.

### Solved regimes

* Equal-width locators ($s_1=\cdots=s_m=s$), arbitrary word widths:
  $O(nA^2)$ (Theorem 24).
* Equal-width locators on uniform word widths: $O(n)$ (Theorem 16).
* Arbitrary profiles on uniform word widths: $O(n\log n)$ (Theorem 10).
* $m=2$ on arbitrary $C$: $O(n)$ (Theorem 19).
* Line count when widths are nonincreasing from the end: $O(n+m)$.

### Barriers

Continuous relaxation, nearest-ideal-point greedy, single-direction
representations, core-sparse products, single- and double-parameter
Lagrangian searches, minimax relaxations, min-plus binary exponentiation,
budget sorting, text reversal, and low-ordinary-rank single-layer reductions
are all excluded by explicit counterexamples (Propositions F–G, J, N, O, P,
R, S, T, V, W, X, Y, Z, AA; Theorems 17–18, 27–28). Mixed-budget
$P(k)$ is nonconvex while single-budget $P_j(k)$ is discrete convex
(Theorem 18). The continuous spline term has strictly negative density at
every interior cell, and the grid quantisation energy prevents a closed-form
convolution power on nonuniform $C$.

### Open Problem C

Decide whether arbitrary profiles $(s_1,\dots,s_m)$ on nonuniform word widths
admit $o(nm)$ for the lexicographic optimum, or prove an $\Omega(nm)$-type
conditional lower bound. Equivalently, decide whether the $(m-1)$-direction
discrete-spline projection onto a nonuniform grid can be computed without
explicitly processing all $m$ prefix layers. Any lower bound must use at
least three composed shifted-hull layers.

## 5. Settled results included for orientation

**Locator problem, base theorem.** An instance $(A,s)$ is feasible iff
$n=0$ and $s\le A$, or $n\ge1$, $\max_t\alpha_t\le A$ and
$\alpha_n\le A-s$. For a feasible instance with $n\ge1$, let
$j^\*=\min\{j: B(j-1,n)\le A-s\}$ be the first word of the longest suffix
that fits on the last line. Then
$\mathrm{OPT}=L_A(j^\*-1)+1$, computable in $O(n)$ time and $O(1)$ extra
space. Greedy wrapping minimises the block count when every word fits. The
two-phase composition $\Phi$ wraps at $A$, and re-wraps the last greedy block
at $A-s$ when its width plus $s$ exceeds $A$; it is legal iff the instance is
feasible, and $\mathrm{lines}(\Phi)\ge\mathrm{OPT}$. These results, together
with paragraph additivity, the constant-total-slack invariant, the
$(i,k)$ DP (naive $O(n^2K)$, $O(nK)$ with a monotone convex-hull trick per
level), the Monge inequality for the mixed body/terminal matrix, the
$O(n)$ whole-problem algorithm obtained by adding a dominating per-line
constant and running an online monotone convex-hull DP, and the Pareto
dominance table with the no-hybrid theorem for the infeasible region, are
proved in `solution.md`.

**Competitive ratio of two-phase composition $\Phi$.** Let
$B_0=A-s$, $\widehat B=\max(B_0,2)$, $D=\widehat B+2$, and write
$A+1=qD+r$ with $0\le r<D$. Then

$$
\rho(A,s)=\max\Bigl(1,\ \tfrac12\bigl(2q+\mathbf 1_{[r\ge2]}\bigr)\Bigr).
$$

The supremum is attained and $\rho$ is nondecreasing in $s$ (Theorem 12).

**Pareto frontier of failure.** In the infeasible region, the maximum number
of Pareto-optimal loss vectors $(V,L,T)$ over $n$-word instances is

$$
\max_w|F(w,A,s)|
=n(s+1)+\sum_{d=1}^{\lceil s/A\rceil}\max\bigl(1,\,s-dA+1\bigr),
$$

equal to $n(A+1)+1$ when $s\le A$ and $\Theta(ns+s^2/A)$ in general
(Theorem 13). An output-sensitive algorithm computes the frontier in
$O((n+\lceil s/A\rceil)(s+1))$ time.

## 6. References

* A. Aggarwal, M. Klawe, S. Moran, P. Shor, R. Wilber, *Geometric
  applications of a matrix-searching algorithm*, Algorithmica 2 (1987).
* A. Aggarwal, B. Schieber, T. Tokuyama, *Finding a minimum-weight
  $K$-link path in graphs with the Monge property*, Discrete & Computational
  Geometry 12 (1994).
* J. Z. Wan, *Finding a shortest $M$-link path in a Monge directed acyclic
  graph*, arXiv:2408.00227 (2024).
* R. Wilber, *The concave least-weight subsequence problem revisited*,
  Journal of Algorithms 9 (1988).
* Z. Galil, K. Park, *A linear-time algorithm for concave one-dimensional
  dynamic programming*, Information Processing Letters 33 (1990).
* H. Gabow, R. Tarjan, *A linear-time algorithm for a special case of
  disjoint set union*, Journal of Computer and System Sciences 30 (1985).
* K. Mehlhorn, S. Näher, H. Alt, *A lower bound for the complexity of the
  union-split-find problem*, SIAM Journal on Computing 17 (1990).
* P. Beame, F. Fich, *Optimal bounds for the predecessor problem and
  related problems*, Journal of Computer and System Sciences 65 (2002).
* M. Pătrașcu, M. Thorup, *Time-space trade-offs for predecessor search*,
  STOC 2006.
* D. E. Knuth, M. F. Plass, *Breaking paragraphs into lines*, Software:
  Practice and Experience 11 (1981).
