# Line breaking with an indivisible locator — solution

Answers to [`spec.md`](spec.md). All statements below are machine-checked where that is feasible; [`scripts/verify_locator_wrap.py`](scripts/verify_locator_wrap.py) re-derives the feasibility criterion, the $\Phi$ blow-up, the $\mathrm{OPT} = L(j^*-1)+1$ formula, both secondary-objective witnesses, the Monge sign of the mixed matrix, and the Pareto/no-go claims by brute force.

Throughout, indices are 1-based for words, $w_t$ is the word and $|w_t|$ its width. For a prefix of length $i$,

$$W_i = \sum_{t=1}^{i} |w_t|, \qquad C_i = W_i + i, \qquad C_0 = 0,$$

so that for $0 \le p < i$

$$B(p+1,\, i) \;=\; C_i - C_p - 1.$$

$L_B(m)$ is the number of blocks produced by greedy wrapping of the first $m$ words at budget $B$; $L(m) := L_A(m)$.

---

## 1. Fact 1, the two-phase algorithm $\Phi$, and paragraphs

### 1.1 Greedy's overrun lemma

**Lemma 1 (structure of greedy blocks).** In greedy wrapping at budget $B_0$, a block has width $> B_0$ iff it is a singleton word whose width is $> B_0$. In particular the last block overruns $B_0$ iff it is a single over-wide word.

*Proof.* Consider the moment greedy starts a block at word $i$.

- If $|w_i| > B_0$, the algorithm places $w_i$ alone in a block. Its width is $|w_i| > B_0$; no further word can be added.
- If $|w_i| \le B_0$, greedy extends the block to the largest $j$ with $B(i,j) \le B_0$. Every added word keeps the span $\le B_0$, so the finished block has width $\le B_0$; if $j < m$ the next word starts a new block.

Hence an overrunning block can only arise in the first case, and then it is exactly a singleton over-wide word. The last block is a special case. $\blacksquare$

Corollary: if all word widths are $\le B_0$, every greedy block has width $\le B_0$.

### 1.2 Feasibility criterion (Fact 1, first half)

**Claim.** The instance is feasible iff either

- $n = 0$ and $s \le A$; or
- $n \ge 1$, $\max_t |w_t| \le A$, and $|w_n| \le A - s$.

*Necessity.* Let a legal layout have blocks $S_1,\dots,S_L$.

- Every block is an interval of consecutive words, so its width is at least the width of each word in it. For $t < L$, $\mathrm{width}(S_t) \le A$; for the last block $\mathrm{width}(S_L) + s \le A$, hence $\mathrm{width}(S_L) \le A - s \le A$. Therefore every word has width $\le A$.
- The last block is a suffix $w_j,\dots,w_n$, so its width is at least $|w_n|$. Hence $|w_n| + s \le \mathrm{width}(S_L) + s \le A$, i.e. $|w_n| \le A - s$.
- If $n = 0$ the unique empty line has width $0$, so legality is exactly $0 + s \le A$.

*Sufficiency.*

- $n = 0,\ s \le A$: the empty line is legal.
- $n \ge 1$: wrap the first $n-1$ words greedily at budget $A$; by the corollary every such block has width $\le A$. Put $w_n$ alone on the last line. Its width is $|w_n| \le A - s$, so $|w_n| + s \le A$. This is a legal layout. $\blacksquare$

### 1.3 Legality equivalence for $\Phi$

Define $\Phi(0)$ to output the unique empty line. For $n \ge 1$, $\Phi$ first greedily wraps all words at $A$, obtaining $G_1,\dots,G_k$. If $B(G_k) + s \le A$ it stops; otherwise it drops $G_k$ and re-wraps the words of $G_k$ at budget $A - s$, splicing the result after $G_1,\dots,G_{k-1}$.

**Claim.** The output of $\Phi$ is legal iff the instance is feasible.

*Proof.* "$\Phi$ legal $\Rightarrow$ feasible" is tautological (a legal layout exists), so by §1.2 the criterion holds. Conversely assume the criterion: $\max_t |w_t| \le A$ and $|w_n| \le A - s$.

- All greedy blocks $G_1,\dots,G_k$ at budget $A$ have width $\le A$.
- If $B(G_k) + s \le A$ the output is unchanged and legal.
- Otherwise $B(G_k) > A - s$. Re-wrap the suffix $H = G_k$ at budget $A - s$. By Lemma 1 any block of this re-wrap that exceeds $A - s$ is a singleton word of width $> A - s$. Such a word has width $\le A$ (global max), so it is legal as an ordinary line. The final block of the re-wrap ends at $w_n$; if it overran $A - s$, Lemma 1 would force $w_n$ itself to be a singleton of width $> A - s$, contradicting $|w_n| \le A - s$. Hence the final block has width $\le A - s$, so the final line plus the locator fits. All earlier re-wrapped blocks have width $\le \max(A - s,\, A) = A$. Therefore the whole output is legal.

So $\Phi$ is legal exactly on feasible instances. $\blacksquare$

### 1.4 Paragraph structure

Let the message be paragraphs $p_1,\dots,p_m$ separated by hard newlines. Paragraph $r$ has $n_r \ge 0$ words $w^{(r)}_1,\dots,w^{(r)}_{n_r}$; $n_r = 0$ means the paragraph contributes one empty line. A layout is the product of the per-paragraph layouts; hard newlines force a break between $p_r$ and $p_{r+1}$.

**Budget rule.** Every line has budget $A$, except the last line of $p_m$, which carries the locator and has body budget $A - s$. Let $M = \max$ over all words in all paragraphs.

**Feasibility criterion.**

$$\text{feasible} \;\iff\; M \le A \;\text{ and }\; \bigl( (n_m \ge 1 \text{ and } |w^{(m)}_{n_m}| \le A - s) \;\text{ or }\; (n_m = 0 \text{ and } s \le A) \bigr).$$

*Necessity:* lines in $p_1,\dots,p_{m-1}$ and all non-final lines of $p_m$ have budget $A$, so every word is $\le A$; the final line of $p_m$ contains its last word, forcing $|w^{(m)}_{n_m}| \le A - s$ (or $s \le A$ if that paragraph is empty).

*Sufficiency:* wrap each $p_r$, $r < m$, greedily at $A$ (empty paragraph $\to$ one empty line). For $p_m$ apply the single-paragraph construction of §1.2: wrap the first $n_m - 1$ words greedily at $A$, then put the last word alone on the final line; if $n_m = 0$, the empty line carries the locator. Hard newlines do not couple the paragraphs, so this is legal.

**Paragraph version of $\Phi$.** Wrap $p_1,\dots,p_{m-1}$ greedily at $A$; apply the single-paragraph $\Phi$ to $p_m$. By §1.3 the whole output is legal iff the paragraph criterion above holds.

**Additivity of line counts.** For any layout the total number of lines is the sum of the lines used per paragraph, because hard newlines make the product decomposition independent. Define

$$\lambda_r \;=\; \begin{cases} 1 & \text{if } n_r = 0, \\ L_A(n_r) & \text{if } n_r \ge 1,\ r < m, \end{cases}$$

and, for the last paragraph, with $j^*_m = \min\{\, j : B_m(j, n_m) \le A - s \,\}$,

$$\lambda_m \;=\; \begin{cases} 1 & \text{if } n_m = 0,\ s \le A, \\ L_A(j^*_m - 1) + 1 & \text{if } n_m \ge 1, \end{cases}$$

(the single-paragraph optimum of §2). Then

$$\mathrm{OPT}_{\text{paras}} \;=\; \sum_{r=1}^{m} \lambda_r.$$

Every paragraph layout contributes at least $\lambda_r$ lines, and concatenating the optimal per-paragraph layouts attains the sum while respecting the budget rule: all lines before $p_m$'s last line have budget $A$, and only $p_m$'s last line has budget $A - s$.

---

## 2. Optimal line count and the $O(n)$ algorithm

### 2.1 A subtlety about $j^*$

As written, $j^* = \min\{\, j : B(j,n) \le A - s \,\}$ is nonempty iff $|w_n| \le A - s$, because $B(n,n) = |w_n|$. But full feasibility also requires $\max_t |w_t| \le A$. If some earlier word is wider than $A$, the set can be nonempty while no legal layout exists. Therefore: $j^*$ and $\mathrm{OPT} = L(j^*-1) + 1$ are used only for **feasible** instances; if the instance is infeasible, $\mathrm{OPT} = \infty$ and $j^*$ is undefined by convention.

Under feasibility $\max_t |w_t| \le A$ and $|w_n| \le A - s$, so $j^*$ exists and the whole analysis is valid.

For $n = 0$, feasible means $s \le A$ and $\mathrm{OPT} = 1$; there is no $j^*$.

### 2.2 Greedy optimality and monotonicity of $L$

**Lemma 2 (greedy minimises block count).** If all word widths are $\le A$, then $L(m)$ is the minimum possible number of blocks of width $\le A$ covering the first $m$ words.

*Proof (exchange).* Let greedy's first block be $[1,g]$, maximal with $B(1,g) \le A$. Take any legal layout of the $m$ words into $\ell$ blocks and let its first block be $[1,r]$. Maximality of $g$ gives $r \le g$. Move the words $r+1,\dots,g$ from the front of later blocks into the first block; removing a prefix from a block cannot increase its width, and empty blocks are deleted, so the number of blocks does not increase. We now have a layout whose first block is $[1,g]$. The remaining words $g+1,\dots,m$ are exactly the input on which greedy continues, so by induction they need at least $L(m) - 1$ blocks. Hence $\ell \ge L(m)$. $\blacksquare$

**Monotonicity.** $L(m)$ is nondecreasing in $m$, and in fact $L(m+1) \in \{L(m),\, L(m)+1\}$. Let the last greedy block of the first $m$ words start at $q$. When $w_{m+1}$ is appended either $B(q, m+1) \le A$, in which case that last block is extended and the count stays $L(m)$, or a new block starts and the count becomes $L(m)+1$. Further words can only keep or increase the count. Hence $L(j-1) \ge L(j^*-1)$ for $j \ge j^*$.

### 2.3 Proof of $\mathrm{OPT} = L(j^*-1) + 1$

Assume feasibility. Note

$$B(j+1,\, n) \;=\; B(j,\, n) - |w_j| - 1 \;\le\; B(j,\, n) - 2,$$

so $B(j,n)$ is strictly decreasing as $j$ increases. Hence the set of starts $j$ whose suffix fits the last line is exactly $\{\, j^*,\, j^*+1,\, \dots,\, n \,\}$. Let any legal layout have last block $w_j,\dots,w_n$. Then $B(j,n) \le A - s$, so $j \ge j^*$. Its prefix $w_1,\dots,w_{j-1}$ is covered by $L - 1$ ordinary budget-$A$ blocks, so by Lemma 2 $L - 1 \ge L(j-1) \ge L(j^*-1)$ using monotonicity. Therefore $L \ge L(j^*-1) + 1$.

Conversely take $j = j^*$. Wrap $w_1,\dots,w_{j^*-1}$ greedily at $A$ (legal because $\max \le A$), giving $L(j^*-1)$ lines, and put $w_{j^*} \dots w_n$ on the last line; its width is $B(j^*,\, n) \le A - s$. This layout is legal with $L(j^*-1) + 1$ lines. Hence $\mathrm{OPT} = L(j^*-1) + 1$. $\blacksquare$

### 2.4 $O(n)$ implementation

One backward pass finds $j^*$ and checks feasibility; one greedy pass over the prefix computes $L(j^*-1)$.

```python
def wrap_locator(w, A, s):
    n = len(w)
    if n == 0:
        return 1 if s <= A else INFEASIBLE
    cur = 0                 # becomes B(j, n) while scanning j downwards
    jstar = None
    maxw = 0
    for j in range(n, 0, -1):
        maxw = max(maxw, w[j - 1])
        cur = w[j - 1] + (1 if j < n else 0) + cur   # = B(j, n)
        if cur <= A - s:
            jstar = j        # last assignment = smallest feasible j
    if maxw > A or jstar is None:
        return INFEASIBLE    # jstar is None iff w_n > A - s
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

Correctness of the backward accumulation: at $j = n$, $\mathrm{cur} = |w_n|$; for $j < n$, $\mathrm{cur} \leftarrow |w_j| + 1 + \mathrm{cur} = B(j,n)$. Since $\mathrm{cur}$ increases as $j$ decreases, the last $j$ with $\mathrm{cur} \le A - s$ is exactly $j^*$. Checking $\mathrm{maxw}$ handles the $\max_t |w_t| \le A$ condition. The greedy pass is $O(m) \le O(n)$. Total time $O(n)$, extra space $O(1)$ beyond the input.

---

## 3. Secondary objective

For a layout with $L$ lines the slack of line $t$ is $\beta_t - \mathrm{width}(S_t)$, where $\beta_t = A$ for $t < L$ and $\beta_L = A - s$. The penalty is the sum of squared slacks.

**Invariant.** For $n \ge 1$ and fixed $L$ the total slack is constant:

$$\sum_{t=1}^{L} \beta_t \;=\; L \cdot A - s, \qquad \sum_{t=1}^{L} \mathrm{width}(S_t) \;=\; \sum_{t=1}^{n} |w_t| + (n - L),$$

so

$$\sum_{t=1}^{L} \bigl(\beta_t - \mathrm{width}(S_t)\bigr) \;=\; L(A + 1) \;-\; \sum_{t=1}^{n} |w_t| \;-\; n \;-\; s.$$

Thus minimising the sum of squares means distributing a fixed total slack as evenly as possible. Greedy plus the longest feasible suffix typically produces very uneven slacks.

### 3(a) Q2's solution is not secondary-optimal — two independent witnesses

**Witness 1: freedom in the last block's start ($L$ flat on $[j^*, j']$).**

$$n = 3, \quad w = (1,1,1), \quad A = 4, \quad s = 1, \quad A - s = 3.$$

- $B(1,3) = 5 > 3$, $\ B(2,3) = 3 \le 3$, $\ B(3,3) = 1 \le 3$, so $j^* = 2$; $L(1) = 1$, $\mathrm{OPT} = 2$.
- $L(2) = 1$ as well ($B(1,2) = 3 \le 4$), so $L$ is flat on $[2,3]$.

Q2's layout ($j^* = 2$): line 1 $[w_1]$ width 1 slack 3; last line $[w_2, w_3]$ width 3 slack 0. Penalty $3^2 + 0^2 = 9$.

Alternative ($j = 3$): line 1 $[w_1, w_2]$ width 3 slack 1; last line $[w_3]$ width 1 slack 2. Penalty $1^2 + 2^2 = 5 < 9$.

Both use 2 lines. $\blacksquare$

**Witness 2: freedom in a non-greedy prefix ($j^*$ forced).**

$$n = 4, \quad w = (2,1,1,1), \quad A = 4, \quad s = 2, \quad A - s = 2.$$

- $B(4,4) = 1 \le 2$ but $B(3,4) = 3 > 2$, so $j^* = 4$ is forced.
- Prefix $(2,1,1)$: greedy at $A = 4$ gives $[w_1, w_2]$ (width 4) and $[w_3]$ (width 1), so $L(3) = 2$, $\mathrm{OPT} = 3$.

Q2's layout: $[w_1, w_2]$ slack 0; $[w_3]$ slack 3; $[w_4]$ slack 1. Penalty $0 + 9 + 1 = 10$.

Another 3-line layout: $[w_1]$ (width 2, slack 2), $[w_2, w_3]$ (width 3, slack 1), then $[w_4]$ (slack 1). Penalty $4 + 1 + 1 = 6 < 10$.

Both use exactly $\mathrm{OPT} = 3$ lines, and here the last block's start is forced. $\blacksquare$

So both sources of freedom — a flat $L$ interval for the last start, and non-greedy $L$-block prefix layouts — can independently make Q2's solution strictly suboptimal.

### 3(b) DP over $(i,k)$

Let $K = \mathrm{OPT} - 1$ be the required number of ordinary budget-$A$ lines. Define

$$\mathrm{opt}_2(i,k) \;=\; \text{minimum squared-slack penalty for the first } i \text{ words using exactly } k \text{ blocks of budget } A.$$

Base cases $\mathrm{opt}_2(0,0) = 0$, $\ \mathrm{opt}_2(i,0) = \infty$ for $i > 0$, $\ \mathrm{opt}_2(0,k) = \infty$ for $k > 0$. For $k \ge 1$

$$\mathrm{opt}_2(i,k) \;=\; \min_{0 \le p < i,\; B(p+1,i) \le A} \Bigl[\; \mathrm{opt}_2(p,\, k-1) + \bigl(A - B(p+1,i)\bigr)^2 \;\Bigr].$$

The last body block is $w_{p+1},\dots,w_i$; its penalty is $(A - B)^2$.

**Correctness.** By induction on $k$. Any layout of the first $i$ words into $k$ body blocks has a last body block $[p+1,i]$ with $B(p+1,i) \le A$; the first $p$ words use $k-1$ blocks and cost at least $\mathrm{opt}_2(p,k-1)$. Conversely, appending $[p+1,i]$ to an optimal $(k-1)$-block layout of the first $p$ words attains the displayed value. Taking the minimum gives the optimum.

Finally append the special last line. For a prefix length $i$ ($0 \le i \le n-1$) the last block is $w_{i+1},\dots,w_n$, legal iff $B(i+1,n) \le A - s$, with penalty $(A - s - B(i+1,n))^2$. Hence

$$\mathrm{ANS} \;=\; \min_{0 \le i \le n-1,\; B(i+1,n) \le A-s} \Bigl[\; \mathrm{opt}_2(i,\, K) + \bigl(A - s - B(i+1,n)\bigr)^2 \;\Bigr].$$

This uses exactly $K + 1 = \mathrm{OPT}$ lines and is legal.

**Complexity.** Naively $i = 1..n$, $k = 1..K$, $p = 0..i-1$: $O(n^2 K)$ time, $O(nK)$ space. With $C_i = W_i + i$ and $D_p = C_p + A + 1$ we have $B(p+1,i) = C_i - C_p - 1$ and body penalty $(D_p - C_i)^2$. For fixed $k-1$ the recurrence is a sliding-window minimum of lines

$$f(p) + (D_p - C_i)^2 \;=\; C_i^2 \;-\; 2\,D_p\,C_i \;+\; \bigl(D_p^2 + f(p)\bigr),$$

with $D_p$ increasing and queries $C_i$ increasing, so a monotone convex-hull trick (deque) evaluates all $i$ in $O(n)$ per level: $O(nK)$ time and $O(n)$ space with rolling arrays. Since $K \le n$, this is $O(n^2)$ in the worst case.

### 3(c) Monge/concavity and the Knuth–Plass speedup

Write the cost of a line from $p+1$ to $i$ with budget $\beta$ as

$$c_\beta(p,i) \;=\; \bigl(\beta + 1 + C_p - C_i\bigr)^2 \;=\; \bigl(D_p^{\beta} - C_i\bigr)^2, \qquad D_p^{\beta} = C_p + \beta + 1,$$

valid when $D_p^{\beta} \ge C_i$. For body rows $\beta = A$; for the final row $\beta_f = A - s$. Note $D_p^{A-s} = D_p^{A} - s$, so the final cost is the body cost evaluated at a shifted coordinate:

$$c_{A-s}(p,n) \;=\; \bigl(D_p^{A} - (C_n + s)\bigr)^2.$$

**The Monge inequality survives.** Let rows $i < i'$ have budgets $\beta_i, \beta_{i'}$ and columns $p < p'$. Put

$$\delta = C_{p'} - C_p > 0, \qquad \varepsilon = C_{i'} - C_i > 0, \qquad \Delta = \beta_{i'} - \beta_i, \qquad u = D_p^{\beta_i} - C_i.$$

Then

$$c_{\beta_i}(p,i) = u^2, \quad c_{\beta_i}(p',i) = (u + \delta)^2, \quad c_{\beta_{i'}}(p,i') = (u + \Delta - \varepsilon)^2, \quad c_{\beta_{i'}}(p',i') = (u + \delta + \Delta - \varepsilon)^2.$$

The Monge difference ($\mathrm{RHS} - \mathrm{LHS}$) is

$$\bigl[\, c(p',i) + c(p,i') \,\bigr] \;-\; \bigl[\, c(p,i) + c(p',i') \,\bigr] \;=\; 2\,\delta\,(\varepsilon - \Delta).$$

- Body–body: $\Delta = 0 \Rightarrow 2\delta\varepsilon > 0$.
- Body–final: $\Delta = -s \Rightarrow 2\delta(\varepsilon + s) > 0$.

Hence for all valid $p < p'$, $i < i'$

$$M[i,p] + M[i',p'] \;\le\; M[i,p'] + M[i',p],$$

i.e. the mixed matrix is Monge / totally monotone, even after adding column constants $f(p)$ and a per-line constant $\lambda$ (both cancel in the quadrangle difference).

**Feasibility restrictions** ($B \le \beta$, i.e. $C_p \ge C_i - \beta - 1$) give column intervals $[\ell_i, u_i]$ with $\ell_i$ and $u_i$ nondecreasing in $i$; for the final row $C_n + s > C_{n-1}$, so its left endpoint is also $\ge$ that of row $n-1$. This staircase preserves total monotonicity with $\infty$ entries. Therefore row minima are attained at nondecreasing column indices — exactly the SMAWK / Knuth–Plass condition.

**Interpretation.** In the slack variable the cost $\varphi(u) = u^2$ is convex, and that convexity is what yields the Monge inequality. (In the usual Knuth–Plass badness parameterisation the same hypothesis is often phrased as concavity; the invariant is the quadrangle/Monge inequality above.) The last-line perturbation introduces no new cost function — it only translates the final query point from $C_n$ to $C_n + s$. One budget *decrease* is Monge-compatible; a budget increase larger than $C_{i'} - C_i$, or several different special budgets, would destroy it.

**Can the speedup be retained? Yes.**

1. *Per DP level.* The recurrence of §3(b) is a Monge row-minima problem; online SMAWK / the convex-hull trick computes each level in $O(n)$ ($O(n \log n)$ with a generic CHT). The layered DP then costs $O(n \cdot \mathrm{OPT})$ instead of $O(n^2 \cdot \mathrm{OPT})$.

2. *Whole problem in $O(n)$ / $O(n \log n)$.* The $\mathrm{OPT}$ factor comes only from the hard cardinality constraint. Add a large constant $\lambda > n \cdot A^2$ per line and solve the unconstrained shortest-path DP

$$f(0) = 0, \qquad f(i) \;=\; \min_{p<i} \Bigl[\; f(p) + \lambda + c_{\beta_i}(p,i) \;\Bigr],$$

with body rows $i < n$ ($\beta = A$) and final row $i = n$ ($\beta = A - s$). Because $\lambda$ exceeds the maximum possible total penalty difference, the optimum first minimises the number of lines, then the secondary penalty among those. The matrix is the same mixed Monge matrix and the final row is the shifted query $x = C_n + s$. An online monotone CHT (deque) inserts one line per $p$, removes infeasible prefixes, and answers the strictly increasing queries $C_1 < C_2 < \dots < C_{n-1} < C_n + s$ in $O(n)$ total time and $O(n)$ space. ($n = 0$ is the trivial empty line.)

So the asymmetric last line does not destroy the Knuth–Plass speedup: the one-dimensional $\min_j$ is precisely the last row of a totally monotone matrix.

---

## 4. Policies in the infeasible region

**Standing assumption.** To keep the policies meaningful assume $\max_t |w_t| \le A$ (otherwise the body already has unavoidable over-wide blocks and the loss vector needs extra coordinates).

The infeasible region is

$$|w_n| > A - s \;\;(n \ge 1), \qquad \text{or } n = 0 \text{ with } s > A.$$

Set

$$\tau = |w_n| + s - A > 0, \qquad \ell = L(n-1) + 1, \qquad \Delta = L(n) - L(n-1) \in \{0,1\}.$$

$\Delta = 0$ iff word $n$ fits into the last greedy block of the first $n-1$ words; equivalently, if that block starts at $a$, $B(a, n-1) + 1 + |w_n| \le A$. Since $|w_n| \ge A - s + 1$, $\Delta = 0$ forces $B(a,n-1) \le s - 2$ (so it needs $s \ge 2$). Also $L(n) + 1 = \ell + \Delta$.

### 4.1 Optimal layout of each policy

Let a general body layout have last block start $j$; its width is $W_j = B(j,n)$. Legality of body lines at $A$ requires $W_j \le A$; since $|w_n| \le A$, $j = n$ is always allowed, with $W_n = |w_n|$. Because $W_j$ is strictly increasing as $j$ decreases, $W_j \ge |w_n|$, with equality iff $j = n$.

**P1 — ignore the discount.** Violation is $W_j + s - A$; minimising it means minimising $W_j$, so the unique optimum is

$$S_L = \{\, w_n \,\} \qquad (\text{for } n = 0 \text{: the empty line}).$$

The prefix $w_1,\dots,w_{n-1}$ is wrapped greedily at $A$, giving $L(n-1)$ lines. Canonical loss vector:

$$P_1 = (\tau,\ \ell,\ 0).$$

**P2 — locator on its own line.** Requires $s \le A$. The body is wrapped greedily at $A$ in $L(n)$ lines; one extra line holds the full locator. Canonical loss vector:

$$P_2 = (0,\ \ell + \Delta,\ 0).$$

If $s > A$, P2 is not an admissible policy.

**P3 — truncate the locator.** With last block width $W$, the last line has $A - W$ free columns; show $\min(s,\, A - W)$ characters, truncating $s - (A - W) = W + s - A$ characters. Again minimising truncation forces $W = |w_n|$, i.e. $\{w_n\}$ last, prefix greedy. Canonical loss vector:

$$P_3 = (0,\ \ell,\ \tau).$$

For $n = 0$, $s > A$: $\tau_0 = s - A$, $\ \ell_0 = 1$, $\ P_1 = (\tau_0, 1, 0)$, $\ P_3 = (0, 1, \tau_0)$, P2 unavailable.

### 4.2 Mutual Pareto dominance

Lower is better; $x$ dominates $y$ iff $x_i \le y_i$ for all $i$ and $<$ for at least one.

| region | P1 | P2 | P3 | dominance |
|---|---|---|---|---|
| $s \le A$, $\ \Delta = 0$ | $(\tau,\ell,0)$ | $(0,\ell,0)$ | $(0,\ell,\tau)$ | P2 strictly dominates P1 and P3; P1 and P3 incomparable |
| $s \le A$, $\ \Delta = 1$ | $(\tau,\ell,0)$ | $(0,\ell+1,0)$ | $(0,\ell,\tau)$ | all three pairwise incomparable |
| $s > A$ | $(\tau,\ell,0)$ | not admissible | $(0,\ell,\tau)$ | P1 and P3 incomparable; $n = 0$ is the special case above |

When $\Delta = 0$, P2 has the same line count as P1/P3 but zero violation and zero truncation. When $\Delta = 1$, P2 buys zero violation/truncation with one extra line, P1 buys fewer lines with $\tau$ violated columns, and P3 buys fewer lines with $\tau$ truncated characters; no vector is componentwise $\le$ another with a strict improvement.

### 4.3 Is there a strictly Pareto-better hybrid?

When all three policies are defined ($s \le A$): **no**.

Allow the most general hybrid: arbitrary body wrapping, arbitrary splitting of the locator across the last body line and/or extra own lines, overflow, and truncation. Let its loss be $(V, L, T)$. To dominate P1 we need $T \le 0$, hence $T = 0$ (full locator shown). To dominate P3 we need $V \le 0$, hence $V = 0$ (no overflow). To dominate P2 we need $V = 0$, $T = 0$, and a strict improvement somewhere; since $V = T = 0$ already match P2, the strict component must be $L < L(n) + 1 = \ell + \Delta$. But any output with $V = 0$ and $T = 0$ has:

- at least $L(n)$ lines containing body words, because with zero overflow every body block has width $\le A$ and greedy minimises their number;
- at least one extra locator-only line, because the last body block has width $W \ge |w_n| > A - s$, so $W + s > A$ and the full locator cannot fit on the last body line without overflow. Even if we place $A - W < s$ characters there, $s - (A - W) > 0$ characters remain and need another line.

Therefore $L \ge L(n) + 1 = \ell + \Delta$, contradicting the required strict inequality. Hence no hybrid — not even one that splits the locator — can dominate P2, and consequently none can dominate all three. In particular "partial truncation" cannot work, because $T > 0$ already fails to dominate P1.

**Remark on $s > A$.** Literally, P2 is not a policy there, so "better than all three" is undefined. Among the two admissible policies P1 and P3 a split-locator hybrid can be better:

$$w = (1,1), \quad A = 4, \quad s = 5 \;(> A), \quad n = 2.$$

Here $L(2) = 1$, $L(1) = 1$, $\ell = 2$, $\tau = 1 + 5 - 4 = 2$.

- P1: $(2, 2, 0)$; P3: $(0, 2, 2)$.
- Hybrid: both words on one line (width 3), append 1 locator character to that line (width 4, no overflow), and put the remaining 4 characters on a second line (width 4). All 5 characters are shown, with no truncation, no overflow, and 2 lines: $(0, 2, 0)$.

This dominates P1 and P3. If one artificially extends P2 to $s > A$ by letting its own line overflow by $s - A = 1$, its loss would be $(1, 2, 0)$, which $(0, 2, 0)$ also dominates.

(The example splits the locator across a line break; with the literal requirement $s \le A$ for P2, the no-go theorem above is the complete answer for the region where all three policies exist.)
