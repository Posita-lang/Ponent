# Line breaking with an indivisible locator — problem statement

Preserved verbatim (as posed). The solution is in [`solution.md`](solution.md);
the code landing is described in [`README.md`](README.md).

---

Line breaking with an indivisible locator: feasibility, optimal line count, and
DP under an asymmetric last-line budget

*Motivation.* When rendering compiler diagnostics in a terminal, each line has a
column budget; the last line of a diagnostic entry must carry an indivisible,
immovable, non-omittable locator (e.g. `main.pn:12:3`) after the body, so the
last line's effective budget differs from that of all other lines.

Definitions. Let A ∈ ℤ_{>0} be the content measure (column budget for ordinary
lines) and s ∈ ℤ_{≥0} the locator width. A message is a word sequence
w₁,…,wₙ (n ≥ 0), each word having display width |w_t| ∈ ℤ_{>0}. For an index
interval [i,j] (1 ≤ i ≤ j ≤ n) define the span width

B(i,j) = Σ_{t=i}^{j} |w_t| + (j−i)

(exactly one space between adjacent words; B(i,i) = |w_i|).

A layout is a partition of {1,…,n} (n ≥ 1) into consecutive nonempty blocks
S₁,…,S_L, with block width B(min S, max S). A layout is legal iff
width(S_t) ≤ A for every t < L and width(S_L) + s ≤ A. An instance is feasible
iff a legal layout exists. For n = 0 the unique layout is a single empty line,
feasible iff s ≤ A.

Greedy wrapping Greedy(m, B₀): scan the first m words left to right, taking each
block maximal under budget B₀; a single word wider than B₀ occupies a block of
its own (budget-violating). Let L(m) be the number of greedy blocks. Known
(folklore; cf. the notes to Knuth–Plass 1981, or a direct exchange argument): if
all word widths are ≤ B₀, greedy minimises the number of blocks.

Two-phase algorithm Φ. (i) Greedily wrap the whole text at budget A, obtaining
blocks G₁,…,G_k. (ii) If B(G_k) + s > A, drop G_k and greedily re-wrap its words
at budget A−s, splicing the resulting blocks after G₁,…,G_{k−1}; otherwise
output unchanged.

Facts (to be made rigorous in Q1).

*Fact 1.* The instance is feasible ⟺ (n = 0 and s ≤ A) or (n ≥ 1,
max_t |w_t| ≤ A, and |w_n| ≤ A − s). Moreover, Φ's output is legal ⟺ the
instance is feasible. (⇒ uses "the last block of any legal layout is a suffix of
the word sequence"; ⇐ uses greedy's property "the last block overruns its budget
iff it is a single over-wide word".)

*Fact 2 (composition degrades without bound).* Take A = 2M+1, s = 2M, and M
words of width 1. Φ outputs M lines; the optimum is 2 lines (the first M−1 words
on one line, width 2M−3 ≤ A; the last word plus locator on one line,
width 1+2M = A). Each phase is optimal for its own budget, yet the composition's
line-count ratio is M/2 → ∞.

Questions.

1. Prove both directions of Fact 1 and the legality equivalence for Φ (first
   prove greedy's "last block overruns ⟺ single over-wide word" property).
   Generalise to paragraph structure: the message is a sequence of paragraphs
   p₁,…,p_m separated by hard newlines; a layout is the product of the
   per-paragraph layouts, with the budget rule "budget A everywhere except the
   last line of the last paragraph". Prove the feasibility criterion and the
   additivity of line counts over paragraphs.

2. Optimal line count. Define j* = min{j : B(j,n) ≤ A−s} (the start of the
   longest feasible suffix; undefined when infeasible). Claim:
   OPT = L(j*−1) + 1. Proof obligation: every legal layout's last block is a
   suffix w_j…wₙ with B(j,n) ≤ A−s, hence j ≥ j*; its prefix requires ≥ L(j−1)
   blocks; L is nondecreasing in the number of words, so j = j* is optimal.
   Give an O(n) implementation (one backward pass accumulating span widths to
   find j*, one greedy pass over the prefix) and prove the monotonicity of L.

3. Secondary objective. Among legal layouts with the minimum number of lines,
   minimise Σ_t (β(S_t) − width(S_t))², where β(S_t) = A for t < L and
   β(S_L) = A − s.

   (a) Construct an instance proving that "greedy prefix + longest feasible
       suffix" (Q2's solution) is strictly suboptimal for the secondary
       objective — minimum-line layouts have freedom from two sources: the
       choice of the last block's start while L is flat on [j*, j'], and
       non-greedy L-block layouts of the prefix.

   (b) Give a DP with state (i,k) (opt₂(i,k) = minimum penalty for the first i
       words using exactly k budget-A blocks); the answer is the minimum over
       feasible last blocks on the level k + 1 = OPT. Prove correctness and give
       the complexity.

   (c) The Knuth–Plass O(n) / O(n log n) speedup relies on
       monotonicity/concavity conditions on the cost (totally monotone matrices,
       SMAWK-type). The last-line budget A−s ≠ A destroys block-uniformity of
       the cost. Characterise which conditions survive the perturbation "uniform
       body budget + a single asymmetric final transition"; does the resulting
       one-dimensional min_j still lie in the Monge/concave framework? Decide
       from this whether the speedup can be retained.

4. Policies in the infeasible region. When |w_n| > A−s (or n = 0 and s > A) no
   legal layout exists, and any output violates the last-line constraint.
   Formalise the three candidate policies as optimisation with violations and
   solve each for its optimal layout:

   P1 *ignore the discount* — wrap everything at A, the locator still follows
   the last line; violation = width(last block) + s − A;

   P2 *locator on its own line* — body wrapped at A, final line contains only
   the locator (requires s ≤ A); line count = L(n) + 1;

   P3 *truncate the locator* — show only what fits on the last line; loss =
   number of truncated characters, violation = 0.

   With loss vector (violated columns, line count, truncated characters):

   • find the layout minimising violation under P1 (what must the last block be?
     verify whether it is {w_n});

   • characterise the mutual dominance of P1, P2, P3 across parameter regions
     under the Pareto order;

   • is there a hybrid policy (e.g. partial truncation + own line) strictly
     Pareto-better than all three? Prove or give a counterexample.
