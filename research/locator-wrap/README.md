# Line breaking with an indivisible locator

`chain.rs` renders a call-chain entry as a wrapped message with an indivisible locator suffix (`main.pn:12:3`, or the kind label) riding the **last** line only. Every other line has the full measure $A$; the last line's body has $A - s$, where $s$ is the suffix width. That asymmetry is the whole problem.

- [`spec.md`](spec.md) — the problem as posed (definitions, $\Phi$, the four questions).
- [`solution.md`](solution.md) — the full answers with proofs.
- [`scripts/verify_locator_wrap.py`](scripts/verify_locator_wrap.py) — brute-force re-derivation of every claim on small instances (`python3`, no dependencies).

## Results in one table

| Question | Answer |
|---|---|
| Feasibility | $n = 0 \Rightarrow s \le A$; $\ n \ge 1 \Rightarrow \max_t w_t \le A$ and $w_n \le A - s$ |
| $\Phi$ (greedy at $A$, then re-wrap the tail at $A - s$) | legal **iff** feasible, but line count is unboundedly bad: $A = 2M+1$, $s = 2M$, $M$ unit words $\Rightarrow M$ lines where $\mathrm{OPT} = 2$ |
| Optimal line count | $\mathrm{OPT} = L(j^* - 1) + 1$ with $j^* = \min\{\, j : B(j,n) \le A - s \,\}$ ($L$ = greedy count at $A$, monotone, exchange-optimal) |
| Complexity | $O(n)$ time (one backward width pass + one greedy prefix pass), $O(1)$ extra space |
| Paragraphs | line counts are additive: body paragraphs at $A$, the last paragraph $\mathrm{OPT} = L(j^*-1)+1$ |
| Secondary objective (squared slack) | Q2's layout is **not** optimal — freedom in both the last block's start (when $L$ is flat on $[j^*, j']$) and the prefix layout; the $(i,k)$ DP is exact, $O(n^2 K)$ naive and $O(nK)$ / $O(n)$ with a monotone CHT |
| Monge/CHT survives the perturbation | yes — the mixed body/final matrix has Monge difference $2\,\delta\,(\varepsilon - \Delta) > 0$ for $\Delta \in \{0, -s\}$, and the feasible window is a monotone staircase, so Knuth–Plass/SMAWK applies; the whole problem is $O(n)$ with $\lambda > nA^2$ per line |
| Infeasible region | $P_1 = (\tau, \ell, 0)$, $P_2 = (0, \ell+\Delta, 0)$, $P_3 = (0, \ell, \tau)$; $P_2$ dominates when $\Delta = 0$, nothing dominates otherwise, and **no hybrid** (not even a split locator) is strictly Pareto-better when $s \le A$ |

The two facts that matter for the renderer:

1. **$\Phi$ is the wrong composition.** Each phase is optimal for its own budget, and the composition still degrades linearly (Fact 2). The correct algorithm is the $j^*$ split: the last line is the *longest feasible suffix*, and the prefix is wrapped at the full measure.

2. **The reserve is Monge-compatible.** A single budget *decrease* on the final row only shifts the query point ($C_n \to C_n + s$); it does not change the convex cost, so every total-monotonicity argument for the unperturbed DP survives. Multiple different special budgets or a budget *increase* would not.

## Landed in the renderer

`src/diagnostics/wrap.rs::wrap_optimal_reserved` implements the $j^*$ split with the existing badness/fitness DP on the prefix (the $O(n)$ greedy split, the DP's layout quality). `chain.rs::render_entries` calls it once per entry, replacing the old two-phase "wrap at $A$, pop the last segment, re-wrap at $A - s$" code and its borrow-ordering dance.

The infeasible branch uses **P1** (ignore the discount: wrap at $A$, let the locator overflow) because it is the minimum-line-count policy and the one the renderer's row-count contract already assumed; P2/P3 are documented but not taken.

The proofs live next to the code: see the comment block on `wrap_optimal_reserved` for why the split point is $j^*$, why the prefix stays at the full measure, and why the infeasible fallback is P1 — plus the Fact 2 instance that the old code reproduced.

