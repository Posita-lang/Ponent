# Optimal line wrapping: why `wrap_dp` stays

## Problem

Wrap one paragraph of words to a display measure. A line covering words
`i..j` has width `base_j - pws[i]`, intrinsic badness `g(width)` (convex,
cubic in the empty/measure ratio), and a fitness class
`c(width) ∈ {0,1,2,3}`. Consecutive lines pay `JUMP[c][pc]`; the first jump is
free, each further class step costs `1/16`. Minimise

```
sum over lines of intrinsic_badness(line) + sum over adjacent lines of JUMP.
```

The DP state is `(j, c)`: the minimum cost of laying out `words[0..j)` with
its last line in class `c`. The transition is

```
dp[j][c] = min_i min_pc ( dp[i][pc] + JUMP[c][pc] + g(base_j - pws[i]) ).
```

## Current algorithm

`wrap_dp` keeps `4(n+1)` states, a monotone window start `i_lo`, and folds the
`pc` scan into a per-`i`, per-class min-plus table. Time is `O(n·k + 16n)`
where `k` is the number of words that fit one line; space is `O(n)`. Widths
and classes are tabulated per measure, so the inner loop is two indexed loads
plus one add and compare. Diagnostics wrap `n` in the 5–130 word range and
`k` in the 5–20 range.

## Can a Monge / Larmore–Schieber DP replace it?

For a fixed class `c`, the transition matrix is

```
M_c[i][j] = best_cost[i][c] + g(base_j - pws[i]).
```

`g` is convex and `base_j - pws[i]` is increasing in `j`, decreasing in `i`;
per-row offsets preserve Monge, so `M_c` is Monge. A per-class online
Larmore–Schieber recurrence is therefore valid in principle.

It does **not** combine into one block-matrix instance. The full block matrix
has row `(i, pc)`, column `(j, c)`, and cost
`JUMP[c][pc] + g(base_j - pws[i])`. `JUMP` is `h(|c - pc|)` with
`h(0)=h(1)=0`, `h(2)=1/16`, `h(3)=2/16`; it is **not additively separable**
(`h(2)+h(0) ≠ h(1)+h(1)`), so the lexicographic block matrix is not Monge.
A unified deque would be unsound; only four independent per-class instances
are available, plus the same `O(16n)` min-plus pass.

The remaining blockers are constant factors and the workload shape:

- Larmore–Schieber costs roughly 30–50 operations per comparison; the current
  monotone-window scan costs roughly 10–15.
- On a non-last line the optimum is the most-filled legal line, so the
  monotone argmin pointer sits at `i_lo` and prunes essentially nothing.
- The current DP is already `O(n·k)` with `k = 5–20`; four per-class deques
  multiply the bookkeeping by four while removing a factor that is already
  small.
- The crossover indicated by these constants is a long unclosed window
  (`n ≳ 10²–10³`, `k ≳ 15`), not the diagnostic corpus (`n ≤ ~130`).

Decision: keep the monotone-window DP. Revisit only with a measured workload
whose window is unclosed and whose `n`, `k` cross the constants above; then
prototype four independent per-class Larmore–Schieber instances, not a block
deque.

## Is an LP useful here?

This DP is a shortest path in a DAG. Its natural LP is integral and does not
remove states, transitions, or reconstruction work; it is not a smaller
algorithm. There is no fractional relaxation to round.

The interval-colouring LP studied elsewhere is a different problem and is
**not** value-exact; it cannot be transplanted into wrapping. If an LP is ever
proposed for wrapping, the acceptance contract is: LP optimum, explicit
integrality check, fall back to `wrap_dp` on any fractional solution, and
byte-identical output against the DP on the snapshot corpus.

`scripts/crossover.py` prints the operation-count model behind the decision.
It is a model, not a benchmark.
