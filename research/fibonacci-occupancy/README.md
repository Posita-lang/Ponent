# Fibonacci-hashing bucket occupancy research

This directory answers a query-engine question, not an underline-lane one: the
`DepGraph` hashes a node index with the Knuth golden-ratio multiplier
`0x9E37_79B9 = floor(2^32/φ)` and takes the top six bits, so how many of the 64
`LINEAR_DEDUP_MAX`/cache slots are occupied by the first `n` indices?

Formally, with `Pₙ(α) = {{iα} : 1 ≤ i ≤ n}` and `Occ(n, m, α)` the number of
non-empty of `m` equal buckets, the exact values at `(64, 64)` are 55 for both
`1/φ` and `0x9E37_79B9/2^32`, but 37 for the near-by rotation
`165902777/2^32`.  `Occ(64,64,1/φ) = 55 = q₉` is an alignment coincidence, not
a theorem; the true statement is that `Occ(q_k, q_k, α) = q_k` for every
irrational `α` at its convergents.

## Map

| Material | Location |
|---|---|
| Complete derivations (exact ℚ(√5) arithmetic, three-gap, Farey walls, corrected Theorem 2.1) | [`derivations.md`](derivations.md) |
| Exact occupancy arithmetic; `--measure` runs the full Farey scan | [`scripts/golden_ratio_occupancy.py`](scripts/golden_ratio_occupancy.py) |
| Corrected Theorem 2.1, float-free | [`scripts/verify_convergent_occ_en.py`](scripts/verify_convergent_occ_en.py) |
| Consumer of these numbers | `src/hir/query/dep_graph.rs` |

## Reproduce

```sh
python3 research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py --measure
python3 research/fibonacci-occupancy/scripts/verify_convergent_occ_en.py
```

## Outcome

The 64-slot occupancy is safe: both the mathematical `1/φ` and the truncated
32-bit multiplier give 55/64, and the counterexamples at `n = 4, 7, 9, 10..12,
14, 56..58, 63` only rule out the naive “`Occ(n,n,1/φ) = q_k`” claim.  The
corrected convergent theorem was verified over all cylinders, both signs of
`ε`, including the wrap branch.