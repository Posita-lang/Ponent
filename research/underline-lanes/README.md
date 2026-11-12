# Underline lane assignment research

Everything about the question “can optimal underline-lane assignments be
minimised through a distributive lattice?” lives in this directory, so a
researcher does not have to search the tree.

## Map

| Material | Location |
|---|---|
| Problem note: why the lattice route is closed, what survives | [`no-distributive-lattice.md`](no-distributive-lattice.md) |
| Lean 0/1 witnesses and hand proofs | [`lean/UnderlineLanes.lean`](lean/UnderlineLanes.lean) |
| Lean project metadata (toolchain, lakefile, manifest) | [`lean/`](lean/) |
| Rust witness tests (Witness A/B/C + Stanley identities) | [`../../src/diagnostics/underline_lanes_witness_tests.rs`](../../src/diagnostics/underline_lanes_witness_tests.rs) |
| Independent enumeration of every note claim | [`scripts/verify_witnesses.py`](scripts/verify_witnesses.py) |

The bucket-occupancy / Fibonacci-hashing material is a different problem and
lives in [`../fibonacci-occupancy/`](../fibonacci-occupancy/README.md).

## Reproduce

```sh
python3 research/underline-lanes/scripts/verify_witnesses.py

(cd research/underline-lanes/lean && lake build)   # Lean witnesses
cargo test --bin ponent underline_lanes            # Rust witness tests
```

The Rust witness module stays in `src/diagnostics/` because cargo compiles it
there; everything else for this problem is contained in this directory.

## Outcome

The distributive-lattice route is closed: three interval-graph witnesses kill
the pointwise lattice, the partition-refinement lattice, and any
symmetry-invariant lattice order.  The surviving fragment is a disjoint union
of distributive lattices indexed by acyclic orientations — real but
exponentially large.  Engineering decision: keep the greedy χ-lane assignment
plus the `O(n log n)` rearrangement post-pass; exact IGMRC is polynomial but
not worth its machinery at `n ≤ ~50`, `χ ≤ ~5`.