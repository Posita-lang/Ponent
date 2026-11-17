# Underline lane assignment: why the distributive-lattice route is closed

> **TL;DR.** The distributive-lattice route to minimising WDS over *all* optimal
> lane assignments is closed: three interval-graph witnesses kill, respectively,
> the pointwise lattice, the partition-refinement lattice, and (strongest) the
> existence of any symmetry-invariant lattice order. What survives is
> `C = ⊔_D C_D`, a disjoint union of distributive lattices indexed by acyclic
> orientations — real, but exponentially many, so industrially useless.
> Engineering decision: keep the greedy χ-lane assignment plus the `O(n log n)`
> rearrangement post-pass; exact IGMRC is NP-hard in general (FPT via the
> `O(n · k! · k)` sweep DP, polynomial only on fully saturated instances), the
> post-pass is exact only among relabellings of the greedy colouring, and the
> exact machinery is not worth it at `n ≤ ~50`, `χ ≤ ~5`. Details and the
> corrected complexity statement live in [`../igmrc/README.md`](../igmrc/README.md).

## 1. Context

`src/diagnostics/glyph.rs::UnderlineLanes::assign` partitions same-row
underlines into lanes, which is interval-graph colouring. The primary
objective — minimise lane count, equal to the chromatic number
`χ = ω = k` — is solved optimally by an earliest-ending greedy sweep. The
secondary objective is the weighted depth sum

$$\mathrm{WDS}(c) \;=\; \sum_{u} w_u \cdot \mathrm{lane}(u) \;=\; \sum_u w_u\, c(u),
\qquad w_u \ge 0.$$

A natural theoretical question is whether one can minimise WDS over *all*
optimal colourings in polynomial time by exploiting a distributive-lattice
structure on the colouring set. This document records the negative answer
and the precise positive fragment that survives.

**Dictionary and conventions.** A *lane* is a colour class (an independent set
of pairwise non-overlapping underlines); a *lane number* is the colour itself,
in `[k] = {1, …, k}`. We write `c(u)` interchangeably with `lane(u)`. A
colouring is **normalised** on a maximum clique `K = {v₁, …, v_k}` when
`c(vᵢ) = i`; this kills the free `S_k` colour gauge on `K`, and it is the only
setting in which pointwise `min/max` are well-defined (§7). `C = C(G, K)`
denotes the set of normalised proper `k`-colourings. "An orientation of every
edge" means an orientation of `E(G)` — not an orientation of a complete graph.

**IGMRC** — *interval-graph minimum-weight recolouring*: minimise `WDS(c)` over
all proper `χ`-colourings `c : V(G) → [χ]`. This is the full-space problem; the
post-pass of §2 optimises only over relabellings of one fixed colouring, and §9
gives the polynomial algorithm and the engineering decision.

## 2. The lane-number problem is already solved, hence out of scope here

The industrial fix does not search all colourings: it keeps the greedy
colouring and relabels lanes by descending visual weight, which is the
rearrangement-inequality optimum on the single colouring the greedy
produced — an `O(n log n)` post-pass that strictly improves WDS whenever a
later-opened lane outweighs an earlier one. It is **not exact**: relabelling
cannot change the partition the greedy chose, and that partition can be
strictly suboptimal already at `χ = 2`. With

```
intervals (0,2) (0,2) (3,5),   w = (1, 2, 1),
```

the greedy takes `{v₁}`, `{v₂}`, then `v₃` joins the first lane, so the
post-pass returns 6 (`Σ w·lane = 1·1 + 2·2 + 1·1`), while the colouring
`v₂ ↦ 1, v₁ ↦ 2, v₃ ↦ 2` costs 5. Reassigning lanes inside a fixed
colouring cannot repair this, so the post-pass is a heuristic with an
unmeasured corpus gap (§9, pinned in `verify_witnesses.py`).

On interval graphs with `χ = ω = k ≤ n` the exact problem is NP-hard in
general (padding reduction, [`../igmrc/README.md`](../igmrc/README.md) §1)
and FPT via the `O(n · k! · k)` sweep DP of §9; it is polynomial only on the
fully saturated subclass. The machinery is not needed for the rows that
occur here. This document only closes the distributive-lattice thread; the
engineering decision is stated once, in §9.

## 3. Negative proposition

**Proposition.**

1. There exist interval graphs `G` with `χ(G) = ω(G) = k` whose normalised
   optimal-colouring set `C` is **not a lattice under the pointwise order**
   (Witness A) and **not a lattice under the partition-refinement order**
   (Witnesses A and B). Witness B also shows the two orders genuinely diverge:
   its pointwise poset is a two-element chain — hence a distributive lattice —
   while its refinement poset is a two-element antichain.
2. A stronger form holds: there is an interval graph whose optimal-colouring
   set admits **no canonical lattice order at all**, distributive or otherwise,
   once the order is required to be invariant under vertex relabelling and
   colour permutation (Witness C). There the normalisation-preserving symmetry
   group acts transitively, forcing any compatible order to be an antichain.

All three witnesses are interval graphs, so the negative conclusion lands on
the class we care about, not merely on chordal or perfect graphs. (The role of
perfectness is discussed once, in §10.)

**Lemma (`k`-block antichain).** Let `P ≠ Q` be partitions of a finite set into
exactly `k` blocks each. Then no `k`-block partition is a common refinement of
`P` and `Q`, and no `k`-block partition is a common coarsening. Consequently
distinct `k`-block partitions are incomparable under refinement and have no
meet or join inside the set of `k`-block partitions; a set of ≥ 2 such
partitions is an antichain and not a lattice.

*Proof.* If `R` refines `P` and `|R| = |P| = k`, then `R = P`; likewise a
`k`-block coarsening of `P` equals `P`. A common refinement (or coarsening)
with `k` blocks would therefore equal both `P` and `Q`. ∎

Every `c ∈ C` uses all `k` colours (a proper colouring of a graph with `χ = k`
into `[k]`), so as a partition it has exactly `k` blocks and the Lemma applies
verbatim. It is used in §§4–6.

## 4. Witness A — five vertices, two triangles sharing one vertex

Take a triangle `{v₁, v₂, v₃}` and another triangle `{u, v, v₃}` sharing
only `v₃`; add no other edges. So `v₁v₂, v₁v₃, v₂v₃, uv₃, vv₃, uv` are
edges and `u, v` are not adjacent to `v₁, v₂`.

- `ω(G) = 3`: no `K₄` exists, since any 4-vertex set containing `u` or `v`
  is missing an edge to `v₁` or `v₂`.
- `χ(G) = 3`: `v₁ = 1, v₂ = 2, v₃ = 3, u = 1, v = 2` is proper.
- The realisation below makes it an interval graph, hence chordal and
  perfect: interval graphs are chordal because the interval with the earliest
  right endpoint is simplicial — every neighbour contains that endpoint, so
  `N(v)` is a clique — and deleting it and iterating gives a perfect
  elimination ordering (Fulkerson–Gross 1965). Hence `χ = ω` comes for free.
- **It is an interval graph.** One realisation is

  ```
  v₃ = [0, 10],  v₁ = [0, 3],  v₂ = [2, 4.5],  u = [5.5, 8],  v = [7, 10]
  ```

  checking: `v₁∩v₂, v₁∩v₃, v₂∩v₃, u∩v₃, v∩v₃, u∩v` are all non-empty and
  `v₁, v₂` are disjoint from `u, v`.

  ```
       |       |   |     |   |     |   |       |
       0       2   3     4.5 5.5   7   8       10
  v₃   [=======================================]  [0,10]
  v₁   [===========]                              [0,3]
  v₂           [=========]                        [2,4.5]
  u                          [=========]          [5.5,8]
  v                                [===========]  [7,10]
  ```

Normalise on `K = {v₁, v₂, v₃}` by `c(vᵢ) = i`. Since `u, v` are both
adjacent to `v₃` (colour 3), each must receive colour 1 or 2; since `uv` is
an edge they must differ. There are exactly two normalised optimal colourings:

```
c = (u ↦ 1, v ↦ 2),     d = (u ↦ 2, v ↦ 1).
```

**Pointwise order** (`c ≤ d` iff `c(x) ≤ d(x)` for all `x`): `c` and `d` are
incomparable (`c(u) < d(u)` but `c(v) > d(v)`). Their pointwise

```
min(c, d) = (u ↦ 1, v ↦ 1),     max(c, d) = (u ↦ 2, v ↦ 2)
```

both violate the edge `uv`. With `C = {c, d}` there is no join and no meet
inside `C`, so `C` is not a lattice.

**Partition-refinement order:** both are 3-block partitions, so by the Lemma
they are an antichain with no internal meet or join. (Explicitly: their common
refinement is the discrete 4-block partition, and their common coarsening
merges adjacent vertices.)

`G` is connected; §10 shows five vertices is minimal for pointwise failure
among connected interval graphs.

## 5. Witness B — four vertices, triangle plus a pendant

Let `G` be a triangle `a, b, c` with a pendant vertex `x` adjacent only to `a`.
Then `ω(G) = χ(G) = 3`.

- **It is an interval graph.** One realisation:

  ```
  a = [0, 4],  b = [1, 3],  c = [2, 5],  x = [0, 0.5]
  ```

  checking: `a∩b, b∩c, a∩c, a∩x` are non-empty and `x` is disjoint from
  `b, c`.

Up to renaming colours, the proper 3-colourings as partitions into independent
sets are

```
π₁ = {{a}, {b, x}, {c}},     π₂ = {{a}, {b}, {c, x}}.
```

Under partition refinement these two are incomparable (Lemma). Their common
refinement is the discrete partition `{{a}, {b}, {c}, {x}}`, which uses 4
colours and so is not a `χ = ω` colouring; their common coarsening merges
`{b, c, x}`, giving `{{a}, {b, c, x}}`, which is not proper because `b` and `c`
are adjacent. Thus the optimal-colouring set has no meet or join inside
itself under refinement.

**But B is a lattice under the pointwise order.** Normalise on the unique
maximum clique `K = {a, b, c}`, so `c(a) = 1, c(b) = 2, c(c) = 3`; `x` is
adjacent only to `a`, hence `c(x) ∈ {2, 3}`:

```
C = { (1, 2, 3, 2), (1, 2, 3, 3) },
```

a two-element chain, hence a distributive lattice (`min` and `max` are the two
elements themselves). So B kills refinement lattice structure *without*
killing pointwise lattice structure — the two orders genuinely diverge. This
is exactly why §3 attributes the pointwise half of the proposition to A alone.

B is connected with four vertices; §10 shows it is minimal for refinement
failure.

## 6. Witness C — `K₃ ⊔ K₃`, no canonical lattice order at all

Let `G = K₃ ⊔ K₃` be two disjoint triangles. Then `χ = ω = 3`, `G` is perfect,
and it is an interval graph (each `K₃` is; place the two realisations on
disjoint real segments). Write `A = {a₁, a₂, a₃}`, `B = {b₁, b₂, b₃}`. Fix the
normalisation `c(aᵢ) = i` on `A` and put `p(i) = c(bᵢ)`. Every optimal
colouring is a bijection on each maximum clique, so each colour class contains
exactly one vertex from each triangle; the normalised optimal colourings are
then in bijection with `S₃` via `p`.

**(a) Pointwise order on colour functions.** Two distinct permutations are
never comparable: if `c ≤ d` coordinatewise and both use the colours
`{1, 2, 3}` on `B`, then `Σ_B c = Σ_B d = 6` forces `c = d`. The set is not
even closed under the operations — `min((1,2,3), (2,1,3)) = (1,1,3)` puts a
single colour on two adjacent vertices of `B`. In fact the pointwise minimum
of *any* two distinct colourings is improper, by the same sum argument.

**(b) Partition-refinement order.** All six colourings have block-type
`(2, 2, 2)`, so by the Lemma they are an antichain; a common refinement would
require `> 3` blocks and a common coarsening would merge adjacent vertices.

**(c) Weak order on `S₃`: a lattice, but not distributive, and convention
matters.** Let `s₁ = (12)`, `s₂ = (23)`, write reduced words in the alphabet
`{1, 2}`, and let `w₀ = 321 = (13)` be the longest element. Take the
**suffix** (often called *left*) weak order: `u ≤ v` iff a reduced word for
`u` is a suffix of one for `v` (equivalently `v = x·u` with
`ℓ(v) = ℓ(x) + ℓ(u)`). Its Hasse diagram is the hexagon

```
              w₀ = 121 = 212
             /             \
        s₂s₁ = 21         s₁s₂ = 12
            |                 |
         s₁ = 1             s₂ = 2
             \             /
                   e
```

In this lattice `s₂ ∧ s₂s₁ = e`, `s₁ ∨ e = s₁`, `s₁ ∨ s₂ = w₀`,
`s₁ ∨ s₂s₁ = s₂s₁` and `w₀ ∧ s₂s₁ = s₂s₁`, so

```
s₁ ∨ (s₂ ∧ s₂s₁) = s₁  ≠  s₂s₁ = (s₁ ∨ s₂) ∧ (s₁ ∨ s₂s₁).
```

Under the **prefix** (*right*) weak order the same expression gives `w₀` on
both sides and is *not* a counterexample; the mirror instance
`s₁ ∨ (s₂ ∧ s₁s₂) = s₁ ≠ s₁s₂ = (s₁ ∨ s₂) ∧ (s₁ ∨ s₁s₂)` works there instead.
Either hexagon contains `N₅` (e.g. `{e, s₁, s₂, s₂s₁, w₀}` in the suffix
order), so neither is distributive. Fixing the convention affects only the
displayed witness, not the conclusion.

**(d) No canonical lattice order exists at all.** Let `Γ` be the subgroup of
`Aut(G) × S₃(colours)` preserving the normalisation `c(aᵢ) = i`: a colour
permutation `σ` must be paired with the vertex automorphism `σ` of the *first*
triangle (otherwise `c(aᵢ) = i` is destroyed), and may be paired with an
arbitrary automorphism `ρ` of the second triangle. In the coordinate `p` this
acts as

```
p ↦ σ ∘ p ∘ ρ⁻¹,
```

so already `σ = q p⁻¹` with `ρ = id` sends `p` to `q`: `Γ` acts
**transitively** on the six colourings.

> **Transitivity lemma.** If a finite group acts transitively on `X` and `≤` is
> an invariant order, then `≤` is trivial. Indeed, if `x < gx` for some `g` of
> finite order `m`, applying `g` repeatedly gives
> `x < gx < g²x < ⋯ < gᵐx = x`, contradicting strictness. A lattice on
> `|X| ≥ 2` elements always has comparable pairs, so no invariant lattice
> order exists.

If one first quotients out the colour-permutation gauge (colourings up to
colour names), the residual vertex symmetry includes the diagonal subgroup,
which acts by **conjugation** `p ↦ σ p σ⁻¹`. Invariance forces each conjugacy
class — `{e}`, the three transpositions, the two 3-cycles — to be an
antichain. A finite lattice has a least element `0` and a greatest element
`1`, and both are fixed by every order automorphism (by uniqueness); hence
both must be fixed points of conjugation. But `Z(S₃) = {e}`, so the only fixed
point is `e`, forcing `0 = 1 = e` on a six-element set — contradiction. (The
bare claim "no lattice on six elements has a 3-element antichain" is false:
`M₃` with a new bottom is a 6-element lattice containing one. The fixed-point
argument is what actually uses invariance.)

Finally, `w₀ = 321 = (13)` is itself a **transposition**, hence not a
conjugation fixed point. Since `w₀` is the top of the weak-order lattice of
(c), this is exactly why that lattice is not invariant under the symmetries we
require — i.e. why it cannot be canonical.

**(e) Normalisation is not the culprit.** A has two maximum cliques
(`{v₁, v₂, v₃}` and `{u, v, v₃}`), and the swap `(v₁, v₂) ↔ (u, v)` is an
automorphism, so normalising on either gives the same two-element antichain.
B's maximum clique `{a, b, c}` is unique. C's transitivity argument does not
depend on which triangle we normalise on (the two are interchangeable by an
automorphism) and in fact needs no normalisation at all: the full 36-element
colouring set is transitive under `Aut(G) × S₃(colours)`. The negative results
are therefore robust to the gauge-fixing choice.

## 7. What is true: `C = ⊔_D C_D`

Fix a maximum clique `K = {v₁, …, v_k}` and normalise every optimal colouring
by `c(vᵢ) = i` (§1). For a colouring `c`, orient each edge `xy` from the
smaller colour to the larger; this gives an acyclic orientation `D_c` of every
edge. For an acyclic orientation `D` of every edge extending
`v₁ → ⋯ → v_k`, define

```
C_D = { c ∈ C : c(x) < c(y)  whenever  x → y  in  D }.
```

`D` makes `V(G)` a poset `P_D` (transitive closure), and `C_D` is exactly the
set of strict order-preserving maps `P_D → [k]` with prescribed values on `K`.
If `f, g ∈ C_D` and `x < y` in `P_D`, write `a = f(x), b = f(y), a' = g(x),
b' = g(y)`, so `a < b` and `a' < b'`. Then `min(a, a')` is `a` or `a'`, and in
either case it is `< b` and `< b'`, hence `< min(b, b')`; dually
`max(a, a') < max(b, b')`. Therefore

```
min(f, g) ∈ C_D,     max(f, g) ∈ C_D,
```

so `C_D` is a sublattice of the distributive lattice `[k]^{V(G)}` — the
classical `P`-partition fact that maps from a poset into a chain form a
distributive lattice.

Every `c ∈ C` lies in exactly one `C_{D_c}`, hence

```
C = ⊔_D  C_D        (over the D with C_D ≠ ∅),
```

a disjoint union of distributive lattices indexed by the induced acyclic
orientations. A disjoint union of lattices is not in general a lattice: in
Witness A the two `C_D` are one-element lattices and the pointwise meet/join
of `c` and `d` leaves `C`.

**Where the decomposition comes from.** The identity actually used above is
elementary, and is *not* a reciprocity theorem: each proper colouring
`c : V(G) → [t]` determines exactly one compatible `D_c`, and `c` is a strict
order-preserving map `P_{D_c} → [t]`, so

```
χ_G(t)  =  Σ_{D acyclic} Ω°_{P_D}(t)          (strict order polynomial).
```

Stanley's reciprocity is the companion identity for *weak* order-preserving
maps (equalities allowed along arcs):

```
(-1)^n χ_G(-t)  =  Σ_{D acyclic} Ω_{P_D}(t)    (order polynomial),
```

together with `#AO(G) = (-1)^n χ_G(-1)`. In particular
`Σ_D t^{#sources(D)}` is *not* equal to `(-1)^n χ_G(-t)`; this is checked on
`K₂` and `K₃` in §12. A related theorem of Propp says that the poset of
acyclic orientations of a **connected** graph with a given sink is itself a
disjoint union of distributive lattices; the published proof carrier is
Ehrenborg–Slone (§11).

**Optimising inside one `C_D` is polynomial — but not because of lattice
closure.** `C_D` is the integer solution set of the difference constraints
`c(y) ≥ c(x) + 1` for every arc `x → y` of `D`, together with `c(vᵢ) = i` and
`1 ≤ c ≤ k`. Its constraint matrix is a network matrix (the node-arc
incidence matrix of a digraph), hence totally unimodular, so the LP relaxation
is integral; equivalently, the lattice bottom is the least solution of this
system, computed by a single longest-path pass over the acyclic digraph `D`.
Since `w ≥ 0`, a linear objective is minimised at that bottom. Coordinatewise
closure alone would not give this.

If colourings are taken up to permutation of colour names, pointwise
`min/max` are not even well-defined, so the lattice question must be asked on
normalised/labelled colourings.

## 8. Why colourings are the wrong shape of object

The phenomenon "the optimal solutions form a distributive lattice" is real
but lives on families of *sets* closed under `∪ / ∩` or `min / max`, forced
by submodularity plus a min–max theorem:

1. **Maximum antichains of a finite poset** form a distributive lattice
   with `A ∧ B = min(A ∪ B)` and `A ∨ B = max(A ∪ B)` (graph-theoretically:
   maximum stable sets of a comparability graph). This is the stable-set
   side of Dilworth–Mirsky — the side where `χ = ω` is guaranteed.
2. **Minimum `s`-`t` cuts** form a distributive lattice under `∪, ∩`: cut
   capacity is submodular, so the union and intersection of two min cuts
   are again min cuts (classical; the basis of Picard–Queyranne-type
   structure theorems).
3. **Maximum-weight closures** of a digraph, by the same
   inclusion–exclusion argument.

In each case distributivity is free because `∪, ∩` distribute, and Birkhoff
then identifies these lattices with ideal-lattices of posets.

A colouring, by contrast, is a *partition*, and properness is a conjunction
of binary `≠` constraints. Such solution sets are not closed under
coordinatewise lattice operations — taking minima creates exactly the
equalities on edges that are forbidden (that is literally what failed in
Witness A and in §6(a)). Contrast Horn formulas, whose satisfying
assignments form sublattices of the Boolean cube and *do* carry canonical
distributive lattices. On top of that, colourings come with a free `S_k`
gauge, so the natural object is closer to a torsor than to a lattice.

## 9. Consequences for IGMRC

- **The distributive-lattice route is closed** (§§3–6): one cannot minimise WDS
  over all optimal colourings by riding a global lattice structure, because
  no such lattice exists in general.
- **IGMRC is FPT, not polynomial** — the sweep DP below is the exact engine,
  not lattice structure. With `k` part of the input the exact-`χ` rank-1
  problem is NP-hard even with unit weights (padding reduction,
  [`../igmrc/README.md`](../igmrc/README.md) §1), and the clique LP is not
  value-exact (explicit `n = 9`, `k = 4` half-integral counterexample, gap
  `1/2`), so there is no polynomial LP engine either. The polynomial case is
  the **fully saturated** subclass (every maximal clique of size exactly `k`);
  in the general case the DP runs in `O(n · k! · k)`, which is polynomial only
  for fixed `k`. Sweep the `2n`
  endpoints in increasing order. At coordinate `t` the active set
  `A(t) = {v : s_v ≤ t ≤ e_v}` is a clique, so `|A(t)| ≤ k`; conversely, by
  Helly's property for intervals, every clique is contained in some `A(t)`, so
  two intervals conflict iff they are simultaneously active at some `t`. The
  sweep state is therefore sufficient. State = an injective map `A(t) → [k]`,
  at most `k! ≤ 120` states for `k ≤ 5`. At each coordinate, process all
  starts `s_v = t` **before** all ends `e_v = t` (so intervals meeting at a
  shared endpoint are still simultaneously active and must get distinct
  colours); for a start, extend the state by any colour not currently used on
  `A(t)` and add `w_v · i`, and for an end delete `v`. Merge equal states
  keeping the minimum accumulated cost. The DP enumerates exactly the proper
  `k`-colourings and returns `min WDS` in `O(n · k! · k)` time. It was
  cross-checked against brute force on 300 random interval families (§12).
  The witnesses above are orthogonal to this: they block one proof strategy,
  not the DP.
- **"The equivalent min-cost flow" is not equivalent.** The classical interval
  flow (Arkin–Silverberg 1987; Carlisle–Lloyd 1995) maximises the weight of a
  `k`-colourable vertex *subset*; it returns a subset, not lane indices, so it
  cannot evaluate `Σ_v w_v c(v)`. Any flow formulation that does evaluate it
  is the DP's `k!`-state graph, i.e. a restatement of `O(n · k! · k)`, not a
  new bound. Only on fully saturated instances does a nested-flow
  (cumulative-variable) formulation of the colouring itself exist.
- **The `C = ⊔_D C_D` decomposition is real but industrially useless**:
  optimising a linear objective inside a single `C_D` is polynomial (§7), but
  the number of acyclic orientations is `(-1)^n χ_G(-1)` — exponential — so
  one cannot enumerate the `D`.
- **Engineering decision (stated once).** Keep the greedy χ-lane assignment
  plus the `O(n log n)` rearrangement post-pass of §2; it is optimal among
  relabellings of the greedy colouring, and for the `n ≤ ~50`, `χ ≤ ~5` rows
  in this codebase it avoids the DP/flow machinery. We have **not** measured
  the remaining gap between the relabelling optimum and exact IGMRC on the
  corpus, so this is a pragmatic complexity trade-off (`O(n log n)` versus
  `O(n · k!)` plus bookkeeping), not a theorem that no further gain exists.

## 10. Notes

- `χ = ω` does not even imply perfectness: `C₅ ⊔ K₃` has `χ = ω = 3` and an
  induced `C₅`. All three witnesses above are perfect (indeed interval), so
  restricting to perfect graphs does not rescue the claim.
- Degenerate positive cases: uniquely `k`-colourable graphs give a
  one-element lattice; when `χ = ω = 2` the colourings form a `ℤ/2`-torsor,
  and a Boolean lattice can be imposed only after non-canonical choices.
- If what one had in mind was **maximum stable sets** rather than
  colourings: yes for comparability graphs (§8 item 1), but again no
  canonical order for perfect graphs in general — in `2K₃` the nine maximum
  stable sets are permuted transitively by automorphisms.
- **Minimality of the witnesses.** Enumerating normalised optimal colourings
  for every connected graph on ≤ 4 vertices with `χ = ω` gives:

  | connected interval graph | n | χ=ω | `C` (pointwise) | `C` (refinement) |
  |---|---|---|---|---|
  | `K₃`, `P₃` | 3 | 3 / 2 | singleton | singleton |
  | `K₄` | 4 | 4 | singleton | singleton |
  | `K₄ − e` (diamond) | 4 | 3 | singleton | singleton |
  | `P₄`, `K₁,₃` | 4 | 2 | singleton | singleton |
  | paw = Witness B | 4 | 3 | 2-chain (distributive lattice) | 2-antichain ← first refinement failure |
  | two triangles at a vertex = Witness A | 5 | 3 | 2-antichain ← first pointwise failure | 2-antichain |

  `C₄` is excluded (not chordal, hence not an interval graph); on ≤ 4 vertices
  no connected graph has a pointwise failure. Dropping connectedness,
  `K₃ ⊔ K₂` (5 vertices, χ = ω = 3) already fails pointwise, so "connected"
  is a real qualifier. All rows were checked by brute-force enumeration
  (`verify_witnesses.py`, §12). Note that normalisation must be on a *maximum*
  clique: normalising on
  a smaller clique leaves the `S_k` gauge only partly fixed and the resulting
  enumeration is not `C`.

## 11. References

- arXiv:1805.06699 — Araújo et al., *Dual parameterization of Weighted
  Coloring* (2018). Guan–Zhu weighting `Σ_i max_{v ∈ S_i} w(v)`, a
  different objective from `Σ_v w_v · c(v)`; interval graphs admit a
  polynomial kernel.
- arXiv:2504.21502 — Bodlaender, Hermelin, van Leeuwen,
  *Concurrency Constrained Scheduling with Tree-Like Constraints* (2025).
  Sum Coloring on bounded treewidth, FPT/XALP dichotomy.
- arXiv:1602.07827 — Hell, Mohammadi Nevisi, *Minimum Cost Homomorphisms
  with Constrained Costs* (2016). Min-cost homomorphism *to* a proper
  interval graph is polynomial (direction opposite to IGMRC).
- R. P. Stanley, *Acyclic orientations of graphs*, Discrete Mathematics
  **5**(2) (1973), **171–178**. Two facts used in §§7, 9:
  (i) `#AO(G) = (-1)^n χ_G(-1)`;
  (ii) reciprocity `(-1)^n χ_G(-t) = Σ_{D acyclic} Ω_{P_D}(t)` (weak order
  polynomial), whose strict-order companion
  `χ_G(t) = Σ_D Ω°_{P_D}(t)` is the decomposition invoked in §7.
- J. Propp, *Acyclic orientations with a fixed sink form a disjoint union
  of distributive lattices* (unpublished note; original source not
  formally published or online-findable). The published proof carrier is
  R. Ehrenborg and M. Slone, *A geometric approach to acyclic
  orientations*, Order **26**(4) (2009), 283–288,
  doi:10.1007/s11083-009-9122-z, arXiv:0907.0046, whose abstract reads:
  "The set of acyclic orientations of a **connected** graph with a given
  sink has a natural poset structure. We give a geometric proof of a result
  of Jim Propp: this poset is the disjoint union of distributive lattices."
  This is the positive counterpart cited in §7.
- D. R. Fulkerson and O. A. Gross, *Incidence matrices and interval
  graphs*, Pacific J. Math. **15**(3) (1965), 835–855 — the interval-graph
  characterisation behind the simplicial-vertex / chordality argument in §4.
- D. Marx, *A short proof of the NP-completeness of minimum sum interval
  coloring*, Oper. Res. Lett. **33**(4) (2005), 382–384,
  doi:10.1016/j.orl.2004.07.006 — unit-weight NP-hardness of sum coloring on
  interval graphs, hence of IGMRC after the exact-`χ` padding of
  [`../igmrc/README.md`](../igmrc/README.md) §1.
- E. M. Arkin and E. B. Silverberg, *Scheduling jobs with fixed start and end
  times*, Discrete Appl. Math. **18**(1) (1987), 1–8, and R. E. Carlisle and
  E. L. Lloyd, *On the k-coloring of intervals*, Discrete Appl. Math. **59**
  (1995), 225–235 — the min-cost-flow algorithms cited in §9. Both compute a
  maximum-weight `k`-colourable **subset**, not a colouring; see the §9
  bullet on why that objective differs.

## 12. Witness tests

All three witnesses are 4–6 vertex graphs, so the mathematical assertions
above are pinned in `src/diagnostics/underline_lanes_witness_tests.rs`. The
file is self-contained (no crate dependencies), so it can be run standalone:

```bash
rustc --test src/diagnostics/underline_lanes_witness_tests.rs -o /tmp/witness_tests && /tmp/witness_tests
```

or, if a `#[test]` is hoisted into the crate's test module, run it with the
usual `cargo test`. This follows the repository habit of pinning
mathematical claims into tests — cf. `wrap.rs`'s
`ascii_fast_path_breaks_match_uax`, `jump_table_matches_fitness_jump` — so
that the document and the code cannot drift apart. For example, the B
contrast of §5 is a handful of lines:

```rust
# [test]
fn witness_b_pointwise_is_a_chain_but_refinement_is_not() {
    let g = Graph::new(4, &[(0, 1), (0, 2), (1, 2), (0, 3)]);
    let c = normalised(&g, 3, &[0, 1, 2]);            // c(x) ∈ {2, 3}
    assert_eq!(c.len(), 2);
    assert!(leq(&c[0], &c[1]) && !leq(&c[1], &c[0])); // pointwise: 2-chain
    assert!(c.contains(&pmin(&c[0], &c[1])));
    assert!(c.contains(&pmax(&c[0], &c[1])));
    let (p, q) = (blocks(&c[0], 3), blocks(&c[1], 3));
    assert!(!refines(&p, &q) && !refines(&q, &p));    // refinement: antichain
}
```

The other tests pin: A's `min/max` falling outside `C`, the antichain, and
independence of the normalising clique; C's six-element pointwise antichain,
the size-6 orbit of `p ↦ σ ∘ p ∘ ρ⁻¹`, and the fact that conjugation fixes
only `e`; and the corrected Stanley identities (decomposition `χ_G(t) =
Σ_D Ω°(t)` and reciprocity `(-1)^n χ_G(-t) = Σ_D Ω(t)`) on `K₂`, `K₃`,
including the failure of `Σ_D t^{#sources(D)}`.

An independent enumeration harness, `research/underline-lanes/scripts/verify_witnesses.py` (`python3
research/underline-lanes/scripts/verify_witnesses.py`, prints `ALL CHECKS PASSED`), additionally checks the
minimality table of §10, both weak-order conventions of §6(c), the interval
realisations of §§4–5, and cross-checks the IGMRC sweep DP of §9 against
brute-force enumeration over all proper colourings on 300 random interval
families.
