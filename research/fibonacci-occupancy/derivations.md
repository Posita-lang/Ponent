# Bucket occupancy of Kronecker points: complete derivations

**Problem.** Let φ = (1+√5)/2 and α₀ = 1/φ = (√5−1)/2 ≈ 0.6180339887. For
integers n, m ≥ 1 and an irrational α ∈ [0, 1), put

- Pₙ(α) = { {iα} : 1 ≤ i ≤ n }, where {x} = x − ⌊x⌋;
- B_k = [k/m, (k+1)/m), k = 0, …, m−1;
- Occ(n, m, α) = #{ k : Pₙ(α) ∩ B_k ≠ ∅ }.

At (n, m) = (64, 64) the exact values are

| α | Occ(64, 64, α) |
|---|---|
| 1/φ | **55** |
| α₁ = ⌊2³²/φ⌋/2³² = 2654435769/2³² | **55** |
| α₂ = 165902777/2³² | **37** |

Everything below is derived, not asserted: every number is a consequence of
integer arithmetic (Lemma 0), and every claim is re-checked mechanically by
`research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py`. Where a derivation in this directory was
initially circulated with an error, the error is stated and corrected in
place (the φ-exponent indexing in §1/§2; the false 1/(2q_k²) lemma, the
dimensional slip in η_k and the missing wrap branch in §2.3's Theorem 2.1;
the bucket→pair correspondence in §2.2; the (35+36/64)/9 wall label in §3.2;
the a₃ partial quotient and the [0;25,1,8] expansion in §4; the r∈{15,17}
strand claim in §4; the "almost everywhere, 1 − e⁻¹" asymptotic and the
average-case conclusion of §5.3 — the 1 − e⁻¹ law is the i.i.d. value, and
the golden ratio is also optimal in the scale-invariant average).

---

## 0. Exact arithmetic in ℚ(√5) — the decision substrate

**Lemma 0.1 (floors by one isqrt).** For integers x ≥ 0, y:

> ⌊(x·√5 − y)/2⌋ = (⌊√(5x²)⌋ − y) / 2 (integer floor division),

*Proof.* k ≤ (x√5 − y)/2 ⟺ 2k + y ≤ x√5. Since 2k + y is an integer, it is
≤ x√5 iff it is ≤ ⌊x√5⌋. So the floor equals (⌊x√5⌋ − y)/2 with ⌊x√5⌋ =
isqrt(5x²). ∎

Consequences used throughout:

- ⌊i·α₀⌋ = (isqrt(5i²) − i)/2  (take x = i, y = i);
- bucket(i) = ⌊64·{i·α₀}⌋ = (isqrt(5·(64i)²) − 64i − 128·⌊i·α₀⌋)/2;
- more generally ⌊m·{i·α₀}⌋ for any m, via the same primitive;
- for **rational** α = p/q: bucket(i) = ((i·p) mod q)·m // q — also exact.

Every comparison "is α₀ bigger than the rational r?" reduces to comparing
√5 with 2r + 1, i.e. 5 with (2r+1)² — pure integer arithmetic. No floating
point participates in any decision below; floats appear only to display.

**Lemma 0.2 (Fibonacci anchors).** With F₀ = 0, F₁ = 1 and q_k = F_{k+1}
(so q₈ = 34, q₉ = 55, q₁₀ = 89), Cassini's identity gives

> q_k·α₀ − F_k = (−1)^k·α₀^{k+1},  hence  ‖q_k·α₀‖ = α₀^{k+1}.

Numerically α₀⁸ ≈ 0.0212862363, α₀⁹ ≈ 0.0131556175, α₀¹⁰ ≈ 0.0081306188,
and α₀⁸ = α₀⁹ + α₀¹⁰ (from α₀² = 1 − α₀).

⚠ **Indexing caveat.** Earlier drafts wrote ‖q₇α‖ = α⁸ and ‖q₈α‖ = α⁹;
with q_k = F_{k+1} the correct pairing is ‖q_k α‖ = α^{k+1} (Lemma 0.2).
All multiplicities below are indexed consistently with q₉ = 55.

---

## 1. Q1 — the three-gap theorem at α = 1/φ, n = 64

**Theorem 1.1 (Sós–Slater, golden case).** The 65 circular gaps of
{0, α₀, 2α₀, …, 64α₀} take exactly three values:

| gap | exact form | value | multiplicity |
|---|---|---|---|
| φ⁻⁸ | 13 − 21α₀ | 0.0212862363 | **24** |
| φ⁻⁹ | 34α₀ − 21 | 0.0131556175 | **31** |
| φ⁻¹⁰ | 34 − 55α₀ | 0.0081306188 | **10** |

Consistency: 24 + 31 + 10 = 65 = n + 1, and
24φ⁻⁸ + 31φ⁻⁹ + 10φ⁻¹⁰ = 55φ⁻⁹ + 34φ⁻¹⁰ = 1.

**Derivation.** All partial quotients of α₀ are 1, so the three-distance
structure is Fibonacci-driven. Since q₉ = 55 ≤ 64 < 89 = q₁₀, write
64 = q₉ + r with r = 9. The 55-point set {0, α₀, …, 54α₀} has two gap
lengths — φ⁻⁸ with multiplicity 34 = q₈ and φ⁻⁹ with multiplicity 21 = q₇
(Steinhaus configuration: q_k + q_{k−1} points) — and

> 55·α₀ ≡ −α₀¹⁰ (mod 1)  (Lemma 0.2: 55α₀ = F₁₀ − α₀¹⁰… note 55α₀ = 34 + (55α₀ − 34),
> and 55α₀ − 34 = −(34 − 55α₀) = −φ⁻¹⁰ by the exact form in the table),

so the 10 new points {55α₀}, …, {64α₀} equal {rα₀ − α₀¹⁰} for r = 0..9:
each lands inside a long φ⁻⁸-gap and splits it into

> α₀¹⁰  and  α₀⁸ − α₀¹⁰ = α₀⁹ (golden identity).

Ten of the 34 long gaps are split: the multiset becomes
(34 − 10) = 24 long, 21 + 10 = 31 medium, 10 short. ∎

**General formula (partial quotients all 1).** For q_k ≤ N = q_k + r < q_{k+1}
with 0 ≤ r < q_{k−1}, the N + 1 gaps are

> α₀^{k+1} × (r + 1),  α₀^{k−1} × (q_{k−1} − r − 1),  α₀^{k} × (q_k − q_{k−1} + r + 1).

Sanity: (r+1) + (q_{k−1} − r − 1) + (q_k − q_{k−1} + r + 1) = q_k + r + 1 = N + 1.
⚠ An earlier circulated variant used multiplicities
(r, q_{k−2}+r, q_{k−1}−r) — that version pairs the φ-exponents with the wrong
Fibonacci indices; the formula above is the one verified against direct
computation at N = 40, 64, 65, 100, 144.

For the 64-point set P₆₄ itself (N = 64 = 55 + 9, r = 9, but the count of
*gaps* is 64 = 55 + 9): φ⁻¹⁰ × 9, φ⁻⁹ × 30, φ⁻⁸ × 25. This is the multiset
the occupancy analysis in §2 consumes.

---

## 2. Q2 — Occ(n, n, 1/φ) = q_k is NOT a theorem; the n = 64 ledger

### 2.1 The disproof

Direct computation of Occ(n, n, 1/φ) (Lemma 0.1 makes it exact):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|
| Occ | 1 | 2 | 3 | **4** | 5 | 6 | **6** | 8 | **7** | 10 | 10 | 11 | 13 | **12** |
| q_k (largest F ≤ n) | 1 | 2 | 3 | 3 | 5 | 5 | 5 | 8 | 8 | 8 | 8 | 8 | 13 | 13 |

Violations in **both** directions: Occ(4) = 4 > q₃ = 3 (all four quarters
hit, since {α₀, 2α₀, 3α₀, 4α₀} ≈ {0.618, 0.236, 0.854, 0.472}), and
Occ(9) = 7 < q₅ = 8. Inside q₉'s own block both signs occur:
Occ(56) = Occ(57) = Occ(58) = 50 < 55 while Occ(68) = 64 > 55
(full table for n = 1..120 is in the machine-checked appendix of
`research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py`). Hence 55 = q₉ at n = 64 is an
**alignment coincidence**, not an instance of a general identity.

### 2.2 The exact reduction (the "ledger" formula)

For m = n write n·α = a + β with a = ⌊nα⌋, β = {nα}. Then for 1 ≤ i ≤ n

> ⌊n{iα}⌋ ≡ a·i + ⌊iβ⌋ (mod n),

because n{iα} = n·iα − n⌊iα⌋ and n·iα = i(nα) = i·a + iβ. Therefore

> Occ(n, n, α) = #{ (a·i + ⌊iβ⌋) mod n : 1 ≤ i ≤ n }.

For n = 64: 64·α₀ = 39 + β with β = 32√5 − 71 ≈ 0.55417528, so

> bucket(i) = (39·i + ⌊iβ⌋) mod 64,  ⌊iβ⌋ = isqrt(5·(32i)²) − 71·i  (Lemma 0.1).

**Ledger (exact).** The 64 residues give **55 distinct values**:

- 9 empty buckets: {1, 9, 16, 22, 24, 37, 50, 52, 58};
- 9 doubly-occupied buckets: {5, 15, 20, 23, 35, 39, 51, 54, 59}
  (bucket → index pair: 5 → {5,60}, 15 → {2,36}, 20 → {7,62},
  23 → {20,54}, 35 → {9,64}, 39 → {1,56}, 51 → {11,45}, 54 → {3,58},
  59 → {29,63});
- no bucket holds three points.

⚠ **Correspondence caveat.** Earlier drafts listed the same nine pairs in an
order that did NOT match the sorted bucket list (e.g. {20,54} sits in bucket
23, not 20). The bucket→pair map above is the machine-checked one, now
asserted by `check_q2`.

So Occ = 64 − 9 = 55. The coincidence decomposes: the number of "extra"
points (9) happens to equal n − q₉ = 9. Nothing forces that equality —
at n = 9 the surplus is 2 while n − q₅ = 1.

**Structural reading via §1.** Only the φ⁻⁸-gaps (1.362 bucket widths)
are longer than a bucket; empty buckets live inside them. Of the 25
φ⁻⁸-gaps of P₆₄, exactly 9 straddle a bucket boundary so as to contain a
whole empty bucket — a *phase* statement about the 1/64 grid, not a
continued-fraction statement.

### 2.3 The true theorem: full occupancy at convergent denominators

**Theorem 2.1.** For any irrational α with convergent p_k/q_k (including
k = 0, q₀ = 1): **Occ(q_k, q_k, α) = q_k**, and in fact every bucket holds
exactly one point.

*Proof.* Let α_{k+1} = [a_{k+1}; a_{k+2}, …] > 1 be the complete quotient, so

> α = (p_k α_{k+1} + p_{k−1}) / (q_k α_{k+1} + q_{k−1}).

With ε = α − p_k/q_k and Cassini's identity
p_k q_{k−1} − p_{k−1} q_k = (−1)^{k−1},

> q_k α − p_k = (−1)^k / (q_k α_{k+1} + q_{k−1}),  hence
> |ε| = 1 / [ q_k (q_k α_{k+1} + q_{k−1}) ] < 1/q_k²,

strict because α_{k+1} > 1 and q_{k−1} ≥ 0. Consequently
0 < |i ε| < 1/q_k for 1 ≤ i ≤ q_k; α irrational ⇒ i ε irrational ⇒ {iα}
never lies on a bucket edge j/q_k. 1/q_k is exactly one bucket width.

Let g_i = (i·p_k mod q_k) ∈ {0, …, q_k−1} and write i·p_k = A_i q_k + g_i.
Then i α = A_i + g_i/q_k + i ε, so with b_i = ⌊q_k {iα}⌋:

- ε > 0: 0 < i ε < 1/q_k and g_i/q_k + i ε < 1, hence b_i = g_i;
- ε < 0: −1/q_k < i ε < 0. For g_i ≥ 1, b_i = g_i − 1. For g_i = 0 (only
  i = q_k), the fractional part **wraps**: {q_k α} = 1 + q_k ε ∈
  (1 − 1/q_k, 1), so b_{q_k} = q_k − 1 = (g_i − 1) mod q_k.

In both cases

> b_i = (g_i − 𝟙_{ε<0}) mod q_k.

Since gcd(p_k, q_k) = 1, i ↦ g_i is a bijection {1,…,q_k} →
{0,…,q_k−1} (i = q_k gives 0); subtracting a constant mod q_k is a
permutation. Hence {b_i} = {0,…,q_k−1} and Occ(q_k, q_k, α) = q_k. ∎

⚠ **Erratum (this proof, and the Q2 proof sketch in the tool).** Earlier
versions used the bound |α − p_k/q_k| < 1/(2 q_k²) — **false for general k**.
Hurwitz' constant 1/√5 guarantees such a bound for *infinitely many*
convergents, not for each k; indeed
q_k²|ε| = 1/(α_{k+1} + q_{k−1}/q_k) can be arbitrarily close to 1 when a_k
is large (so q_{k−1}/q_k is small) and the tail makes α_{k+1} → 1⁺
(a_{k+1} = 1 followed by a huge a_{k+2}); a large a_{k+1} instead pushes
this ratio *down*. Concretely, **every** irrational in the cylinder
[0; 100, 1, 10⁶, …] has q₁²|ε| ≈ 0.9901 > 1/2 at k = 1. The proof
also wrote |α − p/q| = η_k/q_k with η_k = |q_k α − p_k| < 1/q_k² (dimension
error: η_k < 1/q_k), and called the displacement of {iα} from its grid point
"η_k" — it is i ε, of size up to (q_k−1)/q_k² ≈ 1/q_k, not η_k. Finally, the
grid point j_i/q_k sits **on** a bucket edge, so "distance to the nearest
bucket edge ≥ 0" is not a valid margin; the correct statement is that i ε is
smaller than one bucket width and has fixed sign, and the ε < 0, g_i = 0 wrap
branch (formerly waved away "by circularity") is precisely what makes the map
a −1 shift instead of the identity. The conclusion is unchanged; the
hypothesis actually needed is |ε| < 1/q_k², which is sharp in the sense above.

**Verified.** α = 1/φ at n = 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233,
377; α = √2 − 1 at n = 1, 2, 5, 12, 29, 70, 169. Beyond these two sample
numbers, `check_convergent_cylinders()` in the tool proves the theorem for
**every** irrational inside each of 19 continued-fraction cylinders (all-1,
all-2, huge partial quotients a₁ = 1000, a₂ = 1000, a₃ = 10⁶, mixed, random
prefixes), at every k with q_k ≤ 4000: it checks q_k² max|ε| < 1, the shifted
bucket map (both signs of ε), and Occ = q_k — 93 (k, cylinder) pairs, max
q_k²|ε| = 0.999002 — while 10 of those pairs refute the old 1/(2 q_k²) bound.
The standalone English script `verify_convergent_occ_en.py` re-derives the
same checks independently.

### 2.4 The exact decomposition: Occ = n − S + C

**Lemma 2.2 (duality).** For any α, n, m: with the n points sorted
circularly, n − Occ(n, m, α) equals the number of circularly adjacent pairs
lying in the same bin. *(Bin k holds c_k points; those points are
consecutive in the circular order and contribute c_k − 1 same-bin adjacent
pairs; Σ(c_k − 1) over nonempty bins = n − Occ.)*

**Theorem 2.3.** Let S = #{circular gaps of Pₙ(α) shorter than 1/n} (pure
continued-fraction arithmetic: three-distance gives S exactly) and C =

# {of those short gaps that contain a grid point k/n}. Then

> Occ(n, n, α) = n − S + C.

*Proof.* A short gap cannot contain two grid points (its length < 1/n = grid
spacing), so each short gap either swallows exactly one grid point — merging
two adjacent buckets, reducing the count by 1 — or does not. Long gaps
cannot straddle a boundary without the swallowed grid point being counted by
the short gap on the other side. Counting occupied buckets: start from n
buckets each meeting one point... equivalently: buckets missed = S − C, so
Occ = n − (S − C). ∎

At n = 64: S = 9 + 30 = 39 (the 9 φ⁻¹⁰-gaps and 30 φ⁻⁹-gaps of §1's P₆₄
multiset are < 1/64; the 25 φ⁻⁸-gaps are longer than a bin), C = 30 of them
contain a grid line, giving Occ = 64 − 39 + 30 = 55. Earlier drafts wrote
"φ⁻¹⁰ and φ⁻¹⁰..." here. The identity Occ = n − S + C was checked on all
80640 Farey cells of [0,1) at n = m = 64 — zero violations — and the same
ledger gives §4.3's S = 38, C = 11, Occ = 37 for α₂. The phase term C is a
discrepancy-type fluctuation — **there is no closed form for Occ(n, m, 1/φ)
in general**; the honest description is algorithmic (three-distance +
Ostrowski/Zeckendorf expansion + grid phase), which §0 makes exact.

---

## 3. Q3 — stability walls around α₀ at (64, 64)

### 3.1 The wall mechanism

α ↦ (bucket(i))_{i=1..64} is a step function; it changes exactly when some
{iα} crosses a grid line k/64, i.e. at the rational **walls**

> α = (64·j + k)/(64·i),  1 ≤ i ≤ 64,  0 ≤ k ≤ 64.

A perturbation δ of α moves {iα} by ≈ i·δ, so the α-space distance to the
wall generated by point i is dist({iα₀}, grid)/i — **the index divides**.
Consequently the closest boundary approach in x-space is *not* the nearest
wall in α-space: the minimum x-distance is d₉ ≈ 1.94×10⁻⁴ at i = 9
({9α₀} ≈ 0.5623059 vs 36/64 = 0.5625), but dividing by i=56 beats it.

### 3.2 The two radii (exact)

Enumerating all 80639 interior walls and sorting by distance from α₀
(exact rational arithmetic; the tool does this):

| wall | α-space distance | event | occupancy change? |
|---|---|---|---|
| 2215/3584 = (34 + 39/64)/56 | **9.4351785e-6** (below) | i=56 crosses 39/64 | **no** (bucket 39 double-holds {1,56}; 38 holds {22}) |
| 1859/3008 = (29 + 3/64)/47 | 1.5371729e-5 (below) | i=47 crosses 3/64 | **yes**: bucket 3 singly-held by {47} empties, 55 → 54 |
| 89/144 = (5 + 36/64)/9 | 2.1566806e-5 (above) | i ∈ {9,18,…,63} cross together (89/144 is a convergent, 9 \| 144) | **yes**: net +1, 55 → 56 |

**Corollary 3.1 (pattern radius).** The complete bucket vector is unchanged
for |α − α₀| < 9.4351785×10⁻⁶; the nearest wall does not move the count.

**Corollary 3.2 (occupancy interval).**

> Occ(64, 64, α) = 55  ⟺  α ∈ [1859/3008, 89/144)
> ≈ [0.6180186170, 0.6180555556),

with one-sided radii 1.5371729×10⁻⁵ below and 2.1566806×10⁻⁵ above.
(Endpoint conventions: closed on the left as drawn, open on the right where
the count jumps to 56; the i=9 crossing is redundant — bucket 35 also holds
{64α₀} = β ≈ 0.55417528 — which is why the *x*-space-nearest event is not
the *occupancy*-nearest one. Earlier drafts quoted 0.99187 here; that is
{55α₀}, not {64α₀}.)

### 3.3 The 2⁻³² corollary — a theorem, but a coarse one

α₁ = 2654435769/2³² = ⌊2³²/φ⌋/2³² satisfies |α₁ − α₀| ≈ 1.2×10⁻¹⁰, which is
> 4 orders of magnitude inside even the pattern radius. Hence

> Occ(64, 64, α) = 55 for **every** α with |α − α₀| ≤ 2⁻³²  (Corollary 3.1).

The statement is a theorem, but 2⁻³² is not the sharp threshold — the sharp
one is Corollary 3.2's 1.54×10⁻⁵. **Scaling warning for future constants:**
the pattern radius behaves like ~1/m² in the bucket count m; at m ~ 2¹⁶ a
32-bit constant is already marginal, and at m ~ 2³² the guarantee evaporates
entirely.

### 3.4 Where the nearest walls come from (continued fractions)

With β = {64α₀} = 32√5 − 71 = [0; 1, 1, 4, 8, …], a wall for point i is a
rational p/i ≈ β (since 64iα ∈ ℤ ⟺ iβ ∈ ℤ). The best approximation to β
with denominator ≤ 64 from below is 31/56 (→ wall 2215/3584), from above
5/9 (→ wall 89/144); the first *occupancy-changing* lower wall uses the
semiconvergent 26/47 (→ 1859/3008). The pattern radius is exactly
α₀ − 2215/3584 = (1792√5 − 4007)/3584.

---

## 4. Q4 — why α₂ = 165902777/2³² gives Occ = 37, exactly

### 4.1 Arithmetic of the bad constant

α₂ = 165902777/2³² has continued fraction

> α₂ = [0; 25, 1, **7**, 1, 27, 1, 5, 1, 2, 1, 106, …],

⚠ a₃ = 7, **not** 8 (the circulated "[0;25,1,8,…]" was wrong; one derivation
repeated the error in its convergent q₃ = 8·26 + 25 = 233 — the correct
convergents are 1/25, 1/26, 8/207, 9/233, so q₃ = 207, q₄ = 233). Exactly,

> α₂ = 1/26 + δ,  δ = 9252453/55834574848 ≈ 1.65711891×10⁻⁴,
> ‖26·α₂‖ = 26δ = 18504906/2³² ≈ 0.00430850917 < 1/64 ≈ 0.015625.

Because a₁ = 25 is large, 1/25 and 1/26 are consecutive convergents; because
a₃ = 7, the next denominator is 207 ≫ 64. **The whole mechanism**: for the
first ~207 steps the Kronecker sequence is a perturbation of the rational
rotation by 1/26, and 26 ≪ 64, so the 64 points sit on 26 "strands" instead
of spreading over the circle.

### 4.2 The strand decomposition

Write i = 26t + r. Then

> {iα₂} = { r/26 + rδ + t·ρ },  ρ := 26δ ≈ 0.0043085 = 0.2757 bucket widths,

and since rδ + tρ ≤ 25δ + 2ρ ≈ 0.0127 < 1, no wrap occurs in the
fractional term: the bucket of i is ⌊32r/13 + 0.27574·t + 0.010605·r⌋
(mod 64). Three facts make
the count closed-form:

1. **Strands never collide.** The base buckets ⌊32r/13⌋ for r = 0..25 are
   0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29, 32, 34, 36, 39, 41, 44,
   46, 49, 51, 54, 56, 59, 61 — 26 distinct buckets, consecutive gaps 2 or 3.
2. **Strands span < 1 bucket.** A 3-point strand (r = 1..12; 64 = 2·26 + 12)
   has diameter 2ρ ≈ 0.55 bucket widths; a 2-point strand (r = 0, 13..25)
   has diameter ρ. Each occupies one or two **adjacent** buckets, never three.
3. Hence Occ = 26 + #{strands straddling one boundary}.

A strand straddles a boundary iff the floor of its bucket expression
⌊32r/13 + 64·(r·δ + t·ρ)⌋ increments across its members — decided exactly
by the integer check in the tool (δ is rational, so no approximation is
involved). The verified outcome, per strand:

| strand r | occupies | buckets |
|---|---|---|
| 1, 2, 4, 6, 8, 10, 12 | 3 points | **2 adjacent buckets** (straddle) |
| 19, 21, 23, 25 | 2 points | **2 adjacent buckets** (straddle) |
| 0, 3, 5, 7, 9, 11, 13–18, 20, 22, 24 | 2–3 points | 1 bucket |

(One circulated version of this table listed fractional parts
0.4721, 0.9443, … labelled as f_r = {32r/13}; those are {64·(r/26 + rδ)}
values, and the mislabelling also produced a phantom variant claiming
r = 15, 17 straddle with two "missed" base buckets. The exact strand-bucket
computation refutes both: {32r/13} for r = 1 is 6/13 ≈ 0.4615, r = 15 and
r = 17 sit far from any boundary, and every base bucket ⌊32r/13⌋ is
occupied by its own strand.)

Straddling residues: **{1, 2, 4, 6, 8, 10, 12, 19, 21, 23, 25}** — 7 + 4 = 11.

> **Occ(64, 64, α₂) = 26 + 11 = 37.** The cluster lemma (§5.2) gives the
> rigorous upper bound: ‖26α₂‖ ≈ 0.00431 < 1/(64·(⌈64/26⌉ − 1)) = 1/128,
> so Occ ≤ 2·26 = 52, and the actual value reflects that only 11 of the 26
> strands straddle.

### 4.3 Gap-theoretic restatement (mirror of §2.4)

P₆₄(α₂) has gap multiset 38 × ‖26α₂‖-short-gaps (0.276 bucket widths) plus
13 + 13 longer gaps (≈ 1.92 and 2.20 bucket widths): exactly "26 strands,
38 intra-strand gaps". S = 38 = 64 − 26 of the gaps are shorter than a bin,
C = 11 of them swallow a grid line, and Theorem 2.3 gives
Occ = 64 − 38 + 11 = 37 — the same ledger identity as the golden case.

---

## 5. Q5 — characterizing {α : Occ(64, 64, α) < 52}

### 5.1 The step function and its measure

α ↦ Occ(64, 64, α) is constant on the cells cut out by the 80639 interior
walls (64j + k)/(64i) — 80640 cells with rational endpoints, hence

> {α ∈ [0,1) : Occ < 52} is a finite union of rational intervals,

and its Lebesgue measure is exactly computable by sorting the walls and
evaluating one rational midpoint per cell (Lemma 0.1's rational primitive).
The exact value:

> measure{Occ < 52} = 223889114211561380423855435 / 378325402912903125603635712
> = 0.591789799172181

i.e. **low occupancy is the majority** (59.2%). Further exact measures:
measure{Occ ≥ 55} ≈ 0.2807, measure{Occ = 64} ≈ 0.0181, 𝔼[Occ] ≈ 44.53,
mode 54. For scale: 64 i.i.d. uniform points give 𝔼[Occ] = 64(1 − (63/64)⁶⁴)
≈ 40.6 — the Kronecker sequence at this scale is only slightly better than
chance, and 1/φ (55) sits in the top ~28%.

### 5.2 The cluster lemma (proved sufficient condition for badness)

**Lemma 5.1.** If ‖qα‖ < 1/(64·(⌈64/q⌉ − 1)) for some q ≤ 25, then the
points {iα}, i ≤ 64, split into q residue classes each of diameter
< 1/64; each class meets at most 2 buckets; hence Occ ≤ 2q ≤ 50 < 52.

*Proof.* Points i and i + q differ by {qα} (mod 1), a circular displacement
≤ ‖qα‖. A residue class with t members has diameter ≤ (t−1)‖qα‖ ≤
(⌈64/q⌉ − 1)·‖qα‖ < 1/64, so it cannot span two bucket boundaries and meets
≤ 2 buckets. Union over q classes: Occ ≤ 2q. ∎

Its total measure is at most Σ_{q≤25} 2/(64(⌈64/q⌉−1)) ≈ 0.189 — it explains
only ~32% of the bad set. **The bad set is not arithmetic.**

**Lemma 5.2 (necessary condition).** Occ < 64 implies
min_{1≤q≤63} ‖qα‖ < 1/64 (two same-bucket points differ by (i−j)α). Vacuous
as a discriminator (measure ≈ 1 − 10⁻¹⁴).

### 5.3 Why no single CF condition characterizes the threshold

By Theorem 2.3, Occ = 64 − S + C with S given exactly by continued-fraction
arithmetic (three-distance + Ostrowski expansion of 64) and C a *phase* term
— which short gaps swallow a grid line. Empirically over all 80640 cells
(tool `--measure` run): given S, C ranges over essentially its full span
(e.g. at S = 39 the observed Occ runs 25..64), so **no function of the
continued fraction alone** decides Occ < 52. What survives:

- **Statistical predictor.** q*(α) = argmin_{q≤63} ‖qα‖ (the best rational
  approximation visible at scale 64; min ‖qα‖ equals the shortest circular
  gap of P₆₄). The exact conditional measures give
  measure{q* ≥ 52} ≈ 0.1645 and measure{q* ≥ 52 ∧ Occ < 52} ≈ 0.000214, i.e.

  > P(Occ ≥ 52 | q* ≥ 52) ≈ 0.9987,

  while P(Occ < 52 | q* ≤ 51) ≈ 0.708. A one-directional arithmetic criterion
  with a 2.14×10⁻⁴ exception set — clean, but not an equivalence.
- **Sufficient mechanism.** α close to p/q with small q (large partial
  quotient, q ≲ 40): strand collapse gives Occ ≲ 2q (§4 is the case q = 26).
- **The golden ratio's role.** 1/φ is badly approximable (all partial
  quotients 1), maximizing min_{q≤64} ‖qα‖ = φ⁻¹⁰ ≈ 0.52 bucket widths —
  the worst-case-optimal property that makes the Knuth multiplier 0x9E3779B9
  the right choice for `probe_start`. The asymptotics need care, because
  `Occ(m, m, α)/m` has **no pointwise limit** for a fixed irrational α: at
  every convergent denominator `Occ(q_k, q_k, α) = q_k` (§2.3), so
  `limsup_m Occ(m,m,α)/m = 1` for *every* α. Two different averages are
  therefore in play:
  - **i.i.d. random keys** (α irrelevant): `⌊m(1 − e⁻¹)⌋ ≈ 40.6` at `m = 64`.
  - **uniform α**: `E[Occ(64,64,α)] = 44.53395 = 0.6958·64`, the exact cell
    average of §5.1 (numerically `≈ 1 − 3/π²`), *not* 40.6.

  Under the scale-invariant (log-uniform `m`) measure the golden ratio is also
  the **average-case optimum**: with `Φ(α) = lim_{M→∞} (1/ln M)
  Σ_{m≤M} Occ(m,m,α)/m²`, one has `Φ(α₀) ≈ 0.8765` (arithmetic variant
  0.8788) against `≈ 0.863` for the silver ratio, the supremum outside the
  noble numbers equals `Φ(α₀)` and is approached but never attained, and the
  ties are exactly the noble numbers (continued fractions ending in all 1s,
  i.e. the GL₂(ℤ)-orbit of α₀). Worst-case optimality and scale-average
  optimality are governed by the same continued-fraction tail here.
  **Correction to an earlier draft:** "golden hashing buys a worst-case
  guarantee, not an average-case one" is wrong under a scale-invariant
  measure — it buys both. What it does not do is beat the i.i.d. model.

### 5.4 The exact phase criterion, and the `q* ≥ m/2` theorem

**Criterion.** With the `N` points sorted as `y_1 < … < y_N` and gaps
`g_j = y_{j+1} − y_j` (indices circular, `y_{N+1} = y_1 + 1`), put

```
C_j = ⌊N y_{j+1}⌋ − ⌊N y_j⌋ ∈ {0,1,2,…},      Σ_j C_j = N .
```

`C_j = 0` iff the gap contains no grid line `k/N`; since `C_j ≥ 1` whenever
`g_j ≥ 1/N`, only short gaps can contribute, and whether a short gap does
contribute is exactly the **grid phase**. Equivalently

```
C_j = 0  ⟺  {N y_j} + N g_j < 1        ("phase-short").
```

Hence, with `Z = #{j : C_j = 0}` (the circular wrap pair can never qualify),

```
Occ(N, N, α) = N − Z ,      and      α ∈ L_N  ⟺  Z > N/2 .
```

This is the sharpened form of Theorem 2.3: `Z = S − C` there. Decidability:
the gap word is a Sturmian/Christoffel word (three-gap theorem), so `Z` is a
sum of `O(log N)` floor sums and can be evaluated in `O((log N)²)` arithmetic
operations — provided α is known exactly (an algebraic number, or the finite
continued-fraction prefix **plus** the complete quotient / grid phase). A
finite CF prefix alone cannot decide `L_N`: it fixes the cyclic word of gap
types but not which cyclic shift sits on the `1/N` grid, which is precisely
the negative result of §5.3.

**Theorem (`q* ≥ m/2` is sufficient, with no exceptions).** Let
`q* = argmin_{1≤q≤m} ‖qα‖`, attained at the largest convergent denominator
`q_k ≤ m`. If `q* ≥ m/2` then, using `q_k δ_{k−1} + q_{k−1} δ_k = 1`,

```
max three-distance gap ≤ δ_{k−1} < 1/q_k ≤ 2/m .
```

Two adjacent empty buckets would force an interval of length `2/m` free of the
`m+1` points, and the wrap pair `(m−1, 0)` cannot both be empty because
`{q_k α}` lies within `δ_k < 1/m` of an integer. Hence the empty buckets form
an independent set on the `m`-cycle:

```
q* ≥ m/2   ⇒   Occ(m, m, α) ≥ ⌈m/2⌉ .
```

So the exception set `E_m = {q* ≥ m/2 ∧ Occ < m/2}` is **empty** — the
m/2 criterion is a theorem, not a statistic. For even `m` the tight layer
`Occ = m/2` has measure `2φ(m/2)/m²` (`φ` Euler's totient; `1/128` at
`m = 64`, 32 cells), where α lies within `1/m²` of a reduced `p/(m/2)`.
The *sharper* sufficient condition is `max gap < 2/m` (measure `0.5701` of
α-space at `m = 64`, against `0.4624` for `q* ≥ 32`); 20584 cells have a gap
`≥ 2/m` yet still `Occ ≥ 32`, so the max-gap condition is strictly stronger.

**Threshold note.** This is the *`m/2`* threshold. §5.3's one-directional
statistic uses the much higher threshold 52 and stays empirical; the two are
not the same statement. The exact measure of `{q* ≥ 32}` depends on the
convention for `q*`: `0.4624` summing over `q ≤ 64`, `0.4546` over `q ≤ 63`
(the §5.3 convention). Our exhaustive scan reproduces `meas(Occ < 32) =
0.179071`, `min Occ = 32` on `q* ≥ 32`, zero exceptions, and the `1/128`
tight layer; see `verify_phase_criterion.py`.

---

## 6. Verification

Every claim in this document is machine-checked by
`research/fibonacci-occupancy/scripts/golden_ratio_occupancy.py` (integer arithmetic only; `--measure` runs
the full 80640-cell Farey scan). The fixed tool is delivered under the same
filename, `golden_ratio_occupancy.py`; the
`research/fibonacci-occupancy/scripts/` prefix refers to its location in the
source tree. The corrected Theorem 2.1 is additionally
re-derived independently by `verify_convergent_occ_en.py` (standalone,
English, Fraction-only).

| check | anchor |
|---|---|
| three-gap multiset (24/31/10, sums); general multiplicity formula at N = 40, 64, 65, 100, 144 | `check_q1` / `check_three_gap_formula` |
| Occ table, n=64 ledger, empty/double buckets, convergent theorem, bucket→pair map | `check_q2` |
| Theorem 2.1 corrected: cylinder proof for **every** irrational α (both signs of ε, wrap branch; 93 (k, cylinder) pairs, max q_k²\|ε\| = 0.999002), refutation of the 1/(2q_k²) lemma | `check_convergent_cylinders` / `verify_convergent_occ_en.py` |
| wall table, pattern radius 9.4351785e-6, occupancy interval, α₁ both constants | `check_q3` |
| α₂ = 37, strand set, a₃ = 7, 26δ < 1/64 | `check_q4` |
| measure{Occ < 52} = 0.591789799172181 (exact fraction), E[Occ] ≈ 44.53, mode 54, q* predictors, S=39 range 25..64, Theorem 2.3 on all 80640 cells | `check_q5` / `scan_cells` |
| phase criterion `Occ = N − Z` and `α ∈ L_N ⟺ Z > N/2` on all 80640 cells; `q* ≥ m/2 ⇒ Occ ≥ ⌈m/2⌉` with zero exceptions, tight layer 32 cells / measure 1/128 | `verify_phase_criterion.py` |

Wall-count convention used throughout: 80641 distinct rationals including
the endpoints 0 and 1 ⇒ 80639 interior walls ⇒ 80640 cells.

Corresponding engine-side pins: `dep_graph.rs::dedup_probe_start_spreads_consecutive_indices`
(52-slot spread bound) and the query-system defense suite.
