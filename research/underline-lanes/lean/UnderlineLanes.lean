import Mathlib

-- The finite `decide` witnesses below quantify over function spaces
-- (`Fin 6 → Fin 3`), whose instance search recurses far deeper than the
-- default limit.  Set the limit once for the whole file instead of
-- restating it on every example (and keep it *above* the doc comments, so a
-- doc comment is always followed by a declaration keyword).
set_option maxRecDepth 10000

/-!
# Underline lanes: witnesses and general lemmas (Tier 0 + Tier 1)

Companion to `research/underline-lanes/no-distributive-lattice.md` (§12).  Every
`example ... := by decide` below is checked by the Lean kernel; the general
lemmas of §1 are hand proofs.

## Conventions

* Colours are `0`-based (`Fin k`); the note's `[k] = {1..k}` is `Fin k`.
* A graph on `Fin n` is an edge set `E : Finset (Fin n × Fin n)`; loops appear
  only inside the exhaustive minimality quantification, where they discharge
  immediately (a loop makes every proper colouring impossible).
* `Pleq c d` is the pointwise order; `Pmin`/`Pmax` the meet/join candidates;
  `Nrm E K` is the note's `C(G, K)`: proper `k`-colourings normalised on the
  clique `K` (`c (K i) = i`).
* Every proposition that is ever `decide`d is built from `∧ / ∨ / ¬`,
  Fintype quantifiers and Finset membership/cardinality only — no bare
  implications — so decidability needs nothing beyond
  `Fintype.decidableForallFintype` / `Fintype.decidableExistsFintype`.
  `decidableImp` below is a low-priority safety net for the residual
  implication inside elaborated `∀ x ∈ s, ...` forms.
-/
/-- Low-priority safety net: decidability of implication, so `decide` can see
    through the implications elaborated from `∀ x ∈ s, ...` notation.  Being
    low-priority, it never shadows a library instance if one exists. -/
local instance (priority := low) decidableImp {p q : Prop} [dp : Decidable p]
    [dq : Decidable q] : Decidable (p → q) :=
  match dp, dq with
  | isFalse hp, _ => isTrue (fun h => absurd h hp)
  | isTrue _, isTrue hq => isTrue (fun _ => hq)
  | isTrue hp, isFalse hq => isFalse (fun h => hq (h hp))

/-! ## §0 — generic definitions -/

/-- Pointwise order on colourings. -/
def Pleq {n k : ℕ} (a b : Fin n → Fin k) : Prop := ∀ i, a i ≤ b i

instance decPleq {n k : ℕ} (a b : Fin n → Fin k) : Decidable (Pleq a b) :=
  inferInstanceAs (Decidable (∀ i : Fin n, a i ≤ b i))

/-- Pointwise meet candidate. -/
def Pmin {n k : ℕ} (a b : Fin n → Fin k) : Fin n → Fin k := fun i => min (a i) (b i)

/-- Pointwise join candidate. -/
def Pmax {n k : ℕ} (a b : Fin n → Fin k) : Fin n → Fin k := fun i => max (a i) (b i)

/-- Properness (`∨`-form of `∀ e ∈ E, c e.1 ≠ c e.2`). -/
def IsProper {n k : ℕ} (E : Finset (Fin n × Fin n)) (c : Fin n → Fin k) : Prop :=
  ∀ e : Fin n × Fin n, e ∉ E ∨ c e.1 ≠ c e.2

instance decIsProper {n k : ℕ} (E : Finset (Fin n × Fin n)) (c : Fin n → Fin k) :
    Decidable (IsProper E c) :=
  inferInstanceAs (Decidable (∀ e : Fin n × Fin n, e ∉ E ∨ c e.1 ≠ c e.2))

/-- `C(G, K)` (§1 of the note): all proper `k`-colourings with `c (K i) = i`. -/
def Nrm {n k : ℕ} (E : Finset (Fin n × Fin n)) (K : Fin k → Fin n) :
    Finset (Fin n → Fin k) :=
  Finset.univ.filter fun c => IsProper E c ∧ ∀ i : Fin k, c (K i) = i

/-- `m` is the meet of `a` and `b` inside `C` (§4). -/
def IsMeetIn {n k : ℕ} (C : Finset (Fin n → Fin k)) (m a b : Fin n → Fin k) : Prop :=
  m ∈ C ∧ Pleq m a ∧ Pleq m b ∧
    ∀ z ∈ C.filter (fun z => Pleq z a ∧ Pleq z b), Pleq z m

instance decIsMeetIn {n k : ℕ} (C : Finset (Fin n → Fin k)) (m a b : Fin n → Fin k) :
    Decidable (IsMeetIn C m a b) :=
  inferInstanceAs (Decidable (m ∈ C ∧ Pleq m a ∧ Pleq m b ∧
    ∀ z ∈ C.filter (fun z => Pleq z a ∧ Pleq z b), Pleq z m))

/-- Dual: `j` is the join of `a` and `b` inside `C`. -/
def IsJoinIn {n k : ℕ} (C : Finset (Fin n → Fin k)) (j a b : Fin n → Fin k) : Prop :=
  j ∈ C ∧ Pleq a j ∧ Pleq b j ∧
    ∀ z ∈ C.filter (fun z => Pleq a z ∧ Pleq b z), Pleq j z

instance decIsJoinIn {n k : ℕ} (C : Finset (Fin n → Fin k)) (j a b : Fin n → Fin k) :
    Decidable (IsJoinIn C j a b) :=
  inferInstanceAs (Decidable (j ∈ C ∧ Pleq a j ∧ Pleq b j ∧
    ∀ z ∈ C.filter (fun z => Pleq a z ∧ Pleq b z), Pleq j z))

/-- Refinement of colour partitions (§§3–5); `∨`-form of
    `(c v = c v' → d v = d v')`. -/
def Refines {n k : ℕ} (c d : Fin n → Fin k) : Prop :=
  ∀ v v' : Fin n, c v ≠ c v' ∨ d v = d v'

instance decRefines {n k : ℕ} (c d : Fin n → Fin k) : Decidable (Refines c d) :=
  inferInstanceAs (Decidable (∀ v v' : Fin n, c v ≠ c v' ∨ d v = d v'))

theorem refines_iff {n k : ℕ} {c d : Fin n → Fin k} :
    Refines c d ↔ (∀ v v', c v = c v' → d v = d v') := by
  constructor
  · intro h v v' hvv'
    rcases h v v' with h1 | h2
    · exact absurd hvv' h1
    · exact h2
  · intro h v v'
    by_cases hv : c v = c v'
    · exact Or.inr (h v v' hv)
    · exact Or.inl hv

/-- Undirected adjacency. -/
def Adj {n : ℕ} (E : Finset (Fin n × Fin n)) (a b : Fin n) : Prop :=
  (a, b) ∈ E ∨ (b, a) ∈ E

instance decAdj {n : ℕ} (E : Finset (Fin n × Fin n)) (a b : Fin n) : Decidable (Adj E a b) :=
  inferInstanceAs (Decidable ((a, b) ∈ E ∨ (b, a) ∈ E))

/-- One hop of a walk (stay put or cross an edge). -/
def Step {n : ℕ} (E : Finset (Fin n × Fin n)) (a b : Fin n) : Prop :=
  a = b ∨ Adj E a b

instance decStep {n : ℕ} (E : Finset (Fin n × Fin n)) (a b : Fin n) : Decidable (Step E a b) :=
  inferInstanceAs (Decidable (a = b ∨ Adj E a b))

/-- Walk connectivity for graphs on ≤ 5 vertices.

`p : Fin 4 → Fin n` supplies four intermediates, so the definition allows a path
of five edges (`v → p 0 → p 1 → p 2 → p 3 → w`).  That is still a bound, not an
exact statement: a connected `n`-vertex graph has a path with at most `n - 1`
edges, so five steps cover every `n ≤ 6`. -/
def Conn5 {n : ℕ} (E : Finset (Fin n × Fin n)) : Prop :=
  ∀ v w : Fin n, ∃ p : Fin 4 → Fin n, Step E v (p 0) ∧ Step E (p 0) (p 1) ∧
    Step E (p 1) (p 2) ∧ Step E (p 2) (p 3) ∧ Step E (p 3) w

instance decConn5 {n : ℕ} (E : Finset (Fin n × Fin n)) : Decidable (Conn5 E) :=
  inferInstanceAs (Decidable (∀ v w : Fin n, ∃ p : Fin 4 → Fin n,
    Step E v (p 0) ∧ Step E (p 0) (p 1) ∧ Step E (p 1) (p 2) ∧
      Step E (p 2) (p 3) ∧ Step E (p 3) w))

/-- Walk connectivity for graphs on `Fin 4` (three hops suffice). -/
def Conn4 {n : ℕ} (E : Finset (Fin n × Fin n)) : Prop :=
  ∀ v w : Fin n, ∃ p : Fin 3 → Fin n, Step E v (p 0) ∧ Step E (p 0) (p 1) ∧
    Step E (p 1) (p 2) ∧ Step E (p 2) w

instance decConn4 {n : ℕ} (E : Finset (Fin n × Fin n)) : Decidable (Conn4 E) :=
  inferInstanceAs (Decidable (∀ v w : Fin n, ∃ p : Fin 3 → Fin n,
    Step E v (p 0) ∧ Step E (p 0) (p 1) ∧ Step E (p 1) (p 2) ∧ Step E (p 2) w))

/-- Walk connectivity for graphs on `Fin 3` (two hops suffice). -/
def Conn3 {n : ℕ} (E : Finset (Fin n × Fin n)) : Prop :=
  ∀ v w : Fin n, ∃ p : Fin 2 → Fin n, Step E v (p 0) ∧ Step E (p 0) (p 1) ∧
    Step E (p 1) w

instance decConn3 {n : ℕ} (E : Finset (Fin n × Fin n)) : Decidable (Conn3 E) :=
  inferInstanceAs (Decidable (∀ v w : Fin n, ∃ p : Fin 2 → Fin n,
    Step E v (p 0) ∧ Step E (p 0) (p 1) ∧ Step E (p 1) w))

/-- `K` is a clique (`∨`-form of `i ≠ j → Adj (K i) (K j)`). -/
def IsCliqueFn {n m : ℕ} (E : Finset (Fin n × Fin n)) (K : Fin m → Fin n) : Prop :=
  ∀ i j : Fin m, i = j ∨ Adj E (K i) (K j)

instance decIsCliqueFn {n m : ℕ} (E : Finset (Fin n × Fin n)) (K : Fin m → Fin n) :
    Decidable (IsCliqueFn E K) :=
  inferInstanceAs (Decidable (∀ i j : Fin m, i = j ∨ Adj E (K i) (K j)))

/-- Existence of a proper `k`-colouring (`χ ≤ k`). -/
def HasCol {n : ℕ} (E : Finset (Fin n × Fin n)) (k : ℕ) : Prop :=
  ∃ c : Fin n → Fin k, IsProper E c

instance decHasCol {n : ℕ} (E : Finset (Fin n × Fin n)) (k : ℕ) : Decidable (HasCol E k) :=
  inferInstanceAs (Decidable (∃ c : Fin n → Fin k, IsProper E c))

/-- Pointwise chain: every two elements comparable. -/
def ChainPt {n k : ℕ} (S : Finset (Fin n → Fin k)) : Prop :=
  ∀ a b : Fin n → Fin k, a ∉ S ∨ b ∉ S ∨ Pleq a b ∨ Pleq b a

instance decChainPt {n k : ℕ} (S : Finset (Fin n → Fin k)) : Decidable (ChainPt S) :=
  inferInstanceAs (Decidable (∀ a b : Fin n → Fin k, a ∉ S ∨ b ∉ S ∨ Pleq a b ∨ Pleq b a))

/-! ## §1 — general lemmas (Tier 1) -/

/-- Pigeonhole for iterates: in a finite type two iterates of a point collide. -/
theorem exists_iterate_repeat {X : Type*} [Fintype X] (f : X → X) (y : X) :
    ∃ i j : ℕ, i < j ∧ f^[i] y = f^[j] y := by
  by_contra h
  push Not at h
  have hinj : Function.Injective (fun n : Fin (Fintype.card X + 1) => f^[n.val] y) := by
    intro a b hab
    by_contra hne
    have hav : a.val ≠ b.val := fun hc => hne (Fin.eq_of_val_eq hc)
    rcases lt_or_gt_of_ne hav with hlt | hgt
    · exact h a.val b.val hlt hab
    · exact h b.val a.val hgt hab.symm
  have hcard :=
    Fintype.card_le_of_injective (fun n : Fin (Fintype.card X + 1) => f^[n.val] y) hinj
  rw [Fintype.card_fin] at hcard
  omega

/-- A surjective endomap of `Fin k` is injective. -/
theorem surj_inj_fin {k : ℕ} {f : Fin k → Fin k} (hs : Function.Surjective f) :
    Function.Injective f := by
  classical
  -- Pick a right inverse `g` (`f ∘ g = id`).  Then `g` is injective; an
  -- injective self-map of a finite type is surjective, which makes `g` a left
  -- inverse of `f` as well, and injectivity of `f` follows.
  choose g hg using hs
  have hg_inj : Function.Injective g := by
    intro a b hab
    have h : f (g a) = f (g b) := by rw [hab]
    rwa [hg a, hg b] at h
  have hg_surj : Function.Surjective g := Finite.surjective_of_injective hg_inj
  have hleft : ∀ a, g (f a) = a := by
    intro a
    obtain ⟨y, hy⟩ := hg_surj a
    rw [← hy, hg y]
  intro a b hab
  have h : g (f a) = g (f b) := by rw [hab]
  rwa [hleft a, hleft b] at h

/-- §3, "k-block antichain" lemma.  Two colourings that both use every colour
    are comparable under refinement exactly when they are relabellings of one
    another. -/
theorem refines_iff_relabelling {n k : ℕ} {c d : Fin n → Fin k}
    (hc : ∀ j, ∃ v, c v = j) (hd : ∀ j, ∃ v, d v = j) :
    (∀ v v', c v = c v' → d v = d v') ↔ ∃ π : Fin k ≃ Fin k, ∀ v, d v = π (c v) := by
  classical
  constructor
  · intro href
    choose g hg using hc
    have hfc : ∀ v, d (g (c v)) = d v := fun v => href (g (c v)) v (hg (c v))
    have hsurj : Function.Surjective (fun i : Fin k => d (g i)) := by
      intro j
      obtain ⟨v, hv⟩ := hd j
      refine ⟨c v, ?_⟩
      show d (g (c v)) = j
      rw [hfc, hv]
    have hinj := surj_inj_fin hsurj
    refine ⟨Equiv.ofBijective (fun i : Fin k => d (g i)) ⟨hinj, hsurj⟩, fun v => ?_⟩
    show d v = d (g (c v))
    rw [← hfc v]
  · rintro ⟨π, hπ⟩ v v' hvv'
    rw [hπ v, hπ v', hvv']

/-- §3, corollary: relabelling-distinct full-`k` colourings are incomparable
    under refinement — the optimal-colouring set is an antichain. -/
theorem refines_antisymm {n k : ℕ} {c d : Fin n → Fin k}
    (hc : ∀ j, ∃ v, c v = j) (hd : ∀ j, ∃ v, d v = j)
    (hiff : ¬ ∃ π : Fin k ≃ Fin k, ∀ v, d v = π (c v)) :
    ¬ (∀ v v', c v = c v' → d v = d v') ∧ ¬ (∀ v v', d v = d v' → c v = c v') := by
  refine ⟨fun href => hiff ((refines_iff_relabelling hc hd).mp href), fun href => ?_⟩
  obtain ⟨π, hπ⟩ := (refines_iff_relabelling hd hc).mp href
  exact hiff ⟨π.symm, fun v => by rw [hπ v, Equiv.symm_apply_apply]⟩

/-- §6(d), transitivity lemma: an order on a finite type invariant under a
    transitive group of permutations is trivial. -/
theorem invariant_order_trivial {X G : Type*} [Fintype X] [PartialOrder X] [Group G]
    [MulAction G X] (hmono : ∀ (g : G) (x y : X), x ≤ y → (g • x) ≤ (g • y))
    (htrans : ∀ x y : X, ∃ g : G, g • x = y) {x y : X} (hlt : x < y) : False := by
  obtain ⟨g, hg⟩ := htrans x y
  have hinv : ∀ z : X, (g⁻¹ : G) • (g • z) = z := by
    intro z; rw [← mul_smul, inv_mul_cancel, one_smul]
  have hbij : Function.Bijective (fun z : X => g • z) := by
    refine ⟨?_, ?_⟩
    · intro a b hab
      have h := congrArg (fun w : X => (g⁻¹ : G) • w) hab
      rw [hinv, hinv] at h
      exact h
    · intro z
      refine ⟨g⁻¹ • z, ?_⟩
      show g • (g⁻¹ • z) = z
      rw [← mul_smul, mul_inv_cancel, one_smul]
  have h1 : y ≤ g • y := by
    have h2 := hmono g x y hlt.le
    rw [hg] at h2
    exact h2
  have hchain : ∀ n : ℕ, (fun z : X => g • z)^[n] y ≤ (fun z : X => g • z)^[n + 1] y := by
    intro n
    induction n with
    | zero => exact h1
    | succ m ih =>
      -- `f^[m+1] y ≤ f^[m+1+1] y`, i.e. `g • f^[m] y ≤ g • f^[m+1] y`; the
      -- iterate unfolding is left to `simp only` (the `^` notation does not
      -- beta-reduce definitionally here).
      simpa only [Function.iterate_succ_apply'] using hmono g _ _ ih
  have hle : ∀ m n : ℕ, n ≤ m →
      (fun z : X => g • z)^[n] y ≤ (fun z : X => g • z)^[m] y := by
    intro m
    induction m with
    | zero =>
      intro n hn
      have hn0 : n = 0 := le_antisymm hn (Nat.zero_le n)
      subst hn0; exact le_refl _
    | succ m ih =>
      intro n hn
      rcases eq_or_lt_of_le hn with hnm | hlt2
      · subst hnm; exact le_refl _
      · exact le_trans (ih n (by omega)) (hchain m)
  obtain ⟨i, j, hij, hrep⟩ := exists_iterate_repeat (fun z : X => g • z) y
  have hsq : (fun z : X => g • z)^[i] y = (fun z : X => g • z)^[i + 1] y := by
    refine le_antisymm (hchain i) ?_
    have h1' := hle j (i + 1) (by omega)
    rw [← hrep] at h1'
    exact h1'
  have hiterinj : ∀ n : ℕ, Function.Injective ((fun z : X => g • z)^[n]) := by
    intro n
    induction n with
    | zero => exact fun a b h => h
    | succ m ih =>
      intro a b hab
      have hab' : (fun z : X => g • z) ((fun z : X => g • z)^[m] a)
          = (fun z : X => g • z) ((fun z : X => g • z)^[m] b) := by
        simpa only [Function.iterate_succ_apply'] using hab
      exact ih (hbij.1 hab')
  have h3 : y = (fun z : X => g • z) y := by
    apply hiterinj i
    exact hsq.trans (Function.iterate_succ_apply (fun z : X => g • z) i y)
  exact hlt.ne (hbij.1 (hg.trans h3))

/-- §7: pointwise `min`/`max` of two maps strictly increasing along every arc
    again strictly increase along every arc — the closure behind
    "`C_D` is a distributive lattice". -/
theorem orient_mono_minmax {n k : ℕ} (D : Finset (Fin n × Fin n)) (f g : Fin n → Fin k)
    (hf : ∀ e : Fin n × Fin n, e ∉ D ∨ f e.1 < f e.2)
    (hg : ∀ e : Fin n × Fin n, e ∉ D ∨ g e.1 < g e.2) :
    (∀ e : Fin n × Fin n, e ∉ D ∨ min (f e.1) (g e.1) < min (f e.2) (g e.2)) ∧
    (∀ e : Fin n × Fin n, e ∉ D ∨ max (f e.1) (g e.1) < max (f e.2) (g e.2)) := by
  constructor
  · intro e
    rcases hf e with h1 | h1
    · exact Or.inl h1
    · rcases hg e with h2 | h2
      · exact Or.inl h2
      · exact Or.inr (min_lt_min h1 h2)
  · intro e
    rcases hf e with h1 | h1
    · exact Or.inl h1
    · rcases hg e with h2 | h2
      · exact Or.inl h2
      · -- Split the `max` bound into its two components instead of rewriting
        -- with `max_eq_*` (whose argument order differs between Mathlib
        -- versions); `max_lt` is unambiguous.
        refine Or.inr ?_
        exact max_lt (lt_of_lt_of_le h1 (le_max_left _ _))
          (lt_of_lt_of_le h2 (le_max_right _ _))

/-- `C_D` (§7): strict order-preserving maps with prescribed values on `K`. -/
def CDset {n k : ℕ} (D : Finset (Fin n × Fin n)) (K : Fin k → Fin n) :
    Finset (Fin n → Fin k) :=
  Finset.univ.filter fun c =>
    (∀ e : Fin n × Fin n, e ∉ D ∨ c e.1 < c e.2) ∧ ∀ i : Fin k, c (K i) = i

/-- §7: `C_D` is closed under pointwise `min`/`max` — a sublattice of `[k]^V`.
    Acyclicity of `D` is not needed for closure. -/
theorem CDset_closed {n k : ℕ} (D : Finset (Fin n × Fin n)) (K : Fin k → Fin n)
    (f g : Fin n → Fin k) (hf : f ∈ CDset D K) (hg : g ∈ CDset D K) :
    Pmin f g ∈ CDset D K ∧ Pmax f g ∈ CDset D K := by
  obtain ⟨hf1, hf2⟩ := (Finset.mem_filter.mp hf).2
  obtain ⟨hg1, hg2⟩ := (Finset.mem_filter.mp hg).2
  obtain ⟨hm, hj⟩ := orient_mono_minmax D f g hf1 hg1
  constructor
  · simp only [CDset, Finset.mem_filter, Finset.mem_univ, true_and]
    refine ⟨fun e => ?_, fun i => ?_⟩
    · exact hm e
    · show min (f (K i)) (g (K i)) = i
      rw [hf2 i, hg2 i, min_self]
  · simp only [CDset, Finset.mem_filter, Finset.mem_univ, true_and]
    refine ⟨fun e => ?_, fun i => ?_⟩
    · exact hj e
    · show max (f (K i)) (g (K i)) = i
      rw [hf2 i, hg2 i, max_self]

/-- The orientation `D_c`: every edge of `E`, from the smaller to the larger
    colour. -/
def DcOf {n k : ℕ} (E : Finset (Fin n × Fin n)) (c : Fin n → Fin k) :
    Finset (Fin n × Fin n) :=
  E.biUnion fun e => if c e.1 < c e.2 then {(e.1, e.2)} else {(e.2, e.1)}

/-- §7, first half of `C = ⊔_D C_D`: a proper colouring lies in `C_{D_c}`.
    (At most one `D` once `D` is required to orient every edge — immediate,
    left as the remark; disjointness of the pieces follows.) -/
theorem mem_CDset_DcOf {n k : ℕ} (E : Finset (Fin n × Fin n)) (K : Fin k → Fin n)
    (c : Fin n → Fin k) (hc : IsProper E c) (hK : ∀ i : Fin k, c (K i) = i) :
    c ∈ CDset (DcOf E c) K := by
  simp only [CDset, Finset.mem_filter, Finset.mem_univ, true_and]
  refine ⟨fun d => ?_, hK⟩
  by_cases hd : d ∈ DcOf E c
  · simp only [DcOf, Finset.mem_biUnion] at hd
    obtain ⟨e, he, hde⟩ := hd
    rcases lt_trichotomy (c e.1) (c e.2) with hlt | heq | hgt
    · rw [if_pos hlt] at hde
      rw [Finset.mem_singleton.mp hde]
      exact Or.inr hlt
    · -- `e ∈ E` forbids equal colours on an edge — contradiction.
      rcases hc e with h0 | hne
      · exact absurd he h0
      · exact absurd heq hne
    · rw [if_neg (not_lt.mpr (le_of_lt hgt))] at hde
      rw [Finset.mem_singleton.mp hde]
      exact Or.inr hgt
  · exact Or.inl hd

/-! ## §2 — the witnesses, kernel-checked by `decide` -/

/-! ### §4 — Witness A: two triangles sharing one vertex -/

/-- `v₁ = 0, v₂ = 1, v₃ = 2, u = 3, v = 4`. -/
def EA : Finset (Fin 5 × Fin 5) := {(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)}
def CA : Finset (Fin 5 → Fin 3) := Nrm EA ![0, 1, 2]
def cA : Fin 5 → Fin 3 := ![0, 1, 2, 0, 1]
def dA : Fin 5 → Fin 3 := ![0, 1, 2, 1, 0]

/-- Completeness: `C` is exactly the two-point set. -/
example : CA = {cA, dA} := by decide
/-- Pointwise antichain. -/
example : ¬ Pleq cA dA ∧ ¬ Pleq dA cA := by decide
/-- `min`/`max` leave `C` (each violates `uv`). -/
example : ¬ IsProper EA (Pmin cA dA) ∧ ¬ IsProper EA (Pmax cA dA) := by decide
/-- No meet, no join: not a lattice. -/
example : ¬ ∃ m : Fin 5 → Fin 3, IsMeetIn CA m cA dA := by decide
example : ¬ ∃ j : Fin 5 → Fin 3, IsJoinIn CA j cA dA := by decide

/-- Refinement antichain, pinned directly… -/
example : ¬ Refines cA dA ∧ ¬ Refines dA cA := by decide

/-- …and derived from the Tier-1 machinery: A's two colourings use all three
    colours and are not relabellings of one another. -/
example : ¬ Refines cA dA := by
  have h1 : ∀ j : Fin 3, ∃ v, cA v = j := by decide
  have h2 : ∀ j : Fin 3, ∃ v, dA v = j := by decide
  have h3 : ¬ ∃ π : Fin 3 ≃ Fin 3, ∀ v, dA v = π (cA v) := by decide
  intro h
  exact (refines_antisymm h1 h2 h3).1 (refines_iff.mp h)

/-- §6(e): the failure is robust under the other maximum clique
    `{u, v, v₃}`, normalised `u ↦ 0, v ↦ 1, v₃ ↦ 2`. -/
def CA2 : Finset (Fin 5 → Fin 3) := Nrm EA ![3, 4, 2]
def cA2 : Fin 5 → Fin 3 := ![1, 0, 2, 0, 1]

example : CA2 = {cA, cA2} := by decide
example : ¬ Pleq cA cA2 ∧ ¬ Pleq cA2 cA := by decide

/-- Witness A is connected. -/
example : Conn5 EA := by decide

/-! ### §5 — Witness B (the paw) -/

/-- `a = 0, b = 1, c = 2, x = 3`; `x` is the pendant, adjacent only to `a`. -/
def EB : Finset (Fin 4 × Fin 4) := {(0, 1), (0, 2), (1, 2), (0, 3)}
def CB : Finset (Fin 4 → Fin 3) := Nrm EB ![0, 1, 2]
def cB1 : Fin 4 → Fin 3 := ![0, 1, 2, 1]
def cB2 : Fin 4 → Fin 3 := ![0, 1, 2, 2]

/-- Completeness. -/
example : CB = {cB1, cB2} := by decide

/-- §5, **corrected attribution**: under the pointwise order `C` is a
    two-element chain, hence a distributive lattice — the note's claim that B
    fails under *both* orders is wrong; B only kills refinement. -/
example : Pleq cB1 cB2 ∧ cB1 ∈ CB ∧ cB2 ∈ CB ∧ CB.card = 2 := by decide

/-- Meet and join exist in `C` (unlike witness A). -/
example : IsMeetIn CB cB1 cB1 cB2 ∧ IsJoinIn CB cB2 cB1 cB2 := by decide

/-- …while refinement is a two-element antichain: the two orders genuinely
    diverge (§3). -/
example : ¬ Refines cB1 cB2 ∧ ¬ Refines cB2 cB1 := by decide

/-- B's maximum clique is unique: no triangle contains the pendant `x = 3`
    (its only neighbour is `a = 0`). -/
example : ¬ ∃ s : Fin 4 → Bool,
    (Finset.univ.filter (fun x => s x = true)).card = 3 ∧ s 3 = true ∧
      ∀ a b : Fin 4, ¬ (s a = true ∧ s b = true) ∨ a = b ∨ Adj EB a b := by
  decide

example : Conn5 EB := by decide

/-! ### §6 — Witness C (`K₃ ⊔ K₃`): the symmetry obstruction -/

/-- Two disjoint triangles: `A = {0,1,2}` (normalised), `B = {3,4,5}`.  Any
    cross edge would kill one of the six colourings, so this is the minimal
    graph with the note's §6 structure. -/
def EC : Finset (Fin 6 × Fin 6) := {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}

/-- The six permutations of `{0,1,2}`, as functions. -/
def Perms3 : Finset (Fin 3 → Fin 3) :=
  {![0, 1, 2], ![0, 2, 1], ![1, 0, 2], ![1, 2, 0], ![2, 0, 1], ![2, 1, 0]}

/-- The colouring indexed by `p`: `A` fixed, `B` coloured `p`. -/
def cC : (Fin 3 → Fin 3) → (Fin 6 → Fin 3) := fun p => ![0, 1, 2, p 0, p 1, p 2]

def CC : Finset (Fin 6 → Fin 3) := Perms3.image cC

/-- `Perms3` really is `S₃`: it holds six bijective functions. -/
example : Perms3.card = 6 ∧
    ∀ p ∈ Perms3,
      (∀ a b : Fin 3, p a = p b → a = b) ∧ (∀ j : Fin 3, ∃ i : Fin 3, p i = j) := by
  decide

set_option maxRecDepth 8000 in
/-- Completeness: `C` is exactly the normalised optimal set. -/
example : CC = Nrm EC ![0, 1, 2] := by decide
set_option maxRecDepth 8000 in
example : CC.card = 6 := by decide

set_option maxRecDepth 8000 in
set_option maxHeartbeats 1000000 in
/-- §6(a): a six-element pointwise antichain. -/
example : ∀ a b : Fin 6 → Fin 3,
    a ∉ CC ∨ b ∉ CC ∨ a = b ∨ (¬ Pleq a b ∧ ¬ Pleq b a) := by decide

set_option maxRecDepth 8000 in
set_option maxHeartbeats 1000000 in
/-- §6(a): the pointwise `min`/`max` of two distinct elements is never proper
    (the min of two distinct permutations repeats a colour on `B`). -/
example : ∀ a b : Fin 6 → Fin 3,
    a ∉ CC ∨ b ∉ CC ∨ a = b ∨
      (¬ IsProper EC (Pmin a b) ∧ ¬ IsProper EC (Pmax a b)) := by decide

set_option maxRecDepth 8000 in
/-- §6(d): the action is transitive — the colour permutations alone
    (`ρ = id`, the note's `σ = q ∘ p⁻¹`) already give one orbit. -/
example : ∀ p : Fin 3 → Fin 3, p ∉ Perms3 ∨
    Finset.univ.image (fun σ : Equiv.Perm (Fin 3) =>
      ![0, 1, 2, σ (p 0), σ (p 1), σ (p 2)]) = CC := by decide

set_option maxRecDepth 8000 in
/-- §6(d): the only conjugation-fixed permutation is `e`. -/
example : Perms3.filter (fun p => ∀ σ : Equiv.Perm (Fin 3), ∀ i : Fin 3,
    σ (p (σ.symm i)) = p i) = {(![0, 1, 2] : Fin 3 → Fin 3)} := by decide

/-- §6: all six have block type `(2,2,2)` — refinement is trivial on `C`. -/
example : ∀ a b : Fin 6 → Fin 3,
    a ∉ CC ∨ b ∉ CC ∨ a = b ∨ (¬ Refines a b ∧ ¬ Refines b a) := by decide

/-- `K₃ ⊔ K₃` is disconnected; this does not touch §10, whose claim is about
    connected graphs. -/
example : ¬ Conn5 EC := by decide

/-! ### §10 remark — dropping connectedness: `K₃ ⊔ K₂` -/

/-- `K₃ = {0,1,2}`, `K₂ = {3,4}`; `χ = ω = 3`. -/
def ED : Finset (Fin 5 × Fin 5) := {(0, 1), (0, 2), (1, 2), (3, 4)}
def CD : Finset (Fin 5 → Fin 3) := Nrm ED ![0, 1, 2]
def cD : Fin 5 → Fin 3 := ![0, 1, 2, 0, 1]
def dD : Fin 5 → Fin 3 := ![0, 1, 2, 1, 0]

example : cD ∈ CD ∧ dD ∈ CD ∧ CD.card = 6 := by decide
example : ¬ Pleq cD dD ∧ ¬ Pleq dD cD := by decide
example : ¬ IsProper ED (Pmin cD dD) ∧ ¬ IsProper ED (Pmax cD dD) := by decide
/-- The only pointwise lower bound of the pair repeats a colour on the `K₂`
    — no meet in `C`; dually no join. -/
example : ¬ ∃ m : Fin 5 → Fin 3, IsMeetIn CD m cD dD := by decide
example : ¬ ∃ j : Fin 5 → Fin 3, IsJoinIn CD j cD dD := by decide

/-! ## §3 — minimality (§10): no pointwise failure on ≤ 4 connected vertices -/

set_option maxRecDepth 8000 in
/-- §10, n = 3.  Hypotheses folded into the decided disjunction; `χ = k+1`
    makes every `(k+1)`-clique automatically maximum, so no ω-hypothesis is
    needed; looped graphs discharge at `¬ HasCol`. -/
example :
    ∀ E : Finset (Fin 3 × Fin 3),
      ¬ Conn3 E ∨
        ∀ k : Fin 3,
          ¬ HasCol E (k.val + 1) ∨ HasCol E k.val ∨
            ∀ K : Fin (k.val + 1) → Fin 3,
              ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  decide

-- The combined n=4 `decide` exceeded the machine's memory (global OOM at
-- ~4.2 GiB anon-rss even with swap).  Split the `∀ k : Fin 4` into four
-- closed per-`k` goals: the `∀ E` quantifier and the final statement are
-- unchanged, no `native_decide`/`ofReduceBool` is introduced, and each
-- kernel reduction has a smaller peak working set.
set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
private lemma n4_case_0 :
    ∀ E : Finset (Fin 4 × Fin 4),
      (∃ x : Fin 4, (x, x) ∈ E) ∨ ¬ Conn4 E ∨
        ¬ HasCol E 1 ∨ HasCol E 0 ∨
          ∀ K : Fin 1 → Fin 4, ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  decide

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
private lemma n4_case_1 :
    ∀ E : Finset (Fin 4 × Fin 4),
      (∃ x : Fin 4, (x, x) ∈ E) ∨ ¬ Conn4 E ∨
        ¬ HasCol E 2 ∨ HasCol E 1 ∨
          ∀ K : Fin 2 → Fin 4, ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  decide

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
private lemma n4_case_2 :
    ∀ E : Finset (Fin 4 × Fin 4),
      (∃ x : Fin 4, (x, x) ∈ E) ∨ ¬ Conn4 E ∨
        ¬ HasCol E 3 ∨ HasCol E 2 ∨
          ∀ K : Fin 3 → Fin 4, ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  decide

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
private lemma n4_case_3 :
    ∀ E : Finset (Fin 4 × Fin 4),
      (∃ x : Fin 4, (x, x) ∈ E) ∨ ¬ Conn4 E ∨
        ¬ HasCol E 4 ∨ HasCol E 3 ∨
          ∀ K : Fin 4 → Fin 4, ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  decide

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
/-- §10, n = 4.  First disjunct short-circuits looped graphs cheaply.  The
    χ = ω hypotheses (`HasCol (k+1)`, `¬ HasCol k`) are evaluated once per `k`,
    outside the clique loop.  Each `k` is discharged by its own `n4_case_*`
    kernel-checked lemma (the combined `decide` OOMed); the statement and all
    quantifiers are unchanged. -/
example :
    ∀ E : Finset (Fin 4 × Fin 4),
      (∃ x : Fin 4, (x, x) ∈ E) ∨ ¬ Conn4 E ∨
        ∀ k : Fin 4,
          ¬ HasCol E (k.val + 1) ∨ HasCol E k.val ∨
            ∀ K : Fin (k.val + 1) → Fin 4,
              ¬ IsCliqueFn E K ∨ ChainPt (Nrm E K) := by
  intro E
  by_cases hloop : ∃ x : Fin 4, (x, x) ∈ E
  · exact Or.inl hloop
  · right
    by_cases hconn : Conn4 E
    · exact Or.inl hconn
    · right
      intro k
      fin_cases k
      · rcases n4_case_0 E with h | h | h
        · exact absurd h hloop
        · exact absurd h hconn
        · exact h
      · rcases n4_case_1 E with h | h | h
        · exact absurd h hloop
        · exact absurd h hconn
        · exact h
      · rcases n4_case_2 E with h | h | h
        · exact absurd h hloop
        · exact absurd h hconn
        · exact h
      · rcases n4_case_3 E with h | h | h
        · exact absurd h hloop
        · exact absurd h hconn
        · exact h

/-! ## Correspondence with the note

| Note                                              | This file |
| ------------------------------------------------- | --------- |
| §3 "k-block antichain" lemma                      | `refines_iff_relabelling`, `refines_antisymm` |
| §3 corrected attribution (B is a pointwise chain) | `EB`/`CB` claims |
| §4 witness A                                      | `EA`, `CA`, `CA2` claims |
| §5 witness B                                      | `EB`, `CB` claims |
| §6 witness C, transitivity, conjugation           | `EC`, `CC`, `invariant_order_trivial`, claims |
| §7 `C_D` closure; `C = ⊔_D C_D`                   | `orient_mono_minmax`, `CDset_closed`, `DcOf`, `mem_CDset_DcOf` |
| §10 minimality (n ≤ 4, connected)                 | the two `decide` theorems |
| §10 `K₃ ⊔ K₂` remark                              | `ED`/`CD` claims |

Not formalised (Tier 2/3, per scoping): Stanley decomposition `χ_G(t) =
Σ_D Ω°_{P_D}(t)` and reciprocity; Propp's fixed-sink theorem; the §9 LP/TU
remark.  Uniqueness in `C = ⊔_D C_D` is the remark inside
`mem_CDset_DcOf`.
-/

