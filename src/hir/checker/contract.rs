use super::*;

/// A predicate over machine state in SCAP (Feng & Shao 2006 §4.1).
///
/// Three-valued: true (no constraint), false (impossible, ⊥),
/// or a specific type expression representing a state predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Predicate {
    /// The trivially true predicate: holds for every state.
    True,
    /// The false predicate: holds for no state (⊥).
    False,
    /// A specific type expression interpreted as a state predicate.
    Type(TypeId),
}

impl Predicate {
    /// Is this predicate satisfied by at least one state?
    pub fn is_satisfiable(&self) -> bool {
        !matches!(self, Predicate::False)
    }

    /// Does this predicate logically imply `other`?
    ///
    /// Conservatively approximates implication via syntactic equality
    /// for `Type` variants; true → anything is true; false → anything
    /// is true; specific type → only true or the same type.
    pub fn implies(&self, other: &Predicate) -> bool {
        match (self, other) {
            // False implies everything (ex falso quodlibet)
            (Predicate::False, _) => true,
            // True only implies True
            (Predicate::True, Predicate::True) => true,
            (Predicate::True, _) => false,
            // Type(p) implies Type(q) iff p == q
            (Predicate::Type(a), Predicate::Type(b)) => a == b,
            // Type(p) implies True (True is the weakest predicate)
            (Predicate::Type(_), Predicate::True) => true,
            // Nothing implies False except False itself (handled above)
            (_, Predicate::False) => false,
        }
    }
}

/// A SCAP guarantee g : State → State → Prop (Feng & Shao 2006 §4.1).
/// Relates the state at a program point to the state at the return point
/// of the current function. The two predicate fields represent:
///   - `pre`  describes the state *before* the function body runs.
///   - `post` describes the state *at* the return point.
/// If a function has multiple return points, g covers all traces from
/// the current point to any return point.
#[derive(Debug, Clone)]
pub struct Guarantee {
    /// Precondition: predicate over the pre-call machine state.
    pub pre: Predicate,
    /// Postcondition: predicate over the post-return machine state.
    pub post: Predicate,
    /// The set of types / memory regions that are guaranteed to be preserved.
    pub frame: Option<TypeId>,
}

impl Guarantee {
    /// Construct a guarantee from explicit pre/post predicates and an
    /// optional frame type.
    pub fn new(pre: Predicate, post: Predicate, frame: Option<TypeId>) -> Self {
        Guarantee { pre, post, frame }
    }

    /// The identity guarantee: S = S′ (no effect).
    pub fn identity(frame: Option<TypeId>) -> Self {
        Guarantee {
            pre: Predicate::True,
            post: Predicate::True,
            frame,
        }
    }

    /// The empty (unsatisfiable) guarantee: ¬∃S′. g S S′.
    /// No return state is reachable — the function must not return.
    pub fn unsatisfiable(frame: Option<TypeId>) -> Self {
        Guarantee {
            pre: Predicate::True,
            post: Predicate::False,
            frame,
        }
    }

    /// Is this guarantee unsatisfiable? i.e., is there no reachable
    /// return state that satisfies g?
    ///
    /// Equivalent to WFST(0, g, S, Ψ) ≡ ¬∃S′. g S S′:
    /// either the precondition is impossible (no start state) or the
    /// postcondition is impossible (no end state).
    pub fn is_unsatisfiable(&self) -> bool {
        self.pre == Predicate::False || self.post == Predicate::False
    }

    /// Check whether every state satisfying `self.post` also satisfies
    /// `other_pre` — i.e., the postcondition of this guarantee is strong
    /// enough to serve as the precondition `other_pre`.
    ///
    /// This is the logical glue between stacked SCAP guarantees:
    /// the return state S' of the current frame must satisfy the
    /// precondition of the return address (the next frame on the chain).
    pub fn post_implies(&self, other_pre: &Predicate) -> bool {
        self.post.implies(other_pre)
    }
}

/// WFST (Well-Formed Stack) checking for SCAP (Feng & Shao 2006 §4.2).
/// Tracks a chain of guarantees representing the logical control stack.
///
/// WFST(0, g, S, Ψ) ≡ ¬∃S′. g S S′          (outermost, no return address)
/// WFST(n, g, S, Ψ) ≡ ∀S′. g S S′ →
///     S′.R($ra) ∈ dom(Ψ) ∧ p′ S′ ∧ WFST(n−1, g′, S′, Ψ)
///     where (p′, g′) = Ψ(S′.R($ra))
#[derive(Debug, Clone)]
pub struct GuaranteeChain {
    pub stack: Vec<Guarantee>,
}

impl GuaranteeChain {
    pub fn new() -> Self {
        GuaranteeChain { stack: Vec::new() }
    }

    /// Push a callee's guarantee onto the chain (SCAP CALL rule).
    pub fn push(&mut self, g: Guarantee) {
        self.stack.push(g);
    }

    /// Pop the innermost guarantee on return (SCAP RET rule).
    pub fn pop(&mut self) -> Option<Guarantee> {
        self.stack.pop()
    }

    /// The current (innermost) guarantee, if any.
    pub fn current(&self) -> Option<&Guarantee> {
        self.stack.last()
    }

    /// Check WFST condition at depth n:
    ///
    /// At depth 0, verify that `current_guarantee` is unsatisfiable
    /// (the outermost function has no return address — it must not
    /// reach a return state).
    ///
    /// At depth n > 0, verify that every return state through
    /// `current_guarantee` has a valid return address with a matching
    /// precondition, and that the remaining chain is internally
    /// consistent.
    ///
    /// # Invariants
    ///
    /// - `self.stack` must contain at least `depth` entries when
    ///   `depth > 0`.
    /// - For the outermost frame (`depth == 0`), `current_guarantee`
    ///   must be unsatisfiable (post == False).
    /// - For nested frames, `current_guarantee.post` must logically
    ///   imply the precondition of the topmost entry in `self.stack`
    ///   (the immediate caller's expected pre-state after return).
    pub fn check_wfst(&self, depth: usize, current_guarantee: &Guarantee) -> bool {
        if depth == 0 {
            // WFST(0, g, S, Ψ) ≡ ¬∃S′. g S S′
            // The guarantee must be unsatisfiable — no return state exists.
            let ok = current_guarantee.is_unsatisfiable();
            debug_assert!(
                ok,
                "WFST(0): expected unsatisfiable guarantee, but got pre={:?} post={:?}",
                current_guarantee.pre,
                current_guarantee.post
            );
            ok
        } else {
            // WFST(n, g, S, Ψ) ≡ ∀S′. g S S′ →
            //   S′.R($ra) ∈ dom(Ψ) ∧ p′ S′ ∧ WFST(n−1, g′, S′, Ψ)
            //
            // Implemented as two checks:
            //   1. g.post  ⇒  p_top   (return state satisfies caller's pre)
            //   2. For each adjacent pair in the top |depth| entries of
            //      the chain: p_i.post ⇒ p_{i+1}.pre  (internal consistency)

            // Guard: must have enough entries in the chain.
            if self.stack.len() < depth {
                debug_assert!(
                    false,
                    "WFST({depth}): chain has {} entries, need {depth}",
                    self.stack.len()
                );
                return false;
            }

            // Check 1: current_guarantee.post ⇒ top_of_chain.pre
            let top = &self.stack[self.stack.len() - 1];
            if !current_guarantee.post_implies(&top.pre) {
                debug_assert!(
                    false,
                    "WFST({depth}): current post does not imply caller's pre"
                );
                return false;
            }

            // Check 2: internal consistency of the remaining chain
            // For i in 1..depth: stack[last-i+1].post ⇒ stack[last-i].pre
            for i in 1..depth {
                let prev = &self.stack[self.stack.len() - i]; // inner (top is at last)
                let curr = &self.stack[self.stack.len() - 1 - i]; // next outer
                if !prev.post_implies(&curr.pre) {
                    debug_assert!(
                        false,
                        "WFST({depth}): chain entry {i} post does not imply entry {} pre",
                        i + 1
                    );
                    return false;
                }
            }

            true
        }
    }
}
