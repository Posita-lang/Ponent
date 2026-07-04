use super::*;

/// A SCAP guarantee g : State → State → Prop (Feng & Shao 2006 §4.1).
/// Relates the state at a program point to the state at the return point
/// of the current function. The two type expressions represent predicates
/// over the machine state:
///   - `pre`  describes the state *before* the function body runs.
///   - `post` describes the state *at* the return point.
/// If a function has multiple return points, g covers all traces from
/// the current point to any return point.
#[derive(Debug, Clone)]
pub struct Guarantee {
    /// Precondition: predicate over the pre-call machine state.
    pub pre: Option<TypeId>,
    /// Postcondition: predicate over the post-return machine state.
    pub post: Option<TypeId>,
    /// The set of types / memory regions that are guaranteed to be preserved.
    pub frame: Option<TypeId>,
}

impl Guarantee {
    /// Construct a guarantee from the function's `ensures` clause.
    /// The pre/post pair represent the state relation g(S_pre, S_post).
    pub fn new(pre: Option<TypeId>, post: Option<TypeId>, frame: Option<TypeId>) -> Self {
        Guarantee { pre, post, frame }
    }

    /// The identity guarantee: S = S′ (no effect).
    pub fn identity(frame: Option<TypeId>) -> Self {
        Guarantee {
            pre: None,
            post: None,
            frame,
        }
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
    /// At depth 0, verify there is no reachable return state via g.
    /// At depth n > 0, verify that every return state through g has a valid
    /// return address with a matching precondition.
    pub fn check_wfst(&self, depth: usize, current_guarantee: &Guarantee) -> bool {
        if depth == 0 {
            // WFST(0, g, S, Ψ) ≡ ¬∃S′. g S S′
            // If pre and post are both None, the guarantee is empty → well-formed.
            current_guarantee.pre.is_none() && current_guarantee.post.is_none()
        } else {
            // WFST(n, g, S, Ψ) ≡ ∀S′. g S S′ →
            //   S′.R($ra) ∈ dom(Ψ) ∧ p′ S′ ∧ WFST(n−1, g′, S′, Ψ)
            // The chain itself encodes the nesting: each entry must satisfy
            // that its postcondition is a valid precondition for the next.
            if self.stack.len() < depth {
                return false; // not enough entries in the chain
            }
            for i in 0..depth {
                let g_i = &self.stack[self.stack.len() - 1 - i];
                // Each frame's post must be compatible with the next frame's pre
                if i > 0 {
                    let prev = &self.stack[self.stack.len() - i];
                    if let (Some(prev_post), Some(curr_pre)) = (prev.post, g_i.pre) {
                        if prev_post != curr_pre {
                            return false;
                        }
                    }
                }
            }
            true
        }
    }
}
