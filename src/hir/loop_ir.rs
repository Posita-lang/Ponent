//! Edge-based loop transition IR .
//!
//! IR types (this file); the `LoopInstr → BiiLoopProblem`
//! lowering and the independent template verifier live here.
//! The synthesis driver (candidates/limits/leap) lives in `bii.rs`; this
//! module owns the "loop problem representation and transition encoding".

use crate::hir::loop_infer::LoopInstr;
use crate::symbol::Symbol;
use num_bigint::BigInt;
use num_traits::{One, Zero};

/// A scalar variable participating in BII synthesis (aligned with Posita's
/// bit-width and signedness).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiiVar {
    pub symbol: Symbol,
    /// Bit width (1..=64; 128 requires the experimental feature).
    pub bw: u8,
    /// true = Int<N>, false = UInt<N>/unknown.
    pub signed: bool,
}

/// Arithmetic semantics (per-expression; operator suffix > type policy >
/// default trap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithSem {
    /// `+%` / `-%` or type-level `with overflow = wrap`: total modular
    /// arithmetic (BV).
    Wrap,
    /// `+?` / `-?`: clamp, piecewise — not supported by the current
    /// template domain (semantics under discussion).
    Saturate,
    /// Default and `+!`: partial, requires definedness.
    Trap,
}

/// Scalar expression (linear subset for now; Mul/bit ops left as extension
/// points).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarExpr {
    Var(usize),
    Const(BigInt),
    Add(Box<ScalarExpr>, Box<ScalarExpr>, ArithSem),
    Sub(Box<ScalarExpr>, Box<ScalarExpr>, ArithSem),
    /// Conditional choice (for the if-merge route).
    Ite(Box<Cond>, Box<ScalarExpr>, Box<ScalarExpr>),
}

/// Comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
}

/// Boolean condition (for guards / loop_guard / definedness; separated from
/// `ScalarExpr`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    True,
    False,
    Cmp {
        op: CmpOp,
        lhs: Box<ScalarExpr>,
        rhs: Box<ScalarExpr>,
        /// Signedness of the comparison: `bvslt`/`bvsle` vs `bvult`/`bvule`
        /// under BV; fixed by lowering from the operand types.
        signed: bool,
    },
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
}

/// Transition edge kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// Back to the current loop header (participates in inductiveness).
    Back,
    /// Exits the loop (not part of inductiveness; for postcondition
    /// verification).
    Exit,
}

/// One transition: path condition + trap precondition + per-variable next
/// state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionEdge {
    pub kind: EdgeKind,
    /// Path condition (if-branch / continue condition etc.); None = always
    /// true.
    pub guard: Option<Cond>,
    /// Definedness under Trap semantics: no trap arithmetic on this edge
    /// overflows. None for Wrap edges. BII synthesizes under definedness;
    /// trap absence is a separate verification obligation.
    pub definedness: Option<Cond>,
    /// Next-state expression per variable; `next_values[i] = Var(i)` means
    /// unchanged.
    pub next_values: Vec<ScalarExpr>,
}

/// The complete BII synthesis problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BiiLoopProblem {
    /// Loop variables (transitions update their next_values).
    pub vars: Vec<BiiVar>,
    /// External symbolic variables (read-only: enter the template domain,
    /// unchanged by transitions, universally quantified in queries).
    pub params: Vec<BiiVar>,
    /// Pre-loop initial values of the loop variables (constants only for
    /// now; params need no initial constraint).
    pub init: Vec<ScalarExpr>,
    /// Loop-header guard (`while cond`), separated from body path
    /// conditions.
    pub loop_guard: Cond,
    /// All edges back to the loop header (inductiveness only over these).
    pub back_edges: Vec<TransitionEdge>,
    /// Exit edges (optional for BII synthesis; needed for postcondition
    /// verification).
    pub exit_edges: Vec<TransitionEdge>,
    /// Saturating assignments `x_i := x_i +? c` — the template domain
    /// adds Clamp rows for these.
    pub saturates: Vec<(usize, i128)>,
    /// Postcondition (the `Post` of ϕ₃). `None` = no postcondition
    /// (ϕ₃ is vacuously true). Filled by the checker after
    /// `loop_instrs_to_loop_problem` and before
    /// `synthesize_problem_bii`.
    pub post: Option<Cond>,
}

/// Lowering: translate a whitelisted `LoopInstr` sequence
/// (guard-first, produced by `hir_loop_to_loop_instrs`) into a single
/// back-edge `BiiLoopProblem`.
///
/// Semantics are equivalent to `encode_sequential_transition` (SSA
/// read-after-write): all `Test*` go into `loop_guard` (matching
/// `build_refine_query`'s guard collection), and assignments are applied in
/// order to an expression state machine (`values[i]` is variable `i`'s
/// latest-value expression), so `next_values` is the final state machine
/// state. Conservative subset:
///
/// - `init` accepts only `ConstVar` (the checker generates only those;
///   untouched variables stay `Var(i)`, which is the tautology `x_i = x_i`
///   in the Init formula = unconstrained, matching the existing
///   "unknown value omitted" behavior)
/// - `params` are external symbols (read-only variables: enter the template
///   domain, unchanged by transitions, universally quantified in queries).
///   `LoopInstr` variable indices still refer to `vars`; if a guard needs
///   to reference `n` in `i < n`, the caller keeps `n` in `vars` (existing
///   behavior); `params` declares additional read-only variables
/// - `exit_edges` is empty; `definedness` is derived from the trap
///   arithmetic when `trap` is set (stay-in-range
///   conditions for `AddVar`); wrap loops (`use_bv`) pass `trap: false`
///   and carry no definedness
/// - assignment arithmetic is uniformly marked `ArithSem::Wrap` (consistent
///   with the existing synthesis: `bvadd` under BV, mathematical integers
///   under LIA, interpreted by `use_bv` at encoding time)
///
/// Returns `None` for shapes not yet supported (fail-closed).
pub(crate) fn loop_instrs_to_loop_problem(
    vars: &[Symbol],
    init: &[LoopInstr],
    body: &[LoopInstr],
    bit_widths: &[u8],
    signed: &[bool],
    params: &[BiiVar],
    trap: bool,
) -> Option<BiiLoopProblem> {
    // 1. Variable table (all loop variables, no external symbols
    //    beyond the caller-provided `params`).
    let bvars: Vec<BiiVar> = vars
        .iter()
        .enumerate()
        .map(|(i, sym)| BiiVar {
            symbol: *sym,
            bw: bit_widths.get(i).copied().unwrap_or(64),
            signed: signed.get(i).copied().unwrap_or(false),
        })
        .collect();

    // 2. init: accepts constants only; untouched variables stay
    //    `Var(i)` (tautology = unconstrained).
    let mut init_exprs: Vec<ScalarExpr> = (0..vars.len()).map(ScalarExpr::Var).collect();
    for instr in init {
        match instr {
            LoopInstr::ConstVar(i, c) => {
                init_exprs[*i] = ScalarExpr::Const(BigInt::from(*c));
            }
            _ => return None, // non-ConstVar init → fail-closed.
        }
    }

    // 3. loop_guard: conjunction of every `Test*` in body (same as
    //    build_refine_query's guard collection).
    let mut guard_parts: Vec<Cond> = Vec::new();
    // 5. definedness: per-instruction trap conditions —
    //    under the default TRAP overflow semantics, `i := i + c` is
    //    partial (it panics when the result leaves [MIN, MAX]); the edge
    //    carries the stay-in-range condition as a quantified antecedent,
    //    and the verifier's trap-absence check (`A ∧ G ∧ guard ⇒ def`)
    //    discharges it (the loop guard usually implies it).
    let mut def_parts: Vec<Cond> = Vec::new();
    // Saturating assignments: `x_i := x_i +? c`.
    let mut saturates: Vec<(usize, i128)> = Vec::new();
    // 4. next_values: expression state machine (SSA read-after-write).
    let mut values: Vec<ScalarExpr> = (0..vars.len()).map(ScalarExpr::Var).collect();
    for instr in body {
        match instr {
            LoopInstr::TestLe(i, c) => guard_parts.push(Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Var(*i)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(*c))),
                signed: signed.get(*i).copied().unwrap_or(false),
            }),
            LoopInstr::TestGe(i, c) => guard_parts.push(Cond::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ScalarExpr::Var(*i)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(*c))),
                signed: signed.get(*i).copied().unwrap_or(false),
            }),
            LoopInstr::TestDiffLe(i, j, c) => guard_parts.push(Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Sub(
                    Box::new(ScalarExpr::Var(*i)),
                    Box::new(ScalarExpr::Var(*j)),
                    ArithSem::Wrap, // the guard difference is syntactic; the comparison gives the semantics
                )),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(*c))),
                signed: signed.get(*i).copied().unwrap_or(false)
                    || signed.get(*j).copied().unwrap_or(false),
            }),
            LoopInstr::AddVar(i, c) => {
                values[*i] = ScalarExpr::Add(
                    Box::new(values[*i].clone()),
                    Box::new(ScalarExpr::Const(BigInt::from(*c))),
                    ArithSem::Wrap,
                );
                if trap && *c != 0 {
                    // Stay-in-range condition for `i := i + c`, at the
                    // variable's OWN signedness: a signed
                    // Int<N> traps outside [−2^(bw−1), 2^(bw−1)−1],
                    // an unsigned UInt<N> outside [0, 2^bw−1]. The
                    // previous code used the UNSIGNED range for both
                    // — sound for c > 0 (the unsigned bound is WEAKER
                    // than the signed one: more obligation states,
                    // conservative), but UNSOUND for c < 0: `i ≥ |c|`
                    // is STRONGER than the true signed `i ≥ MIN−c`
                    // and EXCLUDES executing (non-trapping) states
                    // from the inductive obligation — the synthesis
                    // then legitimately collapses the BII to the init
                    // (first exposed by a signed-pair test:
                    // `y := y − 1` on Int<8> with def `y ≥ 1` omitted
                    // every reachable state; all rows tightened to
                    // [0,0], the BII lost its over-approximation
                    // guarantee). The old "consistent with the
                    // template rows" note referred to the earlier
                    // unsigned-only rows; signed Interval rows now
                    // carry TRUE signed tops, so
                    // the def range follows the variable.
                    let bw = bit_widths.get(*i).copied().unwrap_or(64) as u32;
                    let is_signed = signed.get(*i).copied().unwrap_or(false);
                    let c_big = BigInt::from(*c);
                    let (min, max) = if is_signed {
                        let half = BigInt::one() << (bw as usize - 1);
                        (-half.clone(), half - 1)
                    } else {
                        (BigInt::zero(), (BigInt::one() << bw as usize) - 1)
                    };
                    if *c > 0 {
                        // no-trap ⟺ i + c ≤ max ⟺ i ≤ max − c.
                        def_parts.push(Cond::Cmp {
                            op: CmpOp::Le,
                            lhs: Box::new(ScalarExpr::Var(*i)),
                            rhs: Box::new(ScalarExpr::Const(max - &c_big)),
                            signed: is_signed,
                        });
                    } else {
                        // no-trap ⟺ i + c ≥ min ⟺ i ≥ min − c
                        // (= min + |c|; unsigned min = 0 reproduces
                        // the old `i ≥ |c|` exactly — byte-identical
                        // for unsigned variables).
                        def_parts.push(Cond::Cmp {
                            op: CmpOp::Ge,
                            lhs: Box::new(ScalarExpr::Var(*i)),
                            rhs: Box::new(ScalarExpr::Const(min - &c_big)),
                            signed: is_signed,
                        });
                    }
                }
            }
            LoopInstr::AddSat(i, c) => {
                // Saturating assignment — the transfer
                // is `x' = clamp(x + c)`; the clamp semantics live in the
                // template's Clamp rows (the antecedent encodes the
                // clamped successor). No trap definedness (saturate is
                // total — never panics).
                values[*i] = ScalarExpr::Add(
                    Box::new(values[*i].clone()),
                    Box::new(ScalarExpr::Const(BigInt::from(*c))),
                    ArithSem::Saturate,
                );
                saturates.push((*i, *c));
            }
            LoopInstr::ConstVar(i, c) => {
                values[*i] = ScalarExpr::Const(BigInt::from(*c));
            }
            LoopInstr::CopyVar(i, j) => {
                // The source may be a separated param (external symbol) —
                // reference it directly instead of indexing `values`
                // (which is sized to the loop variables only).
                values[*i] = if *j < vars.len() {
                    values[*j].clone()
                } else {
                    ScalarExpr::Var(*j)
                };
            }
            LoopInstr::If(cond, then, else_) => {
                // Merge both arms into an `Ite` per
                // assigned variable. NB: trap definedness for assignments
                // inside `if` arms is not generated yet (only the body's
                // top-level `AddVar`s contribute defs).
                let cond = fix_signed(cond, signed, vars.len());
                let then_vals = apply_block(&values, then, signed, vars.len())?;
                let else_vals = apply_block(&values, else_, signed, vars.len())?;
                for i in 0..vars.len() {
                    if then_vals[i] != else_vals[i] {
                        values[i] = ScalarExpr::Ite(
                            Box::new(cond.clone()),
                            Box::new(then_vals[i].clone()),
                            Box::new(else_vals[i].clone()),
                        );
                    }
                }
            }
        }
    }
    let loop_guard = guard_parts.into_iter().fold(Cond::True, |acc, c| {
        if matches!(acc, Cond::True) {
            c
        } else {
            Cond::And(Box::new(acc), Box::new(c))
        }
    });
    let definedness = if def_parts.is_empty() {
        None
    } else if def_parts.len() == 1 {
        Some(def_parts.into_iter().next().unwrap())
    } else {
        Some(def_parts.into_iter().fold(Cond::True, |acc, c| {
            if matches!(acc, Cond::True) {
                c
            } else {
                Cond::And(Box::new(acc), Box::new(c))
            }
        }))
    };

    Some(BiiLoopProblem {
        vars: bvars,
        params: params.to_vec(),
        init: init_exprs,
        loop_guard,
        back_edges: vec![TransitionEdge {
            kind: EdgeKind::Back,
            guard: None,
            definedness,
            next_values: values,
        }],
        exit_edges: vec![TransitionEdge {
            kind: EdgeKind::Exit,
            guard: None, // the exit condition is ¬loop_guard, handled by ϕ₃
            definedness: None,
            next_values: (0..vars.len()).map(ScalarExpr::Var).collect(), // identity transition
        }],
        saturates,
        post: None,
    })
}

/// Split the loop's UNMODIFIED variables (external
/// symbols such as `n` in `while i < n`) out of `vars` into `params`, so
/// the `BiiLoopProblem` path treats them as read-only parameters (their
/// type-range conditions are spliced into the quantified antecedents —
/// and `template_to_invariant_exprs` skips rows that
/// reference them).
///
/// The instruction indices are remapped from the original `vars` order
/// (which `hir_loop_to_loop_instrs` sorts) to the new template-domain
/// order: loop variables first (preserving their original relative
/// order), then params. Returns `None` when every variable is modified —
/// nothing to separate; the caller then uses the original inputs
/// unchanged.
pub(crate) struct SeparatedLoop {
    /// Loop variables only (the modified ones).
    pub vars: Vec<Symbol>,
    /// Remapped init instructions (indices in template-domain order).
    pub init: Vec<LoopInstr>,
    /// Remapped body instructions (same index space as `init`).
    pub body: Vec<LoopInstr>,
    /// Widths in template-domain order (loop vars first, then params).
    pub bit_widths: Vec<u8>,
    /// Signedness in the same order as `bit_widths`.
    pub signed: Vec<bool>,
    /// The separated external-symbol params.
    pub params: Vec<BiiVar>,
}

pub(crate) fn separate_loop_params(
    vars: &[Symbol],
    init: &[LoopInstr],
    body: &[LoopInstr],
    bit_widths: &[u8],
    signed: &[bool],
) -> Option<SeparatedLoop> {
    // 1. Which variables does the body modify? Plus: which have a literal
    //    init (local loop-related variables like `set j = 5`)? Separation
    //    targets only unmodified variables WITHOUT an init — genuine
    //    external symbols (`n` in `while i < n`). An unmodified variable
    //    with a literal init stays in `vars`: its compile-time value is
    //    known and keeps the synthesis tight (discarding it would
    //    degenerate the BII and can break `decreases` verification).
    let mut assigned = vec![false; vars.len()];
    // Recursive: assignments inside `if` arms also mark the variable as
    // loop-carried.
    fn mark_assigned(ins: &LoopInstr, assigned: &mut [bool]) {
        match ins {
            LoopInstr::AddVar(i, _)
            | LoopInstr::AddSat(i, _)
            | LoopInstr::ConstVar(i, _)
            | LoopInstr::CopyVar(i, _) => {
                if *i < assigned.len() {
                    assigned[*i] = true;
                }
            }
            LoopInstr::If(_, then, else_) => {
                for t in then.iter().chain(else_.iter()) {
                    mark_assigned(t, assigned);
                }
            }
            LoopInstr::TestLe(..) | LoopInstr::TestGe(..) | LoopInstr::TestDiffLe(..) => {}
        }
    }
    for ins in body {
        mark_assigned(ins, &mut assigned);
    }
    let mut has_init = vec![false; vars.len()];
    for ins in init {
        if let LoopInstr::ConstVar(i, _) = ins {
            if *i < vars.len() {
                has_init[*i] = true;
            }
        }
    }
    let mut loop_marked = vec![false; vars.len()];
    let mut n_loop = 0usize;
    for i in 0..vars.len() {
        loop_marked[i] = assigned[i] || has_init[i];
        if loop_marked[i] {
            n_loop += 1;
        }
    }
    if n_loop == vars.len() {
        return None; // nothing to separate — caller uses the original inputs.
    }

    // 2. Remap: original index → template-domain index (loop vars first,
    //    preserving original relative order, then params).
    let mut remap = vec![0usize; vars.len()];
    let mut loop_idxs = Vec::new();
    let mut param_idxs = Vec::new();
    for (i, m) in loop_marked.iter().enumerate() {
        if *m {
            loop_idxs.push(i);
        } else {
            param_idxs.push(i);
        }
    }
    for (new, old) in loop_idxs.iter().enumerate() {
        remap[*old] = new;
    }
    for (j, old) in param_idxs.iter().enumerate() {
        remap[*old] = n_loop + j;
    }

    // 3. Rewrite instruction indices to the template-domain order.
    fn remap_ins(ins: &LoopInstr, remap: &[usize]) -> LoopInstr {
        match ins {
            LoopInstr::TestLe(i, c) => LoopInstr::TestLe(remap[*i], *c),
            LoopInstr::TestGe(i, c) => LoopInstr::TestGe(remap[*i], *c),
            LoopInstr::TestDiffLe(i, j, c) => LoopInstr::TestDiffLe(remap[*i], remap[*j], *c),
            LoopInstr::AddVar(i, c) => LoopInstr::AddVar(remap[*i], *c),
            LoopInstr::AddSat(i, c) => LoopInstr::AddSat(remap[*i], *c),
            LoopInstr::ConstVar(i, c) => LoopInstr::ConstVar(remap[*i], *c),
            LoopInstr::CopyVar(i, j) => LoopInstr::CopyVar(remap[*i], remap[*j]),
            // `if` blocks are remapped recursively (both arms).
            LoopInstr::If(c, then, else_) => LoopInstr::If(
                c.clone(),
                then.iter().map(|i| remap_ins(i, remap)).collect(),
                else_.iter().map(|i| remap_ins(i, remap)).collect(),
            ),
        }
    };
    // Init for a separated param is dropped: params have no initial
    // constraint (they are provided by the caller's state). Without this,
    // a `ConstVar` whose index was remapped into the params region would
    // index past `init_exprs` (which is sized to the loop variables) and
    // panic in `loop_instrs_to_loop_problem`.
    let new_init: Vec<LoopInstr> = init
        .iter()
        .filter_map(|ins| match remap_ins(ins, &remap) {
            LoopInstr::ConstVar(i, _) if i >= n_loop => None,
            other => Some(other),
        })
        .collect();
    let new_body: Vec<LoopInstr> = body.iter().map(|ins| remap_ins(ins, &remap)).collect();

    // 4. Template-domain-ordered widths/signedness and the param list.
    let mut new_bws = vec![0u8; vars.len()];
    let mut new_signed = vec![false; vars.len()];
    for (old, &new) in remap.iter().enumerate() {
        new_bws[new] = bit_widths.get(old).copied().unwrap_or(64);
        new_signed[new] = signed.get(old).copied().unwrap_or(false);
    }
    let new_vars: Vec<Symbol> = loop_idxs.iter().map(|i| vars[*i]).collect();
    let params: Vec<BiiVar> = param_idxs
        .iter()
        .map(|i| BiiVar {
            symbol: vars[*i],
            bw: bit_widths.get(*i).copied().unwrap_or(64),
            signed: signed.get(*i).copied().unwrap_or(false),
        })
        .collect();

    Some(SeparatedLoop {
        vars: new_vars,
        init: new_init,
        body: new_body,
        bit_widths: new_bws,
        signed: new_signed,
        params,
    })
}

/// Apply a straight-line instruction block to a copy of the expression
/// state machine (SSA), returning the resulting state.
/// `if` blocks merge both arms into `Ite` expressions per assigned
/// variable; nested `if`s recurse.
fn apply_block(
    base: &[ScalarExpr],
    stmts: &[LoopInstr],
    signed: &[bool],
    vars_len: usize,
) -> Option<Vec<ScalarExpr>> {
    let mut v = base.to_vec();
    for ins in stmts {
        match ins {
            LoopInstr::AddVar(i, c) => {
                v[*i] = ScalarExpr::Add(
                    Box::new(v[*i].clone()),
                    Box::new(ScalarExpr::Const(BigInt::from(*c))),
                    ArithSem::Wrap,
                );
            }
            LoopInstr::AddSat(i, c) => {
                // Saturating assignment inside an `if`
                // arm — the successor is `clamp(x + c)` (Clamp-row
                // semantics; saturates for arms are not collected yet).
                v[*i] = ScalarExpr::Add(
                    Box::new(v[*i].clone()),
                    Box::new(ScalarExpr::Const(BigInt::from(*c))),
                    ArithSem::Saturate,
                );
            }
            LoopInstr::ConstVar(i, c) => {
                v[*i] = ScalarExpr::Const(BigInt::from(*c));
            }
            LoopInstr::CopyVar(i, j) => {
                v[*i] = if *j < vars_len {
                    v[*j].clone()
                } else {
                    ScalarExpr::Var(*j)
                };
            }
            LoopInstr::If(cond, then, else_) => {
                let cond = fix_signed(cond, signed, vars_len);
                let then_vals = apply_block(&v, then, signed, vars_len)?;
                let else_vals = apply_block(&v, else_, signed, vars_len)?;
                for k in 0..v.len() {
                    if then_vals[k] != else_vals[k] {
                        v[k] = ScalarExpr::Ite(
                            Box::new(cond.clone()),
                            Box::new(then_vals[k].clone()),
                            Box::new(else_vals[k].clone()),
                        );
                    }
                }
            }
            LoopInstr::TestLe(..) | LoopInstr::TestGe(..) | LoopInstr::TestDiffLe(..) => {}
        }
    }
    Some(v)
}

/// Fill the comparison signedness of a `Cond` from the per-variable
/// signedness (the if-condition translator has no type information).
fn fix_signed(c: &Cond, signed: &[bool], vars_len: usize) -> Cond {
    match c {
        Cond::Cmp { op, lhs, rhs, .. } => {
            let s = var_signed(lhs, signed, vars_len)
                .or_else(|| var_signed(rhs, signed, vars_len))
                .unwrap_or(false);
            Cond::Cmp {
                op: *op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                signed: s,
            }
        }
        Cond::And(a, b) => Cond::And(
            Box::new(fix_signed(a, signed, vars_len)),
            Box::new(fix_signed(b, signed, vars_len)),
        ),
        Cond::Or(a, b) => Cond::Or(
            Box::new(fix_signed(a, signed, vars_len)),
            Box::new(fix_signed(b, signed, vars_len)),
        ),
        Cond::Not(a) => Cond::Not(Box::new(fix_signed(a, signed, vars_len))),
        Cond::True | Cond::False => c.clone(),
    }
}

fn var_signed(e: &ScalarExpr, signed: &[bool], vars_len: usize) -> Option<bool> {
    match e {
        ScalarExpr::Var(i) if *i < vars_len => signed.get(*i).copied(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Separate the unmodified variable (`n` in
    /// `while i < n { i := i + 1 }`) out of vars into params.
    #[test]
    fn test_separate_loop_params_basic() {
        let vars = vec![Symbol::intern("i"), Symbol::intern("n")]; // sorted
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![
            LoopInstr::TestDiffLe(0, 1, -1), // i - n ≤ -1  ⟺  i < n
            LoopInstr::AddVar(0, 1),
        ];
        let sep = separate_loop_params(&vars, &init, &body, &[8, 8], &[false, false])
            .expect("must separate");
        assert_eq!(sep.vars, vec![Symbol::intern("i")]);
        assert_eq!(sep.params.len(), 1);
        assert_eq!(sep.params[0].symbol, Symbol::intern("n"));
        assert_eq!(sep.params[0].bw, 8);
        // Original order was already loop-first, so indices are stable.
        assert_eq!(sep.body, body);
        assert_eq!(sep.init, init);
        assert_eq!(sep.bit_widths, vec![8, 8]);
        assert_eq!(sep.signed, vec![false, false]);
        // The separated problem lowers with params.
        let problem = loop_instrs_to_loop_problem(
            &sep.vars,
            &sep.init,
            &sep.body,
            &sep.bit_widths,
            &sep.signed,
            &sep.params,
            true,
        )
        .expect("must lower with separated params");
        assert_eq!(problem.vars.len(), 1);
        assert_eq!(problem.params.len(), 1);
        assert_eq!(problem.params[0].symbol, Symbol::intern("n"));
    }

    /// Remap check: when the unmodified variable sorts first (vars =
    /// [n, i]), indices must be rewritten to template-domain order.
    #[test]
    fn test_separate_loop_params_remap() {
        let vars = vec![Symbol::intern("n"), Symbol::intern("i")]; // reversed
        let init = vec![LoopInstr::ConstVar(1, 0)]; // i = 0
        let body = vec![
            LoopInstr::TestDiffLe(1, 0, -1), // i - n ≤ -1
            LoopInstr::AddVar(1, 1),         // i := i + 1
        ];
        let sep = separate_loop_params(&vars, &init, &body, &[8, 8], &[false, false])
            .expect("must separate");
        assert_eq!(sep.vars, vec![Symbol::intern("i")]);
        assert_eq!(sep.params[0].symbol, Symbol::intern("n"));
        assert_eq!(sep.init, vec![LoopInstr::ConstVar(0, 0)]);
        assert_eq!(
            sep.body,
            vec![LoopInstr::TestDiffLe(0, 1, -1), LoopInstr::AddVar(0, 1),]
        );
        assert_eq!(sep.bit_widths, vec![8, 8]); // template-domain order
    }

    /// No unmodified variables → None (caller keeps the original inputs).
    #[test]
    fn test_separate_loop_params_none() {
        let vars = vec![Symbol::intern("i")];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        assert!(separate_loop_params(&vars, &[], &body, &[8], &[false]).is_none());
    }

    /// Basic construction: an `i := 0; while i < 6 { i := i + 1 }` problem.
    #[test]
    fn test_problem_construction() {
        let problem = BiiLoopProblem {
            vars: vec![BiiVar {
                symbol: Symbol::intern("i"),
                bw: 8,
                signed: false,
            }],
            params: vec![],
            init: vec![ScalarExpr::Const(BigInt::from(0))],
            loop_guard: Cond::Cmp {
                op: CmpOp::Lt,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(6))),
                signed: false,
            },
            back_edges: vec![TransitionEdge {
                kind: EdgeKind::Back,
                guard: None,
                definedness: None,
                next_values: vec![ScalarExpr::Add(
                    Box::new(ScalarExpr::Var(0)),
                    Box::new(ScalarExpr::Const(BigInt::from(1))),
                    ArithSem::Wrap,
                )],
            }],
            exit_edges: vec![TransitionEdge {
                kind: EdgeKind::Exit,
                guard: None,
                definedness: None,
                next_values: vec![ScalarExpr::Var(0)],
            }],
            saturates: vec![],
            post: None,
        };
        assert_eq!(problem.vars.len(), 1);
        assert_eq!(problem.back_edges.len(), 1);
        assert_eq!(problem.back_edges[0].kind, EdgeKind::Back);
        assert!(problem.params.is_empty() && problem.post.is_none());
        assert_eq!(problem.exit_edges.len(), 1);
        assert_eq!(problem.exit_edges[0].kind, EdgeKind::Exit);
    }

    /// Cond combination and signedness annotation.
    #[test]
    fn test_cond_combination() {
        let c = Cond::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ScalarExpr::Var(0)),
            rhs: Box::new(ScalarExpr::Const(BigInt::from(0))),
            signed: true,
        };
        let combo = Cond::And(Box::new(c.clone()), Box::new(Cond::Not(Box::new(c))));
        match &combo {
            Cond::And(a, b) => {
                assert!(matches!(**a, Cond::Cmp { .. }));
                assert!(matches!(**b, Cond::Not(_)));
            }
            _ => panic!("expected And"),
        }
        assert_ne!(combo, Cond::True);
    }

    /// Lowering: `i := 0; while i < 6 { i := i + 1 }` → single
    /// back edge; guard = i ≤ 5; next = i + 1 (Wrap).
    #[test]
    fn test_loop_instrs_to_loop_problem_basic() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true)
            .expect("basic loop must lower");
        assert_eq!(problem.vars.len(), 1);
        assert_eq!(problem.vars[0].bw, 8);
        assert_eq!(problem.vars[0].signed, false);
        assert_eq!(problem.init, vec![ScalarExpr::Const(BigInt::from(0))]);
        assert_eq!(
            problem.loop_guard,
            Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(5))),
                signed: false,
            }
        );
        assert_eq!(problem.back_edges.len(), 1);
        assert_eq!(problem.back_edges[0].kind, EdgeKind::Back);
        assert_eq!(
            problem.back_edges[0].next_values,
            vec![ScalarExpr::Add(
                Box::new(ScalarExpr::Var(0)),
                Box::new(ScalarExpr::Const(BigInt::from(1))),
                ArithSem::Wrap,
            )]
        );
    }

    /// Sequential assignments read the latest value (SSA equivalence):
    /// `i = i + 1; j = i` → j's new value is `i + 1`; untouched variables
    /// stay themselves.
    #[test]
    fn test_loop_instrs_to_loop_problem_ssa() {
        let vars = vec![Symbol::intern("i"), Symbol::intern("j")];
        let init = vec![LoopInstr::ConstVar(0, 0), LoopInstr::ConstVar(1, 0)];
        let body = vec![LoopInstr::AddVar(0, 1), LoopInstr::CopyVar(1, 0)];
        let problem =
            loop_instrs_to_loop_problem(&vars, &init, &body, &[8, 8], &[false, false], &[], true)
                .expect("ssa loop must lower");
        assert_eq!(
            problem.back_edges[0].next_values,
            vec![
                ScalarExpr::Add(
                    Box::new(ScalarExpr::Var(0)),
                    Box::new(ScalarExpr::Const(BigInt::from(1))),
                    ArithSem::Wrap,
                ),
                ScalarExpr::Add(
                    Box::new(ScalarExpr::Var(0)),
                    Box::new(ScalarExpr::Const(BigInt::from(1))),
                    ArithSem::Wrap,
                ),
            ],
            "j must read the incremented i (SSA read-after-write)"
        );
    }

    /// Non-ConstVar init → fail-closed (None).
    #[test]
    fn test_loop_instrs_to_loop_problem_init_fail_closed() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::AddVar(0, 1)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        assert!(
            loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true).is_none()
        );
    }

    /// Constant lower-bound guard (TestGe, from `i > 0`) lowers to
    /// `Cond::Ge`.
    #[test]
    fn test_loop_instrs_to_loop_problem_testge() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 5)];
        let body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true)
            .expect("descending loop must lower");
        assert_eq!(
            problem.loop_guard,
            Cond::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(1))),
                signed: false,
            }
        );
        assert_eq!(
            problem.back_edges[0].next_values,
            vec![ScalarExpr::Add(
                Box::new(ScalarExpr::Var(0)),
                Box::new(ScalarExpr::Const(BigInt::from(-1))),
                ArithSem::Wrap,
            )]
        );
    }

    /// params pass-through: external symbols land in problem.params and
    /// synthesis no longer fails closed (the template domain covers loop
    /// variables + parameter variables).
    #[test]
    fn test_loop_instrs_to_loop_problem_params() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let params = vec![BiiVar {
            symbol: Symbol::intern("n"),
            bw: 8,
            signed: false,
        }];
        let problem =
            loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &params, true)
                .expect("must lower with params");
        assert_eq!(problem.params.len(), 1);
        assert_eq!(problem.params[0].symbol, Symbol::intern("n"));
        assert_eq!(problem.params[0].bw, 8);
        assert_eq!(problem.vars.len(), 1);

        // Synthesis (template domain includes params) no longer fails
        // closed on non-empty params. Under LIA, params are now
        // constrained to their type range, so rows
        // referencing them converge and the loop variable's BII is
        // computed normally.
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if solver.check_version() {
            for use_bv in [false, true] {
                let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                    .expect("synthesis with params must not fail closed");
                // Loop variable i's BII is [0, 6] in both modes.
                assert_eq!(tpl.rows[0].ub, BigInt::from(6));
            }
        }
    }

    /// Trap arithmetic emits a definedness condition
    /// (`i := i + 1` on UInt<8> → `i ≤ 254`; `i := i − 1` → `i ≥ 1`);
    /// wrap loops (`trap: false`) carry none.
    #[test]
    fn test_definedness_trap_vs_wrap() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let inc_body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let trap = loop_instrs_to_loop_problem(&vars, &init, &inc_body, &[8], &[false], &[], true)
            .expect("must lower");
        assert_eq!(
            trap.back_edges[0].definedness,
            Some(Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(254))),
                signed: false,
            })
        );
        let wrap = loop_instrs_to_loop_problem(&vars, &init, &inc_body, &[8], &[false], &[], false)
            .expect("must lower");
        assert_eq!(wrap.back_edges[0].definedness, None);
        // Decreasing trap: i := i − 1 on UInt<8> → def = i ≥ 1 (MIN = 0).
        let dec_body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let dec = loop_instrs_to_loop_problem(&vars, &init, &dec_body, &[8], &[false], &[], true)
            .expect("must lower");
        assert_eq!(
            dec.back_edges[0].definedness,
            Some(Cond::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(1))),
                signed: false,
            })
        );
    }

    /// The trap stay-in-range def follows the variable's OWN
    /// signedness. Int<8>: `i := i + 1` → def = i ≤ 126 (MAX−1);
    /// `i := i − 1` → def = i ≥ −127 (MIN+1). The old unsigned-range
    /// def (i ≤ 254 / i ≥ 1) was sound only for c > 0; `i ≥ 1` on a
    /// signed variable EXCLUDED executing states (the BII-collapse
    /// unsoundness exposed by a signed-pair test). Unsigned vars
    /// keep the old constants exactly (pinned by
    /// test_definedness_trap_vs_wrap).
    #[test]
    fn test_definedness_signed_range() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let inc = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let p = loop_instrs_to_loop_problem(&vars, &init, &inc, &[8], &[true], &[], true)
            .expect("must lower");
        assert_eq!(
            p.back_edges[0].definedness,
            Some(Cond::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(126))),
                signed: true,
            })
        );
        let dec = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let p = loop_instrs_to_loop_problem(&vars, &init, &dec, &[8], &[true], &[], true)
            .expect("must lower");
        assert_eq!(
            p.back_edges[0].definedness,
            Some(Cond::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ScalarExpr::Var(0)),
                rhs: Box::new(ScalarExpr::Const(BigInt::from(-127))),
                signed: true,
            })
        );
    }

    /// `if` blocks merge both arms into `Ite` next
    /// values.
    #[test]
    fn test_loop_instrs_to_loop_problem_if_ite() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        // while i < 6 { if i < 3 { i = i + 1 } else { i = i + 2 } }
        let body = vec![
            LoopInstr::TestLe(0, 5),
            LoopInstr::If(
                Box::new(Cond::Cmp {
                    op: CmpOp::Lt,
                    lhs: Box::new(ScalarExpr::Var(0)),
                    rhs: Box::new(ScalarExpr::Const(BigInt::from(3))),
                    signed: false,
                }),
                vec![LoopInstr::AddVar(0, 1)],
                vec![LoopInstr::AddVar(0, 2)],
            ),
        ];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true)
            .expect("must lower with if");
        assert_eq!(
            problem.back_edges[0].next_values,
            vec![ScalarExpr::Ite(
                Box::new(Cond::Cmp {
                    op: CmpOp::Lt,
                    lhs: Box::new(ScalarExpr::Var(0)),
                    rhs: Box::new(ScalarExpr::Const(BigInt::from(3))),
                    signed: false,
                }),
                Box::new(ScalarExpr::Add(
                    Box::new(ScalarExpr::Var(0)),
                    Box::new(ScalarExpr::Const(BigInt::from(1))),
                    ArithSem::Wrap,
                )),
                Box::new(ScalarExpr::Add(
                    Box::new(ScalarExpr::Var(0)),
                    Box::new(ScalarExpr::Const(BigInt::from(2))),
                    ArithSem::Wrap,
                )),
            )]
        );
    }

    /// An `if`-merged loop synthesizes and verifies
    /// (the `Ite` next value is an inductive transition).
    #[test]
    fn test_loop_if_synthesize_verify() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_loop_if_synthesize_verify");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![
            LoopInstr::TestLe(0, 5),
            LoopInstr::If(
                Box::new(Cond::Cmp {
                    op: CmpOp::Lt,
                    lhs: Box::new(ScalarExpr::Var(0)),
                    rhs: Box::new(ScalarExpr::Const(BigInt::from(3))),
                    signed: false,
                }),
                vec![LoopInstr::AddVar(0, 1)],
                vec![LoopInstr::AddVar(0, 2)],
            ),
        ];
        let bw = vec![8u8];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Verified
                ),
                "if-merged template must verify (use_bv={use_bv})"
            );
        }
    }

    /// Signed variables get TRUE signed Interval
    /// bounds — `i := 5; while i > 0 { i := i - 1 }` on `Int<8>` (signed)
    /// synthesizes and verifies under both modes (the signed top is
    /// `[−128, 127]`, and the converged row carries signed bounds).
    #[test]
    fn test_signed_interval_synthesize_verify() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_signed_interval_synthesize_verify");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 5)];
        let body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let bw = vec![8u8];
        // Int<8> signed.
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[true], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Verified
                ),
                "signed-interval template must verify (use_bv={use_bv})"
            );
            // The signed Interval row carries the true signed bounds
            // (init 5, guard i ≥ 1, exit successor 0 → [0, 5]).
            assert_eq!(tpl.rows[0].lb, BigInt::from(0));
            assert_eq!(tpl.rows[0].ub, BigInt::from(5));
        }
    }

    /// `AddSat` lowers to a `Saturate` next value and
    /// records the saturate spec; no trap definedness (saturate is total).
    #[test]
    fn test_loop_instrs_to_loop_problem_addsat() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddSat(0, 1)];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true)
            .expect("must lower");
        assert_eq!(problem.saturates, vec![(0, 1)]);
        assert_eq!(problem.back_edges[0].definedness, None); // saturate: no def
        assert_eq!(
            problem.back_edges[0].next_values,
            vec![ScalarExpr::Add(
                Box::new(ScalarExpr::Var(0)),
                Box::new(ScalarExpr::Const(BigInt::from(1))),
                ArithSem::Saturate,
            )]
        );
    }

    /// A saturating loop synthesizes and verifies —
    /// `i := 0; while i < 6 { i := i +? 1 }` (never clamps on this
    /// range; the Interval row converges to [0, 6]).
    #[test]
    fn test_saturate_synthesize_verify() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_saturate_synthesize_verify");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddSat(0, 1)];
        let bw = vec![8u8];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Verified
                ),
                "saturate template must verify (use_bv={use_bv})"
            );
            // The loop variable's Interval row is [0, 6] (never clamps).
            assert_eq!(tpl.rows[0].ub, BigInt::from(6));
        }
    }

    /// A loop that CLAMPS — `i := 250; while i < 255 {
    /// i := i +? 10 }` — the successor clamps to 255 (UInt<8> MAX), so
    /// the BII must include the clamped value.
    #[test]
    fn test_saturate_clamp_boundary() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_saturate_clamp_boundary");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 250)];
        let body = vec![LoopInstr::TestLe(0, 254), LoopInstr::AddSat(0, 10)];
        let bw = vec![8u8];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Verified
                ),
                "clamp-boundary template must verify (use_bv={use_bv})"
            );
            // i=250 → clamp(260) = 255: the Interval row includes 255.
            assert_eq!(tpl.rows[0].ub, BigInt::from(255));
        }
    }

    /// Oracle — a saturating loop's synthesized
    /// template passes enumeration (the Clamp row evaluates the clamped
    /// successor).
    #[test]
    fn test_oracle_saturate() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_oracle_saturate");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        // UInt<4> — never clamps on this range (the oracle's `next`
        // evaluation is the mathematical addition, equal to the clamp
        // here): i := 0; while i < 6 { i := i +? 1 }.
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddSat(0, 1)];
        let bw = vec![4u8]; // small width: 16 states per variable.
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, false)
            .expect("must synthesize");
        assert!(
            enumerate_verify(&problem, &tpl),
            "saturating-loop template must pass enumeration"
        );
    }

    /// Differential test: the old LoopInstr path (`synthesize_bitwise_bii`)
    /// vs the new BiiLoopProblem path (`synthesize_problem_bii`) synthesize
    /// the same BII. LIA and BV modes; `BiiTemplate` is `PartialEq`, so
    /// compare directly.
    #[test]
    fn test_differential_old_vs_problem_path() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_differential_old_vs_problem_path");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        // Guard `i < 6` → `i ≤ 5`; body `i := i + 1`.
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8];
        for use_bv in [false, true] {
            let tpl_old = crate::hir::bii::synthesize_bitwise_bii(
                &solver,
                &vars,
                &init,
                &body,
                &bw,
                &[false],
                512,
                use_bv,
            )
            .expect("old path must synthesize");
            let problem =
                loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
                    .expect("must lower");
            let tpl_new = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("problem path must synthesize");
            assert_eq!(
                tpl_old, tpl_new,
                "old and problem paths must agree (use_bv={use_bv})"
            );
            assert_eq!(tpl_new.rows[0].ub, BigInt::from(6));
        }
    }

    /// Differential test (BV wrap): `x := 255; while x ≤ 255 { x := x + 1 }`.
    #[test]
    fn test_differential_wrap_bv() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_differential_wrap_bv");
            return;
        }
        let vars = vec![Symbol::intern("x")];
        let init = vec![LoopInstr::ConstVar(0, 255)];
        let body = vec![LoopInstr::TestLe(0, 255), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8];
        let tpl_old = crate::hir::bii::synthesize_bitwise_bii(
            &solver,
            &vars,
            &init,
            &body,
            &bw,
            &[false],
            512,
            true,
        )
        .expect("old path must synthesize");
        // Wrap loop: arithmetic is total (`+%`) — no trap definedness.
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], false)
            .expect("must lower");
        let tpl_new = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, true)
            .expect("problem path must synthesize");
        assert_eq!(tpl_old, tpl_new, "wrap BV paths must agree");
    }

    /// Independent verifier: synthesized results (good templates)
    /// verify; templates with injected bad bounds must be rejected.
    #[test]
    fn test_verify_template_against_problem() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_verify_template_against_problem");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![8u8];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            // Good template: the synthesis result (BII [0, 6]) must verify.
            let tpl_good = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl_good, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Verified
                ),
                "synthesized template must verify (use_bv={use_bv})"
            );
            // Bad template: upper bound 3 ([0,3] is not inductive: i=3 →
            // i'=4 escapes).
            let mut tpl_bad = tpl_good.clone();
            tpl_bad.rows[0].ub = BigInt::from(3);
            assert!(
                matches!(
                    crate::hir::bii::verify_template_against_problem(
                        &solver, &problem, &tpl_bad, use_bv
                    ),
                    crate::hir::bii::VerifyOutcome::Counterexample
                ),
                "broken template must yield a counterexample (use_bv={use_bv})"
            );
        }
    }

    // ── oracle: small-width enumeration (finite-state oracle) ──────────

    use num_traits::ToPrimitive as _;

    /// Evaluate a `ScalarExpr` (mathematical integers; supported shapes).
    fn eval_expr(e: &ScalarExpr, state: &[i128]) -> Option<i128> {
        match e {
            ScalarExpr::Var(i) => state.get(*i).copied(),
            ScalarExpr::Const(c) => c.to_i128(),
            ScalarExpr::Add(l, r, _) => Some(eval_expr(l, state)? + eval_expr(r, state)?),
            ScalarExpr::Sub(l, r, _) => Some(eval_expr(l, state)? - eval_expr(r, state)?),
            ScalarExpr::Ite(c, t, f) => {
                if eval_cond(c, state)? {
                    eval_expr(t, state)
                } else {
                    eval_expr(f, state)
                }
            }
        }
    }

    /// Evaluate a `Cond`.
    fn eval_cond(c: &Cond, state: &[i128]) -> Option<bool> {
        match c {
            Cond::True => Some(true),
            Cond::False => Some(false),
            Cond::Cmp { op, lhs, rhs, .. } => {
                let l = eval_expr(lhs, state)?;
                let r = eval_expr(rhs, state)?;
                Some(match op {
                    CmpOp::Lt => l < r,
                    CmpOp::Le => l <= r,
                    CmpOp::Gt => l > r,
                    CmpOp::Ge => l >= r,
                    CmpOp::Eq => l == r,
                    CmpOp::Neq => l != r,
                })
            }
            Cond::And(a, b) => Some(eval_cond(a, state)? && eval_cond(b, state)?),
            Cond::Or(a, b) => Some(eval_cond(a, state)? || eval_cond(b, state)?),
            Cond::Not(a) => Some(!eval_cond(a, state)?),
        }
    }

    /// Template-row expression value `f_r(X)` (mathematical integers).
    fn eval_row(row: &crate::hir::bii::BiiRow, state: &[i128]) -> Option<i128> {
        use crate::hir::bii::RowKind;
        match row.kind {
            RowKind::Interval(i) => state.get(i).copied(),
            RowKind::Diff(i, j) => Some(state.get(i)? - state.get(j)?),
            RowKind::Sum(i, j) => Some(state.get(i)? + state.get(j)?),
            RowKind::Support3(i, j, k, sj, sk) => {
                let v_i = *state.get(i)?;
                let v_j = if sj { *state.get(j)? } else { -*state.get(j)? };
                let v_k = if sk { *state.get(k)? } else { -*state.get(k)? };
                Some(v_i + v_j + v_k)
            }
            // The clamped successor `clamp(x_i + c)`.
            RowKind::Clamp(i, c) => {
                let v = *state.get(i)? + c as i128;
                let (lo, hi) = if row.signed {
                    let half = 1i128 << (row.bw as usize - 1);
                    (-half, half - 1)
                } else {
                    (0i128, (1i128 << row.bw as usize) - 1)
                };
                Some(v.clamp(lo, hi))
            }
        }
    }

    /// Template membership: `⋀_r lb_r ≤ f_r(X) ≤ ub_r`.
    fn in_invariant(tpl: &crate::hir::bii::BiiTemplate, state: &[i128]) -> bool {
        tpl.rows.iter().all(|row| match eval_row(row, state) {
            Some(v) => {
                let v_big = BigInt::from(v);
                row.lb <= v_big && v_big <= row.ub
            }
            None => false,
        })
    }

    /// Init condition: `⋀_i x_i = init_i`.
    fn init_holds(problem: &BiiLoopProblem, state: &[i128]) -> bool {
        problem
            .init
            .iter()
            .enumerate()
            .all(|(i, e)| match eval_expr(e, state) {
                Some(v) => state.get(i) == Some(&v),
                None => false,
            })
    }

    /// Single-state check: `Init ⇒ Inv` and (inside Inv)
    /// `Inv ∧ guard ∧ def ⇒ Inv'`.
    fn check_state(
        problem: &BiiLoopProblem,
        tpl: &crate::hir::bii::BiiTemplate,
        state: &[i128],
    ) -> bool {
        if init_holds(problem, state) && !in_invariant(tpl, state) {
            return false; // Init ⇒ Inv counterexample.
        }
        if in_invariant(tpl, state) {
            // The loop-header guard (G) is the implicit premise of every
            // back edge (induction: A(X) ∧ G(X) ∧ edge(X,X') ⇒ A(X')).
            // If the guard fails, the body does not run — no successor
            // to check for this state.
            if !eval_cond(&problem.loop_guard, state).unwrap_or(false) {
                return true;
            }
            for edge in &problem.back_edges {
                if edge.kind != EdgeKind::Back {
                    continue;
                }
                if let Some(g) = &edge.guard {
                    if !eval_cond(g, state).unwrap_or(false) {
                        continue;
                    }
                }
                if let Some(d) = &edge.definedness {
                    if !eval_cond(d, state).unwrap_or(false) {
                        continue;
                    }
                }
                // Apply the edge: next_values updates vars (params stay
                // unchanged).
                let mut next = state.to_vec();
                for (i, e) in edge.next_values.iter().enumerate() {
                    match eval_expr(e, state) {
                        Some(v) => next[i] = v,
                        None => return false,
                    }
                }
                if !in_invariant(tpl, &next) {
                    return false; // Inv ∧ edge ⇒ Inv' counterexample.
                }
            }
        }
        true
    }

    /// Enumerate all states (0..2^bw−1 per variable, small widths) and
    /// check the template is an inductive invariant.
    fn enumerate_verify(problem: &BiiLoopProblem, tpl: &crate::hir::bii::BiiTemplate) -> bool {
        let n_total = problem.vars.len() + problem.params.len();
        let widths: Vec<u32> = problem
            .vars
            .iter()
            .chain(problem.params.iter())
            .map(|v| v.bw as u32)
            .collect();
        fn rec(
            problem: &BiiLoopProblem,
            tpl: &crate::hir::bii::BiiTemplate,
            widths: &[u32],
            state: &mut [i128],
            idx: usize,
        ) -> bool {
            if idx == widths.len() {
                return check_state(problem, tpl, state);
            }
            let max = (1i128 << widths[idx]) - 1;
            for v in 0..=max {
                state[idx] = v;
                if !rec(problem, tpl, widths, state, idx + 1) {
                    return false;
                }
            }
            true
        }
        rec(problem, tpl, &widths, &mut vec![0i128; n_total], 0)
    }

    /// oracle: synthesized templates pass enumeration; injected bad bounds
    /// must be rejected. LIA and BV modes.
    #[test]
    fn test_oracle_enumerate_verify() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_oracle_enumerate_verify");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let bw = vec![4u8]; // small width: 16 states per variable.
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        for use_bv in [false, true] {
            let tpl_good = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, use_bv)
                .expect("must synthesize");
            assert!(
                enumerate_verify(&problem, &tpl_good),
                "synthesized template must pass enumeration (use_bv={use_bv})"
            );
            // Bad template: upper bound 3 (i=3 → i'=4 escapes; enumeration
            // must catch it).
            let mut tpl_bad = tpl_good.clone();
            tpl_bad.rows[0].ub = BigInt::from(3);
            assert!(
                !enumerate_verify(&problem, &tpl_bad),
                "broken template must fail enumeration (use_bv={use_bv})"
            );
        }
    }

    /// oracle: descending loop `i := 5; while i > 0 { i := i - 1 }`
    /// (bw=4). The BII includes the exit successor (i=1 → i'=0 must be in
    /// the invariant); enumeration passes.
    #[test]
    fn test_oracle_descending() {
        let solver = crate::hir::smt::SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_oracle_descending");
            return;
        }
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 5)];
        let body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let bw = vec![4u8];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &bw, &[false], &[], true)
            .expect("must lower");
        let tpl = crate::hir::bii::synthesize_problem_bii(&solver, &problem, 512, false)
            .expect("must synthesize");
        assert!(
            enumerate_verify(&problem, &tpl),
            "descending-loop template must pass enumeration"
        );
    }

    /// `loop_instrs_to_loop_problem` sets `post: None`; `exit_edges` contains one Exit edge.
    #[test]
    fn test_problem_post_none_and_exit_edge() {
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 0)];
        let body = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let problem = loop_instrs_to_loop_problem(&vars, &init, &body, &[8], &[false], &[], true)
            .expect("must lower");
        assert!(problem.post.is_none(), "lowering must set post: None");
        assert_eq!(problem.exit_edges.len(), 1);
        assert_eq!(problem.exit_edges[0].kind, EdgeKind::Exit);
        assert_eq!(problem.exit_edges[0].next_values.len(), 1);
    }
}
