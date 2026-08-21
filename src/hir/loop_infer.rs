//! The loop-invariant inference engine — extracted from
//! `type_eq.rs` to separate the abstract-domain substrate (`octagon`)
//! from the fixpoint iteration and the HIR translation.
//!
//! `LoopInstr` is the per-iteration intermediate representation; the
//! checker translates a HIR loop into it and the widened fixpoint
//! (`dbm_fixpoint`) converges to the loop-invariant CANDIDATE (a sound
//! over-approximation). Per the 2026-08-13 committee ruling the
//! candidates are only `@hint`s for the SMT solver — never a discharge by
//! themselves.

use crate::hir::loop_ir::Cond;
use crate::hir::octagon::{Dbm, sat_neg};
use crate::symbol::Symbol;
use num_bigint::BigInt;

pub(crate) fn dbm_fixpoint(
    init: &Dbm,
    step: &dyn Fn(&Dbm) -> Dbm,
    max_iter: usize,
    widen_after: usize,
) -> Option<Dbm> {
    let mut cur = init.clone();
    for iter in 0..max_iter {
        let next = step(&cur);
        if next.eq(&cur) {
            return Some(cur); // a fixpoint — no need to widen.
        }
        // Join first (absorbs one-step tightenings), then widen (freezes
        // relaxing bounds — the termination guarantee).
        let widened = if iter >= widen_after {
            cur.widen(&cur.join(&next))
        } else {
            cur.join(&next)
        };
        if widened.eq(&cur) {
            return Some(widened);
        }
        cur = widened;
    }
    // `max_iter` exhausted without convergence: fail closed. The last
    // matrix is NOT a fixpoint — its bounds are not loop invariants, so
    // the caller must not turn them into candidates.
    None
}

/// One loop-body instruction of the intermediate representation —
/// produced by the checker when translating a HIR loop body; consumed by
/// `infer_loop_invariant_exprs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoopInstr {
    /// `i := i + c`
    AddVar(usize, i128),
    /// `i := i +? c` (saturating arithmetic — clamps to the type range;
    /// the clamp semantics live in the template's Clamp rows).
    AddSat(usize, i128),
    /// `i := c`
    ConstVar(usize, i128),
    /// `i := j`
    CopyVar(usize, usize),
    /// guard test `i ≤ c`
    TestLe(usize, i128),
    /// guard test `i ≥ c` (constant lower bound: `i > c` lowers to `i ≥ c+1`)
    TestGe(usize, i128),
    /// guard test `i - j ≤ c`
    TestDiffLe(usize, usize, i128),
    /// `if cond { then } else { else_ }` — straight-line blocks merged
    /// into `Ite` expressions by the lowering. `cond`
    /// is a comparison over loop variables; the signedness is filled by
    /// the lowering, which has the per-variable types.
    If(Box<Cond>, Vec<LoopInstr>, Vec<LoopInstr>),
}

fn apply_loop_instr(dbm: &Dbm, instr: &LoopInstr) -> Dbm {
    match instr {
        LoopInstr::AddVar(i, c) => dbm.assign_add_var(*i, *c),
        // Saturating arithmetic is not tracked by the
        // DBM fixpoint (clamp is piecewise) — ignore it (the BII path
        // handles saturate via Clamp rows and runs first; a DBM fallback
        // candidate lacking the clamp effect is a weak hint, validated by
        // SMT).
        LoopInstr::AddSat(..) => dbm.clone(),
        LoopInstr::ConstVar(i, c) => dbm.assign_const_var(*i, *c),
        LoopInstr::CopyVar(i, j) => dbm.assign_copy_var(*i, *j),
        LoopInstr::TestLe(i, c) => dbm.test_le_var(*i, *c),
        LoopInstr::TestGe(i, c) => dbm.test_ge_var(*i, *c),
        LoopInstr::TestDiffLe(i, j, c) => dbm.test_diff_le(*i, *j, *c),
        // The DBM fixpoint does not track `if` paths —
        // ignore the branch (the BII path handles `if` via `Ite` and runs
        // first; a DBM fallback candidate lacking the branch effect is a
        // weak hint, still validated by SMT before discharge).
        LoopInstr::If(..) => dbm.clone(),
    }
}

/// Fuse adjacent `LoopInstr`s that can be combined to
/// reduce the number of transfer steps in the fixpoint iteration.
///
/// Currently only merges consecutive `AddVar` on the same variable:
/// AddVar(i, c1); AddVar(i, c2) → AddVar(i, c1 + c2)
/// This is a conservative, semantics‑preserving transformation.
///
/// Other fusion opportunities (e.g. CopyVar chains) are left for future work.
fn fuse_adjacent_instrs(instrs: &[LoopInstr]) -> Vec<LoopInstr> {
    let mut fused = Vec::with_capacity(instrs.len());
    let mut iter = instrs.iter().peekable();
    while let Some(instr) = iter.next() {
        match instr {
            LoopInstr::AddVar(i, c) => {
                // Look ahead for another AddVar on the same variable.
                let mut sum = *c;
                while let Some(&LoopInstr::AddVar(j, d)) = iter.peek() {
                    if *i == *j {
                        sum += d;
                        iter.next(); // consume it
                    } else {
                        break;
                    }
                }
                fused.push(LoopInstr::AddVar(*i, sum));
            }
            // Other instructions are passed through unchanged.
            _ => fused.push(instr.clone()),
        }
    }
    fused
}

/// Apply known type bounds (e.g. from `UInt<N>` or `Int<N>`) to the
/// initial DBM. Each bound is an optional `(min, max)` pair; if present,
/// we add `X ≥ min` and `X ≤ max` as single-variable constraints on the
/// SELF-DUAL edges (paper Figure 5 — there is no implicit zero node):
/// `X ≥ lb ⟺ −2X ≤ −2·lb` rides edge `(2i+1, 2i)` and
/// `X ≤ ub ⟺  2X ≤  2·ub` rides edge `(2i, 2i+1)`.
fn apply_type_bounds(m: &mut Dbm, bounds: &[(Option<i128>, Option<i128>)]) {
    for (i, (lb, ub)) in bounds.iter().enumerate() {
        if let Some(l) = lb {
            m.set_mirrored(2 * i + 1, 2 * i, -2 * l);
        }
        if let Some(u) = ub {
            m.set_mirrored(2 * i, 2 * i + 1, 2 * u);
        }
    }
    // Close to propagate transitive and octagonal implications.
    let _ = m.close();
}

/// The inference kernel of the `@hint(assertion)` injection pipeline —
/// the initial state and loop-body instructions drive the widened fixpoint,
/// and the converged matrix is turned into invariant candidate expressions.
/// The checker translates a HIR loop to `LoopInstr`s and submits the result
/// as a `@hint` (committee 2026-08-13: the candidates never discharge
/// obligations by themselves — the SMT solver stays the authority).
/// Guards (`TestLe`/`TestDiffLe`) are part of the body: the fixpoint absorbs
/// them, so the guard bound is NOT reported as an invariant (it only holds
/// inside the loop).
///
/// `type_bounds` is an optional slice of `(min, max)` pairs for each
/// variable, used to pre‑fill the initial DBM with compile‑time‑known
/// value ranges (e.g. from `UInt<N>` or `Int<N>`). This reduces the
/// number of fixpoint iterations.
pub(crate) fn infer_loop_invariant_exprs<'a>(
    arena: &'a bumpalo::Bump,
    vars: &[Symbol],
    init: &[LoopInstr],
    body: &[LoopInstr],
    max_iter: usize,
    widen_after: usize,
    type_bounds: Option<&[(Option<i128>, Option<i128>)]>,
) -> Vec<&'a crate::ast::Expr<'a>> {
    // 1. Fuse adjacent instructions in the body (semantics‑preserving).
    let fused_body = fuse_adjacent_instrs(body);

    // 2. Build the initial DBM.
    let mut m = Dbm::new(vars.len());

    // 3. Apply type bounds if provided.
    if let Some(bounds) = type_bounds {
        apply_type_bounds(&mut m, bounds);
    }

    // 4. Apply the initialisation instructions (they set up the pre‑state).
    for instr in init {
        m = apply_loop_instr(&m, instr);
    }

    // 5. Define the step function using the fused body.
    let step = |d: &Dbm| {
        let mut r = d.clone();
        for instr in &fused_body {
            r = apply_loop_instr(&r, instr);
        }
        r
    };

    // 6. Compute the widened fixpoint (fail closed).
    let Some(fp) = dbm_fixpoint(&m, &step, max_iter, widen_after) else {
        return Vec::new();
    };

    // 7. Extract invariant expressions from the final DBM.
    dbm_to_invariant_exprs(arena, &fp, vars)
}

/// Infer a `decreases` candidate — a non-negative integer expression
/// that strictly decreases on each iteration. Two shapes are recognized
/// (the guard is the FIRST instruction, as produced by
/// `hir_loop_to_loop_instrs`):
///
/// - Increasing counter: `i := i + c` (c > 0), guard `i ≤ ub` (constant or
///   variable) → `ub - i`: each iteration adds `c`, so `ub - i` decreases
///   by `c` and stays non-negative while `i ≤ ub` holds.
/// - Decreasing counter: `i := i - c` (c > 0, i.e. `AddVar(i, -c)`), guard
///   `i ≥ lb` (constant lower bound) → `i - lb + 1`: the guard guarantees
///   ≥ 1, decreasing by `c` each iteration.
///
/// `None` when no decreasing measure is recognized (fail-closed — the
/// caller then keeps the explicit `decreases` requirement).
pub(crate) fn infer_loop_decreases_expr<'a>(
    arena: &'a bumpalo::Bump,
    vars: &[Symbol],
    instrs: &[LoopInstr],
) -> Option<&'a crate::ast::Expr<'a>> {
    // 1. Increasing counter: `i := i + c` (c > 0), guard `i ≤ ub` → `ub - i`.
    if let Some((i, _c)) = instrs.iter().find_map(|ins| match ins {
        LoopInstr::AddVar(i, c) if *c > 0 => Some((*i, *c)),
        _ => None,
    }) {
        let ub_expr: &'a crate::ast::Expr<'a> = match instrs.first()? {
            LoopInstr::TestLe(j, ub) if *j == i => arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(*ub)),
                crate::ast::Span::new(0, 0),
            )),
            // `i <= j` (`TestDiffLe(i, j, 0)`) and `i < j` (`TestDiffLe(i, j,
            // -1)`, translated as `i - j <= -1`) both give the decreasing
            // measure `j - i`: the measure is ≥ 1 (resp. ≥ 0) while the
            // guard holds and strictly decreases by `c` each iteration.
            LoopInstr::TestDiffLe(j, k, _) if *j == i => arena.alloc(crate::ast::Expr::Ident(
                vars[*k],
                crate::ast::Span::new(0, 0),
            )),
            _ => return None,
        };
        let i_expr = arena.alloc(crate::ast::Expr::Ident(
            vars[i],
            crate::ast::Span::new(0, 0),
        ));
        return Some(arena.alloc(crate::ast::Expr::BinaryOp {
            left: ub_expr,
            op: crate::ast::BinOp::Sub,
            right: i_expr,
            span: crate::ast::Span::new(0, 0),
        }));
    }

    // 2. Decreasing counter: `i := i - c` (c > 0), guard `i ≥ lb` → `i - lb + 1`.
    if let Some((i, _c)) = instrs.iter().find_map(|ins| match ins {
        LoopInstr::AddVar(i, c) if *c < 0 => Some((*i, *c)),
        _ => None,
    }) {
        let lb_expr: &'a crate::ast::Expr<'a> = match instrs.first()? {
            LoopInstr::TestGe(j, lb) if *j == i => arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(*lb)),
                crate::ast::Span::new(0, 0),
            )),
            _ => return None,
        };
        let i_expr = arena.alloc(crate::ast::Expr::Ident(
            vars[i],
            crate::ast::Span::new(0, 0),
        ));
        let diff = arena.alloc(crate::ast::Expr::BinaryOp {
            left: i_expr,
            op: crate::ast::BinOp::Sub,
            right: lb_expr,
            span: crate::ast::Span::new(0, 0),
        });
        let one = arena.alloc(crate::ast::Expr::Literal(
            crate::ast::Literal::Int(crate::ast::IntLit::Small(1)),
            crate::ast::Span::new(0, 0),
        ));
        return Some(arena.alloc(crate::ast::Expr::BinaryOp {
            left: diff,
            op: crate::ast::BinOp::Add,
            right: one,
            span: crate::ast::Span::new(0, 0),
        }));
    }
    None
}

/// Translate a HIR loop (guard `cond` + `body`) into the
/// `LoopInstr` intermediate representation, collecting the integer
/// variables it mentions. Returns `(vars, instrs)` — the variable table
/// and the per-iteration instructions with the GUARD FIRST (each iteration
/// tests the guard, then executes the body assignments). Recognized body
/// shapes: `i := i + c` (→ `AddVar`), `i := c` (→ `ConstVar`),
/// `i := j` (→ `CopyVar`); recognized guards: `i < c` / `i <= c` /
/// `i < j` / `i <= j`. `None` for any other construct (calls, arrays,
/// unknown operators) — the caller skips inference for that loop
/// (fail-closed). The pre-loop initial values are NOT part of this —
/// the caller provides them as `init` to `infer_loop_invariant_exprs`.
pub(crate) fn hir_loop_to_loop_instrs<'input>(
    cond: &crate::hir::hir::HirExpr<'input>,
    body: &[crate::hir::hir::HirStmt<'input>],
    policy_of: &dyn Fn(Symbol) -> Option<crate::ast::OverflowPolicy>,
) -> Option<(Vec<Symbol>, Vec<LoopInstr>)> {
    use crate::hir::hir::{HirExpr, HirStmt};

    // 1. Collect the variables mentioned anywhere in the loop.
    fn collect_expr<'a>(e: &HirExpr<'a>, out: &mut Vec<Symbol>) {
        match e {
            HirExpr::Ident(s, _, _) => out.push(*s),
            HirExpr::Literal(..) => {}
            HirExpr::BinaryOp { left, right, .. } => {
                collect_expr(left, out);
                collect_expr(right, out);
            }
            HirExpr::UnaryOp { expr, .. } => collect_expr(expr, out),
            _ => {}
        }
    }
    fn collect_stmt<'a>(s: &HirStmt<'a>, out: &mut Vec<Symbol>) {
        match s {
            HirStmt::VariableDef { value: Some(v), .. } => collect_expr(v, out),
            // Plain assignments (`i = i + 1`) name the loop-carried
            // variable only in the target/value — collect them too, or
            // `idx()` fails for a variable that appears only in an
            // assignment body and the WHOLE loop is declared
            // untranslatable (every declared `decreases`/`invariant` on
            // such a loop then fails closed, even when the measure is
            // genuine — the BV pipeline stayed inert for the plain
            // syntax).
            HirStmt::Assign { target, value, .. } => {
                collect_expr(target, out);
                collect_expr(value, out);
            }
            // `if` branches mention loop variables too — collect them so
            // `idx()` works for a variable appearing only inside a branch.
            HirStmt::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                collect_expr(cond, out);
                for s in then_branch {
                    collect_stmt(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        collect_stmt(s, out);
                    }
                }
            }
            _ => {}
        }
    }

    let mut vars: Vec<Symbol> = Vec::new();
    collect_expr(cond, &mut vars);
    for s in body {
        collect_stmt(s, &mut vars);
    }
    vars.sort_unstable();
    vars.dedup();

    let idx = |s: Symbol| -> Option<usize> { vars.iter().position(|v| *v == s) };

    let mut instrs = Vec::new();

    // 2. The guard first.
    match cond {
        HirExpr::BinaryOp {
            op, left, right, ..
        } => {
            let (i, j, c) = match (left.as_ref(), right.as_ref()) {
                (HirExpr::Ident(i, _, _), HirExpr::Literal(crate::ast::Literal::Int(c), _, _)) => {
                    (i, None, Some(c.clone()))
                }
                (HirExpr::Ident(i, _, _), HirExpr::Ident(j, _, _)) => (i, Some(j), None),
                // Wrap expressions in conditions (`i +% 0 < j`): only the
                // IDENTITY wrap (`+% 0`) is faithful to the diff-guard
                // abstraction — a non-zero wrap makes the condition a
                // modular relation that `i − j ≤ c` cannot express
                // without wrap drift (the guard would model a different
                // loop). Fail closed (untranslatable) on non-identity
                // wraps rather than verify the wrong loop.
                (
                    HirExpr::BinaryOp {
                        op: crate::ast::BinOp::AddWrap,
                        left,
                        right,
                        ..
                    },
                    HirExpr::Ident(j, _, _),
                ) => {
                    let (
                        HirExpr::Ident(i, _, _),
                        HirExpr::Literal(crate::ast::Literal::Int(c), _, _),
                    ) = (left.as_ref(), right.as_ref())
                    else {
                        return None;
                    };
                    if *c != 0 {
                        return None;
                    }
                    (i, Some(j), None)
                }
                // Mirror: `j > i +% 0` / `j >= i +% 0` — the wrap is on
                // the RIGHT operand; the ordered pair is (inner, outer)
                // so the direction matches the `>`/`>=` arms below.
                (
                    HirExpr::Ident(outer, _, _),
                    HirExpr::BinaryOp {
                        op: crate::ast::BinOp::AddWrap,
                        left,
                        right,
                        ..
                    },
                ) => {
                    let (
                        HirExpr::Ident(inner, _, _),
                        HirExpr::Literal(crate::ast::Literal::Int(c), _, _),
                    ) = (left.as_ref(), right.as_ref())
                    else {
                        return None;
                    };
                    if *c != 0 {
                        return None;
                    }
                    (inner, Some(outer), None)
                }
                _ => return None,
            };

            match (op, j, c) {
                (crate::ast::BinOp::Lt, None, Some(c)) => {
                    // `i < c` ⟺ `i ≤ c-1`; `c = i128::MIN` cannot decrement
                    // — the guard `i < MIN` is always false, so the loop is
                    // untranslatable (fail-closed, no panic/wrap).
                    instrs.push(LoopInstr::TestLe(idx(*i)?, c.checked_sub(1)?));
                }
                (crate::ast::BinOp::Le, None, Some(c)) => {
                    instrs.push(LoopInstr::TestLe(idx(*i)?, c.to_i128()?));
                }
                // Constant lower bound: `i > c` ⟺ `i ≥ c+1`; `c =
                // i128::MAX` cannot increment — `i > MAX` is always false
                // (i is i128), the loop never runs, untranslatable
                // (fail-closed, symmetric with `i < MIN`).
                (crate::ast::BinOp::Gt, None, Some(c)) => {
                    instrs.push(LoopInstr::TestGe(idx(*i)?, c.checked_add(1)?));
                }
                (crate::ast::BinOp::Ge, None, Some(c)) => {
                    instrs.push(LoopInstr::TestGe(idx(*i)?, c.to_i128()?));
                }
                (crate::ast::BinOp::Lt, Some(j), None) => {
                    instrs.push(LoopInstr::TestDiffLe(idx(*i)?, idx(*j)?, -1));
                }
                (crate::ast::BinOp::Le, Some(j), None) => {
                    instrs.push(LoopInstr::TestDiffLe(idx(*i)?, idx(*j)?, 0));
                }
                // `j > i` (i.e. `i < j`): the first match already ordered
                // the pair so the plain `j > i` and the wrap-mirror
                // `j > i +% 0` both land here with the same encoding as
                // the `<` arm.
                (crate::ast::BinOp::Gt, Some(j), None) => {
                    instrs.push(LoopInstr::TestDiffLe(idx(*i)?, idx(*j)?, -1));
                }
                (crate::ast::BinOp::Ge, Some(j), None) => {
                    instrs.push(LoopInstr::TestDiffLe(idx(*i)?, idx(*j)?, 0));
                }
                _ => return None,
            }
        }
        _ => return None,
    }

    // 3. The body assignments.
    for s in body {
        match s {
            HirStmt::VariableDef {
                name: Some(n),
                value: Some(v),
                ..
            } => match v.as_ref() {
                // `i := i + c` (and the wrap variant `i := i +% c` — the
                // wrap loop is then recognized by `hir_uses_wrap` and
                // synthesized/verified under bit-vector semantics; the
                // saturating `i := i +? c` becomes AddSat).
                HirExpr::BinaryOp {
                    op, left, right, ..
                } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                    if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = right.as_ref() {
                        match op {
                            crate::ast::BinOp::AddSaturate => {
                                instrs.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                            }
                            // Type-level policy (suffix > type policy >
                            // default trap) — a
                            // PLAIN `+` on a saturating type is
                            // saturating; `+%` (explicit wrap) overrides
                            // the policy.
                            crate::ast::BinOp::Add
                                if policy_of(*n) == Some(crate::ast::OverflowPolicy::Saturate) =>
                            {
                                instrs.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                            }
                            crate::ast::BinOp::Add | crate::ast::BinOp::AddWrap => {
                                instrs.push(LoopInstr::AddVar(idx(*n)?, c.to_i128()?));
                            }
                            _ => return None,
                        }
                    } else {
                        return None;
                    }
                }
                // `i := i - c` (and `i := i -% c`) — the decreasing-counter
                // shape (`while i > 0 { i = i - 1; }`), without which no
                // descending loop was translatable and a `decreases`
                // clause on it was unconditionally rejected. The
                // saturating `i := i -? c` becomes AddSat(−c).
                HirExpr::BinaryOp {
                    op, left, right, ..
                } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                    if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = right.as_ref() {
                        match op {
                            crate::ast::BinOp::SubSaturate => {
                                instrs.push(LoopInstr::AddSat(idx(*n)?, -c));
                            }
                            // Plain `-` on a saturating type (type-level
                            // policy — suffix overrides).
                            crate::ast::BinOp::Sub
                                if policy_of(*n) == Some(crate::ast::OverflowPolicy::Saturate) =>
                            {
                                instrs.push(LoopInstr::AddSat(idx(*n)?, -c));
                            }
                            crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap => {
                                instrs.push(LoopInstr::AddVar(idx(*n)?, -c));
                            }
                            _ => return None,
                        }
                    } else {
                        return None;
                    }
                }
                // `i := c`
                HirExpr::Literal(crate::ast::Literal::Int(c), _, _) => {
                    instrs.push(LoopInstr::ConstVar(idx(*n)?, c.to_i128()?));
                }
                // `i := j`
                HirExpr::Ident(j, _, _) => {
                    instrs.push(LoopInstr::CopyVar(idx(*n)?, idx(*j)?));
                }
                _ => return None,
            },

            // The standard counter syntax `i = i + 1` / `i += 1` is
            // lowered to `HirStmt::Assign` (not `VariableDef`) — without
            // this branch every ordinary loop was untranslatable and the
            // BII/octagon/invariant pipeline was inert (a declared
            // invariant was unconditionally rejected).
            HirStmt::Assign {
                target, op, value, ..
            } => {
                // Target must be a plain variable (`i`).
                let HirExpr::Ident(n, _, _) = target.as_ref() else {
                    return None;
                };
                match op {
                    // `i += c` (compound assignment): `op: Some(Add)`, and
                    // the wrap variant `i +%= c` (`op: Some(AddWrap)`);
                    // likewise the decreasing forms `i -= c` / `i -%= c`;
                    // the saturating `i +?= c` / `i -?= c` become AddSat.
                    Some(
                        crate::ast::BinOp::Add
                        | crate::ast::BinOp::AddWrap
                        | crate::ast::BinOp::AddSaturate
                        | crate::ast::BinOp::Sub
                        | crate::ast::BinOp::SubWrap
                        | crate::ast::BinOp::SubSaturate,
                    ) => {
                        if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = value.as_ref()
                        {
                            let (saturating, delta) = match op {
                                Some(crate::ast::BinOp::AddSaturate) => (true, c.to_i128()?),
                                Some(crate::ast::BinOp::SubSaturate) => (true, -c),
                                Some(crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap) => {
                                    (false, -c)
                                }
                                // Plain `+=` on a saturating type
                                // (type-level policy — suffix overrides).
                                Some(crate::ast::BinOp::Add)
                                    if policy_of(*n)
                                        == Some(crate::ast::OverflowPolicy::Saturate) =>
                                {
                                    (true, c.to_i128()?)
                                }
                                _ => (false, c.to_i128()?),
                            };
                            if saturating {
                                instrs.push(LoopInstr::AddSat(idx(*n)?, delta));
                            } else {
                                instrs.push(LoopInstr::AddVar(idx(*n)?, delta));
                            }
                        } else {
                            return None;
                        }
                    }
                    None => match value.as_ref() {
                        // `i = i + c` (and the wrap/saturate variants
                        // `i = i +% c` / `i = i +? c`).
                        HirExpr::BinaryOp {
                            op, left, right, ..
                        } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                            if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) =
                                right.as_ref()
                            {
                                match op {
                                    crate::ast::BinOp::AddSaturate => {
                                        instrs.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                                    }
                                    // Plain `+` on a saturating type.
                                    crate::ast::BinOp::Add
                                        if policy_of(*n)
                                            == Some(crate::ast::OverflowPolicy::Saturate) =>
                                    {
                                        instrs.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                                    }
                                    crate::ast::BinOp::Add | crate::ast::BinOp::AddWrap => {
                                        instrs.push(LoopInstr::AddVar(idx(*n)?, c.to_i128()?));
                                    }
                                    _ => return None,
                                }
                            } else {
                                return None;
                            }
                        }
                        // `i = i - c` (and the wrap/saturate variants
                        // `i = i -% c` / `i = i -? c`).
                        HirExpr::BinaryOp {
                            op, left, right, ..
                        } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                            if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) =
                                right.as_ref()
                            {
                                match op {
                                    crate::ast::BinOp::SubSaturate => {
                                        instrs.push(LoopInstr::AddSat(idx(*n)?, -c));
                                    }
                                    // Plain `-` on a saturating type.
                                    crate::ast::BinOp::Sub
                                        if policy_of(*n)
                                            == Some(crate::ast::OverflowPolicy::Saturate) =>
                                    {
                                        instrs.push(LoopInstr::AddSat(idx(*n)?, -c));
                                    }
                                    crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap => {
                                        instrs.push(LoopInstr::AddVar(idx(*n)?, -c));
                                    }
                                    _ => return None,
                                }
                            } else {
                                return None;
                            }
                        }
                        // `i = c`
                        HirExpr::Literal(crate::ast::Literal::Int(c), _, _) => {
                            instrs.push(LoopInstr::ConstVar(idx(*n)?, c.to_i128()?));
                        }
                        // `i = j`
                        HirExpr::Ident(j, _, _) => {
                            instrs.push(LoopInstr::CopyVar(idx(*n)?, idx(*j)?));
                        }
                        _ => return None,
                    },
                    _ => return None,
                }
            }
            // `if` branches: the condition becomes an
            // `Ite`-merged path in the lowering; both arms are
            // straight-line assignment blocks.
            HirStmt::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let c = hir_cond_to_cond(cond.as_ref(), &idx)?;
                let then = translate_assign_block(then_branch, &idx, policy_of)?;
                let else_ = match else_branch {
                    Some(b) => translate_assign_block(b, &idx, policy_of)?,
                    None => Vec::new(),
                };
                instrs.push(LoopInstr::If(Box::new(c), then, else_));
            }
            _ => return None,
        }
    }

    Some((vars, instrs))
}

/// Translate a comparison expression into a `Cond` for an `if` branch
/// condition. Signedness is filled by the lowering
/// (`fix_signed`), which has the per-variable types.
fn hir_cond_to_cond<'input, F>(e: &crate::hir::hir::HirExpr<'input>, idx: &F) -> Option<Cond>
where
    F: Fn(Symbol) -> Option<usize>,
{
    use crate::hir::hir::HirExpr;
    use crate::hir::loop_ir::{CmpOp as O, Cond as C, ScalarExpr as E};

    let HirExpr::BinaryOp {
        op, left, right, ..
    } = e
    else {
        return None;
    };
    let (i, j, c) = match (left.as_ref(), right.as_ref()) {
        (HirExpr::Ident(i, _, _), HirExpr::Literal(crate::ast::Literal::Int(c), _, _)) => {
            (i, None, Some(c.clone()))
        }
        (HirExpr::Ident(i, _, _), HirExpr::Ident(j, _, _)) => (i, Some(j), None),
        _ => return None,
    };
    let lhs = E::Var(idx(*i)?);
    let (op_c, rhs) = match (op, j, c) {
        (crate::ast::BinOp::Lt, None, Some(c)) => (O::Lt, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Le, None, Some(c)) => (O::Le, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Gt, None, Some(c)) => (O::Gt, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Ge, None, Some(c)) => (O::Ge, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Eq, None, Some(c)) => (O::Eq, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Neq, None, Some(c)) => (O::Neq, E::Const(num_bigint::BigInt::from(c))),
        (crate::ast::BinOp::Lt, Some(j), None) => (O::Lt, E::Var(idx(*j)?)),
        (crate::ast::BinOp::Le, Some(j), None) => (O::Le, E::Var(idx(*j)?)),
        (crate::ast::BinOp::Gt, Some(j), None) => (O::Gt, E::Var(idx(*j)?)),
        (crate::ast::BinOp::Ge, Some(j), None) => (O::Ge, E::Var(idx(*j)?)),
        _ => return None,
    };
    Some(C::Cmp {
        op: op_c,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        signed: false, // filled by the lowering (`fix_signed`).
    })
}

/// Translate a straight-line statement block (assignment forms and
/// nested `if`) into loop instructions. Guard
/// translations (`Test*`) come only from the loop-header condition.
fn translate_assign_block<'input, F>(
    stmts: &[crate::hir::hir::HirStmt<'input>],
    idx: &F,
    policy_of: &dyn Fn(Symbol) -> Option<crate::ast::OverflowPolicy>,
) -> Option<Vec<LoopInstr>>
where
    F: Fn(Symbol) -> Option<usize>,
{
    use crate::hir::hir::{HirExpr, HirStmt};
    let mut out = Vec::new();
    for s in stmts {
        match s {
            HirStmt::VariableDef {
                name: Some(n),
                value: Some(v),
                ..
            } => match v.as_ref() {
                // `i := i + c` (and the wrap variant `i := i +% c`).
                HirExpr::BinaryOp {
                    op: crate::ast::BinOp::Add | crate::ast::BinOp::AddWrap,
                    left,
                    right,
                    ..
                } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                    if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = right.as_ref() {
                        out.push(LoopInstr::AddVar(idx(*n)?, c.to_i128()?));
                    } else {
                        return None;
                    }
                }
                // `i := i - c` (and `i := i -% c`).
                HirExpr::BinaryOp {
                    op: crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap,
                    left,
                    right,
                    ..
                } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                    if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = right.as_ref() {
                        out.push(LoopInstr::AddVar(idx(*n)?, -c));
                    } else {
                        return None;
                    }
                }
                // `i := c`
                HirExpr::Literal(crate::ast::Literal::Int(c), _, _) => {
                    out.push(LoopInstr::ConstVar(idx(*n)?, c.to_i128()?));
                }
                // `i := j`
                HirExpr::Ident(j, _, _) => {
                    out.push(LoopInstr::CopyVar(idx(*n)?, idx(*j)?));
                }
                _ => return None,
            },
            HirStmt::Assign {
                target, op, value, ..
            } => {
                let HirExpr::Ident(n, _, _) = target.as_ref() else {
                    return None;
                };
                match op {
                    Some(
                        crate::ast::BinOp::Add
                        | crate::ast::BinOp::AddWrap
                        | crate::ast::BinOp::AddSaturate
                        | crate::ast::BinOp::Sub
                        | crate::ast::BinOp::SubWrap
                        | crate::ast::BinOp::SubSaturate,
                    ) => {
                        if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) = value.as_ref()
                        {
                            let (saturating, delta) = match op {
                                Some(crate::ast::BinOp::AddSaturate) => (true, c.to_i128()?),
                                Some(crate::ast::BinOp::SubSaturate) => (true, -c),
                                Some(crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap) => {
                                    (false, -c)
                                }
                                // Plain `+=` on a saturating type
                                // (type-level policy — suffix overrides).
                                Some(crate::ast::BinOp::Add)
                                    if policy_of(*n)
                                        == Some(crate::ast::OverflowPolicy::Saturate) =>
                                {
                                    (true, c.to_i128()?)
                                }
                                _ => (false, c.to_i128()?),
                            };
                            if saturating {
                                out.push(LoopInstr::AddSat(idx(*n)?, delta));
                            } else {
                                out.push(LoopInstr::AddVar(idx(*n)?, delta));
                            }
                        } else {
                            return None;
                        }
                    }
                    None => match value.as_ref() {
                        HirExpr::BinaryOp {
                            op, left, right, ..
                        } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                            if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) =
                                right.as_ref()
                            {
                                match op {
                                    crate::ast::BinOp::AddSaturate => {
                                        out.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                                    }
                                    // Plain `+` on a saturating type.
                                    crate::ast::BinOp::Add
                                        if policy_of(*n)
                                            == Some(crate::ast::OverflowPolicy::Saturate) =>
                                    {
                                        out.push(LoopInstr::AddSat(idx(*n)?, c.to_i128()?));
                                    }
                                    crate::ast::BinOp::Add | crate::ast::BinOp::AddWrap => {
                                        out.push(LoopInstr::AddVar(idx(*n)?, c.to_i128()?));
                                    }
                                    _ => return None,
                                }
                            } else {
                                return None;
                            }
                        }
                        HirExpr::BinaryOp {
                            op, left, right, ..
                        } if matches!(left.as_ref(), HirExpr::Ident(l, _, _) if l == n) => {
                            if let HirExpr::Literal(crate::ast::Literal::Int(c), _, _) =
                                right.as_ref()
                            {
                                match op {
                                    crate::ast::BinOp::SubSaturate => {
                                        out.push(LoopInstr::AddSat(idx(*n)?, -c));
                                    }
                                    // Plain `-` on a saturating type.
                                    crate::ast::BinOp::Sub
                                        if policy_of(*n)
                                            == Some(crate::ast::OverflowPolicy::Saturate) =>
                                    {
                                        out.push(LoopInstr::AddSat(idx(*n)?, -c));
                                    }
                                    crate::ast::BinOp::Sub | crate::ast::BinOp::SubWrap => {
                                        out.push(LoopInstr::AddVar(idx(*n)?, -c));
                                    }
                                    _ => return None,
                                }
                            } else {
                                return None;
                            }
                        }
                        HirExpr::Literal(crate::ast::Literal::Int(c), _, _) => {
                            out.push(LoopInstr::ConstVar(idx(*n)?, c.to_i128()?));
                        }
                        HirExpr::Ident(j, _, _) => {
                            out.push(LoopInstr::CopyVar(idx(*n)?, idx(*j)?));
                        }
                        _ => return None,
                    },
                    _ => return None,
                }
            }
            // Nested `if` — recursively translated.
            HirStmt::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let c = hir_cond_to_cond(cond.as_ref(), idx)?;
                let then = translate_assign_block(then_branch, idx, policy_of)?;
                let else_ = match else_branch {
                    Some(b) => translate_assign_block(b, idx, policy_of)?,
                    None => Vec::new(),
                };
                out.push(LoopInstr::If(Box::new(c), then, else_));
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Read the inferred facts out of a closed DBM as Posita invariant
/// expressions. Every finite bound `node_a − node_b ≤ c` becomes a
/// comparison expression (`X ≤ c`, `X ≥ c`, `X − Y ≤ c`, `X + Y ≤ c`), so
/// the result is the loop-invariant candidate list. Only the tightest
/// bounds survive the closure, so redundant facts are already eliminated.
/// The candidate is a sound over-approximation — per the 2026-08-13
/// ruling it is only a `@hint` for the SMT solver, never a discharge by
/// itself.
///
/// 2N node numbering (paper §IV.C): `Xᵢ⁺ = 2i`, `Xᵢ⁻ = 2i+1` — there is
/// NO implicit zero node. Single-variable bounds ride the SELF-DUAL
/// edges (paper Figure 5: `v ≤ c` ⟺ `v⁺ − v⁻ ≤ 2c`), so their emission
/// folds the extra factor of 2 back out (`X ≤ floor(c/2)` on the
/// `(2i, 2i+1)` edge; `X ≥ −floor(c/2)` on the mirror). All other edges
/// are plain differences of ±nodes and emit their `node_bound` constant
/// verbatim (`node_bound` already halves the doubled storage once).
pub(crate) fn dbm_to_invariant_exprs<'a>(
    arena: &'a bumpalo::Bump,
    dbm: &Dbm,
    vars: &[Symbol],
) -> Vec<&'a crate::ast::Expr<'a>> {
    // Fail-closed on a shape mismatch (a caller passing a matrix built
    // over a different variable table): emitting from mismatched nodes
    // would index `vars` out of bounds — return no candidates instead.
    if dbm.size != 2 * vars.len() {
        return Vec::new();
    }
    let ident = |s: Symbol| -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::Ident(s, crate::ast::Span::new(0, 0)))
    };
    let lit = |c: i128| -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::Literal(
            crate::ast::Literal::Int(crate::ast::IntLit::Small(c)),
            crate::ast::Span::new(0, 0),
        ))
    };
    let bin = |op: crate::ast::BinOp,
               l: &'a crate::ast::Expr<'a>,
               r: &'a crate::ast::Expr<'a>|
     -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: l,
            op,
            right: r,
            span: crate::ast::Span::new(0, 0),
        })
    };
    // Decode a node into its ±X form: even `2i` ⟹ `+Xᵢ`; odd `2i+1` ⟹
    // `−Xᵢ` (the negation is spelled `0 − Xᵢ` in the AST, so downstream
    // consumers (`expr_entails_typed`'s `linear_of`) see a grammar they
    // already accept).
    let node = |n: usize| -> &'a crate::ast::Expr<'a> {
        if n % 2 == 0 {
            ident(vars[n / 2]) // Xᵢ⁺ = +Xᵢ
        } else {
            bin(crate::ast::BinOp::Sub, lit(0), ident(vars[(n - 1) / 2])) // Xᵢ⁻ = −Xᵢ
        }
    };
    let mut out = Vec::new();
    for a in 0..dbm.size {
        for b in 0..dbm.size {
            if a == b {
                continue;
            }
            // `node_bound` halves the doubled storage ONCE — `c` below is
            // the EDGE constant of `node_a − node_b ≤ c`.
            let Some(c) = dbm.node_bound(a, b) else {
                continue;
            };
            // Self-dual pair (a and b are each other's mirror, the same
            // variable): node_a − node_b = ±2Xᵢ, so the variable bound
            // needs ONE more halving (paper Figure 5).
            if (a ^ 1) == b {
                if a % 2 == 0 {
                    // X⁺ − X⁻ ≤ c ⟺ 2X ≤ c ⟹ X ≤ floor(c/2).
                    let ub = c >> 1;
                    out.push(bin(crate::ast::BinOp::Le, ident(vars[a / 2]), lit(ub)));
                } else {
                    // X⁻ − X⁺ ≤ c ⟺ −2X ≤ c ⟹ X ≥ ceil(−c/2) = −floor(c/2).
                    let lb = sat_neg(c >> 1);
                    out.push(bin(crate::ast::BinOp::Ge, ident(vars[b / 2]), lit(lb)));
                }
            } else {
                // Two distinct nodes: `(±Xᵢ) − (±Xⱼ) ≤ c` — covers the
                // difference (`X−Y ≤ c`, edge (2i, 2j)), the sum
                // (`X+Y ≤ c`, edge (2i, 2j+1)), and their negated mirrors.
                out.push(bin(
                    crate::ast::BinOp::Le,
                    bin(crate::ast::BinOp::Sub, node(a), node(b)),
                    lit(c),
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Type-level overflow policy (suffix > type
    /// policy > default trap): a PLAIN `+` on a saturating type lowers
    /// to `AddSat`; `+%` (explicit wrap) OVERRIDES the policy
    /// (`AddVar`); the default (no policy) is trap (`AddVar`).
    #[test]
    fn test_hir_loop_to_loop_instrs_type_policy() {
        use crate::hir::hir::{HirExpr, HirStmt};
        use crate::hir::types::TypeContext;

        let mut ctx = TypeContext::new();
        let ty = ctx.int(8, true);
        let span = crate::ast::Span::new(0, 0);
        let i = Symbol::intern("i");

        // `i < 255`
        let cond = HirExpr::BinaryOp {
            left: Box::new(HirExpr::Ident(i, ty, span)),
            op: crate::ast::BinOp::Lt,
            right: Box::new(HirExpr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(255)),
                ty,
                span,
            )),
            ty,
            span,
        };

        // `i = i + 10` (plain add) and `i = i +% 10` (wrap suffix).
        let plain = HirStmt::Assign {
            target: Box::new(HirExpr::Ident(i, ty, span)),
            op: None,
            value: Box::new(HirExpr::BinaryOp {
                left: Box::new(HirExpr::Ident(i, ty, span)),
                op: crate::ast::BinOp::Add,
                right: Box::new(HirExpr::Literal(
                    crate::ast::Literal::Int(crate::ast::IntLit::Small(10)),
                    ty,
                    span,
                )),
                ty,
                span,
            }),
            span,
        };
        let wrap = HirStmt::Assign {
            target: Box::new(HirExpr::Ident(i, ty, span)),
            op: None,
            value: Box::new(HirExpr::BinaryOp {
                left: Box::new(HirExpr::Ident(i, ty, span)),
                op: crate::ast::BinOp::AddWrap,
                right: Box::new(HirExpr::Literal(
                    crate::ast::Literal::Int(crate::ast::IntLit::Small(10)),
                    ty,
                    span,
                )),
                ty,
                span,
            }),
            span,
        };
        let plain_stmt = [plain];
        let wrap_stmt = [wrap];

        // Saturating policy + plain `+` → AddSat.
        let (_, instrs) = hir_loop_to_loop_instrs(&cond, &plain_stmt, &|_| {
            Some(crate::ast::OverflowPolicy::Saturate)
        })
        .expect("must lower");
        assert!(instrs.iter().any(|i| matches!(i, LoopInstr::AddSat(0, 10))));

        // Wrap suffix overrides the saturating policy → AddVar.
        let (_, instrs) = hir_loop_to_loop_instrs(&cond, &wrap_stmt, &|_| {
            Some(crate::ast::OverflowPolicy::Saturate)
        })
        .expect("must lower");
        assert!(instrs.iter().any(|i| matches!(i, LoopInstr::AddVar(0, 10))));

        // Default (no policy → trap) + plain `+` → AddVar.
        let (_, instrs) =
            hir_loop_to_loop_instrs(&cond, &plain_stmt, &|_| None).expect("must lower");
        assert!(instrs.iter().any(|i| matches!(i, LoopInstr::AddVar(0, 10))));
    }

    /// Regression: `apply_type_bounds` lower bounds must encode
    /// `X ≥ lb` on the SELF-DUAL edge (paper Figure 5:
    /// `X ≥ lb ⟺ X⁻ − X⁺ ≤ −2·lb`) WITHOUT the spurious mirrored
    /// constraint (which read `X ≤ −lb` and made `lb = 0` pin the
    /// variable to 0, or `lb > 0` unsatisfiable — the historical
    /// mixed-representation bug this test pins).
    #[test]
    fn test_apply_type_bounds_positive_lb() {
        let mut m = Dbm::new(1);
        apply_type_bounds(&mut m, &[(Some(5), Some(10))]);
        assert!(m.close(), "X ∈ [5, 10] must be satisfiable");
        assert_eq!(m.var_ub(0), Some(10), "X ≤ 10");
        assert_eq!(m.var_lb(0), Some(5), "X ≥ 5");
    }

    /// `lb = 0` (UInt-style lower bound) must NOT pin X to 0.
    #[test]
    fn test_apply_type_bounds_zero_lb() {
        let mut m = Dbm::new(1);
        apply_type_bounds(&mut m, &[(Some(0), Some(255))]);
        assert!(m.close());
        assert_eq!(m.var_ub(0), Some(255), "X ≤ 255 must survive");
        assert_eq!(m.var_lb(0), Some(0), "X ≥ 0");
    }

    /// `lb = None` (only an upper bound) is unaffected.
    #[test]
    fn test_apply_type_bounds_ub_only() {
        let mut m = Dbm::new(1);
        apply_type_bounds(&mut m, &[(None, Some(10))]);
        assert!(m.close());
        assert_eq!(m.var_ub(0), Some(10), "X ≤ 10");
        assert_eq!(m.var_lb(0), None, "no lower bound");
    }

    /// Lower-bound guard `i ≥ 5` DBM encoding (paper Figure 5): the
    /// self-dual edge `(X⁻, X⁺)` carries `−2X ≤ −10`; the upper bound
    /// `i ≤ 5` rides `(X⁺, X⁻)` with `2X ≤ 10`. The semantic projections
    /// fold the extra factor of 2 back out.
    #[test]
    fn test_ge_var_encoding() {
        let r = Dbm::new(1).test_ge_var(0, 5);
        assert_eq!(r.var_lb(0), Some(5), "X ≥ 5");
        // Mirrors the upper bound: `i ≤ 5` encodes as 2X ≤ 10.
        let l = Dbm::new(1).test_le_var(0, 5);
        assert_eq!(l.var_ub(0), Some(5), "X ≤ 5");
        // NB: on ⊥, the semantic projections return None rather than
        // panicking out of bounds — the old test's "index out of bounds:
        // len is 0" panic class is gone.
    }

    /// End-to-end: descending counter loop `i := 5; while i > 0 {
    /// i := i - 1 }` (guard lowers to `TestGe(i, 1)`, body
    /// `AddVar(i, -1)`) — the fixpoint converges and yields candidates
    /// (regression: TestGe is consumed by the DBM fixpoint correctly).
    #[test]
    fn test_ge_end_to_end_descending_loop() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let init = vec![LoopInstr::ConstVar(0, 5)];
        let body = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let exprs = infer_loop_invariant_exprs(&arena, &vars, &init, &body, 100, 2, None);
        assert!(
            !exprs.is_empty(),
            "descending loop must produce candidates (i ∈ [1, 5])"
        );
        // Head semantics (fp = (−∞, 5], C-2 decided): must include i ≤ 5;
        // must NOT include the guard bound i ≥ 1. No substring assertions
        // — '1' would be polluted by redundant facts.
        let (mut has_le_5, mut has_ge_1) = (false, false);
        for e in &exprs {
            if let crate::ast::Expr::BinaryOp {
                op, left, right, ..
            } = e
            {
                if matches!(**left, crate::ast::Expr::Ident(s, ..) if s == Symbol::intern("i")) {
                    if let crate::ast::Expr::Literal(
                        crate::ast::Literal::Int(crate::ast::IntLit::Small(n)),
                        ..,
                    ) = &**right
                    {
                        match (op, *n) {
                            (crate::ast::BinOp::Le, 5) => has_le_5 = true,
                            (crate::ast::BinOp::Ge, 1) => has_ge_1 = true,
                            _ => {}
                        }
                    }
                }
            }
        }
        assert!(has_le_5, "head invariant must include i ≤ 5: {exprs:?}");
        assert!(
            !has_ge_1,
            "guard bound i ≥ 1 must NOT be reported: {exprs:?}"
        );
    }

    /// Decreasing counter decreases candidate: `while i > 0 { i := i - 1 }`
    /// (guard `i ≥ 1`, body `AddVar(i, -1)`) → candidate `(i - 1) + 1`.
    #[test]
    fn test_decreases_descending_counter() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let instrs = vec![LoopInstr::TestGe(0, 1), LoopInstr::AddVar(0, -1)];
        let dec = infer_loop_decreases_expr(&arena, &vars, &instrs)
            .expect("descending counter must yield a decreases candidate");
        // Shape `(i - lb) + 1`.
        match dec {
            crate::ast::Expr::BinaryOp {
                op: crate::ast::BinOp::Add,
                left,
                ..
            } => match left {
                crate::ast::Expr::BinaryOp {
                    op: crate::ast::BinOp::Sub,
                    ..
                } => {}
                other => panic!("expected (i - lb) + 1, got {other:?}"),
            },
            other => panic!("expected (i - lb) + 1, got {other:?}"),
        }
    }

    /// Increasing counter decreases candidate regression: `while i < 6 {
    /// i := i + 1 }` → candidate `5 - i` (guard `i ≤ 5`).
    #[test]
    fn test_decreases_increasing_counter() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let instrs = vec![LoopInstr::TestLe(0, 5), LoopInstr::AddVar(0, 1)];
        let dec = infer_loop_decreases_expr(&arena, &vars, &instrs)
            .expect("increasing counter must yield a decreases candidate");
        // Shape `ub - i`.
        match dec {
            crate::ast::Expr::BinaryOp {
                op: crate::ast::BinOp::Sub,
                left,
                ..
            } => match left {
                crate::ast::Expr::Literal(
                    crate::ast::Literal::Int(crate::ast::IntLit::Small(5)),
                    _,
                ) => {}
                other => panic!("expected 5 - i, got {other:?}"),
            },
            other => panic!("expected 5 - i, got {other:?}"),
        }
    }
}
