use super::*;
use crate::ast::{self, BinOp, Literal, Span, VariableKind};
use crate::hir::hir::{HirExpr, HirMatchArm, HirParam, HirPattern, HirStmt};
use crate::hir::types::{TypeContext, TypeId};
use crate::symbol::Symbol;

fn make_int_val<'input>(n: i128, ty: TypeId) -> HirExpr<'input> {
    HirExpr::Literal(Literal::Int(ast::IntLit::Small(n)), ty, Span::new(0, 0))
}

fn make_int<'input>(n: i128) -> HirExpr<'input> {
    make_int_val(n, TypeId::NONE)
}

/// Insert a comptime variable into the evaluator with a unique slot.
/// Used in tests that need to set up variable state before evaluation.
fn insert_var<'a, 'input>(
    ec: &mut ComptimeEvalContext<'a, 'input>,
    name: &str,
    val: ComptimeValue<'input>,
) {
    let slot = ec.allocate_slot();
    ec.cur_slot.insert(Symbol::intern(name), slot);
    ec.variables.insert(slot, val);
}

/// Look up a comptime variable by name via the slot mapping.
/// Returns `None` if the variable doesn't exist in the current scope.
fn get_var<'a, 'input>(
    ec: &'a ComptimeEvalContext<'a, 'input>,
    name: &str,
) -> Option<&'a ComptimeValue<'input>> {
    ec.cur_slot
        .get(&Symbol::intern(name))
        .and_then(|slot| ec.variables.get(slot))
}

fn make_bool<'input>(b: bool) -> HirExpr<'input> {
    HirExpr::Literal(Literal::Bool(b), TypeId::NONE, Span::new(0, 0))
}

fn make_binop_ty<'input>(
    l: HirExpr<'input>,
    op: BinOp,
    r: HirExpr<'input>,
    ty: TypeId,
) -> HirExpr<'input> {
    HirExpr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
        ty,
        span: Span::new(0, 0),
    }
}

fn make_binop<'input>(l: HirExpr<'input>, op: BinOp, r: HirExpr<'input>) -> HirExpr<'input> {
    make_binop_ty(l, op, r, TypeId::NONE)
}

/// Create an Int<32> type and wrap a value in a Literal with that type.
fn make_int32<'input>(ctx: &mut TypeContext<'input>, n: i128) -> HirExpr<'input> {
    let int32 = ctx.int(32, true);
    make_int_val(n, int32)
}

/// Create a BinaryOp with Int<32> as the result type.
fn make_binop32<'input>(
    ctx: &mut TypeContext<'input>,
    l: HirExpr<'input>,
    op: BinOp,
    r: HirExpr<'input>,
) -> HirExpr<'input> {
    let int32 = ctx.int(32, true);
    make_binop_ty(l, op, r, int32)
}

fn make_block<'input>(stmts: Vec<HirStmt<'input>>, last: HirExpr<'input>) -> HirExpr<'input> {
    let mut all = stmts;
    all.push(HirStmt::Expression(Box::new(last)));
    HirExpr::Block(all, TypeId::NONE, Span::new(0, 0))
}

fn make_if<'input>(
    cond: HirExpr<'input>,
    then: HirExpr<'input>,
    els: Option<HirExpr<'input>>,
) -> HirExpr<'input> {
    let then_block = vec![HirStmt::Expression(Box::new(then))];
    let else_block = els.map(|e| vec![HirStmt::Expression(Box::new(e))]);
    HirExpr::If {
        cond: Box::new(cond),
        then_branch: then_block,
        else_branch: else_block,
        is_expression: true,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    }
}

fn eval<'input>(
    ctx: &mut TypeContext<'input>,
    expr: &HirExpr<'input>,
) -> Result<ComptimeValue<'input>, ComptimeError> {
    use crate::diagnostics::DiagCtxt;
    use crate::hir::symbol::SymbolTable;
    use crate::hir::types::{CrateId, DefId};
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut diag = DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(ctx, &symbols, &mut diag);
    ec.eval_expr(expr)
}

#[test]
fn test_eval_int_literal() {
    let mut ctx = TypeContext::new();
    let r = eval(&mut ctx, &make_int(42));
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_bool_literal() {
    let mut ctx = TypeContext::new();
    let r = eval(&mut ctx, &make_bool(true));
    assert!(matches!(r, Ok(ComptimeValue::Bool(true))));
}

#[test]
fn test_eval_add() {
    let mut ctx = TypeContext::new();
    let a = make_int32(&mut ctx, 3);
    let b = make_int32(&mut ctx, 4);
    let expr = make_binop32(&mut ctx, a, BinOp::Add, b);
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(7))));
}

#[test]
fn test_eval_sub() {
    let mut ctx = TypeContext::new();
    let a = make_int32(&mut ctx, 10);
    let b = make_int32(&mut ctx, 3);
    let expr = make_binop32(&mut ctx, a, BinOp::Sub, b);
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(7))));
}

#[test]
fn test_eval_mul() {
    let mut ctx = TypeContext::new();
    let a = make_int32(&mut ctx, 6);
    let b = make_int32(&mut ctx, 7);
    let expr = make_binop32(&mut ctx, a, BinOp::Mul, b);
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_div() {
    let mut ctx = TypeContext::new();
    let a = make_int32(&mut ctx, 10);
    let b = make_int32(&mut ctx, 3);
    let expr = make_binop32(&mut ctx, a, BinOp::Div, b);
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(3))));
}

#[test]
fn test_eval_div_by_zero() {
    let mut ctx = TypeContext::new();
    let expr = make_binop(make_int(1), BinOp::Div, make_int(0));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Err(ComptimeError::DivisionByZero)));
}

#[test]
fn test_eval_int8_overflow() {
    let mut ctx = TypeContext::new();
    let int8_ty = ctx.int(8, true);
    // 100 + 50 = 150, which overflows Int<8> (max 127) but not i128.
    let expr = make_binop_ty(
        make_int_val(100, int8_ty),
        BinOp::Add,
        make_int_val(50, int8_ty),
        int8_ty,
    );
    let r = eval(&mut ctx, &expr);
    assert!(
        matches!(r, Err(ComptimeError::Overflow)),
        "Int<8> 100 + 50 should overflow, got {:?}",
        r
    );
}

#[test]
fn test_eval_nested_arith() {
    let mut ctx = TypeContext::new();
    let a = make_int32(&mut ctx, 1);
    let b = make_int32(&mut ctx, 2);
    let c = make_int32(&mut ctx, 3);
    let inner = make_binop32(&mut ctx, a, BinOp::Add, b);
    let expr = make_binop32(&mut ctx, inner, BinOp::Mul, c);
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(9))));
}

#[test]
fn test_eval_if_true() {
    let mut ctx = TypeContext::new();
    let expr = make_if(make_bool(true), make_int(1), Some(make_int(2)));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(1))));
}

#[test]
fn test_eval_if_false() {
    let mut ctx = TypeContext::new();
    let expr = make_if(make_bool(false), make_int(1), Some(make_int(2)));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(2))));
}

#[test]
fn test_eval_block() {
    let mut ctx = TypeContext::new();
    let expr = make_block(
        vec![HirStmt::Expression(Box::new(make_int(1)))],
        make_int(2),
    );
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(2))));
}

#[test]
fn test_eval_step_limit() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    ec.set_step_limit(0);
    let r = ec.eval_expr(&make_int(42));
    assert!(matches!(r, Err(ComptimeError::StepLimitExceeded)));
}

// ── Phase 2: Variable binding tests ────────────────────────────────

#[test]
fn test_eval_variable_def() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let block = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(42))),
        pattern: None,
        else_branch: None,
        kind: crate::ast::VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[block]);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
    assert!(matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_variable_assign() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "x", ComptimeValue::Int(10));
    ec.memory_used = ComptimeValue::Int(10).memory_size();

    let assign = HirStmt::Assign {
        target: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        value: Box::new(make_int(20)),
        op: None,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[assign]);
    assert!(matches!(r, Ok(ComptimeValue::Int(20))));
    assert!(matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(20))));
}

#[test]
fn test_eval_assign_unknown_variable() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let assign = HirStmt::Assign {
        target: Box::new(HirExpr::Ident(
            "nonexistent".into(),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        value: Box::new(make_int(20)),
        op: None,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[assign]);
    assert!(matches!(r, Err(ComptimeError::UnknownIdentifier(_))));
}

// ── Phase 3: Ident resolution tests ────────────────────────────────

#[test]
fn test_eval_ident_resolves_local_var() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "x", ComptimeValue::Int(99));

    let expr = HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(99))));
}

#[test]
fn test_eval_ident_unknown() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Ident("unknown".into(), TypeId::NONE, Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Err(ComptimeError::UnknownIdentifier(_))));
}

// ── Phase 4: Function call tests ───────────────────────────────────

#[test]
fn test_eval_comptime_fn_call() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Register a comptime function: def double(x: Int<32>) -> Int<32> { 2 * x }
    let body = vec![HirStmt::Expression(Box::new(make_binop_ty(
        make_int_val(2, int32),
        BinOp::Mul,
        HirExpr::Ident("x".into(), int32, Span::new(0, 0)),
        int32,
    )))];
    ec.register_fn("double".into(), vec!["x".into()], body);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("double".into(), int32, Span::new(0, 0))),
        args: vec![make_int_val(21, int32)],
        comptime: true,
        ty: int32,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(
        matches!(r, Ok(ComptimeValue::Int(42))),
        "double(21) = {:?}",
        r
    );
}

#[test]
fn test_eval_comptime_fn_call_unknown() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident(
            "undefined_fn".into(),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        args: vec![],
        comptime: true,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(matches!(r, Err(ComptimeError::UnknownIdentifier(_))));
}

// ── Phase 5: Loop tests ────────────────────────────────────────────

#[test]
fn test_eval_while_loop() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    ec.memory_used = ComptimeValue::Int(0).memory_size();
    // Simulate a while loop: while i < 5 { i = i + 1 }
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(5, int32), int32);
    let body = vec![HirStmt::Assign {
        target: Box::new(i_expr()),
        value: Box::new(make_binop_ty(
            i_expr(),
            BinOp::Add,
            make_int_val(1, int32),
            int32,
        )),
        op: None,
        span: Span::new(0, 0),
    }];

    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_block(&[while_stmt]);
    // After the loop, i should be 5
    assert!(matches!(get_var(&ec, "i"), Some(ComptimeValue::Int(5))));
}

#[test]
fn test_eval_while_scope_isolation() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Outer variable `i` — exists before and should persist after the loop.
    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    ec.memory_used = ComptimeValue::Int(0).memory_size();

    // while i < 1 { set x = 42; i = i + 1 }
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(1, int32), int32);
    let body = vec![
        HirStmt::VariableDef {
            kind: crate::ast::VariableKind::Set,
            mutable: false,
            name: Some(crate::symbol::Symbol::intern("x")),
            pattern: None,
            ty: int32,
            value: Some(Box::new(make_int_val(42, int32))),
            else_branch: None,
            span: Span::new(0, 0),
            type_captures: vec![],
        },
        HirStmt::Assign {
            target: Box::new(i_expr()),
            value: Box::new(make_binop_ty(
                i_expr(),
                BinOp::Add,
                make_int_val(1, int32),
                int32,
            )),
            op: None,
            span: Span::new(0, 0),
        },
    ];

    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_block(&[while_stmt]);

    // After the loop, outer variable `i` should still be accessible and updated.
    let i_val = get_var(&ec, "i");
    assert!(
        i_val.is_some(),
        "outer variable i should still exist after while loop"
    );
    assert!(
        matches!(i_val, Some(ComptimeValue::Int(1))),
        "outer variable i should be 1 after loop, got {:?}",
        i_val
    );

    // After the loop, inner variable `x` must NOT leak.
    assert!(
        get_var(&ec, "x").is_none(),
        "variable x defined inside while body should not leak to outer scope"
    );
}

#[test]
fn test_eval_while_shadowing_same_name() {
    // Verify that a `set` inside a while body shadowing an outer variable
    // by the same name does NOT persist after the loop, while normal
    // modifications to outer variables (via `Assign`) ARE preserved.
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Outer variable `x` (will be shadowed inside the loop) and `i` (counter).
    insert_var(&mut ec, "x", ComptimeValue::Int(5));
    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    ec.memory_used = ComptimeValue::Int(5).memory_size() + ComptimeValue::Int(0).memory_size();

    // while i < 1 { set x = 42; i = i + 1 }
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(1, int32), int32);
    let body = vec![
        // `set x = 42` — shadows outer x inside the loop body
        HirStmt::VariableDef {
            kind: crate::ast::VariableKind::Set,
            mutable: false,
            name: Some(crate::symbol::Symbol::intern("x")),
            pattern: None,
            ty: int32,
            value: Some(Box::new(make_int_val(42, int32))),
            else_branch: None,
            span: Span::new(0, 0),
            type_captures: vec![],
        },
        // `i = i + 1` — modifies outer variable
        HirStmt::Assign {
            target: Box::new(i_expr()),
            value: Box::new(make_binop_ty(
                i_expr(),
                BinOp::Add,
                make_int_val(1, int32),
                int32,
            )),
            op: None,
            span: Span::new(0, 0),
        },
    ];
    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_block(&[while_stmt]);

    // After the loop, outer `x` must be restored to 5 (shadow was cleaned up).
    let x_val = get_var(&ec, "x");
    assert!(
        matches!(x_val, Some(ComptimeValue::Int(5))),
        "outer x should be restored to 5 after while loop, got {:?}",
        x_val,
    );

    // After the loop, `i` must be 1 (modification preserved).
    let i_val = get_var(&ec, "i");
    assert!(
        matches!(i_val, Some(ComptimeValue::Int(1))),
        "outer i should be 1 after while loop, got {:?}",
        i_val,
    );
}

#[test]
fn test_assign_while_internal_var_after_loop() {
    // Regression: assigning to a variable defined inside a while body
    // after the loop ends must return UnknownIdentifier.  The old
    // `cur_slot` leak made such variables appear writable in the outer
    // scope even though they should be gone.
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Counter variable `i` that makes the loop run exactly once.
    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    ec.memory_used = ComptimeValue::Int(0).memory_size();

    // while i < 1 { set x = 42; i = i + 1 }
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(1, int32), int32);
    let body = vec![
        HirStmt::VariableDef {
            kind: crate::ast::VariableKind::Set,
            mutable: false,
            name: Some(crate::symbol::Symbol::intern("x")),
            pattern: None,
            ty: int32,
            value: Some(Box::new(make_int_val(42, int32))),
            else_branch: None,
            span: Span::new(0, 0),
            type_captures: vec![],
        },
        HirStmt::Assign {
            target: Box::new(i_expr()),
            value: Box::new(make_binop_ty(
                i_expr(),
                BinOp::Add,
                make_int_val(1, int32),
                int32,
            )),
            op: None,
            span: Span::new(0, 0),
        },
    ];
    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };

    let _ = ec.eval_block(&[while_stmt]);

    // After the loop, `x` should NOT be assignable — it was scoped inside.
    let assign = HirStmt::Assign {
        target: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        value: Box::new(make_int(99)),
        op: None,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[assign]);
    assert!(
        matches!(r, Err(ComptimeError::UnknownIdentifier(_))),
        "assigning to a while-internal variable after the loop should be UnknownIdentifier, got {:?}",
        r,
    );
}

#[test]
fn test_eval_while_step_limit() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    ec.set_step_limit(5);

    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(100, int32), int32);
    let body = vec![HirStmt::Assign {
        target: Box::new(i_expr()),
        value: Box::new(make_binop_ty(
            i_expr(),
            BinOp::Add,
            make_int_val(1, int32),
            int32,
        )),
        op: None,
        span: Span::new(0, 0),
    }];

    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[while_stmt]);
    assert!(matches!(r, Err(ComptimeError::StepLimitExceeded)));
}

// ── Phase 6: TypeInfo test ─────────────────────────────────────────

#[test]
fn test_eval_type_info() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::TypeInfo(int32, Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok() && matches!(r.unwrap(), ComptimeValue::TypeInfo(_)));
}

// ── Phase 9: Variable scope isolation test ─────────────────────────

#[test]
fn test_eval_fn_call_scope_isolation() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "x", ComptimeValue::Int(1));

    // Register a function that assigns to its own param, not the outer scope
    let body = vec![HirStmt::Assign {
        target: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        value: Box::new(make_int(99)),
        op: None,
        span: Span::new(0, 0),
    }];
    ec.register_fn("mutate_x".into(), vec!["x".into()], body);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident(
            "mutate_x".into(),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        args: vec![make_int(10)],
        comptime: true,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_expr(&call);
    // Outer x should still be 1, not 99
    assert!(matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(1))));
}

// ── Phase 10: Float arithmetic tests ──────────────────────────────

#[test]
fn test_eval_float_add() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::BinaryOp {
        left: Box::new(HirExpr::Literal(
            Literal::Float(1.5),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        op: BinOp::Add,
        right: Box::new(HirExpr::Literal(
            Literal::Float(2.5),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Float(v)) if (v - 4.0).abs() < 1e-10));
}

#[test]
fn test_eval_float_mul() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::BinaryOp {
        left: Box::new(HirExpr::Literal(
            Literal::Float(3.0),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        op: BinOp::Mul,
        right: Box::new(HirExpr::Literal(
            Literal::Float(1.5),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Float(v)) if (v - 4.5).abs() < 1e-10));
}

// ── Phase 11: String operations tests ─────────────────────────────

#[test]
fn test_eval_string_concat() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::BinaryOp {
        left: Box::new(HirExpr::Literal(
            Literal::String("hello ".into()),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        op: BinOp::Add,
        right: Box::new(HirExpr::Literal(
            Literal::String("world".into()),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(&r, Ok(ComptimeValue::String(s)) if s.as_ref() == "hello world"));
}

#[test]
fn test_eval_string_eq() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::BinaryOp {
        left: Box::new(HirExpr::Literal(
            Literal::String("abc".into()),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        op: BinOp::Eq,
        right: Box::new(HirExpr::Literal(
            Literal::String("abc".into()),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Bool(true))));
}

// ── Phase 12: Comparison operators tests ──────────────────────────

#[test]
fn test_eval_int_comparisons() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let a = make_int_val(5, int32);
    let b = make_int_val(10, int32);

    let tests = vec![
        (BinOp::Eq, false),
        (BinOp::Neq, true),
        (BinOp::Lt, true),
        (BinOp::Gt, false),
        (BinOp::Le, true),
        (BinOp::Ge, false),
    ];
    for (op, expected) in tests {
        let expr = make_binop_ty(a.clone(), op, b.clone(), int32);
        let r = eval(&mut ctx, &expr);
        assert!(
            matches!(&r, Ok(ComptimeValue::Bool(v)) if *v == expected),
            "5 {:?} 10 should be {}, got {:?}",
            op,
            expected,
            r
        );
    }
}

#[test]
fn test_eval_float_comparisons() {
    let mut ctx = TypeContext::new();
    let a = HirExpr::Literal(Literal::Float(3.14), TypeId::NONE, Span::new(0, 0));
    let b = HirExpr::Literal(Literal::Float(2.71), TypeId::NONE, Span::new(0, 0));

    let tests = vec![
        (BinOp::Eq, false),
        (BinOp::Neq, true),
        (BinOp::Lt, false),
        (BinOp::Gt, true),
    ];
    for (op, expected) in tests {
        let expr = HirExpr::BinaryOp {
            left: Box::new(a.clone()),
            op,
            right: Box::new(b.clone()),
            ty: TypeId::NONE,
            span: Span::new(0, 0),
        };
        let r = eval(&mut ctx, &expr);
        assert!(
            matches!(&r, Ok(ComptimeValue::Bool(v)) if *v == expected),
            "3.14 {:?} 2.71 should be {}, got {:?}",
            op,
            expected,
            r
        );
    }
}

// ── Phase 13: Pointer tests ───────────────────────────────────────

#[test]
fn test_eval_ref_and_deref() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "x", ComptimeValue::Int(42));

    // &x
    let ref_expr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Ref,
        expr: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&ref_expr);
    assert!(matches!(&r, Ok(ComptimeValue::Pointer { .. })));

    // *ptr
    let deref_expr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(ref_expr),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&deref_expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 13b: Pointer shadowing regression tests ────────────────
// These verify that pointers captured via `&x` / `&mut x` in an outer
// scope correctly resolve to the original slot even when an inner scope
// shadows `x` with a new `set x = ...`.  (Bug: Pointer used to store the
// *name* of the variable, so dereference would find the shadowed value.)

#[test]
fn test_eval_pointer_shadowing_block() {
    // set x = 1; set p = &x; { set x = 2; *p }  →  *p should be 1
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // set x = 1
    let def_x = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(1))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set p = &x
    let ref_x = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Ref,
        expr: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let def_p = HirStmt::VariableDef {
        name: Some("p".into()),
        value: Some(Box::new(ref_x)),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // Inner block: { set x = 2; *p }
    let inner_var = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(2))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let deref_p = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(HirExpr::Ident("p".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let inner_block = HirExpr::Block(
        vec![inner_var, HirStmt::Expression(Box::new(deref_p))],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let block_stmt = HirStmt::Expression(Box::new(inner_block));

    let result = ec.eval_block(&[def_x, def_p, block_stmt]);
    assert!(
        matches!(&result, Ok(ComptimeValue::Int(1))),
        "expected *p == 1 despite shadowing, got {:?}",
        result,
    );
}

#[test]
fn test_eval_pointer_shadowing_write_through() {
    // set x = 1; set p = &mut x; { set x = 2; *p = 10 }
    // After block: x should be 10 (write through pointer), not 2
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // set x = 1
    let def_x = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(1))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set p = &mut x
    let ref_mut_x = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::RefMut,
        expr: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let def_p = HirStmt::VariableDef {
        name: Some("p".into()),
        value: Some(Box::new(ref_mut_x)),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // Inner block: { set x = 2; *p = 10 }
    let inner_var = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(2))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // *p = 10
    let deref_ptr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(HirExpr::Ident("p".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let assign_through_ptr = HirStmt::Assign {
        target: Box::new(deref_ptr),
        value: Box::new(make_int(10)),
        op: None,
        span: Span::new(0, 0),
    };
    let inner_block = HirExpr::Block(
        vec![
            inner_var,
            assign_through_ptr,
            HirStmt::Expression(Box::new(make_int(0))),
        ],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let block_stmt = HirStmt::Expression(Box::new(inner_block));

    let _ = ec.eval_block(&[def_x, def_p, block_stmt]);
    // After the block, outer x should be 10 (written through the pointer)
    assert!(
        matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(10))),
        "expected outer x == 10 after write through pointer, got {:?}",
        get_var(&ec, "x"),
    );
}

#[test]
fn test_eval_pointer_shadowing_nested_blocks() {
    // set x = 1; set p = &x;
    // { set x = 2; { set x = 3; *p } }  →  *p should be 1
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // set x = 1
    let def_x1 = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(1))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set p = &x
    let ref_x = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Ref,
        expr: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let def_p = HirStmt::VariableDef {
        name: Some("p".into()),
        value: Some(Box::new(ref_x)),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // Innermost: { set x = 3; *p }
    let inner_def = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(3))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let deref_p = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(HirExpr::Ident("p".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let innermost = HirExpr::Block(
        vec![inner_def, HirStmt::Expression(Box::new(deref_p))],
        TypeId::NONE,
        Span::new(0, 0),
    );
    // Middle: { set x = 2; <innermost> }
    let mid_def = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(2))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let middle = HirExpr::Block(
        vec![mid_def, HirStmt::Expression(Box::new(innermost))],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let outer_stmt = HirStmt::Expression(Box::new(middle));

    let result = ec.eval_block(&[def_x1, def_p, outer_stmt]);
    assert!(
        matches!(&result, Ok(ComptimeValue::Int(1))),
        "expected *p == 1 through two levels of shadowing, got {:?}",
        result,
    );
}

#[test]
fn test_eval_pointer_shadowing_while() {
    // while i < 3 { set x = 99; i = i + 1 }
    // with outer x and a pointer p = &x taken before the loop.
    // *p should see outer x (unaffected by inner `set x = 99`).
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // set x = 5 (outer)
    let def_x = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int_val(5, int32))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: int32,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set i = 0
    let def_i = HirStmt::VariableDef {
        name: Some("i".into()),
        value: Some(Box::new(make_int_val(0, int32))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: int32,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set p = &x
    let ref_x = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Ref,
        expr: Box::new(HirExpr::Ident("x".into(), int32, Span::new(0, 0))),
        ty: int32,
        span: Span::new(0, 0),
    };
    let def_p = HirStmt::VariableDef {
        name: Some("p".into()),
        value: Some(Box::new(ref_x)),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: int32,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };

    // while body: { set x = 99; i = i + 1 }
    let i = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let shadow_x = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int_val(99, int32))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: int32,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let inc_i = HirStmt::Assign {
        target: Box::new(i()),
        value: Box::new(make_binop_ty(
            i(),
            BinOp::Add,
            make_int_val(1, int32),
            int32,
        )),
        op: None,
        span: Span::new(0, 0),
    };
    let body = vec![shadow_x, inc_i];
    let cond = make_binop_ty(i(), BinOp::Lt, make_int_val(3, int32), int32);
    let while_stmt = HirStmt::While {
        label: None,
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };

    let _ = ec.eval_block(&[def_x, def_i, def_p, while_stmt]);
    // After the loop, outer x should still be 5 (shadow was cleaned up)
    assert!(
        matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(5))),
        "expected outer x == 5 after while loop with shadowing, got {:?}",
        get_var(&ec, "x"),
    );
    // *p should also be 5 (pointer to outer slot)
    let deref_p = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(HirExpr::Ident("p".into(), int32, Span::new(0, 0))),
        ty: int32,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&deref_p);
    assert!(
        matches!(r, Ok(ComptimeValue::Int(5))),
        "expected *p == 5 after while loop, got {:?}",
        r,
    );
}

#[test]
fn test_eval_pointer_shadowing_deref_after_block() {
    // set x = 1; set p = &x; { set x = 2; }  *p  →  should be 1
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // set x = 1
    let def_x = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(1))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // set p = &x
    let ref_x = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Ref,
        expr: Box::new(HirExpr::Ident("x".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let def_p = HirStmt::VariableDef {
        name: Some("p".into()),
        value: Some(Box::new(ref_x)),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    // { set x = 2; }
    let inner_var = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(2))),
        pattern: None,
        else_branch: None,
        kind: VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let block = HirExpr::Block(vec![inner_var], TypeId::NONE, Span::new(0, 0));
    let block_stmt = HirStmt::Expression(Box::new(block));

    // Evaluate: declarations + block
    let _ = ec.eval_block(&[def_x, def_p, block_stmt]);
    // After block, *p should still be 1 (outer x restored, pointer slot valid)
    let deref_p = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Deref,
        expr: Box::new(HirExpr::Ident("p".into(), TypeId::NONE, Span::new(0, 0))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&deref_p);
    assert!(
        matches!(r, Ok(ComptimeValue::Int(1))),
        "expected *p == 1 after block (shadow cleaned up), got {:?}",
        r,
    );
}

// ── Phase 14: Aggregate (struct/tuple/array) tests ────────────────

#[test]
fn test_eval_struct_lit() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::StructLit {
        path: vec![],
        fields: vec![
            ("x".into(), Box::new(make_int(10))),
            ("y".into(), Box::new(make_int(20))),
        ],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok());
    if let Ok(ComptimeValue::Aggregate { fields }) = r {
        assert_eq!(fields.len(), 2);
        assert!(fields[0].0.eq_str("x"));
        assert!(matches!(fields[0].1, ComptimeValue::Int(10)));
        assert!(fields[1].0.eq_str("y"));
        assert!(matches!(fields[1].1, ComptimeValue::Int(20)));
    }
}

#[test]
fn test_eval_tuple() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Tuple(
        vec![make_int(1), make_int(2), make_int(3)],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok());
    if let Ok(ComptimeValue::Aggregate { fields }) = r {
        assert_eq!(fields.len(), 3);
        assert!(fields[0].0.eq_str("_0"));
        assert!(matches!(fields[0].1, ComptimeValue::Int(1)));
    }
}

#[test]
fn test_eval_array() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Array(
        vec![make_int(10), make_int(20), make_int(30)],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok());
    if let Ok(ComptimeValue::Aggregate { fields }) = r {
        assert_eq!(fields.len(), 3);
        assert!(fields[0].0.eq_str("[0]"));
        assert!(matches!(fields[0].1, ComptimeValue::Int(10)));
    }
}

#[test]
fn test_eval_aggregate_field_access() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let struct_expr = HirExpr::StructLit {
        path: vec![],
        fields: vec![
            (
                "name".into(),
                Box::new(HirExpr::Literal(
                    Literal::String("test".into()),
                    TypeId::NONE,
                    Span::new(0, 0),
                )),
            ),
            ("value".into(), Box::new(make_int(42))),
        ],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    // Access field: struct_expr.name
    let field_access = HirExpr::FieldAccess {
        base: Box::new(struct_expr),
        field: "name".into(),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&field_access);
    assert!(matches!(&r, Ok(ComptimeValue::String(s)) if s.as_ref() == "test"));
}

#[test]
fn test_eval_aggregate_index() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let arr_expr = HirExpr::Array(
        vec![make_int(100), make_int(200), make_int(300)],
        TypeId::NONE,
        Span::new(0, 0),
    );
    // Index: arr[1]
    let index = HirExpr::Index {
        base: Box::new(arr_expr),
        index: Box::new(HirExpr::Literal(
            Literal::Int(ast::IntLit::Small(1)),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&index);
    assert!(matches!(r, Ok(ComptimeValue::Int(200))));
}

// ── Phase 15: Type cast tests ─────────────────────────────────────

#[test]
fn test_eval_cast_int_to_int() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let expr = HirExpr::Cast {
        expr: Box::new(make_int_val(42, int32)),
        ty: ctx.int(64, true),
        safe: true,
        rounding: None,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_cast_int_to_float() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let expr = HirExpr::Cast {
        expr: Box::new(make_int_val(42, int32)),
        ty: ctx.float(64),
        safe: true,
        rounding: None,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Float(v)) if (v - 42.0).abs() < 1e-10));
}

// ── Phase 16: TypeAnnotated test ──────────────────────────────────

#[test]
fn test_eval_type_annotated() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::TypeAnnotated {
        expr: Box::new(make_int(42)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 17: EnumLit test ────────────────────────────────────────

#[test]
fn test_eval_enum_lit() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::EnumLit {
        path: vec!["Option".into()],
        variant: "Some".into(),
        payload: Some(Box::new(make_int(42))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok());
    if let Ok(ComptimeValue::Aggregate { fields }) = r {
        assert!(fields[0].0.eq_str("Some"));
        assert!(fields.len() >= 3);
        assert!(matches!(fields[2].1, ComptimeValue::Int(42)));
    }
}

// ── Phase 18: IfLet test ──────────────────────────────────────────

#[test]
fn test_eval_if_let_matches() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let scrutinee = make_int(42);
    let pattern = HirPattern::Wildcard(Span::new(0, 0));
    let then_branch = vec![HirStmt::Expression(Box::new(make_int(1)))];
    let else_branch = vec![HirStmt::Expression(Box::new(make_int(0)))];

    let expr = HirExpr::IfLet {
        pattern,
        scrutinee: Box::new(scrutinee),
        then_branch,
        else_branch: Some(else_branch),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(1))));
}

// ── Phase 19: Match test ──────────────────────────────────────────

#[test]
fn test_eval_match_wildcard() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let scrutinee = make_int(42);
    let arms = vec![HirMatchArm {
        pattern: HirPattern::Wildcard(Span::new(0, 0)),
        guard: None,
        body: Box::new(make_int(99)),
        span: Span::new(0, 0),
    }];
    let expr = HirExpr::Match {
        scrutinee: Box::new(scrutinee),
        arms,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(99))));
}

// ── Phase 20: LayoutOf test ───────────────────────────────────────

#[test]
fn test_eval_layout_of() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Use the correct AST type structure: Type::Generic(Int, [32])
    let path_ty: &'static crate::ast::Type<'static> = Box::leak(Box::new(crate::ast::Type::Path(
        smallvec::smallvec!["Int".into()],
        Span::new(0, 0),
    )));
    let lit_expr: &'static crate::ast::Expr<'static> =
        Box::leak(Box::new(crate::ast::Expr::Literal(
            crate::ast::Literal::Int(ast::IntLit::Small(32)),
            Span::new(0, 0),
        )));
    let expr = HirExpr::LayoutOf(
        Box::new(crate::ast::Type::Generic(
            path_ty,
            vec![crate::ast::GenericArg::Positional(
                crate::ast::Type::Literal(lit_expr, Span::new(0, 0)),
            )],
            Span::new(0, 0),
        )),
        Span::new(0, 0),
    );
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok(), "layout_of!(Int<32>) should succeed: {:?}", r);
    if let Ok(ComptimeValue::LayoutDescriptor(desc)) = r {
        assert_eq!(desc.size, 4, "Int<32> should be 4 bytes");
        assert_eq!(desc.align, 4, "Int<32> should have alignment 4");
    }
}

// ── Phase 29: Nested match test ────────────────────────────────

#[test]
fn test_eval_nested_match() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Build: match (match 42 { _ => 1 }) { _ => 2 }
    let inner_match = HirExpr::Match {
        scrutinee: Box::new(make_int(42)),
        arms: vec![HirMatchArm {
            pattern: HirPattern::Wildcard(Span::new(0, 0)),
            guard: None,
            body: Box::new(make_int(1)),
            span: Span::new(0, 0),
        }],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let outer_match = HirExpr::Match {
        scrutinee: Box::new(inner_match),
        arms: vec![HirMatchArm {
            pattern: HirPattern::Wildcard(Span::new(0, 0)),
            guard: None,
            body: Box::new(make_int(2)),
            span: Span::new(0, 0),
        }],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&outer_match);
    assert!(matches!(r, Ok(ComptimeValue::Int(2))));
}

// ── Phase 21: CompileError test ───────────────────────────────────

#[test]
fn test_eval_compile_error() {
    let mut ctx = TypeContext::new();
    let r = eval(
        &mut ctx,
        &HirExpr::CompileError("test error".into(), Span::new(0, 0)),
    );
    assert!(matches!(r, Err(ComptimeError::AssertionFailed(msg)) if msg == "test error"));
}

// ── Phase 22: Memory limit test ───────────────────────────────────

#[test]
fn test_eval_memory_limit() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    ec.set_memory_limit(1); // Very low limit

    let block = HirStmt::VariableDef {
        name: Some("big".into()),
        value: Some(Box::new(HirExpr::Literal(
            Literal::String("this is a long string that should exceed the memory limit".into()),
            TypeId::NONE,
            Span::new(0, 0),
        ))),
        pattern: None,
        else_branch: None,
        kind: crate::ast::VariableKind::Set,
        ty: TypeId::NONE,
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[block]);
    assert!(matches!(r, Err(ComptimeError::MemoryLimitExceeded(_))));
}

// ── Phase 23: UnsafeBlock test ────────────────────────────────────

#[test]
fn test_eval_unsafe_block() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::UnsafeBlock {
        body: vec![HirStmt::Expression(Box::new(make_int(42)))],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 24: Try expression test ─────────────────────────────────

#[test]
fn test_eval_try_expr() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Try {
        expr: Box::new(make_int(42)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 25: PolyBox/PolyUnbox test ──────────────────────────────

#[test]
fn test_eval_poly_box_unbox() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // poly(42)
    let box_expr = HirExpr::PolyBox {
        expr: Box::new(make_int(42)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&box_expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 26: Zero-param closure test ─────────────────────────────

#[test]
fn test_eval_zero_param_closure() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Closure {
        params: vec![],
        return_type: TypeId::NONE,
        captures: vec![],
        body: vec![HirStmt::Expression(Box::new(make_int(42)))],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_closure_with_params_deferred() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let param = crate::hir::hir::HirParam {
        name: "x".into(),
        ty: TypeId::NONE,
        default: None,
        span: Span::new(0, 0),
    };
    let expr = HirExpr::Closure {
        params: vec![param],
        return_type: TypeId::NONE,
        captures: vec![],
        body: vec![HirStmt::Expression(Box::new(make_int(42)))],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Err(ComptimeError::Deferred)));
}

// ── Phase 27: Pointer index test ──────────────────────────────────

#[test]
fn test_eval_pointer_index() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // Create an array variable
    let arr = ComptimeValue::Aggregate {
        fields: vec![
            (Symbol::intern("[0]"), ComptimeValue::Int(100)),
            (Symbol::intern("[1]"), ComptimeValue::Int(200)),
            (Symbol::intern("[2]"), ComptimeValue::Int(300)),
        ],
    };
    // Create a pointer to arr — capture its slot ID.
    let arr_slot = ec.allocate_slot();
    ec.cur_slot.insert(Symbol::intern("arr"), arr_slot);
    ec.variables.insert(arr_slot, arr.clone());
    let ptr = ComptimeValue::Pointer {
        slot: arr_slot,
        mutable: false,
    };
    insert_var(&mut ec, "ptr", ptr);

    // ptr[1]
    let index = HirExpr::Index {
        base: Box::new(HirExpr::Ident("ptr".into(), TypeId::NONE, Span::new(0, 0))),
        index: Box::new(HirExpr::Literal(
            Literal::Int(ast::IntLit::Small(1)),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&index);
    assert!(matches!(r, Ok(ComptimeValue::Int(200))));
}

// ── Phase 28: Assert built-in test ────────────────────────────────

#[test]
fn test_eval_assert_success() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident(
            "assert".into(),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        args: vec![HirExpr::Literal(
            Literal::Bool(true),
            TypeId::NONE,
            Span::new(0, 0),
        )],
        comptime: true,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(matches!(r, Ok(ComptimeValue::Unit)));
}

#[test]
fn test_eval_assert_failure() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident(
            "assert".into(),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        args: vec![HirExpr::Literal(
            Literal::Bool(false),
            TypeId::NONE,
            Span::new(0, 0),
        )],
        comptime: true,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(matches!(r, Err(ComptimeError::AssertionFailed(_))));
}

// ── Phase 30: Bool and Unit equality tests ─────────────────────

#[test]
fn test_eval_bool_eq() {
    let mut ctx = TypeContext::new();
    let t = HirExpr::Literal(Literal::Bool(true), TypeId::NONE, Span::new(0, 0));
    let f = HirExpr::Literal(Literal::Bool(false), TypeId::NONE, Span::new(0, 0));
    let eq = make_binop(t.clone(), BinOp::Eq, f.clone());
    let neq = make_binop(t.clone(), BinOp::Neq, f.clone());
    let eq2 = make_binop(t.clone(), BinOp::Eq, t.clone());
    assert!(matches!(
        eval(&mut ctx, &eq),
        Ok(ComptimeValue::Bool(false))
    ));
    assert!(matches!(
        eval(&mut ctx, &neq),
        Ok(ComptimeValue::Bool(true))
    ));
    assert!(matches!(
        eval(&mut ctx, &eq2),
        Ok(ComptimeValue::Bool(true))
    ));
}

#[test]
fn test_eval_unit_eq() {
    let mut ctx = TypeContext::new();
    let u = HirExpr::Block(vec![], TypeId::NONE, Span::new(0, 0));
    let eq = make_binop(u.clone(), BinOp::Eq, u.clone());
    let neq = make_binop(u.clone(), BinOp::Neq, u);
    assert!(matches!(eval(&mut ctx, &eq), Ok(ComptimeValue::Bool(true))));
    assert!(matches!(
        eval(&mut ctx, &neq),
        Ok(ComptimeValue::Bool(false))
    ));
}

// ── Phase 31: Unary negation tests ─────────────────────────────

#[test]
fn test_eval_neg_int() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Neg,
        expr: Box::new(make_int(42)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    assert!(matches!(eval(&mut ctx, &expr), Ok(ComptimeValue::Int(-42))));
}

#[test]
fn test_eval_not_bool() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Not,
        expr: Box::new(HirExpr::Literal(
            Literal::Bool(true),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    assert!(matches!(
        eval(&mut ctx, &expr),
        Ok(ComptimeValue::Bool(false))
    ));
}

// ── Phase 32: Variable shadowing tests ─────────────────────────

#[test]
fn test_eval_variable_shadowing() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "x", ComptimeValue::Int(10));

    // Inner block shadows x
    let inner = HirExpr::Block(
        vec![HirStmt::VariableDef {
            name: Some("x".into()),
            pattern: None,
            value: Some(Box::new(make_int(20))),
            else_branch: None,
            kind: crate::ast::VariableKind::Set,
            ty: TypeId::NONE,
            type_captures: vec![],
            mutable: false,
            span: Span::new(0, 0),
        }],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let _ = ec.eval_expr(&inner);
    // Outer x should still be 10 (block scoping prevents leaks)
    assert!(matches!(get_var(&ec, "x"), Some(ComptimeValue::Int(10))));
}

// ── Phase 33: Assign with op tests ─────────────────────────────

// ── Phase 34: Arithmetic edge cases ────────────────────────────
#[test]
fn test_eval_int_min_neg() {
    let mut ctx = TypeContext::new();
    let expr = HirExpr::UnaryOp {
        op: crate::ast::UnaryOp::Neg,
        expr: Box::new(HirExpr::Literal(
            Literal::Int(ast::IntLit::Small(i128::MIN)),
            TypeId::NONE,
            Span::new(0, 0),
        )),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = eval(&mut ctx, &expr);
    assert!(r.is_ok(), "negating i128::MIN should not overflow: {:?}", r);
}

#[test]
fn test_eval_string_neq() {
    let mut ctx = TypeContext::new();
    let a = HirExpr::Literal(Literal::String("abc".into()), TypeId::NONE, Span::new(0, 0));
    let b = HirExpr::Literal(Literal::String("def".into()), TypeId::NONE, Span::new(0, 0));
    let expr = make_binop(a, BinOp::Neq, b);
    assert!(matches!(
        eval(&mut ctx, &expr),
        Ok(ComptimeValue::Bool(true))
    ));
}

#[test]
fn test_eval_float_neq() {
    let mut ctx = TypeContext::new();
    let a = HirExpr::Literal(Literal::Float(3.14), TypeId::NONE, Span::new(0, 0));
    let b = HirExpr::Literal(Literal::Float(2.71), TypeId::NONE, Span::new(0, 0));
    let expr = make_binop(a, BinOp::Neq, b);
    assert!(matches!(
        eval(&mut ctx, &expr),
        Ok(ComptimeValue::Bool(true))
    ));
}

// ── Phase 35: Match pattern tests ──────────────────────────────

#[test]
fn test_eval_match_tuple_pattern() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let scrutinee = HirExpr::Tuple(
        vec![make_int(1), make_int(2)],
        TypeId::NONE,
        Span::new(0, 0),
    );
    let arms = vec![HirMatchArm {
        pattern: HirPattern::Tuple(
            vec![
                HirPattern::Wildcard(Span::new(0, 0)),
                HirPattern::Wildcard(Span::new(0, 0)),
            ],
            Span::new(0, 0),
        ),
        guard: None,
        body: Box::new(make_int(42)),
        span: Span::new(0, 0),
    }];
    let expr = HirExpr::Match {
        scrutinee: Box::new(scrutinee),
        arms,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    assert!(matches!(ec.eval_expr(&expr), Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_match_enum_pattern() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let scrutinee = HirExpr::EnumLit {
        path: vec!["Option".into()],
        variant: "Some".into(),
        payload: Some(Box::new(make_int(42))),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let arms = vec![HirMatchArm {
        pattern: HirPattern::Enum {
            path: vec!["Option".into()],
            variant: "Some".into(),
            inner: Some(Box::new(HirPattern::Wildcard(Span::new(0, 0)))),
            span: Span::new(0, 0),
        },
        guard: None,
        body: Box::new(make_int(99)),
        span: Span::new(0, 0),
    }];
    let expr = HirExpr::Match {
        scrutinee: Box::new(scrutinee),
        arms,
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    assert!(matches!(ec.eval_expr(&expr), Ok(ComptimeValue::Int(99))));
}

// ── Phase 36: Catch expression test ────────────────────────────

#[test]
fn test_eval_catch_expr() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Catch {
        expr: Box::new(make_int(42)),
        branches: vec![],
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    assert!(matches!(ec.eval_expr(&expr), Ok(ComptimeValue::Int(42))));
}

// ── Phase 37: LeaveWith and Await are errors ───────────────────

#[test]
fn test_eval_leave_with_error() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::LeaveWith {
        expr: Box::new(make_int(0)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Err(ComptimeError::NotComptimeAllowed(_))));
}

#[test]
fn test_eval_await_error() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::Await {
        expr: Box::new(make_int(0)),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Err(ComptimeError::NotComptimeAllowed(_))));
}

// ── Phase 38: AttrAccess test ──────────────────────────────────

#[test]
fn test_eval_attr_access_default() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let expr = HirExpr::AttrAccess {
        base: Box::new(make_int(42)),
        attr: "default".into(),
        ty: TypeId::NONE,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

// ── Phase 39: While with netsted loop ──────────────────────────

#[test]
fn test_eval_while_neested() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    insert_var(&mut ec, "i", ComptimeValue::Int(0));
    insert_var(&mut ec, "j", ComptimeValue::Int(0));
    ec.memory_used = ComptimeValue::Int(0).memory_size() + ComptimeValue::Int(0).memory_size();

    let i = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let j = || HirExpr::Ident("j".into(), int32, Span::new(0, 0));
    let outer_cond = make_binop_ty(i(), BinOp::Lt, make_int_val(3, int32), int32);
    let inner_cond = make_binop_ty(j(), BinOp::Lt, make_int_val(2, int32), int32);
    let inner_body = vec![HirStmt::Assign {
        target: Box::new(j()),
        value: Box::new(make_binop_ty(
            j(),
            BinOp::Add,
            make_int_val(1, int32),
            int32,
        )),
        op: None,
        span: Span::new(0, 0),
    }];
    let inner_while = HirStmt::While {
        label: None,
        cond: Box::new(inner_cond),
        body: inner_body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let outer_body = vec![
        inner_while,
        HirStmt::Assign {
            target: Box::new(i()),
            value: Box::new(make_binop_ty(
                i(),
                BinOp::Add,
                make_int_val(1, int32),
                int32,
            )),
            op: None,
            span: Span::new(0, 0),
        },
    ];
    let outer_while = HirStmt::While {
        label: None,
        cond: Box::new(outer_cond),
        body: outer_body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_block(&[outer_while]);
    assert!(matches!(get_var(&ec, "i"), Some(ComptimeValue::Int(3))));
}

// ── Self-recursion test ────────────────────────────────────────

#[test]
fn test_eval_self_recursion() {
    // comptime def fact(n) -> Int<64> {
    //     if n <= 1 { 1 } else { n * fact!(n - 1) }
    // }
    // fact!(5) should return 120
    let mut ctx = TypeContext::new();
    let int64 = ctx.int(64, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    let n_param = HirExpr::Ident("n".into(), int64, Span::new(0, 0));
    let cond = make_binop_ty(n_param.clone(), BinOp::Le, make_int_val(1, int64), int64);
    let recursive_call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("fact".into(), int64, Span::new(0, 0))),
        args: vec![make_binop_ty(
            n_param.clone(),
            BinOp::Sub,
            make_int_val(1, int64),
            int64,
        )],
        comptime: true,
        ty: int64,
        span: Span::new(0, 0),
    };
    let if_expr = HirExpr::If {
        cond: Box::new(cond),
        then_branch: vec![HirStmt::Expression(Box::new(make_int_val(1, int64)))],
        else_branch: Some(vec![HirStmt::Expression(Box::new(make_binop_ty(
            n_param.clone(),
            BinOp::Mul,
            recursive_call,
            int64,
        )))]),
        is_expression: true,
        ty: int64,
        span: Span::new(0, 0),
    };
    let fn_body = vec![HirStmt::Expression(Box::new(if_expr))];
    ec.register_fn("fact".into(), vec!["n".into()], fn_body);

    // fact!(5) = 5 * 4 * 3 * 2 * 1 = 120
    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("fact".into(), int64, Span::new(0, 0))),
        args: vec![make_int_val(5, int64)],
        comptime: true,
        ty: int64,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(
        matches!(r, Ok(ComptimeValue::Int(120))),
        "fact!(5) = 120, got {:?}",
        r
    );
}

// ── Mutual recursion test ──────────────────────────────────────

#[test]
fn test_eval_mutual_recursion() {
    // comptime def is_even(n) -> Bool {
    //     if n == 0 { true } else { is_odd!(n - 1) }
    // }
    // comptime def is_odd(n) -> Bool {
    //     if n == 0 { false } else { is_even!(n - 1) }
    // }
    // is_even!(4) should return true
    let mut ctx = TypeContext::new();
    let int64 = ctx.int(64, true);
    let bool_ty = ctx.bool();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);

    // is_even body: if n == 0 { true } else { is_odd!(n - 1) }
    let n_even = HirExpr::Ident("n".into(), int64, Span::new(0, 0));
    let cond_even = make_binop_ty(n_even.clone(), BinOp::Eq, make_int_val(0, int64), bool_ty);
    let is_odd_call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("is_odd".into(), bool_ty, Span::new(0, 0))),
        args: vec![make_binop_ty(
            n_even.clone(),
            BinOp::Sub,
            make_int_val(1, int64),
            int64,
        )],
        comptime: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    };
    let even_body = vec![HirStmt::Expression(Box::new(HirExpr::If {
        cond: Box::new(cond_even),
        then_branch: vec![HirStmt::Expression(Box::new(HirExpr::Literal(
            crate::ast::Literal::Bool(true),
            bool_ty,
            Span::new(0, 0),
        )))],
        else_branch: Some(vec![HirStmt::Expression(Box::new(is_odd_call))]),
        is_expression: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    }))];

    // is_odd body: if n == 0 { false } else { is_even!(n - 1) }
    let n_odd = HirExpr::Ident("n".into(), int64, Span::new(0, 0));
    let cond_odd = make_binop_ty(n_odd.clone(), BinOp::Eq, make_int_val(0, int64), bool_ty);
    let is_even_call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("is_even".into(), bool_ty, Span::new(0, 0))),
        args: vec![make_binop_ty(
            n_odd.clone(),
            BinOp::Sub,
            make_int_val(1, int64),
            int64,
        )],
        comptime: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    };
    let odd_body = vec![HirStmt::Expression(Box::new(HirExpr::If {
        cond: Box::new(cond_odd),
        then_branch: vec![HirStmt::Expression(Box::new(HirExpr::Literal(
            crate::ast::Literal::Bool(false),
            bool_ty,
            Span::new(0, 0),
        )))],
        else_branch: Some(vec![HirStmt::Expression(Box::new(is_even_call))]),
        is_expression: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    }))];

    ec.register_fn("is_even".into(), vec!["n".into()], even_body);
    ec.register_fn("is_odd".into(), vec!["n".into()], odd_body);

    // is_even!(4) = is_odd!(3) = is_even!(2) = is_odd!(1) = is_even!(0) = true
    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("is_even".into(), bool_ty, Span::new(0, 0))),
        args: vec![make_int_val(4, int64)],
        comptime: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(
        matches!(r, Ok(ComptimeValue::Bool(true))),
        "is_even!(4) = true, got {:?}",
        r
    );

    // is_odd!(3) = is_even!(2) = is_odd!(1) = is_even!(0) = true
    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("is_odd".into(), bool_ty, Span::new(0, 0))),
        args: vec![make_int_val(3, int64)],
        comptime: true,
        ty: bool_ty,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(
        matches!(r, Ok(ComptimeValue::Bool(true))),
        "is_odd!(3) = true, got {:?}",
        r
    );
}

#[test]
/// Verify that deeply recursive comptime function calls trigger
/// StepLimitExceeded (not a stack overflow or infinite hang).
fn test_comptime_recursive_step_limit() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut diag = crate::diagnostics::DiagCtxt::new();
    let int32 = ctx.int(32, true);
    let mut ec = ComptimeEvalContext::new(&mut ctx, &symbols, &mut diag);
    // Set a small step limit — far below what the unbounded recursion needs.
    ec.set_step_limit(100);
    let _ = int32;

    // def recurse(n: Int<32>) -> Int<32> { recurse!(n + 1) }
    let self_call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("recurse".into(), int32, Span::new(0, 0))),
        args: vec![HirExpr::BinaryOp {
            left: Box::new(HirExpr::Ident("n".into(), int32, Span::new(0, 0))),
            op: BinOp::Add,
            right: Box::new(make_int_val(1, int32)),
            ty: int32,
            span: Span::new(0, 0),
        }],
        comptime: true,
        ty: int32,
        span: Span::new(0, 0),
    };
    let fn_body = vec![HirStmt::Expression(Box::new(self_call))];
    ec.register_fn("recurse".into(), vec!["n".into()], fn_body);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("recurse".into(), int32, Span::new(0, 0))),
        args: vec![make_int_val(0, int32)],
        comptime: true,
        ty: int32,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(
        matches!(r, Err(ComptimeError::StepLimitExceeded)),
        "recursive comptime call should hit StepLimitExceeded, got {:?}",
        r
    );
}
