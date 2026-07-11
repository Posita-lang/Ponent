use super::*;
use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::types::{TypeContext, TypeId};
use crate::ast::{BinOp, Literal, Span};

fn make_int_val(n: i128, ty: TypeId) -> HirExpr {
    HirExpr::Literal(Literal::Int(n), ty, Span::new(0, 0))
}

fn make_int(n: i128) -> HirExpr {
    make_int_val(n, TypeId(0))
}

fn make_bool(b: bool) -> HirExpr {
    HirExpr::Literal(Literal::Bool(b), TypeId(0), Span::new(0, 0))
}

fn make_binop_ty(l: HirExpr, op: BinOp, r: HirExpr, ty: TypeId) -> HirExpr {
    HirExpr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
        ty,
        span: Span::new(0, 0),
    }
}

fn make_binop(l: HirExpr, op: BinOp, r: HirExpr) -> HirExpr {
    make_binop_ty(l, op, r, TypeId(0))
}

/// Create an Int<32> type and wrap a value in a Literal with that type.
fn make_int32(ctx: &mut TypeContext, n: i128) -> HirExpr {
    let int32 = ctx.int(32, true);
    make_int_val(n, int32)
}

/// Create a BinaryOp with Int<32> as the result type.
fn make_binop32(ctx: &mut TypeContext, l: HirExpr, op: BinOp, r: HirExpr) -> HirExpr {
    let int32 = ctx.int(32, true);
    make_binop_ty(l, op, r, int32)
}

fn make_block(stmts: Vec<HirStmt>, last: HirExpr) -> HirExpr {
    let mut all = stmts;
    all.push(HirStmt::Expression(Box::new(last)));
    HirExpr::Block(all, TypeId(0), Span::new(0, 0))
}

fn make_if(cond: HirExpr, then: HirExpr, els: Option<HirExpr>) -> HirExpr {
    let then_block = vec![HirStmt::Expression(Box::new(then))];
    let else_block = els.map(|e| vec![HirStmt::Expression(Box::new(e))]);
    HirExpr::If {
        cond: Box::new(cond),
        then_branch: then_block,
        else_branch: else_block,
        is_expression: true,
        ty: TypeId(0),
        span: Span::new(0, 0),
    }
}

fn eval(ctx: &mut TypeContext, expr: &HirExpr) -> Result<ComptimeValue, ComptimeError> {
    use crate::hir::types::{CrateId, DefId};
    use crate::hir::symbol::SymbolTable;
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut ec = ComptimeEvalContext::new(ctx, &symbols);
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
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);
    ec.set_step_limit(0);
    let r = ec.eval_expr(&make_int(42));
    assert!(matches!(r, Err(ComptimeError::StepLimitExceeded)));
}

// ── Phase 2: Variable binding tests ────────────────────────────────

#[test]
fn test_eval_variable_def() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    let block = HirStmt::VariableDef {
        name: Some("x".into()),
        value: Some(Box::new(make_int(42))),
        pattern: None,
        else_branch: None,
        kind: crate::ast::VariableKind::Set,
        ty: TypeId(0),
        type_captures: vec![],
        mutable: false,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[block]);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
    assert!(matches!(ec.variables.get("x"), Some(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_variable_assign() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);
    ec.variables.insert("x".into(), ComptimeValue::Int(10));

    let assign = HirStmt::Assign {
        target: Box::new(HirExpr::Ident("x".into(), TypeId(0), Span::new(0, 0))),
        value: Box::new(make_int(20)),
        op: None,
        span: Span::new(0, 0),
    };
    let r = ec.eval_block(&[assign]);
    assert!(matches!(r, Ok(ComptimeValue::Int(20))));
    assert!(matches!(ec.variables.get("x"), Some(ComptimeValue::Int(20))));
}

#[test]
fn test_eval_assign_unknown_variable() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    let assign = HirStmt::Assign {
        target: Box::new(HirExpr::Ident("nonexistent".into(), TypeId(0), Span::new(0, 0))),
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
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);
    ec.variables.insert("x".into(), ComptimeValue::Int(99));

    let expr = HirExpr::Ident("x".into(), TypeId(0), Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(99))));
}

#[test]
fn test_eval_ident_unknown() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    let expr = HirExpr::Ident("unknown".into(), TypeId(0), Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(matches!(r, Err(ComptimeError::UnknownIdentifier(_))));
}

// ── Phase 4: Function call tests ───────────────────────────────────

#[test]
fn test_eval_comptime_fn_call() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    // Register a comptime function: def double(x: Int<32>) -> Int<32> { 2 * x }
    let body = vec![HirStmt::Expression(Box::new(
        make_binop_ty(
            make_int_val(2, int32),
            BinOp::Mul,
            HirExpr::Ident("x".into(), int32, Span::new(0, 0)),
            int32,
        )
    ))];
    ec.register_fn("double".into(), vec!["x".into()], body);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("double".into(), int32, Span::new(0, 0))),
        args: vec![make_int_val(21, int32)],
        comptime: true,
        ty: int32,
        span: Span::new(0, 0),
    };
    let r = ec.eval_expr(&call);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))), "double(21) = {:?}", r);
}

#[test]
fn test_eval_comptime_fn_call_unknown() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("undefined_fn".into(), TypeId(0), Span::new(0, 0))),
        args: vec![],
        comptime: true,
        ty: TypeId(0),
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
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    ec.variables.insert("i".into(), ComptimeValue::Int(0));
    // Simulate a while loop: while i < 5 { i = i + 1 }
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(5, int32), int32);
    let body = vec![HirStmt::Assign {
        target: Box::new(i_expr()),
        value: Box::new(make_binop_ty(i_expr(), BinOp::Add, make_int_val(1, int32), int32)),
        op: None,
        span: Span::new(0, 0),
    }];

    let while_stmt = HirStmt::While {
        cond: Box::new(cond),
        body,
        invariant: None,
        decreases: None,
        span: Span::new(0, 0),
    };
    let _ = ec.eval_block(&[while_stmt]);
    // After the loop, i should be 5
    assert!(matches!(ec.variables.get("i"), Some(ComptimeValue::Int(5))));
}

#[test]
fn test_eval_while_step_limit() {
    let mut ctx = TypeContext::new();
    let int32 = ctx.int(32, true);
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);
    ec.set_step_limit(5);

    ec.variables.insert("i".into(), ComptimeValue::Int(0));
    let i_expr = || HirExpr::Ident("i".into(), int32, Span::new(0, 0));
    let cond = make_binop_ty(i_expr(), BinOp::Lt, make_int_val(100, int32), int32);
    let body = vec![HirStmt::Assign {
        target: Box::new(i_expr()),
        value: Box::new(make_binop_ty(i_expr(), BinOp::Add, make_int_val(1, int32), int32)),
        op: None,
        span: Span::new(0, 0),
    }];

    let while_stmt = HirStmt::While {
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
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);

    let expr = HirExpr::TypeInfo(int32, Span::new(0, 0));
    let r = ec.eval_expr(&expr);
    assert!(r.is_ok() && matches!(r.unwrap(), ComptimeValue::Type(t) if t == int32));
}

// ── Phase 9: Variable scope isolation test ─────────────────────────

#[test]
fn test_eval_fn_call_scope_isolation() {
    let mut ctx = TypeContext::new();
    let symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(crate::hir::types::DefId(0)));
    let mut ec = ComptimeEvalContext::new(&ctx, &symbols);
    ec.variables.insert("x".into(), ComptimeValue::Int(1));

    // Register a function that assigns to its own param, not the outer scope
    let body = vec![HirStmt::Assign {
        target: Box::new(HirExpr::Ident("x".into(), TypeId(0), Span::new(0, 0))),
        value: Box::new(make_int(99)),
        op: None,
        span: Span::new(0, 0),
    }];
    ec.register_fn("mutate_x".into(), vec!["x".into()], body);

    let call = HirExpr::Call {
        callee: Box::new(HirExpr::Ident("mutate_x".into(), TypeId(0), Span::new(0, 0))),
        args: vec![make_int(10)],
        comptime: true,
        ty: TypeId(0),
        span: Span::new(0, 0),
    };
    let _ = ec.eval_expr(&call);
    // Outer x should still be 1, not 99
    assert!(matches!(ec.variables.get("x"), Some(ComptimeValue::Int(1))));
}
