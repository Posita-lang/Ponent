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
    let mut ec = ComptimeEvalContext::new(ctx);
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
    let mut ec = ComptimeEvalContext::new(&ctx);
    ec.set_step_limit(0);
    let r = ec.eval_expr(&make_int(42));
    assert!(matches!(r, Err(ComptimeError::StepLimitExceeded)));
}
