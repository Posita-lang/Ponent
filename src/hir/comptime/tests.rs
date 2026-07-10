use super::*;
use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::types::{TypeContext, TypeId};
use crate::ast::{BinOp, Literal, Span};

fn make_int(n: i64) -> HirExpr {
    HirExpr::Literal(Literal::Int(n), TypeId(999), Span::new(0, 0))
}

fn make_bool(b: bool) -> HirExpr {
    HirExpr::Literal(Literal::Bool(b), TypeId(999), Span::new(0, 0))
}

fn make_binop(l: HirExpr, op: BinOp, r: HirExpr) -> HirExpr {
    HirExpr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
        ty: TypeId(999),
        span: Span::new(0, 0),
    }
}

fn make_block(stmts: Vec<HirStmt>, last: HirExpr) -> HirExpr {
    let mut all = stmts;
    all.push(HirStmt::Expression(Box::new(last)));
    HirExpr::Block(all, TypeId(999), Span::new(0, 0))
}

fn make_if(cond: HirExpr, then: HirExpr, els: Option<HirExpr>) -> HirExpr {
    let then_block = vec![HirStmt::Expression(Box::new(then))];
    let else_block = els.map(|e| vec![HirStmt::Expression(Box::new(e))]);
    HirExpr::If {
        cond: Box::new(cond),
        then_branch: then_block,
        else_branch: else_block,
        is_expression: true,
        ty: TypeId(999),
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
    let expr = make_binop(make_int(3), BinOp::Add, make_int(4));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(7))));
}

#[test]
fn test_eval_sub() {
    let mut ctx = TypeContext::new();
    let expr = make_binop(make_int(10), BinOp::Sub, make_int(3));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(7))));
}

#[test]
fn test_eval_mul() {
    let mut ctx = TypeContext::new();
    let expr = make_binop(make_int(6), BinOp::Mul, make_int(7));
    let r = eval(&mut ctx, &expr);
    assert!(matches!(r, Ok(ComptimeValue::Int(42))));
}

#[test]
fn test_eval_div() {
    let mut ctx = TypeContext::new();
    let expr = make_binop(make_int(10), BinOp::Div, make_int(3));
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
fn test_eval_nested_arith() {
    let mut ctx = TypeContext::new();
    let expr = make_binop(
        make_binop(make_int(1), BinOp::Add, make_int(2)),
        BinOp::Mul,
        make_int(3),
    );
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
