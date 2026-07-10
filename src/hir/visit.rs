//! HIR walker. Follows the same pattern as `ast::visit`.
//! Each overridden visit method has full control; the default calls `walk_*`.

use crate::ast::visit::VisitorResult;
use crate::ast::*;
use crate::hir::hir::*;
use crate::hir::types::TypeId;

/// HIR visitor. Default implementations call `walk_*` for recursive traversal.
pub trait HirVisitor {
    type Result;

    fn visit_hir_expr(&mut self, expr: &HirExpr) -> Self::Result {
        walk_hir_expr(self, expr)
    }
    fn visit_hir_stmt(&mut self, stmt: &HirStmt) -> Self::Result {
        walk_hir_stmt(self, stmt)
    }
    fn visit_hir_pattern(&mut self, pat: &HirPattern) -> Self::Result {
        walk_hir_pattern(self, pat)
    }
    fn visit_hir_param(&mut self, param: &HirParam) -> Self::Result {
        walk_hir_param(self, param)
    }
    fn visit_type_id(&mut self, _ty: TypeId) -> Self::Result {
        Self::Result::output()
    }
    fn visit_ident(&mut self, _name: &str) -> Self::Result {
        Self::Result::output()
    }
    fn visit_literal(&mut self, _lit: &Literal) -> Self::Result {
        Self::Result::output()
    }
}

// ── Walk functions ───────────────────────────────────────────────

pub fn walk_hir_expr<V: HirVisitor>(visitor: &mut V, expr: &HirExpr) -> V::Result {
    match expr {
        HirExpr::Literal(lit, ty, _) => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_literal(lit)
        }
        HirExpr::Ident(name, ty, _) => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_ident(name)
        }
        HirExpr::TypeAnnotated { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(e)
        }
        HirExpr::BinaryOp { left, right, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(left)?;
            visitor.visit_hir_expr(right)
        }
        HirExpr::UnaryOp { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(e)
        }
        HirExpr::Call { callee, args, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(callee)?;
            for a in args { visitor.visit_hir_expr(a)?; }
            V::Result::output()
        }
        HirExpr::Index { base, index, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(base)?;
            visitor.visit_hir_expr(index)
        }
        HirExpr::FieldAccess { base, field, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(base)?;
            visitor.visit_ident(field)
        }
        HirExpr::AttrAccess { base, attr, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(base)?;
            visitor.visit_ident(attr)
        }
        HirExpr::Cast { expr: base, ty, .. }
        | HirExpr::Deref { expr: base, ty, .. }
        | HirExpr::Ref { expr: base, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(base)
        }
        HirExpr::Range { start, end, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            if let Some(s) = start { visitor.visit_hir_expr(s)?; }
            if let Some(e) = end { visitor.visit_hir_expr(e)?; }
            V::Result::output()
        }
        HirExpr::StructLit { fields, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            for (name, f) in fields {
                visitor.visit_ident(name)?;
                visitor.visit_hir_expr(f)?;
            }
            V::Result::output()
        }
        HirExpr::EnumLit { payload, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            if let Some(p) = payload { visitor.visit_hir_expr(p) }
            else { V::Result::output() }
        }
        HirExpr::Move(e, ty, _) | HirExpr::Await { expr: e, ty, .. }
        | HirExpr::Try { expr: e, ty, .. } | HirExpr::LeaveWith { expr: e, ty, .. }
        | HirExpr::PolyBox { expr: e, ty, .. } | HirExpr::PolyUnbox { expr: e, ty, .. }
        | HirExpr::Old { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(e)
        }
        HirExpr::Tuple(exprs, ty, _) | HirExpr::Array(exprs, ty, _) => {
            visitor.visit_type_id(*ty)?;
            for e in exprs { visitor.visit_hir_expr(e)?; }
            V::Result::output()
        }
        HirExpr::Closure { params, body, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            for p in params { visitor.visit_hir_param(p)?; }
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirExpr::UnsafeBlock { body, ty, .. } | HirExpr::Block(body, ty, _) => {
            visitor.visit_type_id(*ty)?;
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirExpr::Catch { expr: e, branches, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(e)?;
            for b in branches {
                if let Some(ref p) = b.pattern { visitor.visit_hir_pattern(p)?; }
                for s in &b.body { visitor.visit_hir_stmt(s)?; }
            }
            V::Result::output()
        }
        HirExpr::If { cond, then_branch, else_branch, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(cond)?;
            for s in then_branch { visitor.visit_hir_stmt(s)?; }
            if let Some(eb) = else_branch {
                for s in eb { visitor.visit_hir_stmt(s)?; }
            }
            V::Result::output()
        }
        HirExpr::IfLet { pattern, scrutinee, then_branch, else_branch, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(scrutinee)?;
            visitor.visit_hir_pattern(pattern)?;
            for s in then_branch { visitor.visit_hir_stmt(s)?; }
            if let Some(eb) = else_branch {
                for s in eb { visitor.visit_hir_stmt(s)?; }
            }
            V::Result::output()
        }
        HirExpr::Match { scrutinee, arms, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(scrutinee)?;
            for arm in arms {
                visitor.visit_hir_pattern(&arm.pattern)?;
                if let Some(ref g) = arm.guard { visitor.visit_hir_expr(g)?; }
                visitor.visit_hir_expr(&arm.body)?;
            }
            V::Result::output()
        }
        HirExpr::Quantified { range, body, ty, .. } => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_hir_expr(range)?;
            visitor.visit_hir_expr(body)
        }
        HirExpr::Error(_) => V::Result::output(),
        HirExpr::TypeInfo(ty, _) => {
            visitor.visit_type_id(*ty)?;
            V::Result::output()
        }
    }
}

pub fn walk_hir_stmt<V: HirVisitor>(visitor: &mut V, stmt: &HirStmt) -> V::Result {
    match stmt {
        HirStmt::VariableDef { value, pattern, else_branch, .. } => {
            if let Some(e) = value { visitor.visit_hir_expr(e)?; }
            if let Some(p) = pattern { visitor.visit_hir_pattern(p)?; }
            if let Some(eb) = else_branch {
                for s in eb { visitor.visit_hir_stmt(s)?; }
            }
            V::Result::output()
        }
        HirStmt::FunctionDef { name, params, body, finally, .. } => {
            visitor.visit_ident(name)?;
            for p in params { visitor.visit_hir_param(p)?; }
            if let Some(b) = body { for s in b { visitor.visit_hir_stmt(s)?; } }
            if let Some(f) = finally { for s in f { visitor.visit_hir_stmt(s)?; } }
            V::Result::output()
        }
        HirStmt::Expression(expr) => visitor.visit_hir_expr(expr),
        HirStmt::If { cond, then_branch, else_branch, .. } => {
            visitor.visit_hir_expr(cond)?;
            for s in then_branch { visitor.visit_hir_stmt(s)?; }
            if let Some(eb) = else_branch { for s in eb { visitor.visit_hir_stmt(s)?; } }
            V::Result::output()
        }
        HirStmt::IfLet { pattern, scrutinee, then_branch, else_branch, .. } => {
            visitor.visit_hir_expr(scrutinee)?;
            visitor.visit_hir_pattern(pattern)?;
            for s in then_branch { visitor.visit_hir_stmt(s)?; }
            if let Some(eb) = else_branch { for s in eb { visitor.visit_hir_stmt(s)?; } }
            V::Result::output()
        }
        HirStmt::While { cond, body, invariant, decreases, .. } => {
            visitor.visit_hir_expr(cond)?;
            if let Some(ref i) = invariant { visitor.visit_hir_expr(i)?; }
            if let Some(ref d) = decreases { visitor.visit_hir_expr(d)?; }
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirStmt::WhileLet { pattern, scrutinee, body, .. } => {
            visitor.visit_hir_expr(scrutinee)?;
            visitor.visit_hir_pattern(pattern)?;
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirStmt::For { pattern, iterable, body, .. } => {
            visitor.visit_hir_expr(iterable)?;
            visitor.visit_hir_pattern(pattern)?;
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirStmt::Loop { body, .. } | HirStmt::ComptimeBlock { body, .. }
        | HirStmt::ScopeCleanup { body, .. } | HirStmt::Unsafe { body, .. }
        | HirStmt::Isolate { body, .. } => {
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
        HirStmt::Return { value, .. } => {
            if let Some(e) = value { visitor.visit_hir_expr(e) }
            else { V::Result::output() }
        }
        HirStmt::Assign { target, value, .. } => {
            visitor.visit_hir_expr(target)?;
            visitor.visit_hir_expr(value)
        }
        HirStmt::GhostVariableDef { inner, .. } => visitor.visit_hir_stmt(inner),
        HirStmt::Leave { .. } | HirStmt::Continue { .. } | HirStmt::Trigger { .. }
        | HirStmt::Edition(..) | HirStmt::Error => V::Result::output(),
        HirStmt::TypeDef { .. } | HirStmt::TraitDef { .. } | HirStmt::Import { .. }
        | HirStmt::ExternFunction { .. } | HirStmt::Constraint { .. }
        | HirStmt::ImplBlock { .. } => V::Result::output(),
        HirStmt::Generate { body, .. } => {
            for s in body { visitor.visit_hir_stmt(s)?; }
            V::Result::output()
        }
    }
}

pub fn walk_hir_pattern<V: HirVisitor>(visitor: &mut V, pat: &HirPattern) -> V::Result {
    match pat {
        HirPattern::Wildcard(_) | HirPattern::Error(_) => V::Result::output(),
        HirPattern::Ident(name, ty, _) => {
            visitor.visit_type_id(*ty)?;
            visitor.visit_ident(name)
        }
        HirPattern::Literal(expr, _) => visitor.visit_hir_expr(expr),
        HirPattern::Tuple(patterns, _) => {
            for p in patterns {
                visitor.visit_hir_pattern(p)?;
            }
            V::Result::output()
        }
        HirPattern::Slice(before, rest, after, _) => {
            for p in before {
                visitor.visit_hir_pattern(p)?;
            }
            if let Some(r) = rest {
                visitor.visit_hir_pattern(r)?;
            }
            for p in after {
                visitor.visit_hir_pattern(p)?;
            }
            V::Result::output()
        }
        HirPattern::Struct { fields, .. } => {
            for (_, p) in fields { visitor.visit_hir_pattern(p)?; }
            V::Result::output()
        }
        HirPattern::Enum { inner, .. } => {
            if let Some(p) = inner { visitor.visit_hir_pattern(p) }
            else { V::Result::output() }
        }
        HirPattern::Or(patterns, _) => {
            for p in patterns { visitor.visit_hir_pattern(p)?; }
            V::Result::output()
        }
    }
}

pub fn walk_hir_param<V: HirVisitor>(visitor: &mut V, param: &HirParam) -> V::Result {
    visitor.visit_ident(&param.name)?;
    visitor.visit_type_id(param.ty)
}
