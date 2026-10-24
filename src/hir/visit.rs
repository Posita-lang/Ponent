//! HIR walker. Follows the same pattern as `ast::visit`.
//! Each overridden visit method has full control; the default calls `walk_*`.

use crate::ast::visit::VisitorResult;
use crate::ast::*;
use crate::hir::hir::*;
use crate::hir::types::TypeId;
use crate::symbol::Symbol;

/// HIR visitor. Default implementations call `walk_*` for recursive traversal.
pub trait HirVisitor<'hir> {
    fn visit_hir_expr(&mut self, expr: &HirExpr<'hir>)
    where
        Self: Sized,
    {
        walk_hir_expr(self, expr)
    }
    fn visit_hir_stmt(&mut self, stmt: &HirStmt<'hir>)
    where
        Self: Sized,
    {
        walk_hir_stmt(self, stmt)
    }
    fn visit_hir_pattern(&mut self, pat: &HirPattern<'hir>)
    where
        Self: Sized,
    {
        walk_hir_pattern(self, pat)
    }
    fn visit_hir_param(&mut self, param: &HirParam<'hir>)
    where
        Self: Sized,
    {
        walk_hir_param(self, param)
    }
    fn visit_type_id(&mut self, _ty: TypeId) -> () {
        ()
    }
    fn visit_ident(&mut self, _name: &Symbol) -> () {
        ()
    }
    fn visit_literal(&mut self, _lit: &Literal) -> () {
        ()
    }
}

// ── Walk functions ───────────────────────────────────────────────

pub fn walk_hir_expr<'hir, V: HirVisitor<'hir>>(visitor: &mut V, expr: &HirExpr<'hir>) -> () {
    match expr {
        HirExpr::Literal(lit, ty, _) => {
            visitor.visit_type_id(*ty);
            visitor.visit_literal(lit)
        }
        HirExpr::Ident(name, ty, _) => {
            visitor.visit_type_id(*ty);
            visitor.visit_ident(name)
        }
        HirExpr::TypeAnnotated { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(e)
        }
        HirExpr::BinaryOp {
            left, right, ty, ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(left);
            visitor.visit_hir_expr(right)
        }
        HirExpr::UnaryOp { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(e)
        }
        HirExpr::Call {
            callee, args, ty, ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(callee);
            for a in args {
                visitor.visit_hir_expr(a);
            }
            ()
        }
        HirExpr::Index {
            base, index, ty, ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(base);
            visitor.visit_hir_expr(index)
        }
        HirExpr::FieldAccess {
            base, field, ty, ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(base);
            visitor.visit_ident(field)
        }
        HirExpr::AttrAccess { base, attr, ty, .. } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(base);
            visitor.visit_ident(attr)
        }
        HirExpr::Cast { expr: base, ty, .. } | HirExpr::UnaryOp { expr: base, ty, .. } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(base)
        }
        HirExpr::Range { start, end, ty, .. } => {
            visitor.visit_type_id(*ty);
            if let Some(s) = start {
                visitor.visit_hir_expr(s);
            }
            if let Some(e) = end {
                visitor.visit_hir_expr(e);
            }
            ()
        }
        HirExpr::StructLit { fields, ty, .. } => {
            visitor.visit_type_id(*ty);
            for (name, f) in fields {
                visitor.visit_ident(name);
                visitor.visit_hir_expr(f);
            }
            ()
        }
        HirExpr::EnumLit {
            variant,
            payload,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_ident(variant);
            if let Some(p) = payload {
                visitor.visit_hir_expr(p)
            } else {
                ()
            }
        }
        HirExpr::Move(e, ty, _)
        | HirExpr::Await { expr: e, ty, .. }
        | HirExpr::Try { expr: e, ty, .. }
        | HirExpr::LeaveWith { expr: e, ty, .. }
        | HirExpr::PolyBox { expr: e, ty, .. }
        | HirExpr::PolyUnbox { expr: e, ty, .. }
        | HirExpr::Old { expr: e, ty, .. } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(e)
        }
        HirExpr::Tuple(exprs, ty, _) | HirExpr::Array(exprs, ty, _) => {
            visitor.visit_type_id(*ty);
            for e in exprs {
                visitor.visit_hir_expr(e);
            }
            ()
        }
        HirExpr::Closure {
            params, body, ty, ..
        } => {
            visitor.visit_type_id(*ty);
            for p in params {
                visitor.visit_hir_param(p);
            }
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirExpr::UnsafeBlock { body, ty, .. } | HirExpr::Block(body, ty, _) => {
            visitor.visit_type_id(*ty);
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirExpr::Return { value, .. } => {
            visitor.visit_hir_expr(value.as_ref());
        }
        HirExpr::Catch {
            expr: e,
            branches,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(e);
            for b in branches {
                visitor.visit_hir_pattern(&b.pattern);
                for s in &b.body {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(cond);
            for s in then_branch {
                visitor.visit_hir_stmt(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirExpr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(scrutinee);
            visitor.visit_hir_pattern(pattern);
            for s in then_branch {
                visitor.visit_hir_stmt(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirExpr::Match {
            scrutinee,
            arms,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_hir_expr(scrutinee);
            for arm in arms {
                visitor.visit_hir_pattern(&arm.pattern);
                if let Some(ref g) = arm.guard {
                    visitor.visit_hir_expr(g);
                }
                visitor.visit_hir_expr(&arm.body);
            }
            ()
        }
        HirExpr::Quantified {
            quantifier: _,
            binder,
            range,
            body,
            ty,
            ..
        } => {
            visitor.visit_type_id(*ty);
            visitor.visit_ident(binder);
            visitor.visit_hir_expr(range);
            visitor.visit_hir_expr(body)
        }
        HirExpr::Error(_) => (),
        HirExpr::TypeInfo(ty, _) => {
            visitor.visit_type_id(*ty);
            ()
        }
        HirExpr::LayoutOf(_, _) => {
            // LayoutOf stores a Box<crate::ast::Type>, not a TypeId —
            // the AST type is resolved during comptime evaluation.
            // No type_id to visit here.
            ()
        }
        HirExpr::CompileError(_, _) => (),
        HirExpr::Task { block, ty, .. } => {
            visitor.visit_type_id(*ty);
            for s in block {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
    }
}

pub fn walk_hir_stmt<'hir, V: HirVisitor<'hir>>(visitor: &mut V, stmt: &HirStmt<'hir>) -> () {
    match stmt {
        HirStmt::VariableDef {
            name,
            value,
            pattern,
            else_branch,
            ty,
            ..
        } => {
            if let Some(n) = name {
                visitor.visit_ident(&n);
            }
            visitor.visit_type_id(*ty);
            if let Some(e) = value {
                visitor.visit_hir_expr(e);
            }
            if let Some(p) = pattern {
                visitor.visit_hir_pattern(p);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirStmt::LayoutDef { .. } => (),
        HirStmt::FunctionDef {
            name,
            params,
            body,
            finally,
            return_type,
            ..
        } => {
            visitor.visit_ident(name);
            if let Some(rt) = return_type {
                visitor.visit_type_id(*rt);
            }
            for p in params {
                visitor.visit_hir_param(p);
            }
            if let Some(b) = body {
                for s in b {
                    visitor.visit_hir_stmt(s);
                }
            }
            if let Some(f) = finally {
                for s in f {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirStmt::Expression(expr) => visitor.visit_hir_expr(expr),
        HirStmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_hir_expr(cond);
            for s in then_branch {
                visitor.visit_hir_stmt(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirStmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_hir_expr(scrutinee);
            visitor.visit_hir_pattern(pattern);
            for s in then_branch {
                visitor.visit_hir_stmt(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_hir_stmt(s);
                }
            }
            ()
        }
        HirStmt::While {
            cond,
            body,
            invariant,
            decreases,
            ..
        } => {
            visitor.visit_hir_expr(cond);
            if let Some(i) = invariant {
                visitor.visit_hir_expr(i.as_ref());
            }
            if let Some(d) = decreases {
                visitor.visit_hir_expr(d.as_ref());
            }
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirStmt::WhileLet {
            pattern,
            scrutinee,
            body,
            ..
        } => {
            visitor.visit_hir_expr(scrutinee);
            visitor.visit_hir_pattern(pattern);
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirStmt::For {
            pattern,
            iterable,
            body,
            ..
        } => {
            visitor.visit_hir_expr(iterable);
            visitor.visit_hir_pattern(pattern);
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirStmt::Loop { body, .. }
        | HirStmt::ScopeCleanup { body, .. }
        | HirStmt::Unsafe { body, .. }
        | HirStmt::Isolate { body, .. } => {
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirStmt::ComptimeBlock { body, ty, .. } => {
            visitor.visit_type_id(*ty);
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
        HirStmt::Return { value, .. } => {
            if let Some(e) = value {
                visitor.visit_hir_expr(e)
            } else {
                ()
            }
        }
        HirStmt::Assign { target, value, .. } => {
            visitor.visit_hir_expr(target);
            visitor.visit_hir_expr(value)
        }
        HirStmt::GhostVariableDef { inner, .. } => visitor.visit_hir_stmt(inner),
        HirStmt::Leave { .. }
        | HirStmt::Continue { .. }
        | HirStmt::Trigger { .. }
        | HirStmt::Edition(..)
        | HirStmt::Stripped { .. }
        | HirStmt::Error => (),
        HirStmt::TypeDef { .. }
        | HirStmt::TraitDef { .. }
        | HirStmt::Import { .. }
        | HirStmt::ExternFunction { .. }
        | HirStmt::Constraint { .. }
        | HirStmt::ImplBlock { .. } => (),
        HirStmt::Generate { for_type, body, .. } => {
            visitor.visit_type_id(*for_type);
            for s in body {
                visitor.visit_hir_stmt(s);
            }
            ()
        }
    }
}

pub fn walk_hir_pattern<'hir, V: HirVisitor<'hir>>(visitor: &mut V, pat: &HirPattern<'hir>) -> () {
    match pat {
        HirPattern::Wildcard(_) | HirPattern::Error(_) => (),
        HirPattern::Ident(name, ty, _) => {
            visitor.visit_type_id(*ty);
            visitor.visit_ident(name)
        }
        HirPattern::Literal(expr, _) => visitor.visit_hir_expr(expr),
        HirPattern::Tuple(patterns, _) => {
            for p in patterns {
                visitor.visit_hir_pattern(p);
            }
            ()
        }
        HirPattern::Slice(before, rest, after, _) => {
            for p in before {
                visitor.visit_hir_pattern(p);
            }
            if let Some(r) = rest {
                visitor.visit_hir_pattern(r);
            }
            for p in after {
                visitor.visit_hir_pattern(p);
            }
            ()
        }
        HirPattern::Struct { fields, .. } => {
            for (_, p) in fields {
                visitor.visit_hir_pattern(p);
            }
            ()
        }
        HirPattern::Enum { inner, .. } => {
            if let Some(p) = inner {
                visitor.visit_hir_pattern(p)
            } else {
                ()
            }
        }
        HirPattern::Or(patterns, _) => {
            for p in patterns {
                visitor.visit_hir_pattern(p);
            }
            ()
        }
    }
}

pub fn walk_hir_param<'hir, V: HirVisitor<'hir>>(visitor: &mut V, param: &HirParam<'hir>) -> () {
    visitor.visit_ident(&param.name);
    visitor.visit_type_id(param.ty)
}

#[cfg(all(test, not(feature = "shuttle")))]
mod smoke_tests {
    use super::*;

    /// Consumer-style visitor: counts expression nodes via the walker's
    /// default (recursive) traversal.
    struct ExprCounter {
        exprs: usize,
        idents: Vec<String>,
        tys: usize,
    }

    impl HirVisitor<'_> for ExprCounter {
        fn visit_hir_expr(&mut self, expr: &HirExpr<'_>) {
            self.exprs += 1;
            walk_hir_expr(self, expr)
        }
        fn visit_ident(&mut self, name: &Symbol) -> () {
            self.idents.push(format!("{name}"))
        }
        fn visit_type_id(&mut self, _ty: TypeId) -> () {
            self.tys += 1
        }
    }

    #[test]
    fn walker_counts_nested_exprs() {
        // TypeId must be non-zero, so allocate through a real context.
        let mut ctx = crate::hir::types::TypeContext::new();
        let ty0 = ctx.int(32, true);
        // (1 + 2) — BinaryOp over two Literals.
        let lit1 = HirExpr::Literal(Literal::Int(IntLit::Small(1)), ty0, crate::ast::DUMMY_SPAN);
        let lit2 = HirExpr::Literal(Literal::Int(IntLit::Small(2)), ty0, crate::ast::DUMMY_SPAN);
        let sum = HirExpr::BinaryOp {
            op: crate::ast::BinOp::Add,
            left: Box::new(lit1),
            right: Box::new(lit2),
            ty: ty0,
            span: crate::ast::DUMMY_SPAN,
        };
        // f(sum) — Call over Ident + the binary tree.
        let callee = HirExpr::Ident(Symbol::intern("f"), ty0, crate::ast::DUMMY_SPAN);
        let call = HirExpr::Call {
            callee: Box::new(callee),
            args: vec![sum],
            ty: ty0,
            span: crate::ast::DUMMY_SPAN,
            comptime: false,
        };

        let mut counter = ExprCounter {
            exprs: 0,
            idents: Vec::new(),
            tys: 0,
        };
        counter.visit_hir_expr(&call);
        assert_eq!(
            counter.exprs, 5,
            "call + ident + binary + 2 literals = 5 nodes"
        );
        assert_eq!(counter.idents, vec!["f".to_string()]);
        assert_eq!(counter.tys, 5, "every expr node reports its type");
    }
}
