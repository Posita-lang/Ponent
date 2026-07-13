//! AST walker. Follows the same pattern as `rustc`'s visitor.
//! Each overridden visit method has full control; the default calls `walk_*`.

use crate::ast::*;
use crate::symbol::Symbol;

/// AST visitor (immutable, shared references).
pub trait Visitor<'ast>: Sized {
    type Result: VisitorResult;

    fn visit_expr(&mut self, expr: &'ast Expr) -> Self::Result {
        walk_expr(self, expr)
    }
    fn visit_stmt(&mut self, stmt: &'ast Stmt) -> Self::Result {
        walk_stmt(self, stmt)
    }
    fn visit_ty(&mut self, ty: &'ast Type) -> Self::Result {
        walk_ty(self, ty)
    }
    fn visit_pattern(&mut self, pat: &'ast Pattern) -> Self::Result {
        walk_pattern(self, pat)
    }
    fn visit_literal(&mut self, _lit: &'ast Literal) -> Self::Result {
        Self::Result::output()
    }
    fn visit_ident(&mut self, _name: Symbol, _span: &'ast Span) -> Self::Result {
        Self::Result::output()
    }
    fn visit_param(&mut self, param: &'ast Param) -> Self::Result {
        walk_param(self, param)
    }
    fn visit_contract(&mut self, contract: &'ast Contract) -> Self::Result {
        walk_contract(self, contract)
    }
    fn visit_attribute(&mut self, _attr: &'ast Attribute) -> Self::Result {
        Self::Result::output()
    }
}

// ── Immutable walk functions ─────────────────────────────────────

pub fn walk_expr<'ast, V: Visitor<'ast>>(visitor: &mut V, expr: &'ast Expr) -> V::Result {
    match expr {
        Expr::Literal(lit, _) => visitor.visit_literal(lit),
        Expr::Ident(name, span) => visitor.visit_ident(*name, span),
        Expr::TypeAnnotated { expr: e, ty, .. } => {
            visitor.visit_expr(e);
            visitor.visit_ty(ty)
        }
        Expr::BinaryOp { left, right, .. } => {
            visitor.visit_expr(left);
            visitor.visit_expr(right)
        }
        Expr::UnaryOp { expr: e, .. } => visitor.visit_expr(e),
        Expr::Call { callee, args, .. } => {
            visitor.visit_expr(callee);
            for arg in args {
                visitor.visit_expr(arg);
            }
            V::Result::output()
        }
        Expr::Index { base, index, .. } => {
            visitor.visit_expr(base);
            visitor.visit_expr(index)
        }
        Expr::FieldAccess { base, field, span } => {
            visitor.visit_expr(base);
            visitor.visit_ident(*field, span)
        }
        Expr::AttrAccess { base, attr, span } => {
            visitor.visit_expr(base);
            visitor.visit_ident(*attr, span)
        }
        Expr::Cast { expr: base, .. } => visitor.visit_expr(base),
        Expr::Range { start, end, .. } => {
            if let Some(e) = start {
                visitor.visit_expr(e);
            }
            if let Some(e) = end {
                visitor.visit_expr(e);
            }
            V::Result::output()
        }
        Expr::StructLit { fields, .. } => {
            for (_, e) in fields {
                visitor.visit_expr(e);
            }
            V::Result::output()
        }
        Expr::EnumLit { variant, payload, span, .. } => {
            visitor.visit_ident(*variant, span);
            if let Some(e) = payload {
                visitor.visit_expr(e)
            } else {
                V::Result::output()
            }
        }
        Expr::Move(e, _)
        | Expr::Await { expr: e, .. }
        | Expr::Try { expr: e, .. }
        | Expr::LeaveWith { expr: e, .. }
        | Expr::PolyBox { expr: e, .. }
        | Expr::PolyUnbox { expr: e, .. }
        | Expr::Old(e, _) => visitor.visit_expr(e),
        Expr::Task { .. } | Expr::Path(_, _) => V::Result::output(),
        Expr::Tuple(exprs, _) | Expr::Array(exprs, _) => {
            for e in exprs {
                visitor.visit_expr(e);
            }
            V::Result::output()
        }
        Expr::Closure { params, body, .. } => {
            for p in params {
                visitor.visit_param(p);
            }
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr(cond);
            for s in then_branch {
                visitor.visit_stmt(s);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Expr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr(scrutinee);
            visitor.visit_pattern(pattern);
            for s in then_branch {
                visitor.visit_stmt(s);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            visitor.visit_expr(scrutinee);
            for arm in arms {
                visitor.visit_pattern(&arm.pattern);
                if let Some(g) = &arm.guard {
                    visitor.visit_expr(g);
                }
                visitor.visit_expr(&arm.body);
            }
            V::Result::output()
        }
        Expr::Block(stmts, _) | Expr::UnsafeBlock { body: stmts, .. } => {
            for s in stmts {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Expr::Catch {
            expr: e, branches, ..
        } => {
            visitor.visit_expr(e);
            for b in branches {
                for s in &b.body {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Expr::Quantified { range, body, .. } => {
            visitor.visit_expr(range);
            visitor.visit_expr(body)
        }
        Expr::Error(_) => V::Result::output(),
        Expr::CompileError(_, _) => V::Result::output(),
        Expr::TypeInfo(ty, _) => visitor.visit_ty(ty),
    }
}

pub fn walk_stmt<'ast, V: Visitor<'ast>>(visitor: &mut V, stmt: &'ast Stmt) -> V::Result {
    match stmt {
        Stmt::VariableDef {
            value,
            pattern,
            else_branch,
            ..
        } => {
            if let Some(e) = value {
                visitor.visit_expr(e);
            }
            if let Some(p) = pattern {
                visitor.visit_pattern(p);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Stmt::FunctionDef {
            name,
            params,
            body,
            finally,
            span,
            ..
        } => {
            visitor.visit_ident(*name, span);
            for p in params {
                visitor.visit_param(p);
            }
            if let Some(b) = body {
                for s in b {
                    visitor.visit_stmt(s);
                }
            }
            if let Some(f) = finally {
                for s in f {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Stmt::Expression(expr) => visitor.visit_expr(expr),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr(cond);
            for s in then_branch {
                visitor.visit_stmt(s);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Stmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr(scrutinee);
            visitor.visit_pattern(pattern);
            for s in then_branch {
                visitor.visit_stmt(s);
            }
            if let Some(else_b) = else_branch {
                for s in else_b {
                    visitor.visit_stmt(s);
                }
            }
            V::Result::output()
        }
        Stmt::While { cond, body, .. } => {
            visitor.visit_expr(cond);
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Stmt::WhileLet {
            pattern,
            scrutinee,
            body,
            ..
        } => {
            visitor.visit_expr(scrutinee);
            visitor.visit_pattern(pattern);
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Stmt::Loop { body, .. } => {
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Stmt::For {
            pattern,
            iterable,
            body,
            ..
        } => {
            visitor.visit_expr(iterable);
            visitor.visit_pattern(pattern);
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Stmt::Return { value, .. } => {
            if let Some(e) = value {
                visitor.visit_expr(e)
            } else {
                V::Result::output()
            }
        }
        Stmt::Assign { target, value, .. } => {
            visitor.visit_expr(target);
            visitor.visit_expr(value)
        }
        Stmt::ComptimeBlock { body, .. }
        | Stmt::ScopeCleanup { body, .. }
        | Stmt::Unsafe { body, .. }
        | Stmt::Isolate { body, .. } => {
            for s in body {
                visitor.visit_stmt(s);
            }
            V::Result::output()
        }
        Stmt::GhostVariableDef { inner, .. } => visitor.visit_stmt(inner),
        Stmt::Leave { .. }
        | Stmt::Continue { .. }
        | Stmt::Trigger { .. }
        | Stmt::Edition(..)
        | Stmt::LayoutDef { .. }
        | Stmt::Error(_) => V::Result::output(),
        Stmt::Generate { body, .. } => {
            for s in body { visitor.visit_stmt(s); }
            V::Result::output()
        }
        Stmt::TypeDef { .. }
        | Stmt::TraitDef { .. }
        | Stmt::ImplBlock { .. }
        | Stmt::Import { .. }
        | Stmt::ExternFunction { .. }
        | Stmt::Constraint { .. } => V::Result::output(),
    }
}

pub fn walk_ty<'ast, V: Visitor<'ast>>(_visitor: &mut V, _ty: &'ast Type) -> V::Result {
    V::Result::output()
}

pub fn walk_pattern<'ast, V: Visitor<'ast>>(visitor: &mut V, pat: &'ast Pattern) -> V::Result {
    match pat {
        Pattern::Wildcard(_) | Pattern::Error(_) => V::Result::output(),
        Pattern::Ident(name, span) => visitor.visit_ident(*name, span),
        Pattern::Literal(expr, _) => visitor.visit_expr(expr),
        Pattern::Tuple(patterns, _) => {
            for p in patterns {
                visitor.visit_pattern(p);
            }
            V::Result::output()
        }
        Pattern::Slice(before, rest, after, _) => {
            for p in before {
                visitor.visit_pattern(p);
            }
            if let Some(r) = rest {
                visitor.visit_pattern(r);
            }
            for p in after {
                visitor.visit_pattern(p);
            }
            V::Result::output()
        }
        Pattern::Struct { fields, .. } => {
            for (_, p) in fields {
                visitor.visit_pattern(p);
            }
            V::Result::output()
        }
        Pattern::Enum { inner, .. } => {
            if let Some(p) = inner {
                visitor.visit_pattern(p)
            } else {
                V::Result::output()
            }
        }
        Pattern::Or(patterns, _) => {
            for p in patterns {
                visitor.visit_pattern(p);
            }
            V::Result::output()
        }
    }
}

pub fn walk_param<'ast, V: Visitor<'ast>>(visitor: &mut V, param: &'ast Param) -> V::Result {
    visitor.visit_ident(param.name, &param.span);
    if let Some(ty) = &param.ty {
        visitor.visit_ty(ty)
    } else {
        V::Result::output()
    }
}

pub fn walk_contract<'ast, V: Visitor<'ast>>(
    visitor: &mut V,
    contract: &'ast Contract,
) -> V::Result {
    match contract {
        Contract::Requires(expr, _)
        | Contract::Invariant(expr, _)
        | Contract::Decreases(expr, _)
        | Contract::Terminates(expr, _) => visitor.visit_expr(expr),
        Contract::Ensures { expr, .. } => visitor.visit_expr(expr),
    }
}

// ── MutVisitor (mutable, in-place transformation) ───────────────

/// Mutable AST visitor for in-place tree transformations.
/// Each method has a default implementation that recurses via `walk_mut_*`.
pub trait MutVisitor: Sized {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        walk_expr_mut(self, expr)
    }
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        walk_stmt_mut(self, stmt)
    }
    fn visit_pattern_mut(&mut self, pat: &mut Pattern) {
        walk_pattern_mut(self, pat)
    }
    fn visit_param_mut(&mut self, param: &mut Param) {
        walk_param_mut(self, param)
    }
    fn visit_ident_mut(&mut self, name: &mut Symbol) {
        let _ = name;
    }
}

pub fn walk_expr_mut<V: MutVisitor>(visitor: &mut V, expr: &mut Expr) {
    match expr {
        Expr::Literal(_, _) | Expr::Ident(_, _) | Expr::Path(_, _) | Expr::Task { .. } | Expr::Error(_) | Expr::CompileError(_, _) | Expr::TypeInfo(_, _) => {}
        Expr::TypeAnnotated { expr: e, ty: _, .. } => visitor.visit_expr_mut(e),
        Expr::BinaryOp { left, right, .. } => {
            visitor.visit_expr_mut(left);
            visitor.visit_expr_mut(right);
        }
        Expr::UnaryOp { expr: e, .. } => visitor.visit_expr_mut(e),
        Expr::Call { callee, args, .. } => {
            visitor.visit_expr_mut(callee);
            for a in args {
                visitor.visit_expr_mut(a);
            }
        }
        Expr::Index { base, index, .. } => {
            visitor.visit_expr_mut(base);
            visitor.visit_expr_mut(index);
        }
        Expr::FieldAccess { base, field, .. } => {
            visitor.visit_expr_mut(base);
            visitor.visit_ident_mut(field);
        }
        Expr::AttrAccess { base, attr, .. } => {
            visitor.visit_expr_mut(base);
            visitor.visit_ident_mut(attr);
        }
        Expr::Cast { expr: base, .. } => visitor.visit_expr_mut(base),
        Expr::Range { start, end, .. } => {
            if let Some(s) = start {
                visitor.visit_expr_mut(s);
            }
            if let Some(e) = end {
                visitor.visit_expr_mut(e);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, e) in fields {
                visitor.visit_expr_mut(e);
            }
            // NOTE: field names are String, not ident — not visited.
        }
        Expr::EnumLit { variant, payload, .. } => {
            visitor.visit_ident_mut(variant);
            if let Some(e) = payload {
                visitor.visit_expr_mut(e);
            }
        }
        Expr::Move(e, _)
        | Expr::Await { expr: e, .. }
        | Expr::Try { expr: e, .. }
        | Expr::LeaveWith { expr: e, .. }
        | Expr::PolyBox { expr: e, .. }
        | Expr::PolyUnbox { expr: e, .. }
        | Expr::Old(e, _) => visitor.visit_expr_mut(e),
        Expr::Tuple(exprs, _) | Expr::Array(exprs, _) => {
            for e in exprs {
                visitor.visit_expr_mut(e);
            }
        }
        Expr::Closure {
            params, body, ..
        } => {
            for p in params {
                visitor.visit_param_mut(p);
            }
            for s in body {
                visitor.visit_stmt_mut(s);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr_mut(cond);
            for s in then_branch {
                visitor.visit_stmt_mut(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Expr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr_mut(scrutinee);
            visitor.visit_pattern_mut(pattern);
            for s in then_branch {
                visitor.visit_stmt_mut(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            visitor.visit_expr_mut(scrutinee);
            for arm in arms {
                visitor.visit_pattern_mut(&mut arm.pattern);
                if let Some(g) = &mut arm.guard {
                    visitor.visit_expr_mut(g);
                }
                visitor.visit_expr_mut(&mut arm.body);
            }
        }
        Expr::Block(stmts, _) | Expr::UnsafeBlock { body: stmts, .. } => {
            for s in stmts {
                visitor.visit_stmt_mut(s);
            }
        }
        Expr::Catch {
            expr: e, branches, ..
        } => {
            visitor.visit_expr_mut(e);
            for b in branches {
                visitor.visit_pattern_mut(&mut b.pattern);
                for s in &mut b.body {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Expr::Quantified { range, body, .. } => {
            visitor.visit_expr_mut(range);
            visitor.visit_expr_mut(body);
        }
    }
}

pub fn walk_stmt_mut<V: MutVisitor>(visitor: &mut V, stmt: &mut Stmt) {
    match stmt {
        Stmt::VariableDef {
            value,
            pattern,
            else_branch,
            ..
        } => {
            if let Some(e) = value {
                visitor.visit_expr_mut(e);
            }
            if let Some(p) = pattern {
                visitor.visit_pattern_mut(p);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Stmt::FunctionDef {
            name,
            params,
            body,
            finally,
            ..
        } => {
            visitor.visit_ident_mut(name);
            for p in params {
                visitor.visit_param_mut(p);
            }
            if let Some(b) = body {
                for s in b {
                    visitor.visit_stmt_mut(s);
                }
            }
            if let Some(f) = finally {
                for s in f {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Stmt::Expression(expr) => visitor.visit_expr_mut(expr),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr_mut(cond);
            for s in then_branch {
                visitor.visit_stmt_mut(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Stmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            visitor.visit_expr_mut(scrutinee);
            visitor.visit_pattern_mut(pattern);
            for s in then_branch {
                visitor.visit_stmt_mut(s);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    visitor.visit_stmt_mut(s);
                }
            }
        }
        Stmt::While { cond, body, .. } => {
            visitor.visit_expr_mut(cond);
            for s in body {
                visitor.visit_stmt_mut(s);
            }
        }
        Stmt::WhileLet {
            pattern,
            scrutinee,
            body,
            ..
        } => {
            visitor.visit_expr_mut(scrutinee);
            visitor.visit_pattern_mut(pattern);
            for s in body {
                visitor.visit_stmt_mut(s);
            }
        }
        Stmt::For {
            pattern,
            iterable,
            body,
            ..
        } => {
            visitor.visit_expr_mut(iterable);
            visitor.visit_pattern_mut(pattern);
            for s in body {
                visitor.visit_stmt_mut(s);
            }
        }
        Stmt::Loop { body, .. }
        | Stmt::ComptimeBlock { body, .. }
        | Stmt::ScopeCleanup { body, .. }
        | Stmt::Unsafe { body, .. }
        | Stmt::Isolate { body, .. } => {
            for s in body {
                visitor.visit_stmt_mut(s);
            }
        }
        Stmt::Return { value, .. } => {
            if let Some(e) = value {
                visitor.visit_expr_mut(e);
            }
        }
        Stmt::Assign { target, value, .. } => {
            visitor.visit_expr_mut(target);
            visitor.visit_expr_mut(value);
        }
        Stmt::GhostVariableDef { inner, .. } => visitor.visit_stmt_mut(inner),
        Stmt::Leave { .. }
        | Stmt::Continue { .. }
        | Stmt::Trigger { .. }
        | Stmt::Edition(..)
        | Stmt::LayoutDef { .. }
        | Stmt::Error(_)
        | Stmt::TypeDef { .. }
        | Stmt::TraitDef { .. }
        | Stmt::ImplBlock { .. }
        | Stmt::Import { .. }
        | Stmt::ExternFunction { .. }
        | Stmt::Constraint { .. } => {}
        Stmt::Generate { body, .. } => {
            for s in body { visitor.visit_stmt_mut(s); }
        }
    }
}

pub fn walk_pattern_mut<V: MutVisitor>(visitor: &mut V, pat: &mut Pattern) {
    match pat {
        Pattern::Wildcard(_) | Pattern::Error(_) => {}
        Pattern::Ident(name, _) => {
            visitor.visit_ident_mut(name);
        }
        Pattern::Literal(expr, _) => visitor.visit_expr_mut(expr),
        Pattern::Tuple(patterns, _) => {
            for p in patterns {
                visitor.visit_pattern_mut(p);
            }
        }
        Pattern::Slice(before, rest, after, _) => {
            for p in before {
                visitor.visit_pattern_mut(p);
            }
            if let Some(r) = rest {
                visitor.visit_pattern_mut(r);
            }
            for p in after {
                visitor.visit_pattern_mut(p);
            }
        }
        Pattern::Struct { fields, .. } => {
            for (_, p) in fields {
                visitor.visit_pattern_mut(p);
            }
        }
        Pattern::Enum { inner, .. } => {
            if let Some(p) = inner {
                visitor.visit_pattern_mut(p);
            }
        }
        Pattern::Or(patterns, _) => {
            for p in patterns {
                visitor.visit_pattern_mut(p);
            }
        }
    }
}

pub fn walk_param_mut<V: MutVisitor>(visitor: &mut V, param: &mut Param) {
    visitor.visit_ident_mut(&mut param.name);
    if let Some(ty) = &mut param.ty {
        // Types currently have no mutable walker — skip.
        let _ = ty;
    }
}

// ── Result type helper (like rustc's `V::Result::output()`) ─────

/// Trait for visitor result types that can be "output" (no-op continuation).
pub trait VisitorResult {
    fn output() -> Self;
}

impl VisitorResult for () {
    fn output() -> Self {}
}

impl<T> VisitorResult for Option<T> {
    fn output() -> Self {
        None
    }
}

// ── ReplaceIdentVisitor (renames identifiers in-place) ───────────

struct ReplaceIdentVisitor {
    old_name: Symbol,
    new_name: Symbol,
}

impl MutVisitor for ReplaceIdentVisitor {
    fn visit_ident_mut(&mut self, name: &mut Symbol) {
        if *name == self.old_name {
            *name = self.new_name;
        }
    }
}

/// Rename all occurrences of `old_name` to `new_name` in an expression tree.
pub fn replace_ident_in_expr(expr: &mut Expr, old_name: Symbol, new_name: Symbol) {
    let mut v = ReplaceIdentVisitor {
        old_name,
        new_name,
    };
    v.visit_expr_mut(expr);
}

/// Rename all occurrences of `old_name` to `new_name` in a statement tree.
pub fn replace_ident_in_stmt(stmt: &mut Stmt, old_name: Symbol, new_name: Symbol) {
    let mut v = ReplaceIdentVisitor {
        old_name,
        new_name,
    };
    v.visit_stmt_mut(stmt);
}
