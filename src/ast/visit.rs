//! AST walker. Follows the same pattern as `rustc`'s visitor.
//! Each overridden visit method has full control; the default calls `walk_*`.

use crate::ast::*;
use crate::symbol::Symbol;

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

/// AST visitor (immutable, shared references).
pub trait Visitor<'ast, 'input>: Sized {
    type Result: VisitorResult;

    fn visit_expr(&mut self, expr: &'ast Expr<'input>) -> Self::Result {
        walk_expr(self, expr)
    }
    fn visit_stmt(&mut self, stmt: &'ast Stmt<'input>) -> Self::Result {
        walk_stmt(self, stmt)
    }
    fn visit_ty(&mut self, ty: &'ast Type<'input>) -> Self::Result {
        walk_ty(self, ty)
    }
    fn visit_pattern(&mut self, pat: &'ast Pattern<'input>) -> Self::Result {
        walk_pattern(self, pat)
    }
    fn visit_literal(&mut self, _lit: &'ast Literal) -> Self::Result {
        Self::Result::output()
    }
    fn visit_ident(&mut self, _name: Symbol, _span: &'ast Span) -> Self::Result {
        Self::Result::output()
    }
    fn visit_param(&mut self, param: &'ast Param<'input>) -> Self::Result {
        walk_param(self, param)
    }
    fn visit_contract(&mut self, contract: &'ast Contract<'input>) -> Self::Result {
        walk_contract(self, contract)
    }
    fn visit_attribute(&mut self, _attr: &'ast Attribute<'input>) -> Self::Result {
        Self::Result::output()
    }
}

// ── Immutable walk functions ─────────────────────────────────────

pub fn walk_expr<'ast, 'input, V: Visitor<'ast, 'input>>(
    visitor: &mut V,
    expr: &'ast Expr<'input>,
) -> V::Result {
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
        Expr::EnumLit {
            variant,
            payload,
            span,
            ..
        } => {
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
        Expr::LayoutOf(ty, _) => visitor.visit_ty(ty),
    }
}

pub fn walk_stmt<'ast, 'input, V: Visitor<'ast, 'input>>(
    visitor: &mut V,
    stmt: &'ast Stmt<'input>,
) -> V::Result {
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
            for s in body {
                visitor.visit_stmt(s);
            }
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

pub fn walk_ty<'ast, 'input, V: Visitor<'ast, 'input>>(
    _visitor: &mut V,
    _ty: &'ast Type<'input>,
) -> V::Result {
    V::Result::output()
}

pub fn walk_pattern<'ast, 'input, V: Visitor<'ast, 'input>>(
    visitor: &mut V,
    pat: &'ast Pattern<'input>,
) -> V::Result {
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

pub fn walk_param<'ast, 'input, V: Visitor<'ast, 'input>>(
    visitor: &mut V,
    param: &'ast Param<'input>,
) -> V::Result {
    visitor.visit_ident(param.name, &param.span);
    if let Some(ty) = &param.ty {
        visitor.visit_ty(ty)
    } else {
        V::Result::output()
    }
}

pub fn walk_contract<'ast, 'input, V: Visitor<'ast, 'input>>(
    visitor: &mut V,
    contract: &'ast Contract<'input>,
) -> V::Result {
    match contract {
        Contract::Requires(expr, _)
        | Contract::Invariant(expr, _)
        | Contract::Decreases(expr, _)
        | Contract::Terminates(expr, _) => visitor.visit_expr(expr),
        Contract::Ensures { expr, .. } => visitor.visit_expr(expr),
    }
}

/// Rename all occurrences of `old_name` to `new_name` in an expression
/// tree, allocating the NEW tree in `arena` (the AST is arena-backed
/// `&'input` shared references — in-place mutation is not possible, and
/// building a fresh tree requires allocating nodes in the parser's
/// arena).  Used by the `type T = Base where value > 0` desugar
/// (`exists _where_N: Base invariant _where_N > 0`).
/// Collect the variable names BOUND by a pattern — a binder in a pattern
/// shadows the outer scope (capture avoidance: a `match`/`for`/`if let`
/// body must not be renamed into a name its own pattern binds).
fn bound_vars<'input>(p: &Pattern<'input>, out: &mut Vec<Symbol>) {
    match p {
        Pattern::Ident(name, _) => out.push(*name),
        Pattern::Tuple(ps, _) | Pattern::Or(ps, _) => {
            for sub in ps {
                bound_vars(sub, out);
            }
        }
        Pattern::Slice(pre, mid, post, _) => {
            for sub in pre {
                bound_vars(sub, out);
            }
            if let Some(m) = mid {
                bound_vars(m, out);
            }
            for sub in post {
                bound_vars(sub, out);
            }
        }
        Pattern::Struct { fields, .. } => {
            for (_, sub) in fields {
                bound_vars(sub, out);
            }
        }
        Pattern::Enum { inner, .. } => {
            if let Some(sub) = inner {
                bound_vars(sub, out);
            }
        }
        Pattern::Wildcard(_) | Pattern::Literal(..) | Pattern::Error(_) => {}
    }
}

pub fn replace_ident_in_expr<'input>(
    arena: &'input bumpalo::Bump,
    expr: &Expr<'input>,
    old_name: Symbol,
    new_name: Symbol,
) -> &'input Expr<'input> {
    fn go<'a>(arena: &'a bumpalo::Bump, e: &Expr<'a>, old: Symbol, new: Symbol) -> &'a Expr<'a> {
        match e {
            Expr::Ident(name, span) if *name == old => arena.alloc(Expr::Ident(new, *span)),
            Expr::Literal(l, s) => arena.alloc(Expr::Literal(l.clone(), *s)),
            Expr::Ident(name, span) => arena.alloc(Expr::Ident(*name, *span)),
            Expr::TypeAnnotated { expr, ty, span } => arena.alloc(Expr::TypeAnnotated {
                expr: go(arena, expr, old, new),
                // The type annotation is NOT renamed — the desugar only
                // renames value-level identifiers in the invariant
                // expression; type annotations carry no value-level
                // identifiers (type names resolve separately).
                ty: *ty,
                span: *span,
            }),
            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => arena.alloc(Expr::BinaryOp {
                left: go(arena, left, old, new),
                op: *op,
                right: go(arena, right, old, new),
                span: *span,
            }),
            Expr::UnaryOp { op, expr, span } => arena.alloc(Expr::UnaryOp {
                op: *op,
                expr: go(arena, expr, old, new),
                span: *span,
            }),
            Expr::Call {
                callee,
                args,
                comptime,
                span,
            } => arena.alloc(Expr::Call {
                callee: go(arena, callee, old, new),
                args: args
                    .iter()
                    .map(|a| (*go(arena, a, old, new)).clone())
                    .collect(),
                comptime: *comptime,
                span: *span,
            }),
            Expr::Index { base, index, span } => arena.alloc(Expr::Index {
                base: go(arena, base, old, new),
                index: go(arena, index, old, new),
                span: *span,
            }),
            Expr::FieldAccess { base, field, span } => arena.alloc(Expr::FieldAccess {
                base: go(arena, base, old, new),
                field: *field,
                span: *span,
            }),
            Expr::AttrAccess { base, attr, span } => arena.alloc(Expr::AttrAccess {
                base: go(arena, base, old, new),
                attr: *attr,
                span: *span,
            }),
            Expr::Cast {
                expr,
                ty,
                safe,
                rounding,
                span,
            } => arena.alloc(Expr::Cast {
                expr: go(arena, expr, old, new),
                ty: *ty,
                safe: *safe,
                rounding: *rounding,
                span: *span,
            }),
            Expr::Range {
                start,
                end,
                inclusive,
                span,
            } => arena.alloc(Expr::Range {
                start: start.as_ref().map(|s| go(arena, s, old, new)),
                end: end.as_ref().map(|s| go(arena, s, old, new)),
                inclusive: *inclusive,
                span: *span,
            }),
            Expr::StructLit { path, fields, span } => arena.alloc(Expr::StructLit {
                path: path.clone(),
                fields: fields
                    .iter()
                    .map(|(n, e)| (*n, (*go(arena, e, old, new)).clone()))
                    .collect(),
                span: *span,
            }),
            Expr::EnumLit {
                path,
                variant,
                payload,
                span,
            } => arena.alloc(Expr::EnumLit {
                path: path.clone(),
                variant: *variant,
                payload: payload.as_ref().map(|p| go(arena, p, old, new)),
                span: *span,
            }),
            Expr::Move(e, s) => arena.alloc(Expr::Move(go(arena, e, old, new), *s)),
            Expr::Path(p, s) => arena.alloc(Expr::Path(p.clone(), *s)),
            Expr::Tuple(es, s) => arena.alloc(Expr::Tuple(
                es.iter()
                    .map(|x| (*go(arena, x, old, new)).clone())
                    .collect(),
                *s,
            )),
            Expr::Array(es, s) => arena.alloc(Expr::Array(
                es.iter()
                    .map(|x| (*go(arena, x, old, new)).clone())
                    .collect(),
                *s,
            )),
            Expr::Closure {
                params,
                return_type,
                captures,
                body,
                span,
            } => {
                // if a CLOSURE PARAMETER shadows the renamed
                // identifier, the body's bound occurrences must NOT be
                // renamed (the same capture-avoidance as the quantifier
                // binder).
                let shadowed = params.iter().any(|p| p.name == old);
                arena.alloc(Expr::Closure {
                    params: params.clone(),
                    return_type: return_type.clone(),
                    captures: captures.clone(),
                    body: if shadowed {
                        body.iter().map(|s| (*s).clone()).collect()
                    } else {
                        body.iter()
                            .map(|s| replace_ident_in_stmt(arena, s, old, new))
                            .collect()
                    },
                    span: *span,
                })
            }
            Expr::Try { expr, span } => arena.alloc(Expr::Try {
                expr: go(arena, expr, old, new),
                span: *span,
            }),
            Expr::UnsafeBlock { body, span } => arena.alloc(Expr::UnsafeBlock {
                body: body
                    .iter()
                    .map(|s| replace_ident_in_stmt(arena, s, old, new))
                    .collect(),
                span: *span,
            }),
            Expr::Catch {
                expr,
                branches,
                span,
            } => arena.alloc(Expr::Catch {
                expr: go(arena, expr, old, new),
                branches: branches
                    .iter()
                    .map(|b| {
                        // Capture avoidance (same as Match/IfLet): a branch
                        // binding `old` via its pattern or `as` binder
                        // shadows the rename target inside the branch body.
                        let mut bv = Vec::new();
                        bound_vars(&b.pattern, &mut bv);
                        if let Some(bind_sym) = b.bind {
                            bv.push(bind_sym);
                        }
                        let shadowed = bv.contains(&old);
                        CatchBranch {
                            pattern: b.pattern.clone(),
                            bind: b.bind,
                            body: if shadowed {
                                b.body.iter().map(|s| (*s).clone()).collect()
                            } else {
                                b.body
                                    .iter()
                                    .map(|s| replace_ident_in_stmt(arena, s, old, new))
                                    .collect()
                            },
                            span: b.span,
                        }
                    })
                    .collect(),
                span: *span,
            }),
            Expr::LeaveWith {
                expr,
                is_return,
                span,
            } => arena.alloc(Expr::LeaveWith {
                expr: go(arena, expr, old, new),
                is_return: *is_return,
                span: *span,
            }),
            Expr::Await { expr, span } => arena.alloc(Expr::Await {
                expr: go(arena, expr, old, new),
                span: *span,
            }),
            Expr::If {
                cond,
                then_branch,
                else_branch,
                is_expression,
                span,
            } => arena.alloc(Expr::If {
                cond: go(arena, cond, old, new),
                then_branch: then_branch
                    .iter()
                    .map(|s| replace_ident_in_stmt(arena, s, old, new))
                    .collect(),
                else_branch: else_branch.as_ref().map(|eb| {
                    eb.iter()
                        .map(|s| replace_ident_in_stmt(arena, s, old, new))
                        .collect()
                }),
                is_expression: *is_expression,
                span: *span,
            }),
            Expr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                is_expression,
                span,
            } => arena.alloc(Expr::IfLet {
                pattern: pattern.clone(),
                scrutinee: go(arena, scrutinee, old, new),
                then_branch: if {
                    let mut bv = Vec::new();
                    bound_vars(pattern, &mut bv);
                    bv.contains(&old)
                } {
                    then_branch.clone()
                } else {
                    then_branch
                        .iter()
                        .map(|s| replace_ident_in_stmt(arena, s, old, new))
                        .collect()
                },
                // The pattern's bindings are scoped to the THEN branch
                // only (checked after the `with_gadt_arm` scope pops in
                // the checker) — the else branch is in the OUTER scope,
                // so references to `old` there refer to the rename target
                // and MUST be renamed regardless of what the pattern binds.
                else_branch: else_branch.as_ref().map(|eb| {
                    eb.iter()
                        .map(|s| replace_ident_in_stmt(arena, s, old, new))
                        .collect()
                }),
                is_expression: *is_expression,
                span: *span,
            }),
            Expr::Match {
                scrutinee,
                arms,
                span,
            } => arena.alloc(Expr::Match {
                scrutinee: go(arena, scrutinee, old, new),
                arms: arms
                    .iter()
                    .map(|arm| {
                        // Capture avoidance: a pattern that binds `old`
                        // shadows the outer name — its body must NOT be
                        // renamed (the reference is to the local binding).
                        let mut bv = Vec::new();
                        bound_vars(&arm.pattern, &mut bv);
                        let shadowed = bv.contains(&old);
                        MatchArm {
                            pattern: arm.pattern.clone(),
                            // The guard is evaluated in the scope of the
                            // pattern bindings (like Rust) — if the pattern
                            // binds `old`, the guard's reference must NOT be
                            // renamed either (same capture avoidance as the
                            // body).
                            guard: if shadowed {
                                arm.guard.clone()
                            } else {
                                arm.guard.as_ref().map(|g| go(arena, g, old, new))
                            },
                            body: if shadowed {
                                arm.body.clone()
                            } else {
                                (*go(arena, &arm.body, old, new)).clone()
                            },
                            span: arm.span,
                        }
                    })
                    .collect(),
                span: *span,
            }),
            Expr::Block(stmts, s) => arena.alloc(Expr::Block(
                stmts
                    .iter()
                    .map(|x| replace_ident_in_stmt(arena, x, old, new))
                    .collect(),
                *s,
            )),
            Expr::PolyBox { expr, scheme, span } => arena.alloc(Expr::PolyBox {
                expr: go(arena, expr, old, new),
                scheme: scheme.clone(),
                span: *span,
            }),
            Expr::PolyUnbox { expr, scheme, span } => arena.alloc(Expr::PolyUnbox {
                expr: go(arena, expr, old, new),
                scheme: scheme.clone(),
                span: *span,
            }),
            Expr::Quantified {
                quantifier,
                binder,
                range,
                body,
                span,
            } => {
                // if the QUANTIFIER BINDER shadows the renamed
                // identifier, the BOUND occurrences in the body must NOT
                // be renamed — they belong to the binder, not the outer
                // `old` (renaming them would make the bound variable
                // free, silently corrupting the contract).
                let shadowed = *binder == old;
                arena.alloc(Expr::Quantified {
                    quantifier: *quantifier,
                    binder: *binder,
                    range: go(arena, range, old, new),
                    body: if shadowed {
                        body
                    } else {
                        go(arena, body, old, new)
                    },
                    span: *span,
                })
            }
            Expr::Old(e, s) => arena.alloc(Expr::Old(go(arena, e, old, new), *s)),
            Expr::Task { body, span } => arena.alloc(Expr::Task {
                body: body
                    .iter()
                    .map(|s| replace_ident_in_stmt(arena, s, old, new))
                    .collect(),
                span: *span,
            }),
            Expr::TypeInfo(t, s) => arena.alloc(Expr::TypeInfo(*t, *s)),
            Expr::LayoutOf(t, s) => arena.alloc(Expr::LayoutOf(*t, *s)),
            Expr::CompileError(m, s) => arena.alloc(Expr::CompileError(m.clone(), *s)),
            Expr::Error(s) => arena.alloc(Expr::Error(*s)),
        }
    }
    go(arena, expr, old_name, new_name)
}

/// Rename all occurrences of `old_name` to `new_name` in a statement tree,
/// allocating the NEW tree in `arena` (functional — the AST is arena-backed).
pub fn replace_ident_in_stmt<'input>(
    arena: &'input bumpalo::Bump,
    stmt: &Stmt<'input>,
    old_name: Symbol,
    new_name: Symbol,
) -> Stmt<'input> {
    fn go_expr<'a>(arena: &'a bumpalo::Bump, e: &Expr<'a>, old: Symbol, new: Symbol) -> Expr<'a> {
        (*replace_ident_in_expr(arena, e, old, new)).clone()
    }
    fn go_stmts<'a>(
        arena: &'a bumpalo::Bump,
        ss: &[Stmt<'a>],
        old: Symbol,
        new: Symbol,
    ) -> Vec<Stmt<'a>> {
        // Lexical scoping: a binding introduced by one statement shadows
        // the rename target for SUBSEQUENT statements in the block — once
        // `old` is bound locally, later references point at the local
        // binder and must NOT be renamed (a stateless map would sever
        // them, leaving dangling references).
        let mut shadowed: std::collections::HashSet<Symbol> = Default::default();
        ss.iter()
            .map(|s| {
                if shadowed.contains(&old) {
                    return (*s).clone();
                }
                let r = replace_ident_in_stmt(arena, s, old, new);
                match s {
                    Stmt::VariableDef { name: Some(n), .. } if *n == old => {
                        shadowed.insert(old);
                    }
                    Stmt::VariableDef {
                        pattern: Some(p), ..
                    } => {
                        let mut bv = Vec::new();
                        bound_vars(p, &mut bv);
                        if bv.contains(&old) {
                            shadowed.insert(old);
                        }
                    }
                    // Loop variables (`For`/`WhileLet` pattern bindings)
                    // are scoped to the LOOP BODY only — they do NOT leak
                    // to subsequent block statements, so they must not
                    // pollute the block-level `shadowed` set (which would
                    // wrongly skip renaming the rest of the block).
                    _ => {}
                }
                r
            })
            .collect()
    }
    fn go_contract<'a>(
        arena: &'a bumpalo::Bump,
        c: &Contract<'a>,
        old: Symbol,
        new: Symbol,
    ) -> Contract<'a> {
        match c {
            Contract::Requires(e, s) => Contract::Requires(go_expr(arena, e, old, new), *s),
            Contract::Ensures {
                expr,
                span,
                target,
                labels,
            } => Contract::Ensures {
                expr: go_expr(arena, expr, old, new),
                span: *span,
                target: target.clone(),
                labels: labels.clone(),
            },
            Contract::Invariant(e, s) => Contract::Invariant(go_expr(arena, e, old, new), *s),
            Contract::Decreases(e, s) => Contract::Decreases(go_expr(arena, e, old, new), *s),
            Contract::Terminates(e, s) => Contract::Terminates(go_expr(arena, e, old, new), *s),
        }
    }
    match stmt {
        Stmt::VariableDef {
            kind,
            mutable,
            name,
            pattern,
            ty,
            value,
            else_branch,
            span,
            attributes,
            doc,
            type_captures,
            type_modifiers,
        } => Stmt::VariableDef {
            kind: *kind,
            mutable: *mutable,
            name: *name,
            // Patterns are BINDING sites, not use sites: renaming a
            // pattern binding would change the variable's identity, which
            // is incorrect (the `where value > 0` desugar renames the
            // invariant EXPRESSION, never a binder).  The old
            // MutVisitor-based code renamed inside patterns — wrong; the
            // functional version intentionally does not.
            pattern: pattern.clone(),
            ty: ty.clone(),
            value: value
                .as_ref()
                .map(|e| go_expr(arena, e, old_name, new_name)),
            else_branch: else_branch
                .as_ref()
                .map(|ss| go_stmts(arena, ss, old_name, new_name)),
            span: *span,
            attributes: attributes.clone(),
            doc: doc.clone(),
            type_captures: type_captures.clone(),
            type_modifiers: type_modifiers.clone(),
        },
        Stmt::FunctionDef {
            span,
            attributes,
            contracts,
            doc,
            name,
            params,
            return_type,
            body,
            type_params,
            where_clause,
            finally,
            is_comptime,
            is_async,
        } => Stmt::FunctionDef {
            span: *span,
            attributes: attributes.clone(),
            contracts: contracts
                .iter()
                .map(|c| go_contract(arena, c, old_name, new_name))
                .collect(),
            doc: doc.clone(),
            name: *name,
            params: params.clone(),
            return_type: return_type.clone(),
            // Capture avoidance (same as the Closure arm): if a parameter
            // shadows the renamed identifier, the body's bound occurrences
            // must NOT be renamed — otherwise the parameter name survives
            // while its uses point at the fresh name (dangling
            // references).
            body: if params.iter().any(|p| p.name == old_name) {
                body.clone()
            } else {
                body.as_ref()
                    .map(|ss| go_stmts(arena, ss, old_name, new_name))
            },
            type_params: type_params.clone(),
            where_clause: where_clause.clone(),
            finally: if params.iter().any(|p| p.name == old_name) {
                finally.clone()
            } else {
                finally
                    .as_ref()
                    .map(|ss| go_stmts(arena, ss, old_name, new_name))
            },
            is_comptime: *is_comptime,
            is_async: *is_async,
        },
        Stmt::TypeDef {
            span,
            attributes,
            doc,
            name,
            params,
            definition,
            contracts,
        } => Stmt::TypeDef {
            span: *span,
            attributes: attributes.clone(),
            doc: doc.clone(),
            name: *name,
            params: params.clone(),
            definition: definition.clone(),
            contracts: contracts
                .iter()
                .map(|c| go_contract(arena, c, old_name, new_name))
                .collect(),
        },
        Stmt::TraitDef {
            span,
            attributes,
            doc,
            name,
            methods,
            associated_types,
        } => Stmt::TraitDef {
            span: *span,
            attributes: attributes.clone(),
            doc: doc.clone(),
            name: *name,
            methods: methods.clone(),
            associated_types: associated_types.clone(),
        },
        Stmt::Import {
            path,
            items,
            alias,
            span,
        } => Stmt::Import {
            path: path.clone(),
            items: items.clone(),
            alias: *alias,
            span: *span,
        },
        Stmt::ExternFunction {
            abi,
            name,
            params,
            return_type,
            span,
            attributes,
        } => Stmt::ExternFunction {
            abi: abi.clone(),
            name: *name,
            params: params.clone(),
            return_type: return_type.clone(),
            span: *span,
            attributes: attributes.clone(),
        },
        Stmt::Constraint {
            name,
            params,
            predicates,
            span,
        } => Stmt::Constraint {
            name: *name,
            params: params.clone(),
            predicates: predicates.clone(),
            span: *span,
        },
        Stmt::Edition(s, span) => Stmt::Edition(s.clone(), *span),
        Stmt::Expression(e) => Stmt::Expression(go_expr(arena, e, old_name, new_name)),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            span,
        } => Stmt::If {
            cond: go_expr(arena, cond, old_name, new_name),
            then_branch: go_stmts(arena, then_branch, old_name, new_name),
            else_branch: else_branch
                .as_ref()
                .map(|ss| go_stmts(arena, ss, old_name, new_name)),
            span: *span,
        },
        Stmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            span,
        } => {
            // Capture avoidance: a pattern binding `old_name` shadows the
            // outer name — but ONLY within the then branch.  The pattern's
            // bindings are scoped to the then branch (the checker pops the
            // `with_gadt_arm` scope before checking the else branch), so the
            // else branch stays in the outer scope and references to
            // `old_name` there refer to the rename target and MUST be
            // renamed regardless of what the pattern binds.
            let mut bv = Vec::new();
            bound_vars(pattern, &mut bv);
            let shadowed = bv.contains(&old_name);
            Stmt::IfLet {
                pattern: pattern.clone(),
                scrutinee: go_expr(arena, scrutinee, old_name, new_name),
                then_branch: if shadowed {
                    then_branch.clone()
                } else {
                    go_stmts(arena, then_branch, old_name, new_name)
                },
                else_branch: else_branch
                    .as_ref()
                    .map(|ss| go_stmts(arena, ss, old_name, new_name)),
                span: *span,
            }
        }
        Stmt::While {
            label,
            cond,
            body,
            invariant,
            decreases,
            span,
        } => Stmt::While {
            label: *label,
            cond: go_expr(arena, cond, old_name, new_name),
            body: go_stmts(arena, body, old_name, new_name),
            invariant: invariant
                .as_ref()
                .map(|e| go_expr(arena, e, old_name, new_name)),
            decreases: decreases
                .as_ref()
                .map(|e| go_expr(arena, e, old_name, new_name)),
            span: *span,
        },
        Stmt::WhileLet {
            label,
            pattern,
            scrutinee,
            body,
            invariant,
            decreases,
            span,
        } => {
            // Capture avoidance: a pattern binding `old_name`
            // shadows the outer name — the body must NOT be renamed.
            let mut bv = Vec::new();
            bound_vars(pattern, &mut bv);
            let shadowed = bv.contains(&old_name);
            Stmt::WhileLet {
                label: *label,
                pattern: pattern.clone(),
                scrutinee: go_expr(arena, scrutinee, old_name, new_name),
                body: if shadowed {
                    body.clone()
                } else {
                    go_stmts(arena, body, old_name, new_name)
                },
                invariant: invariant
                    .as_ref()
                    .map(|e| go_expr(arena, e, old_name, new_name)),
                decreases: decreases
                    .as_ref()
                    .map(|e| go_expr(arena, e, old_name, new_name)),
                span: *span,
            }
        }
        Stmt::For {
            label,
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            span,
        } => {
            // Capture avoidance: a pattern binding `old_name`
            // shadows the outer name — the loop body must NOT be
            // renamed.
            let mut bv = Vec::new();
            bound_vars(pattern, &mut bv);
            let shadowed = bv.contains(&old_name);
            Stmt::For {
                label: *label,
                pattern: pattern.clone(),
                iterable: go_expr(arena, iterable, old_name, new_name),
                body: if shadowed {
                    body.clone()
                } else {
                    go_stmts(arena, body, old_name, new_name)
                },
                invariant: invariant
                    .as_ref()
                    .map(|e| go_expr(arena, e, old_name, new_name)),
                decreases: decreases
                    .as_ref()
                    .map(|e| go_expr(arena, e, old_name, new_name)),
                span: *span,
            }
        }
        Stmt::Loop { label, body, span } => Stmt::Loop {
            label: *label,
            body: go_stmts(arena, body, old_name, new_name),
            span: *span,
        },
        Stmt::Leave { label, span } => Stmt::Leave {
            label: *label,
            span: *span,
        },
        Stmt::Continue { label, span } => Stmt::Continue {
            label: *label,
            span: *span,
        },
        Stmt::Return {
            value,
            labels,
            span,
        } => Stmt::Return {
            value: value
                .as_ref()
                .map(|e| go_expr(arena, e, old_name, new_name)),
            labels: labels.clone(),
            span: *span,
        },
        Stmt::Assign {
            target,
            op,
            value,
            span,
        } => Stmt::Assign {
            target: replace_ident_in_expr(arena, target, old_name, new_name),
            op: *op,
            value: go_expr(arena, value, old_name, new_name),
            span: *span,
        },
        Stmt::ComptimeBlock {
            captures,
            trusted,
            attributes,
            body,
            span,
        } => Stmt::ComptimeBlock {
            captures: captures.clone(),
            trusted: *trusted,
            attributes: attributes.clone(),
            body: go_stmts(arena, body, old_name, new_name),
            span: *span,
        },
        Stmt::Generate {
            attributes,
            for_type,
            body,
            span,
        } => Stmt::Generate {
            attributes: attributes.clone(),
            for_type: for_type.clone(),
            body: go_stmts(arena, body, old_name, new_name),
            span: *span,
        },
        Stmt::ScopeCleanup {
            name,
            when_condition,
            body,
            propagates,
            overrides,
            span,
        } => Stmt::ScopeCleanup {
            name: *name,
            when_condition: when_condition
                .as_ref()
                .map(|e| replace_ident_in_expr(arena, e, old_name, new_name)),
            body: go_stmts(arena, body, old_name, new_name),
            propagates: *propagates,
            overrides: *overrides,
            span: *span,
        },
        Stmt::Trigger { name, span } => Stmt::Trigger {
            name: *name,
            span: *span,
        },
        Stmt::Unsafe { body, span } => Stmt::Unsafe {
            body: go_stmts(arena, body, old_name, new_name),
            span: *span,
        },
        Stmt::GhostVariableDef { inner, span } => Stmt::GhostVariableDef {
            inner: arena.alloc(replace_ident_in_stmt(arena, inner, old_name, new_name)),
            span: *span,
        },
        Stmt::Isolate {
            attributes,
            body,
            span,
        } => Stmt::Isolate {
            attributes: attributes.clone(),
            body: go_stmts(arena, body, old_name, new_name),
            span: *span,
        },
        Stmt::ImplBlock {
            span,
            attributes,
            trait_path,
            for_type,
            methods,
            associated_types,
            where_clause,
            type_params,
        } => Stmt::ImplBlock {
            span: *span,
            attributes: attributes.clone(),
            trait_path: trait_path.clone(),
            for_type: for_type.clone(),
            methods: methods.clone(),
            associated_types: associated_types.clone(),
            where_clause: where_clause.clone(),
            type_params: type_params.clone(),
        },
        Stmt::LayoutDef {
            name,
            attributes,
            span,
        } => Stmt::LayoutDef {
            name: *name,
            attributes: attributes.clone(),
            span: *span,
        },
        Stmt::Error(span) => Stmt::Error(*span),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// True if any `Expr::Ident` named `name` appears (recursively) in `e`.
    fn expr_has_ident(e: &Expr<'_>, name: Symbol) -> bool {
        match e {
            Expr::Ident(s, _) => *s == name,
            Expr::BinaryOp { left, right, .. } => {
                expr_has_ident(left, name) || expr_has_ident(right, name)
            }
            Expr::UnaryOp { expr, .. } => expr_has_ident(expr, name),
            Expr::Call { callee, args, .. } => {
                expr_has_ident(callee, name) || args.iter().any(|a| expr_has_ident(a, name))
            }
            Expr::Block(stmts, _) => stmts_has_ident(stmts, name),
            Expr::IfLet {
                scrutinee,
                then_branch,
                else_branch,
                ..
            } => {
                expr_has_ident(scrutinee, name)
                    || stmts_has_ident(then_branch, name)
                    || else_branch
                        .as_ref()
                        .is_some_and(|eb| stmts_has_ident(eb, name))
            }
            _ => false,
        }
    }

    fn stmts_has_ident(ss: &[Stmt<'_>], name: Symbol) -> bool {
        ss.iter().any(|s| match s {
            Stmt::Expression(e) => expr_has_ident(e, name),
            Stmt::VariableDef { value: Some(v), .. } => expr_has_ident(v, name),
            Stmt::IfLet {
                scrutinee,
                then_branch,
                else_branch,
                ..
            } => {
                expr_has_ident(scrutinee, name)
                    || stmts_has_ident(then_branch, name)
                    || else_branch
                        .as_ref()
                        .is_some_and(|eb| stmts_has_ident(eb, name))
            }
            _ => false,
        })
    }

    fn parse_program(source: &str) -> (Program<'static>, &'static bumpalo::Bump) {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let mut parser = Parser::new(source, arena);
        let prog = parser
            .parse_program()
            .expect("parse should succeed for test source");
        (prog, arena)
    }

    /// Parse a function body and return its first statement.  Top level
    /// only accepts items, so test statements live inside `def f() { ... }`.
    fn first_fn_stmt(source: &str) -> (Stmt<'static>, &'static bumpalo::Bump) {
        let (prog, arena) = parse_program(source);
        let Stmt::FunctionDef {
            body: Some(body), ..
        } = &prog.items[0]
        else {
            panic!("expected a function definition");
        };
        (body[0].clone(), arena)
    }

    /// Regression test: the `if let` pattern's bindings are scoped to the
    /// THEN branch only (the checker pops the `with_gadt_arm` scope before
    /// checking the else branch).  So when the pattern binds the rename
    /// target `value`, the then-branch keeps its `value` (local binder)
    /// while the else-branch's `value` refers to the OUTER variable and
    /// MUST be renamed — previously the else branch was wrongly skipped.
    #[test]
    fn test_replace_ident_stmt_iflet_else_branch_renamed() {
        let (stmt, arena) =
            first_fn_stmt("def f() { if let Some(value) = opt { value } else { value + 1 } }");
        let old = Symbol::intern("value");
        let new = Symbol::intern("_where_1");

        // Sanity: the pattern really binds `value`.
        let Stmt::IfLet { pattern, .. } = &stmt else {
            panic!("expected Stmt::IfLet, got {:?}", stmt);
        };
        let mut bv = Vec::new();
        bound_vars(pattern, &mut bv);
        assert!(bv.contains(&old), "pattern must bind `value`");

        let renamed = replace_ident_in_stmt(arena, &stmt, old, new);
        let Stmt::IfLet {
            then_branch,
            else_branch,
            ..
        } = &renamed
        else {
            panic!("expected Stmt::IfLet after rename");
        };
        // Then-branch is inside the pattern scope: `value` is the local
        // binder and must be preserved.
        assert!(
            stmts_has_ident(then_branch, old),
            "then branch must keep `value`"
        );
        assert!(
            !stmts_has_ident(then_branch, new),
            "then branch must not contain the fresh binder"
        );
        // Else-branch is OUTSIDE the pattern scope: `value` refers to the
        // outer variable and must be renamed to the fresh binder.
        let eb = else_branch.as_ref().expect("if let has else branch");
        assert!(
            !stmts_has_ident(eb, old),
            "else branch must not keep the outer `value`"
        );
        assert!(stmts_has_ident(eb, new), "else branch must be renamed");
    }

    /// Same regression for the expression-position `if let`.
    #[test]
    fn test_replace_ident_expr_iflet_else_branch_renamed() {
        let (stmt, arena) = first_fn_stmt(
            "def f() { set x = if let Some(value) = opt { value } else { value + 1 }; }",
        );
        let Stmt::VariableDef { value: Some(e), .. } = &stmt else {
            panic!("expected VariableDef");
        };
        let Expr::IfLet {
            pattern,
            then_branch,
            else_branch,
            ..
        } = e
        else {
            panic!("expected Expr::IfLet");
        };
        let old = Symbol::intern("value");
        let new = Symbol::intern("_where_1");

        let mut bv = Vec::new();
        bound_vars(pattern, &mut bv);
        assert!(bv.contains(&old), "pattern must bind `value`");
        assert!(
            stmts_has_ident(then_branch, old),
            "pre-condition: then branch uses `value`"
        );
        assert!(
            stmts_has_ident(else_branch.as_ref().unwrap(), old),
            "pre-condition: else branch uses `value`"
        );

        let renamed = replace_ident_in_expr(arena, e, old, new);
        let Expr::IfLet {
            then_branch,
            else_branch,
            ..
        } = renamed
        else {
            panic!("expected Expr::IfLet after rename");
        };
        assert!(
            stmts_has_ident(then_branch, old),
            "then branch keeps `value`"
        );
        assert!(
            !stmts_has_ident(then_branch, new),
            "then branch must not contain the fresh binder"
        );
        let eb = else_branch.as_ref().expect("if let has else branch");
        assert!(
            !stmts_has_ident(eb, old),
            "else branch must not keep the outer `value`"
        );
        assert!(stmts_has_ident(eb, new), "else branch must be renamed");
    }

    /// Control: when the pattern does NOT bind the rename target, BOTH
    /// branches are renamed (no scope boundary in play).
    #[test]
    fn test_replace_ident_iflet_renames_both_when_unshadowed() {
        let (stmt, arena) = first_fn_stmt(
            "def f() { set x = if let Some(other) = opt { value } else { value + 1 }; }",
        );
        let Stmt::VariableDef { value: Some(e), .. } = &stmt else {
            panic!("expected VariableDef");
        };
        let old = Symbol::intern("value");
        let new = Symbol::intern("_where_1");
        let renamed = replace_ident_in_expr(arena, e, old, new);
        let Expr::IfLet {
            then_branch,
            else_branch,
            ..
        } = renamed
        else {
            panic!("expected Expr::IfLet after rename");
        };
        assert!(stmts_has_ident(then_branch, new), "then branch renamed");
        assert!(
            !stmts_has_ident(then_branch, old),
            "then branch must not keep `value`"
        );
        let eb = else_branch.as_ref().expect("if let has else branch");
        assert!(stmts_has_ident(eb, new), "else branch renamed");
        assert!(
            !stmts_has_ident(eb, old),
            "else branch must not keep `value`"
        );
    }
}
