//! HIR → MIR lowering.
//!
//! Walks a type-checked `HirStmt::FunctionDef` body and produces a flat
//! `MirBody`.  Every intermediate value gets a temp local; every control
//! structure becomes blocks + terminators.  Constructs that MIR does not
//! yet model (closures, try/catch, comptime, quantifiers, …) are emitted
//! as `Statement::Unsupported` / `MirConst::Opaque` rather than panicking,
//! so a partial lowering is always structurally valid.

use super::{
    AggregateKind, BasicBlock, BlockId, BorrowKind, Callee, CastKind, LocalDecl, LocalId, MirBody,
    MirConst, Operand, Place, Rvalue, Statement, SwitchTarget, Terminator,
};
use crate::ast::{BinOp, Literal, Span, UnaryOp};
use crate::hir::hir::{HirExpr, HirMatchArm, HirPattern, HirStmt};
use crate::hir::types::TypeId;
use crate::symbol::Symbol;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct MirBuilder {
    locals: Vec<LocalDecl>,
    blocks: Vec<BasicBlock>,
    /// Block currently receiving emitted statements.
    cur: BlockId,
    /// Source-name → local for identifier resolution.
    var_map: HashMap<Symbol, LocalId>,
    arg_count: usize,
    func_name: Symbol,
}

impl MirBuilder {
    pub fn new(func_name: Symbol) -> Self {
        MirBuilder {
            locals: Vec::new(),
            blocks: Vec::new(),
            cur: BlockId(0),
            var_map: HashMap::new(),
            arg_count: 0,
            func_name,
        }
    }

    fn new_local(&mut self, decl: LocalDecl) -> LocalId {
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(decl);
        id
    }

    fn new_named_local(&mut self, decl: LocalDecl) -> LocalId {
        let id = self.new_local(decl);
        if let Some(name) = self.locals[id.0 as usize].name {
            self.var_map.insert(name, id);
        }
        id
    }

    fn new_temp(&mut self, ty: TypeId, span: Span) -> LocalId {
        let id = self.new_local(LocalDecl {
            name: None,
            ty,
            mutable: false,
            is_temp: true,
            span,
        });
        id
    }

    fn emit(&mut self, stmt: Statement) {
        self.blocks[self.cur.0].stmts.push(stmt);
    }

    /// Close the current block with `terminator` and open a fresh one.
    fn close_block(&mut self, terminator: Terminator) -> BlockId {
        let id = self.cur;
        self.blocks[id.0].terminator = terminator;
        self.cur = BlockId(self.blocks.len());
        self.blocks.push(BasicBlock {
            stmts: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    /// Begin a detached block (for branch targets); becomes current.
    fn start_block(&mut self) -> BlockId {
        self.cur = BlockId(self.blocks.len());
        self.blocks.push(BasicBlock {
            stmts: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        self.cur
    }

    fn patch_terminator(&mut self, block: BlockId, terminator: Terminator) {
        self.blocks[block.0].terminator = terminator;
    }

    /// Allocate an empty block without switching to it (for forward
    /// references in switch/branch terminators).
    fn new_empty_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(BasicBlock {
            stmts: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    /// Make `block` the current emission target.
    fn set_cur(&mut self, block: BlockId) {
        self.cur = block;
    }

    fn finish(mut self) -> MirBody {
        // The current block is the fallthrough continuation; if nothing was
        // emitted into it, it is the function's implicit exit.
        let cur = &mut self.blocks[self.cur.0];
        if cur.stmts.is_empty() && cur.terminator == Terminator::Unreachable {
            cur.terminator = Terminator::Return;
        }
        MirBody {
            blocks: self.blocks,
            locals: self.locals,
            arg_count: self.arg_count,
            entry: BlockId(0),
            func_name: self.func_name,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Lower a type-checked function definition into a `MirBody`.
pub fn lower_function(func: &HirStmt<'_>) -> MirBody {
    let HirStmt::FunctionDef {
        name, params, body, ..
    } = func
    else {
        // Not a function: caller should not call this.  Produce an empty body.
        let mut b = MirBuilder::new(Symbol::from("??"));
        b.close_block(Terminator::Return);
        return b.finish();
    };

    let mut b = MirBuilder::new(*name);
    b.arg_count = params.len();

    for p in params {
        b.new_named_local(LocalDecl {
            name: Some(p.name),
            ty: p.ty,
            mutable: false,
            is_temp: false,
            span: p.span,
        });
    }

    // bb0: entry block (may stay empty for bodiless decls).
    b.start_block();

    match body {
        Some(stmts) => {
            lower_stmts(&mut b, stmts);
        }
        None => {
            // Declaration without body (trait method, extern) — nothing to lower.
        }
    }

    b.finish()
}

fn lower_stmts(b: &mut MirBuilder, stmts: &[HirStmt<'_>]) {
    for stmt in stmts {
        lower_stmt(b, stmt);
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn lower_stmt(b: &mut MirBuilder, stmt: &HirStmt<'_>) {
    match stmt {
        HirStmt::VariableDef {
            mutable,
            name,
            pattern,
            ty,
            value,
            span,
            ..
        } => {
            let local = b.new_named_local(LocalDecl {
                name: *name,
                ty: *ty,
                mutable: *mutable,
                is_temp: false,
                span: *span,
            });
            if let Some(value) = value {
                let operand = lower_expr(b, value);
                b.emit(Statement::Assign(
                    Place::Local(local),
                    Rvalue::Use(operand),
                    *span,
                ));
            }
            // Destructuring patterns beyond a plain identifier are not yet
            // modeled — bindings are registered for `Ident` patterns only.
            if let Some(HirPattern::Ident(bind_name, _, _)) = pattern {
                if *name != Some(*bind_name) {
                    b.emit(Statement::Unsupported(
                        format!("pattern binding {}", bind_name),
                        *span,
                    ));
                }
            } else if pattern.is_some() {
                b.emit(Statement::Unsupported("destructure pattern".into(), *span));
            }
        }

        HirStmt::Assign {
            target,
            op,
            value,
            span,
        } => {
            if let Some(op) = op {
                // Compound assignment `x op= v` desugars to x = x op v.
                let lhs = lower_place(b, target);
                let lhs_ty = target.ty();
                let lhs_val = operand_of_place(lhs.clone(), lhs_ty);
                let rhs = lower_expr(b, value);
                let tmp = b.new_temp(lhs_ty, *span);
                b.emit(Statement::Assign(
                    Place::Local(tmp),
                    Rvalue::BinaryOp(*op, lhs_val, rhs),
                    *span,
                ));
                b.emit(Statement::Assign(
                    lhs,
                    Rvalue::Use(Operand::Copy(Place::Local(tmp))),
                    *span,
                ));
            } else {
                let place = lower_place(b, target);
                let operand = lower_expr(b, value);
                b.emit(Statement::Assign(place, Rvalue::Use(operand), *span));
            }
        }

        HirStmt::Return { value, span, .. } => {
            // Return values are not funneled into a `_ret` local yet.
            if let Some(value) = value {
                let _ = lower_expr(b, value);
            }
            b.close_block(Terminator::Return);
            let _ = span;
        }

        HirStmt::Expression(expr) => {
            let _ = lower_expr(b, expr);
        }

        HirStmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            lower_if(b, cond, then_branch, else_branch.as_ref(), stmt.span());
        }

        HirStmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            span,
        } => {
            lower_if_let(
                b,
                pattern,
                scrutinee,
                then_branch,
                else_branch.as_ref(),
                *span,
            );
        }

        HirStmt::While { cond, body, .. } => {
            lower_while(b, cond, body);
        }

        HirStmt::WhileLet {
            pattern,
            scrutinee,
            body,
            span,
            ..
        } => {
            lower_while_let(b, pattern, scrutinee, body, *span);
        }

        HirStmt::Loop { body, .. } => {
            lower_loop(b, body);
        }

        HirStmt::Leave { .. } => {
            // Label resolution is not modeled yet — conservative exit.
            b.close_block(Terminator::Leave);
        }

        HirStmt::Continue { span, .. } => {
            b.emit(Statement::Unsupported("continue".into(), *span));
        }

        HirStmt::Unsafe { body, .. } => lower_stmts(b, body),

        HirStmt::ScopeCleanup {
            name, body, span, ..
        } => {
            // Cleanup bodies are lowered inline after a Ghost marker so the
            // eventual Drop pass can find them.
            b.emit(Statement::Ghost(format!("scope_cleanup {}", name), *span));
            lower_stmts(b, body);
        }

        HirStmt::GhostVariableDef { inner, span } => {
            b.emit(Statement::Ghost("ghost var".into(), *span));
            lower_stmt(b, inner);
        }

        HirStmt::ComptimeBlock { span, .. } => {
            b.emit(Statement::Ghost("comptime block".into(), *span));
        }

        HirStmt::Trigger { name, span } => {
            b.emit(Statement::Ghost(format!("trigger {}", name), *span));
        }

        HirStmt::Isolate { body, .. } => lower_stmts(b, body),

        // Non-body / declaration statements carry no runtime code.
        HirStmt::FunctionDef { .. }
        | HirStmt::TypeDef { .. }
        | HirStmt::TraitDef { .. }
        | HirStmt::Import { .. }
        | HirStmt::ExternFunction { .. }
        | HirStmt::Constraint { .. }
        | HirStmt::Edition(..)
        | HirStmt::LayoutDef { .. }
        | HirStmt::ImplBlock { .. }
        | HirStmt::Generate { .. }
        | HirStmt::Stripped { .. }
        | HirStmt::Error => {}

        HirStmt::For { span, .. } => {
            // For loops desugar in HIR generation upstream; if one reaches
            // here unlowered, mark it rather than mis-lowering.
            b.emit(Statement::Unsupported("for loop".into(), *span));
        }
    }
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

fn lower_if(
    b: &mut MirBuilder,
    cond: &HirExpr<'_>,
    then_branch: &[HirStmt<'_>],
    else_branch: Option<&Vec<HirStmt<'_>>>,
    span: Span,
) {
    let cond_operand = lower_expr(b, cond);
    // Forward-allocate the arm targets, then close the cond block with the
    // branch (cur is still the cond block).
    let then_block = b.new_empty_block();
    let else_block = b.new_empty_block();
    b.close_block(Terminator::Branch {
        cond: cond_operand,
        then_block,
        else_block,
    });

    // Then arm.
    b.set_cur(then_block);
    lower_stmts(b, then_branch);
    let then_end = b.close_block(Terminator::Unreachable);

    // Else arm.
    let join = b.new_empty_block();
    b.set_cur(else_block);
    if let Some(else_branch) = else_branch {
        lower_stmts(b, else_branch);
        let else_end = b.close_block(Terminator::Unreachable);
        b.patch_terminator(else_end, Terminator::Goto(join));
    } else {
        b.close_block(Terminator::Goto(join));
    }
    b.patch_terminator(then_end, Terminator::Goto(join));
    b.set_cur(join);
    let _ = span;
}

fn lower_if_let(
    b: &mut MirBuilder,
    pattern: &HirPattern<'_>,
    scrutinee: &HirExpr<'_>,
    then_branch: &[HirStmt<'_>],
    else_branch: Option<&Vec<HirStmt<'_>>>,
    span: Span,
) {
    // Conservative first pass: evaluate the scrutinee, register the pattern's
    // identifier bindings (no runtime discrimination yet — pattern
    // refutability is a later pass), then branch as a plain if.
    let _ = lower_expr(b, scrutinee);
    bind_pattern_idents(b, pattern);
    b.emit(Statement::Unsupported("if_let discrimination".into(), span));
    lower_if(
        b,
        &HirExpr::Literal(Literal::Bool(true), TypeId::NONE, span),
        then_branch,
        else_branch,
        span,
    );
}

fn lower_while(b: &mut MirBuilder, cond: &HirExpr<'_>, body: &[HirStmt<'_>]) {
    let head = b.new_empty_block();
    let cond_operand = lower_expr(b, cond);
    let body_block = b.new_empty_block();
    let exit_block = b.new_empty_block();
    b.close_block(Terminator::Branch {
        cond: cond_operand,
        then_block: body_block,
        else_block: exit_block,
    });
    // The cond block jumps back to `head` at loop entry.
    b.patch_terminator(BlockId(head.0 - 1), Terminator::Goto(head));

    b.set_cur(body_block);
    lower_stmts(b, body);
    b.close_block(Terminator::Goto(head));
    b.set_cur(exit_block);
}

fn lower_while_let(
    b: &mut MirBuilder,
    pattern: &HirPattern<'_>,
    scrutinee: &HirExpr<'_>,
    body: &[HirStmt<'_>],
    span: Span,
) {
    let head = b.new_empty_block();
    let _ = lower_expr(b, scrutinee);
    bind_pattern_idents(b, pattern);
    b.emit(Statement::Unsupported(
        "while_let discrimination".into(),
        span,
    ));
    let body_block = b.new_empty_block();
    let exit_block = b.new_empty_block();
    b.close_block(Terminator::Branch {
        cond: Operand::Const(MirConst::Bool(true)),
        then_block: body_block,
        else_block: exit_block,
    });
    b.patch_terminator(BlockId(head.0 - 1), Terminator::Goto(head));

    b.set_cur(body_block);
    lower_stmts(b, body);
    b.close_block(Terminator::Goto(head));
    b.set_cur(exit_block);
}

fn lower_loop(b: &mut MirBuilder, body: &[HirStmt<'_>]) {
    let head = b.start_block();
    lower_stmts(b, body);
    b.close_block(Terminator::Goto(head));
}

fn lower_match(b: &mut MirBuilder, scrutinee: &HirExpr<'_>, arms: &[HirMatchArm<'_>], span: Span) {
    let discr = lower_expr(b, scrutinee);

    // Enum-scrutinee matches without guards lower to a variant switch.
    let all_enum: Vec<(&HirMatchArm<'_>, Symbol)> = arms
        .iter()
        .filter_map(|arm| match (&arm.pattern, &arm.guard) {
            (HirPattern::Enum { variant, .. }, None) => Some((arm, *variant)),
            _ => None,
        })
        .collect();

    if !all_enum.is_empty() && all_enum.len() == arms.len() {
        // Forward-allocate arm blocks and the fallback, close the head with
        // the switch, then fill each arm and converge on the join block.
        let arm_blocks: Vec<BlockId> = (0..all_enum.len()).map(|_| b.new_empty_block()).collect();
        let fallback = b.new_empty_block();

        let head = b.close_block(Terminator::Switch {
            discr,
            targets: SwitchTarget::Enum(
                all_enum.iter().map(|(_, v)| (*v, BlockId(0))).collect(),
                fallback,
            ),
        });
        // Patch the placeholder target ids now that arm blocks are allocated.
        if let Terminator::Switch {
            targets: SwitchTarget::Enum(entries, _),
            ..
        } = &mut b.blocks[head.0].terminator
        {
            for (entry, block) in entries.iter_mut().zip(arm_blocks.iter()) {
                *entry = (entry.0, *block);
            }
        }

        for ((arm, _), arm_block) in all_enum.iter().zip(arm_blocks.iter()) {
            b.set_cur(*arm_block);
            let result = lower_expr(b, &arm.body);
            let arm_tmp = b.new_temp(arm.body.ty(), arm.body.span());
            b.emit(Statement::Assign(
                Place::Local(arm_tmp),
                Rvalue::Use(result),
                arm.body.span(),
            ));
            b.close_block(Terminator::Unreachable);
        }
        // Convergence: patch every arm's tail (and mark fallback) to join.
        let join = b.cur;
        for (i, _) in all_enum.iter().enumerate() {
            let arm_end = BlockId(arm_blocks[i].0 + 1);
            b.patch_terminator(arm_end, Terminator::Goto(join));
        }
        let _ = span;
        return;
    }

    // Generic fallback: sequential evaluation with Unsupported markers.
    for arm in arms {
        if let Some(guard) = &arm.guard {
            let _ = lower_expr(b, guard);
        }
        bind_pattern_idents(b, &arm.pattern);
        b.emit(Statement::Unsupported("match discrimination".into(), span));
        let _ = lower_expr(b, &arm.body);
    }
}

fn bind_pattern_idents(b: &mut MirBuilder, pattern: &HirPattern<'_>) {
    match pattern {
        HirPattern::Ident(name, ty, span) => {
            b.new_named_local(LocalDecl {
                name: Some(*name),
                ty: *ty,
                mutable: false,
                is_temp: false,
                span: *span,
            });
        }
        HirPattern::Tuple(subs, _) | HirPattern::Or(subs, _) => {
            for sub in subs {
                bind_pattern_idents(b, sub);
            }
        }
        HirPattern::Struct { fields, .. } => {
            for (_, sub) in fields {
                bind_pattern_idents(b, sub);
            }
        }
        HirPattern::Enum {
            inner: Some(sub), ..
        } => bind_pattern_idents(b, sub),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// Lower an expression to an operand: a copy/move of a place, or a constant.
/// Intermediate computations are assigned to temps.
fn lower_expr(b: &mut MirBuilder, expr: &HirExpr<'_>) -> Operand {
    match expr {
        HirExpr::Literal(lit, _, _) => Operand::Const(lower_literal(lit)),

        HirExpr::Ident(name, _, _) => {
            match b.var_map.get(name).copied() {
                Some(local) => Operand::Copy(Place::Local(local)),
                None => {
                    // Unresolved here (e.g. fn item reference) — opaque const.
                    Operand::Const(MirConst::Opaque)
                }
            }
        }

        HirExpr::TypeAnnotated { expr, .. } => lower_expr(b, expr),

        HirExpr::BinaryOp {
            left,
            op,
            right,
            ty,
            span,
        } => {
            let l = lower_expr(b, left);
            let r = lower_expr(b, right);
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::BinaryOp(*op, l, r),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::UnaryOp {
            op,
            expr: inner,
            ty,
            span,
        } => {
            match op {
                UnaryOp::Deref => {
                    // `*e` used as a value reads through the pointer.
                    let place = lower_place(b, inner);
                    let tmp = b.new_temp(*ty, *span);
                    b.emit(Statement::Assign(
                        Place::Local(tmp),
                        Rvalue::Use(operand_of_place(place, *ty)),
                        *span,
                    ));
                    Operand::Copy(Place::Local(tmp))
                }
                UnaryOp::Ref | UnaryOp::RefMut | UnaryOp::Ro => {
                    let place = lower_place(b, inner);
                    let kind = match op {
                        UnaryOp::Ref => BorrowKind::Shared,
                        UnaryOp::RefMut => BorrowKind::Mut,
                        _ => BorrowKind::Ro,
                    };
                    let tmp = b.new_temp(*ty, *span);
                    b.emit(Statement::Assign(
                        Place::Local(tmp),
                        Rvalue::Ref(place, kind),
                        *span,
                    ));
                    Operand::Copy(Place::Local(tmp))
                }
                _ => {
                    let operand = lower_expr(b, inner);
                    let tmp = b.new_temp(*ty, *span);
                    b.emit(Statement::Assign(
                        Place::Local(tmp),
                        Rvalue::UnaryOp(*op, operand),
                        *span,
                    ));
                    Operand::Copy(Place::Local(tmp))
                }
            }
        }

        HirExpr::Call {
            callee,
            args,
            ty,
            span,
            ..
        } => {
            let lowered_args: Vec<Operand> = args.iter().map(|a| lower_expr(b, a)).collect();
            let callee_kind = match &**callee {
                HirExpr::Ident(name, _, _) => Callee::Fn(*name),
                HirExpr::FieldAccess { field, .. } | HirExpr::AttrAccess { attr: field, .. } => {
                    Callee::Method {
                        receiver: Operand::Const(MirConst::Opaque),
                        name: *field,
                    }
                }
                _ => Callee::Opaque(Symbol::from("<expr>")),
            };
            let dest = b.new_temp(*ty, *span);
            b.close_block(Terminator::Call {
                callee: callee_kind,
                args: lowered_args,
                destination: Some((Place::Local(dest), b.cur)),
            });
            Operand::Copy(Place::Local(dest))
        }

        HirExpr::Index {
            base,
            index,
            ty,
            span,
        } => {
            let base_place = lower_place(b, base);
            let idx_local = lower_index_operand(b, index);
            let place = Place::Index(Box::new(base_place), idx_local);
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Use(operand_of_place(place, *ty)),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::FieldAccess {
            base,
            field,
            ty,
            span,
        } => {
            let base_place = lower_place(b, base);
            let place = Place::Field(Box::new(base_place), *field, *ty);
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Use(operand_of_place(place, *ty)),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::AttrAccess {
            base,
            attr,
            ty,
            span,
        } => {
            // Attribute access (method/associated item in expression
            // position) is opaque to the first-pass lowering.
            let _ = lower_expr(b, base);
            let _ = attr;
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Unsupported("attr access value".into(), *span));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::Cast {
            expr: inner,
            ty,
            safe,
            span,
            rounding: _,
        } => {
            let operand = lower_expr(b, inner);
            let kind = if *safe {
                CastKind::Prim
            } else {
                CastKind::Unsafe
            };
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Cast(kind, operand, *ty),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::StructLit {
            fields, ty, span, ..
        } => {
            let mut names = Vec::with_capacity(fields.len());
            let mut ops = Vec::with_capacity(fields.len());
            for (name, value) in fields {
                names.push(*name);
                ops.push(lower_expr(b, value));
            }
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Aggregate(AggregateKind::Struct(names), ops),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::EnumLit {
            variant,
            payload,
            ty,
            span,
            ..
        } => {
            let ops: Vec<Operand> = match payload {
                Some(p) => vec![lower_expr(b, p)],
                None => vec![],
            };
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Aggregate(
                    AggregateKind::Enum {
                        variant: *variant,
                        field_count: ops.len(),
                    },
                    ops,
                ),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::Move(inner, _, _) => {
            let place = lower_place(b, inner);
            Operand::Move(place)
        }

        HirExpr::Tuple(elems, ty, span) => {
            let ops: Vec<Operand> = elems.iter().map(|e| lower_expr(b, e)).collect();
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Aggregate(AggregateKind::Tuple, ops),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::Array(elems, ty, span) => {
            let ops: Vec<Operand> = elems.iter().map(|e| lower_expr(b, e)).collect();
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Aggregate(AggregateKind::Array, ops),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::Block(stmts, ty, span) => {
            lower_stmts(b, stmts);
            // Block expressions without a trailing value evaluate to unit.
            let tmp = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(tmp),
                Rvalue::Use(Operand::Const(MirConst::Unit)),
                *span,
            ));
            Operand::Copy(Place::Local(tmp))
        }

        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ty,
            span,
            ..
        } => lower_expr_if(b, cond, then_branch, else_branch.as_ref(), *ty, *span),

        HirExpr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ty,
            span,
        } => {
            let _ = lower_expr(b, scrutinee);
            bind_pattern_idents(b, pattern);
            b.emit(Statement::Unsupported(
                "if_let discrimination".into(),
                *span,
            ));
            lower_expr_if(
                b,
                &HirExpr::Literal(Literal::Bool(true), TypeId::NONE, *span),
                then_branch,
                else_branch.as_ref(),
                *ty,
                *span,
            )
        }

        HirExpr::Match {
            scrutinee,
            arms,
            ty,
            span,
        } => lower_expr_match(b, scrutinee, arms, *ty, *span),

        // Constructs the first-pass lowering does not model: evaluate
        // sub-expressions (so effects/reads still appear in the MIR), mark
        // the value unsupported, and hand back an opaque temp.
        HirExpr::Range { .. }
        | HirExpr::Closure { .. }
        | HirExpr::Try { .. }
        | HirExpr::UnsafeBlock { .. }
        | HirExpr::Catch { .. }
        | HirExpr::LeaveWith { .. }
        | HirExpr::Return { .. }
        | HirExpr::Await { .. }
        | HirExpr::PolyBox { .. }
        | HirExpr::PolyUnbox { .. }
        | HirExpr::Quantified { .. }
        | HirExpr::Old { .. }
        | HirExpr::Task { .. }
        | HirExpr::TypeInfo(..)
        | HirExpr::LayoutOf(..)
        | HirExpr::CompileError(..)
        | HirExpr::Error(_) => lower_expr_opaque(b, expr),
    }
}

/// Lower an if-expression whose value is consumed: each arm assigns its
/// result to a shared temp, then control converges.
fn lower_expr_if(
    b: &mut MirBuilder,
    cond: &HirExpr<'_>,
    then_branch: &[HirStmt<'_>],
    else_branch: Option<&Vec<HirStmt<'_>>>,
    ty: TypeId,
    span: Span,
) -> Operand {
    let result = b.new_temp(ty, span);
    let cond_operand = lower_expr(b, cond);
    let then_block = b.new_empty_block();
    let else_block = b.new_empty_block();
    b.close_block(Terminator::Branch {
        cond: cond_operand,
        then_block,
        else_block,
    });

    // Then arm.
    b.set_cur(then_block);
    lower_stmts(b, then_branch);
    b.emit(Statement::Assign(
        Place::Local(result),
        Rvalue::Use(Operand::Const(MirConst::Unit)),
        span,
    ));
    let then_end = b.close_block(Terminator::Unreachable);

    // Else arm.
    b.set_cur(else_block);
    if let Some(else_branch) = else_branch {
        lower_stmts(b, else_branch);
    }
    b.emit(Statement::Assign(
        Place::Local(result),
        Rvalue::Use(Operand::Const(MirConst::Unit)),
        span,
    ));
    let else_end = b.close_block(Terminator::Unreachable);

    let join = b.new_empty_block();
    b.patch_terminator(then_end, Terminator::Goto(join));
    b.patch_terminator(else_end, Terminator::Goto(join));
    b.set_cur(join);

    Operand::Copy(Place::Local(result))
}

/// Lower a match-expression: enum switches assign per-arm results into a
/// shared temp; other matches degrade to the sequential fallback.
fn lower_expr_match(
    b: &mut MirBuilder,
    scrutinee: &HirExpr<'_>,
    arms: &[HirMatchArm<'_>],
    ty: TypeId,
    span: Span,
) -> Operand {
    let result = b.new_temp(ty, span);
    let discr = lower_expr(b, scrutinee);

    let all_enum: Vec<(&HirMatchArm<'_>, Symbol)> = arms
        .iter()
        .filter_map(|arm| match (&arm.pattern, &arm.guard) {
            (HirPattern::Enum { variant, .. }, None) => Some((arm, *variant)),
            _ => None,
        })
        .collect();

    if !all_enum.is_empty() && all_enum.len() == arms.len() {
        let arm_blocks: Vec<BlockId> = (0..all_enum.len()).map(|_| b.new_empty_block()).collect();
        let fallback = b.new_empty_block();
        let mut arm_ends: Vec<BlockId> = Vec::new();

        // Close the scrutinee block with the variant switch (arm blocks were
        // already forward-allocated, so targets are known).
        b.close_block(Terminator::Switch {
            discr,
            targets: SwitchTarget::Enum(
                all_enum
                    .iter()
                    .map(|(_, v)| *v)
                    .zip(arm_blocks.iter().copied())
                    .collect(),
                fallback,
            ),
        });

        for ((arm, _), arm_block) in all_enum.iter().zip(arm_blocks.iter()) {
            b.set_cur(*arm_block);
            let value = lower_expr(b, &arm.body);
            b.emit(Statement::Assign(
                Place::Local(result),
                Rvalue::Use(value),
                arm.body.span(),
            ));
            let arm_end = b.close_block(Terminator::Unreachable);
            arm_ends.push(arm_end);
        }

        let join = b.new_empty_block();
        for arm_end in arm_ends {
            b.patch_terminator(arm_end, Terminator::Goto(join));
        }
        b.set_cur(join);
        return Operand::Copy(Place::Local(result));
    }

    // Fallback: sequential evaluation, result is whatever the last arm
    // produced; discrimination is marked unsupported.
    for arm in arms {
        if let Some(guard) = &arm.guard {
            let _ = lower_expr(b, guard);
        }
        bind_pattern_idents(b, &arm.pattern);
        b.emit(Statement::Unsupported("match discrimination".into(), span));
        let value = lower_expr(b, &arm.body);
        b.emit(Statement::Assign(
            Place::Local(result),
            Rvalue::Use(value),
            arm.body.span(),
        ));
    }
    Operand::Copy(Place::Local(result))
}

/// For unsupported constructs: surface any nested reads, then produce an
/// opaque temp so the surrounding lowering stays well-formed.
fn lower_expr_opaque(b: &mut MirBuilder, expr: &HirExpr<'_>) -> Operand {
    let ty = expr.ty();
    let span = expr.span();
    b.emit(Statement::Unsupported("expression".into(), span));
    let tmp = b.new_temp(ty, span);
    b.emit(Statement::Assign(
        Place::Local(tmp),
        Rvalue::Use(Operand::Const(MirConst::Opaque)),
        span,
    ));
    Operand::Copy(Place::Local(tmp))
}

/// Lower an expression for use as an lvalue root/projection base.
fn lower_place(b: &mut MirBuilder, expr: &HirExpr<'_>) -> Place {
    match expr {
        HirExpr::Ident(name, _, _) => match b.var_map.get(name).copied() {
            Some(local) => Place::Local(local),
            None => {
                let tmp = b.new_temp(expr.ty(), expr.span());
                b.emit(Statement::Unsupported(
                    format!("unresolved place {}", name),
                    expr.span(),
                ));
                Place::Local(tmp)
            }
        },
        HirExpr::TypeAnnotated { expr, .. } => lower_place(b, expr),
        HirExpr::FieldAccess {
            base, field, ty, ..
        } => Place::Field(Box::new(lower_place(b, base)), *field, *ty),
        HirExpr::Index { base, index, .. } => {
            let idx_local = lower_index_operand(b, index);
            Place::Index(Box::new(lower_place(b, base)), idx_local)
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr: inner,
            ..
        } => Place::Deref(Box::new(lower_place(b, inner))),
        _ => {
            // Rvalue expression as place root: evaluate to a temp first.
            let operand = lower_expr(b, expr);
            match operand {
                Operand::Copy(place) | Operand::Move(place) => place,
                Operand::Const(c) => {
                    let tmp = b.new_temp(expr.ty(), expr.span());
                    b.emit(Statement::Assign(
                        Place::Local(tmp),
                        Rvalue::Use(Operand::Const(c)),
                        expr.span(),
                    ));
                    Place::Local(tmp)
                }
            }
        }
    }
}

/// Lower `a[i]`'s index into a temp local (dynamic index).
fn lower_index_operand(b: &mut MirBuilder, index: &HirExpr<'_>) -> LocalId {
    if let HirExpr::Literal(Literal::Int(small), ty, span) = index {
        if let crate::ast::IntLit::Small(v) = small {
            // Constant index: no temp needed, but callers use
            // Place::Index uniformly — stash it in a temp local.
            let idx = b.new_temp(*ty, *span);
            b.emit(Statement::Assign(
                Place::Local(idx),
                Rvalue::Use(Operand::Const(MirConst::Int(*v))),
                *span,
            ));
            return idx;
        }
    }
    match lower_expr(b, index) {
        Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => l,
        _ => {
            let idx = b.new_temp(index.ty(), index.span());
            b.emit(Statement::Assign(
                Place::Local(idx),
                Rvalue::Use(Operand::Const(MirConst::Opaque)),
                index.span(),
            ));
            idx
        }
    }
}

fn operand_of_place(place: Place, _ty: TypeId) -> Operand {
    Operand::Copy(place)
}

fn lower_literal(lit: &Literal) -> MirConst {
    match lit {
        Literal::Int(crate::ast::IntLit::Small(v)) => MirConst::Int(*v),
        Literal::Int(crate::ast::IntLit::Large(_)) => MirConst::Opaque,
        Literal::Float(v) => MirConst::Float(*v),
        Literal::Bool(v) => MirConst::Bool(*v),
        Literal::Char(v) => MirConst::Char(*v),
        // A byte char `b'x'` is just the integer 0..=255 at runtime.
        Literal::Byte(v) => MirConst::Int(*v as i128),
        Literal::String(s) => MirConst::Str(Symbol::from(s.as_str())),
        Literal::ByteString(_) => MirConst::Opaque,
    }
}
