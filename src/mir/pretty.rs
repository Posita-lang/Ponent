//! Debug pretty-printing for MIR bodies (`ponent mir <file>` output).
//!
//! Shape mirrors rustc's `--emit=mir` dump: one function header, one block
//! per `bbN`, statements indented, terminator suffixed.

use super::{MirBody, Operand, Place, Rvalue, Statement, SwitchTarget, Terminator};
use crate::symbol::Symbol;
use std::fmt::Write;

impl MirBody {
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "fn {}(", self.func_name);
        for (i, local) in self.locals.iter().enumerate() {
            if i < self.arg_count {
                let _ = writeln!(out, "    arg _{}: {:?}", i, local.ty);
            }
        }
        let _ = writeln!(out, ") -> {{");
        for (i, local) in self.locals.iter().enumerate() {
            if i >= self.arg_count {
                let kind = if local.is_temp {
                    "temp"
                } else {
                    match local.mutable {
                        true => "mut",
                        false => "let",
                    }
                };
                let name = match local.name {
                    Some(n) => format!(" {}", n),
                    None => String::new(),
                };
                let _ = writeln!(out, "    {} _{}{}: {:?}", kind, i, name, local.ty);
            }
        }
        let _ = writeln!(out, "}}");

        for (bi, block) in self.blocks.iter().enumerate() {
            let _ = writeln!(out, "bb{}: {{", bi);
            for stmt in &block.stmts {
                let _ = writeln!(out, "    {}", stmt.pretty());
            }
            let _ = writeln!(out, "    {}", block.terminator.pretty());
            let _ = writeln!(out, "}}");
        }
        out
    }
}

impl Statement {
    pub fn pretty(&self) -> String {
        match self {
            Statement::Assign(place, rvalue, _) => format!("{} = {}", place, rvalue.pretty()),
            Statement::StorageLive(l, _) => format!("storage live _{}", l.0),
            Statement::StorageDead(l, _) => format!("storage dead _{}", l.0),
            Statement::Drop(place, _) => format!("drop {}", place),
            Statement::Ghost(msg, _) => format!("ghost {}", msg),
            Statement::Unsupported(msg, _) => format!("unsupported {}", msg),
        }
    }
}

impl Rvalue {
    pub fn pretty(&self) -> String {
        match self {
            Rvalue::Use(op) => op.pretty(),
            Rvalue::Ref(place, kind) => {
                let k = match kind {
                    super::BorrowKind::Shared => "&",
                    super::BorrowKind::Mut => "&mut",
                    super::BorrowKind::Ro => "&ro",
                };
                format!("{}{}", k, place)
            }
            Rvalue::UnaryOp(op, operand) => {
                let s = match op {
                    crate::ast::UnaryOp::Neg => "-",
                    crate::ast::UnaryOp::Not => "!",
                    crate::ast::UnaryOp::BitNot => "~",
                    crate::ast::UnaryOp::Deref => "*",
                    crate::ast::UnaryOp::Ref => "&",
                    crate::ast::UnaryOp::RefMut => "&mut",
                    crate::ast::UnaryOp::Ro => "&ro",
                };
                format!("{}{}", s, operand.pretty())
            }
            Rvalue::BinaryOp(op, l, r) => format!("{} {} {}", l.pretty(), op_str(op), r.pretty()),
            Rvalue::Cast(_, operand, ty) => format!("{} as {:?}", operand.pretty(), ty),
            Rvalue::Aggregate(kind, ops) => {
                let inner = ops
                    .iter()
                    .map(|o| o.pretty())
                    .collect::<Vec<_>>()
                    .join(", ");
                match kind {
                    super::AggregateKind::Tuple => format!("({})", inner),
                    super::AggregateKind::Array => format!("[{}]", inner),
                    super::AggregateKind::Struct(fields) => {
                        let parts: Vec<String> = fields
                            .iter()
                            .zip(ops.iter())
                            .map(|(f, o)| format!("{}: {}", f, o.pretty()))
                            .collect();
                        format!("struct {{{}}}", parts.join(", "))
                    }
                    super::AggregateKind::Enum { variant, .. } => {
                        format!("{}({})", variant, inner)
                    }
                }
            }
        }
    }
}

impl Operand {
    pub fn pretty(&self) -> String {
        match self {
            Operand::Copy(p) => format!("copy {}", p),
            Operand::Move(p) => format!("move {}", p),
            Operand::Const(c) => format!("{}", c),
        }
    }
}

impl Terminator {
    pub fn pretty(&self) -> String {
        match self {
            Terminator::Goto(b) => format!("goto bb{}", b.0),
            Terminator::Branch {
                cond,
                then_block,
                else_block,
            } => format!(
                "branch {} -> [true: bb{}, false: bb{}]",
                cond.pretty(),
                then_block.0,
                else_block.0
            ),
            Terminator::Switch { discr, targets } => {
                let arms = match targets {
                    SwitchTarget::Int(arms, _) => arms
                        .iter()
                        .map(|(v, b)| format!("{} => bb{}", v, b.0))
                        .collect::<Vec<_>>(),
                    SwitchTarget::Enum(arms, _) => arms
                        .iter()
                        .map(|(v, b)| format!("{} => bb{}", v, b.0))
                        .collect::<Vec<_>>(),
                };
                let fallback = match targets {
                    SwitchTarget::Int(_, f) => f.0,
                    SwitchTarget::Enum(_, f) => f.0,
                };
                format!(
                    "switch {} -> [{}] otherwise bb{}",
                    discr.pretty(),
                    arms.join(", "),
                    fallback
                )
            }
            Terminator::Call {
                callee,
                args,
                destination,
            } => {
                let args_str = args
                    .iter()
                    .map(|a| a.pretty())
                    .collect::<Vec<_>>()
                    .join(", ");
                let callee_str = match callee {
                    super::Callee::Fn(s) => format!("{}", s),
                    super::Callee::Method { name, .. } => format!("method {}", name),
                    super::Callee::Opaque(s) => format!("opaque {}", s),
                };
                match destination {
                    Some((place, target)) => {
                        format!("{} = {}({}) -> bb{}", place, callee_str, args_str, target.0)
                    }
                    None => format!("{}({})", callee_str, args_str),
                }
            }
            Terminator::Return => "return".to_string(),
            Terminator::Leave => "leave".to_string(),
            Terminator::Unreachable => "unreachable".to_string(),
        }
    }
}

fn op_str(op: &crate::ast::BinOp) -> &'static str {
    use crate::ast::BinOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Rem => "%",
        AddWrap => "+%",
        SubWrap => "-%",
        MulWrap => "*%",
        AddSaturate => "+?",
        SubSaturate => "-?",
        MulSaturate => "*?",
        AddTrap => "+!",
        SubTrap => "-!",
        MulTrap => "*!",
        BitAnd => "&",
        BitOr => "|",
        BitXor => "^",
        Shl => "<<",
        Shr => ">>",
        Eq => "==",
        Neq => "!=",
        Lt => "<",
        Gt => ">",
        Le => "<=",
        Ge => ">=",
        And => "&&",
        Or => "||",
    }
}

/// Convenience: render a `Symbol` (used by tests and the CLI dump).
pub fn sym_str(s: Symbol) -> String {
    format!("{}", s)
}
