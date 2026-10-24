//! MIR — Mid-level IR for Ponent.
//!
//! Lowers type-checked HIR bodies into flat basic blocks of `Statement`s
//! with `Place`-based operands, following the rustc MIR shape.  Consumers
//! (Polonius facts, move checking, later Cranelift codegen) walk `MirBody`
//! instead of re-traversing `HirExpr` trees.
//!
//! Design notes:
//! - Locals are indexed `LocalId(u32)`; params occupy indices `0..arg_count`.
//! - `RegionId` is reused from `hir::types` so borrow facts can be lifted
//!   back into the existing Polonius pipeline without renumbering.
//! - Ghost/contract statements keep their own `Ghost` variant so the
//!   ghost/runtime separation established in the checker is preserved.

use crate::ast::{BinOp, Span, UnaryOp};
use crate::hir::types::TypeId;
use crate::symbol::Symbol;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Index into `MirBody::locals`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(pub u32);

/// Index into `MirBody::blocks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub usize);

/// Field name within a struct/enum-variant payload (kept symbolic until the
/// layout pass resolves indices).
pub type FieldName = Symbol;

// ---------------------------------------------------------------------------
// Places
// ---------------------------------------------------------------------------

/// An lvalue: a local root plus projections.
#[derive(Debug, Clone, PartialEq)]
pub enum Place {
    Local(LocalId),
    Field(Box<Place>, FieldName, TypeId),
    /// Dynamic index `a[i]` — the index lives in a temp local.
    Index(Box<Place>, LocalId),
    /// Constant index `a[0]` (exact place, per `place_is_prefix_of`).
    ConstIndex(Box<Place>, u64, TypeId),
    Deref(Box<Place>),
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Place::Local(id) => write!(f, "_{}", id.0),
            Place::Field(base, idx, _) => write!(f, "{}.{}", base, idx),
            Place::Index(base, idx) => write!(f, "{}[_{}]", base, idx.0),
            Place::ConstIndex(base, idx, _) => write!(f, "{}[{}]", base, idx),
            Place::Deref(base) => write!(f, "(*{})", base),
        }
    }
}

impl Place {
    /// The local this place is rooted at (follows through all projections).
    pub fn root_local(&self) -> LocalId {
        match self {
            Place::Local(id) => *id,
            Place::Field(base, ..)
            | Place::Index(base, _)
            | Place::ConstIndex(base, ..)
            | Place::Deref(base) => base.root_local(),
        }
    }
}

// ---------------------------------------------------------------------------
// Operands & rvalues
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    Shared,
    Mut,
    /// `&ro` — read-only borrow that freezes a `&mut`.
    Ro,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(MirConst),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Copy(p) => write!(f, "copy {}", p),
            Operand::Move(p) => write!(f, "move {}", p),
            Operand::Const(c) => write!(f, "{}", c),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MirConst {
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(u32),
    Str(Symbol),
    Unit,
    /// Unknown/unlowered constant — never produced by a successful lowering.
    Opaque,
}

impl fmt::Display for MirConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirConst::Int(v) => write!(f, "{}", v),
            MirConst::Float(v) => write!(f, "{}", v),
            MirConst::Bool(v) => write!(f, "{}", v),
            MirConst::Char(v) => write!(f, "{:?}", v),
            MirConst::Str(s) => write!(f, "{:?}", s),
            MirConst::Unit => write!(f, "()"),
            MirConst::Opaque => write!(f, "<opaque>"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CastKind {
    /// Numeric / safe cast.
    Prim,
    /// `as!` unsafe cast.
    Unsafe,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Callee {
    /// Direct call to a resolved function.
    Fn(Symbol),
    /// Method call resolved through the trait env (kept opaque to MIR).
    Method { receiver: Operand, name: Symbol },
    /// Unresolved / foreign / comptime — emitted as an opaque call.
    Opaque(Symbol),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    Ref(Place, BorrowKind),
    UnaryOp(UnaryOp, Operand),
    BinaryOp(BinOp, Operand, Operand),
    Cast(CastKind, Operand, TypeId),
    /// Struct / tuple / array / enum-variant construction.
    Aggregate(AggregateKind, Vec<Operand>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateKind {
    Tuple,
    Array,
    Struct(Vec<Symbol>),
    Enum { variant: Symbol, field_count: usize },
}

// ---------------------------------------------------------------------------
// Statements & terminators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assign(Place, Rvalue, Span),
    StorageLive(LocalId, Span),
    StorageDead(LocalId, Span),
    /// Runtime destructor / `scope_cleanup` body invocation.
    Drop(Place, Span),
    /// Ghost (contract/proof) content — erased before codegen.
    Ghost(String, Span),
    /// Anything the lowering does not yet model; never in a finished pass.
    Unsupported(String, Span),
}

impl Statement {
    pub fn span(&self) -> Span {
        match self {
            Statement::Assign(_, _, s)
            | Statement::StorageLive(_, s)
            | Statement::StorageDead(_, s)
            | Statement::Drop(_, s)
            | Statement::Ghost(_, s)
            | Statement::Unsupported(_, s) => *s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SwitchTarget {
    /// Integer-scrutinee switch: `(value, target)`.
    Int(Vec<(i128, BlockId)>, BlockId),
    /// Enum-scrutinee switch: `(variant_name, target)`.
    Enum(Vec<(Symbol, BlockId)>, BlockId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Goto(BlockId),
    /// Conditional branch on a boolean operand.
    Branch {
        cond: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    Switch {
        discr: Operand,
        targets: SwitchTarget,
    },
    Call {
        callee: Callee,
        args: Vec<Operand>,
        destination: Option<(Place, BlockId)>,
    },
    Return,
    /// Non-local exit: `leave` (to a label) — resolved during lowering.
    Leave,
    Unreachable,
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct LocalDecl {
    pub name: Option<Symbol>,
    pub ty: TypeId,
    pub mutable: bool,
    /// Whether this local is a lowering temp (no source name).
    pub is_temp: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub stmts: Vec<Statement>,
    pub terminator: Terminator,
}

/// A fully lowered function body.
#[derive(Debug, Clone)]
pub struct MirBody {
    pub blocks: Vec<BasicBlock>,
    pub locals: Vec<LocalDecl>,
    pub arg_count: usize,
    pub entry: BlockId,
    /// Name of the function this body belongs to (for pretty printing).
    pub func_name: Symbol,
}

impl MirBody {
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0]
    }

    pub fn successors(&self, id: BlockId) -> Vec<BlockId> {
        match &self.block(id).terminator {
            Terminator::Goto(b) => vec![*b],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Switch { targets, .. } => match targets {
                SwitchTarget::Int(arms, fallback) => {
                    let mut out: Vec<BlockId> = arms.iter().map(|(_, b)| *b).collect();
                    out.push(*fallback);
                    out
                }
                SwitchTarget::Enum(arms, fallback) => {
                    let mut out: Vec<BlockId> = arms.iter().map(|(_, b)| *b).collect();
                    out.push(*fallback);
                    out
                }
            },
            Terminator::Call {
                destination: Some((_, target)),
                ..
            } => vec![*target],
            Terminator::Call {
                destination: None, ..
            } => vec![],
            Terminator::Return | Terminator::Leave | Terminator::Unreachable => vec![],
        }
    }
}

pub mod build;
pub mod pretty;
