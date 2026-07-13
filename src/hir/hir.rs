use crate::ast::*;
use crate::hir::types::{DefId, TypeId};
use crate::symbol::Symbol;

#[derive(Debug, Clone, PartialEq)]
pub struct HirProgram {
    pub items: Vec<HirStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirStmt {
    VariableDef {
        kind: VariableKind,
        mutable: bool,
        name: Option<Symbol>,
        pattern: Option<HirPattern>,
        ty: TypeId,
        value: Option<Box<HirExpr>>,
        else_branch: Option<Vec<HirStmt>>,
        span: Span,
        type_captures: Vec<TypeParam>,
    },
    FunctionDef {
        span: Span,
        attributes: Vec<Attribute>,
        contracts: Vec<Contract>,
        doc: Option<String>,
        name: Symbol,
        params: Vec<HirParam>,
        return_type: TypeId,
        body: Option<Vec<HirStmt>>,
        type_params: Vec<TypeParam>,
        where_clause: Option<()>,
        finally: Option<Vec<HirStmt>>,
        is_comptime: bool,
        is_async: bool,
        is_ieee_contracts: bool,
        hints: Vec<Expr>,
    },
    TypeDef {
        span: Span,
        attributes: Vec<Attribute>,
        doc: Option<String>,
        name: Symbol,
        params: Vec<TypeParam>,
        definition: TypeDefinition,
        contracts: Vec<Contract>,
    },
    TraitDef {
        span: Span,
        attributes: Vec<Attribute>,
        doc: Option<String>,
        name: Symbol,
        methods: Vec<TraitMethod>,
        associated_types: Vec<AssociatedType>,
    },
    Import {
        path: Vec<Symbol>,
        items: Option<Vec<Symbol>>,
        alias: Option<Symbol>,
        span: Span,
    },
    ExternFunction {
        abi: String,
        name: Symbol,
        params: Vec<HirParam>,
        return_type: TypeId,
        span: Span,
        attributes: Vec<Attribute>,
    },
    Constraint {
        name: Symbol,
        bounds: Vec<TypeId>,
        span: Span,
    },
    Edition(String, Span),
    Expression(Box<HirExpr>),
    If {
        cond: Box<HirExpr>,
        then_branch: Vec<HirStmt>,
        else_branch: Option<Vec<HirStmt>>,
        span: Span,
    },
    IfLet {
        pattern: HirPattern,
        scrutinee: Box<HirExpr>,
        then_branch: Vec<HirStmt>,
        else_branch: Option<Vec<HirStmt>>,
        span: Span,
    },
    While {
        cond: Box<HirExpr>,
        body: Vec<HirStmt>,
        invariant: Option<Box<HirExpr>>,
        decreases: Option<Box<HirExpr>>,
        span: Span,
    },
    WhileLet {
        pattern: HirPattern,
        scrutinee: Box<HirExpr>,
        body: Vec<HirStmt>,
        invariant: Option<Box<HirExpr>>,
        decreases: Option<Box<HirExpr>>,
        span: Span,
    },
    For {
        pattern: HirPattern,
        iterable: Box<HirExpr>,
        body: Vec<HirStmt>,
        invariant: Option<Box<HirExpr>>,
        decreases: Option<Box<HirExpr>>,
        span: Span,
    },
    Loop {
        body: Vec<HirStmt>,
        span: Span,
    },
    Leave {
        label: Option<Symbol>,
        span: Span,
    },
    Continue {
        label: Option<Symbol>,
        span: Span,
    },
    Return {
        value: Option<Box<HirExpr>>,
        span: Span,
    },
    Assign {
        target: Box<HirExpr>,
        op: Option<BinOp>,
        value: Box<HirExpr>,
        span: Span,
    },
    ComptimeBlock {
        body: Vec<HirStmt>,
        ty: TypeId,
        span: Span,
    },
    ScopeCleanup {
        name: Symbol,
        body: Vec<HirStmt>,
        propagates: bool,
        overrides: bool,
        span: Span,
    },
    Trigger {
        name: Symbol,
        span: Span,
    },
    Unsafe {
        body: Vec<HirStmt>,
        span: Span,
    },
    GhostVariableDef {
        inner: Box<HirStmt>,
        span: Span,
    },
    Isolate {
        body: Vec<HirStmt>,
        span: Span,
    },
    LayoutDef {
        name: Symbol,
        attributes: Vec<Attribute>,
        span: Span,
    },
    ImplBlock {
        span: Span,
        attributes: Vec<Attribute>,
        trait_path: Option<DefId>,
        for_type: TypeId,
        methods: Vec<ImplMethod>,
        associated_types: Vec<AssociatedType>,
    },
    Generate {
        for_type: TypeId,
        body: Vec<HirStmt>,
        span: Span,
    },
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirExpr {
    Literal(Literal, TypeId, Span),
    Ident(Symbol, TypeId, Span),
    TypeAnnotated {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    BinaryOp {
        left: Box<HirExpr>,
        op: BinOp,
        right: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    Call {
        callee: Box<HirExpr>,
        args: Vec<HirExpr>,
        comptime: bool,
        ty: TypeId,
        span: Span,
    },
    Index {
        base: Box<HirExpr>,
        index: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    FieldAccess {
        base: Box<HirExpr>,
        field: Symbol,
        ty: TypeId,
        span: Span,
    },
    AttrAccess {
        base: Box<HirExpr>,
        attr: Symbol,
        ty: TypeId,
        span: Span,
    },
    Cast {
        expr: Box<HirExpr>,
        ty: TypeId,
        safe: bool,
        rounding: Option<Rounding>,
        span: Span,
    },
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
        inclusive: bool,
        ty: TypeId,
        span: Span,
    },
    StructLit {
        path: Vec<Symbol>,
        fields: Vec<(Symbol, Box<HirExpr>)>,
        ty: TypeId,
        span: Span,
    },
    EnumLit {
        path: Vec<Symbol>,
        variant: Symbol,
        payload: Option<Box<HirExpr>>,
        ty: TypeId,
        span: Span,
    },
    Move(Box<HirExpr>, TypeId, Span),
    Tuple(Vec<HirExpr>, TypeId, Span),
    Array(Vec<HirExpr>, TypeId, Span),
    Closure {
        params: Vec<HirParam>,
        return_type: TypeId,
        captures: Vec<Capture>,
        body: Vec<HirStmt>,
        ty: TypeId,
        span: Span,
    },
    Try {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    UnsafeBlock {
        body: Vec<HirStmt>,
        ty: TypeId,
        span: Span,
    },
    Catch {
        expr: Box<HirExpr>,
        branches: Vec<HirCatchBranch>,
        ty: TypeId,
        span: Span,
    },
    LeaveWith {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    Await {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    If {
        cond: Box<HirExpr>,
        then_branch: Vec<HirStmt>,
        else_branch: Option<Vec<HirStmt>>,
        is_expression: bool,
        ty: TypeId,
        span: Span,
    },
    IfLet {
        pattern: HirPattern,
        scrutinee: Box<HirExpr>,
        then_branch: Vec<HirStmt>,
        else_branch: Option<Vec<HirStmt>>,
        ty: TypeId,
        span: Span,
    },
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
        ty: TypeId,
        span: Span,
    },
    Block(Vec<HirStmt>, TypeId, Span),
    PolyBox {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    PolyUnbox {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    Quantified {
        quantifier: crate::ast::Quantifier,
        binder: Symbol,
        range: Box<HirExpr>,
        body: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    Old {
        expr: Box<HirExpr>,
        ty: TypeId,
        span: Span,
    },
    Task {
        block: Vec<HirStmt>,
        ty: TypeId,
        span: Span,
    },
    TypeInfo(TypeId, Span),
    /// Compile-time error: `@compile_error!("msg")`.
    /// Produced when the comptime evaluator encounters this expression.
    CompileError(String, Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirCatchBranch {
    pub pattern: HirPattern,
    pub bind: Option<Symbol>,
    pub body: Vec<HirStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<Box<HirExpr>>,
    pub body: Box<HirExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirParam {
    pub name: Symbol,
    pub ty: TypeId,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirPattern {
    Wildcard(Span),
    Ident(Symbol, TypeId, Span),
    Literal(Box<HirExpr>, Span),
    Tuple(Vec<HirPattern>, Span),
    Struct {
        path: Vec<Symbol>,
        fields: Vec<(Symbol, Box<HirPattern>)>,
        rest: bool,
        span: Span,
    },
    Enum {
        path: Vec<Symbol>,
        variant: Symbol,
        inner: Option<Box<HirPattern>>,
        span: Span,
    },
    Or(Vec<HirPattern>, Span),
    Slice(
        Vec<HirPattern>,
        Option<Box<HirPattern>>,
        Vec<HirPattern>,
        Span,
    ),
    Error(Span),
}

impl HirExpr {
    pub fn ty(&self) -> TypeId {
        match self {
            HirExpr::Literal(_, ty, _) => *ty,
            HirExpr::Ident(_, ty, _) => *ty,
            HirExpr::TypeAnnotated { ty, .. } => *ty,
            HirExpr::BinaryOp { ty, .. } => *ty,
            HirExpr::UnaryOp { ty, .. } => *ty,
            HirExpr::Call { ty, .. } => *ty,
            HirExpr::Index { ty, .. } => *ty,
            HirExpr::FieldAccess { ty, .. } => *ty,
            HirExpr::AttrAccess { ty, .. } => *ty,
            HirExpr::Cast { ty, .. } => *ty,
            HirExpr::Range { ty, .. } => *ty,
            HirExpr::StructLit { ty, .. } => *ty,
            HirExpr::EnumLit { ty, .. } => *ty,
            HirExpr::Move(_, ty, _) => *ty,
            HirExpr::Tuple(_, ty, _) => *ty,
            HirExpr::Array(_, ty, _) => *ty,
            HirExpr::Closure { ty, .. } => *ty,
            HirExpr::Try { ty, .. } => *ty,
            HirExpr::UnsafeBlock { ty, .. } => *ty,
            HirExpr::Catch { ty, .. } => *ty,
            HirExpr::LeaveWith { ty, .. } => *ty,
            HirExpr::Await { ty, .. } => *ty,
            HirExpr::If { ty, .. } => *ty,
            HirExpr::IfLet { ty, .. } => *ty,
            HirExpr::Match { ty, .. } => *ty,
            HirExpr::Block(_, ty, _) => *ty,
            HirExpr::PolyBox { ty, .. } => *ty,
            HirExpr::PolyUnbox { ty, .. } => *ty,
            HirExpr::Quantified { ty, .. } => *ty,
            HirExpr::Old { ty, .. } => *ty,
            HirExpr::Task { ty, .. } => *ty,
            HirExpr::TypeInfo(_, _) => TypeId(0),
            HirExpr::CompileError(_, _) => TypeId(0),
            HirExpr::Error(_) => TypeId(0),
        }
    }

    pub fn span(&self) -> Span {
        match self {
            HirExpr::Literal(_, _, span) => *span,
            HirExpr::Ident(_, _, span) => *span,
            HirExpr::TypeAnnotated { span, .. } => *span,
            HirExpr::BinaryOp { span, .. } => *span,
            HirExpr::UnaryOp { span, .. } => *span,
            HirExpr::Call { span, .. } => *span,
            HirExpr::Index { span, .. } => *span,
            HirExpr::FieldAccess { span, .. } => *span,
            HirExpr::AttrAccess { span, .. } => *span,
            HirExpr::Cast { span, .. } => *span,
            HirExpr::Range { span, .. } => *span,
            HirExpr::StructLit { span, .. } => *span,
            HirExpr::EnumLit { span, .. } => *span,
            HirExpr::Move(_, _, span) => *span,
            HirExpr::Tuple(_, _, span) => *span,
            HirExpr::Array(_, _, span) => *span,
            HirExpr::Closure { span, .. } => *span,
            HirExpr::Try { span, .. } => *span,
            HirExpr::UnsafeBlock { span, .. } => *span,
            HirExpr::Catch { span, .. } => *span,
            HirExpr::LeaveWith { span, .. } => *span,
            HirExpr::Await { span, .. } => *span,
            HirExpr::If { span, .. } => *span,
            HirExpr::IfLet { span, .. } => *span,
            HirExpr::Match { span, .. } => *span,
            HirExpr::Block(_, _, span) => *span,
            HirExpr::PolyBox { span, .. } => *span,
            HirExpr::PolyUnbox { span, .. } => *span,
            HirExpr::Quantified { span, .. } => *span,
            HirExpr::Old { span, .. } => *span,
            HirExpr::Task { span, .. } => *span,
            HirExpr::TypeInfo(_, span) => *span,
            HirExpr::CompileError(_, span) => *span,
            HirExpr::Error(span) => *span,
        }
    }
}

impl HirStmt {
    pub fn span(&self) -> Span {
        match self {
            HirStmt::VariableDef { span, .. } => *span,
            HirStmt::FunctionDef { span, .. } => *span,
            HirStmt::TypeDef { span, .. } => *span,
            HirStmt::TraitDef { span, .. } => *span,
            HirStmt::Import { span, .. } => *span,
            HirStmt::ExternFunction { span, .. } => *span,
            HirStmt::Constraint { span, .. } => *span,
            HirStmt::Edition(_, span) => *span,
            HirStmt::Expression(expr) => expr.span(),
            HirStmt::If { span, .. } => *span,
            HirStmt::IfLet { span, .. } => *span,
            HirStmt::While { span, .. } => *span,
            HirStmt::WhileLet { span, .. } => *span,
            HirStmt::For { span, .. } => *span,
            HirStmt::Loop { span, .. } => *span,
            HirStmt::Leave { span, .. } => *span,
            HirStmt::Continue { span, .. } => *span,
            HirStmt::Return { span, .. } => *span,
            HirStmt::Assign { span, .. } => *span,
            HirStmt::ComptimeBlock { span, .. } => *span,
            HirStmt::ScopeCleanup { span, .. } => *span,
            HirStmt::Trigger { span, .. } => *span,
            HirStmt::Unsafe { span, .. } => *span,
            HirStmt::GhostVariableDef { span, .. } => *span,
            HirStmt::Isolate { span, .. } => *span,
            HirStmt::ImplBlock { span, .. } => *span,
            HirStmt::LayoutDef { span, .. } => *span,
            HirStmt::Generate { span, .. } => *span,
            HirStmt::Error => Span::new(0, 0),
        }
    }
}
