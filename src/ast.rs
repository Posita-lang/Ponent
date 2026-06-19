use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Char(u8),
    String(String),
    ByteString(Vec<u8>),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    AddWrap,
    SubWrap,
    MulWrap,
    AddSaturate,
    SubSaturate,
    MulSaturate,
    AddTrap,
    SubTrap,
    MulTrap,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Deref,
    Ref,
    RefMut,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal, Span),
    Ident(String, Span),
    BinaryOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
        span: Span,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
        span: Span,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        comptime: bool,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    FieldAccess {
        base: Box<Expr>,
        field: String,
        span: Span,
    },
    AttrAccess {
        base: Box<Expr>,
        attr: String,
        span: Span,
    },
    Cast {
        expr: Box<Expr>,
        ty: Box<Type>,
        safe: bool,
        span: Span,
    },
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
        span: Span,
    },
    StructLit {
        path: Vec<String>,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
    EnumLit {
        path: Vec<String>,
        variant: String,
        payload: Option<Box<Expr>>,
        span: Span,
    },
    Tuple(Vec<Expr>, Span),
    Array(Vec<Expr>, Span),
    Closure {
        params: Vec<Param>,
        return_type: Option<Type>,
        captures: Vec<Capture>,
        body: Vec<Stmt>,
        span: Span,
    },
    Catch {
        expr: Box<Expr>,
        branches: Vec<CatchBranch>,
        span: Span,
    },
    LeaveWith {
        expr: Box<Expr>,
        span: Span,
    },
    Await {
        expr: Box<Expr>,
        span: Span,
    },
    If {
        cond: Box<Expr>,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        is_expression: bool,
        span: Span,
    },
    IfLet {
        pattern: Pattern,
        scrutinee: Box<Expr>,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Block(Vec<Stmt>, Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchBranch {
    pub pattern: Pattern,
    pub bind: Option<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    pub name: String,
    pub mode: CaptureMode,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    ByRef,
    ByMutRef,
    ByCopy,
    ByMove,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    VariableDef {
        kind: VariableKind,
        mutable: bool,
        name: Option<String>,
        pattern: Option<Pattern>,
        ty: Option<Type>,
        value: Option<Expr>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
        attributes: Vec<Attribute>,
        doc: Option<String>,
    },
    FunctionDef {
        span: Span,
        attributes: Vec<Attribute>,
        contracts: Vec<Contract>,
        doc: Option<String>,
        name: String,
        params: Vec<Param>,
        return_type: Type,
        body: Option<Vec<Stmt>>,
    },
    TypeDef {
        span: Span,
        attributes: Vec<Attribute>,
        doc: Option<String>,
        name: String,
        params: Vec<TypeParam>,
        definition: TypeDefinition,
        contracts: Vec<Contract>,
    },
    Import {
        path: Vec<String>,
        items: Option<Vec<String>>,
        alias: Option<String>,
        span: Span,
    },
    ExternFunction {
        abi: String,
        name: String,
        params: Vec<Param>,
        return_type: Type,
        span: Span,
        attributes: Vec<Attribute>,
    },
    Edition(String, Span),
    TestBlock {
        name: String,
        body: Vec<Stmt>,
        span: Span,
    },
    Expression(Expr),
    If {
        cond: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        pattern: Pattern,
        iterable: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Loop {
        body: Vec<Stmt>,
        span: Span,
    },
    Leave {
        label: Option<String>,
        span: Span,
    },
    Continue {
        label: Option<String>,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Assign {
        target: Box<Expr>,
        op: Option<BinOp>,
        value: Expr,
        span: Span,
    },
    ComptimeBlock {
        body: Vec<Stmt>,
        span: Span,
    },
    ScopeCleanup {
        name: String,
        body: Vec<Stmt>,
        span: Span,
    },
    Trigger {
        name: String,
        span: Span,
    },
    Error(Span),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    Let,
    Set,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub bounds: Vec<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDefinition {
    Struct(Vec<StructField>),
    Enum(Vec<EnumVariant>),
    TraitDef {
        methods: Vec<TraitMethod>,
        associated_types: Vec<AssociatedType>,
    },
    ImplBlock {
        trait_path: Option<Vec<String>>,
        for_type: Type,
        methods: Vec<ImplMethod>,
    },
    Constraint(Vec<Type>),
    Alias(Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub payload: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociatedType {
    pub name: String,
    pub default: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplMethod {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Option<Vec<Stmt>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Path(Vec<String>, Span),
    Generic(Box<Type>, Vec<Type>, Span),
    Reference(Box<Type>, bool, Span),
    Pointer(Box<Type>, Span),
    Slice(Box<Type>, Span),
    Array(Box<Type>, Box<Expr>, Span),
    Tuple(Vec<Type>, Span),
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
        span: Span,
    },
    Projection(Box<Type>, String, Span),
    DynTrait(Vec<Type>, Span),
    Exists {
        name: String,
        base: Box<Type>,
        invariant: Box<Expr>,
        span: Span,
    },
    Literal(Box<Expr>, Span),
    Never(Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard(Span),
    Ident(String, Span),
    Literal(Box<Expr>, Span),
    Tuple(Vec<Pattern>, Span),
    Struct {
        path: Vec<String>,
        fields: Vec<(String, Pattern)>,
        span: Span,
    },
    Enum {
        path: Vec<String>,
        variant: String,
        inner: Option<Box<Pattern>>,
        span: Span,
    },
    Or(Vec<Pattern>, Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Contract {
    Requires(Expr, Span),
    Ensures(Expr, Span),
    Invariant(Expr, Span),
    Decreases(Expr, Span),
    Terminates(Expr, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Stmt>,
    pub span: Span,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
