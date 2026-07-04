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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    Round,
    Trunc,
    Ceil,
    Floor,
}

/// Quantifier kind for `forall` / `exists` expressions in contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    Forall,
    Exists,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal, Span),
    Ident(String, Span),
    TypeAnnotated {
        expr: Box<Expr>,
        ty: Box<Type>,
        span: Span,
    },
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
        rounding: Option<Rounding>,
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
    Move(Box<Expr>, Span),
    /// Multi-segment path: `Module::Type::item`. Preserves `::` semantics,
    /// distinct from FieldAccess (`.`). Used for associated fn calls,
    /// enum variant construction, etc.
    Path(Vec<String>, Span),
    Tuple(Vec<Expr>, Span),
    Array(Vec<Expr>, Span),
    Closure {
        params: Vec<Param>,
        return_type: Option<Type>,
        captures: Vec<Capture>,
        body: Vec<Stmt>,
        span: Span,
    },
    Try {
        expr: Box<Expr>,
        span: Span,
    },
    UnsafeBlock {
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
    /// `poly(expr)` — implicit poly box, or `poly : Scheme(expr)` — explicit.
    PolyBox {
        expr: Box<Expr>,
        /// Optional scheme: `forall T1, T2, ... . body`
        scheme: Option<TypeScheme>,
        span: Span,
    },
    /// `unbox(expr)` — implicit poly unbox, or `unbox : Scheme(expr)` — explicit.
    PolyUnbox {
        expr: Box<Expr>,
        /// Optional expected result scheme type.
        scheme: Option<TypeScheme>,
        span: Span,
    },
    /// Quantified expression: `forall i in 0..n: body` or `exists i in range: body`.
    /// Used in contract position (`requires forall i in 0..arr'len: arr[i] > 0`).
    Quantified {
        quantifier: Quantifier,
        binder: String,
        range: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },
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
        type_params: Vec<TypeParam>,
        where_clause: Option<WhereClause>,
        finally: Option<Vec<Stmt>>,
        is_comptime: bool,
        is_async: bool,
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
    TraitDef {
        span: Span,
        attributes: Vec<Attribute>,
        doc: Option<String>,
        name: String,
        methods: Vec<TraitMethod>,
        associated_types: Vec<AssociatedType>,
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
    Constraint {
        name: String,
        bounds: Vec<Type>,
        span: Span,
    },
    Edition(String, Span),
    Expression(Expr),
    If {
        cond: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
    IfLet {
        pattern: Pattern,
        scrutinee: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        invariant: Option<Expr>,
        decreases: Option<Expr>,
        span: Span,
    },
    WhileLet {
        pattern: Pattern,
        scrutinee: Expr,
        body: Vec<Stmt>,
        invariant: Option<Expr>,
        decreases: Option<Expr>,
        span: Span,
    },
    For {
        pattern: Pattern,
        iterable: Expr,
        body: Vec<Stmt>,
        invariant: Option<Expr>,
        decreases: Option<Expr>,
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
        propagates: bool,
        overrides: bool,
        span: Span,
    },
    Trigger {
        name: String,
        span: Span,
    },
    Unsafe {
        body: Vec<Stmt>,
        span: Span,
    },
    GhostVariableDef {
        inner: Box<Stmt>,
        span: Span,
    },
    Isolate {
        body: Vec<Stmt>,
        span: Span,
    },
    ImplBlock {
        span: Span,
        attributes: Vec<Attribute>,
        trait_path: Option<Vec<String>>,
        for_type: Type,
        methods: Vec<ImplMethod>,
        associated_types: Vec<AssociatedType>,
        where_clause: Option<WhereClause>,
        type_params: Vec<TypeParam>,
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
    pub ty: Option<Type>,
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
    Struct(Vec<StructField>, Vec<TypeModifier>),
    Enum(Vec<EnumVariant>, Option<String>, Vec<TypeModifier>),
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
    Alias(Type, Vec<TypeModifier>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeModifier {
    Overflow(OverflowPolicy),
    Default(Expr),
    Validate(Expr),
    NoDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    Wrap,
    Saturate,
    Trap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WherePredicate {
    pub ty: Type,
    pub bounds: Vec<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
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

/// A single generic argument, either positional (`T`) or named (`size = T`).
#[derive(Debug, Clone, PartialEq)]
pub enum GenericArg {
    Positional(Type),
    Named(String, Type),
}

impl GenericArg {
    pub fn ty(&self) -> &Type {
        match self {
            GenericArg::Positional(ty) | GenericArg::Named(_, ty) => ty,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Path(Vec<String>, Span),
    Generic(Box<Type>, Vec<GenericArg>, Span),
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
    /// Qualified path projection: `<ImplType as TraitPath>::AssocName`
    Projection {
        impl_type: Box<Type>,
        trait_path: Box<Type>,
        assoc_name: String,
        span: Span,
    },
    DynTrait(Vec<Type>, Span),
    Exists {
        name: String,
        base: Box<Type>,
        invariant: Box<Expr>,
        span: Span,
    },
    /// Shorthand `type T = Base where value > 0` — the parser produces this instead of
    /// doing semantic name generation. A later desugaring pass rewrites it to `Exists`.
    WhereShorthand {
        base: Box<Type>,
        invariant: Box<Expr>,
        span: Span,
    },
    Literal(Box<Expr>, Span),
    Never(Span),
    Union(Vec<Type>, Span),
    /// A constant expression where a type is expected, e.g. array sizes
    /// `[Int<32>; N + 1]` or generic const args `<Array<Int, N>>`.
    Expr(Box<Expr>, Span),
    Error(Span),
}

/// A polymorphic type scheme: `forall T1, T2, ... . body`
#[derive(Debug, Clone, PartialEq)]
pub struct TypeScheme {
    pub quantifiers: Vec<(Span, String)>,
    pub body: Box<Type>,
    pub span: Span,
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
    Slice(Vec<Pattern>, Option<Box<Pattern>>, Vec<Pattern>, Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<Expr>,
    pub named_args: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnsuresTarget {
    Unconditional,
    OnOk(Option<Pattern>),
    OnErr(Option<Pattern>),
    OnTimeout,
    OnCancel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Contract {
    Requires(Expr, Span),
    Ensures {
        expr: Expr,
        span: Span,
        target: EnsuresTarget,
    },
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

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Path(_, span)
            | Type::Reference(_, _, span)
            | Type::Pointer(_, span)
            | Type::Slice(_, span)
            | Type::Array(_, _, span)
            | Type::Tuple(_, span)
            | Type::Function { span, .. }
            | Type::Projection { span, .. }
            | Type::DynTrait(_, span)
            | Type::Exists { span, .. }
            | Type::WhereShorthand { span, .. }
            | Type::Literal(_, span)
            | Type::Never(span)
            | Type::Union(_, span)
            | Type::Expr(_, span)
            | Type::Error(span) => *span,
            Type::Generic(_, _, span) => *span,
        }
    }
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::VariableDef { span, .. } => *span,
            Stmt::FunctionDef { span, .. } => *span,
            Stmt::TypeDef { span, .. } => *span,
            Stmt::TraitDef { span, .. } => *span,
            Stmt::Import { span, .. } => *span,
            Stmt::ExternFunction { span, .. } => *span,
            Stmt::Constraint { span, .. } => *span,
            Stmt::Edition(_, span) => *span,
            Stmt::Expression(expr) => expr.span(),
            Stmt::If { span, .. } => *span,
            Stmt::IfLet { span, .. } => *span,
            Stmt::While { span, .. } => *span,
            Stmt::WhileLet { span, .. } => *span,
            Stmt::For { span, .. } => *span,
            Stmt::Loop { span, .. } => *span,
            Stmt::Leave { span, .. } => *span,
            Stmt::Continue { span, .. } => *span,
            Stmt::Return { span, .. } => *span,
            Stmt::Assign { span, .. } => *span,
            Stmt::ComptimeBlock { span, .. } => *span,
            Stmt::ScopeCleanup { span, .. } => *span,
            Stmt::Trigger { span, .. } => *span,
            Stmt::Unsafe { span, .. } => *span,
            Stmt::GhostVariableDef { span, .. } => *span,
            Stmt::Isolate { span, .. } => *span,
            Stmt::ImplBlock { span, .. } => *span,
            Stmt::Error(span) => *span,
        }
    }
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal(_, span) => *span,
            Expr::Ident(_, span) => *span,
            Expr::TypeAnnotated { span, .. } => *span,
            Expr::BinaryOp { span, .. } => *span,
            Expr::UnaryOp { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::Index { span, .. } => *span,
            Expr::FieldAccess { span, .. } => *span,
            Expr::AttrAccess { span, .. } => *span,
            Expr::Cast { span, .. } => *span,
            Expr::Range { span, .. } => *span,
            Expr::StructLit { span, .. } => *span,
            Expr::EnumLit { span, .. } => *span,
            Expr::Path(_, span) => *span,
            Expr::Move(_, span) => *span,
            Expr::Tuple(_, span) => *span,
            Expr::Array(_, span) => *span,
            Expr::Closure { span, .. } => *span,
            Expr::Try { span, .. } => *span,
            Expr::UnsafeBlock { span, .. } => *span,
            Expr::Catch { span, .. } => *span,
            Expr::LeaveWith { span, .. } => *span,
            Expr::Await { span, .. } => *span,
            Expr::If { span, .. } => *span,
            Expr::IfLet { span, .. } => *span,
            Expr::Match { span, .. } => *span,
            Expr::Block(_, span) => *span,
            Expr::PolyBox { span, .. } => *span,
            Expr::PolyUnbox { span, .. } => *span,
            Expr::Quantified { span, .. } => *span,
            Expr::Error(span) => *span,
        }
    }
}
