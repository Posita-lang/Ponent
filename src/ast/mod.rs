pub mod visit;
use num_bigint::{BigInt, Sign};

use crate::symbol::Symbol;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
}

/// Sentinel span for "no source location known" (start == end == 0).
/// Used as a fallback in internal error paths that lack a real span;
/// diagnostic emitters treat it as an unanchored location.
pub const DUMMY_SPAN: Span = Span::new(0, 0);

#[derive(Debug, Clone, PartialEq)]
pub enum IntLit {
    Small(i128),
    Large(Box<BigInt>),
}

impl IntLit {
    pub fn normalize(v: BigInt) -> Self {
        use num_traits::ToPrimitive;
        match v.to_i128() {
            Some(n) => IntLit::Small(n),
            None => IntLit::Large(Box::new(v)),
        }
    }

    pub fn fits_unsigned(&self, bits: u64) -> bool {
        match self {
            IntLit::Small(n) => *n >= 0 && (bits >= 127 || (*n as u128) < (1u128 << bits)),
            IntLit::Large(b) => b.sign() != Sign::Minus && b.bits() <= bits,
        }
    }

    pub fn fits_signed(&self, bits: u64) -> bool {
        match self {
            IntLit::Small(n) => match bits {
                0 => false,
                b if b >= 128 => true,
                b => *n >= -(1i128 << (b - 1)) && *n < (1i128 << (b - 1)),
            },
            IntLit::Large(b) => {
                let half = BigInt::from(1) << ((bits - 1) as usize); // 2^(N-1)
                *b >= Box::new(-&half) && *b < Box::new(half)
            }
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self {
            IntLit::Small(n) => u64::try_from(*n).ok(),
            IntLit::Large(b) => {
                use num_traits::ToPrimitive;
                b.to_u64()
            }
        }
    }

    pub fn to_i128(&self) -> Option<i128> {
        match self {
            IntLit::Small(n) => Some(*n),
            IntLit::Large(b) => {
                use num_traits::ToPrimitive;
                b.to_i128()
            }
        }
    }

    /// `self + other` with i128 overflow detection — `None` on overflow
    /// or when `self` is Large (beyond i128; callers fail closed).
    pub fn checked_add(&self, other: i128) -> Option<i128> {
        match self {
            IntLit::Small(n) => n.checked_add(other),
            IntLit::Large(_) => None,
        }
    }

    /// `self - other` with i128 overflow detection — `None` on overflow
    /// or when `self` is Large (callers fail closed).
    pub fn checked_sub(&self, other: i128) -> Option<i128> {
        match self {
            IntLit::Small(n) => n.checked_sub(other),
            IntLit::Large(_) => None,
        }
    }

    /// Absolute value as `u128` (i128's `unsigned_abs`), saturating to
    /// `u128::MAX` for Large values (any threshold check then passes).
    pub fn unsigned_abs(&self) -> u128 {
        match self {
            IntLit::Small(n) => n.unsigned_abs(),
            IntLit::Large(b) => {
                use num_traits::ToPrimitive;
                (**b).to_u128().unwrap_or(u128::MAX)
            }
        }
    }
}

/// Unary negation of a borrowed literal — used in i128 constant contexts
/// (`-c` for descending counters); Large values fail closed to
/// `i128::MIN` (they cannot fit i128).
impl std::ops::Neg for &IntLit {
    type Output = i128;
    fn neg(self) -> i128 {
        match self {
            IntLit::Small(n) => -n,
            IntLit::Large(_) => i128::MIN,
        }
    }
}

impl std::fmt::Display for IntLit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntLit::Small(n) => write!(f, "{n}"),
            IntLit::Large(b) => write!(f, "{b}"),
        }
    }
}

impl From<IntLit> for BigInt {
    fn from(v: IntLit) -> Self {
        match v {
            IntLit::Small(n) => BigInt::from(n),
            IntLit::Large(b) => *b,
        }
    }
}

impl PartialOrd<i32> for IntLit {
    fn partial_cmp(&self, other: &i32) -> Option<Ordering> {
        Some(match self {
            IntLit::Small(n) => n.cmp(&i128::from(*other)),
            IntLit::Large(b) => (**b).cmp(&BigInt::from(*other)),
        })
    }
}

impl PartialEq<i32> for IntLit {
    fn eq(&self, other: &i32) -> bool {
        match self {
            IntLit::Small(n) => *n == i128::from(*other),
            IntLit::Large(b) => **b == BigInt::from(*other),
        }
    }
}

impl PartialEq<i64> for IntLit {
    fn eq(&self, other: &i64) -> bool {
        match self {
            IntLit::Small(n) => *n == i128::from(*other),
            IntLit::Large(b) => **b == BigInt::from(*other),
        }
    }
}

impl PartialOrd<i128> for IntLit {
    fn partial_cmp(&self, other: &i128) -> Option<Ordering> {
        Some(match self {
            IntLit::Small(n) => n.cmp(other),
            IntLit::Large(b) => (**b).cmp(&BigInt::from(*other)),
        })
    }
}

impl PartialEq<i128> for IntLit {
    fn eq(&self, other: &i128) -> bool {
        match self {
            IntLit::Small(n) => *n == *other,
            IntLit::Large(b) => **b == BigInt::from(*other),
        }
    }
}

impl PartialOrd<i64> for IntLit {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        Some(match self {
            IntLit::Small(n) => n.cmp(&i128::from(*other)),
            IntLit::Large(b) => (**b).cmp(&BigInt::from(*other)),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(IntLit),
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
    /// `*?` — saturating multiplication. STILL UNDER RESEARCH: the BII
    /// template domain is linear (multiplication is not in the subset);
    /// lowering fails closed on it. `/?` (saturating division) is not
    /// defined yet — both await a separate RFC.
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
    /// Read-only borrow: `&ro r` freezes a `&mut T` into a `&T`
    /// (SYNTAX.md §"Reference Coercion and Read-Only Borrows").
    Ro,
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
pub enum Expr<'input> {
    Literal(Literal, Span),
    Ident(Symbol, Span),
    TypeAnnotated {
        expr: &'input Expr<'input>,
        ty: &'input Type<'input>,
        span: Span,
    },
    BinaryOp {
        left: &'input Expr<'input>,
        op: BinOp,
        right: &'input Expr<'input>,
        span: Span,
    },
    UnaryOp {
        op: UnaryOp,
        expr: &'input Expr<'input>,
        span: Span,
    },
    Call {
        callee: &'input Expr<'input>,
        args: Vec<Expr<'input>>,
        comptime: bool,
        span: Span,
    },
    Index {
        base: &'input Expr<'input>,
        index: &'input Expr<'input>,
        span: Span,
    },
    FieldAccess {
        base: &'input Expr<'input>,
        field: Symbol,
        span: Span,
    },
    AttrAccess {
        base: &'input Expr<'input>,
        attr: Symbol,
        span: Span,
    },
    Cast {
        expr: &'input Expr<'input>,
        ty: &'input Type<'input>,
        safe: bool,
        rounding: Option<Rounding>,
        span: Span,
    },
    Range {
        start: Option<&'input Expr<'input>>,
        end: Option<&'input Expr<'input>>,
        inclusive: bool,
        span: Span,
    },
    StructLit {
        path: Vec<Symbol>,
        fields: Vec<(Symbol, Expr<'input>)>,
        span: Span,
    },
    EnumLit {
        path: Vec<Symbol>,
        variant: Symbol,
        payload: Option<&'input Expr<'input>>,
        span: Span,
    },
    Move(&'input Expr<'input>, Span),
    /// Multi-segment path: `Module::Type::item`. Preserves `::` semantics,
    /// distinct from FieldAccess (`.`). Used for associated fn calls,
    /// enum variant construction, etc.
    Path(smallvec::SmallVec<[Symbol; 4]>, Span),
    Tuple(Vec<Expr<'input>>, Span),
    Array(Vec<Expr<'input>>, Span),
    Closure {
        params: Vec<Param<'input>>,
        return_type: Option<Type<'input>>,
        captures: Vec<Capture>,
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    Try {
        expr: &'input Expr<'input>,
        span: Span,
    },
    UnsafeBlock {
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    Catch {
        expr: &'input Expr<'input>,
        branches: Vec<CatchBranch<'input>>,
        span: Span,
    },
    LeaveWith {
        expr: &'input Expr<'input>,
        /// `true` if from `return expr` (value return),
        /// `false` if from `leave with expr` (error propagation).
        is_return: bool,
        span: Span,
    },
    Await {
        expr: &'input Expr<'input>,
        span: Span,
    },
    If {
        cond: &'input Expr<'input>,
        then_branch: Vec<Stmt<'input>>,
        else_branch: Option<Vec<Stmt<'input>>>,
        is_expression: bool,
        span: Span,
    },
    IfLet {
        pattern: Pattern<'input>,
        scrutinee: &'input Expr<'input>,
        then_branch: Vec<Stmt<'input>>,
        else_branch: Option<Vec<Stmt<'input>>>,
        is_expression: bool,
        span: Span,
    },
    Match {
        scrutinee: &'input Expr<'input>,
        arms: Vec<MatchArm<'input>>,
        span: Span,
    },
    Block(Vec<Stmt<'input>>, Span),
    /// `poly(expr)` — implicit poly box, or `poly : Scheme(expr)` — explicit.
    PolyBox {
        expr: &'input Expr<'input>,
        /// Optional scheme: `forall T1, T2, ... . body`
        scheme: Option<TypeScheme<'input>>,
        span: Span,
    },
    /// `unbox(expr)` — implicit poly unbox, or `unbox : Scheme(expr)` — explicit.
    PolyUnbox {
        expr: &'input Expr<'input>,
        /// Optional expected result scheme type.
        scheme: Option<TypeScheme<'input>>,
        span: Span,
    },
    /// Quantified expression: `forall i in 0..n: body` or `exists i in range: body`.
    /// Used in contract position (`requires forall i in 0..arr'len: arr[i] > 0`).
    Quantified {
        quantifier: Quantifier,
        binder: Symbol,
        range: &'input Expr<'input>,
        body: &'input Expr<'input>,
        span: Span,
    },
    /// `old(expr)` — captures the value of `expr` at function entry.
    /// Used in `ensures` clauses: `ensures *x == old(*x) + 1`.
    Old(&'input Expr<'input>, Span),
    /// Spawn a task: `task { body }`
    Task {
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    /// Compile-time type reflection: `@typeInfo!(Type)` — returns a
    /// `TypeInfo` value describing the type's structure at comptime.
    /// Inspired by Zig's `@typeInfo`.
    TypeInfo(&'input Type<'input>, Span),
    /// Compile-time layout reflection: `layout_of!(Type)` — returns a
    /// `LayoutDescriptor` describing the type's size, alignment, and
    /// field offsets at comptime.  `layout_of!` is comptime-only and
    /// thus requires `!`.
    LayoutOf(&'input Type<'input>, Span),
    /// Compile-time error: `@compile_error!("msg")` unconditionally halts
    /// compilation with the given message when evaluated (comptime-only).
    /// Parsed as an expression so it can be guarded by `if` / `match`.
    CompileError(String, Span),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchBranch<'input> {
    pub pattern: Pattern<'input>,
    pub bind: Option<Symbol>,
    pub body: Vec<Stmt<'input>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm<'input> {
    pub pattern: Pattern<'input>,
    pub guard: Option<&'input Expr<'input>>,
    pub body: Expr<'input>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    pub name: Symbol,
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
// `#[non_exhaustive]` forces DOWNSTREAM matches to add a `_` arm for
// future variants (within this crate, matches stay exhaustive).  The
// `@auto_ro`/`@auto_coerce` placement validation is kept fail-closed
// via its own wildcard arm (see `validate_auto_ro_placement`).
#[non_exhaustive]
pub enum Stmt<'input> {
    VariableDef {
        kind: VariableKind,
        mutable: bool,
        name: Option<Symbol>,
        pattern: Option<Pattern<'input>>,
        ty: Option<Type<'input>>,
        value: Option<Expr<'input>>,
        else_branch: Option<Vec<Stmt<'input>>>,
        span: Span,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        /// Type/const captures: `set auto<T, N, L> = expr`.
        /// Bound at comptime after inferring `expr`'s type.
        type_captures: Vec<TypeParam<'input>>,
        /// Type-level modifiers (`with overflow = saturate` etc.) on the
        /// annotated type — applied by the checker when the type is
        /// resolved.
        type_modifiers: Vec<TypeModifier<'input>>,
    },
    FunctionDef {
        span: Span,
        attributes: Vec<Attribute<'input>>,
        contracts: Vec<Contract<'input>>,
        doc: Option<String>,
        name: Symbol,
        params: Vec<Param<'input>>,
        return_type: Option<Type<'input>>,
        body: Option<Vec<Stmt<'input>>>,
        type_params: Vec<TypeParam<'input>>,
        where_clause: Option<WhereClause<'input>>,
        finally: Option<Vec<Stmt<'input>>>,
        is_comptime: bool,
        is_async: bool,
    },
    TypeDef {
        span: Span,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        definition: TypeDefinition<'input>,
        contracts: Vec<Contract<'input>>,
    },
    TraitDef {
        span: Span,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        name: Symbol,
        methods: Vec<TraitMethod<'input>>,
        associated_types: Vec<AssociatedType<'input>>,
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
        params: Vec<Param<'input>>,
        return_type: Type<'input>,
        span: Span,
        attributes: Vec<Attribute<'input>>,
    },
    Constraint {
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        predicates: smallvec::SmallVec<[WherePredicate<'input>; 2]>,
        span: Span,
    },
    Edition(String, Span),
    Expression(Expr<'input>),
    If {
        cond: Expr<'input>,
        then_branch: Vec<Stmt<'input>>,
        else_branch: Option<Vec<Stmt<'input>>>,
        span: Span,
    },
    IfLet {
        pattern: Pattern<'input>,
        scrutinee: Expr<'input>,
        then_branch: Vec<Stmt<'input>>,
        else_branch: Option<Vec<Stmt<'input>>>,
        span: Span,
    },
    While {
        label: Option<Symbol>,
        cond: Expr<'input>,
        body: Vec<Stmt<'input>>,
        invariant: Option<Expr<'input>>,
        decreases: Option<Expr<'input>>,
        span: Span,
    },
    WhileLet {
        label: Option<Symbol>,
        pattern: Pattern<'input>,
        scrutinee: Expr<'input>,
        body: Vec<Stmt<'input>>,
        invariant: Option<Expr<'input>>,
        decreases: Option<Expr<'input>>,
        span: Span,
    },
    For {
        label: Option<Symbol>,
        pattern: Pattern<'input>,
        iterable: Expr<'input>,
        body: Vec<Stmt<'input>>,
        invariant: Option<Expr<'input>>,
        decreases: Option<Expr<'input>>,
        span: Span,
    },
    Loop {
        label: Option<Symbol>,
        body: Vec<Stmt<'input>>,
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
        value: Option<Expr<'input>>,
        /// Path labels attached to this return: `return @label1 @label2 expr`.
        /// Used to match specific return paths to `ensures @label` clauses.
        labels: Vec<Symbol>,
        span: Span,
    },
    Assign {
        target: &'input Expr<'input>,
        op: Option<BinOp>,
        value: Expr<'input>,
        span: Span,
    },
    ComptimeBlock {
        /// Variables captured from the enclosing runtime scope.
        /// `comptime [x, y] { ... }` makes `x` and `y` available as
        /// compile‑time constants inside the block.
        /// Each entry carries the variable name and its source span in
        /// the capture list so that errors can point to the specific name.
        captures: Vec<(Symbol, Span)>,
        /// Whether this block is annotated `@trusted`, granting access to
        /// `@trusted` functions and `unsafe` operations during comptime.
        trusted: bool,
        attributes: Vec<Attribute<'input>>,
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    /// A `generate` block: declarative, auditable code generation
    /// attached to a type.  The block is expanded at compile time
    /// to produce module-level declarations (impl, def, type, const).
    /// See SYNTAX.md §1029.
    Generate {
        attributes: Vec<Attribute<'input>>,
        for_type: &'input Type<'input>,
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    ScopeCleanup {
        name: Symbol,
        /// Optional compile-time guard: `scope_cleanup @name when condition { }`
        when_condition: Option<&'input Expr<'input>>,
        body: Vec<Stmt<'input>>,
        propagates: bool,
        overrides: bool,
        span: Span,
    },
    Trigger {
        name: Symbol,
        span: Span,
    },
    Unsafe {
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    GhostVariableDef {
        inner: &'input Stmt<'input>,
        span: Span,
    },
    Isolate {
        attributes: Vec<Attribute<'input>>,
        body: Vec<Stmt<'input>>,
        span: Span,
    },
    ImplBlock {
        span: Span,
        attributes: Vec<Attribute<'input>>,
        /// The trait path (`Add<Int<32>>`), or `None` for inherent impls.
        /// Stored as a `Type` so that generic arguments on the trait are
        /// preserved in the AST (e.g. `impl Add<Int<32>> for Type`).
        trait_path: Option<&'input Type<'input>>,
        for_type: Type<'input>,
        methods: Vec<ImplMethod<'input>>,
        associated_types: Vec<AssociatedType<'input>>,
        where_clause: Option<WhereClause<'input>>,
        type_params: Vec<TypeParam<'input>>,
    },
    /// A layout alias definition: `layout Name { packed, little_endian; }`
    LayoutDef {
        name: Symbol,
        attributes: Vec<Attribute<'input>>,
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
pub struct Param<'input> {
    pub name: Symbol,
    pub ty: Option<Type<'input>>,
    pub default: Option<Expr<'input>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeParamKind<'input> {
    Type,
    Lifetime,
    Const {
        ty: Type<'input>,
        default: Option<&'input Expr<'input>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam<'input> {
    pub name: Symbol,
    pub bounds: Vec<Type<'input>>,
    pub kind: TypeParamKind<'input>,
    pub span: Span,
}

/// An anonymous constant expression — used for const generic arguments,
/// array sizes, etc. Analogous to rustc's `AnonConst<'input>`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnonConst<'input> {
    pub value: &'input Expr<'input>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDefinition<'input> {
    Struct(Vec<StructField<'input>>, Vec<TypeModifier<'input>>),
    Enum(
        Vec<EnumVariant<'input>>,
        Option<String>,
        Vec<TypeModifier<'input>>,
    ),
    TraitDef {
        methods: Vec<TraitMethod<'input>>,
        associated_types: Vec<AssociatedType<'input>>,
    },
    ImplBlock {
        trait_path: Option<Vec<Symbol>>,
        for_type: Type<'input>,
        methods: Vec<ImplMethod<'input>>,
    },
    Constraint(Vec<Type<'input>>),
    Alias(Type<'input>, Vec<TypeModifier<'input>>),
    /// Type alias with `impl Trait` (TAIT) — opaque type.
    /// The `Type` is the trait bound (e.g. `Iterator<Item = u32>`).
    Opaque(Type<'input>, Vec<TypeModifier<'input>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeModifier<'input> {
    Overflow(OverflowPolicy),
    Default(Expr<'input>),
    Validate(Expr<'input>),
    NoDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverflowPolicy {
    Wrap,
    Saturate,
    Trap,
    /// IEEE 754 semantics for floats (`with overflow = ieee` — explicit
    /// opt-in; the default is `trap` per the committee ruling).
    Ieee,
}

/// Byte order for `@endian` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    Little,
    Big,
}

/// Bit field fill order for `@bit_order` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitOrder {
    LsbToMsb,
    MsbToLsb,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WherePredicate<'input> {
    pub ty: Type<'input>,
    pub bounds: Vec<Type<'input>>,
    pub span: Span,
}

/// An equality constraint in a where clause: `where T == Int<32>`.
/// The left side must name a generic parameter; the right side is the
/// concrete type it is constrained to.
#[derive(Debug, Clone, PartialEq)]
pub struct WhereEquality<'input> {
    pub left: Type<'input>,
    pub right: Type<'input>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause<'input> {
    pub predicates: smallvec::SmallVec<[WherePredicate<'input>; 2]>,
    /// Equality constraints: `where T == Int<32>`.
    /// Params constrained here are exempt from the E104 generality check.
    pub equalities: Vec<WhereEquality<'input>>,
    /// LIFETIME outlives constraints: `where 'a: 'b` (each entry is the
    /// left lifetime followed by the lifetimes it must outlive — rustc's
    /// `WherePredicateKind::RegionPredicate` `'a: 'b + 'c`).  Collected
    /// into the region solver's constraint graph and verified at the
    /// signature boundary (SYNTAX.md §Explicit Lifetime Parameters —
    /// "verified by the borrow checker; mismatches cause compile
    /// errors").
    pub lifetime_outlives: Vec<(Symbol, Vec<Symbol>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField<'input> {
    pub name: Symbol,
    pub ty: Type<'input>,
    pub default: Option<Expr<'input>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant<'input> {
    pub name: Symbol,
    pub payload: Option<Type<'input>>,
    /// GADT constraints: `(T, Int<32>)` means `T == Int<32>`.
    /// Parsed from `when T == ConcreteType [and ...]`.
    /// Only type equality constraints are supported.
    pub eq_spec: Vec<(Symbol, Type<'input>)>,
    /// Existentially quantified type variables: `exists X, Y`.
    /// These are scoped to the variant's fields and `when` clause.
    pub exists_params: Vec<Symbol>,
    pub span: Span,
}

impl<'input> EnumVariant<'input> {
    /// Whether this variant is a GADT constructor — i.e. it carries either
    /// `when` type constraints (`eq_spec`) or existentially quantified
    /// type variables (`exists_params`).  An explicit predicate (mirroring
    /// Dromedary's `is_constr_generalized`) so call sites do not have to
    /// inline `!eq_spec.is_empty() || !exists_params.is_empty()`.
    pub fn is_gadt(&self) -> bool {
        !self.eq_spec.is_empty() || !self.exists_params.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod<'input> {
    pub name: Symbol,
    pub params: Vec<Param<'input>>,
    pub return_type: Type<'input>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociatedType<'input> {
    pub name: Symbol,
    /// GAT lifetime parameters: `type Item<'a> [where ...] = ...;`
    /// (SYNTAX.md §GAT Declaration — a GAT may be parameterized by
    /// lifetimes; no new TYPE parameters are permitted).  Empty for a
    /// plain associated type.
    pub lifetime_params: Vec<Symbol>,
    pub default: Option<Type<'input>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplMethod<'input> {
    pub name: Symbol,
    pub attributes: Vec<Attribute<'input>>,
    pub params: Vec<Param<'input>>,
    pub return_type: Type<'input>,
    pub body: Option<Vec<Stmt<'input>>>,
    pub span: Span,
}

/// A single generic argument, either positional (`T`) or named (`size = T`).
#[derive(Debug, Clone, PartialEq)]
pub enum GenericArg<'input> {
    Positional(Type<'input>),
    Named(Symbol, Type<'input>),
    Const(AnonConst<'input>),
}

impl<'input> GenericArg<'input> {
    pub fn ty(&self) -> Cow<'_, Type<'input>> {
        match self {
            GenericArg::Positional(ty) | GenericArg::Named(_, ty) => Cow::Borrowed(ty),
            GenericArg::Const(ac) => Cow::Owned(Type::Expr(ac.value, ac.span)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type<'input> {
    Path(smallvec::SmallVec<[Symbol; 4]>, Span),
    Generic(&'input Type<'input>, Vec<GenericArg<'input>>, Span),
    Reference {
        inner: &'input Type<'input>,
        mutable: bool,
        lifetime: Option<Symbol>,
        span: Span,
    },
    Pointer(&'input Type<'input>, Span),
    Slice(&'input Type<'input>, Span),
    Array(&'input Type<'input>, &'input Expr<'input>, Span),
    Tuple(Vec<Type<'input>>, Span),
    Function {
        params: Vec<Type<'input>>,
        ret: &'input Type<'input>,
        span: Span,
    },
    /// Qualified path projection: `<ImplType as TraitPath>::AssocName`
    Projection {
        impl_type: &'input Type<'input>,
        trait_path: &'input Type<'input>,
        assoc_name: Symbol,
        span: Span,
    },
    DynTrait(Vec<Type<'input>>, Span),
    /// Higher-ranked type: `for<'a> T` (SYNTAX.md §Higher-Ranked Trait
    /// Bounds — "for<'a> introduces one or more lifetime parameters scoped
    /// over the subsequent trait bound").  The lifetime is universally
    /// quantified over `body`; the checker skolemizes it at the call site
    /// (rustc's HRTB instantiation) and rejects if it escapes.
    Forall {
        lifetime: Symbol,
        body: &'input Type<'input>,
        span: Span,
    },
    Exists {
        name: Symbol,
        base: &'input Type<'input>,
        invariant: &'input Expr<'input>,
        span: Span,
    },
    /// Shorthand `type T = Base where value > 0` — the parser produces this instead of
    /// doing semantic name generation. A later desugaring pass rewrites it to `Exists`.
    WhereShorthand {
        base: &'input Type<'input>,
        invariant: &'input Expr<'input>,
        span: Span,
    },
    Literal(&'input Expr<'input>, Span),
    Never(Span),
    Union(Vec<Type<'input>>, Span),
    /// A constant expression where a type is expected, e.g. array sizes
    /// `[Int<32>; N + 1]` or generic const args `<Array<Int, N>>`.
    Expr(&'input Expr<'input>, Span),
    /// A compile-time validated regular expression: `Regex<"pattern">`.
    Regex(String, Span),
    Error(Span),
}

/// A polymorphic type scheme: `forall T1, T2, ... . body`
#[derive(Debug, Clone, PartialEq)]
pub struct TypeScheme<'input> {
    pub quantifiers: Vec<(Span, Symbol)>,
    pub body: &'input Type<'input>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<'input> {
    Wildcard(Span),
    Ident(Symbol, Span),
    Literal(&'input Expr<'input>, Span),
    Tuple(Vec<Pattern<'input>>, Span),
    Struct {
        path: Vec<Symbol>,
        fields: Vec<(Symbol, Pattern<'input>)>,
        rest: bool,
        span: Span,
    },
    Enum {
        path: Vec<Symbol>,
        variant: Symbol,
        inner: Option<&'input Pattern<'input>>,
        span: Span,
    },
    Or(Vec<Pattern<'input>>, Span),
    Slice(
        Vec<Pattern<'input>>,
        Option<&'input Pattern<'input>>,
        Vec<Pattern<'input>>,
        Span,
    ),
    Error(Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute<'input> {
    pub name: Symbol,
    pub args: Vec<Expr<'input>>,
    pub named_args: Vec<(Symbol, Expr<'input>)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnsuresTarget<'input> {
    Unconditional,
    OnOk(Option<Pattern<'input>>),
    OnErr(Option<Pattern<'input>>),
    OnTimeout,
    OnCancel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Contract<'input> {
    Requires(Expr<'input>, Span),
    Ensures {
        expr: Expr<'input>,
        span: Span,
        target: EnsuresTarget<'input>,
        /// Path labels referenced in this ensures clause: `ensures @label expr`.
        /// Each label acts as a placeholder for the value returned on paths
        /// marked with that label.
        labels: Vec<Symbol>,
    },
    Invariant(Expr<'input>, Span),
    Decreases(Expr<'input>, Span),
    Terminates(Expr<'input>, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program<'input> {
    pub items: Vec<Stmt<'input>>,
    pub span: Span,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl<'input> Type<'input> {
    pub fn span(&self) -> Span {
        match self {
            Type::Path(_, span)
            | Type::Reference { span, .. }
            | Type::Pointer(_, span)
            | Type::Slice(_, span)
            | Type::Array(_, _, span)
            | Type::Tuple(_, span)
            | Type::Function { span, .. }
            | Type::Projection { span, .. }
            | Type::DynTrait(_, span)
            | Type::Exists { span, .. }
            | Type::WhereShorthand { span, .. }
            | Type::Forall { span, .. }
            | Type::Literal(_, span)
            | Type::Never(span)
            | Type::Union(_, span)
            | Type::Expr(_, span)
            | Type::Regex(_, span)
            | Type::Error(span) => *span,
            Type::Generic(_, _, span) => *span,
        }
    }
}

impl<'input> Stmt<'input> {
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
            Stmt::LayoutDef { span, .. } => *span,
            Stmt::Generate { span, .. } => *span,
            Stmt::Error(span) => *span,
        }
    }
}

impl<'input> Expr<'input> {
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
            Expr::Old(_, span) => *span,
            Expr::Task { span, .. } => *span,
            Expr::TypeInfo(_, span) => *span,
            Expr::LayoutOf(_, span) => *span,
            Expr::CompileError(_, span) => *span,
            Expr::Error(span) => *span,
        }
    }
}

/// Display an AST type as a human-readable string (path/generic/literal
/// recursion).  Single source of truth — previously duplicated verbatim in
/// `checker::fn_ctxt::ast_type_display` and `resolver::ast_type_display`.
pub(crate) fn ast_type_display(ty: &Type) -> String {
    match ty {
        Type::Path(p, _) => p.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("::"),
        Type::Generic(base, args, _) => {
            let args = args
                .iter()
                .map(|a| match a {
                    GenericArg::Positional(t) => ast_type_display(t),
                    GenericArg::Named(n, t) => {
                        format!("{}: {}", n, ast_type_display(t))
                    }
                    GenericArg::Const(ac) => match ac.value {
                        Expr::Literal(l, _) => format!("{:?}", l),
                        e => format!("{:?}", e),
                    },
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}<{}>", ast_type_display(base), args)
        }
        Type::Literal(e, _) => match e {
            Expr::Literal(l, _) => format!("{:?}", l),
            _ => format!("{:?}", e),
        },
        _ => format!("{:?}", ty),
    }
}
