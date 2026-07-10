use std::fmt;

/// Categorization of compiler error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Lexing or parsing failures.
    Parse,
    /// Name resolution failures.
    Resolution,
    /// Type checker errors.
    Type,
    /// Contract verification errors.
    Contract,
    /// Trait system errors.
    Trait,
    /// Inference engine errors.
    Inference,
    /// Internal compiler errors.
    Internal,
    /// Generic / unclassified errors.
    Generic,
}

impl ErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCategory::Parse => "Parse Error",
            ErrorCategory::Resolution => "Resolution Error",
            ErrorCategory::Type => "Type Error",
            ErrorCategory::Contract => "Contract Error",
            ErrorCategory::Trait => "Trait Error",
            ErrorCategory::Inference => "Inference Error",
            ErrorCategory::Internal => "Internal Compiler Error",
            ErrorCategory::Generic => "Error",
        }
    }

    pub fn ansi_color(&self) -> &'static str {
        match self {
            ErrorCategory::Internal => "\x1b[31;1m", // bright red
            _ => "\x1b[31m",                          // normal red
        }
    }
}

/// All error codes emitted by the compiler, with attached documentation
/// for the `--explain` feature (inspired by `rustc --explain`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // ── Parse errors (E001–E009) ──────────────────────────────
    ExpectedToken,
    UnexpectedEOF,
    UnexpectedToken,
    ParseError,
    ExpectedIdentifier,
    RecursionLimitExceeded,

    // ── Resolution errors (E010–E019) ─────────────────────────
    NoSuchField,
    TypeNotFound,
    NameNotFound,
    UndefinedType,
    GenericArgsOnNonGeneric,
    CannotResolveImport,
    NoDefaultValue,
    ArraySizeNotConstant,
    UnexpectedTopLevel,

    // ── Contract errors (E020–E029) ───────────────────────────
    ContractNonBool,
    EnsuresNonBool,
    DecreasesNonInt,

    // ── Type errors (E030–E099) ──────────────────────────────
    TypeMismatch,
    InvalidBinaryOp,
    InvalidUnaryOp,
    WrongNumberOfArgs,
    ExpectedBool,
    ExpectedInteger,
    ExpectedResult,
    ExpectedFuture,
    ReturnOutsideFunction,
    ReturnWithoutValue,
    LeaveOutsideLoop,
    ContinueOutsideLoop,
    CannotLeaveClosure,
    SetNoPattern,
    LetNeedsInit,
    InvalidLValue,
    ContractBoolAtReturn,

    // ── Trait / impl errors (E100–E199) ───────────────────────
    TraitNotFound,
    ImplMissingMethod,
    ImplMissingAssocType,
    ImplSignatureMismatch,
    OrphanImpl,
    InherentImplOnNonAdt,
    TraitViolatesTermination,

    // ── Cast errors (E600–E699) ──────────────────────────────
    SafeCastFromRef,
    SafeCastNonPrimitive,
    UnsafeCastRefToInt,
    UnsafeCastIncompatible,
    UnknownAttribute,
}

impl ErrorCode {
    /// The short code string, e.g. `"E030"`.
    pub fn code(&self) -> &'static str {
        use ErrorCode::*;
        match self {
            ExpectedToken => "E001",
            UnexpectedEOF => "E002",
            UnexpectedToken => "E003",
            ParseError => "E004",
            ExpectedIdentifier => "E005",
            RecursionLimitExceeded => "E006",

            NoSuchField => "E010",
            TypeNotFound => "E011",
            NameNotFound => "E012",
            UndefinedType => "E013",
            GenericArgsOnNonGeneric => "E014",
            CannotResolveImport => "E015",
            NoDefaultValue => "E016",
            ArraySizeNotConstant => "E017",
            UnexpectedTopLevel => "E018",

            ContractNonBool => "E020",
            EnsuresNonBool => "E021",
            DecreasesNonInt => "E022",

            TypeMismatch => "E030",
            InvalidBinaryOp => "E031",
            InvalidUnaryOp => "E032",
            WrongNumberOfArgs => "E033",
            ExpectedBool => "E034",
            ExpectedInteger => "E035",
            ExpectedResult => "E036",
            ExpectedFuture => "E037",
            ReturnOutsideFunction => "E038",
            ReturnWithoutValue => "E039",
            LeaveOutsideLoop => "E040",
            ContinueOutsideLoop => "E041",
            CannotLeaveClosure => "E042",
            SetNoPattern => "E043",
            LetNeedsInit => "E044",
            InvalidLValue => "E045",
            ContractBoolAtReturn => "E046",

            TraitNotFound => "E100",
            ImplMissingMethod => "E101",
            ImplMissingAssocType => "E102",
            ImplSignatureMismatch => "E103",
            OrphanImpl => "E104",
            InherentImplOnNonAdt => "E105",
            TraitViolatesTermination => "E106",

            SafeCastFromRef => "E600",
            SafeCastNonPrimitive => "E601",
            UnsafeCastRefToInt => "E602",
            UnsafeCastIncompatible => "E603",
            UnknownAttribute => "E604",
        }
    }

    /// Human-readable title for the error.
    pub fn title(&self) -> &'static str {
        use ErrorCode::*;
        match self {
            ExpectedToken => "expected token",
            UnexpectedEOF => "unexpected end of file",
            UnexpectedToken => "unexpected token",
            ParseError => "parse error",
            ExpectedIdentifier => "expected identifier",
            RecursionLimitExceeded => "recursion limit exceeded",

            NoSuchField => "field not found on type",
            TypeNotFound => "type definition not found",
            NameNotFound => "name not found",
            UndefinedType => "undefined type",
            GenericArgsOnNonGeneric => "generic arguments on non-generic type",
            CannotResolveImport => "cannot resolve import",
            NoDefaultValue => "no default value",
            ArraySizeNotConstant => "array size not constant",
            UnexpectedTopLevel => "unexpected top-level item",

            ContractNonBool => "contract condition must be boolean",
            EnsuresNonBool => "ensures clause must be boolean",
            DecreasesNonInt => "decreases/terminates expression must be integer",

            TypeMismatch => "type mismatch",
            InvalidBinaryOp => "invalid operands for binary operator",
            InvalidUnaryOp => "invalid operand for unary operator",
            WrongNumberOfArgs => "wrong number of arguments",
            ExpectedBool => "expected bool",
            ExpectedInteger => "expected integer",
            ExpectedResult => "expected Result type",
            ExpectedFuture => "expected Future type",
            ReturnOutsideFunction => "return outside function",
            ReturnWithoutValue => "return without value in non-unit function",
            LeaveOutsideLoop => "leave outside loop",
            ContinueOutsideLoop => "continue outside loop",
            CannotLeaveClosure => "cannot leave out of closure",
            SetNoPattern => "set does not support destructuring",
            LetNeedsInit => "let requires initializer",
            InvalidLValue => "invalid left-hand side for assignment",
            ContractBoolAtReturn => "ensures condition must be boolean at return",

            TraitNotFound => "trait not found",
            ImplMissingMethod => "impl missing required method",
            ImplMissingAssocType => "impl missing required associated type",
            ImplSignatureMismatch => "impl method signature mismatch",
            OrphanImpl => "orphan impl",
            InherentImplOnNonAdt => "inherent impl on non-struct/enum type",
            TraitViolatesTermination => "impl violates termination rules",

            SafeCastFromRef => "safe cast from reference type",
            SafeCastNonPrimitive => "safe cast only between numeric/boolean types",
            UnsafeCastRefToInt => "unsafe cast from reference to integer",
            UnsafeCastIncompatible => "unsafe cast requires compatible types",
            UnknownAttribute => "unknown attribute",
        }
    }

    /// The category this error belongs to.
    pub fn category(&self) -> ErrorCategory {
        use ErrorCode::*;
        match self {
            ExpectedToken | UnexpectedEOF | UnexpectedToken | ParseError
            | ExpectedIdentifier | RecursionLimitExceeded => ErrorCategory::Parse,

            NoSuchField | TypeNotFound | NameNotFound | UndefinedType
            | GenericArgsOnNonGeneric | CannotResolveImport | NoDefaultValue
            | ArraySizeNotConstant | UnexpectedTopLevel => ErrorCategory::Resolution,

            ContractNonBool | EnsuresNonBool | DecreasesNonInt
            | ContractBoolAtReturn => ErrorCategory::Contract,

            TraitNotFound | ImplMissingMethod | ImplMissingAssocType
            | ImplSignatureMismatch | OrphanImpl | InherentImplOnNonAdt
            | TraitViolatesTermination => ErrorCategory::Trait,

            TypeMismatch | InvalidBinaryOp | InvalidUnaryOp | WrongNumberOfArgs
            | ExpectedBool | ExpectedInteger | ExpectedResult | ExpectedFuture
            | ReturnOutsideFunction | ReturnWithoutValue | LeaveOutsideLoop
            | ContinueOutsideLoop | CannotLeaveClosure | SetNoPattern
            | LetNeedsInit | InvalidLValue => ErrorCategory::Type,

            SafeCastFromRef | SafeCastNonPrimitive | UnsafeCastRefToInt
            | UnsafeCastIncompatible | UnknownAttribute => ErrorCategory::Type,
        }
    }

    /// Full explain text, displayed by `ponent explain E030` or `--explain E030`.
    /// Modeled after `rustc --explain` — provides a description, code examples,
    /// and suggested fixes.
    pub fn explain(&self) -> &'static str {
        match self {
            Self::ExpectedToken => "A specific token was expected at the current parse position but not found.

This typically occurs due to:
  - Missing closing delimiters: `)`, `}`, `]`
  - Missing semicolons at end of statements
  - Incomplete syntax in type or expression

Example of invalid code:
  def foo(x: Int<32 {
    return x;
  }

Fix: add the missing token:
  def foo(x: Int<32>) {
    return x;
  }",

            Self::UnexpectedEOF => "The source file ends before the parser expected it to.

This usually means:
  - A block, parentheses, or bracket was left unclosed
  - A function body or expression was cut off

Example:
  def foo(x: Int<32) {
    return x;
  // Missing `}` at end

Fix: ensure all `{`, `(`, `[` are properly closed.",

            Self::UnexpectedToken => "An unexpected token was encountered during parsing.

The parser found a token that does not fit the grammar at the current position.
This may be a typo, a keyword used as an identifier, or misplaced syntax.

Example:
  def foo(x: Int<32>) {
    return x;
  } def // stray `def`

Fix: remove or reposition the stray token.",

            Self::ParseError => "A general parse error — the input does not conform to the language grammar.

Check for:
  - Missing keywords
  - Incorrect punctuation
  - Malformed expressions or types",

            Self::NoSuchField => "A field access was attempted on a struct or enum type that does not have
a field with that name.

Example:
  type Point = struct { x: Int<32>, y: Int<32> }
  def main() {
    set p = Point { x = 10, y = 20 };
    return p.z; // error: no field `z` on type Point
  }

Fix: use an existing field name. Available fields are listed in the error message.",

            Self::TypeNotFound => "A type definition that was referenced could not be found in the symbol table.

This is an internal compiler error if it occurs after successful name resolution.",

            Self::ContractNonBool => "A contract condition (`requires` or `invariant`) must evaluate to a boolean
value (`Bool`), but a non-boolean expression was provided.

Example:
  def foo(x: Int<32>) -> Int<32>
    requires x + 1  // error: `x + 1` is Int<32>, not Bool
  {
    return x;
  }

Fix: change the condition to a boolean expression, e.g. `requires x > 0`.",

            Self::EnsuresNonBool => "An `ensures` clause must evaluate to a boolean value (`Bool`).

Example:
  def foo(x: Int<32>) -> Int<32>
    ensures x * 2  // error
  { return x; }

Fix: `ensures result > 0` or another boolean predicate.",

            Self::DecreasesNonInt => "A `decreases` or `terminates` expression must evaluate to an integer type,
as it is used as a termination metric for recursive functions.

Example:
  def factorial(n: Int<32>) -> Int<32>
    decreases n >= 0  // error: Bool, not integer
  {
    ...
  }

Fix: use an integer metric: `decreases n`.",

            Self::TypeMismatch => "A type mismatch occurred: the compiler expected a value of one type but
found a value of a different type.

This is the most common kind of error in Posita. It can happen when:
  - Passing an argument of the wrong type to a function
  - Assigning a value of the wrong type to a variable
  - Returning a value that doesn't match the declared return type
  - Using an expression in a context that requires a specific type

Example:
  def add(a: Int<32>, b: Int<32>) -> Int<32> {
    return a + b;
  }
  def main() {
    set result = add(10, true); // error: Bool != Int<32>
  }

Fix: ensure the types match. Use explicit casts with `as` if needed.",

            Self::WrongNumberOfArgs => "A function was called with the wrong number of arguments.

Example:
  def add(a: Int<32>, b: Int<32>) -> Int<32> { ... }
  add(10);      // error: expected 2 args, found 1
  add(10, 20, 30); // error: expected 2 args, found 3

Fix: provide exactly the number of parameters declared in the function signature.",

            Self::ExpectedBool => "A boolean value was expected but a non-boolean expression was provided.
This commonly occurs in:
  - `if` and `while` conditions
  - Contract clauses (`requires`, `ensures`, `invariant`)

Example:
  if 42 { ... }  // error: expected Bool, found Int<32>

Fix: use a comparison: `if x != 0 { ... }`.",

            Self::ReturnOutsideFunction => "A `return` statement was used outside of a function body.

`return` is only valid inside `def` function bodies. It cannot appear at the
top level of a module or inside a `type` definition.

Example:
  return 42; // error: top-level return

Fix: wrap the code in a function, or remove the `return`.",

            Self::LeaveOutsideLoop => "A `leave` statement was used outside of a loop construct
(`while`, `for`, `loop`, or a labeled block).

`leave` is used to exit loops early. Use `return` to exit a function.

Example:
  def foo() {
    leave; // error: no loop to leave
  }

Fix: use `return` instead, or enclose the code in a loop.",

            Self::ImplMissingMethod => "An `impl` block is missing a required method from the trait it implements.

Every method declared in the trait must have a corresponding `def` in the impl block.

Example:
  trait Display {
    def format(&self) -> Str;
  }
  impl Display for MyType {
    // error: missing method `format`
  }

Fix: add the missing method to the impl block.",

            Self::OrphanImpl => "An `impl` block violates the orphan rule: the type and trait must be
defined in the current crate, or the trait must be from the current crate.

This restriction prevents conflicting implementations across crates.",

            _ => "No detailed explanation is available for this error code yet.",
        }
    }
    /// The diagnostic URL for the `--explain` feature.
    /// Returns a URL to the online error code documentation,
    /// matching the pattern of rustc's `https://doc.rust-lang.org/error_codes/E030.html`.
    pub fn url(&self) -> String {
        format!("https://doc.posita-lang.org/error_codes/{}.html", self.code())
    }

    /// The diagnostic URL, formatted as an ANSI hyperlink if the terminal supports it.
    pub fn url_ansi(&self) -> String {
        format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", self.url(), self.code())
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} {}]", self.code(), self.category().as_str())
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
