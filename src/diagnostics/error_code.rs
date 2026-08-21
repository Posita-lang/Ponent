use std::fmt;
use std::sync::OnceLock;

/// The port number of the local explain server, if running.
static EXPLAIN_PORT: OnceLock<u16> = OnceLock::new();

/// Set the port for the local explain server.  Called by the CLI before
/// displaying error code URLs.
pub fn set_explain_port(port: u16) {
    let _ = EXPLAIN_PORT.set(port);
}

/// The URL for the `--explain` feature.  Returns a `localhost` URL when
/// the local explain server is running, otherwise the canonical doc URL.
fn explain_url(code: &str) -> String {
    if let Some(&port) = EXPLAIN_PORT.get() {
        format!("http://127.0.0.1:{port}/{code}")
    } else {
        // We will register this domain and set up a proper documentation
        // website in the future.  For now, the local explain server is the
        // primary way to view error explanations in a browser.
        format!("https://doc.posita-lang.org/error_codes/{code}.html")
    }
}

/// Categorization of compiler error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Parse,
    Resolution,
    Type,
    Contract,
    Trait,
    Inference,
    Internal,
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
            ErrorCategory::Internal => "Internal Error",
            ErrorCategory::Generic => "Error",
        }
    }
}

/// A compiler error or warning code, stored as a string (e.g. "E030", "W113").
/// Metadata (title, category, explanation) is resolved through a lookup table,
/// removing the need for exhaustive match arms on enum variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrCode(String);

/// Error returned when an unknown error code string is used.
#[derive(Debug, Clone)]
pub struct UnknownCode(pub String);

impl fmt::Display for UnknownCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown error code: `{}`", self.0)
    }
}

impl std::error::Error for UnknownCode {}

impl ErrCode {
    /// Create a new `ErrCode` without validation.
    ///
    /// In debug builds, a `debug_assert!` panics if the code is not in the
    /// lookup table.  In release builds the code is accepted as-is and will
    /// silently fall back to "unknown error" for `title()` / `category()` /
    /// `explain()`.
    ///
    /// Prefer [`Self::try_new`] when the input comes from an external source
    /// (user input, CLI arguments, etc.).
    pub fn new(code: impl Into<String>) -> Self {
        let code = code.into();
        debug_assert!(
            lookup(&code).is_some(),
            "unknown error code: {code:?} — must be added to CODE_TABLE in error_code.rs",
        );
        ErrCode(code)
    }

    /// Validate that `code` exists in the lookup table, returning
    /// [`UnknownCode`] if it does not.  This check runs in **all** build
    /// profiles (unlike the `debug_assert!` in [`Self::new`]).
    pub fn try_new(code: impl Into<String>) -> Result<Self, UnknownCode> {
        let code = code.into();
        if lookup(&code).is_some() {
            Ok(ErrCode(code))
        } else {
            Err(UnknownCode(code))
        }
    }

    /// The code string, e.g. "E030" or "W113".
    pub fn code(&self) -> &str {
        &self.0
    }

    /// Short title, e.g. "type mismatch" or "duplicate definition".
    pub fn title(&self) -> &'static str {
        lookup(self.0.as_str())
            .map(|e| e.title)
            .unwrap_or("unknown error")
    }

    /// The error category, e.g. `ErrorCategory::Type`.
    pub fn category(&self) -> ErrorCategory {
        lookup(self.0.as_str())
            .map(|e| e.category)
            .unwrap_or(ErrorCategory::Generic)
    }

    /// Full explanation text, displayed by `ponent --explain E030`.
    pub fn explain(&self) -> &'static str {
        lookup(self.0.as_str())
            .map(|e| e.explain)
            .unwrap_or("No detailed explanation is available for this error code yet.")
    }

    /// The diagnostic URL for the `--explain` feature.
    pub fn url(&self) -> String {
        explain_url(&self.0)
    }

    /// The diagnostic URL, formatted as an ANSI hyperlink if the terminal supports it.
    pub fn url_ansi(&self) -> String {
        format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", self.url(), self.0)
    }
}

/// A single entry in the error code lookup table.
pub(crate) struct CodeEntry {
    pub(crate) code: &'static str,
    pub(crate) title: &'static str,
    pub(crate) category: ErrorCategory,
    explain: &'static str,
}

/// Lookup table for error/warning code metadata.
/// Maps code strings to their title, category, and explanation.
pub(crate) const CODE_TABLE: &[CodeEntry] = &[
    CodeEntry {
        code: "E001",
        title: "expected token",
        category: ErrorCategory::Parse,
        explain: "A specific token was expected at the current parse position but not found.\n\nThis typically occurs due to:\n  - Missing closing delimiters: `)`, `}`, `]`\n  - Missing semicolons at end of statements\n  - Incomplete syntax in type or expression\n\nExample of invalid code:\n  def foo(x: Int<32 {\n    return x;\n  }\n\nFix: add the missing token at the indicated position.",
    },
    CodeEntry {
        code: "E002",
        title: "unexpected end of input",
        category: ErrorCategory::Parse,
        explain: "The parser reached the end of the input while still expecting more tokens.\n\nThis usually means the code is incomplete.",
    },
    CodeEntry {
        code: "E003",
        title: "unexpected token",
        category: ErrorCategory::Parse,
        explain: "An unexpected token was encountered at the current parse position.\n\nThis typically indicates a syntax error such as missing punctuation, extra characters, or incorrect ordering of keywords.\n\nExample:\n  def foo(x: Int<32) { return; } }  // extra `}`\n\nFix: review the syntax at the indicated position and remove or correct the unexpected token.",
    },
    CodeEntry {
        code: "E004",
        title: "parse error",
        category: ErrorCategory::Parse,
        explain: "A general parse error occurred.",
    },
    CodeEntry {
        code: "E005",
        title: "expected identifier",
        category: ErrorCategory::Parse,
        explain: "An identifier (name) was expected but not found.\n\nIdentifiers are used for variable names, function names, type names, etc.\n\nExample:\n  def 123() { }  // error: expected identifier, found number\n\nFix: use a valid identifier name instead.",
    },
    CodeEntry {
        code: "E006",
        title: "recursion limit exceeded",
        category: ErrorCategory::Parse,
        explain: "The parser exceeded the recursion limit.\n\nThis may indicate deeply nested expressions or a bug in the parser.",
    },
    CodeEntry {
        code: "E007",
        title: "expected expression",
        category: ErrorCategory::Parse,
        explain: "An expression was expected at the current position but not found.\n\nExpressions are values, variables, function calls, operators, etc. This typically occurs after an operator or assignment that expects a value.\n\nExample:\n  set x = ;  // error: expected expression after `=`\n\nFix: provide a valid expression at the indicated position.",
    },
    CodeEntry {
        code: "E008",
        title: "return Err is not valid; use leave with",
        category: ErrorCategory::Parse,
        explain: "`return Err(...)` is not a valid error exit in Posita.\n\n`leave with` is the only valid error exit (SYNTAX.md §Error Handling); it\nis recorded as an `ErrorExit` in the control-flow graph for audit.\n\nFix: replace `return Err(x)` with `leave with x`.",
    },
    CodeEntry {
        code: "E009",
        title: "invalid character literal",
        category: ErrorCategory::Parse,
        explain: "Invalid character literal.",
    },
    CodeEntry {
        code: "E010",
        title: "no such field",
        category: ErrorCategory::Resolution,
        explain: "A field access refers to a field that does not exist on the given type.",
    },
    CodeEntry {
        code: "E011",
        title: "type not found",
        category: ErrorCategory::Resolution,
        explain: "A type name could not be resolved in the current scope.",
    },
    CodeEntry {
        code: "E012",
        title: "name not found",
        category: ErrorCategory::Resolution,
        explain: "A name could not be resolved in the current scope.",
    },
    CodeEntry {
        code: "E013",
        title: "undefined type",
        category: ErrorCategory::Resolution,
        explain: "The type has not been defined in the current scope.",
    },
    CodeEntry {
        code: "E014",
        title: "generic args on non-generic",
        category: ErrorCategory::Resolution,
        explain: "Generic type arguments were provided for a non-generic type.",
    },
    CodeEntry {
        code: "E015",
        title: "cannot resolve import",
        category: ErrorCategory::Resolution,
        explain: "An import could not be resolved.",
    },
    CodeEntry {
        code: "E016",
        title: "no default value",
        category: ErrorCategory::Resolution,
        explain: "A type has no default value and no initializer was provided.",
    },
    CodeEntry {
        code: "E017",
        title: "array size not constant",
        category: ErrorCategory::Resolution,
        explain: "Array size must be a constant expression.",
    },
    CodeEntry {
        code: "E018",
        title: "unexpected top-level item",
        category: ErrorCategory::Resolution,
        explain: "`set` and `let` statements are not allowed at the top level; only declarations (`def`, `type`, `trait`, `import`, `impl`, `constraint`, `comptime`, `extern`, `edition`) are permitted here.",
    },
    CodeEntry {
        code: "E019",
        title: "duplicate definition",
        category: ErrorCategory::Resolution,
        explain: "A variable, function, or type has been defined more than once in the same scope.\n\nThis is not allowed because the second definition would shadow the first without\nany way to refer to the original binding.\n\nExample of invalid code:\n  set x = 1;\n  set x = 2;  // error: duplicate definition of `x`\n\nFix: use a different name for the second definition, or remove the first one.",
    },
    CodeEntry {
        code: "E020",
        title: "contract condition must be boolean",
        category: ErrorCategory::Contract,
        explain: "A contract condition (`requires` or `invariant`) must evaluate to a boolean\nvalue (`Bool`), but a non-boolean expression was provided.\n\nExample:\n  def foo(x: Int<32>) -> Int<32>\n    requires x + 1  // error: `x + 1` is Int<32>, not Bool\n\nFix: ensure the condition evaluates to a boolean value.",
    },
    CodeEntry {
        code: "E021",
        title: "ensures clause must be boolean",
        category: ErrorCategory::Contract,
        explain: "An `ensures` clause must evaluate to a boolean value (`Bool`).",
    },
    CodeEntry {
        code: "E022",
        title: "decreases expression must be integer",
        category: ErrorCategory::Contract,
        explain: "A `decreases`/`terminates` expression must be an integer type.",
    },
    CodeEntry {
        code: "E023",
        title: "contract boolean at return",
        category: ErrorCategory::Contract,
        explain: "A contract condition must be boolean.",
    },
    CodeEntry {
        code: "E030",
        title: "type mismatch",
        category: ErrorCategory::Type,
        explain: "The type of an expression does not match the expected type.\n\nThis error occurs when a value is assigned to a variable, passed as an argument,\nor returned from a function with a different type than expected.\n\nExample:\n  def foo() -> Int<32> {\n    return \"hello\";  // error: expected Int<32>, found &Str\n  }\n\nFix: ensure the expression has the correct type, or add an explicit cast.",
    },
    CodeEntry {
        code: "E031",
        title: "kind mismatch",
        category: ErrorCategory::Type,
        explain: "The kind of a type does not match the expected kind.\n\nThis occurs when a type inference variable has a kind constraint (e.g. Integer,\nBool, Float) that conflicts with the resolved concrete type.  For example, using\na string variable where an integer is expected.\n\nExample:\n  set j = \"hello\";\n  set i = j + 1;  // error: expected integer type, found &Str\n\nFix: ensure the value has the correct type kind for the operation.",
    },
    CodeEntry {
        code: "E032",
        title: "operator type error",
        category: ErrorCategory::Type,
        explain: "An operator is applied to incompatible operand types.\n\nThis occurs when binary or unary operators are used with types that do not\nsupport the operation.  For example, adding a string to an integer.\n\nExample:\n  set j = \"hello\";\n  set i = j + 1;  // error: cannot add &Str and Int\n\nFix: ensure both operands have compatible types for the operator.",
    },
    CodeEntry {
        code: "E033",
        title: "cannot infer type",
        category: ErrorCategory::Type,
        explain: "The type of an expression could not be inferred.",
    },
    CodeEntry {
        code: "E034",
        title: "infinite type",
        category: ErrorCategory::Type,
        explain: "Infinite type detected during unification.",
    },
    CodeEntry {
        code: "E035",
        title: "type annotation needed",
        category: ErrorCategory::Type,
        explain: "A type annotation is needed for this expression.",
    },
    CodeEntry {
        code: "E036",
        title: "return type mismatch",
        category: ErrorCategory::Type,
        explain: "The return value type does not match the function's declared return type.\n\nExample:\n  def foo() -> Int<32> {\n    return true;  // error: expected Int<32>, found Bool\n  }\n\nFix: ensure the return value has the correct type, or change the function's\nreturn type annotation.",
    },
    CodeEntry {
        code: "E037",
        title: "argument type mismatch",
        category: ErrorCategory::Type,
        explain: "A function argument's type does not match the parameter type.\n\nExample:\n  def foo(x: Int<32>) { }\n  def main() { foo(true); }  // error: expected Int<32>, found Bool\n\nFix: pass an argument of the correct type, or change the parameter type.",
    },
    CodeEntry {
        code: "E038",
        title: "condition must be boolean",
        category: ErrorCategory::Type,
        explain: "A condition expression (in `if`, `while`, `requires`, `ensures`, etc.) must\nbe of type `Bool`, but a non-boolean expression was provided.\n\nExample:\n  if 42 { }  // error: condition must be Bool, found Int<32>\n\nFix: use a boolean expression as the condition.",
    },
    CodeEntry {
        code: "E039",
        title: "index must be integer",
        category: ErrorCategory::Type,
        explain: "An index expression must be an integer type, but a non-integer type was\nprovided.\n\nExample:\n  arr[\"hello\"]  // error: index must be integer, found &Str\n\nFix: use an integer expression as the index.",
    },
    CodeEntry {
        code: "E040",
        title: "trait not found",
        category: ErrorCategory::Trait,
        explain: "The specified trait could not be found.",
    },
    CodeEntry {
        code: "E041",
        title: "trait not implemented",
        category: ErrorCategory::Trait,
        explain: "A required trait is not implemented for the given type.",
    },
    CodeEntry {
        code: "E042",
        title: "orphan impl",
        category: ErrorCategory::Trait,
        explain: "An `impl` block violates the orphan rule: the type and trait must be\ndefined in the current crate, or the trait must be from the current crate.\n\nThis restriction prevents conflicting implementations across crates.",
    },
    CodeEntry {
        code: "E043",
        title: "conflicting impl",
        category: ErrorCategory::Trait,
        explain: "Conflicting implementations of a trait for the same type.",
    },
    CodeEntry {
        code: "E050",
        title: "inference error",
        category: ErrorCategory::Inference,
        explain: "A type inference error occurred.",
    },
    CodeEntry {
        code: "E051",
        title: "cannot infer type",
        category: ErrorCategory::Inference,
        explain: "The type of an expression could not be inferred. Try adding a type annotation.",
    },
    CodeEntry {
        code: "E052",
        title: "interrupt must satisfy @no_alloc",
        category: ErrorCategory::Resolution,
        explain: "An @interrupt handler must satisfy the @no_alloc capability.\n\nInterrupt handlers run in a constrained context where memory allocation\nis not available. The function must be annotated with @no_alloc or be\nin a scope where @no_alloc is already in effect.\n\nFix: add `@no_alloc` to this function (or remove it if redundant with\n`@no_panic`).",
    },
    CodeEntry {
        code: "E053",
        title: "interrupt must satisfy @no_panic",
        category: ErrorCategory::Resolution,
        explain: "An @interrupt handler must satisfy the @no_panic capability.\n\nInterrupt handlers must not panic because there is no unwind mechanism\navailable. The function must be annotated with @no_panic.\n\nFix: add `@no_panic` to this function.",
    },
    CodeEntry {
        code: "E054",
        title: "interrupt cannot have @alloc",
        category: ErrorCategory::Resolution,
        explain: "An @interrupt handler cannot have the @alloc capability.\n\nInterrupt handlers must not allocate memory. The @alloc annotation is\nincompatible with @interrupt.\n\nFix: remove the `@alloc` annotation from this function.",
    },
    CodeEntry {
        code: "E055",
        title: "interrupt cannot have @io",
        category: ErrorCategory::Resolution,
        explain: "An @interrupt handler cannot have the @io capability.\n\nInterrupt handlers must not perform I/O operations. The @io annotation\nis incompatible with @interrupt.\n\nFix: remove the `@io` annotation from this function.",
    },
    CodeEntry {
        code: "E060",
        title: "GADT variant constraint violation",
        category: ErrorCategory::Type,
        explain: "A GADT variant's `when` constraint is not satisfied after solving\nand defaulting, or cannot be verified because a type argument is\nstill unresolved (fail-closed).\n\nFix: make the type arguments satisfy the variant's `when` constraint,\nor provide concrete type arguments at the construction site.",
    },
    CodeEntry {
        code: "E061",
        title: "unreachable code",
        category: ErrorCategory::Internal,
        explain: "The compiler reached an unreachable code path. This is a bug.",
    },
    CodeEntry {
        code: "E062",
        title: "main function not found",
        category: ErrorCategory::Resolution,
        explain: "The crate does not define a `main` function, which is required for executable output.\n\nEvery executable crate must have a `main` function that serves as the entry\npoint for the program.  The `main` function takes no arguments and returns\nan integer type (e.g. `Int<32>`).\n\nFix: add a `def main() { ... }` function to the crate.",
    },
    CodeEntry {
        code: "E063",
        title: "existential variable shadows enum type parameter",
        category: ErrorCategory::Resolution,
        explain: "An `exists` variable in a GADT variant cannot have the same name as an enum type parameter.\n\nFor example, the following is invalid because `X` is already an enum parameter:\n  type Wrap<X> = enum { Bad(exists X: &[X]) when T == [X] }\n\nFix: rename the `exists` variable or the enum type parameter.",
    },
    CodeEntry {
        code: "E064",
        title: "GADT `when` constraint right-hand side references a same-enum type parameter",
        category: ErrorCategory::Resolution,
        explain: "The right-hand side of a GADT `when` constraint cannot reference another type parameter of the SAME enum.\n\nFor example, the following is invalid because `U` is a type parameter of `Bad`:\n  type Bad<T, U> = enum { Mk(T) when T == U }\n\nA mutual constraint (`when T == U and U == T`) would register a refinement cycle (A → B, B → A) that type resolution would chase until the chain-depth limit.\n\nFix: use a concrete type on the right-hand side, or an `exists` variable scoped to the variant (the witness stays opaque).",
    },
    CodeEntry {
        code: "E065",
        title: "conflicting GADT `when` constraints in variant",
        category: ErrorCategory::Resolution,
        explain: "A GADT variant's `when` constraints force the same type parameter to two different concrete types (e.g. `when T == Int<32> and T == Bool`). The constraint set is unsatisfiable, so the variant cannot be constructed at any instantiation — a logical contradiction, not a style concern.\n\nFix: make the constraints consistent (e.g. `when T == Int<32>`), or drop the redundant one.",
    },
    CodeEntry {
        code: "E066",
        title: "conflicting GADT refinements in or-pattern",
        category: ErrorCategory::Resolution,
        explain: "An or-pattern (`pattern1 | pattern2`) refines the same enum type parameter to DIFFERENT types in different alternatives.\n\nFor example, the following is invalid because `Add` refines `T == Int<32>` while `Eq` refines `T == Bool`:\n  type Expr<T> = enum { Add(...) when T == Int<32>, Eq(...) when T == Bool }\n  match e { Add(_, _) | Eq(_, _) => ... }\n\nA branch guarded by a disjunction may only assume facts true in ALL alternatives. Conflicting refinements are a contradiction.\n\nFix: split the or-pattern into separate match arms, or make the alternatives refine the type parameter consistently.",
    },
    CodeEntry {
        code: "E070",
        title: "unknown edition",
        category: ErrorCategory::Resolution,
        explain: "The `--edition` / `@edition` value is not one of the supported editions (e.g. `\"2024\"` or `\"2026\"`).  The edition selects the language defaults (reserved keywords, overflow policy, …) for the compilation unit.\n\nFix: use a valid edition string.",
    },
    CodeEntry {
        code: "E100",
        title: "trait not found",
        category: ErrorCategory::Trait,
        explain: "A trait referenced in an `impl` or a bound could not be resolved.\n\nFix: check the trait name and ensure it is defined or imported in scope.",
    },
    CodeEntry {
        code: "E101",
        title: "trait impl missing method",
        category: ErrorCategory::Trait,
        explain: "A trait implementation is missing a required method.\n\nEvery trait method must be implemented in the impl block. This error\noccurs when a method declared in the trait is not defined in the impl.\n\nFix: add a `def` for the missing method in this impl block.",
    },
    CodeEntry {
        code: "E102",
        title: "orphan impl or bare type variable",
        category: ErrorCategory::Trait,
        explain: "A trait impl could not be registered: either it violates the orphan rule\n(a type/trait from another crate), or a bare type variable appeared\nwithout sufficient context to determine its kind.\n\nFix: ensure the impl follows the orphan rule, or provide a type\nannotation for the bare type variable.",
    },
    CodeEntry {
        code: "E103",
        title: "trait impl wrong parameter count",
        category: ErrorCategory::Trait,
        explain: "A trait method implementation has a different number of parameters than\nthe trait declaration.\n\nEvery method in a trait impl must have the same number of parameters as\nthe corresponding trait method signature.\n\nFix: adjust the parameter count to match the trait declaration.",
    },
    CodeEntry {
        code: "E104",
        title: "generic parameter constrained by function body",
        category: ErrorCategory::Resolution,
        explain: "The body of a generic function constrains a type parameter to a concrete type (or to another type parameter), so the definition does not type-check for ALL instantiations.\n\nFor example, the following is invalid because the body only works when `T` is `Int<32>`:\n  def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }\n  def g<T>(x: T) -> Int<32> { return add(x, 1); }\n\nBinding one parameter to another is likewise invalid (`T := U`), since the body must be parametric in each distinct parameter:\n  def f<T, U>(x: T, y: U) -> U { return x; }\n\nA generic definition must be valid for every type argument; a body that forces `T = Int<32>` would make `g<Bool>` ill-typed.  Rust (rigid `TyKind::Param`), GHC and OCaml (skolems) all reject such definitions at definition time.\n\nFix: change the signature so the constraint is explicit (e.g. a where-clause), or make the body parametric in `T`.",
    },
    CodeEntry {
        code: "E105",
        title: "or-pattern alternatives must bind the same variables",
        category: ErrorCategory::Resolution,
        explain: "Every alternative of an or-pattern (`pattern1 | pattern2`) must bind the SAME set of variables (SYNTAX.md: \"Both patterns must bind the same set of variables with compatible types\").  A variable bound in one alternative but not another cannot be given a consistent type in the branch body (OCaml reports this as `Orpat_vars`).\n\nFix: bind the same variables in every alternative, or use `_` for alternatives that should not bind.",
    },
    CodeEntry {
        code: "E106",
        title: "or-pattern binding type mismatch",
        category: ErrorCategory::Resolution,
        explain: "A variable bound by an or-pattern has INCOMPATIBLE types in different alternatives (OCaml reports this as `Or_pattern_type_clash`).  The branch body must be able to assume ONE type for each bound variable, so the alternatives' types must unify.\n\nFix: make the alternatives bind the variable at compatible types, or split the or-pattern into separate arms.",
    },
    CodeEntry {
        code: "E107",
        title: "contradictory where equality",
        category: ErrorCategory::Resolution,
        explain: "Two `where` equalities constrain the same type parameter to incompatible types (e.g. `where T == Int<32>, T == Bool`).  The constraints cannot hold simultaneously, so the specification is contradictory — the body cannot be checked under both, and any contract proofs would proceed under an incomplete axiom set.\n\nFix: remove the contradictory equality, or split into separate functions.",
    },
    CodeEntry {
        code: "E108",
        title: "strict mode must_handle violation",
        category: ErrorCategory::Resolution,
        explain: "In strict mode, a call to an `@must_handle` function must handle every marked error variant locally (via `catch`) before propagating.  Propagating a must_handle'd result via `?`, or leaving a marked variant uncaught, is an error in strict mode (a warning otherwise).\n\nFix: add explicit `catch` branches for each `@must_handle` variant, or add `@delegates_must_handle` to this function.",
    },
    CodeEntry {
        code: "E109",
        title: "read of a variable frozen by an active `&mut` borrow",
        category: ErrorCategory::Type,
        explain: "SYNTAX.md §References: an exclusive borrow (`&mut T`) freezes the original variable — neither readable nor writable — while it is live (committee ruling, 2026-08-05).  Reading the source place while the exclusive borrow is live is rejected.\n\nFix: read through the borrow itself (`*r`), or end the borrow's scope before reading the source.",
    },
    CodeEntry {
        code: "E110",
        title: "mutation of a variable frozen by an active borrow",
        category: ErrorCategory::Type,
        explain: "SYNTAX.md §References / §Reference Coercion: the source of an active borrow (`&mut` exclusive, `&ro` / `.freeze!()` read-only) is frozen against mutation while the borrow is live.\n\nFix: mutate through the borrow (for `&mut`), end the borrow's scope first, or use the explicit `&ro`/`.freeze!()` forms deliberately.",
    },
    CodeEntry {
        code: "E111",
        title: "read-only borrow requires a reference operand",
        category: ErrorCategory::Type,
        explain: "The `&ro` operator (and `.freeze!()`) freezes a `&mut T` reference into a `&T` view.  The operand must itself be a reference; borrowing a non-reference value with `&ro` is not meaningful.\n\nFix: apply `&ro` to a reference-typed operand (`&ro r`, `&ro r.f` where the field is a reference).",
    },
    CodeEntry {
        code: "E112",
        title: "overlapping exclusive borrows",
        category: ErrorCategory::Type,
        explain: "SYNTAX.md §References: `&mut T` is exclusive — while it is live, no other borrow (read-only or exclusive) of an overlapping place may exist.\n\nFix: end the first borrow's scope (its borrow variable's last use) before creating the second, or restructure the code to avoid the overlapping exclusive borrows.",
    },
    CodeEntry {
        code: "E113",
        title: "the function's CFG exceeds the point-encoding capacity",
        category: ErrorCategory::Type,
        explain: "The Polonius fact extraction encodes each CFG point as an integer (block << 36 | stmt << 16 | expr).  A function with more than 1,048,575 CFG blocks, more than 1,048,575 statements in a single block, or more than 65,535 expressions in a single statement exceeds this internal encoding capacity — an internal compiler limitation, not a program error.\n\nFix: split the function into smaller functions, or report the issue if the function is not unusually large.",
    },
    CodeEntry {
        code: "E114",
        title: "use of moved value",
        category: ErrorCategory::Type,
        explain: "SYNTAX.md §Move Semantics: after a move, the source is invalidated — a subsequent use is a compile-time error (affine: a non-Copy value is used at most once).\n\nFix: use the value before moving it, or re-initialize the source (assign a new value) before the later use.",
    },
    CodeEntry {
        code: "E115",
        title: "region subset not declared in signature",
        category: ErrorCategory::Type,
        explain: "The Polonius R9 (placeholder-subset rejection) rule: a subset relation between two placeholder (signature-region) origins derived inside the function body must be declared in the signature (`known_placeholder_subset`) — the caller needs the declared relationship to reason about the returned borrows.  An undeclared region subset is rejected.\n\nFix: declare the region relationship in the signature (e.g. `def f<'a, 'b>(...) -> ...` with the regions ordered so the subset holds), or restructure the function so the derived relationship is not required.",
    },
    CodeEntry {
        code: "E116",
        title: "cannot drop a value while its borrow is still live",
        category: ErrorCategory::Type,
        explain: "The Polonius drop rule (`evaluate_drop_errors`): a dropped value whose loan is still live at the drop point — the borrow outlives the value (rustc E0505).  The drop destroys the value while a borrow of it is still live, so the borrow would dangle.\n\nFix: end the borrow's scope (its borrow variable's last use) before the value is dropped, or reorder the code so the drop happens after the borrow is no longer needed.",
    },
    CodeEntry {
        code: "E117",
        title: "side effect in a @pure function",
        category: ErrorCategory::Resolution,
        explain: "A `@pure` function must have no side effects — transitively: it may only call functions/methods whose own `effect_of` labels are empty (no write, no allocation, no panic, no unsafe, no comptime, no mutable-global read, no input/output).  A forbidden effect anywhere in the call chain violates the annotation.\n\nFix: remove the offending call or read, or drop `@pure` from the function.",
    },
    CodeEntry {
        code: "W004",
        title: "must_handle error propagated via `?`",
        category: ErrorCategory::Generic,
        explain: "A call to an `@must_handle` function is propagated via `?`, which does not count as handling the marked error variants.  In strict mode this is an error (E108).\n\nFix: use `catch` to explicitly handle `@must_handle` variants before `?`, or add `@delegates_must_handle` to this function.",
    },
    CodeEntry {
        code: "W005",
        title: "must_handle variant not explicitly caught",
        category: ErrorCategory::Generic,
        explain: "A call to an `@must_handle` function does not have explicit `catch` branches for every marked error variant.  In strict mode this is an error (E108).\n\nFix: add explicit `catch` branches for each `@must_handle` variant.",
    },
    CodeEntry {
        code: "W006",
        title: "must_handle with wildcard-only catch",
        category: ErrorCategory::Generic,
        explain: "A bare `@must_handle` (no specific variants listed) is used with a `catch` that only has a wildcard arm.  A wildcard catch does not make the handled variants explicit.\n\nFix: use specific catch arms like `|Variant| { ... }` instead of `|_| { ... }`.",
    },
    CodeEntry {
        code: "W090",
        title: "use of deprecated function",
        category: ErrorCategory::Generic,
        explain: "A function annotated `@deprecated(\"reason\")` is called.  The annotation marks it for removal; new code should not rely on it.\n\nFix: migrate to the replacement function named in the deprecation reason.",
    },
    CodeEntry {
        code: "W091",
        title: "@cfg condition too complex for precise satisfiability checking",
        category: ErrorCategory::Generic,
        explain: "The SAT-based @cfg reachability checker skipped some constraints because the condition exceeded the complexity budget — the result may be a false positive (the code may be kept even when unsatisfiable).\n\nFix: simplify the condition so every constraint can be checked.",
    },
    CodeEntry {
        code: "W092",
        title: "conservative reachability treatment",
        category: ErrorCategory::Generic,
        explain: "A @cfg condition (or attribute-resolution check) could not be fully verified, so the code is conservatively treated as reachable rather than silently eliminated.\n\nFix: simplify the condition or resolve the incomplete check to suppress the warning.",
    },
    CodeEntry {
        code: "W113",
        title: "variable shadowing",
        category: ErrorCategory::Generic,
        explain: "A variable in the current scope has the same name as a variable in an\nouter scope, which shadows (hides) the outer one.\n\nThis is allowed in Posita, but may indicate a bug if the outer variable was\nstill needed.  Consider renaming one of the variables to avoid confusion.\n\nExample:\n  def f() {\n    set x = 1;\n    if true {\n      set x = 2;  // warning: shadows the outer `x`\n    }\n  }\n\nFix: use a different name for the inner variable, or remove the outer one.",
    },
    CodeEntry {
        code: "W114",
        title: "redundant &ro on immutable reference",
        category: ErrorCategory::Generic,
        explain: "`&ro`'s core purpose is the `&mut T` → `&T` coercion.  Applying it to an\nalready-immutable reference is harmless but redundant — the operand is\nalready `&T`.\n\nThis is allowed in Posita (a warning, not an error), but may indicate that\nthe `&ro` is unnecessary.  Consider removing it and using the plain `&T`\nreference directly.",
    },
    CodeEntry {
        code: "E091",
        title: "strict mode violation",
        category: ErrorCategory::Generic,
        explain: "In strict mode, @trusted functions must have @link_proof or\n@comptime_test evidence linking them to a formal proof.\n\nWithout such evidence, the compiler cannot verify that the function\nmeets its safety guarantees.  Add @link_proof or use @comptime_test\nto provide the required proof linkage.",
    },
    CodeEntry {
        code: "E080",
        title: "comptime evaluation error",
        category: ErrorCategory::Generic,
        explain: "A comptime block or comptime function failed during evaluation.\n\nThe evaluator's error message is attached and is the specific cause.\nCommon causes include:\n- A comptime block containing an item declaration (`type`, `def`, `impl`,\n  `trait`, ...) — comptime blocks only allow expressions, variable\n  definitions, and assignments.  Use a `generate` block for\n  declaration-level code generation.\n- A comptime function or block referencing a value that is not computable\n  at compile time.\nFix the comptime code so it evaluates successfully.",
    },
    CodeEntry {
        code: "E081",
        title: "comptime sandbox violation",
        category: ErrorCategory::Generic,
        explain: "A comptime block attempted to call a @trusted or @io function, which\nis prohibited because comptime code is sandboxed.\n\nComptime blocks can only call comptime functions (declared with\n`comptime def`) or safe built-in operations.  To call a @trusted\nfunction at compile time, use `comptime @trusted { ... }` instead.",
    },
    CodeEntry {
        code: "E082",
        title: "comptime block accesses mutable or unknown state",
        category: ErrorCategory::Generic,
        explain: "A comptime block cannot touch mutable state: assigning to a mutable global, capturing a mutable variable, or naming an unknown variable in the capture list are all rejected — comptime evaluation is sandboxed to compile-time-known values.\n\nFix: only capture comptime-known values, or read the state outside the comptime block.",
    },
    CodeEntry {
        code: "E090",
        title: "invalid @comptime_test function",
        category: ErrorCategory::Resolution,
        explain: "A `@comptime_test` function must have no parameters, must have a body, and must evaluate successfully — it is executed at compile time with no arguments to demonstrate the property.\n\nFix: remove the parameters, provide a body, or fix the failing assertion.",
    },
    CodeEntry {
        code: "E092",
        title: "cfg condition unreachable in strict mode",
        category: ErrorCategory::Generic,
        explain: "In strict mode, the SAT-based cfg reachability checker determined that\na `@cfg(condition)` is unsatisfiable under any target configuration.\n\nThis likely indicates contradictory conditions (e.g.\n`all(target_os == \"linux\", target_os == \"windows\")`).\n\nFix the @cfg condition to be satisfiable under at least one target.",
    },
    CodeEntry {
        code: "E093",
        title: "isolate block accesses external mutable state",
        category: ErrorCategory::Generic,
        explain: "An isolate block must not access or mutate external mutable state\n(SYNTAX.md §Task Isolation — \"does not access any external mutable\nstate\").  Reading or writing a mutable global, or assigning to a\ncaptured outer mutable variable, inside an isolate block is rejected.\n\nDeclare the variable inside the block if it is meant to be internal\nstate, or restructure the code so the isolate block only touches its\nown state.",
    },
    CodeEntry {
        code: "E094",
        title: "use of experimental trait",
        category: ErrorCategory::Trait,
        explain: "The trait is marked experimental — it is not part of the stable language surface and requires the experimental feature flag.\n\nFix: enable experimental features (`--enable-experimental`), or avoid the trait.",
    },
    CodeEntry {
        code: "E099",
        title: "`@compile_error` triggered",
        category: ErrorCategory::Generic,
        explain: "A `@compile_error(\"message\")` expression was evaluated during compilation — it halts compilation unconditionally with the attached message.  This is the intended mechanism for `#error`-style compile-time diagnostics.\n\nFix: remove or guard the `@compile_error` invocation.",
    },
    CodeEntry {
        code: "E601",
        title: "invalid safe cast",
        category: ErrorCategory::Type,
        explain: "A `as` (safe cast) between incompatible types: a reference type requires an explicit dereference or an unsafe `as!` bitcast; safe casts are only allowed between numeric and boolean types; casting a reference to an integer is not yet supported; an unsafe cast requires compatible types (numeric↔numeric, ref↔ptr, ptr↔ptr).\n\nFix: dereference the value first (`*expr as TargetType`), use `as!` for an explicit bitcast, or cast between compatible types.",
    },
];

/// Look up a code string in the table.
pub(crate) fn lookup(code: &str) -> Option<&'static CodeEntry> {
    CODE_TABLE.iter().find(|e| e.code == code)
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
