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
    CodeEntry { code: "E001", title: "expected token", category: ErrorCategory::Parse, explain: "A specific token was expected at the current parse position but not found.\n\nThis typically occurs due to:\n  - Missing closing delimiters: `)`, `}`, `]`\n  - Missing semicolons at end of statements\n  - Incomplete syntax in type or expression\n\nExample of invalid code:\n  def foo(x: Int<32 {\n    return x;\n  }\n\nFix: add the missing token at the indicated position." },
    CodeEntry { code: "E002", title: "unexpected end of input", category: ErrorCategory::Parse, explain: "The parser reached the end of the input while still expecting more tokens.\n\nThis usually means the code is incomplete." },
    CodeEntry { code: "E003", title: "unexpected token", category: ErrorCategory::Parse, explain: "An unexpected token was encountered at the current parse position.\n\nThis typically indicates a syntax error such as missing punctuation, extra characters, or incorrect ordering of keywords.\n\nExample:\n  def foo(x: Int<32) { return; } }  // extra `}`\n\nFix: review the syntax at the indicated position and remove or correct the unexpected token." },
    CodeEntry { code: "E004", title: "parse error", category: ErrorCategory::Parse, explain: "A general parse error occurred." },
    CodeEntry { code: "E005", title: "expected identifier", category: ErrorCategory::Parse, explain: "An identifier (name) was expected but not found.\n\nIdentifiers are used for variable names, function names, type names, etc.\n\nExample:\n  def 123() { }  // error: expected identifier, found number\n\nFix: use a valid identifier name instead." },
    CodeEntry { code: "E006", title: "recursion limit exceeded", category: ErrorCategory::Parse, explain: "The parser exceeded the recursion limit.\n\nThis may indicate deeply nested expressions or a bug in the parser." },
    CodeEntry { code: "E007", title: "expected expression", category: ErrorCategory::Parse, explain: "An expression was expected at the current position but not found.\n\nExpressions are values, variables, function calls, operators, etc. This typically occurs after an operator or assignment that expects a value.\n\nExample:\n  set x = ;  // error: expected expression after `=`\n\nFix: provide a valid expression at the indicated position." },
    CodeEntry { code: "E008", title: "integer overflow", category: ErrorCategory::Parse, explain: "Integer literal overflow." },
    CodeEntry { code: "E009", title: "invalid character literal", category: ErrorCategory::Parse, explain: "Invalid character literal." },

    CodeEntry { code: "E010", title: "no such field", category: ErrorCategory::Resolution, explain: "A field access refers to a field that does not exist on the given type." },
    CodeEntry { code: "E011", title: "type not found", category: ErrorCategory::Resolution, explain: "A type name could not be resolved in the current scope." },
    CodeEntry { code: "E012", title: "name not found", category: ErrorCategory::Resolution, explain: "A name could not be resolved in the current scope." },
    CodeEntry { code: "E013", title: "undefined type", category: ErrorCategory::Resolution, explain: "The type has not been defined in the current scope." },
    CodeEntry { code: "E014", title: "generic args on non-generic", category: ErrorCategory::Resolution, explain: "Generic type arguments were provided for a non-generic type." },
    CodeEntry { code: "E015", title: "cannot resolve import", category: ErrorCategory::Resolution, explain: "An import could not be resolved." },
    CodeEntry { code: "E016", title: "no default value", category: ErrorCategory::Resolution, explain: "A type has no default value and no initializer was provided." },
    CodeEntry { code: "E017", title: "array size not constant", category: ErrorCategory::Resolution, explain: "Array size must be a constant expression." },
    CodeEntry { code: "E018", title: "unexpected top-level item", category: ErrorCategory::Resolution, explain: "`set` and `let` statements are not allowed at the top level; only declarations (`def`, `type`, `trait`, `import`, `impl`, `constraint`, `comptime`, `extern`, `edition`) are permitted here." },
    CodeEntry { code: "E019", title: "duplicate definition", category: ErrorCategory::Resolution, explain: "A variable, function, or type has been defined more than once in the same scope.\n\nThis is not allowed because the second definition would shadow the first without\nany way to refer to the original binding.\n\nExample of invalid code:\n  set x = 1;\n  set x = 2;  // error: duplicate definition of `x`\n\nFix: use a different name for the second definition, or remove the first one." },

    CodeEntry { code: "E020", title: "contract condition must be boolean", category: ErrorCategory::Contract, explain: "A contract condition (`requires` or `invariant`) must evaluate to a boolean\nvalue (`Bool`), but a non-boolean expression was provided.\n\nExample:\n  def foo(x: Int<32>) -> Int<32>\n    requires x + 1  // error: `x + 1` is Int<32>, not Bool\n\nFix: ensure the condition evaluates to a boolean value." },
    CodeEntry { code: "E021", title: "ensures clause must be boolean", category: ErrorCategory::Contract, explain: "An `ensures` clause must evaluate to a boolean value (`Bool`)." },
    CodeEntry { code: "E022", title: "decreases expression must be integer", category: ErrorCategory::Contract, explain: "A `decreases`/`terminates` expression must be an integer type." },
    CodeEntry { code: "E023", title: "contract boolean at return", category: ErrorCategory::Contract, explain: "A contract condition must be boolean." },

    CodeEntry { code: "E030", title: "type mismatch", category: ErrorCategory::Type, explain: "The type of an expression does not match the expected type.\n\nThis error occurs when a value is assigned to a variable, passed as an argument,\nor returned from a function with a different type than expected.\n\nExample:\n  def foo() -> Int<32> {\n    return \"hello\";  // error: expected Int<32>, found &Str\n  }\n\nFix: ensure the expression has the correct type, or add an explicit cast." },
    CodeEntry { code: "E031", title: "kind mismatch", category: ErrorCategory::Type, explain: "The kind of a type does not match the expected kind.\n\nThis occurs when a type inference variable has a kind constraint (e.g. Integer,\nBool, Float) that conflicts with the resolved concrete type.  For example, using\na string variable where an integer is expected.\n\nExample:\n  set j = \"hello\";\n  set i = j + 1;  // error: expected integer type, found &Str\n\nFix: ensure the value has the correct type kind for the operation." },
    CodeEntry { code: "E032", title: "operator type error", category: ErrorCategory::Type, explain: "An operator is applied to incompatible operand types.\n\nThis occurs when binary or unary operators are used with types that do not\nsupport the operation.  For example, adding a string to an integer.\n\nExample:\n  set j = \"hello\";\n  set i = j + 1;  // error: cannot add &Str and Int\n\nFix: ensure both operands have compatible types for the operator." },
    CodeEntry { code: "E033", title: "cannot infer type", category: ErrorCategory::Type, explain: "The type of an expression could not be inferred." },
    CodeEntry { code: "E034", title: "infinite type", category: ErrorCategory::Type, explain: "Infinite type detected during unification." },
    CodeEntry { code: "E035", title: "type annotation needed", category: ErrorCategory::Type, explain: "A type annotation is needed for this expression." },
    CodeEntry { code: "E036", title: "return type mismatch", category: ErrorCategory::Type, explain: "The return value type does not match the function's declared return type.\n\nExample:\n  def foo() -> Int<32> {\n    return true;  // error: expected Int<32>, found Bool\n  }\n\nFix: ensure the return value has the correct type, or change the function's\nreturn type annotation." },
    CodeEntry { code: "E037", title: "argument type mismatch", category: ErrorCategory::Type, explain: "A function argument's type does not match the parameter type.\n\nExample:\n  def foo(x: Int<32>) { }\n  def main() { foo(true); }  // error: expected Int<32>, found Bool\n\nFix: pass an argument of the correct type, or change the parameter type." },
    CodeEntry { code: "E038", title: "condition must be boolean", category: ErrorCategory::Type, explain: "A condition expression (in `if`, `while`, `requires`, `ensures`, etc.) must\nbe of type `Bool`, but a non-boolean expression was provided.\n\nExample:\n  if 42 { }  // error: condition must be Bool, found Int<32>\n\nFix: use a boolean expression as the condition." },
    CodeEntry { code: "E039", title: "index must be integer", category: ErrorCategory::Type, explain: "An index expression must be an integer type, but a non-integer type was\nprovided.\n\nExample:\n  arr[\"hello\"]  // error: index must be integer, found &Str\n\nFix: use an integer expression as the index." },

    CodeEntry { code: "E040", title: "trait not found", category: ErrorCategory::Trait, explain: "The specified trait could not be found." },
    CodeEntry { code: "E041", title: "trait not implemented", category: ErrorCategory::Trait, explain: "A required trait is not implemented for the given type." },
    CodeEntry { code: "E042", title: "orphan impl", category: ErrorCategory::Trait, explain: "An `impl` block violates the orphan rule: the type and trait must be\ndefined in the current crate, or the trait must be from the current crate.\n\nThis restriction prevents conflicting implementations across crates." },
    CodeEntry { code: "E043", title: "conflicting impl", category: ErrorCategory::Trait, explain: "Conflicting implementations of a trait for the same type." },

    CodeEntry { code: "E050", title: "inference error", category: ErrorCategory::Inference, explain: "A type inference error occurred." },
    CodeEntry { code: "E051", title: "cannot infer type", category: ErrorCategory::Inference, explain: "The type of an expression could not be inferred. Try adding a type annotation." },

    CodeEntry { code: "E052", title: "interrupt must satisfy @no_alloc", category: ErrorCategory::Resolution, explain: "An @interrupt handler must satisfy the @no_alloc capability.\n\nInterrupt handlers run in a constrained context where memory allocation\nis not available. The function must be annotated with @no_alloc or be\nin a scope where @no_alloc is already in effect.\n\nFix: add `@no_alloc` to this function (or remove it if redundant with\n`@no_panic`)." },
    CodeEntry { code: "E053", title: "interrupt must satisfy @no_panic", category: ErrorCategory::Resolution, explain: "An @interrupt handler must satisfy the @no_panic capability.\n\nInterrupt handlers must not panic because there is no unwind mechanism\navailable. The function must be annotated with @no_panic.\n\nFix: add `@no_panic` to this function." },
    CodeEntry { code: "E054", title: "interrupt cannot have @alloc", category: ErrorCategory::Resolution, explain: "An @interrupt handler cannot have the @alloc capability.\n\nInterrupt handlers must not allocate memory. The @alloc annotation is\nincompatible with @interrupt.\n\nFix: remove the `@alloc` annotation from this function." },
    CodeEntry { code: "E055", title: "interrupt cannot have @io", category: ErrorCategory::Resolution, explain: "An @interrupt handler cannot have the @io capability.\n\nInterrupt handlers must not perform I/O operations. The @io annotation\nis incompatible with @interrupt.\n\nFix: remove the `@io` annotation from this function." },

    CodeEntry { code: "E060", title: "internal compiler error", category: ErrorCategory::Internal, explain: "An internal compiler error occurred. This is a bug in the compiler.\n\nPlease report this error at https://github.com/posita-lang/ponent/issues." },
    CodeEntry { code: "E061", title: "unreachable code", category: ErrorCategory::Internal, explain: "The compiler reached an unreachable code path. This is a bug." },

    CodeEntry { code: "E062", title: "main function not found", category: ErrorCategory::Resolution, explain: "The crate does not define a `main` function, which is required for executable output.\n\nEvery executable crate must have a `main` function that serves as the entry\npoint for the program.  The `main` function takes no arguments and returns\nan integer type (e.g. `Int<32>`).\n\nFix: add a `def main() { ... }` function to the crate." },

    CodeEntry { code: "E101", title: "trait impl missing method", category: ErrorCategory::Trait, explain: "A trait implementation is missing a required method.\n\nEvery trait method must be implemented in the impl block. This error\noccurs when a method declared in the trait is not defined in the impl.\n\nFix: add a `def` for the missing method in this impl block." },
    CodeEntry { code: "E102", title: "orphan impl or bare type variable", category: ErrorCategory::Trait, explain: "A trait impl could not be registered: either it violates the orphan rule\n(a type/trait from another crate), or a bare type variable appeared\nwithout sufficient context to determine its kind.\n\nFix: ensure the impl follows the orphan rule, or provide a type\nannotation for the bare type variable." },
    CodeEntry { code: "E103", title: "trait impl wrong parameter count", category: ErrorCategory::Trait, explain: "A trait method implementation has a different number of parameters than\nthe trait declaration.\n\nEvery method in a trait impl must have the same number of parameters as\nthe corresponding trait method signature.\n\nFix: adjust the parameter count to match the trait declaration." },

    CodeEntry { code: "W113", title: "variable shadowing", category: ErrorCategory::Generic, explain: "A variable in the current scope has the same name as a variable in an\nouter scope, which shadows (hides) the outer one.\n\nThis is allowed in Posita, but may indicate a bug if the outer variable was\nstill needed.  Consider renaming one of the variables to avoid confusion.\n\nExample:\n  def f() {\n    set x = 1;\n    if true {\n      set x = 2;  // warning: shadows the outer `x`\n    }\n  }\n\nFix: use a different name for the inner variable, or remove the outer one." },
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