# English source messages (rustc-style Fluent). Slugs are stable identifiers: never translated, renamed only with a codebase-wide codemod.

comptime-error = comptime error: { $e }
trait-solver-error = trait solver error: { $msg }
type-mismatch = type mismatch: expected `{ $expected }`, found `{ $found }`
argument-type-mismatch = argument type mismatch in call to `{ $callee }`: parameter `{ $param }` expected `{ $expected }`, found `{ $found }`
contract-non-bool = `{ $clause }` clause must be boolean, found `{ $found }`
impl-missing-method = impl of `{ $trait }` is missing method `{ $method }`
duplicate-definition = duplicate definition of `{ $name }`
field-not-found = no field `{ $field }` found on type `{ $type }`
main-not-found = `main` function not found in crate
set-no-destructuring = `set` does not support pattern destructuring; use `let` instead
let-requires-initializer = `let` requires an explicit initializer; it cannot rely on a type's default value
no-default-value = type has no default value and no initializer provided
shadowing-definition = shadowing definition of `{ $name }`
missing-method = impl missing method `{ $method }` required by trait `{ $trait }`
impl-trait-methods = every trait method must be implemented — add a `def` for it in this impl block
unknown-error-code = unknown error code: `{ $code }`
did-you-mean = did you mean `{ $candidates }`?
run-explain-hint = run `ponent explain <CODE>` with a valid error code
available-error-codes = Available error codes:

summary-errors =
    { $count ->
        [one]   aborting due to previous error
       *[other] aborting due to { $count } previous errors
    }

summary-mixed-errors =
    { $count ->
        [one]   aborting due to { $count } previous error
       *[other] aborting due to { $count } previous errors
    }

summary-warnings =
    { $count ->
        [one]   1 warning emitted
       *[other] { $count } warnings emitted
    }

summary-files =
    { $count ->
        [one]   in 1 file
       *[other] in { $count } files
    }

# Diagnostic level labels (P3-23: i18n for DiagnosticLevel::label)
level-fatal = fatal
level-error = error
level-warning = warning
level-help = help
level-note = note
level-info = info

# Call chain entry kinds (P3-23: i18n for ChainEntryKind::as_str)
chain-call = called from here
chain-inlined = inlined here
chain-reference = referenced here
chain-defines = defined here

# Comptime evaluation reasons (P3-23: i18n for ComptimeReason::as_str)
comptime-reason-block = comptime {"{"} ... {"}"} block
comptime-reason-fndef = comptime def function body
comptime-reason-fncall = comptime function call
comptime-reason-test = @comptime_test function
comptime-reason-assert = assert!()
comptime-reason-typeinfo = @typeInfo!()
comptime-reason-layoutof = layout_of!()

# Error categories (P3-23: i18n for ErrorCategory::label)
category-parse = Parse Error
category-resolution = Resolution Error
category-type = Type Error
category-contract = Contract Error
category-trait = Trait Error
category-inference = Inference Error
category-internal = Internal Error
category-generic = Error

# Humanizer labels (P3-24: i18n for DiagnosticKind::labels)
label-expected = expected { $expected }
label-field-not-found = field not found
label-expected-found = expected `{ $expected }`, found `{ $found }`
label-declared-here = `{ $expected }` declared here
label-not-in-scope = not found in this scope
label-duplicate-definition = duplicate definition
label-first-defined-here = first defined here
label-expected-bool = expected bool
label-method-missing = method missing here
label-required-by-trait = required by trait declaration here
label-comptime-eval-error = comptime evaluation error

# Humanizer help (P3-24: i18n for DiagnosticKind::help)
help-cast-or-change = try using `as` to cast, or change the expression type

# Explain error code / suggestions (P3-23: i18n for explain_error_code/suggest_code)
help-valid-codes = valid error codes include { $codes }
run-explain-hint-detail = run `ponent explain <CODE>` with a valid error code
run-explain-hint-example = run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)
run-explain-detail = Run `ponent explain <CODE>` for details on a specific error code.
unknown-error = unknown error
