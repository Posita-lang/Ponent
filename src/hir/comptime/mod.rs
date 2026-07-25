/// Compile-time evaluation engine for Posita.
///
/// ## Architecture
/// Comptime evaluation is a tree-walking interpreter over HIR (not AST).
/// It runs as a separate pass within the TypeChecker, after name resolution
/// and type checking.  The evaluator has its own variable scope, step/memory
/// limits, and sandbox isolation.
///
/// ## Capabilities (implemented)
/// - Arithmetic: Int, Float, String (binary ops, comparisons)
/// - Control flow: `if`, `while`, `if let`, `match`
/// - Variables: `set`, assignment, variable scoping
/// - Functions: `comptime def` with `!` call marker, forward references
/// - Type reflection: `@typeInfo!(T)`, `layout_of!(T)`
/// - Enum/struct construction: `EnumLit`, `StructLit`, `Tuple`, `Array`
/// - Type casting: `Cast` (Int ↔ Float)
/// - Pointers: `&x`, `*ptr`, `ptr[i]`, mutable references
/// - Aggregate values: struct, tuple, array as `ComptimeValue::Aggregate`
/// - Sandbox: step limit, memory limit, `@io`/`@trusted` blocking
/// - Builtins: `assert!()`, `@compile_error!("msg")`
///
/// ## Capabilities (not yet implemented)
/// - Match on patterns with guards (basic match is implemented)
/// - Symbol table integration for const values
/// - `@compileLog` - debugging output
/// - Closure evaluation in comptime
/// - `AttrAccess` (`T'default`)
/// - `Try`/`Catch`/`Await`/`LeaveWith`
/// - `PolyBox`/`PolyUnbox`
/// - `UnsafeBlock` in comptime
///
/// ## Design notes
/// - The evaluator holds `&'a mut TypeContext` to support type creation during
///   comptime (e.g. `layout_of!` needs to allocate types).  The old
///   `ctx.factory()` approach via `RefCell` is no longer the primary path —
///   the evaluator allocates types directly through `ctx.alloc()`.
/// - Error context backtracking records the chain of comptime blocks/functions
///   via `CtxFrame::comptime_reason` (like Zig's `BlockComptimeReason`).
pub mod error;
pub mod eval;
pub mod value;

pub use error::ComptimeError;
pub use eval::ComptimeEvalContext;
pub use value::ComptimeValue;

#[cfg(test)]
mod tests;
