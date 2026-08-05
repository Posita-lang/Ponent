use crate::diagnostics::DiagCtxt;
use crate::diagnostics::kind::{ComptimeErrorKind, ComptimeReason, DiagnosticKind};
use crate::hir::hir::{HirExpr, HirMatchArm, HirPattern, HirProgram, HirStmt};
use crate::hir::symbol::SymbolTable;
use crate::hir::types::{TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;

use super::error::ComptimeError;
use super::value::{ComptimeValue, SlotId};

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

/// A registered comptime function: (parameter_names, body_statements).
/// The body is wrapped in `Arc` to avoid deep-cloning the AST on every call.
type ComptimeFn = (Vec<Symbol>, Arc<[HirStmt]>);

/// Compute the representable range for a signed integer of `bits` width.
fn signed_range(bits: u8) -> (i128, i128) {
    if bits == 0 {
        (0, 0)
    } else if bits >= 127 {
        (i128::MIN, i128::MAX)
    } else {
        let max = (1i128 << (bits - 1)) - 1;
        let min = -(1i128 << (bits - 1));
        (min, max)
    }
}

/// Compute the representable range for an unsigned integer of `bits` width.
fn unsigned_range(bits: u8) -> (i128, i128) {
    if bits >= 128 {
        (0, i128::MAX)
    } else {
        let max = (1i128 << bits) - 1;
        (0, max)
    }
}

/// Apply the overflow policy to `result` given the type's representable range.
/// Returns the corrected value, or `Overflow` error if the policy is `Trap`.
fn apply_overflow_policy(
    result: i128,
    min: i128,
    max: i128,
    policy: &crate::ast::OverflowPolicy,
) -> Result<i128, ComptimeError> {
    if result >= min && result <= max {
        return Ok(result);
    }
    match policy {
        crate::ast::OverflowPolicy::Wrap => {
            // Two's complement wrapping within [min, max].
            let range = max.wrapping_sub(min).wrapping_add(1);
            if range == 0 {
                return Ok(result);
            }
            Ok(result
                .wrapping_sub(min)
                .wrapping_rem_euclid(range)
                .wrapping_add(min))
        }
        crate::ast::OverflowPolicy::Saturate => {
            if result < min {
                Ok(min)
            } else {
                Ok(max)
            }
        }
        crate::ast::OverflowPolicy::Trap => Err(ComptimeError::Overflow),
    }
}

/// Check `result` against the type's bit width and overflow policy.
/// Returns the (possibly adjusted) value, or `Overflow` if the policy is `Trap`.
fn check_range(result: i128, ty: TypeId, ctx: &TypeContext) -> Result<i128, ComptimeError> {
    match ctx.get(ty) {
        crate::hir::types::TypeData::Int {
            bits,
            overflow_policy,
            ..
        } => {
            let (min, max) = signed_range(*bits);
            apply_overflow_policy(result, min, max, overflow_policy)
        }
        crate::hir::types::TypeData::UInt {
            bits,
            overflow_policy,
            ..
        } => {
            let (min, max) = unsigned_range(*bits);
            apply_overflow_policy(result, min, max, overflow_policy)
        }
        // During comptime evaluation within type checking, the result type
        // may still be an un-resolved InferVar (e.g. TypeVariableKind::Numeric
        // from a BinaryOp).  Skip range checking in that case — the type
        // checker will resolve it later and catch any mismatches.
        crate::hir::types::TypeData::InferVar { .. } => Ok(result),
        _ => Err(ComptimeError::Internal(format!(
            "check_range called on non-integer type: {:?}",
            ctx.get(ty)
        ))),
    }
}

/// Static payload field name to avoid allocation on every enum construction.
pub(crate) const PAYLOAD_FIELD: &str = "payload";

/// Lazy-growing cache for tuple field names (e.g. `_0`, `_1`, …, `_N`).
/// Avoids the `format!` allocation for indices ≥ 64 by memoizing each name
/// on first access.  Thread-safe via `OnceLock<Mutex<Vec<Symbol>>>`; the
/// lock is held only on cache misses, so repeated lookups of the same index
/// are lock-free reads after the first miss.
fn tuple_field_name(i: usize) -> Symbol {
    // Fast path: pre-computed for indices 0–63 (common case for small tuples).
    const PREFIX: [&str; 64] = [
        "_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "_12", "_13",
        "_14", "_15", "_16", "_17", "_18", "_19", "_20", "_21", "_22", "_23", "_24", "_25", "_26",
        "_27", "_28", "_29", "_30", "_31", "_32", "_33", "_34", "_35", "_36", "_37", "_38", "_39",
        "_40", "_41", "_42", "_43", "_44", "_45", "_46", "_47", "_48", "_49", "_50", "_51", "_52",
        "_53", "_54", "_55", "_56", "_57", "_58", "_59", "_60", "_61", "_62", "_63",
    ];
    if i < 64 {
        return Symbol::intern(PREFIX[i]);
    }
    // Slow path: lazily-grown cache for large indices.
    static CACHE: OnceLock<Mutex<Vec<Symbol>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap();
    if i < guard.len() {
        return guard[i];
    }
    let name = Symbol::intern(&format!("_{}", i));
    guard.push(name);
    name
}

/// Lazy-growing cache for array field names (e.g. `[0]`, `[1]`, …, `[N]`).
/// Same design as `tuple_field_name`.
fn array_field_name(i: usize) -> Symbol {
    // Fast path: pre-computed for indices 0–63.
    const PREFIX: [&str; 64] = [
        "[0]", "[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[11]",
        "[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]", "[19]", "[20]", "[21]", "[22]",
        "[23]", "[24]", "[25]", "[26]", "[27]", "[28]", "[29]", "[30]", "[31]", "[32]", "[33]",
        "[34]", "[35]", "[36]", "[37]", "[38]", "[39]", "[40]", "[41]", "[42]", "[43]", "[44]",
        "[45]", "[46]", "[47]", "[48]", "[49]", "[50]", "[51]", "[52]", "[53]", "[54]", "[55]",
        "[56]", "[57]", "[58]", "[59]", "[60]", "[61]", "[62]", "[63]",
    ];
    if i < 64 {
        return Symbol::intern(PREFIX[i]);
    }
    // Slow path: lazily-grown cache for large indices.
    static CACHE: OnceLock<Mutex<Vec<Symbol>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap();
    if i < guard.len() {
        return guard[i];
    }
    let name = Symbol::intern(&format!("[{}]", i));
    guard.push(name);
    name
}

/// Check if a block's statements are all pure expressions (no side effects).
/// Used to refine `is_pure_expr` for `Block` and `If` branches.
fn is_pure_block(stmts: &[HirStmt]) -> bool {
    stmts.iter().all(|stmt| match stmt {
        HirStmt::Expression(expr) => is_pure_expr(expr),
        // VariableDef, Assign, While, and other statement kinds have side effects.
        _ => false,
    })
}

/// Check if a HirExpr is a pure computation with no side effects.
/// Pure expressions can be evaluated and discarded without affecting state.
fn is_pure_expr(expr: &HirExpr) -> bool {
    match expr {
        HirExpr::Literal(..) => true,
        HirExpr::BinaryOp {
            op, left, right, ..
        } => {
            // Division and modulo can trigger DivisionByZero comptime errors,
            // which is a side effect — marking them impure prevents the
            // misleading "this computation has no effect" warning when a
            // division-by-zero error follows immediately after.
            // Also recursively check sub-expressions so that e.g.
            // `(1 / 0) + 1` is not falsely marked as pure.
            !matches!(op, crate::ast::BinOp::Div | crate::ast::BinOp::Rem)
                && is_pure_expr(left)
                && is_pure_expr(right)
        }
        HirExpr::UnaryOp { op, expr, .. } => {
            // Ref/RefMut, Deref, etc. have side effects (capturing/accessing state).
            matches!(op, crate::ast::UnaryOp::Not | crate::ast::UnaryOp::Neg) && is_pure_expr(expr)
        }
        HirExpr::Block(stmts, ..) => is_pure_block(stmts),
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            is_pure_expr(cond)
                && is_pure_block(then_branch)
                && else_branch.as_ref().map_or(true, |b| is_pure_block(b))
        }
        HirExpr::Tuple(..) => true,
        HirExpr::Array(..) => true,
        HirExpr::StructLit { .. } => true,
        HirExpr::EnumLit { .. } => true,
        HirExpr::TypeInfo(..) => true,
        HirExpr::LayoutOf(..) => true,
        HirExpr::CompileError(..) => true,
        HirExpr::TypeAnnotated { expr, .. } => is_pure_expr(expr),
        HirExpr::Cast { .. } => true,
        _ => false,
    }
}

/// Evaluation context for comptime blocks.
/// Tracks step budget and provides expression evaluation.
pub struct ComptimeEvalContext<'a> {
    ctx: &'a mut TypeContext,
    diag: &'a mut DiagCtxt,
    steps: usize,
    step_limit: usize,
    /// Estimated memory used by comptime values (in bytes).
    pub(crate) memory_used: usize,
    /// Maximum memory allowed for comptime evaluation (in bytes).
    memory_limit: usize,
    /// The HIR program, used to lookup comptime function definitions.
    /// Optional because the HirProgram is not available during type checking
    /// (it is the output of check_program).  Will be populated when comptime
    /// function calls are implemented.
    hir_program: Option<&'a HirProgram>,
    /// The symbol table, used for name resolution.
    symbols: &'a SymbolTable,
    /// Variable storage: each comptime variable gets a unique `SlotId`.
    /// The slot ID stays the same for the lifetime of the binding, so pointers
    /// (which hold `SlotId`) are immune to variable shadowing.
    pub variables: HashMap<SlotId, ComptimeValue>,
    /// Maps variable name → current slot ID for name-based lookups (e.g. `Ident`).
    /// When a `VariableDef` shadows an outer variable, this is updated to point
    /// to the new slot; the old slot remains in `variables` until the scope exits.
    pub cur_slot: HashMap<Symbol, SlotId>,
    /// Monotonic counter for allocating unique `SlotId` values.
    next_slot: u32,
    /// Registry of comptime functions: name → (param_names, body).
    /// Populated by the checker as it encounters comptime function definitions.
    fn_registry: HashMap<Symbol, ComptimeFn>,
    /// Call stack for comptime traceback.
    /// Each entry is (function_name, reason, span).
    call_stack: Vec<(Symbol, ComptimeReason, crate::ast::Span)>,
    /// Outer context traceback from the checker's region_tree.
    /// Populated at creation time with comptime block/fn context.
    outer_traceback: Vec<(ComptimeReason, crate::ast::Span)>,
    /// Source text for converting byte offsets to line:column in tracebacks.
    source: Option<&'a str>,
    /// Whether the current comptime block is `@trusted`, granting access to
    /// `@trusted` functions and `unsafe` operations during comptime evaluation.
    allow_trusted: bool,
    /// When set, records the name → original `SlotId` of any variable that is
    /// shadowed via `VariableDef` (not `Assign`) inside a scoped block
    /// (e.g. while body).  After the block exits, `cur_slot` is restored to
    /// point to the original slot ID for each shadowed name, so that subsequent
    /// name-based lookups (e.g. `i = i + 1`) find the outer binding again.
    /// Using a HashMap means only the slot IDs of actually-shadowed variables
    /// are stored — O(k) instead of O(n) per iteration.
    scope_shadows: Vec<HashMap<Symbol, SlotId>>,
}

impl<'a> ComptimeEvalContext<'a> {
    pub fn new(ctx: &'a mut TypeContext, symbols: &'a SymbolTable, diag: &'a mut DiagCtxt) -> Self {
        Self::new_with_source(ctx, symbols, diag, Vec::new(), None)
    }

    pub fn new_with_source(
        ctx: &'a mut TypeContext,
        symbols: &'a SymbolTable,
        diag: &'a mut DiagCtxt,
        outer_traceback: Vec<(ComptimeReason, crate::ast::Span)>,
        source: Option<&'a str>,
    ) -> Self {
        ComptimeEvalContext {
            ctx,
            diag,
            symbols,
            hir_program: None,
            steps: 0,
            step_limit: 10_000,
            memory_used: 0,
            memory_limit: 10 * 1024 * 1024,
            variables: HashMap::new(),
            cur_slot: HashMap::new(),
            next_slot: 0,
            fn_registry: HashMap::new(),
            call_stack: Vec::new(),
            outer_traceback,
            source,
            allow_trusted: false,
            scope_shadows: Vec::new(),
        }
    }

    /// Set whether the current comptime block is `@trusted`.
    /// When `true`, the evaluator allows calls to `@trusted` functions
    /// and `unsafe` operations during comptime evaluation.
    pub fn set_trusted(&mut self, trusted: bool) {
        self.allow_trusted = trusted;
    }

    /// Register a comptime function so it can be called from within comptime blocks.
    pub fn register_fn(&mut self, name: Symbol, params: Vec<Symbol>, body: Vec<HirStmt>) {
        self.fn_registry.insert(name, (params, Arc::from(body)));
    }

    /// Set a custom step limit (for testing).
    pub fn set_step_limit(&mut self, limit: usize) {
        self.step_limit = limit;
    }

    /// Set a custom memory limit in bytes (for testing).
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.memory_limit = limit;
    }

    /// Allocate a new unique `SlotId` for a variable.
    pub fn allocate_slot(&mut self) -> SlotId {
        let id = self.next_slot;
        self.next_slot = self.next_slot.saturating_add(1);
        SlotId(id)
    }

    /// Remove all variables whose slots were created at or after
    /// `next_slot_at_entry`, adjusting `memory_used` accordingly.
    /// Since `SlotId` is monotonic, this is O(m) where m = number of
    /// new slots — the set of "new" slots since that point is exactly
    /// `[next_slot_at_entry, self.next_slot)`.
    fn remove_new_slots_since(&mut self, next_slot_at_entry: u32) {
        for slot_id in next_slot_at_entry..self.next_slot {
            let slot = SlotId(slot_id);
            if let Some(val) = self.variables.remove(&slot) {
                self.memory_used = self.memory_used.saturating_sub(val.memory_size());
            }
        }
    }

    /// Emit a comptime diagnostic via the diagnostic context.
    /// Converts the error to a structured DiagnosticKind::Comptime.
    fn emit_comptime_error(&mut self, err: &ComptimeError, span: crate::ast::Span) {
        use crate::diagnostics::Diagnostic;
        let is_fatal = matches!(
            err,
            ComptimeError::SandboxViolation(_) | ComptimeError::MemoryLimitExceeded(_)
        );
        let kind = match err {
            ComptimeError::StepLimitExceeded => ComptimeErrorKind::StepLimitExceeded,
            ComptimeError::DivisionByZero => ComptimeErrorKind::DivisionByZero,
            ComptimeError::Overflow => ComptimeErrorKind::Overflow,
            ComptimeError::TypeError(s) => ComptimeErrorKind::TypeError(s.clone()),
            ComptimeError::AssertionFailed(s) => ComptimeErrorKind::AssertionFailed(s.clone()),
            ComptimeError::UnknownIdentifier(s) => ComptimeErrorKind::UnknownIdentifier(s.clone()),
            ComptimeError::NotComptimeAllowed(s) => {
                ComptimeErrorKind::NotComptimeAllowed(s.clone())
            }
            ComptimeError::Deferred => ComptimeErrorKind::Deferred,
            ComptimeError::SandboxViolation(s) => ComptimeErrorKind::SandboxViolation(s.clone()),
            ComptimeError::MemoryLimitExceeded(s) => {
                ComptimeErrorKind::MemoryLimitExceeded(s.clone())
            }
            ComptimeError::Internal(s) => ComptimeErrorKind::Internal(s.clone()),
        };
        // Merge outer traceback (from checker's region_tree) with inner call stack.
        let mut traceback = self.outer_traceback.clone();
        traceback.extend(self.format_call_stack_traceback());
        let diag = if is_fatal {
            Diagnostic::fatal_kind(DiagnosticKind::Comptime {
                kind,
                span,
                traceback,
            })
        } else {
            Diagnostic::error_kind(DiagnosticKind::Comptime {
                kind,
                span,
                traceback,
            })
        };
        self.diag.push(diag);
    }

    /// Resolve an AST type to a TypeId for layout_of! evaluation.
    /// Creates the type in the TypeContext arena if needed (via ctx.alloc),
    /// so the returned TypeId is always valid for ctx.get() and layout functions.
    fn resolve_ast_type(&mut self, ty: &crate::ast::Type) -> Result<TypeId, ComptimeError> {
        // ── Future enhancement ─────────────────────────────────────
        // Currently handles Path (Int<32>, MyStruct) and simple Generic
        // (Int<N>, UInt<N>, Float<N>).  Complex type aliases, nested
        // generic projections (e.g. <T as Trait>::Assoc), and multi-
        // segment paths (mod::Type) are not yet supported — they fall
        // through to type_error at the end.  A future pass should
        // resolve these through the symbol table and TypeContext's
        // resolve_binding mechanism.
        // ───────────────────────────────────────────────────────────
        use crate::ast::{GenericArg, Literal, Type};
        match ty {
            Type::Path(path, _) => {
                if path.len() == 1 {
                    let name = path[0].as_str();
                    // Built-in types with dedicated fields.
                    match name.as_str() {
                        "Bool" | "bool" => return Ok(self.ctx.builtin_bool),
                        "Unit" | "unit" | "()" => return Ok(self.ctx.builtin_unit),
                        "Never" | "never" | "!" => return Ok(self.ctx.builtin_never),
                        "String" | "Str" | "string" => return Ok(self.ctx.builtin_str),
                        "Char" => return Ok(self.ctx.builtin_char),
                        "Byte" => return Ok(self.ctx.builtin_byte),
                        "USize" => return Ok(self.ctx.builtin_usize),
                        // Bare names without generic args: defaults Int → Int<32>,
                        // UInt → UInt<32>, Float → Float<64>.
                        "Int" | "int" => {
                            return Ok(self.ctx.alloc(TypeData::Int {
                                bits: 32,
                                signed: true,
                                overflow_policy: crate::ast::OverflowPolicy::Trap,
                            }));
                        }
                        "UInt" | "uint" => {
                            return Ok(self.ctx.alloc(TypeData::UInt {
                                bits: 32,
                                overflow_policy: crate::ast::OverflowPolicy::Trap,
                            }));
                        }
                        "Float" | "float" => {
                            return Ok(self.ctx.alloc(TypeData::Float { bits: 64 }));
                        }
                        _ => {}
                    }
                }
                // Try user-defined types via symbol table.
                if let Some(def_id) = self.symbols.lookup_type_by_path(path)
                    && let Some(ty_id) = self.ctx.get_type_id_for_def_id(def_id)
                {
                    return Ok(ty_id);
                }
                let name_str = path
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                Err(ComptimeError::type_error(format!(
                    "unknown type `{}` in layout_of!",
                    name_str
                )))
            }
            Type::Generic(base, args, _) => {
                // Handle Int<N>, UInt<N>, Float<N>.
                if let Type::Path(path, _) = base.as_ref()
                    && path.len() == 1
                {
                    let name = path[0].as_str();
                    // Extract the bit-width from the first positional argument.
                    let bits = match args.first() {
                        Some(GenericArg::Positional(Type::Literal(expr, _))) => {
                            if let crate::ast::Expr::Literal(Literal::Int(bits), _) = expr.as_ref()
                            {
                                *bits
                            } else {
                                return Err(ComptimeError::type_error(format!(
                                    "expected a numeric literal argument for `{}<N>` in layout_of!, found non-literal expression",
                                    name,
                                )));
                            }
                        }
                        _ => {
                            return Err(ComptimeError::type_error(format!(
                                "expected a numeric literal argument for `{}<N>` in layout_of!",
                                name,
                            )));
                        }
                    };
                    let bits_u8 = u8::try_from(bits).map_err(|_| {
                        ComptimeError::type_error(format!(
                            "invalid bit width {} for `{}` in layout_of!",
                            bits, name,
                        ))
                    })?;
                    match name.as_str() {
                        "Int" => {
                            if bits_u8 < 1 || bits_u8 > 64 {
                                return Err(ComptimeError::type_error(format!(
                                    "Int<{}> is out of range; bits must be 1..64",
                                    bits_u8,
                                )));
                            }
                            return Ok(self.ctx.alloc(TypeData::Int {
                                bits: bits_u8,
                                signed: true,
                                overflow_policy: crate::ast::OverflowPolicy::Trap,
                            }));
                        }
                        "UInt" => {
                            if bits_u8 < 1 || bits_u8 > 64 {
                                return Err(ComptimeError::type_error(format!(
                                    "UInt<{}> is out of range; bits must be 1..64",
                                    bits_u8,
                                )));
                            }
                            return Ok(self.ctx.alloc(TypeData::UInt {
                                bits: bits_u8,
                                overflow_policy: crate::ast::OverflowPolicy::Trap,
                            }));
                        }
                        "Float" => {
                            if bits_u8 != 32 && bits_u8 != 64 {
                                return Err(ComptimeError::type_error(format!(
                                    "Float<{}> is not a valid IEEE 754 type; bits must be 32 or 64",
                                    bits_u8,
                                )));
                            }
                            return Ok(self.ctx.alloc(TypeData::Float { bits: bits_u8 }));
                        }
                        _ => {}
                    }
                }
                // Fall through to error for unknown generic types.
                let name_str = match base.as_ref() {
                    Type::Path(p, _) => p.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("::"),
                    _ => "<complex>".to_string(),
                };
                Err(ComptimeError::type_error(format!(
                    "unknown generic type `{}` in layout_of!",
                    name_str
                )))
            }
            _ => Err(ComptimeError::type_error(
                "complex type expressions in layout_of! are not yet supported in comptime",
            )),
        }
    }

    /// Collect the call stack traceback entries.
    /// Returns structured (ComptimeReason, Span) pairs for flexible rendering.
    fn format_call_stack_traceback(&self) -> Vec<(ComptimeReason, crate::ast::Span)> {
        self.call_stack
            .iter()
            .rev()
            .map(|(_, reason, span)| (*reason, *span))
            .collect()
    }

    /// Track memory usage when adding or replacing a variable.
    /// If `slot` is `Some`, subtracts the old value's memory from that slot
    /// (for updates/assignments).  If `None`, skips subtraction (new variable).
    /// Returns `Err(SandboxViolation)` if the memory limit would be exceeded.
    fn track_variable_memory(
        &mut self,
        slot: Option<&SlotId>,
        new_val: &ComptimeValue,
    ) -> Result<(), ComptimeError> {
        // Subtract old value's memory if a slot is provided and it exists.
        if let Some(slot) = slot
            && let Some(old) = self.variables.get(slot)
        {
            let old_size = old.memory_size();
            // In debug builds, catch accounting errors where memory_used
            // is less than the old value's size.  Only assert when
            // memory_used > 0, since direct HashMap inserts (e.g. in tests)
            // bypass tracking and leave memory_used at 0.
            debug_assert!(
                self.memory_used == 0 || self.memory_used >= old_size,
                "memory accounting error: memory_used ({}) < old value size ({}) for slot {:?}",
                self.memory_used,
                old_size,
                slot,
            );
            // In release builds the assertion is stripped, so we use
            // checked_sub to detect the same accounting bug and surface it
            // as a hard error rather than silently lying about memory usage.
            self.memory_used = self.memory_used.checked_sub(old_size).ok_or_else(|| {
                ComptimeError::Internal(format!(
                    "memory accounting error: memory_used ({}) < old value size ({}) for slot {:?}",
                    self.memory_used, old_size, slot,
                ))
            })?;
        }
        let new_size = new_val.memory_size();
        let new_total = self.memory_used.saturating_add(new_size);
        if new_total > self.memory_limit {
            return Err(ComptimeError::MemoryLimitExceeded(format!(
                "~{} bytes used, limit is {} bytes",
                new_total, self.memory_limit,
            )));
        }
        self.memory_used = new_total;
        Ok(())
    }

    /// Evaluate a comptime block (sequence of statements) and return the result.
    ///
    /// # Errors
    ///
    /// Returns `Err(ComptimeError)` if the step limit is exceeded, a variable
    /// is not found, a type mismatch occurs, or division by zero is attempted.
    #[must_use]
    pub fn eval_block(&mut self, stmts: &[HirStmt]) -> Result<ComptimeValue, ComptimeError> {
        let mut result = ComptimeValue::Unit;
        let len = stmts.len();
        for (i, stmt) in stmts.iter().enumerate() {
            // Count each statement toward the step limit, so a block with 10 000
            // straight-line statements (no loops, no function calls) does not
            // bypass the sandbox.
            self.steps = self.steps.saturating_add(1);
            if self.steps >= self.step_limit {
                return Err(ComptimeError::StepLimitExceeded);
            }
            match stmt {
                HirStmt::Expression(expr) => {
                    // Warn about unused pure expressions in comptime blocks.
                    // A pure expression (literal, arithmetic, comparison) has no
                    // side effects — computing and discarding it is likely a bug.
                    // Only warn for non-last expressions (the last expression is
                    // the block's return value, which is used).
                    if i + 1 < len && is_pure_expr(expr) {
                        self.diag.warn(
                            "unused comptime expression result: this computation has no effect",
                        );
                    }
                    result = self.eval_expr(expr)?;
                }
                HirStmt::VariableDef {
                    name, value, span, ..
                } => {
                    let val = match value {
                        Some(e) => self.eval_expr(e)?,
                        None => {
                            return Err(ComptimeError::not_allowed(
                                "variable definitions in comptime blocks must have a value",
                            ));
                        }
                    };
                    if let Some(n) = name {
                        // Allocate a new slot for this variable definition.
                        // Even if the name shadows an existing variable, the new
                        // slot ensures that pointers to the old slot remain valid.
                        let slot = self.allocate_slot();
                        // Track shadowing: if we're inside a scoped block (while body)
                        // and this `set` shadows an outer variable, record the old
                        // slot ID so `cur_slot` can be restored after the block exits.
                        if let Some(shadows) = self.scope_shadows.last_mut()
                            && let Some(&old_slot) = self.cur_slot.get(n)
                        {
                            shadows.entry(*n).or_insert(old_slot);
                        }
                        // Memory tracking: new slot, no old value to subtract.
                        self.track_variable_memory(None, &val)?;
                        self.cur_slot.insert(*n, slot);
                        self.variables.insert(slot, val.clone());
                        result = val;
                    } else {
                        return Err(ComptimeError::not_allowed(
                            "unnamed variables are not allowed in comptime blocks",
                        ));
                    }
                }
                HirStmt::Assign {
                    target,
                    value,
                    span,
                    ..
                } => {
                    let val = self.eval_expr(value)?;
                    if let HirExpr::Ident(name, _, _) = target.as_ref() {
                        // Assignment to a named variable: look up its current slot.
                        match self.cur_slot.get(name) {
                            Some(&slot) => {
                                self.track_variable_memory(Some(&slot), &val)?;
                                self.variables.insert(slot, val.clone());
                                result = val;
                            }
                            None => {
                                return Err(ComptimeError::UnknownIdentifier(name.as_str()));
                            }
                        }
                    } else if let HirExpr::UnaryOp {
                        op: crate::ast::UnaryOp::Deref,
                        expr: ptr_expr,
                        ..
                    } = target.as_ref()
                    {
                        // *ptr = val — assign through a comptime pointer.
                        let ptr_val = self.eval_expr(ptr_expr)?;
                        match ptr_val {
                            ComptimeValue::Pointer { slot, mutable, .. } => {
                                if !mutable {
                                    return Err(ComptimeError::not_allowed(
                                        "cannot assign through an immutable pointer; \
                                         use `&mut` to create a mutable reference",
                                    ));
                                }
                                self.track_variable_memory(Some(&slot), &val)?;
                                self.variables.insert(slot, val.clone());
                                result = val;
                            }
                            _ => {
                                return Err(ComptimeError::type_error(
                                    "cannot assign through a non-pointer value",
                                ));
                            }
                        }
                    } else {
                        return Err(ComptimeError::not_allowed(
                            "only simple variable assignments and pointer dereference assignments \
                             are supported in comptime blocks",
                        ));
                    }
                }
                HirStmt::While {
                    cond, body, span, ..
                } => {
                    loop {
                        if self.steps >= self.step_limit {
                            return Err(ComptimeError::StepLimitExceeded);
                        }
                        let cond_val = self.eval_expr(cond)?;
                        match cond_val {
                            ComptimeValue::Bool(true) => {
                                // Isolate loop body scope: variables defined inside
                                // the body (set x = 1) must not leak to subsequent
                                // iterations or to code after the loop.  Modifications
                                // to outer variables (e.g. i = i + 1) are preserved.
                                let next_slot_at_entry = self.next_slot;
                                self.scope_shadows.push(HashMap::new());
                                self.eval_block(body)?;
                                // Restore `cur_slot` for variables that were shadowed
                                // by `set` (VariableDef) inside the body — name-based
                                // lookups should find the outer slot again.
                                if let Some(shadows) = self.scope_shadows.last() {
                                    for (name, &old_slot) in shadows {
                                        self.cur_slot.insert(*name, old_slot);
                                    }
                                }
                                self.scope_shadows.pop();
                                // Remove any new slots created inside the body
                                // (O(m) where m = new slots, vs O(n) for retain).
                                self.remove_new_slots_since(next_slot_at_entry);
                                // Also purge `cur_slot` entries pointing to the
                                // now-removed slots, so write-after-scope for
                                // body-internal variables (e.g. assigning to `x`
                                // after the loop ends) correctly fails with
                                // UnknownIdentifier instead of silently succeeding.
                                self.cur_slot
                                    .retain(|_, &mut slot| slot.0 < next_slot_at_entry);
                            }
                            ComptimeValue::Bool(false) => break,
                            ComptimeValue::Float(_) => {
                                return Err(ComptimeError::type_error(
                                    "while condition must be a boolean, found Float",
                                ));
                            }
                            ComptimeValue::String(_) => {
                                return Err(ComptimeError::type_error(
                                    "while condition must be a boolean, found String",
                                ));
                            }
                            _ => {
                                return Err(ComptimeError::type_error(
                                    "while condition must be a boolean",
                                ));
                            }
                        }
                    }
                    result = ComptimeValue::Unit;
                }
                _ => {
                    // Reject unsafe blocks and other prohibited constructs
                    // at runtime as a defense-in-depth measure.
                    return Err(ComptimeError::SandboxViolation(
                        "only expressions, variable definitions, and assignments are allowed in comptime blocks".into(),
                    ));
                }
            }
        }
        Ok(result)
    }

    /// Evaluate a comptime expression to a value.
    ///
    /// # Errors
    ///
    /// Returns `Err(ComptimeError)` if the step limit is exceeded, a variable
    /// is not found, a type mismatch occurs, or division by zero is attempted.
    #[must_use]
    pub fn eval_expr(&mut self, expr: &HirExpr) -> Result<ComptimeValue, ComptimeError> {
        if self.steps >= self.step_limit {
            return Err(ComptimeError::StepLimitExceeded);
        }
        self.steps = self.steps.saturating_add(1);

        match expr {
            HirExpr::Literal(lit, _ty, _span) => match lit {
                crate::ast::Literal::Int(n) => Ok(ComptimeValue::Int(*n)),
                crate::ast::Literal::Float(f) => Ok(ComptimeValue::Float(*f)),
                crate::ast::Literal::Char(c) => Ok(ComptimeValue::Int(*c as i128)),
                crate::ast::Literal::Bool(b) => Ok(ComptimeValue::Bool(*b)),
                crate::ast::Literal::String(s) => Ok(ComptimeValue::String(Arc::from(s.as_str()))),
                crate::ast::Literal::ByteString(b) => Ok(ComptimeValue::String(Arc::from(
                    String::from_utf8_lossy(b).as_ref(),
                ))),
            },
            HirExpr::Block(stmts, _ty, _span) => {
                // Blocks introduce a new lexical scope: variable definitions
                // inside the block must not leak to the outer scope, but
                // modifications to existing variables (e.g. through pointers)
                // must be preserved.
                let next_slot_at_entry = self.next_slot;
                let saved_cur_slot = self.cur_slot.clone();
                let result = self.eval_block(stmts);
                self.remove_new_slots_since(next_slot_at_entry);
                self.cur_slot = saved_cur_slot;
                result
            }
            HirExpr::BinaryOp {
                left,
                op,
                right,
                ty,
                ..
            } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                match (l, r, op) {
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Add) => {
                        let result = a.checked_add(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Sub) => {
                        let result = a.checked_sub(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Mul) => {
                        let result = a.checked_mul(b).ok_or(ComptimeError::Overflow)?;
                        check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Div) => {
                        if b == 0 {
                            Err(ComptimeError::DivisionByZero)
                        } else if a == i128::MIN && b == -1 {
                            // i128::MIN / -1 overflows (can't represent as i128)
                            Err(ComptimeError::Overflow)
                        } else {
                            let result = a / b;
                            check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                        }
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Rem) => {
                        if b == 0 {
                            Err(ComptimeError::DivisionByZero)
                        } else if a == i128::MIN && b == -1 {
                            // i128::MIN % -1 overflows in the same way as division
                            Err(ComptimeError::Overflow)
                        } else {
                            let result = a % b;
                            check_range(result, *ty, self.ctx).map(ComptimeValue::Int)
                        }
                    }
                    // Comparison operators: return Bool
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Lt) => {
                        Ok(ComptimeValue::Bool(a < b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Gt) => {
                        Ok(ComptimeValue::Bool(a > b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Le) => {
                        Ok(ComptimeValue::Bool(a <= b))
                    }
                    (ComptimeValue::Int(a), ComptimeValue::Int(b), crate::ast::BinOp::Ge) => {
                        Ok(ComptimeValue::Bool(a >= b))
                    }
                    // ── Float arithmetic ───────────────────────────────
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Add) => {
                        Ok(ComptimeValue::Float(a + b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Sub) => {
                        Ok(ComptimeValue::Float(a - b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Mul) => {
                        Ok(ComptimeValue::Float(a * b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Div) => {
                        Ok(ComptimeValue::Float(a / b))
                    }
                    // ── Float comparisons ──────────────────────────────
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Lt) => {
                        Ok(ComptimeValue::Bool(a < b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Gt) => {
                        Ok(ComptimeValue::Bool(a > b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Le) => {
                        Ok(ComptimeValue::Bool(a <= b))
                    }
                    (ComptimeValue::Float(a), ComptimeValue::Float(b), crate::ast::BinOp::Ge) => {
                        Ok(ComptimeValue::Bool(a >= b))
                    }
                    // ── String concatenation ───────────────────────────
                    (
                        ComptimeValue::String(a),
                        ComptimeValue::String(b),
                        crate::ast::BinOp::Add,
                    ) => {
                        let mut result = String::with_capacity(a.len() + b.len());
                        result.push_str(&a);
                        result.push_str(&b);
                        Ok(ComptimeValue::String(Arc::from(result)))
                    }
                    // ── String equality ────────────────────────────────
                    (ComptimeValue::String(a), ComptimeValue::String(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (
                        ComptimeValue::String(a),
                        ComptimeValue::String(b),
                        crate::ast::BinOp::Neq,
                    ) => Ok(ComptimeValue::Bool(a != b)),
                    // ── Bool equality ───────────────────────────────────
                    (ComptimeValue::Bool(a), ComptimeValue::Bool(b), crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(a == b))
                    }
                    (ComptimeValue::Bool(a), ComptimeValue::Bool(b), crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(a != b))
                    }
                    // ── Unit equality ────────────────────────────────────
                    (ComptimeValue::Unit, ComptimeValue::Unit, crate::ast::BinOp::Eq) => {
                        Ok(ComptimeValue::Bool(true))
                    }
                    (ComptimeValue::Unit, ComptimeValue::Unit, crate::ast::BinOp::Neq) => {
                        Ok(ComptimeValue::Bool(false))
                    }
                    _ => Err(ComptimeError::type_error("unsupported binary operation")),
                }
            }
            HirExpr::UnaryOp { op, expr, .. } => {
                match op {
                    crate::ast::UnaryOp::Ref | crate::ast::UnaryOp::RefMut => {
                        // &expr / &mut expr — create a comptime pointer.
                        // Evaluate the expression to ensure it exists, then
                        // store it and return a pointer with the slot ID.
                        let val = self.eval_expr(expr)?;
                        let slot = match expr.as_ref() {
                            HirExpr::Ident(n, _, _) => {
                                // Look up the current slot for this variable.
                                *self.cur_slot.get(n).ok_or_else(|| {
                                    ComptimeError::type_error(format!(
                                        "cannot take reference of unknown variable `{}`",
                                        n.as_str(),
                                    ))
                                })?
                            }
                            _ => {
                                return Err(ComptimeError::type_error(
                                    "comptime references can only be taken of simple variables",
                                ));
                            }
                        };
                        Ok(ComptimeValue::Pointer {
                            slot,
                            mutable: *op == crate::ast::UnaryOp::RefMut,
                        })
                    }
                    crate::ast::UnaryOp::Deref => {
                        // *ptr — dereference a comptime pointer.
                        let ptr_val = self.eval_expr(expr)?;
                        match ptr_val {
                            ComptimeValue::Pointer { slot, mutable, .. } => {
                                // Look up the current value by slot ID, so that
                                // shadowing (same name, inner scope) does NOT
                                // affect pointer dereference.
                                match self.variables.get(&slot) {
                                    Some(val) => Ok(val.clone()),
                                    None => Err(ComptimeError::type_error(format!(
                                        "dereferenced pointer to slot {:?} \
                                         points to a variable that went out of \
                                         scope (use-after-scope); comptime pointers \
                                         are tied to the variable's lifetime in \
                                         the current scope",
                                        slot,
                                    ))),
                                }
                            }
                            _ => Err(ComptimeError::type_error(
                                "cannot dereference non-pointer value in comptime",
                            )),
                        }
                    }
                    crate::ast::UnaryOp::Neg => {
                        let val = self.eval_expr(expr)?;
                        match val {
                            ComptimeValue::Int(n) => Ok(ComptimeValue::Int(n.wrapping_neg())),
                            ComptimeValue::Float(f) => Ok(ComptimeValue::Float(-f)),
                            _ => Err(ComptimeError::type_error(
                                "negation is only supported for integers and floats in comptime",
                            )),
                        }
                    }
                    crate::ast::UnaryOp::Not => {
                        let val = self.eval_expr(expr)?;
                        match val {
                            ComptimeValue::Bool(b) => Ok(ComptimeValue::Bool(!b)),
                            ComptimeValue::Int(n) => Ok(ComptimeValue::Int(!n)),
                            _ => Err(ComptimeError::type_error(
                                "logical not is only supported for booleans and integers in comptime",
                            )),
                        }
                    }
                    _ => Err(ComptimeError::Deferred),
                }
            }
            HirExpr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let cond_val = self.eval_expr(cond)?;
                match cond_val {
                    ComptimeValue::Bool(true) => self.eval_block(then_branch),
                    ComptimeValue::Bool(false) => {
                        if let Some(else_branch) = else_branch {
                            self.eval_block(else_branch)
                        } else {
                            Ok(ComptimeValue::Unit)
                        }
                    }
                    ComptimeValue::Float(_) => Err(ComptimeError::type_error(
                        "if condition must be a boolean, found Float",
                    )),
                    ComptimeValue::String(_) => Err(ComptimeError::type_error(
                        "if condition must be a boolean, found String",
                    )),
                    _ => Err(ComptimeError::type_error("if condition must be a boolean")),
                }
            }
            HirExpr::Ident(name, _ty, _span) => {
                // 1. Check local variables first via the name→slot mapping.
                if let Some(&slot) = self.cur_slot.get(name)
                    && let Some(val) = self.variables.get(&slot)
                {
                    return Ok(val.clone());
                }
                // 2. Check if the name is a zero-argument comptime function
                //    (e.g. `comptime def N() -> Int<32> { 5 }` referenced as `N`).
                if let Some((params, body)) = self.fn_registry.get(name)
                    && params.is_empty()
                {
                    let body = Arc::clone(body);
                    let saved = std::mem::take(&mut self.variables);
                    let result = self.eval_block(&body);
                    self.variables = saved;
                    return result;
                }
                // 3. Check the symbol table for runtime variables.
                //    These are variables declared outside the comptime block
                //    (e.g. `let x = 42; comptime { ... x ... }`).
                //    If the variable was declared as `let` with a literal initializer,
                //    the checker already pre-populated `self.variables` with its
                //    comptime value (see task #2 below), so step 1 caught it.
                //    If we reach here, the variable is a true runtime variable whose
                //    value is NOT known at compile time — emit a clear error instead
                //    of silently returning the type.
                if let Some(binding) = self
                    .symbols
                    .lookup_variable(*name, crate::ast::Span::new(0, 0))
                {
                    let mutability = if binding.mutable {
                        "mutable"
                    } else {
                        "immutable"
                    };
                    return Err(ComptimeError::not_allowed(format!(
                        "cannot access {} runtime variable `{}` in comptime context; \
                         its value is not available at compile time. \
                         Use a comptime function parameter or a `let` binding with a \
                         literal initializer to make values available at compile time",
                        mutability, name,
                    )));
                }
                // 4. Check if it's a known function (for better error messages).
                if let Some(_func) = self.symbols.lookup_function(*name) {
                    return Err(ComptimeError::not_allowed(format!(
                        "cannot call function `{}` without `!` in comptime context; use `{}!()`",
                        name, name,
                    )));
                }
                Err(ComptimeError::UnknownIdentifier(name.as_str()))
            }
            HirExpr::Call {
                callee,
                args,
                comptime,
                span,
                ..
            } if *comptime => {
                // Resolve the callee to a function name.
                let fn_name = match callee.as_ref() {
                    HirExpr::Ident(name, _, _) => *name,
                    _ => {
                        return Err(ComptimeError::type_error(
                            "comptime call target must be a simple function name",
                        ));
                    }
                };
                // Built-in: assert(condition)
                if fn_name.eq_str("assert") {
                    if args.len() != 1 {
                        return Err(ComptimeError::type_error(
                            "assert takes exactly one argument",
                        ));
                    }
                    let cond = self.eval_expr(&args[0])?;
                    match cond {
                        ComptimeValue::Bool(true) => Ok(ComptimeValue::Unit),
                        ComptimeValue::Bool(false) => {
                            Err(ComptimeError::AssertionFailed("assertion failed".into()))
                        }
                        _ => Err(ComptimeError::type_error(
                            "assert argument must be a boolean",
                        )),
                    }
                } else {
                    // ── @trusted/@io boundary check ──────────────────
                    // If this comptime block is not @trusted, reject calls
                    // to functions marked @trusted or @io.
                    if !self.allow_trusted
                        && let Some(func) = self.symbols.lookup_function(fn_name)
                    {
                        if func.attributes.iter().any(|a| a.name.eq_str("trusted")) {
                            return Err(ComptimeError::not_allowed(format!(
                                "cannot call @trusted function `{}` from a non-@trusted \
                                 comptime block; annotate the block with `comptime @trusted`",
                                fn_name,
                            )));
                        }
                        if func.attributes.iter().any(|a| a.name.eq_str("io")) {
                            return Err(ComptimeError::not_allowed(format!(
                                "cannot call @io function `{}` from a non-@trusted \
                                 comptime block; annotate the block with `comptime @trusted`",
                                fn_name,
                            )));
                        }
                    }
                    // Look up the function in the registry.
                    if let Some((params, body)) = self.fn_registry.get(&fn_name) {
                        let params = params.clone();
                        let body = Arc::clone(body);
                        // Evaluate arguments.
                        let arg_vals: Vec<ComptimeValue> = args
                            .iter()
                            .map(|a| self.eval_expr(a))
                            .collect::<Result<Vec<_>, _>>()?;
                        if arg_vals.len() != params.len() {
                            return Err(ComptimeError::type_error(format!(
                                "comptime function `{}` expected {} arguments, got {}",
                                fn_name,
                                params.len(),
                                arg_vals.len(),
                            )));
                        }
                        // Save the current variable scope (caller's variables and
                        // cur_slot) so the function body runs in isolation.
                        let saved = std::mem::take(&mut self.variables);
                        let saved_cur_slot = std::mem::take(&mut self.cur_slot);
                        let saved_memory = self.memory_used;
                        // Bind parameters into the now-empty scope.
                        // If any parameter's memory check fails, restore the outer
                        // state before propagating the error so the evaluator is
                        // never left with empty maps after a partial bind.
                        let bind_result: Result<(), ComptimeError> = (|| {
                            for (param, val) in params.iter().zip(arg_vals.into_iter()) {
                                let slot = self.allocate_slot();
                                self.cur_slot.insert(*param, slot);
                                self.track_variable_memory(None, &val)?;
                                self.variables.insert(slot, val);
                            }
                            Ok(())
                        })();
                        let saved = match bind_result {
                            Ok(()) => saved,
                            Err(e) => {
                                self.variables = saved;
                                self.cur_slot = saved_cur_slot;
                                self.memory_used = saved_memory;
                                return Err(e);
                            }
                        };
                        // Push call stack entry for traceback.
                        self.call_stack
                            .push((fn_name, ComptimeReason::ComptimeFnCall, *span));
                        // Evaluate the function body.
                        let result = self.eval_block(&body);
                        // Pop call stack entry.
                        self.call_stack.pop();
                        // Restore the previous variable scope and memory accounting.
                        self.variables = saved;
                        self.cur_slot = saved_cur_slot;
                        self.memory_used = saved_memory;
                        result
                    } else if self.symbols.lookup_function(fn_name).is_some() {
                        // Known function but not a comptime function.
                        Err(ComptimeError::not_allowed(format!(
                            "function `{}` is not defined with `comptime def` and cannot be called with `!` in comptime context",
                            fn_name,
                        )))
                    } else {
                        Err(ComptimeError::UnknownIdentifier(fn_name.as_str()))
                    }
                }
            }
            HirExpr::FieldAccess { base, field, .. } => {
                // Evaluate the base, then resolve the field name.
                let base_val = self.eval_expr(base)?;
                match base_val {
                    ComptimeValue::TypeInfo(info) => {
                        if field.eq_str("name") {
                            Ok(ComptimeValue::String(Arc::from(info.name.as_str())))
                        } else if field.eq_str("kind") {
                            Ok(ComptimeValue::String(Arc::from(format!("{:?}", info.kind))))
                        } else if field.eq_str("bits") {
                            if let Some(b) = info.bits {
                                Ok(ComptimeValue::Int(b as i128))
                            } else {
                                Ok(ComptimeValue::Unit)
                            }
                        } else if field.eq_str("float_bits") {
                            if let Some(b) = info.float_bits {
                                Ok(ComptimeValue::Int(b as i128))
                            } else {
                                Ok(ComptimeValue::Unit)
                            }
                        } else if field.eq_str("params") {
                            // NOTE: Serialized to a comma-separated string.
                            // For programmatic iteration over generic params,
                            // use `generate` blocks instead.
                            let s = info.params.join(", ");
                            Ok(ComptimeValue::String(Arc::from(s)))
                        } else if field.eq_str("fields") {
                            // NOTE: This serializes fields to a display string.
                            // For programmatic iteration over fields (e.g. code
                            // generation per field), use `generate` blocks instead.
                            let s = info
                                .fields
                                .iter()
                                .map(|f| format!("{}: {:?}", f.name, f.ty))
                                .collect::<Vec<_>>()
                                .join(", ");
                            Ok(ComptimeValue::String(Arc::from(s)))
                        } else if field.eq_str("variants") {
                            // NOTE: Serialized to a display string.  For programmatic
                            // iteration over variants, use `generate` blocks instead.
                            let s = info
                                .variants
                                .iter()
                                .map(|v| {
                                    let payload = v
                                        .payload
                                        .iter()
                                        .map(|f| format!("{}: {:?}", f.name, f.ty))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    format!("{}({})", v.name, payload)
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            Ok(ComptimeValue::String(Arc::from(s)))
                        } else {
                            Err(ComptimeError::type_error(format!(
                                "unknown field `{}` on TypeInfo; expected name, kind, bits, float_bits, params, fields, or variants",
                                field.as_str(),
                            )))
                        }
                    }
                    ComptimeValue::LayoutDescriptor(desc) => {
                        if field.eq_str("size") {
                            Ok(ComptimeValue::Int(desc.size as i128))
                        } else if field.eq_str("align") {
                            Ok(ComptimeValue::Int(desc.align as i128))
                        } else if field.eq_str("fields") {
                            let s = desc
                                .fields
                                .iter()
                                .map(|f| format!("{}@{}:{}", f.name, f.offset, f.size))
                                .collect::<Vec<_>>()
                                .join(", ");
                            Ok(ComptimeValue::String(Arc::from(s)))
                        } else {
                            Err(ComptimeError::type_error(format!(
                                "unknown field `{}` on LayoutDescriptor; expected size, align, or fields",
                                field.as_str(),
                            )))
                        }
                    }
                    ComptimeValue::Aggregate { fields } => fields
                        .iter()
                        .find(|(name, _)| name == field)
                        .map(|(_, val)| val.clone())
                        .ok_or_else(|| {
                            ComptimeError::type_error(format!(
                                "unknown field `{}` on aggregate value",
                                field.as_str(),
                            ))
                        }),
                    _ => Err(ComptimeError::type_error(
                        "field access is only supported on TypeInfo, LayoutDescriptor, Aggregate, and Pointer values in comptime",
                    )),
                }
            }
            HirExpr::Index { base, index, .. } => {
                let base_val = self.eval_expr(base)?;
                let idx_val = self.eval_expr(index)?;
                let idx = match idx_val {
                    ComptimeValue::Int(n) => n as usize,
                    _ => {
                        return Err(ComptimeError::type_error(
                            "index must be an integer in comptime",
                        ));
                    }
                };
                match base_val {
                    ComptimeValue::Pointer { slot, .. } => {
                        // ptr[i] — access element through a pointer by slot ID.
                        let arr = self.variables.get(&slot).ok_or_else(|| {
                            ComptimeError::type_error(format!(
                                "pointer to slot {:?} points to a variable that went out of \
                                 scope (use-after-scope) — comptime pointers are tied to \
                                 the variable's lifetime in the current scope",
                                slot,
                            ))
                        })?;
                        if let ComptimeValue::Aggregate { fields } = arr {
                            fields.get(idx).map(|(_, val)| val.clone()).ok_or_else(|| {
                                ComptimeError::type_error(format!(
                                    "index {} out of bounds for array",
                                    idx
                                ))
                            })
                        } else {
                            return Err(ComptimeError::type_error(format!(
                                "cannot index into non-aggregate type (index {}) — \
                                 `ptr[i]` requires a pointer to an array or aggregate",
                                idx,
                            )));
                        }
                    }
                    ComptimeValue::Aggregate { fields } => {
                        fields.get(idx).map(|(_, val)| val.clone()).ok_or_else(|| {
                            ComptimeError::type_error(format!(
                                "index {} out of bounds for array",
                                idx
                            ))
                        })
                    }
                    _ => Err(ComptimeError::type_error(
                        "indexing is only supported on Aggregate and Pointer values in comptime",
                    )),
                }
            }
            HirExpr::TypeInfo(ty, _) => {
                // @typeInfo(T) returns a structured TypeInfo value describing
                // the type's fields, variants, parameters, etc.
                let info = crate::hir::generate::get_type_info(self.ctx, self.symbols, *ty);
                Ok(ComptimeValue::TypeInfo(Box::new(info)))
            }
            HirExpr::LayoutOf(ty, _) => {
                // layout_of!(T) returns a LayoutDescriptor describing the
                // type's size, alignment, and field offsets.
                // Resolve the AST type to a TypeId first.
                let ty_id = self.resolve_ast_type(ty)?;
                // debug_assert — if MutVisitor skips generic substitution
                // inside LayoutOf's AST type (see walk_expr_mut in visit.rs),
                // the resolved TypeId may still contain InferVar.  This
                // catches that case at test/dev time with a clear message.
                debug_assert!(
                    !self.ctx.is_infer_var(ty_id),
                    "layout_of! resolved type contains InferVar — \
                     MutVisitor likely skipped generic substitution \
                     inside LayoutOf's AST type"
                );
                use crate::hir::target::layout::compute_adt_layout;
                let target = self.ctx.target.clone();
                let layout =
                    compute_adt_layout(self.ctx, Some(self.symbols), &target, ty_id, self.diag)
                        .ok_or_else(|| {
                            ComptimeError::type_error(
                                "layout_of! requires a concrete type with known layout",
                            )
                        })?;
                Ok(ComptimeValue::LayoutDescriptor(Box::new(layout)))
            }
            HirExpr::CompileError(msg, _) => Err(ComptimeError::AssertionFailed(msg.clone())),
            HirExpr::StructLit { fields, .. } => {
                let mut agg_fields = Vec::new();
                for (name, expr) in fields {
                    let val = self.eval_expr(expr)?;
                    agg_fields.push((*name, val));
                }
                Ok(ComptimeValue::Aggregate { fields: agg_fields })
            }
            HirExpr::Tuple(elems, _, _) => {
                let mut agg_fields = Vec::with_capacity(elems.len());
                for (i, expr) in elems.iter().enumerate() {
                    let val = self.eval_expr(expr)?;
                    agg_fields.push((tuple_field_name(i), val));
                }
                Ok(ComptimeValue::Aggregate { fields: agg_fields })
            }
            HirExpr::Array(elems, _, _) => {
                let mut agg_fields = Vec::with_capacity(elems.len());
                for (i, expr) in elems.iter().enumerate() {
                    let val = self.eval_expr(expr)?;
                    agg_fields.push((array_field_name(i), val));
                }
                Ok(ComptimeValue::Aggregate { fields: agg_fields })
            }
            HirExpr::TypeAnnotated { expr, .. } => self.eval_expr(expr),
            HirExpr::Cast { expr, ty, .. } => {
                let val = self.eval_expr(expr)?;
                let target_tag = crate::hir::types::TypeTag::from(self.ctx.get(*ty));
                match (val, target_tag) {
                    (ComptimeValue::Int(n), crate::hir::types::TypeTag::Int) => {
                        check_range(n, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(n), crate::hir::types::TypeTag::UInt) => {
                        if n < 0 {
                            return Err(ComptimeError::type_error(
                                "cannot cast negative integer to unsigned type",
                            ));
                        }
                        check_range(n, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Int(n), crate::hir::types::TypeTag::Float) => {
                        Ok(ComptimeValue::Float(n as f64))
                    }
                    (ComptimeValue::Float(f), crate::hir::types::TypeTag::Int) => {
                        if f.is_nan() || f.is_infinite() {
                            return Err(ComptimeError::type_error(format!(
                                "cannot cast {} float to integer",
                                if f.is_nan() { "NaN" } else { "infinite" },
                            )));
                        }
                        // Check range BEFORE the `as i128` cast to avoid misleading
                        // "integer overflow" error when the saturating cast produces
                        // i128::MAX/MIN but the target type is smaller (e.g. Int<32>).
                        let max_int = i128::MAX as f64;
                        let min_int = i128::MIN as f64;
                        if f > max_int || f < min_int {
                            return Err(ComptimeError::type_error(format!(
                                "cannot cast float value {} to integer — value exceeds \
                                 representable integer range ({} .. {})",
                                f,
                                i128::MIN,
                                i128::MAX,
                            )));
                        }
                        let n = f as i128;
                        check_range(n, *ty, self.ctx).map(ComptimeValue::Int)
                    }
                    (ComptimeValue::Float(f), crate::hir::types::TypeTag::Float) => {
                        Ok(ComptimeValue::Float(f))
                    }
                    (v, _) => Ok(v),
                }
            }
            HirExpr::EnumLit {
                path,
                variant,
                payload,
                ..
            } => {
                // Store variant + path + optional payload.
                // The first field is the variant name (used for matching).
                let mut fields = vec![(*variant, ComptimeValue::Unit)];
                // Second field: the type path, so patterns can distinguish
                // e.g. Result::Ok from MyEnum::Ok.
                let path_str = path
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                fields.push((
                    Symbol::intern("$path"),
                    ComptimeValue::String(Arc::from(path_str)),
                ));
                debug_assert_eq!(
                    fields[0].0.as_str(),
                    variant.as_str(),
                    "variant must be at index 0"
                );
                debug_assert_eq!(
                    fields[1].0.as_str(),
                    "$path",
                    "$path metadata must be at index 1"
                );
                if let Some(payload_expr) = payload {
                    let val = self.eval_expr(payload_expr)?;
                    fields.push((Symbol::intern(PAYLOAD_FIELD), val));
                }
                Ok(ComptimeValue::Aggregate { fields })
            }
            HirExpr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                ..
            } => {
                let val = self.eval_expr(scrutinee)?;
                if self.eval_pattern(pattern, &val) {
                    // ⚠️  Pattern bindings are not yet extracted (see
                    // eval_pattern doc) — this is a known limitation until
                    // we have a MIR layer.
                    self.eval_block(then_branch)
                } else if let Some(else_branch) = else_branch {
                    self.eval_block(else_branch)
                } else {
                    Ok(ComptimeValue::Unit)
                }
            }
            HirExpr::Match {
                scrutinee, arms, ..
            } => self.eval_match(scrutinee, arms),
            // ── Partially supported expression types ──────────────────
            HirExpr::Closure { body, params, .. } => {
                if params.is_empty() {
                    // Zero-parameter closure: evaluate the body immediately.
                    self.eval_block(body)
                } else {
                    Err(ComptimeError::Deferred)
                }
            }
            HirExpr::Try { expr, .. } => {
                // `expr?` — in comptime, just evaluate the expression.
                self.eval_expr(expr)
            }
            HirExpr::Catch { expr, branches, .. } => {
                let val = self.eval_expr(expr)?;
                // For comptime, the expression succeeded — return it.
                // If it had failed, we'd match branches, but in comptime
                // errors are propagated via ComptimeError, not catch.
                Ok(val)
            }
            HirExpr::LeaveWith { expr, .. } => {
                // `leave with expr` — comptime error exit.
                // Evaluate the expression for its side effects, then
                // return a type error since comptime can't have error exits.
                let _val = self.eval_expr(expr)?;
                Err(ComptimeError::not_allowed(
                    "`leave with` is not supported in comptime context",
                ))
            }
            HirExpr::Await { expr, .. } => {
                let _val = self.eval_expr(expr)?;
                Err(ComptimeError::not_allowed(
                    "`await` is not supported in comptime context",
                ))
            }
            HirExpr::UnsafeBlock { body, .. } => {
                // Unsafe blocks in comptime: evaluate the body.
                // The checker already verifies that comptime code cannot
                // call @trusted/@io functions, so unsafe is just a block.
                self.eval_block(body)
            }
            HirExpr::PolyBox { expr, .. } => {
                // `poly(expr)` — box a polymorphic expression.
                self.eval_expr(expr)
            }
            HirExpr::PolyUnbox { expr, .. } => {
                // `unbox(expr)` — unbox a polymorphic expression.
                self.eval_expr(expr)
            }
            HirExpr::AttrAccess { base, attr, .. } => {
                // `T'default` — access a type attribute.
                let base_val = self.eval_expr(base)?;
                let attr_name = attr.as_str();
                match attr_name.as_str() {
                    "default" => {
                        // T'default — return the type itself as a value.
                        Ok(base_val)
                    }
                    _ => Err(ComptimeError::type_error(format!(
                        "unknown type attribute `{:?}'{}`",
                        base_val, attr_name,
                    ))),
                }
            }
            _ => Err(ComptimeError::Deferred),
        }
    }

    /// Evaluate a match expression in comptime.
    /// Evaluates the scrutinee, then tries each arm's pattern.
    /// Supports guard expressions (`if guard` on each arm).
    fn eval_match(
        &mut self,
        scrutinee: &HirExpr,
        arms: &[HirMatchArm],
    ) -> Result<ComptimeValue, ComptimeError> {
        let val = self.eval_expr(scrutinee)?;
        for arm in arms {
            if self.eval_pattern(&arm.pattern, &val) {
                // ⚠️  Pattern bindings are not yet extracted (see
                // eval_pattern doc) — this is a known limitation until
                // we have a MIR layer.
                if let Some(guard) = &arm.guard {
                    let guard_val = self.eval_expr(guard)?;
                    match guard_val {
                        ComptimeValue::Bool(true) => return self.eval_expr(&arm.body),
                        ComptimeValue::Bool(false) => continue,
                        _ => {
                            return Err(ComptimeError::type_error(
                                "match arm guard must evaluate to a boolean",
                            ));
                        }
                    }
                }
                return self.eval_expr(&arm.body);
            }
        }
        Err(ComptimeError::type_error(
            "non-exhaustive match: no arm matched the value",
        ))
    }

    /// Check if a comptime value matches a pattern.
    ///
    /// Returns `true` if the pattern matches, `false` otherwise.
    ///
    /// ⚠️  Variable bindings (HirPattern::Ident → named bindings like `Some(n)`)
    /// are NOT extracted here — they are intentionally skipped because we do not
    /// yet have a MIR layer.  In a MIR-based lowering, each arm's bindings would
    /// be assigned to dedicated locals scoped to that arm's block (see rustc's
    /// `bind_matched_candidate_for_arm_body`).  Until then, pattern-bound
    /// variables are simply ignored; code that references them in comptime will
    /// hit "unknown identifier" at eval time.
    // TODO: implement pattern variable binding at the MIR layer.
    fn eval_pattern(&mut self, pattern: &HirPattern, val: &ComptimeValue) -> bool {
        match pattern {
            HirPattern::Wildcard(_) => true,
            HirPattern::Ident(_, _, _) => true,
            HirPattern::Literal(pat_expr, _) => {
                match self.eval_expr(pat_expr) {
                    Ok(pat_val) => {
                        let matched = match (&pat_val, val) {
                            (ComptimeValue::Int(a), ComptimeValue::Int(b)) => a == b,
                            (ComptimeValue::Float(a), ComptimeValue::Float(b)) => {
                                // Exact bitwise comparison for comptime pattern matching.
                                // `a.to_bits() == b.to_bits()` matches identical bit patterns,
                                // including NaN == NaN (unlike IEEE 754).  This is intentional:
                                // comptime code should not depend on floating-point semantics.
                                a.to_bits() == b.to_bits()
                            }
                            (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => a == b,
                            (ComptimeValue::String(a), ComptimeValue::String(b)) => a == b,
                            _ => std::mem::discriminant(&pat_val) == std::mem::discriminant(val),
                        };
                        if matched { true } else { false }
                    }
                    Err(_) => false,
                }
            }
            HirPattern::Enum {
                path,
                variant,
                inner,
                ..
            } => {
                match val {
                    ComptimeValue::Aggregate { fields } => {
                        // The EnumLit handler stores the variant name as the
                        // first field (index 0) — check it by position rather
                        // than scanning all fields by name, avoiding false
                        // matches when a variant name coincides with a
                        // metadata field name like "$path" or "payload".
                        let variant_matches = fields
                            .first()
                            .map(|(name, _)| *name == *variant)
                            .unwrap_or(false);
                        if !variant_matches {
                            return false;
                        }
                        // Also check the enum type path if present in the
                        // pattern (e.g. Result::Ok vs MyEnum::Ok).
                        // Look up "$path" by name rather than relying on position index.
                        let path_matches = if !path.is_empty() {
                            let pattern_path = path
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .join("::");
                            fields.iter().find_map(|(name, val)| {
                                if name.eq_str("$path") {
                                    Some(matches!(val, ComptimeValue::String(s) if s.as_ref() == pattern_path))
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false)
                        } else {
                            true
                        };
                        if !path_matches {
                            return false;
                        }
                        // If there's an inner pattern, match it against the
                        // payload field (stored at the last position only when
                        // the enum variant has a payload).
                        if let Some(inner_pat) = inner {
                            fields
                                .iter()
                                .find(|(name, _)| name.eq_str(PAYLOAD_FIELD))
                                .map(|(_, v)| self.eval_pattern(inner_pat, v))
                                .unwrap_or(false)
                        } else {
                            true
                        }
                    }
                    _ => false,
                }
            }
            HirPattern::Tuple(patterns, _) => match val {
                ComptimeValue::Aggregate { fields } => {
                    fields.len() == patterns.len()
                        && patterns
                            .iter()
                            .zip(fields.iter())
                            .all(|(p, (_, v))| self.eval_pattern(p, v))
                }
                _ => false,
            },
            HirPattern::Struct { fields, rest, .. } => match val {
                ComptimeValue::Aggregate { fields: val_fields } => {
                    for (field_name, pat) in fields {
                        let field_val = val_fields.iter().find(|(n, _)| n == field_name);
                        match field_val {
                            Some((_, v)) => {
                                if !self.eval_pattern(pat, v) {
                                    return false;
                                }
                            }
                            None => {
                                if !rest {
                                    return false;
                                }
                            }
                        }
                    }
                    true
                }
                _ => false,
            },
            HirPattern::Or(alternatives, _) => {
                alternatives.iter().any(|alt| self.eval_pattern(alt, val))
            }
            HirPattern::Slice(before, mid, after, _) => match val {
                ComptimeValue::Aggregate { fields } => {
                    let total = before.len() + after.len();
                    if fields.len() < total {
                        return false;
                    }
                    let remaining = fields.len() - total;
                    if mid.is_none() && remaining != 0 {
                        return false;
                    }
                    let mut pos = 0;
                    for p in before {
                        if !self.eval_pattern(p, &fields[pos].1) {
                            return false;
                        }
                        pos += 1;
                    }
                    if let Some(mid_pat) = mid {
                        let mid_fields: Vec<(Symbol, ComptimeValue)> =
                            fields.iter().skip(pos).take(remaining).cloned().collect();
                        if !self
                            .eval_pattern(mid_pat, &ComptimeValue::Aggregate { fields: mid_fields })
                        {
                            return false;
                        }
                        pos += remaining;
                    }
                    for p in after {
                        if !self.eval_pattern(p, &fields[pos].1) {
                            return false;
                        }
                        pos += 1;
                    }
                    true
                }
                _ => false,
            },
            HirPattern::Error(_) => false,
        }
    }
}
