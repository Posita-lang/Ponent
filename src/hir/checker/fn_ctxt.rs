use super::*;

/// Context for type-checking a single function body.
/// Holds the mutable borrows needed for expression/statement inference
/// and checking, keeping TypeChecker focused on module-level state.
/// Following the rustc `FnCtxt` pattern — see checker/mod.rs for the
/// top-level TypeChecker that owns the global state (ctx, symbols, etc.).
pub struct FnCtxt<'a, 'tcx> {
    pub checker: &'tcx mut TypeChecker<'a>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn new(checker: &'tcx mut TypeChecker<'a>) -> Self {
        FnCtxt { checker }
    }

    /// Convenience accessors that delegate to the underlying TypeChecker.
    fn ctx(&mut self) -> &mut TypeContext { self.checker.ctx }
    fn infer(&mut self) -> &mut InferenceContext { &mut self.checker.infer }

    /// Suggest a cast for common type mismatches (e.g. Int ↔ Float).
    pub fn suggest_cast(&self, expected: TypeId, actual: TypeId) -> Option<String> {
        let (e, a) = (self.checker.ctx.get(expected), self.checker.ctx.get(actual));
        match (e, a) {
            (TypeData::Int { .. }, TypeData::Float { .. })
            | (TypeData::Float { .. }, TypeData::Int { .. }) =>
                Some("try using `as` to cast between integer and float types".into()),
            (TypeData::Bool, TypeData::Int { .. }) =>
                Some("try `x != 0` to convert Int to Bool".into()),
            (TypeData::Int { .. }, TypeData::Bool) =>
                Some("try `if x { 1 } else { 0 }` to convert Bool to Int".into()),
            _ => None,
        }
    }

    pub fn unify(&mut self, expected: TypeId, actual: TypeId, span: Span) -> Result<(), Diagnostic> {
        self.checker.ctx.unify(expected, actual).map(|_| ()).map_err(|_err| {
            let msg = format!("type mismatch: expected {:?}, found {:?}",
                self.checker.ctx.get(expected), self.checker.ctx.get(actual));
            let mut diag = Diagnostic::error(msg).with_code("E030").with_span(span);
            if let Some(suggestion) = self.suggest_cast(expected, actual) {
                diag = diag.with_suggestion(suggestion);
            }
            diag
        })
    }

    pub fn unify_with(&mut self, expected: TypeId, actual: TypeId, span: Span, ctx: TypingContext) -> Result<(), Diagnostic> {
        self.checker.ctx.unify(expected, actual).map(|_| ()).map_err(|_err| {
            let msg = match ctx {
                TypingContext::ReturnValue =>
                    format!("return value type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected), self.checker.ctx.get(actual)),
                TypingContext::StructFieldInit =>
                    format!("field initializer type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected), self.checker.ctx.get(actual)),
                TypingContext::Condition =>
                    format!("condition must be boolean, got {:?}", self.checker.ctx.get(actual)),
                TypingContext::Argument { index, total } =>
                    format!("argument {} of {} has wrong type: expected {:?}, found {:?}",
                        index + 1, total, self.checker.ctx.get(expected), self.checker.ctx.get(actual)),
                TypingContext::ClosureBody =>
                    format!("closure body type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected), self.checker.ctx.get(actual)),
                TypingContext::None =>
                    format!("type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected), self.checker.ctx.get(actual)),
                TypingContext::Index =>
                    format!("index must be an integer, got {:?}", self.checker.ctx.get(actual)),
            };
            let mut diag = Diagnostic::error(msg).with_code("E030").with_span(span)
                .with_label(span, format!("expected {:?}", self.checker.ctx.get(expected)));
            if let Some(suggestion) = self.suggest_cast(expected, actual) {
                diag = diag.with_suggestion(suggestion);
            }
            diag
        })
    }

    // ── Infer expression type ─────────────────────────────────────────────
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<(HirExpr, TypeId), Diagnostic> {
        self.checker.infer_expr(expr)
    }

    /// Check expression against a known type (bidirectional).
    pub fn check_expr(&mut self, expr: &Expr, expected: Expectation, ctx: TypingContext) -> Result<HirExpr, Diagnostic> {
        self.checker.check_expr(expr, expected, ctx)
    }

    /// Check a pattern against an expected type.
    pub fn check_pattern(&mut self, pattern: &Pattern, expected_ty: TypeId) -> Result<HirPattern, Diagnostic> {
        self.checker.check_pattern(pattern, expected_ty)
    }

    /// Resolve a syntactic type to a TypeId.
    pub fn resolve_type(&mut self, ty: &Type) -> Result<TypeId, Diagnostic> {
        self.checker.resolve_type(ty)
    }

    /// Get the type yielded by a block (last expression's type, or unit/never).
    pub fn block_type(&self, stmts: &[HirStmt]) -> TypeId {
        self.checker.block_type(stmts)
    }

    /// Create a fresh inference variable with the given kind.
    pub fn new_infer_var(&mut self, kind: TypeVariableKind) -> TypeId {
        self.checker.infer.new_type_var(self.checker.ctx, kind)
    }

    /// Add a constraint to the inference context.
    pub fn add_constraint(&mut self, c: Constraint) {
        self.checker.infer.add_constraint(c);
    }

    /// Check if a cast between two types is valid.
    pub fn check_cast(&mut self, from: TypeId, to: TypeId, safe: bool, span: Span) -> Result<TypeId, Diagnostic> {
        if safe {
            if (self.ctx().is_numeric(from) && self.ctx().is_numeric(to))
                || (self.ctx().is_bool(from) && self.ctx().is_integer(to))
                || (self.ctx().is_integer(from) && self.ctx().is_bool(to))
            {
                Ok(to)
            } else if self.ctx().is_reference(from) {
                Err(Diagnostic::error("safe cast from reference type requires explicit dereference or unsafe cast")
                    .with_code("E601").with_span(span)
                    .with_suggestion("consider dereferencing first: `*expr as TargetType`")
                    .with_suggestion("or use `as!` for an unsafe bitcast"))
            } else {
                Err(Diagnostic::error("safe cast only allowed between numeric and boolean types")
                    .with_code("E601").with_span(span)
                    .with_suggestion("use `From` trait for non-primitive type conversions"))
            }
        } else {
            if (self.ctx().is_numeric(from) && self.ctx().is_numeric(to))
                || (self.ctx().is_reference(from) && self.ctx().is_pointer(to))
                || (self.ctx().is_pointer(from) && self.ctx().is_reference(to))
            {
                Ok(to)
            } else if self.ctx().is_reference(from) && self.ctx().is_integer(to) {
                Err(Diagnostic::error("unsafe cast from reference to integer not yet supported")
                    .with_code("E601").with_span(span)
                    .with_suggestion("consider using `*expr as usize` via a pointer cast"))
            } else {
                let c = self.ctx();
                match (c.get(from), c.get(to)) {
                    (TypeData::Ptr { .. }, TypeData::Ptr { .. }) => Ok(to),
                    _ => Err(Diagnostic::error("unsafe cast requires compatible types (numeric<->numeric, ref<->ptr, ptr<->ptr)")
                        .with_code("E601").with_span(span)),
                }
            }
        }
    }

    /// Infer the return type of a binary operation.
    pub fn binary_op_type(&mut self, op: BinOp, left: TypeId, right: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        self.checker.binary_op_type(op, left, right, span)
    }

    /// Check a statement (delegates to TypeChecker).
    pub fn check_stmt(&mut self, stmt: &Stmt) -> Result<HirStmt, Diagnostic> {
        self.checker.check_stmt(stmt)
    }

    /// Check a block (delegates to TypeChecker).
    pub fn check_block(&mut self, stmts: &[Stmt]) -> Result<Vec<HirStmt>, Diagnostic> {
        self.checker.check_block(stmts)
    }
}
