use super::{
    CtxKind, EffectSet, Expectation, ScopedVarMap, TypeChecker, TypingContext,
    did_you_mean_suggestion, typing_context_to_type_ctx,
};
use crate::ast::ast_type_display;
use crate::ast::visit::replace_ident_in_expr;
use crate::ast::{Attribute, BinOp, Expr, GenericArg, Literal, Pattern, Span, Stmt, Type, UnaryOp};
use crate::diagnostics::{Applicability, Diagnostic, DiagnosticKind, Suggestion, SuggestionStyle};
use crate::hir::comptime::{ComptimeEvalContext, ComptimeValue};
use crate::hir::hir::{HirCatchBranch, HirExpr, HirMatchArm, HirParam, HirPattern, HirStmt};
use crate::hir::infer::{Constraint, InferenceContext, TypeVariableKind};
use crate::hir::symbol::{TypeBinding, TypeKind};
use crate::hir::types::{Characteristic, DefId, Subst, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;

/// Default bit width for `Int` / `UInt` when no explicit width is given.
const DEFAULT_INT_WIDTH: u32 = 32;
/// Default bit width for `Float` when no explicit width is given.
const DEFAULT_FLOAT_WIDTH: u32 = 64;
/// Maximum allowed bit count for `Rational` integer/fractional parts.
const MAX_RATIONAL_BITS: u32 = 64;
/// Cap on the number of possible values considered for non-exhaustive
/// match diagnostics (beyond this, a wildcard arm is always suggested).
const EXHAUSTIVE_COUNT_CAP: usize = 256;

/// Context for type-checking a single function body.
/// Holds the mutable borrows needed for expression/statement inference
/// and checking, keeping TypeChecker focused on module-level state.
/// Following the rustc `FnCtxt` pattern — see checker/mod.rs for the
/// top-level TypeChecker that owns the global state (ctx, symbols, etc.).
pub struct FnCtxt<'a, 'tcx, 'input> {
    pub checker: &'tcx mut TypeChecker<'a, 'input>,
}

/// Whether a catch-branch pattern covers the given variant name — recurses
/// into or-patterns so `|NetworkError | TimeoutError|` matches both.
fn pattern_covers_variant(p: &crate::ast::Pattern, v: &str) -> bool {
    match p {
        crate::ast::Pattern::Enum { variant, .. } => variant.as_str() == v,
        crate::ast::Pattern::Or(pats, _) => pats.iter().any(|p| pattern_covers_variant(p, v)),
        _ => false,
    }
}

impl<'input: 'a, 'a: 'tcx, 'tcx> FnCtxt<'a, 'tcx, 'input> {
    pub fn new(checker: &'tcx mut TypeChecker<'a, 'input>) -> Self {
        FnCtxt { checker }
    }

    /// Whether a statement BLOCK diverges unconditionally (its
    /// continuation is never reached).  A block diverges if ANY statement
    /// in it diverges — a diverging statement makes everything after it
    /// unreachable, so the whole block never falls through.  (The former
    /// `Expr::If` divergence check only inspected the LAST statement,
    /// missing a `return` in the middle of a block, or a diverging call.)
    fn block_diverges(&self, stmts: &[HirStmt<'input>]) -> bool {
        stmts.iter().any(|s| self.stmt_diverges(s))
    }

    /// Whether a single statement diverges unconditionally (its successor
    /// is unreachable).  Conservative on purpose: `Loop`/`While`/`For`
    /// can be left via `leave`/`continue` or a false condition, so they
    /// are treated as non-diverging — a miss only makes the `if`-branch
    /// type merge imprecise, never unsound.
    fn stmt_diverges(&self, s: &HirStmt<'input>) -> bool {
        match s {
            HirStmt::Return { .. } | HirStmt::Leave { .. } | HirStmt::Continue { .. } => true,
            // An `if` diverges only when BOTH branches diverge.
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.block_diverges(then_branch)
                    && else_branch
                        .as_ref()
                        .is_some_and(|eb| self.block_diverges(eb))
            }
            // A call whose result type is `!` (Never) diverges — e.g.
            // `panic`/`unreachable` or an `@diverges` function.
            HirStmt::Expression(e) => self.checker.ctx.is_never(e.ty()),
            _ => false,
        }
    }

    /// Host variant of `TypeChecker::with_gadt_arm` for the EXPRESSION
    /// sites (`Expr::Match` / `Expr::IfLet`) whose arm bodies use `FnCtxt`
    /// methods (`check_block`, `block_type`, `try_gadt_discharge`).  The
    /// enter/pop/region-restore lifecycle is identical to the TypeChecker
    /// variant (single lifecycle, two hosts).
    fn with_gadt_arm<T>(
        &mut self,
        scrut_ty: TypeId,
        pattern: &crate::ast::Pattern<'input>,
        span: crate::ast::Span,
        body: impl FnOnce(&mut Self, bool) -> Result<T, Diagnostic>,
    ) -> Result<(HirPattern<'input>, T, bool), Diagnostic> {
        let _scope = self.checker.enter_var_scope();
        let (p, gadt_reachable, mut guard) =
            self.checker.begin_gadt_arm(scrut_ty, pattern, span)?;
        let t = body(self, gadt_reachable)?;
        // Pop the refinement BEFORE the caller continues (cross-arm
        // unification / else-branch) so this arm's equalities do not leak.
        // The guard's idempotent depth check handles the early-return path.
        if gadt_reachable {
            self.checker.ctx.pop_gadt_arm();
        }
        // Restore the TcLevel region entered by the arm guard (sets the
        // flag so the guard's Drop does not restore a second time).
        guard.restore_region();
        Ok((p, t, gadt_reachable))
    }

    /// Convenience accessors that delegate to the underlying TypeChecker.
    fn ctx(&mut self) -> &mut TypeContext<'input> {
        self.checker.ctx
    }
    fn infer(&mut self) -> &mut InferenceContext {
        &mut self.checker.infer
    }

    /// Suggest a cast for common type mismatches (e.g. Int ↔ Float).
    pub fn suggest_cast(&self, expected: TypeId, actual: TypeId) -> Option<Suggestion> {
        let (e, a) = (self.checker.ctx.get(expected), self.checker.ctx.get(actual));
        let msg = match (e, a) {
            (TypeData::Int { .. }, TypeData::Float { .. })
            | (TypeData::Float { .. }, TypeData::Int { .. }) => {
                Some("try using `as` to cast between integer and float types")
            }
            (TypeData::Bool, TypeData::Int { .. }) => Some("try `x != 0` to convert Int to Bool"),
            (TypeData::Int { .. }, TypeData::Bool) => {
                Some("try `if x { 1 } else { 0 }` to convert Bool to Int")
            }
            // `&mut T` where `&T` is expected: the implicit freeze is
            // rejected by default (SYNTAX.md §Reference Coercion) — point
            // at the explicit forms.
            (TypeData::Ref { mutable: false, .. }, TypeData::Ref { mutable: true, .. }) => Some(
                "use `&ro expr` or `expr.freeze!()` to surrender mutability explicitly, or annotate the function with `@auto_ro`",
            ),
            _ => None,
        };
        msg.map(|m| Suggestion {
            message: m.into(),
            applicability: Applicability::MaybeIncorrect,
            style: SuggestionStyle::ShowAlways,
        })
    }

    /// Generate a human-readable reason for a type mismatch between two
    /// types, explaining *why* they are incompatible.
    /// Delegates to [`TypeChecker::type_mismatch_reason`].
    fn type_mismatch_reason(&self, expected: TypeId, actual: TypeId) -> Option<String> {
        self.checker.type_mismatch_reason(expected, actual)
    }

    pub fn unify(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
    ) -> Result<(), Diagnostic> {
        self.checker
            .ctx
            .unify_tracked(expected, actual, span)
            .map(|_| ())
            .map_err(|_err| {
                let reason = self.type_mismatch_reason(expected, actual);
                let mut diag = Diagnostic::error_kind(DiagnosticKind::TypeMismatch {
                    expected: self
                        .checker
                        .ctx
                        .get(expected)
                        .display_with(self.checker.ctx, Some(self.checker.symbols)),
                    found: self
                        .checker
                        .ctx
                        .get(actual)
                        .display_with(self.checker.ctx, Some(self.checker.symbols)),
                    span,
                    found_span: None,
                    reason,
                    context: None,
                })
                .with_code_str("E030");
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion.message);
                }
                diag
            })
    }

    pub fn unify_with(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
        ctx: TypingContext,
    ) -> Result<(), Diagnostic> {
        self.checker
            .ctx
            .unify_tracked(expected, actual, span)
            .map(|_| ())
            .map_err(|_err| {
                // Check for the None context first — use structured kind.
                if matches!(ctx, TypingContext::None) {
                    let reason = self.type_mismatch_reason(expected, actual);
                    let type_ctx = typing_context_to_type_ctx(&ctx);
                    return Diagnostic::error_kind(DiagnosticKind::TypeMismatch {
                        expected: self
                            .checker
                            .ctx
                            .get(expected)
                            .display_with(self.checker.ctx, Some(self.checker.symbols)),
                        found: self
                            .checker
                            .ctx
                            .get(actual)
                            .display_with(self.checker.ctx, Some(self.checker.symbols)),
                        span,
                        found_span: None,
                        reason,
                        context: Some(type_ctx),
                    })
                    .with_code_str("E030");
                }
                let msg = match ctx {
                    TypingContext::ReturnValue => format!(
                        "return value type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected),
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::StructFieldInit => format!(
                        "field initializer type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected),
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::Condition => format!(
                        "condition must be boolean, got {:?}",
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::Argument { index, total } => format!(
                        "argument {} of {} has wrong type: expected {:?}, found {:?}",
                        index + 1,
                        total,
                        self.checker.ctx.get(expected),
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::ClosureBody => format!(
                        "closure body type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected),
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::None => unreachable!(), // handled above
                    TypingContext::Index => format!(
                        "index must be an integer, got {:?}",
                        self.checker.ctx.get(actual)
                    ),
                };
                let mut diag = Diagnostic::error(msg).with_code_str("E030").with_span(span);
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion.message);
                }
                diag
            })
    }

    /// Seal-the-wall discharge: when the match/if-let has a known expected
    /// type AND the scrutinee is a GADT enum, unify the arm body against
    /// the expected type WHILE the arm's facts are in scope.  Returns true
    /// if discharged.  Shared by `Expr::Match` and `Expr::IfLet` so the
    /// guards cannot drift between the two sites.
    fn try_gadt_discharge(
        &mut self,
        expected: Option<TypeId>,
        body_ty: TypeId,
        span: crate::ast::Span,
        variant: Option<&crate::ast::EnumVariant>,
    ) -> Result<bool, Diagnostic> {
        if let Some(exp) = expected {
            let exp_resolved = self.checker.ctx.resolve_binding(exp);
            let body_resolved = self.checker.ctx.resolve_binding(body_ty);
            // Only discharge when THIS arm's facts refine the expected type
            // to something concrete (a still-abstract GenericParam means the
            // arm does not constrain the parameter — unifying against it
            // would bind it globally, exactly what the seal forbids).  A
            // GenericParam body is likewise left to the status-quo path.
            let exp_abstract = matches!(
                self.checker.ctx.get_raw(exp_resolved),
                TypeData::GenericParam { .. } | TypeData::InferVar { .. }
            );
            let body_abstract = matches!(
                self.checker.ctx.get_raw(body_resolved),
                TypeData::GenericParam { .. }
            );
            // Deep guard: a compound type with an interior GenericParam the
            // current GADT context cannot refine (no fact, no binding) would
            // be bound into the global table by the in-arm unify — skip the
            // discharge in that case.
            let exp_unrefined = self
                .checker
                .ctx
                .type_contains_unrefined_generic_param(exp_resolved);
            let body_unrefined = self
                .checker
                .ctx
                .type_contains_unrefined_generic_param(body_resolved);
            if !exp_abstract && !body_abstract && !exp_unrefined && !body_unrefined {
                if let Err(mut diag) =
                    self.unify_with(exp_resolved, body_resolved, span, TypingContext::None)
                {
                    // Diagnostic attribution: the mismatch usually stems from
                    // the variant's `when` constraint refining a type parameter
                    // to a type disconnected from the payload.  Point the user
                    // at the constraint source in the variant definition rather
                    // than leaving them to blame the match arm.
                    if let Some(v) = variant {
                        for (pn, ct) in &v.eq_spec {
                            // Opaque `exists`-witness RHS: no refinement claim.
                            if let crate::ast::Type::Path(p, _) = ct
                                && p.len() == 1
                                && v.exists_params.iter().any(|ep| ep == &p[0])
                            {
                                continue;
                            }
                            diag = diag.with_secondary_label(
                                ct.span(),
                                format!(
                                    "`when {} == {}` refines `{}` here — the expected type \
                                     in this branch comes from this constraint, not from the \
                                     match arm.  If the payload type does not match, the \
                                     variant's `when` clause and payload may be disconnected.",
                                    pn,
                                    ast_type_display(ct),
                                    pn,
                                ),
                            );
                        }
                    }
                    return Err(diag);
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn infer_call_expr(
        &mut self,
        expr: &Expr<'input>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::Call {
            callee,
            args,
            comptime,
            span,
        } = expr
        else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        // ── Extract callee name once for all checks ─────────────
        let callee_name: Option<Symbol> = match callee {
            Expr::Ident(name, _) => Some(*name),
            Expr::FieldAccess { field, .. } => Some(*field),
            // Static method call: `Type::method(...)` is `Expr::Path` with len >= 2,
            // where `path[1]` is the method name.  Without this arm, `@must_handle`
            // on a static method would silently bypass the strict error-accountability
            // check (SYNTAX.md — accountability covers ALL call sites).
            Expr::Path(path, _) if path.len() >= 2 => Some(path[1]),
            _ => None,
        };
        let name_str = callee_name
            .as_ref()
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|| "this function".to_string());

        // Look up function attributes once for repeated checks.
        // ⚠️ TODO(security): For `x.foo()` (FieldAccess) forms, `lookup_function`
        //   looks up "foo" as a free function, which will likely return None
        //   for methods (registered under trait/impl).  In those cases we fall
        //   back to `(false, false)` — meaning @trusted and @io attribute checks
        //   are silently skipped for method calls.  Proper fix requires looking
        //   up method attributes through the receiver type's impl blocks.
        let (has_trusted, has_io) = callee_name
            .and_then(|name| self.checker.symbols.lookup_function(name))
            .map(|f| {
                let has_trusted = f.attributes.iter().any(|a| a.name.eq_str("trusted"));
                let has_io = f.attributes.iter().any(|a| a.name.eq_str("io"));
                (has_trusted, has_io)
            })
            .unwrap_or((false, false));

        // In strict mode, warn if we couldn't resolve the callee's attributes
        // (e.g. method call on a FieldAccess target), as sandbox checks rely
        // on knowing whether a function is @trusted or @io.
        if self.checker.strict_mode
            && callee_name.is_some()
            && self
                .checker
                .symbols
                .lookup_function(callee_name.unwrap())
                .is_none()
            && matches!(callee, Expr::FieldAccess { .. })
        {
            // Could also look up through the receiver type's inherent methods
            // here; for now just warn that the check is incomplete.
            self.checker.diagnostics.push(
                        Diagnostic::warning(format!(
                            "cannot verify @trusted/@io attributes for method `{}` — \
                             sandbox checks may be incomplete in strict mode",
                            name_str,
                        ))
                        .with_code_str("W092")
                        .with_span(*span)
                        .with_help("method attributes are resolved through trait/impl lookup, which is not yet supported here"),
                    );
        }

        // ── Comptime sandbox check ─────────────────────────────
        // Inside a comptime block, only comptime function calls
        // (marked with `!`) are allowed.  Calls without `!` are
        // rejected unless they are built-in comptime intrinsics.
        if self.checker.is_in_comptime() && !*comptime {
            if let Err(e) = self.check_call_attribute_violation(
                callee_name
                    .as_ref()
                    .unwrap_or(&Symbol::intern("this_function")),
                false,
                has_trusted,
                has_io,
                *span,
            ) {
                return Ok(e);
            }

            self.checker.diagnostics.push(
                        Diagnostic::error(format!(
                            "cannot call `{}` from comptime context: only comptime functions (called with `!`) are allowed in comptime blocks",
                            name_str,
                        ))
                        .with_code_str("E081")
                        .with_span(*span)
                        .with_suggestion(format!(
                            "use `{}!()` to call a comptime function, or move this call outside the comptime block",
                            name_str,
                        )),
                    );
            return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
        }

        // ── Isolate block check ─────────────────────────────────
        // Inside an isolate block, calling @trusted or @io functions
        // is forbidden because they may access external mutable state.
        // Also reject calls to functions known to access mutable globals.
        if self.checker.is_in_isolate() {
            if let Err(e) = self.check_call_attribute_violation(
                callee_name
                    .as_ref()
                    .unwrap_or(&Symbol::intern("this_function")),
                false,
                has_trusted,
                has_io,
                *span,
            ) {
                return Ok(e);
            }
            // Check if the callee (transitively) touches a mutable
            // global.  The pre-computed `effect_of` map covers
            // INDIRECT access too — A calls B, B reads a mutable
            // global ⇒ A's label carries MUTABLE_GLOBAL and is
            // rejected here, not just direct access (the dynamic
            // `functions_accessing_mutables` set remains as the
            // order-independent-during-checking counterpart).
            if let Some(name) = callee_name
                && let Some(binding) = self.checker.symbols.lookup_function(name)
                && (self
                    .checker
                    .effect_of
                    .get(&binding.def_id)
                    .is_some_and(|e| e.contains(EffectSet::MUTABLE_GLOBAL))
                    || self
                        .checker
                        .functions_accessing_mutables
                        .contains(&binding.def_id))
            {
                self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot call function `{}` inside isolate block: it accesses mutable global state",
                                name_str,
                            ))
                            .with_code_str("E093")
                            .with_span(*span)
                            .with_help("isolate blocks guarantee no external mutable state access; this function reads or writes mutable globals")
                        );
                return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
            }
        }

        // ── Non-comptime function with `!` check ────────────────
        // Calling `f!()` on a function that is NOT a comptime function
        // should be caught at type-checking time, not evaluation time.
        if *comptime
            && let Some(name) = callee_name
            && let Some(binding) = self.checker.symbols.lookup_function(name)
            && !binding.is_comptime
        {
            self.checker.diagnostics.push(
                        Diagnostic::error(format!(
                            "cannot call `{}!()`: `{}` is not a comptime function; remove the `!` to call it at runtime",
                            name, name,
                        ))
                        .with_code_str("E081")
                        .with_span(*span)
                        .with_help("only `comptime def` functions can be called with `!`")
                    );
            return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
        }

        // Check if this is a method call (x.foo()) rather than a free function call
        if let Expr::FieldAccess { base, field, .. } = callee {
            let (base_hir, base_ty) = self.infer_expr(base, None)?;
            // ── Method @trusted/@io attribute check ───────────
            // For method calls, look up the method's attributes through
            // inherent impl blocks, which `lookup_function` cannot find.
            if self.checker.is_in_comptime() || self.checker.is_in_isolate() {
                for method in self
                    .checker
                    .trait_env
                    .lookup_inherent_methods(base_ty, self.checker.ctx)
                {
                    if method.name == *field {
                        let has_trusted =
                            method.attributes.iter().any(|a| a.name.eq_str("trusted"));
                        let has_io = method.attributes.iter().any(|a| a.name.eq_str("io"));
                        if let Err(e) = self.check_call_attribute_violation(
                            field,
                            true,
                            has_trusted,
                            has_io,
                            *span,
                        ) {
                            return Ok(e);
                        }
                        break;
                    }
                }
                // Also check trait impl methods (`impl Trait for Type`),
                // which `lookup_inherent_methods` does not cover.
                let trait_methods: Vec<(Symbol, bool, bool)> = self
                    .checker
                    .trait_env
                    .lookup_impls_for_type(base_ty)
                    .iter()
                    .flat_map(|impl_candidate| &impl_candidate.methods)
                    .filter(|m| m.name == *field)
                    .map(|m| {
                        let has_trusted = m.attributes.iter().any(|a| a.name.eq_str("trusted"));
                        let has_io = m.attributes.iter().any(|a| a.name.eq_str("io"));
                        (m.name, has_trusted, has_io)
                    })
                    .collect::<Vec<_>>();
                for (_, has_trusted, has_io) in &trait_methods {
                    if let Err(e) = self.check_call_attribute_violation(
                        field,
                        true,
                        *has_trusted,
                        *has_io,
                        *span,
                    ) {
                        return Ok(e);
                    }
                }
            }
            // ── Method side-effect check (transitive labels) ───
            // `lookup_method` resolves the method's OWN DefId (the
            // "assoc item" identity), so the pre-computed
            // `method_effect_of` labels (keyed by method DefId) can
            // be enforced at the call site: isolate blocks forbid
            // MUTABLE_GLOBAL; @pure functions forbid the wider set.
            if let Some((_, _, method_def_id)) = self.checker.lookup_method(base_ty, *field)
                && let Some(eff) = self.checker.method_effect_of.get(&method_def_id)
            {
                if self.checker.is_in_isolate() && eff.contains(EffectSet::MUTABLE_GLOBAL) {
                    self.checker.diagnostics.push(
                                Diagnostic::error(format!(
                                    "cannot call method `{}` inside isolate block: it accesses mutable global state",
                                    field,
                                ))
                                .with_code_str("E093")
                                .with_span(*span)
                                .with_help("isolate blocks guarantee no external mutable state access; this method reads or writes mutable globals"),
                            );
                    return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
                }
                if self.checker.current_function_pure
                    && eff.intersects(
                        EffectSet::MUTABLE_GLOBAL
                            | EffectSet::IO
                            | EffectSet::UNSAFE
                            | EffectSet::PANIC
                            | EffectSet::COMPTIME,
                    )
                {
                    self.checker.diagnostics.push(
                                Diagnostic::error(format!(
                                    "`@pure` function cannot call method `{}`: it has a forbidden side effect",
                                    field,
                                ))
                                .with_code_str("E117")
                                .with_span(*span)
                                .with_help(
                                    "a @pure function must have no side effects, transitively — \
                                     remove the offending method call or drop @pure",
                                ),
                            );
                    return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
                }
            }
            if let Some((param_tys, ret_ty, _method_def_id)) =
                self.checker.lookup_method(base_ty, *field)
            {
                // Adjust: method calls pass `self` as the first arg implicitly,
                // so the param list from the declaration includes self.
                // We treat `base` as the receiver and check remaining args.
                // Unify the receiver type with the `self` parameter type.
                if !param_tys.is_empty() {
                    let self_param_ty = param_tys[0];
                    // `@auto_ro`/`@auto_coerce` applies to method
                    // receivers too (SYNTAX.md: "at function call
                    // sites and method resolution").
                    let _receiver_coercion =
                        crate::hir::types::CallSiteCoercion::enter(&self.checker.ctx);
                    // Try direct unification first (self = MyType, receiver = MyType)
                    let mut unified = self
                        .unify_with(self_param_ty, base_ty, *span, TypingContext::None)
                        .is_ok();
                    if !unified {
                        // If self param is a ref and receiver is a value, auto-ref
                        if let TypeData::Ref { ty: inner_ref, .. } =
                            self.checker.ctx.get(self_param_ty)
                        {
                            let ref_base = self.checker.ctx.reference(base_ty, false);
                            unified = self
                                .unify_with(self_param_ty, ref_base, *span, TypingContext::None)
                                .is_ok();
                        }
                    }
                }
                let explicit_param_tys = if param_tys.len() > 1 {
                    &param_tys[1..] // skip self
                } else {
                    &[] // no explicit params besides self
                };
                if explicit_param_tys.len() != args.len() {
                    self.checker.diagnostics.push(
                        Diagnostic::error(format!(
                            "wrong number of arguments: expected {}, found {}",
                            explicit_param_tys.len(),
                            args.len()
                        ))
                        .with_span(*span),
                    );
                }
                // `@auto_ro`'s `&mut T → &T` relaxation applies ONLY
                // at call sites (SYNTAX.md) — mark this arg-check as
                // a call-site unification.
                let hir_args = self.check_call_args(args, &explicit_param_tys, *span)?;
                // Build the HIR: the callee is the field access; we keep it as-is
                let callee_hir = HirExpr::FieldAccess {
                    base: Box::new(base_hir),
                    field: *field,
                    ty: ret_ty,
                    span: *span,
                };
                return Ok((
                    HirExpr::Call {
                        callee: Box::new(callee_hir),
                        args: hir_args,
                        comptime: *comptime,
                        ty: ret_ty,
                        span: *span,
                    },
                    ret_ty,
                ));
            } else {
                // Method not found — collect available method names for a helpful error
                let mut method_names: Vec<Symbol> = Vec::new();
                for ty in self.checker.autoderef_chain(base_ty) {
                    for cand in self.checker.trait_env.lookup_impls_for_type(ty) {
                        for m in &cand.methods {
                            if !method_names.contains(&m.name) {
                                method_names.push(m.name);
                            }
                        }
                    }
                }
                let mut diag =
                    Diagnostic::error(format!("no method named `{}` found for type", field))
                        .with_code_str("E011")
                        .with_span(*span);
                if !method_names.is_empty() {
                    diag = diag.with_suggestion(format!(
                        "available methods: {}",
                        method_names
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                self.checker.diagnostics.push(diag);
                return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
            }
        }

        // Check if this is a static method call: `Type::method(args)`
        if let Expr::Path(path, _) = callee
            && path.len() >= 2
        {
            // Resolve the type from the first path segment.
            let type_name = path[0];
            let method_name = path[1];
            let type_path = Type::Path(smallvec::smallvec![type_name], *span);
            if let Ok(ty) = self.resolve_type(&type_path) {
                // Look up the method on the resolved type.
                // lookup_method also handles inherent methods.
                if let Some((param_tys, ret_ty, _method_def_id)) =
                    self.checker.lookup_method(ty, method_name)
                {
                    // Static method call: no self parameter to skip.
                    // The method's param_tys already reflect the full signature.
                    if param_tys.len() != args.len() {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "wrong number of arguments: expected {}, found {}",
                                param_tys.len(),
                                args.len()
                            ))
                            .with_span(*span),
                        );
                    }
                    let hir_args = self.check_call_args(args, &param_tys, *span)?;
                    let callee_hir = HirExpr::Ident(method_name, ret_ty, *span);
                    return Ok((
                        HirExpr::Call {
                            callee: Box::new(callee_hir),
                            args: hir_args,
                            comptime: *comptime,
                            ty: ret_ty,
                            span: *span,
                        },
                        ret_ty,
                    ));
                }
            }
            // If type resolution or method lookup fails, fall through to
            // normal call handling — infer_expr(Path) will produce a
            // diagnostic about the unresolved path.
        }

        let (callee_hir, callee_ty) = self.infer_expr(callee, None)?;

        // Try local type argument synthesis first: detect polymorphic functions
        // whose parameter types contain GenericParam (type variables that need
        // to be inferred from argument types).
        match self.checker.try_synthesize_type_args(
            &callee_hir,
            callee_ty,
            args,
            *comptime,
            None,
            *span,
        ) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => { /* not polymorphic, fall through */ }
            Err(diag) => {
                self.checker.diagnostics.push(diag);
                return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
            }
        }

        // Normal (non-polymorphic) function call — peel any Forall wrapping
        let inner_call_ty = {
            let mut t = callee_ty;
            loop {
                match self.checker.ctx.get(t) {
                    TypeData::Forall { body, .. } => t = *body,
                    _ => break,
                }
            }
            t
        };
        if let Some(params) = self.checker.ctx.params_of_fn(inner_call_ty) {
            let param_tys = params.to_vec();
            let ret_ty = self
                .checker
                .ctx
                .ret_of_fn(inner_call_ty)
                .unwrap_or(self.checker.ctx.error());
            if param_tys.len() != args.len() {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "wrong number of arguments: expected {}, found {}",
                        param_tys.len(),
                        args.len()
                    ))
                    .with_span(*span),
                );
            }
            let hir_args = self.check_call_args(args, &param_tys, *span)?;
            Ok((
                HirExpr::Call {
                    callee: Box::new(callee_hir),
                    args: hir_args,
                    comptime: *comptime,
                    ty: ret_ty,
                    span: *span,
                },
                ret_ty,
            ))
        } else {
            self.checker
                .diagnostics
                .push(Diagnostic::error("called expression is not a function").with_span(*span));
            Ok((HirExpr::Error(*span), self.checker.ctx.error()))
        }
    }

    fn infer_match_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::Match {
            scrutinee,
            arms,
            span,
        } = expr
        else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee, None)?;
        let mut hir_arms = Vec::new();
        let mut arm_ty = None;
        // Span of the first non-discharging arm that accumulated into
        // arm_ty — used as the error location for the retroactive
        // expected-type check (more precise than the whole-match span).
        let mut first_pre_discharge_span = None;
        // ── Seal-the-wall ────────────────────────────
        // When this match has a known expected type (e.g.
        // `return match ...` against a declared `-> T`) AND the
        // scrutinee is a GADT enum, each arm's body is discharged
        // against the expected type WHILE the arm's facts are in
        // scope — GHC's checkConstraints / OCaml's
        // `type_expect ext_env (mk_expected ty_expected)` model.
        // With the facts active, resolve_binding rewrites
        // GenericParam{T} → Int<32> via the fact registry, so the
        // unify never writes a GenericParam into the global
        // bindings table (guarded by the seal assert in
        // `set_binding`).  Once an arm discharges this way, the
        // match RESULT becomes the expected type itself, making
        // check_expr's post-match `unify_with(expected, result)`
        // an identity instead of a global GenericParam binding.
        // Non-GADT matches keep the status-quo cross-arm
        // accumulation untouched.
        let mut gadt_discharged = false;
        let scrut_is_gadt = self
            .checker
            .lookup_type_binding(self.checker.ctx.resolve_binding(scrut_ty))
            .map_or(false, |b| b.variants.iter().any(|v| v.is_gadt()));
        for arm in arms {
            // GADT refinements are scoped to each arm via the
            // equality registry (see push_gadt_arm).
            // Each arm gets its own scope so that binding a type
            // parameter (e.g., T → Int<32>) in arm 1 does not
            // affect arm 2.  No transaction/rollback needed.
            let _scope = self.checker.enter_var_scope();
            // Shared GADT arm lifecycle (enter/pop/region-restore
            // encapsulated in `with_gadt_arm` — same sequence as the
            // other three pattern-matching sites).
            let (
                pattern_hir,
                (guard_hir, body_hir, arm_body_ty, this_arm_discharged),
                gadt_reachable,
            ) = self.with_gadt_arm(scrut_ty, &arm.pattern, *span, |fctx, gadt_reachable| {
                // Always type-check the guard and body, regardless of
                // reachability, to ensure consistent diagnostics across
                // match, if-let, and while-let.
                let guard_hir = arm
                    .guard
                    .as_ref()
                    .map(|g| {
                        fctx.infer_expr(g, None).map(|(h, ty)| {
                            if !fctx.checker.ctx.is_bool(ty) {
                                fctx.checker.diagnostics.push(
                                    Diagnostic::error("match guard must be boolean")
                                        .with_span(arm.span),
                                );
                            }
                            Box::new(h)
                        })
                    })
                    .transpose()?;
                let (body_hir, body_ty) = fctx.infer_expr(&arm.body, None)?;
                // Seal: per-arm in-scope discharge (BEFORE pop, so
                // the arm's facts are still visible to resolve_binding).
                let variant = fctx
                    .checker
                    .resolve_gadt_variant_info(scrut_ty, &arm.pattern, arm.span)
                    .map(|(_, vd, _)| vd);
                let this_arm_discharged = gadt_reachable
                    && scrut_is_gadt
                    && fctx.try_gadt_discharge(expected, body_ty, arm.span, variant.as_ref())?;
                if this_arm_discharged {
                    gadt_discharged = true;
                }
                // Resolve the body type — follows bindings to the
                // concrete type where available (normal resolution,
                // no longer a GADT-specific capture hack).
                let arm_body_ty = fctx.checker.ctx.resolve_binding(body_ty);
                Ok((guard_hir, body_hir, arm_body_ty, this_arm_discharged))
            })?;
            // Cross-arm unification intentionally runs with TcLevel
            // escape checking DISABLED (`region_tree: None` below):
            // `arm_body_ty` is NOT always fully resolved (e.g. bool
            // literals leave unbound InferVars), and unification
            // binds them here.  No defensive assert — it would fire
            // on legitimate programs.
            hir_arms.push(HirMatchArm {
                pattern: pattern_hir,
                guard: guard_hir,
                body: Box::new(body_hir),
                span: arm.span,
            });
            // Include in cross-arm unification only if reachable,
            // and only when no arm discharged against a shared
            // expected type (GHC/OCaml never unify arm types
            // against each other — each arm satisfies the expected
            // type under its own givens).
            if gadt_reachable {
                if !gadt_discharged {
                    // A diverging arm (`leave with`/`return` — type
                    // `Never`) contributes no value to the cross-arm
                    // type; skip unifying it with the accumulator.
                    if self.checker.ctx.is_never(arm_body_ty) {
                        // diverging arm — non-contributing
                    } else if let Some(prev) = arm_ty {
                        self.unify_with(prev, arm_body_ty, arm.span, TypingContext::None)?;
                    } else {
                        arm_ty = Some(arm_body_ty);
                        first_pre_discharge_span = Some(arm.span);
                    }
                } else if !this_arm_discharged {
                    // A prior arm discharged against the shared expected type,
                    // but THIS arm did not — its body is otherwise never
                    // checked (the cross-arm unify above is skipped once
                    // gadt_discharged).  Match GHC's per-arm expected-type
                    // check: unify the arm body against the expected type
                    // (post-pop, no facts).  A still-abstract GenericParam
                    // expected with a concrete body fires E104 (generality).
                    // gadt_discharged implies expected is Some (the first
                    // discharge required it, and expected is loop-invariant).
                    debug_assert!(expected.is_some());
                    if let Some(exp) = expected {
                        let exp_resolved = self.checker.ctx.resolve_binding(exp);
                        // Direct per-arm mismatch: an abstract GenericParam
                        // expected with a CONCRETE body is rejected HERE —
                        // relying on the deferred E104 would mask this
                        // mismatch for const/where-constrained params (E104
                        // exempts them), silently accepting an ill-typed arm.
                        let exp_abstract = matches!(
                            self.checker.ctx.get_raw(exp_resolved),
                            TypeData::GenericParam { .. }
                        );
                        let body_abstract = matches!(
                            self.checker.ctx.get_raw(arm_body_ty),
                            // An unresolved InferVar is NOT provably
                            // concrete — it may yet resolve to the
                            // expected GenericParam, so it must not
                            // trigger the "produces a concrete type"
                            // E030 prematurely (mirrors the discharge
                            // guard's `exp_abstract`, which includes
                            // InferVar).
                            TypeData::GenericParam { .. } | TypeData::InferVar { .. }
                        );
                        if exp_abstract && !body_abstract {
                            return Err(Diagnostic::error(
                                "match arm does not satisfy the expected type: the arm \
                                         produces a concrete type where the generic parameter \
                                         is expected; each arm must type-check for every \
                                         instantiation",
                            )
                            .with_code_str("E030")
                            .with_span(arm.span));
                        }
                        self.unify_with(exp_resolved, arm_body_ty, arm.span, TypingContext::None)?;
                    }
                }
            }
        }
        let result_ty = if gadt_discharged {
            let exp = expected.expect("gadt_discharged implies an expected type");
            // Retroactively check any arms that accumulated BEFORE the
            // first discharge: their body types were stored in arm_ty
            // but never checked against the expected type once the
            // latch flipped.  A still-abstract GenericParam expected
            // with a concrete accumulated body fires E104 (generality).
            if let Some(accumulated) = arm_ty {
                let exp_resolved = self.checker.ctx.resolve_binding(exp);
                // Same direct per-arm mismatch check as the post-discharge
                // branch (see above): do not rely on the deferred E104,
                // which exempts const/where-constrained params.
                let exp_abstract = matches!(
                    self.checker.ctx.get_raw(exp_resolved),
                    TypeData::GenericParam { .. }
                );
                let acc_abstract = matches!(
                    self.checker.ctx.get_raw(accumulated),
                    TypeData::GenericParam { .. }
                );
                if exp_abstract && !acc_abstract {
                    return Err(Diagnostic::error(
                        "match arm does not satisfy the expected type: the arm \
                                 produces a concrete type where the generic parameter \
                                 is expected; each arm must type-check for every \
                                 instantiation",
                    )
                    .with_code_str("E030")
                    .with_span(first_pre_discharge_span.unwrap_or(*span)));
                }
                self.unify_with(
                    exp_resolved,
                    accumulated,
                    first_pre_discharge_span.unwrap_or(*span),
                    TypingContext::None,
                )?;
            }
            exp
        } else {
            // Divergence discipline (mirrors the Expr::If
            // convention): if EVERY arm body is `Never` (all arms
            // `return`/`leave`), the match itself diverges and
            // must have the bottom type `!` — not `()` (a `!`
            // context would otherwise get a spurious mismatch).
            arm_ty.unwrap_or(self.checker.ctx.never())
        };

        // ── Exhaustiveness check ────────────────────────────
        // Check that all enum variants or finite values are covered
        // by the match arms (unless `_` wildcard present).
        // Use resolve_binding to see through any InferVar bindings.
        let resolved_scrut_ty = self.checker.ctx.resolve_binding(scrut_ty);
        let has_wildcard = hir_arms
            .iter()
            .any(|a| matches!(a.pattern, HirPattern::Wildcard(_)));

        if !has_wildcard {
            // Enumerate checked variants/patterns from all arms
            let mut covered_variants: Vec<String> = Vec::new();
            for arm in &hir_arms {
                match &arm.pattern {
                    HirPattern::Enum { variant, .. } => {
                        if !covered_variants.contains(&variant.as_str()) {
                            covered_variants.push(variant.as_str());
                        }
                    }
                    HirPattern::Or(patterns, _) => {
                        for p in patterns {
                            if let HirPattern::Enum { variant, .. } = p
                                && !covered_variants.contains(&variant.as_str())
                            {
                                covered_variants.push(variant.as_str());
                            }
                        }
                    }
                    HirPattern::Literal(expr, _) => {
                        if let HirExpr::Literal(lit, _, _) = expr.as_ref() {
                            let lit_key = format!("{:?}", lit);
                            if !covered_variants.contains(&lit_key) {
                                covered_variants.push(lit_key);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Path A: type has explicit enum variants
            if let Some(binding) = self.checker.lookup_type_binding(resolved_scrut_ty) {
                // ── GADT dead variant elimination ────────────
                // For GADT enums, filter out variants whose
                // eq_spec contradicts the scrutinee's type args.
                let is_gadt = binding.variants.iter().any(|v| v.is_gadt());
                let reachable_variants: Vec<String> = if is_gadt {
                    binding
                        .variants
                        .iter()
                        .filter_map(|v| {
                            // Reuse is_gadt_variant_reachable which correctly
                            // handles both regular and existential GADT constraints
                            // (unlike a manual resolve_type chain that would fail
                            // on existential param names).
                            let pattern = crate::ast::Pattern::Enum {
                                path: Vec::new(),
                                variant: v.name,
                                inner: None,
                                span: *span,
                            };
                            if self.checker.is_gadt_variant_reachable(
                                resolved_scrut_ty,
                                &pattern,
                                *span,
                            ) {
                                Some(v.name.as_str().to_string())
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    binding
                        .variants
                        .iter()
                        .map(|v| v.name.as_str().to_string())
                        .collect()
                };
                let total_variants = reachable_variants.len();
                // Path A.2: @exhaustive forbids wildcard
                if binding.exhaustive && has_wildcard && total_variants > 0 {
                    self.checker.diagnostics.push(
                                Diagnostic::error(
                                    "`@exhaustive` enum does not allow `_` wildcard; list all variants explicitly"
                                ).with_span(*span)
                            );
                }
                // Compute which reachable variants are actually covered by user arms.
                // Use set difference (not length comparison) because an arm for an
                // unreachable GADT variant inflates covered_variants without covering
                // any reachable variant.
                let missing: Vec<&str> = reachable_variants
                    .iter()
                    .filter_map(|v| {
                        if !covered_variants.contains(v) {
                            Some(v.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if !missing.is_empty() {
                    let msg = binding.missing_match.clone().unwrap_or_else(|| {
                                format!(
                                    "non-exhaustive match: missing variants `{}`; add missing arms or a `_` wildcard",
                                    missing.join(", "),
                                )
                            });
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error(msg).with_span(*span));
                }
            }

            // Path B: small finite type with literal patterns (Bool, etc.)
            // Use characteristic κ after resolving inference variables.
            // For InferVars, also check the variable kind directly
            // (characteristic returns usize::MAX for unresolved infer vars).
            let char = self.checker.ctx.characteristic(resolved_scrut_ty);
            let total_count_from_char = match char {
                Characteristic::FiniteExhaustible(n) => Some(n),
                _ => None,
            };
            let inferred_count: Option<usize> = match self.checker.ctx.get(resolved_scrut_ty) {
                TypeData::InferVar { id, .. } => match self.checker.infer.get_var_kind(*id) {
                    Some(TypeVariableKind::Bool) => Some(2),
                    _ => None,
                },
                _ => None,
            };
            // inferred_count takes priority over characteristic for unresolved vars
            let total_count = inferred_count.or(total_count_from_char);
            // GADT enums are handled by Path A (set-difference with
            // dead-variant elimination) — the value count below would
            // include unreachable variants, producing false
            // non-exhaustiveness for GADT scrutinees.
            match total_count {
                Some(n)
                    if !scrut_is_gadt
                        && n <= EXHAUSTIVE_COUNT_CAP
                        && covered_variants.len() < (n as usize) =>
                {
                    let msg = format!(
                        "non-exhaustive match: covered {}/{} possible values; add more arms or a `_` wildcard",
                        covered_variants.len(),
                        n,
                    );
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error(msg).with_span(*span));
                }
                _ => {}
            }
        }

        Ok((
            HirExpr::Match {
                scrutinee: Box::new(scrut_hir),
                arms: hir_arms,
                ty: result_ty,
                span: *span,
            },
            result_ty,
        ))
    }

    fn infer_enum_lit_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::EnumLit {
            path,
            variant,
            payload,
            span,
        } = expr
        else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        let resolved_ty =
            self.resolve_type(&Type::Path(smallvec::SmallVec::from(path.clone()), *span))?;
        let (def_id, mut args) = self
            .checker
            .resolve_type_to_struct_or_enum(resolved_ty, *span)?;
        // The bare path (`Expr::Lit(...)`) carries no type arguments,
        // so `args` above is empty for a generic enum.  Recover the
        // arguments from the EXPECTED type (e.g. `Expr<T>` when the
        // literal is returned from `fn mk<T>() -> Expr<T>`) so the
        // GADT construction validation below sees the real `T`
        // instead of an empty args list.
        if args.is_empty()
            && let Some(exp) = expected
            && let Ok((exp_def_id, exp_args)) =
                self.checker.resolve_type_to_struct_or_enum(exp, *span)
            && exp_def_id == def_id
        {
            args = exp_args;
        }
        let binding = self
            .checker
            .symbols
            .lookup_type_by_def_id(def_id)
            .ok_or_else(|| Diagnostic::error("type not found").with_span(*span))?;
        // If the type is not an enum, or if the variant is not found among
        // the enum's variants, treat this as a static method call instead.
        if !matches!(binding.kind, TypeKind::Enum)
            || !binding.variants.iter().any(|v| v.name == *variant)
        {
            // Static method call: `Type::method(args)`
            // The payload (if any) is the argument expression.
            if let Some((method_param_tys, ret_ty, _method_def_id)) =
                self.checker.lookup_method(resolved_ty, *variant)
            {
                let mut hir_args = Vec::new();
                // Pass the payload (if any) as the argument.
                if let Some(p) = &payload {
                    let expected = method_param_tys
                        .first()
                        .copied()
                        .unwrap_or(self.checker.ctx.error());
                    let _enum_guard =
                        crate::hir::types::StructuralCoercion::enter(&self.checker.ctx);
                    let hir_arg = self.check_expr(
                        p,
                        Expectation::HasType(expected),
                        TypingContext::Argument { index: 0, total: 1 },
                    )?;
                    hir_args.push(hir_arg);
                }
                let callee_hir = HirExpr::Ident(*variant, ret_ty, *span);
                return Ok((
                    HirExpr::Call {
                        callee: Box::new(callee_hir),
                        args: hir_args,
                        comptime: false,
                        ty: ret_ty,
                        span: *span,
                    },
                    ret_ty,
                ));
            }
            // Fall through: not an enum and not a method — produce a diagnostic below.
            if !matches!(binding.kind, TypeKind::Enum) {
                return Err(Diagnostic::error("not an enum type").with_span(*span));
            }
        }
        let enum_ty = self.checker.ctx.enum_ty(def_id, args.clone());
        let mut subst = Subst::new();
        for (i, _param) in binding.params.iter().enumerate() {
            if let Some(&arg) = args.get(i) {
                subst.insert(i, arg);
            }
        }
        let variant_def = binding
            .variants
            .iter()
            .find(|v| v.name == *variant)
            .ok_or_else(|| {
                Diagnostic::error(format!("variant '{}' not found", variant)).with_span(*span)
            })?;
        // For existential variants, create fresh InferVars for each
        // exist param.  These are shared between the GADT constraint
        // validation and the payload type resolution so that the same
        // (now bound) variables are used in both places.  Stored by
        // binder INDEX (identity, not name — GHC/OCaml convention).
        let mut exist_vars: Vec<TypeId> = Vec::new();
        for _ep in &variant_def.exists_params {
            let var = self.new_infer_var(
                TypeVariableKind::Any,
                crate::hir::infer::VarOrigin::Synthetic,
            );
            exist_vars.push(var);
        }
        // ── GADT construction validation ──────────────────────────
        // Check that the target type's actual type parameters satisfy
        // the variant's `when` constraints (eq_spec).
        // NOTE: re-resolve into a distinct binding so the
        // expected-type-recovered `args` above is not shadowed.
        let (_, validation_args) = self
            .checker
            .resolve_type_to_struct_or_enum(enum_ty, *span)?;
        if !exist_vars.is_empty() {
            // Wrap the constraint loop in a transaction so that
            // partial bindings from try_unify are rolled back if
            // any constraint fails.
            self.checker.ctx.begin_transaction();
            let mut constraint_ok = true;
            for (param_name, concrete_ty) in &variant_def.eq_spec {
                // Check if the constraint targets an enum type parameter
                // (e.g., `T == [X]` where T is an enum param).
                if let Some((param_idx, _)) = binding
                    .params
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.name == *param_name)
                {
                    // Fail-closed: if the constraint's type-argument
                    // slot is absent, the constraint cannot be
                    // verified — reject rather than silently accept
                    // (SYNTAX.md: "The compiler verifies at every
                    // construction site ... Violation results in a
                    // compile-time error").
                    let Some(&actual_arg) = validation_args.get(param_idx) else {
                        constraint_ok = false;
                        break;
                    };
                    let declared = match self.checker.resolve_type_with_skolems(
                        concrete_ty,
                        &variant_def.exists_params,
                        &exist_vars,
                    ) {
                        Some(d) => d,
                        None => match self.resolve_type(concrete_ty) {
                            Ok(d) => d,
                            Err(_) => {
                                constraint_ok = false;
                                break;
                            }
                        },
                    };
                    if self
                        .checker
                        .ctx
                        .try_unify(declared, actual_arg, Some(&self.checker.infer.region_tree))
                        .is_err()
                    {
                        constraint_ok = false;
                        break;
                    }
                } else if let Some(&var) = variant_def
                    .exists_params
                    .iter()
                    .position(|p| p == param_name)
                    .and_then(|i| exist_vars.get(i))
                {
                    let declared = match self.checker.resolve_type_with_skolems(
                        concrete_ty,
                        &variant_def.exists_params,
                        &exist_vars,
                    ) {
                        Some(d) => d,
                        None => match self.resolve_type(concrete_ty) {
                            Ok(d) => d,
                            Err(_) => {
                                constraint_ok = false;
                                break;
                            }
                        },
                    };
                    if self
                        .checker
                        .ctx
                        .try_unify(declared, var, Some(&self.checker.infer.region_tree))
                        .is_err()
                    {
                        constraint_ok = false;
                        break;
                    }
                }
            }
            if !constraint_ok {
                self.checker.ctx.rollback_transaction();
                return Err(Diagnostic::error(format!(
                    "GADT constraint on `{}` not satisfied",
                    variant,
                ))
                .with_code_str("E060")
                .with_span(*span));
            }
            // If any scrutinee type argument is an unresolved variable
            // (GenericParam / InferVar), the constraint check would
            // need to persist `T := Int<32>` to be sound.  Committing
            // pollutes the outer generic; rolling back and accepting
            // forgets the constraint (unsound: `mk<T>()` would be
            // accepted for every T).  Reject the construction so the
            // GADT invariant is preserved — callers must pass
            // concrete type arguments.
            // The committee ruling (solve → default → validate):
            // unresolved type arguments at construction time are NO
            // longer rejected here — the construction is deferred to
            // the post-solve validation phase.
            if args
                .iter()
                .any(|&a| self.checker.type_has_unresolved_vars(a))
            {
                self.checker.pending_gadt_constructs.push(
                    crate::hir::checker::PendingGadtConstruct {
                        enum_ty,
                        variant: *variant,
                        span: *span,
                    },
                );
            }
            self.checker.ctx.commit_transaction();
        } else {
            // Non-existential GADT construction validation.
            // Wrap the whole conjunction in a single transaction
            // so that bindings from earlier constraints are visible
            // to later ones (avoiding per-constraint rollback).
            self.checker.ctx.begin_transaction();
            let mut constraint_ok = true;
            for (param_name, concrete_ty) in &variant_def.eq_spec {
                if let Some((param_idx, _)) = binding
                    .params
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.name == *param_name)
                    && let Some(&actual_arg) = validation_args.get(param_idx)
                {
                    let declared = match self.resolve_type(concrete_ty) {
                        Ok(d) => d,
                        Err(_) => {
                            constraint_ok = false;
                            break;
                        }
                    };
                    if self
                        .checker
                        .ctx
                        .try_unify(declared, actual_arg, Some(&self.checker.infer.region_tree))
                        .is_err()
                    {
                        constraint_ok = false;
                        break;
                    }
                }
            }
            if !constraint_ok {
                self.checker.ctx.rollback_transaction();
                return Err(Diagnostic::error(format!(
                    "GADT constraint on `{}` not satisfied",
                    variant,
                ))
                .with_code_str("E060")
                .with_span(*span));
            }
            // Same as the existential branch: unresolved scrutinee
            // args must be rejected, not silently accepted after
            // rollback (which would forget the GADT constraint and
            // accept `mk<T>()` for every T).
            // The committee ruling (solve → default → validate):
            // unresolved type arguments at construction time are NO
            // longer rejected here — the construction is deferred to
            // the post-solve validation phase.
            if args
                .iter()
                .any(|&a| self.checker.type_has_unresolved_vars(a))
            {
                self.checker.pending_gadt_constructs.push(
                    crate::hir::checker::PendingGadtConstruct {
                        enum_ty,
                        variant: *variant,
                        span: *span,
                    },
                );
            }
            self.checker.ctx.commit_transaction();
        }

        // Resolve the payload type, substituting type params with concrete args.
        // For example, `Option<T>` with `T = Int<32>` means the payload type
        // `T` should resolve to the `GenericParam` TypeId, which will be
        // unified with the concrete arg via the subst.
        //
        // For existential GADT variants (`Slice(exists X: &[X])`), the
        // payload type references the exist param name (`X`), which is
        // not in any symbol table.  Use the construction's own
        // exist_vars (InferVars created above) — NOT the enclosing
        // match arm's stored skolems — to resolve the payload type.
        let payload_ty = variant_def
            .payload
            .as_ref()
            .map(|ty| {
                // If the payload type is a bare type param name (e.g. `T` in
                // `type Option<T> = enum { None, Some(T) }`), resolve it to
                // the corresponding GenericParam TypeId so that substitution
                // with the concrete args works correctly.
                if let Type::Path(p, _) = ty {
                    if p.len() == 1
                        && let Some((i, _)) = binding
                            .params
                            .iter()
                            .enumerate()
                            .find(|(_, tp)| tp.name == p[0])
                    {
                        let gp = self.checker.ctx.generic_param(i, p[0]);
                        let result = self.checker.ctx.subst(gp, &subst);
                        return Ok(result);
                    }
                    // Check if it's an exist param.
                    if p.len() == 1
                        && let Some(&var) = variant_def
                            .exists_params
                            .iter()
                            .position(|ep| ep == &p[0])
                            .and_then(|i| exist_vars.get(i))
                    {
                        return Ok(var);
                    }
                }
                // Use exist_vars for compound types containing exist params.
                if !exist_vars.is_empty()
                    && let Some(d) = self.checker.resolve_type_with_skolems(
                        ty,
                        &variant_def.exists_params,
                        &exist_vars,
                    )
                {
                    return Ok(d);
                }
                self.resolve_type(ty)
            })
            .transpose()?
            .unwrap_or(self.checker.ctx.error());
        let payload_hir = if let Some(payload) = payload {
            let hir = self.check_expr(
                payload,
                Expectation::HasType(payload_ty),
                TypingContext::StructFieldInit,
            )?;
            self.unify_with(payload_ty, hir.ty(), *span, TypingContext::StructFieldInit)?;
            Some(Box::new(hir))
        } else {
            None
        };
        Ok((
            HirExpr::EnumLit {
                path: path.clone(),
                variant: *variant,
                payload: payload_hir,
                ty: enum_ty,
                span: *span,
            },
            enum_ty,
        ))
    }

    fn infer_try_expr(
        &mut self,
        expr: &Expr<'input>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::Try { expr, span } = expr else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        let (hir, ty) = self.infer_expr(expr, None)?;

        // ── @must_handle check ────────────────────────────────────
        // Propagating a must_handle'd error via `?` does not count
        // as handling it.  The caller must use `catch` or
        // `@delegates_must_handle`.
        let callee_name: Option<Symbol> = match expr {
            Expr::Ident(name, _) => Some(*name),
            Expr::Call { callee, .. } => match callee {
                Expr::Ident(name, _) => Some(*name),
                // Method call: `obj.fetch()` is `Call { callee: FieldAccess { field } }`.
                Expr::FieldAccess { field, .. } => Some(*field),
                // Static method call: `Type::method(...)` is `Expr::Path` with len >= 2,
                // where `path[1]` is the method name.  Without this arm, `@must_handle`
                // on a static method would silently bypass the strict error-accountability
                // check (SYNTAX.md — accountability covers ALL call sites).
                Expr::Path(path, _) if path.len() >= 2 => Some(path[1]),
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = callee_name
            && let Some(binding) = self.checker.symbols.lookup_function(name)
            && binding
                .attributes
                .iter()
                .any(|a| a.name.eq_str("must_handle"))
        {
            let msg = format!(
                "call to `{}` has `@must_handle` variants that are propagated via `?`",
                name,
            );
            let diag = if self.checker.strict_mode {
                Diagnostic::error(msg)
                    .with_code_str("E108")
                    .with_help("strict mode requires local handling for `@must_handle` variants")
                    .with_span(*span)
            } else {
                Diagnostic::warning(msg)
                            .with_code_str("W004")
                            .with_help("use `catch` to explicitly handle `@must_handle` variants before `?`")
                            .with_suggestion(format!(
                                "add `catch |variant| {{ ... }}` before `?` to handle the required variants, or add `@delegates_must_handle` to this function"
                            ))
                            .with_span(*span)
            };
            self.checker.diagnostics.push(diag);
        } else if let Expr::Call { callee, .. } = expr
            && let Expr::FieldAccess { field, .. } = callee
            && {
                // Diagnostic-only path: extract the base type from the
                // ALREADY inferred HIR (`hir` — the main
                // `infer_expr(expr, None)` at the top) — NO
                // re-inference, NO transaction, NO side-effect leak
                // (the old rolled-back re-inference leaked
                // InferenceContext variables and duplicated
                // diagnostics — no side-effect leak).
                let base_ty = match &hir {
                    HirExpr::Call { callee, .. } => match callee.as_ref() {
                        HirExpr::FieldAccess { base, .. } => Some(base.ty()),
                        _ => None,
                    },
                    _ => None,
                };
                match base_ty {
                    Some(base_ty) => {
                        // Inherent OR trait methods annotated
                        // `@must_handle` (SYNTAX.md — accountability
                        // covers ALL call sites).
                        let inherent = self
                            .checker
                            .trait_env
                            .lookup_inherent_methods(base_ty, self.checker.ctx)
                            .iter()
                            .any(|m| {
                                m.name == *field
                                    && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            });
                        let trait_impl = self
                            .checker
                            .trait_env
                            .lookup_impls_for_type(base_ty)
                            .iter()
                            .flat_map(|ic| &ic.methods)
                            .any(|m| {
                                m.name == *field
                                    && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            });
                        inherent || trait_impl
                    }
                    None => false,
                }
            }
        {
            // Method call to a `@must_handle` method — fire the check.
            let msg = format!(
                "call to method `{}` has `@must_handle` variants that are propagated via `?`",
                field,
            );
            let diag = if self.checker.strict_mode {
                Diagnostic::error(msg)
                    .with_code_str("E108")
                    .with_span(*span)
            } else {
                Diagnostic::warning(msg)
                    .with_code_str("W004")
                    .with_span(*span)
            };
            self.checker.diagnostics.push(diag);
        } else if let Expr::Ident(name, _) = expr
            && self.checker.must_handle_sources.borrow().contains(name)
        {
            // The `?` operand is a variable that was assigned from
            // a `@must_handle` call — the check must fire even
            // though the variable name doesn't match the function.
            let msg = format!(
                "variable `{}` holds a `@must_handle` result that is propagated via `?`",
                name,
            );
            let diag = if self.checker.strict_mode {
                Diagnostic::error(msg)
                    .with_code_str("E108")
                    .with_span(*span)
            } else {
                Diagnostic::warning(msg)
                    .with_code_str("W004")
                    .with_span(*span)
            };
            self.checker.diagnostics.push(diag);
        }

        let ok_ty = self.checker.check_result_type(ty, *span)?;
        Ok((
            HirExpr::Try {
                expr: Box::new(hir),
                ty: ok_ty,
                span: *span,
            },
            ok_ty,
        ))
    }

    fn infer_catch_expr(
        &mut self,
        expr: &Expr<'input>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::Catch {
            expr,
            branches,
            span,
        } = expr
        else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        let (expr_hir, expr_ty) = self.infer_expr(expr, None)?;
        let (ok_ty, error_ty) = self.checker.extract_result_types(expr_ty, *span)?;

        // ── @must_handle check ────────────────────────────────────
        // Check that all `@must_handle` variants are explicitly
        // handled in the catch branches (not just a wildcard).
        let callee_name: Option<Symbol> = match expr {
            Expr::Call { callee, .. } => match callee {
                Expr::Ident(name, _) => Some(*name),
                // Method call: `obj.fetch()` is `Call { callee: FieldAccess { field } }`.
                Expr::FieldAccess { field, .. } => Some(*field),
                // Static method call: `Type::method(...)` is `Expr::Path` with len >= 2,
                // where `path[1]` is the method name.  Without this arm, `@must_handle`
                // on a static method would silently bypass the strict error-accountability
                // check (SYNTAX.md — accountability covers ALL call sites).
                Expr::Path(path, _) if path.len() >= 2 => Some(path[1]),
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = callee_name
            && let Some(binding) = self.checker.symbols.lookup_function(name)
        {
            // Extract the variant names from @must_handle(...) attribute
            let must_handle_variants: Vec<String> = binding
                .attributes
                .iter()
                .filter(|a| a.name.eq_str("must_handle"))
                .flat_map(|a| &a.args)
                .filter_map(|arg| {
                    if let crate::ast::Expr::Ident(sym, _) = arg {
                        Some(sym.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            if !must_handle_variants.is_empty() {
                // Find which must_handle variants are missing specific branches
                let missing: Vec<&str> = must_handle_variants
                    .iter()
                    .filter(|v| {
                        // Check: is there at least one branch with this variant name?
                        // Recurses into or-patterns — a variant caught
                        // via `|NetworkError | TimeoutError|` must not
                        // be reported as missing.
                        !branches
                            .iter()
                            .any(|b| pattern_covers_variant(&b.pattern, v.as_str()))
                    })
                    .map(|s| s.as_str())
                    .collect();
                if !missing.is_empty() {
                    let msg = format!(
                        "call to `{}` has `@must_handle` variants not explicitly caught: {}",
                        name,
                        missing.join(", "),
                    );
                    // Cap the suggestion so a `@must_handle` list
                    // with many variants does not balloon the
                    // diagnostic output (the message above still
                    // lists every missing variant).
                    const SUGGESTION_CAP: usize = 5;
                    let catch_arms: Vec<String> = missing
                        .iter()
                        .map(|v| format!("|{}| {{ ... }}", v))
                        .collect();
                    let suggestion = if catch_arms.len() > SUGGESTION_CAP {
                        format!(
                            "add catch arms: {} ... and {} more",
                            catch_arms[..SUGGESTION_CAP].join(", "),
                            catch_arms.len() - SUGGESTION_CAP,
                        )
                    } else {
                        format!("add catch arms: {}", catch_arms.join(", "))
                    };
                    let diag = if self.checker.strict_mode {
                        Diagnostic::error(msg)
                                        .with_code_str("E108")
                                        .with_help(
                                            "strict mode requires explicit local handling for `@must_handle` variants",
                                        )
                                        .with_span(*span)
                    } else {
                        Diagnostic::warning(msg)
                            .with_code_str("W005")
                            .with_help(
                                "add explicit catch branches for each `@must_handle` variant",
                            )
                            .with_suggestion(suggestion)
                            .with_span(*span)
                    };
                    self.checker.diagnostics.push(diag);
                }
            } else if binding
                .attributes
                .iter()
                .any(|a| a.name.eq_str("must_handle"))
            {
                // Bare @must_handle (no specific variants): warn if only wildcard catch
                if branches
                    .iter()
                    .all(|b| matches!(&b.pattern, crate::ast::Pattern::Wildcard(_)))
                {
                    self.checker.diagnostics.push(
                                    Diagnostic::warning(format!(
                                        "call to `{}` has `@must_handle` but catch only uses wildcard",
                                        name,
                                    ))
                                    .with_code_str("W006")
                                    .with_help("add explicit catch branches for the required variants")
                                    .with_suggestion("use specific catch arms like `|Variant| { ... }` instead of `|_| { ... }`")
                                    .with_span(*span),
                                );
                }
            }
        } else if let Expr::Call { callee, .. } = expr
            && let Expr::FieldAccess { base, field, .. } = callee
            && {
                // Method call — look up `@must_handle` through the
                // receiver type's impl blocks (mirror of the
                // `Expr::Try` method-call check; SYNTAX.md —
                // accountability covers ALL call sites).
                // Diagnostic-only path: extract the receiver type
                // from the ALREADY inferred HIR (`expr_hir` — the
                // `infer_expr(expr, None)` at the top of this arm) —
                // NO re-inference, NO transaction, NO side-effect
                // leak (the old rolled-back re-inference leaked
                // InferenceContext variables and duplicated
                // diagnostics — no side-effect leak).
                let base_ty = match &expr_hir {
                    HirExpr::Call { callee, .. } => match callee.as_ref() {
                        HirExpr::FieldAccess { base, .. } => Some(base.ty()),
                        _ => None,
                    },
                    _ => None,
                };
                match base_ty {
                    Some(base_ty) => {
                        let inherent = self
                            .checker
                            .trait_env
                            .lookup_inherent_methods(base_ty, self.checker.ctx)
                            .iter()
                            .any(|m| {
                                m.name == *field
                                    && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            });
                        let trait_impl = self
                            .checker
                            .trait_env
                            .lookup_impls_for_type(base_ty)
                            .iter()
                            .flat_map(|ic| &ic.methods)
                            .any(|m| {
                                m.name == *field
                                    && m.attributes.iter().any(|a| a.name.eq_str("must_handle"))
                            });
                        inherent || trait_impl
                    }
                    None => false,
                }
            }
        {
            // Method call to a `@must_handle` method — fire the
            // accountability check (the marked variants are not
            // provably handled by an explicit catch branch).
            let msg = format!(
                "call to method `{}` has `@must_handle` variants used in `catch` without explicit handling",
                field,
            );
            let diag = if self.checker.strict_mode {
                Diagnostic::error(msg)
                    .with_code_str("E108")
                    .with_span(*span)
            } else {
                Diagnostic::warning(msg)
                    .with_code_str("W005")
                    .with_span(*span)
            };
            self.checker.diagnostics.push(diag);
        } else if let Expr::Ident(name, _) = expr
            && self.checker.must_handle_sources.borrow().contains(name)
        {
            // The `catch` operand is a variable that was assigned
            // from a `@must_handle` call — the check must fire even
            // though the variable name doesn't match the function
            // (mirror of the `Expr::Try` variable-based check).
            let msg = format!(
                "variable `{}` holds a `@must_handle` result used in `catch`",
                name,
            );
            let diag = if self.checker.strict_mode {
                Diagnostic::error(msg)
                    .with_code_str("E108")
                    .with_span(*span)
            } else {
                Diagnostic::warning(msg)
                    .with_code_str("W005")
                    .with_span(*span)
            };
            self.checker.diagnostics.push(diag);
        }

        let mut hir_branches = Vec::new();
        for branch in branches {
            let _scope = self.checker.enter_var_scope();
            let exist_depth = self.checker.ctx.gadt.exist_skolems.borrow().len();
            let pattern_hir = self.check_pattern(&branch.pattern, error_ty, exist_depth)?;
            let body_hir = self.check_block(&branch.body)?;
            // SYNTAX.md §Error Handling: "Each branch of `catch`
            // must either diverge or produce a value of type `T`."
            // Unify the branch body type with `ok_ty` unless the
            // branch diverges (type `Never`).
            let body_ty = self.block_type(&body_hir);
            if !self.checker.ctx.is_never(body_ty) {
                self.unify_with(ok_ty, body_ty, branch.span, TypingContext::None)?;
            }
            hir_branches.push(HirCatchBranch {
                pattern: pattern_hir,
                bind: branch.bind,
                body: body_hir,
                span: branch.span,
            });
            // scope drops here — removes pattern + body bindings
        }
        Ok((
            HirExpr::Catch {
                expr: Box::new(expr_hir),
                branches: hir_branches,
                ty: ok_ty,
                span: *span,
            },
            ok_ty,
        ))
    }

    fn infer_if_let_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        let Expr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            is_expression,
            span,
        } = expr
        else {
            unreachable!("infer_expr dispatch guarantees the expression variant");
        };
        // The parser only constructs `Expr::IfLet` in expression
        // position (statement-position `if let` is `Stmt::IfLet`),
        // so `is_expression` is always true here — the statement
        // branch below is dead by construction.
        debug_assert!(*is_expression, "Expr::IfLet is always an expression");
        // SYNTAX.md: "When used as an expression, an `else` branch is mandatory."
        if *is_expression && else_branch.is_none() {
            return Err(Diagnostic::error(
                "`if let` expression without `else` branch is not allowed",
            )
            .with_help("add an `else` branch: `if let Pattern = expr { body } else { fallback }`")
            .with_span(*span));
        }
        let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee, None)?;
        // ── Seal-the-wall ────────────────────────────
        // Same per-branch in-scope discharge as Expr::Match: when
        // the if-let has a known expected type and the scrutinee
        // is a GADT enum, the then-branch's result is discharged
        // against the expected type WHILE the arm's facts are in
        // scope, and the expression's result type becomes the
        // expected type itself (check_expr's post-match unify is
        // then an identity — no global GenericParam binding).
        let scrut_is_gadt = self
            .checker
            .lookup_type_binding(self.checker.ctx.resolve_binding(scrut_ty))
            .map_or(false, |b| b.variants.iter().any(|v| v.is_gadt()));
        let mut gadt_discharged = false;
        // Enter scope so the pattern binding is scoped to the then-branch
        // (lifecycle encapsulated in `with_gadt_arm`).
        let (pattern_hir, then_hir, then_ty, _gadt_reachable) = {
            let (p, (t, ty), gadt_reachable) =
                self.with_gadt_arm(scrut_ty, pattern, *span, |fctx, gadt_reachable| {
                    let t = fctx.check_block(then_branch)?;
                    // Seal: in-scope discharge BEFORE pop (facts visible).
                    let variant = fctx
                        .checker
                        .resolve_gadt_variant_info(scrut_ty, pattern, *span)
                        .map(|(_, vd, _)| vd);
                    if gadt_reachable
                        && scrut_is_gadt
                        && fctx.try_gadt_discharge(
                            expected,
                            fctx.block_type(&t),
                            *span,
                            variant.as_ref(),
                        )?
                    {
                        gadt_discharged = true;
                    }
                    // Resolve the then-branch type — resolves InferVars to
                    // their bound concrete types (normal resolution, not a
                    // GADT-specific hack — the bindings persist because we
                    // no longer roll back after each arm).
                    let ty = fctx.checker.ctx.resolve_binding(fctx.block_type(&t));
                    Ok((t, ty))
                })?;
            (p, t, ty, gadt_reachable)
        }; // _scope dropped inside with_gadt_arm: pattern + then-branch bindings removed
        let else_hir = else_branch
            .as_ref()
            .map(|b| self.check_block(b))
            .transpose()?;
        let ty = if *is_expression {
            // Expression position: unify branch types (else is required
            // and checked above).
            if let Some(ref else_stmts) = else_hir {
                let else_ty = self.block_type(else_stmts);
                let then_diverges = self.checker.ctx.is_never(then_ty);
                let else_diverges = self.checker.ctx.is_never(else_ty);
                // Divergence discipline (Expr::If convention): a
                // `Never` branch contributes no value — skip
                // unification and adopt the other branch's type.
                if !then_diverges && !else_diverges {
                    if gadt_discharged {
                        // Discharge already validated the then-branch
                        // against `expected` IN-SCOPE.  Check the
                        // else branch against `expected` instead of
                        // against `then_ty` — the latter carries a
                        // POPPED GADT refinement (arm_depth == 0, so
                        // the seal would not intercept a GenericParam
                        // binding — a false E104 leak).
                        if let Some(exp) = expected {
                            let exp_resolved = self.checker.ctx.resolve_binding(exp);
                            self.unify_with(exp_resolved, else_ty, *span, TypingContext::None)?;
                        }
                    } else {
                        self.unify_with(then_ty, else_ty, *span, TypingContext::None)?;
                    }
                }
                if gadt_discharged {
                    // Then-branch was discharged against the expected
                    // type in-scope; the result IS the expected type
                    // so check_expr's post-unify is an identity.
                    expected.expect("gadt_discharged implies expected is Some")
                } else if then_diverges {
                    else_ty
                } else {
                    then_ty
                }
            } else {
                then_ty
            }
        } else {
            // Statement position: type is always unit.
            self.checker.ctx.unit()
        };
        Ok((
            HirExpr::IfLet {
                pattern: pattern_hir,
                scrutinee: Box::new(scrut_hir),
                then_branch: then_hir,
                else_branch: else_hir,
                ty,
                span: *span,
            },
            ty,
        ))
    }

    // ── Infer expression type ─────────────────────────────────────────────
    // `expected` is the type expected from the surrounding context (from
    // `check_expr`'s `Expectation`).  It drives context-sensitive inference
    // (e.g. EnumLit / Call / StructLit type-argument synthesis) the same way
    // mainstream industrial compilers propagate expected types (see ante's
    // `infer_expr(item.expr, expected_type)`).  Internal recursion passes
    // `None` for sub-expressions whose expected type is not yet derived.
    //
    // DEEP-RECURSION GUARD (explicit stack): nested source-level calls
    // expand SEVERAL checker frames per call (infer_expr → infer_call_expr
    // → check_call_args → check_call_argument → infer_expr), so a 10-level
    // chain previously overflowed the test runner's default 2 MiB thread
    // stack.  `stacker::maybe_grow` (the same crate rustc uses for deep
    // recursion) checks the remaining stack and grows it on demand, so
    // every recursive call into this dispatcher is stack-safe regardless
    // of the caller's thread-stack default.
    #[must_use]
    pub fn infer_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        // Red zone 256 KiB: when the remaining stack drops below this, grow
        // by a 16 MiB segment (amortized — only triggered on deep chains).
        stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || {
            self.infer_expr_inner(expr, expected)
        })
    }

    /// The recursive inference core, called through the `infer_expr`
    /// stack-growth wrapper above.
    fn infer_expr_inner(
        &mut self,
        expr: &Expr<'input>,
        expected: Option<TypeId>,
    ) -> Result<(HirExpr<'input>, TypeId), Diagnostic> {
        match expr {
            Expr::Literal(lit, span) => {
                match lit {
                    Literal::Int(_) => {
                        let ty = self.new_infer_var(
                            TypeVariableKind::Integer,
                            crate::hir::infer::VarOrigin::Expression(Some(*span)),
                        );
                        // Fix A: the literal unifies with an integer
                        // expected type (e.g. the index expectation) —
                        // otherwise the inference variable dangles and the
                        // integer checks (is_integer) fail on the literal.
                        if let Some(e) = expected {
                            if self.checker.ctx.is_integer(e) {
                                self.unify_with(e, ty, *span, TypingContext::None)?;
                            }
                        }
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                    Literal::Float(_) => {
                        let ty = self.new_infer_var(
                            TypeVariableKind::Float,
                            crate::hir::infer::VarOrigin::Expression(Some(*span)),
                        );
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                    Literal::Bool(_) => {
                        let ty = self.checker.ctx.bool();
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                    Literal::Char(_) => {
                        let ty = self.checker.ctx.char();
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                    Literal::String(_) => {
                        // String literals have type `&Str` (SYNTAX.md).
                        // This is a reference to the built-in `Str` type,
                        // not `&[Byte]` (which is for byte string literals).
                        let ty = self.checker.ctx.str_ref();
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                    Literal::ByteString(_) => {
                        // Byte string literals have type `&[Byte]` (SYNTAX.md).
                        let ty = self.checker.ctx.slice(self.checker.ctx.byte());
                        Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
                    }
                }
            }
            Expr::Ident(name, span) => {
                // Reading a frozen place is enforced by the flow-sensitive
                // point-level borrow-check post-pass (see the checker
                // FunctionDef<'input> arm).
                // Check the local variable type cache first (set by VariableDef)
                if let Some(ty) = self.checker.local_variable_types.get(*name) {
                    // Reading a mutable global outside @trusted is forbidden
                    if self.checker.mutable_globals.contains(name)
                        && !self.checker.current_function_trusted
                    {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot read mutable global `{}` outside `@trusted` function",
                                name,
                            ))
                            .with_code_str("E040")
                            .with_span(*span)
                            .with_help("wrap the function in `@trusted` and add `requires`/`ensures` contracts")
                        );
                    }
                    // Track which functions access mutable globals (for isolate checking)
                    if self.checker.mutable_globals.contains(name)
                        && let Some(def_id) = self.checker.current_function
                    {
                        self.checker.functions_accessing_mutables.insert(def_id);
                    }
                    // Reading a mutable global inside an isolate block is also forbidden
                    if self.checker.mutable_globals.contains(name) && self.checker.is_in_isolate() {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot read mutable global `{}` inside isolate block",
                                name,
                            ))
                            .with_code_str("E093")
                            .with_span(*span)
                            .with_help("isolate blocks must not access external mutable state"),
                        );
                    }
                    Ok((HirExpr::Ident(*name, ty, *span), ty))
                } else if let Some(binding) = self.checker.symbols.lookup_variable(*name, *span) {
                    Ok((HirExpr::Ident(*name, binding.ty, *span), binding.ty))
                } else if let Some(func) = self.checker.symbols.lookup_function(*name) {
                    // ── @deprecated / @experimental check ───────────────
                    self.check_deprecated_experimental(name, &func.attributes, *span);

                    let sig = &func.signature;
                    // Construct the function type: Fn(params..., ret)
                    let mut fn_ty = self.checker.ctx.function(
                        sig.params.iter().map(|p| p.ty).collect(),
                        sig.return_type.get(),
                    );
                    // If the function has type parameters, wrap with Forall:
                    // def foo<T, U>(x: T, y: U) → Forall(0, "T", Forall(1, "U", Fn(...)))
                    if !sig.type_params.is_empty() {
                        for (i, tp) in sig.type_params.iter().enumerate().rev() {
                            fn_ty = self.checker.ctx.forall(i, tp.name, fn_ty);
                        }
                    }
                    Ok((HirExpr::Ident(*name, fn_ty, *span), fn_ty))
                } else {
                    self.checker.diagnostics.push(
                        Diagnostic::error(format!("undefined name: {}", name)).with_span(*span),
                    );
                    Ok((HirExpr::Error(*span), self.checker.ctx.error()))
                }
            }
            Expr::TypeAnnotated { expr, ty, span } => {
                let expected = self.resolve_type(ty)?;
                let hir =
                    self.check_expr(expr, Expectation::HasType(expected), TypingContext::None)?;
                Ok((
                    HirExpr::TypeAnnotated {
                        expr: Box::new(hir),
                        ty: expected,
                        span: *span,
                    },
                    expected,
                ))
            }
            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let (left_hir, left_ty) = self.infer_expr(left, None)?;
                let (right_hir, right_ty) = self.infer_expr(right, None)?;
                let result_ty = self.checker.binary_op_type(
                    *op,
                    left_ty,
                    right_ty,
                    Some(left.span()),
                    Some(right.span()),
                    *span,
                )?;
                Ok((
                    HirExpr::BinaryOp {
                        left: Box::new(left_hir),
                        op: *op,
                        right: Box::new(right_hir),
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::UnaryOp { op, expr, span } => {
                // A borrow operand (`&mut a` / `&ro r`) is not a READ —
                // suppress the read-side freeze while checking it.  The
                // The operand is inferred directly — the old read-side
                // suppression counter is removed (it was
                // incremented/decremented but never read).
                let operand_res = self.infer_expr(expr, None);
                let (hir, ty) = operand_res?;
                let result_ty = match op {
                    UnaryOp::Neg | UnaryOp::BitNot => ty,
                    UnaryOp::Not => self.checker.ctx.bool(),
                    UnaryOp::Deref => self
                        .checker
                        .ctx
                        .pointee_of_ref(ty)
                        .or_else(|| self.checker.ctx.pointee_of_pointer(ty))
                        // The explicit `*x` form through a user-defined
                        // `Deref` impl (SYNTAX.md §Method-Call Auto-
                        // Dereferencing: "method calls through the wrapper
                        // type require explicit `(*x).method()` syntax" —
                        // the unmarked impls).  No `@auto_deref` required
                        // for the EXPLICIT operator.
                        .or_else(|| self.checker.deref_trait_step(ty, false))
                        .unwrap_or(self.checker.ctx.error()),
                    UnaryOp::Ref | UnaryOp::RefMut => {
                        let mutable = matches!(op, UnaryOp::RefMut);
                        // `&mut p` is an exclusive borrow (SYNTAX.md
                        // §References): while it is live, the source place
                        // is frozen — neither readable nor writable — the
                        // freeze (E109/E110) is enforced by the
                        // flow-sensitive post-pass (the explicit
                        // loan registration here is removed — the post-pass
                        // re-collects the loans from the HIR).
                        self.checker.ctx.reference(ty, mutable)
                    }
                    UnaryOp::Ro => {
                        // `&ro r` freezes a `&mut T` into a `&T` (SYNTAX.md
                        // §Reference Coercion): the operand must be a
                        // reference; its mutability is surrendered.  The
                        // source place is FROZEN against mutation for the
                        // borrow's lifetime — register a read-only loan so
                        // a later mutation of the place is rejected.
                        let resolved = self.checker.ctx.resolve_binding(ty);
                        match self.checker.ctx.get(resolved) {
                            TypeData::Ref {
                                ty: inner, mutable, ..
                            } => {
                                // `&ro`'s core purpose is the `&mut T` → `&T`
                                // coercion.  On an already-immutable
                                // reference it is harmless but redundant —
                                // allow but warn (W114).
                                if !mutable {
                                    self.checker.diagnostics.push(
                                        Diagnostic::warning(
                                            "`&ro` on an already-immutable reference is redundant — the operand is already `&T`",
                                        )
                                        .with_code_str("W114")
                                        .with_span(*span),
                                    );
                                }
                                let ty = self.checker.ctx.reference(*inner, false);
                                ty
                            }
                            _ => {
                                self.checker.diagnostics.push(
                                    Diagnostic::error("`&ro` requires a reference operand")
                                        .with_code_str("E111")
                                        .with_span(*span),
                                );
                                self.checker.ctx.error()
                            }
                        }
                    }
                };
                Ok((
                    HirExpr::UnaryOp {
                        op: *op,
                        expr: Box::new(hir),
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::Call { .. } => self.infer_call_expr(expr),
            Expr::Index { base, index, span } => {
                let (base_hir, base_ty) = self.infer_expr(base, None)?;
                // Eager resolution: resolve the base through the GADT fact
                // registry before element lookup — the same discipline as
                // field/method access and binary_op_type operands.  A refined
                // generic param (e.g. `xs: T` with arm fact `T → &[Int<32>]`)
                // must resolve to the slice so indexing works on it.
                let base_ty = self.checker.ctx.resolve_binding(base_ty);
                // Fix A: the index must be an integer — propagate the
                // expected integer type so a literal index (`arr[2]`)
                // unifies with a concrete integer instead of leaving its
                // inference variable dangling (the E030 false positive).
                let (index_hir, index_ty) =
                    self.infer_expr(index, Some(self.checker.ctx.usize()))?;
                // Resolve the inference variable — the literal unifies
                // with the expected usize; the check must see the resolved
                // concrete integer type.
                let index_ty = self.checker.ctx.resolve_binding(index_ty);
                let elem_ty = self
                    .checker
                    .ctx
                    .elem_of_slice(base_ty)
                    .or_else(|| self.checker.ctx.elem_of_array(base_ty))
                    .unwrap_or_else(|| {
                        self.checker.diagnostics.push(
                            Diagnostic::error("indexing on non-array/non-slice type")
                                .with_span(*span),
                        );
                        self.checker.ctx.error()
                    });
                if !self.checker.ctx.is_integer(index_ty) && !self.checker.ctx.is_usize(index_ty) {
                    self.checker.diagnostics.push(
                        Diagnostic::error("index must be an integer")
                            .with_code_str("E030")
                            .with_span(*span)
                            .with_label(
                                index.span(),
                                format!("got {:?}", self.checker.ctx.get(index_ty)),
                            ),
                    );
                }
                Ok((
                    HirExpr::Index {
                        base: Box::new(base_hir),
                        index: Box::new(index_hir),
                        ty: elem_ty,
                        span: *span,
                    },
                    elem_ty,
                ))
            }
            Expr::FieldAccess { base, field, span } => {
                let (base_hir, base_ty) = self.infer_expr(base, None)?;
                // ── Eager resolution: resolve the receiver through
                // the GADT fact registry BEFORE field/method lookup — the
                // same discipline binary_op_type applies to operands.  A
                // refined generic param (e.g. `x: T` with arm fact
                // `T → Int<32>`) must resolve to the concrete type so the
                // method is found on `Int<32>` instead of failing on `T`.
                let base_ty = self.checker.ctx.resolve_binding(base_ty);
                // Try to resolve as a struct field first
                if let Ok(field_ty) = self.checker.lookup_field(base_ty, *field, *span) {
                    return Ok((
                        HirExpr::FieldAccess {
                            base: Box::new(base_hir),
                            field: *field,
                            ty: field_ty,
                            span: *span,
                        },
                        field_ty,
                    ));
                }
                // If not a field, try as a method via autoderef
                if let Some((param_tys, ret_ty, _method_def_id)) =
                    self.checker.lookup_method(base_ty, *field)
                {
                    // Full function type including self parameter: fn(&Obj) -> RetTy
                    let fn_ty = self.checker.ctx.function(param_tys, ret_ty);
                    return Ok((
                        HirExpr::FieldAccess {
                            base: Box::new(base_hir),
                            field: *field,
                            ty: fn_ty,
                            span: *span,
                        },
                        fn_ty,
                    ));
                }
                Err(
                    Diagnostic::error(format!("no field or method '{}' on this type", field))
                        .with_span(*span),
                )
            }
            Expr::AttrAccess { base, attr, span } => {
                let (base_hir, base_ty) = self.infer_expr(base, None)?;
                // Eager resolution: resolve the base through the GADT fact
                // registry before attribute lookup (same discipline as
                // field/method access and binary_op_type operands).
                let base_ty = self.checker.ctx.resolve_binding(base_ty);
                let attr_ty = self.checker.lookup_attr(base_ty, *attr, *span)?;
                Ok((
                    HirExpr::AttrAccess {
                        base: Box::new(base_hir),
                        attr: *attr,
                        ty: attr_ty,
                        span: *span,
                    },
                    attr_ty,
                ))
            }
            Expr::Cast {
                expr,
                ty,
                safe,
                rounding,
                span,
            } => {
                let (hir, actual_ty) = self.infer_expr(expr, None)?;
                let target_ty = self.resolve_type(ty)?;
                let cast_ty = self.check_cast(actual_ty, target_ty, *safe, *span)?;
                Ok((
                    HirExpr::Cast {
                        expr: Box::new(hir),
                        ty: cast_ty,
                        safe: *safe,
                        rounding: *rounding,
                        span: *span,
                    },
                    cast_ty,
                ))
            }
            Expr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                // Keep the endpoints' INFERRED types instead of hardcoding
                // `(Int<32>, Int<32>)` — a `Float`/`UInt` range must not
                // lose its element type.
                let (start_hir, start_ty) = start
                    .as_ref()
                    .map(|s| self.infer_expr(s, None))
                    .transpose()?
                    .map(|(h, t)| (Some(h), Some(t)))
                    .unwrap_or((None, None));
                let (end_hir, end_ty) = end
                    .as_ref()
                    .map(|e| self.infer_expr(e, None))
                    .transpose()?
                    .map(|(h, t)| (Some(h), Some(t)))
                    .unwrap_or((None, None));
                // Range endpoints must share a type: prefer the START's
                // inferred type and unify the END against it (reporting a
                // mismatch), fall back to the END's type, then to the
                // default signed Int width for a fully unbounded `..`.
                let elem_ty = match (start_ty, end_ty) {
                    (Some(st), Some(et)) => {
                        if let Err(_e) = self.checker.ctx.unify_tracked(st, et, *span) {
                            self.checker.diagnostics.push(
                                Diagnostic::error("range endpoints must have the same type")
                                    .with_span(*span)
                                    .with_label(
                                        *span,
                                        format!(
                                            "start: {}, end: {}",
                                            self.checker.ctx.get(st).display_with(
                                                self.checker.ctx,
                                                Some(self.checker.symbols)
                                            ),
                                            self.checker.ctx.get(et).display_with(
                                                self.checker.ctx,
                                                Some(self.checker.symbols)
                                            ),
                                        ),
                                    ),
                            );
                            self.checker.ctx.error()
                        } else {
                            st
                        }
                    }
                    (Some(st), None) => st,
                    (None, Some(et)) => et,
                    (None, None) => self.checker.ctx.int(DEFAULT_INT_WIDTH, true),
                };
                let ty = self.checker.ctx.tuple(vec![elem_ty, elem_ty]);
                Ok((
                    HirExpr::Range {
                        start: start_hir.map(Box::new),
                        end: end_hir.map(Box::new),
                        inclusive: *inclusive,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::StructLit { path, fields, span } => {
                let resolved_ty =
                    self.resolve_type(&Type::Path(smallvec::SmallVec::from(path.clone()), *span))?;
                let (def_id, args) = self
                    .checker
                    .resolve_type_to_struct_or_enum(resolved_ty, *span)?;
                let binding = self
                    .checker
                    .symbols
                    .lookup_type_by_def_id(def_id)
                    .ok_or_else(|| Diagnostic::error("struct not found").with_span(*span))?;
                if !matches!(binding.kind, TypeKind::Struct) {
                    return Err(Diagnostic::error("not a struct type").with_span(*span));
                }
                let struct_ty = self.checker.ctx.struct_ty(def_id, args.clone());
                let mut subst = Subst::new();
                for (i, _param) in binding.params.iter().enumerate() {
                    if let Some(&arg) = args.get(i) {
                        subst.insert(i, arg);
                    }
                }
                let mut hir_fields = Vec::new();
                for (name, value) in fields {
                    let field_def =
                        binding
                            .fields
                            .iter()
                            .find(|f| f.name == *name)
                            .ok_or_else(|| {
                                let field_names: Vec<String> =
                                    binding.fields.iter().map(|f| f.name.as_str()).collect();
                                let type_name = format!("{:?}", def_id);
                                let mut diag =
                                    Diagnostic::error_kind(DiagnosticKind::NoSuchField {
                                        field_name: name.to_string(),
                                        type_name,
                                        span: *span,
                                    })
                                    .with_code_str("E010")
                                    .with_suggestion(format!(
                                        "available fields: {}",
                                        field_names.join(", ")
                                    ));
                                if let Some(suggestion) =
                                    did_you_mean_suggestion(&name.as_str(), &field_names)
                                {
                                    diag = diag.with_suggestion(suggestion);
                                }
                                diag
                            })?;
                    let field_ty = self.checker.ctx.subst(field_def.ty, &subst);
                    let _struct_guard =
                        crate::hir::types::StructuralCoercion::enter(&self.checker.ctx);
                    let hir = self.check_expr(
                        value,
                        Expectation::HasType(field_ty),
                        TypingContext::StructFieldInit,
                    )?;
                    self.unify_with(field_ty, hir.ty(), *span, TypingContext::StructFieldInit)?;
                    hir_fields.push((*name, Box::new(hir)));
                }
                Ok((
                    HirExpr::StructLit {
                        path: path.clone(),
                        fields: hir_fields,
                        ty: struct_ty,
                        span: *span,
                    },
                    struct_ty,
                ))
            }
            Expr::EnumLit { .. } => self.infer_enum_lit_expr(expr, expected),
            Expr::Move(expr, span) => {
                let (hir, ty) = self.infer_expr(expr, None)?;
                Ok((HirExpr::Move(Box::new(hir), ty, *span), ty))
            }
            Expr::Tuple(exprs, span) => {
                let mut hirs = Vec::new();
                let mut types = Vec::new();
                for e in exprs {
                    let (hir, ty) = self.infer_expr(e, None)?;
                    hirs.push(hir);
                    types.push(ty);
                }
                let ty = self.checker.ctx.tuple(types);
                Ok((HirExpr::Tuple(hirs, ty, *span), ty))
            }
            Expr::Array(exprs, span) => {
                let mut hirs = Vec::new();
                let mut elem_ty = None;
                let _array_guard = crate::hir::types::StructuralCoercion::enter(&self.checker.ctx);
                for e in exprs {
                    let (hir, ty) = self.infer_expr(e, None)?;
                    if let Some(et) = elem_ty {
                        self.unify_with(et, ty, *span, TypingContext::None)?;
                    } else {
                        elem_ty = Some(ty);
                    }
                    hirs.push(hir);
                }
                let ty = self.checker.ctx.array(
                    elem_ty.unwrap_or(self.checker.ctx.error()),
                    exprs.len() as u64,
                );
                Ok((HirExpr::Array(hirs, ty, *span), ty))
            }
            Expr::Closure {
                params,
                return_type,
                captures,
                body,
                span,
            } => {
                let mut hir_params = Vec::new();
                let mut param_tys = Vec::new();
                for param in params {
                    let ty = param
                        .ty
                        .as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or_else(|| {
                            Ok(self.new_infer_var(
                                TypeVariableKind::Any,
                                crate::hir::infer::VarOrigin::Expression(Some(param.span)),
                            ))
                        })?;
                    hir_params.push(HirParam {
                        name: param.name,
                        ty,
                        default: None,
                        span: param.span,
                    });
                    param_tys.push(ty);
                }
                // Enter a variable scope for closure parameters so they don't leak
                // into the enclosing function's scope.
                let _closure_scope = self.checker.enter_var_scope();
                for p in &hir_params {
                    self.checker.local_variable_types.insert(p.name, p.ty);
                }
                self.checker.push_ctx(CtxKind::Closure, *span, None);
                let body_hir = self.check_block(body)?;
                let body_ty = self.block_type(&body_hir);
                self.checker.pop_ctx();
                // _closure_scope dropped here — removes closure parameter bindings
                let ret_ty = match return_type {
                    Some(ty) => {
                        let declared = self.resolve_type(ty)?;
                        self.checker.unify_with(
                            declared,
                            body_ty,
                            *span,
                            TypingContext::ClosureBody,
                        )?;
                        declared
                    }
                    None => body_ty,
                };
                let ty = self.checker.ctx.function(param_tys, ret_ty);
                Ok((
                    HirExpr::Closure {
                        params: hir_params,
                        return_type: ret_ty,
                        captures: captures.clone(),
                        body: body_hir,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::Try { .. } => self.infer_try_expr(expr),
            Expr::UnsafeBlock { body, span } => {
                // Strict Mode forbids `unsafe` blocks completely
                // (SYNTAX.md §Strict Mode).
                if self.checker.strict_mode {
                    self.checker.diagnostics.push(
                        Diagnostic::error("`unsafe` blocks are forbidden in Strict Mode")
                            .with_span(*span),
                    );
                }
                let body_hir = self.check_block(body)?;
                let ty = self.checker.ctx.unit();
                Ok((
                    HirExpr::UnsafeBlock {
                        body: body_hir,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::Catch { .. } => self.infer_catch_expr(expr),
            Expr::LeaveWith {
                expr,
                is_return,
                span,
            } => {
                let (hir, err_ty) = self.infer_expr(expr, None)?;
                if let Some(ret_ty) = self.checker.current_return_type {
                    if *is_return {
                        // `return expr` — unify with the full return type.
                        // SYNTAX.md §"Error Handling": "Using `return Err(e)`
                        // in place of `leave with` is a compile-time error."
                        // Unified with the `Stmt::Return` path via
                        // `TypeChecker::is_result_err_constructor` (alias
                        // chain of `Result` — multi-level aliases cannot
                        // bypass the lint).
                        if let crate::hir::hir::HirExpr::EnumLit { variant, path, .. } = &hir
                            && self.checker.is_result_err_constructor(path, variant)
                        {
                            // Push the lint and CONTINUE (uniform error
                            // recovery with the `Stmt::Return` path —
                            // follow-on errors are still reported).
                            self.checker.emit_return_err_lint(*span);
                        }
                        // Detect `return` without value in expression position:
                        // the parser synthesizes an empty tuple `()` as the
                        // expression.  Emit the same E003 diagnostic as the
                        // `Stmt::Return` path instead of a confusing E030 —
                        // BUT only when the return type actually requires a
                        // value: a value-less `return` is legal for `()`- and
                        // `!`-returning functions (mirror of `Stmt::Return`).
                        if matches!(hir, HirExpr::Tuple(ref elems, _, _) if elems.is_empty())
                            && !self.checker.ctx.is_unit(ret_ty)
                            && !self.checker.ctx.is_never(ret_ty)
                        {
                            return Err(Diagnostic::error(
                                "return without value in a function that expects a return value",
                            )
                            // The semantic code for a return
                            // type mismatch is E036 — E003 is "unexpected
                            // token" (parser).
                            .with_code_str("E036")
                            .with_span(*span));
                        }
                        self.unify_with(ret_ty, err_ty, *span, TypingContext::ReturnValue)?;
                    } else {
                        // `leave with expr` — unify with the error type of Result<T, E>
                        if let Ok((_, error_ty)) = self.checker.extract_result_types(ret_ty, *span)
                        {
                            self.unify_with(error_ty, err_ty, *span, TypingContext::None)?;
                        } else {
                            return Err(Diagnostic::error(
                                "`leave with` requires the enclosing function to return `Result<T, E>`"
                            )
                            .with_span(*span)
                            .with_help(
                                "change the function's return type to `Result<T, E>` or use `return` instead"
                            ));
                        }
                    }
                }
                let never = self.checker.ctx.never();
                if *is_return {
                    Ok((
                        HirExpr::Return {
                            value: Box::new(hir),
                            ty: never,
                            span: *span,
                        },
                        never,
                    ))
                } else {
                    Ok((
                        HirExpr::LeaveWith {
                            expr: Box::new(hir),
                            ty: never,
                            span: *span,
                        },
                        never,
                    ))
                }
            }
            Expr::Await { expr, span } => {
                let (hir, ty) = self.infer_expr(expr, None)?;
                let future_ty = self.checker.check_future_type(ty, *span)?;
                Ok((
                    HirExpr::Await {
                        expr: Box::new(hir),
                        ty: future_ty,
                        span: *span,
                    },
                    future_ty,
                ))
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                is_expression,
                span,
            } => {
                let (cond_hir, cond_ty) = self.infer_expr(cond, None)?;
                let cond_is_bool = self.checker.ctx.is_bool(cond_ty)
                    || matches!(self.checker.ctx.get(cond_ty), TypeData::InferVar { id, .. }
                        if self.checker.infer.get_var_kind(*id) == Some(TypeVariableKind::Bool));
                if !cond_is_bool {
                    self.checker.diagnostics.push(
                        Diagnostic::error("if condition must be boolean")
                            .with_code_str("E004")
                            .with_span(*span)
                            .with_label(
                                cond.span(),
                                format!("got {:?}", self.checker.ctx.get(cond_ty)),
                            ),
                    );
                }
                let then_hir = self.check_block(then_branch)?;
                let then_ty = self.block_type(&then_hir);
                let else_hir = else_branch
                    .as_ref()
                    .map(|b| self.check_block(b))
                    .transpose()?;
                let else_ty = else_hir
                    .as_ref()
                    .map(|h| self.block_type(h))
                    .unwrap_or(self.checker.ctx.unit());
                // Divergence detection: a branch diverges if ANY statement
                // in it diverges unconditionally — a `return`/`leave`/
                // `continue` anywhere (later statements are unreachable),
                // an `if` whose both branches diverge, or a `!`-typed call.
                // If both branches diverge the result is never.
                let then_diverges = self.block_diverges(&then_hir);
                let else_diverges = else_hir.as_ref().is_some_and(|h| self.block_diverges(h));
                let both_diverge = then_diverges && else_diverges;
                if *is_expression && !both_diverge && then_ty != else_ty {
                    self.checker.ctx.unify_tracked(then_ty, else_ty, *span).ok();
                }
                let result_ty = if *is_expression {
                    if then_diverges {
                        else_ty
                    } else if else_diverges {
                        then_ty
                    } else {
                        then_ty
                    }
                } else if both_diverge {
                    self.checker.ctx.never()
                } else {
                    self.checker.ctx.unit()
                };
                Ok((
                    HirExpr::If {
                        cond: Box::new(cond_hir),
                        then_branch: then_hir,
                        else_branch: else_hir,
                        is_expression: *is_expression,
                        ty: result_ty,
                        span: *span,
                    },
                    result_ty,
                ))
            }
            Expr::IfLet { .. } => self.infer_if_let_expr(expr, expected),
            Expr::Match { .. } => self.infer_match_expr(expr, expected),
            Expr::Block(stmts, span) => {
                let hir_stmts = self.check_block(stmts)?;
                let ty = self.block_type(&hir_stmts);
                Ok((HirExpr::Block(hir_stmts, ty, *span), ty))
            }
            Expr::Quantified {
                quantifier,
                binder,
                range,
                body,
                span,
            } => {
                let (range_hir, _range_ty) = self.infer_expr(range, None)?;
                let (body_hir, _body_ty) = self.infer_expr(body, None)?;
                let bool_ty = self.checker.ctx.bool();
                Ok((
                    HirExpr::Quantified {
                        quantifier: *quantifier,
                        binder: *binder,
                        range: Box::new(range_hir),
                        body: Box::new(body_hir),
                        ty: bool_ty,
                        span: *span,
                    },
                    bool_ty,
                ))
            }
            Expr::PolyBox {
                expr,
                scheme: _,
                span,
            } => {
                // Infer the inner expression type.
                let (hir_expr, inner_ty) = self.infer_expr(expr, None)?;
                let resolved = self.checker.ctx.resolve_binding(inner_ty);
                match self.checker.ctx.get(resolved).clone() {
                    TypeData::Forall {
                        param_index,
                        param_name,
                        body,
                    } => {
                        // Wrap in Poly — extract quantifier info, reconstruct.
                        let quantifiers = vec![(param_index, param_name)];
                        // Peel any nested Forall layers.
                        let mut all_q = quantifiers;
                        let mut inner_body = body;
                        loop {
                            match self
                                .checker
                                .ctx
                                .get(self.checker.ctx.resolve_binding(inner_body))
                                .clone()
                            {
                                TypeData::Forall {
                                    param_index: pi,
                                    param_name: pn,
                                    body: b,
                                } => {
                                    all_q.push((pi, pn));
                                    inner_body = b;
                                }
                                _ => break,
                            }
                        }
                        let poly_ty = self.checker.ctx.poly(all_q, inner_body);
                        Ok((
                            HirExpr::PolyBox {
                                expr: Box::new(hir_expr),
                                ty: poly_ty,
                                span: *span,
                            },
                            poly_ty,
                        ))
                    }
                    other => {
                        // Not polymorphic — try to box the entire Forall-like structure
                        // or emit an error if the type isn't quantifiable.
                        let msg = format!(
                            "poly(...) requires a polymorphic expression, found non-polymorphic type {:?}",
                            other
                        );
                        self.checker
                            .diagnostics
                            .push(Diagnostic::error(msg).with_span(*span));
                        Ok((HirExpr::Error(*span), self.checker.ctx.error()))
                    }
                }
            }
            Expr::Old(expr, span) => {
                // `old(expr)` captures the value at function entry.
                // Infer the inner expression's type and wrap it.
                let (hir, ty) = self.infer_expr(expr, None)?;
                Ok((
                    HirExpr::Old {
                        expr: Box::new(hir),
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
            Expr::PolyUnbox {
                expr,
                scheme: _,
                span,
            } => {
                let (hir_expr, outer_ty) = self.infer_expr(expr, None)?;
                let resolved = self.checker.ctx.resolve_binding(outer_ty);
                match self.checker.ctx.get(resolved).clone() {
                    TypeData::Poly { quantifiers, body } => {
                        // Instantiate the polytype: replace each GenericParam with a fresh InferVar,
                        // then return a ROOT InferVar unified with the constructed type,
                        // so unification can propagate through the InferVar.
                        let subst_map: Vec<(usize, TypeId)> = quantifiers
                            .iter()
                            .map(|(idx, _name)| {
                                let fresh = self.checker.infer.new_type_var(
                                    self.checker.ctx,
                                    crate::hir::infer::TypeVariableKind::Any,
                                    crate::hir::infer::VarOrigin::Synthetic,
                                );
                                (*idx, fresh)
                            })
                            .collect();
                        let mut inst_ty = body;
                        for (idx, fresh_ty) in &subst_map {
                            inst_ty = self.checker.ctx.replace_generic(inst_ty, *idx, *fresh_ty);
                        }
                        // Create a root InferVar and unify it with the instantiated type,
                        // so the result behaves as an InferVar for unification purposes.
                        let root = self.checker.infer.new_type_var(
                            self.checker.ctx,
                            crate::hir::infer::TypeVariableKind::Any,
                            crate::hir::infer::VarOrigin::Synthetic,
                        );
                        self.checker.ctx.unify_tracked(root, inst_ty, *span).ok();
                        Ok((
                            HirExpr::PolyUnbox {
                                expr: Box::new(hir_expr),
                                ty: root,
                                span: *span,
                            },
                            root,
                        ))
                    }
                    TypeData::InferVar { .. } => {
                        // The operand's polytype is not resolved yet.
                        // Linking the result to the unresolved variable
                        // would make it a permanent alias: when the
                        // variable later resolves to a `Poly`, the result
                        // becomes that SAME boxed polytype with the
                        // instantiation step skipped — breaking
                        // first-class polymorphism (`f = unbox(p); f(42)`
                        // would see a boxed poly instead of a monotype).
                        // Fail closed: require an explicit annotation so
                        // the polytype is known at the unbox site.
                        let msg = format!(
                            "unbox(expr) requires a polytype, but the operand's type is not yet \
                             resolved — add an explicit type annotation on the unbox operand \
                             (e.g. `unbox: Scheme(expr)`) so the polytype is known here"
                        );
                        self.checker
                            .diagnostics
                            .push(Diagnostic::error(msg).with_span(*span));
                        Ok((HirExpr::Error(*span), self.checker.ctx.error()))
                    }
                    other => {
                        let msg = format!("unbox(expr) requires a polytype, found {:?}", other);
                        self.checker
                            .diagnostics
                            .push(Diagnostic::error(msg).with_span(*span));
                        Ok((HirExpr::Error(*span), self.checker.ctx.error()))
                    }
                }
            }
            Expr::Path(path, span) => {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "unresolved path: {}",
                        path.iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join("::")
                    ))
                    .with_span(*span),
                );
                Ok((HirExpr::Error(*span), self.checker.ctx.error()))
            }
            Expr::Error(span) => Ok((HirExpr::Error(*span), self.checker.ctx.error())),
            Expr::TypeInfo(ty, span) => {
                // @typeInfo!(Type) — evaluate at compile time in generate expansion.
                // At type-checking time, we treat it as a deferred comptime expression.
                let ty_id = self.resolve_type(ty)?;
                Ok((HirExpr::TypeInfo(ty_id, *span), self.checker.ctx.unit()))
            }
            Expr::LayoutOf(ty, span) => {
                // layout_of!(Type) — pass the AST type through to HIR so that
                // type expressions can be resolved during comptime evaluation.
                // The expression type is a dedicated LayoutDescriptor (not
                // ctx.error()), so field access on the result type-checks.
                Ok((
                    HirExpr::LayoutOf(Box::new((*ty).clone()), *span),
                    self.checker.ctx.builtin_layout_descriptor,
                ))
            }
            Expr::CompileError(msg, span) => {
                let diag = Diagnostic::error(msg.clone())
                    .with_code_str("E099")
                    .with_help("`@compile_error` halts compilation unconditionally when evaluated")
                    .with_span(*span);
                self.checker.diagnostics.push(diag);
                Ok((
                    HirExpr::CompileError(msg.clone(), *span),
                    self.checker.ctx.error(),
                ))
            }
            Expr::Task { body, span } => {
                let block = self.check_block(body)?;
                let ty = self.checker.ctx.unit();
                Ok((
                    HirExpr::Task {
                        block,
                        ty,
                        span: *span,
                    },
                    ty,
                ))
            }
        }
    }

    /// Check expression against a known type (bidirectional).
    ///
    /// First infers the expression's type, then unifies it with the
    /// expected type when one is provided (e.g. annotated variable
    /// declarations, function argument checking).
    /// Check one call argument: the argument expression is type-checked
    /// WITHOUT the call-site coercion context (internal structural
    /// positions — struct/array/ADT fields — must NOT inherit `@auto_ro`),
    /// then the top-level argument→parameter coercion happens under
    /// `CallSite` — the ONLY place `@auto_ro`'s implicit freeze may apply.
    fn check_call_argument(
        &mut self,
        arg: &crate::ast::Expr<'input>,
        expected: TypeId,
        index: usize,
        total: usize,
        span: crate::ast::Span,
    ) -> Result<HirExpr<'input>, Diagnostic> {
        // `@auto_ro`'s implicit freeze applies at call sites — set the
        // `CallSite` coercion context BEFORE `check_expr` so that context-
        // sensitive inference inside `check_expr` (e.g. `Expr::EnumLit`
        // type-argument recovery) can use the expected type hint without
        // losing the `@auto_ro` freeze (see `check_call_argument` doc).
        // Data-constructor positions (struct fields, ADT payloads, array
        // elements) inside `check_expr` set `Structural` temporarily.
        let _coercion = crate::hir::types::CallSiteCoercion::enter(&self.checker.ctx);
        // Infer the argument type with the expected type hint (for context-
        // sensitive inference like `Expr::EnumLit` type-argument recovery),
        // but do NOT unify yet — the deref coercion (if `@auto_coerce` is
        // active) must be applied BEFORE the final unification.
        let (mut hir_arg, mut ty) = self.infer_expr(arg, Some(expected))?;
        self.check_kind_compat(ty, None, expected, None, hir_arg.span())?;
        self.check_kind_compat(expected, None, ty, None, hir_arg.span())?;
        // `@auto_coerce` also enables deref coercions (SYNTAX.md §Local
        // Relaxation): `&Rc<T>` → `&T` through an `@auto_deref`-marked
        // `Deref` impl.  When the argument is a reference whose pointee
        // derefs to the expected pointee, unify through the deref.
        if self.checker.ctx.auto_coerce.get() {
            if let (TypeData::Ref { ty: p1, .. }, TypeData::Ref { ty: p2, .. }) =
                (self.checker.ctx.get(expected), self.checker.ctx.get(ty))
            {
                // The BUILT-IN REBORROW COERCION (the reference-of-
                // reference deref — rustc's call-site deref coercion):
                // `&mut r` where `r: &mut T` (or `&r` where `r: &T`/
                // `&mut T`) coerces to `&mut *r` / `&*r` — the
                // REFERENT's reference.  The HIR is REWRITTEN (the
                // `*r` deref inserted) so the borrow-check post-pass
                // sees the deref loan, which the referent-resolution
                // machinery lands on the ultimate object (`a = 5` after
                // `set r2 = get(&mut r)` must freeze `a` — E0506).  The
                // original outer mutability is preserved (the freeze is
                // handled by the call-site unification gate); an
                // IMMUTABLE inner reference cannot be reborrowed as
                // mutable (`&mut r` with `r: &T` → E030, like rustc).
                if let HirExpr::UnaryOp {
                    op,
                    expr,
                    ty: ref_ty,
                    span,
                } = &hir_arg
                    && let TypeData::Ref { ty: outer_p, .. } = self.checker.ctx.get(*ref_ty)
                    && let TypeData::Ref {
                        ty: inner,
                        mutable: im,
                        ..
                    } = self.checker.ctx.get(*outer_p)
                    && self.checker.ctx.get(*inner) == self.checker.ctx.get(*p1)
                {
                    let em = matches!(
                        self.checker.ctx.get(expected),
                        TypeData::Ref { mutable: true, .. }
                    );
                    if !(em && !*im) {
                        // The rebuilt reference preserves the ORIGINAL
                        // outer mutability (`&r` stays `&*r` — never
                        // upgraded to `&mut` by the coercion).
                        let arg_mut = matches!(op, UnaryOp::RefMut);
                        let deref = HirExpr::UnaryOp {
                            op: UnaryOp::Deref,
                            expr: expr.clone(),
                            ty: *inner,
                            span: expr.span(),
                        };
                        let coerced_ty = self.checker.ctx.reference(*inner, arg_mut);
                        hir_arg = HirExpr::UnaryOp {
                            op: *op,
                            expr: Box::new(deref),
                            ty: coerced_ty,
                            span: *span,
                        };
                        ty = coerced_ty;
                    }
                } else if *p1 != *p2
                    && let Some(target) = self.checker.try_deref_trait_step(*p2)
                {
                    // Preserve the ORIGINAL mutability of the argument
                    // reference — the deref step only changes the pointee
                    // type, not the mutability.  The `&mut → &` freeze is
                    // handled by the call-site unification gate in
                    // `unify_internal_impl` (coercion_depth == 0 ∧
                    // CallSite ∧ auto_ro ∨ auto_coerce).
                    let arg_mut = matches!(
                        self.checker.ctx.get(ty),
                        TypeData::Ref { mutable: true, .. }
                    );
                    // The HIR is REWRITTEN (the `*w` deref inserted) so
                    // the borrow-check post-pass sees the deref loan —
                    // the SAME invariant as the built-in reborrow branch
                    // above.  An explicit `&w`/`&mut w` argument derefs
                    // its OPERAND (the wrapper value); a bare reference
                    // argument (`r: &Wrapper<T>`) derefs the reference
                    // itself.  Either way the deref loan lands on the
                    // ultimate object via the referent-resolution
                    // machinery (polonius.rs).
                    let (deref_operand, wrap_op) = match &hir_arg {
                        HirExpr::UnaryOp {
                            op: op @ (UnaryOp::Ref | UnaryOp::RefMut),
                            expr,
                            ..
                        } => (expr.clone(), *op),
                        _ => (
                            Box::new(hir_arg.clone()),
                            if arg_mut {
                                UnaryOp::RefMut
                            } else {
                                UnaryOp::Ref
                            },
                        ),
                    };
                    let deref = HirExpr::UnaryOp {
                        op: UnaryOp::Deref,
                        expr: deref_operand,
                        ty: target,
                        span: hir_arg.span(),
                    };
                    let coerced_ty = self.checker.ctx.reference(target, arg_mut);
                    hir_arg = HirExpr::UnaryOp {
                        op: wrap_op,
                        expr: Box::new(deref),
                        ty: coerced_ty,
                        span: hir_arg.span(),
                    };
                    ty = coerced_ty;
                }
            }
        }
        // (committee ruling, 2026-08-05): the implicit `@auto_ro`/`@auto_coerce`
        // downgrade (`&mut T` argument → `&T` parameter) registers NO loan
        // in the type checker (only `unify_with` runs — see the
        // `unify_internal_impl` Ref-arm comment: "does NOT register a
        // loan"); the source remains mutable here.  Any freeze for the
        // implicit form is carried by the borrow-check post-pass.  The gate
        // condition mirrors `unify_internal_impl`: CallSite context
        // (entered above) + expected immutable + arg mutable.
        //
        // Why the missing freeze is sound — two INDEPENDENT guarantees, do
        // not conflate them:
        //  (a) CALL time: the callee receives only `&T`, so the type system
        //      itself proves the source cannot be mutated during the call.
        //  (b) ARGUMENT-EVALUATION time: argument expressions are evaluated
        //      BEFORE the call; any side effect they have happens while the
        //      argument still holds `&mut T` (the downgrade applies at the
        //      parameter), and any such mutation is covered by the ordinary
        //      borrow checker.  The argument is NOT required to be "pure" —
        //      this comment previously claimed it was, which is not
        //      enforced anywhere and is not what soundness needs.
        // The flow-sensitive post-pass scans the HIR (which cannot
        // represent the type-level implicit downgrade), so NO freeze
        // enforcement occurs for the implicit downgrade — sound per (a).
        self.unify_with(expected, ty, span, TypingContext::Argument { index, total })?;
        // §584 (SYNTAX.md "Type Invariants" — implicit invariant
        // propagation): a function with a refined parameter type
        // automatically inherits `requires` that the argument satisfies
        // the type's invariant — passing `0` to `def f(x: NonZeroInt)`
        // is a construction of a refined value and is verified here (the
        // argument path uses `infer_expr` + explicit unify, so it does
        // NOT flow through `check_expr`'s construction-point hook).
        self.check_construction_invariant(expected, &hir_arg, hir_arg.span());
        Ok(hir_arg)
    }

    /// Check a call's arguments against the parameter types (shared by the
    /// direct-call, method-call and field-call paths).
    fn check_call_args(
        &mut self,
        args: &[crate::ast::Expr<'input>],
        param_tys: &[TypeId],
        span: crate::ast::Span,
    ) -> Result<Vec<HirExpr<'input>>, Diagnostic> {
        // (committee ruling, 2026-08-05): NO loan is registered for `@auto_ro`
        // implicit downgrades (the freeze is unobservable in the current
        // grammar — see `check_call_argument`).  Explicit loans
        // (`&ro`/`.freeze!()`/`&mut`) created inside the argument
        // expressions survive the call and live until their borrow
        // variable's LAST USE (point-level liveness — not the
        // enclosing block end).  The call-scope freeze semantics are
        // carried by the borrow-check post-pass (liveness) — there is NO
        // call-time cleanup here (the previous closure wrapper claimed
        // cleanup on `?` early return but had none).
        let mut hir_args = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let expected = param_tys
                .get(i)
                .copied()
                .unwrap_or(self.checker.ctx.error());
            let hir_arg = self.check_call_argument(arg, expected, i, args.len(), span)?;
            hir_args.push(hir_arg);
        }
        Ok(hir_args)
    }

    pub fn check_expr(
        &mut self,
        expr: &Expr<'input>,
        expected: Expectation,
        ctx: TypingContext,
    ) -> Result<HirExpr<'input>, Diagnostic> {
        // For Call expressions, propagate expected type for better type arg synthesis.
        if let Expr::Call {
            callee,
            args: call_args,
            comptime,
            span,
        } = expr
        {
            let expected_ty = match expected {
                Expectation::HasType(ty) => Some(ty),
                _ => None,
            };
            let (callee_hir, callee_ty) = self.infer_expr(callee, None)?;
            // Try type argument synthesis with the expected return type hint.
            // Propagate the Err variant (a synthesis failure such as an
            // arity mismatch is a real diagnostic — discarding it would
            // rely on the fall-through re-inference to re-raise it, losing
            // error locality and emitting the diagnosis late).
            match self.checker.try_synthesize_type_args(
                &callee_hir,
                callee_ty,
                call_args,
                *comptime,
                expected_ty,
                *span,
            ) {
                Ok(Some((hir, _))) => return Ok(hir),
                Ok(None) => {
                    // Non-polymorphic callee — fall through to the
                    // normal call handling via infer_expr.
                }
                Err(diag) => {
                    self.checker.diagnostics.push(diag);
                    return Ok(HirExpr::Error(*span));
                }
            }
            // Fall through to normal call handling via infer_expr.
        }
        // Propagate the expected type into inference: context-sensitive
        // expressions (EnumLit / Call / StructLit) need it to recover type
        // arguments (e.g. `Expr<T>` in `return Expr::Lit(42)`).
        let expected_ty = match expected {
            Expectation::HasType(ty) => Some(ty),
            _ => None,
        };
        let (hir, ty) = self.infer_expr(expr, expected_ty)?;
        if let Expectation::HasType(expected_ty) = expected {
            // Check kind compatibility before unification:
            // if the inferred type is an InferVar with a kind constraint
            // (e.g. Bool from a `true` literal, Integer from `42`),
            // verify that the expected type is compatible with that kind.
            self.check_kind_compat(ty, None, expected_ty, None, hir.span())?;
            self.check_kind_compat(expected_ty, None, ty, None, hir.span())?;
            self.unify_with(expected_ty, ty, hir.span(), ctx)?;
            // §577 (SYNTAX.md "Type Invariants"): every CONSTRUCTION POINT
            // checks the refined type's invariant — a value checked against
            // a type that carries an invariant must satisfy it (literals,
            // struct/enum literal fields, `with default` values, return
            // values, and call arguments all flow through here).
            self.check_construction_invariant(expected_ty, &hir, hir.span());
        }
        Ok(hir)
    }

    /// §577 (SYNTAX.md "Type Invariants"): "The compiler verifies or
    /// enforces the invariant at every construction point."  For a refined
    /// type `exists n: T invariant P(n)`:
    ///
    /// - Constructing from a value whose OWN type carries an ALPHA-EQUIVALENT
    ///   invariant is trivially safe (the value already satisfies it) — the
    ///   fast path.
    /// - Otherwise the constructed value's comptime value is computed and
    ///   the invariant is evaluated with the binder bound to it: provably
    ///   true accepts, provably false errors, and an unverifiable
    ///   construction is REJECTED ("verifies" — the compiler must be able
    ///   to prove the invariant; the value is not silently assumed valid).
    fn check_construction_invariant(
        &mut self,
        expected_ty: TypeId,
        value: &HirExpr<'input>,
        span: Span,
    ) {
        let Some(inv_ast) = self.checker.ctx.get_invariant(expected_ty).cloned() else {
            return; // not a refined type — nothing to check
        };
        // Read the binder from the RAW TypeData (NOT `name_of_exists`,
        // which follows bindings): after `check_expr` unified the value with
        // the expected type, `ctx.get(expected_ty)` may follow a binding to
        // the VALUE's exists type — but `get_invariant` above reads the raw
        // id's meta.  Using `get_raw` keeps the binder consistent with the
        // invariant it belongs to (the alpha-rename below must rewrite the
        // invariant's OWN binder name).
        let Some(binder) = (match self.checker.ctx.get_raw(expected_ty) {
            TypeData::Exists { name, .. } => Some(*name),
            _ => None,
        }) else {
            return;
        };
        let Some(base_ty) = self.checker.ctx.base_of_exists(expected_ty) else {
            return;
        };
        // Fast path: the value's own type carries an alpha-equivalent
        // invariant — `set y: NonZeroInt = x` where `x: NonZeroInt`.
        if let Some(v_inv) = self.checker.ctx.get_invariant(value.ty()).cloned()
            && let Some(v_binder) = (match self.checker.ctx.get_raw(value.ty()) {
                TypeData::Exists { name, .. } => Some(*name),
                _ => None,
            })
            && let Some(arena) = self.checker.ctx.arena
        {
            let renamed = crate::ast::visit::replace_ident_in_expr(arena, &v_inv, v_binder, binder);
            // L1/L2: discreteness-aware — for INTEGER bases `X > 0` ≡
            // `X >= 1`, so `set y: StrictlyPositive = x` (`x: PositiveInt`)
            // is accepted; floats/rationals compare the bounds exactly.
            let is_int = self.checker.ctx.is_integer(base_ty);
            if crate::hir::type_eq::expr_eq_ignoring_spans_typed(&renamed, &inv_ast, is_int) {
                return;
            }
        }
        // The constructed value's comptime value: a literal, a tracked
        // comptime-known variable, or a comptime-evaluable expression.
        let cv = match value {
            HirExpr::Literal(lit, _, _) => match lit {
                crate::ast::Literal::Int(v) => v.to_i128().map(ComptimeValue::Int),
                crate::ast::Literal::Float(v) => Some(ComptimeValue::Float(*v)),
                crate::ast::Literal::Bool(v) => Some(ComptimeValue::Bool(*v)),
                crate::ast::Literal::String(s) => {
                    Some(ComptimeValue::String(std::sync::Arc::from(s.as_str())))
                }
                _ => None,
            },
            HirExpr::Ident(name, _, _) => self.checker.get_literal_value(name).cloned(),
            other => {
                // `-3`, `40 + 2`, `fibonacci!(5)` — evaluate at comptime.
                let mut ec = crate::hir::comptime::ComptimeEvalContext::new(
                    self.checker.ctx,
                    self.checker.symbols,
                    &mut self.checker.diagnostics,
                );
                for (fn_name, (fn_params, fn_body)) in &self.checker.comptime_fn_registry {
                    ec.register_fn(*fn_name, fn_params.clone(), fn_body.clone());
                }
                ec.eval_expr(other).ok()
            }
        };
        let Some(cv) = cv else {
            self.checker.diagnostics.push(
                Diagnostic::error(format!(
                    "cannot prove invariant at construction: the value is not comptime-verifiable for the refined type (binder `{}`)",
                    binder.as_str()
                ))
                .with_span(span),
            );
            return;
        };
        // Evaluate the invariant with the binder bound to the value: rename
        // the binder to a fresh in-scope variable, infer the invariant HIR,
        // then run it through the comptime evaluator with the slot bound.
        let Some(arena) = self.checker.ctx.arena else {
            return;
        };
        // Generate the fresh binder with CAPTURE AVOIDANCE: the name must
        // not occur free in the invariant — a free occurrence of the
        // generated name in `inv_ast` would be accidentally shadowed by
        // the rename below, silently changing the invariant's meaning.
        // The base name is deterministic (`_inv_{span.start}`), so repeated
        // checks are stable; a clash retries with an incrementing suffix.
        // Deterministic GLOBAL counter for the base name — the id depends
        // on compilation order, so user code cannot predict it (unlike
        // the old span-based name), making the retry loop exhaust
        // essentially never.
        let inv_id = self.checker.next_inv_binder_id;
        self.checker.next_inv_binder_id += 1;
        let mut fresh = Symbol::intern(&format!("_inv_{}", inv_id));
        let mut clash = 0;
        // `expr_free_in` is fail-closed (unenumerated shapes report
        // "possibly free"), so the retry loop MUST be bounded — an
        // invariant using e.g. `match` would otherwise spin forever
        // looking for a free name (compiler DoS).
        const MAX_INVARIANT_RENAME_ATTEMPTS: usize = 64;
        while clash < MAX_INVARIANT_RENAME_ATTEMPTS
            && crate::hir::type_eq::expr_free_in(fresh, &inv_ast)
        {
            fresh = Symbol::intern(&format!("_inv_{}_{}", inv_id, clash));
            clash += 1;
        }
        // FAIL-CLOSED: if the bound is exhausted and the last candidate is
        // still "possibly free" (an invariant using e.g. `match`),
        // renaming through it could capture a free occurrence — do NOT
        // proceed; report and skip the construction check.
        if crate::hir::type_eq::expr_free_in(fresh, &inv_ast) {
            self.checker.diagnostics.push(
                Diagnostic::error("invariant too complex to safely generate a fresh binder")
                    .with_span(span),
            );
            return;
        }
        let renamed = crate::ast::visit::replace_ident_in_expr(arena, &inv_ast, binder, fresh);
        let _scope = self.checker.enter_var_scope();
        self.checker.local_variable_types.insert(fresh, base_ty);
        let inv_hir = match self.infer_expr(&renamed, None) {
            Ok((h, _)) => h,
            Err(diag) => {
                self.checker.diagnostics.push(diag);
                return;
            }
        };
        let mut eval = crate::hir::comptime::ComptimeEvalContext::new(
            self.checker.ctx,
            self.checker.symbols,
            &mut self.checker.diagnostics,
        );
        for (fn_name, (fn_params, fn_body)) in &self.checker.comptime_fn_registry {
            eval.register_fn(*fn_name, fn_params.clone(), fn_body.clone());
        }
        let slot = eval.allocate_slot();
        eval.cur_slot.insert(fresh, slot);
        eval.variables.insert(slot, cv);
        match eval.eval_expr(&inv_hir) {
            Ok(ComptimeValue::Bool(true)) => {}
            Ok(ComptimeValue::Bool(false)) => {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "invariant not satisfied at construction: the value violates the refined type's invariant (binder `{}`)",
                        binder.as_str()
                    ))
                    .with_span(span),
                );
            }
            _ => {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot prove invariant at construction: the invariant does not evaluate to a boolean (binder `{}`)",
                        binder.as_str()
                    ))
                    .with_span(span),
                );
            }
        }
    }

    /// Check whether calling a function/method with the given @trusted/@io
    /// attributes violates the current comptime or isolate sandbox.
    /// Returns `Ok(())` if the call is allowed, or the appropriate
    /// `(HirExpr::Error, error_type)` pair if the call should be rejected.
    fn check_call_attribute_violation(
        &mut self,
        name: &Symbol,
        is_method: bool,
        has_trusted: bool,
        has_io: bool,
        span: Span,
    ) -> Result<(), (HirExpr<'input>, TypeId)> {
        let kind = if is_method { "method" } else { "function" };
        if self.checker.is_in_comptime() {
            if has_trusted {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot call @trusted {} `{}` from comptime context: \
                         comptime code is sandboxed and cannot call @trusted functions",
                        kind, name,
                    ))
                    .with_code_str("E081")
                    .with_span(span)
                    .with_help("@trusted functions may perform I/O or unsafe operations, which are prohibited in comptime"),
                );
                return Err((HirExpr::Error(span), self.checker.ctx.error()));
            }
            if has_io {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot call @io {} `{}` from comptime context: \
                         comptime code is sandboxed and cannot perform I/O",
                        kind, name,
                    ))
                    .with_code_str("E081")
                    .with_span(span)
                    .with_help("I/O operations are prohibited in comptime"),
                );
                return Err((HirExpr::Error(span), self.checker.ctx.error()));
            }
        }
        if self.checker.is_in_isolate() {
            if has_trusted {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot call @trusted {} `{}` inside isolate block: \
                         isolate blocks must not access external mutable state",
                        kind, name,
                    ))
                    .with_code_str("E093")
                    .with_span(span)
                    .with_help("isolate blocks guarantee no external mutable state access; @trusted functions may violate this"),
                );
                return Err((HirExpr::Error(span), self.checker.ctx.error()));
            }
            if has_io {
                self.checker.diagnostics.push(
                    Diagnostic::error(format!(
                        "cannot call @io {} `{}` inside isolate block: \
                         isolate blocks must not perform I/O",
                        kind, name,
                    ))
                    .with_code_str("E093")
                    .with_span(span)
                    .with_help("isolate blocks guarantee no external mutable state access; @io functions may perform I/O"),
                );
                return Err((HirExpr::Error(span), self.checker.ctx.error()));
            }
        }
        Ok(())
    }

    /// Check that an InferVar's kind constraint is compatible with the
    /// resolved type of another type.  This prevents situations like
    /// `true` (InferVar with kind Bool) being unified with `Int<32>`.
    /// Only fires when the other side resolves to a concrete (non-type-variable) type.
    fn check_kind_compat(
        &self,
        maybe_var: TypeId,
        maybe_var_span: Option<Span>,
        other: TypeId,
        other_span: Option<Span>,
        span: Span,
    ) -> Result<(), Diagnostic> {
        self.checker
            .check_kind_compat(maybe_var, maybe_var_span, other, other_span, span)
    }

    /// Resolve a syntactic type to a TypeId — actual implementation.
    #[must_use]
    pub fn resolve_type(&mut self, ty: &Type<'input>) -> Result<TypeId, Diagnostic> {
        match ty {
            Type::Path(path, span) => {
                // Lifetime parsed as placeholder path `["'a"]` — skip resolution.
                if let Ok(def_id) = self.checker.resolve_def_id(path) {
                    // Check if this is a generic type parameter (sentinel from resolve_def_id)
                    if def_id == DefId(usize::MAX - 1) {
                        if path.len() == 1
                            && let Some(&ty) = self.checker.local_type_param_cache.get(&path[0])
                        {
                            return Ok(ty);
                        }
                        return Err(Diagnostic::error(format!("type '{}' not found", path[0]))
                            .with_span(*span));
                    }
                    let binding: TypeBinding<'input>;
                    if let Some(b) = self.checker.resolution_map.type_bindings.get(&def_id) {
                        binding = b.clone();
                    } else {
                        binding = self
                            .checker
                            .symbols
                            .lookup_type_by_def_id(def_id)
                            .ok_or_else(|| {
                                Diagnostic::error(format!("type not found: {:?}", path))
                                    .with_span(*span)
                            })?
                            .clone();
                    };
                    // ── @experimental check ───────────────────────────
                    for attr in &binding.attributes {
                        if attr.name.eq_str("experimental") && !self.checker.enable_experimental {
                            self.checker.diagnostics.push(
                                Diagnostic::error(format!(
                                    "use of experimental type `{}`",
                                    path.last()
                                        .map(|s| s.as_str().to_string())
                                        .unwrap_or_else(|| "?".to_string()),
                                ))
                                .with_code_str("E094")
                                .with_span(*span)
                                .with_help("experimental features are not enabled; use `--enable-experimental` to use this type"),
                            );
                        }
                    }
                    match binding.kind {
                        TypeKind::Alias => {
                            if self.checker.resolving_aliases.contains(&def_id) {
                                return Err(
                                    Diagnostic::error("circular alias definition").with_span(*span)
                                );
                            }
                            self.checker.resolving_aliases.insert(def_id);
                            let result = binding
                                .alias_ast
                                .as_ref()
                                .map(|ast| self.resolve_type(ast))
                                .unwrap_or_else(|| {
                                    // Check if this is a type capture name (auto<T>).
                                    // The resolver creates a placeholder alias with no body;
                                    // the actual type is in local_type_param_cache.
                                    if path.len() == 1
                                        && let Some(&ty) =
                                            self.checker.local_type_param_cache.get(&path[0])
                                    {
                                        return Ok(ty);
                                    }
                                    Err(Diagnostic::error("alias has no body").with_span(*span))
                                });
                            self.checker.resolving_aliases.remove(&def_id);
                            result
                        }
                        TypeKind::Struct => {
                            if binding.params.is_empty() {
                                Ok(self.checker.ctx.struct_ty(def_id, vec![]))
                            } else {
                                let args: Vec<TypeId> = (0..binding.params.len())
                                    .map(|_| {
                                        self.new_infer_var(
                                            TypeVariableKind::Unconstrained,
                                            crate::hir::infer::VarOrigin::Synthetic,
                                        )
                                    })
                                    .collect();
                                Ok(self.checker.ctx.struct_ty(def_id, args))
                            }
                        }
                        TypeKind::Enum => {
                            if binding.params.is_empty() {
                                Ok(self.checker.ctx.enum_ty(def_id, vec![]))
                            } else {
                                let args: Vec<TypeId> = (0..binding.params.len())
                                    .map(|_| {
                                        self.new_infer_var(
                                            TypeVariableKind::Unconstrained,
                                            crate::hir::infer::VarOrigin::Synthetic,
                                        )
                                    })
                                    .collect();
                                Ok(self.checker.ctx.enum_ty(def_id, args))
                            }
                        }
                        _ => Err(Diagnostic::error("expected type, found something else")
                            .with_span(*span)),
                    }
                } else {
                    if path[0].eq_str("Bool") {
                        Ok(self.checker.ctx.bool())
                    } else if path[0].eq_str("Char") {
                        Ok(self.checker.ctx.char())
                    } else if path[0].eq_str("Byte") {
                        Ok(self.checker.ctx.byte())
                    } else if path[0].eq_str("USize") {
                        Ok(self.checker.ctx.usize())
                    } else if path[0].eq_str("Unit") {
                        return Err(Diagnostic::error("use `()` instead of `Unit`")
                            .with_code_str("E031")
                            .with_help("Posita uses `()` (empty tuple) to express the unit type")
                            .with_suggestion("replace `Unit` with `()`")
                            .with_span(*span));
                    } else if path[0].eq_str("Never") {
                        Ok(self.checker.ctx.never())
                    } else {
                        // Check if this is a generic type parameter registered in the local cache
                        if path.len() == 1
                            && let Some(&ty) = self.checker.local_type_param_cache.get(&path[0])
                        {
                            return Ok(ty);
                        }
                        Err(Diagnostic::error(format!("type '{}' not found", path[0]))
                            .with_span(*span))
                    }
                }
            }
            Type::Generic(base, args, span) => {
                if let Type::Path(path, _) = base {
                    if path.len() == 1 {
                        if path[0].eq_str("Int") {
                            let width = args
                                .get(0)
                                .and_then(|arg| {
                                    self.checker.extract_int_from_type(arg.ty().as_ref())
                                })
                                .unwrap_or(DEFAULT_INT_WIDTH);
                            return Ok(self.checker.ctx.int(width, true));
                        } else if path[0].eq_str("UInt") {
                            let width = args
                                .get(0)
                                .and_then(|arg| {
                                    self.checker.extract_int_from_type(arg.ty().as_ref())
                                })
                                .unwrap_or(DEFAULT_INT_WIDTH);
                            return Ok(self.checker.ctx.int(width, false));
                        } else if path[0].eq_str("Float") {
                            let width = args
                                .get(0)
                                .and_then(|arg| {
                                    self.checker.extract_int_from_type(arg.ty().as_ref())
                                })
                                .unwrap_or(DEFAULT_FLOAT_WIDTH);
                            return Ok(self.checker.ctx.float(width));
                        } else if path[0].eq_str("Rational") {
                            let p = args.get(0).and_then(|arg| self.checker.extract_int_from_type(arg.ty().as_ref()))
                                .ok_or_else(|| Diagnostic::error("Rational requires a compile-time constant integer bit count for the integer part").with_span(*span))?;
                            let q = args.get(1).and_then(|arg| self.checker.extract_int_from_type(arg.ty().as_ref()))
                                .ok_or_else(|| Diagnostic::error("Rational requires a compile-time constant integer bit count for the fractional part").with_span(*span))?;
                            if p == 0 || p > MAX_RATIONAL_BITS || q == 0 || q > MAX_RATIONAL_BITS {
                                return Err(Diagnostic::error(format!(
                                    "Rational bit counts must be 1..={MAX_RATIONAL_BITS}"
                                ))
                                .with_span(*span));
                            }
                            return Ok(self.checker.ctx.rational(p as u8, q as u8));
                        } else if path[0].eq_str("Ptr") {
                            let mut size = self.checker.ctx.usize();
                            let mut pointee = self.checker.ctx.error();
                            for arg in args {
                                let ty = self.resolve_type(arg.ty().as_ref())?;
                                match arg {
                                    GenericArg::Named(name, _) if name.eq_str("size") => size = ty,
                                    GenericArg::Named(name, _) if name.eq_str("pointee") => {
                                        pointee = ty
                                    }
                                    GenericArg::Positional(_) => {
                                        if self.checker.ctx.is_error(pointee) {
                                            pointee = ty;
                                        } else {
                                            size = ty;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            return Ok(self.checker.ctx.ptr(size, pointee));
                        } else if path[0].eq_str("USize") {
                            return Ok(self.checker.ctx.usize());
                        }
                    }
                }
                let base_ty = self.resolve_type(base)?;
                let expanded = self.expand_base_type(base_ty, *span)?;
                let mut arg_tys = Vec::new();
                for arg in args {
                    arg_tys.push(self.resolve_type(arg.ty().as_ref())?);
                }
                match self.checker.ctx.get(expanded) {
                    TypeData::Adt { def_id, .. } => {
                        let binding = self
                            .checker
                            .symbols
                            .lookup_type_by_def_id(*def_id)
                            .ok_or_else(|| {
                                Diagnostic::error("type definition not found").with_span(*span)
                            })?;
                        if arg_tys.len() != binding.params.len() {
                            return Err(Diagnostic::error(format!(
                                "wrong number of type arguments: expected {}, got {}",
                                binding.params.len(),
                                arg_tys.len()
                            ))
                            .with_span(*span));
                        }
                        match binding.kind {
                            TypeKind::Struct => Ok(self.checker.ctx.struct_ty(*def_id, arg_tys)),
                            TypeKind::Enum => Ok(self.checker.ctx.enum_ty(*def_id, arg_tys)),
                            _ => Err(Diagnostic::error(
                                "generic type arguments on non-generic type",
                            )
                            .with_span(*span)),
                        }
                    }
                    _ => Err(
                        Diagnostic::error("generic type arguments on non-generic type")
                            .with_span(*span),
                    ),
                }
            }
            Type::Reference {
                inner: ty,
                mutable,
                lifetime,
                ..
            } => {
                let inner = self.resolve_type(ty)?;
                // The explicit lifetime annotation (`&'a mut T`) survives
                // into the type (`TypeData::Ref.lifetime`) so the region
                // solver can verify `'a: 'b` outlives constraints — the
                // annotation is NOT dropped here anymore.
                Ok(self
                    .checker
                    .ctx
                    .reference_with_lifetime(inner, *mutable, *lifetime))
            }
            Type::Pointer(ty, _) => {
                let inner = self.resolve_type(ty)?;
                Ok(self.checker.ctx.pointer(inner))
            }
            Type::Slice(ty, _) => {
                let inner = self.resolve_type(ty)?;
                Ok(self.checker.ctx.slice(inner))
            }
            Type::Array(ty, size, span) => {
                let inner = self.resolve_type(ty)?;
                if let Expr::Literal(Literal::Int(size_val), _) = size {
                    match size_val.to_u64() {
                        Some(sz) => Ok(self.checker.ctx.array(inner, sz)),
                        None => Err(Diagnostic::error(
                            "array size must be a valid non-negative integer fitting in u64",
                        )
                        .with_span(*span)),
                    }
                } else {
                    // Non-literal size: try to evaluate it as a comptime expression.
                    match self.infer_expr(size, None) {
                        Ok((size_hir, _size_ty)) => {
                            let mut eval = ComptimeEvalContext::new(
                                self.checker.ctx,
                                self.checker.symbols,
                                &mut self.checker.diagnostics,
                            );
                            for (name, (params, body)) in &self.checker.comptime_fn_registry {
                                eval.register_fn(*name, params.clone(), body.clone());
                            }
                            match eval.eval_expr(&size_hir) {
                                Ok(ComptimeValue::Int(n)) if n >= 0 => {
                                    Ok(self.checker.ctx.array(inner, n as u64))
                                }
                                Ok(ComptimeValue::Int(n)) => Err(Diagnostic::error(format!(
                                    "array size must be a non-negative integer, found {}",
                                    n
                                ))
                                .with_span(*span)),
                                Ok(_) => Err(Diagnostic::error(
                                    "array size must evaluate to an integer at compile time",
                                )
                                .with_span(*span)),
                                Err(e) => Err(Diagnostic::error(format!(
                                    "array size evaluation failed: {}",
                                    e
                                ))
                                .with_span(*span)),
                            }
                        }
                        Err(e) => Err(Diagnostic::error(
                            "array size must be a compile-time constant integer",
                        )
                        .with_span(*span)),
                    }
                }
            }
            Type::Tuple(tys, _) => {
                let elems: Vec<_> = tys
                    .iter()
                    .map(|t| self.resolve_type(t))
                    .collect::<Result<_, _>>()?;
                Ok(self.checker.ctx.tuple(elems))
            }
            Type::Function { params, ret, .. } => {
                let param_tys: Vec<_> = params
                    .iter()
                    .map(|p| self.resolve_type(p))
                    .collect::<Result<_, _>>()?;
                let ret_ty = self.resolve_type(ret)?;
                Ok(self.checker.ctx.function(param_tys, ret_ty))
            }
            Type::Forall { lifetime, body, .. } => {
                // Higher-ranked type `for<'a> T` (SYNTAX.md
                // §Higher-Ranked Trait Bounds): the lifetime is
                // universally quantified over the body — allocate a fresh
                // binder index; the subtype relation skolemizes it at the
                // call site (subtype's Forall arm + check_skolem_escape).
                let body_ty = self.resolve_type(body)?;
                Ok(self.checker.ctx.forall(
                    self.checker.ctx.fresh_param_index(),
                    *lifetime,
                    body_ty,
                ))
            }
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name: name,
                span,
            } => {
                let _impl_ty = self.resolve_type(impl_type)?;
                let _trait_ty = self.resolve_type(trait_path)?;
                let candidates = self.checker.symbols.lookup_traits_by_assoc_type_name(*name);
                match candidates.len() {
                    0 => {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "no trait defines associated type `{}`",
                                name
                            ))
                            .with_span(*span),
                        );
                        Ok(self.checker.ctx.error())
                    }
                    1 => Ok(self
                        .checker
                        .ctx
                        .associated_type(candidates[0], *name, _impl_ty)),
                    _ => {
                        self.checker.diagnostics.push(
                            Diagnostic::error(format!(
                                "ambiguous associated type `{}` found in multiple traits",
                                name
                            ))
                            .with_span(*span),
                        );
                        Ok(self.checker.ctx.error())
                    }
                }
            }
            Type::DynTrait(traits, span) => {
                // SYNTAX.md §Dynamic Dispatch in Strict Mode: OUTSIDE of
                // `@trusted` code, `dyn Trait` is rejected in strict mode
                // (the fail-closed gate was missing; the
                // concrete→dyn coercion path does not exist yet, but the
                // gate must be in place for when it does).
                if self.checker.strict_mode && !self.checker.current_function_trusted {
                    self.checker.diagnostics.push(
                        Diagnostic::error(
                            "`dyn Trait` is not allowed in strict mode outside `@trusted` code",
                        )
                        .with_code_str("E004")
                        .with_span(*span)
                        .with_help(
                            "dynamic dispatch is rejected in strict mode (SYNTAX.md \
                             §Dynamic Dispatch) — use a concrete type or mark the \
                             function `@trusted`",
                        ),
                    );
                }
                let trait_ids: Vec<_> = traits
                    .iter()
                    .filter_map(|t| {
                        if let Type::Path(p, _) = t {
                            self.checker.resolve_def_id(p).ok()
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(self.checker.ctx.dyn_trait(trait_ids))
            }
            Type::Exists {
                name,
                base,
                invariant,
                span,
            } => {
                let base_ty = self.resolve_type(base)?;
                // §577 (SYNTAX.md "Type Invariants"): the invariant may
                // reference the binder (`exists n: Int<32> invariant n != 0`
                // — `n` IS the value being constrained) — register it as a
                // local of the base type so the invariant inference sees it.
                let _binder_scope = self.checker.enter_var_scope();
                self.checker.local_variable_types.insert(*name, base_ty);
                let (inv_hir, inv_ty) = self.infer_expr(invariant, None)?;
                if !self.checker.ctx.is_bool(inv_ty) {
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error("invariant must be boolean").with_span(*span));
                }
                Ok(self.checker.ctx.exists(
                    self.checker.ctx.fresh_param_index(),
                    *name,
                    base_ty,
                    (*invariant).clone(),
                ))
            }
            Type::WhereShorthand {
                base,
                invariant,
                span,
            } => {
                // Desugar `type T = Base where value > 0` into `exists _where_N: Base invariant _where_N > 0`.
                let name = Symbol::intern(&format!("_where_{}", span.start));
                let arena = self
                    .checker
                    .ctx
                    .arena
                    .expect("arena required for the where-invariant desugar");
                let inv = replace_ident_in_expr(arena, &invariant, Symbol::intern("value"), name);
                let base_ty = self.resolve_type(base)?;
                // §577: register the desugared `_where_N` binder so the
                // invariant (`_where_N > 0`) resolves during inference.
                let _binder_scope = self.checker.enter_var_scope();
                self.checker.local_variable_types.insert(name, base_ty);
                let (_, inv_ty) = self.infer_expr(&inv, None)?;
                if !self.checker.ctx.is_bool(inv_ty) {
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error("invariant must be boolean").with_span(*span));
                }
                Ok(self.checker.ctx.exists(
                    self.checker.ctx.fresh_param_index(),
                    name,
                    base_ty,
                    (*inv).clone(),
                ))
            }
            Type::Literal(expr, _) => {
                let (_, ty) = self.infer_expr(expr, None)?;
                Ok(ty)
            }
            Type::Never(_) => Ok(self.checker.ctx.never()),
            Type::Union(tys, span) => {
                let resolved: Vec<TypeId> = tys
                    .iter()
                    .map(|t| self.resolve_type(t))
                    .collect::<Result<Vec<_>, _>>()?;
                if resolved.len() == 1 {
                    Ok(resolved[0])
                } else if resolved.is_empty() {
                    Ok(self.checker.ctx.never())
                } else {
                    // Combine all resolved types into a Coproduct (sum type),
                    // representing the union of all variants.
                    let mut alternatives = Vec::new();
                    for ty in resolved {
                        match self.checker.ctx.get(ty) {
                            TypeData::Adt { .. } => alternatives.push(ty),
                            TypeData::Coproduct { alternatives: alts } => {
                                alternatives.extend(alts.clone());
                            }
                            TypeData::Never => {} // ignore
                            _ => alternatives.push(ty),
                        }
                    }
                    // Deduplicate alternatives
                    alternatives.sort_by_key(|t| t.raw());
                    alternatives.dedup();
                    if alternatives.len() == 1 {
                        Ok(alternatives[0])
                    } else {
                        Ok(self.checker.ctx.coproduct(alternatives))
                    }
                }
            }
            Type::Error(_) => Ok(self.checker.ctx.error()),
            Type::Regex(pattern, _) => {
                // The release-mode regex validation: an invalid pattern is
                // reported with the diagnostics module (friendly error) in
                // ALL builds — not just the debug `debug_assert`.
                if let Err(e) = regex_syntax::parse(pattern) {
                    return Err(Diagnostic::error(format!(
                        "invalid regex pattern: \"{}\"",
                        pattern
                    ))
                    .with_help(format!("regex parse error: {}", e)));
                }
                Ok(self.checker.ctx.regex(pattern.clone()))
            }
            Type::Expr(expr, span) => {
                let (_, ty) = self.infer_expr(expr, None)?;
                Ok(ty)
            }
        }
    }

    /// Expand type aliases: if `ty` is an alias, resolve it to its body.
    #[must_use]
    pub fn expand_base_type(&mut self, ty: TypeId, span: Span) -> Result<TypeId, Diagnostic> {
        if let Some(def_id) = self.checker.ctx.get_def_id_for_type(ty) {
            if let Some(binding) = self.checker.symbols.lookup_type_by_def_id(def_id) {
                if binding.kind == TypeKind::Alias {
                    if self.checker.resolving_aliases.contains(&def_id) {
                        return Err(Diagnostic::error("circular alias definition").with_span(span));
                    }
                    self.checker.resolving_aliases.insert(def_id);
                    let result = binding
                        .alias_ast
                        .as_ref()
                        .map(|ast| self.resolve_type(ast))
                        .unwrap_or(Err(Diagnostic::error("alias has no body").with_span(span)));
                    self.checker.resolving_aliases.remove(&def_id);
                    return result;
                }
            }
        }
        Ok(ty)
    }

    /// Get the type yielded by a block (last expression's type, or unit/never).
    pub fn block_type(&self, stmts: &[HirStmt<'input>]) -> TypeId {
        self.checker.block_type(stmts)
    }

    /// Create a fresh inference variable with the given kind.
    pub fn new_infer_var(
        &mut self,
        kind: TypeVariableKind,
        origin: crate::hir::infer::VarOrigin,
    ) -> TypeId {
        self.checker
            .infer
            .new_type_var(self.checker.ctx, kind, origin)
    }

    /// Add a constraint to the inference context.
    pub fn add_constraint(&mut self, c: Constraint) {
        self.checker.infer.add_constraint(c);
    }

    /// Check if a cast between two types is valid.
    pub fn check_cast(
        &mut self,
        from: TypeId,
        to: TypeId,
        safe: bool,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        if safe {
            if (self.ctx().is_numeric(from) && self.ctx().is_numeric(to))
                || (self.ctx().is_bool(from) && self.ctx().is_integer(to))
                || (self.ctx().is_integer(from) && self.ctx().is_bool(to))
            {
                Ok(to)
            } else if self.ctx().is_reference(from) {
                Err(Diagnostic::error(
                    "safe cast from reference type requires explicit dereference or unsafe cast",
                )
                .with_code_str("E601")
                .with_span(span)
                .with_suggestion("consider dereferencing first: `*expr as TargetType`")
                .with_suggestion("or use `as!` for an unsafe bitcast"))
            } else {
                Err(
                    Diagnostic::error("safe cast only allowed between numeric and boolean types")
                        .with_code_str("E601")
                        .with_span(span)
                        .with_suggestion("use `From` trait for non-primitive type conversions"),
                )
            }
        } else {
            if (self.ctx().is_numeric(from) && self.ctx().is_numeric(to))
                || (self.ctx().is_reference(from) && self.ctx().is_pointer(to))
                || (self.ctx().is_pointer(from) && self.ctx().is_reference(to))
                || (self.ctx().is_integer(from) && self.ctx().is_pointer(to))
                || (self.ctx().is_pointer(from) && self.ctx().is_integer(to))
            {
                Ok(to)
            } else if self.ctx().is_reference(from) && self.ctx().is_integer(to) {
                Err(
                    Diagnostic::error("unsafe cast from reference to integer not yet supported")
                        .with_code_str("E601")
                        .with_span(span)
                        .with_suggestion("consider using `*expr as usize` via a pointer cast"),
                )
            } else {
                let c = self.ctx();
                match (c.get(from), c.get(to)) {
                    (TypeData::Ptr { .. }, TypeData::Ptr { .. }) => Ok(to),
                    _ => Err(Diagnostic::error("unsafe cast requires compatible types (numeric<->numeric, ref<->ptr, ptr<->ptr)")
                        .with_code_str("E601").with_span(span)),
                }
            }
        }
    }

    /// Infer the return type of a binary operation.
    pub fn binary_op_type(
        &mut self,
        op: BinOp,
        left: TypeId,
        right: TypeId,
        left_span: Option<Span>,
        right_span: Option<Span>,
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        self.checker
            .binary_op_type(op, left, right, left_span, right_span, span)
    }

    /// Check a statement (delegates to TypeChecker).
    #[must_use]
    pub fn check_stmt(&mut self, stmt: &Stmt<'input>) -> Result<HirStmt<'input>, Diagnostic> {
        self.checker.check_stmt(stmt)
    }

    /// Check a block — actual implementation, not delegation.
    #[must_use]
    pub fn check_block(
        &mut self,
        stmts: &[Stmt<'input>],
    ) -> Result<Vec<HirStmt<'input>>, Diagnostic> {
        self.checker.push_literal_scope();
        let _scope = self.checker.enter_var_scope();
        // Loan scoping: a borrow created inside a lexical block expires at
        // the block's end (the conservative lexical-block reading of the
        // SYNTAX.md "borrow lifetime").  check_block is the unified block
        // Wrap in a closure so that `?` return inside the loop does not
        // skip cleanup — the outer function always runs pop_literal_scope
        // (the old loan truncation is removed ).
        let result = (|| {
            let mut result = Vec::new();
            for stmt in stmts {
                result.push(self.checker.check_stmt(stmt)?);
            }
            Ok(result)
        })();
        // scope drops here — pops the var scope frame (even on `?` early return)
        drop(_scope);
        self.checker.pop_literal_scope();
        result
    }

    /// Check a pattern against an expected type.
    pub fn check_pattern(
        &mut self,
        pattern: &Pattern<'input>,
        expected_ty: TypeId,
        exist_depth: usize,
    ) -> Result<HirPattern<'input>, Diagnostic> {
        let hir = self.check_pattern_inner(pattern, expected_ty, exist_depth)?;
        // Automatically register all pattern-bound variables into local scope
        // (and as runtime, so they shadow outer ghosts for `when` checking).
        register_pattern_bindings(
            &mut self.checker.local_variable_types,
            &self.checker.runtime_var_scopes,
            &hir,
        );
        Ok(hir)
    }

    /// Inner pattern check, without side-effect variable registration.
    /// Check a pattern against an expected type.
    fn check_pattern_inner(
        &mut self,
        pattern: &Pattern<'input>,
        expected_ty: TypeId,
        exist_depth: usize,
    ) -> Result<HirPattern<'input>, Diagnostic> {
        if self.checker.ctx.is_infer_var(expected_ty) {
            match pattern {
                Pattern::Tuple(patterns, span) => {
                    let elem_tys: Vec<TypeId> = patterns
                        .iter()
                        .map(|_| {
                            self.new_infer_var(
                                TypeVariableKind::Unconstrained,
                                crate::hir::infer::VarOrigin::Synthetic,
                            )
                        })
                        .collect();
                    let tuple_ty = self.checker.ctx.tuple(elem_tys.clone());
                    self.unify_with(expected_ty, tuple_ty, *span, TypingContext::None)?;
                    let mut hir_pats = Vec::new();
                    for (pat, &ety) in patterns.iter().zip(elem_tys.iter()) {
                        hir_pats.push(self.check_pattern(pat, ety, exist_depth)?);
                    }
                    return Ok(HirPattern::Tuple(hir_pats, *span));
                }
                _ => {}
            }
        }
        match pattern {
            Pattern::Wildcard(span) => Ok(HirPattern::Wildcard(*span)),
            Pattern::Ident(name, span) => Ok(HirPattern::Ident(*name, expected_ty, *span)),
            Pattern::Literal(expr, span) => {
                let (hir, ty) = self.infer_expr(expr, None)?;
                self.unify_with(expected_ty, ty, *span, TypingContext::None)?;
                Ok(HirPattern::Literal(Box::new(hir), *span))
            }
            Pattern::Tuple(patterns, span) => {
                let expected_elems = self
                    .checker
                    .ctx
                    .tuple_elems(expected_ty)
                    .map(|e| e.to_vec())
                    .unwrap_or_else(|| vec![self.checker.ctx.error(); patterns.len()]);
                let mut hir_patterns = Vec::new();
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_ty = expected_elems
                        .get(i)
                        .copied()
                        .unwrap_or(self.checker.ctx.error());
                    hir_patterns.push(self.check_pattern(pat, elem_ty, exist_depth)?);
                }
                Ok(HirPattern::Tuple(hir_patterns, *span))
            }
            Pattern::Slice(before, slice, after, span) => {
                let elem_ty = if self.checker.ctx.is_infer_var(expected_ty) {
                    let elem = self.new_infer_var(
                        TypeVariableKind::Any,
                        crate::hir::infer::VarOrigin::Synthetic,
                    );
                    let slice_ty = self.checker.ctx.slice(elem);
                    self.unify_with(expected_ty, slice_ty, *span, TypingContext::None)?;
                    elem
                } else if let Some(elem) = self.checker.ctx.elem_of_slice(expected_ty) {
                    elem
                } else if let Some(elem) = self.checker.ctx.elem_of_array(expected_ty) {
                    elem
                } else {
                    self.checker.diagnostics.push(
                        Diagnostic::error("slice pattern requires array or slice type")
                            .with_span(*span),
                    );
                    self.checker.ctx.error()
                };
                let mut hir_before = Vec::new();
                for pat in before {
                    hir_before.push(self.check_pattern(pat, elem_ty, exist_depth)?);
                }
                let hir_slice = slice
                    .as_ref()
                    .map(|pat| {
                        let slice_ty = self.checker.ctx.slice(elem_ty);
                        let pat: HirPattern<'input> =
                            self.check_pattern(pat, slice_ty, exist_depth)?;
                        Ok(Box::new(pat))
                    })
                    .transpose()?;
                let mut hir_after = Vec::new();
                for pat in after {
                    hir_after.push(self.check_pattern(pat, elem_ty, exist_depth)?);
                }
                Ok(HirPattern::Slice(hir_before, hir_slice, hir_after, *span))
            }
            Pattern::Error(span) => Ok(HirPattern::Error(*span)),
            Pattern::Struct {
                path, fields, span, ..
            } => {
                let def_id = self.checker.resolve_def_id(path)?;
                let binding = self
                    .checker
                    .symbols
                    .lookup_type_by_def_id(def_id)
                    .ok_or_else(|| Diagnostic::error("struct not found").with_span(*span))?;
                if !matches!(binding.kind, TypeKind::Struct) {
                    return Err(Diagnostic::error("pattern type is not a struct").with_span(*span));
                }
                let type_args: Vec<TypeId> = (0..binding.params.len())
                    .map(|_| {
                        self.new_infer_var(
                            TypeVariableKind::Unconstrained,
                            crate::hir::infer::VarOrigin::Synthetic,
                        )
                    })
                    .collect();
                let struct_ty = self.checker.ctx.struct_ty(def_id, type_args.clone());
                self.unify_with(expected_ty, struct_ty, *span, TypingContext::None)?;
                let mut subst = Subst::new();
                for (i, _) in binding.params.iter().enumerate() {
                    subst.insert(i, type_args[i]);
                }
                let mut hir_fields = Vec::new();
                for (name, pat) in fields {
                    let field_def =
                        binding
                            .fields
                            .iter()
                            .find(|f| f.name == *name)
                            .ok_or_else(|| {
                                let type_name = format!("{:?}", def_id);
                                Diagnostic::error_kind(DiagnosticKind::NoSuchField {
                                    field_name: name.to_string(),
                                    type_name,
                                    span: *span,
                                })
                                .with_code_str("E010")
                            })?;
                    let field_ty = self.checker.ctx.subst(field_def.ty, &subst);
                    hir_fields.push((
                        *name,
                        Box::new(self.check_pattern(pat, field_ty, exist_depth)?),
                    ));
                }
                Ok(HirPattern::Struct {
                    path: path.clone(),
                    fields: hir_fields,
                    rest: false, // lower_irrelevance is a type-level concern; rest is informational
                    span: *span,
                })
            }
            Pattern::Enum {
                path,
                variant,
                inner,
                span,
            } => {
                let def_id = if path.is_empty() {
                    // Bare variant like `Some(x)` — infer enum from expected type
                    let resolved = self.checker.ctx.resolve_binding(expected_ty);
                    match self.checker.ctx.get(resolved) {
                        TypeData::Adt { def_id, .. } => *def_id,
                        _ => {
                            return Err(Diagnostic::error(
                                "cannot infer enum type from bare variant pattern; use qualified path like `Opt::Some(x)`",
                            )
                            .with_span(*span));
                        }
                    }
                } else {
                    self.checker.resolve_def_id(path)?
                };
                let binding = self
                    .checker
                    .symbols
                    .lookup_type_by_def_id(def_id)
                    .ok_or_else(|| Diagnostic::error("enum not found").with_span(*span))?;
                if !matches!(binding.kind, TypeKind::Enum) {
                    return Err(Diagnostic::error("pattern type is not an enum").with_span(*span));
                }
                let type_args: Vec<TypeId> = (0..binding.params.len())
                    .map(|_| {
                        self.new_infer_var(
                            TypeVariableKind::Unconstrained,
                            crate::hir::infer::VarOrigin::Synthetic,
                        )
                    })
                    .collect();
                let enum_ty = self.checker.ctx.enum_ty(def_id, type_args.clone());
                self.unify_with(expected_ty, enum_ty, *span, TypingContext::None)?;
                let mut subst = Subst::new();
                for (i, _) in binding.params.iter().enumerate() {
                    subst.insert(i, type_args[i]);
                }
                let variant_def = binding
                    .variants
                    .iter()
                    .find(|v| v.name == *variant)
                    .ok_or_else(|| {
                        Diagnostic::error(format!("variant '{}' not found", variant))
                            .with_span(*span)
                    })?;
                // ── Existential GADT payload resolution ─────────────
                // Each existential variant pushes its OWN skolem scope,
                // unconditionally — skolem IDENTITY is the binder's index
                // in `exists_params`, never its name (GHC `realUnique` /
                // OCaml `id: int`).  Same-named `exists X` in outer and
                // inner variants therefore resolve to independent skolems.
                //
                // Push is done OUTSIDE the closure and pop AFTER the
                // closure returns, so a `?` early return inside the
                // closure cannot leak the scope (error-path safety).
                //
                // Frame identity (OCCURRENCE identity): only reuse the
                // stack top if `precreate_exist_skolems` pushed it for
                // THIS variant occurrence — same enum DefId + variant name
                // AND not yet consumed.  Same-named variants in different
                // enums (DefId differs), nested existential variants, and
                // recursive variants (frame already `used` by the top-level
                // pattern) all push a NEW frame.
                let pushed = if !variant_def.exists_params.is_empty() {
                    let top_is_this_variant = {
                        let stack = self.checker.ctx.gadt.exist_skolems.borrow();
                        // Only frames pushed at or after THIS arm's entry may
                        // be reused — on the shared stack, `last()` can
                        // otherwise be an OUTER arm's frame (nested arm
                        // matching the same variant), which would reuse the
                        // outer witness set instead of fresh skolems.
                        // `>=`: the precreated frame for THIS arm sits exactly
                        // at `exist_depth` (the wrapper captures the depth
                        // after the precreate push), so it must be reusable.
                        stack.len() >= exist_depth
                            && stack.last().map_or(false, |f| {
                                f.variant_name == *variant && f.def_id == def_id && !f.used
                            })
                    };
                    if top_is_this_variant {
                        // Reuse the precreate frame; mark it consumed so a
                        // recursive re-encounter pushes a fresh frame.
                        if let Some(f) = self.checker.ctx.gadt.exist_skolems.borrow_mut().last_mut()
                        {
                            f.used = true;
                        }
                        false
                    } else {
                        let skolems: Vec<TypeId> = variant_def
                            .exists_params
                            .iter()
                            .map(|_| self.checker.ctx.fresh_gadt_skolem())
                            .collect();
                        self.checker.ctx.gadt.exist_skolems.borrow_mut().push(
                            crate::hir::checker::ExistScopeFrame {
                                def_id,
                                variant_name: *variant,
                                used: false,
                                skolems,
                            },
                        );
                        #[cfg(debug_assertions)]
                        crate::hir::anya::trace_skolem_scope(
                            &self.checker.ctx.gadt.exist_skolems.borrow(),
                            &variant_def.exists_params,
                            variant,
                        );
                        true
                    }
                } else {
                    false
                };
                let result = (|| -> Result<HirPattern<'input>, Diagnostic> {
                    let exist_skolems = self
                        .checker
                        .ctx
                        .gadt
                        .exist_skolems
                        .borrow()
                        .last()
                        .map(|f| f.skolems.clone())
                        .unwrap_or_default();
                    // Collect this variant's `when` equalities for nested GADT
                    // refinement (SYNTAX.md §"Nested GADT Refinement"): every
                    // constructor in a nested pattern contributes its
                    // equalities, propagated to the branch body.  The actual
                    // type args come from `enum_ty` (type_args unified with
                    // the expected type).  Registration happens later in
                    // `apply_gadt_refinement` (after `push_gadt_arm`).
                    if variant_def.is_gadt() {
                        if let Ok((_, inner_args)) =
                            self.checker.resolve_type_to_struct_or_enum(enum_ty, *span)
                        {
                            for (pn, ct) in &variant_def.eq_spec {
                                self.checker.ctx.gadt.pending_eqs.push(
                                    crate::hir::checker::PendingInnerGadtEq {
                                        param_name: *pn,
                                        concrete_ty: ct.clone(),
                                        binding: binding.clone(),
                                        args: inner_args.clone(),
                                        exist_params: variant_def.exists_params.clone(),
                                        skolems: exist_skolems.clone(),
                                    },
                                );
                            }
                        }
                    }
                    let inner_ty = variant_def
                        .payload
                        .as_ref()
                        .map(|ty| {
                            // Same logic as EnumLit: substitute type params with concrete args.
                            if let Type::Path(p, _) = ty {
                                if p.len() == 1 {
                                    if let Some((i, _)) = binding
                                        .params
                                        .iter()
                                        .enumerate()
                                        .find(|(_, tp)| tp.name == p[0])
                                    {
                                        let gp = self.checker.ctx.generic_param(i, p[0]);
                                        return Ok(self.checker.ctx.subst(gp, &subst));
                                    }
                                    // Check if the payload type is an existential
                                    // param: resolve by INDEX in exists_params.
                                    if let Some(&skolem) = variant_def
                                        .exists_params
                                        .iter()
                                        .position(|ep| ep == &p[0])
                                        .and_then(|i| exist_skolems.get(i))
                                    {
                                        return Ok(skolem);
                                    }
                                }
                            }
                            // For payload types that reference existential params
                            // (e.g., `&[X]` where X is an `exists` param), use the
                            // skolem-substituted resolution path.
                            if !exist_skolems.is_empty() {
                                self.checker
                                    .resolve_type_with_skolems(
                                        ty,
                                        &variant_def.exists_params,
                                        &exist_skolems,
                                    )
                                    .ok_or_else(|| {
                                        Diagnostic::error(format!(
                                            "cannot resolve type for variant '{}' with existential parameters",
                                            variant,
                                        ))
                                        .with_span(*span)
                                    })
                            } else {
                                self.resolve_type(ty)
                            }
                        })
                        .unwrap_or(Ok(self.checker.ctx.error()))?;
                    let inner_hir = inner
                        .as_ref()
                        .map(|inner| self.check_pattern(inner, inner_ty, exist_depth))
                        .transpose()?;
                    Ok(HirPattern::Enum {
                        path: path.clone(),
                        variant: *variant,
                        inner: inner_hir.map(Box::new),
                        span: *span,
                    })
                })();
                // Pop AFTER the closure returns (success or `?` early
                // return) so the scope never leaks on the error path.
                if pushed {
                    self.checker.ctx.gadt.exist_skolems.borrow_mut().pop();
                }
                result
            }
            Pattern::Or(patterns, span) => {
                let mut hir_patterns = Vec::new();
                // ── or-pattern GADT refinement: per-alternative collection ─
                // Each alternative is checked independently; its GADT `when`
                // equalities are collected in isolation (NOT conjoined into
                // the shared pending_eqs).  Afterwards the intersection is
                // computed (rules 1-6): all alternatives agree → propagate;
                // conflict → E066; some alternative unconstrained → do not
                // propagate (T stays abstract).
                let mut alt_eqs: Vec<Vec<crate::hir::checker::PendingInnerGadtEq>> = Vec::new();
                let mut alt_reachable: Vec<bool> = Vec::new();
                for pat in patterns {
                    // Per-alternative reachability  : only Enum
                    // patterns can be GADT-unreachable under the scrutinee
                    // type; non-Enum alternatives are always reachable.
                    // Unreachable alternatives are warned about and their
                    // equalities are ignored by the intersection.
                    let reachable = match pat {
                        crate::ast::Pattern::Enum { .. } => {
                            self.checker
                                .is_gadt_variant_reachable(expected_ty, pat, *span)
                        }
                        _ => true,
                    };
                    if !reachable {
                        self.checker.diagnostics.push(
                            Diagnostic::warning(
                                "or-pattern alternative is unreachable for the scrutinee type",
                            )
                            .with_span(*span),
                        );
                    }
                    // Use check_pattern_inner (not check_pattern) to avoid
                    // registering each sub-pattern's bindings individually —
                    // or-pattern bindings are collected across ALL
                    // alternatives, checked for name-set agreement and type
                    // compatibility, and registered ONCE into scope by
                    // `register_pattern_bindings`' Or branch.
                    let before = self.checker.ctx.gadt.pending_eqs.len();
                    let p = self.check_pattern_inner(pat, expected_ty, exist_depth)?;
                    let collected: Vec<_> =
                        self.checker.ctx.gadt.pending_eqs.drain(before..).collect();
                    hir_patterns.push(p);
                    alt_eqs.push(collected);
                    alt_reachable.push(reachable);
                }
                self.checker
                    .apply_or_alt_intersection(&alt_eqs, &alt_reachable, *span);
                // ── Or-pattern bindings ─────────────────────────
                // "Both patterns must bind the same set
                // of variables with compatible types."  Collect each
                // alternative's bindings; verify the name sets agree (E105,
                // OCaml's `Orpat_vars`); unify each common name's types
                // across alternatives (E106, OCaml's
                // `Or_pattern_type_clash`).  The unified bindings are
                // registered into scope by `register_pattern_bindings`' Or
                // branch — all alternatives bind the same names, so the
                // first alternative's walk covers every binding.
                let mut name_types: Vec<(Symbol, Vec<TypeId>)> = Vec::new();
                for (i, p) in hir_patterns.iter().enumerate() {
                    let mut binds = Vec::new();
                    collect_pattern_bindings(p, &mut binds);
                    if i == 0 {
                        for (n, ty, _) in &binds {
                            name_types.push((*n, vec![*ty]));
                        }
                    } else {
                        for (n, ty, _) in &binds {
                            if let Some((_, tys)) = name_types.iter_mut().find(|(n0, _)| n0 == n) {
                                tys.push(*ty);
                            } else {
                                self.checker.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "or-pattern alternatives must bind the same variables: `{}` is missing from an earlier alternative",
                                        n,
                                    ))
                                    .with_code_str("E105")
                                    .with_span(*span),
                                );
                            }
                        }
                        for (n, _) in &name_types {
                            if !binds.iter().any(|(n1, _, _)| n1 == n) {
                                self.checker.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "or-pattern alternatives must bind the same variables: `{}` is missing from this alternative",
                                        n,
                                    ))
                                    .with_code_str("E105")
                                    .with_span(*span),
                                );
                            }
                        }
                    }
                }
                // Type compatibility across alternatives (E106).
                for (n, tys) in &name_types {
                    for ty in tys.iter().skip(1) {
                        if self
                            .unify_with(tys[0], *ty, *span, TypingContext::None)
                            .is_err()
                        {
                            self.checker.diagnostics.push(
                                Diagnostic::error(format!(
                                    "or-pattern binding `{}` has incompatible types across alternatives",
                                    n,
                                ))
                                .with_code_str("E106")
                                .with_span(*span),
                            );
                            break;
                        }
                    }
                }
                Ok(HirPattern::Or(hir_patterns, *span))
            }
            _ => {
                self.checker
                    .diagnostics
                    .push(Diagnostic::error("unsupported pattern type").with_span(Span::new(0, 0)));
                Ok(HirPattern::Error(Span::new(0, 0)))
            }
        }
    }

    /// Check if a function name has `@deprecated` or `@experimental` attributes
    /// and emit appropriate warnings/errors.
    fn check_deprecated_experimental(
        &mut self,
        name: &Symbol,
        attributes: &[Attribute<'input>],
        span: Span,
    ) {
        // @deprecated and @experimental are mutually exclusive — a function
        // can't be both "already established but not recommended" and "newly
        // introduced".  @experimental (error) takes priority over @deprecated
        // (warning) when both are present.
        let mut has_experimental_error = false;
        for attr in attributes {
            if attr.name.eq_str("experimental") && !self.checker.enable_experimental {
                has_experimental_error = true;
                let msg = format!("use of experimental function `{}`", name);
                self.checker.diagnostics.push(
                    Diagnostic::error(msg)
                        .with_code_str("E094")
                        .with_span(span)
                        .with_help(
                            "experimental features are not enabled; use `--enable-experimental` to use this function",
                        ),
                );
            }
        }
        if !has_experimental_error {
            for attr in attributes {
                if attr.name.eq_str("deprecated") {
                    let msg = if let Some(Expr::Literal(Literal::String(reason), _)) =
                        attr.args.first()
                    {
                        format!("use of deprecated function `{}`: {}", name, reason)
                    } else {
                        format!("use of deprecated function `{}`", name)
                    };
                    self.checker.diagnostics.push(
                        Diagnostic::warning(msg)
                            .with_code_str("W090")
                            .with_span(span)
                            .with_help("consider migrating to a replacement function"),
                    );
                }
            }
        }
    }
}

/// Collect every variable binding (`HirPattern::Ident`) in a pattern,
/// recursively, into `out` as `(name, type, span)` triples.
fn collect_pattern_bindings<'input>(
    pattern: &HirPattern<'input>,
    out: &mut Vec<(Symbol, TypeId, Span)>,
) {
    // Recurse into every child pattern in the slice.
    fn collect_many<'input>(pats: &[HirPattern<'input>], out: &mut Vec<(Symbol, TypeId, Span)>) {
        for p in pats {
            collect_pattern_bindings(p, out);
        }
    }
    match pattern {
        HirPattern::Ident(name, ty, span) => out.push((*name, *ty, *span)),
        HirPattern::Tuple(elems, _) => collect_many(elems, out),
        HirPattern::Slice(before, rest, after, _) => {
            collect_many(before, out);
            if let Some(p) = rest {
                collect_pattern_bindings(p, out);
            }
            collect_many(after, out);
        }
        HirPattern::Struct { fields, .. } => {
            for (_, p) in fields {
                collect_pattern_bindings(p, out);
            }
        }
        HirPattern::Enum { inner: Some(p), .. } => collect_pattern_bindings(p, out),
        HirPattern::Or(pats, _) => collect_many(pats, out),
        _ => {}
    }
}

/// Walk a checked pattern and register every `HirPattern::Ident` binding
/// into `local_variable_types` (and `runtime_var_scopes`) so the body of
/// if-let / while-let / for / match can reference the bound variable, and
/// so a runtime pattern binding SHADOWS an outer ghost variable during
/// `scope_cleanup when` predicate checking.
pub(super) fn register_pattern_bindings<'input>(
    local_variable_types: &mut ScopedVarMap,
    runtime_var_scopes: &std::rc::Rc<std::cell::RefCell<Vec<std::collections::HashSet<Symbol>>>>,
    pattern: &HirPattern<'input>,
) {
    // Recurse into every child pattern in the slice.
    fn register_many<'input>(
        local_variable_types: &mut ScopedVarMap,
        runtime_var_scopes: &std::rc::Rc<
            std::cell::RefCell<Vec<std::collections::HashSet<Symbol>>>,
        >,
        pats: &[HirPattern<'input>],
    ) {
        for p in pats {
            register_pattern_bindings(local_variable_types, runtime_var_scopes, p);
        }
    }
    match pattern {
        HirPattern::Ident(name, ty, _) => {
            local_variable_types.insert(*name, *ty);
            if let Some(rscope) = runtime_var_scopes.borrow_mut().last_mut() {
                rscope.insert(*name);
            }
        }
        HirPattern::Tuple(patterns, _) => {
            register_many(local_variable_types, runtime_var_scopes, patterns);
        }
        HirPattern::Slice(before, rest, after, _) => {
            register_many(local_variable_types, runtime_var_scopes, before);
            if let Some(p) = rest {
                register_pattern_bindings(local_variable_types, runtime_var_scopes, p);
            }
            register_many(local_variable_types, runtime_var_scopes, after);
        }
        HirPattern::Struct { fields, .. } => {
            for (_, p) in fields {
                register_pattern_bindings(local_variable_types, runtime_var_scopes, p);
            }
        }
        HirPattern::Enum { inner: Some(p), .. } => {
            register_pattern_bindings(local_variable_types, runtime_var_scopes, p);
        }
        HirPattern::Or(patterns, _) => {
            // Or-pattern bindings: all alternatives bind the SAME set
            // of variables (enforced by the `Pattern::Or` checker — E105),
            // with types unified across alternatives (E106), so registering
            // the first alternative's bindings covers every binding and
            // they resolve consistently.
            if let Some(first) = patterns.first() {
                register_pattern_bindings(local_variable_types, runtime_var_scopes, first);
            }
        }
        _ => {}
    }
}
