use super::*;
use crate::ast::visit::replace_ident_in_expr;

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
    fn ctx(&mut self) -> &mut TypeContext {
        self.checker.ctx
    }
    fn infer(&mut self) -> &mut InferenceContext {
        &mut self.checker.infer
    }

    /// Suggest a cast for common type mismatches (e.g. Int ↔ Float).
    pub fn suggest_cast(&self, expected: TypeId, actual: TypeId) -> Option<String> {
        let (e, a) = (self.checker.ctx.get(expected), self.checker.ctx.get(actual));
        match (e, a) {
            (TypeData::Int { .. }, TypeData::Float { .. })
            | (TypeData::Float { .. }, TypeData::Int { .. }) => {
                Some("try using `as` to cast between integer and float types".into())
            }
            (TypeData::Bool, TypeData::Int { .. }) => {
                Some("try `x != 0` to convert Int to Bool".into())
            }
            (TypeData::Int { .. }, TypeData::Bool) => {
                Some("try `if x { 1 } else { 0 }` to convert Bool to Int".into())
            }
            _ => None,
        }
    }

    pub fn unify(
        &mut self,
        expected: TypeId,
        actual: TypeId,
        span: Span,
    ) -> Result<(), Diagnostic> {
        self.checker
            .ctx
            .unify(expected, actual)
            .map(|_| ())
            .map_err(|_err| {
                let msg = format!(
                    "type mismatch: expected {:?}, found {:?}",
                    self.checker.ctx.get(expected),
                    self.checker.ctx.get(actual)
                );
                let mut diag = Diagnostic::error(msg).with_code_str("E030").with_span(span);
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion);
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
            .unify(expected, actual)
            .map(|_| ())
            .map_err(|_err| {
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
                    TypingContext::None => format!(
                        "type mismatch: expected {:?}, found {:?}",
                        self.checker.ctx.get(expected),
                        self.checker.ctx.get(actual)
                    ),
                    TypingContext::Index => format!(
                        "index must be an integer, got {:?}",
                        self.checker.ctx.get(actual)
                    ),
                };
                let mut diag = Diagnostic::error(msg)
                    .with_code_str("E030")
                    .with_span(span);
                if let Some(suggestion) = self.suggest_cast(expected, actual) {
                    diag = diag.with_suggestion(suggestion);
                }
                diag
            })
    }

    // ── Infer expression type ─────────────────────────────────────────────
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<(HirExpr, TypeId), Diagnostic> {
        match expr {
            Expr::Literal(lit, span) => {
                let kind = match lit {
                    Literal::Int(_) => TypeVariableKind::Integer,
                    Literal::Float(_) => TypeVariableKind::Float,
                    Literal::Bool(_) => TypeVariableKind::Bool,
                    Literal::Char(_) | Literal::String(_) | Literal::ByteString(_) => {
                        TypeVariableKind::Any
                    }
                };
                let ty = self.new_infer_var(kind);
                Ok((HirExpr::Literal(lit.clone(), ty, *span), ty))
            }
            Expr::Ident(name, span) => {
                // Check the local variable type cache first (set by VariableDef)
                if let Some(ty) = self.checker.local_variable_types.get(*name) {
                    // Reading a mutable global outside @trusted is forbidden
                    if self.checker.mutable_globals.contains(name) && !self.checker.current_function_trusted {
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
                    Ok((HirExpr::Ident(name.clone(), ty, *span), ty))
                } else if let Some(binding) = self.checker.symbols.lookup_variable(*name, *span) {
                    Ok((HirExpr::Ident(*name, binding.ty, *span), binding.ty))
                } else if let Some(func) = self.checker.symbols.lookup_function(*name) {
                    let sig = &func.signature;
                    // Construct the function type: Fn(params..., ret)
                    let mut fn_ty = self
                        .checker
                        .ctx
                        .function(sig.params.iter().map(|p| p.ty).collect(), sig.return_type);
                    // If the function has type parameters, wrap with Forall:
                    // def foo<T, U>(x: T, y: U) → Forall(0, "T", Forall(1, "U", Fn(...)))
                    if !sig.type_params.is_empty() {
                        for (i, tp) in sig.type_params.iter().enumerate().rev() {
                            fn_ty = self.checker.ctx.forall(i, tp.name.clone(), fn_ty);
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
                let (left_hir, left_ty) = self.infer_expr(left)?;
                let (right_hir, right_ty) = self.infer_expr(right)?;
                let result_ty =
                    self.checker.binary_op_type(*op, left_ty, right_ty, *span)?;
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
                let (hir, ty) = self.infer_expr(expr)?;
                let result_ty = match op {
                    UnaryOp::Neg | UnaryOp::BitNot => ty,
                    UnaryOp::Not => self.checker.ctx.bool(),
                    UnaryOp::Deref => self
                        .checker
                        .ctx
                        .pointee_of_ref(ty)
                        .or_else(|| self.checker.ctx.pointee_of_pointer(ty))
                        .unwrap_or(self.checker.ctx.error()),
                    UnaryOp::Ref | UnaryOp::RefMut => {
                        let mutable = matches!(op, UnaryOp::RefMut);
                        self.checker.ctx.reference(ty, mutable)
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
            Expr::Call {
                callee,
                args,
                comptime,
                span,
            } => {
                // Check if this is a method call (x.foo()) rather than a free function call
                if let Expr::FieldAccess { base, field, .. } = callee.as_ref() {
                    let (base_hir, base_ty) = self.infer_expr(base)?;
                    if let Some((param_tys, ret_ty)) = self.checker.lookup_method(base_ty, *field) {
                        // Adjust: method calls pass `self` as the first arg implicitly,
                        // so the param list from the declaration includes self.
                        // We treat `base` as the receiver and check remaining args.
                        // Unify the receiver type with the `self` parameter type.
                        if !param_tys.is_empty() {
                            let self_param_ty = param_tys[0];
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
                                        .unify_with(
                                            self_param_ty,
                                            ref_base,
                                            *span,
                                            TypingContext::None,
                                        )
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
                        let mut hir_args = Vec::new();
                        for (i, arg) in args.iter().enumerate() {
                            let expected = explicit_param_tys
                                .get(i)
                                .copied()
                                .unwrap_or(self.checker.ctx.error());
                            let hir_arg = self.check_expr(
                                arg,
                                Expectation::HasType(expected),
                                TypingContext::Argument {
                                    index: i,
                                    total: args.len(),
                                },
                            )?;
                            hir_args.push(hir_arg);
                        }
                        // Build the HIR: the callee is the field access; we keep it as-is
                        let callee_hir = HirExpr::FieldAccess {
                            base: Box::new(base_hir),
                            field: field.clone(),
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
                        let mut diag = Diagnostic::error(format!(
                            "no method named `{}` found for type",
                            field
                        ))
                        .with_code_str("E011")
                        .with_span(*span);
                        if !method_names.is_empty() {
                            diag = diag.with_suggestion(format!(
                                "available methods: {}",
                                method_names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                            ));
                        }
                        self.checker.diagnostics.push(diag);
                        return Ok((HirExpr::Error(*span), self.checker.ctx.error()));
                    }
                }

                // Check if this is a static method call: `Type::method(args)`
                if let Expr::Path(path, _) = callee.as_ref() {
                    if path.len() >= 2 {
                        // Resolve the type from the first path segment.
                        let type_name = path[0].clone();
                        let method_name = path[1].clone();
                        let type_path = Type::Path(vec![type_name], *span);
                        if let Ok(ty) = self.resolve_type(&type_path) {
                            // Look up the method on the resolved type.
                            // lookup_method also handles inherent methods.
                            if let Some((param_tys, ret_ty)) = self.checker.lookup_method(ty, method_name) {
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
                                let mut hir_args = Vec::new();
                                for (i, arg) in args.iter().enumerate() {
                                    let expected = param_tys
                                        .get(i)
                                        .copied()
                                        .unwrap_or(self.checker.ctx.error());
                                    let hir_arg = self.check_expr(
                                        arg,
                                        Expectation::HasType(expected),
                                        TypingContext::Argument {
                                            index: i,
                                            total: args.len(),
                                        },
                                    )?;
                                    hir_args.push(hir_arg);
                                }
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
                }

                let (callee_hir, callee_ty) = self.infer_expr(callee)?;

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
                    let mut hir_args = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let expected = param_tys
                            .get(i)
                            .copied()
                            .unwrap_or(self.checker.ctx.error());
                        let hir_arg = self.check_expr(
                            arg,
                            Expectation::HasType(expected),
                            TypingContext::Argument {
                                index: i,
                                total: args.len(),
                            },
                        )?;
                        hir_args.push(hir_arg);
                    }
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
                    self.checker.diagnostics.push(
                        Diagnostic::error("called expression is not a function").with_span(*span),
                    );
                    Ok((HirExpr::Error(*span), self.checker.ctx.error()))
                }
            }
            Expr::Index { base, index, span } => {
                let (base_hir, base_ty) = self.infer_expr(base)?;
                let (index_hir, index_ty) = self.infer_expr(index)?;
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
                let (base_hir, base_ty) = self.infer_expr(base)?;
                // Try to resolve as a struct field first
                if let Ok(field_ty) = self.checker.lookup_field(base_ty, *field, *span) {
                    return Ok((
                        HirExpr::FieldAccess {
                            base: Box::new(base_hir),
                            field: field.clone(),
                            ty: field_ty,
                            span: *span,
                        },
                        field_ty,
                    ));
                }
                // If not a field, try as a method via autoderef
                if let Some((param_tys, ret_ty)) = self.checker.lookup_method(base_ty, *field) {
                    // Full function type including self parameter: fn(&Obj) -> RetTy
                    let fn_ty = self.checker.ctx.function(param_tys, ret_ty);
                    return Ok((
                        HirExpr::FieldAccess {
                            base: Box::new(base_hir),
                            field: field.clone(),
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
                let (base_hir, base_ty) = self.infer_expr(base)?;
                let attr_ty = self.checker.lookup_attr(base_ty, *attr, *span)?;
                Ok((
                    HirExpr::AttrAccess {
                        base: Box::new(base_hir),
                        attr: attr.clone(),
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
                let (hir, actual_ty) = self.infer_expr(expr)?;
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
                let start_hir = start
                    .as_ref()
                    .map(|s| self.infer_expr(s).map(|(h, _)| h))
                    .transpose()?;
                let end_hir = end
                    .as_ref()
                    .map(|e| self.infer_expr(e).map(|(h, _)| h))
                    .transpose()?;
                let int_ty = self.checker.ctx.int(32, true);
                let ty = self.checker.ctx.tuple(vec![int_ty, int_ty]);
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
                let resolved_ty = self.resolve_type(&Type::Path(path.clone(), *span))?;
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
                                let mut diag = Diagnostic::error(format!(
                                    "field '{}' not found in struct",
                                    name
                                ))
                                .with_code_str("E010")
                                .with_span(*span)
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
                    let hir = self.check_expr(
                        value,
                        Expectation::HasType(field_ty),
                        TypingContext::StructFieldInit,
                    )?;
                    self.unify_with(field_ty, hir.ty(), *span, TypingContext::StructFieldInit)?;
                    hir_fields.push((name.clone(), Box::new(hir)));
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
            Expr::EnumLit {
                path,
                variant,
                payload,
                span,
            } => {
                let resolved_ty = self.resolve_type(&Type::Path(path.clone(), *span))?;
                let (def_id, args) = self
                    .checker
                    .resolve_type_to_struct_or_enum(resolved_ty, *span)?;
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
                    if let Some((method_param_tys, ret_ty)) =
                        self.checker.lookup_method(resolved_ty, *variant)
                    {
                        let mut hir_args = Vec::new();
                        // Pass the payload (if any) as the argument.
                        if let Some(p) = &payload {
                            let expected = method_param_tys.first().copied().unwrap_or(self.checker.ctx.error());
                            let hir_arg = self.check_expr(
                                p,
                                Expectation::HasType(expected),
                                TypingContext::Argument { index: 0, total: 1 },
                            )?;
                            hir_args.push(hir_arg);
                        }
                        let callee_hir = HirExpr::Ident(variant.clone(), ret_ty, *span);
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
                        Diagnostic::error(format!("variant '{}' not found", variant))
                            .with_span(*span)
                    })?;
                // Resolve the payload type, substituting type params with concrete args.
                // For example, `Option<T>` with `T = Int<32>` means the payload type
                // `T` should resolve to the `GenericParam` TypeId, which will be
                // unified with the concrete arg via the subst.
                let payload_ty = variant_def
                    .payload
                    .as_ref()
                    .map(|ty| {
                        // If the payload type is a bare type param name (e.g. `T` in
                        // `type Option<T> = enum { None, Some(T) }`), resolve it to
                        // the corresponding GenericParam TypeId so that substitution
                        // with the concrete args works correctly.
                        if let Type::Path(p, _) = ty {
                            if p.len() == 1 {
                                if let Some((i, _)) = binding.params.iter().enumerate().find(|(_, tp)| tp.name == p[0]) {
                                    let gp = self.checker.ctx.generic_param(i, p[0].clone());
                                    let result = self.checker.ctx.subst(gp, &subst);
                                    return Ok(result);
                                }
                            }
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
                        variant: variant.clone(),
                        payload: payload_hir,
                        ty: enum_ty,
                        span: *span,
                    },
                    enum_ty,
                ))
            }
            Expr::Move(expr, span) => {
                let (hir, ty) = self.infer_expr(expr)?;
                Ok((HirExpr::Move(Box::new(hir), ty, *span), ty))
            }
            Expr::Tuple(exprs, span) => {
                let mut hirs = Vec::new();
                let mut types = Vec::new();
                for e in exprs {
                    let (hir, ty) = self.infer_expr(e)?;
                    hirs.push(hir);
                    types.push(ty);
                }
                let ty = self.checker.ctx.tuple(types);
                Ok((HirExpr::Tuple(hirs, ty, *span), ty))
            }
            Expr::Array(exprs, span) => {
                let mut hirs = Vec::new();
                let mut elem_ty = None;
                for e in exprs {
                    let (hir, ty) = self.infer_expr(e)?;
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
                        .unwrap_or_else(|| Ok(self.new_infer_var(TypeVariableKind::Any)))?;
                    hir_params.push(HirParam {
                        name: param.name.clone(),
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
                    self.checker.local_variable_types.insert(p.name.clone(), p.ty);
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
            Expr::Try { expr, span } => {
                let (hir, ty) = self.infer_expr(expr)?;
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
            Expr::UnsafeBlock { body, span } => {
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
            Expr::Catch {
                expr,
                branches,
                span,
            } => {
                let (expr_hir, expr_ty) = self.infer_expr(expr)?;
                let (ok_ty, error_ty) = self.checker.extract_result_types(expr_ty, *span)?;
                let mut hir_branches = Vec::new();
                for branch in branches {
                    let _scope = self.checker.enter_var_scope();
                    let pattern_hir = self.check_pattern(&branch.pattern, error_ty)?;
                    let body_hir = self.check_block(&branch.body)?;
                    hir_branches.push(HirCatchBranch {
                        pattern: pattern_hir,
                        bind: branch.bind.clone(),
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
            Expr::LeaveWith { expr, span } => {
                let (hir, err_ty) = self.infer_expr(expr)?;
                // Validate that the error type matches the function's error type
                if let Some(ret_ty) = self.checker.current_return_type {
                    if let Ok((_, error_ty)) = self.checker.extract_result_types(ret_ty, *span) {
                        self.unify_with(error_ty, err_ty, *span, TypingContext::None)?;
                    }
                }
                let never = self.checker.ctx.never();
                Ok((
                    HirExpr::LeaveWith {
                        expr: Box::new(hir),
                        ty: never,
                        span: *span,
                    },
                    never,
                ))
            }
            Expr::Await { expr, span } => {
                let (hir, ty) = self.infer_expr(expr)?;
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
                let (cond_hir, cond_ty) = self.infer_expr(cond)?;
                let cond_is_bool = self.checker.ctx.is_bool(cond_ty)
                    || matches!(self.checker.ctx.get(cond_ty), TypeData::InferVar { id }
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
                // Divergence detection: if both branches end in return/leave/continue, result is never
                let then_diverges = then_hir.last().map_or(false, |s| {
                    matches!(
                        s,
                        HirStmt::Return { .. } | HirStmt::Leave { .. } | HirStmt::Continue { .. }
                    )
                });
                let else_diverges = else_hir.as_ref().and_then(|h| h.last()).map_or(false, |s| {
                    matches!(
                        s,
                        HirStmt::Return { .. } | HirStmt::Leave { .. } | HirStmt::Continue { .. }
                    )
                });
                let both_diverge = then_diverges && else_diverges;
                if *is_expression && !both_diverge {
                    if then_ty != else_ty {
                        self.checker.ctx.unify(then_ty, else_ty).ok();
                    }
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
            Expr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee)?;
                // Enter scope so the pattern binding is scoped to the then-branch
                let (pattern_hir, then_hir) = {
                    let _scope = self.checker.enter_var_scope();
                    let p = self.check_pattern(pattern, scrut_ty)?;
                    let t = self.check_block(then_branch)?;
                    (p, t)
                }; // _scope dropped: pattern + then-branch bindings removed
                let else_hir = else_branch
                    .as_ref()
                    .map(|b| self.check_block(b))
                    .transpose()?;
                let ty = self.checker.ctx.unit();
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
            Expr::Match {
                scrutinee,
                arms,
                span,
            } => {
                let (scrut_hir, scrut_ty) = self.infer_expr(scrutinee)?;
                let mut hir_arms = Vec::new();
                let mut arm_ty = None;
                for arm in arms {
                    // Each arm introduces pattern bindings in its own scope
                    let _scope = self.checker.enter_var_scope();
                    let pattern_hir = self.check_pattern(&arm.pattern, scrut_ty)?;
                    let guard_hir = arm
                        .guard
                        .as_ref()
                        .map(|g| {
                            self.infer_expr(g).map(|(h, ty)| {
                                if !self.checker.ctx.is_bool(ty) {
                                    self.checker.diagnostics.push(
                                        Diagnostic::error("match guard must be boolean")
                                            .with_span(arm.span),
                                    );
                                }
                                Box::new(h)
                            })
                        })
                        .transpose()?;
                    let (body_hir, body_ty) = self.infer_expr(&arm.body)?;
                    if let Some(prev) = arm_ty {
                        self.unify_with(prev, body_ty, arm.span, TypingContext::None)?;
                    } else {
                        arm_ty = Some(body_ty);
                    }
                    hir_arms.push(HirMatchArm {
                        pattern: pattern_hir,
                        guard: guard_hir,
                        body: Box::new(body_hir),
                        span: arm.span,
                    });
                    // scope drops here — removes pattern + body bindings
                }
                let result_ty = arm_ty.unwrap_or(self.checker.ctx.unit());

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
                                    if let HirPattern::Enum { variant, .. } = p {
                                        if !covered_variants.contains(&variant.as_str()) {
                                            covered_variants.push(variant.as_str());
                                        }
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
                        let total_variants = binding.variants.len();
                        if total_variants > 0 && covered_variants.len() < total_variants {
                            let msg = binding.missing_match.clone().unwrap_or_else(|| {
                                format!(
                                    "non-exhaustive match: covered {}/{} variants; add missing arms or a `_` wildcard",
                                    covered_variants.len(),
                                    total_variants,
                                )
                            });
                            self.checker
                                .diagnostics
                                .push(Diagnostic::error(msg).with_span(*span));
                        }
                        // Path A.2: @exhaustive forbids wildcard
                        if binding.exhaustive && has_wildcard && total_variants > 0 {
                            self.checker.diagnostics.push(
                                Diagnostic::error(
                                    "`@exhaustive` enum does not allow `_` wildcard; list all variants explicitly"
                                ).with_span(*span)
                            );
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
                    let inferred_count: Option<usize> =
                        match self.checker.ctx.get(resolved_scrut_ty) {
                            TypeData::InferVar { id } => {
                                match self.checker.infer.get_var_kind(*id) {
                                    Some(TypeVariableKind::Bool) => Some(2),
                                    _ => None,
                                }
                            }
                            _ => None,
                        };
                    // inferred_count takes priority over characteristic for unresolved vars
                    let total_count = inferred_count.or(total_count_from_char);
                    match total_count {
                        Some(n) if n <= 256 && covered_variants.len() < (n as usize) => {
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
                let (range_hir, _range_ty) = self.infer_expr(range)?;
                let (body_hir, _body_ty) = self.infer_expr(body)?;
                let bool_ty = self.checker.ctx.bool();
                Ok((
                    HirExpr::Quantified {
                        quantifier: *quantifier,
                        binder: binder.clone(),
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
                let (hir_expr, inner_ty) = self.infer_expr(expr)?;
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
                let (hir, ty) = self.infer_expr(expr)?;
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
                let (hir_expr, outer_ty) = self.infer_expr(expr)?;
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
                        );
                        self.checker.ctx.unify(root, inst_ty).ok();
                        Ok((
                            HirExpr::PolyUnbox {
                                expr: Box::new(hir_expr),
                                ty: root,
                                span: *span,
                            },
                            root,
                        ))
                    }
                    TypeData::InferVar { id: _ } => {
                        let result_ty = self.checker.infer.new_type_var(
                            self.checker.ctx,
                            crate::hir::infer::TypeVariableKind::Any,
                        );
                        Ok((
                            HirExpr::PolyUnbox {
                                expr: Box::new(hir_expr),
                                ty: result_ty,
                                span: *span,
                            },
                            result_ty,
                        ))
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
                    Diagnostic::error(format!("unresolved path: {}", path.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("::")))
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
            Expr::CompileError(msg, span) => {
                let diag = Diagnostic::error(msg.clone())
                    .with_code_str("E099")
                    .with_help("`@compile_error` halts compilation unconditionally when evaluated")
                    .with_span(*span);
                self.checker.diagnostics.push(diag);
                Ok((HirExpr::CompileError(msg.clone(), *span), self.checker.ctx.error()))
            }
            Expr::Task { body, span } => {
                let block = self.check_block(body)?;
                let ty = self.checker.ctx.unit();
                Ok((HirExpr::Task { block, ty, span: *span }, ty))
            }
        }
    }

    /// Check expression against a known type (bidirectional).
    ///
    /// First infers the expression's type, then unifies it with the
    /// expected type when one is provided (e.g. annotated variable
    /// declarations, function argument checking).
    pub fn check_expr(
        &mut self,
        expr: &Expr,
        expected: Expectation,
        ctx: TypingContext,
    ) -> Result<HirExpr, Diagnostic> {
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
            let (callee_hir, callee_ty) = self.infer_expr(callee)?;
            // Try type argument synthesis with the expected return type hint.
            if let Ok(Some((hir, _))) = self.checker.try_synthesize_type_args(
                &callee_hir,
                callee_ty,
                call_args,
                *comptime,
                expected_ty,
                *span,
            ) {
                return Ok(hir);
            }
            // Fall through to normal call handling via infer_expr.
        }
        let (hir, ty) = self.infer_expr(expr)?;
        if let Expectation::HasType(expected_ty) = expected {
            // Check kind compatibility before unification:
            // if the inferred type is an InferVar with a kind constraint
            // (e.g. Bool from a `true` literal, Integer from `42`),
            // verify that the expected type is compatible with that kind.
            self.check_kind_compat(ty, expected_ty, hir.span())?;
            self.check_kind_compat(expected_ty, ty, hir.span())?;
            self.unify_with(expected_ty, ty, hir.span(), ctx)?;
        }
        Ok(hir)
    }

    /// Check that an InferVar's kind constraint is compatible with the
    /// resolved type of another type.  This prevents situations like
    /// `true` (InferVar with kind Bool) being unified with `Int<32>`.
    /// Only fires when the other side resolves to a concrete (non-type-variable) type.
    fn check_kind_compat(
        &self,
        maybe_var: TypeId,
        other: TypeId,
        span: Span,
    ) -> Result<(), Diagnostic> {
        self.checker.check_kind_compat(maybe_var, other, span)
    }

    /// Resolve a syntactic type to a TypeId — actual implementation.
    pub fn resolve_type(&mut self, ty: &Type) -> Result<TypeId, Diagnostic> {
        match ty {
            Type::Path(path, span) => {
                // Lifetime parsed as placeholder path `["'a"]` — skip resolution.
                if path.len() == 1 && path[0].as_str().starts_with('\'') {
                    return Ok(self.checker.ctx.unit());
                }
                if let Ok(def_id) = self.checker.resolve_def_id(path) {
                    // Check if this is a generic type parameter (sentinel from resolve_def_id)
                    if def_id == DefId(usize::MAX - 1) {
                        if path.len() == 1 {
                            if let Some(&ty) = self.checker.local_type_param_cache.get(&path[0]) {
                                return Ok(ty);
                            }
                        }
                        return Err(Diagnostic::error(format!("type '{}' not found", path[0]))
                            .with_span(*span));
                    }
                    let binding: TypeBinding;
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
                                .unwrap_or(Err(
                                    Diagnostic::error("alias has no body").with_span(*span)
                                ));
                            self.checker.resolving_aliases.remove(&def_id);
                            result
                        }
                        TypeKind::Struct => {
                            if binding.params.is_empty() {
                                Ok(self.checker.ctx.struct_ty(def_id, vec![]))
                            } else {
                                let args: Vec<TypeId> = (0..binding.params.len())
                                    .map(|_| self.new_infer_var(TypeVariableKind::Unconstrained))
                                    .collect();
                                Ok(self.checker.ctx.struct_ty(def_id, args))
                            }
                        }
                        TypeKind::Enum => {
                            if binding.params.is_empty() {
                                Ok(self.checker.ctx.enum_ty(def_id, vec![]))
                            } else {
                                let args: Vec<TypeId> = (0..binding.params.len())
                                    .map(|_| self.new_infer_var(TypeVariableKind::Unconstrained))
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
                        Ok(self.checker.ctx.unit())
                    } else if path[0].eq_str("Never") {
                        Ok(self.checker.ctx.never())
                    } else {
                            // Check if this is a generic type parameter registered in the local cache
                            if path.len() == 1 {
                                if let Some(&ty) = self.checker.local_type_param_cache.get(&path[0])
                                {
                                    return Ok(ty);
                                }
                            }
                            Err(Diagnostic::error(format!("type '{}' not found", path[0]))
                                .with_span(*span))
                        }
                    }
                }
            Type::Generic(base, args, span) => {
                if let Type::Path(path, _) = base.as_ref() {
                    if path.len() == 1 {
                        if path[0].eq_str("Int") {
                            let width = args
                                .get(0)
                                .and_then(|arg| self.checker.extract_int_from_type(arg.ty()))
                                .unwrap_or(32);
                            return Ok(self.checker.ctx.int(width, true));
                        } else if path[0].eq_str("UInt") {
                            let width = args
                                .get(0)
                                .and_then(|arg| self.checker.extract_int_from_type(arg.ty()))
                                .unwrap_or(32);
                            return Ok(self.checker.ctx.int(width, false));
                        } else if path[0].eq_str("Float") {
                            let width = args
                                .get(0)
                                .and_then(|arg| self.checker.extract_int_from_type(arg.ty()))
                                .unwrap_or(64);
                            return Ok(self.checker.ctx.float(width));
                        } else if path[0].eq_str("Rational") {
                            let p = args.get(0).and_then(|arg| self.checker.extract_int_from_type(arg.ty()))
                                .ok_or_else(|| Diagnostic::error("Rational requires a compile-time constant integer bit count for the integer part").with_span(*span))?;
                            let q = args.get(1).and_then(|arg| self.checker.extract_int_from_type(arg.ty()))
                                .ok_or_else(|| Diagnostic::error("Rational requires a compile-time constant integer bit count for the fractional part").with_span(*span))?;
                            if p == 0 || p > 64 || q == 0 || q > 64 {
                                return Err(Diagnostic::error(
                                    "Rational bit counts must be 1..64",
                                )
                                .with_span(*span));
                            }
                            return Ok(self.checker.ctx.rational(p, q));
                        } else if path[0].eq_str("Ptr") {
                                let mut size = self.checker.ctx.usize();
                                let mut pointee = self.checker.ctx.error();
                                for arg in args {
                                    let ty = self.resolve_type(arg.ty())?;
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
                    arg_tys.push(self.resolve_type(arg.ty())?);
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
                inner: ty, mutable, ..
            } => {
                let inner = self.resolve_type(ty)?;
                Ok(self.checker.ctx.reference(inner, *mutable))
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
                if let Expr::Literal(Literal::Int(size_val), _) = size.as_ref() {
                    Ok(self.checker.ctx.array(inner, *size_val as u64))
                } else {
                    Err(
                        Diagnostic::error("array size must be a compile-time constant integer")
                            .with_span(*span),
                    )
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
                    1 => {
                        Ok(self
                            .checker
                            .ctx
                            .associated_type(candidates[0], name.clone(), _impl_ty))
                    }
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
            Type::DynTrait(traits, _) => {
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
                let (inv_hir, inv_ty) = self.infer_expr(invariant)?;
                if !self.checker.ctx.is_bool(inv_ty) {
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error("invariant must be boolean").with_span(*span));
                }
                Ok(self
                    .checker
                    .ctx
                    .exists(self.checker.ctx.fresh_param_index(), name.clone(), base_ty, *invariant.clone()))
            }
            Type::WhereShorthand {
                base,
                invariant,
                span,
            } => {
                // Desugar `type T = Base where value > 0` into `exists _where_N: Base invariant _where_N > 0`.
                let name = Symbol::intern(&format!("_where_{}", span.start));
                let mut inv = invariant.as_ref().clone();
                replace_ident_in_expr(&mut inv, Symbol::intern("value"), name);
                let base_ty = self.resolve_type(base)?;
                let (_, inv_ty) = self.infer_expr(&inv)?;
                if !self.checker.ctx.is_bool(inv_ty) {
                    self.checker
                        .diagnostics
                        .push(Diagnostic::error("invariant must be boolean").with_span(*span));
                }
                Ok(self.checker.ctx.exists(self.checker.ctx.fresh_param_index(), name, base_ty, inv))
            }
            Type::Literal(expr, _) => {
                let (_, ty) = self.infer_expr(expr)?;
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
            Type::Expr(expr, span) => {
                let (_, ty) = self.infer_expr(expr)?;
                Ok(ty)
            }
        }
    }

    /// Expand type aliases: if `ty` is an alias, resolve it to its body.
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
        span: Span,
    ) -> Result<TypeId, Diagnostic> {
        self.checker.binary_op_type(op, left, right, span)
    }

    /// Check a statement (delegates to TypeChecker).
    pub fn check_stmt(&mut self, stmt: &Stmt) -> Result<HirStmt, Diagnostic> {
        self.checker.check_stmt(stmt)
    }

    /// Check a block — actual implementation, not delegation.
    pub fn check_block(&mut self, stmts: &[Stmt]) -> Result<Vec<HirStmt>, Diagnostic> {
        let _scope = self.checker.enter_var_scope();
        let mut result = Vec::new();
        for stmt in stmts {
            result.push(self.checker.check_stmt(stmt)?);
        }
        // scope drops here — pops the frame (even on `?` early return)
        Ok(result)
    }

    /// Check a pattern against an expected type.
    pub fn check_pattern(
        &mut self,
        pattern: &Pattern,
        expected_ty: TypeId,
    ) -> Result<HirPattern, Diagnostic> {
        let hir = self.check_pattern_inner(pattern, expected_ty)?;
        // Automatically register all pattern-bound variables into local scope
        register_pattern_bindings(&mut self.checker.local_variable_types, &hir);
        Ok(hir)
    }

    /// Inner pattern check, without side-effect variable registration.
    fn check_pattern_inner(
        &mut self,
        pattern: &Pattern,
        expected_ty: TypeId,
    ) -> Result<HirPattern, Diagnostic> {
        if self.checker.ctx.is_infer_var(expected_ty) {
            match pattern {
                Pattern::Tuple(patterns, span) => {
                    let elem_tys: Vec<TypeId> = patterns
                        .iter()
                        .map(|_| self.new_infer_var(TypeVariableKind::Unconstrained))
                        .collect();
                    let tuple_ty = self.checker.ctx.tuple(elem_tys.clone());
                    self.unify_with(expected_ty, tuple_ty, *span, TypingContext::None)?;
                    let mut hir_pats = Vec::new();
                    for (pat, &ety) in patterns.iter().zip(elem_tys.iter()) {
                        hir_pats.push(self.check_pattern(pat, ety)?);
                    }
                    return Ok(HirPattern::Tuple(hir_pats, *span));
                }
                _ => {}
            }
        }
        match pattern {
            Pattern::Wildcard(span) => Ok(HirPattern::Wildcard(*span)),
            Pattern::Ident(name, span) => Ok(HirPattern::Ident(name.clone(), expected_ty, *span)),
            Pattern::Literal(expr, span) => {
                let (hir, ty) = self.infer_expr(expr)?;
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
                    hir_patterns.push(self.check_pattern(pat, elem_ty)?);
                }
                Ok(HirPattern::Tuple(hir_patterns, *span))
            }
            Pattern::Slice(before, slice, after, span) => {
                let elem_ty = if self.checker.ctx.is_infer_var(expected_ty) {
                    let elem = self.new_infer_var(TypeVariableKind::Any);
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
                    hir_before.push(self.check_pattern(pat, elem_ty)?);
                }
                let hir_slice = slice
                    .as_ref()
                    .map(|pat| {
                        let slice_ty = self.checker.ctx.slice(elem_ty);
                        let pat: HirPattern = self.check_pattern(pat, slice_ty)?;
                        Ok(Box::new(pat))
                    })
                    .transpose()?;
                let mut hir_after = Vec::new();
                for pat in after {
                    hir_after.push(self.check_pattern(pat, elem_ty)?);
                }
                Ok(HirPattern::Slice(hir_before, hir_slice, hir_after, *span))
            }
            Pattern::Error(span) => Ok(HirPattern::Error(*span)),
            Pattern::Struct { path, fields, span, .. } => {
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
                    .map(|_| self.new_infer_var(TypeVariableKind::Unconstrained))
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
                                Diagnostic::error(format!("field '{}' not found in struct", name))
                                    .with_code_str("E010")
                                    .with_span(*span)
                            })?;
                    let field_ty = self.checker.ctx.subst(field_def.ty, &subst);
                    hir_fields.push((name.clone(), Box::new(self.check_pattern(pat, field_ty)?)));
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
                    .map(|_| self.new_infer_var(TypeVariableKind::Unconstrained))
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
                let inner_ty = variant_def
                    .payload
                    .as_ref()
                    .map(|ty| {
                        // Same logic as EnumLit: substitute type params with concrete args.
                        if let Type::Path(p, _) = ty {
                            if p.len() == 1 {
                                if let Some((i, _)) = binding.params.iter().enumerate().find(|(_, tp)| tp.name == p[0]) {
                                    let gp = self.checker.ctx.generic_param(i, p[0].clone());
                                    return Ok(self.checker.ctx.subst(gp, &subst));
                                }
                            }
                        }
                        self.resolve_type(ty)
                    })
                    .unwrap_or(Ok(self.checker.ctx.error()))?;
                let inner_hir = inner
                    .as_ref()
                    .map(|inner| self.check_pattern(inner, inner_ty))
                    .transpose()?;
                Ok(HirPattern::Enum {
                    path: path.clone(),
                    variant: variant.clone(),
                    inner: inner_hir.map(Box::new),
                    span: *span,
                })
            }
            Pattern::Or(patterns, span) => {
                let mut hir_patterns = Vec::new();
                for pat in patterns {
                    // Use check_pattern_inner (not check_pattern) to avoid
                    // registering variable bindings from each sub-pattern.
                    // Or-patterns in Posita, like Rust, do not introduce
                    // variable bindings — each alternative binds separately,
                    // and no consistent type can be guaranteed across arms.
                    hir_patterns.push(self.check_pattern_inner(pat, expected_ty)?);
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
}

/// Walk a checked pattern and register every `HirPattern::Ident` binding
/// into `local_variable_types` so the body of if-let / while-let / for / match
/// can reference the bound variable.
pub(super) fn register_pattern_bindings(
    local_variable_types: &mut ScopedVarMap,
    pattern: &HirPattern,
) {
    match pattern {
        HirPattern::Ident(name, ty, _) => {
            local_variable_types.insert(name.clone(), *ty);
        }
        HirPattern::Tuple(patterns, _) => {
            for p in patterns {
                register_pattern_bindings(local_variable_types, p);
            }
        }
        HirPattern::Slice(before, rest, after, _) => {
            for p in before {
                register_pattern_bindings(local_variable_types, p);
            }
            if let Some(p) = rest {
                register_pattern_bindings(local_variable_types, p);
            }
            for p in after {
                register_pattern_bindings(local_variable_types, p);
            }
        }
        HirPattern::Struct { fields, .. } => {
            for (_, p) in fields {
                register_pattern_bindings(local_variable_types, p);
            }
        }
        HirPattern::Enum { inner: Some(p), .. } => {
            register_pattern_bindings(local_variable_types, p);
        }
        HirPattern::Or(patterns, _) => {
            // Or-patterns do not introduce variable bindings, consistent
            // with Rust's semantics.  Each alternative binds separately,
            // and no consistent type can be guaranteed across arms.
        }
        _ => {}
    }
}
