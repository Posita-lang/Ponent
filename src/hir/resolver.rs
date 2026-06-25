use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCollector, DiagnosticLevel};
use crate::hir::symbol::*;
use crate::hir::types::*;

pub struct NameResolver<'a> {
    ctx: &'a mut TypeContext,
    symbols: &'a mut SymbolTable,
    diagnostics: DiagnosticCollector,
    current_scope: usize,
    current_function: Option<DefId>,
    current_type: Option<DefId>,
    import_map: Vec<ImportEntry>,
    next_def_id: usize,
}

struct ImportEntry {
    path: Vec<String>,
    alias: Option<String>,
    items: Option<Vec<String>>,
    span: Span,
}

impl<'a> NameResolver<'a> {
    pub fn new(ctx: &'a mut TypeContext, symbols: &'a mut SymbolTable) -> Self {
        NameResolver {
            ctx,
            symbols,
            diagnostics: DiagnosticCollector::new(),
            current_scope: 0,
            current_function: None,
            current_type: None,
            import_map: Vec::new(),
            next_def_id: 0,
        }
    }

    pub fn resolve_program(&mut self, program: &Program) -> Result<(), DiagnosticCollector> {
        self.enter_scope();
        for item in &program.items {
            self.resolve_item(item);
        }
        self.exit_scope();

        if self.diagnostics.has_errors() {
            Err(std::mem::take(&mut self.diagnostics))
        } else {
            Ok(())
        }
    }

    fn resolve_item(&mut self, item: &Stmt) {
        match item {
            Stmt::FunctionDef {
                span,
                attributes,
                name,
                params,
                return_type,
                body,
                type_params,
                where_clause,
                is_comptime,
                is_async,
                ..
            } => {
                let def_id = self.allocate_def_id();
                let sig = self.collect_function_signature(name, params, return_type, type_params);

                self.symbols.insert_function(
                    name.clone(),
                    FunctionBinding {
                        def_id,
                        signature: sig,
                        is_comptime: *is_comptime,
                        is_async: *is_async,
                        is_pure: self.has_pure_attribute(attributes),
                        contracts: Vec::new(),
                        attributes: attributes.clone(),
                    },
                    *span,
                );

                self.enter_scope();
                self.current_function = Some(def_id);

                for param in params {
                    let ty = self.resolve_type_expr(&param.ty);
                    self.symbols.insert_variable(
                        param.name.clone(),
                        VariableBinding {
                            ty,
                            mutable: false,
                            span: param.span,
                            def_id: self.allocate_def_id(),
                        },
                        param.span,
                    );
                }

                if let Some(body) = body {
                    for stmt in body {
                        self.resolve_stmt(stmt);
                    }
                }

                self.current_function = None;
                self.exit_scope();
            }
            Stmt::TypeDef {
                span,
                name,
                params,
                definition,
                ..
            } => {
                let def_id = self.allocate_def_id();
                let kind = match definition {
                    TypeDefinition::Struct(fields) => TypeKind::Struct,
                    TypeDefinition::Enum(variants, _) => TypeKind::Enum,
                    TypeDefinition::Alias(_, _) => TypeKind::Alias,
                    TypeDefinition::TraitDef { .. } => TypeKind::Trait,
                    TypeDefinition::ImplBlock { .. } => TypeKind::Impl,
                    TypeDefinition::Constraint(_) => TypeKind::Constraint,
                };

                let type_params = params
                    .iter()
                    .map(|tp| {
                        let bounds = tp
                            .bounds
                            .iter()
                            .map(|b| self.resolve_type_expr(b))
                            .collect();
                        TypeParam {
                            name: tp.name.clone(),
                            bounds,
                            span: tp.span,
                        }
                    })
                    .collect();

                self.symbols.insert_type(
                    name.clone(),
                    TypeBinding {
                        def_id,
                        params: type_params,
                        kind,
                        span: *span,
                    },
                    *span,
                );

                self.current_type = Some(def_id);
                self.enter_scope();

                match definition {
                    TypeDefinition::Struct(fields) => {
                        for field in fields {
                            let ty = self.resolve_type_expr(&field.ty);
                            self.symbols.insert_variable(
                                field.name.clone(),
                                VariableBinding {
                                    ty,
                                    mutable: false,
                                    span: field.span,
                                    def_id: self.allocate_def_id(),
                                },
                                field.span,
                            );
                        }
                    }
                    TypeDefinition::Enum(variants, _) => {
                        for variant in variants {
                            if let Some(payload) = &variant.payload {
                                let ty = self.resolve_type_expr(payload);
                                self.symbols.insert_variable(
                                    variant.name.clone(),
                                    VariableBinding {
                                        ty,
                                        mutable: false,
                                        span: variant.span,
                                        def_id: self.allocate_def_id(),
                                    },
                                    variant.span,
                                );
                            }
                        }
                    }
                    TypeDefinition::Alias(ty, _) => {
                        let resolved = self.resolve_type_expr(ty);
                        self.symbols.insert_alias(name.clone(), resolved, *span);
                    }
                    _ => {}
                }

                self.exit_scope();
                self.current_type = None;
            }
            Stmt::TraitDef {
                span,
                name,
                methods,
                associated_types,
                ..
            } => {
                let def_id = self.allocate_def_id();
                let mut method_bindings = Vec::new();
                for method in methods {
                    let sig = self.collect_trait_method_signature(method);
                    method_bindings.push((method.name.clone(), sig));
                }

                self.symbols.insert_trait(
                    name.clone(),
                    TraitBinding {
                        def_id,
                        methods: method_bindings,
                        associated_types: associated_types
                            .iter()
                            .map(|at| (at.name.clone(), at.default.clone()))
                            .collect(),
                        span: *span,
                    },
                    *span,
                );
            }
            Stmt::ImplBlock {
                span,
                attributes,
                trait_path,
                for_type,
                methods,
                ..
            } => {
                let resolved_for = self.resolve_type_expr(for_type);
                let resolved_trait = trait_path.as_ref().map(|path| self.resolve_path(path));

                self.enter_scope();
                self.symbols.insert_impl(
                    resolved_trait,
                    resolved_for,
                    ImplBinding {
                        def_id: self.allocate_def_id(),
                        methods: methods.clone(),
                        span: *span,
                    },
                    *span,
                );
                self.exit_scope();
            }
            Stmt::Import {
                path,
                items,
                alias,
                span,
            } => {
                self.import_map.push(ImportEntry {
                    path: path.clone(),
                    alias: alias.clone(),
                    items: items.clone(),
                    span: *span,
                });
            }
            Stmt::Constraint { name, bounds, span } => {
                let resolved_bounds = bounds.iter().map(|b| self.resolve_type_expr(b)).collect();
                self.symbols.insert_constraint(
                    name.clone(),
                    ConstraintBinding {
                        bounds: resolved_bounds,
                        span: *span,
                    },
                    *span,
                );
            }
            Stmt::ExternFunction {
                abi,
                name,
                params,
                return_type,
                span,
                attributes,
            } => {
                let def_id = self.allocate_def_id();
                let sig = self.collect_function_signature(name, params, return_type, &[]);

                self.symbols.insert_function(
                    name.clone(),
                    FunctionBinding {
                        def_id,
                        signature: sig,
                        is_comptime: false,
                        is_async: false,
                        is_pure: false,
                        contracts: Vec::new(),
                        attributes: attributes.clone(),
                    },
                    *span,
                );
            }
            _ => {
                if let Some(stmt_span) = self.get_stmt_span(item) {
                    self.diagnostics.push(
                        Diagnostic::error("unexpected statement at top level").with_span(stmt_span),
                    );
                }
            }
        }
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VariableDef {
                kind,
                mutable,
                name,
                pattern,
                ty,
                value,
                else_branch,
                span,
                ..
            } => {
                if let Some(name) = name {
                    let ty_id = if let Some(ty) = ty {
                        self.resolve_type_expr(ty)
                    } else {
                        if let Some(value) = value {
                            self.resolve_expr(value).unwrap_or_else(|| self.ctx.error())
                        } else {
                            self.diagnostics.push(
                                Diagnostic::error("cannot infer type without initializer")
                                    .with_span(*span),
                            );
                            self.ctx.error()
                        }
                    };

                    self.symbols.insert_variable(
                        name.clone(),
                        VariableBinding {
                            ty: ty_id,
                            mutable: *mutable,
                            span: *span,
                            def_id: self.allocate_def_id(),
                        },
                        *span,
                    );

                    if let Some(value) = value {
                        self.resolve_expr(value);
                    }
                }

                if let Some(pattern) = pattern {
                    self.resolve_pattern(pattern);
                }

                if let Some(else_branch) = else_branch {
                    self.enter_scope();
                    for stmt in else_branch {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
            }
            Stmt::Expression(expr) => {
                self.resolve_expr(expr);
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                self.resolve_expr(cond);
                self.enter_scope();
                for stmt in then_branch {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();

                if let Some(else_branch) = else_branch {
                    self.enter_scope();
                    for stmt in else_branch {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
            }
            Stmt::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span,
            } => {
                self.resolve_expr(scrutinee);
                self.resolve_pattern(pattern);
                self.enter_scope();
                for stmt in then_branch {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();

                if let Some(else_branch) = else_branch {
                    self.enter_scope();
                    for stmt in else_branch {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
            }
            Stmt::While {
                cond,
                body,
                invariant,
                decreases,
                span,
            } => {
                self.resolve_expr(cond);
                if let Some(inv) = invariant {
                    self.resolve_expr(inv);
                }
                if let Some(dec) = decreases {
                    self.resolve_expr(dec);
                }
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::WhileLet {
                pattern,
                scrutinee,
                body,
                invariant,
                decreases,
                span,
            } => {
                self.resolve_expr(scrutinee);
                self.resolve_pattern(pattern);
                if let Some(inv) = invariant {
                    self.resolve_expr(inv);
                }
                if let Some(dec) = decreases {
                    self.resolve_expr(dec);
                }
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::For {
                pattern,
                iterable,
                body,
                invariant,
                decreases,
                span,
            } => {
                self.resolve_expr(iterable);
                self.resolve_pattern(pattern);
                if let Some(inv) = invariant {
                    self.resolve_expr(inv);
                }
                if let Some(dec) = decreases {
                    self.resolve_expr(dec);
                }
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Loop { body, span } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Leave { label, span } => {}
            Stmt::Continue { label, span } => {}
            Stmt::Return { value, span } => {
                if let Some(value) = value {
                    self.resolve_expr(value);
                }
            }
            Stmt::Assign {
                target,
                op,
                value,
                span,
            } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }
            Stmt::ComptimeBlock { body, span } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::ScopeCleanup {
                name,
                body,
                propagates,
                overrides,
                span,
            } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Unsafe { body, span } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::GhostVariableDef { inner, span } => {
                self.resolve_stmt(inner);
            }
            Stmt::Isolate { body, span } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Trigger { name, span } => {}
            Stmt::Error(span) => {}
            _ => {}
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Option<TypeId> {
        match expr {
            Expr::Literal(lit, span) => {
                let ty = self.literal_type(lit);
                Some(ty)
            }
            Expr::Ident(name, span) => {
                let binding = self.symbols.lookup_variable(name, *span);
                if let Some(binding) = binding {
                    Some(binding.ty)
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined variable: {}", name)).with_span(*span),
                    );
                    Some(self.ctx.error())
                }
            }
            Expr::TypeAnnotated { expr, ty, span } => {
                let resolved_ty = self.resolve_type_expr(ty);
                self.resolve_expr(expr);
                Some(resolved_ty)
            }
            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
                Some(self.ctx.int(32, true))
            }
            Expr::UnaryOp { op, expr, span } => {
                self.resolve_expr(expr);
                Some(self.ctx.int(32, true))
            }
            Expr::Call {
                callee,
                args,
                comptime,
                span,
            } => {
                let callee_ty = self.resolve_expr(callee);
                for arg in args {
                    self.resolve_expr(arg);
                }
                Some(self.ctx.int(32, true))
            }
            Expr::Index { base, index, span } => {
                self.resolve_expr(base);
                self.resolve_expr(index);
                Some(self.ctx.int(32, true))
            }
            Expr::FieldAccess { base, field, span } => {
                self.resolve_expr(base);
                Some(self.ctx.int(32, true))
            }
            Expr::AttrAccess { base, attr, span } => {
                self.resolve_expr(base);
                Some(self.ctx.int(32, true))
            }
            Expr::Cast {
                expr,
                ty,
                safe,
                rounding,
                span,
            } => {
                self.resolve_expr(expr);
                let resolved_ty = self.resolve_type_expr(ty);
                Some(resolved_ty)
            }
            Expr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                if let Some(start) = start {
                    self.resolve_expr(start);
                }
                if let Some(end) = end {
                    self.resolve_expr(end);
                }
                Some(
                    self.ctx
                        .tuple(vec![self.ctx.int(32, true), self.ctx.int(32, true)]),
                )
            }
            Expr::StructLit { path, fields, span } => {
                for (_, value) in fields {
                    self.resolve_expr(value);
                }
                Some(self.ctx.int(32, true))
            }
            Expr::EnumLit {
                path,
                variant,
                payload,
                span,
            } => {
                if let Some(payload) = payload {
                    self.resolve_expr(payload);
                }
                Some(self.ctx.int(32, true))
            }
            Expr::Move(expr, span) => {
                self.resolve_expr(expr);
                Some(self.ctx.int(32, true))
            }
            Expr::Tuple(exprs, span) => {
                let mut elems = Vec::new();
                for e in exprs {
                    if let Some(ty) = self.resolve_expr(e) {
                        elems.push(ty);
                    }
                }
                Some(self.ctx.tuple(elems))
            }
            Expr::Array(exprs, span) => {
                let mut elem_ty = None;
                for e in exprs {
                    if let Some(ty) = self.resolve_expr(e) {
                        if elem_ty.is_none() {
                            elem_ty = Some(ty);
                        }
                    }
                }
                Some(
                    self.ctx
                        .array(elem_ty.unwrap_or(self.ctx.error()), exprs.len() as u64),
                )
            }
            Expr::Closure {
                params,
                return_type,
                captures,
                body,
                span,
            } => {
                self.enter_scope();
                for param in params {
                    let ty = if let Some(ty) = &param.ty {
                        self.resolve_type_expr(ty)
                    } else {
                        self.ctx.error()
                    };
                    self.symbols.insert_variable(
                        param.name.clone(),
                        VariableBinding {
                            ty,
                            mutable: false,
                            span: param.span,
                            def_id: self.allocate_def_id(),
                        },
                        param.span,
                    );
                }
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();

                let ret_ty = if let Some(ret) = return_type {
                    self.resolve_type_expr(ret)
                } else {
                    self.ctx.unit()
                };
                let param_tys = params
                    .iter()
                    .map(|p| {
                        if let Some(ty) = &p.ty {
                            self.resolve_type_expr(ty)
                        } else {
                            self.ctx.error()
                        }
                    })
                    .collect();
                Some(self.ctx.function(param_tys, ret_ty))
            }
            Expr::Try { expr, span } => {
                self.resolve_expr(expr);
                Some(self.ctx.int(32, true))
            }
            Expr::UnsafeBlock { body, span } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
                Some(self.ctx.unit())
            }
            Expr::Catch {
                expr,
                branches,
                span,
            } => {
                self.resolve_expr(expr);
                for branch in branches {
                    self.resolve_pattern(&branch.pattern);
                    if let Some(bind) = &branch.bind {
                        self.symbols.insert_variable(
                            bind.clone(),
                            VariableBinding {
                                ty: self.ctx.error(),
                                mutable: false,
                                span: branch.span,
                                def_id: self.allocate_def_id(),
                            },
                            branch.span,
                        );
                    }
                    self.enter_scope();
                    for stmt in &branch.body {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
                Some(self.ctx.int(32, true))
            }
            Expr::LeaveWith { expr, span } => {
                self.resolve_expr(expr);
                Some(self.ctx.never())
            }
            Expr::Await { expr, span } => {
                self.resolve_expr(expr);
                Some(self.ctx.int(32, true))
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                is_expression,
                span,
            } => {
                self.resolve_expr(cond);
                self.enter_scope();
                for stmt in then_branch {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();

                if let Some(else_branch) = else_branch {
                    self.enter_scope();
                    for stmt in else_branch {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
                Some(self.ctx.unit())
            }
            Expr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span,
            } => {
                self.resolve_expr(scrutinee);
                self.resolve_pattern(pattern);
                self.enter_scope();
                for stmt in then_branch {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();

                if let Some(else_branch) = else_branch {
                    self.enter_scope();
                    for stmt in else_branch {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
                Some(self.ctx.unit())
            }
            Expr::Match {
                scrutinee,
                arms,
                span,
            } => {
                self.resolve_expr(scrutinee);
                for arm in arms {
                    self.resolve_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.resolve_expr(guard);
                    }
                    self.resolve_expr(&arm.body);
                }
                Some(self.ctx.unit())
            }
            Expr::Block(stmts, span) => {
                self.enter_scope();
                let mut last_ty = self.ctx.unit();
                for stmt in stmts {
                    if let Stmt::Expression(expr) = stmt {
                        if let Some(ty) = self.resolve_expr(expr) {
                            last_ty = ty;
                        }
                    } else {
                        self.resolve_stmt(stmt);
                        last_ty = self.ctx.unit();
                    }
                }
                self.exit_scope();
                Some(last_ty)
            }
            Expr::Error(span) => Some(self.ctx.error()),
        }
    }

    fn resolve_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Wildcard(span) => {}
            Pattern::Ident(name, span) => {
                self.symbols.insert_variable(
                    name.clone(),
                    VariableBinding {
                        ty: self.ctx.error(),
                        mutable: false,
                        span: *span,
                        def_id: self.allocate_def_id(),
                    },
                    *span,
                );
            }
            Pattern::Literal(expr, span) => {
                self.resolve_expr(expr);
            }
            Pattern::Tuple(patterns, span) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Struct { path, fields, span } => {
                for (_, p) in fields {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Enum {
                path,
                variant,
                inner,
                span,
            } => {
                if let Some(inner) = inner {
                    self.resolve_pattern(inner);
                }
            }
            Pattern::Or(patterns, span) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Error(span) => {}
        }
    }

    fn resolve_type_expr(&mut self, ty: &Type) -> TypeId {
        match ty {
            Type::Path(path, span) => {
                let def_id = self.resolve_path(path);
                if let Some(def_id) = def_id {
                    self.ctx.int(32, true)
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined type: {}", path.join("::")))
                            .with_span(*span),
                    );
                    self.ctx.error()
                }
            }
            Type::Generic(base, args, span) => {
                let base_ty = self.resolve_type_expr(base);
                let arg_tys: Vec<TypeId> = args.iter().map(|a| self.resolve_type_expr(a)).collect();
                self.ctx.int(32, true)
            }
            Type::Reference(ty, mutable, span) => {
                let inner = self.resolve_type_expr(ty);
                self.ctx.reference(inner, *mutable)
            }
            Type::Pointer(ty, span) => {
                let inner = self.resolve_type_expr(ty);
                self.ctx.pointer(inner)
            }
            Type::Slice(ty, span) => {
                let inner = self.resolve_type_expr(ty);
                self.ctx.slice(inner)
            }
            Type::Array(ty, size, span) => {
                let inner = self.resolve_type_expr(ty);
                if let Expr::Literal(Literal::Int(size_val), _) = size.as_ref() {
                    self.ctx.array(inner, *size_val as u64)
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("array size must be a compile-time constant integer")
                            .with_span(*span),
                    );
                    self.ctx.error()
                }
            }
            Type::Tuple(tys, span) => {
                let elems: Vec<TypeId> = tys.iter().map(|t| self.resolve_type_expr(t)).collect();
                self.ctx.tuple(elems)
            }
            Type::Function { params, ret, span } => {
                let param_tys = params.iter().map(|p| self.resolve_type_expr(p)).collect();
                let ret_ty = self.resolve_type_expr(ret);
                self.ctx.function(param_tys, ret_ty)
            }
            Type::Projection(base, name, span) => {
                let base_ty = self.resolve_type_expr(base);
                self.ctx.int(32, true)
            }
            Type::DynTrait(traits, span) => {
                let trait_ids = traits
                    .iter()
                    .map(|t| {
                        if let Type::Path(path, _) = t {
                            self.resolve_path(path).unwrap_or(DefId(0))
                        } else {
                            DefId(0)
                        }
                    })
                    .collect();
                self.ctx.dyn_trait(trait_ids)
            }
            Type::Exists {
                name,
                base,
                invariant,
                span,
            } => {
                let base_ty = self.resolve_type_expr(base);
                self.ctx
                    .exists(name.clone(), base_ty, invariant.as_ref().clone())
            }
            Type::Literal(expr, span) => {
                self.resolve_expr(expr);
                self.ctx.int(32, true)
            }
            Type::Never(span) => self.ctx.never(),
            Type::Union(tys, span) => self.ctx.int(32, true),
            Type::Error(span) => self.ctx.error(),
        }
    }

    fn resolve_path(&mut self, path: &[String]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        let name = &path[0];
        if let Some(def_id) = self.symbols.lookup_type(name) {
            if path.len() == 1 {
                return Some(def_id);
            }
            let mut current = def_id;
            for i in 1..path.len() {
                if let Some(child) = self.symbols.lookup_child(current, &path[i]) {
                    current = child;
                } else {
                    return None;
                }
            }
            return Some(current);
        }
        None
    }

    fn enter_scope(&mut self) {
        self.current_scope = self.symbols.push_scope();
    }

    fn exit_scope(&mut self) {
        self.symbols.pop_scope();
        self.current_scope = self.symbols.current_scope();
    }

    fn allocate_def_id(&mut self) -> DefId {
        let id = DefId(self.next_def_id);
        self.next_def_id += 1;
        id
    }

    fn get_stmt_span(&self, stmt: &Stmt) -> Option<Span> {
        match stmt {
            Stmt::VariableDef { span, .. } => Some(*span),
            Stmt::FunctionDef { span, .. } => Some(*span),
            Stmt::TypeDef { span, .. } => Some(*span),
            Stmt::TraitDef { span, .. } => Some(*span),
            Stmt::Import { span, .. } => Some(*span),
            Stmt::ExternFunction { span, .. } => Some(*span),
            Stmt::Constraint { span, .. } => Some(*span),
            Stmt::Edition(_, span) => Some(*span),
            _ => None,
        }
    }

    fn has_pure_attribute(&self, attributes: &[Attribute]) -> bool {
        attributes.iter().any(|attr| attr.name == "pure")
    }

    fn literal_type(&self, lit: &Literal) -> TypeId {
        match lit {
            Literal::Int(_) => self.ctx.int(32, true),
            Literal::Float(_) => self.ctx.float(64),
            Literal::Char(_) => self.ctx.char(),
            Literal::String(_) => self.ctx.slice(self.ctx.byte()),
            Literal::ByteString(_) => self.ctx.slice(self.ctx.byte()),
            Literal::Bool(_) => self.ctx.bool(),
        }
    }

    fn collect_function_signature(
        &self,
        name: &str,
        params: &[Param],
        return_type: &Type,
        type_params: &[TypeParam],
    ) -> FunctionSignature {
        FunctionSignature {
            params: params
                .iter()
                .map(|p| {
                    let ty =
                        p.ty.as_ref()
                            .map_or(self.ctx.error(), |t| self.resolve_type_expr(t));
                    Parameter {
                        name: p.name.clone(),
                        ty,
                        span: p.span,
                        default: p.default.clone(),
                    }
                })
                .collect(),
            return_type: self.resolve_type_expr(return_type),
            type_params: type_params
                .iter()
                .map(|tp| TypeParam {
                    name: tp.name.clone(),
                    bounds: tp
                        .bounds
                        .iter()
                        .map(|b| self.resolve_type_expr(b))
                        .collect(),
                    span: tp.span,
                })
                .collect(),
            where_clause: None,
        }
    }

    fn collect_trait_method_signature(&self, method: &TraitMethod) -> FunctionSignature {
        FunctionSignature {
            params: method
                .params
                .iter()
                .map(|p| {
                    let ty =
                        p.ty.as_ref()
                            .map_or(self.ctx.error(), |t| self.resolve_type_expr(t));
                    Parameter {
                        name: p.name.clone(),
                        ty,
                        span: p.span,
                        default: p.default.clone(),
                    }
                })
                .collect(),
            return_type: self.resolve_type_expr(&method.return_type),
            type_params: Vec::new(),
            where_clause: None,
        }
    }
}

impl<'a> From<NameResolver<'a>> for DiagnosticCollector {
    fn from(resolver: NameResolver<'a>) -> Self {
        resolver.diagnostics
    }
}
