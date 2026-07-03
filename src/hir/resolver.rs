use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCollector, DiagnosticLevel};
use crate::hir::symbol::*;
use crate::hir::traits::{ImplCandidate, TraitEnv};
use rustc_hash::FxHashMap;

/// Represents the result of partially resolving a multi-segment path
/// (e.g. `Foo::bar::Baz` where only `Foo` is known at resolver time).
#[derive(Debug, Clone)]
pub enum PartialRes {
    /// Fully resolved — all path segments are known.
    Full(Res),
    /// Only the prefix `base` could be resolved; `remaining` segments
    /// must be resolved during type-checking.
    Unresolved { base: Res, remaining: usize },
    /// Resolution encountered an error.
    Err,
}

/// A resolved name/item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Res {
    Def(DefId),
    Type(DefId),
    Module(DefId),
    Primitive,
}

/// Pre-resolved name resolution results, populated by NameResolver and consumed by TypeChecker.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResolutionMap {
    pub variable_types: FxHashMap<String, TypeId>,
    pub type_def_ids: FxHashMap<String, DefId>,
    pub type_bindings: FxHashMap<DefId, TypeBinding>,
    /// Partial resolution of value paths (multi-segment), keyed by the first segment.
    pub value_resolutions: FxHashMap<String, PartialRes>,
    /// Partial resolution of type paths (multi-segment), keyed by the first segment.
    pub type_resolutions: FxHashMap<String, PartialRes>,
}
use crate::hir::types::*;
use rustc_hash::FxHashMap as HashMap;

pub struct NameResolver<'a> {
    ctx: &'a mut TypeContext,
    symbols: SymbolTable,
    trait_env: TraitEnv,
    diagnostics: DiagnosticCollector,
    current_scope: usize,
    current_function: Option<DefId>,
    current_type: Option<DefId>,
    import_map: Vec<ImportEntry>,
    local_crate_id: CrateId,
    /// Temporary mapping of type parameter names to GenericParam TypeIds
    /// used when resolving types inside an `impl<T>` block.
    current_impl_type_params: Option<HashMap<String, TypeId>>,
    /// Pre-resolved name resolutions for the type checker.
    resolution_map: ResolutionMap,
}

struct ImportEntry {
    path: Vec<String>,
    alias: Option<String>,
    items: Option<Vec<String>>,
    span: Span,
}

impl<'a> NameResolver<'a> {
    pub fn new(ctx: &'a mut TypeContext, local_crate_id: CrateId) -> Self {
        NameResolver {
            ctx,
            symbols: SymbolTable::new(local_crate_id),
            trait_env: TraitEnv::new(),
            diagnostics: DiagnosticCollector::new(),
            current_scope: 0,
            current_function: None,
            current_type: None,
            import_map: Vec::new(),
            local_crate_id,
            current_impl_type_params: None,
            resolution_map: ResolutionMap::default(),
        }
    }

    pub fn resolve_program(
        &mut self,
        program: &Program,
    ) -> Result<(SymbolTable, TraitEnv, DiagnosticCollector, ResolutionMap), DiagnosticCollector> {
        for item in &program.items {
            self.resolve_item(item);
        }

        if self.diagnostics.has_errors() {
            Err(std::mem::take(&mut self.diagnostics))
        } else {
            let symbols =
                std::mem::replace(&mut self.symbols, SymbolTable::new(self.local_crate_id));
            let trait_env = std::mem::replace(&mut self.trait_env, TraitEnv::new());
            let resolution_map = std::mem::take(&mut self.resolution_map);
            Ok((symbols, trait_env, std::mem::take(&mut self.diagnostics), resolution_map))
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
                contracts,
                ..
            } => {
                let def_id = self.allocate_def_id();

                // Register generic parameters BEFORE collecting the function signature,
                // so that resolve_type_expr can resolve T in `def foo<T>(x: T) -> T`.
                let mut param_map = HashMap::default();
                for (i, tp) in type_params.iter().enumerate() {
                    let ty_id = self.ctx.generic_param(i, tp.name.clone());
                    param_map.insert(tp.name.clone(), ty_id);
                }
                self.current_impl_type_params = Some(param_map);

                let sig = self.collect_function_signature(name, params, return_type, type_params);

                let binding = FunctionBinding {
                    def_id,
                    signature: sig,
                    is_comptime: *is_comptime,
                    is_async: *is_async,
                    is_pure: self.has_pure_attribute(attributes),
                    contracts: contracts.clone(),
                    attributes: attributes.clone(),
                };
                if let Err(diag) = self.symbols.insert_function(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }

                self.enter_scope();
                self.current_function = Some(def_id);

                for param in params {
                    let ty = if let Some(ty) = &param.ty {
                        self.resolve_type_expr(ty)
                    } else {
                        self.ctx.error()
                    };
                    let binding = VariableBinding {
                        ty,
                        mutable: false,
                        span: param.span,
                        def_id: self.allocate_def_id(),
                    };
                    self.resolution_map.variable_types.insert(param.name.clone(), ty);
                    if let Err(diag) =
                        self.symbols
                            .insert_variable(param.name.clone(), binding, param.span)
                    {
                        self.diagnostics.push(diag);
                    }
                }

                if let Some(body) = body {
                    for stmt in body {
                        self.resolve_stmt(stmt);
                    }
                }

                self.current_function = None;
                self.current_impl_type_params = None;
                self.exit_scope();
            }
            Stmt::TypeDef {
                span,
                attributes,
                name,
                params,
                definition,
                ..
            } => {
                let def_id = self.allocate_def_id();
                // Register type name in the resolution map for the type checker
                self.resolution_map.type_def_ids.insert(name.clone(), def_id);
                let type_params = params.clone();
                let kind = match definition {
                    TypeDefinition::Struct(_) => TypeKind::Struct,
                    TypeDefinition::Enum(_, _) => TypeKind::Enum,
                    TypeDefinition::Alias(_, _) => TypeKind::Alias,
                    TypeDefinition::TraitDef { .. } => TypeKind::Trait,
                    TypeDefinition::ImplBlock { .. } => TypeKind::Impl,
                    TypeDefinition::Constraint(_) => TypeKind::Constraint,
                };

                let mut fields = Vec::new();
                let mut variants = Vec::new();
                let mut alias_ast = None;
                let mut invariant = None;
                let mut default_value = None;
                let mut no_default = false;
                let mut missing_match = None;
                let exhaustive = attributes.iter().any(|a| a.name == "exhaustive");

                match definition {
                    TypeDefinition::Struct(fields_def) => {
                        fields = fields_def
                            .iter()
                            .map(|f| {
                                let field_ty = self.resolve_type_expr(&f.ty);
                                FieldBinding {
                                    name: f.name.clone(),
                                    ty: field_ty,
                                    default: f.default.clone(),
                                    span: f.span,
                                }
                            })
                            .collect();
                    }
                    TypeDefinition::Enum(variants_def, mm) => {
                        variants = variants_def.clone();
                        missing_match = mm.clone();
                    }
                    TypeDefinition::Alias(ty, mods) => {
                        alias_ast = Some(ty.clone());
                        for m in mods {
                            match m {
                                TypeModifier::Default(expr) => default_value = Some(expr.clone()),
                                TypeModifier::NoDefault => no_default = true,
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }

                let binding = TypeBinding {
                    def_id,
                    params: type_params,
                    kind,
                    span: *span,
                    alias_ast,
                    fields,
                    variants,
                    invariant,
                    default_value,
                    no_default,
                    crate_id: self.symbols.local_crate_id,
                    missing_match,
                    exhaustive,
                };
                if let Err(diag) = self.symbols.insert_type(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
                // Populate the resolution map for the type checker
                if let Some(b) = self.symbols.lookup_type(&name) {
                    self.resolution_map.type_bindings.insert(def_id, b.clone());
                }
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

                let binding = TraitBinding {
                    def_id,
                    methods: method_bindings,
                    associated_types: associated_types
                        .iter()
                        .map(|at| (at.name.clone(), at.default.clone()))
                        .collect(),
                    span: *span,
                    crate_id: self.symbols.local_crate_id,
                };
                if let Err(diag) = self.symbols.insert_trait(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
            }
            Stmt::ImplBlock {
                span,
                attributes,
                trait_path,
                for_type,
                methods,
                where_clause,
                type_params,
                ..
            } => {
                // Build mapping of type parameter names for this impl block
                let mut param_map = HashMap::default();
                for (i, tp) in type_params.iter().enumerate() {
                    let ty_id = self.ctx.generic_param(i, tp.name.clone());
                    param_map.insert(tp.name.clone(), ty_id);
                }
                self.current_impl_type_params = Some(param_map);

                let resolved_for = self.resolve_type_expr(for_type);
                let resolved_trait = trait_path
                    .as_ref()
                    .and_then(|path| self.resolve_trait_path(path));

                // Collect context types from where clause for Paterson/Coverage checking
                let mut context = Vec::new();
                if let Some(wc) = where_clause {
                    for pred in &wc.predicates {
                        let pred_ty = self.resolve_type_expr(&pred.ty);
                        context.push(pred_ty);
                    }
                }
                // Also add type params that have bounds to the context,
                // since `impl<T: Bar>` implicitly constrains T.
                for tp in type_params {
                    if !tp.bounds.is_empty() {
                        if let Some(ty_id) = self.current_impl_type_params.as_ref()
                            .and_then(|m| m.get(&tp.name))
                        {
                            context.push(*ty_id);
                        }
                    }
                }

                self.current_impl_type_params = None;

                self.enter_scope();
                let binding = ImplBinding {
                    def_id: self.allocate_def_id(),
                    methods: methods.clone(),
                    span: *span,
                };
                self.symbols.insert_impl(binding, *span);

                let has_auto_deref = attributes.iter().any(|a| a.name == "auto_deref");

                if let Some(trait_id) = resolved_trait {
                    let candidate = ImplCandidate {
                        trait_id,
                        for_type: resolved_for,
                        methods: methods.clone(),
                        assoc_tys: Vec::new(),
                        has_auto_deref,
                        context,
                        span: *span,
                    };
                    if let Err(err) = self.trait_env
                        .add_impl(candidate, &self.symbols, self.ctx, false)
                    {
                        // Convert OrphanError to a diagnostic for proper error reporting
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "impl for trait on type violates termination rules: {}",
                                err
                            ))
                            .with_span(*span)
                        );
                    }
                }

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
            Stmt::Edition(..) => {}
            Stmt::Constraint { name, bounds, span } => {
                let resolved_bounds: Vec<TypeId> =
                    bounds.iter().map(|b| self.resolve_type_expr(b)).collect();
                let binding = ConstraintBinding {
                    bounds: resolved_bounds,
                    span: *span,
                };
                if let Err(diag) = self.symbols.insert_constraint(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
            }
            Stmt::ExternFunction {
                abi,
                name,
                params,
                return_type,
                span,
                attributes,
                ..
            } => {
                let def_id = self.allocate_def_id();
                let sig = self.collect_function_signature(name, params, return_type, &[]);
                let binding = FunctionBinding {
                    def_id,
                    signature: sig,
                    is_comptime: false,
                    is_async: false,
                    is_pure: false,
                    contracts: Vec::new(),
                    attributes: attributes.clone(),
                };
                if let Err(diag) = self.symbols.insert_function(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
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

                    let binding = VariableBinding {
                        ty: ty_id,
                        mutable: *mutable,
                        span: *span,
                        def_id: self.allocate_def_id(),
                    };
                    // Pre-populate resolution map for the type checker
                    self.resolution_map.variable_types.insert(name.clone(), ty_id);
                    if let Err(diag) = self.symbols.insert_variable(name.clone(), binding, *span) {
                        self.diagnostics.push(diag);
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
                ..
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
                ..
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
                ..
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
                ..
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
                ..
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
            Stmt::Loop { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Leave { .. } => {}
            Stmt::Continue { .. } => {}
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    self.resolve_expr(value);
                }
            }
            Stmt::Assign { target, value, .. } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }
            Stmt::ComptimeBlock { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::ScopeCleanup { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Unsafe { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::GhostVariableDef { inner, .. } => {
                self.resolve_stmt(inner);
            }
            Stmt::Isolate { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
            }
            Stmt::Trigger { .. } => {}
            _ => {}
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) -> Option<TypeId> {
        match expr {
            Expr::Literal(lit, _span) => {
                let ty = self.literal_type(lit);
                Some(ty)
            }
            Expr::Ident(name, span) => {
                if let Some(binding) = self.symbols.lookup_variable(name, *span) {
                    Some(binding.ty)
                } else if let Some(func) = self.symbols.lookup_function(name) {
                    let sig = func.signature.clone();
                    let ty = self
                        .ctx
                        .function(sig.params.iter().map(|p| p.ty).collect(), sig.return_type);
                    Some(ty)
                } else if let Some(_ty_binding) = self.symbols.lookup_type(name) {
                    None
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined name: {}", name)).with_span(*span),
                    );
                    Some(self.ctx.error())
                }
            }
            Expr::TypeAnnotated { expr, ty, .. } => {
                let _ = self.resolve_type_expr(ty);
                self.resolve_expr(expr)
            }
            Expr::BinaryOp { left, right, .. } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
                None
            }
            Expr::UnaryOp { expr, .. } => {
                self.resolve_expr(expr);
                None
            }
            Expr::Call { callee, args, .. } => {
                self.resolve_expr(callee);
                for arg in args {
                    self.resolve_expr(arg);
                }
                None
            }
            Expr::Index { base, index, .. } => {
                self.resolve_expr(base);
                self.resolve_expr(index);
                None
            }
            Expr::FieldAccess { base, .. } => {
                self.resolve_expr(base);
                None
            }
            Expr::AttrAccess { base, .. } => {
                self.resolve_expr(base);
                None
            }
            Expr::Cast { expr, ty, .. } => {
                self.resolve_expr(expr);
                let _ = self.resolve_type_expr(ty);
                None
            }
            Expr::Range { start, end, .. } => {
                if let Some(start) = start {
                    self.resolve_expr(start);
                }
                if let Some(end) = end {
                    self.resolve_expr(end);
                }
                None
            }
            Expr::StructLit { path, fields, .. } => {
                let def_id = self.resolve_type_path(path);
                for (_, value) in fields {
                    self.resolve_expr(value);
                }
                if let Some(def_id) = def_id {
                    if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id) {
                        if binding.kind == TypeKind::Struct {
                            return Some(self.ctx.struct_ty(def_id, vec![]));
                        }
                    }
                }
                None
            }
            Expr::EnumLit {
                path,
                variant,
                payload,
                ..
            } => {
                if let Some(payload) = payload {
                    self.resolve_expr(payload);
                }
                if let Some(def_id) = self.resolve_type_path(path) {
                    if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id) {
                        if binding.kind == TypeKind::Enum {
                            return Some(self.ctx.enum_ty(def_id, vec![]));
                        }
                    }
                }
                None
            }
            Expr::Move(expr, ..) => {
                self.resolve_expr(expr);
                None
            }
            Expr::Tuple(exprs, ..) => {
                let mut elems = Vec::new();
                for e in exprs {
                    if let Some(ty) = self.resolve_expr(e) {
                        elems.push(ty);
                    } else {
                        elems.push(self.ctx.error());
                    }
                }
                Some(self.ctx.tuple(elems))
            }
            Expr::Array(exprs, ..) => {
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
                body,
                ..
            } => {
                self.enter_scope();
                for param in params {
                    let ty = if let Some(ty) = &param.ty {
                        self.resolve_type_expr(ty)
                    } else {
                        self.ctx.error()
                    };
                    let binding = VariableBinding {
                        ty,
                        mutable: false,
                        span: param.span,
                        def_id: self.allocate_def_id(),
                    };
                    self.resolution_map.variable_types.insert(param.name.clone(), ty);
                    if let Err(diag) =
                        self.symbols
                            .insert_variable(param.name.clone(), binding, param.span)
                    {
                        self.diagnostics.push(diag);
                    }
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
                let param_tys: Vec<TypeId> = params
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
            Expr::Try { expr, .. } => {
                self.resolve_expr(expr);
                None
            }
            Expr::UnsafeBlock { body, .. } => {
                self.enter_scope();
                for stmt in body {
                    self.resolve_stmt(stmt);
                }
                self.exit_scope();
                Some(self.ctx.unit())
            }
            Expr::Catch { expr, branches, .. } => {
                self.resolve_expr(expr);
                for branch in branches {
                    self.resolve_pattern(&branch.pattern);
                    if let Some(bind) = &branch.bind {
                        let binding = VariableBinding {
                            ty: self.ctx.error(),
                            mutable: false,
                            span: branch.span,
                            def_id: self.allocate_def_id(),
                        };
                        if let Err(diag) =
                            self.symbols
                                .insert_variable(bind.clone(), binding, branch.span)
                        {
                            self.diagnostics.push(diag);
                        }
                    }
                    self.enter_scope();
                    for stmt in &branch.body {
                        self.resolve_stmt(stmt);
                    }
                    self.exit_scope();
                }
                None
            }
            Expr::LeaveWith { expr, .. } => {
                self.resolve_expr(expr);
                Some(self.ctx.never())
            }
            Expr::Await { expr, .. } => {
                self.resolve_expr(expr);
                None
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
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
                None
            }
            Expr::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                ..
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
                None
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.resolve_expr(scrutinee);
                for arm in arms {
                    self.resolve_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.resolve_expr(guard);
                    }
                    self.resolve_expr(&arm.body);
                }
                None
            }
            Expr::Block(stmts, ..) => {
                self.enter_scope();
                let mut last_ty = None;
                for stmt in stmts {
                    if let Stmt::Expression(expr) = stmt {
                        last_ty = self.resolve_expr(expr);
                    } else {
                        self.resolve_stmt(stmt);
                    }
                }
                self.exit_scope();
                last_ty
            }
            Expr::PolyBox { expr, .. } => {
                self.resolve_expr(expr);
                Some(self.ctx.error())
            }
            Expr::PolyUnbox { expr, .. } => {
                self.resolve_expr(expr);
                Some(self.ctx.error())
            }
            Expr::Error(..) => Some(self.ctx.error()),
        }
    }

    fn resolve_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Wildcard(..) => {}
            Pattern::Ident(name, span) => {
                let binding = VariableBinding {
                    ty: self.ctx.error(),
                    mutable: false,
                    span: *span,
                    def_id: self.allocate_def_id(),
                };
                if let Err(diag) = self.symbols.insert_variable(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
            }
            Pattern::Literal(expr, ..) => {
                self.resolve_expr(expr);
            }
            Pattern::Tuple(patterns, ..) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Struct { fields, .. } => {
                for (_, p) in fields {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Enum { inner, .. } => {
                if let Some(inner) = inner {
                    self.resolve_pattern(inner);
                }
            }
            Pattern::Or(patterns, ..) => {
                for p in patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Slice(before, slice, after, ..) => {
                for p in before {
                    self.resolve_pattern(p);
                }
                if let Some(slice) = slice {
                    self.resolve_pattern(slice);
                }
                for p in after {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Error(..) => {}
        }
    }

    fn resolve_type_expr(&mut self, ty: &Type) -> TypeId {
        match ty {
            Type::Path(path, span) => {
                // Check if this name refers to an impl type parameter (e.g. `T` in `impl<T>`)
                if path.len() == 1 {
                    if let Some(ref param_map) = self.current_impl_type_params {
                        if let Some(&ty_id) = param_map.get(&path[0]) {
                            return ty_id;
                        }
                    }
                }
                if let Some(def_id) = self.resolve_type_path(path) {
                    let alias = self
                        .symbols
                        .lookup_type_by_def_id(def_id)
                        .and_then(|b| b.alias_ast.clone());
                    if let Some(alias) = alias {
                        self.resolve_type_expr(&alias)
                    } else if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id) {
                        match binding.kind {
                            TypeKind::Struct => self.ctx.struct_ty(def_id, vec![]),
                            TypeKind::Enum => self.ctx.enum_ty(def_id, vec![]),
                            _ => self.ctx.error(),
                        }
                    } else {
                        self.ctx.error()
                    }
                } else {
                    // Check for built-in types
                    let name = &path[0];
                    match name.as_str() {
                        "Bool" => self.ctx.bool(),
                        "Char" => self.ctx.char(),
                        "Byte" => self.ctx.byte(),
                        "USize" => self.ctx.usize(),
                        "Unit" => self.ctx.unit(),
                        "Never" => self.ctx.never(),
                        "Int" | "UInt" | "Float" | "Rational" => {
                            // These require type arguments; handled in Type::Generic
                            self.ctx.error()
                        }
                        _ => {
                            self.diagnostics.push(
                                Diagnostic::error(format!("undefined type: {}", path.join("::")))
                                    .with_span(*span),
                            );
                            self.ctx.error()
                        }
                    }
                }
            }
            Type::Generic(base, args, span) => {
                // Handle generic built-in types (Int, UInt, Float) by matching base path
                if let Type::Path(path, _) = base.as_ref() {
                    if path.len() == 1 {
                        match path[0].as_str() {
                            "Int" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(32);
                                return self.ctx.int(bits, true);
                            }
                            "UInt" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(32);
                                return self.ctx.int(bits, false);
                            }
                            "Float" => {
                                let bits = self.extract_int_from_type(&args[0]).unwrap_or(64);
                                return self.ctx.float(bits);
                            }
                            "Rational" => {
                                let p = self.extract_int_from_type(&args[0]).unwrap_or(16);
                                let q = self.extract_int_from_type(&args[1]).unwrap_or(16);
                                return self.ctx.rational(p, q);
                            }
                            _ => {}
                        }
                    }
                }
                let base_ty = self.resolve_type_expr(base);
                if let Some(def_id) = self.ctx.get_def_id_for_type(base_ty) {
                    let binding = self.symbols.lookup_type_by_def_id(def_id).cloned();
                    if let Some(binding) = binding {
                        let arg_tys: Vec<TypeId> =
                            args.iter().map(|a| self.resolve_type_expr(a)).collect();
                        match binding.kind {
                            TypeKind::Struct => self.ctx.struct_ty(def_id, arg_tys),
                            TypeKind::Enum => self.ctx.enum_ty(def_id, arg_tys),
                            _ => {
                                self.diagnostics.push(
                                    Diagnostic::error("generic type arguments on non-generic type")
                                        .with_span(*span),
                                );
                                self.ctx.error()
                            }
                        }
                    } else {
                        self.diagnostics
                            .push(Diagnostic::error("type definition not found").with_span(*span));
                        self.ctx.error()
                    }
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("expected a path type for generic base").with_span(*span),
                    );
                    self.ctx.error()
                }
            }
            Type::Reference(ty, mutable, ..) => {
                let inner = self.resolve_type_expr(ty);
                self.ctx.reference(inner, *mutable)
            }
            Type::Pointer(ty, ..) => {
                let inner = self.resolve_type_expr(ty);
                self.ctx.pointer(inner)
            }
            Type::Slice(ty, ..) => {
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
            Type::Tuple(tys, ..) => {
                let elems: Vec<TypeId> = tys.iter().map(|t| self.resolve_type_expr(t)).collect();
                self.ctx.tuple(elems)
            }
            Type::Function { params, ret, .. } => {
                let param_tys = params.iter().map(|p| self.resolve_type_expr(p)).collect();
                let ret_ty = self.resolve_type_expr(ret);
                self.ctx.function(param_tys, ret_ty)
            }
            Type::Projection(base, name, span) => {
                let _base_ty = self.resolve_type_expr(base);
                self.ctx.error()
            }
            Type::DynTrait(traits, ..) => {
                let trait_ids: Vec<DefId> = traits
                    .iter()
                    .filter_map(|t| {
                        if let Type::Path(path, _) = t {
                            self.resolve_type_path(path)
                        } else {
                            None
                        }
                    })
                    .collect();
                self.ctx.dyn_trait(trait_ids)
            }
            Type::Exists {
                name,
                base,
                invariant,
                ..
            } => {
                let base_ty = self.resolve_type_expr(base);
                self.ctx
                    .exists(name.clone(), base_ty, invariant.as_ref().clone())
            }
            Type::Literal(expr, ..) => self.resolve_expr(expr).unwrap_or(self.ctx.error()),
            Type::Never(..) => self.ctx.never(),
            Type::Union(tys, ..) => self.ctx.error(),
            Type::Error(..) => self.ctx.error(),
        }
    }

    fn resolve_type_path(&mut self, path: &[String]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        self.symbols.lookup_type_by_path(path)
    }

    fn resolve_trait_path(&mut self, path: &[String]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        self.symbols.lookup_trait_by_path(path)
    }

    fn extract_int_from_type(&self, ty: &Type) -> Option<u8> {
        match ty {
            Type::Literal(expr, _) => {
                if let Expr::Literal(Literal::Int(val), _) = expr.as_ref() {
                    Some(*val as u8)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn enter_scope(&mut self) {
        self.current_scope = self.symbols.push_scope();
    }

    fn exit_scope(&mut self) {
        self.symbols.pop_scope();
    }

    fn allocate_def_id(&mut self) -> DefId {
        self.symbols.allocate_def_id()
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

    fn literal_type(&mut self, lit: &Literal) -> TypeId {
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
        &mut self,
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
            type_params: type_params.to_vec(),
            where_clause: None,
        }
    }

    fn collect_trait_method_signature(&mut self, method: &TraitMethod) -> FunctionSignature {
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

    pub fn into_symbols(self) -> SymbolTable {
        self.symbols
    }

    pub fn diagnostics(&self) -> &DiagnosticCollector {
        &self.diagnostics
    }
}
// Use crate-level module instead
