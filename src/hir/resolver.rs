use crate::ast::visit::replace_ident_in_expr;
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
    /// Current module path for registering full-qualified type paths.
    module_path: Vec<String>,
    /// Layout aliases defined with `layout Name { ... }`.
    layout_aliases: HashMap<String, Vec<Attribute>>,
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
            module_path: Vec::new(),
            layout_aliases: HashMap::default(),
        }
    }

    pub fn resolve_program(
        &mut self,
        program: &Program,
    ) -> Result<(SymbolTable, TraitEnv, DiagnosticCollector, ResolutionMap), DiagnosticCollector>
    {
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
            Ok((
                symbols,
                trait_env,
                std::mem::take(&mut self.diagnostics),
                resolution_map,
            ))
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
                    is_ieee_contracts: self.has_ieee_contracts_attribute(attributes),
                    hints: self.extract_hints(attributes),
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
                self.resolution_map
                    .type_def_ids
                    .insert(name.clone(), def_id);
                let type_params = params.clone();
                let kind = match definition {
                    TypeDefinition::Struct(_, _) => TypeKind::Struct,
                    TypeDefinition::Enum(_, _, _) => TypeKind::Enum,
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
                    TypeDefinition::Struct(fields_def, _) => {
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
                    TypeDefinition::Enum(variants_def, mm, _) => {
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

                let mut c_layout = false;
                let mut transparent = false;
                let mut packed = false;
                let mut endian = None;
                let mut bit_order = None;
                let mut align = None;
                let mut pad = None;
                let mut expanded_attrs = attributes.clone();
                for attr in attributes {
                    if attr.name == "layout" {
                        for arg in &attr.args {
                            if let crate::ast::Expr::Ident(name, _) = arg {
                                if name == "C" {
                                    c_layout = true;
                                } else if let Some(alias_attrs) = self.layout_aliases.get(name.as_str()) {
                                    for alias_attr in alias_attrs {
                                        if !expanded_attrs.iter().any(|a| a.name == alias_attr.name) {
                                            expanded_attrs.push(alias_attr.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if attr.name == "transparent" { transparent = true; }
                    if attr.name == "packed" { packed = true; }
                    if attr.name == "endian" {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Ident(name, _)) if name == "little" => {
                                endian = Some(crate::ast::Endianness::Little);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) if name == "big" => {
                                endian = Some(crate::ast::Endianness::Big);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("`@endian` expects `little` or `big`, got `{}`", name))
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@endian(little)` or `@endian(big)`"),
                                );
                            }
                            Some(_) => {
                                self.diagnostics.push(
                                    Diagnostic::error("`@endian` requires an identifier argument (`little` or `big`)")
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@endian(little)` or `@endian(big)`"),
                                );
                            }
                            None => {
                                self.diagnostics.push(
                                    Diagnostic::error("`@endian` requires an argument")
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@endian(little)` or `@endian(big)`"),
                                );
                            }
                        }
                    }
                    if attr.name == "bit_order" {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Ident(name, _)) if name == "lsb_to_msb" => {
                                bit_order = Some(crate::ast::BitOrder::LsbToMsb);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) if name == "msb_to_lsb" => {
                                bit_order = Some(crate::ast::BitOrder::MsbToLsb);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("`@bit_order` expects `lsb_to_msb` or `msb_to_lsb`, got `{}`", name))
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@bit_order(lsb_to_msb)` or `@bit_order(msb_to_lsb)`"),
                                );
                            }
                            Some(_) => {
                                self.diagnostics.push(
                                    Diagnostic::error("`@bit_order` requires an identifier argument")
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@bit_order(lsb_to_msb)` or `@bit_order(msb_to_lsb)`"),
                                );
                            }
                            None => {
                                self.diagnostics.push(
                                    Diagnostic::error("`@bit_order` requires an argument")
                                        .with_code_str("E061")
                                        .with_span(attr.span)
                                        .with_suggestion("write `@bit_order(lsb_to_msb)` or `@bit_order(msb_to_lsb)`"),
                                );
                            }
                        }
                    }
                    if attr.name == "align" || attr.name == "pad" {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Literal(crate::ast::Literal::Int(n), _)) => {
                                if attr.name == "align" { align = Some(*n as u64); }
                                if attr.name == "pad" { pad = Some(*n as u64); }
                            }
                            Some(_) => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("`@{}` requires an integer argument", attr.name))
                                        .with_code_str("E060")
                                        .with_span(attr.span)
                                        .with_suggestion(format!("write `@{}(N)` where N is a power of two", attr.name)),
                                );
                            }
                            None => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("`@{}` requires an integer argument", attr.name))
                                        .with_code_str("E060")
                                        .with_span(attr.span)
                                        .with_suggestion(format!("write `@{}(128)`", attr.name)),
                                );
                            }
                        }
                    }
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
                    c_layout,
                    transparent,
                    expanded_layout_attrs: expanded_attrs,
                    packed,
                    endian,
                    bit_order,
                    align,
                    pad,
                };
                if let Err(diag) = self.symbols.insert_type(name.clone(), binding, *span) {
                    self.diagnostics.push(diag);
                }
                // Register the fully-qualified path for multi-segment resolution.
                {
                    let mut full_path = self.module_path.clone();
                    full_path.push(name.clone());
                    let full = full_path.join("::");
                    self.symbols.register_full_path(full, def_id);
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
                associated_types,
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
                        if let Some(ty_id) = self
                            .current_impl_type_params
                            .as_ref()
                            .and_then(|m| m.get(&tp.name))
                        {
                            context.push(*ty_id);
                        }
                    }
                }

                self.enter_scope();
                let binding = ImplBinding {
                    def_id: self.allocate_def_id(),
                    methods: methods.clone(),
                    span: *span,
                };
                self.symbols.insert_impl(binding, *span);

                // Keep current_impl_type_params active so method body resolution
                // can resolve type parameters like `T` in `impl<T> Foo for Bar { ... }`.
                // It is cleared after the impl block is fully processed.

                let has_auto_deref = attributes.iter().any(|a| a.name == "auto_deref");

                // Pre-resolve method param types using the impl's type param mapping,
                // so generic params like `T` are properly substituted in lookup_method.
                let mut resolved_methods = Vec::new();
                for method in methods {
                    let mut param_tys = Vec::with_capacity(method.params.len());
                    for p in &method.params {
                        if let Some(ref param_ty) = p.ty {
                            // Substitute `Self` with the concrete for_type AST before resolving,
                            // since resolve_type_expr cannot resolve `Self` on its own.
                            let resolved_ty = self.resolve_self_in_type(param_ty, for_type);
                            param_tys.push(self.resolve_type_expr(&resolved_ty));
                        } else {
                            param_tys.push(self.ctx.error());
                        }
                    }
                    let resolved_ret = self.resolve_self_in_type(&method.return_type, for_type);
                    let ret_ty = self.resolve_type_expr(&resolved_ret);
                    resolved_methods.push(crate::hir::traits::MethodInfo {
                        name: method.name.clone(),
                        param_tys,
                        ret_ty,
                        span: method.span,
                        has_auto_deref,
                    });
                }

                // Resolve associated types from the impl block.
                let mut assoc_tys = Vec::new();
                for at in associated_types {
                    if let Some(ref default) = at.default {
                        let resolved = self.resolve_type_expr(default);
                        assoc_tys.push((at.name.clone(), resolved));
                    }
                }

                if let Some(trait_id) = resolved_trait {
                    let candidate = ImplCandidate {
                        trait_id,
                        for_type: resolved_for,
                        methods: methods.clone(),
                        resolved_methods,
                        assoc_tys,
                        has_auto_deref,
                        context,
                        span: *span,
                    };
                    if let Err(err) =
                        self.trait_env
                            .add_impl(candidate, &self.symbols, self.ctx, false)
                    {
                        // Convert OrphanError to a diagnostic for proper error reporting
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "impl for trait on type violates termination rules: {}",
                                err
                            ))
                            .with_span(*span),
                        );
                    }
                }

                // Clear impl type params so they don't leak into subsequent statements.
                self.current_impl_type_params = None;

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
                // Resolve the import path against the symbol table and register
                // the imported symbols in the current scope.
                let resolved = self.resolve_import_path(path, items, alias, *span);
                if let Err(diag) = resolved {
                    self.diagnostics.push(diag);
                }
            }
            Stmt::Edition(version, span) => {
                match crate::hir::types::Edition::from_str(version) {
                    Some(ed) => self.ctx.set_edition(ed),
                    None => {
                        self.diagnostics.push(
                            Diagnostic::error(format!("unknown edition `{}`", version))
                                .with_code_str("E070")
                                .with_span(*span)
                                .with_suggestion("use a valid edition: `\"2024\"` or `\"2026\"`"),
                        );
                    }
                }
            }
            Stmt::LayoutDef { name, attributes, .. } => {
                // Register a layout alias so that @layout(AliasName) can be expanded.
                if self.layout_aliases.contains_key(name) {
                    self.diagnostics.push(
                        Diagnostic::error(format!("duplicate layout alias `{}`", name))
                            .with_span(Span::new(0, 0)),
                    );
                } else {
                    self.layout_aliases.insert(name.clone(), attributes.clone());
                }
            }
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
                    is_ieee_contracts: self.has_ieee_contracts_attribute(attributes),
                    hints: self.extract_hints(attributes),
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
                type_captures,
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
                    if let Err(diag) = self.symbols.insert_variable(name.clone(), binding, *span) {
                        self.diagnostics.push(diag);
                    }
                }

                // `set auto<T> = expr` — register each capture name as a type
                // in the resolution map so that comptime code can reference it.
                for cap in type_captures {
                    // Placeholder entry: the checker overwrites this with the inferred
                    // type after evaluating the expression.  We seed it with error()
                    // so that if the checker doesn't override it (e.g. due to a bug),
                    // the capture name resolves to Error at the use site, triggering
                    // a downstream compile error instead of silent miscompilation.
                    let _placeholder = self.ctx.error();
                    self.resolution_map
                        .type_def_ids
                        .insert(cap.name.clone(), DefId(usize::MAX));
                    // The actual type binding will be updated by the checker
                    // after inferring the expression's type.
                    let binding = TypeBinding {
                        def_id: DefId(usize::MAX),
                        params: vec![],
                        kind: TypeKind::Alias,
                        span: *span,
                        alias_ast: None,
                        fields: vec![],
                        variants: vec![],
                        invariant: None,
                        default_value: None,
                        no_default: true,
                        crate_id: self.local_crate_id,
                        missing_match: None,
                        exhaustive: false,
                        c_layout: false,
                        transparent: false,
                        expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
                    };
                    self.symbols
                        .insert_type(cap.name.clone(), binding, *span)
                        .ok();
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
                } else if name == "result" {
                    // `result` in `ensures` clauses is resolved by the checker.
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
            Expr::Quantified { range, body, .. } => {
                self.resolve_expr(range);
                self.resolve_expr(body);
                Some(self.ctx.error())
            }
            Expr::PolyBox { expr, .. } => {
                self.resolve_expr(expr);
                Some(self.ctx.error())
            }
            Expr::PolyUnbox { expr, .. } => {
                self.resolve_expr(expr);
                Some(self.ctx.error())
            }
            Expr::Old(expr, _) => {
                self.resolve_expr(expr);
                None
            }
            Expr::Path(path, _) => {
                self.diagnostics.push(
                    Diagnostic::error(format!("unresolved path: {}", path.join("::")))
                        .with_span(Span::new(0, 0)),
                );
                Some(self.ctx.error())
            }
            Expr::Error(..) => Some(self.ctx.error()),
            Expr::TypeInfo(ty, _) => {
                // @typeInfo!(Type) — resolve the type argument, return Unit.
                self.resolve_type_expr(ty);
                Some(self.ctx.unit())
            }
            Expr::Task { body, .. } => {
                for s in body { self.resolve_stmt(s); }
                Some(self.ctx.unit())
            }
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
                                let bits = self.extract_int_from_type(args[0].ty()).unwrap_or(32);
                                return self.ctx.int(bits, true);
                            }
                            "UInt" => {
                                let bits = self.extract_int_from_type(args[0].ty()).unwrap_or(32);
                                return self.ctx.int(bits, false);
                            }
                            "Float" => {
                                let bits = self.extract_int_from_type(args[0].ty()).unwrap_or(64);
                                return self.ctx.float(bits);
                            }
                            "Rational" => {
                                let p = self.extract_int_from_type(args[0].ty()).unwrap_or(16);
                                let q = self.extract_int_from_type(args[1].ty()).unwrap_or(16);
                                return self.ctx.rational(p, q);
                            }
                            "Ptr" => {
                                let size = args
                                    .get(0)
                                    .map(|a| self.resolve_type_expr(a.ty()))
                                    .unwrap_or(self.ctx.usize());
                                let pointee = args
                                    .get(1)
                                    .map(|a| self.resolve_type_expr(a.ty()))
                                    .unwrap_or(self.ctx.error());
                                return self.ctx.ptr(size, pointee);
                            }
                            "USize" => {
                                return self.ctx.usize();
                            }
                            _ => {}
                        }
                    }
                }
                let base_ty = self.resolve_type_expr(base);
                if let Some(def_id) = self.ctx.get_def_id_for_type(base_ty) {
                    let binding = self.symbols.lookup_type_by_def_id(def_id).cloned();
                    if let Some(binding) = binding {
                        let arg_tys: Vec<TypeId> = args
                            .iter()
                            .map(|a| self.resolve_type_expr(a.ty()))
                            .collect();
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
            Type::Reference {
                inner: ty, mutable, ..
            } => {
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
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name: name,
                span,
            } => {
                let _impl_ty = self.resolve_type_expr(impl_type);
                let _trait_ty = self.resolve_type_expr(trait_path);
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
                    .exists(self.ctx.fresh_param_index(), name.clone(), base_ty, invariant.as_ref().clone())
            }
            Type::WhereShorthand {
                base,
                invariant,
                span,
            } => {
                // Desugar `type T = Base where value > 0` into `exists _where_N: Base invariant _where_N > 0`.
                let name = format!("_where_{}", span.start);
                let mut inv = invariant.as_ref().clone();
                replace_ident_in_expr(&mut inv, "value", &name);
                let base_ty = self.resolve_type_expr(base);
                self.ctx.exists(self.ctx.fresh_param_index(), name, base_ty, inv)
            }
            Type::Literal(expr, ..) => self.resolve_expr(expr).unwrap_or(self.ctx.error()),
            Type::Never(..) => self.ctx.never(),
            Type::Union(tys, ..) => self.ctx.error(),
            Type::Error(..) => self.ctx.error(),
            Type::Expr(expr, ..) => self.resolve_expr(expr).unwrap_or(self.ctx.error()),
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
                    if *val > 64 {
                        return None;
                    }
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

    fn has_ieee_contracts_attribute(&self, attributes: &[Attribute]) -> bool {
        attributes.iter().any(|attr| attr.name == "ieee_contracts")
    }

    fn extract_hints(&self, attributes: &[Attribute]) -> Vec<Expr> {
        attributes
            .iter()
            .filter(|attr| attr.name == "hint")
            .flat_map(|attr| attr.args.clone())
            .collect()
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

    /// Recursively substitute `Self` in an AST type with `self_ty`.
    /// Needed for resolving method signatures in impl blocks, where
    /// `&self` desugars to `Self` which resolve_type_expr cannot handle.
    fn resolve_self_in_type(&self, ty: &Type, self_ty: &Type) -> Type {
        match ty {
            Type::Path(p, s) if p.len() == 1 && (p[0] == "Self" || p[0] == "self") => {
                self_ty.clone()
            }
            Type::Reference {
                inner,
                mutable,
                span: s,
                ..
            } => Type::Reference {
                inner: Box::new(self.resolve_self_in_type(inner, self_ty)),
                mutable: *mutable,
                lifetime: None,
                span: *s,
            },
            Type::Pointer(inner, s) => {
                Type::Pointer(Box::new(self.resolve_self_in_type(inner, self_ty)), *s)
            }
            Type::Generic(base, args, span) => {
                let new_base = self.resolve_self_in_type(base, self_ty);
                let new_args: Vec<GenericArg> = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Positional(t) => {
                            GenericArg::Positional(self.resolve_self_in_type(t, self_ty))
                        }
                        GenericArg::Named(n, t) => {
                            GenericArg::Named(n.clone(), self.resolve_self_in_type(t, self_ty))
                        }
                    })
                    .collect();
                Type::Generic(Box::new(new_base), new_args, *span)
            }
            Type::Tuple(tys, span) => Type::Tuple(
                tys.iter()
                    .map(|t| self.resolve_self_in_type(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Slice(inner, span) => {
                Type::Slice(Box::new(self.resolve_self_in_type(inner, self_ty)), *span)
            }
            Type::Array(inner, size, span) => Type::Array(
                Box::new(self.resolve_self_in_type(inner, self_ty)),
                size.clone(),
                *span,
            ),
            Type::DynTrait(traits, span) => Type::DynTrait(
                traits
                    .iter()
                    .map(|t| self.resolve_self_in_type(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Function { params, ret, span } => Type::Function {
                params: params
                    .iter()
                    .map(|p| self.resolve_self_in_type(p, self_ty))
                    .collect(),
                ret: Box::new(self.resolve_self_in_type(ret, self_ty)),
                span: *span,
            },
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name,
                span,
            } => Type::Projection {
                impl_type: Box::new(self.resolve_self_in_type(impl_type, self_ty)),
                trait_path: Box::new(self.resolve_self_in_type(trait_path, self_ty)),
                assoc_name: assoc_name.clone(),
                span: *span,
            },
            other => other.clone(),
        }
    }

    /// Resolve an import path against the symbol table and register aliases
    /// in the current scope.  Supports:
    ///   `import path::to::item;`           → alias = last segment
    ///   `import path::to::item as alias;`  → alias = explicit name
    ///   `from path::to import { a, b };`   → each item by explicit name
    fn resolve_import_path(
        &mut self,
        path: &[String],
        items: &Option<Vec<String>>,
        alias: &Option<String>,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let import_name = alias.as_ref().or_else(|| path.last()).cloned();

        // First try to resolve as a type.
        if let Some(def_id) = self.symbols.lookup_type_by_path(path) {
            if let Some(name) = &import_name {
                self.resolution_map
                    .type_def_ids
                    .insert(name.clone(), def_id);
                if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id).cloned() {
                    self.symbols.insert_type(name.clone(), binding, span).ok();
                }
                // Register the import's original full path for re-export resolution.
                let full_path = path.join("::");
                self.symbols.register_full_path(full_path, def_id);
            }
            // `from path import { items }`
            if let Some(item_list) = items {
                for item in item_list {
                    let item_path = [item.clone()];
                    if let Some(item_def_id) = self.symbols.lookup_type_by_path(&item_path) {
                        self.resolution_map
                            .type_def_ids
                            .insert(item.clone(), item_def_id);
                        if let Some(binding) =
                            self.symbols.lookup_type_by_def_id(item_def_id).cloned()
                        {
                            self.symbols.insert_type(item.clone(), binding, span).ok();
                        }
                        // Register the full path: path::to::item
                        let mut full_item_path = path.to_vec();
                        full_item_path.push(item.clone());
                        self.symbols
                            .register_full_path(full_item_path.join("::"), item_def_id);
                    }
                }
            }
            return Ok(());
        }

        // Try as a trait — supports multi-segment paths.
        if let Some(trait_def_id) = self.symbols.lookup_trait_by_path(path) {
            if let Some(trait_binding) = self.symbols.lookup_trait_by_def_id(trait_def_id).cloned() {
                if let Some(name) = &import_name {
                    self.symbols
                        .insert_trait(name.clone(), trait_binding, span)
                        .ok();
                }
                // `from path import { items }` — also import traits
                if let Some(item_list) = items {
                    for item in item_list {
                        let item_path = [item.clone()];
                        if let Some(item_def_id) = self.symbols.lookup_trait_by_path(&item_path) {
                            if let Some(item_binding) =
                                self.symbols.lookup_trait_by_def_id(item_def_id).cloned()
                            {
                                self.symbols
                                    .insert_trait(item.clone(), item_binding, span)
                                    .ok();
                            }
                        }
                    }
                }
                return Ok(());
            }
        }

        // Try as a function — single-segment only for now;
        // multi-segment function imports require module hierarchy support.
        if path.len() == 1 {
            if let Some(func_binding) = self.symbols.lookup_function(&path[0]).cloned() {
                if let Some(name) = &import_name {
                    if let Err(diag) =
                        self.symbols
                            .insert_function(name.clone(), func_binding, span)
                    {
                        self.diagnostics.push(diag);
                    }
                }
                return Ok(());
            }
        }

        Err(
            Diagnostic::error(format!("cannot resolve import `{}`", path.join("::"),))
                .with_span(span),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// Parse and resolve a Posita source, returning the resolver's symbol table.
    fn resolve_source(source: &str) -> Result<(SymbolTable, TraitEnv, ResolutionMap, TypeContext), Vec<String>> {
        let mut ctx = TypeContext::new();
        let mut parser = Parser::new(source);
        let program = parser
            .parse_program()
            .map_err(|diags| diags.into_iter().map(|d| d.message).collect::<Vec<_>>())?;
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, trait_env, _diags, resolution_map) = resolver
            .resolve_program(&program)
            .map_err(|diags| {
                diags
                    .into_inner()
                    .into_iter()
                    .map(|d| d.message)
                    .collect::<Vec<_>>()
            })?;
        Ok((symbols, trait_env, resolution_map, ctx))
    }

    #[test]
    fn test_resolve_empty_program() {
        let result = resolve_source("");
        assert!(result.is_ok(), "empty program: {:?}", result.err());
    }

    #[test]
    fn test_resolve_function_def() {
        let result = resolve_source("def main() -> Int<32> { return 0; }");
        assert!(result.is_ok(), "function def: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let func = symbols.lookup_function("main");
        assert!(func.is_some(), "main should be registered");
    }

    #[test]
    fn test_resolve_type_def_struct() {
        let result = resolve_source("type Point = struct { x: Int<32>, y: Int<32> }");
        assert!(result.is_ok(), "struct type: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type("Point");
        assert!(binding.is_some(), "Point should be registered");
        if let Some(b) = binding {
            assert_eq!(b.fields.len(), 2, "Point should have 2 fields");
        }
    }

    #[test]
    fn test_resolve_type_def_enum() {
        let result = resolve_source("type Option<T> = enum { None, Some(T) }");
        assert!(result.is_ok(), "enum type: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type("Option");
        assert!(binding.is_some(), "Option should be registered");
        if let Some(b) = binding {
            assert_eq!(b.params.len(), 1, "Option should have 1 type param");
            assert_eq!(b.variants.len(), 2, "Option should have 2 variants");
        }
    }

    #[test]
    fn test_resolve_type_alias() {
        let result = resolve_source("type MyInt = Int<32>");
        assert!(result.is_ok(), "type alias: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type("MyInt");
        assert!(binding.is_some(), "MyInt should be registered");
        assert!(binding.unwrap().alias_ast.is_some(), "MyInt should have an alias AST");
    }

    #[test]
    fn test_resolve_layout_alias() {
        let result = resolve_source(
            "layout Mmio {
                 packed,
                 little_endian;
             }",
        );
        assert!(result.is_ok(), "layout alias: {:?}", result.err());
    }

    #[test]
    fn test_resolve_transparent_attr() {
        let result = resolve_source(
            "@transparent
             type Wrapper = struct { inner: Int<32> }",
        );
        assert!(result.is_ok(), "transparent: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type("Wrapper");
        assert!(binding.unwrap().transparent, "Wrapper should be transparent");
    }

    #[test]
    fn test_resolve_layout_c_attr() {
        let result = resolve_source(
            "@layout(C)
             type CStruct = struct { x: Int<32> }",
        );
        assert!(result.is_ok(), "layout(C): {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type("CStruct");
        assert!(binding.unwrap().c_layout, "CStruct should have c_layout");
    }

    #[test]
    fn test_resolve_generic_function() {
        let result = resolve_source("def id<T>(x: T) -> T { return x; }");
        assert!(result.is_ok(), "generic function: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let func = symbols.lookup_function("id");
        assert!(func.is_some(), "id should be registered");
        assert!(!func.unwrap().signature.type_params.is_empty(), "id should have type params");
    }

    #[test]
    fn test_resolve_trait_and_impl() {
        let result = resolve_source(
            "trait Show { }
             impl Show for Int<32> { }",
        );
        assert!(result.is_ok(), "trait + impl: {:?}", result.err());
    }

    #[test]
    fn test_resolve_duplicate_function() {
        let result = resolve_source(
            "def f() -> Int<32> { return 0; }
             def f() -> Int<32> { return 1; }",
        );
        assert!(result.is_err(), "duplicate function should error");
    }
}
