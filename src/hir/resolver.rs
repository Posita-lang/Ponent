use crate::ast::visit::replace_ident_in_expr;
use crate::ast::*;
use crate::diagnostics::{DiagCtxt, Diagnostic, DiagnosticLevel};
use crate::hir::builtins;
use crate::hir::symbol::*;
use crate::hir::traits::{ImplCandidate, TraitEnv};
use crate::symbol::Symbol;
use regex_syntax;

/// Recursively search an AST type for a reference to one of `params`
/// (the enum's declared type parameters).  Used to reject GADT `when`
/// constraints whose RIGHT-HAND side references another type parameter
/// of the same enum (`when T == U and U == T` would register a mutual
/// refinement cycle that `resolve_binding_tail` chases until
/// `MAX_CHAIN_DEPTH`).  `exists` variables are NOT in `params`, so
/// `when T == X` (witness stays opaque) is unaffected.
fn find_param_ref_in_type<'input>(
    ty: &Type<'input>,
    params: &[TypeParam<'input>],
) -> Option<Symbol> {
    match ty {
        Type::Path(path, _) if path.len() == 1 => params
            .iter()
            .find(|tp| tp.name == path[0])
            .map(|tp| tp.name),
        Type::Generic(base, args, _) => find_param_ref_in_type(base, params).or_else(|| {
            args.iter().find_map(|arg| match arg {
                GenericArg::Positional(t) | GenericArg::Named(_, t) => {
                    find_param_ref_in_type(t, params)
                }
                GenericArg::Const(_) => None,
            })
        }),
        Type::Tuple(elems, _) => elems.iter().find_map(|e| find_param_ref_in_type(e, params)),
        Type::Slice(elem, _) => find_param_ref_in_type(elem, params),
        Type::Array(elem, _, _) => find_param_ref_in_type(elem, params),
        Type::Reference { inner, .. } => find_param_ref_in_type(inner, params),
        Type::Pointer(inner, _) => find_param_ref_in_type(inner, params),
        Type::Function {
            params: ps, ret, ..
        } => ps
            .iter()
            .find_map(|e| find_param_ref_in_type(e, params))
            .or_else(|| find_param_ref_in_type(ret, params)),
        Type::Projection {
            impl_type,
            trait_path,
            ..
        } => find_param_ref_in_type(impl_type, params)
            .or_else(|| find_param_ref_in_type(trait_path, params)),
        Type::DynTrait(bounds, _) => bounds
            .iter()
            .find_map(|e| find_param_ref_in_type(e, params)),
        // Defensive: `Type::Expr` arises from const-generic / array-size
        // positions (`[T; N]`'s `N`).  Unreachable as a `when` RHS via
        // `parse_type`, but a recursive walk keeps E064 complete if a
        // future syntax route reaches here.
        Type::Expr(expr, _) => find_param_ref_in_expr(expr, params),
        // Existential binding: search the base type for same-enum parameter
        // references (e.g. `exists X: T` where `T` is a same-enum param).
        Type::Exists { base, .. } => find_param_ref_in_type(base, params),
        _ => None,
    }
}

/// A GADT `when` RHS is "provably concrete" for the E065 contradiction check
/// when it mentions no `exists` witness (at ANY depth — an opaque witness may
/// equal anything) and every path it references resolves to a non-alias
/// binding (two different alias names may still alias the same type).  Any
/// other shape (forward references, unregistered names, projection/exists
/// forms) is not provable at the resolver stage and must NOT trigger a hard
/// error — a false positive (rejecting valid code) is worse than a missed
/// diagnostic.
fn rhs_is_provably_concrete<'input>(
    ct: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> bool {
    match ct {
        crate::ast::Type::Path(p, _) => {
            // A bare single-segment path naming an `exists` witness is opaque.
            if p.len() == 1 && exists_params.iter().any(|ep| ep == &p[0]) {
                return false;
            }
            // The builtin primitives are concrete by construction (they are
            // not symbol-table bindings, so the lookup below would fail).
            if p.len() == 1
                && (p[0].eq_str("Int")
                    || p[0].eq_str("UInt")
                    || p[0].eq_str("Float")
                    || p[0].eq_str("Bool")
                    || p[0].eq_str("Char")
                    || p[0].eq_str("Byte")
                    || p[0].eq_str("USize"))
            {
                return true;
            }
            // The path must resolve to a concrete (non-alias) binding.
            match symbols
                .lookup_type_by_path(p)
                .and_then(|d| symbols.lookup_type_by_def_id(d))
            {
                Some(b) => !matches!(b.kind, TypeKind::Alias),
                None => false,
            }
        }
        crate::ast::Type::Generic(base, args, _) => {
            rhs_is_provably_concrete(base, exists_params, symbols)
                && args.iter().all(|a| match a {
                    crate::ast::GenericArg::Positional(t) | crate::ast::GenericArg::Named(_, t) => {
                        rhs_is_provably_concrete(t, exists_params, symbols)
                    }
                    crate::ast::GenericArg::Const(ac) => {
                        crate::hir::type_eq::const_expr_is_ground(&ac.value)
                    }
                })
        }
        crate::ast::Type::Reference { inner, .. } => {
            rhs_is_provably_concrete(inner, exists_params, symbols)
        }
        crate::ast::Type::Pointer(t, _) => rhs_is_provably_concrete(t, exists_params, symbols),
        crate::ast::Type::Slice(t, _) => rhs_is_provably_concrete(t, exists_params, symbols),
        crate::ast::Type::Array(t, size, _) => {
            rhs_is_provably_concrete(t, exists_params, symbols)
                && crate::hir::type_eq::const_expr_is_ground(size)
        }
        crate::ast::Type::Tuple(es, _) => es
            .iter()
            .all(|e| rhs_is_provably_concrete(e, exists_params, symbols)),
        crate::ast::Type::Function { params, ret, .. } => {
            params
                .iter()
                .all(|p| rhs_is_provably_concrete(p, exists_params, symbols))
                && rhs_is_provably_concrete(ret, exists_params, symbols)
        }
        crate::ast::Type::Union(es, _) => es
            .iter()
            .all(|e| rhs_is_provably_concrete(e, exists_params, symbols)),
        crate::ast::Type::Never(_)
        | crate::ast::Type::Literal(..)
        | crate::ast::Type::Regex(..)
        | crate::ast::Type::Error(_) => true,
        // Projection, DynTrait, Exists, WhereShorthand, Expr<'input>: not provably
        // concrete at the resolver stage — be conservative (skip the check).
        _ => false,
    }
}

/// Expression-side counterpart of `find_param_ref_in_type`: recursively
/// search an AST expression for an `Ident` whose name is one of `params`.
/// Used for `Type::Expr` (const-generic / array-size) positions.
fn find_param_ref_in_expr<'input>(
    expr: &Expr<'input>,
    params: &[TypeParam<'input>],
) -> Option<Symbol> {
    match expr {
        Expr::Ident(name, _) => params.iter().find(|tp| tp.name == *name).map(|tp| tp.name),
        Expr::Literal(_, _) => None,
        Expr::TypeAnnotated { expr: inner, .. }
        | Expr::UnaryOp { expr: inner, .. }
        | Expr::Cast { expr: inner, .. } => find_param_ref_in_expr(inner, params),
        Expr::BinaryOp { left, right, .. } => {
            find_param_ref_in_expr(left, params).or_else(|| find_param_ref_in_expr(right, params))
        }
        Expr::Call { callee, args, .. } => find_param_ref_in_expr(callee, params)
            .or_else(|| args.iter().find_map(|a| find_param_ref_in_expr(a, params))),
        Expr::Index { base, index, .. } => {
            find_param_ref_in_expr(base, params).or_else(|| find_param_ref_in_expr(index, params))
        }
        Expr::FieldAccess { base, .. } | Expr::AttrAccess { base, .. } => {
            find_param_ref_in_expr(base, params)
        }
        Expr::Range { start, end, .. } => start
            .as_ref()
            .and_then(|s| find_param_ref_in_expr(s, params))
            .or_else(|| end.as_ref().and_then(|e| find_param_ref_in_expr(e, params))),
        Expr::Tuple(elems, _) | Expr::Array(elems, _) => {
            elems.iter().find_map(|e| find_param_ref_in_expr(e, params))
        }
        Expr::Block(stmts, _) => stmts.iter().find_map(|s| match s {
            Stmt::Expression(e) => find_param_ref_in_expr(e, params),
            // Defensive: `Type::Expr` cannot currently appear as a `when`
            // RHS (`parse_type` does not route through `parse_expr`), so
            // this branch is unreachable today — but if a future grammar
            // change makes it reachable, every statement's expression
            // positions must be traversed or E064's param-reference check
            // silently under-reports.  The arms below keep the traversal
            // complete for the statement kinds that can carry expressions.
            Stmt::VariableDef { value: Some(v), .. } => find_param_ref_in_expr(v, params),
            Stmt::If { cond, .. } => find_param_ref_in_expr(cond, params),
            _ => None,
        }),
        _ => None,
    }
}
use rustc_hash::FxHashMap;
use std::cell::Cell;
use std::rc::Rc;

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
pub(crate) struct ResolutionMap<'input> {
    pub type_def_ids: FxHashMap<Symbol, DefId>,
    pub type_bindings: FxHashMap<DefId, TypeBinding<'input>>,
    /// Partial resolution of value paths (multi-segment), keyed by the first segment.
    pub value_resolutions: FxHashMap<Symbol, PartialRes>,
    /// Partial resolution of type paths (multi-segment), keyed by the first segment.
    pub type_resolutions: FxHashMap<Symbol, PartialRes>,
    /// Whether a top-level `main` function was found during resolution.
    pub has_main: bool,
}
use crate::hir::types::*;
use rustc_hash::FxHashMap as HashMap;

pub struct NameResolver<'a, 'input> {
    ctx: &'a mut TypeContext<'input>,
    symbols: SymbolTable<'input>,
    trait_env: TraitEnv<'input>,
    diagnostics: DiagCtxt,
    current_scope: usize,
    current_function: Option<DefId>,
    current_type: Option<DefId>,
    import_map: Vec<ImportEntry>,
    local_crate_id: CrateId,
    /// Whether a top-level function named `main` was resolved.
    has_main: bool,
    /// Temporary mapping of type parameter names to GenericParam TypeIds
    /// used when resolving types inside an `impl<T>` block.
    current_impl_type_params: Option<HashMap<Symbol, TypeId>>,
    /// The resolved `for_type` of the current `impl` block, used to resolve
    /// `Self` in method bodies and associated type defaults.
    current_impl_for_type: Option<TypeId>,
    /// Pre-resolved name resolutions for the type checker.
    resolution_map: ResolutionMap<'input>,
    /// Current module path for registering full-qualified type paths.
    module_path: Vec<Symbol>,
    /// Layout aliases defined with `layout Name { ... }`.
    layout_aliases: HashMap<Symbol, Vec<Attribute<'input>>>,
    /// The builtin `Drop` trait's DefId — LAZILY resolved on the first
    /// `is_drop` check and cached (the builtin traits may not all be
    /// registered by `register_builtins` at `NameResolver::new` time;
    /// by the first `impl` resolution the symbols are populated).  The
    /// `is_drop` comparison uses this anchor instead of re-querying the
    /// symbol table on every `impl`.
    builtin_drop_def_id: Option<DefId>,
}

struct ImportEntry {
    path: Vec<Symbol>,
    alias: Option<Symbol>,
    items: Option<Vec<Symbol>>,
    span: Span,
}

impl<'input: 'a, 'a> NameResolver<'a, 'input> {
    pub fn new(ctx: &'a mut TypeContext<'input>, local_crate_id: CrateId) -> Self {
        let mut symbols = SymbolTable::new(local_crate_id);
        let mut trait_env = TraitEnv::new();
        // Register built-in types (Result, Option, Channel, etc.) so that
        // the resolver can resolve them in type annotations like
        // `fn f() -> Result<(), Int<32>>`.
        builtins::register_builtins(&mut symbols, &mut trait_env, ctx);
        NameResolver {
            ctx,
            symbols,
            trait_env,
            diagnostics: DiagCtxt::new(),
            current_scope: 0,
            current_function: None,
            current_type: None,
            import_map: Vec::new(),
            local_crate_id,
            has_main: false,
            current_impl_type_params: None,
            current_impl_for_type: None,
            resolution_map: ResolutionMap::default(),
            module_path: Vec::new(),
            layout_aliases: HashMap::default(),
            builtin_drop_def_id: None,
        }
    }

    pub fn resolve_program(
        &mut self,
        program: &Program<'input>,
    ) -> (
        SymbolTable<'input>,
        TraitEnv<'input>,
        DiagCtxt,
        ResolutionMap<'input>,
    ) {
        for item in &program.items {
            self.resolve_item(item);
        }

        let symbols = std::mem::replace(&mut self.symbols, SymbolTable::new(self.local_crate_id));
        let trait_env = std::mem::replace(&mut self.trait_env, TraitEnv::new());
        let mut resolution_map = std::mem::take(&mut self.resolution_map);
        resolution_map.has_main = self.has_main;
        let diags = std::mem::take(&mut self.diagnostics);

        (symbols, trait_env, diags, resolution_map)
    }

    /// Incremental resolution: processes only `new_items` and merges the
    /// results into the pre-existing `symbols`, `trait_env`, and `resolution_map`
    /// from a previous `resolve_program` call.
    ///
    /// Use this after Phase 2 `generate` expansion, where only newly generated
    /// items need name resolution — the existing declarations are already resolved.
    ///
    /// ⚠️  Generated items are processed in order: if a type alias references
    /// another generated type, the referenced type MUST appear first in `new_items`.
    /// Function bodies are deferred (signature-only resolution), so mutual
    /// function references are safe regardless of order.
    ///
    /// The global DefId allocator (`allocate_def_id`) continues from where the
    /// previous resolution left off, so there is no DefId collision risk.
    pub fn resolve_incremental(
        &mut self,
        new_items: &[Stmt<'input>],
        existing_symbols: SymbolTable<'input>,
        existing_trait_env: TraitEnv<'input>,
        existing_resolution_map: ResolutionMap<'input>,
    ) -> (
        SymbolTable<'input>,
        TraitEnv<'input>,
        DiagCtxt,
        ResolutionMap<'input>,
    ) {
        // Restore existing resolution state.
        self.has_main = existing_resolution_map.has_main;
        self.symbols = existing_symbols;
        self.trait_env = existing_trait_env;
        self.resolution_map = existing_resolution_map;
        self.diagnostics = DiagCtxt::new();
        self.current_scope = 0;

        // Only resolve the newly generated items.
        for item in new_items {
            self.resolve_item(item);
        }

        // Take results (same pattern as resolve_program).
        let symbols = std::mem::replace(&mut self.symbols, SymbolTable::new(self.local_crate_id));
        let trait_env = std::mem::replace(&mut self.trait_env, TraitEnv::new());
        let mut resolution_map = std::mem::take(&mut self.resolution_map);
        resolution_map.has_main = self.has_main;
        let diags = std::mem::take(&mut self.diagnostics);

        (symbols, trait_env, diags, resolution_map)
    }

    fn resolve_item(&mut self, item: &Stmt<'input>) {
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
                    let ty_id = self.ctx.generic_param(i, tp.name);
                    param_map.insert(tp.name, ty_id);
                }
                self.current_impl_type_params = Some(param_map);

                // Compute the attribute-derived flags BEFORE the mutable
                // `collect_function_signature` call (which holds a `&mut
                // self` borrow through its returned signature).
                let is_pure = self.has_pure_attribute(attributes);
                let is_ieee_contracts = self.has_ieee_contracts_attribute(attributes);
                let hints = self.extract_hints(attributes);

                let sig = self.collect_function_signature(
                    *name,
                    params,
                    return_type.as_ref(),
                    type_params,
                );
                // Own the signature so the `&mut self` borrow from
                // `collect_function_signature` ends before the root insert
                // below re-borrows `self.symbols`.
                let sig = sig.clone();

                let binding = FunctionBinding {
                    def_id,
                    signature: sig,
                    is_comptime: *is_comptime,
                    is_async: *is_async,
                    is_pure,
                    is_ieee_contracts,
                    hints,
                    contracts: contracts.clone(),
                    attributes: attributes.clone(),
                };
                // Nested function definitions are resolved via `resolve_stmt`
                // inside an enclosing function's body scope — but the checker
                // patches the return type via `update_function_return_type`
                // starting from the ROOT scope, so nested defs must be
                // registered at the root too (rustc-style item collection).
                // (The resolver's own `current_scope` does NOT control the
                // SymbolTable<'input>'s scope — hence the dedicated root insert.)
                if let Err(diag) = self.symbols.insert_function_at_root(*name, binding, *span) {
                    self.diagnostics.push(diag);
                }
                if name.eq_str("main") {
                    self.has_main = true;
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
                    if let Err(diag) = self
                        .symbols
                        .insert_variable(param.name, binding, param.span)
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
                self.resolution_map.type_def_ids.insert(*name, def_id);
                let type_params = params.clone();

                // Register generic parameters so that resolve_type_expr can
                // resolve T in `type Option<T> = enum { None, Some(T) }`.
                let mut param_map = HashMap::default();
                for (i, tp) in params.iter().enumerate() {
                    let ty_id = self.ctx.generic_param(i, tp.name);
                    param_map.insert(tp.name, ty_id);
                }
                // Save the previous param map and install the new one.
                let prev_param_map = self.current_impl_type_params.take();
                self.current_impl_type_params = Some(param_map);
                let kind = match definition {
                    TypeDefinition::Struct(_, _) => TypeKind::Struct,
                    TypeDefinition::Enum(_, _, _) => TypeKind::Enum,
                    TypeDefinition::Alias(_, _) => TypeKind::Alias,
                    TypeDefinition::Opaque(_, _) => TypeKind::Opaque,
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
                let exhaustive = attributes.iter().any(|a| a.name.eq_str("exhaustive"));

                match definition {
                    TypeDefinition::Struct(fields_def, _) => {
                        fields = fields_def
                            .iter()
                            .map(|f| {
                                let field_ty = self.resolve_type_expr(&f.ty);
                                FieldBinding {
                                    name: f.name,
                                    ty: field_ty,
                                    default: f.default.clone(),
                                    span: f.span,
                                }
                            })
                            .collect();
                    }
                    TypeDefinition::Enum(variants_def, mm, modifiers) => {
                        variants = variants_def.clone();
                        // The §Copy derivation: a generic enum is Copy iff
                        // EVERY variant payload is Copy (the value holds
                        // one variant) — collect the payloads into the ADT
                        // fields so `adt_is_copy` checks them uniformly.
                        for v in variants_def {
                            if let Some(p) = &v.payload {
                                // Recursive payloads (`Cons((Int<32>, &mut List))`)
                                // reference the ADT itself, which is not yet in
                                // the symbol table at this point — drop the
                                // "undefined type" diagnostics produced here;
                                // the field's type is then the error type
                                // (conservative for the §Copy derivation).
                                // The type checker RE-VALIDATES the field type
                                // later, so a genuine error (e.g. a misspelled
                                // payload type) is reported there, not lost.
                                let before = self.diagnostics.unreported_len();
                                let pt = self.resolve_type_expr(p);
                                if self.diagnostics.unreported_len() > before {
                                    self.diagnostics.truncate_unreported(before);
                                }
                                fields.push(FieldBinding {
                                    name: v.name,
                                    ty: pt,
                                    default: None,
                                    span: v.span,
                                });
                            }
                        }
                        missing_match = mm.clone();
                        // ── GADT constraint parameter name validation ───
                        // Check that every `when` constraint parameter is a
                        // declared type parameter of this enum, OR an
                        // existentially quantified variable in this variant.
                        for v in &variants {
                            for (pn, ct) in &v.eq_spec {
                                let in_params = params.iter().any(|tp| tp.name == *pn);
                                let in_exists = v.exists_params.iter().any(|ep| ep == pn);
                                if !in_params && !in_exists {
                                    self.diagnostics.push(
                                        Diagnostic::error(format!(
                                            "unknown type parameter `{}` in GADT `when` constraint",
                                            pn,
                                        ))
                                        .with_code_str("E062")
                                        .with_span(*span)
                                        .with_help(
                                            format!(
                                                "`{}` is not a type parameter of this enum \
                                                 or an `exists` variable in this variant; \
                                                 use one of: {}",
                                                pn,
                                                params
                                                    .iter()
                                                    .map(|tp| tp.name.as_str())
                                                    .chain(
                                                        v.exists_params
                                                            .iter()
                                                            .map(|ep| ep.as_str())
                                                    )
                                                    .collect::<Vec<_>>()
                                                    .join(", "),
                                            ),
                                        ),
                                    );
                                }
                                // ── GADT constraint RHS validation ──
                                // The right-hand side of a `when` constraint
                                // must NOT reference another type parameter
                                // of the SAME enum: `when T == U and U == T`
                                // would register a mutual refinement cycle
                                // (A → B, B → A) that `resolve_binding_tail`
                                // would chase until MAX_CHAIN_DEPTH.  An
                                // `exists` variable IS allowed on the RHS
                                // (`when T == X` — the witness stays opaque).
                                if let Some(bad) = find_param_ref_in_type(ct, &params) {
                                    self.diagnostics.push(
                                        Diagnostic::error(format!(
                                            "GADT `when` constraint right-hand side references \
                                             type parameter `{}` of the same enum",
                                            bad,
                                        ))
                                        .with_code_str("E064")
                                        .with_span(ct.span())
                                        .with_help(
                                            "the right-hand side of `when` must be a concrete \
                                             type or an `exists` variable, not another type \
                                             parameter of this enum",
                                        ),
                                    );
                                }
                            }
                            // ── Exists parameter name duplication check ──
                            // An `exists X` variable must not shadow an enum
                            // type parameter name.
                            for ep in &v.exists_params {
                                if params.iter().any(|tp| tp.name == *ep) {
                                    self.diagnostics.push(
                                        Diagnostic::error(format!(
                                            "`exists {}` shadows enum type parameter `{}`",
                                            ep, ep,
                                        ))
                                        .with_code_str("E063")
                                        .with_span(*span)
                                        .with_help(
                                            format!(
                                                "rename the `exists` variable or the type parameter"
                                            ),
                                        ),
                                    );
                                }
                            }
                            // ── GADT constraint satisfiability check (E065) ──
                            // A variant whose `when` constraints force the same
                            // type parameter to two DIFFERENT concrete types
                            // (`when T == Int<32> and T == Bool`) is unsatisfiable:
                            // the variant cannot be constructed at ANY instantiation
                            // — a provable logical contradiction (unlike the
                            // payload/constraint heuristic, which stays a warning).
                            for (i, (pn, ct1)) in v.eq_spec.iter().enumerate() {
                                for (pn2, ct2) in v.eq_spec.iter().skip(i + 1) {
                                    if pn2 != pn {
                                        continue;
                                    }
                                    // Only fire when BOTH RHS are provably concrete:
                                    // an `exists` witness (at ANY depth — the witness
                                    // may equal anything) or an alias path (two names
                                    // may alias the same type) keeps the pair
                                    // satisfiable, so the syntactic inequality is not
                                    // a provable contradiction.
                                    if !rhs_is_provably_concrete(
                                        ct1,
                                        &v.exists_params,
                                        &self.symbols,
                                    ) || !rhs_is_provably_concrete(
                                        ct2,
                                        &v.exists_params,
                                        &self.symbols,
                                    ) {
                                        continue;
                                    }
                                    // Nominal comparison: the RESOLVED top-level
                                    // constructor identity (`Int` and `core::Int`
                                    // are the same type), then the generic args.
                                    // `None` (unresolvable form) is conservative —
                                    // no contradiction asserted.
                                    let differ = match (
                                        crate::hir::type_eq::concrete_ctor_key(
                                            ct1,
                                            &v.exists_params,
                                            &self.symbols,
                                        ),
                                        crate::hir::type_eq::concrete_ctor_key(
                                            ct2,
                                            &v.exists_params,
                                            &self.symbols,
                                        ),
                                    ) {
                                        (Some(k1), Some(k2)) => {
                                            k1 != k2
                                                || !crate::hir::type_eq::type_args_eq_ignoring_spans(
                                                    ct1,
                                                    ct2,
                                                    &v.exists_params,
                                                    &self.symbols,
                                                )
                                        }
                                        _ => false,
                                    };
                                    if differ {
                                        self.diagnostics.push(
                                            Diagnostic::error(format!(
                                                "GADT `when` constraints on `{}` are \
                                                 contradictory ({} vs {}); the variant cannot \
                                                 be constructed at any instantiation",
                                                pn,
                                                ast_type_display(ct1),
                                                ast_type_display(ct2),
                                            ))
                                            .with_code_str("E065")
                                            .with_span(ct2.span())
                                            .with_help(
                                                "the same type parameter is constrained to two \
                                                 different concrete types; make the constraints \
                                                 consistent",
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                        // ── GADT + `with default` prohibition (SYNTAX.md §"Type-level Default Values") ──
                        // Only *generic* GADT enums whose `when` constraints involve
                        // their type parameters (or `exists` variables) cannot have a
                        // default value — a single default cannot satisfy all possible
                        // instantiations.  Non-generic enums whose constraints reference
                        // only global constants are unaffected.
                        let constraints_involve_type_params = !params.is_empty()
                            && variants.iter().any(|v| {
                                v.eq_spec.iter().any(|(pn, _)| {
                                    params.iter().any(|tp| tp.name == *pn)
                                        || v.exists_params.contains(pn)
                                })
                            });
                        if constraints_involve_type_params {
                            for m in modifiers {
                                if matches!(m, crate::ast::TypeModifier::Default(_)) {
                                    self.diagnostics.push(
                                        Diagnostic::error(
                                            "GADT enum with generic type parameters cannot have a `with default` clause"
                                        )
                                        .with_code_str("E061")
                                        .with_span(*span)
                                        .with_help(
                                            "a GADT enum's type parameters are constrained by `when` clauses, \
                                             so a single default value cannot satisfy all variants"
                                        )
                                        .with_suggestion(
                                            "remove the `with default` clause, or make the enum non-generic"
                                        ),
                                    );
                                }
                            }
                        }
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
                    if attr.name.eq_str("layout") {
                        for arg in &attr.args {
                            if let crate::ast::Expr::Ident(name, _) = arg {
                                if name.eq_str("C") {
                                    c_layout = true;
                                } else if let Some(alias_attrs) = self.layout_aliases.get(&name) {
                                    for alias_attr in alias_attrs {
                                        if !expanded_attrs.iter().any(|a| a.name == alias_attr.name)
                                        {
                                            expanded_attrs.push(alias_attr.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if attr.name.eq_str("transparent") {
                        transparent = true;
                    }
                    if attr.name.eq_str("packed") {
                        packed = true;
                    }
                    if attr.name.eq_str("endian") {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Ident(name, _)) if name.eq_str("little") => {
                                endian = Some(crate::ast::Endianness::Little);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) if name.eq_str("big") => {
                                endian = Some(crate::ast::Endianness::Big);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "`@endian` expects `little` or `big`, got `{}`",
                                        name
                                    ))
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
                                        .with_suggestion(
                                            "write `@endian(little)` or `@endian(big)`",
                                        ),
                                );
                            }
                        }
                    }
                    if attr.name.eq_str("bit_order") {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Ident(name, _)) if name.eq_str("lsb_to_msb") => {
                                bit_order = Some(crate::ast::BitOrder::LsbToMsb);
                            }
                            Some(crate::ast::Expr::Ident(name, _)) if name.eq_str("msb_to_lsb") => {
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
                    if attr.name.eq_str("align") || attr.name.eq_str("pad") {
                        match attr.args.first() {
                            Some(crate::ast::Expr::Literal(crate::ast::Literal::Int(n), _)) => {
                                if attr.name.eq_str("align") {
                                    align = Some(n.to_u64().unwrap_or(0));
                                }
                                if attr.name.eq_str("pad") {
                                    pad = Some(n.to_u64().unwrap_or(0));
                                }
                            }
                            Some(_) => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "`@{}` requires an integer argument",
                                        attr.name
                                    ))
                                    .with_code_str("E060")
                                    .with_span(attr.span)
                                    .with_suggestion(format!(
                                        "write `@{}(N)` where N is a power of two",
                                        attr.name
                                    )),
                                );
                            }
                            None => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "`@{}` requires an integer argument",
                                        attr.name
                                    ))
                                    .with_code_str("E060")
                                    .with_span(attr.span)
                                    .with_suggestion(format!("write `@{}(128)`", attr.name)),
                                );
                            }
                        }
                    }
                }

                // Register the ADT definition (struct/enum) — the §Copy
                // derivation (`type_is_copy`) queries the fields.
                if matches!(kind, TypeKind::Struct | TypeKind::Enum) {
                    self.ctx.register_adt(
                        def_id,
                        crate::hir::types::AdtDef {
                            fields: fields.iter().map(|f| f.ty).collect(),
                            has_drop: false,
                        },
                    );
                }

                let binding = TypeBinding {
                    def_id,
                    params: type_params,
                    kind,
                    span: *span,
                    alias_ast,
                    attributes: attributes.clone(),
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
                if let Err(diag) = self.symbols.insert_type(*name, binding, *span) {
                    self.diagnostics.push(diag);
                }
                // Register the fully-qualified path for multi-segment resolution.
                {
                    let mut full_path = self.module_path.clone();
                    full_path.push(*name);
                    let full = Symbol::intern(
                        &full_path
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join("::"),
                    );
                    self.symbols.register_full_path(full, def_id);
                }
                // Populate the resolution map for the type checker
                if let Some(b) = self.symbols.lookup_type(*name) {
                    self.resolution_map.type_bindings.insert(def_id, b.clone());
                }
                // Restore the previous type param map (if any).
                self.current_impl_type_params = prev_param_map;
            }
            Stmt::TraitDef {
                span,
                name,
                methods,
                associated_types,
                attributes,
                ..
            } => {
                let def_id = self.allocate_def_id();
                // Set current_impl_for_type to a fresh generic param for Self,
                // so that trait method signatures can reference `Self` in their
                // parameter and return types (e.g. `def clone(self) -> Self`).
                let self_param = self.ctx.generic_param(0, Symbol::intern("Self"));
                self.current_impl_for_type = Some(self_param);
                let mut method_bindings = Vec::new();
                for method in methods {
                    // Own the signature so the `&mut self` borrow from
                    // `collect_trait_method_signature` ends before the next
                    // loop iteration.
                    let sig = self.collect_trait_method_signature(method).clone();
                    method_bindings.push((method.name, sig));
                }
                self.current_impl_for_type = None;

                let binding = TraitBinding {
                    def_id,
                    methods: method_bindings,
                    associated_types: associated_types
                        .iter()
                        .map(|at| (at.name, at.default.clone()))
                        .collect(),
                    super_traits: vec![],
                    span: *span,
                    attributes: attributes.clone(),
                    crate_id: self.symbols.local_crate_id,
                };
                if let Err(diag) = self.symbols.insert_trait(*name, binding, *span) {
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
                    let ty_id = self.ctx.generic_param(i, tp.name);
                    param_map.insert(tp.name, ty_id);
                }
                self.current_impl_type_params = Some(param_map);

                let resolved_for = self.resolve_type_expr(for_type);
                self.current_impl_for_type = Some(resolved_for);
                let resolved_trait = trait_path.as_ref().and_then(|tp| {
                    // Extract the path from the trait type for lookup.
                    match tp {
                        Type::Path(path, _) => self.resolve_trait_path(path),
                        _ => {
                            // For complex trait types (e.g. `Add<Int<32>>`),
                            // resolve as a type expression and get the DefId.
                            let ty = self.resolve_type_expr(tp);
                            self.ctx.get_def_id_for_type(ty)
                        }
                    }
                });

                // The §Copy derivation: an `impl Drop for T` marks the
                // ADT as non-Copy (`type_is_copy` consults `has_drop`).
                if let Some(rt) = &resolved_trait {
                    // The fully precise check: compare the RESOLVED trait
                    // DefId against the builtin `Drop` DefId.  The anchor
                    // is lazily resolved from the resolver's own `symbols`
                    // (the builtin traits are registered into it; NOT the
                    // TypeChecker's `builtin_registry`, which is populated
                    // only AFTER name resolution — the resolver runs
                    // first) and cached — a user-defined trait named
                    // `Drop` has a DIFFERENT DefId and no longer marks
                    // the ADT non-Copy.
                    if self.builtin_drop_def_id.is_none() {
                        self.builtin_drop_def_id = self
                            .symbols
                            .lookup_trait(Symbol::intern("Drop"))
                            .map(|b| b.def_id);
                    }
                    let is_drop = Some(*rt) == self.builtin_drop_def_id;
                    if is_drop {
                        if let crate::hir::types::TypeData::Adt { def_id, .. } =
                            *self.ctx.get(resolved_for)
                        {
                            self.ctx.set_adt_has_drop(def_id);
                        }
                    }
                    let _ = rt;
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

                let has_auto_deref = attributes.iter().any(|a| a.name.eq_str("auto_deref"));

                // Pre-resolve method param types using the impl's type param mapping,
                // so generic params like `T` are properly substituted in lookup_method.
                // Each method ALSO gets its OWN DefId here — the "assoc item"
                // identity (mirroring rustc's `AssocItem.def_id`) — registered
                // under (receiver type DefId, name) so the AST pre-scan
                // (`collect_function_effects`) and the checker agree on the
                // method's identity even before the impl is registered.
                let receiver_def = self.ctx.get_def_id_for_type(resolved_for);
                let mut resolved_methods = Vec::new();
                for method in methods {
                    let method_def = self.allocate_def_id();
                    if let Some(receiver_def) = receiver_def {
                        self.symbols
                            .insert_method_def_id(receiver_def, method.name, method_def);
                    }
                    let mut param_tys = Vec::with_capacity(method.params.len());
                    for p in &method.params {
                        if let Some(ref param_ty) = p.ty {
                            let resolved_ty = self.resolve_self_in_type(param_ty, for_type);
                            param_tys.push(self.resolve_type_expr(&resolved_ty));
                        } else {
                            param_tys.push(self.ctx.error());
                        }
                    }
                    let resolved_ret = self.resolve_self_in_type(&method.return_type, for_type);
                    let ret_ty = self.resolve_type_expr(&resolved_ret);
                    resolved_methods.push(crate::hir::traits::MethodInfo {
                        def_id: method_def,
                        name: method.name,
                        param_tys,
                        ret_ty,
                        span: method.span,
                        attributes: method.attributes.clone(),
                        has_auto_deref,
                    });
                }

                // Resolve associated types from the impl block.
                let mut assoc_tys = Vec::new();
                for at in associated_types {
                    if let Some(ref default) = at.default {
                        let resolved = self.resolve_type_expr(default);
                        assoc_tys.push((at.name, resolved));
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
                        // context, arity, trait_args, where_clause_bounds are
                        // populated by the checker (checker/mod.rs) which
                        // performs the actual add_impl call.  The resolver
                        // no longer registers impls — it only resolves method
                        // signatures for method lookup.  See the deferral
                        // NOTE below.
                        context: vec![],
                        arity: 0,
                        trait_args: vec![],
                        where_clause_bounds: vec![],
                        span: *span,
                    };
                    // NOTE: Impl registration is deferred to the type checker
                    // (checker/mod.rs).  Registering here AND in the checker
                    // would cause double registration, triggering false
                    // positives in the overlap check.  The candidate is
                    // discarded after resolution.
                    let _ = candidate;
                }

                // Clear impl type params so they don't leak into subsequent statements.
                self.current_impl_type_params = None;
                self.current_impl_for_type = None;

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
                    alias: *alias,
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
            Stmt::Edition(version, span) => match crate::hir::types::Edition::from_str(version) {
                Some(ed) => self.ctx.set_edition(ed),
                None => {
                    self.diagnostics.push(
                        Diagnostic::error(format!("unknown edition `{}`", version))
                            .with_code_str("E070")
                            .with_span(*span)
                            .with_suggestion("use a valid edition: `\"2024\"` or `\"2026\"`"),
                    );
                }
            },
            Stmt::LayoutDef {
                name, attributes, ..
            } => {
                // Register a layout alias so that @layout(AliasName) can be expanded.
                if self.layout_aliases.contains_key(name) {
                    self.diagnostics.push(
                        Diagnostic::error(format!("duplicate layout alias `{}`", name))
                            .with_span(Span::new(0, 0)),
                    );
                } else {
                    self.layout_aliases.insert(*name, attributes.clone());
                }
            }
            Stmt::Constraint {
                name,
                params,
                predicates,
                span,
                ..
            } => {
                // Register type parameters so T resolves in constraint body.
                let mut param_map = HashMap::default();
                for (i, tp) in params.iter().enumerate() {
                    let ty_id = self.ctx.generic_param(i, tp.name);
                    param_map.insert(tp.name, ty_id);
                }
                self.current_impl_type_params = Some(param_map);
                let resolved_predicates: Vec<ConstraintPredicate> = predicates
                    .iter()
                    .map(|p| {
                        let subject = self.resolve_type_expr(&p.ty);
                        let bounds: Vec<TypeId> = p
                            .bounds
                            .iter()
                            .map(|b| {
                                // Before attempting type resolution, check whether this
                                // bound is a trait name — traits are not registered as
                                // types, so resolve_type_expr would produce an error.
                                if let Type::Path(path, _) = b
                                    && path.len() == 1
                                    && let Some(trait_binding) = self.symbols.lookup_trait(path[0])
                                {
                                    return self.ctx.dyn_trait(vec![trait_binding.def_id]);
                                }
                                self.resolve_type_expr(b)
                            })
                            .collect();
                        ConstraintPredicate { subject, bounds }
                    })
                    .collect();
                self.current_impl_type_params = None;
                let binding = ConstraintBinding {
                    predicates: resolved_predicates,
                    params: params.clone(),
                    span: *span,
                };
                if let Err(diag) = self.symbols.insert_constraint(*name, binding, *span) {
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
                // Compute attribute-derived flags BEFORE the mutable
                // `collect_function_signature` call.
                let is_ieee_contracts = self.has_ieee_contracts_attribute(attributes);
                let hints = self.extract_hints(attributes);
                let sig = self.collect_function_signature(*name, params, Some(return_type), &[]);
                let binding = FunctionBinding {
                    def_id,
                    signature: sig,
                    is_comptime: false,
                    is_async: false,
                    is_pure: false,
                    is_ieee_contracts,
                    hints,
                    contracts: Vec::new(),
                    attributes: attributes.clone(),
                };
                if let Err(diag) = self.symbols.insert_function(*name, binding, *span) {
                    self.diagnostics.push(diag);
                }
            }
            _ => {
                if let Some(stmt_span) = self.get_stmt_span(item) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "`set` and `let` statements are not allowed at the top level; only declarations (`def`, `type`, `trait`, `import`, `impl`, `constraint`, `comptime`, `extern`, `edition`) are permitted here",
                        )
                        .with_code_str("E018")
                        .with_span(stmt_span),
                    );
                }
            }
        }
    }

    fn resolve_stmt(&mut self, stmt: &Stmt<'input>) {
        match stmt {
            // Nested function definitions: register them like items
            // (rustc-style — items in blocks are collected and
            // referenced).  Without this, a nested `def` would never be
            // registered and the checker's `update_function_return_type`
            // would panic ("function not found in any scope").
            Stmt::FunctionDef { .. } => self.resolve_item(stmt),
            // Nested type definitions — register them like items
            // (rustc-style, mirroring the FunctionDef<'input> arm above).  Before
            // this arm existed, a `type R = ...` inside a function body was
            // silently dropped: `R` was never registered in the symbol
            // table, so a later reference (`set x: R`) reported
            // "undefined type: R".
            Stmt::TypeDef { .. } => self.resolve_item(stmt),
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
                    if let Err(diag) = self.symbols.insert_variable(*name, binding, *span) {
                        self.diagnostics.push(diag);
                    }
                }

                // `set auto<T> = expr` — register each capture name as a type
                // in the resolution map so that comptime code can reference it.
                for cap in type_captures {
                    // Allocate a unique DefId for each capture so that
                    // placeholder bindings don't collide in the type_defs map.
                    let cap_def_id = self.allocate_def_id();
                    let _placeholder = self.ctx.error();
                    self.resolution_map
                        .type_def_ids
                        .insert(cap.name, cap_def_id);
                    // The actual type binding will be updated by the checker
                    // after inferring the expression's type.
                    let binding = TypeBinding {
                        def_id: cap_def_id,
                        params: vec![],
                        kind: TypeKind::Alias,
                        span: *span,
                        alias_ast: None,
                        attributes: vec![],
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
                    self.symbols.insert_type(cap.name, binding, *span).ok();
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

    fn resolve_expr(&mut self, expr: &Expr<'input>) -> Option<TypeId> {
        match expr {
            Expr::Literal(lit, _span) => {
                let ty = self.literal_type(lit);
                Some(ty)
            }
            Expr::Ident(name, span) => {
                if let Some(binding) = self.symbols.lookup_variable(*name, *span) {
                    Some(binding.ty)
                } else if let Some(func) = self.symbols.lookup_function(*name) {
                    let sig = func.signature.clone();
                    let ty = self.ctx.function(
                        sig.params.iter().map(|p| p.ty).collect(),
                        sig.return_type.get(),
                    );
                    Some(ty)
                } else if let Some(_ty_binding) = self.symbols.lookup_type(*name) {
                    None
                } else if name.eq_str("result") {
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
                if let Some(def_id) = def_id
                    && let Some(binding) = self.symbols.lookup_type_by_def_id(def_id)
                    && binding.kind == TypeKind::Struct
                {
                    return Some(self.ctx.struct_ty(def_id, vec![]));
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
                if let Some(def_id) = self.resolve_type_path(path)
                    && let Some(binding) = self.symbols.lookup_type_by_def_id(def_id)
                    && binding.kind == TypeKind::Enum
                {
                    return Some(self.ctx.enum_ty(def_id, vec![]));
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
                    if let Some(ty) = self.resolve_expr(e)
                        && elem_ty.is_none()
                    {
                        elem_ty = Some(ty);
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
                    if let Err(diag) = self
                        .symbols
                        .insert_variable(param.name, binding, param.span)
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
                        if let Err(diag) = self.symbols.insert_variable(*bind, binding, branch.span)
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
                    Diagnostic::error(format!(
                        "unresolved path: {}",
                        path.iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join("::")
                    ))
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
            Expr::LayoutOf(ty, _) => {
                // layout_of!(Type) — resolve the type argument, return error type
                // as a placeholder (the actual LayoutDescriptor is computed at
                // comptime, but the type system cannot determine it here).
                self.resolve_type_expr(ty);
                Some(self.ctx.error())
            }
            Expr::CompileError(msg, span) => {
                // @compile_error!("msg") — emit an error and continue (deferred to checker).
                Some(self.ctx.error())
            }
            Expr::Task { body, .. } => {
                for s in body {
                    self.resolve_stmt(s);
                }
                Some(self.ctx.unit())
            }
        }
    }

    fn resolve_pattern(&mut self, pattern: &Pattern<'input>) {
        match pattern {
            Pattern::Wildcard(..) => {}
            Pattern::Ident(name, span) => {
                let binding = VariableBinding {
                    ty: self.ctx.error(),
                    mutable: false,
                    span: *span,
                    def_id: self.allocate_def_id(),
                };
                if let Err(diag) = self.symbols.insert_variable(*name, binding, *span) {
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

    fn resolve_type_expr(&mut self, ty: &Type<'input>) -> TypeId {
        match ty {
            Type::Path(path, span) => {
                // Check if this name refers to `Self` in an impl block.
                if path.len() == 1
                    && path[0].eq_str("Self")
                    && let Some(self_ty) = self.current_impl_for_type
                {
                    return self_ty;
                }
                // Check if this name refers to an impl type parameter (e.g. `T` in `impl<T>`)
                if path.len() == 1
                    && let Some(ref param_map) = self.current_impl_type_params
                    && let Some(&ty_id) = param_map.get(&path[0])
                {
                    return ty_id;
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
                    let name = path[0];
                    if name.eq_str("Bool") {
                        self.ctx.bool()
                    } else if name.eq_str("Char") {
                        self.ctx.char()
                    } else if name.eq_str("Byte") {
                        self.ctx.byte()
                    } else if name.eq_str("USize") {
                        self.ctx.usize()
                    } else if name.eq_str("Unit") {
                        self.diagnostics.push(
                            Diagnostic::error("use `()` instead of `Unit`")
                                .with_code_str("E031")
                                .with_help(
                                    "Posita uses `()` (empty tuple) to express the unit type",
                                )
                                .with_suggestion("replace `Unit` with `()`"),
                        );
                        self.ctx.error()
                    } else if name.eq_str("Never") {
                        self.ctx.never()
                    } else if name.eq_str("Int")
                        || name.eq_str("UInt")
                        || name.eq_str("Float")
                        || name.eq_str("Rational")
                    {
                        // These require type arguments; handled in Type::Generic
                        self.ctx.error()
                    } else {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "undefined type: {}",
                                path.iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join("::")
                            ))
                            .with_span(*span),
                        );
                        self.ctx.error()
                    }
                }
            }
            Type::Generic(base, args, span) => {
                // Handle generic built-in types (Int, UInt, Float) by matching base path
                if let Type::Path(path, _) = base
                    && path.len() == 1
                {
                    if path[0].eq_str("Int") {
                        let bits = self
                            .extract_int_from_type(args[0].ty().as_ref())
                            .unwrap_or(32);
                        return self.ctx.int(bits, true);
                    } else if path[0].eq_str("UInt") {
                        let bits = self
                            .extract_int_from_type(args[0].ty().as_ref())
                            .unwrap_or(32);
                        return self.ctx.int(bits, false);
                    } else if path[0].eq_str("Float") {
                        let bits = self
                            .extract_int_from_type(args[0].ty().as_ref())
                            .unwrap_or(64);
                        return self.ctx.float(bits);
                    } else if path[0].eq_str("Rational") {
                        let p = self
                            .extract_int_from_type(args[0].ty().as_ref())
                            .unwrap_or(16);
                        let q = self
                            .extract_int_from_type(args[1].ty().as_ref())
                            .unwrap_or(16);
                        return self.ctx.rational(p as u8, q as u8);
                    } else if path[0].eq_str("Ptr") {
                        let size = args
                            .get(0)
                            .map(|a| self.resolve_type_expr(a.ty().as_ref()))
                            .unwrap_or(self.ctx.usize());
                        let pointee = args
                            .get(1)
                            .map(|a| self.resolve_type_expr(a.ty().as_ref()))
                            .unwrap_or(self.ctx.error());
                        return self.ctx.ptr(size, pointee);
                    } else if path[0].eq_str("USize") {
                        return self.ctx.usize();
                    }
                }
                let base_ty = self.resolve_type_expr(base);
                if let Some(def_id) = self.ctx.get_def_id_for_type(base_ty) {
                    let binding = self.symbols.lookup_type_by_def_id(def_id).cloned();
                    if let Some(binding) = binding {
                        let arg_tys: Vec<TypeId> = args
                            .iter()
                            .map(|a| self.resolve_type_expr(a.ty().as_ref()))
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
                inner: ty,
                mutable,
                lifetime,
                ..
            } => {
                let inner = self.resolve_type_expr(ty);
                // Keep the explicit lifetime annotation (`&'a T`) in the
                // resolved type for the region solver.
                self.ctx.reference_with_lifetime(inner, *mutable, *lifetime)
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
                if let Expr::Literal(Literal::Int(size_val), _) = size {
                    self.ctx.array(inner, size_val.to_u64().unwrap_or(0))
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
            Type::Forall { lifetime, body, .. } => {
                // Higher-ranked type `for<'a> T`: the lifetime is
                // universally quantified — allocate a fresh binder index
                // (the checker skolemizes it at the call site).
                let body_ty = self.resolve_type_expr(body);
                self.ctx
                    .forall(self.ctx.fresh_param_index(), *lifetime, body_ty)
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
                self.ctx.exists(
                    self.ctx.fresh_param_index(),
                    *name,
                    base_ty,
                    (*invariant).clone(),
                )
            }
            Type::WhereShorthand {
                base,
                invariant,
                span,
            } => {
                // Desugar `type T = Base where value > 0` into `exists _where_N: Base invariant _where_N > 0`.
                let name = Symbol::intern(&format!("_where_{}", span.start));
                let arena = self
                    .ctx
                    .arena
                    .expect("arena required for the where-invariant desugar");
                let inv = replace_ident_in_expr(arena, &invariant, Symbol::intern("value"), name);
                let base_ty = self.resolve_type_expr(base);
                self.ctx
                    .exists(self.ctx.fresh_param_index(), name, base_ty, (*inv).clone())
            }
            Type::Literal(expr, ..) => self.resolve_expr(expr).unwrap_or(self.ctx.error()),
            Type::Never(..) => self.ctx.never(),
            Type::Union(tys, ..) => self.ctx.error(),
            Type::Error(..) => self.ctx.error(),
            Type::Expr(expr, ..) => self.resolve_expr(expr).unwrap_or(self.ctx.error()),
            Type::Regex(pattern, _) => {
                // Pattern<'input> validated by the parser at parse time.  `Type::Regex` is
                // only constructed by the parser — there is no macro expansion or
                // deserialization path that produces `ast::Type::Regex` nodes.
                // The `debug_assert!` below is a safety net for debug builds only.
                debug_assert!(
                    regex_syntax::parse(pattern).is_ok(),
                    "Regex pattern should have been validated at parse time: {}",
                    pattern,
                );
                self.ctx.regex(pattern.clone())
            }
        }
    }

    fn resolve_type_path(&mut self, path: &[Symbol]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        self.symbols.lookup_type_by_path(path)
    }

    fn resolve_trait_path(&mut self, path: &[Symbol]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        self.symbols.lookup_trait_by_path(path)
    }

    /// Resolve a trait path from a bound `Type` (e.g. `Foo` or `Add<Int<32>>`).
    /// Extracts the path from the `Type` and calls `resolve_trait_path`.
    fn resolve_trait_path_from_bound(&mut self, bound: &Type<'input>) -> Option<DefId> {
        let path = match bound {
            Type::Path(path, _) => path,
            Type::Generic(base, _, _) => match base {
                Type::Path(path, _) => path,
                _ => return None,
            },
            _ => return None,
        };
        self.resolve_trait_path(path)
    }

    fn extract_int_from_type(&self, ty: &Type<'input>) -> Option<u32> {
        match ty {
            Type::Literal(expr, _) => {
                if let Expr::Literal(Literal::Int(val), _) = expr {
                    if *val > 64 {
                        return None;
                    }
                    val.to_u64().and_then(|n| u32::try_from(n).ok())
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

    fn get_stmt_span(&self, stmt: &Stmt<'input>) -> Option<Span> {
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

    fn has_pure_attribute(&self, attributes: &[Attribute<'input>]) -> bool {
        attributes.iter().any(|attr| attr.name.eq_str("pure"))
    }

    fn has_ieee_contracts_attribute(&self, attributes: &[Attribute<'input>]) -> bool {
        attributes
            .iter()
            .any(|attr| attr.name.eq_str("ieee_contracts"))
    }

    fn extract_hints(&self, attributes: &[Attribute<'input>]) -> Vec<Expr<'input>> {
        attributes
            .iter()
            .filter(|attr| attr.name.eq_str("hint"))
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
        name: Symbol,
        params: &[Param<'input>],
        return_type: Option<&Type<'input>>,
        type_params: &[TypeParam<'input>],
    ) -> FunctionSignature<'input> {
        FunctionSignature {
            params: params
                .iter()
                .map(|p| {
                    let ty =
                        p.ty.as_ref()
                            .map_or(self.ctx.error(), |t| self.resolve_type_expr(t));
                    Parameter {
                        name: p.name,
                        ty,
                        span: p.span,
                        default: p.default.clone(),
                    }
                })
                .collect(),
            return_type: Rc::new(Cell::new(match return_type {
                Some(t) => self.resolve_type_expr(t),
                None => self.ctx.unit(),
            })),
            type_params: type_params.to_vec(),
            where_clause: None,
        }
    }

    fn collect_trait_method_signature(
        &mut self,
        method: &TraitMethod<'input>,
    ) -> FunctionSignature<'input> {
        FunctionSignature {
            params: method
                .params
                .iter()
                .map(|p| {
                    let ty =
                        p.ty.as_ref()
                            .map_or(self.ctx.error(), |t| self.resolve_type_expr(t));
                    Parameter {
                        name: p.name,
                        ty,
                        span: p.span,
                        default: p.default.clone(),
                    }
                })
                .collect(),
            return_type: Rc::new(Cell::new(self.resolve_type_expr(&method.return_type))),
            type_params: Vec::new(),
            where_clause: None,
        }
    }

    pub fn into_symbols(self) -> SymbolTable<'input> {
        self.symbols
    }

    pub fn diagnostics(&self) -> &DiagCtxt {
        &self.diagnostics
    }

    /// Recursively substitute `Self` in an AST type with `self_ty`.
    /// Needed for resolving method signatures in impl blocks, where
    /// `&self` desugars to `Self` which resolve_type_expr cannot handle.
    fn resolve_self_in_type(&self, ty: &Type<'input>, self_ty: &Type<'input>) -> Type<'input> {
        match ty {
            Type::Path(p, s) if p.len() == 1 && (p[0].eq_str("Self") || p[0].eq_str("self")) => {
                self_ty.clone()
            }
            Type::Reference {
                inner,
                mutable,
                span: s,
                ..
            } => Type::Reference {
                inner: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(inner, self_ty)),
                mutable: *mutable,
                lifetime: None,
                span: *s,
            },
            Type::Pointer(inner, s) => Type::Pointer(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(inner, self_ty)),
                *s,
            ),
            Type::Generic(base, args, span) => {
                let new_base = self.resolve_self_in_type(base, self_ty);
                let new_args: Vec<GenericArg<'input>> = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Positional(t) => {
                            GenericArg::Positional(self.resolve_self_in_type(t, self_ty))
                        }
                        GenericArg::Named(n, t) => {
                            GenericArg::Named(*n, self.resolve_self_in_type(t, self_ty))
                        }
                        GenericArg::Const(ac) => GenericArg::Const(crate::ast::AnonConst {
                            value: ac.value,
                            span: ac.span,
                        }),
                    })
                    .collect();
                Type::Generic(
                    self.ctx
                        .arena
                        .expect("arena required for type construction")
                        .alloc(new_base),
                    new_args,
                    *span,
                )
            }
            Type::Tuple(tys, span) => Type::Tuple(
                tys.iter()
                    .map(|t| self.resolve_self_in_type(t, self_ty))
                    .collect(),
                *span,
            ),
            Type::Slice(inner, span) => Type::Slice(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(inner, self_ty)),
                *span,
            ),
            Type::Array(inner, size, span) => Type::Array(
                self.ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(inner, self_ty)),
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
                ret: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(ret, self_ty)),
                span: *span,
            },
            Type::Projection {
                impl_type,
                trait_path,
                assoc_name,
                span,
            } => Type::Projection {
                impl_type: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(impl_type, self_ty)),
                trait_path: self
                    .ctx
                    .arena
                    .expect("arena required for type construction")
                    .alloc(self.resolve_self_in_type(trait_path, self_ty)),
                assoc_name: *assoc_name,
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
        path: &[Symbol],
        items: &Option<Vec<Symbol>>,
        alias: &Option<Symbol>,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let import_name = alias.as_ref().or_else(|| path.last()).copied();

        // First try to resolve as a type.
        if let Some(def_id) = self.symbols.lookup_type_by_path(path) {
            if let Some(name) = &import_name {
                self.resolution_map.type_def_ids.insert(*name, def_id);
                if let Some(binding) = self.symbols.lookup_type_by_def_id(def_id).cloned() {
                    self.symbols.insert_type(*name, binding, span).ok();
                }
                // Register the import's original full path for re-export resolution.
                let full_path = Symbol::intern(
                    &path
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join("::"),
                );
                self.symbols.register_full_path(full_path, def_id);
            }
            // `from path import { items }`
            if let Some(item_list) = items {
                for item in item_list {
                    let item_path = [*item];
                    if let Some(item_def_id) = self.symbols.lookup_type_by_path(&item_path) {
                        self.resolution_map.type_def_ids.insert(*item, item_def_id);
                        if let Some(binding) =
                            self.symbols.lookup_type_by_def_id(item_def_id).cloned()
                        {
                            self.symbols.insert_type(*item, binding, span).ok();
                        }
                        // Register the full path: path::to::item
                        let mut full_item_path = path.to_vec();
                        full_item_path.push(*item);
                        let full_item_str = Symbol::intern(
                            &full_item_path
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .join("::"),
                        );
                        self.symbols.register_full_path(full_item_str, item_def_id);
                    }
                }
            }
            return Ok(());
        }

        // Try as a trait — supports multi-segment paths.
        if let Some(trait_def_id) = self.symbols.lookup_trait_by_path(path)
            && let Some(trait_binding) = self.symbols.lookup_trait_by_def_id(trait_def_id).cloned()
        {
            if let Some(name) = &import_name {
                self.symbols.insert_trait(*name, trait_binding, span).ok();
            }
            // `from path import { items }` — also import traits
            if let Some(item_list) = items {
                for item in item_list {
                    let item_path = [*item];
                    if let Some(item_def_id) = self.symbols.lookup_trait_by_path(&item_path)
                        && let Some(item_binding) =
                            self.symbols.lookup_trait_by_def_id(item_def_id).cloned()
                    {
                        self.symbols.insert_trait(*item, item_binding, span).ok();
                    }
                }
            }
            return Ok(());
        }

        // Try as a function — single-segment only for now;
        // multi-segment function imports require module hierarchy support.
        if path.len() == 1
            && let Some(func_binding) = self.symbols.lookup_function(path[0]).cloned()
        {
            if let Some(name) = &import_name
                && let Err(diag) = self.symbols.insert_function(*name, func_binding, span)
            {
                self.diagnostics.push(diag);
            }
            return Ok(());
        }

        Err(Diagnostic::error(format!(
            "cannot resolve import `{}`",
            path.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("::"),
        ))
        .with_span(span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// Parse and resolve a Posita source, returning the resolver's symbol table.
    fn resolve_source(
        source: &str,
    ) -> Result<
        (
            SymbolTable<'static>,
            TraitEnv<'static>,
            ResolutionMap<'static>,
            TypeContext<'static>,
        ),
        Vec<String>,
    > {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let mut ctx = TypeContext::new();
        let mut parser = Parser::new(source, arena);
        let program = parser.parse_program().map_err(|diags| {
            diags
                .into_iter()
                .map(|d| d.message().to_string())
                .collect::<Vec<_>>()
        })?;
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, trait_env, diags, resolution_map) = resolver.resolve_program(&program);
        if diags.has_errors() {
            return Err(diags
                .into_inner()
                .into_iter()
                .map(|d| d.message().to_string())
                .collect::<Vec<_>>());
        }
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
        let func = symbols.lookup_function(Symbol::intern("main"));
        assert!(func.is_some(), "main should be registered");
    }

    #[test]
    fn test_resolve_type_def_struct() {
        let result = resolve_source("type Point = struct { x: Int<32>, y: Int<32> }");
        assert!(result.is_ok(), "struct type: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type(Symbol::intern("Point"));
        assert!(binding.is_some(), "Point should be registered");
        if let Some(b) = binding {
            assert_eq!(b.fields.len(), 2, "Point should have 2 fields");
        }
    }

    #[test]
    fn test_resolve_type_def_enum() {
        let result = resolve_source("type MyOption<T> = enum { None, Some(T) }");
        assert!(result.is_ok(), "enum type: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type(Symbol::intern("MyOption"));
        assert!(binding.is_some(), "MyOption should be registered");
        if let Some(b) = binding {
            assert_eq!(b.params.len(), 1, "MyOption should have 1 type param");
            assert_eq!(b.variants.len(), 2, "MyOption should have 2 variants");
        }
    }

    #[test]
    fn test_resolve_type_alias() {
        let result = resolve_source("type MyInt = Int<32>");
        assert!(result.is_ok(), "type alias: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type(Symbol::intern("MyInt"));
        assert!(binding.is_some(), "MyInt should be registered");
        assert!(
            binding.unwrap().alias_ast.is_some(),
            "MyInt should have an alias AST"
        );
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
        let binding = symbols.lookup_type(Symbol::intern("Wrapper"));
        assert!(
            binding.unwrap().transparent,
            "Wrapper should be transparent"
        );
    }

    #[test]
    fn test_resolve_layout_c_attr() {
        let result = resolve_source(
            "@layout(C)
             type CStruct = struct { x: Int<32> }",
        );
        assert!(result.is_ok(), "layout(C): {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let binding = symbols.lookup_type(Symbol::intern("CStruct"));
        assert!(binding.unwrap().c_layout, "CStruct should have c_layout");
    }

    #[test]
    fn test_resolve_generic_function() {
        let result = resolve_source("def id<T>(x: T) -> T { return x; }");
        assert!(result.is_ok(), "generic function: {:?}", result.err());
        let (symbols, _, _, _) = result.unwrap();
        let func = symbols.lookup_function(Symbol::intern("id"));
        assert!(func.is_some(), "id should be registered");
        assert!(
            !func.unwrap().signature.type_params.is_empty(),
            "id should have type params"
        );
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
