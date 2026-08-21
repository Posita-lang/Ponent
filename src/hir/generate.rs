use crate::ast::*;
use crate::hir::symbol::SymbolTable;
use crate::hir::types::{TypeContext, TypeData, TypeId};
use std::borrow::Cow;

/// Information about a type's structure, computed from a `TypeId` at compile time.
/// Equivalent to Zig's `@typeInfo` result.
///
/// String fields use `Cow<'input, str>` so that primitive type names
/// (`Int<32>`, `Bool`, `()`, …) can be returned as `Cow::Borrowed` from
/// static tables without allocating.  ADT-derived names (struct/enum/field/
/// variant names) still go through `Symbol::as_str() -> String` and are
/// stored as `Cow::Owned`, since the interner does not expose a
/// lifetime-erased `&'input str` view.
#[derive(Debug, Clone)]
pub struct TypeInfo<'input> {
    pub name: Cow<'input, str>,
    pub params: Vec<Cow<'input, str>>,
    pub fields: Vec<FieldInfo<'input>>,
    pub variants: Vec<VariantInfo<'input>>,
    pub kind: TypeKind,
    /// For Int<Bits> / UInt<Bits>, the bit width.
    pub bits: Option<u32>,
    /// For Float<Bits>, the bit width.
    pub float_bits: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct FieldInfo<'input> {
    pub name: Cow<'input, str>,
    pub ty: TypeId,
}

#[derive(Debug, Clone)]
pub struct VariantInfo<'input> {
    pub name: Cow<'input, str>,
    pub payload: Vec<FieldInfo<'input>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Struct,
    Enum,
    /// A primitive type such as Int, Bool, etc.
    Primitive,
    /// A type alias or other opaque type.
    Other,
}

/// The `generate` block expansion engine.
///
/// Operates on the AST **before** name resolution (Phase 1), and again
/// **after** name resolution (Phase 2) when the `SymbolTable<'input>` is available.
///
/// Pipeline: parse → Phase 1 expand → resolve → Phase 2 expand → type check.
pub struct GenerateExpander<'a, 'input> {
    ctx: &'a mut TypeContext<'input>,
    symbols: Option<&'a SymbolTable<'input>>,
}

/// A value produced by evaluating a generate-block condition.
/// Conditions are not always boolean — `@typeInfo!(T).name` evaluates
/// to a string, and `@typeInfo!(T).fields'len` evaluates to an integer.
/// `BinOp::Eq`/`BinOp::Neq` compare these values directly rather than
/// first coercing everything to bool (which would lose information).
#[derive(Debug, Clone, PartialEq, Eq)]
enum CondValue<'input> {
    Bool(bool),
    Int(i128),
    Str(Cow<'input, str>),
}

impl<'input> CondValue<'input> {
    fn truthy(&self) -> bool {
        match self {
            CondValue::Bool(b) => *b,
            CondValue::Int(n) => *n != 0,
            CondValue::Str(s) => !s.is_empty(),
        }
    }

    fn from_bool(b: bool) -> Self {
        CondValue::Bool(b)
    }
}

impl<'a, 'input> GenerateExpander<'a, 'input> {
    /// Create a new expander for Phase 1 (before name resolution).
    /// The `SymbolTable<'input>` is not yet available, so `@typeInfo!` evaluations
    /// will return placeholder data.
    pub fn new_phase1(ctx: &'a mut TypeContext<'input>) -> Self {
        GenerateExpander { ctx, symbols: None }
    }

    /// Create a new expander for Phase 2 (after name resolution).
    /// The `SymbolTable<'input>` is available, so `@typeInfo!` evaluations return
    /// real type information.
    pub fn new_phase2(ctx: &'a mut TypeContext<'input>, symbols: &'a SymbolTable<'input>) -> Self {
        GenerateExpander {
            ctx,
            symbols: Some(symbols),
        }
    }

    /// Expand all `Generate` blocks in a list of AST statements.
    /// Called after parsing, before name resolution (Phase 1).
    /// In Phase 1, the body is simply passed through since type info
    /// is not yet available.  Conditional generation and name-mapped
    /// templates are expanded in Phase 2 (after name resolution).
    pub fn expand_program(
        &mut self,
        items: Vec<Stmt<'input>>,
    ) -> (Vec<Stmt<'input>>, Vec<Stmt<'input>>) {
        let mut result = Vec::new();
        let mut new_items = Vec::new();
        for item in items {
            match item {
                Stmt::Generate {
                    attributes,
                    for_type,
                    body,
                    span,
                } => {
                    let before = result.len();
                    if self.symbols.is_some() {
                        // Phase 2: full expansion with @typeInfo! support.
                        self.expand_generate_block(&for_type, body, span, &mut result);
                    } else {
                        // Phase 1: preserve the Generate node so Phase 2 can
                        // evaluate @typeInfo! conditions and expand it properly.
                        result.push(Stmt::Generate {
                            attributes,
                            for_type,
                            body,
                            span,
                        });
                    }
                    // Track newly generated items so resolve_incremental can
                    // process only these (a clone — the originals stay in result
                    // for the final program returned to the caller).
                    new_items.extend(result[before..].iter().cloned());
                }
                _ => result.push(item),
            }
        }
        (result, new_items)
    }

    /// Expand a single `generate for <Type> { ... }` block using the
    /// available `SymbolTable<'input>` (Phase 2).  Evaluates `@typeInfo!`
    /// expressions, processes conditional generation, and expands
    /// name-mapped templates.
    fn expand_generate_block(
        &mut self,
        for_type: &Type<'input>,
        body: Vec<Stmt<'input>>,
        _span: Span,
        result: &mut Vec<Stmt<'input>>,
    ) {
        let symbols = match self.symbols {
            Some(s) => s,
            None => {
                // Phase 1 fallback: just pass through.
                result.extend(body);
                return;
            }
        };

        // Resolve the `for_type` to a TypeId so we can call @typeInfo! on it.
        // We need to resolve the AST type to a TypeId.  Use the resolve_type
        // helper which tries to look up the type in the symbol table.
        let for_type_id = self.resolve_type_for_generate(&for_type, symbols);

        // Walk the body, evaluating @typeInfo! expressions and conditions.
        let expanded = self.expand_generate_body(&body, for_type_id, symbols);
        result.extend(expanded);
    }

    /// Walk a `generate` block body, expanding @typeInfo! expressions,
    /// evaluating `if` conditions, and expanding name-mapped templates.
    fn expand_generate_body(
        &mut self,
        stmts: &[Stmt<'input>],
        for_type_id: Option<TypeId>,
        symbols: &SymbolTable<'input>,
    ) -> Vec<Stmt<'input>> {
        let mut result = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::If {
                    cond,
                    then_branch,
                    else_branch,
                    span,
                    ..
                } => {
                    // Evaluate the condition.
                    let cond_true = self
                        .eval_generate_condition(cond, for_type_id, symbols)
                        .truthy();
                    if cond_true {
                        // Expand the then-branch recursively.
                        result.extend(self.expand_generate_body(then_branch, for_type_id, symbols));
                    } else if let Some(else_branch) = else_branch {
                        result.extend(self.expand_generate_body(else_branch, for_type_id, symbols));
                    }
                    // If condition is false and no else branch, emit nothing.
                }
                _ => {
                    // For non-conditional statements, emit them as-is.
                    result.push(stmt.clone());
                }
            }
        }
        result
    }

    /// Evaluate a condition expression in a `generate` block.
    /// Returns a `CondValue` (bool, int, or string) for use in comparisons.
    fn eval_generate_condition(
        &mut self,
        cond: &Expr<'input>,
        for_type_id: Option<TypeId>,
        _symbols: &SymbolTable<'input>,
    ) -> CondValue<'input> {
        match cond {
            Expr::BinaryOp {
                left, op, right, ..
            } => {
                let l = self.eval_generate_condition(left, for_type_id, _symbols);
                let r = self.eval_generate_condition(right, for_type_id, _symbols);
                match op {
                    BinOp::And => CondValue::Bool(l.truthy() && r.truthy()),
                    BinOp::Or => CondValue::Bool(l.truthy() || r.truthy()),
                    BinOp::Eq => CondValue::Bool(l == r),
                    BinOp::Neq => CondValue::Bool(l != r),
                    _ => CondValue::Bool(false),
                }
            }
            Expr::UnaryOp { op, expr, .. } => {
                let v = self.eval_generate_condition(expr, for_type_id, _symbols);
                match op {
                    UnaryOp::Not => CondValue::Bool(!v.truthy()),
                    _ => CondValue::Bool(false),
                }
            }
            Expr::TypeInfo(ty, _) => {
                // @typeInfo!(T) — A type info is truthy if the type exists.
                if self.resolve_type_ast(ty, _symbols).is_some() {
                    CondValue::Bool(true)
                } else {
                    CondValue::Bool(false)
                }
            }
            Expr::FieldAccess { base, field, .. } => {
                // Evaluate field access for @typeInfo!(T).field patterns.
                // `'len` (compile-time property) is parsed as AttrAccess,
                // `.len` (runtime syntax) is parsed as FieldAccess.
                if field.eq_str("fields.len") {
                    // `@typeInfo!(T).fields.len` — runtime syntax, rejected.
                    return CondValue::Bool(false);
                }
                if field.eq_str("variants.len") {
                    // `@typeInfo!(T).variants.len` — runtime syntax, rejected.
                    return CondValue::Bool(false);
                }
                if field.eq_str("name")
                    && let Expr::TypeInfo(ty, _) = base
                    && let Some(ty_id) = self.resolve_type_ast(ty, _symbols)
                {
                    return CondValue::Str(type_name(self.ctx, ty_id, Some(_symbols)));
                }
                // Unknown field access — return empty string (falsy).
                CondValue::Str(Cow::Borrowed(""))
            }
            Expr::AttrAccess { base, attr, .. } => {
                // `@typeInfo!(T).fields'len` is parsed as:
                //   AttrAccess { base: FieldAccess { base: TypeInfo, field: "fields" }, attr: "len" }
                // Same for `variants'len`.
                if attr.eq_str("len")
                    && let Expr::FieldAccess {
                        base: inner_base,
                        field,
                        ..
                    } = base
                {
                    if field.eq_str("fields")
                        && let Expr::TypeInfo(ty, _) = inner_base
                        && let Some(ty_id) = self.resolve_type_ast(ty, _symbols)
                    {
                        let info = crate::hir::generate::get_type_info(self.ctx, _symbols, ty_id);
                        return CondValue::Int(info.fields.len() as i128);
                    } else if field.eq_str("variants")
                        && let Expr::TypeInfo(ty, _) = inner_base
                        && let Some(ty_id) = self.resolve_type_ast(ty, _symbols)
                    {
                        let info = crate::hir::generate::get_type_info(self.ctx, _symbols, ty_id);
                        return CondValue::Int(info.variants.len() as i128);
                    }
                }
                CondValue::Str(Cow::Borrowed(""))
            }
            Expr::Literal(Literal::Bool(b), _) => CondValue::Bool(*b),
            Expr::Literal(Literal::Int(n), _) => CondValue::Int(
                n.to_i128()
                    .unwrap_or_else(|| if *n < 0 { i128::MIN } else { i128::MAX }),
            ),
            // `Literal::String(String)` — the AST owns the storage, so we must
            // clone into a `Cow::Owned`.  (If the AST ever switches to
            // `Literal::String(&'input str)`, this becomes `Cow::Borrowed(*s)`.)
            Expr::Literal(Literal::String(s), _) => CondValue::Str(Cow::Owned(s.clone())),
            _ => CondValue::Bool(false),
        }
    }

    /// Resolve an AST type to a TypeId using the symbol table.
    /// This is a simplified resolver that handles basic type paths.
    fn resolve_type_for_generate(
        &mut self,
        ty: &Type<'input>,
        symbols: &SymbolTable<'input>,
    ) -> Option<TypeId> {
        self.resolve_type_ast(ty, symbols)
    }

    /// Resolve an AST type to a TypeId.
    fn resolve_type_ast(
        &mut self,
        ty: &Type<'input>,
        symbols: &SymbolTable<'input>,
    ) -> Option<TypeId> {
        match ty {
            Type::Path(path, _) => {
                // Single-segment path: look up by name.
                if path.len() == 1 {
                    let def_id = symbols.lookup_type_by_path(path)?;
                    self.ctx.get_type_id_for_def_id(def_id)
                } else {
                    None
                }
            }
            Type::Generic(base, args, _) => {
                // Handle generic types like `Option<Int<32>>`.
                // Resolve the base type to get its DefId.
                let base_path = match base {
                    Type::Path(p, _) => p.clone(),
                    _ => return None,
                };
                let def_id = symbols.lookup_type_by_path(&base_path)?;
                // Resolve each generic argument.
                let arg_ids: Vec<TypeId> = args
                    .iter()
                    .filter_map(|arg| match arg {
                        GenericArg::Positional(t) => self.resolve_type_ast(t, symbols),
                        _ => None,
                    })
                    .collect();
                // Determine if the base is a struct or enum.
                let binding = symbols.lookup_type_by_def_id(def_id)?;
                match binding.kind {
                    crate::hir::symbol::TypeKind::Struct => {
                        Some(self.ctx.struct_ty(def_id, arg_ids))
                    }
                    crate::hir::symbol::TypeKind::Enum => Some(self.ctx.enum_ty(def_id, arg_ids)),
                    _ => self.ctx.get_type_id_for_def_id(def_id),
                }
            }
            _ => None,
        }
    }
}

/// Compute the full `TypeInfo` for a resolved `TypeId`.
///
/// Uses the `TypeContext<'input>` for structural type data and the `SymbolTable<'input>`
/// to look up `TypeBinding<'input>` (field lists, enum variants, generic params).
pub fn get_type_info<'input>(
    ctx: &mut TypeContext<'input>,
    symbols: &SymbolTable<'input>,
    ty: TypeId,
) -> TypeInfo<'input> {
    let resolved = ctx.resolve_binding(ty);
    let name = type_name(ctx, resolved, Some(symbols));
    let kind = determine_kind(ctx, resolved);

    // Collect fields, variants, params from the TypeBinding<'input> (if ADT).
    let mut fields: Vec<FieldInfo<'input>> = Vec::new();
    let mut variants: Vec<VariantInfo<'input>> = Vec::new();
    let mut params: Vec<Cow<'input, str>> = Vec::new();
    let mut bits: Option<u32> = None;
    let mut float_bits: Option<u32> = None;

    match ctx.get(resolved) {
        TypeData::Adt { def_id, .. } => {
            if let Some(binding) = symbols.lookup_type_by_def_id(*def_id) {
                // Generic parameters.  `Symbol::as_str()` returns `String`
                // (the interner does not expose a `&'input str` view), so we
                // store these as `Cow::Owned`.
                for p in &binding.params {
                    params.push(Cow::Owned(p.name.as_str()));
                }
                // Struct fields
                for f in &binding.fields {
                    fields.push(FieldInfo {
                        name: Cow::Owned(f.name.as_str()),
                        ty: f.ty,
                    });
                }
                // Enum variants
                for v in &binding.variants {
                    let payload = match &v.payload {
                        Some(ty) => {
                            // Collect fields from the payload type (usually a tuple type).
                            tuple_fields(ctx, ty, symbols)
                        }
                        None => Vec::new(),
                    };
                    variants.push(VariantInfo {
                        name: Cow::Owned(v.name.as_str()),
                        payload,
                    });
                }
            }
        }
        TypeData::Int { bits: b, .. } | TypeData::UInt { bits: b, .. } => {
            bits = Some(*b);
        }
        TypeData::Float { bits: b } => {
            float_bits = Some(*b);
        }
        _ => {}
    }

    TypeInfo {
        name,
        params,
        fields,
        variants,
        kind,
        bits,
        float_bits,
    }
}

/// Pre-computed type names for common integer/float widths (8, 16, 32, 64, 128).
/// Avoids `format!` allocations for the ~95% common case; custom widths fall back.
const INT_TYPE_NAMES: [&str; 5] = ["Int<8>", "Int<16>", "Int<32>", "Int<64>", "Int<128>"];
const UINT_TYPE_NAMES: [&str; 5] = ["UInt<8>", "UInt<16>", "UInt<32>", "UInt<64>", "UInt<128>"];
/// Pre-computed type names for IEEE 754 float widths (32, 64).
/// Float<8>, Float<16>, Float<128> are not valid IEEE 754 types
/// and are rejected by the type checker.  Custom widths fall back to format!.
const FLOAT_TYPE_NAMES: [&str; 2] = ["Float<32>", "Float<64>"];

fn int_type_name<'input>(bits: u32, signed: bool) -> Cow<'input, str> {
    let idx = match bits {
        8 => Some(0),
        16 => Some(1),
        32 => Some(2),
        64 => Some(3),
        128 => Some(4),
        _ => None,
    };
    if let Some(i) = idx {
        Cow::Borrowed(if signed {
            INT_TYPE_NAMES[i]
        } else {
            UINT_TYPE_NAMES[i]
        })
    } else if signed {
        Cow::Owned(format!("Int<{}>", bits))
    } else {
        Cow::Owned(format!("UInt<{}>", bits))
    }
}

fn float_type_name<'input>(bits: u32) -> Cow<'input, str> {
    // FLOAT_TYPE_NAMES only caches the 32-bit and 64-bit widths of the IEEE 754 standard.
    match bits {
        32 => Cow::Borrowed(FLOAT_TYPE_NAMES[0]),
        64 => Cow::Borrowed(FLOAT_TYPE_NAMES[1]),
        _ => Cow::Owned(format!("Float<{}>", bits)),
    }
}

/// Determine a human-readable name for a type.
/// Takes a pre-resolved `TypeId` to avoid redundant `resolve_binding` calls.
/// When `symbols` is `Some`, ADT types (structs, enums) are rendered with
/// their actual name (e.g. `MyStruct`) instead of a generic `DefId(N)`.
fn type_name<'input>(
    ctx: &TypeContext<'input>,
    resolved: TypeId,
    symbols: Option<&SymbolTable<'input>>,
) -> Cow<'input, str> {
    match ctx.get(resolved) {
        TypeData::Int { bits, signed, .. } => int_type_name(*bits, *signed),
        TypeData::UInt { bits, .. } => int_type_name(*bits, false),
        TypeData::Float { bits } => float_type_name(*bits),

        TypeData::Bool => Cow::Borrowed("Bool"),
        TypeData::Char => Cow::Borrowed("Char"),
        TypeData::Byte => Cow::Borrowed("Byte"),
        TypeData::USize => Cow::Borrowed("usize"),
        TypeData::Never => Cow::Borrowed("!"),
        TypeData::Unit => Cow::Borrowed("()"),

        TypeData::Adt { def_id, .. } => {
            if let Some(symbols) = symbols {
                if let Some(name) = symbols.type_name_by_def_id(*def_id) {
                    // `Symbol::as_str()` returns `String` (the interner does
                    // not expose a `&'input str` view), so we must own it.
                    return Cow::Owned(name.as_str());
                }
            }
            Cow::Owned(format!("{:?}", def_id))
        }

        _ => Cow::Owned(format!("{:?}", ctx.get(resolved))),
    }
}

/// Determine the `TypeKind` for a type.
/// Takes a pre-resolved `TypeId` to avoid redundant `resolve_binding` calls.
fn determine_kind<'input>(ctx: &TypeContext<'input>, resolved: TypeId) -> TypeKind {
    match ctx.get(resolved) {
        TypeData::Adt { kind: adt_kind, .. } => match adt_kind {
            crate::hir::types::AdtKind::Struct => TypeKind::Struct,
            crate::hir::types::AdtKind::Enum => TypeKind::Enum,
        },
        TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::Float { .. } | TypeData::Bool => {
            TypeKind::Primitive
        }
        _ => TypeKind::Other,
    }
}

/// Extract tuple fields from a resolution-phase `Type`.
fn tuple_fields<'input>(
    ctx: &mut TypeContext<'input>,
    ty: &Type<'input>,
    symbols: &SymbolTable<'input>,
) -> Vec<FieldInfo<'input>> {
    match ty {
        Type::Tuple(tys, _) => tys
            .iter()
            .enumerate()
            .map(|(i, elem_ty)| FieldInfo {
                name: Cow::Owned(format!("_{}", i)),
                ty: resolve_ast_type_to_typeid(ctx, elem_ty, symbols),
            })
            .collect(),
        // Single unnamed field (no tuple syntax)
        _ => vec![FieldInfo {
            name: Cow::Borrowed("value"),
            ty: resolve_ast_type_to_typeid(ctx, ty, symbols),
        }],
    }
}

/// Resolve an AST `Type` to a `TypeId` using the symbol table.
/// Falls back to `ctx.error()` if resolution fails.
fn resolve_ast_type_to_typeid<'input>(
    ctx: &mut TypeContext<'input>,
    ty: &Type<'input>,
    symbols: &SymbolTable<'input>,
) -> TypeId {
    match ty {
        Type::Path(path, _) => {
            if let Some(def_id) = symbols.lookup_type_by_path(path)
                && let Some(ty_id) = ctx.get_type_id_for_def_id(def_id)
            {
                return ty_id;
            }
            // Try to match known type names (bare names without generic args).
            if path.len() == 1 {
                let name = path[0].as_str();
                match name.as_str() {
                    // Bare `Int` defaults to `Int<32>`.  If `Int` is bound to a
                    // generic parameter in the current scope, `lookup_type_by_path`
                    // above will resolve it first, so this fallback only triggers
                    // when `Int` is truly unbound.  This matches SYNTAX.md: "a bare
                    // `Int` is equivalent to `Int<32>`".
                    "Int" => return ctx.int(32, true),
                    "UInt" => return ctx.uint(32),
                    "Float" => return ctx.float(64),
                    "Bool" => return ctx.bool(),
                    "String" => return ctx.builtin_str,
                    _ => {}
                }
            }
            ctx.error()
        }
        Type::Generic(base, args, _) => {
            // Handle Int<N>, UInt<N>, Float<N>.
            if let Type::Path(path, _) = base
                && path.len() == 1
            {
                let name = path[0].as_str();
                let bits = args.first().and_then(|arg| {
                    if let GenericArg::Positional(Type::Literal(expr, _)) = arg
                        && let Expr::Literal(Literal::Int(bits), _) = expr
                    {
                        return bits.to_u64().and_then(|n| u32::try_from(n).ok());
                    }
                    None
                });
                if let Some(bits) = bits {
                    match name.as_str() {
                        "Int" if bits >= 1 && bits <= 64 => return ctx.int(bits, true),
                        "Int" => return ctx.error(), // out-of-range Int rejected
                        "UInt" if bits >= 1 && bits <= 64 => return ctx.uint(bits),
                        "UInt" => return ctx.error(), // out-of-range UInt rejected
                        "Float" if bits == 32 || bits == 64 => return ctx.float(bits),
                        "Float" => return ctx.error(), // non-32/64 Float rejected
                        _ => {}
                    }
                }
            }
            // Fallback: resolve the base type (ignoring args).
            resolve_ast_type_to_typeid(ctx, base, symbols)
        }
        Type::Reference {
            inner,
            mutable,
            lifetime,
            ..
        } => {
            let inner_id = resolve_ast_type_to_typeid(ctx, inner, symbols);
            ctx.reference_with_lifetime(inner_id, *mutable, *lifetime)
        }
        _ => ctx.error(),
    }
}
