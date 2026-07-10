use crate::ast::*;
use crate::hir::types::{TypeContext, TypeData, TypeId};

/// Information about a type's structure, computed from a `TypeId` at compile time.
/// Equivalent to Zig's `@typeInfo` result.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: String,
    pub params: Vec<String>,
    pub fields: Vec<FieldInfo>,
    pub kind: TypeKind,
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub ty: TypeId,
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
/// Operates on the AST **before** name resolution.  Structural type
/// information (field lists, parameter names) will be embedded in
/// `TypeContext` in a future change; for now `get_type_info` returns
/// placeholder data and the expander simply passes the body through.
///
/// Pipeline: parse → expand generate blocks → resolve → type check.
pub struct GenerateExpander<'a> {
    ctx: &'a mut TypeContext,
}

impl<'a> GenerateExpander<'a> {
    pub fn new(ctx: &'a mut TypeContext) -> Self {
        GenerateExpander { ctx }
    }

    /// Expand all `Generate` blocks in a list of AST statements.
    /// Called after parsing, before name resolution.
    pub fn expand_program(&self, items: Vec<Stmt>) -> Vec<Stmt> {
        let mut result = Vec::new();
        for item in items {
            match item {
                Stmt::Generate { body, .. } => {
                    // TODO: implement actual expansion:
                    // 1. Evaluate @typeInfo!(T) for the for_type
                    // 2. Unroll declarative loops
                    // 3. Expand name-mapped templates
                    // 4. Log to audit
                    result.extend(body);
                }
                _ => result.push(item),
            }
        }
        result
    }

    /// Get type info for a given type at compile time.
    /// This is the implementation of `@typeInfo!(Type)`.
    /// Returns placeholder data until structural type information is
    /// embedded in `TypeContext` (see `collect_fields` / `collect_params`).
    pub fn get_type_info(&self, ty: TypeId) -> TypeInfo {
        let name = format!("{:?}", self.ctx.get(ty));
        let kind = match self.ctx.get(ty) {
            TypeData::Adt { .. } => {
                if self.ctx.is_struct(ty) {
                    TypeKind::Struct
                } else if self.ctx.is_enum(ty) {
                    TypeKind::Enum
                } else {
                    TypeKind::Other
                }
            }
            TypeData::Int { .. } | TypeData::Float { .. } | TypeData::Bool => TypeKind::Primitive,
            _ => TypeKind::Other,
        };
        // TODO: collect real fields and params from TypeContext once
        // structural type information is available pre-resolution.
        TypeInfo {
            name,
            params: Vec::new(),
            fields: Vec::new(),
            kind,
        }
    }
}