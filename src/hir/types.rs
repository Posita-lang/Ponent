use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeData {
    Int {
        bits: u8,
        signed: bool,
    },
    Float {
        bits: u8,
    },
    Bool,
    Char,
    Byte,
    USize,
    Struct {
        def_id: DefId,
        args: Vec<TypeId>,
    },
    Enum {
        def_id: DefId,
        args: Vec<TypeId>,
    },
    Tuple {
        elems: Vec<TypeId>,
    },
    Array {
        elem: TypeId,
        size: u64,
    },
    Slice {
        elem: TypeId,
    },
    Ref {
        ty: TypeId,
        mutable: bool,
    },
    Pointer {
        ty: TypeId,
    },
    Ptr {
        size: TypeId,
        pointee: TypeId,
    },
    Fn {
        params: Vec<TypeId>,
        ret: TypeId,
    },
    DynTrait {
        traits: Vec<DefId>,
    },
    Exists {
        name: String,
        base: TypeId,
        invariant: Expr,
    },
    Alias {
        def_id: DefId,
        args: Vec<TypeId>,
    },
    GenericParam {
        index: usize,
        name: String,
    },
    AssociatedType {
        trait_id: DefId,
        name: String,
        self_ty: TypeId,
    },
    Never,
    Unit,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub usize);

pub struct TypeContext {
    types: Vec<Arc<TypeData>>,
    type_map: HashMap<TypeData, TypeId>,
}

impl TypeContext {
    pub fn new() -> Self {
        let mut ctx = TypeContext {
            types: Vec::new(),
            type_map: HashMap::new(),
        };
        let unit = ctx.alloc(TypeData::Unit);
        let never = ctx.alloc(TypeData::Never);
        let error = ctx.alloc(TypeData::Error);
        let bool_ty = ctx.alloc(TypeData::Bool);
        let char_ty = ctx.alloc(TypeData::Char);
        let byte_ty = ctx.alloc(TypeData::Byte);
        let usize_ty = ctx.alloc(TypeData::USize);
        ctx
    }

    pub fn alloc(&mut self, data: TypeData) -> TypeId {
        if let Some(&id) = self.type_map.get(&data) {
            return id;
        }
        let id = TypeId(self.types.len());
        self.types.push(Arc::new(data.clone()));
        self.type_map.insert(data, id);
        id
    }

    pub fn get(&self, id: TypeId) -> &TypeData {
        &self.types[id.0]
    }

    pub fn int(&mut self, bits: u8, signed: bool) -> TypeId {
        self.alloc(TypeData::Int { bits, signed })
    }

    pub fn uint(&mut self, bits: u8) -> TypeId {
        self.int(bits, false)
    }

    pub fn float(&mut self, bits: u8) -> TypeId {
        self.alloc(TypeData::Float { bits })
    }

    pub fn bool(&self) -> TypeId {
        TypeId(3)
    }

    pub fn char(&self) -> TypeId {
        TypeId(4)
    }

    pub fn byte(&self) -> TypeId {
        TypeId(5)
    }

    pub fn usize(&self) -> TypeId {
        TypeId(6)
    }

    pub fn unit(&self) -> TypeId {
        TypeId(0)
    }

    pub fn never(&self) -> TypeId {
        TypeId(1)
    }

    pub fn error(&self) -> TypeId {
        TypeId(2)
    }

    pub fn struct_ty(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        self.alloc(TypeData::Struct { def_id, args })
    }

    pub fn enum_ty(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        self.alloc(TypeData::Enum { def_id, args })
    }

    pub fn tuple(&mut self, elems: Vec<TypeId>) -> TypeId {
        self.alloc(TypeData::Tuple { elems })
    }

    pub fn array(&mut self, elem: TypeId, size: u64) -> TypeId {
        self.alloc(TypeData::Array { elem, size })
    }

    pub fn slice(&mut self, elem: TypeId) -> TypeId {
        self.alloc(TypeData::Slice { elem })
    }

    pub fn reference(&mut self, ty: TypeId, mutable: bool) -> TypeId {
        self.alloc(TypeData::Ref { ty, mutable })
    }

    pub fn pointer(&mut self, ty: TypeId) -> TypeId {
        self.alloc(TypeData::Pointer { ty })
    }

    pub fn ptr(&mut self, size: TypeId, pointee: TypeId) -> TypeId {
        self.alloc(TypeData::Ptr { size, pointee })
    }

    pub fn function(&mut self, params: Vec<TypeId>, ret: TypeId) -> TypeId {
        self.alloc(TypeData::Fn { params, ret })
    }

    pub fn dyn_trait(&mut self, traits: Vec<DefId>) -> TypeId {
        self.alloc(TypeData::DynTrait { traits })
    }

    pub fn exists(&mut self, name: String, base: TypeId, invariant: Expr) -> TypeId {
        self.alloc(TypeData::Exists {
            name,
            base,
            invariant,
        })
    }

    pub fn alias(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        self.alloc(TypeData::Alias { def_id, args })
    }

    pub fn generic_param(&mut self, index: usize, name: String) -> TypeId {
        self.alloc(TypeData::GenericParam { index, name })
    }

    pub fn associated_type(&mut self, trait_id: DefId, name: String, self_ty: TypeId) -> TypeId {
        self.alloc(TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        })
    }

    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
        let data_a = self.get(a).clone();
        let data_b = self.get(b).clone();
        if data_a == data_b {
            return Ok(a);
        }
        if let TypeData::Alias { def_id, args } = &data_a {
            let expanded = self.expand_alias(*def_id, args.clone());
            return self.unify(expanded, b);
        }
        if let TypeData::Alias { def_id, args } = &data_b {
            let expanded = self.expand_alias(*def_id, args.clone());
            return self.unify(a, expanded);
        }
        match (&data_a, &data_b) {
            (TypeData::Error, _) => Ok(b),
            (_, TypeData::Error) => Ok(a),
            (
                TypeData::GenericParam { index: i1, .. },
                TypeData::GenericParam { index: i2, .. },
            ) if i1 == i2 => Ok(a),
            (TypeData::GenericParam { .. }, _) => {
                self.types[b.0] = Arc::new(data_a);
                self.type_map.insert(data_a, b);
                Ok(b)
            }
            (_, TypeData::GenericParam { .. }) => {
                self.types[a.0] = Arc::new(data_b);
                self.type_map.insert(data_b, a);
                Ok(a)
            }
            _ => Err(TypeError::Mismatch {
                expected: b,
                found: a,
                span: Span::new(0, 0),
            }),
        }
    }

    pub fn expand_alias(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        todo!("expand type alias")
    }

    pub fn subtype(&self, sub: TypeId, sup: TypeId) -> bool {
        if sub == sup {
            return true;
        }
        match (self.get(sub), self.get(sup)) {
            (TypeData::Error, _) => true,
            (_, TypeData::Error) => true,
            (TypeData::Never, _) => true,
            (TypeData::Alias { def_id, args }, _) => {
                let expanded = self.expand_alias_no_alloc(*def_id, args);
                self.subtype(expanded, sup)
            }
            (_, TypeData::Alias { def_id, args }) => {
                let expanded = self.expand_alias_no_alloc(*def_id, args);
                self.subtype(sub, expanded)
            }
            _ => false,
        }
    }

    fn expand_alias_no_alloc(&self, def_id: DefId, args: &[TypeId]) -> TypeId {
        todo!("expand type alias without allocation")
    }

    pub fn subst(&self, ty: TypeId, subst: &Subst) -> TypeId {
        match self.get(ty) {
            TypeData::GenericParam { index, .. } => subst.get(*index).copied().unwrap_or(ty),
            TypeData::Int { bits, signed } => TypeId(
                self.types
                    .iter()
                    .position(|t| {
                        **t == TypeData::Int {
                            bits: *bits,
                            signed: *signed,
                        }
                    })
                    .unwrap_or(usize::MAX),
            ),
            TypeData::Float { bits } => TypeId(
                self.types
                    .iter()
                    .position(|t| **t == TypeData::Float { bits: *bits })
                    .unwrap_or(usize::MAX),
            ),
            TypeData::Bool => ty,
            TypeData::Char => ty,
            TypeData::Byte => ty,
            TypeData::USize => ty,
            TypeData::Never => ty,
            TypeData::Unit => ty,
            TypeData::Error => ty,
            TypeData::Struct { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.struct_ty_no_alloc(*def_id, new_args)
            }
            TypeData::Enum { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.enum_ty_no_alloc(*def_id, new_args)
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems.iter().map(|&e| self.subst(e, subst)).collect();
                self.tuple_ty_no_alloc(new_elems)
            }
            TypeData::Array { elem, size } => {
                self.array_ty_no_alloc(self.subst(*elem, subst), *size)
            }
            TypeData::Slice { elem } => self.slice_ty_no_alloc(self.subst(*elem, subst)),
            TypeData::Ref { ty, mutable } => self.ref_ty_no_alloc(self.subst(*ty, subst), *mutable),
            TypeData::Pointer { ty } => self.pointer_ty_no_alloc(self.subst(*ty, subst)),
            TypeData::Ptr { size, pointee } => {
                self.ptr_ty_no_alloc(self.subst(*size, subst), self.subst(*pointee, subst))
            }
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> =
                    params.iter().map(|&p| self.subst(p, subst)).collect();
                self.fn_ty_no_alloc(new_params, self.subst(*ret, subst))
            }
            TypeData::DynTrait { traits } => ty,
            TypeData::Exists {
                name,
                base,
                invariant: _,
            } => self.exists_ty_no_alloc(name.clone(), self.subst(*base, subst)),
            TypeData::Alias { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.alias_ty_no_alloc(*def_id, new_args)
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => self.associated_ty_no_alloc(*trait_id, name.clone(), self.subst(*self_ty, subst)),
            _ => ty,
        }
    }

    fn struct_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Struct { def_id: d, args: a } = &**t {
                        *d == def_id && *a == args
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn enum_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Enum { def_id: d, args: a } = &**t {
                        *d == def_id && *a == args
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn tuple_ty_no_alloc(&self, elems: Vec<TypeId>) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Tuple { elems: e } = &**t {
                        *e == elems
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn array_ty_no_alloc(&self, elem: TypeId, size: u64) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Array { elem: e, size: s } = &**t {
                        *e == elem && *s == size
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn slice_ty_no_alloc(&self, elem: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Slice { elem: e } = &**t {
                        *e == elem
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn ref_ty_no_alloc(&self, ty: TypeId, mutable: bool) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Ref { ty: t, mutable: m } = &**t {
                        *t == ty && *m == mutable
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn pointer_ty_no_alloc(&self, ty: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Pointer { ty: t } = &**t {
                        *t == ty
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn ptr_ty_no_alloc(&self, size: TypeId, pointee: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Ptr {
                        size: s,
                        pointee: p,
                    } = &**t
                    {
                        *s == size && *p == pointee
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn fn_ty_no_alloc(&self, params: Vec<TypeId>, ret: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Fn { params: p, ret: r } = &**t {
                        *p == params && *r == ret
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn exists_ty_no_alloc(&self, name: String, base: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Exists {
                        name: n,
                        base: b,
                        invariant: _,
                    } = &**t
                    {
                        *n == name && *b == base
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn alias_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::Alias { def_id: d, args: a } = &**t {
                        *d == def_id && *a == args
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    fn associated_ty_no_alloc(&self, trait_id: DefId, name: String, self_ty: TypeId) -> TypeId {
        TypeId(
            self.types
                .iter()
                .position(|t| {
                    if let TypeData::AssociatedType {
                        trait_id: t_id,
                        name: n,
                        self_ty: s,
                    } = &**t
                    {
                        *t_id == trait_id && *n == name && *s == self_ty
                    } else {
                        false
                    }
                })
                .unwrap_or(usize::MAX),
        )
    }

    pub fn instantiate(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        todo!("instantiate generic type")
    }

    pub fn is_numeric(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. } => true,
            TypeData::Float { .. } => true,
            _ => false,
        }
    }

    pub fn is_integer(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. } => true,
            _ => false,
        }
    }

    pub fn is_unsigned(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { signed, .. } => !signed,
            TypeData::USize => true,
            _ => false,
        }
    }

    pub fn is_signed(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { signed, .. } => *signed,
            _ => false,
        }
    }

    pub fn is_float(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Float { .. } => true,
            _ => false,
        }
    }

    pub fn is_bool(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Bool)
    }

    pub fn is_char(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Char)
    }

    pub fn is_byte(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Byte)
    }

    pub fn is_usize(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::USize)
    }

    pub fn is_unit(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Unit)
    }

    pub fn is_never(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Never)
    }

    pub fn is_error(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Error)
    }

    pub fn is_reference(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Ref { .. })
    }

    pub fn is_pointer(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Pointer { .. })
    }

    pub fn is_struct(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Struct { .. })
    }

    pub fn is_enum(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Enum { .. })
    }

    pub fn is_tuple(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Tuple { .. })
    }

    pub fn is_array(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Array { .. })
    }

    pub fn is_slice(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Slice { .. })
    }

    pub fn is_fn(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Fn { .. })
    }

    pub fn is_dyn_trait(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::DynTrait { .. })
    }

    pub fn is_exists(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Exists { .. })
    }

    pub fn is_alias(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Alias { .. })
    }

    pub fn is_generic_param(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::GenericParam { .. })
    }

    pub fn is_associated_type(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::AssociatedType { .. })
    }

    pub fn bits_of_int(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Int { bits, .. } => Some(*bits),
            _ => None,
        }
    }

    pub fn signedness_of_int(&self, ty: TypeId) -> Option<bool> {
        match self.get(ty) {
            TypeData::Int { signed, .. } => Some(*signed),
            _ => None,
        }
    }

    pub fn bits_of_float(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Float { bits } => Some(*bits),
            _ => None,
        }
    }

    pub fn size_of_array(&self, ty: TypeId) -> Option<u64> {
        match self.get(ty) {
            TypeData::Array { size, .. } => Some(*size),
            _ => None,
        }
    }

    pub fn elem_of_array(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Array { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    pub fn elem_of_slice(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Slice { elem } => Some(*elem),
            _ => None,
        }
    }

    pub fn pointee_of_ref(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Ref { ty: t, .. } => Some(*t),
            _ => None,
        }
    }

    pub fn mutability_of_ref(&self, ty: TypeId) -> Option<bool> {
        match self.get(ty) {
            TypeData::Ref { mutable, .. } => Some(*mutable),
            _ => None,
        }
    }

    pub fn pointee_of_pointer(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Pointer { ty: t } => Some(*t),
            _ => None,
        }
    }

    pub fn params_of_fn(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.get(ty) {
            TypeData::Fn { params, .. } => Some(params),
            _ => None,
        }
    }

    pub fn ret_of_fn(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Fn { ret, .. } => Some(*ret),
            _ => None,
        }
    }

    pub fn tuple_elems(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.get(ty) {
            TypeData::Tuple { elems } => Some(elems),
            _ => None,
        }
    }

    pub fn base_of_exists(&self, ty: TypeId) -> Option<TypeId> {
        match self.get(ty) {
            TypeData::Exists { base, .. } => Some(*base),
            _ => None,
        }
    }

    pub fn name_of_exists(&self, ty: TypeId) -> Option<&String> {
        match self.get(ty) {
            TypeData::Exists { name, .. } => Some(name),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Subst {
    map: HashMap<usize, TypeId>,
}

impl Subst {
    pub fn new() -> Self {
        Subst {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, index: usize, ty: TypeId) {
        self.map.insert(index, ty);
    }

    pub fn get(&self, index: usize) -> Option<&TypeId> {
        self.map.get(&index)
    }

    pub fn extend(&mut self, other: &Subst) {
        for (&k, &v) in other.map.iter() {
            self.map.insert(k, v);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Default for Subst {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch {
        expected: TypeId,
        found: TypeId,
        span: Span,
    },
    UndefinedName {
        name: String,
        span: Span,
        suggestions: Vec<String>,
    },
    TypeNotFound {
        name: String,
        span: Span,
    },
    CannotInfer {
        span: Span,
    },
    GenericArgumentCount {
        expected: usize,
        found: usize,
        span: Span,
    },
    TraitNotImplemented {
        ty: TypeId,
        trait_name: String,
        span: Span,
    },
    InvariantViolation {
        ty: TypeId,
        expr: String,
        span: Span,
    },
    MutableBorrow {
        span: Span,
    },
    ImmutableBorrow {
        span: Span,
    },
    OutOfBounds {
        index: u64,
        size: u64,
        span: Span,
    },
    DivisionByZero {
        span: Span,
    },
    Overflow {
        span: Span,
    },
    NeverType {
        span: Span,
    },
    CircularDependency {
        name: String,
        span: Span,
    },
    DuplicateDefinition {
        name: String,
        span: Span,
        previous: Span,
    },
    PrivateField {
        name: String,
        span: Span,
    },
    PrivateType {
        name: String,
        span: Span,
    },
    PrivateFunction {
        name: String,
        span: Span,
    },
    PatternNotExhaustive {
        span: Span,
    },
    PatternRedundant {
        span: Span,
    },
    PatternTypeMismatch {
        expected: TypeId,
        found: TypeId,
        span: Span,
    },
}

use crate::ast::Expr;
use crate::ast::Span;
