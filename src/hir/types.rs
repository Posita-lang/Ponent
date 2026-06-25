use std::collections::{HashMap, HashSet};
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
    bindings: HashMap<TypeId, TypeId>,
    invariants: HashMap<TypeId, Expr>,
    alias_cache: HashMap<(DefId, Vec<TypeId>), TypeId>,
    instantiate_cache: HashMap<(DefId, Vec<TypeId>), TypeId>,
    symbol_provider: Option<Box<dyn SymbolProvider>>,
    expanding: HashSet<DefId>,
}

#[derive(Debug, Clone)]
pub enum AliasError {
    NotFound(DefId),
    ParameterCountMismatch {
        expected: usize,
        found: usize,
        def_id: DefId,
    },
    Recursive(DefId),
    SymbolProviderNotSet,
}

pub trait SymbolProvider {
    fn get_alias_body(&self, def_id: DefId) -> Option<(Vec<TypeParam>, TypeId)>;
    fn get_struct_definition(&self, def_id: DefId) -> Option<(Vec<TypeParam>, Vec<StructField>)>;
    fn get_enum_definition(&self, def_id: DefId) -> Option<(Vec<TypeParam>, Vec<EnumVariant>)>;
    fn is_type_alias(&self, def_id: DefId) -> bool;
}

struct ExpandingGuard<'a> {
    ctx: &'a mut TypeContext,
    def_id: DefId,
    removed: bool,
}

impl<'a> ExpandingGuard<'a> {
    fn new(ctx: &'a mut TypeContext, def_id: DefId) -> Self {
        ctx.expanding.insert(def_id);
        ExpandingGuard {
            ctx,
            def_id,
            removed: false,
        }
    }
}

impl<'a> Drop for ExpandingGuard<'a> {
    fn drop(&mut self) {
        if !self.removed {
            self.ctx.expanding.remove(&self.def_id);
            self.removed = true;
        }
    }
}

impl TypeContext {
    pub fn new() -> Self {
        let mut ctx = TypeContext {
            types: Vec::new(),
            type_map: HashMap::new(),
            bindings: HashMap::new(),
            invariants: HashMap::new(),
            alias_cache: HashMap::new(),
            instantiate_cache: HashMap::new(),
            symbol_provider: None,
            expanding: HashSet::new(),
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

    pub fn set_symbol_provider(&mut self, provider: impl SymbolProvider + 'static) {
        self.symbol_provider = Some(Box::new(provider));
    }

    pub fn get_invariant(&self, id: TypeId) -> Option<&Expr> {
        self.invariants.get(&id)
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

    pub fn get(&mut self, id: TypeId) -> &TypeData {
        self.resolve_binding(id)
    }

    fn resolve_binding(&mut self, id: TypeId) -> &TypeData {
        if let Some(&bound) = self.bindings.get(&id) {
            let root = self.resolve_binding(bound);
            if root != bound {
                self.bindings.insert(id, root);
            }
            root
        } else {
            &self.types[id.0]
        }
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
        let id = self.alloc(TypeData::Exists { name, base });
        self.invariants.insert(id, invariant);
        id
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

    fn occurs_check(&mut self, param: TypeId, ty: TypeId) -> bool {
        if param == ty {
            return true;
        }
        match self.resolve_binding(ty) {
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                args.iter().any(|&a| self.occurs_check(param, a))
            }
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.occurs_check(param, e)),
            TypeData::Array { elem, .. } => self.occurs_check(param, *elem),
            TypeData::Slice { elem } => self.occurs_check(param, *elem),
            TypeData::Ref { ty, .. } => self.occurs_check(param, *ty),
            TypeData::Pointer { ty } => self.occurs_check(param, *ty),
            TypeData::Ptr { size, pointee } => {
                self.occurs_check(param, *size) || self.occurs_check(param, *pointee)
            }
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.occurs_check(param, p))
                    || self.occurs_check(param, *ret)
            }
            TypeData::Alias { args, .. } => args.iter().any(|&a| self.occurs_check(param, a)),
            TypeData::Exists { base, .. } => self.occurs_check(param, *base),
            TypeData::AssociatedType { self_ty, .. } => self.occurs_check(param, *self_ty),
            TypeData::GenericParam { .. } => false,
            TypeData::Int { .. }
            | TypeData::Float { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::DynTrait { .. } => false,
        }
    }

    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
        let data_a = self.resolve_binding(a).clone();
        let data_b = self.resolve_binding(b).clone();

        if data_a == data_b {
            return Ok(a);
        }

        if let TypeData::Alias { def_id, args } = &data_a {
            match self.expand_alias_def(*def_id, args.clone()) {
                Ok(expanded) => return self.unify(expanded, b),
                Err(AliasError::NotFound(_)) => {
                    return Err(TypeError::TypeNotFound {
                        name: format!("alias {:?}", def_id),
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::ParameterCountMismatch {
                    expected,
                    found,
                    def_id,
                }) => {
                    return Err(TypeError::GenericArgumentCount {
                        expected,
                        found,
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::Recursive(def_id)) => {
                    return Err(TypeError::CircularDependency {
                        name: format!("alias {:?}", def_id),
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::SymbolProviderNotSet) => {
                    return Err(TypeError::TypeNotFound {
                        name: format!("alias {:?} (symbol provider not set)", def_id),
                        span: Span::new(0, 0),
                    });
                }
            }
        }

        if let TypeData::Alias { def_id, args } = &data_b {
            match self.expand_alias_def(*def_id, args.clone()) {
                Ok(expanded) => return self.unify(a, expanded),
                Err(AliasError::NotFound(_)) => {
                    return Err(TypeError::TypeNotFound {
                        name: format!("alias {:?}", def_id),
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::ParameterCountMismatch {
                    expected,
                    found,
                    def_id,
                }) => {
                    return Err(TypeError::GenericArgumentCount {
                        expected,
                        found,
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::Recursive(def_id)) => {
                    return Err(TypeError::CircularDependency {
                        name: format!("alias {:?}", def_id),
                        span: Span::new(0, 0),
                    });
                }
                Err(AliasError::SymbolProviderNotSet) => {
                    return Err(TypeError::TypeNotFound {
                        name: format!("alias {:?} (symbol provider not set)", def_id),
                        span: Span::new(0, 0),
                    });
                }
            }
        }

        match (&data_a, &data_b) {
            (TypeData::Error, _) => Ok(b),
            (_, TypeData::Error) => Ok(a),
            (
                TypeData::GenericParam { index: i1, .. },
                TypeData::GenericParam { index: i2, .. },
            ) if i1 == i2 => Ok(a),
            (
                TypeData::GenericParam {
                    index: i1,
                    name: n1,
                },
                _,
            ) => {
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: Span::new(0, 0),
                    });
                }
                self.bindings.insert(a, b);
                self.type_map.remove(&TypeData::GenericParam {
                    index: *i1,
                    name: n1.clone(),
                });
                self.alias_cache.clear();
                self.instantiate_cache.clear();
                Ok(b)
            }
            (
                _,
                TypeData::GenericParam {
                    index: i2,
                    name: n2,
                },
            ) => {
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: Span::new(0, 0),
                    });
                }
                self.bindings.insert(b, a);
                self.type_map.remove(&TypeData::GenericParam {
                    index: *i2,
                    name: n2.clone(),
                });
                self.alias_cache.clear();
                self.instantiate_cache.clear();
                Ok(a)
            }
            _ => Err(TypeError::Mismatch {
                expected: b,
                found: a,
                span: Span::new(0, 0),
            }),
        }
    }

    fn expand_alias_def(&mut self, def_id: DefId, args: Vec<TypeId>) -> Result<TypeId, AliasError> {
        let cache_key = (def_id, args.clone());
        if let Some(&cached) = self.alias_cache.get(&cache_key) {
            return Ok(cached);
        }

        if self.expanding.contains(&def_id) {
            return Err(AliasError::Recursive(def_id));
        }

        let _guard = ExpandingGuard::new(self, def_id);

        let provider = self
            .symbol_provider
            .as_ref()
            .ok_or(AliasError::SymbolProviderNotSet)?;
        let (params, body) = provider
            .get_alias_body(def_id)
            .ok_or(AliasError::NotFound(def_id))?;

        if params.len() != args.len() {
            return Err(AliasError::ParameterCountMismatch {
                expected: params.len(),
                found: args.len(),
                def_id,
            });
        }

        let mut subst = Subst::new();
        for (i, _) in params.iter().enumerate() {
            subst.insert(i, args[i]);
        }
        let expanded = self.subst(body, &subst);

        self.alias_cache.insert(cache_key, expanded);
        Ok(expanded)
    }

    pub fn subtype(&mut self, sub: TypeId, sup: TypeId) -> bool {
        if sub == sup {
            return true;
        }

        let sub_data = self.resolve_binding(sub);
        let sup_data = self.resolve_binding(sup);

        match (sub_data, sup_data) {
            (TypeData::Error, _) => true,
            (_, TypeData::Error) => true,
            (TypeData::Never, _) => true,
            (TypeData::Unit, TypeData::Unit) => true,
            (TypeData::Alias { def_id, args }, _) => {
                if let Ok(expanded) = self.expand_alias_def(*def_id, args.clone()) {
                    self.subtype(expanded, *sup)
                } else {
                    false
                }
            }
            (_, TypeData::Alias { def_id, args }) => {
                if let Ok(expanded) = self.expand_alias_def(*def_id, args.clone()) {
                    self.subtype(*sub, expanded)
                } else {
                    false
                }
            }
            (
                TypeData::Ref {
                    ty: t1,
                    mutable: m1,
                },
                TypeData::Ref {
                    ty: t2,
                    mutable: m2,
                },
            ) => {
                if *m1 == *m2 {
                    self.subtype(*t1, *t2)
                } else {
                    false
                }
            }
            (
                TypeData::Fn {
                    params: p1,
                    ret: r1,
                },
                TypeData::Fn {
                    params: p2,
                    ret: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return false;
                }
                for (a, b) in p1.iter().zip(p2.iter()) {
                    if !self.subtype(*a, *b) {
                        return false;
                    }
                }
                self.subtype(*r1, *r2)
            }
            (TypeData::Array { elem: e1, size: s1 }, TypeData::Array { elem: e2, size: s2 }) => {
                *s1 == *s2 && self.subtype(*e1, *e2)
            }
            (TypeData::Slice { elem: e1 }, TypeData::Slice { elem: e2 }) => self.subtype(*e1, *e2),
            (TypeData::Tuple { elems: e1 }, TypeData::Tuple { elems: e2 }) => {
                if e1.len() != e2.len() {
                    return false;
                }
                e1.iter().zip(e2.iter()).all(|(a, b)| self.subtype(*a, *b))
            }
            (
                TypeData::Int {
                    bits: b1,
                    signed: s1,
                },
                TypeData::Int {
                    bits: b2,
                    signed: s2,
                },
            ) => *s1 == *s2 && *b1 == *b2,
            (TypeData::Float { bits: b1 }, TypeData::Float { bits: b2 }) => *b1 == *b2,
            _ => false,
        }
    }

    fn find_type(&self, data: &TypeData) -> Option<TypeId> {
        self.type_map.get(data).copied()
    }

    pub fn subst(&self, ty: TypeId, subst: &Subst) -> TypeId {
        match self.resolve_binding(ty) {
            TypeData::GenericParam { index, .. } => subst.get(*index).copied().unwrap_or(ty),
            TypeData::Int { bits, signed } => {
                let data = TypeData::Int {
                    bits: *bits,
                    signed: *signed,
                };
                self.find_type(&data).unwrap_or_else(|| {
                    panic!(
                        "Int type with bits={} signed={} not found in intern table",
                        bits, signed
                    )
                })
            }
            TypeData::Float { bits } => {
                let data = TypeData::Float { bits: *bits };
                self.find_type(&data).unwrap_or_else(|| {
                    panic!("Float type with bits={} not found in intern table", bits)
                })
            }
            TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error => ty,
            TypeData::Struct { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.struct_ty_no_alloc(*def_id, new_args)
                    .unwrap_or_else(|| {
                        panic!(
                            "Struct type with def_id={:?} and args={:?} not found in intern table",
                            def_id, new_args
                        )
                    })
            }
            TypeData::Enum { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.enum_ty_no_alloc(*def_id, new_args).unwrap_or_else(|| {
                    panic!(
                        "Enum type with def_id={:?} and args={:?} not found in intern table",
                        def_id, new_args
                    )
                })
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems.iter().map(|&e| self.subst(e, subst)).collect();
                self.tuple_ty_no_alloc(new_elems).unwrap_or_else(|| {
                    panic!(
                        "Tuple type with elems={:?} not found in intern table",
                        new_elems
                    )
                })
            }
            TypeData::Array { elem, size } => {
                let new_elem = self.subst(*elem, subst);
                self.array_ty_no_alloc(new_elem, *size).unwrap_or_else(|| {
                    panic!(
                        "Array type with elem={:?} size={} not found in intern table",
                        new_elem, size
                    )
                })
            }
            TypeData::Slice { elem } => {
                let new_elem = self.subst(*elem, subst);
                self.slice_ty_no_alloc(new_elem).unwrap_or_else(|| {
                    panic!(
                        "Slice type with elem={:?} not found in intern table",
                        new_elem
                    )
                })
            }
            TypeData::Ref { ty, mutable } => {
                let new_ty = self.subst(*ty, subst);
                self.ref_ty_no_alloc(new_ty, *mutable).unwrap_or_else(|| {
                    panic!(
                        "Ref type with ty={:?} mutable={} not found in intern table",
                        new_ty, mutable
                    )
                })
            }
            TypeData::Pointer { ty } => {
                let new_ty = self.subst(*ty, subst);
                self.pointer_ty_no_alloc(new_ty).unwrap_or_else(|| {
                    panic!(
                        "Pointer type with ty={:?} not found in intern table",
                        new_ty
                    )
                })
            }
            TypeData::Ptr { size, pointee } => {
                let new_size = self.subst(*size, subst);
                let new_pointee = self.subst(*pointee, subst);
                self.ptr_ty_no_alloc(new_size, new_pointee)
                    .unwrap_or_else(|| {
                        panic!(
                            "Ptr type with size={:?} pointee={:?} not found in intern table",
                            new_size, new_pointee
                        )
                    })
            }
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> =
                    params.iter().map(|&p| self.subst(p, subst)).collect();
                let new_ret = self.subst(*ret, subst);
                self.fn_ty_no_alloc(new_params, new_ret).unwrap_or_else(|| {
                    panic!(
                        "Fn type with params={:?} ret={:?} not found in intern table",
                        new_params, new_ret
                    )
                })
            }
            TypeData::DynTrait { traits } => ty,
            TypeData::Exists { name, base } => {
                let new_base = self.subst(*base, subst);
                self.exists_ty_no_alloc(name.clone(), new_base)
                    .unwrap_or_else(|| {
                        panic!(
                            "Exists type with name={:?} base={:?} not found in intern table",
                            name, new_base
                        )
                    })
            }
            TypeData::Alias { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.alias_ty_no_alloc(*def_id, new_args)
                    .unwrap_or_else(|| {
                        panic!(
                            "Alias type with def_id={:?} args={:?} not found in intern table",
                            def_id, new_args
                        )
                    })
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let new_self = self.subst(*self_ty, subst);
                self.associated_ty_no_alloc(*trait_id, name.clone(), new_self)
                    .unwrap_or_else(|| {
                        panic!(
                            "AssociatedType with trait_id={:?} name={:?} self_ty={:?} not found",
                            trait_id, name, new_self
                        )
                    })
            }
            _ => ty,
        }
    }

    fn struct_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Struct { def_id, args })
    }

    fn enum_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Enum { def_id, args })
    }

    fn tuple_ty_no_alloc(&self, elems: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Tuple { elems })
    }

    fn array_ty_no_alloc(&self, elem: TypeId, size: u64) -> Option<TypeId> {
        self.find_type(&TypeData::Array { elem, size })
    }

    fn slice_ty_no_alloc(&self, elem: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Slice { elem })
    }

    fn ref_ty_no_alloc(&self, ty: TypeId, mutable: bool) -> Option<TypeId> {
        self.find_type(&TypeData::Ref { ty, mutable })
    }

    fn pointer_ty_no_alloc(&self, ty: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Pointer { ty })
    }

    fn ptr_ty_no_alloc(&self, size: TypeId, pointee: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Ptr { size, pointee })
    }

    fn fn_ty_no_alloc(&self, params: Vec<TypeId>, ret: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Fn { params, ret })
    }

    fn exists_ty_no_alloc(&self, name: String, base: TypeId) -> Option<TypeId> {
        self.find_type(&TypeData::Exists { name, base })
    }

    fn alias_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Alias { def_id, args })
    }

    fn associated_ty_no_alloc(
        &self,
        trait_id: DefId,
        name: String,
        self_ty: TypeId,
    ) -> Option<TypeId> {
        self.find_type(&TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        })
    }

    pub fn instantiate(&mut self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        let cache_key = (def_id, args.clone());
        if let Some(&cached) = self.instantiate_cache.get(&cache_key) {
            return Some(cached);
        }

        let provider = self.symbol_provider.as_ref()?;

        if provider.is_type_alias(def_id) {
            let result = self.expand_alias_def(def_id, args).ok()?;
            self.instantiate_cache.insert(cache_key, result);
            return Some(result);
        }

        if let Some((params, _)) = provider.get_struct_definition(def_id) {
            if params.len() != args.len() {
                return None;
            }
            let result = self.struct_ty(def_id, args);
            self.instantiate_cache.insert(cache_key, result);
            return Some(result);
        }

        if let Some((params, _)) = provider.get_enum_definition(def_id) {
            if params.len() != args.len() {
                return None;
            }
            let result = self.enum_ty(def_id, args);
            self.instantiate_cache.insert(cache_key, result);
            return Some(result);
        }

        None
    }

    pub fn is_numeric(&self, ty: TypeId) -> bool {
        match self.resolve_binding(ty) {
            TypeData::Int { .. } | TypeData::Float { .. } => true,
            _ => false,
        }
    }

    pub fn is_integer(&self, ty: TypeId) -> bool {
        match self.resolve_binding(ty) {
            TypeData::Int { .. } | TypeData::USize => true,
            _ => false,
        }
    }

    pub fn is_unsigned(&self, ty: TypeId) -> bool {
        match self.resolve_binding(ty) {
            TypeData::Int { signed, .. } => !signed,
            TypeData::USize => true,
            _ => false,
        }
    }

    pub fn is_signed(&self, ty: TypeId) -> bool {
        match self.resolve_binding(ty) {
            TypeData::Int { signed, .. } => *signed,
            _ => false,
        }
    }

    pub fn is_float(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Float { .. })
    }

    pub fn is_bool(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Bool)
    }

    pub fn is_char(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Char)
    }

    pub fn is_byte(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Byte)
    }

    pub fn is_usize(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::USize)
    }

    pub fn is_unit(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Unit)
    }

    pub fn is_never(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Never)
    }

    pub fn is_error(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Error)
    }

    pub fn is_reference(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Ref { .. })
    }

    pub fn is_pointer(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Pointer { .. })
    }

    pub fn is_struct(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Struct { .. })
    }

    pub fn is_enum(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Enum { .. })
    }

    pub fn is_tuple(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Tuple { .. })
    }

    pub fn is_array(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Array { .. })
    }

    pub fn is_slice(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Slice { .. })
    }

    pub fn is_fn(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Fn { .. })
    }

    pub fn is_dyn_trait(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::DynTrait { .. })
    }

    pub fn is_exists(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Exists { .. })
    }

    pub fn is_alias(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::Alias { .. })
    }

    pub fn is_generic_param(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::GenericParam { .. })
    }

    pub fn is_associated_type(&self, ty: TypeId) -> bool {
        matches!(self.resolve_binding(ty), TypeData::AssociatedType { .. })
    }

    pub fn bits_of_int(&self, ty: TypeId) -> Option<u8> {
        match self.resolve_binding(ty) {
            TypeData::Int { bits, .. } => Some(*bits),
            _ => None,
        }
    }

    pub fn signedness_of_int(&self, ty: TypeId) -> Option<bool> {
        match self.resolve_binding(ty) {
            TypeData::Int { signed, .. } => Some(*signed),
            _ => None,
        }
    }

    pub fn bits_of_float(&self, ty: TypeId) -> Option<u8> {
        match self.resolve_binding(ty) {
            TypeData::Float { bits } => Some(*bits),
            _ => None,
        }
    }

    pub fn size_of_array(&self, ty: TypeId) -> Option<u64> {
        match self.resolve_binding(ty) {
            TypeData::Array { size, .. } => Some(*size),
            _ => None,
        }
    }

    pub fn elem_of_array(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Array { elem, .. } => Some(*elem),
            _ => None,
        }
    }

    pub fn elem_of_slice(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Slice { elem } => Some(*elem),
            _ => None,
        }
    }

    pub fn pointee_of_ref(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Ref { ty: t, .. } => Some(*t),
            _ => None,
        }
    }

    pub fn mutability_of_ref(&self, ty: TypeId) -> Option<bool> {
        match self.resolve_binding(ty) {
            TypeData::Ref { mutable, .. } => Some(*mutable),
            _ => None,
        }
    }

    pub fn pointee_of_pointer(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Pointer { ty: t } => Some(*t),
            _ => None,
        }
    }

    pub fn params_of_fn(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.resolve_binding(ty) {
            TypeData::Fn { params, .. } => Some(params),
            _ => None,
        }
    }

    pub fn ret_of_fn(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Fn { ret, .. } => Some(*ret),
            _ => None,
        }
    }

    pub fn tuple_elems(&self, ty: TypeId) -> Option<&[TypeId]> {
        match self.resolve_binding(ty) {
            TypeData::Tuple { elems } => Some(elems),
            _ => None,
        }
    }

    pub fn base_of_exists(&self, ty: TypeId) -> Option<TypeId> {
        match self.resolve_binding(ty) {
            TypeData::Exists { base, .. } => Some(*base),
            _ => None,
        }
    }

    pub fn name_of_exists(&self, ty: TypeId) -> Option<&String> {
        match self.resolve_binding(ty) {
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
    RecursiveType {
        ty: TypeId,
        span: Span,
    },
}

use crate::ast::EnumVariant;
use crate::ast::Expr;
use crate::ast::Span;
use crate::ast::StructField;
use crate::ast::TypeParam;
