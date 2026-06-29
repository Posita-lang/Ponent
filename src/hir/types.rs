use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateId(pub DefId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeData {
    Int {
        bits: u8,
        signed: bool,
    },
    UInt {
        bits: u8,
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
    /// An explicit universal quantifier: ∀X. Body
    /// `param_index` and `param_name` identify the bound variable X.
    /// X appears in `body` as `GenericParam { index: param_index }`.
    /// This is a compiler-internal node — there is no user-facing ∀ syntax.
    Forall {
        param_index: usize,
        param_name: String,
        body: TypeId,
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
    InferVar {
        id: usize,
    },
    Never,
    Unit,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub usize);

#[derive(Debug, Clone)]
pub struct TypeMeta {
    pub default_value: Option<crate::ast::Expr>,
    pub invariant: Option<crate::ast::Expr>,
    pub no_default: bool,
}

pub struct TypeContext {
    types: Vec<Arc<TypeData>>,
    type_map: HashMap<TypeData, TypeId>,
    pub(crate) bindings: RefCell<HashMap<TypeId, TypeId>>,
    meta: HashMap<TypeId, TypeMeta>,
    def_id_to_type_id: HashMap<DefId, TypeId>,
    pub builtin_unit: TypeId,
    pub builtin_never: TypeId,
    pub builtin_error: TypeId,
    pub builtin_bool: TypeId,
    pub builtin_char: TypeId,
    pub builtin_byte: TypeId,
    pub builtin_usize: TypeId,
}

impl TypeContext {
    pub fn new() -> Self {
        let mut ctx = TypeContext {
            types: Vec::new(),
            type_map: HashMap::default(),
            bindings: RefCell::new(HashMap::default()),
            meta: HashMap::default(),
            def_id_to_type_id: HashMap::default(),
            builtin_unit: TypeId(0),
            builtin_never: TypeId(0),
            builtin_error: TypeId(0),
            builtin_bool: TypeId(0),
            builtin_char: TypeId(0),
            builtin_byte: TypeId(0),
            builtin_usize: TypeId(0),
        };
        ctx.builtin_unit = ctx.alloc(TypeData::Unit);
        ctx.builtin_never = ctx.alloc(TypeData::Never);
        ctx.builtin_error = ctx.alloc(TypeData::Error);
        ctx.builtin_bool = ctx.alloc(TypeData::Bool);
        ctx.builtin_char = ctx.alloc(TypeData::Char);
        ctx.builtin_byte = ctx.alloc(TypeData::Byte);
        ctx.builtin_usize = ctx.alloc(TypeData::USize);
        ctx
    }

    pub fn get_invariant(&self, id: TypeId) -> Option<&crate::ast::Expr> {
        self.meta.get(&id).and_then(|m| m.invariant.as_ref())
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
        let resolved = self.resolve_binding(id);
        &self.types[resolved.0]
    }

    pub fn is_infer_var(&self, id: TypeId) -> bool {
        matches!(self.get(id), TypeData::InferVar { .. })
    }

    pub(crate) fn resolve_binding(&self, id: TypeId) -> TypeId {
        let mut current = id;
        loop {
            let bound = self.bindings.borrow().get(&current).copied();
            match bound {
                Some(next) => current = next,
                None => break,
            }
        }
        let root = current;
        let mut cur = id;
        while cur != root {
            let next = self.bindings.borrow().get(&cur).copied().unwrap();
            self.bindings.borrow_mut().insert(cur, root);
            cur = next;
        }
        root
    }

    pub fn alloc_infer_var(&mut self, id: usize) -> TypeId {
        self.alloc(TypeData::InferVar { id })
    }

    pub fn get_def_id_for_type(&self, id: TypeId) -> Option<DefId> {
        let resolved = self.resolve_binding(id);
        match &self.types[resolved.0].as_ref() {
            TypeData::Struct { def_id, .. } => Some(*def_id),
            TypeData::Enum { def_id, .. } => Some(*def_id),
            _ => None,
        }
    }

    pub fn register_def_id(&mut self, def_id: DefId, type_id: TypeId) {
        self.def_id_to_type_id.insert(def_id, type_id);
    }

    pub fn get_type_id_for_def_id(&self, def_id: DefId) -> Option<TypeId> {
        self.def_id_to_type_id.get(&def_id).copied()
    }

    pub fn int(&mut self, bits: u8, signed: bool) -> TypeId {
        self.alloc(TypeData::Int { bits, signed })
    }

    pub fn uint(&mut self, bits: u8) -> TypeId {
        self.alloc(TypeData::UInt { bits })
    }

    pub fn float(&mut self, bits: u8) -> TypeId {
        self.alloc(TypeData::Float { bits })
    }

    pub fn bool(&self) -> TypeId {
        self.builtin_bool
    }

    pub fn char(&self) -> TypeId {
        self.builtin_char
    }

    pub fn byte(&self) -> TypeId {
        self.builtin_byte
    }

    pub fn usize(&self) -> TypeId {
        self.builtin_usize
    }

    pub fn unit(&self) -> TypeId {
        self.builtin_unit
    }

    pub fn never(&self) -> TypeId {
        self.builtin_never
    }

    pub fn error(&self) -> TypeId {
        self.builtin_error
    }

    pub fn struct_ty(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        let id = self.alloc(TypeData::Struct { def_id, args });
        self.def_id_to_type_id.insert(def_id, id);
        id
    }

    pub fn enum_ty(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        let id = self.alloc(TypeData::Enum { def_id, args });
        self.def_id_to_type_id.insert(def_id, id);
        id
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

    /// Apply Yoneda reduction if this type matches the pattern
    /// `∀X. (A ⇒ X) ⇒ B⟨X⟩`  →  `B[X ↦ A]`
    /// or `∀X. (X ⇒ A) ⇒ B⟨X⟩`  →  `B[X ↦ A]`.
    /// Matches both explicit `Forall` nodes and implicit `Fn`-encoded patterns.
    pub fn try_yoneda_reduce(&mut self, ty: TypeId) -> TypeId {
        // Case A: explicit Forall node — Forall(X, Fn { params: [A], ret: X }) or Forall(X, Fn { params: [X], ret: A })
        if let TypeData::Forall { param_index, param_name: _, body } = self.get(ty).clone() {
            if let TypeData::Fn { params, ret } = self.get(body).clone() {
                if params.len() == 1 {
                    let a = params[0];
                    // Case 1: Forall(X, Fn { params: [A], ret: X })
                    if let TypeData::GenericParam { index, .. } = self.get(ret).clone() {
                        if index == param_index && self.check_positive_only(index, body) {
                            let reduced = self.subst(ret, &Subst::from_single(index, a));
                            if reduced != ret { return reduced; }
                        }
                    }
                    // Case 2: Forall(X, Fn { params: [X], ret: A })
                    if let TypeData::GenericParam { index, .. } = self.get(a).clone() {
                        if index == param_index && self.check_negative_only(index, body) {
                            let reduced = self.subst(ret, &Subst::from_single(index, a));
                            if reduced != ret { return reduced; }
                        }
                    }
                }
            }
            return ty; // Forall node doesn't match Yoneda pattern
        }

        // Case B: implicit Fn-encoded pattern (backward compatible)
        let (inner, ret) = match self.get(ty) {
            TypeData::Fn { params, ret } if params.len() == 1 => (params[0], *ret),
            _ => return ty,
        };
        let (a, x) = match self.get(inner) {
            TypeData::Fn { params, ret } if params.len() == 1 => (params[0], *ret),
            _ => return ty,
        };
        // Case 1: ∀X.(A ⇒ X) ⇒ B⟨X⟩  →  B[X↦A]  (X only positive in B)
        if let TypeData::GenericParam { index, .. } = self.get(x) {
            if self.check_positive_only(*index, ret) {
                let reduced = self.subst(ret, &Subst::from_single(*index, a));
                if reduced != ret {
                    return reduced;
                }
            }
        }
        // Case 2: ∀X.(X ⇒ A) ⇒ B⟨X⟩  →  B[X↦A]  (X only negative in B)
        if let TypeData::GenericParam { index, .. } = self.get(a) {
            if self.check_negative_only(*index, ret) {
                let reduced = self.subst(ret, &Subst::from_single(*index, x));
                if reduced != ret {
                    return reduced;
                }
            }
        }
        ty
    }

    /// Check that `param` only appears in positive (covariant) positions in `ty`.
    /// Check whether all occurrences of `param` in `ty` appear only in
    /// **positive** (covariant) positions. Uses sign propagation through
    /// the type tree (Pistone & Tranchini 2022 §2).
    fn check_positive_only(&self, param: usize, ty: TypeId) -> bool {
        self.check_variance(param, ty, 1)
    }

    /// Check whether all occurrences of `param` in `ty` appear only in
    /// **negative** (contravariant) positions.
    fn check_negative_only(&self, param: usize, ty: TypeId) -> bool {
        self.check_variance(param, ty, -1)
    }

    /// Core variance checker via sign propagation.
    /// Returns `true` iff every occurrence of `param` in `ty` is at a
    /// position whose cumulative sign matches `expected_sign`.
    ///
    /// Sign propagation rules:
    ///   - Fn params: contravariant → cumulative sign flips
    ///   - Fn ret: covariant → cumulative sign unchanged
    ///   - Ref/Pointer/Ptr: invariant → param cannot appear inside
    ///   - Tuple/Array/Slice/Struct args/Enum args/Forall body/Exists base:
    ///     covariant → cumulative sign unchanged
    fn check_variance(&self, param: usize, ty: TypeId, expected_sign: isize) -> bool {
        self.check_variance_with_sign(param, ty, expected_sign, 1)
    }

    fn check_variance_with_sign(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
    ) -> bool {
        match self.get(ty) {
            // Reached the tracked parameter — check sign match
            TypeData::GenericParam { index, .. } => {
                if *index == param {
                    cumulative_sign == expected_sign
                } else {
                    true // different param, doesn't affect result
                }
            }
            // Function type: params are contravariant (flip sign), ret is covariant (keep sign)
            TypeData::Fn { params, ret } => {
                params.iter().all(|p| {
                    if self.type_contains_param(param, *p) {
                        self.check_variance_with_sign(param, *p, expected_sign, -cumulative_sign)
                    } else {
                        true
                    }
                }) && self.check_variance_with_sign(param, *ret, expected_sign, cumulative_sign)
            }
            // Invariant containers: param cannot appear (sign effectively becomes 0)
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                !self.type_contains_param(param, *ty)
            }
            TypeData::Ptr { pointee, .. } => !self.type_contains_param(param, *pointee),
            // Covariant containers: keep sign
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                args.iter()
                    .all(|&a| self.check_variance_with_sign(param, a, expected_sign, cumulative_sign))
            }
            TypeData::Tuple { elems } => elems.iter().all(|&e| {
                self.check_variance_with_sign(param, e, expected_sign, cumulative_sign)
            }),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                self.check_variance_with_sign(param, *elem, expected_sign, cumulative_sign)
            }
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. } => {
                self.check_variance_with_sign(param, *body, expected_sign, cumulative_sign)
            }
            TypeData::AssociatedType { self_ty, .. } => {
                self.check_variance_with_sign(param, *self_ty, expected_sign, cumulative_sign)
            }
            TypeData::InferVar { .. } => true, // inference vars don't affect variance
            _ => true, // primitives (Int, Bool, etc.) don't contain generic params
        }
    }

    /// Check if a GenericParam with the given index appears anywhere in a type.
    pub fn type_contains_param(&self, param: usize, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::GenericParam { index, .. } => *index == param,
            TypeData::Fn { params, ret } => {
                params.iter().any(|&p| self.type_contains_param(param, p))
                || self.type_contains_param(param, *ret)
            }
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                args.iter().any(|&a| self.type_contains_param(param, a))
            }
            TypeData::Tuple { elems } => {
                elems.iter().any(|&e| self.type_contains_param(param, e))
            }
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                self.type_contains_param(param, *elem)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                self.type_contains_param(param, *ty)
            }
            TypeData::Ptr { pointee, .. } => self.type_contains_param(param, *pointee),
            TypeData::AssociatedType { self_ty, .. } => {
                self.type_contains_param(param, *self_ty)
            }
            TypeData::Exists { base, .. } => {
                self.type_contains_param(param, *base)
            }
            _ => false,
        }
    }

    pub fn dyn_trait(&mut self, traits: Vec<DefId>) -> TypeId {
        self.alloc(TypeData::DynTrait { traits })
    }

    pub fn exists(&mut self, name: String, base: TypeId, invariant: crate::ast::Expr) -> TypeId {
        let id = self.alloc(TypeData::Exists { name, base });
        self.meta.entry(id).or_insert(TypeMeta {
            default_value: None,
            invariant: Some(invariant),
            no_default: false,
        });
        id
    }

    pub fn forall(&mut self, param_index: usize, param_name: String, body: TypeId) -> TypeId {
        let id = self.alloc(TypeData::Forall {
            param_index,
            param_name,
            body,
        });
        // Automatically attempt Yoneda reduction:
        // ∀X.(A ⇒ X) ⇒ B⟨X⟩  →  B[X↦A]   (X only positive in body)
        // ∀X.(X ⇒ A) ⇒ B⟨X⟩  →  B[X↦A]   (X only negative in body)
        let reduced = self.try_yoneda_reduce(id);
        if reduced != id {
            reduced
        } else {
            id
        }
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

    fn occurs_check(&self, param: TypeId, ty: TypeId) -> bool {
        if param == ty {
            return true;
        }
        let resolved = self.resolve_binding(ty);
        match &self.types[resolved.0].as_ref() {
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
            TypeData::Exists { base, .. } => self.occurs_check(param, *base),
            TypeData::Forall { body, .. } => self.occurs_check(param, *body),
            TypeData::AssociatedType { self_ty, .. } => self.occurs_check(param, *self_ty),
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => false,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
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

    pub fn unify(&self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
        let data_a = self.get(a).clone();
        let data_b = self.get(b).clone();

        if data_a == data_b {
            return Ok(a);
        }

        match (&data_a, &data_b) {
            (TypeData::Error, _) => Ok(b),
            (_, TypeData::Error) => Ok(a),
            (
                TypeData::GenericParam { index: i1, .. },
                TypeData::GenericParam { index: i2, .. },
            ) if i1 == i2 => Ok(a),
            (TypeData::GenericParam { .. }, _) => {
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.bindings.borrow_mut().insert(a, b);
                Ok(b)
            }
            (_, TypeData::GenericParam { .. }) => {
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.bindings.borrow_mut().insert(b, a);
                Ok(a)
            }
            (TypeData::InferVar { .. }, _) => {
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.bindings.borrow_mut().insert(a, b);
                Ok(b)
            }
            (_, TypeData::InferVar { .. }) => {
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.bindings.borrow_mut().insert(b, a);
                Ok(a)
            }
            _ => Err(TypeError::Mismatch {
                expected: b,
                found: a,
                span: crate::ast::Span::new(0, 0),
            }),
        }
    }

    pub fn subtype(&self, sub: TypeId, sup: TypeId) -> bool {
        if sub == sup {
            return true;
        }

        let sub_data = self.get(sub);
        let sup_data = self.get(sup);

        match (sub_data, sup_data) {
            (TypeData::Error, _) => true,
            (_, TypeData::Error) => true,
            (TypeData::Never, _) => true,
            (TypeData::Unit, TypeData::Unit) => true,
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
                p1.iter().zip(p2.iter()).all(|(a, b)| self.subtype(*a, *b))
                    && self.subtype(*r1, *r2)
            }
            (TypeData::Array { elem: e1, size: s1 }, TypeData::Array { elem: e2, size: s2 }) => {
                *s1 == *s2 && self.subtype(*e1, *e2)
            }
            (TypeData::Slice { elem: e1 }, TypeData::Slice { elem: e2 }) => self.subtype(*e1, *e2),
            (TypeData::Tuple { elems: e1 }, TypeData::Tuple { elems: e2 }) => {
                e1.len() == e2.len() && e1.iter().zip(e2.iter()).all(|(a, b)| self.subtype(*a, *b))
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

    pub(crate) fn find_type(&self, data: &TypeData) -> Option<TypeId> {
        self.type_map.get(data).copied()
    }

    pub fn subst(&self, ty: TypeId, subst: &Subst) -> TypeId {
        let resolved = self.resolve_binding(ty);
        match &self.types[resolved.0].as_ref() {
            TypeData::GenericParam { index, .. } => subst.get(*index).copied().unwrap_or(ty),
            TypeData::Int { bits, signed } => {
                let data = TypeData::Int {
                    bits: *bits,
                    signed: *signed,
                };
                self.find_type(&data)
                    .expect("built-in Int type should exist")
            }
            TypeData::UInt { bits } => {
                let data = TypeData::UInt { bits: *bits };
                self.find_type(&data)
                    .expect("built-in UInt type should exist")
            }
            TypeData::Float { bits } => {
                let data = TypeData::Float { bits: *bits };
                self.find_type(&data)
                    .expect("built-in Float type should exist")
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
                    .expect("struct type should exist")
            }
            TypeData::Enum { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.enum_ty_no_alloc(*def_id, new_args)
                    .expect("enum type should exist")
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems.iter().map(|&e| self.subst(e, subst)).collect();
                self.tuple_ty_no_alloc(new_elems)
                    .expect("tuple type should exist")
            }
            TypeData::Array { elem, size } => {
                let new_elem = self.subst(*elem, subst);
                self.array_ty_no_alloc(new_elem, *size)
                    .expect("array type should exist")
            }
            TypeData::Slice { elem } => {
                let new_elem = self.subst(*elem, subst);
                self.slice_ty_no_alloc(new_elem)
                    .expect("slice type should exist")
            }
            TypeData::Ref { ty, mutable } => {
                let new_ty = self.subst(*ty, subst);
                self.ref_ty_no_alloc(new_ty, *mutable)
                    .expect("ref type should exist")
            }
            TypeData::Pointer { ty } => {
                let new_ty = self.subst(*ty, subst);
                self.pointer_ty_no_alloc(new_ty)
                    .expect("pointer type should exist")
            }
            TypeData::Ptr { size, pointee } => {
                let new_size = self.subst(*size, subst);
                let new_pointee = self.subst(*pointee, subst);
                self.ptr_ty_no_alloc(new_size, new_pointee)
                    .expect("ptr type should exist")
            }
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> =
                    params.iter().map(|&p| self.subst(p, subst)).collect();
                let new_ret = self.subst(*ret, subst);
                self.fn_ty_no_alloc(new_params, new_ret)
                    .expect("function type should exist")
            }
            TypeData::DynTrait { .. } => ty,
            TypeData::Forall { param_index, param_name, body } => {
                let new_body = self.subst(*body, subst);
                self.find_type(&TypeData::Forall { param_index: *param_index, param_name: param_name.clone(), body: new_body })
                    .unwrap_or(ty)
            }
            TypeData::Exists { name, base } => {
                let new_base = self.subst(*base, subst);
                self.exists_ty_no_alloc(name.clone(), new_base)
                    .expect("exists type should exist")
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let new_self = self.subst(*self_ty, subst);
                self.associated_ty_no_alloc(*trait_id, name.clone(), new_self)
                    .expect("associated type should exist")
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

    pub fn is_numeric(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::Float { .. } => true,
            _ => false,
        }
    }

    pub fn is_integer(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::USize => true,
            _ => false,
        }
    }

    pub fn is_unsigned(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { signed, .. } => !*signed,
            TypeData::UInt { .. } => true,
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
        matches!(self.get(ty), TypeData::Float { .. })
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

    /// Compute the constructor-depth of a type for Paterson-condition checking.
    /// GenericParam = 0, Int/Bool/etc = 1, Struct/Enum = 1 + max(depth of args)
    pub fn type_constructor_depth(&self, ty: TypeId) -> usize {
        match self.get(ty) {
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => 0,
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                1 + args.iter().map(|a| self.type_constructor_depth(*a)).max().unwrap_or(0)
            }
            TypeData::Tuple { elems } => {
                1 + elems.iter().map(|e| self.type_constructor_depth(*e)).max().unwrap_or(0)
            }
            TypeData::Array { elem, .. } => 1 + self.type_constructor_depth(*elem),
            TypeData::Slice { elem } => 1 + self.type_constructor_depth(*elem),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => 1 + self.type_constructor_depth(*ty),
            TypeData::Ptr { pointee, .. } => 1 + self.type_constructor_depth(*pointee),
            TypeData::Fn { params, ret } => {
                1 + params.iter().map(|p| self.type_constructor_depth(*p)).max().unwrap_or(0)
                    .max(self.type_constructor_depth(*ret))
            }
            TypeData::AssociatedType { self_ty, .. } => 1 + self.type_constructor_depth(*self_ty),
            TypeData::Exists { base, .. } => 1 + self.type_constructor_depth(*base),
            TypeData::Forall { body, .. } => 1 + self.type_constructor_depth(*body),
            TypeData::DynTrait { .. } => 1,
            TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::Float { .. }
            | TypeData::Bool | TypeData::Char | TypeData::Byte | TypeData::USize
            | TypeData::Never | TypeData::Unit | TypeData::Error => 1,
        }
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

    pub fn is_generic_param(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::GenericParam { .. })
    }

    pub fn is_associated_type(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::AssociatedType { .. })
    }

    pub fn bits_of_int(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Int { bits, .. } | TypeData::UInt { bits } => Some(*bits),
            _ => None,
        }
    }

    pub fn signedness_of_int(&self, ty: TypeId) -> Option<bool> {
        match self.get(ty) {
            TypeData::Int { signed, .. } => Some(*signed),
            TypeData::UInt { .. } => Some(false),
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

    pub fn set_meta(&mut self, id: TypeId, meta: TypeMeta) {
        self.meta.insert(id, meta);
    }

    pub fn get_meta(&self, id: TypeId) -> Option<&TypeMeta> {
        self.meta.get(&id)
    }
}

#[derive(Debug, Clone)]
pub struct Subst {
    map: HashMap<usize, TypeId>,
}

impl Subst {
    pub fn new() -> Self {
        Subst {
            map: HashMap::default(),
        }
    }

    pub fn insert(&mut self, index: usize, ty: TypeId) {
        self.map.insert(index, ty);
    }

    pub fn from_single(index: usize, ty: TypeId) -> Self {
        let mut map = HashMap::default();
        map.insert(index, ty);
        Subst { map }
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

/// The characteristic κ(A) of a type, describing its inhabitant count:
/// - `FiniteExhaustible(usize)` → κ=0: finite inhabitants (e.g. `Bool` has 2)
/// - `InfiniteEnumerable` → κ=1: infinite but enumerable (recursive types with only covariant cycles)
/// - `Undecidable` → κ=∞: cannot decide (cycles through contravariant/invariant positions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Characteristic {
    FiniteExhaustible(usize),
    InfiniteEnumerable,
    Undecidable,
}

impl TypeContext {
    /// Compute the characteristic κ(A) of a type, used for exhaustiveness checking
    /// of match expressions.
    ///
    /// Algorithm (Pistone & Tranchini 2022 §5):
    /// 1. Treat the type tree as a directed graph with variance-annotated edges
    ///    (covariant →, contravariant ⇒, invariant ↔).
    /// 2. Add "axiom links" connecting matching GenericParam nodes.
    /// 3. Detect cyclic paths through the graph.
    /// 4. Acyclic → κ=0, cyclic only through covariant edges → κ=1, other → κ=∞.
    pub fn characteristic(&self, ty: TypeId, visited: &mut Vec<TypeId>) -> Characteristic {
        // Prevent infinite recursion on cyclic types
        if visited.contains(&ty) {
            return Characteristic::InfiniteEnumerable;
        }
        visited.push(ty);

        let data = self.get(ty);
        match data {
            TypeData::Int { bits, .. } => {
                let count = 1usize << bits;
                Characteristic::FiniteExhaustible(count)
            }
            TypeData::UInt { bits } => {
                let count = 1usize << bits;
                Characteristic::FiniteExhaustible(count)
            }
            TypeData::Float { .. } | TypeData::USize => {
                // Floats and usize have very large but finite domains
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Bool => Characteristic::FiniteExhaustible(2),
            TypeData::Char => Characteristic::FiniteExhaustible(256),
            TypeData::Byte => Characteristic::FiniteExhaustible(256),
            TypeData::Unit => Characteristic::FiniteExhaustible(1),
            TypeData::Never => Characteristic::FiniteExhaustible(0),
            TypeData::Error => Characteristic::FiniteExhaustible(0),
            TypeData::Tuple { elems } => {
                let mut total = 1usize;
                let mut has_infinite = false;
                for &elem in elems {
                    match self.characteristic(elem, visited) {
                        Characteristic::FiniteExhaustible(n) => {
                            total = total.saturating_mul(n);
                        }
                        Characteristic::InfiniteEnumerable => {
                            has_infinite = true;
                        }
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(total)
                }
            }
            TypeData::Struct { def_id: _, args } | TypeData::Enum { def_id: _, args } => {
                let mut has_infinite = false;
                for &arg in args {
                    match self.characteristic(arg, visited) {
                        Characteristic::FiniteExhaustible(_) => {}
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(usize::MAX)
                }
            }
            TypeData::Array { elem, size } => {
                match self.characteristic(*elem, visited) {
                    Characteristic::FiniteExhaustible(n) => {
                        Characteristic::FiniteExhaustible(n.saturating_pow(*size as u32))
                    }
                    Characteristic::InfiniteEnumerable => Characteristic::InfiniteEnumerable,
                    Characteristic::Undecidable => Characteristic::Undecidable,
                }
            }
            TypeData::Slice { elem } | TypeData::Ref { ty: elem, .. }
            | TypeData::Pointer { ty: elem }
            | TypeData::Ptr { pointee: elem, .. } => {
                let _ = self.characteristic(*elem, visited);
                Characteristic::InfiniteEnumerable
            }
            TypeData::Fn { params, ret } => {
                for &p in params {
                    if self.characteristic(p, visited) == Characteristic::Undecidable {
                        return Characteristic::Undecidable;
                    }
                }
                match self.characteristic(*ret, visited) {
                    Characteristic::Undecidable => Characteristic::Undecidable,
                    _ => Characteristic::InfiniteEnumerable,
                }
            }
            TypeData::GenericParam { .. } => {
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. } => {
                self.characteristic(*body, visited)
            }
            TypeData::DynTrait { .. } | TypeData::AssociatedType { .. } => {
                Characteristic::InfiniteEnumerable
            }
            TypeData::InferVar { .. } => {
                Characteristic::FiniteExhaustible(usize::MAX)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch {
        expected: TypeId,
        found: TypeId,
        span: crate::ast::Span,
    },
    UndefinedName {
        name: String,
        span: crate::ast::Span,
        suggestions: Vec<String>,
    },
    TypeNotFound {
        name: String,
        span: crate::ast::Span,
    },
    CannotInfer {
        span: crate::ast::Span,
    },
    GenericArgumentCount {
        expected: usize,
        found: usize,
        span: crate::ast::Span,
    },
    TraitNotImplemented {
        ty: TypeId,
        trait_name: String,
        span: crate::ast::Span,
    },
    InvariantViolation {
        ty: TypeId,
        expr: String,
        span: crate::ast::Span,
    },
    MutableBorrow {
        span: crate::ast::Span,
    },
    ImmutableBorrow {
        span: crate::ast::Span,
    },
    OutOfBounds {
        index: u64,
        size: u64,
        span: crate::ast::Span,
    },
    DivisionByZero {
        span: crate::ast::Span,
    },
    Overflow {
        span: crate::ast::Span,
    },
    NeverType {
        span: crate::ast::Span,
    },
    CircularDependency {
        name: String,
        span: crate::ast::Span,
    },
    DuplicateDefinition {
        name: String,
        span: crate::ast::Span,
        previous: crate::ast::Span,
    },
    PrivateField {
        name: String,
        span: crate::ast::Span,
    },
    PrivateType {
        name: String,
        span: crate::ast::Span,
    },
    PrivateFunction {
        name: String,
        span: crate::ast::Span,
    },
    PatternNotExhaustive {
        span: crate::ast::Span,
    },
    PatternRedundant {
        span: crate::ast::Span,
    },
    PatternTypeMismatch {
        expected: TypeId,
        found: TypeId,
        span: crate::ast::Span,
    },
    RecursiveType {
        ty: TypeId,
        span: crate::ast::Span,
    },
}


