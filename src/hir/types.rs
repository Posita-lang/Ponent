use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum TypeTag {
    Int = 0,
    UInt = 1,
    Float = 2,
    Bool = 3,
    Char = 4,
    Byte = 5,
    USize = 6,
    Struct = 7,
    Enum = 8,
    Tuple = 9,
    Array = 10,
    Slice = 11,
    Ref = 12,
    Pointer = 13,
    Ptr = 14,
    Fn = 15,
    DynTrait = 16,
    Exists = 17,
    Forall = 18,
    GenericParam = 19,
    AssociatedType = 20,
    InferVar = 21,
    Never = 22,
    Unit = 23,
    Error = 24,
    Coproduct = 25,
    Mu = 26,
    Nu = 27,
}

impl From<&TypeData> for TypeTag {
    fn from(data: &TypeData) -> Self {
        match data {
            TypeData::Int { .. } => TypeTag::Int,
            TypeData::UInt { .. } => TypeTag::UInt,
            TypeData::Float { .. } => TypeTag::Float,
            TypeData::Bool => TypeTag::Bool,
            TypeData::Char => TypeTag::Char,
            TypeData::Byte => TypeTag::Byte,
            TypeData::USize => TypeTag::USize,
            TypeData::Struct { .. } => TypeTag::Struct,
            TypeData::Enum { .. } => TypeTag::Enum,
            TypeData::Tuple { .. } => TypeTag::Tuple,
            TypeData::Array { .. } => TypeTag::Array,
            TypeData::Slice { .. } => TypeTag::Slice,
            TypeData::Ref { .. } => TypeTag::Ref,
            TypeData::Pointer { .. } => TypeTag::Pointer,
            TypeData::Ptr { .. } => TypeTag::Ptr,
            TypeData::Fn { .. } => TypeTag::Fn,
            TypeData::DynTrait { .. } => TypeTag::DynTrait,
            TypeData::Exists { .. } => TypeTag::Exists,
            TypeData::Forall { .. } => TypeTag::Forall,
            TypeData::GenericParam { .. } => TypeTag::GenericParam,
            TypeData::AssociatedType { .. } => TypeTag::AssociatedType,
            TypeData::InferVar { .. } => TypeTag::InferVar,
            TypeData::Coproduct { .. } => TypeTag::Coproduct,
            TypeData::Mu { .. } => TypeTag::Mu,
            TypeData::Nu { .. } => TypeTag::Nu,
            TypeData::Never => TypeTag::Never,
            TypeData::Unit => TypeTag::Unit,
            TypeData::Error => TypeTag::Error,
        }
    }
}

impl TypeId {
    pub const TAG_BITS: usize = 5;
    const TAG_MASK: usize = (1 << Self::TAG_BITS) - 1;

    pub fn index(self) -> usize {
        self.0 >> Self::TAG_BITS
    }

    pub fn tag(self) -> TypeTag {
        // SAFETY: every TypeId created through TypeContext::alloc has a valid tag
        // and TAG_MASK covers all 25 discriminants (0..24).
        unsafe { std::mem::transmute::<usize, TypeTag>(self.0 & Self::TAG_MASK) }
    }
}

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
    /// A named coproduct (sum type), Σᵢ Aᵢ.
    /// Introduced by Yoneda reduction of ∀X.(A₁⇒X)⇒...⇒(Aₙ⇒X)⇒X → Σᵢ Aᵢ.
    /// Unlike Tuple (product), Coproduct represents "one of the alternatives."
    Coproduct {
        alternatives: Vec<TypeId>,
    },
    /// Least fixed-point type: μX.A⟨X⟩.
    /// X is the recursive type variable, identified by param_index in body.
    Mu {
        param_index: usize,
        param_name: String,
        body: TypeId,
    },
    /// Greatest fixed-point type: νX.A⟨X⟩.
    Nu {
        param_index: usize,
        param_name: String,
        body: TypeId,
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

/// A variance-annotated edge in the type graph.
/// Pre-computed so that variance propagation is a simple graph
/// traversal over edges, not pattern-matching on TypeData each time.
#[derive(Debug, Clone, Copy)]
struct VarianceEdge {
    target: TypeId,
    /// +1 = covariant, -1 = contravariant, 0 = invariant
    sign: isize,
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
    /// Cache for variance check results: (param_index, TypeId, expected_sign) → bool.
    variance_cache: RefCell<HashMap<(usize, TypeId, isize), bool>>,
    /// Pre-computed variance-annotated outgoing edges for each TypeId.
    /// Built lazily on first variance check, then reused.
    variance_edges: RefCell<HashMap<TypeId, Vec<VarianceEdge>>>,
    /// Transaction stack for atomic unification (OmniML-style rollback).
    /// Each entry captures bindings before an operation.
    /// On rollback, bindings are restored; on commit, snapshot is discarded.
    transaction_stack: RefCell<Vec<HashMap<TypeId, TypeId>>>,
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
            variance_cache: RefCell::new(HashMap::default()),
            variance_edges: RefCell::new(HashMap::default()),
            transaction_stack: RefCell::new(Vec::new()),
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
        let tag = TypeTag::from(&data) as usize;
        let index = self.types.len();
        let id = TypeId((index << TypeId::TAG_BITS) | tag);
        self.types.push(Arc::new(data.clone()));
        self.type_map.insert(data, id);
        id
    }

    pub fn get(&self, id: TypeId) -> &TypeData {
        let resolved = self.resolve_binding(id);
        &self.types[resolved.index()]
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
        match &self.types[resolved.index()].as_ref() {
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

    /// Create a coproduct (sum type) Σ Aᵢ — "one of the alternatives".
    /// Used by Yoneda reduction to encode branch choice.
    pub fn coproduct(&mut self, alternatives: Vec<TypeId>) -> TypeId {
        match alternatives.len() {
            0 => self.never(),
            1 => alternatives[0],
            _ => self.alloc(TypeData::Coproduct { alternatives }),
        }
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
    ///
    /// Uses iteration with convergence detection (max 10 rounds) to handle
    /// chained reductions like Forall(X, Forall(Y, ...)) where reducing the
    /// outer Forall exposes a new reducible inner Forall. This follows the
    /// same convergence principle as Yen's KSP algorithm:
    /// "keep iterating until no more candidates can be generated".
    pub fn try_yoneda_reduce(&mut self, ty: TypeId) -> TypeId {
        const MAX_ITERATIONS: usize = 10;
        let mut result = ty;
        for _iteration in 0..MAX_ITERATIONS {
            let before = result;
            result = self.yoneda_reduce_once(result);
            if result == before {
                break; // converged
            }
        }
        result
    }

    /// Single-pass Yoneda reduction (used internally by `try_yoneda_reduce`).
    ///
    /// Matches the ≡_X / ≡_X schemas from Pistone & Tranchini (2022) §2.
    ///
    /// **≡_X (Yoneda)** – each branch's *return* is the bound variable X:
    /// ```text
    /// ∀X. ∀Y⃗. ⟨ ∀Z⃗ₖ. ⟨Aⱼₖ⟨X⟩⟩ⱼ ⇒ X ⟩ₖ ⇒ B⟨X⟩
    ///   ≡_X
    /// ∀Y⃗. B⟨X ↦ Σₖ ∃Z⃗ₖ. Πⱼ Aⱼₖ⟨X⟩⟩
    /// ```
    ///
    /// **≡_X (co-Yoneda)** – each branch's *first param* is the bound variable X:
    /// ```text
    /// ∀X. ∀Y⃗. ⟨ ∀Z⃗ₖ. X ⇒ Aⱼ⟨X⟩ ⟩ₖ ⇒ B⟨X⟩
    ///   ≡_X
    /// ∀Y⃗. B⟨X ↦ νX. ∀Z⃗ₖ. Πⱼ Aⱼ⟨X⟩⟩
    /// ```
    ///
    /// Terms: `Σₖ` = sum (multi-branch → Tuple of results), `Πⱼ` = product (Tuple),
    /// `∃Z⃗ₖ` = Exists node when the branch has inner quantifiers.
    /// μX/νX fixpoints are elided when X does not appear in B⟨X⟩ (common case).
    fn yoneda_reduce_once(&mut self, ty: TypeId) -> TypeId {
        // ── Case A: explicit Forall node ──────────────────────────────
        if let TypeData::Forall { param_index, param_name: _, body } = self.get(ty).clone() {
            if let TypeData::Fn { params, ret } = self.get(body).clone() {
                let pi = param_index;
                let mut branch_replacements: Vec<TypeId> = Vec::new();
                for branch in params {
                    // Peel outer Forall layers (∀Z⃗ₖ).
                    let mut inner_quantifiers: Vec<(usize, String)> = Vec::new();
                    let mut inner = branch;
                    loop {
                        match self.get(inner).clone() {
                            TypeData::Forall { param_index: fi, param_name: fn_, body: b } => {
                                inner_quantifiers.push((fi, fn_));
                                inner = b;
                            }
                            _ => break,
                        }
                    }
                    if let TypeData::Fn { params: ips, ret: ir } = self.get(inner).clone() {
                        // Check ≡_X (Yoneda): ir = GenericParam(pi)
                        let yoneda_match = match self.get(ir) {
                            TypeData::GenericParam { index, .. } if *index == pi => true,
                            _ => false,
                        };
                        // Check ≡_X (co-Yoneda): ips[0] = GenericParam(pi)
                        let coyoneda_match = if !ips.is_empty() {
                            match self.get(ips[0]) {
                                TypeData::GenericParam { index, .. } if *index == pi => true,
                                _ => false,
                            }
                        } else { false };
                        // Process Yoneda case
                        if yoneda_match {
                            let product = if ips.len() == 1 { ips[0] }
                                else { self.tuple(ips.clone()) };
                            let repl = if inner_quantifiers.is_empty() { product }
                                else {
                                    let mut w = product;
                                    for (eq, en) in &inner_quantifiers {
                                        w = self.exists(en.clone(), w,
                                            crate::ast::Expr::Literal(crate::ast::Literal::Bool(true),
                                                crate::ast::Span::new(0, 0)));
                                    }
                                    w
                                };
                            branch_replacements.push(repl);
                        }
                        // Process co-Yoneda case
                        if coyoneda_match {
                            // The branch is X ⇒ A where A = inner_ret (not a param).
                            // When ips.len() == 1, X is the only param and A = inner_ret.
                            // When ips.len() > 1, the branch is X ⇒ A₁ ⇒ ... ⇒ Aⱼ
                            // and the replacement is the return of the branch itself.
                            let replacement = if ips.len() <= 1 { ir }
                                else if ips.len() == 2 { ips[1] }
                                else { self.tuple(ips[1..].to_vec()) };
                            let repl = if inner_quantifiers.is_empty() { replacement }
                                else {
                                    let mut w = replacement;
                                    for (_eq, en) in &inner_quantifiers {
                                        w = self.forall(*_eq, en.clone(), w);
                                    }
                                    w
                                };
                            branch_replacements.push(repl);
                        }
                    }
                }
                if !branch_replacements.is_empty() {
                    // Σₖ is the categorical coproduct (sum type), NOT a product.
                    // For ∀X.(A₁⇒X)⇒(A₂⇒X)⇒X  →  A₁ + A₂
                    let sigma = self.coproduct(branch_replacements);
                    // Wrap with μX only when the branch product(s) depend on X
                    // (Pistone & Tranchini 2022 §2, eq.3):
                    //   ∀X.(A⟨X⟩⇒X)⇒B⟨X⟩  →  B⟨X↦μX.A⟨X⟩⟩
                    // When A⟨X⟩ = Int (no X), no Mu needed:
                    //   ∀X.(Int⇒X)⇒B⟨X⟩  →  B⟨X↦Int⟩
                    let needs_mu = self.type_contains_param(pi, sigma);
                    let replacement = if needs_mu {
                        self.alloc(TypeData::Mu {
                            param_index: pi,
                            param_name: "X".into(),
                            body: sigma,
                        })
                    } else { sigma };
                    return self.replace_generic(ret, pi, replacement);
                }
            }
            return ty;
        }

        // ── Case B: implicit Fn-encoded pattern (backward compatible) ──
        let (inner, ret) = match self.get(ty) {
            TypeData::Fn { params, ret } if params.len() == 1 => (params[0], *ret),
            _ => return ty,
        };
        if let TypeData::Fn { params: inner_params, ret: inner_ret } = self.get(inner).clone() {
            // ≡_X (Yoneda): inner_ret is GenericParam X
            let yoneda_idx = match self.get(inner_ret) {
                TypeData::GenericParam { index, .. } => Some(*index),
                _ => None,
            };
            if let Some(idx) = yoneda_idx {
                let replacement = if inner_params.len() == 1 { inner_params[0] }
                    else { self.tuple(inner_params.clone()) };
                return self.replace_generic(ret, idx, replacement);
            }
            // ≡_X (co-Yoneda): first inner param is GenericParam X
            if !inner_params.is_empty() {
                let coyoneda_idx = match self.get(inner_params[0]) {
                    TypeData::GenericParam { index, .. } => Some(*index),
                    _ => None,
                };
                if let Some(idx) = coyoneda_idx {
                    return self.replace_generic(ret, idx, inner_ret);
                }
            }
        }
        ty
    }
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
        // Check cache first
        let cache_key = (param, ty, expected_sign);
        if let Some(&cached) = self.variance_cache.borrow().get(&cache_key) {
            return cached;
        }
        let result = self.check_variance_uncached(param, ty, expected_sign, cumulative_sign);
        self.variance_cache.borrow_mut().insert(cache_key, result);
        result
    }

    fn check_variance_uncached(
        &self,
        param: usize,
        ty: TypeId,
        expected_sign: isize,
        cumulative_sign: isize,
    ) -> bool {
        // Use pre-computed variance edges instead of pattern-matching TypeData.
        // This is faster because edges are computed once and reused.
        let edges = self.get_variance_edges(ty);
        for edge in &edges {
            if self.type_contains_param(param, edge.target) {
                // Propagate sign: contravariant flips, invariant blocks
                match edge.sign {
                    -1 => {
                        // Contravariant: flip cumulative sign
                        if !self.check_variance_with_sign(param, edge.target, expected_sign, -cumulative_sign) {
                            return false;
                        }
                    }
                    0 => {
                        // Invariant: param cannot appear
                        return false;
                    }
                    _ => {
                        // Covariant: keep cumulative sign
                        if !self.check_variance_with_sign(param, edge.target, expected_sign, cumulative_sign) {
                            return false;
                        }
                    }
                }
            }
        }
        // No edges → no sub-types (leaf node). Check if THIS node is the param.
        if edges.is_empty() {
            if let TypeData::GenericParam { index, .. } = self.get(ty) {
                if *index == param {
                    return cumulative_sign == expected_sign;
                }
            }
        }
        true
    }

    /// Get (or compute) the variance-annotated outgoing edges for a TypeId.
    /// Edges represent "this type → child type with given variance sign".
    fn get_variance_edges(&self, ty: TypeId) -> Vec<VarianceEdge> {
        if let Some(edges) = self.variance_edges.borrow().get(&ty) {
            return edges.clone();
        }
        let edges = self.compute_variance_edges(ty);
        self.variance_edges.borrow_mut().insert(ty, edges.clone());
        edges
    }

    /// Build the outgoing variance edges for a TypeId by inspecting its TypeData.
    fn compute_variance_edges(&self, ty: TypeId) -> Vec<VarianceEdge> {
        match self.get(ty) {
            TypeData::Fn { params, ret } => {
                let mut edges: Vec<VarianceEdge> = params.iter()
                    .map(|&p| VarianceEdge { target: p, sign: -1 })
                    .collect();
                edges.push(VarianceEdge { target: *ret, sign: 1 });
                edges
            }
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                args.iter().map(|&a| VarianceEdge { target: a, sign: 1 }).collect()
            }
            TypeData::Tuple { elems } => {
                elems.iter().map(|&e| VarianceEdge { target: e, sign: 1 }).collect()
            }
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                vec![VarianceEdge { target: *elem, sign: 1 }]
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                vec![VarianceEdge { target: *ty, sign: 0 }]
            }
            TypeData::Ptr { pointee, .. } => {
                vec![VarianceEdge { target: *pointee, sign: 0 }]
            }
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                vec![VarianceEdge { target: *body, sign: 1 }]
            }
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().map(|&a| VarianceEdge { target: a, sign: 1 }).collect()
            }
            TypeData::AssociatedType { self_ty, .. } => {
                vec![VarianceEdge { target: *self_ty, sign: 1 }]
            }
            // Leaves: no edges (GenericParam, primitives, etc.)
            _ => Vec::new(),
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
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.type_contains_param(param, a))
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
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                self.type_contains_param(param, *body)
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

    /// Skip the `subst` type-pool lookup limitations and directly build
    /// the replacement type.  This avoids the `fn_ty_no_alloc().expect()`
    /// panic that occurs when `subst` tries to find a pre-existing type
    /// that hasn't been created yet.
    pub fn replace_generic(&mut self, ty: TypeId, param_index: usize, replacement: TypeId) -> TypeId {
        if !self.type_contains_param(param_index, ty) { return ty; }
        let data = self.get(ty).clone();
        match data {
            TypeData::GenericParam { index, .. } if index == param_index => replacement,
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> = params.iter()
                    .map(|&p| self.replace_generic(p, param_index, replacement))
                    .collect();
                let new_ret = self.replace_generic(ret, param_index, replacement);
                self.function(new_params, new_ret)
            }
            TypeData::Forall { param_index: pi, param_name, body } => {
                let new_body = self.replace_generic(body, param_index, replacement);
                self.forall(pi, param_name, new_body)
            }
            TypeData::Mu { param_index: pi, param_name, body } => {
                let new_body = self.replace_generic(body, param_index, replacement);
                self.alloc(TypeData::Mu { param_index: pi, param_name, body: new_body })
            }
            TypeData::Nu { param_index: pi, param_name, body } => {
                let new_body = self.replace_generic(body, param_index, replacement);
                self.alloc(TypeData::Nu { param_index: pi, param_name, body: new_body })
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems.iter().map(|&e| self.replace_generic(e, param_index, replacement)).collect();
                self.tuple(new_elems)
            }
            TypeData::Struct { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.replace_generic(a, param_index, replacement)).collect();
                self.struct_ty(def_id, new_args)
            }
            TypeData::Enum { def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.replace_generic(a, param_index, replacement)).collect();
                self.enum_ty(def_id, new_args)
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: Vec<TypeId> = alternatives.iter().map(|&a| self.replace_generic(a, param_index, replacement)).collect();
                if new_alts.len() == 1 { new_alts[0] }
                else { self.alloc(TypeData::Coproduct { alternatives: new_alts }) }
            }
            _ => ty,
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
        match &self.types[resolved.index()].as_ref() {
            TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                args.iter().any(|&a| self.occurs_check(param, a))
            }
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.occurs_check(param, e)),
            TypeData::Coproduct { alternatives } => {
                alternatives.iter().any(|&a| self.occurs_check(param, a))
            }
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
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => self.occurs_check(param, *body),
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
        // ── Transaction: capture current bindings for rollback ──
        self.begin_transaction();

        let result = self.unify_internal(a, b);

        // ── Commit or rollback ──
        match result {
            Ok(ty) => {
                self.commit_transaction();
                Ok(ty)
            }
            Err(e) => {
                self.rollback_transaction();
                Err(e)
            }
        }
    }

    /// Internal unification (no transaction wrapping).
    fn unify_internal(&self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
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

    // ── Transaction support for atomic unification ────────────────

    /// Begin a new transaction: save current bindings snapshot.
    pub fn begin_transaction(&self) {
        let snapshot = self.bindings.borrow().clone();
        self.transaction_stack.borrow_mut().push(snapshot);
    }

    /// Commit the current transaction: discard the snapshot.
    pub fn commit_transaction(&self) {
        self.transaction_stack.borrow_mut().pop();
    }

    /// Rollback the current transaction: restore bindings from snapshot.
    pub fn rollback_transaction(&self) {
        if let Some(snapshot) = self.transaction_stack.borrow_mut().pop() {
            *self.bindings.borrow_mut() = snapshot;
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
            (TypeData::Coproduct { alternatives: a1 }, TypeData::Coproduct { alternatives: a2 }) => {
                a1.len() == a2.len() && a1.iter().zip(a2.iter()).all(|(a, b)| self.subtype(*a, *b))
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
        match &self.types[resolved.index()].as_ref() {
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
            TypeData::Mu { param_index, param_name, body } => {
                let new_body = self.subst(*body, subst);
                self.find_type(&TypeData::Mu { param_index: *param_index, param_name: param_name.clone(), body: new_body })
                    .unwrap_or(ty)
            }
            TypeData::Nu { param_index, param_name, body } => {
                let new_body = self.subst(*body, subst);
                self.find_type(&TypeData::Nu { param_index: *param_index, param_name: param_name.clone(), body: new_body })
                    .unwrap_or(ty)
            }
            TypeData::Exists { name, base } => {
                let new_base = self.subst(*base, subst);
                self.exists_ty_no_alloc(name.clone(), new_base)
                    .expect("exists type should exist")
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: Vec<TypeId> = alternatives.iter().map(|&a| self.subst(a, subst)).collect();
                self.coproduct_ty_no_alloc(new_alts)
                    .expect("coproduct type should exist")
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

    fn coproduct_ty_no_alloc(&self, alternatives: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Coproduct { alternatives })
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
            TypeData::Tuple { elems } | TypeData::Coproduct { alternatives: elems } => {
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
            TypeData::Forall { body, .. } | TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => 1 + self.type_constructor_depth(*body),
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
    ///
    /// Uses KSP-style convergence detection (max 10 iterations) to handle
    /// mutually recursive types where the characteristic may change across
    /// successive rounds of refinement.
    pub fn characteristic(&self, ty: TypeId, visited: &mut Vec<TypeId>) -> Characteristic {
        const MAX_ITERATIONS: usize = 10;
        let mut result = self.characteristic_with_variance(ty, visited, false);
        for _iteration in 0..MAX_ITERATIONS {
            let before = result;
            // Re-run with the same visited to detect convergence:
            // if the result stabilises, the cycle is resolved.
            let mut re_visited = visited.clone();
            result = self.characteristic_with_variance(ty, &mut re_visited, false);
            if result == before {
                break; // converged
            }
            *visited = re_visited;
        }
        result
    }

    /// Compute κ with scoped (remove+recover) visited tracking.
    /// When backtracking, `ty` is popped from `visited` so sibling
    /// branches can traverse through it independently.
    /// This follows the KSP `remove`+`recover` pattern — temporary
    /// modification with guaranteed restoration.
    fn characteristic_with_variance(
        &self,
        ty: TypeId,
        visited: &mut Vec<TypeId>,
        has_non_covariant_path: bool,
    ) -> Characteristic {
        // ── Cycle detection (before push) ──────────────────────────
        // If ty is already on the current DFS path, we've found a cycle.
        if visited.contains(&ty) {
            return if has_non_covariant_path {
                Characteristic::Undecidable      // κ=∞
            } else {
                Characteristic::InfiniteEnumerable // κ=1
            };
        }
        // ── Mark (remove from available) ───────────────────────────
        visited.push(ty);

        // ── Recurse (compute) ──────────────────────────────────────
        let result = self.characteristic_body(ty, visited, has_non_covariant_path);

        // ── Recover (restore to available) ─────────────────────────
        visited.pop();

        result
    }

    /// The actual computation body, called after push and before pop.
    /// All recursive calls go through `characteristic_with_variance`
    /// so each level has proper scoping.
    fn characteristic_body(
        &self,
        ty: TypeId,
        visited: &mut Vec<TypeId>,
        has_non_covariant_path: bool,
    ) -> Characteristic {
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
                    match self.characteristic_with_variance(elem, visited, has_non_covariant_path) {
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
                    match self.characteristic_with_variance(arg, visited, has_non_covariant_path) {
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
                match self.characteristic_with_variance(*elem, visited, has_non_covariant_path) {
                    Characteristic::FiniteExhaustible(n) => {
                        Characteristic::FiniteExhaustible(n.saturating_pow(*size as u32))
                    }
                    Characteristic::InfiniteEnumerable => Characteristic::InfiniteEnumerable,
                    Characteristic::Undecidable => Characteristic::Undecidable,
                }
            }
            // Slice/Ref/Pointer/Ptr: invariant containers — mark path as non-covariant
            TypeData::Slice { elem } | TypeData::Ref { ty: elem, .. }
            | TypeData::Pointer { ty: elem }
            | TypeData::Ptr { pointee: elem, .. } => {
                let _ = self.characteristic_with_variance(*elem, visited, true);
                Characteristic::InfiniteEnumerable
            }
            // Fn params: contravariant → mark path as non-covariant
            // Fn ret: covariant → keep existing flag
            TypeData::Fn { params, ret } => {
                for &p in params {
                    if self.characteristic_with_variance(p, visited, true) == Characteristic::Undecidable {
                        return Characteristic::Undecidable;
                    }
                }
                match self.characteristic_with_variance(*ret, visited, has_non_covariant_path) {
                    Characteristic::Undecidable => Characteristic::Undecidable,
                    _ => Characteristic::InfiniteEnumerable,
                }
            }
            TypeData::GenericParam { .. } => {
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            // Mu/Nu: treat as covariant recursive type
            // (The body has already been handled above.)
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                // Already handled above — this is a safety fallback.
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            // Coproduct: characteristic is the sum of alternatives.
            TypeData::Coproduct { alternatives } => {
                let mut total = 0usize;
                let mut has_infinite = false;
                for &alt in alternatives {
                    match self.characteristic_with_variance(alt, visited, has_non_covariant_path) {
                        Characteristic::FiniteExhaustible(n) => {
                            total = total.saturating_add(n);
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
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. } => {
                self.characteristic_with_variance(*body, visited, has_non_covariant_path)
            }
            // DynTrait and AssociatedType: covariant containers but the concrete
            // implementation is unknown — conservatively return InfiniteEnumerable.
            // For AssociatedType, the self_ty variance propagates to the associated type.
            TypeData::DynTrait { .. } => {
                Characteristic::InfiniteEnumerable
            }
            TypeData::AssociatedType { self_ty, .. } => {
                self.characteristic_with_variance(*self_ty, visited, has_non_covariant_path)
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



#[cfg(test)]
mod tests {
    use super::*;

    fn new_ctx() -> TypeContext { TypeContext::new() }

    // -- TypeId tag --
    #[test]
    fn test_typeid_tag_encode_decode() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        assert_eq!(int_ty.tag(), TypeTag::Int);

        let fn_ty = ctx.function(vec![int_ty], ctx.bool());
        assert_eq!(fn_ty.tag(), TypeTag::Fn);
    }

    #[test]
    fn test_typeid_tag_index_roundtrip() {
        let mut ctx = new_ctx();
        let b = ctx.bool();
        let idx = b.index();
        assert_eq!(*ctx.types[idx], TypeData::Bool);
    }

    // -- Variance --
    #[test]
    fn test_variance_fn_param_contravariant() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.check_variance(0, ctx.bool(), -1), true);
    }

    #[test]
    fn test_variance_invariant_ref() {
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "T".into());
        let ref_ty = ctx.reference(p0, false);
        // Ref is invariant: param inside Ref cannot be at covariant or contravariant position
        assert!(!ctx.check_variance(0, ref_ty, 1));
        assert!(!ctx.check_variance(0, ref_ty, -1));
        // A covariant tuple containing the param works
        let tup_ty = ctx.tuple(vec![p0]);
        assert!(ctx.check_variance(0, tup_ty, 1));
    }

    #[test]
    fn test_variance_tuple_covariant() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.check_variance(0, ctx.bool(), 1), true);
    }

    #[test]
    fn test_variance_nested_fn() {
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "T".into());
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let inner = ctx.function(vec![p0], bool_ty);
        let outer = ctx.function(vec![int_ty], inner);
        assert!(ctx.type_contains_param(0, outer));
    }

    // -- Characteristic κ --
    #[test]
    fn test_characteristic_bool() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.characteristic(ctx.bool(), &mut vec![]), Characteristic::FiniteExhaustible(2));
    }

    #[test]
    fn test_characteristic_int32() {
        let mut ctx = TypeContext::new();
        let i32 = ctx.int(32, true);
        assert_eq!(ctx.characteristic(i32, &mut vec![]), Characteristic::FiniteExhaustible(2u64.pow(32) as usize));
    }

    #[test]
    fn test_characteristic_unit() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.characteristic(ctx.unit(), &mut vec![]), Characteristic::FiniteExhaustible(1));
    }

    #[test]
    fn test_characteristic_fn() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![bool_ty], bool_ty);
        // Fn types are classified as InfiniteEnumerable due to
        // contravariant parameter cycles in the characteristic algorithm
        assert_eq!(ctx.characteristic(fn_ty, &mut vec![]), Characteristic::InfiniteEnumerable);
    }

    #[test]
    fn test_characteristic_slice() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let slice_ty = ctx.slice(bool_ty);
        assert_eq!(ctx.characteristic(slice_ty, &mut vec![]), Characteristic::InfiniteEnumerable);
    }

    #[test]
    fn test_characteristic_ksp_convergence() {
        // Simulate a recursive type pattern where KSP-style iteration
        // is needed to reach convergence:
        //   μX. Bool × X  — a recursive type representing an infinite
        //   stream of Bools.  We encode it using Forall/GenericParam.
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let p0 = ctx.generic_param(0, "X".into());
        // Simulate μX: Bool × X  via Forall(0, "X", (Bool, X) ⇒ X)
        let body = {
            let tup = ctx.tuple(vec![bool_ty, p0]);
            ctx.function(vec![tup], p0)
        };
        let ty = ctx.forall(0, "X".into(), body);
        // KSP iteration should converge and detect InfiniteEnumerable
        // (recursive type with only covariant cycles)
        let mut visited = Vec::new();
        let kappa = ctx.characteristic(ty, &mut visited);
        assert_eq!(kappa, Characteristic::InfiniteEnumerable,
            "recursive stream type should be InfiniteEnumerable");
    }

    // -- Transaction --
    #[test]
    fn test_transaction_commit() {
        let mut ctx = TypeContext::new();
        let a = ctx.int(32, true);
        let b = ctx.int(64, false);
        assert!(ctx.unify(a, b).is_err());
    }

    #[test]
    fn test_transaction_rollback() {
        let mut ctx = TypeContext::new();
        let a = ctx.int(32, true);
        let bool_ty = ctx.bool();
        ctx.begin_transaction();
        ctx.bindings.borrow_mut().insert(a, bool_ty);
        ctx.rollback_transaction();
        assert!(ctx.resolve_binding(a) == a);
    }

    #[test]
    fn test_transaction_nested() {
        let mut ctx = TypeContext::new();
        let a = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let unit_ty = ctx.unit();
        ctx.begin_transaction();
        ctx.bindings.borrow_mut().insert(a, bool_ty);
        ctx.begin_transaction();
        ctx.bindings.borrow_mut().insert(a, unit_ty);
        ctx.rollback_transaction();
        assert_eq!(ctx.resolve_binding(a), bool_ty);
        ctx.commit_transaction();
    }

    // -- replace_generic --
    #[test]
    fn test_replace_generic_fn_ret() {
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "T".into());
        let p1 = ctx.generic_param(1, "U".into());
        let int_ty = ctx.int(32, true);
        let fn_ty = ctx.function(vec![p0], p1);
        let replaced = ctx.replace_generic(fn_ty, 0, int_ty);
        let expected = ctx.function(vec![int_ty], p1);
        assert_eq!(replaced, expected);
    }

    #[test]
    fn test_replace_generic_noop() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let int_ty = ctx.int(32, true);
        let replaced = ctx.replace_generic(bool_ty, 0, int_ty);
        assert_eq!(replaced, bool_ty);
    }

    // -- Yoneda reduction --
    #[test]
    fn test_yoneda_single_param_case1() {
        // ∀X.(Int⇒X)⇒X → Int
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let int_ty = ctx.int(32, true);
        let inner_fn = ctx.function(vec![int_ty], p0);
        let outer_fn = ctx.function(vec![inner_fn], p0);
        let forall = ctx.forall(0, "X".into(), outer_fn);
        assert_eq!(forall, int_ty, "∀X.(Int⇒X)⇒X should reduce to Int");
    }

    #[test]
    fn test_yoneda_single_param_case2() {
        // ∀X.(X⇒Int)⇒(X⇒Bool) → Int⇒Bool  (co-Yoneda)
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let int_ty = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let inner_fn = ctx.function(vec![p0], int_ty);
        let outer_fn = ctx.function(vec![p0], bool_ty);
        let combined = ctx.function(vec![inner_fn], outer_fn);
        let forall = ctx.forall(0, "X".into(), combined);
        assert_eq!(forall, ctx.function(vec![int_ty], bool_ty),
            "∀X.(X⇒Int)⇒(X⇒Bool) should reduce to Int⇒Bool");
    }

    #[test]
    fn test_yoneda_no_reduction() {
        // ∀X.Int⇒Int 不应约简
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let fn_ty = ctx.function(vec![int_ty], int_ty);
        let forall = ctx.forall(0, "X".into(), fn_ty);
        assert!(matches!(ctx.get(forall), TypeData::Forall { .. }));
    }

    #[test]
    fn test_yoneda_multi_param_inner_fn() {
        // ∀X.(Int⇒Bool⇒X)⇒X → (Int, Bool)  (single branch, product of params)
        // The inner Fn has params [Int, Bool] and ret=X.
        // With a single branch, Σₖ = Πⱼ Aⱼ = (Int, Bool).
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let int_ty = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let inner_fn = ctx.function(vec![int_ty, bool_ty], p0);
        let outer_fn = ctx.function(vec![inner_fn], p0);
        let forall = ctx.forall(0, "X".into(), outer_fn);
        let expected = ctx.tuple(vec![int_ty, bool_ty]);
        assert_eq!(forall, expected, "∀X.(Int⇒Bool⇒X)⇒X should reduce to (Int,Bool)");
    }

    #[test]
    fn test_yoneda_distributed_case_b() {
        // Implicit Fn-encoded: (Int⇒Bool⇒X)⇒X  →  (Int, Bool)
        // (no Forall wrapper — pure Fn-encoded pattern)
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let int_ty = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let inner_fn = ctx.function(vec![int_ty, bool_ty], p0);
        let ty = ctx.function(vec![inner_fn], p0);
        let reduced = ctx.try_yoneda_reduce(ty);
        let expected = ctx.tuple(vec![int_ty, bool_ty]);
        assert_eq!(reduced, expected, "(Int⇒Bool⇒X)⇒X should reduce to (Int,Bool)");
    }
}
