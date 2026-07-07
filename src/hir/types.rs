use std::cell::Cell;
use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;
use std::collections::HashSet;
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
    Tuple = 7,
    Array = 8,
    Slice = 9,
    Ref = 10,
    Pointer = 11,
    Ptr = 12,
    Fn = 13,
    DynTrait = 14,
    Exists = 15,
    Forall = 16,
    GenericParam = 17,
    AssociatedType = 18,
    InferVar = 19,
    Never = 20,
    Unit = 21,
    Error = 22,
    Coproduct = 23,
    Mu = 24,
    Nu = 25,
    Poly = 26,
    Rational = 27,
    /// Algebraic data type — a struct, enum, or other named type applied
    /// to its generic arguments.  Follows rustc's single-`Adt` convention
    /// rather than separate `Struct`/`Enum` variants.
    Adt = 28,
    SkolemVar = 29,
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
            TypeData::Adt { .. } => TypeTag::Adt,
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
            TypeData::Poly { .. } => TypeTag::Poly,
            TypeData::Rational { .. } => TypeTag::Rational,
            TypeData::SkolemVar { .. } => TypeTag::SkolemVar,
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
        // and TAG_MASK covers all 31 discriminants (0..30).
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
    /// An algebraic data type (ADT): struct, enum, or other named type
    /// applied to its generic arguments.  Rustc-style single variant for
    /// all named types: `Adt(def_id, [args...])`.
    /// When the type has no generic parameters, `args` is empty.
    /// Examples:
    ///   `String`          → Adt { def_id: StringDefId, args: [] }
    ///   `Option<Int<32>>` → Adt { def_id: OptionDefId, args: [Int<32>] }
    Adt {
        kind: AdtKind,
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
    /// A polytype: `[∀ᾱ. τ]` — a boxed first-class polymorphic type.
    /// `quantifiers` lists the universally quantified variables as (index, name) pairs.
    /// `body` is the inner type, referencing quantifiers via `GenericParam`.
    /// See OmniML §3.1 (O'Brien, Rémy & Scherer).
    Poly {
        quantifiers: Vec<(usize, String)>,
        body: TypeId,
    },
    /// Fixed-precision rational type: `Rational<p, q>`.
    /// `int_bits` = number of integer bits (p), `frac_bits` = number of fractional bits (q).
    /// Arithmetic is exact over the rational domain for contracts.
    /// Default overflow policy is `saturate`.
    Rational {
        int_bits: u8,
        frac_bits: u8,
    },
    SkolemVar {
        id: usize,
        universe_num: usize,
    },
    Never,
    Unit,
    Error,
}

/// Distinguishes between struct and enum ADT kinds (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdtKind {
    Struct,
    Enum,
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

/// Variance for type unification: controls how subtyping propagates
/// through compound types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variance {
    /// T <: U — the type is in a covariant position (e.g. function return, tuple element).
    Covariant,
    /// T :> U (i.e. U <: T) — the type is in a contravariant position (e.g. function parameter).
    Contravariant,
    /// T == U — strict equality required (default for unification).
    Invariant,
}

impl Variance {
    /// Transform variance when going through a position of `self` variance.
    /// For example, if we are in an Invariant context and encounter a Covariant
    /// position (Fn return), the result is Invariant * Covariant = Covariant.
    /// If we are in a Covariant context and encounter a Contravariant position
    /// (Fn parameter), the result is Covariant * Contravariant = Contravariant.
    pub fn xform(self, position: Variance) -> Variance {
        match (self, position) {
            (Variance::Invariant, _) => position,
            (Variance::Covariant, Variance::Covariant) => Variance::Covariant,
            (Variance::Covariant, Variance::Contravariant) => Variance::Contravariant,
            (Variance::Covariant, Variance::Invariant) => Variance::Invariant,
            (Variance::Contravariant, Variance::Covariant) => Variance::Contravariant,
            (Variance::Contravariant, Variance::Contravariant) => Variance::Covariant,
            (Variance::Contravariant, Variance::Invariant) => Variance::Invariant,
        }
    }
}

#[derive(Clone)]
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
    /// Built-in string slice type `Str`.
    pub builtin_str: TypeId,
    /// Built-in reference to string slice `&Str` — a `Ref { ty: Str, mutable: false }`.
    pub builtin_str_ref: TypeId,
    /// Cache for variance check results: (param_index, TypeId, expected_sign, cumulative_sign) → bool.
    variance_cache: RefCell<HashMap<(usize, TypeId, isize, isize), bool>>,
    /// Pre-computed variance-annotated outgoing edges for each TypeId.
    /// Built lazily on first variance check, then reused.
    variance_edges: RefCell<HashMap<TypeId, Vec<VarianceEdge>>>,
    /// Transaction stack for atomic unification (OmniML-style rollback via undo log).
    /// Each entry is a list of (key, old_value) pairs recording every binding
    /// change made during that transaction.  On rollback the changes are undone
    /// in reverse order; on commit the log is discarded.
    /// This is O(changes) instead of O(total_bindings) — a significant saving
    /// when the binding table is large and transactions are frequent.
    transaction_stack: RefCell<Vec<Vec<(TypeId, Option<TypeId>)>>>,
    /// Cache for unification with variance: prevents infinite recursion on
    /// self-referential types.  Keyed by (a, b, variance_tag) where
    /// variance_tag = 0 for Invariant, 1 for Covariant, 2 for Contravariant.
    unify_seen: RefCell<HashSet<(TypeId, TypeId, u8)>>,
    /// Cache for κ(A) characteristic results.  Cleared when bindings change.
    kappa_cache: RefCell<HashMap<TypeId, Characteristic>>,
    /// Universe counter for Higher-Ranked Type skolemization (rustc-style).
    /// Each `for<'a>` binder comparison enters a fresh universe.
    next_universe: Cell<usize>,
    /// Pre-allocated pool of skolem placeholder types for HRTB comparison.
    skolem_pool: RefCell<Vec<TypeId>>,
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
            builtin_str: TypeId(0),
            builtin_str_ref: TypeId(0),
            variance_cache: RefCell::new(HashMap::default()),
            variance_edges: RefCell::new(HashMap::default()),
            transaction_stack: RefCell::new(Vec::new()),
            unify_seen: RefCell::new(HashSet::default()),
            kappa_cache: RefCell::new(HashMap::default()),
            next_universe: Cell::new(0),
            skolem_pool: RefCell::new(Vec::new()),
        };
        ctx.builtin_unit = ctx.alloc(TypeData::Unit);
        ctx.builtin_never = ctx.alloc(TypeData::Never);
        ctx.builtin_error = ctx.alloc(TypeData::Error);
        ctx.builtin_bool = ctx.alloc(TypeData::Bool);
        ctx.builtin_char = ctx.alloc(TypeData::Char);
        ctx.builtin_byte = ctx.alloc(TypeData::Byte);
        ctx.builtin_usize = ctx.alloc(TypeData::USize);
        // Str type: represented as a zero-sized struct with a sentinel DefId.
        ctx.builtin_str = ctx.alloc(TypeData::Adt {
            kind: AdtKind::Struct,
            def_id: DefId(usize::MAX),
            args: vec![],
        });
        // &Str = Ref { ty: Str, mutable: false }
        ctx.builtin_str_ref = ctx.reference(ctx.builtin_str, false);
        let mut pool = Vec::new();
        for i in 0..16 {
            let skolem = ctx.alloc(TypeData::SkolemVar { id: i, universe_num: i });
            pool.push(skolem);
        }
        ctx.skolem_pool = RefCell::new(pool);
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

    /// Returns an `Arc<TypeData>` instead of a borrow, enabling cheap clone via
    /// `Arc::clone` (reference-count bump only).  Use this instead of
    /// `self.get(ty).clone()` on hot paths (substitution, Yoneda reduction,
    /// unification) to avoid deep-copying `Vec<TypeId>` and `String` fields.
    pub fn get_arc(&self, id: TypeId) -> Arc<TypeData> {
        let resolved = self.resolve_binding(id);
        Arc::clone(&self.types[resolved.index()])
    }

    pub fn is_infer_var(&self, id: TypeId) -> bool {
        matches!(self.get(id), TypeData::InferVar { .. })
    }

    pub(crate) fn resolve_binding(&self, id: TypeId) -> TypeId {
        // Safety: guard against infinite loops from circular bindings.
        // 10 000 is generous enough for any real program while preventing
        // a maliciously constructed chain from DoS-ing the compiler.
        const MAX_CHAIN_DEPTH: usize = 10_000;

        // First pass: follow the binding chain to the root with a single
        // immutable borrow.  This is a simple linked-list traversal through
        // the bindings map until we reach an unbound TypeId.
        let root = {
            let bindings = self.bindings.borrow();
            let mut current = id;
            let mut depth = 0;
            while let Some(&next) = bindings.get(&current) {
                current = next;
                depth += 1;
                if depth > MAX_CHAIN_DEPTH {
                    // Cycle detected — break and return what we have.
                    break;
                }
            }
            current
        };

        // Second pass: path compression.  Point every node along the chain
        // directly to the root so that future lookups are O(1) instead of
        // O(depth).  Uses a single mutable borrow for all updates.
        if root != id {
            let mut bindings = self.bindings.borrow_mut();
            let mut current = id;
            let mut depth = 0;
            while current != root {
                if let Some(&next) = bindings.get(&current) {
                    bindings.insert(current, root);
                    current = next;
                    depth += 1;
                    if depth > MAX_CHAIN_DEPTH {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        root
    }

    pub fn alloc_infer_var(&mut self, id: usize) -> TypeId {
        self.alloc(TypeData::InferVar { id })
    }

    pub fn get_def_id_for_type(&self, id: TypeId) -> Option<DefId> {
        let resolved = self.resolve_binding(id);
        match &self.types[resolved.index()].as_ref() {
            TypeData::Adt { def_id, .. } => Some(*def_id),
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

    pub fn str_ref(&self) -> TypeId {
        self.builtin_str_ref
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
        let id = self.alloc(TypeData::Adt { kind: AdtKind::Struct, def_id, args });
        self.def_id_to_type_id.insert(def_id, id);
        id
    }

    pub fn enum_ty(&mut self, def_id: DefId, args: Vec<TypeId>) -> TypeId {
        let id = self.alloc(TypeData::Adt { kind: AdtKind::Enum, def_id, args });
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

    /// Allocate a polytype `[∀ᾱ. τ]` — a boxed first-class polymorphic type.
    /// `quantifiers` are (index, name) pairs for universally quantified variables.
    /// `body` references them via `GenericParam`.
    pub fn poly(&mut self, quantifiers: Vec<(usize, String)>, body: TypeId) -> TypeId {
        self.alloc(TypeData::Poly { quantifiers, body })
    }

    pub fn rational(&mut self, int_bits: u8, frac_bits: u8) -> TypeId {
        self.alloc(TypeData::Rational {
            int_bits,
            frac_bits,
        })
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
        // Limit iterations to prevent DoS from maliciously constructed types.
        // In practice, Yoneda/co-Yoneda reduction converges in ≤3 iterations
        // because each pass either eliminates a Forall node or reaches a
        // fixed point.  10 is a generous safety ceiling.
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
        let ty_data = self.get_arc(ty);
        if let TypeData::Forall {
            param_index,
            param_name: _,
            body,
        } = &*ty_data
        {
            let body_data = self.get_arc(*body);
            if let TypeData::Fn { params, ret } = &*body_data {
                let pi = *param_index;
                let ret = *ret;
                let mut branch_replacements: Vec<TypeId> = Vec::new();
                let mut is_coyoneda = false;
                for &branch in params {
                    // Peel outer Forall layers (∀Z⃗ₖ).
                    let mut inner_quantifiers: Vec<(usize, String)> = Vec::new();
                    let mut inner = branch;
                    loop {
                        let inner_data = self.get_arc(inner);
                        match &*inner_data {
                            TypeData::Forall {
                                param_index: fi,
                                param_name: fn_,
                                body: b,
                            } => {
                                inner_quantifiers.push((*fi, fn_.clone()));
                                inner = *b;
                            }
                            _ => break,
                        }
                    }
                    let inner_data = self.get_arc(inner);
                    if let TypeData::Fn {
                        params: ips,
                        ret: ir,
                    } = &*inner_data
                    {
                        // Check ≡_X (Yoneda): ir = GenericParam(pi)
                        let yoneda_match = match self.get(*ir) {
                            TypeData::GenericParam { index, .. } if *index == pi => true,
                            _ => false,
                        };
                        // Check ≡_X (co-Yoneda): ips[0] = GenericParam(pi)
                        let coyoneda_match = if !ips.is_empty() {
                            match self.get(ips[0]) {
                                TypeData::GenericParam { index, .. } if *index == pi => true,
                                _ => false,
                            }
                        } else {
                            false
                        };
                        // Process Yoneda case
                        if yoneda_match {
                            let product = if ips.len() == 1 {
                                ips[0]
                            } else {
                                self.tuple(ips.clone())
                            };
                            let repl = if inner_quantifiers.is_empty() {
                                product
                            } else {
                                let mut w = product;
                                for (eq, en) in &inner_quantifiers {
                                    w = self.exists(
                                        en.clone(),
                                        w,
                                        crate::ast::Expr::Literal(
                                            crate::ast::Literal::Bool(true),
                                            crate::ast::Span::new(0, 0),
                                        ),
                                    );
                                }
                                w
                            };
                            branch_replacements.push(repl);
                        }
                        // Process co-Yoneda case
                        if coyoneda_match {
                            is_coyoneda = true;
                            // The branch is X ⇒ A where A = inner_ret (not a param).
                            // When ips.len() == 1, X is the only param and A = inner_ret.
                            // When ips.len() > 1, the branch is X ⇒ A₁ ⇒ ... ⇒ Aⱼ ⇒ ir,
                            // and the replacement is A₁ ⇒ (A₂ ⇒ ... ⇒ (Aⱼ ⇒ ir)).
                            let replacement = if ips.len() <= 1 {
                                *ir
                            } else {
                                self.function(ips[1..].to_vec(), *ir)
                            };
                            let repl = if inner_quantifiers.is_empty() {
                                replacement
                            } else {
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
                    // Wrap with μX/νX only when the branch product(s) depend on X
                    // (Pistone & Tranchini 2022 §2, eq.3 & eq.4):
                    //   Yoneda (A⟨X⟩⇒X):    B⟨X⟩ → B⟨X↦μX.A⟨X⟩⟩
                    //   co-Yoneda (X⇒A⟨X⟩): B⟨X⟩ → B⟨X↦νX.A⟨X⟩⟩
                    // When A⟨X⟩ = Int (no X), no fixpoint needed:
                    //   ∀X.(Int⇒X)⇒B⟨X⟩  →  B⟨X↦Int⟩
                    let needs_fix = self.type_contains_param(pi, sigma);
                    let replacement = if needs_fix {
                        if is_coyoneda {
                            self.alloc(TypeData::Nu {
                                param_index: pi,
                                param_name: "X".into(),
                                body: sigma,
                            })
                        } else {
                            self.alloc(TypeData::Mu {
                                param_index: pi,
                                param_name: "X".into(),
                                body: sigma,
                            })
                        }
                    } else {
                        sigma
                    };
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
        let inner_data = self.get_arc(inner);
        if let TypeData::Fn {
            params: inner_params,
            ret: inner_ret,
        } = &*inner_data
        {
            // ≡_X (Yoneda): inner_ret is GenericParam X
            let yoneda_idx = match self.get(*inner_ret) {
                TypeData::GenericParam { index, .. } => Some(*index),
                _ => None,
            };
            if let Some(idx) = yoneda_idx {
                let replacement = if inner_params.len() == 1 {
                    inner_params[0]
                } else {
                    self.tuple(inner_params.clone())
                };
                return self.replace_generic(ret, idx, replacement);
            }
            // ≡_X (co-Yoneda): first inner param is GenericParam X
            if !inner_params.is_empty() {
                let coyoneda_idx = match self.get(inner_params[0]) {
                    TypeData::GenericParam { index, .. } => Some(*index),
                    _ => None,
                };
                if let Some(idx) = coyoneda_idx {
                    return self.replace_generic(ret, idx, *inner_ret);
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
        let cache_key = (param, ty, expected_sign, cumulative_sign);
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
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            -cumulative_sign,
                        ) {
                            return false;
                        }
                    }
                    0 => {
                        // Invariant: param cannot appear
                        return false;
                    }
                    _ => {
                        // Covariant: keep cumulative sign
                        if !self.check_variance_with_sign(
                            param,
                            edge.target,
                            expected_sign,
                            cumulative_sign,
                        ) {
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
                let mut edges: Vec<VarianceEdge> = params
                    .iter()
                    .map(|&p| VarianceEdge {
                        target: p,
                        sign: -1,
                    })
                    .collect();
                edges.push(VarianceEdge {
                    target: *ret,
                    sign: 1,
                });
                edges
            }
            TypeData::Adt { args, .. } => args
                .iter()
                .map(|&a| VarianceEdge { target: a, sign: 0 }) // invariant — nominal types have invariant params
                .collect(),
            TypeData::Tuple { elems } => elems
                .iter()
                .map(|&e| VarianceEdge { target: e, sign: 1 })
                .collect(),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                vec![VarianceEdge {
                    target: *elem,
                    sign: 1,
                }]
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                vec![VarianceEdge {
                    target: *ty,
                    sign: 0,
                }]
            }
            TypeData::Ptr { pointee, .. } => {
                vec![VarianceEdge {
                    target: *pointee,
                    sign: 0,
                }]
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. }
            | TypeData::Poly { body, .. } => {
                vec![VarianceEdge {
                    target: *body,
                    sign: 1,
                }]
            }
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .map(|&a| VarianceEdge { target: a, sign: 1 })
                .collect(),
            TypeData::AssociatedType { self_ty, .. } => {
                vec![VarianceEdge {
                    target: *self_ty,
                    sign: 1,
                }]
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
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.type_contains_param(param, a)),
            TypeData::Tuple { elems } => elems.iter().any(|&e| self.type_contains_param(param, e)),
            TypeData::Coproduct { alternatives } => alternatives
                .iter()
                .any(|&a| self.type_contains_param(param, a)),
            TypeData::Array { elem, .. } | TypeData::Slice { elem } => {
                self.type_contains_param(param, *elem)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                self.type_contains_param(param, *ty)
            }
            TypeData::Ptr { pointee, .. } => self.type_contains_param(param, *pointee),
            TypeData::AssociatedType { self_ty, .. } => self.type_contains_param(param, *self_ty),
            TypeData::Poly { body, .. } => self.type_contains_param(param, *body),
            TypeData::Exists { base, .. } => self.type_contains_param(param, *base),
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
        if reduced != id { reduced } else { id }
    }

    /// Skip the `subst` type-pool lookup limitations and directly build
    /// the replacement type.  This avoids the `fn_ty_no_alloc().expect()`
    /// panic that occurs when `subst` tries to find a pre-existing type
    /// that hasn't been created yet.
    pub fn replace_generic(
        &mut self,
        ty: TypeId,
        param_index: usize,
        replacement: TypeId,
    ) -> TypeId {
        if !self.type_contains_param(param_index, ty) {
            return ty;
        }
        let data = self.get_arc(ty);
        match &*data {
            TypeData::GenericParam { index, .. } if *index == param_index => replacement,
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> = params
                    .iter()
                    .map(|&p| self.replace_generic(p, param_index, replacement))
                    .collect();
                let new_ret = self.replace_generic(*ret, param_index, replacement);
                self.function(new_params, new_ret)
            }
            TypeData::Forall {
                param_index: pi,
                param_name,
                body,
            } => {
                let new_body = self.replace_generic(*body, param_index, replacement);
                self.forall(*pi, param_name.clone(), new_body)
            }
            TypeData::Mu {
                param_index: pi,
                param_name,
                body,
            } => {
                let new_body = self.replace_generic(*body, param_index, replacement);
                self.alloc(TypeData::Mu {
                    param_index: *pi,
                    param_name: param_name.clone(),
                    body: new_body,
                })
            }
            TypeData::Nu {
                param_index: pi,
                param_name,
                body,
            } => {
                let new_body = self.replace_generic(*body, param_index, replacement);
                self.alloc(TypeData::Nu {
                    param_index: *pi,
                    param_name: param_name.clone(),
                    body: new_body,
                })
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems
                    .iter()
                    .map(|&e| self.replace_generic(e, param_index, replacement))
                    .collect();
                self.tuple(new_elems)
            }
            TypeData::Adt { kind, def_id, args } => {
                let new_args: Vec<TypeId> = args
                    .iter()
                    .map(|&a| self.replace_generic(a, param_index, replacement))
                    .collect();
                self.alloc(TypeData::Adt {
                    kind: *kind,
                    def_id: *def_id,
                    args: new_args,
                })
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: Vec<TypeId> = alternatives
                    .iter()
                    .map(|&a| self.replace_generic(a, param_index, replacement))
                    .collect();
                if new_alts.len() == 1 {
                    new_alts[0]
                } else {
                    self.alloc(TypeData::Coproduct {
                        alternatives: new_alts,
                    })
                }
            }
            TypeData::Poly { quantifiers, body } => {
                let new_body = self.replace_generic(*body, param_index, replacement);
                self.poly(quantifiers.clone(), new_body)
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

    /// Check whether `param` occurs inside `ty` (the "occurs check").
    ///
    /// # Why no `visited` set is needed
    ///
    /// The `types` arena (`Vec<Arc<TypeData>>`) is physically a DAG — every
    /// `TypeData` is allocated before any cycles could exist, and the only
    /// way to form a cycle is through the `bindings` table.  Since this
    /// function calls `self.resolve_binding(ty)` first, the incoming `ty`
    /// is already dereferenced past any binding chain, making the recursive
    /// walk of the type structure **acyclic by construction**.
    ///
    /// A naive reader might be tempted to add a `visited: HashSet<TypeId>`
    /// to guard against infinite recursion.  **Do not.**  It would add O(n)
    /// memory overhead and mask the fact that the real cycle-safety proof
    /// lives upstream, in the binding layer.
    fn occurs_check(&self, param: TypeId, ty: TypeId) -> bool {
        if param == ty {
            return true;
        }
        let resolved = self.resolve_binding(ty);
        match &self.types[resolved.index()].as_ref() {
            TypeData::Adt { args, .. } => args.iter().any(|&a| self.occurs_check(param, a)),
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
            TypeData::Poly { body, .. } => self.occurs_check(param, *body),
            TypeData::Exists { base, .. } => self.occurs_check(param, *base),
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => self.occurs_check(param, *body),
            TypeData::AssociatedType { self_ty, .. } => self.occurs_check(param, *self_ty),
            TypeData::GenericParam { .. } | TypeData::InferVar { .. } => false,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::DynTrait { .. }
            | TypeData::SkolemVar { .. } => false,
        }
    }

    pub fn unify(&self, a: TypeId, b: TypeId) -> Result<TypeId, TypeError> {
        // ── Transaction: capture current bindings for rollback ──
        self.begin_transaction();

        // Clear the seen-set before each top-level unification.
        self.unify_seen.borrow_mut().clear();

        let result = self.unify_internal(a, b, Variance::Invariant);

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

    fn variance_tag(v: Variance) -> u8 {
        match v {
            Variance::Invariant => 0,
            Variance::Covariant => 1,
            Variance::Contravariant => 2,
        }
    }

    /// Internal unification with variance-aware subtyping.
    /// Recursively decomposes compound types and unifies sub-components
    /// according to the given variance.
    ///
    /// Variance propagation rules:
    /// - Invariant: all sub-components unified with Invariant (strict equality)
    /// - Covariant (T <: U): sub-components in covariant positions keep Covariant,
    ///   those in contravariant positions flip to Contravariant
    /// - Contravariant (T :> U): sub-components in covariant positions flip to
    ///   Contravariant, those in contravariant positions flip to Covariant
    fn unify_internal(
        &self,
        a: TypeId,
        b: TypeId,
        variance: Variance,
    ) -> Result<TypeId, TypeError> {
        // ── Caching: skip if we've already checked this (a, b, variance) pair ──
        let tag = Self::variance_tag(variance);
        let key = (a, b, tag);
        if !self.unify_seen.borrow_mut().insert(key) {
            // Already visited this pair — assume success to break cycles.
            return Ok(a);
        }

        let result = self.unify_internal_impl(a, b, variance);

        // On error, remove the cache entry so future attempts can retry.
        if result.is_err() {
            self.unify_seen.borrow_mut().remove(&key);
        }
        result
    }

    /// The actual unification logic, called by `unify_internal` which wraps
    /// it with cache management.
    fn unify_internal_impl(
        &self,
        a: TypeId,
        b: TypeId,
        variance: Variance,
    ) -> Result<TypeId, TypeError> {
        let data_a = self.get_arc(a);
        let data_b = self.get_arc(b);

        if *data_a == *data_b {
            return Ok(a);
        }

        match (&*data_a, &*data_b) {
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
                self.set_binding(a, b);
                Ok(b)
            }
            (_, TypeData::GenericParam { .. }) => {
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.set_binding(b, a);
                Ok(a)
            }
            (TypeData::InferVar { .. }, _) => {
                if self.occurs_check(a, b) {
                    return Err(TypeError::RecursiveType {
                        ty: a,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.set_binding(a, b);
                Ok(b)
            }
            (_, TypeData::InferVar { .. }) => {
                if self.occurs_check(b, a) {
                    return Err(TypeError::RecursiveType {
                        ty: b,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                self.set_binding(b, a);
                Ok(a)
            }

            // ── Compound types: same variant, recursive sub-component unification ──

            // Adt (struct/enum): same def_id, same args length, unify args pairwise (invariant).
            (
                TypeData::Adt {
                    kind: _,
                    def_id: d1,
                    args: a1,
                },
                TypeData::Adt {
                    kind: _,
                    def_id: d2,
                    args: a2,
                },
            ) if d1 == d2 && a1.len() == a2.len() => {
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify_internal(*t1, *t2, Variance::Invariant)?;
                }
                self.set_binding(a, b);
                Ok(b)
            }

            // Tuple: same length, elements are COVARIANT
            (TypeData::Tuple { elems: e1 }, TypeData::Tuple { elems: e2 })
                if e1.len() == e2.len() =>
            {
                let elem_variance = variance.xform(Variance::Covariant);
                for (t1, t2) in e1.iter().zip(e2.iter()) {
                    self.unify_internal(*t1, *t2, elem_variance)?;
                }
                self.set_binding(a, b);
                Ok(b)
            }

            // Function: params are CONTRAVARIANT, return is COVARIANT
            (
                TypeData::Fn {
                    params: p1,
                    ret: r1,
                },
                TypeData::Fn {
                    params: p2,
                    ret: r2,
                },
            ) if p1.len() == p2.len() => {
                let param_variance = variance.xform(Variance::Contravariant);
                for (t1, t2) in p1.iter().zip(p2.iter()) {
                    self.unify_internal(*t1, *t2, param_variance)?;
                }
                let ret_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*r1, *r2, ret_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Array: same size, element is COVARIANT
            (TypeData::Array { elem: e1, size: s1 }, TypeData::Array { elem: e2, size: s2 })
                if s1 == s2 =>
            {
                let elem_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*e1, *e2, elem_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Slice: element is COVARIANT
            (TypeData::Slice { elem: e1 }, TypeData::Slice { elem: e2 }) => {
                let elem_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*e1, *e2, elem_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Ref: pointee is INVARIANT (per compute_variance_edges signing it sign: 0).
            // MUTABILITY:
            // - &mut T <: &T allowed in Covariant direction (borrow shortening)
            // - &T <: &mut T NEVER allowed
            //
            // NOTE on mutable subtyping: this language permits &mut T <: &T in
            // covariant contexts (a "borrow shortening" rule).  This is NOT the
            // same as Rust's semantics where &mut T is invariant; it is a
            // deliberate design choice to support safe temporary reborrowing.
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
                let allow_mutable_coerce = match variance {
                    Variance::Invariant => *m1 == *m2,
                    Variance::Covariant => !(*m1 == false && *m2 == true),
                    Variance::Contravariant => !(*m2 == false && *m1 == true),
                };
                if !allow_mutable_coerce {
                    return Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: crate::ast::Span::new(0, 0),
                    });
                }
                let ty_variance = variance.xform(Variance::Invariant);
                self.unify_internal(*t1, *t2, ty_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Pointer: INVARIANT (per compute_variance_edges signing it sign: 0).
            // While some languages treat raw pointers as covariant, this design
            // conservatively marks them invariant for type safety.
            (TypeData::Pointer { ty: t1 }, TypeData::Pointer { ty: t2 }) => {
                let ty_variance = variance.xform(Variance::Invariant);
                self.unify_internal(*t1, *t2, ty_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Ptr: invariant for safety
            (
                TypeData::Ptr {
                    size: s1,
                    pointee: p1,
                },
                TypeData::Ptr {
                    size: s2,
                    pointee: p2,
                },
            ) => {
                self.unify_internal(*s1, *s2, Variance::Invariant)?;
                self.unify_internal(*p1, *p2, Variance::Invariant)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Coproduct: same length, alternatives COVARIANT
            (
                TypeData::Coproduct { alternatives: a1 },
                TypeData::Coproduct { alternatives: a2 },
            ) if a1.len() == a2.len() => {
                let alt_variance = variance.xform(Variance::Covariant);
                for (t1, t2) in a1.iter().zip(a2.iter()) {
                    self.unify_internal(*t1, *t2, alt_variance)?;
                }
                self.set_binding(a, b);
                Ok(b)
            }

            // Forall: same param_index + body is COVARIANT
            (
                TypeData::Forall {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Forall {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) if pi1 == pi2 && pn1 == pn2 => {
                let body_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*b1, *b2, body_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Exists: same name + base is COVARIANT
            (TypeData::Exists { name: n1, base: b1 }, TypeData::Exists { name: n2, base: b2 })
                if n1 == n2 =>
            {
                let base_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*b1, *b2, base_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Poly: same quantifiers list length + content + body is COVARIANT
            (
                TypeData::Poly {
                    quantifiers: q1,
                    body: b1,
                },
                TypeData::Poly {
                    quantifiers: q2,
                    body: b2,
                },
            ) if q1.len() == q2.len() && q1.iter().zip(q2.iter()).all(|(a, b)| a == b) => {
                let body_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*b1, *b2, body_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Mu: same param_index + body is COVARIANT
            (
                TypeData::Mu {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Mu {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) if pi1 == pi2 && pn1 == pn2 => {
                let body_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*b1, *b2, body_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Nu: same param_index + body is COVARIANT
            (
                TypeData::Nu {
                    param_index: pi1,
                    param_name: pn1,
                    body: b1,
                },
                TypeData::Nu {
                    param_index: pi2,
                    param_name: pn2,
                    body: b2,
                },
            ) if pi1 == pi2 && pn1 == pn2 => {
                let body_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*b1, *b2, body_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // Rational: same int_bits and frac_bits (invariant)
            (
                TypeData::Rational {
                    int_bits: i1,
                    frac_bits: f1,
                },
                TypeData::Rational {
                    int_bits: i2,
                    frac_bits: f2,
                },
            ) if i1 == i2 && f1 == f2 => {
                self.set_binding(a, b);
                Ok(b)
            }

            // DynTrait: same trait list (invariant)
            (TypeData::DynTrait { traits: t1 }, TypeData::DynTrait { traits: t2 }) if t1 == t2 => {
                self.set_binding(a, b);
                Ok(b)
            }

            // AssociatedType: same trait_id + name, self_ty is COVARIANT
            (
                TypeData::AssociatedType {
                    trait_id: ti1,
                    name: n1,
                    self_ty: s1,
                },
                TypeData::AssociatedType {
                    trait_id: ti2,
                    name: n2,
                    self_ty: s2,
                },
            ) if ti1 == ti2 && n1 == n2 => {
                let self_variance = variance.xform(Variance::Covariant);
                self.unify_internal(*s1, *s2, self_variance)?;
                self.set_binding(a, b);
                Ok(b)
            }

            // ── Under non-Invariant variance, try subtype fallback ──
            _ if variance != Variance::Invariant => {
                let (sub, sup) = match variance {
                    Variance::Covariant => (a, b),
                    Variance::Contravariant => (b, a),
                    _ => unreachable!(),
                };
                if self.subtype(sub, sup) {
                    self.set_binding(a, b);
                    Ok(b)
                } else {
                    Err(TypeError::Mismatch {
                        expected: b,
                        found: a,
                        span: crate::ast::Span::new(0, 0),
                    })
                }
            }

            _ => Err(TypeError::Mismatch {
                expected: b,
                found: a,
                span: crate::ast::Span::new(0, 0),
            }),
        }
    }

    // ── Transaction support for atomic unification (Undo Log) ─────

    /// Begin a new transaction: push an empty undo log onto the stack.
    /// All subsequent binding changes (via `set_binding`) will be recorded
    /// for potential rollback, without cloning the entire binding table.
    pub fn begin_transaction(&self) {
        self.transaction_stack.borrow_mut().push(Vec::new());
    }

    /// Commit the current transaction: discard the undo log.
    pub fn commit_transaction(&self) {
        // Pop the current (innermost) transaction's undo log.
        let committed = self.transaction_stack.borrow_mut().pop();
        // Merge its entries into the parent transaction's log so that if
        // the parent later rolls back, it also undoes changes that were
        // committed by the inner transaction.
        //
        // Without this merge, the inner transaction's log is discarded on
        // commit, leaving the parent unaware of those changes.  A subsequent
        // parent rollback would then only undo the parent's own direct
        // changes, leaving the inner transaction's modifications in place
        // — a semantic mismatch with the original full-snapshot behaviour.
        if let Some(committed_log) = committed {
            if let Some(parent_log) = self.transaction_stack.borrow_mut().last_mut() {
                parent_log.extend(committed_log);
            }
        }
        // κ cache may be invalidated by binding changes across transaction boundaries.
        self.kappa_cache.borrow_mut().clear();
    }

    /// Rollback the current transaction: reverse-apply every binding change
    /// recorded in this transaction's undo log.
    /// Also clears the unification cache so subsequent attempts re-evaluate.
    pub fn rollback_transaction(&self) {
        if let Some(log) = self.transaction_stack.borrow_mut().pop() {
            let mut bindings = self.bindings.borrow_mut();
            for (key, old) in log.into_iter().rev() {
                match old {
                    Some(v) => bindings.insert(key, v),
                    None => bindings.remove(&key),
                };
            }
        }
        self.unify_seen.borrow_mut().clear();
        self.kappa_cache.borrow_mut().clear();
    }

    /// Insert a binding, recording the old value in the current transaction's
    /// undo log if one is active.  Always use this instead of
    /// `self.bindings.borrow_mut().insert(...)` so that transactions can
    /// correctly roll back.
    pub(crate) fn set_binding(&self, key: TypeId, value: TypeId) {
        if let Some(log) = self.transaction_stack.borrow_mut().last_mut() {
            let old = self.bindings.borrow().get(&key).copied();
            log.push((key, old));
        }
        self.bindings.borrow_mut().insert(key, value);
    }

    /// When `self_ty` resolves to a concrete ADT, return its `DefId`.
    /// Full projection resolution (finding the impl's concrete associated
    /// type) requires `TraitEnv` and is performed by the checker.
    pub fn try_normalize_associated_type_def_id(&self, self_ty: TypeId) -> Option<DefId> {
        let resolved = self.resolve_binding(self_ty);
        match self.get(resolved) {
            TypeData::Adt { def_id, .. } => Some(*def_id),
            _ => None,
        }
    }

    pub fn enter_universe(&self) -> (usize, TypeId) {
        let universe = self.next_universe.get();
        self.next_universe.set(universe + 1);
        let pool = self.skolem_pool.borrow();
        let idx = universe % pool.len();
        (universe, pool[idx])
    }

    pub fn check_skolem_escape(&self, ty: TypeId, max_universe: usize) -> Option<usize> {
        let resolved = self.resolve_binding(ty);
        match self.get(resolved) {
            TypeData::SkolemVar { universe_num, .. } if *universe_num > max_universe => Some(*universe_num),
            TypeData::Adt { args, .. } | TypeData::Tuple { elems: args, .. } | TypeData::Coproduct { alternatives: args, .. } => {
                for &a in args { if let Some(u) = self.check_skolem_escape(a, max_universe) { return Some(u); } }
                None
            }
            TypeData::Fn { params, ret } => {
                for &p in params { if let Some(u) = self.check_skolem_escape(p, max_universe) { return Some(u); } }
                self.check_skolem_escape(*ret, max_universe)
            }
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } | TypeData::Array { elem: ty, .. } | TypeData::Slice { elem: ty } | TypeData::Ptr { pointee: ty, .. } => self.check_skolem_escape(*ty, max_universe),
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. } | TypeData::Mu { body, .. } | TypeData::Nu { body, .. } | TypeData::Poly { body, .. } => self.check_skolem_escape(*body, max_universe),
            TypeData::AssociatedType { self_ty, .. } => self.check_skolem_escape(*self_ty, max_universe),
            _ => None,
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

            // ── Higher-Ranked Types: `∀X.T <: ∀Y.U` ────────────
            (
                TypeData::Forall {
                    param_index: pi1,
                    param_name: _,
                    body: b1,
                },
                TypeData::Forall {
                    param_index: pi2,
                    param_name: _,
                    body: b2,
                },
            ) => *pi1 == *pi2 && self.subtype(*b1, *b2),
            (TypeData::Forall { body, .. }, _) => self.subtype(*body, sup),
            (_, TypeData::Forall { body, .. }) => self.subtype(sub, *body),

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
                // Aligned with unify_internal_impl's Ref handling:
                // - &mut T <: &T allowed (borrow shortening), invariant inner type
                // - &T <: &mut T NEVER allowed
                // - same mutability → invariant inner type
                if *m1 == *m2 {
                    *t1 == *t2 // same mutability, invariant
                } else if *m1 == true && *m2 == false {
                    *t1 == *t2 // &mut T <: &T, invariant
                } else {
                    false // &T <: &mut T: never allowed
                }
            }
            (TypeData::Pointer { ty: t1 }, TypeData::Pointer { ty: t2 }) => *t1 == *t2, // invariant — exact equality required
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
                p1.iter().zip(p2.iter()).all(|(a, b)| self.subtype(*b, *a))
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
                TypeData::Coproduct { alternatives: a1 },
                TypeData::Coproduct { alternatives: a2 },
            ) => {
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
            (
                TypeData::Rational {
                    int_bits: p1,
                    frac_bits: q1,
                },
                TypeData::Rational {
                    int_bits: p2,
                    frac_bits: q2,
                },
            ) => *p1 == *p2 && *q1 == *q2,
            _ => false,
        }
    }

    pub(crate) fn find_type(&self, data: &TypeData) -> Option<TypeId> {
        self.type_map.get(data).copied()
    }

    pub fn subst(&mut self, ty: TypeId, subst: &Subst) -> TypeId {
        let resolved = self.resolve_binding(ty);
        // Clone the data to avoid borrow conflicts when calling self.subst() recursively.
        let data = self.types[resolved.index()].clone();
        match &*data {
            TypeData::GenericParam { index, .. } => subst.get(*index).copied().unwrap_or(ty),
            TypeData::Int { bits, signed } => self.int(*bits, *signed),
            TypeData::UInt { bits } => self.uint(*bits),
            TypeData::Float { bits } => self.float(*bits),
            TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error => ty,
            TypeData::Adt { kind, def_id, args } => {
                let new_args: Vec<TypeId> = args.iter().map(|&a| self.subst(a, subst)).collect();
                self.alloc(TypeData::Adt {
                    kind: *kind,
                    def_id: *def_id,
                    args: new_args,
                })
            }
            TypeData::Tuple { elems } => {
                let new_elems: Vec<TypeId> = elems.iter().map(|&e| self.subst(e, subst)).collect();
                self.tuple(new_elems)
            }
            TypeData::Array { elem, size } => {
                let new_elem = self.subst(*elem, subst);
                self.array(new_elem, *size)
            }
            TypeData::Slice { elem } => {
                let new_elem = self.subst(*elem, subst);
                self.slice(new_elem)
            }
            TypeData::Ref { ty, mutable } => {
                let new_ty = self.subst(*ty, subst);
                self.reference(new_ty, *mutable)
            }
            TypeData::Pointer { ty } => {
                let new_ty = self.subst(*ty, subst);
                self.pointer(new_ty)
            }
            TypeData::Ptr { size, pointee } => {
                let new_size = self.subst(*size, subst);
                let new_pointee = self.subst(*pointee, subst);
                self.ptr(new_size, new_pointee)
            }
            TypeData::Fn { params, ret } => {
                let new_params: Vec<TypeId> =
                    params.iter().map(|&p| self.subst(p, subst)).collect();
                let new_ret = self.subst(*ret, subst);
                self.function(new_params, new_ret)
            }
            TypeData::Poly { quantifiers, body } => {
                let new_body = self.subst(*body, subst);
                self.poly(quantifiers.clone(), new_body)
            }
            TypeData::DynTrait { .. } => ty,
            TypeData::Forall {
                param_index,
                param_name,
                body,
            } => {
                let new_body = self.subst(*body, subst);
                self.alloc(TypeData::Forall {
                    param_index: *param_index,
                    param_name: param_name.clone(),
                    body: new_body,
                })
            }
            TypeData::Mu {
                param_index,
                param_name,
                body,
            } => {
                let new_body = self.subst(*body, subst);
                self.alloc(TypeData::Mu {
                    param_index: *param_index,
                    param_name: param_name.clone(),
                    body: new_body,
                })
            }
            TypeData::Nu {
                param_index,
                param_name,
                body,
            } => {
                let new_body = self.subst(*body, subst);
                self.alloc(TypeData::Nu {
                    param_index: *param_index,
                    param_name: param_name.clone(),
                    body: new_body,
                })
            }
            TypeData::Exists { name, base } => {
                let new_base = self.subst(*base, subst);
                self.alloc(TypeData::Exists {
                    name: name.clone(),
                    base: new_base,
                })
            }
            TypeData::Coproduct { alternatives } => {
                let new_alts: Vec<TypeId> =
                    alternatives.iter().map(|&a| self.subst(a, subst)).collect();
                self.coproduct(new_alts)
            }
            TypeData::AssociatedType {
                trait_id,
                name,
                self_ty,
            } => {
                let new_self = self.subst(*self_ty, subst);
                self.associated_type(*trait_id, name.clone(), new_self)
            }
            _ => ty,
        }
    }

    fn struct_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Adt { kind: AdtKind::Struct, def_id, args })
    }

    fn enum_ty_no_alloc(&self, def_id: DefId, args: Vec<TypeId>) -> Option<TypeId> {
        self.find_type(&TypeData::Adt { kind: AdtKind::Enum, def_id, args })
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

    fn rational_ty_no_alloc(&self, int_bits: u8, frac_bits: u8) -> Option<TypeId> {
        self.find_type(&TypeData::Rational {
            int_bits,
            frac_bits,
        })
    }

    pub fn is_numeric(&self, ty: TypeId) -> bool {
        match self.get(ty) {
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. } => true,
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
            TypeData::Adt { args, .. } => {
                1 + args
                    .iter()
                    .map(|a| self.type_constructor_depth(*a))
                    .max()
                    .unwrap_or(0)
            }
            TypeData::Tuple { elems }
            | TypeData::Coproduct {
                alternatives: elems,
            } => {
                1 + elems
                    .iter()
                    .map(|e| self.type_constructor_depth(*e))
                    .max()
                    .unwrap_or(0)
            }
            TypeData::Array { elem, .. } => 1 + self.type_constructor_depth(*elem),
            TypeData::Slice { elem } => 1 + self.type_constructor_depth(*elem),
            TypeData::Ref { ty, .. } | TypeData::Pointer { ty } => {
                1 + self.type_constructor_depth(*ty)
            }
            TypeData::Ptr { pointee, .. } => 1 + self.type_constructor_depth(*pointee),
            TypeData::Fn { params, ret } => {
                1 + params
                    .iter()
                    .map(|p| self.type_constructor_depth(*p))
                    .max()
                    .unwrap_or(0)
                    .max(self.type_constructor_depth(*ret))
            }
            TypeData::AssociatedType { self_ty, .. } => 1 + self.type_constructor_depth(*self_ty),
            TypeData::Exists { base, .. } => 1 + self.type_constructor_depth(*base),
            TypeData::Poly {
                quantifiers: _,
                body,
            } => 1 + self.type_constructor_depth(*body),
            TypeData::Forall { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => 1 + self.type_constructor_depth(*body),
            TypeData::DynTrait { .. } => 1,
            TypeData::Int { .. }
            | TypeData::UInt { .. }
            | TypeData::Float { .. }
            | TypeData::Rational { .. }
            | TypeData::Bool
            | TypeData::Char
            | TypeData::Byte
            | TypeData::USize
            | TypeData::Never
            | TypeData::Unit
            | TypeData::Error
            | TypeData::SkolemVar { .. } => 1,
        }
    }

    pub fn is_struct(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Adt { kind: AdtKind::Struct, .. })
    }

    pub fn is_enum(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Adt { kind: AdtKind::Enum, .. })
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

    pub fn is_poly(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Poly { .. })
    }

    pub fn is_rational(&self, ty: TypeId) -> bool {
        matches!(self.get(ty), TypeData::Rational { .. })
    }

    pub fn bits_of_rational_int(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Rational { int_bits, .. } => Some(*int_bits),
            _ => None,
        }
    }

    pub fn bits_of_rational_frac(&self, ty: TypeId) -> Option<u8> {
        match self.get(ty) {
            TypeData::Rational { frac_bits, .. } => Some(*frac_bits),
            _ => None,
        }
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

/// A variance-annotated type graph used for κ(A) computation.
/// Nodes are TypeIds; edges carry a variance sign (+1 covariant, -1 contravariant, 0 invariant).
struct KappaGraph {
    nodes: Vec<TypeId>,
    /// (from_idx, to_idx, sign)
    edges: Vec<(usize, usize, isize)>,
}

impl TypeContext {
    /// Compute the characteristic κ(A) of a type, used for exhaustiveness checking
    /// of match expressions.
    ///
    /// Algorithm (Pistone & Tranchini 2022 §5) — global fixed-point iteration:
    /// 1. Build the type graph G_A with variance edges and axiom links.
    /// 2. Mark all "leaf" types (Bool, Int, …) as determined.
    /// 3. Propagate: when every child of a node has κ fixed, compute the node's κ.
    /// 4. Nodes that remain unmarked form cycles → classify by edge variance:
    ///    all-covariant → InfiniteEnumerable (κ=1), else → Undecidable (κ=∞).
    pub fn characteristic(&self, ty: TypeId) -> Characteristic {
        // Check cache first.
        if let Some(&cached) = self.kappa_cache.borrow().get(&ty) {
            return cached;
        }
        let graph = self.build_kappa_graph(ty);
        let result = self.solve_kappa(&graph);
        // Cache the result.
        self.kappa_cache.borrow_mut().insert(ty, result);
        result
    }

    /// Build the type graph from root, collecting all reachable nodes,
    /// variance edges, and axiom links for bound GenericParam occurrences.
    fn build_kappa_graph(&self, root: TypeId) -> KappaGraph {
        use std::collections::HashSet as Set;

        let mut nodes: Vec<TypeId> = Vec::new();
        let mut edges: Vec<(usize, usize, isize)> = Vec::new();
        let mut node_map: HashMap<TypeId, usize> = HashMap::default();
        let mut visited: Set<TypeId> = Set::default();
        // Stack of active binder scopes: (param_index, binder_node_idx)
        let mut binder_stack: Vec<(usize, usize)> = Vec::new();
        // GenericParam occurrences grouped by (param_index, binder_node_idx).
        // Each entry collects all occurrences of a specific variable bound by a specific binder.
        let mut param_occurrences: HashMap<(usize, usize), Vec<usize>> = HashMap::default();

        // Recursive traversal.
        fn traverse(
            ty: TypeId,
            ctx: &TypeContext,
            nodes: &mut Vec<TypeId>,
            edges: &mut Vec<(usize, usize, isize)>,
            node_map: &mut HashMap<TypeId, usize>,
            visited: &mut Set<TypeId>,
            binder_stack: &mut Vec<(usize, usize)>,
            param_occurrences: &mut HashMap<(usize, usize), Vec<usize>>,
        ) -> usize {
            if let Some(&idx) = node_map.get(&ty) {
                return idx;
            }
            let idx = nodes.len();
            nodes.push(ty);
            node_map.insert(ty, idx);
            visited.insert(ty);

            let data = ctx.get(ty);
            match data {
                TypeData::GenericParam { index, .. } => {
                    // Check if this GPIO is bound by an active binder.
                    if let Some(&(pi, binder_idx)) =
                        binder_stack.iter().rev().find(|(p, _)| *p == *index)
                    {
                        param_occurrences
                            .entry((pi, binder_idx))
                            .or_default()
                            .push(idx);
                        // Add a self-loop to mark this GPIO as bound.
                        // This prevents leaf_kappa from resolving it immediately
                        // and ensures the binder's fixed-point cycle is detected.
                        edges.push((idx, idx, 1));
                    }
                }
                TypeData::Forall {
                    param_index, body, ..
                }
                | TypeData::Mu {
                    param_index, body, ..
                }
                | TypeData::Nu {
                    param_index, body, ..
                } => {
                    // Push binder FIRST, then traverse body so GenericParam
                    // occurrences register with the correct binder scope.
                    binder_stack.push((*param_index, idx));
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    // Binder → body (covariant)
                    edges.push((idx, body_idx, 1));
                    binder_stack.pop();
                }
                TypeData::Poly { quantifiers, body } => {
                    // Push all quantifier indices as binders for the body.
                    for &(pi, _) in quantifiers {
                        binder_stack.push((pi, idx));
                    }
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    for _ in quantifiers {
                        binder_stack.pop();
                    }
                    // Poly → body (covariant)
                    edges.push((idx, body_idx, 1));
                }
                TypeData::Exists { base: body, .. } => {
                    // Not introducing a binder for GenericParam — treat body as covariant child.
                    let body_idx = traverse(
                        *body,
                        ctx,
                        nodes,
                        edges,
                        node_map,
                        visited,
                        binder_stack,
                        param_occurrences,
                    );
                    edges.push((idx, body_idx, 1));
                }
                _ => {
                    // Generic case: emit variance edges for all children.
                    let variance_edges = ctx.compute_variance_edges(ty);
                    for ve in &variance_edges {
                        let child_idx = traverse(
                            ve.target,
                            ctx,
                            nodes,
                            edges,
                            node_map,
                            visited,
                            binder_stack,
                            param_occurrences,
                        );
                        edges.push((idx, child_idx, ve.sign));
                    }
                }
            }
            idx
        }

        traverse(
            root,
            self,
            &mut nodes,
            &mut edges,
            &mut node_map,
            &mut visited,
            &mut binder_stack,
            &mut param_occurrences,
        );

        // Build axiom links: for each (variable, binder), connect all GPIO occurrences
        // pairwise as bidirectional covariant edges so they participate in the fixed-point solver.
        for (_key, occurrences) in &param_occurrences {
            for i in 0..occurrences.len() {
                for j in (i + 1)..occurrences.len() {
                    let a = occurrences[i];
                    let b = occurrences[j];
                    edges.push((a, b, 1)); // a → b (covariant)
                    edges.push((b, a, 1)); // b → a (covariant)
                }
            }
        }

        KappaGraph { nodes, edges }
    }

    /// Solve κ for a graph using fixed-point iteration.
    /// Returns the κ of the root node (graph.nodes[0]).
    fn solve_kappa(&self, graph: &KappaGraph) -> Characteristic {
        let n = graph.nodes.len();
        // result[i] = None (unknown) or Some(κ)
        let mut result: Vec<Option<Characteristic>> = vec![None; n];
        // Maps TypeId → Characteristic for quick child lookup during combine.
        let mut type_kappa: HashMap<TypeId, Characteristic> = HashMap::default();

        let mut out_degree: Vec<usize> = vec![0; n];
        for &(from, _to, _sign) in &graph.edges {
            out_degree[from] += 1;
        }

        // Determine initial κ for base-type leaf nodes (out_degree == 0).
        let mut queue: Vec<usize> = Vec::new();
        for i in 0..n {
            if out_degree[i] == 0 {
                let k = self.leaf_kappa(graph.nodes[i]);
                result[i] = Some(k);
                type_kappa.insert(graph.nodes[i], k);
                queue.push(i);
            }
        }

        // Build reverse adjacency: for each node, which nodes have an edge TO it?
        let mut reverse_edges: Vec<Vec<(usize, isize)>> = vec![Vec::new(); n];
        for &(from, to, sign) in &graph.edges {
            reverse_edges[to].push((from, sign));
        }

        // Track how many outgoing edges are still unresolved for each node.
        let mut unresolved_count: Vec<usize> = out_degree.clone();

        // BFS-based propagation: pop determined nodes and check their predecessors.
        while let Some(determined) = queue.pop() {
            let det_kappa = result[determined].unwrap();

            // Check all predecessors (nodes that depend on `determined`).
            for &(pred, _sign) in &reverse_edges[determined] {
                if result[pred].is_some() {
                    continue;
                }
                unresolved_count[pred] = unresolved_count[pred].saturating_sub(1);
                if unresolved_count[pred] == 0 {
                    let k = self.combine_kappa(graph.nodes[pred], &type_kappa);
                    result[pred] = Some(k);
                    type_kappa.insert(graph.nodes[pred], k);
                    queue.push(pred);
                }
            }
        }

        // After propagation, check for remaining undetermined nodes (cycles).
        let undetermined: Vec<usize> = (0..n).filter(|i| result[*i].is_none()).collect();

        if undetermined.is_empty() {
            return result[0].unwrap();
        }

        // Phase 2: classify remaining cycle(s).  Check edge variance.
        use std::collections::HashSet as Set;
        let undetermined_set: Set<usize> = undetermined.iter().copied().collect();
        let mut has_non_covariant = false;

        for &(from, to, sign) in &graph.edges {
            // Only consider edges where BOTH ends are in the remaining subgraph.
            if undetermined_set.contains(&from) && undetermined_set.contains(&to) {
                if sign != 1 {
                    has_non_covariant = true;
                    break;
                }
            }
        }

        if has_non_covariant {
            for &i in &undetermined {
                result[i] = Some(Characteristic::Undecidable);
            }
        } else {
            for &i in &undetermined {
                result[i] = Some(Characteristic::InfiniteEnumerable);
            }
        }

        result[0].unwrap()
    }

    /// Return the κ of a leaf type (no outgoing edges).
    fn leaf_kappa(&self, ty: TypeId) -> Characteristic {
        // Does NOT go through `characteristic_body` — this is the base case.
        let data = self.get(ty);
        match data {
            TypeData::Int { bits, .. } => Characteristic::FiniteExhaustible(1usize << bits),
            TypeData::UInt { bits } => Characteristic::FiniteExhaustible(1usize << bits),
            TypeData::Float { .. } | TypeData::USize => {
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Rational {
                int_bits,
                frac_bits,
                ..
            } => {
                let total_bits = *int_bits as u32 + *frac_bits as u32;
                // Use (usize::BITS - 1) so we can safely represent 1 << total_bits.
                // The previous hard-coded threshold of 16 would misclassify even
                // modest fixed-point types like Rational<8,8> as `usize::MAX`,
                // degrading pattern-match exhaustiveness precision.
                if total_bits >= (usize::BITS - 1) {
                    Characteristic::FiniteExhaustible(usize::MAX)
                } else {
                    Characteristic::FiniteExhaustible(1usize << total_bits)
                }
            }
            TypeData::Bool => Characteristic::FiniteExhaustible(2),
            TypeData::Char => Characteristic::FiniteExhaustible(256),
            TypeData::Byte => Characteristic::FiniteExhaustible(256),
            TypeData::Unit => Characteristic::FiniteExhaustible(1),
            TypeData::Never => Characteristic::FiniteExhaustible(0),
            TypeData::Error => Characteristic::FiniteExhaustible(0),
            TypeData::GenericParam { .. } => {
                // GenericParam with no axiom links → unknown but finite.
                // (If it HAS axiom links, it'll be part of a cycle and get
                //  classified during Phase 2.)
                Characteristic::FiniteExhaustible(usize::MAX)
            }
            TypeData::Adt { .. } => Characteristic::FiniteExhaustible(usize::MAX),
            TypeData::InferVar { .. } => Characteristic::FiniteExhaustible(usize::MAX),
            TypeData::DynTrait { .. } => Characteristic::InfiniteEnumerable,

            // The following types are NOT leaf types in practice because they
            // have outgoing edges.  This arm is a fallback.
            _ => Characteristic::FiniteExhaustible(usize::MAX),
        }
    }

    /// Combine children κ values into a node's κ, given the type constructor.
    /// Called when all of a node's outgoing edges point to determined nodes.
    /// Combine children κ values into a node's κ, given the type constructor.
    /// Called when all of a node's outgoing edges point to determined nodes.
    /// `kappa_map` maps child TypeId → determined Characteristic.
    fn combine_kappa(
        &self,
        ty: TypeId,
        kappa_map: &HashMap<TypeId, Characteristic>,
    ) -> Characteristic {
        /// Helper: look up a child's κ — must be resolved at this point.
        fn ck(
            ctx: &TypeContext,
            child: TypeId,
            map: &HashMap<TypeId, Characteristic>,
        ) -> Characteristic {
            *map.get(&child)
                .expect("child kappa not resolved: graph construction missed a dependency edge")
        }

        let data = self.get(ty);
        match data {
            TypeData::Tuple { elems } => {
                let mut total = 1usize;
                let mut has_infinite = false;
                for &e in elems {
                    match ck(self, e, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => total = total.saturating_mul(n),
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(total)
                }
            }
            TypeData::Adt { args, .. } => {
                let mut has_infinite = false;
                for &a in args {
                    match ck(self, a, kappa_map) {
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
            TypeData::Array { elem, size } => match ck(self, *elem, kappa_map) {
                Characteristic::FiniteExhaustible(n) => {
                    Characteristic::FiniteExhaustible(n.saturating_pow(*size as u32))
                }
                Characteristic::InfiniteEnumerable => Characteristic::InfiniteEnumerable,
                Characteristic::Undecidable => Characteristic::Undecidable,
            },
            TypeData::Slice { .. }
            | TypeData::Ref { .. }
            | TypeData::Pointer { .. }
            | TypeData::Ptr { .. } => Characteristic::InfiniteEnumerable,
            TypeData::Fn { params, ret } => {
                let mut domain_product = 1usize;
                let mut domain_infinite = false;
                for &p in params {
                    match ck(self, p, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => {
                            domain_product = domain_product.saturating_mul(n)
                        }
                        Characteristic::InfiniteEnumerable => domain_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                match ck(self, *ret, kappa_map) {
                    Characteristic::Undecidable => Characteristic::Undecidable,
                    Characteristic::FiniteExhaustible(c) => {
                        if domain_product == 0 {
                            Characteristic::FiniteExhaustible(1)
                        } else if domain_infinite {
                            if c == 0 {
                                Characteristic::FiniteExhaustible(0)
                            } else if c == 1 {
                                Characteristic::FiniteExhaustible(1)
                            } else {
                                Characteristic::InfiniteEnumerable
                            }
                        } else {
                            Characteristic::FiniteExhaustible(
                                c.saturating_pow(domain_product as u32),
                            )
                        }
                    }
                    Characteristic::InfiniteEnumerable => {
                        if domain_product == 0 {
                            Characteristic::FiniteExhaustible(1)
                        } else {
                            Characteristic::InfiniteEnumerable
                        }
                    }
                }
            }
            TypeData::Coproduct { alternatives } => {
                let mut total = 0usize;
                let mut has_infinite = false;
                for &a in alternatives {
                    match ck(self, a, kappa_map) {
                        Characteristic::FiniteExhaustible(n) => total = total.saturating_add(n),
                        Characteristic::InfiniteEnumerable => has_infinite = true,
                        Characteristic::Undecidable => return Characteristic::Undecidable,
                    }
                }
                if has_infinite {
                    Characteristic::InfiniteEnumerable
                } else {
                    Characteristic::FiniteExhaustible(total)
                }
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. } => ck(self, *body, kappa_map),
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => ck(self, *body, kappa_map),
            TypeData::AssociatedType { self_ty, .. } => ck(self, *self_ty, kappa_map),
            _ => Characteristic::FiniteExhaustible(usize::MAX),
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

    fn new_ctx() -> TypeContext {
        TypeContext::new()
    }

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
        assert_eq!(
            ctx.characteristic(ctx.bool()),
            Characteristic::FiniteExhaustible(2)
        );
    }

    #[test]
    fn test_characteristic_int32() {
        let mut ctx = TypeContext::new();
        let i32 = ctx.int(32, true);
        assert_eq!(
            ctx.characteristic(i32),
            Characteristic::FiniteExhaustible(2u64.pow(32) as usize)
        );
    }

    #[test]
    fn test_characteristic_unit() {
        let ctx = TypeContext::new();
        assert_eq!(
            ctx.characteristic(ctx.unit()),
            Characteristic::FiniteExhaustible(1)
        );
    }

    #[test]
    fn test_characteristic_fn() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let fn_ty = ctx.function(vec![bool_ty], bool_ty);
        // Bool → Bool has 2^2 = 4 inhabitants.
        assert_eq!(
            ctx.characteristic(fn_ty),
            Characteristic::FiniteExhaustible(4)
        );
    }

    #[test]
    fn test_characteristic_slice() {
        let mut ctx = TypeContext::new();
        let bool_ty = ctx.bool();
        let slice_ty = ctx.slice(bool_ty);
        assert_eq!(
            ctx.characteristic(slice_ty),
            Characteristic::InfiniteEnumerable
        );
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
        // With axiom links connecting GPIO occurrences as bidirectional edges,
        // the remaining cycle includes a contravariant edge (Fn param → Tuple),
        // so the result is Undecidable.
        let kappa = ctx.characteristic(ty);
        assert_eq!(
            kappa,
            Characteristic::Undecidable,
            "recursive stream type with contravariant path should be Undecidable"
        );
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
        ctx.set_binding(a, bool_ty);
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
        ctx.set_binding(a, bool_ty);
        ctx.begin_transaction();
        ctx.set_binding(a, unit_ty);
        ctx.rollback_transaction();
        assert_eq!(ctx.resolve_binding(a), bool_ty);
        ctx.commit_transaction();
    }

    #[test]
    /// Inner commit + outer rollback: verify that the outer transaction's undo log
    /// correctly absorbs the inner transaction's changes on commit.  Without the
    /// log-merge in `commit_transaction`, the outer rollback would leave the inner
    /// transaction's modifications in place, breaking the atomicity semantics.
    fn test_transaction_nested_commit_outer_rollback() {
        let mut ctx = TypeContext::new();
        let a = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let unit_ty = ctx.unit();

        // Outer: set a → Bool
        ctx.begin_transaction();
        ctx.set_binding(a, bool_ty);
        assert_eq!(ctx.resolve_binding(a), bool_ty);

        // Inner: set a → Unit, then commit
        ctx.begin_transaction();
        ctx.set_binding(a, unit_ty);
        assert_eq!(ctx.resolve_binding(a), unit_ty);
        ctx.commit_transaction();

        // After inner commit, a is still Unit
        assert_eq!(ctx.resolve_binding(a), unit_ty);

        // Outer rollback: should restore a to its state BEFORE the outer began
        ctx.rollback_transaction();
        assert_eq!(ctx.resolve_binding(a), a);
    }

    #[test]
    /// Three-level nested transaction with commit/rollback at each layer.
    /// Verifies that the undo-log merge across three levels correctly restores
    /// the outermost state when the outermost rolls back.
    ///
    /// Layer-3 commit → merge into Layer-2 log.
    /// Layer-2 commit → merge (Layer-3 ∪ Layer-2) into Layer-1 log.
    /// Layer-1 rollback → reverse-apply the combined log → initial state.
    fn test_transaction_nested_three_level() {
        let mut ctx = TypeContext::new();
        let a = ctx.int(32, true);
        let bool_ty = ctx.bool();
        let unit_ty = ctx.unit();
        let int64 = ctx.int(64, true);

        // L1: a → Bool
        ctx.begin_transaction();
        ctx.set_binding(a, bool_ty);

        // L2: a → Unit
        ctx.begin_transaction();
        ctx.set_binding(a, unit_ty);

        // L3: a → Int64, then commit L3
        ctx.begin_transaction();
        ctx.set_binding(a, int64);
        assert_eq!(ctx.resolve_binding(a), int64);
        ctx.commit_transaction(); // L3 log merged into L2

        // L2 commit → merged log (L3+L2) merged into L1
        assert_eq!(ctx.resolve_binding(a), int64);
        ctx.commit_transaction();

        // After L2 commit, a is still Int64
        assert_eq!(ctx.resolve_binding(a), int64);

        // L1 rollback → should undo everything
        ctx.rollback_transaction();
        assert_eq!(ctx.resolve_binding(a), a);
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
        assert_eq!(
            forall,
            ctx.function(vec![int_ty], bool_ty),
            "∀X.(X⇒Int)⇒(X⇒Bool) should reduce to Int⇒Bool"
        );
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
        assert_eq!(
            forall, expected,
            "∀X.(Int⇒Bool⇒X)⇒X should reduce to (Int,Bool)"
        );
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
        assert_eq!(
            reduced, expected,
            "(Int⇒Bool⇒X)⇒X should reduce to (Int,Bool)"
        );
    }

    #[test]
    fn test_coyoneda_multi_param_preserves_return() {
        // co-Yoneda: ∀X.(X ⇒ Int ⇒ Float) ⇒ X
        //   ips = [X(pi), Int], ir = Float
        //   Correct: replacement = Int → Float = Fn(Int, Float)
        //   Bug:     replacement = Int (drops Float!)
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let int_ty = ctx.int(32, true);
        let float_ty = ctx.float(64);
        // branch: X ⇒ Int → Float
        let branch = ctx.function(vec![int_ty], float_ty);
        // outer: X ⇒ branch  →  but co-Yoneda needs X as FIRST param
        let outer_fn = ctx.function(vec![p0, branch], p0); // p0 is ret, but it doesn't matter
        // Actually build the right shape: ∀X.(X ⇒ Int → Float) ⇒ X
        // The branch param list is [X, Int] with ret=Float
        let inner_fn = ctx.function(vec![p0, int_ty], float_ty);
        let outer = ctx.function(vec![inner_fn], p0);
        let forall = ctx.forall(0, "X".into(), outer);
        let reduced = ctx.try_yoneda_reduce(forall);
        // Expected: Int → Float
        let expected = ctx.function(vec![int_ty], float_ty);
        assert_eq!(
            reduced, expected,
            "∀X.(X⇒Int⇒Float)⇒X should reduce to Int→Float, not lose Float"
        );
    }

    // ── HRTB / Forall subtype tests ──────────────────────────────

    #[test]
    fn test_subtype_forall_identical() {
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let fn_ty = ctx.function(vec![p0], p0);
        let forall = ctx.forall(0, "X".into(), fn_ty);
        assert!(ctx.subtype(forall, forall));
    }

    #[test]
    fn test_subtype_forall_body_subtype() {
        let mut ctx = TypeContext::new();
        let p0 = ctx.generic_param(0, "X".into());
        let never = ctx.never();
        let int32 = ctx.int(32, true);
        let sub_fn = ctx.function(vec![p0], never);
        let sup_fn = ctx.function(vec![p0], int32);
        let sub_forall = ctx.forall(0, "X".into(), sub_fn);
        let sup_forall = ctx.forall(0, "X".into(), sup_fn);
        assert!(ctx.subtype(sub_forall, sup_forall));
    }

    #[test]
    fn test_subtype_forall_peel_sup() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int(32, true);
        let forall_ty = ctx.forall(0, "X".into(), int_ty);
        assert!(ctx.subtype(int_ty, forall_ty));
    }

    #[test]
    fn test_normalize_associated_type_concrete_self() {
        let mut ctx = TypeContext::new();
        let def_id = DefId(42);
        let int_ty = ctx.int(32, true);
        let adt_ty = ctx.alloc(TypeData::Adt { kind: AdtKind::Struct, def_id, args: vec![int_ty] });
        assert_eq!(ctx.try_normalize_associated_type_def_id(adt_ty), Some(def_id));
    }

    #[test]
    fn test_normalize_associated_type_abstract_self() {
        let mut ctx = TypeContext::new();
        let var_id = ctx.alloc(TypeData::InferVar { id: 0 });
        assert_eq!(ctx.try_normalize_associated_type_def_id(var_id), None);
    }
}
