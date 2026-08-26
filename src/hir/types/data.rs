use rustc_hash::FxHashMap as HashMap;
use std::cell::Cell;
use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use super::*;
use crate::ast::OverflowPolicy;
use crate::symbol::Symbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(NonZeroUsize);

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
    /// A compile-time validated regular expression: `Regex<"pattern">`.
    /// Replaces the old `Reserved30` slot — the last valid discriminant is
    /// now `Regex = 30`; only `Reserved31 = 31` remains as a padding variant.
    Regex = 30,
    /// `Type` — the type of types, used as a first-class value in comptime.
    TypeKind = 31,
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
            TypeData::Regex { .. } => TypeTag::Regex,
            TypeData::Type => TypeTag::TypeKind,
            TypeData::Opaque { .. } => TypeTag::TypeKind,
        }
    }
}

impl TypeId {
    /// Number of bits reserved for the type tag in the lower bits of TypeId.
    /// Must be large enough to hold all TypeTag variants (currently 32, max 31
    /// with 5 bits).  Bump to 6 if more than 31 variants are needed.
    pub const TAG_BITS: usize = 5;
    const TAG_MASK: usize = (1 << Self::TAG_BITS) - 1;
    /// The last valid discriminant value (not the mask — those differ when
    /// reserved variants fill the gap to the bit boundary).  Set to the
    /// highest *real* variant; `Reserved31` exists only to make every 5-bit
    /// pattern a valid `TypeTag` for transmute safety.
    const TAG_LAST_VARIANT: usize = TypeTag::Regex as usize;
    /// Bounds check: every valid tag must be ≤ `TAG_LAST_VARIANT`.
    const TAG_MAX: usize = Self::TAG_LAST_VARIANT;

    /// Sentinel representing "no type" (error nodes, untyped expressions).
    /// Occupies the `NonZeroUsize` niche so that `Option<TypeId>` is 8 bytes
    /// instead of 16 — every type allocated through `TypeContext::alloc` has
    /// `raw >= 32` (since `index` is offset by 1).
    ///
    /// Chosen as `usize::MAX` so that a stray call to `tag()` or `index()`
    /// on this sentinel produces an obviously-wrong value: `tag()` returns
    /// discriminant 31 (past the last real variant `SkolemVar = 29`, but
    /// within the reserved range `Reserved31 = 31`), and `index()` returns
    /// `usize::MAX >> 5 - 1`, an unreachable index.
    /// Both paths are guarded by `debug_assert!` checks that fire before
    /// any unsound action.
    pub const NONE: TypeId = TypeId(unsafe {
        // SAFETY: `usize::MAX` is non-zero, so `NonZeroUsize::new_unchecked`
        // never creates an invalid value; the encoding is a sentinel whose
        // `index()` lands on the reserved `Reserved31` variant (guarded by
        // `debug_assert!` in `index()`).
        NonZeroUsize::new_unchecked(usize::MAX)
    });

    /// Create a `TypeId` from a raw encoded value.
    /// Panics if `raw` is zero (which can never be a valid `NonZeroUsize`).
    #[inline]
    pub fn from_raw(raw: usize) -> Self {
        TypeId(NonZeroUsize::new(raw).expect("TypeId raw value must be non-zero"))
    }

    /// Return the raw underlying encoded value.
    #[inline]
    pub fn raw(self) -> usize {
        self.0.get()
    }

    /// Decode the arena index.
    ///
    /// The index stored in the raw encoding is biased by +1 so that the
    /// base value is never zero (guaranteeing `NonZeroUsize` validity).
    /// This method subtracts the bias to recover the true 0-based index.
    pub fn index(self) -> usize {
        assert!(
            self.0.get() != usize::MAX,
            "TypeId::index() called on sentinel NONE"
        );
        (self.0.get() >> Self::TAG_BITS) - 1
    }

    pub fn tag(self) -> TypeTag {
        let raw = self.0.get();
        debug_assert!(raw != usize::MAX, "TypeId::tag() called on sentinel NONE");
        let tag_val = raw & Self::TAG_MASK;
        // Catch tag overflow in debug builds: if a new TypeTag variant pushes
        // the discriminant past TAG_LAST_VARIANT, this assert fires immediately
        // rather than silently producing a value outside the intended range.
        debug_assert!(
            tag_val <= Self::TAG_MAX,
            "TypeTag discriminant {} exceeds TAG_LAST_VARIANT={}; \
             increase TAG_BITS to accommodate more variants",
            tag_val,
            Self::TAG_MAX,
        );
        // SAFETY: every TypeId created through TypeContext::alloc has a valid
        // tag in 0..TAG_LAST_VARIANT, enforced by TypeTag's explicit discriminants
        // and debug_assert above.  TAG_MASK covers all 5-bit patterns, and
        // Reserved30/Reserved31 ensure that even out-of-range values (30, 31)
        // produce a valid, albeit unreachable, variant — never UB.
        unsafe { std::mem::transmute::<usize, TypeTag>(tag_val) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrateId(pub DefId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeData {
    Int {
        bits: u32,
        signed: bool,
        overflow_policy: OverflowPolicy,
    },
    UInt {
        bits: u32,
        overflow_policy: OverflowPolicy,
    },
    Float {
        bits: u32,
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
    /// (Mac Lane §III.4): the product a × b is the universal
    /// object with projections p: a×b → a, q: a×b → b, satisfying
    /// C(c, a×b) ≅ C(c, a) × C(c, b).  Dual to Coproduct (§III.3).    
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
        /// The explicit lifetime annotation (`&'a mut T` — `Some('a)`),
        /// or `None` for an elided/inferred lifetime.  Previously
        /// dropped in the AST→HIR lowering (`Type::Reference { .. }`
        /// swallowed it); kept now so the borrow checker can map the
        /// annotation to the UniversalRegions early-bound region.
        lifetime: Option<Symbol>,
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
        param_index: usize,
        name: Symbol,
        base: TypeId,
    },
    /// An explicit universal quantifier: ∀X. Body
    /// `param_index` and `param_name` identify the bound variable X.
    /// X appears in `body` as `GenericParam { index: param_index }`.
    /// This is a compiler-internal node — there is no user-facing ∀ syntax.
    Forall {
        param_index: usize,
        param_name: Symbol,
        body: TypeId,
    },
    GenericParam {
        index: usize,
        name: Symbol,
    },
    AssociatedType {
        trait_id: DefId,
        name: Symbol,
        self_ty: TypeId,
    },
    InferVar {
        id: usize,
        /// The per-variable universe (rustc's `CanonicalVarKind::Ty { ui }`):
        /// 0 for ordinary variables; higher for HRTB/forall-introduced
        /// variables — the unify check forbids binding a LOWER-universe
        /// variable to a higher-universe type (escape).
        universe: usize,
    },
    /// A named coproduct (sum type), Σᵢ Aᵢ.
    /// Introduced by Yoneda reduction of ∀X.(A₁⇒X)⇒...⇒(Aₙ⇒X)⇒X → Σᵢ Aᵢ.
    /// Unlike Tuple (product), Coproduct represents "one of the alternatives."
    ///
    /// (Mac Lane §III.3): the coproduct a ⊔ b is the universal
    /// object receiving injections i: a → a⊔b, j: b → a⊔b, satisfying
    /// C(a⊔b, c) ≅ C(a, c) × C(b, c).  The Yoneda-reduced form
    /// ∀X.(A₁⇒X)⇒⋯⇒(Aₙ⇒X)⇒X is exactly this universal property internalized
    /// as a polymorphic type.
    Coproduct {
        alternatives: Vec<TypeId>,
    },
    /// Least fixed-point type: μX.A⟨X⟩.
    /// X is the recursive type variable, identified by param_index in body.
    ///
    /// (Mac Lane §VI.2): μX.F(X) is the initial algebra of
    /// the endofunctor F — the least fixed point in the category of F-algebras.
    /// Produced by Yoneda reduction (≡_X) when the branch product depends on X.
    Mu {
        param_index: usize,
        param_name: Symbol,
        body: TypeId,
    },
    /// Greatest fixed-point type: νX.A⟨X⟩.
    ///
    /// (Mac Lane §VI.2, dual): νX.F(X) is the final coalgebra
    /// of F — the greatest fixed point.  Produced by co-Yoneda reduction (≡^X).
    Nu {
        param_index: usize,
        param_name: Symbol,
        body: TypeId,
    },
    /// A polytype: `[∀ᾱ. τ]` — a boxed first-class polymorphic type.
    /// `quantifiers` lists the universally quantified variables as (index, name) pairs.
    /// `body` is the inner type, referencing quantifiers via `GenericParam`.
    /// See OmniML §3.1 (O'Brien, Rémy & Scherer).
    ///
    /// # Invariant (closedness)
    /// The body must be a *closed* type: all free variables are bound by
    /// `quantifiers`.  Inference variables (`InferVar`) must NOT appear in
    /// the body — only `GenericParam` references bound by the quantifiers.
    /// This matches the OmniML reference implementation's invariant
    /// (omniml/lib/constraint_solver/principal_shape.ml):
    /// ```ocaml
    /// | Poly _ ->
    ///   (* Invariant: no occurrences of [Poly]. *)
    ///   assert false
    /// ```
    /// Functions that substitute into types treat `Poly` differently
    /// depending on what they replace:
    /// - `subst()` replaces `GenericParam` indices and **does** recurse
    ///   into the body (with binder shadowing), because `GenericParam`
    ///   CAN appear in the body as bound variables.
    /// - `replace_infer()` replaces `InferVar` IDs and does **not** recurse,
    ///   because `InferVar` must never appear in a closed polytope body.
    ///
    /// # Defense-in-depth
    /// The reference implementation enforces the closedness invariant
    /// explicitly via `Poly.invariant` (principal_shape.ml).  Ponent does
    /// NOT have a runtime check for this invariant.  If a bug elsewhere in
    /// the compiler creates a `Poly` whose body contains an `InferVar`,
    /// `replace_infer()` will silently fail to replace it, leaving a stale
    /// inference variable in the type.  Adding an invariant assertion
    /// (e.g. in `TypeContext::poly()`) would catch such violations at the
    /// construction site rather than silently propagating them.
    Poly {
        quantifiers: Vec<(usize, Symbol)>,
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
    /// A compile-time validated regular expression pattern: `Regex<"pattern">`.
    Regex {
        pattern: String,
    },
    /// `type` — the type of types, used as a first-class value in comptime
    /// contexts.  When a comptime function returns `type`, it returns a
    /// `TypeId` value that can be used in type declarations.
    /// Corresponds to `Token::Type` in the parser and `ComptimeValue::Type`.
    Type,
    /// An opaque type defined via `type T = impl Trait` (TAIT).
    /// `def_id` identifies the opaque type.  `hidden` is:
    /// - `None` inside the defining scope (type is opaque)
    /// - `Some(hidden_ty)` outside the defining scope (revealed)
    Opaque {
        def_id: DefId,
        hidden: Option<TypeId>,
    },
}

impl fmt::Display for TypeTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeTag::Int => write!(f, "Int"),
            TypeTag::UInt => write!(f, "UInt"),
            TypeTag::Float => write!(f, "Float"),
            TypeTag::Bool => write!(f, "Bool"),
            TypeTag::Char => write!(f, "Char"),
            TypeTag::Byte => write!(f, "Byte"),
            TypeTag::USize => write!(f, "USize"),
            TypeTag::Adt => write!(f, "struct/enum"),
            TypeTag::Tuple => write!(f, "tuple"),
            TypeTag::Array => write!(f, "array"),
            TypeTag::Slice => write!(f, "slice"),
            TypeTag::Ref => write!(f, "ref"),
            TypeTag::Pointer => write!(f, "ptr"),
            TypeTag::Fn => write!(f, "fn"),
            TypeTag::InferVar => write!(f, "?"),
            TypeTag::GenericParam => write!(f, "generic"),
            TypeTag::SkolemVar => write!(f, "skolem"),
            TypeTag::Never => write!(f, "!"),
            TypeTag::Error => write!(f, "!!"),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl fmt::Display for TypeData {
    /// Human-readable type name.  For types that contain nested `TypeId`
    /// values (e.g. `Ref`, `Slice`, `Adt`), use `display_with(ctx)` instead
    /// — this fallback shows the `TypeTag` of the element type.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeData::Regex { pattern } => write!(f, "Regex<\"{}\">", pattern),
            TypeData::Int { bits, signed, .. } => {
                if *signed {
                    write!(f, "Int<{}>", bits)
                } else {
                    write!(f, "UInt<{}>", bits)
                }
            }
            TypeData::UInt { bits, .. } => write!(f, "UInt<{}>", bits),
            TypeData::Float { bits } => write!(f, "Float<{}>", bits),
            TypeData::Bool => write!(f, "Bool"),
            TypeData::Char => write!(f, "Char"),
            TypeData::Byte => write!(f, "Byte"),
            TypeData::USize => write!(f, "USize"),
            TypeData::Adt { kind, def_id, .. } => {
                if def_id == &DefId::SENTINEL_STR {
                    write!(f, "Str")
                } else {
                    match kind {
                        AdtKind::Struct => write!(f, "struct"),
                        AdtKind::Enum => write!(f, "enum"),
                    }
                }
            }
            TypeData::Tuple { elems } => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.tag())?;
                }
                if elems.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            TypeData::Array { elem, size } => write!(f, "[{}; {}]", elem.tag(), size),
            TypeData::Slice { elem } => write!(f, "&[{}]", elem.tag()),
            TypeData::Ref { ty, mutable, .. } => {
                if *mutable {
                    write!(f, "&mut {}", ty.tag())
                } else {
                    write!(f, "&{}", ty.tag())
                }
            }
            TypeData::Pointer { ty } => write!(f, "*{}", ty.tag()),
            TypeData::Ptr { size, pointee } => {
                write!(f, "Ptr<size={}, pointee={}>", size.tag(), pointee.tag())
            }
            TypeData::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.tag())?;
                }
                write!(f, ") -> {}", ret.tag())
            }
            TypeData::DynTrait { .. } => write!(f, "dyn Trait"),
            TypeData::Exists { name, .. } => write!(f, "exists {}", name),
            TypeData::Forall { .. } => write!(f, "forall ..."),
            TypeData::Poly { .. } => write!(f, "poly"),
            TypeData::InferVar { id, .. } => write!(f, "infer#{}", id),
            TypeData::GenericParam { index, .. } => write!(f, "T{}", index),
            TypeData::SkolemVar { .. } => write!(f, "skolem"),
            TypeData::Never => write!(f, "!"),
            TypeData::Error => write!(f, "!!"),
            TypeData::Rational {
                int_bits,
                frac_bits,
                ..
            } => write!(f, "Rational<{}, {}>", int_bits, frac_bits),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl TypeData {
    /// Render the type to a human-readable string, resolving nested `TypeId`
    /// values through `ctx` so that e.g. `&Str` is printed as `&Str` rather
    /// than `&struct/enum`.  This is the preferred formatting method when a
    /// `TypeContext<'input>` is available.
    ///
    /// When `symbols` is `Some`, ADT types (structs, enums) are rendered with
    /// their actual name (e.g. `MyStruct`) instead of a generic `struct`/`enum`.
    pub fn display_with<'input>(
        &self,
        ctx: &TypeContext<'input>,
        symbols: Option<&crate::hir::symbol::SymbolTable>,
    ) -> String {
        match self {
            TypeData::Ref { ty, mutable, .. } => {
                let inner = ctx.get(*ty).display_with(ctx, symbols);
                if *mutable {
                    format!("&mut {}", inner)
                } else {
                    format!("&{}", inner)
                }
            }
            TypeData::Slice { elem } => {
                let inner = ctx.get(*elem).display_with(ctx, symbols);
                format!("&[{}]", inner)
            }
            TypeData::Array { elem, size } => {
                let inner = ctx.get(*elem).display_with(ctx, symbols);
                format!("[{}; {}]", inner, size)
            }
            TypeData::Tuple { elems } => {
                let parts: Vec<String> = elems
                    .iter()
                    .map(|e| ctx.get(*e).display_with(ctx, symbols))
                    .collect();
                format!("({})", parts.join(", "))
            }
            TypeData::Fn { params, ret } => {
                let params: Vec<String> = params
                    .iter()
                    .map(|p| ctx.get(*p).display_with(ctx, symbols))
                    .collect();
                let ret = ctx.get(*ret).display_with(ctx, symbols);
                format!("fn({}) -> {}", params.join(", "), ret)
            }
            TypeData::Ptr { size, pointee } => {
                format!(
                    "Ptr<size={}, pointee={}>",
                    ctx.get(*size).display_with(ctx, symbols),
                    ctx.get(*pointee).display_with(ctx, symbols)
                )
            }
            TypeData::Pointer { ty } => format!("*{}", ctx.get(*ty).display_with(ctx, symbols)),
            TypeData::Adt { kind, def_id, .. } => {
                if def_id == &DefId::SENTINEL_STR {
                    "Str".to_string()
                } else if let Some(symbols) = symbols {
                    if let Some(name) = symbols.type_name_by_def_id(*def_id) {
                        name.as_str().to_string()
                    } else {
                        match kind {
                            AdtKind::Struct => "struct".to_string(),
                            AdtKind::Enum => "enum".to_string(),
                        }
                    }
                } else {
                    match kind {
                        AdtKind::Struct => "struct".to_string(),
                        AdtKind::Enum => "enum".to_string(),
                    }
                }
            }
            TypeData::Int { bits, signed, .. } => {
                if *signed {
                    format!("Int<{}>", bits)
                } else {
                    format!("UInt<{}>", bits)
                }
            }
            TypeData::UInt { bits, .. } => format!("UInt<{}>", bits),
            TypeData::Float { bits } => format!("Float<{}>", bits),
            TypeData::Bool => "Bool".to_string(),
            TypeData::Char => "Char".to_string(),
            TypeData::Byte => "Byte".to_string(),
            TypeData::USize => "USize".to_string(),
            TypeData::Never => "!".to_string(),
            TypeData::Error => "!!".to_string(),
            TypeData::Rational {
                int_bits,
                frac_bits,
                ..
            } => format!("Rational<{}, {}>", int_bits, frac_bits),
            TypeData::InferVar { id, .. } => format!("infer#{}", id),
            // For all other types, use the Display trait (which shows TypeTag
            // for leaf types and doesn't need TypeContext<'input>).
            other => format!("{}", other),
        }
    }
}

/// Distinguishes between struct and enum ADT kinds (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdtKind {
    Struct,
    Enum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub usize);
impl DefId {
    pub const SENTINEL_STR: DefId = DefId(usize::MAX);
    pub const SENTINEL_LAYOUT_DESC: DefId = DefId(usize::MAX - 1);
}

#[derive(Debug, Clone)]
pub struct TypeMeta<'input> {
    pub default_value: Option<crate::ast::Expr<'input>>,
    pub invariant: Option<crate::ast::Expr<'input>>,
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
pub(crate) struct VarianceEdge {
    pub(crate) target: TypeId,
    /// +1 = covariant, -1 = contravariant, 0 = invariant
    pub(crate) sign: isize,
}

/// A type factory that can create new types with shared (immutable) access.
/// The type arena is wrapped in `RefCell` so that types can be created
/// without `&mut` access to the `TypeContext<'input>`.
#[derive(Debug)]
pub struct TypeFactory {
    types: RefCell<Vec<Arc<TypeData>>>,
    pub(crate) type_map: RefCell<HashMap<TypeData, TypeId>>,
    /// Index up to which types have been synced by TypeContext<'input>.
    /// `drain_new_types()` returns all types from this index onward
    /// and advances it.  This avoids the O(n) while loop in
    /// `TypeContext::alloc` that previously scanned the entire arena.
    sync_index: Cell<usize>,
}

impl<'input> TypeFactory {
    pub fn new() -> Self {
        TypeFactory {
            types: RefCell::new(Vec::new()),
            type_map: RefCell::new(HashMap::default()),
            sync_index: Cell::new(0),
        }
    }

    pub fn alloc(&self, data: TypeData) -> (TypeId, Arc<TypeData>) {
        // ═══════════════════════════════════════════════════════════════
        // Borrow order constraint: immutable borrow for volatile check
        // MUST be dropped before the mutable borrow for push below.
        // Rust's RefCell does not allow upgrading a shared borrow to an
        // exclusive one — the two must be sequential, not concurrent.
        // ═══════════════════════════════════════════════════════════════
        let types_borrow = self.types.borrow();
        let can_cache = !is_type_volatile_inner(&data, Some(&types_borrow));
        drop(types_borrow); // ← release before mutable borrow
        if can_cache {
            let type_map = self.type_map.borrow();
            if let Some(&id) = type_map.get(&data) {
                drop(type_map); // ← release before borrowing types below
                // Cache hit — return the existing Arc from the types vec.
                let types = self.types.borrow();
                return (id, types[id.index()].clone());
            }
        }
        let tag = TypeTag::from(&data) as usize;
        let mut types = self.types.borrow_mut();
        let index = types.len();
        let id = TypeId::from_raw(((index + 1) << TypeId::TAG_BITS) | tag);
        let arc = Arc::new(data.clone()); // clone for Arc (HashMap needs original by value)
        types.push(arc.clone());
        drop(types);
        if can_cache {
            self.type_map.borrow_mut().insert(data, id);
        }
        (id, arc)
    }

    pub fn len(&self) -> usize {
        self.types.borrow().len()
    }

    /// Drain newly created types since the last call to this method.
    /// Returns `Arc` clones for all types from the current sync index
    /// to the end of the arena, then advances the sync index.
    /// Used by `TypeContext<'input>` to keep its `self.types` cache in sync
    /// without scanning the entire arena on every allocation.
    pub fn drain_new_types(&self) -> Vec<Arc<TypeData>> {
        let types = self.types.borrow();
        let start = self.sync_index.get();
        let end = types.len();
        if start >= end {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            result.push(types[i].clone());
        }
        self.sync_index.set(end);
        result
    }

    pub fn borrow_types(&self) -> std::cell::Ref<'_, Vec<Arc<TypeData>>> {
        self.types.borrow()
    }

    pub fn int(&self, bits: u32, signed: bool) -> TypeId {
        self.alloc(TypeData::Int {
            bits,
            signed,
            overflow_policy: OverflowPolicy::Trap,
        })
        .0
    }

    pub fn uint(&self, bits: u32) -> TypeId {
        self.alloc(TypeData::UInt {
            bits,
            overflow_policy: OverflowPolicy::Trap,
        })
        .0
    }

    pub fn float(&self, bits: u32) -> TypeId {
        self.alloc(TypeData::Float { bits }).0
    }

    pub fn tuple(&self, elems: Vec<TypeId>) -> TypeId {
        self.alloc(TypeData::Tuple { elems }).0
    }

    pub fn array(&self, elem: TypeId, size: u64) -> TypeId {
        self.alloc(TypeData::Array { elem, size }).0
    }

    pub fn slice(&self, elem: TypeId) -> TypeId {
        self.alloc(TypeData::Slice { elem }).0
    }

    pub fn reference(&self, ty: TypeId, mutable: bool) -> TypeId {
        self.reference_with_lifetime(ty, mutable, None)
    }

    /// `&'a T` / `&'a mut T` with an EXPLICIT lifetime annotation (the
    /// region name — `'a` → interned `Symbol`), or `None` for an
    /// elided/inferred region.  The explicit annotation survives into the
    /// type so the region solver can map it to the function's early-bound
    /// UniversalRegion and verify `'a: 'b` outlives constraints
    /// (SYNTAX.md §Explicit Lifetime Parameters: "verified by the borrow
    /// checker; mismatches cause compile errors").
    pub fn reference_with_lifetime(
        &self,
        ty: TypeId,
        mutable: bool,
        lifetime: Option<Symbol>,
    ) -> TypeId {
        self.alloc(TypeData::Ref {
            ty,
            mutable,
            lifetime,
        })
        .0
    }

    pub fn pointer(&self, ty: TypeId) -> TypeId {
        self.alloc(TypeData::Pointer { ty }).0
    }

    pub fn ptr(&self, size: TypeId, pointee: TypeId) -> TypeId {
        self.alloc(TypeData::Ptr { size, pointee }).0
    }

    pub fn function(&self, params: Vec<TypeId>, ret: TypeId) -> TypeId {
        self.alloc(TypeData::Fn { params, ret }).0
    }

    pub fn regex(&self, pattern: String) -> TypeId {
        debug_assert!(
            regex_syntax::parse(&pattern).is_ok(),
            "Regex pattern must be valid"
        );
        self.alloc(TypeData::Regex { pattern }).0
    }
}
