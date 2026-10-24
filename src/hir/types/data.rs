use crate::sync::Mutex;
use rustc_hash::FxHashMap as HashMap;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;

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
    /// highest *real* variant.  B2: `TypeKind = 31` is a REAL variant
    /// (`TypeData::Type` / `TypeData::Opaque` map to it), so the bound is
    /// TypeKind, not Regex — the old `Regex` bound fired the `tag()`
    /// debug assert on every `Type`/`Opaque`-tagged `TypeId` (TAITs are a
    /// real code path).  All 32 five-bit patterns are real variants now;
    /// there is no padding tag left, so the NEXT variant must bump
    /// `TAG_BITS` to 6.
    const TAG_LAST_VARIANT: usize = TypeTag::TypeKind as usize;
    /// Bounds check: every valid tag must be ≤ `TAG_LAST_VARIANT`.
    const TAG_MAX: usize = Self::TAG_LAST_VARIANT;

    /// Sentinel representing "no type" (error nodes, untyped expressions).
    /// Occupies the `NonZeroUsize` niche so that `Option<TypeId>` is 8 bytes
    /// instead of 16 — every type allocated through `TypeContext::alloc` has
    /// `raw >= 32` (since `index` is offset by 1).
    ///
    /// Chosen as `usize::MAX` so that a stray call to `tag()` or `index()`
    /// on this sentinel produces an obviously-wrong value: `tag()` returns
    /// discriminant 31 (`TypeKind` — a nonsense tag for a sentinel; the
    /// `debug_assert!` at the top of `tag()` fires first anyway), and
    /// `index()` returns `usize::MAX >> 5 - 1`, an unreachable index.
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
        args: SmallVec<[TypeId; 4]>,
    },
    /// (Mac Lane §III.4): the product a × b is the universal
    /// object with projections p: a×b → a, q: a×b → b, satisfying
    /// C(c, a×b) ≅ C(c, a) × C(c, b).  Dual to Coproduct (§III.3).    
    Tuple {
        elems: SmallVec<[TypeId; 4]>,
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
        params: SmallVec<[TypeId; 4]>,
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
        alternatives: SmallVec<[TypeId; 4]>,
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
        quantifiers: SmallVec<[(usize, Symbol); 2]>,
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
        pattern: Symbol,
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
    ///
    /// Returns [`Cow`] so that constant renderings (and interned ADT names)
    /// borrow a `&'static str` instead of allocating; only computed
    /// renderings own a `String`.
    pub fn display_with<'input>(
        &self,
        ctx: &TypeContext<'input>,
        symbols: Option<&crate::hir::symbol::SymbolTable>,
    ) -> Cow<'static, str> {
        match self {
            TypeData::Ref { ty, mutable, .. } => {
                let inner = ctx.get(*ty).display_with(ctx, symbols);
                if *mutable {
                    Cow::Owned(format!("&mut {}", inner))
                } else {
                    Cow::Owned(format!("&{}", inner))
                }
            }
            TypeData::Slice { elem } => {
                let inner = ctx.get(*elem).display_with(ctx, symbols);
                Cow::Owned(format!("&[{}]", inner))
            }
            TypeData::Array { elem, size } => {
                let inner = ctx.get(*elem).display_with(ctx, symbols);
                Cow::Owned(format!("[{}; {}]", inner, size))
            }
            TypeData::Tuple { elems } => {
                let parts: Vec<Cow<'static, str>> = elems
                    .iter()
                    .map(|e| ctx.get(*e).display_with(ctx, symbols))
                    .collect();
                Cow::Owned(format!("({})", parts.join(", ")))
            }
            TypeData::Fn { params, ret } => {
                let params: Vec<Cow<'static, str>> = params
                    .iter()
                    .map(|p| ctx.get(*p).display_with(ctx, symbols))
                    .collect();
                let ret = ctx.get(*ret).display_with(ctx, symbols);
                Cow::Owned(format!("fn({}) -> {}", params.join(", "), ret))
            }
            TypeData::Ptr { size, pointee } => Cow::Owned(format!(
                "Ptr<size={}, pointee={}>",
                ctx.get(*size).display_with(ctx, symbols),
                ctx.get(*pointee).display_with(ctx, symbols)
            )),
            TypeData::Pointer { ty } => {
                Cow::Owned(format!("*{}", ctx.get(*ty).display_with(ctx, symbols)))
            }
            TypeData::Adt { kind, def_id, .. } => {
                if def_id == &DefId::SENTINEL_STR {
                    Cow::Borrowed("Str")
                } else if let Some(symbols) = symbols {
                    if let Some(name) = symbols.type_name_by_def_id(*def_id) {
                        Cow::Borrowed(name.as_str())
                    } else {
                        match kind {
                            AdtKind::Struct => Cow::Borrowed("struct"),
                            AdtKind::Enum => Cow::Borrowed("enum"),
                        }
                    }
                } else {
                    match kind {
                        AdtKind::Struct => Cow::Borrowed("struct"),
                        AdtKind::Enum => Cow::Borrowed("enum"),
                    }
                }
            }
            TypeData::Int { bits, signed, .. } => {
                if *signed {
                    Cow::Owned(format!("Int<{}>", bits))
                } else {
                    Cow::Owned(format!("UInt<{}>", bits))
                }
            }
            TypeData::UInt { bits, .. } => Cow::Owned(format!("UInt<{}>", bits)),
            TypeData::Float { bits } => Cow::Owned(format!("Float<{}>", bits)),
            TypeData::Bool => Cow::Borrowed("Bool"),
            TypeData::Char => Cow::Borrowed("Char"),
            TypeData::Byte => Cow::Borrowed("Byte"),
            TypeData::USize => Cow::Borrowed("USize"),
            TypeData::Never => Cow::Borrowed("!"),
            TypeData::Error => Cow::Borrowed("!!"),
            TypeData::Rational {
                int_bits,
                frac_bits,
                ..
            } => Cow::Owned(format!("Rational<{}, {}>", int_bits, frac_bits)),
            TypeData::InferVar { id, .. } => Cow::Owned(format!("infer#{}", id)),
            // For all other types, use the Display trait (which shows TypeTag
            // for leaf types and doesn't need TypeContext<'input>).
            other => Cow::Owned(format!("{}", other)),
        }
    }
}

/// Distinguishes between struct and enum ADT kinds (rustc-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdtKind {
    Struct,
    Enum,
}

/// Rigidity of a type variable, distinguishing user-specified (rigid) types
/// from inferred (wobbly) types (Peyton Jones et al. PLDI'06 §2).
///
/// In GADT type inference, type refinement applies only to *rigid* types —
/// those explicitly specified by the user in annotations, signatures, or
/// `when` constraints.  *Wobbly* types (created by the inference engine)
/// are never refined by GADT equalities, preventing unsound refinement of
/// inferred types that the programmer did not commit to.
///
/// # Examples
/// - A user-written `fn foo(x: Int<32>)` declares `x`'s type as **Rigid**.
/// - An inferred type variable `?a` created by the engine is **Wobbly**.
/// - A GADT `when T == Int<32>` constraint refines `T` only if `T` is rigid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rigidity {
    /// The type was explicitly specified by the user (annotation, signature,
    /// or `when` constraint).  GADT refinement applies to rigid types.
    Rigid,
    /// The type was inferred by the type inference engine.  GADT refinement
    /// must not apply to wobbly types — they represent unconstrained
    /// inference variables that the programmer has not committed to.
    Wobbly,
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

/// The SHARED half of the type system: the interner.
///
/// One `TypeFactory` is created per compilation and held behind an `Arc` by
/// every `TypeContext<'input>` — the global context and each per-query one.
/// Sharing it is what keeps `TypeId`s globally comparable: two contexts
/// that intern `Int<32>` separately would mint two different ids and every
/// `TypeId` equality check would silently disagree.
///
/// The arena is a `boxcar::Vec`: an append-only, chunked vec whose reads
/// are lock-free and whose `push` takes `&self`.  So the interner grows
/// *without* a read lock, and `TypeContext::get` still hands out a plain
/// `&TypeData` — the two properties the query system needs.
///
/// Everything in here is either read-mostly (behind a `RwLock`, so a
/// concurrent reader never blocks another reader) or atomic.  Nothing here
/// is per-query inference state: that lives on `TypeContext<'input>`, so
/// each query gets its own bindings and can run in parallel.
pub struct TypeFactory<'input> {
    pub(crate) types: boxcar::Vec<Arc<TypeData>>,
    /// The intern map (structural type → id).  A `Mutex`, not a `RwLock`:
    /// `alloc` is a check-then-push, and two threads that both miss would
    /// push two identical `TypeData`s and hand out two different `TypeId`s
    /// — silently breaking every `TypeId` equality check downstream.  The
    /// lock serialises the whole `alloc`, so interning stays atomic.  It is
    /// uncontended today (queries run serially) and costs one
    /// uncontended acquire per new type.
    pub(crate) type_map: Mutex<HashMap<TypeData, TypeId>>,
    /// The regex-pattern validation cache (pattern → valid).
    pub(crate) regex_cache: RwLock<HashMap<Symbol, bool>>,
    /// Per-`TypeId` refinement metadata (invariant, default value).
    pub(crate) meta: RwLock<HashMap<TypeId, TypeMeta<'input>>>,
    /// `DefId` → the `TypeId` minted for it.
    pub(crate) def_id_to_type_id: RwLock<HashMap<DefId, TypeId>>,
    /// Registered ADT definitions (def_id → fields + Drop).
    pub(crate) adt_defs: RwLock<HashMap<DefId, AdtDef>>,
    /// (param_index, TypeId, expected_sign, cumulative_sign) → variance check result.
    pub(crate) variance_cache: RwLock<HashMap<(usize, TypeId, isize, isize), bool>>,
    /// Pre-computed variance-annotated outgoing edges per `TypeId`.
    pub(crate) variance_edges: RwLock<HashMap<TypeId, Vec<VarianceEdge>>>,
    /// Universe counter for Higher-Ranked Type skolemization.  Shared so
    /// that universes are unique across per-query contexts.
    pub(crate) next_universe: AtomicUsize,
    /// Counter for GADT existential skolem IDs.
    pub(crate) next_gadt_skolem_id: AtomicUsize,
    /// Counter for generating fresh parameter indices (Exists/Forall).
    pub(crate) next_param_index: AtomicUsize,
    /// Counter for overlap/specialization freshening variables.  Base
    /// offset mirrors the former global `OVERLAP_FRESH_VAR_ID` (1_000_000).
    pub(crate) next_overlap_fresh_id: AtomicUsize,
}

impl<'input> TypeFactory<'input> {
    pub fn new() -> Self {
        TypeFactory {
            types: boxcar::Vec::new(),
            type_map: Mutex::new(HashMap::default()),
            regex_cache: RwLock::new(HashMap::default()),
            meta: RwLock::new(HashMap::default()),
            def_id_to_type_id: RwLock::new(HashMap::default()),
            adt_defs: RwLock::new(HashMap::default()),
            variance_cache: RwLock::new(HashMap::default()),
            variance_edges: RwLock::new(HashMap::default()),
            next_universe: AtomicUsize::new(0),
            next_gadt_skolem_id: AtomicUsize::new(0),
            next_param_index: AtomicUsize::new(0),
            next_overlap_fresh_id: AtomicUsize::new(1_000_000),
        }
    }

    pub fn alloc(&self, data: TypeData) -> (TypeId, Arc<TypeData>) {
        let can_cache = !is_type_volatile_inner(&data, Some(&self.types));
        if can_cache {
            let mut type_map = self.type_map.lock();
            if let Some(&id) = type_map.get(&data) {
                // Cache hit — return the existing Arc from the arena.
                return (id, self.types.get(id.index()).expect("cached id").clone());
            }
            let tag = TypeTag::from(&data) as usize;
            let arc = Arc::new(data.clone()); // clone for Arc (HashMap needs original by value)
            // `push` returns the index it claimed, so the id is derived from
            // the allocation itself rather than from a racy `len()` read.
            let index = self.types.push(arc.clone());
            let id = TypeId::from_raw(((index + 1) << TypeId::TAG_BITS) | tag);
            type_map.insert(data, id);
            return (id, arc);
        }
        let tag = TypeTag::from(&data) as usize;
        let arc = Arc::new(data);
        let index = self.types.push(arc.clone());
        let id = TypeId::from_raw(((index + 1) << TypeId::TAG_BITS) | tag);
        (id, arc)
    }

    pub fn len(&self) -> usize {
        self.types.count()
    }

    pub fn borrow_types(&self) -> &boxcar::Vec<Arc<TypeData>> {
        &self.types
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

    pub fn tuple(&self, elems: SmallVec<[TypeId; 4]>) -> TypeId {
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

    pub fn function(&self, params: SmallVec<[TypeId; 4]>, ret: TypeId) -> TypeId {
        self.alloc(TypeData::Fn { params, ret }).0
    }

    pub fn regex(&self, pattern: Symbol) -> TypeId {
        debug_assert!(
            regex_syntax::parse(pattern.as_str()).is_ok(),
            "Regex pattern must be valid"
        );
        self.alloc(TypeData::Regex { pattern }).0
    }
}
