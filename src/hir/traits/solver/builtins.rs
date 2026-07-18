use crate::hir::types::{DefId, TypeContext, TypeData, TypeId};
use crate::symbol::Symbol;
use rustc_hash::FxHashMap as HashMap;

/// Registry of builtin trait DefIds, populated at startup.
/// Maps DefId → BuiltinTrait for fast O(1) lookup during candidate assembly.
pub struct BuiltinTraitRegistry {
    map: HashMap<DefId, BuiltinTrait>,
}

impl BuiltinTraitRegistry {
    pub fn new() -> Self {
        BuiltinTraitRegistry {
            map: HashMap::default(),
        }
    }

    /// Register a builtin trait by its DefId and name.
    /// Returns `true` if the name is a known builtin, `false` otherwise.
    pub fn register(&mut self, def_id: DefId, name: &Symbol) -> bool {
        if let Some(builtin) = BuiltinTrait::identify(name) {
            self.map.insert(def_id, builtin);
            true
        } else {
            false
        }
    }

    /// Look up a builtin trait by its DefId.
    pub fn lookup(&self, def_id: DefId) -> Option<BuiltinTrait> {
        self.map.get(&def_id).copied()
    }

    /// Check if a DefId corresponds to a known builtin trait.
    pub fn is_builtin(&self, def_id: DefId) -> bool {
        self.map.contains_key(&def_id)
    }

    /// Clear the registry (for testing or re-initialization).
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

/// Built-in traits known to the compiler.
///
/// These correspond to the traits listed in SYNTAX.md § Traits and
/// Implementations — Built-in Traits: Add, Sub, Mul, Div, Rem, Eq, Ord,
/// Copy, Clone, Default, Drop, Deref, Display, Serialize, Write.
///
/// The `identify` function maps a trait DefId to its BuiltinTrait kind
/// by checking the trait's name against a predefined set. This is a
/// data-driven approach: new builtins are added by extending this enum
/// and the `identify` function, not by scattering match arms across the
/// solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinTrait {
    // ── Marker traits ──
    Sized,
    Copy,
    Clone,
    Drop,
    Default,

    // ── Operator traits ──
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Eq,
    Ord,

    // ── Indexing ──
    Index,
    IndexMut,

    // ── Deref ──
    Deref,

    // ── Display ──
    Display,

    // ── Serialization ──
    Serialize,
    Write,
}

impl BuiltinTrait {
    /// Try to identify a builtin trait by its name.
    /// Returns `None` if the trait is user-defined.
    pub fn identify(name: &Symbol) -> Option<BuiltinTrait> {
        match &*name.as_str() {
            "Sized" => Some(BuiltinTrait::Sized),
            "Copy" => Some(BuiltinTrait::Copy),
            "Clone" => Some(BuiltinTrait::Clone),
            "Drop" => Some(BuiltinTrait::Drop),
            "Default" => Some(BuiltinTrait::Default),
            "Add" => Some(BuiltinTrait::Add),
            "Sub" => Some(BuiltinTrait::Sub),
            "Mul" => Some(BuiltinTrait::Mul),
            "Div" => Some(BuiltinTrait::Div),
            "Rem" => Some(BuiltinTrait::Rem),
            "Neg" => Some(BuiltinTrait::Neg),
            "Eq" => Some(BuiltinTrait::Eq),
            "Ord" => Some(BuiltinTrait::Ord),
            "Index" => Some(BuiltinTrait::Index),
            "IndexMut" => Some(BuiltinTrait::IndexMut),
            "Deref" => Some(BuiltinTrait::Deref),
            "Display" => Some(BuiltinTrait::Display),
            "Serialize" => Some(BuiltinTrait::Serialize),
            "Write" => Some(BuiltinTrait::Write),
            _ => None,
        }
    }

    /// Whether this builtin trait is coinductive (auto‑trait / marker).
    /// Coinductive traits treat cycles in the proof tree as success.
    pub fn is_coinductive(&self) -> bool {
        matches!(
            self,
            BuiltinTrait::Sized | BuiltinTrait::Copy | BuiltinTrait::Clone
        )
    }
}

/// Determine whether a type is `Sized` by construction.
///
/// All concrete types are Sized. The only unsized types in Posita are:
/// - `[T]` (slices, no length)
/// - `dyn Trait` (trait objects)
/// - `Str` (unsized string slice)
///
/// Inference variables are conservatively considered Sized (they will be
/// resolved to concrete types). This is sound because Posita does not
/// allow unsized inference variables — all unsized types are explicit.
pub fn compute_sized(ty: TypeId, ctx: &TypeContext) -> bool {
    !is_unsized(ty, ctx)
}

/// Check whether a type is unsized (cannot be used as a value type).
///
/// Returns `true` for:
/// - `[T]` (slices)
/// - `dyn Trait` (trait objects)
/// - `Str` (string slices — unsized in Posita per SYNTAX.md)
///
/// Inference variables are conservatively considered Sized (returns `false`).
pub fn is_unsized(ty: TypeId, ctx: &TypeContext) -> bool {
    match ctx.get(ty) {
        TypeData::Slice { .. } | TypeData::DynTrait { .. } => true,
        _ => {
            // Str is an unsized string slice type in Posita, stored as an Adt.
            // Check by comparing with the builtin_str TypeId.
            ty == ctx.builtin_str
        }
    }
}

/// Determine whether a type is `Copy` by structural analysis.
///
/// A type is Copy if:
/// 1. It is a primitive (`Int`, `UInt`, `Float`, `Bool`, `Char`, `Byte`, `USize`)
/// 2. It is a tuple of Copy types
/// 3. It is an array of Copy types
/// 4. It is a reference (`&T` is always Copy; `&mut T` is NOT Copy)
/// 5. It is a struct/enum where ALL fields/variants are Copy AND
///    the type does NOT implement `Drop`
///
/// If the type is an inference variable, returns `false` conservatively.
///
/// NOTE: For ADTs (structs/enums), this function conservatively returns `false`
/// because checking field types requires access to the SymbolTable for type
/// bindings, and checking for Drop impls requires access to the TraitEnv.
/// A full implementation would:
///   - Look up the ADT definition via SymbolTable
///   - Recursively check that every field is Copy
///   - Reject any ADT with an explicit Drop impl
/// Until then, returning `false` is the safe conservative choice — denying
/// Copy is always sound (it just means more explicit Clone calls).
pub fn compute_copy(ty: TypeId, ctx: &TypeContext) -> bool {
    match ctx.get(ty) {
        // Primitives: always Copy
        TypeData::Int { .. }
        | TypeData::UInt { .. }
        | TypeData::Float { .. }
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit => true,

        // Immutable references are always Copy
        TypeData::Ref { mutable: false, .. } => true,

        // Mutable references are NOT Copy (exclusive borrow)
        TypeData::Ref { mutable: true, .. } => false,

        // Raw pointers are Copy (bitwise copy is safe)
        TypeData::Pointer { .. } | TypeData::Ptr { .. } => true,

        // Tuple: Copy iff all elements are Copy
        TypeData::Tuple { elems } => elems.iter().all(|e| compute_copy(*e, ctx)),

        // Array: Copy iff element is Copy
        TypeData::Array { elem, .. } => compute_copy(*elem, ctx),

        // Slice: NOT Copy (unsized, cannot be copied)
        TypeData::Slice { .. } => false,

        // ADT: conservatively NOT Copy.
        // A full implementation would check all fields recursively and
        // verify no Drop impl, but that requires SymbolTable + TraitEnv
        // access which this function doesn't have.  Denying Copy is the
        // sound conservative choice.
        TypeData::Adt { .. } => false,

        // Functions: always Copy
        TypeData::Fn { .. } => true,

        // DynTrait: NOT Copy (trait objects are reference-like)
        TypeData::DynTrait { .. } => false,

        // Everything else: conservatively not Copy
        _ => false,
    }
}

/// Determine whether a type automatically derives `Clone`.
///
/// In Posita, if a type is `Copy`, then `Clone` is automatically derived
/// with `fn clone(&self) -> Self { *self }` (see SYNTAX.md § Automatic
/// Clone for Copy Types).
pub fn compute_clone(ty: TypeId, ctx: &TypeContext) -> bool {
    compute_copy(ty, ctx)
}
