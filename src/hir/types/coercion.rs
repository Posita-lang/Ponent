use super::*;

// ── §2.6 pointer-conversion contract (SYNTAX.md 2026-10-28) ──────────────
// Landing points for the migration slice (none of these are IN this file;
// recorded here so the contract survives until each lands):
//   • subtype.rs: `*mut T → *const T` and `Ptr<..., mut> → Ptr<..., const>`
//     are SAFE implicit coercions; the reverse requires `as!` (bitcast,
//     capsa-audit-flagged) — same pointee AND size must match (§2.6 table).
//   • checker (extern "C" ABI, §4.6): `*const T → const T*`,
//     `*mut T → T*`; `&T/&[T] → *const T`, `&mut/&mut[T] → *mut T`; a bare
//     `Ptr` whose size ≠ platform word width is FORBIDDEN in an extern "C"
//     signature — the ABI has no representation for it.
//   • checker (UnsafeCell::get, MMIO `@io`): mint via
//     `TypeContext::pointer_mut` — going through `pointer()` silently
//     loses the write intent (§2.3: mutability is part of the type).
//   • parser: bare `*T` is a DEPRECATED alias for `*const T` — emit the
//     deprecation warning; capsa audit flags it (§2.2).
// Consult `TypeContext::mutability_of_pointer` for the mutability bit —
// never pattern-match `mutable: false` (that reads `*mut` as `*const`).
//
// Types-layer support landed (use these — do not re-derive):
//   raw_ptr_shape(ty) / extern_c_ptr_size_ok(size, word_bits)  (mod.rs)
// Per-slice acceptance (each lands WITH its assertions):
//  1. checker extern "C" (§4.6): every signature arg satisfies
//     raw_ptr_shape == None || (Explicit ⇒ extern_c_ptr_size_ok(size,
//     target word bits)); render WordConst→`const T*`, WordMut→`T*`.
//     Test: extern "C" fn(Ptr<Int<8>, T>) REJECTED; fn(*mut T) renders `T*`.
//  2. UnsafeCell::get / MMIO `@io`: mint via `pointer_mut` — assertion:
//     mutability_of_pointer(result) == Some(true) (a pointer() mint
//     silently loses write intent, §2.3).
//  3. parser bare `*T`: parse as *const + deprecation warning —
//     assertion: lowered type is Pointer{mutable:false} AND warning
//     count == 1 per item (capsa audit flags it, §2.2).
//
// ── Ref three-layer consistency ledger (CLOSED as a
// documented decision) ───────────────────────────────────────────────
// The three layers answer DIFFERENT questions and are each internally
// sound; they diverge on shared-ref covariance BY DESIGN:
//  • subtype.rs: &T <: &U requires T = U — a conservative RELATION
//    (lifetime covariance handled separately via region_subtype_collect;
//    aligned with unify).
//  • polar_action.rs / variance: &T covariant, &mut T invariant — the
//    FUNCTOR semantics (θ substitution, §2.4-analog).
//  • kappa.rs: Ref is a CONSTANT constructor (severed edges, IE leaf) —
//    inhabitant counting never consumes the pointee.
// If a §2.4-class spec ever fixes Ref's variance normatively, sync all
// THREE before touching any one of them.

// The context of a unification: whether it is a call-site argument check
// (where `@auto_ro`'s `&mut T → &T` relaxation applies — SYNTAX.md "at
// function call sites and method resolution") or a structural position
// (array/ADT elements, struct fields, ...) where the relaxation must NOT
// apply.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CoercionContext {
    CallSite,
    Structural,
}

impl CoercionContext {
    /// Stored as an `AtomicU8` rather than a `Cell`: the interner half of
    /// `TypeContext` must be `Sync`, and `Cell` never is.  Only the owning
    /// thread reads/writes it, so `Relaxed` is sufficient — no data is
    /// published through this flag.
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            CoercionContext::CallSite => 0,
            CoercionContext::Structural => 1,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            0 => CoercionContext::CallSite,
            _ => CoercionContext::Structural,
        }
    }
}

// The coercion context is managed by a SCOPED closure API instead of
// raw-pointer RAII guards.  Two `FnCtxt` methods (see
// `hir/checker/fn_ctxt.rs`) swap the flag, run a closure, and restore the
// previous value:

//   * `FnCtxt::with_call_site_coercion(f)` — `CallSite` scope
//   * `FnCtxt::with_structural_coercion(f)` — `Structural` scope

// The sequential-borrow shape — swap, drop the borrow, run `f` with the
// full `&mut self`, re-borrow to store — needs NO unsafe pointer: no
// borrow of the flag lives across the closure, so `&mut self` calls
// inside `f` never conflict (the E0502 problem that motivated the old
// raw-pointer guards, whose "a held `&AtomicU8` must live across the
// loop" premise was false).  `?` inside the closure cannot skip the
// restore — it happens here, around the call — the same guarantee the
// `Drop` guards gave.

// Why the helpers live on `FnCtxt` rather than here: the flag is a field
// of `TypeContext` reached through `FnCtxt`'s `&mut self`, so a free
// function taking `&mut AtomicU8` would have to hold that borrow across
// `f` — re-creating the split-borrow conflict.  Owning the whole
// `&mut self` for the closure's duration is the only split-free shape.
