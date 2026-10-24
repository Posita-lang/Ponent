use super::*;

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

// B7: the coercion context is managed by a SCOPED closure API instead of
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
