use super::*;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering as AtomicOrdering;

/// The context of a unification: whether it is a call-site argument check
/// (where `@auto_ro`'s `&mut T → &T` relaxation applies — SYNTAX.md "at
/// function call sites and method resolution") or a structural position
/// (array/ADT elements, struct fields, ...) where the relaxation must NOT
/// apply.
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

/// RAII: mark the enclosing unification as a call-site argument check for
/// the duration of the scope.  `@auto_ro`'s implicit freeze is thereby
/// confined to function call sites (SYNTAX.md), never structural positions.
///
/// The guard holds a raw pointer to the context FLAG (not the whole
/// `TypeContext<'input>`).  WHY a raw pointer rather than a shared `&AtomicU8`
/// reference: a held `&AtomicU8` borrow of
/// `self.checker.ctx.current_coercion_ctx` would live across the fallible
/// argument-checking loop and conflict with the enclosing `&mut self`
/// calls inside it (E0502 — the field is reached through `&mut self`, so
/// the borrow checker cannot split it while the guard is alive).  The raw
/// address is borrow-free: `?` early-returns cannot skip the restore, and
/// the enclosing `&mut self` calls do not conflict.  (SAFETY: the flag
/// belongs to the `TypeContext<'input>`, which outlives this method-local guard;
/// atomicity makes the write well-defined under `&TypeContext<'input>`.)
pub(crate) struct CallSiteCoercion {
    ctx: *const AtomicU8,
    prev: CoercionContext,
}

impl CallSiteCoercion {
    pub fn enter<'input>(ctx: &TypeContext<'input>) -> Self {
        let prev = CoercionContext::from_u8(
            ctx.current_coercion_ctx
                .swap(CoercionContext::CallSite.to_u8(), AtomicOrdering::Relaxed),
        );
        CallSiteCoercion {
            ctx: &ctx.current_coercion_ctx as *const AtomicU8,
            prev,
        }
    }
}

impl Drop for CallSiteCoercion {
    fn drop(&mut self) {
        // SAFETY: the flag belongs to the `TypeContext<'input>` that outlives this
        // method-local guard; the pointer was taken from a shared borrow of
        // the field inside `enter`, and atomicity makes the write
        // well-defined under `&TypeContext<'input>`.
        unsafe { &*self.ctx }.store(self.prev.to_u8(), AtomicOrdering::Relaxed);
    }
}

/// RAII: mark the enclosing unification as a structural position (not a
/// call site) for the duration of the scope.  Data-constructor positions
/// (struct fields, ADT payloads, array elements) must NOT inherit the
/// `CallSite` coercion context — if they did, `@auto_ro`'s implicit freeze
/// would be applied to nested positions, bypassing SYNTAX.md's scoping
/// requirement (freeze only at function call sites).
///
/// Same lifetime discipline as `CallSiteCoercion`: the raw pointer avoids
/// re-borrowing the flag under a live `&mut self` in the enclosing
/// method (see `CallSiteCoercion` doc).
pub(crate) struct StructuralCoercion {
    ctx: *const AtomicU8,
    prev: CoercionContext,
}

impl StructuralCoercion {
    pub fn enter<'input>(ctx: &TypeContext<'input>) -> Self {
        let prev = CoercionContext::from_u8(
            ctx.current_coercion_ctx
                .swap(CoercionContext::Structural.to_u8(), AtomicOrdering::Relaxed),
        );
        StructuralCoercion {
            ctx: &ctx.current_coercion_ctx as *const AtomicU8,
            prev,
        }
    }
}

impl Drop for StructuralCoercion {
    fn drop(&mut self) {
        // SAFETY: same as `CallSiteCoercion` — the flag belongs to the
        // `TypeContext<'input>` that outlives this method-local guard.
        unsafe { &*self.ctx }.store(self.prev.to_u8(), AtomicOrdering::Relaxed);
    }
}
