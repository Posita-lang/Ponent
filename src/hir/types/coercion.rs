use super::*;

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

/// RAII: mark the enclosing unification as a call-site argument check for
/// the duration of the scope.  `@auto_ro`'s implicit freeze is thereby
/// confined to function call sites (SYNTAX.md), never structural positions.
///
/// The guard holds a raw pointer to the context CELL (not the whole
/// `TypeContext<'input>`).  WHY a raw pointer rather than a shared `&Cell`
/// reference: a held `&Cell<CoercionContext>` borrow of
/// `self.checker.ctx.current_coercion_ctx` would live across the fallible
/// argument-checking loop and conflict with the enclosing `&mut self`
/// calls inside it (E0502 — the field is reached through `&mut self`, so
/// the borrow checker cannot split it while the guard is alive).  The raw
/// address is borrow-free: `?` early-returns cannot skip the restore, and
/// the enclosing `&mut self` calls do not conflict.  (SAFETY: the `Cell`
/// belongs to the `TypeContext<'input>`, which outlives this method-local guard;
/// the interior mutability makes the write well-defined under
/// `&TypeContext<'input>`.)
pub(crate) struct CallSiteCoercion {
    ctx: *const std::cell::Cell<CoercionContext>,
    prev: CoercionContext,
}

impl CallSiteCoercion {
    pub fn enter<'input>(ctx: &TypeContext<'input>) -> Self {
        let prev = ctx.current_coercion_ctx.replace(CoercionContext::CallSite);
        CallSiteCoercion {
            ctx: &ctx.current_coercion_ctx as *const std::cell::Cell<CoercionContext>,
            prev,
        }
    }
}

impl Drop for CallSiteCoercion {
    fn drop(&mut self) {
        // SAFETY: the cell belongs to the `TypeContext<'input>` that outlives this
        // method-local guard; the pointer was taken from a shared borrow of
        // the field inside `enter`, and the cell's interior mutability makes
        // the write well-defined under `&TypeContext<'input>`.
        unsafe { &*self.ctx }.set(self.prev);
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
/// re-borrowing the `Cell` under a live `&mut self` in the enclosing
/// method (see `CallSiteCoercion` doc).
pub(crate) struct StructuralCoercion {
    ctx: *const std::cell::Cell<CoercionContext>,
    prev: CoercionContext,
}

impl StructuralCoercion {
    pub fn enter<'input>(ctx: &TypeContext<'input>) -> Self {
        let prev = ctx
            .current_coercion_ctx
            .replace(CoercionContext::Structural);
        StructuralCoercion {
            ctx: &ctx.current_coercion_ctx as *const std::cell::Cell<CoercionContext>,
            prev,
        }
    }
}

impl Drop for StructuralCoercion {
    fn drop(&mut self) {
        // SAFETY: same as `CallSiteCoercion` — the cell belongs to the
        // `TypeContext<'input>` that outlives this method-local guard.
        unsafe { &*self.ctx }.set(self.prev);
    }
}
