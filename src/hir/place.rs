//! Shared place utilities — the `place_is_prefix_of` predicate is used by
//! both the checker and the CFG borrow-check post-pass.  It lives in its
//! own leaf module so the checker's module cycle with `cfg_graph` does not
//! force a duplicated copy  .

use crate::hir::types::FrozenPlace;

/// Whether `target` is a prefix of (or equal to) `frozen` — i.e. whether
/// accessing `target` would touch the storage of the frozen place.
///
/// The invariant: `target` is a prefix of `frozen` iff `target == frozen`
/// OR `target` is a prefix of `frozen`'s immediate base.  The previous
/// arm-by-arm structural matching compared FIELD NAMES at each level
/// (`tf == ff && ...`), which conflated "same field name" with "same
/// structural position" — `a.c` was wrongly a prefix of `a.b.c` (a false
/// positive) and `a.b` was wrongly NOT a prefix of `a.b.c.d` (a false
/// negative).
///
/// Index semantics (mirroring rustc's `ProjectionElem::Index` vs
/// `ConstantIndex`): a CONSTANT index (`a[0]`) is an exact place, so
/// `a[0]` is NOT a prefix of `a[1]`; a DYNAMIC index (`a[i]`) may equal
/// ANY element, so it conservatively overlaps every constant index on the
/// same base (in both directions — freezing `a[i]` freezes `a[0]`, and
/// freezing `a[0]` freezes `a[i]`).
pub(crate) fn place_is_prefix_of(target: &FrozenPlace, frozen: &FrozenPlace) -> bool {
    if target == frozen {
        return true;
    }
    // Dynamic-index conservatism: `a[i]` may equal `a[0]` (or any other
    // element), so a dynamic and a constant index on the same base overlap
    // in BOTH directions.
    match (target, frozen) {
        (FrozenPlace::Index(t), FrozenPlace::ConstIndex(f, _))
        | (FrozenPlace::ConstIndex(f, _), FrozenPlace::Index(t)) => {
            if place_is_prefix_of(t, f) || place_is_prefix_of(f, t) {
                return true;
            }
        }
        _ => {}
    }
    match frozen {
        FrozenPlace::Field(base, _)
        | FrozenPlace::Index(base)
        | FrozenPlace::ConstIndex(base, _)
        | FrozenPlace::Deref(base) => place_is_prefix_of(target, base),
        _ => false,
    }
}
