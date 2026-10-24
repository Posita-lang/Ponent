//! Value erasure for the query cache: the two-path split behind
//! `provided_to_erased`.
//!
//! A query's `compute` returns a *provided* value; the cache stores an
//! *erased* one.  For most queries those coincide up to the single `Arc`
//! wrap the framework has always done.  For arena queries the provided
//! value is already a borrow (`&'q V`) and wrapping it in an `Arc` would
//! be pure overhead — the erased form is the reference itself, passed
//! through with zero cost.
//!
//! The split is selected per query by [`Query::Storage`][crate::hir::query::Query::Storage]:
//!
//! - [`Owned`] — today's path: `provided_to_erased` is the one `Arc::new`
//!   wrap, `erased_to_provided` the one value clone (the cache always
//!   holds a handle, so the `Arc` never unwraps — see the impl).
//! - [`Arena`] — the borrow path: both directions are identity (a lifetime
//!   shortening for `&'a V` with `'a: 'q`, a copy for the `&'q V` handle).
//!
//! The two markers make the paths coherence-disjoint (`Owned` vs `Arena`
//! as `Self`), so no specialization is needed: a query whose value is a
//! non-`'static` borrow simply fails the `Owned` bound at its `impl` and
//! the compiler points it at `Arena`.

use std::fmt::Debug;
use std::sync::Arc;

use super::{QueryValue, StableHash};

/// Converts a query's provided value into the form the cache stores,
/// and back.  Implemented once per storage path ([`Owned`], [`Arena`]),
/// selected by the query — never per query type, so query authors only
/// name the path.
///
/// `Provided` is what `compute` returns; `Erased` is what the merged
/// cache, the waiter latches, and `get_shared` traffic in.  Both stay
/// under the arena lifetime `'q`: erased handles never outlive the
/// arena the values borrow from.
pub trait Erasable<'q, Provided>: Sized {
    /// What the cache holds.  `Arc<V>` on the owned path, `&'q V` on
    /// the arena path (a `Copy` handle — no refcount, no allocation).
    type Erased: Clone + Debug + Send + Sync + 'q;

    /// Provider side → cache side.  Runs once per fresh compute, inside
    /// the staging critical section.
    fn provided_to_erased(provided: Provided) -> Self::Erased;
    /// Cache side → provider side.  Runs on every serve; must stay
    /// cheap.  "Cheap" is per-storage, not per-trait: a refcount bump
    /// (or a plain copy) on [`Owned`], a bare pointer copy on
    /// [`Arena`].  The [`Owned`] impl does clone the VALUE (see its own
    /// note) — that is the one deep clone in the system, which is why
    /// hot read paths prefer [`QuerySystem::get_shared`]: the erased
    /// handle itself is always just an `Arc` bump.
    fn erased_to_provided(erased: &Self::Erased) -> Provided;
}

/// Today's path: the provided value is owned (possibly
/// borrowing the arena through its own parameters, e.g. `FnBodyCheck<'q>`)
/// and the cache holds the single shared `Arc` the framework has always
/// stored.  No `'static` bound: the `Arc` lives in `'q`-parameterized
/// tables next to the arena its referent borrows from.
pub struct Owned;

impl<'q, V> Erasable<'q, V> for Owned
where
    V: QueryValue<'q>,
{
    type Erased = Arc<V>;

    #[inline]
    fn provided_to_erased(provided: V) -> Self::Erased {
        Arc::new(provided)
    }

    #[inline]
    fn erased_to_provided(erased: &Self::Erased) -> V {
        // The cache holds a handle by construction (the merged entry, the
        // waiter latches and `get_shared` traffic all keep one), so
        // `erased`'s referent can never be the last `Arc`.  Going through
        // `Arc::unwrap_or_clone(erased.clone())` therefore always takes
        // the clone branch — but pays two atomic RMWs (the temp clone's
        // increment and its drop's decrement) for the `try_unwrap` that
        // can never succeed.  Clone the value directly instead.
        (**erased).clone()
    }
}

/// The borrow path: the provided value is already `&'q V` (allocated by
/// `compute` in the session arena, e.g. via [`crate::arena::SyncArena`])
/// and the cache stores the reference itself — no `Arc`, no wrap.
///
/// `compute` hands back some `&'a V` with `'a: 'q`; the shortening
/// coercion at the call boundary makes it `&'q V`, so both directions
/// below are exact-`'q` identities (a copy out, a copy back).
pub struct Arena;

impl<'q, V> Erasable<'q, &'q V> for Arena
where
    V: Debug + Send + Sync + 'q + StableHash,
{
    type Erased = &'q V;

    #[inline]
    fn provided_to_erased(provided: &'q V) -> Self::Erased {
        provided
    }

    #[inline]
    fn erased_to_provided(erased: &Self::Erased) -> &'q V {
        // `erased` is `&&'q V` and the result is `&'q V`, so this is one
        // pointer copy either way: the explicit `*` shown here, or the
        // deref coercion Rust would otherwise apply at the return
        // position.  Written out deliberately — a reader who "simplifies"
        // it to a `(**erased).clone()` would introduce the one deep clone
        // the trait exists to keep OFF this path.
        *erased
    }
}
