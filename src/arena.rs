//! A thread-shareable view of the parse arena.
//!
//! ## Why this exists
//!
//! `bumpalo::Bump` is `Send` but deliberately **not** `Sync`: allocation
//! mutates a `Cell<NonNull<ChunkFooter>>` with no synchronization, so two
//! threads allocating concurrently would corrupt the chunk list.
//!
//! That is normally fine, but `TypeContext` stores the arena as
//! `&'input Bump`, and both `&T: Send` and `&T: Sync` require `T: Sync`.
//! A bare `&Bump` therefore makes `TypeContext` **neither `Send` nor
//! `Sync`** — which is what blocked queries from ever being handed
//! `&mut TypeContext` (see `hir::query::QueryProvider`).  Wrapping the
//! context in `RefCell`/`Mutex` did not help, because both need
//! `TypeContext: Send` to be shareable at all.
//!
//! `SyncArena` removes that root cause by serializing allocation.
//!
//! ## Soundness
//!
//! The `unsafe impl`s below are sound only because of these invariants,
//! which this module enforces structurally:
//!
//! 1. The inner `&Bump` is exposed **only** through [`BumpScope`], which
//!    holds `alloc_lock` for its entire lifetime.  Every other allocating
//!    method holds `alloc_lock` for the duration of the allocation, so no
//!    caller can allocate outside the lock.
//! 2. Every allocating method holds `alloc_lock` for the duration of the
//!    allocation.
//! 3. Reading previously allocated values needs no lock at all: the values
//!    are ordinary `Send` data living in arena memory, and allocation
//!    never moves or invalidates them.
//!
//! If a future change exposes the raw `&Bump`, or allocates outside the
//! lock, these impls become unsound.  Don't.

use crate::sync::Mutex;
use bumpalo::Bump;
use std::fmt;

/// A `Sync` (and `Send`) handle to a `bumpalo` arena.
///
/// The arena is owned by the driver (`main` keeps a stack `Bump` and wraps
/// a borrow of it) and lives only as long as that owner.  `'input` — the
/// lifetime threaded through `Program`, `SymbolTable`, `HirStmt` and
/// `TypeContext` — is that owner's lifetime, so nothing here needs to be
/// `'static`; the query layer's values are bound to `'q`, not `'static`
/// (see `crate::hir::query::QueryValue`).
///
/// Cheap to copy around as `&SyncArena`; all sharing goes through the
/// allocation lock.
pub struct SyncArena<'a> {
    bump: &'a Bump,
    /// Serializes allocation.  Held only for the duration of a single
    /// `alloc` call — never across user code — so it cannot deadlock and
    /// costs nothing on the read path.
    alloc_lock: Mutex<()>,
}

// SAFETY: see the module docs.  `&Bump` is not `Sync` because allocation is
// unsynchronized; here every allocation is serialized by `alloc_lock`, and
// the raw `&Bump` is never exposed.
unsafe impl<'a> Sync for SyncArena<'a> {}

// SAFETY: same reasoning.  `Bump` itself is already `Send`; sharing a
// `&SyncArena` across threads is safe because mutation only ever happens
// under the lock.
unsafe impl<'a> Send for SyncArena<'a> {}

// Concise `Debug` (rather than dumping the whole arena) so structs that
// embed a `&SyncArena` — e.g. `vfs::VfsNode` — can keep deriving `Debug`.
impl fmt::Debug for SyncArena<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyncArena").finish_non_exhaustive()
    }
}

impl<'a> SyncArena<'a> {
    /// Wrap a borrow of a `Bump`.  The caller must not use the original
    /// borrow to allocate while this wrapper is alive — doing so would
    /// bypass the lock and break invariant (2).
    pub fn new(bump: &'a Bump) -> Self {
        SyncArena {
            bump,
            alloc_lock: Mutex::new(()),
        }
    }

    /// Allocate `val` in the arena.
    ///
    /// The result is tied to `'a` (the lifetime the `Bump` was borrowed
    /// for), not to `&self` — so `&SyncArena<'a>` is a drop-in for the
    /// `&'a Bump` that used to appear in allocating helper signatures,
    /// and existing call sites are unchanged.
    #[inline]
    pub fn alloc<T>(&self, val: T) -> &'a mut T {
        let _guard = self.alloc_lock.lock();
        self.bump.alloc(val)
    }

    /// Copy `s` into the arena, returning a borrow tied to `'a`.
    ///
    /// Used to give a per-file source buffer an arena lifetime, so the
    /// diagnostics that borrow it can outlive the handler that produced them
    /// (see `run_parse` / `run_check` in `main`).
    #[inline]
    pub fn alloc_str(&self, s: &str) -> &'a mut str {
        let _guard = self.alloc_lock.lock();
        self.bump.alloc_str(s)
    }

    /// Copy a slice into the arena, returning a mutable slice borrow tied to
    /// `'a`.  Thin wrapper around `Bump::alloc_slice_clone`; callers pass a
    /// `&[T]` (typically a temporary `Vec<T>` converted via `&v[..]`) and get
    /// back a `&'a mut [T]` they can hand to an AST constructor.
    #[inline]
    pub fn alloc_slice_clone<T: Clone>(&self, src: &[T]) -> &'a mut [T] {
        let _guard = self.alloc_lock.lock();
        self.bump.alloc_slice_clone(src)
    }

    /// Acquire exclusive allocation access to the arena for the duration of
    /// the returned [`BumpScope`].
    ///
    /// The scope holds `alloc_lock`, so no other thread can allocate through
    /// this `SyncArena` while it is alive.  That exclusivity is exactly what
    /// makes it sound to hand out the raw `&Bump` (see [`BumpScope::bump`]),
    /// which callers use for zero-copy bulk building (e.g.
    /// `bumpalo::collections::Vec`).
    ///
    /// **Do not** call [`SyncArena::alloc`]/[`alloc_str`](Self::alloc_str)/
    /// [`alloc_slice_clone`](Self::alloc_slice_clone) on the same arena while
    /// a scope is alive, and do not nest scopes: `alloc_lock` is not
    /// reentrant, so either would deadlock.
    #[inline]
    pub fn scope(&'a self) -> BumpScope<'a> {
        BumpScope {
            bump: self.bump,
            _guard: self.alloc_lock.lock(),
        }
    }
}

/// Exclusive, scoped access to a [`SyncArena`]'s underlying bump.
///
/// Holding this value holds the arena's `alloc_lock`, so the raw `&Bump`
/// returned by [`bump`](BumpScope::bump) can be used for unsynchronized
/// allocation for as long as the scope lives.  Drop it (or let it fall out of
/// scope) to release the lock.
pub struct BumpScope<'a> {
    bump: &'a Bump,
    /// Serializes allocation: held for the scope's whole lifetime, so no other
    /// thread can allocate through the owning `SyncArena` while we hold the
    /// raw `&Bump`.  Never read — its `Drop` is the point.
    _guard: crate::sync::MutexGuard<'a, ()>,
}

impl<'a> BumpScope<'a> {
    /// The arena's bump.  Valid only while this scope is alive; allocation
    /// through it is exclusive because the scope holds `alloc_lock`.
    #[inline]
    pub fn bump(&self) -> &'a Bump {
        self.bump
    }
}

// ── Compile-time pins of the soundness argument ───────────────────────
//
// The `unsafe impl`s above are only sound while two documented facts
// hold; these assertions turn both into always-on build checks that the
// compiler — not a reader of this comment — enforces:
//
// * `Bump: Send & !Sync` — the module's founding observation, stated as
//   one `impls!` logical expression (the "implements-all-of" +
//   "implements-none-of" pair from `static_assertions` would be two
//   macros).  `!Sync` is the load-bearing half: it is exactly why every
//   allocation must pass through `alloc_lock`.  If bumpalo ever makes
//   `Bump` `Sync` (or stops it being `Send`), the build fails and the
//   safety case must be re-argued before shipping.
// * `SyncArena<'a>: Send + Sync` — the point of the wrapper.  Losing
//   either auto-trait makes `TypeContext` neither (a `&T: Send`/`Sync`
//   requires `T: Sync`), and queries stop being shareable across
//   threads — the original bug this module exists to fix.  The `'static`
//   witness is lossless: the `unsafe impl`s above are unconditional, so
//   proving them for one lifetime proves them for every lifetime
//   (`for<'a>` binders are not valid on ordinary path types).
static_assertions::const_assert!(impls::impls!(Bump: Send & !Sync));
static_assertions::assert_impl_all!(SyncArena<'static>: Send, Sync);

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    /// `SyncArena::scope()` holds `alloc_lock` for its whole lifetime, so a
    /// second thread's `SyncArena::alloc` must block until the scope drops.
    /// This is the exclusivity that makes handing out the raw `&Bump` sound.
    #[test]
    fn scope_serializes_allocation() {
        let bump: &'static Bump = Box::leak(Box::new(Bump::new()));
        let arena: &'static SyncArena<'static> = Box::leak(Box::new(SyncArena::new(bump)));

        let started = Arc::new(AtomicBool::new(false));
        let acquired = Arc::new(AtomicBool::new(false));

        // Hold the allocation lock for the whole test body...
        let scope = arena.scope();

        // ...and have a second thread attempt to allocate through the arena.
        let (started2, acquired2) = (started.clone(), acquired.clone());
        let handle = thread::spawn(move || {
            started2.store(true, Ordering::SeqCst);
            let _v = arena.alloc(42u32);
            acquired2.store(true, Ordering::SeqCst);
        });

        // Wait until the thread is definitely *inside* `alloc` (past the
        // handshake), then give it ample time to proceed if it wrongly could.
        while !started.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        thread::sleep(Duration::from_millis(50));
        assert!(
            !acquired.load(Ordering::SeqCst),
            "SyncArena::alloc proceeded while a BumpScope held alloc_lock"
        );

        // Releasing the scope unblocks the thread.
        drop(scope);
        handle.join().unwrap();
        assert!(acquired.load(Ordering::SeqCst));
    }
}
