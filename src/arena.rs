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
//! 1. The inner `&Bump` is **never** handed out.  There is no accessor for
//!    it, and it is a private field, so no caller can allocate without
//!    going through [`SyncArena::alloc`].
//! 2. Every allocating method holds `alloc_lock` for the duration of the
//!    allocation.
//! 3. Reading previously allocated values needs no lock at all: the values
//!    are ordinary `Send` data living in arena memory, and allocation
//!    never moves or invalidates them.
//!
//! If a future change exposes the raw `&Bump`, or allocates outside the
//! lock, these impls become unsound.  Don't.

use bumpalo::Bump;
use parking_lot::Mutex;
use std::fmt;

/// Leak a fresh arena pair that lives for the rest of the process.
///
/// Everything the parser produces borrows the arena, so `'input` — the
/// lifetime threaded through `Program`, `SymbolTable`, `HirStmt` and
/// `TypeContext` — is the arena's lifetime.  Leaking it makes that
/// `'static`, which is what lets lowered HIR be a query value
/// (`Query::Value: Send + 'static`).
///
/// Call this **once** and reuse the result for every file: memory then
/// grows with total input and is never reclaimed per file.  That is the
/// same bargain rustc makes (its arena lives in the session, not per
/// file), and it is what makes per-function queries possible at all.
///
/// Deliberately not a `static`: `Bump` is not `Sync`, so it cannot be
/// stored in one without an `unsafe impl`.  Leaking in `main` and passing
/// the `&'static` references down needs no such escape hatch.
///
/// Do **not** call `reset` on the returned arena: results from earlier
/// files are still alive in it.
pub fn leaked() -> (&'static Bump, &'static SyncArena<'static>) {
    let bump: &'static Bump = Box::leak(Box::new(Bump::new()));
    (bump, Box::leak(Box::new(SyncArena::new(bump))))
}

/// A `Sync` (and `Send`) handle to a `bumpalo` arena.
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
}
