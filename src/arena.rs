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
//! `SyncArena` removes that root cause with a hybrid allocator:
//!
//! * Small allocations use a lock-free CAS on a cursor inside a *slab*
//!   reserved from the `Bump`.  This is the hot path taken by parallel
//!   query evaluation.
//! * Reserving a slab, growing the `Bump`, and allocating anything larger
//!   than a slab allocation take `alloc_lock`.  Payload copies (strings,
//!   slices) happen *outside* the lock, so a large `alloc_str` only holds
//!   the lock for the O(1) cursor bump, not for the memcpy.
//!
//! ## Soundness
//!
//! The `unsafe impl`s below are sound only because of these invariants,
//! which this module enforces structurally:
//!
//! 1. The inner `&Bump` is exposed **only** through [`BumpScope`], which
//!    holds `alloc_lock` for its entire lifetime.  The sole widened
//!    accessor, [`BumpScope::bump_unchecked`], is `unsafe` and its safety
//!    contract is discharged by the parser at its one call site.
//! 2. Every mutation of the `Bump`'s chunk cursor happens under
//!    `alloc_lock`.  The lock-free fast path does not touch that cursor at
//!    all: it CASes an offset inside a slab that was reserved from the
//!    `Bump` while the lock was held.  The `Bump` therefore cannot hand
//!    those bytes to any other call, and concurrent fast-path allocations
//!    own disjoint ranges of the slab.
//! 3. Reading previously allocated values needs no lock at all: the values
//!    are ordinary `Send` data living in arena memory, and allocation
//!    never moves or invalidates them.
//!
//! If a future change exposes the raw `&Bump` safely, or mutates the
//! `Bump`'s cursor outside `alloc_lock`, these impls become unsound.  Don't.

use crate::sync::{Mutex, MutexGuard};
use bumpalo::Bump;
use std::alloc::Layout;
use std::fmt;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Bytes reserved from the backing `Bump` at a time for the lock-free path.
///
/// The tail of a slab that is too small for the next allocation is wasted;
/// keeping the slab modest bounds that waste while still amortizing the
/// reservation over thousands of small allocations.
const SLAB_SIZE: usize = 16 * 1024;

/// Largest allocation served from a slab.  Anything larger is reserved
/// directly from the `Bump` under `alloc_lock` (the payload copy, if any,
/// still happens outside the lock).
const MAX_SLAB_ALLOC: usize = SLAB_SIZE / 8;

/// Alignment guaranteed by a slab.  Over-aligned layouts take the locked
/// path; `bumpalo` handles arbitrary alignment there.
const SLAB_ALIGN: usize = 64;

/// One slab of memory reserved from the `Bump`.
///
/// Slabs are allocated from the arena and never freed, moved, or reused, so
/// a thread may keep using a stale `&FastSlab` after a newer slab has been
/// published.  Each slab owns its own cursor; only `base` and `cap` are
/// immutable after publication.
struct FastSlab {
    base: usize,
    cap: usize,
    cursor: AtomicUsize,
}

/// Cache-line padding for a contested field.
///
/// The allocation cursor and `alloc_lock` are written by many threads;
/// keeping them on separate lines stops a store to one from invalidating
/// the other on every core.
#[repr(align(64))]
struct Padded<T>(T);

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
/// allocation fast path or `alloc_lock`.
pub struct SyncArena<'a> {
    bump: &'a Bump,
    /// The most recently installed fast slab.  `null` until the first
    /// small allocation; installed under `alloc_lock`, read lock-free.
    fast: Padded<AtomicPtr<FastSlab>>,
    /// Serializes every mutation of the backing `Bump`: slab installation,
    /// chunk growth, and large allocations.  Never held while copying a
    /// payload and never held across user code.
    alloc_lock: Padded<Mutex<()>>,
    /// Test-only: incremented when the slow path observes `alloc_lock`
    /// already held, so the scope-blocking test can wait for a definite
    /// contention event instead of sleeping.
    #[cfg(all(test, not(feature = "shuttle")))]
    slow_contended: AtomicUsize,
}

// SAFETY: see the module docs.  `&Bump` is not `Sync` because allocation is
// unsynchronized; here every mutation of the `Bump` is serialized by
// `alloc_lock`, the raw `&Bump` is never exposed safely, and the lock-free
// path only writes into slabs exclusively reserved from the `Bump`.
unsafe impl<'a> Sync for SyncArena<'a> {}

// SAFETY: same reasoning.  `Bump` itself is already `Send`; sharing a
// `&SyncArena` across threads is safe because mutation only ever happens
// under the lock or inside a slab owned by this arena.
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
            fast: Padded(AtomicPtr::new(ptr::null_mut())),
            alloc_lock: Padded(Mutex::new(())),
            #[cfg(all(test, not(feature = "shuttle")))]
            slow_contended: AtomicUsize::new(0),
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
        self.alloc_with(move || val)
    }

    /// Construct `T` directly in arena memory.
    ///
    /// Reserving the space and writing the value are separate steps, so
    /// large values are not moved through a stack temporary and copied
    /// again; `f` runs exactly once, after the reservation succeeds.
    #[inline]
    pub fn alloc_with<F, T>(&self, f: F) -> &'a mut T
    where
        F: FnOnce() -> T,
    {
        let ptr = self.alloc_layout(Layout::new::<T>()).cast::<T>();
        // SAFETY: `alloc_layout` returned a writable, suitably aligned
        // region of at least `size_of::<T>()` bytes that no other
        // allocation can claim.  `ptr::write` does not read the old
        // contents, which are uninitialized.
        unsafe {
            ptr.as_ptr().write(f());
            &mut *ptr.as_ptr()
        }
    }

    /// Copy `s` into the arena, returning a borrow tied to `'a`.
    ///
    /// Used to give a per-file source buffer an arena lifetime, so the
    /// diagnostics that borrow it can outlive the handler that produced them
    /// (see `run_parse` / `run_check` in `main`).
    ///
    /// The reservation is made under `alloc_lock`; the `memcpy` runs after
    /// the guard is released.  A reservation is always *before* the cursor
    /// every later allocation sees, so the destination cannot alias.
    #[inline]
    pub fn alloc_str(&self, s: &str) -> &'a mut str {
        let layout = Layout::for_value(s.as_bytes());
        // Deliberately always take the locked reservation path: callers use
        // this for whole-file source buffers, so the copy dominates and we
        // avoid burning a slab on a per-file arena that only copies text.
        let dst = {
            let _guard = self.alloc_lock.0.lock();
            self.bump.alloc_layout(layout)
        };
        // SAFETY: `dst` reserves `s.len()` bytes.  The copy runs after the
        // lock is released, but the reserved range is fixed: the next
        // caller's cursor starts at or after its end, so nobody else can
        // write here.  The bytes come from a `str`, so the result is UTF-8.
        unsafe {
            ptr::copy_nonoverlapping(s.as_ptr(), dst.as_ptr(), s.len());
            std::str::from_utf8_unchecked_mut(std::slice::from_raw_parts_mut(dst.as_ptr(), s.len()))
        }
    }

    /// Clone a slice into the arena, returning a mutable slice borrow tied
    /// to `'a`.
    ///
    /// `T::clone` runs outside the lock.  If a clone panics, the elements
    /// cloned so far are leaked — the same behavior as
    /// `Bump::alloc_slice_clone`, and safe because they are never observed.
    #[inline]
    pub fn alloc_slice_clone<T: Clone>(&self, src: &[T]) -> &'a mut [T] {
        let dst = self.alloc_layout(Layout::for_value(src)).cast::<T>();
        // SAFETY: the reservation covers `src.len()` contiguous `T`s, and
        // each index is written exactly once before the slice is created.
        unsafe {
            for (index, value) in src.iter().cloned().enumerate() {
                dst.as_ptr().add(index).write(value);
            }
            std::slice::from_raw_parts_mut(dst.as_ptr(), src.len())
        }
    }

    /// Copy a `Copy` slice into the arena with one `memcpy`.
    ///
    /// Prefer this over [`alloc_slice_clone`](Self::alloc_slice_clone) when
    /// `T: Copy`: it avoids a per-element `clone` call and keeps the copy
    /// outside `alloc_lock`.
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> &'a mut [T] {
        let dst = self.alloc_layout(Layout::for_value(src)).cast::<T>();
        // SAFETY: the reservation covers `src.len()` contiguous `T`s, and
        // `src` cannot overlap arena memory reserved by this call.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src.len());
            std::slice::from_raw_parts_mut(dst.as_ptr(), src.len())
        }
    }

    /// Acquire exclusive access to the backing `Bump` for the duration of
    /// the returned [`BumpScope`].
    ///
    /// The scope holds `alloc_lock`, so while it is alive no slow-path
    /// allocation (slab installation, chunk growth, over-large values) and
    /// no [`alloc_str`](Self::alloc_str) can touch the `Bump`.  Once a slab
    /// has been installed, small [`alloc`](Self::alloc) calls may still
    /// proceed concurrently: they allocate from private slabs and never
    /// mutate the `Bump` cursor.  That is what makes it sound to hand out
    /// the raw `&Bump` through [`BumpScope::bump_unchecked`].  (A cold arena's first
    /// small allocation installs its slab under `alloc_lock`, so it waits
    /// for the scope like any other slow-path allocation.)
    ///
    /// **Do not** call [`SyncArena::alloc_str`] or allocate an over-large
    /// value on the same thread while a scope is alive: those take
    /// `alloc_lock`, which is not reentrant, so they would deadlock.  Do not
    /// nest scopes for the same reason.
    #[inline]
    pub fn scope(&'a self) -> BumpScope<'a> {
        BumpScope {
            bump: self.bump,
            _guard: self.alloc_lock.0.lock(),
        }
    }

    /// Reserve `layout` bytes without copying anything.
    #[inline]
    fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        if let Some(ptr) = self.try_alloc_fast(layout) {
            return ptr;
        }
        self.alloc_layout_slow(layout)
    }

    /// Lock-free bump within the current slab.
    ///
    /// Returns `None` when the layout is too large for a slab, when no slab
    /// has been installed yet, or when the current slab is exhausted.
    #[inline]
    fn try_alloc_fast(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() > MAX_SLAB_ALLOC || layout.align() > SLAB_ALIGN {
            return None;
        }

        let slab = self.fast.0.load(Ordering::Acquire);
        if slab.is_null() {
            return None;
        }
        // SAFETY: slabs are allocated from the `Bump`, which outlives `'a`,
        // and are never freed or moved while the arena is alive.
        let slab = unsafe { &*slab };

        let mut cursor = slab.cursor.load(Ordering::Relaxed);
        loop {
            let aligned = (cursor + (layout.align() - 1)) & !(layout.align() - 1);
            let next = aligned.checked_add(layout.size())?;
            if next > slab.cap {
                return None;
            }
            match slab.cursor.compare_exchange_weak(
                cursor,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: the successful CAS reserved `[aligned, next)`,
                    // which no other allocation can overlap.  `base` is
                    // `SLAB_ALIGN`-aligned, so the offset preserves the
                    // requested alignment.
                    let ptr = (slab.base + aligned) as *mut u8;
                    return NonNull::new(ptr);
                }
                Err(actual) => cursor = actual,
            }
        }
    }

    /// Locked path: install a new slab or reserve directly from the `Bump`.
    #[cold]
    #[inline(never)]
    fn alloc_layout_slow(&self, layout: Layout) -> NonNull<u8> {
        // Another thread may have installed a fresh slab between the fast
        // attempt in `alloc_layout` and this point.
        if let Some(ptr) = self.try_alloc_fast(layout) {
            return ptr;
        }

        #[cfg(all(test, not(feature = "shuttle")))]
        self.note_slow_contention();

        let _guard = self.alloc_lock.0.lock();

        // A thread that held the lock while we waited may have installed a
        // slab with room for this layout.
        if let Some(ptr) = self.try_alloc_fast(layout) {
            return ptr;
        }

        if layout.size() <= MAX_SLAB_ALLOC && layout.align() <= SLAB_ALIGN {
            let slab_layout = Layout::from_size_align(SLAB_SIZE, SLAB_ALIGN)
                .expect("SLAB_SIZE is a multiple of SLAB_ALIGN");
            let memory = self.bump.alloc_layout(slab_layout);
            let slab = self.bump.alloc(FastSlab {
                base: memory.as_ptr() as usize,
                cap: SLAB_SIZE,
                cursor: AtomicUsize::new(0),
            });
            let slab_ptr: *mut FastSlab = slab;
            self.fast.0.store(slab_ptr, Ordering::Release);
            if let Some(ptr) = self.try_alloc_fast(layout) {
                return ptr;
            }
        }

        self.bump.alloc_layout(layout)
    }

    /// Record that the slow path found `alloc_lock` held.  Test-only; the
    /// `try_lock` is not part of the production path.
    #[cfg(all(test, not(feature = "shuttle")))]
    fn note_slow_contention(&self) {
        if self.alloc_lock.0.try_lock().is_none() {
            self.slow_contended.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Scoped access to a [`SyncArena`]'s underlying bump.
///
/// Holding this value holds the arena's `alloc_lock`, so the raw `&Bump`
/// returned by [`bump_unchecked`](BumpScope::bump_unchecked) can be used for
/// unsynchronized allocation for as long as the scope lives.  Drop it (or
/// let it fall out of scope) to release the lock.
pub struct BumpScope<'a> {
    bump: &'a Bump,
    /// Serializes access to the `Bump`'s cursor: held for the scope's whole
    /// lifetime, so no other thread can grow the arena or allocate directly
    /// from it while we hold the raw `&Bump`.  Never read — its `Drop` is
    /// the point.
    _guard: MutexGuard<'a, ()>,
}

impl<'a> BumpScope<'a> {
    /// The arena's bump widened to the arena lifetime.
    ///
    /// This exists for bulk builders that must return values tied to the
    /// arena (`'a`) rather than to the scope borrow — notably
    /// `bumpalo::collections::Vec<'a, T>` and the parser's AST.  The
    /// reference itself must **not** escape the scope.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the returned `&'a Bump` is used only
    /// while this `BumpScope` is alive, and is never stored anywhere that
    /// can outlive it.  Allocating through the reference after the scope is
    /// dropped would mutate the `Bump` without `alloc_lock` and make the
    /// `Sync`/`Send` impls on [`SyncArena`] unsound.
    ///
    /// Values *allocated* through the reference may outlive the scope: the
    /// arena owns their memory for `'a`, and reading them needs no lock.
    #[inline]
    pub unsafe fn bump_unchecked(&self) -> &'a Bump {
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
//   mutation of the `Bump` must pass through `alloc_lock`.  If bumpalo ever
//   makes `Bump` `Sync` (or stops it being `Send`), the build fails and the
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
// The fast-path metadata must stay shareable without a lock; a raw pointer
// creeping back into `FastSlab` would silently break that.
static_assertions::assert_impl_all!(FastSlab: Send, Sync);

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    fn leaked_arena() -> &'static SyncArena<'static> {
        let bump: &'static Bump = Box::leak(Box::new(Bump::new()));
        Box::leak(Box::new(SyncArena::new(bump)))
    }

    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Aligned64(u64);

    #[repr(align(128))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Aligned128(u64);

    /// Larger than `MAX_SLAB_ALLOC`, so allocation must take the slow path.
    #[repr(C)]
    struct Big([u8; MAX_SLAB_ALLOC + 1]);

    /// Many threads hammering the CAS cursor must never receive overlapping
    /// ranges.  Every allocation stays alive until the end of the test, so a
    /// repeated address is a real overlap, not allocator reuse.
    #[test]
    fn concurrent_fast_path_allocations_are_disjoint() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 4096;

        let arena = leaked_arena();
        let handles: Vec<_> = (0..THREADS)
            .map(|thread_idx| {
                thread::spawn(move || {
                    let mut addresses = Vec::with_capacity(PER_THREAD);
                    for value_idx in 0..PER_THREAD {
                        let value = arena.alloc((thread_idx, value_idx));
                        assert_eq!(*value, (thread_idx, value_idx));
                        addresses.push(value as *mut (usize, usize) as usize);
                    }
                    addresses
                })
            })
            .collect();

        let mut addresses = Vec::with_capacity(THREADS * PER_THREAD);
        for handle in handles {
            addresses.extend(handle.join().unwrap());
        }
        addresses.sort_unstable();
        let allocated = addresses.len();
        addresses.dedup();
        assert_eq!(
            addresses.len(),
            allocated,
            "two allocations returned the same address"
        );
    }

    /// The scope only excludes mutations of the backing `Bump`; small
    /// allocations from a private slab must not block behind it.
    #[test]
    fn scope_does_not_block_fast_path_allocation() {
        let arena = leaked_arena();
        // Install the first slab before taking the scope: installing a slab
        // is itself a slow-path operation and correctly waits for the lock.
        let _ = arena.alloc(0u8);
        let scope = arena.scope();

        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let value = arena.alloc(7u64);
            tx.send(*value).unwrap();
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5))
                .expect("fast-path allocation blocked behind a BumpScope"),
            7
        );

        drop(scope);
        handle.join().unwrap();
    }

    /// Allocations that must grow the `Bump` still serialize with the scope.
    #[test]
    fn scope_blocks_slow_path_allocation() {
        let arena = leaked_arena();
        let scope = arena.scope();

        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let value = arena.alloc(Big([0; MAX_SLAB_ALLOC + 1]));
            tx.send(value.0[0]).unwrap();
        });

        // Wait until the worker has actually reached the slow path and
        // observed the held lock.  At that point it cannot have completed,
        // so the empty-channel assertion below is deterministic rather than
        // a sleep-based guess.
        let deadline = Instant::now() + Duration::from_secs(5);
        while arena.slow_contended.load(Ordering::SeqCst) == 0 {
            assert!(
                Instant::now() < deadline,
                "worker never reached the slow allocation path"
            );
            thread::yield_now();
        }

        match rx.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {}
            Ok(_) => panic!("slow-path allocation completed while the scope held alloc_lock"),
            Err(mpsc::TryRecvError::Disconnected) => panic!("allocation worker panicked"),
        }

        drop(scope);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
        handle.join().unwrap();
    }

    #[test]
    fn alloc_str_and_slices_round_trip() {
        let arena = leaked_arena();

        let short = arena.alloc_str("hello");
        assert_eq!(short, "hello");

        // Larger than a slab, so this exercises reserve-under-lock followed
        // by a copy outside it.
        let long_source = "x".repeat(MAX_SLAB_ALLOC * 4);
        let long = arena.alloc_str(&long_source);
        assert_eq!(long, long_source.as_str());

        let values = [String::from("alpha"), String::from("beta")];
        let cloned = arena.alloc_slice_clone(&values);
        assert_eq!(cloned, values.as_slice());

        let copied = arena.alloc_slice_copy(&[1u32, 2, 3]);
        assert_eq!(copied, &[1, 2, 3]);
    }

    #[test]
    fn alloc_with_constructs_in_place_and_honors_alignment() {
        let arena = leaked_arena();

        let calls = AtomicUsize::new(0);
        let built = arena.alloc_with(|| {
            calls.fetch_add(1, Ordering::SeqCst);
            String::from("built")
        });
        assert_eq!(built, "built");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        for _ in 0..64 {
            let fast = arena.alloc(Aligned64(1));
            assert_eq!(fast as *mut Aligned64 as usize % 64, 0);
            // Alignment beyond the slab guarantee takes the locked path.
            let slow = arena.alloc(Aligned128(2));
            assert_eq!(slow as *mut Aligned128 as usize % 128, 0);
            assert_eq!(fast.0, 1);
            assert_eq!(slow.0, 2);
        }
    }
}
