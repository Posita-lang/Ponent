//! Crate-wide synchronization primitives with a feature-gated backend.
//!
//! Default build: a direct re-export of `parking_lot` — same types, zero
//! wrappers, zero behavior change. With `--features shuttle` (the dual-track
//! lane) the same names become parking_lot-shaped newtypes over
//! `shuttle::sync`, so every use site compiles unchanged while any blocking
//! that happens inside a `shuttle::check` execution is visible to the
//! deterministic scheduler. Poisoning is flattened to parking_lot semantics
//! (`into_inner`), because shuttle's backend mirrors `std` and parking_lot
//! has no poisoning at all.

#[cfg(not(feature = "shuttle"))]
pub use parking_lot::{Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "shuttle")]
mod shuttle_backend {
    use core::ops::{Deref, DerefMut};

    use shuttle::sync as sh;

    #[derive(Debug)]
    pub struct Mutex<T: ?Sized>(sh::Mutex<T>);

    impl<T> Mutex<T> {
        #[must_use]
        pub const fn new(value: T) -> Self {
            Self(sh::Mutex::new(value))
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> MutexGuard<'_, T> {
            MutexGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut().unwrap_or_else(|e| e.into_inner())
        }
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    pub struct MutexGuard<'a, T: ?Sized>(sh::MutexGuard<'a, T>);

    impl<T: ?Sized> Deref for MutexGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }

    #[derive(Debug)]
    pub struct RwLock<T: ?Sized>(sh::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        pub const fn new(value: T) -> Self {
            Self(sh::RwLock::new(value))
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> RwLockReadGuard<'_, T> {
            RwLockReadGuard(self.0.read().unwrap_or_else(|e| e.into_inner()))
        }

        pub fn write(&self) -> RwLockWriteGuard<'_, T> {
            RwLockWriteGuard(self.0.write().unwrap_or_else(|e| e.into_inner()))
        }
    }

    impl<T: Default> Default for RwLock<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    pub struct RwLockReadGuard<'a, T: ?Sized>(sh::RwLockReadGuard<'a, T>);

    impl<T: ?Sized> Deref for RwLockReadGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &T {
            &self.0
        }
    }

    pub struct RwLockWriteGuard<'a, T: ?Sized>(sh::RwLockWriteGuard<'a, T>);

    impl<T: ?Sized> Deref for RwLockWriteGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl<T: ?Sized> DerefMut for RwLockWriteGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }

    #[derive(Debug)]
    pub struct Condvar(sh::Condvar);

    impl Condvar {
        #[must_use]
        pub const fn new() -> Self {
            Self(sh::Condvar::new())
        }

        /// In-place wait matching parking_lot's shape: the guard never moves
        /// ownership, so call sites compile unchanged across backends.
        pub fn wait<T>(&self, guard: &mut MutexGuard<'_, T>) {
            // SAFETY: `guard` is exclusively borrowed for the whole call, so
            // no alias can observe the moved-out slot; the value is written
            // back before the borrow ends (no panic path between read/write —
            // `into_inner` on the poison error returns normally).
            let owned = unsafe { core::ptr::read(&guard.0) };
            let owned = self.0.wait(owned).unwrap_or_else(|e| e.into_inner());
            unsafe {
                core::ptr::write(&mut guard.0, owned);
            }
        }

        pub fn notify_all(&self) {
            self.0.notify_all();
        }
    }

    impl Default for Condvar {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "shuttle")]
pub use shuttle_backend::{Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

// ── Compile-time backend-parity contracts ─────────────────────────────
//
// The re-exports/newtypes above are a promise: whatever the default
// parking_lot backend does, the `shuttle_backend` newtypes must do
// identically — otherwise `--features shuttle` stops being a faithful
// mirror and its deterministic-scheduler runs stop being representative.
// These `static_assertions` checks compile under BOTH feature
// configurations, so a backend that silently loses `Send`/`Sync` (e.g.
// by wrapping a type that holds a raw pointer) fails the build here, at
// the contract site, instead of at some far-away feature-gated call site.
//
// `String` is the witness type: an ordinary `Send + Sync` payload, the
// narrowest stand-in for the bounds each primitive documents.
//
// The shared part of the contract — the locks and the condition variable
// themselves, which threads share across the whole crate:
static_assertions::assert_impl_all!(Condvar: Send, Sync);
static_assertions::assert_impl_all!(Mutex<String>: Send, Sync);
static_assertions::assert_impl_all!(RwLock<String>: Send, Sync);

// Guards' surface is pinned per backend — and the pins turned out to be
// IDENTICAL on both sides: every guard here is `!Send + Sync`.
// parking_lot's guards are `!Send` through lock_api's `GuardNoSend(*mut
// ())` phantom — the same pthread release-on-acquiring-thread
// restriction that makes `std`'s guards `!Send` (the shuttle mirror
// therefore inherits it) — and both are `Sync` once the data is.  These
// assertions caught exactly this during introduction: the first draft
// pinned parking_lot's guards as `Send + Sync`, and the build rejected
// it right here, at the contract site.  Call sites must hold guards in
// place across backends anyway (see `Condvar::wait` above), so only the
// locks carry the shared contract.
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_not_impl_all!(MutexGuard<'static, String>: Send);
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_impl_all!(MutexGuard<'static, String>: Sync);
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_not_impl_all!(RwLockReadGuard<'static, String>: Send);
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_impl_all!(RwLockReadGuard<'static, String>: Sync);
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_not_impl_all!(RwLockWriteGuard<'static, String>: Send);
#[cfg(not(feature = "shuttle"))]
static_assertions::assert_impl_all!(RwLockWriteGuard<'static, String>: Sync);

#[cfg(feature = "shuttle")]
static_assertions::assert_not_impl_all!(MutexGuard<'static, String>: Send);
#[cfg(feature = "shuttle")]
static_assertions::assert_impl_all!(MutexGuard<'static, String>: Sync);
#[cfg(feature = "shuttle")]
static_assertions::assert_not_impl_all!(RwLockReadGuard<'static, String>: Send);
#[cfg(feature = "shuttle")]
static_assertions::assert_impl_all!(RwLockReadGuard<'static, String>: Sync);
#[cfg(feature = "shuttle")]
static_assertions::assert_not_impl_all!(RwLockWriteGuard<'static, String>: Send);
#[cfg(feature = "shuttle")]
static_assertions::assert_impl_all!(RwLockWriteGuard<'static, String>: Sync);
