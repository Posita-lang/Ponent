//! Ponent compiler — library root.
//!
//! Re-exports dependencies that are consumed by exported macros so that
//! downstream crates using `ponent::tr!` do not need to declare them
//! themselves (avoids macro-dependency leakage).
pub use smallvec;
