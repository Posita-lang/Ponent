pub mod checker;
pub mod cfg;
pub mod cfg_graph;
pub mod type_eq;
mod comptime;
mod generate;
mod hir;
mod infer;
pub(crate) mod loop_ir;
pub(crate) mod octagon;
mod resolver;
mod shape_var;
pub mod target;
mod smt;
mod symbol;
mod traits;
mod types;
pub mod visit;

// (｡•̀ᴗ-)✧  Waku waku!  Only compiles in debug builds.
#[cfg(debug_assertions)]
pub mod anya;

pub use comptime::*;
pub use generate::*;
pub use hir::*;
pub use infer::*;
pub use resolver::*;
pub use shape_var::*;
pub use smt::*;
pub use symbol::*;
pub use traits::*;
pub use types::*;
