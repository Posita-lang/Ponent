pub mod error;
pub mod value;
pub mod eval;

pub use error::ComptimeError;
pub use value::ComptimeValue;
pub use eval::ComptimeEvalContext;

#[cfg(test)]
mod tests;
