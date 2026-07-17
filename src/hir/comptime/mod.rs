pub mod error;
pub mod eval;
pub mod value;

pub use error::ComptimeError;
pub use eval::ComptimeEvalContext;
pub use value::ComptimeValue;

#[cfg(test)]
mod tests;
