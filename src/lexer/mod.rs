//! Lexer module — tokenizes Posita source code.
//!
//! Architecture:
//!   token.rs   — `Token` enum (Logos), `TokenClass`, `token_class()`
//!   literal.rs — escape-sequence parsing for string/char/number literals
//!   keyword.rs — `Token::as_ident_symbol`: keyword→Symbol mapping
//!   display.rs — `Token::to_user_string`: user-friendly token display

pub mod display;
pub mod keyword;
pub mod literal;
pub mod token;

pub use token::{Token, TokenClass, token_class};

#[cfg(test)]
mod tests;
