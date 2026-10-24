//! Diagnostic renderers — pure formatters, one per output format.

pub mod html;
pub mod human;
pub mod json;

pub use html::HtmlRenderer;
pub use human::HumanRenderer;
pub use json::JsonRenderer;
