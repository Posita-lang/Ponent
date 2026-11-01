//! Diagnostic renderers — pure formatters, one per output format.

pub mod fmt;
pub mod html;
pub mod human;
pub mod json;
pub mod short;

pub use html::HtmlRenderer;
pub use human::HumanRenderer;
pub use json::JsonRenderer;
pub use short::ShortRenderer;
