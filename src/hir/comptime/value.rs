use crate::hir::generate::TypeInfo;
use crate::hir::target::layout::LayoutDescriptor;
use crate::hir::types::TypeId;
use crate::symbol::Symbol;
use std::sync::Arc;

/// The result of evaluating a comptime expression.
#[derive(Debug, Clone)]
pub enum ComptimeValue {
    /// Unit `()` value.
    Unit,
    /// A boolean literal.
    Bool(bool),
    /// An integer literal.
    Int(i128),
    /// A floating-point literal.
    Float(f64),
    /// A string literal, stored as `Arc<str>` to avoid deep copies on clone.
    String(Arc<str>),
    /// A type value (returned by type factories).
    Type(TypeId),
    /// A structured type info value (returned by `@typeInfo!`).
    TypeInfo(Box<TypeInfo>),
    /// A layout descriptor (returned by `layout_of!`).
    LayoutDescriptor(Box<LayoutDescriptor>),
    /// A comptime pointer value (created by `&x` / `&mut x`).
    /// `mutable` indicates whether the reference is `&mut` (can be written through).
    /// Dereference and index operations always read the current value from the
    /// evaluator's variable table, not from a snapshot stored in the pointer.
    Pointer { name: Symbol, mutable: bool },
    /// A comptime aggregate value: struct, tuple, or array.
    /// Fields are named for structs, positional for tuples/arrays.
    Aggregate {
        fields: Vec<(Symbol, ComptimeValue)>,
    },
}

impl ComptimeValue {
    /// Estimate the "own size" of this value in bytes — the memory that would
    /// be freed if this specific reference were dropped.
    ///
    /// For `Arc<str>` this includes both the shared string content and the Arc
    /// metadata (pointer + reference counts + length).  Counting shared data
    /// prevents large strings from trivially bypassing the memory limit, and
    /// the slight over-counting across duplicate Arcs is acceptable for an
    /// approximate limit.
    ///
    /// Used to enforce comptime memory limits.
    pub fn memory_size(&self) -> usize {
        match self {
            ComptimeValue::Unit => 0,
            ComptimeValue::Bool(_) => 1,
            ComptimeValue::Int(_) => 16,  // i128
            ComptimeValue::Float(_) => 8, // f64
            // Arc<str>: metadata (16 B) + string content length.
            // Including the content prevents large strings from
            // trivially bypassing the memory limit.  Over-counting
            // across shared Arcs is acceptable for a limit that
            // only needs to be approximate.
            ComptimeValue::String(s) => {
                // Arc<str>: metadata (16 B) + string content length.
                // Including the content prevents large strings from
                // trivially bypassing the memory limit.  Over-counting
                // across shared Arcs is acceptable for a limit that
                // only needs to be approximate.
                16 + s.len()
            }
            ComptimeValue::Type(_) => 8, // TypeId
            ComptimeValue::TypeInfo(info) => {
                let mut size = 64; // TypeInfo struct overhead
                for f in &info.fields {
                    size += f.name.len() + 8; // name + TypeId
                }
                for v in &info.variants {
                    size += v.name.len() + 8;
                    for f in &v.payload {
                        size += f.name.len() + 8;
                    }
                }
                size
            }
            ComptimeValue::LayoutDescriptor(desc) => {
                let mut size = 32; // LayoutDescriptor struct overhead
                for f in &desc.fields {
                    size += f.name.len() + 24; // name + offset + size + align
                }
                size
            }
            ComptimeValue::Pointer { .. } => 16, // name (Symbol) + mutable (bool)
            ComptimeValue::Aggregate { fields } => {
                let mut size = 32; // Vec overhead
                // Field names are Symbol (u32, Copy), zero-cost to clone.
                // NOTE: Vec capacity may exceed `len`, so actual memory
                // usage is slightly underestimated (typically by ≤1×).
                // This is acceptable for an approximate sandbox limit.
                for (_, val) in fields {
                    size += 4; // Symbol is a u32
                    size += val.memory_size();
                }
                size
            }
        }
    }
}
