use crate::diagnostics::DiagCtxt;
use crate::hir::symbol::SymbolTable;
use crate::hir::target::Target;
use crate::hir::types::{TypeContext, TypeData, TypeId};

/// Compute the ABI size (in bytes) of a type on the given target.
///
/// For ADT types (structs/enums), requires a `SymbolTable` to resolve
/// type bindings.  When `symbols` is `None`, falls back to a pointer-sized
/// estimate for ADT types.
pub fn abi_size(
    ctx: &mut TypeContext,
    target: &Target,
    ty: TypeId,
    symbols: Option<&SymbolTable>,
    diag: &mut DiagCtxt,
) -> Option<u64> {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::Int { bits, .. } | TypeData::UInt { bits, .. } => Some(target.int_abi_size(bits)),
        TypeData::Float { bits } => Some(target.float_abi_size(bits)),
        TypeData::Bool | TypeData::Byte | TypeData::Char => Some(1),
        TypeData::USize => Some(target.ptr_size()),
        TypeData::Ptr { size, .. } => abi_size(ctx, target, size, symbols, diag),
        TypeData::Pointer { .. } => Some(target.ptr_size()),
        TypeData::Ref { .. } => Some(target.ptr_size()),
        TypeData::Fn { .. } => Some(target.ptr_size()),
        TypeData::Array { elem, size } => {
            abi_size(ctx, target, elem, symbols, diag).map(|s| s * size)
        }
        TypeData::Slice { .. } => Some(target.ptr_size() * 2),
        TypeData::Tuple { elems } => {
            let mut total: u64 = 0;
            let mut max_align: u64 = 1;
            for e in &elems {
                let field_size = abi_size(ctx, target, *e, symbols, diag)?;
                let field_align = abi_alignment(ctx, target, *e, symbols, diag)?;
                total = align_up(total, field_align);
                total += field_size;
                max_align = max_align.max(field_align);
            }
            let struct_align = max_align.min(target.max_align());
            total = align_up(total, struct_align);
            Some(total)
        }
        TypeData::Adt { .. } => {
            // Delegate to the layout engine when the SymbolTable is available.
            if let Some(s) = symbols {
                let layout =
                    crate::hir::target::layout::compute_adt_layout(ctx, Some(s), target, ty, diag)?;
                Some(layout.size)
            } else {
                // Without a SymbolTable we cannot resolve TypeBindings, and
                // a pointer-sized fallback would be wildly wrong for large
                // structs/enums.  Returning None forces callers to handle
                // the un-resolved-ADT case explicitly.
                None
            }
        }
        TypeData::Never | TypeData::Unit => Some(0),
        TypeData::GenericParam { .. }
        | TypeData::AssociatedType { .. }
        | TypeData::InferVar { .. }
        | TypeData::SkolemVar { .. } => None,
        _ => None,
    }
}

/// Compute the ABI alignment (in bytes) of a type on the given target.
pub fn abi_alignment(
    ctx: &mut TypeContext,
    target: &Target,
    ty: TypeId,
    symbols: Option<&SymbolTable>,
    diag: &mut DiagCtxt,
) -> Option<u64> {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::Int { bits, .. } | TypeData::UInt { bits, .. } => {
            Some(target.int_abi_align(bits))
        }
        TypeData::Float { bits } => Some(target.float_abi_align(bits)),
        TypeData::Bool | TypeData::Byte | TypeData::Char => Some(1),
        TypeData::USize => Some(target.ptr_align()),
        TypeData::Ptr { size, .. } => abi_alignment(ctx, target, size, symbols, diag),
        TypeData::Pointer { .. } => Some(target.ptr_align()),
        TypeData::Ref { .. } => Some(target.ptr_align()),
        TypeData::Fn { .. } => Some(target.ptr_align()),
        TypeData::Array { elem, .. } => abi_alignment(ctx, target, elem, symbols, diag),
        TypeData::Slice { .. } => Some(target.ptr_align()),
        TypeData::Tuple { elems } => {
            let mut max_align: u64 = 1;
            for e in &elems {
                if let Some(a) = abi_alignment(ctx, target, *e, symbols, diag) {
                    max_align = max_align.max(a);
                }
            }
            Some(max_align.min(target.max_align()))
        }
        TypeData::Adt { .. } => {
            if let Some(s) = symbols {
                let layout =
                    crate::hir::target::layout::compute_adt_layout(ctx, Some(s), target, ty, diag)?;
                Some(layout.align)
            } else {
                Some(target.ptr_align())
            }
        }
        TypeData::Never | TypeData::Unit => Some(1),
        TypeData::GenericParam { .. }
        | TypeData::AssociatedType { .. }
        | TypeData::InferVar { .. }
        | TypeData::SkolemVar { .. } => None,
        _ => None,
    }
}

/// Round `offset` up to the next multiple of `alignment`.
pub fn align_up(offset: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    if alignment <= 1 {
        return offset;
    }
    let mask = alignment - 1;
    let addend = offset.checked_add(mask).unwrap_or(u64::MAX);
    addend & !mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::target::Target;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
    }
}
