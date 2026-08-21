use crate::ast::{GenericArg, Type};
use crate::diagnostics::DiagCtxt;
use crate::hir::symbol::SymbolTable;
use crate::hir::target::Target;
use crate::hir::types::{TypeContext, TypeData, TypeId};

/// The computed layout of a type: its size, alignment, and field offsets.
#[derive(Debug, Clone)]
pub struct LayoutDescriptor {
    /// Total size in bytes.
    pub size: u64,
    /// ABI alignment in bytes.
    pub align: u64,
    /// Field layouts (empty for primitives, populated for structs/tuples).
    pub fields: Vec<FieldLayout>,
}

/// Layout information for a single field.
#[derive(Debug, Clone)]
pub struct FieldLayout {
    /// Field name (e.g. `"x"`, `"_0"` for tuple fields).
    pub name: String,
    /// Byte offset from the start of the containing type.
    pub offset: u64,
    /// Size of the field in bytes.
    pub size: u64,
    /// Alignment of the field in bytes.
    pub align: u64,
}

/// The computed layout of an enum type.
#[derive(Debug, Clone)]
pub struct EnumLayout {
    /// Total size in bytes.
    pub size: u64,
    /// ABI alignment in bytes.
    pub align: u64,
    /// Discriminant size in bytes.
    pub discr_size: u64,
    /// Discriminant alignment in bytes.
    pub discr_align: u64,
    /// Layout of each variant's payload.
    pub variants: Vec<VariantLayout>,
}

/// Layout information for a single enum variant's payload.
#[derive(Debug, Clone)]
pub struct VariantLayout {
    /// Variant name.
    pub name: String,
    /// Byte offset of this variant's payload within the overall enum.
    pub offset: u64,
    /// Size of this variant's payload.
    pub size: u64,
    /// Fields within this variant's payload.
    pub fields: Vec<FieldLayout>,
}

// ── Public API ───────────────────────────────────────────────────

/// Compute the full layout of any ADT type (struct, enum, or tuple)
/// by looking up the type definition in the symbol table.
///
/// Returns `None` if the type is not an ADT or the symbol table is
/// unavailable.
pub fn compute_adt_layout<'input>(
    ctx: &mut TypeContext<'input>,
    symbols: Option<&SymbolTable<'input>>,
    target: &Target,
    ty: TypeId,
    diag: &mut DiagCtxt,
) -> Option<LayoutDescriptor> {
    let resolved = ctx.resolve_binding(ty);
    let def_id = match ctx.get(resolved) {
        TypeData::Adt { def_id, .. } => *def_id,
        TypeData::Tuple { elems } => {
            // Treat tuples as anonymous structs with _0, _1, ... fields.
            let field_types: Vec<(String, TypeId)> = elems
                .iter()
                .enumerate()
                .map(|(i, e)| (format!("_{}", i), *e))
                .collect();
            return compute_struct_layout(
                ctx,
                target,
                &field_types,
                false,
                None,
                false,
                false,
                None,
                diag,
            );
        }
        _ => {
            // Primitive types (Int, Float, Bool, Pointer, etc.) — return
            // a simple layout with their ABI size and alignment.
            let size = super::abi::abi_size(ctx, target, ty, symbols, diag)?;
            let align = super::abi::abi_alignment(ctx, target, ty, symbols, diag)?;
            return Some(LayoutDescriptor {
                size,
                align,
                fields: Vec::new(),
            });
        }
    };

    let binding = symbols?.lookup_type_by_def_id(def_id)?;
    match binding.kind {
        crate::hir::symbol::TypeKind::Struct => {
            let field_types: Vec<(String, TypeId)> = binding
                .fields
                .iter()
                .map(|f| (f.name.as_str(), f.ty))
                .collect();
            compute_struct_layout(
                ctx,
                target,
                &field_types,
                binding.packed,
                binding.align,
                binding.c_layout,
                binding.transparent,
                symbols,
                diag,
            )
        }
        crate::hir::symbol::TypeKind::Enum => {
            // Compute the discriminant size: default to Int<32> (4 bytes).
            let discr_size = target.int_abi_size(32);
            let discr_align = target.int_abi_align(32);

            // Collect variant payloads with full type resolution.
            let variant_payloads: Vec<(String, Vec<(String, TypeId)>)> = binding
                .variants
                .iter()
                .map(|v| match &v.payload {
                    Some(payload_ty) => {
                        let fields = resolve_ast_type_payload(payload_ty, ctx, symbols);
                        (v.name.as_str(), fields)
                    }
                    None => (v.name.as_str(), Vec::new()),
                })
                .collect();
            // Sync any types created by resolve_ast_type_payload via ctx.alloc()
            // into ctx.types so they are visible to subsequent ctx.get() calls.
            ctx.sync_factory();

            let enum_layout = compute_enum_layout(
                ctx,
                target,
                discr_size,
                discr_align,
                &variant_payloads,
                binding.packed,
                binding.align,
                symbols,
                diag,
            )?;

            // Convert EnumLayout to LayoutDescriptor for the struct-like case.
            Some(LayoutDescriptor {
                size: enum_layout.size,
                align: enum_layout.align,
                fields: vec![FieldLayout {
                    name: "discriminant".to_string(),
                    offset: 0,
                    size: enum_layout.discr_size,
                    align: enum_layout.discr_align,
                }],
            })
        }
        _ => {
            // Other type kinds (Alias, Trait, etc.) — use pointer-sized.
            Some(LayoutDescriptor {
                size: target.ptr_size(),
                align: target.ptr_align(),
                fields: Vec::new(),
            })
        }
    }
}

/// Compute the full layout of a struct type.
///
/// `field_types` is a list of (field_name, TypeId) pairs.
/// `packed`, `align_override`, `c_layout`, and `transparent` control
/// layout attributes.
pub fn compute_struct_layout<'input>(
    ctx: &mut TypeContext<'input>,
    target: &Target,
    field_types: &[(String, TypeId)],
    packed: bool,
    align_override: Option<u64>,
    c_layout: bool,
    transparent: bool,
    symbols: Option<&SymbolTable<'input>>,
    diag: &mut DiagCtxt,
) -> Option<LayoutDescriptor> {
    if transparent && field_types.len() == 1 {
        // @transparent: layout is identical to the sole field.
        let (name, ty) = &field_types[0];
        let field_size = super::abi::abi_size(ctx, target, *ty, symbols, diag)?;
        let field_align = super::abi::abi_alignment(ctx, target, *ty, symbols, diag)?;
        return Some(LayoutDescriptor {
            size: field_size,
            align: field_align,
            fields: vec![FieldLayout {
                name: name.clone(),
                offset: 0,
                size: field_size,
                align: field_align,
            }],
        });
    } else if transparent {
        diag.warn(format!(
            "@transparent struct with {} fields — only single-field transparent \
             is supported; falling back to default layout",
            field_types.len(),
        ));
    }

    let mut offset: u64 = 0;
    let mut max_align: u64 = 1;
    let mut fields: Vec<FieldLayout> = Vec::with_capacity(field_types.len());

    if packed {
        // Packed layout: no padding between fields, all align = 1.
        for (name, ty) in field_types {
            let field_size = super::abi::abi_size(ctx, target, *ty, symbols, diag)?;
            fields.push(FieldLayout {
                name: name.clone(),
                offset,
                size: field_size,
                align: 1,
            });
            offset += field_size;
        }
        let struct_align = align_override.unwrap_or(1);
        let total_size = align_up(offset, struct_align);
        return Some(LayoutDescriptor {
            size: total_size,
            align: struct_align,
            fields,
        });
    }

    // Common field layout loop — identical for both C and default layouts.
    // Only the subsequent struct_align calculation differs.
    for (name, ty) in field_types {
        let field_size = super::abi::abi_size(ctx, target, *ty, symbols, diag)?;
        let field_align = super::abi::abi_alignment(ctx, target, *ty, symbols, diag)?;
        offset = align_up(offset, field_align);
        fields.push(FieldLayout {
            name: name.clone(),
            offset,
            size: field_size,
            align: field_align,
        });
        offset += field_size;
        max_align = max_align.max(field_align);
    }

    // Overall struct alignment.
    let struct_align = align_override.unwrap_or_else(|| {
        if c_layout {
            max_align
        } else {
            max_align.min(target.max_align())
        }
    });

    // Total size is rounded up to the struct alignment.
    let total_size = align_up(offset, struct_align);

    Some(LayoutDescriptor {
        size: total_size,
        align: struct_align,
        fields,
    })
}

/// Compute the full layout of an enum type.
///
/// Current layout: `discriminant | (alignment padding) | payloads`
///
/// ── Future work: Niche optimization ─────────────────────────
/// Some types have "niche" bit patterns that are semantically
/// invalid (e.g. null for a pointer, or all-ones for Int<1>).
/// When exactly one variant has a payload with a known niche,
/// the discriminant can be encoded INTO that niche space,
/// eliminating the tag field entirely:
///
///   Option<Ptr<T>>  discr(4)+pad(4)+payload(8)=16
///                →  payload(8)  null=None, addr=Some
///
/// This is analogous to rustc's `HasOptimableNiche` / `Niche`.
/// If you're reading this and want a bite-sized feature, this
/// is it — see also `TypeId(NonZeroUsize)` in types.rs for an
/// encoding-level niche trick.
/// ─────────────────────────────────────────────────────────────
pub fn compute_enum_layout<'input>(
    ctx: &mut TypeContext<'input>,
    target: &Target,
    discr_size: u64,
    discr_align: u64,
    variant_payloads: &[(String, Vec<(String, TypeId)>)],
    packed: bool,
    align_override: Option<u64>,
    symbols: Option<&SymbolTable<'input>>,
    diag: &mut DiagCtxt,
) -> Option<EnumLayout> {
    // Compute layout of each variant's payload.
    let mut variants: Vec<VariantLayout> = Vec::with_capacity(variant_payloads.len());
    let mut max_payload_size: u64 = 0;
    let mut max_payload_align: u64 = 1;

    for (name, payload_fields) in variant_payloads {
        if payload_fields.is_empty() {
            // No payload — just the discriminant.
            variants.push(VariantLayout {
                name: name.to_string(),
                offset: 0,
                size: 0,
                fields: Vec::new(),
            });
            continue;
        }
        // Compute layout of the payload as if it were a struct.
        let payload_layout = compute_struct_layout(
            ctx,
            target,
            payload_fields,
            packed,
            None,
            false,
            false,
            symbols,
            diag,
        )?;
        let payload_align = if packed { 1 } else { payload_layout.align };
        max_payload_size = max_payload_size.max(payload_layout.size);
        max_payload_align = max_payload_align.max(payload_align);
        variants.push(VariantLayout {
            name: name.to_string(),
            offset: 0, // Will be calculated below
            size: payload_layout.size,
            fields: payload_layout.fields,
        });
    }

    // Enum layout: discriminant first, then the largest variant payload.
    // For simplicity, we use a tagged union layout:
    //   [ discriminant ] [ padding ] [ max_payload ]
    // The payload starts at an offset aligned to max_payload_align.
    let payload_offset = if packed {
        discr_size
    } else {
        align_up(discr_size, max_payload_align)
    };

    // Update variant offsets.
    for v in &mut variants {
        v.offset = payload_offset;
    }

    let total_size = if packed {
        discr_size + max_payload_size
    } else {
        let total = payload_offset + max_payload_size;
        let enum_align = align_override
            .unwrap_or_else(|| discr_align.max(max_payload_align.min(target.max_align())));
        align_up(total, enum_align)
    };

    let enum_align = if packed {
        1
    } else {
        align_override.unwrap_or_else(|| discr_align.max(max_payload_align.min(target.max_align())))
    };

    Some(EnumLayout {
        size: total_size,
        align: enum_align,
        discr_size,
        discr_align,
        variants,
    })
}

fn align_up(offset: u64, alignment: u64) -> u64 {
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

/// Resolve an AST `Type` (enum variant payload) to a list of (name, TypeId) fields.
/// Handles tuples, named types, and basic primitives.
fn resolve_ast_type_payload<'input>(
    ty: &Type<'input>,
    ctx: &mut TypeContext<'input>,
    symbols: Option<&SymbolTable<'input>>,
) -> Vec<(String, TypeId)> {
    match ty {
        Type::Tuple(tys, _) => tys
            .iter()
            .enumerate()
            .map(|(i, elem_ty)| {
                let ty_id = resolve_single_ast_type(elem_ty, ctx, symbols);
                (format!("_{}", i), ty_id)
            })
            .collect(),
        _ => {
            vec![(
                "value".to_string(),
                resolve_single_ast_type(ty, ctx, symbols),
            )]
        }
    }
}

/// Resolve a single AST `Type` to a `TypeId`.
fn resolve_single_ast_type<'input>(
    ty: &Type<'input>,
    ctx: &mut TypeContext<'input>,
    symbols: Option<&SymbolTable<'input>>,
) -> TypeId {
    match ty {
        Type::Path(path, _) => {
            if let Some(symbols) = symbols
                && let Some(def_id) = symbols.lookup_type_by_path(path)
                && let Some(ty_id) = ctx.get_type_id_for_def_id(def_id)
            {
                return ty_id;
            }
            // Fallback: try to match known names.
            if path.len() == 1 {
                let name = path[0].as_str();
                match name.as_str() {
                    "Int" | "Int<32>" => return ctx.int(32, true),
                    "UInt" | "UInt<32>" => return ctx.uint(32),
                    "Int<8>" => return ctx.int(8, true),
                    "UInt<8>" => return ctx.uint(8),
                    "Int<16>" => return ctx.int(16, true),
                    "UInt<16>" => return ctx.uint(16),
                    "Int<64>" => return ctx.int(64, true),
                    "UInt<64>" => return ctx.uint(64),
                    "Float<32>" => return ctx.float(32),
                    "Float<64>" => return ctx.float(64),
                    "Bool" => return ctx.bool(),
                    "String" => return ctx.builtin_str,
                    _ => {}
                }
            }
            ctx.int(32, true) // default fallback
        }
        Type::Generic(base, args, _) => {
            // Simplified: resolve the base type, ignore args for now.
            resolve_single_ast_type(base, ctx, symbols)
        }
        Type::Reference { inner, .. } => {
            let inner_id = resolve_single_ast_type(inner, ctx, symbols);
            ctx.reference(inner_id, false)
        }
        _ => ctx.int(32, true), // default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::target::Target;

    #[test]
    fn test_align_up_edge() {
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
    }

    #[test]
    fn test_compute_struct_layout_basic() {
        // Simulate a struct with two Int<32> fields.
        // Each is 4 bytes, aligned to 4.
        // Total = 8, align = 4.
        let target = Target::host();
        let mut ctx = TypeContext::new_with_target(target.clone());
        let i32 = ctx.int(32, true);
        let fields = vec![("x".to_string(), i32), ("y".to_string(), i32)];
        let mut diag = DiagCtxt::new();
        let layout = compute_struct_layout(
            &mut ctx, &target, &fields, false, None, false, false, None, &mut diag,
        )
        .expect("layout should succeed");
        assert_eq!(layout.size, 8);
        assert_eq!(layout.align, 4);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.fields[1].offset, 4);
        assert_eq!(layout.fields[1].size, 4);
    }

    #[test]
    fn test_compute_struct_layout_packed() {
        // Packed struct: no padding.
        let target = Target::host();
        let mut ctx = TypeContext::new_with_target(target.clone());
        let i32 = ctx.int(32, true);
        let i8 = ctx.int(8, true);
        let fields = vec![("a".to_string(), i32), ("b".to_string(), i8)];
        let mut diag = DiagCtxt::new();
        // Normal: a at 0, b at 4 (aligned to 4), total = 8.
        let normal = compute_struct_layout(
            &mut ctx, &target, &fields, false, None, false, false, None, &mut diag,
        )
        .expect("normal layout");
        assert_eq!(normal.fields[0].offset, 0);
        assert_eq!(normal.fields[1].offset, 4);
        assert_eq!(normal.size, 8);

        // Packed: a at 0, b at 4, total = 5.
        let packed = compute_struct_layout(
            &mut ctx, &target, &fields, true, None, false, false, None, &mut diag,
        )
        .expect("packed layout");
        assert_eq!(packed.fields[0].offset, 0);
        assert_eq!(packed.fields[1].offset, 4);
        assert_eq!(packed.size, 5);
        assert_eq!(packed.align, 1);
    }

    #[test]
    fn test_compute_struct_layout_align_override() {
        // @align(16) overrides struct alignment.
        let target = Target::host();
        let mut ctx = TypeContext::new_with_target(target.clone());
        let i8 = ctx.int(8, true);
        let fields = vec![("x".to_string(), i8)];
        let mut diag = DiagCtxt::new();
        let layout = compute_struct_layout(
            &mut ctx,
            &target,
            &fields,
            false,
            Some(16),
            false,
            false,
            None,
            &mut diag,
        )
        .expect("layout with align override");
        assert_eq!(layout.align, 16);
        assert_eq!(layout.size, 16); // padded to 16
    }

    #[test]
    fn test_compute_enum_layout_basic() {
        // Enum with empty variants.
        let target = Target::host();
        let mut ctx = TypeContext::new_with_target(target.clone());
        let variants = vec![
            ("None".to_string(), vec![]),
            (
                "Some".to_string(),
                vec![("value".to_string(), ctx.int(32, true))],
            ),
        ];
        let mut diag = DiagCtxt::new();
        let layout = compute_enum_layout(
            &mut ctx, &target, 4, 4, &variants, false, None, None, &mut diag,
        )
        .expect("enum layout");
        assert_eq!(layout.discr_size, 4);
        // Payload: Int<32> = 4 bytes, aligned to 4.
        // Total = 4 (discr) + 4 (payload) = 8, align = 4.
        assert_eq!(layout.size, 8);
        assert_eq!(layout.align, 4);
        assert_eq!(layout.variants.len(), 2);
        assert_eq!(layout.variants[0].size, 0); // None has no payload
        assert_eq!(layout.variants[1].size, 4); // Some has Int<32>
    }
}
