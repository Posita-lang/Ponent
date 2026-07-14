use crate::ast::{EnumVariant, Span, Type, TypeParam};
use crate::hir::symbol::{SymbolTable, TraitBinding, TypeBinding, TypeKind};
use crate::hir::traits::{ImplCandidate, TraitEnv};
use crate::hir::types::{DefId, TypeContext};
use crate::symbol::Symbol;

/// Insert a trait with no associated types into the symbol table.
fn insert_trait(symbols: &mut SymbolTable, name: &str, def_id: &mut DefId) {
    *def_id = symbols.allocate_def_id();
    let binding = TraitBinding {
        def_id: *def_id,
        methods: vec![],
        associated_types: vec![],
        super_traits: vec![],
        span: Span::new(0, 0),
        crate_id: symbols.local_crate_id,
    };
    symbols
        .insert_trait(Symbol::intern(name), binding, Span::new(0, 0))
        .ok();
}

/// Insert a trait with associated types into the symbol table.
fn insert_trait_with_assoc_types(
    symbols: &mut SymbolTable,
    name: &str,
    def_id: &mut DefId,
    associated_types: Vec<(Symbol, Option<Type>)>,
) {
    *def_id = symbols.allocate_def_id();
    let binding = TraitBinding {
        def_id: *def_id,
        methods: vec![],
        associated_types,
        super_traits: vec![],
        span: Span::new(0, 0),
        crate_id: symbols.local_crate_id,
    };
    symbols
        .insert_trait(Symbol::intern(name), binding, Span::new(0, 0))
        .ok();
}

pub fn register_builtins(
    symbols: &mut SymbolTable,
    trait_env: &mut TraitEnv,
    ctx: &mut TypeContext,
) {
    let mut add_id = DefId(0);
    let mut sub_id = DefId(0);
    let mut mul_id = DefId(0);
    let mut div_id = DefId(0);
    let mut rem_id = DefId(0);
    let mut bitand_id = DefId(0);
    let mut bitor_id = DefId(0);
    let mut bitxor_id = DefId(0);
    let mut shl_id = DefId(0);
    let mut shr_id = DefId(0);
    let mut eq_id = DefId(0);
    let mut neq_id = DefId(0);
    let mut lt_id = DefId(0);
    let mut gt_id = DefId(0);
    let mut le_id = DefId(0);
    let mut ge_id = DefId(0);
    let mut and_id = DefId(0);
    let mut or_id = DefId(0);
    let mut ord_id = DefId(0);
    let mut neg_id = DefId(0);
    let mut not_id = DefId(0);

    insert_trait(symbols, "Add", &mut add_id);
    insert_trait(symbols, "Sub", &mut sub_id);
    insert_trait(symbols, "Mul", &mut mul_id);
    insert_trait(symbols, "Div", &mut div_id);
    insert_trait(symbols, "Rem", &mut rem_id);
    insert_trait(symbols, "BitAnd", &mut bitand_id);
    insert_trait(symbols, "BitOr", &mut bitor_id);
    insert_trait(symbols, "BitXor", &mut bitxor_id);
    insert_trait(symbols, "Shl", &mut shl_id);
    insert_trait(symbols, "Shr", &mut shr_id);
    insert_trait(symbols, "Eq", &mut eq_id);
    insert_trait(symbols, "Neq", &mut neq_id);
    insert_trait(symbols, "Lt", &mut lt_id);
    insert_trait(symbols, "Gt", &mut gt_id);
    insert_trait(symbols, "Le", &mut le_id);
    insert_trait(symbols, "Ge", &mut ge_id);
    insert_trait(symbols, "And", &mut and_id);
    insert_trait(symbols, "Or", &mut or_id);
    insert_trait(symbols, "Ord", &mut ord_id);
    insert_trait(symbols, "Neg", &mut neg_id);
    insert_trait(symbols, "Not", &mut not_id);

    let int_arith_traits = [
        add_id, sub_id, mul_id, div_id, rem_id, bitand_id, bitor_id, bitxor_id, shl_id, shr_id,
        eq_id, ord_id, not_id,
    ];
    // Register arithmetic/bitwise trait impls for all common integer types.
    // Signed: Int<8>, Int<16>, Int<32>, Int<64>
    // Unsigned: UInt<8>, UInt<16>, UInt<32>, UInt<64>
    for &(bits, signed) in &[
        (8, true), (16, true), (32, true), (64, true),
        (8, false), (16, false), (32, false), (64, false),
    ] {
        let int_ty = ctx.int(bits, signed);
        for &trait_id in &int_arith_traits {
            trait_env
                .add_impl(
                    ImplCandidate {
                        trait_id,
                        for_type: int_ty,
                        methods: vec![],
                        resolved_methods: vec![],
                        assoc_tys: vec![],
                        has_auto_deref: false,
                        context: vec![],
                        span: Span::new(0, 0),
                    },
                    symbols,
                    ctx,
                    false,
                )
                .ok();
        }
        // Neg (unary `-`) only applies to signed integers.
        if signed {
            trait_env
                .add_impl(
                    ImplCandidate {
                        trait_id: neg_id,
                        for_type: int_ty,
                        methods: vec![],
                        resolved_methods: vec![],
                        assoc_tys: vec![],
                        has_auto_deref: false,
                        context: vec![],
                        span: Span::new(0, 0),
                    },
                    symbols,
                    ctx,
                    false,
                )
                .ok();
        }
    }

    // Register Not impl for Bool.  And/Or are handled directly by the
    // type checker and do not route through traits.
    let bool_ty = ctx.bool();
    for &trait_id in &[not_id] {
        trait_env
            .add_impl(
                ImplCandidate {
                    trait_id,
                    for_type: bool_ty,
                    methods: vec![],
                    resolved_methods: vec![],
                    assoc_tys: vec![],
                    has_auto_deref: false,
                    context: vec![],
                    span: Span::new(0, 0),
                },
                symbols,
                ctx,
                false,
            )
            .ok();
    }

    let float64 = ctx.float(64);
    for &trait_id in &[
        add_id, sub_id, mul_id, div_id, rem_id, eq_id, ord_id, neg_id,
    ] {
        trait_env
            .add_impl(
                ImplCandidate {
                    trait_id,
                    for_type: float64,
                    methods: vec![],
                    resolved_methods: vec![],
                    assoc_tys: vec![],
                    has_auto_deref: false,
                    context: vec![],
                    span: Span::new(0, 0),
                },
                symbols,
                ctx,
                false,
            )
            .ok();
    }

    // Register built-in Rational<p,q> types with arithmetic trait impls.
    let rational_arith_traits = [
        add_id, sub_id, mul_id, div_id, rem_id, eq_id, ord_id, neg_id,
    ];
    for &(p, q) in &[(8, 8), (16, 16), (32, 16), (32, 32)] {
        let rty = ctx.rational(p, q);
        for &trait_id in &rational_arith_traits {
            trait_env
                .add_impl(
                    ImplCandidate {
                        trait_id,
                        for_type: rty,
                        methods: vec![],
                        resolved_methods: vec![],
                        assoc_tys: vec![],
                        has_auto_deref: false,
                        context: vec![],
                        span: Span::new(0, 0),
                    },
                    symbols,
                    ctx,
                    false,
                )
                .ok();
        }
    }

    // ── Standard library types: Result, Option ────────────────────────

    // Result<T, E>
    {
        let def_id = symbols.allocate_def_id();
        let result_t = TypeParam {
            name: Symbol::intern("T"),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let result_e = TypeParam {
            name: Symbol::intern("E"),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let ok_variant = EnumVariant {
            name: Symbol::intern("Ok"),
            payload: Some(Type::Path(vec![Symbol::intern("T")], Span::new(0, 0))),
            span: Span::new(0, 0),
        };
        let err_variant = EnumVariant {
            name: Symbol::intern("Err"),
            payload: Some(Type::Path(vec![Symbol::intern("E")], Span::new(0, 0))),
            span: Span::new(0, 0),
        };
        let binding = TypeBinding {
            def_id,
            params: vec![result_t, result_e],
            kind: TypeKind::Enum,
            span: Span::new(0, 0),
            alias_ast: None,
            fields: vec![],
            variants: vec![ok_variant, err_variant],
            invariant: None,
            default_value: None,
            no_default: true,
            crate_id: symbols.local_crate_id,
            missing_match: None,
            exhaustive: false,
            c_layout: false,
                        transparent: false,
                        expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
        };
        symbols
            .insert_type(Symbol::intern("Result"), binding, Span::new(0, 0))
            .ok();
    }

    // Option<T>
    {
        let def_id = symbols.allocate_def_id();
        let option_t = TypeParam {
            name: Symbol::intern("T"),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let none_variant = EnumVariant {
            name: Symbol::intern("None"),
            payload: None,
            span: Span::new(0, 0),
        };
        let some_variant = EnumVariant {
            name: Symbol::intern("Some"),
            payload: Some(Type::Path(vec![Symbol::intern("T")], Span::new(0, 0))),
            span: Span::new(0, 0),
        };
        let binding = TypeBinding {
            def_id,
            params: vec![option_t],
            kind: TypeKind::Enum,
            span: Span::new(0, 0),
            alias_ast: None,
            fields: vec![],
            variants: vec![none_variant, some_variant],
            invariant: None,
            default_value: None,
            no_default: true,
            crate_id: symbols.local_crate_id,
            missing_match: None,
            exhaustive: false,
            c_layout: false,
                        transparent: false,
                        expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
        };
        symbols
            .insert_type(Symbol::intern("Option"), binding, Span::new(0, 0))
            .ok();
    }

    // ── Future trait ──────────────────────────────────────────────────
    //
    // Future is a trait with an associated type `Output`.
    //   trait Future {
    //       type Output;
    //   }
    //
    // async functions return `Future<T>` (a built-in enum type that serves
    // as the concrete return type).  The trait is used by `await` to extract
    // the Output type via projection: given `Future<T>`, `Future::Output = T`.
    //
    // The `Future<T>` enum with a single `Output(T)` variant stores the
    // eventual value.  This mirrors Rust's design where `Future` is a trait
    // and async fn returns `impl Future<Output = T>`.

    // 1) Register the Future enum type (concrete return type of async fns)
    let future_enum_def_id = symbols.allocate_def_id();
    {
        let future_t = TypeParam {
            name: Symbol::intern("T"),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let output_variant = EnumVariant {
            name: Symbol::intern("Output"),
            payload: Some(Type::Path(vec![Symbol::intern("T")], Span::new(0, 0))),
            span: Span::new(0, 0),
        };
        let binding = TypeBinding {
            def_id: future_enum_def_id,
            params: vec![future_t],
            kind: TypeKind::Enum,
            span: Span::new(0, 0),
            alias_ast: None,
            fields: vec![],
            variants: vec![output_variant],
            invariant: None,
            default_value: None,
            no_default: true,
            crate_id: symbols.local_crate_id,
            missing_match: None,
            exhaustive: false,
            c_layout: false,
                        transparent: false,
                        expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
        };
        symbols
            .insert_type(Symbol::intern("Future"), binding, Span::new(0, 0))
            .ok();
    }

    // 2) Register the Future trait with associated type `Output`
    let mut future_trait_id = DefId(0);
    insert_trait_with_assoc_types(
        symbols,
        "Future",
        &mut future_trait_id,
        vec![(Symbol::intern("Output"), None)],
    );

    // 3) Register `impl Future for Future<T>` where `Output = T`.
    //    This is the key bridge: for any concrete `Future<T>` type,
    //    the `Output` associated type resolves to `T`.
    //
    //    We register a generic impl using a generic param index 0.
    let future_output_ty = ctx.generic_param(0, Symbol::intern("T"));
    let future_for_ty = ctx.enum_ty(future_enum_def_id, vec![future_output_ty]);
    trait_env
        .add_impl(
            ImplCandidate {
                trait_id: future_trait_id,
                for_type: future_for_ty,
                methods: vec![],
                resolved_methods: vec![],
                assoc_tys: vec![(Symbol::intern("Output"), future_output_ty)],
                has_auto_deref: false,
                context: vec![],
                span: Span::new(0, 0),
            },
            symbols,
            ctx,
            true, // trusted — built-in impl
        )
        .ok();

    // Register standard traits for error suggestions and future use
    insert_trait(symbols, "From", &mut DefId(0));
    insert_trait(symbols, "Into", &mut DefId(0));
    insert_trait(symbols, "Sized", &mut DefId(0));
    insert_trait(symbols, "Deref", &mut DefId(0));
    insert_trait(symbols, "DerefMut", &mut DefId(0));

    // ── Channel<T> type ──────────────────────────────────────────
    // `Channel<T>` is a built-in type constructor for typed channels.
    // Syntax: `Channel<Int<32>>::new(16)` → (Sender, Receiver)
    // Currently registered as a minimal placeholder (no variants, no methods).
    // A full implementation will add Send(T)/Recv(T) payload variants
    // and the associated new() method.
    {
        let channel_t = TypeParam {
            name: Symbol::intern("T"),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let binding = TypeBinding {
            def_id: symbols.allocate_def_id(),
            params: vec![channel_t],
            kind: TypeKind::Enum,
            span: Span::new(0, 0),
            alias_ast: None,
            fields: vec![],
            variants: vec![],
            invariant: None,
            default_value: None,
            no_default: true,
            crate_id: symbols.local_crate_id,
            missing_match: None,
            exhaustive: false,
            c_layout: false,
            transparent: false,
            expanded_layout_attrs: vec![],
            packed: false,
            endian: None,
            bit_order: None,
            align: None,
            pad: None,
        };
        symbols
            .insert_type(Symbol::intern("Channel"), binding, Span::new(0, 0))
            .ok();
    }
}
