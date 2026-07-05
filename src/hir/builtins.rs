use crate::ast::{EnumVariant, Span, Type, TypeParam};
use crate::hir::symbol::{SymbolTable, TraitBinding, TypeBinding, TypeKind};
use crate::hir::traits::{ImplCandidate, TraitEnv};
use crate::hir::types::{DefId, TypeContext};

/// Insert a trait with no associated types into the symbol table.
fn insert_trait(symbols: &mut SymbolTable, name: &str, def_id: &mut DefId) {
    *def_id = symbols.allocate_def_id();
    let binding = TraitBinding {
        def_id: *def_id,
        methods: vec![],
        associated_types: vec![],
        span: Span::new(0, 0),
        crate_id: symbols.local_crate_id,
    };
    symbols
        .insert_trait(name.to_string(), binding, Span::new(0, 0))
        .ok();
}

/// Insert a trait with associated types into the symbol table.
fn insert_trait_with_assoc_types(
    symbols: &mut SymbolTable,
    name: &str,
    def_id: &mut DefId,
    associated_types: Vec<(String, Option<Type>)>,
) {
    *def_id = symbols.allocate_def_id();
    let binding = TraitBinding {
        def_id: *def_id,
        methods: vec![],
        associated_types,
        span: Span::new(0, 0),
        crate_id: symbols.local_crate_id,
    };
    symbols
        .insert_trait(name.to_string(), binding, Span::new(0, 0))
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

    let int32 = ctx.int(32, true);
    for &trait_id in &[
        add_id, sub_id, mul_id, div_id, rem_id, bitand_id, bitor_id, bitxor_id, shl_id, shr_id,
        eq_id, neq_id, lt_id, gt_id, le_id, ge_id, and_id, or_id,
    ] {
        trait_env
            .add_impl(
                ImplCandidate {
                    trait_id,
                    for_type: int32,
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
        add_id, sub_id, mul_id, div_id, rem_id, eq_id, neq_id, lt_id, gt_id, le_id, ge_id,
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
        add_id, sub_id, mul_id, div_id, rem_id, eq_id, neq_id, lt_id, gt_id, le_id, ge_id,
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
            name: "T".to_string(),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let result_e = TypeParam {
            name: "E".to_string(),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let ok_variant = EnumVariant {
            name: "Ok".to_string(),
            payload: Some(Type::Path(vec!["T".to_string()], Span::new(0, 0))),
            span: Span::new(0, 0),
        };
        let err_variant = EnumVariant {
            name: "Err".to_string(),
            payload: Some(Type::Path(vec!["E".to_string()], Span::new(0, 0))),
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
        };
        symbols
            .insert_type("Result".to_string(), binding, Span::new(0, 0))
            .ok();
    }

    // Option<T>
    {
        let def_id = symbols.allocate_def_id();
        let option_t = TypeParam {
            name: "T".to_string(),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let none_variant = EnumVariant {
            name: "None".to_string(),
            payload: None,
            span: Span::new(0, 0),
        };
        let some_variant = EnumVariant {
            name: "Some".to_string(),
            payload: Some(Type::Path(vec!["T".to_string()], Span::new(0, 0))),
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
        };
        symbols
            .insert_type("Option".to_string(), binding, Span::new(0, 0))
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
            name: "T".to_string(),
            bounds: vec![],
            is_lifetime: false,
            span: Span::new(0, 0),
        };
        let output_variant = EnumVariant {
            name: "Output".to_string(),
            payload: Some(Type::Path(vec!["T".to_string()], Span::new(0, 0))),
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
        };
        symbols
            .insert_type("Future".to_string(), binding, Span::new(0, 0))
            .ok();
    }

    // 2) Register the Future trait with associated type `Output`
    let mut future_trait_id = DefId(0);
    insert_trait_with_assoc_types(
        symbols,
        "Future",
        &mut future_trait_id,
        vec![("Output".to_string(), None)],
    );

    // 3) Register `impl Future for Future<T>` where `Output = T`.
    //    This is the key bridge: for any concrete `Future<T>` type,
    //    the `Output` associated type resolves to `T`.
    //
    //    We register a generic impl using a generic param index 0.
    let future_output_ty = ctx.generic_param(0, "T".to_string());
    let future_for_ty = ctx.enum_ty(future_enum_def_id, vec![future_output_ty]);
    trait_env
        .add_impl(
            ImplCandidate {
                trait_id: future_trait_id,
                for_type: future_for_ty,
                methods: vec![],
                resolved_methods: vec![],
                assoc_tys: vec![("Output".to_string(), future_output_ty)],
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
}
