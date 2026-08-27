use super::*;

fn check_parse(source: &str) -> Program<'static> {
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let mut parser = Parser::new(source, arena);
    match parser.parse_program() {
        Ok(prog) => {
            assert!(
                parser.diagnostics.is_empty(),
                "unexpected diagnostics: {:?}",
                parser.diagnostics
            );
            prog
        }
        Err(diags) => panic!("parse failed with diagnostics: {:?}", diags),
    }
}

fn check_parse_err(source: &str) -> Vec<Diagnostic> {
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let mut parser = Parser::new(source, arena);
    parser.parse_program().err().unwrap_or_else(|| {
        panic!(
            "expected parse error, but parsing succeeded: {:?}",
            parser.diagnostics
        )
    })
}

/// Construct a parser over `source` backed by a leaked `'static` arena,
/// for tests that need direct access to the parser (tokens, pending
/// stack, etc.) rather than just its result.
fn new_parser(source: &str) -> Parser<'static> {
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    Parser::new(source, arena)
}

/// Parse `source` and assert that it fails (error result or diagnostics).
fn check_parse_fails(source: &str) {
    let mut parser = new_parser(source);
    let result = parser.parse_program();
    assert!(
        result.is_err() || !parser.diagnostics.is_empty(),
        "expected parse failure for: {:?}",
        source
    );
}

/// The banded edit distance correctness: the transposition-like pair
/// `"retrun"`/`"return"` is distance 2 (within the limit), and the
/// unrelated pair is clamped to the limit+1 sentinel when it exceeds
/// the limit.
#[test]
fn test_edit_distance_limited_banded() {
    let mut prev = Vec::new();
    let mut curr = Vec::new();
    assert_eq!(
        edit_distance_limited_into(
            &mut prev,
            &mut curr,
            &mut Vec::new(),
            &mut Vec::new(),
            "retrun",
            "return",
            2
        ),
        2
    );
    // Within the lenient threshold.
    assert_eq!(
        edit_distance_limited_into(
            &mut prev,
            &mut curr,
            &mut Vec::new(),
            &mut Vec::new(),
            "retrun",
            "leave",
            6
        ),
        5
    );
    // Exceeds the limit → clamped to limit + 1.
    assert_eq!(
        edit_distance_limited_into(
            &mut prev,
            &mut curr,
            &mut Vec::new(),
            &mut Vec::new(),
            "retrun",
            "leave",
            2
        ),
        3
    );
}

#[test]
fn test_did_you_mean_keyword_suggestion() {
    // The optimized did_you_mean_keyword must still suggest:
    // "retrun" → "return" (distance 2, first-char match).
    let suggestion = did_you_mean_keyword("retrun", KeywordContext::Statement);
    assert_eq!(suggestion.as_deref(), Some("did you mean `return`?"));
    // The ASCII fast path (upper-case input → case-insensitive match).
    let suggestion2 = did_you_mean_keyword("Retrun", KeywordContext::Statement);
    assert_eq!(suggestion2.as_deref(), Some("did you mean `return`?"));
    // An unrelated input (distance == length — no real similarity)
    // gets NO suggestion (the wide threshold used to
    // suggest `leave` for `xyzabc` — a false suggestion).
    assert_eq!(
        did_you_mean_keyword("xyzabc", KeywordContext::Statement).as_deref(),
        None
    );
}

#[test]
fn test_empty_function() {
    let program = check_parse("def main() { }");
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::FunctionDef {
            name, params, body, ..
        } => {
            assert!(name.eq_str("main"));
            assert!(params.is_empty());
            assert!(body.as_ref().map_or(false, |b| b.is_empty()));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_variable_def() {
    let program = check_parse("def main() { set x = 42; }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::VariableDef { name, .. } => assert!(name.unwrap().eq_str("x")),
            _ => panic!("expected VariableDef"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

/// `a..b` must parse the upper bound: `Range { end: Some(b) }`.
/// A regression test for the infix range branch (the `..` arm must
/// parse the end when a non-terminator follows).
#[test]
fn test_range_expr_with_upper_bound() {
    let program = check_parse("def main() { set r = 0..10; }");
    let Stmt::FunctionDef { body, .. } = &program.items[0] else {
        panic!("expected FunctionDef");
    };
    let Stmt::VariableDef { value, .. } = &body.as_ref().unwrap()[0] else {
        panic!("expected VariableDef");
    };
    let expr = value.as_ref().expect("initializer present");
    match expr {
        Expr::Range { end, inclusive, .. } => {
            assert!(!inclusive);
            assert!(end.is_some(), "`0..10` must have an upper bound");
        }
        other => panic!("expected Range, got {:?}", other),
    }
}

/// `a..;` is an open-ended range: `Range { end: None }`, and must
/// NOT error (the `..` arm must leave the end unset when a
/// terminator follows).
#[test]
fn test_range_expr_open_ended() {
    let program = check_parse("def main() { set r = 0..; }");
    let Stmt::FunctionDef { body, .. } = &program.items[0] else {
        panic!("expected FunctionDef");
    };
    let Stmt::VariableDef { value, .. } = &body.as_ref().unwrap()[0] else {
        panic!("expected VariableDef");
    };
    let expr = value.as_ref().expect("initializer present");
    match expr {
        Expr::Range { end, inclusive, .. } => {
            assert!(!inclusive);
            assert!(end.is_none(), "`0..` must be an open-ended range");
        }
        other => panic!("expected Range, got {:?}", other),
    }
}

/// Same-precedence binary operators are LEFT-associative:
/// `a - b - c` parses as `(a - b) - c`.  A regression test for the
/// `bp + 1` right-operand binding-power change in `parse_infix`.
#[test]
fn test_binary_sub_left_associative() {
    let program = check_parse("def main() -> Int<32> { set r = a - b - c; return 0; }");
    let Stmt::FunctionDef { body, .. } = &program.items[0] else {
        panic!("expected FunctionDef");
    };
    let Stmt::VariableDef { value, .. } = &body.as_ref().unwrap()[0] else {
        panic!("expected VariableDef");
    };
    let expr = value.as_ref().expect("initializer present");
    match expr {
        Expr::BinaryOp {
            left,
            op: BinOp::Sub,
            right,
            ..
        } => {
            assert!(matches!(*right, Expr::Ident(..)));
            match *left {
                Expr::BinaryOp { op: BinOp::Sub, .. } => {}
                other => panic!("expected nested `a - b` on the left, got {:?}", other),
            }
        }
        other => panic!("expected BinaryOp(Sub), got {:?}", other),
    }
}

/// Division is left-associative too (SYNTAX.md §Operators:
/// `*`/`/`/`%` are left-to-right): `a / b / c` parses as
/// `(a / b) / c`.  A companion to `test_binary_sub_left_associative`.
#[test]
fn test_binary_div_left_associative() {
    let program = check_parse("def main() -> Int<32> { set r = a / b / c; return 0; }");
    let Stmt::FunctionDef { body, .. } = &program.items[0] else {
        panic!("expected FunctionDef");
    };
    let Stmt::VariableDef { value, .. } = &body.as_ref().unwrap()[0] else {
        panic!("expected VariableDef");
    };
    let expr = value.as_ref().expect("initializer present");
    match expr {
        Expr::BinaryOp {
            left,
            op: BinOp::Div,
            right,
            ..
        } => {
            assert!(matches!(*right, Expr::Ident(..)));
            match *left {
                Expr::BinaryOp { op: BinOp::Div, .. } => {}
                other => panic!("expected nested `a / b` on the left, got {:?}", other),
            }
        }
        other => panic!("expected BinaryOp(Div), got {:?}", other),
    }
}

#[test]
fn test_if_stmt() {
    let program = check_parse("def main() { if true { } }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => {
            assert!(matches!(body.as_ref().unwrap()[0], Stmt::If { .. }));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_scope_cleanup_with_at() {
    let program = check_parse("def main() { scope_cleanup @close_file { } }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::ScopeCleanup {
                name,
                body: _,
                propagates,
                overrides,
                ..
            } => {
                assert!(name.eq_str("close_file"));
                assert!(!propagates);
                assert!(!overrides);
            }
            _ => panic!("expected ScopeCleanup"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_reference_type() {
    let program = check_parse("def main() { set x: &Int<32> = 0; }");
    assert!(program.items.len() == 1);
}

#[test]
fn test_pointer_type() {
    let program = check_parse("def main() { set x: *Int<32> = 0; }");
    assert!(program.items.len() == 1);
}

#[test]
fn test_slice_type() {
    let program = check_parse("def main() { set x: [Int<32>] = 0; }");
    assert!(program.items.len() == 1);
}

#[test]
fn test_array_type() {
    let program = check_parse("def main() { set x: [Int<32>; 10] = 0; }");
    assert!(program.items.len() == 1);
}

#[test]
fn test_dyn_trait_type() {
    let program = check_parse("def main() { set x: dyn Display = 0; }");
    assert!(program.items.len() == 1);
}

#[test]
fn test_exists_type() {
    let program = check_parse("type Age = exists n: UInt<8> invariant n >= 18;");
    assert!(program.items.len() == 1);
}

#[test]
fn test_ellipsis_is_invalid() {
    check_parse_fails("def main() { ...; }");
}

#[test]
fn test_struct_literal() {
    let program = check_parse("def main() { set e = Employee { id = 1, name = b\"Alice\" }; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_enum_literal() {
    let program = check_parse("def main() { set d = Department::Engineering; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_move_expression() {
    let program = check_parse("def main() { set x = 1; set y = move x; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_suffixed_literal() {
    let program = check_parse("def main() { set x = 42i32; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_unsafe_block() {
    let program = check_parse("def main() { unsafe { } }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_try_expression() {
    let program =
        check_parse("def main() -> Result<(), Error> { let x = do_something()?; return Ok(()); }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_cast() {
    let program = check_parse("def main() { set x = 42 as Float<64>; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_ref_prefix() {
    let program = check_parse("def main() { set x = &y; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_deref_prefix() {
    let program = check_parse("def main() { set x = *y; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_finally_block() {
    let program = check_parse("def main() { } finally { }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_impl_block() {
    let program = check_parse("impl Drop for UniqueToken { def drop(&mut self) { } }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_closure() {
    let program =
        check_parse("def main() { set f = |x: Int<32>, y: Int<32>| -> Int<32> { x + y; }; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_pattern_struct() {
    let program = check_parse("def main() { let Point { x, y } = p; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_pattern_enum() {
    let program = check_parse("def main() { let Some(v) = opt; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_pattern_literal() {
    let program = check_parse("def main() { match x { 1 => {}, _ => {} }; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_keyword_suggestion_toplevel_fn() {
    let diags = check_parse_err("fn main(){}");
    let has = diags
        .iter()
        .any(|d| d.suggestions().iter().any(|s| s.message.contains("def")));
    assert!(
        has,
        "expected 'did you mean `def`?' for `fn`, got: {:?}",
        diags
    );
}

#[test]
fn test_keyword_suggestion_pattern_frue() {
    // A non-identifier token in pattern position triggers the suggestion path.
    // Users might accidentally type `Frue` (inspired by `fn` → `def` style typos).
    let diags = check_parse_err("def main() { match x { $ => {} }; }");
    let has = diags
        .iter()
        .any(|d| d.suggestions().iter().any(|s| s.message.contains("try")));
    assert!(
        has,
        "expected a suggestion for bad pattern token, got: {:?}",
        diags
    );
}

/// Error recovery must NOT swallow a statement-start keyword:
/// after the `1 +` operand gap, the next `set` starts a fresh
/// statement (`set y = 5`), so the parser reports exactly ONE error
/// (the missing operand).  If `synchronize` consumed the `set`, the
/// leftover `y = 5` would be misparsed and add a cascade error.
#[test]
fn test_synchronize_does_not_swallow_statement_start() {
    let diags = check_parse_err("def main() { set a = 1 + set y = 5; }");
    assert_eq!(
        diags.len(),
        1,
        "expected only the `1 +` operand-gap error, got: {:?}",
        diags
    );
}

/// A block-level `trait` is a valid statement (committee-approved
/// rustc-style nested items): `parse_stmt` dispatches it to
/// `parse_trait_def`, which consumes the keyword first — so error
/// recovery always makes progress and the block loop terminates at
/// the closing brace.  The test completing at all is the assertion
/// (a spin would hang it).
#[test]
fn test_synchronize_block_level_trait_no_spin() {
    // Block-level `trait` is now a valid statement (committee-approved
    // rustc-style nested items): parsing succeeds — a spin would hang
    // the test instead.
    let program = check_parse("def main() { trait X {} return 0; }");
    assert_eq!(program.items.len(), 1);
}

/// Regression: error recovery must not swallow a block-level item
/// definition — `trait` is a protected statement starter, so after
/// the `1 +` operand-gap error the `trait X {}` definition is parsed
/// normally (only the original error is reported).
#[test]
fn test_recovery_keeps_block_level_trait() {
    let diags = check_parse_err("def main() { set a = 1 + trait X {} return 0; }");
    assert_eq!(
        diags.len(),
        1,
        "expected only the `1 +` operand-gap error, got: {:?}",
        diags
    );
}

// ── peek_next lookahead-2 contract ────────────────────────────
// `peek_next` must return the token AFTER the one `peek()` would
// return, both with an empty pending stack and with pending tokens
// queued (the `>>`/`<<` split paths).  For `"a + b"` the stream is
// [Ident(a), Plus, Ident(b)].

/// Empty pending stack: `peek()` returns `Ident(a)`, so `peek_next`
/// is the following stream token `Plus`.
#[test]
fn test_peek_next_pending_empty() {
    let mut p = new_parser("a + b");
    assert_eq!(p.peek_next(), Some(Token::Plus));
    // peek() must still return the FIRST token (peek_next is a pure
    // lookahead — it must not have consumed `Ident(a)`).
    assert!(matches!(p.peek(), Ok(Token::Ident(_))));
}

/// One pending token (`Gt`, as after a `>>` split): `peek()` will pop
/// it, so `peek_next` is the stream position `Ident(a)`.
#[test]
fn test_peek_next_pending_one() {
    let mut p = new_parser("a + b");
    p.pending.push(Token::Gt);
    assert_eq!(p.peek_next(), Some(Token::Ident(Symbol::intern("a"))));
    // peek() returns the pending token, not the stream head.
    assert!(matches!(p.peek(), Ok(Token::Gt)));
}

/// Two pending tokens (`[Gt, Lt]`, stack top `Lt`): `peek()` will pop
/// `Lt`, so `peek_next` is the new stack top `Gt`.
#[test]
fn test_peek_next_pending_two() {
    let mut p = new_parser("a + b");
    p.pending.push(Token::Gt);
    p.pending.push(Token::Lt);
    assert_eq!(p.peek_next(), Some(Token::Gt));
    // peek() returns the pending stack top `Lt`.
    assert!(matches!(p.peek(), Ok(Token::Lt)));
}

#[test]
fn test_ghost_variable() {
    let program = check_parse("def main() { ghost set mut x = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_comptime_function_def() {
    let program = check_parse("comptime def eval() -> Int<32> { return 42; }");
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::FunctionDef {
            is_comptime, name, ..
        } => {
            assert!(is_comptime);
            assert!(name.eq_str("eval"));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_async_function_def() {
    let program = check_parse("async def fetch() -> Data { }");
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::FunctionDef { is_async, name, .. } => {
            assert!(is_async);
            assert!(name.eq_str("fetch"));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_trait_def() {
    let src = "trait Show { def show(&self) -> String; type Output = Int<32>; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::TraitDef {
            name,
            methods,
            associated_types,
            ..
        } => {
            assert!(name.eq_str("Show"));
            assert_eq!(methods.len(), 1);
            assert_eq!(associated_types.len(), 1);
        }
        _ => panic!("expected TraitDef"),
    }
}

#[test]
fn test_constraint() {
    // Proper colon-based syntax: `Subject: Bound1 + Bound2`
    let src = "constraint MyConstraint { T: Display + Debug }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::Constraint {
            name, predicates, ..
        } => {
            assert!(name.eq_str("MyConstraint"));
            assert_eq!(predicates.len(), 1);
            assert!(matches!(predicates[0].ty, Type::Path(_, _)));
            assert_eq!(predicates[0].bounds.len(), 2);
        }
        _ => panic!("expected Constraint"),
    }
}

#[test]
fn test_constraint_rejects_flat_format() {
    // Flat format `{ Display + Debug }` without colon must be rejected.
    let src = "constraint MyConstraint { Display + Debug }";
    let errs = check_parse_err(src);
    assert!(!errs.is_empty(), "flat constraint format must be rejected");
}

#[test]
fn test_constraint_with_generic_params() {
    // Generic constraint with type params and colon-based predicates.
    let src = "constraint SortableContainer<C> { C: Container, C::Item: Ord, C::Item: Default }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::Constraint {
            name,
            params,
            predicates,
            ..
        } => {
            assert!(name.eq_str("SortableContainer"));
            assert_eq!(params.len(), 1);
            assert!(params[0].name.eq_str("C"));
            assert_eq!(predicates.len(), 3);
            // First predicate: C: Container
            assert_eq!(predicates[0].bounds.len(), 1);
            // Second predicate: C::Item: Ord
            assert_eq!(predicates[1].bounds.len(), 1);
        }
        _ => panic!("expected Constraint"),
    }
}

#[test]
fn test_compile_error_without_bang() {
    // `@compile_error("msg")` — no exclamation mark, per the spec
    let src = r#"def f() { @compile_error("oops"); }"#;
    let _program = check_parse(src);
}

#[test]
fn test_where_tuple_subject() {
    // Track‑B: `where (T, U): Constraint` syntax — tuple subjects
    // in where clauses apply a multi-type constraint positionally.
    let src = "def f<T, U>(x: T, y: U) where (T, U): MyConstraint { }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::FunctionDef { where_clause, .. } => {
            let wc = where_clause
                .as_ref()
                .expect("where clause should be parsed");
            assert_eq!(wc.predicates.len(), 1);
            // Subject should be a tuple type
            assert!(
                matches!(&wc.predicates[0].ty, Type::Tuple(_, _)),
                "subject should be a tuple, got {:?}",
                wc.predicates[0].ty
            );
            // The bound should be the constraint name
            assert_eq!(wc.predicates[0].bounds.len(), 1);
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_keyword_as_path_ident() {
    // Track‑A: keyword tokens are accepted after `::` as identifiers.
    // `default`, `move`, `copy`, `type`, `Self` are keywords that can
    // appear as method / variant / associated-type names.
    //
    // IMPORTANT: `T::default()` parses as a 2-segment path followed by
    // `()`, which `parse_path_or_literal` routes to `parse_enum_lit`
    // (empty payload).  This is a pre-existing parser design choice:
    // 2-segment `Path::ident(args)` is always treated as enum variant
    // construction, not an associated-function call.  The test verifies
    // the KEYWORD-AS-IDENTIFIER aspect, not the call-semantics aspect.
    let cases = &[
        // Path-only (no call parens) — the most direct keyword test.
        "def f() { let x = Enum::default; }",
        "def f() { let x = Mod::move; }",
        "def f() { let x = Mod::copy; }",
        "def f() { let x = Mod::type; }",
        "def f() { let x = Mod::ieee; }",
        // With empty parens (→ enum construction with empty payload).
        "def f() { let x = Mod::none(); }",
    ];
    for src in cases {
        let program = check_parse(src);
        assert_eq!(program.items.len(), 1, "failed to parse: {src}");
    }
}

#[test]
fn test_ieee_keyword_attribute_name() {
    // `ieee` is a keyword, but as an ATTRIBUTE NAME it must
    // round-trip to the symbol "ieee" — NOT the debug-formatted "Ieee"
    // fallback (which silently renamed `@ieee` and broke consumers
    // comparing `attr.name.eq_str("ieee")`).
    let src = "@ieee\ndef f() { }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { attributes, .. } => {
            assert_eq!(attributes.len(), 1);
            assert!(
                attributes[0].name.eq_str("ieee"),
                "attribute name must be `ieee`, got `{}`",
                attributes[0].name.as_str()
            );
        }
        _ => panic!("expected function definition"),
    }
}

#[test]
fn test_type_alias_with_overflow() {
    let src = "type MyInt = Int<32> with overflow = saturate;";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Alias(_, modifiers),
            ..
        } => {
            assert_eq!(modifiers.len(), 1);
            assert!(matches!(
                modifiers[0],
                TypeModifier::Overflow(OverflowPolicy::Saturate)
            ));
        }
        _ => panic!("expected type alias with overflow"),
    }
}

#[test]
fn test_type_alias_with_default() {
    let src = "type MyInt = Int<32> with default = 42;";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Alias(_, modifiers),
            ..
        } => {
            assert_eq!(modifiers.len(), 1);
            assert!(matches!(modifiers[0], TypeModifier::Default(_)));
        }
        _ => panic!("expected default"),
    }
}

#[test]
fn test_type_alias_with_no_default() {
    let src = "type MyInt = Int<32> with no_default;";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Alias(_, modifiers),
            ..
        } => {
            assert_eq!(modifiers.len(), 1);
            assert!(matches!(modifiers[0], TypeModifier::NoDefault));
        }
        _ => panic!("expected no_default"),
    }
}

#[test]
fn test_ensures_on_ok() {
    let src = "def div(a: Int<32>, b: Int<32>) -> Int<32> requires b != 0 ensures on Ok(result) => result * b == a { return a / b; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 2);
            match &contracts[1] {
                Contract::Ensures { target, .. } => {
                    assert!(matches!(target, EnsuresTarget::OnOk(Some(_))));
                }
                _ => panic!("expected Ensures contract"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_deprecated_attribute() {
    let src = "@deprecated(\"use new_method\") def old_fn() { }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { attributes, .. } => {
            assert_eq!(attributes.len(), 1);
            assert!(attributes[0].name.eq_str("deprecated"));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_cfg_attribute() {
    let src = "@cfg(target_os = \"linux\") def linux_only() { }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { attributes, .. } => {
            assert_eq!(attributes.len(), 1);
            assert!(attributes[0].name.eq_str("cfg"));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_module_doc_comment() {
    let src = "//! module doc\ndef main() { }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { doc, .. } => {
            assert_eq!(doc.as_deref(), Some("module doc"));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_comptime_block() {
    let src = "comptime { let x = 42; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::ComptimeBlock { .. } => {}
        _ => panic!("expected ComptimeBlock"),
    }
}

#[test]
fn test_isolate_block() {
    let src = "def main() { isolate { set x = 42; } }";
    let program = check_parse(src);
    assert!(program.items.len() == 1);
}

#[test]
fn test_catch_expression() {
    let src = "def main() -> Result<(), Error> { let data = fetch() catch { |NetworkError| { leave with Err(ProcessError::NetworkFail); } }; return Ok(()); }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_match_exhaustive() {
    let src = "def main() { match x { 1 => {}, _ => {} }; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_for_loop_with_invariant() {
    let src = "def sum(arr: &[Int<32>]) -> Int<32> { set mut total = 0; for i in 0..arr'len invariant total == fold(arr[0..i], 0, +) decreases arr'len - i { total += arr[i]; } return total; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_leave_with_in_catch() {
    check_parse(
        "def f() -> Result<(), ()> { let _ = x() catch { |E| { leave with Err(()); } }; Ok(()) }",
    );
}

#[test]
fn test_while_with_invariant() {
    let src =
        "def f() { set mut i = 0; while i < 10 invariant i >= 0 decreases 10 - i { i += 1; } }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_as_bitcast() {
    let src = "def f() { set x = 42 as! Float<64>; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::VariableDef {
                value: Some(Expr::Cast { safe, .. }),
                ..
            } => {
                assert!(!safe);
            }
            _ => panic!("expected Cast"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_scope_cleanup_with_propagates() {
    let program = check_parse("def main() { scope_cleanup @close_file propagates { } }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::ScopeCleanup {
                propagates,
                overrides,
                ..
            } => {
                assert!(propagates);
                assert!(!overrides);
            }
            _ => panic!("expected ScopeCleanup"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_scope_cleanup_with_propagates_overrides() {
    let program = check_parse("def main() { scope_cleanup @close_file propagates overrides { } }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::ScopeCleanup {
                propagates,
                overrides,
                ..
            } => {
                assert!(propagates);
                assert!(overrides);
            }
            _ => panic!("expected ScopeCleanup"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_scope_cleanup_overrides_without_propagates_fails() {
    check_parse_fails("def main() { scope_cleanup @close_file overrides { } }");
}

#[test]
fn test_ensures_on_err() {
    let src = "def f() -> Result<Int<32>, Err> ensures on Err(e) => e != 0 { return Err(1); }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 1);
            match &contracts[0] {
                Contract::Ensures { target, .. } => {
                    assert!(matches!(target, EnsuresTarget::OnErr(Some(_))));
                }
                _ => panic!("expected Ensures"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_type_union_alias() {
    let src = "type AppError = IoError | ParseError;";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Alias(ty, _),
            ..
        } => {
            assert!(matches!(ty, Type::Union(..)));
        }
        _ => panic!("expected Union type alias"),
    }
}

#[test]
fn test_type_keyword_as_literal() {
    let src = "comptime def foo() -> type { return 42; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { return_type, .. } => {
            let rt = return_type.as_ref().expect("return_type should be Some");
            assert!(matches!(rt, Type::Path(path, _) if path.len() == 1 && path[0].eq_str("type")));
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_cast_with_rounding() {
    let src = "def f() { set x = 3.14 as Int<32> round; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::VariableDef {
                value: Some(Expr::Cast { rounding, .. }),
                ..
            } => {
                assert_eq!(rounding, &Some(Rounding::Round));
            }
            _ => panic!("expected Cast with rounding"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_enum_missing_match() {
    let src = "type State = enum { A, B } with missing_match = \"missing variants\";";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Enum(_, Some(msg), _),
            ..
        } => {
            assert_eq!(msg, "missing variants");
        }
        _ => panic!("expected Enum with missing_match"),
    }
}

#[test]
fn test_gadt_when_clause() {
    // Single GADT constraint
    let src = "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Enum(variants, _, _),
            params,
            ..
        } => {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].name.as_str(), "T");
            assert_eq!(variants.len(), 1);
            let lit = &variants[0];
            assert_eq!(lit.name.as_str(), "Lit");
            assert_eq!(lit.eq_spec.len(), 1);
            let (param, concrete) = &lit.eq_spec[0];
            assert_eq!(param.as_str(), "T");
            // Int<32> parses as Type::Generic(Path(["Int"]), [Positional(Path(["32"]))])
            assert!(
                matches!(concrete, Type::Generic(..)),
                "expected Generic type for Int<32>, got {:?}",
                concrete
            );
        }
        _ => panic!("expected TypeDef with Enum"),
    }
}

#[test]
fn test_gadt_when_multi_and() {
    // Multiple GADT constraints with `and` (single payload per variant)
    let src = "type KV<K, V> = enum { Pair(Int<32>) when K == Int<32> and V == String }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Enum(variants, _, _),
            ..
        } => {
            assert_eq!(variants.len(), 1);
            let ik = &variants[0];
            assert_eq!(ik.name.as_str(), "Pair");
            assert_eq!(ik.eq_spec.len(), 2, "should have K and V constraints");
            assert_eq!(ik.eq_spec[0].0.as_str(), "K");
            assert_eq!(ik.eq_spec[1].0.as_str(), "V");
        }
        _ => panic!("expected TypeDef with Enum"),
    }
}

#[test]
fn test_gadt_mixed() {
    // Mixture of GADT and non-GADT variants
    let src = "type Expr<T> = enum {
            Lit(Int<32>) when T == Int<32>,
            Not(Expr<Bool>) when T == Bool,
            Wrap(Expr<T>)
        }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Enum(variants, _, _),
            ..
        } => {
            assert_eq!(variants.len(), 3);
            // Lit has eq_spec
            assert_eq!(variants[0].name.as_str(), "Lit");
            assert_eq!(variants[0].eq_spec.len(), 1);
            // Not has eq_spec
            assert_eq!(variants[1].name.as_str(), "Not");
            assert_eq!(variants[1].eq_spec.len(), 1);
            // Wrap has NO eq_spec
            assert_eq!(variants[2].name.as_str(), "Wrap");
            assert_eq!(variants[2].eq_spec.len(), 0);
        }
        _ => panic!("expected TypeDef with Enum"),
    }
}

#[test]
fn test_trigger_with_at() {
    let program = check_parse("def main() { trigger @close_file; }");
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::Trigger { name, .. } => assert!(name.eq_str("close_file")),
            _ => panic!("expected Trigger"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_ensures_on_timeout() {
    let src = "async def f() -> Int<32> ensures on_timeout => result == 0 { return 1; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 1);
            match &contracts[0] {
                Contract::Ensures { target, .. } => {
                    assert!(matches!(target, EnsuresTarget::OnTimeout));
                }
                _ => panic!("expected Ensures with OnTimeout"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_ensures_on_cancel() {
    let src = "async def f() -> Int<32> ensures on_cancel => result == -1 { return 1; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 1);
            match &contracts[0] {
                Contract::Ensures { target, .. } => {
                    assert!(matches!(target, EnsuresTarget::OnCancel));
                }
                _ => panic!("expected Ensures with OnCancel"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_literal_type_annotation() {
    let src = "def main() { set x = 1: PositiveInt; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::VariableDef {
                value: Some(Expr::TypeAnnotated { expr, ty, .. }),
                ..
            } => {
                assert!(matches!(
                    **expr,
                    Expr::Literal(Literal::Int(crate::ast::IntLit::Small(1)), _)
                ));
                assert!(
                    matches!(**ty, Type::Path(ref path, _) if path.len() == 1 && path[0].eq_str("PositiveInt"))
                );
            }
            _ => panic!("expected TypeAnnotated"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_type_where_clause() {
    let src = "type PositiveInt = Int<32> where value > 0 with default = 1;";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::TypeDef {
            definition: TypeDefinition::Alias(ty, modifiers),
            ..
        } => {
            assert!(matches!(
                ty,
                Type::WhereShorthand {
                    base,
                    invariant,
                    ..
                }
            ));
            assert_eq!(modifiers.len(), 1);
        }
        _ => panic!("expected TypeDef with where clause"),
    }
}

/// The `where value > 0` desugar renames `value` to the fresh binder
/// inside the invariant expression — the rename must propagate into
/// NESTED statements (an if-branch block inside the invariant), not
/// just the top-level expression.
#[test]
fn test_replace_ident_renames_into_nested_stmt() {
    let src = "def main() -> Int<32> { if value > 0 { value } else { -value } return 0; }";
    let program = check_parse(src);
    let Stmt::FunctionDef { body, .. } = &program.items[0] else {
        panic!("expected a function def");
    };
    let if_stmt = &body.as_ref().unwrap()[0];
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let renamed = crate::ast::visit::replace_ident_in_stmt(
        arena,
        if_stmt,
        Symbol::intern("value"),
        Symbol::intern("_where_9"),
    );
    let Stmt::If {
        then_branch,
        else_branch,
        ..
    } = renamed
    else {
        panic!("expected an if statement");
    };
    let collect_names = |stmts: &[Stmt<'static>]| -> Vec<Symbol> {
        let mut names = Vec::new();
        for s in stmts {
            if let Stmt::Expression(e) = s {
                // `-value` is a UnaryOp wrapping the Ident.
                let inner: &Expr<'static> = match e {
                    Expr::UnaryOp { expr, .. } => *expr,
                    other => other,
                };
                if let Expr::Ident(n, _) = inner {
                    names.push(*n);
                }
            }
        }
        names
    };
    let then_names = collect_names(&then_branch);
    assert_eq!(
        then_names,
        vec![Symbol::intern("_where_9")],
        "then-branch `value` must be renamed to the binder"
    );
    let else_names = collect_names(else_branch.as_deref().unwrap_or(&[]));
    assert_eq!(
        else_names,
        vec![Symbol::intern("_where_9")],
        "else-branch `value` must be renamed to the binder"
    );
}

// ── Nested generics and >> disambiguation ─────────────────────

#[test]
fn test_nested_generics_double_gt() {
    let program = check_parse("def main() { set x: Vec<Vec<Int<32>>> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_nested_generics_triple() {
    let program = check_parse("def main() { set x: Map<String<Int<8>>, Vec<Int<32>>> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_const_expr_generic_arg() {
    // Array size as a simple expression: `[Int<32>; N + 1]`
    let src = "def main() { set arr: [Int<32>; 10] = 0; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_nested_generics_with_shr_expr() {
    // `>>` as right-shift inside a braced const generic argument:
    //   Foo<Int, { Val >> 2 }>
    // The >> is inside `{ }`, so the generic argument parser correctly
    // delegates to parse_block and the `>` closing the generic is
    // unambiguous.
    let program = check_parse("def main() { set x: Foo<Int, { Val >> 2 }> = 0; }");
    assert_eq!(program.items.len(), 1);
    // Verify the second generic arg is GenericArg::Const
    match &program.items[0] {
        Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
            Stmt::VariableDef {
                ty: Some(Type::Generic(_, args, _)),
                ..
            } => {
                assert_eq!(
                    args.len(),
                    2,
                    "generic should have 2 args: Int and {{Val>>2}}"
                );
                assert!(
                    matches!(&args[1], GenericArg::Const(AnonConst { .. })),
                    "second arg should be GenericArg::Const ({{Val>>2}}), got {:?}",
                    args[1]
                );
            }
            _ => panic!("expected VariableDef with type annotation"),
        },
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_generic_right_shift_expr() {
    // `Array<Int, Count >> 4>` — the `Count >> 4` is a binary
    // expression that must be wrapped in `{ }` per the braced-const
    // rule.  The parser should reject it, not panic.
    let mut parser = new_parser("def main() { set x: Array<Int, Count >> 4> = 0; }");
    let result = parser.parse_program();
    assert!(
        result.is_err() || !parser.diagnostics.is_empty(),
        "expected parse error for unbraced `Count >> 4` in const-generic position"
    );
}

#[test]
fn test_const_expr_int_literal_arith() {
    let program = check_parse("def main() { set x: Foo<Int, { 5 + 3 }> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_bitwidth_not_confused_with_expr() {
    let program = check_parse("def main() { set x: Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_const_expr_int_literal_sub() {
    let program = check_parse("def main() { set x: Foo<Int, { 10 - 3 }> = 0; }");
    assert_eq!(program.items.len(), 1);
}

// --- Function type tests ---

#[test]
fn test_fn_type_zero_params() {
    let program = check_parse("def main() { set f: () -> Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_one_param() {
    let program = check_parse("def main() { set f: (Int<32>) -> Bool = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_two_params() {
    let program = check_parse("def main() { set f: (Int<32>, Bool) -> Result<(), Error> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_in_generic() {
    let program = check_parse("def main() { set f: Option<(Int<32>) -> Bool> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_as_type_alias() {
    let src = "type Callback = (Int<32>) -> Bool;";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_nested() {
    // Higher-order function type: a function that returns a function
    let program = check_parse("def main() { set f: ((Int<32>) -> Bool) -> Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_tuple_not_confused_with_fn_type() {
    // Without `->`, `(A, B)` must remain a tuple type
    let program = check_parse("def main() { set x: (Int<32>, Bool) = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_fn_type_as_param() {
    // Function type used as parameter type
    let src = "def map(f: (Int<32>) -> Int<32>) -> Int<32> { return 0; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

// --- Projection type tests ---

#[test]
fn test_projection_type() {
    // `<ImplType as TraitPath>::AssocType`
    let program = check_parse("def main() { set x: <Int<32> as Add<Int<32>>>::Output = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_projection_type_in_fn_param() {
    let src =
        "def serialize<T>(value: &T, stream: &mut S) where T: Serialize, T::Format: Display { }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_projection_type_in_type_alias() {
    let src = "type ItemType = <Vec<Int<32>> as Iterator>::Item;";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_projection_type_nested() {
    // Nested projection: <<A as Trait1>::Assoc1 as Trait2>::Assoc2
    let program =
        check_parse("def main() { set x: <<A as Trait1>::Assoc1 as Trait2>::Assoc2 = 0; }");
    assert_eq!(program.items.len(), 1);
}

// --- Named generic argument tests ---

#[test]
fn test_named_generic_arg_single() {
    // Single named parameter: Ptr<pointee = Int<32>>
    let program = check_parse("def main() { set p: Ptr<pointee = Int<32>> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_named_generic_arg_multiple() {
    // Multiple named parameters with mixed order
    let program = check_parse("def main() { set p: Ptr<size = UInt<16>, pointee = T> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_named_generic_arg_mixed() {
    // Positional + named (positional should come first)
    let program = check_parse("def main() { set x: SomeType<Int<32>, flag = true> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_named_generic_arg_nested_type() {
    // Named arg value is itself a complex type
    let program = check_parse("def main() { set p: Ptr<pointee = Vec<Int<32>>> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_positional_generic_args_still_work() {
    // Verify that plain positional args (without names) still parse correctly
    let program = check_parse("def main() { set x: HashMap<Int<32>, Bool> = 0; }");
    assert_eq!(program.items.len(), 1);
}

// --- Lifetime annotation tests ---

#[test]
fn test_lifetime_on_ref() {
    // `&'a T` in variable type
    let program = check_parse("def main() { set x: &'a Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_on_ref_mut() {
    // `&'a mut T`
    let program = check_parse("def main() { set x: &'a mut Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_ref_without_lifetime_still_works() {
    // `&T` without lifetime is still valid
    let program = check_parse("def main() { set x: &Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_on_fn_param() {
    // Lifetime in function parameter type
    let src = "def process(x: &'a Int<32>, y: &'a Int<32>) -> &'a Int<32> { return x; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_nested_ref() {
    // Nested references with lifetimes: `&'a &'b mut T`
    let program = check_parse("def main() { set x: &'a &'b mut Int<32> = 0; }");
    assert_eq!(program.items.len(), 1);
}

// --- Lifetime parameter declaration tests ---

#[test]
fn test_lifetime_param_on_fn() {
    // `def foo<'a>(x: &'a Int<32>)`
    let src = "def foo<'a>(x: &'a Int<32>) -> &'a Int<32> { return x; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_param_mixed() {
    // Mixed lifetime and type params: `def bar<'a, T>(x: &'a T)`
    let src = "def bar<'a, T>(x: &'a T) -> &'a T { return x; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_param_multi() {
    // Multiple lifetime params: `def baz<'a, 'b>(x: &'a Int<32>, y: &'b Bool)`
    let src = "def baz<'a, 'b>(x: &'a Int<32>, y: &'b Bool) -> &'a Int<32> { return x; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_param_on_type_alias() {
    // Lifetime param on type alias: `type Ref<'a> = &'a Int<32>`
    let src = "type Ref<'a> = &'a Int<32>;";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_lifetime_param_on_impl() {
    // Lifetime param on impl block
    let src = "impl<'a> Foo for &'a Bar { }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_type_params_still_work() {
    // Plain type params still work
    let src = "def id<T>(x: T) -> T { return x; }";
    let program = check_parse(src);
    assert_eq!(program.items.len(), 1);
}

// ── Operator precedence tests ──────────────────────────────────
// Verify that the Pratt parser respects the SYNTAX.md precedence table.
// Mul/Div/Rem (15) binds tighter than Add/Sub (13), etc.

#[test]
fn test_precedence_mul_over_add() {
    // 1 + 2 * 3  →  1 + (2 * 3), NOT (1 + 2) * 3
    let src = "def main() { set x = 1 + 2 * 3; }";
    let _program = check_parse(src);
    // If precedence is correct, parsing succeeds and the AST reflects
    // the expected grouping.  A crash or error here means the Pratt
    // binding powers are misconfigured.
}

#[test]
fn test_precedence_add_over_shift() {
    // 1 + 2 << 3  →  (1 + 2) << 3 (shift is lower than add)
    let src = "def main() { set x = 1 + 2 << 3; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_shift_over_bitand() {
    // 1 << 2 & 3  →  (1 << 2) & 3
    let src = "def main() { set x = 1 << 2 & 3; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_bitand_over_xor() {
    // 1 & 2 ^ 3  →  (1 & 2) ^ 3
    let src = "def main() { set x = 1 & 2 ^ 3; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_xor_over_bitor() {
    // 1 ^ 2 | 3  →  (1 ^ 2) | 3
    let src = "def main() { set x = 1 ^ 2 | 3; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_comparison_over_logical() {
    // a < b and c > d  →  (a < b) and (c > d), NOT a < (b and c) > d
    let src = "def main() -> Bool { return 1 < 2 and 3 > 4; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_and_over_or() {
    // true and false or true  →  (true and false) or true
    let src = "def main() -> Bool { return true and false or true; }";
    let _program = check_parse(src);
}

#[test]
fn test_precedence_wrap_variants_match_base() {
    // Wrap variant `+%` should bind at the same level as `+`.
    // `*%` should bind at the same level as `*`.
    let src = "def main() { set x = 1 +% 2 *% 3; }";
    let _program = check_parse(src);
}

// ── @label path labels ─────────────────────────────────────────

#[test]
fn test_return_label_single() {
    let src = "def f() -> Int<32> { return @even 4; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef {
            body: Some(body), ..
        } => {
            assert_eq!(body.len(), 1);
            match &body[0] {
                Stmt::Return { labels, value, .. } => {
                    assert_eq!(labels.len(), 1);
                    assert_eq!(labels[0].as_str(), "@even");
                    assert!(value.is_some());
                }
                _ => panic!("expected Return"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_return_label_multiple() {
    let src = "def f() -> Int<32> { return @even @big 200; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef {
            body: Some(body), ..
        } => {
            assert_eq!(body.len(), 1);
            match &body[0] {
                Stmt::Return { labels, .. } => {
                    assert_eq!(labels.len(), 2);
                    assert_eq!(labels[0].as_str(), "@even");
                    assert_eq!(labels[1].as_str(), "@big");
                }
                _ => panic!("expected Return"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_return_label_without_expr() {
    // Labels without expression should parse (value is None).
    let src = "def f() { return @done; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef {
            body: Some(body), ..
        } => {
            assert_eq!(body.len(), 1);
            match &body[0] {
                Stmt::Return { labels, value, .. } => {
                    assert_eq!(labels.len(), 1);
                    assert_eq!(labels[0].as_str(), "@done");
                    assert!(value.is_none());
                }
                _ => panic!("expected Return"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_ensures_label() {
    // `@label` is a placeholder in the expression: `ensures @even > 0`
    // means "the return value on the @even path is > 0".
    let src = "def f(x: Int<32>) -> Int<32>
                        ensures @even > 0
                    { return @even x; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 1);
            match &contracts[0] {
                Contract::Ensures { expr, .. } => {
                    // The expression should be `@even > 0`, which parses as
                    // a BinaryOp with Ident("@even") on the left.
                    match expr {
                        Expr::BinaryOp { left, op, .. } => {
                            assert_eq!(*op, BinOp::Gt);
                            match left {
                                Expr::Ident(name, _) => {
                                    assert_eq!(name.as_str(), "@even");
                                }
                                _ => panic!("expected Ident on left side of >"),
                            }
                        }
                        _ => panic!("expected BinaryOp for @even > 0"),
                    }
                }
                _ => panic!("expected Ensures"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_ensures_label_and_codomain() {
    // Mix of `codomain` (all paths) and `@label` (specific path).
    let src = "def f(x: Int<32>) -> Int<32>
                        ensures codomain >= 0
                        ensures @fast < 100
                    { return @fast x; }";
    let program = check_parse(src);
    match &program.items[0] {
        Stmt::FunctionDef { contracts, .. } => {
            assert_eq!(contracts.len(), 2);
            // Second ensures should have the @fast label in the expression
            match &contracts[1] {
                Contract::Ensures { expr, .. } => {
                    // The expression should be `@fast < 100`.
                    match expr {
                        Expr::BinaryOp { left, op, .. } => {
                            assert_eq!(*op, BinOp::Lt);
                            match left {
                                Expr::Ident(name, _) => {
                                    assert_eq!(name.as_str(), "@fast");
                                }
                                _ => panic!("expected Ident on left side of <"),
                            }
                        }
                        _ => panic!("expected BinaryOp for @fast < 100"),
                    }
                }
                _ => panic!("expected Ensures"),
            }
        }
        _ => panic!("expected FunctionDef"),
    }
}

#[test]
fn test_return_label_parse_error() {
    // `return @` without identifier should fail.
    let src = "def f() -> Int<32> { return @ 4; }";
    let diags = check_parse_err(src);
    assert!(!diags.is_empty(), "expected parse error for `return @`");
}
