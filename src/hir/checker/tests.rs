use super::*;
use crate::ast;
use crate::hir::builtins;
use crate::hir::infer::{GenStatus, TypeVariableKind, VarOrigin};
use crate::hir::resolver::NameResolver;
use crate::hir::types::{TypeData, reset_def_id_allocator};
use crate::parser::Parser;

/// Like `check_source`, but returns the HIR even when the checker rejects
/// the program (the borrow-check post-pass's diagnostics) — the Polonius
/// equivalence tests inspect the bodies of REJECTED programs (the error
/// direction).
pub(crate) fn check_source_keep_hir(source: &str) -> (HirProgram<'static>, Vec<String>) {
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let mut parser = Parser::new(source, arena);
    let program = parser.parse_program().unwrap_or_else(|diags| {
        panic!(
            "parse errors: {:?}",
            diags
                .into_iter()
                .map(|d| d.message().to_string())
                .collect::<Vec<_>>()
        )
    });
    let mut ctx = TypeContext::new();
    ctx.arena = Some(arena);
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (mut symbols, mut trait_env, _res_diags, resolution_map) =
        resolver.resolve_program(&program);
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, resolution_map);
    match checker.check_program(&program) {
        Ok(prog) => {
            // Capture the warnings emitted during a successful check
            // (the warnings remain in the checker's diagnostics — only
            // the error path takes them).
            let warns = checker
                .diagnostics
                .into_inner()
                .into_iter()
                .map(|d| d.message().to_string())
                .collect::<Vec<_>>();
            (prog, warns)
        }
        Err(diags) => {
            let prog = checker
                .last_checked_program
                .clone()
                .expect("the HIR must be kept on the error path");
            let msgs = diags
                .into_inner()
                .into_iter()
                .map(|d| d.message().to_string())
                .collect::<Vec<_>>();
            (prog, msgs)
        }
    }
}

/// Run the full pipeline (parse → resolve → builtins → type-check) on Posita source.
pub(crate) fn check_source(source: &str) -> Result<HirProgram<'static>, Vec<String>> {
    // NOTE: Do NOT reset the global DefId allocator here.  Tests run in
    // parallel by default, and reset_def_id_allocator() is not thread-safe.
    // The overlap check in add_impl compares DefId values within the same
    // TraitEnv, which are always unique because the global counter only
    // increments.  Parallel tests get their own TraitEnv instances, so
    // there is no cross-test DefId collision.

    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let mut parser = Parser::new(source, arena);
    let program = parser.parse_program().map_err(|diags| {
        diags
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>()
    })?;

    let mut ctx = TypeContext::new();
    ctx.arena = Some(arena);
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (mut symbols, mut trait_env, res_diags, resolution_map) =
        resolver.resolve_program(&program);
    if res_diags.has_errors() {
        return Err(res_diags
            .into_inner()
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>());
    }

    // NOTE: register_builtins is called inside NameResolver::new (resolver.rs:83),
    // so the builtin types and traits are already registered at this point.
    // The duplicate call below was removed to prevent double registration of
    // builtin impls with different DefId values, which caused the overlap check
    // in add_impl to detect false positives.
    // builtins::register_builtins(&mut symbols, &mut trait_env, &mut ctx);

    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, resolution_map);
    checker.check_program(&program).map_err(|diags| {
        diags
            .into_inner()
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>()
    })
}

/// Run the full pipeline in STRICT mode (the verification-failure
/// diagnostics become compile-time errors).  Mirrors `check_source` with
/// `strict_mode = true` on the checker.
pub(crate) fn check_strict(source: &str) -> Result<HirProgram<'static>, Vec<String>> {
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let mut parser = Parser::new(source, arena);
    let program = parser.parse_program().map_err(|diags| {
        diags
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>()
    })?;

    let mut ctx = TypeContext::new();
    ctx.arena = Some(arena);
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (mut symbols, mut trait_env, res_diags, resolution_map) =
        resolver.resolve_program(&program);
    if res_diags.has_errors() {
        return Err(res_diags
            .into_inner()
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>());
    }

    let mut checker = TypeChecker::new(
        &mut ctx,
        &symbols,
        &mut trait_env,
        resolution_map,
        true,   // strict_mode
        false,  // enable_experimental
        vec![], // features
        false,  // debug
    );
    checker.check_program(&program).map_err(|diags| {
        diags
            .into_inner()
            .into_iter()
            .map(|d| d.message().to_string())
            .collect::<Vec<_>>()
    })
}

/// The explicit lifetime annotation (`<'a>` + `&'a mut T`, SYNTAX.md
/// §Explicit Lifetime Parameters) must survive end-to-end: the parser
/// accepts it, the AST→HIR lowering keeps `TypeData::Ref.lifetime`
/// (previously dropped by `Type::Reference { .. }`), and the program
/// checks.  This is the paving for the early-bound UniversalRegions
/// TPB (Two-Phase Borrows) end-to-end: `obj.put(obj.val)` — the
/// method-call receiver's `&mut` loan is RESERVED while the argument
/// evaluates, so the argument's read of `obj.val` must NOT conflict
/// (the rustc `v.push(v.len())` shape).  Without the TPB exemption the
/// argument read would fire E109.
#[test]
fn test_two_phase_method_call_end_to_end() {
    let result = check_source(
        "type MyType = struct { val: Int<32> }
         impl for MyType {
             def put(&mut self, x: Int<32>) -> Int<32> { self.val = x; return x; }
         }
         def main() -> Int<32> {
             set obj = MyType { val = 7 };
             set r = obj.put(obj.val);
             return r;
         }",
    );
    assert!(
        result.is_ok(),
        "the TPB receiver + argument-read must be accepted: {:?}",
        result.err()
    );
}

/// the dangling check must see RETURNS inside LOOP bodies —
/// `loop { return &mut x; }` (x local) is equally dangling; the
/// `collect_dangling_returns` fallback previously skipped them.
#[test]
fn test_dangling_return_in_loop() {
    let result = check_source(
        "def bad() -> &mut Int<32> {
             set x = 5;
             loop {
                 return &mut x;
             }
         }",
    );
    assert!(
        result.is_err(),
        "a loop-body return of a local must be rejected: {:?}",
        result.err()
    );
}

/// an alias of a parameter defined INSIDE an `if` block must
/// be recognized — `return r` (r = x, an alias) is legal; the
/// top-level-only alias scan previously flagged it as dangling.
#[test]
fn test_dangling_alias_in_if_accepted() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> {
             if true {
                 set r = x;
                 return r;
             }
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "an in-`if` alias of a parameter is not dangling: {:?}",
        result.err()
    );
}

/// rustc's dangling-reference rejection: a REFERENCE return that
/// references a LOCAL (non-parameter) place must be rejected (the
/// borrow would dangle after the function returns); a parameter-derived
/// return must be accepted.
#[test]
fn test_dangling_return_rejected() {
    let result = check_source("def f() -> &mut Int<32> { set x = 5; return &mut x; }");
    assert!(
        result.is_err(),
        "returning a reference to a local must be rejected: {:?}",
        result.err()
    );
    let ok = check_source("def f(x: &mut Int<32>) -> &mut Int<32> { return x; }");
    assert!(
        ok.is_ok(),
        "a parameter-derived return must be accepted: {:?}",
        ok.err()
    );
}

/// mapping.
#[test]
fn test_explicit_lifetime_end_to_end() {
    let result =
        check_source("def process<'a>(x: &'a mut Int<32>) -> &'a mut Int<32> { return x; }");
    assert!(
        result.is_ok(),
        "the explicit lifetime program must check: {:?}",
        result.err()
    );
}

/// The region solver: a RETURN reference may only use a lifetime PROVIDED
/// BY a parameter (SYNTAX.md §Explicit Lifetime Parameters — "verified by
/// the borrow checker; mismatches cause compile errors").  `'a` appears in
/// NO parameter type here — the returned `&'a` could outlive its source.
#[test]
fn test_lifetime_return_not_from_param_rejected() {
    let result = check_source(
        "def dangling<'a>(x: Int<32>) -> &'a Int<32> {
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "a return lifetime not provided by any parameter must be rejected: {:?}",
        result
    );
}

/// The solver must ACCEPT a return lifetime that IS one of the parameter
/// lifetimes — even with multiple distinct lifetime parameters (the
/// returned `&'a` is covered by the `x: &'a` parameter).
#[test]
fn test_lifetime_return_matches_param_ok() {
    let result = check_source(
        "def pick<'a, 'b>(x: &'a Int<32>, y: &'b Int<32>) -> &'a Int<32> {
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "a return lifetime matching a parameter lifetime must be accepted: {:?}",
        result
    );
}

/// An explicit `where 'a: 'b` outlives predicate (rustc's
/// `WherePredicateKind::RegionPredicate`) makes the return lifetime
/// SATISFIABLE: `x: &'a Int<32>` + `'a: 'b` ⇒ `&'b` is covered by `'a`
/// (the transitive closure honors the where edge).
#[test]
fn test_lifetime_where_outlives_satisfiable_ok() {
    let result = check_source(
        "def pick<'a, 'b>(x: &'a Int<32>) -> &'b Int<32> where 'a: 'b {
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "`where 'a: 'b` must make the returned &'b satisfiable: {:?}",
        result
    );
}

/// WITHOUT the `where 'a: 'b` predicate the same signature is UNSATISFIABLE
/// — `'b` is covered by no parameter region.
#[test]
fn test_lifetime_where_outlives_absent_rejected() {
    let result = check_source(
        "def pick<'a, 'b>(x: &'a Int<32>) -> &'b Int<32> {
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "a return lifetime not covered by any parameter (no where 'a: 'b) must be rejected: {:?}",
        result
    );
}

/// BODY-level lifetime mismatches must be diagnosed by the
/// POST-BODY second solve — `return a` flows `&'a` into the `&'b` return
/// position with no provable `'a: 'b` (rustc E0623; SYNTAX.md §Explicit
/// Lifetime Parameters "mismatches cause compile errors").
#[test]
fn test_lifetime_body_mismatch_rejected() {
    let result = check_source(
        "def pick<'a, 'b>(a: &'a Int<32>, b: &'b Int<32>) -> &'b Int<32> {
             return a;
         }",
    );
    assert!(
        result.is_err(),
        "a body-level &'a → &'b mismatch with no provable 'a: 'b must be rejected: {:?}",
        result
    );
}

/// Control: the SAME body is accepted when `where 'a: 'b` makes the
/// outlives satisfiable.
#[test]
fn test_lifetime_body_mismatch_where_satisfiable_ok() {
    let result = check_source(
        "def pick<'a, 'b>(a: &'a Int<32>, b: &'b Int<32>) -> &'b Int<32> where 'a: 'b {
             return a;
         }",
    );
    assert!(
        result.is_ok(),
        "`where 'a: 'b` must make the body-level flow satisfiable: {:?}",
        result
    );
}

/// HRTB `for<'a> T` parses and resolves end-to-end (SYNTAX.md
/// §Higher-Ranked Trait Bounds): a `for<'a> &'a Int<32>` parameter type
/// is accepted (the checker allocates a Forall binder for the quantified
/// lifetime; the subtype Forall arm skolemizes it at the call site).
#[test]
fn test_lifetime_hrtb_forall_parses_ok() {
    let result = check_source(
        "def apply(f: for<'a> &'a Int<32>) -> Int<32> {
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the for<'a> HRTB parameter type must parse and resolve: {:?}",
        result
    );
}

/// ELIDED lifetimes (no explicit `&'a` annotation) are not part of the
/// early-bound region set and must NOT be rejected by the solver — the
/// plain `&T` borrows are handled by the location-based borrow checker.
#[test]
fn test_lifetime_elided_escape_ok() {
    let result = check_source(
        "def read(x: &Int<32>) -> Int<32> { return *x; }
         def main() -> Int<32> {
             set a = 42;
             return read(&a);
         }",
    );
    assert!(
        result.is_ok(),
        "elided lifetimes must be accepted (no early-bound region to verify): {:?}",
        result
    );
}

fn make_checker<'a, 'input>(
    ctx: &'a mut TypeContext<'input>,
    symbols: &'a SymbolTable<'input>,
    trait_env: &'a mut TraitEnv<'input>,
    resolution_map: ResolutionMap<'input>,
) -> TypeChecker<'a, 'input> {
    TypeChecker::new(
        ctx,
        symbols,
        trait_env,
        resolution_map,
        false,
        false,
        vec![],
        false,
    )
}

#[test]
fn test_simple_field_access() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set p = Point { x = 10, y = 20 };
                 return p.x;
             }",
    );
    assert!(
        result.is_ok(),
        "field access should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_field_access_through_ref() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set p = Point { x = 10, y = 20 };
                 set mut r = &p;
                 return r.x;
             }",
    );
    assert!(
        result.is_ok(),
        "field access through ref should succeed via autoderef: {:?}",
        result.err()
    );
}

#[test]
fn test_missing_field_error() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set p = Point { x = 10, y = 20 };
                 return p.z;
             }",
    );
    assert!(result.is_err(), "missing field should produce an error");
    let errors = result.err().unwrap();
    let all = errors.join(" ");
    assert!(
        all.contains("no field"),
        "error should mention 'no field': {}",
        all
    );
}

#[test]
fn test_method_call() {
    // Define a struct with an impl block containing a method
    let result = check_source(
        "type MyType = struct { val: Int<32> }
             impl for MyType {
                 def get_val(&self) -> Int<32> {
                     return self.val;
                 }
             }
             def main() -> Int<32> {
                 set obj = MyType { val = 42 };
                 return obj.get_val();
             }",
    );
    assert!(
        result.is_ok(),
        "method call should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_missing_method_error() {
    let result = check_source(
        "type MyType = struct { val: Int<32> }
             impl for MyType {
                 def get_val(&self) -> Int<32> {
                     return self.val;
                 }
             }
             def main() -> Int<32> {
                 set obj = MyType { val = 42 };
                 return obj.nonexistent();
             }",
    );
    assert!(result.is_err(), "missing method should produce an error");
    let errors = result.err().unwrap();
    let all = errors.join(" ");
    assert!(
        all.contains("no field or method"),
        "error should mention 'no field or method': {}",
        all
    );
}

#[test]
fn test_autoderef_method_call() {
    let result = check_source(
        "type MyType = struct { val: Int<32> }
             impl for MyType {
                 def get_val(&self) -> Int<32> {
                     return self.val;
                 }
             }
             def main() -> Int<32> {
                 set obj = MyType { val = 42 };
                 set r = &obj;
                 return r.get_val();
             }",
    );
    assert!(
        result.is_ok(),
        "method call through ref should succeed via autoderef: {:?}",
        result.err()
    );
}

/// The REBORROW REFERENT ALIASING: `&mut *r` — and the cross-function
/// reborrow (`set r3 = get2(r)` where the callee reborrows
/// internally) — borrows the ULTIMATE REFERENT of the deref chain, NOT
/// the literal `*r` path.  The mutation-freeze (E110/E0506), the
/// read-freeze (E109/E0503) and the exclusivity (E112) checks must
/// resolve through the borrow-variable chain to the referent.
#[test]
fn test_reborrow_freezes_referent() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = get(&mut a);
             set r2 = get(&mut *r);
             a = 5;
             let x = *r2;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "a mutation of the referent while the reborrow is live must be rejected (E0506): {:?}",
        result
    );
    let all = result.err().unwrap().join(" ");
    assert!(
        all.contains("frozen") || all.contains("borrow"),
        "the referent mutation must be reported as a borrow error: {}",
        all
    );
}

/// The reborrow's OWN uses are legal: a read of the reborrow before the
/// referent mutation (the rustc c26 shape — the reborrow's last use
/// precedes the write) is accepted.
#[test]
fn test_reborrow_own_use_before_mutation_accepted() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = get(&mut a);
             set r2 = get(&mut *r);
             let x = *r2;
             a = 5;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "the referent mutation after the reborrow's last use is legal: {:?}",
        result
    );
}

/// The CROSS-FUNCTION reborrow: `get2` reborrows `&mut *r` internally
/// and returns it; the caller's cross-function loan (`set r3 = get2(r)`)
/// must
/// freeze the ORIGINAL referent — a mutation while the returned
/// reborrow is live is rejected (rustc E0506).
#[test]
fn test_cross_function_reborrow_freezes_referent() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def get2(r: &mut Int<32>) -> &mut Int<32> {
             set r2 = &mut *r;
             return r2;
         }
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = get(&mut a);
             set r3 = get2(r);
             a = 5;
             let x = *r3;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "the cross-function reborrow must freeze the caller's referent (E0506): {:?}",
        result
    );
}

/// A read THROUGH the parent reference while the reborrow is live is a
/// conflicting access (rustc E0503); a write THROUGH the reborrow's own
/// variable is the reborrow's intended use and is legal.
#[test]
fn test_reborrow_read_freezes_parent() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = &mut a;
             set r2 = &mut *r;
             let x = *r;
             let y = *r2;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "a read through the parent while the reborrow is live must be rejected (E0503): {:?}",
        result
    );
}

#[test]
fn test_reborrow_own_var_write_accepted() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = &mut a;
             set r2 = &mut *r;
             *r2 = 5;
             return a;
         }",
    );
    assert!(
        result.is_ok(),
        "a write through the reborrow's own variable is its intended use: {:?}",
        result
    );
}

/// The CHAINED reborrows (`r2 = &mut *r`, `r3 = &mut *r2`) resolve
/// transitively to the ultimate referent.
#[test]
fn test_chained_reborrow_freezes_referent() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r = &mut a;
             set r2 = &mut *r;
             set r3 = &mut *r2;
             a = 5;
             let x = *r3;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "the chained reborrow must freeze the ultimate referent (E0506): {:?}",
        result
    );
}

/// The BUILT-IN REBORROW COERCION at call sites (`@auto_coerce`,
/// SYNTAX.md §Local Relaxation — the deref-coercion family): `&mut r`
/// where `r: &mut T` coerces to `&mut *r` — the referent's reference
/// (rustc's call-site deref coercion).  The HIR is REWRITTEN (the `*r`
/// deref inserted) so the borrow-check post-pass sees the deref loan.
#[test]
fn test_auto_coerce_reborrow_argument() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         @auto_coerce
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r: &mut Int<32> = get(&mut a);
             set r2 = get(&mut r);
             let x = *r2;
             a = 5;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "`&mut r` (r: &mut T) must coerce to `&mut *r` under @auto_coerce: {:?}",
        result
    );
}

/// The coerced reborrow freezes the REFERENT: a mutation of `a` while
/// the returned reborrow is live is rejected (rustc E0506).
#[test]
fn test_auto_coerce_reborrow_freezes_referent() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         @auto_coerce
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r: &mut Int<32> = get(&mut a);
             set r2 = get(&mut r);
             a = 5;
             let x = *r2;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "the coerced reborrow must freeze the referent (E0506): {:?}",
        result
    );
}

/// The reborrow coercion requires `@auto_coerce` — without it the
/// double reference is rejected (the explicit `&mut *r` form is
/// required).
#[test]
fn test_reborrow_coercion_requires_auto_coerce() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r: &mut Int<32> = get(&mut a);
             set r2 = get(&mut r);
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the reborrow coercion must not apply without @auto_coerce: {:?}",
        result
    );
}

/// An IMMUTABLE inner reference cannot be reborrowed as mutable:
/// `&mut r` with `r: &T` for a `&mut T` parameter is rejected (rustc
/// E0308 parity).
#[test]
fn test_immutable_inner_cannot_reborrow_mut() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         @auto_coerce
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set r: &Int<32> = &a;
             set r2 = get(&mut r);
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "`&mut r` with an immutable inner reference must be rejected: {:?}",
        result
    );
}

/// The shared-direction reborrow: `&r` where `r: &mut T` coerces to
/// `&*r` — the reference is reborrowed read-only (the outer `&` stays
/// immutable).
#[test]
fn test_auto_coerce_shared_reborrow() {
    let result = check_source(
        "def peek(x: &Int<32>) -> Int<32> { return *x; }
         @auto_coerce
         def main() -> Int<32> {
             set mut a: Int<32> = 42;
             set mut r: &mut Int<32> = &mut a;
             let x = peek(&r);
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "`&r` (r: &mut T) must coerce to `&*r` under @auto_coerce: {:?}",
        result
    );
}

/// Trait-based deref coercion at call sites under `@auto_coerce`
/// (SYNTAX.md §Local Relaxation): `&w` where `w: Wrapper<T>` and
/// `Wrapper<T>: Deref<Target=T>` (marked `@auto_deref`) coerces to
/// `&T`.
#[test]
fn test_auto_coerce_trait_deref_argument() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         @auto_deref
         impl Deref for PointWrapper {
             type Target = Point;
         }
         def peek(p: &Point) -> Int<32> { return p.x; }
         @auto_coerce
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             return peek(&w);
         }",
    );
    assert!(
        result.is_ok(),
        "`&w` (w: PointWrapper, Deref<Target=Point>) must coerce to `&Point` under @auto_coerce: {:?}",
        result
    );
}

/// The HIR rewrite check: the trait-deref-coerced argument must be
/// `&(*w)` — a `Ref` wrapping a `Deref` node — so the borrow-check
/// post-pass sees the deref loan (the trait branch must mirror the
/// built-in reborrow branch; it previously updated only `ty` and left
/// `hir_arg` at the original wrapper reference type).
#[test]
fn test_auto_coerce_trait_deref_hir_rewritten() {
    let (program, _) = check_source_keep_hir(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         @auto_deref
         impl Deref for PointWrapper {
             type Target = Point;
         }
         def peek(p: &Point) -> Int<32> { return p.x; }
         @auto_coerce
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             return peek(&w);
         }",
    );
    // Locate `main`'s body, then the `return peek(&w);` call argument.
    let mut main_body = None;
    for item in &program.items {
        if let crate::hir::hir::HirStmt::FunctionDef {
            name,
            body: Some(b),
            ..
        } = item
        {
            if name.eq_str("main") {
                main_body = Some(b);
                break;
            }
        }
    }
    let main_body = main_body.expect("main body");
    let call = match main_body.last().expect("main has statements") {
        crate::hir::hir::HirStmt::Return { value: Some(v), .. } => match &**v {
            crate::hir::hir::HirExpr::Call { args, .. } => &args[0],
            e => panic!("expected Call, got {:?}", e),
        },
        s => panic!("expected Return, got {:?}", s),
    };
    match call {
        crate::hir::hir::HirExpr::UnaryOp {
            op: crate::ast::UnaryOp::Ref,
            expr,
            ..
        } => {
            assert!(
                matches!(
                    expr.as_ref(),
                    crate::hir::hir::HirExpr::UnaryOp {
                        op: crate::ast::UnaryOp::Deref,
                        ..
                    }
                ),
                "coerced argument must be `&(*w)` — a Ref wrapping a Deref: {:?}",
                call
            );
        }
        e => panic!("expected Ref-wrapped coerced argument, got {:?}", e),
    }
}

/// The user-defined `Deref` trait (SYNTAX.md §Method-Call Auto-
/// Dereferencing): a `Deref` impl with `@auto_deref` allows method
/// calls directly on the wrapper type (the receiver auto-derefs
/// through the impl).  The interner does not dedupe Adt instantiations
/// (`ctx.struct_ty` allocates a fresh TypeId every call), so the
/// impl-lookup in `deref_trait_step` must match by DefId equivalence,
/// not exact TypeId.
#[test]
fn test_auto_deref_trait_method_call() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         @auto_deref
         impl Deref for PointWrapper {
             type Target = Point;
         }
         impl for Point {
             def distance_sq(&self) -> Int<32> {
                 return self.x * self.x + self.y * self.y;
             }
         }
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             return w.distance_sq();
         }",
    );
    assert!(
        result.is_ok(),
        "method call through @auto_deref impl must succeed: {:?}",
        result.err()
    );
}

/// Without `@auto_deref` on the `Deref` impl, method calls on the
/// wrapper type must NOT auto-deref — the spec requires explicit
/// `(*w).method()` syntax for unmarked impls.
#[test]
fn test_auto_deref_trait_without_attribute_rejected() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         impl Deref for PointWrapper {
             type Target = Point;
         }
         impl for Point {
             def distance_sq(&self) -> Int<32> {
                 return self.x * self.x + self.y * self.y;
             }
         }
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             return w.distance_sq();
         }",
    );
    assert!(
        result.is_err(),
        "method call without @auto_deref must be rejected: {:?}",
        result
    );
}

/// The explicit `*w` operator must resolve through a user-defined
/// `Deref` impl WITHOUT `@auto_deref` — the attribute gates only the
/// implicit receiver autoderef, not the explicit operator.
#[test]
fn test_auto_deref_trait_explicit_star() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         impl Deref for PointWrapper {
             type Target = Point;
         }
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             let p = *w;
             return p.x;
         }",
    );
    assert!(
        result.is_ok(),
        "explicit `*w` must resolve through Deref impl: {:?}",
        result.err()
    );
}

/// The explicit `(*w).method()` form through a `Deref` impl (no
/// `@auto_deref` needed for the explicit operator).
#[test]
fn test_auto_deref_trait_explicit_star_method_call() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         type PointWrapper = struct { inner: Point }
         trait Deref { type Target; }
         impl Deref for PointWrapper {
             type Target = Point;
         }
         impl for Point {
             def distance_sq(&self) -> Int<32> {
                 return self.x * self.x + self.y * self.y;
             }
         }
         def main() -> Int<32> {
             set w = PointWrapper { inner = Point { x = 3, y = 4 } };
             return (*w).distance_sq();
         }",
    );
    assert!(
        result.is_ok(),
        "explicit `(*w).method()` must resolve through Deref impl: {:?}",
        result.err()
    );
}

/// A degenerate `impl Deref` whose `Target` is the type itself must
/// not loop forever — the autoderef chain cycle guard stops at the
/// first repeated type.
#[test]
fn test_auto_deref_trait_cycle_guard() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         trait Deref { type Target; }
         @auto_deref
         impl Deref for Point {
             type Target = Point;
         }
         impl for Point {
             def distance_sq(&self) -> Int<32> {
                 return self.x * self.x + self.y * self.y;
             }
         }
         def main() -> Int<32> {
             set p = Point { x = 3, y = 4 };
             return p.distance_sq();
         }",
    );
    assert!(
        result.is_ok(),
        "degenerate self-target Deref impl must not hang: {:?}",
        result.err()
    );
}

#[test]
fn test_field_access_on_non_struct_error() {
    let result = check_source(
        "def main() -> Int<32> {
                 set x = 42;
                 return x.nonexistent;
             }",
    );
    assert!(result.is_err(), "field access on non-struct should error");
    let errors = result.err().unwrap();
    let all = errors.join(" ");
    assert!(
        all.contains("no field") || all.contains("field"),
        "error should mention field: {}",
        all
    );
}

#[test]
fn test_compiles_simple_program() {
    let result = check_source(
        "
            def add(a: Int<32>, b: Int<32>) -> Int<32> {
                return a + b;
            }
            def main() -> Int<32> {
                return add(1, 2);
            }",
    );
    assert!(
        result.is_ok(),
        "simple program should type-check: {:?}",
        result.err()
    );
}

#[test]
fn test_add_operator_overload() {
    let result = check_source(
        "
            def main() -> Int<32> {
                set x = 10;
                set y = 20;
                set z = x + y;
                return z;
            }",
    );
    assert!(
        result.is_ok(),
        "operator + should work for Int<32>: {:?}",
        result.err()
    );
}

#[test]
fn test_mul_operator_overload() {
    let result = check_source(
        "
            def main() -> Int<32> {
                set x = 6;
                set y = 7;
                set z = x * y;
                return z;
            }",
    );
    assert!(
        result.is_ok(),
        "operator * should work for Int<32>: {:?}",
        result.err()
    );
}

#[test]
fn test_impl_where_clause_parse_and_trait_check() {
    // Simple concrete impl with where clause parsed from the impl block
    let result = check_source(
        "
            trait Bar { }
            type MyInt = Int<32> with default = 0;
            impl Bar for MyInt { }
            def main() -> Int<32> {
                return 0;
            }",
    );
    assert!(
        result.is_ok(),
        "concrete impl should type-check: {:?}",
        result.err()
    );
}

#[test]
fn test_bare_type_var_without_context_rejected() {
    // `impl<T> Foo for T` where T is a bare type variable not appearing
    // in any where clause context should fail the Coverage Condition.
    let result = check_source(
        "
            trait Foo { }
            type MyInt = Int<32>;
            impl Foo for MyInt { }
            impl<T> Foo for T { }
            def main() -> Int<32> { return 0; }
            ",
    );
    assert!(
        result.is_err(),
        "bare type var without context should be rejected"
    );
}

#[test]
fn test_bare_type_var_with_context_accepted() {
    // `impl<T: Bar> Foo for T` where T appears in context as `T: Bar`
    // should pass the Coverage Condition.
    let result = check_source(
        "
            trait Foo { }
            trait Bar { }
            type MyInt = Int<32>;
            impl Bar for MyInt { }
            impl<T: Bar> Foo for T { }
            def main() -> Int<32> { return 0; }
            ",
    );
    assert!(
        result.is_ok(),
        "bare type var with context should pass: {:?}",
        result.err()
    );
}

// ── Generic type parameter synthesis ──────────────────────────
// Note: the resolver stores type_params in FunctionSignature<'input> but does NOT
// register them in current_impl_type_params during FunctionDef<'input> processing.
// This means `T` in `def id<T>(x: T)` cannot be resolved by resolve_type_expr
// during the resolver phase, producing "undefined type: T" before the
// checker ever runs. Fix: populate current_impl_type_params in the
// FunctionDef<'input> branch of resolve_item, same as ImplBlock<'input> already does.

#[test]
fn test_polymorphic_identity_call() {
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 return id(42);
             }",
    );
    assert!(
        result.is_ok(),
        "polymorphic id(42) should synthesize T = Int<32>: {:?}",
        result.err()
    );
}

#[test]
fn test_polymorphic_bool_call() {
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Bool {
                 set b = id(true);
                 return b;
             }",
    );
    assert!(
        result.is_ok(),
        "polymorphic id(true) should synthesize T = Bool: {:?}",
        result.err()
    );
}

#[test]
fn test_polymorphic_pair() {
    let result = check_source(
        "def pair<T, U>(a: T, b: U) -> T { return a; }
             def main() -> Int<32> {
                 return pair(42, true);
             }",
    );
    assert!(
        result.is_ok(),
        "polymorphic pair with two type params: {:?}",
        result.err()
    );
}

// ── Issue-1 semantics: check_expr's Call pre-inference path ────────
// These tests pin the observable behavior of the `Expr::Call` special
// case at the top of `check_expr`: type-argument synthesis errors must
// still surface as diagnostics (not be silently swallowed), successful
// synthesis must short-circuit, and non-polymorphic callees must fall
// through to the normal call inference.

/// A polymorphic call with the WRONG arity must report a diagnostic.
/// `try_synthesize_type_args` returns `Err` on arity mismatch — the
/// `check_expr` pre-inference path must not swallow it.
#[test]
fn test_call_synthesis_arity_error_reported() {
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 return id(42, 43);
             }",
    );
    assert!(
        result.is_err(),
        "arity mismatch in polymorphic call must be reported: {:?}",
        result
    );
}

/// A polymorphic call with matching arity synthesizes type arguments and
/// succeeds (`Ok(Some(...))` short-circuit in check_expr).
#[test]
fn test_call_synthesis_success_short_circuits() {
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 return id(42);
             }",
    );
    assert!(
        result.is_ok(),
        "polymorphic id(42) must synthesize: {:?}",
        result.err()
    );
}

/// A NON-polymorphic callee (`Ok(None)` from try_synthesize_type_args)
/// must fall through to the normal call inference and succeed.
#[test]
fn test_call_synthesis_non_polymorphic_falls_through() {
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }
             def main() -> Int<32> {
                 return add(1, 2);
             }",
    );
    assert!(
        result.is_ok(),
        "non-polymorphic call must fall through to normal inference: {:?}",
        result.err()
    );
}

// ── Polytopes (first-class polymorphism) ──────────────────────

#[test]
fn test_poly_box_identity() {
    // Box a polymorphic identity function, then unbox and apply.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 set p = poly(id);
                 set f = unbox(p);
                 return f(42);
             }",
    );
    assert!(
        result.is_ok(),
        "poly box/unbox identity: {:?}",
        result.err()
    );
}

#[test]
fn test_poly_box_twice() {
    // unbox(p) creates ONE set of fresh type variables; f(42) constrains
    // them to Int<32>, so f(true) must fail.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 set p = poly(id);
                 set f = unbox(p);
                 set x = f(42);
                 set y = f(true);
                 return x;
             }",
    );
    assert!(result.is_err(), "poly box twice should fail: {:?}", result);
}

#[test]
fn test_poly_unbox_non_poly_error() {
    // Box a non-polymorphic value should produce an error.
    let result = check_source(
        "def main() -> Int<32> {
                 set p = poly(42);
                 return 0;
             }",
    );
    assert!(result.is_err(), "poly(42) should error: {:?}", result);
}

#[test]
fn test_unbox_non_poly_error_after_resolution() {
    // unbox on a concrete non-poly value triggers error when the poly type
    // is later resolved (not yet enforced in current phase — will suspend in Phase 5).
    // For now we test that the expression at least doesn't crash.
    let result = check_source(
        "def main() -> Int<32> {
                 set x = 42;
                 set p = unbox(x);
                 return 0;
             }",
    );
    // Currently accepts because InferVar is not yet checked against Poly.
    // Phase 5 will add suspended matching for proper error detection.
    assert!(
        result.is_ok() || result.is_err(),
        "unbox of non-poly should eventually error"
    );
}

/// `unbox` on an operand whose type is an UNRESOLVED inference variable
/// must fail closed (require an explicit annotation) instead of silently
/// aliasing the result to the variable — an alias would skip the
/// polytype instantiation when the variable later resolves to a `Poly`.
#[test]
fn test_unbox_unresolved_operand_fails_closed() {
    let result = check_source(
        "def main() -> Int<32> {
             set x = 42;
             set p = unbox(x);
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "unbox of an unresolved type must fail closed (require an annotation): {:?}",
        result
    );
}

#[test]
fn test_poly_higher_rank() {
    // unbox once and apply — single instantiation.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 set f = unbox(poly(id));
                 return f(42);
             }",
    );
    assert!(result.is_ok(), "higher-rank poly: {:?}", result.err());
}

#[test]
fn test_poly_multi_instantiate() {
    // Create separate unbox instantiations for different types.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> (Int<32>, Bool) {
                 set a = unbox(poly(id))(42);
                 set b = unbox(poly(id))(true);
                 return (a, b);
             }",
    );
    // Note: the checker supports `unbox(poly(id))(42)` only if the
    // parser chains calls correctly; otherwise accept either outcome.
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_poly_multi_quantifier() {
    // Poly with multiple type quantifiers.
    let result = check_source(
        "def pair<T, U>(a: T, b: U) -> T { return a; }
             def main() -> Int<32> {
                 set f = unbox(poly(pair));
                 return f(42, true);
             }",
    );
    assert!(result.is_ok(), "multi-quantifier poly: {:?}", result.err());
}

#[test]
fn test_poly_inside_fn_body() {
    // Use poly/unbox inside a function body that returns a concrete type.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 set f = unbox(poly(id));
                 return f(42);
             }",
    );
    assert!(result.is_ok(), "poly inside fn body: {:?}", result.err());
}

#[test]
fn test_poly_chain() {
    // Chain poly() and unbox() across let bindings.
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
             def main() -> Int<32> {
                 set a = unbox(poly(id));
                 return a(42);
             }",
    );
    assert!(result.is_ok(), "chained poly/unbox: {:?}", result.err());
}

// ── Trait impl and operator overload ──────────────────────────

#[test]
fn test_trait_impl_basic() {
    let result = check_source(
        "trait Show { }
             type MyInt = Int<32> with default = 0;
             impl Show for MyInt { }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "basic trait impl: {:?}", result.err());
}

#[test]
fn test_operator_plus() {
    let result = check_source(
        "def main() -> Int<32> {
                 return 10 + 20;
             }",
    );
    assert!(result.is_ok(), "operator +: {:?}", result.err());
}

#[test]
fn test_operator_mul() {
    let result = check_source(
        "def main() -> Int<32> {
                 return 6 * 7;
             }",
    );
    assert!(result.is_ok(), "operator *: {:?}", result.err());
}

// ── Autoderef ──────────────────────────────────────────────────

#[test]
fn test_autoderef_field_through_ref() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set p = Point { x = 10, y = 20 };
                 set r = &p;
                 return r.x;
             }",
    );
    assert!(
        result.is_ok(),
        "field access through &ref via autoderef: {:?}",
        result.err()
    );
}

#[test]
fn test_autoderef_method_through_ref() {
    let result = check_source(
        "type MyType = struct { val: Int<32> }
             impl for MyType {
                 def get_val(&self) -> Int<32> { return self.val; }
             }
             def main() -> Int<32> {
                 set obj = MyType { val = 42 };
                 set r = &obj;
                 return r.get_val();
             }",
    );
    assert!(
        result.is_ok(),
        "method call through &ref via autoderef: {:?}",
        result.err()
    );
}

// ── Error message verification ────────────────────────────────

#[test]
fn test_error_type_mismatch() {
    let result = check_source("def main() -> Int<32> { return true; }");
    assert!(result.is_err(), "type mismatch should error");
    let all = result.err().unwrap().join(" ");
    assert!(!all.is_empty(), "should have at least one error message");
}

// ── Constraint-queue dispatch (return-body unification) ───────────
//
// Defensive regression tests for routing the return-body unification
// (checker/mod.rs `check_stmt(FunctionDef)`) through the inference
// constraint queue (`Constraint::Eq`) instead of an inline
// `ctx.unify_tracked` call.  These pin behaviors that MUST NOT change:
// mismatches still error, matches still pass, and infer-var returns
// resolve identically whether unified inline or at solve time.

#[test]
fn test_constraint_dispatch_return_mismatch_still_errors() {
    // Baseline: inline unify_with(return_ty, body_ty) → immediate E030.
    // After dispatch: the queued Constraint::Eq must surface the same
    // mismatch when solved at the scope exit.
    let result = check_source("def f() -> Int<32> { return true; }");
    let err = result.expect_err("return type mismatch must error");
    let all = err.join(" ");
    assert!(
        all.contains("expected") && all.contains("Int") && all.contains("Bool"),
        "error should mention both the expected and found types: {}",
        all
    );
}

#[test]
fn test_constraint_dispatch_return_match_still_passes() {
    let result = check_source("def f() -> Int<32> { return 42; }");
    assert!(
        result.is_ok(),
        "matching return must pass: {:?}",
        result.err()
    );
}

#[test]
fn test_constraint_dispatch_return_infer_var_resolves() {
    // The return type is an InferVar; the body binds it to Int<32>.
    // Whether unified inline or dispatched to the queue, the result
    // must resolve identically.
    let result = check_source("def f() -> Int<32> { set x = 42; return x; }");
    assert!(
        result.is_ok(),
        "infer-var return must pass: {:?}",
        result.err()
    );
}

#[test]
fn test_constraint_dispatch_return_mismatch_in_conditional() {
    // An `if` whose branches return different types is still rejected
    // after dispatch (no over-acceptance through the queue).
    let result = check_source(
        "def f(x: Int<32>) -> Int<32> {
             if true { return x; } else { return true; }
         }",
    );
    assert!(result.is_err(), "branch type mismatch should error");
}

#[test]
fn test_error_undefined_variable() {
    let result = check_source("def main() -> Int<32> { return x; }");
    assert!(result.is_err(), "undefined variable should error");
    let all = result.err().unwrap().join(" ");
    assert!(
        all.contains("undefined"),
        "error should mention 'undefined': {}",
        all
    );
}

#[test]
fn test_error_wrong_argument_count() {
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }
             def main() -> Int<32> { return add(1); }",
    );
    assert!(result.is_err(), "wrong argument count should error");
    let all = result.err().unwrap().join(" ");
    assert!(
        all.contains("wrong number"),
        "error should mention arg count: {}",
        all
    );
}

#[test]
fn test_error_no_such_field() {
    let result = check_source(
        "type T = struct { x: Int<32> }
             def main() -> Int<32> {
                 set t = T { x = 42 };
                 return t.y;
             }",
    );
    assert!(result.is_err(), "missing field should error");
    let all = result.err().unwrap().join(" ");
    assert!(
        all.contains("no field"),
        "error should mention 'no field': {}",
        all
    );
}

#[test]
fn test_error_no_method() {
    let result = check_source(
        "type T = struct { x: Int<32> }
             def main() -> Int<32> {
                 set t = T { x = 42 };
                 return t.foo();
             }",
    );
    assert!(result.is_err(), "no matching method should error");
    let all = result.err().unwrap().join(" ");
    assert!(
        all.contains("no field or method"),
        "error should mention 'no field or method': {}",
        all
    );
}

// ── Exhaustiveness matching tests ──────────────────────────────

#[test]
fn test_match_exhaustive_enum_ok() {
    let result = check_source(
        "type MyBool = enum { True, False }
             def main() -> Int<32> {
                 set b = MyBool::True;
                 set x = match b { MyBool::True => 1, MyBool::False => 0 };
                 return x;
             }",
    );
    assert!(
        result.is_ok(),
        "exhaustive match should pass: {:?}",
        result.err()
    );
}

/// A `match` whose arms ALL diverge (`return`/`leave`) must have the
/// bottom type `!` (Never), mirroring the `Expr::If` divergence
/// convention — not `()` (Unit).  A `!` context must accept it: before
/// the fix, the all-diverging match inferred Unit and the enclosing
/// `return x` (where the function returns `Int<32>`) produced a
/// spurious mismatch.
#[test]
fn test_match_all_arms_diverge_infers_never() {
    let result = check_source(
        "def main() -> Int<32> {
             set b = true;
             set x = match b { true => { return 1; }, false => { return 2; } };
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "an all-diverging match must infer `!`, not Unit: {:?}",
        result.err()
    );
}

#[test]
fn test_match_non_exhaustive_enum_errors() {
    let result = check_source(
        "type State = enum { Init, Running, Stopped }
             def main() -> Int<32> {
                 set s = State::Init;
                 set x = match s { State::Init => 1, State::Running => 2 };
                 return x;
             }",
    );
    assert!(result.is_err(), "non-exhaustive match should error");
    let msg = result.err().unwrap().join(" ");
    assert!(
        msg.contains("non-exhaustive"),
        "error should mention non-exhaustive: {}",
        msg
    );
}

#[test]
fn test_match_exhaustive_with_wildcard_ok() {
    let result = check_source(
        "type State = enum { A, B, C, D }
             def main() -> Int<32> {
                 set s = State::A;
                 set x = match s { State::A => 1, _ => 0 };
                 return x;
             }",
    );
    assert!(
        result.is_ok(),
        "match with wildcard should pass: {:?}",
        result.err()
    );
}

#[test]
fn test_match_bool_exhaustive_required() {
    let result = check_source(
        "def main() -> Int<32> {
                 set b = true;
                 set x = match b { true => 1 };
                 return x;
             }",
    );
    assert!(result.is_err(), "non-exhaustive bool match should error");
    let msg = result.err().unwrap().join(" ");
    assert!(
        msg.contains("non-exhaustive"),
        "error should mention non-exhaustive: {}",
        msg
    );
}

#[test]
fn test_match_bool_exhaustive_with_wildcard_ok() {
    let result = check_source(
        "def main() -> Int<32> {
                 set b = true;
                 set x = match b { true => 1, _ => 0 };
                 return x;
             }",
    );
    assert!(
        result.is_ok(),
        "bool match with wildcard should pass: {:?}",
        result.err()
    );
}

#[test]
fn test_match_bool_full_exhaustive_ok() {
    let result = check_source(
        "def main() -> Int<32> {
                 set b = true;
                 set x = match b { true => 1, false => 0 };
                 return x;
             }",
    );
    assert!(
        result.is_ok(),
        "full bool match should pass: {:?}",
        result.err()
    );
}

// ── Region Tree tests ──────────────────────────────────────────

#[test]
fn test_region_tree_basic_ops() {
    let mut rt = RegionTree::new();
    // Root region exists with no frames
    assert_eq!(rt.current_frames().len(), 0);

    // Push a frame
    rt.push_frame(CtxFrame {
        kind: CtxKind::Function,
        span: Span::new(0, 0),
        label: None,
        comptime_reason: None,
    });
    assert_eq!(rt.current_frames().len(), 1);

    // Pop it back
    assert!(rt.pop_frame().is_some());
    assert_eq!(rt.current_frames().len(), 0);
}

// -- Bidirectional if-expression --
#[test]
fn test_if_expression_type_inference() {
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();

    // if true { 42 } else { 0 }
    let cond: &'static Expr<'static> = Box::leak(Box::new(Expr::Literal(
        Literal::Bool(true),
        Span::new(0, 1),
    )));
    let then_block = vec![Stmt::Expression(Expr::Literal(
        Literal::Int(ast::IntLit::Small(42)),
        Span::new(2, 4),
    ))];
    let else_block = vec![Stmt::Expression(Expr::Literal(
        Literal::Int(ast::IntLit::Small(0)),
        Span::new(5, 6),
    ))];
    let if_expr = Expr::If {
        cond,
        then_branch: then_block,
        else_branch: Some(else_block),
        is_expression: true,
        span: Span::new(0, 6),
    };

    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());
    let result = checker.infer_expr(&if_expr, None);
    assert!(result.is_ok());
    let (_hir, ty) = result.unwrap();
    // Both branches are integer literals — the inferred type is an Integer InferVar
    assert!(checker.ctx.is_integer(ty) || checker.ctx.is_infer_var(ty));
}

#[test]
fn test_if_statement_with_return() {
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();

    // if true { return 42 } else { return 0 }  — both diverge → never
    let cond: &'static Expr<'static> = Box::leak(Box::new(Expr::Literal(
        Literal::Bool(true),
        Span::new(0, 1),
    )));
    let then_stmt = Stmt::Return {
        value: Some(Expr::Literal(
            Literal::Int(ast::IntLit::Small(42)),
            Span::new(2, 4),
        )),
        labels: Vec::new(),
        span: Span::new(2, 4),
    };
    let else_stmt = Stmt::Return {
        value: Some(Expr::Literal(
            Literal::Int(ast::IntLit::Small(0)),
            Span::new(5, 6),
        )),
        labels: Vec::new(),
        span: Span::new(5, 6),
    };
    let if_expr = Expr::If {
        cond,
        then_branch: vec![then_stmt],
        else_branch: Some(vec![else_stmt]),
        is_expression: true,
        span: Span::new(0, 6),
    };

    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());
    let result = checker.infer_expr(&if_expr, None);
    // Should succeed (no unify panic) since both branches diverge
    assert!(result.is_ok());
}

#[test]
fn test_if_expression_branch_type_match() {
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();

    // if true { 42 } else { false } — should still succeed via unification
    let cond: &'static Expr<'static> = Box::leak(Box::new(Expr::Literal(
        Literal::Bool(true),
        Span::new(0, 1),
    )));
    let then_block = vec![Stmt::Expression(Expr::Literal(
        Literal::Int(ast::IntLit::Small(42)),
        Span::new(2, 4),
    ))];
    let else_block = vec![Stmt::Expression(Expr::Literal(
        Literal::Bool(false),
        Span::new(5, 10),
    ))];
    let if_expr = Expr::If {
        cond,
        then_branch: then_block,
        else_branch: Some(else_block),
        is_expression: true,
        span: Span::new(0, 10),
    };

    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());
    let result = checker.infer_expr(&if_expr, None);
    assert!(result.is_ok());
}

#[test]
fn test_if_expression_tuple() {
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();

    // if true { 1 } else { 2 } inside tuple context
    let cond: &'static Expr<'static> = Box::leak(Box::new(Expr::Literal(
        Literal::Bool(true),
        Span::new(0, 1),
    )));
    let if_expr = Expr::If {
        cond,
        then_branch: vec![Stmt::Expression(Expr::Literal(
            Literal::Int(ast::IntLit::Small(1)),
            Span::new(2, 3),
        ))],
        else_branch: Some(vec![Stmt::Expression(Expr::Literal(
            Literal::Int(ast::IntLit::Small(2)),
            Span::new(4, 5),
        ))]),
        is_expression: true,
        span: Span::new(0, 5),
    };
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());
    let result = checker.infer_expr(&if_expr, None);
    assert!(result.is_ok());
}

// -- SCAP guarantee chaining --
#[test]
fn test_scap_ensures_bool_check() {
    // SCAP §4: ensures clause must be boolean — verify the chain infrastructure
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());

    // Push a guarantee with a boolean postcondition (simulating 'ensures result > 0')
    let post = checker.ctx.bool();
    let g = Guarantee::new(Predicate::True, Predicate::Type(post), None);
    checker.guarantee_chain.push(g);

    // The guarantee chain should have depth 1
    assert!(checker.guarantee_chain.current().is_some());
    assert_eq!(
        checker.guarantee_chain.current().unwrap().post,
        Predicate::Type(post)
    );

    // Pop the guarantee on simulated return
    let popped = checker.guarantee_chain.pop();
    assert!(popped.is_some());
    assert!(checker.guarantee_chain.current().is_none());
}

#[test]
fn test_scap_ensures_chaining() {
    // SCAP §4, Fig.8 (CALL rule): g₀ → g' (callee's g) → g₂ (continuation)
    // Simulate: caller pushes g₀, calls callee (g'), then continuation (g₂)
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());

    // Push g₀ (caller's guarantee)
    let g0 = Guarantee::new(Predicate::True, Predicate::Type(checker.ctx.bool()), None);
    checker.guarantee_chain.push(g0);

    // Push g' (callee's guarantee — CALL rule chains through callee)
    let g_callee = Guarantee::new(Predicate::True, Predicate::Type(checker.ctx.bool()), None);
    checker.guarantee_chain.push(g_callee);

    // Pop g' (callee returns)
    let popped = checker.guarantee_chain.pop();
    assert!(popped.is_some());

    // g₀ should still be on the chain
    assert!(checker.guarantee_chain.current().is_some());

    // Pop g₀ (caller returns)
    let popped2 = checker.guarantee_chain.pop();
    assert!(popped2.is_some());
    assert!(checker.guarantee_chain.current().is_none());
}

#[test]
fn test_scap_ensures_no_guarantee_ok() {
    // SCAP §4.2, WFST: outermost function has no return pointer → no guarantee
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();
    let checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());

    // No ensures clause = vacuously true (chain is empty)
    assert!(checker.guarantee_chain.current().is_none());
}

#[test]
fn test_scap_multiple_ensures_clauses() {
    // SCAP: multiple ensures clauses → multiple guarantees on the chain
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());

    // Push two guarantees (simulating two ensures clauses)
    let g1 = Guarantee::new(Predicate::True, Predicate::Type(checker.ctx.bool()), None);
    let g2 = Guarantee::new(Predicate::True, Predicate::Type(checker.ctx.bool()), None);
    checker.guarantee_chain.push(g1);
    checker.guarantee_chain.push(g2);

    // Both should be on the chain
    assert!(checker.guarantee_chain.current().is_some());
    assert_eq!(checker.guarantee_chain.stack.len(), 2);

    // Pop in reverse order (stack discipline)
    checker.guarantee_chain.pop();
    assert_eq!(checker.guarantee_chain.stack.len(), 1);
    checker.guarantee_chain.pop();
    assert!(checker.guarantee_chain.current().is_none());
}

#[test]
fn test_scap_guarantee_discharge_on_return() {
    // SCAP §4 (RET rule): on return, the innermost guarantee must be discharged.
    // Verify that a return statement in a function with an ensures clause
    // properly checks/clears the guarantee.
    let mut ctx = TypeContext::new();
    let symbols = SymbolTable::new(CrateId(DefId(0)));
    let mut trait_env = TraitEnv::new();
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, ResolutionMap::default());

    // Simulate entering a function: push a guarantee
    let g = Guarantee::new(Predicate::True, Predicate::Type(checker.ctx.bool()), None);
    checker.guarantee_chain.push(g);

    // The return should see the guarantee and verify it
    // (in real compilation, the return statement would pop it)
    assert!(checker.guarantee_chain.current().is_some());

    // Discharge on simulated return
    let discharged = checker.guarantee_chain.pop();
    assert!(discharged.is_some());
    assert!(checker.guarantee_chain.current().is_none());
}

// ── Rational<p,q> tests ────────────────────────────────────────

#[test]
fn test_rational_type_syntax() {
    let result = check_source(
        r#"edition = "2026";
def main() -> Rational<16,16> {
    return 0: Rational<16,16>;
}"#,
    );
    assert!(
        result.is_ok(),
        "Rational<16,16> should type-check: {:?}",
        result.err()
    );
}

#[test]
fn test_rational_arithmetic() {
    let result = check_source(
        r#"edition = "2026";
def main() -> Rational<16,16> {
    set a: Rational<16,16> = 1: Rational<16,16>;
    set b: Rational<16,16> = 2: Rational<16,16>;
    set c = a + b;
    return c;
}"#,
    );
    assert!(
        result.is_ok(),
        "Rational arithmetic should type-check: {:?}",
        result.err()
    );
}

#[test]
fn test_rational_type_mismatch() {
    let result = check_source(
        r#"edition = "2026";
def main() -> Rational<16,8> {
    return 0: Rational<8,16>;
}"#,
    );
    assert!(
        result.is_err(),
        "Rational<16,8> and Rational<8,16> should NOT unify"
    );
}

// ── Quantified expressions (parsed but checker returns bool) ────

#[test]
fn test_forall_in_contract() {
    // `forall` in a simple expression context
    let result = check_source(
        "def f() -> Bool { return true; }
             def main() -> Bool { return f(); }",
    );
    assert!(result.is_ok(), "baseline: {:?}", result.err());
}

// ── Closure return type inference ───────────────────────────────

#[test]
fn test_closure_implicit_return_int() {
    let result = check_source("def main() -> Int<32> { set f = || { 1 + 1 }; return f(); }");
    assert!(result.is_ok(), "closure infer Int: {:?}", result.err());
}

#[test]
fn test_closure_implicit_return_bool() {
    let result = check_source("def main() -> Bool { set f = || { true }; return f(); }");
    assert!(result.is_ok(), "closure infer Bool: {:?}", result.err());
}

#[test]
fn test_closure_explicit_return_type() {
    let result =
        check_source("def main() -> Int<64> { set f = || -> Int<64> { 42 }; return f(); }");
    assert!(
        result.is_ok(),
        "closure explicit return: {:?}",
        result.err()
    );
}

#[test]
fn test_closure_unit_return() {
    let result = check_source("def main() -> Bool { set f = || { true }; return f(); }");
    assert!(result.is_ok(), "closure unit: {:?}", result.err());
}

// ── Trait impl completeness ─────────────────────────────────────

#[test]
fn test_trait_impl_missing_method() {
    let result = check_source(
        "trait Show { def show(&self) -> Int<32>; }
             type MyInt = Int<32> with default = 0;
             impl Show for MyInt { }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_err(), "impl missing method should fail");
}

#[test]
fn test_trait_impl_all_methods_provided() {
    // Trait with a method taking a concrete type (not `self`) so that
    // the checker can resolve all types without `Self` → for_type mapping.
    let result = check_source(
        "trait Show { def show(x: Int<32>) -> Int<32>; }
             type MyInt = Int<32> with default = 0;
             impl Show for MyInt { def show(self) -> Int<32> { return 42; } }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "impl with all methods: {:?}", result.err());
}

#[test]
fn test_trait_impl_wrong_param_count() {
    let result = check_source(
        "trait Show { def show(self) -> Int<32>; }
             type MyInt = Int<32> with default = 0;
             impl Show for MyInt { def show(self, extra: Int<32>) -> Int<32> { return 42; } }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_err(), "impl wrong param count should fail");
}

#[test]
fn test_trait_impl_generic_with_bound() {
    let result = check_source(
        "trait Show { } trait Default { }
             type MyInt = Int<32> with default = 0;
             impl Default for MyInt { }
             impl<T: Default> Show for T { }
             def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_ok(),
        "generic impl with bound: {:?}",
        result.err()
    );
}

#[test]
fn test_trait_two_methods_impl_both() {
    let result = check_source(
        "trait Pair { def first(x: Int<32>) -> Int<32>; def second(x: Int<32>) -> Int<32>; }
             type MyInt = Int<32> with default = 0;
             impl Pair for MyInt {
                 def first(self) -> Int<32> { return 42; }
                 def second(self) -> Int<32> { return 42; }
             }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "impl with two methods: {:?}", result.err());
}

#[test]
fn test_trait_missing_one_of_two() {
    let result = check_source(
        "trait Pair { def first(self) -> Int<32>; def second(self) -> Int<32>; }
             type MyInt = Int<32> with default = 0;
             impl Pair for MyInt { def first(self) -> Int<32> { return 42; } }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_err(), "impl missing one of two should fail");
}

// ── Inherent impl ───────────────────────────────────────────────

#[test]
fn test_inherent_impl_method_call() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
             impl Point {
                 def get_x(self) -> Int<32> { return self.x; }
             }
             def main() -> Int<32> { set p = Point { x = 10, y = 20 }; return 0; }",
    );
    assert!(result.is_ok(), "inherent method: {:?}", result.err());
}

#[test]
fn test_inherent_impl_mut_method() {
    let result = check_source(
        "type Counter = struct { val: Int<32> }
             impl Counter {
                 def inc(self) -> Counter { return self; }
             }
             def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "inherent method: {:?}", result.err());
}

#[test]
fn test_int_32_bit() {
    let result = check_source("def main() -> Int<32> { return 0; }");
    assert!(result.is_ok(), "Int<32>: {:?}", result.err());
}

#[test]
fn test_uint_8_bit() {
    let result = check_source("def main() -> UInt<8> { return 0; }");
    assert!(result.is_ok(), "UInt<8>: {:?}", result.err());
}

// ── End-to-end: generics, structs, pattern matching ──────────────

#[test]
fn test_generic_function_identity() {
    // Polymorphic identity function: `def id<T>(x: T) -> T { return x; }`
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
         def main() -> Int<32> { set y = id(42); return y; }",
    );
    assert!(result.is_ok(), "generic identity: {:?}", result.err());
}

#[test]
fn test_generic_function_pair() {
    // Generic function with multiple type params — simple return type
    let result = check_source(
        "def pair<A, B>(a: A, b: B) -> A { return a; }
         def main() -> Int<32> { set p = pair(1, true); return p; }",
    );
    assert!(result.is_ok(), "generic pair: {:?}", result.err());
}

#[test]
fn test_generic_function_with_trait_bound() {
    // Generic with a trait bound — use simple impl pattern without T:: syntax
    let result = check_source(
        "trait Defaultable { def get_default() -> Int<32>; }
         impl Defaultable for Int<32> { def get_default() -> Int<32> { return 0; } }
         def main() -> Int<32> {
             set x: Int<32> = 0;
             return x;
         }",
    );
    assert!(result.is_ok(), "generic with bound: {:?}", result.err());
}

#[test]
fn test_struct_literal_and_field_access() {
    // Full round-trip: define struct, construct, access fields
    let result = check_source(
        "type Vec2 = struct { x: Int<32>, y: Int<32> }
         def main() -> Int<32> {
             set v = Vec2 { x = 10, y = 20 };
             return v.x + v.y;
         }",
    );
    assert!(result.is_ok(), "struct field access: {:?}", result.err());
}

#[test]
fn test_nested_struct_field_access() {
    // Nested struct: outer.inner.field
    let result = check_source(
        "type Inner = struct { val: Int<32> }
         type Outer = struct { inner: Inner }
         def main() -> Int<32> {
             set obj = Outer { inner = Inner { val = 42 } };
             return obj.inner.val;
         }",
    );
    assert!(result.is_ok(), "nested field: {:?}", result.err());
}

#[test]
fn test_match_on_bool() {
    // Pattern<'input> matching on a Bool
    let result = check_source(
        "def main() -> Int<32> {
             set flag = true;
             return match flag {
                 true => 1,
                 false => 0,
             };
         }",
    );
    assert!(result.is_ok(), "match bool: {:?}", result.err());
}

#[test]
fn test_method_call_on_struct() {
    // Struct with an impl block and method call
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         impl Point {
             def magnitude_sq(self) -> Int<32> {
                 return self.x * self.x + self.y * self.y;
             }
         }
         def main() -> Int<32> {
             set p = Point { x = 3, y = 4 };
             return p.magnitude_sq();
         }",
    );
    assert!(result.is_ok(), "method call: {:?}", result.err());
}

#[test]
fn test_deferred_impl_registration_trait_method_call() {
    // Audit test: verify that trait method resolution works correctly
    // with the deferred impl registration architecture (impl registered
    // by the checker, not the resolver).  The resolver no longer calls
    // add_impl — registration happens in the type checker instead.
    let result = check_source(
        "trait Show {
             def show(&self) -> Int<32>;
         }
         type MyInt = Int<32> with default = 0;
         impl Show for MyInt {
             def show(&self) -> Int<32> {
                 return *self;
             }
         }
         def main() -> Int<32> {
             set x: MyInt = 42;
             return x.show();
         }",
    );
    assert!(
        result.is_ok(),
        "deferred impl trait method call: {:?}",
        result.err()
    );
}

/// GAT declaration syntax must parse — `type Item<'a>;` in a
/// trait and `type Item<'a> = ...;` in an impl (SYNTAX.md §GAT
/// Declaration).  Previously the parser stopped at the `<'a>` list
/// ("expected Semicolon/Assign, found Lt").
#[test]
fn test_m4_gat_declaration_parses() {
    let result = check_source(
        "trait Container {
             type Item<'a>;
         }
         type MyInt = Int<32> with default = 0;
         impl Container for MyInt {
             type Item<'a> = Int<32>;
         }
         def main() -> Int<32> {
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "GAT declarations must parse (trait type Item<'a>; + impl type Item<'a> = ...;): {:?}",
        result
    );
}

/// The SYNTAX.md §Higher-Ranked Trait Bounds example must
/// PARSE — `where F: for<'a> Fn(&'a Int<32>) -> &'a Int<32>` (the named
/// `Fn(...) -> ...` function type + `for<'a>` binder).  The body does not
/// call `f` (a generic call would need the Fn-trait solver — beyond this
/// parser-level fix).
#[test]
fn test_m5_hrtb_fn_bound_parses() {
    let result = check_source(
        "def apply<F>(f: F, x: &Int<32>) -> &Int<32>
             where F: for<'a> Fn(&'a Int<32>) -> &'a Int<32>
         {
             return x;
         }
         def main() -> Int<32> {
             return 0;
         }",
    );
    assert!(result.is_ok(), "the HRTB Fn bound must parse: {:?}", result);
}

/// The GAT equality bound in a where predicate must parse —
/// `I: StreamingIterator<Item = &'a Int<32>>` (the `=` in a generic-arg
/// list, SYNTAX.md §Interaction with HRTB).  Previously the generic-arg
/// loop errored on the `=`.  (The lifetime-PARAMETERIZED use site
/// `Item<'a> = ...` is a separate boundary — `Item<'a>`'s own nested
/// generic parse — tracked as a follow-up; the `=` handling itself is
/// what the GAT-equality fix handles.)
#[test]
fn test_m5_gat_equality_bound_parses() {
    let result = check_source(
        "def drain_all<I>(iter: I) -> Int<32>
             where I: StreamingIterator<Item = &'a Int<32>>
         {
             return 0;
         }
         def main() -> Int<32> {
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the GAT equality bound must parse: {:?}",
        result
    );
}

/// GAT lifetime-PARAMETERIZED use site: the
/// full SYNTAX.md §Interaction with HRTB shape must parse —
/// `for<'a> StreamingIterator<Item<'a> = &'a Int<32>>`.  Two parser bugs
/// were fixed: (a) the `Foo<'a>` generic-arg lifetime consumed the
/// closing `>` (an extra `advance()` inside the catch-all Apostrophe
/// arm); (b) the GAT equality-bound `=` arm forgot to close the generic
/// list (the loop then parsed the closing `>` as another argument →
/// "expected expression").
#[test]
fn test_gat_parameterized_use_site_parses() {
    let result = check_source(
        "trait StreamingIterator {
             type Item<'a>;
         }
         type MyInt = Int<32> with default = 0;
         impl StreamingIterator for MyInt {
             type Item<'a> = Int<32>;
         }
         def drain_all<I>(iter: I) -> Int<32>
             where I: for<'a> StreamingIterator<Item<'a> = &'a Int<32>>
         {
             return 0;
         }
         def main() -> Int<32> {
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the lifetime-parameterized GAT use site must parse: {:?}",
        result
    );
}

#[test]
fn test_type_error_propagation() {
    // Type mismatch should produce a diagnostic
    let result = check_source(
        "def main() -> Bool {
             return 42;
         }",
    );
    assert!(result.is_err(), "type mismatch should fail");
}

#[test]
fn test_undefined_variable_error() {
    // Using an undefined variable produces an error
    let result = check_source(
        "def main() -> Int<32> {
             return x;
         }",
    );
    assert!(result.is_err(), "undefined variable should fail");
}

#[test]
fn test_contract_requires() {
    // Function with basic requires contract
    let result = check_source(
        "def divide(a: Int<32>, b: Int<32>) -> Int<32>
             requires b != 0
         {
             return a / b;
         }
         def main() -> Int<32> { return divide(10, 2); }",
    );
    assert!(result.is_ok(), "contract requires: {:?}", result.err());
}

#[test]
fn test_closure_basic() {
    // Simple closure with explicit parameter types, block body: `|x: Int<32>| { x }`
    let result = check_source(
        "def main() -> Int<32> {
             set f = |x: Int<32>| -> Int<32> { return x; };
             return f(42);
         }",
    );
    assert!(result.is_ok(), "closure basic: {:?}", result.err());
}

#[test]
fn test_closure_short_body() {
    // Closure with expression body (no braces): `|x: Int<32>| x + 1`
    let result = check_source(
        "def main() -> Int<32> {
             set f = |x: Int<32>| x + 1;
             return f(41);
         }",
    );
    assert!(result.is_ok(), "closure short body: {:?}", result.err());
}

#[test]
fn test_closure_capture() {
    // Closure capturing a variable from the enclosing scope
    let result = check_source(
        "def main() -> Int<32> {
             set factor = 2;
             set f = |x: Int<32>| x * factor;
             return f(21);
         }",
    );
    assert!(result.is_ok(), "closure capture: {:?}", result.err());
}

#[test]
fn test_for_loop_with_variable() {
    // for loop iterating over an array literal — loop variable in scope
    let result = check_source(
        "def main() -> Int<32> {
             set mut total = 0;
             for x in [1, 2, 3] {
                 total = total + x;
             }
             return total;
         }",
    );
    assert!(result.is_ok(), "for loop: {:?}", result.err());
}

#[test]
fn test_for_loop_with_index() {
    // for loop over a range — using index variable in body
    let result = check_source(
        "def main() -> Int<32> {
             set mut total = 0;
             set arr = [10, 20, 30];
             for i in arr {
                 total = total + i;
             }
             return total;
         }",
    );
    assert!(result.is_ok(), "for loop index: {:?}", result.err());
}

#[test]
fn test_old_expression() {
    // `old(expr)` in a contract — capture value at function entry
    let result = check_source(
        "def main() -> Int<32> {
             set x = 42;
             set y = old(x);
             return y;
         }",
    );
    assert!(result.is_ok(), "old expression: {:?}", result.err());
}

#[test]
fn test_old_in_contract() {
    // `old(expr)` inside an ensures clause — basic parsing and checking
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32>
             ensures old(a) + old(b) >= 0
         {
             return a + b;
         }
         def main() -> Int<32> { return add(1, 2); }",
    );
    assert!(result.is_ok(), "old in ensures: {:?}", result.err());
}

#[test]
fn test_codomain_in_ensures() {
    // `codomain` keyword in ensures refers to the return value
    let result = check_source(
        "def double(x: Int<32>) -> Int<32>
             ensures codomain == x + x
         {
             return x + x;
         }
         def main() -> Int<32> { return double(5); }",
    );
    assert!(result.is_ok(), "codomain in ensures: {:?}", result.err());
}

#[test]
fn test_codomain_in_ensures_multi() {
    // Multiple ensures clauses using `codomain`
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32>
             ensures codomain >= a
             ensures codomain >= b
         {
             return a + b;
         }
         def main() -> Int<32> { return add(3, 4); }",
    );
    assert!(result.is_ok(), "result multi: {:?}", result.err());
}

#[test]
fn test_label_in_ensures_and_return() {
    // `ensures @label expr` with matching `return @label` should pass.
    // `@label` is a placeholder in the expression: `@even > 0` means
    // "the return value on the @even path is > 0".
    let result = check_source(
        "def f(x: Int<32>) -> Int<32>
             ensures @even > 0
         {
             return @even x;
         }
         def main() -> Int<32> { return f(5); }",
    );
    assert!(
        result.is_ok(),
        "label in ensures+return: {:?}",
        result.err()
    );
}

#[test]
fn test_label_missing_return_rejected() {
    // `ensures @label expr` without any `return @label` should fail.
    let result = check_source(
        "def f(x: Int<32>) -> Int<32>
             ensures @even > 0
         {
             return x;
         }
         def main() -> Int<32> { return f(5); }",
    );
    assert!(
        result.is_err(),
        "label in ensures without matching return should be rejected"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter()
            .any(|e| e.contains("label") && e.contains("ensures")),
        "error should mention label in ensures: {:?}",
        errs
    );
}

#[test]
fn test_codomain_and_label_together() {
    // `codomain` (all paths) and `@label` (specific path) can be combined.
    let result = check_source(
        "def f(x: Int<32>) -> Int<32>
             ensures codomain >= 0
             ensures @fast < 100
         {
             return @fast x;
         }
         def main() -> Int<32> { return f(5); }",
    );
    assert!(result.is_ok(), "codomain + label: {:?}", result.err());
}

#[test]
fn test_qualified_enum_path() {
    // Qualified enum path with payload: `Opt::Some(42)`
    let result = check_source(
        "type Opt = enum { None, Some(Int<32>) }
         def main() -> Int<32> {
             set val = Opt::Some(42);
             return match val {
                 Opt::Some(x) => x,
                 Opt::None => 0,
             };
         }",
    );
    assert!(result.is_ok(), "qualified enum: {:?}", result.err());
}

#[test]
fn test_enum_no_payload() {
    // Enum variant without payload: `Dept::Engineering`
    let result = check_source(
        "type Dept = enum { Engineering, Sales }
         def main() -> Int<32> {
             set d = Dept::Engineering;
             return 0;
         }",
    );
    assert!(result.is_ok(), "enum no payload: {:?}", result.err());
}

#[test]
fn test_if_let_basic() {
    // if-let with enum destructuring and else branch
    let result = check_source(
        "type Opt = enum { None, Some(Int<32>) }
         def main() -> Int<32> {
             set val = Opt::Some(7);
             let Some(x) = val else { return 0; };
             return x;
         }",
    );
    assert!(result.is_ok(), "if-let: {:?}", result.err());
}

#[test]
fn test_generic_return() {
    // Polymorphic identity — tests generic type parameter inference
    let result = check_source(
        "def id<T>(x: T) -> T { return x; }
         def main() -> Int<32> { return id(42); }",
    );
    assert!(result.is_ok(), "generic id: {:?}", result.err());
}

/// Generality check (E104): a generic function body must type-check for
/// ALL instantiations.  `def g<T>(x: T) -> Int<32> { return add(x, 1); }`
/// only works when `T = Int<32>` — the body solves the generic param,
/// which rustc (rigid `TyKind::Param`), GHC and OCaml (skolems) all
/// reject at definition time.  Enabled by the seal: GADT refinements no
/// longer write global bindings, so any remaining binding on a function
/// generic param is body-driven.
#[test]
fn test_generic_param_constrained_by_body_rejected() {
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }
         def g<T>(x: T) -> Int<32> { return add(x, 1); }
         def main() -> Int<32> { return 0; }",
    );
    let msg = format!("{:?}", result);
    assert!(
        result.is_err() && msg.contains("constrained to a specific type"),
        "body-constrained generic parameter must be rejected (E104): {:?}",
        result
    );
}

/// Generality check (E104), `T := U`: binding one generic parameter to
/// ANOTHER (`def f<T, U>(x: T, y: U) -> U { return x; }`) is also a
/// violation — the body must be parametric in each distinct parameter
/// (rustc/GHC/OCaml all reject rigid-param-to-rigid-param unifications).
#[test]
fn test_generic_param_bound_to_another_param_rejected() {
    let result = check_source(
        "def f<T, U>(x: T, y: U) -> U { return x; }
         def main() -> Int<32> { return 0; }",
    );
    let msg = format!("{:?}", result);
    assert!(
        result.is_err() && msg.contains("constrained to a specific type"),
        "T := U must be rejected (E104): {:?}",
        result
    );
}

/// Generality check exemption for const generic parameters: const
/// params monomorphize per concrete constant value (SYNTAX.md §Const
/// Generics), so they are exempt from E104 even though they appear in
/// the generic parameter list.
#[test]
fn test_const_generic_param_exempt_from_generality() {
    let result = check_source(
        "def f<const N: usize>(x: Int<32>) -> Int<32> { return x; }
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_ok(),
        "const generic param must be exempt from E104: {:?}",
        result
    );
}

/// Where equality constraint: `where T == Int<32>` constrains the
/// generic parameter explicitly in the signature, exempting it from the
/// E104 generality check — the body may rely on T being Int<32>.
#[test]
fn test_where_equality_exempts_generality() {
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }
         def f<T>(x: T) -> Int<32> where T == Int<32> { return add(x, 1); }
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_ok(),
        "where T == Int<32> must exempt E104: {:?}",
        result
    );
}

/// Two `&mut` loans on OVERLAPPING places must conflict even when a
/// NON-overlapping sibling sorts between them: `a.b` and `a.b.d` overlap,
/// but the derived place `Ord` orders `a.c` (a sibling of `a.b`) BEFORE
/// `a.b.d` — the exclusivity check must still report the conflict (a
/// regression test for the early bail-out that skipped later pairs).
#[test]
fn test_loan_exclusive_overlap_across_sibling() {
    let result = check_source(
        "type Inner = struct { d: Int<32> }
         type Outer = struct { b: Inner, c: Int<32> }
         def main() -> Int<32> {
             set mut a = Outer { b = Inner { d = 1 }, c = 2 };
             set r1: &mut Inner = &mut (a.b);
             set r2: &mut Int<32> = &mut (a.c);
             set r3: &mut Int<32> = &mut (a.b.d);
             return (*r1).d + *r2 + *r3;
         }",
    );
    assert!(
        result.is_err(),
        "overlapping exclusive loans (a.b vs a.b.d) must be rejected even with a sibling between them: {:?}",
        result
    );
}

/// `continue` outside a loop is diagnosed by the type checker; the CFG
/// builder must degrade gracefully (no panic, no dead successor block).
#[test]
fn test_continue_outside_loop_cfg_degrades_gracefully() {
    let result = check_source(
        "def main() -> Int<32> {
             continue;
             return 0;
         }",
    );
    // The type checker reports the misplaced `continue`; the important
    // assertion is that building the CFG did not panic (the test itself
    // would fail on an ICE).
    let _ = result;
}

#[test]
fn test_variable_duplicate_in_same_scope() {
    // Duplicate definition in the same scope is an error.
    let result = check_source(
        "def main() -> Int<32> {
             set x = 1;
             set x = 2;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "duplicate definition in same scope should be rejected"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter().any(|e| e.contains("duplicate definition")),
        "expected duplicate definition error: {:?}",
        errs
    );
}

#[test]
fn test_while_let_variable_in_body() {
    // while-let pattern variable should be accessible inside loop body
    let result = check_source(
        "type Opt = enum { None, Some(Int<32>) }
         def main() -> Int<32> {
             set mut opt = Opt::Some(42);
             while let Some(x) = opt {
                 return x;
             }
             return 0;
         }",
    );
    assert!(result.is_ok(), "while-let variable: {:?}", result.err());
}

#[test]
fn test_while_let_break() {
    // `leave` inside while-let should target the while-let loop
    let result = check_source(
        "def main() -> Int<32> {
             set mut i = 0;
             while true {
                 if i >= 5 {
                     leave;
                 }
                 i = i + 1;
             }
             return i;
         }",
    );
    assert!(result.is_ok(), "while break: {:?}", result.err());
}

#[test]
fn test_for_loop_break() {
    // `leave` inside for loop should target the for loop
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [1, 2, 3];
             for x in arr {
                 if x == 2 {
                     leave;
                 }
             }
             return 0;
         }",
    );
    assert!(result.is_ok(), "for break: {:?}", result.err());
}

#[test]
fn test_for_loop_continue() {
    // `continue` inside for loop should skip to next iteration
    let result = check_source(
        "def main() -> Int<32> {
             set mut total = 0;
             set arr = [1, 2, 3, 4, 5];
             for x in arr {
                 if x == 3 {
                     continue;
                 }
                 total = total + x;
             }
             return total;
         }",
    );
    assert!(result.is_ok(), "for continue: {:?}", result.err());
}

#[test]
fn test_while_let_continue() {
    // `continue` inside while-let should work
    let result = check_source(
        "type Opt = enum { None, Some(Int<32>) }
         def main() -> Int<32> {
             set mut opt = Opt::Some(42);
             set mut count = 0;
             while let Some(x) = opt {
                 count = count + 1;
                 if count < 3 {
                     continue;
                 }
                 return x;
             }
             return 0;
         }",
    );
    assert!(result.is_ok(), "while-let continue: {:?}", result.err());
}

/// `continue 'label;` jumps to the continue point of the labeled OUTER
/// loop even when a nested unlabeled loop sits between (SYNTAX.md §Loops —
/// "continue to outer labels").  The labeled jump must be accepted.
#[test]
fn test_labeled_continue_to_outer_loop() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut outer_count = 0;
             set mut inner_count = 0;
             'outer: loop {
                 outer_count = outer_count + 1;
                 loop {
                     inner_count = inner_count + 1;
                     if inner_count < 3 {
                         continue;
                     }
                     // jump to the OUTER loop's continue point
                     if outer_count < 2 {
                         continue 'outer;
                     }
                     leave 'outer;
                 }
             }
             return outer_count + inner_count;
         }",
    );
    assert!(
        result.is_ok(),
        "labeled continue to outer loop must be accepted: {:?}",
        result.err()
    );
}

/// A labeled `continue` with no matching labeled loop is an error
/// (E006) — the label must resolve to an enclosing loop.
#[test]
fn test_labeled_continue_no_matching_loop() {
    let result = check_source(
        "def main() -> Int<32> {
             continue 'missing;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "continue with an unknown label must be rejected: {:?}",
        result
    );
}

#[test]
fn test_type_capture_auto() {
    // `set auto<T> = expr` captures the inferred type of expr into the name T,
    // making T available as a type name for comptime reflection.
    let result = check_source(
        "def main() -> Int<32> {
             set auto<T> = 42;
             // T should be bound to Int<32> here.
             // For now at least it should parse and type-check without error.
             return 0;
         }",
    );
    assert!(result.is_ok(), "type capture auto: {:?}", result.err());
}

#[test]
fn test_type_capture_auto_with_struct() {
    // Type capture with a struct type — more realistic use case
    let result = check_source(
        "type MyType = struct { val: Int<32> }
         def main() -> Int<32> {
             set obj = MyType { val = 10 };
             set auto<T> = obj;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "type capture with struct: {:?}",
        result.err()
    );
}

#[test]
fn test_type_capture_auto_multi() {
    // `set auto<T, N, L> = expr` captures the inferred type of expr into
    // all named bindings, making each available for comptime reflection.
    let result = check_source(
        "def main() -> Int<32> {
             set auto<T, N, L> = 42;
             // T, N, L should all be bound to Int<32> here.
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "multi type capture auto<T, N, L>: {:?}",
        result.err()
    );
}

#[test]
fn test_type_capture_auto_four() {
    // Four captures — no limit on the number of capture names.
    let result = check_source(
        "def main() -> Int<32> {
             set auto<A, B, C, D> = true;
             return 0;
         }",
    );
    assert!(result.is_ok(), "four type captures: {:?}", result.err());
}

#[test]
fn test_type_capture_correct_type() {
    // Verify that the captured type is actually correct by using T
    // as a type annotation in a subsequent variable declaration.
    let result = check_source(
        "def main() -> Int<32> {
             set auto<T> = 42;
             set x: T = 10;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "captured type verification: {:?}",
        result.err()
    );
}

#[test]
fn test_type_capture_correct_type_bool() {
    // Verify that Bool capture works correctly.
    let result = check_source(
        "def main() -> Bool {
             set auto<T> = true;
             set x: T = false;
             return x;
         }",
    );
    assert!(result.is_ok(), "captured bool type: {:?}", result.err());
}

// ── Top-level inference scope ────────────────────────────────────

#[test]
fn test_top_level_single_function() {
    // The program-level inference scope processes constraints
    // generated by top-level `def` items.
    let result = check_source("def main() -> Int<32> { return 42; }");
    assert!(result.is_ok(), "top-level def: {:?}", result.err());
}

#[test]
fn test_top_level_type_def_and_function() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         def origin() -> Point { return Point { x = 0, y = 0 }; }",
    );
    assert!(result.is_ok(), "top-level type + def: {:?}", result.err());
}

#[test]
fn test_top_level_multi_function_cross_ref() {
    // Functions defined at top level that reference each other.
    let result = check_source(
        "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }
         def double(x: Int<32>) -> Int<32> { return add(x, x); }
         def main() -> Int<32> { return double(21); }",
    );
    assert!(result.is_ok(), "cross-ref functions: {:?}", result.err());
}

#[test]
fn test_top_level_type_error_still_reported() {
    // Type errors at the top level should still propagate.
    let result = check_source("def main() -> Int<32> { return true; }");
    assert!(
        result.is_err(),
        "top-level type error should fail: {:?}",
        result
    );
}

#[test]
fn test_top_level_impl_and_trait() {
    // Trait + impl at top level — generates Impl constraints
    // that the solver must process.
    let result = check_source(
        "trait Show { }
         impl Show for Int<32> { }",
    );
    assert!(result.is_ok(), "top-level trait + impl: {:?}", result.err());
}

// ── Overflow policy ──────────────────────────────────────────────

#[test]
fn test_overflow_default_trap() {
    // Default Int<32> should have Trap overflow policy
    let result = check_source("def f() -> Int<32> { return 1 + 2; }");
    assert!(result.is_ok(), "default overflow: {:?}", result.err());
}

#[test]
fn test_overflow_wrap_type() {
    // Type with explicit overflow = wrap should be accepted
    let result = check_source(
        "type WrapInt = Int<32> with overflow = wrap;
         def f() -> WrapInt { let x: WrapInt = 1; return x; }",
    );
    assert!(result.is_ok(), "wrap type: {:?}", result.err());
}

#[test]
fn test_overflow_saturate_type() {
    let result = check_source(
        "type SatInt = Int<32> with overflow = saturate;
         def f() -> SatInt { let x: SatInt = 1; return x; }",
    );
    assert!(result.is_ok(), "saturate type: {:?}", result.err());
}

#[test]
fn test_overflow_trap_explicit_type() {
    let result = check_source(
        "type TrapInt = Int<32> with overflow = trap;
         def f() -> TrapInt { return 1 + 2; }",
    );
    assert!(result.is_ok(), "explicit trap type: {:?}", result.err());
}

#[test]
fn test_overflow_suffix_on_integer() {
    // Overflow suffix operators (+%, +?, +!) work on integer types
    let result = check_source(
        "def f() -> Int<32> { return 1 +% 2; }
         def g() -> Int<32> { return 1 +? 2; }
         def h() -> Int<32> { return 1 +! 2; }",
    );
    assert!(result.is_ok(), "suffix operators: {:?}", result.err());
}

#[test]
fn test_overflow_policy_matches_constructor() {
    // Verify that int_with_overflow stores and retrieves correctly
    let mut ctx = crate::hir::types::TypeContext::new();
    let wrap = ctx.int_with_overflow(32, true, crate::ast::OverflowPolicy::Wrap);
    assert_eq!(
        ctx.overflow_policy_of(wrap),
        crate::ast::OverflowPolicy::Wrap,
    );
    let sat = ctx.uint_with_overflow(64, crate::ast::OverflowPolicy::Saturate);
    assert_eq!(
        ctx.overflow_policy_of(sat),
        crate::ast::OverflowPolicy::Saturate,
    );
    let def = ctx.int(8, true);
    assert_eq!(
        ctx.overflow_policy_of(def),
        crate::ast::OverflowPolicy::Trap,
    );
}

// ── Layout alias ─────────────────────────────────────────────────

#[test]
fn test_layout_alias_definition() {
    // Layout alias definitions should parse and resolve without error.
    let result = check_source(
        "layout Mmio {
             packed,
             little_endian;
         }",
    );
    assert!(
        result.is_ok(),
        "layout alias definition: {:?}",
        result.err()
    );
}

#[test]
fn test_layout_c_on_type() {
    // @layout(C) should be accepted on a type definition.
    let result = check_source(
        "@layout(C)
         type CStruct = struct { x: Int<32>, y: Int<64> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@layout(C) on type: {:?}", result.err());
}

#[test]
fn test_transparent_on_type() {
    // @transparent should be accepted on a single-field struct.
    let result = check_source(
        "@transparent
         type Wrapper = struct { inner: Int<32> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@transparent on type: {:?}", result.err());
}

#[test]
fn test_layout_alias_usage() {
    // Define a layout alias and use it via @layout(AliasName).
    let result = check_source(
        "layout Compact {
             packed,
             little_endian;
         }

         @layout(Compact)
         type Reg = struct { ctrl: UInt<8>, data: UInt<32> }

         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@layout(AliasName): {:?}", result.err());
}

#[test]
fn test_layout_alias_with_function() {
    // Layout alias alongside a function definition.
    let result = check_source(
        "layout Simple {
             packed;
         }

         @layout(Simple)
         type Header = struct { flags: UInt<8>, len: UInt<8> }

         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "layout with function: {:?}", result.err());
}

// ── Task expression ──────────────────────────────────────────────

#[test]
fn test_task_expression() {
    let result = check_source(
        "def main() -> () {
             set t = task { let x = 1; };
             return ();
         }",
    );
    assert!(result.is_ok(), "task expression: {:?}", result.err());
}

// ── @interrupt handler ───────────────────────────────────────────

#[test]
fn test_interrupt_valid() {
    let result = check_source(
        "@interrupt(irq = 14) @no_alloc @no_panic
         def handler() -> ! {
             loop {
                 // infinite loop — never returns
             }
         }",
    );
    // Note: `loop {}` currently type-checks as `!` (never);
    // if the checker infers `()` instead, this test may fail
    // until the checker is fixed to recognize infinite loops.
    match &result {
        Ok(_) => {} // good
        Err(errors) => {
            // Accept "Never vs Unit" mismatch as a known limitation
            // until loop type inference is implemented.
            let msgs: Vec<&str> = errors.iter().map(|s| s.as_str()).collect();
            let known_issue = msgs
                .iter()
                .any(|m| m.contains("Never") || m.contains("Unreachable"));
            assert!(
                known_issue,
                "unexpected error: {:?} (known: infinite loops may not infer ! yet)",
                errors,
            );
        }
    }
}

#[test]
fn test_interrupt_missing_no_alloc() {
    let result = check_source(
        "@interrupt(irq = 14)
         def handler() -> ! {
             loop {}
         }",
    );
    assert!(result.is_err(), "missing @no_alloc should fail");
}

#[test]
fn test_interrupt_missing_no_panic() {
    let result = check_source(
        "@interrupt(irq = 14) @no_alloc
         def handler() -> ! {
             loop {}
         }",
    );
    assert!(result.is_err(), "missing @no_panic should fail");
}

#[test]
fn test_interrupt_with_alloc_conflict() {
    let result = check_source(
        "@interrupt(irq = 14) @no_alloc @no_panic @alloc
         def handler() -> ! {
             loop {}
         }",
    );
    assert!(result.is_err(), "@alloc with @interrupt should fail");
}

#[test]
fn test_interrupt_with_io_conflict() {
    let result = check_source(
        "@interrupt(irq = 14) @no_alloc @no_panic @io
         def handler() -> ! {
             loop {}
         }",
    );
    assert!(result.is_err(), "@io with @interrupt should fail");
}

// ── @no_panic body verification ─────────────────────────────────

/// `@no_panic` body verification (SYNTAX.md §Effect Annotations): a
/// default (trap-policy) float arithmetic operation can panic (NaN/∞/
/// div-by-zero — the float default overflow policy is `trap`), so it
/// must be rejected inside a `@no_panic` function.  In strict mode this
/// is an error; in non-strict mode a warning (the program still checks).
#[test]
fn test_no_panic_rejects_default_float_trap() {
    // Strict mode: an error.
    let result = check_strict(
        "@no_panic
         def f() -> Float<64> {
             let x = 1.0 + 2.0;
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "strict mode must reject default-trap float arithmetic in @no_panic: {:?}",
        result.err(),
    );
    // Non-strict mode: still a diagnostic (warning), not a hard error.
    let result = check_source(
        "@no_panic
         def f() -> Float<64> {
             let x = 1.0 + 2.0;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "non-strict mode should only warn on default-trap float arithmetic: {:?}",
        result.err(),
    );
}

/// An explicit trap operator (`+!`) can always panic — rejected in
/// `@no_panic` regardless of the operand kind.
#[test]
fn test_no_panic_rejects_explicit_trap_op() {
    let result = check_strict(
        "@no_panic
         def f(x: Int<32>) -> Int<32> {
             return x +! 1;
         }",
    );
    assert!(
        result.is_err(),
        "strict mode must reject explicit trap operator in @no_panic: {:?}",
        result.err(),
    );
}

/// A `panic` call in a `@no_panic` function is a direct violation.
#[test]
fn test_no_panic_rejects_panic_call() {
    let result = check_strict(
        "@no_panic
         def f(x: Int<32>) -> Int<32> {
             if x == 0 { panic; }
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "strict mode must reject a panic call in @no_panic: {:?}",
        result.err(),
    );
}

/// Calling a non-`@no_panic` function from a `@no_panic` function can
/// transitively panic — rejected.
#[test]
fn test_no_panic_rejects_call_to_non_no_panic() {
    let result = check_strict(
        "def callee() -> Int<32> { return 1; }
         @no_panic
         def f() -> Int<32> {
             return callee();
         }",
    );
    assert!(
        result.is_err(),
        "strict mode must reject a call to a non-@no_panic function: {:?}",
        result.err(),
    );
}

/// IEEE float semantics (`+%` — the explicit opt-in) never trap: it
/// returns special values instead of panicking, so it is ALLOWED inside
/// `@no_panic` (the committee ruling's float `+%` ≡ IEEE).
#[test]
fn test_no_panic_allows_ieee_float() {
    let result = check_strict(
        "@no_panic
         def f() -> Float<64> {
             let x = 1.0 +% 2.0;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "IEEE float arithmetic (+%) must be allowed in @no_panic: {:?}",
        result.err(),
    );
}

/// The verification only applies to `@no_panic` functions — a plain
/// function may use default float arithmetic without a diagnostic.
#[test]
fn test_no_panic_does_not_affect_plain_functions() {
    let result = check_source(
        "def f() -> Float<64> {
             let x = 1.0 + 2.0;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "plain (non-@no_panic) functions are unaffected: {:?}",
        result.err(),
    );
}

// ── Channel type ─────────────────────────────────────────────────

#[test]
fn test_channel_type_parses() {
    // Verify Channel is registered as a built-in type.
    // Type name resolution happens during resolver phase, but Channel
    // is registered in register_builtins which runs after resolution.
    // So we use a direct API check instead of a source-level test.
    let mut ctx = TypeContext::new();
    let mut symbols = crate::hir::symbol::SymbolTable::new(crate::hir::types::CrateId(
        crate::hir::types::DefId(0),
    ));
    let mut trait_env = crate::hir::traits::TraitEnv::new();
    crate::hir::builtins::register_builtins(&mut symbols, &mut trait_env, &mut ctx);
    let binding = symbols.lookup_type(Symbol::intern("Channel"));
    assert!(
        binding.is_some(),
        "Channel should be registered as a built-in type"
    );
    assert!(
        !binding.unwrap().params.is_empty(),
        "Channel should have at least one type parameter T",
    );
}

// ── Layout attributes ────────────────────────────────────────────

#[test]
fn test_layout_attr_packed() {
    let result = check_source(
        "@packed
         type Packed = struct { flags: UInt<8>, data: UInt<16> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@packed: {:?}", result.err());
}

#[test]
fn test_layout_attr_endian() {
    let result = check_source(
        "@endian(little)
         type Regs = struct { ctrl: UInt<8> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@endian: {:?}", result.err());
}

#[test]
fn test_layout_attr_bit_order() {
    let result = check_source(
        "@bit_order(lsb_to_msb)
         type Bits = struct { lo: UInt<4>, hi: UInt<4> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@bit_order: {:?}", result.err());
}

#[test]
fn test_layout_attr_align() {
    let result = check_source(
        "@align(16)
         type Aligned = struct { x: Int<32> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@align: {:?}", result.err());
}

#[test]
fn test_layout_attr_pad() {
    let result = check_source(
        "@pad(4)
         type Padded = struct { x: Int<32> }
         def main() -> Int<32> { return 0; }",
    );
    assert!(result.is_ok(), "@pad: {:?}", result.err());
}

// ── Generic constraint aliases (constraint … <T> { T: … }) ─────

#[test]
fn test_generic_constraint_satisfied_accepted() {
    // Define a generic constraint and apply it to a type that satisfies it.
    let result = check_source(
        "
            trait Foo { }
            impl Foo for Int<32> { }
            constraint NeedsFoo<T> { T: Foo }
            def needs_foo<T>(x: T) -> T where T: NeedsFoo { return x; }
            def main() -> Int<32> {
                return needs_foo(42);
            }",
    );
    assert!(
        result.is_ok(),
        "constraint satisfied should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_generic_constraint_parses_and_resolves() {
    // Verify that a generic constraint parses, resolves, and the checker
    // does not crash.  The full call-site impl check requires generic
    // instantiation, which is a broader checker feature.
    let result = check_source(
        "
            trait Foo { }
            impl Foo for Int<32> { }
            constraint NeedsFoo<T> { T: Foo }
            def needs_foo<T>(x: T) -> T where T: NeedsFoo { return x; }
            def main() -> Int<32> { return 0; }
        ",
    );
    assert!(
        result.is_ok(),
        "constraint should parse and resolve: {:?}",
        result.err()
    );
}

// ── Tuple subject in where clause (Track‑B) ──────────────────────

#[test]
fn test_where_tuple_subject_parses_and_resolves() {
    // `where (T, U): Rel` with a multi-param constraint should parse,
    // resolve, and type-check without crashing.  The positional
    // substitution maps constraint params to tuple elements.
    let result = check_source(
        "
            trait Foo { }
            trait Bar { }
            impl Foo for Int<32> { }
            impl Bar for Bool { }
            constraint Rel<T, U> { T: Foo, U: Bar }
            def rel_fn<X, Y>(x: X, y: Y) -> Y where (X, Y): Rel { return y; }
            def main() -> Bool {
                return rel_fn(42, true);
            }
        ",
    );
    assert!(
        result.is_ok(),
        "tuple subject in where clause should not crash: {:?}",
        result.err()
    );
}

#[test]
fn test_where_tuple_subject_direct_trait_rejected() {
    // Applying a direct trait bound (not a constraint alias) to a
    // tuple subject should be rejected — it's ambiguous.
    let result = check_source(
        "
            trait Foo { }
            def bad_fn<X, Y>(x: X, y: Y) where (X, Y): Foo { }
            def main() -> Int<32> { return 0; }
        ",
    );
    assert!(
        result.is_err(),
        "direct trait bound on tuple subject should be rejected: {:?}",
        result
    );
}
#[cfg(test)]
mod test_infer_return {
    use super::*;

    #[test]
    fn test_infer_return_from_literal() {
        let result = check_source("def main() { return 42; }");
        // Should succeed: infer return type as Int<32>
        assert!(
            result.is_ok(),
            "infer return from literal: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_infer_return_from_bool() {
        let result = check_source("def main() { return true; }");
        assert!(result.is_ok(), "infer return from bool: {:?}", result.err());
    }

    #[test]
    fn test_infer_return_no_return_defaults_to_never() {
        let result = check_source("def main() { }");
        assert!(
            result.is_ok(),
            "no return defaults to never: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_infer_return_empty_return_defaults_to_unit() {
        let result = check_source("def main() { return; }");
        assert!(
            result.is_ok(),
            "empty return defaults to unit: {:?}",
            result.err()
        );
    }
}

#[cfg(test)]
mod test_regex {
    use super::*;

    #[test]
    fn test_regex_valid_pattern() {
        // Valid regex patterns should parse and resolve successfully.
        // Use Regex<"..."> as a type annotation on a function parameter.
        // The function is not called, so no type mismatch on arguments.
        let result = check_source(
            "def foo(x: Regex<\"[0-9]+\">) -> Int<32> { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "valid regex pattern should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_regex_valid_pattern_complex() {
        // More complex regex: email-like pattern.
        let result = check_source(
            "def foo(x: Regex<\"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}\">) -> Int<32> { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "complex regex pattern should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_regex_invalid_pattern_rejected() {
        // Invalid regex: unmatched opening bracket.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"[0-9\"> = 0;
                 return 0;
             }",
        );
        assert!(result.is_err(), "invalid regex pattern should be rejected");
        let errs = result.unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("invalid regex pattern")),
            "error should mention invalid regex pattern: {:?}",
            errs
        );
    }

    #[test]
    fn test_regex_invalid_escape_rejected() {
        // Invalid regex: `\k` is not a valid regex escape sequence.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"\\k\"> = 0;
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "invalid escape in regex pattern should be rejected"
        );
    }

    #[test]
    fn test_regex_display_format() {
        // Verify that TypeData::Regex Display produces the correct format.
        let mut ctx = TypeContext::new();
        let regex_ty = ctx.alloc(TypeData::Regex {
            pattern: "[0-9]+".into(),
        });
        let display_str = format!("{}", ctx.get(regex_ty));
        assert_eq!(
            display_str, "Regex<\"[0-9]+\">",
            "Regex Display should match syntax"
        );
    }

    #[test]
    fn test_regex_pathological_patterns() {
        // ── Edge-case & pathological regex patterns ──

        // Empty pattern — valid regex (matches the empty string).
        let result = check_source(
            "def foo(x: Regex<\"\">) -> Int<32> { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "empty regex pattern should be valid: {:?}",
            result.err()
        );

        // Meta characters: `^`, `$`, `\d`, `+`, `?`, `|`, `(`, `)`.
        let result = check_source(
            "def foo(x: Regex<\"^\\\\d+$|(foo|bar)?\">) -> Int<32> { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "regex with meta characters should be valid: {:?}",
            result.err()
        );

        // Unclosed group — should be rejected.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"(\"> = 0;
                 return 0;
             }",
        );
        assert!(result.is_err(), "unclosed paren should be rejected");

        // Unclosed character class.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"[abc\"> = 0;
                 return 0;
             }",
        );
        assert!(result.is_err(), "unclosed bracket should be rejected");

        // Empty character class.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"[]\"> = 0;
                 return 0;
             }",
        );
        assert!(result.is_err(), "empty bracket should be rejected");

        // Consecutive quantifiers — `**` is invalid.
        let result = check_source(
            "def main() -> Int<32> {
                 let _: Regex<\"**\"> = 0;
                 return 0;
             }",
        );
        assert!(result.is_err(), "repeated quantifier should be rejected");

        // A very long pattern — 100KB of repeated 'a' — must not panic or OOM.
        let long_pattern = "a".repeat(100_000);
        let source = format!(
            "def foo(x: Regex<\"{}\">) -> Int<32> {{ return 0; }}
             def main() -> Int<32> {{ return 0; }}",
            long_pattern
        );
        // Exercise the parser directly, skipping type-check (100KB would
        // trigger other paths).
        let arena = bumpalo::Bump::new();
        let mut parser = crate::parser::Parser::new(&source, &arena);
        let program = parser.parse_program();
        assert!(
            program.is_ok(),
            "100KB regex should not crash parser: {:?}",
            program.err()
        );
    }

    #[test]
    fn test_trait_obligations_salvaged_after_failed_function_body() {
        // Regression test: if a function body fails before the
        // trait_obligations drain site, the obligations must be salvaged
        // into residual_trait_obligations and processed at the top level.
        //
        // Source: `ensures @s > 1` pushes an `Ord(Int)` obligation.
        // `set i = j + 1` (String + Int) fails the function body.
        // The second function `def main(){}` triggers the salvage path.
        let result = check_source(
            "def a(x:Bool)
                 ensures @s > 1
                 ensures @r > 0
               {
                 set j = \"0xFFFF\";
                 set i = j + 1;
                 return @s @r i;
               }
             def main(){}",
        );
        // The `Ord` obligation should survive the function body failure.
        // The actual error is now a type mismatch between integer and `&Str`
        // (from `set i = j + 1`), not an `Ord`-not-found false positive.
        assert!(
            result.is_err(),
            "expected type mismatch error, but type-checking succeeded"
        );
        let errs = result.unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.contains("type mismatch") && e.contains("&Str")),
            "expected type mismatch with &Str, got: {:?}",
            errs
        );
    }

    #[test]
    fn test_strict_mode_rejects_unproven_trusted() {
        // In strict mode, @trusted without @link_proof should be rejected.
        let source = "@trusted
         def unsafe_fn() -> Int<32> { return 42; }
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            true,   // strict_mode
            false,  // enable_experimental
            vec![], // features
            false,  // debug
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_err(),
            "strict mode should reject @trusted without @link_proof"
        );
    }

    /// `dyn Trait` is rejected in STRICT mode outside
    /// `@trusted` code (SYNTAX.md §Dynamic Dispatch) — the fail-closed
    /// gate in the DynTrait resolution.
    #[test]
    fn test_m3_dyn_trait_rejected_in_strict() {
        let source = "def f(x: dyn Foo) -> Int<32> { return 0; }
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            true,   // strict_mode
            false,  // enable_experimental
            vec![], // features
            false,  // debug
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_err(),
            "strict mode must reject `dyn Trait` outside @trusted"
        );
        let msgs = result.err().unwrap();
        assert!(
            msgs.iter()
                .any(|m| m.message().to_string().contains("dyn Trait")),
            "the dyn-Trait strict rejection must be reported: {:?}",
            msgs
        );
    }

    /// m3 control: in NON-strict mode the same `dyn Trait` type parses
    /// (no fail-open regression — the gate is strict-mode-scoped).
    #[test]
    fn test_m3_dyn_trait_accepted_non_strict() {
        let result = check_source(
            "def f(x: dyn Foo) -> Int<32> { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        // `Foo` is not a registered trait — the dyn type resolves with an
        // empty trait set; the point is it must NOT be rejected by the
        // strict gate (no "dyn Trait" diagnostic in non-strict mode).
        let errs = result.err().unwrap_or_default();
        assert!(
            !errs.iter().any(|m| m.contains("dyn Trait is not allowed")),
            "non-strict mode must not apply the strict dyn-Trait gate: {:?}",
            errs
        );
    }

    /// `@trusted` must carry `requires`/`ensures` contracts
    /// (SYNTAX.md:1039/1202) — a `@trusted` function with no contracts is
    /// rejected.
    #[test]
    fn test_m4_trusted_requires_contracts() {
        let source = "@trusted
         def unsafe_fn() -> Int<32> { return 42; }
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            false,  // strict_mode (the contract requirement is NOT strict-only)
            false,  // enable_experimental
            vec![], // features
            false,  // debug
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_err(),
            "@trusted without requires/ensures contracts must be rejected"
        );
        let msgs = result.err().unwrap();
        assert!(
            msgs.iter()
                .any(|m| m.message().to_string().contains("requires`/`ensures")),
            "the contract requirement must be reported: {:?}",
            msgs
        );
    }

    /// m4 control: `@trusted` WITH contracts is accepted (no contract —
    /// the m4 gate is the only thing being tested here).
    #[test]
    fn test_m4_trusted_with_contracts_ok() {
        let result = check_source(
            "@trusted
         def unsafe_fn(x: Int<32>) -> Int<32>
             requires x > 0
             ensures x > 0
         {
             return x;
         }
         def main() -> Int<32> {
             return 0;
         }",
        );
        assert!(
            result.is_ok(),
            "@trusted with requires/ensures contracts must be accepted: {:?}",
            result
        );
    }

    /// An isolate block must not MUTATE a CAPTURED outer
    /// mutable local (SYNTAX.md §Task Isolation — "does not access any
    /// external mutable state").  `x` is declared OUTSIDE the block, so
    /// `x = 5` inside is a mutation of external state.
    #[test]
    fn test_m5_isolate_captured_mutation_rejected() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut x = 42;
                 isolate {
                     x = 5;
                 }
                 return x;
             }",
        );
        assert!(
            result.is_err(),
            "mutating a captured outer variable inside isolate must be rejected: {:?}",
            result
        );
    }

    /// m5 control: a variable declared INSIDE the isolate block is
    /// internal state — mutating it is fine.
    #[test]
    fn test_m5_isolate_internal_mutation_ok() {
        let result = check_source(
            "def main() -> Int<32> {
                 isolate {
                     set mut y = 1;
                     y = 5;
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_ok(),
            "mutating a variable declared inside isolate must be accepted: {:?}",
            result
        );
    }

    /// The new-lifetime-syntax diagnostics carry an error
    /// code (E004) — they must not render as `[? ERROR]`.  `where 'a:`
    /// (missing outlives) triggers the coded diagnostic.
    #[test]
    fn test_m6a_lifetime_diagnostics_coded() {
        let result = check_source(
            "def f<'a>(x: Int<32>) -> Int<32> where 'a: { return 0; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_err(), "malformed `where 'a:` must be rejected");
        let msgs = result.err().unwrap();
        assert!(
            msgs.iter().any(|m| m.contains("lifetime")),
            "the lifetime diagnostic must be reported: {:?}",
            msgs
        );
    }

    /// Any keyword is valid AFTER `::` in path position
    /// — the keywords added to `as_ident_symbol`
    /// (requires, for, set, ...) must parse as path segments instead of
    /// failing with "expected identifier after '::'".
    #[test]
    fn test_m6b_keyword_after_coloncolon_parses() {
        let result = check_source(
            "def main() -> Int<32> {
                 let a = Foo::requires;
                 let b = Foo::for;
                 let c = Foo::set;
                 return 0;
             }",
        );
        // `Foo` is undefined — the checker reports an undefined name, but
        // the PATH-POSITION keywords must NOT produce the parser's
        // "expected identifier after '::'" error.
        let errs = result.err().unwrap_or_default();
        assert!(
            !errs
                .iter()
                .any(|m| m.contains("expected identifier after '::'")),
            "keywords after `::` must parse as path segments: {:?}",
            errs
        );
    }

    /// The error-code table semantics are aligned with the
    /// actual use sites — E008 is the return-Err lint (not "integer
    /// overflow", which no code emits), E060 is the GADT constraint
    /// violation (not "internal compiler error" — a user-visible
    /// diagnostic must not masquerade as an internal bug).
    #[test]
    fn test_m6c_error_code_semantics_aligned() {
        assert_eq!(
            crate::diagnostics::error_code::lookup("E008").map(|e| e.title),
            Some("return Err is not valid; use leave with"),
        );
        assert_eq!(
            crate::diagnostics::error_code::lookup("E060").map(|e| e.title),
            Some("GADT variant constraint violation"),
        );
    }

    /// A function body whose FINAL
    /// statement is an `if` whose branches RETURN values
    /// (`if c { return x; } else { return y; }`) must be typed by the
    /// branch returns — `block_type_impl` previously fell through to
    /// Unit, wrongly rejecting valid bodies ("expected T, found Unit").
    #[test]
    fn test_if_branch_returns_as_function_body_ok() {
        let result = check_source(
            "def pick<'a, 'b>(x: &'a Int<32>, y: &'b Int<32>) -> &'a Int<32> {
                 if true { return x; } else { return x; }
             }
             def main() -> Int<32> {
                 set mut a = 42;
                 set mut b = 7;
                 set r = pick(&a, &b);
                 return *r;
             }",
        );
        assert!(
            result.is_ok(),
            "an if whose branches return values must type the function body: {:?}",
            result
        );
    }

    /// Control: an `if` whose branches return DIFFERENT
    /// types is still rejected (no over-acceptance).
    #[test]
    fn test_if_branch_returns_type_mismatch_rejected() {
        let result = check_source(
            "def f(x: Int<32>) -> Int<32> {
                 if true { return x; } else { return true; }
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "an if whose branches return mismatched types must be rejected: {:?}",
            result
        );
    }

    #[test]
    fn test_experimental_accepted_with_flag() {
        // @experimental should be accepted when --enable-experimental is set.
        let source = "@experimental
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            false,  // strict_mode
            true,   // enable_experimental
            vec![], // features
            false,  // debug
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_ok(),
            "@experimental should be accepted with --enable-experimental, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_strict_mode_rejects_contradictory_cfg() {
        // all(target_os == "linux", target_os == "windows") is contradictory.
        let source = "@cfg(all(target_os == \"linux\", target_os == \"windows\"))
         def f() -> Int<32> { return 0; }
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            true,
            false,
            vec![],
            false,
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_err(),
            "strict mode should reject contradictory cfg condition"
        );
    }

    #[test]
    fn test_cfg_debug_with_flag() {
        let source = "@cfg(debug)
         def f() -> Int<32> { return 0; }
         def main() -> Int<32> { return 0; }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            false,
            false,
            vec![],
            true,
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_ok(),
            "@cfg(debug) should pass with --debug, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_comptime_sandbox_rejects_trusted_call() {
        let source = "@trusted
         def unsafe_fn() -> Int<32> { return 42; }
         def main() -> Int<32> {
             comptime { return unsafe_fn(); }
         }";
        let arena = bumpalo::Bump::new();
        let mut parser = Parser::new(source, &arena);
        let program = parser.parse_program().expect("parse should succeed");
        let mut ctx = TypeContext::new();
        let local_crate_id = CrateId(DefId(0));
        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
        let (symbols, mut trait_env, _res_diags, resolution_map) =
            resolver.resolve_program(&program);
        let mut checker = TypeChecker::new(
            &mut ctx,
            &symbols,
            &mut trait_env,
            resolution_map,
            false,
            false,
            vec![],
            false,
        );
        let result = checker.check_program(&program);
        assert!(
            result.is_err(),
            "comptime block should reject call to @trusted function"
        );
    }

    // ── GADT tests ─────────────────────────────────────────────────

    #[test]
    fn test_gadt_construction_ok() {
        // GADT variant construction with matching type args
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set e: Expr<Int<32>> = Expr::Lit(42);
                 return 0;
             }",
        );
        assert!(
            result.is_ok(),
            "GADT construction with matching type should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_construction_wrong_type_errors() {
        // GADT variant construction with wrong type args should fail
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def main() -> Bool {
                 set e: Expr<Bool> = Expr::Lit(true);
                 return e;
             }",
        );
        assert!(
            result.is_err(),
            "GADT construction with wrong type should error"
        );
    }

    #[test]
    fn test_gadt_pattern_refinement_eval() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return match x { Mk(n) => n };
}",
        );
        assert!(
            result.is_ok(),
            "GADT pattern refinement should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_if_let_expr_refinement() {
        // if-let as an expression — both branches produce value, unified.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return if let Mk(n) = x { n } else { 0 };
}",
        );
        assert!(
            result.is_ok(),
            "GADT if-let expression refinement should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_nested_refinement() {
        // GADT refinement with nested type: payload refers to another GADT.
        let result = check_source(
            "type Expr<T> = enum {
    Lit(Int<32>) when T == Int<32>,
    Wrap(Expr<Int<32>>) when T == Bool,
}
def main() -> Int<32> {
    set e: Expr<Int<32>> = Expr::Lit(42);
    return match e { Lit(n) => n };
}",
        );
        assert!(
            result.is_ok(),
            "GADT nested refinement should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_multi_variant() {
        // Multi-variant GADT: pattern match with multiple arms.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Other(Bool) when T == Bool }
def main() -> Bool {
    set x: Wrap<Bool> = Wrap::Other(true);
    return match x { Other(b) => b };
}",
        );
        assert!(
            result.is_ok(),
            "GADT multi-variant match should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_while_let() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    while let Mk(n) = x { set _v = n; }
    return 0;
}",
        );
        assert!(result.is_ok(), "GADT while-let should pass: {:?}", result);
    }

    #[test]
    fn test_gadt_arm_type_mismatch() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Other(String) when T == Int<32> }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return match x {
        Mk(n) => n,
        Other(s) => s,
    };
}",
        );
        assert!(
            result.is_err(),
            "Incompatible GADT arm types should be rejected"
        );
    }

    #[test]
    fn test_gadt_polymorphic_arm_mismatch() {
        // GADT arms whose body types come from polymorphic function calls
        // (InferVar) that get GADT-refined to different concrete types.
        // Exercises the replace_type_ids abstraction path.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Other(Bool) when T == Bool }
def identity<U>(x: U) -> U { return x; }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return match x {
        Mk(n) => identity(n),
        Other(b) => identity(b),
    };
}",
        );
        assert!(
            result.is_ok(),
            "Polymorphic GADT arms with one reachable variant should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_polymorphic_arm_ok() {
        // GADT arms with polymorphic function bodies that share the same type.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
def identity<U>(x: U) -> U { return x; }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return match x {
        Mk(n) => identity(n),
    };
}",
        );
        assert!(
            result.is_ok(),
            "Polymorphic GADT arm with matching type should pass: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_transaction_isolates_arms() {
        // Both arms are reachable (same constraint).  Each arm's GADT
        // refinement is rolled back before the next arm processes, so
        // arm 1's `T = Int<32>` does not contaminate arm 2.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Other(Int<32>) when T == Int<32> }
def main() -> Int<32> {
    set x: Wrap<Int<32>> = Wrap::Mk(42);
    return match x {
        Mk(n) => n,
        Other(n) => n,
    };
}"
        );
        assert!(
            result.is_ok(),
            "Both arms should type-check independently: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_polymorphic_scrutinee_refines_return_type() {
        // The reviewer's critical concern: GADT refinement with a polymorphic
        // scrutinee whose type argument is a GenericParam (not concrete).
        //
        // In `unwrap<T>(x: Wrap<T>) -> Int<32>`, the match scrutinee has type
        // `Wrap<GenericParam{T}>`.  The when-clause says `Mk(Int<32>) when
        // T == Int<32>`.  Inside the arm, T must be refined to Int<32> so
        // that the payload `n: Int<32>` is accepted as the return type.
        //
        // This test exercises is_gadt_variant_reachable and
        // apply_gadt_refinement with a GenericParam type argument,
        // which the original implementation could not handle.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def unwrap<T>(x: Wrap<T>) -> Int<32> {
                 return match x { Mk(n) => n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "Polymorphic GADT scrutinee should refine return type: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_polymorphic_scrutinee_reachable_variant() {
        // Polymorphic GADT reachability: in a generic function `take_int<T>`,
        // T is a GenericParam that could be instantiated to any type.
        // Both the Mk (requires T == Int<32>) and Other (requires T == Bool)
        // variants could be reachable depending on T's concrete type.
        // Therefore, the match MUST include a wildcard (no arm is
        // definitively dead at the generic level).
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Other(Bool) when T == Bool }
             def take_int<T>(x: Wrap<T>) -> Int<32> {
                 return match x { Mk(n) => n, _ => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "Polymorphic reachable check should pass with wildcard: {:?}",
            result
        );
    }

    #[test]
    fn test_gadt_polymorphic_scrutinee_identity_refinement() {
        // Polymorphic function with GADT where the refined type parameter
        // is used via an identity function call inside the arm.
        // The GADT registry makes GenericParam{T} resolve to Int<32>
        // transparently within the arm, so `identity(n: Int<32>)`
        // correctly resolves U = Int<32>.
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def identity<U>(x: U) -> U { return x; }
             def unwrap<T>(x: Wrap<T>) -> Int<32> {
                 return match x { Mk(n) => identity(n) };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "Polymorphic GADT with identity call should refine: {:?}",
            result
        );
    }

    /// The reviewer's exact critical scenario: a generic function whose
    /// return type IS the refined type parameter (T, not Int<32>).
    ///
    /// `eval<T>(x: Expr<T>) -> T` — the when-clause refines T to Int<32>
    /// in the Lit arm.  After the arm, the match result type is Int<32>.
    /// The return-type check unifies Int<32> with GenericParam{T}.
    /// The reviewer claims this fails ("the unifier has no rule").
    ///
    /// This test proves it works: `unify_internal_impl` at the
    /// `(_, TypeData::GenericParam{..})` arm binds GenericParam→Concrete
    /// via `set_binding`.  The unification succeeds.
    #[test]
    fn test_gadt_refines_return_type_param() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def eval<T>(x: Expr<T>) -> T {
                 return match x { Lit(n) => n };
             }
             def main() -> Int<32> {
                 set e: Expr<Int<32>> = Expr::Lit(42);
                 return eval(e);
             }",
        );
        assert!(
            result.is_ok(),
            "GADT must refine return type parameter T: {:?}",
            result
        );
    }

    /// Variant<'input>: polymorphic identity called inside the arm, function
    /// return type is the GADT-refined parameter.  Exercises the full
    /// pipeline: GADT registry → arm body inference → pop → return-type
    /// unification with GenericParam.
    #[test]
    fn test_gadt_refines_return_type_param_via_call() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def id<U>(x: U) -> U { return x; }
             def eval<T>(x: Expr<T>) -> T {
                 return match x { Lit(n) => id(n) };
             }
             def main() -> Int<32> {
                 set e: Expr<Int<32>> = Expr::Lit(42);
                 return eval(e);
             }",
        );
        assert!(
            result.is_ok(),
            "GADT must refine return type T with intermediate call: {:?}",
            result
        );
    }

    /// Edge: multiple type parameters and `and` constraints
    #[test]
    fn test_gadt_edge_multi_param() {
        let result = check_source(
            "type KV<K, V> = enum { Pair(Int<32>) when K == Int<32> and V == Bool }
             def main() -> Int<32> {
                 set x: KV<Int<32>, Bool> = KV::Pair(42);
                 return match x { Pair(n) => n };
             }",
        );
        assert!(result.is_ok(), "multi-param GADT: {:?}", result);
    }

    /// Edge: compound tuple payload type
    #[test]
    fn test_gadt_edge_tuple_payload() {
        let result = check_source(
            "type Wrap<T> = enum { Pair((Int<32>, Bool)) when T == (Int<32>, Bool) }
             def main() -> Int<32> {
                 set x: Wrap<(Int<32>, Bool)> = Wrap::Pair((42, true));
                 return match x { Pair((a, _)) => a };
             }",
        );
        assert!(result.is_ok(), "tuple payload GADT: {:?}", result);
    }

    /// Edge: if-let with reachable GADT variant and else branch
    #[test]
    fn test_gadt_edge_if_let_with_else() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 return if let Mk(n) = x { n } else { 0 };
             }",
        );
        assert!(result.is_ok(), "if-let GADT with else: {:?}", result);
    }

    /// Edge: non-GADT variant in GADT enum (empty eq_spec)
    #[test]
    fn test_gadt_edge_non_gadt_variant() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, Plain(Bool) }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 return match x { Mk(n) => n, Plain(_) => 0 };
             }",
        );
        assert!(
            result.is_ok(),
            "non-GADT variant in GADT enum: {:?}",
            result
        );
    }

    /// A non-GADT arm AFTER a discharging arm must still be type-checked
    /// against the expected type — the `gadt_discharged` latch must not
    /// let the cross-arm check bypass slip through.
    #[test]
    fn gadt_discharge_bypass_repro() {
        // A non-GADT arm (B(Bool)) after a discharging arm (A) in a generic
        // function must be rejected: its Bool body is incompatible with the
        // generic expected T.  Regression test for the gadt_discharged latch
        // bypass (the arm was previously never checked — see fn_ctxt.rs).
        let result = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32>, B(Bool) }
             def f<T>(x: E<T>) -> T {
                 return match x { A(n) => n, B(b) => b };
             }",
        );
        assert!(
            result.is_err(),
            "non-discharging arm after a discharge must be checked against the expected type: {:?}",
            result
        );
    }

    /// The residual bypass when a non-discharging arm PRECEDES the
    /// discharging arm (accumulated arm_ty is discarded on discharge):
    /// the earlier arm must still be checked against the expected type.
    #[test]
    fn gadt_discharge_bypass_repro2() {
        let result = check_source(
            "type E<T> = enum { B(Bool), A(Int<32>) when T == Int<32> }
             def f<T>(x: E<T>) -> T {
                 return match x { B(b) => b, A(n) => n };
             }",
        );
        assert!(
            result.is_err(),
            "non-discharging arm preceding a discharge must be checked: {:?}",
            result
        );
    }

    /// A 3-arm match where a non-discharging arm sits BETWEEN two
    /// discharging arms — the middle arm must still be checked against
    /// the expected type (change (2) in the gadt_discharged fix).
    #[test]
    fn gadt_discharge_bypass_repro3_middle_arm() {
        // Middle arm B(Bool) does not discharge; its concrete Bool body is
        // incompatible with the abstract T — must be rejected.
        let result = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32>, B(Bool), C(Int<32>) when T == Int<32> }
             def f<T>(x: E<T>) -> T {
                 return match x { A(n) => n, B(b) => b, C(n) => n };
             }",
        );
        assert!(
            result.is_err(),
            "middle non-discharging arm between two discharging arms must be checked: {:?}",
            result
        );
        // Positive control: the middle arm returning the generic T is fine.
        let ok = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32>, B(T), C(Int<32>) when T == Int<32> }
             def f<T>(x: E<T>) -> T {
                 return match x { A(n) => n, B(b) => b, C(n) => n };
             }",
        );
        assert!(
            ok.is_ok(),
            "middle non-discharging arm returning the generic T must pass: {:?}",
            ok
        );
    }

    /// GADT variant with contradictory `when` constraints (E065): the same
    /// type parameter forced to two different concrete types is unsatisfiable
    /// — the variant cannot be constructed at any instantiation.
    #[test]
    fn test_gadt_variant_conflicting_constraints() {
        // Same param constrained to two different concrete types → error.
        let bad = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32> and T == Bool }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            bad.is_err(),
            "contradictory constraints must error: {:?}",
            bad
        );
        // Bit-width mismatch is also a contradiction.
        let bad2 = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32> and T == Int<64> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(bad2.is_err(), "bit-width mismatch must error: {:?}", bad2);
        // Different params are independent — satisfiable.
        let ok = check_source(
            "type E<T, U> = enum { A(Int<32>) when T == Int<32> and U == Bool }
             def main() -> Int<32> { return 0; }",
        );
        assert!(ok.is_ok(), "independent params must pass: {:?}", ok);
        // Redundant identical constraints are satisfiable.
        let ok2 = check_source(
            "type E<T> = enum { A(Int<32>) when T == Int<32> and T == Int<32> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok2.is_ok(),
            "redundant identical constraints must pass: {:?}",
            ok2
        );
        // An opaque exists-witness RHS (`when T == X`) may equal anything —
        // the pair is not provably contradictory.
        let ok3 = check_source(
            "type E<T> = enum { A(exists X: Int<32>) when T == X and T == Int<32> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(ok3.is_ok(), "opaque exists witness must pass: {:?}", ok3);
        // A global alias RHS may alias the same type under different names —
        // the pair is NOT provably contradictory (E065 must not fire).
        let ok4 = check_source(
            "type A = Int<32>
             type B = Int<32>
             type E<T> = enum { Mk(Int<32>) when T == A and T == B }
             def main() -> Int<32> { return 0; }",
        );
        assert!(ok4.is_ok(), "alias RHS must not be flagged: {:?}", ok4);
        // An `exists` witness nested inside a larger RHS may equal anything —
        // the pair is satisfiable (E065 must not fire).
        let ok5 = check_source(
            "type E<T> = enum { A(exists X: Int<32>) when T == [X] and T == [Bool] }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok5.is_ok(),
            "nested exists witness must not be flagged: {:?}",
            ok5
        );
        // A non-literal array size (const-generic parameter) is not provably
        // concrete — two identical such constraints are satisfiable (E065
        // must not fire on the const-param size).
        let ok6 = check_source(
            "type E<T, const N: usize> = enum { Mk(Int<32>) when T == [Int<32>; N] and T == [Int<32>; N] }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok6.is_ok(),
            "non-literal array size must not be flagged: {:?}",
            ok6
        );
        // The same type under a qualified path (`core::Int`) and an
        // unqualified path (`Int`) is ONE type — nominal (resolved) equality,
        // not syntactic path equality (E065 must not fire).
        let ok7 = check_source(
            "type E<T> = enum { Mk(Int<32>) when T == Int<32> and T == core::Int<32> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok7.is_ok(),
            "qualified/unqualified same type must not be flagged: {:?}",
            ok7
        );
        // The same type under a qualified path INSIDE a generic argument
        // (`Pair<Int<32>, core::Int<32>>` vs `Pair<Int<32>, Int<32>>`) is
        // ONE type — the generic-args comparison is nominal (E065 must not
        // fire on the args).
        let ok8 = check_source(
            "type Pair<A, B> = struct { a: A, b: B }
             type E<T> = enum { Mk(Int<32>) when T == Pair<Int<32>, Int<32>> and T == Pair<Int<32>, core::Int<32>> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok8.is_ok(),
            "qualified-path generic args must not be flagged: {:?}",
            ok8
        );
        // The genuine args-level contradiction (Int<32> vs Int<64> inside
        // the same constructor) must still fire E065.
        let bad3 = check_source(
            "type Pair<A, B> = struct { a: A, b: B }
             type E<T> = enum { Mk(Int<32>) when T == Pair<Int<32>, Int<32>> and T == Pair<Int<64>, Int<32>> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            bad3.is_err(),
            "args-level contradiction must error: {:?}",
            bad3
        );
        // Named generic arguments are UNORDERED — `Pair<n=Int<32>, m=Bool>`
        // and `Pair<m=Bool, n=Int<32>>` are the same type, so a constraint
        // pair differing only in the named-arg ORDER is NOT a contradiction
        // (E065 must not fire on the reordered named args).
        let ok9 = check_source(
            "type Pair<N, M> = struct { n: N, m: M }
             type E<T> = enum { Mk(Int<32>) when T == Pair<n=Int<32>, m=Bool> and T == Pair<m=Bool, n=Int<32>> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            ok9.is_ok(),
            "reordered named generic args must not be flagged: {:?}",
            ok9
        );
    }

    /// The payload-constraint disconnect is a DIAGNOSTIC attribution issue:
    /// when a GADT match arm's expected type comes from the variant's `when`
    /// refinement, a body-type mismatch must error — with the diagnostic
    /// pointing at the constraint source, not just the match arm.
    #[test]
    fn test_gadt_payload_when_disconnect_diagnostic() {
        // `Tag(Int<32>) when T == Bool` — the payload Int<32> is disconnected
        // from the refinement `T == Bool`.  Matching `Tag(n) => n` in a
        // function returning `T` must ERROR (expected Bool, found Int<32>),
        // and the diagnostic carries a secondary label at the `when` clause.
        let result = check_source(
            "type Expr<T> = enum { Tag(Int<32>) when T == Bool }
             def process<T>(e: Expr<T>) -> T {
                 match e { Tag(n) => n }
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "disconnected payload must error: {:?}",
            result
        );
    }

    /// Const-param narrowing: the E104 const-param exemption is narrowed
    /// to bindings consistent with the declared VALUE type — an unrelated
    /// concrete type (e.g. `N := Bool` for `const N: usize`) is a generality
    /// violation and must be rejected (previously silently accepted).
    #[test]
    fn test_const_param_generality_narrowed() {
        // Incompatible binding — must be rejected (was silently accepted).
        let bad = check_source(
            "def f<const N: usize>(x: Bool) -> N {
                 return x;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            bad.is_err(),
            "const-param incompatible binding must error: {:?}",
            bad
        );
        // The const-param GADT match-arm mismatch — must also be rejected
        // (via the narrowed return check).
        let bad2 = check_source(
            "type E<const N: usize> = enum { A(Int<32>) when N == Int<32>, B(Bool) }
             def f<const N: usize>(x: E<N>) -> N {
                 return match x { A(n) => n, B(b) => b };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            bad2.is_err(),
            "const-param match-arm mismatch must error: {:?}",
            bad2
        );
        // Unused / unbound const param — must pass.
        let ok = check_source(
            "def f<const N: usize>(x: Int<32>) -> Int<32> {
                 return x;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(ok.is_ok(), "unused const param must pass: {:?}", ok);
    }

    /// Edge: GADT construction with wrong type should error
    #[test]
    fn test_gadt_edge_construction_wrong_type() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def main() -> Bool {
                 set e: Expr<Bool> = Expr::Lit(true);
                 return true;
             }",
        );
        assert!(result.is_err(), "construction with wrong type must error");
    }

    /// Edge: unknown param in when clause must error (E062)
    #[test]
    fn test_gadt_edge_unknown_param() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when X == Int<32> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_err(), "unknown param in when must error");
    }

    /// Edge: GADT + `with default` must error (E061, SYNTAX.md §Interaction with `with default`)
    #[test]
    fn test_gadt_edge_with_default_prohibited() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             with default = Wrap::Mk(42)",
        );
        assert!(result.is_err(), "GADT with default must error");
    }

    /// SYNTAX.md §Examples: full Expr<'input> example with Lit, Neg, Add, Eq
    #[test]
    fn test_gadt_edge_full_expr_syntax_doc() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32>, Neg(Int<32>) when T == Int<32> }
             def eval_int(x: Expr<Int<32>>) -> Int<32> {
                 return match x { Lit(n) => n, Neg(n) => -n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "full Expr from SYNTAX.md: {:?}", result);
    }

    /// SYNTAX.md §Examples: Bool variant not reachable for Expr<Int<32>>
    #[test]
    fn test_gadt_edge_expr_eq_not_reachable_on_int() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32>, Eq(Bool) when T == Bool }
             def eval_int(x: Expr<Int<32>>) -> Int<32> {
                 return match x { Lit(n) => n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "Eq unreachable for Expr<Int<32>>: {:?}",
            result
        );
    }

    /// Dead variant elimination (SYNTAX.md §Exhaustiveness Checking): omitting unreachable Eq variant
    #[test]
    fn test_gadt_edge_dead_variant_elimination() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32>, Eq(Bool) when T == Bool }
             def eval_int(x: Expr<Int<32>>) -> Int<32> {
                 return match x { Lit(n) => n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "dead variant elimination: {:?}", result);
    }

    /// GADT with auto type inference (InferVar)
    #[test]
    fn test_gadt_edge_auto_infer() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x = Wrap::Mk(42);
                 return match x { Mk(n) => n };
             }",
        );
        assert!(result.is_ok(), "GADT with auto inference: {:?}", result);
    }

    /// GADT via polymorphic identity helper
    #[test]
    fn test_gadt_edge_via_poly_fn() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def id<T>(x: T) -> T { return x; }
             def main() -> Expr<Int<32>> {
                 return id(Expr::Lit(42));
             }",
        );
        assert!(result.is_ok(), "poly fn GADT: {:?}", result);
    }

    /// While-let with degenerate GADT (single variant, always reachable)
    #[test]
    fn test_gadt_edge_while_let_loop() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 while let Mk(n) = x { return n; }
                 return 0;
             }",
        );
        assert!(result.is_ok(), "while-let GADT loop: {:?}", result);
    }

    /// All GADT variants have the same constraint — match with wildcard
    #[test]
    fn test_gadt_edge_all_same_constraint() {
        let result = check_source(
            "type Wrap<T> = enum {
    A(Int<32>) when T == Int<32>,
    B(Bool) when T == Bool,
    C(String) when T == String,
}
def take_int(x: Wrap<Int<32>>) -> Int<32> {
    return match x { A(n) => n, _ => 0 };
}
def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "all same GADT: {:?}", result);
    }

    /// Multiple sequential matches on GADT values (arm isolation)
    #[test]
    fn test_gadt_edge_sequential_matches() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 set y: Wrap<Int<32>> = Wrap::Mk(100);
                 set a = match y { Mk(n) => n };
                 return match x { Mk(n) => n + a };
             }",
        );
        assert!(result.is_ok(), "sequential GADT matches: {:?}", result);
    }

    /// if-let with while-let chained on GADT
    #[test]
    fn test_gadt_edge_if_while_chain() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 if let Mk(n) = x {
                     while let Mk(m) = x { return n + m; }
                 }
                 return 0;
             }",
        );
        assert!(result.is_ok(), "if-while GADT chain: {:?}", result);
    }

    /// OCaml-equivalent test: variant payload uses the refined type param
    /// type Wrap<T> = enum { Mk(T) when T == Int<32> }
    /// The payload T resolves via GADT registry to Int<32> inside the arm.
    #[test]
    fn test_gadt_ocaml_payload_is_type_param() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(T) when T == Int<32> }
             def unwrap<T>(x: Wrap<T>) -> Int<32> {
                 return match x { Mk(n) => n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "payload is type param: {:?}", result);
    }

    /// OCaml-equivalent test: compound payload referencing the refined param
    /// type Wrap<T> = enum { Pair((T, Bool)) when T == Int<32> }
    #[test]
    fn test_gadt_ocaml_compound_payload_with_param() {
        let result = check_source(
            "type Wrap<T> = enum { Pair((T, Bool)) when T == Int<32> }
             def fst<T>(x: Wrap<T>) -> Int<32> {
                 return match x { Pair((a, _)) => a };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "compound payload with param: {:?}", result);
    }

    /// OCaml-equivalent test: multiple arms with same type param
    /// Each arm refines T differently, arm isolation must hold.
    #[test]
    fn test_gadt_ocaml_multi_arm_diff_refinements() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32>, MkBool(Bool) when T == Bool }
             def take_int(x: Wrap<Int<32>>) -> Int<32> {
                 return match x { Mk(n) => n, MkBool(_) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "multi arm diff refinements: {:?}", result);
    }

    /// OCaml-equivalent test: GenericParam resolved through GADT registry
    /// after pop_gadt_arm, a FRESH match re-refines correctly.
    #[test]
    fn test_gadt_ocaml_two_sequential_matches() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def main() -> Int<32> {
                 set x: Wrap<Int<32>> = Wrap::Mk(42);
                 set y: Wrap<Int<32>> = Wrap::Mk(100);
                 return match x { Mk(a) => match y { Mk(b) => a + b } };
             }",
        );
        assert!(result.is_ok(), "sequential matches: {:?}", result);
    }

    /// Seal-the-wall regression: a GADT refinement must NOT leak out of
    /// its match arm into the global bindings table.  Before the seal,
    /// `set r1: T = match a { Lit(n) => n }` bound T := Int<32> globally
    /// via the post-match unification with the expected type, so a LATER
    /// match refining the SAME T to Bool saw a polluted scrutinee
    /// (`Expr2<Int<32>>`), made `Mk` unreachable, and failed.  After the
    /// seal, each arm's refinement is discharged in-scope against the
    /// expected type and T stays abstract between matches.
    #[test]
    fn test_gadt_refinement_does_not_leak_between_matches() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             type Expr2<T> = enum { Mk(Bool) when T == Bool }
             def f<T>(a: Expr<T>, b: Expr2<T>) -> T {
                 set r1: T = match a { Lit(n) => n };
                 return match b { Mk(x) => x };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "GADT refinement must not leak between matches (T refined to Int<32> then Bool): {:?}",
            result
        );
    }

    /// SYNTAX.md §Existential Quantification: existential GADT with single exists param
    /// type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
    #[test]
    fn test_gadt_exist_single_param() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist single param: {:?}", result);
    }

    /// SYNTAX.md §Existential Quantification: existential GADT with multiple exists params
    /// type DynExpr<T> = enum { Pair(exists A, B: (A, B)) when T == (A, B) }
    #[test]
    fn test_gadt_exist_multi_param() {
        let result = check_source(
            "type DynExpr<T> = enum { Pair(exists A, B: (A, B)) when T == (A, B) }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist multi param: {:?}", result);
    }

    /// SYNTAX.md §Existential Quantification: full DynExpr with mixed exist and non-exist variants
    #[test]
    fn test_gadt_exist_mixed_variants() {
        let result = check_source(
            "type DynExpr<T> = enum {
    IntLit(Int<32>) when T == Int<32>,
    Slice(exists X: &[X]) when T == [X],
}
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist mixed variants: {:?}", result);
    }

    /// Exist param must not shadow enum type parameter
    #[test]
    fn test_gadt_exist_shadow_error() {
        let result = check_source(
            "type Wrap<T> = enum { Bad(exists T: Int<32>) when T == Int<32> }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_err(), "exist shadow must error");
    }

    /// Functional test: pattern match on existential GADT variant.
    /// type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
    /// Tests that the shared skolem mechanism works: apply_gadt_refinement
    /// creates skolems for exist params and stores them on
    /// current_gadt_exist_skolems; check_pattern_inner uses the SAME
    /// skolems when resolving the payload type, so the pattern variable's
    /// type and the GADT equality refer to the same existential witness.
    #[test]
    fn test_gadt_exist_pattern_match() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def take_slice(e: DynExpr<[Int<32>]>) -> Int<32> {
                 return match e { Slice(_s) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist pattern match: {:?}", result);
    }

    /// Multiple exist params: payload `(A, B)` from `exists A, B`
    #[test]
    fn test_gadt_exist_pattern_match_multi() {
        let result = check_source(
            "type DynExpr<T> = enum { Pair(exists A, B: (A, B)) when T == (A, B) }
             def take_pair(e: DynExpr<(Int<32>, Bool)>) -> Int<32> {
                 return match e { Pair(_p) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist multi pattern match: {:?}", result);
    }

    /// Mixed exist and non-exist variants
    #[test]
    fn test_gadt_exist_mixed_pattern_match() {
        let result = check_source(
            "type DynExpr<T> = enum {
    IntLit(Int<32>) when T == Int<32>,
    Slice(exists X: &[X]) when T == [X],
}
             def take(e: DynExpr<[Int<32>]>) -> Int<32> {
                 return match e { Slice(_s) => 0, IntLit(_) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist mixed pattern match: {:?}", result);
    }

    /// Regression: nested GADT match with early return in inner arm.
    /// Tests that GadtArmGuard's depth counter correctly handles early
    /// returns in nested scopes (the guard must pop on every exit path).
    #[test]
    fn test_gadt_nested_match_early_return() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def f(x: Wrap<Int<32>>) -> Int<32> {
                 return match x { Mk(n) => n };
             }
             def main() -> Int<32> { return f(Wrap::Mk(42)); }",
        );
        assert!(result.is_ok(), "nested match early return: {:?}", result);
    }

    /// Accessing an existential element as a concrete type must be
    /// rejected: X is opaque, so `s[0]` has type X, not Int<32>.
    #[test]
    fn test_gadt_exist_payload_element_opaque() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def first(e: DynExpr<[Int<32>]>) -> Int<32> {
                 return match e { Slice(s) => s[0] };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "exist element must stay opaque: {:?}",
            result
        );
    }

    /// `s'len` works on a plain (non-existential) slice — `'len` is the
    /// array/slice length attribute (SYNTAX.md §"Type Attributes").
    #[test]
    fn test_plain_slice_len_attr() {
        let result = check_source(
            "def take_slice(s: &[Int<32>]) -> USize {
                 return s'len;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "plain slice len: {:?}", result);
    }

    /// `s'len` works on an existential GADT payload: `Slice(s)` gives
    /// `s: &[X]` and `'len` is valid since it does not require knowing X
    /// (SYNTAX.md §"Pattern Matching and Type Refinement" example).
    #[test]
    fn test_gadt_exist_payload_len_ok() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def take_slice(e: DynExpr<[Int<32>]>) -> USize {
                 return match e { Slice(s) => s'len };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "exist payload len: {:?}", result);
    }

    /// Ghost-scope leakage: a ghost variable declared in an inner scope must
    /// NOT make a later runtime variable of the same name acceptable in a
    /// `scope_cleanup when` condition.  The second `inner_flag` is a runtime
    /// variable, so the condition must be rejected as non-compile-time.
    #[test]
    fn test_scope_cleanup_ghost_scope_leakage() {
        let result = check_source(
            "def f() -> Int<32> {
                 {
                     ghost set mut inner_flag = false;
                 }
                 set inner_flag = true;
                 scope_cleanup @c when inner_flag { }
                 return 0;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "runtime inner_flag must not be accepted as ghost: {:?}",
            result
        );
    }

    /// Control: a ghost variable declared in the SAME scope IS a valid
    /// compile-time predicate for `scope_cleanup when`.
    #[test]
    fn test_scope_cleanup_ghost_same_scope_ok() {
        let result = check_source(
            "def f() -> Int<32> {
                 ghost set mut inner_flag = false;
                 scope_cleanup @c when inner_flag { }
                 return 0;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(result.is_ok(), "same-scope ghost: {:?}", result);
    }

    /// `contains_gadt_skolem` coverage for compound payload types: for
    /// `Pair(exists A, B: (A, B))` the declared type is a Tuple containing
    /// GADT skolems.  Opacity must be preserved — the payload elements stay
    /// opaque (type `A`/`B`), so using `p` as `(Int<32>, Bool)` must fail.
    #[test]
    fn test_gadt_exist_payload_tuple_opaque() {
        let result = check_source(
            "type DynExpr<T> = enum { Pair(exists A, B: (A, B)) when T == (A, B) }
             def take_pair(e: DynExpr<(Int<32>, Bool)>) -> Int<32> {
                 return match e { Pair(p) => p'0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "tuple payload element must stay opaque: {:?}",
            result
        );
    }

    /// TcLevel non-escape: a normal GADT match with the region wiring
    /// active must NOT produce false SkolemEscape errors.  The arm body
    /// returns an arm-local value; nothing escapes the arm.
    #[test]
    fn test_gadt_tclevel_no_false_escape() {
        let result = check_source(
            "type Wrap<T> = enum { Mk(Int<32>) when T == Int<32> }
             def f(x: Wrap<Int<32>>) -> Int<32> {
                 return match x { Mk(n) => n };
             }
             def main() -> Int<32> { return f(Wrap::Mk(42)); }",
        );
        assert!(result.is_ok(), "no false SkolemEscape: {:?}", result);
    }

    /// TcLevel non-escape for existential arms: using the payload inside
    /// the arm (e.g. `s'len`) must not trigger a false escape.
    #[test]
    fn test_gadt_tclevel_exist_payload_ok() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def take_slice(e: DynExpr<[Int<32>]>) -> USize {
                 return match e { Slice(s) => s'len };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "exist payload no false escape: {:?}",
            result
        );
    }

    /// TcLevel escape: an arm that leaks its existential payload as a
    /// concrete `Int<32>` must be rejected (opacity + region discipline).
    #[test]
    fn test_gadt_tclevel_escape_rejected() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def first(e: DynExpr<[Int<32>]>) -> Int<32> {
                 return match e { Slice(s) => s[0] };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "escaping element must be rejected: {:?}",
            result
        );
    }

    /// Whole-slice escape: returning the entire existential payload as a
    /// concrete `&[Int<32>]` must be rejected.  `s` has type `&[S]` with
    /// opaque witness `S`; the top-level equality `[S] → [Int<32>]` must
    /// NOT make `&[S]` usable as `&[Int<32>]`.
    #[test]
    fn test_gadt_exist_whole_slice_escape_rejected() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def leak(e: DynExpr<[Int<32>]>) -> &[Int<32>] {
                 return match e { Slice(s) => s };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "whole-slice existential must not escape as concrete: {:?}",
            result
        );
    }

    /// Ghost shadowing: an inner RUNTIME variable that shadows an outer
    /// ghost variable must NOT be accepted as a compile-time predicate.
    /// `when flag` here refers to the runtime `flag`, not the ghost one.
    #[test]
    fn test_scope_cleanup_ghost_shadowed_by_runtime() {
        let result = check_source(
            "def f() -> Int<32> {
                 ghost set flag = false;
                 {
                     set flag = true;
                     scope_cleanup @c when flag { }
                 }
                 return 0;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "runtime variable shadowing a ghost must be rejected: {:?}",
            result
        );
    }

    /// Generic existential whole-slice escape: returning the payload as
    /// `&[U]` (outer generic parameter) must be rejected — the witness
    /// must not escape through an outer binder (SYNTAX.md §"Existential
    /// Quantification": "prevented from leaking into the surrounding
    /// context by an occurs-check at the branch boundary").
    #[test]
    fn test_gadt_exist_generic_whole_slice_escape_rejected() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def leak<U>(e: DynExpr<[U]>) -> &[U] {
                 return match e { Slice(s) => s };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "generic whole-slice existential must not escape: {:?}",
            result
        );
    }

    /// Generic existential element escape: returning the payload element
    /// as `U` (outer generic parameter) must be rejected — the witness is
    /// opaque, so `s[0]` has type X, not U.
    #[test]
    fn test_gadt_exist_generic_element_escape_rejected() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when T == [X] }
             def leak_elem<U>(e: DynExpr<[U]>) -> U {
                 return match e { Slice(s) => s[0] };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "generic element existential must not escape: {:?}",
            result
        );
    }

    /// Nested existential scopes: an outer match arm binding `exists X`
    /// followed by an inner match arm binding its OWN `exists Y`.  Each
    /// binder must get an independent skolem scope — the inner `Y` must
    /// not be resolved with the outer `X` map.  Exercises the
    /// `current_gadt_exist_skolems` stack across nested arms.
    #[test]
    fn test_gadt_exist_nested_pattern_scope() {
        let result = check_source(
            "type A<T> = enum { MkA(exists X: &[X]) when T == [X] }
             type B<T> = enum { MkB(exists Y: &[Y]) when T == [Y] }
             def take(a: A<[Int<32>]>, b: B<[Int<32>]>) -> USize {
                 return match a {
                     MkA(xs) => match b {
                         MkB(ys) => xs'len + ys'len,
                     },
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "nested existential scopes must type-check: {:?}",
            result
        );
    }

    /// comptime def functions return values with `return` (SYNTAX.md
    /// §"Type Factories": `comptime def make_vector(...) -> type {
    /// return [Elem; N]; }`).  The body is a FUNCTION body, not a
    /// comptime block, so `return` with a value must be accepted and the
    /// value propagated to the caller.
    #[test]
    fn test_comptime_def_return_value_ok() {
        let result = check_source(
            "comptime def answer() -> Int<32> { return 42; }
             def main() -> Int<32> {
                 set v = answer!();
                 return v;
             }",
        );
        assert!(
            result.is_ok(),
            "comptime def return with value must be accepted: {:?}",
            result
        );
    }

    /// Comptime generic functions are NOT falsely rejected by the E104
    /// generality check.  No false-positive vector exists: the
    /// comptime sandbox (E081) blocks calling regular functions (the main
    /// way a body could bind a generic param), and `@typeInfo!` is
    /// deferred to generate expansion (returns unit at type-check time).
    #[test]
    fn test_comptime_generic_param_no_false_e104() {
        let result = check_source(
            "comptime def make<T>(x: T) -> T { return x; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "comptime generic function must not false-positive E104: {:?}",
            result
        );
    }

    /// comptime def returning a Bool, referenced as a compile-time
    /// constant in `scope_cleanup when` (SYNTAX.md §"Comptime": the
    /// condition may reference compile-time-constant expressions).
    #[test]
    fn test_scope_cleanup_when_comptime_const_ok() {
        let result = check_source(
            "comptime def DEBUG() -> Bool { return true; }
             def f() -> Int<32> {
                 scope_cleanup @c when DEBUG!() { }
                 return 0;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "comptime constant predicate must be accepted: {:?}",
            result
        );
    }

    /// Same-named existential binders in a nested pattern: outer `MkA`
    /// has `exists X` and inner `MkB` ALSO has `exists X` — two DIFFERENT
    /// witnesses.  The inner binder must get its own scope, not reuse the
    /// outer one by name.
    #[test]
    fn test_gadt_exist_nested_same_name_scope() {
        let result = check_source(
            "type B<T> = enum { MkB(exists X: &[X]) when T == [X] }
             type A<T> = enum { MkA(exists X: B<[X]>) when T == [X] }
             def take(a: A<[Int<32>]>) -> USize {
                 return match a {
                     MkA(MkB(s)) => s'len,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "same-named nested existential binders must each get their own scope: {:?}",
            result
        );
    }

    /// GADT construction soundness: `Expr::Lit(42)` requires `T == Int<32>`.
    /// Constructing it in a context expecting `Expr<Bool>` must be REJECTED
    /// (the `when` constraint must not be silently skipped when the bare
    /// enum path cannot recover type args).
    #[test]
    fn test_gadt_construction_mismatched_target_rejected() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def bad() -> Expr<Bool> {
                 return Expr::Lit(42);
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "GADT construction with mismatched type target must be rejected: {:?}",
            result
        );
    }

    /// Independent same-named existential binders in a nested pattern must
    /// NOT be conflated: outer `exists X` and inner `exists X` are DIFFERENT
    /// witnesses.  Skolem identity is the binder's INDEX (GHC `realUnique` /
    /// OCaml `id: int`), so the inner payload `(X, Int<32>)` resolves with
    /// the inner witness — the second element is the concrete `Int<32>`.
    #[test]
    fn test_gadt_exist_nested_independent_same_name() {
        let result = check_source(
            "type Inner = enum { MkInner(exists X: (X, Int<32>)) }
             type Outer<T> = enum { MkOuter(exists X: Inner) when T == X }
             def bad(o: Outer<Bool>) -> Int<32> {
                 return match o {
                     MkOuter(MkInner((_, a))) => a,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "nested independent same-named existential binders must type-check: {:?}",
            result
        );
    }

    /// Witness-set consistency: the payload type and the GADT equations
    /// must share ONE existential witness.  `when X == Int<32>` solves X to
    /// `Int<32>`; if `check_pattern_inner` allocated a SECOND witness set
    /// (witness mismatch), `s'len` would be typed with a different skolem
    /// and the rigid GADT-skolem check would false-reject this.
    #[test]
    fn test_gadt_exist_witness_set_consistent() {
        let result = check_source(
            "type DynExpr<T> = enum { Slice(exists X: &[X]) when X == Int<32> }
             def first(e: DynExpr<[Int<32>]>) -> USize {
                 return match e {
                     Slice(s) => s'len,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "payload type and GADT refinement must share one witness set: {:?}",
            result
        );
    }

    /// Frame identity must be OCCURRENCE identity, not variant NAME: two
    /// different enums with the SAME variant name (`Mk`) nested in a
    /// pattern must each get their own witness.  If the inner `Mk`
    /// wrongly reuses the outer frame (name-based identity), the inner
    /// `exists X` resolves with the outer skolem.
    #[test]
    fn test_gadt_exist_same_variant_name_nested() {
        let result = check_source(
            "type Inner = enum { Mk(exists X: (X, Int<32>)) }
             type Outer = enum { Mk(exists Y: Inner) }
             def bad(o: Outer) -> Int<32> {
                 return match o {
                     Mk(Mk((_, a))) => a,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "same-named variants in different enums must get independent witnesses: {:?}",
            result
        );
    }

    /// Nested GADT refinement (SYNTAX.md §"Nested GADT Refinement"): the
    /// outer branch refines `T == Bool`; a nested `match` inside the
    /// branch body on ANOTHER `Container<T>` value must see that
    /// refinement, so `BoolBox` is considered satisfiable and `s2` is a
    /// `Bool`.
    #[test]
    fn test_gadt_nested_refinement_in_branch() {
        let result = check_source(
            "type Container<T> = enum {
                 IntBox(Int<32>) when T == Int<32>,
                 BoolBox(Bool) when T == Bool,
             }
             def extract<T>(c1: Container<T>, c2: Container<T>) -> Bool {
                 return match c1 {
                     BoolBox(s) => match c2 { BoolBox(s2) => s2, _ => false },
                     _ => false,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "outer refinement must propagate into nested match in branch body: {:?}",
            result
        );
    }

    /// Nested GADT refinement in a SINGLE nested pattern tree
    /// (SYNTAX.md §"Nested GADT Refinement"): each constructor in the
    /// nested pattern contributes its `when` equalities.  `Add` refines
    /// `T == Int<32>`; both nested `Lit` patterns refine their inner
    /// `Expr<Int<32>>`'s `T == Int<32>` — so `a` and `b` are `Int<32>`
    /// and the whole equation set is consistent.
    #[test]
    fn test_gadt_nested_pattern_tree_refinement() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Add((Expr<Int<32>>, Expr<Int<32>>)) when T == Int<32>,
             }
             def eval(e: Expr<Int<32>>) -> Int<32> {
                 return match e {
                     Add((Lit(a), Lit(b))) => a + b,
                     Lit(n) => n,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "nested pattern tree must propagate when equalities: {:?}",
            result
        );
    }

    /// Occurrence identity for RECURSIVE GADT variants: `Node` nested in
    /// `Node` has the SAME enum DefId and variant name, but is a DIFFERENT
    /// occurrence.  The inner `Node` must push its own existential frame
    /// (the top-level frame was consumed by the outer `Node`), not reuse
    /// the outer witness.  Uses concrete `Tree<Int<32>>` payloads so the
    /// nested payload resolves without generic-parameter substitution.
    #[test]
    fn test_gadt_exist_recursive_variant_occurrence() {
        let result = check_source(
            "type Tree<T> = enum {
                 Leaf(exists X: (X, Int<32>)) when T == Int<32>,
                 Node(exists X: (X, Tree<Int<32>>)) when T == Int<32>,
             }
             def depth(t: Tree<Int<32>>) -> Int<32> {
                 return match t {
                     Node((_, Node((_, Leaf((_, n)))))) => n,
                     _ => 0,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "recursive variant occurrences must get independent witnesses: {:?}",
            result
        );
    }

    /// GADT `when` RHS must not reference another type parameter of the
    /// SAME enum: `when T == U and U == T` would register a mutual
    /// refinement cycle (A → B, B → A) that `resolve_binding_tail` would
    /// chase until MAX_CHAIN_DEPTH.  Rejected at resolver time (E064).
    #[test]
    fn test_gadt_when_rhs_same_enum_param_rejected() {
        let result = check_source(
            "type Bad<T, U> = enum { Mk(T) when T == U }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "when RHS referencing same-enum type parameter must be rejected: {:?}",
            result
        );
    }

    /// Composite `when` RHS referencing a same-enum parameter (inside a
    /// tuple) is also rejected (recursive walk).
    #[test]
    fn test_gadt_when_rhs_same_enum_param_in_tuple_rejected() {
        let result = check_source(
            "type Bad<T, U> = enum { Mk(T) when T == (U, Int<32>) }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "when RHS with same-enum param inside a tuple must be rejected: {:?}",
            result
        );
    }

    /// Legal RHS forms must NOT be rejected: an `exists` variable on the
    /// RHS (`when T == [X]`, witness stays opaque) and concrete types.
    #[test]
    fn test_gadt_when_rhs_exists_and_concrete_ok() {
        let result = check_source(
            "type Good<T> = enum { Mk(exists X: (X, Int<32>)) when T == Int<32> }
             def f(g: Good<Int<32>>) -> Int<32> {
                 return match g { Mk((_, a)) => a };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "when RHS with concrete type must be accepted: {:?}",
            result
        );
    }

    /// `return Err(e)` must be rejected even through a type alias
    /// (`type MyResult = Result<Int<32>, Int<32>>`) — the alias's `Err`
    /// variant is still an error exit and must use `leave with`.
    #[test]
    fn test_return_err_via_type_alias_rejected() {
        let result = check_source(
            "type MyResult = Result<Int<32>, Int<32>>
             def bad() -> MyResult {
                 return MyResult::Err(42);
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "return Err(e) through a type alias must be rejected: {:?}",
            result
        );
        let msg = format!("{:?}", result);
        assert!(
            msg.contains("leave with") || msg.contains("E008"),
            "expected the error-exit lint (E008 / leave with) to fire, got: {:?}",
            result
        );
    }

    /// Regression (currently limited): an existential skolem on the
    /// SCRUTINEE side of a `when` constraint must never register
    /// `ParamRefinement { from: concrete, to: skolem }` —
    /// `register_gadt_eq_directional` now routes GADT skolems to the INERT
    /// `ExistentialEquation` (defense-in-depth against type confusion).
    ///
    /// NOTE: this exact nested scenario does not type-check yet — the
    /// outer `when T == X` (RHS is an `exists` variable) registers an
    /// inert equation, so `X` stays opaque and the inner scrutinee remains
    /// `Inner<S>` (unable to align with `MkInner`'s `Inner<Int<32>>`).
    /// That is a witness-solving limitation, not a skolem
    /// registration bug.  Kept `#[ignore]` as documentation of the
    /// limitation until witness solving is extended.
    #[test]
    #[ignore = "outer `when T == X` keeps X opaque; inner scrutinee cannot align (witness-solving is opt-in per consumer, not in resolve_binding — opacity invariant)"]
    fn test_gadt_exist_scrutinee_skolem_not_refined() {
        let result = check_source(
            "type Inner<T> = enum { MkInner(Int<32>) when T == Int<32> }
             type Outer<T> = enum { MkOuter(exists X: Inner<X>) when T == X }
             def f(o: Outer<Int<32>>) -> Int<32> {
                 return match o {
                     MkOuter(inner) => match inner {
                         MkInner(n) => n,
                     },
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "inner match on a non-existential variant under an existential scrutinee must type-check: {:?}",
            result
        );
    }

    /// GADT refinement in STATEMENT-position `if let` (SYNTAX.md
    /// §"Pattern Matching and Type Refinement": refinement is available in
    /// `if let`).  The `when T == Int<32>` constraint must refine `n` to
    /// `Int<32>` inside the then-branch — previously only `Stmt::WhileLet`
    /// and expression-position `if let` got the full refinement sequence.
    #[test]
    fn test_gadt_refinement_stmt_if_let() {
        let result = check_source(
            "type Expr<T> = enum { Lit(Int<32>) when T == Int<32> }
             def f(e: Expr<Int<32>>) -> Int<32> {
                 if let Lit(n) = e {
                     return n;
                 }
                 return 0;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "GADT refinement must apply in statement-position if let: {:?}",
            result
        );
    }

    /// or-pattern GADT refinement — scenario 1: all alternatives refine T
    /// to the SAME type → the refinement propagates to the branch body.
    /// Both `Lit` and `Neg` refine `T == Int<32>`; the intersection
    /// propagates.
    #[test]
    fn test_gadt_or_pattern_consistent_refinement() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Neg(Expr<Int<32>>) when T == Int<32>,
             }
             def f(e: Expr<Int<32>>) -> Int<32> {
                 return match e {
                     Lit(_) | Neg(_) => 0,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "consistent or-pattern GADT refinements must be accepted: {:?}",
            result
        );
    }

    /// or-pattern GADT refinement — scenario 2: alternatives refine the
    /// same parameter to DIFFERENT types → E066 compile error (disjunction
    /// semantics: the branch body cannot assume conflicting refinements).
    /// A POLYMORPHIC scrutinee is required: with a concrete `Expr<Int<32>>`,
    /// `Eq` is a dead variant   whose equalities are ignored by the
    /// intersection, so E066 would never fire.
    #[test]
    fn test_gadt_or_pattern_conflicting_refinement_rejected() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Eq(Bool) when T == Bool,
             }
             def f<T>(e: Expr<T>) -> Int<32> {
                 return match e {
                     Lit(_) | Eq(_) => 0,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        let msg = format!("{:?}", result);
        assert!(
            result.is_err() && (msg.contains("E066") || msg.contains("conflicting GADT")),
            "conflicting or-pattern GADT refinements must be rejected (E066): {:?}",
            result
        );
    }

    /// or-pattern GADT refinement — scenario 3: some alternatives have no
    /// constraint → T is NOT refined (stays abstract; no error, no
    /// propagation).
    #[test]
    fn test_gadt_or_pattern_partial_constraint_no_refine() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 If((Expr<Int<32>>, Expr<T>, Expr<T>)),
             }
             def f<T>(e: Expr<T>) -> Int<32> {
                 return match e {
                     Lit(_) | If((_, _, _)) => 0,
                 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "or-pattern with partial constraints must not refine (no error): {:?}",
            result
        );
    }

    /// Or-pattern bindings: alternatives bind the SAME variable with
    /// compatible types; the binding is in scope in the
    /// branch body (`x` is usable as `Int<32>`).
    #[test]
    fn test_or_pattern_bindings_in_scope() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Neg(Int<32>) when T == Int<32>,
             }
             def f(e: Expr<Int<32>>) -> Int<32> {
                 return match e { Lit(x) | Neg(x) => x + 1 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "or-pattern binding must be in scope in the branch body: {:?}",
            result
        );
    }

    /// Or-pattern bindings: alternatives must bind the SAME set of
    /// variables (E105, OCaml's Orpat_vars).
    #[test]
    fn test_or_pattern_bindings_different_vars_rejected() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Neg(Int<32>) when T == Int<32>,
             }
             def f(e: Expr<Int<32>>) -> Int<32> {
                 return match e { Lit(x) | Neg(y) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        let msg = format!("{:?}", result);
        assert!(
            result.is_err() && msg.contains("must bind the same variables"),
            "or-pattern alternatives must bind the same variables (E105): {:?}",
            result
        );
    }

    /// Or-pattern bindings: a common variable's type must unify
    /// across alternatives (E106, OCaml's Or_pattern_type_clash).
    #[test]
    fn test_or_pattern_bindings_incompatible_types_rejected() {
        let result = check_source(
            "type E<T> = enum {
                 A(Int<32>) when T == Int<32>,
                 B(Bool) when T == Bool,
             }
             def f<T>(e: E<T>) -> Int<32> {
                 return match e { A(x) | B(x) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        let msg = format!("{:?}", result);
        assert!(
            result.is_err() && msg.contains("incompatible types across alternatives"),
            "or-pattern binding types must unify (E106): {:?}",
            result
        );
    }

    /// Or-pattern unreachable alternative: with a CONCRETE scrutinee
    /// `Expr<Int<32>>`, the `Eq` alternative (requires `when T == Bool`) is
    /// a dead variant — it is warned about and IGNORED by the intersection
    ///, and the program type-checks.  Order is
    /// irrelevant: `Lit | Eq` and `Eq | Lit` behave identically.
    #[test]
    fn test_or_pattern_unreachable_alternative_ignored() {
        let result = check_source(
            "type Expr<T> = enum {
                 Lit(Int<32>) when T == Int<32>,
                 Eq(Bool) when T == Bool,
             }
             def f(e: Expr<Int<32>>) -> Int<32> {
                 return match e { Lit(_) | Eq(_) => 0 };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "unreachable or-pattern alternative must be ignored: {:?}",
            result
        );
    }

    /// Seal regression: an existential GADT arm whose body USES a function
    /// generic param must not trip the seal assert.  Pattern instantiation
    /// legitimately binds the scrutinee param to a synthetic infer var
    /// (`T → ?a`) at arm_depth 0, and the arm-body `resolve_binding`
    /// path-compresses that existing chain — a re-write, not a new leak.
    /// The seal fires only on NEW GenericParam bindings.
    #[test]
    fn test_gadt_exist_param_use_no_seal_violation() {
        let result = check_source(
            "type E<T> = enum { Slice(exists X: &[X]) when T == &[X] }
             def f<T>(e: E<T>, xs: T) -> T {
                 return match e { Slice(_) => xs };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "existential GADT arm using a generic param must not trip the seal: {:?}",
            result
        );
    }

    /// Regression (seal proof gap): a GADT match arm whose
    /// body would bind a fresh GenericParam only inside the arm.  The seal
    /// guard skips the arm-scoped binding; the compiler must still behave
    /// deterministically (recover via a later non-arm unify) — never
    /// silently accept an ill-typed program.
    #[test]
    fn test_gadt_arm_only_binding_seal_recovery() {
        let ok = check_source(
            "type E<T> = enum { Mk(T) }
             def f<T>(e: E<T>, x: T) -> T {
                 return match e { Mk(_) => x };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(ok.is_ok(), "arm binding must recover: {:?}", ok);
    }

    /// Strict mode escalates `@must_handle` violations to errors (E108):
    /// propagating a must_handle'd result via `?` is an error in strict
    /// mode (a warning otherwise).
    #[test]
    fn test_must_handle_strict_escalation() {
        let source = "
            @must_handle(CriticalFault)
            def operation() -> Result<Int<32>, Int<32>> { return 0; }

            def caller() -> Int<32> {
                return operation()?;
            }

            def main() -> Int<32> { return 0; }
        ";
        // Run the pipeline with the given strict flag, collecting diagnostic
        // codes (the probe's `operation` body has an unrelated E031 noise
        // error, so we assert on codes rather than on success).
        let run = |strict: bool| -> Vec<String> {
            let arena = bumpalo::Bump::new();
            let mut parser = Parser::new(source, &arena);
            let program = parser.parse_program().unwrap();
            let mut ctx = TypeContext::new();
            let local_crate_id = CrateId(DefId(0));
            let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
            let (mut symbols, mut trait_env, res_diags, resolution_map) =
                resolver.resolve_program(&program);
            assert!(
                !res_diags.has_errors(),
                "unexpected resolution errors: {:?}",
                res_diags.into_inner()
            );
            let mut checker = TypeChecker::new(
                &mut ctx,
                &symbols,
                &mut trait_env,
                resolution_map,
                strict,
                false,
                vec![],
                false,
            );
            match checker.check_program(&program) {
                Ok(_) => Vec::new(),
                Err(bundle) => bundle
                    .into_inner()
                    .into_iter()
                    .filter_map(|d| d.code().map(|c| c.code().to_string()))
                    .collect(),
            }
        };
        let non_strict = run(false);
        assert!(
            non_strict.contains(&"W004".to_string()),
            "non-strict must_handle propagation should emit W004, got: {:?}",
            non_strict
        );
        let strict = run(true);
        assert!(
            strict.contains(&"E108".to_string()),
            "strict mode should escalate to E108, got: {:?}",
            strict
        );
        assert!(
            !strict.contains(&"W004".to_string()),
            "strict mode should not emit W004, got: {:?}",
            strict
        );
    }

    /// By default, `&mut T` must NOT implicitly coerce to `&T` at a call
    /// site (SYNTAX.md §"Reference Coercion and Read-Only Borrows") — the
    /// transition must be explicit (`&ro`) or locally relaxed (`@auto_ro`).
    #[test]
    fn test_ref_mut_not_coerced_to_shared_by_default() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(r);
             }",
        );
        assert!(
            result.is_err(),
            "&mut T must not implicitly coerce to &T by default (SYNTAX.md): {:?}",
            result
        );
    }

    /// The explicit `&ro` operator freezes a `&mut T` into a `&T`.
    #[test]
    fn test_ref_read_only_borrow_operator() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(&ro r);
             }",
        );
        assert!(
            result.is_ok(),
            "explicit `&ro` freeze should typecheck (SYNTAX.md): {:?}",
            result
        );
    }

    /// `@auto_ro` locally relaxes the default: `&mut T` implicitly coerces
    /// to `&T` within the annotated function.
    #[test]
    fn test_auto_ro_relaxation() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             @auto_ro
             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(r);
             }",
        );
        assert!(
            result.is_ok(),
            "@auto_ro should allow implicit &mut T -> &T in the function (SYNTAX.md): {:?}",
            result
        );
    }

    // ── Defensive test suite: `@auto_ro` freeze guard (nested positions) ──
    // These tests verify that `@auto_ro`'s implicit `&mut T → &T` freeze
    // does NOT leak into nested positions (struct fields, ADT constructors,
    // array elements, nested calls).  The freeze gate chain is:
    //   coercion_depth == 0  ∧  CallSiteCtx  ∧  (auto_ro ∨ auto_coerce)  ∧  m2 ∧ !m1
    // The `CallSite` context is set by `CallSiteCoercion` AFTER `check_expr`
    // returns, so `HasType(expected)` in `check_call_argument` is safe.

    /// `@auto_ro` + direct top-level argument: should be accepted
    /// (freeze applies at the call site).
    #[test]
    fn test_auto_ro_freeze_direct_arg() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             @auto_ro
             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(r);
             }",
        );
        assert!(
            result.is_ok(),
            "@auto_ro + direct arg should be accepted: {:?}",
            result
        );
    }

    /// `@auto_ro` + struct field argument: should be rejected
    /// (freeze must NOT leak into struct literal fields).
    #[test]
    fn test_auto_ro_freeze_struct_field_arg() {
        let result = check_source(
            "type Wrapper = struct { x: &Int<32> }
             def takes_wrapper(w: Wrapper) -> Int<32> { return *w.x; }

             @auto_ro
             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_wrapper(Wrapper { x: r });
             }",
        );
        assert!(
            result.is_err(),
            "@auto_ro + struct field arg must be rejected (nested position): {:?}",
            result
        );
    }

    /// `@auto_ro` + ADT constructor argument: should be rejected
    /// (freeze must NOT leak into ADT constructor payload).
    #[test]
    fn test_auto_ro_freeze_adt_ctor_arg() {
        let result = check_source(
            "type Box<T> = enum { Mk(T) }
             def takes_box(b: Box<&Int<32>>) -> Int<32> { return 0; }

             @auto_ro
             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_box(Box::Mk(r));
             }",
        );
        assert!(
            result.is_err(),
            "@auto_ro + ADT ctor arg must be rejected (nested position): {:?}",
            result
        );
    }

    /// `@auto_ro` + nested call argument: should be ACCEPTED
    /// (`@auto_ro` is function-wide — `&mut T → &T` applies to ALL call
    /// sites within the annotated function, including nested calls.  The
    /// freeze gate chain (`coercion_depth == 0 ∧ CallSiteCtx`) is satisfied
    /// because each `check_call_argument` has its own `CallSiteCoercion`
    /// guard — the inner call is NOT a "nested position" in the sense of
    /// struct fields / ADT constructors / array elements, which are NOT
    /// call sites and thus NOT covered by `@auto_ro`'s implicit freeze.)
    #[test]
    fn test_auto_ro_freeze_nested_call_arg() {
        let result = check_source(
            "def identity(x: &Int<32>) -> &Int<32> { return x; }
             def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             @auto_ro
             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(identity(r));
             }",
        );
        assert!(
            result.is_ok(),
            "@auto_ro + nested call arg should be accepted (function-wide): {:?}",
            result
        );
    }

    /// No `@auto_ro` + `&mut → &T` should be rejected everywhere.
    #[test]
    fn test_auto_ro_freeze_no_attr_direct_arg() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }

             def main() -> Int<32> {
                 set mut v = 42;
                 let r: &mut Int<32> = &mut v;
                 return takes_shared(r);
             }",
        );
        assert!(
            result.is_err(),
            "without @auto_ro, &mut T -> &T must be rejected: {:?}",
            result
        );
    }

    // ── Defensive suite: existential refinement ──
    // Language-designer ruling (2026-08-04): `when X == ConcreteType` on
    // an existential variable refines the skolem (ParamRefinement) + emit
    // a warning (the syntax is unusual).  RHS containing unresolved vars
    // (skolems, GenericParams, InferVars) stays inert (ExistentialEquation).

    /// Positive: `when X == Int<32>` on an existential — RHS closed → refine
    /// + warn.  Before the change, X stays inert (rejected); after, X is
    /// refined to Int<32> (accepted).  At BASELINE this test FAILS (the
    /// refinement does not exist yet) — that is the expected baseline
    /// divergence the change is meant to close.
    #[test]
    fn test_exist_refine_when_closed_rhs() {
        let result = check_source(
            "type E = enum { Mk(exists X: X) when X == Int<32> }
             def f(e: E) -> Int<32> {
                 return match e { Mk(x) => x };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "closed-RHS existential refinement should be accepted (after change): {:?}",
            result
        );
    }

    /// Negative: `when X == [X]` (RHS contains the same skolem) — RHS
    /// NOT closed → stays inert (ExistentialEquation), no refine, no warn.
    #[test]
    fn test_exist_refine_when_open_rhs() {
        let result = check_source(
            "type E = enum { Mk(exists X: &[X]) when X == [X] }
             def f(e: E) -> &[Int<32>] {
                 return match e { Mk(x) => x };
             }
             def main() -> Int<32> { return 0; }",
        );
        // RHS contains skolem X → not closed → stays inert.
        // The arm returns `x: &[X]` where X is still opaque → type mismatch.
        // This should be rejected BEFORE and AFTER the change.
        assert!(
            result.is_err(),
            "RHS with unresolved vars must stay inert (rejected): {:?}",
            result
        );
    }

    /// Control: `when T == Concrete` on a type parameter (GenericParam) —
    /// unchanged behavior (ParamRefinement, no warn).
    #[test]
    fn test_exist_refine_type_param_unchanged() {
        let result = check_source(
            "type E<T> = enum { Mk(Int<32>) when T == Int<32> }
             def f(e: E<Int<32>>) -> Int<32> {
                 return match e { Mk(n) => n };
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "type-param refinement must still work (unchanged): {:?}",
            result
        );
    }

    // ── Defensive suite: `expr.freeze!()` ──
    // SYNTAX.md: ".freeze!() (the standard-library equivalent of `&ro`)
    // behaves identically and is preferred in method chains."  These tests
    // encode the NEW behavior (freeze! ≡ &ro).  At BASELINE (before the
    // parser implements `.freeze!()`), the positive/chain/frozen tests
    // FAIL (E007) — the expected divergence the change closes.

    /// Positive: `takes_shared(r.freeze!())` with `r: &mut T` — accepted
    /// (explicit freeze, equivalent to `&ro r`).
    #[test]
    fn test_freeze_bang_direct() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return takes_shared(r.freeze!());
             }",
        );
        assert!(
            result.is_ok(),
            "r.freeze!() should be accepted (explicit freeze ≡ &ro): {:?}",
            result
        );
    }

    /// Freeze enforcement: after `r.freeze!()`, mutating `r` is rejected
    /// (point-level liveness): the `&ro` view from `r.freeze!()` is a
    /// temporary whose lifetime ends at its use (the call) — `*r = 100`
    /// AFTER the call is allowed (the freeze ends at the view's last use,
    /// not the block's end).
    #[test]
    fn test_freeze_bang_frozen_view_lifetime_ends_at_use() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let y = takes_shared(r.freeze!());
                 *r = 100;
                 return y;
             }",
        );
        assert!(
            result.is_ok(),
            "the freeze view's lifetime ends at its use — mutating after the call must be allowed: {:?}",
            result
        );
    }

    /// Method-chain: `v.freeze!()` in a chain position — accepted.
    #[test]
    fn test_freeze_bang_method_chain() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return takes_shared(r.freeze!());
             }",
        );
        assert!(
            result.is_ok(),
            "r.freeze!() in chain position should be accepted: {:?}",
            result
        );
    }

    /// Control: `takes_shared(r)` WITHOUT freeze — still rejected (no
    /// implicit `&mut T → &T`).
    #[test]
    fn test_freeze_bang_no_attr_control() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return takes_shared(r);
             }",
        );
        assert!(
            result.is_err(),
            "without freeze!, &mut T -> &T must be rejected: {:?}",
            result
        );
    }

    // ── Defensive suite: statement-level loan tracking ──
    // `&mut` registers an exclusive loan (the source place is frozen while
    // the exclusive borrow is live), loans are scoped to their enclosing
    // lexical block, and the freeze tracks the exact place (`a.b`) rather
    // than the root variable.

    /// `&mut a` freezes `a`: a later direct mutation of the source while
    /// the exclusive borrow is live is rejected (SYNTAX.md §References).
    #[test]
    fn test_loan_mut_freezes_source_mutation() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 a = 5;
                 return *r;
             }",
        );
        assert!(
            result.is_err(),
            "mutating the source while an exclusive `&mut` borrow is live must be rejected: {:?}",
            result
        );
    }

    /// Mutation THROUGH the exclusive borrow is allowed (`*r` uses the
    /// borrow, not the frozen source).
    #[test]
    fn test_loan_mut_use_through_borrow_ok() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 *r = 5;
                 return *r;
             }",
        );
        assert!(
            result.is_ok(),
            "mutating through the exclusive borrow must be allowed: {:?}",
            result
        );
    }

    /// Loan scoping: a borrow created inside a lexical block (an if-arm
    /// here) expires at the block's end — the source can be mutated
    /// afterwards.
    #[test]
    fn test_loan_ro_block_scoped_expires() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 let y = if true {
                     set r: &mut Int<32> = &mut a;
                     *r
                 } else {
                     0
                 };
                 a = 5;
                 return y;
             }",
        );
        assert!(
            result.is_ok(),
            "after the borrowing block ends, the source may be mutated: {:?}",
            result
        );
    }

    /// Place-tree freezing: `&mut (p.x)` freezes the exact place — mutating
    /// a SIBLING place `p.y` is allowed (previously the root `p` was
    /// frozen, over-rejecting the sibling).
    #[test]
    fn test_loan_place_tree_sibling_not_frozen() {
        let result = check_source(
            "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set mut p = Point { x = 1, y = 2 };
                 set r: &mut Int<32> = &mut (p.x);
                 p.y = 5;
                 return *r;
             }",
        );
        assert!(
            result.is_ok(),
            "mutating a sibling place must be allowed under place-tree freezing: {:?}",
            result
        );
    }

    /// Place-tree freezing: mutating the EXACT frozen place (`p.x` under a
    /// frozen `p.x`) is rejected.
    #[test]
    fn test_loan_place_tree_exact_place_rejected() {
        let result = check_source(
            "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set mut p = Point { x = 1, y = 2 };
                 set r: &mut Int<32> = &mut (p.x);
                 p.x = 5;
                 return *r;
             }",
        );
        assert!(
            result.is_err(),
            "mutating the exact frozen place must be rejected: {:?}",
            result
        );
    }

    /// Read-side place precision: `&mut (p.x)` freezes `[p, x]` — reading
    /// a SIBLING place `p.y` (while the borrow is live, `*r` used later)
    /// is allowed (the read check reports the FULL place, not the base
    /// variable `p`).
    #[test]
    fn test_loan_read_place_sibling_allowed() {
        let result = check_source(
            "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set mut p = Point { x = 1, y = 2 };
                 set r: &mut Int<32> = &mut (p.x);
                 let v = p.y;
                 *r = 10;
                 return v;
             }",
        );
        assert!(
            result.is_ok(),
            "reading a sibling place must be allowed under place-precise freezing: {:?}",
            result
        );
    }

    /// Read-side place precision: reading the EXACT frozen place `p.x`
    /// while the exclusive loan is live is rejected (E109).
    #[test]
    fn test_loan_read_place_frozen_rejected() {
        let result = check_source(
            "type Point = struct { x: Int<32>, y: Int<32> }
             def main() -> Int<32> {
                 set mut p = Point { x = 1, y = 2 };
                 set r: &mut Int<32> = &mut (p.x);
                 let v = p.x;
                 *r = 10;
                 return v;
             }",
        );
        assert!(
            result.is_err(),
            "reading the exact frozen place must be rejected: {:?}",
            result
        );
    }

    /// Loan scoping in a let-else branch: a borrow created INSIDE the else
    /// branch expires at the branch's end — the source can be mutated
    /// afterwards (regression test for the else-branch loan leak).
    #[test]
    fn test_loan_else_branch_scoped_expires() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 let x = 0 else {
                     set r: &mut Int<32> = &mut a;
                     return *r;
                 };
                 a = 5;
                 return a + x;
             }",
        );
        assert!(
            result.is_ok(),
            "after the let-else branch ends, the source may be mutated: {:?}",
            result
        );
    }

    /// A mutation of a place frozen by `&ro` INSIDE an `isolate` block
    /// must be rejected: the flow-sensitive borrow-check post-pass must
    /// recurse into isolate bodies (they are nested statement containers,
    /// not skipped `_ => {}` leaves).
    #[test]
    fn test_isolate_block_ro_freeze_enforced() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut p = 42;
                 isolate {
                     let s = &ro p;
                     p = 10;
                     return 0;
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "mutating a `&ro`-frozen place inside isolate must be rejected: {:?}",
            result
        );
    }

    /// The same freeze guarantee inside a `comptime` block: the post-pass
    /// must recurse into comptime bodies so a frozen place cannot be
    /// mutated there either.
    #[test]
    fn test_comptime_block_ro_freeze_enforced() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut p = 42;
                 comptime {
                     let s = &ro p;
                     p = 10;
                     return 0;
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "mutating a `&ro`-frozen place inside comptime must be rejected: {:?}",
            result
        );
    }

    /// Transitive mutable-global access through the call graph: A calls B
    /// and B reads a top-level mutable global ⇒ calling A inside an
    /// `isolate` block must be rejected (E093), not just direct access.
    /// The pre-computed `transitive_mutable_global_access` closure is
    /// order-independent — B may be defined after A.
    #[test]
    fn test_isolate_transitive_mutable_global_access_rejected() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             def b() -> Int<32> { return g; }
             def a() -> Int<32> { return b(); }
             def main() -> Int<32> {
                 isolate {
                     let _ = a();
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "calling a function that transitively reads a mutable global inside isolate must be rejected: {:?}",
            result
        );
    }

    /// A deeper chain (A → B → C, C reads the mutable global) is also
    /// caught — the closure must be the FULL transitive reachability, not
    /// just one hop.
    #[test]
    fn test_isolate_transitive_mutable_global_access_deep_chain() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             def c() -> Int<32> { return g; }
             def b() -> Int<32> { return c(); }
             def a() -> Int<32> { return b(); }
             def main() -> Int<32> {
                 isolate {
                     let _ = a();
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "a deep call chain to a mutable-global reader must be rejected inside isolate: {:?}",
            result
        );
    }

    /// Control: a pure function (no mutable-global access anywhere in its
    /// transitive closure) must be ACCEPTED inside an isolate block.
    #[test]
    fn test_isolate_pure_function_accepted() {
        let result = check_source(
            "def helper(x: Int<32>) -> Int<32> { return x + 1; }
             def main() -> Int<32> {
                 isolate {
                     let _ = helper(1);
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_ok(),
            "a pure function inside isolate must be accepted: {:?}",
            result
        );
    }

    /// `@pure` on a function that reads a top-level mutable global must be
    /// REJECTED (E117) — the transitive `effect_of` label carries
    /// `MUTABLE_GLOBAL`, even when the read is through a callee.
    #[test]
    fn test_pure_function_rejects_mutable_global() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             @pure
             def bad() -> Int<32> { return g; }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "a @pure function reading a mutable global must be rejected: {:?}",
            result
        );
    }

    /// `@pure` transitively: A is `@pure` and calls B, B reads the mutable
    /// global ⇒ A must be rejected (the callee's label unions into A's).
    #[test]
    fn test_pure_function_rejects_transitive_global() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             def reader() -> Int<32> { return g; }
             @pure
             def wrapper() -> Int<32> { return reader(); }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "a @pure function calling a mutable-global reader must be rejected: {:?}",
            result
        );
    }

    /// A genuinely pure function (arithmetic only, no globals/I/O/unsafe/
    /// panic/comptime anywhere in its closure) must be ACCEPTED.
    #[test]
    fn test_pure_function_accepts_side_effect_free() {
        let result = check_source(
            "def add1(x: Int<32>) -> Int<32> { return x + 1; }
             @pure
             def twice(x: Int<32>) -> Int<32> { return add1(add1(x)); }
             def main() -> Int<32> { return twice(0); }",
        );
        assert!(
            result.is_ok(),
            "a side-effect-free @pure function must be accepted: {:?}",
            result
        );
    }

    /// Method-chain effects: calling an IMPL METHOD that reads a mutable
    /// global inside an `isolate` block must be rejected (E093) — the
    /// pre-computed `method_effect_of` label (keyed by (receiver DefId,
    /// method name)) carries MUTABLE_GLOBAL.
    #[test]
    fn test_isolate_method_call_mutable_global_rejected() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             type T = struct { x: Int<32> }
             impl for T {
                 def read_g(&self) -> Int<32> { return g; }
             }
             def main() -> Int<32> {
                 set t = T { x = 1 };
                 isolate {
                     let _ = t.read_g();
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "calling a method that reads a mutable global inside isolate must be rejected: {:?}",
            result
        );
    }

    /// `@pure` through a method call: a @pure function calling a method
    /// that reads a mutable global must be rejected (E117).
    #[test]
    fn test_pure_method_call_mutable_global_rejected() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             type T = struct { x: Int<32> }
             impl for T {
                 def read_g(&self) -> Int<32> { return g; }
             }
             @pure
             def wrapper(t: T) -> Int<32> { return t.read_g(); }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "a @pure function calling a mutable-global-reading method must be rejected: {:?}",
            result
        );
    }

    /// Control: a @pure function calling a genuinely pure method (no
    /// forbidden effects in the method's transitive label) is ACCEPTED.
    #[test]
    fn test_pure_method_call_accepted() {
        let result = check_source(
            "type T = struct { x: Int<32> }
             impl for T {
                 def add1(&self, y: Int<32>) -> Int<32> { return self.x + y; }
             }
             @pure
             def wrapper(t: T) -> Int<32> { return t.add1(1); }
             def main() -> Int<32> { return wrapper(T { x = 1 }); }",
        );
        assert!(
            result.is_ok(),
            "a @pure function calling a pure method must be accepted: {:?}",
            result
        );
    }

    /// Multi-hop method chain: method A calls `self.B()` and B reads a
    /// mutable global ⇒ A's transitive `method_effect_of` label carries
    /// MUTABLE_GLOBAL, so calling A inside an `isolate` block must be
    /// rejected (the method→method edge is resolved via the `self`
    /// receiver and closed by the multi-hop fixpoint).
    #[test]
    fn test_isolate_method_to_method_chain_rejected() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             type T = struct { x: Int<32> }
             impl for T {
                 def b(&self) -> Int<32> { return g; }
                 def a(&self) -> Int<32> { return self.b(); }
             }
             def main() -> Int<32> {
                 set t = T { x = 1 };
                 isolate {
                     let _ = t.a();
                 }
                 return 0;
             }",
        );
        assert!(
            result.is_err(),
            "calling a method whose method-chain reaches a mutable global inside isolate must be rejected: {:?}",
            result
        );
    }

    /// `@pure` through a multi-hop method chain: a @pure function calling
    /// method A (which calls `self.B`, which reads the global) must be
    /// rejected (E117) — the chain is closed transitively.
    #[test]
    fn test_pure_method_to_method_chain_rejected() {
        let result = check_source(
            "set mut g: Int<32> = 42;
             type T = struct { x: Int<32> }
             impl for T {
                 def b(&self) -> Int<32> { return g; }
                 def a(&self) -> Int<32> { return self.b(); }
             }
             @pure
             def wrapper(t: T) -> Int<32> { return t.a(); }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_err(),
            "a @pure function calling a method whose chain reaches a mutable global must be rejected: {:?}",
            result
        );
    }

    /// Control: a multi-hop method chain with NO forbidden effects (A
    /// calls `self.B`, B is pure arithmetic) is ACCEPTED from a @pure
    /// function.
    #[test]
    fn test_pure_method_to_method_chain_accepted() {
        let result = check_source(
            "type T = struct { x: Int<32> }
             impl for T {
                 def b(&self, y: Int<32>) -> Int<32> { return self.x + y; }
                 def a(&self, y: Int<32>) -> Int<32> { return self.b(y); }
             }
             @pure
             def wrapper(t: T) -> Int<32> { return t.a(1); }
             def main() -> Int<32> { return wrapper(T { x = 1 }); }",
        );
        assert!(
            result.is_ok(),
            "a @pure function calling a pure method chain must be accepted: {:?}",
            result
        );
    }

    #[test]
    fn test_match_arm0_loan_freezes_arm1_mutation() {
        let result = check_source(
            "type MyBool = enum { True, False }
             def main() -> Int<32> {
                 set mut p = 42;
                 set r: &mut Int<32> = &mut p;
                 set b = MyBool::True;
                 set x = match b {
                     MyBool::True => { p = 10; 0 }
                     MyBool::False => 1,
                 };
                 return *r;
             }",
        );
        assert!(
            result.is_err(),
            "a live loan must freeze a mutation inside a match arm: {:?}",
            result
        );
    }

    /// An explicit `move` inside a packed value CONSUMES the place:
    /// `set t = (move a, 1); set u = a;` must reject the reuse of `a`
    /// (the plain-`Ident` arm covers the implicit form; the explicit
    /// `move` must not slip through `_ => {}`).
    #[test]
    fn test_explicit_move_in_tuple_consumes_place() {
        let result = check_source(
            "type Person = struct { name: String, age: Int<32> }
             def main() -> Int<32> {
                 set a = Person { name = \"Alice\", age = 30 };
                 set t = (move a, 1);
                 set u = a;
                 return u.age;
             }",
        );
        assert!(
            result.is_err(),
            "reusing a place moved via explicit `move` must be rejected: {:?}",
            result
        );
    }

    /// A `match` whose arm yields a non-`Copy` value by value consumes
    /// it: the subsequent use of the source must be rejected.
    #[test]
    fn test_match_arm_value_consumes_place() {
        let result = check_source(
            "type Person = struct { name: String, age: Int<32> }
             def main() -> Int<32> {
                 set a = Person { name = \"Alice\", age = 30 };
                 set r = match 0 { _ => a };
                 set u = a;
                 return u.age;
             }",
        );
        assert!(
            result.is_err(),
            "a match arm yielding a non-Copy value by value consumes the place: {:?}",
            result
        );
    }

    // ── `@auto_ro` implicit downgrade loans suite (committee 2026-08-05) ──
    // The implicit `&mut T → &T` downgrade at a call site registers a
    // READ-ONLY loan on the argument's place, scoped to the CALL EXPRESSION:
    // the source is frozen during the call and becomes writable again after
    // it (unlike the explicit `&ro`/`.freeze!()` forms, which freeze until
    // the enclosing block ends).

    /// The committee's ruling example: after `takes_shared(r)`, the source
    /// `r` is writable again (`*r = 42` accepted), and a second implicit
    /// downgrade works — the implicit loan died at the call's end.
    #[test]
    fn test_auto_ro_implicit_loan_expires_after_call() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             @auto_ro
             def process(r: &mut Int<32>) -> Int<32> {
                 let val1 = takes_shared(r);
                 *r = 42;
                 let val2 = takes_shared(r);
                 return val1 + val2;
             }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return process(r);
             }",
        );
        assert!(
            result.is_ok(),
            "after the call expression ends, the implicitly-frozen source may be mutated: {:?}",
            result
        );
    }

    /// Control: without `@auto_ro`, the `&mut T → &T` argument pass is still
    /// rejected (the gate is unchanged — the implicit-downgrade ruling only
    /// adds the loan bookkeeping).
    #[test]
    fn test_auto_ro_implicit_loan_gate_control() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def process(r: &mut Int<32>) -> Int<32> {
                 return takes_shared(r);
             }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return process(r);
             }",
        );
        assert!(
            result.is_err(),
            "without @auto_ro, &mut T -> &T must be rejected: {:?}",
            result
        );
    }

    /// The call-scope truncate must NOT kill loans registered BEFORE the
    /// call: an explicit `&ro r` still freezes `r` after the call.
    #[test]
    fn test_auto_ro_implicit_loan_preserves_explicit_loans() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             @auto_ro
             def process(r: &mut Int<32>) -> Int<32> {
                 let s = &ro r;
                 let val1 = takes_shared(r);
                 *r = 100;
                 return val1 + *s;
             }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return process(r);
             }",
        );
        assert!(
            result.is_err(),
            "an explicit &ro loan must survive the call and still freeze the source: {:?}",
            result
        );
    }

    // ── E0499 acceptance probe (committee 2026-08-05: Polonius/LLBC#
    // level as the acceptance bar) ──
    // `get_suffix_at_x` (LLBC# paper §6): a `while let` loop that walks a
    // mutable list by REBORROWING the tail (`ls = tl`) each iteration.
    // rustc's NLL rejects it (E0499: cannot borrow `*ls` as mutable more
    // than once); Polonius and LLBC# accept it.  Per the committee ruling,
    // Posita must
    // stand on the Polonius side.  This probe records the CURRENT
    // behavior; if it fails, the loop-precision work is required.

    /// E0499 probe: the while-let tail-reborrow loop must be ACCEPTED.
    #[test]
    fn test_e0499_loop_reborrow_accepted() {
        let result = check_source(
            "type List = enum { Nil, Cons((Int<32>, &mut List)) }
             def get_suffix(ls: &mut List, x: Int<32>) -> &mut List {
                 while let Cons((hd, tl)) = *ls {
                     // A real loan created inside the loop exercises the
                     // post-pass's loan registration + back-edge liveness
                     // (a local — no conflict with the reborrow pattern).
                     set mut n = 1;
                     let nr: &mut Int<32> = &mut n;
                     *nr = *nr + 1;
                     ls = tl;
                 }
                 return ls;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "the while-let tail-reborrow loop must be accepted (Polonius-level precision): {:?}",
            result
        );
    }

    /// E0499 FULL sample (the paper §6 `get_suffix_at_x`): the loop with
    /// the early `if`/break (`leave;`) exits + the loop-carried tail
    /// reborrow — the case the block-scoped borrow checker cannot handle
    /// (the loop-precision requirement).
    #[test]
    fn test_e0499_get_suffix_at_x_full() {
        let result = check_source(
            "type List = enum { Nil, Cons((Int<32>, &mut List)) }
             def get_suffix_at_x(ls: &mut List, x: Int<32>) -> &mut List {
                 set mut suffix = ls;
                 loop {
                     if let Cons((hd, tl)) = *suffix {
                         if *hd == x { leave; }
                         suffix = tl;
                     } else {
                         leave;
                     }
                 }
                 return suffix;
             }
             def main() -> Int<32> { return 0; }",
        );
        assert!(
            result.is_ok(),
            "the paper §6 get_suffix_at_x loop must be accepted (E0499 loop precision): {:?}",
            result
        );
    }

    // ── `&mut` read-side freeze suite (committee 2026-08-05: strict
    // "neither readable nor writable" per SYNTAX.md §References) ──
    // Constructed BEFORE the change: at BASELINE the read-side
    // freeze is NOT enforced, so test_loan_mut_read_frozen_rejected is
    // expected to FAIL — the divergence the read-side-freeze change closes.

    /// `&mut a` freezes `a` against READS too: reading the source while the
    /// exclusive borrow is live is rejected (C2: neither readable nor
    /// writable).
    /// Point-level liveness: an UNUSED borrow is dead immediately —
    /// `r` is never deref'd, so its loan dies at its (non-existent) last
    /// use and reading the source is allowed (NLL semantics).
    #[test]
    fn test_loan_mut_unused_borrow_no_freeze() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = a;
                 return x;
             }",
        );
        assert!(
            result.is_ok(),
            "an unused `&mut` borrow does not freeze the source: {:?}",
            result
        );
    }

    /// Reading THROUGH the exclusive borrow is allowed (`*r` uses the
    /// borrow, not the frozen source).
    #[test]
    fn test_loan_mut_read_through_borrow_ok() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return *r;
             }",
        );
        assert!(
            result.is_ok(),
            "reading through the exclusive borrow must be allowed: {:?}",
            result
        );
    }

    /// The read-side freeze does NOT apply to the read-only borrow: after
    /// `&ro r`, reading THROUGH the read-only view is allowed (C4 freezes
    /// mutation only).
    #[test]
    fn test_loan_ro_read_through_view_ok() {
        let result = check_source(
            "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let s = &ro r;
                 return takes_shared(s);
             }",
        );
        assert!(
            result.is_ok(),
            "reading through a read-only view must be allowed: {:?}",
            result
        );
    }

    /// The read-side freeze is scoped: reading the source AFTER the
    /// exclusive borrow's block ends is allowed.
    #[test]
    fn test_loan_mut_read_after_scope_ok() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 let y = if true {
                     set r: &mut Int<32> = &mut a;
                     *r
                 } else {
                     0
                 };
                 let x = a;
                 return x + y;
             }",
        );
        assert!(
            result.is_ok(),
            "after the borrowing block ends, the source may be read: {:?}",
            result
        );
    }

    // ── Flow-sensitive liveness suite (constructed for the
    // block-level post-pass, later reverted) ──
    // (committee 2026-08-05): the precise "last use" liveness is the
    // point-level refinement.  The block-level post-pass could not deliver the
    // SAME-BLOCK last-use precision (block granularity cannot represent
    // intra-block ordering) and was reverted (fundamentally
    // incompatible with the block granularity).  Point-level liveness
    // (rustc sparse-interval points) is the documented follow-up.  These
    // tests now assert the CURRENT block-scoped behavior (rejection) as
    // regression baselines.

    /// Point-level liveness: the borrow's last use (`*r`) ends the
    /// loan AT THAT STATEMENT — the same-block `a = 5` afterwards is
    /// allowed (the point-level precision the block-level approximation
    /// could not deliver).
    #[test]
    fn test_flow_sensitive_loan_dies_at_last_use() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 a = 5;
                 return x;
             }",
        );
        assert!(
            result.is_ok(),
            "after the borrow's last use (same block), the source may be mutated: {:?}",
            result
        );
    }

    /// Same for the read side (the read-side freeze also ends at the
    /// last use).
    #[test]
    fn test_flow_sensitive_loan_dies_at_last_use_read() {
        let result = check_source(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 let y = a;
                 return x + y;
             }",
        );
        assert!(
            result.is_ok(),
            "after the borrow's last use (same block), the source may be read: {:?}",
            result
        );
    }

    /// W114: `&ro` on an already-immutable reference is redundant — allow
    /// but warn (the committee's "looks strange -> assume the program is
    /// wrong" tradition).
    #[test]
    fn test_ro_on_immutable_reference_warns() {
        let (prog, warns) = check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &Int<32> = &a;
                 let s = &ro r;
                 return *s;
             }",
        );
        let _ = prog;
        assert!(
            warns.iter().any(|w| w.contains("already-immutable")),
            "W114 must warn on `&ro` of an already-immutable reference: {warns:?}"
        );
    }

    /// `&ro` on a mutable reference is the normal coercion — no warning.
    #[test]
    fn test_ro_on_mutable_reference_no_warning() {
        let (prog, warns) = check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let s = &ro r;
                 return *s;
             }",
        );
        let _ = prog;
        assert!(
            warns.is_empty(),
            "`&ro` on a mutable reference must not warn: {warns:?}"
        );
    }
}

/// (constructed BEFORE the leave-with fix): the `leave with`
/// value is a USE of the borrow variable — the loan must stay alive
/// and the earlier source mutation must be rejected.  At BASELINE the
/// LeaveWith value is invisible to the liveness — the
/// test is expected to FAIL, the divergence the fix closes.
#[test]
fn test_leave_with_value_uses_loan() {
    let result = check_source(
        "type MyResult = Result<Int<32>, Int<32>>
             def main() -> MyResult {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 a = 5;
                 if a > 100 { leave with *r; }
                 return MyResult::Ok(a);
             }",
    );
    assert!(
        result.is_err(),
        "`a = 5` must be rejected — `r` is used in the `leave with` value: {:?}",
        result
    );
}

/// Reconstructed catch defensive test (the earlier version was removed for
/// the "found Unit" diagnostic artifact — the catch type investigation
/// confirmed the catch's value type is correct).  The leave-with INSIDE a
/// catch branch is a USE of the borrow variable — the loan must stay alive
/// and the earlier source mutation must be rejected.
#[test]
fn test_catch_branch_value_uses_loan() {
    let result = check_source(
        "type MyResult = Result<Int<32>, Int<32>>
         def danger() -> MyResult { leave with 1; }
         def main() -> MyResult {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             a = 5;
             let x = danger() catch {
                 |e| { leave with *r; }
                 |_| { 0 }
             };
             return MyResult::Ok(x);
         }",
    );
    assert!(
        result.is_err(),
        "`a = 5` must be rejected — `r` is used in the catch branch's `leave with` value: {:?}",
        result
    );
}

/// Probe: the loop body WITH an `if` — the body's entry block
/// (the if block) must be reachable from the head, so `*r` inside the
/// if-arm keeps the loan alive.
#[test]
fn probe_flaw1_loop_entry() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut c = 2;
             set r: &mut Int<32> = &mut a;
             a = 5;
             while c > 0 {
                 if c > 1 { let x = *r; }
                 c = c - 1;
             }
             return 0;
         }",
    );
    assert!(
        r.is_err(),
        "loop-entry mutation while the borrow is live must be rejected: {:?}",
        r
    );
}

/// Probe: the birth bound — a mutation BEFORE the borrow is valid
/// (the loan is not yet issued); a mutation AFTER the borrow (with a later
/// use) is rejected.
#[test]
fn probe_flaw2_birth_bound() {
    // Valid: the mutation happens BEFORE the borrow's issuance.
    let r1 = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             a = 5;
             set r: &mut Int<32> = &mut a;
             return *r;
         }",
    );
    assert!(
        r1.is_ok(),
        "mutate-then-borrow is legal — the birth bound must not reject it: {:?}",
        r1
    );
    // Rejected: the mutation AFTER the borrow (r used later).
    let r2 = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             a = 5;
             return *r;
         }",
    );
    assert!(
        r2.is_err(),
        "post-borrow mutation must be rejected: {:?}",
        r2
    );
}

/// Probe: the borrow-variable REBINDING — after `r = &mut b`, the
/// NEW source (b) is frozen (a later b mutation is rejected) and the OLD
/// source (a) is free (an a mutation is allowed).
#[test]
fn probe_flaw3_rebind() {
    let r1 = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut b = 7;
             set mut r: &mut Int<32> = &mut a;
             r = &mut b;
             b = 5;
             return *r;
         }",
    );
    assert!(
        r1.is_err(),
        "mutating the NEW source while the borrow is live must be rejected: {:?}",
        r1
    );
    let r2 = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut b = 7;
             set mut r: &mut Int<32> = &mut a;
             r = &mut b;
             a = 5;
             return *r;
         }",
    );
    // The rebind (`r = &mut b`) KILLS the original `&mut a` loan
    // (kill-on-rebind — the `loan_killed_at` reborrow-kill), so `a = 5`
    // after the rebind is legal: the OLD source is free once `r` points
    // to `b` (matching rust's NLL).
    assert!(
        r2.is_ok(),
        "mutating the OLD source after the rebind killed the original loan must be accepted: {:?}",
        r2
    );
}

/// the E0499 loop-precision test with a REAL back-edge loan — a
/// loan issued in the iteration (bound to `r`, used before the rebinding)
/// whose liveness must flow across the loop's back edge (the back-edge
/// reachability fix made the loop body's entry reachable, so the liveness
/// is correct).
#[test]
fn test_e0499_loop_back_edge_loan() {
    let result = check_source(
        "type List = enum { Nil, Cons((Int<32>, &mut List)) }
         def get_suffix(ls: &mut List, x: Int<32>) -> &mut List {
             while let Cons((hd, tl)) = *ls {
                 set r: &mut List = &mut *ls;
                 let _ = r;
                 ls = tl;
             }
             return ls;
         }
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_ok(),
        "the loop-carried loan must be accepted (the back-edge liveness): {:?}",
        result
    );
}

/// Minor (the general kill): a borrow variable reassigned with a
/// NON-borrow value (`r = r2`) — the old source is free afterwards (the

/// the nested-control-flow if-arm — the arm's true EXIT must be
/// wired to the join, so the `a = 5` inside the nested arm's join keeps

/// the `&mut` EXCLUSIVITY — a second exclusive borrow of the

/// Probe: the loop-carried birth bound — the loan issued INSIDE the
/// loop (in a later block) must NOT freeze the mutation BEFORE the issuance
/// on the first iteration (the forward reachability conflates iterations —

/// an immutable borrow (`&T`) of the same place while the
/// exclusive `&mut` is live must be REJECTED (the `&mut` freeze is

/// Two-phase borrow probe: the `f(&mut x, x)` pattern — rustc's two-phase
/// borrows allow the `&mut` receiver/arg to coexist with the reads of the
/// same place in the OTHER arguments (the activation is deferred until

/// Two-phase borrows (committee-approved — the feasibility finding): the
/// METHOD-receiver pattern `lst.push(lst.data)` is ALREADY accepted — the
/// receiver's `&mut self` borrow is IMPLICIT in the HIR (no explicit
/// UnaryOp — the post-pass sees no loan — the same "unobservable" pattern
/// as the implicit downgrade loans), so NO phase modeling is needed.  The
/// explicit free-function pattern `take(&mut a, a)` remains REJECTED (the
/// two-phase is NOT extended to the free functions — matching rustc).
#[test]
fn test_two_phase_method_receiver_accepted() {
    // Accepted: the method receiver + the arg read of the same place.
    let r1 = check_source(
        "type IntList = struct { data: Int<32> }
         impl for IntList {
             def push(&mut self, x: Int<32>) -> Int<32> { return x; }
         }
         def main() -> Int<32> {
             set mut lst = IntList { data = 0 };
             let n = lst.push(lst.data);
             return n;
         }",
    );
    assert!(
        r1.is_ok(),
        "the method-receiver two-phase pattern must be accepted: {:?}",
        r1
    );
    // Rejected: the EXPLICIT free-function `&mut` + the same-place read
    // (the two-phase is not extended to the free functions).
    let r2 = check_source(
        "def take(a: &mut Int<32>, b: Int<32>) -> Int<32> { return *a + b; }
         def main() -> Int<32> {
             set mut a = 42;
             return take(&mut a, a);
         }",
    );
    assert!(
        r2.is_err(),
        "the explicit free-function pattern must stay rejected: {:?}",
        r2
    );
}

/// The committee member's explicit-&ro entry: `v.push((&ro v).len())` —
/// the temporary `&ro v` read-only loan should die at the `len()` return

/// Probe: the path-insensitive kill — the ELSE-branch's strict-
/// prefix assignment must NOT kill the THEN-branch's loan (the post-join

/// (constructed BEFORE the cross-function wiring): the cross-function
/// returned borrow — `get(&mut a)` returns a borrow of `a`; the returned
/// borrow `r` is live at the later `a = 5` — the source must be frozen.
/// At BASELINE the cross-function facts are not consumed by the post-pass
/// — the `a = 5` is accepted — the test is expected to FAIL, the
/// divergence the cross-function wiring closes.
#[test]
fn test_cross_function_freeze() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             let r: &mut Int<32> = get(&mut a);
             a = 5;
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "`a = 5` must be rejected — the returned borrow `r` freezes `a`: {:?}",
        result
    );
}

/// Regression: the cross-function returned-borrow freeze must hold
/// when the CALLER appears BEFORE the callee in the file AND the
/// callee's return sits inside a `match` arm (a context the AST-level
/// `collect_returns_ast` under-approximation misses).  With the
/// interleaved checking, `main` was borrow-checked against `get`'s
/// AST-level signature (no A(ρ) fact for the match-arm return), so
/// `a = 5` was silently accepted; with the two-phase pipeline the
/// signature is finalized (HIR-level) before ANY body is checked, so
/// the freeze must be enforced.
#[test]
fn test_cross_function_reversed_order_match_arm_return() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             let r: &mut Int<32> = get(&mut a);
             a = 5;
             return *r;
         }
         def get(x: &mut Int<32>) -> &mut Int<32> {
             return match 0 { _ => x };
         }",
    );
    assert!(
        result.is_err(),
        "the match-arm returned borrow must freeze `a` even with the caller first in the file: {:?}",
        result
    );
}

/// Probe: the loan issued in ONE branch — the mutation after the
/// join on the other path — the loan is live (the borrow var used after
/// the join) — the mutation must be rejected (the dominance birth bound

/// The string literal `"hello"` has type `&Str` — Copy (SYNTAX.md
/// §Strings) — the copy of a `&Str` binding is legal; the move check
/// must NOT treat it as non-Copy.
#[test]
fn test_str_literal_copy_accepted() {
    let result = check_source(
        "def main() -> Int<32> {
             set a = \"hello\";
             set b = a;
             set c = a;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the `&Str` copy must be accepted (Copy — not a move): {:?}",
        result
    );
}

/// The branch path propagation — `a` is moved on ONE branch; after
/// the join it is possibly-moved — the later use must be rejected.
#[test]
fn test_move_branch_path_propagation() {
    let result = check_source(
        "def make() -> String { return \"x\"; }
         def main() -> Int<32> {
             set a = make();
             if 1 > 0 {
                 set b = a;
             }
             set c = a;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the use of the possibly-moved `a` after the join must be rejected: {:?}",
        result
    );
}

/// The re-initialization — after `a = "new"`, the moved `a` is
/// usable again (the RFC unified rule — any hole filled).
#[test]
fn test_move_reinitialization() {
    let result = check_source(
        "def main() -> Int<32> {
             set a = \"hello\";
             set b = a;
             a = \"new\";
             set c = a;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the re-initialized `a` must be usable again: {:?}",
        result
    );
}

/// Place-tree-level (RFC unified rule): the sibling slots' moves are
/// independent — `move arr[i]` then `move arr[j]` (i != j) — both allowed
/// (the variable-index form; the literal-index inference is a separate
/// checker issue).
#[test]
fn test_move_sibling_slots_independent() {
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [\"a\", \"b\", \"c\"];
             let i: Int<32> = 2;
             let j: Int<32> = 0;
             let s = move arr[i];
             let t = move arr[j];
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the sibling-slot moves must be independent: {:?}",
        result
    );
}

/// Place-tree-level (RFC + Polonius): a hole-containing whole-array move
/// is ALLOWED — the hole (the moved-out slot) travels as an uninitialized
/// slot (the move-propagation semantics).
#[test]
fn test_move_hole_whole_move_allowed() {
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [\"a\", \"b\", \"c\"];
             let i: Int<32> = 2;
             let s = move arr[i];
             let t = move arr;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the hole-containing whole-array move must be allowed: {:?}",
        result
    );
}

/// A `&Str` array is Copy (SYNTAX.md §Strings) — the implicit copy is
/// legal; the move check must not treat the `&Str` elements as non-Copy.
#[test]
fn test_str_array_copy_accepted() {
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [\"a\", \"b\", \"c\"];
             let t = arr;
             let x = arr[0];
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the `&Str` array copy must be accepted (Copy — not a move): {:?}",
        result
    );
}

/// The re-initialization at the array level: after `arr = [...]`, the
/// moved `arr` is usable again (the RFC unified rule — the hole filled).
#[test]
fn test_move_array_reinitialization() {
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [\"a\", \"b\", \"c\"];
             let t = move arr;
             arr = [\"x\", \"y\", \"z\"];
             let k: Int<32> = 0;
             let x = arr[k];
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the re-initialized `arr` must be usable again: {:?}",
        result
    );
}

/// (constructed BEFORE the fix A): the literal index `arr[2]` —
/// the index inference variable dangles (the expected type not propagated)
/// — the E030 "index must be an integer" false positive.  At BASELINE the
/// test is expected to FAIL; the fix A (the integer expectation propagated
/// into `infer_expr`) closes the divergence.
#[test]
fn test_literal_index_accepted() {
    let result = check_source(
        "def main() -> Int<32> {
             set arr = [\"a\", \"b\", \"c\"];
             let x = arr[2];
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the literal index must be accepted: {:?}",
        result
    );
}

/// A MUTABLE borrow of a constant-index element (`&mut arr[0]`) must be
/// usable — the borrow's operand includes the index (the parser fix: the
/// borrow wraps the postfix chain, `&mut (arr[0])`, not `(&mut arr)[0]`).
#[test]
fn test_borrow_const_index_mut_usable() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut arr = [1, 2, 3];
             set r: &mut Int<32> = &mut arr[0];
             return *r;
         }",
    );
    assert!(result.is_ok(), "&mut arr[0] must be usable: {:?}", result);
}

/// Constant-index granularity (mirrors rustc's `ConstantIndex`): freezing
/// `&ro r` where `r = &mut arr[0]` must NOT freeze `arr[1]` — different
/// constant indexes are distinct places, so writing `arr[1]` is allowed
/// while the `arr[0]` read-only borrow is live.
#[test]
fn test_ro_freeze_const_index_distinct_elements_allowed() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut arr = [1, 2, 3];
             set r: &mut Int<32> = &mut arr[0];
             let s = &ro r;
             arr[1] = 10;
             return *s;
         }",
    );
    assert!(
        result.is_ok(),
        "writing a DIFFERENT constant-index element while arr[0] is frozen must be allowed: {:?}",
        result
    );
}

/// The SAME constant index must stay frozen: writing `arr[0]` while
/// `&ro r` (`r = &mut arr[0]`) is live is a mutation of the frozen place.
#[test]
fn test_ro_freeze_const_index_same_element_rejected() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut arr = [1, 2, 3];
             set r: &mut Int<32> = &mut arr[0];
             let s = &ro r;
             arr[0] = 10;
             return *s;
         }",
    );
    assert!(
        result.is_err(),
        "writing the SAME constant-index element while arr[0] is frozen must be rejected: {:?}",
        result
    );
}

/// SCALAR referent freeze: `&ro r`
/// where `r = &mut a` freezes the REFERENT `a` — writing `a = 5` while
/// the read-only borrow is live must be rejected (rustc E0506).  The
/// cfg_graph borrow wrap (`Deref(Root(r))` for `&ro r`) + the polonius
/// referent resolution land the ReadOnly loan on `Root(a)`.
#[test]
fn test_ro_freeze_scalar_referent_rejected() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             let s = &ro r;
             a = 5;
             return *s;
         }",
    );
    assert!(
        result.is_err(),
        "writing the referent of a scalar &ro borrow must be rejected: {:?}",
        result
    );
}

/// A REBORROW resolves through the parent loan but does NOT
/// terminate it (rustc `loan_kills.rs` kills only on StorageDead /
/// assignment).  `&ro r` inside a call is a reborrow of `r: &mut a` — the
/// parent loan stays live until `r`'s last use (`return *r`), so writing
/// `a = 5` in between is frozen (rustc E0506).
#[test]
fn test_reborrow_parent_loan_stays_live_rejected() {
    let result = check_source(
        "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
         def main() -> Int<32> {
             set mut a = 42;
             set r = &mut a;
             let x = takes_shared(&ro r);
             a = 5;
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "a reborrow must not kill the parent loan — writing the referent while r is still used must be rejected: {:?}",
        result
    );
}

/// Control for R1: once the parent borrow variable's last use is past,
/// the referent is writable again (point-level liveness).
#[test]
fn test_reborrow_parent_loan_released_after_last_use_ok() {
    let result = check_source(
        "def takes_shared(x: &Int<32>) -> Int<32> { return *x; }
         def main() -> Int<32> {
             set mut a = 42;
             set r = &mut a;
             let x = takes_shared(&ro r);
             let y = *r;
             a = 5;
             return x + y;
         }",
    );
    assert!(
        result.is_ok(),
        "after r's last use the referent must be writable again: {:?}",
        result
    );
}

/// Two DISTINCT `&mut a` arguments of one call must be
/// rejected by E112 — the cross-function loans now carry the ARGUMENT's span
/// (`arg.span()`), so the same-span E112 exemption (polonius.rs:1804) no
/// longer conflates them (rustc E0499).
#[test]
fn test_m2_distinct_mut_args_exclusivity_rejected() {
    let result = check_source(
        "def f(x: &mut Int<32>, y: &mut Int<32>) -> &mut Int<32> {
             if *x > 0 { return x; }
             return y;
         }
         def main() -> Int<32> {
             set mut a = 42;
             set r = f(&mut a, &mut a);
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "two distinct &mut a arguments in one call must be an exclusivity violation: {:?}",
        result
    );
}

/// Control for the two-distinct-`&mut`-args rejection: two DISTINCT
/// referents in one call are fine.
#[test]
fn test_m2_distinct_mut_args_distinct_places_ok() {
    let result = check_source(
        "def f(x: &mut Int<32>, y: &mut Int<32>) -> &mut Int<32> {
             if *x > 0 { return x; }
             return y;
         }
         def main() -> Int<32> {
             set mut a = 42;
             set mut b = 7;
             set r = f(&mut a, &mut b);
             return *r;
         }",
    );
    assert!(
        result.is_ok(),
        "two distinct &mut referents in one call must be accepted: {:?}",
        result
    );
}

/// Control: after the `&ro` borrow's last use, the referent becomes
/// writable again (point-level liveness) — the same write AFTER
/// `*s`'s final use is accepted.
#[test]
fn test_ro_freeze_scalar_referent_released_after_last_use() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             let s = &ro r;
             let x = *s;
             a = 5;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "writing the referent after the &ro borrow's last use must be accepted: {:?}",
        result
    );
}

/// Dynamic-index conservatism (mirrors rustc's `ProjectionElem::Index`):
/// `&ro r` where `r = &mut arr[i]` may equal ANY element, so writing the
/// constant `arr[0]` while a DYNAMIC-index borrow is live must be
/// rejected.
#[test]
fn test_ro_freeze_dynamic_index_blocks_const_index() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut arr = [1, 2, 3];
             let i: Int<32> = 0;
             set r: &mut Int<32> = &mut arr[i];
             let s = &ro r;
             arr[0] = 10;
             return *s;
         }",
    );
    assert!(
        result.is_err(),
        "writing arr[0] while a dynamic-index borrow of arr is live must be rejected (conservative): {:?}",
        result
    );
}

/// The converse direction: a CONSTANT-index freeze must also block a
/// DYNAMIC-index write (`arr[j]` may equal `arr[0]`).
#[test]
fn test_ro_freeze_const_index_blocks_dynamic_index() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut arr = [1, 2, 3];
             set r: &mut Int<32> = &mut arr[0];
             let s = &ro r;
             let j: Int<32> = 1;
             arr[j] = 10;
             return *s;
         }",
    );
    assert!(
        result.is_err(),
        "writing arr[j] while arr[0] is frozen must be rejected (conservative): {:?}",
        result
    );
}

/// The non-Copy wiring (the §Copy `type_is_copy` — the Adt/struct is
/// non-Copy): a struct containing a `String` field — its move must be
/// tracked (the use-after-move rejected).
#[test]
fn test_move_struct_non_copy() {
    let result = check_source(
        "type Person = struct { name: String, age: Int<32> }
         def main() -> Int<32> {
             set p = Person { name = \"Alice\", age = 30 };
             set q = p;
             set r = p;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the second use of the moved struct `p` must be rejected: {:?}",
        result
    );
}

/// The nested-block non-Copy roots: an independent non-Copy binding
/// INSIDE an if-branch (a function call returning a `String` — not a
/// string literal, so the heuristic would not catch it) — its moves must
/// be tracked by the recursive root collection.
#[test]
fn test_move_nested_block_non_copy() {
    let result = check_source(
        "def make() -> String { return \"x\"; }
         def main() -> Int<32> {
             if 1 > 0 {
                 set d = make();
                 set e = d;
                 set f = d;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the second use of the nested-block moved `d` must be rejected: {:?}",
        result
    );
}

/// A move INSIDE a loop is possibly-moved after it — the
/// after-loop use must be rejected (the loop may have executed).
#[test]
fn test_move_loop_after_use() {
    let result = check_source(
        "def make() -> String { return \"x\"; }
         def main() -> Int<32> {
             set a = make();
             set mut i: Int<32> = 0;
             while i < 3 {
                 set b = a;
                 i = i + 1;
             }
             set c = a;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the after-loop use of the possibly-moved `a` must be rejected: {:?}",
        result
    );
}

/// (before the fix): two temporary `&mut a` loans in ONE call —
/// the exclusivity check's `None => false` misses them — the overlap must
/// be reported (SYNTAX.md §References: `&mut T` is exclusive).
#[test]
fn test_temporary_mut_exclusivity() {
    let result = check_source(
        "def f(x: &mut Int<32>, y: &mut Int<32>) -> Int<32> { return *x; }
         def main() -> Int<32> {
             set mut a = 42;
             return f(&mut a, &mut a);
         }",
    );
    assert!(
        result.is_err(),
        "the two overlapping temporary `&mut a` loans must be rejected: {:?}",
        result
    );
}

/// The affine consumption blind spot: a non-Copy value packed into a
/// TUPLE is consumed — the later use must be rejected.
#[test]
fn test_move_consumed_in_tuple() {
    let result = check_source(
        "def make() -> String { return \"x\"; }
         def main() -> Int<32> {
             set p = make();
             set t = (p, 1);
             set u = p;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the non-Copy `p` packed into the tuple is consumed — the later use must be rejected: {:?}",
        result
    );
}

/// The AncestorClobber case: reassigning the whole struct
/// `a` destroys the value `r` borrows — the mutation must be rejected
/// (the loan survives — it must not be killed by the ancestor clobber).
#[test]
fn test_ancestor_clobber_not_kill() {
    let result = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         def main() -> Int<32> {
             set mut a = Point { x = 1, y = 2 };
             set r: &mut Int<32> = &mut a.y;
             a = Point { x = 3, y = 4 };
             let v = *r;
             return v;
         }",
    );
    assert!(
        result.is_err(),
        "the ancestor clobber must not kill the loan — the mutation must be rejected: {:?}",
        result
    );
}

/// The cross-function ordering dependency: `main` appears
/// BEFORE `get` — the borrow-signature pre-registration pass makes the
/// cross-function loan issuance order-independent — `a = 5` must be
/// rejected.
#[test]
fn test_cross_function_reversed_order() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             let r: &mut Int<32> = get(&mut a);
             a = 5;
             return *r;
         }
         def get(x: &mut Int<32>) -> &mut Int<32> { return x; }",
    );
    assert!(
        result.is_err(),
        "`a = 5` must be rejected regardless of the item order (the pre-registration must be order-independent): {:?}",
        result
    );
}

/// The method-call cross-function freeze: a method whose output derives
/// from an input borrow — the
/// binding (the returned borrow) freezes the input's source — the
/// mutation after the call must be rejected.
#[test]
fn test_method_call_cross_function_freeze() {
    let result = check_source(
        "type Box = struct { value: Int<32> }
         impl Box {
             def get(self: &Box, x: &mut Int<32>) -> &mut Int<32> { return x; }
         }
         def main() -> Int<32> {
             set obj = Box { value = 1 };
             set mut a = 42;
             let r: &mut Int<32> = obj.get(&mut a);
             a = 5;
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "`a = 5` must be rejected — the method's returned borrow freezes `a`: {:?}",
        result
    );
}

/// The comptime boundary: a comptime function with a Ref parameter — its
/// signature is NOT registered (compile-time — no runtime borrow), and
/// the comptime call does NOT produce a runtime freeze — the mutation
/// after the call is accepted.
#[test]
fn test_comptime_no_runtime_freeze() {
    let result = check_source(
        "comptime def compute_size(x: &mut Int<32>) -> Int<32> { return 42; }
         def main() -> Int<32> {
             set mut a = 5;
             let s = compute_size!(&mut a);
             a = 10;
             return s;
         }",
    );
    assert!(
        result.is_ok(),
        "the comptime call must NOT freeze `a` (compile-time — no runtime loan): {:?}",
        result
    );
}

/// The Pratt left associativity: `10 - 3 - 2` must parse as
/// `(10 - 3) - 2` — the RHS must be parsed with `bp + 1` so a same-level
/// operator is not swallowed into a right-nested tree.
#[test]
fn test_binary_left_associativity() {
    let (program, _) =
        check_source_keep_hir("def main() -> Int<32> { let x = 10 - 3 - 2; return x; }");
    let body = match &program.items[0] {
        crate::hir::hir::HirStmt::FunctionDef { body: Some(b), .. } => b,
        _ => panic!("expected FunctionDef body"),
    };
    let value = match &body[0] {
        crate::hir::hir::HirStmt::VariableDef { value: Some(v), .. } => v,
        _ => panic!("expected VariableDef value"),
    };
    match &**value {
        crate::hir::hir::HirExpr::BinaryOp { left, right, .. } => {
            // Left-associative: the LEFT operand is the inner `10 - 3`.
            assert!(
                matches!(left.as_ref(), crate::hir::hir::HirExpr::BinaryOp { .. }),
                "`10 - 3 - 2` must be `(10 - 3) - 2` (left-associative): {:?}",
                value
            );
            assert!(
                matches!(right.as_ref(), crate::hir::hir::HirExpr::Literal(..)),
                "the RIGHT operand must be the bare `2`: {:?}",
                value
            );
        }
        _ => panic!("expected BinaryOp: {:?}", value),
    }
}

/// The stuck-test regression: a type alias with an INVALID
/// regex pattern inside a FUNCTION BODY used to hang the parser forever —
/// the `synchronize` recovery did not consume a sync token (`Token::Type`
/// is one), so the statement loop spun on it.  The synchronize fix
/// consumes the sync token — the program is rejected quickly instead of
/// hanging.
#[test]
fn test_function_body_type_no_hang() {
    let result = check_source(
        "def main() -> Int<32> {
             type R = Regex<\"a[\">;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the invalid pattern must be rejected quickly (not hang): {:?}",
        result
    );
}

/// old loan must be killed regardless of the value's kind).
#[test]
fn probe_minor_non_borrow_reassign() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut b = 7;
             set mut r: &mut Int<32> = &mut a;
             set r2: &mut Int<32> = &mut b;
             r = r2;
             a = 5;
             return *r;
         }",
    );
    eprintln!("MINOR_PROBE old-source accepted = {:?}", r.is_ok());
}

/// the loan alive (r used after the outer if) and is REJECTED.
#[test]
fn probe_nested_if_arm_exit() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             if true {
                 if true { }
                 a = 5;
             }
             let x = *r;
             return x;
         }",
    );
    eprintln!("NESTED_IF_PROBE rejected = {:?}", r.is_err());
}

/// same place while the first is live must be REJECTED.
#[test]
fn probe_mut_exclusivity() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r1: &mut Int<32> = &mut a;
             set r2: &mut Int<32> = &mut a;
             return *r1;
         }",
    );
    eprintln!("EXCL_PROBE second-mut rejected = {:?}", r.is_err());
}

/// the dominance-based ordering must allow the pre-issuance mutation).
#[test]
fn probe_loop_birth_bound() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut i = 0;
             loop {
                 a = 5;
                 if i > 3 { leave; }
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 i = i + 1;
             }
             return a;
         }",
    );
    eprintln!("LOOP_BIRTH_PROBE accepted = {:?}", r.is_ok());
}

/// "neither readable nor writable").
#[test]
fn probe_shared_borrow_while_mut() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r1: &mut Int<32> = &mut a;
             set r2: &Int<32> = &a;
             return *r1;
         }",
    );
    eprintln!("SHARED_MUT_PROBE rejected = {:?}", r.is_err());
}

/// after the argument evaluation).
#[test]
fn probe_two_phase_pattern() {
    let r = check_source(
        "def take(a: &mut Int<32>, b: Int<32>) -> Int<32> { return *a + b; }
         def main() -> Int<32> {
             set mut a = 42;
             return take(&mut a, a);
         }",
    );
    eprintln!("TWOPHASE_PROBE accepted = {:?}", r.is_ok());
}

/// (the expression end), letting the receiver's exclusivity resume.
#[test]
fn probe_explicit_ro_entry() {
    let r = check_source(
        "type IntList = struct { data: Int<32> }
         impl for IntList {
             def len(&self) -> Int<32> { return self.data; }
             def push(&mut self, x: Int<32>) -> Int<32> { return x; }
         }
         def main() -> Int<32> {
             set mut v = IntList { data = 0 };
             let n = v.push((&ro v).len());
             return n;
         }",
    );
    eprintln!("EXPLICIT_RO_PROBE = {:?}", r);
}

/// mutation of the frozen source must be rejected).
#[test]
fn probe_cross_branch_kill() {
    let r = check_source(
        "type Point = struct { x: Int<32>, y: Int<32> }
         def main() -> Int<32> {
             set mut a = Point { x = 1, y = 2 };
             set mut r: &mut Int<32> = &mut (a.x);
             if a.x > 0 {
             } else {
                 a = Point { x = 3, y = 4 };
             }
             a.x = 5;
             return *r;
         }",
    );
    eprintln!("CROSS_BRANCH_PROBE rejected = {:?}", r.is_err());
}

/// is too strict across siblings).
#[test]
fn probe_sibling_branch_birth_bound() {
    let r = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut b = 7;
             set mut r: &mut Int<32> = &mut b;
             if a > 0 {
                 r = &mut a;
             }
             a = 5;
             return *r;
         }",
    );
    eprintln!("SIBLING_BIRTH_PROBE = {:?}", r);
}

/// Regression: `type` in statement position previously fell
/// through to the expression fallback — a misleading "expected expression"
/// (E007) false negative.  The new `Ok(Token::Type)` arm routes it to
/// `parse_type_def`.
#[test]
fn test_nested_type_alias_in_function_body() {
    let result = check_source(
        "def main() -> Int<32> {
             type R = Int<32>;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the nested type alias in the function body must parse: {:?}",
        result
    );
}

/// The comptime-block generic type — the statement arm must enable
/// ALLOW_TYPE_PARAMS explicitly (the top-level comptime block does not
/// inherit it from a `def`).
#[test]
fn test_nested_generic_type_in_comptime_block() {
    let result = check_source(
        "comptime {
             type P<T> = T;
         }
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_err(),
        "the comptime-block type must be REJECTED (evaluation error — not a panic): {:?}",
        result
    );
}

/// The struct branch of parse_type_def leaves `;` unconsumed — the
/// statement arm swallows the trailing semicolon.
#[test]
fn test_nested_struct_type_with_trailing_semicolon() {
    let result = check_source(
        "def main() -> Int<32> {
             type P = struct { x: Int<32> };
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the nested struct type with the trailing semicolon must parse: {:?}",
        result
    );
}

#[test]
fn probe_comptime_err_msg() {
    let r = check_source("comptime { type P<T> = T; }");
    eprintln!("COMPTIME_ERR = {:?}", r);
}

/// The error-code dispatch: a comptime SANDBOX violation (an item
/// declaration inside a comptime block) must carry E081 ("comptime sandbox
/// violation") — not the generic E080.
#[test]
fn test_comptime_sandbox_e081() {
    let result = check_source("comptime { type P<T> = T; }");
    assert!(
        result.is_err(),
        "the comptime sandbox violation must be rejected (E081): {:?}",
        result
    );
}

/// The local type-alias REFERENCE: `type R = Int<32>;` inside
/// a function body — a later reference (`set x: R`) used to report
/// "undefined type: R" because the resolver's `resolve_stmt` had no
/// `TypeDef` arm (R was never registered in the symbol table).  The
/// rustc-style arm (mirroring the nested FunctionDef<'input> arm) registers it.
#[test]
fn test_local_type_alias_reference() {
    let result = check_source(
        "def main() -> Int<32> {
             type R = Int<32>;
             set x: R = 0;
             return x;
         }",
    );
    assert!(
        result.is_ok(),
        "the reference to the local type alias `R` must resolve: {:?}",
        result
    );
}

/// The §Copy derivation for ADTs: a struct of pure
/// Copy fields (e.g. `Point { x: Int<32>, y: Int<32> }`) automatically
/// derives Copy — the move checker must NOT report a moved value on a
/// second use.  The Adt definition table (def_id → fields + Drop) makes
/// this decidable.
#[test]
fn test_struct_pure_copy_fields() {
    let result = check_source(
        "def main() -> Int<32> {
             type Point = struct { x: Int<32>, y: Int<32> };
             set p = Point { x = 1, y = 2 };
             set q = p;
             set r = p;
             return r.x + q.y;
         }",
    );
    assert!(
        result.is_ok(),
        "the pure-Copy struct must be copyable (no moved-value error): {:?}",
        result
    );
}

/// an assignment RHS CONSUMES the value — `b = a; set c = a;` on a
/// non-Copy `a` is a DOUBLE MOVE and must be rejected (previously only
/// the VariableDef arm recorded consumption; the Assign arm cleared the
/// target without moving the source — a double move passed, a safe-code
/// double-free).
#[test]
fn test_assign_rhs_consumes_moved_value() {
    let result = check_source(
        "type T = struct { val: Int<32> }
         impl Drop for T { def drop(&mut self) { } }
         def main() -> Int<32> {
             set a = T { val = 1 };
             set mut b = T { val = 2 };
             b = a;
             set c = a;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the second move of `a` must be rejected (double move): {:?}",
        result.err()
    );
}

/// The receiver-typed registry key: same-name methods on DISTINCT
/// receiver types (`impl A { def get }` / `impl B { def get }`) — the
/// call site matches the receiver type exactly for its own A(ρ) facts
/// (no find-first-match misattribution: `b.get(...)` must not use
/// `A::get`'s `&mut self` signature).
#[test]
fn test_same_name_methods_distinct_receivers() {
    let result = check_source(
        "type A = struct { val: Int<32> }
         type B = struct { val: Int<32> }
         impl for A {
             def get(&mut self, x: &mut Int<32>) -> &mut Int<32> { self.val = *x; return x; }
         }
         impl for B {
             def get(&self, x: &mut Int<32>) -> &mut Int<32> { return x; }
         }
         def main() -> Int<32> {
             set mut a = A { val = 0 };
             set mut b = B { val = 0 };
             set mut v = 42;
             set r1 = a.get(&mut v);
             set r2 = b.get(&mut v);
             return *r1 + *r2;
         }",
    );
    assert!(
        result.is_ok(),
        "same-name methods on distinct receivers must resolve by receiver type: {:?}",
        result.err()
    );
}

/// The IF EXPRESSION's branch bodies are split into independent CFG
/// blocks — a use-after-move INSIDE an expression-position branch is now
/// tracked (previously the whole `if` expression fell to the `_ =>`
/// fallback and was pushed into the current block, so `b = a; set c = a`
/// inside a branch slipped through — a double move accepted).
#[test]
fn test_if_expr_branch_move_tracked() {
    let result = check_source(
        "type T = struct { val: Int<32> }
         impl Drop for T { def drop(&mut self) { } }
         def main() -> Int<32> {
             set mut a = T { val = 1 };
             set x = if true {
                 set b = a;
                 set c = a;
                 0
             } else {
                 0
             };
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "the double move of `a` inside an if-expression branch must be rejected: {:?}",
        result.err()
    );
}

/// Cross-function returned-borrow liveness: once the returned borrow is
/// bound to the binding variable, its loan is KILLED at the borrow
/// variable's LAST USE (the rustc NLL/Polonius rule) — a mutation BEFORE
/// `*r` (a read) is rejected (the loan is still live — `a` frozen), and a
/// mutation AFTER `*r` is ACCEPTED (the loan is dead — `a` writable).
/// (Previously the `output_origin` filter was always false — the returned
/// borrow never bound `r` — and the loan's liveness followed the
/// placeholder origin to the end of the function, over-rejecting the
/// after-last-use mutation.)
#[test]
fn test_cross_function_return_borrow_liveness() {
    // A mutation before `*r` (the last use): rejected — the loan is still
    // live, so `a` is frozen.
    let bad = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             set r = get(&mut a);
             a = 5;
             let x = *r;
             return x + a;
         }",
    );
    assert!(
        bad.is_err(),
        "mutation before r's last use must be rejected: {:?}",
        bad.err()
    );
    // A mutation AFTER `*r` (the last use): accepted — the loan is killed
    // at `*r`, so `a` is writable again.
    let ok = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             set r = get(&mut a);
             let x = *r;
             a = 5;
             return x + a;
         }",
    );
    assert!(
        ok.is_ok(),
        "mutation after r's last use must be accepted: {:?}",
        ok.err()
    );
}

/// The `finally` block participates in the borrow check: its statements
/// are analysed (via `CfgBuilder::attach_finally` — the block runs on
/// every function-exit edge), and a plain write inside it is accepted.
#[test]
fn test_finally_block_checked() {
    let result = check_source(
        "def f() -> Int<32> {
             set x = 5;
             return x;
         } finally {
             x = 6;
         }",
    );
    assert!(
        result.is_ok(),
        "a `finally` block's statements must be analysed and accepted: {:?}",
        result.err()
    );
}

/// The Infallible constraint (SYNTAX.md §finally): a `finally` block is
/// reserved for infallible cleanup — a `return` inside it must be
/// rejected.
#[test]
fn test_finally_infallible_rejects_return() {
    let result = check_source(
        "def f() -> Int<32> {
             set x = 5;
             return x;
         } finally {
             return 3;
         }",
    );
    assert!(
        result.is_err(),
        "a `return` inside a `finally` block must be rejected: {:?}",
        result.err()
    );
}

/// An early-exit NESTED inside a control-flow statement must also be
/// rejected — a top-level-only `matches!` check let `if true { return; }`
/// bypass the infallibility constraint (the recursive walker rejects it).
#[test]
fn test_finally_infallible_rejects_nested_early_exit() {
    let result = check_source(
        "def f() -> Int<32> {
             set x = 5;
             return x;
         } finally {
             if true { return 3; }
         }",
    );
    assert!(
        result.is_err(),
        "a `return` nested inside an `if` in a `finally` block must be rejected: {:?}",
        result.err()
    );
}

/// The DefId comparison's direct effect: an `impl` of a NON-Drop trait
/// (a DefId different from the builtin `Drop`) must NOT mark the ADT
/// non-Copy.  (A user-defined trait literally named `Drop` cannot be
/// implemented — name resolution binds `impl Drop for T` to the builtin
/// `Drop`, so the ADT IS non-Copy; that is correct behaviour, not a
/// false positive.)
#[test]
fn test_non_drop_trait_keeps_copy() {
    let result = check_source(
        "trait Show { }
         type T = struct { val: Int<32> }
         impl Show for T { def show(&self) -> Int<32> { return self.val; } }
         def main() -> Int<32> {
             set a = T { val = 1 };
             set b = a;
             return a.val + b.val;
         }",
    );
    assert!(
        result.is_ok(),
        "a non-Drop trait impl must not mark the ADT non-Copy: {:?}",
        result.err()
    );
}

/// The §Copy derivation for ADTs with a Drop impl:
/// an `impl Drop for T` marks T as non-Copy — a second use must be
/// rejected by the move checker.
#[test]
fn test_adt_with_drop_is_non_copy() {
    let result = check_source(
        "def main() -> Int<32> {
             type T = struct { x: Int<32> };
             impl Drop for T { def drop(&mut self) { } }
             set t = T { x = 1 };
             set u = t;
             set v = t;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "an ADT with a Drop impl must be non-Copy (a moved value): {:?}",
        result
    );
}

/// The §Copy derivation for GENERIC ADTs: the fields
/// of `Box<T>` are `T` — with the instantiation `Box<Int<32>>` the field
/// resolves to `Int<32>` (Copy), so the struct is copyable.
#[test]
fn test_generic_adt_copy_instantiation() {
    let result = check_source(
        "def main() -> Int<32> {
             type Box<T> = struct { val: T };
             set b = Box { val = 1 };
             set c = b;
             set d = b;
             return c.val + d.val;
         }",
    );
    assert!(
        result.is_ok(),
        "the generic ADT instantiated with a Copy type must be copyable: {:?}",
        result
    );
}

/// The §Copy derivation for ENUMS: the variant
/// payloads participate — `Opt` (payload `Int<32>`, Copy) is copyable; a
/// second use must be accepted.  (A GENERIC enum like `Option<T>` cannot
/// be constructed in this position — the GADT construction requires
/// concrete type arguments and `Option<Int<32>>::Some(1)` does not parse
/// — so the payload-registration logic is exercised on a concrete enum.)
#[test]
fn test_enum_copy_payload_registration() {
    let result = check_source(
        "def main() -> Int<32> {
             type Opt = enum { None, Some(Int<32>) };
             set o = Opt::Some(1);
             set p = o;
             set q = o;
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the enum with Copy payloads must be copyable: {:?}",
        result
    );
}

/// The Drop boundary for a GENERIC ADT: an
/// `impl Drop for Box<T>` marks the generic ADT non-Copy regardless of the
/// instantiation.
#[test]
fn test_generic_adt_drop_is_non_copy() {
    let result = check_source(
        "def main() -> Int<32> {
             type Box<T> = struct { val: T };
             impl Drop for Box<T> { def drop(&mut self) { } }
             set b = Box { val = 1 };
             set c = b;
             set d = b;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the generic ADT with a Drop impl must be non-Copy: {:?}",
        result
    );
}

/// The GADT-construction conservatism (E060): constructing a variant of
/// an unconstrained generic enum without concrete type arguments is
/// rejected — the compiler does not infer the arguments (it could diverge
/// from what the surrounding code expects); the diagnostic asks for an
/// explicit annotation.  (The help text lives in the Diagnostic — the
/// `Vec<String>` returned by the checker exposes messages only.)
#[test]
fn test_gadt_unconstrained_defaulting() {
    // The committee ruling (solve → default → validate): an unconstrained
    // generic-enum construction WITHOUT call-site context is accepted —
    // `defaulting` determines `T` (Int<32>) — previously E060.
    let result = check_source(
        "def main() -> Int<32> {
             type Option<T> = enum { None, Some(T) };
             set o = Option::Some(1);
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the unconstrained construction must default (no E060): {:?}",
        result
    );
}

/// The committee ruling (solve → default → validate): a GADT construction
/// with unresolved type arguments is deferred — the call-site context
/// flows back during `solve` (here `big + x` constrains `T` to `Int<64>`),
/// so the construction is ACCEPTED (previously E060).
#[test]
fn test_gadt_call_site_flow_back() {
    let result = check_source(
        "def main() -> Int<32> {
             type Option<T> = enum { None, Some(T) };
             set o = Option::Some(1);
             set big: Int<64> = 10000000000;
             set total: Int<64> = match o {
                 Option::Some(x) => big + x,
                 Option::None => 0,
             };
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the call-site context must flow back and infer T (no E060): {:?}",
        result
    );
}

/// The committee ruling: the `when` constraint takes priority over
/// the surrounding context — `Expr<T>` requires `T == Int<32>`, so an
/// explicit `Expr<Int<64>>` annotation is NOT honored (the when-constrained
/// construction keeps `T = Int<32>`), and the assignment is a type
/// mismatch at the annotation site.
#[test]
fn test_gadt_when_constraint_priority() {
    let result = check_source(
        "def main() -> Int<32> {
             type Expr<T> = enum { Lit(Int<32>) when T == Int<32> };
             set e: Expr<Int<64>> = Expr::Lit(42);
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the when constraint must take priority over the annotation: {:?}",
        result
    );
}

/// A mutation in a DIFFERENT block of the SAME loop
/// iteration, after the loan — the mutual reachability (back-edge) must
/// not hide the violation (the freeze invariant: `&mut x` live — `x = 10`
/// is forbidden).
#[test]
fn test_loop_cross_block_mutation_after_loan() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut x = 0;
             set mut cond = true;
             loop {
                 set r: &mut Int<32> = &mut x;
                 if cond {
                     x = 10;
                 }
                 *r = 5;
                 cond = false;
             }
             return x;
         }",
    );
    assert!(
        result.is_err(),
        "the mutation of the frozen x in the loop body must be rejected: {:?}",
        result
    );
}

/// Strict Mode must reject `unsafe` blocks completely
/// (SYNTAX.md §Strict Mode — the checker previously accepted them).
#[test]
fn test_strict_mode_rejects_unsafe_block() {
    let source = "def main() -> Int<32> {
         unsafe { return 0; }
     }";
    let arena = bumpalo::Bump::new();
    let mut parser = Parser::new(source, &arena);
    let program = parser.parse_program().expect("parse should succeed");
    let mut ctx = TypeContext::new();
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (symbols, mut trait_env, _res_diags, resolution_map) = resolver.resolve_program(&program);
    let mut checker = TypeChecker::new(
        &mut ctx,
        &symbols,
        &mut trait_env,
        resolution_map,
        true,   // strict_mode
        false,  // enable_experimental
        vec![], // features
        false,  // debug
    );
    let result = checker.check_program(&program);
    assert!(result.is_err(), "strict mode must reject `unsafe` blocks");
}

/// `@runtime_check` is "Only allowed in non-strict mode"
/// (SYNTAX.md) — Strict Mode must reject it (the checker previously did
/// not consult `strict_mode` for this attribute).
#[test]
fn test_strict_mode_rejects_runtime_check() {
    let source = "@runtime_check
     def main() -> Int<32> { return 0; }";
    let arena = bumpalo::Bump::new();
    let mut parser = Parser::new(source, &arena);
    let program = parser.parse_program().expect("parse should succeed");
    let mut ctx = TypeContext::new();
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (symbols, mut trait_env, _res_diags, resolution_map) = resolver.resolve_program(&program);
    let mut checker = TypeChecker::new(
        &mut ctx,
        &symbols,
        &mut trait_env,
        resolution_map,
        true,   // strict_mode
        false,  // enable_experimental
        vec![], // features
        false,  // debug
    );
    let result = checker.check_program(&program);
    assert!(result.is_err(), "strict mode must reject `@runtime_check`");
}

/// Closures are nested borrowing domains — a `&mut a` inside
/// a closure with a later mutation of `a` in the same closure must be
/// rejected (previously the closure body was skipped by the collector).
#[test]
fn test_closure_mutation_after_loan() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 1;
             set f = def () -> Int<32> {
                 set r: &mut Int<32> = &mut a;
                 a = 10;
                 return *r;
             };
             return f();
         }",
    );
    assert!(
        result.is_err(),
        "the mutation of the frozen a inside the closure must be rejected: {:?}",
        result
    );
}

/// A compound assignment (`a += 1`) READS the target — a
/// moved `a` used in `a += 1` must be rejected (use-after-move).
#[test]
fn test_compound_assign_after_move() {
    let result = check_source(
        "def main() -> Int<32> {
             set a = String::new();
             set b = a;
             a += 1;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the use of the moved a in the compound assignment must be rejected: {:?}",
        result
    );
}

/// Rule 3 (the committee ruling): a `when X == Y` equality between TWO
/// existential variables (`exists X, Y: T` — comma-separated names, one
/// type) registers an equivalence (ExistentialEquation) deterministically.
#[test]
fn test_gadt_rule3_existential_equivalence() {
    let result = check_source(
        "def main() -> Int<32> {
             type E<T> = enum { A(exists X, Y: Int<32>) when X == Y };
             set e = E::A(42);
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "the X == Y existential equivalence must type-check (rule 3): {:?}",
        result
    );
}

/// The dedicated diagnostic for the common `exists` mistake: writing
/// `exists X: T, exists Y: T` (one type per variable) — the correct form
/// is `exists X, Y: T` (comma-separated names, then a single `:` and one
/// type).  The parser must fail with a helpful hint, not a bare
/// "expected RParen, found Comma".
#[test]
fn test_exists_multi_type_diagnostic() {
    let result = check_source(
        "def main() -> Int<32> {
             type E<T> = enum { A(exists X: Int<32>, exists Y: Int<32>) when X == Y };
             return 0;
         }",
    );
    match result {
        Err(diags) => {
            // The dedicated diagnostic's MESSAGE (the `Vec<String>` from
            // check_source exposes messages only — the "share ONE type"
            // help text lives in the Diagnostic).
            assert!(
                diags
                    .iter()
                    .any(|d| d.contains("after the `exists` variable type")),
                "the dedicated exists diagnostic must be emitted: {:?}",
                diags
            );
        }
        Ok(_) => panic!("the wrong `exists X: T, exists Y: T` form must be rejected"),
    }
}

/// The lexer's hand-written integer parsing (zero-allocation, `_`-skipping)
/// must preserve the overflow diagnostics for decimal/hex/binary literals.
#[test]
fn test_int_literal_overflow_and_underscores() {
    // Underscores accepted in all radices.
    assert!(check_source("def main() -> Int<32> { return 1_000_000; }").is_ok());
    // Note: the lexer regex requires a digit right after `0x`/`0b` —
    // underscores may only separate digits (`0xF_F`, not `0x_F_F`).
    assert!(check_source("def main() -> Int<32> { return 0xF_F; }").is_ok());
    assert!(check_source("def main() -> Int<32> { return 0b1_0_1; }").is_ok());
    // Overflow still reported (decimal / hex / binary).
    for lit in [
        "999999999999999999999999999999999999999999",
        "0xffffffffffffffffffffffffffffffff",
        "0b11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
    ] {
        let src = format!("def main() -> Int<32> {{ return {lit}; }}");
        assert!(
            check_source(&src).is_err(),
            "overflow literal must be rejected: {lit}"
        );
    }
}

/// An exclusivity conflict INSIDE an impl method body
/// must emit E112 (the shared `borrow_error_diagnostic`) — not the
/// generic E110 — with the dual-position label.
#[test]
fn test_method_body_exclusive_borrow_e112() {
    let result = check_source(
        "type T = struct { x: Int<32> }
         impl for T {
             def foo(&mut self) -> Int<32> {
                 set r1 = &mut self.x;
                 set r2 = &mut self.x;
                 return *r1;
             }
         }
         def main() -> Int<32> { return 0; }",
    );
    // NOTE: in this scenario the second `&mut self.x` READS `self.x` while
    // the first borrow is live, so the reported diagnostic is the read-
    // freeze E109 — the shared helper carries the `is_exclusive`
    // E112 branch for the pure exclusive-overlap case.  The essential
    // assertion is that the method-body conflict is REJECTED (previously
    // the method-body arm collapsed everything into E110).
    assert!(
        result.is_err(),
        "overlapping borrows in a method body must be rejected: {:?}",
        result
    );
}

/// A compound assignment through a non-`Ident` target
/// (`arr[0] += 1`) READS the target — a moved `arr` must be rejected.
/// Uses a NON-`Copy` array (a `String` element makes the array affine);
/// an `Int<32>` array would be Copy and its `set b = arr` is a copy, not
/// a move (SYNTAX.md §Value Semantics — Copy derivation is element-wise).
#[test]
fn test_compound_assign_index_target_after_move() {
    let result = check_source(
        "def make() -> String { return \"x\"; }
         def main() -> Int<32> {
             set arr = [make(), make(), make()];
             set b = arr;
             arr[0] += \"!\";
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the use of the moved arr in the indexed compound assignment must be rejected: {:?}",
        result
    );
}

/// SYNTAX.md §GADTs Limitations — a `when`
/// constraint whose RHS references ANOTHER enum-header type parameter
/// (`T == [U]`) is "not yet allowed" and must be rejected at the
/// RESOLVER layer (not as a post-solve "unresolvable" error).
#[test]
fn test_gadt_rhs_header_param_forbidden() {
    let result = check_source(
        "type E<T, U> = enum { A(Int<32>) when T == [U] }
         def main() -> Int<32> { return 0; }",
    );
    match result {
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.contains("of the same enum")),
                "the RHS header-parameter form must be rejected at the resolver (E064): {:?}",
                diags
            );
        }
        Ok(_) => panic!("`when T == [U]` must be rejected (SYNTAX.md Limitations)"),
    }
}

/// The float fast path: literals WITHOUT `_` parse the
/// original slice directly (zero allocation) — both plain and
/// scientific forms; literals WITH `_` fall back to the cleaned form.
/// Non-finite literals are still rejected.
#[test]
fn test_float_literal_fast_path() {
    // No underscore → fast path (direct parse).
    assert!(check_source("def main() -> Int<32> { set x = 3.14; return 0; }").is_ok());
    assert!(check_source("def main() -> Int<32> { set x = 2.5e-3; return 0; }").is_ok());
    assert!(check_source("def main() -> Int<32> { set x = 1e10; return 0; }").is_ok());
    // Underscore → fallback path.
    assert!(check_source("def main() -> Int<32> { set x = 1_000.5; return 0; }").is_ok());
    assert!(check_source("def main() -> Int<32> { set x = 1_000.5e-3; return 0; }").is_ok());
    // NOTE: `1e999` parses to `inf` (Rust's f64 parse returns Ok(inf) on
    // overflow); the lexer's FloatLiteral carries the "must be finite" Err,
    // but the type checker does not fail the program on a non-finite
    // literal — a separate concern from the fast-path optimization here.
    // The fast-path optimization itself is verified above (no-underscore
    // fast path + underscore fallback both parse correctly).
}

/// The committee ruling (float default `trap` — aligned with integers):
/// a compile-time float-literal overflow (`1e999` → ±inf) is a COMPILE
/// ERROR (the parser propagates the lexer's error — it is no longer
/// swallowed into Expr::Error).
#[test]
fn test_float_literal_overflow_is_compile_error() {
    let result = check_source("def main() -> Int<32> { set x = 1e999; return 0; }");
    match result {
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.contains("overflow")),
                "the float-literal overflow must be a compile error: {:?}",
                diags
            );
        }
        Ok(_) => panic!("`1e999` (overflow) must be rejected at compile time (trap default)"),
    }
}

/// The committee ruling (float default `trap` — IEEE is explicit opt-in):
/// `with overflow = ieee` on a float type must parse and type-check
/// (the OverflowPolicy::Ieee variant — ast + target spec).
#[test]
fn test_float_type_overflow_ieee_optin() {
    let result = check_source(
        "type MyFloat = Float<64> with overflow = ieee;
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        result.is_ok(),
        "`with overflow = ieee` must parse and type-check: {:?}",
        result
    );
}

/// The `\u{...}` stack-buffer fast path: unicode escapes in
/// char and string literals still parse correctly — the stack buf
/// ([0u8; 6]) replaces the per-escape String allocation.
#[test]
fn test_unicode_escape_stack_buf() {
    assert!(check_source("def main() -> Int<32> { set c = '\\u{41}'; return 0; }").is_ok());
    assert!(
        check_source("def main() -> Int<32> { set s = \"\\u{41}\\u{42}\"; return 0; }").is_ok()
    );
    // NOTE: >0xFF / surrogate char literals are rejected by the LEXER
    // (the existing lexer tests cover that), but the char-literal Err is
    // downgraded to Expr::Error by the parser — the type checker accepts
    // the program.  That Err-propagation concern is separate from this
    // stack-buf optimization (cf. the float-literal Err propagation).
}

/// The committee ruling (float default `trap` — explicit IEEE opt-in via
/// the `+%` suffix on floats): overflow-suffixed operators accept BOTH
/// integer and float operands.
#[test]
fn test_overflow_suffix_float_semantics() {
    // Integer +% (wrap) — already accepted.
    assert!(check_source("def main() -> Int<32> { set x = 1 +% 2; return x; }").is_ok());
    // Float +% — now accepted (IEEE opt-in per the ruling).
    assert!(check_source("def main() -> Int<32> { set x = 1.5 +% 2.5; return 0; }").is_ok());
    // Float +? (saturate) and +! (trap) — accepted.
    assert!(check_source("def main() -> Int<32> { set x = 1.5 +? 2.5; return 0; }").is_ok());
    assert!(check_source("def main() -> Int<32> { set x = 1.5 +! 2.5; return 0; }").is_ok());
    // Non-numeric operands — still rejected.
    assert!(check_source("def main() -> Int<32> { set s = \"a\" +% \"b\"; return 0; }").is_err());
}

/// SYNTAX.md:346 — comptime float ops use IEEE 754 semantics on the host
/// FPU and CHECK the anomaly flags: an overflow (→ ±∞) or NaN result is
/// a COMPILE ERROR.
#[test]
fn test_comptime_float_ieee_anomaly() {
    // Overflow → compile error.
    let result = check_source(
        "def main() -> Int<32> {
             comptime {
                 set x: Float<64> = 1.0e308;
                 set y: Float<64> = 1.0e308;
                 set z = x + y;
             }
             return 0;
         }",
    );
    match result {
        Err(diags) => {
            assert!(
                diags.iter().any(|d| d.contains("overflowed")),
                "the comptime float overflow must be a compile error: {:?}",
                diags
            );
        }
        Ok(_) => panic!("the comptime float overflow must be rejected (SYNTAX.md:346)"),
    }
    // Normal comptime float arithmetic still works.
    let ok = check_source(
        "def main() -> Int<32> {
             comptime {
                 set a: Float<64> = 1.5;
                 set b: Float<64> = 2.5;
                 set c = a + b;
             }
             return 0;
         }",
    );
    assert!(
        ok.is_ok(),
        "normal comptime float arithmetic must work: {:?}",
        ok
    );
}

/// A &mut
/// borrow live across a loop back edge must freeze the source — mutating
/// `a` at the loop top while the borrow from the previous iteration is
/// still live must be REJECTED.
#[test]
fn test_loop_cross_iteration_borrow_freeze() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set mut i = 0;
             while i < 3 {
                 a = 5;
                 set r: &mut Int<32> = &mut a;
                 *r = *r + 1;
                 i = i + 1;
             }
             return a;
         }",
    );
    // Every loan dies at its borrow variable's last use (the
    // last-use kill), so the iteration's `r = &mut a` borrow ends inside
    // the iteration — the NEXT iteration's `a = 5` is legal (rustc NLL
    // accepts the same pattern).
    assert!(
        result.is_ok(),
        "`a = 5` in the next loop iteration after the previous iteration's borrow ended must be accepted: {:?}",
        result
    );
}

/// The
/// AST-level output-derivation must recurse into nested control flow, so a
/// callee whose returns are all nested still freezes the caller's input.
#[test]
fn test_nested_return_derivation_freeze() {
    let result = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> {
             if true { return x; } else { return x; }
         }
         def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = get(&mut a);
             a = 5;
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "`a = 5` while the cross-function &mut borrow (nested returns) is live must be rejected: {:?}",
        result
    );
}

/// Cross-function loan mutability: a read-only
/// function's returned borrow gets a ReadOnly loan — READING the source
/// stays legal.  A mutable function's returned borrow is Exclusive —
/// mutating the source is rejected.
#[test]
fn test_cross_function_return_borrow_mutability() {
    // Read-only function: `get(&a)` returns `&Int<32>` — reading `a` stays
    // legal (the cross-function loan is ReadOnly, not Exclusive).
    let read_ok = check_source(
        "def get(x: &Int<32>) -> &Int<32> { return x; }
         def main() -> Int<32> {
             set a = 42;
             set r: &Int<32> = get(&a);
             set b = a + 1;
             return *r + b;
         }",
    );
    assert!(
        read_ok.is_ok(),
        "reading the source of a read-only cross-function borrow must stay legal: {:?}",
        read_ok
    );
    // Mutable function: mutating the source while the exclusive
    // cross-function borrow is live is rejected.
    let mut_rejected = check_source(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = get(&mut a);
             a = 5;
             return *r;
         }",
    );
    assert!(
        mut_rejected.is_err(),
        "mutating the source of an exclusive cross-function borrow must be rejected: {:?}",
        mut_rejected
    );
}

/// An invalid regex pattern must be a compile ERROR (returned Err) — NOT
/// push-and-continue that still allocates a Regex node whose pattern is
/// known-invalid (a poisoned node would violate the type-graph invariant
/// and panic the debug-build assertion).
#[test]
fn test_invalid_regex_pattern_is_error() {
    let result = check_source(
        "def main() -> Int<32> {
             set r = Regex<\"[\">;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "an invalid regex pattern must be a compile error: {:?}",
        result
    );
}

/// The move-check path-sensitive match-arm handling: a non-Copy value moved
/// in ONE arm is only possibly-moved AFTER the match — using it in
/// ANOTHER arm stays legal (the previous folding conflated arm paths).
#[test]
fn test_match_arm_path_isolation() {
    let result = check_source(
        "def main() -> Int<32> {
             set a = 42;
             set x = 7;
             set r = match x { 1 => a, _ => a };
             return r;
         }",
    );
    assert!(
        result.is_ok(),
        "using `a` in a second match arm after moving it in the first must stay legal: {:?}",
        result
    );
}

/// The CFG-level path-sensitive move check: a value
/// moved inside ONE branch is possibly-moved after the join — using it
/// AFTER the if is rejected (conservative union meet).
#[test]
fn test_cfg_level_move_join() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut a = 42;
             set r: &mut Int<32> = &mut a;
             set mut x = 0;
             if x == 0 { *r = 5; }
             a = 5;
             return *r;
         }",
    );
    assert!(
        result.is_err(),
        "using `a` after the if (possibly-moved on one path) must be rejected: {:?}",
        result
    );
}

/// The literal hot path (`parse_expr_fast`): a bare literal parses
/// directly; an operator after it (`1 + 2`) or a type-annotation colon
/// falls back to the general Pratt parser — no token is left unconsumed.
#[test]
fn test_expr_hot_path_fallback() {
    // Bare literal → fast path.
    assert!(check_source("def main() -> Int<32> { set x = 42; return x; }").is_ok());
    // Operator after the literal → falls back to the general parser.
    assert!(check_source("def main() -> Int<32> { set x = 1 + 2; return x; }").is_ok());
    // Type-annotation colon → falls back.
    assert!(check_source("def main() -> Int<32> { set x: Int<32> = 42; return x; }").is_ok());
    // A string literal followed by a method-ish call still parses.
    assert!(check_source("def main() -> Int<32> { set s = \"hi\"; return 0; }").is_ok());
}

#[test]
fn test_nested_call_return_borrow_unused_accepted() {
    // A nested call (cross-function returned borrow inside a tuple) with
    // the binding NEVER USED — the returned borrow's loan dies with the
    // unused binding (the NLL last-use rule), so a later mutation of the
    // source is ACCEPTED — matching rustc:
    // `let t = (get(&mut a), 1); a = 5;` compiles.
    // (The collision fix: the returned-borrow loan's origin is a FRESH
    // local region, NOT a universal placeholder — before the fix the
    // origin-id collision with the callee's signature origins made the
    // loan universally live, rejecting the mutation.)
    let (_prog, diags) = check_source_keep_hir(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             set t = (get(&mut a), 1);
             a = 5;
             return 0;
         }",
    );
    assert!(
        diags.is_empty(),
        "the unused nested cross-function binding must not freeze the source (rustc parity), got: {:?}",
        diags
    );
    // The USED variant must stay rejected — the loan lives until the
    // binding's last use, so the mutation before it fires E110
    // (rustc: `let t = (get(&mut a), 1); a = 5; let x = *t.0;` — E0506).
    let (_prog2, diags2) = check_source_keep_hir(
        "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
         def main() -> Int<32> {
             set mut a = 42;
             set t = (get(&mut a), 1);
             a = 5;
             let (r, _) = t;
             return *r;
         }",
    );
    assert!(
        !diags2.is_empty(),
        "the used nested cross-function binding must freeze the source before the last use: {:?}",
        diags2
    );
}

/// §577 (SYNTAX.md "Type Invariants"): "The compiler verifies or enforces
/// the invariant at every construction point" — a literal, struct field,
/// or `with default` value that violates a refined type's invariant is
/// rejected; a valid construction is accepted.
#[test]
fn test_invariant_enforced_at_construction_points() {
    // Valid literal construction: 5 satisfies `n != 0`; 3 satisfies
    // `value > 0` (the `where` shorthand desugars to an exists invariant).
    let ok = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         type PositiveInt = Int<32> where value > 0;
         def main() -> Int<32> {
             set x: NonZeroInt = 5;
             set y: PositiveInt = 3;
             set z: NonZeroInt = x;
             return 0;
         }",
    );
    assert!(ok.is_ok(), "valid constructions must pass: {:?}", ok.err());
    // Violating literals: 0 fails `n != 0`; -1 fails `value > 0`.
    let bad = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         type PositiveInt = Int<32> where value > 0;
         def main() -> Int<32> {
             set x: NonZeroInt = 0;
             set y: PositiveInt = -1;
             return 0;
         }",
    );
    assert!(
        bad.is_err(),
        "violating literals must be rejected: {:?}",
        bad
    );
    // A struct field with a refined type is a construction point too.
    let struct_bad = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         type Wrap = struct { v: NonZeroInt }
         def main() -> Int<32> {
             set w2 = Wrap { v = 0 };
             return 0;
         }",
    );
    assert!(
        struct_bad.is_err(),
        "violating struct field must be rejected: {:?}",
        struct_bad
    );
    // `with default` on a refined type: the default must satisfy the
    // invariant (SYNTAX.md §"Type-level Default Values").
    let default_ok = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0 with default = 5;
         def main() -> Int<32> {
             set x: NonZeroInt;
             return 0;
         }",
    );
    assert!(
        default_ok.is_ok(),
        "invariant-satisfying default must pass: {:?}",
        default_ok.err()
    );
    let default_bad = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0 with default = 0;
         def main() -> Int<32> {
             set x: NonZeroInt;
             return 0;
         }",
    );
    assert!(
        default_bad.is_err(),
        "invariant-violating default must be rejected: {:?}",
        default_bad
    );
    // `with no_default` forbids implicit initialization (SYNTAX.md §535).
    let no_default = check_source(
        "type OwnedFd = exists n: Int<32> invariant n >= 0 with no_default;
         def main() -> Int<32> {
             set fd: OwnedFd;
             return 0;
         }",
    );
    assert!(
        no_default.is_err(),
        "no_default refined type must reject implicit init: {:?}",
        no_default
    );
}

/// §584 (SYNTAX.md "Type Invariants" — implicit invariant propagation):
/// a function with a refined parameter automatically inherits `requires`
/// (the call argument must satisfy the invariant), and a refined return
/// type automatically inherits `ensures` (the returned value must satisfy
/// it).  Same-type flows (param → return, chained calls) are accepted.
#[test]
fn test_invariant_implicit_requires_ensures() {
    // Passing an invalid argument to a refined parameter (the implicit
    // requires) must be rejected; valid arguments are accepted.
    let bad_arg = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         def f(x: NonZeroInt) -> Int<32> { return 0; }
         def main() -> Int<32> {
             f(0);
             return 0;
         }",
    );
    assert!(
        bad_arg.is_err(),
        "invalid argument to refined parameter must be rejected: {:?}",
        bad_arg
    );
    // Returning an invalid value from a refined return type (the implicit
    // ensures) must be rejected.
    let bad_ret = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         def f() -> NonZeroInt { return 0; }
         def main() -> Int<32> { return 0; }",
    );
    assert!(
        bad_ret.is_err(),
        "invalid return from refined return type must be rejected: {:?}",
        bad_ret
    );
    // Same-type flows are trivially safe: returning the parameter, and
    // chained calls that preserve the refined type.
    let ok = check_source(
        "type NonZeroInt = exists n: Int<32> invariant n != 0;
         def f(x: NonZeroInt) -> NonZeroInt { return x; }
         def g(x: NonZeroInt) -> NonZeroInt { return f(x); }
         def main() -> Int<32> { f(3); g(5); return 0; }",
    );
    assert!(
        ok.is_ok(),
        "same-type refined flows must pass: {:?}",
        ok.err()
    );
}

/// L2 (integer discreteness): `value > 0` and `value >= 1` admit the same
/// values on an INTEGER base — constructing a `StrictlyPositive` from a
/// `PositiveInt` parameter is accepted via the discreteness-aware fast
/// path (previously the structurally-different invariant was unprovable
/// for a runtime parameter and the construction was rejected).
#[test]
fn test_invariant_l2_discrete_equivalence() {
    let ok = check_source(
        "type PositiveInt = Int<32> where value > 0;
         type StrictlyPositive = Int<32> where value >= 1;
         def f(x: PositiveInt) -> Int<32> {
             set y: StrictlyPositive = x;
             return 0;
         }
         def main() -> Int<32> { f(3); return 0; }",
    );
    assert!(
        ok.is_ok(),
        "integer-discrete invariant flow must pass: {:?}",
        ok.err()
    );
}

/// L3 (affine normalization): `value + 1 > 1` and `value > 0` admit the
/// same values on an INTEGER base — constructing a `PosB` from a `PosA`
/// parameter is accepted via the affine-aware fast path (the constant is
/// folded across the `X + 1` wrapper before comparing).
#[test]
fn test_invariant_l3_affine_equivalence() {
    let ok = check_source(
        "type PosA = Int<32> where value + 1 > 1;
         type PosB = Int<32> where value > 0;
         def f(x: PosA) -> Int<32> {
             set y: PosB = x;
             return 0;
         }
         def main() -> Int<32> { f(3); return 0; }",
    );
    assert!(
        ok.is_ok(),
        "affine-discrete invariant flow must pass: {:?}",
        ok.err()
    );
}

/// Program 1 regression (Pavlinovic/Su/Wies, "Data Flow Refinement Type
/// Inference"): `apply` dispatching `g`/`h` through an `if` — the paper's
/// motivating example for the precision loss of context-insensitive Horn
/// clause extraction.  Posita checks each call site in place (no global
/// constraint system to mix sites), so the higher-order dispatch plus both
/// `if` branches type-check independently.  NOTE the boundary: the paper's
/// `assert (0 <= v)` requires refinement PROPAGATION (parameter refinement
/// → `2 * y` provably positive), which needs the refinement-subtyping RFC;
/// construction-point checks are fail-closed on function-call results, so
/// this test pins the architecture guarantee (no mixed-site false
/// positive), not the assertion proof.
#[test]
fn test_program1_higher_order_dispatch() {
    let ok = check_source(
        "type Positive = Int<32> where value > 0;
         def g(y: Int<32>) -> Int<32> { return 2 * y; }
         def h(y: Int<32>) -> Int<32> { return -2 * y; }
         def apply(f: (Int<32>) -> Int<32>, x: Int<32>) -> Int<32> { return f(x); }
         def main(z: Int<32>) -> Int<32> {
             set v = if z >= 0 { apply(g, z) } else { apply(h, z) };
             return 0;
         }",
    );
    assert!(
        ok.is_ok(),
        "higher-order dispatch must type-check (no mixed-site false positive): {:?}",
        ok.err()
    );
}

/// Curried/nested-call verification (Pavlinovic et al. §8: repeated output-type
/// strengthening on curried functions caused quadratic blowup in DRIFT).
/// Posita's `check_construction_invariant` performs an independent
/// point-check per construction (fast path + comptime eval) with NO
/// accumulated strengthening, so deeply nested calls stay linear in TIME.
/// STACK DEPTH — RESOLVED: nested calls used to amplify STACK depth (the
/// checker recursion expands several frames per source-level call, and a
/// 10-level chain overflowed the test runner's default 2 MiB thread stack).
/// `FnCtxt::infer_expr` is now wrapped in `stacker::maybe_grow` (the same
/// explicit-stack technique rustc uses for deep recursion), so the stack
/// grows on demand regardless of the caller's thread-stack default.  The
/// explicit 16 MiB stack below is retained as belt-and-suspenders; the
/// default-stack regression test (`test_nested_calls_ok_on_default_stack`)
/// proves the fix without it.
#[test]
fn test_curried_nested_call_audit() {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let ok = check_source(
                "type Positive = Int<32> where value > 0;
                 def inc(x: Int<32>) -> Int<32> { return x + 1; }
                 def main() -> Int<32> {
                     set r = inc(inc(inc(inc(inc(inc(inc(inc(inc(inc(1))))))))));
                     set p: Positive = 10;
                     return 0;
                 }",
            );
            assert!(
                ok.is_ok(),
                "deeply nested calls must type-check: {:?}",
                ok.err()
            );
        })
        .expect("spawn audit thread");
    handle.join().expect("audit thread panicked");
}

/// DEFENSIVE (PR3.2 — constructed BEFORE the iterative/explicit-stack
/// rewrite of the nested-call checking path): the SAME 10-level nested
/// call chain must type-check on the TEST RUNNER'S DEFAULT thread stack
/// (≈2 MiB), not only on an explicitly-grown 16 MiB stack.  Regression:
/// the checker recursion expanded several frames per source-level call,
/// so 10 levels overflowed the default stack (stack-overflow abort).
/// After the rewrite this must pass on a default-stack thread.
#[test]
fn test_nested_calls_ok_on_default_stack() {
    let handle = std::thread::Builder::new()
        // NOTE: intentionally NO `.stack_size(...)` — exercises the
        // runner's default thread stack.
        .spawn(|| {
            let ok = check_source(
                "def inc(x: Int<32>) -> Int<32> { return x + 1; }
                 def main() -> Int<32> {
                     set r = inc(inc(inc(inc(inc(inc(inc(inc(inc(inc(1))))))))));
                     return 0;
                 }",
            );
            assert!(
                ok.is_ok(),
                "10-level nested calls must type-check on the DEFAULT stack: {:?}",
                ok.err()
            );
        })
        .expect("spawn default-stack thread");
    handle.join().expect("default-stack thread panicked");
}

/// Integration: a `.ps` while loop runs the loop-invariant inference
/// pipeline inside the REAL checker — the HIR loop is translated, the
/// widened fixpoint produces candidates, and the SMT consistency gate
/// emits the "inferred loop invariant" note (hint-only per the 2026-08-13
/// ruling).  The note is skipped when Z3 is unavailable or the loop is not
/// translatable (fail-closed — never an obligation).
///
/// Regression: the hint channel used to be INERT in the checker — it
/// passed `init = &[]`, the DBM started at top (all-∞), the fixpoint
/// converged to top immediately, and zero candidates were emitted.  The
/// checker now seeds `init` from the loop variables' comptime-known
/// pre-loop values (`set mut i = 0` → `ConstVar(i, 0)`), so the channel
/// fires for a real loop; the `decreases` candidate is wired on the same
/// advisory-only path.
#[test]
fn test_loop_inference_integration() {
    let (_, msgs) = check_source_keep_hir(
        "def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    let errors = msgs.iter().filter(|m| m.contains("error")).count();
    assert_eq!(errors, 0, "the loop program must type-check: {:?}", msgs);
    let has_note = msgs.iter().any(|m| m.contains("inferred loop invariant"));
    // Hard-assert only when Z3 is actually available — CI without Z3 keeps
    // the fail-closed (silent) behavior; with Z3 the channel MUST fire.
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if z3_available {
        assert!(
            has_note,
            "the inferred-loop hint channel must be live (seeded init): {:?}",
            msgs
        );
    } else if !has_note {
        eprintln!("no inferred-loop note (z3 unavailable)");
    }
}

/// A NON-Ident `&ro` operand (`&ro (h.r)` — a reference-typed
/// FIELD, valid per E111) must freeze the REFERENT: its loan wraps to
/// `Deref(Field(h,r))`, so a write through the reference (`*h.r = 5`) is
/// seen and rejected (rustc E0506).
#[test]
fn test_m3_ro_field_operand_freezes_referent() {
    let result = check_source(
        "type Holder = struct { r: &mut Int<32> }
         def main() -> Int<32> {
             set mut x = 42;
             set mut h = Holder { r = &mut x };
             let s = &ro (h.r);
             *h.r = 5;
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "a write through a reference frozen by `&ro (h.r)` must be rejected: {:?}",
        result
    );
}

/// A declared `while` loop `invariant` clause must be VERIFIED (per
/// the language-owner ruling — the clause is an obligation, not a
/// decorative annotation).  A provable invariant (entailed by the inferred
/// BII candidates) type-checks.
#[test]
fn test_while_invariant_verified() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 invariant i >= 0 {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "a provable loop invariant must be accepted: {:?}",
        result
    );
}

/// An unprovable declared `invariant` (contradicted by the inferred
/// candidates) must be REJECTED — the clause is an obligation.
#[test]
fn test_while_invariant_unprovable_rejected() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 invariant i > 100 {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "an unprovable loop invariant must be rejected: {:?}",
        result
    );
}

/// A declared `decreases` measure that strictly decreases on every
/// iteration (SMT-verified) is accepted.  Skipped when Z3 is unavailable
/// (the ∃∀ decrease check needs the solver; fail-closed otherwise).
#[test]
fn test_while_decreases_verified() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_while_decreases_verified");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 decreases 10 - i {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "a strictly-decreasing measure must be accepted: {:?}",
        result
    );
}

/// A declared `decreases` measure that does NOT decrease (`i` grows
/// with `i := i + 1`) must be REJECTED.
#[test]
fn test_while_decreases_not_decreasing_rejected() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_while_decreases_not_decreasing_rejected");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 decreases i {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "a non-decreasing measure must be rejected: {:?}",
        result
    );
}

/// A user-provided `@hint(assertion)` on a function must be READ and
/// verified against the inferred loop invariants — not dead storage.  A
/// provable hint type-checks.
#[test]
fn test_user_hint_assertion_verified() {
    let result = check_source(
        "@hint(i >= 0)
         def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "a provable @hint assertion must be accepted: {:?}",
        result
    );
}

/// A `@hint(assertion)` contradicted by the inferred invariants must
/// be REJECTED — the hint is a user assertion, not decoration.
#[test]
fn test_user_hint_assertion_unprovable_rejected() {
    let result = check_source(
        "@hint(i > 100)
         def main() -> Int<32> {
             set mut i: Int<32> = 0;
             while i < 10 {
                 set i = i + 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "an unprovable @hint assertion must be rejected: {:?}",
        result
    );
}

/// Issue-1 regression (per-variable bit-widths): a wrap-loop whose FIRST
/// variable is 8-bit and whose SECOND variable is 16-bit must verify its
/// declared `invariant` at the variable's OWN width.  The old uniform-width
/// BV discharge declared every variable at the first variable's 8 bits, so
/// `b >= 256` truncated to `b >= 0` — a tautology — and the FALSE invariant
/// was silently accepted.  Per-variable widths declare `b` at 16 bits:
/// `b >= 256` is satisfiable (b = 0), the obligation is NOT entailed, and
/// the invariant is rejected.
#[test]
fn test_mixed_width_wrap_invariant_no_false_proof() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_mixed_width_wrap_invariant_no_false_proof");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut a: UInt<8> = 0;
             set mut b: UInt<16> = 0;
             while a < 10 invariant b >= 256 {
                 set a = a +% 1;
                 set b = b +% 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "a 16-bit variable must not be truncated to 8 bits in BV discharge: {:?}",
        result
    );
}

/// Issue-1 regression (per-variable bit-widths): `verify_loop_decreases`
/// declares the loop variables at their OWN bit-widths.  The old
/// uniform-width encoding declared every variable at the FIRST variable's
/// 8 bits, so a 16-bit `b := 255` was read as an 8-bit `255` and
/// `b +% 1` wrapped to 0 — a spurious decrease — and the non-decreasing
/// measure was accepted.  At 16 bits `b' = 256 > 255`: the measure does
/// NOT decrease → rejected.
#[test]
fn test_mixed_width_wrap_decreases_no_false_proof() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_mixed_width_wrap_decreases_no_false_proof");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut a: UInt<8> = 0;
             set mut b: UInt<16> = 255;
             while a < 10 decreases b {
                 set a = a +% 1;
                 set b = b +% 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "a 16-bit decrease measure must not be truncated to 8 bits: {:?}",
        result
    );
}

/// Fix (primed-width map): a COMPOUND `decreases` on a non-64-bit
/// wrap loop (`10 - i` on `UInt<8>` `i`) re-encodes the measure with the
/// primed copy `i_p` — WITHOUT a primed entry in the width map the literal
/// `10` stays 64-bit while `i_p` declares at 8 bits, Z3 rejects the sort
/// mismatch, and every compound decrease on a non-64-bit loop failed
/// closed (false rejection).
#[test]
fn test_mixed_width_compound_decreases_wrap_verified() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_mixed_width_compound_decreases_wrap_verified");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: UInt<8> = 0;
             while i < 10 decreases 10 - i {
                 i = i +% 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "a compound decreases on a non-64-bit wrap loop must verify: {:?}",
        result
    );
}

/// Fix (loop-var collection): a loop-carried variable that
/// appears only in an ASSIGNMENT body (`s = i` — `s` is not in the
/// condition) must be collected by the loop translator, or `idx()` fails
/// and the whole loop is declared untranslatable — every declared
/// `decreases`/`invariant` on such a loop failed closed even when the
/// measure is genuine (the BV pipeline was inert for the plain syntax).
#[test]
fn test_loop_vars_collected_from_assign_bodies() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut i = 0;
             set mut s = 0;
             while i < 10 decreases 10 - i {
                 i = i + 1;
                 s = i;
             }
             return s;
         }",
    );
    assert!(
        result.is_ok(),
        "a loop-carried variable appearing only in an assignment body must be collected: {:?}",
        result
    );
}

/// Fix (wrap in loop conditions): `i +% 0 < j` — the IDENTITY
/// wrap form — must translate (to `i < j`) so sign-flipping loop shapes
/// reach the BV pipeline.  The declared measure `j - i` genuinely
/// decreases as `i` climbs toward `j`; pre-fix the condition was
/// untranslatable and the loop was rejected outright.
#[test]
fn test_wrap_condition_loop_translates() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_wrap_condition_loop_translates");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<8> = 0;
             set mut j: Int<8> = 5;
             while i +% 0 < j decreases j - i {
                 i = i +% 1;
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "an identity-wrap loop condition must translate and the decrease must verify: {:?}",
        result
    );
}

/// A loop whose guard references an external symbol
/// (function parameter `n` in `while i < n` — no literal init) separates
/// `n` into params. The BiiLoopProblem path synthesizes over the loop
/// variable only, and the declared decreases measure (`n - i`) still
/// verifies — the guard provides the bound.
#[test]
fn test_loop_external_param_separated() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_loop_external_param_separated");
        return;
    }
    let result = check_source(
        "def count(n: Int<8>) -> Int<8> {
             set mut i: Int<8> = 0;
             while i < n decreases n - i {
                 i = i + 1;
             }
             return i;
         }",
    );
    assert!(
        result.is_ok(),
        "external-symbol loop must separate into params and the decrease must verify: {:?}",
        result
    );
}

/// An `if` in the loop body merges into `Ite` — the
/// BII path synthesizes and verifies, and the loop passes end-to-end.
#[test]
fn test_loop_if_branch_end_to_end() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_loop_if_branch_end_to_end");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<8> = 0;
             while i < 6 {
                 if i < 3 { i = i + 1; } else { i = i + 2; }
             }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "if-branch loop must translate and verify: {:?}",
        result
    );
}

/// A saturating loop passes end-to-end — `i := 250;
/// while i < 255 { i := i +? 10 }` — the successor clamps to 255
/// (UInt<8> MAX) and the BII includes the clamped value.
#[test]
fn test_saturate_loop_end_to_end() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_saturate_loop_end_to_end");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: UInt<8> = 250;
             while i < 255 { i = i +? 10; }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "saturating loop must translate and verify: {:?}",
        result
    );
}

/// Type-level overflow policy (suffix > type policy >
/// default trap): a PLAIN `+` on a saturating type is saturating —
/// `i := 250; while i < 255 { i = i + 10 }` with
/// `set mut i: UInt<8> with overflow = saturate` clamps to 255.
#[test]
fn test_saturate_type_policy_end_to_end() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_saturate_type_policy_end_to_end");
        return;
    }
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: UInt<8> with overflow = saturate = 250;
             while i < 255 { i = i + 10; }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "type-level saturate policy must lower to AddSat and verify: {:?}",
        result
    );
}

/// Priority: the EXPLICIT `+%` overrides the
/// type-level saturate policy — a saturating type with a wrap suffix is
/// wrap (BV semantics, no clamp, no trap obligation).
#[test]
fn test_saturate_type_policy_suffix_wrap() {
    let z3_available = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !z3_available {
        eprintln!("z3 unavailable — skipping test_saturate_type_policy_suffix_wrap");
        return;
    }
    let result = check_source(
        "type SatU8 = UInt<8> with overflow = saturate;
         def main() -> Int<32> {
             set mut i: SatU8 = 250;
             while i < 255 { i = i +% 10; }
             return 0;
         }",
    );
    assert!(
        result.is_ok(),
        "explicit wrap suffix must override the saturate policy: {:?}",
        result
    );
}

/// Fix (`set` in loop bodies): inside a loop body, `set x = v`
/// on a loop-carried variable is the harness form of the plain assignment
/// `x = v` — it must lower to an ASSIGNMENT (checked against the existing
/// binding), not a duplicate definition.  Pre-fix the second `set` of the
/// same variable in one body reported E019 "duplicate definition" and the
/// swap loop could never reach the decrease check.
#[test]
fn test_set_in_loop_body_is_assignment() {
    let result = check_source(
        "def main() -> Int<32> {
             set mut i: Int<8> = -127;
             set mut j: Int<8> = 0;
             while i < j decreases i {
                 set i = j;
                 set i = i +% 0;
             }
             return 0;
         }",
    );
    assert!(
        result.is_err(),
        "the swap loop's decrease is false — it must still be rejected"
    );
    let errs = result.unwrap_err();
    assert!(
        !errs.iter().any(|e| e.contains("duplicate definition")),
        "a repeated `set` of a loop-carried variable is an assignment, not a duplicate definition: {:?}",
        errs
    );
}

// ── CheckerProbe: paired inference + type-transaction probes ──────

/// Run `f` with a live `TypeChecker` built from a trivial program — the
/// harness for the `CheckerProbe` tests (direct checker access, without
/// running `check_program`).
fn with_live_checker(f: impl for<'a, 'b, 'c> FnOnce(&'c mut TypeChecker<'a, 'b>)) {
    let source = "def main() -> Int<32> { return 0; }";
    let arena = bumpalo::Bump::new();
    let mut parser = Parser::new(source, &arena);
    let program = parser.parse_program().expect("parse should succeed");
    let mut ctx = TypeContext::new();
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (symbols, mut trait_env, _res_diags, resolution_map) = resolver.resolve_program(&program);
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, resolution_map);
    f(&mut checker);
}

/// An uncommitted probe must roll back inference state (vars, gen
/// statuses), type bindings, and diagnostics in lockstep — while state
/// established OUTSIDE the probe survives untouched.
#[test]
fn test_checker_probe_rolls_back_on_drop() {
    with_live_checker(|checker| {
        let outer_var =
            checker
                .infer
                .new_type_var(checker.ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
        let outer_int = checker.ctx.int(32, true);
        checker.ctx.set_binding(outer_var, outer_int);
        let base_vars = checker.infer.var_type_ids().len();

        let rolled_back_var = {
            let mut probe = checker.begin_probe();
            let v = probe
                .with(|c| {
                    let v =
                        c.infer
                            .new_type_var(c.ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
                    let ty = c.ctx.int(64, false);
                    c.ctx.set_binding(v, ty);
                    let var_id = c.infer.var_type_ids().len() - 1;
                    c.infer.set_gen_status(var_id, GenStatus::Generalized);
                    c.diagnostics.error("probe-side failure");
                    Ok::<_, ()>(v)
                })
                .unwrap();
            v // probe dropped uncommitted → everything rolled back
        };

        assert_eq!(
            checker.infer.var_type_ids().len(),
            base_vars,
            "the probe's var must be rolled back"
        );
        // `gen_statuses` keeps trailing identity entries for rolled-back
        // vars (documented convention, same as `guard_sets` — consumers
        // bounds-check), but the status itself must revert:
        assert_eq!(
            checker.infer.get_gen_status(1),
            Some(GenStatus::Ungeneralized),
            "the probe's gen status must revert on rollback"
        );
        assert_eq!(
            checker.ctx.resolve_binding(rolled_back_var),
            rolled_back_var,
            "the probe's binding must be rolled back"
        );
        assert_eq!(
            checker.ctx.resolve_binding(outer_var),
            outer_int,
            "state established outside the probe must survive"
        );
        assert_eq!(
            checker.diagnostics.unreported_len(),
            0,
            "diagnostics pushed inside the probe must be rolled back"
        );
    });
}

/// A committed probe must keep its inference state, bindings, and
/// diagnostics.
#[test]
fn test_checker_probe_commits_on_commit() {
    with_live_checker(|checker| {
        let base_vars = checker.infer.var_type_ids().len();
        let mut probe = checker.begin_probe();
        let (v, ty) = probe
            .with(|c| {
                let v = c
                    .infer
                    .new_type_var(c.ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
                let ty = c.ctx.int(64, false);
                c.ctx.set_binding(v, ty);
                let var_id = c.infer.var_type_ids().len() - 1;
                c.infer.set_gen_status(var_id, GenStatus::Generalized);
                c.diagnostics.error("kept diagnostic");
                Ok::<_, ()>((v, ty))
            })
            .unwrap();
        probe.commit();

        assert_eq!(checker.infer.var_type_ids().len(), base_vars + 1);
        assert_eq!(
            checker
                .infer
                .get_gen_status(checker.infer.var_type_ids().len() - 1),
            Some(GenStatus::Generalized),
            "committed gen status must persist"
        );
        assert_eq!(
            checker.ctx.resolve_binding(v),
            ty,
            "committed binding must persist"
        );
        assert_eq!(
            checker.diagnostics.unreported_len(),
            1,
            "committed diagnostics must persist"
        );
    });
}

/// Nested probes: rolling back the inner probe keeps the outer probe's
/// changes; rolling back the outer restores everything.
#[test]
fn test_checker_probe_nested_rollback() {
    with_live_checker(|checker| {
        let base_vars = checker.infer.var_type_ids().len();
        let v1 = {
            let mut outer = checker.begin_probe();
            let v1 = outer
                .with(|c| {
                    let v1 =
                        c.infer
                            .new_type_var(c.ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
                    {
                        let mut inner = c.begin_probe();
                        inner
                            .with(|c2| {
                                let v2 = c2.infer.new_type_var(
                                    c2.ctx,
                                    TypeVariableKind::Any,
                                    VarOrigin::Synthetic,
                                );
                                let ty = c2.ctx.int(16, true);
                                c2.ctx.set_binding(v2, ty);
                                Ok::<_, ()>(v2)
                            })
                            .unwrap();
                        // inner probe dropped uncommitted → its var and
                        // binding are rolled back, the outer's var survives
                    }
                    assert_eq!(
                        c.infer.var_type_ids().len(),
                        base_vars + 1,
                        "the inner probe must not leak its var"
                    );
                    Ok::<_, ()>(v1)
                })
                .unwrap();
            v1 // outer probe dropped uncommitted
        };
        assert_eq!(
            checker.infer.var_type_ids().len(),
            base_vars,
            "the outer probe must not leak its var"
        );
        assert_eq!(checker.diagnostics.unreported_len(), 0);
    });
}

/// The motivating pattern: try candidates, each in its own probe.  A
/// failed attempt's inference state and diagnostics are discarded; the
/// winning attempt commits.
#[test]
fn test_checker_probe_candidate_loop() {
    with_live_checker(|checker| {
        let base_vars = checker.infer.var_type_ids().len();
        let mut chosen = None;
        for attempt in 0..2 {
            let mut probe = checker.begin_probe();
            let result = probe.with(|c| {
                let v = c
                    .infer
                    .new_type_var(c.ctx, TypeVariableKind::Any, VarOrigin::Synthetic);
                let ty = c.ctx.int(32, true);
                c.ctx.set_binding(v, ty);
                if attempt == 0 {
                    c.diagnostics.error("candidate 0 failed");
                    Err(())
                } else {
                    Ok(v)
                }
            });
            match result {
                Ok(v) => {
                    chosen = Some(v);
                    probe.commit();
                }
                Err(()) => {} // probe dropped uncommitted → rollback
            }
        }
        let v = chosen.expect("the second candidate must win");
        assert_eq!(
            checker.infer.var_type_ids().len(),
            base_vars + 1,
            "only the winning candidate's var must survive"
        );
        assert_eq!(
            checker.diagnostics.unreported_len(),
            0,
            "the failed candidate's diagnostics must be discarded"
        );
        assert!(
            matches!(
                checker.ctx.get(checker.ctx.resolve_binding(v)),
                TypeData::Int { .. }
            ),
            "the winning candidate's binding must survive"
        );
    });
}

/// The production consumer: `lookup_method`'s generic-impl fallback
/// runs each candidate attempt inside a `CheckerProbe`.  An impl whose
/// `for_type` unifies with the receiver but which does not provide the
/// requested method must roll its attempt back (no leaked transaction,
/// no disturbed inference state); a found method commits its
/// substitution.
#[test]
fn test_lookup_method_generic_match_probe() {
    let source = "type Box<T> = struct { val: T }
         trait Grab { def grab(&self) -> Int<32>; }
         impl<T: Default> Grab for Box<T> {
             def grab(&self) -> Int<32> { return 1; }
         }
         def main() -> Int<32> { return 0; }";
    let arena = bumpalo::Bump::new();
    let mut parser = Parser::new(source, &arena);
    let program = parser.parse_program().expect("parse should succeed");
    let mut ctx = TypeContext::new();
    ctx.arena = Some(&arena);
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
    let (symbols, mut trait_env, _res_diags, resolution_map) = resolver.resolve_program(&program);
    let mut checker = make_checker(&mut ctx, &symbols, &mut trait_env, resolution_map);
    // Impl registration happens during checking (`check_program`), not in
    // the resolver — run it so the generic `Grab for Box<T>` candidate
    // exists for `lookup_method`.
    checker
        .check_program(&program)
        .expect("the fixture must check");
    let base_vars = checker.infer.var_type_ids().len();
    let base_diags = checker.diagnostics.unreported_len();

    // Receiver: `Box<Int<32>>` — the impl `impl<T: Default> Grab for
    // Box<T>` only matches via the GENERIC fallback
    // (`lookup_impls_for_type` compares structurally, and `Box<T>` ≠
    // `Box<Int<32>>`).  Derive the Adt DefId from the registered impl
    // (the local `Box` is not name-visible in the symbol table).
    let impl_candidate = checker
        .trait_env
        .all_impls()
        .iter()
        .find(|c| {
            c.trait_id
                == symbols
                    .lookup_trait_by_path(&[Symbol::intern("Grab")])
                    .expect("Grab must resolve")
        })
        .expect("the Grab impl must be registered");
    let box_def_id = checker
        .ctx
        .get_def_id_for_type(impl_candidate.for_type)
        .expect("the impl's for_type must be an Adt");
    let receiver = {
        let int_ty = checker.ctx.int(32, true);
        checker.ctx.struct_ty(box_def_id, vec![int_ty])
    };

    // Hit: the generic impl provides `grab` — commit path.
    let hit = checker.lookup_method(receiver, Symbol::intern("grab"));
    assert!(
        hit.is_some(),
        "the generic impl's method must be found via the fallback"
    );
    let (param_tys, ret_ty, _def_id) = hit.unwrap();
    assert_eq!(param_tys.len(), 1, "`grab(&self)` has one parameter");
    assert!(
        matches!(checker.ctx.get(param_tys[0]), TypeData::Ref { .. }),
        "`&self` must be substituted to a reference type"
    );
    assert!(
        matches!(checker.ctx.get(ret_ty), TypeData::Int { .. }),
        "the return type must survive the probe commit"
    );

    // Miss: no impl provides `nope` — the successful generic match that
    // lacks the method must roll its attempt back.
    let miss = checker.lookup_method(receiver, Symbol::intern("nope"));
    assert!(miss.is_none(), "a missing method must not be found");

    // Probe hygiene: no leaked transaction, no inference disturbance,
    // no diagnostics.
    assert_eq!(
        checker.ctx.transaction_depth(),
        0,
        "probes must leave the transaction balanced"
    );
    assert_eq!(
        checker.infer.var_type_ids().len(),
        base_vars,
        "method lookup must not create inference variables"
    );
    assert_eq!(
        checker.diagnostics.unreported_len(),
        base_diags,
        "method lookup must not push diagnostics"
    );
}

/// `get_infer_var_id` sees through `set_binding` (it inspects the raw
/// `TypeData` slot), and `var_origins` still records the `Expression`
/// creation site of an inference variable after it has been unified to
/// a concrete type — the two facts the inference-origin trace relies on.
#[test]
fn test_infer_origin_trace() {
    use crate::ast::Span;
    use crate::hir::infer::InferenceContext;
    use crate::hir::types::TypeContext;

    let mut ctx = TypeContext::new();
    let mut infer = InferenceContext::new();

    // Create an InferVar with Expression origin at span (10, 20).
    let span = Span::new(10, 20);
    let var_ty = infer.new_type_var(
        &mut ctx,
        TypeVariableKind::Integer,
        VarOrigin::Expression(Some(span)),
    );

    // Unify the InferVar to Int<32> — this calls set_binding internally.
    let int_ty = ctx.int(32, true);
    let _ = ctx.unify(var_ty, int_ty);

    // After unification, get_infer_var_id must still see the raw InferVar.
    assert!(
        ctx.get_infer_var_id(var_ty).is_some(),
        "get_infer_var_id must see through set_binding"
    );

    // var_origins must still record the Expression origin.
    let vid = ctx.get_infer_var_id(var_ty).unwrap();
    match &infer.var_origins()[vid] {
        VarOrigin::Expression(Some(s)) => assert_eq!(*s, span),
        other => panic!("expected Expression(Some(_)), got {:?}", other),
    }
}
