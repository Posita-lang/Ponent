//! Span-insensitive, nominally-resolved structural equality for AST types.
//!
//! Shared by the resolver (E065 contradiction detection) and the checker
//! (or-pattern equality). Residing in a common module keeps the HIR
//! checker from reaching back into the resolver (layering), giving the
//! comparison family a single authoritative home.

use crate::hir::octagon::Dbm;
use crate::hir::symbol::*;
use crate::hir::types::DefId;
use crate::symbol::Symbol;

/// Structural equality of AST types IGNORING source spans: two `Int<32>`
/// written at different positions are the same type. Used by the E065
/// contradiction check so identical RHS constraints are not flagged merely
/// because their spans differ.
pub(crate) fn type_eq_ignoring_spans<'input>(
    a: &crate::ast::Type,
    b: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> bool {
    type_eq_ignoring_spans_renamed(a, b, exists_params, symbols, &mut Vec::new())
}

/// The recursive core of `type_eq_ignoring_spans`, threading an
/// alpha-renaming map for the `exists` BINDERS: entering `exists X. T`,
/// `X` is bound to the other side's binder name while the bases are
/// compared, so `exists X. Expr<X>` and `exists Y. Expr<Y>` are the same
/// type — the name of a bound variable is irrelevant to structural
/// equality (alpha-conversion). `rename` holds the (a-binder, b-binder)
/// pairs; the Path arm compares a binder reference `X` in `a` as if it
/// were the mapped name.
fn type_eq_ignoring_spans_renamed<'input>(
    a: &crate::ast::Type,
    b: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
    rename: &mut Vec<(Symbol, Symbol)>,
) -> bool {
    match (a, b) {
        (crate::ast::Type::Path(p1, _), crate::ast::Type::Path(p2, _)) => {
            // Nominal equality: the RESOLVED constructor identity — `Int`
            // and `core::Int` are the same type, and two aliases to the
            // same type compare equal. Opaque exists witnesses (or an
            // unresolvable path) fall back to the syntactic comparison
            // (modulo the active alpha-renaming map — a binder reference
            // in `a`'s base compares equal to `b`'s mapped binder name).
            match (
                path_ctor_key(p1, exists_params, symbols),
                path_ctor_key(p2, exists_params, symbols),
            ) {
                (Some(k1), Some(k2)) => k1 == k2,
                _ => renamed_paths_eq(p1, p2, rename),
            }
        }
        (crate::ast::Type::Generic(b1, a1, _), crate::ast::Type::Generic(b2, a2, _)) => {
            type_eq_ignoring_spans_renamed(b1, b2, exists_params, symbols, rename)
                && generic_args_eq_ignoring_spans_renamed(a1, a2, exists_params, symbols, rename)
        }
        (
            crate::ast::Type::Reference {
                inner: i1,
                mutable: m1,
                ..
            },
            crate::ast::Type::Reference {
                inner: i2,
                mutable: m2,
                ..
            },
        ) => m1 == m2 && type_eq_ignoring_spans_renamed(i1, i2, exists_params, symbols, rename),
        (crate::ast::Type::Pointer(t1, _), crate::ast::Type::Pointer(t2, _)) => {
            type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
        }
        (crate::ast::Type::Slice(t1, _), crate::ast::Type::Slice(t2, _)) => {
            type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
        }
        (crate::ast::Type::Array(t1, e1, _), crate::ast::Type::Array(t2, e2, _)) => {
            type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
                && const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Tuple(e1, _), crate::ast::Type::Tuple(e2, _)) => {
            e1.len() == e2.len()
                && e1.iter().zip(e2).all(|(x, y)| {
                    type_eq_ignoring_spans_renamed(x, y, exists_params, symbols, rename)
                })
        }
        (
            crate::ast::Type::Function {
                params: p1,
                ret: r1,
                ..
            },
            crate::ast::Type::Function {
                params: p2,
                ret: r2,
                ..
            },
        ) => {
            p1.len() == p2.len()
                && p1.iter().zip(p2).all(|(x, y)| {
                    type_eq_ignoring_spans_renamed(x, y, exists_params, symbols, rename)
                })
                && type_eq_ignoring_spans_renamed(r1, r2, exists_params, symbols, rename)
        }
        (
            crate::ast::Type::Projection {
                impl_type: i1,
                trait_path: t1,
                assoc_name: n1,
                ..
            },
            crate::ast::Type::Projection {
                impl_type: i2,
                trait_path: t2,
                assoc_name: n2,
                ..
            },
        ) => {
            type_eq_ignoring_spans_renamed(i1, i2, exists_params, symbols, rename)
                && type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
                && n1 == n2
        }
        (crate::ast::Type::DynTrait(es1, _), crate::ast::Type::DynTrait(es2, _)) => {
            es1.len() == es2.len()
                && es1.iter().zip(es2).all(|(x, y)| {
                    type_eq_ignoring_spans_renamed(x, y, exists_params, symbols, rename)
                })
        }
        // The `invariant` clause (`exists n: T invariant P(n)`) IS part of
        // the type's identity — the set of valid instances. Two exists
        // types are equal only if their bases are alpha-equivalent AND
        // their invariants are alpha-equivalent (`exists X. Int invariant
        // X > 0` and `exists Y. Int invariant Y < 0` are DIFFERENT types).
        (
            crate::ast::Type::Exists {
                name: n1,
                base: b1,
                invariant: i1,
                ..
            },
            crate::ast::Type::Exists {
                name: n2,
                base: b2,
                invariant: i2,
                ..
            },
        ) => {
            let is_int = type_is_integer(b1);
            if n1 == n2 {
                // Register the identity mapping: the innermost binder with
                // this name must resolve to ITSELF (the `rev()` lookups pick
                // the innermost entry), shadowing any stale outer `(X, _)`
                // mapping — otherwise `exists X. exists X. exists X. Expr<X>`
                // vs `exists Y. exists Z. exists X. Expr<Z>` would compare
                // equal (a's base X resolves through the middle `(X, Z)`
                // entry) when they are NOT alpha-equivalent.
                rename.push((*n1, *n2));
                let eq = type_eq_ignoring_spans_renamed(b1, b2, exists_params, symbols, rename)
                    && expr_eq_ignoring_spans_renamed_typed(i1, i2, rename, is_int);
                rename.pop();
                eq
            } else if !binder_free_in(*n2, b1)
                && !binder_free_in(*n1, b2)
                && !expr_free_in(*n2, i1)
                && !expr_free_in(*n1, i2)
            {
                // Alpha-renaming: `exists X. T invariant P(X)` and
                // `exists Y. T invariant P(Y)` are the SAME type — the
                // bound variable name is irrelevant to structural equality.
                // Bind the two binder names and compare the bases AND the
                // invariants through the map. The guard rejects the
                // capture case (one binder name FREE in the other's base or
                // invariant — then the types are genuinely different).
                rename.push((*n1, *n2));
                let eq = type_eq_ignoring_spans_renamed(b1, b2, exists_params, symbols, rename)
                    && expr_eq_ignoring_spans_renamed_typed(i1, i2, rename, is_int);
                rename.pop();
                eq
            } else {
                false
            }
        }
        (
            crate::ast::Type::WhereShorthand { base: b1, .. },
            crate::ast::Type::WhereShorthand { base: b2, .. },
        ) => type_eq_ignoring_spans_renamed(b1, b2, exists_params, symbols, rename),
        (crate::ast::Type::Literal(e1, _), crate::ast::Type::Literal(e2, _)) => {
            const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Never(_), crate::ast::Type::Never(_)) => true,
        (crate::ast::Type::Union(es1, _), crate::ast::Type::Union(es2, _)) => {
            es1.len() == es2.len()
                && es1.iter().zip(es2).all(|(x, y)| {
                    type_eq_ignoring_spans_renamed(x, y, exists_params, symbols, rename)
                })
        }
        (crate::ast::Type::Expr(e1, _), crate::ast::Type::Expr(e2, _)) => {
            const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Regex(s1, _), crate::ast::Type::Regex(s2, _)) => s1 == s2,
        (crate::ast::Type::Error(_), crate::ast::Type::Error(_)) => true,
        _ => false,
    }
}

/// Compare two paths with the alpha-renaming map applied to BOTH sides: a
/// binder reference `X` in `a`'s base (mapped to `Y`) compares equal to a
/// reference `Y` in `b`'s base — `exists X. X` and `exists Y. Y` have
/// identical structure up to the binder names.
fn renamed_paths_eq(p1: &[Symbol], p2: &[Symbol], rename: &[(Symbol, Symbol)]) -> bool {
    if p1 == p2 {
        return true;
    }
    if p1.len() != p2.len() {
        return false;
    }
    let rename_seg = |s: Symbol| {
        // The map is ordered outermost binder → innermost binder (pushed in
        // recursion order); a nested `exists X` shadows an outer `exists X`,
        // so the LAST (innermost) mapping wins. `.find` returns the
        // outermost mapping and mis-compares nested same-name binders — a
        // false positive (or negative) in GADT pending-eq / or-pattern
        // equality.
        rename
            .iter()
            .rev()
            .find(|(f, _)| *f == s)
            .map(|(_, t)| *t)
            .unwrap_or(s)
    };
    p1.iter()
        .zip(p2)
        .all(|(s1, s2)| rename_seg(*s1) == rename_seg(*s2))
}

/// Whether `name` occurs as a FREE single-segment path in `ty` — a binder
/// reference NOT shadowed by a nested `exists` with the same name. The
/// alpha-renaming guard: renaming `n1` to `n2` must not capture a free
/// `n2` in the other type's base (`exists X. Y` vs `exists Y. Y` are NOT
/// alpha-equivalent — the first `Y` is free, the second bound).
/// Const expressions (array sizes, generic const args) are NOT scanned:
/// their comparison is name-identity (`const_expr_eq_ignoring_spans`), so
/// a miss here fails closed (the pair compares unequal) — never a false
/// positive.
fn binder_free_in(name: Symbol, ty: &crate::ast::Type) -> bool {
    match ty {
        crate::ast::Type::Path(p, _) => p.len() == 1 && p[0] == name,
        crate::ast::Type::Exists {
            name: inner,
            base,
            invariant,
            ..
        } => {
            // The nested binder shadows the outer one.
            if *inner == name {
                false
            } else {
                // The invariant is part of the type's structure — a free
                // occurrence of `name` inside a NESTED exists invariant
                // would be captured by the outer renaming and must be
                // detected by the guard.
                binder_free_in(name, base) || expr_free_in(name, invariant)
            }
        }
        crate::ast::Type::Generic(b, args, _) => {
            binder_free_in(name, b)
                || args.iter().any(|a| match a {
                    crate::ast::GenericArg::Positional(t) | crate::ast::GenericArg::Named(_, t) => {
                        binder_free_in(name, t)
                    }
                    crate::ast::GenericArg::Const(_) => false,
                })
        }
        crate::ast::Type::Reference { inner, .. } => binder_free_in(name, inner),
        crate::ast::Type::Pointer(t, _) => binder_free_in(name, t),
        crate::ast::Type::Slice(t, _) => binder_free_in(name, t),
        crate::ast::Type::Array(t, _, _) => binder_free_in(name, t),
        crate::ast::Type::Tuple(es, _) => es.iter().any(|t| binder_free_in(name, t)),
        crate::ast::Type::Function { params, ret, .. } => {
            params.iter().any(|t| binder_free_in(name, t)) || binder_free_in(name, ret)
        }
        crate::ast::Type::Projection {
            impl_type,
            trait_path,
            ..
        } => binder_free_in(name, impl_type) || binder_free_in(name, trait_path),
        crate::ast::Type::DynTrait(es, _) | crate::ast::Type::Union(es, _) => {
            es.iter().any(|t| binder_free_in(name, t))
        }
        crate::ast::Type::WhereShorthand { base, .. } => binder_free_in(name, base),
        _ => false,
    }
}

/// Compare generic arguments ignoring spans (const args compare by shape).
fn generic_args_eq_ignoring_spans<'input>(
    a: &[crate::ast::GenericArg],
    b: &[crate::ast::GenericArg],
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> bool {
    generic_args_eq_ignoring_spans_renamed(a, b, exists_params, symbols, &mut Vec::new())
}

/// The recursive core of `generic_args_eq_ignoring_spans` — threads the
/// alpha-renaming map, and compares NAMED generic arguments by NAME, not
/// position: `MyType<n=Int, m=Bool>` and `MyType<m=Bool, n=Int>` are the
/// same type (named arguments are unordered — SYNTAX.md). The non-named
/// (positional/const) arguments keep their order.
fn generic_args_eq_ignoring_spans_renamed<'input>(
    a: &[crate::ast::GenericArg],
    b: &[crate::ast::GenericArg],
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
    rename: &mut Vec<(Symbol, Symbol)>,
) -> bool {
    fn args_eq<'input>(
        x: &crate::ast::GenericArg,
        y: &crate::ast::GenericArg,
        exists_params: &[Symbol],
        symbols: &SymbolTable<'input>,
        rename: &mut Vec<(Symbol, Symbol)>,
    ) -> bool {
        match (x, y) {
            (crate::ast::GenericArg::Positional(t1), crate::ast::GenericArg::Positional(t2)) => {
                type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
            }
            (crate::ast::GenericArg::Named(n1, t1), crate::ast::GenericArg::Named(n2, t2)) => {
                n1 == n2 && type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
            }
            (crate::ast::GenericArg::Const(ac1), crate::ast::GenericArg::Const(ac2)) => {
                const_expr_eq_ignoring_spans(&ac1.value, &ac2.value)
            }
            _ => false,
        }
    }
    // Purely positional (or const) arguments: compare by index — the order
    // is semantically meaningful for positional arguments.
    if !a
        .iter()
        .any(|x| matches!(x, crate::ast::GenericArg::Named(..)))
        && !b
            .iter()
            .any(|x| matches!(x, crate::ast::GenericArg::Named(..)))
    {
        return a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| args_eq(x, y, exists_params, symbols, rename));
    }
    // The non-named (positional/const) args keep their order; the named
    // args compare pairwise by name.
    let a_pos: Vec<&crate::ast::GenericArg> = a
        .iter()
        .filter(|x| !matches!(x, crate::ast::GenericArg::Named(..)))
        .collect();
    let b_pos: Vec<&crate::ast::GenericArg> = b
        .iter()
        .filter(|x| !matches!(x, crate::ast::GenericArg::Named(..)))
        .collect();
    if a_pos.len() != b_pos.len()
        || !a_pos
            .iter()
            .zip(&b_pos)
            .all(|(x, y)| args_eq(x, y, exists_params, symbols, rename))
    {
        return false;
    }
    let a_named: Vec<(&Symbol, &crate::ast::Type)> = a
        .iter()
        .filter_map(|x| match x {
            crate::ast::GenericArg::Named(n, t) => Some((n, t)),
            _ => None,
        })
        .collect();
    let b_named: Vec<(&Symbol, &crate::ast::Type)> = b
        .iter()
        .filter_map(|x| match x {
            crate::ast::GenericArg::Named(n, t) => Some((n, t)),
            _ => None,
        })
        .collect();
    a_named.len() == b_named.len()
        && a_named.iter().all(|(n1, t1)| {
            b_named
                .iter()
                .find(|(n2, _)| *n2 == *n1)
                .is_some_and(|(_, t2)| {
                    type_eq_ignoring_spans_renamed(t1, t2, exists_params, symbols, rename)
                })
        })
}

/// Compare const expressions (the common case: literal values) ignoring spans.
fn const_expr_eq_ignoring_spans(a: &crate::ast::Expr, b: &crate::ast::Expr) -> bool {
    match (a, b) {
        (crate::ast::Expr::Literal(l1, _), crate::ast::Expr::Literal(l2, _)) => l1 == l2,
        // Identically named identifiers (e.g. the same const-generic parameter
        // referenced twice) are structurally equal — defensive completeness
        // in case a future relaxation treats non-ground consts as ground.
        (crate::ast::Expr::Ident(s1, _), crate::ast::Expr::Ident(s2, _)) => s1 == s2,
        _ => false,
    }
}

/// Span-insensitive structural equality for general expressions — used to
/// compare `exists` INVARIANTS (`exists n: T invariant P(n)`): two refined
/// types are alpha-equivalent when their invariant bodies are structurally
/// identical after renaming one side's binder. The shapes invariants
/// actually use (comparisons, boolean/arithmetic operators, literals,
/// idents, comptime calls) compare exactly; anything else fails closed
/// (`false` — conservative: unproven equality is NOT equality).
pub(crate) fn expr_eq_ignoring_spans<'a, 'b: 'a, 'c, 'd: 'c>(
    a: &'a crate::ast::Expr<'b>,
    b: &'c crate::ast::Expr<'d>,
) -> bool {
    expr_eq_ignoring_spans_renamed_int(a, b, &[], false)
}

/// L1/L2-aware equality with the base type's discreteness: with
/// `is_int = true`, `X > 0` ≡ `X >= 1` (integer discreteness — the bounds
/// differ by one); floats/rationals are dense, so the two bounds genuinely
/// differ and compare exactly.
pub(crate) fn expr_eq_ignoring_spans_typed<'a, 'b: 'a, 'c, 'd: 'c>(
    a: &'a crate::ast::Expr<'b>,
    b: &'c crate::ast::Expr<'d>,
    is_int: bool,
) -> bool {
    expr_eq_ignoring_spans_renamed_int(a, b, &[], is_int)
}

/// The recursive core of `expr_eq_ignoring_spans`, applying the active
/// alpha-renaming map to identifier comparisons: a binder reference `X` in
/// `a`'s invariant (mapped to `Y`) compares equal to `Y` in `b`'s — so
/// `exists X. Int invariant X != 0` and `exists Y. Int invariant Y != 0`
/// have identical invariant structure up to the binder names.
pub(crate) fn expr_eq_ignoring_spans_renamed<'a, 'b: 'a, 'c, 'd: 'c>(
    a: &'a crate::ast::Expr<'b>,
    b: &'c crate::ast::Expr<'d>,
    rename: &[(Symbol, Symbol)],
) -> bool {
    expr_eq_ignoring_spans_renamed_int(a, b, rename, false)
}

/// `expr_eq_ignoring_spans_renamed` + the L1/L2 normalization (see
/// `expr_eq_ignoring_spans_typed` for the `is_int` semantics).
pub(crate) fn expr_eq_ignoring_spans_renamed_typed<'a, 'b: 'a, 'c, 'd: 'c>(
    a: &'a crate::ast::Expr<'b>,
    b: &'c crate::ast::Expr<'d>,
    rename: &[(Symbol, Symbol)],
    is_int: bool,
) -> bool {
    expr_eq_ignoring_spans_renamed_int(a, b, rename, is_int)
}

/// L1: strip arithmetic/boolean IDENTITY wrappers before comparing —
/// `0 + X`/`X + 0` → `X`, `X - 0` → `X`, `1 * X`/`X * 1` → `X`,
/// `X / 1` → `X`, `X and true`/`true and X` → `X`, `X or false`/
/// `false or X` → `X`, `not not X` → `X`. Pure syntactic normalization
/// (NO allocation — the sub-node reference is returned directly), applied
/// repeatedly.
fn peel_identity<'a, 'b: 'a>(e: &'a crate::ast::Expr<'b>) -> &'a crate::ast::Expr<'b> {
    let is_zero = |x: &crate::ast::Expr| {
        matches!(
            x,
            crate::ast::Expr::Literal(crate::ast::Literal::Int(crate::ast::IntLit::Small(0)), _)
        )
    };
    let is_one = |x: &crate::ast::Expr| {
        matches!(
            x,
            crate::ast::Expr::Literal(crate::ast::Literal::Int(crate::ast::IntLit::Small(1)), _)
        )
    };
    let is_true = |x: &crate::ast::Expr| {
        matches!(
            x,
            crate::ast::Expr::Literal(crate::ast::Literal::Bool(true), _)
        )
    };
    let is_false = |x: &crate::ast::Expr| {
        matches!(
            x,
            crate::ast::Expr::Literal(crate::ast::Literal::Bool(false), _)
        )
    };
    let mut cur = e;
    loop {
        let next = match cur {
            crate::ast::Expr::BinaryOp {
                op, left, right, ..
            } => match op {
                crate::ast::BinOp::Add if is_zero(left) => Some(right),
                crate::ast::BinOp::Add if is_zero(right) => Some(left),
                crate::ast::BinOp::Sub if is_zero(right) => Some(left),
                crate::ast::BinOp::Mul if is_one(left) => Some(right),
                crate::ast::BinOp::Mul if is_one(right) => Some(left),
                crate::ast::BinOp::Div if is_one(right) => Some(left),
                crate::ast::BinOp::And if is_true(left) => Some(right),
                crate::ast::BinOp::And if is_true(right) => Some(left),
                crate::ast::BinOp::Or if is_false(left) => Some(right),
                crate::ast::BinOp::Or if is_false(right) => Some(left),
                _ => None,
            },
            crate::ast::Expr::UnaryOp {
                op: crate::ast::UnaryOp::Not,
                expr,
                ..
            } => match expr {
                crate::ast::Expr::UnaryOp {
                    op: crate::ast::UnaryOp::Not,
                    expr: inner,
                    ..
                } => Some(inner),
                _ => None,
            },
            _ => None,
        };
        match next {
            Some(n) => cur = n,
            None => return cur,
        }
    }
}

/// L3: an affine (linear) form `coeff * core + offset` extracted from an
/// invariant expression built out of the binder and literal arithmetic —
/// `X + 1`, `2 * X - 3`, `-X`. Used to normalize the non-constant side of
/// an invariant comparison so the constant can be moved across (`X + 1 > 1`
/// ≡ `X > 0`, `2 * X > 5` ≡ `X > 2` on integer bases). Extraction fails
/// (returns `None`) on overflow or on shapes that are not affine in a
/// single variable — comparisons then fall back to structural equality
/// (fail-closed).
#[derive(Clone, Copy)]
struct Affine<'a, 'b> {
    core: &'a crate::ast::Expr<'b>,
    coeff: i128,
    offset: i128,
}

fn affine_of<'a, 'b>(e: &'a crate::ast::Expr<'b>) -> Option<Affine<'a, 'b>> {
    let lit = |x: &crate::ast::Expr| match x {
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => Some(v.clone()),
        _ => None,
    };
    match e {
        crate::ast::Expr::Ident(..) => Some(Affine {
            core: e,
            coeff: 1,
            offset: 0,
        }),
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => {
            match (op, lit(left), lit(right)) {
                // `(X + k)` / `(k + X)`: fold the constant into the offset.
                (crate::ast::BinOp::Add, Some(k), None) => {
                    let a = affine_of(right)?;
                    Some(Affine {
                        offset: a.offset.checked_add(k.to_i128()?)?,
                        ..a
                    })
                }
                (crate::ast::BinOp::Add, None, Some(k)) => {
                    let a = affine_of(left)?;
                    Some(Affine {
                        offset: a.offset.checked_add(k.to_i128()?)?,
                        ..a
                    })
                }
                // `X - k`: subtract from the offset.
                (crate::ast::BinOp::Sub, None, Some(k)) => {
                    let a = affine_of(left)?;
                    Some(Affine {
                        offset: a.offset.checked_sub(k.to_i128()?)?,
                        ..a
                    })
                }
                // `k * X` / `X * k` (k ≠ 0): scale coefficient AND offset.
                (crate::ast::BinOp::Mul, Some(k), None) if k != 0 => {
                    let a = affine_of(right)?;
                    Some(Affine {
                        coeff: a.coeff.checked_mul(k.to_i128()?)?,
                        offset: a.offset.checked_mul(k.to_i128()?)?,
                        ..a
                    })
                }
                (crate::ast::BinOp::Mul, None, Some(k)) if k != 0 => {
                    let a = affine_of(left)?;
                    Some(Affine {
                        coeff: a.coeff.checked_mul(k.to_i128()?)?,
                        offset: a.offset.checked_mul(k.to_i128()?)?,
                        ..a
                    })
                }
                _ => None,
            }
        }
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr,
            ..
        } => {
            let a = affine_of(expr)?;
            Some(Affine {
                coeff: a.coeff.checked_neg()?,
                offset: a.offset.checked_neg()?,
                ..a
            })
        }
        _ => None,
    }
}

/// Mirror a comparison operator for the affine sign flip (`-X > 0` ⟺
/// `X < 0`).
fn mirror(op: NormCmpOp) -> NormCmpOp {
    match op {
        NormCmpOp::Gt => NormCmpOp::Lt,
        NormCmpOp::Lt => NormCmpOp::Gt,
        NormCmpOp::Ge => NormCmpOp::Le,
        NormCmpOp::Le => NormCmpOp::Ge,
        o => o,
    }
}

/// L3.5: a multi-variable linear form `Σ aᵢXᵢ + c` (free variables with
/// coefficients plus a constant) — extracted from expressions that the
/// single-variable `Affine` cannot express (`X - Y + 1`). Two linear forms
/// are merged across a comparison to the canonical form `Σ aᵢXᵢ + c op 0`,
/// so `X - Y + 1 > 1` ≡ `X > Y`. Extraction fails (returns `None`) on
/// overflow or non-linear shapes (division, `X * Y`, calls) — comparisons
/// then fall back to structural equality (fail-closed).
#[derive(Clone)]
struct Linear {
    coeffs: smallvec::SmallVec<[(Symbol, i128); 2]>,
    constant: i128,
}

impl Linear {
    fn add(mut self, other: &Linear) -> Option<Linear> {
        for (s, k) in &other.coeffs {
            self.upsert(*s, *k)?;
        }
        self.constant = self.constant.checked_add(other.constant)?;
        Some(self)
    }
    fn sub(mut self, other: &Linear) -> Option<Linear> {
        for (s, k) in &other.coeffs {
            self.upsert(*s, k.checked_neg()?)?;
        }
        self.constant = self.constant.checked_sub(other.constant)?;
        Some(self)
    }
    fn scale(mut self, k: i128) -> Option<Linear> {
        for (_, c) in self.coeffs.iter_mut() {
            *c = c.checked_mul(k)?;
        }
        self.constant = self.constant.checked_mul(k)?;
        Some(self)
    }
    /// Add `d` to the coefficient of `s`; a coefficient that reaches zero
    /// is removed.
    fn upsert(&mut self, s: Symbol, d: i128) -> Option<()> {
        let mut i = 0;
        while i < self.coeffs.len() {
            if self.coeffs[i].0 == s {
                let c = self.coeffs[i].1.checked_add(d)?;
                if c == 0 {
                    self.coeffs.swap_remove(i);
                } else {
                    self.coeffs[i].1 = c;
                }
                return Some(());
            }
            i += 1;
        }
        if d != 0 {
            self.coeffs.push((s, d));
        }
        Some(())
    }
}

pub(crate) fn linear_of<'a, 'b>(e: &'a crate::ast::Expr<'b>) -> Option<Linear> {
    let lit = |x: &crate::ast::Expr| match x {
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => Some(v.clone()),
        _ => None,
    };
    match e {
        crate::ast::Expr::Ident(s, _) => Some(Linear {
            coeffs: smallvec::smallvec![(*s, 1)],
            constant: 0,
        }),
        crate::ast::Expr::Literal(crate::ast::Literal::Int(c), _) => Some(Linear {
            coeffs: smallvec::SmallVec::new(),
            constant: c.to_i128().unwrap_or(0),
        }),
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => match op {
            crate::ast::BinOp::Add => linear_of(left)?.add(&linear_of(right)?),
            crate::ast::BinOp::Sub => linear_of(left)?.sub(&linear_of(right)?),
            crate::ast::BinOp::Mul => match (lit(left), lit(right)) {
                // `k * X` / `X * k` (k ≠ 0): scale every coefficient.
                (Some(k), None) if k != 0 => linear_of(right)?.scale(k.to_i128()?),
                (None, Some(k)) if k != 0 => linear_of(left)?.scale(k.to_i128()?),
                // Non-linear (`X * Y`) or degenerate (`X * 0`).
                _ => None,
            },
            // Division, remainder, logic, nested comparisons: not affine.
            _ => None,
        },
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr,
            ..
        } => linear_of(expr)?.scale(-1),
        _ => None,
    }
}

/// L3.5: canonicalize a comparison whose BOTH sides are affine onto the
/// multi-variable linear form `Σ aᵢXᵢ + c op 0` — `X > Y`, `X - Y + 1 > 1`
/// (offset folding across the subtraction), `X >= Y` ≡ `X > Y - 1`
/// (discreteness). Returns `None` for single-variable or non-affine
/// comparisons (they fall back to the L2/L3 `Simple` path or structural
/// equality). Integer bases only.
fn linear_cmp<'a, 'b>(
    op: crate::ast::BinOp,
    left: &'a crate::ast::Expr<'b>,
    right: &'a crate::ast::Expr<'b>,
    is_int: bool,
) -> Option<NormCmp<'a, 'b>> {
    if !is_int {
        return None;
    }
    let lit = |x: &crate::ast::Expr| match x {
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => Some(v.clone()),
        _ => None,
    };
    // Mirror the constant to the right (`c < X` ⟺ `X > c`).
    let (op, left, right) = match (op, lit(left), lit(right)) {
        (crate::ast::BinOp::Lt, Some(_), None) => (crate::ast::BinOp::Gt, right, left),
        (crate::ast::BinOp::Le, Some(_), None) => (crate::ast::BinOp::Ge, right, left),
        (crate::ast::BinOp::Gt, Some(_), None) => (crate::ast::BinOp::Lt, right, left),
        (crate::ast::BinOp::Ge, Some(_), None) => (crate::ast::BinOp::Le, right, left),
        (crate::ast::BinOp::Eq, Some(_), None) => (crate::ast::BinOp::Eq, right, left),
        (crate::ast::BinOp::Neq, Some(_), None) => (crate::ast::BinOp::Neq, right, left),
        _ => (op, left, right),
    };
    // Merge: `lhs op rhs` ⟺ `(lhs - rhs) op 0`.
    let merged = linear_of(left)?.sub(&linear_of(right)?)?;
    // Single-variable shapes go through the L2/L3 `Simple` path instead.
    if merged.coeffs.len() < 2 {
        return None;
    }
    // Discreteness + canonical form `E op 0`:
    // `E >= 0` ⟺ `E + 1 > 0`; `E <= 0` ⟺ `E - 1 < 0`.
    let (op, c) = match (op, merged.constant) {
        (crate::ast::BinOp::Gt, c) => (NormCmpOp::Gt, c),
        (crate::ast::BinOp::Ge, c) => (NormCmpOp::Gt, c.checked_add(1)?),
        (crate::ast::BinOp::Lt, c) => (NormCmpOp::Lt, c),
        (crate::ast::BinOp::Le, c) => (NormCmpOp::Lt, c.checked_sub(1)?),
        (crate::ast::BinOp::Eq, c) => (NormCmpOp::Eq, c),
        (crate::ast::BinOp::Neq, c) => (NormCmpOp::Neq, c),
        _ => return None,
    };
    Some(NormCmp::Linear {
        op,
        coeffs: merged.coeffs,
        rhs: c,
    })
}

/// The widened fixpoint iteration
/// over a loop body's transfer function.
///
/// `step` abstracts one iteration of the loop body (`&Dbm → Dbm`). The
/// iteration joins the new state into the running one; once `widen_after`
/// iterations have passed it switches to `widen`, whose relaxation
/// guarantees that any ascending chain stabilizes (Miné) — so the loop
/// terminates even when the invariant is unbounded (e.g. `while true {
/// i := i + 1 }` converges to `i ≥ 0`). Convergence is detected by matrix
/// equality; `max_iter` bounds the iterations defensively.
///
/// The result is the loop-invariant CANDIDATE (a sound over-approximation).
/// Per the 2026-08-13 committee ruling it must NOT discharge obligations by
/// itself — it is only a `@hint` for the SMT solver.

/// The inference kernel of the `@hint(assertion)` injection pipeline —
/// the initial state and loop-body instructions drive the widened fixpoint,
/// and the converged matrix is turned into invariant candidate
/// expressions. The checker translates a HIR loop to `LoopInstr`s and
/// submits the result as a `@hint` (committee 2026-08-13: the candidates
/// never discharge obligations by themselves — the SMT solver stays the
/// authority). Guards (`TestLe`/`TestDiffLe`) are part of the body: the
/// fixpoint absorbs them, so the guard bound is NOT reported as an
/// invariant (it only holds inside the loop).

/// Flatten a conjunction `A and B and C` into its clauses (arbitrary
/// nesting); a non-conjunction is its own single clause.
pub(crate) fn and_clause_list<'a, 'b: 'a>(
    e: &'a crate::ast::Expr<'b>,
) -> Vec<&'a crate::ast::Expr<'b>> {
    match e {
        crate::ast::Expr::BinaryOp {
            op: crate::ast::BinOp::And,
            left,
            right,
            ..
        } => {
            let mut v = and_clause_list(left);
            v.extend(and_clause_list(right));
            v
        }
        _ => vec![e],
    }
}

/// Encode `Σ aᵢXᵢ ≤ d` (aᵢ ∈ {±1}, at most two variables) into the DBM —
/// the octagon row for one bound, following the paper's Figure 5
/// translation onto the 2N node numbering (`Xᵢ⁺ = 2i`, `Xᵢ⁻ = 2i+1`,
/// NO implicit zero node):
///
/// - `X ≤ d`    ⟺ `2X ≤ 2d`     (self-dual edge `(2i, 2i+1)`, constant `2d`)
/// - `−X ≤ d`   ⟺ `−2X ≤ 2d`    (self-dual edge `(2i+1, 2i)`, constant `2d`)
/// - `X − Y ≤ d` ⟺ `X⁺ − Y⁺ ≤ d` (edge `(2i, 2j)`)
/// - `X + Y ≤ d` ⟺ `X⁺ − Y⁻ ≤ d` (edge `(2i, 2j+1)`)
/// - `−X − Y ≤ d` ⟺ `X⁻ − Y⁺ ≤ d` (edge `(2i+1, 2j)`)
///
/// `set_mirrored` maintains the coherent mirror edge `(j̄, ī)` with the
/// same constant. `None` for non-octagon shapes (a coefficient with
/// |aᵢ| ≠ 1, three or more variables): the caller falls back to structural
/// equality (fail-closed).
pub(crate) fn dbm_leq(
    dbm: &mut Dbm,
    coeffs: &[(Symbol, i128)],
    d: i128,
    var_of: &std::collections::HashMap<Symbol, usize>,
) -> Option<()> {
    let i = |s: &Symbol| -> Option<usize> { var_of.get(s).copied() };
    match coeffs {
        [] => {
            // `0 ≤ d`: a tautology when `d ≥ 0`; otherwise the constant
            // contradiction is left to the closure's negative-cycle check.
            // A variable-less (size 0) DBM has no node to carry the
            // contradiction — fail closed (the caller keeps the
            // conservative path; a contradictory premise yields `None`
            // downstream anyway).
            if d < 0 {
                if dbm.size == 0 {
                    return None;
                }
                dbm.set(0, 0, d); // node₀ − node₀ ≤ d < 0 — any diagonal
            }
            Some(())
        }
        // X ≤ d ⟺ 2X ≤ 2d (paper Figure 5: `v ≤ c` rides `v⁺ − v⁻ ≤ 2c`).
        [(s, 1)] => {
            dbm.set_mirrored(2 * i(s)?, 2 * i(s)? + 1, 2 * d);
            Some(())
        }
        // −X ≤ d ⟺ X ≥ −d ⟺ −2X ≤ 2d (edge (X⁻, X⁺)).
        // The bound IS `2d` — the sign lives in the node selection, not
        // the constant. (A historical sign-flip bug here encoded `X ≥ d`
        // instead of `X ≥ −d`, weakening every negative-coefficient
        // premise by 2d — sound but catastrophically lossy for the
        // exact entailment path.)
        [(s, -1)] => {
            dbm.set_mirrored(2 * i(s)? + 1, 2 * i(s)?, 2 * d);
            Some(())
        }
        // X − Y ≤ d ⟺ X⁺ − Y⁺ ≤ d.
        [(s1, 1), (s2, -1)] => {
            dbm.set_mirrored(2 * i(s1)?, 2 * i(s2)?, d);
            Some(())
        }
        // X + Y ≤ d ⟺ X⁺ − Y⁻ ≤ d.
        [(s1, 1), (s2, 1)] => {
            dbm.set_mirrored(2 * i(s1)?, 2 * i(s2)? + 1, d);
            Some(())
        }
        // −X + Y ≤ d ⟺ Y − X ≤ d ⟺ Y⁺ − X⁺ ≤ d.
        [(s1, -1), (s2, 1)] => {
            dbm.set_mirrored(2 * i(s2)?, 2 * i(s1)?, d);
            Some(())
        }
        // −X − Y ≤ d ⟺ X⁻ − Y⁺ ≤ d (paper Figure 5: `−vᵢ − vⱼ ≤ c`).
        [(s1, -1), (s2, -1)] => {
            dbm.set_mirrored(2 * i(s1)? + 1, 2 * i(s2)?, d);
            Some(())
        }
        _ => None,
    }
}

/// Encode `Σ aᵢXᵢ + c op 0` (from a normalized clause) as ≤ bounds:
/// `E > 0` ⟺ `-E ≤ -1`, `E < 0` ⟺ `E ≤ -1`, `E == 0` ⟺ `E ≤ 0 ∧ -E ≤ 0`
/// (integers). `!=` is not convex and cannot be encoded — `None` (the
/// comparison falls back to structural equality).
pub(crate) fn dbm_encode(
    dbm: &mut Dbm,
    op: NormCmpOp,
    coeffs: &[(Symbol, i128)],
    c: i128,
    var_of: &std::collections::HashMap<Symbol, usize>,
) -> Option<()> {
    let neg = |cs: &[(Symbol, i128)]| -> Vec<(Symbol, i128)> {
        cs.iter().map(|(s, k)| (*s, -k)).collect()
    };
    match op {
        NormCmpOp::Gt => {
            let d = c.checked_sub(1)?;
            dbm_leq(dbm, &neg(coeffs), d, var_of)
        }
        NormCmpOp::Lt => {
            let d = c.checked_neg()?.checked_sub(1)?;
            dbm_leq(dbm, coeffs, d, var_of)
        }
        NormCmpOp::Eq => {
            let d = c.checked_neg()?;
            dbm_leq(dbm, coeffs, d, var_of)?;
            dbm_leq(dbm, &neg(coeffs), c, var_of)
        }
        // Non-convex (`!=`) and the closed forms (Ge/Le never survive the
        // integer discreteness normalization) are not encoded.
        NormCmpOp::Neq | NormCmpOp::Ge | NormCmpOp::Le => None,
    }
}

/// Encode the NEGATION of `Σ aᵢXᵢ + c op 0` — the entailment check
/// `a ⟹ b` tests `a ∧ ¬bᵢ` unsatisfiability for every conclusion clause
/// `bᵢ`. `E == 0` cannot be negated to a convex form (`E ≠ 0`), so the
/// check falls back (fail-closed); `E ≠ 0` negates to `E == 0` (two
/// bounds). Ge/Le never survive the discreteness normalization.
pub(crate) fn dbm_encode_negated(
    dbm: &mut Dbm,
    op: NormCmpOp,
    coeffs: &[(Symbol, i128)],
    c: i128,
    var_of: &std::collections::HashMap<Symbol, usize>,
) -> Option<()> {
    let neg = |cs: &[(Symbol, i128)]| -> Vec<(Symbol, i128)> {
        cs.iter().map(|(s, k)| (*s, -k)).collect()
    };
    match op {
        // ¬(E > 0) = E ≤ 0 ⟺ ΣaᵢXᵢ ≤ -c.
        NormCmpOp::Gt => {
            let d = c.checked_neg()?;
            dbm_leq(dbm, coeffs, d, var_of)
        }
        // ¬(E < 0) = E ≥ 0 ⟺ -E ≤ 0 ⟺ Σ(-aᵢ)Xᵢ ≤ c.
        NormCmpOp::Lt => dbm_leq(dbm, &neg(coeffs), c, var_of),
        // ¬(E == 0) = E ≠ 0 — non-convex, not encodable (fail-closed).
        NormCmpOp::Eq => None,
        // ¬(E ≠ 0) = E == 0 ⟺ E ≤ 0 ∧ -E ≤ 0.
        NormCmpOp::Neq => {
            let d = c.checked_neg()?;
            dbm_leq(dbm, coeffs, d, var_of)?;
            dbm_leq(dbm, &neg(coeffs), c, var_of)
        }
        NormCmpOp::Ge | NormCmpOp::Le => None,
    }
}

/// The entailment check `a ⟹ b` on INTEGER bases — EXACT within the
/// difference-constraint sub-language: for every conclusion clause `bᵢ`,
/// the premise together with `¬bᵢ` must be unsatisfiable (a negative
/// cycle in the closed DBM). Per the founder's bifurcation, an exact
/// decision like this is self-verifying (may discharge directly); a
/// `None` result (non-linear shapes, `E == 0` negated to `E ≠ 0`, dense
/// bases, a contradictory premise) falls back (fail-closed).
pub(crate) fn expr_entails_typed(
    a: &crate::ast::Expr,
    b: &crate::ast::Expr,
    is_int: bool,
) -> Option<bool> {
    if !is_int {
        return None;
    }
    let ca = and_clause_list(a);
    let cb = and_clause_list(b);
    // Collect the variables of both sides.
    let mut vars: Vec<Symbol> = Vec::new();
    for c in &ca {
        clause_vars(&norm_cmp(c, true)?, &[], &mut vars)?;
    }
    for c in &cb {
        clause_vars(&norm_cmp(c, true)?, &[], &mut vars)?;
    }
    vars.sort_unstable();
    vars.dedup();
    let var_of: std::collections::HashMap<Symbol, usize> =
        vars.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    // The premise `a`.
    let mut premise = Dbm::new(vars.len());
    for c in &ca {
        dbm_encode_clause(&mut premise, c, &[], &var_of)?;
    }
    // A contradictory premise entails everything — fail-closed: do not
    // judge (the caller keeps the conservative path).
    if !premise.close() {
        return None;
    }
    // `a ⟹ b` ⟺ for every conclusion clause `bᵢ`, `a ∧ ¬bᵢ` is unsat.
    for bi in &cb {
        let mut m = premise.clone();
        let n = norm_cmp(bi, true)?;
        let (op, coeffs, c) = match n {
            NormCmp::Simple { op, lhs, rhs } => {
                let lf = linear_of(lhs)?;
                (op, lf.coeffs.into_vec(), lf.constant.checked_sub(rhs)?)
            }
            NormCmp::Linear { op, coeffs, rhs } => (op, coeffs.into_vec(), rhs),
        };
        dbm_encode_negated(&mut m, op, &coeffs, c, &var_of)?;
        if m.close() {
            return Some(false); // a counter-model exists.
        }
    }
    Some(true)
}

/// Collect the (renamed) variables occurring in a normalized clause's
/// octagon constraints.
pub(crate) fn clause_vars<'a, 'b>(
    n: &NormCmp<'a, 'b>,
    rename: &[(Symbol, Symbol)],
    out: &mut Vec<Symbol>,
) -> Option<()> {
    let rename_seg = |s: Symbol| {
        // Innermost mapping wins (nested `exists` shadows the outer one) —
        // see `renamed_paths_eq`.
        rename
            .iter()
            .rev()
            .find(|(f, _)| *f == s)
            .map(|(_, t)| *t)
            .unwrap_or(s)
    };
    match n {
        NormCmp::Simple { lhs, .. } => {
            for (s, _) in &linear_of(lhs)?.coeffs {
                out.push(rename_seg(*s));
            }
        }
        NormCmp::Linear { coeffs, .. } => {
            for (s, _) in coeffs {
                out.push(rename_seg(*s));
            }
        }
    }
    Some(())
}

/// Encode one clause (`lhs op rhs` or `Σcoeffs + c op 0`) into the DBM.
/// The coefficient symbols are normalized through `rename` FIRST (exactly
/// like the variable collection in `clause_vars`) — otherwise the `var_of`
/// lookup misses alpha-renamed binders and the DBM would silently fall
/// back to structural equality.
pub(crate) fn dbm_encode_clause(
    dbm: &mut Dbm,
    clause: &crate::ast::Expr,
    rename: &[(Symbol, Symbol)],
    var_of: &std::collections::HashMap<Symbol, usize>,
) -> Option<()> {
    let rename_seg = |s: Symbol| {
        // Innermost mapping wins (nested `exists` shadows the outer one) —
        // see `renamed_paths_eq`.
        rename
            .iter()
            .rev()
            .find(|(f, _)| *f == s)
            .map(|(_, t)| *t)
            .unwrap_or(s)
    };
    let renamed = |cs: &[(Symbol, i128)]| -> Vec<(Symbol, i128)> {
        cs.iter().map(|(s, k)| (rename_seg(*s), *k)).collect()
    };
    match norm_cmp(clause, true)? {
        NormCmp::Simple { op, lhs, rhs } => {
            // `lhs op rhs` ⟺ E = linear_of(lhs) - rhs, then `E op 0`.
            let lf = linear_of(lhs)?;
            let c = lf.constant.checked_sub(rhs)?;
            let cs = renamed(&lf.coeffs);
            dbm_encode(dbm, op, &cs, c, var_of)
        }
        NormCmp::Linear { op, coeffs, rhs } => {
            let cs = renamed(&coeffs);
            dbm_encode(dbm, op, &cs, rhs, var_of)
        }
    }
}

/// Compare two clause sets through octagon (DBM) closure — redundancy
/// elimination (`X > 0 and X > 1` ≡ `X > 1`), transitivity
/// (`X - Y < 5 and Y - Z < 3` ≡ `X - Z < 7`), and order-insensitivity
/// (`A and B` ≡ `B and A`). Returns:
/// - `None` — a clause is not octagon-expressible (division, `X * Y`,
///   `!=`, non-unit coefficients) or a side is unsatisfiable: the caller
///   falls back to structural equality (fail-closed);
/// - `Some(eq)` — the closed matrices are/aren't equal (an EXACT check
///   within the difference-constraint sub-language).
fn dbm_clauses_eq(
    cl1: &[&crate::ast::Expr],
    cl2: &[&crate::ast::Expr],
    rename: &[(Symbol, Symbol)],
    is_int: bool,
) -> Option<bool> {
    if !is_int {
        return None;
    }
    // Renamed variable sets — different sets mean genuinely different
    // constraints.
    let mut vars1 = Vec::new();
    let mut vars2 = Vec::new();
    for c in cl1 {
        clause_vars(&norm_cmp(c, true)?, rename, &mut vars1)?;
    }
    for c in cl2 {
        clause_vars(&norm_cmp(c, true)?, rename, &mut vars2)?;
    }
    vars1.sort_unstable();
    vars1.dedup();
    vars2.sort_unstable();
    vars2.dedup();
    if vars1 != vars2 {
        return Some(false);
    }
    let var_of: std::collections::HashMap<Symbol, usize> =
        vars1.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    // Build and strongly close both sides.
    let mut dbm1 = Dbm::new(vars1.len());
    let mut dbm2 = Dbm::new(vars1.len());
    for c in cl1 {
        dbm_encode_clause(&mut dbm1, c, rename, &var_of)?;
    }
    for c in cl2 {
        dbm_encode_clause(&mut dbm2, c, rename, &var_of)?;
    }
    // An unsatisfiable side (negative cycle) is out of scope — fall back.
    if !dbm1.close() || !dbm2.close() {
        return None;
    }
    Some(dbm1.eq(&dbm2))
}

/// The canonical comparison form (L2): the constant is moved to the RIGHT
/// (`c < X` mirrors to `X > c`) and, for INTEGER bases, the bound is
/// normalized to the strict (open-interval) form — `X >= c` → `X > c-1`
/// and `X <= c` → `X < c+1`, so `X > 0` ≡ `X >= 1`. `None` for
/// non-comparison expressions or non-constant comparisons (compared
/// structurally instead).
#[derive(PartialEq, Clone, Copy)]
enum NormCmpOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Neq,
}

/// The canonical form of an invariant comparison.
enum NormCmp<'a, 'b> {
    /// Single-variable canonical form `X op c` (L2/L3): the constant on the
    /// right, integer bases on the open-interval form (`X > 0` ≡ `X >= 1`).
    Simple {
        op: NormCmpOp,
        lhs: &'a crate::ast::Expr<'b>,
        rhs: i128,
    },
    /// Multi-variable linear canonical form `Σ aᵢXᵢ + c op 0` (L3.5): both
    /// sides of the comparison are affine, merged onto the left — so
    /// `X - Y + 1 > 1` ≡ `X > Y` (same coefficient table + constant).
    Linear {
        op: NormCmpOp,
        coeffs: smallvec::SmallVec<[(Symbol, i128); 2]>,
        rhs: i128,
    },
}

pub(crate) fn norm_cmp<'a, 'b: 'a>(
    e: &'a crate::ast::Expr<'b>,
    is_int: bool,
) -> Option<NormCmp<'a, 'b>> {
    let (op, left, right) = match e {
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => (op, left, right),
        _ => return None,
    };
    let lit = |x: &crate::ast::Expr| match x {
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => Some(v.clone()),
        _ => None,
    };
    // L3.5: multi-variable linear — both sides affine (e.g. `X > Y`,
    // `X - Y + 1 > 1`), merged onto the canonical form `Σ aᵢXᵢ + c op 0`.
    // Single-variable shapes return `None` and fall through to L2/L3 below.
    if let Some(n) = linear_cmp(*op, left, right, is_int) {
        return Some(n);
    }
    // Move the constant to the right (`c < X` ⟺ `X > c`).
    let (op, lhs, rhs) = match (op, lit(left), lit(right)) {
        (crate::ast::BinOp::Lt, Some(c), None) => (NormCmpOp::Gt, *right, c.to_i128()?),
        (crate::ast::BinOp::Lt, None, Some(c)) => (NormCmpOp::Lt, *left, c.to_i128()?),
        (crate::ast::BinOp::Gt, Some(c), None) => (NormCmpOp::Lt, *right, c.to_i128()?),
        (crate::ast::BinOp::Gt, None, Some(c)) => (NormCmpOp::Gt, *left, c.to_i128()?),
        (crate::ast::BinOp::Le, Some(c), None) => (NormCmpOp::Ge, *right, c.to_i128()?),
        (crate::ast::BinOp::Le, None, Some(c)) => (NormCmpOp::Le, *left, c.to_i128()?),
        (crate::ast::BinOp::Ge, Some(c), None) => (NormCmpOp::Le, *right, c.to_i128()?),
        (crate::ast::BinOp::Ge, None, Some(c)) => (NormCmpOp::Ge, *left, c.to_i128()?),
        (crate::ast::BinOp::Eq, Some(c), None) => (NormCmpOp::Eq, *right, c.to_i128()?),
        (crate::ast::BinOp::Eq, None, Some(c)) => (NormCmpOp::Eq, *left, c.to_i128()?),
        (crate::ast::BinOp::Neq, Some(c), None) => (NormCmpOp::Neq, *right, c.to_i128()?),
        (crate::ast::BinOp::Neq, None, Some(c)) => (NormCmpOp::Neq, *left, c.to_i128()?),
        _ => return None,
    };
    // Integer discreteness: canonicalize the bound to the open interval —
    // `X >= c` ⟺ `X > c-1`, `X <= c` ⟺ `X < c+1` (floats/rationals keep
    // the exact bound). Checked arithmetic: `c = i128::MIN/MAX` cannot be
    // adjusted — the comparison falls back to structural equality
    // (fail-closed) instead of panicking or wrapping.
    let (op, rhs) = match (op, is_int) {
        (NormCmpOp::Ge, true) => (NormCmpOp::Gt, rhs.checked_sub(1)?),
        (NormCmpOp::Le, true) => (NormCmpOp::Lt, rhs.checked_add(1)?),
        (o, _) => (o, rhs),
    };
    // L3: affine normalization (integer bases only) — fold the constant
    // across `X + k` / `k * X` / `-X` so `X + 1 > 1` ≡ `X > 0` and
    // `2 * X > 5` ≡ `X > 2`. Floats/rationals keep the exact shape
    // (no affine rewrite — fall back to structural comparison).
    let (op, lhs, rhs) = match affine_of(lhs) {
        Some(a) if is_int => {
            // `k * X + o op c` ⟺ `k * X op (c - o)`.
            let c = rhs.checked_sub(a.offset)?;
            // Negative coefficients mirror the comparison: `-X > 0` ⟺ `X < 0`.
            let (op, k, c) = if a.coeff < 0 {
                (mirror(op), a.coeff.checked_neg()?, c.checked_neg()?)
            } else {
                (op, a.coeff, c)
            };
            // Integer discreteness first (`X >= c` ⟺ `X > c-1`), then scale
            // the bound: `k * X > c` ⟺ `X > floor(c / k)` (k > 0; a unit
            // coefficient is a no-op). Equality only normalizes the unit
            // coefficient — `2 * X == 2` vs `X == 1` stays structural
            // (fail-closed).
            let c = match (op, k) {
                (NormCmpOp::Ge, _) => c.checked_sub(1)?,
                (NormCmpOp::Le, _) => c.checked_add(1)?,
                (o, _) => c,
            };
            let (op, c) = match (op, k) {
                (NormCmpOp::Gt, 1) => (NormCmpOp::Gt, c),
                (NormCmpOp::Gt, k) => (NormCmpOp::Gt, c.div_euclid(k)),
                (NormCmpOp::Lt, 1) => (NormCmpOp::Lt, c),
                (NormCmpOp::Lt, k) => {
                    (NormCmpOp::Lt, c.checked_neg()?.div_euclid(k).checked_neg()?)
                }
                (NormCmpOp::Eq | NormCmpOp::Neq, 1) => (op, c),
                _ => return None,
            };
            (op, a.core, c)
        }
        _ => (op, lhs, rhs),
    };
    Some(NormCmp::Simple { op, lhs, rhs })
}

fn expr_eq_ignoring_spans_renamed_int<'a, 'b: 'a, 'c, 'd: 'c>(
    a: &'a crate::ast::Expr<'b>,
    b: &'c crate::ast::Expr<'d>,
    rename: &[(Symbol, Symbol)],
    is_int: bool,
) -> bool {
    let rename_seg = |s: Symbol| {
        // Innermost mapping wins (nested `exists` shadows the outer one) —
        // see `renamed_paths_eq`.
        rename
            .iter()
            .rev()
            .find(|(f, _)| *f == s)
            .map(|(_, t)| *t)
            .unwrap_or(s)
    };
    // L1: strip identity wrappers before comparing.
    let a = peel_identity(a);
    let b = peel_identity(b);
    match (a, b) {
        (crate::ast::Expr::Literal(l1, _), crate::ast::Expr::Literal(l2, _)) => l1 == l2,
        (crate::ast::Expr::Ident(s1, _), crate::ast::Expr::Ident(s2, _)) => {
            rename_seg(*s1) == rename_seg(*s2)
        }
        (crate::ast::Expr::BinaryOp { .. }, crate::ast::Expr::BinaryOp { .. }) => {
            // L2/L3/L3.5: canonical comparison form (`X > 0` ≡ `X >= 1`,
            // `X - Y + 1 > 1` ≡ `X > Y` on integer bases).
            match (norm_cmp(a, is_int), norm_cmp(b, is_int)) {
                (
                    Some(NormCmp::Simple {
                        op: o1,
                        lhs: l1,
                        rhs: r1,
                    }),
                    Some(NormCmp::Simple {
                        op: o2,
                        lhs: l2,
                        rhs: r2,
                    }),
                ) => {
                    return o1 == o2
                        && r1 == r2
                        && expr_eq_ignoring_spans_renamed_int(l1, l2, rename, is_int);
                }
                (
                    Some(NormCmp::Linear {
                        op: o1,
                        coeffs: c1,
                        rhs: r1,
                    }),
                    Some(NormCmp::Linear {
                        op: o2,
                        coeffs: c2,
                        rhs: r2,
                    }),
                ) => {
                    // Same operator, same constant, and the same coefficient
                    // table up to alpha-renaming (order-insensitive).
                    return o1 == o2
                        && r1 == r2
                        && c1.len() == c2.len()
                        && c1.iter().all(|(s1, k1)| {
                            c2.iter().any(|(s2, k2)| rename_seg(*s1) == *s2 && k1 == k2)
                        });
                }
                // Simple vs Linear (one side single-variable, the other
                // multi-variable) or non-comparison: structural below.
                _ => {}
            }
            // L3.5+: conjunction — both sides flatten to clause sets
            // compared through octagon (DBM) closure: redundancy
            // elimination (`X > 0 and X > 1` ≡ `X > 1`), transitivity
            // (`X - Y < 5 and Y - Z < 3` ≡ `X - Z < 7`), and
            // order-insensitivity (`A and B` ≡ `B and A`). Only when at
            // least one side is a real conjunction — single clauses are
            // already handled by the canonical forms above. A non-octagon
            // clause falls through to structural equality (fail-closed).
            let ca = and_clause_list(a);
            let cb = and_clause_list(b);
            if ca.len() > 1 || cb.len() > 1 {
                if let Some(eq) = dbm_clauses_eq(&ca, &cb, rename, is_int) {
                    return eq;
                }
            }
            // Non-comparison binary ops: structural.
            if let (
                crate::ast::Expr::BinaryOp {
                    left: l1,
                    right: r1,
                    op: o1,
                    ..
                },
                crate::ast::Expr::BinaryOp {
                    left: l2,
                    right: r2,
                    op: o2,
                    ..
                },
            ) = (a, b)
            {
                o1 == o2
                    && expr_eq_ignoring_spans_renamed_int(l1, l2, rename, is_int)
                    && expr_eq_ignoring_spans_renamed_int(r1, r2, rename, is_int)
            } else {
                false
            }
        }
        (
            crate::ast::Expr::UnaryOp {
                op: o1, expr: e1, ..
            },
            crate::ast::Expr::UnaryOp {
                op: o2, expr: e2, ..
            },
        ) => o1 == o2 && expr_eq_ignoring_spans_renamed_int(e1, e2, rename, is_int),
        (
            crate::ast::Expr::Call {
                callee: c1,
                args: a1,
                comptime: t1,
                ..
            },
            crate::ast::Expr::Call {
                callee: c2,
                args: a2,
                comptime: t2,
                ..
            },
        ) => {
            t1 == t2
                && expr_eq_ignoring_spans_renamed_int(c1, c2, rename, is_int)
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2)
                    .all(|(x, y)| expr_eq_ignoring_spans_renamed_int(x, y, rename, is_int))
        }
        _ => false,
    }
}

/// Whether an AST type's base is an INTEGER (discrete ordered) type —
/// `Int<N>` / `UInt<N>` / `Byte` / `Char` / `USize`. The L2 invariant
/// canonicalization (`X > 0` ≡ `X >= 1`) applies ONLY to integer bases:
/// floats and rationals are dense, so the two bounds genuinely differ.
fn type_is_integer(ty: &crate::ast::Type) -> bool {
    let path = match ty {
        crate::ast::Type::Path(p, _) => p.as_ref(),
        crate::ast::Type::Generic(base, _, _) => return type_is_integer(base),
        _ => return false,
    };
    let last = path[path.len() - 1];
    last.eq_str("Int")
        || last.eq_str("UInt")
        || last.eq_str("Byte")
        || last.eq_str("Char")
        || last.eq_str("USize")
}

/// Whether `name` occurs as a bare identifier in `e` — used by the
/// alpha-renaming guard for `exists` INVARIANTS: renaming the binder must
/// not capture a free occurrence of the other binder in the invariant
/// (`exists X. Int invariant Y > 0` vs `exists Y. Int invariant Y > 0`
/// are NOT alpha-equivalent — the first `Y` is free).

/// Does `name` appear FREE in a pattern (as a binder that shadows the
/// outer scope)? Mirrors `ast::visit::bound_vars` — used by the
/// capture-avoidance of the fresh-binder rename in
/// `check_construction_invariant` (a pattern binder shadows the outer
/// name inside the branch body).
fn pattern_binds(name: Symbol, p: &crate::ast::Pattern) -> bool {
    match p {
        crate::ast::Pattern::Ident(s, _) => *s == name,
        crate::ast::Pattern::Tuple(ps, _) | crate::ast::Pattern::Or(ps, _) => {
            ps.iter().any(|sub| pattern_binds(name, sub))
        }
        crate::ast::Pattern::Slice(pre, mid, post, _) => {
            pre.iter().any(|sub| pattern_binds(name, sub))
                || mid.is_some_and(|m| pattern_binds(name, m))
                || post.iter().any(|sub| pattern_binds(name, sub))
        }
        crate::ast::Pattern::Struct { fields, .. } => {
            fields.iter().any(|(_, sub)| pattern_binds(name, sub))
        }
        crate::ast::Pattern::Enum { inner, .. } => {
            inner.is_some_and(|sub| pattern_binds(name, sub))
        }
        crate::ast::Pattern::Wildcard(_)
        | crate::ast::Pattern::Literal(..)
        | crate::ast::Pattern::Error(_) => false,
    }
}

/// Does `name` appear FREE in a statement (traversing its embedded
/// expressions)? Mirrors the capture-avoidance of
/// `ast::visit::replace_ident_in_stmt`: a name bound by the statement's
/// own pattern (`if let`/`while let`/`for`) shadows the outer name inside
/// the branch body. Statements without value-level expressions cannot
/// reference a fresh binder; unknown future variants fail closed.
fn stmt_free_in(name: Symbol, s: &crate::ast::Stmt) -> bool {
    match s {
        crate::ast::Stmt::VariableDef {
            value, else_branch, ..
        } => {
            value.as_ref().is_some_and(|v| expr_free_in(name, v))
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| eb.iter().any(|s| stmt_free_in(name, s)))
        }
        crate::ast::Stmt::Expression(e) => expr_free_in(name, e),
        crate::ast::Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            expr_free_in(name, cond)
                || then_branch.iter().any(|s| stmt_free_in(name, s))
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| eb.iter().any(|s| stmt_free_in(name, s)))
        }
        crate::ast::Stmt::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            expr_free_in(name, scrutinee)
                || (!pattern_binds(name, pattern)
                    && (then_branch.iter().any(|s| stmt_free_in(name, s))
                        || else_branch
                            .as_ref()
                            .is_some_and(|eb| eb.iter().any(|s| stmt_free_in(name, s)))))
        }
        crate::ast::Stmt::While {
            cond,
            body,
            invariant,
            decreases,
            ..
        } => {
            expr_free_in(name, cond)
                || body.iter().any(|s| stmt_free_in(name, s))
                || invariant.as_ref().is_some_and(|i| expr_free_in(name, i))
                || decreases.as_ref().is_some_and(|d| expr_free_in(name, d))
        }
        crate::ast::Stmt::WhileLet {
            pattern,
            scrutinee,
            body,
            invariant,
            decreases,
            ..
        } => {
            expr_free_in(name, scrutinee)
                || (!pattern_binds(name, pattern)
                    && (body.iter().any(|s| stmt_free_in(name, s))
                        || invariant.as_ref().is_some_and(|i| expr_free_in(name, i))
                        || decreases.as_ref().is_some_and(|d| expr_free_in(name, d))))
        }
        crate::ast::Stmt::For {
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            ..
        } => {
            expr_free_in(name, iterable)
                || (!pattern_binds(name, pattern)
                    && (body.iter().any(|s| stmt_free_in(name, s))
                        || invariant.as_ref().is_some_and(|i| expr_free_in(name, i))
                        || decreases.as_ref().is_some_and(|d| expr_free_in(name, d))))
        }
        crate::ast::Stmt::Loop { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        crate::ast::Stmt::Return { value, .. } => {
            value.as_ref().is_some_and(|v| expr_free_in(name, v))
        }
        crate::ast::Stmt::Assign { target, value, .. } => {
            expr_free_in(name, target) || expr_free_in(name, value)
        }
        crate::ast::Stmt::ComptimeBlock { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        crate::ast::Stmt::ScopeCleanup {
            when_condition,
            body,
            ..
        } => {
            when_condition
                .as_ref()
                .is_some_and(|c| expr_free_in(name, c))
                || body.iter().any(|s| stmt_free_in(name, s))
        }
        crate::ast::Stmt::Unsafe { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        crate::ast::Stmt::GhostVariableDef { inner, .. } => stmt_free_in(name, inner),
        crate::ast::Stmt::Isolate { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        // Statements without value-level expressions cannot reference a
        // fresh invariant binder; `Stmt` is #[non_exhaustive], so unknown
        // future variants fail closed.
        _ => false,
    }
}

/// Does `name` appear FREE in the expression `e`? Returns `true` if any
/// using occurrence of `name` is not bound by a local binder (closure
/// parameter, match/if-let pattern, quantifier binder, catch branch).
///
/// Enumerated shapes mirror `ast::visit::replace_ident_in_expr` exactly
/// (the same shadowing rules), so a `false` here means the rename in
/// `check_construction_invariant` is capture-free. `Path` components are
/// module/type names, not value identifiers — they never rename. Unknown
/// future variants fail closed (`true` = "possibly free" — can only block
/// a renaming, never allow a capture).
pub(crate) fn expr_free_in(name: Symbol, e: &crate::ast::Expr) -> bool {
    match e {
        crate::ast::Expr::Ident(s, _) => *s == name,
        crate::ast::Expr::Literal(..) => false,
        crate::ast::Expr::BinaryOp { left, right, .. } => {
            expr_free_in(name, left) || expr_free_in(name, right)
        }
        crate::ast::Expr::UnaryOp { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Call { callee, args, .. } => {
            expr_free_in(name, callee) || args.iter().any(|a| expr_free_in(name, a))
        }
        // Common container variants: traverse precisely (so legitimate
        // renamings over e.g. `if`-free invariants are not blocked).
        crate::ast::Expr::TypeAnnotated { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Index { base, index, .. } => {
            expr_free_in(name, base) || expr_free_in(name, index)
        }
        crate::ast::Expr::FieldAccess { base, .. } => expr_free_in(name, base),
        crate::ast::Expr::AttrAccess { base, .. } => expr_free_in(name, base),
        crate::ast::Expr::Cast { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Range { start, end, .. } => {
            start.is_some_and(|s| expr_free_in(name, s))
                || end.is_some_and(|e| expr_free_in(name, e))
        }
        crate::ast::Expr::StructLit { fields, .. } => {
            fields.iter().any(|(_, e)| expr_free_in(name, e))
        }
        crate::ast::Expr::EnumLit { payload, .. } => payload.is_some_and(|p| expr_free_in(name, p)),
        crate::ast::Expr::Try { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Move(e, _) => expr_free_in(name, e),
        // Multi-segment path (`Module::Type::item`): the components are
        // module/type/method names, never value bindings — the rename
        // never touches them.
        crate::ast::Expr::Path(..) => false,
        crate::ast::Expr::Tuple(es, _) => es.iter().any(|x| expr_free_in(name, x)),
        crate::ast::Expr::Array(es, _) => es.iter().any(|x| expr_free_in(name, x)),
        crate::ast::Expr::Closure {
            params,
            captures,
            body,
            ..
        } => {
            // A closure parameter (or an explicit capture of the outer
            // name) binds/references `name` — body occurrences of a
            // shadowed parameter are bound, not free; an explicit capture
            // IS a free reference to the outer variable.
            captures.iter().any(|c| c.name == name)
                || (!params.iter().any(|p| p.name == name)
                    && body.iter().any(|s| stmt_free_in(name, s)))
        }
        crate::ast::Expr::UnsafeBlock { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        crate::ast::Expr::Catch { expr, branches, .. } => {
            expr_free_in(name, expr)
                || branches.iter().any(|b| {
                    let shadowed =
                        pattern_binds(name, &b.pattern) || b.bind.is_some_and(|s| s == name);
                    !shadowed && b.body.iter().any(|s| stmt_free_in(name, s))
                })
        }
        crate::ast::Expr::LeaveWith { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Await { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            expr_free_in(name, cond)
                || then_branch.iter().any(|s| stmt_free_in(name, s))
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| eb.iter().any(|s| stmt_free_in(name, s)))
        }
        crate::ast::Expr::IfLet {
            pattern,
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            expr_free_in(name, scrutinee)
                || (!pattern_binds(name, pattern)
                    && (then_branch.iter().any(|s| stmt_free_in(name, s))
                        || else_branch
                            .as_ref()
                            .is_some_and(|eb| eb.iter().any(|s| stmt_free_in(name, s)))))
        }
        crate::ast::Expr::Match {
            scrutinee,
            arms,
            span: _,
        } => {
            expr_free_in(name, scrutinee)
                || arms.iter().any(|arm| {
                    // The guard is evaluated in the scope of the pattern
                    // bindings (like Rust) — a pattern binding `name`
                    // shadows it inside both the guard and the body.
                    let shadowed = pattern_binds(name, &arm.pattern);
                    (!shadowed
                        && (arm.guard.is_some_and(|g| expr_free_in(name, g))
                            || expr_free_in(name, &arm.body)))
                })
        }
        crate::ast::Expr::Block(stmts, _) => stmts.iter().any(|s| stmt_free_in(name, s)),
        crate::ast::Expr::PolyBox { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::PolyUnbox { expr, .. } => expr_free_in(name, expr),
        crate::ast::Expr::Quantified {
            binder,
            range,
            body,
            ..
        } => {
            // The quantifier binder shadows the name inside the body (the
            // bound occurrences belong to the binder, not the outer name);
            // the range is still evaluated in the outer scope.
            expr_free_in(name, range) || (*binder != name && expr_free_in(name, body))
        }
        crate::ast::Expr::Old(e, _) => expr_free_in(name, e),
        crate::ast::Expr::Task { body, .. } => body.iter().any(|s| stmt_free_in(name, s)),
        // Type-level reflections carry no value-level identifiers.
        crate::ast::Expr::TypeInfo(..) | crate::ast::Expr::LayoutOf(..) => false,
        crate::ast::Expr::CompileError(..) | crate::ast::Expr::Error(_) => false,
        // FAIL-CLOSED: any other variant (if/match/closure/quantified/block/
        // tuple/array/path/move/await/catch/...) is treated as "possibly
        // free". In a capture-avoidance guard `false` means "safe to
        // rename", so returning `true` for unknown shapes can only block a
        // renaming, never allow a capture.
        _ => true,
    }
}

/// A const expression (array size, const-generic argument) is "ground" — a
/// fixed value — only when it is a literal. An identifier (const parameter)
/// or a computation may depend on parameters, so a type containing it is not
/// provably concrete: two syntactically different such types could still
/// unify, and E065 must not fire on them (false positive > false negative).
pub(crate) fn const_expr_is_ground(e: &crate::ast::Expr) -> bool {
    matches!(e, crate::ast::Expr::Literal(..))
}

/// The nominal identity of a provably-concrete `when` RHS's top-level type
/// constructor: the resolved DefId for symbol-table types, or the primitive
/// name for the builtin primitives (so `Int` and `core::Int` compare equal —
/// the same type under a qualified path must not be flagged as two types).
/// `None` for forms without a single constructor (tuple, array, ...) or an
/// opaque `exists` witness — the caller must NOT assert a contradiction then.
#[derive(PartialEq)]
pub(crate) enum ConcreteCtor {
    Def(DefId),
    Primitive(Symbol),
}

/// Maximum alias-expansion depth for nominal constructor resolution. Cyclic
/// or pathologically deep aliases must fail closed (return None) rather than
/// recurse forever — the compiler must fail gracefully on invalid inputs.
/// 32 layers is far beyond any real alias chain (each expansion consumes a
/// distinct `type` name, so a chain this deep is already pathological);
/// the bound exists only to stop infinite recursion on cycles.
const MAX_ALIAS_DEPTH: usize = 32;

/// The nominal constructor identity of a PATH: the resolved DefId for
/// symbol-table types, or the primitive name for the builtin primitives (so
/// `Int` and `core::Int` compare equal). `None` for an opaque `exists`
/// witness, an alias cycle, or an unresolvable path.
fn path_ctor_key<'input>(
    path: &[Symbol],
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> Option<ConcreteCtor> {
    path_ctor_key_depth(path, exists_params, symbols, 0)
}

fn path_ctor_key_depth<'input>(
    path: &[Symbol],
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
    depth: usize,
) -> Option<ConcreteCtor> {
    if depth > MAX_ALIAS_DEPTH {
        return None;
    }
    // An opaque `exists` witness has no concrete constructor.
    if path.len() == 1 && exists_params.iter().any(|ep| ep == &path[0]) {
        return None;
    }
    // The path must resolve to a concrete (non-alias) binding.
    if let Some(def_id) = symbols.lookup_type_by_path(path) {
        if let Some(binding) = symbols.lookup_type_by_def_id(def_id) {
            if !matches!(binding.kind, TypeKind::Alias) {
                return Some(ConcreteCtor::Def(def_id));
            }
            // Type aliases are TRANSPARENT (SYNTAX.md): resolve the alias to
            // its underlying type's constructor — `MyInt` and `Int<32>` are
            // the same type, so `when T == MyInt and T == Int<32>` is NOT a
            // contradiction. Depth-limited so cyclic aliases fail closed.
            if let Some(alias_ast) = &binding.alias_ast {
                return concrete_ctor_key_depth(alias_ast, exists_params, symbols, depth + 1);
            }
            return None;
        }
    }
    // Unresolved path: the builtin primitives are not symbol-table bindings —
    // identify by the LAST segment name so the qualified `core::Int` and the
    // bare `Int` compare equal (the same type under a qualified path must
    // never be flagged as two different constructors).
    let last = path[path.len() - 1];
    if last.eq_str("Int")
        || last.eq_str("UInt")
        || last.eq_str("Float")
        || last.eq_str("Bool")
        || last.eq_str("Char")
        || last.eq_str("Byte")
        || last.eq_str("USize")
    {
        return Some(ConcreteCtor::Primitive(last));
    }
    None
}

/// The nominal constructor identity of a TYPE (the top-level constructor):
/// see `path_ctor_key`. `None` for tuples/arrays or an unresolvable path.
pub(crate) fn concrete_ctor_key<'input>(
    ct: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> Option<ConcreteCtor> {
    concrete_ctor_key_depth(ct, exists_params, symbols, 0)
}

fn concrete_ctor_key_depth<'input>(
    ct: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
    depth: usize,
) -> Option<ConcreteCtor> {
    if depth > MAX_ALIAS_DEPTH {
        return None;
    }
    let path = match ct {
        crate::ast::Type::Path(p, _) => p,
        crate::ast::Type::Generic(base, _, _) => {
            return concrete_ctor_key_depth(base, exists_params, symbols, depth + 1);
        }
        _ => return None,
    };
    path_ctor_key_depth(path, exists_params, symbols, depth)
}

/// Expand a top-level type-alias path (following alias chains) to the
/// underlying AST type — aliases are TRANSPARENT (SYNTAX.md), so `MyInt`
/// and its expansion `Int<32>` are the same type. `None` when the type is
/// not an alias, or an alias cycle / pathological depth (fail closed).
fn normalize_alias<'a, 'input>(
    ty: &'a crate::ast::Type<'input>,
    symbols: &'a SymbolTable<'input>,
) -> Option<&'a crate::ast::Type<'input>> {
    let mut current = ty;
    for _ in 0..MAX_ALIAS_DEPTH {
        let path = match current {
            crate::ast::Type::Path(p, _) => p,
            _ => return Some(current),
        };
        let def_id = symbols.lookup_type_by_path(path)?;
        let binding = symbols.lookup_type_by_def_id(def_id)?;
        if !matches!(binding.kind, TypeKind::Alias) {
            return None; // not an alias — nothing to expand
        }
        let Some(alias_ast) = binding.alias_ast.as_ref() else {
            return None;
        };
        current = alias_ast;
    }
    None // alias cycle / pathological depth — fail closed
}

/// Compare the generic arguments of two same-constructor types (positions,
/// named args, and const values); bare paths without args are equal.
pub(crate) fn type_args_eq_ignoring_spans<'input>(
    a: &crate::ast::Type<'input>,
    b: &crate::ast::Type<'input>,
    exists_params: &[Symbol],
    symbols: &SymbolTable<'input>,
) -> bool {
    match (a, b) {
        (crate::ast::Type::Generic(_, a1, _), crate::ast::Type::Generic(_, a2, _)) => {
            generic_args_eq_ignoring_spans(a1, a2, exists_params, symbols)
        }
        (crate::ast::Type::Path(..), crate::ast::Type::Path(..)) => true,
        // Alias transparency: `MyInt` (a Path that is an alias) and its
        // expanded underlying type (`Int<32>` — a Generic) are the SAME
        // type — normalize either side's top-level alias and retry.
        _ => match (normalize_alias(a, symbols), normalize_alias(b, symbols)) {
            (Some(na), Some(nb)) => type_eq_ignoring_spans(na, nb, exists_params, symbols),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::loop_infer::{
        LoopInstr, dbm_fixpoint, dbm_to_invariant_exprs, infer_loop_decreases_expr,
        infer_loop_invariant_exprs,
    };
    use crate::hir::types::{CrateId, DefId};

    fn path<'a>(arena: &'a bumpalo::Bump, seg: &str) -> &'a crate::ast::Type<'a> {
        arena.alloc(crate::ast::Type::Path(
            smallvec::smallvec![Symbol::intern(seg)],
            crate::ast::Span::new(0, 0),
        ))
    }

    fn generic<'a>(
        arena: &'a bumpalo::Bump,
        base: &'a crate::ast::Type<'a>,
        args: Vec<crate::ast::GenericArg<'a>>,
    ) -> &'a crate::ast::Type<'a> {
        arena.alloc(crate::ast::Type::Generic(
            base,
            args,
            crate::ast::Span::new(0, 0),
        ))
    }

    /// `exists <name>. <base>` with a dummy invariant.
    fn exists<'a>(
        arena: &'a bumpalo::Bump,
        name: &str,
        base: &'a crate::ast::Type<'a>,
    ) -> &'a crate::ast::Type<'a> {
        let inv = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("_dummy"),
            crate::ast::Span::new(0, 0),
        ));
        arena.alloc(crate::ast::Type::Exists {
            name: Symbol::intern(name),
            base,
            invariant: inv,
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `exists <name>. <base> invariant <inv>`.
    fn exists_inv<'a>(
        arena: &'a bumpalo::Bump,
        name: &str,
        base: &'a crate::ast::Type<'a>,
        inv: &'a crate::ast::Expr<'a>,
    ) -> &'a crate::ast::Type<'a> {
        arena.alloc(crate::ast::Type::Exists {
            name: Symbol::intern(name),
            base,
            invariant: inv,
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `<binder> > 0` — an invariant referencing the exists binder.
    fn gt_zero<'a>(arena: &'a bumpalo::Bump, binder: &str) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Ident(
                Symbol::intern(binder),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Gt,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `<binder> < 0` — an invariant referencing the exists binder.
    fn lt_zero<'a>(arena: &'a bumpalo::Bump, binder: &str) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Ident(
                Symbol::intern(binder),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Lt,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `<binder> >= 1` — an invariant referencing the exists binder.
    fn ge_one<'a>(arena: &'a bumpalo::Bump, binder: &str) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Ident(
                Symbol::intern(binder),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Ge,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(1)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `0 < <binder>` — the constant on the LEFT (the mirror case).
    fn zero_lt<'a>(arena: &'a bumpalo::Bump, binder: &str) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Lt,
            right: arena.alloc(crate::ast::Expr::Ident(
                Symbol::intern(binder),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `inner + c` — a literal-addition wrapper (L3 affine normalization).
    fn add_lit<'a>(
        arena: &'a bumpalo::Bump,
        inner: &'a crate::ast::Expr<'a>,
        c: i128,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: inner,
            op: crate::ast::BinOp::Add,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(c)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `inner - c` — a literal-subtraction wrapper (L3 affine).
    fn sub_lit<'a>(
        arena: &'a bumpalo::Bump,
        inner: &'a crate::ast::Expr<'a>,
        c: i128,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: inner,
            op: crate::ast::BinOp::Sub,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(c)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `c * inner` — a literal-multiplication wrapper (L3 affine).
    fn mul_lit<'a>(
        arena: &'a bumpalo::Bump,
        inner: &'a crate::ast::Expr<'a>,
        c: i128,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(c)),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Mul,
            right: inner,
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `-inner` — a negation wrapper (L3 affine).
    fn neg<'a>(
        arena: &'a bumpalo::Bump,
        inner: &'a crate::ast::Expr<'a>,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr: inner,
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `<inner> <op> c` — a comparison with a literal constant on the RIGHT.
    fn cmp_lit<'a>(
        arena: &'a bumpalo::Bump,
        op: crate::ast::BinOp,
        inner: &'a crate::ast::Expr<'a>,
        c: i128,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: inner,
            op,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(c)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `a and b` — a conjunction (L3.5+ DBM closure).
    fn and_expr<'a>(
        arena: &'a bumpalo::Bump,
        a: &'a crate::ast::Expr<'a>,
        b: &'a crate::ast::Expr<'a>,
    ) -> &'a crate::ast::Expr<'a> {
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: a,
            op: crate::ast::BinOp::And,
            right: b,
            span: crate::ast::Span::new(0, 0),
        })
    }

    /// `(0 + <binder>) > 0` — the L1 identity wrapper inside a comparison.
    fn zero_plus_gt_zero<'a>(arena: &'a bumpalo::Bump, binder: &str) -> &'a crate::ast::Expr<'a> {
        let add = arena.alloc(crate::ast::Expr::BinaryOp {
            left: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                crate::ast::Span::new(0, 0),
            )),
            op: crate::ast::BinOp::Add,
            right: arena.alloc(crate::ast::Expr::Ident(
                Symbol::intern(binder),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        });
        arena.alloc(crate::ast::Expr::BinaryOp {
            left: add,
            op: crate::ast::BinOp::Gt,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                crate::ast::Span::new(0, 0),
            )),
            span: crate::ast::Span::new(0, 0),
        })
    }

    fn symbols() -> SymbolTable<'static> {
        SymbolTable::new(CrateId(DefId(0)))
    }

    /// `exists` binder names are irrelevant to structural equality:
    /// `exists X. Expr<X>` and `exists Y. Expr<Y>` are the SAME type
    /// (alpha-conversion) — the previous strict `n1 == n2` comparison
    /// rejected alpha-equivalent types as unequal.
    #[test]
    fn test_exists_alpha_renaming() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let syms = symbols();
        let a = exists(
            &arena,
            "X",
            generic(
                &arena,
                path(&arena, "Expr"),
                vec![crate::ast::GenericArg::Positional(
                    path(&arena, "X").clone(),
                )],
            ),
        );
        let b = exists(
            &arena,
            "Y",
            generic(
                &arena,
                path(&arena, "Expr"),
                vec![crate::ast::GenericArg::Positional(
                    path(&arena, "Y").clone(),
                )],
            ),
        );
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "`exists X. Expr<X>` and `exists Y. Expr<Y>` must be alpha-equivalent"
        );
        // The binder is unused: `exists X. Int` and `exists Y. Int`.
        let c = exists(&arena, "X", path(&arena, "Int"));
        let d = exists(&arena, "Y", path(&arena, "Int"));
        assert!(type_eq_ignoring_spans(c, d, &[], &syms));
        // The capture guard: `exists X. Y` (Y FREE in the base) and
        // `exists Y. Y` (Y bound) are NOT alpha-equivalent.
        let e = exists(&arena, "X", path(&arena, "Y"));
        let f = exists(&arena, "Y", path(&arena, "Y"));
        assert!(
            !type_eq_ignoring_spans(e, f, &[], &syms),
            "a free binder name must not be captured by the renaming"
        );
    }

    /// Nested SAME-NAME exists binders: the inner binder shadows the outer
    /// one, so the inner renaming must win. Regression for rename_seg
    /// using `.find()` (which returned the OUTERMOST mapping): without the
    /// fix `exists X. exists X. Expr<X>` vs `exists Y. exists Z. Expr<Z>`
    /// compared the inner `X` against `Y` instead of `Z` and reported
    /// unequal (a false negative in GADT pending-eq / or-pattern equality).
    #[test]
    fn test_nested_same_name_exists_shadowing() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let syms = symbols();

        // `exists X. exists X. Expr<X>` — the inner X shadows the outer.
        let a = exists(
            &arena,
            "X",
            exists(
                &arena,
                "X",
                generic(
                    &arena,
                    path(&arena, "Expr"),
                    vec![crate::ast::GenericArg::Positional(
                        path(&arena, "X").clone(),
                    )],
                ),
            ),
        );
        // `exists Y. exists Z. Expr<Z>` — alpha-equivalent to `a`.
        let b = exists(
            &arena,
            "Y",
            exists(
                &arena,
                "Z",
                generic(
                    &arena,
                    path(&arena, "Expr"),
                    vec![crate::ast::GenericArg::Positional(
                        path(&arena, "Z").clone(),
                    )],
                ),
            ),
        );
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "the inner same-name binder must map through the innermost renaming"
        );
        // The mirror (the same-name binder on the left side this time).
        assert!(type_eq_ignoring_spans(b, a, &[], &syms));

        // Same-shadowing through the DBM/invariant machinery: the inner
        // invariant `X > 0` (inner X) must compare equal to `Z > 0`,
        // while the outer invariants reference the OUTER binders.
        let a = exists_inv(
            &arena,
            "X",
            exists_inv(&arena, "X", path(&arena, "Int"), gt_zero(&arena, "X")),
            gt_zero(&arena, "X"),
        );
        let b = exists_inv(
            &arena,
            "Y",
            exists_inv(&arena, "Z", path(&arena, "Int"), gt_zero(&arena, "Z")),
            gt_zero(&arena, "Y"),
        );
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "nested same-name invariant binders must alpha-compare through the innermost renaming"
        );

        // And the negative direction: the SHADOWED outer binder is NOT the
        // one referenced — `exists X. exists X. Expr<X>` vs
        // `exists X. exists Z. Expr<Z>` is still equal (the outer name is
        // unused), but `exists X. exists X. Expr<W>` (free W) is NOT equal
        // to `exists Y. exists Z. Expr<Z>` — the free W must not be
        // captured by either renaming.
        let w_free = exists(
            &arena,
            "X",
            exists(
                &arena,
                "X",
                generic(
                    &arena,
                    path(&arena, "Expr"),
                    vec![crate::ast::GenericArg::Positional(
                        path(&arena, "W").clone(),
                    )],
                ),
            ),
        );
        assert!(
            !type_eq_ignoring_spans(w_free, b, &[], &syms),
            "a free name must not be captured by the renaming"
        );

        // THREE levels with EQUAL names at the innermost position: the
        // innermost `X` must resolve to ITSELF (identity), NOT through a
        // stale outer `(X, _)` mapping. `exists X. exists X. exists X.
        // Expr<X>` (base X = innermost binder) vs `exists Y. exists Z.
        // exists X. Expr<Z>` (base Z = MIDDLE binder) are NOT
        // alpha-equivalent — before the identity push, the innermost
        // equal-name pair left no mapping, so `a`'s base X resolved through
        // the middle `(X, Z)` entry and the types compared EQUAL (a false
        // positive introduced by the `rev()` lookup change).
        let triple = exists(
            &arena,
            "X",
            exists(
                &arena,
                "X",
                exists(
                    &arena,
                    "X",
                    generic(
                        &arena,
                        path(&arena, "Expr"),
                        vec![crate::ast::GenericArg::Positional(
                            path(&arena, "X").clone(),
                        )],
                    ),
                ),
            ),
        );
        let triple_mid = exists(
            &arena,
            "Y",
            exists(
                &arena,
                "Z",
                exists(
                    &arena,
                    "X",
                    generic(
                        &arena,
                        path(&arena, "Expr"),
                        vec![crate::ast::GenericArg::Positional(
                            path(&arena, "Z").clone(),
                        )],
                    ),
                ),
            ),
        );
        assert!(
            !type_eq_ignoring_spans(triple, triple_mid, &[], &syms),
            "the innermost equal-name binder must not resolve through a stale outer mapping"
        );
        // NOTE: the mirror direction (triple_mid, triple) is a PRE-EXISTING
        // name-collision false positive (the b-side repeats `X` at multiple
        // depths while `a`'s base maps to the name `X`), present before the
        // `rev()` change too — the name-keyed map cannot distinguish binder
        // positions when BOTH sides repeat the same name. Out of scope for
        // the identity-push fix (which repairs the a-side-shadowing case).

        // The depth-matched triple IS alpha-equivalent: `exists X. exists X.
        // exists X. Expr<X>` vs `exists Y. exists Z. exists W. Expr<W>`.
        let triple_inner = exists(
            &arena,
            "Y",
            exists(
                &arena,
                "Z",
                exists(
                    &arena,
                    "W",
                    generic(
                        &arena,
                        path(&arena, "Expr"),
                        vec![crate::ast::GenericArg::Positional(
                            path(&arena, "W").clone(),
                        )],
                    ),
                ),
            ),
        );
        assert!(
            type_eq_ignoring_spans(triple, triple_inner, &[], &syms),
            "depth-matched equal-name triples must still compare equal"
        );
        assert!(type_eq_ignoring_spans(triple_inner, triple, &[], &syms));
    }

    /// The `invariant` clause IS part of the type's identity: two exists
    /// types are equal only if their invariants are alpha-equivalent —
    /// `exists X. Int invariant X > 0` and `exists Y. Int invariant Y > 0`
    /// are the same type, but `... invariant Y < 0` is a DIFFERENT type.
    #[test]
    fn test_exists_invariant_identity() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let syms = symbols();
        // Alpha-equivalent invariants: `X > 0` ≡ `Y > 0`.
        let a = exists_inv(&arena, "X", path(&arena, "Int"), gt_zero(&arena, "X"));
        let b = exists_inv(&arena, "Y", path(&arena, "Int"), gt_zero(&arena, "Y"));
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "alpha-equivalent invariants must compare equal"
        );
        // Different invariants: `X > 0` vs `Y < 0` — DIFFERENT types.
        let c = exists_inv(&arena, "X", path(&arena, "Int"), gt_zero(&arena, "X"));
        let d = exists_inv(&arena, "Y", path(&arena, "Int"), lt_zero(&arena, "Y"));
        assert!(
            !type_eq_ignoring_spans(c, d, &[], &syms),
            "different invariants must compare unequal"
        );
        // Base AND invariant both alpha-equivalent:
        // `exists X. Expr<X> invariant X > 0` ≡ `exists Y. Expr<Y> invariant Y > 0`.
        let e = exists_inv(
            &arena,
            "X",
            generic(
                &arena,
                path(&arena, "Expr"),
                vec![crate::ast::GenericArg::Positional(
                    path(&arena, "X").clone(),
                )],
            ),
            gt_zero(&arena, "X"),
        );
        let f = exists_inv(
            &arena,
            "Y",
            generic(
                &arena,
                path(&arena, "Expr"),
                vec![crate::ast::GenericArg::Positional(
                    path(&arena, "Y").clone(),
                )],
            ),
            gt_zero(&arena, "Y"),
        );
        assert!(
            type_eq_ignoring_spans(e, f, &[], &syms),
            "base + invariant both alpha-equivalent must compare equal"
        );
        // The invariant capture guard: `exists X. Int invariant Y > 0`
        // (Y FREE in the invariant) vs `exists Y. Int invariant Y > 0`
        // (Y bound) are NOT alpha-equivalent.
        let g = exists_inv(&arena, "X", path(&arena, "Int"), gt_zero(&arena, "Y"));
        let h = exists_inv(&arena, "Y", path(&arena, "Int"), gt_zero(&arena, "Y"));
        assert!(
            !type_eq_ignoring_spans(g, h, &[], &syms),
            "a free binder name in the invariant must not be captured"
        );
    }

    /// L1 (identity peeling) + L2 (integer-discrete comparison
    /// canonicalization): `X > 0` ≡ `X >= 1` on INTEGER bases (the bounds
    /// differ by one), `0 < X` ≡ `X > 0` (constant mirrored to the right),
    /// `(0 + X) > 0` ≡ `X >= 1` (peel then canonicalize). Floats/rationals
    /// are dense — the bounds genuinely differ and compare exactly.
    #[test]
    fn test_expr_l1_l2_normalization() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        // L2: integer discreteness — `X > 0` ≡ `X >= 1`.
        assert!(
            expr_eq_ignoring_spans_typed(gt_zero(&arena, "X"), ge_one(&arena, "X"), true),
            "X > 0 must equal X >= 1 on integer bases"
        );
        // Dense bases: the two bounds genuinely differ.
        assert!(
            !expr_eq_ignoring_spans_typed(gt_zero(&arena, "X"), ge_one(&arena, "X"), false),
            "X > 0 must NOT equal X >= 1 on dense (float/rational) bases"
        );
        // L2: the constant is mirrored to the right — `0 < X` ≺ `X > 0`.
        assert!(
            expr_eq_ignoring_spans_typed(zero_lt(&arena, "X"), gt_zero(&arena, "X"), true),
            "0 < X must equal X > 0"
        );
        // L1 + L2: peel `0 + X` then canonicalize — `(0 + X) > 0` ≡ `X >= 1`.
        assert!(
            expr_eq_ignoring_spans_typed(zero_plus_gt_zero(&arena, "X"), ge_one(&arena, "X"), true,),
            "(0 + X) > 0 must equal X >= 1 on integer bases"
        );
        // Alpha-renaming + L2 at the type level: `exists X. Int invariant
        // X > 0` ≡ `exists Y. Int invariant Y >= 1` (Int is discrete).
        let syms = symbols();
        let a = exists_inv(&arena, "X", path(&arena, "Int"), gt_zero(&arena, "X"));
        let b = exists_inv(&arena, "Y", path(&arena, "Int"), ge_one(&arena, "Y"));
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "Int base: X > 0 must equal Y >= 1"
        );
        // Float base: the same invariant pair genuinely differs.
        let c = exists_inv(&arena, "X", path(&arena, "Float"), gt_zero(&arena, "X"));
        let d = exists_inv(&arena, "Y", path(&arena, "Float"), ge_one(&arena, "Y"));
        assert!(
            !type_eq_ignoring_spans(c, d, &[], &syms),
            "Float base: X > 0 must NOT equal Y >= 1"
        );
    }

    /// L3 (affine normalization): the constant is folded across `X + k`,
    /// `k * X`, and `-X` on INTEGER bases — `X + 1 > 1` ≡ `X > 0`,
    /// `2 * X > 5` ≡ `X > 2`, `-X < 0` ≡ `X > 0`. Floats/rationals and
    /// multi-variable shapes keep the exact structural comparison
    /// (fail-closed: unproven equality is NOT equality).
    #[test]
    fn test_expr_l3_affine_normalization() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        // Offset folding: `X + 1 > 1` ≡ `X > 0`.
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, add_lit(&arena, x, 1), 1),
                gt_zero(&arena, "X"),
                true,
            ),
            "X + 1 > 1 must equal X > 0"
        );
        // Offset + discreteness: `X - 1 >= 1` ≡ `X > 1`.
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Ge, sub_lit(&arena, x, 1), 1),
                cmp_lit(&arena, crate::ast::BinOp::Gt, x, 1),
                true,
            ),
            "X - 1 >= 1 must equal X > 1"
        );
        // Coefficient scaling: `2 * X > 5` ≡ `X > 2` and `2 * X >= 6` ≡
        // `X > 2` (discrete division).
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, mul_lit(&arena, x, 2), 5),
                cmp_lit(&arena, crate::ast::BinOp::Gt, x, 2),
                true,
            ),
            "2 * X > 5 must equal X > 2"
        );
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Ge, mul_lit(&arena, x, 2), 6),
                cmp_lit(&arena, crate::ast::BinOp::Gt, x, 2),
                true,
            ),
            "2 * X >= 6 must equal X > 2"
        );
        // Sign flip: `-X < 0` ≡ `X > 0` and `-2 * X > 5` ≡ `X < -2`.
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Lt, neg(&arena, x), 0),
                gt_zero(&arena, "X"),
                true,
            ),
            "-X < 0 must equal X > 0"
        );
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(
                    &arena,
                    crate::ast::BinOp::Gt,
                    neg(&arena, mul_lit(&arena, x, 2)),
                    5,
                ),
                cmp_lit(&arena, crate::ast::BinOp::Lt, x, -2),
                true,
            ),
            "-2 * X > 5 must equal X < -2"
        );
        // Dense (float/rational) bases: NO affine rewrite — `X + 1 > 1` vs
        // `X > 0` compare structurally (fail-closed).
        assert!(
            !expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, add_lit(&arena, x, 1), 1),
                gt_zero(&arena, "X"),
                false,
            ),
            "dense base: X + 1 > 1 must NOT equal X > 0"
        );
        // Multi-variable shapes are now handled by the L3.5 LINEAR
        // normalization: `X + Y > 0` ≡ `X > -Y` (transposition — both merge
        // to the same coefficient table `{X: 1, Y: 1}`).
        let y = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Y"),
            crate::ast::Span::new(0, 0),
        ));
        let xpy = arena.alloc(crate::ast::Expr::BinaryOp {
            left: x,
            op: crate::ast::BinOp::Add,
            right: y,
            span: crate::ast::Span::new(0, 0),
        });
        let x_gt_neg_y = arena.alloc(crate::ast::Expr::BinaryOp {
            left: x,
            op: crate::ast::BinOp::Gt,
            right: neg(&arena, y),
            span: crate::ast::Span::new(0, 0),
        });
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, xpy, 0),
                x_gt_neg_y,
                true,
            ),
            "multi-variable: X + Y > 0 must equal X > -Y"
        );
    }

    /// L3.5 (multi-variable LINEAR normalization): both sides of a
    /// comparison are affine and merged onto the left —
    /// `X - Y + 1 > 1` ≡ `X > Y` (offset folding), `X - Y >= 1` ≡ `X > Y`
    /// (discreteness), `X >= Y` ≡ `X > Y - 1`. Coefficient tables that
    /// differ (`2X - 2Y > 0` vs `X > Y`) and non-linear shapes stay
    /// structural (fail-closed).
    #[test]
    fn test_expr_l3_5_linear_normalization() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        let y = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Y"),
            crate::ast::Span::new(0, 0),
        ));
        let bin = |op: crate::ast::BinOp,
                   l: &'static crate::ast::Expr<'static>,
                   r: &'static crate::ast::Expr<'static>| {
            arena.alloc(crate::ast::Expr::BinaryOp {
                left: l,
                op,
                right: r,
                span: crate::ast::Span::new(0, 0),
            })
        };
        let xmy = bin(crate::ast::BinOp::Sub, x, y);
        // Offset folding across the subtraction: `X - Y + 1 > 1` ≡ `X > Y`.
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, add_lit(&arena, xmy, 1), 1),
                bin(crate::ast::BinOp::Gt, x, y),
                true,
            ),
            "X - Y + 1 > 1 must equal X > Y"
        );
        // Discreteness on the merged form: `X - Y >= 1` ≡ `X > Y`.
        assert!(
            expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Ge, xmy, 1),
                bin(crate::ast::BinOp::Gt, x, y),
                true,
            ),
            "X - Y >= 1 must equal X > Y"
        );
        // `X >= Y` ≡ `X > Y - 1` (both merge to `X - Y + 1 > 0`).
        assert!(
            expr_eq_ignoring_spans_typed(
                bin(crate::ast::BinOp::Ge, x, y),
                bin(crate::ast::BinOp::Gt, x, sub_lit(&arena, y, 1)),
                true,
            ),
            "X >= Y must equal X > Y - 1"
        );
        // Coefficient tables that differ stay unequal (no gcd reduction):
        // `2X - 2Y > 0` vs `X > Y`.
        let two_xmy = bin(
            crate::ast::BinOp::Sub,
            mul_lit(&arena, x, 2),
            mul_lit(&arena, y, 2),
        );
        assert!(
            !expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, two_xmy, 0),
                bin(crate::ast::BinOp::Gt, x, y),
                true,
            ),
            "2X - 2Y > 0 must NOT equal X > Y (no gcd reduction)"
        );
        // `X > Y` vs `X >= Y` genuinely differ.
        assert!(
            !expr_eq_ignoring_spans_typed(
                bin(crate::ast::BinOp::Gt, x, y),
                bin(crate::ast::BinOp::Ge, x, y),
                true,
            ),
            "X > Y must NOT equal X >= Y"
        );
        // Dense (float/rational) bases: NO linear rewrite — structural.
        assert!(
            !expr_eq_ignoring_spans_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, add_lit(&arena, xmy, 1), 1),
                bin(crate::ast::BinOp::Gt, x, y),
                false,
            ),
            "dense base: X - Y + 1 > 1 must NOT equal X > Y"
        );
        // Non-linear shapes stay structural: `X * Y > 0` vs `X > 0`.
        assert!(
            !expr_eq_ignoring_spans_typed(
                cmp_lit(
                    &arena,
                    crate::ast::BinOp::Gt,
                    bin(crate::ast::BinOp::Mul, x, y),
                    0,
                ),
                gt_zero(&arena, "X"),
                true,
            ),
            "X * Y > 0 must NOT equal X > 0 (non-linear)"
        );
    }

    /// L3.5+ (octagon / DBM closure on conjunctions): `A and B` compares
    /// through the closed difference-bound matrix — redundancy elimination
    /// (`X > 0 and X > 1` ≡ `X > 1`), discreteness + order-insensitivity
    /// (`(X > 0) and (X < 10)` ≡ `(X <= 9) and (X >= 1)`), and transitivity
    /// (`X - Y < 5 and Y - Z < 3` ≡ `X - Z < 7`). Tautological clauses
    /// vanish; non-octagon shapes (`!=`, non-unit coefficients) fall back
    /// to structural equality (fail-closed).
    #[test]
    fn test_expr_dbm_octagon_closure() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        let y = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Y"),
            crate::ast::Span::new(0, 0),
        ));
        let z = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Z"),
            crate::ast::Span::new(0, 0),
        ));
        let bin = |op: crate::ast::BinOp,
                   l: &'static crate::ast::Expr<'static>,
                   r: &'static crate::ast::Expr<'static>| {
            arena.alloc(crate::ast::Expr::BinaryOp {
                left: l,
                op,
                right: r,
                span: crate::ast::Span::new(0, 0),
            })
        };
        let x_gt_0 = cmp_lit(&arena, crate::ast::BinOp::Gt, x, 0);
        let x_gt_1 = cmp_lit(&arena, crate::ast::BinOp::Gt, x, 1);
        let x_lt_10 = cmp_lit(&arena, crate::ast::BinOp::Lt, x, 10);
        let x_ge_1 = cmp_lit(&arena, crate::ast::BinOp::Ge, x, 1);
        let x_le_9 = cmp_lit(&arena, crate::ast::BinOp::Le, x, 9);
        let xmy = bin(crate::ast::BinOp::Sub, x, y);
        let ymz = bin(crate::ast::BinOp::Sub, y, z);
        let xmz = bin(crate::ast::BinOp::Sub, x, z);

        // Redundancy elimination: `X > 0 and X > 1` ≡ `X > 1` (the stronger
        // bound absorbs the weaker one in the closure).
        assert!(
            expr_eq_ignoring_spans_typed(and_expr(&arena, x_gt_0, x_gt_1), x_gt_1, true,),
            "X > 0 and X > 1 must equal X > 1 (redundancy elimination)"
        );
        // Discreteness + order-insensitivity: `(X > 0) and (X < 10)` ≡
        // `(X <= 9) and (X >= 1)` (clauses reordered on the right).
        assert!(
            expr_eq_ignoring_spans_typed(
                and_expr(&arena, x_gt_0, x_lt_10),
                and_expr(&arena, x_le_9, x_ge_1),
                true,
            ),
            "(X > 0) and (X < 10) must equal (X <= 9) and (X >= 1)"
        );
        // Transitivity + redundancy: the closure infers `X - Z ≤ 6` from
        // `X - Y ≤ 4 ∧ Y - Z ≤ 2`, making the explicit `X - Z < 7` clause
        // on the right redundant — the two conjunctions are equal. (A
        // bare `X - Z < 7` WITHOUT the `Y` clauses is NOT equal under
        // Posita's all-quantified free-variable semantics: the left side
        // additionally constrains `Y`.)
        assert!(
            expr_eq_ignoring_spans_typed(
                and_expr(
                    &arena,
                    cmp_lit(&arena, crate::ast::BinOp::Lt, xmy, 5),
                    cmp_lit(&arena, crate::ast::BinOp::Lt, ymz, 3),
                ),
                and_expr(
                    &arena,
                    and_expr(
                        &arena,
                        cmp_lit(&arena, crate::ast::BinOp::Lt, xmz, 7),
                        cmp_lit(&arena, crate::ast::BinOp::Lt, xmy, 5),
                    ),
                    cmp_lit(&arena, crate::ast::BinOp::Lt, ymz, 3),
                ),
                true,
            ),
            "X - Y < 5 and Y - Z < 3 must equal X - Z < 7 and X - Y < 5 and Y - Z < 3 (transitivity)"
        );
        // A tautological clause vanishes: `(X > 0) and (X - X < 1)` ≡
        // `X > 0` (the `X - X` clause collapses to `0 < 1`).
        assert!(
            expr_eq_ignoring_spans_typed(
                and_expr(
                    &arena,
                    x_gt_0,
                    cmp_lit(
                        &arena,
                        crate::ast::BinOp::Lt,
                        bin(crate::ast::BinOp::Sub, x, x),
                        1,
                    ),
                ),
                x_gt_0,
                true,
            ),
            "(X > 0) and (X - X < 1) must equal X > 0 (tautology)"
        );
        // Non-convex `!=` cannot be encoded: `(X != 0) and (X > 0)` vs
        // `X > 0` stays structural (fail-closed — NOT equal).
        assert!(
            !expr_eq_ignoring_spans_typed(
                and_expr(
                    &arena,
                    cmp_lit(&arena, crate::ast::BinOp::Neq, x, 0),
                    x_gt_0,
                ),
                x_gt_0,
                true,
            ),
            "X != 0 and X > 0 must NOT equal X > 0 (non-convex, fail-closed)"
        );
        // Non-unit coefficients are not octagon: `(2X - 2Y > 0) and
        // (Y > 0)` vs `(X > Y) and (Y > 0)` stays structural.
        let two_xmy = bin(
            crate::ast::BinOp::Sub,
            mul_lit(&arena, x, 2),
            mul_lit(&arena, y, 2),
        );
        assert!(
            !expr_eq_ignoring_spans_typed(
                and_expr(
                    &arena,
                    cmp_lit(&arena, crate::ast::BinOp::Gt, two_xmy, 0),
                    cmp_lit(&arena, crate::ast::BinOp::Gt, y, 0),
                ),
                and_expr(
                    &arena,
                    bin(crate::ast::BinOp::Gt, x, y),
                    cmp_lit(&arena, crate::ast::BinOp::Gt, y, 0),
                ),
                true,
            ),
            "2X - 2Y > 0 and Y > 0 must NOT equal X > Y and Y > 0 (non-unit)"
        );
    }

    /// Review-fix regression: alpha-renaming must survive the DBM
    /// encoding — `(X > 0) and (X < 10)` vs `(Y < 10) and (Y > 0)`
    /// (renamed binder + REORDERED clauses) are equal via the octagon
    /// closure. Before the fix the encode phase looked up UN-renamed
    /// symbols in `var_of` (which holds the renamed set) and silently fell
    /// back to structural equality — which fails on reordered clauses.
    #[test]
    fn test_expr_dbm_renamed_conjunction() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        let y = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Y"),
            crate::ast::Span::new(0, 0),
        ));
        let rename = [(Symbol::intern("X"), Symbol::intern("Y"))];
        // Expression level: renamed binder, clauses reordered on the right.
        let a = and_expr(
            &arena,
            cmp_lit(&arena, crate::ast::BinOp::Gt, x, 0),
            cmp_lit(&arena, crate::ast::BinOp::Lt, x, 10),
        );
        let b = and_expr(
            &arena,
            cmp_lit(&arena, crate::ast::BinOp::Lt, y, 10),
            cmp_lit(&arena, crate::ast::BinOp::Gt, y, 0),
        );
        assert!(
            expr_eq_ignoring_spans_renamed_typed(a, b, &rename, true),
            "renamed, reordered conjunction must be equal via DBM closure"
        );
        // Type level: `exists X. Int invariant (X > 0) and (X < 10)` vs
        // `exists Y. Int invariant (Y < 10) and (Y > 0)`.
        let syms = symbols();
        let ta = exists_inv(
            &arena,
            "X",
            path(&arena, "Int"),
            and_expr(
                &arena,
                cmp_lit(&arena, crate::ast::BinOp::Gt, x, 0),
                cmp_lit(&arena, crate::ast::BinOp::Lt, x, 10),
            ),
        );
        let tb = exists_inv(
            &arena,
            "Y",
            path(&arena, "Int"),
            and_expr(
                &arena,
                cmp_lit(&arena, crate::ast::BinOp::Lt, y, 10),
                cmp_lit(&arena, crate::ast::BinOp::Gt, y, 0),
            ),
        );
        assert!(
            type_eq_ignoring_spans(ta, tb, &[], &syms),
            "exists-level renamed conjunction must be equal"
        );
    }

    /// Review-fix regression: extreme `i128` literals must not panic or
    /// wrap during the discreteness normalization — `X >= i128::MIN`
    /// cannot be adjusted to `X > i128::MIN - 1`, so the comparison falls
    /// back to structural equality (fail-closed). Before the fix this
    /// panicked in debug builds (and silently wrapped to `X > MAX` in
    /// release).
    #[test]
    fn test_expr_extreme_literal_bounds() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        // `X >= i128::MIN` — the `c - 1` discreteness adjustment overflows.
        let ge_min = cmp_lit(&arena, crate::ast::BinOp::Ge, x, i128::MIN);
        // Must NOT panic; the two sides compare equal via the structural
        // fallback.
        assert!(
            expr_eq_ignoring_spans_typed(ge_min, ge_min, true),
            "X >= i128::MIN must equal itself (no panic on overflow)"
        );
        // `X <= i128::MAX` — the `c + 1` adjustment overflows.
        let le_max = cmp_lit(&arena, crate::ast::BinOp::Le, x, i128::MAX);
        assert!(
            expr_eq_ignoring_spans_typed(le_max, le_max, true),
            "X <= i128::MAX must equal itself (no panic on overflow)"
        );
        // Different extreme bounds stay unequal (no silent wrapping).
        assert!(
            !expr_eq_ignoring_spans_typed(ge_min, le_max, true),
            "X >= i128::MIN must NOT equal X <= i128::MAX"
        );
    }

    /// Review-fix regression: the capture-avoidance guard must scan
    /// NESTED exists invariants — `binder_free_in` previously discarded
    /// the invariant field (`..`), so a free occurrence of the other
    /// side's binder inside a nested exists invariant was invisible to
    /// the guard and the alpha-renaming captured it. Constructed at the
    /// AST level (the free symbols are abstract in a type-pattern
    /// comparison): `W` is FREE in A's nested invariant, `X` is FREE in
    /// B's — renaming `X → W` must NOT be allowed.
    #[test]
    fn test_nested_exists_invariant_capture_guard() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let syms = symbols();
        let a = exists_inv(
            &arena,
            "X",
            exists_inv(&arena, "Y", path(&arena, "Int"), gt_zero(&arena, "W")),
            gt_zero(&arena, "X"),
        );
        let b = exists_inv(
            &arena,
            "W",
            exists_inv(&arena, "Y", path(&arena, "Int"), gt_zero(&arena, "X")),
            gt_zero(&arena, "W"),
        );
        assert!(
            !type_eq_ignoring_spans(a, b, &[], &syms),
            "free variables in nested invariants must not be captured by alpha-renaming"
        );
    }

    /// Review-fix regression: `expr_free_in` must be FAIL-CLOSED on
    /// unenumerated expression shapes — a name inside an `if` expression is
    /// reported as "possibly free" (`true`), so the capture-avoidance guard
    /// blocks the renaming. Before the fix the `_ => false` wildcard
    /// ignored `if`/`match`/etc., permitting captures.
    #[test]
    fn test_expr_free_in_fail_closed() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let span = crate::ast::Span::new(0, 0);
        let x = arena.alloc(crate::ast::Expr::Ident(Symbol::intern("X"), span));
        let gt0 = arena.alloc(crate::ast::Expr::BinaryOp {
            left: x,
            op: crate::ast::BinOp::Gt,
            right: arena.alloc(crate::ast::Expr::Literal(
                crate::ast::Literal::Int(crate::ast::IntLit::Small(0)),
                span,
            )),
            span,
        });
        let inv5 = arena.alloc(crate::ast::Expr::Ident(Symbol::intern("_inv_5"), span));
        // Precise path: a name inside an enumerated container (Index) is
        // found.
        let idx = arena.alloc(crate::ast::Expr::Index {
            base: x,
            index: inv5,
            span,
        });
        assert!(
            expr_free_in(Symbol::intern("_inv_5"), idx),
            "a name inside an Index expression must be found"
        );
        // Fail-closed: a name inside an UNENUMERATED shape (If) is treated
        // as possibly free — the guard must NOT allow the rename.
        let if_expr = arena.alloc(crate::ast::Expr::If {
            cond: gt0,
            then_branch: vec![crate::ast::Stmt::Expression(inv5.clone())],
            else_branch: None,
            is_expression: true,
            span,
        });
        assert!(
            expr_free_in(Symbol::intern("_inv_5"), if_expr),
            "a name inside an if expression must be reported free (fail-closed)"
        );
    }

    /// Founder's-bifurcation test: the entailment check `a ⟹ b` is EXACT
    /// within the difference-constraint sub-language (self-verifying, may
    /// discharge directly); non-expressible shapes (non-linear premises
    /// or conclusions, `E == 0` negated to `E ≠ 0`, dense bases) fall
    /// back to `None` (fail-closed).
    #[test]
    fn test_expr_entails_typed() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        let y = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("Y"),
            crate::ast::Span::new(0, 0),
        ));
        let bin = |op: crate::ast::BinOp,
                   l: &'static crate::ast::Expr<'static>,
                   r: &'static crate::ast::Expr<'static>|
         -> &'static crate::ast::Expr<'static> {
            arena.alloc(crate::ast::Expr::BinaryOp {
                left: l,
                op,
                right: r,
                span: crate::ast::Span::new(0, 0),
            })
        };
        let x_gt_0 = cmp_lit(&arena, crate::ast::BinOp::Gt, x, 0);
        let x_ge_1 = cmp_lit(&arena, crate::ast::BinOp::Ge, x, 1);
        let x_ge_0 = cmp_lit(&arena, crate::ast::BinOp::Ge, x, 0);
        let x_gt_5 = cmp_lit(&arena, crate::ast::BinOp::Gt, x, 5);
        let x_ne_0 = cmp_lit(&arena, crate::ast::BinOp::Neq, x, 0);
        let x_eq_1 = cmp_lit(&arena, crate::ast::BinOp::Eq, x, 1);
        // Exact entailments (integer discreteness).
        assert_eq!(
            expr_entails_typed(x_gt_0, x_ge_1, true),
            Some(true),
            "X > 0 ⟹ X >= 1"
        );
        assert_eq!(
            expr_entails_typed(x_gt_0, x_ge_0, true),
            Some(true),
            "X > 0 ⟹ X >= 0"
        );
        assert_eq!(
            expr_entails_typed(x_gt_0, x_ne_0, true),
            Some(true),
            "X > 0 ⟹ X != 0"
        );
        assert_eq!(
            expr_entails_typed(x_gt_0, x_gt_5, true),
            Some(false),
            "X > 0 does NOT entail X > 5"
        );
        // Multi-variable difference entailment: `X - Y > 0 ⟹ X > Y`.
        let xmy = bin(crate::ast::BinOp::Sub, x, y);
        assert_eq!(
            expr_entails_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, xmy, 0),
                bin(crate::ast::BinOp::Gt, x, y),
                true,
            ),
            Some(true),
            "X - Y > 0 ⟹ X > Y"
        );
        // Fail-closed fallbacks.
        assert_eq!(
            expr_entails_typed(x_gt_0, x_eq_1, true),
            None,
            "E == 0 negated to E ≠ 0 is non-convex (fail-closed)"
        );
        let xmul = bin(crate::ast::BinOp::Mul, x, y);
        assert_eq!(
            expr_entails_typed(
                cmp_lit(&arena, crate::ast::BinOp::Gt, xmul, 0),
                x_gt_0,
                true
            ),
            None,
            "a non-linear premise is not expressible (fail-closed)"
        );
        assert_eq!(
            expr_entails_typed(x_gt_0, x_ge_1, false),
            None,
            "dense bases are not discretely entailed (fail-closed)"
        );
    }

    /// The lattice operations and transfer
    /// functions on `Dbm` — meet (tighter), join (looser), widen
    /// (termination-guaranteed relaxation), and the assignment/test
    /// transfer functions (`X := X + c`, `X := c`, `X := Y`, `X ≤ c`,
    /// `X - Y ≤ c`). Node numbering (paper §IV.C): `Xᵢ⁺ = 2i`,
    /// `Xᵢ⁻ = 2i+1` — single-variable bounds ride the SELF-DUAL edges
    /// (paper Figure 5: `v ≤ c` ⟺ `v⁺ − v⁻ ≤ 2c`, so `X ≤ c` is
    /// `set(2i, 2i+1, 2c)` and `X ≥ c` is `set(2i+1, 2i, −2c)`).
    /// All reads go through the semantic projections — the raw cells
    /// carry the doubled storage (4× on the self-dual interval edges)
    /// and must never be read directly.
    #[test]
    fn test_dbm_octagon_operations() {
        // meet: `X ≥ 1` ⊓ `X ≤ 10` → `X ∈ [1, 10]`.
        let mut lo = Dbm::new(1);
        lo.set(1, 0, -2); // X⁻ − X⁺ ≤ −2 ⟺ −2X ≤ −2 ⟺ X ≥ 1
        lo.close();
        let mut hi = Dbm::new(1);
        hi.set(0, 1, 20); // X⁺ − X⁻ ≤ 20 ⟺ 2X ≤ 20 ⟺ X ≤ 10
        hi.close();
        let m = lo.meet(&hi);
        assert_eq!(m.var_ub(0), Some(10), "meet keeps X ≤ 10");
        assert_eq!(m.var_lb(0), Some(1), "meet keeps X ≥ 1");
        // join: `X ≤ 5` ⊔ `X ≤ 10` → `X ≤ 10` (the looser bound).
        let mut a = Dbm::new(1);
        a.set(0, 1, 10); // 2X ≤ 10 ⟺ X ≤ 5
        a.close();
        let mut b = Dbm::new(1);
        b.set(0, 1, 20); // 2X ≤ 20 ⟺ X ≤ 10
        b.close();
        let j = a.join(&b);
        assert_eq!(j.var_ub(0), Some(10), "join keeps the looser X ≤ 10");
        // widen: per the paper, `n ≤ old` keeps OLD, otherwise ∞. The old
        // assertion expected lo.widen(&hi) to keep hi's upper bound — but
        // lo's upper bound is ∞ and 20 ≤ ∞ keeps ∞, so keeping 10 would
        // violate Theorem 8.1's m ▽ n ⊒ n. Use a stable/relaxing pair:
        // cur = {X≤5, X≥0}, next = {X≤10, X≥0} — the upper bound relaxes
        // (20 ≤ 10 false) → ∞; X ≥ 0 is stable.
        let mut cur = Dbm::new(1);
        cur.set(0, 1, 10); // X ≤ 5
        cur.set(1, 0, 0); // X ≥ 0
        cur.close();
        let mut next = Dbm::new(1);
        next.set(0, 1, 20); // X ≤ 10
        next.set(1, 0, 0); // X ≥ 0
        next.close();
        let w = cur.widen(&next);
        assert_eq!(w.var_ub(0), None, "widen drops the relaxing X ≤ 5→10");
        assert_eq!(w.var_lb(0), Some(0), "widen keeps the stable X ≥ 0");
        // Transfer: `X := X + 1` from `X ≥ 0` → `X ≥ 1`.
        let mut ge0 = Dbm::new(1);
        ge0.set(1, 0, 0); // −2X ≤ 0 ⟺ X ≥ 0
        ge0.close();
        let t = ge0.assign_add_var(0, 1);
        assert_eq!(t.var_lb(0), Some(1), "X := X + 1 turns X ≥ 0 into X ≥ 1");
        // Transfer: `X := 5` pins X to [5, 5].
        let c5 = ge0.assign_const_var(0, 5);
        assert_eq!(c5.var_ub(0), Some(5), "X := 5 sets X ≤ 5");
        assert_eq!(c5.var_lb(0), Some(5), "X := 5 sets X ≥ 5");
        // Transfer: `X := Y` with `Y ≥ 3` (two variables: X⁺=0, X⁻=1,
        // Y⁺=2, Y⁻=3 in the 2N numbering).
        let mut yge3 = Dbm::new(2);
        yge3.set(3, 2, -6); // Y⁻ − Y⁺ ≤ −6 ⟺ −2Y ≤ −6 ⟺ Y ≥ 3
        yge3.close();
        let cp = yge3.assign_copy_var(0, 1);
        assert_eq!(cp.var_lb(0), Some(3), "X := Y inherits Y ≥ 3");
        // Transfer: test `X ≤ 7` and `X - Y ≤ 2`.
        let top = Dbm::new(2).top();
        let tle = top.test_le_var(0, 7);
        assert_eq!(tle.var_ub(0), Some(7), "test X ≤ 7 adds the upper bound");
        let tdiff = top.test_diff_le(0, 1, 2);
        assert_eq!(
            tdiff.diff_bound(0, 1),
            Some(2),
            "test X - Y ≤ 2 adds the difference"
        );
    }

    /// The widened fixpoint iteration — `while true { i := i + 1 }`
    /// (init `i = 0`, body `i := i + 1`) converges to the unbounded
    /// invariant `i ≥ 0`, not to an ever-growing interval.
    #[test]
    fn test_dbm_fixpoint() {
        let mut init = Dbm::new(1);
        init.set(0, 1, 0); // 2X ≤ 0 ⟺ X ≤ 0
        init.set(1, 0, 0); // −2X ≤ 0 ⟺ X ≥ 0   (X = 0)
        init.close();
        let step = |d: &Dbm| d.assign_add_var(0, 1);
        let fp = dbm_fixpoint(&init, &step, 100, 2).expect("i := i + 1 converges");
        assert_eq!(fp.var_lb(0), Some(0), "fixpoint keeps i ≥ 0");
        assert_eq!(fp.var_ub(0), None, "fixpoint widens the upper bound");
    }

    /// Fail-closed: a fixpoint that does NOT converge within `max_iter`
    /// returns `None` — the last matrix is not a fixpoint, so its bounds
    /// must not be emitted as invariant candidates.
    #[test]
    fn test_dbm_fixpoint_non_convergence_fails_closed() {
        let mut init = Dbm::new(1);
        init.set(0, 1, 0); // X ≤ 0
        init.set(1, 0, 0); // X ≥ 0
        init.close();
        let step = |d: &Dbm| d.assign_add_var(0, 1);
        assert!(
            dbm_fixpoint(&init, &step, 3, 3).is_none(),
            "a non-converging step must yield None (fail closed)"
        );
        // The old construction (a diagonal set(0,0,-1) decreasing) built a
        // hidden-⊥: join(cur, ⊥) = cur ⟹ pseudo-convergence on cur, so
        // `Some` was the SOUND behavior — it was never a fail-closed test.
        // The new construction: widen_after(3) > max_iter(3), widening never
        // kicks in, join makes cur strictly change every round
        // ([0,0]→[0,1]→[0,2]→[0,3]), the budget runs out → None.
    }

    /// The candidate inverse transform — a closed DBM (`X ∈ [1, 10]`)
    /// yields invariant expressions including `X ≥ 1` and `X ≤ 10`.
    #[test]
    fn test_dbm_to_invariant_exprs() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let mut d = Dbm::new(1);
        d.set(0, 1, 20); // 2X ≤ 20 ⟺ X ≤ 10
        d.set(1, 0, -2); // −2X ≤ −2 ⟺ X ≥ 1
        d.close();
        let vars = vec![Symbol::intern("X")];
        let exprs = dbm_to_invariant_exprs(&arena, &d, &vars);
        let x = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("X"),
            crate::ast::Span::new(0, 0),
        ));
        let le10 = cmp_lit(&arena, crate::ast::BinOp::Le, x, 10);
        let ge1 = cmp_lit(&arena, crate::ast::BinOp::Ge, x, 1);
        assert!(
            exprs.len() >= 2,
            "X ∈ [1, 10] yields at least two facts (got {})",
            exprs.len()
        );
        assert!(
            exprs.iter().any(|e| expr_eq_ignoring_spans(e, le10)),
            "one fact is X ≤ 10"
        );
        assert!(
            exprs.iter().any(|e| expr_eq_ignoring_spans(e, ge1)),
            "one fact is X ≥ 1"
        );
    }

    /// The inference kernel end-to-end — `i = 0; while i < 10 {
    /// i := i + 1 }` yields the invariant candidate `i ≥ 0`; the guard
    /// bound `i ≤ 9` is absorbed by the fixpoint and NOT reported (it only
    /// holds inside the loop, so it is not an invariant).
    #[test]
    fn test_infer_loop_invariant_exprs() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let exprs = infer_loop_invariant_exprs(
            &arena,
            &vars,
            &[LoopInstr::ConstVar(0, 0)],
            &[LoopInstr::TestLe(0, 9), LoopInstr::AddVar(0, 1)],
            100,
            2,
            None,
        );
        let i = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("i"),
            crate::ast::Span::new(0, 0),
        ));
        let ge0 = cmp_lit(&arena, crate::ast::BinOp::Ge, i, 0);
        let le9 = cmp_lit(&arena, crate::ast::BinOp::Le, i, 9);
        assert!(
            exprs.iter().any(|e| expr_eq_ignoring_spans(e, ge0)),
            "the invariant candidate includes i ≥ 0"
        );
        assert!(
            !exprs.iter().any(|e| expr_eq_ignoring_spans(e, le9)),
            "the guard bound i ≤ 9 must NOT leak into the invariant"
        );
    }

    /// The `decreases` inference — `i = 0; while i < 10 { i := i + 1 }`
    /// yields the candidate `9 - i` (the guard bound minus the counter);
    /// a loop without an increasing step has no candidate.
    #[test]
    fn test_infer_loop_decreases() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i")];
        let instrs = vec![LoopInstr::TestLe(0, 9), LoopInstr::AddVar(0, 1)];
        let dec = infer_loop_decreases_expr(&arena, &vars, &instrs);
        assert!(dec.is_some(), "a decreasing measure exists");
        let i = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("i"),
            crate::ast::Span::new(0, 0),
        ));
        let lit9 = arena.alloc(crate::ast::Expr::Literal(
            crate::ast::Literal::Int(crate::ast::IntLit::Small(9)),
            crate::ast::Span::new(0, 0),
        ));
        let expected = arena.alloc(crate::ast::Expr::BinaryOp {
            left: lit9,
            op: crate::ast::BinOp::Sub,
            right: i,
            span: crate::ast::Span::new(0, 0),
        });
        assert!(
            expr_eq_ignoring_spans(dec.unwrap(), expected),
            "decreases candidate is 9 - i"
        );
        // No increasing step → no candidate.
        let no_step = vec![LoopInstr::TestLe(0, 9)];
        assert!(
            infer_loop_decreases_expr(&arena, &vars, &no_step).is_none(),
            "no decreasing measure without an increasing step"
        );
    }

    /// Regression: `while i < j { i := i + 1 }` translates the strict
    /// guard to `TestDiffLe(i, j, -1)`. The decreases inference must
    /// still produce the candidate `j - i` — an offset-0-only match used to
    /// silently drop strict guards (no manual `decreases` annotation
    /// required).
    #[test]
    fn test_infer_loop_decreases_strict_diff_guard() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let vars = vec![Symbol::intern("i"), Symbol::intern("j")];
        // Guard `i < j` → `i - j <= -1`; body `i := i + 1`.
        let instrs = vec![LoopInstr::TestDiffLe(0, 1, -1), LoopInstr::AddVar(0, 1)];
        let dec = infer_loop_decreases_expr(&arena, &vars, &instrs);
        assert!(dec.is_some(), "a decreasing measure exists for `i < j`");
        let i = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("i"),
            crate::ast::Span::new(0, 0),
        ));
        let j = arena.alloc(crate::ast::Expr::Ident(
            Symbol::intern("j"),
            crate::ast::Span::new(0, 0),
        ));
        let expected = arena.alloc(crate::ast::Expr::BinaryOp {
            left: j,
            op: crate::ast::BinOp::Sub,
            right: i,
            span: crate::ast::Span::new(0, 0),
        });
        assert!(
            expr_eq_ignoring_spans(dec.unwrap(), expected),
            "decreases candidate is j - i"
        );
    }

    /// Named generic arguments are identified by NAME, not position:
    /// `Foo<n=Int, m=Bool>` and `Foo<m=Bool, n=Int>` are the same type —
    /// the previous zip-by-index comparison rejected reordered named args.
    #[test]
    fn test_named_generic_args_unordered() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let syms = symbols();
        let named = |n: &str, t: &'static crate::ast::Type<'static>| {
            crate::ast::GenericArg::Named(Symbol::intern(n), t.clone())
        };
        let a = generic(
            &arena,
            path(&arena, "Foo"),
            vec![
                named("n", path(&arena, "Int")),
                named("m", path(&arena, "Bool")),
            ],
        );
        let b = generic(
            &arena,
            path(&arena, "Foo"),
            vec![
                named("m", path(&arena, "Bool")),
                named("n", path(&arena, "Int")),
            ],
        );
        assert!(
            type_eq_ignoring_spans(a, b, &[], &syms),
            "reordered named args must compare equal"
        );
        // Different names are different types.
        let c = generic(
            &arena,
            path(&arena, "Foo"),
            vec![named("n", path(&arena, "Int"))],
        );
        let d = generic(
            &arena,
            path(&arena, "Foo"),
            vec![named("m", path(&arena, "Int"))],
        );
        assert!(!type_eq_ignoring_spans(c, d, &[], &syms));
        // Positional args keep their order.
        let e = generic(
            &arena,
            path(&arena, "Foo"),
            vec![crate::ast::GenericArg::Positional(
                path(&arena, "Int").clone(),
            )],
        );
        let f = generic(
            &arena,
            path(&arena, "Foo"),
            vec![crate::ast::GenericArg::Positional(
                path(&arena, "Int").clone(),
            )],
        );
        assert!(type_eq_ignoring_spans(e, f, &[], &syms));
        // Mixed positional + named (the named ones are unordered).
        let g = generic(
            &arena,
            path(&arena, "Foo"),
            vec![
                crate::ast::GenericArg::Positional(path(&arena, "Int").clone()),
                named("m", path(&arena, "Bool")),
            ],
        );
        let h = generic(
            &arena,
            path(&arena, "Foo"),
            vec![
                named("m", path(&arena, "Bool")),
                crate::ast::GenericArg::Positional(path(&arena, "Int").clone()),
            ],
        );
        assert!(type_eq_ignoring_spans(g, h, &[], &syms));
    }
}
