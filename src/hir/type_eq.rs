//! Span-insensitive, nominally-resolved structural equality for AST types.
//!
//! Shared by the resolver (E065 contradiction detection) and the checker
//! (or-pattern equality).  Residing in a common module keeps the HIR
//! checker from reaching back into the resolver (layering), giving the
//! comparison family a single authoritative home.

use crate::hir::symbol::*;
use crate::hir::types::DefId;
use crate::symbol::Symbol;

/// Structural equality of AST types IGNORING source spans: two `Int<32>`
/// written at different positions are the same type.  Used by the E065
/// contradiction check so identical RHS constraints are not flagged merely
/// because their spans differ.
pub(crate) fn type_eq_ignoring_spans(
    a: &crate::ast::Type,
    b: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable,
) -> bool {
    match (a, b) {
        (crate::ast::Type::Path(p1, _), crate::ast::Type::Path(p2, _)) => {
            // Nominal equality: the RESOLVED constructor identity — `Int`
            // and `core::Int` are the same type, and two aliases to the
            // same type compare equal.  Opaque exists witnesses (or an
            // unresolvable path) fall back to the syntactic comparison.
            match (
                path_ctor_key(p1, exists_params, symbols),
                path_ctor_key(p2, exists_params, symbols),
            ) {
                (Some(k1), Some(k2)) => k1 == k2,
                _ => p1 == p2,
            }
        }
        (crate::ast::Type::Generic(b1, a1, _), crate::ast::Type::Generic(b2, a2, _)) => {
            type_eq_ignoring_spans(b1, b2, exists_params, symbols)
                && generic_args_eq_ignoring_spans(a1, a2, exists_params, symbols)
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
        ) => m1 == m2 && type_eq_ignoring_spans(i1, i2, exists_params, symbols),
        (crate::ast::Type::Pointer(t1, _), crate::ast::Type::Pointer(t2, _)) => {
            type_eq_ignoring_spans(t1, t2, exists_params, symbols)
        }
        (crate::ast::Type::Slice(t1, _), crate::ast::Type::Slice(t2, _)) => {
            type_eq_ignoring_spans(t1, t2, exists_params, symbols)
        }
        (crate::ast::Type::Array(t1, e1, _), crate::ast::Type::Array(t2, e2, _)) => {
            type_eq_ignoring_spans(t1, t2, exists_params, symbols)
                && const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Tuple(e1, _), crate::ast::Type::Tuple(e2, _)) => {
            e1.len() == e2.len()
                && e1
                    .iter()
                    .zip(e2)
                    .all(|(x, y)| type_eq_ignoring_spans(x, y, exists_params, symbols))
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
                && p1
                    .iter()
                    .zip(p2)
                    .all(|(x, y)| type_eq_ignoring_spans(x, y, exists_params, symbols))
                && type_eq_ignoring_spans(r1, r2, exists_params, symbols)
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
            type_eq_ignoring_spans(i1, i2, exists_params, symbols)
                && type_eq_ignoring_spans(t1, t2, exists_params, symbols)
                && n1 == n2
        }
        (crate::ast::Type::DynTrait(es1, _), crate::ast::Type::DynTrait(es2, _)) => {
            es1.len() == es2.len()
                && es1
                    .iter()
                    .zip(es2)
                    .all(|(x, y)| type_eq_ignoring_spans(x, y, exists_params, symbols))
        }
        (
            crate::ast::Type::Exists {
                name: n1, base: b1, ..
            },
            crate::ast::Type::Exists {
                name: n2, base: b2, ..
            },
        ) => n1 == n2 && type_eq_ignoring_spans(b1, b2, exists_params, symbols),
        (
            crate::ast::Type::WhereShorthand { base: b1, .. },
            crate::ast::Type::WhereShorthand { base: b2, .. },
        ) => type_eq_ignoring_spans(b1, b2, exists_params, symbols),
        (crate::ast::Type::Literal(e1, _), crate::ast::Type::Literal(e2, _)) => {
            const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Never(_), crate::ast::Type::Never(_)) => true,
        (crate::ast::Type::Union(es1, _), crate::ast::Type::Union(es2, _)) => {
            es1.len() == es2.len()
                && es1
                    .iter()
                    .zip(es2)
                    .all(|(x, y)| type_eq_ignoring_spans(x, y, exists_params, symbols))
        }
        (crate::ast::Type::Expr(e1, _), crate::ast::Type::Expr(e2, _)) => {
            const_expr_eq_ignoring_spans(e1, e2)
        }
        (crate::ast::Type::Regex(s1, _), crate::ast::Type::Regex(s2, _)) => s1 == s2,
        (crate::ast::Type::Error(_), crate::ast::Type::Error(_)) => true,
        _ => false,
    }
}

/// Compare generic arguments ignoring spans (const args compare by shape).
fn generic_args_eq_ignoring_spans(
    a: &[crate::ast::GenericArg],
    b: &[crate::ast::GenericArg],
    exists_params: &[Symbol],
    symbols: &SymbolTable,
) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (crate::ast::GenericArg::Positional(t1), crate::ast::GenericArg::Positional(t2)) => {
                type_eq_ignoring_spans(t1, t2, exists_params, symbols)
            }
            (crate::ast::GenericArg::Named(n1, t1), crate::ast::GenericArg::Named(n2, t2)) => {
                n1 == n2 && type_eq_ignoring_spans(t1, t2, exists_params, symbols)
            }
            (crate::ast::GenericArg::Const(ac1), crate::ast::GenericArg::Const(ac2)) => {
                const_expr_eq_ignoring_spans(&ac1.value, &ac2.value)
            }
            _ => false,
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

/// A const expression (array size, const-generic argument) is "ground" — a
/// fixed value — only when it is a literal.  An identifier (const parameter)
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

/// Maximum alias-expansion depth for nominal constructor resolution.  Cyclic
/// or pathologically deep aliases must fail closed (return None) rather than
/// recurse forever — the compiler must fail gracefully on invalid inputs.
const MAX_ALIAS_DEPTH: usize = 32;

/// The nominal constructor identity of a PATH: the resolved DefId for
/// symbol-table types, or the primitive name for the builtin primitives (so
/// `Int` and `core::Int` compare equal).  `None` for an opaque `exists`
/// witness, an alias cycle, or an unresolvable path.
fn path_ctor_key(
    path: &[Symbol],
    exists_params: &[Symbol],
    symbols: &SymbolTable,
) -> Option<ConcreteCtor> {
    path_ctor_key_depth(path, exists_params, symbols, 0)
}

fn path_ctor_key_depth(
    path: &[Symbol],
    exists_params: &[Symbol],
    symbols: &SymbolTable,
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
            // contradiction.  Depth-limited so cyclic aliases fail closed.
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
/// see `path_ctor_key`.  `None` for tuples/arrays or an unresolvable path.
pub(crate) fn concrete_ctor_key(
    ct: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable,
) -> Option<ConcreteCtor> {
    concrete_ctor_key_depth(ct, exists_params, symbols, 0)
}

fn concrete_ctor_key_depth(
    ct: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable,
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
/// and its expansion `Int<32>` are the same type.  `None` when the type is
/// not an alias, or an alias cycle / pathological depth (fail closed).
fn normalize_alias<'a>(
    ty: &'a crate::ast::Type,
    symbols: &'a SymbolTable,
) -> Option<&'a crate::ast::Type> {
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
pub(crate) fn type_args_eq_ignoring_spans(
    a: &crate::ast::Type,
    b: &crate::ast::Type,
    exists_params: &[Symbol],
    symbols: &SymbolTable,
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
