use crate::ast::{Attribute, BinOp, Expr, Literal, UnaryOp};
use crate::diagnostics::{DiagCtxt, Diagnostic};
use crate::hir::target::Target;
use crate::sat::Solver;
use crate::symbol::Symbol;
use std::collections::HashMap;

/// A key-value pair representing a cfg condition name and its expected value.
/// For conditions without a value (e.g. `debug`), the value is `None`.
type CfgKey = (String, Option<String>);

/// Evaluate a `@cfg(condition)` attribute against the current target.
///
/// Returns `true` if the condition is met (the item should be compiled),
/// `false` if it should be skipped.
///
/// Supported conditions:
/// - `@cfg(target_os = "linux")` — match target OS
/// - `@cfg(target_arch = "x86_64")` — match target architecture
/// - `@cfg(feature = "logging")` — match a user-defined feature flag
/// - `@cfg(debug)` — true in debug builds
/// - `@cfg(all(...))` — all conditions must be true
/// - `@cfg(any(...))` — any condition must be true
/// - `@cfg(not(...))` — negate a condition
pub fn eval_cfg(
    attr: &Attribute,
    target: &Target,
    features: &[String],
    debug: bool,
    diag: &mut DiagCtxt,
) -> bool {
    // Find the first `@cfg(...)` attribute.
    if !attr.name.eq_str("cfg") {
        return true; // Not a cfg attribute — always include.
    }

    // @cfg with no arguments is always true.
    if attr.args.is_empty() && attr.named_args.is_empty() {
        return true;
    }

    // @cfg with named args like `@cfg(target_os = "linux")`
    let mut cfg_met = true;
    if !attr.named_args.is_empty() {
        let mut warned_keys = std::collections::HashSet::new();
        for (key, value) in &attr.named_args {
            let key_str = key.as_str();
            let val_str = match value {
                Expr::Literal(Literal::String(s), _) => s.clone(),
                _ => {
                    // Non-string literal (e.g. @cfg(target_os = 42)) —
                    // the condition is clearly not met.
                    diag.warn(format!(
                        "cfg key `{}` requires a string value, found non-string literal; \
                         this condition will never match — use a quoted string, \
                         e.g. `@cfg({} = \"value\")`",
                        key_str, key_str,
                    ));
                    cfg_met = false;
                    continue;
                }
            };
            let cfg_key = (key_str, Some(val_str));
            if !eval_cfg_key(&cfg_key, target, features, debug, diag, &mut warned_keys) {
                cfg_met = false;
            }
        }
    }

    // @cfg with positional args like `@cfg(all(...))` or `@cfg(debug)`
    // Also evaluated when named_args are present — both must be met.
    if let Some(expr) = attr.args.first() {
        let mut warned_keys = std::collections::HashSet::new();
        let positional_ok = eval_cfg_expr(expr, target, features, debug, diag, &mut warned_keys);
        cfg_met = cfg_met && positional_ok;
    }

    cfg_met
}

/// Evaluate a cfg expression.
fn eval_cfg_expr(
    expr: &Expr,
    target: &Target,
    features: &[String],
    debug: bool,
    diag: &mut DiagCtxt,
    warned_keys: &mut std::collections::HashSet<String>,
) -> bool {
    match expr {
        Expr::Ident(name, _) => {
            // Simple identifier: `@cfg(debug)`
            let key = (name.as_str(), None);
            eval_cfg_key(&key, target, features, debug, diag, warned_keys)
        }
        Expr::Literal(Literal::Bool(b), _) => *b,
        Expr::Literal(Literal::Int(n), _) => *n != 0,
        Expr::Call { callee, args, .. } => {
            let callee_name = match callee.as_ref() {
                Expr::Ident(name, _) => name.as_str(),
                _ => return false,
            };
            match callee_name.as_str() {
                "all" => args
                    .iter()
                    .all(|a| eval_cfg_expr(a, target, features, debug, diag, warned_keys)),
                "any" => args
                    .iter()
                    .any(|a| eval_cfg_expr(a, target, features, debug, diag, warned_keys)),
                // `not(...)` is handled by the parser as `Expr::UnaryOp { op: Not }`,
                // NOT as `Expr::Call`.  The `not` identifier is recognized during
                // prefix-expression parsing (see `parse_prefix` in parser.rs) and
                // produces a `UnaryOp(Not, expr)` node directly — it never becomes
                // a `Call { callee: "not", ... }`.  Therefore this `Call` branch
                // only needs to match `all(...)` and `any(...)`; `not(...)` is
                // handled by the `Expr::UnaryOp { op: Not }` arm below.
                _ => {
                    // If this fires, the parser changed `not` parsing — see comment above.
                    debug_assert!(
                        callee_name.as_str() != "not",
                        "`not` should be parsed as UnaryOp(Not, ...), not Call"
                    );
                    false
                }
            }
        }
        Expr::BinaryOp {
            left, op, right, ..
        } => {
            match op {
                BinOp::Eq => {
                    // `key = "value"` — compare left identifier to right string
                    let key = match left.as_ref() {
                        Expr::Ident(name, _) => name.as_str(),
                        _ => return false,
                    };
                    let val = match right.as_ref() {
                        Expr::Literal(Literal::String(s), _) => s.clone(),
                        _ => return false,
                    };
                    eval_cfg_key(
                        &(key.to_string(), Some(val)),
                        target,
                        features,
                        debug,
                        diag,
                        warned_keys,
                    )
                }
                BinOp::Neq => {
                    let key = match left.as_ref() {
                        Expr::Ident(name, _) => name.as_str(),
                        _ => return false,
                    };
                    let val = match right.as_ref() {
                        Expr::Literal(Literal::String(s), _) => s.clone(),
                        _ => return false,
                    };
                    !eval_cfg_key(
                        &(key.to_string(), Some(val)),
                        target,
                        features,
                        debug,
                        diag,
                        warned_keys,
                    )
                }
                _ => false,
            }
        }
        Expr::UnaryOp { op, expr, .. } => match op {
            UnaryOp::Not => !eval_cfg_expr(expr, target, features, debug, diag, warned_keys),
            _ => false,
        },
        _ => false,
    }
}

/// Evaluate a single cfg key-value pair.
fn eval_cfg_key(
    key: &CfgKey,
    target: &Target,
    features: &[String],
    debug: bool,
    diag: &mut DiagCtxt,
    warned_keys: &mut std::collections::HashSet<String>,
) -> bool {
    let (name, expected_val) = key;

    match name.as_str() {
        "target_os" => {
            if let Some(expected) = expected_val {
                target.spec.os == *expected
            } else {
                false // target_os requires a value
            }
        }
        "target_arch" => {
            if let Some(expected) = expected_val {
                target.spec.arch == *expected
            } else {
                false
            }
        }
        "target_abi" => {
            if let Some(expected) = expected_val {
                target.spec.abi == *expected
            } else {
                false
            }
        }
        "target_endian" => {
            if let Some(expected) = expected_val {
                let endian_str = match target.spec.endian {
                    crate::hir::target::spec::Endian::Little => "little",
                    crate::hir::target::spec::Endian::Big => "big",
                };
                endian_str == expected.as_str()
            } else {
                false
            }
        }
        "target_pointer_width" => {
            if let Some(expected) = expected_val {
                *expected == target.spec.pointer_width.to_string()
            } else {
                false
            }
        }
        "feature" => {
            if let Some(expected) = expected_val {
                features.iter().any(|f| f == expected)
            } else {
                false
            }
        }
        "debug" => {
            // debug is a boolean flag, not a key=value pair.
            // @cfg(debug) works, but @cfg(debug = "true") does not.
            if let Some(val) = expected_val {
                // Warn the user so misuse doesn't silently skip items.
                diag.warn(format!(
                    "cfg key `debug` is a boolean flag and does not take a value; \
                     `@cfg(debug = \"{}\")` will never match — use `@cfg(debug)` instead",
                    val,
                ));
                false
            } else {
                debug
            }
        }
        _ => {
            // Unknown cfg key — warn the user so misspellings don't
            // silently skip items.  Only warn once per attribute to avoid
            // noise when multiple `@cfg` attributes on different items
            // reference the same unknown key.
            if warned_keys.insert(name.clone()) {
                diag.warn(format!(
                    "unknown cfg key `{}` — this condition will never be met; \
                     check for typos or use a known key (target_os, target_arch, \
                     target_abi, target_endian, target_pointer_width, feature, debug, etc.)",
                    name,
                ));
            }
            false
        }
    }
}

/// Check if a `@cfg` condition is satisfiable (there exists at least one
/// configuration under which it could be true), for strict mode.
///
/// NOTE: This only checks self-consistency (satisfiability), not whether the
/// condition is actually met on the current target.  It does NOT take a
/// `Target` parameter — it merely verifies that the user's cfg expression
/// is not contradictory (e.g. `all(target_os = "linux", target_os = "windows")`).
///
/// Uses the DPLL SAT solver (`Solver`) to check satisfiability:
/// 1. Each `key = "value"` pair gets a boolean variable
/// 2. Mutual exclusion constraints: at most one value per key
/// 3. Target architecture axioms: `arch → pointer_width`
/// 4. The cfg expression is converted to CNF clauses
/// 5. `solver.solve()` determines satisfiability
pub fn is_provably_reachable(attr: &Attribute, strict_mode: bool, diag: &mut DiagCtxt) -> bool {
    if !attr.name.eq_str("cfg") {
        return true;
    }

    if attr.args.is_empty() && attr.named_args.is_empty() {
        return true;
    }

    let mut solver = Solver::new();
    let mut var_map: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut var_counter: i32 = 1;

    // Get or create a variable for a key=value pair.
    let mut var_for = |key: &str, val: &str| -> i32 {
        let name = format!("cfg_{}_{}", key, val);
        let idx = var_map.len() as i32 + 1;
        *var_map.entry(name).or_insert_with(|| {
            let id = var_counter;
            var_counter += 1;
            id
        })
    };

    // Process named args: @cfg(target_os = "linux", ...)
    // Also populate var_keyvals for mutual exclusion (duplicate keys).
    let mut var_keyvals: std::collections::HashMap<i32, (String, String)> =
        std::collections::HashMap::new();
    for (key, value) in &attr.named_args {
        let key_str = key.as_str();
        if let Expr::Literal(Literal::String(s), _) = value {
            let lit = var_for(key_str.as_str(), s.as_str());
            var_keyvals.insert(lit, (key_str.to_string(), s.clone()));
            solver.add_unit(lit);
        }
    }

    // Process positional args: @cfg(all(...)) or @cfg(debug)
    if let Some(expr) = attr.args.first() {
        // Convert the expression to clauses and add to solver.
        let mut limit_exceeded = false;
        let clauses = build_clauses(
            expr,
            &mut var_map,
            &mut var_counter,
            &mut var_keyvals,
            &mut limit_exceeded,
        );
        if limit_exceeded {
            diag.push(
                Diagnostic::warning(
                    "@cfg condition is too complex for precise satisfiability checking — \
                     some constraints were skipped; the result may be a false positive",
                )
                .with_code_str("W091"),
            );
            if strict_mode {
                return false;
            }
        }
        for clause in clauses {
            solver.add_clause(&clause);
        }

        // Register all variables created by build_clauses with the solver,
        // so the solver's assignment vector is large enough for solve().
        // build_clauses uses `var_counter` for variable IDs, which is
        // independent from solver.new_var() — if we don't sync them,
        // solver.solve() will panic with index out of bounds.
        while (solver.new_var()) < var_counter as usize {}

        // Add mutual exclusion: at most one value per key group.
        let mut key_groups: std::collections::HashMap<String, Vec<i32>> =
            std::collections::HashMap::new();
        for (id, (key, _val)) in &var_keyvals {
            key_groups.entry(key.clone()).or_default().push(*id);
        }
        for (_key, lits) in &key_groups {
            if lits.len() > 1 {
                solver.add_at_most_one(lits);
            }
        }
    }

    // Add target architecture axioms: arch → pointer_width.
    if let Some(arch) = attr
        .named_args
        .iter()
        .find(|(k, _)| k.eq_str("target_arch"))
    {
        if let Expr::Literal(Literal::String(s), _) = &arch.1 {
            let arch_name = format!("cfg_target_arch_{}", s);
            if let Some(&arch_lit) = var_map.get(&arch_name) {
                // Look up the pointer width from target specs instead of
                // hardcoding arch→pointer_width mappings here.
                if let Some(width) = crate::hir::target::Target::arch_pointer_width(s.as_str()) {
                    let width_name = format!("cfg_target_pointer_width_{}", width);
                    if let Some(&width_lit) = var_map.get(&width_name) {
                        solver.add_implies(arch_lit, width_lit);
                    }
                }
            }
        }
    }

    solver.solve().is_some()
}

/// Build a list of CNF clauses from a cfg expression.
/// Variable indices are resolved via `var_map`.
fn build_clauses(
    expr: &Expr,
    var_map: &mut std::collections::HashMap<String, i32>,
    var_counter: &mut i32,
    var_keyvals: &mut std::collections::HashMap<i32, (String, String)>,
    limit_exceeded: &mut bool,
) -> Vec<Vec<i32>> {
    let mut clauses = Vec::new();
    build_clauses_inner(
        expr,
        var_map,
        var_counter,
        &mut clauses,
        false,
        var_keyvals,
        limit_exceeded,
    );
    clauses
}

fn build_clauses_inner(
    expr: &Expr,
    var_map: &mut std::collections::HashMap<String, i32>,
    var_counter: &mut i32,
    clauses: &mut Vec<Vec<i32>>,
    negated: bool,
    var_keyvals: &mut std::collections::HashMap<i32, (String, String)>,
    limit_exceeded: &mut bool,
) {
    match expr {
        Expr::Call { callee, args, .. } => {
            if let Expr::Ident(name, _) = callee.as_ref() {
                let name_s = name.as_str();
                match name_s.as_str() {
                    "all" => {
                        // all(a, b) → a ∧ b
                        if negated {
                            // ¬(a ∧ b) ≡ ¬a ∨ ¬b
                            // In CNF, each ¬arg_i is itself a set of clauses.
                            // The disjunction of clause sets is their cross-product:
                            //   (x1 ∧ x2) ∨ (y1)  =  (x1 ∨ y1) ∧ (x2 ∨ y1)
                            // So we collect each arg's negated clauses, then
                            // cross-product them into the final clause list.
                            let mut arg_clause_sets: Vec<Vec<Vec<i32>>> = Vec::new();
                            for arg in args {
                                let mut sub_clauses = Vec::new();
                                build_clauses_inner(
                                    arg,
                                    var_map,
                                    var_counter,
                                    &mut sub_clauses,
                                    true,
                                    var_keyvals,
                                    limit_exceeded,
                                );
                                if sub_clauses.is_empty() {
                                    // Expression has no representation in SAT
                                    // (e.g. unsupported pattern). Skip it rather
                                    // than silently producing incomplete clauses.
                                    continue;
                                }
                                arg_clause_sets.push(sub_clauses);
                            }
                            if arg_clause_sets.is_empty() {
                                // All args empty → ¬all() ≡ true → skip
                            } else {
                                // Cross-product of all arg clause sets.
                                // Bound the result to prevent exponential blowup
                                // from deeply nested `not(all(...))` conditions.
                                const CLAUSE_PRODUCT_LIMIT: usize = 1024;
                                let mut merged: Vec<Vec<i32>> = vec![Vec::new()];
                                for cl_set in &arg_clause_sets {
                                    // Check before cross-product: would this step
                                    // exceed the limit?
                                    if merged.len().saturating_mul(cl_set.len())
                                        > CLAUSE_PRODUCT_LIMIT
                                    {
                                        // Too many clauses — skip this arg's
                                        // contribution rather than discarding
                                        // all previously processed constraints.
                                        // The SAT result may be a false positive
                                        // (reports reachable when it isn't), but
                                        // only for this particular arg; earlier
                                        // args still contribute their constraints.
                                        //
                                        // Why not Tseitin transformation?
                                        // Each ¬argᵢ is already a set of CNF
                                        // clauses (conjunction of disjunctions).
                                        // Encoding "all clauses in set S are
                                        // false" as a single Tseitin variable
                                        // requires expressing a conjunction
                                        // (¬clause₁ ∧ ¬clause₂ ∧ …) — but each
                                        // ¬clauseⱼ = ¬(l₁ ∨ l₂ ∨ …) = l₁ ∧ l₂ ∧ …
                                        // is itself a conjunction of literals.
                                        // This bottoms out as a DNF explosion
                                        // that defeats the purpose. Tseitin
                                        // works best before CNF conversion,
                                        // not on already‑CNF sub‑formulas.
                                        *limit_exceeded = true;
                                        continue;
                                    }
                                    let mut next = Vec::with_capacity(
                                        merged.len().saturating_mul(cl_set.len()),
                                    );
                                    for existing in &merged {
                                        for new_cl in cl_set {
                                            let mut combined = existing.clone();
                                            combined.extend_from_slice(new_cl);
                                            next.push(combined);
                                        }
                                    }
                                    merged = next;
                                }
                                clauses.extend(merged);
                            }
                        } else {
                            for arg in args {
                                build_clauses_inner(
                                    arg,
                                    var_map,
                                    var_counter,
                                    clauses,
                                    false,
                                    var_keyvals,
                                    limit_exceeded,
                                );
                            }
                        }
                    }
                    "any" => {
                        // any(a, b) → a ∨ b
                        if negated {
                            // ¬(a ∨ b) ≡ ¬a ∧ ¬b
                            for arg in args {
                                build_clauses_inner(
                                    arg,
                                    var_map,
                                    var_counter,
                                    clauses,
                                    true,
                                    var_keyvals,
                                    limit_exceeded,
                                );
                            }
                        } else {
                            let mut clause = Vec::new();
                            for arg in args {
                                push_literals(
                                    arg,
                                    var_map,
                                    var_counter,
                                    &mut clause,
                                    false,
                                    var_keyvals,
                                );
                            }
                            if !clause.is_empty() {
                                clauses.push(clause);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Expr::BinaryOp {
            left, op, right, ..
        } if *op == BinOp::Eq => {
            // key = "value" → unit clause
            if let (Expr::Ident(key, _), Expr::Literal(Literal::String(val), _)) =
                (left.as_ref(), right.as_ref())
            {
                let name = format!("cfg_{}_{}", key.as_str(), val);
                let lit = *var_map.entry(name).or_insert_with(|| {
                    let id = *var_counter;
                    *var_counter += 1;
                    id
                });
                var_keyvals
                    .entry(lit)
                    .or_insert_with(|| (key.as_str().to_string(), val.clone()));
                if negated {
                    clauses.push(vec![-lit]);
                } else {
                    clauses.push(vec![lit]);
                }
            }
        }
        Expr::BinaryOp {
            left, op, right, ..
        } if *op == BinOp::Neq => {
            // key != "value" → ¬variable
            if let (Expr::Ident(key, _), Expr::Literal(Literal::String(val), _)) =
                (left.as_ref(), right.as_ref())
            {
                let name = format!("cfg_{}_{}", key.as_str(), val);
                let lit = *var_map.entry(name).or_insert_with(|| {
                    let id = *var_counter;
                    *var_counter += 1;
                    id
                });
                var_keyvals
                    .entry(lit)
                    .or_insert_with(|| (key.as_str().to_string(), val.clone()));
                if negated {
                    clauses.push(vec![lit]);
                } else {
                    clauses.push(vec![-lit]);
                }
            }
        }
        Expr::UnaryOp { op, expr, .. } if *op == UnaryOp::Not => {
            // not(expr) — flip negation.
            build_clauses_inner(
                expr,
                var_map,
                var_counter,
                clauses,
                !negated,
                var_keyvals,
                limit_exceeded,
            );
        }
        _ => {}
    }
}

/// Push literals for an expression into a clause (for `any()`).
fn push_literals(
    expr: &Expr,
    var_map: &mut std::collections::HashMap<String, i32>,
    var_counter: &mut i32,
    clause: &mut Vec<i32>,
    negated: bool,
    var_keyvals: &mut std::collections::HashMap<i32, (String, String)>,
) {
    match expr {
        Expr::BinaryOp {
            left, op, right, ..
        } if *op == BinOp::Eq => {
            if let (Expr::Ident(key, _), Expr::Literal(Literal::String(val), _)) =
                (left.as_ref(), right.as_ref())
            {
                let name = format!("cfg_{}_{}", key.as_str(), val);
                let lit = *var_map.entry(name).or_insert_with(|| {
                    let id = *var_counter;
                    *var_counter += 1;
                    id
                });
                var_keyvals
                    .entry(lit)
                    .or_insert_with(|| (key.as_str().to_string(), val.clone()));
                if negated {
                    clause.push(-lit);
                } else {
                    clause.push(lit);
                }
            }
        }
        Expr::UnaryOp { op, expr, .. } if *op == UnaryOp::Not => {
            push_literals(expr, var_map, var_counter, clause, !negated, var_keyvals);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::hir::target::Target;

    fn make_target() -> Target {
        Target::builtin("x86_64-linux-gnu").expect("builtin target")
    }

    fn make_cfg_attr(named_args: Vec<(&str, &str)>) -> Attribute {
        Attribute {
            name: Symbol::intern("cfg"),
            args: Vec::new(),
            named_args: named_args
                .into_iter()
                .map(|(k, v)| {
                    (
                        Symbol::intern(k),
                        Expr::Literal(Literal::String(v.into()), Span::new(0, 0)),
                    )
                })
                .collect(),
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn test_cfg_target_os_match() {
        let target = make_target();
        let attr = make_cfg_attr(vec![("target_os", "linux")]);
        assert!(eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_target_os_no_match() {
        let target = make_target();
        let attr = make_cfg_attr(vec![("target_os", "windows")]);
        assert!(!eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_target_arch_match() {
        let target = make_target();
        let attr = make_cfg_attr(vec![("target_arch", "x86_64")]);
        assert!(eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_feature() {
        let target = make_target();
        let attr = make_cfg_attr(vec![("feature", "logging")]);
        assert!(eval_cfg(
            &attr,
            &target,
            &["logging".into()],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
        assert!(!eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_debug() {
        let target = make_target();
        let attr = make_cfg_attr(vec![("debug", "true")]);
        // debug is a special case — it's a boolean flag, not a key=value
        // Actually, `debug` is typically used as a bare identifier, not key=value.
        // This test verifies the key=value form doesn't work for debug.
        assert!(!eval_cfg(
            &attr,
            &target,
            &[],
            true,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_debug_bare_identifier_true() {
        // @cfg(debug) when debug=true — should be true
        let target = make_target();
        let attr = Attribute {
            name: Symbol::intern("cfg"),
            args: vec![Expr::Ident(Symbol::intern("debug"), Span::new(0, 0))],
            named_args: Vec::new(),
            span: Span::new(0, 0),
        };
        assert!(eval_cfg(
            &attr,
            &target,
            &[],
            true,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_cfg_debug_bare_identifier_false() {
        // @cfg(debug) when debug=false — should be false
        let target = make_target();
        let attr = Attribute {
            name: Symbol::intern("cfg"),
            args: vec![Expr::Ident(Symbol::intern("debug"), Span::new(0, 0))],
            named_args: Vec::new(),
            span: Span::new(0, 0),
        };
        assert!(!eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    fn test_non_cfg_attr() {
        let target = make_target();
        let attr = Attribute {
            name: Symbol::intern("deprecated"),
            args: Vec::new(),
            named_args: Vec::new(),
            span: Span::new(0, 0),
        };
        assert!(eval_cfg(
            &attr,
            &target,
            &[],
            false,
            &mut crate::diagnostics::DiagCtxt::new()
        ));
    }

    #[test]
    /// Verify that two values for the same key containing underscores
    /// are correctly grouped for mutual exclusion.
    ///
    /// Old bug: `rfind('_')` on `cfg_target_arch_x86_64` split at the
    /// wrong underscore, producing key = "target_arch_x86" instead of
    /// "target_arch". This broke the at-most-one constraint, allowing
    /// contradictory `target_arch` values to both be true.
    ///
    /// This test constructs `all(target_arch = "x86_64", target_arch = "aarch64")`
    /// and verifies the SAT solver correctly reports it as unsatisfiable,
    /// because `x86_64` and `aarch64` are mutually exclusive.
    fn test_cfg_mutual_exclusion_with_underscores() {
        let expr = Expr::Call {
            callee: Box::new(Expr::Ident(Symbol::intern("all"), Span::new(0, 0))),
            args: vec![
                Expr::BinaryOp {
                    left: Box::new(Expr::Ident(Symbol::intern("target_arch"), Span::new(0, 0))),
                    op: BinOp::Eq,
                    right: Box::new(Expr::Literal(
                        Literal::String("x86_64".into()),
                        Span::new(0, 0),
                    )),
                    span: Span::new(0, 0),
                },
                Expr::BinaryOp {
                    left: Box::new(Expr::Ident(Symbol::intern("target_arch"), Span::new(0, 0))),
                    op: BinOp::Eq,
                    right: Box::new(Expr::Literal(
                        Literal::String("aarch64".into()),
                        Span::new(0, 0),
                    )),
                    span: Span::new(0, 0),
                },
            ],
            comptime: false,
            span: Span::new(0, 0),
        };

        let mut var_map = std::collections::HashMap::new();
        let mut var_counter = 1;
        let mut var_keyvals = std::collections::HashMap::new();

        let mut solver = crate::sat::Solver::new();
        let clauses = build_clauses(
            &expr,
            &mut var_map,
            &mut var_counter,
            &mut var_keyvals,
            &mut false,
        );
        assert_eq!(
            clauses.len(),
            2,
            "each target_arch = value should produce one unit clause"
        );

        // Verify key grouping: both variables map to "target_arch"
        // (not "target_arch_x86" or "target_arch_aarch").
        let mut key_groups: std::collections::HashMap<String, Vec<i32>> =
            std::collections::HashMap::new();
        for (&id, (key, _)) in &var_keyvals {
            key_groups.entry(key.clone()).or_default().push(id);
        }
        let group = key_groups
            .get("target_arch")
            .expect("both variables should be grouped under key 'target_arch'");
        assert_eq!(
            group.len(),
            2,
            "both target_arch variables must be in the SAME group for at-most-one to work"
        );

        // Register all variables with the solver before adding clauses.
        let max_var = var_map.values().copied().max().unwrap_or(0);
        for _ in 0..max_var {
            solver.new_var();
        }
        for clause in &clauses {
            solver.add_clause(clause);
        }
        // Add at-most-one constraint (the same one eval_cfg uses).
        if group.len() > 1 {
            solver.add_at_most_one(group);
        }
        assert!(
            solver.solve().is_none(),
            "all(target_arch = \"x86_64\", target_arch = \"aarch64\") \
             should be UNSATISFIABLE: x86_64 and aarch64 are mutually exclusive"
        );
    }
}
