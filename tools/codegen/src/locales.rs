//! Validation of `locales/*.ftl` against the English source of truth and
//! per-locale translation coverage.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use fluent_syntax::ast;

/// Translation coverage for one locale file: how many `en.ftl` messages
/// this locale defines (`present`) out of the total in `en.ftl`.
pub struct CoverageRow {
    pub locale: String,
    pub present: usize,
    pub total: usize,
}

/// Result of validating the locales directory.
///
/// Hard `errors`: parse failures, placeholder mismatches against `en.ftl`
/// (both missing and extra variables — the runtime loader silently falls
/// back to raw text on extra variables), and orphan messages that `en.ftl`
/// does not define.  A message merely *missing* from a translation is not
/// an error: it shows up in `coverage` instead.
pub struct Locales {
    pub errors: Vec<String>,
    pub coverage: Vec<CoverageRow>,
}

/// Validate every `*.ftl` in `dir`.  `en.ftl` is the source of truth.
pub fn validate_locales(dir: &Path) -> Locales {
    let mut errors = Vec::new();

    let en_src = match fs::read_to_string(dir.join("en.ftl")) {
        Ok(s) => s,
        Err(e) => {
            errors.push(format!("locales/en.ftl: {e}"));
            return Locales {
                errors,
                coverage: Vec::new(),
            };
        }
    };
    let (en_res, en_parse) = parse(&en_src);
    push_parse_errors(&mut errors, "en.ftl", &en_parse);
    let en: BTreeMap<&str, Vec<String>> = messages(&en_res);

    // #13: `messages()` only extracts variables, so `A = { B }` previously
    // passed even when `B` did not exist (Fluent then renders the raw id at
    // runtime).  Check every message's references: each referrer is visited,
    // so a missing node anywhere in the reference graph is reported with the
    // slug that points at it (which is the transitive closure for existence).
    for (from, to) in message_refs(&en_res) {
        if !en.contains_key(to) {
            errors.push(format!(
                "locales/en.ftl: {from}: message reference to undefined slug {to:?}"
            ));
        }
    }

    let mut coverage = vec![CoverageRow {
        locale: "en".to_string(),
        present: en.len(),
        total: en.len(),
    }];

    let mut others: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|f| {
            let name = f.file_name().to_str()?.to_string();
            (name.ends_with(".ftl") && name != "en.ftl").then_some(name)
        })
        .collect();
    others.sort();

    for file in others {
        let Ok(src) = fs::read_to_string(dir.join(&file)) else {
            errors.push(format!("locales/{file}: cannot read file"));
            continue;
        };
        let (res, parse_errors) = parse(&src);
        push_parse_errors(&mut errors, &file, &parse_errors);
        let locale: BTreeMap<&str, Vec<String>> = messages(&res);
        let locale_name = file.trim_end_matches(".ftl");

        for (slug, vars) in &locale {
            match en.get(slug) {
                None => errors.push(format!(
                    "locales/{file}: orphan message {slug:?} (not defined in en.ftl)"
                )),
                Some(en_vars) if {
                    // Order differs by design (each locale's first-appearance
                    // order); the placeholder contract is set equality.
                    let en_set: BTreeSet<&str> = en_vars.iter().map(String::as_str).collect();
                    let loc_set: BTreeSet<&str> = vars.iter().map(String::as_str).collect();
                    en_set != loc_set
                } =>
                {
                    let en_set: BTreeSet<&str> = en_vars.iter().map(String::as_str).collect();
                    let loc_set: BTreeSet<&str> = vars.iter().map(String::as_str).collect();
                    let missing: Vec<String> = en_set
                        .difference(&loc_set)
                        .map(|v| format!("${v}"))
                        .collect();
                    let extra: Vec<String> = loc_set
                        .difference(&en_set)
                        .map(|v| format!("${v}"))
                        .collect();
                    let mut parts = Vec::new();
                    if !missing.is_empty() {
                        parts.push(format!("missing {}", missing.join(", ")));
                    }
                    if !extra.is_empty() {
                        parts.push(format!("extra {}", extra.join(", ")));
                    }
                    errors.push(format!(
                        "locales/{file}: {slug}: placeholder mismatch vs en.ftl ({})",
                        parts.join("; ")
                    ));
                }
                Some(_) => {}
            }
        }

        let present = locale.keys().filter(|k| en.contains_key(**k)).count();
        coverage.push(CoverageRow {
            locale: locale_name.to_string(),
            present,
            total: en.len(),
        });
    }

    Locales { errors, coverage }
}

fn parse(src: &str) -> (ast::Resource<&str>, Vec<fluent_syntax::parser::ParserError>) {
    match fluent_syntax::parser::parse(src) {
        Ok(res) => (res, Vec::new()),
        Err((res, errs)) => (res, errs),
    }
}

fn push_parse_errors(
    errors: &mut Vec<String>,
    file: &str,
    parse_errors: &[fluent_syntax::parser::ParserError],
) {
    const MAX_SHOWN: usize = 5;
    for err in parse_errors.iter().take(MAX_SHOWN) {
        errors.push(format!(
            "locales/{file}: parse error at byte {}: {err}",
            err.pos.start
        ));
    }
    if parse_errors.len() > MAX_SHOWN {
        errors.push(format!(
            "locales/{file}: ... and {} more parse errors",
            parse_errors.len() - MAX_SHOWN
        ));
    }
}

/// Message slug → set of `$variable` placeholders (from message values and
/// attributes; terms and function names are not placeholders).
fn messages<'a>(res: &'a ast::Resource<&'a str>) -> BTreeMap<&'a str, Vec<String>> {
    let mut out = BTreeMap::new();
    for entry in &res.body {
        if let ast::Entry::Message(msg) = entry {
            // First-appearance order, not `BTreeSet` order: callers of the
            // generated typed functions pass arguments positionally, so the
            // parameter order must match the FTL pattern (and the `tr!` docs).
            let mut vars = Vec::new();
            let mut seen = HashSet::new();
            if let Some(value) = &msg.value {
                pattern_vars(value, &mut vars, &mut seen);
            }
            for attr in &msg.attributes {
                pattern_vars(&attr.value, &mut vars, &mut seen);
            }
            out.insert(msg.id.name, vars);
        }
    }
    out
}

fn pattern_vars(
    pattern: &ast::Pattern<&str>,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    for element in &pattern.elements {
        if let ast::PatternElement::Placeable { expression } = element {
            expr_vars(expression, out, seen);
        }
    }
}

fn expr_vars(expr: &ast::Expression<&str>, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    match expr {
        ast::Expression::Select { selector, variants } => {
            inline_vars(selector, out, seen);
            for variant in variants {
                pattern_vars(&variant.value, out, seen);
            }
        }
        ast::Expression::Inline(inline) => inline_vars(inline, out, seen),
    }
}

fn inline_vars(
    inline: &ast::InlineExpression<&str>,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    match inline {
        ast::InlineExpression::VariableReference { id } => {
            let name = id.name.to_string();
            if seen.insert(name.clone()) {
                out.push(name);
            }
        }
        ast::InlineExpression::Placeable { expression } => expr_vars(expression, out, seen),
        ast::InlineExpression::FunctionReference { arguments, .. } => {
            call_vars(arguments, out, seen)
        }
        ast::InlineExpression::TermReference {
            arguments: Some(args),
            ..
        } => call_vars(args, out, seen),
        _ => {}
    }
}

fn call_vars(args: &ast::CallArguments<&str>, out: &mut Vec<String>, seen: &mut HashSet<String>) {
    for arg in &args.positional {
        inline_vars(arg, out, seen);
    }
    for arg in &args.named {
        inline_vars(&arg.value, out, seen);
    }
}

/// Every `$variable` used as a `select` / plural **selector** in `res`.
///
/// These are the keys that must reach Fluent as `FluentValue::Number` — a
/// selector passed as a string silently falls to the `other` CLDR category,
/// so the runtime numeric-key set has to cover them.  Derived from the
/// parsed AST (not a hardcoded list) so adding a plural message with a new
/// selector name cannot drift out of sync.
fn select_selector_keys(res: &ast::Resource<&str>) -> BTreeSet<String> {
    fn walk(pattern: &ast::Pattern<&str>, out: &mut BTreeSet<String>) {
        for element in &pattern.elements {
            if let ast::PatternElement::Placeable { expression } = element {
                selector_vars(expression, out);
            }
        }
    }
    // Selector names only: a select whose selector is a plain `$variable`
    // contributes that name; nested selects inside variants are picked up
    // recursively.  Everything else (functions, terms) is not a runtime
    // argument key.
    fn selector_vars(expr: &ast::Expression<&str>, out: &mut BTreeSet<String>) {
        match expr {
            ast::Expression::Select { selector, variants } => {
                if let ast::InlineExpression::VariableReference { id } = selector {
                    out.insert(id.name.to_string());
                }
                for variant in variants {
                    walk(&variant.value, out);
                }
            }
            // #14: parity with `expr_vars` — a select nested inside a
            // placeable must still contribute its selector key.
            ast::Expression::Inline(ast::InlineExpression::Placeable { expression }) => {
                selector_vars(expression, out);
            }
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    for entry in &res.body {
        if let ast::Entry::Message(msg) = entry {
            if let Some(value) = &msg.value {
                walk(value, &mut out);
            }
            for attr in &msg.attributes {
                walk(&attr.value, &mut out);
            }
        }
    }
    out
}

/// Every `(referrer, referenced)` message-slug pair in `res`.
///
/// Collects `MessageReference`s from message values, attributes, select
/// variants, and call arguments.  Terms and function names are not message
/// slugs and are ignored (term existence is checked by Fluent itself).
fn message_refs<'a>(res: &'a ast::Resource<&'a str>) -> Vec<(&'a str, &'a str)> {
    fn walk_pattern<'a>(
        p: &'a ast::Pattern<&'a str>,
        from: &'a str,
        out: &mut Vec<(&'a str, &'a str)>,
    ) {
        for el in &p.elements {
            if let ast::PatternElement::Placeable { expression } = el {
                walk_expr(expression, from, out);
            }
        }
    }
    fn walk_expr<'a>(
        e: &'a ast::Expression<&'a str>,
        from: &'a str,
        out: &mut Vec<(&'a str, &'a str)>,
    ) {
        match e {
            ast::Expression::Select { selector, variants } => {
                walk_inline(selector, from, out);
                for v in variants {
                    walk_pattern(&v.value, from, out);
                }
            }
            ast::Expression::Inline(i) => walk_inline(i, from, out),
        }
    }
    fn walk_inline<'a>(
        i: &'a ast::InlineExpression<&'a str>,
        from: &'a str,
        out: &mut Vec<(&'a str, &'a str)>,
    ) {
        match i {
            ast::InlineExpression::MessageReference { id, .. } => out.push((from, id.name)),
            ast::InlineExpression::Placeable { expression } => walk_expr(expression, from, out),
            ast::InlineExpression::FunctionReference { arguments, .. } => {
                walk_args(arguments, from, out)
            }
            ast::InlineExpression::TermReference {
                arguments: Some(args),
                ..
            } => walk_args(args, from, out),
            _ => {}
        }
    }
    fn walk_args<'a>(
        args: &'a ast::CallArguments<&'a str>,
        from: &'a str,
        out: &mut Vec<(&'a str, &'a str)>,
    ) {
        for a in &args.positional {
            walk_inline(a, from, out);
        }
        for a in &args.named {
            walk_inline(&a.value, from, out);
        }
    }
    let mut out = Vec::new();
    for entry in &res.body {
        if let ast::Entry::Message(msg) = entry {
            let from = msg.id.name;
            if let Some(v) = &msg.value {
                walk_pattern(v, from, &mut out);
            }
            for attr in &msg.attributes {
                walk_pattern(&attr.value, from, &mut out);
            }
        }
    }
    out
}

/// Generate typed Rust functions for every Fluent slug in `en.ftl`.
///
/// Each function takes exactly the `$placeholders` the slug declares (in
/// order of first appearance), so a `tr!("type-mismatch", expected = x,
/// found = y)` call becomes `fluent::type_mismatch(x, y)` — a wrong key
/// name or missing argument is a **compile error**, not a silent
/// Fluent fallback.
///
/// The generated file is included by `i18n.rs` as `pub mod fluent`.
pub fn generate_fluent_functions(dir: &Path) -> String {
    let en_src = match fs::read_to_string(dir.join("en.ftl")) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let (res, _) = parse(&en_src);
    let msgs = messages(&res);

    // Numeric argument keys, derived from the source of truth: every `$var`
    // used as a select/plural selector in *any* locale (union, not just
    // en.ftl), plus the legacy count names.  Unioning across locales matters:
    // a translation that introduces a differently-named plural selector
    // (e.g. `{ $pluralCount }` in ru.ftl) would otherwise arrive as a
    // `FluentValue::String` and silently fall to the `other` CLDR branch —
    // exactly the failure this mechanism exists to prevent.
    let mut numeric: BTreeSet<String> = select_selector_keys(&res);
    // #3: union across *every* locale, not a hardcoded ru/zh pair.  A new
    // `fr.ftl` introducing a differently-named plural selector must extend
    // the numeric key set just like the existing locales do.
    let mut others: Vec<std::path::PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|f| f.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("ftl")
                && p.file_name().and_then(|n| n.to_str()) != Some("en.ftl")
        })
        .collect();
    others.sort();
    for path in &others {
        if let Ok(src) = fs::read_to_string(path) {
            let (other, _) = parse(&src);
            numeric.extend(select_selector_keys(&other));
        }
    }
    numeric.extend(["count", "n", "num"].map(str::to_string));
    let numeric_keys: Vec<String> = numeric.iter().map(|k| format!("{k:?}")).collect();

    let mut rs = String::from(
        "// @generated by build.rs from locales/en.ftl — DO NOT EDIT.\n\
         // Edit that file instead; this module is regenerated on every build.\n\
         use crate::diagnostics::i18n::{render, TrValue};\n\
         use std::borrow::Cow;\n\n",
    );
    // Bitmap/match lookup instead of a linear scan over a slice: the checker
    // hits this once per argument of every slug message, and the key set is
    // tiny and closed at build time.  Emitting a `match` lets the compiler
    // build a length-dispatched decision tree (no per-call slice walk).
    let match_arms = numeric_keys
        .iter()
        .map(|k| format!("        {k} => true,\n"))
        .collect::<String>();
    let _ = std::fmt::Write::write_fmt(
        &mut rs,
        format_args!(
            "/// Whether `key` must reach Fluent as a number (select / plural\n\
             /// selectors in any locale, plus the legacy count names), so CLDR\n\
             /// plural categories are actually selected.\n\
             #[inline]\n\
             pub(crate) fn is_numeric_arg_key(key: &str) -> bool {{\n\
                 match key {{\n{}\
                     _ => false,\n\
                 }}\n\
             }}\n\n",
            match_arms
        ),
    );

    // Zero-argument slugs get a build-time direct-addressed cache: the slug
    // set is closed (it comes from en.ftl), so we emit a per-slug slot
    // holding one `String` per language.  First call per (slug, lang)
    // formats through the bundle; every later call is one array read and
    // returns `Cow::Borrowed` — zero allocation, zero formatting.  This
    // generalizes the hand-rolled `cached_label!` tables to every slug.
    let zero_arg: Vec<&str> =
        msgs.iter().filter(|(_, v)| v.is_empty()).map(|(s, _)| *s).collect();
    if !zero_arg.is_empty() {
        rs.push_str(
            "static ZERO_ARG_TABLE: [[std::sync::OnceLock<String>; 3]; ",
        );
        rs.push_str(&zero_arg.len().to_string());
        rs.push_str("] = [const { [const { std::sync::OnceLock::new() }; 3] }; ");
        rs.push_str(&zero_arg.len().to_string());
        rs.push_str("];\n\n");
        rs.push_str("pub(crate) fn zero_arg_slug(slug: &str) -> Option<Cow<'static, str>> {\n");
        rs.push_str("    let idx = match slug {\n");
        for (i, s) in zero_arg.iter().enumerate() {
            let _ = std::fmt::Write::write_fmt(&mut rs, format_args!("        {s:?} => {i},\n"));
        }
        rs.push_str("        _ => return None,\n");
        rs.push_str("    };\n");
        rs.push_str(
            // #4: `Lang::index()` is zero-based; the old `as usize - 1`
            // underflowed if `En` ever took the default discriminant 0.
            // #10: per-(slug, lang) `OnceLock` instead of eagerly translating
            // all three languages on the first access to any one of them.
            "    let li = super::Lang::current().index();\n\
             \x20   Some(Cow::Borrowed(\n\
             \x20       ZERO_ARG_TABLE[idx][li]\n\
             \x20           .get_or_init(|| match super::format_slug(super::lang_id_at(li), slug, None) {\n\
             \x20               Some(s) => s,\n\
             \x20               None => slug.to_string(),\n\
             \x20           })\n\
             \x20           .as_str(),\n\
             \x20   ))\n\
             }\n\n",
        );
    }

    for (slug, vars) in &msgs {
        let fn_name = slug.replace('-', "_");
        let params: Vec<&String> = vars.iter().collect();
        if params.is_empty() {
            // #7: route through the build-time zero-argument cache instead of
            // re-formatting through Fluent on every call.
            let _ = std::fmt::Write::write_fmt(
                &mut rs,
                format_args!(
                    "pub fn {fn_name}() -> Cow<'static, str> {{\n    \
                     zero_arg_slug({slug:?}).unwrap_or_else(|| render({slug:?}, &[]))\n}}\n\n",
                ),
            );
        } else {
            // One lifetime parameter per argument: each value keeps its own
            // borrow (a Cow argument stays borrowed), and the generated
            // into_cow() calls meet at a common shorter region inside the
            // pair array.
            let lifetimes: Vec<String> = (0..params.len()).map(|i| format!("'a{i}")).collect();
            let args: Vec<String> = params
                .iter()
                .zip(&lifetimes)
                .map(|(p, lt)| format!("{}: impl TrValue<{lt}>", escape_kw(p)))
                .collect();
            let pairs: Vec<String> = params
                .iter()
                .map(|p| format!("(\"{}\", {}.into_cow())", p, escape_kw(p)))
                .collect();
            let _ = std::fmt::Write::write_fmt(
                &mut rs,
                format_args!(
                    "pub fn {fn_name}<{generics}>({args}) -> Cow<'static, str> {{\n    \
                     render({slug:?}, &[{pairs}])\n}}\n\n",
                    generics = lifetimes.join(", "),
                    args = args.join(", "),
                    pairs = pairs.join(", "),
                ),
            );
        }
    }
    rs
}

/// Append `_` to Rust keywords so they can be used as parameter names.
fn escape_kw(name: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn",
        "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "dyn", "abstract", "become", "box", "do",
        "final", "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try", "gen",
    ];
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}
