//! Validation of `locales/*.ftl` against the English source of truth and
//! per-locale translation coverage.

use std::collections::{BTreeMap, BTreeSet};
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
    let en: BTreeMap<&str, BTreeSet<String>> = messages(&en_res);

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
        let locale: BTreeMap<&str, BTreeSet<String>> = messages(&res);
        let locale_name = file.trim_end_matches(".ftl");

        for (slug, vars) in &locale {
            match en.get(slug) {
                None => errors.push(format!(
                    "locales/{file}: orphan message {slug:?} (not defined in en.ftl)"
                )),
                Some(en_vars) if en_vars != vars => {
                    let missing: Vec<String> =
                        en_vars.difference(vars).map(|v| format!("${v}")).collect();
                    let extra: Vec<String> =
                        vars.difference(en_vars).map(|v| format!("${v}")).collect();
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
fn messages<'a>(res: &'a ast::Resource<&'a str>) -> BTreeMap<&'a str, BTreeSet<String>> {
    let mut out = BTreeMap::new();
    for entry in &res.body {
        if let ast::Entry::Message(msg) = entry {
            let mut vars = BTreeSet::new();
            if let Some(value) = &msg.value {
                pattern_vars(value, &mut vars);
            }
            for attr in &msg.attributes {
                pattern_vars(&attr.value, &mut vars);
            }
            out.insert(msg.id.name, vars);
        }
    }
    out
}

fn pattern_vars(pattern: &ast::Pattern<&str>, out: &mut BTreeSet<String>) {
    for element in &pattern.elements {
        if let ast::PatternElement::Placeable { expression } = element {
            expr_vars(expression, out);
        }
    }
}

fn expr_vars(expr: &ast::Expression<&str>, out: &mut BTreeSet<String>) {
    match expr {
        ast::Expression::Select { selector, variants } => {
            inline_vars(selector, out);
            for variant in variants {
                pattern_vars(&variant.value, out);
            }
        }
        ast::Expression::Inline(inline) => inline_vars(inline, out),
    }
}

fn inline_vars(inline: &ast::InlineExpression<&str>, out: &mut BTreeSet<String>) {
    match inline {
        ast::InlineExpression::VariableReference { id } => {
            out.insert(id.name.to_string());
        }
        ast::InlineExpression::Placeable { expression } => expr_vars(expression, out),
        ast::InlineExpression::FunctionReference { arguments, .. } => call_vars(arguments, out),
        ast::InlineExpression::TermReference {
            arguments: Some(args),
            ..
        } => call_vars(args, out),
        _ => {}
    }
}

fn call_vars(args: &ast::CallArguments<&str>, out: &mut BTreeSet<String>) {
    for arg in &args.positional {
        inline_vars(arg, out);
    }
    for arg in &args.named {
        inline_vars(&arg.value, out);
    }
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

    let mut rs = String::from(
        "// @generated by build.rs from locales/en.ftl — DO NOT EDIT.\n\
         // Edit that file instead; this module is regenerated on every build.\n\
         use crate::diagnostics::i18n::{render, TrValue};\n\
         use std::borrow::Cow;\n\n",
    );

    for (slug, vars) in &msgs {
        let fn_name = slug.replace('-', "_");
        let params: Vec<&String> = vars.iter().collect();
        if params.is_empty() {
            let _ = std::fmt::Write::write_fmt(
                &mut rs,
                format_args!(
                    "pub fn {fn_name}() -> Cow<'static, str> {{\n    \
                     render({slug:?}, &[])\n}}\n\n",
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
        "final", "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try",
    ];
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}
