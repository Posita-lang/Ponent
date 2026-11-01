//! Validation of `error_codes/codes.toml` (+ referenced `error_codes/*.md`)
//! and generation of the `error_codes_gen.rs` table source.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const CTX: &str = "error_codes/codes.toml";

/// Legal `category` values in `error_codes/codes.toml` → `ErrorCategory`
/// variant name in the generated table.
pub const CATEGORIES: &[(&str, &str)] = &[
    ("parse", "Parse"),
    ("resolution", "Resolution"),
    ("type", "Type"),
    ("contract", "Contract"),
    ("trait", "Trait"),
    ("inference", "Inference"),
    ("internal", "Internal"),
    ("generic", "Generic"),
];

/// Minimum acceptable `error_codes/*.md` size (bytes).  Guards against
/// placeholder stubs like "" or "TODO" sneaking into the registry.
pub const MIN_EXPLAIN_BYTES: usize = 10;

/// One validated `[codes.EXXX]` entry, in registry document order.
pub struct RegistryEntry {
    pub code: String,
    pub title: String,
    /// The `ErrorCategory` variant name (e.g. `"Type"`).
    pub category_variant: String,
    pub explain: String,
}

/// Result of loading the registry.
///
/// `generated` is the full `error_codes_gen.rs` source, or empty when
/// `errors` is non-empty.  `lints` holds duplicate titles and
/// unreferenced `*.md` files (hygiene, not invalid data).
pub struct Registry {
    pub entries: Vec<RegistryEntry>,
    pub errors: Vec<String>,
    pub lints: Vec<String>,
    pub generated: String,
}

impl Registry {
    fn broken(errors: Vec<String>, lints: Vec<String>) -> Self {
        Registry {
            entries: Vec::new(),
            errors,
            lints,
            generated: String::new(),
        }
    }
}

/// Validate `dir` (the `error_codes/` directory) and build the generated
/// `CODE_TABLE` source in document order.
pub fn load_registry(dir: &Path) -> Registry {
    let raw = match fs::read_to_string(dir.join("codes.toml")) {
        Ok(s) => s,
        Err(e) => return Registry::broken(vec![format!("{CTX}: {e}")], Vec::new()),
    };
    let doc: toml::Value = match toml::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return Registry::broken(vec![format!("{CTX}: {e}")], Vec::new()),
    };

    let schema = match doc.get("schema").and_then(toml::Value::as_integer) {
        Some(s) => s,
        None => {
            return Registry::broken(
                vec![format!("{CTX}: `schema = <integer>` missing")],
                Vec::new(),
            );
        }
    };
    if schema != 1 {
        return Registry::broken(
            vec![format!("{CTX}: unsupported schema {schema} (expected 1)")],
            Vec::new(),
        );
    }
    let table = match doc.get("codes").and_then(toml::Value::as_table) {
        Some(t) => t,
        None => {
            return Registry::broken(
                vec![format!("{CTX}: no [codes.<CODE>] sections")],
                Vec::new(),
            );
        }
    };
    if table.is_empty() {
        return Registry::broken(vec![format!("{CTX}: [codes.*] table is empty")], Vec::new());
    }

    let mut errors: Vec<String> = Vec::new();
    let mut lints: Vec<String> = Vec::new();
    let mut entries: Vec<RegistryEntry> = Vec::new();
    let mut titles: HashMap<&str, &str> = HashMap::new();
    let mut referenced: Vec<&str> = Vec::new();

    for (code, meta) in table {
        // Byte-wise so a non-ASCII code (e.g. "é12", 4 bytes) is rejected
        // instead of panicking on a non-char-boundary slice.
        let b = code.as_bytes();
        let ok_format =
            b.len() == 4 && matches!(b[0], b'E' | b'W') && b[1..].iter().all(u8::is_ascii_digit);
        if !ok_format {
            errors.push(format!(
                "{CTX}: invalid code {code:?} (expected ^[EW]\\d{{3}}$)"
            ));
            continue;
        }
        let Some(entry) = meta.as_table() else {
            errors.push(format!("{CTX}: [{code}] must be a table"));
            continue;
        };

        let Some(title) = entry
            .get("title")
            .and_then(toml::Value::as_str)
            .filter(|t| !t.trim().is_empty())
        else {
            errors.push(format!("{CTX}: [{code}] missing/non-string `title`"));
            continue;
        };
        if let Some(prev) = titles.insert(title, code) {
            lints.push(format!(
                "{CTX}: duplicate title {title:?} used by both {prev} and {code}"
            ));
        }

        let Some(explain) = entry.get("explain").and_then(toml::Value::as_str) else {
            errors.push(format!("{CTX}: [{code}] missing/non-string `explain`"));
            continue;
        };
        referenced.push(explain);
        if explain.contains("..")
            || explain.starts_with('/')
            || !explain
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            errors.push(format!("{CTX}: [{code}] unsafe explain path {explain:?}"));
            continue;
        }
        let explain_path = dir.join(explain);
        let bytes = match fs::read(&explain_path) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{CTX}: [{code}] explain file {explain:?}: {e}"));
                continue;
            }
        };
        if bytes.len() < MIN_EXPLAIN_BYTES {
            errors.push(format!(
                "{CTX}: [{code}] explain file {explain:?} is {} bytes (< {MIN_EXPLAIN_BYTES})",
                bytes.len()
            ));
            continue;
        }

        let Some(cat) = entry
            .get("category")
            .and_then(toml::Value::as_str)
            .and_then(|c| CATEGORIES.iter().find(|(k, _)| *k == c))
            .map(|(_, v)| *v)
        else {
            errors.push(format!(
                "{CTX}: [{code}] `category` must be one of parse|resolution|type|contract|trait|inference|internal|generic"
            ));
            continue;
        };

        entries.push(RegistryEntry {
            code: code.clone(),
            title: title.to_string(),
            category_variant: cat.to_string(),
            explain: explain.to_string(),
        });
    }

    // Every `*.md` beside the registry should be reachable from some
    // `[codes.*].explain`; a leftover file means a removed/renamed code.
    let mut orphans: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|f| {
            let name = f.file_name().to_str()?.to_string();
            (name.ends_with(".md") && !referenced.contains(&name.as_str())).then_some(name)
        })
        .collect();
    orphans.sort();
    for orphan in orphans {
        lints.push(format!("{CTX}: unreferenced explain file {orphan:?}"));
    }

    let generated = if errors.is_empty() {
        generate_table(&entries)
    } else {
        String::new()
    };
    Registry {
        entries,
        errors,
        lints,
        generated,
    }
}

/// Render the validated entries as `error_codes_gen.rs` source.  Byte-for-
/// byte identical to what the inline `build.rs` generator produced.
fn generate_table(entries: &[RegistryEntry]) -> String {
    let mut rs = String::from(
        "// @generated by build.rs from error_codes/codes.toml + error_codes/*.md — DO NOT EDIT.\n\
         // Edit those files instead; this file is regenerated on every build.\n\
         /// Lookup table for error/warning code metadata.\n\
         /// Maps code strings to their title, category, and explanation.\n\
         ///\n\
         /// Source of truth: `error_codes/codes.toml` (registry) + per-code\n\
         /// `error_codes/*.md` explanation files; document order follows the\n\
         /// `[codes.*]` sections of the registry.\n",
    );
    rs.push_str("pub(crate) const CODE_TABLE: &[CodeEntry] = &[\n");
    for e in entries {
        let _ = writeln!(rs, "    CodeEntry {{");
        let _ = writeln!(rs, "        code: {:?},", e.code);
        let _ = writeln!(rs, "        title: {:?},", e.title);
        let _ = writeln!(
            rs,
            "        category: ErrorCategory::{},",
            e.category_variant
        );
        let _ = writeln!(
            rs,
            "        explain: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/error_codes/{}\")),",
            e.explain
        );
        let _ = writeln!(rs, "    }},");
    }
    rs.push_str("];\n");

    // Emit one `ErrCode` associated constant per entry so production call
    // sites write `ErrCode::E030` (a typo is a compile error) instead of
    // `"E030"` (a string literal with no typo safety).  The constant is
    // a `&'static str` that `with_code_str` validates at runtime; the
    // compile-time safety comes from the constant name, not from skipping
    // the lookup.
    rs.push_str("\nimpl ErrCode {\n");
    for e in entries.iter() {
        let _ = writeln!(rs, "    pub const {}: &'static str = {:?};", e.code, e.code);
    }
    rs.push_str("}\n");

    rs
}
