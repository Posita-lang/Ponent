//! Validation of `error_codes/codes.toml` (+ referenced `error_codes/*.md`)
//! and generation of the `error_codes_gen.rs` table source.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use swrite::{SWrite, swrite, swriteln};

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
    /// Validated UTF-8 contents of `explain`, cached so the site renderer
    /// does not re-open and re-read every file the registry just read.
    pub content: String,
    /// Codes this one links to in "see also" (validated against the table).
    pub related: Vec<String>,
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
    let mut titles: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut referenced: HashSet<&str> = HashSet::new();

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
        titles.entry(title).or_default().push(code);

        let Some(explain) = entry.get("explain").and_then(toml::Value::as_str) else {
            errors.push(format!("{CTX}: [{code}] missing/non-string `explain`"));
            continue;
        };
        let explain_path = dir.join(explain);
        if explain.is_empty() {
            // #8: `""` passes the byte-class check vacuously and then
            // `dir.join("")` is the directory itself (EISDIR).
            errors.push(format!("{CTX}: [{code}] `explain` must not be empty"));
            continue;
        }
        if explain.contains("..")
            || explain.starts_with('/')
            || !explain
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            errors.push(format!("{CTX}: [{code}] unsafe explain path {explain:?}"));
            continue;
        }
        let bytes = match fs::read(&explain_path) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{CTX}: [{code}] explain file {explain:?}: {e}"));
                continue;
            }
        };
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                errors.push(format!(
                    "{CTX}: [{code}] explain file {explain:?} is not valid UTF-8"
                ));
                continue;
            }
        };
        if content.len() < MIN_EXPLAIN_BYTES {
            errors.push(format!(
                "{CTX}: [{code}] explain file {explain:?} is {} bytes (< {MIN_EXPLAIN_BYTES})",
                content.len()
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

        // `related` is optional; when present it must be an array of
        // shape-valid codes.  Reference existence is checked after the
        // whole table is loaded (a forward reference is legal).
        let related: Vec<String> = match entry.get("related") {
            None => Vec::new(),
            Some(v) => {
                let Some(items) = v.as_array() else {
                    errors.push(format!("{CTX}: [{code}] `related` must be an array"));
                    continue;
                };
                let mut out = Vec::with_capacity(items.len());
                let mut bad = false;
                for item in items {
                    let Some(rel) = item.as_str() else {
                        errors.push(format!("{CTX}: [{code}] `related` entries must be strings"));
                        bad = true;
                        break;
                    };
                    let rb = rel.as_bytes();
                    if !(rb.len() == 4
                        && matches!(rb[0], b'E' | b'W')
                        && rb[1..].iter().all(u8::is_ascii_digit))
                    {
                        errors.push(format!(
                            "{CTX}: [{code}] invalid related code {rel:?} (expected ^[EW]\\d{{3}}$)"
                        ));
                        bad = true;
                        break;
                    }
                    out.push(rel.to_string());
                }
                if bad {
                    continue;
                }
                out
            }
        };

        referenced.insert(explain);
        entries.push(RegistryEntry {
            code: code.clone(),
            title: title.to_string(),
            category_variant: cat.to_string(),
            explain: explain.to_string(),
            content,
            related,
        });
    }

    // Every `related` target must exist in the registry.  Done after the
    // loop so order-independent forward references are fine.
    {
        let known: std::collections::HashSet<&str> =
            entries.iter().map(|e| e.code.as_str()).collect();
        for e in &entries {
            for rel in &e.related {
                if !known.contains(rel.as_str()) {
                    errors.push(format!(
                        "{CTX}: [{}] `related` names unregistered code {rel:?}",
                        e.code
                    ));
                }
            }
        }
    }

    // #15: report every user of a duplicated title in one lint, not just
    // consecutive pairs (`A`, `B`, `C` used to yield `A/B` and `B/C` only).
    let mut dup: Vec<(&str, &Vec<&str>)> = titles
        .iter()
        .filter(|(_, codes)| codes.len() > 1)
        .map(|(title, codes)| (*title, codes))
        .collect();
    dup.sort_by_key(|(title, _)| *title);
    for (title, codes) in dup {
        lints.push(format!(
            "{CTX}: duplicate title {title:?} used by {}",
            codes.join(", ")
        ));
    }

    // Every `*.md` beside the registry should be reachable from some
    // `[codes.*].explain`; a leftover file means a removed/renamed code.
    let mut orphans: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|f| {
            let name = f.file_name().to_str()?.to_string();
            // #16: a *directory* named `foo.md` must not count as an orphan
            // explain file (and must not be read as one by the site renderer).
            let is_file = f.file_type().map(|t| t.is_file()).unwrap_or(false);
            (is_file && name.ends_with(".md") && !referenced.contains(name.as_str()))
                .then_some(name)
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
/// Documentation-URL prefix, shared by the generated payloads and by the
/// runtime builder for unknown spellings ([`crate::diagnostics`] uses the same
/// constant; `doc_url_prefix_matches` pins the agreement).
const DOC_URL_PREFIX: &str = "https://doc.posita-lang.org/error_codes/";

/// FNV-1a over the canonical spelling.
///
/// Must equal `error_code::code_hash` — the runtime precomputes the same value
/// for interned spellings, and `code_inner_hashes_match_code_hash` pins it.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

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
        swriteln!(rs, "    CodeEntry {{");
        swriteln!(rs, "        code: {:?},", e.code);
        let b = e.code.as_bytes();
        let prefix = if b[0] == b'E' { 0 } else { 1 };
        let n = (b[1] - b'0') as usize * 100 + (b[2] - b'0') as usize * 10 + (b[3] - b'0') as usize;
        swriteln!(rs, "        slot: {},", prefix * 1000 + n);
        swriteln!(rs, "        title: {:?},", e.title);
        swriteln!(
            rs,
            "        category: ErrorCategory::{},",
            e.category_variant
        );
        swrite!(rs, "        related: &[");
        for (i, rel) in e.related.iter().enumerate() {
            if i > 0 {
                rs.push_str(", ");
            }
            swrite!(rs, "{rel:?}");
        }
        swriteln!(rs, "],");
        swrite!(rs, "        siblings: &[");
        let mut first = true;
        for other in entries
            .iter()
            .filter(|other| other.category_variant == e.category_variant && other.code != e.code)
        {
            if !first {
                rs.push_str(", ");
            }
            first = false;
            swrite!(rs, "{:?}", other.code);
        }
        swriteln!(rs, "],");
        swriteln!(
            rs,
            "        explain: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/error_codes/{}\")),",
            e.explain
        );
        swriteln!(rs, "    }},");
    }
    rs.push_str("];\n");

    // `ErrCode` is one `&'static ErrCodeInner`, i.e. a single word.
    //
    // Everything a code needs at render time except its explain body is known
    // here: the two spellings, the resolved entry, the `Hash` value, and both
    // documentation-URL forms (plain and OSC 8).  Generating them is what lets
    // `ErrCode::url()`/`url_ansi()` read a field instead of consulting a runtime
    // cache — the "explain port not set" quadrant of those caches is deleted
    // rather than tuned.
    rs.push_str("pub(crate) const CODE_INNER: &[ErrCodeInner] = &[\n");
    for (i, e) in entries.iter().enumerate() {
        swriteln!(rs, "    ErrCodeInner {{");
        swriteln!(rs, "        display: {:?},", e.code);
        swriteln!(rs, "        canonical: {:?},", e.code);
        // The slot is emitted literally (not re-derived through `CODE_INDEX`):
        // `ErrCode` is one `&'static ErrCodeInner`, so storing it here turns
        // `slot()` for a registry code into a single field load instead of an
        // entry -> slot two-hop.
        {
            let b = e.code.as_bytes();
            let prefix = if b[0] == b'E' { 0 } else { 1 };
            let n =
                (b[1] - b'0') as usize * 100 + (b[2] - b'0') as usize * 10 + (b[3] - b'0') as usize;
            swriteln!(rs, "        slot: {},", prefix * 1000 + n);
        }
        swriteln!(rs, "        entry: Some(&CODE_TABLE[{i}]),");
        swriteln!(rs, "        known: true,");
        let doc = format!("{DOC_URL_PREFIX}{}.html", e.code);
        swriteln!(rs, "        doc_url: {doc:?},");
        let ansi = format!("\u{1b}]8;;{doc}\u{1b}\\{}\u{1b}]8;;\u{1b}\\", e.code);
        swriteln!(rs, "        doc_url_ansi: {ansi:?},");

        swriteln!(rs, "    }},");
    }
    rs.push_str("];\n");

    // Generated "see also" data now lives inside each `CodeEntry.related`;
    // no separate lookup table is emitted.

    // Emit one `ErrCode` associated constant per entry so production call
    // sites write `ErrCode::E030` (a typo is a compile error) instead of
    // `"E030"` (a string literal with no typo safety).  The constant is
    // a `&'static str` that `with_code_str` validates at runtime; the
    // compile-time safety comes from the constant name, not from skipping
    // the lookup.
    rs.push_str("\nimpl ErrCode {\n");
    for e in entries.iter() {
        swriteln!(rs, "    pub const {}: &'static str = {:?};", e.code, e.code);
    }
    rs.push_str("}\n");

    // Build-time `known_code_ranges`, so the "valid codes are …" hint is a
    // `&'static str` instead of a first-use sort of ~450 entries plus a heap
    // `String`.  The runtime `format_code_ranges` stays as the *differential
    // oracle*: `known_code_ranges_match_the_runtime_formatter` asserts the two
    // agree, so this duplication cannot drift silently.
    let ranges = format_ranges(entries);
    rs.push_str(
        "\n/// Every registry code, summarised as contiguous numeric ranges.\n\
         ///\n\
         /// Generated: see `tools/codegen/src/codes.rs::format_ranges`, whose output is\n\
         /// pinned against the runtime `diagnostics::format_code_ranges` by a test.\n",
    );
    swriteln!(
        rs,
        "pub(crate) const KNOWN_CODE_RANGES: &str = {:?};",
        ranges
    );

    rs
}

/// Summarise the registry's codes as contiguous numeric ranges.
///
/// Must stay byte-identical to `src/diagnostics/mod.rs::format_code_ranges`
/// (pinned by `known_code_ranges_match_the_runtime_formatter`): the same sort by
/// `(prefix, numeric)`, the same `start–end` grouping for adjacent numerics, and
/// the same `", "` join.  The `–` is U+2013.
fn format_ranges(entries: &[RegistryEntry]) -> String {
    let mut keyed: Vec<(u8, u64, &str)> = entries
        .iter()
        .map(|e| {
            let c = e.code.as_str();
            (
                c.as_bytes().first().copied().unwrap_or(b'?'),
                c.get(1..)
                    .and_then(|s| s.parse::<u32>().ok())
                    .map(u64::from)
                    .unwrap_or(u64::MAX),
                c,
            )
        })
        .collect();
    keyed.sort_unstable();
    let numeric = |c: &str| {
        c.get(1..)
            .and_then(|s| s.parse::<u32>().ok())
            .map(u64::from)
    };
    let mut out = String::with_capacity(keyed.len().saturating_mul(9));
    let mut i = 0;
    while i < keyed.len() {
        let start = keyed[i].2;
        let prefix = start.get(..1).unwrap_or("?");
        let mut end = start;
        while i + 1 < keyed.len() {
            let next = keyed[i + 1].2;
            if next.get(..1).unwrap_or("?") != prefix {
                break;
            }
            match (numeric(end), numeric(next)) {
                (Some(a), Some(b)) if a.checked_add(1) == Some(b) => {
                    end = next;
                    i += 1;
                }
                _ => break,
            }
        }
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(start);
        if start != end {
            out.push('\u{2013}');
            out.push_str(end);
        }
        i += 1;
    }
    out
}
