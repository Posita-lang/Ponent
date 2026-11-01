//! Static HTML renderer for the error-code explain pages (`cargo xtask site`).

use std::fs;
use std::path::Path;

use crate::codes::{CATEGORIES, RegistryEntry, load_registry};

/// Pages written by [`render_site`] (one per code plus the index).
pub struct SiteReport {
    pub pages: usize,
}

/// Render `root`'s registry into `out_dir`: `index.html` (grouped by
/// category) plus `{code}.html` per entry.  Locale validation is the job
/// of `codegen --check`; the site only needs the registry to be valid.
pub fn render_site(root: &Path, out_dir: &Path) -> Result<SiteReport, String> {
    let registry = load_registry(&root.join("error_codes"));
    if !registry.errors.is_empty() {
        return Err(registry.errors.join("\n"));
    }
    fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;

    let mut pages = 0;
    for entry in &registry.entries {
        let md_path = root.join("error_codes").join(&entry.explain);
        let md = fs::read_to_string(&md_path).map_err(|e| format!("{}: {e}", md_path.display()))?;
        let body = markdown_html(&preprocess(&md));
        let page = page_html(entry, &body);
        let dest = out_dir.join(format!("{}.html", entry.code));
        fs::write(&dest, &page).map_err(|e| format!("{}: {e}", dest.display()))?;
        pages += 1;
    }
    let index = index_html(&registry.entries);
    fs::write(out_dir.join("index.html"), &index)
        .map_err(|e| format!("{}: {e}", out_dir.display()))?;
    pages += 1;
    Ok(SiteReport { pages })
}

/// The terminal renderer (`explain_error_code`) treats every line indented
/// by two spaces as a code block; CommonMark only honours four, so runs of
/// two-space-indented lines are rewritten as fenced blocks, dedented by
/// two.  All other lines pass through untouched.
fn preprocess(md: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for line in md.lines() {
        let is_code = line.starts_with("  ") && !line.trim().is_empty();
        if is_code != in_code {
            out.push_str("```\n");
            in_code = is_code;
        }
        if is_code {
            out.push_str(&line[2..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    if in_code {
        out.push_str("```\n");
    }
    out
}

fn markdown_html(md: &str) -> String {
    let parser = pulldown_cmark::Parser::new_ext(md, pulldown_cmark::Options::all());
    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, parser);
    html
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

const STYLE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #16161a; }
code, pre { font-family: ui-monospace, "Cascadia Code", Menlo, monospace; font-size: .92em; }
pre { background: #f4f4f5; border: 1px solid #e4e4e7; border-radius: 6px; padding: .75rem .9rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
code { background: #f4f4f5; border-radius: 4px; padding: .1em .3em; }
header { border-bottom: 1px solid #e4e4e7; margin-bottom: 1.5rem; padding-bottom: .75rem; }
header .cat { color: #71717a; font-size: .9em; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.4rem; margin: .4rem 0 0; }
h2 { font-size: 1.1rem; margin-top: 2rem; }
ul { padding-left: 1.2rem; }
</style>
</head>
<body>
"#;

/// Substitute `{TITLE}` in the shared head.
fn head(title: &str) -> String {
    STYLE.replace("{TITLE}", &escape(title))
}

fn page_html(entry: &RegistryEntry, body: &str) -> String {
    let title = format!("{}: {} — ponent error codes", entry.code, entry.title);
    format!(
        "{}<header><a href=\"index.html\">&larr; all codes</a> &nbsp; \
         <code>{}</code> <span class=\"cat\">({})</span>\
         <h1>{}: {}</h1></header>\n<main>\n{}\n</main>\n</body>\n</html>\n",
        head(&title),
        entry.code,
        entry.category_variant,
        entry.code,
        escape(&entry.title),
        body,
    )
}

fn index_html(entries: &[RegistryEntry]) -> String {
    let mut out = head("ponent error codes");
    out.push_str("<header><h1>ponent error codes</h1></header>\n");
    for (_, variant) in CATEGORIES {
        let group: Vec<&RegistryEntry> = entries
            .iter()
            .filter(|e| e.category_variant == *variant)
            .collect();
        if group.is_empty() {
            continue;
        }
        out.push_str(&format!("<h2>{variant}</h2>\n<ul>\n"));
        for e in group {
            out.push_str(&format!(
                "<li><a href=\"{}.html\"><code>{}</code></a> — {}</li>\n",
                e.code,
                e.code,
                escape(&e.title),
            ));
        }
        out.push_str("</ul>\n");
    }
    out.push_str("</body>\n</html>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::preprocess;

    #[test]
    fn two_space_runs_become_fenced_blocks_dedented_by_two() {
        let md = "Intro text.\n  def f() {\n    x\n  }\n\nTail.\n";
        let out = preprocess(md);
        assert_eq!(out, "Intro text.\n```\ndef f() {\n  x\n}\n```\n\nTail.\n");
    }

    #[test]
    fn indented_bullets_follow_the_terminal_rule() {
        // The terminal renderer highlights these as code, so the site must
        // mirror it rather than reinterpret them as markdown lists.
        let out = preprocess("before\n  - one\n  - two\nafter\n");
        assert_eq!(out, "before\n```\n- one\n- two\n```\nafter\n");
    }
}
