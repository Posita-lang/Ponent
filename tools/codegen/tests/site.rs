//! HTML site rendering tests (feature `site`).

#![cfg(feature = "site")]

mod common;

use common::{Fixture, repo_root};
use ponent_codegen::render_site;

#[test]
fn renders_real_repo() {
    let out = tempfile::tempdir().expect("tempdir");
    let report = render_site(&repo_root(), out.path()).expect("render");
    // 89 code pages + index.
    assert_eq!(report.pages, 90);

    let index = std::fs::read_to_string(out.path().join("index.html")).expect("index");
    assert!(index.contains("href=\"E001.html\""), "index links E001");
    assert!(index.contains("E051"), "retitled code present");

    let page = std::fs::read_to_string(out.path().join("E001.html")).expect("E001 page");
    assert!(page.contains("<h1>E001:"), "page heading");
    assert!(page.contains("</html>"));
    // Two-space-indented example code must arrive as a fenced <pre> block.
    let code_page = std::fs::read_to_string(out.path().join("E036.html")).expect("E036 page");
    assert!(
        code_page.contains("<pre>"),
        "E036's indented example renders as a code block"
    );
}

#[test]
fn titles_are_escaped() {
    let f = Fixture::new();
    f.base();
    f.write(
        "error_codes/codes.toml",
        "schema = 1\n\n[codes.E001]\ntitle = \"A & B <c>\"\ncategory = \"parse\"\nexplain = \"E001.md\"\n",
    );

    let out = tempfile::tempdir().expect("tempdir");
    render_site(f.root(), out.path()).expect("render");

    let page = std::fs::read_to_string(out.path().join("E001.html")).expect("page");
    assert!(page.contains("A &amp; B &lt;c&gt;"), "escaped title");
    assert!(!page.contains("<c>"), "raw angle brackets must not leak");

    let index = std::fs::read_to_string(out.path().join("index.html")).expect("index");
    assert!(index.contains("A &amp; B &lt;c&gt;"));
}

#[test]
fn invalid_registry_fails_the_render() {
    let f = Fixture::new(); // empty fixture: no codes.toml at all
    let out = tempfile::tempdir().expect("tempdir");
    assert!(render_site(f.root(), out.path()).is_err());
}
