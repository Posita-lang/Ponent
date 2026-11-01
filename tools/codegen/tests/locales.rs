//! Fluent locale validation tests (en.ftl = source of truth).

mod common;

use common::Fixture;
use ponent_codegen::validate_locales;

#[test]
fn placeholder_mismatch_is_an_error() {
    let f = Fixture::new();
    f.base();
    f.write("locales/en.ftl", "greet = Hello { $name }!\n");
    f.write("locales/ru.ftl", "greet = Privet { $nami }!\n");

    let out = validate_locales(&f.root().join("locales"));
    assert_eq!(out.errors.len(), 1, "errors: {:#?}", out.errors);
    assert!(
        out.errors[0].contains("locales/ru.ftl: greet: placeholder mismatch vs en.ftl")
            && out.errors[0].contains("missing $name")
            && out.errors[0].contains("extra $nami"),
        "unexpected error: {}",
        out.errors[0]
    );
}

#[test]
fn extra_placeholder_alone_is_still_an_error() {
    // The runtime loader resolves missing variables to `None` and falls
    // back to raw placeholder text — silent breakage, so hard-fail here.
    let f = Fixture::new();
    f.base();
    f.write("locales/en.ftl", "plain = No variables here.\n");
    f.write("locales/zh.ftl", "plain = Placeholder { $oops }.\n");

    let out = validate_locales(&f.root().join("locales"));
    assert_eq!(out.errors.len(), 1, "errors: {:#?}", out.errors);
    assert!(
        out.errors[0].contains("placeholder mismatch") && out.errors[0].contains("extra $oops"),
        "unexpected error: {}",
        out.errors[0]
    );
}

#[test]
fn orphan_slug_is_an_error() {
    let f = Fixture::new();
    f.base();
    f.write("locales/en.ftl", "known = v\n");
    f.write("locales/ru.ftl", "known = v\nrenamed = v\n");

    let out = validate_locales(&f.root().join("locales"));
    assert_eq!(out.errors.len(), 1, "errors: {:#?}", out.errors);
    assert!(
        out.errors[0] == "locales/ru.ftl: orphan message \"renamed\" (not defined in en.ftl)",
        "unexpected error: {}",
        out.errors[0]
    );
}

#[test]
fn missing_slug_only_affects_coverage() {
    let f = Fixture::new();
    f.base();
    f.write("locales/en.ftl", "one = v\ntwo = v\n");
    f.write("locales/ru.ftl", "one = v\n");

    let out = validate_locales(&f.root().join("locales"));
    assert!(out.errors.is_empty(), "errors: {:#?}", out.errors);
    let ru = out
        .coverage
        .iter()
        .find(|r| r.locale == "ru")
        .expect("ru row");
    assert_eq!((ru.present, ru.total), (1, 2));
    let en = out
        .coverage
        .iter()
        .find(|r| r.locale == "en")
        .expect("en row");
    assert_eq!((en.present, en.total), (2, 2));
}

#[test]
fn parse_error_is_an_error() {
    let f = Fixture::new();
    f.base();
    f.write("locales/en.ftl", "k = v\n");
    f.write("locales/ru.ftl", "broken = { $x\n");

    let out = validate_locales(&f.root().join("locales"));
    assert!(!out.errors.is_empty(), "expected parse error");
    assert!(
        out.errors[0].contains("locales/ru.ftl: parse error at byte "),
        "unexpected error: {}",
        out.errors[0]
    );
}

#[test]
fn missing_en_is_an_error() {
    let f = Fixture::new();
    f.write("error_codes/codes.toml", "schema = 1\n");

    let out = validate_locales(&f.root().join("locales"));
    assert_eq!(out.errors.len(), 1);
    assert!(
        out.errors[0].starts_with("locales/en.ftl: "),
        "unexpected error: {}",
        out.errors[0]
    );
    assert!(out.coverage.is_empty());
}

#[test]
fn select_expression_variables_are_compared() {
    let f = Fixture::new();
    f.base();
    // The selector variable AND both variant bodies must match.
    f.write("locales/en.ftl", "sum = { $count ->\n   *[other] Many\n}\n");
    f.write("locales/ru.ftl", "sum = { $n ->\n   *[other] Many\n}\n");

    let out = validate_locales(&f.root().join("locales"));
    assert_eq!(out.errors.len(), 1, "errors: {:#?}", out.errors);
    assert!(
        out.errors[0].contains("missing $count") && out.errors[0].contains("extra $n"),
        "unexpected error: {}",
        out.errors[0]
    );
}
