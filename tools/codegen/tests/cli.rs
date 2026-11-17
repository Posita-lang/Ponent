//! End-to-end tests of the `ponent-codegen` binary (the `cargo xtask` CLI).

mod common;

use std::process::Command;

use common::repo_root;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ponent-codegen"))
}

#[test]
fn check_passes_on_the_real_repo() {
    let out = bin()
        .args(["codegen", "--check"])
        .current_dir(repo_root())
        .output()
        .expect("run binary");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr:\n{stderr}\nstdout:\n{stdout}");
    // Counts drift as messages/codes are added; assert the *shape* (every
    // locale covers the full en set) instead of freezing today's numbers.
    for locale in ["en", "ru", "zh"] {
        let prefix = format!("{locale}: ");
        let line = stdout
            .lines()
            .find(|l| l.starts_with(&prefix))
            .unwrap_or_else(|| panic!("missing locale line {locale:?} in:\n{stdout}"));
        let (present, total) = line[prefix.len()..]
            .split_once('/')
            .unwrap_or_else(|| panic!("malformed coverage line: {line:?}"));
        assert_eq!(
            present, total,
            "locale {locale} does not cover every message: {line:?}"
        );
    }
    assert!(stdout.contains("codegen check OK ("), "stdout:\n{stdout}");
}

#[test]
fn check_from_a_subdirectory_still_finds_the_root() {
    let out = bin()
        .args(["codegen", "--check"])
        .current_dir(repo_root().join("src"))
        .output()
        .expect("run binary");
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn unknown_subcommand_exits_2() {
    let out = bin().arg("bogus").output().expect("run binary");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn site_renders_via_cli() {
    let dest = tempfile::tempdir().expect("tempdir");
    let out = bin()
        .args(["site", "--out"])
        .arg(dest.path().join("site"))
        .current_dir(repo_root())
        .output()
        .expect("run binary");
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dest.path().join("site/index.html").is_file());
    assert!(dest.path().join("site/E001.html").is_file());
}
