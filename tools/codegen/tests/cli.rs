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
    assert!(stdout.contains("en: 19/19"), "stdout:\n{stdout}");
    assert!(stdout.contains("ru: 19/19"), "stdout:\n{stdout}");
    assert!(stdout.contains("zh: 19/19"), "stdout:\n{stdout}");
    assert!(
        stdout.contains("codegen check OK (89 codes)"),
        "stdout:\n{stdout}"
    );
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
