//! `cargo xtask` entry point: registry/locale validation gate and doc-site
//! renderer.  See `--help`.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
usage: ponent-codegen codegen --check
       ponent-codegen site --out DIR

subcommands:
  codegen --check  validate error_codes/ + locales/; print coverage (exit 0 ok, 1 invalid)
  site --out DIR   render the HTML explain site (default DIR: target/error_codes_site)
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help" | "-h") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("codegen") => cmd_codegen(&args[1..]),
        Some("site") => cmd_site(&args[1..]),
        _ => {
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn cmd_codegen(rest: &[String]) -> ExitCode {
    if !rest.is_empty() && !(rest.len() == 1 && rest[0] == "--check") {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }
    let root = match find_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let outcome = ponent_codegen::collect(&root);
    for lint in &outcome.lints {
        eprintln!("lint: {lint}");
    }
    for error in &outcome.errors {
        eprintln!("error: {error}");
    }
    for row in &outcome.coverage {
        println!("{}: {}/{}", row.locale, row.present, row.total);
    }

    if outcome.errors.is_empty() && outcome.lints.is_empty() {
        let codes = outcome.generated.matches("include_str!").count();
        println!("codegen check OK ({codes} codes)");
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "codegen check FAILED ({} errors, {} lints)",
            outcome.errors.len(),
            outcome.lints.len()
        );
        ExitCode::FAILURE
    }
}

#[cfg(feature = "site")]
fn cmd_site(rest: &[String]) -> ExitCode {
    let mut out = PathBuf::from("target/error_codes_site");
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--out" => {
                i += 1;
                let Some(path) = rest.get(i) else {
                    eprintln!("error: --out requires a path");
                    return ExitCode::from(2);
                };
                out = PathBuf::from(path);
            }
            other => {
                eprintln!("error: unknown argument {other:?}\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let root = match find_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    match ponent_codegen::render_site(&root, &out) {
        Ok(report) => {
            println!("rendered {} pages to {}", report.pages, out.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "site"))]
fn cmd_site(_rest: &[String]) -> ExitCode {
    eprintln!("error: ponent-codegen was built without the `site` feature");
    ExitCode::from(2)
}

/// Walk up from the current directory to the workspace root (the nearest
/// directory containing `error_codes/codes.toml`).
fn find_root() -> Result<PathBuf, String> {
    let mut dir = env::current_dir().map_err(|e| format!("cannot read cwd: {e}"))?;
    loop {
        if dir.join("error_codes").join("codes.toml").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Err(
                "cannot locate the workspace root: no error_codes/codes.toml in this directory or its parents".to_string(),
            );
        }
    }
}
