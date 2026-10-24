use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

const URL: &str = "https://raw.githubusercontent.com/flowlog-rs/flowlog/main/example/program_analysis/polonius_int.dl";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=POLONIUS_DL_LOCAL");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");
    println!("cargo:rerun-if-changed=error_codes");
    // `static_loader!` include_str!s each existing ftl (rustc-tracked), but a
    // brand-new file only reaches the macro when this script re-runs.
    println!("cargo:rerun-if-changed=locales");

    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set — run from a build.rs"));

    // Validate the error-code registry before anything else so data errors
    // surface even when the polonius download below would fail (offline …).
    generate_error_codes(&out_dir)?;

    let dl_path = out_dir.join("polonius_int.dl");

    // Allow overriding with a local file (offline dev, internal mirror, etc.).
    if let Some(local) = env::var_os("POLONIUS_DL_LOCAL") {
        fs::copy(&local, &dl_path)?;
        flowlog_build::compile(&dl_path)?;
        return Ok(());
    }

    // Non-empty check, not `exists()`: an interrupted download (ctrl-C during
    // `fs::write`) leaves a truncated file that would otherwise be trusted
    // forever and compile into a stub engine.
    if fs::metadata(&dl_path).map(|m| m.len() == 0).unwrap_or(true) {
        let offline = env::var_os("CARGO_NET_OFFLINE").is_some();
        if offline {
            return Err(
                "CARGO_NET_OFFLINE is set but polonius_int.dl is missing from OUT_DIR — \
                 set POLONIUS_DL_LOCAL to a local path, or build without --offline"
                    .into(),
            );
        }
        let client = reqwest::blocking::Client::builder()
            // TLS certificate verification is enabled by default via rustls-tls.
            // Previous `danger_accept_invalid_certs(true)` exposed a supply-chain
            // MITM vector; the downloaded polonius_int.dl is fed into the Datalog
            // compiler, so a forged file could inject arbitrary build-time logic.
            .timeout(Duration::from_secs(20))
            .connect_timeout(Duration::from_secs(5))
            .build()?;
        let resp = client.get(URL).send()?;
        if !resp.status().is_success() {
            return Err(format!("download failed: HTTP {}", resp.status()).into());
        }
        fs::write(&dl_path, resp.bytes()?)?;
    }

    flowlog_build::compile(&dl_path)?;
    Ok(())
}

/// Validate `error_codes/codes.toml` (+ referenced `*.md` files) and emit
/// `OUT_DIR/error_codes_gen.rs` containing the `CODE_TABLE` const.
///
/// Validation: `schema`, code format `^[EW]\d{3}$` (unique by TOML map key),
/// legal category, explain file exists and clears the size floor, duplicate
/// titles reported as `cargo:warning`.  Explain bodies are embedded via
/// `include_str!` with an absolute `CARGO_MANIFEST_DIR`-based path so they
/// resolve correctly from the OUT_DIR-generated file.
///
/// Registry + locale checks live in `ponent-codegen` (shared with
/// `cargo xtask codegen --check`); this front-end downgrades hygiene lints
/// to `cargo:warning` and hard-fails on validation errors.
fn generate_error_codes(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR not set")?;
    let outcome = ponent_codegen::collect(&PathBuf::from(manifest));
    for lint in &outcome.lints {
        println!("cargo:warning={lint}");
    }
    if !outcome.errors.is_empty() {
        return Err(outcome.errors.join("\n").into());
    }
    ponent_codegen::write_generated(&out_dir.join("error_codes_gen.rs"), &outcome.generated)?;
    Ok(())
}
