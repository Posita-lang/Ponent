use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const URL: &str = "https://raw.githubusercontent.com/flowlog-rs/flowlog/main/example/program_analysis/polonius_int.dl";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=POLONIUS_DL_LOCAL");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");

    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set — run from a build.rs"));
    let dl_path = out_dir.join("polonius_int.dl");

    // Allow overriding with a local file (offline dev, internal mirror, etc.).
    if let Some(local) = env::var_os("POLONIUS_DL_LOCAL") {
        fs::copy(&local, &dl_path)?;
        flowlog_build::compile(&dl_path)?;
        return Ok(());
    }

    if !dl_path.exists() {
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
