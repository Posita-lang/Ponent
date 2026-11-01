//! Build-time validation and code generation for ponent's error-code
//! registry (`error_codes/codes.toml` + `error_codes/*.md`) and Fluent
//! locales (`locales/*.ftl`).
//!
//! Two consumers share this crate:
//! - the root `build.rs`, which hard-fails the build on invalid data and
//!   writes the generated `CODE_TABLE` source into `OUT_DIR`
//! - the `cargo xtask` CLI (`cargo xtask codegen --check`), which adds
//!   duplicate-title / unreferenced-file lints as hard errors for CI,
//!   prints the translation coverage report, and renders the HTML site
//!   (feature `site`).

mod codes;
mod locales;
#[cfg(feature = "site")]
mod site;

pub use codes::{CATEGORIES, MIN_EXPLAIN_BYTES, Registry, RegistryEntry, load_registry};
pub use locales::{CoverageRow, Locales, generate_fluent_functions, validate_locales};
#[cfg(feature = "site")]
pub use site::{SiteReport, render_site};

use std::fs;
use std::path::Path;

/// Everything `collect` found in one pass over `error_codes/` + `locales/`.
///
/// `errors` are hard failures (invalid data — the build / `--check` must
/// fail).  `lints` are hygiene findings that `build.rs` downgrades to
/// `cargo:warning` but `codegen --check` treats as failures.
/// `generated` is the table source, empty when `errors` is non-empty.
pub struct Outcome {
    pub errors: Vec<String>,
    pub lints: Vec<String>,
    pub coverage: Vec<CoverageRow>,
    pub generated: String,
}

/// Validate the registry and locales under `root` (a directory containing
/// `error_codes/` and `locales/`) and produce the generated table source.
///
/// Never panics: all problems are reported through [`Outcome::errors`] /
/// [`Outcome::lints`].
pub fn collect(root: &Path) -> Outcome {
    let registry = codes::load_registry(&root.join("error_codes"));
    let locales = locales::validate_locales(&root.join("locales"));

    let mut errors = registry.errors;
    errors.extend(locales.errors);

    Outcome {
        errors,
        lints: registry.lints,
        coverage: locales.coverage,
        generated: registry.generated,
    }
}

/// Write `generated` to `out` unless the file already has that exact
/// content.  Returns whether the file was written.  Empty `generated`
/// (invalid registry) is never written.
pub fn write_generated(out: &Path, generated: &str) -> std::io::Result<bool> {
    if generated.is_empty() {
        return Ok(false);
    }
    if fs::read_to_string(out).ok().as_deref() == Some(generated) {
        return Ok(false);
    }
    fs::write(out, generated)?;
    Ok(true)
}
