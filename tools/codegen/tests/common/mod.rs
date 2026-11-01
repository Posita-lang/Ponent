//! Shared fixture helpers for ponent-codegen integration tests.

// Each integration-test binary compiles this module independently and
// uses a different subset of the helpers.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

/// The real ponent workspace root (this crate lives at `tools/codegen`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonicalize repo root")
}

/// Temporary workspace with an `error_codes/` + `locales/` skeleton.
pub struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        Fixture {
            dir: tempfile::tempdir().expect("tempdir"),
        }
    }

    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// Write `content` at `rel` (relative to the fixture root), creating
    /// parent directories.
    pub fn write(&self, rel: &str, content: &str) {
        let path = self.dir.path().join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dirs");
        }
        fs::write(path, content).expect("write fixture file");
    }

    /// Minimal valid workspace: one code, one explain file, `en.ftl`.
    pub fn base(&self) -> &Self {
        self.write(
            "error_codes/codes.toml",
            "schema = 1\n\n[codes.E001]\ntitle = \"Alpha\"\ncategory = \"parse\"\nexplain = \"E001.md\"\n",
        );
        self.write(
            "error_codes/E001.md",
            "This is the E001 explanation body.\n",
        );
        self.write("locales/en.ftl", "k = v\n");
        self
    }
}
