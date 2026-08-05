use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ponent", about = "Posita compiler toolchain")]
#[command(subcommand_required = true)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    /// Target specification (e.g. "x86_64-linux-gnu" or path to a .json file).
    /// If omitted, the host target is detected automatically.
    #[arg(long, global = true)]
    pub target: Option<String>,

    /// Enable strict mode: all @trusted functions must have @link_proof or
    /// @comptime_test evidence.  Unsafe operations and dynamic dispatch are
    /// also restricted.  Recommended for safety-critical builds.
    #[arg(long, global = true)]
    pub strict: bool,

    /// Enable experimental features and items marked with @experimental.
    /// Without this flag, any use of an @experimental item is a compile error.
    #[arg(long, global = true)]
    pub enable_experimental: bool,

    /// Conditional compilation features: `--feature logging`.
    /// Can be specified multiple times.  Used by `@cfg(feature = "...")`.
    /// Each value is a feature name (e.g. `logging`), NOT a `key=value` pair.
    #[arg(long, global = true, value_name = "FEATURE")]
    pub feature: Vec<String>,

    /// Enable `@cfg(debug)` evaluation (NOT debug output).
    ///
    /// When set, `@cfg(debug)` conditions in source code evaluate to `true`.
    /// Without this flag, `@cfg(debug)` is always `false`.
    /// This flag does NOT enable verbose logging or debug-level diagnostics.
    #[arg(long, global = true)]
    pub debug: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Lex {
        file: String,
    },
    Parse {
        file: String,
        #[arg(long)]
        ast: bool,
        #[arg(long, help = "Output diagnostics as JSON (instead of colored text)")]
        json: bool,
    },
    /// Parse, resolve, and type-check a source file without generating code.
    /// Exits 0 on success, 1 if any errors were reported.
    Check {
        file: String,
        #[arg(long, help = "Output diagnostics as JSON (instead of colored text)")]
        json: bool,
    },
    /// Show detailed explanation for an error code (e.g. `ponent explain E030`).
    /// Without a code, lists all available error codes.
    Explain {
        /// Error code to explain, e.g. "E030" or "W113".
        code: Option<String>,
    },
}
