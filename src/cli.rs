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

    /// Colour output policy: `auto` (default, off when piped), `always`, or `never`.
    #[arg(long, global = true, value_name = "WHEN", default_value = "auto")]
    pub color: String,

    /// Only emit diagnostics with one of these error codes (repeatable, or
    /// comma-separated: `--only-code=E030,E031`).
    #[arg(long, global = true, value_name = "CODE", value_delimiter = ',')]
    pub only_code: Vec<String>,

    /// Only emit diagnostics in this error category: `parse`, `resolution`,
    /// `type`, `contract`, `trait`, `inference`, `internal`, `generic`
    /// (repeatable, comma-separated).
    #[arg(long, global = true, value_name = "CATEGORY", value_delimiter = ',')]
    pub only_category: Vec<String>,

    /// Only emit diagnostics whose file path contains this substring
    /// (repeatable, comma-separated).
    #[arg(long, global = true, value_name = "PATTERN", value_delimiter = ',')]
    pub only_file: Vec<String>,

    /// Suppress diagnostics with these error codes entirely (rustc-style
    /// `-A`).  Applied before `--only-*` filters.
    #[arg(long, global = true, value_name = "CODE", value_delimiter = ',')]
    pub allow: Vec<String>,

    /// Suppress `help:` / `note:` / suggestions / sub-diagnostics and the
    /// batch summary.  The error header and source context stay visible.
    #[arg(long, short = 'q', global = true)]
    pub quiet: bool,

    /// Do not soft-wrap message text (copy-paste-friendly).  Hard newlines
    /// are preserved.
    #[arg(long, global = true)]
    pub no_wrap: bool,

    /// Apply machine-applicable suggestions to the source file in place.
    /// Only safe edits by default; use `--fix-hints` to widen.
    #[arg(long, global = true)]
    pub fix: bool,

    /// Which suggestion applicability levels `--fix` accepts:
    /// `auto` (default, safe only), `review` (also maybe-incorrect), `all`.
    #[arg(long, global = true, value_name = "HINTS", default_value = "auto")]
    pub fix_hints: String,

    /// Output format: `human` (default), `short` (one line per diagnostic), `json`.
    #[arg(long, global = true, value_name = "FORMAT", default_value = "human")]
    pub error_format: String,

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
        /// Run as a persistent local server: the index page at `/` lists
        /// every code, and `/<code>` serves that code's explanation.  Opens
        /// `code`'s page (or the index, if no code is given) in the browser
        /// and blocks until Ctrl+C.
        #[arg(long)]
        serve: bool,
    },
    /// Parse, resolve, and type-check a file, then dump the MIR for every
    /// function body to stdout (debug aid for the lowering).
    Mir {
        file: String,
        #[arg(long, help = "Output diagnostics as JSON (instead of colored text)")]
        json: bool,
    },
}
