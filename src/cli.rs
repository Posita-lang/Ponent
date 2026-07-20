use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ponent", about = "Posita compiler toolchain")]
#[command(subcommand_required = true)]
#[command(arg_required_else_help = true)]
pub struct Cli {
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
    /// Show detailed explanation for an error code (e.g. `ponent explain E030`).
    /// Without a code, lists all available error codes.
    Explain {
        /// Error code to explain, e.g. "E030" or "W113".
        code: Option<String>,
    },
}
