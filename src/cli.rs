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
}
