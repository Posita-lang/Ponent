mod ast;
mod cli;
mod diagnostics;
mod hir {
    pub mod builtins;
    pub mod checker;
    pub mod hir;
    pub mod infer;
    pub mod resolver;
    pub mod shape_var;
    pub mod smt;
    pub mod symbol;
    pub mod traits;
    pub mod types;
}
mod lexer;
mod parser;
mod vfs;

use crate::diagnostics::Diagnostic;
use clap::Parser;
use hir::checker::TypeChecker;
use hir::resolver::NameResolver;
use hir::types::{CrateId, DefId, TypeContext};
use logos::Logos;
use std::fs;
use std::process;

fn emit_error_diags(diags: &[Diagnostic]) {
    for diag in diags {
        if diag.is_error() {
            eprintln!("{}", diag);
        }
    }
}

fn main() {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Lex { file } => {
            let source = fs::read_to_string(&file).expect("failed to read file");
            let lexer = lexer::Token::lexer(&source);
            for result in lexer {
                match result {
                    Ok(token) => {
                        if token != lexer::Token::WhitespaceOrComment {
                            println!("{:?}", token);
                        }
                    }
                    Err(()) => eprintln!("ERROR: invalid token"),
                }
            }
        }
        cli::Command::Parse { file, ast } => {
            let source = fs::read_to_string(&file).expect("failed to read file");
            let mut parser = parser::Parser::new(&source);
            match parser.parse_program() {
                Ok(program) => {
                    if ast {
                        println!("{:#?}", program);
                    } else {
                        let mut ctx = TypeContext::new();
                        let local_crate_id = CrateId(DefId(0));
                        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
                        match resolver.resolve_program(&program) {
                            Ok((mut symbols, mut trait_env, _diags, resolution_map)) => {
                                // Register built-in traits and impls before type checking
                                hir::builtins::register_builtins(
                                    &mut symbols,
                                    &mut trait_env,
                                    &mut ctx,
                                );

                                let mut checker = TypeChecker::new(&mut ctx, &symbols, &mut trait_env, resolution_map);
                                match checker.check_program(&program) {
                                    Ok(_hir_program) => {
                                        println!("Type checking succeeded.");
                                    }
                                    Err(errors) => {
                                        emit_error_diags(&errors.into_inner());
                                        process::exit(1);
                                    }
                                }
                            }
                            Err(errors) => {
                                emit_error_diags(&errors.into_inner());
                                process::exit(1);
                            }
                        }
                    }
                }
                Err(diagnostics) => {
                    emit_error_diags(&diagnostics);
                    process::exit(1);
                }
            }
        }
    }
}
