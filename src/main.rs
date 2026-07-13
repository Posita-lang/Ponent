mod ast;
mod cli;
mod diagnostics;
mod hir {
    pub mod builtins;
    pub mod checker;
    pub mod comptime;
    pub mod generate;
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
mod symbol;
mod vfs;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::Parser;
use diagnostics::{ColoredEmitter, Diagnostic, DiagnosticEmitter, JsonEmitter};
use hir::checker::TypeChecker;
use hir::resolver::NameResolver;
use hir::types::{CrateId, DefId, TypeContext};
use logos::Logos;
use std::fs;
use std::process;

/// Attach source text to all diagnostics in a slice so the emitter can
/// render `^`-underline annotations (Rust-style source context).
fn attach_source_to_diags(diags: &mut [Diagnostic], source: &str) {
    for diag in diags.iter_mut() {
        if diag.source.is_none() {
            diag.source = Some(source.to_string());
        }
    }
}

fn make_emitter(json: bool) -> Box<dyn DiagnosticEmitter> {
    if json {
        Box::new(JsonEmitter::new_compact())
    } else {
        Box::new(ColoredEmitter::new())
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
        cli::Command::Parse { file, ast, json } => {
            let source = fs::read_to_string(&file).expect("failed to read file");
            let mut parser = parser::Parser::new(&source);
            match parser.parse_program() {
                Ok(program) => {
                    if ast {
                        println!("{:#?}", program);
                    } else {
                        let mut ctx = TypeContext::new();
                        let local_crate_id = CrateId(DefId(0));

                        // Expand `generate` blocks before name resolution,
                        // so the resolver sees only concrete declarations.
                        let expander = hir::generate::GenerateExpander::new(&mut ctx);
                        let expanded_items = expander.expand_program(program.items);
                        let program = ast::Program {
                            items: expanded_items,
                            span: program.span,
                        };

                        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
                        match resolver.resolve_program(&program) {
                            Ok((mut symbols, mut trait_env, _diags, resolution_map)) => {
                                let mut diags = _diags.into_inner();
                                attach_source_to_diags(&mut diags, &source);

                                // Register built-in traits and impls before type checking
                                hir::builtins::register_builtins(
                                    &mut symbols,
                                    &mut trait_env,
                                    &mut ctx,
                                );

                                let mut checker = TypeChecker::new(
                                    &mut ctx,
                                    &symbols,
                                    &mut trait_env,
                                    resolution_map,
                                );
                                match checker.check_program(&program) {
                                    Ok(_hir_program) => {
                                        println!("Type checking succeeded.");
                                    }
                                    Err(errors) => {
                                        let mut diags = errors.into_inner();
                                        attach_source_to_diags(&mut diags, &source);
                                        let mut emitter = make_emitter(json);
                                        emitter.emit_all(&diags);
                                        process::exit(1);
                                    }
                                }
                            }
                            Err(errors) => {
                                let mut diags = errors.into_inner();
                                attach_source_to_diags(&mut diags, &source);
                                let mut emitter = make_emitter(json);
                                emitter.emit_all(&diags);
                                process::exit(1);
                            }
                        }
                    }
                }
                Err(mut diagnostics) => {
                    attach_source_to_diags(&mut diagnostics, &source);
                    let mut emitter = make_emitter(json);
                    emitter.emit_all(&diagnostics);
                    process::exit(1);
                }
            }
        }
    }
}
