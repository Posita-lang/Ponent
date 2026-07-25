#![deny(clippy::correctness)]
#![warn(clippy::suspicious, clippy::style, clippy::complexity, clippy::perf)]

mod ast;
mod cli;
mod diagnostics;
mod hir {
    pub mod builtins;
    pub mod cfg;
    pub mod checker;
    pub mod comptime;
    pub mod generate;
    pub mod hir;
    pub mod infer;
    #[cfg(test)]
    mod infer_tests;
    pub mod query;
    pub mod resolver;
    pub mod shape_var;
    pub mod smt;
    pub mod symbol;
    pub mod target;
    pub mod traits;
    pub mod types;
}
mod lexer;
mod parser;
mod sat;
mod symbol;
mod vfs;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::Parser;
use diagnostics::{ColoredEmitter, DiagCtxt, Diagnostic, DiagnosticEmitter, JsonEmitter};
use hir::checker::TypeChecker;
use hir::resolver::NameResolver;
use hir::types::{CrateId, DefId, TypeContext};
use logos::Logos;
use std::fs;
use std::process;

/// Attach source text to all diagnostics in a slice so the emitter can
/// render `^`-underline annotations (Rust-style source context).
fn attach_source_to_diags(diags: &mut [Diagnostic], source: &str, file_name: &str) {
    for diag in diags.iter_mut() {
        if diag.source().is_none() {
            diag.set_source(Some(source.to_string()));
        }
        if diag.file_name() == "<input>" {
            diag.set_file_name(file_name.to_string());
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

/// Resolve a target from CLI argument: load a JSON file, look up a builtin,
/// or fall back to the host target.
fn resolve_target(target_opt: &Option<String>) -> Result<hir::target::Target, String> {
    match target_opt {
        Some(name) => {
            let path = std::path::Path::new(name);
            if path.exists() {
                // Try loading as a JSON target spec.
                match hir::target::Target::from_json(path) {
                    Ok(target) => Ok(target),
                    Err(e) => {
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                        Err(if ext != "json" {
                            format!(
                                "`{}` is not a valid target spec (expected `.json` file)\n\
                                     note: file extension is `.{}`\n\
                                     help: rename the file to `{}` or convert it to JSON.",
                                name,
                                ext,
                                path.with_extension("json").display()
                            )
                        } else {
                            format!("invalid target spec `{}`: {}", name, e)
                        })
                    }
                }
            } else {
                // Try as a built-in target name.
                hir::target::Target::builtin(name).ok_or_else(|| {
                    let targets = [
                        "  x86_64-linux-gnu",
                        "  aarch64-linux-gnu",
                        "  riscv64",
                        "  wasm32",
                        "  arm-eabi",
                        "  custom-template",
                    ];
                    format!("unknown target `{}`\navailable built-in targets:\n{}\n  or provide a path to a .json target spec file.",
                        name, targets.join("\n"))
                })
            }
        }
        None => Ok(hir::target::Target::host()),
    }
}

fn main() {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Lex { file } => {
            let source = match fs::read_to_string(&file) {
                Ok(s) => s,
                Err(e) => {
                    let mut diag = Diagnostic::error(format!("failed to read `{}`: {}", file, e));
                    // Don't set source — the file couldn't be read, so there's
                    // no source code to display.  The emitter will render the
                    // error without source context and close the box with └─.
                    let mut emitter = ColoredEmitter::new();
                    emitter.emit(&diag);
                    emitter.emit_summary(1, 0);
                    process::exit(1);
                }
            };
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
            let source = match fs::read_to_string(&file) {
                Ok(s) => s,
                Err(e) => {
                    let mut diag = Diagnostic::error(format!("failed to read `{}`: {}", file, e));
                    // Don't set source — the file couldn't be read, so there's
                    // no source code to display.  The emitter will render the
                    // error without source context and close the box with └─.
                    let mut emitter = make_emitter(json);
                    emitter.emit(&diag);
                    emitter.emit_summary(1, 0);
                    process::exit(1);
                }
            };
            let mut parser = parser::Parser::new(&source);
            match parser.parse_program() {
                Ok(program) => {
                    if ast {
                        println!("{:#?}", program);
                    } else {
                        let target = match resolve_target(&cli.target) {
                            Ok(t) => t,
                            Err(msg) => {
                                let mut diag = Diagnostic::error(msg);
                                let mut emitter = make_emitter(json);
                                emitter.emit(&diag);
                                emitter.emit_summary(1, 0);
                                process::exit(1);
                            }
                        };
                        let mut ctx = TypeContext::new_with_target(target);
                        let local_crate_id = CrateId(DefId(0));

                        // Expand `generate` blocks before name resolution,
                        // so the resolver sees only concrete declarations.
                        let mut expander = hir::generate::GenerateExpander::new_phase1(&mut ctx);
                        let (expanded_items, _) = expander.expand_program(program.items);
                        let program = ast::Program {
                            items: expanded_items,
                            span: program.span,
                        };

                        let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
                        let (symbols, mut trait_env, resolver_diags, resolution_map) =
                            resolver.resolve_program(&program);
                        let mut all_diags = resolver_diags.into_inner();
                        attach_source_to_diags(&mut all_diags, &source, &file);

                        // Phase 2: expand `generate` blocks with name resolution
                        // complete — `@typeInfo!` can now return real type data.
                        let mut expander =
                            hir::generate::GenerateExpander::new_phase2(&mut ctx, &symbols);
                        let (expanded_items, new_items) = expander.expand_program(program.items);
                        let program = ast::Program {
                            items: expanded_items,
                            span: program.span,
                        };

                        // Phase 2 expansion may have introduced new AST nodes
                        // (e.g. from conditional branches).  Re-run the resolver
                        // so that any new type/function references are resolved
                        // before type checking.
                        let (symbols, mut trait_env, resolver_diags, resolution_map) =
                            if !new_items.is_empty() {
                                // Incremental resolution: only process the newly expanded
                                // items, reusing the existing symbol table from Phase 1.
                                let mut resolver = NameResolver::new(&mut ctx, local_crate_id);
                                resolver.resolve_incremental(
                                    &new_items,
                                    symbols,
                                    trait_env,
                                    resolution_map,
                                )
                            } else {
                                // No new items — keep the Phase 1 results unchanged.
                                (symbols, trait_env, DiagCtxt::new(), resolution_map)
                            };
                        let mut phase2_diags = resolver_diags.into_inner();
                        attach_source_to_diags(&mut phase2_diags, &source, &file);
                        all_diags.extend(phase2_diags);

                        let has_main = resolution_map.has_main;
                        let mut checker = TypeChecker::new_with_source(
                            &mut ctx,
                            &symbols,
                            &mut trait_env,
                            resolution_map,
                            cli.strict,
                            cli.enable_experimental,
                            cli.feature,
                            cli.debug,
                            Some(&source),
                        );
                        let checker_result = checker.check_program(&program);
                        // Extract checker diagnostics (warnings, notes) even on success.
                        let checker_diags = checker.take_diagnostics();
                        let mut checker_diag_vec: Vec<Diagnostic> = checker_diags.into_inner();
                        attach_source_to_diags(&mut checker_diag_vec, &source, &file);
                        if !checker_diag_vec.is_empty() {
                            let mut emitter = make_emitter(json);
                            emitter.emit_all(&checker_diag_vec);
                        }
                        match checker_result {
                            Ok(_hir_program) => {
                                // Checker succeeded — report any resolver diagnostics.
                                if !all_diags.is_empty() {
                                    let mut emitter = make_emitter(json);
                                    emitter.emit_all(&all_diags);
                                }
                                // Verify that a `main` function exists.
                                if !has_main {
                                    let mut diags = vec![
                                        Diagnostic::error(
                                            "`main` function not found in crate",
                                        )
                                        .with_code_str("E062")
                                        .with_help(
                                            "add a `def main() { ... }` function as the entry point",
                                        ),
                                    ];
                                    attach_source_to_diags(&mut diags, &source, &file);
                                    let mut emitter = make_emitter(json);
                                    emitter.emit_all(&diags);
                                    process::exit(1);
                                }
                                if all_diags.is_empty() {
                                    println!("Type checking succeeded.");
                                } else {
                                    process::exit(1);
                                }
                            }
                            Err(errors) => {
                                let mut diags = errors.into_inner();
                                attach_source_to_diags(&mut diags, &source, &file);
                                // Merge resolver diagnostics into checker diagnostics.
                                diags.extend(all_diags);
                                let mut emitter = make_emitter(json);
                                emitter.emit_all(&diags);
                                process::exit(1);
                            }
                        }
                    }
                }
                Err(mut diagnostics) => {
                    attach_source_to_diags(&mut diagnostics, &source, &file);
                    let mut emitter = make_emitter(json);
                    emitter.emit_all(&diagnostics);
                    process::exit(1);
                }
            }
        }
        cli::Command::Explain { code } => {
            match code {
                Some(code) => {
                    // Start the local explain server on a random port.
                    let listener = std::net::TcpListener::bind("127.0.0.1:0")
                        .expect("failed to bind explain server");
                    let port = listener.local_addr().unwrap().port();
                    diagnostics::error_code::set_explain_port(port);

                    let explanation = diagnostics::explain_error_code(&code);
                    println!("{}", explanation);

                    let explain_code = code.clone();
                    let url = format!("http://127.0.0.1:{port}/{explain_code}");
                    // Open the browser in a background thread so the server
                    // can accept connections.
                    let url_bg = url.clone();
                    std::thread::spawn(move || {
                        let _ = open_browser(&url_bg);
                    });

                    // Serve the explain page using the already-bound listener.
                    diagnostics::explain_server::serve_explain(listener, &explain_code)
                        .unwrap_or_else(|e| {
                            eprintln!("warning: explain server error: {e}");
                        });
                }
                None => {
                    print!("{}", diagnostics::list_error_codes());
                }
            }
        }
    }
}

/// Open a URL in the user's default browser.
#[cfg(target_os = "linux")]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("xdg-open").arg(url).spawn()?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("open").arg(url).spawn()?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) -> std::io::Result<()> {
    std::process::Command::new("cmd")
        .args(["/c", "start", url])
        .spawn()?;
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn open_browser(_url: &str) -> std::io::Result<()> {
    Ok(()) // silently no-op on unsupported platforms
}
