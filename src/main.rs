#![deny(clippy::correctness)]
#![warn(
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::inline_trait_bounds,
    clippy::undocumented_unsafe_blocks,
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::mem_forget,
    clippy::large_enum_variant
)]

mod arena;
mod ast;
mod cli;
mod diagnostics;
mod hir {
    pub mod bii;
    pub mod builtins;
    pub mod cfg;
    pub mod cfg_graph;
    pub mod checker;
    pub mod comptime;
    pub mod generate;
    pub mod hir;
    pub mod infer;
    #[cfg(test)]
    mod infer_tests;
    pub mod loop_infer;
    pub(crate) mod loop_ir;
    pub mod octagon;
    pub mod parametricity;
    pub mod place;
    pub mod polonius;
    pub mod query;
    pub mod resolver;
    pub mod shape_var;
    pub mod smt;
    pub mod symbol;
    pub mod target;
    pub mod traits;
    pub mod type_eq;
    pub mod types;
    pub mod visit;
    // Debug-only tracing / diagnostics module.
    #[cfg(debug_assertions)]
    pub mod anya;
}
mod lexer;
mod parser;
mod sat;
mod session;
mod sexp_ext;
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
use session::CompilerSession;
use std::fs;
use std::process;

/// Attach source text to all diagnostics in a slice so the emitter can
/// render `^`-underline annotations (Rust-style source context).
///
/// Diagnostics borrow the source buffer directly (`&'src str`) instead of
/// holding an owned copy, so this is a cheap pointer write per diagnostic.
fn attach_source_to_diags<'src>(
    diags: &mut [Diagnostic<'src>],
    source: &'src str,
    file_name: &str,
) {
    for diag in diags.iter_mut() {
        if diag.source().is_none() {
            diag.set_source(Some(source));
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

/// Phase 1 (generate expansion) → name resolution → Phase 2 (generate
/// expansion with resolution) → incremental resolution — all inside one
/// function so each phase's `&mut ctx` borrow is scoped sequentially.
fn resolve_and_expand<'src, 'input>(
    ctx: &mut TypeContext<'input>,
    program: ast::Program<'input>,
    local_crate_id: CrateId,
    source: &'src str,
    file_name: &str,
) -> (
    ast::Program<'input>,
    hir::symbol::SymbolTable<'input>,
    hir::traits::TraitEnv<'input>,
    Vec<Diagnostic<'src>>,
    hir::resolver::ResolutionMap<'input>,
) {
    // Phase 1: expand `generate` blocks before name resolution.
    let mut expander = hir::generate::GenerateExpander::new_phase1(ctx);
    let (expanded_items, _) = expander.expand_program(program.items);
    let program = ast::Program {
        items: expanded_items,
        span: program.span,
        doc_map: program.doc_map,
    };
    // Name resolution (the resolver's `&mut ctx` borrow ends here).
    let (symbols, trait_env, resolver_diags, resolution_map) = {
        let mut resolver = NameResolver::new(ctx, local_crate_id, Some(source));
        resolver.resolve_program(&program)
    };
    let mut all_diags = resolver_diags.into_inner();
    attach_source_to_diags(&mut all_diags, source, file_name);
    // Phase 2: expand with name resolution complete.
    let mut expander = hir::generate::GenerateExpander::new_phase2(ctx, &symbols);
    let (expanded_items, new_items) = expander.expand_program(program.items);
    let program = ast::Program {
        items: expanded_items,
        span: program.span,
        doc_map: program.doc_map,
    };
    // Incremental resolution for newly expanded items.
    let (symbols, trait_env, resolver_diags, resolution_map) = if !new_items.is_empty() {
        let mut resolver = NameResolver::new(ctx, local_crate_id, Some(source));
        resolver.resolve_incremental(&new_items, symbols, trait_env, resolution_map)
    } else {
        (symbols, trait_env, DiagCtxt::new(), resolution_map)
    };
    let mut phase2_diags = resolver_diags.into_inner();
    attach_source_to_diags(&mut phase2_diags, source, file_name);
    all_diags.extend(phase2_diags);
    (program, symbols, trait_env, all_diags, resolution_map)
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

/// Compiler options shared by the `Parse` and `Check` handlers.
struct CompilerOptions {
    target: Option<String>,
    strict: bool,
    enable_experimental: bool,
    feature: Vec<String>,
    debug: bool,
}

fn main() {
    // One arena for the whole process, leaked so `'input` is `'static` and
    // lowered HIR can be a query value.  See `crate::arena::leaked`.  Every
    // allocation goes through the `SyncArena` (never the bare `&Bump`), so
    // the arena's allocation lock is always respected.
    let (_bump, session_sync_arena) = crate::arena::leaked();

    let cli = cli::Cli::parse();
    let opts = CompilerOptions {
        target: cli.target,
        strict: cli.strict,
        enable_experimental: cli.enable_experimental,
        feature: cli.feature,
        debug: cli.debug,
    };

    // Each handler returns the diagnostics to render instead of exiting:
    // `process::exit` from a rayon worker would tear down the whole batch.
    // The single process exit lives at the end of `main`.
    let code = match cli.command {
        cli::Command::Lex { file } => report(run_lex(&file), false),
        cli::Command::Parse { file, ast, json } => {
            let code = report(run_parse(&file, ast, &opts, session_sync_arena), json);
            if code == 0 && !ast {
                println!("Type checking succeeded.");
            }
            code
        }
        cli::Command::Check { file, json } => {
            let code = report(run_check(&file, &opts, session_sync_arena), json);
            if code == 0 {
                println!("Type checking succeeded.");
            }
            code
        }
        cli::Command::Explain { code } => {
            run_explain(code);
            0
        }
    };
    if code != 0 {
        process::exit(code);
    }
}

/// Render a handler's diagnostics and return the process exit code: `1` when
/// any diagnostic is an error, `0` otherwise (warnings alone do not fail).
fn report(outcome: Result<(), Vec<Diagnostic<'static>>>, json: bool) -> i32 {
    let diags = match outcome {
        Ok(()) => return 0,
        Err(diags) => diags,
    };
    let mut emitter = make_emitter(json);
    emitter.emit_all(&diags);
    i32::from(diags.iter().any(|d| d.is_error()))
}

/// Lex a file, printing each token.  Never exits.
fn run_lex(file: &str) -> Result<(), Vec<Diagnostic<'static>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        // Don't attach source — the file couldn't be read, so there is no
        // source code to display.  The emitter closes the box with └─.
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
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
    Ok(())
}

/// Parse a file, and unless `--ast` was given, resolve and type-check it.
///
/// Returns the diagnostics to render; never calls `process::exit`, so it is
/// safe to invoke from a rayon worker.
fn run_parse(
    file: &str,
    ast: bool,
    opts: &CompilerOptions,
    sync_arena: &'static arena::SyncArena<'static>,
) -> Result<(), Vec<Diagnostic<'static>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
    // Copy the source into the leaked arena so the diagnostics that borrow it
    // can outlive this call and be rendered by `main`.
    let source: &'static str = sync_arena.alloc_str(&source);

    let mut parser = parser::Parser::new(source, sync_arena);
    let program = match parser.parse_program() {
        Ok(p) => p,
        Err(mut diagnostics) => {
            attach_source_to_diags(&mut diagnostics, source, file);
            return Err(diagnostics);
        }
    };
    if ast {
        println!("{:#?}", program);
        return Ok(());
    }

    let target = match resolve_target(&opts.target) {
        Ok(t) => t,
        Err(msg) => return Err(vec![Diagnostic::error(msg)]),
    };
    let mut ctx = TypeContext::new_with_target(target.clone());
    // A `SyncArena`, not a bare `&Bump`: a bare `&Bump` makes `TypeContext`
    // neither `Send` nor `Sync`, which is what blocked queries from being
    // handed it.  See `crate::arena`.
    ctx.arena = Some(sync_arena);
    let local_crate_id = CrateId(DefId(0));

    // Phase 1 → resolve → Phase 2 → incremental resolve, all inside a helper
    // so each phase's `&mut ctx` borrow is scoped (the helper takes `ctx`
    // once and the phases reborrow it sequentially).
    let (program, symbols, trait_env, mut all_diags, resolution_map) =
        resolve_and_expand(&mut ctx, program, local_crate_id, source, file);
    attach_source_to_diags(&mut all_diags, source, file);

    // From here the session owns the pipeline state, so a query can be handed
    // all of it through one provider.
    let mut session = CompilerSession::new(ctx, symbols, trait_env, resolution_map, target);

    let has_main = session.resolution_map.has_main;
    let mut checker = TypeChecker::new_with_source(
        &mut session.ctx,
        &session.symbols,
        &mut session.trait_env,
        // `TypeChecker` takes the map by value, so the session keeps its own
        // copy for queries to read.
        session.resolution_map.clone(),
        opts.strict,
        opts.enable_experimental,
        opts.feature.clone(),
        opts.debug,
        Some(source),
    );
    let checker_result = checker.check_program(&program);
    // Extract checker diagnostics (warnings, notes) even on success.
    let mut checker_diag_vec = checker.take_diagnostics().into_inner();
    attach_source_to_diags(&mut checker_diag_vec, source, file);
    match checker_result {
        Ok(_hir_program) => {
            // Checker succeeded — report any resolver diagnostics.
            let mut diags = checker_diag_vec;
            diags.extend(all_diags);
            // Verify that a `main` function exists.
            if !has_main {
                let mut main_diag = vec![
                    Diagnostic::error("`main` function not found in crate")
                        .with_code_str("E062")
                        .with_help("add a `def main() { ... }` function as the entry point"),
                ];
                attach_source_to_diags(&mut main_diag, source, file);
                diags.extend(main_diag);
            }
            if diags.is_empty() { Ok(()) } else { Err(diags) }
        }
        Err(errors) => {
            let mut diags = errors.into_inner();
            attach_source_to_diags(&mut diags, source, file);
            // Merge checker diagnostics (warnings accumulated before the
            // fatal error) and resolver diagnostics — neither may be dropped.
            diags.extend(checker_diag_vec);
            diags.extend(all_diags);
            Err(diags)
        }
    }
}

/// Parse, resolve, and type-check a file.  Never calls `process::exit`.
fn run_check(
    file: &str,
    opts: &CompilerOptions,
    sync_arena: &'static arena::SyncArena<'static>,
) -> Result<(), Vec<Diagnostic<'static>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
    let source: &'static str = sync_arena.alloc_str(&source);

    let mut parser = parser::Parser::new(source, sync_arena);
    let program = match parser.parse_program() {
        Ok(p) => p,
        Err(mut diagnostics) => {
            attach_source_to_diags(&mut diagnostics, source, file);
            return Err(diagnostics);
        }
    };
    let target = match resolve_target(&opts.target) {
        Ok(t) => t,
        Err(msg) => return Err(vec![Diagnostic::error(msg)]),
    };
    let mut ctx = TypeContext::new_with_target(target.clone());
    ctx.arena = Some(sync_arena);
    let local_crate_id = CrateId(DefId(0));
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id, Some(source));
    let (symbols, trait_env, resolver_diags, resolution_map) = resolver.resolve_program(&program);
    let mut all_diags = resolver_diags.into_inner();
    attach_source_to_diags(&mut all_diags, source, file);

    // Same session plumbing as the `Parse` path above.
    let mut session = CompilerSession::new(ctx, symbols, trait_env, resolution_map, target);

    let mut checker = TypeChecker::new_with_source(
        &mut session.ctx,
        &session.symbols,
        &mut session.trait_env,
        // `TypeChecker` takes the map by value, so the session keeps its own
        // copy for queries to read.
        session.resolution_map.clone(),
        opts.strict,
        opts.enable_experimental,
        opts.feature.clone(),
        opts.debug,
        Some(source),
    );
    let checker_result = checker.check_program(&program);
    let mut checker_diag_vec = checker.take_diagnostics().into_inner();
    attach_source_to_diags(&mut checker_diag_vec, source, file);
    match checker_result {
        Ok(_hir) => {
            let mut diags = checker_diag_vec;
            diags.extend(all_diags);
            if diags.is_empty() { Ok(()) } else { Err(diags) }
        }
        Err(errors) => {
            let mut diags = errors.into_inner();
            attach_source_to_diags(&mut diags, source, file);
            diags.extend(all_diags);
            Err(diags)
        }
    }
}

/// Show an error-code explanation (blocking while the explain server runs).
fn run_explain(code: Option<String>) {
    match code {
        Some(code) => {
            // Start the local explain server on a random port.
            let listener =
                std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind explain server");
            let port = listener.local_addr().unwrap().port();
            diagnostics::error_code::set_explain_port(port);

            let explanation = diagnostics::explain_error_code(&code);
            println!("{}", explanation);

            let explain_code = code.clone();
            let url = format!("http://127.0.0.1:{port}/{explain_code}");
            // Open the browser in a background thread so the server can
            // accept connections.
            let url_bg = url.clone();
            std::thread::spawn(move || {
                let _ = open_browser(&url_bg);
            });

            // Serve the explain page using the already-bound listener.
            diagnostics::explain_server::serve_explain(&listener, &explain_code).unwrap_or_else(
                |e| {
                    eprintln!("warning: explain server error: {e}");
                },
            );
        }
        None => {
            print!("{}", diagnostics::list_error_codes());
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
