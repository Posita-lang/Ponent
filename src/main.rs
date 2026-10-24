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

pub use smallvec;

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
    #[cfg(all(test, not(feature = "shuttle")))]
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
mod loader;
mod mir;
mod parser;
mod sat;
mod session;
mod sexp_ext;
mod source_map;
mod sourcemap;
mod symbol;
mod sync;
mod vfs;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::Parser;
use diagnostics::{ColorChoice, DiagCtxt, DiagOpts, Diagnostic, Emitter, HumanRenderer, JsonRenderer, OutputFormat, sink};
use hir::checker::TypeChecker;
use hir::resolver::NameResolver;
use hir::types::{CrateId, DefId, TypeContext};
use logos::Logos;
use session::CompilerSession;
use std::fs;
use std::io::{IsTerminal, Write as _};
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
            diag.set_file_name(file_name);
        }
    }
}

/// Phase 1 (generate expansion) → name resolution → Phase 2 (generate
/// expansion with resolution) → incremental resolution — all inside one
/// function so each phase's `&mut ctx` borrow is scoped sequentially.
fn resolve_and_expand<'src, 'input>(
    ctx: &mut TypeContext<'input>,
    program: ast::Program<'input, 'src>,
    local_crate_id: CrateId,
    source: &'src str,
    file_name: &str,
    enable_experimental: bool,
) -> (
    ast::Program<'input, 'src>,
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
        let mut resolver = NameResolver::new(ctx, local_crate_id, Some(source), enable_experimental);
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
        let mut resolver = NameResolver::new(ctx, local_crate_id, Some(source), enable_experimental);
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
    // The parse arena lives only for this `main` call — nothing borrows it
    // past process exit, so there is no reason to leak it, and `'input` is a
    // plain local lifetime rather than `'static`.  Every allocation goes
    // through the `SyncArena` (never the bare `&Bump`), so the arena's
    // allocation lock is always respected.
    let bump = bumpalo::Bump::new();
    let session_sync_arena = crate::arena::SyncArena::new(&bump);

    let cli = cli::Cli::parse();
    let color_choice = ColorChoice::from_str(&cli.color);
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
        cli::Command::Lex { file } => report(run_lex(&file), false, color_choice),
        cli::Command::Parse { file, ast, json } => {
            let code = report(run_parse(&file, ast, &opts, &session_sync_arena), json, color_choice);
            if code == 0 && !ast {
                println!("Type checking succeeded.");
            }
            code
        }
        cli::Command::Check { file, json } => {
            let code = report(run_check(&file, &opts, &session_sync_arena), json, color_choice);
            if code == 0 {
                println!("Type checking succeeded.");
            }
            code
        }
        cli::Command::Explain { code } => {
            run_explain(code, color_choice);
            0
        }
        cli::Command::Mir { file, json } => report(run_mir(&file, &opts, &session_sync_arena), json, color_choice),
    };
    if code != 0 {
        process::exit(code);
    }
}

/// Render a handler's diagnostics and return the process exit code: `1` when
/// any diagnostic is an error, `0` otherwise (warnings alone do not fail).
fn report<'a>(outcome: Result<(), Vec<Diagnostic<'a>>>, json: bool, color: ColorChoice) -> i32 {
    let mut diags = match outcome {
        Ok(()) => return 0,
        Err(diags) => diags,
    };
    let mut opts = DiagOpts::detect();
    opts.use_color = color.resolve();
    match OutputFormat::from_json_flag(json) {
        OutputFormat::Json => {
            let mut emitter = Emitter::new(JsonRenderer::new_compact(), sink::stderr(), opts);
            let _ = emitter.emit_all(&mut diags);
        }
        OutputFormat::Human => {
            let use_color = opts.use_color;
            let renderer = HumanRenderer::with_opts(use_color, Some(&opts));
            let mut emitter = Emitter::new(renderer, sink::stderr(), opts);
            let _ = emitter.emit_all(&mut diags);
        }
    }
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
fn run_parse<'a>(
    file: &str,
    ast: bool,
    opts: &CompilerOptions,
    sync_arena: &'a arena::SyncArena<'a>,
) -> Result<(), Vec<Diagnostic<'a>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
    // Register the file with a SourceMap so every span carries a real
    // FileId, then copy the source into the leaked arena so the diagnostics
    // that borrow it can outlive this call and be rendered by `main`.
    let mut source_map = sourcemap::SourceMap::new();
    let file_id = source_map.add_file(file, source).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to register `{}`: {}",
            file, e
        ))]
    })?;
    let source: &'a str = sync_arena.alloc_str(source_map.file(file_id).source());

    let mut parser = parser::Parser::new(source, file_id, sync_arena);
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
        resolve_and_expand(
            &mut ctx,
            program,
            local_crate_id,
            source,
            file,
            opts.enable_experimental,
        );
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
fn run_check<'a>(
    file: &str,
    opts: &CompilerOptions,
    sync_arena: &'a arena::SyncArena<'a>,
) -> Result<(), Vec<Diagnostic<'a>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
    let mut source_map = sourcemap::SourceMap::new();
    let file_id = source_map.add_file(file, source).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to register `{}`: {}",
            file, e
        ))]
    })?;
    let source: &'a str = sync_arena.alloc_str(source_map.file(file_id).source());

    let mut parser = parser::Parser::new(source, file_id, sync_arena);
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
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id, Some(source), opts.enable_experimental);
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

fn run_mir<'a>(
    file: &str,
    opts: &CompilerOptions,
    sync_arena: &'a arena::SyncArena<'a>,
) -> Result<(), Vec<Diagnostic<'a>>> {
    let source = fs::read_to_string(file).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to read `{}`: {}",
            file, e
        ))]
    })?;
    let mut source_map = sourcemap::SourceMap::new();
    let file_id = source_map.add_file(file, source).map_err(|e| {
        vec![Diagnostic::error(format!(
            "failed to register `{}`: {}",
            file, e
        ))]
    })?;
    let source: &'a str = sync_arena.alloc_str(source_map.file(file_id).source());

    let mut parser = parser::Parser::new(source, file_id, sync_arena);
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
    let mut resolver = NameResolver::new(&mut ctx, local_crate_id, Some(source), opts.enable_experimental);
    let (symbols, trait_env, resolver_diags, resolution_map) = resolver.resolve_program(&program);
    let mut all_diags = resolver_diags.into_inner();
    attach_source_to_diags(&mut all_diags, source, file);

    let mut session = CompilerSession::new(ctx, symbols, trait_env, resolution_map, target);

    let mut checker = TypeChecker::new_with_source(
        &mut session.ctx,
        &session.symbols,
        &mut session.trait_env,
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
        Ok(hir_program) => {
            let mut diags = checker_diag_vec;
            diags.extend(all_diags);
            if diags.iter().any(|d| d.is_error()) {
                return Err(diags);
            }
            for item in &hir_program.items {
                if matches!(item, hir::hir::HirStmt::FunctionDef { .. }) {
                    println!("{}", mir::build::lower_function(item).pretty());
                }
            }
            if diags.is_empty() { Ok(()) } else { Err(diags) }
        }
        Err(errors) => {
            let mut diags = errors.into_inner();
            attach_source_to_diags(&mut diags, source, file);
            diags.extend(checker_diag_vec);
            diags.extend(all_diags);
            Err(diags)
        }
    }
}

/// Show an error-code explanation (blocking while the explain server runs).
fn run_explain(code: Option<String>, color: ColorChoice) {
    match code {
        Some(code) => {
            let use_color = color.resolve();
            // Start the local explain server on a random port.
            let listener =
                std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind explain server");
            let port = listener.local_addr().unwrap().port();
            // Called once per process; a re-set would be a bug, so surface it.
            if let Err(previous) = diagnostics::error_code::set_explain_port(port) {
                eprintln!("warning: explain port already set to {previous}, ignoring {port}");
            }

            let explanation = diagnostics::explain_error_code(&code, use_color);
            println!("{}", explanation);

            let explain_code = code.clone();
            // The code is untrusted CLI input; `explain_server_url` encodes it
            // as a single path segment so it cannot escape the path or smuggle
            // control bytes into the URL handed to the browser.
            let url = diagnostics::error_code::explain_server_url(port, &explain_code);
            // Open the browser in a background thread so the server can
            // accept connections.
            let url_bg = url.clone();
            std::thread::spawn(move || {
                let _ = open_browser(&url_bg);
            });

            // Spin a one-line ASCII spinner on stderr while the server waits
            // for the browser to connect (~2 s), so the terminal does not
            // look frozen.  Only when stderr is a real terminal: a pipe or
            // log file must never receive cursor-movement control bytes.
            let spinner = if std::io::stderr().is_terminal() {
                Some(spawn_explain_spinner(&url, &explain_code))
            } else {
                None
            };

            // Serve the explain page using the already-bound listener.
            let served =
                diagnostics::explain_server::serve_explain(&listener, &explain_code);
            if let Some((handle, stop)) = spinner {
                stop_explain_spinner(handle, &stop);
            }
            served.unwrap_or_else(|e| {
                eprintln!("warning: explain server error: {e}");
            });
        }
        None => {
            print!("{}", diagnostics::list_error_codes());
        }
    }
}

/// Open a URL in the user's default browser.

/// Animate a one-line spinner on stderr while the explain server waits for
/// the browser to connect, e.g.
///
/// ```text
/// ⠋ opening E030 explanation on http://127.0.0.1:41234/explain/E030 …
/// ```
///
/// The URL is wrapped in an OSC 8 hyperlink so terminals that support it
/// make it clickable.  Frames are redrawn in place with a carriage return;
/// the caller stops the thread via [`stop_explain_spinner`], which clears
/// the line so the explanation text that follows stands alone.  Callers
/// must gate spawning on stderr being a terminal.
fn spawn_explain_spinner(
    url: &str,
    code: &str,
) -> (std::thread::JoinHandle<()>, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(80);

    // OSC 8 hyperlink: ST (`ESC \`) terminator, not BEL — the diagnostics
    // layer's sanitizer rules out BEL for good reason, and ST is the
    // better-supported form anyway.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_bg = stop.clone();
    let url = url.to_string();
    let code = code.to_string();
    let handle = std::thread::spawn(move || {
        let mut stderr = std::io::stderr().lock();
        for frame in FRAMES.iter().cycle() {
            if stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let _ = write!(
                stderr,
                "\r{frame} opening {code} explanation on \
                 \x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\ …\x1b[K"
            );
            let _ = stderr.flush();
            std::thread::sleep(FRAME_INTERVAL);
        }
    });
    (handle, stop)
}

/// Stop the spinner thread and erase its line from the terminal.
fn stop_explain_spinner(handle: std::thread::JoinHandle<()>, stop: &std::sync::atomic::AtomicBool) {
    // The thread checks the flag every frame (~80 ms), so the join below
    // cannot block meaningfully.
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = handle.join();
    let mut stderr = std::io::stderr().lock();
    let _ = write!(stderr, "\r\x1b[2K");
    let _ = stderr.flush();
}

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
