//! Multi-file loading and parallel parsing.
//!
//! Pipeline role: given a module plan (`ModulePath ↔ PathBuf` pairs),
//! read each file through a [`VfsBackend`], register it in the
//! [`SourceMap`] (idempotently), and parse it on its own arena.
//! Parsing is embarrassingly parallel — the `SourceMap` is
//! concurrency-safe by design, and every file gets a private arena, so
//! the only shared write point is `Symbol::intern` (sharded, safe).
//!
//! Lifetime strategy: the source text and each file's arena are
//! `Box::leak`ed and tracked by the returned [`LoadedFile`]; the
//! session drops the `LoadedFile`s (and thereby frees the arenas) when
//! it ends.  This mirrors the `Box::leak` pattern already used
//! throughout the checker tests.

use crate::ast::{FileId, Program, Stmt};
use crate::source_map::{ModulePath, SourceMap};
use crate::vfs::VfsBackend;
use crate::diagnostics::Diagnostic;
use crate::hir::types::Edition;
use crate::parser::Parser;
use crate::symbol::Symbol;
use std::path::{Path, PathBuf};

/// One parsed file plus everything needed to keep its memory alive
/// until session end.
pub struct LoadedFile {
    /// The `SourceMap` id this file was registered under.
    pub id: FileId,
    /// Parsed program (`'static` because text and arena are leaked).
    pub program: Program<'static, 'static>,
    /// The file's arena handle (`SyncArena` is `Send + Sync`, so parsed
    /// programs can travel out of a scoped thread); the session drops
    /// the `LoadedFile`s when it ends.
    pub _arena: &'static crate::arena::SyncArena<'static>,
}

/// A fatal parse failure, kept out of band: the caller decides how to
/// surface it (typically push onto the diagnostic context).
pub struct LoadOutcome {
    pub loaded: Vec<LoadedFile>,
    /// Fatal diagnostics per failed file, in plan order.
    pub fatal: Vec<Diagnostic<'static>>,
}

/// Read, register, and parse one file.
///
/// Returns `None` only when the file cannot be read or registered
/// (registration failures leave the dummy placeholder the `SourceMap`
/// documented — diagnostics must tolerate the "hole").  A fatal parse
/// error still yields a `LoadedFile` with an error program plus the
/// diagnostics in `fatal`, so a single broken file does not contaminate
/// the rest of the tree (parser recovery + sink nodes).
pub fn load_and_parse(
    backend: &dyn VfsBackend,
    sourcemap: &SourceMap,
    module: ModulePath,
    path: PathBuf,
    default_edition: Edition,
) -> (Option<LoadedFile>, Vec<Diagnostic<'static>>) {
    let (raw, mtime) = match backend.read_file_with_mtime(&path.to_string_lossy()) {
        Ok(pair) => pair,
        Err(e) => {
            return (
                None,
                vec![Diagnostic::fatal(format!(
                    "cannot read '{}': {:?}",
                    path.display(),
                    e
                ))],
            );
        }
    };
    let id = match sourcemap.insert_or_get(
        path.clone(),
        module,
        raw.as_bytes(),
        default_edition,
        mtime,
    ) {
        Ok(id) => id,
        Err(e) => {
            return (
                None,
                vec![Diagnostic::fatal(format!(
                    "cannot register '{}': {e}",
                    path.display()
                ))],
            );
        }
    };

    // Session-lifetime text + private arena per file (Box::leak pattern
    // as in the checker tests; `LoadedFile` tracks both for the session
    // to release).
    let text: &'static str = Box::leak(raw.into_boxed_str());
    let bump: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
    let arena: &'static crate::arena::SyncArena<'static> =
        Box::leak(Box::new(crate::arena::SyncArena::new(bump)));

    let mut parser = Parser::new(text, id, arena);
    let program = match parser.parse_program() {
        Ok(program) => program,
        Err(diags) => {
            // Keep going with an error program: resolver skips error items.
            let program = Program {
                items: Vec::new(),
                span: crate::ast::Span::new(id, 0, 0),
                doc_map: crate::ast::DocMap {
                    file: id,
                    ..Default::default()
                },
            };
            return (
                Some(LoadedFile { id, program, _arena: arena }),
                coerce_diags(diags),
            );
        }
    };
    (Some(LoadedFile { id, program, _arena: arena }), Vec::new())
}

/// Erase the borrow the parser hangs on the (leaked) source text.
fn coerce_diags(diags: Vec<Diagnostic<'_>>) -> Vec<Diagnostic<'static>> {
    // SAFETY: the diagnostics borrow only the file's source text, which
    // was `Box::leak`ed for the whole session above, so every borrow is
    // valid for `'static` as far as any observer can tell.  Kept in one
    // place so an owned-`Diagnostic` refactor touches a single fn.
    diags
        .into_iter()
        .map(|d| unsafe { std::mem::transmute::<Diagnostic<'_>, Diagnostic<'static>>(d) })
        .collect()
}

/// Parse a whole module plan, in parallel.
///
/// Results keep plan order regardless of thread completion order, so
/// downstream consumers see a deterministic file sequence; sort
/// diagnostics by span for full determinism before emitting.
pub fn load_tree(
    backend: &(impl VfsBackend + Sync),
    sourcemap: &SourceMap,
    plan: Vec<(ModulePath, PathBuf)>,
    default_edition: Edition,
) -> LoadOutcome {
    let sourcemap = &sourcemap;
    let backend = &backend;
    let mut loaded: Vec<Option<LoadedFile>> = (0..plan.len()).map(|_| None).collect();
    let mut fatal: Vec<Vec<Diagnostic<'static>>> = (0..plan.len()).map(|_| Vec::new()).collect();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (idx, (module, path)) in plan.into_iter().enumerate() {
            handles.push((
                idx,
                s.spawn(move || load_and_parse(*backend, sourcemap, module, path, default_edition)),
            ));
        }
        for (idx, handle) in handles {
            let (file, diags) = handle.join().unwrap_or((None, Vec::new()));
            loaded[idx] = file;
            fatal[idx] = diags;
        }
    });
    LoadOutcome {
        loaded: loaded.into_iter().flatten().collect(),
        fatal: fatal.into_iter().flatten().collect(),
    }
}

/// One `import` statement, as an edge in the module import graph.
#[derive(Debug, Clone)]
pub struct ImportSite {
    /// The file containing the `import`.
    pub from: FileId,
    /// Byte span of the whole import statement (primary label anchor).
    pub span: crate::ast::Span,
    /// The `a::b::c` module path segments, in order.
    pub path: Vec<Symbol>,
    /// `import a::b as x` — the local alias.
    pub alias: Option<Symbol>,
    /// `import a::b::{x, y}` — the imported items, if any.
    pub items: Option<Vec<Symbol>>,
}

/// The canonical `::`-joined module path an import refers to.
pub fn site_module_path(site: &ImportSite) -> ModulePath {
    let joined = site
        .path
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("::");
    ModulePath::new(joined)
}

/// Collect the import graph edges of a program (top-level items only —
/// `import` is a module-level statement in Posita).
pub fn import_sites(prog: &Program<'_, '_>) -> Vec<ImportSite> {
    let mut sites = Vec::new();
    for item in &prog.items {
        if let Stmt::Import { path, items, alias, span } = item {
            sites.push(ImportSite {
                from: span.file,
                span: *span,
                path: path.clone(),
                alias: *alias,
                items: items.clone(),
            });
        }
    }
    sites
}

/// Resolve every import site against the `SourceMap`, reporting:
///
/// - `E-import-unresolved` when the target module is not registered,
///   with a "did you mean" help line for the closest registered module
///   path (OSA distance ≤ 2, `MaybeIncorrect`-style suggestion);
/// - `E-import-cycle` — one diagnostic per cyclic strongly-connected
///   component, carrying a secondary "imported here" label at every
///   site on the cycle (cross-file labels; the renderer groups them by
///   file).
///
/// Deterministic: diagnostics come back sorted by (file, start, end).
pub fn resolve_and_check(
    programs: &[&Program<'_, '_>],
    sourcemap: &SourceMap,
) -> Vec<Diagnostic<'static>> {
    let all_sites: Vec<Vec<ImportSite>> =
        programs.iter().map(|p| import_sites(p)).collect();
    let flat: Vec<&ImportSite> = all_sites.iter().flatten().collect();

    let mut diags = Vec::new();

    // 1) Module resolution — record which module each site resolves to
    //    (unresolved sites are dropped from the graph so the cycle pass
    //    only sees real edges).
    let mut edges: Vec<(FileId, FileId, &ImportSite)> = Vec::new();
    for site in &flat {
        let target = site_module_path(site);
        match sourcemap.file_id_by_module(&target) {
            Some(to) => edges.push((site.from, to, site)),
            None => {
                let mut diag = Diagnostic::error(format!(
                    "unresolved module `{target}`"
                ))
                .with_code_str("E-import-unresolved")
                .with_span(site.span)
                .with_help(format!("no module `{target}` is registered in this build"));
                if let Some(near) = closest_module(&target, sourcemap) {
                    diag = diag.with_suggestion(format!(
                        "a similarly named module exists: `{near}` — did you mean it?"
                    ));
                }
                diags.push(coerce_one(diag));
            }
        }
    }

    // 2) Cycle detection — iterative Tarjan SCC over the module graph.
    for cycle in cycles_in(&edges) {
        // `cycle` is a list of import sites forming the ring (site i's
        // target imports site i+1's source).  One primary anchor, one
        // secondary label per contributing site.
        let mut diag = Diagnostic::error("cyclic module import")
            .with_code_str("E-import-cycle");
        let mut first = true;
        for site in &cycle {
            let msg = format!("`{}` imports `{}`", site_module_path(site), {
                // the module this site's file lives in
                let from = sourcemap.file(site.from).module_path.clone();
                from.to_string()
            });
            diag = if first {
                first = false;
                diag.with_span(site.span).with_label(site.span, msg)
            } else {
                diag.with_secondary_label(site.span, msg)
            };
        }
        diag = diag.with_help(
            "break the cycle by moving shared declarations into a lower-level module",
        );
        diags.push(coerce_one(diag));
    }

    diags.sort_by(|a, b| {
        let sa = a.spans().primary().first().copied();
        let sb = b.spans().primary().first().copied();
        sa.cmp(&sb)
    });
    diags
}

fn coerce_one(d: Diagnostic<'_>) -> Diagnostic<'static> {
    // SAFETY: the message strings are owned `String`s and no source text
    // has been attached yet at load time, so the `'src` parameter is
    // vacuous here.  Single conversion point, as in `coerce_diags`.
    unsafe { std::mem::transmute::<Diagnostic<'_>, Diagnostic<'static>>(d) }
}

/// The registered module path closest to `name` (OSA distance ≤ 2),
/// or `None`.  Ties break on the lexicographically smaller path so the
/// suggestion is deterministic.
fn closest_module(name: &ModulePath, sourcemap: &SourceMap) -> Option<String> {
    let chars: Vec<char> = name.as_str().chars().collect();
    let mut best: Option<(usize, String)> = None;
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    let mut p3 = Vec::new();
    for module in sourcemap.module_paths() {
        if module == *name {
            continue;
        }
        let cand: Vec<char> = module.as_str().chars().collect();
        let d = crate::parser::osa_distance_limited_into(
            &mut p1, &mut p2, &mut p3, &chars, &cand, 2,
        );
        if d <= 2 && best.as_ref().map_or(true, |(bd, _)| d < *bd || (d == *bd && module.as_str() < best.as_ref().unwrap().1.as_str())) {
            best = Some((d, module.as_str().to_string()));
        }
    }
    best.map(|(_, m)| m)
}

/// The import cycles among `edges` (from-file, to-file, site), as lists
/// of the sites forming each ring.  Iterative Tarjan SCC; a cycle is an
/// SCC with more than one node or a single node with a self-edge.
fn cycles_in(edges: &[(FileId, FileId, &ImportSite)]) -> Vec<Vec<ImportSite>> {
    use std::collections::HashMap;
    // Build adjacency: from-file → (to-file, site).
    let mut adj: HashMap<FileId, Vec<(FileId, &ImportSite)>> = HashMap::new();
    let mut self_loop: Vec<FileId> = Vec::new();
    for (from, to, site) in edges {
        if from == to {
            self_loop.push(*from);
        }
        adj.entry(*from).or_default().push((*to, site));
    }

    // Iterative Tarjan.
    let mut index: HashMap<FileId, usize> = HashMap::new();
    let mut low: HashMap<FileId, usize> = HashMap::new();
    let mut on_stack: rustc_hash::FxHashSet<FileId> = Default::default();
    let mut stack: Vec<FileId> = Vec::new();
    let mut sccs: Vec<Vec<FileId>> = Vec::new();
    let mut next_index = 0usize;

    let mut roots: Vec<FileId> = adj.keys().copied().collect();
    roots.sort_unstable_by_key(|f| f.as_u32());
    for root in roots {
        if index.contains_key(&root) {
            continue;
        }
        // Frame: (node, next child slot).
        let mut work: Vec<(FileId, usize)> = vec![(root, 0)];
        while let Some((v, mut ci)) = work.pop() {
            if ci == 0 {
                index.insert(v, next_index);
                low.insert(v, next_index);
                next_index += 1;
                stack.push(v);
                on_stack.insert(v);
            }
            let children = adj.get(&v).cloned().unwrap_or_default();
            let mut recursed = false;
            while ci < children.len() {
                let (w, _) = children[ci];
                ci += 1;
                if !index.contains_key(&w) {
                    work.push((v, ci));
                    work.push((w, 0));
                    recursed = true;
                    break;
                } else if on_stack.contains(&w) {
                    let li = low[&v].min(index[&w]);
                    low.insert(v, li);
                }
            }
            if recursed {
                continue;
            }
            // All children processed: pop the SCC if v is a root.
            if low[&v] == index[&v] {
                let mut scc = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack.remove(&w);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                sccs.push(scc);
            }
            if let Some((parent, _)) = work.last().copied() {
                let li = low[&parent].min(low[&v]);
                low.insert(parent, li);
            }
        }
    }

    // Materialise rings: for each cyclic SCC pick one import site per
    // edge so the diagnostic can label every hop.  Walk the SCC's edges
    // in adjacency order to produce a site sequence.
    let mut out = Vec::new();
    for scc in &sccs {
        let multi = scc.len() > 1;
        let single_self = scc.len() == 1 && self_loop.contains(&scc[0]);
        if !multi && !single_self {
            continue;
        }
        let set: rustc_hash::FxHashSet<FileId> = scc.iter().copied().collect();
        let mut ring: Vec<ImportSite> = Vec::new();
        let mut visited: rustc_hash::FxHashSet<FileId> = Default::default();
        // Start at the smallest file id for determinism; follow in-SCC
        // edges until we return to start or exhaust.
        let start = *scc.iter().min_by_key(|f| f.as_u32()).unwrap();
        let mut cur = start;
        loop {
            visited.insert(cur);
            let next = adj
                .get(&cur)
                .and_then(|outs| {
                    outs.iter()
                        .filter(|(w, _)| set.contains(w))
                        .find(|(w, _)| {
                            // New node, or the edge that closes the ring
                            // back to start (a self-loop is the degenerate
                            // case: scc.len() == 1, closable immediately).
                            !visited.contains(w)
                                || (*w == start && ring.len() + 1 >= scc.len())
                        })
                        .copied()
                });
            match next {
                Some((w, site)) => {
                    ring.push(site.clone());
                    if w == start {
                        break;
                    }
                    cur = w;
                }
                None => break,
            }
        }
        if !ring.is_empty() {
            out.push(ring);
        }
    }
    out.sort_by_key(|ring| ring[0].span.start);
    out
}

/// Convenience: build a plan by walking `root` for `*.pos` files whose
/// module path is their `::`-joined path relative to `root` (file-style
/// modules, no wildcards — `posita.toml`'s job is done by the caller).
pub fn plan_from_dir(
    backend: &dyn VfsBackend,
    root: &Path,
) -> Result<Vec<(ModulePath, PathBuf)>, String> {
    let mut plan = Vec::new();
    fn walk(
        backend: &dyn VfsBackend,
        dir: &Path,
        prefix: &str,
        out: &mut Vec<(ModulePath, PathBuf)>,
    ) -> Result<(), String> {
        for entry in backend
            .read_dir(&dir.to_string_lossy())
            .map_err(|e| format!("cannot scan '{}': {:?}", dir.display(), e))?
        {
            let child = dir.join(&entry.name);
            match entry.kind {
                crate::vfs::VfsNodeKind::Directory
                | crate::vfs::VfsNodeKind::Root
                | crate::vfs::VfsNodeKind::SrcDirectory => {
                    let sub = if prefix.is_empty() {
                        entry.name.clone()
                    } else {
                        format!("{prefix}::{}", entry.name)
                    };
                    walk(backend, &child, &sub, out)?;
                }
                crate::vfs::VfsNodeKind::File => {
                    if entry.name.ends_with(".pos") {
                        let stem = entry.name.trim_end_matches(".pos");
                        let module = if prefix.is_empty() {
                            stem.to_string()
                        } else {
                            format!("{prefix}::{stem}")
                        };
                        out.push((ModulePath::new(module), child));
                    }
                }
            }
        }
        Ok(())
    }
    walk(backend, root, "", &mut plan)?;
    plan.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(plan)
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;
    use crate::diagnostics::AnnotationKind;
    use crate::vfs::MemoryBackend;
    use std::sync::Arc;

    fn backend() -> Arc<MemoryBackend> {
        let mut b = MemoryBackend::new();
        b.insert_file(
            "/src/main.pos",
            "import utils::math;\ndef main() -> Int<32> { return 0; }",
        );
        b.insert_file("/src/utils/math.pos", "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }");
        Arc::new(b)
    }

    #[test]
    fn load_tree_parses_all_files_and_keeps_plan_order() {
        let b = backend();
        let sm = SourceMap::new();
        let plan = plan_from_dir(b.as_ref(), Path::new("/src")).expect("plan");
        assert_eq!(plan.len(), 2);

        let outcome = load_tree(b.as_ref(), &sm, plan, Edition::Year2026);
        assert!(outcome.fatal.is_empty(), "no fatal diags: {:?}", outcome.fatal);
        assert_eq!(outcome.loaded.len(), 2);
        // Plan order (sorted by path): main.pos before utils/math.pos.
        assert_eq!(outcome.loaded[0].program.items.len(), 2); // import + def
        assert_eq!(outcome.loaded[1].program.items.len(), 1);

        // Both registered in the SourceMap under their module paths.
        assert!(sm.file_id_by_module(&ModulePath::new("main")).is_some());
        assert!(sm.file_id_by_module(&ModulePath::new("utils::math")).is_some());
    }

    #[test]
    fn import_sites_capture_edges() {
        let b = backend();
        let sm = SourceMap::new();
        let plan = plan_from_dir(b.as_ref(), Path::new("/src")).expect("plan");
        let outcome = load_tree(b.as_ref(), &sm, plan, Edition::Year2026);
        let main = &outcome.loaded[0];
        let sites = import_sites(&main.program);
        assert_eq!(sites.len(), 1);
        assert_eq!(site_module_path(&sites[0]), ModulePath::new("utils::math"));
        assert!(sites[0].items.is_none());
        assert!(sites[0].alias.is_none());
        assert_eq!(sites[0].from, main.id);
    }

    #[test]
    fn parallel_matches_sequential() {
        // Parse the same plan sequentially (single-element chunks) and
        // in one parallel batch; item counts and module registrations
        // must agree exactly.
        let b = backend();
        let plan = plan_from_dir(b.as_ref(), Path::new("/src")).expect("plan");

        let sm_seq = SourceMap::new();
        let mut seq_counts = Vec::new();
        for (m, p) in plan.clone() {
            let (loaded, fatal) =
                load_and_parse(b.as_ref(), &sm_seq, m, p, Edition::Year2026);
            assert!(fatal.is_empty());
            seq_counts.push(loaded.expect("sequential load succeeded").program.items.len());
        }

        let sm_par = SourceMap::new();
        let outcome = load_tree(b.as_ref(), &sm_par, plan, Edition::Year2026);
        let par_counts: Vec<_> = outcome.loaded.iter().map(|l| l.program.items.len()).collect();
        assert_eq!(seq_counts, par_counts);
    }

    #[test]
    fn fatal_parse_error_does_not_contaminate_other_files() {
        let mut b = MemoryBackend::new();
        b.insert_file("/src/good.pos", "def f() -> Int<32> { return 1; }");
        b.insert_file("/src/bad.pos", "def g( {"); // broken syntax
        let sm = SourceMap::new();
        let plan = vec![
            (ModulePath::new("good"), PathBuf::from("/src/good.pos")),
            (ModulePath::new("bad"), PathBuf::from("/src/bad.pos")),
        ];
        let outcome = load_tree(&b, &sm, plan, Edition::Year2026);
        assert!(!outcome.fatal.is_empty(), "bad.pos must produce fatal diags");
        assert_eq!(outcome.loaded.len(), 2, "both files yield programs");
        let good = outcome
            .loaded
            .iter()
            .find(|l| sm.file(l.id).module_path == ModulePath::new("good"))
            .expect("good.pos loaded");
        assert_eq!(good.program.items.len(), 1);
    }

    fn parse_registered(b: &MemoryBackend, sm: &SourceMap, path: &str, module: &str) -> LoadedFile {
        let raw = b.read_file(path).expect("file exists");
        let id = sm
            .insert_or_get(PathBuf::from(path), ModulePath::new(module), raw.as_bytes(), Edition::Year2026, None)
            .expect("register");
        let text: &'static str = Box::leak(raw.into_boxed_str());
        let bump: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let arena: &'static crate::arena::SyncArena<'static> =
            Box::leak(Box::new(crate::arena::SyncArena::new(bump)));
        let mut parser = Parser::new(text, id, arena);
        let program = parser.parse_program().expect("parse ok");
        LoadedFile { id, program, _arena: arena }
    }

    #[test]
    fn unresolved_import_reports_and_suggests() {
        let mut b = MemoryBackend::new();
        b.insert_file("/src/a.pos", "import utils::mathh;\ndef f() -> Int<32> { return 0; }");
        b.insert_file("/src/utils/math.pos", "def g() -> Int<32> { return 1; }");
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let _u = parse_registered(&b, &sm, "/src/utils/math.pos", "utils::math");

        let diags = resolve_and_check(&[&a.program], &sm);
        assert_eq!(diags.len(), 1, "{:?}", diags);
        assert!(
            diags[0].message().contains("unresolved module `utils::mathh`"),
            "{:?}",
            diags[0]
        );
        assert_eq!(
            diags[0].code(),
            Some(&crate::diagnostics::ErrCode::new("E-import-unresolved"))
        );
        assert!(
            diags[0]
                .suggestions()
                .iter()
                .any(|s| format!("{s:?}").contains("utils::math")),
            "did-you-mean missing: {:?}",
            diags[0].suggestions()
        );
    }

    #[test]
    fn import_cycle_reports_one_diag_with_secondary_labels() {
        // a imports b, b imports a.
        let mut b = MemoryBackend::new();
        b.insert_file("/src/a.pos", "import b;\ndef fa() -> Int<32> { return 0; }");
        b.insert_file("/src/b.pos", "import a;\ndef fb() -> Int<32> { return 0; }");
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let bb = parse_registered(&b, &sm, "/src/b.pos", "b");

        let diags = resolve_and_check(&[&a.program, &bb.program], &sm);
        assert_eq!(diags.len(), 1, "one cycle diag, got: {:?}", diags);
        assert_eq!(
            diags[0].code(),
            Some(&crate::diagnostics::ErrCode::new("E-import-cycle"))
        );
        // Primary + one secondary label per ring member.
        assert!(diags[0].spans().primary().len() >= 1);
        assert!(
            diags[0]
                .labels()
                .iter()
                .any(|l| l.kind == AnnotationKind::Secondary),
            "ring must carry secondary labels: {:?}",
            diags[0].labels()
        );
    }

    #[test]
    fn self_import_is_a_cycle() {
        let mut b = MemoryBackend::new();
        b.insert_file("/src/a.pos", "import a;\ndef f() -> Int<32> { return 0; }");
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let diags = resolve_and_check(&[&a.program], &sm);
        assert_eq!(diags.len(), 1);
        assert_eq!(
            diags[0].code(),
            Some(&crate::diagnostics::ErrCode::new("E-import-cycle"))
        );
    }

    #[test]
    fn clean_imports_produce_no_diagnostics() {
        let b = backend();
        let sm = SourceMap::new();
        let plan = plan_from_dir(b.as_ref(), Path::new("/src")).expect("plan");
        let outcome = load_tree(b.as_ref(), &sm, plan, Edition::Year2026);
        let refs: Vec<&Program<'_, '_>> = outcome.loaded.iter().map(|l| &l.program).collect();
        let diags = resolve_and_check(&refs, &sm);
        assert!(diags.is_empty(), "clean tree must have no import diags: {:?}", diags);
    }
}
