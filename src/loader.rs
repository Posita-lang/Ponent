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
//! `Box::leak`ed and tracked by the returned [`LoadedFile`].  `Box::leak` is
//! permanent by definition, so dropping a `LoadedFile` frees **nothing**:
//! this memory is reclaimed only at process exit.  That is deliberate for the
//! one-shot compiler driver (parsing is the last consumer), but a long-lived
//! LSP session that reloads files must account for it — see the `LoadedFile`
//! doc for the escape-hatch caveat.  This mirrors the `Box::leak` pattern
//! already used throughout the checker tests.

use crate::ast::{FileId, Item, Program};
use crate::diagnostics::Diagnostic;
use crate::hir::types::Edition;
use crate::parser::Parser;
use crate::source_map::{ModulePath, SourceMap};
use crate::symbol::Symbol;
use crate::vfs::VfsBackend;
use std::path::{Path, PathBuf};

/// One parsed file plus everything needed to keep its memory alive
/// for the lifetime of the process.
///
/// All three fields are process-lifetime: `program` borrows the leaked text
/// and arena, and `_arena` is a *reference* to memory that was leaked (so
/// dropping this struct frees nothing).  Because `program` is `pub`, a caller
/// can smuggle `'static` borrows out; safety of any future reclamation
/// scheme therefore depends on sealing that field first.
pub struct LoadedFile {
    /// The `SourceMap` id this file was registered under.
    pub id: FileId,
    /// Parsed program (`'static` because text and arena are leaked).
    pub program: Program<'static, 'static>,
    /// The file's arena handle (`SyncArena` is `Send + Sync`, so parsed
    /// programs can travel out of a worker thread).  Leaked, not owned:
    /// dropping this handle does not free the arena.
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
    let (raw, mtime) = match backend.read_file_bytes_with_mtime(&path.to_string_lossy()) {
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
    // Byte-exact: a file that is not valid UTF-8 gets a diagnostic instead of
    // a silent U+FFFD decode (which would make every byte span wrong relative
    // to the on-disk file).
    let text_owned = match String::from_utf8(raw) {
        Ok(text) => text,
        Err(e) => {
            let at = e.utf8_error().valid_up_to();
            return (
                None,
                vec![Diagnostic::fatal(format!(
                    "invalid UTF-8 in '{}' at byte {at}",
                    path.display()
                ))],
            );
        }
    };
    let id = match sourcemap.insert_or_get(
        path.clone(),
        module,
        // The validated text's bytes are the original file bytes.
        text_owned.as_bytes(),
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
    let text: &'static str = Box::leak(text_owned.into_boxed_str());
    let bump: &'static mut bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
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
            // `text` was leaked to `'static` above, so the parser was
            // instantiated at `'static` and `diags` already has the
            // session lifetime — no coercion needed.
            return (
                Some(LoadedFile {
                    id,
                    program,
                    _arena: arena,
                }),
                diags,
            );
        }
    };
    (
        Some(LoadedFile {
            id,
            program,
            _arena: arena,
        }),
        Vec::new(),
    )
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
    // Deterministic duplicate-module handling: keep the lexicographically
    // smallest path per module and report the rest, instead of letting
    // `insert_or_get` decide (nondeterministically under parallel load) which
    // file imports resolve to.
    let mut order: Vec<usize> = (0..plan.len()).collect();
    order.sort_by(|&a, &b| {
        plan[a]
            .0
            .as_str()
            .cmp(plan[b].0.as_str())
            .then_with(|| plan[a].1.cmp(&plan[b].1))
    });
    let mut keep = vec![false; plan.len()];
    let mut last_kept: Option<usize> = None;
    let mut duplicates: Vec<Diagnostic<'static>> = Vec::new();
    for &idx in &order {
        match last_kept {
            Some(prev) if plan[prev].0 == plan[idx].0 => {
                duplicates.push(
                    Diagnostic::error(format!("duplicate module `{}`", plan[idx].0))
                        .with_code_str("E-module-duplicate")
                        .with_help(format!(
                            "already provided by '{}'; this file is ignored for import resolution",
                            plan[prev].1.display()
                        )),
                );
            }
            _ => {
                keep[idx] = true;
                last_kept = Some(idx);
            }
        }
    }
    let plan: Vec<(ModulePath, PathBuf)> = plan
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, entry)| entry)
        .collect();
    // One worker per file is unbounded: a 10 000-file monorepo spawned
    // 10 000 OS threads (tens of MB of stack reservations each) and paid
    // spawn cost that dominated small parses.  Rayon's indexed `collect` over
    // a `Vec` preserves plan order, so determinism is unchanged, and the pool
    // is bounded.  Each map is wrapped in `catch_unwind` so a panicking parser
    // still becomes a diagnostic instead of aborting the whole load.
    use rayon::prelude::*;
    let results: Vec<(Option<LoadedFile>, Vec<Diagnostic<'static>>)> = plan
        .into_par_iter()
        .map(|(module, path)| {
            let display = path.display().to_string();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_and_parse(*backend, sourcemap, module, path, default_edition)
            })) {
                Ok(result) => result,
                Err(_) => (
                    None,
                    vec![Diagnostic::fatal(format!(
                        "internal error: parser panicked while loading '{display}'"
                    ))],
                ),
            }
        })
        .collect();
    let mut loaded: Vec<LoadedFile> = Vec::with_capacity(results.len());
    let mut fatal: Vec<Diagnostic<'static>> = Vec::new();
    for (file, diags) in results {
        if let Some(file) = file {
            loaded.push(file);
        }
        fatal.extend(diags);
    }
    fatal.extend(duplicates);
    LoadOutcome { loaded, fatal }
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
    /// The canonical `::`-joined target, computed once at construction so
    /// resolution and cycle labels do not rebuild it per use.
    pub module: ModulePath,
    /// `import a::b as x` — the local alias.
    pub alias: Option<Symbol>,
    /// `import a::b::{x, y}` — the imported items, if any.
    pub items: Option<Vec<Symbol>>,
}

/// The canonical `::`-joined module path an import refers to.
pub fn site_module_path(site: &ImportSite) -> ModulePath {
    site.module.clone()
}

/// Collect the import graph edges of a program (top-level items only —
/// `import` is a module-level statement in Posita).
pub fn import_sites(prog: &Program<'_, '_>) -> Vec<ImportSite> {
    let mut sites = Vec::new();
    for item in &prog.items {
        if let Item::Import {
            path,
            items,
            alias,
            span,
        } = item
        {
            sites.push(ImportSite {
                from: span.file,
                span: *span,
                module: ModulePath::new(
                    path.iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join("::"),
                ),
                path: path.to_vec(),
                alias: *alias,
                items: items.map(|i| i.to_vec()),
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
    let all_sites: Vec<Vec<ImportSite>> = programs.iter().map(|p| import_sites(p)).collect();
    let flat: Vec<&ImportSite> = all_sites.iter().flatten().collect();

    let mut diags = Vec::new();

    // Target program per file id, and a lazily-built export-name set per
    // file: `import m::{x}` used to check only that `m` exists, silently
    // accepting a non-existent item.
    let by_id: rustc_hash::FxHashMap<FileId, &Program<'_, '_>> =
        programs.iter().map(|p| (p.span.file, *p)).collect();
    let mut export_names: rustc_hash::FxHashMap<FileId, rustc_hash::FxHashSet<Symbol>> =
        Default::default();

    // 1) Module resolution — record which module each site resolves to
    //    (unresolved sites are dropped from the graph so the cycle pass
    //    only sees real edges).
    let mut edges: Vec<(FileId, FileId, &ImportSite)> = Vec::new();
    for site in &flat {
        let target = &site.module;
        match sourcemap.file_id_by_module(target) {
            Some(to) => {
                edges.push((site.from, to, site));
                if let Some(items) = &site.items
                    && let Some(prog) = by_id.get(&to)
                {
                    let names = export_names
                        .entry(to)
                        .or_insert_with(|| prog.items.iter().filter_map(item_name).collect());
                    for item in items {
                        if names.contains(item) {
                            continue;
                        }
                        let mut diag = Diagnostic::error(format!(
                            "module `{target}` has no item `{}`",
                            item.as_str()
                        ))
                        .with_code_str("E-import-no-item")
                        .with_span(site.span)
                        .with_help(format!("`{target}` does not export `{}`", item.as_str()));
                        if let Some(near) = closest_item(*item, names) {
                            diag = diag.with_suggestion(format!(
                                "a similarly named item exists: `{near}` — did you mean it?"
                            ));
                        }
                        diags.push(diag);
                    }
                }
            }
            None => {
                let mut diag = Diagnostic::error(format!("unresolved module `{target}`"))
                    .with_code_str("E-import-unresolved")
                    .with_span(site.span)
                    .with_help(format!("no module `{target}` is registered in this build"));
                if let Some(near) = closest_module(target, sourcemap) {
                    diag = diag.with_suggestion(format!(
                        "a similarly named module exists: `{near}` — did you mean it?"
                    ));
                }
                diags.push(diag);
            }
        }
    }

    // 2) Cycle detection — iterative Tarjan SCC over the module graph.
    for cycle in cycles_in(&edges) {
        // `cycle` is every import site whose edge lies inside one cyclic SCC
        // (not necessarily a simple ring: SCCs are not Hamiltonian in general).
        // One primary anchor, one secondary label per contributing site.
        let mut diag = Diagnostic::error("cyclic module import").with_code_str("E-import-cycle");
        let mut first = true;
        // Per-file "this file's module" memo: the previous loop rebuilt the
        // same `ModulePath` string once per label.
        let mut from_names: rustc_hash::FxHashMap<FileId, String> = Default::default();
        for site in &cycle {
            let from_name = from_names
                .entry(site.from)
                .or_insert_with(|| sourcemap.file_or_dummy(site.from).module_path.to_string());
            let msg = format!("`{}` imports `{}`", &site.module, from_name);
            diag = if first {
                first = false;
                diag.with_span(site.span).with_label(site.span, msg)
            } else {
                diag.with_secondary_label(site.span, msg)
            };
        }
        diag = diag
            .with_help("break the cycle by moving shared declarations into a lower-level module");
        diags.push(diag);
    }

    // Explicit `(file, start, end)` key: relying on `Span`'s derived `Ord`
    // made the documented order depend on field order, and `None`-first was
    // accidental.
    diags.sort_by_key(|d| {
        d.spans()
            .primary_ref()
            .first()
            .map(|s| (s.file.as_u32(), s.start, s.end))
    });
    diags
}

/// The registered module path closest to `name` (OSA distance ≤ 2),
/// or `None`.  Ties break on the lexicographically smaller path so the
/// suggestion is deterministic.
fn closest_module(name: &ModulePath, sourcemap: &SourceMap) -> Option<String> {
    use smallvec::SmallVec;
    let chars: Vec<char> = name.as_str().chars().collect();
    let mut best: Option<(usize, String)> = None;
    let mut p1: SmallVec<[usize; 16]> = SmallVec::new();
    let mut p2: SmallVec<[usize; 16]> = SmallVec::new();
    let mut p3: SmallVec<[usize; 16]> = SmallVec::new();
    for module in sourcemap.module_paths() {
        if module == *name {
            continue;
        }
        // OSA distance is at least the char-length difference: reject far
        // candidates before collecting them into a `Vec<char>` (the common
        // case, and it removes an allocation per candidate).
        let cand_len = module.as_str().chars().count();
        if cand_len.abs_diff(chars.len()) > 2 {
            continue;
        }
        let cand: Vec<char> = module.as_str().chars().collect();
        let d =
            crate::parser::osa_distance_limited_into(&mut p1, &mut p2, &mut p3, &chars, &cand, 2);
        if d <= 2
            && best.as_ref().map_or(true, |(bd, _)| {
                d < *bd || (d == *bd && module.as_str() < best.as_ref().unwrap().1.as_str())
            })
        {
            best = Some((d, module.as_str().to_string()));
        }
    }
    best.map(|(_, m)| m)
}

/// The name an item exports at module scope, if it has one.
fn item_name(item: &Item<'_>) -> Option<Symbol> {
    match item {
        Item::FunctionDef { name, .. }
        | Item::TypeDef { name, .. }
        | Item::TraitDef { name, .. }
        | Item::ExternFunction { name, .. }
        | Item::Constraint { name, .. }
        | Item::LayoutDef { name, .. } => Some(*name),
        _ => None,
    }
}

/// The exported name closest to `item` (OSA distance ≤ 2) for a
/// did-you-mean suggestion; ties break lexicographically for determinism.
fn closest_item(item: Symbol, names: &rustc_hash::FxHashSet<Symbol>) -> Option<String> {
    use smallvec::SmallVec;
    let chars: Vec<char> = item.as_str().chars().collect();
    let mut p1: SmallVec<[usize; 16]> = SmallVec::new();
    let mut p2: SmallVec<[usize; 16]> = SmallVec::new();
    let mut p3: SmallVec<[usize; 16]> = SmallVec::new();
    let mut best: Option<(usize, &str)> = None;
    for name in names {
        let cand: Vec<char> = name.as_str().chars().collect();
        if cand.len().abs_diff(chars.len()) > 2 {
            continue;
        }
        let d =
            crate::parser::osa_distance_limited_into(&mut p1, &mut p2, &mut p3, &chars, &cand, 2);
        if d <= 2
            && best.as_ref().map_or(true, |(bd, bs)| {
                d < *bd || (d == *bd && name.as_str() < *bs)
            })
        {
            best = Some((d, name.as_str()));
        }
    }
    best.map(|(_, name)| name.to_string())
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
            // Borrow the adjacency slice: cloning it on every pop made the
            // DFS O(V·E) worst case with one allocation per pop.
            let children: &[(FileId, &ImportSite)] = adj.get(&v).map(Vec::as_slice).unwrap_or(&[]);
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

    // Materialise one site list per cyclic SCC.  An SCC need not have a
    // Hamiltonian cycle (A→B, B→A, B→C, C→B is one SCC with no simple cycle
    // through all nodes), so walking a "ring" silently dropped edges.  Every
    // intra-SCC import edge contributes a label instead.
    let mut out = Vec::new();
    for scc in &sccs {
        let multi = scc.len() > 1;
        let single_self = scc.len() == 1 && self_loop.contains(&scc[0]);
        if !multi && !single_self {
            continue;
        }
        let set: rustc_hash::FxHashSet<FileId> = scc.iter().copied().collect();
        let mut sites: Vec<ImportSite> = edges
            .iter()
            .filter(|(from, to, _)| set.contains(from) && set.contains(to))
            .map(|(_, _, site)| (*site).clone())
            .collect();
        sites.sort_by_key(|s| (s.span.file.as_u32(), s.span.start, s.span.end));
        sites.dedup_by_key(|s| (s.span.file.as_u32(), s.span.start, s.span.end));
        if !sites.is_empty() {
            out.push(sites);
        }
    }
    out.sort_by_key(|sites| sites[0].span.start);
    out
}

/// Convenience: build a plan by walking `root` for `*.pos` files whose
/// module path is their `::`-joined path relative to `root` (file-style
/// modules, no wildcards — `posita.toml`'s job is done by the caller).
pub fn plan_from_dir(
    backend: &dyn VfsBackend,
    root: &Path,
) -> Result<Vec<(ModulePath, PathBuf)>, String> {
    /// Guard against symlink cycles and pathological nesting: the walk is an
    /// explicit stack, so a deep tree neither overflows the native stack nor
    /// spins forever.
    const MAX_DEPTH: usize = 256;
    fn join_module(prefix: &str, name: &str) -> String {
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}::{name}")
        }
    }

    let mut plan = Vec::new();
    let mut stack: Vec<(PathBuf, String, usize)> = vec![(root.to_path_buf(), String::new(), 0)];
    while let Some((dir, prefix, depth)) = stack.pop() {
        if depth > MAX_DEPTH {
            return Err(format!(
                "directory nesting exceeds {MAX_DEPTH} at '{}'",
                dir.display()
            ));
        }
        for entry in backend
            .read_dir(&dir.to_string_lossy())
            .map_err(|e| format!("cannot scan '{}': {:?}", dir.display(), e))?
        {
            let child = dir.join(&entry.name);
            match entry.kind {
                crate::vfs::VfsNodeKind::Directory
                | crate::vfs::VfsNodeKind::Root
                | crate::vfs::VfsNodeKind::SrcDirectory => {
                    stack.push((child, join_module(&prefix, &entry.name), depth + 1));
                }
                crate::vfs::VfsNodeKind::File => {
                    if let Some(stem) = entry.name.strip_suffix(".pos") {
                        // `mod.pos` names its *parent* module (`utils/mod.pos`
                        // is `utils`), mirroring `main.pos` for the root; a
                        // root-level `mod.pos` has no module name to claim.
                        let module = if stem == "mod" {
                            if prefix.is_empty() {
                                continue;
                            }
                            prefix.clone()
                        } else {
                            join_module(&prefix, stem)
                        };
                        plan.push((ModulePath::new(module), child));
                    }
                }
            }
        }
    }
    // Sort by module path, not by `Path`: `Path`'s order is component-wise and
    // platform-sensitive (separators/case), while the module path is what
    // downstream consumers key on.
    plan.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()).then_with(|| a.1.cmp(&b.1)));
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
        b.insert_file(
            "/src/utils/math.pos",
            "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }",
        );
        Arc::new(b)
    }

    #[test]
    fn load_tree_parses_all_files_and_keeps_plan_order() {
        let b = backend();
        let sm = SourceMap::new();
        let plan = plan_from_dir(b.as_ref(), Path::new("/src")).expect("plan");
        assert_eq!(plan.len(), 2);

        let outcome = load_tree(b.as_ref(), &sm, plan, Edition::Year2026);
        assert!(
            outcome.fatal.is_empty(),
            "no fatal diags: {:?}",
            outcome.fatal
        );
        assert_eq!(outcome.loaded.len(), 2);
        // Plan order (sorted by path): main.pos before utils/math.pos.
        assert_eq!(outcome.loaded[0].program.items.len(), 2); // import + def
        assert_eq!(outcome.loaded[1].program.items.len(), 1);

        // Both registered in the SourceMap under their module paths.
        assert!(sm.file_id_by_module(&ModulePath::new("main")).is_some());
        assert!(
            sm.file_id_by_module(&ModulePath::new("utils::math"))
                .is_some()
        );
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
            let (loaded, fatal) = load_and_parse(b.as_ref(), &sm_seq, m, p, Edition::Year2026);
            assert!(fatal.is_empty());
            seq_counts.push(
                loaded
                    .expect("sequential load succeeded")
                    .program
                    .items
                    .len(),
            );
        }

        let sm_par = SourceMap::new();
        let outcome = load_tree(b.as_ref(), &sm_par, plan, Edition::Year2026);
        let par_counts: Vec<_> = outcome
            .loaded
            .iter()
            .map(|l| l.program.items.len())
            .collect();
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
        assert!(
            !outcome.fatal.is_empty(),
            "bad.pos must produce fatal diags"
        );
        assert_eq!(outcome.loaded.len(), 2, "both files yield programs");
        let good = outcome
            .loaded
            .iter()
            .find(|l| sm.file_or_dummy(l.id).module_path == ModulePath::new("good"))
            .expect("good.pos loaded");
        assert_eq!(good.program.items.len(), 1);
    }

    fn parse_registered(b: &MemoryBackend, sm: &SourceMap, path: &str, module: &str) -> LoadedFile {
        let raw = b.read_file(path).expect("file exists");
        let id = sm
            .insert_or_get(
                PathBuf::from(path),
                ModulePath::new(module),
                raw.as_bytes(),
                Edition::Year2026,
                None,
            )
            .expect("register");
        let text: &'static str = Box::leak(raw.into_boxed_str());
        let bump: &'static mut bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let arena: &'static crate::arena::SyncArena<'static> =
            Box::leak(Box::new(crate::arena::SyncArena::new(bump)));
        let mut parser = Parser::new(text, id, arena);
        let program = parser.parse_program().expect("parse ok");
        LoadedFile {
            id,
            program,
            _arena: arena,
        }
    }

    #[test]
    fn unresolved_import_reports_and_suggests() {
        let mut b = MemoryBackend::new();
        b.insert_file(
            "/src/a.pos",
            "import utils::mathh;\ndef f() -> Int<32> { return 0; }",
        );
        b.insert_file("/src/utils/math.pos", "def g() -> Int<32> { return 1; }");
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let _u = parse_registered(&b, &sm, "/src/utils/math.pos", "utils::math");

        let diags = resolve_and_check(&[&a.program], &sm);
        assert_eq!(diags.len(), 1, "{:?}", diags);
        assert!(
            diags[0]
                .message()
                .contains("unresolved module `utils::mathh`"),
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
        assert!(diags[0].spans().primary_ref().len() >= 1);
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
        assert!(
            diags.is_empty(),
            "clean tree must have no import diags: {:?}",
            diags
        );
    }

    #[test]
    fn duplicate_module_paths_are_reported_and_deduplicated() {
        let mut b = MemoryBackend::new();
        b.insert_file("/src/one.pos", "def f() -> Int<32> { return 0; }");
        b.insert_file("/src/two.pos", "def g() -> Int<32> { return 1; }");
        let sm = SourceMap::new();
        // The lexicographically smallest path wins; the loser is reported
        // instead of letting parallel registration pick nondeterministically.
        let plan = vec![
            (ModulePath::new("dup"), PathBuf::from("/src/two.pos")),
            (ModulePath::new("dup"), PathBuf::from("/src/one.pos")),
        ];
        let outcome = load_tree(&b, &sm, plan, Edition::Year2026);
        assert_eq!(outcome.loaded.len(), 1, "one file kept per module");
        assert!(
            outcome
                .fatal
                .iter()
                .any(|d| d.message().contains("duplicate module `dup`")),
            "missing duplicate diagnostic: {:?}",
            outcome.fatal
        );
    }

    #[test]
    fn import_missing_item_reports_with_suggestion() {
        let mut b = MemoryBackend::new();
        b.insert_file(
            "/src/a.pos",
            "import utils::math::{ad};\ndef f() -> Int<32> { return 0; }",
        );
        b.insert_file(
            "/src/utils/math.pos",
            "def add(a: Int<32>, b: Int<32>) -> Int<32> { return a + b; }",
        );
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let m = parse_registered(&b, &sm, "/src/utils/math.pos", "utils::math");
        let diags = resolve_and_check(&[&a.program, &m.program], &sm);
        assert!(
            diags
                .iter()
                .any(|d| d.message().contains("has no item `ad`")),
            "missing item must be diagnosed: {:?}",
            diags
        );
        assert!(
            diags.iter().any(|d| d
                .suggestions()
                .iter()
                .any(|s| format!("{s:?}").contains("add"))),
            "did-you-mean for the item missing: {:?}",
            diags
        );
    }

    #[test]
    fn scc_without_hamiltonian_cycle_labels_every_edge() {
        // a→b, b→a, b→c, c→b is one SCC with *no* simple cycle through all
        // three nodes; every intra-SCC edge must still be labelled.
        let mut b = MemoryBackend::new();
        b.insert_file("/src/a.pos", "import b;\ndef fa() -> Int<32> { return 0; }");
        b.insert_file(
            "/src/b.pos",
            "import a;\nimport c;\ndef fb() -> Int<32> { return 0; }",
        );
        b.insert_file("/src/c.pos", "import b;\ndef fc() -> Int<32> { return 0; }");
        let sm = SourceMap::new();
        let a = parse_registered(&b, &sm, "/src/a.pos", "a");
        let bb = parse_registered(&b, &sm, "/src/b.pos", "b");
        let c = parse_registered(&b, &sm, "/src/c.pos", "c");
        let diags = resolve_and_check(&[&a.program, &bb.program, &c.program], &sm);
        assert_eq!(diags.len(), 1, "one SCC diag, got: {:?}", diags);
        assert_eq!(
            diags[0].labels().len(),
            4,
            "every intra-SCC edge must be labelled: {:?}",
            diags[0].labels()
        );
    }
}
