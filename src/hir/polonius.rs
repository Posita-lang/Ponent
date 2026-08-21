//! The Polonius cross-validation pipeline (the committee two-solver
//! comparison).
//!
//! Extracts the loan / liveness facts from a function body (the CFG + the
//! borrow sites + the access events) into the Polonius input schema and
//! evaluates the rules (R1-R9, hand-translated from the FlowLog
//! `polonius_int.dl`) IN PROCESS, so the point-level borrow-check
//! post-pass can be checked against the authoritative rules.
//!
//! # Why a custom implementation (deliberate, not a shortcut)
//! The rules are hand-translated into the crate ON PURPOSE: we may make
//! CUSTOM changes to Polonius itself down the line (Posita-specific rule
//! extensions — loan kinds, two-phase borrows, the signature derivation,
//! the `known_placeholder_subset` handling, ...).  An in-process Rust
//! rule set stays freely editable without being bound to the official
//! engine's internals; the official `polonius-engine` keeps serving as
//! the differential-testing oracle so any drift from the baseline rules
//! is caught by `test_oracle_official_engine_agrees`.
//!
//! # Enforcement authority (the ENGINE switch)
//! `polonius::borrow_check_function` is now the ENGINE path — it runs
//! `extract_facts` + `evaluate_rules` + `rules_to_borrow_errors`, which
//! produce the user-facing three-way diagnostics (E109 read-freeze /
//! E110 mutation-freeze / E112 exclusivity, with loan kind + span).
//! This module is BOTH the production engine AND the differential oracle
//! layer:
//!
//! - `evaluate_rules` / `evaluate_subset_errors` (R1-R9) are the
//!   production decision layer, cross-checked on every covered shape
//!   (`test_engine_switch_equivalence`);
//! - the OFFICIAL rustc `polonius-engine` (crates.io, dev-dependency) is
//!   itself used as an oracle against our translation
//!   (`test_oracle_official_engine_agrees`) — it shares none of our
//!   CFG/liveness infrastructure, so shared-infrastructure bugs cannot
//!   mask drift;
//! - `rules_to_borrow_errors` maps the engine output to
//!   `BorrowError`s (currently the E110 category), the bridge for a
//!   future authority switch once E109/E112 mapping and the name-keyed
//!   signature-registry collision (the DefId-keying follow-up) land.
//!
//! Remaining pre-switch work: (a) E109/E112 diagnostic mapping, (b) the
//! registry keyed by DefId instead of `Symbol`, (c) `var_defined_at` /
//! `var_dropped_at` and path-level facts (child_path/path_is_var) for
//! full `AllFacts` parity.

use crate::ast::UnaryOp;
use crate::hir::cfg_graph::{BlockId, CfgGraph, Point, collect_borrow_data};
use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::place::place_is_prefix_of;
use crate::hir::types::{FrozenPlace, LoanKind, SignatureFacts, TypeData, TypeId};
use crate::symbol::Symbol;
use std::collections::{HashMap, HashSet};

/// A point encoded as a plain integer (the polonius_int.dl point ids):
/// `block << 36 | stmt << 16 | expr` — a 64-bit encoding with generous
/// per-function budgets (1M blocks, 1M statements per block, 65536
/// expressions per statement) so that legitimate large statements (big
/// array/struct literals, wide `match` arms) never overflow.  The order
/// of the packed fields is lexicographic in (block, stmt, expr) — the
/// CFG order — which the extraction's point-ordering queries rely on
/// (`< ip` / `max_by_key`).  Returns `Err` (a friendly message) when the
/// encoding capacity is exceeded — an internal compiler limitation that
/// the extraction reports as E113 (the caller decides how to present it).
fn point_id(p: Point) -> Result<u64, String> {
    // Expression-level points: `block << 36 | stmt << 16 | expr` — the
    // low 16 bits are the expression index within a statement, so a
    // statement's write/read/borrow operations are individually ordered
    // (aligned with rustc's CFG points — a same-statement write-then-
    // reborrow is decidable without the R8 same-point exemption).
    if p.stmt >= (1 << 20) {
        return Err(format!(
            "the function's CFG exceeds the point-encoding capacity: statement index {} is beyond the 2^20 bound",
            p.stmt
        ));
    }
    if p.expr >= (1 << 16) {
        return Err(format!(
            "the function's CFG exceeds the point-encoding capacity: expression index {} is beyond the 2^16 bound",
            p.expr
        ));
    }
    if p.block.0 >= (1 << 20) {
        return Err(format!(
            "the function's CFG exceeds the point-encoding capacity: block index {} is beyond the 2^20 bound",
            p.block.0
        ));
    }
    Ok(((p.block.0 as u64) << 36) | ((p.stmt as u64) << 16) | (p.expr as u64))
}

/// The Polonius facts for one function body — the polonius_int.dl inputs
/// the current model needs (the remaining inputs — subset_base,
/// universal_region, loan_killed_at, known_placeholder_subset,
/// var_dropped_at, drop_of_var_derefs_origin, var_defined_at, child_path,
/// path_*_at_base, path_is_var — are empty in this model).
#[derive(Default)]
pub struct PoloniusFacts {
    pub cfg_edge: Vec<(u64, u64)>,
    pub loan_issued_at: Vec<(u32, u32, u64)>, // (loan, origin, point)
    pub loan_invalidated_at: Vec<(u32, u64)>, // (loan, point)
    pub var_used_at: Vec<(u32, u64)>,         // (var, point)
    pub use_of_var_derefs_origin: Vec<(u32, u32)>, // (var, origin)
    /// The reborrow parent/child pairs — (source_loan, reborrow_loan):
    /// the loans whose referent resolution (the `Deref(Root(v))` place
    /// rewrite) derived the child from the parent.  The E112 exclusivity
    /// check exempts the pair (a child derives from its parent and
    /// cannot conflict with it); the parent is killed at the child's
    /// issuance point (the reborrow-kill), so its liveness does not
    /// leak past the reborrow.  The official-engine oracle does not
    /// consume this vector (it has no Polonius equivalent — the path
    /// aliasing is a facts-level concern).
    pub reborrow_sources: Vec<(u32, u32)>,
    /// The per-point subset facts (the region abstraction) — the
    /// call-site instantiations (R1's base).
    pub subset_base: Vec<(u32, u32, u64)>, // (origin1, origin2, point)
    /// The placeholder (universal) input origins.
    pub universal_region: Vec<u32>,
    /// The DECLARED signature-level subset relations (the A(ρ)
    /// constraints from the `SignatureFacts` — R9's whitelist).
    pub known_placeholder_subset: Vec<(u32, u32)>,
    /// The loans KILLED at a point (the R5 `!loan_killed_at` condition) —
    /// a killed loan is no longer tracked (e.g. a reassignment clears it).
    pub loan_killed_at: Vec<(u32, u64)>,
    /// `var_defined_at(var, point)` — the variable is (re)initialized at
    /// `point` (a `set`/`def` binding or an assignment to the bare
    /// identifier — the R3 liveness restart point).
    pub var_defined_at: Vec<(u32, u64)>,
    /// `var_dropped_at(var, point)` — the variable's old value is dropped
    /// at `point` (an overwrite — the R3 liveness kill point).
    pub var_dropped_at: Vec<(u32, u64)>,
    /// `drop_of_var_derefs_origin(var, origin)` — the drop of `var`
    /// dereferences `origin` (the drop-side analog of
    /// `use_of_var_derefs_origin`).
    pub drop_of_var_derefs_origin: Vec<(u32, u32)>,
    /// `child_path(child, parent)` — the DIRECT path-parent relation
    /// (e.g. `x.y` → `x`; not `x.y.z` → `x`).
    pub child_path: Vec<(u32, u32)>,
    /// `path_is_var(path, var)` — the root path `path` starts at `var`.
    pub path_is_var: Vec<(u32, u32)>,
}

/// A loan with its Polonius ids (for the equivalence correlation).
#[derive(Clone, Debug)]
pub struct LoanInfo {
    pub id: u32,
    pub origin: u32,
    pub place: FrozenPlace,
    pub borrow_var: Option<Symbol>,
    pub kind: LoanKind,
    pub point: Point,
    /// The loan's source span (the diagnostic anchor for the
    /// rules→BorrowError mapping).
    pub span: crate::ast::Span,
    /// TWO-PHASE BORROW (TPB) marker: the loan is the receiver of a
    /// METHOD CALL (`v.push(...)` — `&mut v` is RESERVED until the
    /// arguments are evaluated; a read of `v` in the args, e.g.
    /// `v.len()`, must not conflict).  Mirrors the official Polonius
    /// `MutBorrowKind::TwoPhaseBorrow` / `Reservation` modelling.
    pub two_phase: bool,
}

// ────────────────────────────────────────────────────────────────────────
// The function-boundary region abstraction (aeneas `A(ρ)`).
// Signature-level facts: the input borrows are placeholder origins
// (`universal_region`), and the A(ρ) constraint "output borrow alive ⟹
// input borrow considered alive" is encoded as `subset_base(input,
// output)` — the input's loans propagate to the output origin (R4), so
// the input loan stays live wherever the output is live.
// ────────────────────────────────────────────────────────────────────────

/// A function's borrow signature — the resolved reference-ness of the
/// parameters and the return (the caller resolves the types).
#[derive(Clone, Debug, Default)]
pub struct BorrowSignature {
    /// The reference parameters (`&mut`/`&T`/`&ro` — the input origins /
    /// placeholders), each with its MUTABILITY (`&mut` = true) — the
    /// cross-function loans use it to issue `Exclusive` only for
    /// mutable inputs (read-only inputs get `ReadOnly` —
    /// previously every return-borrow froze the source exclusively).
    pub input_borrows: Vec<(Symbol, bool)>,
    /// The explicit lifetime of each input borrow (`&'a mut T` →
    /// `Some('a)`), aligned with `input_borrows` by index.  `None` = an
    /// elided/inferred lifetime.  The shared-lifetime grouping (task:
    /// early-bound UniversalRegions mapping) maps every reference with
    /// the SAME explicit `'a` to the SAME placeholder origin.
    pub input_lifetimes: Vec<Option<Symbol>>,
    /// The reference RETURN(s) — the output origins.  Plural: a
    /// function can return several references (a tuple/struct of refs);
    /// each gets its own placeholder origin and its own A(ρ) relation.
    /// (The rustc `UniversalRegions` analog: the output region(s) of the
    /// normalized signature — the earlier single-output model was the
    /// simplified form.)
    pub output_borrows: Vec<Symbol>,
}

/// Generate the signature facts from a borrow signature + the
/// output-derivation analysis (the aeneas `inst_sig` precision — NOT the
/// conservative "every output constrains every input" encoding).  All
/// input borrows are placeholder origins; the `known_placeholder_subset`
/// relations are ONLY the DERIVING inputs (the output borrow alive ⟹ the
/// deriving input considered alive).  The input origins are `0..n`; the
/// output origin is `n`.
pub fn signature_facts(sig: &BorrowSignature, deriving: &[usize]) -> SignatureFacts {
    let mut facts = SignatureFacts::default();
    let n = sig.input_borrows.len();
    // EARLY-BOUND mapping: every input with the SAME explicit lifetime
    // (`&'a mut T` — `input_lifetimes[i] == Some('a)`) shares ONE
    // placeholder origin (the rustc `UniversalRegions` early-bound
    // region); elided/inferred lifetimes (None) get their own origin.
    let mut origin_of: HashMap<Option<Symbol>, u32> = HashMap::new();
    let mut next_origin = 0u32;
    let mut input_origins: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let lt = sig.input_lifetimes.get(i).copied().flatten();
        let origin = match lt {
            Some(sym) => *origin_of.entry(Some(sym)).or_insert_with(|| {
                let o = next_origin;
                next_origin += 1;
                o
            }),
            None => {
                let o = next_origin;
                next_origin += 1;
                o
            }
        };
        facts.universal_region.push(origin);
        facts.input_borrow_mutable.push(sig.input_borrows[i].1);
        input_origins.push(origin);
    }
    // The output origins follow the (deduplicated) input origins — each
    // output is a distinct placeholder (out_start + k).
    let out_start = next_origin;
    for k in 0..sig.output_borrows.len() {
        facts.universal_region.push(out_start + k as u32);
    }
    // A(ρ): every deriving input flows into every output origin (the
    // cross-function returned-borrow freeze), using the input's ACTUAL
    // (possibly shared) origin.
    for i in 0..n {
        let o = input_origins[i];
        for k in 0..sig.output_borrows.len() {
            let out_origin = out_start + k as u32;
            if deriving.contains(&i) {
                facts.known_placeholder_subset.push((o, out_origin));
            }
        }
    }
    facts
}

/// The number of reference positions in a return type — the multi-output
/// extension of the single `return_is_ref` flag.  Recurses into tuples,
/// arrays/slices and struct (ADT) fields: every `&T`/`&mut T` position is
/// a distinct output origin (`__ret0`, `__ret1`, …).  Generic-parameter
/// fields are conservatively counted as 0 (an unresolved field cannot be
/// asserted to be a reference without the substitution).
pub fn count_return_refs(ctx: &crate::hir::types::TypeContext, ty: TypeId) -> usize {
    match ctx.get(ctx.resolve_binding(ty)) {
        TypeData::Ref { .. } => 1,
        TypeData::Tuple { elems } => elems.iter().map(|e| count_return_refs(ctx, *e)).sum(),
        TypeData::Array { elem, .. } | TypeData::Slice { elem } => count_return_refs(ctx, *elem),
        TypeData::Adt { def_id, args, .. } => ctx
            .adt_def(*def_id)
            .map(|def| {
                def.fields
                    .iter()
                    .map(|f| {
                        // Substitute the generic-parameter fields with
                        // the concrete args (the same pattern as
                        // `adt_field_is_copy`): a field `a: T` whose
                        // arg is a reference counts as an output.
                        let substituted = match ctx.get(*f) {
                            TypeData::GenericParam { index, .. } => {
                                args.get(*index).copied().unwrap_or(*f)
                            }
                            _ => *f,
                        };
                        count_return_refs(ctx, substituted)
                    })
                    .sum()
            })
            .unwrap_or(0),
        _ => 0,
    }
}

/// The AST-level count of reference positions in a return type — the
/// `pre_register_signatures` counterpart of `count_return_refs` (the
/// HIR-level resolution cannot run in the AST pre-pass, so the direct
/// `Type::Reference`/`Tuple`/`Array`/`Slice` shapes are counted here;
/// struct-path fields are left to the HIR-level pass).
pub fn count_return_refs_ast(ty: &crate::ast::Type) -> usize {
    match ty {
        crate::ast::Type::Reference { .. } => 1,
        crate::ast::Type::Tuple(elems, _) => elems.iter().map(count_return_refs_ast).sum(),
        crate::ast::Type::Array(elem, _, _) | crate::ast::Type::Slice(elem, _) => {
            count_return_refs_ast(elem)
        }
        _ => 0,
    }
}

/// Resolve a function's borrow signature: the reference parameters (the
/// input origins) + the reference return (the output origin).  The
/// per-parameter reference-ness is supplied by the caller (the checker
/// resolves `TypeData::Ref` via its context).
pub fn extract_borrow_signature(
    params: &[(Symbol, bool, bool, Option<Symbol>)],
    return_refs: usize,
) -> BorrowSignature {
    let mut sig = BorrowSignature::default();
    for (name, is_ref, mutable, lifetime) in params {
        if *is_ref {
            sig.input_borrows.push((*name, *mutable));
            sig.input_lifetimes.push(*lifetime);
        }
    }
    // Multi-output: every reference position in the return type gets its
    // own output origin (`__ret0`, `__ret1`, …) — a tuple/struct return
    // of several references is no longer collapsed to a single output.
    for k in 0..return_refs {
        sig.output_borrows
            .push(Symbol::intern(&format!("__ret{k}")));
    }
    sig
}

/// The call-site instantiation (aeneas `synthesize_function_call`).
/// At a call to a function with a borrow signature (the explicit
/// borrow parameters only — the `@auto_ro` path is untouched):
/// - the argument borrow at each input-borrow parameter position issues
///   its loan in the INPUT origin (the universal placeholder region),
/// - the DECLARED A(ρ) relations (`known_placeholder_subset(input,
///   output)`) become per-point `subset_base(input, output, call_point)`
///   facts — the instantiation that drives the R4 loan-via-subset
///   propagation ("output alive ⟹ input alive").
/// The cross-function extraction — the body facts + the call-site
/// integration (the calls to known-borrow-signature functions map their
/// argument borrows to the input origins + instantiate the A(ρ) subset
/// facts at the call points) + the callee signature facts merged.
/// `registry`: the (function name, input-borrow parameter positions, the
/// A(ρ) signature facts) triples — from the checker's `signature_facts`.
pub fn extract_cross_function<'input>(
    cfg: &CfgGraph<'input>,
    body: &[HirStmt<'input>],
    finally: &[HirStmt<'input>],
    registry: &[(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)],
    ctx: &crate::hir::types::TypeContext<'input>,
) -> Result<(PoloniusFacts, Vec<LoanInfo>), String> {
    let (mut facts, infos, _events) = extract_facts(cfg, body, finally, registry, ctx)?;
    // The call-site integration: for each statement, walk the expressions
    // for the calls to the known-signature functions.
    for (bi, blk) in cfg.blocks().iter().enumerate() {
        for (si, stmt) in blk.stmts.iter().enumerate() {
            let pt = point_id(Point {
                block: BlockId(bi),
                stmt: si,
                expr: 0,
            })?;
            collect_call_sites(stmt, pt, registry, &mut facts, infos.len() as u32);
        }
    }
    Ok((facts, infos))
}

/// Walk a statement's expressions for the calls to the known-signature
/// functions — the argument loans in the input origins + the A(ρ) subset
/// instantiations at the call point.
fn collect_call_sites<'input>(
    stmt: &HirStmt<'input>,
    pt: u64,
    registry: &[(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)],
    facts: &mut PoloniusFacts,
    base: u32,
) {
    let exprs: Vec<&HirExpr<'input>> = match stmt {
        HirStmt::VariableDef { value: Some(v), .. } => vec![v],
        HirStmt::Assign { value, .. } => vec![value],
        HirStmt::Return { value: Some(v), .. } => vec![v],
        HirStmt::Expression(e) => vec![e],
        _ => Vec::new(),
    };
    for e in exprs {
        walk_call_expr(e, pt, registry, facts, base);
    }
}

fn walk_call_expr<'input>(
    e: &HirExpr<'input>,
    pt: u64,
    registry: &[(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)],
    facts: &mut PoloniusFacts,
    base: u32,
) {
    match e {
        HirExpr::Call { callee, args, .. } => {
            // The callee's name — the registry lookup.
            if let HirExpr::Ident(name, _, _) = callee.as_ref() {
                if let Some((_, _, _, positions, sig)) = registry
                    .iter()
                    .find(|(n, is_m, _, _, _)| n == name && !is_m)
                {
                    let (loans, subset) = call_site_facts(
                        positions,
                        pt,
                        &sig.known_placeholder_subset,
                        // the call-site loan/origin base — the body's
                        // loan count, so the injected facts do not collide
                        // with the body's 0..K ID space.
                        base,
                    );
                    facts.loan_issued_at.extend(loans);
                    facts.subset_base.extend(subset);
                    // The callee's signature origins are shifted INTO the
                    // caller's region space ABOVE the body's 0..K loan
                    // origins (the same `base` offset as the injected
                    // loans) — WITHOUT the shift, the signature's
                    // placeholder origins (0..m) collide with the body
                    // loans' own origin ids (0..K), making an UNUSED
                    // returned-borrow loan accidentally universal: it
                    // stays live forever and a later mutation of the
                    // source is rejected (rustc accepts —
                    // `set r = get(&mut a); a = 5;` with `r` never used).
                    // The shifted universe also keeps the injected loans
                    // (base+i) disjoint from the universals' unshifted
                    // value space.
                    facts
                        .universal_region
                        .extend(sig.universal_region.iter().map(|o| o + base));
                    facts.known_placeholder_subset.extend(
                        sig.known_placeholder_subset
                            .iter()
                            .map(|&(i, o)| (i + base, o + base)),
                    );
                }
            }
            // Recurse into the arguments for nested calls.
            for a in args {
                walk_call_expr(a, pt, registry, facts, base);
            }
        }
        // Aggregates / composite expressions — recurse so a cross-function
        // call nested
        // inside a tuple, array, struct literal, binary op, index, move,
        // or unary op is issued (and later invalidated) like a top-level
        // call.  This keeps issuance and invalidation walks in sync.
        HirExpr::Tuple(elems, _, _) | HirExpr::Array(elems, _, _) => {
            for el in elems {
                walk_call_expr(el, pt, registry, facts, base);
            }
        }
        HirExpr::StructLit { fields, .. } => {
            for (_, v) in fields {
                walk_call_expr(v, pt, registry, facts, base);
            }
        }
        HirExpr::BinaryOp { left, right, .. } => {
            walk_call_expr(left, pt, registry, facts, base);
            walk_call_expr(right, pt, registry, facts, base);
        }
        HirExpr::Index {
            base: ib, index, ..
        } => {
            walk_call_expr(ib, pt, registry, facts, base);
            walk_call_expr(index, pt, registry, facts, base);
        }
        HirExpr::Move(inner, _, _) => {
            walk_call_expr(inner, pt, registry, facts, base);
        }
        HirExpr::UnaryOp { expr, .. } => {
            walk_call_expr(expr, pt, registry, facts, base);
        }
        _ => {}
    }
}

/// Returns the `(loan_issued_at, subset_base)` additions.
pub fn call_site_facts(
    input_positions: &[usize],
    call_point: u64,
    arho: &[(u32, u32)],
    base: u32,
) -> (Vec<(u32, u32, u64)>, Vec<(u32, u32, u64)>) {
    let mut loans = Vec::new();
    let mut subset = Vec::new();
    for (i, _pos) in input_positions.iter().enumerate() {
        // the call-site loans/origins are offset by `base` (the
        // body's loan count) — previously the parameter position was
        // used directly as the loan/origin ID, colliding with the
        // body's 0..K space (rules_to_borrow_errors attributed the
        // wrong loan to the wrong place/span).
        let input_origin = base + i as u32;
        loans.push((base + i as u32, input_origin, call_point));
        // The A(ρ) instantiation: subset_base(input, output, call_point).
        // The output origin is the callee's signature origin, shifted by
        // the same `base` as the caller-side universal extension (see
        // `walk_call_expr`) — keeping the subset's endpoints in the
        // shifted placeholder space.
        for &(ai, ao) in arho {
            if ai == i as u32 {
                subset.push((input_origin, ao + base, call_point));
            }
        }
    }
    (loans, subset)
}

/// The root symbol of a place chain (`x`, `x.f`, `*x`, `&mut *x.f`).
/// A fresh (non-place) value → `None`.
fn place_root<'input>(e: &HirExpr<'input>) -> Option<Symbol> {
    match e {
        HirExpr::Ident(name, _, _) => Some(*name),
        HirExpr::FieldAccess { base, .. } => place_root(base),
        HirExpr::Index { base, .. } => place_root(base),
        HirExpr::UnaryOp {
            op: UnaryOp::Deref | UnaryOp::Ref | UnaryOp::RefMut | UnaryOp::Ro,
            expr,
            ..
        } => place_root(expr),
        _ => None,
    }
}

/// The output-derivation analysis (the aeneas `inst_sig` precision —
/// NOT the conservative "every output constrains every input" encoding).
/// Determines which input parameters the returned borrow can derive from:
/// the return expression's place ROOT mapped to the parameter positions.
/// `params` is the FULL parameter list (names); the returned indices are
/// the deriving inputs (empty = the return is a fresh value with no input
/// derivation).
pub fn derive_output_origins<'input>(body: &[HirStmt<'input>], params: &[Symbol]) -> Vec<usize> {
    // The UNION of ALL the returns' derivation origins   + Issue
    // 2): the returns nested inside the control flow (if/while/for/loop
    // arms) are walked recursively — the aeneas `inst_sig` analysis
    // considers every exit path.  The returned borrow may also derive
    // through a LOCAL ALIAS of the input (`set r2 = &mut *r; return
    // r2;` — the reborrow): `collect_param_alias_map` maps each alias
    // local to the input it derives from, and the returns resolve
    // through it (the cross-function reborrow — the caller must freeze
    // the ORIGINAL referent, E0506 parity).
    let alias_map = collect_param_alias_map(body, params);
    let mut deriving = Vec::new();
    collect_returns(body, params, &alias_map, &mut deriving);
    deriving
}

/// The local → input mapping: every local variable whose value derives
/// from an input borrow (`set r2 = &mut *r` — the reborrow — or
/// `set suffix = ls` — a plain copy) maps to the input's parameter
/// index; chains resolve transitively (`set r3 = &mut *r2`).
fn collect_param_alias_map<'input>(
    stmts: &[HirStmt<'input>],
    params: &[Symbol],
) -> HashMap<Symbol, usize> {
    let mut map: HashMap<Symbol, usize> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        collect_alias_map_in_stmts(stmts, params, &mut map, &mut changed);
    }
    map
}

fn collect_alias_map_in_stmts<'input>(
    stmts: &[HirStmt<'input>],
    params: &[Symbol],
    map: &mut HashMap<Symbol, usize>,
    changed: &mut bool,
) {
    for stmt in stmts {
        match stmt {
            HirStmt::VariableDef {
                name: Some(n),
                value: Some(v),
                ..
            } => {
                if let Some(root) = place_root(v) {
                    let idx = params
                        .iter()
                        .position(|p| *p == root)
                        .or_else(|| map.get(&root).copied());
                    if let Some(i) = idx
                        && map.get(n) != Some(&i)
                    {
                        map.insert(*n, i);
                        *changed = true;
                    }
                }
            }
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            }
            | HirStmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_alias_map_in_stmts(then_branch, params, map, changed);
                if let Some(e) = else_branch {
                    collect_alias_map_in_stmts(e, params, map, changed);
                }
            }
            HirStmt::While { body, .. }
            | HirStmt::WhileLet { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Loop { body, .. } => {
                collect_alias_map_in_stmts(body, params, map, changed);
            }
            _ => {}
        }
    }
}

/// rustc's dangling-reference rejection: collect the spans of the
/// RETURNS that reference a LOCAL (non-parameter) place — the returned
/// borrow would DANGLE after the function returns (the borrow cannot
/// outlive the function's frame).  A local variable that ALIASES a
/// parameter (`set suffix = ls` — the value is the parameter's borrow)
/// is NOT dangling: it points at the parameter's data, which outlives
/// the function.
pub fn dangling_return_spans<'input>(
    body: &[HirStmt<'input>],
    params: &[Symbol],
) -> Vec<crate::ast::Span> {
    let aliases = collect_param_aliases(body, params);
    let mut spans = Vec::new();
    collect_dangling_returns(body, params, &aliases, &mut spans);
    spans
}

/// The transitive set of local variables aliasing a parameter
/// (`set suffix = ls` — the value is the parameter's borrow): returning
/// them is legal (the aliased data outlives the function).
fn collect_param_aliases<'input>(stmts: &[HirStmt<'input>], params: &[Symbol]) -> HashSet<Symbol> {
    let mut aliases: HashSet<Symbol> = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        collect_aliases_in_stmts(stmts, params, &mut aliases, &mut changed);
    }
    aliases
}

/// Recursive alias collection (mirrors the `collect_dangling_returns`
/// recursion): an alias defined INSIDE an `if`/loop body is still an
/// alias — the previous top-level-only scan missed them and wrongly
/// flagged their returns as dangling.
fn collect_aliases_in_stmts<'input>(
    stmts: &[HirStmt<'input>],
    params: &[Symbol],
    aliases: &mut HashSet<Symbol>,
    changed: &mut bool,
) {
    for stmt in stmts {
        match stmt {
            HirStmt::VariableDef {
                name: Some(n),
                value: Some(v),
                ..
            } => {
                if let Some(root) = place_root(v) {
                    let roots_param = params.iter().any(|p| *p == root);
                    if (roots_param || aliases.contains(&root)) && !aliases.contains(n) {
                        aliases.insert(*n);
                        *changed = true;
                    }
                }
            }
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            }
            | HirStmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_aliases_in_stmts(then_branch, params, aliases, changed);
                if let Some(e) = else_branch {
                    collect_aliases_in_stmts(e, params, aliases, changed);
                }
            }
            HirStmt::While { body, .. }
            | HirStmt::WhileLet { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Loop { body, .. } => {
                collect_aliases_in_stmts(body, params, aliases, changed);
            }
            _ => {}
        }
    }
}

fn collect_dangling_returns<'input>(
    stmts: &[HirStmt<'input>],
    params: &[Symbol],
    aliases: &HashSet<Symbol>,
    out: &mut Vec<crate::ast::Span>,
) {
    for stmt in stmts {
        match stmt {
            HirStmt::Return {
                value: Some(v),
                span,
                ..
            } => {
                if let Some(root) = place_root(v) {
                    let root_is_param = params.iter().any(|p| *p == root);
                    if !root_is_param && !aliases.contains(&root) {
                        out.push(*span);
                    }
                }
            }
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            }
            | HirStmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_dangling_returns(then_branch, params, aliases, out);
                if let Some(e) = else_branch {
                    collect_dangling_returns(e, params, aliases, out);
                }
            }
            // The loop bodies (mirrors `collect_returns`): a `return` of a
            // local inside a `loop`/`while`/`for` is equally dangling —
            // the previous `_ => {}` fallback silently missed them.
            HirStmt::While { body, .. }
            | HirStmt::WhileLet { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Loop { body, .. } => {
                collect_dangling_returns(body, params, aliases, out);
            }
            _ => {}
        }
    }
}

fn collect_returns<'input>(
    stmts: &[HirStmt<'input>],
    params: &[Symbol],
    alias_map: &HashMap<Symbol, usize>,
    out: &mut Vec<usize>,
) {
    for stmt in stmts {
        match stmt {
            HirStmt::Return { value: Some(v), .. } => {
                if let Some(root) = place_root(v) {
                    // The derivation resolves through the local ALIASES
                    // too: `return r2` where `r2 = &mut *r` derives from
                    // the input `r` (the cross-function reborrow).
                    let idx = params
                        .iter()
                        .position(|p| *p == root)
                        .or_else(|| alias_map.get(&root).copied());
                    if let Some(i) = idx {
                        if !out.contains(&i) {
                            out.push(i);
                        }
                    }
                } else {
                    // The return value is not a plain place — it is a
                    // composite/control-flow expression (a `match`/`if`/
                    // block).  Recurse so every arm/branch derivation is
                    // collected (the output-derivation union — a return
                    // nested inside a match arm must count).
                    collect_returns_in_expr(v, params, alias_map, out);
                }
            }
            HirStmt::If {
                then_branch,
                else_branch,
                ..
            }
            | HirStmt::IfLet {
                then_branch,
                else_branch,
                ..
            } => {
                collect_returns(then_branch, params, alias_map, out);
                if let Some(e) = else_branch {
                    collect_returns(e, params, alias_map, out);
                }
            }
            HirStmt::While { body, .. }
            | HirStmt::WhileLet { body, .. }
            | HirStmt::For { body, .. }
            | HirStmt::Loop { body, .. } => {
                collect_returns(body, params, alias_map, out);
            }
            HirStmt::Expression(e) => collect_returns_in_expr(e, params, alias_map, out),
            _ => {}
        }
    }
}

fn collect_returns_in_expr<'input>(
    e: &HirExpr<'input>,
    params: &[Symbol],
    alias_map: &HashMap<Symbol, usize>,
    out: &mut Vec<usize>,
) {
    // A single-place value (an Ident / field access / deref) derives from
    // its root — handled here so nested arms (`match` bodies, block tails)
    // that yield a plain place are collected.
    if let Some(root) = place_root(e) {
        let idx = params
            .iter()
            .position(|p| *p == root)
            .or_else(|| alias_map.get(&root).copied());
        if let Some(i) = idx {
            if !out.contains(&i) {
                out.push(i);
            }
        }
        return;
    }
    match e {
        HirExpr::If {
            then_branch,
            else_branch,
            ..
        }
        | HirExpr::IfLet {
            then_branch,
            else_branch,
            ..
        } => {
            collect_returns(then_branch, params, alias_map, out);
            if let Some(eb) = else_branch {
                collect_returns(eb, params, alias_map, out);
            }
        }
        // `return match ... { _ => x }` — the arm bodies are return paths;
        // each arm's body derivation is part of the output-derivation
        // union (the aeneas inst_sig precision).
        HirExpr::Match { arms, .. } => {
            for arm in arms {
                collect_returns_in_expr(&arm.body, params, alias_map, out);
            }
        }
        HirExpr::Block(stmts, _, _) => collect_returns(stmts, params, alias_map, out),
        _ => {}
    }
}

/// The ROOT variable of a place (`a.b[0]` → `a`) — the loan's SOURCE
/// variable (what the drop-while-borrowed rule checks).
fn place_root_var(p: &FrozenPlace) -> Option<Symbol> {
    match p {
        FrozenPlace::Root(v) => Some(*v),
        FrozenPlace::Field(base, _)
        | FrozenPlace::Index(base)
        | FrozenPlace::ConstIndex(base, _)
        | FrozenPlace::Deref(base) => place_root_var(base),
    }
}

/// Resolve a READ/WRITE event's place through the reborrow aliasing:
/// `*r` where `r` holds a borrow of `a` refers to the object `a` — the
/// event's place is rewritten to the CURRENT loan's resolved place (the
/// referent).  The ORIGINAL root variable is returned alongside (the
/// own-use check: an access through the loan's own borrow variable is
/// the loan's intended use, not a conflicting access), plus a flag
/// whether the event WAS rewritten (a deref through a live borrow
/// variable — only rewritten events are eligible for the own-use
/// exemption; a plain ROOT read while an exclusive loan on the same
/// root is live is a genuine conflict — the E0502-family freeze, which
/// the TPB reservation window alone exempts).
fn resolve_event_place(
    place: &FrozenPlace,
    pt: &Point,
    infos: &[LoanInfo],
) -> Result<(FrozenPlace, Option<Symbol>, bool), String> {
    let root = place_root_var(place);
    let FrozenPlace::Deref(base) = place else {
        return Ok((place.clone(), root, false));
    };
    let FrozenPlace::Root(v) = base.as_ref() else {
        return Ok((place.clone(), root, false));
    };
    let ip = point_id(*pt)?;
    // Precompute the issuance ids ONCE (fail-closed: any overflow aborts
    // via `?` instead of silently degrading to point 0 — the closures
    // cannot use `?`).
    let mut candidates: Vec<(usize, u64)> = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        if info.borrow_var == Some(*v) && point_id(info.point)? < ip {
            candidates.push((i, point_id(info.point)?));
        }
    }
    match candidates.into_iter().max_by_key(|(_, id)| *id) {
        Some((i, _)) => Ok((infos[i].place.clone(), root, true)),
        None => Ok((place.clone(), root, false)),
    }
}

/// Extract the Polonius facts from a function body.
/// Collect the METHOD-CALL receiver places (`v.push(...)` — the base of a
/// `FieldAccess` callee), for the TPB marking of the receiver loans.
fn collect_method_receiver_places<'input>(
    stmts: &[HirStmt<'input>],
    out: &mut HashSet<FrozenPlace>,
) {
    for s in stmts {
        match s {
            HirStmt::VariableDef { value: Some(v), .. } => collect_expr_receiver(v, out),
            HirStmt::Assign { value, .. } => collect_expr_receiver(value, out),
            HirStmt::Expression(e) => collect_expr_receiver(e, out),
            HirStmt::Return { value: Some(v), .. } => collect_expr_receiver(v, out),
            _ => {}
        }
    }
}

fn collect_expr_receiver<'input>(e: &HirExpr<'input>, out: &mut HashSet<FrozenPlace>) {
    match e {
        HirExpr::Call { callee, args, .. } => {
            if let HirExpr::FieldAccess { base, .. } = callee.as_ref()
                && let Some(p) = crate::hir::cfg_graph::hir_expr_place(base)
            {
                out.insert(p);
            }
            collect_expr_receiver(callee, out);
            for a in args {
                collect_expr_receiver(a, out);
            }
        }
        _ => {}
    }
}

pub fn extract_facts<'input>(
    cfg: &CfgGraph<'input>,
    body: &[HirStmt<'input>],
    finally: &[HirStmt<'input>],
    registry: &[(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)],
    ctx: &crate::hir::types::TypeContext<'input>,
) -> Result<
    (
        PoloniusFacts,
        Vec<LoanInfo>,
        Vec<(FrozenPlace, Point, bool, crate::ast::Span)>,
    ),
    String,
> {
    let live = cfg.compute_point_liveness();
    let (loans, events, kills, counts) = collect_borrow_data(cfg, registry, ctx);
    let mut facts = PoloniusFacts::default();
    // cfg_edge: the EXPRESSION-LEVEL chains — within each statement the
    // head (expr 0) → first expression → ... → the last expression
    // (intra-statement ordering), then the statement → the next
    // statement's head; the terminator's head → its condition chain →
    // each successor's entry.  A same-statement write-then-reborrow is
    // therefore decidable WITHOUT the statement-level same-point
    // exemptions (the official polonius engine keeps the R4-R8 rules
    // pure — the precision lives in the facts).
    for b in 0..cfg.blocks().len() {
        let blk = BlockId(b);
        let n = cfg.block(blk).stmts.len();
        for i in 0..n {
            let e = *counts.get(&(b, i)).unwrap_or(&1);
            for k in 0..e - 1 {
                facts.cfg_edge.push((
                    point_id(Point {
                        block: blk,
                        stmt: i,
                        expr: k,
                    })?,
                    point_id(Point {
                        block: blk,
                        stmt: i,
                        expr: k + 1,
                    })?,
                ));
            }
            facts.cfg_edge.push((
                point_id(Point {
                    block: blk,
                    stmt: i,
                    expr: e - 1,
                })?,
                point_id(Point {
                    block: blk,
                    stmt: i + 1,
                    expr: 0,
                })?,
            ));
        }
        // The terminator's condition chain (if present), then the
        // terminator → each successor's entry.
        let et = *counts.get(&(b, n)).unwrap_or(&1);
        for k in 0..et - 1 {
            facts.cfg_edge.push((
                point_id(Point {
                    block: blk,
                    stmt: n,
                    expr: k,
                })?,
                point_id(Point {
                    block: blk,
                    stmt: n,
                    expr: k + 1,
                })?,
            ));
        }
        let term = point_id(Point {
            block: blk,
            stmt: n,
            expr: et - 1,
        })?;
        for &s in cfg.successors(blk) {
            facts.cfg_edge.push((
                term,
                point_id(Point {
                    block: s,
                    stmt: 0,
                    expr: 0,
                })?,
            ));
        }
    }
    // The loans → ids + origins + the variable facts.
    let mut infos = Vec::new();
    let mut temp_counter = 1_000_000u32;
    // TPB (Two-Phase Borrow): collect the METHOD-CALL receiver places —
    // the receiver loans (`v.push(...)` — the implicit `&mut v`) are
    // marked `two_phase` so their reserved-phase reads are exempt (the
    // E109 exemption in `rules_to_borrow_errors`).
    let mut receiver_places: HashSet<FrozenPlace> = HashSet::new();
    collect_method_receiver_places(body, &mut receiver_places);
    let mut reborrow_sources: Vec<(u32, u32)> = Vec::new();
    // REBORROW PLACE RESOLUTION (the referent aliasing): a `&mut *r`
    // borrow — and the cross-function call-return loan of a bare-reference
    // argument (`set r3 = get2(r)`, wrapped as `Deref(Root(r))` by the
    // caller) — borrows the ULTIMATE REFERENT of the deref chain: `*r`
    // where `r` holds a borrow of `a` is the object `a` itself.  The
    // loans are registered on the literal `Deref(Root(r))` path, which
    // never prefix-overlaps the referent's writes (`a = 5`) or reads
    // (`*r`) — the mutation/read-freeze checks would miss the referent
    // entirely, accepting shapes rustc rejects (E0506/E0503: a reborrow
    // freezes the referent while the reborrow is live).
    //
    // The resolution walks the borrow-variable chain: the current loan
    // of `v` (the latest loan bound to `v` issued before the resolving
    // loan) is the referent's borrow — its PLACE is the referent.  The
    // reborrow also KILLS that source loan at the child's issuance
    // point (rustc's reborrow-kill: the child takes over the referent's
    // borrow), and the (parent, child) pair is recorded in
    // `reborrow_sources` so the E112 exclusivity check exempts it.
    let mut resolved_places: Vec<FrozenPlace> = Vec::with_capacity(loans.len());
    for (i, (place, _var, _kind, pt, _span, _is_pb)) in loans.iter().enumerate() {
        let resolved = match place {
            FrozenPlace::Deref(base) => match base.as_ref() {
                FrozenPlace::Root(v) => {
                    // The current loan of `v` at `pt`: the LATEST loan
                    // bound to `v` issued strictly before `pt` (a loan
                    // cannot reborrow its own issuance point; the
                    // same-statement temporary loans are bound to `None`
                    // and never match).
                    let ip = point_id(*pt)?;
                    // Precompute the prior loan points ONCE (fail-closed:
                    // any overflow aborts the extraction via `?` instead
                    // of silently degrading to point 0).
                    let mut prior: Vec<(usize, Option<Symbol>, u64)> = Vec::with_capacity(i);
                    for (j, (_, var, _, jpt, _, _)) in loans[..i].iter().enumerate() {
                        prior.push((j, *var, point_id(*jpt)?));
                    }
                    match prior
                        .iter()
                        .filter(|(_, var, jid)| *var == Some(*v) && *jid < ip)
                        .max_by_key(|(j, _, _)| *j)
                    {
                        Some((j, _, _)) => {
                            // A reborrow RESOLVES through the parent loan
                            // but does NOT terminate it (rustc
                            // `loan_kills.rs` kills loans only on
                            // `StorageDead` and on assignment — never at a
                            // reborrow; SoundBorrowChecking.md:173-176 —
                            // ℓ0 survives ℓ1, it must be ended
                            // separately).  The parent stays live until
                            // its own borrow variable's last use (the
                            // last-use kill vector below), so a write to
                            // the referent while the parent borrow var is
                            // still used afterwards is frozen (E0506).
                            // `reborrow_sources` keeps the E112 exemption
                            // between the parent/child pair.
                            reborrow_sources.push((*j as u32, i as u32));
                            resolved_places[*j].clone()
                        }
                        None => place.clone(),
                    }
                }
                _ => place.clone(),
            },
            _ => place.clone(),
        };
        resolved_places.push(resolved);
    }
    for (i, (place, var, kind, pt, span, _is_pb)) in loans.iter().enumerate() {
        let id = i as u32;
        let origin = i as u32;
        match var {
            Some(v) => {
                // The borrow variable's per-point uses → var_used_at —
                // at the STATEMENT's FINAL expression point (the use
                // covers the whole statement: the backward liveness
                // closure reaches every expression of the statement and
                // the preceding ones).
                for (bi, blk_uses) in live.var_uses().iter().enumerate() {
                    for (si, used) in blk_uses.iter().enumerate() {
                        if used.contains(v) {
                            let e = *counts.get(&(bi, si)).unwrap_or(&1);
                            facts.var_used_at.push((
                                v.to_u32(),
                                point_id(Point {
                                    block: BlockId(bi),
                                    stmt: si,
                                    expr: e - 1,
                                })?,
                            ));
                        }
                    }
                }
                facts.use_of_var_derefs_origin.push((v.to_u32(), origin));
                // Every loan dies at its borrow variable's last use
                // (matching rust's NLL/Polonius: a loan flows along the
                // origin's liveness and stops at its last use), so a
                // mutation after that is accepted.  A loop iteration's
                // borrow therefore does NOT freeze the NEXT iteration —
                // `a = 5` at the top of a loop after `let r = &mut a`
                // in the previous iteration is legal (rustc NLL accepts
                // it).  A Field/Index loan (borrowed content is destroyed,
                // not rebound) is excluded — same AncestorClobber
                // discipline as the write-event logic: it lives on for
                // the E110 invalidation.
                if let Some(&(_, last_pt)) = facts
                    .var_used_at
                    .iter()
                    .filter(|(vv, _)| *vv == v.to_u32())
                    .max_by_key(|(_, p)| *p)
                    && !matches!(
                        place,
                        FrozenPlace::Field(..)
                            | FrozenPlace::Index(..)
                            | FrozenPlace::ConstIndex(..)
                    )
                {
                    facts.loan_killed_at.push((id, last_pt));
                }
            }
            None => {
                // A temporary loan: a fresh variable used at the END of
                // its statement — the loan is live through the whole
                // statement (its expression ordering: a later argument
                // read still sees an earlier `&mut` argument loan — the
                // `take(&mut a, a)` E109), and dies at the statement's
                // final point.
                let tv = temp_counter;
                temp_counter += 1;
                let stmt_expr = *counts.get(&(pt.block.0, pt.stmt)).unwrap_or(&1);
                let end_pt = point_id(Point {
                    block: pt.block,
                    stmt: pt.stmt,
                    expr: stmt_expr - 1,
                })?;
                facts.var_used_at.push((tv, end_pt));
                facts.use_of_var_derefs_origin.push((tv, origin));
                // A borrow-operand temporary (`&mut a` passed to a call)
                // lives only THROUGH the statement that contains the
                // borrow: the caller-side loan dies at the statement's
                // END (the call has returned — rustc NLL ends the
                // argument loan at the call expression).  Previously the
                // temporary loan followed its origin's liveness to the
                // END of the function, over-freezing mutations after the
                // call (an E110 false positive — e.g. `set r =
                // get(&mut a); let x = *r; a = 5;` rejected the trailing
                // mutation).
                // The AncestorClobber exclusion (mirroring the mutation-
                // event logic): a Field/Index temporary loan is NOT
                // killed at the borrow statement — the borrowed content
                // is destroyed, not rebound — it lives on (the E110
                // invalidation handles it).
                if !matches!(
                    place,
                    FrozenPlace::Field(..) | FrozenPlace::Index(..) | FrozenPlace::ConstIndex(..)
                ) {
                    facts.loan_killed_at.push((id, end_pt));
                }
            }
        }
        facts.loan_issued_at.push((id, origin, point_id(*pt)?));
        infos.push(LoanInfo {
            id,
            origin,
            place: resolved_places[i].clone(),
            borrow_var: *var,
            kind: *kind,
            point: *pt,
            span: *span,
            two_phase: receiver_places.contains(place),
        });
    }
    // The reborrow-KILLS case: an assignment to the
    // borrow VARIABLE (`r = r2`) kills the OLD loan — the kill target has
    // no prefix relation with the loan source, so the mutation-event
    // logic above cannot reconstruct it.  Previously the kills vector was
    // dropped (`_`), so the old source stayed frozen past the reborrow
    // (a behaviour drift from the handwritten pass).
    for (kplace, kvar, kpt) in &kills {
        for info in &infos {
            let var_matches = match kvar {
                Some(k) => {
                    // Only the covered OLD loan (issued in an EARLIER
                    // statement) is killed by the reborrow; the FRESH
                    // loan of the same expression (`r = &mut b`'s loan —
                    // same statement as the assignment) is the reborrow's
                    // NEW borrow and must stay alive until the borrow
                    // variable's last use (so a later mutation of `b` is
                    // still rejected — E110).
                    info.borrow_var.as_ref() == Some(k)
                        && !(info.point.block == kpt.block && info.point.stmt == kpt.stmt)
                }
                None => {
                    // The same AncestorClobber exclusion as the mutation-
                    // event logic above: a clobber of a FIELD/INDEX
                    // ancestor (`arr = ...` clobbers the loan on `arr[1]`)
                    // must NOT kill the loan — the borrowed content is
                    // destroyed, not rebound.
                    let strict_prefix =
                        place_is_prefix_of(kplace, &info.place) && kplace != &info.place;
                    if strict_prefix
                        && matches!(
                            &info.place,
                            FrozenPlace::Field(..)
                                | FrozenPlace::Index(..)
                                | FrozenPlace::ConstIndex(..)
                        )
                    {
                        false
                    } else {
                        // A write TO the borrowed path itself (kplace ==
                        // info.place) is NOT a kill — it INVALIDATES the
                        // loan (the mutation-freeze E110 is reported by the
                        // mutation-event logic); only a clobber of an
                        // ANCESTOR (or a prefix relation with a different
                        // path) kills.
                        (place_is_prefix_of(kplace, &info.place)
                            || place_is_prefix_of(&info.place, kplace))
                            && kplace != &info.place
                    }
                }
            };
            if var_matches {
                facts.loan_killed_at.push((info.id, point_id(*kpt)?));
            }
        }
    }
    // The path-level facts (child_path/path_is_var) from the loan
    // places — every unique place gets a path id; the root starts at its
    // variable, and each projection (Field/Index/Deref) is the direct
    // child of its base.
    // var_defined_at / var_dropped_at from the statements — a
    // `set`/`def` binding or an assignment to the bare identifier
    // (re)initializes the variable (dropping its old value).
    for (bi, blk) in cfg.blocks().iter().enumerate() {
        for (si, stmt) in blk.stmts.iter().enumerate() {
            let pt = point_id(Point {
                block: BlockId(bi),
                stmt: si,
                expr: 0,
            })?;
            match stmt {
                HirStmt::VariableDef { name: Some(n), .. } => {
                    facts.var_defined_at.push((n.to_u32(), pt));
                }
                HirStmt::Assign { target, .. } => {
                    if let HirExpr::Ident(n, _, _) = target.as_ref() {
                        facts.var_dropped_at.push((n.to_u32(), pt));
                        facts.var_defined_at.push((n.to_u32(), pt));
                    }
                }
                _ => {}
            }
        }
    }
    // The the drop-side derefs — the loan SOURCE variable (the borrowed
    // place's root) dropped while the loan is live.  The borrow
    // VARIABLE's reassignment (`r = r2`) is a reborrow-KILL, NOT a drop
    // of the source — it must not fire the drop-while-borrowed error
    // (the previous borrow_var-based extraction misfired on it).
    for (i, (place, _, _, _, _, _)) in loans.iter().enumerate() {
        if let Some(source_var) = place_root_var(place)
            && facts
                .var_dropped_at
                .iter()
                .any(|(v2, _)| *v2 == source_var.to_u32())
        {
            facts
                .drop_of_var_derefs_origin
                .push((source_var.to_u32(), i as u32));
        }
    }
    // child_path / path_is_var — the path-level facts (the DIRECT
    // path-parent relation + the root variable of every path), extracted
    // from every loan's place.  Previously empty (the `path_*_at_base`
    // family is a to_official-only concern); the path relations let the
    // official path-level facts be instantiated.  A complex-place
    // assignment (`a.f = x`, `arr[i] = x`) is a PATH-level update — it
    // does NOT re-initialize the whole root variable, so it is tracked
    // here (the path family), NOT in `var_defined_at`/`var_dropped_at`
    // (which stay variable-level: a bare-identifier reassignment).
    let mut path_ids: HashMap<FrozenPlace, u32> = HashMap::new();
    let mut next_path = 0u32;
    for (i, (_place, _, _, _, _, _)) in loans.iter().enumerate() {
        let mut cur = resolved_places[i].clone();
        loop {
            let cur_id = *path_ids.entry(cur.clone()).or_insert_with(|| {
                let id = next_path;
                next_path += 1;
                id
            });
            match &cur {
                FrozenPlace::Field(base, _)
                | FrozenPlace::Index(base)
                | FrozenPlace::ConstIndex(base, _)
                | FrozenPlace::Deref(base) => {
                    let pid = *path_ids.entry((**base).clone()).or_insert_with(|| {
                        let id = next_path;
                        next_path += 1;
                        id
                    });
                    facts.child_path.push((cur_id, pid));
                    cur = (**base).clone();
                }
                FrozenPlace::Root(v) => {
                    facts.path_is_var.push((cur_id, v.to_u32()));
                    break;
                }
            }
        }
    }
    // loan_invalidated_at / loan_killed_at  : the MUTATION events
    // — an assignment to a STRICT PREFIX of the borrowed path KILLS the
    // loan (the reborrow-kill — "mutations to the path that was borrowed
    // no longer invalidate the loan"); a mutation OF the borrowed content
    // (the loan's place is a prefix of the target) invalidates it.
    for (place, pt, is_read, _) in &events {
        if *is_read {
            continue;
        }
        // A borrow VARIABLE being (re)assigned (`rf = &mut v0`) covers the
        // old loan it held — kill every loan whose `borrow_var` is the
        // assigned root AND whose issuance point precedes the assignment
        // (rust's `loan_killed_at`: a prefix of the borrowed path is
        // assigned/overwritten, so the loan no longer needs to be
        // tracked).  Restricted to DIRECT root assignments
        // (`FrozenPlace::Root`) — a `a.b = ...` field write must NOT kill
        // a loan borrowed through `a` (the Field/Index AncestorClobber
        // exclusion), and the loan issued by the SAME expression
        // (`rf = &mut v0`'s fresh loan) is not killed either.
        if let FrozenPlace::Root(root) = place {
            let cur = point_id(*pt)?;
            for info in &infos {
                if info.borrow_var == Some(*root)
                    && point_id(info.point)? < cur
                    && !(info.point.block == pt.block && info.point.stmt == pt.stmt)
                {
                    facts.loan_killed_at.push((info.id, cur));
                }
            }
        }
        // The REBORROW-ALIASING for the mutation events: a write
        // THROUGH a reference (`*r = 5`) refers to the referent of the
        // deref chain — resolve the event's place the same way as the
        // loans (see the resolution pre-pass above), and skip the
        // loan's OWN use: a write through the loan's own borrow
        // variable (`*r2 = 5` while the r2 loan is live) is the
        // borrow's intended use (rustc allows it); a write through
        // ANOTHER variable (`*r = 5` while the r2 reborrow is live —
        // the referent frozen) invalidates the reborrow's loan.
        let (eplace, oroot, rewritten) = resolve_event_place(place, pt, &infos)?;
        for info in &infos {
            if rewritten && oroot == info.borrow_var {
                continue;
            }
            let strict_prefix = place_is_prefix_of(&eplace, &info.place) && eplace != info.place;
            if !strict_prefix && !place_is_prefix_of(&info.place, &eplace) {
                continue;
            }
            if strict_prefix {
                // Mirror the primary checker's AncestorClobber exclusion
                // (cfg_graph.rs): an assignment to a FIELD/INDEX ancestor
                // does NOT kill the loan — the borrowed content is
                // destroyed, not rebound — so the loan survives and a
                // later use through it is rejected (E109).
                if !matches!(
                    &info.place,
                    FrozenPlace::Field(..) | FrozenPlace::Index(..) | FrozenPlace::ConstIndex(..)
                ) {
                    facts.loan_killed_at.push((info.id, point_id(*pt)?));
                } else {
                    facts.loan_invalidated_at.push((info.id, point_id(*pt)?));
                }
            } else {
                facts.loan_invalidated_at.push((info.id, point_id(*pt)?));
            }
        }
    }
    // Structured troubleshooting output (PONENT_TRACE=1): the extracted
    // facts — the anya module (debug builds only).
    #[cfg(debug_assertions)]
    if crate::hir::anya::tracing_enabled() {
        crate::hir::anya::log_facts(&facts);
    }
    facts.reborrow_sources = reborrow_sources;
    Ok((facts, infos, events))
}

/// The origin liveness: `var_live` (the backward closure over var_used_at
/// and the cfg edges) + `origin_live` (via use_of_var_derefs_origin) —
/// shared by the loan rules and the subset rules.
fn origin_liveness(facts: &PoloniusFacts) -> HashSet<(u32, u64)> {
    // R3: var_live closure — var_used_at + cfg-edge propagation.
    // The rule is BACKWARD: `var_live(p1) :- var_live(p2), cfg_edge(p1, p2)`
    // — a variable live at an edge's target is also live at its source
    // (liveness flows backward against the edges, from the uses).
    let mut var_live: HashSet<(u32, u64)> = facts.var_used_at.iter().cloned().collect();
    loop {
        let mut new = HashSet::new();
        for &(v, p1) in &var_live {
            for &(e1, p2) in &facts.cfg_edge {
                if p2 == p1 {
                    new.insert((v, e1));
                }
            }
        }
        let before = var_live.len();
        var_live.extend(new);
        if var_live.len() == before {
            break;
        }
    }
    // origin_live_on_entry.
    let mut origin_live: HashSet<(u32, u64)> = HashSet::new();
    for &(var, point) in &var_live {
        for &(var2, origin) in &facts.use_of_var_derefs_origin {
            if var2 == var {
                origin_live.insert((origin, point));
            }
        }
    }
    // Placeholder origins (universal_region) are live at ALL points
    // (authoritative R3/R6/R7: `placeholder(Origin, _)` is a liveness
    // alternative — the pure var-use closure missed them).
    let all_points: HashSet<u64> = facts.cfg_edge.iter().flat_map(|&(a, b)| [a, b]).collect();
    for &origin in &facts.universal_region {
        for &pt in &all_points {
            origin_live.insert((origin, pt));
        }
    }
    // NOTE (reference-verified): the official engine's `origin_live_on_entry`
    // (rustc-generated input) keeps an origin alive ONLY at var-use points
    // (backward closure) — NOT unconditionally at every block entry.  A
    // mutation in a join block right after a branch is therefore ACCEPTED
    // by the official engine (the loan is not live at the exact
    // invalidation point).  Unconditional block-exit propagation
    // over-rejects that shape, so it is intentionally NOT applied here.
    origin_live
}

/// The subset closure (R1-R3 from polonius_int.dl): the per-point
/// `subset_base` facts, the transitive step (R2), and the cfg-edge
/// propagation (R3) with both origins live at the target.
///
/// NOTE: the datafrog_opt dying-region pruning is deliberately NOT applied
/// here — unlike the loan `contains` closure (whose errors only fire at
/// the INVALIDATED points, so the pruning is semantic-preserving), the R9
/// `subset_errors` fire at EVERY subset point between two live
/// placeholder origins.  Pruning the propagation across constant-liveness
/// edges could MISS such errors.  The unpruned closure is correct and
/// small (the `subset_base` facts are empty until the region-abstraction
/// call-site integration feeds them).
fn subset_closure(
    facts: &PoloniusFacts,
    origin_live: &HashSet<(u32, u64)>,
) -> HashSet<(u32, u32, u64)> {
    let mut subset: HashSet<(u32, u32, u64)> = facts.subset_base.iter().cloned().collect();
    loop {
        let mut new = HashSet::new();
        // R2: the transitive step (subset × subset — the authoritative
        // Polonius rule `subset(A,C,P) :- subset(A,B,P), subset(B,C,P)`).
        // Joining with `subset_base` would under-approximate the closure:
        // a fact derived by R3 (the cfg propagation) could not participate
        // as the right side of a transitivity step  .
        for &(a, b, p) in &subset {
            for &(b2, c, p2) in &subset {
                if b == b2 && p == p2 && a != c {
                    new.insert((a, c, p));
                }
            }
        }
        // R3: the cfg-edge propagation.
        for &(a, b, p1) in &subset {
            for &(e1, p2) in &facts.cfg_edge {
                if e1 == p1 && origin_live.contains(&(a, p2)) && origin_live.contains(&(b, p2)) {
                    new.insert((a, b, p2));
                }
            }
        }
        let before = subset.len();
        subset.extend(new);
        if subset.len() == before {
            break;
        }
    }
    subset
}

/// The authoritative Polonius loan rules (R1-R9 from polonius_int.dl),
/// evaluated in process.  Returns the `errors(loan, point)` pairs.
pub fn evaluate_rules(facts: &PoloniusFacts) -> Vec<(u32, u64)> {
    let origin_live = origin_liveness(facts);
    // Relevant points (the datafrog_opt pruning idea): the loan contains
    // propagation only needs to reach the points from which an
    // INVALIDATED point is forward-reachable — the contains at any other
    // point can never fire an error.  Computed as the backward closure
    // from the invalidated points along the cfg edges.
    // PRUNING INVARIANT: `relevant` must contain EVERY
    // point that can forward-reach an invalidated point — a future rule
    // that fires errors at non-invalidated points must extend this set.
    let mut relevant: HashSet<u64> = facts.loan_invalidated_at.iter().map(|(_, p)| *p).collect();
    loop {
        let mut new = HashSet::new();
        for &(e1, p2) in &facts.cfg_edge {
            if relevant.contains(&p2) {
                new.insert(e1);
            }
        }
        let before = relevant.len();
        relevant.extend(new);
        if relevant.len() == before {
            break;
        }
    }
    // The subset closure (R1-R3) — feeds R4 (the loan-via-subset step).
    let subset = subset_closure(facts, &origin_live);
    // R4-R6: origin_contains_loan_on_entry closure — issued + cfg-edge
    // propagation (R5) with the origin live at the target + the loan
    // propagation via the subset (R4), RESTRICTED to the relevant points
    // (semantically identical — the loan liveness outside the relevant
    // points cannot fire errors).
    let mut contains: HashSet<(u32, u32, u64)> = facts
        .loan_issued_at
        .iter()
        .cloned()
        .filter(|(_, _, p)| relevant.contains(p))
        .collect();
    loop {
        let mut new = HashSet::new();
        for &(loan, origin, p1) in &contains {
            for &(e1, p2) in &facts.cfg_edge {
                if e1 == p1
                    && !facts.loan_killed_at.contains(&(loan, p1))
                    && origin_live.contains(&(origin, p2))
                    && relevant.contains(&p2)
                {
                    new.insert((loan, origin, p2));
                }
            }
        }
        // R4: the loan-via-subset propagation.
        for &(loan, origin, p) in &contains {
            for &(s1, o2, sp) in &subset {
                if s1 == origin && sp == p {
                    new.insert((loan, o2, p));
                }
            }
        }
        let before = contains.len();
        contains.extend(new);
        if contains.len() == before {
            break;
        }
    }
    // R7: loan_live_at.
    let mut loan_live: HashSet<(u32, u64)> = HashSet::new();
    for &(loan, origin, point) in &contains {
        if origin_live.contains(&(origin, point)) {
            loan_live.insert((loan, point));
        }
    }
    // R8: errors(loan, point) :- invalidated(loan, point), live(loan, point).
    // The statement-level same-point exemption is GONE: the expression-
    // level points order a statement's operations (the synthetic WRITE
    // node of an assignment is at the statement's final expression
    // index, after the value's borrows), so a same-statement
    // write-then-reborrow is decidable without it — matching the
    // official engine's pure R8.
    let out: Vec<(u32, u64)> = facts
        .loan_invalidated_at
        .iter()
        .filter(|(l, p)| loan_live.contains(&(*l, *p)))
        .cloned()
        .collect();
    out
}

/// The the drop-while-borrowed errors — a dropped value whose loan is
/// still live at the drop point (the borrow outlives the value).  Kept
/// SEPARATE from `evaluate_rules` (the oracle-comparison baseline): the
/// official engine has no such error rule, so mixing it in would make
/// the differential oracle disagree.
pub fn evaluate_drop_errors(facts: &PoloniusFacts) -> Vec<(u32, u64)> {
    let full_live = loan_live_at(facts);
    let loan_origin: HashMap<u32, u32> = facts
        .loan_issued_at
        .iter()
        .map(|&(l, o, _)| (l, o))
        .collect();
    let mut errs = Vec::new();
    for (v, origin) in &facts.drop_of_var_derefs_origin {
        for &(dv, point) in &facts.var_dropped_at {
            if dv == *v {
                for &(loan, lpt) in &full_live {
                    if lpt == point && loan_origin.get(&loan) == Some(origin) {
                        errs.push((loan, point));
                    }
                }
            }
        }
    }
    errs
}

/// The FULL `loan_live_at(loan, point)` relation (R4-R7, not restricted
/// to the invalidated points) — the basis for the E109 read-freeze and
/// E112 exclusivity diagnostics (a read/mutation event at a point where
/// an overlapping loan is live, or two loans overlapping in a live
/// region).
pub fn loan_live_at(facts: &PoloniusFacts) -> HashSet<(u32, u64)> {
    let origin_live = origin_liveness(facts);
    let subset = subset_closure(facts, &origin_live);
    let mut contains: HashSet<(u32, u32, u64)> = facts.loan_issued_at.iter().cloned().collect();
    loop {
        let mut new = HashSet::new();
        for &(loan, origin, p1) in &contains {
            for &(e1, p2) in &facts.cfg_edge {
                if e1 == p1
                    && !facts.loan_killed_at.contains(&(loan, p1))
                    && origin_live.contains(&(origin, p2))
                {
                    new.insert((loan, origin, p2));
                }
            }
        }
        // R4: the loan-via-subset propagation.
        for &(loan, origin, p) in &contains {
            for &(s1, o2, sp) in &subset {
                if s1 == origin && sp == p {
                    new.insert((loan, o2, p));
                }
            }
        }
        let before = contains.len();
        contains.extend(new);
        if contains.len() == before {
            break;
        }
    }
    let mut loan_live: HashSet<(u32, u64)> = HashSet::new();
    for &(loan, origin, point) in &contains {
        if origin_live.contains(&(origin, point)) {
            loan_live.insert((loan, point));
        }
    }
    loan_live
}

/// Run the flow-sensitive borrow check over a function body.
///
/// Lives in `polonius` (the rules engine), not `cfg_graph`, because it is
/// the ENGINE path: it drives the fact extraction + the R1-R9 rules +
/// the diagnostic mapping.  Hosting it in `cfg_graph` forced a
/// `cfg_graph` → `polonius` dependency on the CFG module (a module-level
/// cycle with the rules engine that consumes the CFG).
pub fn borrow_check_function<'input>(
    body: &[HirStmt<'input>],
    finally: &[HirStmt<'input>],
    registry: &[(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) -> Vec<crate::hir::cfg_graph::BorrowError> {
    // The ENGINE path (the Polonius production switch): the facts
    // extraction + the R1-R9 rules + the diagnostic mapping replace
    // the handwritten flow-sensitive pass.  Equivalence with the old
    // pass on every covered shape is enforced by
    // `test_engine_switch_equivalence` (same accept/reject + error
    // categories).
    let cfg = CfgGraph::build_function(body, finally);
    let (facts, infos) = match extract_cross_function(&cfg, body, finally, registry, ctx) {
        Ok(x) => x,
        Err(e) => {
            // The raw stderr trace is a DEBUGGING aid only — it bypasses
            // the diagnostic pipeline.  The sentinel BorrowError below is
            // what surfaces the structured E113 diagnostic; the trace is
            // gated behind the tracing flag (and stripped entirely in
            // release builds — `anya` only compiles under
            // `debug_assertions`).
            #[cfg(debug_assertions)]
            if crate::hir::anya::tracing_enabled() {
                eprintln!("borrow check: {e}");
            }
            return vec![crate::hir::cfg_graph::BorrowError {
                place: FrozenPlace::Root(Symbol::intern("<internal>")),
                is_read: false,
                is_exclusive: false,
                is_drop: false,
                loan_kind: None,
                span1: None,
                span: crate::ast::Span::new(0, 0),
            }];
        }
    };
    let (loans, events, _, _) = collect_borrow_data(&cfg, registry, ctx);
    // Structured troubleshooting output (PONENT_TRACE=1): the CFG + the
    // collected loans/events — the anya module (debug builds only).
    #[cfg(debug_assertions)]
    if crate::hir::anya::tracing_enabled() {
        crate::hir::anya::log_cfg(&cfg);
        crate::hir::anya::log_borrow_data(&loans, &events);
    }
    match rules_to_borrow_errors(&facts, &infos, &events) {
        Ok(errs) => errs,
        Err(e) => {
            // Fail-closed (E113): the mapping layer degraded silently on
            // an oversized/foreign point id (`unwrap_or(0)`) — surface
            // the E113 sentinel instead of a mis-ordered diagnostic.
            #[cfg(debug_assertions)]
            if crate::hir::anya::tracing_enabled() {
                eprintln!("borrow check: {e}");
            }
            vec![crate::hir::cfg_graph::BorrowError {
                place: FrozenPlace::Root(Symbol::intern("<internal>")),
                is_read: false,
                is_exclusive: false,
                is_drop: false,
                loan_kind: None,
                span1: None,
                span: crate::ast::Span::new(0, 0),
            }]
        }
    }
}

/// The diagnostic mapping layer — translate the rules output into
/// user-facing `BorrowError`s:
/// - E110 (mutation-freeze): the R8 errors — the loan was INVALIDATED at
///   a point where it is still live;
/// - E109 (read-freeze): a READ event at a point where an overlapping
///   EXCLUSIVE (`&mut`) loan is live;
/// - E112 (exclusivity): two overlapping loans with one live at the
///   other's issuance point.
/// Anchored with the loan's place/kind/span — the bridge that lets the
/// engine replace the handwritten pass's diagnostics.
pub fn rules_to_borrow_errors(
    facts: &PoloniusFacts,

    infos: &[LoanInfo],
    events: &[(FrozenPlace, Point, bool, crate::ast::Span)],
) -> Result<Vec<crate::hir::cfg_graph::BorrowError>, String> {
    let mut out = Vec::new();
    let loan_live = loan_live_at(facts);
    // The loan-ISSUANCE points: an invalidation AT a loan-issuance point
    // is a BORROW of the place (a second `&mut` argument), not a
    // mutation — the E112 exclusivity diagnostic reports it, and the
    // E110 mutation-freeze would duplicate it (rustc reports E0499 for
    // `f(&mut a, &mut a)`, not E0506).
    let issued_points: HashSet<u64> = facts.loan_issued_at.iter().map(|(_, _, p)| *p).collect();
    // E110: the mutation-freeze (invalidated + live).
    for (loan, point) in evaluate_rules(facts) {
        let Some(info) = infos.get(loan as usize) else {
            continue;
        };
        if issued_points.contains(&point) {
            continue;
        }
        out.push(crate::hir::cfg_graph::BorrowError {
            place: info.place.clone(),
            is_read: false,
            is_exclusive: false,
            is_drop: false,
            loan_kind: Some(info.kind),
            span1: None,
            span: info.span,
        });
    }
    // E109: a READ event at a point where an overlapping EXCLUSIVE loan
    // is live (`&mut` freezes reads — the read-side freeze).
    for (place, pt, is_read, span) in events {
        if !is_read {
            continue;
        }
        let pt_id = point_id(*pt)?;
        // The REBORROW-ALIASING for the read events: `*r2` resolves to
        // the referent (the reborrow's place) — and the loan's OWN use
        // (an access through its own borrow variable — reading `*r2`
        // while the r2 loan is live) is exempt; reading through ANOTHER
        // variable (`*r` after the reborrow) is a conflicting access
        // (rustc E0503).
        let (eplace, oroot, rewritten) = resolve_event_place(place, pt, infos)?;
        for (i, info) in infos.iter().enumerate() {
            if info.kind != LoanKind::Exclusive {
                continue;
            }
            if rewritten && oroot == info.borrow_var {
                continue;
            }
            // TWO-PHASE BORROW (reserved): a method-call receiver loan
            // (`v.push(v.len())` — `&mut v` reserved while the args
            // evaluate) does NOT freeze READS during the reservation: the
            // callee read at the receiver loan's OWN expression point and
            // the argument reads at LATER points of the same statement
            // (`v.len()` reads `v` legally during the reservation).
            // Mirrors Polonius' `Reservation` access kind for
            // `TwoPhaseBorrow`.
            if info.two_phase
                && pt.block == info.point.block
                && pt.stmt == info.point.stmt
                && pt.expr >= info.point.expr
            {
                continue;
            }
            let overlaps = place_is_prefix_of(&eplace, &info.place)
                || place_is_prefix_of(&info.place, &eplace);
            if overlaps && loan_live.contains(&(i as u32, pt_id)) {
                out.push(crate::hir::cfg_graph::BorrowError {
                    place: eplace.clone(),
                    is_read: true,
                    is_exclusive: false,
                    is_drop: false,
                    loan_kind: Some(LoanKind::Exclusive),
                    span1: None,
                    span: *span,
                });
            }
        }
    }
    // E112: two overlapping loans, one live at the other's issuance
    // point (the exclusivity conflict — dual-position diagnostics).
    for i in 0..infos.len() {
        for j in (i + 1)..infos.len() {
            let a = &infos[i];
            let b = &infos[j];
            let overlaps =
                place_is_prefix_of(&a.place, &b.place) || place_is_prefix_of(&b.place, &a.place);
            if !overlaps {
                continue;
            }
            // Same-SPAN loans (`a.span == b.span`) come from the SAME
            // borrow expression — e.g. `rf = &mut v0` produces both the
            // rf-held loan and the argument-temporary loan (var `None`);
            // they cannot overlap (there is only one borrow, two
            // bookkeeping entries for it).  Two DISTINCT `&mut a`
            // arguments of one call have different spans and DO overlap.
            if a.span == b.span {
                continue;
            }
            // A loan that is KILLED at (or after) the other's issuance
            // point is not live there — e.g. `rf = &mut v0` (re)assigns
            // the borrow variable at the statement's final expression
            // point, killing the earlier `&mut v0` loan it held, so the
            // two loans do NOT overlap (the reborrow-kill of
            // `loan_killed_at` — matching rust's NLL/Polonius behavior
            // for branch-merged reborrows).  The exemption applies ONLY
            // when the two loans share the SAME borrow variable (a
            // re-assignment); a temporary loan's statement-end kill must
            // not exempt a genuine overlap with another `&mut` loan.
            // TWO-PHASE RESERVATION (TPB, mirroring the E109 window): a
            // two-phase receiver loan (`obj.put(...)` — `&mut obj`
            // reserved while the args evaluate) is NOT yet active during
            // its reservation window (same statement, at-or-after its
            // issuance) — a later overlapping loan in the window (e.g.
            // the nested `obj.get()` receiver of `obj.put(obj.get())`)
            // does not conflict with it.  rustc accepts the shape; the
            // activation at the statement's end is what matters.
            let a_reserved = a.two_phase
                && b.point.block == a.point.block
                && b.point.stmt == a.point.stmt
                && b.point.expr >= a.point.expr;
            let b_reserved = b.two_phase
                && a.point.block == b.point.block
                && a.point.stmt == b.point.stmt
                && a.point.expr >= b.point.expr;
            // The REBORROW parent/child pair (the referent resolution
            // pre-pass): the child loan derives FROM the parent (the
            // parent is killed at the child's issuance) — they cannot
            // conflict, and the parent's liveness must not trip the
            // exclusivity check (rustc accepts `let r2 = &mut *r;` — the
            // reborrow takes over the referent's borrow).
            if facts.reborrow_sources.contains(&(i as u32, j as u32)) {
                continue;
            }
            let b_pt = point_id(b.point)?;
            let a_pt = point_id(a.point)?;
            // The borrow-variable REASSIGNMENT exemption: `rf = &mut v0`
            // (re)assigns the borrow variable at the assignment's
            // statement, killing the EARLIER `&mut v0` loan it held — the
            // two loans do not overlap (matching rust's NLL/Polonius
            // branch-merged reborrows).  The exemption fires ONLY when
            // the kill lands in the SAME STATEMENT as the other loan's
            // issuance (`k >> 16 == b_pt >> 16` — the point encoding is
            // `block<<36 | stmt<<16 | expr`).  The LAST-USE kill (a loan
            // dying at its borrow variable's final use, e.g. `return *r`
            // in a later statement) is NOT a reassignment and must NOT
            // exempt a genuine overlap — two distinct `&mut a` arguments
            // of one call (`set r = f(&mut a, &mut a)`, rustc E0499)
            // share the borrow var `r`, and without this statement-scope
            // restriction their last-use kills (in `return *r`) would
            // falsely exempt the exclusivity conflict.
            let a_live_at_b = !a_reserved
                && loan_live.contains(&(i as u32, b_pt))
                && !(facts
                    .loan_killed_at
                    .iter()
                    .any(|&(la, k)| la == i as u32 && (k >> 16) == (b_pt >> 16))
                    && a.borrow_var.is_some()
                    && a.borrow_var == b.borrow_var);
            let b_live_at_a = !b_reserved
                && loan_live.contains(&(j as u32, a_pt))
                && !(facts
                    .loan_killed_at
                    .iter()
                    .any(|&(lb, k)| lb == j as u32 && (k >> 16) == (a_pt >> 16))
                    && a.borrow_var.is_some()
                    && a.borrow_var == b.borrow_var);
            if a_live_at_b || b_live_at_a {
                out.push(crate::hir::cfg_graph::BorrowError {
                    place: b.place.clone(),
                    is_read: false,
                    is_exclusive: true,
                    is_drop: false,
                    loan_kind: None,
                    span1: Some(a.span),
                    span: b.span,
                });
            }
        }
    }
    // R9 (rejection): the placeholder subset errors — a subset
    // relation between two placeholder (signature-region) origins NOT
    // declared in the signature (`known_placeholder_subset`) is rejected:
    // the signature must carry every cross-region relationship (the
    // caller needs it to reason about the returned borrows).  The
    // closure propagates one error PER reachable point — deduplicate
    // (one identical diagnostic per violating pair, not per point).
    let mut subset_errs: Vec<(u32, u32)> = Vec::new();
    for (o1, o2, _p) in evaluate_subset_errors(facts) {
        if !subset_errs.contains(&(o1, o2)) {
            subset_errs.push((o1, o2));
        }
    }
    for (_o1, _o2) in subset_errs {
        out.push(crate::hir::cfg_graph::BorrowError {
            place: FrozenPlace::Root(Symbol::intern("<region>")),
            is_read: false,
            is_exclusive: false,
            is_drop: false,
            loan_kind: None,
            span1: None,
            span: crate::ast::Span::new(0, 0),
        });
    }
    // E116: the drop-while-borrowed errors — a dropped value whose loan
    // is still live at the drop point (the borrow outlives the value).
    // Anchored at the loan's place/kind/span (the drop's own point is
    // not an event in the `events` list).  `evaluate_drop_errors` is
    // deliberately separate from `evaluate_rules` (the official engine
    // has no such rule — the differential oracle must not see it), but
    // the mapping layer is not oracle-compared: wire it in here so
    // the violation is actually REJECTED, not just computed.
    for (loan, _point) in evaluate_drop_errors(facts) {
        let Some(info) = infos.get(loan as usize) else {
            continue;
        };
        out.push(crate::hir::cfg_graph::BorrowError {
            place: info.place.clone(),
            is_read: false,
            is_exclusive: false,
            is_drop: true,
            loan_kind: Some(info.kind),
            span1: None,
            span: info.span,
        });
    }
    // Deduplicate IDENTICAL diagnostics: the rules emit one error PER
    // (loan, point) pair, so a loan invalidated at several points (E110)
    // or dropped while live at several points (E116) would otherwise
    // surface the same message repeatedly at the same span.  The
    // place/span/category sextet is the diagnostic identity (span1 is
    // included: two E112 exclusivity errors at the same b.span differ in
    // their first-loan span; the spans are keyed as (start, end) —
    // `Span`/`LoanKind` have no `Hash` impl, so the kind is keyed by
    // name).
    let mut seen: HashSet<(
        FrozenPlace,
        bool,
        bool,
        Option<&'static str>,
        Option<(u64, u64)>,
        (u64, u64),
    )> = HashSet::default();
    let mut deduped = Vec::with_capacity(out.len());
    for e in out {
        let key = (
            e.place.clone(),
            e.is_read,
            e.is_exclusive || e.is_drop,
            e.loan_kind.map(|k| k.as_str()),
            e.span1.map(|s| (s.start as u64, s.end as u64)),
            (e.span.start as u64, e.span.end as u64),
        );
        if seen.insert(key) {
            deduped.push(e);
        }
    }
    Ok(deduped)
}

/// R9 (the placeholder-subset rejection — the committee ruling): the
/// placeholder subset
/// errors — a subset relation between two placeholder origins NOT
/// declared in the signature (`known_placeholder_subset`) is rejected.
pub fn evaluate_subset_errors(facts: &PoloniusFacts) -> Vec<(u32, u32, u64)> {
    let origin_live = origin_liveness(facts);
    let subset = subset_closure(facts, &origin_live);
    let placeholder: HashSet<u32> = facts.universal_region.iter().cloned().collect();
    // known_placeholder_subset: the transitive closure.
    let mut known: HashSet<(u32, u32)> = facts.known_placeholder_subset.iter().cloned().collect();
    loop {
        let mut new = HashSet::new();
        for &(a, b) in &known {
            for &(b2, c) in &facts.known_placeholder_subset {
                if b == b2 {
                    new.insert((a, c));
                }
            }
        }
        let before = known.len();
        known.extend(new);
        if known.len() == before {
            break;
        }
    }
    let mut errs = Vec::new();
    for &(o1, o2, p) in &subset {
        if o1 != o2
            && placeholder.contains(&o1)
            && placeholder.contains(&o2)
            && !known.contains(&(o1, o2))
        {
            errs.push((o1, o2, p));
        }
    }
    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::checker::tests::check_source;

    fn first_body(src: &str) -> Vec<HirStmt<'static>> {
        let prog = check_source(src).expect("program must check");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        body.clone().unwrap_or_default()
    }

    /// The FactTypes instantiation for the official oracle — a newtype
    /// wrapper (a bare `usize` would violate the orphan rule for the
    /// external `Atom` trait).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct OracleId(usize);
    impl polonius_engine::Atom for OracleId {
        fn index(self) -> usize {
            self.0
        }
    }
    impl From<usize> for OracleId {
        fn from(v: usize) -> Self {
            OracleId(v)
        }
    }
    impl From<OracleId> for usize {
        fn from(v: OracleId) -> Self {
            v.0
        }
    }
    #[derive(Debug, Clone, Copy)]
    struct OracleTypes;
    impl polonius_engine::FactTypes for OracleTypes {
        type Origin = OracleId;
        type Loan = OracleId;
        type Point = OracleId;
        type Variable = OracleId;
        type Path = OracleId;
    }

    /// Transpose OUR `PoloniusFacts` into the official `AllFacts` shape.
    /// Note the tuple-order differences: official `loan_issued_at` is
    /// `(origin, loan, point)` (ours is `(loan, origin, point)`) and
    /// `loan_invalidated_at` is `(point, loan)` (ours is `(loan, point)`).
    fn to_official(facts: &PoloniusFacts) -> polonius_engine::AllFacts<OracleTypes> {
        let id = |v: &u32| OracleId(*v as usize);
        let idp = |v: &u64| OracleId(*v as usize);
        polonius_engine::AllFacts {
            cfg_edge: facts
                .cfg_edge
                .iter()
                .map(|&(a, b)| (idp(&a), idp(&b)))
                .collect(),
            loan_issued_at: facts
                .loan_issued_at
                .iter()
                .map(|&(loan, origin, point)| (id(&origin), id(&loan), idp(&point)))
                .collect(),
            universal_region: facts.universal_region.iter().map(id).collect(),
            loan_killed_at: facts
                .loan_killed_at
                .iter()
                .map(|&(loan, point)| (id(&loan), idp(&point)))
                .collect(),
            subset_base: facts
                .subset_base
                .iter()
                .map(|&(o1, o2, p)| (id(&o1), id(&o2), idp(&p)))
                .collect(),
            loan_invalidated_at: facts
                .loan_invalidated_at
                .iter()
                .map(|&(loan, point)| (idp(&point), id(&loan)))
                .collect(),
            var_used_at: facts
                .var_used_at
                .iter()
                .map(|&(v, p)| (id(&v), idp(&p)))
                .collect(),
            use_of_var_derefs_origin: facts
                .use_of_var_derefs_origin
                .iter()
                .map(|&(v, o)| (id(&v), id(&o)))
                .collect(),
            known_placeholder_subset: facts
                .known_placeholder_subset
                .iter()
                .map(|&(a, b)| (id(&a), id(&b)))
                .collect(),
            // Fields we do not extract yet (the path-assignment remainder): the
            // path-ASSIGNMENT events (the official engine's
            // path_assigned_at_base / path_moved_at_base /
            // path_accessed_at_base — only consumed by the path-level
            // variants it offers; Naive's core R1-R9 does not need them).
            placeholder: Vec::new(),
            var_defined_at: facts
                .var_defined_at
                .iter()
                .map(|&(v, p)| (id(&v), idp(&p)))
                .collect(),
            var_dropped_at: facts
                .var_dropped_at
                .iter()
                .map(|&(v, p)| (id(&v), idp(&p)))
                .collect(),
            drop_of_var_derefs_origin: facts
                .drop_of_var_derefs_origin
                .iter()
                .map(|&(v, o)| (id(&v), id(&o)))
                .collect(),
            child_path: facts
                .child_path
                .iter()
                .map(|&(c, p)| (id(&c), id(&p)))
                .collect(),
            path_is_var: facts
                .path_is_var
                .iter()
                .map(|&(p, v)| (id(&p), id(&v)))
                .collect(),
            path_assigned_at_base: Vec::new(),
            path_moved_at_base: Vec::new(),
            path_accessed_at_base: Vec::new(),
        }
    }

    /// Differential oracle: the OFFICIAL rustc Polonius engine
    /// (`Algorithm::Naive`) must agree with OUR `evaluate_rules` on
    /// accept/reject for the same bodies.  This is the strongest
    /// validation — the official engine shares none of our CFG/liveness
    /// infrastructure, so shared-infrastructure bugs cannot mask drift.
    #[test]
    fn test_oracle_official_engine_agrees() {
        // (source, expected acceptance) — OUR pass is the reference.
        let cases: &[(&str, bool)] = &[
            // Same-block last use: the loan dies at `*r`, `a = 5` is fine.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     let x = *r;
                     a = 5;
                     return x;
                 }",
                true,
            ),
            // A plain through-borrow.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     return *r;
                 }",
                true,
            ),
            // Mutation while the borrow is live must be rejected.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     a = 5;
                     return *r;
                 }",
                false,
            ),
            // R5 (CFG-edge propagation): a borrow inside an `if` branch.
            // NOTE: the OFFICIAL engine ACCEPTS a mutation in the join
            // block (the loan lives at the block ENTRY points, not at the
            // in-block invalidation point — R8 needs the loan live at the
            // exact point).  Our engine now matches that acceptance.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     if true { set r: &mut Int<32> = &mut a; *r = 1; }
                     a = 5;
                     return a;
                 }",
                true,
            ),
            // R4 (subset propagation): a cross-function returned borrow
            // must propagate A(ρ) — a mutation of the source after the
            // call is rejected.  (The harness now drives the engine with
            // the real registry, so the cross-function loan + placeholder input
            // origin are injected and R8 fires — matching the official
            // engine's rejection.)
            (
                "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
                 def main() -> Int<32> {
                     set mut a = 42;
                     set r = get(&mut a);
                     a = 5;
                     return *r;
                 }",
                false,
            ),
        ];
        for (src, expect_accept) in cases {
            let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(src);
            let Some(item) = prog.items.iter().find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            }) else {
                panic!("expected a main function def");
            };
            let HirStmt::FunctionDef { body, .. } = item else {
                panic!("expected a function def");
            };
            let body = body.clone().unwrap_or_default();
            let cfg = CfgGraph::build_function(&body, &[]);
            // The differential harness must drive the engine through the
            // cross-function path (`extract_cross_function` with a real
            // registry) so the cross-function loan (and its placeholder input
            // origin) is injected — `extract_facts` with an EMPTY registry never
            // sees cross-function returned-borrow loans, which made the R4
            // case diverge from the official engine (test-harness gap).
            let registry: Vec<(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)> = vec![(
                Symbol::intern("get"),
                false,
                None,
                vec![0],
                SignatureFacts {
                    universal_region: vec![0],
                    known_placeholder_subset: vec![(0, 1)],
                    input_borrow_mutable: vec![true],
                },
            )];
            let (facts, _infos) = extract_cross_function(
                &cfg,
                &body,
                &[],
                &registry,
                &crate::hir::types::TypeContext::new(),
            )
            .expect("the cross-function extraction must succeed");
            let ours = evaluate_rules(&facts);
            let official = polonius_engine::Output::compute(
                &to_official(&facts),
                polonius_engine::Algorithm::Naive,
                false,
            );
            let official_errs: Vec<(OracleId, OracleId)> = official
                .errors
                .iter()
                .flat_map(|(&point, loans)| loans.iter().map(move |&loan| (loan, point)))
                .collect();
            assert_eq!(
                ours.is_empty(),
                *expect_accept,
                "our evaluate_rules acceptance mismatch for {src:?} (diags: {diags:?})"
            );
            assert_eq!(
                official_errs.is_empty(),
                *expect_accept,
                "official engine disagrees on {src:?} (ours: {ours:?}, official: {official_errs:?})"
            );
        }
    }

    /// Differential harness (the first step of wiring the Polonius
    /// engine into production): the handwritten flow-sensitive pass
    /// (`borrow_check_function`) and the R1-R9 rules engine
    /// (`evaluate_rules`) must AGREE on accept/reject for the same
    /// bodies.  This makes drift between the two implementations
    /// detectable instead of silent.  The handwritten pass stays the
    /// production authority until the rules engine covers every shape
    /// (the ancestor-clobber shape is a known gap — see the ignored
    /// `test_polonius_ancestor_clobber_exclusion`).
    ///
    /// Uses `check_source_keep_hir` (not `first_body`/`check_source`)
    /// so rejected programs still yield their HIR body for the
    /// differential run — `check_source` drops bodies on borrow errors.
    #[test]
    fn test_differential_handwritten_vs_rules() {
        // (source, expected acceptance) — the handwritten pass is the
        // production authority; the rules engine must agree.
        let cases: &[(&str, bool)] = &[
            // Same-block last use: the loan dies at `*r`, `a = 5` is fine.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     let x = *r;
                     a = 5;
                     return x;
                 }",
                true,
            ),
            // A plain through-borrow (no mutation after).
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     return *r;
                 }",
                true,
            ),
            // Mutation while the borrow is live must be rejected.
            (
                "def main() -> Int<32> {
                     set mut a = 42;
                     set r: &mut Int<32> = &mut a;
                     a = 5;
                     return *r;
                 }",
                false,
            ),
        ];
        for (src, expect_accept) in cases {
            let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(src);
            let Some(item) = prog.items.iter().find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            }) else {
                panic!("expected a main function def");
            };
            let HirStmt::FunctionDef { body, .. } = item else {
                panic!("expected a function def");
            };
            let body = body.clone().unwrap_or_default();
            let hand =
                borrow_check_function(&body, &[], &[], &crate::hir::types::TypeContext::new());
            let cfg = CfgGraph::build_function(&body, &[]);
            let (facts, _infos, _events) = extract_facts(
                &cfg,
                &body,
                &[],
                &[],
                &crate::hir::types::TypeContext::new(),
            )
            .expect("the facts extraction must succeed");
            let rules = evaluate_rules(&facts);
            assert_eq!(
                hand.is_empty(),
                *expect_accept,
                "handwritten pass acceptance mismatch for {src:?} (diags: {diags:?})"
            );
            assert_eq!(
                rules.is_empty(),
                *expect_accept,
                "rules engine disagrees with the handwritten pass on {src:?} (drift!)"
            );
        }
    }

    /// The cross-function extraction — `get(x: &mut T) -> &mut T`
    /// called as `get(&mut a)` — the argument loan in the input origin +
    /// the A(ρ) subset_base instantiation at the call point + the callee
    /// signature facts merged.
    #[test]
    fn test_extract_cross_function() {
        // The get signature: x: &mut T (position 0) + the return borrow.
        let sig = signature_facts(
            &BorrowSignature {
                input_borrows: vec![(Symbol::intern("x"), false)],
                input_lifetimes: vec![None],
                output_borrows: vec![Symbol::intern("__ret")],
            },
            &[0],
        );
        let registry = vec![(Symbol::intern("get"), false, None, vec![0usize], sig)];
        // The main body (the caller with the call-site).
        let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(
            "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = get(&mut a);
                 *r = 10;
                 return *r;
             }",
        );
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            })
            .expect("main function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos) = extract_cross_function(
            &cfg,
            &body,
            &[],
            &registry,
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the cross-function extraction must succeed");
        assert!(
            !facts.subset_base.is_empty(),
            "the A(ρ) subset_base must be instantiated at the call point"
        );
        assert!(
            !facts.universal_region.is_empty(),
            "the callee's universal_region must be merged"
        );
    }

    /// The cross-function wiring — the ENGINE (not just the fact
    /// extraction) must detect the cross-function returned-borrow freeze:
    /// `get(&mut a)` instantiates the callee's A(ρ) subset at the call
    /// point (`subset_base`), so a mutation of `a` while the returned
    /// borrow `r` is live must be rejected by `evaluate_rules`.
    #[test]
    fn test_cross_function_rules_detection() {
        let sig = signature_facts(
            &BorrowSignature {
                input_borrows: vec![(Symbol::intern("x"), false)],
                input_lifetimes: vec![None],
                output_borrows: vec![Symbol::intern("__ret")],
            },
            &[0],
        );
        let registry = vec![(Symbol::intern("get"), false, None, vec![0usize], sig)];
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def get(x: &mut Int<32>) -> &mut Int<32> { return x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = get(&mut a);
                 a = 5;
                 return *r;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            })
            .expect("main function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos) = extract_cross_function(
            &cfg,
            &body,
            &[],
            &registry,
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the cross-function extraction must succeed");
        assert!(
            !facts.subset_base.is_empty(),
            "the call site must instantiate the A(ρ) subset_base"
        );
        assert!(
            !evaluate_rules(&facts).is_empty(),
            "the rules engine must detect the cross-function returned-borrow freeze"
        );
    }

    /// Soundness direction: on programs the borrow checker ACCEPTS, the
    /// authoritative Polonius rules must also find NO errors (no false
    /// positives).  (The error direction needs a diagnostics-returning
    /// checker hook — the post-pass's errors are what rejects programs, so
    /// check_source drops their bodies; documented as a follow-up.)
    #[test]
    fn test_polonius_equivalence_accepted() {
        for src in [
            // Same-block last use: the loan dies at `*r`, `a = 5` is fine.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 a = 5;
                 return x;
             }",
            // A plain through-borrow.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return *r;
             }",
            // A loop without borrows.
            "def main() -> Int<32> {
                 set mut a = 42;
                 while a > 0 { a = a - 1; }
                 return a;
             }",
        ] {
            let body = first_body(src);
            let cfg = CfgGraph::build_function(&body, &[]);
            let (facts, _infos, _events) = extract_facts(
                &cfg,
                &body,
                &[],
                &[],
                &crate::hir::types::TypeContext::new(),
            )
            .expect("the facts extraction must succeed");
            let errs = evaluate_rules(&facts);
            assert!(
                errs.is_empty(),
                "the authoritative Polonius rules must find no errors on accepted programs: \
                 {src:?} — errs: {errs:?}"
            );
        }
    }

    /// The CSV writer produces the polonius_int.dl input schema: all 17
    /// input files exist, and the populated relations are non-empty.
    #[test]
    fn test_write_facts_csv() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return *r;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        let dir = std::env::temp_dir().join(format!("posita_csv_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        write_facts_csv(&facts, &dir).expect("write the CSV inputs");
        for name in POLONIUS_INPUTS {
            assert!(dir.join(name).exists(), "missing input file {name}");
        }
        assert!(
            !std::fs::read_to_string(dir.join("cfg_edge.csv"))
                .unwrap()
                .is_empty(),
            "cfg_edge must be populated"
        );
        assert!(
            !std::fs::read_to_string(dir.join("loan_issued_at.csv"))
                .unwrap()
                .is_empty(),
            "loan_issued_at must be populated"
        );
    }

    /// Generate a LARGE function body with `loans` borrow bindings and
    /// `points` statements — the `loans × points` product scales the
    /// facts (the `cfg_edge`/`var_used_at`/`loan_issued_at` relations)
    /// that both engines must solve.  The program is NOT intended to
    /// type-check cleanly (mutations overlap the borrows) —
    /// `check_source_keep_hir` keeps the body regardless of diagnostics.
    fn gen_large_body(loans: usize, points: usize) -> Vec<HirStmt<'static>> {
        let mut src = String::from("def main() -> Int<32> {\n");
        for i in 0..loans {
            src.push_str(&format!("set mut v{i} = {i};\n"));
            src.push_str(&format!("set r{i} = &mut v{i};\n"));
        }
        // The `points` statements: mutations over the borrowed vars (each
        // is a loan event + a point — the solver's work scales with them).
        for i in 0..points {
            src.push_str(&format!("v{} = {};\n", i % loans, i));
        }
        src.push_str("return 0;\n}");
        let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(&src);
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            })
            .expect("main function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        body.clone().unwrap_or_default()
    }

    /// The FlowLog LIBRARY MODE smoke test: the vendored
    /// `polonius_int.dl` (compiled by build.rs via flowlog-build) must
    /// run end-to-end — feed the facts via `insert_*` (the generated
    /// `rel` tuples), `run()` the `DatalogBatchEngine`, and observe the
    /// `errors_size` relation.  This is the base for the self-written vs
    /// FlowLog timing benchmark.
    mod flowlog_libmode {
        include!(concat!(env!("OUT_DIR"), "/polonius_int.rs"));

        /// The self-written evaluator vs the FlowLog library-mode engine:
        /// for the same generated large body, time `extract_facts` +
        /// `evaluate_rules` (ours) against the `DatalogBatchEngine`
        /// (feed all 17 input relations, `run()`).  The tuple order for
        /// `loan_issued_at`/`loan_invalidated_at` follows the official
        /// `polonius_int.dl` semantics (origin, loan, point) / (point,
        /// loan) — the same transpose `to_official` uses.
        #[test]
        fn bench_self_vs_flowlog() {
            use std::time::Instant;
            for (loans, points) in [(10usize, 20usize), (50, 100), (200, 400)] {
                let body = super::gen_large_body(loans, points);
                let cfg = crate::hir::cfg_graph::CfgGraph::build_function(&body, &[]);
                // Self-written: facts extraction + the R1-R9 evaluator.
                let t0 = Instant::now();
                let (facts, _infos, _events) = super::extract_facts(
                    &cfg,
                    &body,
                    &[],
                    &[],
                    &crate::hir::types::TypeContext::new(),
                )
                .expect("facts");
                let self_errs = super::evaluate_rules(&facts).len();
                let self_time = t0.elapsed();
                // FlowLog library mode: feed all input relations, run.
                let t1 = Instant::now();
                let mut engine = DatalogBatchEngine::new(1);
                engine.insert_cfg_edge(
                    facts
                        .cfg_edge
                        .iter()
                        .map(|&(a, b)| (a as i64, b as i64))
                        .collect(),
                );
                engine.insert_loan_issued_at(
                    facts
                        .loan_issued_at
                        .iter()
                        .map(|&(loan, origin, point)| (origin as i32, loan as i32, point as i64))
                        .collect(),
                );
                engine.insert_universal_region(
                    facts
                        .universal_region
                        .iter()
                        .map(|&o| (o as i32,))
                        .collect(),
                );
                engine.insert_loan_killed_at(
                    facts
                        .loan_killed_at
                        .iter()
                        .map(|&(l, p)| (l as i32, p as i64))
                        .collect(),
                );
                engine.insert_subset_base(
                    facts
                        .subset_base
                        .iter()
                        .map(|&(a, b, c)| (a as i32, b as i32, c as i64))
                        .collect(),
                );
                engine.insert_loan_invalidated_at(
                    facts
                        .loan_invalidated_at
                        .iter()
                        .map(|&(l, p)| (l as i32, p as i64))
                        .collect(),
                );
                engine.insert_var_used_at(
                    facts
                        .var_used_at
                        .iter()
                        .map(|&(v, p)| (v as i32, p as i64))
                        .collect(),
                );
                engine.insert_use_of_var_derefs_origin(
                    facts
                        .use_of_var_derefs_origin
                        .iter()
                        .map(|&(v, o)| (v as i32, o as i32))
                        .collect(),
                );
                engine.insert_known_placeholder_subset(
                    facts
                        .known_placeholder_subset
                        .iter()
                        .map(|&(a, b)| (a as i32, b as i32))
                        .collect(),
                );
                engine.insert_var_defined_at(
                    facts
                        .var_defined_at
                        .iter()
                        .map(|&(v, p)| (v as i32, p as i64))
                        .collect(),
                );
                engine.insert_var_dropped_at(
                    facts
                        .var_dropped_at
                        .iter()
                        .map(|&(v, p)| (v as i32, p as i64))
                        .collect(),
                );
                engine.insert_drop_of_var_derefs_origin(
                    facts
                        .drop_of_var_derefs_origin
                        .iter()
                        .map(|&(v, o)| (v as i32, o as i32))
                        .collect(),
                );
                engine.insert_child_path(
                    facts
                        .child_path
                        .iter()
                        .map(|&(c, p)| (c as i32, p as i32))
                        .collect(),
                );
                engine.insert_path_is_var(
                    facts
                        .path_is_var
                        .iter()
                        .map(|&(p, v)| (p as i32, v as i32))
                        .collect(),
                );
                engine.insert_path_moved_at_base(vec![]);
                engine.insert_path_assigned_at_base(vec![]);
                engine.insert_path_accessed_at_base(vec![]);
                let results = engine.run();
                let flowlog_time = t1.elapsed();
                eprintln!(
                    "loans×points = {loans}×{points}: self = {self_time:?} ({self_errs} errs), flowlog = {flowlog_time:?} ({:?} errs)",
                    results.errors_size
                );
            }
        }

        #[test]
        fn test_flowlog_library_mode_smoke() {
            let mut engine = DatalogBatchEngine::new(1);
            // The minimal violation: a loan issued at 0, live through
            // point 2 (the var used there), and invalidated at 2 while
            // still live — must fire the errors relation.
            engine.insert_cfg_edge(vec![(0, 1), (1, 2)]);
            engine.insert_loan_issued_at(vec![(0, 0, 0)]);
            engine.insert_var_used_at(vec![(99, 2)]);
            engine.insert_use_of_var_derefs_origin(vec![(99, 0)]);
            engine.insert_loan_invalidated_at(vec![(0, 2)]);
            let results = engine.run();
            assert!(
                results.errors_size >= 1,
                "the invalidated-live loan must fire the errors relation (got errors_size={})",
                results.errors_size
            );
        }
    }

    /// The external FlowLog engine (when available) must agree with the
    /// in-process evaluator — no errors on accepted programs.  If FlowLog
    /// is unavailable, the runner returns None and the test passes
    /// gracefully (the in-process evaluator already validated the rules).
    #[test]
    fn test_flowlog_runner_equivalence() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 a = 5;
                 return x;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        assert!(
            evaluate_rules(&facts).is_empty(),
            "in-process evaluator: no errors on the accepted program"
        );
        if let Some(n) = run_flowlog(&facts) {
            assert_eq!(
                n, 0,
                "the external FlowLog engine must agree with the in-process evaluator"
            );
        }
    }

    /// Error direction: a program the post-pass REJECTS — the
    /// authoritative Polonius rules must ALSO find the violation (no false
    /// negatives).  Uses `check_source_keep_hir` (the post-pass's
    /// diagnostics reject the program, but the HIR body is kept).
    #[test]
    fn test_polonius_equivalence_rejected() {
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 a = 5;
                 return *r;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        let errs = evaluate_rules(&facts);
        assert!(
            !errs.is_empty(),
            "the Polonius rules must find the violation (no false negatives)"
        );
        if let Some(n) = run_flowlog(&facts) {
            assert!(
                n > 0,
                "the external FlowLog engine must also find the violation"
            );
        }
    }

    /// The diagnostic mapping layer — a rejected program must yield
    /// user-facing `BorrowError`s (E110 mutation-freeze) carrying the
    /// loan's place/kind/span, so the engine can drive diagnostics
    /// directly (the bridge to replacing the handwritten pass).
    #[test]
    fn test_rules_to_borrow_errors_mapping() {
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 a = 5;
                 return *r;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        let (_, events, _, _) =
            collect_borrow_data(&cfg, &[], &crate::hir::types::TypeContext::new());
        let errs = rules_to_borrow_errors(&facts, &infos, &events)
            .expect("the mapping must succeed (the facts are pre-validated)");
        assert!(
            !errs.is_empty(),
            "the mapped errors must be non-empty for a rejected program"
        );
        assert!(
            errs.iter()
                .any(|e| matches!(e.loan_kind, Some(LoanKind::Exclusive))),
            "the freeze loan must be classified as Exclusive (&mut): {:?}",
            errs
        );
        assert!(
            errs.iter().all(|e| e.span.start != 0 || e.span.end != 0),
            "every mapped error must carry a real span (the diagnostic anchor)"
        );
    }

    /// TDD switch — RED: the engine path (`extract_cross_function` +
    /// `evaluate_rules` + `rules_to_borrow_errors`) must be EQUIVALENT
    /// to the handwritten `borrow_check_function` on every covered shape:
    /// same accept/reject AND the same error categories (is_read /
    /// is_exclusive / loan_kind) — the "no regression" contract for the
    /// production switch.
    fn engine_borrow_check<'input>(
        body: &[HirStmt<'input>],
        registry: &[(Symbol, bool, Option<TypeId>, Vec<usize>, SignatureFacts)],
    ) -> Vec<crate::hir::cfg_graph::BorrowError> {
        let cfg = CfgGraph::build_function(body, &[]);
        let (facts, infos) = extract_cross_function(
            &cfg,
            body,
            &[],
            registry,
            &crate::hir::types::TypeContext::new(),
        )
        .expect("facts extraction");
        let (_, events, _, _) =
            collect_borrow_data(&cfg, registry, &crate::hir::types::TypeContext::new());
        rules_to_borrow_errors(&facts, &infos, &events)
            .expect("the mapping must succeed (the facts are pre-validated)")
    }

    fn err_signature(errs: &[crate::hir::cfg_graph::BorrowError]) -> Vec<(bool, bool, u8)> {
        let kind = |k: Option<LoanKind>| match k {
            None => 0,
            Some(LoanKind::Exclusive) => 1,
            Some(LoanKind::ReadOnly) => 2,
        };
        let mut sig: Vec<(bool, bool, u8)> = errs
            .iter()
            .map(|e| (e.is_read, e.is_exclusive, kind(e.loan_kind)))
            .collect();
        sig.sort();
        sig
    }

    #[test]
    fn test_engine_switch_equivalence() {
        let cases: &[&str] = &[
            // Accepted: the loan dies at its last use.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = *r;
                 a = 5;
                 return x;
             }",
            // Accepted: a plain through-borrow.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return *r;
             }",
            // Rejected (E110): mutation while the borrow is live.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 a = 5;
                 return *r;
             }",
            // Rejected (E109): a READ of a &mut-frozen place.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = a;
                 *r = x;
                 return x;
             }",
            // Rejected (E112): two overlapping &mut loans, one live at
            // the other's issuance.
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 set s: &mut Int<32> = &mut a;
                 *s = 1;
                 return *r;
             }",
            // Rejected (ancestor-clobber): the loan survives the ancestor
            // clobber, the deref-write through it is a use-after-move.
            "def main() -> Int<32> {
                 set mut arr = [1, 2, 3];
                 set r: &mut Int<32> = &mut arr[1];
                 arr = [4, 5, 6];
                 *r = 7;
                 return 0;
             }",
            // Accepted: a loan whose lifetime spans a loop back-edge
            // (the borrow variable's last use inside the loop body).
            "def main() -> Int<32> {
                 set mut a = 0;
                 set r: &mut Int<32> = &mut a;
                 while a < 10 { *r = *r + 1; }
                 return *r;
             }",
            // Rejected (E112): TWO TEMPORARY loans at the same point
            // (the same-statement exclusivity — `f(&mut a, &mut a)`).
            "def f(x: &mut Int<32>, y: &mut Int<32>) -> Int<32> { *x = *y; return *x; }
             def main() -> Int<32> {
                 set mut a = 42;
                 return f(&mut a, &mut a);
             }",
            // NOTE: this probe is ILL-TYPED — `&ro a` requires a
            // REFERENCE operand (`a: Int<32>` — the E111 diagnostic
            // fires), so it does NOT test the claimed "a `&ro` loan does
            // not freeze mutations" proposition.  It is kept as an
            // engine-agreement probe: both engines must agree on the
            // borrow-check layer regardless of the E111 diagnostic.
            // (The proposition itself is tested elsewhere on a legal
            // `&ro` of a reference operand.)
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r = &ro a;
                 a = 5;
                 return *r;
             }",
        ];
        for src in cases {
            let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(src);
            let Some(item) = prog.items.iter().find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            }) else {
                panic!("expected a main function def");
            };
            let HirStmt::FunctionDef { body, .. } = item else {
                panic!("expected a function def");
            };
            let body = body.clone().unwrap_or_default();
            let hand =
                borrow_check_function(&body, &[], &[], &crate::hir::types::TypeContext::new());
            let engine = engine_borrow_check(&body, &[]);
            assert_eq!(
                hand.is_empty(),
                engine.is_empty(),
                "accept/reject mismatch on {src:?}: hand={:?} engine={:?}",
                err_signature(&hand),
                err_signature(&engine)
            );
            assert_eq!(
                err_signature(&hand),
                err_signature(&engine),
                "error-category mismatch on {src:?}"
            );
        }
    }

    #[test]
    fn test_mapping_e109_read_freeze() {
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 let x = a;
                 *r = x;
                 return x;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        let (_, events, _, _) =
            collect_borrow_data(&cfg, &[], &crate::hir::types::TypeContext::new());
        let errs = rules_to_borrow_errors(&facts, &infos, &events)
            .expect("the mapping must succeed (the facts are pre-validated)");
        assert!(
            errs.iter().any(|e| e.is_read),
            "the read of a &mut-frozen place must surface as E109 (is_read): {:?}",
            errs
        );
    }

    /// The E112 exclusivity mapping — two overlapping loans with one
    /// live at the other's issuance must surface as `is_exclusive`.
    #[test]
    fn test_mapping_e112_exclusivity() {
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 set s: &mut Int<32> = &mut a;
                 *s = 1;
                 return *r;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        let (_, events, _, _) =
            collect_borrow_data(&cfg, &[], &crate::hir::types::TypeContext::new());
        let errs = rules_to_borrow_errors(&facts, &infos, &events)
            .expect("the mapping must succeed (the facts are pre-validated)");
        assert!(
            errs.iter().any(|e| e.is_exclusive),
            "the overlapping live loans must surface as E112 (is_exclusive): {:?}",
            errs
        );
    }

    /// The AncestorClobber divergence: a loan on `p.y`
    /// survives an ancestor clobber `p = ...` — the primary checker does
    /// NOT kill it (Field/Index exclusion) — so the later use through the
    /// loan is rejected.  The Polonius extractor must mirror that
    /// exclusion, and the R1-R9 rules must now also detect the violation
    /// (previously `evaluate_rules` returned no error for this shape —
    /// the borrow-variable binding was lost when the HIR wrapped the
    /// borrow in an `Index`/`TypeAnnotated` node, so the loan degraded
    /// to a temporary with no liveness connection).
    #[test]
    fn test_polonius_ancestor_clobber_exclusion() {
        let (prog, diags) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut arr = [1, 2, 3];
                 set r: &mut Int<32> = &mut arr[1];
                 arr = [4, 5, 6];
                 *r = 7;
                 return 0;
             }",
        );
        assert!(!diags.is_empty(), "the post-pass must reject the program");
        let HirStmt::FunctionDef { body, .. } = &prog.items[0] else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("the facts extraction must succeed");
        // The exclusion itself  : the ancestor clobber must
        // INVALIDATE the Field/Index loan — not KILL it — mirroring the
        // primary checker's AncestorClobber exclusion.
        assert!(
            !facts.loan_invalidated_at.is_empty(),
            "the loan must be invalidated (not killed) by the ancestor clobber"
        );
        assert!(
            facts.loan_killed_at.is_empty(),
            "the Field/Index exclusion must prevent the kill"
        );
        // The rule-level detection: the borrow-variable binding must
        // survive the `Index`/`TypeAnnotated` HIR wrapping so the loan is
        // live through its deref-write — `evaluate_rules` must report the
        // violation (previously ERRS stayed empty for this shape).
        assert!(
            !evaluate_rules(&facts).is_empty(),
            "the rules engine must detect the ancestor-clobber use"
        );
    }

    /// The A(ρ) signature facts — `fn get(x: &mut T) -> &mut T` —
    /// `universal_region(x)` + `subset_base(x, output)`: the output borrow
    /// alive ⟹ the input borrow considered alive.
    #[test]
    fn test_signature_facts_borrow_return() {
        let sig = BorrowSignature {
            input_borrows: vec![(Symbol::intern("x"), false)],
            input_lifetimes: vec![None],
            output_borrows: vec![Symbol::intern("__ret")],
        };
        let facts = signature_facts(&sig, &[0]);
        assert_eq!(
            facts.universal_region,
            vec![0, 1],
            "the input AND the output are placeholder origins (multi-output)"
        );
        assert_eq!(
            facts.known_placeholder_subset,
            vec![(0, 1)],
            "known_placeholder_subset(input, output) — the DECLARED A(ρ) constraint"
        );
    }

    /// A signature without borrows produces no facts.
    #[test]
    fn test_signature_facts_no_borrows() {
        let facts = signature_facts(&BorrowSignature::default(), &[]);
        assert!(facts.universal_region.is_empty());
        assert!(facts.known_placeholder_subset.is_empty());
    }

    /// A function with TWO input borrows
    /// where the output derives from `x` only — `y` is NOT constrained
    /// (the conservative all-pairs encoding would wrongly freeze `y` while
    /// the output is alive).
    #[test]
    fn test_signature_facts_derivation_precision() {
        let sig = BorrowSignature {
            input_borrows: vec![(Symbol::intern("x"), false), (Symbol::intern("y"), true)],
            input_lifetimes: vec![None, None],
            output_borrows: vec![Symbol::intern("__ret")],
        };
        // The output derives from x (position 0) only.
        let facts = signature_facts(&sig, &[0]);
        assert_eq!(
            facts.universal_region,
            vec![0, 1, 2],
            "both inputs AND the output are placeholder origins"
        );
        assert_eq!(
            facts.known_placeholder_subset,
            vec![(0, 2)],
            "only the DERIVING input x constrains the output — y is free"
        );
    }

    /// The output-derivation analysis — `return x` derives from the
    /// `x` parameter; a local borrow return derives from nothing.
    #[test]
    fn test_derive_output_origins() {
        let body = first_body(
            "def get(x: &mut Int<32>, y: &mut Int<32>) -> &mut Int<32> {
                 return x;
             }",
        );
        let params = vec![Symbol::intern("x"), Symbol::intern("y")];
        assert_eq!(
            derive_output_origins(&body, &params),
            vec![0],
            "`return x` derives from the x parameter (position 0)"
        );
        // A local borrow return (a fresh value derived from a local) — no
        // input derivation.  The dangling check REJECTS this program
        // (returning a reference to a local), so the body is taken via
        // `check_source_keep_hir` (the derivation analysis is independent
        // of the dangling diagnostic).
        let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(
            "def get(x: &mut Int<32>) -> &mut Int<32> {
                 set mut a = 42;
                 set r: &mut Int<32> = &mut a;
                 return r;
             }",
        );
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("get"))
            })
            .expect("get function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body2 = body.clone().unwrap_or_default();
        assert!(
            derive_output_origins(&body2, &[Symbol::intern("x")]).is_empty(),
            "a local borrow return has no input derivation"
        );
    }

    /// The borrow signature extraction with the injected reference
    /// resolution.
    #[test]
    /// The multi-output extraction: TWO reference returns (`__ret0`,
    /// `__ret1`) must each register its own output origin and its own
    /// A(ρ) subset from every deriving input — the tuple/struct-return
    /// shape is no longer collapsed to a single output.
    #[test]
    fn test_extract_borrow_signature_multi_output() {
        let sig = extract_borrow_signature(
            &[
                (Symbol::intern("x"), true, false, None),
                (Symbol::intern("y"), true, true, None),
            ],
            2,
        );
        assert_eq!(
            sig.output_borrows,
            vec![Symbol::intern("__ret0"), Symbol::intern("__ret1")],
            "two output origins"
        );
        let facts = signature_facts(&sig, &[0, 1]);
        // inputs 0,1 + outputs 2,3 — each pushed once.
        assert_eq!(facts.universal_region, vec![0, 1, 2, 3]);
        // 2 inputs × 2 outputs = 4 A(ρ) subsets (x→ret0, x→ret1, y→ret0, y→ret1).
        assert_eq!(facts.known_placeholder_subset.len(), 4);
        for &(i, o) in &[(0u32, 2u32), (0, 3), (1, 2), (1, 3)] {
            assert!(
                facts.known_placeholder_subset.contains(&(i, o)),
                "missing A(ρ) {i}→{o}: {:?}",
                facts.known_placeholder_subset
            );
        }
    }

    /// The recursive return-type counting: a tuple of two references
    /// counts 2; a struct field reference counts via the ADT definition.
    #[test]
    fn test_count_return_refs_recursive() {
        use crate::ast::{Span, Type};
        let path = Type::Path(smallvec::smallvec![], Span::new(0, 1));
        let r1 = Type::Reference {
            inner: &path,
            mutable: true,
            lifetime: None,
            span: Span::new(0, 1),
        };
        assert_eq!(count_return_refs_ast(&r1), 1);
        let r2 = Type::Reference {
            inner: &path,
            mutable: false,
            lifetime: None,
            span: Span::new(0, 1),
        };
        let tup = Type::Tuple(vec![r1, r2], Span::new(0, 1));
        assert_eq!(
            count_return_refs_ast(&tup),
            2,
            "a tuple of two references counts 2 outputs"
        );
    }

    /// The EARLY-BOUND mapping: two inputs with the SAME explicit
    /// lifetime (`&'a mut T` and `&'a T`) share ONE placeholder origin —
    /// the rustc `UniversalRegions` early-bound region analog.  The
    /// elided-lifetime case (all `None`) keeps one origin per input.
    #[test]
    fn test_shared_lifetime_same_origin() {
        let a = Symbol::intern("a");
        let sig = BorrowSignature {
            input_borrows: vec![(Symbol::intern("x"), true), (Symbol::intern("y"), false)],
            input_lifetimes: vec![Some(a), Some(a)],
            output_borrows: vec![Symbol::intern("__ret0")],
        };
        let facts = signature_facts(&sig, &[0, 1]);
        // BOTH inputs push origin 0 (the shared 'a — the duplicate IS
        // the sharing evidence); the output is 1.
        assert_eq!(
            facts.universal_region,
            vec![0, 0, 1],
            "the shared-'a inputs must map to ONE origin: {:?}",
            facts.universal_region
        );
        assert!(
            facts.known_placeholder_subset.contains(&(0, 1)),
            "both deriving inputs flow to the output: {:?}",
            facts.known_placeholder_subset
        );
    }

    /// The ADT generic-parameter substitution: a struct field `a: T`
    /// whose arg is a reference must count as an output (the
    /// `count_return_refs` Adt branch substitutes GenericParam with the
    /// concrete args).
    #[test]
    fn test_count_return_refs_adt_generic() {
        // Build a TypeContext with a struct `S { a: T }` where T is bound
        // to a reference — assert the field substitution counts it.
        let mut ctx = crate::hir::types::TypeContext::new();
        let t_param = ctx.alloc(crate::hir::types::TypeData::GenericParam {
            index: 0,
            name: Symbol::intern("T"),
        });
        let s_def = crate::hir::types::AdtDef {
            fields: vec![t_param],
            has_drop: false,
        };
        let ref_id = ctx.alloc(crate::hir::types::TypeData::Ref {
            ty: t_param,
            mutable: true,
            lifetime: None,
        });
        let s_id = ctx.alloc(crate::hir::types::TypeData::Adt {
            kind: crate::hir::types::AdtKind::Struct,
            def_id: crate::hir::types::DefId(1),
            args: vec![ref_id],
        });
        ctx.register_adt(crate::hir::types::DefId(1), s_def);
        assert_eq!(
            count_return_refs(&ctx, s_id),
            1,
            "the generic field T substituted with a &mut T must count 1"
        );
    }

    /// A reborrow-kill regression guard: the reborrow-kill (`r = r2` —
    /// an assignment to the
    /// borrow VARIABLE kills the OLD loan).  The kills vector was
    /// previously dropped in `extract_facts`, so the old source stayed
    /// frozen past the reborrow (a false positive).  After the fix, the
    /// old loan is dead at the reassignment and the source is writable.
    #[test]
    fn test_reborrow_kill_engine_path() {
        let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(
            "def main() -> Int<32> {
                 set mut a = 1;
                 set mut b = 2;
                 set r = &mut a;
                 set r2 = &mut b;
                 r = r2;
                 a = 5;
                 *r = 9;
                 return a;
             }",
        );
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            })
            .expect("main function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        let (facts, _infos, _events) = extract_facts(
            &cfg,
            &body,
            &[],
            &[],
            &crate::hir::types::TypeContext::new(),
        )
        .expect("facts extraction");
        assert!(
            evaluate_rules(&facts).is_empty(),
            "the reborrow `r = r2` must kill the OLD loan — `a = 5` afterwards is legal: {:?}",
            evaluate_rules(&facts)
        );
    }

    /// TPB (Two-Phase Borrow) unit: a method-call receiver loan marked
    /// `two_phase` must NOT freeze a READ at its own issuance point — the
    /// argument evaluation (`v.push(v.len())` reads `v` during the
    /// reservation).  Without the flag, the same read fires E109.
    #[test]
    fn test_two_phase_reserved_read_exempt() {
        use crate::hir::cfg_graph::Point as CfgPoint;
        let pt0 = CfgPoint {
            block: crate::hir::cfg_graph::BlockId(0),
            stmt: 0,
            expr: 0,
        };
        let span = crate::ast::Span::new(0, 1);
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 1)],
            loan_issued_at: vec![(0, 0, 0)],
            var_used_at: vec![(99, 0)],
            use_of_var_derefs_origin: vec![(99, 0)],
            ..Default::default()
        };
        let place = FrozenPlace::Root(Symbol::intern("v"));
        let infos = vec![LoanInfo {
            id: 0,
            origin: 0,
            place: place.clone(),
            borrow_var: Some(Symbol::intern("v")),
            kind: LoanKind::Exclusive,
            point: pt0,
            span,
            two_phase: true,
        }];
        // A READ of `v` AT the loan's issuance point (the reserved phase).
        let events = vec![(place, pt0, true, span)];
        let errs = rules_to_borrow_errors(&facts, &infos, &events)
            .expect("the mapping must succeed (the facts are pre-validated)");
        assert!(
            !errs.iter().any(|e| e.is_read),
            "the reserved-phase read must be exempt (TPB): {:?}",
            errs
        );
        // Without the TPB flag the same read fires E109.
        let mut plain = infos.clone();
        plain[0].two_phase = false;
        let errs_plain = rules_to_borrow_errors(&facts, &plain, &events)
            .expect("the mapping must succeed (the facts are pre-validated)");
        assert!(
            errs_plain.iter().any(|e| e.is_read),
            "a non-TPB exclusive loan must freeze the read: {:?}",
            errs_plain
        );
    }

    /// The METHOD-CALL RECEIVER loan end-to-end: `obj.put(5)` (a
    /// `&mut self` method) — the receiver place `obj` must be collected
    /// as a two-phase receiver by `collect_method_receiver_places` (the
    /// TPB marking input).  The receiver LOAN itself is issued on the
    /// cross-function path (`extract_cross_function`, where the method
    /// signature registry provides the receiver mutability); plain
    /// `extract_facts` has no registry and issues no receiver loan.
    #[test]
    fn test_method_receiver_loan_two_phase() {
        let (prog, _) = crate::hir::checker::tests::check_source_keep_hir(
            "type MyType = struct { val: Int<32> }
             impl for MyType {
                 def put(&mut self, x: Int<32>) -> Int<32> { self.val = x; return x; }
             }
             def main() -> Int<32> {
                 set obj = MyType { val = 0 };
                 set r = obj.put(5);
                 return r;
             }",
        );
        let item = prog
            .items
            .iter()
            .find(|i| {
                matches!(i, HirStmt::FunctionDef { name, .. } if name == &Symbol::intern("main"))
            })
            .expect("main function");
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let mut receiver_places: HashSet<FrozenPlace> = HashSet::new();
        collect_method_receiver_places(&body, &mut receiver_places);
        let obj = FrozenPlace::Root(Symbol::intern("obj"));
        assert!(
            receiver_places.contains(&obj),
            "the method-call receiver place `obj` must be collected for TPB: {:?}",
            receiver_places
        );
    }

    /// The drop-while-borrowed — a dropped value whose origin still
    /// carries a LIVE loan must fire an error (the borrow outlives the
    /// value).  The `drop_of_var_derefs_origin` facts were extracted but
    /// previously had NO rule consuming them.
    #[test]
    fn test_drop_while_borrowed() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 1)],
            // (origin, loan, point): the loan 0 (origin 0) issued at 0.
            loan_issued_at: vec![(0, 0, 0)],
            var_used_at: vec![(99, 0)],
            use_of_var_derefs_origin: vec![(99, 0)],
            // The value 99 is DROPPED at point 0 — its origin (0) still
            // carries the live loan 0 → the drop-while-borrowed error.
            var_dropped_at: vec![(99, 0)],
            drop_of_var_derefs_origin: vec![(99, 0)],
            ..Default::default()
        };
        let errs = evaluate_drop_errors(&facts);
        // Hand-computed oracle (independent of the official polonius-engine,
        // which has NO drop-while-borrowed rule — the drop rule is a
        // Posita-specific extension, so the differential oracle cannot
        // cross-validate it): the loan 0 (origin 0, issued at point 0) is
        // still live at point 0, where the value 99 (origin 0) is dropped —
        // exactly the (loan, point) pair (0, 0).
        assert_eq!(
            errs,
            vec![(0, 0)],
            "the drop-while-borrowed must yield exactly the (loan, point) pair (0, 0)"
        );
    }

    fn test_extract_borrow_signature() {
        let sig = extract_borrow_signature(
            &[
                (Symbol::intern("x"), true, false, None),
                (Symbol::intern("n"), false, false, None),
            ],
            1,
        );
        assert_eq!(sig.input_borrows, vec![(Symbol::intern("x"), false)]);
        assert!(!sig.output_borrows.is_empty());
        let sig2 = extract_borrow_signature(&[(Symbol::intern("x"), true, true, None)], 0);
        assert_eq!(sig2.input_borrows.len(), 1);
        assert!(sig2.output_borrows.is_empty());
    }

    /// The A(ρ) subset semantics — an input loan (issued in the
    /// universal input origin 0) propagates to the output origin 1 (R4 via
    /// `subset_base(0, 1, _)`), so it stays live wherever the OUTPUT
    /// origin is live — the input source stays frozen while the returned
    /// borrow is alive.
    #[test]
    fn test_subset_arho_propagation() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 1), (1, 2)],
            subset_base: vec![(0, 1, 0)], // the call-site instantiation
            universal_region: vec![0],
            loan_issued_at: vec![(0, 0, 0)], // the input loan in origin 0
            loan_invalidated_at: vec![(0, 2)], // the source mutated at 2
            var_used_at: vec![(99, 2)],      // the output var used at 2
            use_of_var_derefs_origin: vec![(99, 1)], // the output var → origin 1
            known_placeholder_subset: vec![(0, 1)], // the DECLARED A(ρ) relation
            loan_killed_at: vec![],
            var_defined_at: vec![],
            var_dropped_at: vec![],
            drop_of_var_derefs_origin: vec![],
            child_path: vec![],
            path_is_var: vec![],
            reborrow_sources: vec![],
        };
        let errs = evaluate_rules(&facts);
        assert!(
            errs.contains(&(0, 2)),
            "the input loan must be live at the output's live point (A(ρ) via R4)"
        );
        // The declared relation → no subset errors.
        assert!(evaluate_subset_errors(&facts).is_empty());
    }

    /// The subset-rejection case: a subset relation between two
    /// placeholder origins NOT declared in the signature fires the R9
    /// subset_errors.
    #[test]
    fn test_subset_errors_r9() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 1)],
            subset_base: vec![(0, 2, 0)], // input 0 ⊆ input 2 — NOT declared
            universal_region: vec![0, 2], // both placeholders
            ..Default::default()
        };
        let errs = evaluate_subset_errors(&facts);
        assert!(
            errs.contains(&(0, 2, 0)),
            "the undeclared placeholder subset must be rejected"
        );
    }

    /// R9 wiring verification: an undeclared placeholder subset must
    /// surface through `rules_to_borrow_errors` as an E115 borrow error
    /// (the `Root("<region>")` marker that the diagnostic maps to
    /// "region subset not declared in signature").
    #[test]
    fn test_r9_wired_into_borrow_errors() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 1)],
            subset_base: vec![(0, 2, 0)], // input 0 ⊆ input 2 — NOT declared
            universal_region: vec![0, 2], // both placeholders
            ..Default::default()
        };
        let errs = rules_to_borrow_errors(&facts, &[], &[])
            .expect("the mapping must succeed (the facts are pre-validated)");
        let r9 = errs
            .iter()
            .find(|e| matches!(&e.place, FrozenPlace::Root(n) if n.eq_str("<region>")));
        assert!(
            r9.is_some(),
            "the undeclared placeholder subset must surface as an R9 borrow error"
        );
        let diag = crate::hir::cfg_graph::borrow_error_diagnostic(r9.unwrap());
        assert!(
            format!("{diag}").contains("region subset not declared in signature"),
            "the R9 diagnostic must carry the E115 message"
        );
    }

    /// E116 wiring verification: a dropped value whose loan is still live
    /// at the drop point must surface through `rules_to_borrow_errors` as
    /// a `is_drop` borrow error (the marker `borrow_error_diagnostic`
    /// maps to "cannot drop a value while its borrow is still live").
    #[test]
    fn test_drop_errors_wired_into_borrow_errors() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 16), (16, 32)],
            loan_issued_at: vec![(0, 0, 16)],
            loan_invalidated_at: vec![(0, 32)],
            var_used_at: vec![(99, 32)],
            use_of_var_derefs_origin: vec![(99, 0)],
            // The DROP facts: var 99 (the loan's origin holder) is dropped
            // at 32 while its loan is live there — the drop-while-borrowed
            // violation.  The drop derefs origin 0 (the loan's origin),
            // so the loan is live at the drop point.
            var_dropped_at: vec![(99, 32)],
            drop_of_var_derefs_origin: vec![(99, 0)],
            ..Default::default()
        };
        let infos = vec![LoanInfo {
            id: 0,
            origin: 0,
            place: FrozenPlace::Root(Symbol::intern("v0")),
            borrow_var: None,
            kind: LoanKind::Exclusive,
            point: Point {
                block: crate::hir::cfg_graph::BlockId(0),
                stmt: 0,
                expr: 16,
            },
            span: crate::ast::Span::new(10, 20),
            two_phase: false,
        }];
        let errs = rules_to_borrow_errors(&facts, &infos, &[])
            .expect("the mapping must succeed (the facts are pre-validated)");
        let drop_err = errs.iter().find(|e| e.is_drop);
        assert!(
            drop_err.is_some(),
            "the drop-while-borrowed violation must surface as an E116 borrow error: {:?}",
            errs
        );
        let diag = crate::hir::cfg_graph::borrow_error_diagnostic(drop_err.unwrap());
        assert!(
            format!("{diag}").contains("cannot drop a value while its borrow is still live"),
            "the E116 diagnostic must carry the drop-while-borrowed message"
        );
    }

    /// The dedup pass: multiple violation points for the SAME loan must
    /// collapse into ONE identical diagnostic (a loan invalidated at
    /// several points would otherwise spam the same E110 message).
    #[test]
    fn test_borrow_errors_dedup() {
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 16), (16, 32), (32, 48)],
            loan_issued_at: vec![(0, 0, 16)],
            loan_invalidated_at: vec![(0, 32), (0, 48)], // two violations
            var_used_at: vec![(99, 32), (99, 48)],
            use_of_var_derefs_origin: vec![(99, 0)],
            ..Default::default()
        };
        let infos = vec![LoanInfo {
            id: 0,
            origin: 0,
            place: FrozenPlace::Root(Symbol::intern("v0")),
            borrow_var: None,
            kind: LoanKind::Exclusive,
            point: Point {
                block: crate::hir::cfg_graph::BlockId(0),
                stmt: 0,
                expr: 16,
            },
            span: crate::ast::Span::new(10, 20),
            two_phase: false,
        }];
        let errs = rules_to_borrow_errors(&facts, &infos, &[])
            .expect("the mapping must succeed (the facts are pre-validated)");
        let e110 = errs
            .iter()
            .filter(|e| !e.is_read && !e.is_exclusive && !e.is_drop);
        assert_eq!(
            e110.count(),
            1,
            "the two identical E110 violations must deduplicate into one: {:?}",
            errs
        );
    }

    /// `loan_killed_at`: a loan KILLED before its invalidated point is NOT
    /// live there — no error (the R5 `!loan_killed_at` condition).
    #[test]
    fn test_loan_killed_semantics() {
        // The statement-level points (16-aligned, matching the CFG
        // encoding): issuance at 16, kill at 16, invalidation at 32.
        let facts = PoloniusFacts {
            cfg_edge: vec![(0, 16), (16, 32)],
            loan_issued_at: vec![(0, 0, 16)],
            loan_invalidated_at: vec![(0, 32)],
            loan_killed_at: vec![(0, 16)], // killed before the invalidated point
            var_used_at: vec![(99, 32)],
            use_of_var_derefs_origin: vec![(99, 0)],
            ..Default::default()
        };
        let errs = evaluate_rules(&facts);
        assert!(
            !errs.contains(&(0, 32)),
            "the killed loan must not be live at the invalidated point"
        );
    }

    /// The call-site instantiation — `get(&mut a)` with the A(ρ)
    /// relation (input 0 → output 1): the argument loan issued in the
    /// input origin 0 at the call point + the `subset_base(0, 1, point)`
    /// instantiation.
    #[test]
    fn test_call_site_facts() {
        let (loans, subset) = call_site_facts(&[0], 7, &[(0, 1)], 0);
        assert_eq!(
            loans,
            vec![(0, 0, 7)],
            "the argument loan issued in the input origin 0 at the call point"
        );
        assert_eq!(
            subset,
            vec![(0, 1, 7)],
            "the A(ρ) instantiation: subset_base(0, 1, call_point)"
        );
        // No input borrows → no facts.
        let (loans2, subset2) = call_site_facts(&[], 7, &[(0, 1)], 0);
        assert!(loans2.is_empty());
        assert!(subset2.is_empty());
    }

    /// The full A(ρ) cross-function scenario — `get(&mut a); *r = 10;
    /// a = 5;` — the input loan (of `a`) propagates to the output origin
    /// (R4 via the call-site `subset_base` instantiation), so mutating `a`
    /// while the returned borrow `r` is alive is an error.
    #[test]
    fn test_cross_function_arho_scenario() {
        // Signature facts: get(x: &mut T) -> &mut T — universal_region [0],
        // known_placeholder_subset [(0, 1)].
        let sig = signature_facts(
            &BorrowSignature {
                input_borrows: vec![(Symbol::intern("x"), false)],
                input_lifetimes: vec![None],
                output_borrows: vec![Symbol::intern("__ret")],
            },
            &[0],
        );
        // The call-site facts at point 16: the argument loan (0) in origin
        // 0 + subset_base(0, 1, 16) — the A(ρ) instantiation.
        let (call_loans, call_subset) = call_site_facts(&[0], 16, &sig.known_placeholder_subset, 0);
        // The body facts: the statement-level points 0,16,32,48,64; the
        // returned borrow's var used at 64 (its last use); the source
        // mutated at 32.
        let mut facts = PoloniusFacts {
            cfg_edge: vec![(0, 16), (16, 32), (32, 48), (48, 64)],
            var_used_at: vec![(99, 64)],
            use_of_var_derefs_origin: vec![(99, 1)],
            loan_invalidated_at: vec![(0, 32)],
            ..Default::default()
        };
        facts.universal_region.extend(sig.universal_region);
        facts
            .known_placeholder_subset
            .extend(sig.known_placeholder_subset);
        facts.loan_issued_at.extend(call_loans);
        facts.subset_base.extend(call_subset);
        // The mutation of `a` at 32: the input loan (0) is live there
        // (propagated to the output origin 1 — live via the returned
        // borrow's use at 64) → an error.
        let errs = evaluate_rules(&facts);
        assert!(
            errs.contains(&(0, 32)),
            "the A(ρ) semantics: the input source stays frozen while the returned borrow is alive"
        );
        assert!(
            evaluate_subset_errors(&facts).is_empty(),
            "the declared A(ρ) relation must not fire subset errors"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
// The FlowLog runner: write the facts as the polonius_int.dl CSV inputs
// and invoke the external FlowLog engine (compiled to an executable).
// ────────────────────────────────────────────────────────────────────────

use std::path::Path;

/// The polonius_int.dl input file names (the full 17-input schema).
pub const POLONIUS_INPUTS: &[&str] = &[
    "subset_base.csv",
    "cfg_edge.csv",
    "loan_issued_at.csv",
    "universal_region.csv",
    "var_used_at.csv",
    "loan_killed_at.csv",
    "known_placeholder_subset.csv",
    "var_dropped_at.csv",
    "drop_of_var_derefs_origin.csv",
    "var_defined_at.csv",
    "child_path.csv",
    "path_moved_at_base.csv",
    "path_assigned_at_base.csv",
    "path_accessed_at_base.csv",
    "path_is_var.csv",
    "loan_invalidated_at.csv",
    "use_of_var_derefs_origin.csv",
];

/// Write the facts as the polonius_int.dl CSV inputs in `dir` (all 17
/// inputs — the unpopulated ones are empty files).
pub fn write_facts_csv(facts: &PoloniusFacts, dir: &Path) -> std::io::Result<()> {
    let mut s = String::new();
    for (a, b) in &facts.cfg_edge {
        s.push_str(&format!("{a},{b}\n"));
    }
    std::fs::write(dir.join("cfg_edge.csv"), &s)?;
    s.clear();
    for (l, o, p) in &facts.loan_issued_at {
        // The polonius_int.dl / official AllFacts schema is
        // `loan_issued_at(origin, loan, point)` — the internal
        // (loan, origin, point) order must be TRANSPOSED here (the
        // `to_official` helper does the same for the polonius-engine
        // oracle; previously the CSV was written in the internal order,
        // corrupting the FlowLog differential input).
        s.push_str(&format!("{o},{l},{p}\n"));
    }
    std::fs::write(dir.join("loan_issued_at.csv"), &s)?;
    s.clear();
    for (l, p) in &facts.loan_invalidated_at {
        // The official schema is `loan_invalidated_at(point, loan)` —
        // transposed from the internal (loan, point) order.
        s.push_str(&format!("{p},{l}\n"));
    }
    std::fs::write(dir.join("loan_invalidated_at.csv"), &s)?;
    s.clear();
    for (v, p) in &facts.var_used_at {
        s.push_str(&format!("{v},{p}\n"));
    }
    std::fs::write(dir.join("var_used_at.csv"), &s)?;
    s.clear();
    for (v, o) in &facts.use_of_var_derefs_origin {
        s.push_str(&format!("{v},{o}\n"));
    }
    std::fs::write(dir.join("use_of_var_derefs_origin.csv"), &s)?;
    s.clear();
    for (a, b, p) in &facts.subset_base {
        s.push_str(&format!("{a},{b},{p}\n"));
    }
    std::fs::write(dir.join("subset_base.csv"), &s)?;
    s.clear();
    for o in &facts.universal_region {
        s.push_str(&format!("{o}\n"));
    }
    std::fs::write(dir.join("universal_region.csv"), &s)?;
    s.clear();
    for (a, b) in &facts.known_placeholder_subset {
        s.push_str(&format!("{a},{b}\n"));
    }
    std::fs::write(dir.join("known_placeholder_subset.csv"), &s)?;
    s.clear();
    for (l, p) in &facts.loan_killed_at {
        s.push_str(&format!("{l},{p}\n"));
    }
    std::fs::write(dir.join("loan_killed_at.csv"), &s)?;
    // The remaining inputs are empty in this model.
    for name in POLONIUS_INPUTS {
        if !dir.join(name).exists() {
            std::fs::write(dir.join(name), "")?;
        }
    }
    Ok(())
}

/// Run the external FlowLog engine on the facts (the polonius_int.dl
/// program compiled to an executable, executed with the CSV inputs) and
/// return the number of `errors(loan, point)` pairs.  Returns `None`
/// whenever FlowLog is unavailable (binary or .dl missing, or the
/// invocation fails) — the test then skips gracefully.
pub fn run_flowlog(facts: &PoloniusFacts) -> Option<usize> {
    // The FlowLog install location — configured ONLY via the
    // `PONENT_FLOWLOG_DIR` environment variable; an industrial-grade
    // compiler must never fall back to a hardcoded local path.  When
    // unset, FlowLog is unavailable and the cross-validation skips
    // gracefully.
    let base = std::env::var("PONENT_FLOWLOG_DIR").ok()?;
    let flowlog = Path::new(&base).join("target/release/flowlog-compiler");
    let dl = Path::new(&base).join("example/program_analysis/polonius_int.dl");
    if !flowlog.exists() || !dl.exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("posita_flowlog_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    write_facts_csv(facts, &dir).ok()?;
    let exe = dir.join("polonius_prog");
    let out = dir.join("out");
    let _ = std::fs::create_dir_all(&out);
    // Compile the .dl program to an executable.
    let status = std::process::Command::new(flowlog)
        .arg("-o")
        .arg(&exe)
        .arg("-F")
        .arg(&dir)
        .arg("-D")
        .arg(&out)
        .arg(dl)
        .status()
        .ok()?;
    if !status.success() || !exe.exists() {
        return None;
    }
    // Run the executable.
    let run = std::process::Command::new(&exe).status().ok()?;
    if !run.success() {
        return None;
    }
    // Parse the errors relation size from the output directory.
    for entry in std::fs::read_dir(&out).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.contains("errors") {
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                for line in text.lines() {
                    if let Some(n) = line.trim().strip_prefix("errors:") {
                        return n.trim().parse().ok();
                    }
                }
            }
        }
    }
    None
}
