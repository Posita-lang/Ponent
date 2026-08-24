//! Control-flow graph (CFG) infrastructure for Ponent.
//!
//! NOTE: this module is named `cfg_graph` because `src/hir/cfg.rs` is
//! already taken by the `@cfg` CONDITIONAL-COMPILATION evaluator (a
//! different "cfg").
//!
//! The graph is the standard block-and-terminator design (rustc MIR:
//! straight-line statements per block, a terminator deciding the
//! successors; ante: index-based block ids; GHC Cmm: open/close blocks).
//! It is built per function body from the HIR and is the shared
//! infrastructure for any flow-sensitive pass (borrow-check liveness,
//! reachability, dead-code, comptime analysis).

use crate::ast::UnaryOp;
use crate::diagnostics::Diagnostic;
use crate::hir::hir::{HirExpr, HirStmt};
use crate::hir::place::place_is_prefix_of;
use crate::hir::types::{FrozenPlace, LoanKind};
use crate::symbol::Symbol;
use std::collections::{HashMap, HashSet};

/// Index-based block identifier (ante-style: cheap, stable).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockId(pub usize);

/// Terminator<'input> of a basic block: decides the successors.
#[derive(Clone, Debug)]
pub enum Terminator<'input> {
    /// Unconditional jump.
    Goto(BlockId),
    /// Conditional branch (`if` / `if let`): cond true → `then_`, false → `else_`.
    Branch {
        cond: Box<HirExpr<'input>>,
        then_: BlockId,
        else_: BlockId,
    },
    /// Conservative dual-edge branch for a `match` arm whose pattern is
    /// REFUTABLE but cannot be synthesized into a boolean `HirExpr`
    /// condition (Enum / Struct / Tuple / Or / Slice).  Both successors
    /// are possible — the pattern may match OR fail — so both the arm
    /// body (`then_`) and the fall-through (`else_` = the next arm's test
    /// block / the match exit) stay reachable.  A `Goto(then_)` here
    /// would sever the edge to later arms, making them unreachable for
    /// the flow-sensitive analyses (borrow / move) — an under-
    /// approximation that hides use-after-move and freeze conflicts.
    Switch { then_: BlockId, else_: BlockId },
    /// Function exit (`return`).
    Return,
    /// `leave with` exit.
    Leave,
    /// No valid successor (statement-list end without return).
    Unreachable,
}

/// A basic block: a maximal straight-line sequence of statements plus a
/// terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock<'input> {
    pub stmts: Vec<HirStmt<'input>>,
    pub terminator: Option<Terminator<'input>>,
}

impl<'input> BasicBlock<'input> {
    fn new() -> Self {
        BasicBlock {
            stmts: Vec::new(),
            terminator: None,
        }
    }
}

/// The control-flow graph of one function body.
#[derive(Clone, Debug)]
pub struct CfgGraph<'input> {
    blocks: Vec<BasicBlock<'input>>,
    entry: BlockId,
    successors: Vec<Vec<BlockId>>,
    predecessors: Vec<Vec<BlockId>>,
    /// Back edges: `(from, to)` where `to` is an ancestor of `from` in a
    /// DFS from the entry — these form the loop edges.
    back_edges: Vec<(BlockId, BlockId)>,
}

impl<'input> CfgGraph<'input> {
    pub fn build_function(
        body: &[HirStmt<'input>],
        finally: &[HirStmt<'input>],
    ) -> CfgGraph<'input> {
        let mut b = CfgBuilder::new();
        // The entry is the FIRST block (build_seq always starts a new block
        // for the sequence).  The return value of build_seq is the JOIN
        // point (or None when the sequence ended in return/leave), NOT the
        // entry.
        b.build_seq(body, None);
        b.attach_finally(finally);
        b.finish(BlockId(0))
    }

    pub fn entry(&self) -> BlockId {
        self.entry
    }

    pub fn blocks(&self) -> &[BasicBlock<'input>] {
        &self.blocks
    }

    pub fn block(&self, id: BlockId) -> &BasicBlock<'input> {
        &self.blocks[id.0]
    }

    pub fn successors(&self, id: BlockId) -> &[BlockId] {
        &self.successors[id.0]
    }

    pub fn predecessors(&self, id: BlockId) -> &[BlockId] {
        &self.predecessors[id.0]
    }

    /// Whether `to` is reachable from `from` via the FORWARD edges only
    /// (back-edges excluded) — the path-order birth bound.  A back-edge
    /// conflates a PRE-issuance execution in an earlier iteration with a
    /// same-iteration path after the issuance (Case B: the
    /// mutual reachability wrongly hid a mutation of a frozen variable in
    /// a different block of the same loop iteration).
    pub fn forward_reaches(&self, from: BlockId, to: BlockId) -> bool {
        let mut visited = vec![false; self.blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            if b == to {
                return true;
            }
            if visited[b.0] {
                continue;
            }
            visited[b.0] = true;
            for &s in self.successors(b) {
                if self.back_edges.contains(&(b, s)) {
                    continue; // skip the back-edge (the loop conflation)
                }
                stack.push(s);
            }
        }
        false
    }

    /// Whether `to` is reachable from `from` via the forward edges (the
    /// CFG forward reachability — for the loan issuance-bound check).
    pub fn reaches(&self, from: BlockId, to: BlockId) -> bool {
        let mut visited = vec![false; self.blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            if b == to {
                return true;
            }
            if visited[b.0] {
                continue;
            }
            visited[b.0] = true;
            for &s in self.successors(b) {
                stack.push(s);
            }
        }
        false
    }

    /// Whether `from` dominates `to` — every path from the ENTRY block to
    /// `to` passes through `from`.  The loan birth bound needs this
    /// Back edges (loop edges) of the graph.
    pub fn back_edges(&self) -> &[(BlockId, BlockId)] {
        &self.back_edges
    }

    /// Compute the per-block variable liveness (backward dataflow): a
    /// variable is live at a block's entry if it is used in the block or
    /// live at the entry of a successor.  This is the "live until last
    /// use" machinery (rustc NLL: sparse interval sets; here: block-level
    /// fixpoint over the CFG).
    ///
    /// The liveness is an over-approximation (defs are not subtracted —
    /// a variable counts as live from the entry of any block that uses it
    /// or a successor), which is the conservative direction for the
    /// borrow-check consumer: a loan (source S, borrow variable B) is
    /// live at block Q iff B is live at Q.
    pub fn compute_var_liveness(&self) -> VarLiveness {
        let n = self.blocks.len();
        // uses(b): variables appearing in block b's statements/terminator.
        let mut uses: Vec<HashSet<Symbol>> = (0..n).map(|_| HashSet::new()).collect();
        for (i, blk) in self.blocks.iter().enumerate() {
            for stmt in &blk.stmts {
                for v in used_vars_in_stmt(stmt) {
                    uses[i].insert(v);
                }
            }
            if let Some(Terminator::Branch { cond, .. }) = &blk.terminator {
                for v in used_vars_in_expr(cond) {
                    uses[i].insert(v);
                }
            }
        }
        // Backward fixpoint: live_in(b) = uses(b) ∪ ⋃ live_in(succ(b)).
        let mut live_in: Vec<HashSet<Symbol>> = (0..n).map(|_| HashSet::new()).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                let mut cur: HashSet<Symbol> = uses[i].clone();
                for &s in &self.successors[i] {
                    cur.extend(live_in[s.0].iter().cloned());
                }
                if cur != live_in[i] {
                    live_in[i] = cur;
                    changed = true;
                }
            }
        }
        VarLiveness { live_in }
    }

    /// Point-level variable liveness (backward dataflow at statement
    /// granularity): a variable is live at a statement point if it is used
    /// at that statement, at a later statement in the same block, or at
    /// the entry of a successor block.  This delivers the point-level
    /// "last use"
    /// precision even WITHIN a block (the borrow variable's last use ends
    /// the loan at the exact statement, not the block).
    pub fn compute_point_liveness(&self) -> PointLiveness {
        let n = self.blocks.len();
        // uses(b)[i]: variables used by block b's i-th statement, with the
        // terminator's uses at index stmts.len().
        let mut uses: Vec<Vec<HashSet<Symbol>>> = Vec::new();
        for blk in &self.blocks {
            let mut b: Vec<HashSet<Symbol>> = blk
                .stmts
                .iter()
                .map(|s| used_vars_in_stmt(s).into_iter().collect())
                .collect();
            let mut t = HashSet::new();
            if let Some(Terminator::Branch { cond, .. }) = &blk.terminator {
                for v in used_vars_in_expr(cond) {
                    t.insert(v);
                }
            }
            b.push(t);
            uses.push(b);
        }
        // Backward fixpoint over points: live(b,i) = uses(b,i) ∪ live(b,i+1)
        // within the block; the terminator point joins the successors.
        let mut live_at: Vec<Vec<HashSet<Symbol>>> = uses.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for b in 0..n {
                let len = live_at[b].len();
                let mut cur: HashSet<Symbol> = uses[b][len - 1].clone();
                for &s in &self.successors[b] {
                    cur.extend(live_at[s.0][0].iter().cloned());
                }
                if cur != live_at[b][len - 1] {
                    live_at[b][len - 1] = cur;
                    changed = true;
                }
                for i in (0..len - 1).rev() {
                    let mut cur: HashSet<Symbol> = uses[b][i].clone();
                    cur.extend(live_at[b][i + 1].iter().cloned());
                    if cur != live_at[b][i] {
                        live_at[b][i] = cur;
                        changed = true;
                    }
                }
            }
        }
        // Compress the live sets into per-variable sparse intervals
        // (rustc's SparseIntervalMatrix concept — a variable is typically
        // live over a contiguous range, from its definition to its last
        // use).
        let mut intervals: Vec<HashMap<Symbol, Vec<(usize, usize)>>> = Vec::new();
        for blk_live in &live_at {
            let mut by_var: HashMap<Symbol, Vec<usize>> = HashMap::new();
            for (i, vars) in blk_live.iter().enumerate() {
                for v in vars {
                    by_var.entry(*v).or_default().push(i);
                }
            }
            let mut runs_map = HashMap::new();
            for (v, mut pts) in by_var {
                pts.sort_unstable();
                let mut runs: Vec<(usize, usize)> = Vec::new();
                for p in pts {
                    match runs.last_mut() {
                        Some((_, end)) if *end + 1 == p => *end = p,
                        _ => runs.push((p, p)),
                    }
                }
                runs_map.insert(v, runs);
            }
            intervals.push(runs_map);
        }
        PointLiveness { uses, intervals }
    }
}

/// Per-block variable liveness (live at block entry).
#[derive(Clone, Debug)]
pub struct VarLiveness {
    live_in: Vec<HashSet<Symbol>>,
}

/// A point in the CFG: a statement position within a block.  Point-level
/// liveness (rustc's point granularity) delivers the point-level
/// "last use"
/// precision even WITHIN a block — the block-level approximation could
/// not distinguish the borrow's last use from a later statement in the
/// same block.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Point {
    pub block: BlockId,
    pub stmt: usize,
    /// The expression index WITHIN the statement — expression-level
    /// points (aligned with rustc's CFG points): a statement's write,
    /// read, and borrow operations each get their own point, so their
    /// relative ORDER is decidable (a same-statement write-then-reborrow
    /// no longer needs the R8 same-point exemption).  `0` for
    /// statement-level uses (kills, drops, etc.).
    pub expr: usize,
}

/// Point-level variable liveness: `live_at[block][stmt]` holds the
/// variables live at that statement point; `uses` holds the per-point
/// variable uses (exposed for the Polonius fact extractor).
#[derive(Clone, Debug)]
pub struct PointLiveness {
    uses: Vec<Vec<HashSet<Symbol>>>,
    /// Sparse-interval compression of the point liveness (rustc's
    /// `SparseIntervalMatrix` concept): per block, per variable, the
    /// contiguous runs of live statement points.  A variable is typically
    /// live over a range (from its definition to its last use), so this
    /// compresses the dense per-point sets.  `is_live_at` queries this.
    intervals: Vec<HashMap<Symbol, Vec<(usize, usize)>>>,
}

impl PointLiveness {
    /// Whether `var` is live at `point` (a use of `var` is reachable from
    /// this statement — the "live until last use" notion, point-precise).
    pub fn is_live_at(&self, var: Symbol, point: Point) -> bool {
        self.intervals
            .get(point.block.0)
            .and_then(|blk| blk.get(&var))
            .is_some_and(|runs| {
                runs.iter()
                    .any(|&(s, e)| point.stmt >= s && point.stmt <= e)
            })
    }

    /// The per-point variable USES (`uses[block][stmt]` = the variables
    /// used at that statement; the last index per block is the terminator).
    pub fn var_uses(&self) -> &Vec<Vec<HashSet<Symbol>>> {
        &self.uses
    }

    /// The per-variable live intervals (per block, the contiguous runs of
    /// live statement points) — the sparse-interval liveness, exposed for
    /// the anya troubleshooting output.
    pub(crate) fn live_intervals(&self) -> &Vec<HashMap<Symbol, Vec<(usize, usize)>>> {
        &self.intervals
    }
}

impl VarLiveness {
    /// Whether `var` is live at the entry of `block` (a use of `var` is
    /// reachable from `block` — the "live until last use" notion).
    pub fn is_live_at(&self, var: Symbol, block: BlockId) -> bool {
        self.live_in.get(block.0).is_some_and(|s| s.contains(&var))
    }
}

/// Collect the variables USED (read) in a statement.
/// The static move check (the CFG-level affine use-after-move).
/// String-literal-bound variables are non-Copy (the `String`); the RHS
/// `Ident` of a non-Copy source MOVES it, and a later use of a moved
/// variable is a use-after-move error.  The CFG-level version runs a
/// forward dataflow over the function's basic blocks (each block's
/// moved-set merges its predecessors' — path-sensitive across branches
/// and loops).
pub(crate) fn check_function_moves<'input>(
    body: &[HirStmt<'input>],
    finally: &[HirStmt<'input>],
    non_copy_roots: &[Symbol],
) -> Vec<String> {
    // The CFG-level path-sensitive analysis (the flow-sensitive move
    // check): build the function's
    // CFG and run a forward dataflow over basic blocks, merging (union)
    // predecessor moved-sets at joins.  This subsumes the old straight-
    // line pass (which only handled a single statement list).
    // The `finally` block participates in the move check (SYNTAX.md
    // §finally — it runs on every function-exit edge, so a moved value
    // used inside it is a use-after-move).
    let cfg = CfgGraph::build_function(body, finally);
    check_cfg_moves(&cfg, non_copy_roots)
}

/// CFG-level forward move analysis: each basic block carries a moved-set
/// that is the UNION of its predecessors' moved-sets (conservative meet),
/// then the block's statements are checked against it (path-sensitive —
/// a value moved on one path is possibly-moved after the join; a branch
/// whose arms isolate moved-sets gets the precise per-arm treatment from
/// `check_stmt_moves`).
pub(crate) fn check_cfg_moves(cfg: &CfgGraph, non_copy_roots: &[Symbol]) -> Vec<String> {
    let n = cfg.blocks.len();
    let mut block_moved: Vec<HashSet<FrozenPlace>> = vec![HashSet::new(); n];
    // First pass: forward-propagate the "is non-Copy" predicate so each
    // block's entry snapshot contains the non-Copy variables introduced in
    // ANY reaching block — NOT whatever the fixpoint order happened to have
    // processed so far.  (The previous global `&mut non_copy` let a later
    // block's introduction leak into earlier predecessors — an over-
    // approximation that flagged false use-after-moves on those paths.)
    let mut block_non_copy: Vec<HashSet<Symbol>> = vec![HashSet::new(); n];
    block_non_copy[0].extend(non_copy_roots.iter().copied());
    loop {
        let mut changed = false;
        for b in 0..n {
            let mut entry: HashSet<Symbol> = HashSet::new();
            for &p in cfg.predecessors(BlockId(b)) {
                entry.extend(block_non_copy[p.0].iter().cloned());
            }
            let mut out = entry.clone();
            propagate_block_non_copy(&cfg.blocks[b].stmts, &mut out);
            if out != block_non_copy[b] {
                block_non_copy[b] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Second pass: the moved-set fixpoint.  Each block uses its OWN entry
    // non-Copy snapshot (block-local growth never leaks into predecessors —
    // the propagation was already settled by the first pass).
    // Iterate to a fixpoint (loop back-edges can grow the moved-sets).
    // the errors are NOT collected inside the loop — collecting
    // them there re-issued each use-after-move once per iteration
    // (duplicate E114s on loop back-edges).  The final pass below
    // issues each error exactly once.
    loop {
        let mut changed = false;
        for b in 0..n {
            let mut in_set: HashSet<FrozenPlace> = HashSet::new();
            for &p in cfg.predecessors(BlockId(b)) {
                in_set.extend(block_moved[p.0].iter().cloned());
            }
            let mut out = in_set.clone();
            check_block_moves(
                &cfg.blocks[b].stmts,
                &mut out,
                &mut block_non_copy[b].clone(),
                &mut Vec::new(),
            );
            if out != block_moved[b] {
                block_moved[b] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The FINAL single pass — each error is issued exactly once.
    let mut errs = Vec::new();
    for b in 0..n {
        let mut in_set: HashSet<FrozenPlace> = HashSet::new();
        for &p in cfg.predecessors(BlockId(b)) {
            in_set.extend(block_moved[p.0].iter().cloned());
        }
        let mut out = in_set.clone();
        check_block_moves(
            &cfg.blocks[b].stmts,
            &mut out,
            &mut block_non_copy[b].clone(),
            &mut errs,
        );
    }
    errs
}

/// The recursive block walk: the moved-state's propagation through the
/// nested statements.
/// Forward-propagate the "is non-Copy" predicate within a block: a
/// VariableDef whose source is non-Copy (a known non-Copy variable or an
/// explicit `move`) introduces a non-Copy binding.  Extracted from the
/// moved-set pass so the fixpoint iteration does not conflate the global
/// non-Copy set with the per-block entry snapshot.
fn propagate_block_non_copy<'input>(stmts: &[HirStmt<'input>], non_copy: &mut HashSet<Symbol>) {
    for stmt in stmts {
        if let HirStmt::VariableDef {
            name: Some(n),
            value: Some(v),
            ..
        } = stmt
        {
            let src_is_non_copy = matches!(
                v.as_ref(),
                HirExpr::Ident(src, _, _) if non_copy.contains(src)
            ) || matches!(v.as_ref(), HirExpr::Move(..));
            if src_is_non_copy {
                non_copy.insert(*n);
            }
        }
    }
}

fn check_block_moves<'input>(
    stmts: &[HirStmt<'input>],
    moved: &mut HashSet<FrozenPlace>,
    non_copy: &mut HashSet<Symbol>,
    errs: &mut Vec<String>,
) {
    for stmt in stmts {
        check_stmt_moves(stmt, moved, non_copy, errs);
    }
}

/// The RHS consumption of a non-Copy value (shared by VariableDef /
/// Assign / Return / expression statements): an Ident of a non-Copy
/// source moves it, an explicit `move place` moves the place, and any
/// other expression is scanned for consumed places.  Previously only
/// the VariableDef arm recorded consumption — Assign/Return/expression
/// statements let a moved value be used again (a double move).
fn mark_value_consumed<'input>(
    v: &HirExpr<'input>,
    non_copy: &HashSet<Symbol>,
    moved: &mut HashSet<FrozenPlace>,
) {
    match v {
        HirExpr::Ident(src, _, _) => {
            if non_copy.contains(src) {
                moved.insert(FrozenPlace::Root(*src));
            }
        }
        HirExpr::Move(inner, _, _) => {
            if let Some(p) = hir_expr_place(inner) {
                moved.insert(p);
            }
        }
        _ => mark_consumed_places(v, non_copy, moved),
    }
}

/// Format a `FrozenPlace` into a human-readable string for error messages.
fn format_frozen_place(place: &FrozenPlace, out: &mut String) {
    match place {
        FrozenPlace::Root(v) => {
            out.push_str(&v.as_str());
        }
        FrozenPlace::Field(base, field) => {
            format_frozen_place(base, out);
            out.push('.');
            out.push_str(&field.as_str());
        }
        FrozenPlace::Index(base) => {
            format_frozen_place(base, out);
            out.push_str("[..]");
        }
        FrozenPlace::ConstIndex(base, idx) => {
            format_frozen_place(base, out);
            out.push('[');
            out.push_str(&idx.to_string());
            out.push(']');
        }
        FrozenPlace::Deref(base) => {
            out.push('*');
            format_frozen_place(base, out);
        }
    }
}

fn check_stmt_moves<'input>(
    stmt: &HirStmt<'input>,
    moved: &mut HashSet<FrozenPlace>,
    non_copy: &mut HashSet<Symbol>,
    errs: &mut Vec<String>,
) {
    // The statement's reads — a use of a moved place is the error.
    // Collect precise read places (FrozenPlace) rather than bare symbols,
    // so partial move checking can distinguish arr[0] (hollow) from arr[1]
    // (available).
    let read_places: Vec<FrozenPlace> = match stmt {
        HirStmt::Assign {
            value, target, op, ..
        } => {
            let mut out = Vec::new();
            collect_read_places_into(value, &mut out);
            if op.is_some() {
                // Compound assignment reads the whole target's old value.
                collect_read_places_into(target, &mut out);
            } else {
                // Plain write: only reads the target's index/base (not the
                // target itself — the written place is not read).
                collect_read_places_in_write_target(target, &mut out);
            }
            out
        }
        _ => {
            let mut out = Vec::new();
            collect_read_places_in_stmt(stmt, &mut out);
            out
        }
    };
    for read_place in &read_places {
        // Bidirectional prefix check:
        // - moved place is prefix of read → read touches moved sub-place
        // - read is prefix of moved → read touches whole containing a moved part
        // For ConstIndex: arr[0] and arr[1] are NOT prefixes of each other,
        // so moving arr[0] does not block reading arr[1].
        if moved
            .iter()
            .any(|mp| place_is_prefix_of(mp, read_place) || place_is_prefix_of(read_place, mp))
        {
            let mut place_str = String::new();
            format_frozen_place(read_place, &mut place_str);
            errs.push(format!("use of moved value: `{}`", place_str));
        }
    }
    match stmt {
        HirStmt::VariableDef {
            name: Some(n),
            value: Some(v),
            ..
        } => {
            let src_is_non_copy = matches!(
                v.as_ref(),
                HirExpr::Ident(src, _, _) if non_copy.contains(src)
            ) || matches!(v.as_ref(), HirExpr::Move(..));
            mark_value_consumed(v.as_ref(), non_copy, moved);
            if src_is_non_copy {
                non_copy.insert(*n);
            }
            // Re-initialization: clear ALL moved marks for this variable
            // (including sub-places like ConstIndex, Field, etc.).
            moved.retain(|mp| place_root_symbol(mp) != Some(*n));
        }
        // The branch path propagation — the branches' moved states
        // merge (the union) at the join: a variable moved on ANY branch
        // is possibly-moved after the join.  This is CONSERVATIVE (the
        // Control-flow statements (`If`/`While`/`Loop`/`For`) NEVER appear
        // in a `BasicBlock`'s `stmts` — `CfgBuilder::build_seq` flattens
        // them into separate blocks with `Terminator::Branch` / loop
        // edges.  (Dead arms for them were removed; the move analysis is
        // path-sensitive over the CFG via `check_cfg_moves`.)
        // A MATCH expression: each arm is a separate path — a value
        // moved in ONE arm is only possibly-moved after the match
        // (path-sensitive; the move check: the previous folding
        // conflated arm paths, causing false positives on re-borrows
        // inside arms).
        HirStmt::Expression(e) if matches!(e.as_ref(), HirExpr::Match { .. }) => {
            let HirExpr::Match {
                scrutinee, arms, ..
            } = e.as_ref()
            else {
                unreachable!()
            };
            // The scrutinee CONSUMES before any arm runs (`match move x
            // { ... }`) — it was previously ignored (use-after-move false
            // negative).
            let mut pre = moved.clone();
            mark_value_consumed(scrutinee, non_copy, &mut pre);
            let mut merged: HashSet<FrozenPlace> = HashSet::new();
            for arm in arms {
                let mut am = pre.clone();
                let mut nc_arm = non_copy.clone();
                let tmp_stmt = HirStmt::Expression(arm.body.clone());
                let tmp_arr = [tmp_stmt];
                check_block_moves(&tmp_arr, &mut am, &mut nc_arm, errs);
                merged.extend(am);
            }
            *moved = merged;
        }
        // The re-initialization — an assignment to a moved variable
        // re-initializes it. Use precise place-based removal:
        // - arr[0] = value → removes ConstIndex(Root(arr), 0) exactly
        // - arr = [1,2,3]  → removes Root(arr) + all sub-places (whole overwrite)
        // - *r = value     → removes Deref(Root(r)) exactly
        // - p.f = value    → removes Field(Root(p), f) exactly
        HirStmt::Assign { target, value, .. } => {
            if let Some(target_place) = hir_expr_place(target) {
                moved.remove(&target_place);
                // If target is a Root (whole-variable assignment), clear all
                // sub-place hollow marks for that variable.
                if matches!(target_place, FrozenPlace::Root(_)) {
                    moved
                        .retain(|mp| !place_is_prefix_of(&target_place, mp) && *mp != target_place);
                }
            }
            // The RHS of an assignment CONSUMES the value — `b = a`
            // moves `a` (previously unrecorded → double moves passed).
            mark_value_consumed(value, non_copy, moved);
        }
        // A `return` / bare expression statement also CONSUMES its
        // value (previously `_ => {}` — double moves passed).
        HirStmt::Return { value: Some(v), .. } => {
            mark_value_consumed(v, non_copy, moved);
        }
        HirStmt::Expression(e) if !matches!(e.as_ref(), HirExpr::Match { .. }) => {
            mark_value_consumed(e, non_copy, moved);
        }
        _ => {}
    }
}

fn used_vars_in_stmt<'input>(stmt: &HirStmt<'input>) -> Vec<Symbol> {
    let mut out = Vec::new();
    match stmt {
        HirStmt::VariableDef { value, .. } => {
            if let Some(v) = value {
                used_vars_in_expr_into(v, &mut out);
            }
        }
        HirStmt::Assign { target, value, .. } => {
            used_vars_in_expr_into(target, &mut out);
            used_vars_in_expr_into(value, &mut out);
        }
        HirStmt::Return { value, .. } => {
            if let Some(v) = value {
                used_vars_in_expr_into(v, &mut out);
            }
        }
        // Expression statements (bare `expr;` — including the `leave with
        // e;` error exit) USE their expression's variables.
        HirStmt::Expression(expr) => used_vars_in_expr_into(expr, &mut out),
        HirStmt::If { .. } | HirStmt::IfLet { .. } => {
            let (cond, then_branch, else_branch) = match stmt {
                HirStmt::If {
                    cond,
                    then_branch,
                    else_branch,
                    ..
                } => (cond, then_branch, else_branch),
                HirStmt::IfLet {
                    scrutinee,
                    then_branch,
                    else_branch,
                    ..
                } => (scrutinee, then_branch, else_branch),
                _ => unreachable!(),
            };
            used_vars_in_expr_into(cond, &mut out);
            for s in then_branch {
                out.extend(used_vars_in_stmt(s));
            }
            if let Some(e) = else_branch {
                for s in e {
                    out.extend(used_vars_in_stmt(s));
                }
            }
        }
        HirStmt::While { .. } | HirStmt::WhileLet { .. } => {
            let (cond, body) = match stmt {
                HirStmt::While { cond, body, .. } => (cond, body),
                HirStmt::WhileLet {
                    scrutinee, body, ..
                } => (scrutinee, body),
                _ => unreachable!(),
            };
            used_vars_in_expr_into(cond, &mut out);
            for s in body {
                out.extend(used_vars_in_stmt(s));
            }
        }
        HirStmt::For { iterable, body, .. } => {
            // The iterable's variables are USED (`for x in a` reads `a` —
            // the iterable was previously discarded).
            used_vars_in_expr_into(iterable, &mut out);
            for s in body {
                out.extend(used_vars_in_stmt(s));
            }
        }
        HirStmt::Loop { body, .. } => {
            for s in body {
                out.extend(used_vars_in_stmt(s));
            }
        }
        _ => {}
    }
    out
}

/// Collect the variables used (read) in an expression.
fn used_vars_in_expr<'input>(expr: &HirExpr<'input>) -> Vec<Symbol> {
    let mut out = Vec::new();
    used_vars_in_expr_into(expr, &mut out);
    out
}

fn used_vars_in_expr_into<'input>(expr: &HirExpr<'input>, out: &mut Vec<Symbol>) {
    match expr {
        HirExpr::Ident(name, _, _) => out.push(*name),
        // Closures/tasks — their bodies' variables participate
        // in the liveness analysis.
        HirExpr::Closure { body, .. } => {
            for stmt in body {
                out.extend(used_vars_in_stmt(stmt));
            }
        }
        HirExpr::Task { block, .. } => {
            for stmt in block {
                out.extend(used_vars_in_stmt(stmt));
            }
        }
        HirExpr::BinaryOp { left, right, .. } => {
            used_vars_in_expr_into(left, out);
            used_vars_in_expr_into(right, out);
        }
        HirExpr::UnaryOp { expr, .. } => used_vars_in_expr_into(expr, out),
        HirExpr::FieldAccess { base, .. } => used_vars_in_expr_into(base, out),
        HirExpr::Index { base, index, .. } => {
            used_vars_in_expr_into(base, out);
            used_vars_in_expr_into(index, out);
        }
        HirExpr::Call { callee, args, .. } => {
            used_vars_in_expr_into(callee, out);
            for a in args {
                used_vars_in_expr_into(a, out);
            }
        }
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            used_vars_in_expr_into(scrutinee, out);
            // The match arms' guards and bodies USE their variables
            // — the arms were previously discarded by `..`).
            for a in arms {
                if let Some(g) = &a.guard {
                    used_vars_in_expr_into(g, out);
                }
                used_vars_in_expr_into(&a.body, out);
            }
        }
        // The EXPRESSION-form if/if-let branches USE their variables
        //   — they were previously invisible to the liveness).
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            used_vars_in_expr_into(cond, out);
            for s in then_branch {
                out.extend(used_vars_in_stmt(s));
            }
            if let Some(e) = else_branch {
                for s in e {
                    out.extend(used_vars_in_stmt(s));
                }
            }
        }
        HirExpr::IfLet {
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            used_vars_in_expr_into(scrutinee, out);
            for s in then_branch {
                out.extend(used_vars_in_stmt(s));
            }
            if let Some(e) = else_branch {
                for s in e {
                    out.extend(used_vars_in_stmt(s));
                }
            }
        }
        // The error-exit value (`leave with e`) is a USE of its expression
        // (the value must stay visible to the liveness).
        HirExpr::LeaveWith { expr, .. } => used_vars_in_expr_into(expr, out),
        // The `catch` branches (the error handlers) USE their bodies'
        // variables — the branch bodies are part of the function body.
        HirExpr::Catch { expr, branches, .. } => {
            used_vars_in_expr_into(expr, out);
            for b in branches {
                for s in &b.body {
                    out.extend(used_vars_in_stmt(s));
                }
            }
        }
        HirExpr::Block(stmts, _, _) => {
            for s in stmts {
                out.extend(used_vars_in_stmt(s));
            }
        }
        // Aggregates and other expression forms: the liveness must see
        // uses INSIDE tuple/struct/array/cast/range literals — a borrow's
        // last use hidden in an aggregate would otherwise be invisible,
        // killing the loan at issuance (a false negative on the freeze).
        HirExpr::Tuple(elems, _, _) | HirExpr::Array(elems, _, _) => {
            for el in elems {
                used_vars_in_expr_into(el, out);
            }
        }
        HirExpr::StructLit { fields, .. } => {
            for (_, val) in fields {
                used_vars_in_expr_into(val, out);
            }
        }
        HirExpr::EnumLit { payload, .. } => {
            if let Some(p) = payload {
                used_vars_in_expr_into(p, out);
            }
        }
        HirExpr::Move(inner, _, _) => used_vars_in_expr_into(inner, out),
        HirExpr::Cast { expr, .. }
        | HirExpr::TypeAnnotated { expr, .. }
        | HirExpr::Try { expr, .. }
        | HirExpr::Await { expr, .. }
        | HirExpr::Old { expr, .. }
        | HirExpr::PolyBox { expr, .. }
        | HirExpr::PolyUnbox { expr, .. }
        | HirExpr::Return { value: expr, .. } => used_vars_in_expr_into(expr, out),
        HirExpr::Range { start, end, .. } => {
            if let Some(s) = start {
                used_vars_in_expr_into(s, out);
            }
            if let Some(e) = end {
                used_vars_in_expr_into(e, out);
            }
        }
        HirExpr::AttrAccess { base, .. } => used_vars_in_expr_into(base, out),
        HirExpr::Quantified { range, body, .. } => {
            used_vars_in_expr_into(range, out);
            used_vars_in_expr_into(body, out);
        }
        HirExpr::UnsafeBlock { body, .. } => {
            for s in body {
                out.extend(used_vars_in_stmt(s));
            }
        }
        _ => {}
    }
}

/// The USES inside a plain ASSIGNMENT TARGET (`*r = 7`, `arr[i] = x`,
/// `p.f = x`): the DEREF/INDEX/FIELD BASES must be read to reach the
/// write location (a loan on `r` stays live through its deref-write),
/// but the final leaf `Ident` — the written place itself — is NOT read
/// (`a = 5` must not keep a loan on `a` alive).  Mirrors the
/// `mark_consumed_places` write-target discipline for the liveness side.
fn used_vars_in_write_target<'input>(expr: &HirExpr<'input>, out: &mut Vec<Symbol>) {
    match expr {
        HirExpr::Ident(..) => {}
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr: inner,
            ..
        } => used_vars_in_expr_into(inner, out),
        HirExpr::Index { base, index, .. } => {
            used_vars_in_expr_into(base, out);
            used_vars_in_expr_into(index, out);
        }
        HirExpr::FieldAccess { base, .. } => used_vars_in_expr_into(base, out),
        _ => used_vars_in_expr_into(expr, out),
    }
}

/// Collect all precise read places (`FrozenPlace`) from an expression.
/// Top-level places (e.g. `arr[0]`) are collected as their full
/// `ConstIndex`/`Index` form, not as `Root(arr)` — enabling the partial
/// move check to distinguish `arr[0]` from `arr[1]`.
fn collect_read_places_into<'input>(expr: &HirExpr<'input>, out: &mut Vec<FrozenPlace>) {
    collect_read_places_into_inner(expr, false, out);
}

fn collect_read_places_into_inner<'input>(
    expr: &HirExpr<'input>,
    in_place_base: bool,
    out: &mut Vec<FrozenPlace>,
) {
    match expr {
        HirExpr::Ident(name, _, _) => {
            if !in_place_base {
                out.push(FrozenPlace::Root(*name));
            }
        }
        HirExpr::FieldAccess { base, .. } => {
            if !in_place_base {
                if let Some(p) = hir_expr_place(expr) {
                    out.push(p);
                }
            }
            collect_read_places_into_inner(base, true, out);
        }
        HirExpr::Index { base, index, .. } => {
            if !in_place_base {
                if let Some(p) = hir_expr_place(expr) {
                    out.push(p);
                }
            }
            collect_read_places_into_inner(base, true, out);
            collect_read_places_into_inner(index, false, out);
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr: inner,
            ..
        } => {
            if !in_place_base {
                if let Some(p) = hir_expr_place(expr) {
                    out.push(p);
                }
            }
            collect_read_places_into_inner(inner, true, out);
        }
        HirExpr::UnaryOp {
            op: UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref,
            expr: inner,
            ..
        } => {
            collect_read_places_into_inner(inner, false, out);
        }
        HirExpr::UnaryOp { expr: inner, .. } => {
            collect_read_places_into_inner(inner, false, out);
        }
        HirExpr::BinaryOp { left, right, .. } => {
            collect_read_places_into_inner(left, false, out);
            collect_read_places_into_inner(right, false, out);
        }
        HirExpr::Move(inner, _, _) => {
            collect_read_places_into_inner(inner, in_place_base, out);
        }
        HirExpr::Call { callee, args, .. } => {
            collect_read_places_into_inner(callee, false, out);
            for a in args {
                collect_read_places_into_inner(a, false, out);
            }
        }
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            collect_read_places_into_inner(scrutinee, false, out);
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_read_places_into_inner(g, false, out);
                }
                collect_read_places_in_stmt(&HirStmt::Expression(a.body.clone()), out);
            }
        }
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        }
        | HirExpr::IfLet {
            scrutinee: cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_read_places_into_inner(cond, false, out);
            for s in then_branch {
                collect_read_places_in_stmt(s, out);
            }
            if let Some(e) = else_branch {
                for s in e {
                    collect_read_places_in_stmt(s, out);
                }
            }
        }
        HirExpr::LeaveWith { expr: inner, .. } => {
            collect_read_places_into_inner(inner, false, out);
        }
        HirExpr::Catch { expr, branches, .. } => {
            collect_read_places_into_inner(expr, false, out);
            for b in branches {
                for s in &b.body {
                    collect_read_places_in_stmt(s, out);
                }
            }
        }
        HirExpr::Block(stmts, _, _) => {
            for s in stmts {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirExpr::Tuple(elems, _, _) | HirExpr::Array(elems, _, _) => {
            for el in elems {
                collect_read_places_into_inner(el, false, out);
            }
        }
        HirExpr::StructLit { fields, .. } => {
            for (_, val) in fields {
                collect_read_places_into_inner(val, false, out);
            }
        }
        HirExpr::EnumLit { payload, .. } => {
            if let Some(p) = payload {
                collect_read_places_into_inner(p, false, out);
            }
        }
        HirExpr::Cast { expr: inner, .. }
        | HirExpr::TypeAnnotated { expr: inner, .. }
        | HirExpr::Try { expr: inner, .. }
        | HirExpr::Await { expr: inner, .. }
        | HirExpr::Old { expr: inner, .. }
        | HirExpr::PolyBox { expr: inner, .. }
        | HirExpr::PolyUnbox { expr: inner, .. }
        | HirExpr::Return { value: inner, .. } => {
            collect_read_places_into_inner(inner, false, out);
        }
        HirExpr::Range { start, end, .. } => {
            if let Some(s) = start {
                collect_read_places_into_inner(s, false, out);
            }
            if let Some(e) = end {
                collect_read_places_into_inner(e, false, out);
            }
        }
        HirExpr::AttrAccess { base, .. } => {
            collect_read_places_into_inner(base, false, out);
        }
        HirExpr::Quantified { range, body, .. } => {
            collect_read_places_into_inner(range, false, out);
            collect_read_places_into_inner(body, false, out);
        }
        HirExpr::UnsafeBlock { body, .. } => {
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirExpr::Closure { body, .. } => {
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirExpr::Task { block, .. } => {
            for s in block {
                collect_read_places_in_stmt(s, out);
            }
        }
        _ => {}
    }
}

/// Statement-level precise read place collection.
fn collect_read_places_in_stmt<'input>(stmt: &HirStmt<'input>, out: &mut Vec<FrozenPlace>) {
    match stmt {
        HirStmt::VariableDef { value: Some(v), .. } => {
            collect_read_places_into(v, out);
        }
        HirStmt::Assign {
            target, value, op, ..
        } => {
            collect_read_places_into(value, out);
            if op.is_some() {
                collect_read_places_into(target, out);
            } else {
                collect_read_places_in_write_target(target, out);
            }
        }
        HirStmt::Return { value: Some(v), .. } => {
            collect_read_places_into(v, out);
        }
        HirStmt::Expression(expr) => {
            collect_read_places_into(expr, out);
        }
        HirStmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        }
        | HirStmt::IfLet {
            scrutinee: cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_read_places_into(cond, out);
            for s in then_branch {
                collect_read_places_in_stmt(s, out);
            }
            if let Some(e) = else_branch {
                for s in e {
                    collect_read_places_in_stmt(s, out);
                }
            }
        }
        HirStmt::While { cond, body, .. }
        | HirStmt::WhileLet {
            scrutinee: cond,
            body,
            ..
        } => {
            collect_read_places_into(cond, out);
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirStmt::For { iterable, body, .. } => {
            collect_read_places_into(iterable, out);
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirStmt::Loop { body, .. } => {
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirStmt::ComptimeBlock { body, .. }
        | HirStmt::Isolate { body, .. }
        | HirStmt::Unsafe { body, .. } => {
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirStmt::ScopeCleanup {
            when_condition,
            body,
            ..
        } => {
            if let Some(cond) = when_condition {
                collect_read_places_into(cond, out);
            }
            for s in body {
                collect_read_places_in_stmt(s, out);
            }
        }
        HirStmt::GhostVariableDef { inner, .. } => {
            collect_read_places_in_stmt(inner, out);
        }
        _ => {}
    }
}

/// Read-place collection for plain assignment targets.
/// Only collects index expressions (not the target itself or its base),
/// matching the semantics of `used_vars_in_write_target`.
fn collect_read_places_in_write_target<'input>(expr: &HirExpr<'input>, out: &mut Vec<FrozenPlace>) {
    match expr {
        HirExpr::Ident(..) => {}
        HirExpr::Index { base, index, .. } => {
            collect_read_places_into(index, out);
            collect_read_places_in_write_target(base, out);
        }
        HirExpr::FieldAccess { base, .. } => {
            collect_read_places_in_write_target(base, out);
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr: inner,
            ..
        } => {
            collect_read_places_in_write_target(inner, out);
        }
        _ => {
            collect_read_places_into(expr, out);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Flow-sensitive borrow-check post-pass (the CFG consumer).
//
// Loans die at their borrow variable's LAST USE (committee ruling
// 2026-08-05): a loan (source place S, borrow variable B) is live at a
// block iff B is live there (a use of B is reachable).  Temporary
// borrows (in-expression, not bound) are live only in their own block.
// This replaces the block-scoped interleaved approximation.
// ────────────────────────────────────────────────────────────────────────

/// A borrow-check error from the post-pass: reading or mutating a place
/// that is frozen by a live loan.
#[derive(Clone, Debug)]
pub struct BorrowError {
    pub place: FrozenPlace,
    pub is_read: bool,
    /// The exclusive-borrow conflict (two live loans on an overlapping
    /// place) — a distinct diagnostic category from the freeze
    /// mutation/read errors (E112 vs E109/E110).
    pub is_exclusive: bool,
    /// The drop-while-borrowed error (E116): a dropped value whose loan
    /// is still live at the drop point — the borrow outlives the value.
    /// Distinct from the freeze/exclusivity categories (the drop is not
    /// a read or mutation event).
    pub is_drop: bool,
    /// The kind of the loan that FREEZES the place (E109/E110): `None`
    /// for the exclusivity errors (two loans, no single kind).  Lets the
    /// diagnostic name the mechanism — `&ro`/`.freeze!()` (ReadOnly)
    /// vs `&mut` (Exclusive) — instead of a generic "active borrow".
    pub loan_kind: Option<LoanKind>,
    /// The FIRST loan's issuance span (the dual-position diagnostics —
    /// the exclusivity errors report BOTH borrow sites); `None` for the
    /// freeze errors.
    pub span1: Option<crate::ast::Span>,
    pub span: crate::ast::Span,
}

/// Shared `BorrowError → Diagnostic` mapping: both
/// borrow-check call-sites (free functions and method bodies) must emit
/// IDENTICAL diagnostics — the three discriminable states (read /
/// exclusive / mutation) with the correct codes (E109 / E112 / E110) and
/// the dual-position label for exclusivity conflicts.  Inlining this at
/// each call-site previously let `check_method_body` drift.
pub(crate) fn borrow_error_diagnostic(err: &BorrowError) -> Diagnostic {
    // The fail-closed sentinel: the borrow check could not be performed
    // (the function exceeded the analysis capacity) — a distinct,
    // explicit diagnostic instead of a generic freeze message.
    if let FrozenPlace::Root(n) = &err.place
        && n.eq_str("<internal>")
    {
        return Diagnostic::error(
            "borrow check could not be performed: the function exceeds the analysis capacity",
        )
        // E113 (registered — "the function's CFG exceeds the
        // point-encoding capacity") is the sentinel's semantics.
        .with_code_str("E113")
        .with_help(
            "the point encoding is bounded at 65536 expressions per statement, \
         1M statements per block, and 1M blocks per function — split an \
         oversized statement (e.g. a very large literal or match) into \
         several statements",
        )
        .with_span(err.span);
    }
    // R9: the placeholder subset error — a region subset derived
    // inside the body but NOT declared in the signature is rejected
    // (E115 — registered; the signature must carry every cross-region
    // relationship).
    if let FrozenPlace::Root(n) = &err.place
        && n.eq_str("<region>")
    {
        return Diagnostic::error("region subset not declared in signature")
            .with_code_str("E115")
            .with_span(err.span);
    }
    // The drop-while-borrowed error (E116 — registered): a dropped value
    // whose loan is still live at the drop point — the borrow outlives
    // the value (rustc E0505).
    if err.is_drop {
        return Diagnostic::error("cannot drop a value while its borrow is still live")
            .with_code_str("E116")
            .with_span(err.span);
    }
    let mut diag = Diagnostic::error(if err.is_read {
        "cannot read a variable frozen by an active `&mut` borrow"
    } else if err.is_exclusive {
        "cannot borrow this place as mutable more than once while an exclusive borrow is live"
    } else {
        // Name the freeze mechanism: `&ro`/`.freeze!()` (ReadOnly) vs
        // `&mut` (Exclusive) — the mutation may be frozen by either.
        match err.loan_kind {
            Some(LoanKind::ReadOnly) => "cannot mutate a variable frozen by an active `&ro` borrow",
            Some(LoanKind::Exclusive) => {
                "cannot mutate a variable frozen by an active `&mut` borrow"
            }
            None => "cannot mutate a variable frozen by an active borrow",
        }
    })
    .with_code_str(if err.is_read {
        "E109"
    } else if err.is_exclusive {
        "E112"
    } else {
        "E110"
    })
    .with_span(err.span);
    // The dual-position diagnostics: the exclusivity errors annotate the
    // FIRST borrow's site as well.
    if let Some(s1) = err.span1 {
        diag = diag.with_label(s1, "the first exclusive borrow is here");
    }
    diag
}

/// Whether the issuance point `lpt` is AT-OR-BEFORE the event point `pt`
/// in the CFG — the loan's birth bound (the loan is live only AFTER its
/// issuance; the backward liveness alone would over-freeze pre-issuance
/// mutations ).
/// Collect the borrow loans and the access events of a function body
/// (shared by the borrow-check post-pass and the Polonius fact extractor).
///
/// NOTE: the HANDWRITTEN birth-bound check (`issued_at_or_before` —
/// forward-reachability + loop-cycle detection + sibling-branch
/// dominance) was REMOVED after the Polonius engine switch: the engine
/// derives the birth bounds from the `cfg_edge` + `loan_issued_at`
/// facts (R4-R6 contains propagation), so the orphaned hand-written
/// logic was dead code and a maintenance hazard.
pub(crate) fn collect_borrow_data<'input>(
    cfg: &CfgGraph<'input>,
    registry: &[(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) -> (
    Vec<(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )>,
    Vec<(FrozenPlace, Point, bool, crate::ast::Span)>,
    Vec<(FrozenPlace, Option<Symbol>, Point)>,
    // The per-statement EXPRESSION counts (block, stmt) → the number of
    // expression points — the polonius cfg_edge extraction builds the
    // intra-statement chains from these (the low 16 bits of a point id
    // are the expression index — see `point_id`).
    HashMap<(usize, usize), usize>,
) {
    let mut loans: Vec<(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )> = Vec::new();
    let mut events: Vec<(FrozenPlace, Point, bool, crate::ast::Span)> = Vec::new();
    // The KILL events (the reborrow-kill + the borrow-variable
    // reassignment): the loans are NOT removed from the vector — a
    // path-insensitive removal would let a sibling branch's assignment
    // kill a loan issued on another path  ).
    let mut kills: Vec<(FrozenPlace, Option<Symbol>, Point)> = Vec::new();
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for (bi, blk) in cfg.blocks().iter().enumerate() {
        let b = BlockId(bi);
        for (si, stmt) in blk.stmts.iter().enumerate() {
            // The expression index counter: a statement's HEAD is
            // expression 0 (the chain landing pad); the value
            // expressions start at 1, in evaluation order (rustc's
            // CFG: callee/receiver first, then the arguments
            // left-to-right; an assignment's target-index reads, then
            // the value, then the synthetic WRITE point).
            let mut next = 1usize;
            collect_stmt(
                stmt,
                Point {
                    block: b,
                    stmt: si,
                    expr: 0,
                },
                &mut next,
                &mut loans,
                &mut events,
                &mut kills,
                registry,
                ctx,
            );
            counts.insert((bi, si), next);
        }
        // The terminator's condition reads (if / while) are reads too.
        if let Some(Terminator::Branch { cond, .. }) = &blk.terminator {
            let mut next = 1usize;
            collect_expr(
                cond,
                Point {
                    block: b,
                    stmt: blk.stmts.len(),
                    expr: 0,
                },
                &mut next,
                false,
                None,
                &mut loans,
                &mut events,
                &mut kills,
                registry,
                ctx,
            );
            counts.insert((bi, blk.stmts.len()), next);
        }
    }
    (loans, events, kills, counts)
}

/// Extract the PLACE from a HIR expression (the storage path being read,
/// borrowed or mutated).
pub(crate) fn hir_expr_place<'input>(e: &HirExpr<'input>) -> Option<FrozenPlace> {
    match e {
        HirExpr::Ident(name, _, _) => Some(FrozenPlace::Root(*name)),
        HirExpr::FieldAccess { base, field, .. } => {
            hir_expr_place(base).map(|p| FrozenPlace::Field(Box::new(p), *field))
        }
        HirExpr::Index { base, index, .. } => {
            // A CONSTANT index (`a[0]`, `a[3]`) is statically known, so the
            // element is an exact place (`a[0]` vs `a[1]` are distinct —
            // freezing one does NOT freeze the other, mirroring rustc's
            // `ProjectionElem::ConstantIndex`).  A DYNAMIC index (`a[i]`)
            // is conservatively treated as touching every element.
            match index.as_ref() {
                HirExpr::Literal(crate::ast::Literal::Int(n), _, _) => {
                    // Attempt to convert the value into a u64 type. If the value is negative or exceeds the range of u64, fall back to dynamic indexing.
                    if let Some(idx) = n.to_u64() {
                        hir_expr_place(base).map(|p| FrozenPlace::ConstIndex(Box::new(p), idx))
                    } else {
                        hir_expr_place(base).map(|p| FrozenPlace::Index(Box::new(p)))
                    }
                }
                _ => hir_expr_place(base).map(|p| FrozenPlace::Index(Box::new(p))),
            }
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr,
            ..
        } => hir_expr_place(expr).map(|p| FrozenPlace::Deref(Box::new(p))),
        // The borrow operators strip the operator — the `&mut a` argument
        // (and `&a` / `&ro r`) refers to the operand's place (the
        // cross-function call-site loan detection needs this).
        HirExpr::UnaryOp {
            op: UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref,
            expr,
            ..
        } => hir_expr_place(expr),
        // `move expr` refers to the same place as `expr` — strip the
        // wrapper so `move arr[0]` yields `ConstIndex(Root(arr), 0)`.
        HirExpr::Move(inner, _, _) => hir_expr_place(inner),
        _ => None,
    }
}

/// Extract the root `Symbol` from a `FrozenPlace`.
/// `Root(a)` → `Some(a)`; `Field`/`Index`/`ConstIndex`/`Deref` recurse to base.
fn place_root_symbol(place: &FrozenPlace) -> Option<Symbol> {
    match place {
        FrozenPlace::Root(v) => Some(*v),
        FrozenPlace::Field(base, _)
        | FrozenPlace::Index(base)
        | FrozenPlace::ConstIndex(base, _)
        | FrozenPlace::Deref(base) => place_root_symbol(base),
    }
}

fn borrow_kind(op: &UnaryOp) -> LoanKind {
    if matches!(op, UnaryOp::RefMut) {
        LoanKind::Exclusive
    } else {
        LoanKind::ReadOnly
    }
}

fn collect_stmt<'input>(
    stmt: &HirStmt<'input>,
    pt: Point,
    next: &mut usize,
    loans: &mut Vec<(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )>,
    events: &mut Vec<(FrozenPlace, Point, bool, crate::ast::Span)>,
    kills: &mut Vec<(FrozenPlace, Option<Symbol>, Point)>,
    registry: &[(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) {
    match stmt {
        HirStmt::VariableDef {
            name,
            value: Some(v),
            ..
        } => {
            // A top-level borrow binding: the loan's borrow variable is
            // the binding name — its last use ends the loan.  Peel a
            // `TypeAnnotated` wrapper (`set r: &mut Int<32> = &mut x`
            // wraps the borrow in an annotation) so the loan still binds
            // to `Some(r)` — otherwise it degrades to a temporary loan
            // with no borrow variable, and the liveness/freeze chain
            // loses the connection (the Polonius ancestor-clobber
            // shape).
            let annotated = match v.as_ref() {
                HirExpr::TypeAnnotated { expr, .. } => expr.as_ref(),
                other => other,
            };
            let vpt = Point { expr: *next, ..pt };
            if let Some(n) = name
                && let HirExpr::UnaryOp { op, expr, span, .. } = annotated
                && matches!(op, UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref)
                && let Some(p) = hir_expr_place(expr)
            {
                // A `&ro r` operand (`r` a REFERENCE VARIABLE — `&ro`
                // requires a reference operand, E111) freezes the
                // referent, not the variable cell: wrap the place in a
                // deref so the referent resolution (polonius.rs) lands
                // the loan on the ultimate object (`a` for `r: &mut a` —
                // the `a = 5` mutation-freeze after `&ro r` parity,
                // rustc E0506).  Without the wrap the loan stays on
                // `Root(r)` and the write to the referent is never seen.
                // `&mut a` / `&a` (a possibly an ORDINARY variable) keep
                // the plain place — wrapping THOSE would turn an
                // ordinary-variable loan into `Deref(Root(a))`, which the
                // AncestorClobber exclusion (a write to the root is then a
                // strict prefix of the deref'd loan) mis-kills instead of
                // invalidating — the E110 regression.
                //
                // The `&ro` wrap applies to ANY operand — a
                // non-Ident operand (`&ro (a.b)`, `&ro (arr[0])`) is
                // still a REFERENCE-TYPED place (fn_ctxt E111 requires
                // the `&ro` operand to be a reference type), so its loan
                // must land on the REFERENT: `&ro X` → `Deref(place(X))`.
                // Without this, `&ro (a.b)` registered a loan on the CELL
                // `Field(a,b)` and a write through the reference
                // (`*(a.b) = 5`) was never seen (rustc E0506).
                let place = match (op, expr.as_ref()) {
                    (UnaryOp::Ro, HirExpr::Ident(name, _, _)) => {
                        FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                    }
                    (UnaryOp::Ro, _) => FrozenPlace::Deref(Box::new(p)),
                    _ => p,
                };
                loans.push((place, Some(*n), borrow_kind(op), pt, *span, false));
            } else if let Some(n) = name
                // `set r = &mut self.x` — the HIR is
                // `FieldAccess(base: UnaryOp(RefMut, self), x)`: the borrow
                // is on the BASE (`&mut self`), the field projects after.
                // Bind the loan to `Some(r)` with the projected place (the
                // Polonius ancestor-clobber shape) — otherwise the loan
                // degrades to a temporary `Root(self)` loan that dies at
                // the next statement, losing the freeze chain.
                && let HirExpr::FieldAccess { base, field, .. } = annotated
                && let HirExpr::UnaryOp { op, expr, span, .. } = base.as_ref()
                && matches!(op, UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref)
                && let Some(p) = hir_expr_place(expr)
            {
                loans.push((
                    FrozenPlace::Field(Box::new(p), *field),
                    Some(*n),
                    borrow_kind(op),
                    pt,
                    *span,
                    false,
                ));
            } else if let Some(n) = name
                // `set r: &mut T = &mut arr[1]` — the HIR is
                // `Index(base: UnaryOp(RefMut, arr), 1)`: the borrow is on
                // the BASE (`&mut arr`), the index applies after.  Bind the
                // loan to `Some(r)` with the indexed place so the liveness/
                // freeze chain keeps the connection (the Polonius
                // ancestor-clobber shape).
                && let HirExpr::Index {
                    base,
                    index,
                    ..
                } = annotated
                && let HirExpr::UnaryOp { op, expr, span, .. } = base.as_ref()
                && matches!(op, UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref)
                && let Some(p) = hir_expr_place(expr)
            {
                loans.push((
                    FrozenPlace::Index(Box::new(p)),
                    Some(*n),
                    borrow_kind(op),
                    vpt,
                    *span,
                    false,
                ));
                // Traverse the index expression — reads/temporary loans there must
                // be observed (the top-level borrow operand itself is not re-walked).
                collect_expr(
                    index,
                    pt,
                    next,
                    false,
                    Some(*n),
                    loans,
                    events,
                    kills,
                    registry,
                    ctx,
                );
            } else if let Some(n) = name
                && let HirExpr::Call {
                    callee,
                    args,
                    // The comptime calls (`compute_size!()`) are
                    // compile-time — they never produce a runtime loan,
                    // so the cross-function A(ρ) freeze does not apply.
                    comptime: false,
                    ..
                } = v.as_ref()
                && let Some((_is_method, receiver, positions, sig)) =
                    resolve_call(callee, registry, ctx)
            {
                // The METHOD-CALL RECEIVER borrow (`v.push(...)` — the
                // implicit `&mut v` / `&v`, rustc's receiver autoref):
                // the mutability comes from the method signature's `self`
                // parameter (`&mut self` → Exclusive, `&self` →
                // ReadOnly).  The loan makes the receiver freeze while
                // the method runs.  Issued at the CALLEE's expression
                // point (the `obj.put` FieldAccess node): the same-point
                // callee read is the TPB reservation (exempt), and the
                // argument reads at LATER points of the same statement
                // are exempt too — the two-phase exemption is ordered by
                // the expression index.
                if let Some(rcv) = receiver {
                    if let Some(p) = hir_expr_place(rcv) {
                        // `self` is the FIRST input borrow of the method
                        // signature (`&mut self` → Exclusive, `&self` →
                        // ReadOnly) — its mutability is
                        // `input_borrow_mutable[0]`.
                        let self_mut = sig.input_borrow_mutable.first().copied().unwrap_or(false);
                        let kind = if self_mut {
                            LoanKind::Exclusive
                        } else {
                            LoanKind::ReadOnly
                        };
                        // The receiver autoref is a TEMPORARY in rustc
                        // (NLL): it dies at the end of the call's
                        // statement, regardless of later uses of the
                        // receiver variable — a later `obj.v = 2` is
                        // legal.  Binding it to the receiver variable
                        // would keep it live until the variable's last
                        // use (the `b.put(b.get()); b.v = 2;` E110
                        // over-rejection — rustc accepts).
                        loans.push((p, None, kind, vpt, callee.span(), false));
                    }
                }
                // The cross-function returned borrow — the callee's
                // output derives from an input borrow (the A(ρ) — the
                // known_placeholder_subset); the binding (the returned
                // borrow) freezes the input's source.
                // The `out_origin == universal_region.len()` filter was
                // ALWAYS FALSE — the output origins are the LAST entries of
                // `universal_region`, so `len()` is one past the last
                // `out_origin` (the block never fired: the returned-borrow
                // loan was never bound to the binding variable, degrading
                // to a temporary loan with no liveness connection).
                // Iterate the subset directly — it already encodes the
                // (input, output) origin pairs.
                // The returned-borrow-covered argument positions: their top-level
                // borrow IS the returned-borrow loan (registered at the
                // argument's expression point below) — a second,
                // temporary loan for the same argument would
                // SELF-OVERLAP at the issuance point (the E112 exclusivity
                // check fires on two same-place loans — `set r =
                // get(&mut a); let x = *r; a = 5;` spuriously rejected the
                // trailing mutation).  Their sub-expressions (index
                // expressions, deref operands) are still reads and are
                // traversed below.
                let mut pb_loans: HashMap<usize, (FrozenPlace, LoanKind)> = HashMap::new();
                for &(input_origin, _out_origin) in &sig.known_placeholder_subset {
                    if let Some(&pos) = positions.get(input_origin as usize)
                        && let Some(arg) = args.get(pos)
                        && let Some(place) = hir_expr_place(arg)
                    {
                        // The input borrow's MUTABILITY decides the loan
                        // kind: `&mut` inputs freeze the source exclusively;
                        // read-only inputs (`&T`/`&ro`) only get a
                        // `ReadOnly` loan (previously every
                        // return-borrow was `Exclusive`, over-freezing
                        // read-only functions).
                        let kind = if sig
                            .input_borrow_mutable
                            .get(input_origin as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            LoanKind::Exclusive
                        } else {
                            LoanKind::ReadOnly
                        };
                        // A BARE-REFERENCE argument (`set r3 = get2(r)` —
                        // the callee's output derives from the input's
                        // REFERENT, reborrowed inside the callee, not from
                        // the reference CELL): wrap the place in a deref so
                        // the referent resolution (polonius.rs) lands the
                        // loan on the ultimate object (`a` for `r: &mut a`)
                        // — the `a = 5` mutation-freeze parity (rustc
                        // E0506).  Without the wrap the returned-borrow loan stays on
                        // `Root(r)` and the write to the referent is never
                        // seen (the cross-function reborrow hole).
                        let place = match arg {
                            HirExpr::Ident(name, _, _) => {
                                FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                            }
                            // An explicit `&ro r` ARGUMENT (`set s =
                            // takes_shared(&ro r)` — the `&ro` itself is
                            // the borrow, `r` a reference variable):
                            // same referent-resolution wrap, so the
                            // temporary returned-borrow loan lands on `a` and the
                            // `a = 5` mutation-freeze is seen (E0506) —
                            // mirroring the VariableDef/expression-level
                            // borrow wraps above.
                            HirExpr::UnaryOp {
                                op: crate::ast::UnaryOp::Ro,
                                expr,
                                ..
                            } => match expr.as_ref() {
                                HirExpr::Ident(name, _, _) => {
                                    FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                                }
                                // A non-Ident `&ro` ARGUMENT operand
                                // (still reference-typed per E111) must
                                // also land on the referent.
                                _ => FrozenPlace::Deref(Box::new(place)),
                            },
                            _ => place,
                        };
                        pb_loans.insert(pos, (place, kind));
                    }
                }
                // Traverse the call's sub-expressions (callee + args) so
                // reads and temporary loans in arguments are observed —
                // the top-level borrow registration above is not a
                // substitute for sub-expression coverage.  A returned-borrow-covered
                // argument is NOT re-registered as a temporary loan (the
                // same underlying borrow); only its operand's
                // sub-expressions (reads) are traversed.
                collect_expr(
                    callee, pt, next, false, None, loans, events, kills, registry, ctx,
                );
                for (idx, arg) in args.iter().enumerate() {
                    if let Some((place, kind)) = pb_loans.get(&idx) {
                        // The returned-borrow loan is issued at the ARGUMENT's
                        // expression point (its root index, consumed
                        // here); the operand's sub-expressions are still
                        // walked (reads).
                        loans.push((
                            place.clone(),
                            Some(*n),
                            *kind,
                            Point { expr: *next, ..pt },
                            // Use the ARGUMENT's span — the
                            // whole-call `v.span()` gave every returned-borrow loan
                            // in one call the SAME span, so the E112
                            // same-span exemption (polonius.rs:1804)
                            // skipped the exclusivity conflict between
                            // two DISTINCT `&mut a` arguments
                            // (`set r = f(&mut a, &mut a)` — rustc
                            // E0499).  Mirrors the expression arm below.
                            arg.span(),
                            true,
                        ));
                        *next += 1;
                        if let HirExpr::UnaryOp {
                            op: UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref,
                            expr,
                            ..
                        } = arg
                        {
                            collect_target_reads(
                                expr, pt, next, None, loans, events, kills, registry, ctx,
                            );
                        } else {
                            collect_expr(
                                arg, pt, next, false, None, loans, events, kills, registry, ctx,
                            );
                        }
                    } else {
                        collect_expr(
                            arg, pt, next, false, None, loans, events, kills, registry, ctx,
                        );
                    }
                }
            } else if let Some(n) = name {
                collect_expr(
                    v,
                    pt,
                    next,
                    false,
                    Some(*n),
                    loans,
                    events,
                    kills,
                    registry,
                    ctx,
                );
            } else {
                collect_expr(
                    v, pt, next, false, None, loans, events, kills, registry, ctx,
                );
            }
        }
        HirStmt::Assign {
            target,
            value,
            span,
            ..
        } => {
            // The target's INDEX expressions are reads (`arr[idx] = 5`
            // reads `idx`) — reading a frozen index must be rejected.
            // They evaluate BEFORE the value (rustc's order), so the
            // target reads come first.
            collect_target_reads(target, pt, next, None, loans, events, kills, registry, ctx);
            // If the value is a NEW borrow — bind it to the target.
            if let HirExpr::UnaryOp {
                op,
                expr,
                span: vspan,
                ..
            } = value.as_ref()
                && matches!(op, UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref)
                && let Some(place) = hir_expr_place(expr)
                && let Some(name) = match hir_expr_place(target) {
                    Some(FrozenPlace::Root(n)) => Some(n),
                    _ => None,
                }
            {
                loans.push((
                    place,
                    Some(name),
                    borrow_kind(op),
                    Point { expr: *next, ..pt },
                    *vspan,
                    false,
                ));
            }
            collect_expr(
                value, pt, next, false, None, loans, events, kills, registry, ctx,
            );
            // The WRITE event + the kills at the statement's FINAL
            // expression point (the synthetic write node — after the
            // value's evaluation): the expression-level ordering makes a
            // same-statement write-then-reborrow decidable.
            let wpt = Point { expr: *next, ..pt };
            *next += 1;
            // Mutation event on the target.
            if let Some(p) = hir_expr_place(target) {
                events.push((p.clone(), wpt, false, *span));
                // The reborrow-kill : an assignment to a STRICT PREFIX
                // of a loan's place KILLS the loan (e.g. `ls = tl`
                // reassigns `ls`, killing the `&mut *ls` loan — the
                // reborrow loop; the liveness alone cannot distinguish
                // the loop iterations).  The reborrow-kill recorded as a
                // KILL EVENT — the loans are NOT removed (the
                // path-insensitive removal would let a sibling branch's
                // assignment kill a loan issued on another path).
                kills.push((p, None, wpt));
            }
            // (the general kill — the Minor fix): ANY reassignment
            // of a borrow variable kills its old loans — regardless of the
            // value's kind (`r = r2` overwrites the borrow; the old source
            // is no longer borrowed through it — the old loan would
            // otherwise stay live until the variable's last use).
            let target_name = match hir_expr_place(target) {
                Some(FrozenPlace::Root(n)) => Some(n),
                _ => None,
            };
            if let Some(name) = target_name {
                // The borrow-variable reassignment recorded as a KILL
                // EVENT (the same path-sensitivity as the reborrow-kill).
                kills.push((FrozenPlace::Root(name), Some(name), wpt));
            }
        }
        HirStmt::Return { value: Some(v), .. } => {
            collect_expr(
                v, pt, next, false, None, loans, events, kills, registry, ctx,
            );
        }
        // Expression statements (bare `expr;` — including `leave with e;`)
        // READ their expression — the borrow-check must see the value's
        // reads/loans.
        HirStmt::Expression(expr) => collect_expr(
            expr, pt, next, false, None, loans, events, kills, registry, ctx,
        ),
        // Nested statement containers: recurse into their bodies so that
        // loans/events INSIDE comptime/isolate/unsafe blocks and nested
        // statement lists are collected.  The checker-global `frozen_vars`
        // previously covered these via check_stmt; the flow-sensitive
        // post-pass must preserve that coverage (a `_ => {}` fallback would
        // silently narrow freeze enforcement to top-level statements only).
        HirStmt::ComptimeBlock { body, .. }
        | HirStmt::Isolate { body, .. }
        | HirStmt::Unsafe { body, .. } => {
            for stmt in body {
                let mut next2 = 1usize;
                collect_stmt(stmt, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        HirStmt::ScopeCleanup {
            when_condition,
            body,
            ..
        } => {
            if let Some(cond) = when_condition {
                let mut next2 = 1usize;
                collect_expr(
                    cond, pt, &mut next2, false, None, loans, events, kills, registry, ctx,
                );
            }
            for stmt in body {
                let mut next2 = 1usize;
                collect_stmt(stmt, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        HirStmt::GhostVariableDef { inner, .. } => {
            collect_stmt(inner, pt, next, loans, events, kills, registry, ctx);
        }
        _ => {}
    }
}

fn collect_expr<'input>(
    e: &HirExpr<'input>,
    pt: Point,
    next: &mut usize,
    in_place_base: bool,
    bind: Option<Symbol>,
    loans: &mut Vec<(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )>,
    events: &mut Vec<(FrozenPlace, Point, bool, crate::ast::Span)>,
    kills: &mut Vec<(FrozenPlace, Option<Symbol>, Point)>,
    registry: &[(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) {
    // Every expression node consumes its own point (the pre-order
    // evaluation order — rustc's CFG points).
    let ept = Point { expr: *next, ..pt };
    *next += 1;
    match e {
        // Closures/tasks are nested borrowing domains — their
        // bodies' loans/events must be collected.  Previously the `_ => {}`
        // fallback skipped them, weakening the freeze guarantee inside
        // closures (the old interleaved `frozen_vars` mechanism did cover
        // them via check_stmt).
        HirExpr::Closure { body, .. } => {
            for stmt in body {
                let mut next2 = 1usize;
                collect_stmt(stmt, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        HirExpr::Task { block, .. } => {
            for stmt in block {
                let mut next2 = 1usize;
                collect_stmt(stmt, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        HirExpr::Ident(name, _, span) => {
            // A plain variable read — UNLESS it is the base of a larger
            // place (`a` in `a.b`), where the FULL place is reported at
            // the top node (so `&mut a.b` frozen does not over-reject
            // reading a sibling `a.c`).
            if !in_place_base {
                events.push((FrozenPlace::Root(*name), ept, true, *span));
            }
        }
        HirExpr::UnaryOp { op, expr, span, .. }
            if matches!(op, UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref) =>
        {
            // A borrow site: register a temporary loan (live only in this
            // block — the borrow operand is not a read).
            if let Some(p) = hir_expr_place(expr) {
                // Mirror the VariableDef borrow wrap above: an EXPRESSION-
                // level `&ro r` (`takes_shared(&ro r)` — `r` a reference
                // variable, `&ro` requires a reference operand, E111)
                // freezes the REFERENT, so wrap the place in a deref for
                // the polonius referent resolution — otherwise the
                // temporary loan stays on `Root(r)` and a later write to
                // the referent (`a = 5` after `&ro r` in an argument) is
                // accepted (the expression-level reborrow hole).  Any
                // `&ro X` operand (not just Ident) wraps to the referent.
                let place = match (op, expr.as_ref()) {
                    (UnaryOp::Ro, HirExpr::Ident(name, _, _)) => {
                        FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                    }
                    (UnaryOp::Ro, _) => FrozenPlace::Deref(Box::new(p)),
                    _ => p,
                };
                loans.push((place, None, borrow_kind(op), ept, *span, false));
            }
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr,
            span,
            ..
        } => {
            // The top of a deref-place read: report the FULL place.
            if !in_place_base {
                if let Some(p) = hir_expr_place(e) {
                    events.push((p, ept, true, *span));
                }
            }
            collect_expr(
                expr, pt, next, true, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::FieldAccess { base, span, .. } => {
            if !in_place_base {
                if let Some(p) = hir_expr_place(e) {
                    events.push((p, ept, true, *span));
                }
            }
            collect_expr(
                base, pt, next, true, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::Index {
            base, index, span, ..
        } => {
            if !in_place_base {
                if let Some(p) = hir_expr_place(e) {
                    events.push((p, ept, true, *span));
                }
            }
            collect_expr(
                base, pt, next, true, bind, loans, events, kills, registry, ctx,
            );
            collect_expr(
                index, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::UnaryOp { expr, .. } => collect_expr(
            expr, pt, next, false, bind, loans, events, kills, registry, ctx,
        ),
        HirExpr::BinaryOp { left, right, .. } => {
            collect_expr(
                left, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            collect_expr(
                right, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::Call { callee, args, .. } => {
            // The METHOD-CALL RECEIVER loan (`obj.put(x)` — the implicit
            // `&mut obj` / `&obj` receiver autoref): issued at the
            // CALLEE's expression point (the FieldAccess node) — the
            // same-point `obj.put` place read is the TPB reservation
            // (exempt), and the argument reads at LATER points of the
            // same statement are exempt too (the two-phase exemption is
            // ordered by the expression index).  The VariableDef arm
            // handles the CALL-AS-VALUE case itself (it does not recurse
            // here), so there is no duplicate registration.
            if let Some((is_method, receiver, positions, sig)) = resolve_call(callee, registry, ctx)
            {
                if is_method
                    && let Some(rcv) = receiver
                    && let Some(p) = hir_expr_place(rcv)
                {
                    let self_mut = sig.input_borrow_mutable.first().copied().unwrap_or(false);
                    let kind = if self_mut {
                        LoanKind::Exclusive
                    } else {
                        LoanKind::ReadOnly
                    };
                    // The receiver autoref is a TEMPORARY in rustc (NLL):
                    // it dies at the end of the call's statement (see the
                    // collect_stmt Call branch for the rationale).
                    loans.push((p, None, kind, ept, callee.span(), false));
                }
                // The returned-borrow-covered argument positions (a call NESTED inside
                // an aggregate/argument — `set t = (get(&mut a), 1)`): the
                // argument borrow IS the returned borrow of the enclosing
                // binding — register it VAR-BOUND to the binding name
                // (same semantics as the VariableDef CALL-AS-VALUE arm), so
                // the returned borrow freezes the source while the binding
                // lives.  Without the binding, a `&mut a` inside a tuple
                // would degrade to a temporary loan that dies at the
                // statement's end — a mutation in the NEXT statement would
                // be silently accepted (the temporary-borrow collapse).
                let mut pb_loans: HashMap<usize, (FrozenPlace, LoanKind)> = HashMap::new();
                for &(input_origin, _out_origin) in &sig.known_placeholder_subset {
                    if let Some(&pos) = positions.get(input_origin as usize)
                        && let Some(arg) = args.get(pos)
                        && let Some(place) = hir_expr_place(arg)
                    {
                        let kind = if sig
                            .input_borrow_mutable
                            .get(input_origin as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            LoanKind::Exclusive
                        } else {
                            LoanKind::ReadOnly
                        };
                        let place = match arg {
                            HirExpr::Ident(name, _, _) => {
                                FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                            }
                            // An explicit `&ro r` ARGUMENT (`set s =
                            // takes_shared(&ro r)` — the `&ro` itself is
                            // the borrow, `r` a reference variable):
                            // same referent-resolution wrap, so the
                            // temporary returned-borrow loan lands on `a` and the
                            // `a = 5` mutation-freeze is seen (E0506) —
                            // mirroring the VariableDef/expression-level
                            // borrow wraps above.
                            HirExpr::UnaryOp {
                                op: crate::ast::UnaryOp::Ro,
                                expr,
                                ..
                            } => match expr.as_ref() {
                                HirExpr::Ident(name, _, _) => {
                                    FrozenPlace::Deref(Box::new(FrozenPlace::Root(*name)))
                                }
                                // A non-Ident `&ro` ARGUMENT operand
                                // (still reference-typed per E111) must
                                // also land on the referent.
                                _ => FrozenPlace::Deref(Box::new(place)),
                            },
                            _ => place,
                        };
                        pb_loans.insert(pos, (place, kind));
                    }
                }
                collect_expr(
                    callee, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
                for (idx, arg) in args.iter().enumerate() {
                    if let Some((place, kind)) = pb_loans.get(&idx) {
                        loans.push((
                            place.clone(),
                            bind,
                            *kind,
                            Point { expr: *next, ..pt },
                            arg.span(),
                            true,
                        ));
                        *next += 1;
                        if let HirExpr::UnaryOp {
                            op: UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref,
                            expr,
                            ..
                        } = arg
                        {
                            collect_target_reads(
                                expr, pt, next, bind, loans, events, kills, registry, ctx,
                            );
                        } else {
                            collect_expr(
                                arg, pt, next, false, bind, loans, events, kills, registry, ctx,
                            );
                        }
                    } else {
                        collect_expr(
                            arg, pt, next, false, bind, loans, events, kills, registry, ctx,
                        );
                    }
                }
            } else {
                collect_expr(
                    callee, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
                for a in args {
                    collect_expr(
                        a, pt, next, false, bind, loans, events, kills, registry, ctx,
                    );
                }
            }
        }
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            collect_expr(
                scrutinee, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            // The match arms' guards and bodies READ their expressions —
            // the borrow-check must see the arms' reads/loans  .
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_expr(
                        g, pt, next, false, bind, loans, events, kills, registry, ctx,
                    );
                }
                collect_expr(
                    &a.body, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
        }
        // The EXPRESSION-form if/if-let branches READ their statements —
        // the borrow-check must see the branches' reads/loans  .
        HirExpr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_expr(
                cond, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            for s in then_branch {
                let mut next2 = 1usize;
                collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
            }
            if let Some(e) = else_branch {
                for s in e {
                    let mut next2 = 1usize;
                    collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
                }
            }
        }
        HirExpr::IfLet {
            scrutinee,
            then_branch,
            else_branch,
            ..
        } => {
            collect_expr(
                scrutinee, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            for s in then_branch {
                let mut next2 = 1usize;
                collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
            }
            if let Some(e) = else_branch {
                for s in e {
                    let mut next2 = 1usize;
                    collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
                }
            }
        }
        // The error-exit value (`leave with e`) is a READ of its expression
        // — the borrow-check must see the value's reads/loans.
        HirExpr::LeaveWith { expr, .. } => collect_expr(
            expr, pt, next, false, bind, loans, events, kills, registry, ctx,
        ),
        // The `catch` branches (the error handlers) READ their bodies —
        // the borrow-check must see the branch bodies' reads/loans.
        HirExpr::Catch { expr, branches, .. } => {
            collect_expr(
                expr, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            for b in branches {
                for s in &b.body {
                    let mut next2 = 1usize;
                    collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
                }
            }
        }
        HirExpr::Block(stmts, _, _) => {
            for s in stmts {
                let mut next2 = 1usize;
                collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        // Aggregates and other expression forms: recurse so loans/events
        // INSIDE tuple/struct/array/cast/range literals are collected — a
        // borrow's last use hidden in an aggregate must keep the loan
        // live (a `_ => {}` fallback would silently narrow the freeze).
        HirExpr::Tuple(elems, _, _) | HirExpr::Array(elems, _, _) => {
            for el in elems {
                collect_expr(
                    el, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
        }
        HirExpr::StructLit { fields, .. } => {
            for (_, val) in fields {
                collect_expr(
                    val, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
        }
        HirExpr::EnumLit { payload, .. } => {
            if let Some(p) = payload {
                collect_expr(
                    p, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
        }
        HirExpr::Move(inner, _, _) => collect_expr(
            inner, pt, next, false, bind, loans, events, kills, registry, ctx,
        ),
        HirExpr::Cast { expr, .. }
        | HirExpr::TypeAnnotated { expr, .. }
        | HirExpr::Try { expr, .. }
        | HirExpr::Await { expr, .. }
        | HirExpr::Old { expr, .. }
        | HirExpr::PolyBox { expr, .. }
        | HirExpr::PolyUnbox { expr, .. }
        | HirExpr::Return { value: expr, .. } => collect_expr(
            expr, pt, next, false, bind, loans, events, kills, registry, ctx,
        ),
        HirExpr::Range { start, end, .. } => {
            if let Some(s) = start {
                collect_expr(
                    s, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
            if let Some(e) = end {
                collect_expr(
                    e, pt, next, false, bind, loans, events, kills, registry, ctx,
                );
            }
        }
        HirExpr::AttrAccess { base, .. } => collect_expr(
            base, pt, next, false, bind, loans, events, kills, registry, ctx,
        ),
        HirExpr::Quantified { range, body, .. } => {
            collect_expr(
                range, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
            collect_expr(
                body, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::UnsafeBlock { body, .. } => {
            for s in body {
                let mut next2 = 1usize;
                collect_stmt(s, pt, &mut next2, loans, events, kills, registry, ctx);
            }
        }
        _ => {}
    }
}

/// Collect the READ events inside an assign target: the INDEX expressions
/// (`arr[idx] = 5` reads `idx`) — the place base chain is the write
/// target, not a read (reading a frozen index must be rejected — E109).
fn collect_target_reads<'input>(
    e: &HirExpr<'input>,
    pt: Point,
    next: &mut usize,
    bind: Option<Symbol>,
    loans: &mut Vec<(
        FrozenPlace,
        Option<Symbol>,
        LoanKind,
        Point,
        crate::ast::Span,
        bool,
    )>,
    events: &mut Vec<(FrozenPlace, Point, bool, crate::ast::Span)>,
    kills: &mut Vec<(FrozenPlace, Option<Symbol>, Point)>,
    registry: &[(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) {
    match e {
        HirExpr::Index { base, index, .. } => {
            collect_target_reads(base, pt, next, bind, loans, events, kills, registry, ctx);
            collect_expr(
                index, pt, next, false, bind, loans, events, kills, registry, ctx,
            );
        }
        HirExpr::FieldAccess { base, .. } => {
            collect_target_reads(base, pt, next, bind, loans, events, kills, registry, ctx)
        }
        HirExpr::UnaryOp {
            op: UnaryOp::Deref,
            expr,
            ..
        } => collect_target_reads(expr, pt, next, bind, loans, events, kills, registry, ctx),
        _ => {}
    }
}

/// Resolve the callee against the signature registry with a STRUCTURAL
/// receiver-type match: the registry key includes the RECEIVER type
/// (methods only — `None` for free functions), so same-name methods on
/// DIFFERENT receivers (`impl A { def get }` / `impl B { def get }`)
/// match exactly instead of find-first-match.  The receiver match is
/// structural because `TypeId` is an arena index (not hash-consed) —
/// two separately-resolved copies of the same nominal type have
/// different ids, so they must be compared via the resolved `TypeData`.
fn resolve_call<'a, 'input>(
    callee: &'a HirExpr<'input>,
    registry: &'a [(
        Symbol,
        bool,
        Option<crate::hir::types::TypeId>,
        Vec<usize>,
        crate::hir::types::SignatureFacts,
    )],
    ctx: &crate::hir::types::TypeContext<'input>,
) -> Option<(
    bool,
    Option<&'a HirExpr<'input>>,
    &'a Vec<usize>,
    &'a crate::hir::types::SignatureFacts,
)> {
    let (fname, is_method, receiver) = match callee {
        HirExpr::Ident(fn_name, _, _) => (fn_name, false, None),
        // The method calls: `obj.method(args)` — the callee is a field
        // access; the method name is the field, the BASE is the receiver
        // (its implicit borrow — rustc's autoref).
        HirExpr::FieldAccess { base, field, .. } => (field, true, Some(base.as_ref())),
        _ => return None,
    };
    registry
        .iter()
        .find(|(nm, is_m, rcv, _, _)| {
            nm == fname
                && *is_m == is_method
                && match (is_method, rcv) {
                    (true, Some(rty)) => match receiver {
                        Some(r) => ctx.get(*rty) == ctx.get(r.ty()),
                        None => false,
                    },
                    (false, None) => true,
                    _ => false,
                }
        })
        .map(|(_, _, _, positions, sig)| (is_method, receiver, positions, sig))
}

struct CfgBuilder<'input> {
    blocks: Vec<BasicBlock<'input>>,
    /// Enclosing loop-head stack, for `continue` targets.
    loop_stack: Vec<(BlockId, Option<Symbol>)>,
    /// Enclosing loop-EXIT stack, for `leave;` targets (SYNTAX.md:
    /// `leave;` exits the enclosing loop — the exit block).  Carries the
    /// loop's label so `leave 'label;` can target an OUTER loop, mirroring
    /// `loop_stack`'s labeled entries for `continue`.
    loop_exits: Vec<(BlockId, Option<Symbol>)>,
}

impl<'input> CfgBuilder<'input> {
    fn new() -> Self {
        CfgBuilder {
            blocks: Vec::new(),
            loop_stack: Vec::new(),
            loop_exits: Vec::new(),
        }
    }

    /// Wire a `finally` block onto every function-exit edge: each block
    /// whose terminator is a function exit (`Return` — the `leave with`
    /// exit is `Leave`) jumps into the finally block FIRST, which then
    /// terminates the function (SYNTAX.md §finally — "at scope exit on
    /// all paths").
    fn attach_finally(&mut self, finally: &[HirStmt<'input>]) {
        if finally.is_empty() {
            return;
        }
        // Each exit path gets its OWN finally sequence, terminated with
        // the ORIGINAL exit kind — a single shared sequence cannot
        // preserve `leave with` (error) vs `return` (success) distinctions
        // (a shared tail hardcoded to `Return` would swallow error exits).
        let exits: Vec<(BlockId, Terminator<'input>)> = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, blk)| match &blk.terminator {
                Some(t @ (Terminator::Return | Terminator::Leave)) => Some((BlockId(i), t.clone())),
                _ => None,
            })
            .collect();
        for (blk, original_term) in exits {
            // A FRESH finally sequence per exit path, so each path can
            // resume with its own exit kind afterwards.
            let (fentry, fexit) = self.build_seq(finally, None);
            // INTENTIONAL replacement: the exit edge is re-routed through
            // the finally block — the block is ALREADY terminated
            // (Return/Leave), so `set_terminator`'s `debug_assert!` (which
            // requires an unterminated block) must be bypassed with a
            // direct assignment.
            self.blocks[blk.0].terminator = Some(Terminator::Goto(fentry));
            // The finally sequence's tail terminates the function with the
            // ORIGINAL exit kind (`leave with` stays an error exit — it
            // must NOT be silently converted into a successful `return`).
            if let Some(fexit) = fexit
                && self.blocks[fexit.0].terminator.is_none()
            {
                self.set_terminator(fexit, original_term);
            }
        }
    }

    fn new_block(&mut self) -> BlockId {
        self.blocks.push(BasicBlock::new());
        BlockId(self.blocks.len() - 1)
    }

    fn set_terminator(&mut self, blk: BlockId, term: Terminator<'input>) {
        debug_assert!(
            self.blocks[blk.0].terminator.is_none(),
            "block {blk:?} already terminated"
        );
        self.blocks[blk.0].terminator = Some(term);
    }

    /// Build the CFG for a statement sequence.  Returns the block after the
    /// sequence (the join point), or `None` if the sequence ends in a
    /// terminator (`return`/`leave`).
    /// Returns `(entry, exit)`: the sequence's FIRST block (the entry —
    /// what predecessors must target) and its LAST block (the exit — where
    /// back edges attach; `None` when the sequence ended in return/leave).
    /// The previous single-block return conflated the entry with the join,
    /// making the loop/if bodies with control flow have UNREACHABLE entries
    ///   — false accepts + the `unwrap()` ICE on leave/return bodies).
    fn build_seq(
        &mut self,
        stmts: &[HirStmt<'input>],
        loop_head: Option<BlockId>,
    ) -> (BlockId, Option<BlockId>) {
        let entry = self.new_block();
        let mut cur = entry;
        for stmt in stmts {
            match stmt {
                // ── Control-flow statements ──
                HirStmt::If { .. } | HirStmt::IfLet { .. } => {
                    let (cond, then_branch, else_branch) = match stmt {
                        HirStmt::If {
                            cond,
                            then_branch,
                            else_branch,
                            ..
                        } => (cond, then_branch, else_branch),
                        HirStmt::IfLet {
                            scrutinee,
                            then_branch,
                            else_branch,
                            ..
                        } => (scrutinee, then_branch, else_branch),
                        _ => unreachable!(),
                    };
                    // Terminate the current block with a branch.
                    let (then_, then_exit) = self.build_seq(then_branch, loop_head);
                    let (else_, else_exit) = match else_branch.as_deref() {
                        Some(e) => self.build_seq(e, loop_head),
                        None => {
                            let b = self.new_block();
                            (b, Some(b))
                        }
                    };
                    let cond = Box::new(cond.as_ref().clone());
                    self.set_terminator(cur, Terminator::Branch { cond, then_, else_ });
                    // The join block after the if: the arms' EXITS flow
                    // into it (unless an arm already ends in return/leave).
                    // Wiring the ENTRIES would skip arms with nested control
                    // flow (the entry already carries a Branch terminator)
                    // and orphan their true exit — a severed CFG edge
                    //   — false acceptance).
                    let join = self.new_block();
                    for arm_exit in [then_exit, else_exit] {
                        if let Some(exit) = arm_exit {
                            if self.blocks[exit.0].terminator.is_none() {
                                self.blocks[exit.0].terminator = Some(Terminator::Goto(join));
                            }
                        }
                    }
                    cur = join;
                }
                HirStmt::While { label, body, .. }
                | HirStmt::WhileLet { label, body, .. }
                | HirStmt::For { label, body, .. }
                | HirStmt::Loop { label, body, .. } => {
                    // Loop: a head block (condition for while/while-let,
                    // plain goto-head for for/loop), a body, a back edge
                    // from the body's end back to the head, and an exit
                    // block (the cond-fail + the `leave;` target) that
                    // flows into the after-loop code.
                    let head = self.new_block();
                    let exit = self.new_block();
                    let after_loop = self.new_block();
                    self.loop_stack.push((head, *label));
                    self.loop_exits.push((exit, *label));
                    let (body_entry, body_exit) = self.build_seq(body, Some(head));
                    self.loop_stack.pop();
                    self.loop_exits.pop();
                    // The body's LAST block (the exit) jumps back to the
                    // head (back edge); the head's branch targets the
                    // body's ENTRY.
                    if let Some(last_body) = body_exit {
                        self.set_terminator(last_body, Terminator::Goto(head));
                    }
                    // Head terminates with a conditional branch (or, for
                    // for/loop with no cond, a plain goto into the body).
                    match stmt {
                        HirStmt::While { .. } | HirStmt::WhileLet { .. } => {
                            let cond: Box<HirExpr<'input>> = match stmt {
                                HirStmt::While { cond, .. } => Box::new(cond.as_ref().clone()),
                                // WhileLet: the scrutinee is used as the
                                // BRANCH CONDITION (a boolean branch on
                                // the scrutinee's truthiness).  The
                                // pattern-MATCHING semantics are handled
                                // by the type checker; the CFG only
                                // models the control-flow structure
                                // (loop head branches to body or exit).
                                HirStmt::WhileLet { scrutinee, .. } => {
                                    Box::new(scrutinee.as_ref().clone())
                                }
                                _ => unreachable!(),
                            };
                            self.set_terminator(
                                head,
                                Terminator::Branch {
                                    cond,
                                    then_: body_entry,
                                    else_: exit,
                                },
                            );
                        }
                        _ => {
                            // for / loop: unconditional entry into the body
                            // (the entry always exists — no unwrap needed).
                            self.set_terminator(head, Terminator::Goto(body_entry));
                        }
                    }
                    // The For's ITERABLE is evaluated ONCE in a PRE-HEADER
                    // block — re-evaluating it in the head on every
                    // iteration would over-approximate the liveness and
                    // re-issue the iterable's loans per iteration.  The
                    // back edge still targets the head.
                    if let HirStmt::For { iterable, .. } = stmt {
                        let pre_header = self.new_block();
                        self.blocks[pre_header.0]
                            .stmts
                            .push(HirStmt::Expression(iterable.clone()));
                        self.set_terminator(pre_header, Terminator::Goto(head));
                        self.set_terminator(cur, Terminator::Goto(pre_header));
                    } else {
                        // The pre-loop block jumps to the head; the exit
                        // (the cond-fail / `leave;`) flows into the
                        // after-loop code.
                        self.set_terminator(cur, Terminator::Goto(head));
                    }
                    self.set_terminator(exit, Terminator::Goto(after_loop));
                    cur = after_loop;
                }
                HirStmt::Return { .. } => {
                    // Keep the Return STATEMENT (with its value expression)
                    // in the block's statements — the return value's
                    // variable uses must be visible to the liveness and
                    // the borrow-check pass.  The terminator is set to
                    // Return (no successors).
                    self.blocks[cur.0].stmts.push(stmt.clone());
                    self.set_terminator(cur, Terminator::Return);
                    return (entry, None);
                }
                HirStmt::Leave { label, .. } => {
                    // The leave STATEMENT stays in the block's statements
                    // (consistent with the Return/LeaveWith value-fidelity
                    // pattern); the control flow is the terminator.
                    self.blocks[cur.0].stmts.push(stmt.clone());
                    // `leave;` exits the ENCLOSING LOOP (SYNTAX.md §Loops:
                    // "leave; (exits for, while, or loop)") — jump to the
                    // loop's exit block; `leave 'label;` targets the
                    // labeled outer loop's exit.  Outside a loop it is a
                    // function error exit.
                    let target = match label {
                        Some(l) => self
                            .loop_exits
                            .iter()
                            .rev()
                            .find(|(_, lbl)| *lbl == Some(*l))
                            .map(|(e, _)| *e),
                        None => self.loop_exits.last().map(|(e, _)| *e),
                    };
                    if let Some(exit) = target {
                        self.set_terminator(cur, Terminator::Goto(exit));
                    } else {
                        self.set_terminator(cur, Terminator::Leave);
                    }
                    return (entry, None);
                }
                HirStmt::Continue { label, .. } => {
                    // Resolve the continue target: a labeled
                    // `continue 'l;` jumps to the enclosing loop whose
                    // label matches `l` (SYNTAX.md §Loops — "continue to
                    // outer labels"); an unlabeled `continue;` jumps to
                    // the nearest enclosing loop.
                    let target = match label {
                        Some(l) => self
                            .loop_stack
                            .iter()
                            .rev()
                            .find(|(_, lbl)| *lbl == Some(*l))
                            .map(|(h, _)| *h),
                        None => self.loop_stack.last().map(|(h, _)| *h),
                    };
                    match target {
                        Some(target) => {
                            self.set_terminator(cur, Terminator::Goto(target));
                            cur = self.new_block();
                        }
                        None => {
                            // `continue` outside a loop is already
                            // diagnosed by the type checker (which
                            // returns `Ok` for recovery) — the CFG must
                            // degrade gracefully instead of panicking
                            // (the ICE).  The terminator is Unreachable
                            // and no successor block is allocated — the
                            // unreachable tail is dropped rather than
                            // building a dead block.
                            self.set_terminator(cur, Terminator::Unreachable);
                            return (entry, None);
                        }
                    }
                }
                // ── Straight-line statements ──
                // Item declarations are not executed; skip them.
                HirStmt::FunctionDef { .. }
                | HirStmt::TypeDef { .. }
                | HirStmt::TraitDef { .. }
                | HirStmt::Import { .. }
                | HirStmt::ExternFunction { .. } => {}
                // Everything else is straight-line.
                // `leave with e;` DIVERGES (the function error exit — the
                // Leave terminator) — no subsequent statement in the block
                // is reachable  .
                HirStmt::Expression(e) if matches!(e.as_ref(), HirExpr::LeaveWith { .. }) => {
                    self.blocks[cur.0].stmts.push(stmt.clone());
                    self.set_terminator(cur, Terminator::Leave);
                    return (entry, None);
                }
                // A MATCH expression is a control-flow node: each arm gets
                // its OWN block so loans inside one arm are not conflated
                // with another arm's (the previous folding
                // previous folding forced arm-local liveness into a single
                // point, over-freezing across arms).
                HirStmt::Expression(e) if matches!(e.as_ref(), HirExpr::Match { .. }) => {
                    let HirExpr::Match {
                        scrutinee, arms, ..
                    } = e.as_ref()
                    else {
                        unreachable!()
                    };
                    self.blocks[cur.0]
                        .stmts
                        .push(HirStmt::Expression(scrutinee.clone()));
                    let mut arm_entries = Vec::new();
                    let mut arm_exits = Vec::new();
                    for arm in arms {
                        let (arm_entry, arm_exit) =
                            self.build_seq(&[HirStmt::Expression(arm.body.clone())], loop_head);
                        arm_entries.push(arm_entry);
                        if let Some(exit) = arm_exit {
                            arm_exits.push(exit);
                        }
                    }
                    // Chain the conditional Branches from the LAST arm
                    // backwards: a Literal pattern gets a precise
                    // `scrutinee == literal` condition (then_ = this arm,
                    // else_ = the previous arm in the chain / match exit);
                    // Ident/Wildcard (and destructuring) are treated as
                    // trivially true — the arm stays isolated either way.
                    //
                    // Chain invariant: arm k's ELSE falls through to arm
                    // k-1's TEST BLOCK (arms are tested in source order —
                    // arm 0 first, whose test block is the scrutinee block
                    // itself).  Each non-first arm gets a fresh test block;
                    // the last arm's else is the match exit.
                    let scrutinee_block = cur;
                    let mut else_target: Option<BlockId> = None;
                    for (idx, arm) in arms.iter().enumerate().rev() {
                        let then_ = arm_entries[idx];
                        let else_ = else_target.unwrap_or_else(|| self.new_block());
                        // Arm 0 is tested in the scrutinee block; every
                        // later arm gets its own dispatch block.
                        let test_block = if idx == 0 {
                            scrutinee_block
                        } else {
                            self.new_block()
                        };
                        // Three-way dispatch on the pattern's refutability:
                        //  - Literal: a precise `scrutinee == lit` boolean
                        //    condition (then_ = this arm, else_ = fall-through).
                        //  - Ident / Wildcard: IRREFUTABLE — always matches,
                        //    so an unconditional Goto into the arm body is
                        //    sound (no else_ path exists).
                        //  - Any other REFUTABLE pattern (Enum / Struct /
                        //    Tuple / Or / Slice): no boolean HIR condition
                        //    can be synthesized, but the pattern may match
                        //    OR fail — emit a conservative dual-edge
                        //    `Switch` so BOTH the arm body and the fall-
                        //    through (later arms / match exit) stay
                        //    reachable.  (These previously fell into the
                        //    `_ => None` catch-all and got `Goto(then_)`,
                        //    severing the edge to every later arm and
                        //    hiding their moves/borrows from the
                        //    flow-sensitive analyses.)
                        match arm.pattern {
                            crate::hir::hir::HirPattern::Literal(ref lit, _) => {
                                let scrut_ty = scrutinee.ty();
                                self.set_terminator(
                                    test_block,
                                    Terminator::Branch {
                                        cond: Box::new(HirExpr::BinaryOp {
                                            left: scrutinee.clone(),
                                            op: crate::ast::BinOp::Eq,
                                            right: lit.clone(),
                                            ty: scrut_ty,
                                            span: arm.span,
                                        }),
                                        then_,
                                        else_,
                                    },
                                );
                            }
                            crate::hir::hir::HirPattern::Ident(..)
                            | crate::hir::hir::HirPattern::Wildcard(_) => {
                                self.set_terminator(test_block, Terminator::Goto(then_));
                            }
                            _ => {
                                self.set_terminator(
                                    test_block,
                                    Terminator::Switch { then_, else_ },
                                );
                            }
                        }
                        // The NEXT (earlier) arm falls through to THIS
                        // arm's test block — not to this arm's else
                        // (storing `else_` here orphaned every non-last
                        // dispatch block and inverted the test order).
                        else_target = Some(test_block);
                    }
                    cur = scrutinee_block;
                    // The match's arms join at the exit; if there are no
                    // exits (all arms diverge), the current block is
                    // unreachable.
                    if let Some(&first_exit) = arm_exits.first() {
                        for &exit in arm_exits.iter() {
                            if exit != first_exit {
                                self.set_terminator(exit, Terminator::Goto(first_exit));
                            }
                        }
                        cur = first_exit;
                    } else {
                        // ALL arms diverge: the scrutinee block is ALREADY
                        // terminated (the dispatch chain) — a following
                        // control-flow statement would set_terminator on
                        // it (debug ICE / release silent overwrite of the
                        // dispatch chain).  Start a FRESH dead block
                        // instead (finish terminalizes it Unreachable),
                        // matching the `Return` arm's stop semantics.
                        cur = self.new_block();
                    }
                }
                HirStmt::Expression(e) if matches!(e.as_ref(), HirExpr::If { .. }) => {
                    // The IF EXPRESSION (`let y = if c { .. } else { .. }`)
                    // gets the SAME block splitting as the if STATEMENT:
                    // the then/else branch bodies become independent CFG
                    // blocks.  Previously the whole expression fell to
                    // `_ =>` — pushed into the current block — so the
                    // branch bodies' moves were never tracked by the
                    // block-level dataflow and a use-after-move inside an
                    // expression-position branch slipped through.
                    let HirExpr::If {
                        cond,
                        then_branch,
                        else_branch,
                        ..
                    } = e.as_ref()
                    else {
                        unreachable!()
                    };
                    let (then_, then_exit) = self.build_seq(then_branch, loop_head);
                    let (else_, else_exit) = match else_branch.as_deref() {
                        Some(e) => self.build_seq(e, loop_head),
                        None => {
                            let b = self.new_block();
                            (b, Some(b))
                        }
                    };
                    let cond = Box::new(cond.as_ref().clone());
                    self.set_terminator(cur, Terminator::Branch { cond, then_, else_ });
                    // The join block after the if expression: the arms'
                    // EXITS flow into it (unless an arm already ends in
                    // return/leave) — mirroring the if-statement wiring.
                    let join = self.new_block();
                    for arm_exit in [then_exit, else_exit] {
                        if let Some(exit) = arm_exit {
                            if self.blocks[exit.0].terminator.is_none() {
                                self.blocks[exit.0].terminator = Some(Terminator::Goto(join));
                            }
                        }
                    }
                    cur = join;
                }
                _ => self.blocks[cur.0].stmts.push(stmt.clone()),
            }
        }
        (entry, Some(cur))
    }

    fn finish(mut self, entry: BlockId) -> CfgGraph<'input> {
        let n = self.blocks.len();
        let mut successors = vec![Vec::new(); n];
        let mut predecessors = vec![Vec::new(); n];
        // Terminalize blocks without a terminator.
        for i in 0..n {
            if self.blocks[i].terminator.is_none() {
                self.blocks[i].terminator = Some(Terminator::Unreachable);
            }
        }
        // Collect edges.
        for i in 0..n {
            let succs: Vec<BlockId> = match self.blocks[i].terminator.as_ref().unwrap() {
                Terminator::Goto(t) => vec![*t],
                Terminator::Branch { then_, else_, .. } => vec![*then_, *else_],
                // Conservative dual-edge: both the arm body and the
                // fall-through stay reachable (refutable pattern).
                Terminator::Switch { then_, else_ } => vec![*then_, *else_],
                Terminator::Return | Terminator::Leave | Terminator::Unreachable => vec![],
            };
            for s in &succs {
                successors[i].push(*s);
                predecessors[s.0].push(BlockId(i));
            }
        }
        // Back edges: an edge (u, v) where v is an ancestor of u in a DFS
        // from the entry (standard loop-edge detection).
        let mut back_edges = Vec::new();
        {
            let mut visited = vec![false; n];
            let mut on_stack = vec![false; n];
            fn dfs(
                u: BlockId,
                graph: &[Vec<BlockId>],
                visited: &mut [bool],
                on_stack: &mut [bool],
                back_edges: &mut Vec<(BlockId, BlockId)>,
            ) {
                visited[u.0] = true;
                on_stack[u.0] = true;
                for &v in &graph[u.0] {
                    if !visited[v.0] {
                        dfs(v, graph, visited, on_stack, back_edges);
                    } else if on_stack[v.0] {
                        back_edges.push((u, v));
                    }
                }
                on_stack[u.0] = false;
            }
            dfs(
                entry,
                &successors,
                &mut visited,
                &mut on_stack,
                &mut back_edges,
            );
        }
        CfgGraph {
            blocks: self.blocks,
            entry,
            successors,
            predecessors,
            back_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::checker::tests::check_source;

    /// Extract the body of the FIRST function in the program.
    fn first_body(src: &str) -> Vec<HirStmt<'static>> {
        let prog = check_source(src).expect("program must check");
        let item = &prog.items[0];
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        body.clone().unwrap_or_default()
    }

    /// Regression: `attach_finally` must preserve the ORIGINAL exit kind
    /// — a `leave with` (error) exit routed through the `finally`
    /// sequence must still terminate with `Leave`, NOT be silently
    /// converted into a successful `Return` (which would swallow the
    /// error).
    #[test]
    fn test_attach_finally_preserves_leave_kind() {
        // A minimal `finally` body (any statement — the sequence must be
        // non-empty for `attach_finally` to engage).
        let fin_body = vec![HirStmt::Edition(
            "finally".to_string(),
            crate::ast::Span::new(0, 0),
        )];
        let mut builder = CfgBuilder::new();
        let ret_blk = builder.new_block();
        let leave_blk = builder.new_block();
        builder.set_terminator(ret_blk, Terminator::Return);
        builder.set_terminator(leave_blk, Terminator::Leave);
        builder.attach_finally(&fin_body);
        // Follow each exit's Goto chain: the finally sequence's tail must
        // terminate with the ORIGINAL exit kind.
        for (start, expected) in [
            (ret_blk, Terminator::Return),
            (leave_blk, Terminator::Leave),
        ] {
            let mut cur = start;
            let mut hops = 0;
            loop {
                hops += 1;
                assert!(hops < 100, "unexpected cycle in finally routing");
                match builder.blocks[cur.0].terminator.as_ref().unwrap().clone() {
                    Terminator::Goto(next) => cur = next,
                    t => {
                        let is_expected = match expected {
                            Terminator::Return => matches!(t, Terminator::Return),
                            Terminator::Leave => matches!(t, Terminator::Leave),
                            _ => unreachable!(),
                        };
                        assert!(
                            is_expected,
                            "finally tail must resume with the original exit kind"
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Straight-line body: one entry block, no back edges.
    #[test]
    fn test_cfg_straight_line() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 a = a + 1;
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        assert_eq!(cfg.blocks().len(), 1, "straight-line body is one block");
        assert!(cfg.back_edges().is_empty(), "no loops");
        // Entry block terminates with Return.
        assert!(matches!(
            cfg.block(cfg.entry()).terminator,
            Some(Terminator::Return)
        ));
    }

    /// `if` produces a branch: entry → then / else, joined after.
    #[test]
    fn test_cfg_if_branch() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 if a > 0 {
                     a = 1;
                 } else {
                     a = 2;
                 }
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        let succ = cfg.successors(cfg.entry());
        assert_eq!(succ.len(), 2, "if produces two successors");
        assert!(cfg.back_edges().is_empty(), "no loops");
    }

    /// `while` produces a loop: a back edge from the body back to the head.
    #[test]
    fn test_cfg_while_back_edge() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 while a > 0 {
                     a = a - 1;
                 }
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        assert_eq!(cfg.back_edges().len(), 1, "one loop back edge");
        let (from, to) = cfg.back_edges()[0];
        assert!(
            matches!(cfg.block(to).terminator, Some(Terminator::Branch { .. })),
            "back edge targets the loop head (a branch); got {from:?} -> {to:?}"
        );
    }

    /// `leave;` exits the ENCLOSING LOOP (SYNTAX.md §Loops: "leave; (exits
    /// for, while, or loop)"): the leave block's terminator must be a Goto
    /// to the loop's exit block (NOT the function-exit Leave), so the
    /// after-loop code stays reachable.
    #[test]
    fn test_cfg_leave_loop_exit() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 while a > 0 {
                     if a == 1 { leave; }
                     a = a - 1;
                 }
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        // The block containing the `leave;` statement.
        let leave_block = cfg
            .blocks()
            .iter()
            .position(|b| b.stmts.iter().any(|s| matches!(s, HirStmt::Leave { .. })))
            .expect("a block with the `leave;` statement");
        // The leave block must Goto the loop's exit (not terminate the
        // function with Leave) — the after-loop code is reachable.
        assert!(
            matches!(
                cfg.block(BlockId(leave_block)).terminator,
                Some(Terminator::Goto(_))
            ),
            "`leave;` must jump to the loop's exit block"
        );
    }

    /// Find the block containing the first `Assign` statement.
    fn assign_block(cfg: &CfgGraph) -> BlockId {
        let i = cfg
            .blocks()
            .iter()
            .position(|b| b.stmts.iter().any(|s| matches!(s, HirStmt::Assign { .. })))
            .expect("an Assign block");
        BlockId(i)
    }

    /// Set of blocks reachable from `start` via forward CFG edges.
    fn reachable_from(cfg: &CfgGraph, start: BlockId) -> HashSet<BlockId> {
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        while let Some(b) = stack.pop() {
            if seen.insert(b) {
                stack.extend(cfg.successors(b).iter().copied());
            }
        }
        seen
    }

    /// Regression: a match arm with a REFUTABLE (non-Literal) pattern —
    /// Enum, Struct, Tuple, Slice, Or — must keep the fall-through edge
    /// alive.  The dispatch block must be a `Switch` (dual edge, both the
    /// arm body AND the next arm reachable), NOT a `Goto(then_)` which
    /// severed the path to every later arm and hid their moves/borrows
    /// from the flow-sensitive analyses.
    #[test]
    fn test_cfg_match_refutable_arm_keeps_fallthrough() {
        // NOTE: the match must be a BARE expression statement
        // (`match x { ... };`) — build_seq's match branch only fires for
        // `HirStmt::Expression(HirExpr::Match)`.  `set a = match ...`
        // (VariableDef form) stays a straight-line statement and never
        // reaches the dispatch-chain code.
        let prog = check_source(
            "type Maybe = enum { Nothing, Just(Int<32>) }
             def f(x: Maybe) -> Int<32> {
                 match x {
                     Maybe::Just(v) => { set a = v; },
                     Maybe::Nothing => { set a = 0; },
                 };
                 return 0;
             }",
        )
        .expect("program must check");
        let item = &prog.items[1];
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        // Every refutable arm's dispatch must be a Switch.
        let switch_blocks: Vec<BlockId> = cfg
            .blocks()
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b.terminator, Some(Terminator::Switch { .. })))
            .map(|(i, _)| BlockId(i))
            .collect();
        assert!(
            !switch_blocks.is_empty(),
            "refutable match arms must emit Switch dispatch (found: none)"
        );
        for s in &switch_blocks {
            assert_eq!(
                cfg.successors(*s).len(),
                2,
                "each Switch keeps both the arm body and the fall-through reachable"
            );
        }
        // The whole dispatch chain must be reachable from the entry — no
        // severed arms.  (A Goto-only chain left the later arms' dispatch
        // blocks with no predecessors.)
        let reachable = reachable_from(&cfg, cfg.entry());
        for s in switch_blocks {
            assert!(
                reachable.contains(&s),
                "Switch dispatch block {s:?} must be reachable from the entry"
            );
        }
        // Every non-entry block that carries statements is reachable (no
        // orphaned arm body).
        let reachable = reachable_from(&cfg, cfg.entry());
        for (i, b) in cfg.blocks().iter().enumerate() {
            if i != cfg.entry().0 && !b.stmts.is_empty() {
                assert!(
                    reachable.contains(&BlockId(i)),
                    "block {i} (with statements) must be reachable from the entry"
                );
            }
        }
    }

    /// Control: an IRREFUTABLE arm (Ident / Wildcard) still gets an
    /// unconditional Goto — always matches, so no fall-through edge.
    #[test]
    fn test_cfg_match_irrefutable_arm_goto() {
        let prog = check_source(
            "def f(x: Int<32>) -> Int<32> {
                 set a = match x { _ => 7 };
                 return a;
             }",
        )
        .expect("program must check");
        let item = &prog.items[0];
        let HirStmt::FunctionDef { body, .. } = item else {
            panic!("expected a function def");
        };
        let body = body.clone().unwrap_or_default();
        let cfg = CfgGraph::build_function(&body, &[]);
        // No Switch dispatch anywhere: wildcard always matches.
        for b in cfg.blocks() {
            assert!(
                !matches!(b.terminator, Some(Terminator::Switch { .. })),
                "irrefutable wildcard arm must not emit Switch"
            );
        }
    }

    /// Variable liveness (the mechanism behind loan liveness): after the
    /// variable's LAST USE in a DIFFERENT block (the if-arm), it is dead
    /// at the later `a = 5` join block.  Plain variables — no borrows, so
    /// the checker accepts the program.
    #[test]
    fn test_liveness_var_dies_after_last_use() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r = 7;
                 if a > 0 { let x = r; }
                 a = 5;
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        let live = cfg.compute_var_liveness();
        let r = Symbol::intern("r");
        let blk = assign_block(&cfg);
        assert!(
            !live.is_live_at(r, blk),
            "r must be dead (no use reachable) at the `a = 5` join block"
        );
    }

    /// Variable liveness: `r` IS live at the `a = 5` block when it is
    /// used later (the if-arm is a successor) — the mutation must be
    /// rejected there.
    #[test]
    fn test_liveness_var_live_until_last_use() {
        let body = first_body(
            "def main() -> Int<32> {
                 set mut a = 42;
                 set r = 7;
                 a = 5;
                 if a > 0 { let x = r; }
                 return a;
             }",
        );
        let cfg = CfgGraph::build_function(&body, &[]);
        let live = cfg.compute_var_liveness();
        let r = Symbol::intern("r");
        let blk = assign_block(&cfg);
        assert!(
            live.is_live_at(r, blk),
            "r is used later (in the if-arm) — must be live at the `a = 5` block"
        );
    }

    /// The sparse-interval liveness query: the borrow variable is live at
    /// its use point and dead at the later statement (the same-block last
    /// use — the point-level precision, via the compressed intervals).
    #[test]
    fn test_liveness_intervals_point_precision() {
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
        let live = cfg.compute_point_liveness();
        let r = Symbol::intern("r");
        // The block: [set a(0), set r(1), let x=*r(2), a=5(3), return x(4)].
        assert!(
            live.is_live_at(
                r,
                Point {
                    block: BlockId(0),
                    stmt: 2,
                    expr: 0,
                }
            ),
            "r is live at its use (`*r`)"
        );
        assert!(
            !live.is_live_at(
                r,
                Point {
                    block: BlockId(0),
                    stmt: 3,
                    expr: 0,
                }
            ),
            "r is dead at the later `a = 5` (last use passed)"
        );
    }

    // ── Partial move (hollow) tests ──────────────────────────────────
    // NOTE: We use `[&mut Int<32>; N]` as the element type because
    // `&mut T` is non-Copy (mutable references are affine).  Struct types
    // like `String` are NOT registered as type names in the resolver.

    /// Explicit move of arr[0] then read arr[0] — must be rejected.
    #[test]
    fn test_partial_move_element_rejected() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut target = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut target, &mut target, &mut target];
                let s = move arr[0];
                let t = move arr[0];
                return 0;
            }",
        );
        assert!(
            result.is_err(),
            "reading a moved element must be rejected: {:?}",
            result
        );
    }

    /// Explicit move of arr[0] then read arr[1] — must be accepted.
    #[test]
    fn test_partial_move_sibling_ok() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut a = 0;
                set mut b = 0;
                set mut c = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut a, &mut b, &mut c];
                let s = move arr[0];
                let t = move arr[1];
                return 0;
            }",
        );
        assert!(
            result.is_ok(),
            "sibling element must still be accessible: {:?}",
            result
        );
    }

    /// Move arr[0], re-fill via whole-array assignment of a fresh array, then read arr[0] — must be accepted.
    #[test]
    fn test_partial_move_refill() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut a = 0;
                set mut b = 0;
                set mut c = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut a, &mut b, &mut c];
                let s = move arr[0];
                set mut d = 0;
                set mut e = 0;
                set mut f = 0;
                arr = [&mut d, &mut e, &mut f];
                let t = move arr[0];
                return 0;
            }",
        );
        assert!(
            result.is_ok(),
            "re-filled element must be accessible again: {:?}",
            result
        );
    }

    /// Move arr[0], whole-array overwrite, then read arr[0] — accepted.
    #[test]
    fn test_partial_move_whole_overwrite() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut t1 = 0;
                set mut t2 = 0;
                set mut t3 = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut t1, &mut t2, &mut t3];
                let s = move arr[0];
                arr = [&mut t1, &mut t2, &mut t3];
                let t = move arr[0];
                return 0;
            }",
        );
        assert!(
            result.is_ok(),
            "whole-array overwrite must clear hollow marks: {:?}",
            result
        );
    }

    /// Move arr[0], then read whole arr — must be rejected (hollow element
    /// invalidates the whole variable).
    #[test]
    fn test_partial_move_read_whole_array() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut target = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut target, &mut target, &mut target];
                let s = move arr[0];
                let t = arr;
                return 0;
            }",
        );
        assert!(
            result.is_err(),
            "reading whole array with hollow element must be rejected: {:?}",
            result
        );
    }

    /// Dynamic index move: `move arr[i]` then `move arr[i]` — must be
    /// rejected (dynamic index is conservatively any element).
    #[test]
    fn test_partial_move_dynamic_index() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut target = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut target, &mut target, &mut target];
                let i: Int<32> = 0;
                let s = move arr[i];
                let t = move arr[i];
                return 0;
            }",
        );
        assert!(
            result.is_err(),
            "dynamic index after move must be rejected: {:?}",
            result
        );
    }

    /// Dynamic index move then different literal index — must be rejected
    /// (dynamic is conservatively ANY element).
    #[test]
    fn test_partial_move_dynamic_then_literal() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut target = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut target, &mut target, &mut target];
                let i: Int<32> = 0;
                let s = move arr[i];
                let t = move arr[1];
                return 0;
            }",
        );
        assert!(
            result.is_err(),
            "dynamic index move must conservatively block literal index: {:?}",
            result
        );
    }

    /// Whole-array reassignment clears all moved marks for that variable.
    #[test]
    fn test_partial_move_whole_reassign() {
        let result = check_source(
            "def main() -> Int<32> {
                set mut t1 = 0;
                set mut t2 = 0;
                set mut t3 = 0;
                set mut arr: [&mut Int<32>; 3] = [&mut t1, &mut t2, &mut t3];
                let s = move arr[0];
                arr = [&mut t1, &mut t2, &mut t3];
                let t = move arr[0];
                return 0;
            }",
        );
        assert!(
            result.is_ok(),
            "whole-array reassignment must clear all moved marks: {:?}",
            result
        );
    }
}

/// The affine consumption walk: a non-Copy leaf packed into the RHS
/// (tuple, struct literal, array, enum variant, call argument) is
/// CONSUMED — its ownership transfers into the value.  The borrow
/// operators (`&mut`/`&`/`&ro`) do NOT consume the operand.
fn mark_consumed_places<'input>(
    e: &HirExpr<'input>,
    non_copy: &HashSet<Symbol>,
    moved: &mut HashSet<FrozenPlace>,
) {
    match e {
        HirExpr::Ident(name, _, _) => {
            if non_copy.contains(name) {
                moved.insert(FrozenPlace::Root(*name));
            }
        }
        // The borrow operators do NOT consume the operand.
        HirExpr::UnaryOp {
            op: UnaryOp::RefMut | UnaryOp::Ro | UnaryOp::Ref,
            ..
        } => {}

        // Aggregate values: each child is consumed.
        HirExpr::Tuple(elems, _, _) => mark_all(elems, non_copy, moved),
        HirExpr::Call { args, .. } => mark_all(args, non_copy, moved),
        HirExpr::Array(elems, _, _) => mark_all(elems, non_copy, moved),
        HirExpr::StructLit { fields, .. } => {
            mark_all(fields.iter().map(|(_, v)| v.as_ref()), non_copy, moved);
        }
        HirExpr::EnumLit { payload, .. } => {
            if let Some(p) = payload {
                mark_consumed_places(p, non_copy, moved);
            }
        }
        // `match` arms yield their tail value by value: if ANY arm
        // consumes `a`, the whole match possibly-moves `a` — mark it
        // (conservative/rejecting direction, CC #3).
        HirExpr::Match {
            scrutinee, arms, ..
        } => {
            // The scrutinee may CONSUME (`match move x { ... }`); it was
            // previously discarded by `..` — a use-after-move false
            // negative.
            mark_consumed_places(scrutinee, non_copy, moved);
            mark_all(arms.iter().map(|a| a.body.as_ref()), non_copy, moved);
        }

        // An explicit `move a` CONSUMES `a` exactly like a plain `a`
        // (the plain-Ident arm above already handles the implicit form —
        // the explicit form must not slip through `_ => {}`).
        HirExpr::Move(inner, _, _) => {
            mark_consumed_places(inner, non_copy, moved);
        }
        // The error-propagation / conversion forms consume their inner
        // value (`return expr` / `leave with expr` / casts / `?` /
        // `await`) — previously `_ => {}` (a moved value inside them
        // passed).
        HirExpr::Return { value: e, .. } | HirExpr::LeaveWith { expr: e, .. } => {
            mark_consumed_places(e, non_copy, moved);
        }
        HirExpr::Cast { expr: e, .. }
        | HirExpr::Try { expr: e, .. }
        | HirExpr::Await { expr: e, .. } => {
            mark_consumed_places(e, non_copy, moved);
        }

        // `if`/`if-let` expressions: a branch's tail value consumes.
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
            let branches = [Some(then_branch.as_slice()), else_branch.as_deref()];
            for branch in branches.into_iter().flatten() {
                if let Some(t) = branch.last()
                    && let HirStmt::Expression(expr) = t
                {
                    mark_consumed_places(expr.as_ref(), non_copy, moved);
                }
            }
        }
        // A block expression: ALL statements may consume (a `move` or a
        // `VariableDef` whose value moves) — not just the tail expression,
        // which is what the previous version inspected (a non-tail move
        // was invisible → use-after-move false negative).
        HirExpr::Block(stmts, _, _) => {
            for s in stmts {
                mark_consumed_stmt(s, non_copy, moved);
            }
        }
        // Catch/Closure/Task/UnsafeBlock bodies are NOT flattened into CFG
        // blocks either — visit their statements so a `move` inside a
        // catch branch or closure body is recorded.
        HirExpr::Catch { expr, branches, .. } => {
            mark_consumed_places(expr.as_ref(), non_copy, moved);
            for b in branches {
                for s in &b.body {
                    mark_consumed_stmt(s, non_copy, moved);
                }
            }
        }
        HirExpr::Closure { body, .. } | HirExpr::UnsafeBlock { body, .. } => {
            for s in body {
                mark_consumed_stmt(s, non_copy, moved);
            }
        }
        HirExpr::Task { block, .. } => {
            for s in block {
                mark_consumed_stmt(s, non_copy, moved);
            }
        }

        // Array element move: `let x = arr[0]` or `let x = move arr[0]`.
        // Mark the exact element place (ConstIndex/Index) as moved when the
        // root is non-Copy. The `move` keyword on a Copy value is still a
        // copy (ownership is not transferred), so we check non_copy in both
        // cases. Do NOT recurse to base — `arr` itself is not consumed.
        HirExpr::Index { index, .. } => {
            if let Some(place) = hir_expr_place(e) {
                if let Some(root) = place_root_symbol(&place) {
                    if non_copy.contains(&root) {
                        moved.insert(place);
                    }
                }
            }
            // The index expression may contain consumable sub-expressions.
            mark_consumed_places(index, non_copy, moved);
        }
        // Struct field implicit move: `let x = obj.field` (when obj is non-Copy).
        // Mark the exact field place as moved. Recurse into base because the
        // base may itself be a consumed expression (e.g. function call result).
        HirExpr::FieldAccess { base, .. } => {
            if let Some(place) = hir_expr_place(e) {
                if let Some(root) = place_root_symbol(&place) {
                    if non_copy.contains(&root) {
                        moved.insert(place);
                    }
                }
            }
            mark_consumed_places(base, non_copy, moved);
        }

        _ => {}
    }
}

/// A statement inside a nested expression container (`Block`/`Catch`/
/// `Closure`/`Task`/`UnsafeBlock`) is NOT flattened into a CFG block —
/// the move checker must visit its consuming sub-expressions directly: an
/// `Expression`'s value and a `VariableDef`'s initializer may both
/// `move` a non-Copy variable.
fn mark_consumed_stmt<'a>(
    s: &HirStmt<'a>,
    non_copy: &HashSet<Symbol>,
    moved: &mut HashSet<FrozenPlace>,
) {
    match s {
        HirStmt::Expression(expr) => mark_consumed_places(expr.as_ref(), non_copy, moved),
        HirStmt::VariableDef { value: Some(v), .. } => {
            mark_consumed_places(v.as_ref(), non_copy, moved)
        }
        _ => {}
    }
}

fn mark_all<'a, 'b: 'a>(
    exprs: impl IntoIterator<Item = &'a HirExpr<'b>>,
    non_copy: &HashSet<Symbol>,
    moved: &mut HashSet<FrozenPlace>,
) {
    for e in exprs {
        mark_consumed_places(e, non_copy, moved);
    }
}
